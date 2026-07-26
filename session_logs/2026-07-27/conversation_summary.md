# 2026-07-27 SIMQ 性能优化对话记录

## 参与者
- 用户（陆亦帆）
- Claude（助手）

## 背景
SIMQ 是 vsag 项目里的多向量检索索引（ColBERT-style）。用户的参考实现 `SIMQ_try_3_temp4`（也叫 v4）跑 35 QPS，但 SIMQ 只有 10-12 QPS。用户想找出差距原因并优化。

## 这一轮做了什么

### 1. 初识问题：fio 硬件极限测试
- 用 fio 测 mv_codes.bin 的随机读性能
- 结果：BW=1.67 GB/s, IOPS=49K（34KB 块，iodepth=400）
- 发现：SIMQ 实际 mv_io=60-74ms，远低于 fio 测出的硬件极限

### 2. 排查文件系统差异
- `/tmp`（SIMQ 数据）vs `/dataset`（原始数据）挂载点对比
- 碎片对比：`/tmp` 19121 extents vs `/dataset` 1135 extents
- 结论：碎片化严重，但 fio 测得两个路径速度一样

### 3. 尝试的优化（大多失败或收益小）

#### 杠杆 1：粗搜用扁平数组替换 hash（✅ 成功）
- 文件：`src/algorithm/simq/simq.cpp` + `simq.h`
- 粗搜阶段每 query token 用 O(1) 数组查分，替换 unordered_map/set
- 收益：粗搜从 ~几ms 降到 <1ms

#### 杠杆 2：消除 O_DIRECT（❌ 效果小）
- 让读走 page cache，理论上数据热时内存读
- 实测：QPS 从 10 涨到 12（只 20%），用户说效果不明显

#### 杠杆 3：按 offset 排序（✅ 实现但收益小）
- MultiRead 前按字节偏移排序，让 SSD 顺序读
- 实测：mv_io 从 75.67ms 降到 73.85ms（只 2.4%）
- 原因：即使排序，doc 仍散布在 34GB 文件里，本质还是随机读

#### 杠杆 4：增大 DEFAULT_REQUEST_COUNT（❌ 没效果）
- 400 → 4096，让单次 io_submit 提交所有 2322 个请求
- 实测：速度基本没变，甚至稍慢
- 原因：chunking 不是主瓶颈，函数指针开销才是

#### 杠杆 5：try_3_temp4 风格 IO + 内联 SIMD（❌ 实测更慢，已撤回）
- 给每个 doc 分配 per-doc buffer，单次 io_submit
- ComputeDist 改 inline scalar loop（编译器自动 SIMD）
- 恢复 O_DIRECT
- 实测：反而变慢！用户让撤回了
- 原因：posix_memalign × 2322 开销大 + 单次提交 2322 个请求内核处理慢

### 4. 当前状态
- 已撤回 try_3_temp4 风格改动
- 已加 per-query IO 诊断输出（stderr），打印 IOPS 和带宽
- 等待用户跑测看实际数字

## 关键文件改动（vs origin/main）

### src/algorithm/simq/simq.cpp
- 新增 `coarse_score_buf_`, `coarse_seen_buf_`, `coarse_dirty_`, `coarse_seen_dirty_` 成员
- 重写 `coarse_search` 用扁平数组替换 hash 容器

### src/algorithm/simq/simq.h
- 对应成员声明

### src/datacell/multi_vector_datacell.inl
- 加 `<numeric>` include
- Query 加 sort-by-offset（lever 3）
- Query 加 per-query IO 诊断输出

### src/io/async_io/async_io.cpp
- 当前是 page cache 模式（无 O_DIRECT）

## 关键发现

### 性能差距的真正原因
try_3_temp4 比 SIMQ 快的原因（按重要性排序）：
1. **Compute 多线程**（SIMQ 单线程，try_3_temp4 用满所有核）
2. **单次 io_submit**（SIMQ 切 400 chunk 6 次，try_3_temp4 一次提交全部）
3. **无 memcpy 开销**（SIMQ 把数据拷贝到 all_codes，try_3_temp4 直接用 per-doc buffer）
4. **内联 SIMD**（SIMQ 用函数指针 FP32ComputeIP，try_3_temp4 是 inline scalar loop 自动向量化）

### 为什么不能简单模仿 try_3_temp4
- per-doc posix_memalign × 2322 实测反而更慢
- 单次 io_submit 2322 个请求内核处理也慢
- 所以 try_3_temp4 的"优势"不一定能直接搬到 SIMQ

## 待办
- [ ] 看 per-query IO 诊断输出，确定实际 IOPS 和带宽
- [ ] 评估是否要加多线程 compute
- [ ] 评估是否要改 chunking 策略
- [ ] 评估是否要换内联 SIMD（但要避开上次的性能陷阱）
