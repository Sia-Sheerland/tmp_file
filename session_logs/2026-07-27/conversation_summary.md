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

## 5. 后续：IOPS / 带宽输出改为按 coarse_k 聚合

### 问题
per-query 的 stderr 输出（每个查询一行 `[SIMQ Query] candidates=... iops=... bw_mb_s=...`）刷屏不美观。用户希望每个 coarse_k 档位只输出一行平均值。

### 用户反馈
- 用户要求"直接改 pipeline"——即尽量在 `simq_full_pipeline.cpp` 层面解决，不要动底层
- IOPS 可以直接从已有 stats（`mv_candidates` / `mv_io_ms`）算出：`avg_iops = sum_candidates / sum_io_ms * 1000`
- 带宽需要 `total_bytes`，只有 `.inl` 内部知道，必须至少改一个底层文件透传出来

### 最终改动（commit 3a611f31，分支 perf/coarse-search-flat-array）

**`src/query_context.h`**
- 新增字段 `std::atomic<uint64_t> mv_io_bytes{0}`
- `Dump()` 中新增 `j["mv_io_bytes"].SetUint64(...)` 序列化

**`src/datacell/multi_vector_datacell.inl`**
- 删掉 per-query `fprintf(stderr, "[SIMQ Query] ...")` 块
- stats 块中加一行 `stats->mv_io_bytes.fetch_add(total_size, std::memory_order_relaxed);`

**`examples/cpp/simq_full_pipeline.cpp`**
- 新增累加器 `mv_io_bytes_sum`
- `GetStatistics` 增加 `"mv_io_bytes"` key（注意：这个 key 来自 `stats.Dump()`，没有 `simq_` 前缀）
- 每个 coarse_k 循环结束后计算：
  - `avg_iops = mv_candidates_sum / (mv_io_ms_sum / 1000)`
  - `avg_bw_mb_s = mv_io_bytes_sum / (mv_io_ms_sum / 1000) / 1e6`
- stdout 摘要行和 CSV 都新增 `avg_iops` 和 `avg_bw_mb_s` 两列

**`src/algorithm/simq/simq.cpp`**
- 没动。`dump_simq_statistics` 内部调用 `stats.Dump()` 生成 JSON，新字段自动出现在顶层，可以直接 `GetStatistics({"mv_io_bytes"})` 读到

### 关键技巧
`SearchStatistics::Dump()` 的字段会自动进入最终 JSON，不需要在 `dump_simq_statistics` 里显式透传——只要字段名不带 `simq_` 前缀就能直接被 `GetStatistics` 读到

### 编译说明
增量编译即可，不需要 `find build-release -name "*.cpp.o" -delete`：
```bash
make release -j$(nproc)
g++ -std=c++17 -O2 -I include -I /usr/include/hdf5/serial \
    examples/cpp/simq_full_pipeline.cpp \
    -L build-release/src -lvsag \
    -L /usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 \
    -lpthread -o /tmp/simq_full_new
```

### 新输出格式示例
```
coarse=50 avg_ms=12.3 qps=81.3 ... mv_cands=1234 iops=18500.3 bw_mb_s=512.7 r10=0.85 ...
```
每个 coarse_k 一行，不再刷屏

### 待办（更新）
- [ ] 跑一轮新 benchmark，看各 coarse_k 下的 avg IOPS 和 avg bandwidth 实际数字
- [ ] 评估是否要加多线程 compute
- [ ] 评估是否要改 chunking 策略
- [ ] 评估是否要换内联 SIMD（但要避开上次的性能陷阱）
