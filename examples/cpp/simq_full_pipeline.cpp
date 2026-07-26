#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

#include <hdf5.h>

#include "vsag/binaryset.h"
#include "vsag/dataset.h"
#include "vsag/index.h"
#include "vsag/vsag.h"

static double now(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static bool file_exists(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

static std::vector<int64_t> read_i64_1d(hid_t f, const char* name) {
    hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
    if (d < 0) throw std::runtime_error(std::string("open dataset failed: ") + name);

    hid_t s = H5Dget_space(d);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(s, &n, nullptr);

    std::vector<int64_t> out(static_cast<size_t>(n));
    if (H5Dread(d, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
        throw std::runtime_error(std::string("read dataset failed: ") + name);
    }

    H5Sclose(s);
    H5Dclose(d);
    return out;
}

static std::vector<float> read_f32_2d(hid_t f,
                                      const char* name,
                                      hsize_t r0,
                                      hsize_t rn,
                                      hsize_t dim) {
    hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
    if (d < 0) throw std::runtime_error(std::string("open dataset failed: ") + name);

    hid_t fs = H5Dget_space(d);

    hsize_t st[2] = {r0, 0};
    hsize_t ct[2] = {rn, dim};
    H5Sselect_hyperslab(fs, H5S_SELECT_SET, st, nullptr, ct, nullptr);

    hsize_t md[2] = {rn, dim};
    hid_t ms = H5Screate_simple(2, md, nullptr);

    std::vector<float> out(static_cast<size_t>(rn) * static_cast<size_t>(dim));

    if (H5Dread(d, H5T_NATIVE_FLOAT, ms, fs, H5P_DEFAULT, out.data()) < 0) {
        throw std::runtime_error(std::string("read f32 dataset failed: ") + name);
    }

    H5Sclose(ms);
    H5Sclose(fs);
    H5Dclose(d);
    return out;
}

static std::vector<int64_t> read_i64_2d_all(hid_t f,
                                            const char* name,
                                            hsize_t& rows,
                                            hsize_t& cols) {
    hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
    if (d < 0) throw std::runtime_error(std::string("open dataset failed: ") + name);

     hid_t s = H5Dget_space(d);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(s, dims, nullptr);
    rows = dims[0];
    cols = dims[1];

    std::vector<int64_t> out(static_cast<size_t>(rows) * static_cast<size_t>(cols));

    if (H5Dread(d, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
        throw std::runtime_error(std::string("read i64 2d dataset failed: ") + name);
    }

    H5Sclose(s);
    H5Dclose(d);
    return out;
}

static void write_u64(std::ofstream& out, uint64_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(uint64_t));
}

static uint64_t read_u64(std::ifstream& in) {
    uint64_t x = 0;
    in.read(reinterpret_cast<char*>(&x), sizeof(uint64_t));
    return x;
}

static void save_binary_set(const vsag::BinarySet& bs, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open index output file: " + path);

    auto keys = bs.GetKeys();
    write_u64(out, static_cast<uint64_t>(keys.size()));

    for (const auto& key : keys) {
        auto bin = bs.Get(key);

        write_u64(out, static_cast<uint64_t>(key.size()));
        out.write(key.data(), static_cast<std::streamsize>(key.size()));

        write_u64(out, bin.size);
        if (bin.size > 0) {
            out.write(reinterpret_cast<const char*>(bin.data.get()),
                      static_cast<std::streamsize>(bin.size));
        }
    }
}

static vsag::BinarySet load_binary_set(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open index input file: " + path);

    vsag::BinarySet bs;

    uint64_t n = read_u64(in);
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t key_size = read_u64(in);
        std::string key(key_size, '\0');
        in.read(&key[0], static_cast<std::streamsize>(key_size));

        uint64_t bin_size = read_u64(in);

        std::shared_ptr<int8_t[]> data;
        if (bin_size > 0) {
            data.reset(new int8_t[bin_size]);
            in.read(reinterpret_cast<char*>(data.get()), static_cast<std::streamsize>(bin_size));
        }

        vsag::Binary bin;
        bin.data = data;
        bin.size = bin_size;
        bs.Set(key, bin);
    }

    return bs;
}

static std::string make_build_param(const std::string& mv_file_path,
                                    int64_t rerank_k_default) {
    return "{"
           "\"dtype\":\"float32\","
           "\"metric_type\":\"ip\","
           "\"dim\":256,"
           "\"index_param\":{"
           "\"base_io_type\":\"async_io\","
           "\"base_file_path\":\"" + mv_file_path + "\","
           "\"init_cluster_ratio\":0.1,"
           "\"max_cluster_size\":160,"
           "\"split_start_idx\":80,"
           "\"coarse_k\":50,"
           "\"rerank_k\":" + std::to_string(rerank_k_default) +
           "}}";
}

static std::string make_search_param(int coarse_k, int64_t rerank_k) {
    return "{\"simq\":{\"coarse_k\":" + std::to_string(coarse_k) +
           ",\"rerank_k\":" + std::to_string(rerank_k) + "}}";
}

int main(int argc, char** argv) {
    std::string h5_path = "/dataset/multi_vec_20260513.hdf5";
    int64_t base_docs = 1000000;
    std::string mode = "auto";  // auto | rebuild | load

    const int dim = 256;
    const int search_topk = 100;

    const std::string index_file = "/tmp/simq_index.bin";
    const std::string mv_file = "/tmp/simq_mv_codes.bin";
    const std::string out_txt = "/tmp/simq_eval.txt";

    if (argc >= 2) h5_path = argv[1];
    if (argc >= 3) base_docs = std::atoll(argv[2]);
    if (argc >= 4) mode = argv[3];

    hid_t f = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) throw std::runtime_error("failed to open hdf5: " + h5_path);

    auto doc_counts = read_i64_1d(f, "doc_counts");
    auto doc_offsets = read_i64_1d(f, "doc_offsets");
    auto test_counts = read_i64_1d(f, "test_counts");
    auto test_offsets = read_i64_1d(f, "test_offsets");

    hsize_t gt_rows = 0;
    hsize_t gt_cols = 0;
    auto gt_neighbors = read_i64_2d_all(f, "neighbors", gt_rows, gt_cols);

    base_docs = std::min<int64_t>(base_docs, static_cast<int64_t>(doc_counts.size()));

    int64_t qtokens = 0;
    for (auto c : test_counts) qtokens += c;

    auto test = read_f32_2d(
        f, "test", 0, static_cast<hsize_t>(qtokens), static_cast<hsize_t>(dim));

    int64_t rerank_full = base_docs;
    std::string build_param = make_build_param(mv_file, rerank_full);

    bool need_build = (mode == "rebuild") || (!file_exists(index_file));
    if (mode == "load") need_build = false;

    vsag::IndexPtr index;

    if (need_build) {
        std::cout << "mode=build\n";

        int64_t start = doc_offsets[0];
        int64_t end = doc_offsets[base_docs - 1] + doc_counts[base_docs - 1];
        int64_t base_tokens = end - start;

        std::cout << "base_docs=" << base_docs << "\n";
        std::cout << "base_tokens=" << base_tokens << "\n";

        auto t_load_train = std::chrono::steady_clock::now();
        auto train = read_f32_2d(f,
                                 "train",
                                 static_cast<hsize_t>(start),
                                 static_cast<hsize_t>(base_tokens),
                                 static_cast<hsize_t>(dim));
        std::cout << "load_train_s=" << now(t_load_train) << "\n";

        std::vector<vsag::MultiVector> base_mvs(static_cast<size_t>(base_docs));
        std::vector<int64_t> ids(static_cast<size_t>(base_docs));

        for (int64_t i = 0; i < base_docs; ++i) {
            base_mvs[static_cast<size_t>(i)].len_ = static_cast<uint32_t>(doc_counts[i]);
            base_mvs[static_cast<size_t>(i)].vectors_ =
                train.data() + (doc_offsets[i] - start) * dim;
            ids[static_cast<size_t>(i)] = i;
        }

        auto base_dataset = vsag::Dataset::Make();
        base_dataset->NumElements(base_docs)
            ->Dim(dim)
            ->Ids(ids.data())
            ->MultiVectors(base_mvs.data())
            ->MultiVectorDim(dim)
            ->Owner(false);

        auto create_res = vsag::Factory::CreateIndex("simq", build_param);
        if (!create_res.has_value()) {
            std::cerr << "CreateIndex failed\n";
            return 2;
        }
        index = create_res.value();

        auto t_build = std::chrono::steady_clock::now();
        auto build_res = index->Build(base_dataset);
        if (!build_res.has_value()) {
            std::cerr << "Build failed\n";
            return 3;
        }
        std::cout << "build_s=" << now(t_build) << "\n";

        auto ser_res = index->Serialize();
        if (!ser_res.has_value()) {
            std::cerr << "Serialize failed\n";
            return 4;
        }

        save_binary_set(ser_res.value(), index_file);
        std::cout << "saved_index=" << index_file << "\n";
    } else {
        std::cout << "mode=load\n";
    }

    H5Fclose(f);

    if (!index) {
        auto create_res = vsag::Factory::CreateIndex("simq", build_param);
        if (!create_res.has_value()) {
            std::cerr << "CreateIndex for load failed\n";
            return 5;
        }
        index = create_res.value();

        auto bs = load_binary_set(index_file);
        auto deser_res = index->Deserialize(bs);
        if (!deser_res.has_value()) {
            std::cerr << "Deserialize failed\n";
            return 6;
        }
        std::cout << "loaded_index=" << index_file << "\n";
    }

    int64_t qnum = std::min<int64_t>(
        static_cast<int64_t>(test_counts.size()), static_cast<int64_t>(gt_rows));

    std::vector<vsag::MultiVector> qv(static_cast<size_t>(qnum));
    for (int64_t i = 0; i < qnum; ++i) {
        qv[static_cast<size_t>(i)].len_ = static_cast<uint32_t>(test_counts[i]);
        qv[static_cast<size_t>(i)].vectors_ = test.data() + test_offsets[i] * dim;
    }

    // Sweep coarse_k (= HNSW m in try_3 reference): probes per query token
    //   2 (min)
    //   5,10,...,50  step 5
    //   60,70,...,100  step 10
    //   150,200,...,1000  step 50
    //   1250,1500,...,3000  step 250
    std::vector<int> sweep;
    sweep.push_back(2);
    for (int v = 5; v <= 50; v += 5) sweep.push_back(v);
    for (int v = 60; v <= 100; v += 10) sweep.push_back(v);
    for (int v = 150; v <= 1000; v += 50) sweep.push_back(v);
    for (int v = 1250; v <= 3000; v += 250) sweep.push_back(v);

    std::vector<int> eval_ks = {10, 20, 50, 100};

    std::ofstream log(out_txt);
    log << "coarse_k,avg_ms,qps,avg_nret,avg_coarse_ms,avg_query_ms,avg_sort_ms,avg_mv_io_ms,avg_mv_compute_ms,avg_mv_candidates,avg_iops,avg_bw_mb_s,recall_10_at_10,recall_20_at_20,recall_50_at_50,recall_100_at_100\n";

    for (int ck : sweep) {
        std::string search_param = make_search_param(ck, rerank_full);

        double total_s = 0.0;
        std::vector<double> recall_sum(eval_ks.size(), 0.0);
        double nret_sum = 0.0;
        double coarse_ms_sum = 0.0;
        double query_ms_sum = 0.0;
        double sort_ms_sum = 0.0;
        double mv_io_ms_sum = 0.0;
        double mv_compute_ms_sum = 0.0;
        double mv_candidates_sum = 0.0;
        double mv_io_bytes_sum = 0.0;

        for (int64_t qi = 0; qi < qnum; ++qi) {
            auto ds = vsag::Dataset::Make();
            ds->NumElements(1)
                ->Dim(dim)
                ->MultiVectors(&qv[static_cast<size_t>(qi)])
                ->MultiVectorDim(dim)
                ->Owner(false);

            auto t1 = std::chrono::steady_clock::now();
            auto sr = index->KnnSearch(ds, search_topk, search_param, vsag::FilterPtr(nullptr));
            total_s += now(t1);

            if (!sr.has_value()) {
                std::cerr << "Search failed at q=" << qi << ", coarse=" << ck << "\n";
                return 7;
            }

            auto result = sr.value();
            const int64_t* ret_ids = result->GetIds();
            int64_t nret = result->GetDim();  // KNN result dim = topK; num_elements = query count
            nret_sum += static_cast<double>(nret);

            // Read per-phase timings from statistics JSON
            auto timing_vals = result->GetStatistics({"simq_coarse_ms",
                                                      "simq_query_ms",
                                                      "simq_sort_ms",
                                                      "simq_mv_io_ms",
                                                      "simq_mv_compute_ms",
                                                      "simq_mv_candidates",
                                                      "mv_io_bytes"});
            if (timing_vals.size() == 7) {
                if (!timing_vals[0].empty()) coarse_ms_sum += std::stod(timing_vals[0]);
                if (!timing_vals[1].empty()) query_ms_sum += std::stod(timing_vals[1]);
                if (!timing_vals[2].empty()) sort_ms_sum += std::stod(timing_vals[2]);
                if (!timing_vals[3].empty()) mv_io_ms_sum += std::stod(timing_vals[3]);
                if (!timing_vals[4].empty()) mv_compute_ms_sum += std::stod(timing_vals[4]);
                if (!timing_vals[5].empty()) mv_candidates_sum += std::stod(timing_vals[5]);
                if (!timing_vals[6].empty()) mv_io_bytes_sum += std::stod(timing_vals[6]);
            }

            for (size_t ei = 0; ei < eval_ks.size(); ++ei) {
                int k = eval_ks[ei];

                std::unordered_set<int64_t> gt;
                for (int j = 0; j < k && j < static_cast<int>(gt_cols); ++j) {
                    gt.insert(gt_neighbors[static_cast<size_t>(qi) * static_cast<size_t>(gt_cols) +
                                           static_cast<size_t>(j)]);
                }

                int hit = 0;
                int lim = std::min<int64_t>(nret, k);
                for (int j = 0; j < lim; ++j) {
                    if (gt.count(ret_ids[j])) ++hit;
                }

                recall_sum[ei] += static_cast<double>(hit) / static_cast<double>(k);
            }
        }

        double avg_ms = total_s * 1000.0 / static_cast<double>(qnum);
        double qps = static_cast<double>(qnum) / total_s;
        double avg_coarse_ms = coarse_ms_sum / qnum;
        double avg_query_ms = query_ms_sum / qnum;
        double avg_sort_ms = sort_ms_sum / qnum;
        double avg_mv_io_ms = mv_io_ms_sum / qnum;
        double avg_mv_compute_ms = mv_compute_ms_sum / qnum;
        double avg_mv_candidates = mv_candidates_sum / qnum;
        double avg_iops = (mv_io_ms_sum > 0.0)
                              ? mv_candidates_sum / (mv_io_ms_sum / 1000.0)
                              : 0.0;
        double avg_bw_mb_s = (mv_io_ms_sum > 0.0)
                                 ? mv_io_bytes_sum / (mv_io_ms_sum / 1000.0) / 1e6
                                 : 0.0;

        std::cout << "coarse=" << ck
                  << " avg_ms=" << avg_ms
                  << " qps=" << qps
                  << " avg_nret=" << nret_sum / qnum
                  << " coarse=" << avg_coarse_ms << "ms"
                  << " query=" << avg_query_ms << "ms"
                  << " sort=" << avg_sort_ms << "ms"
                  << " mv_io=" << avg_mv_io_ms << "ms"
                  << " mv_compute=" << avg_mv_compute_ms << "ms"
                  << " mv_cands=" << avg_mv_candidates
                  << " iops=" << avg_iops
                  << " bw_mb_s=" << avg_bw_mb_s
                  << " r10=" << recall_sum[0] / qnum
                  << " r20=" << recall_sum[1] / qnum
                  << " r50=" << recall_sum[2] / qnum
                  << " r100=" << recall_sum[3] / qnum
                  << "\n";

        log << ck << ","
            << avg_ms << ","
            << qps << ","
            << nret_sum / qnum << ","
            << avg_coarse_ms << ","
            << avg_query_ms << ","
            << avg_sort_ms << ","
            << avg_mv_io_ms << ","
            << avg_mv_compute_ms << ","
            << avg_mv_candidates << ","
            << avg_iops << ","
            << avg_bw_mb_s << ","
            << recall_sum[0] / qnum << ","
            << recall_sum[1] / qnum << ","
            << recall_sum[2] / qnum << ","
            << recall_sum[3] / qnum << "\n";
        log.flush();
    }

    log.close();
    std::cout << "saved_eval=" << out_txt << "\n";

    return 0;
}
