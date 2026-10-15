// moe-l2ab -- G3-0a micro-benchmark: "미스난 전문가를 어디서 계산할 것인가" A/B 실측
//
//   A (CPU 제자리) : ggml CPU 백엔드(N스레드)로 gate/up/down GEMV x N전문가
//   B (pinned->GPU): pinned 호스트 버퍼 적재 -> H2D -> CUDA GEMV -> D2H, 복사/연산 분리 계측
//                     + pageable 호스트 대비 H2D 대역 비교
//
// 대상 소스 트리: llama.cpp b10057 (commit 0bd0ec609 -- 프리빌드 D:\moe-tools\llama-b10057\ 와
// 동일, G1에서 기검증). 링크 대상: 로컬 빌드(build/ggml/src/{ggml,ggml-base}.lib) -- CUDA는
// 정적 링크하지 않고 ggml_backend_load_all()의 런타임 동적 로딩(레지스트리)으로 해결한다.
// 근거: ggml/src/ggml-backend-reg.cpp의 ggml_backend_load_best()가 실행파일 디렉터리 +
// 현재 작업 디렉터리를 검색해 "ggml-<name>-*.dll"(스코어 최고) 또는 "ggml-<name>.dll"을
// 로드한다 (동 파일 480~559행). CPU 스레드 수 설정도 ggml-cpu.lib를 직접 링크하지 않고
// llama.cpp 1차 소스의 확립된 패턴(src/llama-context.cpp:359)을 그대로 따라
// ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")로 동적 획득한다.
// 이 프로그램은 ggml.h/ggml-alloc.h/ggml-backend.h(+ggml-cpp.h)만 포함하며 ggml-cpu.h/
// ggml-cuda.h는 포함하지 않는다(GGML_BACKEND_API 심볼 불필요 -> GGML_BACKEND_SHARED 의존 없음).
//
// 실행 배치: 이 exe는 D:\moe-tools\llama-b10057\ 안(또는 CWD를 그 경로로)에서 실행해야
// ggml_backend_load_all()이 그 폴더의 ggml-cuda.dll / ggml-cpu-<isa 최적>.dll을 찾는다.
//
// 치수(1차 소스: bench_results/g1/gguf_map_gpt-oss-120b.json summary/per_layer):
//   gate/up/down 각각 MXFP4 [2880,2880] = 4,406,400B + F32 bias[2880] = 11,520B
//   전문가당 합계 = 3*(4,406,400+11,520) = 13,253,760B ("per_expert_slice_bytes_avg"와 일치)
//
// 단순화(의도적, 문서화): down의 입력은 실제로는 gate/up 이후의 SwiGLU 활성값이지만, 이
// 모델은 feed_forward_length == embedding_length == 2880이라 down 가중치도 [2880,2880]로
// 동일 shape이다. 활성화 함수 체이닝은 GEMV 처리시간에 영향을 주지 않으므로(과제 지시:
// "타이밍은 데이터 무관") down도 동일 x[2880]을 입력으로 재사용한다 -- 3개 행렬은 서로
// 독립적인 GEMV로 취급(실제 FFN의 gate*up 결합/활성화는 모델링하지 않음).
//
// ---------------------------------------------------------------------------------------
// !!! 절대계약 (G3-0a 작업지시): G2-3b 타이밍 실측 진행 중 GPU를 만지는 어떤 실행도 금지.
// 이 파일을 작성한 에이전트는 컴파일/링크 성공과 `--help`(백엔드 초기화 이전, 인자 파싱
// 수준) 실행만 확인했다. 실제 벤치 실행(--mode cpu/gpu/all, --list-devices 포함 -- 이들은
// ggml_backend_load_all() 이후 backend/device를 실제로 초기화한다)은 메인(Fable)이
// G2-3b 종료 후 수행한다.
// ---------------------------------------------------------------------------------------

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// 상수 (1차 소스: bench_results/g1/gguf_map_gpt-oss-120b.json)
// ============================================================================
static constexpr int64_t DIM               = 2880;               // embedding_length == feed_forward_length
static constexpr size_t  WEIGHT_BYTES       = 4406400;            // MXFP4 [2880,2880] (2880*2880/32*17)
static constexpr size_t  BIAS_BYTES         = DIM * sizeof(float); // F32 [2880]
static constexpr size_t  EXPERT_BYTES       = 3 * (WEIGHT_BYTES + BIAS_BYTES); // 13,253,760B
static constexpr int     MAX_N_EXPERTS_CAP  = 64; // 그래프/컨텍스트 사이즈 여유분 (과제 요구 N<=16의 4배)

// ============================================================================
// 유틸
// ============================================================================
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static double mean_of(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double)v.size();
}

static std::string now_utc_iso() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return std::string(buf);
}

// ============================================================================
// CLI 인자 (여기까지는 ggml/backend 호출 없음 -- --help/파싱 실패는 반드시 백엔드
// 초기화 이전에 리턴해야 한다: G3-0a 절대계약의 "인자 파싱 수준 실행" 허용 범위)
// ============================================================================
struct Args {
    bool help = false;
    bool ok = true;
    std::string err;

    std::string mode = "all";       // cpu | gpu | all
    std::vector<int> n_list = {1, 4, 16};
    int reps = 50;
    int warmup = 5;
    int threads = 8;
    std::string host_buf = "both";  // pinned | pageable | both (gpu 전용)
    std::string json_out;           // 비어있으면 저장 안 함
    unsigned seed = 12345;
    bool list_devices = false;
};

static void print_usage(const char * argv0) {
    std::printf(
        "moe-l2ab -- G3-0a L2 경로 A/B 마이크로벤치 (CPU 제자리 GEMV vs pinned H2D+GPU GEMV)\n"
        "\n"
        "usage: %s [옵션]\n"
        "  --mode {cpu|gpu|all}                실행 경로 (기본: all)\n"
        "  --n <n1,n2,...>                      배치 전문가 수 목록, 콤마구분 (기본: 1,4,16)\n"
        "  --reps <N>                           웜업 후 반복 횟수 (기본: 50)\n"
        "  --warmup <N>                         웜업 반복 횟수 (기본: 5)\n"
        "  --threads <N>                        CPU 백엔드 스레드 수 (기본: 8)\n"
        "  --host-buf {pinned|pageable|both}    gpu 모드 H2D 소스 버퍼 (기본: both)\n"
        "  --json <path>                        결과 JSON 저장 경로 (미지정시 콘솔 표만)\n"
        "  --seed <N>                           난수 시드 (기본: 12345)\n"
        "  --list-devices                       백엔드 로드 후 디바이스 목록만 출력\n"
        "  --help                               이 도움말 출력 (백엔드 초기화 이전 종료)\n"
        "\n"
        "치수(1차 소스: bench_results/g1/gguf_map_gpt-oss-120b.json): 전문가당 gate/up/down\n"
        "MXFP4 [2880,2880](각 %zuB) + F32 bias[2880](각 %zuB) = %zuB/expert\n"
        "\n"
        "실행 위치: 이 exe는 D:\\moe-tools\\llama-b10057\\ 안에서(또는 그 경로를 CWD로) 실행해야\n"
        "ggml_backend_load_all()이 그 폴더의 ggml-cuda.dll / ggml-cpu-<isa>.dll을 찾는다.\n",
        argv0, WEIGHT_BYTES, BIAS_BYTES, EXPERT_BYTES);
}

static std::vector<int> parse_csv_int(const std::string & s, bool & ok) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        try {
            size_t pos = 0;
            int v = std::stoi(tok, &pos);
            if (pos != tok.size() || v <= 0) { ok = false; return {}; }
            out.push_back(v);
        } catch (...) {
            ok = false;
            return {};
        }
    }
    if (out.empty()) ok = false;
    return out;
}

static bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need_val = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                a.ok = false;
                a.err = std::string(name) + " 옵션에 값이 필요합니다";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--mode") {
            const char * v = need_val("--mode");
            if (!v) return false;
            a.mode = v;
            if (a.mode != "cpu" && a.mode != "gpu" && a.mode != "all") {
                a.ok = false; a.err = "--mode 는 cpu|gpu|all 중 하나여야 합니다"; return false;
            }
        } else if (arg == "--n") {
            const char * v = need_val("--n");
            if (!v) return false;
            bool ok = true;
            a.n_list = parse_csv_int(v, ok);
            if (!ok) { a.ok = false; a.err = "--n 파싱 실패 (예: 1,4,16)"; return false; }
            for (int n : a.n_list) {
                if (n > MAX_N_EXPERTS_CAP) {
                    a.ok = false;
                    a.err = "--n 값이 상한(" + std::to_string(MAX_N_EXPERTS_CAP) + ")을 초과합니다";
                    return false;
                }
            }
        } else if (arg == "--reps") {
            const char * v = need_val("--reps"); if (!v) return false;
            a.reps = std::atoi(v);
            if (a.reps <= 0) { a.ok = false; a.err = "--reps 는 양수여야 합니다"; return false; }
        } else if (arg == "--warmup") {
            const char * v = need_val("--warmup"); if (!v) return false;
            a.warmup = std::atoi(v);
            if (a.warmup < 0) { a.ok = false; a.err = "--warmup 은 0 이상이어야 합니다"; return false; }
        } else if (arg == "--threads") {
            const char * v = need_val("--threads"); if (!v) return false;
            a.threads = std::atoi(v);
            if (a.threads <= 0) { a.ok = false; a.err = "--threads 는 양수여야 합니다"; return false; }
        } else if (arg == "--host-buf") {
            const char * v = need_val("--host-buf"); if (!v) return false;
            a.host_buf = v;
            if (a.host_buf != "pinned" && a.host_buf != "pageable" && a.host_buf != "both") {
                a.ok = false; a.err = "--host-buf 는 pinned|pageable|both 중 하나여야 합니다"; return false;
            }
        } else if (arg == "--json") {
            const char * v = need_val("--json"); if (!v) return false;
            a.json_out = v;
        } else if (arg == "--seed") {
            const char * v = need_val("--seed"); if (!v) return false;
            a.seed = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (arg == "--list-devices") {
            a.list_devices = true;
        } else {
            a.ok = false;
            a.err = "알 수 없는 옵션: " + arg;
            return false;
        }
    }
    return true;
}

// ============================================================================
// 랜덤 데이터 채움 (ggml_quantize_chunk 경유 -- tests/test-backend-ops.cpp의
// init_tensor_uniform() 과 동일한 "32개 블록을 1개 row로 취급" 방식을 따른다.
// 원시 랜덤 바이트를 MXFP4 블록에 직접 채우지 않는 이유: 지수 바이트(e8m0)가 극단값이면
// 역양자화 시 NaN/Inf/서브노멀이 나올 수 있어 -- 정식 양자화 경로가 더 안전하다.)
// ============================================================================
static void fill_f32_uniform(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    int64_t nels = ggml_nelements(t);
    std::vector<float> data(nels);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int64_t i = 0; i < nels; i++) data[i] = dist(rng);
    ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
}

static void fill_mxfp4_uniform(ggml_tensor * t, std::mt19937 & rng) {
    int64_t nels = ggml_nelements(t);
    const int64_t blck = ggml_blck_size(t->type);
    GGML_ASSERT(nels % blck == 0);

    std::vector<float> data(nels);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < nels; i++) data[i] = dist(rng);

    std::vector<uint8_t> dataq(ggml_row_size(t->type, nels));
    // 블록 단위를 "1 row = blck개 원소"로 취급해 양자화 (test-backend-ops.cpp 패턴과 동일)
    ggml_quantize_chunk(t->type, data.data(), dataq.data(), 0, nels / blck, blck, /*imatrix=*/nullptr);
    ggml_backend_tensor_set(t, dataq.data(), 0, dataq.size());
}

// ============================================================================
// 전문가 슬랩: gate/up/down 가중치(MXFP4)+bias(F32) 텐서 묶음
// ============================================================================
struct ExpertSlab {
    ggml_tensor * gate_w = nullptr, * gate_b = nullptr;
    ggml_tensor * up_w   = nullptr, * up_b   = nullptr;
    ggml_tensor * down_w = nullptr, * down_b = nullptr;
};

static ExpertSlab make_expert_slab(ggml_context * ctx, int idx) {
    ExpertSlab e;
    char name[64];

    e.gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, DIM, DIM);
    std::snprintf(name, sizeof(name), "e%d_gate_w", idx); ggml_set_name(e.gate_w, name);
    e.gate_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, DIM);
    std::snprintf(name, sizeof(name), "e%d_gate_b", idx); ggml_set_name(e.gate_b, name);

    e.up_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, DIM, DIM);
    std::snprintf(name, sizeof(name), "e%d_up_w", idx); ggml_set_name(e.up_w, name);
    e.up_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, DIM);
    std::snprintf(name, sizeof(name), "e%d_up_b", idx); ggml_set_name(e.up_b, name);

    e.down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, DIM, DIM);
    std::snprintf(name, sizeof(name), "e%d_down_w", idx); ggml_set_name(e.down_w, name);
    e.down_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, DIM);
    std::snprintf(name, sizeof(name), "e%d_down_b", idx); ggml_set_name(e.down_b, name);

    return e;
}

static void fill_expert_slab(ExpertSlab & e, std::mt19937 & rng) {
    fill_mxfp4_uniform(e.gate_w, rng); fill_f32_uniform(e.gate_b, rng, -0.1f, 0.1f);
    fill_mxfp4_uniform(e.up_w,   rng); fill_f32_uniform(e.up_b,   rng, -0.1f, 0.1f);
    fill_mxfp4_uniform(e.down_w, rng); fill_f32_uniform(e.down_b, rng, -0.1f, 0.1f);
}

static size_t expert_slab_bytes(const ExpertSlab & e) {
    return ggml_nbytes(e.gate_w) + ggml_nbytes(e.gate_b)
         + ggml_nbytes(e.up_w)   + ggml_nbytes(e.up_b)
         + ggml_nbytes(e.down_w) + ggml_nbytes(e.down_b);
}

// 그래프/컨텍스트 여유 사이즈 (모든 N<=MAX_N_EXPERTS_CAP 케이스를 커버하는 고정 상한 --
// no_alloc 컨텍스트라 메타데이터만 차지, 넉넉히 잡아도 비용이 거의 없다)
static ggml_context * make_meta_ctx(size_t graph_size) {
    // 텐서 수 상한: x(1) + 전문가당 leaf 6개 + 전문가당 op 6개(3 mul_mat + 3 add) = 13/전문가
    size_t n_tensors_budget = 1 + 13 * MAX_N_EXPERTS_CAP + 64;
    ggml_init_params params;
    params.mem_size   = ggml_tensor_overhead() * n_tensors_budget + ggml_graph_overhead_custom(graph_size, false);
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    return ggml_init(params);
}

// ============================================================================
// Path A: CPU 제자리 GEMV
// ============================================================================
struct TimingStat {
    double p50_us = 0, p95_us = 0, mean_us = 0;
    int    reps = 0;
};

static TimingStat summarize(const std::vector<double> & samples_us) {
    TimingStat s;
    s.reps    = (int)samples_us.size();
    s.p50_us  = percentile(samples_us, 0.50);
    s.p95_us  = percentile(samples_us, 0.95);
    s.mean_us = mean_of(samples_us);
    return s;
}

struct CpuCaseResult {
    int n_experts = 0;
    size_t batch_bytes = 0;
    TimingStat compute;
};

static bool run_cpu_case(ggml_backend_t backend_cpu, int n_experts, const Args & args,
                          CpuCaseResult & out, std::string & err) {
    const size_t graph_size = std::max<size_t>(256, (size_t)(8 * n_experts + 32));
    ggml_context_ptr ctx(make_meta_ctx(graph_size));
    if (!ctx) { err = "ggml_init 실패 (cpu case)"; return false; }

    ggml_tensor * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, DIM);
    ggml_set_name(x, "x_in");

    std::vector<ExpertSlab> experts;
    experts.reserve(n_experts);
    for (int i = 0; i < n_experts; i++) experts.push_back(make_expert_slab(ctx.get(), i));

    std::vector<ggml_tensor *> outs;
    outs.reserve(n_experts * 3);
    for (int i = 0; i < n_experts; i++) {
        ggml_tensor * o_gate = ggml_add(ctx.get(), ggml_mul_mat(ctx.get(), experts[i].gate_w, x), experts[i].gate_b);
        ggml_tensor * o_up   = ggml_add(ctx.get(), ggml_mul_mat(ctx.get(), experts[i].up_w,   x), experts[i].up_b);
        ggml_tensor * o_down = ggml_add(ctx.get(), ggml_mul_mat(ctx.get(), experts[i].down_w, x), experts[i].down_b);
        outs.push_back(o_gate); outs.push_back(o_up); outs.push_back(o_down);
    }

    // 할당 (CPU 백엔드 기본 버퍼 타입)
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend_cpu));
    if (!buf) { err = "ggml_backend_alloc_ctx_tensors 실패 (cpu case)"; return false; }

    std::mt19937 rng(args.seed + (unsigned)n_experts);
    fill_f32_uniform(x, rng, -1.0f, 1.0f);
    for (auto & e : experts) fill_expert_slab(e, rng);

    size_t batch_bytes = ggml_nbytes(x);
    for (auto & e : experts) batch_bytes += expert_slab_bytes(e);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_size, false);
    for (ggml_tensor * o : outs) ggml_build_forward_expand(gf, o);

    // 웜업
    for (int w = 0; w < args.warmup; w++) {
        ggml_status st = ggml_backend_graph_compute(backend_cpu, gf);
        if (st != GGML_STATUS_SUCCESS) { err = std::string("웜업 compute 실패: ") + ggml_status_to_string(st); return false; }
    }

    std::vector<double> samples;
    samples.reserve(args.reps);
    for (int r = 0; r < args.reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_status st = ggml_backend_graph_compute(backend_cpu, gf);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (st != GGML_STATUS_SUCCESS) { err = std::string("compute 실패: ") + ggml_status_to_string(st); return false; }
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    out.n_experts   = n_experts;
    out.batch_bytes = batch_bytes;
    out.compute     = summarize(samples);
    return true;
}

// ============================================================================
// Path B: pinned/pageable -> H2D -> GPU GEMV -> D2H
// ============================================================================
struct GpuCaseResult {
    int n_experts = 0;
    size_t batch_weight_bytes = 0; // x 제외, 이번 배치에서 새로 옮기는 전문가 가중치+bias 총량
    bool has_pinned = false, has_pageable = false;
    TimingStat h2d_pinned, h2d_pageable;
    double bw_pinned_gbps = 0, bw_pageable_gbps = 0;
    TimingStat compute;
    TimingStat d2h;
    double bw_d2h_gbps = 0;
};

// 지정한 buffer type으로 x + N전문가 슬랩을 담는 host-side 컨텍스트를 만들고 채운다.
struct HostStaging {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * x = nullptr;
    std::vector<ExpertSlab> experts;
};

static bool build_host_staging(ggml_backend_buffer_type_t buft, int n_experts,
                                const Args & args, HostStaging & hs, std::string & err) {
    hs.ctx.reset(make_meta_ctx(64));
    if (!hs.ctx) { err = "ggml_init 실패 (host staging)"; return false; }

    hs.x = ggml_new_tensor_1d(hs.ctx.get(), GGML_TYPE_F32, DIM);
    ggml_set_name(hs.x, "x_host");
    hs.experts.reserve(n_experts);
    for (int i = 0; i < n_experts; i++) hs.experts.push_back(make_expert_slab(hs.ctx.get(), i));

    hs.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(hs.ctx.get(), buft));
    if (!hs.buf) { err = "ggml_backend_alloc_ctx_tensors_from_buft 실패 (host staging)"; return false; }

    std::mt19937 rng(args.seed + 777u + (unsigned)n_experts);
    fill_f32_uniform(hs.x, rng, -1.0f, 1.0f);
    for (auto & e : hs.experts) fill_expert_slab(e, rng);
    return true;
}

static bool run_gpu_case(ggml_backend_dev_t dev_gpu, ggml_backend_t backend_gpu,
                          int n_experts, const Args & args, GpuCaseResult & out, std::string & err) {
    out.n_experts = n_experts;

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev_gpu);
    ggml_backend_buffer_type_t pageable_buft = ggml_backend_cpu_buffer_type();
    if (!host_buft) { err = "이 GPU 디바이스는 pinned host buffer type을 지원하지 않습니다"; return false; }

    bool want_pinned   = (args.host_buf == "pinned"   || args.host_buf == "both");
    bool want_pageable = (args.host_buf == "pageable" || args.host_buf == "both");

    HostStaging hs_pinned, hs_pageable;
    if (want_pinned) {
        if (!build_host_staging(host_buft, n_experts, args, hs_pinned, err)) return false;
    }
    if (want_pageable) {
        if (!build_host_staging(pageable_buft, n_experts, args, hs_pageable, err)) return false;
    }

    // 디바이스측 상주 텐서 (x + N전문가) -- 기본(디바이스) 버퍼 타입
    const size_t graph_size = std::max<size_t>(256, (size_t)(8 * n_experts + 32));
    ggml_context_ptr ctx_dev(make_meta_ctx(graph_size));
    if (!ctx_dev) { err = "ggml_init 실패 (device ctx)"; return false; }

    ggml_tensor * x_dev = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_F32, DIM);
    ggml_set_name(x_dev, "x_dev");
    std::vector<ExpertSlab> dev_experts;
    dev_experts.reserve(n_experts);
    for (int i = 0; i < n_experts; i++) dev_experts.push_back(make_expert_slab(ctx_dev.get(), i));

    std::vector<ggml_tensor *> outs;
    outs.reserve(n_experts * 3);
    for (int i = 0; i < n_experts; i++) {
        ggml_tensor * o_gate = ggml_add(ctx_dev.get(), ggml_mul_mat(ctx_dev.get(), dev_experts[i].gate_w, x_dev), dev_experts[i].gate_b);
        ggml_tensor * o_up   = ggml_add(ctx_dev.get(), ggml_mul_mat(ctx_dev.get(), dev_experts[i].up_w,   x_dev), dev_experts[i].up_b);
        ggml_tensor * o_down = ggml_add(ctx_dev.get(), ggml_mul_mat(ctx_dev.get(), dev_experts[i].down_w, x_dev), dev_experts[i].down_b);
        outs.push_back(o_gate); outs.push_back(o_up); outs.push_back(o_down);
    }

    ggml_backend_buffer_ptr buf_dev(ggml_backend_alloc_ctx_tensors(ctx_dev.get(), backend_gpu));
    if (!buf_dev) { err = "ggml_backend_alloc_ctx_tensors 실패 (device ctx)"; return false; }

    // x는 배치 상주 활성값으로 간주 -- 타이밍 루프 밖에서 1회만 적재(H2D 측정 대상은
    // "새로 당겨야 하는 전문가 가중치"이지 상주 활성값이 아님)
    {
        ggml_tensor * x_src = want_pinned ? hs_pinned.x : hs_pageable.x;
        ggml_backend_tensor_set(x_dev, x_src->data, 0, ggml_nbytes(x_dev));
    }

    size_t batch_weight_bytes = 0;
    for (auto & e : dev_experts) batch_weight_bytes += expert_slab_bytes(e);
    out.batch_weight_bytes = batch_weight_bytes;

    auto h2d_from = [&](HostStaging & hs) -> std::vector<double> {
        std::vector<double> samples;
        samples.reserve(args.reps);
        // 웜업
        for (int w = 0; w < args.warmup; w++) {
            for (int i = 0; i < n_experts; i++) {
                ggml_backend_tensor_set(dev_experts[i].gate_w, hs.experts[i].gate_w->data, 0, ggml_nbytes(dev_experts[i].gate_w));
                ggml_backend_tensor_set(dev_experts[i].gate_b, hs.experts[i].gate_b->data, 0, ggml_nbytes(dev_experts[i].gate_b));
                ggml_backend_tensor_set(dev_experts[i].up_w,   hs.experts[i].up_w->data,   0, ggml_nbytes(dev_experts[i].up_w));
                ggml_backend_tensor_set(dev_experts[i].up_b,   hs.experts[i].up_b->data,   0, ggml_nbytes(dev_experts[i].up_b));
                ggml_backend_tensor_set(dev_experts[i].down_w, hs.experts[i].down_w->data, 0, ggml_nbytes(dev_experts[i].down_w));
                ggml_backend_tensor_set(dev_experts[i].down_b, hs.experts[i].down_b->data, 0, ggml_nbytes(dev_experts[i].down_b));
            }
        }
        for (int r = 0; r < args.reps; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_experts; i++) {
                ggml_backend_tensor_set(dev_experts[i].gate_w, hs.experts[i].gate_w->data, 0, ggml_nbytes(dev_experts[i].gate_w));
                ggml_backend_tensor_set(dev_experts[i].gate_b, hs.experts[i].gate_b->data, 0, ggml_nbytes(dev_experts[i].gate_b));
                ggml_backend_tensor_set(dev_experts[i].up_w,   hs.experts[i].up_w->data,   0, ggml_nbytes(dev_experts[i].up_w));
                ggml_backend_tensor_set(dev_experts[i].up_b,   hs.experts[i].up_b->data,   0, ggml_nbytes(dev_experts[i].up_b));
                ggml_backend_tensor_set(dev_experts[i].down_w, hs.experts[i].down_w->data, 0, ggml_nbytes(dev_experts[i].down_w));
                ggml_backend_tensor_set(dev_experts[i].down_b, hs.experts[i].down_b->data, 0, ggml_nbytes(dev_experts[i].down_b));
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        return samples;
    };

    if (want_pinned) {
        auto s = h2d_from(hs_pinned);
        out.has_pinned = true;
        out.h2d_pinned = summarize(s);
        out.bw_pinned_gbps = (out.h2d_pinned.p50_us > 0)
            ? ((double)batch_weight_bytes / 1e9) / (out.h2d_pinned.p50_us / 1e6) : 0.0;
    }
    if (want_pageable) {
        auto s = h2d_from(hs_pageable);
        out.has_pageable = true;
        out.h2d_pageable = summarize(s);
        out.bw_pageable_gbps = (out.h2d_pageable.p50_us > 0)
            ? ((double)batch_weight_bytes / 1e9) / (out.h2d_pageable.p50_us / 1e6) : 0.0;
    }

    // 연산 타이밍 (이 시점에는 위 h2d_from()의 마지막 호출로 디바이스 텐서가 채워진 상태)
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_dev.get(), graph_size, false);
    for (ggml_tensor * o : outs) ggml_build_forward_expand(gf, o);

    for (int w = 0; w < args.warmup; w++) {
        ggml_status st = ggml_backend_graph_compute(backend_gpu, gf);
        if (st != GGML_STATUS_SUCCESS) { err = std::string("웜업 GPU compute 실패: ") + ggml_status_to_string(st); return false; }
    }
    std::vector<double> compute_samples;
    compute_samples.reserve(args.reps);
    for (int r = 0; r < args.reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_status st = ggml_backend_graph_compute(backend_gpu, gf);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (st != GGML_STATUS_SUCCESS) { err = std::string("GPU compute 실패: ") + ggml_status_to_string(st); return false; }
        compute_samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    out.compute = summarize(compute_samples);

    // D2H: 결과 3N벡터 회수 (일반 malloc 호스트 버퍼 -- 이 항목은 pinned/pageable 비교 대상이 아님)
    size_t out_bytes_total = 0;
    for (ggml_tensor * o : outs) out_bytes_total += ggml_nbytes(o);
    std::vector<uint8_t> host_result(out_bytes_total);

    for (int w = 0; w < args.warmup; w++) {
        size_t off = 0;
        for (ggml_tensor * o : outs) { ggml_backend_tensor_get(o, host_result.data() + off, 0, ggml_nbytes(o)); off += ggml_nbytes(o); }
    }
    std::vector<double> d2h_samples;
    d2h_samples.reserve(args.reps);
    for (int r = 0; r < args.reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        size_t off = 0;
        for (ggml_tensor * o : outs) { ggml_backend_tensor_get(o, host_result.data() + off, 0, ggml_nbytes(o)); off += ggml_nbytes(o); }
        auto t1 = std::chrono::high_resolution_clock::now();
        d2h_samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    out.d2h = summarize(d2h_samples);
    out.bw_d2h_gbps = (out.d2h.p50_us > 0) ? ((double)out_bytes_total / 1e9) / (out.d2h.p50_us / 1e6) : 0.0;

    return true;
}

// ============================================================================
// 출력
// ============================================================================
static void print_cpu_table(const std::vector<CpuCaseResult> & rows) {
    std::printf("\n[A: CPU 제자리 GEMV] (gate+up+down x N전문가, 배치 합산 1회 compute 타이밍)\n");
    std::printf("%-10s %-14s %-12s %-12s %-12s\n", "N전문가", "배치bytes", "p50(us)", "p95(us)", "mean(us)");
    for (auto & r : rows) {
        std::printf("%-10d %-14zu %-12.1f %-12.1f %-12.1f\n",
                    r.n_experts, r.batch_bytes, r.compute.p50_us, r.compute.p95_us, r.compute.mean_us);
    }
}

static void print_gpu_table(const std::vector<GpuCaseResult> & rows) {
    std::printf("\n[B: pinned/pageable -> H2D -> GPU GEMV -> D2H]\n");
    std::printf("%-8s %-14s %-16s %-12s %-16s %-12s %-12s %-12s %-16s %-12s\n",
                "N", "가중치bytes", "H2D-pinned p50", "GB/s", "H2D-pageable p50", "GB/s",
                "compute p50", "p95", "D2H p50", "GB/s");
    for (auto & r : rows) {
        std::printf("%-8d %-14zu %-16.1f %-12.2f %-16.1f %-12.2f %-12.1f %-12.1f %-16.1f %-12.2f\n",
                    r.n_experts, r.batch_weight_bytes,
                    r.has_pinned   ? r.h2d_pinned.p50_us   : -1.0, r.bw_pinned_gbps,
                    r.has_pageable ? r.h2d_pageable.p50_us : -1.0, r.bw_pageable_gbps,
                    r.compute.p50_us, r.compute.p95_us,
                    r.d2h.p50_us, r.bw_d2h_gbps);
    }
}

static void write_json(const std::string & path, const Args & args,
                        const std::vector<CpuCaseResult> & cpu_rows,
                        const std::vector<GpuCaseResult> & gpu_rows) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "경고: JSON 저장 실패 (%s)\n", path.c_str()); return; }

    f << "{\n";
    f << "  \"meta\": {\n";
    f << "    \"tool\": \"moe-l2ab\",\n";
    f << "    \"llama_build\": \"b10057\",\n";
    f << "    \"commit\": \"0bd0ec609\",\n";
    f << "    \"generated_utc\": \"" << now_utc_iso() << "\",\n";
    f << "    \"threads_cpu\": " << args.threads << ",\n";
    f << "    \"reps\": " << args.reps << ",\n";
    f << "    \"warmup\": " << args.warmup << ",\n";
    f << "    \"seed\": " << args.seed << ",\n";
    f << "    \"dim\": " << DIM << ",\n";
    f << "    \"weight_bytes_per_matrix\": " << WEIGHT_BYTES << ",\n";
    f << "    \"bias_bytes_per_matrix\": " << BIAS_BYTES << ",\n";
    f << "    \"bytes_per_expert\": " << EXPERT_BYTES << ",\n";
    f << "    \"note_down_input\": \"down GEMV reuses x (SwiGLU 활성화 체이닝 미모델링 -- shape 동일: feed_forward_length==embedding_length==2880, 타이밍은 데이터/체이닝 무관)\"\n";
    f << "  },\n";

    f << "  \"cpu\": [\n";
    for (size_t i = 0; i < cpu_rows.size(); i++) {
        auto & r = cpu_rows[i];
        f << "    {\"n_experts\": " << r.n_experts
          << ", \"batch_bytes\": " << r.batch_bytes
          << ", \"p50_us\": " << r.compute.p50_us
          << ", \"p95_us\": " << r.compute.p95_us
          << ", \"mean_us\": " << r.compute.mean_us
          << ", \"reps\": " << r.compute.reps << "}"
          << (i + 1 < cpu_rows.size() ? ",\n" : "\n");
    }
    f << "  ],\n";

    f << "  \"gpu\": [\n";
    for (size_t i = 0; i < gpu_rows.size(); i++) {
        auto & r = gpu_rows[i];
        f << "    {\"n_experts\": " << r.n_experts
          << ", \"batch_weight_bytes\": " << r.batch_weight_bytes
          << ", \"h2d_pinned\": {\"p50_us\": " << r.h2d_pinned.p50_us
          << ", \"p95_us\": " << r.h2d_pinned.p95_us
          << ", \"bandwidth_gbps\": " << r.bw_pinned_gbps
          << ", \"available\": " << (r.has_pinned ? "true" : "false") << "},\n"
          << "     \"h2d_pageable\": {\"p50_us\": " << r.h2d_pageable.p50_us
          << ", \"p95_us\": " << r.h2d_pageable.p95_us
          << ", \"bandwidth_gbps\": " << r.bw_pageable_gbps
          << ", \"available\": " << (r.has_pageable ? "true" : "false") << "},\n"
          << "     \"compute\": {\"p50_us\": " << r.compute.p50_us << ", \"p95_us\": " << r.compute.p95_us << "},\n"
          << "     \"d2h\": {\"p50_us\": " << r.d2h.p50_us << ", \"p95_us\": " << r.d2h.p95_us
          << ", \"bandwidth_gbps\": " << r.bw_d2h_gbps << "}}"
          << (i + 1 < gpu_rows.size() ? ",\n" : "\n");
    }
    f << "  ]\n";
    f << "}\n";

    std::printf("\nJSON 저장: %s\n", path.c_str());
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char ** argv) {
    Args args;
    bool parsed = parse_args(argc, argv, args);

    // ---- 여기까지: 순수 인자 파싱, ggml/backend 호출 전무 ----
    if (args.help || !parsed) {
        if (!parsed && !args.err.empty()) std::fprintf(stderr, "인자 오류: %s\n\n", args.err.c_str());
        print_usage(argv[0]);
        return parsed ? 0 : 1;
    }

    // -----------------------------------------------------------------------
    // 아래부터 backend 초기화(ggml_backend_load_all -> CPU/CUDA 디바이스 init).
    // G3-0a 절대계약: 이 지점 이후는 구현 에이전트가 실행하지 않는다(메인이 실행).
    // -----------------------------------------------------------------------
    ggml_backend_load_all();

    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t dev_gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

    if (args.list_devices) {
        size_t n = ggml_backend_dev_count();
        std::printf("등록된 backend device: %zu개\n", n);
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(d, &props);
            std::printf("  [%zu] %-10s type=%d  %-40s free=%.2fGiB total=%.2fGiB\n",
                        i, ggml_backend_dev_name(d), (int)props.type, props.description,
                        props.memory_free / 1073741824.0, props.memory_total / 1073741824.0);
        }
        return 0;
    }

    bool need_cpu = (args.mode == "cpu" || args.mode == "all");
    bool need_gpu = (args.mode == "gpu" || args.mode == "all");

    if (need_cpu && !dev_cpu) { std::fprintf(stderr, "오류: CPU backend device를 찾지 못했습니다\n"); return 1; }
    if (need_gpu && !dev_gpu) {
        std::fprintf(stderr, "오류: GPU(CUDA) backend device를 찾지 못했습니다 -- ggml-cuda.dll이 exe와 같은 폴더/CWD에 있는지 확인하세요\n");
        return 1;
    }

    ggml_backend_ptr backend_cpu, backend_gpu;
    if (need_cpu) {
        backend_cpu.reset(ggml_backend_dev_init(dev_cpu, nullptr));
        if (!backend_cpu) { std::fprintf(stderr, "오류: CPU backend 초기화 실패\n"); return 1; }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev_cpu);
        auto set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads_fn) {
            set_n_threads_fn(backend_cpu.get(), args.threads);
        } else {
            std::fprintf(stderr, "경고: ggml_backend_set_n_threads proc address를 찾지 못해 스레드 수 기본값 사용\n");
        }
    }
    if (need_gpu) {
        backend_gpu.reset(ggml_backend_dev_init(dev_gpu, nullptr));
        if (!backend_gpu) { std::fprintf(stderr, "오류: GPU backend 초기화 실패\n"); return 1; }
    }

    std::printf("moe-l2ab: mode=%s n=[", args.mode.c_str());
    for (size_t i = 0; i < args.n_list.size(); i++) std::printf("%d%s", args.n_list[i], i + 1 < args.n_list.size() ? "," : "");
    std::printf("] reps=%d warmup=%d threads=%d host-buf=%s seed=%u\n",
                args.reps, args.warmup, args.threads, args.host_buf.c_str(), args.seed);

    std::vector<CpuCaseResult> cpu_rows;
    std::vector<GpuCaseResult> gpu_rows;

    if (need_cpu) {
        for (int n : args.n_list) {
            CpuCaseResult r; std::string err;
            if (!run_cpu_case(backend_cpu.get(), n, args, r, err)) {
                std::fprintf(stderr, "CPU case(N=%d) 실패: %s\n", n, err.c_str());
                return 1;
            }
            cpu_rows.push_back(r);
        }
        print_cpu_table(cpu_rows);
    }

    if (need_gpu) {
        for (int n : args.n_list) {
            GpuCaseResult r; std::string err;
            if (!run_gpu_case(dev_gpu, backend_gpu.get(), n, args, r, err)) {
                std::fprintf(stderr, "GPU case(N=%d) 실패: %s\n", n, err.c_str());
                return 1;
            }
            gpu_rows.push_back(r);
        }
        print_gpu_table(gpu_rows);
    }

    if (!args.json_out.empty()) {
        write_json(args.json_out, args, cpu_rows, gpu_rows);
    }

    return 0;
}
