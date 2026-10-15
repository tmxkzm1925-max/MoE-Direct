// MoE 프리페치 Phase A — 무행동 계측 모듈 본체.
//
// 사양: PREFETCH_DESIGN.md §3(FROZEN v1) · wire: bench/moe-direct/SPEC_PHASEA_WIRE.md
//
// 이 모듈은 "관측만" 한다 — 캐시 상태·읽기 시퀀스·LRU tick·출력 토큰 중 어느 것도 바꾸지 않는다.
// accuracy 모드의 eval callback은 backend split을 graph view로 재분할하므로 **타이밍만** 바뀐다
// (그래서 metrics에 schedule_perturbed=true를 박고 성능·창 산출에서 제외한다 — 사양 §3.3).
//
// ★동시성: 자체 뮤텍스 없음. 근거·전제는 ggml-moe-phasea.h 상단 주석 참조.
// Windows 전용 — 비Windows는 파일 하단 스텁.

#include "ggml-moe-phasea.h"
#include "ggml-moe-direct.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

// ===========================================================================
// §1. wire 구조체 (SPEC_PHASEA_WIRE.md §2~§6 — 리틀엔디언·고정폭)
// ===========================================================================

#pragma pack(push, 1)

struct PaHeader {
    char     magic[8];          // "MOEPA1\0\0"
    uint32_t layout_version;
    uint32_t file_kind;         // 1=events 2=ops 3=ledger 4=sets
    uint32_t record_bytes;
    uint32_t phasea_mode;
    uint64_t record_count;
    uint64_t payload_digest;
    uint64_t qpc_frequency;
    uint64_t t0_qpc;
    uint32_t pid;
    uint32_t complete;
    uint32_t n_expert;
    uint32_t n_moe_layers;
    uint32_t n_slots;
    uint32_t n_expert_used;
    uint64_t boot_id;
    uint64_t total_requests;
    uint64_t dropped_records;
    uint8_t  reserved[24];
};

struct PaEvent {
    uint32_t graph_seq;
    uint16_t layer;
    uint8_t  flags;             // b0 phase(1=prefill) b1 had_miss
    uint8_t  part;
    uint64_t d_qpc;
    uint64_t r_qpc;
    uint64_t s_qpc;
    uint64_t c_qpc;
    uint64_t q_busy_qpc;
    uint32_t q_epochs;
    uint32_t n_ids;
    uint32_t n_miss;
    uint32_t n_hit;
    uint64_t miss_bytes;
    uint32_t request_index;
    uint32_t reserved;
};

struct PaOp {
    uint64_t record_id;
    uint64_t file_offset;
    uint32_t bytes;
    uint16_t expert;
    uint16_t layer;
    uint32_t graph_seq;
    uint32_t batch_order;
    uint64_t submit_seq;
    uint64_t submit_qpc;
    uint64_t retire_seq;
    uint64_t retire_qpc;
    int32_t  slot;
    uint8_t  flags;             // b0 pending b1 phase(prefill)
    uint8_t  part;
    uint16_t reserved;
};

struct PaLedger {
    uint8_t  kind;              // 1 ACQ_CALL 2 ACQ_EXPERT 3 EVICT 4 REL_CALL 5 REL_EXPERT 6 SLOT_INIT
    uint8_t  flags;             // b0 hit b1 canonical b2 phase(prefill)
    uint16_t expert;
    uint16_t layer;
    uint16_t part;
    int32_t  slot;
    uint32_t c32;               // graph_seq (SLOT_INIT: owner_record)
    uint64_t t_qpc;
    uint64_t val;
};

struct PaSet {
    uint32_t graph_seq;
    uint16_t layer;
    uint8_t  flags;             // b0 phase b1 pred_a0_valid b2 pred_b_valid b3 actual_truncated
    uint8_t  n_actual;
    uint8_t  n_miss;
    uint8_t  n_pred;
    uint16_t reserved;
    uint32_t request_index;
    uint16_t actual [16];
    uint16_t miss   [16];
    uint16_t pred_a0[16];
    uint16_t pred_b [16];
};

#pragma pack(pop)

static_assert(sizeof(PaHeader) == 128, "PaHeader must be exactly 128 bytes (ABI contract)");
static_assert(sizeof(PaEvent)  ==  80, "PaEvent must be exactly 80 bytes (ABI contract)");
static_assert(sizeof(PaOp)     ==  72, "PaOp must be exactly 72 bytes (ABI contract)");
static_assert(sizeof(PaLedger) ==  32, "PaLedger must be exactly 32 bytes (ABI contract)");
static_assert(sizeof(PaSet)    == 144, "PaSet must be exactly 144 bytes (ABI contract)");

enum : uint8_t {
    LED_ACQ_CALL   = 1,
    LED_ACQ_EXPERT = 2,
    LED_EVICT      = 3,
    LED_REL_CALL   = 4,
    LED_REL_EXPERT = 5,
    LED_SLOT_INIT  = 6,
};

constexpr uint16_t PA_NA_U16  = 0xFFFFu;
constexpr uint32_t PA_NA_U32  = 0xFFFFFFFFu;
constexpr int32_t  PA_PRED_K  = 16;     // top-16 저장(앞 8 = @8)
constexpr int32_t  PA_SET_CAP = 16;     // set 레코드 actual/miss 고정폭
constexpr uint64_t FNV_SEED   = 0xcbf29ce484222325ull;

// ===========================================================================
// §2. 유틸
// ===========================================================================

inline uint64_t fnv1a64(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::string to_hex64(uint64_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%016llx", (unsigned long long)v);
    return std::string(buf);
}

void pa_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[moe-phasea] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

uint64_t qpc_now_raw() {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (uint64_t)li.QuadPart;
}

// ===========================================================================
// §3. 전역 상태
// ===========================================================================

struct SigAcc {
    // A1 (per target mlpos)
    std::vector<uint64_t> inter8, inter16, denom, full8, full16, samples;
    std::vector<uint64_t> absent8_cnt, absent16_cnt, absent8_bytes, absent16_bytes;
    // A1-m (per target mlpos)
    std::vector<uint64_t> mcov8, mcov16, mdenom, mfull8, mfull16, msamples;
    // joint histogram(전 층 합산) — 행=|miss_set| 0..8, 열=covered 0..8
    uint64_t joint8[9][9];
    uint64_t joint16[9][9];

    void resize(size_t n) {
        for (std::vector<uint64_t> * v : { &inter8, &inter16, &denom, &full8, &full16, &samples,
                                           &absent8_cnt, &absent16_cnt, &absent8_bytes, &absent16_bytes,
                                           &mcov8, &mcov16, &mdenom, &mfull8, &mfull16, &msamples }) {
            v->assign(n, 0);
        }
        std::memset(joint8,  0, sizeof(joint8));
        std::memset(joint16, 0, sizeof(joint16));
    }
    void clear() { resize(inter8.size()); }
};

struct PredSlot {
    bool     valid_a0 = false;
    bool     valid_b  = false;
    uint16_t a0[PA_PRED_K];
    uint16_t b [PA_PRED_K];
    uint32_t absent_a0_8 = 0, absent_a0_16 = 0;
    uint32_t absent_b_8  = 0, absent_b_16  = 0;
};

// 진행 중인 acquire 1건의 문맥(전 구간 g_mod.mutex 보유이므로 단일 슬롯으로 충분).
struct CurAcquire {
    bool     open      = false;
    bool     canonical = false;
    int32_t  mlpos     = -1;
    int32_t  part      = -1;
    int32_t  n_ids     = 0;
    uint8_t  phase     = 0;   // 0 decode 1 prefill
    uint32_t graph_seq = 0;   // ★E — enter 시 포획한 소유 그래프(exit 가 경계를 넘어도 불변)
    uint64_t d_qpc     = 0;
    uint64_t s_qpc     = 0;   // 첫 ReadFile 직전
    uint64_t c_qpc     = 0;   // 마지막 회수
    uint64_t q_busy    = 0;
    uint32_t q_epochs  = 0;
    int32_t  q_last_n  = 0;
    uint64_t q_last_t  = 0;
    int32_t  n_hit     = 0;
    std::vector<uint16_t> actual;   // canonical 전용(요청 순서 = 오름차순)
    std::vector<uint16_t> miss;
};

struct PaState {
    // ── 모드/식별 ──
    bool        resolved = false;
    int32_t     mode     = GGML_MOE_PHASEA_OFF;
    std::string out_prefix;
    uint32_t    buf_mb   = 1024;
    uint64_t    boot_id  = 0;
    uint64_t    qpc_freq = 0;
    uint64_t    t0_qpc   = 0;
    bool        eval_cb_installed = false;

    // ── 모델 상수(seal 통지) ──
    bool     sealed = false;
    int32_t  n_expert = 0, n_expert_used = 0, n_moe_layers = 0, n_slots = 0;
    std::vector<int32_t>  moe_layers;       // mlpos -> 실제 layer id
    std::vector<uint64_t> layer_stride;     // mlpos -> record stride bytes
    std::vector<int32_t>  layer_to_mlpos;   // layer id -> mlpos (없으면 -1)

    // ── 버퍼 ──
    std::vector<uint8_t> buf_events, buf_ops, buf_ledger, buf_sets;
    uint64_t n_events = 0, n_ops = 0, n_ledger = 0, n_sets = 0;
    uint64_t dropped  = 0;
    bool     complete = true;
    bool     buf_overflowed = false;   // ★latch — 최초 상한 초과 후 모든 append 거부(Codex R1 권고)
    uint64_t buf_cap_bytes = 0;

    // ── run/graph 문맥 ──
    uint32_t graph_seq_next = 0;
    uint32_t request_next   = 0;
    bool     run_active     = false;
    uint32_t request_index  = PA_NA_U32;
    uint64_t t_run_begin    = 0;
    bool     in_graph       = false;
    uint32_t cur_graph_seq  = PA_NA_U32;
    int32_t  cur_n_outputs  = -1;
    // ★r1 — 섬도 드래프트 트리거용 identity 스냅샷(적격 decode graph 종료 시에만 갱신)
    int32_t  id_valid       = 0;
    uint32_t id_request     = PA_NA_U32;
    uint32_t id_graph_seq   = PA_NA_U32;
    uint32_t id_decode_ord  = 0;
    uint8_t  cur_phase      = 0;
    uint32_t cur_n_tokens   = 0;

    // ── graph 스코프 ──
    // ★훅 재개봉 라운드 E — graph-local 플래그를 **소유권(graph_seq) 판정**으로 교체한다.
    //   구조: bare boolean + graph_begin 일괄 리셋은 "리셋을 못 받은 그래프"(= graph_begin 없이
    //   계산되는 prelude/capability 그래프)나 경계를 넘는 exit 에서 **잔존 플래그**가 다음 그래프의
    //   canonical/emit 판정을 오염시킨다. 소유권 키는 리셋 여부와 무관하게 "이 그래프의 것인가"만
    //   보므로 그 원천이 사라진다(값 = 그 mlpos 를 마지막으로 소유한 graph_seq · NA = 미소유).
    std::vector<uint32_t> layer_seen_gs;   // mlpos -> canonical acquire 를 낸 graph_seq
    std::vector<uint32_t> set_emit_gs;     // mlpos -> set 레코드를 낸(또는 억제 판정한) graph_seq
    std::vector<PredSlot> pred;         // target mlpos -> 예측

    CurAcquire cur;
    int32_t    rel_mlpos = -1;          // release 문맥(acquire 와 분리)
    int32_t    rel_part  = -1;
    uint64_t   rel_t     = 0;           // REL_CALL 진입 시각(REL_EXPERT "소속 CALL과 동일" 계약용)

    // ── 요청 단위 집계(run_begin에서 리셋) ──
    SigAcc acc_a0, acc_b;
    std::vector<uint64_t> a3_miss_count, a3_miss_bytes;
    uint64_t r_decode_graphs = 0, r_prefill_graphs = 0;
    uint64_t r_acquire_calls = 0, r_acquire_canonical = 0, r_release_calls = 0;
    uint64_t r_join_samples = 0, r_join_dup = 0, r_join_unmatched = 0, r_join_overflow = 0;
    uint64_t r_ev = 0, r_op = 0, r_led = 0, r_set = 0;
    uint64_t r_nodes_digest = FNV_SEED, r_splits_digest = FNV_SEED, r_graphs = 0;
    uint64_t r_pred_calls = 0, r_pred_ns_total = 0, r_pred_ns_max = 0;
    uint64_t r_tap_calls = 0, r_tap_bytes = 0, r_tap_skipped = 0;

    // ── accuracy 가중치 스냅샷 ──
    bool     snap_configured = false;
    bool     snap_sealed     = false;
    int32_t  n_layer_exec = 0, n_embd = 0;
    float    rms_eps = 1e-6f;
    std::vector<float>   snap_router;   // (n_layer_exec-1) * n_embd * n_expert
    std::vector<float>   snap_norm;     // (n_layer_exec-1) * n_embd
    std::vector<uint8_t> snap_have;
    std::vector<float>   tap_buf, norm_buf, logit_buf;
    std::vector<char>    topk_used;

    // ── flush ──
    std::vector<json> run_records;
    bool flushed = false;
};

PaState g_pa;

// ===========================================================================
// §4. env 해소
// ===========================================================================

void resolve_env_once() {
    if (g_pa.resolved) return;
    g_pa.resolved = true;

    std::string m = env_str("MOE_DIRECT_PHASEA");
    const size_t b = m.find_first_not_of(" \t\r\n");
    const size_t e = m.find_last_not_of(" \t\r\n");
    m = (b == std::string::npos) ? std::string() : m.substr(b, e - b + 1);
    for (auto & c : m) c = (char)tolower((unsigned char)c);

    if      (m.empty() || m == "off" || m == "0") g_pa.mode = GGML_MOE_PHASEA_OFF;
    else if (m == "accuracy")                     g_pa.mode = GGML_MOE_PHASEA_ACCURACY;
    else if (m == "timing")                       g_pa.mode = GGML_MOE_PHASEA_TIMING;
    else {
        g_pa.mode = GGML_MOE_PHASEA_OFF;
        pa_log("MOE_DIRECT_PHASEA unknown value '%s' - treating as off", m.c_str());
    }

    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_pa.qpc_freq = (uint64_t)f.QuadPart;
    g_pa.t0_qpc   = qpc_now_raw();

    if (g_pa.mode == GGML_MOE_PHASEA_OFF) return;

    g_pa.out_prefix = env_str("MOE_DIRECT_PHASEA_OUT");
    g_pa.buf_mb = 1024;
    {
        const std::string s = env_str("MOE_DIRECT_PHASEA_BUF_MB");
        if (!s.empty()) {
            char * endp = nullptr;
            const long v = strtol(s.c_str(), &endp, 10);
            if (endp && *endp == '\0' && v > 0) g_pa.buf_mb = (uint32_t)v;
        }
    }
    g_pa.buf_cap_bytes = (uint64_t)g_pa.buf_mb * 1024ull * 1024ull;

    // boot_id = t0_qpc 와 pid 를 섞은 프로세스 고유값(사이드카↔metrics 조인 키).
    uint64_t h = FNV_SEED;
    const uint32_t pid = (uint32_t)GetCurrentProcessId();
    h = fnv1a64(h, &g_pa.t0_qpc, sizeof(g_pa.t0_qpc));
    h = fnv1a64(h, &pid, sizeof(pid));
    g_pa.boot_id = h;

    if (g_pa.out_prefix.empty()) {
        pa_log("MOE_DIRECT_PHASEA is enabled but MOE_DIRECT_PHASEA_OUT is not set - sidecar not written"
               " (complete=0; this arm is not judgeable)");
        g_pa.complete = false;
    }
}

inline bool pa_on() { resolve_env_once(); return g_pa.mode != GGML_MOE_PHASEA_OFF; }

// ===========================================================================
// §5. 버퍼 append (상한 초과 = 기록 중단 + complete=0. 요청 중 분할 flush 금지)
// ===========================================================================

uint64_t buf_total_bytes() {
    return (uint64_t)g_pa.buf_events.size() + g_pa.buf_ops.size()
         + g_pa.buf_ledger.size() + g_pa.buf_sets.size();
}

bool buf_append(std::vector<uint8_t> & v, const void * rec, size_t n, uint64_t * counter) {
    if (g_pa.buf_overflowed || buf_total_bytes() + n > g_pa.buf_cap_bytes) {
        if (!g_pa.buf_overflowed) {
            pa_log("sidecar buffer cap (%u MiB) exceeded - recording stopped, complete=0 "
                   "(splitting a flush mid-request is forbidden: PREFETCH_DESIGN section 3.4 item 6 re-review condition)", g_pa.buf_mb);
        }
        g_pa.buf_overflowed = true;   // ★latch — 이후 더 작은 레코드도 전부 거부(부분 기록 방지)
        g_pa.complete = false;
        g_pa.dropped++;
        return false;
    }
    const uint8_t * p = (const uint8_t *)rec;
    v.insert(v.end(), p, p + n);
    (*counter)++;
    return true;
}

inline void led_push(const PaLedger & r) {
    if (buf_append(g_pa.buf_ledger, &r, sizeof(r), &g_pa.n_ledger) && g_pa.run_active) g_pa.r_led++;
}

// ===========================================================================
// §6. 사이드카 파일 기록
// ===========================================================================

bool write_sidecar(const std::string & path, uint32_t kind, uint32_t record_bytes,
                   uint64_t record_count, const std::vector<uint8_t> & payload,
                   uint64_t * out_digest) {
    PaHeader h{};
    std::memcpy(h.magic, "MOEPA1\0\0", 8);
    h.layout_version  = 1;
    h.file_kind       = kind;
    h.record_bytes    = record_bytes;
    h.phasea_mode     = (uint32_t)g_pa.mode;
    h.record_count    = record_count;
    h.payload_digest  = fnv1a64(FNV_SEED, payload.data(), payload.size());
    h.qpc_frequency   = g_pa.qpc_freq;
    h.t0_qpc          = g_pa.t0_qpc;
    h.pid             = (uint32_t)GetCurrentProcessId();
    h.complete        = g_pa.complete ? 1u : 0u;
    h.n_expert        = (uint32_t)g_pa.n_expert;
    h.n_moe_layers    = (uint32_t)g_pa.n_moe_layers;
    h.n_slots         = (uint32_t)g_pa.n_slots;
    h.n_expert_used   = (uint32_t)g_pa.n_expert_used;
    h.boot_id         = g_pa.boot_id;
    h.total_requests  = g_pa.request_next;
    h.dropped_records = g_pa.dropped;
    if (out_digest) *out_digest = h.payload_digest;

    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0);
    std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), &w[0], n);

    HANDLE f = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        pa_log("cannot create sidecar (CREATE_NEW) path=%s err=%lu", path.c_str(),
               (unsigned long)GetLastError());
        return false;
    }
    DWORD wr = 0;
    bool ok = WriteFile(f, &h, sizeof(h), &wr, nullptr) != 0 && wr == sizeof(h);
    if (ok && !payload.empty()) {
        size_t off = 0;
        while (off < payload.size()) {
            const DWORD chunk = (DWORD)std::min<size_t>(payload.size() - off, (size_t)32 * 1024 * 1024);
            if (!WriteFile(f, payload.data() + off, chunk, &wr, nullptr) || wr != chunk) { ok = false; break; }
            off += chunk;
        }
    }
    if (ok) ok = FlushFileBuffers(f) != 0;
    CloseHandle(f);
    if (!ok) pa_log("cannot write sidecar path=%s err=%lu", path.c_str(), (unsigned long)GetLastError());
    return ok;
}

// ===========================================================================
// §7. 신호 계산 (accuracy 모드)
//     y = RMSNorm_post[t](x) ;  logits[e] = <W_router[t][e], y> ;  top-16(내림차순)
// ===========================================================================

inline int32_t target_index(int32_t target_layer) { return target_layer - 1; }

void compute_topk(const float * x, int32_t target_layer, uint16_t out[PA_PRED_K]) {
    const int32_t ne = g_pa.n_embd;
    const int32_t nx = g_pa.n_expert;
    const int32_t ti = target_index(target_layer);
    const float * w_norm = g_pa.snap_norm.data()   + (size_t)ti * (size_t)ne;
    const float * w_rt   = g_pa.snap_router.data() + (size_t)ti * (size_t)ne * (size_t)nx;

    // RMSNorm(ggml_rms_norm 과 동일 산식: x * 1/sqrt(mean(x^2)+eps), 이어서 weight 곱)
    double ss = 0.0;
    for (int32_t i = 0; i < ne; i++) ss += (double)x[i] * (double)x[i];
    const float scale = (float)(1.0 / std::sqrt(ss / (double)ne + (double)g_pa.rms_eps));
    float * y = g_pa.norm_buf.data();
    for (int32_t i = 0; i < ne; i++) y[i] = x[i] * scale * w_norm[i];

    // gemv: logits[e] = dot(W[e*ne ..], y)
    float * lg = g_pa.logit_buf.data();
    for (int32_t e = 0; e < nx; e++) {
        const float * row = w_rt + (size_t)e * (size_t)ne;
        float acc = 0.0f;
        for (int32_t i = 0; i < ne; i++) acc += row[i] * y[i];
        lg[e] = acc;
    }

    // top-16 부분 선택(동점은 낮은 expert id 우선 — 결정론).
    char * used = g_pa.topk_used.data();
    std::memset(used, 0, (size_t)nx);
    const int32_t k = std::min<int32_t>(PA_PRED_K, nx);
    for (int32_t r = 0; r < k; r++) {
        int32_t best = -1;
        float   bv   = 0.0f;
        for (int32_t e = 0; e < nx; e++) {
            if (used[e]) continue;
            if (best < 0 || lg[e] > bv) { best = e; bv = lg[e]; }
        }
        used[best] = 1;
        out[r] = (uint16_t)best;
    }
    for (int32_t r = k; r < PA_PRED_K; r++) out[r] = PA_NA_U16;
}

// 신호 시점 미상주 예측 expert 수(@8 · @16).
void absent_of(int32_t target_mlpos, const uint16_t pred[PA_PRED_K], uint32_t * a8, uint32_t * a16) {
    int32_t ids[PA_PRED_K];
    for (int32_t i = 0; i < PA_PRED_K; i++) ids[i] = (pred[i] == PA_NA_U16) ? -1 : (int32_t)pred[i];
    *a8  = (uint32_t)ggml_moe_direct_absent_count(target_mlpos, ids, 8);
    *a16 = (uint32_t)ggml_moe_direct_absent_count(target_mlpos, ids, PA_PRED_K);
}

// ===========================================================================
// §8. eval callback 이름 파싱 — "attn_residual-<il>"(a0 · h_l) / "l_out-<il>"(b · x_{l+1})
// ===========================================================================

int parse_tap_name(const char * name, int32_t * out_il) {
    const char * digits = nullptr;
    int kind = 0;
    if      (std::strncmp(name, "attn_residual-", 14) == 0) { kind = 1; digits = name + 14; }
    else if (std::strncmp(name, "l_out-",          6) == 0) { kind = 2; digits = name + 6;  }
    else return 0;
    if (!*digits) return 0;
    int32_t v = 0;
    for (const char * p = digits; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        v = v * 10 + (*p - '0');
        if (v > 100000) return 0;
    }
    *out_il = v;
    return kind;
}

// ===========================================================================
// §9. 요청 레코드 조립 헬퍼
// ===========================================================================

json sig_to_json(const SigAcc & a, size_t n) {
    auto slice = [&](const std::vector<uint64_t> & v) {
        json arr = json::array();
        for (size_t i = 1; i < n; i++) arr.push_back(v[i]);   // target mlpos 1..n-1
        return arr;
    };
    auto hist = [&](const uint64_t h[9][9]) {
        json rows = json::array();
        for (int i = 0; i < 9; i++) {
            json r = json::array();
            for (int j = 0; j < 9; j++) r.push_back(h[i][j]);
            rows.push_back(r);
        }
        return rows;
    };
    return json{
        {"a1", json{
            {"inter8", slice(a.inter8)}, {"inter16", slice(a.inter16)}, {"denom", slice(a.denom)},
            {"full_cover8", slice(a.full8)}, {"full_cover16", slice(a.full16)}, {"samples", slice(a.samples)},
            {"absent_pred8_count", slice(a.absent8_cnt)}, {"absent_pred16_count", slice(a.absent16_cnt)},
            {"absent_pred8_bytes", slice(a.absent8_bytes)}, {"absent_pred16_bytes", slice(a.absent16_bytes)},
        }},
        {"a1m", json{
            {"cov8", slice(a.mcov8)}, {"cov16", slice(a.mcov16)}, {"denom", slice(a.mdenom)},
            {"full_cover8", slice(a.mfull8)}, {"full_cover16", slice(a.mfull16)}, {"samples", slice(a.msamples)},
            {"joint_hist8", hist(a.joint8)}, {"joint_hist16", hist(a.joint16)},
        }},
    };
}

// canonical acquire 종료 시 set 레코드 방출 + A1/A1-m 조인 집계.
void emit_join(void) {
    CurAcquire & c = g_pa.cur;
    const int32_t mlpos = c.mlpos;
    // ★E — 소유권 판정: 이 **acquire 자신의 graph_seq**(enter 시 포획) 기준으로 중복을 가른다.
    //   live `cur_graph_seq` 를 쓰면 경계를 넘은 exit 가 다음 그래프의 칸을 선점해(마크-후-필터
    //   순서와 겹쳐) 그 층의 set 을 조용히 떨어뜨린다 — 관측된 결손의 구조적 원천.
    if (g_pa.set_emit_gs[(size_t)mlpos] == c.graph_seq) { if (g_pa.run_active) g_pa.r_join_dup++; return; }
    g_pa.set_emit_gs[(size_t)mlpos] = c.graph_seq;

    // ★set 레코드는 decode graph 전용(prefill actual 집합은 ledger 가 권위 — SPEC_PHASEA_WIRE §6).
    if (c.phase != 0) return;

    const bool truncated = (int32_t)c.actual.size() > PA_SET_CAP || (int32_t)c.miss.size() > PA_SET_CAP;
    if (truncated && g_pa.run_active) g_pa.r_join_overflow++;

    PaSet s{};
    s.graph_seq     = c.graph_seq;   // ★E — 자기 acquire 의 그래프(무-straddle 런은 cur 와 동일)
    s.layer         = (uint16_t)g_pa.moe_layers[(size_t)mlpos];
    s.n_actual      = (uint8_t)std::min<size_t>(c.actual.size(), 255);
    s.n_miss        = (uint8_t)std::min<size_t>(c.miss.size(),   255);
    s.request_index = g_pa.request_index;
    for (int i = 0; i < PA_SET_CAP; i++) {
        s.actual[i] = PA_NA_U16; s.miss[i] = PA_NA_U16;
        s.pred_a0[i] = PA_NA_U16; s.pred_b[i] = PA_NA_U16;
    }
    for (size_t i = 0; i < c.actual.size() && i < (size_t)PA_SET_CAP; i++) s.actual[i] = c.actual[i];
    for (size_t i = 0; i < c.miss.size()   && i < (size_t)PA_SET_CAP; i++) s.miss[i]   = c.miss[i];
    s.flags = (uint8_t)(truncated ? 0x08 : 0x00);

    const PredSlot * ps = (mlpos >= 1 && (size_t)mlpos < g_pa.pred.size())
                        ? &g_pa.pred[(size_t)mlpos] : nullptr;
    if (ps && ps->valid_a0) { s.flags |= 0x02; for (int i = 0; i < PA_PRED_K; i++) s.pred_a0[i] = ps->a0[i]; }
    if (ps && ps->valid_b)  { s.flags |= 0x04; for (int i = 0; i < PA_PRED_K; i++) s.pred_b[i]  = ps->b[i];  }
    s.n_pred = (uint8_t)((ps && (ps->valid_a0 || ps->valid_b)) ? PA_PRED_K : 0);

    if (buf_append(g_pa.buf_sets, &s, sizeof(s), &g_pa.n_sets) && g_pa.run_active) g_pa.r_set++;
    if (mlpos >= 1 && g_pa.run_active) g_pa.r_join_samples++;

    if (!g_pa.run_active || !ps) return;

    auto accumulate = [&](SigAcc & acc, const uint16_t pred[PA_PRED_K], bool valid,
                          uint32_t abs8, uint32_t abs16) {
        if (!valid) return;
        const size_t li = (size_t)mlpos;
        uint64_t i8 = 0, i16 = 0, m8 = 0, m16 = 0;
        for (size_t a = 0; a < c.actual.size(); a++) {
            int at = -1;
            for (int p = 0; p < PA_PRED_K; p++) if (pred[p] == c.actual[a]) { at = p; break; }
            if (at >= 0) { i16++; if (at < 8) i8++; }
        }
        for (size_t a = 0; a < c.miss.size(); a++) {
            int at = -1;
            for (int p = 0; p < PA_PRED_K; p++) if (pred[p] == c.miss[a]) { at = p; break; }
            if (at >= 0) { m16++; if (at < 8) m8++; }
        }
        acc.inter8[li]  += i8;
        acc.inter16[li] += i16;
        acc.denom[li]   += (uint64_t)c.actual.size();
        acc.samples[li] += 1;
        if (i8  == (uint64_t)c.actual.size()) acc.full8[li]  += 1;
        if (i16 == (uint64_t)c.actual.size()) acc.full16[li] += 1;
        acc.absent8_cnt[li]    += abs8;
        acc.absent16_cnt[li]   += abs16;
        acc.absent8_bytes[li]  += (uint64_t)abs8  * g_pa.layer_stride[li];
        acc.absent16_bytes[li] += (uint64_t)abs16 * g_pa.layer_stride[li];

        acc.mcov8[li]    += m8;
        acc.mcov16[li]   += m16;
        acc.mdenom[li]   += (uint64_t)c.miss.size();
        acc.msamples[li] += 1;
        if (m8  == (uint64_t)c.miss.size()) acc.mfull8[li]  += 1;
        if (m16 == (uint64_t)c.miss.size()) acc.mfull16[li] += 1;
        const size_t hm = std::min<size_t>(c.miss.size(), 8);
        acc.joint8 [hm][std::min<uint64_t>(m8,  8)]++;
        acc.joint16[hm][std::min<uint64_t>(m16, 8)]++;
    };
    accumulate(g_pa.acc_a0, ps->a0, ps->valid_a0, ps->absent_a0_8, ps->absent_a0_16);
    accumulate(g_pa.acc_b,  ps->b,  ps->valid_b,  ps->absent_b_8,  ps->absent_b_16);
}

} // anonymous namespace

// ===========================================================================
// §10. 공개 API
// ===========================================================================

extern "C" {

int32_t  ggml_moe_phasea_mode(void)     { resolve_env_once(); return g_pa.mode; }
bool     ggml_moe_phasea_enabled(void)  { return pa_on(); }
uint64_t ggml_moe_phasea_boot_id(void)  { resolve_env_once(); return g_pa.boot_id; }
uint32_t ggml_moe_phasea_buf_mb(void)   { resolve_env_once(); return g_pa.buf_mb; }
uint64_t ggml_moe_phasea_now_qpc(void)  { return qpc_now_raw(); }

bool ggml_moe_phasea_eval_callback_installed(void)      { resolve_env_once(); return g_pa.eval_cb_installed; }
void ggml_moe_phasea_note_eval_callback_installed(void) { resolve_env_once(); g_pa.eval_cb_installed = true; }

const char * ggml_moe_phasea_mode_name(void) {
    resolve_env_once();
    switch (g_pa.mode) {
        case GGML_MOE_PHASEA_ACCURACY: return "accuracy";
        case GGML_MOE_PHASEA_TIMING:   return "timing";
        default:                       return "off";
    }
}

const char * ggml_moe_phasea_out_prefix(void) {
    resolve_env_once();
    return g_pa.out_prefix.c_str();
}

// ---------------------------------------------------------------------------
// seal 통지
// ---------------------------------------------------------------------------
void ggml_moe_phasea_note_seal(int32_t n_expert, int32_t n_expert_used, int32_t n_moe_layers,
                               int32_t n_slots, const int32_t * moe_layers,
                               const uint64_t * layer_stride_bytes) {
    if (!pa_on()) return;
    if (n_expert <= 0 || n_moe_layers <= 0 || n_slots <= 0 || !moe_layers || !layer_stride_bytes) return;

    g_pa.n_expert      = n_expert;
    g_pa.n_expert_used = n_expert_used;
    g_pa.n_moe_layers  = n_moe_layers;
    g_pa.n_slots       = n_slots;
    g_pa.moe_layers.assign(moe_layers, moe_layers + n_moe_layers);
    g_pa.layer_stride.assign(layer_stride_bytes, layer_stride_bytes + n_moe_layers);

    int32_t max_layer = 0;
    for (int32_t i = 0; i < n_moe_layers; i++) max_layer = std::max(max_layer, moe_layers[i]);
    g_pa.layer_to_mlpos.assign((size_t)max_layer + 1, -1);
    for (int32_t i = 0; i < n_moe_layers; i++) g_pa.layer_to_mlpos[(size_t)moe_layers[i]] = i;

    g_pa.layer_seen_gs.assign((size_t)n_moe_layers, PA_NA_U32);
    g_pa.set_emit_gs.assign((size_t)n_moe_layers, PA_NA_U32);
    g_pa.pred.assign((size_t)n_moe_layers, PredSlot{});
    g_pa.acc_a0.resize((size_t)n_moe_layers);
    g_pa.acc_b.resize((size_t)n_moe_layers);
    g_pa.a3_miss_count.assign((size_t)n_moe_layers, 0);
    g_pa.a3_miss_bytes.assign((size_t)n_moe_layers, 0);
    if ((int32_t)g_pa.topk_used.size() < n_expert) g_pa.topk_used.assign((size_t)n_expert, 0);
    g_pa.sealed = true;

    // ★arm 시작 초기 상태 = seal 직후 전 슬롯 empty(owner NA · last_used 0 · lease 0).
    //   ledger 의 다른 어떤 레코드보다 앞선다(SPEC_PHASEA_WIRE §5 순서 계약 1).
    const uint64_t t = qpc_now_raw();
    for (int32_t s = 0; s < n_slots; s++) {
        PaLedger r{};
        r.kind   = LED_SLOT_INIT;
        r.expert = 0;               // seal 직후 lease = 0
        r.layer  = PA_NA_U16;
        r.part   = PA_NA_U16;
        r.slot   = s;
        r.c32    = PA_NA_U32;       // owner_record 없음
        r.t_qpc  = t;
        r.val    = 0;               // last_used = 0
        led_push(r);
    }
}

// ---------------------------------------------------------------------------
// 요청(run) 경계
// ---------------------------------------------------------------------------
void ggml_moe_phasea_run_begin(void) {
    if (!pa_on()) return;
    g_pa.run_active    = true;
    g_pa.request_index = g_pa.request_next++;
    g_pa.id_valid      = 0;   // ★요청 경계 — 이전 요청의 identity 재사용 금지
    g_pa.t_run_begin   = qpc_now_raw();

    g_pa.acc_a0.clear();
    g_pa.acc_b.clear();
    std::fill(g_pa.a3_miss_count.begin(), g_pa.a3_miss_count.end(), 0ull);
    std::fill(g_pa.a3_miss_bytes.begin(), g_pa.a3_miss_bytes.end(), 0ull);
    g_pa.r_decode_graphs = g_pa.r_prefill_graphs = 0;
    g_pa.r_acquire_calls = g_pa.r_acquire_canonical = g_pa.r_release_calls = 0;
    g_pa.r_join_samples = g_pa.r_join_dup = g_pa.r_join_unmatched = g_pa.r_join_overflow = 0;
    g_pa.r_ev = g_pa.r_op = g_pa.r_led = g_pa.r_set = 0;
    g_pa.r_nodes_digest = g_pa.r_splits_digest = FNV_SEED;
    g_pa.r_graphs = 0;
    g_pa.r_pred_calls = g_pa.r_pred_ns_total = g_pa.r_pred_ns_max = 0;
    g_pa.r_tap_calls = g_pa.r_tap_bytes = g_pa.r_tap_skipped = 0;
}

void ggml_moe_phasea_run_end(uint32_t predicted_n, uint32_t prompt_n) {
    if (!pa_on() || !g_pa.run_active) return;
    const uint64_t t_end = qpc_now_raw();
    const size_t   n     = (size_t)g_pa.n_moe_layers;

    json counts;
    counts["n_layer_exec"]  = g_pa.n_moe_layers;
    counts["n_expert"]      = g_pa.n_expert;
    counts["n_expert_used"] = g_pa.n_expert_used;
    counts["n_slots"]       = g_pa.n_slots;
    {
        json ids = json::array(), tids = json::array();
        for (size_t i = 0; i < n; i++) ids.push_back(g_pa.moe_layers[i]);
        for (size_t i = 1; i < n; i++) tids.push_back(g_pa.moe_layers[i]);
        counts["layer_ids"]        = ids;
        counts["target_layer_ids"] = tids;
    }
    counts["n_graphs"] = json{{"decode", g_pa.r_decode_graphs}, {"prefill", g_pa.r_prefill_graphs}};
    counts["acquire_calls"]           = g_pa.r_acquire_calls;
    counts["acquire_calls_canonical"] = g_pa.r_acquire_canonical;
    counts["release_calls"]           = g_pa.r_release_calls;

    const uint64_t expected = g_pa.r_decode_graphs * (n > 0 ? (uint64_t)(n - 1) : 0ull);
    counts["join"] = json{
        {"samples",   g_pa.r_join_samples},
        {"duplicate", g_pa.r_join_dup},
        {"unmatched", g_pa.r_join_unmatched},
        {"missing",   expected > g_pa.r_join_samples ? expected - g_pa.r_join_samples : 0ull},
        {"overflow",  g_pa.r_join_overflow},
        {"expected",  expected},
    };
    {
        const json ja0 = sig_to_json(g_pa.acc_a0, n);
        const json jb  = sig_to_json(g_pa.acc_b,  n);
        counts["a1"]  = json{{"a0", ja0["a1"]},  {"b", jb["a1"]}};
        counts["a1m"] = json{{"a0", ja0["a1m"]}, {"b", jb["a1m"]}};
    }
    {
        json mc = json::array(), mb = json::array();
        for (size_t i = 0; i < n; i++) { mc.push_back(g_pa.a3_miss_count[i]); mb.push_back(g_pa.a3_miss_bytes[i]); }
        counts["a3"] = json{{"miss_count", mc}, {"miss_bytes", mb}};
    }
    counts["sidecar_records"] = json{
        {"events", g_pa.r_ev}, {"ops", g_pa.r_op}, {"ledger", g_pa.r_led}, {"sets", g_pa.r_set}};
    counts["graph_shape"] = json{
        {"n_graphs",      g_pa.r_graphs},
        {"nodes_digest",  to_hex64(g_pa.r_nodes_digest)},
        {"splits_digest", to_hex64(g_pa.r_splits_digest)}};

    // ── ★accuracy 관측 완전성 게이트(Codex R1 필수 2 — fail-open 봉쇄) ──
    //    콜백 미설치·형상 불일치 스킵·tap 결손은 로그가 아니라 arm 무효(complete=0)다.
    //    expected = 2신호 × target (n-1)층 × decode graph 수(ask 게이트가 decode 전용이므로).
    if (g_pa.mode == GGML_MOE_PHASEA_ACCURACY) {
        const uint64_t expected_taps = 2ull * g_pa.r_decode_graphs * (n > 0 ? (uint64_t)(n - 1) : 0ull);
        const bool pass = g_pa.eval_cb_installed && g_pa.r_tap_skipped == 0 &&
                          g_pa.r_tap_calls == expected_taps && g_pa.r_pred_calls == g_pa.r_tap_calls;
        counts["accuracy_gate"] = json{
            {"expected_taps", expected_taps},
            {"tap_calls",     g_pa.r_tap_calls},
            {"tap_skipped",   g_pa.r_tap_skipped},
            {"installed",     g_pa.eval_cb_installed ? 1 : 0},
            {"pass",          pass ? 1 : 0}};
        if (!pass) {
            if (g_pa.complete) {
                pa_log("accuracy observation incomplete - installed=%d expected_taps=%llu tap_calls=%llu "
                       "skipped=%llu -> complete=0 (fail-close, SPEC_PHASEA_WIRE section 9)",
                       g_pa.eval_cb_installed ? 1 : 0,
                       (unsigned long long)expected_taps, (unsigned long long)g_pa.r_tap_calls,
                       (unsigned long long)g_pa.r_tap_skipped);
            }
            g_pa.complete = false;
        }
    }

    json timing;
    timing["qpc_frequency"]      = g_pa.qpc_freq;
    timing["t_run_begin_qpc"]    = g_pa.t_run_begin;
    timing["t_run_end_qpc"]      = t_end;
    timing["predicted_n"]        = predicted_n;
    timing["prompt_n"]           = prompt_n;
    timing["predictor_calls"]    = g_pa.r_pred_calls;
    timing["predictor_ns_total"] = g_pa.r_pred_ns_total;
    timing["predictor_ns_max"]   = g_pa.r_pred_ns_max;
    timing["tap_calls"]          = g_pa.r_tap_calls;
    timing["tap_bytes"]          = g_pa.r_tap_bytes;
    timing["tap_skipped"]        = g_pa.r_tap_skipped;

    json rec;
    rec["phasea_mode"]             = ggml_moe_phasea_mode_name();
    rec["phasea_boot_id"]          = to_hex64(g_pa.boot_id);
    rec["request_index"]           = g_pa.request_index;
    rec["schedule_perturbed"]      = (g_pa.mode == GGML_MOE_PHASEA_ACCURACY);
    rec["eval_callback_installed"] = g_pa.eval_cb_installed;
    rec["counts"]                  = counts;
    rec["timing"]                  = timing;
    g_pa.run_records.push_back(std::move(rec));

    g_pa.run_active    = false;
    g_pa.request_index = PA_NA_U32;
    g_pa.id_valid      = 0;
}

// ---------------------------------------------------------------------------
// graph 경계
// ---------------------------------------------------------------------------
void ggml_moe_phasea_graph_begin(uint8_t phase, uint32_t n_tokens) {
    if (!pa_on()) return;
    g_pa.in_graph      = true;
    g_pa.cur_graph_seq = g_pa.graph_seq_next++;
    g_pa.cur_phase     = phase ? 1 : 0;
    g_pa.cur_n_tokens  = n_tokens;
    if (g_pa.sealed) {
        // ★E — layer_seen/set_emitted 일괄 리셋 폐기: 소유권 키(graph_seq)가 그래프 경계를
        //   스스로 가르므로 리셋이 필요 없고, "리셋을 못 받는 그래프"의 잔존 오염도 사라진다.
        for (auto & p : g_pa.pred) { p.valid_a0 = false; p.valid_b = false; }
    }
    if (g_pa.run_active) {
        if (g_pa.cur_phase) g_pa.r_prefill_graphs++; else g_pa.r_decode_graphs++;
    }
}

void ggml_moe_phasea_note_outputs(int32_t n_outputs) {
    if (!pa_on()) return;
    g_pa.cur_n_outputs = n_outputs;
}

void ggml_moe_phasea_identity(struct ggml_moe_phasea_ident * out) {
    if (!out) return;
    if (!pa_on()) { out->valid = 0; out->request_index = 0; out->graph_seq = 0; out->decode_ordinal = 0; return; }
    out->valid          = g_pa.id_valid;
    out->request_index  = g_pa.id_request;
    out->graph_seq      = g_pa.id_graph_seq;
    out->decode_ordinal = g_pa.id_decode_ord;
}

void ggml_moe_phasea_graph_end(int32_t n_nodes, int32_t n_splits) {
    if (!pa_on()) return;
    // 예측은 계산됐는데 대응 canonical acquire가 없던 표본 = join unmatched.
    if (g_pa.sealed && g_pa.run_active && g_pa.cur_phase == 0) {
        for (size_t i = 1; i < g_pa.pred.size(); i++) {
            if ((g_pa.pred[i].valid_a0 || g_pa.pred[i].valid_b) && g_pa.set_emit_gs[i] != g_pa.cur_graph_seq) {
                g_pa.r_join_unmatched++;
            }
        }
    }
    if (g_pa.run_active) {
        const uint32_t nn = (uint32_t)n_nodes;
        const uint32_t ns = (uint32_t)n_splits;
        g_pa.r_nodes_digest  = fnv1a64(g_pa.r_nodes_digest,  &nn, sizeof(nn));
        g_pa.r_splits_digest = fnv1a64(g_pa.r_splits_digest, &ns, sizeof(ns));
        g_pa.r_graphs++;
    }
    // ★r1 차단 1 — 적격 decode graph(phase=decode ∧ n_tokens=1 ∧ n_outputs=1) 가 끝난 순간에만
    //   identity 를 갱신한다. 서버 기동 prelude decode · 프리필 · 다중 출력 batch 는 제외된다.
    if (g_pa.run_active && g_pa.cur_phase == 0 && g_pa.cur_n_tokens == 1 && g_pa.cur_n_outputs == 1) {
        g_pa.id_valid      = 1;
        g_pa.id_request    = g_pa.request_index;
        g_pa.id_graph_seq  = g_pa.cur_graph_seq;
        g_pa.id_decode_ord = g_pa.r_decode_graphs > 0 ? (uint32_t)(g_pa.r_decode_graphs - 1) : 0u;
    } else {
        g_pa.id_valid = 0;
    }
    g_pa.cur_n_outputs = -1;
    g_pa.in_graph      = false;
    g_pa.cur_graph_seq = PA_NA_U32;
}

// ---------------------------------------------------------------------------
// accuracy 가중치 스냅샷
// ---------------------------------------------------------------------------
bool ggml_moe_phasea_snapshot_needed(void) {
    resolve_env_once();
    return g_pa.mode == GGML_MOE_PHASEA_ACCURACY && !g_pa.snap_sealed;
}

int ggml_moe_phasea_snapshot_configure(int32_t n_layer_exec, int32_t n_embd,
                                       int32_t n_expert, int32_t n_expert_used, float rms_eps) {
    resolve_env_once();
    if (g_pa.mode != GGML_MOE_PHASEA_ACCURACY || g_pa.snap_sealed) return 0;
    if (n_layer_exec < 2 || n_embd <= 0 || n_expert <= 0) return -1;

    g_pa.n_layer_exec = n_layer_exec;
    g_pa.n_embd       = n_embd;
    g_pa.rms_eps      = rms_eps;
    if (g_pa.n_expert == 0)      g_pa.n_expert      = n_expert;
    if (g_pa.n_expert_used == 0) g_pa.n_expert_used = n_expert_used;

    const size_t nt = (size_t)(n_layer_exec - 1);
    const size_t rt = nt * (size_t)n_embd * (size_t)n_expert;
    try {
        g_pa.snap_router.assign(rt, 0.0f);
        g_pa.snap_norm.assign(nt * (size_t)n_embd, 0.0f);
        g_pa.snap_have.assign(nt, 0);
        g_pa.tap_buf.assign((size_t)n_embd, 0.0f);
        g_pa.norm_buf.assign((size_t)n_embd, 0.0f);
        g_pa.logit_buf.assign((size_t)n_expert, 0.0f);
        g_pa.topk_used.assign((size_t)n_expert, 0);
    } catch (const std::exception & ex) {
        pa_log("weight snapshot allocation failed(%.1f MiB): %s",
               (double)(rt * sizeof(float)) / (1024.0 * 1024.0), ex.what());
        return -1;
    }
    g_pa.snap_configured = true;
    pa_log("weight snapshot ready - target layers=%zu, n_embd=%d, n_expert=%d, %.2f MiB",
           nt, n_embd, n_expert,
           (double)((rt + nt * (size_t)n_embd) * sizeof(float)) / (1024.0 * 1024.0));
    return 0;
}

int ggml_moe_phasea_snapshot_layer(int32_t target_layer, const struct ggml_tensor * router,
                                   const struct ggml_tensor * post_norm) {
    resolve_env_once();
    if (g_pa.mode != GGML_MOE_PHASEA_ACCURACY || !g_pa.snap_configured || g_pa.snap_sealed) return 0;
    if (target_layer < 1 || target_layer >= g_pa.n_layer_exec) return 0;
    if (!router || !post_norm) return -1;
    if (router->type != GGML_TYPE_F32 || post_norm->type != GGML_TYPE_F32) {
        pa_log("router/post_norm is not F32 (layer=%d) - accuracy signal cannot be computed", target_layer);
        return -1;
    }
    if (router->ne[0] != (int64_t)g_pa.n_embd || router->ne[1] != (int64_t)g_pa.n_expert ||
        post_norm->ne[0] != (int64_t)g_pa.n_embd) {
        pa_log("router/post_norm shape mismatch (layer=%d)", target_layer);
        return -1;
    }
    const size_t ti = (size_t)target_index(target_layer);
    ggml_backend_tensor_get((ggml_tensor *)router,
                            g_pa.snap_router.data() + ti * (size_t)g_pa.n_embd * (size_t)g_pa.n_expert,
                            0, (size_t)g_pa.n_embd * (size_t)g_pa.n_expert * sizeof(float));
    ggml_backend_tensor_get((ggml_tensor *)post_norm,
                            g_pa.snap_norm.data() + ti * (size_t)g_pa.n_embd,
                            0, (size_t)g_pa.n_embd * sizeof(float));
    g_pa.snap_have[ti] = 1;
    return 0;
}

int ggml_moe_phasea_snapshot_seal(void) {
    resolve_env_once();
    if (g_pa.mode != GGML_MOE_PHASEA_ACCURACY) return 0;
    if (g_pa.snap_sealed) return 0;
    if (!g_pa.snap_configured) return -1;
    for (size_t i = 0; i < g_pa.snap_have.size(); i++) {
        if (!g_pa.snap_have[i]) {
            pa_log("weight snapshot incomplete - target layer %zu not filled (accuracy signal disabled, complete=0)", i + 1);
            g_pa.complete = false;
            return -1;
        }
    }
    g_pa.snap_sealed = true;
    return 0;
}

uint64_t ggml_moe_phasea_snapshot_bytes(void) {
    return (uint64_t)(g_pa.snap_router.size() + g_pa.snap_norm.size()) * sizeof(float);
}

// ---------------------------------------------------------------------------
// eval callback (accuracy 전용) — h_l·l_out tap
// ---------------------------------------------------------------------------
bool ggml_moe_phasea_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;
    if (!ask) {
        // 데이터 사용 가능 시점. 무슨 일이 있어도 그래프 평가는 계속(true).
        if (!g_pa.snap_sealed || !g_pa.sealed || !t) return true;
        int32_t il = -1;
        const int kind = parse_tap_name(t->name, &il);
        if (!kind) return true;
        const int32_t target = il + 1;
        if (target < 1 || target >= g_pa.n_layer_exec) return true;
        const int32_t tmlpos = (target < (int32_t)g_pa.layer_to_mlpos.size())
                             ? g_pa.layer_to_mlpos[(size_t)target] : -1;
        if (tmlpos < 1 || tmlpos >= (int32_t)g_pa.pred.size()) return true;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != (int64_t)g_pa.n_embd || !ggml_is_contiguous(t) ||
            t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) {
            g_pa.r_tap_skipped++;
            return true;
        }
        const uint64_t t0 = qpc_now_raw();
        ggml_backend_tensor_get(t, g_pa.tap_buf.data(), 0, (size_t)g_pa.n_embd * sizeof(float));
        g_pa.r_tap_calls++;
        g_pa.r_tap_bytes += (uint64_t)g_pa.n_embd * sizeof(float);

        PredSlot & ps = g_pa.pred[(size_t)tmlpos];
        compute_topk(g_pa.tap_buf.data(), target, (kind == 1) ? ps.a0 : ps.b);
        if (kind == 1) { ps.valid_a0 = true; absent_of(tmlpos, ps.a0, &ps.absent_a0_8, &ps.absent_a0_16); }
        else           { ps.valid_b  = true; absent_of(tmlpos, ps.b,  &ps.absent_b_8,  &ps.absent_b_16);  }

        const uint64_t t1 = qpc_now_raw();
        if (g_pa.qpc_freq) {
            const uint64_t ns = (uint64_t)((double)(t1 - t0) * 1e9 / (double)g_pa.qpc_freq);
            g_pa.r_pred_calls++;
            g_pa.r_pred_ns_total += ns;
            if (ns > g_pa.r_pred_ns_max) g_pa.r_pred_ns_max = ns;
        }
        return true;
    }

    // ask 단계 — 필요한 노드에만 true(그만큼 split이 잘게 쪼개져 타이밍이 변한다).
    if (g_pa.mode != GGML_MOE_PHASEA_ACCURACY || !g_pa.snap_sealed || !g_pa.sealed) return false;
    if (!g_pa.in_graph || g_pa.cur_phase != 0) return false;   // decode graph 만 tap
    if (!t) return false;
    int32_t il = -1;
    if (!parse_tap_name(t->name, &il)) return false;
    const int32_t target = il + 1;
    return target >= 1 && target < g_pa.n_layer_exec;
}

// ---------------------------------------------------------------------------
// acquire / release 훅
// ---------------------------------------------------------------------------
void ggml_moe_phasea_acquire_enter(int32_t mlpos, int32_t part, int32_t n_ids,
                                   const int32_t * expert_ids, int32_t is_prefill,
                                   uint64_t t_enter_qpc, int32_t * out_is_canonical) {
    if (out_is_canonical) *out_is_canonical = 0;
    if (!pa_on() || !g_pa.sealed) return;
    if (mlpos < 0 || mlpos >= g_pa.n_moe_layers || n_ids <= 0 || !expert_ids) return;

    const uint8_t phase = g_pa.in_graph ? g_pa.cur_phase : (uint8_t)(is_prefill ? 1 : 0);
    // ★E — 소유권 판정(bare boolean 폐기). graph_begin 을 받지 못한 그래프(capability prelude 등)
    //   에서도 "이 graph_seq 가 이 층을 이미 소유했는가"만 보므로 경계 잔존이 성립하지 않는다.
    const bool canonical = g_pa.layer_seen_gs[(size_t)mlpos] != g_pa.cur_graph_seq;
    if (canonical) g_pa.layer_seen_gs[(size_t)mlpos] = g_pa.cur_graph_seq;
    if (out_is_canonical) *out_is_canonical = canonical ? 1 : 0;

    CurAcquire & c = g_pa.cur;
    c.open      = true;
    c.canonical = canonical;
    c.mlpos     = mlpos;
    c.part      = part;
    c.n_ids     = n_ids;
    c.phase     = phase;
    c.graph_seq = g_pa.cur_graph_seq;   // ★E — 소유 그래프 포획
    c.d_qpc     = t_enter_qpc;
    c.s_qpc     = 0;
    c.c_qpc     = 0;
    c.q_busy    = 0;
    c.q_epochs  = 0;
    c.q_last_n  = 0;
    c.q_last_t  = 0;
    c.n_hit     = 0;
    c.actual.clear();
    c.miss.clear();

    if (g_pa.run_active) {
        g_pa.r_acquire_calls++;
        if (canonical) g_pa.r_acquire_canonical++;
    }

    PaLedger r{};
    r.kind   = LED_ACQ_CALL;
    r.flags  = (uint8_t)((canonical ? 0x02 : 0x00) | (phase ? 0x04 : 0x00));
    r.expert = PA_NA_U16;
    r.layer  = (uint16_t)g_pa.moe_layers[(size_t)mlpos];
    r.part   = (uint16_t)(part < 0 ? PA_NA_U16 : part);
    r.slot   = -1;
    r.c32    = g_pa.cur_graph_seq;
    r.t_qpc  = t_enter_qpc;
    r.val    = ((uint64_t)g_pa.request_index << 32) | (uint64_t)(uint32_t)n_ids;
    led_push(r);
}

void ggml_moe_phasea_acquire_evict(int32_t slot, int64_t evicted_record_idx, int32_t new_expert) {
    if (!pa_on() || !g_pa.sealed || !g_pa.cur.open) return;
    PaLedger r{};
    r.kind   = LED_EVICT;
    r.flags  = (uint8_t)(g_pa.cur.phase ? 0x04 : 0x00);
    r.expert = (uint16_t)(new_expert < 0 ? PA_NA_U16 : new_expert);
    r.layer  = (uint16_t)g_pa.moe_layers[(size_t)g_pa.cur.mlpos];
    r.part   = (uint16_t)(g_pa.cur.part < 0 ? PA_NA_U16 : g_pa.cur.part);
    r.slot   = slot;
    r.c32    = g_pa.cur_graph_seq;
    r.t_qpc  = qpc_now_raw();
    r.val    = (uint64_t)evicted_record_idx;
    led_push(r);
}

void ggml_moe_phasea_acquire_expert(int32_t expert, int32_t is_hit, int32_t slot, uint64_t logical_tick) {
    if (!pa_on() || !g_pa.sealed || !g_pa.cur.open) return;
    CurAcquire & c = g_pa.cur;

    PaLedger r{};
    r.kind   = LED_ACQ_EXPERT;
    r.flags  = (uint8_t)((is_hit ? 0x01 : 0x00) | (c.phase ? 0x04 : 0x00));
    r.expert = (uint16_t)expert;
    r.layer  = (uint16_t)g_pa.moe_layers[(size_t)c.mlpos];
    r.part   = (uint16_t)(c.part < 0 ? PA_NA_U16 : c.part);
    r.slot   = slot;
    r.c32    = g_pa.cur_graph_seq;
    r.t_qpc  = c.d_qpc;
    r.val    = logical_tick;
    led_push(r);

    if (is_hit) c.n_hit++;
    if (c.canonical) {
        c.actual.push_back((uint16_t)expert);
        if (!is_hit) c.miss.push_back((uint16_t)expert);
    }
}

void ggml_moe_phasea_io_qd(int32_t n_outstanding, uint64_t t_qpc) {
    if (!pa_on() || !g_pa.sealed || !g_pa.cur.open) return;
    CurAcquire & c = g_pa.cur;
    if (c.q_last_n > 0 && c.q_last_t != 0 && t_qpc > c.q_last_t) c.q_busy += t_qpc - c.q_last_t;
    if (c.q_last_n == 0 && n_outstanding > 0) c.q_epochs++;
    c.q_last_n = n_outstanding;
    c.q_last_t = t_qpc;
}

void ggml_moe_phasea_io_op(int64_t record_idx, uint64_t file_offset, uint32_t bytes,
                           int32_t slot, uint32_t batch_order,
                           uint64_t submit_seq, uint64_t submit_qpc,
                           uint64_t retire_seq, uint64_t retire_qpc,
                           int32_t was_pending) {
    if (!pa_on() || !g_pa.sealed || !g_pa.cur.open) return;
    CurAcquire & c = g_pa.cur;
    if (c.s_qpc == 0 || submit_qpc < c.s_qpc) c.s_qpc = submit_qpc;
    if (retire_qpc > c.c_qpc) c.c_qpc = retire_qpc;

    PaOp o{};
    o.record_id   = (uint64_t)record_idx;
    o.file_offset = file_offset;
    o.bytes       = bytes;
    o.expert      = (uint16_t)(g_pa.n_expert > 0 ? (record_idx % g_pa.n_expert) : 0);
    o.layer       = (uint16_t)g_pa.moe_layers[(size_t)c.mlpos];
    o.graph_seq   = g_pa.cur_graph_seq;
    o.batch_order = batch_order;
    o.submit_seq  = submit_seq;
    o.submit_qpc  = submit_qpc;
    o.retire_seq  = retire_seq;
    o.retire_qpc  = retire_qpc;
    o.slot        = slot;
    o.flags       = (uint8_t)((was_pending ? 0x01 : 0x00) | (c.phase ? 0x02 : 0x00));
    o.part        = (uint8_t)(c.part < 0 ? 0xFF : c.part);
    if (buf_append(g_pa.buf_ops, &o, sizeof(o), &g_pa.n_ops) && g_pa.run_active) g_pa.r_op++;
}

void ggml_moe_phasea_acquire_exit(int32_t n_miss, uint64_t miss_bytes, uint64_t t_exit_qpc) {
    if (!pa_on() || !g_pa.sealed || !g_pa.cur.open) return;
    CurAcquire & c = g_pa.cur;

    // ── A3 층별 miss 분포(실행층 전수 · 양 모드 공통) ──
    if (g_pa.run_active && n_miss > 0) {
        g_pa.a3_miss_count[(size_t)c.mlpos] += (uint64_t)n_miss;
        g_pa.a3_miss_bytes[(size_t)c.mlpos] += miss_bytes;
    }

    if (c.canonical) {
        PaEvent e{};
        e.graph_seq     = g_pa.cur_graph_seq;
        e.layer         = (uint16_t)g_pa.moe_layers[(size_t)c.mlpos];
        e.flags         = (uint8_t)((c.phase ? 0x01 : 0x00) | (n_miss > 0 ? 0x02 : 0x00));
        e.part          = (uint8_t)(c.part < 0 ? 0xFF : c.part);
        e.d_qpc         = c.d_qpc;
        e.r_qpc         = t_exit_qpc;
        e.s_qpc         = c.s_qpc;
        e.c_qpc         = c.c_qpc;
        e.q_busy_qpc    = c.q_busy;
        e.q_epochs      = c.q_epochs;
        e.n_ids         = (uint32_t)c.n_ids;
        e.n_miss        = (uint32_t)n_miss;
        e.n_hit         = (uint32_t)c.n_hit;
        e.miss_bytes    = miss_bytes;
        e.request_index = g_pa.request_index;
        if (buf_append(g_pa.buf_events, &e, sizeof(e), &g_pa.n_events) && g_pa.run_active) g_pa.r_ev++;
        emit_join();
    }
    c.open = false;
}

void ggml_moe_phasea_release_enter(int32_t mlpos, int32_t part, int32_t n_ids, uint64_t t_qpc) {
    if (!pa_on() || !g_pa.sealed) return;
    if (mlpos < 0 || mlpos >= g_pa.n_moe_layers || n_ids <= 0) return;
    if (g_pa.run_active) g_pa.r_release_calls++;
    g_pa.rel_mlpos = mlpos;
    g_pa.rel_part  = part;
    g_pa.rel_t     = t_qpc;

    PaLedger r{};
    r.kind   = LED_REL_CALL;
    r.flags  = (uint8_t)((g_pa.in_graph && g_pa.cur_phase) ? 0x04 : 0x00);
    r.expert = PA_NA_U16;
    r.layer  = (uint16_t)g_pa.moe_layers[(size_t)mlpos];
    r.part   = (uint16_t)(part < 0 ? PA_NA_U16 : part);
    r.slot   = -1;
    r.c32    = g_pa.cur_graph_seq;
    r.t_qpc  = t_qpc;
    r.val    = ((uint64_t)g_pa.request_index << 32) | (uint64_t)(uint32_t)n_ids;
    led_push(r);
}

void ggml_moe_phasea_release_expert(int32_t expert, int32_t slot, int32_t lease_before, int32_t lease_after) {
    if (!pa_on() || !g_pa.sealed) return;
    if (g_pa.rel_mlpos < 0 || g_pa.rel_mlpos >= g_pa.n_moe_layers) return;
    PaLedger r{};
    r.kind   = LED_REL_EXPERT;
    r.flags  = (uint8_t)((g_pa.in_graph && g_pa.cur_phase) ? 0x04 : 0x00);
    r.expert = (uint16_t)expert;
    r.layer  = (uint16_t)g_pa.moe_layers[(size_t)g_pa.rel_mlpos];
    r.part   = (uint16_t)(g_pa.rel_part < 0 ? PA_NA_U16 : g_pa.rel_part);
    r.slot   = slot;
    r.c32    = g_pa.cur_graph_seq;
    r.t_qpc  = g_pa.rel_t;              // ★소속 REL_CALL 과 동일 시각(SPEC_PHASEA_WIRE §5)
    r.val    = ((uint64_t)(uint32_t)lease_before << 32) | (uint64_t)(uint32_t)lease_after;
    led_push(r);
}

// ---------------------------------------------------------------------------
// flush — ggml_moe_direct_finalize()가 락 밖에서 1회 호출(멱등)
// ---------------------------------------------------------------------------
void ggml_moe_phasea_flush(void) {
    if (!pa_on() || g_pa.flushed) return;
    g_pa.flushed = true;

    uint64_t dg_ev = 0, dg_op = 0, dg_led = 0, dg_set = 0;
    int wr_ev = 0, wr_op = 0, wr_led = 0, wr_set = 0;

    if (!g_pa.out_prefix.empty()) {
        const std::string & p = g_pa.out_prefix;
        wr_ev  = write_sidecar(p + "_events.pa", 1, sizeof(PaEvent),  g_pa.n_events, g_pa.buf_events, &dg_ev)  ? 1 : 0;
        wr_op  = write_sidecar(p + "_ops.pa",    2, sizeof(PaOp),     g_pa.n_ops,    g_pa.buf_ops,    &dg_op)  ? 1 : 0;
        wr_led = write_sidecar(p + "_ledger.pa", 3, sizeof(PaLedger), g_pa.n_ledger, g_pa.buf_ledger, &dg_led) ? 1 : 0;
        wr_set = write_sidecar(p + "_sets.pa",   4, sizeof(PaSet),    g_pa.n_sets,   g_pa.buf_sets,   &dg_set) ? 1 : 0;
    }

    const json sc = {
        {"events", json{{"records", g_pa.n_events}, {"written", wr_ev},  {"digest", to_hex64(dg_ev)}}},
        {"ops",    json{{"records", g_pa.n_ops},    {"written", wr_op},  {"digest", to_hex64(dg_op)}}},
        {"ledger", json{{"records", g_pa.n_ledger}, {"written", wr_led}, {"digest", to_hex64(dg_led)}}},
        {"sets",   json{{"records", g_pa.n_sets},   {"written", wr_set}, {"digest", to_hex64(dg_set)}}},
        {"complete",        g_pa.complete ? 1 : 0},
        {"dropped_records", g_pa.dropped},
        {"buffer_bytes",    buf_total_bytes()},
        {"snapshot_bytes",  ggml_moe_phasea_snapshot_bytes()},
        {"out_prefix",      g_pa.out_prefix},
    };

    for (auto & r : g_pa.run_records) {
        r["sidecar"] = sc;
        const std::string body = r.dump();
        ggml_moe_direct_metrics_write_record("phase_a_run", body.c_str());
    }

    pa_log("flush done - mode=%s events=%llu ops=%llu ledger=%llu sets=%llu buffer=%.2f MiB "
           "snapshot=%.2f MiB complete=%d dropped=%llu requests=%u",
           ggml_moe_phasea_mode_name(),
           (unsigned long long)g_pa.n_events, (unsigned long long)g_pa.n_ops,
           (unsigned long long)g_pa.n_ledger, (unsigned long long)g_pa.n_sets,
           (double)buf_total_bytes() / (1024.0 * 1024.0),
           (double)ggml_moe_phasea_snapshot_bytes() / (1024.0 * 1024.0),
           g_pa.complete ? 1 : 0, (unsigned long long)g_pa.dropped, g_pa.request_next);
}

// ---------------------------------------------------------------------------
// 테스트 전용
// ---------------------------------------------------------------------------
void ggml_moe_phasea_reset_for_test(void) {
    g_pa = PaState{};
}

void ggml_moe_phasea_test_counts(uint64_t * out_events, uint64_t * out_ops,
                                 uint64_t * out_ledger, uint64_t * out_sets,
                                 uint64_t * out_dropped, int32_t * out_complete) {
    if (out_events)   *out_events   = g_pa.n_events;
    if (out_ops)      *out_ops      = g_pa.n_ops;
    if (out_ledger)   *out_ledger   = g_pa.n_ledger;
    if (out_sets)     *out_sets     = g_pa.n_sets;
    if (out_dropped)  *out_dropped  = g_pa.dropped;
    if (out_complete) *out_complete = g_pa.complete ? 1 : 0;
}

} // extern "C"

#else // !_WIN32

extern "C" {

int32_t  ggml_moe_phasea_mode(void)    { return GGML_MOE_PHASEA_OFF; }
bool     ggml_moe_phasea_enabled(void) { return false; }
const char * ggml_moe_phasea_mode_name(void)  { return "off"; }
uint64_t ggml_moe_phasea_boot_id(void) { return 0; }
void ggml_moe_phasea_note_outputs(int32_t) {}
void ggml_moe_phasea_identity(struct ggml_moe_phasea_ident * o) { if (o) { o->valid=0; o->request_index=0; o->graph_seq=0; o->decode_ordinal=0; } }
const char * ggml_moe_phasea_out_prefix(void) { return ""; }
uint32_t ggml_moe_phasea_buf_mb(void)  { return 0; }
bool     ggml_moe_phasea_eval_callback_installed(void) { return false; }
uint64_t ggml_moe_phasea_now_qpc(void) { return 0; }
void ggml_moe_phasea_note_seal(int32_t, int32_t, int32_t, int32_t, const int32_t *, const uint64_t *) {}
void ggml_moe_phasea_run_begin(void) {}
void ggml_moe_phasea_run_end(uint32_t, uint32_t) {}
void ggml_moe_phasea_graph_begin(uint8_t, uint32_t) {}
void ggml_moe_phasea_graph_end(int32_t, int32_t) {}
int  ggml_moe_phasea_snapshot_configure(int32_t, int32_t, int32_t, int32_t, float) { return 0; }
bool ggml_moe_phasea_snapshot_needed(void) { return false; }
int  ggml_moe_phasea_snapshot_layer(int32_t, const struct ggml_tensor *, const struct ggml_tensor *) { return 0; }
int  ggml_moe_phasea_snapshot_seal(void) { return 0; }
uint64_t ggml_moe_phasea_snapshot_bytes(void) { return 0; }
bool ggml_moe_phasea_eval_callback(struct ggml_tensor *, bool ask, void *) { return !ask; }
void ggml_moe_phasea_note_eval_callback_installed(void) {}
void ggml_moe_phasea_acquire_enter(int32_t, int32_t, int32_t, const int32_t *, int32_t, uint64_t, int32_t * o) { if (o) *o = 0; }
void ggml_moe_phasea_acquire_evict(int32_t, int64_t, int32_t) {}
void ggml_moe_phasea_acquire_expert(int32_t, int32_t, int32_t, uint64_t) {}
void ggml_moe_phasea_io_qd(int32_t, uint64_t) {}
void ggml_moe_phasea_io_op(int64_t, uint64_t, uint32_t, int32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, int32_t) {}
void ggml_moe_phasea_acquire_exit(int32_t, uint64_t, uint64_t) {}
void ggml_moe_phasea_release_enter(int32_t, int32_t, int32_t, uint64_t) {}
void ggml_moe_phasea_release_expert(int32_t, int32_t, int32_t, int32_t) {}
void ggml_moe_phasea_flush(void) {}
void ggml_moe_phasea_reset_for_test(void) {}
void ggml_moe_phasea_test_counts(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d,
                                 uint64_t * e, int32_t * f) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0;
}

} // extern "C"

#endif // _WIN32
