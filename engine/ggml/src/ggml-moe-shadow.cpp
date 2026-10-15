// MoE 섀도 드래프트 관측 모듈 본체 — 사양: bench/techdev/decode_hunt/SPEC_SHADOW_DRAFT.md ★v1.0
//
// 이 파일이 소유하는 것: env 게이트 2축(실행·계측) · kind5/6/7/8 wire 버퍼 · 사이드카 기록 ·
// 스텝/층/스팬/원가 레코드 조립 · 층 타임라인 커서 · CPU 시계 조회.
// 이 파일이 소유하지 않는 것: C3 캐시 view(구현 = ggml-moe-direct.cpp — 캐시 내부 상태 소유자) ·
// 섀도 드라이버(src/llama-context.cpp) · 마스크-재정규화 forward(src/llama-graph.cpp).
//
// ★동시성: 섀도 드라이버는 토큰 확정 직후 **동기** 실행이므로(SPEC D6) 이 모듈은 단일 스레드
// 진입을 전제한다. 방어적으로 자체 mutex 를 두되 락 안에서 다른 모듈을 호출하지 않는다.
//
// Windows 전용 — 비Windows 는 파일 하단 스텁.

#include "ggml-moe-shadow.h"
#include "ggml-moe-phasea.h"
#include "ggml.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>          // std::bad_alloc — buf_append 의 catch + r3 M1 음성 창구
#include <string>
#include <vector>

namespace {

// ===========================================================================
// §1. wire 구조체 (SPEC §2 — 리틀엔디언·고정폭)
// ===========================================================================

#pragma pack(push, 1)

// 공통 128B 헤더 — Phase A 관례(PaHeader)와 **바이트 동형**이어야 한다(같은 판독기가 읽는다).
struct ShHeader {
    char     magic[8];          // "MOEPA1\0\0"                      @0
    uint32_t layout_version;    // 1                                 @8
    uint32_t file_kind;         // 5=shadow layer · 6=shadow step    @12
    uint32_t record_bytes;      // 96 / 64                           @16
    uint32_t phasea_mode;       // Phase A 모드 echo(진단)            @20
    uint64_t record_count;      //                                   @24
    uint64_t payload_digest;    // FNV-1a 64                         @32
    uint64_t qpc_frequency;     //                                   @40
    uint64_t t0_qpc;            //                                   @48
    uint32_t pid;               //                                   @56
    uint32_t complete;          // 1=완전 0=버퍼 상한 초과 → fail-close @60
    uint32_t n_expert;          //                                   @64
    uint32_t n_moe_layers;      //                                   @68
    uint32_t n_slots;           //                                   @72
    uint32_t n_expert_used;     //                                   @76
    uint64_t boot_id;           // ★조인 키(pa_sets.pa 와 동일값)     @80
    uint64_t total_requests;    //                                   @88
    uint64_t dropped_records;   //                                   @96
    uint8_t  reserved[24];      //                                   @104..127
};

// kind=5 · 96B — 섀도 스텝 j 의 층 1개
struct ShLayer {
    uint32_t request_index;          // @0
    uint32_t origin_graph_seq;       // @4   진단용
    uint32_t origin_decode_ordinal;  // @8
    uint32_t target_decode_ordinal;  // @12  = origin_decode_ordinal + j
    uint16_t layer;                  // @16  실제 layer id
    uint8_t  j;                      // @18  1..D
    uint8_t  flags;                  // @19  b0=all_dropped · 잔여 0
    uint8_t  n_pred;                 // @20  <= 16
    uint8_t  n_present;              // @21  = popcount(present_mask)
    uint16_t present_mask;           // @22  pred 칸 1:1 · bits[n_pred..15]=0
    uint16_t pred[GGML_MOE_SHADOW_PRED_CAP];  // @24..55  미사용 = 0xFFFF
    uint16_t gate_mass_present_f16;  // @56  진단 전용(재정규화 계산에 사용 금지)
    uint16_t reserved0;              // @58  = 0
    uint32_t record_stride_bytes;    // @60  B[e] — byte 가중 recall 분모/분자 가중
    uint64_t route_ready_qpc;        // @64
    uint64_t layer_begin_qpc;        // @72
    uint64_t layer_end_qpc;          // @80
    uint8_t  reserved1[8];           // @88..95 = 0
};

// kind=6 · 64B — 섀도 스텝(j) 당 1행
struct ShStep {
    uint32_t request_index;          // @0
    uint32_t origin_decode_ordinal;  // @4
    uint32_t target_decode_ordinal;  // @8
    uint8_t  j;                      // @12
    uint8_t  flags;                  // @13  잔여 0
    uint16_t reserved0;              // @14  = 0
    uint64_t step_begin_qpc;         // @16
    uint64_t step_end_qpc;           // @24
    uint32_t resident_records;       // @32
    uint32_t dropped_records;        // @36
    uint64_t resident_gemv_bytes;    // @40
    uint32_t origin_graph_seq;       // @48  진단용
    uint32_t reserved1;              // @52  = 0
    uint64_t reserved2;              // @56  = 0
};

// kind=7 · 64B — 계측 스팬 1건 (★r1 차단 6 / SPEC §4 G-SD3 ⑥①)
//   kind5/6 ABI 는 동결이므로 clone/checkpoint · dispatcher drain · device sync · writer ·
//   target span 은 이 **별도 축**에 싣는다. target_decode_ordinal 은 호출 인자가 아니라
//   (origin, j) 에서 파생한다 — kind5/6 의 조인 계약과 어긋날 수 없게 하기 위해서다.
struct ShSpan {
    uint32_t request_index;          // @0   프로세스 수준 스팬 = 0xFFFFFFFF
    uint32_t origin_decode_ordinal;  // @4   〃
    uint32_t target_decode_ordinal;  // @8   = origin + j (특례 없음 · origin==NA 일 때만 0xFFFFFFFF)
                                     //      ★캐노니컬 pairing 키 = 이 필드다. TARGET(j=0) 은
                                     //      target == origin 이므로 같은 키로 떨어진다.
    uint8_t  j;                      // @12  0 = 스텝과 무관
    uint8_t  span_kind;              // @13  enum ggml_moe_shadow_span_kind
    uint8_t  flags;                  // @14  b0=BAD_ORDER · b1=UNKNOWN_KIND · 잔여 0
    uint8_t  reserved0;              // @15  = 0
    uint64_t t_begin_qpc;            // @16
    uint64_t t_end_qpc;              // @24
    uint64_t aux_u64;                // @32  종류별 보조량(WRITER = payload bytes) · 없으면 0
    uint32_t origin_graph_seq;       // @40  진단용(열린 스텝이 있으면 그 값, 없으면 0xFFFFFFFF)
    uint32_t seq_no;                 // @44  프로세스 내 발행 순번(동일 QPC 동률 정렬·dup 판별)
    uint64_t reserved1;              // @48  = 0
    uint64_t reserved2;              // @56  = 0
};

// kind=8 · 96B — G-SD3 ⑦ CPU/backend 원가 1건 (★r2 M2 신설 축)
//   kind=7 은 64B 로 동결·static_assert 되어 있으므로 필드를 넓히지 않고 별도 kind 를 만든다.
//   같은 (request_index, target_decode_ordinal, span_kind, j) 로 kind=7 의 같은 행과 1:1 조인된다.
struct ShCost {
    uint32_t request_index;          // @0   프로세스 수준 = 0xFFFFFFFF
    uint32_t origin_decode_ordinal;  // @4
    uint32_t target_decode_ordinal;  // @8   = origin + j (kind=7 과 동일 파생식)
    uint8_t  j;                      // @12
    uint8_t  span_kind;              // @13  enum ggml_moe_shadow_span_kind
    uint8_t  flags;                  // @14  COST_FLAG_* (kind=7 flags 와 값 도메인이 다르다)
    uint8_t  reserved0;              // @15  = 0
    uint64_t t_begin_qpc;            // @16  ★kind=7 과 같은 값(축 간 대조용 · 자기완결)
    uint64_t t_end_qpc;              // @24
    uint64_t thread_cycles;          // @32  QueryThreadCycleTime  delta(호출 스레드)
    uint64_t proc_cycles;            // @40  QueryProcessCycleTime delta(전 스레드)
    uint64_t thread_cpu_100ns;       // @48  GetThreadTimes(kernel+user)  delta
    uint64_t proc_cpu_100ns;         // @56  GetProcessTimes(kernel+user) delta
    uint64_t backend_wait_qpc;       // @64  Σ 호스트가 sched_synchronize 에 블록된 시간
    uint64_t device_interval_qpc;    // @72  last_sync_end − first_submit (host 관측 창)
    uint32_t n_submits;              // @80
    uint32_t n_syncs;                // @84
    uint32_t seq_no;                 // @88  프로세스 내 발행 순번
    uint32_t reserved1;              // @92  = 0
};

#pragma pack(pop)

static_assert(sizeof(ShHeader) == 128, "ShHeader must be exactly 128 bytes (ABI contract)");
static_assert(sizeof(ShLayer)  == GGML_MOE_SHADOW_LAYER_BYTES,
              "ShLayer must be exactly 96 bytes (SPEC_SHADOW_DRAFT section 2 ABI contract)");
static_assert(sizeof(ShStep)   == GGML_MOE_SHADOW_STEP_BYTES,
              "ShStep must be exactly 64 bytes (SPEC_SHADOW_DRAFT section 2 ABI contract)");
static_assert(sizeof(ShSpan)   == GGML_MOE_SHADOW_SPAN_BYTES,
              "ShSpan must be exactly 64 bytes (kind=7 telemetry axis ABI contract)");
static_assert(sizeof(ShCost)   == GGML_MOE_SHADOW_COST_BYTES,
              "ShCost must be exactly 96 bytes (kind=8 cost axis ABI contract)");

// 오프셋 계약을 컴파일 타임에 잠근다(사양 문면의 @off 표와 1:1).
static_assert(offsetof(ShLayer, layer)                 == 16, "ShLayer.layer @16");
static_assert(offsetof(ShLayer, present_mask)          == 22, "ShLayer.present_mask @22");
static_assert(offsetof(ShLayer, pred)                  == 24, "ShLayer.pred @24");
static_assert(offsetof(ShLayer, gate_mass_present_f16) == 56, "ShLayer.gate_mass_present_f16 @56");
static_assert(offsetof(ShLayer, record_stride_bytes)   == 60, "ShLayer.record_stride_bytes @60");
static_assert(offsetof(ShLayer, route_ready_qpc)       == 64, "ShLayer.route_ready_qpc @64");
static_assert(offsetof(ShLayer, layer_end_qpc)         == 80, "ShLayer.layer_end_qpc @80");
static_assert(offsetof(ShLayer, reserved1)             == 88, "ShLayer.reserved1 @88");
static_assert(offsetof(ShStep,  step_begin_qpc)        == 16, "ShStep.step_begin_qpc @16");
static_assert(offsetof(ShStep,  resident_records)      == 32, "ShStep.resident_records @32");
static_assert(offsetof(ShStep,  resident_gemv_bytes)   == 40, "ShStep.resident_gemv_bytes @40");
static_assert(offsetof(ShStep,  origin_graph_seq)      == 48, "ShStep.origin_graph_seq @48");
static_assert(offsetof(ShStep,  reserved2)             == 56, "ShStep.reserved2 @56");

// kind=7 오프셋 계약(신설 축 — kind5/6 과 같은 강도로 잠근다).
static_assert(offsetof(ShSpan,  request_index)         ==  0, "ShSpan.request_index @0");
static_assert(offsetof(ShSpan,  origin_decode_ordinal) ==  4, "ShSpan.origin_decode_ordinal @4");
static_assert(offsetof(ShSpan,  target_decode_ordinal) ==  8, "ShSpan.target_decode_ordinal @8");
static_assert(offsetof(ShSpan,  j)                     == 12, "ShSpan.j @12");
static_assert(offsetof(ShSpan,  span_kind)             == 13, "ShSpan.span_kind @13");
static_assert(offsetof(ShSpan,  flags)                 == 14, "ShSpan.flags @14");
static_assert(offsetof(ShSpan,  reserved0)             == 15, "ShSpan.reserved0 @15");
static_assert(offsetof(ShSpan,  t_begin_qpc)           == 16, "ShSpan.t_begin_qpc @16");
static_assert(offsetof(ShSpan,  t_end_qpc)             == 24, "ShSpan.t_end_qpc @24");
static_assert(offsetof(ShSpan,  aux_u64)               == 32, "ShSpan.aux_u64 @32");
static_assert(offsetof(ShSpan,  origin_graph_seq)      == 40, "ShSpan.origin_graph_seq @40");
static_assert(offsetof(ShSpan,  seq_no)                == 44, "ShSpan.seq_no @44");
static_assert(offsetof(ShSpan,  reserved1)             == 48, "ShSpan.reserved1 @48");
static_assert(offsetof(ShSpan,  reserved2)             == 56, "ShSpan.reserved2 @56");

// kind=8 오프셋 계약(신설 축 — 기존 축과 같은 강도로 잠근다).
static_assert(offsetof(ShCost,  request_index)         ==  0, "ShCost.request_index @0");
static_assert(offsetof(ShCost,  origin_decode_ordinal) ==  4, "ShCost.origin_decode_ordinal @4");
static_assert(offsetof(ShCost,  target_decode_ordinal) ==  8, "ShCost.target_decode_ordinal @8");
static_assert(offsetof(ShCost,  j)                     == 12, "ShCost.j @12");
static_assert(offsetof(ShCost,  span_kind)             == 13, "ShCost.span_kind @13");
static_assert(offsetof(ShCost,  flags)                 == 14, "ShCost.flags @14");
static_assert(offsetof(ShCost,  reserved0)             == 15, "ShCost.reserved0 @15");
static_assert(offsetof(ShCost,  t_begin_qpc)           == 16, "ShCost.t_begin_qpc @16");
static_assert(offsetof(ShCost,  t_end_qpc)             == 24, "ShCost.t_end_qpc @24");
static_assert(offsetof(ShCost,  thread_cycles)         == 32, "ShCost.thread_cycles @32");
static_assert(offsetof(ShCost,  proc_cycles)           == 40, "ShCost.proc_cycles @40");
static_assert(offsetof(ShCost,  thread_cpu_100ns)      == 48, "ShCost.thread_cpu_100ns @48");
static_assert(offsetof(ShCost,  proc_cpu_100ns)        == 56, "ShCost.proc_cpu_100ns @56");
static_assert(offsetof(ShCost,  backend_wait_qpc)      == 64, "ShCost.backend_wait_qpc @64");
static_assert(offsetof(ShCost,  device_interval_qpc)   == 72, "ShCost.device_interval_qpc @72");
static_assert(offsetof(ShCost,  n_submits)             == 80, "ShCost.n_submits @80");
static_assert(offsetof(ShCost,  n_syncs)               == 84, "ShCost.n_syncs @84");
static_assert(offsetof(ShCost,  seq_no)                == 88, "ShCost.seq_no @88");
static_assert(offsetof(ShCost,  reserved1)             == 92, "ShCost.reserved1 @92");

constexpr uint16_t SH_NA_U16 = 0xFFFFu;
constexpr uint32_t SH_NA_U32 = GGML_MOE_SHADOW_ID_NA;
constexpr uint64_t FNV_SEED  = 0xcbf29ce484222325ull;

// ===========================================================================
// §2. 유틸
// ===========================================================================

inline uint64_t fnv1a64(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

void sh_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[moe-shadow] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// ===========================================================================
// §3. 전역 상태
// ===========================================================================

struct ShState {
    // ── 게이트 ──
    bool    resolved = false;
    bool    on       = false;      // MOE_SHADOW_DRAFT — 섀도 **실행**
    bool    bench    = false;      // ★MOE_SHADOW_BENCH — **계측만**(실행과 독립 · r2 M2)
    int32_t depth    = 0;
    int32_t every    = 0;
    bool    forced   = false;      // 테스트 강제 게이트(env 재해석 금지)
    std::string out_prefix;
    uint32_t buf_mb  = 512;
    uint64_t buf_cap_bytes = 0;
    uint64_t boot_id = 0;

    // ── 버퍼 ──
    std::vector<uint8_t> buf_layer, buf_step, buf_span, buf_cost;
    uint64_t n_layer = 0, n_step = 0, n_span = 0, n_cost = 0;
    uint32_t span_seq = 0;          // 스팬 발행 순번(wire 의 seq_no)
    uint32_t cost_seq = 0;          // 원가 발행 순번(kind=8 의 seq_no)
    uint64_t n_target_spans = 0;    // ★TARGET 스팬 수(flush 경고용 — 0 이면 ⑨ 분모가 없다)
    uint64_t dropped = 0;
    bool     complete        = true;
    bool     buf_overflowed  = false;   // ★latch — 부분 기록 방지
    bool     flushed         = false;

    // ── 진행 중인 섀도 스텝 1건 ──
    bool     step_open   = false;
    uint32_t s_request   = 0xFFFFFFFFu;
    uint32_t s_origin_gs = 0xFFFFFFFFu;
    uint32_t s_origin_ord = 0;
    uint32_t s_target_ord = 0;
    int32_t  s_j          = 0;
    uint64_t s_begin_qpc  = 0;
    // 층 레코드가 누적하는 스텝 비용 3항(kind=6 의 원천).
    uint32_t s_resident   = 0;
    uint32_t s_dropped    = 0;
    uint64_t s_gemv_bytes = 0;
    // ★층 타임라인 커서(r1 차단 6) — "다음 층이 시작된 시각". step_begin 에서 step_begin_qpc 로
    //   초기화하고 층 레코드마다 layer_end_qpc 로 단조 전진한다. 스텝 밖 = 0.
    uint64_t s_layer_window_qpc = 0;

    // ── ★런 오류 latch (r1 차단 2·3) ──
    uint64_t run_errors = 0;
    bool     run_error_logged = false;

    // ── 음성 대조 주입 ──
    int32_t  fault_bits = 0;

    std::mutex mutex;
};

ShState g_sh;

// ---------------------------------------------------------------------------
// env 파싱: "depth<D>,every<N>" — 키 순서 무관·공백 허용·대소문자 무시.
// 파싱 실패 = 완전 OFF + 경고 1회(정적 기본값으로 조용히 켜지는 것을 금한다).
// ---------------------------------------------------------------------------
bool parse_spec(const std::string & raw, int32_t & out_depth, int32_t & out_every) {
    std::string s;
    s.reserve(raw.size());
    for (char c : raw) {
        if (!std::isspace((unsigned char) c)) s.push_back((char) std::tolower((unsigned char) c));
    }
    if (s.empty()) return false;

    int32_t d = -1, e = -1;
    size_t  i = 0;
    while (i <= s.size()) {
        const size_t comma = s.find(',', i);
        const std::string tok = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        if (!tok.empty()) {
            const char * key = nullptr;
            int32_t *    dst = nullptr;
            if (tok.compare(0, 5, "depth") == 0)      { key = "depth"; dst = &d; }
            else if (tok.compare(0, 5, "every") == 0) { key = "every"; dst = &e; }
            if (!key || !dst) return false;
            const std::string num = tok.substr(5);
            if (num.empty()) return false;
            for (char c : num) if (c < '0' || c > '9') return false;
            errno = 0;
            const long v = std::strtol(num.c_str(), nullptr, 10);
            if (errno != 0 || v <= 0 || v > 4096) return false;
            if (*dst >= 0) return false;   // 중복 키 = 오류
            *dst = (int32_t) v;
        }
        if (comma == std::string::npos) break;
        i = comma + 1;
    }
    if (d <= 0 || e <= 0) return false;
    out_depth = d;
    out_every = e;
    return true;
}

// ---------------------------------------------------------------------------
// ★계측 게이트 파싱(MOE_SHADOW_BENCH) — 실행 게이트와 별개 축(r2 M2).
//   엄격 파싱: 아는 값만 받고 나머지는 OFF + 경고 1회(오타로 조용히 켜지거나 꺼지는 것 금지).
// ---------------------------------------------------------------------------
bool parse_bench(const std::string & raw, bool & out_on) {
    std::string s;
    s.reserve(raw.size());
    for (char c : raw) {
        if (!std::isspace((unsigned char) c)) s.push_back((char) std::tolower((unsigned char) c));
    }
    if (s.empty())                                        { out_on = false; return true; }
    if (s == "1" || s == "on"  || s == "true"  || s == "yes") { out_on = true;  return true; }
    if (s == "0" || s == "off" || s == "false" || s == "no")  { out_on = false; return true; }
    out_on = false;
    return false;
}

void resolve_env_once() {
    if (g_sh.resolved) return;
    g_sh.resolved = true;

    // ── ① 섀도 **실행** 게이트 ──
    const std::string raw = env_str("MOE_SHADOW_DRAFT");
    if (!raw.empty()) {
        int32_t d = 0, e = 0;
        if (parse_spec(raw, d, e)) {
            g_sh.on    = true;
            g_sh.depth = d;
            g_sh.every = e;
        } else {
            sh_log("MOE_SHADOW_DRAFT='%s' unparsable (expected 'depth<D>,every<N>') - shadow stays OFF",
                   raw.c_str());
        }
    }

    // ── ①' ★benchmark **계측** 게이트(실행과 독립 — SPEC §4 G-SD3 ⑨ 의 OFF 팔 분모) ──
    //   실행이 켜지면 계측은 상위집합으로 자동 포함된다(ON 팔이 OFF 팔보다 축이 적을 수 없다).
    {
        const std::string braw = env_str("MOE_SHADOW_BENCH");
        bool b = false;
        if (!parse_bench(braw, b)) {
            sh_log("MOE_SHADOW_BENCH='%s' unrecognised (expected 1/0/on/off/true/false/yes/no) - "
                   "benchmark telemetry stays OFF", braw.c_str());
        }
        g_sh.bench = b || g_sh.on;
    }

    if (!g_sh.on && !g_sh.bench) {
        // ★제품 완전 OFF — 어떤 상태도 할당하지 않는다(섀도 흔적 0 계약).
        return;
    }

    // ── ★r4 R4' — 음성 대조 주입 창구의 **env 표면**(`MOE_SHADOW_FAULT=0x10` 등) ──
    //   phase-0 의 NO_SYNC 음성은 소비자가 `llama_context::synchronize()` 라 모델 적재 서버
    //   프로세스에서만 도달한다. 그 프로세스에는 테스트 API(`ggml_moe_shadow_test_fault`)를
    //   부를 지점이 없으므로, 무장 수단이 없으면 phase-0 명령 자체가 성립하지 않는다
    //   (교차 r4 [차단] "실행 가능한 전용 명령").
    //   ★게이트 안쪽에서만 읽는다 — 제품 완전 OFF(위 early return)에서는 파싱조차 하지 않으므로
    //   "섀도 흔적 0" 계약은 그대로다. 미설정/파싱 실패 = 0(주입 없음).
    {
        const std::string fraw = env_str("MOE_SHADOW_FAULT");
        if (!fraw.empty()) {
            char * end = nullptr;
            const long v = std::strtol(fraw.c_str(), &end, 0);   // 0x.. 16진 · 10진 모두 허용
            if (end && *end == '\0' && v > 0 && v <= 0x7F) {
                g_sh.fault_bits = (int32_t) v;
                sh_log("MOE_SHADOW_FAULT=0x%02x armed - THIS IS A NEGATIVE-CONTROL RUN, "
                       "its observations are intentionally broken and must not be used as data",
                       (unsigned) v);
            } else {
                sh_log("MOE_SHADOW_FAULT='%s' invalid (expected 0x01..0x7f) - no fault injected",
                       fraw.c_str());
            }
        }
    }

    // 버퍼 상한(MiB). 미설정/부정형 = 기본 512.
    const std::string mb = env_str("MOE_SHADOW_BUF_MB");
    if (!mb.empty()) {
        const long v = std::strtol(mb.c_str(), nullptr, 10);
        if (v > 0 && v <= 65536) g_sh.buf_mb = (uint32_t) v;
        else sh_log("MOE_SHADOW_BUF_MB='%s' invalid - keeping default %u", mb.c_str(), g_sh.buf_mb);
    }
    g_sh.buf_cap_bytes = (uint64_t) g_sh.buf_mb * 1024ull * 1024ull;

    // 사이드카 접두사는 Phase A 와 공유한다(같은 run 디렉터리·같은 조인 대상).
    g_sh.out_prefix = env_str("MOE_DIRECT_PHASEA_OUT");
    if (g_sh.out_prefix.empty()) {
        // ★complete 는 "기록한 레코드에 결손이 있는가"만 뜻한다(버퍼 상한 초과 = 0).
        //   출력 접두사 미설정은 **쓸 곳이 없다**는 별개 사실이므로 여기서 complete 를 내리지
        //   않는다 — 내리면 명시 접두사로 정상 기록한 파일까지 헤더가 "결손"이라 거짓말한다.
        //   실 런에서 접두사가 없으면 애초에 파일이 생기지 않아 판독기가 그 자체로 fail-close 한다.
        sh_log("MOE_DIRECT_PHASEA_OUT unset - shadow sidecars will not be written at finalize");
    }

    // 조인 키. Phase A 가 켜져 있으면 그 boot_id 를 그대로 쓴다.
    g_sh.boot_id = ggml_moe_phasea_boot_id();
    if (g_sh.boot_id == 0) {
        // Phase A OFF — 자체 생성(파일 내부 정합만 성립·pa_sets 조인 불가).
        const uint64_t t0  = ggml_moe_phasea_now_qpc();
        const uint32_t pid = (uint32_t) GetCurrentProcessId();
        uint64_t h = FNV_SEED;
        h = fnv1a64(h, &t0,  sizeof(t0));
        h = fnv1a64(h, &pid, sizeof(pid));
        g_sh.boot_id = h;
        sh_log("Phase A is OFF - shadow uses a self-generated boot_id; join against pa_sets.pa "
               "will fail-close in the reader (this is intended, not a silent pass)");
    }
    sh_log("armed: exec=%s bench=%s depth=%d every=%d buf_mb=%u out='%s'",
           g_sh.on ? "on" : "off", g_sh.bench ? "on" : "off", g_sh.depth, g_sh.every, g_sh.buf_mb,
           g_sh.out_prefix.c_str());
}

inline bool sh_on()    { resolve_env_once(); return g_sh.on; }
inline bool sh_bench() { resolve_env_once(); return g_sh.bench; }
// 실행 OR 계측 — 스팬/원가 축이 살아 있는가.
inline bool sh_any()   { resolve_env_once(); return g_sh.on || g_sh.bench; }

// ★버퍼 상한은 4축 합계로 잰다(kind=7/8 도 같은 예산 안에서 산다 — 새 축이 상한을 우회하면
//   MOE_SHADOW_BUF_MB 계약이 거짓말이 된다).
uint64_t buf_total_bytes() {
    return (uint64_t) g_sh.buf_layer.size() + (uint64_t) g_sh.buf_step.size() +
           (uint64_t) g_sh.buf_span.size()  + (uint64_t) g_sh.buf_cost.size();
}

bool buf_append(std::vector<uint8_t> & v, const void * rec, size_t n, uint64_t * counter) {
    if (g_sh.buf_overflowed || (g_sh.buf_cap_bytes && buf_total_bytes() + n > g_sh.buf_cap_bytes)) {
        if (!g_sh.buf_overflowed) {
            sh_log("shadow buffer cap (%u MiB) exceeded - recording stopped, complete=0", g_sh.buf_mb);
        }
        g_sh.buf_overflowed = true;   // ★latch — 이후 더 작은 레코드도 전부 거부
        g_sh.complete = false;
        g_sh.dropped++;
        return false;
    }
    // ★r1 차단 3 수리 — insert 는 재할당에서 bad_alloc 을 던질 수 있다. 여기서 예외가
    //   빠져나가면 backend 콜백(custom op) 한복판이라 view 가 열린 채 스택이 풀린다.
    //   던지는 대신 overflow 와 동일하게 처분한다(latch + complete=0).
    const uint8_t * p = (const uint8_t *) rec;
    try {
        // ★r3 M1 음성 대조 창구(test-only) — **실제 std::bad_alloc 을 던져** 아래 catch 를 그대로
        //   태운다. 처분을 흉내 내는 별도 분기를 만들지 않는 이유: 그러면 음성이 프로덕션 코드가
        //   아니라 그 흉내 분기를 검사하게 된다. 프로덕션 경로는 fault_bits==0 이라 무영향이다.
        if (g_sh.fault_bits & GGML_MOE_SHADOW_FAULT_WRITER_OOM) {
            throw std::bad_alloc();
        }
        v.insert(v.end(), p, p + n);
    } catch (const std::bad_alloc &) {
        g_sh.buf_overflowed = true;
        g_sh.complete = false;
        g_sh.dropped++;
        sh_log("out of memory while appending a shadow record - recording stopped, complete=0");
        return false;
    }
    (*counter)++;
    return true;
}

// ---------------------------------------------------------------------------
// ★스팬 조립(kind=7) — 락을 **이미 쥔 상태**에서만 부른다.
//   flush() 가 락 안에서 writer 스팬을 남겨야 하므로 공개 API 와 분리했다(std::mutex 는
//   재귀 락이 아니다 — 공개 API 가 그대로 재진입하면 교착이다).
// ---------------------------------------------------------------------------
void span_note_locked(int32_t span_kind, uint32_t request_index, uint32_t origin_decode_ordinal,
                      int32_t j, uint64_t t_begin_qpc, uint64_t t_end_qpc, uint64_t aux_u64) {
    ShSpan r{};

    uint8_t flags = 0;
    int32_t k = span_kind;
    if (k <= (int32_t) GGML_MOE_SHADOW_SPAN_NONE || k >= (int32_t) GGML_MOE_SHADOW_SPAN_KIND_COUNT) {
        // ABI 불일치(구 호출자·오타)를 조용히 정상 스팬으로 접지 않는다 — 판독기가 fail-close
        // 할 수 있게 종류는 NONE 으로 접고 플래그로 사실을 남긴다.
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_SPAN_FLAG_UNKNOWN_KIND);
        k = (int32_t) GGML_MOE_SHADOW_SPAN_NONE;
    }
    if (t_end_qpc < t_begin_qpc) {
        // 시계 혼용/짝 어긋남. 값은 그대로 싣고(원자료 보존) 사실만 표시한다.
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_SPAN_FLAG_BAD_ORDER);
    }

    const int32_t jj = j < 0 ? 0 : (j > 255 ? 255 : j);

    r.request_index         = request_index;
    r.origin_decode_ordinal = origin_decode_ordinal;
    // target = origin + j. ★r2 M2 — 다만 **j=0 이면 TARGET 만** target 을 싣는다.
    //   TARGET 은 판독기가 `(request_index, target_decode_ordinal)` 로 ON/OFF 짝을 맞추므로
    //   j=0 이어도 키가 있어야 하고(= target==origin), 그 밖의 j=0 스팬(TRIGGER·⑥ 비용
    //   구성요소)은 판독기 계약이 NA 를 요구한다. ordinal 자체는 origin 필드가 그대로 들고
    //   있으므로 정보 손실은 없다. 이 규칙은 소비자(shadow_reader.py) 의 하드 계약과 1:1이며,
    //   어긋나면 그 런 전체가 fail-close 한다 — 엔진이 맞추는 쪽이 안전한 수렴이다.
    r.target_decode_ordinal = (origin_decode_ordinal == SH_NA_U32) ? SH_NA_U32
                            : ((jj == 0 && k != (int32_t) GGML_MOE_SHADOW_SPAN_TARGET)
                                   ? SH_NA_U32
                                   : (origin_decode_ordinal + (uint32_t) jj));
    r.j                     = (uint8_t) jj;
    r.span_kind             = (uint8_t) k;
    r.flags                 = flags;
    r.reserved0             = 0;
    r.t_begin_qpc           = t_begin_qpc;
    r.t_end_qpc             = t_end_qpc;
    // ★r2 M2 — aux 는 **WRITER 전용**(판독기 하드 계약: "WRITER 외에는 0"). 다른 종류가
    //   보조량을 실어 보내면 wire 에서 떨군다(조용한 계약 위반보다 명시적 0 이 낫다).
    r.aux_u64               = (k == (int32_t) GGML_MOE_SHADOW_SPAN_WRITER) ? aux_u64 : 0;
    r.origin_graph_seq      = g_sh.step_open ? g_sh.s_origin_gs : SH_NA_U32;
    r.seq_no                = g_sh.span_seq++;
    r.reserved1             = 0;
    r.reserved2             = 0;

    if (buf_append(g_sh.buf_span, &r, sizeof(r), &g_sh.n_span) &&
        r.span_kind == (uint8_t) GGML_MOE_SHADOW_SPAN_TARGET) {
        g_sh.n_target_spans++;
    }
}

// ---------------------------------------------------------------------------
// ★원가 조립(kind=8) — 락을 **이미 쥔 상태**에서만 부른다(span_note_locked 와 동일 규약).
//   차분은 여기서 계산한다: 호출측이 begin/end 스냅샷만 넘기면 되므로 뺄셈 실수가 원천 봉쇄되고,
//   역행(감소)한 카운터는 0 으로 접고 CPU_UNAVAIL 을 세운다(u64 랩어라운드로 천문학적 값이
//   실리는 것을 금한다 — 판독기가 그 값을 "측정됨"으로 오독하면 ⑦ 판정이 통째로 거짓이 된다).
// ---------------------------------------------------------------------------
uint64_t cost_delta(uint64_t a, uint64_t b, uint8_t & flags) {
    if (b < a) {
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_COST_FLAG_CPU_UNAVAIL);
        return 0;
    }
    return b - a;
}

void cost_note_locked(int32_t span_kind, uint32_t request_index, uint32_t origin_decode_ordinal,
                      int32_t j, uint64_t t_begin_qpc, uint64_t t_end_qpc,
                      const struct ggml_moe_shadow_cpu_times   * cpu_begin,
                      const struct ggml_moe_shadow_cpu_times   * cpu_end,
                      const struct ggml_moe_shadow_backend_acc * backend) {
    ShCost r{};

    uint8_t flags = 0;
    int32_t k = span_kind;
    if (k <= (int32_t) GGML_MOE_SHADOW_SPAN_NONE || k >= (int32_t) GGML_MOE_SHADOW_SPAN_KIND_COUNT) {
        k = (int32_t) GGML_MOE_SHADOW_SPAN_NONE;   // kind=7 과 동일 처분(조용한 정상화 금지)
    }
    if (t_end_qpc < t_begin_qpc) {
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_COST_FLAG_BAD_ORDER);
    }
    // ★섀도 **실행**이 꺼진 팔에서 수집한 행임을 wire 에 못박는다 — 판독기가 ON/OFF 팔을
    //   파일 밖 메타(디렉터리명 등)에 의존하지 않고 레코드 자체로 확인할 수 있어야 한다.
    if (!g_sh.on) {
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_COST_FLAG_BENCH_ONLY);
    }
    if (!cpu_begin || !cpu_end || !cpu_begin->ok || !cpu_end->ok) {
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_COST_FLAG_CPU_UNAVAIL);
    } else if (cpu_begin->thread_id != cpu_end->thread_id) {
        // 사이클이 다른 스레드에서 닫혔다 — thread_* 차분은 의미가 없다(proc_* 는 유효).
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_COST_FLAG_CROSS_THREAD);
    }

    const int32_t jj = j < 0 ? 0 : (j > 255 ? 255 : j);

    r.request_index         = request_index;
    r.origin_decode_ordinal = origin_decode_ordinal;
    // ★kind=7 과 **같은 파생식**을 쓴다(두 축이 바이트 수준에서 어긋나면 1:1 조인이 깨진다).
    //   조인 키는 `(request_index, origin_decode_ordinal, span_kind, j)` 다 — j=0 비-TARGET
    //   스팬은 target 이 NA 라 target 을 키로 쓰면 같은 요청의 여러 ordinal 이 충돌한다.
    r.target_decode_ordinal = (origin_decode_ordinal == SH_NA_U32) ? SH_NA_U32
                            : ((jj == 0 && k != (int32_t) GGML_MOE_SHADOW_SPAN_TARGET)
                                   ? SH_NA_U32
                                   : (origin_decode_ordinal + (uint32_t) jj));
    r.j                     = (uint8_t) jj;
    r.span_kind             = (uint8_t) k;
    r.reserved0             = 0;
    r.t_begin_qpc           = t_begin_qpc;
    r.t_end_qpc             = t_end_qpc;

    if (cpu_begin && cpu_end && cpu_begin->ok && cpu_end->ok) {
        r.thread_cycles    = cost_delta(cpu_begin->thread_cycles,    cpu_end->thread_cycles,    flags);
        r.proc_cycles      = cost_delta(cpu_begin->proc_cycles,      cpu_end->proc_cycles,      flags);
        r.thread_cpu_100ns = cost_delta(cpu_begin->thread_cpu_100ns, cpu_end->thread_cpu_100ns, flags);
        r.proc_cpu_100ns   = cost_delta(cpu_begin->proc_cpu_100ns,   cpu_end->proc_cpu_100ns,   flags);
    }
    // CROSS_THREAD 면 thread_* 는 남기되 플래그로 무효를 알린다(원자료 보존 · 판독기가 배제).

    if (backend) {
        r.backend_wait_qpc = backend->wait_qpc;
        r.n_submits        = backend->n_submits;
        r.n_syncs          = backend->n_syncs;
        if (backend->first_submit_qpc != 0 && backend->last_sync_end_qpc > backend->first_submit_qpc) {
            r.device_interval_qpc = backend->last_sync_end_qpc - backend->first_submit_qpc;
        }
    }
    if (r.n_submits == 0 || r.n_syncs == 0 || r.device_interval_qpc == 0) {
        // ★"0" 을 측정값으로 읽히게 두지 않는다 — 미관측과 진짜 0 은 다른 사실이다.
        flags = (uint8_t) (flags | GGML_MOE_SHADOW_COST_FLAG_NO_BACKEND);
    }

    r.flags     = flags;
    r.seq_no    = g_sh.cost_seq++;
    r.reserved1 = 0;

    buf_append(g_sh.buf_cost, &r, sizeof(r), &g_sh.n_cost);
}

// ===========================================================================
// §4. 사이드카 기록 (Phase A write_sidecar 와 동일 규약: CREATE_NEW·128B 헤더·32MiB 청크)
// ===========================================================================

bool write_sidecar(const std::string & path, uint32_t kind, uint32_t record_bytes,
                   uint64_t record_count, const std::vector<uint8_t> & payload) {
    ShHeader h{};
    std::memcpy(h.magic, "MOEPA1\0\0", 8);
    h.layout_version  = 1;
    h.file_kind       = kind;
    h.record_bytes    = record_bytes;
    h.phasea_mode     = (uint32_t) ggml_moe_phasea_mode();
    h.record_count    = record_count;
    h.payload_digest  = fnv1a64(FNV_SEED, payload.data(), payload.size());
    h.qpc_frequency   = 0;
    h.t0_qpc          = 0;
    {
        LARGE_INTEGER f;
        if (QueryPerformanceFrequency(&f)) h.qpc_frequency = (uint64_t) f.QuadPart;
    }
    h.pid             = (uint32_t) GetCurrentProcessId();
    h.complete        = g_sh.complete ? 1u : 0u;
    // n_expert / n_moe_layers / n_slots / n_expert_used 는 이 모듈이 소유하지 않는다(0 = 미기재).
    // 판독기는 같은 run 의 pa_sets.pa 헤더에서 읽는다(boot_id 로 묶인 동일 프로세스).
    h.boot_id         = g_sh.boot_id;
    h.total_requests  = 0;
    h.dropped_records = g_sh.dropped;

    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int) path.size(), nullptr, 0);
    std::wstring w((size_t) (n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int) path.size(), &w[0], n);

    HANDLE f = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        sh_log("cannot create sidecar (CREATE_NEW) path=%s err=%lu", path.c_str(),
               (unsigned long) GetLastError());
        return false;
    }
    DWORD wr = 0;
    bool ok = WriteFile(f, &h, sizeof(h), &wr, nullptr) != 0 && wr == sizeof(h);
    if (ok && !payload.empty()) {
        size_t off = 0;
        while (off < payload.size()) {
            const DWORD chunk = (DWORD) std::min<size_t>(payload.size() - off, (size_t) 32 * 1024 * 1024);
            if (!WriteFile(f, payload.data() + off, chunk, &wr, nullptr) || wr != chunk) { ok = false; break; }
            off += chunk;
        }
    }
    if (ok) ok = FlushFileBuffers(f) != 0;
    CloseHandle(f);
    if (!ok) sh_log("cannot write sidecar path=%s err=%lu", path.c_str(), (unsigned long) GetLastError());
    return ok;
}

} // namespace

// ===========================================================================
// §5. 공개 API — 게이트
// ===========================================================================

bool ggml_moe_shadow_enabled(void) { return sh_on(); }

// ★계측 게이트 — 섀도 실행과 독립(r2 M2). 실행이 켜져 있으면 자동으로 true 다(상위집합).
bool ggml_moe_shadow_bench_enabled(void) { return sh_bench(); }
bool ggml_moe_shadow_any_gate(void)      { return sh_any(); }

int32_t ggml_moe_shadow_depth(void) { return sh_on() ? g_sh.depth : 0; }
int32_t ggml_moe_shadow_every(void) { return sh_on() ? g_sh.every : 0; }

const char * ggml_moe_shadow_out_prefix(void) { resolve_env_once(); return g_sh.out_prefix.c_str(); }

uint64_t ggml_moe_shadow_boot_id(void) { resolve_env_once(); return g_sh.boot_id; }

// ===========================================================================
// §6. 공개 API — 스텝/층 레코드
// ===========================================================================

void ggml_moe_shadow_step_begin(uint32_t request_index, uint32_t origin_graph_seq,
                                uint32_t origin_decode_ordinal, uint32_t target_decode_ordinal,
                                int32_t j) {
    if (!sh_on()) return;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (g_sh.step_open) {
        // 짝이 안 맞은 채 새 스텝이 열리는 것은 드라이버 결함이다 — 조용히 덮지 않는다.
        sh_log("step_begin while a step is still open (j=%d) - previous step record is dropped", g_sh.s_j);
        g_sh.dropped++;
        g_sh.complete = false;
    }
    g_sh.step_open    = true;
    g_sh.s_request    = request_index;
    g_sh.s_origin_gs  = origin_graph_seq;
    g_sh.s_origin_ord = origin_decode_ordinal;
    g_sh.s_target_ord = target_decode_ordinal;
    g_sh.s_j          = j;
    g_sh.s_begin_qpc  = ggml_moe_phasea_now_qpc();
    g_sh.s_resident   = 0;
    g_sh.s_dropped    = 0;
    g_sh.s_gemv_bytes = 0;
    // ★층 타임라인 커서 초기화 — 스텝의 **첫 층**은 step_begin 부터 시작한 것으로 본다.
    g_sh.s_layer_window_qpc = g_sh.s_begin_qpc;
}

void ggml_moe_shadow_step_end(void) {
    if (!sh_on()) return;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (!g_sh.step_open) return;
    const uint32_t resident_records     = g_sh.s_resident;
    const uint32_t dropped_records      = g_sh.s_dropped;
    const uint64_t resident_gemv_bytes  = g_sh.s_gemv_bytes;

    ShStep r{};
    r.request_index         = g_sh.s_request;
    r.origin_decode_ordinal = g_sh.s_origin_ord;
    r.target_decode_ordinal = g_sh.s_target_ord;
    r.j                     = (uint8_t) (g_sh.s_j < 0 ? 0 : (g_sh.s_j > 255 ? 255 : g_sh.s_j));
    r.flags                 = 0;
    r.reserved0             = 0;
    r.step_begin_qpc        = g_sh.s_begin_qpc;
    r.step_end_qpc          = ggml_moe_phasea_now_qpc();
    r.resident_records      = resident_records;
    r.dropped_records       = dropped_records;
    r.resident_gemv_bytes   = resident_gemv_bytes;
    r.origin_graph_seq      = g_sh.s_origin_gs;
    r.reserved1             = 0;
    r.reserved2             = 0;

    buf_append(g_sh.buf_step, &r, sizeof(r), &g_sh.n_step);
    g_sh.step_open = false;
    g_sh.s_layer_window_qpc = 0;   // 스텝 밖 — 커서 무효
}

// ★층 타임라인 창(r1 차단 6) — 다음 층 레코드가 layer_begin_qpc 로 실어야 할 값.
uint64_t ggml_moe_shadow_layer_window_qpc(void) {
    if (!sh_on()) return 0;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    return g_sh.step_open ? g_sh.s_layer_window_qpc : 0;
}

void ggml_moe_shadow_layer_record(int32_t layer, const int32_t * pred, int32_t n_pred,
                                  uint16_t present_mask, float gate_mass_present,
                                  int32_t all_dropped, uint32_t record_stride_bytes,
                                  uint64_t route_ready_qpc, uint64_t layer_begin_qpc,
                                  uint64_t layer_end_qpc) {
    if (!sh_on()) return;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (!g_sh.step_open) return;   // 스텝 밖 호출은 무시(드라이버 계약 위반 — 기록 오염 금지)

    const int32_t np = n_pred < 0 ? 0 : (n_pred > GGML_MOE_SHADOW_PRED_CAP ? GGML_MOE_SHADOW_PRED_CAP : n_pred);

    ShLayer r{};
    r.request_index         = g_sh.s_request;
    r.origin_graph_seq      = g_sh.s_origin_gs;
    r.origin_decode_ordinal = g_sh.s_origin_ord;
    r.target_decode_ordinal = g_sh.s_target_ord;
    r.layer                 = (uint16_t) (layer < 0 ? SH_NA_U16 : (layer > 0xFFFE ? 0xFFFE : layer));
    r.j                     = (uint8_t) (g_sh.s_j < 0 ? 0 : (g_sh.s_j > 255 ? 255 : g_sh.s_j));
    r.flags                 = (uint8_t) (all_dropped ? 0x01 : 0x00);
    r.n_pred                = (uint8_t) np;

    // present_mask 계약: pred 칸 1:1 · bits[n_pred..15] = 0.
    const uint16_t mask = (uint16_t) (np >= 16 ? present_mask
                                               : (uint16_t) (present_mask & (uint16_t) ((1u << np) - 1u)));
    r.present_mask = mask;
    uint8_t pc = 0;
    for (int b = 0; b < 16; b++) if (mask & (1u << b)) pc++;
    r.n_present = pc;

    for (int i = 0; i < GGML_MOE_SHADOW_PRED_CAP; i++) {
        r.pred[i] = (i < np && pred && pred[i] >= 0) ? (uint16_t) pred[i] : SH_NA_U16;
    }

    // mass 는 F32 로 계산된 값을 IEEE binary16 으로 직렬화한다(진단 전용).
    r.gate_mass_present_f16 = (uint16_t) ggml_fp32_to_fp16(gate_mass_present);
    r.reserved0             = 0;
    r.record_stride_bytes   = record_stride_bytes;
    r.route_ready_qpc       = route_ready_qpc;
    r.layer_begin_qpc       = layer_begin_qpc;
    r.layer_end_qpc         = layer_end_qpc;
    std::memset(r.reserved1, 0, sizeof(r.reserved1));

    // 스텝 비용 3항 누적(kind=6 원천) — 층 기록과 스텝 합계가 구조적으로 일치한다.
    g_sh.s_resident   += (uint32_t) pc;
    g_sh.s_dropped    += (uint32_t) (np - (int32_t) pc);
    g_sh.s_gemv_bytes += (uint64_t) pc * (uint64_t) record_stride_bytes;

    // ★층 타임라인 커서 전진(r1 차단 6) — 다음 층의 창은 이 층이 끝난 시각부터다.
    //   단조만 허용한다(역행 값은 무시 — 창이 음수 폭이 되는 것을 원천 차단).
    if (layer_end_qpc > g_sh.s_layer_window_qpc) {
        g_sh.s_layer_window_qpc = layer_end_qpc;
    }

    buf_append(g_sh.buf_layer, &r, sizeof(r), &g_sh.n_layer);
}

// ===========================================================================
// §6'. 공개 API — 스팬(kind=7 · G-SD3 축)
// ===========================================================================

uint64_t ggml_moe_shadow_now_qpc(void) { return ggml_moe_phasea_now_qpc(); }

void ggml_moe_shadow_span_note(int32_t span_kind, uint32_t request_index,
                               uint32_t origin_decode_ordinal, int32_t j,
                               uint64_t t_begin_qpc, uint64_t t_end_qpc, uint64_t aux_u64) {
    // ★r2 M2 — 실행 게이트가 아니라 **계측 게이트**로 연다. 그래야 짝런 OFF 팔(섀도 미실행)도
    //   TARGET 스팬을 낼 수 있고 ⑨ 의 분모가 생긴다. 둘 다 OFF 면 여전히 완전 no-op 이다.
    if (!sh_any()) return;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    span_note_locked(span_kind, request_index, origin_decode_ordinal, j,
                     t_begin_qpc, t_end_qpc, aux_u64);
}

// ===========================================================================
// §6''. 공개 API — 원가(kind=8 · G-SD3 ⑦ 축)
// ===========================================================================

void ggml_moe_shadow_cpu_now(struct ggml_moe_shadow_cpu_times * out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->thread_id = (uint64_t) GetCurrentThreadId();

    int filled = 0;
    ULONG64 cyc = 0;
    if (QueryThreadCycleTime(GetCurrentThread(), &cyc))  { out->thread_cycles = (uint64_t) cyc; filled++; }
    cyc = 0;
    if (QueryProcessCycleTime(GetCurrentProcess(), &cyc)) { out->proc_cycles  = (uint64_t) cyc; filled++; }

    FILETIME ct, et, kt, ut;
    if (GetThreadTimes(GetCurrentThread(), &ct, &et, &kt, &ut)) {
        const uint64_t k = ((uint64_t) kt.dwHighDateTime << 32) | (uint64_t) kt.dwLowDateTime;
        const uint64_t u = ((uint64_t) ut.dwHighDateTime << 32) | (uint64_t) ut.dwLowDateTime;
        out->thread_cpu_100ns = k + u;
        filled++;
    }
    if (GetProcessTimes(GetCurrentProcess(), &ct, &et, &kt, &ut)) {
        const uint64_t k = ((uint64_t) kt.dwHighDateTime << 32) | (uint64_t) kt.dwLowDateTime;
        const uint64_t u = ((uint64_t) ut.dwHighDateTime << 32) | (uint64_t) ut.dwLowDateTime;
        out->proc_cpu_100ns = k + u;
        filled++;
    }
    // 전건 성공일 때만 ok — 부분 성공을 "측정됨"으로 올리지 않는다(판독기 fail-close 유지).
    out->ok = (filled == 4) ? 1 : 0;
}

void ggml_moe_shadow_cost_note(int32_t span_kind, uint32_t request_index,
                               uint32_t origin_decode_ordinal, int32_t j,
                               uint64_t t_begin_qpc, uint64_t t_end_qpc,
                               const struct ggml_moe_shadow_cpu_times   * cpu_begin,
                               const struct ggml_moe_shadow_cpu_times   * cpu_end,
                               const struct ggml_moe_shadow_backend_acc * backend) {
    if (!sh_any()) return;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    cost_note_locked(span_kind, request_index, origin_decode_ordinal, j,
                     t_begin_qpc, t_end_qpc, cpu_begin, cpu_end, backend);
}

// ===========================================================================
// §7. flush
// ===========================================================================

// ★런 오류 latch — 게이트와 무관하게 삼키지 않는다(배선 결함을 숨기지 않기 위해).
void ggml_moe_shadow_run_error(const char * reason) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    g_sh.run_errors++;
    g_sh.complete = false;   // ★latch — 이 프로세스의 사이드카는 fail-close 대상이 된다
    if (!g_sh.run_error_logged) {
        g_sh.run_error_logged = true;
        sh_log("run error latched (sidecars become complete=0): %s", reason ? reason : "(no reason)");
    }
}

uint64_t ggml_moe_shadow_run_error_count(void) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    return g_sh.run_errors;
}

void ggml_moe_shadow_flush(void) {
    if (!sh_any()) return;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (g_sh.flushed) return;
    g_sh.flushed = true;

    if (g_sh.step_open) {
        sh_log("flush with an open step (j=%d) - its kind=6 row is missing, complete=0", g_sh.s_j);
        g_sh.complete = false;
        g_sh.step_open = false;
    }
    if (g_sh.out_prefix.empty()) {
        sh_log("no out prefix - %llu layer / %llu step / %llu span / %llu cost records discarded",
               (unsigned long long) g_sh.n_layer, (unsigned long long) g_sh.n_step,
               (unsigned long long) g_sh.n_span,  (unsigned long long) g_sh.n_cost);
        return;
    }
    // ★계측이 무장됐는데 TARGET 이 한 건도 없으면 ⑨ 의 분모(또는 분자)가 통째로 비었다는 뜻이다.
    //   가장 흔한 원인은 Phase A 미가동(= decode identity 가 영영 valid 가 되지 않음)이다.
    //   complete 는 내리지 않는다(기록한 레코드 자체엔 결손이 없다) — 대신 크게 경고하고
    //   판독기가 "축 미수집"으로 fail-close 하게 둔다.
    if (g_sh.n_target_spans == 0) {
        sh_log("WARNING: telemetry armed but 0 TARGET spans were recorded - the paired ON/OFF "
               "denominator (SPEC G-SD3 (9)) will be empty. Most likely Phase A is not running "
               "(MOE_DIRECT_PHASEA), so the decode identity never became valid.");
    }
    const std::string p = g_sh.out_prefix;

    // ★writer 비용도 G-SD3 ⑥의 계측 대상이다 — kind=5/6/8 기록 구간을 그 자리에서 스팬으로
    //   남기고, 그 스팬까지 담은 kind=7 을 **마지막에** 쓴다(스팬 파일 자신의 기록 비용은
    //   자기참조라 담을 수 없다 — 판독기는 그 한 건만 미계측으로 본다).
    //
    // ★r2 M2 — kind=5/6 은 **섀도가 실제로 실행된 팔에서만** 만든다. 짝런 OFF 팔에 0행짜리
    //   kind=5/6 을 깔면 "파일이 있다 = 섀도가 돌았다" 라는 읽기가 무너져, 팔 라벨링 오류가
    //   조용히 통과할 수 있다. OFF 팔의 축은 kind=7/8 뿐이다.
    bool wl = true, ws = true;
    if (g_sh.on) {
        const uint64_t w0 = ggml_moe_phasea_now_qpc();
        wl = write_sidecar(p + "_shadow.pa",       GGML_MOE_SHADOW_KIND_LAYER,
                           (uint32_t) sizeof(ShLayer), g_sh.n_layer, g_sh.buf_layer);
        const uint64_t w1 = ggml_moe_phasea_now_qpc();
        ws = write_sidecar(p + "_shadow_steps.pa", GGML_MOE_SHADOW_KIND_STEP,
                           (uint32_t) sizeof(ShStep),  g_sh.n_step,  g_sh.buf_step);
        const uint64_t w2 = ggml_moe_phasea_now_qpc();
        span_note_locked(GGML_MOE_SHADOW_SPAN_WRITER, SH_NA_U32, SH_NA_U32, 0, w0, w1,
                         (uint64_t) g_sh.buf_layer.size());
        span_note_locked(GGML_MOE_SHADOW_SPAN_WRITER, SH_NA_U32, SH_NA_U32, 0, w1, w2,
                         (uint64_t) g_sh.buf_step.size());
    }

    const uint64_t w3 = ggml_moe_phasea_now_qpc();
    const bool wc = write_sidecar(p + "_shadow_cpugpu.pa", GGML_MOE_SHADOW_KIND_COST,
                                  (uint32_t) sizeof(ShCost), g_sh.n_cost, g_sh.buf_cost);
    const uint64_t w4 = ggml_moe_phasea_now_qpc();
    span_note_locked(GGML_MOE_SHADOW_SPAN_WRITER, SH_NA_U32, SH_NA_U32, 0, w3, w4,
                     (uint64_t) g_sh.buf_cost.size());

    // 스팬이 0행이어도 **파일은 만든다** — "스팬이 없다"와 "이 축이 아예 없다"는 판독기에서
    // 서로 다른 판정(후자는 fail-close)이어야 하기 때문이다.
    const bool wp = write_sidecar(p + "_shadow_spans.pa", GGML_MOE_SHADOW_KIND_SPAN,
                                  (uint32_t) sizeof(ShSpan), g_sh.n_span, g_sh.buf_span);
    sh_log("flush(exec=%s bench=%s): layer_records=%llu written=%d · step_records=%llu written=%d · "
           "span_records=%llu written=%d (target=%llu) · cost_records=%llu written=%d · "
           "dropped=%llu complete=%d",
           g_sh.on ? "on" : "off", g_sh.bench ? "on" : "off",
           (unsigned long long) g_sh.n_layer, g_sh.on ? (wl ? 1 : 0) : -1,
           (unsigned long long) g_sh.n_step,  g_sh.on ? (ws ? 1 : 0) : -1,
           (unsigned long long) g_sh.n_span,  wp ? 1 : 0,
           (unsigned long long) g_sh.n_target_spans,
           (unsigned long long) g_sh.n_cost,  wc ? 1 : 0,
           (unsigned long long) g_sh.dropped, g_sh.complete ? 1 : 0);
}

// ===========================================================================
// §8. 테스트 전용 창구
// ===========================================================================

void ggml_moe_shadow_reset_for_test(void) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    g_sh.resolved = false;
    g_sh.on       = false;
    g_sh.bench    = false;
    g_sh.depth    = 0;
    g_sh.every    = 0;
    g_sh.forced   = false;
    g_sh.out_prefix.clear();
    g_sh.buf_mb   = 512;
    g_sh.buf_cap_bytes = 0;
    g_sh.boot_id  = 0;
    g_sh.buf_layer.clear();
    g_sh.buf_step.clear();
    g_sh.buf_span.clear();
    g_sh.buf_cost.clear();
    g_sh.buf_layer.shrink_to_fit();
    g_sh.buf_step.shrink_to_fit();
    g_sh.buf_span.shrink_to_fit();
    g_sh.buf_cost.shrink_to_fit();
    g_sh.n_layer = g_sh.n_step = g_sh.n_span = g_sh.n_cost = g_sh.dropped = 0;
    g_sh.n_target_spans = 0;
    g_sh.span_seq       = 0;
    g_sh.cost_seq       = 0;
    g_sh.complete       = true;
    g_sh.buf_overflowed = false;
    g_sh.flushed        = false;
    g_sh.step_open      = false;
    g_sh.s_layer_window_qpc = 0;
    g_sh.fault_bits     = 0;
    g_sh.run_errors     = 0;
    g_sh.run_error_logged = false;
}

namespace {
// 강제 게이트가 켜질 때 필요한 최소 상태(버퍼 상한·조인 키)를 지연 확보한다.
void sh_arm_state_locked() {
    if (g_sh.buf_cap_bytes == 0) g_sh.buf_cap_bytes = (uint64_t) g_sh.buf_mb * 1024ull * 1024ull;
    if (g_sh.boot_id == 0) {
        g_sh.boot_id = ggml_moe_phasea_boot_id();
        if (g_sh.boot_id == 0) {
            const uint64_t t0  = ggml_moe_phasea_now_qpc();
            const uint32_t pid = (uint32_t) GetCurrentProcessId();
            uint64_t h = FNV_SEED;
            h = fnv1a64(h, &t0,  sizeof(t0));
            h = fnv1a64(h, &pid, sizeof(pid));
            g_sh.boot_id = h;
        }
    }
}
} // namespace

void ggml_moe_shadow_test_force_gate(int32_t on, int32_t depth, int32_t every) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    g_sh.resolved = true;               // env 재해석 금지
    if (!on) {
        // 헤더 문면대로 **완전 OFF** 로 되돌린다(계측 축 포함). 짝런 OFF 팔을 재현하려면
        // 이 호출 뒤에 ggml_moe_shadow_test_force_bench(1) 을 명시적으로 부른다.
        g_sh.on = false; g_sh.bench = false; g_sh.depth = 0; g_sh.every = 0; g_sh.forced = false;
        return;
    }
    g_sh.on     = true;
    g_sh.bench  = true;                 // 실행 ON = 계측 상위집합(env 경로와 동일 규약)
    g_sh.forced = true;
    g_sh.depth  = depth > 0 ? depth : 8;
    g_sh.every  = every > 0 ? every : 8;
    sh_arm_state_locked();
}

// ★짝런 OFF 팔 재현용 — 섀도 실행은 그대로 두고 계측만 켠다/끈다.
void ggml_moe_shadow_test_force_bench(int32_t on) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    g_sh.resolved = true;               // env 재해석 금지
    if (!on) {
        // 실행이 켜져 있으면 계측은 상위집합이라 끌 수 없다(ON 팔이 OFF 팔보다 축이 적을 수 없다).
        g_sh.bench = g_sh.on;
        return;
    }
    g_sh.bench = true;
    sh_arm_state_locked();
}

void ggml_moe_shadow_test_counts(uint64_t * out_layers, uint64_t * out_steps,
                                 uint64_t * out_dropped, int32_t * out_complete) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (out_layers)   *out_layers   = g_sh.n_layer;
    if (out_steps)    *out_steps    = g_sh.n_step;
    if (out_dropped)  *out_dropped  = g_sh.dropped;
    if (out_complete) *out_complete = g_sh.complete ? 1 : 0;
}

int ggml_moe_shadow_test_layer_bytes(uint64_t idx, void * out, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (!out || bytes != sizeof(ShLayer) || idx >= g_sh.n_layer) return -1;
    std::memcpy(out, g_sh.buf_layer.data() + (size_t) idx * sizeof(ShLayer), sizeof(ShLayer));
    return 0;
}

int ggml_moe_shadow_test_step_bytes(uint64_t idx, void * out, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (!out || bytes != sizeof(ShStep) || idx >= g_sh.n_step) return -1;
    std::memcpy(out, g_sh.buf_step.data() + (size_t) idx * sizeof(ShStep), sizeof(ShStep));
    return 0;
}

int ggml_moe_shadow_test_span_bytes(uint64_t idx, void * out, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (!out || bytes != sizeof(ShSpan) || idx >= g_sh.n_span) return -1;
    std::memcpy(out, g_sh.buf_span.data() + (size_t) idx * sizeof(ShSpan), sizeof(ShSpan));
    return 0;
}

int ggml_moe_shadow_test_cost_bytes(uint64_t idx, void * out, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    if (!out || bytes != sizeof(ShCost) || idx >= g_sh.n_cost) return -1;
    std::memcpy(out, g_sh.buf_cost.data() + (size_t) idx * sizeof(ShCost), sizeof(ShCost));
    return 0;
}

uint64_t ggml_moe_shadow_test_span_count(void) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    return g_sh.n_span;
}

uint64_t ggml_moe_shadow_test_cost_count(void) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    return g_sh.n_cost;
}

int ggml_moe_shadow_test_write_to(const char * prefix) {
    if (!prefix || !*prefix) return -1;
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    const std::string p = prefix;
    // ★픽스처 경로는 WRITER 스팬을 만들지 않는다 — 만들면 "기록 → 버퍼 증가 → 재기록마다
    //   다른 파일"이 되어 실 emitter→reader 왕복의 재현성이 깨진다.
    //   ★flush() 와 달리 여기서는 4종을 **무조건** 쓴다(픽스처는 팔 라벨링 대상이 아니다).
    const bool wl = write_sidecar(p + "_shadow.pa",        GGML_MOE_SHADOW_KIND_LAYER,
                                  (uint32_t) sizeof(ShLayer), g_sh.n_layer, g_sh.buf_layer);
    const bool ws = write_sidecar(p + "_shadow_steps.pa",  GGML_MOE_SHADOW_KIND_STEP,
                                  (uint32_t) sizeof(ShStep),  g_sh.n_step,  g_sh.buf_step);
    const bool wp = write_sidecar(p + "_shadow_spans.pa",  GGML_MOE_SHADOW_KIND_SPAN,
                                  (uint32_t) sizeof(ShSpan),  g_sh.n_span,  g_sh.buf_span);
    const bool wc = write_sidecar(p + "_shadow_cpugpu.pa", GGML_MOE_SHADOW_KIND_COST,
                                  (uint32_t) sizeof(ShCost),  g_sh.n_cost,  g_sh.buf_cost);
    return (wl && ws && wp && wc) ? 0 : -2;
}

void ggml_moe_shadow_test_fault(int32_t bits) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    g_sh.fault_bits = bits;
}

int32_t ggml_moe_shadow_test_fault_bits(void) {
    std::lock_guard<std::mutex> lock(g_sh.mutex);
    return g_sh.fault_bits;
}

#else  // !_WIN32 — 비Windows 스텁

bool     ggml_moe_shadow_enabled(void) { return false; }
bool     ggml_moe_shadow_bench_enabled(void) { return false; }
bool     ggml_moe_shadow_any_gate(void) { return false; }
int32_t  ggml_moe_shadow_depth  (void) { return 0; }
int32_t  ggml_moe_shadow_every  (void) { return 0; }
const char * ggml_moe_shadow_out_prefix(void) { return ""; }
uint64_t ggml_moe_shadow_boot_id(void) { return 0; }

void ggml_moe_shadow_step_begin(uint32_t, uint32_t, uint32_t, uint32_t, int32_t) {}
void ggml_moe_shadow_step_end  (void) {}
void ggml_moe_shadow_layer_record(int32_t, const int32_t *, int32_t, uint16_t, float, int32_t,
                                  uint32_t, uint64_t, uint64_t, uint64_t) {}
uint64_t ggml_moe_shadow_layer_window_qpc(void) { return 0; }
uint64_t ggml_moe_shadow_now_qpc(void) { return 0; }
void ggml_moe_shadow_span_note(int32_t, uint32_t, uint32_t, int32_t, uint64_t, uint64_t,
                               uint64_t) {}
void ggml_moe_shadow_cpu_now(struct ggml_moe_shadow_cpu_times * out) {
    if (out) {
        out->thread_cycles = out->thread_cpu_100ns = 0;
        out->proc_cycles   = out->proc_cpu_100ns   = 0;
        out->thread_id     = 0;
        out->ok            = 0;
        out->reserved      = 0;
    }
}
void ggml_moe_shadow_cost_note(int32_t, uint32_t, uint32_t, int32_t, uint64_t, uint64_t,
                               const struct ggml_moe_shadow_cpu_times *,
                               const struct ggml_moe_shadow_cpu_times *,
                               const struct ggml_moe_shadow_backend_acc *) {}
void ggml_moe_shadow_flush(void) {}
void ggml_moe_shadow_run_error(const char *) {}
uint64_t ggml_moe_shadow_run_error_count(void) { return 0; }

void ggml_moe_shadow_reset_for_test(void) {}
void ggml_moe_shadow_test_force_gate(int32_t, int32_t, int32_t) {}
void ggml_moe_shadow_test_force_bench(int32_t) {}
void ggml_moe_shadow_test_counts(uint64_t * a, uint64_t * b, uint64_t * c, int32_t * d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
int  ggml_moe_shadow_test_layer_bytes(uint64_t, void *, uint64_t) { return -1; }
int  ggml_moe_shadow_test_step_bytes (uint64_t, void *, uint64_t) { return -1; }
int  ggml_moe_shadow_test_span_bytes (uint64_t, void *, uint64_t) { return -1; }
int  ggml_moe_shadow_test_cost_bytes (uint64_t, void *, uint64_t) { return -1; }
uint64_t ggml_moe_shadow_test_span_count(void) { return 0; }
uint64_t ggml_moe_shadow_test_cost_count(void) { return 0; }
int  ggml_moe_shadow_test_write_to(const char *) { return -1; }
void ggml_moe_shadow_test_fault(int32_t) {}
int32_t ggml_moe_shadow_test_fault_bits(void) { return 0; }

#endif // _WIN32
