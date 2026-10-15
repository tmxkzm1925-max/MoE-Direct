// MoE 프리페치 Phase B — 층-선행 프리페치 신호/관측 모듈 본체(B2).
//
// 사양: bench/moe-direct/PHASEB_DESIGN.md(★FROZEN v3) · wire: bench/moe-direct/SPEC_PHASEB_WIRE.md
//       (v4 ★FROZEN). 설계와 충돌하면 PHASEB_DESIGN이 이긴다.
//
// 이 모듈이 소유하는 것(§2.9 B2 확장 자원 전부):
//   capture_q(48 슬롯 · 각 슬롯이 key + capture_width F32 사본을 **자기 것으로** 소유 — 단일 버퍼 금지
//             · capture_width = 계열 adapter descriptor 의 n_embd · guard ≤8192)
//   intent_q(48 PrefetchIntent · overflow=조용한 drop 금지 → request disable + 계수 + run 무효 latch)
//   predictor worker 1(전용 스레드 · g_mod.mutex/q_mutex 어느 쪽도 보유하지 않는다)
//   router(l=1..n-1) + post-norm(l=0..n-1) host 스냅샷
//   C3 카운터 · 5종 digest 중 intent digest · a0_equiv 사이드카 · 요청별 phase_b_run 누적
//
// ★관측 모드(MOE_DIRECT_PHASEB)는 **관측만** 가른다 — trace on/off가 intent 생성·admission·
//   I/O 순서를 바꾸면 C4′ 위반이다(§9 게이트 12). 그래서 카운터·digest·intent 경로는 모드와
//   무관하게 항상 돌고, **레코드 버퍼링만** 모드로 가린다.
//
// Windows 전용 — 비Windows는 파일 하단 스텁.

#include "ggml-moe-phaseb.h"
#include "ggml-moe-phasea.h"   // ★R1 #4 — 프로세스 identity(boot_id) 공유
#include "ggml-moe-direct.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <process.h>

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

static_assert(sizeof(ggml_moe_phaseb_header)  == 128, "PbHeader must be exactly 128 bytes (ABI contract, section 2)");
static_assert(sizeof(ggml_moe_phaseb_a0equiv) ==  80, "PbA0Equiv must be exactly 80 bytes (ABI contract, section 3)");
static_assert(sizeof(ggml_moe_phaseb_sample)  == 112, "PbSample must be exactly 112 bytes (ABI contract, section 4)");
static_assert(sizeof(ggml_moe_phaseb_io)      == 112, "PbIo must be exactly 112 bytes (ABI contract, section 5)");
static_assert(sizeof(ggml_moe_phaseb_observe) ==  64, "PbObserve must be exactly 64 bytes (ABI contract, section 13.2)");
static_assert(offsetof(ggml_moe_phaseb_header, signal_observe_on) == 120, "section 13.1 offset 120");
static_assert(offsetof(ggml_moe_phaseb_header, adapter_id)        == 121, "section 13.1 offset 121");
static_assert(offsetof(ggml_moe_phaseb_header, capture_width)     == 122, "section 13.1 offset 122");
static_assert(offsetof(ggml_moe_phaseb_header, reserved2)         == 124, "section 13.1 offset 124");
static_assert(offsetof(ggml_moe_phaseb_observe, pred)             ==  16, "section 13.2 offset 16");
static_assert(offsetof(ggml_moe_phaseb_observe, predict_end_qpc)  ==  48, "section 13.2 offset 48");
// ★FROZEN_PB_OBSV_V2 section 2.2 — V2 는 폭·offset 을 바꾸지 않는다(NO_OUTPUT 은 아래 2축만 사용).
static_assert(offsetof(ggml_moe_phaseb_observe, flags)            ==  14, "PB_OBSV_V2 offset 14");
static_assert(offsetof(ggml_moe_phaseb_observe, invalid_reason)   ==  15, "PB_OBSV_V2 offset 15");
static_assert(offsetof(ggml_moe_phaseb_observe, reserved)         ==  56, "PB_OBSV_V2 offset 56");

constexpr uint32_t PB_PREDICTOR_STACK = 1u << 19;   // worker 스택 예약(mem echo 정확값)
constexpr int32_t  PB_K              = GGML_MOE_PHASEB_TOPK;
// ★observe 레코드 버퍼의 **seal 정적 예약량**(§13.4 `observe_buf_bytes` — "seal 시 정적" 계약).
//   실제 vector::reserve 값과 metrics echo 가 같은 상수를 쓴다(수기 가산 금지 — §7.2 관행).
//   초과분은 BUF_MB 상한(§1) 아래에서 자연 증가하며, 상한 초과는 기존대로 complete=0 이다.
constexpr uint64_t PB_OBSERVE_RESERVE_RECORDS = 4096;

inline uint64_t fnv1a64(uint64_t h, const void * d, size_t n) {
    const uint8_t * p = (const uint8_t *)d;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
constexpr uint64_t FNV_INIT = 0xcbf29ce484222325ull;

void pb_log(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[moe-phaseb] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

// ===========================================================================
// §1. 자료구조
// ===========================================================================

// ★5c-T OFF 바이트 동일성 잠금 — `intent_record_bytes`(metrics `phaseb_mem`/`b1_mem`)가 이
//   sizeof 를 그대로 echo 하므로, 스탬프 N 필드는 기존 정렬 패딩만 채워야 한다(88 불변).
static_assert(sizeof(ggml_moe_phaseb_intent) == 88,
              "ggml_moe_phaseb_intent must stay 88 bytes - metrics intent_record_bytes echoes it "
              "(OFF-run wire byte identity, SPEC_PHASEB_WIRE appendix 14)");
// ledger 레코드 폭 고정(부록 14.3 — 사이드카 헤더 record_bytes 와 1:1).
static_assert(sizeof(ggml_moe_phaseb_ledger_owner) == 16, "pb_ledger owner ref must be 16 bytes");
static_assert(sizeof(ggml_moe_phaseb_ledger) == 320, "pb_ledger record must be 320 bytes");

// capture 슬롯 — 각 슬롯이 자기 사본 버퍼를 소유한다(단일 버퍼 금지 — §2.9).
struct PbCaptureSlot {
    uint32_t request_epoch = 0;
    uint32_t graph_epoch   = 0;
    uint32_t graph_seq     = 0;
    int32_t  source_mlpos  = -1;
    int32_t  source_layer  = -1;
    int32_t  n_embd        = 0;
    uint64_t capture_qpc   = 0;
    std::vector<float> x;              // capture_width(=descriptor n_embd) 고정 — 슬롯 소유
};

// ★NO_OUTPUT 권위 표면 원소(FROZEN_AUTHORITY_V3 §3.2.1 A5 — exact object).
//   `request_index` 는 부모 phase_b_run 레코드가 소유하므로 원소에 중복 탑재하지 않는다.
//   `n_outputs` 는 정의상 0 이라 상수로 방출한다(서버 확정값 — 사후 추론이 아니다).
struct PbNoOutputEntry {
    uint32_t graph_seq      = 0;
    uint32_t decode_ordinal = 0;
    uint16_t target_layer   = 0;   // ★물리 layer id(= moe_layer_ids[-1] · mlpos 금지)
};

// 요청별 phase_b_run 누적(§7.4 · 귀속 = 보존 request_epoch).
struct PbRunAcc {
    bool     used = false;
    uint64_t intents = 0;
    uint64_t skip[GGML_MOE_PB_SKIP_COUNT] = {0};
    uint64_t c3[GGML_MOE_PB_C_COUNT] = {0};
    uint64_t split_required = 0, split_ready = 0, split_join = 0, split_demand_read = 0, split_join_attempt = 0;
    uint64_t test_hint_applied = 0;   // ★정정 6호 — 이 요청 귀속 post-transform enqueue 성공 수(§7.4)
    uint64_t sc_a0equiv = 0, sc_sample = 0, sc_io = 0, sc_observe = 0;
    uint64_t blocked_us = 0, blocked_count = 0;
    // ── ★5c-T 수집 텔레메트리(부록 14.2 — 수집 ON 에서만 채워지고 방출된다) ──
    int32_t  outcome = GGML_MOE_PB_OUTCOME_UNKNOWN;   // 서버 release 판정(미전달 = unknown)
    // ★r1 M1 — generation lifecycle(run_begin 발급 / run_end 폐쇄). controller §1 stamp 의
    //   `admission_seq` 는 **이 단일 generation 순번**이며 candidate range 가 아니다.
    uint64_t admission_seq  = GGML_MOE_PHASEB_NA_U64;
    uint64_t admission_qpc  = 0;
    uint64_t terminal_seq   = GGML_MOE_PHASEB_NA_U64;
    uint64_t terminal_qpc   = 0;
    // controller §2 결정 표본 축(candidate rank 순번 범위) — stamp 가 아니라 진단 원료다.
    uint64_t decision_seq_first = GGML_MOE_PHASEB_NA_U64;
    uint64_t decision_seq_last  = GGML_MOE_PHASEB_NA_U64;
    uint64_t blocked_intervals  = 0;
    // ★r1 M2 — generation admission 시점에 **박제된** 튜플(controller §1 "이미 admission 된
    //   generation 의 스탬프는 이후 어떤 조정으로도 불변"). 방출 시점에 전역을 다시 읽지 않는다.
    int32_t  stamp_k = 0, stamp_n = 0, stamp_qd = 0, stamp_rev = 0;
    std::vector<uint64_t> exp_sum, exp_max, exp_n;   // predictor 노출(target 1..n-1)
    // ★`phase_b_run.no_output_graphs[]`(FROZEN_AUTHORITY_V3 §3.2.1) — 이 요청에서 서버가
    //   `n_outputs==0` 으로 확정한 decode graph 들. **summary 카운터가 대체할 수 없는 권위 표면**이다.
    std::vector<PbNoOutputEntry> no_output;
};

// a0_equiv 대조 표(현재 graph) — target layer별 ref/cpu.
struct PbEquivRow {
    bool     have_ref = false, have_cpu = false;
    uint8_t  reason   = 0;             // 우선 기록된 invalid 사유
    uint16_t ref[PB_K];
    uint16_t cpu[PB_K];
};

// ★5c-A1 `activation_bundle_view`(SPEC_PREFETCH_5CA §1-6 · §1-6-1 VW-10 12키) — 런처가
//   explicit environment block 으로 넘기는 검증 객체의 엔진측 표현. 엔진은 payload 파일을
//   읽지 않으므로(I-2) 이 구조체가 identity 축의 유일한 입력이며, `digest` 는 런처가 계산해
//   보낸 값을 **옮기는** 자리이지 엔진이 계산하는 자리가 아니다(엔진에 SHA-256 없음 — §6-3).
struct PbBundleView {
    bool        accepted = false;      // V-1~V-3 전건 통과 = accepted view
    std::string refusal;               // 미통과 사유(관측 · 통과 시 빈 문자열)
    uint32_t    esv = 0, asv = 0;      // expected/actual schema_version
    std::string ecv, acv;              // expected/actual constants_version
    std::string escope, ascope;        // 64hex scope_id
    std::string eph, aph;              // 64hex payload_hash
    std::string kn;                    // VW-8 `K:N` 목록 — 이 사양에서 kn 의 단일 권위
    std::string digest;                // 64hex identity_digest — echo 공급점
};

struct PbState {
    // ── env/mode ──
    bool        env_resolved = false;
    int32_t     mode = GGML_MOE_PHASEB_OFF;
    std::string out_prefix;
    uint32_t    buf_mb = 512;
    uint64_t    boot_id = 0;
    uint64_t    qpc_freq = 0;
    uint64_t    t0_qpc = 0;

    // ── ★정정 6호 오힌트 test 표면(SPEC_PHASEB_TEST_HINT §1~§3 · qualification 전용) ──
    //   hint_raw = env 원값(미설정 = 빈 문자열 · "설정됐는가"의 권위) / hint_rule = 해소 규칙
    //   / hint_offset = ⌊n_expert/2⌋(seal 확정) / hint_applied = §7.3 프로세스 누적 카운터.
    std::string hint_raw;
    std::string hint_fail_reason;
    int32_t     hint_rule   = GGML_MOE_PHASEB_TEST_HINT_NONE;
    int32_t     hint_offset = 0;
    std::atomic<uint64_t> hint_applied{0};

    // ── prefetch k/N(§1 · seal 1회 해소) ──
    bool        pf_resolved = false;
    bool        pf_on = false;
    int32_t     pf_k = 0, pf_n = 0;
    std::string pf_off_reason = "unset";

    // ── ★5c-T 수집 게이트(부록 14.4) ──
    //   collect_raw = env 원값 해소(pb_resolve_env — out_prefix 결정에 필요하므로 여기서 읽는다)
    //   collect_on  = **seal 1회 해소**된 실효값(이후 immutable · header echo 의 권위).
    //   기존 `MOE_DIRECT_PHASEB` 모드와 독립이라 mode=off + collect=on 조합이 성립한다.
    bool        collect_raw = false;
    bool        collect_on  = false;

    // ── ★5c-A1 adapt 축(SPEC_PREFETCH_5CA §1-6 · §6-3 — collect 축과 **동형 독립 심볼**) ──
    //   adapt_raw = `activation_grant` verifier 단 0~4 **전건 통과**(env 해소 시점 판정).
    //   adapt_on  = **seal 1회 해소**된 실효값(이후 immutable).
    //   ★5c-A1 에는 이 축을 켜는 producer 가 없다 — 발급자는 제품 런처 단일 지점(§6-3)인데
    //     5c-A1 런은 grant 를 싣지 않아 단 1(부재)에서 멈춘다. 여기 서 있는 것은 음성 경로다.
    bool        adapt_raw = false;
    bool        adapt_on  = false;
    //   view 는 env 해소 시점 파싱값(raw)과 **seal 1회 복사본**(snapshot)을 분리해 둔다 —
    //   snapshot 을 고치는 심볼은 seal 과 시나리오 리셋 밖에 존재하지 않는다(§1-6).
    PbBundleView view_raw;
    PbBundleView view;
    bool         view_sealed = false;
    //   grant verifier 거부 사유(§1-7 activation namespace 어휘) — 발급 자체가 없으면 부재다.
    std::string  grant_refusal = "activation_grant_absent";
    //   §6-3 "전건 통과해야 controller 객체 최초 할당" 의 계수. 5c-A1 은 그 객체를 만들지
    //   않으므로(그 시공은 뒤 원자 소관) 이 값은 **할당을 여는 admission** 의 수이며 0 이다.
    std::atomic<uint64_t> controller_alloc{0};

    // ── ★5c-T ledger 사이드카(부록 14.3 — 수집 ON 에서만) ──
    std::mutex  led_mu;                                   // leaf lock
    std::vector<ggml_moe_phaseb_ledger> led_recs;
    std::atomic<uint64_t> led_event_seq{0};
    // ★r1 M1 — generation lifecycle 순번(run_begin admission / run_end terminal 공용 단조 열)과
    //   candidate 결정 표본 순번을 **분리된 두 축**으로 둔다.
    std::atomic<uint64_t> gen_seq_next{0};
    std::atomic<uint64_t> decision_seq_next{0};
    // ★테스트 전용 one-shot 제출 실패 주입(oracle §6-2 ⓑ) — 프로덕션 0.
    std::atomic<int32_t>  test_submit_fail_next{0};
    // ★★R-2 게이트 전용 one-shot predictor worker 생성 실패 주입 — 프로덕션 0.
    std::atomic<int32_t>  test_worker_fail_next{0};
    // ★★U-1(교차 r4) — predictor 정지가 **timeout 으로 끝난** 사실의 sticky 래치.
    //   이게 서면 `ggml_moe_phaseb_reset_for_test()` 첫 줄의 **INFINITE join 을 건너뛴다**.
    //   구 판은 direct 쪽 래치만 있어서, predictor 정지 회귀가 그 첫 줄에서 다시 매달렸다
    //   (= `run_group` 종단 가드에 **도달하기 전** 교착 — r4 U-1 ①).
    std::atomic<int32_t>  stop_timed_out{0};
    // ★테스트 전용 owner 튜플 override(r3 수리 2) — 프로덕션 0(k=0 = 비활성).
    std::atomic<int32_t>  test_stamp_k{0}, test_stamp_n{0}, test_stamp_qd{0};
    // ★테스트 전용 중복 terminal 전이 프로브(r2 M7-1) — 프로덕션 0.
    std::atomic<int32_t>  test_dup_term_arm{0};
    std::atomic<int32_t>  test_dup_term_hits{0};
    std::atomic<int32_t>  test_dup_term_status{0};
    std::atomic<uint64_t> test_dup_term_ctr{0};
    std::atomic<uint64_t> test_dup_term_led{0};
    std::atomic<int32_t>  test_dup_term_qd{0};

    // ── ★계열 신호 capability adapter(SPEC_FAMILY_SIGNAL v0.2 §1 — seal 1회 해소·이후 immutable) ──
    //   fam_* = 생산자가 선언한 descriptor 입력(원천: live GGUF hparams + 로드 텐서 인벤토리).
    //   ad_*  = 닫힌 매핑 표에서 해소된 런타임 능력(norm 텐서·선택식·bias 유무·capture 폭).
    bool        fam_declared = false;
    std::string fam_arch;
    int32_t     fam_gating = 0, fam_groups = 0;
    int32_t     fam_n_embd = 0, fam_n_expert = 0, fam_n_expert_used = 0;
    int32_t     fam_n_layer_exec = 0;
    // ★DSV4 capability 입력(§1.5 확정 목록) — 선언 descriptor 사본. 요구값이 명시된 행에서만 대조.
    int32_t     fam_hash_layers = 0, fam_n_expert_shared = 0, fam_hc_mult = 0, fam_weights_norm = 0;
    float       fam_weights_scale = 0.0f;
    std::vector<int32_t> fam_moe_layers;   // ★실제 moe_layers[](생산자 live 스캔 사본)
    // ★presence / validity 분리 집계 — present∧¬valid(=invalid) 는 **부재가 아니다**.
    struct PbInvSum { int32_t present = 0, valid = 0, invalid = 0; std::string why; };
    PbInvSum    inv_norm_attn, inv_norm_ffn, inv_router, inv_bias_gi, inv_bias_ep;
    // ★텐서 파티션 축(DSV4 §1.5) — **층별** 상태(0=부재 1=valid 2=present∧invalid). 집계(inv_*)는
    //   층 정보를 잃어 "몇 개"만 알 뿐이라 배타 파티션(0..hash-1 ↔ hash..n-1)을 판정할 수 없다.
    //   ★범위는 il = 0..n_layer_exec-1 **전 층**이다(집계 inv_bias_ep 의 il>=1 범위와 별개 축 —
    //   기존 계열의 집계 의미·값은 이 배열 도입으로 전혀 바뀌지 않는다).
    static const uint8_t INV_ABSENT = 0, INV_VALID = 1, INV_INVALID = 2;
    std::vector<uint8_t> inv_tid_state, inv_ep_state;
    std::string inv_tid_why, inv_ep_why;
    // ── ★denominator/source norm 원소별 exact-zero inventory(FROZEN_NORM_INVENTORY §3·§4) ──
    //   nz_* = note_family 의 **기존 finite 전량 순회에 합류**해 모은 층별 zero 좌표
    //   (0-based 오름차순 u32 · 전량 · 절단 없음 · 인덱스 = il). 이 시점엔 norm_kind 가 아직
    //   해소되지 않았으므로 두 후보 텐서를 모두 모아 두고, seal 해소(pb_resolve_adapter)에서
    //   채택 계열의 것만 계상한다.
    std::vector<std::vector<uint32_t>> nz_attn, nz_ffn;
    //   nv_* = 해소 산출(= metrics `observe_norm_inventory` exact block 의 단일 출처).
    //   nv_measured=false 는 "zero 0건"이 아니라 **미측정**이다(판독기 fail-close · 기본값 0
    //   합성 금지 — §4 "echo 부재 = 결손").
    struct PbNormZeroRec {
        int32_t  source_layer    = -1;
        int32_t  affected_target = -1;   // ★규범식 moe_layer_ids[mlpos+1](산술 source+1 아님)
        uint64_t digest          = 0;    // §3.3 FNV-1a 64
        std::vector<uint32_t> idx;       // 오름차순 0-based · 전량
    };
    bool        nv_measured = false;
    int32_t     nv_scanned  = 0;         // source_layers_scanned = |moe_layer_ids[:-1]|
    uint64_t    nv_zero_elements = 0;
    std::vector<PbNormZeroRec> nv_zero;
    int32_t     ad_id          = GGML_MOE_PHASEB_ADAPTER_NONE;
    const char *ad_name        = "";
    int32_t     ad_norm_kind   = GGML_MOE_PHASEB_NORM_NONE;
    int32_t     ad_selection_fn= GGML_MOE_PHASEB_SEL_NONE;
    int32_t     ad_bias_kind   = GGML_MOE_PHASEB_BIAS_NONE;
    // ★capability `output_crop_before_last_moe`(FROZEN_PB_OBSV_V2 §7 E6) — 표의 명시값 사본.
    bool        ad_out_crop    = false;
    // ★층 마스크(DSV4 §1.5) — 해소된 표 행의 `first_target_layer`. 미해소 = 1(중립 = 종전 의미).
    int32_t     ad_first_target = 1;
    // ★router dtype allowlist(SPEC_FAMILY_SIGNAL §1.6) — 해소된 표 행의 `router_bf16_ok` 사본.
    //   미해소 = false(전역 완화 금지 · 역할×계열로만 열린다).
    bool        ad_router_bf16_ok = false;
    int32_t     capture_width  = 0;
    std::string ad_reason      = "descriptor_absent";

    // ── ★observe 축(SPEC_PHASEB_WIRE §13 · SPEC_FAMILY_SIGNAL §2.1) ──
    //   obs_on   = 실행 축(seal 1회 확정) · obs_reason = seal 전제 위반 사유(fail-close 메시지)
    //   obs_recs = 사이드카 레코드 버퍼(행은 graph_begin 에서 **전 target 선생성**하고 capture/
    //              worker 가 사후 patch 한다 — 드레인 없이 coverage 를 구조로 보장한다)
    bool                  obs_on = false;
    std::string           obs_reason;
    std::mutex            obs_mu;                 // leaf lock — cap_mu → obs_mu 방향만 존재
    std::vector<ggml_moe_phaseb_observe> obs_recs;
    std::unordered_map<uint32_t, uint32_t> obs_base_by_graph;   // graph_seq → 첫 행 index
    // ★NO_OUTPUT(FROZEN_PB_OBSV_V2 §4.2 E3) — 서버 확정 zero-output 으로 판정된 graph 집합.
    //   worker 가 "이 graph 의 마지막 관리 MoE target 은 측정 대상 부재"임을 **intent enqueue 이전**에
    //   확인하는 창구다(§1.5 생산 조건 "enqueue 전 억제"). obs_mu 가 지킨다(leaf lock).
    std::unordered_set<uint32_t> obs_no_out_graph;
    // ★억제 **발동** 계수(진단·selftest 전용 — wire/metrics 표면 아님). 억제 가드가 실제로
    //   intent 발행 경로를 끊은 횟수다. observe 축은 speculative enqueue 가 구조적으로 0 이라
    //   "enqueue 0" 검사만으로는 가드 유무를 구분할 수 없다 — 이 계수가 그 구분자다.
    std::atomic<uint64_t> obs_no_out_suppressed{0};
    std::atomic<uint64_t> obs_ctr[GGML_MOE_PB_O_COUNT];
    // ★훅 재개봉 라운드 B — replay 추정 ②③의 **U18 predictor cost 입력**(PREFETCH_DESIGN §3.2
    //   "상한값"). 의미 = capture_qpc → predict_end_qpc 구간의 **warmup 포함 전 표본 단일
    //   최댓값**(ns) + 표본 수. ★승격 지표 `predictor_exposed`(§7.5 = predict_end − R_qpc(source))
    //   와는 **다른 축**이다(이쪽은 predictor 자체 비용·저쪽은 demand 대기와의 교차).
    //   관측 무개입 — 이미 찍고 있는 QPC 2개의 집계일 뿐 신규 계측 지점이 없다.
    std::atomic<uint64_t> obs_cost_ns_max{0};
    std::atomic<uint64_t> obs_cost_samples{0};

    // ── seal 상수 ──
    bool     sealed = false;
    int32_t  n_expert = 0, n_expert_used = 0, n_moe_layers = 0, n_slots = 0, qd_effective = 0;
    std::vector<int32_t>  moe_layers;
    std::vector<uint64_t> strides;
    std::vector<int32_t>  layer_to_mlpos;   // 실제 layer id → mlpos(-1)

    // ── 스냅샷(§3) ──
    bool     snap_configured = false, snap_sealed = false;
    int32_t  n_layer_exec = 0, n_embd = 0;
    float    rms_eps = 1e-6f;
    std::vector<float> snap_router;   // (n_layer_exec-1) × n_embd × n_expert  (layer 1..n-1)
    std::vector<float> snap_norm;     // n_layer_exec × n_embd                 (layer 0..n-1)
    // ★계열 selection-bias 스냅샷(SPEC_FAMILY_SIGNAL §1) — (n_layer_exec-1) × n_expert (layer 1..n-1).
    //   bias 를 요구하지 않는 계열(qwen35moe)에서는 비어 있고 회계값도 0이다.
    std::vector<float> snap_bias;
    std::vector<uint8_t> snap_router_ok, snap_norm_ok, snap_bias_ok;

    // ── epoch(§2.1) ──
    std::atomic<uint32_t> request_index{GGML_MOE_PHASEB_NA_U32};
    std::atomic<uint32_t> decode_ordinal{GGML_MOE_PHASEB_NA_U32};
    std::atomic<uint32_t> graph_seq{0};
    std::atomic<uint32_t> cur_graph_seq{GGML_MOE_PHASEB_NA_U32};
    std::atomic<uint8_t>  cur_phase{0};
    uint32_t  next_request_index = 0;
    uint32_t  next_decode_ordinal = 0;
    uint64_t  total_requests = 0;
    bool      run_active = false;
    bool      graph_active = false;

    // ── capture_q ──
    std::mutex              cap_mu;
    std::condition_variable cap_cv;      // worker 기상
    std::condition_variable cap_idle_cv; // 테스트 idle 대기
    std::vector<PbCaptureSlot> cap_slots;
    std::vector<int32_t>    cap_free;
    std::vector<int32_t>    cap_ring;
    int32_t   cap_head = 0, cap_count = 0;
    // ★R2 #1 — PB_FINALIZE_CUT ① 선형화점의 **단일 원자 래치**. capture 훅과 direct 의 cq_alloc
    //   이 **이 하나**를 확인한다(래치가 둘이면 그 사이에 시작된 capture 가 command CUT 이후에도
    //   수용되는 창이 생긴다 — R2 적발). 세터는 cap_mu 를 잡고 store 하므로, cap_mu 임계 구간
    //   안에서 enqueue 하는 capture 는 "래치 전 완료" 또는 "래치를 보고 skip" 둘 중 하나다.
    std::atomic<bool> cut_closed{false};
    // ★보완(R4) 테스트 전용 one-shot barrier — capture 를 (1) cap_mu 획득 **전** 또는
    //   (2) 등록(래치 확인 통과) **직후** 에 정지시켜 CUT 와의 승패를 결정론적으로 통제한다.
    //   프로덕션 0(reset 복귀) · 기존 test_force_sync_next/stop_race 훅 선례와 동형.
    std::atomic<int32_t> test_cap_barrier{0};
    void * cap_bar_hit = nullptr;
    void * cap_bar_go  = nullptr;
    // ★R7 — barrier 는 **release 호출로만** 풀린다(자가 해제 금지). 아래 둘은 진단용:
    //   bar_held      = 현재 barrier 안에 정지한 스레드 수(>0 이면 mode-2 는 cap_mu 보유 중)
    //   bar_safety    = 안전 타임아웃(60s)이 발동했는가(발동 = 픽스처 결함 · 케이스 FAIL 신호)
    std::atomic<int32_t> bar_held{0};
    std::atomic<int32_t> bar_safety{0};
    // E-2a (Codex r2 blocker 2) - test-only one-shot park inside the worker's REAL busy window
    // (pb_worker_process, no mutex held), so a stop's predictor drain has something genuine to
    // spend its budget on. Same shape as the cap barrier above: armed by the test, released by
    // the test, with a hard bound so a fixture that forgets cannot hang a run. Production keeps
    // the arm at 0 and pays one relaxed load per processed capture.
    std::atomic<int32_t> test_worker_hold{0};
    void * whold_hit = nullptr;
    void * whold_go  = nullptr;
    std::atomic<int32_t> whold_safety{0};
    // E-2a (Codex r2 blocker 2) - the stages of the last predictor drain, in entry order:
    //   [0] capture-idle wait   [1] worker join
    // Each row is the ONE deadline that stage measured against and what it was handed at its
    // own entry. This is what lets a gate say "the multi-stage path actually ran" instead of
    // assuming it did - and, since a stage that is skipped records nothing, it separates
    // "drained fast" from "never entered".
    std::atomic<int32_t> drain_stage_n{0};
    std::atomic<int64_t> drain_stage_deadline[2];
    std::atomic<int32_t> drain_stage_left[2];
    // ★R5 테스트 전용 rendezvous — CUT 가 cap_mu 를 **시도**한 순간과 **취득**한 순간을 각각 노출해,
    //   "CUT 가 실제로 경합 위치에서 막혀 있다"를 Sleep 추정이 아니라 **관측**으로 확정한다.
    //   프로덕션 오버헤드 = relaxed atomic 로드 1회(플래그 off 면 완전 no-op).
    std::atomic<int32_t> test_cut_rv{0};
    void * cut_attempted = nullptr;
    void * cut_contended = nullptr;   // ★R6 — try_lock 실패 = "실제로 막혔다"의 **양성** 증거
    void * cut_acquired  = nullptr;
    bool      worker_busy = false;
    std::vector<uint8_t> cap_layer_seen; // graph 내 층별 capture 중복 방지(canonical part 1회)
    uint32_t  cap_layer_seen_graph = GGML_MOE_PHASEB_NA_U32;

    // ── predictor worker ──
    uintptr_t worker_thread = 0;
    std::atomic<bool> worker_running{false};
    std::atomic<bool> worker_stop{false};
    std::vector<float> w_norm_buf, w_logit_buf;
    std::vector<char>  w_used_buf;

    // ── intent_q ──
    std::mutex int_mu;
    std::vector<ggml_moe_phaseb_intent> int_slots;
    int32_t  int_head = 0, int_count = 0;
    uint64_t intent_ordinal_next = 0;
    std::atomic<uint32_t> disabled_request{GGML_MOE_PHASEB_NA_U32}; // ★R1 #11 — worker 쓰기/훅 읽기 교차
    bool     lora_swapped = false;

    // ── 카운터/digest ──
    std::atomic<uint64_t> ctr[GGML_MOE_PB_C_COUNT];
    std::atomic<uint64_t> skip_ctr[GGML_MOE_PB_SKIP_COUNT];
    std::atomic<uint64_t> cap_overflow{0};
    std::atomic<uint64_t> blocked_us_total{0}, blocked_count_total{0};
    std::atomic<uint32_t> qd_max_demand{0}, qd_max_prefetch{0}, qd_max_total{0};
    uint64_t dg_intent = FNV_INIT, dg_disp = FNV_INIT, dg_trans = FNV_INIT,
             dg_class = FNV_INIT, dg_readfile = FNV_INIT;
    uint64_t dg_readfile_ops = 0;
    uint64_t intent_count = 0;

    // ── 요청별 누적 ──
    std::mutex run_mu;
    std::vector<PbRunAcc> runs;

    // ── a0_equiv ──
    std::mutex equiv_mu;
    std::vector<PbEquivRow> equiv_rows;    // 현재 graph의 target layer별(1..n_layer_exec-1)
    std::vector<ggml_moe_phaseb_a0equiv> equiv_recs;

    // ── predictor 노출 timing 조인(§7.5) + capture 결손 사유 전달 ──
    // ★R1 #11 — pend_by_target/skip_by_target/overflow_by_target 은 worker(쓰기)와
    //   graph_end·dispatcher(읽기)가 공유하는 mailbox다. 전용 mutex로 규율한다(trace 모드 포함).
    std::mutex tm_mu;
    std::vector<uint64_t> r_qpc_by_layer;  // 현재 graph(decode 스레드 전용)
    std::vector<uint64_t> pend_by_target;  // 현재 graph predict_end(worker→graph_end)
    std::vector<uint8_t>  skip_by_target;  // ★R1 #10 ② 실제 elig_skip 사유(capture/worker→행)
    std::vector<uint8_t>  ovf_by_target;   // ★R1 #10 ③ intent_dropped_overflow bit producer
    uint32_t              mailbox_graph = GGML_MOE_PHASEB_NA_U32;
    // ★R2 #5 — 비동기 사유(worker NONFINITE·intent overflow)는 graph 가 바뀐 뒤에도 소실되면
    //   안 되고 **이미 생성된 pb_sample 행**까지 갱신해야 한다. mailbox(현 graph 전용)와 별도로
    //   (graph_seq, target) 를 담은 상태 큐를 두고 **dispatcher 가 g_mod.mutex 아래에서 소비**한다
    //   — worker 는 tm_mu 만 잡고 append 하므로 락 순서(g_mod→q→int→{run,tm})가 역전되지 않는다.
    //   ★누수 수리(R2 후속) — vector append/head 전진 방식은 head가 정확히 따라잡을 때만
    //   clear() 되어 **size()가 단조 증가**했다(생산자가 계속 밀면 메모리 무한 증가). 고정 링으로
    //   바꿔 상한을 구조로 못박는다. 링 만석 = 관측 결손이므로 complete=0 을 latch 한다.
    struct PbStatusEvt { uint32_t graph_seq; int32_t target_mlpos; uint8_t reason; uint8_t overflow; };
    static const size_t      STATUS_RING = 2048;
    std::vector<PbStatusEvt> status_q;      // 크기 = STATUS_RING(고정 · 최초 사용 시 1회 할당)
    size_t                   status_head = 0, status_count = 0;

    // ★R1 #2 ① — intent enqueue 가 dispatcher를 깨우는 통로(lost-wake 규율: enqueue+SetEvent를
    //   int_mu 아래에서 한 묶음으로 하고, dispatcher는 command·channel·intent 3원 공백을
    //   q_mutex→int_mu 순으로 확인한 뒤에만 ResetEvent 한다 — direct.cpp disp_peek).
    void * wake_event = nullptr;
    // ★R1 #12 — "PB off ∧ prefetch off" fast-gate(캐시된 bool 1분기). 훅 최전방에서 즉시 반환.
    std::atomic<bool> all_off{true};

    // ── 버퍼 상한 ──
    std::atomic<uint64_t> buf_used{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<bool>     complete{true};
    // ★incomplete 사유 마스크(FROZEN_INCOMPLETE_REASON_MASK §3·§4.3) — sticky(회복 없음).
    //   complete 와 **같은 수명**을 갖고 같은 자리에서만 리셋된다(항등식이 항상 성립하도록).
    std::atomic<uint32_t> incomplete_mask{0};
    std::atomic<bool>     run_invalid{false};
    std::atomic<bool>     predictor_drained{false};   // ★R1 #3 ② CUT ② 완료 래치
};

PbState g_pb;

// ===========================================================================
// §2. env 해소
// ===========================================================================

std::string pb_getenv(const char * n) { const char * v = std::getenv(n); return v ? std::string(v) : std::string(); }

std::string pb_trim_lower(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    std::string o = s.substr(a, b - a);
    for (auto & c : o) c = (char)std::tolower((unsigned char)c);
    return o;
}

void pb_acquire_identity();

// ===========================================================================
// §2-A. 닫힌 grammar 파서 — `activation_bundle_view`(§1-6-1) · `activation_grant`(§6-3)
//   ★엔진 파서는 **1종**이다(I-5 "공유가 아니라 단일화") — 두 payload 가 같은 문법을 쓰고
//     키 표만 다르다. canonical JSON loader 는 런처 1곳뿐이라 여기서 JSON 을 파싱하지 않는다.
//   ★두 verifier 모두 **fail-closed** 이고 어느 단에서도 allocation 을 하지 않는다.
// ===========================================================================

constexpr char PB_KV_SEP = '\x1f';        // VW-3 unit separator(1바이트)

// VW-4 — 값 문자 집합 `[0-9a-zA-Z._:,-]`. 전건 ASCII 단일 바이트라 UTF-8/UTF-16 왕복에서
//        값이 변형되지 않는다(집합 밖 1바이트라도 = 거부).
bool pb_kv_charset_ok(const std::string & v) {
    if (v.empty()) return false;
    for (const char ch : v) {
        const unsigned char c = (unsigned char)ch;
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        c == '.' || c == '_' || c == ':' || c == ',' || c == '-';
        if (!ok) return false;
    }
    return true;
}

// VW-5 — 10진 · 선행 0 금지 · 부호 금지 · padding 금지(`sv=01`·`sv=+1` 은 문법 위반).
bool pb_kv_uint_ok(const std::string & v, uint32_t * out) {
    if (v.empty() || v.size() > 9) return false;
    if (v.size() > 1 && v[0] == '0') return false;
    uint32_t acc = 0;
    for (const char ch : v) {
        if (ch < '0' || ch > '9') return false;
        acc = acc * 10u + (uint32_t)(ch - '0');
    }
    if (out) *out = acc;
    return true;
}

// VW-6 — hex 는 **소문자만** · 길이 exact(대문자는 "같은 값의 다른 표기"가 아니라 문법 위반).
bool pb_kv_hex_ok(const std::string & v, size_t n) {
    if (v.size() != n) return false;
    for (const char ch : v) {
        const bool ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        if (!ok) return false;
    }
    return true;
}

// VW-8 — `K:N` 쌍을 `,` 로 연결 · payload 원소 순서 보존 · 중복 원소 거부 · 빈 값 거부.
bool pb_kv_kn_ok(const std::string & v) {
    if (v.empty()) return false;
    std::vector<std::string> seen;
    size_t i = 0;
    while (true) {
        const size_t j = v.find(',', i);
        const std::string elem = v.substr(i, (j == std::string::npos ? v.size() : j) - i);
        const size_t c = elem.find(':');
        if (c == std::string::npos) return false;
        if (!pb_kv_uint_ok(elem.substr(0, c), nullptr)) return false;
        if (!pb_kv_uint_ok(elem.substr(c + 1), nullptr)) return false;
        for (const std::string & s : seen) { if (s == elem) return false; }
        seen.push_back(elem);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return true;
}

// VW-9 / PF-6·PF-7 — preimage 의 fnv1a64 를 **16자리 소문자 hex(선행 0 유지)** 로.
//   알고리즘은 이 파일의 원시 `fnv1a64` 자신이라 런처와 바이트 단위로 같은 축을 쓴다.
std::string pb_kv_fnv_hex(const std::string & preimage) {
    const uint64_t h = fnv1a64(FNV_INIT, preimage.data(), preimage.size());
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

// VW-3·VW-7 — `k=v` 쌍을 separator 로 끊어 **키 순서 고정 · unknown/결손 0** 으로 읽는다.
//   선두·말미·연속 separator 는 **빈 조각**을 만들므로 그 자리에서 거부된다.
bool pb_kv_split(const std::string & s, const char * const * keys, size_t n_keys,
                 std::vector<std::string> & out) {
    out.clear();
    if (s.empty()) return false;
    size_t i = 0, idx = 0;
    while (true) {
        const size_t j = s.find(PB_KV_SEP, i);
        const std::string pair = s.substr(i, (j == std::string::npos ? s.size() : j) - i);
        if (pair.empty() || idx >= n_keys) return false;
        const size_t e = pair.find('=');
        if (e == std::string::npos) return false;
        if (pair.compare(0, e, keys[idx]) != 0) return false;   // 키 순서 고정 · unknown 거부
        const std::string val = pair.substr(e + 1);
        if (!pb_kv_charset_ok(val)) return false;
        out.push_back(val);
        idx++;
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return idx == n_keys;                                        // 결손 거부
}

// PF-1 / VW-9 — preimage = 마지막 `fnv` 쌍과 **그 앞 separator 를 뺀** 바이트열.
//   재직렬화가 아니라 원문 절단이라 표기 차이가 끼어들 자리가 없다.
bool pb_kv_preimage(const std::string & s, std::string & out) {
    const size_t p = s.rfind(PB_KV_SEP);
    if (p == std::string::npos || p == 0) return false;
    out = s.substr(0, p);
    return true;
}

// ★§1-6-1 VW-10 — view 키 표(순서 고정 · 정확 12키).
const char * const kViewKeys[12] = {
    "v", "esv", "ecv", "escope", "eph", "asv", "acv", "ascope", "aph", "kn", "digest", "fnv"
};
// ★§6-3 — grant 키 표(순서 고정 · 정확 8키 · r4 에서 `kn` 제거 = snapshot 단일 권위).
const char * const kGrantKeys[8] = {
    "v", "mode", "attempt", "sv", "cv", "scope", "ph", "fnv"
};

// ★view verifier — fail-closed 3단(V-1~V-3). 실패는 전부 **view 부재로 취급**되며
//   그 귀결은 `prefetch_identity_digest = null` 이다(adapt 불가).
PbBundleView pb_activation_bundle_view_parse(const std::string & raw) {
    PbBundleView v;
    std::vector<std::string> f;
    // V-1 — 존재 ∧ v==1 ∧ 키 집합 정확 12 ∧ unknown/결손 0 ∧ VW-4~VW-8 형식 전건.
    if (raw.empty())                         { v.refusal = "view_absent";    return v; }
    if (!pb_kv_split(raw, kViewKeys, 12, f)) { v.refusal = "view_malformed"; return v; }
    uint32_t vv = 0, esv = 0, asv = 0;
    if (!pb_kv_uint_ok(f[0], &vv) || vv != 1) { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_uint_ok(f[1], &esv))           { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_charset_ok(f[2]))              { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_hex_ok(f[3], 64))              { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_hex_ok(f[4], 64))              { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_uint_ok(f[5], &asv))           { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_charset_ok(f[6]))              { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_hex_ok(f[7], 64))              { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_hex_ok(f[8], 64))              { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_kn_ok(f[9]))                   { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_hex_ok(f[10], 64))             { v.refusal = "view_malformed"; return v; }
    if (!pb_kv_hex_ok(f[11], 16))             { v.refusal = "view_malformed"; return v; }
    // V-2 — `fnv` 재계산 일치(전달 완전성 축 · 위조 저항이 아니다).
    std::string pre;
    if (!pb_kv_preimage(raw, pre) || pb_kv_fnv_hex(pre) != f[11]) {
        v.refusal = "view_integrity"; return v;
    }
    // V-3 — L-C 의 4필드 재확인. 두 벌이 **각각** 도착해야 성립하며, 한 벌만 보내고
    //       "일치했으니 보냈다" 를 믿는 형태는 자기대조라 계약이 아니다.
    if (esv != asv || f[2] != f[6] || f[3] != f[7] || f[4] != f[8]) {
        v.refusal = "view_identity_mismatch"; return v;
    }
    v.esv = esv;  v.asv = asv;
    v.ecv = f[2]; v.escope = f[3]; v.eph = f[4];
    v.acv = f[6]; v.ascope = f[7]; v.aph = f[8];
    v.kn  = f[9]; v.digest = f[10];
    v.accepted = true;
    return v;
}

// ★grant verifier — fail-closed 4단 + 단 0(§6-3). 반환 = 전건 통과 여부이고,
//   어느 단에서 멈추든 **allocation 은 0** 이다. 거부 사유는 activation namespace 어휘다.
bool pb_activation_grant_verify(const std::string & raw, const PbBundleView & view,
                                std::string & refusal) {
    // 단 0 — accepted view 선행. view 가 없으면 단 4 가 비교 상대(expected[4])를 갖지 못한다.
    if (!view.accepted) { refusal = "activation_grant_identity_mismatch"; return false; }
    // 단 1 — 존재 ∧ v==1 ∧ 키 집합 정확 8 ∧ unknown/결손 0 ∧ VW-4~VW-7 형식 전건.
    if (raw.empty())    { refusal = "activation_grant_absent";            return false; }
    std::vector<std::string> f;
    if (!pb_kv_split(raw, kGrantKeys, 8, f)) { refusal = "activation_grant_malformed"; return false; }
    uint32_t vv = 0, sv = 0;
    if (!pb_kv_uint_ok(f[0], &vv) || vv != 1) { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_charset_ok(f[1]))              { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_hex_ok(f[2], 32))              { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_uint_ok(f[3], &sv))            { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_charset_ok(f[4]))              { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_hex_ok(f[5], 64))              { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_hex_ok(f[6], 64))              { refusal = "activation_grant_malformed"; return false; }
    if (!pb_kv_hex_ok(f[7], 16))              { refusal = "activation_grant_malformed"; return false; }
    // 단 2 — `fnv` 재계산 일치(PF-1~PF-7).
    std::string pre;
    if (!pb_kv_preimage(raw, pre) || pb_kv_fnv_hex(pre) != f[7]) {
        refusal = "activation_grant_malformed"; return false;
    }
    // 단 3 — `mode == "product"`. 검증 모드는 `mode` 에 자기 이름을 싣고 엔진이 그것을 보고
    //        거부한다 — 런처가 실수로 넘겨도 여기서 막힌다(운영 규율이 구조가 되는 지점).
    if (f[1] != "product") { refusal = "activation_grant_mode_not_product"; return false; }
    // 단 4 — `sv`·`cv`·`scope`·`ph` == snapshot 의 **expected[4]**. actual 과 비교하면
    //        "bundle 이 말한 값과 bundle 에서 읽은 값" 의 자기대조가 된다.
    if (sv != view.esv || f[4] != view.ecv || f[5] != view.escope || f[6] != view.eph) {
        refusal = "activation_grant_identity_mismatch"; return false;
    }
    refusal.clear();
    return true;
}

void pb_resolve_env() {
    if (g_pb.env_resolved) return;
    g_pb.env_resolved = true;

    const std::string m = pb_trim_lower(pb_getenv("MOE_DIRECT_PHASEB"));
    if      (m.empty() || m == "off") g_pb.mode = GGML_MOE_PHASEB_OFF;
    else if (m == "trace")            g_pb.mode = GGML_MOE_PHASEB_TRACE;
    else if (m == "equiv")            g_pb.mode = GGML_MOE_PHASEB_EQUIV;
    else if (m == "observe")          g_pb.mode = GGML_MOE_PHASEB_OBSERVE;   // ★8호
    else {
        g_pb.mode = GGML_MOE_PHASEB_OFF;
        pb_log("MOE_DIRECT_PHASEB=\"%s\" unknown value - using off (off|trace|equiv|observe)", m.c_str());
    }
    // ★5c-T(부록 14.4) — 수집 플래그 env 원값 해소. `MOE_DIRECT_PHASEB` 모드와 **독립**이므로
    //   mode=off + collect=on 조합에서도 사이드카 경로(out_prefix)가 살아 있어야 한다.
    //   ★유효값은 exact `on` 1종(그 외·미설정 = off) — 기본 OFF 가 계약이다.
    g_pb.collect_raw = (pb_trim_lower(pb_getenv("MOE_DIRECT_PHASEB_COLLECT")) == "on");
    g_pb.out_prefix = (g_pb.mode == GGML_MOE_PHASEB_OFF && !g_pb.collect_raw)
                          ? std::string() : pb_getenv("MOE_DIRECT_PHASEB_OUT");
    // ★5c-A1 identity 축(§1-6 · §1-6-1 · §6-3) — 런처가 조립한 explicit environment block 의
    //   두 키를 **엔진이 판정**한다. 디스크 재독 0 · payload 파싱 0(I-2) — 도착한 바이트만 본다.
    //   view 는 accepted load 마다, grant 는 활성화가 승인된 런에만 온다(수명·존재 조건이 다르다).
    g_pb.view_raw  = pb_activation_bundle_view_parse(pb_getenv("MOE_DIRECT_PREFETCH_BUNDLE_VIEW"));
    g_pb.adapt_raw = pb_activation_grant_verify(pb_getenv("MOE_DIRECT_PREFETCH_ACTIVATION_GRANT"),
                                                g_pb.view_raw, g_pb.grant_refusal);
    if (g_pb.adapt_raw) {
        // §6-3 — verifier 전건 통과가 controller 객체 최초 할당을 여는 유일한 문이다. 그 객체
        //   자체는 이 원자의 범위 밖이므로 여기서 세는 것은 **할당을 여는 admission** 이고,
        //   발급자가 없는 5c-A1 런에서는 이 분기에 도달하지 않는다(계수 전건 0).
        g_pb.controller_alloc.fetch_add(1, std::memory_order_relaxed);
    }
    {
        uint32_t mb = 512;
        const std::string s = pb_getenv("MOE_DIRECT_PHASEB_BUF_MB");
        if (!s.empty()) {
            char * e = nullptr;
            const long v = strtol(s.c_str(), &e, 10);
            if (e && *e == '\0' && v > 0) mb = (uint32_t)v;
            else pb_log("MOE_DIRECT_PHASEB_BUF_MB=\"%s\" is invalid - using the default 512", s.c_str());
        }
        g_pb.buf_mb = mb;
    }
    // ★정정 6호 — 오힌트 test 표면(SPEC_PHASEB_TEST_HINT §1). 유효값은 **exact `rot` 1종**이며
    //   그 외 비어있지 않은 값은 rule=NONE 으로 남겨 seal precheck 가 fail-close 시킨다(여기서
    //   조용히 off 로 흡수하면 무효값 런이 그대로 통과한다). 게이팅(qualifier 결속·equiv 배타)은
    //   precheck 가 담당하고, 이 절은 값 해소만 한다.
    g_pb.hint_raw    = pb_getenv("MOE_DIRECT_PHASEB_TEST_HINT");
    g_pb.hint_rule   = (g_pb.hint_raw == "rot") ? GGML_MOE_PHASEB_TEST_HINT_ROT
                                                : GGML_MOE_PHASEB_TEST_HINT_NONE;
    g_pb.hint_offset = 0;   // n_expert 의존 — note_seal 에서 확정
    LARGE_INTEGER f{}, c{};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    g_pb.qpc_freq = (uint64_t)f.QuadPart;
    g_pb.t0_qpc   = (uint64_t)c.QuadPart;
    pb_acquire_identity();
}

// ★R1 #4 / R2 #3 / v4 — 프로세스 identity 획득의 **단일 지점**. wire v4 기준 **공유 대상은
//   boot_id 축뿐**이다(§9 게이트 6 = boot_id 3자 일치). t0_qpc 는 v4에서 PB-local 기준시각으로
//   확정됐고, qpc_frequency 는 시스템 전역 상수라 공유 절차가 필요 없다.
//     · qpc_frequency : QueryPerformanceFrequency 는 시스템 전역 불변 상수라 PA/PB 값이
//                       **구조적으로 동일**하다(별도 소비 경로 불요).
//     · boot_id       : PA 활성 시 ggml_moe_phasea_boot_id() 를 그대로 소비한다.
//     · t0_qpc        : ★v4 확정 — **PB-local 기준시각**이다(PA 와 공유하지 않는다). PA 공개
//                       API 에 t0 getter 가 없고 boot_id 는 t0 로 역산되지 않으므로, wire v4 가
//                       t0 를 "자체 기준시각"으로 정정했다. 조인 키는 boot_id 축뿐이다(게이트 6).
//   PA 비활성(PB 단독)일 때의 boot_id 산식도 PA와 **동일 순서 (t0_qpc, pid)** 로 통일한다.
void pb_acquire_identity() {
    if (ggml_moe_phasea_enabled()) {
        g_pb.boot_id = ggml_moe_phasea_boot_id();
        return;
    }
    uint64_t b = FNV_INIT;
    const uint32_t pid = (uint32_t)GetCurrentProcessId();
    b = fnv1a64(b, &g_pb.t0_qpc, sizeof(g_pb.t0_qpc));
    b = fnv1a64(b, &pid, sizeof(pid));
    g_pb.boot_id = b;
}

// ★R1 #12 — fast-gate 재계산(env/prefetch 해소·seal 시점마다).
void pb_refresh_all_off() {
    // ★r1 M5 — 수집 축은 PB 모드·프리페치와 **독립**이다(부록 14.4). collect 를 무시하면
    //   `mode=off + prefetch=off + collect=on` 조합에서 run lifecycle 이 발급되지 않아 회계가
    //   NA request(bucket 0)로 뭉개진다. collect_raw 를 함께 보는 이유는 이 함수가 seal 이전
    //   (resolve_prefetch)에도 호출되기 때문이다 — 수집 OFF 런에서는 두 값 모두 false 라
    //   게이트 값이 종전과 바이트 단위로 동일하다.
    const bool collect = g_pb.collect_raw || g_pb.collect_on;
    g_pb.all_off.store(g_pb.mode == GGML_MOE_PHASEB_OFF && !g_pb.pf_on && !collect,
                       std::memory_order_relaxed);
}

// ===========================================================================
// §2.1 ★계열 신호 capability 표 (SPEC_FAMILY_SIGNAL v0.2 §1 — 닫힌 매핑)
// ===========================================================================
//   행 = 계열 1종. `registered=false` 행은 **자리만** 만들어 둔 것으로, 해소 시 항상
//   `none`(사유 adapter_unregistered)이 된다 — 3단계(gpt-oss)·4단계(K2.6)에서 켠다.
//   ★표의 값이 곧 계약이다: norm 텐서 선택·선택식·bias 유무를 코드 분기가 아니라 이 행에서
//   읽는다(qwen 경로는 아래 첫 행 그대로 = 기존 하드코딩과 동일 텐서·동일 산식).
struct PbAdapterSpec {
    int32_t      id;
    const char * name;
    const char * arch;           // general.architecture 완전 일치
    int32_t      gating_func;    // 요구 expert_gating_func(qwen35moe·gpt-oss = 미설정 0)
    int32_t      norm_kind;
    int32_t      selection_fn;
    int32_t      bias_kind;      // 요구 bias 종류(NONE = **부재가 필수**)
    int32_t      leading_dense;  // 요구 첫 관리 MoE 층(qwen=0 · K2.6=1)
    bool         registered;     // 2단계 등재 여부
    // ★capability `output_crop_before_last_moe`(FROZEN_PB_OBSV_V2 §7 · E6) — **행마다 명시**한다.
    //   그래프 빌더가 마지막 MoE 층 **앞에서** 출력 행을 축소하는 계열인가. true 인 계열에서만
    //   `n_outputs==0` graph 의 마지막 관리 MoE 행이 NO_OUTPUT 으로 전환된다. 값이 명시·검증되지
    //   않은 계열은 자동 false 상속이 아니라 adapter 미해소(=observe 신호 비활성)로 처리된다.
    bool         output_crop_before_last_moe;
    // ── ★층 마스크(DSV4_ADAPTER_DESIGN §1.5) — learned target 의 최소 물리 층. 기존 계열 1. ──
    int32_t      first_target_layer;
    // ── ★hparams 검증형 capability 확정 목록(DSV4_ADAPTER_DESIGN §1.5). **-1 = 미검사**이며
    //    기존 3행은 전건 미검사라 종전 거동이 바이트 동일하다. 값이 박힌 행은 그 값과 정확히
    //    일치할 때만 해소된다(불일치 = 비활성 fail-close). ──
    int32_t      req_hash_layer_count;   // deepseek4.hash_layer_count
    int32_t      req_n_layer_exec;
    int32_t      req_n_expert;
    int32_t      req_n_expert_used;
    int32_t      req_n_embd;
    int32_t      req_n_expert_shared;
    int32_t      req_hc_mult;
    int32_t      req_expert_weights_norm;  // 0/1
    float        req_expert_weights_scale; // < 0 = 미검사
    int32_t      req_n_expert_groups;      // -1 = 미검사(공통 {0,1} guard 만 적용)
    // 텐서 파티션 배타 검사(tid2eid = [0,hash) · exp_probs_b = [hash,n) · 상호 배타)를 켜는 행인가.
    bool         partition_tid2eid;
    // ── ★router dtype allowlist(SPEC_FAMILY_SIGNAL §1.6 — **역할×계열**) ──────────────────
    //   true = 이 계열의 **router 항목에 한해** {F32, BF16} 수용(내부에서 canonical F32 로 승격).
    //   norm/bias/tid2eid 와 이 값이 false 인 계열은 종전 그대로다(F32/I32) — 전역 완화 금지.
    bool         router_bf16_ok;
};

// ★미검사 센티널(가독성 — 표에서 "이 계열은 이 축을 잠그지 않는다"를 한눈에 보이게 한다).
constexpr int32_t PB_REQ_ANY   = -1;
constexpr float   PB_REQF_ANY  = -1.0f;

const PbAdapterSpec PB_ADAPTERS[] = {
    { GGML_MOE_PHASEB_ADAPTER_QWEN35MOE,  "qwen35moe_v1",     "qwen35moe", 0,
      GGML_MOE_PHASEB_NORM_ATTN_POST, GGML_MOE_PHASEB_SEL_TOPK_RAW,
      GGML_MOE_PHASEB_BIAS_NONE,        0, true,  /*output_crop_before_last_moe=*/false,
      /*first_target_layer=*/1,
      PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY,
      PB_REQ_ANY, PB_REQF_ANY, PB_REQ_ANY, /*partition_tid2eid=*/false,
      /*router_bf16_ok=*/false },
    // ★3단계 등재(1차 소스 직독 — `src/models/openai-moe.cpp:127-131` router 입력 = attn_post_norm
    //   RMSNorm · `src/llama-graph.cpp:1840-1841` logits += ffn_gate_inp_b **후** top_k · 동 파일
    //   1853-1856 SOFTMAX_WEIGHT = probs 그대로 → 선택식 = TopK(Wx + gate_b). openai-moe.cpp 는
    //   LLM_KV_EXPERT_GATING_FUNC 를 읽지 않으므로 hparams 기본값 0(NONE)·n_expert_groups 기본 0).
    //   ★output_crop_before_last_moe=true — 마지막 MoE 층 **선행** 출력 축소 계열(FROZEN_PB_OBSV_V2 §1).
    { GGML_MOE_PHASEB_ADAPTER_OPENAI_MOE, "openai_moe_v1",    "gpt-oss",   0,
      GGML_MOE_PHASEB_NORM_ATTN_POST, GGML_MOE_PHASEB_SEL_TOPK_BIAS,
      GGML_MOE_PHASEB_BIAS_GATE_INP_B,  0, true,  /*output_crop_before_last_moe=*/true,
      /*first_target_layer=*/1,
      PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY,
      PB_REQ_ANY, PB_REQF_ANY, PB_REQ_ANY, /*partition_tid2eid=*/false,
      /*router_bf16_ok=*/false },
    { GGML_MOE_PHASEB_ADAPTER_DEEPSEEK2,  "deepseek2_k26_v1", "deepseek2", 2,
      GGML_MOE_PHASEB_NORM_FFN,       GGML_MOE_PHASEB_SEL_TOPK_SIGMOID_BIAS,
      GGML_MOE_PHASEB_BIAS_EXP_PROBS_B, 1, false, /*output_crop_before_last_moe=*/false,
      /*first_target_layer=*/1,
      PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY, PB_REQ_ANY,
      PB_REQ_ANY, PB_REQF_ANY, PB_REQ_ANY, /*partition_tid2eid=*/false,
      /*router_bf16_ok=*/false },
    // ★DSV4 등재(1차 소스 직독 — `src/models/deepseek4.cpp`):
    //   · :1136 신호 지점 = `build_norm(hc_ffn_pre, ffn_norm)` → expert MMID 입력(norm=ffn_norm)
    //   · :1142~1145 hash 층(il < hash_layer_count)은 tid2eid 정적 룩업 + exp_probs_b **우회**
    //     → learned target 은 `first_target_layer = 3`(= hash_layer_count) 이상뿐이다
    //   · :51~52 gating_func = SQRT_SOFTPLUS 강제(로더가 다른 값이면 예외)
    //   · llama-graph.cpp:1858/:1868/:1912 → 선택식 = TopK(sqrt(softplus(Wx)) + exp_probs_b)
    //   · :132~135 텐서 파티션 — tid2eid = [0,hash) · exp_probs_b = [hash,n_layer) **상호 배타**
    //   · :1181 inp_out_ids crop 이 43개 MoE 실행 **뒤** → output_crop_before_last_moe=false
    { GGML_MOE_PHASEB_ADAPTER_DEEPSEEK4,  "deepseek4_v1",     "deepseek4", 4,
      GGML_MOE_PHASEB_NORM_FFN,       GGML_MOE_PHASEB_SEL_TOPK_SQRT_SOFTPLUS_BIAS,
      GGML_MOE_PHASEB_BIAS_EXP_PROBS_B, 0, true,  /*output_crop_before_last_moe=*/false,
      /*first_target_layer=*/3,
      /*req_hash_layer_count=*/3, /*req_n_layer_exec=*/43, /*req_n_expert=*/256,
      /*req_n_expert_used=*/6,    /*req_n_embd=*/4096,     /*req_n_expert_shared=*/1,
      /*req_hc_mult=*/4,          /*req_expert_weights_norm=*/1, /*req_expert_weights_scale=*/1.5f,
      /*req_n_expert_groups=*/0,  /*partition_tid2eid=*/true,
      // ★router {F32, BF16}(SPEC_FAMILY_SIGNAL §1.6) — 업스트림 DSV4 변환기가 소스 dtype 을
      //   보존해 실물 router 43/43 이 BF16 이다(conversion/deepseek.py:533-539·:761-769).
      //   BF16→F32 는 표현상 무손실 승격(BF16 = F32 상위 16비트)이라 GEMV·wire 는 불변이다.
      /*router_bf16_ok=*/true },
};

// ★닫힌 매핑(arch → 표 행)의 **단일 출처**. 해소(seal)와 인벤토리 주사(note_family)가 같은
//   함수를 쓰게 해 "두 표면이 다른 행을 본다"는 표류를 구조적으로 막는다. 없으면 nullptr.
const PbAdapterSpec * pb_find_adapter_spec(const std::string & arch) {
    for (const auto & a : PB_ADAPTERS) if (arch == a.arch) return &a;
    return nullptr;
}

void pb_adapter_clear() {
    g_pb.ad_id           = GGML_MOE_PHASEB_ADAPTER_NONE;
    g_pb.ad_name         = "";
    g_pb.ad_norm_kind    = GGML_MOE_PHASEB_NORM_NONE;
    g_pb.ad_selection_fn = GGML_MOE_PHASEB_SEL_NONE;
    g_pb.ad_bias_kind    = GGML_MOE_PHASEB_BIAS_NONE;
    g_pb.ad_out_crop     = false;   // ★미해소 = 신호 비활성(자동 false 상속이 아니다 — E6)
    g_pb.ad_first_target = 1;       // ★중립값(마스크 미선언 = 종전 "target 1..n-1")
    g_pb.ad_router_bf16_ok = false; // ★미해소 = BF16 불허(§1.6 — 자동 상속 없음)
    g_pb.capture_width   = 0;
    // ★해소 산출인 norm exact-zero inventory 도 함께 초기화한다 — adapter 미해소 =
    //   **미측정**(echo null)이지 "zero 0건"이 아니다.
    g_pb.nv_measured     = false;
    g_pb.nv_scanned      = 0;
    g_pb.nv_zero_elements= 0;
    g_pb.nv_zero.clear();
}

// norm 텐서 종류 → 이름(§13.4 echo·로그의 **단일 출처**).
const char * pb_norm_kind_name(int32_t kind) {
    switch (kind) {
        case GGML_MOE_PHASEB_NORM_ATTN_POST: return "attn_post_norm";
        case GGML_MOE_PHASEB_NORM_FFN:       return "ffn_norm";
        default:                             return "";
    }
}

// ★tensor 항목 1건 판정(§1 "F32/형상/finite" + 이름 완전일치). 반환 nullptr = valid,
//   그 외 = 무효 사유 문자열. **presence 판정(포인터 non-null)은 호출자 몫**이다.
//   형상은 ne[0..3] **전 차원**을 봉인한다(부분 봉인은 3·4차원 오염을 통과시킨다).
//   ★`zeros`(non-null) = exact-zero 좌표 수집 출력. **별도 패스를 만들지 않고** 아래 finite
//   전량 순회에 합류한다(FROZEN_NORM_INVENTORY N8). 술어는 엔진 보정 루프와 **동일한**
//   `== 0.0f` 정확 비교다(epsilon 금지 — 다른 술어를 쓰면 인벤토리의 예측력이 깨진다.
//   `-0.0f == 0.0f` 는 IEEE 754 에서 참이므로 부호 있는 0 도 zero 로 계상된다).
//   ★`bf16_ok`(SPEC_FAMILY_SIGNAL §1.6) = **역할×계열 allowlist**의 이 항목분이다. true 인
//     호출(= deepseek4_v1 의 router 항목)만 BF16 을 추가로 받고, 그 경우 finite/zero 판정은
//     **청크 단위 F32 승격**으로 수행한다(영구 사본 비보존 — 승격본은 `scratch` 에만 산다).
//     기본값 false 라 기존 호출부(norm/bias·타 계열)는 종전 "F32 아니면 type" 그대로다.
const char * pb_inv_check(const ggml_tensor * t, const std::string & want_name,
                          int64_t e0, int64_t e1, std::vector<float> & scratch,
                          std::vector<uint32_t> * zeros = nullptr, bool bf16_ok = false) {
    if (want_name != ggml_get_name(t))                    return "name";
    const bool is_bf16 = (t->type == GGML_TYPE_BF16);
    if (t->type != GGML_TYPE_F32 && !(bf16_ok && is_bf16)) return "type";
    if (t->ne[0] != e0 || t->ne[1] != e1 ||
        t->ne[2] != 1  || t->ne[3] != 1)                  return "shape";
    if (!ggml_is_contiguous(t))                           return "noncontig";
    if (t->data == nullptr)                               return "unallocated";
    const int64_t n = e0 * e1;
    // ★BF16 항목(§1.6 수용 규칙 ①) — host/비-host **두 버퍼 경로 모두** 같은 청크 승격으로 돈다.
    //   승격 뒤 술어는 F32 와 완전히 동일하다(isfinite · `== 0.0f` exact 비교).
    if (is_bf16) {
        const int64_t CH = 1 << 20;   // 원소 수 청크(승격본 4MiB — 상주 상한을 구조로 못박는다)
        const int64_t m0 = std::min<int64_t>(CH, n);
        scratch.resize((size_t)m0);
        const bool host = (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer));
        std::vector<ggml_bf16_t> raw;             // 비-host 원본 staging(함수 종료 시 소멸)
        if (!host) raw.resize((size_t)m0);
        for (int64_t off = 0; off < n; off += CH) {
            const int64_t m = std::min<int64_t>(CH, n - off);
            const ggml_bf16_t * src;
            if (host) {
                src = (const ggml_bf16_t *)t->data + off;
            } else {
                ggml_backend_tensor_get(t, raw.data(), (size_t)off * sizeof(ggml_bf16_t),
                                        (size_t)m * sizeof(ggml_bf16_t));
                src = raw.data();
            }
            ggml_bf16_to_fp32_row(src, scratch.data(), m);
            for (int64_t i = 0; i < m; i++) {
                if (!std::isfinite(scratch[(size_t)i]))  return "nonfinite";
                if (zeros && scratch[(size_t)i] == 0.0f) zeros->push_back((uint32_t)(off + i));
            }
        }
        return nullptr;
    }
    // ★finite 는 **seal/inventory 시점**에 전량 검사한다(구: worker 도달 후 표본별 skip뿐 —
    //   그건 신호를 조용히 갉아먹을 뿐 fail-close 가 아니었다).
    if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
        const float * p = (const float *)t->data;
        for (int64_t i = 0; i < n; i++) {
            if (!std::isfinite(p[i]))          return "nonfinite";
            if (zeros && p[i] == 0.0f)         zeros->push_back((uint32_t)i);
        }
        return nullptr;
    }
    const int64_t CH = 1 << 20;   // 4MiB 청크(비-host 버퍼 — 상주 메모리 상한을 구조로 못박는다)
    scratch.resize((size_t)std::min<int64_t>(CH, n));
    for (int64_t off = 0; off < n; off += CH) {
        const int64_t m = std::min<int64_t>(CH, n - off);
        ggml_backend_tensor_get(t, scratch.data(), (size_t)off * sizeof(float), (size_t)m * sizeof(float));
        for (int64_t i = 0; i < m; i++) {
            if (!std::isfinite(scratch[(size_t)i]))  return "nonfinite";
            if (zeros && scratch[(size_t)i] == 0.0f) zeros->push_back((uint32_t)(off + i));
        }
    }
    return nullptr;
}

// 반환 = 이 항목의 층별 상태(PbState::INV_ABSENT/INV_VALID/INV_INVALID). 기존 호출부는 반환을
//   무시하며 집계(s) 갱신 의미는 전건 불변이다 — 파티션 축이 층 정보를 얻기 위한 추가 산출이다.
uint8_t pb_inv_accum(PbState::PbInvSum & s, const ggml_tensor * t, const std::string & want,
                     int64_t e0, int64_t e1, std::vector<float> & scratch,
                     std::vector<uint32_t> * zeros = nullptr, bool bf16_ok = false) {
    if (!t) return PbState::INV_ABSENT;   // ABSENT — presence 자체가 없다(유일한 "부재")
    s.present++;
    // ★zero 좌표는 **valid 일 때만** 확정한다 — invalid 항목의 부분 주사 결과를 인벤토리로
    //   승격시키지 않는다(중간에 nonfinite 로 끊긴 순회는 전량 근거가 아니다).
    std::vector<uint32_t> z;
    const char * why = pb_inv_check(t, want, e0, e1, scratch, zeros ? &z : nullptr, bf16_ok);
    if (why) { s.invalid++; if (s.why.empty()) s.why = why; return PbState::INV_INVALID; }
    s.valid++;
    if (zeros) *zeros = std::move(z);
    return PbState::INV_VALID;
}

// ★tid2eid(토큰ID → expert id 정적 룩업 표) 1건 판정 — DSV4 텐서 파티션 축 전용.
//   신호 경로는 이 표를 **읽지 않는다**(§1.5). 여기서 보는 것은 "선언된 그 텐서가 맞는가"이며
//   술어는 F32 항목과 **같은 엄밀도**로 맞춘다:
//     이름 완전일치 · dtype(I32) · 형상 ne[0]=n_expert_used · ne[2]=ne[3]=1 · contiguous ·
//     할당됨 · 값 전량이 [0, n_expert) — 마지막 항은 F32 의 `finite 전량` 에 대응하는
//     **정수 축 술어**다(표의 값이 expert id 공간을 벗어나면 그 모델은 우리가 검증한 형상이
//     아니다). ne[1](=n_vocab)은 신호 축 상수가 아니고 descriptor 에 실려 있지 않으므로
//     "양수"까지만 봉인한다(없는 값을 합성해 대조하지 않는다).
const char * pb_inv_check_tid2eid(const ggml_tensor * t, const std::string & want_name,
                                  int64_t e0, int32_t n_expert, std::vector<int32_t> & scratch) {
    if (want_name != ggml_get_name(t))                    return "name";
    if (t->type != GGML_TYPE_I32)                         return "type";
    if (t->ne[0] != e0 || t->ne[1] <= 0 ||
        t->ne[2] != 1  || t->ne[3] != 1)                  return "shape";
    if (!ggml_is_contiguous(t))                           return "noncontig";
    if (t->data == nullptr)                               return "unallocated";
    const int64_t n = t->ne[0] * t->ne[1];
    if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
        const int32_t * p = (const int32_t *)t->data;
        for (int64_t i = 0; i < n; i++)
            if (p[i] < 0 || p[i] >= n_expert)             return "range";
        return nullptr;
    }
    const int64_t CH = 1 << 20;   // 4MiB 청크(비-host 버퍼 — 상주 메모리 상한을 구조로 못박는다)
    scratch.resize((size_t)std::min<int64_t>(CH, n));
    for (int64_t off = 0; off < n; off += CH) {
        const int64_t m = std::min<int64_t>(CH, n - off);
        ggml_backend_tensor_get(t, scratch.data(), (size_t)off * sizeof(int32_t), (size_t)m * sizeof(int32_t));
        for (int64_t i = 0; i < m; i++)
            if (scratch[(size_t)i] < 0 || scratch[(size_t)i] >= n_expert) return "range";
    }
    return nullptr;
}

// ★denominator/source norm 의 원소별 exact-zero inventory 확정(FROZEN_NORM_INVENTORY §3·§4).
//   **adapter 해소 성공 직후 1회** 호출한다(대상 텐서가 norm_kind 로 정해져야 계상이 성립).
//   · source 층 = `moe_layer_ids[:-1]` — 마지막 관리 MoE 층은 affected target 이 없어
//     coverage-limiting zero source 가 아니다 → `source_layers_scanned` 에도 넣지 않는다.
//   · affected_target = `moe_layer_ids[mlpos+1]` (**규범식** · 산술 `source+1` 금지 —
//     leading-dense 혼재 형상에서는 물리 id 가 연속하지 않는다).
//   · zero 좌표 = 오름차순 0-based u32 **전량**(상위 N 절단 없음) · digest = §3.3.
//   · 스캔 대상 source 층이 인벤토리 근거 범위(0..n_layer_exec-1) 밖이면 **미측정**이다
//     (실행 범위 밖 관리 층이 섞인 형상 — 0 을 합성하지 않고 fail-close 한다).
void pb_build_norm_inventory(int32_t n_layer_exec) {
    g_pb.nv_measured     = false;
    g_pb.nv_scanned      = 0;
    g_pb.nv_zero_elements= 0;
    g_pb.nv_zero.clear();

    const std::vector<std::vector<uint32_t>> & nz =
        (g_pb.ad_norm_kind == GGML_MOE_PHASEB_NORM_FFN) ? g_pb.nz_ffn : g_pb.nz_attn;
    const std::vector<int32_t> & ml = g_pb.moe_layers;
    if (ml.size() < 2) return;            // source 층이 없다 = 측정 자체가 성립하지 않는다

    const int32_t n_src = (int32_t)ml.size() - 1;      // = |moe_layer_ids[:-1]|
    int32_t scanned = 0;
    for (int32_t mlpos = 0; mlpos < n_src; mlpos++) {
        const int32_t src = ml[(size_t)mlpos];
        // ★층 마스크(DSV4 §1.5 "전 표면 동일 마스크") — 정적 제외 target 을 가진 source 는
        //   coverage-limiting 이 될 수 없다(그 target 은 애초에 예측하지 않는다). 스캔 모집단
        //   에서 빼지 않으면 그 층의 zero 가 거짓 `coverage_limited` 를 만든다.
        if (ml[(size_t)mlpos + 1] < g_pb.ad_first_target) continue;
        scanned++;
        if (src < 0 || src >= n_layer_exec || (size_t)src >= nz.size()) {
            g_pb.nv_zero.clear();
            g_pb.nv_zero_elements = 0;
            return;                        // 근거 부재 = 미측정(fail-close)
        }
        const std::vector<uint32_t> & idx = nz[(size_t)src];
        if (idx.empty()) continue;
        PbState::PbNormZeroRec r;
        r.source_layer    = src;
        r.affected_target = ml[(size_t)mlpos + 1];
        r.idx             = idx;
        // §3.3 — little-endian `(u16 source_layer)(u32 zero_count)(u32 zero_index...)` 연접의 FNV-1a 64.
        const uint16_t sl = (uint16_t)src;
        const uint32_t zc = (uint32_t)r.idx.size();
        uint64_t h = FNV_INIT;
        h = fnv1a64(h, &sl, sizeof(sl));
        h = fnv1a64(h, &zc, sizeof(zc));
        for (size_t k = 0; k < r.idx.size(); k++) h = fnv1a64(h, &r.idx[k], sizeof(uint32_t));
        r.digest = h;
        g_pb.nv_zero_elements += (uint64_t)zc;
        g_pb.nv_zero.push_back(std::move(r));
    }
    g_pb.nv_scanned  = scanned;   // ★마스크 적용 모집단(기존 계열 = n_src 그대로)
    g_pb.nv_measured = true;
    // seal 로그 1줄(§4 표) — 사람이 즉시 읽는 요약. echo 권위는 metrics 쪽이다.
    pb_log("norm inventory: tensor=%s scanned_source_layers=%d zero_source_layers=%d "
           "zero_elements=%llu affected_targets=%d",
           pb_norm_kind_name(g_pb.ad_norm_kind), (int)g_pb.nv_scanned,
           (int)g_pb.nv_zero.size(), (unsigned long long)g_pb.nv_zero_elements,
           (int)g_pb.nv_zero.size());
}

// seal 시 1회 해소. 순서 = ①닫힌 매핑(arch→행) ②등재 여부 ③hparams guard ④moe_layers 구조
// ⑤tensor inventory ⑥seal 대조(선언 descriptor ↔ 카탈로그 seal 표류). 어긋나면 `none`.
void pb_resolve_adapter(int32_t seal_n_expert, int32_t seal_n_expert_used,
                        int32_t seal_n_layer_exec, int32_t seal_n_embd) {
    pb_adapter_clear();
    if (!g_pb.fam_declared) { g_pb.ad_reason = "descriptor_absent"; return; }

    const PbAdapterSpec * sp = pb_find_adapter_spec(g_pb.fam_arch);
    if (!sp)             { g_pb.ad_reason = "arch_unmapped";        return; }
    if (!sp->registered) { g_pb.ad_reason = "adapter_unregistered"; return; }

    // ③ hparams guard(§1 검증 잠금).
    if (g_pb.fam_gating != sp->gating_func)   { g_pb.ad_reason = "gating_func_mismatch";      return; }
    // ★그룹 top-k 미구현 — 지원값은 정확히 {0(KV 부재), 1} 뿐이다(음수·2 이상 전부 거부).
    if (!(g_pb.fam_groups == 0 || g_pb.fam_groups == 1))
                                              { g_pb.ad_reason = "expert_groups_unsupported"; return; }
    if (g_pb.fam_n_expert_used < 1 || g_pb.fam_n_expert_used > GGML_MOE_PHASEB_TOPK)
                                              { g_pb.ad_reason = "n_expert_used_unsupported"; return; }
    if (g_pb.fam_n_embd <= 0 || g_pb.fam_n_embd > GGML_MOE_PHASEB_CAPTURE_DIM_MAX)
                                              { g_pb.ad_reason = "n_embd_unsupported";        return; }
    if (g_pb.fam_n_expert <= 0)               { g_pb.ad_reason = "n_expert_unsupported";      return; }
    if (g_pb.fam_n_layer_exec < 2)            { g_pb.ad_reason = "n_layer_unsupported";       return; }
    // ★③′ 계열 고정 hparams(DSV4_ADAPTER_DESIGN §1.5 확정 목록) — **-1 = 미검사**라 기존 3행은
    //   이 블록을 전부 통과한다(거동 불변). 값이 박힌 행은 정확히 일치할 때만 해소된다.
    if (sp->req_hash_layer_count >= 0 && g_pb.fam_hash_layers != sp->req_hash_layer_count)
                                              { g_pb.ad_reason = "hash_layer_count_mismatch";  return; }
    if (sp->req_n_layer_exec >= 0 && g_pb.fam_n_layer_exec != sp->req_n_layer_exec)
                                              { g_pb.ad_reason = "n_layer_exec_mismatch";      return; }
    if (sp->req_n_expert >= 0 && g_pb.fam_n_expert != sp->req_n_expert)
                                              { g_pb.ad_reason = "n_expert_mismatch";          return; }
    if (sp->req_n_expert_used >= 0 && g_pb.fam_n_expert_used != sp->req_n_expert_used)
                                              { g_pb.ad_reason = "n_expert_used_mismatch";     return; }
    if (sp->req_n_embd >= 0 && g_pb.fam_n_embd != sp->req_n_embd)
                                              { g_pb.ad_reason = "n_embd_mismatch";            return; }
    if (sp->req_n_expert_shared >= 0 && g_pb.fam_n_expert_shared != sp->req_n_expert_shared)
                                              { g_pb.ad_reason = "n_expert_shared_mismatch";   return; }
    if (sp->req_hc_mult >= 0 && g_pb.fam_hc_mult != sp->req_hc_mult)
                                              { g_pb.ad_reason = "hc_mult_mismatch";           return; }
    if (sp->req_expert_weights_norm >= 0 && g_pb.fam_weights_norm != sp->req_expert_weights_norm)
                                              { g_pb.ad_reason = "expert_weights_norm_mismatch"; return; }
    if (sp->req_expert_weights_scale >= 0.0f && g_pb.fam_weights_scale != sp->req_expert_weights_scale)
                                              { g_pb.ad_reason = "expert_weights_scale_mismatch"; return; }
    // ★group routing 활성 시 §1.5 선택 산식이 불완전해진다 — 요구값이 박힌 행은 정확히 그 값.
    if (sp->req_n_expert_groups >= 0 && g_pb.fam_groups != sp->req_n_expert_groups)
                                              { g_pb.ad_reason = "expert_groups_unsupported";  return; }
    // ★층 마스크 성립 전제 — first_target_layer 가 실행 층 범위 안이어야 target 이 존재한다.
    if (sp->first_target_layer < 1 || sp->first_target_layer >= g_pb.fam_n_layer_exec)
                                              { g_pb.ad_reason = "layer_mask_unsupported";     return; }

    // ④ ★실제 moe_layers[] 구조(§1) — 오름차순·중복 없음 · [0]=leading-dense ·
    //    [leading_dense, n_layer_exec) **전 층 연속**(capture 훅의 target=l+1 연속성 계약과 동일).
    {
        const std::vector<int32_t> & ml = g_pb.fam_moe_layers;
        if (ml.empty())                { g_pb.ad_reason = "moe_layers_absent";    return; }
        if (ml.front() < 0)            { g_pb.ad_reason = "moe_layers_unordered"; return; }
        for (size_t i = 1; i < ml.size(); i++)
            if (ml[i] <= ml[i - 1])    { g_pb.ad_reason = "moe_layers_unordered"; return; }
        if (ml.front() != sp->leading_dense)
                                       { g_pb.ad_reason = "leading_dense_mismatch"; return; }
        size_t k = 0;
        for (int32_t l = sp->leading_dense; l < g_pb.fam_n_layer_exec; l++) {
            while (k < ml.size() && ml[k] < l) k++;
            if (k >= ml.size() || ml[k] != l) { g_pb.ad_reason = "moe_layers_gap"; return; }
        }
    }

    // ⑤ tensor inventory — 요구 norm 전층(0..n-1)·router 전층(1..n-1) + bias 프로필 정확 일치.
    //    ★present∧¬valid(=invalid)는 **부재가 아니라 결함**이다 — 사유를 실어 fail-close.
    const PbState::PbInvSum & nrm = (sp->norm_kind == GGML_MOE_PHASEB_NORM_FFN)
                                    ? g_pb.inv_norm_ffn : g_pb.inv_norm_attn;
    if (nrm.invalid > 0)  { g_pb.ad_reason = "inventory_invalid_norm_" + nrm.why;   return; }
    if (nrm.valid != g_pb.fam_n_layer_exec)      { g_pb.ad_reason = "norm_inventory_mismatch";   return; }
    if (g_pb.inv_router.invalid > 0)
                          { g_pb.ad_reason = "inventory_invalid_router_" + g_pb.inv_router.why; return; }
    if (g_pb.inv_router.valid != g_pb.fam_n_layer_exec - 1)
                                                 { g_pb.ad_reason = "router_inventory_mismatch"; return; }
    // ★요구 bias 가 NONE 이면 **presence 자체가 0**이어야 한다 — 비-null·오이름 bias 가
    //   "이름이 안 맞으니 없는 것"으로 흡수되던 구멍을 여기서 닫는다.
    {
        const PbState::PbInvSum & gi = g_pb.inv_bias_gi;
        const PbState::PbInvSum & ep = g_pb.inv_bias_ep;
        const PbState::PbInvSum & want = (sp->bias_kind == GGML_MOE_PHASEB_BIAS_GATE_INP_B) ? gi : ep;
        const PbState::PbInvSum & othr = (sp->bias_kind == GGML_MOE_PHASEB_BIAS_GATE_INP_B) ? ep : gi;
        if (sp->bias_kind == GGML_MOE_PHASEB_BIAS_NONE) {
            if (gi.present != 0 || ep.present != 0) { g_pb.ad_reason = "bias_profile_mismatch"; return; }
        } else {
            if (othr.present != 0)   { g_pb.ad_reason = "bias_profile_mismatch"; return; }
            if (want.invalid > 0)    { g_pb.ad_reason = "inventory_invalid_bias_" + want.why; return; }
            // ★층 마스크(§1.5) — 요구 bias 층수 = **learned target 수**다(정적 제외 target 은
            //   bias 가 존재하지 않는 것이 계약이다 — deepseek4 hash 층). 기존 계열은
            //   first_target_layer=1 이라 종전 `n_layer_exec-1` 과 완전히 같은 식이다.
            if (want.valid != g_pb.fam_n_layer_exec - sp->first_target_layer)
                                     { g_pb.ad_reason = "bias_profile_mismatch"; return; }
        }
    }
    // ⑤′ ★텐서 파티션 배타(DSV4 §1.5) — tid2eid = [0, hash) · exp_probs_b = [hash, n_layer_exec)
    //    이며 **층별로 정확히 하나만** 존재해야 한다. 집계(present/valid 수)로는 "개수만 맞고
    //    위치가 뒤바뀐" 형상이 통과하므로 층별 상태 배열로 판정한다.
    if (sp->partition_tid2eid) {
        const int32_t n  = g_pb.fam_n_layer_exec;
        const int32_t hl = sp->req_hash_layer_count >= 0 ? sp->req_hash_layer_count : g_pb.fam_hash_layers;
        if ((int32_t)g_pb.inv_tid_state.size() != n || (int32_t)g_pb.inv_ep_state.size() != n) {
            g_pb.ad_reason = "partition_inventory_absent"; return;
        }
        for (int32_t il = 0; il < n; il++) {
            const bool want_tid = (il < hl);
            const uint8_t ts = g_pb.inv_tid_state[(size_t)il];
            const uint8_t es = g_pb.inv_ep_state[(size_t)il];
            if (want_tid) {
                if (ts == PbState::INV_INVALID) { g_pb.ad_reason = "inventory_invalid_tid2eid_" + g_pb.inv_tid_why; return; }
                if (ts != PbState::INV_VALID)   { g_pb.ad_reason = "tid2eid_partition_mismatch"; return; }
                if (es != PbState::INV_ABSENT)  { g_pb.ad_reason = "exp_probs_b_partition_mismatch"; return; }
            } else {
                if (ts != PbState::INV_ABSENT)  { g_pb.ad_reason = "tid2eid_partition_mismatch"; return; }
                if (es == PbState::INV_INVALID) { g_pb.ad_reason = "inventory_invalid_bias_" + g_pb.inv_ep_why; return; }
                if (es != PbState::INV_VALID)   { g_pb.ad_reason = "exp_probs_b_partition_mismatch"; return; }
            }
        }
    }

    // ⑥ 선언 descriptor ↔ seal 상수 대조(두 생산자 간 표류 = fail-close).
    if (g_pb.fam_n_expert      != seal_n_expert      ||
        g_pb.fam_n_expert_used != seal_n_expert_used ||
        g_pb.fam_n_layer_exec  != seal_n_layer_exec  ||
        g_pb.fam_n_embd        != seal_n_embd)       { g_pb.ad_reason = "seal_descriptor_drift"; return; }
    // ★moe_layers 표류 — 실행 범위(< n_layer_exec) 집합이 두 생산자에서 동일해야 한다.
    //   (실행 범위 밖 = NextN 등 scope 필터 대상이라 비교축에서 제외 — 오탐 방지.)
    {
        std::vector<int32_t> a, b;
        for (int32_t x : g_pb.fam_moe_layers) if (x < seal_n_layer_exec) a.push_back(x);
        for (int32_t x : g_pb.moe_layers)     if (x < seal_n_layer_exec) b.push_back(x);
        if (a != b) { g_pb.ad_reason = "seal_moe_layers_drift"; return; }
    }

    g_pb.ad_id           = sp->id;
    g_pb.ad_name         = sp->name;
    g_pb.ad_norm_kind    = sp->norm_kind;
    g_pb.ad_selection_fn = sp->selection_fn;
    g_pb.ad_bias_kind    = sp->bias_kind;
    g_pb.ad_out_crop     = sp->output_crop_before_last_moe;   // ★표의 명시값만(재추론 금지 — E6)
    g_pb.ad_first_target = sp->first_target_layer;            // ★층 마스크 = 표의 명시값(§1.5)
    g_pb.ad_router_bf16_ok = sp->router_bf16_ok;              // ★router dtype allowlist(§1.6)
    g_pb.capture_width   = g_pb.fam_n_embd;   // ★capture 폭 = descriptor n_embd(guard ≤8192)
    g_pb.ad_reason       = "";
    // ⑦ ★denominator norm exact-zero inventory 확정 + seal 로그(FROZEN_NORM_INVENTORY §4).
    //    **zero 존재는 해소를 탈락시키지 않는다**(즉 FAIL 아님) — 위 해소 결과를 바꾸지 않고
    //    측정값만 싣는다. 판정(coverage_limited/HOLD)은 판독기 축이다(§5).
    pb_build_norm_inventory(seal_n_layer_exec);
}

} // namespace

// ===========================================================================
// §3. 공개 API — env/mode
// ===========================================================================

extern "C" {

int32_t ggml_moe_phaseb_mode(void)          { pb_resolve_env(); return g_pb.mode; }
bool    ggml_moe_phaseb_enabled(void)       { return ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_OFF; }
const char * ggml_moe_phaseb_mode_name(void) {
    switch (ggml_moe_phaseb_mode()) {
        case GGML_MOE_PHASEB_TRACE:   return "trace";
        case GGML_MOE_PHASEB_EQUIV:   return "equiv";
        case GGML_MOE_PHASEB_OBSERVE: return "observe";
        default:                      return "off";
    }
}
uint64_t     ggml_moe_phaseb_boot_id(void)    { pb_resolve_env(); return g_pb.boot_id; }
const char * ggml_moe_phaseb_out_prefix(void) { pb_resolve_env(); return g_pb.out_prefix.c_str(); }
uint32_t     ggml_moe_phaseb_buf_mb(void)     { pb_resolve_env(); return g_pb.buf_mb; }
uint64_t     ggml_moe_phaseb_now_qpc(void)    { LARGE_INTEGER c{}; QueryPerformanceCounter(&c); return (uint64_t)c.QuadPart; }

// ★프리페치 활성 조건(§1): K·N **둘 다 설정·둘 다 유효**일 때만 ON. 하나라도 미설정/무효
//   (범위 밖·비정수)면 OFF + off_reason echo. 기동 1회 해소·런 중 적응 금지.
void ggml_moe_phaseb_resolve_prefetch(int32_t qd_effective) {
    pb_resolve_env();
    if (g_pb.pf_resolved) return;
    g_pb.pf_resolved = true;
    g_pb.pf_on = false; g_pb.pf_k = 0; g_pb.pf_n = 0;

    pb_refresh_all_off();
    const std::string ks = pb_getenv("MOE_DIRECT_PREFETCH_K");
    const std::string ns = pb_getenv("MOE_DIRECT_PREFETCH_N");
    if (ks.empty() || ns.empty()) { g_pb.pf_off_reason = "unset"; return; }

    char * e = nullptr;
    const long kv = strtol(ks.c_str(), &e, 10);
    if (!e || *e != '\0')          { g_pb.pf_off_reason = "invalid_k"; return; }
    if (kv < 1 || kv > 16)         { g_pb.pf_off_reason = "invalid_range"; return; }
    e = nullptr;
    const long nv = strtol(ns.c_str(), &e, 10);
    if (!e || *e != '\0')          { g_pb.pf_off_reason = "invalid_n"; return; }
    if (nv < 1 || nv >= (long)qd_effective) { g_pb.pf_off_reason = "invalid_range"; return; }

    g_pb.pf_k = (int32_t)kv;
    g_pb.pf_n = (int32_t)nv;
    g_pb.pf_on = true;
    g_pb.pf_off_reason = "";
    pb_refresh_all_off();
}

// ★계열 신호 descriptor 선언(SPEC_FAMILY_SIGNAL §1) — seal **이전**에 1회. 재선언=덮어쓰기,
//   nullptr=선언 취소. 값 복사만 하고 해소는 note_seal 에서 한다(seal 상수 대조가 필요하므로).
void ggml_moe_phaseb_note_family(const struct ggml_moe_phaseb_family_in * in) {
    // 전건 초기화(재선언 = 덮어쓰기 · nullptr = 선언 취소).
    g_pb.fam_declared = false;
    g_pb.fam_arch.clear();
    g_pb.fam_moe_layers.clear();
    g_pb.fam_gating = g_pb.fam_groups = 0;
    g_pb.fam_n_embd = g_pb.fam_n_expert = g_pb.fam_n_expert_used = 0;
    g_pb.fam_n_layer_exec = 0;
    g_pb.fam_hash_layers = g_pb.fam_n_expert_shared = g_pb.fam_hc_mult = g_pb.fam_weights_norm = 0;
    g_pb.fam_weights_scale = 0.0f;
    g_pb.inv_norm_attn = g_pb.inv_norm_ffn = g_pb.inv_router = PbState::PbInvSum{};
    g_pb.inv_bias_gi   = g_pb.inv_bias_ep  = PbState::PbInvSum{};
    g_pb.inv_tid_state.clear();
    g_pb.inv_ep_state.clear();
    g_pb.inv_tid_why.clear();
    g_pb.inv_ep_why.clear();
    g_pb.nz_attn.clear();
    g_pb.nz_ffn.clear();
    pb_adapter_clear();
    g_pb.ad_reason = "descriptor_absent";
    if (!in) return;

    g_pb.fam_declared      = true;
    g_pb.fam_arch          = in->arch ? std::string(in->arch) : std::string();
    g_pb.fam_gating        = in->expert_gating_func;
    g_pb.fam_groups        = in->n_expert_groups;
    g_pb.fam_n_embd        = in->n_embd;
    g_pb.fam_n_expert      = in->n_expert;
    g_pb.fam_n_expert_used = in->n_expert_used;
    g_pb.fam_n_layer_exec  = in->n_layer_exec;
    g_pb.fam_hash_layers   = in->hash_layer_count;
    g_pb.fam_n_expert_shared = in->n_expert_shared;
    g_pb.fam_hc_mult       = in->hc_mult;
    g_pb.fam_weights_norm  = in->expert_weights_norm;
    g_pb.fam_weights_scale = in->expert_weights_scale;
    if (in->moe_layers && in->n_moe_layers > 0)
        g_pb.fam_moe_layers.assign(in->moe_layers, in->moe_layers + in->n_moe_layers);

    if (!in->layers || in->n_layer_exec <= 0) return;   // 인벤토리 미제공 = 전 항목 ABSENT

    // ★판정은 **모듈이** 한다(생산자 계수 위임 금지 — presence/validity 분리의 근거).
    //   ★norm 두 후보의 exact-zero 좌표는 이 **같은 순회**에서 함께 모은다(별도 패스 없음 —
    //   FROZEN_NORM_INVENTORY N8). 계상 대상 층의 선별은 seal 해소가 한다.
    g_pb.nz_attn.assign((size_t)in->n_layer_exec, std::vector<uint32_t>());
    g_pb.nz_ffn .assign((size_t)in->n_layer_exec, std::vector<uint32_t>());
    // ★파티션 축(DSV4 §1.5) — 층별 상태는 **층 0 을 포함한 전 층**을 본다(집계 inv_* 의 il>=1
    //   범위는 손대지 않는다 — 기존 계열의 판정 입력이 바뀌면 안 된다).
    g_pb.inv_tid_state.assign((size_t)in->n_layer_exec, PbState::INV_ABSENT);
    g_pb.inv_ep_state .assign((size_t)in->n_layer_exec, PbState::INV_ABSENT);
    // ★router dtype allowlist(SPEC_FAMILY_SIGNAL §1.6) — **역할×계열**. 해소는 seal 시점이므로
    //   여기서는 해소와 **같은 닫힌 매핑**(arch→표 행)으로 행을 찾아 그 행의 명시값만 읽는다
    //   (전역 완화 금지 · 미매핑 arch = false = 종전 F32 전용). 아래에서 router 항목에만 준다.
    const PbAdapterSpec * sp_dt = pb_find_adapter_spec(g_pb.fam_arch);
    const bool router_bf16_ok = sp_dt && sp_dt->router_bf16_ok;
    std::vector<float>   scratch;
    std::vector<int32_t> scratch_i32;
    for (int32_t il = 0; il < in->n_layer_exec; il++) {
        const ggml_moe_phaseb_layer_in & L = in->layers[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        // ── ★파티션 축 층별 판정 — tid2eid 는 이 축에만 존재하므로 전 층을 여기서 본다.
        if (L.ffn_gate_tid2eid) {
            const char * why = pb_inv_check_tid2eid(L.ffn_gate_tid2eid, b + "ffn_gate_tid2eid.weight",
                                                    in->n_expert_used, in->n_expert, scratch_i32);
            g_pb.inv_tid_state[(size_t)il] = why ? PbState::INV_INVALID : PbState::INV_VALID;
            if (why && g_pb.inv_tid_why.empty()) g_pb.inv_tid_why = why;
        }
        // exp_probs_b 의 층 0 분(집계 범위 밖) — 파티션 축에만 싣는다(집계 inv_bias_ep 불변).
        if (il == 0 && L.ffn_exp_probs_b) {
            PbState::PbInvSum tmp;
            g_pb.inv_ep_state[0] =
                pb_inv_accum(tmp, L.ffn_exp_probs_b, b + "exp_probs_b.bias", in->n_expert, 1, scratch);
            if (g_pb.inv_ep_state[0] == PbState::INV_INVALID && g_pb.inv_ep_why.empty())
                g_pb.inv_ep_why = tmp.why;
        }
        pb_inv_accum(g_pb.inv_norm_attn, L.attn_post_norm, b + "post_attention_norm.weight",
                     in->n_embd, 1, scratch, &g_pb.nz_attn[(size_t)il]);
        pb_inv_accum(g_pb.inv_norm_ffn,  L.ffn_norm,       b + "ffn_norm.weight",
                     in->n_embd, 1, scratch, &g_pb.nz_ffn[(size_t)il]);
        // ★기록(3단계 교차 F7) — §1 문면의 bias "전층 존재 필수" 대비 이 주사는 router 와 동일
        //   범위(il≥1)만 돈다. layer 0 bias 누락은 **로더가 선차단**한다(gpt-oss 는
        //   `src/models/openai-moe.cpp` 가 `ffn_gate_inp.bias` 를 전층 필수 텐서로 create_tensor
        //   하므로 부재 모델은 로드 자체가 실패) · 신호 산식도 target=l+1(≥1) 만 소비한다.
        //   = 위협모델 밖 차이(수정 대상 아님·차이만 명시).
        if (il >= 1) {
            pb_inv_accum(g_pb.inv_router,  L.ffn_gate_inp,    b + "ffn_gate_inp.weight",
                         in->n_embd, in->n_expert, scratch, nullptr, router_bf16_ok);
            pb_inv_accum(g_pb.inv_bias_gi, L.ffn_gate_inp_b,  b + "ffn_gate_inp.bias",
                         in->n_expert, 1, scratch);
            // ★파티션 축 층별 상태는 **이 호출의 반환**으로 얻는다(중복 주사 없음 — 집계와
            //   층별 배열이 같은 1회 판정에서 나오므로 두 축이 어긋날 수 없다).
            g_pb.inv_ep_state[(size_t)il] =
                pb_inv_accum(g_pb.inv_bias_ep, L.ffn_exp_probs_b, b + "exp_probs_b.bias",
                             in->n_expert, 1, scratch);
            if (g_pb.inv_ep_state[(size_t)il] == PbState::INV_INVALID && g_pb.inv_ep_why.empty())
                g_pb.inv_ep_why = g_pb.inv_bias_ep.why;
        }
    }
}

int32_t      ggml_moe_phaseb_adapter_id(void)       { return g_pb.ad_id; }
const char * ggml_moe_phaseb_adapter_name(void)     { return g_pb.ad_name; }
const char * ggml_moe_phaseb_adapter_reason(void)   { return g_pb.ad_reason.c_str(); }
int32_t      ggml_moe_phaseb_norm_tensor_kind(void) { return g_pb.ad_norm_kind; }
int32_t      ggml_moe_phaseb_selection_fn(void)     { return g_pb.ad_selection_fn; }
int32_t      ggml_moe_phaseb_bias_kind(void)        { return g_pb.ad_bias_kind; }
int32_t      ggml_moe_phaseb_capture_width(void)    { return g_pb.capture_width; }
const char * ggml_moe_phaseb_family_arch(void)      { return g_pb.fam_arch.c_str(); }
const char * ggml_moe_phaseb_norm_tensor_name(void) { return pb_norm_kind_name(g_pb.ad_norm_kind); }
const char * ggml_moe_phaseb_selection_fn_name(void) {
    switch (g_pb.ad_selection_fn) {
        case GGML_MOE_PHASEB_SEL_TOPK_RAW:          return "topk_raw";
        case GGML_MOE_PHASEB_SEL_TOPK_BIAS:         return "topk_bias";
        case GGML_MOE_PHASEB_SEL_TOPK_SIGMOID_BIAS: return "topk_sigmoid_bias";
        case GGML_MOE_PHASEB_SEL_TOPK_SQRT_SOFTPLUS_BIAS: return "topk_sqrt_softplus_bias";
        default:                                    return "";
    }
}
const char * ggml_moe_phaseb_bias_snapshot_name(void) {
    switch (g_pb.ad_bias_kind) {
        case GGML_MOE_PHASEB_BIAS_GATE_INP_B:  return "ffn_gate_inp_b[]";
        case GGML_MOE_PHASEB_BIAS_EXP_PROBS_B: return "ffn_exp_probs_b[]";
        default:                               return "";
    }
}
int32_t ggml_moe_phaseb_family_n_expert(void)      { return g_pb.fam_n_expert; }
int32_t ggml_moe_phaseb_family_n_expert_used(void) { return g_pb.fam_n_expert_used; }
int32_t ggml_moe_phaseb_moe_layer_count(void)      { return (int32_t)g_pb.moe_layers.size(); }
int32_t ggml_moe_phaseb_moe_layer_at(int32_t i) {
    return (i >= 0 && i < (int32_t)g_pb.moe_layers.size()) ? g_pb.moe_layers[(size_t)i] : -1;
}
// ★capability echo 의 단일 출처(§7 E6) — 해소된 adapter 행의 명시값 그대로(재추론 없음).
bool ggml_moe_phaseb_output_crop_before_last_moe(void) { return g_pb.ad_out_crop; }

// ★층 마스크 창구(DSV4_ADAPTER_DESIGN §1.5) — 전 표면이 이 **하나**를 본다.
int32_t ggml_moe_phaseb_signal_first_target_layer(void) { return g_pb.ad_first_target; }
bool    ggml_moe_phaseb_signal_layer_mask_active(void)  { return g_pb.ad_first_target > 1; }
bool    ggml_moe_phaseb_signal_target_eligible(int32_t target_layer) {
    return target_layer >= 1 && target_layer >= g_pb.ad_first_target;
}

// ★norm exact-zero inventory 접근자(FROZEN_NORM_INVENTORY §4 — metrics `observe_norm_inventory`
//   exact block 의 **단일 출처**. direct 는 여기 값을 옮겨 적기만 하고 재계산하지 않는다).
//   미측정(measured=false)에서는 전 계수가 0 이지만, 그 0 은 **echo 되어서는 안 된다**
//   (키 부재/`null` 과 "zero 0건"의 구분이 §4 계약 — 호출자가 measured 를 먼저 본다).
bool     ggml_moe_phaseb_norm_inventory_measured(void)      { return g_pb.nv_measured; }
int32_t  ggml_moe_phaseb_norm_inventory_scanned(void)       { return g_pb.nv_measured ? g_pb.nv_scanned : 0; }
int32_t  ggml_moe_phaseb_norm_inventory_zero_layers(void) {
    return g_pb.nv_measured ? (int32_t)g_pb.nv_zero.size() : 0;
}
uint64_t ggml_moe_phaseb_norm_inventory_zero_elements(void) {
    return g_pb.nv_measured ? g_pb.nv_zero_elements : 0;
}
int ggml_moe_phaseb_norm_inventory_at(int32_t i, int32_t * out_source_layer,
                                      int32_t * out_affected_target, uint32_t * out_zero_count,
                                      uint64_t * out_zero_digest) {
    if (!g_pb.nv_measured || i < 0 || i >= (int32_t)g_pb.nv_zero.size()) return -1;
    const PbState::PbNormZeroRec & r = g_pb.nv_zero[(size_t)i];
    if (out_source_layer)    *out_source_layer    = r.source_layer;
    if (out_affected_target) *out_affected_target = r.affected_target;
    if (out_zero_count)      *out_zero_count      = (uint32_t)r.idx.size();
    if (out_zero_digest)     *out_zero_digest     = r.digest;
    return 0;
}
uint32_t ggml_moe_phaseb_norm_inventory_zero_index(int32_t i, uint32_t k) {
    if (!g_pb.nv_measured || i < 0 || i >= (int32_t)g_pb.nv_zero.size()) return GGML_MOE_PHASEB_NA_U32;
    const std::vector<uint32_t> & v = g_pb.nv_zero[(size_t)i].idx;
    return (k < (uint32_t)v.size()) ? v[(size_t)k] : GGML_MOE_PHASEB_NA_U32;
}

bool         ggml_moe_phaseb_signal_observe_on(void) { return g_pb.obs_on; }
// ★observe seal 전제(SPEC_FAMILY_SIGNAL §2.1 · SPEC_PHASEB_WIRE §13) — note_seal 이 이미 판정해
//   둔 사유를 그대로 돌려준다(판정 시점 = 자원 할당 **이전** 이라 fail-close 가 구조로 성립).
int ggml_moe_phaseb_observe_precheck(const char ** out_reason) {
    if (ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_OBSERVE) return 0;
    if (g_pb.obs_on) return 0;
    if (out_reason) {
        *out_reason = g_pb.obs_reason.empty() ? "observe seal precondition not evaluated (note_seal missing)"
                                              : g_pb.obs_reason.c_str();
    }
    return -1;
}
uint64_t ggml_moe_phaseb_observe_counter(int32_t ctr) {
    if (ctr < 0 || ctr >= GGML_MOE_PB_O_COUNT) return 0;
    return g_pb.obs_ctr[ctr].load(std::memory_order_relaxed);
}
uint64_t ggml_moe_phaseb_observe_buf_bytes(void) {
    return g_pb.obs_on ? PB_OBSERVE_RESERVE_RECORDS * (uint64_t)sizeof(ggml_moe_phaseb_observe) : 0ull;
}
uint64_t ggml_moe_phaseb_observe_records(void) {
    std::lock_guard<std::mutex> lk(g_pb.obs_mu);
    return (uint64_t)g_pb.obs_recs.size();
}
// ★권위 표면 조회(`phase_b_run.no_output_graphs[]` — FROZEN_AUTHORITY_V3 §3.2.1 A5).
uint64_t ggml_moe_phaseb_no_output_count(uint32_t request_index) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    if ((size_t)request_index >= g_pb.runs.size()) return 0;
    return (uint64_t)g_pb.runs[(size_t)request_index].no_output.size();
}
// selftest 전용 — 억제 가드가 intent 발행 경로를 끊은 횟수(§1.5 "enqueue 전 억제"의 직접 관측).
uint64_t ggml_moe_phaseb_test_no_output_suppressed(void) {
    return g_pb.obs_no_out_suppressed.load(std::memory_order_relaxed);
}
int ggml_moe_phaseb_no_output_at(uint32_t request_index, uint64_t i,
                                 uint32_t * out_graph_seq, uint32_t * out_decode_ordinal,
                                 uint16_t * out_target_layer) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    if ((size_t)request_index >= g_pb.runs.size()) return -1;
    const std::vector<PbNoOutputEntry> & v = g_pb.runs[(size_t)request_index].no_output;
    if (i >= (uint64_t)v.size()) return -1;
    if (out_graph_seq)      *out_graph_seq      = v[(size_t)i].graph_seq;
    if (out_decode_ordinal) *out_decode_ordinal = v[(size_t)i].decode_ordinal;
    if (out_target_layer)   *out_target_layer   = v[(size_t)i].target_layer;
    return 0;
}
int ggml_moe_phaseb_test_observe_record(uint64_t idx, void * out_record, uint32_t bytes) {
    if (!out_record || bytes != sizeof(ggml_moe_phaseb_observe)) return -1;
    std::lock_guard<std::mutex> lk(g_pb.obs_mu);
    if (idx >= (uint64_t)g_pb.obs_recs.size()) return -1;
    std::memcpy(out_record, &g_pb.obs_recs[(size_t)idx], sizeof(ggml_moe_phaseb_observe));
    return 0;
}

bool         ggml_moe_phaseb_prefetch_on(void)  { return g_pb.pf_on; }
int32_t      ggml_moe_phaseb_prefetch_k(void)   { return g_pb.pf_k; }
int32_t      ggml_moe_phaseb_prefetch_n(void)   { return g_pb.pf_n; }
const char * ggml_moe_phaseb_prefetch_off_reason(void) { return g_pb.pf_off_reason.c_str(); }

// equiv 모드 = A′ 동등성 게이트 전용 1런(§1 ★R1 #5): ①PA accuracy ②유효 K/N 둘 다 필수.
int ggml_moe_phaseb_equiv_precheck(bool phasea_accuracy, const char ** out_reason) {
    if (ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_EQUIV) return 0;
    if (!phasea_accuracy) {
        if (out_reason) *out_reason = "MOE_DIRECT_PHASEB=equiv requires MOE_DIRECT_PHASEA=accuracy";
        return -1;
    }
    if (!g_pb.pf_on) {
        if (out_reason) *out_reason = "MOE_DIRECT_PHASEB=equiv requires a valid MOE_DIRECT_PREFETCH_K/N"
                                      " (blocks passing the gate with a no-op predictor)";
        return -1;
    }
    return 0;
}

// ★정정 6호 — 오힌트 test 표면 게이팅(SPEC_PHASEB_TEST_HINT §1). 미설정이면 즉시 통과(표면
//   완전 비활성). 설정됐다면 다음 3종 전건이 init fail-close 다(호출자가 seal 을 거부한다 —
//   resource/worker 기동 **전**에 불려야 구조 잠금이 성립한다).
//     ① MOE_DIRECT_QUALIFY=1 부재  ② 무효값(rot 외)  ③ MOE_DIRECT_PHASEB=equiv 동시 설정
int ggml_moe_phaseb_test_hint_precheck(bool qualify, const char ** out_reason) {
    pb_resolve_env();
    if (g_pb.hint_raw.empty()) return 0;
    if (!qualify) {
        g_pb.hint_fail_reason = "MOE_DIRECT_PHASEB_TEST_HINT is a qualification-only surface - "
                                "startup is refused unless MOE_DIRECT_QUALIFY=1 is set as well";
        if (out_reason) *out_reason = g_pb.hint_fail_reason.c_str();
        return -1;
    }
    if (g_pb.hint_rule == GGML_MOE_PHASEB_TEST_HINT_NONE) {
        g_pb.hint_fail_reason = "MOE_DIRECT_PHASEB_TEST_HINT=\"" + g_pb.hint_raw + "\" is an invalid value (the only valid value is rot)";
        if (out_reason) *out_reason = g_pb.hint_fail_reason.c_str();
        return -1;
    }
    if (g_pb.mode == GGML_MOE_PHASEB_EQUIV) {
        g_pb.hint_fail_reason = "MOE_DIRECT_PHASEB_TEST_HINT and MOE_DIRECT_PHASEB=equiv cannot be set together"
                                " (keeps the A' authenticity gate free of injected hints)";
        if (out_reason) *out_reason = g_pb.hint_fail_reason.c_str();
        return -1;
    }
    return 0;
}

int32_t      ggml_moe_phaseb_test_hint_rule(void)   { pb_resolve_env(); return g_pb.hint_rule; }
const char * ggml_moe_phaseb_test_hint_rule_name(void) {
    pb_resolve_env();
    return g_pb.hint_rule == GGML_MOE_PHASEB_TEST_HINT_ROT ? "rot" : "";
}
int32_t      ggml_moe_phaseb_test_hint_offset(void)  { return g_pb.hint_offset; }
uint64_t     ggml_moe_phaseb_test_hint_applied(void) { return g_pb.hint_applied.load(std::memory_order_relaxed); }

} // extern "C"

// ===========================================================================
// §4. 내부 — capture_q / predictor worker / intent_q
// ===========================================================================

namespace {

int32_t pb_cap_alloc_locked() {
    if (g_pb.cap_free.empty()) return -1;
    const int32_t i = g_pb.cap_free.back();
    g_pb.cap_free.pop_back();
    return i;
}

// 요청별 누적 버킷 확보(§0.1 — 보존 request_epoch 귀속).
PbRunAcc & pb_run_acc_locked(uint32_t req) {
    const size_t idx = (req == GGML_MOE_PHASEB_NA_U32) ? 0 : (size_t)req;
    if (g_pb.runs.size() <= idx) g_pb.runs.resize(idx + 1);
    PbRunAcc & a = g_pb.runs[idx];
    if (!a.used) {
        a.used = true;
        const size_t nt = g_pb.n_moe_layers > 1 ? (size_t)(g_pb.n_moe_layers - 1) : 0;
        a.exp_sum.assign(nt, 0); a.exp_max.assign(nt, 0); a.exp_n.assign(nt, 0);
        // ★r1 M2 — generation 튜플은 이 버킷이 **생기는 순간** 박제된다(controller §1 "이미
        //   admission 된 generation 의 스탬프는 이후 어떤 조정으로도 불변"). run_begin 만이
        //   버킷 생성자가 아니므로(카운터도 lazy 생성) 여기가 유일한 박제 지점이다.
        if (g_pb.collect_on || g_pb.adapt_on) {
            a.stamp_k   = g_pb.pf_on ? g_pb.pf_k : 0;
            a.stamp_n   = g_pb.pf_on ? g_pb.pf_n : 0;
            a.stamp_qd  = g_pb.qd_effective;
            // ★5c-A1 — `stamp_rev` 리터럴 대입은 제거했다(A3′ ⓑ). 값은 그대로 0 이지만
            //   그것은 이제 `PbRunAcc` 필드 기본값이지 이 자리의 상수가 아니다 — adapt 축이
            //   열리는 원자에서 revision 을 여기 못 박아 두면 그 값이 곧 거짓이 된다.
            // ★r3 수리 2 — 테스트 override(프로덕션 0). generation 마다 다른 튜플을 박제해
            //   실 producer 가 mixed 한 wire 를 만들게 한다.
            const int32_t ok_ = g_pb.test_stamp_k.load(std::memory_order_relaxed);
            if (ok_ > 0) {
                a.stamp_k  = ok_;
                a.stamp_n  = g_pb.test_stamp_n.load(std::memory_order_relaxed);
                a.stamp_qd = g_pb.test_stamp_qd.load(std::memory_order_relaxed);
            }
        }
    }
    return a;
}

// top-16 부분 선택(내림차순 · 동점 = 낮은 expert id 우선 — 결정론).
void pb_topk(const float * lg, int32_t n_expert, uint16_t out[PB_K]) {
    char * used = g_pb.w_used_buf.data();
    std::memset(used, 0, (size_t)n_expert);
    const int32_t k = std::min<int32_t>(PB_K, n_expert);
    for (int32_t r = 0; r < k; r++) {
        int32_t best = -1; float bv = 0.0f;
        for (int32_t e = 0; e < n_expert; e++) {
            if (used[e]) continue;
            if (best < 0 || lg[e] > bv) { best = e; bv = lg[e]; }
        }
        used[best] = 1;
        out[r] = (uint16_t)best;
    }
    for (int32_t r = k; r < PB_K; r++) out[r] = GGML_MOE_PHASEB_NA_U16;
}

// ★정정 6호 — 오힌트 변형(SPEC_PHASEB_TEST_HINT §2 규칙 `rot`):
//     ids'[i] = (ids[i] + ⌊n_expert/2⌋) mod n_expert   — 전 층·전 intent.
//   전 ID 상이·범위/유일성/순서 보존·가역 결정론(128E: 0→64·63→127·64→0·127→63).
//   비활성 런의 비용 = 이 분기 1회. ★교차 R1 필수 1 — `hint_rule` 판정은 성공-intent 경로에서
//   이 함수의 단 1회뿐이어야 한다(§1 "로드 1회" 계약): 반환값(변형 적용 여부)을 호출자가
//   latch 해 applied 계수 가드에 재사용한다(g_pb.hint_rule 재판정 금지).
bool pb_apply_test_hint(uint16_t ids[PB_K], int32_t n_expert) {
    if (g_pb.hint_rule != GGML_MOE_PHASEB_TEST_HINT_ROT || n_expert <= 0) return false;
    const uint32_t off = (uint32_t)(n_expert / 2);
    for (int32_t r = 0; r < PB_K; r++) {
        if (ids[r] == GGML_MOE_PHASEB_NA_U16) continue;   // n_expert<16 의 NA 패딩은 변형 대상 아님
        ids[r] = (uint16_t)(((uint32_t)ids[r] + off) % (uint32_t)n_expert);
    }
    return true;
}

// ★정정 6호 §3 — `test_hint_applied` 는 변형 호출 수가 아니라 **ordinal 발급·enqueue 성공한
//   effective intent 수**다. 호출은 enqueue 성공 경로에서만(overflow 경로 제외).
void pb_note_test_hint_applied(uint32_t request_epoch) {
    g_pb.hint_applied.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);   // 락 순서: {run_mu} 는 최말단(int_mu 해제 뒤 호출)
    pb_run_acc_locked(request_epoch).test_hint_applied++;
}

// ★R4/R7 테스트 barrier — 지정 mode 에 도달하면 1회 정지한다. 해제는 `cap_barrier_release()`
//   **호출로만**(자가 해제 없음). 대기에는 60s **안전 타임아웃**만 두며, 발동하면 sticky
//   `bar_safety` 가 서서 해당 arm 이 FAIL 로 잡힌다(=selftest 영구 정지 방지용이지 정상 경로 아님).
void pb_cap_barrier(int32_t mode) {
    if (g_pb.test_cap_barrier.load(std::memory_order_relaxed) != mode) return;
    g_pb.test_cap_barrier.store(0, std::memory_order_relaxed);   // one-shot
    // ★R7 — **release-bound**: 정상 해제 경로는 `cap_barrier_release()` 뿐이다. 이전의 5s
    //   자가 해제는 fin_th 지연 같은 극단 스케줄에서 barrier 가 스스로 풀린 뒤 worker 가
    //   cap_mu 를 점유해 contended 양성이 **오귀속**될 수 있었다(결정성이 다시 시간 의존).
    //   안전 타임아웃은 픽스처 결함 시 selftest 영구 정지를 막기 위한 것뿐이며, 발동하면
    //   bar_safety 로 드러나 해당 케이스가 FAIL 로 잡힌다.
    g_pb.bar_held.fetch_add(1, std::memory_order_release);
    if (g_pb.cap_bar_hit) SetEvent((HANDLE)g_pb.cap_bar_hit);
    if (g_pb.cap_bar_go) {
        if (WaitForSingleObject((HANDLE)g_pb.cap_bar_go, 60000) != WAIT_OBJECT_0)
            g_pb.bar_safety.store(1, std::memory_order_relaxed);
    }
    g_pb.bar_held.fetch_sub(1, std::memory_order_release);
}

// ★R2 #5 — 상태 큐 append(tm_mu 아래). mailbox(현 graph)와 큐(전 graph 보존) 둘 다 갱신한다.
void pb_note_target_status(uint32_t graph_seq, int32_t target_mlpos, uint8_t reason, uint8_t overflow) {
    // ★status fanout 은 **소비자가 실재하는 모드에서만**(FROZEN_INCOMPLETE_REASON_MASK §6
    //   재개봉 1). 이 상태 이벤트의 유일한 소비자는 pb_sample 행인데 그 행은 `pb_row_locked` 의
    //   모드 게이트 때문에 trace/equiv 에서만 생성된다 — 그 밖의 모드(OFF·collect-only·observe)는
    //   **소비자가 구조적으로 부재**하다. 소비자 없는 생산은 링(2048)·보류함(4096)만 채우다
    //   complete=0 을 만든다(observe = 397B 실측 · OFF ∧ collect=on = G-4a 실측 26-08-28 —
    //   후자가 재개봉 사유다. 구 판의 "observe 한정" 게이트는 소비자 부재 조건의 부분집합이라
    //   collect-only 수집 런을 그대로 흘려 PENDING_CAP 을 래치했다).
    //   ★술어를 `pb_row_locked` 와 **같은 형태**로 두어 소비자 실재 조건과 1:1 로 묶는다.
    //     수집 축(collect)은 조건에 넣지 않는다 — collect + trace/equiv 는 소비자가 실재하므로
    //     fanout 을 유지한다. 게이트를 진입점 1곳에 두어 §6.2 전 호출부(worker degenerate·GEMV
    //     overflow·intent overflow·capture skip·대체 intent)를 한 번에 닫는다 — trace/equiv 는
    //     조건 밖이므로 호출 경로·순서·인자·소비 로직이 전건 불변이다.
    if (g_pb.mode != GGML_MOE_PHASEB_TRACE && g_pb.mode != GGML_MOE_PHASEB_EQUIV) return;
    if (target_mlpos < 1 || graph_seq == GGML_MOE_PHASEB_NA_U32) return;
    std::lock_guard<std::mutex> lk(g_pb.tm_mu);
    if (g_pb.mailbox_graph == graph_seq) {
        if (reason && target_mlpos < (int32_t)g_pb.skip_by_target.size())
            g_pb.skip_by_target[(size_t)target_mlpos] = reason;
        if (overflow && target_mlpos < (int32_t)g_pb.ovf_by_target.size())
            g_pb.ovf_by_target[(size_t)target_mlpos] = 1;
    }
    // ★고정 링 — 상한을 구조로 못박는다(정상 운전은 dispatcher 가 매 pass 비운다).
    //   만석 = 관측 결손이므로 조용히 버리지 않고 complete=0 을 latch 한다(§9 게이트 2).
    if (g_pb.status_q.empty()) g_pb.status_q.assign(PbState::STATUS_RING, PbState::PbStatusEvt{});
    if (g_pb.status_count >= PbState::STATUS_RING) {
        ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_RING_OVERFLOW);
        return;
    }
    g_pb.status_q[(g_pb.status_head + g_pb.status_count) % PbState::STATUS_RING] =
        PbState::PbStatusEvt{ graph_seq, target_mlpos, reason, overflow };
    g_pb.status_count++;
}

// a0_equiv 표에 산출값 기록(equiv 모드 전용).
void pb_equiv_record(int32_t target_layer, bool is_ref, const uint16_t v[PB_K], uint8_t reason) {
    if (g_pb.mode != GGML_MOE_PHASEB_EQUIV) return;
    std::lock_guard<std::mutex> lk(g_pb.equiv_mu);
    if (target_layer < 1 || target_layer >= (int32_t)g_pb.equiv_rows.size() + 1) return;
    PbEquivRow & r = g_pb.equiv_rows[(size_t)(target_layer - 1)];
    if (reason && r.reason == 0) r.reason = reason;
    if (!v) return;
    if (is_ref) { std::memcpy(r.ref, v, sizeof(r.ref)); r.have_ref = true; }
    else        { std::memcpy(r.cpu, v, sizeof(r.cpu)); r.have_cpu = true; }
}

// ── ★observe 행(SPEC_PHASEB_WIRE §13.2) ───────────────────────────────────
//   설계: **행을 graph_begin 에서 전 target 선생성**하고 capture/worker 가 사후 patch 한다.
//   ⓐcoverage(레코드 수 = Σ decode graph × (n_moe_layers−1))가 worker 진행 상태와 무관하게
//     구조로 성립한다 ⓑgraph 경계에서 predictor 를 드레인할 필요가 없다(= 관측이 실행 스케줄에
//     개입하지 않는다 — §1 "관측만 가른다") ⓒ늦게 도착한 worker 결과도 finalize 전이면 제 행에
//     반영된다(a0_equiv 의 graph-local 표와 다른 점).
//   ★FROZEN_PB_OBSV_V2 §4.2/§5.2(E3·E4) — `n_outputs==0` ∧ capability `output_crop_before_last_moe`
//     인 decode graph 는 **전 target 행을 그대로 선생성**한 뒤 **마지막 관리 MoE 물리층 행 1개만**
//     NO_OUTPUT 으로 **전환**한다(행 신설·생략이 아니다 → 레코드 수 등식 불변).
void pb_observe_open_graph(uint32_t gs, uint32_t req, uint32_t dec, bool no_output) {
    if (!g_pb.obs_on || g_pb.n_moe_layers < 2 || gs == GGML_MOE_PHASEB_NA_U32) return;
    const int32_t n_rows = g_pb.n_moe_layers - 1;
    uint32_t made = 0;
    bool first_no_output = false;   // 권위 배열 등재 여부 — **행 생성과 독립** · graph 당 1회
    {
        std::lock_guard<std::mutex> lk(g_pb.obs_mu);
        if (g_pb.obs_base_by_graph.count(gs)) return;   // 같은 graph 재개봉 금지(dup 0)
        const uint32_t base = (uint32_t)g_pb.obs_recs.size();
        for (int32_t ml = 1; ml <= n_rows; ml++) {
            g_pb.obs_ctr[GGML_MOE_PB_O_EXPECTED].fetch_add(1, std::memory_order_relaxed);
            if (!ggml_moe_phaseb_buffer_reserve(sizeof(ggml_moe_phaseb_observe))) {
                ggml_moe_phaseb_note_dropped(1);        // 상한 초과 = complete=0(§9 게이트 2)
                continue;
            }
            ggml_moe_phaseb_observe r{};
            r.graph_seq      = gs;
            r.request_index  = req;
            r.decode_ordinal = dec;
            r.target_layer   = (uint16_t)g_pb.moe_layers[(size_t)ml];
            r.flags          = 0;
            r.invalid_reason = (uint8_t)GGML_MOE_PB_OBS_CAPTURE_MISS;   // 기본 = 관측 결손
            for (int i = 0; i < PB_K; i++) r.pred[i] = GGML_MOE_PHASEB_NA_U16;
            r.predict_end_qpc = 0;
            g_pb.obs_recs.push_back(r);
            made++;
        }
        if (made == (uint32_t)n_rows) {
            g_pb.obs_base_by_graph[gs] = base;
            if (no_output) {
                // ★전 target 행 중 **마지막 1개**(= moe_layer_ids[-1])의 disposition 만 바꾼다.
                //   표현 계약(§4.1 E2): b1=1 ⇔ reason=6 · valid=0 · pred 전량 NA · qpc=0.
                //   같은 graph 의 다른 행은 **무변경**(NO_OUTPUT 전파 금지).
                //   ★wire 태그는 **완전 생성일 때만** 기입한다 — 부분 생성 graph 에는 마지막 행이
                //     존재하지 않을 수 있어 태그를 실을 자리가 없다(허위 태그 금지).
                ggml_moe_phaseb_observe & last = g_pb.obs_recs[(size_t)base + (size_t)(n_rows - 1)];
                last.flags          = GGML_MOE_PB_OBS_FLAG_NO_OUTPUT;   // valid(b0)=0
                last.invalid_reason = (uint8_t)GGML_MOE_PB_OBS_NO_OUTPUT;
                for (int i = 0; i < PB_K; i++) last.pred[i] = GGML_MOE_PHASEB_NA_U16;
                last.predict_end_qpc = 0;
            }
        } else {
            ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_RECORD_DROP);   // 부분 생성 = 결손
        }
        // ★억제 창구는 **행 생성 성공 여부와 독립**이다(FROZEN_PB_OBSV_V2 §1.5 생산 조건 3항에
        //   "행 생성 성공"이 없고, §6 게이트 ④ 문언도 "**zero-output graph 에서** 억제"다).
        //   버퍼 상한으로 행이 부분 생성된 런에서 억제가 풀리면, 서버가 실행하지도 않을 층의
        //   전문가를 다시 프리페치하게 된다 — 이 계약이 막으려는 낭비가 열화 경로에서 부활한다.
        //   insert 의 반환으로 **정확히 1회** 등재를 보장한다(완전 생성 경로의 obs_base_by_graph
        //   dup 가드가 부분 생성 경로에는 없기 때문에 여기서 dedupe 한다).
        if (no_output) first_no_output = g_pb.obs_no_out_graph.insert(gs).second;
        g_pb.obs_ctr[GGML_MOE_PB_O_RECORDED].fetch_add(made, std::memory_order_relaxed);
    }
    for (uint32_t i = 0; i < made; i++) ggml_moe_phaseb_note_sidecar_record(4, req);
    // ★권위 표면 등재(`phase_b_run.no_output_graphs[]` — FROZEN_AUTHORITY_V3 §3.2.1 A5).
    //   ★**행 생성 성공 여부와 독립**이다: 이 배열은 "서버가 확정한 값"을 싣는 표면이고(A5),
    //     서버의 `n_outputs==0` 확정은 엔진 버퍼 상태의 함수가 아니다. 부분 생성(RECORD_DROP)
    //     런에서는 `배열 길이 1 ∧ wire 태그 0` 이 되어 판독기의 3자 폐합이 FAIL 하는데, 이것이
    //     **의도된 fail-close** 다(그 런은 이미 complete=0 이라 자격 경로가 닫혀 있다). 반대로
    //     배열에서 빼면 "서버가 zero-output 을 확정했다"는 사실 자체가 소실돼 은폐가 된다.
    //   obs_mu 를 놓은 뒤 run_mu 를 잡는다(락 순서 규율: obs_mu 는 leaf —
    //   note_sidecar_record 호출 배치와 같은 이유다).
    if (first_no_output) {
        PbNoOutputEntry e;
        e.graph_seq      = gs;
        e.decode_ordinal = dec;
        e.target_layer   = (uint16_t)g_pb.moe_layers[(size_t)(g_pb.n_moe_layers - 1)];
        std::lock_guard<std::mutex> lk(g_pb.run_mu);
        pb_run_acc_locked(req).no_output.push_back(e);
    }
}

// ★(graph, target) 가 서버 확정 zero-output 의 **태깅 대상 행**인가(§1.5 "enqueue 전 억제" 판정).
//   대상 = 그 graph 의 마지막 관리 MoE 물리층(target_mlpos == n_moe_layers-1) 하나뿐이다.
bool pb_observe_is_no_output_target(uint32_t graph_seq, int32_t target_mlpos) {
    if (graph_seq == GGML_MOE_PHASEB_NA_U32) return false;
    if (target_mlpos != g_pb.n_moe_layers - 1) return false;
    std::lock_guard<std::mutex> lk(g_pb.obs_mu);
    return g_pb.obs_no_out_graph.count(graph_seq) != 0;
}

// (graph_seq, target_mlpos) 행 patch. reason=0 ∧ pred 있음 = valid · 그 외 = invalid(NA 규약 유지).
//   ★NA 규약(§13.2): valid=0 ⇒ pred 전부 0xFFFF ∧ qpc=0 / valid=1 ⇒ pred 산출값 ∧ qpc>0.
void pb_observe_patch(uint32_t graph_seq, int32_t target_mlpos, uint8_t reason,
                      const uint16_t * pred, uint64_t predict_end_qpc) {
    if (!g_pb.obs_on || target_mlpos < 1 || graph_seq == GGML_MOE_PHASEB_NA_U32) return;
    std::lock_guard<std::mutex> lk(g_pb.obs_mu);
    auto it = g_pb.obs_base_by_graph.find(graph_seq);
    if (it == g_pb.obs_base_by_graph.end()) return;      // prefill graph 등 — 행이 없다
    if (target_mlpos > g_pb.n_moe_layers - 1) return;
    const size_t idx = (size_t)it->second + (size_t)(target_mlpos - 1);
    if (idx >= g_pb.obs_recs.size()) return;
    ggml_moe_phaseb_observe & r = g_pb.obs_recs[idx];
    // ★NO_OUTPUT 행은 **확정 disposition** 이다 — 늦게 도착한 worker 결과·사유가 덮어쓰지 못한다
    //   (§4.1 동치 `valid=1 ∧ b1=1` 불법 조합의 구조적 차단).
    if (r.flags & GGML_MOE_PB_OBS_FLAG_NO_OUTPUT) return;
    if (reason == GGML_MOE_PB_OBS_NONE && pred && predict_end_qpc != 0) {
        r.flags          = 0x01;
        r.invalid_reason = (uint8_t)GGML_MOE_PB_OBS_NONE;
        std::memcpy(r.pred, pred, sizeof(r.pred));
        r.predict_end_qpc = predict_end_qpc;
    } else {
        r.flags          = 0x00;
        r.invalid_reason = reason ? reason : (uint8_t)GGML_MOE_PB_OBS_CAPTURE_MISS;
        for (int i = 0; i < PB_K; i++) r.pred[i] = GGML_MOE_PHASEB_NA_U16;
        r.predict_end_qpc = 0;
    }
}

// capture 훅의 elig_skip enum(§4.1) → observe invalid_reason(§13.2) 사상.
//   실제 큐 만석만 capture_miss(5) 이고 나머지 eligibility 위반은 전부 not_eligible(4) 이다.
uint8_t pb_observe_reason_from_skip(int32_t skip_reason) {
    return (skip_reason == GGML_MOE_PB_SKIP_CAPTURE_QUEUE_FULL)
           ? (uint8_t)GGML_MOE_PB_OBS_CAPTURE_MISS : (uint8_t)GGML_MOE_PB_OBS_NOT_ELIGIBLE;
}

// ★finalize 시 1회 — 레코드 전수에서 분류 카운터를 확정한다(§13.4 closure 4등식이 구조로 성립).
//   eligible = expected − not_eligible · captured = eligible − capture_miss 는 **파생**이고,
//   나머지는 레코드 사유의 배타 계수다(판독기가 .pb 로부터 독립 재계산해 대조한다 — §2.3 ⓙ).
void pb_observe_tally() {
    if (!g_pb.obs_on) return;
    uint64_t not_elig = 0, cap_miss = 0, valid = 0, inv_nz = 0, inv_nf = 0, inv_ov = 0, no_out = 0;
    {
        std::lock_guard<std::mutex> lk(g_pb.obs_mu);
        for (const auto & r : g_pb.obs_recs) {
            if (r.flags & GGML_MOE_PB_OBS_FLAG_VALID) { valid++; continue; }
            switch (r.invalid_reason) {
                case GGML_MOE_PB_OBS_NORM_ZERO:    inv_nz++;   break;
                case GGML_MOE_PB_OBS_NONFINITE:    inv_nf++;   break;
                case GGML_MOE_PB_OBS_OVERFLOW:     inv_ov++;   break;
                case GGML_MOE_PB_OBS_NOT_ELIGIBLE: not_elig++; break;
                case GGML_MOE_PB_OBS_NO_OUTPUT:    no_out++;   break;   // ★V2 독립 버킷(§5.2)
                default:                           cap_miss++; break;
            }
        }
    }
    // ★§5.2 5등식 — `no_output` 은 `expected`/`recorded` 에만 들고 `eligible`/`captured`/`Σinvalid_*`
    //   에는 들지 않는다. 따라서 파생식이 둘 다 no_output 을 먼저 뺀다:
    //     eligible = expected − not_eligible − no_output · captured = eligible − capture_miss
    //   (나머지 3등식 `expected=recorded`·`eligible=captured+capture_miss`·`captured=valid+Σinv` 는 불변)
    const uint64_t expected = g_pb.obs_ctr[GGML_MOE_PB_O_EXPECTED].load(std::memory_order_relaxed);
    const uint64_t elig_sub = not_elig + no_out;
    const uint64_t eligible = expected >= elig_sub ? expected - elig_sub : 0;
    const uint64_t captured = eligible >= cap_miss ? eligible - cap_miss : 0;
    g_pb.obs_ctr[GGML_MOE_PB_O_NO_OUTPUT].store(no_out, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_NOT_ELIGIBLE].store(not_elig, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_CAPTURE_MISS].store(cap_miss, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_VALID].store(valid, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_INVALID_NORM_ZERO].store(inv_nz, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_INVALID_NONFINITE].store(inv_nf, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_INVALID_OVERFLOW].store(inv_ov, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_ELIGIBLE].store(eligible, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_CAPTURED].store(captured, std::memory_order_relaxed);
    g_pb.obs_ctr[GGML_MOE_PB_O_CAPTURE_Q_OVERFLOW].store(
        g_pb.cap_overflow.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

// ★softplus — **엔진 CPU 커널과 바이트 동일 정의**(1차 소스: ggml/src/ggml-cpu/unary-ops.cpp:79
//   `op_softplus`). 자체 근사·다른 분기점을 쓰면 선택 ID 가 갈릴 수 있으므로 그대로 옮긴다.
inline float pb_softplus(float x) { return (x > 20.0f) ? x : logf(1.0f + expf(x)); }

// ── predictor worker 본체(§3) ─────────────────────────────────────────────
// 보정식: a0_norm[i] = src1[i] × w_{l+1}[i] / w_l[i]  →  TopK16(W_router[l+1] · a0_norm).
// ★GEMV는 반드시 이 worker에서(dispatcher 스레드 GEMV 금지 — §3).
void pb_worker_process(PbCaptureSlot & s) {
    // E-2a (Codex r2 blocker 2) - test-only one-shot park, inside the REAL busy window. The
    // caller set worker_busy under cap_mu and then released both mutexes before calling here,
    // which is the same window a long GEMV occupies, so a worker parked at this line really is
    // busy and really is unjoinable - the predictor drain's idle wait and its worker join now
    // have something genuine to spend a budget on. What the test controls is the duration, not
    // the state. Same shape as the cap barrier: production leaves the arm at 0 and pays one
    // relaxed load per processed capture, the park is released only by _release, and the hard
    // 5 s bound means a fixture that forgets cannot hang a run - it reports through _safety.
    if (g_pb.test_worker_hold.load(std::memory_order_relaxed) != 0) {
        g_pb.test_worker_hold.store(0, std::memory_order_relaxed);   // one-shot
        if (g_pb.whold_hit) SetEvent((HANDLE)g_pb.whold_hit);
        if (g_pb.whold_go) {
            if (WaitForSingleObject((HANDLE)g_pb.whold_go, 5000) != WAIT_OBJECT_0)
                g_pb.whold_safety.store(1, std::memory_order_relaxed);
        }
    }
    const int32_t ne = g_pb.n_embd;
    const int32_t nx = g_pb.n_expert;
    const int32_t tgt_layer = s.source_layer + 1;
    if (!g_pb.snap_sealed || ne <= 0 || nx <= 0 ||
        tgt_layer < 1 || tgt_layer >= g_pb.n_layer_exec) return;
    // ★층 마스크(DSV4 §1.5) — predictor eligibility 표면. capture 훅이 이미 같은 술어로 걸러
    //   내지만(정적 제외 target 은 capture 자체가 not_eligible), 산식 표면에도 같은 술어를 둬
    //   "표면별 마스크 표류"를 구조적으로 막는다(전 표면 동일 마스크 의무).
    if (tgt_layer < g_pb.ad_first_target) return;

    // ★NO_OUTPUT 억제(FROZEN_PB_OBSV_V2 §1.5 · §6 게이트 ④) — 서버가 `n_outputs==0` 으로 확정한
    //   graph 의 **마지막 관리 MoE target** 은 측정 대상 부재다. 그 행의 disposition 은 graph_begin
    //   에서 이미 NO_OUTPUT 으로 확정됐으므로 여기서는
    //     ⓐ 예측 산출·행 patch 를 하지 않고(확정 disposition 보존)
    //     ⓑ demand/prefetch intent 를 **intent_q enqueue 이전에** 억제한다
    //       (ordinal 발급·intent digest·INTENT_PREDICTED 계수도 발생하지 않는다).
    if (pb_observe_is_no_output_target(s.graph_seq, s.source_mlpos + 1)) {
        g_pb.obs_no_out_suppressed.fetch_add(1, std::memory_order_relaxed);   // 억제 발동 관측 창구
        return;
    }

    const float * w_cur  = g_pb.snap_norm.data()   + (size_t)s.source_layer * (size_t)ne;
    const float * w_next = g_pb.snap_norm.data()   + (size_t)tgt_layer      * (size_t)ne;
    const float * w_rt   = g_pb.snap_router.data() + (size_t)(tgt_layer - 1) * (size_t)ne * (size_t)nx;

    float * y = g_pb.w_norm_buf.data();
    bool degenerate = false;
    // ★observe 사유 분해(§13.2 enum) — 카운터·기존 skip 처분은 **그대로**(NONFINITE 1종)이고,
    //   분해된 사유는 observe 레코드에만 실린다(qwen 프리페치 경로 거동 무변화).
    uint8_t obs_reason = GGML_MOE_PB_OBS_NONE;
    for (int32_t i = 0; i < ne; i++) {
        const float d = w_cur[i];
        if (d == 0.0f) { degenerate = true; obs_reason = GGML_MOE_PB_OBS_NORM_ZERO; break; }
        if (!std::isfinite(d) || !std::isfinite(w_next[i]) || !std::isfinite(s.x[(size_t)i])) {
            degenerate = true; obs_reason = GGML_MOE_PB_OBS_NONFINITE; break;
        }
        y[i] = (s.x[(size_t)i] * w_next[i]) / d;
    }
    if (degenerate) {
        ggml_moe_phaseb_count_skip(GGML_MOE_PB_SKIP_NONFINITE, s.request_epoch);
        pb_note_target_status(s.graph_seq, s.source_mlpos + 1, GGML_MOE_PB_SKIP_NONFINITE, 0); // ★R2 #5
        pb_equiv_record(tgt_layer, /*is_ref=*/false, nullptr, /*nonfinite=*/2);
        pb_observe_patch(s.graph_seq, s.source_mlpos + 1, obs_reason, nullptr, 0);
        return;
    }

    // ★계열 selection-bias(SPEC_FAMILY_SIGNAL §1) — `topk_bias` 는 GEMV 결과에 층별 bias 를
    //   더한 뒤 top-k 한다(= 1차 소스 llama-graph.cpp:1840 `logits = logits + gate_inp_b` 후
    //   top_k 와 동일 순서). bias 를 요구하지 않는 계열은 이 포인터가 null 이고 가산도 없다.
    //   ★DSV4(`topk_sqrt_softplus_bias`)는 **가산 시점이 다르다**: bias 는 logit 이 아니라
    //   변환(sqrt∘softplus) **뒤의 probs** 에 더해지는 selection bias 다(1차 소스
    //   llama-graph.cpp:1858 probs · :1868 selection_probs = probs + exp_probs_b · :1912 top_k).
    //   두 시점을 뒤바꾸면 ordered Top16 이 실제로 갈린다(음성 컨트롤로 실증).
    const bool  bias_pre  = (g_pb.ad_selection_fn == GGML_MOE_PHASEB_SEL_TOPK_BIAS);
    const bool  bias_post = (g_pb.ad_selection_fn == GGML_MOE_PHASEB_SEL_TOPK_SQRT_SOFTPLUS_BIAS);
    const bool  want_bias = bias_pre || bias_post;
    const float * bias    = nullptr;
    if (want_bias) {
        if (g_pb.snap_bias.size() < (size_t)tgt_layer * (size_t)nx) {
            ggml_moe_phaseb_count_skip(GGML_MOE_PB_SKIP_NONFINITE, s.request_epoch);
            pb_observe_patch(s.graph_seq, s.source_mlpos + 1, GGML_MOE_PB_OBS_NONFINITE, nullptr, 0);
            return;   // bias 스냅샷 결손 = 산식 불성립(조용한 무-bias 산출 금지)
        }
        bias = g_pb.snap_bias.data() + (size_t)(tgt_layer - 1) * (size_t)nx;
    }
    float * lg = g_pb.w_logit_buf.data();
    for (int32_t e = 0; e < nx; e++) {
        const float * row = w_rt + (size_t)e * (size_t)ne;
        float acc = 0.0f;
        for (int32_t i = 0; i < ne; i++) acc += row[i] * y[i];
        if (bias && bias_pre) acc += bias[e];       // logit bias(topk_bias)
        if (!std::isfinite(acc)) { degenerate = true; obs_reason = GGML_MOE_PB_OBS_OVERFLOW; break; }
        if (bias_post) {                            // selection bias(topk_sqrt_softplus_bias)
            acc = sqrtf(pb_softplus(acc)) + (bias ? bias[e] : 0.0f);
            if (!std::isfinite(acc)) { degenerate = true; obs_reason = GGML_MOE_PB_OBS_OVERFLOW; break; }
        }
        lg[e] = acc;
    }
    if (degenerate) {
        ggml_moe_phaseb_count_skip(GGML_MOE_PB_SKIP_NONFINITE, s.request_epoch);
        pb_note_target_status(s.graph_seq, s.source_mlpos + 1, GGML_MOE_PB_SKIP_NONFINITE, 0); // ★R2 #5
        pb_equiv_record(tgt_layer, /*is_ref=*/false, nullptr, /*nonfinite=*/2);
        pb_observe_patch(s.graph_seq, s.source_mlpos + 1, obs_reason, nullptr, 0);
        return;
    }

    ggml_moe_phaseb_intent it{};
    it.request_epoch = s.request_epoch;
    it.graph_epoch   = s.graph_epoch;
    it.graph_seq     = s.graph_seq;
    it.source_layer  = (uint16_t)s.source_layer;
    it.target_layer  = (uint16_t)tgt_layer;
    it.source_mlpos  = s.source_mlpos;
    it.target_mlpos  = s.source_mlpos + 1;
    it.k             = (uint32_t)g_pb.pf_k;
    // ★5c-T §1 — owner generation 스탬프 N 결속(K 와 동형·발행 시 1회·이후 불변).
    //   admission 은 이 값을 읽고 전역 `pb_n` 을 직독하지 않는다.
    it.n             = (uint32_t)g_pb.pf_n;
    it.capture_qpc   = s.capture_qpc;
    // ★계열 adapter 선택식(SPEC_FAMILY_SIGNAL §1 `selection_fn`) — 코드 분기가 아니라 표에서
    //   읽은 값으로 가른다. 2단계 등재는 `topk_raw`(qwen35moe_v1) 1종이며, 그 분기는 구
    //   하드코딩과 **같은 함수·같은 인자**다(동일 산식·동일 tie-break).
    //   bias 소비 계열(topk_bias·topk_sigmoid_bias)은 descriptor 가 미등재라 해소 자체가
    //   되지 않으므로 여기 도달하지 않는다 — 산식·bias 스냅샷 시공은 3·4단계 몫이다.
    switch (g_pb.ad_selection_fn) {
        case GGML_MOE_PHASEB_SEL_TOPK_RAW:     // qwen35moe_v1 — TopK(Wx)
        case GGML_MOE_PHASEB_SEL_TOPK_BIAS:    // ★openai_moe_v1 — TopK(Wx + gate_b)(가산은 위 GEMV)
        case GGML_MOE_PHASEB_SEL_TOPK_SQRT_SOFTPLUS_BIAS:
            // ★deepseek4_v1 — TopK(sqrt(softplus(Wx)) + exp_probs_b)(변환·가산은 위 GEMV 루프)
            pb_topk(lg, nx, it.ids);
            break;
        default:
            return;   // 미등재 선택식 = intent 미발행(조용한 오산출 금지)
    }
    // ★observe 축(§2.1) — 예측만 산출하고 **dispatcher enqueue 는 하지 않는다**(speculative I/O 0).
    //   intent ordinal·digest·INTENT_PREDICTED 도 발생시키지 않는다(§13.3 ④ 의 wire 증명).
    if (g_pb.obs_on) {
        const uint64_t pend = ggml_moe_phaseb_now_qpc();
        // ★B — U18 입력 집계(capture→predict_end · warmup 포함 전 표본 단일 최댓값).
        if (g_pb.qpc_freq && pend > s.capture_qpc) {
            const uint64_t ns = (pend - s.capture_qpc) * 1000000000ull / g_pb.qpc_freq;
            uint64_t cur = g_pb.obs_cost_ns_max.load(std::memory_order_relaxed);
            while (ns > cur && !g_pb.obs_cost_ns_max.compare_exchange_weak(cur, ns,
                                                                          std::memory_order_relaxed)) {}
        }
        g_pb.obs_cost_samples.fetch_add(1, std::memory_order_relaxed);
        pb_observe_patch(s.graph_seq, s.source_mlpos + 1, GGML_MOE_PB_OBS_NONE, it.ids, pend);
        return;
    }
    // ★정정 6호 — 오힌트 변형은 **pb_topk 직후·predict_end_qpc 기록 전**(변형 비용이 실제
    //   predictor timing 에 포함되도록). 이후 소비처(digest·ordinal·pb_sample·dispatcher)는
    //   전부 post-transform ids 만 본다 = 유일 canonical intent.
    //   ★교차 R1 필수 1 — 반환값 latch(§1 "로드 1회": 아래 applied 가드에서 재판정 금지).
    const bool hint_transformed = pb_apply_test_hint(it.ids, nx);
    it.predict_end_qpc = ggml_moe_phaseb_now_qpc();

    // a0_cpu = 실 production 경로 산출값(§3 A′ 동등성 게이트).
    pb_equiv_record(tgt_layer, /*is_ref=*/false, it.ids, 0);

    // ★intent_ordinal 발급점 = intent digest 갱신점(§6). intent_q enqueue 성공과 원자적으로 묶는다.
    bool overflowed = false;
    {
        std::lock_guard<std::mutex> lk(g_pb.int_mu);
        const int32_t cap = (int32_t)g_pb.int_slots.size();
        if (g_pb.int_count >= cap) {
            // overflow — 조용한 drop 금지: 해당 request 프리페치 disable + 계수 + run 무효 latch.
            g_pb.disabled_request.store(s.request_epoch, std::memory_order_relaxed);
            g_pb.run_invalid.store(true, std::memory_order_relaxed);
            ggml_moe_phaseb_count(GGML_MOE_PB_C_INTENT_OVERFLOW, s.request_epoch, 1);
            overflowed = true;   // ★R2 #5 — 상태 기록은 int_mu 해제 뒤(락 보유 최소화)
        } else {
        it.intent_ordinal = g_pb.intent_ordinal_next++;
        g_pb.int_slots[(size_t)((g_pb.int_head + g_pb.int_count) % cap)] = it;
        g_pb.int_count++;
        // intent digest(§8) — ordinal 발급 순서대로 46B.
        uint64_t h = g_pb.dg_intent;
        const uint32_t re = it.request_epoch, ge = it.graph_epoch;
        const uint16_t sl = it.source_layer, tl = it.target_layer, ke = (uint16_t)it.k;
        h = fnv1a64(h, &re, sizeof(re));
        h = fnv1a64(h, &ge, sizeof(ge));
        h = fnv1a64(h, &sl, sizeof(sl));
        h = fnv1a64(h, &tl, sizeof(tl));
        h = fnv1a64(h, &ke, sizeof(ke));
        h = fnv1a64(h, it.ids, sizeof(it.ids));
        g_pb.dg_intent = h;
        g_pb.intent_count++;
        // ★R1 #2 ① — enqueue와 **같은 mutex 아래** 에서 dispatcher를 깨운다(lost-wake 방지).
        if (g_pb.wake_event) SetEvent((HANDLE)g_pb.wake_event);
        }
    }
    if (overflowed) {
        // ★R1 #10 ③ / R2 #5 — intent_dropped_overflow bit(0x04) producer.
        pb_note_target_status(s.graph_seq, it.target_mlpos, 0, 1);
        return;
    }
    ggml_moe_phaseb_count(GGML_MOE_PB_C_INTENT_PREDICTED, s.request_epoch, 1);
    if (hint_transformed) pb_note_test_hint_applied(s.request_epoch);   // ★교차 R1 필수 1 — latch 재사용


    // §7.5 predictor 노출 조인 재료(현재 graph의 target별 predict_end) — ★R1 #11 mutex 규율.
    {
        std::lock_guard<std::mutex> lk(g_pb.tm_mu);
        if (g_pb.mailbox_graph == s.graph_seq &&
            it.target_mlpos >= 1 && it.target_mlpos < (int32_t)g_pb.pend_by_target.size()) {
            g_pb.pend_by_target[(size_t)it.target_mlpos] = it.predict_end_qpc;
        }
    }
}

unsigned __stdcall pb_worker_main(void *) {
    for (;;) {
        PbCaptureSlot * slot = nullptr;
        int32_t si = -1;
        {
            std::unique_lock<std::mutex> lk(g_pb.cap_mu);
            g_pb.cap_cv.wait(lk, [] {
                return g_pb.cap_count > 0 || g_pb.worker_stop.load(std::memory_order_relaxed);
            });
            if (g_pb.cap_count == 0) {
                if (g_pb.worker_stop.load(std::memory_order_relaxed)) break;
                continue;
            }
            si = g_pb.cap_ring[(size_t)g_pb.cap_head];
            g_pb.cap_head = (g_pb.cap_head + 1) % (int32_t)g_pb.cap_slots.size();
            g_pb.cap_count--;
            g_pb.worker_busy = true;
            slot = &g_pb.cap_slots[(size_t)si];
        }
        pb_worker_process(*slot);            // ★양쪽 mutex 무보유 구간(GEMV)
        {
            std::lock_guard<std::mutex> lk(g_pb.cap_mu);
            g_pb.cap_free.push_back(si);
            g_pb.worker_busy = false;
            g_pb.cap_idle_cv.notify_all();
        }
    }
    return 0;
}

// ★★PB virtual A5(`SPEC_PB_VIRTUAL_RUNTIME_R1` §2-2 "worker 생성 실패=명시 실패 전파") —
//   반환 false = worker 미기동. 구 판은 로그만 남기고 void 로 빠져서 `note_seal` 이 0(성공)을
//   돌려줬다: 그러면 프리페치 ON 으로 seal 이 서고 intent 생산자가 없는 **조용한 no-op** 이
//   된다(설정은 켜졌는데 기능이 죽은 상태 — 사양이 금지하는 바로 그 형상).
bool pb_worker_start() {
    if (g_pb.worker_running.load(std::memory_order_acquire)) return true;
    // ★★R-2 게이트 — one-shot 실패 주입. `_beginthreadex` 실패는 합성으로 만들 수 없는데,
    //   바로 그 실패가 `pb_alloc_b2_resources` 의 **자기 unwind** 를 가르는 유일한 분기다
    //   (이 주입 없이는 §2-2 "실패 후 자원 잔존 0" 이 코드로만 참이고 관문이 없다).
    //   ★위치 구속: 스레드를 만들기 **전**에 소비한다 — 실 실패와 같은 지점이어야 잔존 형상이
    //     같다(만든 뒤 지우면 그건 다른 사건이다).
    if (g_pb.test_worker_fail_next.load(std::memory_order_relaxed) != 0) {
        g_pb.test_worker_fail_next.store(0, std::memory_order_relaxed);   // one-shot
        pb_log("cannot create the predictor worker thread (test injection)");
        return false;
    }
    g_pb.worker_stop.store(false, std::memory_order_relaxed);
    unsigned tid = 0;
    g_pb.worker_thread = _beginthreadex(nullptr, PB_PREDICTOR_STACK, pb_worker_main, nullptr,
                                        STACK_SIZE_PARAM_IS_A_RESERVATION, &tid);
    if (g_pb.worker_thread == 0) { pb_log("cannot create the predictor worker thread"); return false; }
    g_pb.worker_running.store(true, std::memory_order_release);
    return true;
}

// ★★T-1(교차 r3) — **유계 stop/join**. 반환 1=회수 완료 / 0=timeout(스레드 미회수).
//   `timeout_ms < 0` = INFINITE(프로덕션 경로·거동 불변).
//   ★사유: dispatcher 유계 종료가 **timeout 계측 전에** 이 함수를 동기 호출하므로,
//   여기가 INFINITE 면 predictor 정지 회귀는 **timeout 에 도달하기도 전에 교착**한다
//   (r3 S-2 ①). 유계화해야 종단 유계가 성립한다.
//   ★timeout 시 핸들을 닫지 않고 `worker_running` 도 내리지 않는다 — 스레드가 아직
//     그 상태를 쓰고 있을 수 있어 정리하면 use-after-free 다.
// E-2a (Codex r2 blocker 2) - the ONE budget, as an absolute instant on the steady clock, and
// what is left of it right now. Every stage of a stop asks this at its own entry instead of
// starting a clock of its own; that is the whole difference between "one 30 s bound" and
// "30 s per stage".
//   deadline_abs_ms <  0 : INFINITE, reported as -1, which every bounded wait below reads as
//                          "wait forever" - the unchanged production stop policy.
//   otherwise            : deadline - now, floored at 0. 0 is an answer, not an error: it means
//                          check once and do not wait, which is what an exhausted budget buys.
// The clock is the same std::chrono::steady_clock the caller in ggml-moe-direct.cpp reads, so
// the instant crosses the translation-unit boundary unchanged - no re-derivation, no drift.
inline int64_t pb_steady_ms() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline int32_t pb_budget_left_ms(int64_t deadline_abs_ms) {
    if (deadline_abs_ms < 0) return -1;
    const int64_t left = deadline_abs_ms - pb_steady_ms();
    if (left <= 0)         return 0;
    if (left > 0x7fffffff) return 0x7fffffff;
    return (int32_t)left;
}
// Record one stage of the current predictor drain: what it measured against, and what it was
// handed at its own entry. Test-only observation - nothing branches on it.
inline void pb_note_drain_stage(int32_t idx, int64_t deadline_abs_ms, int32_t left_ms) {
    if (idx < 0 || idx >= 2) return;
    g_pb.drain_stage_deadline[(size_t)idx].store(deadline_abs_ms, std::memory_order_relaxed);
    g_pb.drain_stage_left[(size_t)idx].store(left_ms, std::memory_order_relaxed);
    if (g_pb.drain_stage_n.load(std::memory_order_relaxed) < idx + 1) {
        g_pb.drain_stage_n.store(idx + 1, std::memory_order_relaxed);
    }
}

// T-1 (cross r3) - the bounded stop/join. 1 = the thread was reclaimed, 0 = timeout.
// E-2a (Codex r2 blocker 2) - it now takes the ONE deadline instead of a relative timeout. The
//   old form was handed the caller's timeout_ms again, from scratch, so the join spent a fresh
//   full budget on top of whatever the capture-idle wait above had already spent.
//   deadline_abs_ms < 0 = INFINITE (the production path, behaviour unchanged).
// On timeout the handle is NOT closed and worker_running is NOT lowered: the thread may still
// be using that state, so tidying it up here would be a use-after-free.
int pb_worker_stop_join_until(int64_t deadline_abs_ms) {
    if (!g_pb.worker_running.load(std::memory_order_acquire)) return 1;
    {
        std::lock_guard<std::mutex> lk(g_pb.cap_mu);
        g_pb.worker_stop.store(true, std::memory_order_relaxed);
        g_pb.cap_cv.notify_all();
    }
    const int32_t left = pb_budget_left_ms(deadline_abs_ms);
    pb_note_drain_stage(/*idx=*/1, deadline_abs_ms, left);
    const DWORD wr = WaitForSingleObject((HANDLE)g_pb.worker_thread,
                                         left < 0 ? INFINITE : (DWORD)left);
    if (wr != WAIT_OBJECT_0) { g_pb.stop_timed_out.store(1, std::memory_order_release); return 0; }
    CloseHandle((HANDLE)g_pb.worker_thread);
    g_pb.worker_thread = 0;
    g_pb.worker_running.store(false, std::memory_order_release);
    return 1;
}

void pb_worker_stop_join() { (void)pb_worker_stop_join_until(/*deadline_abs_ms=*/-1); }

// capture 풀·intent_q 할당.
//   ★§7.2 규칙 분할(SPEC_PHASEB_WIRE §13.4 8호): **signal-path**(capture 풀·스냅샷·predictor·
//   observe 버퍼) = `prefetch_on || signal_observe_on` · **intent_q·speculative dispatcher 상태**
//   = `prefetch_on` 만. 둘 다 off 면 §7.2 게이트 ⓐ 의 "B2 확장 블록 전부 0" 그대로다.
// ★★PB virtual A5 — 반환 false = **worker 생성 실패**(그 외 경로는 전부 true).
//   자원 미생성(신호 축 off · adapter 미해소 · 이미 확보)은 실패가 아니다.
// ★★R-2(FROZEN §2-2 "실패 후 자원 잔존 0" · 교차 r1 F1) — `pb_alloc_b2_resources` 가 잡는
//   자원의 **정확한 역연산**. 트랜잭션 경계가 그 함수이므로 되돌림도 그 함수가 만든 것만
//   대상으로 한다(worker 는 호출자가 먼저 접었거나 애초에 못 떴다).
void pb_free_b2_resources() {
    g_pb.cap_slots.clear();
    g_pb.cap_slots.shrink_to_fit();
    g_pb.cap_free.clear();
    g_pb.cap_ring.clear();
    g_pb.cap_head = 0; g_pb.cap_count = 0;
    g_pb.int_slots.clear();
    g_pb.int_slots.shrink_to_fit();
    g_pb.int_head = 0; g_pb.int_count = 0;
    {
        std::lock_guard<std::mutex> lk(g_pb.obs_mu);
        g_pb.obs_recs.clear();
        g_pb.obs_recs.shrink_to_fit();
    }
    g_pb.cut_closed.store(false, std::memory_order_relaxed);
}

bool pb_alloc_b2_resources() {
    const bool sig = g_pb.pf_on || g_pb.obs_on;
    if (!sig) return true;
    if (g_pb.capture_width <= 0) return true;   // adapter 미해소 = 자원 미생성(pf_on 과 동반 불변)
    if (!g_pb.cap_slots.empty()) return true;
    g_pb.cap_slots.resize(GGML_MOE_PHASEB_CAPTURE_SLOTS);
    // ★capture 폭 = descriptor n_embd(SPEC_FAMILY_SIGNAL §1) — qwen(3072)에서 구 고정값과 동일.
    for (auto & s : g_pb.cap_slots) s.x.assign((size_t)g_pb.capture_width, 0.0f);
    g_pb.cap_free.clear();
    g_pb.cap_free.reserve(GGML_MOE_PHASEB_CAPTURE_SLOTS);
    for (int32_t i = GGML_MOE_PHASEB_CAPTURE_SLOTS - 1; i >= 0; i--) g_pb.cap_free.push_back(i);
    g_pb.cap_ring.assign(GGML_MOE_PHASEB_CAPTURE_SLOTS, -1);
    g_pb.cap_head = 0; g_pb.cap_count = 0;
    g_pb.cut_closed.store(false, std::memory_order_relaxed);
    if (g_pb.pf_on) {   // ★intent_q = speculative 축 전용(observe 축은 0 — §2.1)
        g_pb.int_slots.assign(GGML_MOE_PHASEB_INTENT_SLOTS, ggml_moe_phaseb_intent{});
        g_pb.int_head = 0; g_pb.int_count = 0;
    }
    if (g_pb.obs_on) {  // ★observe 레코드 버퍼 — seal 정적 예약(§13.4 `observe_buf_bytes`)
        std::lock_guard<std::mutex> lk(g_pb.obs_mu);
        g_pb.obs_recs.reserve((size_t)PB_OBSERVE_RESERVE_RECORDS);
    }
    // ★★R-2 — **자기 unwind**. worker 는 이 함수의 마지막 단계라, 여기서 실패하면 위에서 잡은
    //   capture 풀·intent_q·observe 버퍼가 그대로 남는다. 구 판은 그 잔존을 두고 false 만
    //   돌려줬다 ⇒ 같은 프로세스에서 seal 을 재시도하면 `cap_slots` 가 이미 차 있어 위의
    //   `if (!cap_slots.empty()) return true;` 로 **worker 재기동 없이 통과**하고, 프리페치가
    //   조용한 no-op 이 된다(교차 r1 F1 의 2차 영향). 실패는 흔적을 남기지 않는다.
    if (!pb_worker_start()) { pb_free_b2_resources(); return false; }
    return true;
}

} // namespace

// ===========================================================================
// §5. 공개 API — seal / 경계 / 스냅샷
// ===========================================================================

extern "C" {

int ggml_moe_phaseb_note_seal(int32_t n_expert, int32_t n_expert_used,
                              int32_t n_moe_layers, int32_t n_slots,
                              int32_t qd_effective, const int32_t * moe_layers,
                              const uint64_t * layer_stride_bytes,
                              int32_t n_layer_exec, int32_t n_embd, float rms_eps) {
    pb_resolve_env();
    g_pb.n_expert      = n_expert;
    g_pb.n_expert_used = n_expert_used;
    g_pb.n_moe_layers  = n_moe_layers;
    g_pb.n_slots       = n_slots;
    g_pb.qd_effective  = qd_effective;
    g_pb.moe_layers.assign(moe_layers, moe_layers + n_moe_layers);
    g_pb.strides.assign(layer_stride_bytes, layer_stride_bytes + n_moe_layers);
    int32_t max_layer = 0;
    for (int32_t i = 0; i < n_moe_layers; i++) max_layer = std::max(max_layer, moe_layers[i]);
    g_pb.layer_to_mlpos.assign((size_t)max_layer + 1, -1);
    for (int32_t i = 0; i < n_moe_layers; i++) g_pb.layer_to_mlpos[(size_t)moe_layers[i]] = i;
    g_pb.sealed = true;
    // ★5c-T(부록 14.4) — 수집 실효값 **seal 1회 해소**(이후 immutable · header echo 권위).
    g_pb.collect_on = g_pb.collect_raw;
    // ★5c-A1(§1-6 · §6-3) — adapt 실효값 **seal 1회 해소**(collect 축과 동형 · 이후 immutable).
    //   같은 자리에서 view 를 snapshot 으로 1회 복사한다 — 이 스냅샷을 고치는 심볼은 없다.
    g_pb.adapt_on = g_pb.adapt_raw;
    g_pb.view = g_pb.view_raw;
    g_pb.view_sealed = true;
    // ★정정 6호 — 오힌트 offset 확정(§2·§3 echo). n_expert 는 seal 에서만 알 수 있다.
    g_pb.hint_offset = (g_pb.hint_rule == GGML_MOE_PHASEB_TEST_HINT_ROT && n_expert > 0) ? n_expert / 2 : 0;
    // rank < k 전건이 실 expert id를 가져야 §4.2 닫힌 enum 처분이 성립한다 — n_expert < k면 OFF.
    if (g_pb.pf_on && n_expert < g_pb.pf_k) {
        pb_log("n_expert=%d < MOE_DIRECT_PREFETCH_K=%d - prefetch OFF(invalid_range)",
               (int)n_expert, (int)g_pb.pf_k);
        g_pb.pf_on = false;
        g_pb.pf_off_reason = "invalid_range";
    }
    // ★계열 신호 adapter 해소(SPEC_FAMILY_SIGNAL §1) — seal 1회. **미해소(none) = 신호 전면
    //   비활성**: 프리페치를 OFF 로 내려 기존 non-qwen fail-safe 경로(§7.2 게이트 ⓐ "OFF면 B2
    //   확장 전부 0" · capture 훅 PREFETCH_OFF skip)와 **동일 거동**으로 수렴시킨다.
    //   ★위치 구속: 아래 pb_refresh_all_off/스냅샷 형상 확정/자원 할당보다 **앞**이어야 한다.
    pb_resolve_adapter(n_expert, n_expert_used, n_layer_exec, n_embd);
    if (g_pb.pf_on && g_pb.ad_id == GGML_MOE_PHASEB_ADAPTER_NONE) {
        pb_log("family signal adapter unresolved(%s) - prefetch signal disabled (fail-close)", g_pb.ad_reason.c_str());
        g_pb.pf_on = false;
        g_pb.pf_off_reason = "adapter_none";
    }
    // ★observe 실행 축 확정(SPEC_FAMILY_SIGNAL §2.1 seal 전제 · SPEC_PHASEB_WIRE §13) —
    //   여기서 **자원 할당보다 먼저** 판정하므로 전제 위반 런은 capture 풀/worker/버퍼를 만들지
    //   않은 상태로 호출자(direct.cpp)의 init fail-close 를 맞는다.
    //   전제: ①supported adapter ②MOE_DIRECT_PHASEA=timing ③TEST_HINT 부재
    //         ④MOE_DIRECT_PREFETCH_K/N raw key **둘 다 부재**(빈 값으로 존재해도 실패).
    g_pb.obs_on = false;
    g_pb.obs_reason.clear();
    if (g_pb.mode == GGML_MOE_PHASEB_OBSERVE) {
        if (std::getenv("MOE_DIRECT_PREFETCH_K") != nullptr ||
            std::getenv("MOE_DIRECT_PREFETCH_N") != nullptr) {
            g_pb.obs_reason = "observe mode forbids MOE_DIRECT_PREFETCH_K/N (raw key must be absent)";
        } else if (!g_pb.hint_raw.empty()) {
            g_pb.obs_reason = "observe mode forbids MOE_DIRECT_PHASEB_TEST_HINT";
        } else if (ggml_moe_phasea_mode() != GGML_MOE_PHASEA_TIMING) {
            g_pb.obs_reason = "observe mode requires MOE_DIRECT_PHASEA=timing (oracle = timing sets.actual)";
        } else if (g_pb.ad_id == GGML_MOE_PHASEB_ADAPTER_NONE) {
            g_pb.obs_reason = "observe mode requires a supported family adapter (adapter_reason="
                              + g_pb.ad_reason + ")";
        } else if (n_layer_exec < 2 || n_embd <= 0 || g_pb.n_moe_layers < 2) {
            g_pb.obs_reason = "observe mode requires n_layer_exec>=2, n_embd>0, n_moe_layers>=2";
        } else {
            g_pb.obs_on = true;
        }
        if (!g_pb.obs_on) pb_log("observe seal precondition failed - %s", g_pb.obs_reason.c_str());
    }
    pb_refresh_all_off();
    // ★R1 #12 — PB off ∧ prefetch off 면 PB 벡터를 **아예 할당하지 않는다**(프로덕션 기본 경로).
    if (!g_pb.all_off.load(std::memory_order_relaxed)) {
        g_pb.r_qpc_by_layer.assign((size_t)n_moe_layers, 0);
        g_pb.pend_by_target.assign((size_t)n_moe_layers, 0);
        g_pb.skip_by_target.assign((size_t)n_moe_layers, 0);
        g_pb.ovf_by_target.assign((size_t)n_moe_layers, 0);
        g_pb.cap_layer_seen.assign((size_t)n_moe_layers, 0);
        g_pb.cap_layer_seen_graph = GGML_MOE_PHASEB_NA_U32;
    }
    // ★R1 #5 ② — router/norm 스냅샷의 **형상 할당(bytes 확정)을 seal 이전(= metrics header 기록
    //   전)** 으로 옮긴다. header phaseb_mem 에 실값이 실려야 §7.2 게이트 ⓐ가 성립한다.
    //   컨텍스트 생성 단계(snapshot_layer)는 이제 데이터 채움만 담당한다.
    if ((g_pb.pf_on || g_pb.obs_on) && n_layer_exec >= 2 && n_embd > 0) {
        ggml_moe_phaseb_snapshot_configure(n_layer_exec, n_embd, n_expert, n_expert_used, rms_eps);
    }
    // ★★PB virtual A5 — 자원·worker 확보 실패는 **호출자에게 전파**한다(§2-2). 구 판은 반환이
    //   상수 0 이라 predictor 없는 프리페치 ON 이 조용히 성립했다. 호출자(direct.cpp seal)가
    //   이 값을 보고 fail-close·rollback 을 수행한다.
    if (!pb_alloc_b2_resources()) return -1;
    return 0;
}

void ggml_moe_phaseb_run_begin(void) {
    if (g_pb.all_off.load(std::memory_order_relaxed) && g_pb.env_resolved) return;  // ★R1 #12 fast-gate
    pb_resolve_env();
    g_pb.request_index.store(g_pb.next_request_index++, std::memory_order_relaxed);
    g_pb.next_decode_ordinal = 0;
    g_pb.total_requests++;
    g_pb.run_active = true;
    g_pb.disabled_request.store(GGML_MOE_PHASEB_NA_U32, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    PbRunAcc & a = pb_run_acc_locked(g_pb.request_index.load(std::memory_order_relaxed));
    // ★r1 M1 — generation admission 순번·시각 발급(controller §1 stamp 원소 · 5c-T §4-①).
    //   수집 OFF 에서는 이 두 값이 방출되지 않으므로 wire 는 종전과 동일하다.
    if ((g_pb.collect_on || g_pb.adapt_on) && a.admission_seq == GGML_MOE_PHASEB_NA_U64) {
        a.admission_seq = g_pb.gen_seq_next.fetch_add(1, std::memory_order_relaxed);
        a.admission_qpc = ggml_moe_phaseb_now_qpc();
    }
}

void ggml_moe_phaseb_run_end(void) {
    // ★r1 M1 — generation terminal 순번·시각 폐쇄. run_active 해제 의미는 종전 그대로다
    //   (취소 통지 ≠ censor cut — 부록 14.5).
    if (g_pb.collect_on || g_pb.adapt_on) {
        std::lock_guard<std::mutex> lk(g_pb.run_mu);
        PbRunAcc & a = pb_run_acc_locked(g_pb.request_index.load(std::memory_order_relaxed));
        if (a.terminal_seq == GGML_MOE_PHASEB_NA_U64) {
            a.terminal_seq = g_pb.gen_seq_next.fetch_add(1, std::memory_order_relaxed);
            a.terminal_qpc = ggml_moe_phaseb_now_qpc();
        }
    }
    g_pb.run_active = false;
}

void ggml_moe_phaseb_set_wake_event(void * h) { std::lock_guard<std::mutex> lk(g_pb.int_mu); g_pb.wake_event = h; }
bool ggml_moe_phaseb_all_off(void)  { return g_pb.all_off.load(std::memory_order_relaxed); }

void ggml_moe_phaseb_graph_begin(uint8_t phase, uint32_t n_tokens, uint32_t n_outputs) {
    (void)n_tokens;
    if (g_pb.all_off.load(std::memory_order_relaxed) && g_pb.env_resolved) return;  // ★R1 #12 fast-gate
    pb_resolve_env();
    const uint32_t gs = g_pb.graph_seq.fetch_add(1, std::memory_order_relaxed);
    g_pb.cur_graph_seq.store(gs, std::memory_order_relaxed);
    g_pb.cur_phase.store(phase, std::memory_order_relaxed);
    g_pb.decode_ordinal.store(phase ? GGML_MOE_PHASEB_NA_U32 : g_pb.next_decode_ordinal,
                              std::memory_order_relaxed);
    if (!phase) g_pb.next_decode_ordinal++;
    g_pb.graph_active = true;
    {
        std::lock_guard<std::mutex> lk(g_pb.tm_mu);
        std::fill(g_pb.r_qpc_by_layer.begin(), g_pb.r_qpc_by_layer.end(), 0);
        std::fill(g_pb.pend_by_target.begin(), g_pb.pend_by_target.end(), 0);
        std::fill(g_pb.skip_by_target.begin(), g_pb.skip_by_target.end(), 0);
        std::fill(g_pb.ovf_by_target.begin(), g_pb.ovf_by_target.end(), 0);
        g_pb.mailbox_graph = gs;
    }
    {
        std::lock_guard<std::mutex> lk(g_pb.cap_mu);
        std::fill(g_pb.cap_layer_seen.begin(), g_pb.cap_layer_seen.end(), 0);
        g_pb.cap_layer_seen_graph = gs;
    }
    if (g_pb.mode == GGML_MOE_PHASEB_EQUIV && g_pb.n_layer_exec > 1) {
        std::lock_guard<std::mutex> lk(g_pb.equiv_mu);
        g_pb.equiv_rows.assign((size_t)(g_pb.n_layer_exec - 1), PbEquivRow{});
        for (auto & r : g_pb.equiv_rows) {
            for (int i = 0; i < PB_K; i++) { r.ref[i] = GGML_MOE_PHASEB_NA_U16; r.cpu[i] = GGML_MOE_PHASEB_NA_U16; }
        }
    }
    // ★observe 행 선생성(§13.2 coverage) — decode graph 에 한해 target=layer_ids[1:] 전건.
    //   ★NO_OUTPUT 생산 조건(FROZEN_PB_OBSV_V2 §1.5) 3항 **동시 성립**에서만 태깅한다:
    //     ① 서버가 이 graph 의 `n_outputs` 를 확정해 전달했고 그 값이 0(사후 추론 금지)
    //     ② 계열 capability `output_crop_before_last_moe` 가 **명시 true**(자동 상속 없음)
    //     ③ decode graph(prefill 은 observe 행 자체가 없다)
    if (!phase) {
        const bool no_output = (n_outputs == 0) && g_pb.ad_out_crop;
        pb_observe_open_graph(gs, g_pb.request_index.load(std::memory_order_relaxed),
                              g_pb.decode_ordinal.load(std::memory_order_relaxed), no_output);
    }
}

void ggml_moe_phaseb_graph_end(void) {
    if (g_pb.all_off.load(std::memory_order_relaxed)) return;   // ★R1 #12 fast-gate
    if (!g_pb.graph_active) return;
    g_pb.graph_active = false;
    const uint32_t gs  = g_pb.cur_graph_seq.load(std::memory_order_relaxed);
    const uint32_t req = g_pb.request_index.load(std::memory_order_relaxed);
    const uint32_t dec = g_pb.decode_ordinal.load(std::memory_order_relaxed);
    const uint8_t  ph  = g_pb.cur_phase.load(std::memory_order_relaxed);

    // ★equiv 모드는 결정론 대조를 위해 worker drain을 기다린다(schedule_perturbed=true 런).
    if (g_pb.mode == GGML_MOE_PHASEB_EQUIV) ggml_moe_phaseb_test_wait_idle(10000);

    // §7.5 predictor 노출 = max(0, predict_end(target) − R_qpc(source=target−1)).
    if (!ph) {
        // ★R1 #11 — mailbox 는 tm_mu 아래에서만 읽는다(worker 병행 쓰기).
        std::vector<uint64_t> pe_snap, r_snap;
        {
            std::lock_guard<std::mutex> lkt(g_pb.tm_mu);
            if (g_pb.mailbox_graph == gs) { pe_snap = g_pb.pend_by_target; r_snap = g_pb.r_qpc_by_layer; }
        }
        std::lock_guard<std::mutex> lk(g_pb.run_mu);
        PbRunAcc & a = pb_run_acc_locked(req);
        for (size_t t = 1; t < pe_snap.size(); t++) {
            const uint64_t pe = pe_snap[t];
            const uint64_t r  = r_snap[t - 1];
            if (pe == 0 || r == 0) continue;                 // intent 없는 행은 표본 제외
            const uint64_t d  = pe > r ? pe - r : 0;
            const uint64_t us = g_pb.qpc_freq ? (d * 1000000ull / g_pb.qpc_freq) : 0;
            if (t - 1 < a.exp_sum.size()) {
                a.exp_sum[t - 1] += us;
                if (us > a.exp_max[t - 1]) a.exp_max[t - 1] = us;
                a.exp_n[t - 1]++;
            }
        }
    }

    // a0_equiv 레코드 방출 — eligibility 통과분(valid=1)과 미통과분(valid=0·사유) **전부**.
    if (g_pb.mode == GGML_MOE_PHASEB_EQUIV && !ph && g_pb.n_layer_exec > 1) {
        std::lock_guard<std::mutex> lk(g_pb.equiv_mu);
        for (size_t i = 0; i < g_pb.equiv_rows.size(); i++) {
            PbEquivRow & r = g_pb.equiv_rows[i];
            ggml_moe_phaseb_a0equiv rec{};
            rec.graph_seq      = gs;
            rec.request_index  = req;
            rec.decode_ordinal = dec;
            rec.target_layer   = (uint16_t)(i + 1);
            uint8_t reason = r.reason;
            if (!r.have_ref && !r.have_cpu)      reason = reason ? reason : 5;  // capture_miss
            else if (r.have_ref != r.have_cpu)   reason = reason ? reason : 4;  // eligibility_drift
            const bool valid = (r.have_ref && r.have_cpu && reason == 0);
            rec.flags          = valid ? 0x01 : 0x00;
            rec.invalid_reason = valid ? 0 : (reason ? reason : 5);
            std::memcpy(rec.a0_ref, r.ref, sizeof(rec.a0_ref));
            std::memcpy(rec.a0_cpu, r.cpu, sizeof(rec.a0_cpu));
            if (ggml_moe_phaseb_buffer_reserve(sizeof(rec))) {
                g_pb.equiv_recs.push_back(rec);
                ggml_moe_phaseb_note_sidecar_record(1, req);
            } else {
                ggml_moe_phaseb_note_dropped(1);
            }
        }
    }
    g_pb.cur_graph_seq.store(GGML_MOE_PHASEB_NA_U32, std::memory_order_relaxed);
}

void ggml_moe_phaseb_current_epoch(uint32_t * out_request_index, uint32_t * out_decode_ordinal,
                                   uint32_t * out_graph_seq, uint8_t * out_phase) {
    if (out_request_index)  *out_request_index  = g_pb.request_index.load(std::memory_order_relaxed);
    if (out_decode_ordinal) *out_decode_ordinal = g_pb.decode_ordinal.load(std::memory_order_relaxed);
    if (out_graph_seq)      *out_graph_seq      = g_pb.cur_graph_seq.load(std::memory_order_relaxed);
    if (out_phase)          *out_phase          = g_pb.cur_phase.load(std::memory_order_relaxed);
}

uint64_t ggml_moe_phaseb_total_requests(void) { return g_pb.total_requests; }

// ── 스냅샷 ─────────────────────────────────────────────────────────────────
bool ggml_moe_phaseb_snapshot_needed(void) {
    pb_resolve_env();
    return (g_pb.pf_on || g_pb.obs_on) && !g_pb.snap_sealed;   // ★signal-path 축(§7.2 분할)
}

int ggml_moe_phaseb_snapshot_configure(int32_t n_layer_exec, int32_t n_embd,
                                       int32_t n_expert, int32_t n_expert_used, float rms_eps) {
    pb_resolve_env();
    if (!(g_pb.pf_on || g_pb.obs_on)) return 0;
    if (n_layer_exec < 2 || n_embd <= 0 || n_expert <= 0) return -1;
    // ★capture 폭은 adapter 해소로 확정된다(=descriptor n_embd·guard ≤8192). 여기 들어온
    //   n_embd 가 그 폭과 다르면 배선 표류이므로 fail-close(구 3072 고정 상한 검사의 후신).
    if (g_pb.capture_width <= 0 || n_embd != g_pb.capture_width) {
        pb_log("n_embd=%d != capture width %d - prefetch disabled (capture slot shape mismatch)",
               (int)n_embd, (int)g_pb.capture_width);
        g_pb.pf_on = false; g_pb.pf_off_reason = "invalid_range";
        // ★observe 축도 같은 배선 표류에서 성립하지 않는다 — 축을 내리고 런을 무효로 표시한다
        //   (관측 결손을 조용히 남기지 않는다 · §9 게이트 2).
        if (g_pb.obs_on) {
            g_pb.obs_on = false;
            ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_SNAPSHOT_MISMATCH);
        }
        return -1;
    }
    // ★R1 #5 ② — seal에서 형상이 이미 확정됐다면 재할당하지 않는다(데이터 채움만 남는다).
    if (g_pb.snap_configured && g_pb.n_layer_exec == n_layer_exec && g_pb.n_embd == n_embd) return 0;
    g_pb.n_layer_exec  = n_layer_exec;
    g_pb.n_embd        = n_embd;
    g_pb.n_expert      = n_expert;
    g_pb.n_expert_used = n_expert_used;
    g_pb.rms_eps       = rms_eps;
    const size_t rt = (size_t)(n_layer_exec - 1) * (size_t)n_embd * (size_t)n_expert;
    g_pb.snap_router.assign(rt, 0.0f);
    g_pb.snap_norm.assign((size_t)n_layer_exec * (size_t)n_embd, 0.0f);
    g_pb.snap_router_ok.assign((size_t)(n_layer_exec - 1), 0);
    g_pb.snap_norm_ok.assign((size_t)n_layer_exec, 0);
    // ★selection-bias 스냅샷 — 요구 계열에서만 할당(§1 bias_snapshot · 없으면 회계 0).
    if (g_pb.ad_bias_kind != GGML_MOE_PHASEB_BIAS_NONE) {
        g_pb.snap_bias.assign((size_t)(n_layer_exec - 1) * (size_t)n_expert, 0.0f);
        g_pb.snap_bias_ok.assign((size_t)(n_layer_exec - 1), 0);
    } else {
        g_pb.snap_bias.clear();
        g_pb.snap_bias_ok.clear();
    }
    g_pb.w_norm_buf.assign((size_t)n_embd, 0.0f);
    g_pb.w_logit_buf.assign((size_t)n_expert, 0.0f);
    g_pb.w_used_buf.assign((size_t)n_expert, 0);
    g_pb.snap_configured = true;
    g_pb.snap_sealed = false;
    return 0;
}

int ggml_moe_phaseb_snapshot_layer(int32_t layer, const struct ggml_tensor * router,
                                   const struct ggml_tensor * post_norm) {
    if (!g_pb.snap_configured) return -1;
    if (layer < 0 || layer >= g_pb.n_layer_exec) return -1;
    if (!post_norm) return -1;
    if (post_norm->type != GGML_TYPE_F32 || post_norm->ne[0] != (int64_t)g_pb.n_embd) return -1;
    ggml_backend_tensor_get((ggml_tensor *)post_norm,
                            g_pb.snap_norm.data() + (size_t)layer * (size_t)g_pb.n_embd,
                            0, (size_t)g_pb.n_embd * sizeof(float));
    g_pb.snap_norm_ok[(size_t)layer] = 1;
    // ★층 마스크(DSV4 §1.5) — 정적 제외 target 층은 router 스냅샷을 **요구하지 않는다**
    //   (그 층의 예측 자체가 없다). 생산자가 null 을 넘겨도 결손이 아니며 seal 도 요구하지 않는다.
    if (layer >= 1 && layer >= g_pb.ad_first_target) {
        if (!router) return -1;
        // ★router dtype(SPEC_FAMILY_SIGNAL §1.6) — 해소된 계열 행이 허용할 때만 BF16 을 추가 수용.
        const bool rt_bf16 = (router->type == GGML_TYPE_BF16) && g_pb.ad_router_bf16_ok;
        if ((router->type != GGML_TYPE_F32 && !rt_bf16) ||
            router->ne[0] != (int64_t)g_pb.n_embd || router->ne[1] != (int64_t)g_pb.n_expert) return -1;
        const size_t nrt = (size_t)g_pb.n_embd * (size_t)g_pb.n_expert;
        float * dst = g_pb.snap_router.data() + (size_t)(layer - 1) * nrt;
        if (rt_bf16) {
            // ★§1.6 수용 규칙 ② — 여기서 **1회** canonical F32 로 승격해 기존 snap_router<float>
            //   에 적재한다. 원본 staging 은 이 스코프에서 소멸(영구 BF16/추가 F32 사본 없음)이며
            //   스냅샷 바이트 회계(phaseb_mem exact-key)는 F32 소스일 때와 완전히 같다.
            std::vector<ggml_bf16_t> raw(nrt);
            ggml_backend_tensor_get((ggml_tensor *)router, raw.data(), 0, nrt * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(raw.data(), dst, (int64_t)nrt);
        } else {
            ggml_backend_tensor_get((ggml_tensor *)router, dst, 0, nrt * sizeof(float));
        }
        g_pb.snap_router_ok[(size_t)(layer - 1)] = 1;
    }
    return 0;
}

// ★계열 selection-bias 스냅샷(§1) — layer 1..n-1 · {n_expert} F32. 요구 없는 계열은 -1.
int ggml_moe_phaseb_snapshot_bias(int32_t layer, const struct ggml_tensor * bias) {
    if (!g_pb.snap_configured) return -1;
    if (g_pb.ad_bias_kind == GGML_MOE_PHASEB_BIAS_NONE) return -1;
    if (layer < 1 || layer >= g_pb.n_layer_exec) return -1;
    if (layer < g_pb.ad_first_target) return -1;   // ★층 마스크 — 정적 제외 target 은 bias 대상 아님
    if (!bias) return -1;
    if (bias->type != GGML_TYPE_F32 || bias->ne[0] != (int64_t)g_pb.n_expert) return -1;
    if (g_pb.snap_bias.size() < (size_t)layer * (size_t)g_pb.n_expert) return -1;
    ggml_backend_tensor_get((ggml_tensor *)bias,
                            g_pb.snap_bias.data() + (size_t)(layer - 1) * (size_t)g_pb.n_expert,
                            0, (size_t)g_pb.n_expert * sizeof(float));
    g_pb.snap_bias_ok[(size_t)(layer - 1)] = 1;
    return 0;
}

int ggml_moe_phaseb_test_snapshot_bias_raw(int32_t layer, const float * bias) {
    if (!g_pb.snap_configured || !bias) return -1;
    if (g_pb.ad_bias_kind == GGML_MOE_PHASEB_BIAS_NONE) return -1;
    if (layer < 1 || layer >= g_pb.n_layer_exec) return -1;
    if (g_pb.snap_bias.size() < (size_t)layer * (size_t)g_pb.n_expert) return -1;
    std::memcpy(g_pb.snap_bias.data() + (size_t)(layer - 1) * (size_t)g_pb.n_expert, bias,
                (size_t)g_pb.n_expert * sizeof(float));
    g_pb.snap_bias_ok[(size_t)(layer - 1)] = 1;
    return 0;
}

int ggml_moe_phaseb_test_snapshot_raw(int32_t layer, const float * router, const float * post_norm) {
    if (!g_pb.snap_configured) return -1;
    if (layer < 0 || layer >= g_pb.n_layer_exec) return -1;
    if (post_norm) {
        std::memcpy(g_pb.snap_norm.data() + (size_t)layer * (size_t)g_pb.n_embd, post_norm,
                    (size_t)g_pb.n_embd * sizeof(float));
        g_pb.snap_norm_ok[(size_t)layer] = 1;
    }
    if (layer >= 1 && router) {
        std::memcpy(g_pb.snap_router.data() + (size_t)(layer - 1) * (size_t)g_pb.n_embd * (size_t)g_pb.n_expert,
                    router, (size_t)g_pb.n_embd * (size_t)g_pb.n_expert * sizeof(float));
        g_pb.snap_router_ok[(size_t)(layer - 1)] = 1;
    }
    return 0;
}

int ggml_moe_phaseb_snapshot_seal(void) {
    if (!g_pb.snap_configured) return -1;
    // ★층 마스크(DSV4 §1.5 · snapshot seal 표면) — 요구 범위는 **신호 대상 층뿐**이다:
    //     norm   = [first_target-1, n_layer_exec)   (source 층 ∪ target 층)
    //     router = [first_target,   n_layer_exec)   (인덱스 = layer-1)
    //     bias   = 〃
    //   기존 계열(first_target=1)에서는 norm 0..n-1 · router/bias 1..n-1 = 종전 전 범위 그대로다.
    const int32_t ft = g_pb.ad_first_target;
    for (size_t i = 0; i < g_pb.snap_norm_ok.size(); i++)
        if ((int32_t)i >= ft - 1 && !g_pb.snap_norm_ok[i])   return -1;
    for (size_t i = 0; i < g_pb.snap_router_ok.size(); i++)
        if ((int32_t)i + 1 >= ft && !g_pb.snap_router_ok[i]) return -1;
    // ★bias 요구 계열은 (마스크 안의) 전 층 채움이 확인돼야 seal 이다(부분 bias = 산식 오염).
    for (size_t i = 0; i < g_pb.snap_bias_ok.size(); i++)
        if ((int32_t)i + 1 >= ft && !g_pb.snap_bias_ok[i])   return -1;
    g_pb.snap_sealed = true;
    return 0;
}

uint64_t ggml_moe_phaseb_router_snapshot_bytes(void) { return (uint64_t)g_pb.snap_router.size() * sizeof(float); }
uint64_t ggml_moe_phaseb_norm_snapshot_bytes(void)   { return (uint64_t)g_pb.snap_norm.size()   * sizeof(float); }
uint64_t ggml_moe_phaseb_bias_snapshot_bytes(void)   { return (uint64_t)g_pb.snap_bias.size()   * sizeof(float); }

// ── capture 훅(§3) ─────────────────────────────────────────────────────────
int ggml_moe_phaseb_capture(int32_t src_layer, int32_t src_mlpos,
                            const void * src1_data, int32_t src1_type,
                            int32_t n_embd, int32_t n_rows, int32_t n_tokens,
                            int32_t src1_contiguous) {
    // ★R1 #12 — fast-gate: PB off ∧ prefetch off 면 캐시된 bool 1분기로 즉시 반환.
    if (g_pb.all_off.load(std::memory_order_relaxed)) return 0;
    if (!g_pb.sealed) return 0;
    const uint32_t req = g_pb.request_index.load(std::memory_order_relaxed);
    // 호출자가 mlpos를 모르면(ggml-cpu.c 훅) 실제 layer id로 해소한다(seal 이후 불변 표 조회).
    if (src_mlpos < 0) {
        src_mlpos = (src_layer >= 0 && src_layer < (int32_t)g_pb.layer_to_mlpos.size())
                    ? g_pb.layer_to_mlpos[(size_t)src_layer] : -1;
    }
    const uint32_t gs = g_pb.cur_graph_seq.load(std::memory_order_relaxed);

    pb_cap_barrier(1);   // 테스트 전용 — cap_mu 획득 전(= CUT 선승 arm)

    // ★★필수 ①(Codex R4) — CUT straddler 봉인: **래치 확인(등록)과 그 capture 의 전 관측
    //   mutation 을 하나의 동기화 경계(cap_mu) 안에 넣는다.**
    //     · 등록 성공(래치 열림 확인) ⇒ 이 capture 의 counter·elig_skip·status/run·dedupe·enqueue
    //       ·INTENT_CAPTURED 가 **전부 CUT 보다 앞**임이 보장된다.
    //     · CUT(`cut_close_capture`)도 cap_mu 를 잡으므로, 등록된 pre-CUT capture 가 mutation 을
    //       끝낼 때까지 **자동으로 대기**한다(별도 등록 카운터/CV 불요 — 상호배제가 곧 드레인).
    //     · 선점으로 늦게 재개된 capture 는 여기서 래치를 보고 **조용한 무동작**으로 끝난다.
    //   락 순서: cap_mu → {run_mu, tm_mu} — 선언된 전역 순서 그대로(역방향 경로 없음).
    std::lock_guard<std::mutex> lk(g_pb.cap_mu);
    if (g_pb.cut_closed.load(std::memory_order_acquire)) return 0;   // 등록 실패 = 무동작

    pb_cap_barrier(2);   // 테스트 전용 — 등록 직후(cap_mu 보유 · = capture 선승 arm)

    // ★R1 #10 ① — canonical part dedupe 를 eligibility 검사 **앞**으로. 훅은 층당 등록 part
    //   3회(gate/up/down) 불리므로 여기서 걸러야 skip 계수·행 전달이 (graph,layer)당 1회가 된다.
    const bool have_key = (src_mlpos >= 0 && src_mlpos < g_pb.n_moe_layers && gs != GGML_MOE_PHASEB_NA_U32);
    if (have_key) {
        if (g_pb.cap_layer_seen.empty()) return 0;
        if (g_pb.cap_layer_seen_graph != gs) {
            g_pb.cap_layer_seen_graph = gs;
            std::fill(g_pb.cap_layer_seen.begin(), g_pb.cap_layer_seen.end(), 0);
        }
        if (g_pb.cap_layer_seen[(size_t)src_mlpos]) return 0;   // 이 (graph, 층) 이미 처리
        g_pb.cap_layer_seen[(size_t)src_mlpos] = 1;
    }
    // ★R1 #10 ② — 사유는 카운터와 **pb_sample 행 양쪽**에 전달한다(target = src_mlpos+1).
    const int32_t tgt_ml = src_mlpos >= 0 ? src_mlpos + 1 : -1;
    auto skip = [&](int32_t reason) -> int {
        ggml_moe_phaseb_count_skip(reason, req);                 // cap_mu → run_mu
        pb_note_target_status(gs, tgt_ml, (uint8_t)reason, 0);   // cap_mu → tm_mu
        // ★observe 행 사유 전달(§13.2) — eligibility 위반=not_eligible(4)·큐 만석=capture_miss(5).
        pb_observe_patch(gs, tgt_ml, pb_observe_reason_from_skip(reason), nullptr, 0); // cap_mu → obs_mu
        return 0;
    };

    // ★eligibility(§1) — 위반 = 무동작 + reason별 계수(§4.1 enum 1:1). 관측 모드와 무관하게 계수한다
    //   (단 all_off fast-gate[R1 #12 — 이 함수 최전방]가 훅 자체를 반환시키는 전부-off 형상
    //    [phaseb_mode=off ∧ prefetch_on=false]은 예외 — SPEC_PHASEB_WIRE 정정 7호 구조적 N/A).
    if (n_tokens > 1) return skip(GGML_MOE_PB_SKIP_PREFILL);
    // ★signal-path 축(§7.2 분할) — prefetch 또는 observe 중 하나라도 켜져 있으면 capture 한다.
    if (!(g_pb.pf_on || g_pb.obs_on) || !g_pb.snap_sealed) return skip(GGML_MOE_PB_SKIP_PREFETCH_OFF);
    {
        const uint32_t dr = g_pb.disabled_request.load(std::memory_order_relaxed);
        if (dr != GGML_MOE_PHASEB_NA_U32 && dr == req) return skip(GGML_MOE_PB_SKIP_REQUEST_DISABLED);
    }
    if (g_pb.lora_swapped)  return skip(GGML_MOE_PB_SKIP_LORA_SWAP);
    if (n_rows != 1)        return skip(GGML_MOE_PB_SKIP_MULTI_TOKEN);
    if (src1_type != (int32_t)GGML_TYPE_F32) return skip(GGML_MOE_PB_SKIP_SRC1_TYPE);
    if (!src1_contiguous)   return skip(GGML_MOE_PB_SKIP_NONCONTIG);
    // trunk 범위 — source는 마지막 관리 층일 수 없고 target은 연속(l+1)이어야 한다.
    if (src_mlpos < 0 || src_mlpos + 1 >= g_pb.n_moe_layers ||
        g_pb.moe_layers[(size_t)src_mlpos + 1] != src_layer + 1 ||
        src_layer + 1 >= g_pb.n_layer_exec) return skip(GGML_MOE_PB_SKIP_SOURCE_RANGE);
    // ★층 마스크(DSV4 §1.5) — 정적 제외 target(= learned 라우팅이 아닌 층)은 capture 자체를
    //   하지 않는다. 행은 그대로 존재하고 처분만 `not_eligible` 이 된다(coverage 등식 불변 ·
    //   pb_observe_reason_from_skip: SOURCE_RANGE → NOT_ELIGIBLE).
    if (src_layer + 1 < g_pb.ad_first_target) return skip(GGML_MOE_PB_SKIP_SOURCE_RANGE);
    if (n_embd != g_pb.n_embd || n_embd > g_pb.capture_width) return skip(GGML_MOE_PB_SKIP_SOURCE_RANGE);
    if (!src1_data) return skip(GGML_MOE_PB_SKIP_SRC1_TYPE);
    if (gs == GGML_MOE_PHASEB_NA_U32) return skip(GGML_MOE_PB_SKIP_SOURCE_RANGE);

    const int32_t si = pb_cap_alloc_locked();
    if (si < 0) {
        g_pb.cap_overflow.fetch_add(1, std::memory_order_relaxed);
        g_pb.run_invalid.store(true, std::memory_order_relaxed);
        return skip(GGML_MOE_PB_SKIP_CAPTURE_QUEUE_FULL);   // 실제 큐 만석에만 쓴다
    }
    PbCaptureSlot & s = g_pb.cap_slots[(size_t)si];
    s.request_epoch = req;
    s.graph_epoch   = g_pb.decode_ordinal.load(std::memory_order_relaxed);
    s.graph_seq     = gs;
    s.source_mlpos  = src_mlpos;
    s.source_layer  = src_layer;
    s.n_embd        = n_embd;
    s.capture_qpc   = ggml_moe_phaseb_now_qpc();
    std::memcpy(s.x.data(), src1_data, (size_t)n_embd * sizeof(float));
    g_pb.cap_ring[(size_t)((g_pb.cap_head + g_pb.cap_count) % (int32_t)g_pb.cap_slots.size())] = si;
    g_pb.cap_count++;
    g_pb.cap_cv.notify_one();
    // ★필수 ① ⓑ — INTENT_CAPTURED 증가도 **cap_mu 안**에서(이전엔 해제 뒤라 flush 와 경합했다).
    ggml_moe_phaseb_count(GGML_MOE_PB_C_INTENT_CAPTURED, req, 1);
    return 1;
}

void ggml_moe_phaseb_note_lora_swap(void) { g_pb.lora_swapped = true; }

// a0_ref — 기존 Phase A accuracy callback 경로와 **동일 산식**(RMSNorm(h_l; w_{l+1}) → router[l+1]).
void ggml_moe_phaseb_tap_a0(int32_t src_layer, const float * h, int32_t n) {
    if (g_pb.mode != GGML_MOE_PHASEB_EQUIV) return;
    if (!g_pb.snap_sealed || !h || n != g_pb.n_embd) return;
    const int32_t tgt = src_layer + 1;
    if (tgt < 1 || tgt >= g_pb.n_layer_exec) return;
    const int32_t ne = g_pb.n_embd, nx = g_pb.n_expert;
    const float * w_norm = g_pb.snap_norm.data()   + (size_t)tgt * (size_t)ne;
    const float * w_rt   = g_pb.snap_router.data() + (size_t)(tgt - 1) * (size_t)ne * (size_t)nx;

    double ss = 0.0;
    for (int32_t i = 0; i < ne; i++) ss += (double)h[i] * (double)h[i];
    if (ss == 0.0) { pb_equiv_record(tgt, true, nullptr, /*norm_zero=*/1); return; }
    const float scale = (float)(1.0 / std::sqrt(ss / (double)ne + (double)g_pb.rms_eps));
    std::vector<float> y((size_t)ne);
    for (int32_t i = 0; i < ne; i++) y[(size_t)i] = h[i] * scale * w_norm[i];
    std::vector<float> lg((size_t)nx);
    for (int32_t e = 0; e < nx; e++) {
        const float * row = w_rt + (size_t)e * (size_t)ne;
        float acc = 0.0f;
        for (int32_t i = 0; i < ne; i++) acc += row[i] * y[(size_t)i];
        lg[(size_t)e] = acc;
    }
    // top-k 는 worker 버퍼를 쓰지 않는다(tap은 별도 스레드에서 올 수 있다).
    std::vector<char> used((size_t)nx, 0);
    uint16_t out[PB_K];
    const int32_t k = std::min<int32_t>(PB_K, nx);
    for (int32_t r = 0; r < k; r++) {
        int32_t best = -1; float bv = 0.0f;
        for (int32_t e = 0; e < nx; e++) {
            if (used[(size_t)e]) continue;
            if (best < 0 || lg[(size_t)e] > bv) { best = e; bv = lg[(size_t)e]; }
        }
        used[(size_t)best] = 1;
        out[r] = (uint16_t)best;
    }
    for (int32_t r = k; r < PB_K; r++) out[r] = GGML_MOE_PHASEB_NA_U16;
    pb_equiv_record(tgt, /*is_ref=*/true, out, 0);
}

// ── intent_q(dispatcher 소비) ──────────────────────────────────────────────
int ggml_moe_phaseb_intent_peek(struct ggml_moe_phaseb_intent * out) {
    std::lock_guard<std::mutex> lk(g_pb.int_mu);
    if (g_pb.int_count == 0) return 0;
    if (out) *out = g_pb.int_slots[(size_t)g_pb.int_head];
    return 1;
}

void ggml_moe_phaseb_intent_pop(void) {
    std::lock_guard<std::mutex> lk(g_pb.int_mu);
    if (g_pb.int_count == 0) return;
    g_pb.int_head = (g_pb.int_head + 1) % (int32_t)g_pb.int_slots.size();
    g_pb.int_count--;
}

int ggml_moe_phaseb_intent_depth(void) {
    std::lock_guard<std::mutex> lk(g_pb.int_mu);
    return g_pb.int_count;
}

// ── PB_FINALIZE_CUT ────────────────────────────────────────────────────────
void ggml_moe_phaseb_cut_close_capture(void) {
    // cap_mu 아래에서 세운다 — 이미 임계 구간에 들어간 capture 는 **전 관측 mutation** 을 마치고,
    // 그 뒤 진입하는 capture 는 반드시 래치를 본다(§0.1 ① 선형화점 · R4 등록/드레인 프로토콜).
    // ★프로덕션 경로 비용 = relaxed load 1회(이벤트 호출 0 · 잠금 형태도 기존 lock_guard 그대로).
    const bool rv = g_pb.test_cut_rv.load(std::memory_order_relaxed) != 0;
    if (!rv) {
        std::lock_guard<std::mutex> lk(g_pb.cap_mu);
        g_pb.cut_closed.store(true, std::memory_order_release);
        return;
    }
    // ★R6 테스트 무장 경로 — `attempted`(시도 직전)만으로는 "막혔다"의 양성 증거가 못 된다
    //   (SetEvent 직후 선점되면 실 contention 없이 지나갈 수 있다). try_lock 을 선행해
    //   **실패했을 때만** `contended` 를 세워, capture 가 cap_mu 를 쥔 상태에서 CUT 가 실제로
    //   시도해 막혔음을 양성으로 증명한다. 실패 시엔 통상 lock() 으로 그대로 진행한다.
    if (g_pb.cut_attempted) SetEvent((HANDLE)g_pb.cut_attempted);
    std::unique_lock<std::mutex> lk(g_pb.cap_mu, std::try_to_lock);
    if (!lk.owns_lock()) {
        if (g_pb.cut_contended) SetEvent((HANDLE)g_pb.cut_contended);        // ★양성 증거
        lk.lock();
    }
    if (g_pb.cut_acquired) SetEvent((HANDLE)g_pb.cut_acquired);              // 취득 **직후**
    g_pb.cut_closed.store(true, std::memory_order_release);
}

// ★R1 #3 ② — 이 래치가 서기 전에는 dispatcher graceful exit 금지(후발 intent 미처분 경합).
bool ggml_moe_phaseb_predictor_drained(void) {
    // worker가 애초에 없으면(프리페치 OFF·이미 join) drain 할 것이 없다 = 이미 완료.
    return !g_pb.worker_running.load(std::memory_order_acquire) ||
           g_pb.predictor_drained.load(std::memory_order_acquire);
}

// ★R2 #1 — cq_alloc(acquire 경로)도 **같은 래치**를 확인한다(mutex 불필요·atomic 1회 로드).
bool ggml_moe_phaseb_capture_closed(void) {
    return g_pb.cut_closed.load(std::memory_order_acquire);
}

// ② capture_q/predictor drain — worker가 잔여 capture 전건 소진·intent 발행 완결 후 stop/join.
// ★★R-2(FROZEN §2-2 "실패 후 자원 잔존 0" · 교차 r1 F1) — **실패한 seal 의 되돌림**.
//   `note_seal` 은 `sealed`·모델 배열·adapter·스냅샷 형상·capture/intent 자원·worker 를 **먼저
//   게시**한다(그것이 정상 경로의 설계다). 그래서 그 뒤 단계(metrics·dispatcher)가 실패하면
//   호출자는 게시된 것을 전부 되돌려야 §2-2 를 지킨다.
//   ★구 판의 되돌림은 direct 쪽 `pb_*` 사본과 predictor 정지뿐이었고 PhaseB 내부의
//     `sealed`·snapshot·`cap_slots`·`int_slots`·CUT 래치는 그대로 남았다 — 전건 초기화 수단이
//     test-only `reset_for_test` 밖에 없었다는 것이 교차 r1 F1 의 지적이다.
//   ★`reset_for_test` 와 다른 점: 카운터·digest·ledger 같은 **관측 이력은 건드리지 않는다**.
//     실패한 seal 이라도 그때까지의 진단 계수는 호출자가 로그로 낼 수 있어야 하고, 이 함수의
//     계약은 "**자원과 seal 상태**의 잔존 0"이지 이력 소거가 아니다.
void ggml_moe_phaseb_unseal_failed(void) {
    // ① 신규 capture 를 먼저 닫고 worker 를 접는다(정상 CUT ①②와 같은 순서).
    g_pb.cut_closed.store(true, std::memory_order_release);
    if (g_pb.worker_running.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(g_pb.cap_mu);
        g_pb.cap_idle_cv.wait_for(lk, std::chrono::milliseconds(10000),
                                  [] { return g_pb.cap_count == 0 && !g_pb.worker_busy; });
    }
    pb_worker_stop_join();
    // ② 자원 반납(자기 unwind 와 같은 역연산 — 사본 축소).
    pb_free_b2_resources();
    // ③ 스냅샷 형상 해제.
    g_pb.snap_router.clear();      g_pb.snap_router.shrink_to_fit();
    g_pb.snap_norm.clear();        g_pb.snap_norm.shrink_to_fit();
    g_pb.snap_bias.clear();        g_pb.snap_bias.shrink_to_fit();
    g_pb.snap_router_ok.clear();   g_pb.snap_norm_ok.clear();   g_pb.snap_bias_ok.clear();
    g_pb.w_norm_buf.clear();       g_pb.w_logit_buf.clear();    g_pb.w_used_buf.clear();
    g_pb.snap_configured = false;  g_pb.snap_sealed = false;
    // ④ seal 게시분 원복.
    //   ★정확한 재시도 계약(문면 정정) — 이건 "fresh seal 과 동일"이 **아니다**. `pf_resolved`
    //     는 프로세스 1회성 env 해소 캐시라 여기서 안 연다(`resolve_prefetch` 의 조기반환).
    //     따라서 같은 프로세스의 재시도는 **프리페치가 OFF 인 채로** 출발한다 — 되살아나지
    //     않는다. 그게 이 함수가 노리는 fail-safe 쪽이고(header 가 prefetch OFF 를 echo 하므로
    //     "ON 이라 적혀 있는데 producer 가 없는" 조용한 no-op 이 성립하지 않는다), 그래서
    //     `pf_resolved` 를 일부러 건드리지 않는다. §2-2 가 요구하는 것은 **자원**의 잔존 0 이지
    //     env 해소 캐시의 원복이 아니다.
    g_pb.sealed = false;
    g_pb.pf_on  = false;
    g_pb.obs_on = false;
    g_pb.moe_layers.clear();
    g_pb.strides.clear();
    g_pb.layer_to_mlpos.clear();
    g_pb.r_qpc_by_layer.clear();
    g_pb.pend_by_target.clear();
    g_pb.skip_by_target.clear();
    g_pb.ovf_by_target.clear();
    g_pb.cap_layer_seen.clear();
    g_pb.predictor_drained.store(false, std::memory_order_release);
    pb_adapter_clear();
    pb_refresh_all_off();
}

// T-1 - the bounded drain. 1 = drained, 0 = timeout.
// E-2a (Codex r2 blocker 2) - ONE budget, threaded through. This drain has two waits of its own
//   (the capture-idle wait, then the worker join) and the old form handed the caller's
//   timeout_ms to BOTH of them from scratch, so a remaining R could cost min(R,10s) + R and
//   still return success. Here the caller pins one absolute instant and each stage spends only
//   what is LEFT of it at its own entry. Nothing left = one check and no wait, which is the
//   same "budget 0" meaning drain_finish gives deadline_ms <= 0 one level up.
//   deadline_abs_ms < 0 = INFINITE (the production path, behaviour unchanged).
int ggml_moe_phaseb_cut_drain_predictor_until(int64_t deadline_abs_ms) {
    g_pb.drain_stage_n.store(0, std::memory_order_relaxed);   // start this drain's stage record
    if (g_pb.worker_running.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(g_pb.cap_mu);
        const int32_t left = pb_budget_left_ms(deadline_abs_ms);
        pb_note_drain_stage(/*idx=*/0, deadline_abs_ms, left);
        // The 10 s cap is unchanged - it keeps this one wait from swallowing the whole budget.
        // What changed is WHAT it caps: the remaining budget, not the caller's timeout_ms.
        const int32_t wait_ms = (left < 0) ? 10000 : (left < 10000 ? left : 10000);
        g_pb.cap_idle_cv.wait_for(lk, std::chrono::milliseconds(wait_ms),
                                  [] { return g_pb.cap_count == 0 && !g_pb.worker_busy; });
    }
    if (!pb_worker_stop_join_until(deadline_abs_ms)) return 0;   // timeout propagates as-is
    g_pb.predictor_drained.store(true, std::memory_order_release);
    return 1;
}

// Relative-timeout wrapper: pin one deadline at entry, and pass only that one downwards.
int ggml_moe_phaseb_cut_drain_predictor_bounded(int32_t timeout_ms) {
    return ggml_moe_phaseb_cut_drain_predictor_until(
        timeout_ms < 0 ? -1 : pb_steady_ms() + (int64_t)timeout_ms);
}

void ggml_moe_phaseb_cut_drain_predictor(void) {
    (void)ggml_moe_phaseb_cut_drain_predictor_until(/*deadline_abs_ms=*/-1);
}

// ── 카운터/digest ──────────────────────────────────────────────────────────
void ggml_moe_phaseb_count(int32_t ctr, uint32_t request_epoch, uint64_t delta) {
    if (ctr < 0 || ctr >= GGML_MOE_PB_C_COUNT || delta == 0) return;
    g_pb.ctr[ctr].fetch_add(delta, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    PbRunAcc & a = pb_run_acc_locked(request_epoch);
    a.c3[ctr] += delta;
    if (ctr == GGML_MOE_PB_C_INTENT_PREDICTED) a.intents += delta;
}

uint64_t ggml_moe_phaseb_counter(int32_t ctr) {
    if (ctr < 0 || ctr >= GGML_MOE_PB_C_COUNT) return 0;
    return g_pb.ctr[ctr].load(std::memory_order_relaxed);
}

void ggml_moe_phaseb_count_skip(int32_t reason, uint32_t request_epoch) {
    if (reason <= 0 || reason >= GGML_MOE_PB_SKIP_COUNT) return;
    g_pb.skip_ctr[reason].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    pb_run_acc_locked(request_epoch).skip[reason]++;
}

uint64_t ggml_moe_phaseb_skip_counter(int32_t reason) {
    if (reason < 0 || reason >= GGML_MOE_PB_SKIP_COUNT) return 0;
    return g_pb.skip_ctr[reason].load(std::memory_order_relaxed);
}

void ggml_moe_phaseb_count_split(uint32_t request_epoch, uint32_t required, uint32_t ready,
                                 uint32_t join, uint32_t demand_read, uint32_t join_attempt) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    PbRunAcc & a = pb_run_acc_locked(request_epoch);
    a.split_required += required; a.split_ready += ready; a.split_join += join;
    a.split_demand_read += demand_read; a.split_join_attempt += join_attempt;
}

void ggml_moe_phaseb_count_blocked(uint32_t request_epoch, uint64_t us, uint32_t count) {
    if (us == 0 && count == 0) return;
    g_pb.blocked_us_total.fetch_add(us, std::memory_order_relaxed);
    g_pb.blocked_count_total.fetch_add(count, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    PbRunAcc & a = pb_run_acc_locked(request_epoch);
    a.blocked_us += us; a.blocked_count += count;
}

void ggml_moe_phaseb_note_sidecar_record(int32_t file_kind, uint32_t request_epoch) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    PbRunAcc & a = pb_run_acc_locked(request_epoch);
    if      (file_kind == 1) a.sc_a0equiv++;
    else if (file_kind == 2) a.sc_sample++;
    else if (file_kind == 3) a.sc_io++;
    else if (file_kind == 4) a.sc_observe++;   // ★8호
}

void ggml_moe_phaseb_note_qd(uint32_t qd_demand, uint32_t qd_prefetch, uint32_t qd_total) {
    auto bump = [](std::atomic<uint32_t> & a, uint32_t v) {
        uint32_t cur = a.load(std::memory_order_relaxed);
        while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
    };
    bump(g_pb.qd_max_demand, qd_demand);
    bump(g_pb.qd_max_prefetch, qd_prefetch);
    bump(g_pb.qd_max_total, qd_total);
}

void ggml_moe_phaseb_qd_max(uint32_t * out_demand, uint32_t * out_prefetch, uint32_t * out_total) {
    if (out_demand)   *out_demand   = g_pb.qd_max_demand.load(std::memory_order_relaxed);
    if (out_prefetch) *out_prefetch = g_pb.qd_max_prefetch.load(std::memory_order_relaxed);
    if (out_total)    *out_total    = g_pb.qd_max_total.load(std::memory_order_relaxed);
}

void ggml_moe_phaseb_blocked_totals(uint64_t * out_us, uint64_t * out_count) {
    if (out_us)    *out_us    = g_pb.blocked_us_total.load(std::memory_order_relaxed);
    if (out_count) *out_count = g_pb.blocked_count_total.load(std::memory_order_relaxed);
}

void ggml_moe_phaseb_note_demand_ready(uint32_t graph_seq, int32_t layer, uint64_t r_qpc) {
    if (g_pb.all_off.load(std::memory_order_relaxed)) return;
    if (layer < 0) return;
    const int32_t ml = (layer < (int32_t)g_pb.layer_to_mlpos.size()) ? g_pb.layer_to_mlpos[(size_t)layer] : -1;
    std::lock_guard<std::mutex> lk(g_pb.tm_mu);   // ★R1 #11
    if (g_pb.mailbox_graph != graph_seq) return;
    if (ml < 0 || ml >= (int32_t)g_pb.r_qpc_by_layer.size()) return;
    if (g_pb.r_qpc_by_layer[(size_t)ml] == 0) g_pb.r_qpc_by_layer[(size_t)ml] = r_qpc;
}

// ★R1 #10 ②③ — 이 (graph, target 층)에 기록된 실제 skip 사유·overflow bit를 행 생성 시 회수.
void ggml_moe_phaseb_target_status(uint32_t graph_seq, int32_t target_mlpos,
                                   uint8_t * out_reason, uint8_t * out_overflow) {
    uint8_t r = 0, o = 0;
    {
        std::lock_guard<std::mutex> lk(g_pb.tm_mu);
        if (g_pb.mailbox_graph == graph_seq && target_mlpos >= 0 &&
            target_mlpos < (int32_t)g_pb.skip_by_target.size()) {
            r = g_pb.skip_by_target[(size_t)target_mlpos];
            o = g_pb.ovf_by_target[(size_t)target_mlpos];
        }
    }
    if (out_reason)   *out_reason   = r;
    if (out_overflow) *out_overflow = o;
}

// ★R2 #5 — dispatcher(g_mod.mutex 보유)가 소비하는 상태 큐 pop. 1=이벤트 있음.
int ggml_moe_phaseb_status_pop(uint32_t * out_graph_seq, int32_t * out_target_mlpos,
                               uint8_t * out_reason, uint8_t * out_overflow) {
    std::lock_guard<std::mutex> lk(g_pb.tm_mu);
    if (g_pb.status_count == 0) return 0;
    const PbState::PbStatusEvt e = g_pb.status_q[g_pb.status_head];
    g_pb.status_head = (g_pb.status_head + 1) % PbState::STATUS_RING;
    g_pb.status_count--;
    if (out_graph_seq)     *out_graph_seq     = e.graph_seq;
    if (out_target_mlpos)  *out_target_mlpos  = e.target_mlpos;
    if (out_reason)        *out_reason        = e.reason;
    if (out_overflow)      *out_overflow      = e.overflow;
    return 1;
}

void ggml_moe_phaseb_digest_disposition(uint64_t intent_ordinal, uint16_t rank, uint8_t disp) {
    uint64_t h = g_pb.dg_disp;
    h = fnv1a64(h, &intent_ordinal, sizeof(intent_ordinal));
    h = fnv1a64(h, &rank, sizeof(rank));
    h = fnv1a64(h, &disp, sizeof(disp));
    g_pb.dg_disp = h;
}

void ggml_moe_phaseb_digest_transition(uint8_t ev, uint64_t record_id, int32_t slot,
                                       uint64_t generation, uint64_t intent_ordinal) {
    uint64_t h = g_pb.dg_trans;
    h = fnv1a64(h, &ev, sizeof(ev));
    h = fnv1a64(h, &record_id, sizeof(record_id));
    h = fnv1a64(h, &slot, sizeof(slot));
    h = fnv1a64(h, &generation, sizeof(generation));
    h = fnv1a64(h, &intent_ordinal, sizeof(intent_ordinal));
    g_pb.dg_trans = h;
}

void ggml_moe_phaseb_digest_cacheclass(uint32_t idx, uint8_t cls, uint64_t intent_ordinal) {
    uint64_t h = g_pb.dg_class;
    h = fnv1a64(h, &idx, sizeof(idx));
    h = fnv1a64(h, &cls, sizeof(cls));
    h = fnv1a64(h, &intent_ordinal, sizeof(intent_ordinal));
    g_pb.dg_class = h;
}

void ggml_moe_phaseb_digest_readfile(uint16_t layer, uint16_t expert, uint64_t file_offset,
                                     uint32_t bytes, uint8_t origin) {
    uint64_t h = g_pb.dg_readfile;
    h = fnv1a64(h, &layer, sizeof(layer));
    h = fnv1a64(h, &expert, sizeof(expert));
    h = fnv1a64(h, &file_offset, sizeof(file_offset));
    h = fnv1a64(h, &bytes, sizeof(bytes));
    h = fnv1a64(h, &origin, sizeof(origin));
    g_pb.dg_readfile = h;
    g_pb.dg_readfile_ops++;
}

void ggml_moe_phaseb_digests(uint64_t * out_intent, uint64_t * out_disposition,
                             uint64_t * out_transition, uint64_t * out_cacheclass,
                             uint64_t * out_readfile, uint64_t * out_readfile_ops,
                             uint64_t * out_intent_count) {
    if (out_intent)       *out_intent       = g_pb.dg_intent;
    if (out_disposition)  *out_disposition  = g_pb.dg_disp;
    if (out_transition)   *out_transition   = g_pb.dg_trans;
    if (out_cacheclass)   *out_cacheclass   = g_pb.dg_class;
    if (out_readfile)     *out_readfile     = g_pb.dg_readfile;
    if (out_readfile_ops) *out_readfile_ops = g_pb.dg_readfile_ops;
    if (out_intent_count) *out_intent_count = g_pb.intent_count;
}

// ── 버퍼 상한 ──────────────────────────────────────────────────────────────
bool ggml_moe_phaseb_buffer_reserve(uint64_t add_bytes) {
    // ★5c-T(부록 14.4) — 수집은 PB 모드와 독립이므로 mode=off + collect=on 에서도 버퍼가 열린다.
    //   수집 OFF 런에서는 조건이 종전과 완전히 동일하다.
    if (g_pb.mode == GGML_MOE_PHASEB_OFF && !g_pb.collect_on) return false;
    const uint64_t cap = (uint64_t)g_pb.buf_mb * 1024ull * 1024ull;
    uint64_t cur = g_pb.buf_used.load(std::memory_order_relaxed);
    for (;;) {
        if (cur + add_bytes > cap) {
            // 상한 초과 = 레코드 drop(§2 인벤토리 #5) — 사유 비트 선행 후 complete 하강.
            ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_RECORD_DROP);
            return false;
        }
        if (g_pb.buf_used.compare_exchange_weak(cur, cur + add_bytes, std::memory_order_relaxed)) return true;
    }
}

// ★권고 ⓒ(R3) — 이 값은 **계약상 charge 합계**(레코드 폭 + 인덱스 노드 보수 추정)이지
//   allocator 실측치가 아니다. §7.2 게이트는 `used <= cap` 만 요구하므로 charge 기준으로 충분하다.
uint64_t ggml_moe_phaseb_buffer_used(void) { return g_pb.buf_used.load(std::memory_order_relaxed); }
uint64_t ggml_moe_phaseb_buffer_cap(void)  { pb_resolve_env(); return (uint64_t)g_pb.buf_mb * 1024ull * 1024ull; }
// ★acquire — 판독기가 complete=false 를 본 순간 그 원인 비트도 **반드시 보이도록** mark_incomplete
//   의 release store 와 짝을 이룬다(§4.2 "비트 없이 complete 만 0 인 창" 금지).
bool     ggml_moe_phaseb_complete(void)    { return g_pb.complete.load(std::memory_order_acquire); }
// ★사유 인자 필수 단일 API(§4.2 M3) — atomic OR 가 complete=false 보다 **선행**한다.
void     ggml_moe_phaseb_mark_incomplete(uint32_t reason_bit) {
    g_pb.incomplete_mask.fetch_or(reason_bit, std::memory_order_relaxed);
    g_pb.complete.store(false, std::memory_order_release);
}
uint32_t ggml_moe_phaseb_incomplete_mask(void) { return g_pb.incomplete_mask.load(std::memory_order_relaxed); }
uint64_t ggml_moe_phaseb_dropped_records(void) { return g_pb.dropped.load(std::memory_order_relaxed); }
void     ggml_moe_phaseb_note_dropped(uint64_t n) {
    if (n == 0) return;
    g_pb.dropped.fetch_add(n, std::memory_order_relaxed);
    // ★§4.2 — note_dropped 도 내부에서 RECORD_DROP 을 세운 뒤 complete 를 내린다(호출부 무변경).
    ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_RECORD_DROP);
}
bool     ggml_moe_phaseb_run_invalid(void) { return g_pb.run_invalid.load(std::memory_order_relaxed); }

} // extern "C"

// ===========================================================================
// §6. 사이드카 파일 기록 (§2 공통 헤더 · CREATE_NEW · 전량 메모리 버퍼 후 finalize flush)
// ===========================================================================

namespace {

const char * pb_kind_tag(int32_t kind) {
    switch (kind) {
        case 1: return "a0_equiv";
        case 2: return "PB_SAMPLE_V1";
        case 3: return "PB_IO_V1";
        case 4: return "PB_OBSV_V2";   // ★8호(§13.1) → ★FROZEN_PB_OBSV_V2 §2.1 버전 상승(V1→V2)
        case 5: return "PB_LEDGER_V1"; // ★5c-T(부록 14.3)
        default: return "";
    }
}
const char * pb_kind_suffix(int32_t kind) {
    switch (kind) {
        case 1: return "_a0equiv.pb";
        case 2: return "_sample.pb";
        case 3: return "_io.pb";
        case 4: return "_observe.pb";  // ★8호(§1.1)
        case 5: return "_ledger.pb";   // ★5c-T(부록 14.3)
        default: return "_unknown.pb";
    }
}

// ★모드↔파일 허용 조합(§1.1 · §9 게이트 1-c) — 생산 측에서 구조로 잠근다:
//   trace = sample+io / equiv = 3종 / observe = observe 1종 정확히(sample·io·a0equiv 부재).
bool pb_kind_allowed(int32_t kind) {
    // ★5c-T(부록 14.3) — `pb_ledger`(kind 5)는 **수집 축** 전용이며 PB 모드와 직교한다.
    //   수집 OFF 에서는 어떤 모드에서도 생성되지 않는다(기존 조합 표 무변경).
    if (kind == 5) return g_pb.collect_on;
    switch (g_pb.mode) {
        case GGML_MOE_PHASEB_TRACE:   return kind == 2 || kind == 3;
        case GGML_MOE_PHASEB_EQUIV:   return kind == 1 || kind == 2 || kind == 3;
        case GGML_MOE_PHASEB_OBSERVE: return kind == 4;
        default:                      return false;
    }
}

bool pb_write_file(const std::string & path, const ggml_moe_phaseb_header & h,
                   const void * payload, size_t payload_bytes) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0);
    std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), &w[0], n);
    HANDLE f = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        pb_log("cannot create sidecar (CREATE_NEW) path=%s err=%lu", path.c_str(), (unsigned long)GetLastError());
        return false;
    }
    DWORD wr = 0;
    bool ok = WriteFile(f, &h, sizeof(h), &wr, nullptr) != 0 && wr == sizeof(h);
    if (ok && payload_bytes) {
        size_t off = 0;
        const uint8_t * p = (const uint8_t *)payload;
        while (off < payload_bytes) {
            const DWORD chunk = (DWORD)std::min<size_t>(payload_bytes - off, (size_t)32 * 1024 * 1024);
            if (!WriteFile(f, p + off, chunk, &wr, nullptr) || wr != chunk) { ok = false; break; }
            off += chunk;
        }
    }
    if (ok) ok = FlushFileBuffers(f) != 0;
    CloseHandle(f);
    if (!ok) pb_log("cannot write sidecar path=%s err=%lu", path.c_str(), (unsigned long)GetLastError());
    return ok;
}

} // namespace

extern "C" {

int ggml_moe_phaseb_write_sidecar(int32_t file_kind, const void * payload,
                                  uint64_t record_count, uint32_t record_bytes,
                                  uint64_t * out_digest) {
    const uint64_t d = fnv1a64(FNV_INIT, payload, (size_t)(record_count * record_bytes));
    if (out_digest) *out_digest = d;
    // ★5c-T — kind 5(ledger)는 수집 축이라 mode=off 에서도 기록 대상이다(부록 14.3·14.4).
    //   kind 1~4 의 조건은 종전과 문자 그대로 동일하다.
    if ((g_pb.mode == GGML_MOE_PHASEB_OFF && file_kind != 5) || g_pb.out_prefix.empty()) return 0;
    if (!pb_kind_allowed(file_kind)) return 0;   // ★모드↔파일 조합(§1.1) — 허용 밖은 미기록

    ggml_moe_phaseb_header h{};
    std::memcpy(h.magic, "MOEPB1\0\0", 8);
    h.layout_version = 1;
    h.file_kind      = (uint32_t)file_kind;
    h.record_bytes   = record_bytes;
    h.phaseb_mode    = (uint32_t)g_pb.mode;
    h.record_count   = record_count;
    h.payload_digest = d;
    h.qpc_frequency  = g_pb.qpc_freq;
    h.t0_qpc         = g_pb.t0_qpc;
    h.pid            = (uint32_t)GetCurrentProcessId();
    h.complete       = g_pb.complete.load(std::memory_order_relaxed) ? 1u : 0u;
    h.n_expert       = (uint32_t)g_pb.n_expert;
    h.n_moe_layers   = (uint32_t)g_pb.n_moe_layers;
    h.n_slots        = (uint32_t)g_pb.n_slots;
    h.n_expert_used  = (uint32_t)g_pb.n_expert_used;
    h.boot_id        = g_pb.boot_id;
    h.total_requests = g_pb.total_requests;
    h.dropped_records= g_pb.dropped.load(std::memory_order_relaxed);
    h.prefetch_on    = g_pb.pf_on ? 1 : 0;
    h.prefetch_k     = (uint8_t)g_pb.pf_k;
    h.prefetch_n     = (uint8_t)g_pb.pf_n;
    h.qd_effective   = (uint8_t)g_pb.qd_effective;
    std::memset(h.kind_tag, 0, sizeof(h.kind_tag));
    const char * tag = pb_kind_tag(file_kind);
    std::memcpy(h.kind_tag, tag, std::min<size_t>(std::strlen(tag), sizeof(h.kind_tag)));
    // ★§13.1 헤더 delta — **kind 4 파일에만** 정의값을 싣는다. kind 1~3 은 off 120~127 이 전부 0
    //   으로 남아 구파일·기존 게이트 19("reserved 전부 0")와 바이트 단위로 동일하다.
    if (file_kind == 4) {
        h.signal_observe_on = 1;
        h.adapter_id        = (uint8_t)g_pb.ad_id;
        h.capture_width     = (uint16_t)g_pb.capture_width;
    }

    const std::string path = g_pb.out_prefix + pb_kind_suffix(file_kind);
    return pb_write_file(path, h, payload, (size_t)(record_count * record_bytes)) ? 1 : 0;
}

int ggml_moe_phaseb_write_a0equiv(uint64_t * out_records, uint64_t * out_digest) {
    if (out_records) *out_records = (uint64_t)g_pb.equiv_recs.size();
    // trace 모드에서는 a0_equiv를 기록하지 않는다(§1.1 모드↔파일 허용 조합).
    if (g_pb.mode != GGML_MOE_PHASEB_EQUIV) {
        if (out_digest) *out_digest = FNV_INIT;
        if (out_records) *out_records = 0;
        return 0;
    }
    return ggml_moe_phaseb_write_sidecar(1, g_pb.equiv_recs.data(), (uint64_t)g_pb.equiv_recs.size(),
                                         (uint32_t)sizeof(ggml_moe_phaseb_a0equiv), out_digest);
}

// ★observe 사이드카(모듈 소유 버퍼) — mode!=observe 면 미기록·0건(§1.1 파일 조합).
int ggml_moe_phaseb_write_observe(uint64_t * out_records, uint64_t * out_digest) {
    pb_observe_tally();   // ★flush 직전 1회 — 레코드 전수 분류 확정(§13.4 closure)
    std::lock_guard<std::mutex> lk(g_pb.obs_mu);
    if (g_pb.mode != GGML_MOE_PHASEB_OBSERVE) {
        if (out_digest)  *out_digest  = FNV_INIT;
        if (out_records) *out_records = 0;
        return 0;
    }
    if (out_records) *out_records = (uint64_t)g_pb.obs_recs.size();
    return ggml_moe_phaseb_write_sidecar(4, g_pb.obs_recs.data(), (uint64_t)g_pb.obs_recs.size(),
                                         (uint32_t)sizeof(ggml_moe_phaseb_observe), out_digest);
}

// ★5c-T `pb_ledger` 사이드카(부록 14.3) — 수집 ON 에서만 버퍼링·기록. 기존 사이드카 계약 동형
//   (전량 메모리 버퍼 → finalize 1회 flush · CREATE_NEW · 공통 128B 헤더 · buf 상한 청구).
bool ggml_moe_phaseb_collect_on(void) { return g_pb.collect_on; }

// ★5c-A1 adapt 축(§1-6 · §6-3) — seal 해소된 실효값. 발급자가 없는 5c-A1 런에서는 false 다.
bool ggml_moe_phaseb_adapt_on(void) { return g_pb.adapt_on; }

// ★§1-6-1 — view snapshot 이 accepted 인가(V-1~V-3 전건 통과). seal 전에는 false.
bool ggml_moe_phaseb_bundle_view_accepted(void) { return g_pb.view.accepted; }

// ★§6-3 — `activation_grant` verifier 가 멈춘 단의 거부 사유(통과 시 빈 문자열).
//   활성화 거부는 침묵이 아니라 **이름 있는 사유**로 관측된다(§1-7 activation namespace).
const char * ggml_moe_phaseb_activation_grant_refusal(void) { return g_pb.grant_refusal.c_str(); }

// ★§6-3 — controller 객체 할당 계수(읽기 전용). 5c-A1 은 그 객체를 만들지 않으므로 0 이다.
uint64_t ggml_moe_phaseb_controller_alloc_count(void) {
    return g_pb.controller_alloc.load(std::memory_order_relaxed);
}

uint64_t ggml_moe_phaseb_ledger_next_event_seq(void) {
    return g_pb.led_event_seq.fetch_add(1, std::memory_order_relaxed);
}

int ggml_moe_phaseb_ledger_push(const struct ggml_moe_phaseb_ledger * rec) {
    if (!g_pb.collect_on || rec == nullptr) return 0;
    return ggml_moe_phaseb_ledger_push_group(rec, 1);
}

// ★r1 M4 — 연속 레코드 그룹의 **원자 적재**. 그룹 전량을 한 번에 예약해, 실패하면 한 건도
//   적지 않고 0 을 돌려준다. 부분 방출은 "완전한 참여 집합"으로 오독되므로 금지하며, 드롭은
//   `note_dropped` 가 RECORD_DROP 비트를 세워 사이드카 헤더 `complete=0` 으로 fail-close 된다.
int ggml_moe_phaseb_ledger_push_group(const struct ggml_moe_phaseb_ledger * recs, uint32_t n_recs) {
    if (!g_pb.collect_on || recs == nullptr || n_recs == 0) return 0;
    const uint64_t need = (uint64_t)n_recs * sizeof(ggml_moe_phaseb_ledger);
    if (!ggml_moe_phaseb_buffer_reserve(need)) {
        ggml_moe_phaseb_note_dropped(n_recs);
        return 0;
    }
    std::lock_guard<std::mutex> lk(g_pb.led_mu);
    g_pb.led_recs.insert(g_pb.led_recs.end(), recs, recs + n_recs);
    return 1;
}

uint64_t ggml_moe_phaseb_ledger_records(void) {
    std::lock_guard<std::mutex> lk(g_pb.led_mu);
    return (uint64_t)g_pb.led_recs.size();
}

int ggml_moe_phaseb_write_ledger(uint64_t * out_records, uint64_t * out_digest) {
    std::lock_guard<std::mutex> lk(g_pb.led_mu);
    if (!g_pb.collect_on) {
        if (out_digest)  *out_digest  = FNV_INIT;
        if (out_records) *out_records = 0;
        return 0;
    }
    if (out_records) *out_records = (uint64_t)g_pb.led_recs.size();
    return ggml_moe_phaseb_write_sidecar(5, g_pb.led_recs.data(), (uint64_t)g_pb.led_recs.size(),
                                         (uint32_t)sizeof(ggml_moe_phaseb_ledger), out_digest);
}

// ★★R-2 게이트 — predictor worker 생성 실패 1회 주입(`pb_worker_start` 가 소비한다).
void ggml_moe_phaseb_test_worker_start_fail_next(int32_t on) {
    g_pb.test_worker_fail_next.store(on ? 1 : 0, std::memory_order_relaxed);
}

void ggml_moe_phaseb_test_submit_fail_next(int32_t on) {
    g_pb.test_submit_fail_next.store(on ? 1 : 0, std::memory_order_relaxed);
}
int ggml_moe_phaseb_test_submit_fail_take(void) {
    if (g_pb.test_submit_fail_next.load(std::memory_order_relaxed) == 0) return 0;
    g_pb.test_submit_fail_next.store(0, std::memory_order_relaxed);   // one-shot
    return 1;
}

void ggml_moe_phaseb_test_stamp_override(int32_t k, int32_t n, int32_t qd) {
    g_pb.test_stamp_k.store(k, std::memory_order_relaxed);
    g_pb.test_stamp_n.store(n, std::memory_order_relaxed);
    g_pb.test_stamp_qd.store(qd, std::memory_order_relaxed);
}

void ggml_moe_phaseb_test_dup_terminal_arm(int32_t on) {
    g_pb.test_dup_term_arm.store(on ? 1 : 0, std::memory_order_relaxed);
    if (on) {
        g_pb.test_dup_term_hits.store(0, std::memory_order_relaxed);
        g_pb.test_dup_term_status.store(0, std::memory_order_relaxed);
        g_pb.test_dup_term_ctr.store(0, std::memory_order_relaxed);
        g_pb.test_dup_term_led.store(0, std::memory_order_relaxed);
        g_pb.test_dup_term_qd.store(0, std::memory_order_relaxed);
    }
}
int ggml_moe_phaseb_test_dup_terminal_take(void) {
    if (g_pb.test_dup_term_arm.load(std::memory_order_relaxed) == 0) return 0;
    g_pb.test_dup_term_arm.store(0, std::memory_order_relaxed);   // one-shot
    return 1;
}
void ggml_moe_phaseb_test_dup_terminal_note(int32_t status, uint64_t ctr_delta,
                                            uint64_t ledger_delta, int32_t qd_delta) {
    g_pb.test_dup_term_hits.fetch_add(1, std::memory_order_relaxed);
    g_pb.test_dup_term_status.store(status, std::memory_order_relaxed);
    g_pb.test_dup_term_ctr.store(ctr_delta, std::memory_order_relaxed);
    g_pb.test_dup_term_led.store(ledger_delta, std::memory_order_relaxed);
    g_pb.test_dup_term_qd.store(qd_delta, std::memory_order_relaxed);
}
int ggml_moe_phaseb_test_dup_terminal_result(int32_t * out_status, uint64_t * out_ctr_delta,
                                             uint64_t * out_ledger_delta, int32_t * out_qd_delta) {
    if (out_status)       *out_status       = g_pb.test_dup_term_status.load(std::memory_order_relaxed);
    if (out_ctr_delta)    *out_ctr_delta    = g_pb.test_dup_term_ctr.load(std::memory_order_relaxed);
    if (out_ledger_delta) *out_ledger_delta = g_pb.test_dup_term_led.load(std::memory_order_relaxed);
    if (out_qd_delta)     *out_qd_delta     = g_pb.test_dup_term_qd.load(std::memory_order_relaxed);
    return g_pb.test_dup_term_hits.load(std::memory_order_relaxed);
}

int ggml_moe_phaseb_test_ledger_record(uint64_t idx, void * out_record, uint32_t bytes) {
    if (!out_record || bytes < sizeof(ggml_moe_phaseb_ledger)) return -1;
    std::lock_guard<std::mutex> lk(g_pb.led_mu);
    if (idx >= (uint64_t)g_pb.led_recs.size()) return -1;
    std::memcpy(out_record, &g_pb.led_recs[(size_t)idx], sizeof(ggml_moe_phaseb_ledger));
    return 0;
}

uint64_t ggml_moe_phaseb_next_decision_seq(uint32_t request_epoch) {
    const uint64_t s = g_pb.decision_seq_next.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    PbRunAcc & a = pb_run_acc_locked(request_epoch);
    if (a.decision_seq_first == GGML_MOE_PHASEB_NA_U64) a.decision_seq_first = s;
    a.decision_seq_last = s;
    return s;
}

// ★r1 M1 — owner generation 의 admission 순번(스탬프 원소). owner request_epoch 로만 읽으며
//   ambient 현재 요청을 보지 않는다. 미발급(수집 OFF·lifecycle 밖) = NA.
uint64_t ggml_moe_phaseb_run_admission_seq(uint32_t request_epoch) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    return pb_run_acc_locked(request_epoch).admission_seq;
}

// ★r1 M2 — owner generation 의 **박제 튜플**(K/N/QD/tuple_revision)을 owner epoch 로 되읽는다.
//   run_begin 이 admission 시점에 고정한 값이므로 이후 어떤 조정으로도 바뀌지 않는다.
void ggml_moe_phaseb_run_stamp(uint32_t request_epoch, int32_t * out_k, int32_t * out_n,
                               int32_t * out_qd, int32_t * out_rev, uint64_t * out_admission_seq) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    const PbRunAcc & a = pb_run_acc_locked(request_epoch);
    if (out_k)   *out_k   = a.stamp_k;
    if (out_n)   *out_n   = a.stamp_n;
    if (out_qd)  *out_qd  = a.stamp_qd;
    if (out_rev) *out_rev = a.stamp_rev;
    if (out_admission_seq) *out_admission_seq = a.admission_seq;
}

void ggml_moe_phaseb_note_blocked_interval(uint32_t request_epoch) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    pb_run_acc_locked(request_epoch).blocked_intervals++;
}

// ★부록 14.2 — 서버 release 시점 outcome 전달. **현 요청**(run_end 직전) 귀속이며 엔진은
//   판정하지 않는다(전달값 그대로 기록). 미호출 = unknown 유지.
void ggml_moe_phaseb_run_outcome(int32_t outcome) {
    if (!(g_pb.collect_on || g_pb.adapt_on)) return;
    if (outcome < GGML_MOE_PB_OUTCOME_UNKNOWN || outcome > GGML_MOE_PB_OUTCOME_FAILED) return;
    const uint32_t req = g_pb.request_index.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    pb_run_acc_locked(req).outcome = outcome;
}

int ggml_moe_phaseb_test_run_outcome(uint32_t request_epoch) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    return pb_run_acc_locked(request_epoch).outcome;
}

uint64_t ggml_moe_phaseb_test_run_blocked_intervals(uint32_t request_epoch) {
    std::lock_guard<std::mutex> lk(g_pb.run_mu);
    return pb_run_acc_locked(request_epoch).blocked_intervals;
}

void ggml_moe_phaseb_emit_runs(const char * a0equiv_json, const char * sample_json,
                               const char * io_json, const char * observe_json, int32_t complete,
                               uint64_t dropped, uint64_t buffer_bytes) {
    char bid[32];
    snprintf(bid, sizeof(bid), "0x%016llx", (unsigned long long)g_pb.boot_id);
    static const char * kSkipName[GGML_MOE_PB_SKIP_COUNT] = {
        "NONE", "SOURCE_RANGE", "PREFILL", "MULTI_TOKEN", "LORA_SWAP", "SRC1_TYPE",
        "NONCONTIG", "NONFINITE", "CAPTURE_QUEUE_FULL", "REQUEST_DISABLED", "PREFETCH_OFF", "NO_SOURCE",
        "NCAP_ZERO",   // ★★PB virtual — SPEC_PB_VIRTUAL_RUNTIME_R1 §2-0 A0-5
    };
    // c3 블록 = §7.4 열거(★wasted·capacity_cancel_attempt 제외 — 둘 다 summary 전용).
    static const std::pair<int, const char *> kC3[] = {
        { GGML_MOE_PB_C_CANDIDATE,                 "candidate" },
        { GGML_MOE_PB_C_ADMITTED,                  "admitted" },
        { GGML_MOE_PB_C_ISSUED,                    "issued" },
        { GGML_MOE_PB_C_SUBMIT_FAILED,             "submit_failed" },
        { GGML_MOE_PB_C_DROPPED_NO_SLOT_QD,        "dropped_no_slot_qd" },
        { GGML_MOE_PB_C_DROPPED_PREFETCH_CAP,      "dropped_prefetch_cap" },
        { GGML_MOE_PB_C_DROPPED_RESERVE,           "dropped_reserve" },
        { GGML_MOE_PB_C_PRE_DROPPED_TOO_LATE,      "pre_dropped_too_late" },
        { GGML_MOE_PB_C_PRE_DROPPED_STALE,         "pre_dropped_stale" },
        { GGML_MOE_PB_C_PRE_SKIPPED_RESIDENT,      "pre_skipped_resident" },
        { GGML_MOE_PB_C_PRE_SKIPPED_PENDING,       "pre_skipped_pending" },
        { GGML_MOE_PB_C_USED_JOIN,                 "used_join" },
        { GGML_MOE_PB_C_USED_PROMOTED,             "used_promoted" },
        { GGML_MOE_PB_C_INTENT_OVERFLOW,           "intent_overflow" },
        { GGML_MOE_PB_C_PREFETCH_ERROR_UNPUBLISH,  "prefetch_error_unpublish" },
        { GGML_MOE_PB_C_PREFETCH_RETRY,            "prefetch_retry" },
        { GGML_MOE_PB_C_DEMAND_PREFLIGHT_REJECT,   "demand_preflight_reject" },
    };
    // ★5c-T 수집 전용 c3 확장(부록 14.1 5종 · 수집 ON 에서만 방출).
    static const std::pair<int, const char *> kC3T[] = {
        { GGML_MOE_PB_C_RESOLVED_WASTED,           "resolved_wasted" },
        { GGML_MOE_PB_C_CENSORED,                  "censored" },
        { GGML_MOE_PB_C_ELIGIBLE_DEMAND_ACQUIRE,   "eligible_demand_acquire" },
        { GGML_MOE_PB_C_WAITED_ACQUIRE,            "waited_acquire" },
        { GGML_MOE_PB_C_OBSERVED_FRONTIER_MATCH,   "observed_frontier_match" },
    };
    std::vector<PbRunAcc> snapshot;
    {
        std::lock_guard<std::mutex> lk(g_pb.run_mu);
        snapshot = g_pb.runs;
    }
    json ja0 = json::parse(a0equiv_json ? a0equiv_json : "{}", nullptr, false);
    json jsm = json::parse(sample_json  ? sample_json  : "{}", nullptr, false);
    json jio = json::parse(io_json      ? io_json      : "{}", nullptr, false);
    json job = json::parse(observe_json ? observe_json : "{}", nullptr, false);   // ★8호
    for (size_t r = 0; r < snapshot.size(); r++) {
        const PbRunAcc & a = snapshot[r];
        if (!a.used) continue;
        json j;
        j["phaseb_mode"]        = ggml_moe_phaseb_mode_name();
        j["phaseb_boot_id"]     = std::string(bid);
        j["request_index"]      = (uint64_t)r;
        j["schedule_perturbed"] = (g_pb.mode == GGML_MOE_PHASEB_EQUIV);
        json counts;
        counts["prefetch_on"] = g_pb.pf_on ? 1 : 0;
        counts["prefetch_k"]  = g_pb.pf_k;
        counts["prefetch_n"]  = g_pb.pf_n;
        counts["intents"]     = a.intents;
        counts["test_hint_applied"] = a.test_hint_applied;   // ★정정 6호(§7.4 · 요청 델타·비활성 0)
        json jskip = json::object();
        for (int i = 1; i < GGML_MOE_PB_SKIP_COUNT; i++) jskip[kSkipName[i]] = a.skip[i];
        counts["elig_skip"] = jskip;
        json jc3 = json::object();
        for (const auto & kv : kC3) jc3[kv.second] = a.c3[kv.first];
        // ★5c-T(부록 14.2 "c3 확장") — 수집 ON 에서만 5종을 c3 블록에 병기한다.
        //   수집 OFF 런에서는 키 자체가 없어 종전 바이트와 동일하다.
        if (g_pb.collect_on) {
            for (const auto & kv : kC3T) jc3[kv.second] = a.c3[kv.first];
        }
        counts["c3"] = jc3;
        counts["split"] = json{ {"required", a.split_required}, {"ready", a.split_ready},
                                {"join", a.split_join}, {"demand_read", a.split_demand_read},
                                {"join_attempt", a.split_join_attempt} };
        counts["sidecar_records"] = json{ {"a0equiv", a.sc_a0equiv}, {"sample", a.sc_sample},
                                          {"io", a.sc_io}, {"observe", a.sc_observe} };
        j["counts"] = counts;
        json timing;
        timing["predictor_exposed_us"] = json{ {"sum", a.exp_sum}, {"max", a.exp_max}, {"n", a.exp_n} };
        timing["blocked_by_prefetch_us"]    = a.blocked_us;
        timing["blocked_by_prefetch_count"] = a.blocked_count;
        // ★B — replay U18 입력(프로세스 전역 단일 최댓값·warmup 포함). 요청별 분해가 아니라
        //   "이 런의 predictor cost 상한"이므로 전 레코드에 같은 값이 실린다(어느 레코드로도
        //   소비 가능). 0 표본이면 samples=0 이며 소비자는 **0ns 대체 없이 NOT_RUN** 이어야 한다.
        timing["observe_predictor_cost_ns_max"]  = g_pb.obs_cost_ns_max.load(std::memory_order_relaxed);
        timing["observe_predictor_cost_samples"] = g_pb.obs_cost_samples.load(std::memory_order_relaxed);
        j["timing"] = timing;
        // ★NO_OUTPUT 권위 표면(FROZEN_AUTHORITY_V3 §3.2.1 A5) — 요청별 배열. **summary 카운터로
        //   대체할 수 없는 유일 권위**이며, 원소는 exact object
        //   `{graph_seq, decode_ordinal, target_layer, n_outputs:0}` 다. `request_index` 는 이 부모
        //   레코드가 소유하므로 원소에 중복 탑재하지 않는다. zero-output 이 없는 요청은 **빈 배열**
        //   이다(키 자체는 항상 존재 — 판독기의 결손 검출용).
        json jno = json::array();
        for (const auto & e : a.no_output) {
            jno.push_back(json{ { "graph_seq",      e.graph_seq },
                                { "decode_ordinal", e.decode_ordinal },
                                { "target_layer",   e.target_layer },
                                { "n_outputs",      0 } });
        }
        j["no_output_graphs"] = jno;
        // ★5c-T additive(부록 14.2) — 수집 ON 에서만 존재하는 최상위 3키.
        //   기존 소비자는 미지 키를 무시하고, 수집 OFF 런에는 키 자체가 없다.
        if (g_pb.collect_on || g_pb.adapt_on) {
            // controller §1 불변 스탬프 10필드 **전부 의무**. 5c-T 시점의 명시 N/A = JSON null
            //   (필드 부재가 아니다) · constants_version 은 전원 unsealed.
            json stamp;
            stamp["controller_version"] = std::string(GGML_MOE_PB_CONTROLLER_VERSION);
            stamp["constants_version"]  = std::string("unsealed");
            stamp["session_id"]         = std::string(bid);   // = phaseb_boot_id
            stamp["epoch_id"]           = nullptr;            // 5c-T = online epoch 없음(명시 N/A)
            stamp["tuple_revision"]     = a.stamp_rev;
            stamp["K"]                  = a.stamp_k;    // ★owner 박제값(ambient 재조회 아님)
            stamp["N"]                  = a.stamp_n;
            stamp["QD"]                 = a.stamp_qd;
            // ★r1 M1/M6 — controller §1 이 요구하는 **단일 generation `admission_seq`**.
            //   구 판의 `admission_seq_range`(candidate 범위)는 상위 사양을 덮어쓰는 오용이었다.
            if (a.admission_seq == GGML_MOE_PHASEB_NA_U64) stamp["admission_seq"] = nullptr;
            else                                           stamp["admission_seq"] = a.admission_seq;
            stamp["parent_action_id"]   = nullptr;            // actuation 전면 OFF(명시 N/A)
            j["stamp"] = stamp;
            // generation lifecycle(5c-T §4-① "admission/종결 순번·시각") — stamp 밖 별도 블록.
            json life;
            life["admission_seq"] = a.admission_seq == GGML_MOE_PHASEB_NA_U64
                                        ? json(nullptr) : json(a.admission_seq);
            life["admission_qpc"] = a.admission_qpc;
            life["terminal_seq"]  = a.terminal_seq == GGML_MOE_PHASEB_NA_U64
                                        ? json(nullptr) : json(a.terminal_seq);
            life["terminal_qpc"]  = a.terminal_qpc;
            life["decision_seq_range"] = a.decision_seq_first == GGML_MOE_PHASEB_NA_U64
                                        ? json(nullptr)
                                        : json(json::array({ a.decision_seq_first, a.decision_seq_last }));
            j["generation_lifecycle"] = life;
            static const char * kOutcome[4] = { "unknown", "normal", "cancel", "failed" };
            const int oc = (a.outcome >= 0 && a.outcome <= 3) ? a.outcome : 0;
            j["outcome"] = std::string(kOutcome[oc]);
            j["blocked_intervals"] = a.blocked_intervals;
            // ★5c-A1 additive(§1-6 · §3-2) — accepted view 의 `identity_digest` 를 **최상위**에
            //   echo 한다(stamp 안이 아니다). 엔진은 이 값을 만들지 않고 **옮기기만** 하며,
            //   accepted view 가 없으면 명시 N/A = JSON null 이다(키 부재가 아니다).
            j["prefetch_identity_digest"] = g_pb.view.accepted ? json(g_pb.view.digest)
                                                               : json(nullptr);
        }
        json sc;
        sc["a0equiv"] = ja0.is_object() ? ja0 : json::object();
        sc["sample"]  = jsm.is_object() ? jsm : json::object();
        sc["io"]      = jio.is_object() ? jio : json::object();
        sc["observe"] = job.is_object() ? job : json::object();   // ★8호(§13.3 게이트 5 준용)
        sc["complete"] = complete;
        sc["dropped_records"] = dropped;
        sc["buffer_bytes"] = buffer_bytes;
        sc["out_prefix"] = g_pb.out_prefix;
        j["sidecar"] = sc;
        ggml_moe_direct_metrics_write_record("phase_b_run", j.dump().c_str());
    }
}

// ★§7.2 규칙 분할(8호) — capture 풀은 signal-path 축(prefetch_on || signal_observe_on).
uint32_t ggml_moe_phaseb_capture_pool_slots(void) {
    return ((g_pb.pf_on || g_pb.obs_on) && !g_pb.cap_slots.empty())
           ? (uint32_t)GGML_MOE_PHASEB_CAPTURE_SLOTS : 0u;
}
uint32_t ggml_moe_phaseb_intent_q_slots(void) {   // speculative 축 전용(observe 에서 0)
    return (g_pb.pf_on && !g_pb.int_slots.empty()) ? (uint32_t)GGML_MOE_PHASEB_INTENT_SLOTS : 0u;
}

void ggml_moe_phaseb_mem_echo(uint64_t * out_capture_slot_bytes, uint64_t * out_capture_pool_bytes,
                              uint64_t * out_router_bytes, uint64_t * out_norm_bytes,
                              uint64_t * out_intent_record_bytes, uint64_t * out_intent_q_bytes,
                              uint32_t * out_predictor_threads, uint64_t * out_predictor_stack,
                              uint64_t * out_capture_overflow, uint64_t * out_intent_overflow) {
    // ★§7.2 규칙 분할(8호 · SPEC_PHASEB_WIRE §13.4): signal-path(capture/스냅샷/predictor)는
    //   `prefetch_on || signal_observe_on` · intent_q 는 `prefetch_on` 만.
    const bool sig = (g_pb.pf_on || g_pb.obs_on) && !g_pb.cap_slots.empty();
    const bool spec_on = g_pb.pf_on && !g_pb.int_slots.empty();
    const uint64_t slot_bytes = sig
        ? (uint64_t)(sizeof(PbCaptureSlot) + (size_t)g_pb.capture_width * sizeof(float)) : 0;
    const uint64_t irec = spec_on ? (uint64_t)sizeof(ggml_moe_phaseb_intent) : 0;
    if (out_capture_slot_bytes)   *out_capture_slot_bytes   = slot_bytes;
    if (out_capture_pool_bytes)   *out_capture_pool_bytes   = slot_bytes * (sig ? GGML_MOE_PHASEB_CAPTURE_SLOTS : 0);
    // ★R1 #5 ① — 두 축 모두 OFF = B2 확장 블록 **전부 0**(슬롯 수 포함 — §7.2 게이트 ⓐ).
    (void)0;
    if (out_router_bytes)         *out_router_bytes         = sig ? ggml_moe_phaseb_router_snapshot_bytes() : 0;
    if (out_norm_bytes)           *out_norm_bytes           = sig ? ggml_moe_phaseb_norm_snapshot_bytes()   : 0;
    if (out_intent_record_bytes)  *out_intent_record_bytes  = irec;
    if (out_intent_q_bytes)       *out_intent_q_bytes       = irec * (spec_on ? GGML_MOE_PHASEB_INTENT_SLOTS : 0);
    if (out_predictor_threads)    *out_predictor_threads    = g_pb.worker_running.load(std::memory_order_acquire) ? 1u : 0u;
    if (out_predictor_stack)      *out_predictor_stack      = sig ? (uint64_t)PB_PREDICTOR_STACK : 0;
    if (out_capture_overflow)     *out_capture_overflow     = g_pb.cap_overflow.load(std::memory_order_relaxed);
    if (out_intent_overflow)      *out_intent_overflow      = g_pb.ctr[GGML_MOE_PB_C_INTENT_OVERFLOW].load(std::memory_order_relaxed);
}

// ★R4 테스트 barrier 제어. arm: mode(1=cap_mu 전 · 2=등록 직후) 무장 후 즉시 반환.
// ★R5 rendezvous 제어 — arm 후 CUT 가 cap_mu 를 시도/취득하는 시점을 관측할 수 있다.
int ggml_moe_phaseb_test_cut_rendezvous_arm(void) {
    if (!g_pb.cut_attempted) g_pb.cut_attempted = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.cut_contended) g_pb.cut_contended = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.cut_acquired)  g_pb.cut_acquired  = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.cut_attempted || !g_pb.cut_contended || !g_pb.cut_acquired) return -1;
    ResetEvent((HANDLE)g_pb.cut_attempted);
    ResetEvent((HANDLE)g_pb.cut_contended);
    ResetEvent((HANDLE)g_pb.cut_acquired);
    g_pb.test_cut_rv.store(1, std::memory_order_relaxed);
    return 0;
}
int ggml_moe_phaseb_test_cut_wait_attempted(int32_t timeout_ms) {
    if (!g_pb.cut_attempted) return -1;
    return WaitForSingleObject((HANDLE)g_pb.cut_attempted,
                               (DWORD)(timeout_ms > 0 ? timeout_ms : 1000)) == WAIT_OBJECT_0 ? 0 : -1;
}
// ★R6 — CUT 가 cap_mu 를 **시도했다가 실패(경합)** 했음을 관측한다. 0 = 양성 확인.
int ggml_moe_phaseb_test_cut_wait_contended(int32_t timeout_ms) {
    if (!g_pb.cut_contended) return -1;
    return WaitForSingleObject((HANDLE)g_pb.cut_contended,
                               (DWORD)(timeout_ms > 0 ? timeout_ms : 1000)) == WAIT_OBJECT_0 ? 0 : -1;
}
int ggml_moe_phaseb_test_cut_wait_acquired(int32_t timeout_ms) {
    if (!g_pb.cut_acquired) return -1;
    return WaitForSingleObject((HANDLE)g_pb.cut_acquired,
                               (DWORD)(timeout_ms > 0 ? timeout_ms : 1000)) == WAIT_OBJECT_0 ? 0 : -1;
}
// 1 = CUT 가 이미 cap_mu 를 취득했다 / 0 = 아직(= 경합 위치에서 막혀 있다).
int ggml_moe_phaseb_test_cut_acquired(void) {
    if (!g_pb.cut_acquired) return -1;
    return WaitForSingleObject((HANDLE)g_pb.cut_acquired, 0) == WAIT_OBJECT_0 ? 1 : 0;
}
void ggml_moe_phaseb_test_cut_rendezvous_disarm(void) {
    g_pb.test_cut_rv.store(0, std::memory_order_relaxed);
}

// E-2a (Codex r2 blocker 2) - the window that parks the worker once, in its real busy state.
// arm -> submit one capture -> when _wait returns 0 the worker is inside that window, and from
// there both of the predictor drain's waits actually block. Skipping _release does not hang the
// run: the park self-clears at the 5 s bound and _safety_fired reports that it had to.
int ggml_moe_phaseb_test_worker_hold_arm(void) {
    if (!g_pb.whold_hit) g_pb.whold_hit = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.whold_go)  g_pb.whold_go  = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.whold_hit || !g_pb.whold_go) return -1;
    ResetEvent((HANDLE)g_pb.whold_hit);
    ResetEvent((HANDLE)g_pb.whold_go);
    g_pb.whold_safety.store(0, std::memory_order_relaxed);
    g_pb.test_worker_hold.store(1, std::memory_order_relaxed);
    return 0;
}
int ggml_moe_phaseb_test_worker_hold_wait(int32_t timeout_ms) {
    if (!g_pb.whold_hit) return -1;
    return WaitForSingleObject((HANDLE)g_pb.whold_hit,
                               (DWORD)(timeout_ms > 0 ? timeout_ms : 1000)) == WAIT_OBJECT_0 ? 0 : -1;
}
void ggml_moe_phaseb_test_worker_hold_release(void) {
    if (g_pb.whold_go) SetEvent((HANDLE)g_pb.whold_go);
}
int ggml_moe_phaseb_test_worker_hold_safety_fired(void) {
    return g_pb.whold_safety.load(std::memory_order_relaxed);
}
int ggml_moe_phaseb_test_drain_stages(int64_t * out_deadline_abs_ms, int32_t * out_left_ms,
                                      int32_t max_n) {
    const int32_t n = g_pb.drain_stage_n.load(std::memory_order_relaxed);
    const int32_t k = (n < max_n) ? n : max_n;
    for (int32_t i = 0; i < k; i++) {
        if (out_deadline_abs_ms) {
            out_deadline_abs_ms[i] = g_pb.drain_stage_deadline[(size_t)i].load(std::memory_order_relaxed);
        }
        if (out_left_ms) {
            out_left_ms[i] = g_pb.drain_stage_left[(size_t)i].load(std::memory_order_relaxed);
        }
    }
    return n;
}

int ggml_moe_phaseb_test_cap_barrier_arm(int32_t mode) {
    if (mode != 1 && mode != 2) return -1;
    if (!g_pb.cap_bar_hit) g_pb.cap_bar_hit = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.cap_bar_go)  g_pb.cap_bar_go  = (void *)CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_pb.cap_bar_hit || !g_pb.cap_bar_go) return -1;
    ResetEvent((HANDLE)g_pb.cap_bar_hit);
    ResetEvent((HANDLE)g_pb.cap_bar_go);
    g_pb.test_cap_barrier.store(mode, std::memory_order_relaxed);
    return 0;
}
int ggml_moe_phaseb_test_cap_barrier_wait(int32_t timeout_ms) {
    if (!g_pb.cap_bar_hit) return -1;
    return WaitForSingleObject((HANDLE)g_pb.cap_bar_hit,
                               (DWORD)(timeout_ms > 0 ? timeout_ms : 1000)) == WAIT_OBJECT_0 ? 0 : -1;
}
void ggml_moe_phaseb_test_cap_barrier_release(void) {
    if (g_pb.cap_bar_go) SetEvent((HANDLE)g_pb.cap_bar_go);
}
// ★R7 진단 — 지금 barrier 안에 정지한 스레드가 있는가(mode-2 면 그 스레드가 cap_mu 보유자다).
int ggml_moe_phaseb_test_cap_barrier_held(void) {
    return g_pb.bar_held.load(std::memory_order_acquire) > 0 ? 1 : 0;
}
// ★R7 진단 — 안전 타임아웃(60s) 발동 여부. 1 = 픽스처가 release 를 부르지 못했다(케이스 FAIL).
int ggml_moe_phaseb_test_cap_barrier_safety_fired(void) {
    return g_pb.bar_safety.load(std::memory_order_relaxed);
}

void ggml_moe_phaseb_reset_for_test(void) {
    // ★★U-1(교차 r4 ①) — predictor 정지가 이미 timeout 했다면 **여기서 다시 무한 join 하지
    //   않는다**. 구 판은 이 첫 줄이 INFINITE 라, predictor 정지 회귀 시 하네스가 종단 가드에
    //   도달하기도 전에 매달렸다. 래치가 서 있으면 worker 회수를 포기하고 나머지 상태만
    //   되돌린다(스레드는 살아 있지만 그 런은 이미 붉고 곧 종료된다).
    if (g_pb.stop_timed_out.load(std::memory_order_acquire) == 0) {
        pb_worker_stop_join();
    }
    g_pb.env_resolved = false;
    g_pb.mode = GGML_MOE_PHASEB_OFF;
    g_pb.out_prefix.clear();
    g_pb.buf_mb = 512;
    g_pb.pf_resolved = false; g_pb.pf_on = false; g_pb.pf_k = 0; g_pb.pf_n = 0;
    g_pb.pf_off_reason = "unset";
    // ★5c-T 수집 축 전건 초기화(시나리오 격리 — env 는 위 env_resolved=false 로 재해소된다).
    g_pb.collect_raw = false;
    g_pb.collect_on  = false;
    // ★5c-A1 adapt 축·view 전건 초기화(시나리오 격리 — env 는 위 env_resolved=false 로 재해소).
    g_pb.adapt_raw = false;
    g_pb.adapt_on  = false;
    g_pb.view_raw  = PbBundleView();
    g_pb.view      = PbBundleView();
    g_pb.view_sealed   = false;
    g_pb.grant_refusal = "activation_grant_absent";
    g_pb.controller_alloc.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_pb.led_mu);
        g_pb.led_recs.clear();
    }
    g_pb.led_event_seq.store(0, std::memory_order_relaxed);
    g_pb.gen_seq_next.store(0, std::memory_order_relaxed);
    g_pb.decision_seq_next.store(0, std::memory_order_relaxed);
    g_pb.test_submit_fail_next.store(0, std::memory_order_relaxed);
    g_pb.test_worker_fail_next.store(0, std::memory_order_relaxed);
    g_pb.test_dup_term_arm.store(0, std::memory_order_relaxed);
    g_pb.test_dup_term_hits.store(0, std::memory_order_relaxed);
    g_pb.test_stamp_k.store(0, std::memory_order_relaxed);
    g_pb.test_stamp_n.store(0, std::memory_order_relaxed);
    g_pb.test_stamp_qd.store(0, std::memory_order_relaxed);
    // ★계열 adapter — 선언 descriptor·인벤토리 집계·해소 결과 전건 초기화(시나리오 격리).
    ggml_moe_phaseb_note_family(nullptr);
    // ★정정 6호 — 오힌트 표면 전건 초기화(env 는 위 env_resolved=false 로 재해소된다).
    g_pb.hint_raw.clear(); g_pb.hint_fail_reason.clear();
    g_pb.hint_rule = GGML_MOE_PHASEB_TEST_HINT_NONE;
    g_pb.hint_offset = 0;
    g_pb.hint_applied.store(0, std::memory_order_relaxed);
    g_pb.sealed = false;
    g_pb.n_expert = g_pb.n_expert_used = g_pb.n_moe_layers = g_pb.n_slots = g_pb.qd_effective = 0;
    g_pb.moe_layers.clear(); g_pb.strides.clear(); g_pb.layer_to_mlpos.clear();
    g_pb.snap_configured = g_pb.snap_sealed = false;
    g_pb.n_layer_exec = g_pb.n_embd = 0;
    g_pb.snap_router.clear(); g_pb.snap_norm.clear(); g_pb.snap_bias.clear();
    g_pb.snap_router_ok.clear(); g_pb.snap_norm_ok.clear(); g_pb.snap_bias_ok.clear();
    // ★observe 축 전건 초기화(시나리오 격리 — 카운터·행·graph 색인).
    g_pb.obs_on = false;
    g_pb.obs_reason.clear();
    {
        std::lock_guard<std::mutex> lk(g_pb.obs_mu);
        g_pb.obs_recs.clear();
        g_pb.obs_base_by_graph.clear();
        g_pb.obs_no_out_graph.clear();   // ★NO_OUTPUT graph 집합도 시나리오 격리 대상
    }
    g_pb.obs_no_out_suppressed.store(0, std::memory_order_relaxed);
    for (int i = 0; i < GGML_MOE_PB_O_COUNT; i++) g_pb.obs_ctr[i].store(0, std::memory_order_relaxed);
    g_pb.obs_cost_ns_max.store(0, std::memory_order_relaxed);
    g_pb.obs_cost_samples.store(0, std::memory_order_relaxed);
    g_pb.request_index.store(GGML_MOE_PHASEB_NA_U32, std::memory_order_relaxed);
    g_pb.decode_ordinal.store(GGML_MOE_PHASEB_NA_U32, std::memory_order_relaxed);
    g_pb.graph_seq.store(0, std::memory_order_relaxed);
    g_pb.cur_graph_seq.store(GGML_MOE_PHASEB_NA_U32, std::memory_order_relaxed);
    g_pb.cur_phase.store(0, std::memory_order_relaxed);
    g_pb.next_request_index = 0; g_pb.next_decode_ordinal = 0; g_pb.total_requests = 0;
    g_pb.run_active = false; g_pb.graph_active = false;
    g_pb.cap_slots.clear(); g_pb.cap_free.clear(); g_pb.cap_ring.clear();
    g_pb.cap_head = g_pb.cap_count = 0; g_pb.worker_busy = false;
    g_pb.cut_closed.store(false, std::memory_order_relaxed);
    g_pb.cap_layer_seen.clear(); g_pb.cap_layer_seen_graph = GGML_MOE_PHASEB_NA_U32;
    g_pb.w_norm_buf.clear(); g_pb.w_logit_buf.clear(); g_pb.w_used_buf.clear();
    g_pb.int_slots.clear(); g_pb.int_head = g_pb.int_count = 0;
    g_pb.intent_ordinal_next = 0;
    g_pb.disabled_request.store(GGML_MOE_PHASEB_NA_U32, std::memory_order_relaxed);
    g_pb.lora_swapped = false;
    g_pb.wake_event = nullptr;
    g_pb.test_cap_barrier.store(0, std::memory_order_relaxed);
    g_pb.bar_held.store(0, std::memory_order_relaxed);
    g_pb.bar_safety.store(0, std::memory_order_relaxed);
    if (g_pb.cap_bar_hit) { CloseHandle((HANDLE)g_pb.cap_bar_hit); g_pb.cap_bar_hit = nullptr; }
    if (g_pb.cap_bar_go)  { CloseHandle((HANDLE)g_pb.cap_bar_go);  g_pb.cap_bar_go  = nullptr; }
    // E-2a - the worker park follows the same discipline, with one difference: the join above
    // is SKIPPED when stop_timed_out is set, so a worker can still be sitting in the park. Free
    // it before closing the handles; closing first leaves that thread waiting on a dead handle
    // until its 5 s bound.
    g_pb.test_worker_hold.store(0, std::memory_order_relaxed);
    if (g_pb.whold_go) SetEvent((HANDLE)g_pb.whold_go);
    g_pb.whold_safety.store(0, std::memory_order_relaxed);
    if (g_pb.whold_hit) { CloseHandle((HANDLE)g_pb.whold_hit); g_pb.whold_hit = nullptr; }
    if (g_pb.whold_go)  { CloseHandle((HANDLE)g_pb.whold_go);  g_pb.whold_go  = nullptr; }
    g_pb.drain_stage_n.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < 2; i++) {
        g_pb.drain_stage_deadline[i].store(0, std::memory_order_relaxed);
        g_pb.drain_stage_left[i].store(0, std::memory_order_relaxed);
    }
    g_pb.test_cut_rv.store(0, std::memory_order_relaxed);
    if (g_pb.cut_attempted) { CloseHandle((HANDLE)g_pb.cut_attempted); g_pb.cut_attempted = nullptr; }
    if (g_pb.cut_contended) { CloseHandle((HANDLE)g_pb.cut_contended); g_pb.cut_contended = nullptr; }
    if (g_pb.cut_acquired)  { CloseHandle((HANDLE)g_pb.cut_acquired);  g_pb.cut_acquired  = nullptr; }
    g_pb.all_off.store(true, std::memory_order_relaxed);
    g_pb.predictor_drained.store(false, std::memory_order_relaxed);
    g_pb.skip_by_target.clear(); g_pb.ovf_by_target.clear();
    g_pb.status_q.clear(); g_pb.status_head = 0; g_pb.status_count = 0;
    g_pb.mailbox_graph = GGML_MOE_PHASEB_NA_U32;
    for (int i = 0; i < GGML_MOE_PB_C_COUNT; i++)   g_pb.ctr[i].store(0, std::memory_order_relaxed);
    for (int i = 0; i < GGML_MOE_PB_SKIP_COUNT; i++) g_pb.skip_ctr[i].store(0, std::memory_order_relaxed);
    g_pb.cap_overflow.store(0, std::memory_order_relaxed);
    g_pb.blocked_us_total.store(0, std::memory_order_relaxed);
    g_pb.blocked_count_total.store(0, std::memory_order_relaxed);
    g_pb.qd_max_demand.store(0, std::memory_order_relaxed);
    g_pb.qd_max_prefetch.store(0, std::memory_order_relaxed);
    g_pb.qd_max_total.store(0, std::memory_order_relaxed);
    g_pb.dg_intent = g_pb.dg_disp = g_pb.dg_trans = g_pb.dg_class = g_pb.dg_readfile = FNV_INIT;
    g_pb.dg_readfile_ops = 0; g_pb.intent_count = 0;
    g_pb.runs.clear();
    g_pb.equiv_rows.clear(); g_pb.equiv_recs.clear();
    g_pb.r_qpc_by_layer.clear(); g_pb.pend_by_target.clear();
    g_pb.buf_used.store(0, std::memory_order_relaxed);
    g_pb.dropped.store(0, std::memory_order_relaxed);
    g_pb.complete.store(true, std::memory_order_relaxed);
    g_pb.incomplete_mask.store(0, std::memory_order_relaxed);   // ★sticky 해제는 이 리셋뿐(§4.3)
    g_pb.run_invalid.store(false, std::memory_order_relaxed);
}

int ggml_moe_phaseb_test_push_intent(uint32_t request_epoch, uint32_t graph_epoch,
                                     uint32_t graph_seq, int32_t source_mlpos,
                                     int32_t target_mlpos, const int32_t * ids, int32_t n) {
    if (!g_pb.pf_on || g_pb.int_slots.empty()) return -1;
    if (target_mlpos < 0 || target_mlpos >= g_pb.n_moe_layers) return -1;
    ggml_moe_phaseb_intent it{};
    it.request_epoch = request_epoch;
    it.graph_epoch   = graph_epoch;
    it.graph_seq     = graph_seq;
    it.source_mlpos  = source_mlpos;
    it.target_mlpos  = target_mlpos;
    it.source_layer  = (uint16_t)(source_mlpos >= 0 ? g_pb.moe_layers[(size_t)source_mlpos] : 0);
    it.target_layer  = (uint16_t)g_pb.moe_layers[(size_t)target_mlpos];
    it.k             = (uint32_t)g_pb.pf_k;
    it.n             = (uint32_t)g_pb.pf_n;   // ★5c-T §1 스탬프 N(본선과 동일 결속)
    it.capture_qpc   = ggml_moe_phaseb_now_qpc();
    it.predict_end_qpc = it.capture_qpc;
    for (int i = 0; i < PB_K; i++) it.ids[i] = (i < n && ids[i] >= 0) ? (uint16_t)ids[i] : GGML_MOE_PHASEB_NA_U16;
    bool ovf = false;
    {
        std::lock_guard<std::mutex> lk(g_pb.int_mu);
        const int32_t cap = (int32_t)g_pb.int_slots.size();
        if (g_pb.int_count >= cap) {
            g_pb.disabled_request.store(request_epoch, std::memory_order_relaxed);
            g_pb.run_invalid.store(true, std::memory_order_relaxed);
            ggml_moe_phaseb_count(GGML_MOE_PB_C_INTENT_OVERFLOW, request_epoch, 1);
            ovf = true;
        } else {
        it.intent_ordinal = g_pb.intent_ordinal_next++;
        g_pb.int_slots[(size_t)((g_pb.int_head + g_pb.int_count) % cap)] = it;
        g_pb.int_count++;
        uint64_t h = g_pb.dg_intent;
        const uint32_t re = it.request_epoch, ge = it.graph_epoch;
        const uint16_t sl = it.source_layer, tl = it.target_layer, ke = (uint16_t)it.k;
        h = fnv1a64(h, &re, sizeof(re));
        h = fnv1a64(h, &ge, sizeof(ge));
        h = fnv1a64(h, &sl, sizeof(sl));
        h = fnv1a64(h, &tl, sizeof(tl));
        h = fnv1a64(h, &ke, sizeof(ke));
        h = fnv1a64(h, it.ids, sizeof(it.ids));
        g_pb.dg_intent = h;
        g_pb.intent_count++;
        if (g_pb.wake_event) SetEvent((HANDLE)g_pb.wake_event);   // ★R1 #2 ①
        }
    }
    if (ovf) { pb_note_target_status(graph_seq, target_mlpos, 0, 1); return -2; }   // ★R2 #5
    ggml_moe_phaseb_count(GGML_MOE_PB_C_INTENT_PREDICTED, request_epoch, 1);
    return 0;
}

void ggml_moe_phaseb_test_counts(uint64_t * out_captured, uint64_t * out_predicted,
                                 uint64_t * out_intent_overflow, uint64_t * out_capture_overflow,
                                 int32_t * out_intent_depth, int32_t * out_worker_running) {
    if (out_captured)        *out_captured        = g_pb.ctr[GGML_MOE_PB_C_INTENT_CAPTURED].load(std::memory_order_relaxed);
    if (out_predicted)       *out_predicted       = g_pb.ctr[GGML_MOE_PB_C_INTENT_PREDICTED].load(std::memory_order_relaxed);
    if (out_intent_overflow) *out_intent_overflow = g_pb.ctr[GGML_MOE_PB_C_INTENT_OVERFLOW].load(std::memory_order_relaxed);
    if (out_capture_overflow)*out_capture_overflow= g_pb.cap_overflow.load(std::memory_order_relaxed);
    if (out_intent_depth)    *out_intent_depth    = ggml_moe_phaseb_intent_depth();
    if (out_worker_running)  *out_worker_running  = g_pb.worker_running.load(std::memory_order_acquire) ? 1 : 0;
}

// selftest 전용 — 상태 링 잔여 이벤트 수(observe fanout 차단 게이트의 관측 창구).
uint32_t ggml_moe_phaseb_test_status_depth(void) {
    std::lock_guard<std::mutex> lk(g_pb.tm_mu);
    return (uint32_t)g_pb.status_count;
}

int ggml_moe_phaseb_test_wait_idle(uint32_t timeout_ms) {
    if (!g_pb.worker_running.load(std::memory_order_acquire)) return 1;
    std::unique_lock<std::mutex> lk(g_pb.cap_mu);
    return g_pb.cap_idle_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                                     [] { return g_pb.cap_count == 0 && !g_pb.worker_busy; }) ? 1 : 0;
}

} // extern "C"

#else // !_WIN32 — 비Windows 스텁

extern "C" {

int32_t      ggml_moe_phaseb_mode(void) { return 0; }
bool         ggml_moe_phaseb_enabled(void) { return false; }
const char * ggml_moe_phaseb_mode_name(void) { return "off"; }
uint64_t     ggml_moe_phaseb_boot_id(void) { return 0; }
const char * ggml_moe_phaseb_out_prefix(void) { return ""; }
uint32_t     ggml_moe_phaseb_buf_mb(void) { return 0; }
uint64_t     ggml_moe_phaseb_now_qpc(void) { return 0; }
void         ggml_moe_phaseb_resolve_prefetch(int32_t) {}
bool         ggml_moe_phaseb_prefetch_on(void) { return false; }
int32_t      ggml_moe_phaseb_prefetch_k(void) { return 0; }
int32_t      ggml_moe_phaseb_prefetch_n(void) { return 0; }
const char * ggml_moe_phaseb_prefetch_off_reason(void) { return "unset"; }
int          ggml_moe_phaseb_equiv_precheck(bool, const char **) { return 0; }
// ★계열 신호 adapter — 비Windows 는 신호 경로 자체가 없으므로 항상 미해소(none).
void         ggml_moe_phaseb_note_family(const struct ggml_moe_phaseb_family_in *) {}
int32_t      ggml_moe_phaseb_adapter_id(void) { return GGML_MOE_PHASEB_ADAPTER_NONE; }
const char * ggml_moe_phaseb_adapter_name(void) { return ""; }
const char * ggml_moe_phaseb_adapter_reason(void) { return "descriptor_absent"; }
int32_t      ggml_moe_phaseb_norm_tensor_kind(void) { return GGML_MOE_PHASEB_NORM_NONE; }
int32_t      ggml_moe_phaseb_selection_fn(void) { return GGML_MOE_PHASEB_SEL_NONE; }
int32_t      ggml_moe_phaseb_bias_kind(void) { return GGML_MOE_PHASEB_BIAS_NONE; }
int32_t      ggml_moe_phaseb_capture_width(void) { return 0; }
const char * ggml_moe_phaseb_family_arch(void) { return ""; }
const char * ggml_moe_phaseb_norm_tensor_name(void) { return ""; }
const char * ggml_moe_phaseb_selection_fn_name(void) { return ""; }
const char * ggml_moe_phaseb_bias_snapshot_name(void) { return ""; }
int32_t      ggml_moe_phaseb_family_n_expert(void) { return 0; }
int32_t      ggml_moe_phaseb_family_n_expert_used(void) { return 0; }
int32_t      ggml_moe_phaseb_moe_layer_count(void) { return 0; }
int32_t      ggml_moe_phaseb_moe_layer_at(int32_t) { return -1; }
bool         ggml_moe_phaseb_output_crop_before_last_moe(void) { return false; }
// ★층 마스크 — 비Windows 는 adapter 가 항상 미해소라 중립값(1)·마스크 비활성이다.
int32_t      ggml_moe_phaseb_signal_first_target_layer(void) { return 1; }
bool         ggml_moe_phaseb_signal_layer_mask_active(void) { return false; }
bool         ggml_moe_phaseb_signal_target_eligible(int32_t target_layer) { return target_layer >= 1; }
// ★norm exact-zero inventory — 비Windows 는 신호 경로가 없으므로 항상 **미측정**이다.
bool         ggml_moe_phaseb_norm_inventory_measured(void) { return false; }
int32_t      ggml_moe_phaseb_norm_inventory_scanned(void) { return 0; }
int32_t      ggml_moe_phaseb_norm_inventory_zero_layers(void) { return 0; }
uint64_t     ggml_moe_phaseb_norm_inventory_zero_elements(void) { return 0; }
int          ggml_moe_phaseb_norm_inventory_at(int32_t, int32_t *, int32_t *, uint32_t *, uint64_t *) { return -1; }
uint32_t     ggml_moe_phaseb_norm_inventory_zero_index(int32_t, uint32_t) { return GGML_MOE_PHASEB_NA_U32; }
bool         ggml_moe_phaseb_signal_observe_on(void) { return false; }
int          ggml_moe_phaseb_observe_precheck(const char **) { return 0; }
uint64_t     ggml_moe_phaseb_observe_counter(int32_t) { return 0; }
uint64_t     ggml_moe_phaseb_observe_buf_bytes(void) { return 0; }
uint64_t     ggml_moe_phaseb_observe_records(void) { return 0; }
int          ggml_moe_phaseb_write_observe(uint64_t * r, uint64_t * d) { if (r) *r = 0; if (d) *d = 0; return 0; }
int          ggml_moe_phaseb_test_observe_record(uint64_t, void *, uint32_t) { return -1; }
uint64_t     ggml_moe_phaseb_no_output_count(uint32_t) { return 0; }
int          ggml_moe_phaseb_no_output_at(uint32_t, uint64_t, uint32_t *, uint32_t *, uint16_t *) { return -1; }
uint64_t     ggml_moe_phaseb_test_no_output_suppressed(void) { return 0; }
int          ggml_moe_phaseb_snapshot_bias(int32_t, const struct ggml_tensor *) { return -1; }
uint64_t     ggml_moe_phaseb_bias_snapshot_bytes(void) { return 0; }
int          ggml_moe_phaseb_test_snapshot_bias_raw(int32_t, const float *) { return -1; }
int32_t      ggml_moe_phaseb_test_hint_rule(void) { return 0; }
const char * ggml_moe_phaseb_test_hint_rule_name(void) { return ""; }
int32_t      ggml_moe_phaseb_test_hint_offset(void) { return 0; }
uint64_t     ggml_moe_phaseb_test_hint_applied(void) { return 0; }
int          ggml_moe_phaseb_test_hint_precheck(bool, const char **) { return 0; }
int          ggml_moe_phaseb_note_seal(int32_t, int32_t, int32_t, int32_t, int32_t,
                                       const int32_t *, const uint64_t *,
                                       int32_t, int32_t, float) { return 0; }
void         ggml_moe_phaseb_set_wake_event(void *) {}
bool         ggml_moe_phaseb_all_off(void) { return true; }
bool         ggml_moe_phaseb_predictor_drained(void) { return true; }
void         ggml_moe_phaseb_target_status(uint32_t, int32_t, uint8_t * a, uint8_t * b) { if (a) *a = 0; if (b) *b = 0; }
int          ggml_moe_phaseb_status_pop(uint32_t *, int32_t *, uint8_t *, uint8_t *) { return 0; }
int          ggml_moe_phaseb_test_cut_rendezvous_arm(void) { return -1; }
int          ggml_moe_phaseb_test_cut_wait_attempted(int32_t) { return -1; }
int          ggml_moe_phaseb_test_cut_wait_contended(int32_t) { return -1; }
int          ggml_moe_phaseb_test_cut_wait_acquired(int32_t) { return -1; }
int          ggml_moe_phaseb_test_cut_acquired(void) { return -1; }
void         ggml_moe_phaseb_test_cut_rendezvous_disarm(void) {}
int          ggml_moe_phaseb_test_cap_barrier_arm(int32_t) { return -1; }
int          ggml_moe_phaseb_test_cap_barrier_wait(int32_t) { return -1; }
void         ggml_moe_phaseb_test_cap_barrier_release(void) {}
int          ggml_moe_phaseb_test_cap_barrier_held(void) { return 0; }
int          ggml_moe_phaseb_test_cap_barrier_safety_fired(void) { return 0; }
uint32_t     ggml_moe_phaseb_capture_pool_slots(void) { return 0; }
uint32_t     ggml_moe_phaseb_intent_q_slots(void) { return 0; }
void         ggml_moe_phaseb_run_begin(void) {}
void         ggml_moe_phaseb_run_end(void) {}
void         ggml_moe_phaseb_graph_begin(uint8_t, uint32_t, uint32_t) {}
void         ggml_moe_phaseb_graph_end(void) {}
void         ggml_moe_phaseb_current_epoch(uint32_t * a, uint32_t * b, uint32_t * c, uint8_t * d) {
    if (a) *a = 0xFFFFFFFFu; if (b) *b = 0xFFFFFFFFu; if (c) *c = 0xFFFFFFFFu; if (d) *d = 0;
}
uint64_t     ggml_moe_phaseb_total_requests(void) { return 0; }
int          ggml_moe_phaseb_snapshot_configure(int32_t, int32_t, int32_t, int32_t, float) { return 0; }
bool         ggml_moe_phaseb_snapshot_needed(void) { return false; }
int          ggml_moe_phaseb_snapshot_layer(int32_t, const struct ggml_tensor *, const struct ggml_tensor *) { return 0; }
int          ggml_moe_phaseb_snapshot_seal(void) { return 0; }
uint64_t     ggml_moe_phaseb_router_snapshot_bytes(void) { return 0; }
uint64_t     ggml_moe_phaseb_norm_snapshot_bytes(void) { return 0; }
int          ggml_moe_phaseb_test_snapshot_raw(int32_t, const float *, const float *) { return 0; }
int          ggml_moe_phaseb_capture(int32_t, int32_t, const void *, int32_t, int32_t, int32_t, int32_t, int32_t) { return 0; }
void         ggml_moe_phaseb_note_lora_swap(void) {}
void         ggml_moe_phaseb_tap_a0(int32_t, const float *, int32_t) {}
int          ggml_moe_phaseb_intent_peek(struct ggml_moe_phaseb_intent *) { return 0; }
void         ggml_moe_phaseb_intent_pop(void) {}
int          ggml_moe_phaseb_intent_depth(void) { return 0; }
void         ggml_moe_phaseb_cut_close_capture(void) {}
void         ggml_moe_phaseb_cut_drain_predictor(void) {}
int          ggml_moe_phaseb_cut_drain_predictor_bounded(int32_t) { return 1; }
int          ggml_moe_phaseb_cut_drain_predictor_until(int64_t) { return 1; }
void         ggml_moe_phaseb_unseal_failed(void) {}
bool         ggml_moe_phaseb_capture_closed(void) { return true; }
void         ggml_moe_phaseb_count(int32_t, uint32_t, uint64_t) {}
uint64_t     ggml_moe_phaseb_counter(int32_t) { return 0; }
void         ggml_moe_phaseb_count_skip(int32_t, uint32_t) {}
uint64_t     ggml_moe_phaseb_skip_counter(int32_t) { return 0; }
void         ggml_moe_phaseb_count_split(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}
void         ggml_moe_phaseb_count_blocked(uint32_t, uint64_t, uint32_t) {}
// ★5c-T 수집 텔레메트리(부록 14) — 비Windows 스텁(전 표면 no-op·수집 OFF 고정).
bool         ggml_moe_phaseb_collect_on(void) { return false; }
// ★5c-A1 identity 축 — 비Windows 스텁(adapt OFF 고정 · view 미수용 · 계수 0).
bool         ggml_moe_phaseb_adapt_on(void) { return false; }
bool         ggml_moe_phaseb_bundle_view_accepted(void) { return false; }
const char * ggml_moe_phaseb_activation_grant_refusal(void) { return "activation_grant_absent"; }
uint64_t     ggml_moe_phaseb_controller_alloc_count(void) { return 0; }
void         ggml_moe_phaseb_run_outcome(int32_t) {}
uint64_t     ggml_moe_phaseb_run_admission_seq(uint32_t) { return 0xFFFFFFFFFFFFFFFFull; }
uint64_t     ggml_moe_phaseb_next_decision_seq(uint32_t) { return 0; }
void         ggml_moe_phaseb_run_stamp(uint32_t, int32_t * a, int32_t * b, int32_t * c,
                                       int32_t * d, uint64_t * e) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
    if (e) *e = 0xFFFFFFFFFFFFFFFFull;
}
void         ggml_moe_phaseb_note_blocked_interval(uint32_t) {}
int          ggml_moe_phaseb_ledger_push(const struct ggml_moe_phaseb_ledger *) { return 0; }
int          ggml_moe_phaseb_ledger_push_group(const struct ggml_moe_phaseb_ledger *, uint32_t) { return 0; }
uint64_t     ggml_moe_phaseb_ledger_next_event_seq(void) { return 0; }
uint64_t     ggml_moe_phaseb_ledger_records(void) { return 0; }
int          ggml_moe_phaseb_write_ledger(uint64_t * r, uint64_t * d) { if (r) *r = 0; if (d) *d = 0; return 0; }
void         ggml_moe_phaseb_test_submit_fail_next(int32_t) {}
void         ggml_moe_phaseb_test_worker_start_fail_next(int32_t) {}
int          ggml_moe_phaseb_test_submit_fail_take(void) { return 0; }
void         ggml_moe_phaseb_test_stamp_override(int32_t, int32_t, int32_t) {}
void         ggml_moe_phaseb_test_dup_terminal_arm(int32_t) {}
int          ggml_moe_phaseb_test_dup_terminal_take(void) { return 0; }
void         ggml_moe_phaseb_test_dup_terminal_note(int32_t, uint64_t, uint64_t, int32_t) {}
int          ggml_moe_phaseb_test_dup_terminal_result(int32_t * a, uint64_t * b, uint64_t * c, int32_t * d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; return 0;
}
int          ggml_moe_phaseb_test_ledger_record(uint64_t, void *, uint32_t) { return -1; }
int          ggml_moe_phaseb_test_run_outcome(uint32_t) { return 0; }
uint64_t     ggml_moe_phaseb_test_run_blocked_intervals(uint32_t) { return 0; }
void         ggml_moe_phaseb_note_sidecar_record(int32_t, uint32_t) {}
void         ggml_moe_phaseb_note_qd(uint32_t, uint32_t, uint32_t) {}
void         ggml_moe_phaseb_qd_max(uint32_t * a, uint32_t * b, uint32_t * c) { if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; }
void         ggml_moe_phaseb_blocked_totals(uint64_t * a, uint64_t * b) { if (a) *a = 0; if (b) *b = 0; }
void         ggml_moe_phaseb_note_demand_ready(uint32_t, int32_t, uint64_t) {}
void         ggml_moe_phaseb_digest_disposition(uint64_t, uint16_t, uint8_t) {}
void         ggml_moe_phaseb_digest_transition(uint8_t, uint64_t, int32_t, uint64_t, uint64_t) {}
void         ggml_moe_phaseb_digest_cacheclass(uint32_t, uint8_t, uint64_t) {}
void         ggml_moe_phaseb_digest_readfile(uint16_t, uint16_t, uint64_t, uint32_t, uint8_t) {}
void         ggml_moe_phaseb_digests(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d,
                                     uint64_t * e, uint64_t * f, uint64_t * g) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0; if (g) *g = 0;
}
bool         ggml_moe_phaseb_buffer_reserve(uint64_t) { return false; }
uint64_t     ggml_moe_phaseb_buffer_used(void) { return 0; }
uint64_t     ggml_moe_phaseb_buffer_cap(void) { return 0; }
bool         ggml_moe_phaseb_complete(void) { return false; }
void         ggml_moe_phaseb_mark_incomplete(uint32_t) {}
uint32_t     ggml_moe_phaseb_incomplete_mask(void) { return 0; }
uint64_t     ggml_moe_phaseb_dropped_records(void) { return 0; }
void         ggml_moe_phaseb_note_dropped(uint64_t) {}
bool         ggml_moe_phaseb_run_invalid(void) { return false; }
int          ggml_moe_phaseb_write_sidecar(int32_t, const void *, uint64_t, uint32_t, uint64_t * d) { if (d) *d = 0; return 0; }
int          ggml_moe_phaseb_write_a0equiv(uint64_t * r, uint64_t * d) { if (r) *r = 0; if (d) *d = 0; return 0; }
void         ggml_moe_phaseb_emit_runs(const char *, const char *, const char *, const char *,
                                       int32_t, uint64_t, uint64_t) {}
void         ggml_moe_phaseb_mem_echo(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d, uint64_t * e,
                                      uint64_t * f, uint32_t * g, uint64_t * h, uint64_t * i, uint64_t * j) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0;
    if (f) *f = 0; if (g) *g = 0; if (h) *h = 0; if (i) *i = 0; if (j) *j = 0;
}
void         ggml_moe_phaseb_reset_for_test(void) {}
int          ggml_moe_phaseb_test_push_intent(uint32_t, uint32_t, uint32_t, int32_t, int32_t, const int32_t *, int32_t) { return -1; }
void         ggml_moe_phaseb_test_counts(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d, int32_t * e, int32_t * f) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0;
}
int          ggml_moe_phaseb_test_wait_idle(uint32_t) { return 1; }
uint32_t     ggml_moe_phaseb_test_status_depth(void) { return 0; }
int          ggml_moe_phaseb_test_worker_hold_arm(void) { return -1; }
int          ggml_moe_phaseb_test_worker_hold_wait(int32_t) { return -1; }
void         ggml_moe_phaseb_test_worker_hold_release(void) {}
int          ggml_moe_phaseb_test_worker_hold_safety_fired(void) { return 0; }
int          ggml_moe_phaseb_test_drain_stages(int64_t *, int32_t *, int32_t) { return 0; }

} // extern "C"

#endif // _WIN32
