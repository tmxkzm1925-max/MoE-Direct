// MoE 프리페치 Phase B — 층-선행 프리페치 신호/관측 모듈(B2).
//
// 사양 권위: bench/moe-direct/PHASEB_DESIGN.md(★FROZEN v3 — §2.9 자료구조·§3 신호 A′·§4 정책·
//            §5 오류/취소·§6 계측)
// wire 계약: bench/moe-direct/SPEC_PHASEB_WIRE.md(v4 ★FROZEN — 필드·게이트·카운터·digest 권위)
//
// 역할 분담(SPEC_PHASEB_WIRE §11):
//   ① 이 모듈           — env/mode 해소·boot_id·router/norm 스냅샷·capture_q·predictor worker·
//                          intent_q·intent 발급/digest·a0_equiv·C3 카운터·요청별 phase_b_run
//                          누적·사이드카 버퍼 상한·파일 기록
//   ② ggml-moe-direct.cpp — dispatcher 결속분(admission·사전 처분·disposition/transition digest·
//                          pb_io 계측·PB_FINALIZE_CUT·metrics header/summary 필드)
//   ③ ggml-cpu.c          — capture 훅 호출부(ith==0·row grouping 후·acquire 앞)
//   ④ llama-context.cpp   — 스냅샷 배선 + a0_ref tap(equiv 모드)
//   ⑤ server-context.cpp  — 요청(run) 경계
//
// ★★ 동시성 계약 ★★ (R2 권고 1 — 실제 획득 순서로 갱신)
//   참여자:
//     · capture 훅(compute 스레드)      : cap_mu → {run_mu, tm_mu}
//     · predictor worker(전용 스레드 1) : cap_mu(pop) → [무락 GEMV] → int_mu(push) → {run_mu, tm_mu}
//     · dispatcher(g_mod.mutex 보유)    : g_mod.mutex → q_mutex → int_mu → {run_mu, tm_mu}
//     · graph/run 경계(decode 스레드)   : tm_mu → run_mu
//   ★전역 락 순서(획득 방향 단방향 · 역방향 없음):
//        g_mod.mutex → q_mutex → int_mu → { run_mu, tm_mu }
//        cap_mu                        → { run_mu, tm_mu }
//     (cap_mu 와 int_mu 는 **동시에 보유하지 않는다** — worker 가 capture 를 pop 한 뒤 cap_mu 를
//      놓고 GEMV 를 돌린 다음에야 int_mu 를 잡는다.)
//   worker는 g_mod.mutex·q_mutex 어느 쪽도 잡지 않는다(PHASEB_DESIGN §2.9 "양쪽 mutex 무보유").
//   ★비동기 사유(NONFINITE·intent overflow)의 pb_sample 행 반영은 worker 가 g_mod.mutex 를 잡는
//     대신 상태 큐(status_q · tm_mu)에 append 하고 dispatcher 가 소비한다(R2 #5 — 역전 방지).
//
// Windows 전용 — 비Windows는 전부 no-op 스텁으로 컴파일된다.
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 모드 (env MOE_DIRECT_PHASEB — off[기본] / trace / equiv / observe)
//   ★observe(3) = 계열 확장 qualification 전용(SPEC_PHASEB_WIRE §13 · SPEC_FAMILY_SIGNAL §2):
//     speculative I/O 0 · 사이드카는 `<P>_observe.pb` **1종만** · seal 전제 위반 = init fail-close.
// ---------------------------------------------------------------------------
#define GGML_MOE_PHASEB_OFF     0
#define GGML_MOE_PHASEB_TRACE   1
#define GGML_MOE_PHASEB_EQUIV   2
#define GGML_MOE_PHASEB_OBSERVE 3

// capture 슬롯이 소유하는 활성값 사본 차원의 **상한 guard**(SPEC_FAMILY_SIGNAL v0.2 §1 —
//   실 폭은 descriptor 의 n_embd 로 seal 에서 해소한다. 구 고정값 3072 = qwen n_embd 였다).
#define GGML_MOE_PHASEB_CAPTURE_DIM_MAX 8192
#define GGML_MOE_PHASEB_CAPTURE_SLOTS 48
#define GGML_MOE_PHASEB_INTENT_SLOTS  48
#define GGML_MOE_PHASEB_TOPK          16
#define GGML_MOE_PHASEB_NA_U16        ((uint16_t)0xFFFF)
#define GGML_MOE_PHASEB_NA_U32        ((uint32_t)0xFFFFFFFFu)
#define GGML_MOE_PHASEB_NA_U64        ((uint64_t)0xFFFFFFFFFFFFFFFFull)

// SPEC_PHASEB_WIRE §4.1 — elig_skip_reason 닫힌 enum(summary 카운터와 1:1).
enum ggml_moe_phaseb_skip {
    GGML_MOE_PB_SKIP_NONE               = 0,
    GGML_MOE_PB_SKIP_SOURCE_RANGE       = 1,
    GGML_MOE_PB_SKIP_PREFILL            = 2,
    GGML_MOE_PB_SKIP_MULTI_TOKEN        = 3,
    GGML_MOE_PB_SKIP_LORA_SWAP          = 4,
    GGML_MOE_PB_SKIP_SRC1_TYPE          = 5,
    GGML_MOE_PB_SKIP_NONCONTIG          = 6,
    GGML_MOE_PB_SKIP_NONFINITE          = 7,
    GGML_MOE_PB_SKIP_CAPTURE_QUEUE_FULL = 8,
    GGML_MOE_PB_SKIP_REQUEST_DISABLED   = 9,
    GGML_MOE_PB_SKIP_PREFETCH_OFF       = 10,
    GGML_MOE_PB_SKIP_NO_SOURCE          = 11,
    // ★★PB virtual 배선(권위 = `bench/techdev/repack_space/SPEC_PB_VIRTUAL_RUNTIME_R1.md`
    //   §2-0 A0-5 · 확장 근거 = A0-6): `N_cap(l) = 0` 인 층 = 그 층 프리페치 기능 OFF.
    //   층별 정적 상한이 0 이라 candidate 가 서지도 못하는 상태이므로 disposition(C3 열)이
    //   아니라 **eligibility skip** 이다(C3 배타 closure 불변).
    GGML_MOE_PB_SKIP_NCAP_ZERO          = 12,
    GGML_MOE_PB_SKIP_COUNT              = 13,
};

// SPEC_PHASEB_WIRE §4.2 — rank_disposition 닫힌 enum.
enum ggml_moe_phaseb_disp {
    GGML_MOE_PB_D_RANK_NA            = 0,
    GGML_MOE_PB_D_PRE_TOO_LATE       = 1,
    GGML_MOE_PB_D_PRE_STALE          = 2,
    GGML_MOE_PB_D_PRE_RESIDENT       = 3,
    GGML_MOE_PB_D_PRE_PENDING        = 4,
    GGML_MOE_PB_D_DROP_NO_SLOT_QD    = 5,
    GGML_MOE_PB_D_DROP_PREFETCH_CAP  = 6,
    GGML_MOE_PB_D_DROP_RESERVE       = 7,
    GGML_MOE_PB_D_ISSUED             = 8,
    GGML_MOE_PB_D_SUBMIT_FAILED      = 9,
};

// SPEC_PHASEB_WIRE §5.2 — terminal_status 닫힌 enum.
enum ggml_moe_phaseb_term {
    GGML_MOE_PB_T_SUCCESS_PUBLISHED       = 1,
    GGML_MOE_PB_T_ERROR_UNPUBLISHED       = 2,
    GGML_MOE_PB_T_CANCELLED_UNPUBLISHED   = 3,
    GGML_MOE_PB_T_CANCEL_SUCCESS_UNPUB    = 4,
};

// SPEC_PHASEB_WIRE §8 — transition digest event_type.
enum ggml_moe_phaseb_ev {
    GGML_MOE_PB_EV_PUBLISH_DEMAND    = 1,
    GGML_MOE_PB_EV_PUBLISH_PROBATION = 2,
    GGML_MOE_PB_EV_CONSUME_JOIN      = 3,
    GGML_MOE_PB_EV_CONSUME_PROMOTION = 4,
    GGML_MOE_PB_EV_EVICT             = 5,
    GGML_MOE_PB_EV_UNPUBLISH_ERROR   = 6,
    GGML_MOE_PB_EV_RETRY_ISSUE       = 7,
    GGML_MOE_PB_EV_CANCEL            = 8,
};

// ★SPEC_PHASEB_WIRE §13.2 — observe 레코드 invalid_reason 닫힌 enum(a0_equiv 계열 승계·
//   4만 의미 교체[observe 엔 양경로 drift 개념이 없다]).
//   ★FROZEN_PB_OBSV_V2 §3(E1) — V2 에서 `NO_OUTPUT=6` 확장(닫힌 enum 의 다음 미사용 값).
enum ggml_moe_phaseb_obs_reason {
    GGML_MOE_PB_OBS_NONE         = 0,
    GGML_MOE_PB_OBS_NORM_ZERO    = 1,
    GGML_MOE_PB_OBS_NONFINITE    = 2,
    GGML_MOE_PB_OBS_OVERFLOW     = 3,
    GGML_MOE_PB_OBS_NOT_ELIGIBLE = 4,
    GGML_MOE_PB_OBS_CAPTURE_MISS = 5,
    GGML_MOE_PB_OBS_NO_OUTPUT    = 6,   // ★V2 신설 — 서버 확정 zero-output graph(측정 대상 부재)
};

// ★FROZEN_PB_OBSV_V2 §4 — `flags` 비트(V2). b0 = valid(V1 승계) · b1 = NO_OUTPUT(V2 신설) ·
//   b2~b7 = reserved(전부 0). **V1 파일에서는 b1~b7 전부 0 이 계약이며 묵시 재사용은 금지**다
//   (태그 도입은 `kind_tag` 버전 상승으로만 관측된다 — §4-3).
#define GGML_MOE_PB_OBS_FLAG_VALID     0x01
#define GGML_MOE_PB_OBS_FLAG_NO_OUTPUT 0x02

// ★SPEC_PHASEB_WIRE §13.4 — observe 카운터(summary 키와 1:1 · closure 등식 §13.4).
//   ★FROZEN_PB_OBSV_V2 §5.1 — V2 에서 `observe_no_output`(12번째·필수 키) 추가.
enum ggml_moe_phaseb_obs_ctr {
    GGML_MOE_PB_O_EXPECTED = 0,
    GGML_MOE_PB_O_ELIGIBLE,
    GGML_MOE_PB_O_NOT_ELIGIBLE,
    GGML_MOE_PB_O_CAPTURED,
    GGML_MOE_PB_O_CAPTURE_MISS,
    GGML_MOE_PB_O_RECORDED,
    GGML_MOE_PB_O_VALID,
    GGML_MOE_PB_O_INVALID_NORM_ZERO,
    GGML_MOE_PB_O_INVALID_NONFINITE,
    GGML_MOE_PB_O_INVALID_OVERFLOW,
    GGML_MOE_PB_O_CAPTURE_Q_OVERFLOW,
    GGML_MOE_PB_O_NO_OUTPUT,            // ★V2 신설(§5.2 — expected/recorded 에만 포함)
    GGML_MOE_PB_O_COUNT,
};

// 슬롯 cache class(SPEC_PHASEB_WIRE §8 cache-class digest · PHASEB_DESIGN §4 상태기계).
#define GGML_MOE_PB_CLASS_NONE      0
#define GGML_MOE_PB_CLASS_DEMAND    1
#define GGML_MOE_PB_CLASS_PROBATION 2

// C3 카운터(SPEC_PHASEB_WIRE §6 표 — summary 키와 1:1·배타 증가).
enum ggml_moe_phaseb_ctr {
    GGML_MOE_PB_C_INTENT_CAPTURED = 0,
    GGML_MOE_PB_C_INTENT_PREDICTED,
    GGML_MOE_PB_C_INTENT_OVERFLOW,
    GGML_MOE_PB_C_PRE_DROPPED_TOO_LATE,
    GGML_MOE_PB_C_PRE_DROPPED_STALE,
    GGML_MOE_PB_C_PRE_SKIPPED_RESIDENT,
    GGML_MOE_PB_C_PRE_SKIPPED_PENDING,
    GGML_MOE_PB_C_CANDIDATE,
    GGML_MOE_PB_C_DROPPED_NO_SLOT_QD,
    GGML_MOE_PB_C_DROPPED_PREFETCH_CAP,
    GGML_MOE_PB_C_DROPPED_RESERVE,
    GGML_MOE_PB_C_ADMITTED,
    GGML_MOE_PB_C_ISSUED,
    GGML_MOE_PB_C_SUBMIT_FAILED,
    GGML_MOE_PB_C_USED_JOIN,
    GGML_MOE_PB_C_USED_PROMOTED,
    GGML_MOE_PB_C_PREFETCH_ERROR_UNPUBLISH,
    GGML_MOE_PB_C_PREFETCH_RETRY,
    GGML_MOE_PB_C_CAPACITY_CANCEL_ATTEMPT,
    GGML_MOE_PB_C_DEMAND_PREFLIGHT_REJECT,
    // ★SPEC_PHASEB_WIRE 부록 14.1(5c-T additive) — 수집 ON 에서만 증가한다(OFF = 전건 0 ·
    //   summary/phase_b_run 표면도 수집 ON 에서만 나타나므로 OFF 런 wire 는 바이트 동일).
    GGML_MOE_PB_C_RESOLVED_WASTED,          // used latch 없이 사용 불가 **확정** 종결
    GGML_MOE_PB_C_CENSORED,                 // censor cut 잔여 OPEN 일괄(관측 종료·미결)
    GGML_MOE_PB_C_ELIGIBLE_DEMAND_ACQUIRE,  // canonical demand acquire 분모
    GGML_MOE_PB_C_WAITED_ACQUIRE,           // 그중 실제 blocked 발생
    GGML_MOE_PB_C_OBSERVED_FRONTIER_MATCH,  // waited 중 required = pred[K] 정확 1칸
    GGML_MOE_PB_C_COUNT,
};

// ★부록 14.2 stamp — `controller_version`(SPEC_PREFETCH_CONTROLLER v1.0 동결본 기준).
//   5c-T 는 actuation 전면 OFF 라 상수는 전원 unsealed(`constants_version="unsealed"`)다.
#define GGML_MOE_PB_CONTROLLER_VERSION "1.0"

// ★부록 14.1 — issued op 의 배타 4분할(SPEC_PREFETCH_CONTROLLER §2 분류 원리).
//   NA = 아직 미확정(성공 publish 후 READY_PROBATION 대기 — 승격/축출/census 로 확정된다).
enum ggml_moe_phaseb_partition {
    GGML_MOE_PB_P_NA              = 0,
    GGML_MOE_PB_P_USED_JOIN       = 1,
    GGML_MOE_PB_P_USED_PROMOTED   = 2,
    GGML_MOE_PB_P_RESOLVED_WASTED = 3,
    GGML_MOE_PB_P_CENSORED        = 4,
};

// ★부록 14.2 — 서버 release 시점 outcome(엔진은 판정하지 않고 전달만 받는다).
enum ggml_moe_phaseb_outcome {
    GGML_MOE_PB_OUTCOME_UNKNOWN = 0,   // 미전달(명시값)
    GGML_MOE_PB_OUTCOME_NORMAL  = 1,
    GGML_MOE_PB_OUTCOME_CANCEL  = 2,
    GGML_MOE_PB_OUTCOME_FAILED  = 3,
};

// ★부록 14.3 — `pb_ledger` 사이드카 사건 6종(닫힌 enum).
enum ggml_moe_phaseb_ledger_kind {
    GGML_MOE_PB_L_ADMISSION_DECISION = 1,
    GGML_MOE_PB_L_INFLIGHT_DELTA     = 2,
    GGML_MOE_PB_L_BLOCKED_INTERVAL   = 3,
    GGML_MOE_PB_L_OP_TERMINAL        = 4,
    GGML_MOE_PB_L_READY_EVICT        = 5,
    GGML_MOE_PB_L_CENSOR_CENSUS      = 6,
};

// ledger `flags` 비트.
#define GGML_MOE_PB_L_F_EVER_JOINED   0x01
#define GGML_MOE_PB_L_F_HAD_WAITER    0x02
#define GGML_MOE_PB_L_F_CONTINUATION  0x04   // 참여 집합 연속 레코드(같은 event_seq)
#define GGML_MOE_PB_L_F_DEFERRED      0x08   // 종결 시점이 아닌 사후 분류 확정(승격 — §14.3 ④)
#define GGML_MOE_PB_L_F_UNSUBMITTED   0x10   // admit+claim 후 제출에 도달하지 못한 채 종결
                                             // (abort cut 등 — disposition 미확정 0xFF 와 세트)

// ---------------------------------------------------------------------------
// wire 레코드 (SPEC_PHASEB_WIRE §2~§5 — 리틀엔디언·고정폭)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)

struct ggml_moe_phaseb_header {         // 128 byte
    char     magic[8];                  // "MOEPB1\0\0"
    uint32_t layout_version;            // 1
    uint32_t file_kind;                 // 1=a0_equiv 2=sample 3=io 4=observe(★8호)
    uint32_t record_bytes;              // 80 / 112 / 112 / 64
    uint32_t phaseb_mode;               // 1=trace 2=equiv 3=observe(★8호)
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
    uint8_t  prefetch_on;
    uint8_t  prefetch_k;
    uint8_t  prefetch_n;
    uint8_t  qd_effective;
    char     kind_tag[12];              // "a0_equiv" / "PB_SAMPLE_V1" / "PB_IO_V1" / "PB_OBSV_V2"(NUL 패딩)
                                        // ★FROZEN_PB_OBSV_V2 §2.1(E0) — kind 4 세대 판별자는 이 필드
                                        //   단독이다(layout_version 은 1 유지 · kind 1~3 불변).
    // ★SPEC_PHASEB_WIRE §13.1 — 구 reserved(off 120·8B) 분할. **kind 1~3 파일은 8B 전부 0**
    //   (구파일·기존 게이트 19 "reserved 전부 0" 불변) · kind 4 파일만 아래 정의값을 싣는다.
    uint8_t  signal_observe_on;         // off 120 — 1=observe 실행 축 활성(kind 4 = 항상 1)
    uint8_t  adapter_id;                // off 121 — 0=none 1=qwen35moe_v1 2=openai_moe_v1
                                        //           3=deepseek2_k26_v1 4=deepseek4_v1
    uint16_t capture_width;             // off 122 — 해소된 capture 폭(=n_embd · guard ≤8192)
    uint8_t  reserved2[4];              // off 124 — 전부 0(게이트 19의 kind 4 적용 범위)
};

struct ggml_moe_phaseb_a0equiv {        // 80 byte · file_kind=1
    uint32_t graph_seq;
    uint32_t request_index;
    uint32_t decode_ordinal;
    uint16_t target_layer;
    uint8_t  flags;                     // b0 valid
    uint8_t  invalid_reason;            // 0 none 1 norm_zero 2 nonfinite 3 overflow
                                        // 4 eligibility_drift 5 capture_miss
    uint16_t a0_ref[16];
    uint16_t a0_cpu[16];
};

struct ggml_moe_phaseb_sample {         // 112 byte · file_kind=2 · "PB_SAMPLE_V1"
    uint32_t graph_seq;
    uint32_t request_index;
    uint32_t decode_ordinal;            // prefill = NA
    uint16_t layer;
    uint16_t source_layer;
    uint8_t  flags;                     // b0 intent_present b1 pf_enabled
                                        // b2 intent_dropped_overflow b3 phase(1=prefill)
    uint8_t  elig_skip_reason;
    uint8_t  k_effective;
    uint8_t  reserved0;
    uint16_t n_required;
    uint16_t n_ready;
    uint16_t n_join;
    uint16_t n_demand_read;
    uint16_t n_join_attempt;
    uint16_t reserved1;
    uint64_t intent_ordinal;
    uint16_t intent_ids[16];
    uint8_t  rank_disposition[16];
    uint64_t capture_qpc;
    uint64_t predict_end_qpc;
    uint32_t blocked_us;
    uint16_t blocked_count;
    uint16_t reserved2;
};

struct ggml_moe_phaseb_io {             // 112 byte · file_kind=3 · "PB_IO_V1"
    uint64_t record_id;
    uint64_t file_offset;
    uint32_t bytes;
    uint16_t expert;
    uint16_t layer;
    uint32_t graph_seq;                 // 보존 owner epoch
    uint32_t request_index;             // 〃
    uint64_t intent_ordinal;            // demand = NA
    uint16_t rank;                      // demand = NA
    uint8_t  origin;                    // 0=demand 1=prefetch
    uint8_t  flags;                     // b0 pending b1 phase(prefill) b2 is_retry
                                        // b3 ever_joined b4 ever_promoted b5 had_waiter_at_terminal
    int32_t  slot;
    uint64_t generation;
    uint64_t submit_event_seq;
    uint64_t submit_qpc;
    uint64_t terminal_event_seq;
    uint64_t terminal_qpc;
    uint64_t publish_seq;               // 0 = 미publish(1-based)
    uint64_t retry_of_submit_event_seq; // is_retry=0 → NA
    uint8_t  qd_demand_at_submit;
    uint8_t  qd_prefetch_at_submit;
    uint8_t  qd_total_at_submit;
    uint8_t  qd_demand_at_terminal;
    uint8_t  qd_prefetch_at_terminal;
    uint8_t  qd_total_at_terminal;
    uint8_t  terminal_status;
    uint8_t  reserved;
};

// ★SPEC_PHASEB_WIRE §13.2 — observe 레코드(file_kind=4 · sentinel "PB_OBSV_V2" · 64 byte).
//   per (warmup 포함 전 요청, decode graph, target = PA layer_ids[1:] 의 각 물리 층) 1개 —
//   eligible/비eligible 표본 **전부** 기록한다(coverage 가 판독의 분모).
//   ★FROZEN_PB_OBSV_V2 §2.2 — 폭 64B 불변(신규 필드 없음). NO_OUTPUT 은 `flags`·`invalid_reason`
//   두 축으로만 표현하며 `flags.b1=1 ⇔ invalid_reason=6` 완전 동치다(§4.1 E2).
struct ggml_moe_phaseb_observe {        // 64 byte · file_kind=4 · "PB_OBSV_V2"
    uint32_t graph_seq;
    uint32_t request_index;
    uint32_t decode_ordinal;
    uint16_t target_layer;              // ★실제 물리 layer id(= moe_layers[target_mlpos])
    uint8_t  flags;                     // b0 valid · b1 NO_OUTPUT(V2) · b2-7 reserved=0
    uint8_t  invalid_reason;            // enum ggml_moe_phaseb_obs_reason(0..6)
    uint16_t pred[16];                  // ordered top-16(내림차순·동점 낮은 id · 잔여 0xFFFF)
    uint64_t predict_end_qpc;           // 진단 전용(0=무효)
    uint8_t  reserved[8];               // 전부 0
};

// ★SPEC_PHASEB_WIRE 부록 14.3 — `pb_ledger` 사이드카(file_kind=5 · "PB_LEDGER_V1" · 304 byte).
//   **수집 ON 에서만 생성**되며 기존 사이드카 계약과 동형이다(전량 메모리 버퍼 → finalize 1회
//   flush · CREATE_NEW · 공통 128B 헤더). 엔진은 원시 사건만 적는다 — 시간가중 동시성·window
//   union·mixed_tuple 은 offline 재구성 몫이다(controller §6).
struct ggml_moe_phaseb_ledger_owner {   // 16 byte — owner ref(참여 집합 원소)
    uint32_t request_epoch;             // 보존 owner epoch
    uint32_t graph_seq;
    uint64_t intent_ordinal;            // demand-origin = NA(u64)
};

// 한 레코드에 담는 참여 집합 원소 수. 초과분은 **같은 `event_seq` 의 연속 레코드**(flags b2)로
// 이어 적으며 **총 레코드 수 상한은 없다** — 참여 집합은 실 유계(in-flight ≤ QD · READY ≤
// n_slots)이므로 전량을 적는 것이 유계다(★r1 M4: 인위적 48개 절단은 launcher 실물 형상
// slots=648/128 에서 실도달 결손이었다). 버퍼가 모자라면 **부분 방출로 위장하지 않고**
// 그룹 전체를 버리고 `complete=0`(RECORD_DROP) 으로 fail-close 한다.
#define GGML_MOE_PB_LEDGER_PART_MAX  12

struct ggml_moe_phaseb_ledger {         // 320 byte · file_kind=5 · "PB_LEDGER_V1"
    uint64_t event_seq;                 // ledger 전용 단조 순번(6종 공용 · 연속 레코드는 동일값)
    uint64_t qpc;                       // 사건 발생 시각(방출 시각 아님)
    uint64_t io_event_seq;              // §5.1 공용 io namespace 상관자(무관 사건 = NA u64)
    uint64_t intent_ordinal;            // demand/NA = NA u64
    int64_t  record_idx;                // -1 = NA
    uint64_t start_qpc;                 // BLOCKED_INTERVAL 전용(그 외 0)
    uint64_t end_qpc;                   // 〃
    // ★r1 M1 — generation 과 decision 의 순번을 **분리**한다(구 판: candidate 순번을
    //   admission_seq 로 오용). `admission_seq` = run_begin 이 발급한 **generation** 순번
    //   (controller §1 stamp 원소) · `decision_seq` = candidate rank 순번(controller §2 결정 표본).
    uint64_t admission_seq;             // owner generation 순번(전 kind 보유 · NA u64 = 미해소)
    uint64_t decision_seq;              // ADMISSION_DECISION 전용(그 외 NA u64)
    uint32_t request_epoch;
    uint32_t graph_seq;
    uint32_t graph_epoch;
    uint32_t n_participants;            // 이 사건의 참여자 총수(연속 레코드 합과 일치)
    uint16_t rank;                      // NA u16 = 없음
    uint16_t layer;                     // 실제 layer id(NA u16 = 없음)
    uint16_t pred_rank;                 // §14.6 경량 판정(NA u16 = 판정 불가)
    uint16_t n_free_slots;              // admission ③ E — **mutation 전** 계산값(전 처분 공통)
    uint16_t n_part_stored;             // 이 레코드가 실제로 담은 참여자 수
    uint8_t  kind;                      // enum ggml_moe_phaseb_ledger_kind
    uint8_t  origin;                    // 0=demand 1=prefetch (무관 = 0xFF)
    uint8_t  disposition;               // enum ggml_moe_phaseb_disp(무관/미확정 = 0xFF)
    uint8_t  terminal_status;           // enum ggml_moe_phaseb_term(무관 = 0)
    uint8_t  partition;                 // enum ggml_moe_phaseb_partition
    int8_t   delta;                     // INFLIGHT_DELTA 전용(+1/−1 · 그 외 0)
    uint8_t  qd_demand;                 // snapshot
    uint8_t  qd_prefetch;
    uint8_t  qd_total;
    uint8_t  cap_total;                 // admission ① 검사 시점 cap
    uint8_t  stamp_k;                   // owner 스탬프 K(§1 — ambient 재조회 금지)
    uint8_t  stamp_n;                   // owner 스탬프 N
    uint8_t  stamp_qd;                  // owner 스탬프 QD(seal 고정 상수)
    uint8_t  tuple_revision;            // owner 스탬프 tuple_revision(5c-T = 0)
    uint8_t  n_expert_used;
    uint8_t  flags;                     // GGML_MOE_PB_L_F_*
    uint8_t  reserved[14];              // 전부 0
    struct ggml_moe_phaseb_ledger_owner participants[GGML_MOE_PB_LEDGER_PART_MAX];
};

#pragma pack(pop)

// PrefetchIntent(C1′) — 결정론 키 = (request_epoch, graph_epoch, source_layer, target_layer, rank).
struct ggml_moe_phaseb_intent {
    uint32_t request_epoch;             // = request_index
    uint32_t graph_epoch;               // = decode_ordinal
    uint32_t graph_seq;
    uint16_t source_layer;              // 실제 layer id
    uint16_t target_layer;              // 실제 layer id
    int32_t  source_mlpos;
    int32_t  target_mlpos;
    uint32_t k;
    // ★5c-T §1 — owner generation 스탬프 N(admission 이 전역 `pb_n` 을 직독하지 않게 하는 결속).
    //   기존 `k` 와 동형으로 intent 발행 시 1회 박히며 이후 불변이다. ★이 필드는 `k` 뒤의
    //   **기존 정렬 패딩(4B)** 을 채우므로 `sizeof(ggml_moe_phaseb_intent)` 는 88 로 불변이다
    //   (metrics `phaseb_mem.intent_record_bytes` 가 이 값을 echo 하므로 OFF 런 바이트 동일성의
    //   전제 — ggml-moe-phaseb.cpp 의 static_assert 가 잠근다). intent digest 입력에는 넣지
    //   않는다(§8 digest 열 불변).
    uint32_t n;
    uint64_t intent_ordinal;
    uint64_t capture_qpc;
    uint64_t predict_end_qpc;
    uint16_t ids[16];                   // ordered top-16(내림차순·동점 낮은 id)
};

// ---------------------------------------------------------------------------
// env/mode 해소 (최초 호출 1회 캐시 — 테스트는 reset_for_test로 재해소)
// ---------------------------------------------------------------------------
GGML_API int32_t      ggml_moe_phaseb_mode(void);
GGML_API bool         ggml_moe_phaseb_enabled(void);       // mode != off
GGML_API const char * ggml_moe_phaseb_mode_name(void);
GGML_API uint64_t     ggml_moe_phaseb_boot_id(void);
GGML_API const char * ggml_moe_phaseb_out_prefix(void);
GGML_API uint32_t     ggml_moe_phaseb_buf_mb(void);
GGML_API uint64_t     ggml_moe_phaseb_now_qpc(void);

// 프리페치 k/N 해소(seal에서 1회 · qd_effective 의존) — 이후 immutable.
GGML_API void         ggml_moe_phaseb_resolve_prefetch(int32_t qd_effective);
GGML_API bool         ggml_moe_phaseb_prefetch_on(void);
GGML_API int32_t      ggml_moe_phaseb_prefetch_k(void);
GGML_API int32_t      ggml_moe_phaseb_prefetch_n(void);
GGML_API const char * ggml_moe_phaseb_prefetch_off_reason(void);

// equiv 모드 fail-close 선검사(§1) — 통과 0, 실패 <0(reason에 사유 문자열).
GGML_API int          ggml_moe_phaseb_equiv_precheck(bool phasea_accuracy, const char ** out_reason);

// ---------------------------------------------------------------------------
// ★정정 6호 — 오힌트 test 표면(SPEC_PHASEB_TEST_HINT.md §1~§3 · qualification 전용)
//   env `MOE_DIRECT_PHASEB_TEST_HINT` 유효값 = exact `rot` 1종. 미설정 = 표면 완전 비활성
//   (production 경로 비용 = relaxed 로드 1회). 활성 시 predictor 의 top-k 산출 직후에
//   `ids'[i] = (ids[i] + n_expert/2) mod n_expert` 를 적용하며, **post-transform ids 가
//   유일 canonical intent** 다(digest·count·ordinal·pb_sample·dispatcher 전부 변형 후 소비).
// ---------------------------------------------------------------------------
#define GGML_MOE_PHASEB_TEST_HINT_NONE 0
#define GGML_MOE_PHASEB_TEST_HINT_ROT  1

GGML_API int32_t      ggml_moe_phaseb_test_hint_rule(void);        // 0=none 1=rot
GGML_API const char * ggml_moe_phaseb_test_hint_rule_name(void);   // "" | "rot"
GGML_API int32_t      ggml_moe_phaseb_test_hint_offset(void);      // ⌊n_expert/2⌋ (비활성 0 · seal 확정)
// summary `test_hint_applied`(SPEC_PHASEB_WIRE §7.3) — ordinal 발급·enqueue **성공**한
// post-transform intent 총수(변형 호출 수 아님 · 프로세스 누적 · reset_for_test 로 0).
GGML_API uint64_t     ggml_moe_phaseb_test_hint_applied(void);
// init fail-close 선검사(§1) — 통과 0, 실패 <0(reason에 사유 문자열).
//   qualify = 호출자(direct.cpp)의 `MOE_DIRECT_QUALIFY=1` latch.
//   실패 3종: ①qualifier 부재 ②무효값 ③`MOE_DIRECT_PHASEB=equiv` 동시 설정.
GGML_API int          ggml_moe_phaseb_test_hint_precheck(bool qualify, const char ** out_reason);

// ---------------------------------------------------------------------------
// ★계열 신호 capability adapter (SPEC_FAMILY_SIGNAL v0.2 §1 — 2단계 "qwen 무변경 adapter화")
//   seal **이전**에 `ggml_moe_phaseb_note_family()` 로 descriptor 입력을 선언하면
//   `ggml_moe_phaseb_note_seal()` 이 **닫힌 매핑**(general.architecture + hparams +
//   tensor inventory)으로 adapter_id 를 1회 해소한다(이후 immutable).
//   미선언·불일치 = `none` = **신호 전면 비활성**(프리페치 OFF + off_reason `adapter_none`).
//   기동은 계속된다(프리페치는 부가 기능 · 직독 본체 무영향).
//   ★3단계 등재는 `qwen35moe_v1`·`openai_moe_v1` 2종이다 — deepseek2_k26_v1 은 표에 자리만
//   있고 미등재(해소 시 `none`)이며 4단계에서 활성화한다.
//   ★DSV4 단계 등재는 `deepseek4_v1`(DSV4_ADAPTER_DESIGN v0.2 §1.5) — hash 라우팅 층(0..2)을
//   신호 대상에서 정적 제외하는 **층 마스크**를 가진 첫 계열이다.
// ---------------------------------------------------------------------------
#define GGML_MOE_PHASEB_ADAPTER_NONE       0
#define GGML_MOE_PHASEB_ADAPTER_QWEN35MOE  1   // 등재(2단계)
#define GGML_MOE_PHASEB_ADAPTER_OPENAI_MOE 2   // 등재(3단계)
#define GGML_MOE_PHASEB_ADAPTER_DEEPSEEK2  3   // 미등재(4단계)
#define GGML_MOE_PHASEB_ADAPTER_DEEPSEEK4  4   // 등재(DSV4 — V4-Flash)

// norm_tensor(§1) — 스냅샷 source norm 텐서 선택(llama-context.cpp 배선이 소비).
#define GGML_MOE_PHASEB_NORM_NONE      0
#define GGML_MOE_PHASEB_NORM_ATTN_POST 1   // blk.%d.post_attention_norm.weight
#define GGML_MOE_PHASEB_NORM_FFN       2   // blk.%d.ffn_norm.weight

// selection_fn(§1) — predictor 의 top-k 산출식.
#define GGML_MOE_PHASEB_SEL_NONE              0
#define GGML_MOE_PHASEB_SEL_TOPK_RAW          1   // qwen35moe: TopK(Wx)
#define GGML_MOE_PHASEB_SEL_TOPK_BIAS         2   // gpt-oss  : TopK(Wx + ffn_gate_inp_b)   [3단계]
#define GGML_MOE_PHASEB_SEL_TOPK_SIGMOID_BIAS 3   // K2.6     : TopK(sigmoid(Wx) + exp_probs_b) [4단계]
// ★DSV4(deepseek4_v1) — TopK(sqrt(softplus(Wx)) + exp_probs_b).
//   1차 소스: src/llama-graph.cpp:1858(probs = sqrt(softplus(logits))) · :1868(selection_probs =
//   probs + exp_probs_b) · :1912(argsort_top_k(selection_probs)). `weights_norm`/`scale` 은 선택
//   ID 가 아니라 혼합 가중치에만 적용되므로 신호 산식에 들어가지 않는다.
//   ★bias 가산 시점이 `topk_bias`(GEMV 직후 = logit bias)와 **다르다**(변환 뒤 = selection bias).
#define GGML_MOE_PHASEB_SEL_TOPK_SQRT_SOFTPLUS_BIAS 4

// bias_snapshot(§1) — 선택식이 요구하는 bias 텐서 종류(부재 포함이 계약의 일부).
#define GGML_MOE_PHASEB_BIAS_NONE        0
#define GGML_MOE_PHASEB_BIAS_GATE_INP_B  1   // blk.%d.ffn_gate_inp.bias
#define GGML_MOE_PHASEB_BIAS_EXP_PROBS_B 2   // blk.%d.exp_probs_b.bias

// 층별 tensor inventory 입력(§1) — 생산자는 **포인터만** 넘기고 판정은 전부 모듈이 한다.
//   ★계수 위임 금지: 생산자가 미리 센 개수를 받으면 "비-null 인데 이름이 다른" 텐서가 계수 0
//   으로 흡수돼 **부재로 오인**된다(qwen 의 "bias 부재 필수"가 뚫린다). 그래서 presence
//   (포인터 non-null)와 validity(이름 완전일치 + F32 + ne[0..3] 전량 + contiguous + finite
//   전량)를 모듈이 **분리 판정**하고, present∧¬valid = fail-close 로 잡는다.
struct ggml_moe_phaseb_layer_in {
    const struct ggml_tensor * attn_post_norm;   // blk.%d.post_attention_norm.weight {n_embd}
    const struct ggml_tensor * ffn_norm;         // blk.%d.ffn_norm.weight            {n_embd}
    const struct ggml_tensor * ffn_gate_inp;     // blk.%d.ffn_gate_inp.weight        {n_embd, n_expert}
    const struct ggml_tensor * ffn_gate_inp_b;   // blk.%d.ffn_gate_inp.bias          {n_expert}
    const struct ggml_tensor * ffn_exp_probs_b;  // blk.%d.exp_probs_b.bias           {n_expert}
    // ★DSV4 텐서 파티션 축(DSV4_ADAPTER_DESIGN §1.5) — hash 라우팅 층의 토큰ID 정적 룩업 표
    //   `blk.%d.ffn_gate_tid2eid.weight` {n_expert_used, n_vocab} I32. 신호 경로는 이 텐서를
    //   **읽지 않는다** — `exp_probs_b` 와의 **상호 배타 파티션**(0..hash-1 ↔ hash..n-1)을
    //   capability 로 봉인하기 위한 presence/validity 입력이다. 다른 계열은 nullptr.
    const struct ggml_tensor * ffn_gate_tid2eid;
};

// descriptor 입력(§1) — 생산자가 **live GGUF hparams + 로드된 텐서 포인터**로 채운다.
//   `layers`/`moe_layers` 는 이 호출이 반환할 때까지만 유효하면 된다(모듈이 즉시 판정·복사).
struct ggml_moe_phaseb_family_in {
    const char * arch;               // general.architecture
    int32_t expert_gating_func;      // llama_expert_gating_func_type(0=none 1=softmax 2=sigmoid …)
    int32_t n_expert_groups;         // ★0(KV 부재) 또는 1 만 지원(그룹 top-k 미구현) — 그 외 비활성
    int32_t n_embd;
    int32_t n_expert;
    int32_t n_expert_used;
    int32_t n_layer_exec;            // NextN 제외 실행 층 수
    int32_t n_moe_layers;            // ★실제 moe_layers[] 길이(live routed 텐서 스캔 결과)
    const int32_t * moe_layers;      // 〃 오름차순 물리 layer id(leading-dense·연속성·seal 대조 원천)
    const struct ggml_moe_phaseb_layer_in * layers;   // n_layer_exec 개(인덱스 = il)
    // ── ★DSV4 capability 입력(DSV4_ADAPTER_DESIGN §1.5 "hparams 검증형 capability" 확정 목록) ──
    //   전 계열 공통으로 채우되(값 없는 계열은 hparams 기본값 0), **요구값이 명시된 행에서만**
    //   대조된다(기존 3행은 전건 "미검사" = 거동 불변).
    int32_t hash_layer_count;        // deepseek4.hash_layer_count(hash 라우팅 층 수 · 기타 계열 0)
    int32_t n_expert_shared;         // hparams.n_expert_shared
    int32_t hc_mult;                 // deepseek4 하이퍼커넥션 스트림 수(hparams.dsv4_hc_mult)
    int32_t expert_weights_norm;     // hparams.expert_weights_norm(0/1)
    float   expert_weights_scale;    // hparams.expert_weights_scale
};

// seal 이전 1회 선언(재선언 = 덮어쓰기). nullptr = 선언 취소.
// ★이 호출이 인벤토리를 **즉시 전량 판정**한다(finite 스캔 포함 — 호출 순서·env 상태에
//   의존하지 않는다). 비용 = router+norm 텐서 1회 선형 주사(모델 로드 1회·추론 경로 0).
GGML_API void         ggml_moe_phaseb_note_family(const struct ggml_moe_phaseb_family_in * in);
GGML_API int32_t      ggml_moe_phaseb_adapter_id(void);       // 미해소 = NONE
GGML_API const char * ggml_moe_phaseb_adapter_name(void);     // "" | "qwen35moe_v1" | …
GGML_API const char * ggml_moe_phaseb_adapter_reason(void);   // 해소 성공 = "" · 실패 = 사유
GGML_API int32_t      ggml_moe_phaseb_norm_tensor_kind(void);
GGML_API int32_t      ggml_moe_phaseb_selection_fn(void);
GGML_API int32_t      ggml_moe_phaseb_bias_kind(void);
GGML_API int32_t      ggml_moe_phaseb_capture_width(void);    // 해소된 capture 폭(=n_embd) · 미해소 0
// ★observe_descriptor(§13.4 exact object) 의 wire echo 원천 — 표가 단일 출처다.
GGML_API const char * ggml_moe_phaseb_family_arch(void);       // 선언 descriptor 의 general.architecture
GGML_API const char * ggml_moe_phaseb_norm_tensor_name(void);  // "" | "attn_post_norm" | "ffn_norm"
GGML_API const char * ggml_moe_phaseb_selection_fn_name(void); // "" | "topk_raw" | "topk_bias" |
                                                               // "topk_sigmoid_bias" | "topk_sqrt_softplus_bias"
GGML_API const char * ggml_moe_phaseb_bias_snapshot_name(void);// "" | "ffn_gate_inp_b[]" | "ffn_exp_probs_b[]"
GGML_API int32_t      ggml_moe_phaseb_family_n_expert(void);
GGML_API int32_t      ggml_moe_phaseb_family_n_expert_used(void);
GGML_API int32_t      ggml_moe_phaseb_moe_layer_count(void);   // seal moe_layers[] 길이
GGML_API int32_t      ggml_moe_phaseb_moe_layer_at(int32_t i); // seal moe_layers[i](-1=범위 밖)
// ★계열 capability `output_crop_before_last_moe`(FROZEN_PB_OBSV_V2 §7 · E6) — **폐쇄형 선언**이다.
//   표에 명시된 계열만 값을 가지며(openai_moe_v1=true · qwen35moe_v1=false), 미등재·미해소 계열은
//   "자동 false 상속"이 아니라 **observe 신호 비활성**(adapter 해소 실패 = obs_on=false)이다.
//   판독기는 adapter_id 로 이 값을 재추론하지 않고 descriptor echo 된 값만 소비한다.
GGML_API bool         ggml_moe_phaseb_output_crop_before_last_moe(void);

// ---------------------------------------------------------------------------
// ★계열 층 마스크(DSV4_ADAPTER_DESIGN v0.2 §1.5 — "전 표면 동일 마스크 의무")
// ---------------------------------------------------------------------------
//   일부 계열은 앞쪽 층의 전문가 선택이 학습 라우팅이 아니다(deepseek4 hash 층 = 토큰ID 정적
//   룩업). 그런 층을 target 으로 하는 예측은 **정의상 성립하지 않으므로** 신호 축에서 정적
//   제외한다. 마스크는 표에서 읽은 `first_target_layer` 하나로 표현되며(그 미만 target 전부
//   제외), resolver·tensor inventory·snapshot seal·capture eligibility·predictor·descriptor
//   echo 가 **모두 같은 술어**를 쓴다(표면별 재해석 금지).
//   · 기존 3계열 = 1(= 종전 "target 은 1..n-1" 그대로 · 거동 불변)
//   · deepseek4_v1 = 3(hash_layer_count) → learned target 3..n-1 · capture source 2..n-2
GGML_API int32_t      ggml_moe_phaseb_signal_first_target_layer(void);  // 미해소 = 1(중립)
// 표가 1 보다 큰 마스크를 선언한 계열인가(= descriptor 조건부 필드 방출 조건).
GGML_API bool         ggml_moe_phaseb_signal_layer_mask_active(void);
// 이 물리 target 층이 신호 대상인가(target < first_target_layer = 정적 제외).
GGML_API bool         ggml_moe_phaseb_signal_target_eligible(int32_t target_layer);

// ---------------------------------------------------------------------------
// ★denominator/source norm 원소별 exact-zero inventory
//   (FROZEN_NORM_INVENTORY §3·§4 — metrics `observe_norm_inventory` exact block 의 단일 출처)
// ---------------------------------------------------------------------------
//   엔진의 보정 분모는 **source 층의 norm 가중치**이고 원소가 하나라도 `0.0f` 이면 그 층을
//   source 로 쓰는 target 의 예측이 전량 무효(NORM_ZERO)가 된다. 런타임은 첫 zero 에서
//   break 하므로 "몇 개·어디인지"는 **seal 시점 static 인벤토리만** 알 수 있다.
//   · 술어      : `== 0.0f` 정확 비교(epsilon 금지 · `-0.0f` 도 zero)
//   · 스캔 시점 : note_family 의 finite 전량 순회에 **합류**(별도 패스 없음)
//   · source 층 : `moe_layer_ids[:-1]`(마지막 관리 MoE 층은 affected target 부재로 제외)
//   · 좌표      : 0-based 오름차순 u32 **전량**(절단 없음) · digest = FNV-1a 64,
//                 입력 바이트 = little-endian `(u16 source_layer)(u32 zero_count)(u32 index...)`
//   ★zero 존재는 adapter 해소를 **탈락시키지 않는다**(즉 FAIL 아님) — 측정값일 뿐이다.
//   ★`measured=false` 는 "zero 0건"이 아니라 **미측정**이다(판독기 fail-close · 기본값 0 합성
//     금지). 소비자는 반드시 measured 를 먼저 확인한다.
#define GGML_MOE_PHASEB_NORM_INVENTORY_VERSION 1

GGML_API bool     ggml_moe_phaseb_norm_inventory_measured(void);
GGML_API int32_t  ggml_moe_phaseb_norm_inventory_scanned(void);        // source_layers_scanned
GGML_API int32_t  ggml_moe_phaseb_norm_inventory_zero_layers(void);    // = total_zero_source_layers
GGML_API uint64_t ggml_moe_phaseb_norm_inventory_zero_elements(void);  // total_zero_elements
// i 번째 zero source 층 레코드. 0=성공 · <0=미측정·범위 밖.
// ★`source_layer` 와 `affected_target` 은 **둘 다 물리 layer id** 이며 동시 탑재가 계약이다
//   (source 축과 target 축의 표기 혼선을 구조적으로 제거한다).
GGML_API int      ggml_moe_phaseb_norm_inventory_at(int32_t i, int32_t * out_source_layer,
                                                    int32_t * out_affected_target,
                                                    uint32_t * out_zero_count,
                                                    uint64_t * out_zero_digest);
// i 번째 레코드의 k 번째 zero 좌표(오름차순). 범위 밖 = GGML_MOE_PHASEB_NA_U32.
GGML_API uint32_t ggml_moe_phaseb_norm_inventory_zero_index(int32_t i, uint32_t k);

// ---------------------------------------------------------------------------
// ★observe 모드(SPEC_PHASEB_WIRE §13 · SPEC_FAMILY_SIGNAL §2) — 계열 신호 qualification
// ---------------------------------------------------------------------------
// observe 실행 축(§2.1). seal 에서 1회 확정: mode==observe ∧ 전제 통과 ∧ adapter 해소.
GGML_API bool         ggml_moe_phaseb_signal_observe_on(void);
// seal 전제 검사(위반 = init fail-close). ★note_seal **뒤**에 호출한다(adapter 해소 필요).
//   전제: supported adapter · MOE_DIRECT_PHASEA=timing · TEST_HINT 부재 ·
//         MOE_DIRECT_PREFETCH_K/N raw key 둘 다 부재(빈 값으로 존재해도 실패).
//   통과 0 · 실패 <0(out_reason 에 영문 사유).
GGML_API int          ggml_moe_phaseb_observe_precheck(const char ** out_reason);
GGML_API uint64_t     ggml_moe_phaseb_observe_counter(int32_t ctr);   // enum ggml_moe_phaseb_obs_ctr
GGML_API uint64_t     ggml_moe_phaseb_observe_buf_bytes(void);        // seal 정적 예약 바이트(§13.4)
GGML_API uint64_t     ggml_moe_phaseb_observe_records(void);
// observe 사이드카(모듈 소유 버퍼) 기록. 반환 written(1/0) · 레코드 수/digest 출력.
GGML_API int          ggml_moe_phaseb_write_observe(uint64_t * out_records, uint64_t * out_digest);
// selftest 전용 — observe 레코드 직독(i 번째). 0=성공.
GGML_API int          ggml_moe_phaseb_test_observe_record(uint64_t idx, void * out_record, uint32_t bytes);
// ★NO_OUTPUT 권위 표면(`phase_b_run.no_output_graphs[]` · FROZEN_AUTHORITY_V3 §3.2.1 A5) 조회 창구.
//   metrics 레코드로 나가는 것과 **같은 원천**을 selftest 가 직독한다(요청별 배열 · request_index 는
//   부모 레코드 소유이므로 원소에 중복 탑재하지 않는다).
GGML_API uint64_t     ggml_moe_phaseb_no_output_count(uint32_t request_index);
GGML_API int          ggml_moe_phaseb_no_output_at(uint32_t request_index, uint64_t i,
                                                   uint32_t * out_graph_seq, uint32_t * out_decode_ordinal,
                                                   uint16_t * out_target_layer);
// selftest 전용 — 억제 가드가 intent 발행 경로를 끊은 횟수(§1.5 "enqueue 전 억제"의 **직접** 관측).
//   observe 축은 speculative enqueue 가 구조적으로 0 이라 "enqueue 0" 검사만으로는 가드의 유무를
//   구분하지 못한다(가드를 지워도 통과). 이 계수가 그 구분자다 — wire/metrics 표면이 아니다.
GGML_API uint64_t     ggml_moe_phaseb_test_no_output_suppressed(void);

// ---------------------------------------------------------------------------
// seal 통지 — 모델 상수 확정 + (프리페치 ON이면) capture 풀/intent_q/worker 기동
// ---------------------------------------------------------------------------
// ★R1 #5 ② — n_layer_exec/n_embd/rms_eps 까지 받아 **seal 시점에 스냅샷 형상(bytes)을 확정**한다
//   (metrics header phaseb_mem 에 실값이 실려야 §7.2 게이트 ⓐ가 성립). 컨텍스트 생성 단계는
//   데이터 채움(snapshot_layer)만 담당한다. n_layer_exec<2 또는 n_embd<=0 이면 형상 미확정.
GGML_API int  ggml_moe_phaseb_note_seal(int32_t n_expert, int32_t n_expert_used,
                                        int32_t n_moe_layers, int32_t n_slots,
                                        int32_t qd_effective, const int32_t * moe_layers,
                                        const uint64_t * layer_stride_bytes,
                                        int32_t n_layer_exec, int32_t n_embd, float rms_eps);

// ★R1 #2 ① — intent enqueue 가 dispatcher 를 깨우는 이벤트 등록(direct.cpp command_event).
GGML_API void ggml_moe_phaseb_set_wake_event(void * event_handle);
// ★R1 #12 — "PB off ∧ prefetch off" fast-gate(호출부가 1분기로 전부 건너뛴다).
GGML_API bool ggml_moe_phaseb_all_off(void);

// ---------------------------------------------------------------------------
// 요청(run)/graph 경계 — epoch 투영(§2.1) 원천
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_phaseb_run_begin(void);
GGML_API void ggml_moe_phaseb_run_end(void);
// ★n_outputs = **서버가 확정한** 이 graph(ubatch)의 출력 행 수(FROZEN_PB_OBSV_V2 §1.5 생산 조건).
//   사후 추론 금지 계약이므로 인자로 **강제 전달**한다(생략 가능한 별도 setter 를 두지 않는 이유 —
//   호출부가 빠지면 컴파일이 실패해야 "미전달=0 으로 오인"이 원천 차단된다).
GGML_API void ggml_moe_phaseb_graph_begin(uint8_t phase, uint32_t n_tokens, uint32_t n_outputs);
GGML_API void ggml_moe_phaseb_graph_end(void);
GGML_API void ggml_moe_phaseb_current_epoch(uint32_t * out_request_index, uint32_t * out_decode_ordinal,
                                            uint32_t * out_graph_seq, uint8_t * out_phase);
GGML_API uint64_t ggml_moe_phaseb_total_requests(void);

// ---------------------------------------------------------------------------
// router/post-norm host 스냅샷 (§3 — router l=1..n-1 · post-norm l=0..n-1)
// ---------------------------------------------------------------------------
GGML_API int  ggml_moe_phaseb_snapshot_configure(int32_t n_layer_exec, int32_t n_embd,
                                                 int32_t n_expert, int32_t n_expert_used,
                                                 float rms_eps);
GGML_API bool ggml_moe_phaseb_snapshot_needed(void);
GGML_API int  ggml_moe_phaseb_snapshot_layer(int32_t layer, const struct ggml_tensor * router,
                                             const struct ggml_tensor * post_norm);
// ★계열 selection-bias 스냅샷(SPEC_FAMILY_SIGNAL §1 `bias_snapshot`) — bias 를 요구하는 adapter
//   에서만 배선한다(layer 1..n-1 · 요구 없는 계열은 호출 자체가 없어야 하고 호출되면 -1).
GGML_API int  ggml_moe_phaseb_snapshot_bias(int32_t layer, const struct ggml_tensor * bias);
GGML_API int  ggml_moe_phaseb_snapshot_seal(void);
GGML_API uint64_t ggml_moe_phaseb_router_snapshot_bytes(void);
GGML_API uint64_t ggml_moe_phaseb_norm_snapshot_bytes(void);
GGML_API uint64_t ggml_moe_phaseb_bias_snapshot_bytes(void);

// selftest 전용 — 합성 가중치 직접 주입(실 텐서 없이 A′ 경로를 행사).
GGML_API int  ggml_moe_phaseb_test_snapshot_raw(int32_t layer, const float * router,
                                                const float * post_norm);
// selftest 전용 — 합성 selection-bias 직접 주입(bias 계열 산식 행사).
GGML_API int  ggml_moe_phaseb_test_snapshot_bias_raw(int32_t layer, const float * bias);

// ---------------------------------------------------------------------------
// capture 훅 (ggml-cpu.c · ith==0 · row grouping 후 · acquire 앞)
//   src_layer/src_mlpos = 이 mul_mat_id가 속한 층. n_tokens = ids->ne[1] 원값.
//   반환 = 실제로 capture_q에 실렸으면 1(진단용).
// ---------------------------------------------------------------------------
GGML_API int  ggml_moe_phaseb_capture(int32_t src_layer, int32_t src_mlpos,
                                      const void * src1_data, int32_t src1_type,
                                      int32_t n_embd, int32_t n_rows, int32_t n_tokens,
                                      int32_t src1_contiguous);

// LoRA 교체 감지 통지(런 중 1회라도 서면 이후 전 capture는 LORA_SWAP skip).
GGML_API void ggml_moe_phaseb_note_lora_swap(void);

// equiv 모드 a0_ref tap — llama-context의 eval callback 래퍼가 "attn_residual-<il>"에서 호출.
GGML_API void ggml_moe_phaseb_tap_a0(int32_t src_layer, const float * h, int32_t n);

// ---------------------------------------------------------------------------
// intent_q — dispatcher 소비(g_mod.mutex 보유 상태에서 호출)
// ---------------------------------------------------------------------------
// 헤드 intent 조회(pop 아님). 없으면 0. 있으면 1 + out 채움.
GGML_API int  ggml_moe_phaseb_intent_peek(struct ggml_moe_phaseb_intent * out);
GGML_API void ggml_moe_phaseb_intent_pop(void);
GGML_API int  ggml_moe_phaseb_intent_depth(void);

// ---------------------------------------------------------------------------
// PB_FINALIZE_CUT 협조 (§0.1)
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_phaseb_cut_close_capture(void);   // ① 신규 capture 거절
GGML_API void ggml_moe_phaseb_cut_drain_predictor(void); // ② capture drain + worker stop/join
// ★★T-1(교차 r3 S-2 ①) — 위의 **유계판**. 반환 1=drain 완료 / 0=timeout.
//   dispatcher 유계 종료가 timeout 계측 **전에** predictor drain 을 동기 호출하므로,
//   이게 INFINITE 면 predictor 정지 회귀가 **timeout 전에 교착**해 종단 유계가 깨진다.
//   `timeout_ms < 0` = 구 거동(INFINITE) — 프로덕션 경로는 래퍼로 불변.
GGML_API int  ggml_moe_phaseb_cut_drain_predictor_bounded(int32_t timeout_ms);
// E-2a (Codex r2 blocker 2) - the SAME drain, taking the ONE deadline instead of a relative
// timeout. The predictor drain has two internal waits (capture idle, then the worker join) and
// the caller has a third (the dispatcher join). Handing each of them the caller's timeout_ms
// gives every stage a FRESH full budget, so a 30 s bound can spend 10 s + 30 s + 30 s and still
// report success. This form is handed an absolute instant on the steady clock and every stage
// spends what is LEFT of it.
//   deadline_abs_ms <  0 : no bound at all (INFINITE) - the unchanged production stop policy.
//   deadline_abs_ms >= 0 : milliseconds on std::chrono::steady_clock, same epoch the caller
//                          reads. Already past = budget 0 = check once, do not wait.
// `_bounded` above is now the thin wrapper that turns a relative timeout into one of these.
GGML_API int  ggml_moe_phaseb_cut_drain_predictor_until(int64_t deadline_abs_ms);
GGML_API bool ggml_moe_phaseb_capture_closed(void);
// ★R1 #3 ② — 이 래치가 서기 전에는 dispatcher graceful exit 금지.
GGML_API bool ggml_moe_phaseb_predictor_drained(void);

// ★★R-2(권위 = `bench/techdev/repack_space/SPEC_PB_VIRTUAL_RUNTIME_R1.md` §2-2 "실패 후 자원
//   잔존 0" · 교차 r1 F1) — **실패한 seal 의 되돌림**(프로덕션 경로 · test-only 아님).
//   `note_seal` 은 `sealed`·모델 배열·adapter·스냅샷 형상·capture/intent 자원·predictor worker 를
//   **먼저 게시**하므로, 그 뒤 단계(metrics·dispatcher 기동)가 실패하면 호출자가 전건을 되돌려야
//   한다. 구 판에는 그 수단이 test-only `reset_for_test` 밖에 없었다.
//   ★`reset_for_test` 와 다른 점: 카운터·digest·ledger 등 **관측 이력은 보존**한다. 이 함수의
//     계약은 "**자원과 seal 상태**의 잔존 0"이지 이력 소거가 아니다(실패 진단을 지우지 않는다).
//   ★멱등 — 미seal 상태에서 불러도 안전하다.
GGML_API void ggml_moe_phaseb_unseal_failed(void);

// ---------------------------------------------------------------------------
// 카운터 / digest / 요청별 귀속(§0.1 보존 epoch)
// ---------------------------------------------------------------------------
GGML_API void     ggml_moe_phaseb_count(int32_t ctr, uint32_t request_epoch, uint64_t delta);
GGML_API uint64_t ggml_moe_phaseb_counter(int32_t ctr);
GGML_API void     ggml_moe_phaseb_count_skip(int32_t reason, uint32_t request_epoch);
GGML_API uint64_t ggml_moe_phaseb_skip_counter(int32_t reason);
GGML_API void     ggml_moe_phaseb_count_split(uint32_t request_epoch, uint32_t required, uint32_t ready,
                                              uint32_t join, uint32_t demand_read, uint32_t join_attempt);
GGML_API void     ggml_moe_phaseb_count_blocked(uint32_t request_epoch, uint64_t us, uint32_t count);
GGML_API void     ggml_moe_phaseb_note_sidecar_record(int32_t file_kind, uint32_t request_epoch);
GGML_API void     ggml_moe_phaseb_note_qd(uint32_t qd_demand, uint32_t qd_prefetch, uint32_t qd_total);
GGML_API void     ggml_moe_phaseb_qd_max(uint32_t * out_demand, uint32_t * out_prefetch, uint32_t * out_total);
GGML_API void     ggml_moe_phaseb_blocked_totals(uint64_t * out_us, uint64_t * out_count);

// ---------------------------------------------------------------------------
// ★5c-T 수집 텔레메트리(SPEC_PHASEB_WIRE 부록 14 · additive only)
//   전 표면이 **수집 ON 에서만** 산출된다 — OFF 런의 wire 는 바이트 수준 종전과 동일하다.
// ---------------------------------------------------------------------------
// §14.4 — env `MOE_DIRECT_PHASEB_COLLECT`(off 기본|on) · seal 1회 해소 · 이후 immutable.
GGML_API bool     ggml_moe_phaseb_collect_on(void);

// ---------------------------------------------------------------------------
// ★5c-A1 identity·adapt 축(SPEC_PREFETCH_5CA §1-6 · §1-6-1 · §6-3 · additive only)
//   런처가 explicit environment block 으로 넘기는 `activation_bundle_view`(12키)와
//   `activation_grant`(8키)를 엔진이 닫힌 grammar 로 읽고 fail-closed 로 판정한다.
//   엔진은 payload 파일을 읽지 않고 SHA-256 을 계산하지도 않는다 — digest 는 **옮기는 값**이다.
// ---------------------------------------------------------------------------
// §6-3 — grant verifier 단 0~4 전건 통과에서만 켜지고 seal 1회로 해소된다(이후 immutable).
//   ★5c-A1 에는 발급자가 없어 실측 전건 false 이며, 여기 서 있는 것은 음성 경로다.
GGML_API bool     ggml_moe_phaseb_adapt_on(void);
// §1-6-1 — view snapshot 이 accepted 인가(V-1 형식 · V-2 fnv · V-3 expected==actual).
GGML_API bool     ggml_moe_phaseb_bundle_view_accepted(void);
// §6-3 — grant verifier 가 멈춘 단의 거부 사유(통과 시 빈 문자열). 프로세스 수명 문자열.
GGML_API const char * ggml_moe_phaseb_activation_grant_refusal(void);
// §6-3 — controller 객체 할당 계수(읽기 전용 · 5c-A1 전건 0).
GGML_API uint64_t ggml_moe_phaseb_controller_alloc_count(void);
// §14.2 — 서버 release 시점 outcome 전달(현 요청 귀속). 미호출 = "unknown".
GGML_API void     ggml_moe_phaseb_run_outcome(int32_t outcome);
// ★r1 M1 — generation lifecycle. `run_begin` 이 generation admission 순번+QPC 를 발급하고
//   `run_end` 가 terminal 순번+QPC 를 닫는다(5c-T §4-① "모든 admission 의 순번·시각").
//   owner 회계는 request_epoch 로 이 값을 되읽는다(ambient 재조회 아님 — owner 고정값).
GGML_API uint64_t ggml_moe_phaseb_run_admission_seq(uint32_t request_epoch);
// ★r1 M2 — owner generation 의 박제 튜플 되읽기(K/N/QD/tuple_revision/admission_seq).
//   owner epoch 로만 접근하며 전역 현재값(ambient)을 보지 않는다.
GGML_API void     ggml_moe_phaseb_run_stamp(uint32_t request_epoch, int32_t * out_k, int32_t * out_n,
                                            int32_t * out_qd, int32_t * out_rev,
                                            uint64_t * out_admission_seq);
// ★r1 M1 — candidate rank 순번은 generation 순번과 **분리**된 결정 표본 축이다(controller §2).
GGML_API uint64_t ggml_moe_phaseb_next_decision_seq(uint32_t request_epoch);
// §14.2 — 이 요청의 blocked interval 1건 통지(phase_b_run.blocked_intervals).
GGML_API void     ggml_moe_phaseb_note_blocked_interval(uint32_t request_epoch);
// §14.3 — ledger 사건 1건 적재. 수집 OFF = no-op. 반환 1=적재 0=버퍼 부족(드롭·fail-close).
GGML_API int      ggml_moe_phaseb_ledger_push(const struct ggml_moe_phaseb_ledger * rec);
// ★r1 M4 — 연속 레코드 그룹의 **원자 적재**: 전량 예약에 실패하면 한 건도 적지 않고 0 을
//   돌려준다(부분 방출로 완전한 ledger 를 위장하지 않는다 · complete=0 으로 fail-close).
GGML_API int      ggml_moe_phaseb_ledger_push_group(const struct ggml_moe_phaseb_ledger * recs,
                                                    uint32_t n_recs);
GGML_API uint64_t ggml_moe_phaseb_ledger_next_event_seq(void);
GGML_API uint64_t ggml_moe_phaseb_ledger_records(void);
// finalize 1회 flush(다른 사이드카와 동형 · 수집 OFF = 미기록·0건).
GGML_API int      ggml_moe_phaseb_write_ledger(uint64_t * out_records, uint64_t * out_digest);
// ★테스트 전용 one-shot 제출 실패 주입(oracle matrix §6-2 ⓑ "submit_failed 주입 시 candidate
//   closure"). 다음 **speculative(fail-soft) 제출 1건**을 ReadFile 실패로 강등한다. 프로덕션은
//   항상 0이며 소비 비용은 relaxed 로드 1회다(기존 test_force_sync_next/test_pb_error_next 선례
//   동형). ★_take 는 소비자(direct.cpp io_submit_locked) 전용 — 1회 소비 후 자동 해제.
GGML_API void     ggml_moe_phaseb_test_submit_fail_next(int32_t on);
GGML_API int      ggml_moe_phaseb_test_submit_fail_take(void);

// ★★R-2 게이트 전용 one-shot **predictor worker 생성 실패** 주입(권위 =
//   `bench/techdev/repack_space/SPEC_PB_VIRTUAL_RUNTIME_R1.md` §2-2 "실패 후 자원 잔존 0").
//   arm 후 다음 `pb_worker_start` 1건이 스레드를 만들지 않고 실패를 돌려준다 — `note_seal` 은
//   그때 rc=-1 을 내보내고 `pb_alloc_b2_resources` 는 **이미 잡은** capture 풀·intent_q·observe
//   버퍼를 자기 unwind 로 되돌려야 한다. 실 `_beginthreadex` 실패는 합성으로 만들 수 없으므로
//   이 주입 없이는 그 분기에 관문을 세울 수 없다(교차 r1 F1 이 놓친 이유이기도 하다 — M4 는
//   첫 seal 거부만 봤고 rollback 잔존은 못 봤다). 프로덕션은 항상 0 · 소비=relaxed 로드 1회.
GGML_API void     ggml_moe_phaseb_test_worker_start_fail_next(int32_t on);

// ★테스트 전용 **중복 terminal 전이** 프로브(oracle r2 M7-1). arm 후 다음 prefetch-origin
//   terminal 1건에서 엔진이 **같은 live op 의 4분할 전이를 두 번째로 시도**하고, 그 2차 시도가
//   카운터·QD·ledger 를 하나도 늘리지 않았음을 실측해 결과로 남긴다(배타 래치의 양성 증명).
//   프로덕션은 항상 0이며 소비 비용은 relaxed 로드 1회다.
// ★테스트 전용 owner 튜플 override(oracle r3 수리 2). 설정하면 **다음에 생성되는 generation
//   버킷**의 박제 튜플이 이 값으로 고정된다 — 서로 다른 owner stamp 를 **실 producer 가 실
//   ledger 파일에 쓰게** 만들어 mixed_tuple 양성 반례를 합성 없이 세우기 위한 창구다.
//   `k<=0` = 해제(이후 seal 값 사용). 프로덕션은 항상 비활성이다.
GGML_API void     ggml_moe_phaseb_test_stamp_override(int32_t k, int32_t n, int32_t qd);

GGML_API void     ggml_moe_phaseb_test_dup_terminal_arm(int32_t on);
GGML_API int      ggml_moe_phaseb_test_dup_terminal_take(void);
GGML_API void     ggml_moe_phaseb_test_dup_terminal_note(int32_t status, uint64_t ctr_delta,
                                                         uint64_t ledger_delta, int32_t qd_delta);
// 반환 = 2차 전이를 실제로 시도한 횟수(0 = 프로브 미발동 → oracle 은 FAIL 로 취급해야 한다).
GGML_API int      ggml_moe_phaseb_test_dup_terminal_result(int32_t * out_status, uint64_t * out_ctr_delta,
                                                           uint64_t * out_ledger_delta,
                                                           int32_t * out_qd_delta);

// 테스트 전용 판독 창구(읽기만).
GGML_API int      ggml_moe_phaseb_test_ledger_record(uint64_t idx, void * out_record, uint32_t bytes);
GGML_API int      ggml_moe_phaseb_test_run_outcome(uint32_t request_epoch);
GGML_API uint64_t ggml_moe_phaseb_test_run_blocked_intervals(uint32_t request_epoch);

// demand R_j 시각 통지(§7.5 predictor 노출 조인 — target = layer+1).
GGML_API void     ggml_moe_phaseb_note_demand_ready(uint32_t graph_seq, int32_t layer, uint64_t r_qpc);

// ★R1 #10 ②③ — 이 (graph, target 층)에 기록된 실제 elig_skip 사유·intent overflow bit 회수.
GGML_API void     ggml_moe_phaseb_target_status(uint32_t graph_seq, int32_t target_mlpos,
                                                uint8_t * out_reason, uint8_t * out_overflow);
// ★R2 #5 — 비동기 사유 상태 큐 pop(dispatcher 가 g_mod.mutex 아래에서 소비). 1=이벤트 있음.
//   graph 가 바뀐 뒤에도 (graph_seq, target) 가 보존되므로 **기생성 행**까지 사후 갱신할 수 있다.
GGML_API int      ggml_moe_phaseb_status_pop(uint32_t * out_graph_seq, int32_t * out_target_mlpos,
                                             uint8_t * out_reason, uint8_t * out_overflow);
// ★R1 #5 ① — 프리페치 OFF = 0(§7.2 게이트 ⓐ "OFF면 B2 확장 블록 전부 0").
GGML_API uint32_t ggml_moe_phaseb_capture_pool_slots(void);
GGML_API uint32_t ggml_moe_phaseb_intent_q_slots(void);

GGML_API void     ggml_moe_phaseb_digest_disposition(uint64_t intent_ordinal, uint16_t rank, uint8_t disp);
GGML_API void     ggml_moe_phaseb_digest_transition(uint8_t ev, uint64_t record_id, int32_t slot,
                                                    uint64_t generation, uint64_t intent_ordinal);
GGML_API void     ggml_moe_phaseb_digest_cacheclass(uint32_t idx, uint8_t cls, uint64_t intent_ordinal);
GGML_API void     ggml_moe_phaseb_digest_readfile(uint16_t layer, uint16_t expert, uint64_t file_offset,
                                                  uint32_t bytes, uint8_t origin);
GGML_API void     ggml_moe_phaseb_digests(uint64_t * out_intent, uint64_t * out_disposition,
                                          uint64_t * out_transition, uint64_t * out_cacheclass,
                                          uint64_t * out_readfile, uint64_t * out_readfile_ops,
                                          uint64_t * out_intent_count);

// ---------------------------------------------------------------------------
// 사이드카 버퍼 상한(§1 MOE_DIRECT_PHASEB_BUF_MB) — 초과 = 기록 중단 + complete=0
// ---------------------------------------------------------------------------
// ★incomplete 사유 마스크(FROZEN_INCOMPLETE_REASON_MASK §3 동결 enum · u32 sticky).
//   `complete == 1  <=>  pb_incomplete_reason_mask == 0` 이 양방향 계약이며, 모든
//   complete->false 전환은 **원인 비트 set 이 선행**한다(§4.2 — atomic OR 가 store 보다 앞).
//   b5~b31 = reserved(전부 0). 신규 사유 추가는 계약 재개봉으로만(§3 M7).
enum ggml_moe_pb_incomplete_reason {
    GGML_MOE_PB_INC_RING_OVERFLOW     = 0x01,  // b0 상태 링(2048) 만석 = 비동기 사유 관측 결손
    GGML_MOE_PB_INC_PENDING_CAP       = 0x02,  // b1 보류함(4096) 초과
    GGML_MOE_PB_INC_CUT_RESIDUAL      = 0x04,  // b2 CUT 6 잔여 자원(세부 = pb_cut_residual)
    GGML_MOE_PB_INC_RECORD_DROP       = 0x08,  // b3 버퍼 상한 drop + graph 행 부분 생성(단일 비트)
    GGML_MOE_PB_INC_SNAPSHOT_MISMATCH = 0x10,  // b4 capture 폭/스냅샷 형상 표류로 축 하강
};

GGML_API bool     ggml_moe_phaseb_buffer_reserve(uint64_t add_bytes);
GGML_API uint64_t ggml_moe_phaseb_buffer_used(void);
GGML_API uint64_t ggml_moe_phaseb_buffer_cap(void);
GGML_API bool     ggml_moe_phaseb_complete(void);
// ★사유 인자 필수 단일 API(§4.2 M3) — 무인자 API·별도 setter 는 존재하지 않는다.
GGML_API void     ggml_moe_phaseb_mark_incomplete(uint32_t reason_bit);
// sticky 마스크 조회(§7 M4 — metrics summary `pb_incomplete_reason_mask` 의 단일 출처).
GGML_API uint32_t ggml_moe_phaseb_incomplete_mask(void);
GGML_API uint64_t ggml_moe_phaseb_dropped_records(void);
GGML_API void     ggml_moe_phaseb_note_dropped(uint64_t n);
GGML_API bool     ggml_moe_phaseb_run_invalid(void);

// 사이드카 파일 1개 기록(헤더는 모듈이 채운다). 반환 = written(1/0).
GGML_API int      ggml_moe_phaseb_write_sidecar(int32_t file_kind, const void * payload,
                                                uint64_t record_count, uint32_t record_bytes,
                                                uint64_t * out_digest);
// a0_equiv 사이드카(모듈 소유 버퍼)를 기록한다. 반환 written(1/0), 레코드 수/digest 출력.
GGML_API int      ggml_moe_phaseb_write_a0equiv(uint64_t * out_records, uint64_t * out_digest);

// phase_b_run JSON 레코드 일괄 방출(요청 순서대로) — direct.cpp가 CUT ⑧에서 호출.
// sidecar 블록에 실을 값은 호출자가 준다(사이드카 소유가 나뉘어 있으므로).
GGML_API void     ggml_moe_phaseb_emit_runs(const char * a0equiv_json, const char * sample_json,
                                            const char * io_json, const char * observe_json,
                                            int32_t complete,
                                            uint64_t dropped, uint64_t buffer_bytes);

// mem echo(§7.2) 항목 조회.
GGML_API void     ggml_moe_phaseb_mem_echo(uint64_t * out_capture_slot_bytes, uint64_t * out_capture_pool_bytes,
                                           uint64_t * out_router_bytes, uint64_t * out_norm_bytes,
                                           uint64_t * out_intent_record_bytes, uint64_t * out_intent_q_bytes,
                                           uint32_t * out_predictor_threads, uint64_t * out_predictor_stack,
                                           uint64_t * out_capture_overflow, uint64_t * out_intent_overflow);

// selftest 전용 — CUT straddler 봉인 회귀용 one-shot barrier(R4).
//   mode 1 = cap_mu 획득 **전** 정지(CUT 선승 arm) · mode 2 = 등록 직후 정지(capture 선승 arm).
// selftest 전용 — CUT 가 cap_mu 를 **시도/취득**하는 시점 rendezvous(R5). arm 뒤에만 유효하며
//   프로덕션 경로 오버헤드는 relaxed 로드 1회다. "CUT 가 경합 위치에서 막혀 있음"을 Sleep 추정이
//   아니라 관측으로 확정하기 위한 창구.
GGML_API int  ggml_moe_phaseb_test_cut_rendezvous_arm(void);
GGML_API int  ggml_moe_phaseb_test_cut_wait_attempted(int32_t timeout_ms);
// ★R6 — CUT 가 cap_mu 를 시도했다가 **실패(경합)** 했음의 양성 증거. 0 = 관측.
GGML_API int  ggml_moe_phaseb_test_cut_wait_contended(int32_t timeout_ms);
GGML_API int  ggml_moe_phaseb_test_cut_wait_acquired(int32_t timeout_ms);
GGML_API int  ggml_moe_phaseb_test_cut_acquired(void);   // 1=취득함 0=아직(막힘)
GGML_API void ggml_moe_phaseb_test_cut_rendezvous_disarm(void);

GGML_API int  ggml_moe_phaseb_test_cap_barrier_arm(int32_t mode);
GGML_API int  ggml_moe_phaseb_test_cap_barrier_wait(int32_t timeout_ms);
GGML_API void ggml_moe_phaseb_test_cap_barrier_release(void);
// ★R7 — barrier 는 release 호출로만 풀린다(자가 해제 없음). 아래는 소유자 귀속·결함 진단용.
GGML_API int  ggml_moe_phaseb_test_cap_barrier_held(void);          // 1 = barrier 안에 정지 중
GGML_API int  ggml_moe_phaseb_test_cap_barrier_safety_fired(void);  // 1 = 60s 안전 타임아웃 발동

// selftest 전용 — 전역 상태 완전 초기화 + env 재해소.
GGML_API void     ggml_moe_phaseb_reset_for_test(void);
// selftest 전용 — 합성 intent 직접 주입(실 dispatcher admission 경로 행사용 test hint source).
GGML_API int      ggml_moe_phaseb_test_push_intent(uint32_t request_epoch, uint32_t graph_epoch,
                                                   uint32_t graph_seq, int32_t source_mlpos,
                                                   int32_t target_mlpos, const int32_t * ids, int32_t n);
// selftest 전용 — 관측 창구.
GGML_API void     ggml_moe_phaseb_test_counts(uint64_t * out_captured, uint64_t * out_predicted,
                                              uint64_t * out_intent_overflow, uint64_t * out_capture_overflow,
                                              int32_t * out_intent_depth, int32_t * out_worker_running);
// selftest 전용 — worker가 큐를 비울 때까지 대기(결정론 확보). timeout_ms 초과 시 0.
GGML_API int      ggml_moe_phaseb_test_wait_idle(uint32_t timeout_ms);
// selftest 전용 — 상태 링 잔여 이벤트 수(observe fanout 차단의 관측 창구 · §6.3 게이트 1).
GGML_API uint32_t ggml_moe_phaseb_test_status_depth(void);

// ---------------------------------------------------------------------------
// E-2a (Codex r2 blocker 2) - selftest only: the predictor drain's ONE budget
// ---------------------------------------------------------------------------
// A stop can only be caught spending a budget twice if a stage actually blocks, and the only
// thing the predictor drain blocks on is a worker that is busy and does not exit. This is a
// one-shot park inside the REAL busy window (pb_worker_process, both mutexes released, the
// same window a long GEMV occupies): the worker is genuinely busy and genuinely unjoinable,
// so the drain's capture-idle wait and its worker join both have something real to spend on.
// The test controls the duration, not the state. Same idiom as the cap barrier above -
// production leaves the arm at 0 and pays one relaxed load per processed capture, the park is
// released only by _release, and a hard 5 s safety bound means a fixture that forgets cannot
// hang a run (it reports through _safety_fired instead).
GGML_API int  ggml_moe_phaseb_test_worker_hold_arm(void);              // 0 = armed
GGML_API int  ggml_moe_phaseb_test_worker_hold_wait(int32_t timeout_ms); // 0 = worker parked
GGML_API void ggml_moe_phaseb_test_worker_hold_release(void);
GGML_API int  ggml_moe_phaseb_test_worker_hold_safety_fired(void);     // 1 = 5s bound fired
// The stages of the last predictor drain, in entry order: 0 = capture-idle wait, 1 = worker
// join. Each row is the deadline that stage measured against and the milliseconds it was
// handed at its own entry. Returns the number of stages recorded (0..2). A drain that never
// entered a stage records nothing, which is how a gate tells "the path ran" from "the path was
// skipped" - the distinction Codex r2 said XSTORE24/25/44 could not make.
GGML_API int  ggml_moe_phaseb_test_drain_stages(int64_t * out_deadline_abs_ms,
                                                int32_t * out_left_ms, int32_t max_n);

#ifdef __cplusplus
}
#endif
