// G3-2c 겹침 — 계측 인프라(Phase 1). 사양: bench/moe-expert-stream/bench/moe-direct/SPEC_G32C.md v4.
//
// ★설계 원칙(사양 서문): 계측을 ggml-moe-direct.cpp(모듈 로직)와 별도 파일로 분리한다 —
// 사양 §7의 sensor_id(baseline 승계 조건)를 "이 파일들의 SHA"로 단순 정의할 수 있게 하기
// 위함이다. 나중에 겹침 구현(Phase 2)이 ggml-moe-direct.cpp에 추가돼도 이 파일들이 불변이면
// baseline(A_pre_ref·delta_pre)을 승계한다.
//
// env MOE_P1_TRACE=1일 때만 활성(배선 2단계에서 MOE_TRACE→MOE_P1_TRACE 개명 — 같은 트리의
// examples/moe-trace/가 MOE_TRACE_OUT·MOE_TRACE_OFF를 쓰고 있어 이름 근접 혼동을 피한다).
// 미설정 시 이 헤더의 모든 API는 즉시 반환하고 실행 경로·바이트가 현행과 동일하다(사양
// §3-5-7). 판정은 std::call_once로 1회 캐시(ggml-moe-direct.cpp의 ggml_moe_direct_enabled()와
// 동일 관례 — 그 함수처럼 프로세스 생애 동안 고정된다).
//
// Windows 전용 — 비Windows 빌드는 ggml-moe-trace.cpp가 전부 no-op/실패 스텁으로 컴파일된다.
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 상수 (C ABI 정수 — ggml-moe-direct.cpp 내부의 C++ enum class Phase와는 별도 표현)
// ---------------------------------------------------------------------------

enum {
    MOE_TRACE_PHASE_DECODE  = 0,
    MOE_TRACE_PHASE_PREFILL = 1,
    MOE_TRACE_PHASE_UNKNOWN = 2,
};

// event_kind — FIXED 레코드의 관측점 식별자(사양 §3-1 전 항목).
enum {
    MOE_TRACE_EV_GRAPH_BEGIN         = 0,
    MOE_TRACE_EV_GRAPH_END           = 1,
    MOE_TRACE_EV_RUN_BEGIN           = 2,
    MOE_TRACE_EV_RUN_END             = 3,
    MOE_TRACE_EV_IO_SUBMIT           = 4,
    MOE_TRACE_EV_IO_COMPLETE         = 5,
    MOE_TRACE_EV_LOCK_WAIT_BEGIN     = 6,
    MOE_TRACE_EV_LOCK_ACQUIRED       = 7,
    MOE_TRACE_EV_DATA_WAIT_BEGIN     = 8,
    MOE_TRACE_EV_DATA_WAIT_END       = 9,
    MOE_TRACE_EV_COMPUTE_ENTER       = 10,
    MOE_TRACE_EV_COMPUTE_EXIT        = 11,
    MOE_TRACE_EV_READY_PUBLISH       = 12,
    MOE_TRACE_EV_FIRST_ARITH_CONSUME = 13,
};

// io_complete의 status(사양 §3-1 O2 "전송 바이트·status 병기"). 이 모듈은 실패 시
// GGML_ABORT로 프로세스를 즉시 종료하므로(SPEC.md v1 §3-1 계약 승계) 정상 경로에서
// 관측되는 값은 사실상 OK뿐이다 — ERROR는 방어적 여지로만 남겨둔다.
enum {
    MOE_TRACE_IO_STATUS_OK    = 0,
    MOE_TRACE_IO_STATUS_ERROR = 1,
};

// 8/16진법 무관 sentinel — layer/tensor_part/expert 필드가 "해당 없음"일 때.
#define MOE_TRACE_NA_U8 ((uint8_t)0xFFu)

// FixedRecord.flags 비트 정의(배선 S7a에서 명시 상수화 — 이전엔 리터럴 0x1만 사용).
#define MOE_TRACE_FLAG_WAS_PENDING        0x01u // io_complete 전용(기존 동작 불변)
#define MOE_TRACE_FLAG_ANCHOR_FIRST_NEG   0x02u // ★S7a: COMPUTE_EXIT 전용 — first anchor 역전
#define MOE_TRACE_FLAG_ANCHOR_LAST_NEG    0x04u // ★S7a: COMPUTE_EXIT 전용 — last anchor 역전

enum {
    MOE_TRACE_RT_FIXED         = 0,
    MOE_TRACE_RT_DEMAND_HDR    = 1,
    MOE_TRACE_RT_DEMAND_CONT   = 2,
    MOE_TRACE_RT_SNAPSHOT_HDR  = 3,
    MOE_TRACE_RT_SNAPSHOT_CONT = 4,
    MOE_TRACE_RT_WALLSTATE_AGG = 5, // 배선 4a 신설 — 사양 §2-2/§6-5 온라인 wall-state 집계
    MOE_TRACE_RT_IOSTATE_AGG   = 6, // 배선 4d 신설 — 사양 §6-5 read_bytes/hit/miss run별 델타
    MOE_TRACE_RT_WALLSTATE_WIN_AGG = 7, // 배선 R9 신설 — §6-2 P6 재정의(표본↔집계 tick 정확
                                         // 일치)용 창 구간 전용 wall-state 집계. 레이아웃은
                                         // ggml_moe_trace_wallstate_agg_record를 그대로 재사용
                                         // (아래 struct 주석 참조 — type 필드 값만 다르다).
    MOE_TRACE_RT_IOSTATE_WIN_AGG   = 8, // 배선 S3a 신설(외부 심의 OV6 E절 차단 1) — U2(demand
                                         // 수 == hard hit+miss) 판정용 창 구간 전용 IO(read_bytes/
                                         // hit/miss) 집계. WALLSTATE_WIN_AGG와 동일 원칙: 레이아웃은
                                         // ggml_moe_trace_iostate_agg_record를 그대로 재사용(아래
                                         // struct 주석 참조 — type 필드 값만 다르다).
};

// demand-set 엔트리 — 공개 API(ggml_moe_trace_demand_digest) 파라미터 타입 겸 아래
// wire format(ggml_moe_trace_demand_cont_record.entries[])의 원소 타입. 하나로 통일해
// 이중 정의를 피한다 — 배열 원소로 쓰이므로 wire format 섹션보다 먼저 와야 한다(완전한
// 타입이 필요).
struct ggml_moe_trace_demand_entry {
    uint8_t layer;
    uint8_t tensor_part;
    uint8_t expert;
};

// ---------------------------------------------------------------------------
// 레코드 wire format(사양 §5-1) — 전부 64B 고정. ggml-moe-trace.cpp와 판독기(향후 배치)·
// selftest가 공유하는 단일 소스(중복 정의 금지). #pragma pack(1)로 컴파일러 패딩 삽입을
// 차단해 필드 나열 순서가 곧 파일상의 바이트 오프셋이 되게 한다 — sensor_id(사양 §7)
// 해시 대상에 이 레이아웃이 포함된다(변경하면 baseline을 승계할 수 없다).
// ---------------------------------------------------------------------------
#pragma pack(push, 1)

// ★코드로 확정(사양 §5-1 "결속 필드"의 저장 표현 — 최종 보고서 표 참조): op_id/
// wait_episode_id/lock_episode_id/compute_op_seq 4종을 각각의 8B 필드로 두지 않고 단일
// correlation_id(8B)로 통합했다. 한 레코드는 항상 정확히 하나의 event_kind이므로(O1 XOR
// O3a XOR O3 XOR O5 ...) 그 시점에 의미 있는 상관자는 항상 하나뿐이다 — event_kind가 이미
// 판별자이므로 값을 이 필드 하나로 겹쳐 써도 모호함이 없다. "여는" 관측점(O1·O3a·O3·O5·
// graph_begin·run_begin)은 correlation_id = 자기 자신의 event_seq를 재사용(별도 ID
// 카운터 불요 — event_seq 자체가 이미 전역 유일·단조). "닫는" 관측점(O2·O3b·O4·O6·
// graph_end·run_end)은 여는 쪽이 반환한 값을 그대로 받아 덮어쓴다.
struct ggml_moe_trace_fixed_record {
    uint8_t  type;                 // MOE_TRACE_RT_FIXED
    uint8_t  event_kind;           // MOE_TRACE_EV_*
    uint8_t  phase;                // MOE_TRACE_PHASE_*
    uint8_t  tensor_part;          // 0..5, 해당없음=MOE_TRACE_NA_U8
    uint8_t  layer;                // 0..35(프로덕션 상수 범위), 해당없음=MOE_TRACE_NA_U8
    uint8_t  expert;                // 0..127, 해당없음=MOE_TRACE_NA_U8
    uint8_t  flags;   // MOE_TRACE_FLAG_* 비트합. bit0=was_pending(io_complete 전용),
                      // bit1/bit2=anchor 역전(COMPUTE_EXIT 전용, 배선 S7a). 그 외 비트는
                      // 예약(항상 0). ★판독기의 미정의 비트 fail-close 검사는 **S7b에서
                      // 배선**한다 — 그 전까지 이 줄은 발행측 계약일 뿐 검증되지 않는다
                      // (메인 검수 26-07-21: 주석이 미구현 동작을 기정사실로 쓰지 않도록).
    uint8_t  status;                  // MOE_TRACE_IO_STATUS_*
    int64_t  qpc_tick;                 // QueryPerformanceCounter 원값(틱 정수, 환산 금지)
    uint64_t event_seq;                 // 전역 단조 증가 — canonical 순서(사양 §3-5-2)
    uint64_t correlation_id;             // op_id/lock_episode_id/wait_episode_id/compute_op_seq(위 주석)
    uint32_t thread_id;                   // GetCurrentThreadId()
    uint32_t generation_low32;             // 슬롯 generation 하위 32비트(사양 §3-3)
    uint32_t bytes_requested;               // io_submit에서만 의미(그 외 0) — 예외(event_kind별):
                                             //   RUN_END     = prompt_n(서버 권위값, 배선 2단계,
                                             //                 사양 §4-3 대조용 — run_end 함수 주석)
                                             //   GRAPH_BEGIN = 그 graph의 ubatch.n_tokens(배선 R1,
                                             //                 RUN_END와 동일 관례 — graph_begin 함수 주석)
                                             //   GRAPH_END   = chunk_call_count 세션 누적 스냅샷의
                                             //                 하위 32비트(배선 R7a — 그 graph의 마지막
                                             //                 chunk 호출 이후 채취, §6-2 P2 등식 대조용
                                             //                 — graph_end 함수 주석 참조)
    uint32_t bytes_transferred;              // io_complete에서만 의미(그 외 0) — 예외(event_kind별):
                                              //   RUN_END      = predicted_n(서버 권위값, 위와 동일)
                                              //   COMPUTE_EXIT = 이 워커가 이 op에서 실제로 수행한
                                              //                  작업 단위 수(배선 R2 — mul_mat_id =
                                              //                  chunk 호출 수, add_id = 처리한 행 수)
                                              //   GRAPH_BEGIN  = chunk_call_count 세션 누적 스냅샷의
                                              //                  하위 32비트(배선 R7a — 그 graph의 첫
                                              //                  chunk 호출보다 앞서 채취, §6-2 P2 등식
                                              //                  대조용 — graph_begin 함수 주석 참조)
    uint32_t graph_seq;                       // ★프로세스 전역 단조 증가(배선 2단계 확정 — run
                                               // 경계에서 리셋하지 않는다. 사양 §4-1 "코드로 확정")
    uint16_t run_index;                        // 세션 내 0-based run 인덱스(프로세스 전역 단조)
    int16_t  qd_after;                          // 이 이벤트 처리 직후 app_qd 스냅샷(포화 변환)
    int16_t  compute_active_after;               // 이 이벤트 처리 직후 useful_compute_active 스냅샷
    uint8_t  _pad[6];                             // ★배선 S6(외부 심의 OV6 B-1 잔여 — O5/O6
                                                   // duration 정직성의 "실제 작업 시점 결속" 요구):
                                                   // event_kind==MOE_TRACE_EV_COMPUTE_EXIT 레코드에서만
                                                   // 의미를 가진다(그 외 event_kind는 여전히 값 0
                                                   // 고정). [0..2] = first_work_delta, [3..5] =
                                                   // last_work_delta — 각각 3바이트 부호없는 리틀
                                                   // 엔디언 정수: first_work_delta = (첫 작업 단위
                                                   // 시작 tick) − (O5의 qpc_tick), last_work_delta =
                                                   // (O6의 qpc_tick) − (마지막 작업 단위 종료 tick).
                                                   // 3바이트 상한(0xFFFFFF = 16,777,215 tick)을
                                                   // 넘으면 포화(0xFFFFFF 그대로) — 포화 자체가
                                                   // O5를 일찍 열었거나 O6를 늦게 닫은 우회 의심
                                                   // 신호다(판독기 소비는 후속 배치). work_unit_count
                                                   // ==0(정상 호출점은 항상 >=1 — 방어적 케이스)이면
                                                   // 두 값 모두 0(별도 sentinel 없음 — 판독기는 이
                                                   // 경우를 "정상 소구간"과 구분할 수 없다). 계산·
                                                   // 인코딩은 ggml-moe-trace.cpp 안에서만 한다.
    // ★배선 S7a(v3): 이 필드의 해석은 layout_version==3에서만 유효하다. 또한 raw delta가
    // 음수(역전)인 경우 이전에는 0으로 클램프되어 정상 0-delta와 구분되지 않았으나, 이제
    // 해당 side의 flags 비트(MOE_TRACE_FLAG_ANCHOR_*_NEG)가 서고 header의
    // anchor_negative_count가 증가한다. wire delta 값 자체는 계속 0을 싣는다(레코드 크기 유지).
    // ★배선 S12(v4 slot identity — SPEC_G32C §8 U6'): event_kind==MOE_TRACE_EV_IO_SUBMIT(O1)/
    // MOE_TRACE_EV_IO_COMPLETE(O2) 레코드에서만 _pad[0..3] = slot_index(리틀엔디언 uint32,
    // victim 슬롯 인덱스), _pad[4..5] = 0(reserved). layout_version>=4에서만 유효하다(v3 이하
    // O1/O2는 _pad 전량 0). COMPUTE_EXIT의 anchor delta와는 event_kind가 배타적이라(레코드당
    // event_kind 하나) 같은 _pad를 겹쳐 써도 모호함이 없다 — S6이 O6 anchor delta를 _pad에
    // 실은 것과 동일 수법. 인코딩은 ggml-moe-trace.cpp 안에서만 한다.
};

struct ggml_moe_trace_demand_hdr_record {
    uint8_t  type;         // MOE_TRACE_RT_DEMAND_HDR
    uint8_t  phase;
    uint16_t run_index;
    uint32_t graph_seq;
    uint64_t qpc_tick;
    uint64_t event_seq;    // continuation들이 header_event_seq로 참조
    uint32_t n_entries;    // 전체 continuation 합산 엔트리 수
    uint32_t n_cont;       // 뒤따르는 DEMAND_CONT 레코드 수
    uint8_t  _pad[32];
};

#define MOE_TRACE_DEMAND_CONT_OVERHEAD (1u + 2u + 1u + 8u) /* type+cont_index+entry_count+header_event_seq */
#define MOE_TRACE_DEMAND_K ((64u - MOE_TRACE_DEMAND_CONT_OVERHEAD) / 3u) /* = 17 */

struct ggml_moe_trace_demand_cont_record {
    uint8_t  type;              // MOE_TRACE_RT_DEMAND_CONT
    uint16_t cont_index;        // 0-based
    uint8_t  entry_count;       // 이 레코드에 실제로 채운 엔트리 수(<=MOE_TRACE_DEMAND_K)
    uint64_t header_event_seq;  // 결속 — ggml_moe_trace_demand_hdr_record.event_seq
    struct ggml_moe_trace_demand_entry entries[MOE_TRACE_DEMAND_K]; // = 17
    uint8_t  _pad[64 - MOE_TRACE_DEMAND_CONT_OVERHEAD - MOE_TRACE_DEMAND_K * 3];
};

struct ggml_moe_trace_snapshot_hdr_record {
    uint8_t  type;        // MOE_TRACE_RT_SNAPSHOT_HDR
    uint8_t  _reserved0;
    uint16_t run_index;
    uint32_t graph_seq;
    uint64_t qpc_tick;
    uint64_t event_seq;
    uint32_t n_ops;        // outstanding op_id 목록 길이
    uint32_t n_waits;       // 활성 wait_episode_id 목록 길이
    uint32_t n_cont;         // 뒤따르는 SNAPSHOT_CONT 총수(두 리스트 합산)
    uint8_t  _pad[28];
};

#define MOE_TRACE_SNAPSHOT_CONT_OVERHEAD (1u + 1u + 2u + 1u + 8u) /* type+list_kind+cont_index+entry_count+header_event_seq */
#define MOE_TRACE_SNAPSHOT_K ((64u - MOE_TRACE_SNAPSHOT_CONT_OVERHEAD) / 8u) /* = 6 */

struct ggml_moe_trace_snapshot_cont_record {
    uint8_t  type;              // MOE_TRACE_RT_SNAPSHOT_CONT
    uint8_t  list_kind;          // 0=outstanding op_id, 1=active wait_episode_id
    uint16_t cont_index;
    uint8_t  entry_count;         // <= MOE_TRACE_SNAPSHOT_K
    uint64_t header_event_seq;
    uint64_t values[MOE_TRACE_SNAPSHOT_K]; // = 6
    uint8_t  _pad[64 - MOE_TRACE_SNAPSHOT_CONT_OVERHEAD - MOE_TRACE_SNAPSHOT_K * 8];
};

// 배선 4a 신설(사양 §2-2 wall-state 분해·§3-2 Lcrit 두 성분·§6-5 A_pre_ref 집계) — run_end에서
// phase(decode/prefill)당 1개씩 발행하는 온라인 집계 결과. FIXED와 달리 "여는/닫는" 짝이 없는
// 단발 요약 레코드라 tagged-union의 별도 종류로 둔다(§5-1). 필드는 이 run·이 phase 동안의
// *기여분*(run_begin 시점 세션 누적 스냅샷과의 차분)이다 — 세션 전체 누적값은
// ggml_moe_trace_wall_state_totals()로 별도 조회한다(ggml-moe-direct.cpp의 metrics JSONL이
// 그쪽을 쓴다 — 판독기가 이 레코드의 run별 합과 metrics의 세션 누계를 대조할 수 있게).
struct ggml_moe_trace_wallstate_agg_record {
    uint8_t  type;             // MOE_TRACE_RT_WALLSTATE_AGG
    uint8_t  phase;             // MOE_TRACE_PHASE_DECODE 또는 MOE_TRACE_PHASE_PREFILL만(UNKNOWN 없음)
    uint16_t run_index;
    uint64_t qpc_tick;           // 이 AGG 레코드를 발행한 시점(run_end 처리 중)의 QPC
    uint64_t event_seq;           // 전역 단조 — canonical 순서(§3-5-2)
    int64_t  c_ticks;              // |C| 기여분(tick) — 사양 §2-2
    int64_t  lcrit_ticks;           // |Lcrit| 기여분
    int64_t  other_ticks;            // |other| 기여분(C+Lcrit+other == 이 run의 이 phase 총 graph 구간 길이)
    int64_t  l_lock_ticks;            // L_lock 기여분(사양 §3-2 — 락 대기가 "존재한" 시간)
    int64_t  l_data_ticks;             // L_data 기여분(데이터 대기가 "존재한" 시간)
    uint32_t wall_state_mismatch_count; // 이 run·이 phase 동안 §6-2 P4 항등식 불일치로 잡힌 graph 수
}; // 크기 static_assert는 다른 레코드들과 동일 관례로 ggml-moe-trace.cpp에 둔다(이 헤더는
   // C 소비자와도 호환돼야 하므로 static_assert를 여기 두지 않는다 — 기존 레코드들 참조).

// 배선 R9 신설(§6-2 P6 재정의) — 표본 창 gating(배선 R1) 도입 이후 P6("표본↔집계: tick 정확
// 일치")이 판정 불가가 됐다: 온라인 누적(위 WallStateAggRecord)은 run 전체를 담지만 표본
// (O5/O6 등 고빈도 이벤트)은 표본 창 안에만 존재해, 판독기가 "같은 범위"의 온라인 집계값을
// 가질 수 없었다. MOE_TRACE_RT_WALLSTATE_WIN_AGG는 그 수리 — 창 구간(g_window_mode != 0)에만
// 귀속된 별도 누산기(g_wall_accum_window)의 run별 기여분을 담는다. ★레이아웃은 위
// ggml_moe_trace_wallstate_agg_record를 그대로 재사용한다(새 struct를 정의하지 않음) — 판독기가
// 같은 파서로 두 레코드 타입을 모두 처리할 수 있게 하기 위함이며, type 필드 값(5 vs 7)만으로
// 세션-scope와 창-scope를 구분한다.

// 배선 4d 신설(사양 §6-5 "read_bytes delta_pre") — run_end에서 phase(decode/prefill)당
// 1개씩 발행하는 read_bytes/hit/miss의 run별 기여분(baseline diff — WALLSTATE_AGG와 동일
// 패턴). 원본 카운터는 ggml-moe-direct.cpp의 PhaseCounters에 있으므로, trace 모듈이
// ggml_moe_direct_phase_read_bytes_hit_miss()로 조회한 세션 누적값을 run_begin에서
// baseline 스냅샷하고 run_end에서 차분해 여기 싣는다. 값은 항상 비음수이지만(모듈 hard
// counter는 단조 증가) WallStateAggRecord와 동일하게 int64_t로 둔다 — 뺄셈 결과가 음수가
// 되는 것은 그 자체로 버그 신호이므로, unsigned wraparound로 침묵시키지 않고 판독기가
// 음수를 그대로 관측해 적발할 수 있게 하기 위함이다.
// ★R3 배선(외부 심의 OV5 차단 3 — 사양 §3): 이 레코드가 싣는 값의 출처가 바뀌었다. 이전에는
// n_tokens_hint 기반(부정확) 카운터를 조회했으나, 이제는 graph 문맥(ggml_moe_trace_current_
// graph_context) 기반 정확 카운터(ggml_moe_direct_phase_read_bytes_hit_miss_exact)를 조회한다.
// wire format(이 struct)과 필드 의미는 변경되지 않았다 — 판독기는 이 레코드를 항상 "graph
// 문맥 기반 정확 분류"로 해석하면 된다.
struct ggml_moe_trace_iostate_agg_record {
    uint8_t  type;             // MOE_TRACE_RT_IOSTATE_AGG
    uint8_t  phase;             // MOE_TRACE_PHASE_DECODE 또는 MOE_TRACE_PHASE_PREFILL만
    uint16_t run_index;
    uint64_t qpc_tick;           // 이 레코드를 발행한 시점(run_end 처리 중)의 QPC
    uint64_t event_seq;           // 전역 단조 — canonical 순서(§3-5-2)
    int64_t  read_bytes_delta;     // |read_bytes| 기여분(모듈 hard counter, §3-5-6 요구)
    int64_t  hit_delta;             // |hit| 기여분
    int64_t  miss_delta;             // |miss| 기여분
    uint8_t  _pad[20];
};

// 배선 S3a 신설(외부 심의 OV6 E절 차단 1 — U2 "demand 수 == hard hit+miss" 판정 복구) — 표본
// 창 gating(배선 R1) 도입 이후 U2가 판정 불가가 됐다: DEMAND는 고빈도군이라 표본 창 안에만
// 존재하는데, 비교 대상이던 IOSTATE_AGG(세션-scope 누적)는 run 전체를 담아 scope가 어긋난다.
// MOE_TRACE_RT_IOSTATE_WIN_AGG는 그 수리 — 창 구간(g_window_mode != 0)에만 귀속된 별도 IO
// 누산기(ggml-moe-direct.cpp의 phase_exact_window[])의 run별 기여분을 담는다. ★레이아웃은 위
// ggml_moe_trace_iostate_agg_record를 그대로 재사용한다(새 struct를 정의하지 않음) — 판독기가
// 같은 파서로 두 레코드 타입을 모두 처리할 수 있게 하기 위함이며, type 필드 값(6 vs 8)만으로
// 세션-scope와 창-scope를 구분한다(WALLSTATE_WIN_AGG가 5 vs 7로 쓰는 것과 동일 관례).

// 트레이스 파일 header(사양 §3-5-6, layout_version 2 — 배선 R1 §6). 이 header 자체도 64B로
// 맞춰 "선두 64B=header, 이후 64B 단위 레코드 배열"이라는 단일 불변식만 기억하면 되게 한다.
// v1 대비 마지막 _pad[16]을 소진해 prefill 세그먼트·덮어쓰기 계수용 4필드를 신설했다(R1
// 사양 §6 — 판독기는 layout_version 1/2를 모두 수용해야 한다. ★S7a 정정: 이제 1/2/3을
// 수용하되 **anchor 판정 대상은 3뿐**이다 — 아래 v3 문단 참조).
// v3(배선 S7a): _reserved0을 anchor_negative_count로 소진하고, FixedRecord.flags에 anchor
// 음수 표시 bit 2개를 정의했다. **layout_version 3 = "O6의 _pad에 anchor delta가 실려 있고
// 음수 역전이 검출된다"는 capability 선언**이다 — v2 이하 파일은 anchor 판정 대상이 아니다
// (구버전 판독기는 미지원 버전으로 즉시 거부하므로 오인 경로가 없다).
// v4(배선 S12 slot identity — SPEC_G32C §8 U6'): header 레이아웃·필드 의미는 v3와 완전히
// 동일하고(anchor_negative_count·tail 4필드 그대로 상속), **layout_version 값만 4**다. 유일한
// wire 차이는 O1/O2 FixedRecord의 _pad에 slot_index가 실린다는 것(위 FixedRecord._pad 주석).
// 판독기는 v4를 v3의 전 capability(anchor 해석 포함)를 상속한 상위 집합으로 처리한다.
struct ggml_moe_trace_file_header {
    char     magic[8];              // "MOETRC1" + NUL
    uint32_t layout_version;        // = 4 (배선 S12 v4 — 위 v4 문단·SPEC_G32C §8 U6')
    uint32_t record_size;           // = 64 — 판독기 자기검증용
    uint64_t qpc_frequency;         // QueryPerformanceFrequency 원값
    uint32_t endianness_marker;     // 0x01020304 리틀엔디언 자기검증용
    uint32_t anchor_negative_count; // ★배선 S7a(v3 신설) — anchor delta raw가 음수였던
                                    // COMPUTE_EXIT 레코드 수. **세션 누적**(overwrite_count·
                                    // prefill_overflow_count와 동일 관례 — flush마다 리셋하지
                                    // 않는다). O6 하나당 1회만 증가한다(first/last가 둘 다
                                    // 음수여도 1). uint32 포화 변환(saturate_u64_to_u32).
                                    // ★판독기 소비 계약: 파일별 **합산 금지**(누적이라 중복
                                    // 계산이 된다) — 값>0이면 그 자체로 실패다.
                                    // v1/v2에서는 이 자리가 0(미사용)이었다.
    uint64_t record_count;          // ★ring 세그먼트 + prefill 세그먼트 합계(기존 소비자 호환)
    uint64_t total_write_pos;       // 래핑 포함 누적 write_pos(참고용)
    uint32_t prefill_seg_record_count; // 파일 끝쪽 prefill 세그먼트 레코드 수
    uint32_t overwrite_count;          // 링 덮어쓰기 횟수(포화 변환 — uint32 상한에서 클램프)
    uint32_t prefill_overflow_count;   // prefill 버퍼 포화 횟수(동일 변환)
    uint16_t decode_window_begin;      // = DECODE_WIN_BEGIN(128)
    uint16_t decode_window_end;        // = DECODE_WIN_END(192)
};

#pragma pack(pop)

// ---------------------------------------------------------------------------
// 마스터 스위치 + 게이지 조회 (락 없는 원자적 조회)
// ---------------------------------------------------------------------------

GGML_API bool ggml_moe_trace_enabled(void);

GGML_API int32_t ggml_moe_trace_gauge_compute_active(void); // useful_compute_active 현재값
GGML_API int32_t ggml_moe_trace_gauge_app_qd(void);         // app_qd 현재값
GGML_API int32_t ggml_moe_trace_gauge_wait_active(void);    // wait_active 현재값(배선 4a 신설 — 사양 §2-1)

// 배선 S3a 신설(외부 심의 OV6 E절 차단 1) — 창 활성 여부 조회(off이거나 창 밖이면 false).
// g_window_mode는 off 상태에서는 결코 갱신되지 않으므로(graph_begin/end가 trace_active_and_
// ready() 게이트를 먼저 통과해야 함) 이 함수 자체는 내부에서 별도로 ggml_moe_trace_enabled()를
// 재확인하지 않는다(다른 게이지 조회 함수들과 동일 관례). ggml-moe-direct.cpp의 호출자
// (count_hit_locked/count_miss_locked — ★QD 레버가 구 ensure_locked의 계수 분기를 분리 승계)가
// 이미 ggml_moe_trace_enabled() 게이트 안에서만 이 함수를 호출한다(strict flag-off 유지 —
// 배선 R2 §4-d와 동일 원칙).
GGML_API bool ggml_moe_trace_window_active(void);

// ---------------------------------------------------------------------------
// 초기화 · 플러시 · 종료
// ---------------------------------------------------------------------------

// 링버퍼(256 MiB) + prefill capture 버퍼(32 MiB×2, 배선 R1 §2) 선할당 + QPC 주파수 캐치 +
// header 확정. MOE_P1_TRACE=1일 때만 실제 동작.
// 이후 API 호출들도 최초 1회 호출 시 이 초기화를 지연 수행하므로(내부 call_once) 명시
// 호출은 필수가 아니다 — selftest가 결정적 타이밍으로 확인하고 싶을 때 쓰는 편의 진입점.
// 반환: 성공 또는 off(무동작)=0, 실패(<0, 예: VirtualAlloc 실패).
GGML_API int ggml_moe_trace_init(void);

// flush: 명시 호출로만(사양 §3-5-5 — 측정 중 파일 I/O 금지, run_end 이후·다음 run_begin
// 전에만 호출할 것). 현재까지 기록된 링버퍼 내용을 파일 header(QPC 주파수·layout
// version·각 레코드 sizeof·엔디언) + 기록 순서 그대로 이진 파일로 쓴다. 쓰기 실패는
// GGML_ABORT(계측 파일 쓰기 실패를 침묵 무시하지 않는다 — ggml-moe-direct.cpp의
// metrics_write_line 관례와 동일). off 상태면 무동작.
GGML_API void ggml_moe_trace_flush(const char * path);

// 종료 finalize — 멱등. 슬롯/링버퍼는 프로세스 종료 시 OS가 회수(seal()과 동일 철학).
GGML_API void ggml_moe_trace_finalize(void);

// ---------------------------------------------------------------------------
// 관측점 API (사양 §3-1) — 각 함수는 off일 때 얕은 게이트 하나만 거치고 즉시 반환한다.
// 짝을 이루는 관측점은 "여는 쪽"이 상관 ID(해당 이벤트 자신의 event_seq를 재사용 — 아래
// "코드로 확정" 사항 참조)를 반환하고, "닫는 쪽"이 그 값을 인자로 받는다.
//
// ★반환값 0의 의미: off 상태(또는 준비 미완료로 인한 drop) — 실제 event_seq는 항상 1부터
// 시작하므로(내부 카운터가 1에서 출발) 0은 어느 관측점에서도 진짜 ID와 절대 충돌하지
// 않는 sentinel이다. 호출측이 "닫는" 함수에 0을 그대로 넘기면 그 함수도 동일한 off
// 게이트를 통과하며 조용히 무시한다(별도 널 체크 불요).
// ---------------------------------------------------------------------------

// ★배선 2단계: graph_begin은 호출과 동시에 {run_index, graph_seq, phase}를 아래
// ggml_moe_trace_current_graph_context()가 읽는 전역 스냅샷에 발행(release-store)한다 —
// ggml CPU worker 스레드(io/wait/lock 관측점이 실행되는 곳)가 이 값을 얻는 유일한 경로다
// (무비판적 TLS는 worker에서 값을 잃으므로 배제 — 사양 §4-1 "코드로 확정", 최종 보고서
// 참조). graph_end는 phase를 MOE_TRACE_PHASE_UNKNOWN으로 되돌려 "활성 graph 없음"을
// 표시한다(방어적 — 관측점은 구조상 graph 구간 밖에서 fire할 수 없지만, 만약 fire한다면
// phase 필드에 그 이상이 드러난다).
// ★배선 R1: n_tokens는 이 graph의 ubatch.n_tokens 원값(FixedRecord.bytes_requested에 실림 —
// 위 wire format 섹션 필드 주석 참조). ★기본값 없음 — 이 헤더는 순수 C 소비자(ggml-cpu.c)도
// #include하므로 C++ 전용 기능인 default argument를 쓸 수 없다(C에서는 구문 오류). 모든
// 호출부가 명시적으로 넘겨야 한다.
GGML_API uint64_t ggml_moe_trace_graph_begin(uint16_t run_index, uint32_t graph_seq, uint8_t phase, uint32_t n_tokens);
GGML_API void     ggml_moe_trace_graph_end  (uint64_t graph_op_id, uint16_t run_index, uint32_t graph_seq, uint8_t phase);

// ★배선 2단계: predicted_n/prompt_n은 서버 권위값(server_slot::release()가 timing 갱신
// 후·reset 전에 읽은 n_decoded/n_prompt_tokens_processed)이며 FixedRecord의 bytes_requested/
// bytes_transferred 필드에 실린다(사양 §4-3 "decode graph == predicted_n-1" 등 대조용 — 위
// wire format 섹션의 필드 주석 참조).
GGML_API uint64_t ggml_moe_trace_run_begin(uint16_t run_index);
GGML_API void     ggml_moe_trace_run_end  (uint64_t run_op_id, uint16_t run_index, uint32_t predicted_n, uint32_t prompt_n);

// ---------------------------------------------------------------------------
// run_index/graph_seq 카운터 + graph 문맥 조회(배선 2단계 신설 — 사양 §4-1/§4-2 "코드로
// 확정"). 이 세 함수는 ggml-moe-trace.cpp가 소유한 전역 상태를 노출한다: run_index는
// server-context.cpp(run_begin 호출부)가 발급하고 llama-context.cpp(graph_begin 호출부)가
// 읽는다 — 두 TU가 서로 다른 링크 단위(서버 exe ↔ libllama)이므로 ggml(둘의 공통 하위
// 의존)에 상태를 둬야 공유가 가능하다(직접 파라미터 전달 경로 없음 — llama_decode()가
// run_index를 모른다).
// ---------------------------------------------------------------------------

// run_begin 직전에 호출 — 세션 내 0-based run_index를 새로 발급한다(off면 0·카운터 불변).
GGML_API uint16_t ggml_moe_trace_next_run_index(void);

// 마지막으로 발급된 run_index를 증가 없이 조회한다(graph_begin 호출부가 사용 — 그 시점엔
// 이미 서버가 run_begin에서 발급을 마친 뒤이므로 항상 "현재 run"을 가리킨다).
GGML_API uint16_t ggml_moe_trace_current_run_index(void);

// graph_begin~graph_end 구간 동안 불변인 문맥을 조회한다(ggml CPU worker 스레드에서 호출 —
// 사양 §4-1 worker 문맥 전파). 게이지 조회 함수들과 동일하게 게이트 없이 항상 즉시 반환한다
// (off 상태면 graph_begin이 한 번도 스냅샷을 갱신하지 않았으므로 초기값 {0,0,UNKNOWN}이
// 그대로 읽힌다 — 호출측이 이미 ggml_moe_trace_enabled()로 게이트한 뒤에만 부르므로 무해).
GGML_API void ggml_moe_trace_current_graph_context(
    uint16_t * out_run_index, uint32_t * out_graph_seq, uint8_t * out_phase);

// io_submit/io_complete(O1/O2). 반환값(op_id) = io_complete에 그대로 넘길 상관자.
// ★배선 S12(v4 slot identity — SPEC_G32C §8 U6'): slot_index(victim 슬롯 인덱스)를 O1/O2 양쪽에
// 동일 값으로 탑재한다(O2도 받아 O1↔O2 slot 일치 검사를 가능케 한다). wire 표현은 각 레코드의
// _pad[0..3](리틀엔디언 uint32 — 위 FixedRecord._pad 주석). off 상태에서는 무시된다.
GGML_API uint64_t ggml_moe_trace_io_submit(
    uint16_t run_index, uint32_t graph_seq, uint8_t phase,
    uint8_t layer, uint8_t tensor_part, uint8_t expert,
    uint32_t generation_low32, uint32_t bytes_requested, uint32_t slot_index);

GGML_API void ggml_moe_trace_io_complete(
    uint64_t op_id, uint8_t status, uint32_t bytes_transferred, bool was_pending, uint32_t slot_index);

// lock_wait_begin/lock_acquired(O3a/O3b). 반환값 = lock_episode_id.
GGML_API uint64_t ggml_moe_trace_lock_wait_begin(uint16_t run_index, uint32_t graph_seq, uint8_t phase);
GGML_API void     ggml_moe_trace_lock_acquired  (uint64_t lock_episode_id);

// data_wait_begin/data_wait_end(O3/O4) — record 단위 발급(batch acquire 1회가 여러
// record를 대기할 수 있으므로 — 사양 §3-2). hit 경로에서는 절대 호출하지 않는다.
GGML_API uint64_t ggml_moe_trace_data_wait_begin(
    uint16_t run_index, uint32_t graph_seq, uint8_t phase,
    uint8_t layer, uint8_t tensor_part, uint8_t expert, uint32_t generation_low32);
GGML_API void ggml_moe_trace_data_wait_end(uint64_t wait_episode_id);

// ready_publish(O7) — 슬롯 READY release-store 직후. 단발 이벤트(짝 없음). qd_after/
// compute_active_after는 다른 관측점과 동일하게 이 함수 내부에서 게이지를 직접 읽어
// 채운다(호출측이 별도로 넘길 필요 없음 — 호출측은 게이지를 직접 들고 있지 않다).
GGML_API void ggml_moe_trace_ready_publish(
    uint16_t run_index, uint32_t graph_seq, uint8_t phase,
    uint8_t layer, uint8_t tensor_part, uint8_t expert, uint32_t generation_low32);

// compute_enter/compute_exit(O5/O6) — op×worker 단위(사양 §2-1, chunk 단위 아님).
// expert 필드는 이 관측점에서 의미가 없으므로(op 전체에 걸쳐 여러 expert를 순회) 내부에서
// MOE_TRACE_NA_U8로 고정 기록한다. nth는 상한 검사(useful_compute_active<=nth)에 쓰인다.
// 반환값 = compute_op_seq. ★배선 R2(사양 §3): work_unit_count는 이 워커가 이 op에서 실제로
// 수행한 작업 단위 수(위 wire format 섹션 bytes_transferred 필드 주석 참조) — close 레코드의
// bytes_transferred에 그대로 실린다.
GGML_API uint64_t ggml_moe_trace_compute_enter(
    uint16_t run_index, uint32_t graph_seq, uint8_t phase,
    uint8_t layer, uint8_t tensor_part, int32_t nth);

// ★배선 S6(사양 §1/§2/§3 — 외부 심의 OV6 B-1 잔여): first_work_tick/last_work_tick은
// 호출측(ggml-cpu.c/ops.cpp)이 아래 ggml_moe_trace_query_tick()으로 채취한 원시 QPC 값을
// 그대로 넘긴다(첫 작업 단위 시작 직전 / 마지막 작업 단위 종료 직후) — 델타 계산(O5 자신의
// qpc_tick과의 차·포화 변환)은 이 함수 내부에서만 한다(호출점이 여러 곳으로 흩어지면
// 어긋나므로). 결과는 close 레코드의 _pad[6]에 실린다(위 wire format 섹션 FixedRecord._pad
// 주석 참조). work_unit_count==0이면 두 값 모두 0(위 _pad 주석과 동일 사유).
GGML_API void ggml_moe_trace_compute_exit(
    uint64_t compute_op_seq, uint32_t work_unit_count,
    int64_t first_work_tick, int64_t last_work_tick);

// ★배선 S6 신설 — 호출점(ggml-cpu.c/ops.cpp)이 QueryPerformanceCounter를 직접 부르지 않도록
// 이 모듈이 소유한 조회 함수. off 상태(또는 미준비)면 0(호출측은 이 값을 다시 위
// ggml_moe_trace_compute_exit()에 원시 tick으로만 전달하면 되고, off 상태의 0은 그쪽에서
// work_unit_count 게이트에 의해 어차피 무시된다). 다른 게이지 조회 함수들과 달리 이 함수는
// 선형화 스핀락(§5) 밖에서 호출된다 — event_seq 발급과 결속될 필요가 없는 순수 anchor
// 채취이기 때문이다(그 결속은 compute_enter/compute_exit 자신의 tick이 담당).
GGML_API int64_t ggml_moe_trace_query_tick(void);

// first_arith_consume(O8) — 그 expert의 첫 실제 산술 진입. 단발 이벤트(짝 없음).
GGML_API void ggml_moe_trace_first_arith_consume(
    uint16_t run_index, uint32_t graph_seq, uint8_t phase,
    uint8_t layer, uint8_t tensor_part, uint8_t expert, uint32_t generation_low32);

// 배선 3단계 신설 — O8 "처음" 판정(SPEC_G32C.md v4 §3-1 O8 "그 expert의 첫 실제 산술 진입" +
// §8 U5 "정확히 1회"). (layer, tensor_part, expert)가 이번 graph_seq에서 처음 호출되면 true를
// 반환하며 원자적으로 "이미 봄" 상태로 표시한다. 내부 표는 (layer, tensor_part, expert)마다
// "마지막으로 이 조합을 claim한 graph_seq"를 저장하는 방식이라 그래프마다 별도 리셋이
// 불필요하다(graph_seq는 프로세스 전역 단조 증가 — 다른 그래프의 값과 절대 같을 수 없으므로
// 이전 값이 남아 있어도 새 graph_seq와 비교하면 자연히 "처음"으로 판정된다). 동시 호출 경합
// 시(같은 op의 여러 chunk를 서로 다른 워커가 동시에 처리하는 경우) 하드웨어가 직렬화하는
// atomic exchange 순서상 정확히 하나만 true를 받는다(구현 주석 참조) — 별도 CAS 재시도 루프
// 불필요. 호출측은 true일 때만 ggml_moe_trace_first_arith_consume()을 호출할 것. off 상태면
// 항상 false(부작용 없음).
GGML_API bool ggml_moe_trace_first_arith_claim(
    uint32_t graph_seq, uint8_t layer, uint8_t tensor_part, uint8_t expert);

// demand-set digest(사양 §3-4) — (layer, tensor_part, expert) 목록을 DEMAND
// header+continuation으로 분할 기록한다. entries는 호출 전 정렬 불요(판독기가 정렬).
// (struct ggml_moe_trace_demand_entry는 위 wire format 섹션에서 이미 정의됨.)
GGML_API void ggml_moe_trace_demand_digest(
    uint16_t run_index, uint32_t graph_seq, uint8_t phase,
    const struct ggml_moe_trace_demand_entry * entries, uint32_t n_entries);

// window-start snapshot(사양 §5-1) — outstanding op_id 목록 + 활성 wait_episode_id 목록을
// SNAPSHOT header+continuation으로 분할 기록한다.
GGML_API void ggml_moe_trace_snapshot(
    uint16_t run_index, uint32_t graph_seq,
    const uint64_t * outstanding_op_ids, uint32_t n_ops,
    const uint64_t * active_wait_episode_ids, uint32_t n_waits);

// ---------------------------------------------------------------------------
// 온라인 wall-state 집계 조회(배선 4a 신설 — 사양 §2-2/§3-2/§6-5). 세션 누적(cumulative)
// 값을 반환한다 — phase는 MOE_TRACE_PHASE_DECODE/PREFILL만 유효(그 외 인자는 전부 0으로
// 채워 반환·부작용 없음). ggml-moe-direct.cpp의 metrics JSONL이 이 값을 phase별 누계로
// 싣는다(run별 기여분은 트레이스 스트림의 MOE_TRACE_RT_WALLSTATE_AGG 레코드가 담당 — 판독기가
// 두 경로를 대조할 수 있게 하기 위한 의도적 이중화, 위 wire format 섹션 주석 참조). off
// 상태거나 out 포인터가 널이면 그 포인터만 건너뛴다(부분 조회 허용 — 호출측이 관심 필드만
// 넘겨도 안전).
GGML_API void ggml_moe_trace_wall_state_totals(
    uint8_t phase,
    int64_t * out_c_ticks, int64_t * out_lcrit_ticks, int64_t * out_other_ticks,
    int64_t * out_l_lock_ticks, int64_t * out_l_data_ticks, uint64_t * out_mismatch_count);

// ---------------------------------------------------------------------------
// acquire() 호출 통계(사양 §3-5-6 metrics 출력 요구) — ★배선 R2 정정(OV5 차단 6): 이 문구는
// 과거 "MOE_P1_TRACE 게이트와 무관하게 항상 집계된다"였으나, 그 전제가 OFF에서 실행 경로·
// 바이트가 불변이어야 한다는 §3-5-7 요구를 깨는 결함의 근거였다. 이제 호출측
// (ggml_moe_direct_acquire)과 이 함수 구현 내부 양쪽에서 ggml_moe_trace_enabled() 게이트를
// 거친 뒤에만 집계된다(이중 게이트 — 사양 §4-d). ggml-moe-direct.cpp의 metrics JSON이 조회한다.
// ---------------------------------------------------------------------------

GGML_API void     ggml_moe_trace_note_acquire_call(int32_t n_ids); // acquire() 진입마다 1회 호출
GGML_API uint64_t ggml_moe_trace_acquire_call_count(void);
// out8[0..6] = n_ids 1..7의 호출 수, out8[7] = n_ids>=8 총합.
GGML_API void     ggml_moe_trace_n_ids_histogram(uint64_t out8[8]);

// 배선 3단계 신설(사양 §6-2 P2 "독립 chunk-call counter와 O5 수 정합") — mul_mat_id의 chunk
// while 루프가 실제로 one_chunk를 호출할 때마다 1회(등록 텐서·트레이스 ON 한정으로 호출측이
// 게이트) 집계하는 순수 대조용 카운터. O5(compute_enter)는 op×worker당 최대 1회만 발화하므로
// (사양 §2-1) 이 값과 O5 발화 수를 비교하면 게이지가 실수로 chunk 단위로 회귀하지 않았음을
// selftest가 독립적으로 검증할 수 있다 — trace 레코드(FixedRecord)와는 별도의 단순 atomic
// 카운터라 판독기 파싱 없이 즉시 비교 가능. off 상태에서는 호출측이 게이트하므로 이 함수
// 자체도 무동작(카운트 불변).
GGML_API void     ggml_moe_trace_note_chunk_call(void);
GGML_API uint64_t ggml_moe_trace_chunk_call_count(void);

// ---------------------------------------------------------------------------
// 카운터 · 통계 조회(selftest·metrics 출력용)
// ---------------------------------------------------------------------------

GGML_API uint64_t ggml_moe_trace_event_count(void);          // 기록 성공한 레코드(슬롯) 총수
// ★배선 R1(사양 §4): 준비 안 된 상태에서의 기록 시도 + 링 덮어쓰기 + prefill 버퍼 포화의
// 합성값. 표본 유실 경로가 하나라도 발생하면 0이 아니게 되어 §6-2 P6(timeline_drop_count==0)
// 관문이 자동으로 실패하도록 만든다 — 개별 원인은 아래 두 API로 분해 조회한다.
GGML_API uint64_t ggml_moe_trace_timeline_drop_count(void);
GGML_API uint64_t ggml_moe_trace_qpc_frequency(void);        // QueryPerformanceFrequency 원값(캐시, off면 0)
GGML_API uint64_t ggml_moe_trace_ring_capacity_records(void);// 링버퍼 총 슬롯 수(off라도 상수 반환)

// 배선 R1 신설(사양 §4) — timeline_drop_count 합성값의 개별 성분 조회. 비Windows 스텁도
// 대칭으로 0을 반환한다.
GGML_API uint64_t ggml_moe_trace_overwrite_count(void);          // 링을 한 바퀴 넘게 덮어쓴 횟수
GGML_API uint64_t ggml_moe_trace_prefill_overflow_count(void);   // prefill 버퍼가 가득 차 못 쓴 횟수
GGML_API uint64_t ggml_moe_trace_anchor_negative_count(void);    // ★S7a — anchor delta raw 음수 COMPUTE_EXIT 수(세션 누적)

// ---------------------------------------------------------------------------
// 테스트 전용
// ---------------------------------------------------------------------------

// 링버퍼·카운터·게이지·ready 상태를 리셋한다. ★env 캐시(ggml_moe_trace_enabled()의 최초
// 판정)는 리셋하지 않는다 — ggml_moe_direct_reset_for_test()와 동일 관례(모듈 §13 참조).
// MOE_P1_TRACE=1 프로세스 안에서 반복 테스트할 때 쓴다.
GGML_API void ggml_moe_trace_reset_for_test(void);

// 링버퍼의 논리 인덱스(0=최초 기록, 래핑 무관 누적치) 위치의 64B 원본을 복사한다.
// 반환: 성공 true(아직 한 번도 안 쓰였거나 이미 링 밖으로 밀려난 인덱스가 아닐 때) / false.
GGML_API bool ggml_moe_trace_debug_copy_record(uint64_t logical_index, void * out64);

// 이 프로세스에서 MOE_P1_TRACE가 활성 판정으로 캐시되었는지(디버그 표시용 — 값 자체는
// ggml_moe_trace_enabled()와 동일하되, 이 함수는 아직 캐시가 없다면 "판정을 트리거하지
// 않고" false를 반환한다는 차이가 없다: 캐시 여부 확인 목적이 아니라 가독성용 별칭이다).
GGML_API bool ggml_moe_trace_ready_for_test(void); // init까지 끝났는지(링버퍼 유효)

#ifdef __cplusplus
}
#endif
