// MoE 섀도 드래프트 관측 모듈 (D-A1 판별 실험) — 사양: bench/techdev/decode_hunt/SPEC_SHADOW_DRAFT.md ★v1.0
//
// 이 모듈은 "관측만" 한다. 메인 디코드의 어떤 상태·관측·출력도 바꾸지 않으며(SPEC §3 전건 게이트),
// 힌트 소비(축출/프리페치 개입) 배선은 존재하지 않는다(SPEC §0).
//
// 구성 요소
//   ① env 게이트  MOE_SHADOW_DRAFT=depth<D>,every<N>   (미설정 = 완전 OFF · 어떤 상태도 할당 안 함)
//   ①' ★benchmark 계측 게이트  MOE_SHADOW_BENCH=1      (섀도 **실행**과 독립 — 아래 표)
//   ② C3 캐시 view  shadow_view_begin/end — dispatcher drain + mutation epoch 동결 +
//      (record, slot, generation, ptr) 스냅샷. view 중 demand/prefetch/claim/victim/tick/lease/
//      class/pending mutation 금지. 조회는 tick-중립·읽기 제출 0.
//      ★view_begin/end 의 **구현은 ggml-moe-direct.cpp** 에 있다(캐시 내부 상태 소유자).
//   ③ 계측 wire  pa_shadow.pa(kind=5·96B) · pa_shadow_steps.pa(kind=6·64B) ·
//      pa_shadow_spans.pa(kind=7·64B — G-SD3 스팬 축) ·
//      pa_shadow_cpugpu.pa(kind=8·96B — G-SD3 ⑦ CPU/backend 원가 축)
//      — 공통 128B 헤더 관례(MOEPA1 · boot_id 조인 · complete · record_bytes · kind)를 따른다.
//      kind=5/6/7 의 ABI 는 동결이다(96/64/64B·오프셋). 계측이 더 필요하면 **새 kind** 를 만든다.
//
// ★★★ 게이트 2축 행렬 (r2 M2 — "계측 모순" 수리 · SPEC §4 G-SD3 ⑨)
//   ⑨ 는 **동일 ordinal 의 OFF/ON 짝 delta** 를 요구한다. 그런데 TARGET 을 섀도 실행 게이트에
//   묶어 두면 OFF 팔에서는 엔진이 분모를 만들 수 없다(생산자↔소비자 비양립). 그래서 **계측
//   게이트를 실행 게이트에서 떼어낸다**. 제품 OFF(둘 다 미설정)의 "섀도 흔적 0" 계약은 불변이다.
//
//     MOE_SHADOW_BENCH | MOE_SHADOW_DRAFT | 동작
//     -----------------+------------------+------------------------------------------------------
//     미설정           | 미설정           | ★제품 OFF — 흔적 0. QPC 호출·상태 할당·사이드카 전무.
//     미설정           | 설정             | 섀도 실행 ON. 계측은 실행이 켜지면 자동 포함(상위집합)
//                      |                  | 이라 kind5/6/7/8 을 전부 낸다.
//     설정             | 미설정           | ★짝런 OFF 팔 — 섀도는 **돌지 않고** TARGET(kind7)+
//                      |                  | 원가(kind8)만 수집한다. kind5/6 은 만들지 않는다
//                      |                  | ("kind5/6 이 있다 ⟺ 섀도가 실행됐다" 를 유지).
//     설정             | 설정             | ★짝런 ON 팔 — 4종 전부.
//
//   TARGET 스팬의 구간 정의도 함께 교정됐다: 비동기 제출 둘레가 아니라 **토큰 사이클 전체**
//   (ordinal k 의 graph 진입 ~ ordinal k+1 의 graph 진입)다. 섀도 실행은 sampler accept 뒤
//   서버에서 일어나므로, 이 정의라야 ON/OFF 비율이 섀도의 실비용을 담는다. 구간 산출과 사이클
//   회전은 src/llama-context.cpp 가 소유하고, 이 모듈은 wire 만 소유한다.
//
// Windows 전용: 비Windows 빌드에서는 전부 no-op/실패 스텁으로 컴파일된다(Phase A 관례 동일).
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// wire 상수 (SPEC §2 — static_assert 로 .cpp 에서 잠근다)
// ---------------------------------------------------------------------------
#define GGML_MOE_SHADOW_KIND_LAYER   5    // pa_shadow.pa
#define GGML_MOE_SHADOW_KIND_STEP    6    // pa_shadow_steps.pa
#define GGML_MOE_SHADOW_KIND_SPAN    7    // pa_shadow_spans.pa  ★r1 차단 6(G-SD3) 신설 축
#define GGML_MOE_SHADOW_KIND_COST    8    // pa_shadow_cpugpu.pa ★r2 M2(G-SD3 ⑦) 신설 축
#define GGML_MOE_SHADOW_LAYER_BYTES 96
#define GGML_MOE_SHADOW_STEP_BYTES  64
#define GGML_MOE_SHADOW_SPAN_BYTES  64
#define GGML_MOE_SHADOW_COST_BYTES  96
#define GGML_MOE_SHADOW_PRED_CAP    16    // n_pred <= 16 · 미사용 칸 = 0xFFFF

// 스팬 식별자 "해당 없음"(프로세스 수준 스팬 — writer 등). j 는 0 을 쓴다.
#define GGML_MOE_SHADOW_ID_NA       0xFFFFFFFFu

// 섀도 그래프 역할값(C0) — ggml_backend_sched / llama_context 에 전달되는 값과 동일 도메인.
#define GGML_MOE_ROLE_MAIN    0
#define GGML_MOE_ROLE_SHADOW  1

// ---------------------------------------------------------------------------
// ① env 게이트 (값은 최초 호출 시 1회 캐시 — 기본 완전 OFF)
// ---------------------------------------------------------------------------

// MOE_SHADOW_DRAFT 이 유효 파싱되면 true. 미설정·빈 문자열·파싱 실패 = false(경고 1회).
// 파라미터 일반형: "depth<D>,every<N>" (키 순서 무관·공백 허용·대소문자 무시).
// 예) MOE_SHADOW_DRAFT=depth8,every8
GGML_API bool    ggml_moe_shadow_enabled(void);
GGML_API int32_t ggml_moe_shadow_depth(void);    // D — OFF 면 0
GGML_API int32_t ggml_moe_shadow_every(void);    // N — OFF 면 0

// ---------------------------------------------------------------------------
// ①' ★benchmark 계측 게이트 (MOE_SHADOW_BENCH=1 · 섀도 **실행** 게이트와 독립)
// ---------------------------------------------------------------------------
// true 면 kind=7 스팬 축과 kind=8 원가 축을 수집·기록한다. 섀도 실행(kind5/6·사설 컨텍스트·
// view·checkpoint)은 **켜지 않는다** — 그건 오직 MOE_SHADOW_DRAFT 의 일이다.
// 파싱: "1"/"on"/"true"/"yes" = ON · "0"/"off"/"false"/"no"/미설정 = OFF · 그 외 = OFF + 경고 1회.
// ★이 게이트가 존재하는 이유는 SPEC §4 G-SD3 ⑨ 의 분모(OFF 팔 TARGET)를 엔진이 만들 수
//   있게 하기 위해서다. 제품 런(둘 다 미설정)은 이 함수가 false 를 돌려주고 아무 상태도
//   만들지 않으므로 "섀도 흔적 0" 계약이 그대로다.
GGML_API bool ggml_moe_shadow_bench_enabled(void);

// 실행 게이트 OR 계측 게이트 — 사이드카를 하나라도 쓸 상황인가(내부 배선 편의용).
GGML_API bool ggml_moe_shadow_any_gate(void);

// 사이드카 접두사(MOE_DIRECT_PHASEA_OUT 재사용 — 같은 run 의 조인 대상과 같은 디렉터리에 쓴다).
GGML_API const char * ggml_moe_shadow_out_prefix(void);

// 조인 키. Phase A 가 켜져 있으면 그 boot_id 를 그대로 쓴다(= pa_sets.pa 와 동일값).
// Phase A 가 OFF 면 자체 생성값을 쓰며 이 경우 pa_sets.pa 조인은 성립하지 않는다(판독기 fail-close).
GGML_API uint64_t ggml_moe_shadow_boot_id(void);

// ---------------------------------------------------------------------------
// ② C3 캐시 view — 구현은 ggml-moe-direct.cpp (캐시 상태 소유자)
// ---------------------------------------------------------------------------

// dispatcher drain + mutation epoch 동결 + (record, slot, generation, ptr) view 구성.
// 성공 0 / 실패 <0 (-1 미seal · -2 중첩 진입 · -3 drain 실패). 실패 시 view 는 열리지 않는다.
GGML_API int  ggml_moe_shadow_view_begin(void);

// backend synchronize 는 **호출측(드라이버)** 이 먼저 수행하고 이 함수를 부른다 —
// 이 함수는 generation 재검증 + epoch 해제만 한다. 성공 0 / generation drift <0(fail-close).
GGML_API int  ggml_moe_shadow_view_end(void);

GGML_API bool ggml_moe_shadow_view_active(void);

// tick-중립 residency 조회. 반환 = present 개수(<0 = 오류), out_mask 비트 i = experts[i] 상주 여부.
// 캐시/LRU/lease/class 를 일절 건드리지 않으며 읽기를 제출하지 않는다.
GGML_API int32_t ggml_moe_shadow_view_present_mask(int32_t mlpos, const int32_t * experts,
                                                   int32_t n, uint16_t * out_mask);

// view 스냅샷 포인터(무lease). view 활성 중에만 유효하며, 스냅샷 시점 generation 과 현재
// generation 이 다르면 NULL(= 동결 위반 탐지). ggml_moe_direct_ptr() 단독 재사용 금지의 대체물.
GGML_API const void * ggml_moe_shadow_view_ptr(const struct ggml_tensor * t, int32_t expert_id);

// C3 진입/탈출 등식 판정용 스냅샷 — slot owner/generation/last_used/lease/class · record table ·
// tick · pending I/O · control queue · cache/read-order digest 를 한 값으로 접는다.
GGML_API void ggml_moe_shadow_state_digest(uint64_t * out_cache_digest, uint64_t * out_tick,
                                           uint64_t * out_readfile_ops, uint64_t * out_pending);

// 층별 레코드 stride(byte) — G-SD1 byte 가중 recall 의 B[e]. 범위 밖 = 0.
GGML_API uint64_t ggml_moe_shadow_layer_stride_bytes(int32_t mlpos);

// 등록된 전문가 가중치 텐서의 mlpos(= moe_layers 내 위치). 미등록/미seal = -1.
// 섀도 forward 가 present 조회에 쓸 인덱스를 얻는 유일 창구다.
GGML_API int32_t ggml_moe_shadow_mlpos_of(const struct ggml_tensor * t);

// ---------------------------------------------------------------------------
// ③ 계측 wire — 섀도 드라이버가 호출한다
// ---------------------------------------------------------------------------

// 섀도 스텝 j 시작. origin = 확정 토큰을 산출한 main graph. target = origin_decode_ordinal + j.
GGML_API void ggml_moe_shadow_step_begin(uint32_t request_index,
                                         uint32_t origin_graph_seq,
                                         uint32_t origin_decode_ordinal,
                                         uint32_t target_decode_ordinal,
                                         int32_t  j);

// 섀도 스텝 j 종료 → kind=6 레코드 1행. 비용 3항(resident_records·dropped_records·
// resident_gemv_bytes)은 층 레코드가 누적한 값을 그대로 쓴다 — 드라이버가 따로 셀 필요가 없고,
// 층 기록과 스텝 합계가 구조적으로 어긋날 수 없다.
GGML_API void ggml_moe_shadow_step_end(void);

// 층 1개의 마스크-재정규화 결과 → kind=5 레코드 1행. step_begin~step_end 사이에서만 유효.
//   pred            = all-expert 라우터 top-k P(pre-mask · 라우터 선택 순서 그대로)
//   present_mask    = pred 칸 1:1 비트(bits[n_pred..15]=0)
//   gate_mass_present = Σ(present 원 가중)/Σ(전체 선택 가중) — 진단 전용(F32 계산 후 f16 직렬화)
//   all_dropped     = present mass 0 으로 sparse 기여를 0 처리했으면 1
GGML_API void ggml_moe_shadow_layer_record(int32_t  layer,
                                           const int32_t * pred, int32_t n_pred,
                                           uint16_t present_mask,
                                           float    gate_mass_present,
                                           int32_t  all_dropped,
                                           uint32_t record_stride_bytes,
                                           uint64_t route_ready_qpc,
                                           uint64_t layer_begin_qpc,
                                           uint64_t layer_end_qpc);

// ★층 타임라인 창(r1 차단 6 수리 — layer_begin_qpc 출처 정정).
//   반환 = "현 스텝에서 **아직 기록되지 않은 다음 층**이 시작된 시각" =
//   직전 층 레코드의 layer_end_qpc(스텝의 첫 층이면 step_begin_qpc). 스텝이 열려 있지 않거나
//   게이트 OFF 면 0(호출측이 fallback 을 고른다).
//   이 커서를 layer_begin_qpc 로 실으면 한 스텝의 층 span 들이 [step_begin, step_end] 를
//   겹침 없이 분할하고, (route_ready − layer_begin) 이 **라우터 앞 구간**을 실제로 덮는다.
//   커서는 layer_record() 가 layer_end_qpc 로 전진시킨다(단조 — 역행 값은 무시).
GGML_API uint64_t ggml_moe_shadow_layer_window_qpc(void);

// ---------------------------------------------------------------------------
// ③' 계측 wire 제2축 — 스팬(kind=7 · 64B · `<prefix>_shadow_spans.pa`)
//     ★r1 차단 6(SPEC §4 G-SD3 ⑥①) — kind5/6 ABI 는 동결이므로 clone/checkpoint ·
//     dispatcher drain · device sync · writer · target span 은 **별도 축**에 싣는다.
//     헤더 관례는 kind5/6 과 동일(MOEPA1 · layout_version=1 · FNV-1a-64 payload digest ·
//     complete · boot_id), 버퍼 상한·complete latch·run_error latch 도 동일 규약을 공유하며
//     기존 ggml_moe_shadow_flush() 가 함께 기록한다.
// ---------------------------------------------------------------------------

// 스팬 종류. wire 에는 u8 로 실린다(범위 밖 값 = NONE 으로 기록 + UNKNOWN_KIND 플래그).
enum ggml_moe_shadow_span_kind {
    GGML_MOE_SHADOW_SPAN_NONE        = 0,
    GGML_MOE_SHADOW_SPAN_TARGET      = 1,  // ① target span — 해당 ordinal 의 main decode 구간
    GGML_MOE_SHADOW_SPAN_TRIGGER     = 2,  // 섀도 트리거 1회 전체(j=1..D 묶음)
                                           // ★⑨ paired delta 의 축은 TARGET 이다 — TRIGGER 는
                                           //   OFF 팔에 아예 존재하지 않아 짝을 이룰 수 없다.
                                           //   TRIGGER 는 ON 팔 전용 섀도 비용(진단)으로만 쓴다.
    GGML_MOE_SHADOW_SPAN_CLONE       = 3,  // ⑥ 사설 컨텍스트/시퀀스 clone
    GGML_MOE_SHADOW_SPAN_CHECKPOINT  = 4,  // ⑥ KV full sequence checkpoint 뜨기
    GGML_MOE_SHADOW_SPAN_RESTORE     = 5,  // ⑥ checkpoint 심기/되읽기 + 사설 seq 폐기
    GGML_MOE_SHADOW_SPAN_DRAIN       = 6,  // ⑥ dispatcher drain(view_begin 내부)
    GGML_MOE_SHADOW_SPAN_DEVICE_SYNC = 7,  // ⑥ backend synchronize
    GGML_MOE_SHADOW_SPAN_VIEW        = 8,  // view_begin..view_end 전 구간
    GGML_MOE_SHADOW_SPAN_WRITER      = 9,  // ⑥ 사이드카 writer(aux = 기록 payload 바이트)
    GGML_MOE_SHADOW_SPAN_KIND_COUNT  = 10
};

// wire flags(u8)
#define GGML_MOE_SHADOW_SPAN_FLAG_BAD_ORDER    0x01  // t_end < t_begin — 시계 혼용/짝 어긋남
#define GGML_MOE_SHADOW_SPAN_FLAG_UNKNOWN_KIND 0x02  // span_kind 범위 밖(ABI 불일치)

// 관측 시계 — kind5/6/7 의 모든 QPC 는 이 시계에서 나와야 한다(Phase A OFF 여도 유효).
GGML_API uint64_t ggml_moe_shadow_now_qpc(void);

// 스팬 1건 기록. ★begin/end 짝 API 를 두지 않는 이유: 섀도 드라이버는 OOM·예외·조기
// 반환이 있는 경로(P43j)라 "열어 두고 못 닫는" 상태가 실재한다 — 무상태 note 는 짝이
// 어긋날 수 없고, 재진입·중첩도 자동으로 안전하다. 호출측은 ggml_moe_shadow_now_qpc() 로
// t0/t1 을 직접 집는다.
//   span_kind             = enum ggml_moe_shadow_span_kind
//   request_index/origin_decode_ordinal = 조인 키. 프로세스 수준 스팬은 GGML_MOE_SHADOW_ID_NA.
//   j                     = 섀도 스텝 번호(1..D). 스텝과 무관하면 0.
//                           TARGET/TRIGGER·트리거 1회 설정 비용(clone/checkpoint/drain/sync)은
//                           j=0 + origin=해당 ordinal 로 낸다.
//   target_decode_ordinal = **파생값** origin + j — 호출측 인자로 받지 않아 kind5/6 의 조인
//                           계약과 어긋날 수 없다. ★r2 M2 정정: origin==NA 면 NA 이고,
//                           **j==0 이면 TARGET 만** target 을 싣는다(나머지 j=0 스팬은 NA).
//                           TARGET 은 판독기가 (request_index, target_decode_ordinal) 로 ON/OFF
//                           짝을 맞추므로 j=0 이어도 키가 필요하고, 그 밖의 j=0 스팬은 소비자
//                           계약이 NA 를 요구한다(ordinal 은 origin 필드가 그대로 들고 있다).
//   aux_u64               = ★WRITER 전용(payload bytes). 그 외 종류에 실린 값은 wire 에서
//                           떨군다 — 소비자(shadow_reader.py)의 하드 계약이 "WRITER 외 0" 이고,
//                           어기면 그 런 전체가 fail-close 하기 때문이다.
// 게이트(실행 OR 계측) OFF 면 완전 no-op. 버퍼 상한 초과·OOM 은 kind5/6 과 동일하게
// latch + complete=0.
GGML_API void ggml_moe_shadow_span_note(int32_t  span_kind,
                                        uint32_t request_index,
                                        uint32_t origin_decode_ordinal,
                                        int32_t  j,
                                        uint64_t t_begin_qpc,
                                        uint64_t t_end_qpc,
                                        uint64_t aux_u64);

// ---------------------------------------------------------------------------
// ③'' 계측 wire 제3축 — 원가(kind=8 · 96B · `<prefix>_shadow_cpugpu.pa`)
//     ★r2 M2 / SPEC §4 G-SD3 ⑦ "CPU thread time + GPU/backend interval".
//     wall-QPC 스팬(kind=7)만으로는 ⑦을 대체할 수 없다 — 같은 wall 안에서 CPU 를 얼마나
//     태웠는지, device 가 얼마나 물려 있었는지를 분리해야 겹침(overlap) 투영이 "낙관 상한"
//     딱지를 뗄 수 있기 때문이다. kind=7 은 64B 로 **동결·static_assert** 되어 있으므로
//     필드를 넓히지 않고 같은 조인 키를 쓰는 **별도 kind** 를 만든다.
//     조인: `(request_index, origin_decode_ordinal, span_kind, j)` — kind=7 의 같은 행과 1:1.
//       ※ **target 이 아니라 origin 이 키**다: j=0 인 비-TARGET 스팬은 target 이 NA 라(kind=7 과
//         동일 파생식) target 을 키로 쓰면 같은 요청의 여러 ordinal 이 한 칸에 뭉친다.
//         TARGET 행은 origin==target 이라 어느 쪽으로 봐도 같다.
//     헤더 관례·버퍼 상한·complete latch·run_error latch·boot_id 는 kind5/6/7 과 공유한다.
//     발행 지점(현재): TARGET(토큰 사이클 · llama-context.cpp 회전) · TRIGGER(j-루프) ·
//       VIEW(섀도 view 창 — 섀도 쪽 backend interval 의 유일한 완결 행).
// ---------------------------------------------------------------------------

// wire flags(u8) — kind=8 전용(kind=7 의 flags 와 값 도메인이 다르다).
#define GGML_MOE_SHADOW_COST_FLAG_BAD_ORDER   0x01  // t_end < t_begin
#define GGML_MOE_SHADOW_COST_FLAG_CPU_UNAVAIL 0x02  // CPU 시계 조회 실패 → cpu_* 필드 무의미
#define GGML_MOE_SHADOW_COST_FLAG_CROSS_THREAD 0x04 // begin/end 가 다른 스레드 → thread_* 무의미
#define GGML_MOE_SHADOW_COST_FLAG_NO_BACKEND  0x08  // 구간 안에 submit/sync 관측 0 → device_* 미측정
#define GGML_MOE_SHADOW_COST_FLAG_BENCH_ONLY  0x10  // 섀도 실행 OFF 인 팔에서 수집됨(짝런 OFF 분모)

// CPU 시계 스냅샷. 호출 스레드 + 프로세스 전체를 한 번에 집는다.
//   thread_* = 이 스레드만 — 메인 디코드 스레드가 곧 섀도 드라이버 스레드이므로(동기 실행 D6)
//              토큰 사이클의 thread CPU 는 섀도의 호스트측 비용을 **포함**한다.
//   proc_*   = 프로세스 전 스레드(CPU backend worker·CUDA 드라이버 스레드 포함).
//   *_cycles = QueryThread/ProcessCycleTime(고해상도) · *_cpu_100ns = Get*Times(커널+유저,
//              시스템 틱 해상도라 거칠다 — 교차 검산용이지 주 지표가 아니다).
struct ggml_moe_shadow_cpu_times {
    uint64_t thread_cycles;
    uint64_t thread_cpu_100ns;
    uint64_t proc_cycles;
    uint64_t proc_cpu_100ns;
    uint64_t thread_id;        // GetCurrentThreadId — CROSS_THREAD 판정용
    int32_t  ok;               // 0 = 조회 실패(호출측이 채우지 않아도 안전하게 0)
    int32_t  reserved;
};

// 한 스팬 구간 안에서 관측한 backend 이벤트 누적. **호출측이 소유**한다(이 모듈은 시계만 제공).
//   wait_qpc          = ggml_backend_sched_synchronize() 에 호스트가 블록된 시간의 합
//   first_submit_qpc  = 구간 안 첫 graph_compute 제출 시각(0 = 관측 없음)
//   last_sync_end_qpc = 구간 안 마지막 sync 반환 시각(0 = 관측 없음)
//   device interval   = last_sync_end − first_submit (이 모듈이 계산) — ★host 관측 창이지
//                       GPU 타임스탬프가 아니다(정확한 GPU 구간은 CUDA event 가 필요하고 그건
//                       ggml-cuda 소유다). 판독기는 이를 "backend interval(host-observable)"
//                       로만 읽어야 한다.
//   ★계약(r2 마감·판독기 결속): 이 값의 판독기 출력 라벨은 **`host_observed_device_span`** 로
//     고정한다. `gpu_busy`·`device_busy`·`gpu_time` 등 GPU 점유를 함의하는 이름으로 보고하는
//     것을 금한다 — submit~sync 사이에는 큐 대기·다른 스트림·유휴가 섞이므로 이 값은 GPU busy
//     의 **상한**이지 측정치가 아니다. SPEC §4 G-SD3 의 "낙관 상한" 딱지가 여기에도 그대로
//     적용된다(진짜 GPU 구간은 CUDA event 도입 전까지 미측정으로 남는다).
struct ggml_moe_shadow_backend_acc {
    uint64_t wait_qpc;
    uint64_t first_submit_qpc;
    uint64_t last_sync_end_qpc;
    uint32_t n_submits;
    uint32_t n_syncs;
};

// 현재 시각의 CPU 시계를 집는다. 실패해도 out 은 항상 정의된 값으로 채워진다(ok=0).
GGML_API void ggml_moe_shadow_cpu_now(struct ggml_moe_shadow_cpu_times * out);

// kind=8 레코드 1행. 차분(end−begin)은 이 함수가 계산한다 — 호출측 산술 실수를 원천 차단하고,
// 역행(감소)한 카운터는 0 으로 접으며 CPU_UNAVAIL 을 세운다(음수 랩어라운드 금지).
// 게이트(실행 OR 계측)가 전부 OFF 면 완전 no-op. 버퍼 상한·OOM 처분은 kind5/6/7 과 동일.
GGML_API void ggml_moe_shadow_cost_note(int32_t  span_kind,
                                        uint32_t request_index,
                                        uint32_t origin_decode_ordinal,
                                        int32_t  j,
                                        uint64_t t_begin_qpc,
                                        uint64_t t_end_qpc,
                                        const struct ggml_moe_shadow_cpu_times   * cpu_begin,
                                        const struct ggml_moe_shadow_cpu_times   * cpu_end,
                                        const struct ggml_moe_shadow_backend_acc * backend);

// ★런 오류 latch (r1 차단 2·3 수리) — 관측이 신뢰할 수 없게 된 순간을 사이드카에 못박는다.
//   한 번 서면 이 프로세스의 사이드카는 `complete=0` 으로 기록되어 판독기가 fail-close 한다.
//   메인 토큰 송신·회계에는 영향을 주지 않는다(섬도 오류가 서빙을 멈추지 않는다는 계약).
GGML_API void     ggml_moe_shadow_run_error(const char * reason);
GGML_API uint64_t ggml_moe_shadow_run_error_count(void);

// 사이드카 기록(멱등). ggml_moe_direct_finalize() 가 Phase A flush 뒤에 1회 호출.
//   · 섀도 실행 ON  : kind=5/6/8/7 4종
//   · 계측만 ON     : kind=8/7 2종 (kind5/6 은 만들지 않는다 — "있다 ⟺ 섀도가 돌았다")
//   · 둘 다 OFF     : 아무것도 하지 않는다(제품 OFF 흔적 0)
// kind=5/6/8 기록 비용은 그 자리에서 WRITER 스팬으로 남고 kind=7 을 **마지막에** 쓴다(스팬 파일
// 자신의 기록 비용은 자기참조라 남길 수 없다 — 판독기는 그 한 건만 미계측으로 본다).
GGML_API void ggml_moe_shadow_flush(void);

// ===========================================================================
// 테스트 전용 진입점 — 프로덕션 코드는 아래 심볼을 절대 호출하지 마라.
// ===========================================================================

GGML_API void ggml_moe_shadow_reset_for_test(void);

// 게이트 강제(env 무시). on=0 이면 완전 OFF 로 되돌린다.
GGML_API void ggml_moe_shadow_test_force_gate(int32_t on, int32_t depth, int32_t every);

// ★계측 게이트 강제(env 무시) — 짝런 OFF 팔(섀도 실행 OFF · 계측 ON)을 테스트에서 재현한다.
GGML_API void ggml_moe_shadow_test_force_bench(int32_t on);

// 버퍼 계수 조회. complete=0 이면 버퍼 상한 초과(fail-close 대상).
GGML_API void ggml_moe_shadow_test_counts(uint64_t * out_layers, uint64_t * out_steps,
                                          uint64_t * out_dropped, int32_t * out_complete);

// 버퍼 레코드 1건 원바이트 복사(bytes = 96 / 64 / 64 / 96). 성공 0 / 범위 밖·크기 불일치 <0.
GGML_API int ggml_moe_shadow_test_layer_bytes(uint64_t idx, void * out, uint64_t bytes);
GGML_API int ggml_moe_shadow_test_step_bytes (uint64_t idx, void * out, uint64_t bytes);
GGML_API int ggml_moe_shadow_test_span_bytes (uint64_t idx, void * out, uint64_t bytes);
GGML_API int ggml_moe_shadow_test_cost_bytes (uint64_t idx, void * out, uint64_t bytes);

// 스팬/원가 버퍼 행 수(테스트 전용 — test_counts 의 시그니처는 동결이라 별도 창구로 뺀다).
GGML_API uint64_t ggml_moe_shadow_test_span_count(void);
GGML_API uint64_t ggml_moe_shadow_test_cost_count(void);

// 지정 접두사로 즉시 기록(실 emitter→reader 왕복 픽스처용 — kind=5/6/7/8 4종).
// 픽스처 재현성을 위해 이 경로는 WRITER 스팬을 추가로 만들지 않는다. 성공 0 / 실패 <0.
GGML_API int ggml_moe_shadow_test_write_to(const char * prefix);

// 음성 대조 주입 — DSV4 compressor 1바이트 변조 / observer guard 제거 / sync 관측 누락.
// 프로덕션 경로는 항상 0.
// ★r4 R4' — env 표면 `MOE_SHADOW_FAULT=0x10`(16진/10진 · 1..0x7f)도 같은 비트를 세운다. 모델
//   적재 서버 프로세스에는 이 C API 를 부를 지점이 없어서, phase-0 의 음성처럼 **실 런에서만
//   도달하는 소비자**를 무장할 수단이 없으면 phase-0 명령 자체가 성립하지 않는다.
//   env 는 게이트(실행 OR 계측)가 켜진 뒤에만 읽으므로 제품 완전 OFF 의 "흔적 0" 계약은
//   불변이고, 무장되면 stderr 에 "이 런의 관측은 의도적으로 깨져 있으니 데이터로 쓰지 마라"는
//   경고를 1회 남긴다.
GGML_API void ggml_moe_shadow_test_fault(int32_t bits);
GGML_API int32_t ggml_moe_shadow_test_fault_bits(void);

#define GGML_MOE_SHADOW_FAULT_NONE            0
#define GGML_MOE_SHADOW_FAULT_NO_OBS_GUARD    0x01  // C0 관측기 가드 제거 → P43c FAIL 유발
#define GGML_MOE_SHADOW_FAULT_TICK_BUMP       0x02  // view 중 tick 증가      → P43e FAIL 유발
#define GGML_MOE_SHADOW_FAULT_KV_BYTE         0x04  // DSV4 compressor 1바이트 → P43f FAIL 유발
#define GGML_MOE_SHADOW_FAULT_MAIN_OUT        0x08  // main output overwrite  → P43d FAIL 유발
// ★r6 P0 — 0x10 **재정의**(구 `NO_SYNC` 폐지 · 진단 교차 `codex_shadow_phase0_r1.md` §1-C).
//   구 비트는 process-global 이라 `llama_context::synchronize()` 가 역할을 가리지 않고 실
//   fence 를 건너뛰었고, 서버는 섀도 호출 **전** 메인 샘플링에서 `llama_synchronize(ctx_tgt)`
//   를 부르므로(server-context.cpp:3825 · sampling.cpp:540) **메인 토큰열이 깨질 수 있었다** —
//   하위 음성 기법이 상위 계약(SPEC §3 C1 "동기화 후 폐기 · main continuation")을 뒤집는
//   구조였다. 게다가 실 fence 를 건너뛴 뒤에도 `moe_bench_note_sync()` 는 `n_syncs` 를 무조건
//   올려 판독기가 PASS 했다(= 음성이 아무것도 검사하지 못했다).
//   지금은 **실 fence 를 언제나 실행**하고(llama-context.cpp `synchronize()`), SHADOW 역할의
//   **sync 관측 기록만** 생략한다(`moe_bench_note_sync()`). 그래서 이 비트는 "메인 무관 ·
//   관측 누락"이며, 판독기가 그 누락을 fail-close 하는지(= 판독 경로 검증)가 음성의 본래
//   목적 그대로다. 소비자 계약: 무장 시 SHADOW 누산기의 `n_syncs` 가 증가하지 않아 VIEW 원가
//   행이 `n_submits>0 ∧ n_syncs==0 ∧ device_interval==0` 으로 남고, 판독기의 VIEW 하드 계약이
//   그 행을 잡는다. main 역할 누산기·실 fence·메인 출력에는 **어떤 영향도 없다**.
//   ★구 파괴적 동작(전역 fence 생략)은 **되살리지 않는다** — 남기려면 별도
//   `GLOBAL_NO_SYNC_DESTRUCTIVE` 로 명명하고 게이트·데이터에는 쓰지 말라는 것이 교차 권고다.
#define GGML_MOE_SHADOW_FAULT_SYNC_OBS_DROP   0x10  // SHADOW sync **관측**만 누락 → reader FAIL
// ★r3 M1 / ★r4 R1' — "실 OOM 음성"의 두 창구. 둘 다 **실 실패 분기 그 자체**로 떨어뜨린다:
//   무장 시 프로덕션 `try` 안에서 실제 `std::bad_alloc` 을 던져 프로덕션 `catch` 가 처분한다.
//   처분을 흉내 내는 별도 분기는 두지 않는다(그러면 음성이 프로덕션과 다른 코드를 검사한다).
#define GGML_MOE_SHADOW_FAULT_SCRATCH_OOM     0x20  // D2 스크래치 resize try 안에서 실 bad_alloc
                                                    //  → catch(...) 처분: ready=false·latch·rc<0
#define GGML_MOE_SHADOW_FAULT_WRITER_OOM      0x40  // writer buf_append try 안에서 실 bad_alloc
                                                    //  → catch(bad_alloc) 처분: complete=0 latch

#ifdef __cplusplus
}
#endif
