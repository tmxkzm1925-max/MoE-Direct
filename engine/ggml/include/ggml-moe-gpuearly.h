// MoE GPU측 조기 전달 계측(v2) 모듈 — 사양: bench/techdev/repack_space/PROBE_GPU_EARLY_SPEC_R1.md
// ★FROZEN v1.0 (교차 r1~r6 완주). 이 모듈은 사양 §4 의 "C wire" 를 소유한다.
//
// 이 모듈은 "관측만" 한다. 캐시·플래너·PhaseA/PhaseB·IO3·lead-time 의 어떤 상태도 읽거나 바꾸지
// 않으며, 조기 D2H 를 실제로 수행하는 배선은 **존재하지 않는다**(이 라운드는 계측이지 시공이 아니다).
// 기본 OFF 에서 기존 거동은 완전 불변이다.
//
// 재는 것(사양 §1):
//   Δ_tail_gpu(i) = G_boundary(i) − G_router(i)
//   — 같은 stream 의 GPU 이벤트 쌍 ElapsedTime(단일 클럭 도메인·해상도 ~0.5µs).
//     G_router   = 라우터 확정(fused topk-moe 또는 argsort 커널 완료)
//     G_boundary = 그 split 의 GPU 작업 끝(= host 가 `cudaStreamSynchronize` 로 기다리는 지점)
//   host 스탬프(`sync_return_qpc`·`a1_qpc`)는 별도 병록한다.
//   ★Δ_tail_gpu 는 "조기 D2H 후보 구간"의 **상한**이지 실회수 δ 가 아니다(sync 반환·D2H 비용
//     미포함 — 사양 §1 의 해석 제한을 판독기·보고가 그대로 승계해야 한다).
//
// ★역할 분담(이 파일이 소유하는 것 / 아닌 것)
//   소유:   env 게이트 · POD 링 버퍼(arm 시 전량 선할당) · 원자 인덱스 예약 · 층별 서수(occurrence)
//           · boundary_seq 발번 · pending 회계 · 물리층 결산 카운터 · 사이드카 기록 · ms→ns 변환 규칙
//   비소유: CUDA 이벤트 소유·생성·기록(= `ggml/src/ggml-cuda/moe-gpuearly.cu`) · 훅 지점 ·
//           판독(오프라인 python) · 산술층 결산(판독기).
//   ★이 분리가 계약이다: ggml-base 는 CUDA 를 링크하지 않으므로, GPU 이벤트를 다루는 코드는
//     이 모듈에 있을 수 없다. 대신 CUDA 백엔드가 `set_consumer` 로 소비자를 등록하고,
//     `ggml-backend.cpp` 의 split 경계 훅이 `consume()` 를 부르면 그 소비자가 호출된다.
//
// ★★hot-path 계약(lead-time 모듈과 동형 — 위반은 이 프로브의 존재 이유를 무너뜨린다)
//   ① 버퍼는 **arm 시점(=seal)에 전량 선할당**된 고정 폭 POD 링이다. 칸 예약은 `fetch_add` 1회.
//   ② hot hook 안에서 **문자열 포맷·파일 I/O·동적 할당·getenv·락 획득이 전부 0** 이다.
//      그래서 env 해소는 hook 이 아니라 `ggml_moe_gpuearly_note_seal()` 만 한다.
//   ③ OFF 계약 = "QPC·lock·allocation 0회 + 예측 가능한 fast-gate(캐시 bool) 1회".
//      `ggml_moe_gpuearly_enabled()` 는 해소 여부와 무관하게 원자 로드 1회로 끝나며(미해소=false),
//      그 뒤 훅은 실행문 0 이다.
//
// ★★★publication 계약 — lead-time R3 판과 **동일 기제**(그 라운드의 교차 r1 BLOCKER·r2 R3-1/R3-2
//   폐합분을 그대로 승계한다. 새로 발명하지 않는다):
//     · 커밋은 게이트 확인 **이전에** `w_enter` 를 올리고, 본문 memcpy 뒤 `w_exit` 를 release 로 올린다.
//     · flush 는 ①게이트를 먼저 닫고 ②`w_enter == w_exit` 까지 drain 하고 ③그 뒤에야 record_count
//       를 확정한다. drain 실패 = payload 를 **읽지 않고** `record_count=0 · complete=0 ·
//       dropped=미커밋 수` 인 헤더만 기록한다(live writer 와의 비원자 동시 읽기 회피).
//     · 사이드카는 2단계로 쓴다: 헤더 자리 예약(0) → payload → **최종 재검** → 헤더 backpatch.
//       ⇒ payload 기록 창 안에 끼어든 late 커밋도 파일 헤더에 반영된다.
//
// ★사양 §3 "시간 단위" 의 유일 구현 지점이 이 모듈이다(`ms_to_ns`) — 변환 규칙이 CUDA 쪽과
//   판독기 쪽에 각각 손으로 복제되면 그 순간 사본 드리프트다. 엔진 내부 정규 단위 = ns(u64).
//
// Windows 전용: 비Windows 빌드에서는 전부 no-op 스텁으로 컴파일된다(lead-time·shadow 관례 동일).
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// wire 상수 (사양 §4 — .cpp 에서 static_assert 로 폭·오프셋을 동결한다)
// ---------------------------------------------------------------------------
#define GGML_MOE_GPUEARLY_HDR_BYTES   128
#define GGML_MOE_GPUEARLY_REC_BYTES   64
#define GGML_MOE_GPUEARLY_VERSION     1

// record.router_path (u8)
#define GGML_MOE_GPUEARLY_ROUTER_ARGSORT 0
#define GGML_MOE_GPUEARLY_ROUTER_FUSED   1

// record.flags (u8) — ★분류 우선순위(사양 §3 물리층 · 동결): invalid > direct > warmup.
//   b0=0 이면 invalid(그 행의 Δ 는 무의미하므로 b1/b2 를 **무시**한다) · elif b1 이면 direct ·
//   elif b2 이면 warmup · else eligible. 헤더 카운터도 **같은 우선순위로** 하나만 올린다
//   (그래야 `eligible = record_count − invalid − direct − warmup` 파생식이 성립한다 — 비트별로
//    세면 두 비트가 선 행이 이중 계상되어 그 식이 깨진다).
#define GGML_MOE_GPUEARLY_FLAG_VALID           0x01
#define GGML_MOE_GPUEARLY_FLAG_EXCLUDED_DIRECT 0x02
#define GGML_MOE_GPUEARLY_FLAG_EXCLUDED_WARMUP 0x04

// 층별 서수(occurrence) 테이블 폭. 층 번호가 이 값 이상이면 서수를 발급할 수 없으므로
// drop + fail-close 로 정직 처분한다(조용한 접기 금지). 실 모델은 수십~수백 층이다.
#define GGML_MOE_GPUEARLY_MAX_LAYERS  4096

// ---------------------------------------------------------------------------
// ① 게이트 (env: `MOE_DIRECT_GPUEARLY=1` · 그 외 = off + 경고 1회 · 미설정 = 조용한 off)
//    `MOE_DIRECT_GPUEARLY_OUT`     미설정 = 카운터만(파일 미생성 — lead-time 관례)
//    `MOE_DIRECT_GPUEARLY_MAX_MB`  기본 64 (초과 = drop 계수 + complete=0)
// ---------------------------------------------------------------------------

// ★hot-path 게이트. 원자 로드 1회 + 비교 1회. **해소하지 않는다**(미해소 = false).
//   ★flush 가 게이트를 닫은 뒤에도 false 다(publication 계약 — 폐쇄 = 신규 커밋 차단).
GGML_API bool ggml_moe_gpuearly_enabled(void);

// 관측 시계 — 이 프로브의 모든 QPC 는 여기서 나온다(raw QueryPerformanceCounter · 게이트 무관).
// 호출측은 `enabled()` 가 참일 때만 부른다(OFF 계약의 "QPC 0회"가 그렇게 성립한다).
// ★lead-time 모듈과 **같은 QPC 도메인**이다(둘 다 QueryPerformanceCounter) — 그래서
//   `a1_qpc` 가 lead-time A1 스탬프와 같은 사건을 같은 시계로 가리킨다(사양 §3 `H_i` 의 전제).
GGML_API uint64_t ggml_moe_gpuearly_now_qpc(void);

// 조인 키. IO3 가 무장돼 있으면 `note_seal` 이 그 boot_id 를 그대로 채택한다(= 같은 값).
GGML_API uint64_t ggml_moe_gpuearly_boot_id(void);

// ---------------------------------------------------------------------------
// ② arm — env 해소 + 버퍼 전량 선할당 + 조인 키 채택. seal 경로가 1회 부른다.
//    ★IO3 의 boot_id 해소 **뒤**에 불러야 조인 키를 그대로 채택할 수 있다(사양 §4 "boot_id =
//      기존 A/B wire 동일 계약"). join_boot_id == 0 이면 자체 생성값을 쓰며 IO3 조인은 성립하지
//      않는다(판독기 fail-close — 의도된 침묵 금지이므로 경고 1줄을 남긴다).
//    멱등: 커밋이 시작된 뒤의 재호출은 조인 키를 바꾸지 않는다.
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_gpuearly_note_seal(uint64_t join_boot_id);

// ---------------------------------------------------------------------------
// ③ hot hook — 게이트가 참일 때만 호출된다. 락·할당·포맷 0.
// ---------------------------------------------------------------------------

// 층별 FIFO 서수 발급(사양 §4 occurrence 규칙 · 0-base · run 전체에서 reset 없음).
// ★"그 층의 router anchor 발화 전건이 서수를 소비한다" — prefill/decode/hit-only 구분 없이,
//   direct/warmup 제외 표본도, invalid/drop 행도 서수를 소비한다. 그래야 판독기가 B 축을 같은
//   규칙으로 세어 cardinality 를 대조할 수 있다(불일치 = fail-close · 95% 허용 없음).
// 반환 = 이 발화에 할당된 서수. layer 가 테이블 폭 밖이면 UINT32_MAX 를 돌려주고 drop 계상한다.
GGML_API uint32_t ggml_moe_gpuearly_next_occurrence(uint32_t layer);

// boundary_seq 발번 + pending 개시(사양 §2 replay 상태기계). n_anchors = 이 launch 가 소비할
// 앵커 수(consume 실패 시 결손 계상량). 이미 pending 이 열려 있으면 **그 pending 을 유실로
// 정직 처분**하고(pending_unconsumed_count++ · dropped += 직전 n_anchors · complete=0) 새로 연다.
GGML_API uint64_t ggml_moe_gpuearly_pending_begin(uint64_t n_anchors);

// ★부분 소비 반영(교차 r1 차단① — 대상 결속). 한 launch 의 앵커가 여러 층이면 그 층의 ids 가
// 실체화되는 split 경계마다 **자기 층 몫만** 소비된다. 남은 앵커 수를 그만큼 줄이고, 0 이 되면
// pending 을 정상 종료한다. 남은 수를 정확히 유지해야 중도 유실 시 `dropped` 가 **이미 커밋된
// 앵커까지 과다 계상**하지 않는다 — 유실 회계의 정확성이 이 API 의 존재 이유다.
GGML_API void ggml_moe_gpuearly_pending_consumed(uint64_t n_consumed);

// pending 정상 종료(consume 완료). 열려 있지 않으면 무동작.
GGML_API void ggml_moe_gpuearly_pending_end(void);

// pending 을 유실로 처분한다(재launch·recapture·eviction·파괴 중 pending — 사양 §2 fail-close).
// ★호출측이 **먼저** 유실 앵커 각각에 대해 `next_occurrence()` 를 소비한 뒤 부른다(서수 장부 정합).
GGML_API void ggml_moe_gpuearly_abandon_pending(void);

// 레코드 1행 커밋(hot). flags 는 위 3비트. 사양 §4 레코드 필드 순서 그대로.
GGML_API void ggml_moe_gpuearly_commit(uint64_t boundary_seq,
                                       uint32_t layer,
                                       uint32_t router_path,
                                       uint32_t flags,
                                       uint32_t occurrence,
                                       uint64_t delta_tail_gpu_ns,
                                       uint64_t sync_return_qpc,
                                       uint64_t a1_qpc,
                                       uint64_t graph_uid);

// ★사양 §3 시간 단위 변환의 **유일 구현**: CUDA `cudaEventElapsedTime` 의 float ms → u64 ns.
//   규칙(동결): finite·비음수 검사 → `llround(ms × 1e6)` → 2^63 초과 = invalid.
//   반환 0 = 정상(`*out_ns` 유효) · <0 = invalid(해당 표본은 flags b0=0 행으로 커밋해야 한다).
GGML_API int ggml_moe_gpuearly_ms_to_ns(float elapsed_ms, uint64_t * out_ns);

// 상호배타(fused XOR argsort) 위반 계상 — 같은 capture 안에서 한 층이 두 경로로 앵커를 남기면
// 사양 §6 게이트("메타데이터 기준 fused XOR argsort")가 깨진 것이므로 fail-close 한다.
GGML_API void ggml_moe_gpuearly_note_exclusivity_violation(void);

// 진단 계수 — 라우터 경로별 앵커 발화 수(상호배타 검증의 관측 창구·§7 원자⑤).
GGML_API void ggml_moe_gpuearly_note_anchor(uint32_t router_path);

// 일반 결손 계상(버퍼 밖 사유 — 이벤트 자원 부족·capture 무장 실패 등). complete=0 으로 내린다.
GGML_API void ggml_moe_gpuearly_note_drop(uint64_t n);

// ★결손 **행 수를 늘리지 않는** 무결성 강하(교차 r1 권고 — 전 오류 경로 정직 처분).
//   쓰임: 이벤트 풀 확보 실패처럼 "행은 direct 로 정상 기록되지만 이 런의 측정 전제가 깨진"
//   경우. `note_drop` 을 쓰면 그 행들이 direct 로도 실리고 dropped 로도 세어져 **이중 계상**이
//   되므로, 그런 자리에는 이것을 쓴다(계수는 진단 창구에만 노출·헤더는 `complete=0` 으로 전달).
GGML_API void ggml_moe_gpuearly_note_fault(void);

// ★층 이름 파서의 **유일 구현**(교차 r1 차단① — 대상 결속). `^<prefix><digits>$` 정확 매칭.
//   접미사(` (reshaped)` 등)·비숫자·테이블 폭 초과는 전부 비매칭이다.
//   판독기 정규식(`^ffn_moe_topk-(\d+)$`)과 **같은 계약**이며, 앵커 훅과 consume 대상 결속이
//   이 하나를 공유한다 — 각자 손으로 파싱하면 그 순간 사본 드리프트다.
//   반환 0 = 매칭(`*out_layer` 유효) · <0 = 비매칭.
GGML_API int ggml_moe_gpuearly_parse_layer(const char * name, const char * prefix,
                                           uint32_t * out_layer);

// ---------------------------------------------------------------------------
// ④ 소비자 시임 — CUDA 백엔드가 자기 소비자를 등록하고, split 경계(ggml-backend.cpp A1)가
//    `consume()` 를 부른다. ggml-base 는 CUDA 심볼을 링크하지 않으므로 이 간접이 필요하다.
//    ★consume 은 "GPU 를 방금 동기화한 split 경계"에서만 소비해야 한다 — 그래야 사양 §2 의
//      "`cudaErrorNotReady` = 결손 + fail-close" 가 거짓 격발 없이 성립한다. 그 판별을 위해
//      **방금 동기화한 backend 포인터**(`synced_backend`)를 그대로 넘긴다: 소비자가 자기
//      pending 을 띄운 backend 와 **포인터 동일성**으로 대조한다.
//      ★왜 device type 질의가 아니라 포인터인가: `ggml_backend_dev_type()` 은 CUDA 에서
//      `cudaGetDeviceProperties` 를 **매 호출** 수행한다(ggml-cuda.cu). 그걸 split 경계마다
//      부르면 계측이 스스로 측정을 교란한다 — 바로 §5 A/A 게이트가 잡으려는 그 교란이다.
//      포인터 비교는 정확하고(그 pending 을 띄운 바로 그 backend) 비용이 0 이다.
//
//    ★★**대상 결속 3조건 AND**(교차 r1 차단① — 이것이 없으면 R_UB 가 조용히 축소된다):
//      한 CPU MoE split 은 generic 동기 복사를 **여러 번** 한다(활성화가 먼저, ids 가 나중).
//      backend 포인터만 보면 **첫 복사**(활성화)에서 pending 이 닫혀 `a1_qpc` 가 ids 실체화보다
//      이르게 잡힌다 — M1 실측: topk 26,448행 중 26,160행이 같은 층 `attn_post_norm-<L>
//      (reshaped)` 복사 직후이고 중앙값 **6.7µs** 선행. `H_i = T1 − a1_qpc` 가 그만큼 **커지고**
//      `R_UB = Σ min(Δ, max(0, D−H))` 가 **작아진다**(과소 보고 — 침묵하는 방향이라 더 위험).
//      ⇒ ①`input_name` 이 `^ffn_moe_topk-(\d+)$` 매칭 ∧ ②그 층이 pending graph 메타데이터의
//      앵커 층과 일치 ∧ ③backend 포인터 동일 — **셋 다 참일 때만** 소비한다.
//      그래서 `input_name` 이 ABI 에 있다(§3 `H_i` 정의가 요구하는 "ids 실체화 시각" 결속).
// ---------------------------------------------------------------------------
typedef void (*ggml_moe_gpuearly_consume_fn)(void * user, void * synced_backend,
                                             const char * input_name,
                                             uint64_t sync_return_qpc, uint64_t a1_qpc);

GGML_API void ggml_moe_gpuearly_set_consumer(ggml_moe_gpuearly_consume_fn fn, void * user);
GGML_API void ggml_moe_gpuearly_consume(void * synced_backend, const char * input_name,
                                        uint64_t sync_return_qpc, uint64_t a1_qpc);

// ---------------------------------------------------------------------------
// ⑤ flush — 사이드카 기록(멱등·once 래치). `ggml_moe_direct_finalize()` 가 dispatcher drain 과
//    IO3/PA/shadow/lead-time flush **뒤**에 1회 호출한다.
//    pending 이 열린 채 도달하면 `pending_unconsumed_count` 를 올리고 complete=0 으로 내린다
//    (사양 §3 "pending_unconsumed_count = 0 유효 조건").
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_gpuearly_flush(void);

// ===========================================================================
// 시나리오 teardown — 전역 상태·버퍼 해제 + env 재해소 가능 상태로 되돌린다.
// ★공개 헤더에 남기는 유일한 테스트 진입점(`ggml_moe_direct_reset_for_test()` 가 부른다 —
//   순수 teardown 이며 회계를 어긋내는 시임이 아니다. lead-time/phaseb/shadow 선례와 같은 자리).
GGML_API void ggml_moe_gpuearly_reset_for_test(void);

// ===========================================================================
// ★관측 창구와 **시임**은 이 공개 헤더에 없다 — `ggml/src/ggml-moe-gpuearly-internal.h`
//   (설치·배포 대상이 아닌 내부 헤더). 근거 = lead-time 라운드 교차 r2 R3-4 의 같은 처분.
// ===========================================================================

#ifdef __cplusplus
}
#endif
