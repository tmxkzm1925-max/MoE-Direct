// MoE GPU측 조기 전달 계측(v2) — ★테스트 전용 진입점(내부 헤더).
//
// 사양: bench/techdev/repack_space/PROBE_GPU_EARLY_SPEC_R1.md ★FROZEN v1.0
// 배치 근거: lead-time 라운드 교차 r2 [REVISE] R3-4 — 시임이 공개 API 표면에 노출되면 안 된다.
//
// ★이 헤더는 `ggml/include/` 에 없다(설치·배포 대상 아님). 프로덕션 코드는 include 하지 않으며,
//   소비자는 `moe-direct-selftest` 하나뿐이다(상대경로 include — 같은 트리의
//   `ggml-backend-impl.h` · `ggml-moe-leadtime-internal.h` 접근 관례와 동일).
//
// ★정직 고지(ABI): 심볼은 공유 DLL 구조상 여전히 `ggml-base.dll` 에서 export 된다(selftest 가
//   별 프로세스로 그 DLL 에 링크하므로 export 없이는 링크 불가). 이 격리가 없애는 것은
//   **공개 헤더 표면**이다 — lead-time 내부 헤더의 같은 고지를 승계한다.
//
// ★위험 고지: `..._test_stall_writer` 는 publication 회계(`w_enter`/`w_exit`)를 의도적으로
//   어긋나게 만드는 시임이다. 실 런에서 호출하면 flush 의 drain 이 거짓 실패하고 사이드카가
//   `complete=0` 으로 무효화된다.
#pragma once

#include "ggml-moe-gpuearly.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── 물리층 결산 창구(사양 §3) ────────────────────────────────────────────────
// record_count(=커밋된 행 수) · dropped · complete · capacity · 그리고 분류 카운터 4종.
// ★`eligible` 은 **파생값**이라 창구에 없다 — 판독기와 selftest 가 같은 식으로 산출한다:
//   `eligible = record_count − invalid − excluded_direct − excluded_warmup`.
//   (헤더 필드로 두면 그 자체가 사본이 되어 드리프트 대상이 된다 — 사양 §3 의 "판독기 산출".)
GGML_API void ggml_moe_gpuearly_test_counts(uint64_t * out_rows,
                                            uint64_t * out_dropped,
                                            int32_t  * out_complete,
                                            uint64_t * out_capacity,
                                            uint64_t * out_invalid,
                                            uint64_t * out_excl_direct,
                                            uint64_t * out_excl_warmup,
                                            uint64_t * out_pending_unconsumed);

// 레코드 1건 원바이트 복사(bytes = 64). 성공 0 / 범위 밖·크기 불일치 <0.
GGML_API int ggml_moe_gpuearly_test_row_bytes(uint64_t idx, void * out, uint64_t bytes);

// env 해소 횟수 — OFF 계약의 "fast-gate 1회"를 기계로 확인하는 창구(해소가 반복되면 hot path
// 에서 getenv 가 도는 것이므로 계약 위반이고, 이 수가 그 사실을 드러낸다).
GGML_API uint64_t ggml_moe_gpuearly_test_resolve_count(void);

// 공개 `now_qpc()` 호출 횟수 — OFF 에서 0 이어야 한다(모듈이 스스로 QPC 를 부르지 않는다는 증명).
GGML_API uint64_t ggml_moe_gpuearly_test_qpc_calls(void);

// publication 계약 관측 창구 — `w_enter`/`w_exit` 와 게이트 상태(-1 미해소·0 off·1 on·2 closed).
// flush 뒤에는 `gate==2 ∧ enter==exit` 여야 한다(drain 성립의 기계 증거).
GGML_API void ggml_moe_gpuearly_test_publication(uint64_t * out_enter, uint64_t * out_exit,
                                                 int32_t * out_gate);

// 라우터 경로별 앵커 발화 수 + 상호배타 위반 수(사양 §7 원자⑤ "상호배타 카운터 검증").
GGML_API void ggml_moe_gpuearly_test_anchor_counts(uint64_t * out_argsort, uint64_t * out_fused,
                                                   uint64_t * out_exclusivity_violations);

// `note_fault` 계수 — 행 수를 늘리지 않는 무결성 강하가 몇 번 일어났나(헤더엔 complete=0 만 나간다).
GGML_API uint64_t ggml_moe_gpuearly_test_faults(void);

// 등록된 소비자 조회 — 테스트가 자기 소비자를 끼워 넣기 **전에 저장하고 끝나면 되돌리기** 위한
// 창구다(`reset_for_test` 는 소비자를 지우지 않는 것이 계약이므로, 저장/복원이 없으면 테스트가
// 프로덕션 결선을 조용히 파괴한다).
GGML_API void ggml_moe_gpuearly_test_get_consumer(ggml_moe_gpuearly_consume_fn * out_fn,
                                                  void ** out_user);

// pending 상태 창구 — 열림 여부·현재 boundary_seq·그 pending 이 **아직 안고 있는** 앵커 수
// (부분 소비가 반영된 잔여 — 유실 처분 시 이 값만 `dropped` 로 간다).
GGML_API void ggml_moe_gpuearly_test_pending(int32_t * out_open, uint64_t * out_seq,
                                             uint64_t * out_anchors);

// 층별 서수 테이블의 현재 값(다음에 발급될 서수). 범위 밖 층은 UINT32_MAX.
GGML_API uint32_t ggml_moe_gpuearly_test_occurrence_peek(uint32_t layer);

// ★시임 ① — **커밋되지 않은 예약이 남은 상태**를 결정론적으로 합성한다(`w_enter` 만 n 올린다).
// 되돌리기 = 같은 n 으로 `_release`. lead-time 시임 ① 과 동형.
GGML_API void ggml_moe_gpuearly_test_stall_writer(int32_t n);
GGML_API void ggml_moe_gpuearly_test_release_stalled_writer(int32_t n);

// ★시임 ② — flush 의 **payload 기록 창 안에서** late 커밋을 주입한다. 최종 재검이 그것을 잡아
// 헤더를 `complete=0` 으로 내리는지 검사하는 유일한 수단이다(그 창은 실 경로에서 외부가 건드릴
// 수 없다). 1회 소비 후 자동 해제. lead-time 시임 ③ 과 동형.
GGML_API void ggml_moe_gpuearly_test_arm_late_commit(int32_t n);

#ifdef __cplusplus
}
#endif
