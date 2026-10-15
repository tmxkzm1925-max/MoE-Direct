// MoE lead-time 계측 모듈 — ★테스트 전용 진입점(내부 헤더).
//
// 사양: bench/techdev/repack_space/PROBE_LEADTIME_SPEC_R1.md ★FROZEN v1.0
// 신설 근거: 교차 r2 [REVISE] R3-4 — 시임이 공개 API 표면에 노출되면 안 된다.
//
// ★이 헤더는 `ggml/include/` 에 없다(설치·배포 대상 아님). 프로덕션 코드는 이 헤더를
//   include 하지 않으며, 소비자는 `moe-direct-selftest` 하나뿐이다(상대경로 include —
//   같은 트리의 `ggml-backend-impl.h` 접근 관례와 동일).
//
// ★정직 고지(ABI): 심볼은 공유 DLL 구조상 여전히 `ggml-base.dll` 에서 export 된다.
//   selftest 는 별 프로세스로 그 DLL 에 링크하므로 export 없이는 링크가 성립하지 않는다.
//   ⇒ 이 격리가 없애는 것은 **공개 헤더 표면**이고, ABI 표면까지 없애려면 테스트 전용
//   별도 DLL 빌드가 필요하다(현 빌드 구조 변경 = 이 라운드 범위 밖).
//
// ★위험 고지: 아래 `..._test_stall_writer` / `..._test_stamp_b_delayed` 는 publication
//   회계(`w_enter`/`w_exit`)를 의도적으로 어긋나게 만드는 시임이다. 실 런에서 호출하면
//   flush 의 drain 이 거짓 실패하고 사이드카가 `complete=0` 으로 무효화된다.
#pragma once

#include "ggml-moe-leadtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ★`ggml_moe_leadtime_reset_for_test()` 는 공개 헤더에 남아 있다(순수 teardown · 다른 TU 소비자
//   있음 · phaseb/shadow 선례). 여기 있는 것은 **관측 창구와 시임**뿐이다.

// 버퍼 계수. complete=0 이면 결손(버퍼 상한 초과 · 할당 실패 · drop) → fail-close 대상.
// cap_a/cap_b = **선할당된 칸 수**(헤드룸 감사 — 실 런에서 drop 이 나기 전에 알 수 있다).
GGML_API void ggml_moe_leadtime_test_counts(uint64_t * out_a, uint64_t * out_b,
                                            uint64_t * out_dropped, int32_t * out_complete,
                                            uint64_t * out_cap_a, uint64_t * out_cap_b);

// 레코드 1건 원바이트 복사(bytes = 64 / 32). 성공 0 / 범위 밖·크기 불일치 <0.
GGML_API int ggml_moe_leadtime_test_a_bytes(uint64_t idx, void * out, uint64_t bytes);
GGML_API int ggml_moe_leadtime_test_b_bytes(uint64_t idx, void * out, uint64_t bytes);

// env 해소 횟수 — OFF 계약의 "fast-gate 1회"를 기계로 확인하는 창구다(해소가 반복되면 hot path
// 에서 getenv 가 도는 것이므로 계약 위반이고, 이 수가 그 사실을 드러낸다).
GGML_API uint64_t ggml_moe_leadtime_test_resolve_count(void);

// 공개 `now_qpc()` 호출 횟수 — OFF 에서 0 이어야 한다(모듈이 스스로 QPC 를 부르지 않는다는 증명).
// 내부 t0 채취는 이 카운터를 쓰지 않는 raw 경로라 arm 만으로는 증가하지 않는다.
GGML_API uint64_t ggml_moe_leadtime_test_qpc_calls(void);

// publication 계약 관측 창구 — `w_enter`/`w_exit` 와 게이트 상태(-1 미해소·0 off·1 on·2 closed).
// flush 뒤에는 `gate==2 ∧ enter==exit` 여야 한다(drain 성립의 기계 증거).
GGML_API void ggml_moe_leadtime_test_publication(uint64_t * out_enter, uint64_t * out_exit,
                                                 int32_t * out_gate);

// ★시임 ① — **커밋되지 않은 예약이 남은 상태**를 결정론적으로 합성한다(`w_enter` 만 n 올린다).
// 칸 예약도 본문 쓰기도 하지 않으므로 "drain 이 실패하는가"만 검사한다. 되돌리기 = 같은 n 으로 `_release`.
GGML_API void ggml_moe_leadtime_test_stall_writer(int32_t n);
GGML_API void ggml_moe_leadtime_test_release_stalled_writer(int32_t n);

// ★시임 ② — **진짜 in-flight writer**(교차 r2 R3-2): 실제로 칸을 예약한 뒤 `delay_ms` 동안
// 본문을 쓰지 않고 머문다. 시임 ① 이 못 만드는 상태 — "예약이 공개됐고 그 칸이 아직 쓰이는 중"
// — 를 재현하므로, flush 가 그 칸을 읽지 않는지(비원자 동시 읽기 회피)를 검사할 수 있다.
// 호출측이 **별 스레드**에서 부르고, 본체 스레드가 그 사이에 flush 를 호출하는 형태로 쓴다.
GGML_API void ggml_moe_leadtime_test_stamp_b_delayed(uint64_t qpc, int32_t mlpos, int32_t part_idx,
                                                     int32_t n_ids, int32_t delay_ms);

// ★시임 ③ — flush 의 **payload 기록 창 안에서** late stamp 를 주입한다(교차 r2 R3-1).
// n>0 으로 무장하면 flush 가 drain 성립 확인 후·헤더 확정 전에 stamp 를 n 회 시도한다.
// 최종 재검이 그것을 잡아 헤더를 `complete=0` 으로 내리는지 검사하는 유일한 수단이다
// (그 창은 실 코드 경로에서는 외부에서 건드릴 수 없다). 1회 소비 후 자동 해제.
GGML_API void ggml_moe_leadtime_test_arm_late_stamp(int32_t n);

#ifdef __cplusplus
}
#endif
