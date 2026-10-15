// MoE lead-time 계측 모듈 — 사양: bench/techdev/repack_space/PROBE_LEADTIME_SPEC_R1.md ★FROZEN v1.0
//
// 이 모듈은 "관측만" 한다. 캐시·플래너·PhaseA/PhaseB·IO3 의 어떤 상태도 읽거나 바꾸지 않으며,
// 힌트 소비(프리페치 개입) 배선은 존재하지 않는다. 기본 OFF 에서 기존 거동은 완전 불변이다.
//
// 재는 것(사양 §1 — "host-visible exact-id lead"):
//   T0 = ids 의 host 실체화 완료      → stamp A (2지점: A1=decode generic 동기 복사 직후 /
//                                       A2=offload split 의 explicit ids D2H synchronize 직후)
//   T1 = 그 층 첫 acquire 진입(pre-lock) → stamp B
//   T2/T3(첫 child submit · burst 마지막 publish)는 **기존 IO3** 가 이미 낸다 — 이 모듈은
//   T0·T1 두 축만 신설하고, 판독기가 IO3 와 boot_id·QPC 도메인으로 조인한다(사양 §2 join 계약).
//
// 왜 PhaseA 를 안 쓰는가(사양 §2 · COE 사실 4-ⓐ): PA timing 도 `phasea_on` 을 세우고,
// `disp_can_plan_demand` 가 `phasea_on ∧ demand_outstanding>0` 이면 계획을 지연시킨다 —
// 관측이 거동을 바꾼다. 이 모듈은 PA 와 완전히 독립된 게이트를 쓰고 PA 코드·의미론을
// 한 줄도 건드리지 않는다.
//
// ★★hot-path 계약(사양 §2 — 위반은 이 프로브의 존재 이유를 무너뜨린다)
//   ① 버퍼는 **arm 시점(=seal)에 전량 선할당**된 고정 폭 POD 링이다. A/B 는 **분리 버퍼**이며
//      각자 원자 인덱스 예약(fetch_add)으로 칸을 잡는다 — mutex 교차 경합이 없다.
//   ② hot hook 안에서 **문자열 포맷·파일 I/O·동적 할당·getenv·락 획득이 전부 0** 이다.
//      그래서 env 해소는 hook 이 아니라 `ggml_moe_leadtime_note_seal()` 만 한다(아래 ★).
//   ③ OFF 계약 = "QPC·registry·lock·allocation 0회 + 예측 가능한 fast-gate(캐시 bool) 1회".
//      문자적 제로가 아니라 **검증 가능형**이다: `ggml_moe_leadtime_enabled()` 는 해소 여부와
//      무관하게 원자 로드 1회로 끝나며(미해소 = false), 그 뒤 훅은 실행문 0 이다.
//      ★훅은 `enabled()` 가 참일 때만 `stamp_*` 를 부르므로, 아래 publication 카운터의 비용은
//      **게이트가 켜진 런에만** 든다(OFF 런의 훅 비용은 원자 로드 1회 그대로다).
//
// ★★★publication 계약(교차 r1 BLOCKER 폐합 — **불변식: torn/late 행은 `complete=1` 파일에
//      실릴 수 없다**)
//   무엇이 문제였나: 칸 예약(`next.fetch_add`)이 본문 memcpy **보다 먼저** 공개되므로, 그 사이에
//   flush 가 `next` 를 record_count 로 삼으면 **아직 안 쓰인 칸**이 유효한 행으로 파일에 실린다.
//   게다가 flush 가 게이트를 닫지 않으면 **flush 이후 도착한 스탬프**가 버퍼에만 남고
//   `complete` 도 안 내려간다(요구된 관측이 조용히 사라진다).
//   폐합 기제 — **진입/완료 카운터 쌍**:
//     · 스탬프는 게이트 확인 **이전에** `w_enter` 를 올린다(순서가 계약이다 — "게이트를 통과했지만
//       아직 예약하지 않은" 스레드가 drain 창 밖으로 빠져나가지 못하게 한다).
//     · 본문 memcpy 를 마친 뒤 `w_exit` 를 **release** 로 올린다(스코프 소멸자 — 전 반환 경로).
//     · flush 는 ①게이트를 **먼저 닫고**(신규 스탬프 차단) ②`w_enter == w_exit` 까지 대기
//       (acquire — 이 시점에 기록된 전 본문이 flush 스레드에 가시적이다) ③그 뒤에야 record_count
//       를 확정한다. 미완 예약이 남은 채 대기가 끝나면 **그 수만큼 dropped 계상 + `complete=0`**.
//   ⇒ drain 이 성립한 파일의 모든 행은 **커밋 완료 행**이고, drain 이 실패한 파일은 `complete=0`
//      이라 판독기가 fail-close 한다. 게이트 폐쇄 후 도착한 스탬프도 dropped + `complete=0` 로
//      정직 처분된다(실 런타임에서 finalize 이후 스탬프는 도달 불가하지만 fail-loud 로 둔다).
//
// ★**env 해소 지점이 계약의 일부다**: `ggml_moe_leadtime_enabled()` 는 **절대 해소하지 않는다**
//   (미해소 상태를 그냥 false 로 돌려준다). 해소·선할당은 `ggml_moe_leadtime_note_seal()` 과
//   테스트 전용 리셋만 수행한다. 이렇게 두면 "첫 hook 진입이 VirtualAlloc 을 유발"하는 경로가
//   **구조적으로 존재할 수 없다** — 사양의 "hot hook 내 동적 할당 금지"를 코드 형상으로 잠근다.
//   대가: virtual seal 에 도달하지 않은 런은 env 를 켜도 무장되지 않는다(= 기록 0). 그 침묵을
//   막기 위해 `ggml_moe_leadtime_flush()` 가 미해소 상태에서 진단 1줄을 남긴다(§flush 주석).
//
// Windows 전용: 비Windows 빌드에서는 전부 no-op 스텁으로 컴파일된다(Phase A·shadow 관례 동일).
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// wire 상수 (.cpp 에서 static_assert 로 폭·오프셋을 동결한다)
// ---------------------------------------------------------------------------
#define GGML_MOE_LEADTIME_KIND_A      1     // `<prefix>_lead_a.lt` — T0 스탬프
#define GGML_MOE_LEADTIME_KIND_B      2     // `<prefix>_lead_b.lt` — T1 스탬프
#define GGML_MOE_LEADTIME_A_BYTES     64
#define GGML_MOE_LEADTIME_B_BYTES     32
#define GGML_MOE_LEADTIME_HDR_BYTES   128
#define GGML_MOE_LEADTIME_NAME_BYTES  48    // 사양 §2 "tensor_name[<=48B 정확 복사]"

// stamp A 채취 지점(wire 에 u8). 판독기는 A1=decode 경로 / A2=prefill 경로로 읽는다.
#define GGML_MOE_LEADTIME_SITE_A1     1     // ggml-backend.cpp generic 동기 복사 직후
#define GGML_MOE_LEADTIME_SITE_A2     2     // ggml-backend.cpp offload split explicit ids D2H 직후

// stamp A flags(u8)
#define GGML_MOE_LEADTIME_A_FLAG_NAME_TRUNCATED 0x01  // 원 이름 길이 > 48B(앞 48B 만 실렸다)
#define GGML_MOE_LEADTIME_A_FLAG_NAME_NULL      0x02  // 이름 포인터가 NULL(있을 수 없는 입력의 정직 표시)

// stamp B flags(u8)
#define GGML_MOE_LEADTIME_B_FLAG_PREFILL        0x01  // n_tokens_hint > 1 (= Phase::PREFILL)

// ---------------------------------------------------------------------------
// ① 게이트 (env: `MOE_DIRECT_LEADTIME=1` · 그 외 = off + 경고 1회 · 미설정 = 조용한 off)
//    `MOE_DIRECT_LEADTIME_OUT`     미설정 = 카운터만(파일 미생성 — LMTRACE 관례)
//    `MOE_DIRECT_LEADTIME_MAX_MB`  기본 64 (A 3/4 · B 1/4 로 분할 선할당)
// ---------------------------------------------------------------------------

// ★hot-path 게이트. 원자 로드 1회 + 비교 1회. **해소하지 않는다**(미해소 = false).
//   ★flush 가 게이트를 닫은 뒤에도 false 다(publication 계약 ① — 폐쇄 = 신규 스탬프 차단).
GGML_API bool ggml_moe_leadtime_enabled(void);

// 관측 시계 — 이 프로브의 모든 QPC 는 여기서 나온다(raw QueryPerformanceCounter · 게이트 무관).
// 호출측은 `enabled()` 가 참일 때만 부른다(OFF 계약의 "QPC 0회"가 그렇게 성립한다).
GGML_API uint64_t ggml_moe_leadtime_now_qpc(void);

// 조인 키. IO3 가 무장돼 있으면 `note_seal` 이 그 boot_id 를 그대로 채택한다(= 같은 값).
// 미무장이면 0. ★flush 로 게이트가 닫힌 뒤에도 값이 유지된다(사후 대조용 — 게이트 상태가 아니라
// "이 런이 무장됐었나"를 뜻하는 값이다).
GGML_API uint64_t ggml_moe_leadtime_boot_id(void);

// ---------------------------------------------------------------------------
// ② arm — env 해소 + 버퍼 전량 선할당 + 조인 키 채택. seal 경로가 1회 부른다.
//    ★IO3 의 boot_id 해소 **뒤**에 불러야 조인 키를 그대로 채택할 수 있다.
//      join_boot_id == 0 이면 자체 생성값을 쓰며, 그 경우 IO3 조인은 성립하지 않는다
//      (판독기 fail-close — 의도된 침묵 금지이므로 경고 1줄을 남긴다).
//    engine_mode: 1=bin 2=virtual (헤더 echo · 진단용) · n_expert/n_moe_layers 도 echo.
//    멱등: 두 번째 호출은 계수만 로그하고 상태를 바꾸지 않는다.
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_leadtime_note_seal(uint64_t join_boot_id, uint32_t engine_mode,
                                          uint32_t n_expert, uint32_t n_moe_layers);

// ---------------------------------------------------------------------------
// ③ hot hook — 게이트가 참일 때만 호출된다. 락·할당·포맷 0.
// ---------------------------------------------------------------------------

// stamp A: ids 의 host 실체화 시각. site = GGML_MOE_LEADTIME_SITE_A1|A2.
// tensor_name 은 호출 시점에 **정확 복사**된다(<=48B · 초과분은 절단 + 플래그).
// 판독기는 `^ffn_moe_topk-(\d+)$` 정확 매칭으로 대상 층을 고르고, 비대상 이름 행은 배제한다
// (사양 §2 NIT — 엔진은 필터하지 않는다. 필터를 엔진에 두면 이름 규칙 변경이 조용한 결손이 된다).
GGML_API void ggml_moe_leadtime_stamp_a(int32_t site, uint64_t qpc, const char * tensor_name);

// stamp B: 그 층 acquire 의 pre-lock 진입 시각 + registry 해소값.
// qpc 는 **호출측이 진입부에서 미리 채취한 값**을 그대로 싣는다(락 대기·registry 조회 시간이
// T1 에 섞이지 않게 하는 것이 사양 §1 의 "pre-lock" 의미다).
GGML_API void ggml_moe_leadtime_stamp_b(uint64_t qpc, int32_t mlpos, int32_t part_idx,
                                        int32_t n_ids, int32_t is_prefill);

// ---------------------------------------------------------------------------
// ④ flush — 사이드카 기록(멱등·once 래치). `ggml_moe_direct_finalize()` 가 dispatcher drain 과
//    IO3 flush **뒤**에 1회 호출한다(같은 boot_id 를 쓰므로 IO3 가 먼저 닫혀 있어야 한다).
//    drop 이 한 건이라도 있으면 헤더 `complete=0` 으로 정직하게 강하한다(사양 §4b: lead 지표는
//    분포 자체가 산출물이라 결손 비허용 — 판독기가 fail-close 한다).
//    ★미해소(=무장 실패) 상태에서는 여기서만 env 를 진단 목적으로 1회 들여다보고, `=1` 이었으면
//      "켰는데 무장되지 않았다"를 1줄 남긴다. finalize 는 hot path 가 아니므로 계약 위반이 아니다.
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_leadtime_flush(void);

// ===========================================================================
// 시나리오 teardown — 전역 상태·버퍼 해제 + env 재해소 가능 상태로 되돌린다.
// ★이것만 공개 헤더에 남긴다: `ggml_moe_direct_reset_for_test()`(다른 TU)가 부르고, 회계를
//   어긋나게 만드는 시임이 아니라 순수 teardown 이며, `ggml_moe_phaseb_reset_for_test` ·
//   `ggml_moe_shadow_reset_for_test` 가 이미 같은 자리에 있는 선례를 따른다.
GGML_API void ggml_moe_leadtime_reset_for_test(void);

// ===========================================================================
// ★관측 창구와 **시임**은 **이 공개 헤더에 없다**(교차 r2 R3-4 폐합).
//   위치 = `ggml/src/ggml-moe-leadtime-internal.h`(설치·배포 대상이 아닌 내부 헤더).
//   이유: `test_stall_writer` 류는 publication 회계를 임의로 어긋나게 만들 수 있는 시임이라
//   **생산 API 표면에 광고되면 안 된다**. selftest 는 그 내부 헤더를 상대경로로 include 해
//   접근한다(같은 트리의 `ggml-backend-impl.h` 접근 관례와 동일).
//   ★정직 고지: 심볼 자체는 공유 DLL 구조상 여전히 export 된다(selftest 가 별 프로세스로
//   `ggml-base.dll` 에 링크하므로 export 없이는 링크가 불가능하다). 이 수리가 없애는 것은
//   **공개 헤더 표면**이며, ABI 표면까지 없애려면 테스트 전용 별도 DLL 빌드가 필요하다.
// ===========================================================================

#ifdef __cplusplus
}
#endif
