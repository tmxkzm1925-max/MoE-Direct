> ---
> **[Published design note - as-is working record]** A page from the developer's lab notes,
> published with the original Korean body preserved unchanged. The design and the decisions
> here are the developer's; AI assistants draft and cross-check under his direction.
> "FROZEN" means the document text was locked after multi-round cross-review;
> implementation and measurement status, including deferred items, is stated in each note.
> References to internal files (HANDOFF, SESSION_STATE, reviews/) point to the private
> workspace and are left in place on purpose - they show how the records are kept.
>
> **TL;DR:** Frozen erratum 2 to the e2e protocol: the INTERRUPTED terminal (a slot that started but did not finish is never confused with one that never started), terminal-to-phase attribution rules, and binding the erratum into the effective protocol SHA.
> ---

# SPEC_E2E_PARAM_R1 부속 정오 2 — 중단(INTERRUPTED) 전용 종점 신설 (v3 **FROZEN** · 26-08-14 · 리드 Fable · 사용자 승인 26-08-14 15:1x)

> **지위**: `SPEC_E2E_PARAM_R1.md` v0.12(FROZEN)의 부속 정오. 본체·부속 정오 1 v2 는 무변경.
> **동결 상태**: ★**v3 FROZEN(26-08-14 19:0x — r47 교차 [ACCEPT]·전사=
> `reviews/codex_e2e_runner_r47.md`)**. 이후 문면 변경=재심 경유.
> **발단**: r44 심의(전사=`reviews/codex_e2e_runner_r44.md`) — 중단된 슬롯을 §3 ⓐ행
> (`PLAN_ABORT`=실행 전 차단·미착수)으로 재사용하면 공식 판독기가 "E3 not started" 로
> 출력하는 실 소비자 혼동 경로 확정. 리드·Codex 판정 일치 + 사용자 승인.
> **v1→v2**: r45(전사=`reviews/codex_e2e_runner_r45.md`)의 동결 불가 판정 3건 폐합 —
> ①슬롯① terminal→phase 귀속 규칙(§1-2 신설) ②E2 중단 최종 verdict 확정(§1-3)
> ③effective protocol SHA 결속(§5 신설). C0 지배 우선순위(§1-2)도 r45 BLOCKER 1 의 계약
> 근거로 명문화.
> **v2→v3**: r46(전사=`reviews/codex_e2e_runner_r46.md`)의 유일 동결 공백 폐합 —
> `SENTINEL_UNOBSERVABLE` 귀속을 §1-2 에 명문화(예외적 slot-global — r46 심의가 "건전한
> 보수"로 지지한 현행 코드 선택을 사양이 유일하게 도출하도록) + 재분석 병기값 순수성 1문.

## 1. §3 "비정상 종료 후 파라미터 처분 통일" ⓐ~ⓓ 에 ⓔ 신설

### 1-1 정의

**ⓔ 중단(terminal `INTERRUPTED`)** — 슬롯의 봉인 실행열이 **착수된 뒤**(해당 슬롯의
**현행 인스턴스**에 cell 이 존재) 완주하지 못했고, 그 **인스턴스·phase 에 귀속된** terminal
도 기록되지 않은 상태를 judge 가 적발한 경우. 그 phase 의 terminal 은 `INTERRUPTED` 다.

- **처분(ⓐ 준용)**: `io_qd_total` 최종 상태 `policy_fixed(8)` 존치 · 미완 · 원인 해소 후
  재시도 가능 · 이미 기록된 cell 은 이력으로 보존(무효화하지 않음 — 소비 여부는 기존
  `run_key` 회계가 지배).
- **ⓐ와의 구별(이 정오의 존재 이유)**: ⓐ=**미착수**(cell 0·실행 전 차단) / ⓔ=**착수 후
  미완**(cell 존재·마무리 증거 누락). 어휘를 분리할 뿐 파라미터 처분은 동일하다.

### 1-2 슬롯① terminal→phase 귀속 규칙 (v2 신설 — r45 문면 결함 폐합)

슬롯①은 C0 phase 와 E2 phase 를 **순차 공급**한다. 슬롯 상태 레코드는 `phase_context`
(와 인스턴스)를 기록하며, 귀속은 다음과 같다:

- **phase-local terminal**(ⓐ `PLAN_ABORT` · ⓒ `INCONCLUSIVE` · ⓔ `INTERRUPTED`):
  **그 레코드의 `phase_context` 가 가리키는 phase 만** 지배한다. 이미 정상 완료된 다른
  phase 의 기록을 **소급 변경하지 않는다**(예: C0 정상 완료 → plan-main 봉인 → E2 패스
  gate abort 의 경우, `PLAN_ABORT` 는 E2 에만 귀속되고 C0 는 `NORMAL` 을 유지한다).
- **slot-global terminal**(ⓑ 정합성 FAIL · ⓓ sentinel `NONSTATIONARY` · sentinel
  `UNOBSERVABLE`): 기존 §3 ⓑⓓ 처분 그대로 — 그 슬롯 전체에 미친다(ⓑ=슬롯 전체 중단·조사
  착수 / ⓓ=그 슬롯 전체 중단).
  ★**`SENTINEL_UNOBSERVABLE` 의 예외 귀속(v3 명문 — r46 심의 반영)**: 관측 불능의
  **처분은 ⓒ(`INCONCLUSIVE` 계열·존치)를 따르지만 귀속은 예외적으로 slot-global** 이다 —
  sentinel 은 phase 별 성능이 아니라 S_start 기준 **슬롯 전체의 정상성**을 감시하므로,
  허용 재실행 후에도 관측 불능이면 그 슬롯의 증거 전체가 불완전하다(본체 §3 "그 슬롯
  `INCONCLUSIVE`" 문면 승계).
- **지배 우선순위(r45 BLOCKER 1 폐합)**: phase 에 귀속된 ⓐ~ⓔ terminal 은 judge 의
  **재분석 결과보다 우선**한다 — 재분석(5쌍 미달 `INCONCLUSIVE` 등)은 검증 대조이지 종점
  결정이 아니다. 귀속 terminal 이 있으면 그것이 phase terminal 이고, 재분석 결과는 근거
  필드로만 병기한다. ★**병기값 순수성(v3)**: 병기되는 재분석 값은 **지배 적용 전의 순수
  재분석 결과**여야 한다(지배 terminal 로 먼저 덮어쓴 값을 병기하는 것은 계약 위반).

### 1-3 최종 verdict 매핑 (v2 확정 — r45 문면 결함 폐합)

- **E3 중단** = 최종 verdict `PARTIAL`(ⓐ 미착수와 동일 매핑·문면만 분리).
- **E2 중단** = 최종 verdict **`INCONCLUSIVE`**(E2 가 정상 종점에 도달하지 못한 모든 경우와
  동일 매핑 — 본체 H8 원칙 "COMPLETE 는 E2 정상 종점 도달 시에만" 승계).
- **C0 중단** = E2/E3 는 캠페인 미착수 규칙(§4)이 지배하며 최종 verdict 는 `INCONCLUSIVE`.
- 어느 경우든 중단 상태에서 `COMPLETE` 로 떨어질 수 없다.

## 2. 재개·재착수와의 관계

- `INTERRUPTED` 는 **미재개 상태에 대한 judge 시점의 판독 결과**이지 영구 종점이 아니다 —
  중단 슬롯은 기존 재개 기제(소비된 `run_key` 스킵)로 이어 달릴 수 있고, 완주하면 judge 는
  정상 판정한다.
- **E3 전체 재착수 상한(총 1회·§3)은 그대로 지배한다** — 중단이 상한을 우회하는 경로가
  되어서는 안 된다(구현·검증 대상).

## 3. 판독기(공식 .md RESULT 소비 표면) 계약

- `INTERRUPTED` 는 미착수("not started")와 **구별되는 문면**으로 표기한다 — 중단임을 명시
  하고 **누락 증거 목록**(missing sentinel/atom)을 함께 출력한다.
- 중단 슬롯에 대해 "not started" 류 문면을 출력하는 것은 계약 위반이다. **C0 중단도 동일
  적용**(중단 절 출력 — v2 명시).

## 4. 적용 범위

- 이 정오는 judge 의 슬롯 완결성 판독(r43 신설·r44~r45 정밀화)과 그 어휘·귀속에만 관여한다.
  §3 ⓐ~ⓓ 의 기존 정의·처분·E3 terminal 전수 분리 조항은 무변경.

## 5. 프로토콜 결속 (v2 신설 — r45 BLOCKER 2 폐합)

- 이 정오는 러너의 **적용 정오 목록(APPLIED_ERRATA)에 등재**되어 **effective protocol SHA
  산입 대상**이다(정오 1 의 결속 계약 승계 — 후속 정오는 effective SHA 로 신규 캠페인을
  구성한다).
- 결속 시점 전에 발행된 e2e 캠페인 plan 은 존재하지 않는다(첫 캠페인 착수 전 동결) —
  소급 재봉인 대상 없음.
