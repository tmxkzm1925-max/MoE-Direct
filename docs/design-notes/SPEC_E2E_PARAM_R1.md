> ---
> **[Published design note - as-is working record]** A page from the developer's lab notes,
> published with the original Korean body preserved unchanged. The design and the decisions
> here are the developer's; AI assistants draft and cross-check under his direction.
> "FROZEN" means the document text was locked after multi-round cross-review;
> implementation and measurement status, including deferred items, is stated in each note.
> References to internal files (HANDOFF, SESSION_STATE, reviews/) point to the private
> workspace and are left in place on purpose - they show how the records are kept.
>
> **TL;DR:** Preregistered end-to-end parameter search + A/A calibration protocol: SHA-sealed plans, sentinel runs, a full disposition matrix for every abnormal terminal, and a judge that fail-closes on incomplete evidence.
> ---

# SPEC_E2E_PARAM_R1 — 실측 순서 ⑶ e2e 파라미터 탐색+A/A 캘리브 사양 (v0.12 · 26-08-14 · 리드 Fable · r29~r39 전건 반영)

> ★★**동결 스탬프(리드 기입 26-08-14 11:4x)**: **v0.12 = FROZEN** — r40 **[동결 가·신규
> 0·처분 매트릭스 미정의 칸 0]**(`reviews/codex_e2e_param_r40.md` — 매트릭스 전문 그
> 전사). 이 스탬프 기입 커밋부터 본 파일 무수정 — 개정은 부속 정오로만(correctness 선례).

> **지위**: r20 채택 6단계의 ⑶. 목적 2개 — ①`SPEC_REPACK_V3.md` §5 파라미터의 처분
> (`measured_selected`/`policy_fixed`/`integration_deferred`) ②§9-4 밴드 산식에 넣을
> **A/A noise calibration**. **bin↔virtual 효과값 비교가 아니다**(그건 ⑸⑹ — 그 전까지
> B/A 효과값 열람 금지[r20 Q4]. 이 문서의 스윕은 virtual 팔 내부 파라미터 선택).
> **v0.12 = r29~r39 교차(`reviews/codex_e2e_param_r{29..39}.md`) 전건 수용본**.
> 동결=r40 [ACCEPT] 후 스탬프(기입이 무수정 시작점 — correctness 방식 승계).

## §1. 축 (r29 Q1 확정 — ②part_parallelism·④submit order·worker sweep 보류 유지)

| 축 | 내용 | 처분 형태 |
|---|---|---|
| C0 | qwen122-nonextn virtual·현행 all-parts·coalescing OFF·QD 8 — **A/A matched pair 5쌍** | §9-4 noise 입력·E2 표본 산식 입력 |
| E2 | `io_qd_total` **단일 factor 3수준** {4,8,16} — QD 외 argv/env·prompt·순서·resolved IO 파라미터 전건 동일(header echo 확인) | `measured_selected` 또는 `policy_fixed` |
| E3 | gpt-oss-120b virtual·QD 8 — 성능런(IO3 off)+**별도 프로세스 IO3 진단런 1회**(고유 CREATE_NEW prefix) | ★`io_bounce_copy_qd`=**`policy_fixed(nominal=2, effective=max(nominal, staging_floor))`** — 이 축에서 `measured_selected` 를 주장하지 않는다(비교축 부재). 바운스 비용은 diagnostic 기록 |
| E4 | 생략 — coalescing=`policy_fixed(OFF)` | 처분 즉시 확정 |

## §2. 측정 계약 (r29 Q2 대체 문면)

- **payload 고정**: `cache_prompt=false` · `temperature=0` · `top_k=1` · `seed=7` ·
  **`ignore_eos=true`** · `stream=true` · `return_tokens=true`.
- 부하 세트=`e2e_prompts_r1.json`(SHA 봉인): L1 장문 프리필(`n_predict=32`) · L2 decode
  256 · L3 decode 256. **L1 `prompt_n` 은 "~급" 표기 금지 — plan 에 사전 확인 실값을
  봉인**(r29 실측 참고치: raw 1,618·현 chat-template 렌더 1,627 — plan 생성기가 `/tokenize`
  로 재확인해 봉인).
- **요청별 요구**(불충족=해당 cell 무효): L1/L2/L3 각각 `prompt_n=<봉인 실값>` ·
  `predicted_n={32,256,256}` 정확 일치 · `cache_n=0` · `truncated=false` ·
  `stop_type=limit` · ★**수치 건전성(r32)**: `timings` 의 `prompt_per_second`·
  `predicted_per_second` 가 **양수·유한**(0·음수·NaN/Inf=해당 cell 무효 — 열 교체 규칙
  적용).
- **estimand**: §5 선택 primary = **L3 terminal `log(timings.predicted_per_second)`**.
  L1(process-cold 진단 — SSD/OS cold 아님)·L2(1차 decode)는 진단 병기.
  **§9-4 A/A 입력 = L1 `log(timings.prompt_per_second)` 와 L3
  `log(timings.predicted_per_second)` 를 별도 strata 로 산출·합치지 않는다**(prefill/decode
  밴드 분리 — r20 Q4 계승).
- rep=fresh 프로세스 1회(①② 는 KV 워밍이 아니라 MoE cache·OS·SSD 상태의 고정 priming —
  `cache_prompt=false` 라 KV 재사용 없음).

## §3. 실행 순서·통제 (r29 Q3 + r30 B1·B2·M1 실값 반영)

- ★**E2 실행열(r30 B1 대체 문면)**: 다음 **6열을 하나의 Williams superblock** 으로
  고정한다: `[4,8,16]` `[8,16,4]` `[16,4,8]` `[16,8,4]` `[8,4,16]` `[4,16,8]`.
  `N`(QD당 유효 cell 수)은 **`N = 6×ceil(N_raw/6)`**(§4 산식). 실제 시각 순서가 plan 열과
  다르면 캠페인 무효. C0 의 QD 8 관측치는 E2 효과 추정에 재사용하지 않는다(§4).
- ★**분석 정책 실값(r30 B2 — plan 생성기는 결정권 없음·본문 실값의 복사·검증만.
  빈 값·작성자 선택지=`PlanAbort`)**:
  - **QD16 지위=(B) 확정**: 후보={4,8} · **QD16=veto probe**(선례
    `AMENDMENT_HOP2_ALLOWLIST.md`:342 — 후보 포함 시 veto 발동 불가 확정). veto 규칙:
    QD16 의 mean `log(pps_L3)` 가 선택값 대비 **+log(1.10) 초과**면 선택 무효 →
    `INCONCLUSIVE`(QD 도메인 재설계 재심行).
  - **CI=Welch t 양측 95%**, 비교 통계량 `Δ = mean(log pps_L3 | QD4) − mean(log pps_L3 |
    QD8)`(E2 관측치로 산출).
  - ★**판정 그래프(단일 순서 — r31 B2·이 순서 외 분기 없음)**:
    ① digest·건전성 검사(§3 fail-close) — 불일치=즉시 FAIL·캠페인 중단.
    ② CI half-width > `log(1.05)` → `INCONCLUSIVE` → `policy_fixed(8)`.
    ③ CI 가 0 을 제외 ∧ `|Δ| > log(1.03)` → 승자 `measured_selected`.
    ④ 그 외(유의하지 않거나 `|Δ| ≤ log(1.03)`) → 실질 동률 → `policy_fixed(8)`.
    ⑤ veto probe(최후 평가): QD16 mean `log(pps_L3)` 가 **②③④의 채택값(선택값 또는
    존치 8)** 대비 `+log(1.10)` 초과 → 채택 무효·`INCONCLUSIVE`(QD 도메인 재설계 재심行).
  - **tolerance 의미(명시)**: `log(1.03)` 은 최소 검출 목표가 아니라 **실질 동등 문턱**
    (통계적으로 유의해도 3% 이하 차이는 운영상 동률로 처분). power 목표(§4)는
    `δ=log(1.05)` 에 대한 것이다.
  - ★**CI 정의불가 분기(r32 BLOCKER 3 — ② 앞에 평가)**: 유효 cell 로 Welch CI 를 산출할
    수 없는 상태(양 후보 표본분산 모두 0·자유도 정의불가 등)=`INCONCLUSIVE` →
    `policy_fixed(8)`(④ "그 외"로 흘리지 않는다 — 명시 분기).
  - ★**비정상 종료 후 파라미터 처분 통일(r32 BLOCKER 1 + r33 BLOCKER 1 — 종점 전수)**:
    **아래 ⓐ~ⓓ 비정상 terminal 상태에서는 `io_qd_total` 의 최종 상태가 `policy_fixed(8)`
    존치**다(정상 종료의 `measured_selected` 는 판정 그래프 ③이 그대로 지배 — r34 MINOR)
    (r19 운영 처분표의 현행값 유지 원칙 승계·`integration_deferred` 아님·"폐합 행 미발행"
    없음). 경로별 부가 처분:
    ⓐ **실행 전 차단**(`PlanAbort`·비-virtual/bin/`ab_attempts` 입력 즉시 거부) —
    존치+캠페인 미착수 취급·원인 해소 후 재시도 가능·E3 독립 진행 가능.
    ⓑ **정합성 FAIL**(digest·건전성 불일치) — 존치+**그 슬롯 전체 중단·조사 착수**(E3
    포함·재개=리드 판정. 성능 이전에 정합성 사고다).
    ⓒ **veto·순서 불일치·replacement 소진·CI 정의불가·기타 `INCONCLUSIVE`** — 존치.
    veto 는 추가로 "QD 도메인 재설계 재심"을 장부 등재. ★E3 독립 진행의 범위는
    **E2 착수 이후 발생한 `INCONCLUSIVE`** 에 한정한다(r35) — **C0 5쌍 미달과
    `N_raw > N_max` 는 제외**되며 그 두 경로는 §4 의 "E2/E3 캠페인 미착수" 가 지배한다.
    ★★**E3 슬롯 terminal 전수 분리(r37·r38·r39 일반화)** — **E3 슬롯에서 발생하는 모든
    terminal(`INCONCLUSIVE`[§4 성능 표본 미달·sentinel 관측 불능]·**sentinel
    `NONSTATIONARY`**)은 ⓒⓓ 비대상**이며, **기존 E2 처분(`measured_selected` 포함)을
    변경하지 않고 E3 만 그 상태로 종료**한다(`io_bounce_copy_qd` 의 `policy_fixed` 값도
    불변 — §4 E3 처분 조항이 지배). ★**E3 재착수 상한=총 1회**(어느 terminal 에서든 —
    다음 슬롯·자기 sentinel 세트 동반). 재착수도 terminal 이면 **E3 영구 종료**
    ("diagnostic absent·확인 미완" 정직 기록·값 불변).
    ⓓ **sentinel NONSTATIONARY(★C0/E2 슬롯 한정 — E3 슬롯은 위 전수 분리 조항이 지배)**
    — 존치+그 슬롯 전체 중단(E3 는 다음 슬롯에서 자기 sentinel 세트와 함께 별도 착수 —
    위 재착수 상한 산입).
  - **시작 온도 band ≤60°C · 안전 천장 78°C**(초과 cell 무효) · CPU affinity·power plan·
    PCIe gen/width·engine modules·manifest·prompts·plan SHA 전건 봉인.
  - **외부 I/O(정확식 승계)**: `external_io_ratio = (max(0, physical_read_delta −
    runner_completed_bytes) + physical_write_delta) / runner_completed_bytes` — 동일 cell
    측정창에서 **>0.01 이면 해당 block 무효**.
  - ★**sentinel 실행계약(실값 — r32 BLOCKER 2 + r33 BLOCKER 2·3)**: sentinel = **QD 8·
    부하 세트 동일·fresh process 1 run**(pair 아님). **슬롯 단위 규칙(분할 대응 — ★r34 원자 경계 재정의)**:
    각 실행 슬롯은 자기 sentinel 세트를 가진다. 배치 단위는 cell 이 아니라 **원자**다 —
    원자 = C0 의 matched pair(2 run)·E2 의 3-cell 열·E3 의 개별 run. **S_start·S_end
    필수 + 그 슬롯의 원자 수가 6 초과면 S_mid(⌈원자 수/2⌉번째 원자 **완료 직후** —
    원자 내부 절단 금지)**. 슬롯 분할도 **원자 경계에서만** 허용한다. baseline = **그
    슬롯의 S_start**(슬롯 간 비교 없음 — 슬롯 간 상태 차는 fresh process 설계가 흡수).
    ★**S_mid 위치는 인과적으로 사전 고정(r36 B1)**: 산입 대상은 **plan 이 봉인한 기본
    실행열의 원자만**이다 — 무효 시도·replacement 원자는 **산입하지 않으며**, S_mid 는
    "⌈기본 원자 수/2⌉번째 **기본** 원자 완료 직후"로 봉인 시점에 위치가 확정된다(이후
    교체가 몇 번 일어나도 불변 — 미래 지식 불요).
    예: 슬롯①(N=6)=기본 원자 11(C0 pair 5+E2 열 6) → S_mid 는 6번째 기본 원자(E2 첫 열)
    완료 직후 / N=12=기본 원자 17 → 9번째 완료 직후. E3 슬롯=기본 원자 4(run 4)≤6 →
    S_start·S_end 2개. **E2/E3 효과 통계에
    불포함**(드리프트 감시 전용).
    ★**판정 함수(단일 — U4 실값 승계)**: `d_i = |log(pps_L3,i / pps_L3,S_start)|` —
    배타 구간: `d_i ≤ ln(1.05)`=정상 / `ln(1.05) < d_i ≤ ln(1.10)`=**warning**(기록·
    계속) / `d_i > ln(1.10)`=**NONSTATIONARY**(그 슬롯 무효 — §3 ⓓ 처분). 절댓값이므로
    하락·상승 대칭. sentinel cell 자체가 무효(§3 무효 정의·수치 건전성 포함)면 재실행
    1회, 재무효=**"sentinel 관측 불능" → 그 슬롯 `INCONCLUSIVE`**(NONSTATIONARY 아님).
- **무효·교체(r30 B1)**: 무효 cell 은 그 **3-cell 열 전체 무효** · 교체는 plan 에 미리
  열거한 **동일 열 replacement slot 에서만·최대 2열** · 교체 실패로 superblock 불완전=
  `INCONCLUSIVE`.
- ★**digest fail-close(r30 M1 대체 문면)**: token/content/stop 동일성은 **같은 model·
  manifest·prompt·seed 를 공유하는 새 perf 캠페인의 모든 비교 가능 cell 사이**에서
  검사한다(C0 pair·E2 QD4/8/16 전 반복·E3 성능 반복+IO3 진단 대응 cell). 불일치=즉시
  FAIL. E2↔E3 모델 간 교차 비교와 a1 봉인물 열람은 하지 않는다.
- 실측 슬롯=사용자 터미널에서 **runner 호출**(서버 spawn 은 runner 소유 — fresh process·
  env 봉인 유지)·전면 배타·matched block 중간 삽입 금지.

## §4. 2단계 사전등록·표본 (r29 Q4 대체 문면)

- **Phase C0(캘리브)**: **A/A matched pair 5쌍**(쌍=동일 형상 fresh process 2회 연속 —
  10 run). pair 통계량 `d_i = log(T_{i,2}/T_{i,1})`(strata 별: L1 prompt_pps·L3
  predicted_pps). **σ 는 plug-in 점추정**(상한 아님 — r31 정정): `s_d` = d_i 표본
  표준편차 → per-cell noise `σ = s_d/√2`.
- ★**σ 입력 단일화(r31 B2)**: **N·ceiling 산정에는 L3(decode) strata 의 σ 만** 투입한다
  (E2 primary estimand 가 L3 이므로). L1 strata 의 σ·d_i 는 §9-4 prefill 밴드 입력
  전용이다.
- ★**C0 무효 pair closure(r31 BLOCKER)**: pair 중 1 cell 이라도 무효(§3 무효 정의 준용)
  =**pair 전체 제외**. 교체는 plan 에 미리 열거한 **replacement pair 최대 2쌍**에서만.
  최종 5쌍 미달=**C0 `INCONCLUSIVE`** → 전 pending 축 `policy_fixed` 존치·E2/E3 캠페인
  미착수.
- ★**N 산식(실값)**: `N_raw = ceil( 15.68 × σ² / δ² )`(양측 α=0.05·power 80%·z 근사
  2(1.96+0.84)²=15.68) · 탐지 목표 `δ = log(1.05)` · **N = 6×ceil(N_raw/6)** · 최소
  **N=6** · **N_max=12**. ★**noise ceiling(r31 모순 해소 — 단일 조건)**: 별도 σ 임계를 두지 않는다 — **운용
  게이트는 `N_raw > N_max(=12)` 하나**다(성립 시 E2 착수 전 `INCONCLUSIVE` → 전 pending
  축 `policy_fixed` 존치·캠페인 미착수). 참고 유도: 이 게이트는 `σ ≲ δ×√(N_max/15.68)
  ≈ 0.0427` 과 동치이나 판정은 N_raw 정수 비교로만 한다(경계 부동소수 이중 임계 제거 —
  리드 검산 26-08-14). 이 정의라 N∈{6,12} 적응 설계가 살아 있다.
- C0 종료 후 **A/A 값만** 산식에 넣어 최종 N 과 E2/E3 plan SHA 를 **새로 봉인**한다(효과값
  을 보기 전에). ★**그 시점에 최종 N 으로 ETA 를 재산출한다**(r30 m1). C0 관측치는 E2
  효과 추정에 재사용하지 않는다.
- decode/prefill 밴드는 **각각** 계산한다(§2 strata).
- E3=성능 3 rep+IO3 진단 1(`policy_fixed` 확인·diagnostic 용도 — §1).
  ★**E3 무효 run 처분(r36 B2 — 실값)**: E3 의 원자=개별 run 이므로 열 교체 규칙 대신 —
  ⓐ성능 run 무효(§2 요청 요구·수치 건전성·온도·외부 I/O 위반)=**동일 형상 재실행 최대
  1회/run**(plan 에 예비 3 run 열거). 재무효 또는 최종 유효 성능 run<3 → **E3
  `INCONCLUSIVE`**(diagnostic 미확정 — `io_bounce_copy_qd` 의 `policy_fixed` 값 자체는
  E3 결과와 무관하게 불변[§1 처분 형태]) ⓑIO3 진단 run 무효 또는 판독기 완전성 실패=
  재실행 최대 1회·재실패=**"diagnostic absent" 정직 기록 후 E3 종료**(유효 성능 run≥3
  이면 E3 는 완료로 닫힌다 — IO3 는 부가 진단).
- **ETA(sentinel 포함 — r33 MINOR 정정)**: ⑵ a1 실측(팔당 ≈3분) 역산 보수 밴드 run 당
  5~10분. 슬롯①=C0 10 + E2 18(N=6) + sentinel 3 = **31 run ≈2시간35분~5시간10분**
  (**C0 후 최종 N 확정 시 재산출·슬롯 분할 가능** — 분할 시 각 슬롯이 자기 sentinel
  세트를 추가 부담) · 슬롯②=E3 4 + sentinel 2 = **6 run ≈0.5~1h**. **첫 rep 후 ETA
  재확정은 슬롯 분할·예약 시간에만 영향** — 표본·요청·판정식은 변경하지 않는다
  (outcome-driven 변경 금지).

## §5. 러너·경계 (r29 Q5 대체 문면)

- **correctness 자산(`SPEC_AB_CORRECTNESS_R1.md`·`ab_runner.py`·`ab_attempts/`)은 본
  단계에서 수정·재사용하지 않는다**(FROZEN identity 보존). 신규 **`e2e_perf_runner.py`** 와
  **`e2e_attempts/`** 를 만들고, correctness 에서 검증된 동작 계약(HTTP framing·ready·
  graceful shutdown·env sanitize 방식)만 **계약으로** 재사용한다(코드 공유 아님 — 독립
  runner identity).
- perf plan 은 독립 protocol/runner/extractor SHA·engine modules·manifest·prompts·argv/
  env·순서·N·분석식을 봉인한다. **⑶에서 `mode=virtual` 만 허용** — bin 아티팩트 또는
  `ab_attempts/` 입력은 즉시 거부(fail-close). a1 의 봉인 성능값은 읽지 않는다.
- 성능런=IO3 off. **진단런=별도 프로세스·IO3 on·고유 CREATE_NEW prefix** — 값은 **전용
  판독기(`io3_read.py` — 신규 시공·`SPEC_IO_METRICS_V3` §5 계약 준수: `complete`·dropped·
  digest·join/cap 검증)** 의 완전성 검사를 통과한 것만 diagnostic 으로 기록.
- ★**E3 env 함정**: `MOE_DIRECT_IO_BOUNCE_COPY_QD` 를 **명시하지 않는다**(명시 2=shape
  floor 미달로 기동 거부) — 키 부재로 두고 header `io_params.io_bounce_copy_qd={value:14,
  source:"default"}` 를 요구한다.

## §6. 개정 이력

v0.1(26-08-14 새벽·미확정 3 명시형) → 미확정 해소(a1 역산) → v0.2(r29 전건 반영) →
**v0.3(r30 B1·B2·M1·m1 반영 — Williams superblock 6열·분석 정책 전건 실값화[QD16=(B)
probe 확정·CI Welch t 95%·tolerance 3%·veto 10%·δ=log(1.05)·N=6×ceil(N_raw/6)·N_max=12·
noise ceiling 0.0244·온도 band ≤60°C·외부 I/O 정확식]·digest 범위=캠페인 전 비교 가능
cell·C0 후 ETA 재산출) → **v0.4(r31 반영 — 판정 그래프 단일 순서 ①~⑤·tolerance=실질
동등 문턱 명시·σ=plug-in 점추정 명칭 정정·σ 입력=L3 단일화·C0 무효 pair closure[pair
제외·교체 2쌍·미달=INCONCLUSIVE]·ceiling=N_raw>N_max 단일 조건[리드 검산으로 경계
모순 제거]) → **v0.5(r32 반영 — 비정상 종료 전 경로 `policy_fixed(8)` 통일+E3 독립·
sentinel 실행계약 실값[형상·배치·통계 불포함·관측 불능 처분·ETA 포함]·수치 건전성
cell 무효·CI 정의불가 명시 분기·지위 문구 정합) → **v0.6(r33 반영 — terminal 종점
전수 폐합[실행 전 차단/정합성 FAIL/INCONCLUSIVE 계열/NONSTATIONARY 4분류·전부
policy_fixed(8) 존치+부가 처분]·sentinel 판정 함수 단일화[절댓값·배타 구간·U4 실값
승계]·슬롯 단위 sentinel 세트[분할·E3 대응]·ETA sentinel 포함 재산)** → **v0.7(r34 반영 — sentinel·분할 경계=원자 단위[C0 pair·
E2 3-cell 열·원자 내부 절단 금지·수기 대입 예시 병기]·비정상 처분 범위 문구 ⓐ~ⓓ 한정)** → **v0.8(r35 —
ⓒ E3 독립 진행 범위를 "E2 착수 이후 INCONCLUSIVE"로 한정·C0 미달과 N_raw>N_max 는 §4
미착수 지배 명시)** → **v0.9(r36 — S_mid=기본 원자만 산입·봉인 시점 위치 고정[인과성]·
E3 무효 run 처분 실값[재실행 1회/예비 3·유효<3=INCONCLUSIVE·IO3 재실패=diagnostic absent
정직 기록])** → **v0.10(r37 — ⓒ 적용 범위=E2 측 INCONCLUSIVE 한정·E3 자체 INCONCLUSIVE
는 E2 처분 불변경·E3 종료만[§4 지배])** → **v0.11(r38 — 그 조항을 E3 슬롯의 모든
INCONCLUSIVE[sentinel 관측 불능 포함]로 일반화)** → **v0.12(r39 — E3 슬롯 terminal
전수 분리[INCONCLUSIVE+NONSTATIONARY 전부 ⓒⓓ 비대상·E2 처분 불변]·E3 재착수 상한=총
1회·재차 terminal=영구 종료·ⓓ=C0/E2 슬롯 한정 명시)**. 동결=r40 [ACCEPT] 후 스탬프.
