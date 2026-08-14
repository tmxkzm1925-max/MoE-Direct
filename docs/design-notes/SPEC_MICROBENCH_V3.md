> ---
> **[Published design note - as-is working record]** A page from the developer's lab notes,
> published with the original Korean body preserved unchanged. The design and the decisions
> here are the developer's; AI assistants draft and cross-check under his direction.
> "FROZEN" means the document text was locked after multi-round cross-review;
> implementation and measurement status, including deferred items, is stated in each note.
> References to internal files (HANDOFF, SESSION_STATE, reviews/) point to the private
> workspace and are left in place on purpose - they show how the records are kept.
>
> **TL;DR:** Runner spec for a preregistered 1,059-cell device-ceiling decomposition campaign, including a 90-coordinate Stage A1 grid, that separates what the NVMe device can do from what the software stack loses (queueing, submission width, dispatcher serialization) before any tuning decision is allowed.
> ---

# SPEC_MICROBENCH_V3 — 장치 천장 분해 마이크로벤치 러너 사양 (**동결판 v1.0** · 26-08-07 · 작성 Fable)

> **지위: 동결**(Codex r4 **[ACCEPT — freeze-ready]** — r3 3건 [CLOSED]·신규 결함 0·좌표
> 총람 62/186 독립 재산출 일치. 심의 계보: r1 [MODIFY 4·5·2] → r2 [MODIFY 2·2] → r3
> [MODIFY 3] → r4 [ACCEPT] — 전 라운드 전건 수용·전사=`reviews/codex_microbench_v3_{r1,r2,
> r3,r4}.md`. 사용자 수렴 26-08-06: 셀 시간 규칙=Q5 원안·C: 코퍼스 64GiB — planner ETA>5h면
> 재수렴). **이후 변경은 신규 심의로만.**
> ★**부속 정오 1(26-08-07 — 심의=코드 교차 r1 `reviews/codex_ubench_impl_r1.md`·확인=수리
> 재교차 r2)**: ①§3-4 Stage C 좌표 총람 규범 신설(16 unique/48 executions+A/A 1 — 완전
> 본 런=705 executions·시간 하한 7,050s) ②§4-3 corpus topology 규범(2×32GiB 파일·D→C
> source 전단사) ③§3-1 A1 대형 블록 라벨의 규범 바이트 명시 ④§6-1 commit headroom 하한
> 8GiB 승인.
> ★**부속 정오 2(26-08-07 — 심의=수리 재교차 r2 `reviews/codex_ubench_impl_r2.md`·사용자
> 수렴[cap]·확인=r3)**: ⓐ§4-2 **trace 실행 cap 규범**(봉인 모집단 131,072 보존·실행용
> 결정론 표본 cap **main 11,912[≈64GiB/pass]·quick 1,485[≈8GiB]**·전역 stratified window
> 추출·paired 양팔 동일 실행 표본 digest — 구 whole-pass 전량 소비는 본 런 ~71.6TiB로
> §8 임계 충돌) ⓑ§10-6 **r5 원천 정정**(구 후보 events_decomp 합산 공식은 diag 실측
> 6.85GB/s로 4.98과 불일치 — 실원천=`bench_results/decode_hunt/events_seal.json:98-120`
> 계열 seal·**per-token p50 공식**·raw 대응 time-window join 규범화) ⓒ§6-1 **quick cells
> JSONL 보존 계약**(threshold 생성~최종 judge 종료까지 immutable·content digest+quick
> plan/run SHA 결속) ⓓ§3-3 **workset digest 계약**(동일 workset 대조=physical tuple
> multiset digest — 단 **C-solo 제외·C34는 logical payload digest**[단일 record vs parts
> 표현차로 physical 구조 불일치]) ⓔ§4-2 prefetch_on=**operator attestation**(PA 헤더에
> 필드 부재 — 1차 소스 확인·증거 파일은 optional).
> ★**부속 정오 3(26-08-07 — 심의=코드 r3 `reviews/codex_ubench_impl_r3.md`·확인=r4. 전부
> 기수렴 범위 내 기술 정밀화 — Codex 확정 문안 채택)**: ⓐ정오 2-ⓐ의 실행 cap
> 11,912/1,485는 **nominal record target**으로 격하 — main 규범=**"64GiB ≤ one-pass useful
> bytes ≤ 76.8GiB·record 편차 ≤20%"의 byte-floor 우선**(실측 11,912×평균 5,753,124B=
> 63.82GiB<floor — 두 조항 동시 충족 불가 해소) ⓑ**R5-ref=origin-bound replay**(§3-2 총람
> `R5-ref × diag`·§6-0 판독 5의 diag 문면 교체 — trace_scope·prefetch_on·ops path/digest는
> `r5_reference.origin_seal_stream`과 같은 run의 `replay_ops_source`가 정한다. 현
> qwen122_bench seal=prod trace-bench·prefetch_on=true. **diag_cpuonly는 4.98 reference가
> 아니다**) ⓒ**engine/raw 분모 분리**: engine token 값=`miss_bytes/Σ(R−D)` · raw token
> 값=같은 full graph_seq miss-op 집합의 `useful_bytes/[first submit, last publish)` — 두
> 분포의 **p50 비교**·pooled-sum ratio 금지 ⓓ`R5_SAMPLE_RULE` 문면 정정 — "flags
> bit0==0" 필터는 seal 생성기·parser 모두 실제 미적용(실물: prefill 3,120행 포함·
> seal-as-built p50=4.976054 vs decode-only 4.974244) → **4.98 seal-as-built 규범 유지·
> selection 문면을 실 공식에 정합** ⓔ**quick 규범 교체**: `--quick`=25 smoke coordinate
> 각 1회·§3-1의 10초/64GiB floor 미적용·coordinate당 atomic 실행 단위 보존 deterministic
> quick workset **1 pass만**(2번째 pass 진입 금지=hard pass limit)·expected completed
> bytes(**read_len 합**) ≤8GiB·총 ≤200GiB·목표 15분·hard ceiling 20분(도달 시 cell 경계
> 중단·불완전 quick=threshold/ETA 원천 금지). 절단 단위: A1=block·A2/팔3~8=record/
> coalesce group·trace=whole token·Stage C=전단사 대응 record 쌍. calibrate=quick
> plan/manifest 결속(`plan.quick=true`·25셀 전건·latest-valid·동일 plan/run) 필수.
> judge 최소표본 `MIN_JOINED_TOKENS=30` 등재. ★수리 3차 재유도(확인=r4): quick trace
> nominal 1,485→**1,119**(표본 오차+브라켓 포함 read_len 합이 8GiB를 넘지 않도록 예산
> 6GiB 재유도 — ⓐ의 nominal 격하 논리 동일 적용)·quick per-cell stall guard 300s.
> ★**부속 정오 4(26-08-07 — 심의=게이트 판독 r1 `reviews/codex_ubench_gate_r1.md`
> [MODIFY 2: D1·D2 MODIFY/D3 ACCEPT]·사용자 수렴 19:14[경로 승인·재팩 실측 우선]·
> ★확정=게이트 r2 `reviews/codex_ubench_gate_r2.md` [MODIFY — 전건 수용]·★**시공 교차=
> 게이트 r3 `reviews/codex_ubench_gate_r3.md` [MODIFY — 전건 수용·아래 문안에 반영 완료]**)**:
> 발단=본 런 1회(`ub_20260807T050917Z`)가 게이트에서 **valid 7/705** —
> 지배 원인=quick 유도 임계의 지속 체제 비대표성(quick 실 IO 합 **35.5초** vs 본 런
> 3.14h·D: 실측 50~70°C. 왜곡 양성 증거 미발견: 클린 rep bw-온도 r=0.0100·PCIe 드랍 0).
> 인시던트 증거=`ubench/diag_contamination_r1/`(**보존·판독 인용 금지**·외부 IO 인시던트
> 집합=position 301~462, 162셀 fail-close·판독 1~8 전건 인용 불가[유효 A/A 1/0/0]).
> ⓐ**quick 격하(§3-5·§6-1·§8·§9·§10)**: "`--quick` 측정값은 파이프라인·권한·센서
> preflight 및 §8 비게이트 ETA 산출에만 사용할 수 있다. 온도 임계·start band·판독 원천
> 사용은 금지한다." ★**부속 정오 2-ⓒ·정오 3-ⓔ 중 quick→온도 thresholds/calibrate 결속은
> 본 정오가 폐기한다** — quick cells/plan 결속은 §8 비게이트 ETA 산출에만 잔존.
> ⓑ**steady-state 캘리브레이션(임계 유도 규범 — r2 확정 산식)**:
> `run_kind=steady_state_calibration`·D/C 세그먼트 순차 단일 슬롯·전용 plan(기존 러너·셀
> 스키마 운반). ★**단일 프로세스 계약(r4 — `--stage`/`--cell` 분할이 정상 CLI로 도달
> 가능했음·메모리 검산으로 epoch 검증 통과 실증 / r5 정밀화)**: 측정 실행은 **새 출력 파일을
> 원자적으로 생성해 보유하는 단일 full-plan D→C 프로세스**여야 한다. 원자 claim=
> `CreateFileW(GENERIC_WRITE, **FILE_SHARE_READ**, CREATE_NEW)`를 **plan identity 확인과
> `--list` 반환 직후·모든 work/source 접근 전에** 실행하고 handle을 run 종료까지 보유한다
> (★r5: 구 순서는 claim이 work digest·source open 뒤라, 경쟁 프로세스의 검증 read가 진행 중
> epoch의 외부 IO를 오염시킬 수 있었다 — `fopen_s` 존재검사는 공유 위반을 "부재"로 오독하므로
> **보증 수단이 아니다**). `FILE_SHARE_READ`는 reader만 허용하고 `FILE_SHARE_WRITE/DELETE`가
> 없어 두 번째 writer·append·rename/delete는 계속 거부 — 분할 차단력 무훼손.
> `--resume`·`--stage`·`--cell`·**기존 `--out`**은 source open·셀 실행 전에 **exit 2** 거부
> (`--list`는 비측정 예외). ★**exit taxonomy(r5·r6 확정)**: **exit 1 = 검증된 JSONL을 보존한
> calibration VOID에서만** 발생(아래 조기 중단) / **exit 2 = `--selftest` 실패를 포함한
> 진단·CLI·plan·identity·output 계약 실패 전부**(필수 `--plan`/`--out` 누락·write/flush/close
> 실패 포함). 출력 오류 시 보존 성공문을 내지 않는다. ★**조기 중단(r4 — 구 "조기 종료 없음" 문구 교체)**: 첫 `CellResult.valid=false`
> 셀의 JSON을 보존한 뒤 **다음 셀 전에 중단**하고(Stage C면 그 셀까지 flush·공통 정리 경로
> 경유) — ★r5: **완전한 JSONL 기록과 flush/close 성공이 확인된 경우에만** — **exit 1**. 근거=확정 VOID 세그먼트를 계속 도는 것은 최대 ~120분 순손실이며 epoch
> 계약상 어차피 전량 재실행이다. 중단된 epoch는 **보존**하고 새 plan/run_id/output으로 D
> 첫 셀부터 재실행한다. workload 고정: **D:=A1 2MiB·qd8·t1 /
> C:=C_solo_C(c_drive_c 전단사 workset)·qd8·t1**. 규모=**세그먼트당 60셀×60s**(★r3: 구
> 66셀은 t\*가 3,600s 이하 경계만 쓰므로 61~66번째가 W 표본에 기여하지 못한 채 오염 표면·
> 슬롯만 늘림 — 슬롯 floor 132→**120분**). 판정시각 **t\*=세그먼트 시작 후 60분 이하의
> 마지막 완결 셀 경계**(과거 통과 창 검색 금지)·**W=[t\*−15분, t\*]**. `plateau_max`=W 겹침
> 셀 `temp_max_c` 최댓값·`plateau_min`=W의 `temp_start/end_c` 최솟값·첫/끝 5분 중앙값도 W
> 경계 온도 표본(★r3: **각 edge 경계 표본 <5개면 `CALIBRATION_INCONCLUSIVE`** — 구 ≥1
> 검사 강화. ★실측 여유[r4 정정]: 본 런 실측 1-pass D 12.2~14.6s·C ~3.5s → 셀 길이
> 61~73s → edge당 표본 **8~10개**[위상 미정렬 보수 검산·구 "2배 이상"은 상한 근사]로
> 게이트 5를 넘으므로 `CALIBRATION_CELL_SECONDS` 조정 불요. 편차가 커지면 게이트가
> fail-close). 성공=**W 범위≤2°C ∧ 첫/끝 5분 중앙차≤1°C**. 실패·센서 누락·계획 셀 누락=
> `CALIBRATION_INCONCLUSIVE`(임계 미산출·fail-close). 세그먼트 내 외부 IO ratio
> unavailable/1% 초과 셀이 **1개라도 있으면 fail-close**(오염 셀 제외 후 창 선별 금지).
> ★**epoch 연속성(r3 — 정상 입력 도달 결함 처방)**: calibration segment는 **분할 불가능한
> 연속 측정 epoch**다. segment 중간의 프로세스 재시작·closure-invalid·개별 셀 재시도는
> `CALIBRATION_INCONCLUSIVE`이며 해당 segment를 **fresh epoch로 전량 재실행**한다.
> calibrator는 segment epoch 단일성·plan 순서와 시간 순서 일치·**D 종료 후 C 시작**을
> 검증하고, 외부 IO gate는 **폐기된 attempt까지 포함한 epoch 전건**에 적용한다(구 구현은
> latest-valid 선택으로 비연속 epoch를 정상 봉인할 수 있었음).
> `temp_limit_c=plateau_max+5` **고정식**(margin 인터페이스는 5 외 거부)·
> `start_band_c=[plateau_min, plateau_max]`. calibration plan/run/cells SHA·유도식 버전을
> thresholds(`ubench-thresholds-2`)에 봉인 — **이후 생성된 새 main plan SHA에만 효력**.
> ★순환 차단(실문장): 본 런 r1의 온도·BW 자료는 기존 quick 유도식의 대표성 반증과 차기
> calibration 설계 입력으로만 사용한다. r1에서 숫자 임계 선택·BW 기반 셀 선별·r1 validity
> 변경·판독 1~8 인용을 금지한다. r1 기반 후보는 `diagnostic_only=true`·
> `promotion_gate_status=NOT_EVALUATED`로 남기며, 독립 확인된 임계는 그 이후 생성된 새
> main plan SHA에만 효력을 갖는다.
> ⓒ**stage-entry warming precondition(r2 확정 — 러너 내장·구 10분 상한 기각[68°C 도달
> 49.3분 실측과 충돌])**: 정상 연속 런에서는 D:를 첫 Stage A 측정 전, C:를 첫 Stage C
> 측정 전에 precondition한다. 재시작·resume 시 첫 미완결 측정 셀의 대상 드라이브에
> 재적용한다. 온도가 band 하한 미만이면 calibration과 동일한 고정 workset/QD의 **비계측
> 워밍 패스**를 반복하고, pass-end 온도가 **3회 연속 band 안**일 때만 측정에 진입한다.
> ★**마감·완결성(r3 — 도달 결함 처방)**: 워밍 성공은 **완결된 1-pass closure에 한해서만
> 계수**한다(불완전 workset·duration cap·closure-invalid 패스=
> `abort_warming_pass_incomplete` fail-close). **세 번째 in-band pass-end 시각이
> precondition 시작 후 3,600초 이하일 때만** 진입하고, 초과 시 성공 판정보다 **먼저**
> `AbortTimeout`으로 종료한다(구 구현은 3연속 성공을 timeout 검사보다 먼저 반환해 60분
> 초과 성공이 가능했음).
> 상한 초과 시작·워밍 중 `temp_limit_c` 초과·외부 IO ratio>1%·**60분 내 3연속 진입 실패**
> 는 본 셀 실행 전 abort한다. 비계측 패스는 705 실행·판독 표본에 불포함하며(★r3 실물 검증:
> 705 회계·드라이브 카운터·외부 IO 비율에 **누수 없음** 확인) 첫 측정 셀에
> `stage_entry_precondition` 증거(workload identity·시각·온도 표본·외부 IO·종료 사유)를
> 기록한다. ★**게이트 의미 분리**: `start_band_c`=**stage-entry precondition 전용** /
> `temp_limit_c`=**각 측정 셀 전용** — judge의 per-cell start-band 게이트는 제거한다
> (미분리 시 warming을 도입해도 후속 셀 대량 무효 재발·`ubench_judge.py:302`).
> ⓓ**셀 절대시각 스키마(r2 확정 — `ubench-cell-2` bump)**: `identity.t_start_utc`/
> `t_end_utc`(`YYYY-MM-DDTHH:MM:SS.sssZ`) 필수·실 I/O 구간과 동일 지점 수집·
> `t_start≤t_end` 검증·duration 권위는 QPC 유지(UTC=외부 창 교집합 전용). 새
> plan/manifest는 `cell_schema_version=ubench-cell-2` 봉인·새 run의 v1/v2 혼합 거부·
> `calibrate`=v2만 수용·`eta`=v1/v2 수용(v1=절대창 산출 불가 표시)·judge=v1 자료는 진단
> 전용(`absolute_time_unavailable` 표시·v2 요구 plan에서 v1 행=fail-close)·resume=plan
> 선언 schema와 기존 행 schema 일치 필수(혼합 append 금지).
> ⓔ**선별 재실행 문면 정정(§6-2)+judge rerun 출력 분리(r2)**: 러너 resume done=
> complete∧closure.valid(`ubench_io.cpp:3289·3861`) — judge-invalid·closure-valid 셀은
> 현 러너로 선별 재실행 **불가**. judge rerun 출력은 **`selective_rerunnable`
> (closure-invalid)/`new_plan_required`(closure-valid·judge-invalid)로 분리**(현행
> rerun_semantics의 일괄 --resume 안내는 거짓 — 수정). 러너 done 의미 개정=차기 개정
> 백로그. 이번 사이클은 **새 thresholds→새 plan SHA→705 전량 재실행**으로 우회 확정.
> ★시공 품목(r2 확정 표=전사 §요청 3): P0 문서 폐합·P1 planner/calibrator
> (`ubench-thresholds-2`)·P2 runner(cell-2·warming 게이트)·P3 judge(v1/v2 정책·start-band
> 게이트 제거·rerun 분리)·P4 회귀 묶음·P5 새 산출물 디렉토리(★r3 갱신 — **3개**:
> `plan_quick_r2/`→`plan_calibration_r2/`→`plan_main_r2/`. 기존 `plan_quick/`·`plan_main/`·
> `diag_contamination_r1/` 불변 보존).
> ⓕ**ETA의 warming 정직 표기(r3)**: ETA는 측정 셀 시간만 합산하므로 `eta.json`에
> `eta_excludes_stage_entry_warming: true`·`continuous_run_warming_budget_seconds: [0,7200]`·
> `resume_warming_note`(재개마다 첫 미완결 셀이 touch하는 드라이브별 최대 3,600s 재적용)를
> 기록한다. ★7,200s 상한 선언은 ⓒ의 마감 수리가 반영된 뒤에만 참이다.
> ⓖ**새 빌드 quick 1회 필수(r3 요청 3-1 확정)**: 보존된 구 quick cells(v1)는 §8 비게이트
> ETA 원천으로 **재사용 가능**하나(실물 검사: plan.quick=true·25셀 전건 closure-valid·결속
> 일치), 구 plan은 새 러너가 거부하므로 **새 빌드의 pipeline·권한·센서 smoke와 cell-2
> writer 실 경로를 시험하지 못한다** ⇒ §8 "시공 후 quick preflight" 이행에 `plan_quick_r2/`
> 생성+1회 실행이 **필수**. 정상 시 ETA도 새 v2 quick을 쓰고 구 v1은 보존 fallback·회귀
> 증거로만 둔다.
> ★★**부속 정오 5(26-08-08 — r2 본 런[`ub_20260807T203020Z`대 상당] 판독 실패의 원인 규명
> 3라운드 심의로 확정: verdict r1[ⓒ 부분 봉인]·power r1[판별력 MODIFY]·space r1[공간축
> MODIFY]. 전사=`reviews/codex_ubench_{verdict,power}_r1.md`·`codex_space_axis_r1.md` —
> 조항 원문(확정 문안)은 전사가 권위·아래는 규범 요지. 사용자 수렴 26-08-08: 실행 시간
> 5~6h 승인·§5 3상태 승인·QD fallback 승인·SP-A 문턱=성능 무손실 우선)**:
> **1** r2 런=`PARTIAL_READOUT` 봉인·소급 재판정 금지(validity·threshold 불변).
> **2** ★**판정식 교체**: A/A max band **폐기** → `log(B/A)` 대칭화·superiority=one-sided
> 95% LCB>0·비열화=90% CI ⊂ ±10%(사전등록 margin — 하류 GATE qwen 15%/D·K 10% 근거)·
> 다중비교=사전등록 max-t/wild-cluster bootstrap·**A/A는 비정상성 sentinel/QC로만**(임계
> 생성 금지). 사전등록 파일(α=.05·power=.80·margin·좌표·가중치·seed·제외규칙·CI 알고리즘·
> bootstrap 횟수·interaction 처리·fallback)을 **plan SHA 전 동결**. 근거=현 판정식 MDE80
> 판독 1 기준 40.8%(p95 74.9%)로 필요 해상도 10%의 4배·**n 증설은 역효과**(max band
> 비감소+전건 통과 요건 — n=5에서 44.0%).
> **3** ★**A1 격자 재설계**: 전역 sweep(fwd→rev→random — 동일 좌표 반복이 런 전체에 흩어져
> 드리프트 흡수·실측 원거리 산포 11.75% vs 근거리 5.76%) 폐기 → **(block,T) 18 strata별
> QD 5개 1블록·coordinate-local 반복·5×5 cyclic Latin square(절반 reverse — 위치·1차
> carryover 균형)**.
> **4** B/C 인접 pairing 유지·**핵심 좌표 n=5/진단 n=3**. QD 판독=18 strata equal-weight
> marginal mean에서 **최고값 대비 10% 이내 최소 QD 선택**(QD16이 10% 초과 시 veto)·Stage B
> primary QD {2,4,8}(QD1/16=interaction 진단 분리)·**좌표별 방향 반전 시 전역 승자 강제
> 금지**(QD-조건부 정책 또는 policy_fixed+e2e gate 귀결 — C7 실측 QD1 +30.9%/QD4 +11.6%/
> QD8 −1.7% interaction이 근거).
> **5** **drift pilot**(★**hard ceiling 60분·목표 55분** — 26-08-08 16:2x 정정: 구 "≤55분"은
> 메인이 정오 5 전사 시 권위(`codex_ubench_power_r1.md:136` **"1시간 이내 가능: 목표 55분
> 이하"**)를 좁혀 적은 것이며, 그 결과 warming 예산이 3,300−2,910=**390s < 실측 D: warming
> 429s**로 **콜드 스타트 시 구조적 abort**가 됐다[Phase A 수리자 적발]. 권위 내역
> 7+36+6+6=55는 anchor가 grid 안에 포함되는 구현과 어긋나 slack이 가산되지 않는다. 정정 후
> warming allowance=3,600−2,910=**690s**[실측 429s 대비 여유 261s]. 총 상한은 권위의 "1시간
> 이내"를 그대로 쓴다)·별도 artifact·임계 조정 사용 금지: D: A1 2MiB/QD8/T1 anchor
> 13 시간점×연속 2·사이 12구간=SAME-LBA/ROTATE-LBA/IDLE 각 4(대칭쌍 배정 — ★**정오 6-①로
> 교체**: IDLE 6/SAME 3/ROTATE 3·상수 배정)·margin 5% 고정·
> 판정=CONFIRMED/NOT_MATERIAL_WITHIN_48M/UNRESOLVED(+THERMALLY_CONFOUNDED). 상세 표=power
> r1 §3.
> **6** ★**온도 계약 교체**: `entry_band`(현 캘리브 원자료 유지 — D 63~65·C 62~63·stage-entry
> 전용)와 `safety_ceiling`(**장치 경고선−5: D 78·C 75** — Kioxia 경고 83/임계 85·P41 80대
> [사용자 조사]) **분리**. ceiling 아래 온도=셀 제거 근거 금지·**covariate 기록만**(근거=
> 71°C 셀이 저온 반복 대비 p50 +7.7%·하락 0건 — 구 게이트가 고성능 셀 선택 제거로 중앙값
> −4.81% 하향 편향 유발). ceiling 초과=셀 선별 제거가 아니라 **paired block 전체 fail-stop**.
> **7** **§5 폐합=3상태**(`measured_selected|policy_fixed|integration_deferred`·bare
> not_identified=0 — 사용자 승인·상세=SPEC_REPACK_V3 부속 정오 1-ⓐ). 측정 우선 핵심 5=
> `io_qd_total`·`io_part_parallelism`·`io_submit_order`·`io_bounce_copy_workers`·
> **`io_bounce_copy_qd`(matrix 신설 — 러너 기지원·planner만 확장)**.
> **8** split/resume이라도 동일 plan SHA·manifest·append-only 로그면 **한 번의 replacement
> main**으로 정의(슬롯 분할 허용 근거).
> **9** ETA·셀 감축 규칙=실행 전 동결·결과 본 뒤 표본 축소 금지.
> **10** ★**C34/C45/C35 의미 계약**: C34=one-read API-shape control↔serial 3-part·C45=
> serial↔concurrent — 어느 하나도 단독으로 SP-A kill-screen 통과를 뜻하지 않음.
> **`C35=arm3↔arm5(QD8·T1·arm당 matched n=5) 20셀 신설**(의도된 v3 baseline의 직접
> I/O-shape 입력). cross-contrast 곱셈·추이식 판정 금지.
> **11** r2 C34 지위=`diagnostic_only/PARTIAL_READOUT`(설계·사전등록 입력 인용 가 /
> measured_selected·kill-screen PASS·SP-A_MODE_GATE 근거 불가).
> **12** **프리필 정정·소유권**: 구 mmap QD 0.26~0.35의 direct+QD8 이식 **철회**(현 프리필
> =demand만으로 QD 7.85/8 포화 — `codex_prefill_axis_xcheck.md`). 단 record-QD와 v3
> child-QD는 별개 estimand라 C34 QD8 수치의 프리필 e2e 이식도 금지. **S2 최종 판정=구현 후
> 재팩 소유 prefill bin↔virtual A/B 전속**(C7=coalescing microbench 입력까지).
> **13** **bounce 범위**: C6+copy-QD matrix=worker/QD 기본값 판정까지. gpt-oss SP-A 승격=
> 구현 후 profile A/B 전속(copy GB/s·p99·e2e — 실패 시 해당 프로파일만 bin 회귀).
> **14** **Stage C 의미 제한**: C_static/C_dynamic=**D/C 복제 요청열의 mirror-scheduler 팔**
> (1× 정적 분할 아님) — dual-drive additivity(실측 98.4~98.8%)·mirror 내부 dispatch 증거로만
> 사용. 1× 분할·R-3 재개봉·공간 절감 근거 인용 금지. D-SC1 판정=min RΣ+p99·D-A2=별도 e2e.
> **15** **게이트 지위 분리**: microbench는 §5 폐합만 발행 — SP-A_MODE_GATE·D-A2_PROMOTION_
> GATE를 발행하지 않음. 효과 없음·equivalence·policy_fixed도 정상 폐합. **QD fallback=
> `io_qd_total=8_pending_e2e`(사용자 승인)**.
> **16** **셀 총람=1,059**(A 649[A1 450+A2 180+anchor 12+sentinel 7]/B 359[C35 20·bounce-QD
> 40 포함]/C 51)·한 프로세스 ~4h46m·슬롯 3분할(①pilot **≤60분(목표 55분 — 조항 5 정정)**
> ②Main-1 Stage A ~2h55m
> ③Main-2 `--resume` B+C ~1h58m)·**캘리브 재실행 불요**·소요 권위=planner ETA. 성공 판정
> 9요건=power r1 §5(roll-call 정확·핵심 n=5 전건 closure-valid·sentinel 비발동·15행 전건
> 3상태·"효과 없음"도 정상 결론).
> 이행 근거=`SPEC_REPACK_V3.md`(동결 v1.0)
> §9-3·§6-3. 설계 기반=r2 Q5·Q6(`reviews/codex_repack_speed_r2.md:212-343`). 지표 정의=
> `SPEC_IO_METRICS_V3.md`(동결 v1.0) §6 공유(러너는 IO3 wire **미사용** — 동 §10-2 결속).
> ★**Q5 원문과의 명시 상충·확장 목록**(r1 필수 5 — 본 문서가 우선하는 지점): ①Stage A
> 반복=정/역/무작위 3 sweep(Q5 "ABBA"의 격자 대체 — drift-balanced ordering·ABBA 인터리브는
> 팔 비교 전용) ②D-SC1 solo QD에 `{16}` 추가(원문 `{2,4,8}` — 확장) ③K2.6 exact=합성
> (원본 부재 — 결론 범위 `device_size_only`) ④qwen exact=두 실측 stride(원문 단일 5.71MB
> 표기는 48층 평균으로 판명 — r1 검산).

> ★★**부속 정오 6(26-08-08 — drift pilot size 게이트 심의 3라운드로 확정: r1[ⓑ 재설계]→
> r2[HOLD 유지·ⓐ 설계 수정]→r3[MODIFY 후 ⓔ 복합]. 전사=`reviews/codex_ubench_size_gate_
> r{1..3}.md`(조항 원문 권위). ★메인(Fable) 독립 재검산 전건 일치 — CP=별도 코드 경로·
> leverage=정확 분수 연산·N=[3,000, 23,250] 전수 스캔·봉인 digest 재유도. Fable 동결 26-08-08)**:
> **①** drift pilot 12 state interval 배정=**상수 핀** `IDLE={2,4,6,7,9,11}·SAME_LBA={1,8,10}·
> ROTATE_LBA={3,5,12}`(구 정오 5-⑤ "각 4" 문면 폐기 — 6/3/3 기하는 U1[r2] 승계). 기하 불변식=
> idle 합 39=active 합 39·SAME 19·ROTATE 20. leverage 봉인(정확 분수)=base elapsed max
> **512/2571=0.1991443018**(1/13의 2.589×)·position 0.0769339392·condition 전 cluster **1/12**.
> ★적격 배정 10개 중 최소최대 동률이 **정확히 2개**(핀 배정과 `IDLE={3,4,5,8,9,10}·SAME=
> {1,7,11}·ROTATE={2,6,12}` — 정확 분수 동일)이므로 **seed 동률 해소 금지**(명명 배정과 설계
> digest가 분리될 위험) — 최소최대 규칙은 도출 기록·회귀 가드로만 유지(적격 10·동률 집합 2·
> 핀∈동률 집합).
> **②** size calibration=`ubench-drift-size-calibration-4`·**N=23,207 exact 상수**(`N≥` 표기
> 금지 — 통과확률 비단조: 23,206·23,208~23,210 미달·[3,000, 23,206] 전수 통과 無)·
> `assurance_kind=region`·**assurance_region=[0.09, 0.11]**·outer acceptance [0.08, 0.12]·
> PASS/FAIL/HOLD/CALIBRATION_ERROR 4상태·q=FWER 0.01/(2·3) 불변. 봉인 수치: pass count band
> **[1980, 2640]**·P_PASS(.09)=0.9941341·P_PASS(.11)=0.9666843·familywise 하한 **0.9000530**
> (region 내부 최소=끝점 0.11·여유 +1.8e-5 — 그래서 exact 상수여야 한다).
> ★**provenance(확정 문안)**: region [0.09,0.11]은 **prereg-6 HOLD 이후 채택한 adaptive
> precision-planning target**이다 — U3 미이행의 이행도, G=13에서 유도된 size 상한도, prereg-6
> 결과의 재해석도 아니다. N은 fresh draws에 앞서 봉인하며 이전 count 합산·연장 금지. 관측 CI
> 초과분만 메우는 N≈6,234 계열은 결과를 본 뒤 N을 고르는 절차라 금지.
> **③** calibration artifact 무결성: cold write·cache hit **양 경로 모두** 파일 raw SHA-256
> 계산·반환, plan emit 전 64자 SHA 존재+현재 파일 bytes 재계산 일치 강제(부재·불일치=
> fail-close)+cold↔warm 동일성 회귀. 근거=현 warm-cache 경로가 SHA 없는 문서 반환(정상 재사용
> 경로라 도달 가능 — on-disk artifact에 키 부재 실측).
> **④** prereg 세대: prereg-6=**`CALIBRATION_HOLD / PILOT_NOT_EMITTED`** append-only 종결
> (건전 계약의 정확한 음성 증거 — scheduled 635/5881=0.107975·CI 상한 0.120346이 0.12 경계
> 0.000346 초과·나머지 2 대비 PASS·artifact raw SHA `8fab7a34…`). `ubench-drift-prereg-7`
> 신설·fresh calibration seed(배정 seed는 상수 핀으로 선택 미사용 — 기록 전용). **draws 재사용
> 금지**: prereg-5/r2·prereg-6/r3 전량 합산·연장·재생성 금지·기존 PASS 2건 승계 금지·
> deterministic leverage 계산만 설계 입력 재사용 가.
> **⑤** 라운드 상한: size 게이트 **r4=마지막 geometry/precision 라운드**. 비PASS 시 추가 N·
> 배정 튜닝 금지·질적으로 다른 method-replacement 라운드 **1회만** 허용·그것도 실패하면 이번
> 사이클 confirmatory pilot 종료(추가 정적 라운드 절대 상한 2회).
> **⑥** 예산 불변: pilot 38셀·정적 2,910s+warming allowance 690s=hard ceiling 3,600s·Phase B
> 3분기 불변. calibration은 plan 전 합성 계산(직렬 단일 코어 ~75~95분)이며 pilot ceiling과 별도.
>
> ★★**부속 정오 7(26-08-09 — stage-entry warming 의무의 범위를 기계적으로 못박는다. 계기=재팩
> A3 drift pilot 교착 · 심의=Codex r8 Q6(선택지)→**r9 Q4 안전성 CONFIRM**·r9 M4/M5/M6 ·
> 처분=리드 판정[A3=ⓑ 러너 정정 · r9 필수 8항 전건 수용] · 전사=`reviews/codex_verifier_
> contract_r{8,9,10}.md` · ★★**동결(26-08-10 00:5x 리드(Fable) 확정 — r10 Q5 심의·r11 정정
> [M3 용어·M4 분리 기재] 반영 완료 · 부속 정오 2 와 같은 커밋 원자 동결·이후 재심 경유)**)**:
>
> **①★워밍 의무의 범위 = "밴드 ∩ 측정 셀 I/O 가능 드라이브"**(구 러너 구현의 "밴드를 가진 모든
> 드라이브"를 폐기) `[[C:repack.warm-touch|src]]`. 러너는 plan load 시 그 집합을 먼저 구하고,
> **그 집합에 대해서만** 워밍
> 워크로드의 존재를 fail-close 검증한다. 밴드가 있어도 측정 셀이 I/O 하지 않는 드라이브(예:
> 캘리브에서 상속된 `C:` 밴드를 지닌 D:-only drift pilot)에는 워밍을 요구하지 **않는다** —
> 정오 4-ⓒ 가 의무를 처음부터 "D: 를 첫 Stage A 측정 전, C: 를 첫 Stage C 측정 전"으로 적었고,
> **측정 셀이 I/O 하지 않는 드라이브**의 준비를 요구하는 것은 계약이 아니라 계약의 오독이기
> 때문이다(★r10 M3 — 구 문면 "열지 않는 드라이브"는 바로 아래 ②의 literal-open 부정과
> 모순됐다: 러너는 `plan.sources` 전부를 실제로 연다).
> **②★`touch`(=I/O 가능)의 기계적 정의** — 용어가 정의되지 않으면 다음 구현자가 다시 갈린다:
> > `CellSpec.source_ids`(비어 있으면 `spans[].source_id` 에서 복원) → 대응 `SourceSpec.drive_key`
> > (비어 있으면 `volume`) → **`plan.cells` 전건에 대한 합집합**. 이는 **literal handle open 집합이
> > 아니다** — 러너는 셀 선택과 무관하게 `plan.sources` 전부를 시작 시 열고 identity 를 재해시하므로,
> > 의무의 기준은 "핸들을 여는가"가 아니라 **"측정 셀의 요청 스트림이 I/O 할 수 있는가"**다.
> **③합집합은 stage/`--cell` 필터·resume 과 무관하게 plan 전 셀에서 취한다.** 실행 셀 집합은
> 런타임에 달라지므로 실행 대상만 보면 의무가 런마다 흔들린다.
> **④검사는 load 시점에 남는다**(삭제·지연 금지). 첫 측정 셀 **전에** 거부하는 것이 이 검사의
> 존재 이유다 — "첫 사용 시점까지 지연"은 앞선 측정을 남긴 채 중간에 실패해 **부분 표본**을
> 만든다(r8 이 "명백히 안전하지 않다"로 분류한 3안 중 하나).
> **⑤런타임 stage-entry fail-close 는 존치한다**(첫 touch 시 워밍 워크로드 부재=즉시 중단).
> load 검사의 중복처럼 보이나 **계약층**이며, 정상 load 를 통과한 불변 plan 에서만 도달 불가다.
> **⑥★preflight 와의 관계(정직 표기)**: 플래너 preflight 는 전 셀 union 이 아니라 **`plan.sources`
> 의 drive 집합**과 밴드를 교차한다. 현 D:-only trusted drift 발행기에서 **결과가 같을 뿐**이며,
> 두 알고리즘이 일반적으로 동일하다고 적으면 안 된다.
> **적용 — ★역사 스냅샷(r9 시점 기록 · 소급 수정 금지)**: `ubench_io.cpp`
> `drives_touched_by_plan()`·`missing_warming_drives()` 신설 ·
> runner selftest 4건 신설(64/64) · `ubench_plan.py` **무접촉**(terminal 등록 유지).
> **현행 상태(r10 N2 이후)**: warm obligation 계열 **11건** · 제출 실측 총 **71 PASS**
> `[[C:repack.selftest-counts]]` — 위 스냅샷은 그 시점 기록으로 보존하고 현행은 분리 기재한다
> (r10 M4-ⓑ·리드 판정). 이 수치의 **원본은 산문이 아니라 실측**이며 repacker selftest
> `CONTRACT-REGISTRY-②` 가 `ubench_io.cpp` 의 `"warm obligation:` 검사 수를 세어 직접 대조한다.
> `ubench_plan.py` 무접촉은 현행에서도 유지된다.

## §0. 한 줄 정의

엔진 밖 합성 I/O 러너로 "정격 10GB/s → fio 6.5~8.1 → 엔진 실효 R 4.98~7.6" 갭을 층별로
분해한다 — 장치/스택 천장(Stage A), v3 구현 형상별 비용(Stage B), 이중 드라이브 협주(Stage C).
러너 산출은 **증거(evidence)**다 — 게이트 판정(D-A2 승격 등)은 하지 않는다(§6-4).

## §1. 도구 구성 (시공 발주 단위)

| 구성물 | 언어·역할 |
|---|---|
| `ubench_io.exe` | C++ 단일 파일(MSVC) — I/O 코어: `CreateFileW(GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NO_BUFFERING\|OVERLAPPED)`·수동 event·스레드 토폴로지(§3-2 팔 8·§3-3 — ★r2 인용 정정)·QPC 계시·affinity 고정·plan 입력→JSONL 출력 |
| `ubench_plan.py` | planner — 실물 GGUF/expect에서 offset 스케줄·셀 목록·ETA 산출·preflight(§6-1). ★파서 재사용 **함수 단위 고정**(r2): `bench/repack/repack_experts.py`의 `parse_gguf_header()`(:279)·`load_model_shards()`(:353 — qwen multi-shard)·`per_expert_slice_bytes()`(:429)·`compute_record_layout()`(:606) |
| `ubench_judge.py` | 판독기 — §6 판독·무효·A/A noise band·파라미터 표 산출 |

- 전부 `bench/techdev/repack_space/ubench/`(신설·재팩 도메인). 시공=Opus 위임 1건(사양 밖
  변경 금지·스폰 시 사유·결과 보고 의무).
- buffered fallback **금지**(열기 실패=셀 fail). 대상 파일 전부 read-only(쓰기 핸들 자체를
  안 연다) — 예외=Stage C 코퍼스 **생성 단계**(§4-3)와 결과 JSONL(C: 프로젝트 볼륨).

## §2. 공통 I/O·로그 계약

- offset·buffer 주소·길이는 런타임 sector/alignment 정렬 — nominal과 실제 정렬 바이트 병기.
- 결과 로그는 C: 프로젝트 디렉토리. ★**Stage C 구간은 셀 단위 flush 금지 — Stage C 전체
  종료 후 일괄 flush**(r1: 셀 사이 C: 쓰기가 다음 P41 셀의 외부 I/O·온도를 오염).
- 각 셀 시작·종료 env 스냅샷: SSD 온도(start/max/end)·PCIe link gen/width(baseline/current)·
  전원 계획·해당 볼륨 외부 I/O 카운터 델타(§6-2 산식용 raw 병기).

## §3. 스테이지 구조 (★r1 차단 1·3 반영 — 격자/팔 재구성)

### §3-1. Stage A — 장치 천장 (A1 nominal + A2 raw-exact + sequential anchor)

- **A1 nominal 격자 90좌표**: block `{512KiB, 1MiB, 2MiB, 5.3MB, 13.4MB, 18.9MB}` × 전역
  QD `{1,2,4,8,16}` × 스레드 `{1,2,4}`. ★대형 3종의 규범 바이트(부속 정오 1-③ — 구 "MiB"
  표기는 오기): **5,308,416B·13,369,344B·18,923,520B**(=§4-1 exact 3종과 동일값·10진 MB). 접근=**random/scattered ceiling**(명명 한정 — r1):
  working set(≥128GiB 구간·부재 시 파일 전체) 블록 단위 **무작위 순열**(시드 기록·동일 비교
  간 span·시드·시작 LBA 목록 재사용).
- **A2 raw-exact 60좌표**: exact 크기 4종(§4-1 — qwen 2·DSV4 1·K2.6 합성 1) × QD
  `{1,2,4,8,16}` × 스레드 `{1,2,4}` — Q5 "exact 셀 별도 추가"의 이행(판독 5·6의 raw
  exact/QD16/T4 전제 복원).
- **sequential anchor**(★r1 — 판독 8 "fio·정격 층" 판별용): 순차 대구간 read(블록 `{1MiB,
  13MiB}` × QD `{1,8}` × T1) — 기존 fio 조건과 동일 축(direct·span). fio sealed
  reference(§10-1) 부재 시 판독 8=`INCONCLUSIVE`.
- 반복=**3 sweep(정/역/무작위 셀 순서)** — Q5 ABBA의 격자 대체(drift-balanced·상충 목록 ①).
- 셀 시간=`max(10초, completed 64GiB)`(수렴 완결 26-08-06 — Q5 원안).
- 대상: qwen35-122b 2-shard(A1·A2 qwen) · **DSV4(A2 DSV4·K2.6 합성 — ★r2 상충 해소: 합성
  오프셋의 소스는 §4-1 권위대로 DSV4 shard)** · 필요 시 대용량 잔존 파일.

### §3-2. Stage B — 구현 팔 (★프로파일별 matrix — Cartesian 아님·r1 차단 1)

| 팔 | 정의·형상 계약(SPEC_REPACK_V3 §3 대응 — r1 차단 2 반영) | offset·크기 | QD | T | 비교 계약 |
|---|---|---|---|---|---|
| 1 | **trace replay** — 실측 demand miss 열의 비연속 offset 순서(§4-2 trace 고정·decode flag만·`submit_seq` 순) | qwen 24:24 혼합 stride(실측 열 그대로) | {1,4,8,16} | 1 | 단독(운영 형상 기준선) |
| 2 | 동일 offset 집합의 source별 오름차순 재정렬 | 팔 1과 동일 집합 | {1,4,8,16} | 1 | 팔 1과 paired(§3-3) — 판독 7(layout) |
| 3 | **record-sized one-read I/O-shape control**(★"engine-shaped"에서 개명 — 임의 구간 stride 크기 읽기는 API 형상 통제군이지 bin 물리 layout이 아님·r1) | qwen 두 stride 각각 별도 셀 | {1,4,8,16} | 1 | 팔 4·5의 대조 기준 |
| 4 | v3 part serial — record당 3 part를 **브라켓 주소 규약대로**(`read_offset=abs_offset+e×slice−h`·`read_len=align_up(h+slice,A)` — SPEC_REPACK_V3 §2-4) 순차 read·record barrier | qwen 실물 텐서 abs_offset | {1,4,8,16} | 1 | 팔 3↔4↔5 paired — 판독(part 분할 비용) |
| 5 | v3 part concurrent — 동일 record 3 child 동시 제출·전역 child-QD 관리·all-parts READY | 〃 | {1,4,8,16} | 1 | 〃 |
| 6 | v3 bounce — **gpt-oss 6-child·`io_part_parallelism=3` two-wave**·브라켓 read→staging 착지·**채널 즉시 반환 후 bounded copy queue**(`copy_qd=2`)·staging lifetime=copy terminal·record당 13,253,760B payload copy | gpt-oss-120b 실물(유일 비4K) | {4,8,16} | 1 | copy workers **{1,2}** paired — 판독(비4K 비용·kill-screen ③ 증거) |
| 7 | prefill coalescing — **동일 source/tensor part의 연속 expert run별** 대구간 read(min/max=8/64MiB·window=0µs — SPEC_REPACK_V3 §5·§3-6 7조건 전부 재현)+exact slice scatter | qwen 인접 run | {1,4,8} | 1 | ★**matched UNCOALESCED 대조와 ABBA**(같은 record 집합·같은 바이트 — r1: 대조 없으면 판독 7 불가) |
| 8 | 스레드 토폴로지 — 팔 5 형상 고정·T만 변경(**handle 수·전역 QD 불변** — worker별 disjoint channel/event 소유만 분할·r1 차단 3) | 팔 5와 동일 | {8,16} | {1,2,4} | T1↔T2↔T4 paired — 판독 4(dispatcher 상한) |

★**Stage B cell key·contrast graph(r2 차단 1 — planner가 사양 밖 판단 없이 좌표를 산출
가능하게 폐합)**:

- cell key = `(arm_variant, trace_scope, stride_class, qd, threads, contrast_id)`.
  `trace_scope ∈ {prod(trace-bench·prefetch-on), diag(diag_cpuonly·prefetch-off),
  **sensitivity**(§4-2 권고 2 여벌 trace), synthetic(fallback)}` — 팔 1·2에만 유효(그 외
  `none`·★r3: sensitivity를 값 집합에 정식 등재). `stride_class ∈ {qwen_s(5,308,416),
  qwen_l(6,119,424), mixed(실측 열), **qwen_run**(연속 expert run 대구간 — 팔 7 전용·양
  stride 층에서 추출·UNCOALESCED/COALESCED 양팔 동일 record 집합), dsv4, k26_synth,
  gptoss}` — 팔 3·4·5는 qwen_s/qwen_l **각각 별도 셀**(혼합 아님), 팔 1·2=mixed,
  ★팔 7=qwen_run, ★팔 8=**qwen_l 단일 고정**(stride 승계 안 함 — 판독 4의 목적은 T
  스레딩 효과이지 stride 민감도가 아님·r3 이원 해석 해소).
- **contrast graph(열거 — 이 목록이 전부·전 pairwise 아님)**: `C12-prod`=팔1(prod)↔팔2(prod)
  · `C34-qwen_s`/`C34-qwen_l`=팔3↔팔4 · `C45-qwen_s`/`C45-qwen_l`=팔4↔팔5 · `C6`=팔6
  copy_workers 1↔2 · `C7`=UNCOALESCED↔COALESCED · `C8a`=T1↔T2 · `C8b`=T1↔T4.
- **서로 다른 `contrast_id` 사이 실행 재사용 금지**(팔 8의 T1 셀도 팔 5 셀과 별도 실행).
- ★**Stage B 좌표 총람(r3 — planner 산출의 유일해를 문서가 직접 고정)**:

| contrast | 좌표 구성 | unique | executions(×3) |
|---|---|---:|---:|
| C12-prod | 2팔 × QD{1,4,8,16} × mixed × T1 | 8 | 24 |
| C34-qwen_s + C34-qwen_l | 2팔 × 2stride × QD{1,4,8,16} | 16 | 48 |
| C45-qwen_s + C45-qwen_l | 〃 | 16 | 48 |
| C6 | 2variant(copy_workers 1·2) × QD{4,8,16} × gptoss | 6 | 18 |
| C7 | 2variant(UNCOAL·COAL) × QD{1,4,8} × qwen_run | 6 | 18 |
| C8a | 2T(T1·T2) × QD{8,16} × qwen_l | 4 | 12 |
| C8b | 2T(T1·T4) × QD{8,16} × qwen_l | 4 | 12 |
| R5-ref(단독) | 팔1 × diag × QD8 × T1 × mixed | 1 | 3 |
| SENS(단독) | 팔1 × sensitivity × QD8 × T1 × mixed | 1 | 3 |
| **합계** | | **62** | **186** |

  (paired contrast는 §3-3의 6실행=팔당 3회 규칙과 동일 — 표의 executions는 좌표당 3회 표기.
  planner 산출이 이 표와 불일치=plan 생성 중단.)
- ★**축소 축의 외삽 금지(r2)**: 팔 7의 QD16·팔 8의 QD4 등 matrix에 없는 조합은
  `not_identified/INCONCLUSIVE` — 통합 기본값 동결에 외삽 인용 금지.
- ★**qwen runtime 대조(판독 5) 재정의(r2 차단 2 — 분포 불일치 해소)**: 1차 대조=
  **diag scope 팔 1(prefetch-off·QD8·T1·혼합 trace replay) ↔ 같은 raw run에서 동결한
  engine 실효 R 4.98 reference**(동일 분자·동일 시간창 정의 — §4-2 identity 결속). A2
  qwen 단일-size 두 셀은 **size별 장치 천장 보조 근거**일 뿐 — 서로 평균하거나 팔 1과
  paired ratio를 만들지 않는다(혼합 요청의 동시 서비스 시간은 단일-size 평균으로 보존되지
  않음). A2 직접 비교가 필요해지면 팔 1과 동일한 `read_len` 열·순서·총 useful bytes를
  재현하는 `raw_mixed` 셀을 별도 추가(기본 비활성).
- `io_handles_per_source` 변화는 본 사이클 비스코프(별도 팔 신설 필요 — SPEC_REPACK_V3 §5
  "T2/T4 우세 시 {2,4} 재개봉" 조건부).

### §3-3. paired 비교 실행 계약 (★r1 — "ABBA·3반복" 모호 해소)

- 한 contrast(A팔↔B팔)=**6 실행**: 홀수 번째 contrast는 `A B B A A B`, 짝수 번째는 역순
  `B A A B B A` — 각 팔 3회씩. trace·시드·QD·시작 온도 band를 pair-match.
- 유효 반복이 팔당 3회 미만이면 재실행 후에도 미달 시 해당 contrast=`INCONCLUSIVE`(§6-3).

### §3-4. Stage C — 이중 드라이브 (D-SC1/D-A2 증거 공급)

- 두 물리 NVMe = D:(Kioxia·실물 GGUF) + C:(P41·합성 코퍼스 §4-3).
- exact 유효 셀로: ①solo QD `{2,4,8,16}`(드라이브별 — ★원문 D-SC1 `{2,4,8}`에 16 확장·상충
  목록 ②) ②simultaneous static 50:50 ③dynamic least-outstanding. 기존 D-SC1 사양(direct/
  no-cache·드라이브별+합산 BW·p99·온도·`min RΣ` 판정 — `codex_techdev_decode_r2.md:161-179`)
  재사용·paired 계약=§3-3.
- ★**C/D 동일 요청열 조건**(r1): 두 드라이브에 같은 `read_len`/논리 offset 패턴/source 전환
  열을 준다 — **D: qwen shard source index 0/1의 전환열을 C: corpus source index 0/1에 고정
  전단사로 매핑**(read_len·bracket head·논리 offset 패턴·source transition 열 전부 보존 —
  부속 정오 1-②·`i%2` 임의 교대 금지). corpus extent 수·할당 크기가 실물 GGUF와 크게
  다르면 결과를 "drive+volume+layout" 차이로 표시(순수 SSD 차이 단정 금지).
- ★**Stage C 좌표 총람(규범 — 부속 정오 1-①·코드 교차 r1 문안)**:

| contrast | 좌표 구성 | unique | executions(×3) |
|---|---|---:|---:|
| C-solo | 2 variants(D·C) × QD `{2,4,8,16}` — ★각 QD에서 **D↔C를 하나의 paired contrast**로 §3-3 ABBAAB/BAABBA 적용 | 8 | 24 |
| C-static-dynamic | 2 variants(static 50:50·dynamic least-outstanding) × QD `{2,4,8,16}` | 8 | 24 |
| **합계** | | **16** | **48** |

  Stage C A/A 대표 셀=별도 1 unique/3 executions(총람 합계 불포함). planner 불일치=plan 생성
  중단. ⇒ **완전 본 런 = A(465)+B(186+A/A 3)+C(48+A/A 3) = 705 executions·시간 하한
  7,050s**(구 6,540s는 Stage C 누락값 — 정정). **corpus 부재=quick/main 공히 `PlanAbort`**
  (Stage C 생략 강등 금지 — r1 B4).

### §3-5. 스모크 `--quick`

1반복·completed 상한 8GiB·A1 대표 12셀+A2 4셀+팔 8종 각 1셀+Stage C 1조합. ★부속 정오
4-ⓐ(r2 확정 문안): **`--quick` 측정값은 파이프라인·권한·센서 preflight 및 §8 비게이트 ETA
산출에만 사용할 수 있다. 온도 임계·start band·판독 원천 사용은 금지한다**(임계 원천=
steady-state 캘리브레이션[§6-1]뿐). 판정 인용 금지(judge 무효 태그).

## §4. 입력 데이터 (실물 전제 — 26-08-06 조회 확정)

### §4-1. exact 셀 (★r1 차단 2 — qwen 이원화)

| 값 | 모델 | 소스·비고 |
|---|---|---|
| **5,308,416B**(=3×1,769,472) | qwen122 Q4/Q4/Q4 층(24개) | 실물 GGUF — planner가 헤더+expect로 재검증 |
| **6,119,424B**(=2×1,769,472+2,580,480) | qwen122 Q4/Q4/Q6 층(24개) | 〃. ★48층 평균 5,713,920B는 `display_mean_bytes`로만 기록 — **제출 길이로 사용 금지** |
| 13,369,344B(=3×4,456,448) | DSV4 record | 실물 `deepseek-v4-flash`(145.6GB) |
| 18,923,520B(=3×6,307,840) | K2.6 record | ★합성(원본 부재) — DSV4 shard 위 스트라이드 정렬 오프셋·`synthetic_offsets=true`·**결론 범위=`device_size_only`**(K2 layout·shard·e2e 판단 인용 금지 — r1) |

### §4-2. 팔 1 trace 소스 (★r1 필수 — plan SHA에 신원 고정)

- **고정 항목**: `path`·file SHA/payload digest·wire version·record count·`complete`·
  `dropped_records`·`prefetch_on`·routed scope — plan SHA에 포함. `submit_seq` 순 정렬·
  decode flag만 선택(PA ops wire: layer/expert/submit_seq/flags — `ggml-moe-phasea.cpp:89`).
- **후보 2종은 causal scope가 달라 분리 사용**(r1 실물 확인):
  ①production-shape 판정=`bench_results/a_axis/trace-bench_20260731T144240Z_f893dbed`(검증·
  동결·complete=1·841,040 records — **prefetch_on=true**) ②4.98 causal 대조(판독 5)=
  `bench_results/diag_cpuonly`(prefetch_on=false — r2의 4.98 표본과 scope 일치. ★본 런 전
  validation identity[SHA·complete] 확정=preflight 항목). 두 결과 합산 금지.
- ★**판독 5 engine reference 계약(r3 차단 2 — manifest 필드 동결·값은 preflight 채움)**:
  run manifest에 `r5_reference` 블록 필수 — `path/digest/run_id`(diag 원자료)·선택 decode
  표본 identity·**numerator 필드·산식 식별자**(4.98 원측 판독 스크립트의 파일:행 — §10-6
  preflight에서 확정)·**denominator 시작/종료 필드**(시간창)·단위(GB/s). judge는 이 블록으로
  4.98을 재산출해 대조하고, raw diag 팔 1도 **동일 표본 행·동일 분자·대응 시간창 산식**으로
  산출한다. 블록 미완=판독 5 `INCONCLUSIVE`(fail-closed — 유사 산식 대체 금지).
- 두 후보 모두 부적합 시 합성 fallback(층 순회×top-8 비복원·시드 기록) — 단 **runtime
  underfill·layout 판정 금지**(`INCONCLUSIVE` — 합성은 동등 대체가 아님·r1).
- (권고 2 등재) 여벌 qwen workload trace 1개=sensitivity run — 기본값 동결 판정은 주 trace
  단독·sensitivity는 방향 반전 여부만 보고.

### §4-3. Stage C 코퍼스 (C: P41 — 64GiB·수렴 완결)

- ★**topology 규범(부속 정오 1-② — 코드 교차 r1 문안)**: 총 64GiB = **32GiB fully-allocated
  파일 2개**. 각 파일=하나의 logical source·source당 handle 1개(`io_handles_per_source=1`
  의미 보존·qwen 2-shard source transition 모사). 동일 파일 2회 열기 방식은 본 사이클
  미사용. 전환열 매핑=§3-4 전단사 규범.
- 생성 계약(★r1): **non-sparse·non-compressed·fully allocated**로 1회 생성→flush/close→
  **background write/GC·온도 안정 대기 후** 측정 시작(안정 판정=온도가 시작 band 복귀+C:
  write 카운터 정지). 이후 read-only. extent 수·할당 크기를 manifest에 기록.
- 위치=`bench/techdev/repack_space/ubench/corpus/`(측정 후 사용자 판단 삭제 가능·재생성 수 분).

## §5. 출력 스키마 (★r1 필수 3 — 필드 그룹 동결)

**셀 레코드 JSONL**(append-only) — 5그룹:

- **신원**: `schema_version`·`run_id`·`plan_sha`·`cell_id`·`contrast_id`·`order_position`·
  `stage/arm/block_label/qd/threads/rep`·`seed`·`source_sha/size`·`trace_id/path/digest`·
  `synthetic_scope`(none|device_size_only|synthetic_trace).
- **I/O**: `submit_bytes`·`useful_bytes`·`physical_bytes`·`amplification`·실제
  `read_offset/read_len` 규약(nominal↔정렬 실값)·`qd_records/qd_children/qd_bytes` 시간가중
  평균·최대(적분=SPEC_IO_METRICS_V3 §6 — ★명칭 통일: op-QD 아님 `qd_children`)·
  `handle_count`·`channel_count`·`thread_topology`.
- **계시**(QPC·★r1 개명): `qpc_frequency`·`queue`·`submit_call`(begin/return)·
  **`service_observed`**(ReadFile 반환→event 관측 — 장치 완료시각 아님을 명명으로 고정)·
  `event_observed_to_retire`·`retire`(begin/end)·**`publish`**(record 단위 — Q5 필수 출력
  복원)·(팔 6·7) `copy_queue`·`copy_worker` — 각 count/p50/p95/p99+histogram overflow.
- **drive/env**: 물리 disk identity·volume·firmware/드라이버·per-drive bytes/ops/BW·온도
  start/max/end·link baseline/current·외부 I/O raw delta+ratio(§6-2 산식).
- **폐합**: `valid`·`invalid_reasons[]`·completed/expected bytes·ops·short/error/immediate/
  pending 수·`run_complete`.

**run manifest**: 도구 소스 SHA·plan SHA·전원/코어 파킹/LSPM resolved 값·드라이버·affinity
맵·BitLocker/FS/cluster/extent·HMB(enabled/size/source — 조회 불가 시 `unavailable`)·팬
프로파일·fio sealed reference(§10-1)·전 셀 목록.

## §6. preflight·무효·판독·산출물 (★r1 차단 4 — 순환 제거·폐합)

### §6-0. 판독 1~8 normative 정의 (★r2 차단 2 — judge 구현 계약으로 직접 열거)

| # | 판독 | 종류(§6-3) | 입력 셀 | 판정식(★r3 — 판독별 열거) | INCONCLUSIVE 조건 |
|---|---|---|---|---|---|
| 1 | QD 병렬도 | grid-matched | A1(같은 block·T) | QD 인접쌍(1→2,2→4,4→8,8→16) ratio를 동일 rep sweep 3개 각각 산출 — 3개 전부 band 밖 상승=상승, 전부 band 내=평탄. 평탄화 지점=첫 band-내 인접쌍의 QD | matched observation<3 |
| 2 | byte-depth 지배 | grid-matched | A1 동일 outstanding-bytes 그룹 G1={(512KiB,16),(1MiB,8),(2MiB,4)}·G2={(1MiB,16),(2MiB,8)} | 그룹 내 전 쌍이 3 sweep 전부 band 내=byte-depth 지배 | 〃 |
| 3 | block-size 효과 | grid-matched | 판독 2의 동일 그룹 | 그룹 내 큰 block 우세가 3 sweep 전부 band 밖=per-op/펌웨어 효과 | 〃 |
| 4 | dispatcher 상한 | paired-contrast | C8a·C8b | 같은 전역 QD에서 T2/T4 BW 우세(§6-3 ①) ∧ `event_observed_to_retire` 감소 | paired 유효<3 |
| 5 | qwen runtime underfill | sealed-reference | R5-ref(diag 팔1·QD8·T1) ↔ engine 4.98 reference(§4-2 계약 — 동일 표본·분자·시간창) | raw 3반복 각각이 reference 대비 band 밖 상회(≥7GB/s급)인데 engine=4.98이면 엔진 층(계획·byte-QD·retire) | reference 블록 미완·synthetic fallback |
| 6 | stack 한계 | grid-matched+sealed | A2 qwen_s·qwen_l 각각(QD16·T4) ↔ fio sealed reference 하한(6.5GB/s) | 각 stride별 최상 sweep BW가 fio 하한 대비 band 이상 낮으면 그 stride에서 stack 한계. ★결합 규칙: **둘 다 성립=확정·하나만=partial(사이즈 의존) 표기** | 해당 셀 무효·fio 부재 |
| 7a | ordering 기회 | paired-contrast | C12-prod | ascending 우세(§6-3 ①)=offset ordering 근거 | paired 유효<3 |
| 7b | coalescing 기회 | paired-contrast | C7 | COALESCED 우세(§6-3 ①)=coalescing 근거. ★7a와 **별도 결과**(하나만 우세해도 각자 판정 — r3) | 〃 |
| 8 | fio/정격 층 | sealed-reference | sequential anchor ↔ fio sealed reference(§10-1) | anchor 3반복 각각이 fio 6.5~8.1 범위와 band 내 일치=장치·PCIe·FS 층, 그 위만 엔진 층 | fio reference 부재 |

### §6-1. preflight (하나라도 불가=본 런 시작 금지)

thermal 센서·임계 출처·PCIe link collector·볼륨 외부 I/O 카운터·(§4-2 ②) diag trace
validation identity·commit headroom(**하한 8GiB — 부속 정오 1-④**·pagefile 이동/비활성
금지·r1)·**Stage C corpus 실재·크기 검증**(부재=PlanAbort — §3-4). **온도 임계·시작 온도
band는 steady-state 캘리브레이션(★부속 정오 4-ⓑ — 구 `--quick` 원천 폐지) 후·본 런 plan
SHA 잠금 전에 고정 — 본 런 이후 변경 금지**(사후 조정=판정 순환·r1 차단 4). 스테이지 진입
워밍 precondition=정오 4-ⓒ.

### §6-2. 셀 무효 조건 (판독기 게이트)

`short_read>0` · `error>0` · 목표 duration/bytes 미달 · 전역 QD 초과 관측 · source/plan
identity 변화 · PCIe baseline 대비 gen/width 하락 · 전원/affinity/드라이버 drift · 온도
임계 초과(§6-1 고정값) · **외부 I/O ratio 초과**: `(max(0, 물리 디스크 read delta − 러너
completed bytes) + write delta) / 러너 completed bytes > 1%`(Stage C는 드라이브별 산정) ·
Windows Update/Delivery Optimization/Optimize Drives/Defender/인덱싱의 대상 경로 I/O
검출(§7). 무효 셀=재실행 목록 출력(★부속 정오 4-ⓔ: 이 목록의 선별 재실행은 **closure-invalid
셀에만 성립** — closure-valid·judge-invalid 셀은 러너 resume이 done으로 스킵하므로 새 plan
전량 재실행이 정합 경로).

### §6-3. 판정 규율 (★r3 — 3종 분리)

- stage별 대표 **A/A 셀**을 시작·중간·끝 3회 — `noise_band = max(2%, 최대 |A/A ratio−1|)`.
- **① paired-contrast**(판독 4·7a·7b·Stage C): §3-3의 matched A/B ratio 3쌍 — **전부 같은
  방향** ∧ 최악 ratio도 band 밖일 때만 우세. 아니면 `INCONCLUSIVE`.
- **② grid-matched**(판독 1·2·3·6): 동일 rep의 세 sweep(정/역/무작위) 셀을 matched
  observation으로 삼는다 — 비교 edge/그룹·판정식은 §6-0 표의 판독별 열거를 따른다(paired라
  부르지 않음).
- **③ sealed-reference**(판독 5·8): 유효한 현재 반복 3개 **각각**을 manifest 결속
  reference 산식·범위와 비교(§4-2 r5_reference·§10-1 fio) — paired라 부르지 않음.
- 각 종류 공통: 유효 matched observation <3(재실행 후에도)=해당 판독 `INCONCLUSIVE`.
  `--quick` 산출=판정 인용 금지.

### §6-4. 산출물

- ①Stage A 등고선+판독 8건 판정(각 판정에 근거 셀 인용·판별 전제 미충족 시 `INCONCLUSIVE`
  명시) ②**SPEC_REPACK_V3 §5 파라미터 표** — 항목별 `measured | policy_fixed |
  not_identified` 태그(★r1: 이 러너는 `io_qd_demand_reserve`·`io_qd_prefetch_child_max`·
  coalesce window/gap/buffers·`io_bounce_copy_qd`·`io_handles_per_source`를 **식별하지
  못한다** — `policy_fixed`/`not_identified`로 정직 표기·측정된 것만 `measured`)
  ③D-SC1/D-A2 공급=**`D_A2_MICROBENCH_EVIDENCE`+projection만** —
  `promotion_gate_status=NOT_EVALUATED` 고정(승격 판정은 SPEC_REPACK_V3 §6-2 게이트 전속).

## §7. 환경 통제 (★r1 필수 4 — 이 머신 확정 통제)

- Q5 승계: 대용량 write 직후 측정 금지·write preconditioning 없음·**K3 수신과 본 런 병행
  금지**·머신 배타(selftest 배타 규약 동일)·전원/드라이버/affinity 전 반복 고정.
- **전원·코어**: 전원 계획+CPU 최소 상태+코어 파킹+PCIe Link State Power Management의
  resolved 값을 plan에 기록·반복 간 drift=무효. 러너 스레드는 서로 다른 물리 코어에
  고정(SMT sibling·core 0 회피)·priority 고정(Real-time 금지).
- **백그라운드 억제**: 측정 슬롯 동안 Windows Update/Delivery Optimization/Optimize Drives
  예약 실행 방지(불가 시 §6-2 검출·폐기). Defender=전체 비활성화 대신 D: 모델 경로+C:
  corpus 경로 **슬롯 한정 exclusion**(권한 없으면 해당 프로세스 I/O 검출·재실행). Windows
  Search 인덱싱도 두 경로 제외 또는 검출.
- **C: 특칙**: pagefile 이동·비활성 금지(commit headroom preflight로 대체·OS I/O 1% 초과
  셀만 재실행). corpus 생성→안정화 대기는 §4-3(★"read-only라 SLC 무관" 문구 폐기 — 직전
  64GiB write의 folding/GC·열이 read 런을 오염할 수 있음·r1).
- **HMB**(Kioxia G4=DRAM-less): 생산 기본 상태 유지·임의 변경 금지 — 조회 가능하면
  enabled/size/source 기록·불가면 `unavailable`.
- BIOS 변경·BitLocker 토글 요구 없음(상태 기록만). 팬 프로파일·시작 온도 band 고정.

## §8. 실행 계획 (🧑사용자 슬롯)

- ★소요는 문서 추정치를 쓰지 않는다(r1: v0.1의 "Stage B ≈30분"은 검산 실패 — 문면 120
  좌표/회면 최소 60~80분). **planner가 `unique_coordinates`·`executions_after_repetition`·
  `minimum_seconds`·quick 실측 기반 `eta_seconds`를 산출**하고, §8 시간표는 그 이후 확정.
  **ETA>5h면 사용자 재수렴**(수렴 완결분=Q5 셀 시간 규칙·64GiB 코퍼스 — 총 소요 상한은
  미수렴 상태로 정직 표기).
- 실행 순서(★부속 정오 4 — r3 확정 6단계·각 단계 fail-close 점검 목록=`reviews/
  codex_ubench_gate_r3.md` §요청 3-3 표):
  **0** 수리·synthetic 회귀(warming 마감/완결성·calibration epoch 연속성·stale 문구 0) →
  **1** `plan_quick_r2/`+`plan_calibration_r2/` 생성(수 초 — GGUF 헤더·corpus·`--probe-env`
  실경로가 여기서 처음 밟히므로 조기 실패 지점) → **2** 🧑새 v2 quick 25셀(preflight·비게이트
  ETA — 임계/판독 사용 0) → **3** 🧑calibration **D→C**(세그먼트당 60×60s·중간 resume 금지·
  전 셀 closure-valid·외부 IO≤1%·edge 표본≥5 → `calibrate --confirm` status OK) → **4**
  `plan_main_r2/`+preflight+ETA(thresholds digest 불변·새 plan SHA·705 roll-call·warming
  제외 표기) → **5** 🧑705 본 런 → judge(A/A stage별 3/3·matched≥3·비의도 INCONCLUSIVE 0).
- ★★**정적 감사 종료·실측 전환 선언(r6 확정 문안)**: "두 국소 수리(exit taxonomy·help
  walker)의 표적 회귀 후 **추가 전면 정적 라운드는 종료**하고 1~2단계 실측으로 전환한다.
  1~2단계는 `Planner.build`·실 collector·quick I/O를 검증하며, **실 warming loop 검증은
  thresholds가 결속된 main run 단계에 귀속**한다." ★도달 범위 정정(r6): 1단계=Planner.build·
  GGUF header·corpus·one-shot probe / 2단계=실 I/O·셀별 온도 수집(**band·warming 없음** —
  quick/calibration plan 은 thresholds·warming 이 의도적으로 None) / **실 드라이브 warming
  loop 은 calibration 으로 band 를 만든 뒤 main run 에서 최초 도달**.
- ★**진행 관측 규범(r5 — read 공유 도입과 세트)**: calibration/main 출력은 실행 중 read-only
  열람이 가능하다(writer 배제는 유지). 단 **첫 Stage C 진입 시점부터 JSONL tail을 중단**하고
  콘솔 `[cell]` 진행만 본다 — ⓐStage C 레코드는 오염 방지를 위해 **런 종료 후 일괄 flush**
  되므로 그 구간엔 새 행이 나오지 않고 ⓑ결과 로그가 C: 에 있어 **반복 polling 자체가 Stage C
  의 외부 IO 게이트를 오염**시킬 수 있다.
- ★**프리즈 대응 규범(26-08-07 하드 프리즈 1회·원인 미상·Event 41/덤프 없음)**: 진행은
  감수하되 완화 3종 — ⓐ새 v2 quick 을 canary 로 먼저 완주 ⓑcalibration 은 재부팅 직후·
  sleep/update/병행 작업 없는 슬롯에서 ⓒD 구간만 관측. quick 전후로 WHEA·disk/volmgr 오류가
  새로 생기면 calibration 진입 중단. **2회째 발생 시 즉시 실측 HOLD** 전환하고 dump/pagefile·
  전원·RAM·firmware/driver 순으로 원인 수집(장시간 epoch 추가 시도 금지). ★calibration 의
  D/C epoch 분리는 완화책으로 **기각**(r5 — 단일 full-plan 계약 재개봉이라 provenance·
  coverage·digest·gap 주장 변경이 세트로 필요).

## §9. 시공·검증 순서

1. 본 v0.2 → Codex r2 재교차(max — r1 11건 폐합+신규 스캔) → [ACCEPT] 동결.
2. 시공=Opus 위임 1건 → 메인 diff 검수 → Codex 코드 교차.
3. (★부속 정오 4 — r3 확정) §8 실행 순서 0~5 이행 → judge → **SPEC_REPACK_V3 §5 기본값
   동결**(Codex 교차·§9-4 게이트 수치 사양 입력) → 장부 SP-A/D-A2/D-SC1 절 갱신.

## §10. 미결

1. **fio sealed reference**: 기존 fio 6.5~8.1 실측의 config/원본 결과 경로·SHA를 manifest에
   결속 — 본 런 전 preflight에서 확정(부재 시 판독 8=`INCONCLUSIVE` 유지·러너는 sequential
   anchor 자체 실측으로 대체 보고).
2. `diag_cpuonly` trace의 validation identity(SHA·complete) 확정=preflight 항목(§4-2).
3. throttle 임계=steady-state 캘리브레이션 후 고정(★부속 정오 4-ⓑ — 구 `--quick` 원천
   폐지·§6-1) — 문서엔 수치 미기재가 정상.
4. (권고 1 종결) 실물 qwen `experts.bin` 부재(디스크 재팩본 0 — 26-08-06 실사) → engine-
   layout 대조군 `3b` 미채택·팔 3 개명으로 종결. bin 재생성 시 재개봉 가능.
5. (권고 2) sensitivity trace 후보 선정=본 런 전 preflight에서.
6. ★**판독 5 분자·산식 원천(r3 → ★부속 정오 2-ⓑ로 확정)**: ~~r4 정적 식별 후보
   (events_decomp.py 합산 공식)~~는 **기각** — 수리 재교차 r2가 diag 실물에 적용해
   6.852GB/s를 얻어 4.98과 불일치 확인. **실원천=`bench_results/decode_hunt/
   events_seal.json:98-120` 계열 seal·per-token p50 공식** — r5_reference는 이
   공식·소스로 결속하고 raw 팔도 대응 time-window join으로 동일 정의 산출(세부 구현=수리
   2차·확인=r3).
