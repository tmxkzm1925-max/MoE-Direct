> ---
> **[Published design note - as-is working record]** A page from the developer's lab notes,
> published with the original Korean body preserved unchanged. The design and the decisions
> here are the developer's; AI assistants draft and cross-check under his direction.
> "FROZEN" means the document text was locked after multi-round cross-review;
> implementation and measurement status, including deferred items, is stated in each note.
> References to internal files (HANDOFF, SESSION_STATE, reviews/) point to the private
> workspace and are left in place on purpose - they show how the records are kept.
>
> **TL;DR:** Frozen erratum 1 to the e2e protocol: external-I/O gate demotion, implementation rules, and the missing-copies disposition.
> ---

# SPEC_E2E_PARAM_R1 부속 정오 1 (v2 · 26-08-14 · 리드 Fable)

> ★★**동결 스탬프(리드 기입 26-08-14 13:5x)**: **v2 = FROZEN** — r42 **[정오 동결 가]**
> (`reviews/codex_e2e_runner_r42.md`). 이 스탬프가 마지막 편집·이후 무수정. 판정 의미
> 변경은 후속 정오로만(effective SHA 기제로 자동 신규 캠페인).

> 본체=`SPEC_E2E_PARAM_R1.md` v0.12 FROZEN(무수정 — 개정은 부속 정오로만). 계기=러너
> 시공 질문 8건 중 본체 문면이 미결정이던 지점의 성문화(구현 재량 승인이 아니라 리드
> 판정의 등재). 시공 전사=e2e-runner-impl 납품 보고 26-08-14.

## 정오 1-ⓐ v2 — 외부 I/O 게이트의 ⑶ 지위: **비게이팅 진단으로 강등** (r41 #2 반영)

본체의 external_io_ratio 정확식은 ubench(러너 자신이 유일한 I/O 주체) 계보라 서빙 런에
건전하게 이식할 수 없음이 2단으로 확정됐다: ①전체 창 적용=서버 자신의 dense 로드(mmap·
수 GB)가 "외부"로 오분류(전 cell 무효화 — 도달 확실) ②v1 이 시도한 프로세스 IO 카운터
분모도 불건전 — `GetProcessIoCounters` 는 파일·네트워크·장치 전체 범위라 대상 PhysicalDrive
귀속을 보장하지 않고, mmap hard fault 는 별도 메모리 관리 경로의 디스크 읽기다(r41 #2 —
MS 1차 문서 근거·전사 참조). 건전한 측정량(target-volume 프로세스 물리량 또는 엔진 내부
completed physical child bytes)은 현 계측 표면에 없다.
**재정의(v2)**: ⑶ 성능런에서 외부 I/O 자동 게이트는 **비게이팅 진단 기록**이다 —
- 채록 3종(요청 구간 창): 물리 디스크 카운터 델타 · 서버 프로세스 IO 카운터 델타 ·
  metrics 논리 read_bytes. **cell 무효 조건에서 제외**(본체 §3 의 ">0.01 block 무효" 는
  ⑶ 에 적용하지 않는다).
- 오염 방어의 집행 수단: ①운영 전면 배타(§2-1 — 착수 전 상위 프로세스 전수 확인·단독
  머신) ②sentinel A/A 드리프트 감시(성능 오염의 실질 백스톱 — NONSTATIONARY 게이팅 유지).
- 재승격 조건: 엔진이 physical child bytes 를 mode 공통으로 노출하거나 target-volume
  프로세스 귀속 계측이 확보되는 시점(차기 엔진 라운드 후보).

## 정오 1-ⓑ — 구현 규정 성문화 (시공 질문 처분)

1. **N=12 실행열** = 본체 6열 Williams superblock 을 **동일 순서로 2회 반복**(각 QD 가
   각 시간 위치에 2회 — 균형 유지).
2. **슬롯② sentinel 의 모델 = 그 슬롯의 모델(gpt-oss)** — "부하 세트 동일" 은 슬롯 내
   기준이다(sentinel 은 그 슬롯 장치 상태 감시).
3. **sentinel 봉인 시점**: 슬롯①의 S_start 는 c0_plan 이, S_mid·S_end 는 plan-main 이
   봉인한다 — 둘 다 해당 sentinel **실행 전** 봉인이므로 본체 r36 인과성 조항과 정합
   (S_mid 위치는 기본 원자 수(5+N)가 확정되는 plan-main 시점에만 계산 가능).
4. **시작 온도 band 밖 처분**: 자동 대기 정책은 창작하지 않는다 — 기본=중단(재시도는
   운영자 몫). `--cooldown-wait-seconds` 는 **시작 게이트 재검사 대기**로만 허용(plan
   불변·측정 의미 무영향).
5. **budget_mb 실값**: qwen nonextn=**8205**(slot_count 1403 — a1 correctness B팔과 동일
   기하·§9-4 비교 가능성) · gpt-oss=**8192**(스모크 실증 계승).
6. **ⓐ/ⓑ 경계**: 기동 전 identity 드리프트·첫 `/completion` 전 header 게이트 불일치=
   ⓐ(PLAN_ABORT·cell 미소비) / 측정된 cell 간 digest 불일치=ⓑ(INTEGRITY_FAIL).

## 정오 1-ⓒ — E3 copy 축 관측 불능의 지위 (엔진 계측 결손 등재)

엔진 IO3 copies emitter 결손(평시 바운스 경로에서 copies 행 바인딩 미배선 —
`ggml-moe-direct.cpp` `io3_chan_copy_row` 전 대입이 -1·호출부 :6193 out_copy_row 미전달·
스모크 r2 실물이 격발 증거)으로 **⑶의 E3 진단에서 copy GB/s·p99 는 구조적으로 null**
이다. 처분:
- 판독기는 **비게이팅 anomaly 2종**(`bounce_copy_obligations_never_transitioned`·
  `copy_axis_unobservable`)을 RESULT 본문까지 올린다(null 을 "빠름"으로 오독 방지).
- E3 는 본체 문면대로 완료 가능("diagnostic absent" 계열 정직 기록) — **E3 무효화하지
  않는다**(copy 축은 §6-1 ⓓ confirmatory 의 입력이지 ⑶의 판정 입력이 아님).
- ★**엔진 수리 = 차기 엔진 라운드 필수 선행**(⑹ gpt-oss confirmatory 는 ⓓ 바운스 비용
  을 요구한다) — seal anchor unfiltered marker 1줄 후보와 같은 라운드 묶음.
