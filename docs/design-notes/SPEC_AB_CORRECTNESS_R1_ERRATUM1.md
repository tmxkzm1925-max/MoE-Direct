> ---
> **[Published design note - as-is working record]** A page from the developer's lab notes,
> published with the original Korean body preserved unchanged. The design and the decisions
> here are the developer's; AI assistants draft and cross-check under his direction.
> "FROZEN" means the document text was locked after multi-round cross-review;
> implementation and measurement status, including deferred items, is stated in each note.
> References to internal files (HANDOFF, SESSION_STATE, reviews/) point to the private
> workspace and are left in place on purpose - they show how the records are kept.
>
> **TL;DR:** Frozen erratum 1 to the correctness protocol: evidence registration details.
> ---

# SPEC_AB_CORRECTNESS_R1 부속 정오 1 (26-08-14 · 리드 Fable)

> ★★**동결 스탬프(리드 기입 26-08-14 05:5x)**: **FROZEN** — r27 **[정오 1 동결 가]**
> (`reviews/codex_ab_runner_r27.md` — 1-ⓐ~ⓓ 전건 적합 판정). 이 스탬프가 마지막 편집이며
> 이후 무수정. 판정 의미 변경은 후속 정오로만(1-ⓑ 유효 SHA 기제로 자동 신규 캠페인).

> 본체=`SPEC_AB_CORRECTNESS_R1.md` v0.4 FROZEN(무수정 원칙 — 개정은 이 부속 정오로만·선례
> =SPEC_REPACK_V3 부속 정오 체계). 계기=r25 코드 감사 D4: §7-1 의 "request payload SHA 를
> plan 안의 실값으로"가 **문자 그대로는 이행 불가**함을 리드가 확인 — full `/completion`
> payload 는 서버가 렌더링한 prompt 를 포함하는데, plan 생성 시점(첫 팔 실행 전)에는
> 서버가 없다. 사양 결함이지 구현 재량 사항이 아니다(시공이 무단 재해석한 것은 별도 통제
> — 정오는 그 재해석의 사후 승인이 아니라 문면의 이행 가능한 재정의다).

## 정오 1-ⓐ — §7-1 "request payload SHA" 의 이행 가능 재정의

- plan 실값 의무(사전 고정 가능한 전부): ①4개 `/apply-template` 요청체(messages 원문)의
  canonical JSON SHA-256 ②`/completion` 고정 파라미터 블록(`n_predict`·`temperature`·
  `top_k`·`seed`·`cache_prompt`·`stream`·`return_tokens`)의 canonical JSON SHA-256
  ③`ab_prompts_r1.json` 파일 SHA-256.
- full `/completion` payload SHA(렌더 포함)는 **각 요청 발행 직전 receipt 에 채록**하고,
  extract 가 ①A/B 동일 ②고정 파라미터 블록이 plan ② 값과 일치 ③prompt 부분이 그 팔의
  `/apply-template` 채록 렌더값과 바이트 일치 — 3중 대조로 판정한다. 셋 중 하나라도
  불일치=즉시 FAIL(적격 실행 기준).
- 취지 보존: 조작 가능한 자유도(메시지 원문·파라미터)는 전부 사전 봉인되고, 서버 산출
  렌더값은 교차 대조로 잠긴다 — "추후 채록 금지"의 방어 대상(사후 payload 조정)은 그대로
  차단된다.

## 정오 1-ⓑ — §6 캠페인 정의 명문화 (★r26 대체 문면 반영 — 유효 프로토콜 식별자)

- **유효 프로토콜 식별자**: prereg plan 은 본체 protocol SHA 와 **적용 부속 정오 SHA 전부를
  순서대로** 봉인하고, 그 ordered bundle 의 SHA-256 을 `effective_protocol_sha256` 으로
  봉인한다. arm preflight·extract 는 본체와 정오 전부를 **재해시**해 대조한다. RESULT 에도
  본체·정오·유효 SHA 를 직접 기록한다.
- **캠페인 = `effective_protocol_sha256` 이 같은 attempt 전부**(`ab_attempts/` 하위). 다른
  유효 SHA 의 attempt 는 서로 격리된다. **판정 의미를 바꾸는 후속 정오는 유효 SHA 를
  바꾸므로 기계적으로 새 캠페인이 된다**(별도 리셋 선언 불요 — 자동 격리).
- 집계 강제(기록이 아니라 차단): ①유효 pair 의 FAIL 이 캠페인에 존재하면 이후 arm 실행은
  preflight abort(사유="campaign already FAIL")·캠페인 판정은 영구 FAIL ②소비된 pair 가
  3에 도달하면(최초+교체 2) 이후 arm 실행은 preflight abort·유효 pair 부재 시 캠페인
  INCONCLUSIVE ③★**pair 소비의 정의(r26 대체 문면)**=어느 한 팔이라도 첫 `/completion`
  의 **transport 호출 직전 durable dispatch-intent(소비 마커)가 기록된** attempt — 이후의
  연결 실패도 소비다(선기록=과소집계 불가 방향의 fail-close·예산은 보수적으로 소모된다).

## 정오 1-ⓒ — §4 사전 관측 게이트의 관측 수단(구현 규정 · ★r26 대체 문면 반영)

- "첫 추론 요청 전에 확인 가능한" header 조건의 관측 수단: ⓐmetrics JSONL header 행
  read-only 직독(재시도 포함) → ⓑ불가 시 서버 로그의 seal 줄 파싱(★seal 줄은
  `slot_count` 만 관측 가능 — 나머지 필드는 명시적 unobservable 기록 후 사후 FAIL 백스톱).
- ★**관측 불가와 관측된 위반의 경계**: 첫 완전 행이 header 가 아니거나 JSON·필수 필드·
  값이 부적합하면 그것은 **관측된 위반 = 즉시 preflight abort** 다. **OS-level read 불가
  또는 재시도 후에도 완전 행이 없는 경우만** 관측 불가이며, 그때만 §4 의 사후 FAIL 경로가
  적용된다(무확인 진행이 아니라 관측 불가의 정직 기록 의무).
- ⓐ가 성공하면 게이트는 **live header 에서 관측 가능한 §7-4 기대 조건 전부**를 검사한다
  (slot_count·mode·prefetch_on 에 한정하지 않는다 — 관측했는데 안 본 필드가 pair 를
  소비시키는 것을 금지).

## 정오 1-ⓓ — §8 RESULT 불변 보존(구현 규정)

- RESULT `.md`/`.json` 은 write-once. 존재 시 덮어쓰기 금지 — 재판독은
  `AB_CORRECTNESS_R1_RESULT_<attempt_id>__x<N>.md` 연번 신규 파일로만(전 판본 불변 보존).
