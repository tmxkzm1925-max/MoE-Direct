> ---
> **[Published design note - as-is working record]** A page from the developer's lab notes,
> published with the original Korean body preserved unchanged. The design and the decisions
> here are the developer's; AI assistants draft and cross-check under his direction.
> "FROZEN" means the document text was locked after multi-round cross-review;
> implementation and measurement status, including deferred items, is stated in each note.
> References to internal files (HANDOFF, SESSION_STATE, reviews/) point to the private
> workspace and are left in place on purpose - they show how the records are kept.
>
> **TL;DR:** A preregistered four-prompt A/B protocol requiring identical raw token-ID and generated-text digests, stop outcomes, and determinism metrics between the virtual and baseline paths.
> ---

# SPEC_AB_CORRECTNESS_R1 — SP-A_MODE_GATE ⓐ 전용 correctness 하위 프로토콜 (v0.4 · 26-08-14 · 작성 리드 Fable · r21~r23 전건 반영)

> ★★**동결 스탬프(리드 기입 26-08-14 03:5x)**: **v0.4 = FROZEN** — r24 **[ACCEPT — 동결
> 가·신규 결함 없음]**(`reviews/codex_ab_correctness_r24.md`). 이 스탬프 기입 커밋부터
> 본 파일 무수정(§8) — 결과는 별도 RESULT 파일로만.

> **지위**: `SPEC_REPACK_V3.md` §6-1 SP-A_MODE_GATE 의 **ⓐ(digest 완전 동일) 축 전용 하위
> 프로토콜**. ⓒ 비열화 밴드·성능 판정과 **완전 분리**(r20 Q1 — 이 런의 성능값은
> `diagnostic_only` 영구 봉인). **v0.4 = r21 Q1~Q5 대체 문면 + r22 잔여 5건 + r23 신규
> BLOCKER 2건 전건 수용본**(전사=`reviews/codex_ab_correctness_r{21,22,23}.md`).
> **동결은 r24 [ACCEPT] 후에만** — [ACCEPT] 시 리드가 이 머리에 동결 스탬프(버전·라운드·
> 커밋)를 기입하며, 그 기입이 §8 "동결 후 무수정"의 시작점이다. matched pair 실행은 그 뒤.
> digest 판정축·비교 조건은 `SPEC_IO_METRICS_V3.md` §2(r1 B5) 동결 문면 승계(새 판정축
> 신설 없음).

## §1. 대상 짝 (qwen35-122b-nonextn — A/B 1호)

| 팔 | 아티팩트 | identity |
|---|---|---|
| A=bin | `D:\moe-models\qwen35-122b\repack_nonextn\`(experts.bin+manifest v2) | manifest SHA `fc3e588d9928db2f4d6251b061ca7c011658c16148cdf2f9db432f398acbecff` · slot stride **6,119,424** |
| B=virtual | `D:\moe-models\qwen35-122b\repack-virtual-nonextn\`(manifest v3·experts.bin 없음) | manifest SHA `c6abae64a21b3fdb3346a76ceb6bc6a03b98e374316a32ab380e490cd4e02fb5` · `legacy_align_source=paired_v2`(A manifest SHA 결속) · slot stride **6,131,712** |

- 소스 identity: 두 팔 같은 GGUF shard 2개(경로·bytes·mtime·헤더 digest — manifest/plan_report
  기록·부팅 fail-close 대조).
- **P-1(충족 26-08-14 02:54 KST)**: bin fresh `--verify-only` PASS — verify_report 2번째
  레코드(36,864/36,864·manifest SHA 불변). 채록 의무=§7-2.
- **P-2**: 준비 슬롯 대용량 I/O 후 **장치 안정화 대기**는 §1 권고(성능 교란 방지)이며
  correctness 무효 사유가 아니다(§6 — r21 Q4).

## §2. 실행 상수 (직접 기동·env 계약 — r21 Q1·Q3 대체 문면)

- **launcher 사용 금지 — 직접 기동**. 근거: launcher 는 이 프로파일의 catalog-fixed K8/N4 를
  K/N env 로 주입하는데, virtual 은 K/N raw key 존재 자체를 기동 거부한다(r21 Q1 인용).
- **엔진**: `D:\moe-tools\llama.cpp-src-b10057\build-moedirect-s0-cuda\bin\llama-server.exe`
  (SHA-256 은 §7-1 prereg plan 의 실값). 현행 트리=`BASELINE_TREE_CHILDSCHED.md` §B U3 after.
- **argv(동결·`-m` 명시 포함 — ★코드 스팬 무개행: 아래 한 줄이 실값이다)**:
  `-m "D:\moe-models\qwen35-122b\Qwen_Qwen3.5-122B-A10B-Q4_K_M\Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf" -ngl 99 --n-cpu-moe 49 -c 12288 -t 8 -b 2048 -ub 512 -fa on -np 1 --host 127.0.0.1 --port 8093 --no-webui --no-warmup -lv 4`
- **env 계약**: child 환경은 호스트 환경에서 **모든 `MOE_*` 및 `GGML_BACKEND_PATH` 를 제거**한
  뒤 아래 allowlist 만 재주입한다(★각 실값 코드 스팬은 무개행 한 줄 — r22 BLOCKER 1 반영).
  - 공통: `MOE_DIRECT=1` · `MOE_DIRECT_EXPECTS_DIR=C:\Users\tmxkz\Claude\Projects\moe-expert-stream\bench\repack\expects` · `MOE_DIRECT_QD=8` · `MOE_NO_PREFETCH=1`.
  - 팔별: `MOE_DIRECT_DIR`(§1 팔별 디렉터리) · `MOE_DIRECT_METRICS`(아래 규칙) · A `MOE_DIRECT_BUDGET_MB=8192` · B `MOE_DIRECT_BUDGET_MB=8205`.
  - ★**`MOE_DIRECT_METRICS` 규칙(r22 BLOCKER 2 반영)**: `<attempt_id>\<arm>.jsonl` 의
    attempt×arm 고유 경로로 한다. **대상이 이미 존재하면 삭제하지 않고 preflight abort** 하며,
    prereg 이후 생성된 모든 시도 아티팩트는 불변 보존한다(사전 삭제 없음 — CREATE_NEW 는
    고유 경로로 충족).
  - K/N·PhaseA/B·IO_PATH·QUALIFY·trace·coalescing·전용 `MOE_DIRECT_IO_*`·IO3 계열은 **raw
    key 부재**로 고정한다(미설정=off).
- ★**budget 비대칭의 근거(동일 `n_slots` 조건 — B5)**: `n_slots=floor(budget_bytes ÷
  slot_stride_max)` 인데 두 팔 stride 가 다르다(§1 표). 8192MiB 면 A=1403·B=1400 으로
  갈라진다(비교 부적격). **A 8192MiB→1403 · B 8205MiB→1403** 로 등가화한다(산술 검증:
  8,589,934,592÷6,119,424=1403.7→1403 · 8,603,566,080÷6,131,712=1403.1→1403).
- **header 적격 검사(양팔·§7-4 채록)**: `slot_count=1403` · `prefetch_on=false` ·
  `io_path=persistent` · `io3_enabled=false` · A: `qd_effective=8`/`qd_source=env` ·
  B: `qd_effective=8`/`qd_source=legacy_alias` + `io_params.io_qd_total={value:8,
  source:"legacy_alias"}`. `b1_mem.slot_pool_bytes` 는 stride 차로 팔 간 동일값을 요구하지
  않는다 — 판정은 header 의 **`slot_count` 직접 필드**로 한다(r21 Q3).

## §3. 요청 계약 (결정론 입력 고정 — r21 Q2 대체 문면)

- ready 후 **`/props`** 응답의 `total_slots=1`·`model_path`·`build_info`·`chat_template`
  SHA-256 을 채록·양팔 교차 확인한다.
- 고정 messages 4건(`ab_prompts_r1.json` — SHA 는 §7-1 plan 실값)을 각각 **`/apply-template`**
  에 넣고 반환 prompt 의 UTF-8 SHA-256 을 양팔 비교한다. 같은 prompt 를 **`/tokenize`**
  (`add_special=true`·`parse_special=true`)에 넣어 **입력 token ID 전열**도 양팔 비교한다.
- 실제 추론은 **`/completion`** 에 정확히
  `{"prompt":<rendered>, "n_predict":128, "temperature":0, "top_k":1, "seed":7,
  "cache_prompt":false, "stream":true, "return_tokens":true}` 를 보낸다(`cache_prompt=false`
  =KV 재사용 차단으로 동일 record 순서 보장·`return_tokens=true`=raw token 열 수신).
  이전 요청의 terminal SSE 수신 후에만 다음 요청을 발행한다(직렬). 추론 요청은 팔당 정확히
  4개 — 다른 클라이언트·워밍업 요청 금지(§7-5).
- 각 팔 = fresh 프로세스 1회 기동 → 위 사슬 → **graceful shutdown**(§7-6 clean terminal).
  팔 순서 A→B 1쌍.

## §4. 판정축 (r21 Q1 대체 문면 — B5 승계)

**primary equality bundle** (요청별+summary — A/B 완전 동일=필수):
1. **token digest**: 응답 **raw token ID 배열**의 공백 없는 canonical JSON UTF-8 SHA-256
   (재토크나이즈 금지 — `return_tokens` 원열).
2. **content digest**: 순서대로 연결한 non-terminal SSE `content` 의 UTF-8 SHA-256.
   terminal stop 정보(stop 종류·stopping word 유무)도 A/B 동일해야 한다.
3. **metrics 결정론 묶음**(terminal summary): `moe_tick_final` · `moe_lru_tick_digest` ·
   `moe_graph_nodes_digest` · `moe_graph_splits_digest` · `moe_graph_count`.

**별도 공통 논리 불변식**(A/B 동일=필수 — layout 무관 논리 계수): prefill/decode 의
`hit`·`miss`·`read_count`·`read_bytes`(virtual 도 legacy_record_bytes 로 계수 —
`SPEC_IO_METRICS_V3` §2 계약).

**구조 건전성**(양팔 각자): `fallback_count=0` · `touch_events=0` · `touch_pages=0` ·
`dispatcher_abort_terminal=0`.

**mode-local 제외(교차 비교 금지)**: `moe_readfile_digest`/`moe_readfile_ops`(bin=record-op
16B 축·virtual=child-op 18B 축으로 재정의 — r21 Q1 인용) · virtual child 물리 계수 ·
`moe_pb_*`(PB off) · `io3_*`(off) · 모든 시간·속도 필드(§5 봉인).

**LRU 비교 적격 조건(r23 BLOCKER 2 대체 문면)**: 양팔 header `prefetch_on=false` ∧
`slot_count=1403` ∧ §3 입력·요청 receipt 동일 ∧ §7-6 clean terminal. **첫 추론 요청 전에
확인 가능한 header·입력·요청 조건 불충족은 preflight abort** 로 별도 기록하고, 교체는 전체
pair 로만 한다. **정상 closure 후 확인된 적격 조건 불충족은 즉시 FAIL** 한다. **clean
terminal 불충족만 §6 의 closure failure 무효**로 분류하며, 적격 실행의 digest 불일치도
즉시 FAIL 한다.

## §5. 성능값 봉인 (r21 Q4 대체 문면)

서버 stdout/stderr 와 원본 metrics 는 시작 시부터 파일로 redirect 하고 해시 봉인한다.
correctness extractor 는 allowlist 된 identity·건전성·digest 필드만 읽으며 시간·속도·wait
수치를 화면·판정 보고서에 노출하지 않는다. 이 실행에서 생성된 성능값은 이후 threshold·
confirmatory 성능 판정에도 **영구 제외**한다.

## §6. 무효·FAIL·교체 (r21 Q4 대체 문면)

- 최초 pair 외 최대 **2개의 전체 pair replacement** 만 허용한다(최대 3 pair). **한 팔만
  다시 돌리지 않는다**. 모든 시도는 attempt ID·artifact hash·자동 분류 사유와 함께 불변
  보존한다.
- 정상 비교 가능한 실행에서 digest 불일치 또는 구조 건전성 불일치는 **즉시 FAIL** — 재실행
  으로 덮을 수 없고, FAIL 은 이후 PASS 보다 우선한다.
- **무효**는 closure failure 에 한정한다: 요청 완료 전 프로세스 crash/HTTP 단절 · terminal
  summary 부재 · 강제 종료. 사전 identity·CREATE_NEW 검사는 요청 전 **preflight abort** 로
  별도 기록한다. **외부 I/O 는 무효 사유가 아니다**(안정화는 §1 권고).
- 두 replacement 후에도 valid pair 가 없으면 **INCONCLUSIVE**.

## §7. 필수 채록물 (r21 Q5 대체 문면 — evidence manifest)

1. 첫 팔 실행 전 **prereg plan** 을 생성한다: protocol(이 문서)·engine exe·runner·extractor·
   `ab_prompts_r1.json`·양 manifest·virtual `plan_report.json`·expect 파일의 SHA-256, 전체
   argv/env, 팔 순서, 예상 `slot_count=1403`, request payload SHA — 전부 **plan 안의 실값**
   ("추후 채록" 금지). plan 자체 SHA 를 봉인한다.
2. fresh bin verify 마지막 JSONL record 의 SHA·offset 을 보존하고 `pass=true` ·
   `pairs_total=pairs_pass=expected_pairs=36,864` · 오류 배열 3종 empty · reference_lock·
   manifest SHA 일치를 확인한다. **그 fresh verify 는 source GGUF shard 와 `experts.bin` 을
   재독하여 per-layer part SHA 를 비교한 `--verify-only` 실행이어야 한다**(r22 Q5 보완).
3. verify 직후와 A 종료 후 `experts.bin` 의 절대경로·bytes·mtime 을 기록하고 **불변**을
   요구한다(엔진은 기동 시 크기만 재확인 — r21 Q5 인용). virtual 은 `plan_report.pass=true`·
   records 36,864·paired v2 manifest SHA 를 기록한다.
4. 각 metrics 파일에서 **정확히 하나의 header·하나의 terminal summary·단조 `seq`·summary=
   마지막 record** 를 확인한다. header 의 mode/schema/manifest SHA/reference lock/routed
   scope/slot_count/QD/prefetch/phase/io-path/IO3 echo 를 예상값과 대조한다(periodic record
   허용).
5. `/props`·`/apply-template`·`/tokenize` 응답과 4개 completion request/SSE 원문을 팔별
   보존한다. 추론 요청은 정확히 4개 — 다른 클라이언트·워밍업 요청 금지.
6. **clean terminal 정의**: graceful shutdown·exit code 0·강제 kill 부재·terminal summary
   존재(서버 정상 cleanup 이 finalize 를 호출한다 — r21 Q5 인용).
7. stdout/stderr 원문은 성능 봉인 영역에 보존하고 SHA 만 공개한다. correctness 용 오류·경고
   allowlist 추출본과 extractor SHA 를 결과에 포함한다.

## §8. 판정 지위·결과 기록 (r22 BLOCKER 3 반영)

- ⓐ PASS 시에도 **SP-A_MODE_GATE 전체 PASS 아님**(ⓑ fail-close 전건 실증 지속·ⓒⓓ는
  `SPEC_REPACK_V3` §9-4 동결 후 confirmatory A/B 소관 — r20 Q1).
- ★**본 protocol 파일은 동결 후 수정하지 않는다**(prereg plan 이 이 파일 SHA 를 봉인한다).
  결과는 별도 `AB_CORRECTNESS_R1_RESULT_<attempt_id>.md`(+동형 `.json`)에 기록하고, 그
  결과물과 frozen protocol 의 SHA 를 `HANDOFF_DEV.md` 에 포인터로 남긴다.
