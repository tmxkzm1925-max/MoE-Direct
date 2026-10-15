#pragma once
// ggml-moe-direct-internal.h - test-only entry points, split out of the public header by
// R-B3a. The block below is the public header's test-only region, moved BYTE FOR BYTE:
// RB3A_ATOM_SPEC.json records its sha256 and rb_move_proof.py compares it on every leg, so
// an edit here is a failed proof and not a diff somebody has to notice.
//
// INCLUDE ALLOWLIST (RB_DESIGN 2-0 premise (b), role-based): the translation unit that
// DEFINES these entry points, the selftest monolith, the selftest harness, and the R-B3b
// test TU. No other translation unit may include this header - rb_static_gate.py layer r3-1
// walks the tree and fails on any other consumer.
//
// The declarations keep GGML_API: they are exported by ggml-base today and R-B3a does not
// move the export set. That is what makes this header a different contract from E-2c's
// ggml-moe-direct-test.h, whose six demoted names are selftest-local plain extern "C".

#include "ggml-moe-direct.h"

#ifdef __cplusplus
extern "C" {
#endif
// ===========================================================================
// 테스트 전용 진입점 — 프로덕션 코드는 아래 심볼을 절대 호출하지 마라.
// ===========================================================================

// dir 아래 v2 manifest.json + experts.bin(합성)을 읽어 seal 로직을 실행한다. sector 재질의는
// IOCTL 실패 시 GetDiskFreeSpaceW 폴백을 허용한다(프로덕션 seal()은 폴백 금지). model/sources는
// 합성 manifest와 정합해야 한다. budget_bytes로 슬롯 수를 정밀 제어(슬롯 용량 384/383 케이스).
// expects_dir=합성 expect 디렉토리, test_profile_id/test_digest=테스트 전용 단일 엔트리 카탈로그
// 주입(프로덕션 EXPECT_CATALOG 우회 아님 — F1 reference_lock 결속 행사). 성공=0, 실패(<0).
// test_scope: ★부록A — 테스트 전용 카탈로그 주입 엔트리의 scope("all"|"execution", null/빈="all").
// reference_lock 검증의 카탈로그 scope == manifest routed_scope 대조를 합성 픽스처로 행사한다.
GGML_API int ggml_moe_direct_seal_for_test(
    const char * dir,
    const struct ggml_moe_direct_model *  model,
    const struct ggml_moe_direct_source * sources,
    int32_t                               n_sources,
    uint64_t                              budget_bytes,
    uint32_t                              touch_abort,
    const char *                          expects_dir,
    const char *                          test_profile_id,
    const char *                          test_digest,
    const char *                          test_scope);

// 모듈 전역 상태를 완전히 초기화(selftest 전용 훅 — 여러 시나리오 순차 실행용).
GGML_API void ggml_moe_direct_reset_for_test(void);

// ★§Z-⑥ — 직전 seal 시도가 mode=virtual 관문을 어디까지 통과했는지(관측 전용).
//   0=v3 경로 미진입(v2 manifest 이거나 모드 게이트에서 거부) · 1=mode virtual 확정 ·
//   2=init 거부 4조합 통과(SPEC_IO_METRICS_V3 §3) · 3=manifest v3 검증 통과(§2-2·§2-4) ·
//   4=reference_lock 결속 + experts.bin 부재 통과 · 5=plan_report 소비 결속 통과(§4-3) ·
//   6=A_runtime 사다리 통과 · 7=등록 3집합(M_all==L_scope) 통과 ·
//   8=§2-5 live GGUF 재파싱 + DF-1 digest 통과(**A1 완주**) ·
//   9=N-shard 소스 핸들 집합 확보(§3-1) · 10=슬롯 풀 확보 + 레이아웃 확정(§4-7) ·
//   11=registry·attestation·보호 확정(**A2 완주**) ·
//   12=QD 팬아웃 하한·VEH·dispatcher 기동(**A3 완주 = seal 성공**).
//   프로덕션 경로는 이 값을 읽지 않는다.
GGML_API int32_t ggml_moe_direct_test_v3_stage(void);

// ---------------------------------------------------------------------------
// OPEN_ARCH B axis machine-readable probe (OPENARCH_B_SPEC_DRAFT v0.2 section 2-5 / D2).
// Returns a static JSON string (valid for the process lifetime, never freed):
//   {"abi":"open-arch-template/1","templates":[{"template_id","template_version","weight_kinds",
//    "bias","layer_rule","nextn"}, ...]}
// Consumer split: the abi field is FOR THE M5 ASSEMBLER (the A/B/C identity comparison is the
//                 assembly gate's job) - the templates rows are FOR GATE G (table synchronisation
//                 against the A-axis ARCH_TEMPLATES).
// It is not used for seal-time wire validation.
GGML_API const char * ggml_moe_direct_openarch_probe_json(void);

// OPEN_ARCH B axis - the inventory digest the last template seal recomputed independently.
// Empty string on the model-pin path and when not sealed. This is the observation window for the
// cross-language digest comparison (Gate C item 4).
// Returns the written string length on success, <0 on failure (buffer too small / bad argument).
// out is always NUL-terminated.
GGML_API int ggml_moe_direct_test_template_inventory_sha256(char * out, int32_t cap);

// ★§Z-⑥ 원자 A2 — 직전 mode=virtual seal 이 확보한 자원의 스냅샷(관측 전용). 자원 확보
//   직후에 찍히므로 seal 이 성공하든(A3) 그 뒤에서 실패하든 같은 값이 남는다.
//   스냅샷 부재(=자원 확보 미도달) = 반환 <0.
GGML_API int ggml_moe_direct_test_v3_probe(int32_t * n_handles, int32_t * n_slots, int32_t * n_parts,
                                           int32_t * att_managed, int32_t * att_host,
                                           int32_t * n_prot_intervals,
                                           uint64_t * pool_bytes, uint64_t * slot_stride_max,
                                           uint64_t * handle_bytes_total);

// 위 스냅샷의 주소 계약 1건 — flat_index = (mlpos 오름차순, parts 배열순) flatten 순서.
// 반환 offset/bytes 는 `ggml_moe_direct_part_info()` 와 **같은 분기**를 통과한 값이다.
GGML_API int ggml_moe_direct_test_v3_probe_part(int32_t flat_index, uint64_t * offset, uint64_t * bytes);

// ★v0.3.1 이월 원자 ⓐ — 직전 seal 이 **남긴** 공용 자원 상태를 그 자리에서 읽는다(관측 전용).
//   위 `..._test_v3_probe` 와 성격이 다르다: 저건 확보 **직후**의 스냅샷이고 이건 **지금 살아
//   있는 상태**다. 실패한 seal 의 rollback 이 전건을 되돌렸는지는 후속 seal 이 전역을 재게시
//   하면(seal_common_locked_v2 의 레이아웃 확정 블록) 관측에서 사라지므로, **실패 직후**가
//   유일한 창이다. 전 out 파라미터 nullable · g_mod.mutex 하 필드 복사만(부작용·판정 0).
//   축 11 = bin/virtual 공용 9 + virtual 전용 2(source_files·veh_handle).
GGML_API void ggml_moe_direct_test_seal_residue(int32_t * out_data_file_valid,
                                                int32_t * out_pool_base_nonnull,
                                                uint64_t * out_pool_bytes,
                                                int32_t * out_n_slots,
                                                int32_t * out_n_slot_entries,
                                                int32_t * out_n_records,
                                                int32_t * out_n_registry,
                                                int32_t * out_n_prot_intervals,
                                                int32_t * out_n_reg_tensor_info,
                                                int32_t * out_n_source_files,
                                                int32_t * out_veh_handle_nonnull);

// ★§Z-⑥ 원자 A3 — mode=virtual child 팬아웃 관측(테스트 전용). 제출·회수 child 수와
//   op terminal·publish 수. "record 1건 = 물리 read N건"이 실제로 그렇게 돌았는지의 판정 근거다.
GGML_API void ggml_moe_direct_test_v3_child_stats(uint64_t * submitted, uint64_t * retired,
                                                  uint64_t * op_terminals, uint64_t * publishes);

// ★★R-1(FROZEN `SPEC_PB_VIRTUAL_RUNTIME_R1` §2-0 A0-2) — **wave 발행 수**.
//   child 총수만으로는 "한 번에 P 개"와 "W 개씩 여러 번"이 **구분되지 않는다**(둘 다 같다).
//   이 계수가 wave 순차 제출이 실제로 돌았다는 유일한 직접 증거다: 같은 형상에서
//   `io_part_parallelism` 만 낮추면 `waves = ceil(P/W) × records` 로 늘어야 한다.
//   `out_part_parallelism` = 해소된 `io_part_parallelism`(= wave 폭 상한)을 같이 낸다.
GGML_API void ggml_moe_direct_test_v3_wave_stats(uint64_t * out_waves, int32_t * out_part_parallelism);

// ★§Z-⑥ 원자 A3 — child 경로 one-shot 결함 주입(테스트 전용 · 프로덕션 0). 실 저장장치에서는
//   완료 순서·부분 read 를 재현할 수 없어서 §6-4 ⑩ 이 요구하는 경로를 이 창구로 발화시킨다.
//     reverse_children    1 = 다음 op 의 part→채널 배정을 역순으로(= 완료 순서 역전) · <0 = 무시
//     fail_child_part   >=0 = 다음 op 의 그 part child 를 실패로 강등 · -1 = 해제 · <-1 = 무시
//     fail_after_acquire  1 = 다음 mode=virtual seal 을 자원 확보 직후 실패시킨다 · <0 = 무시
GGML_API void ggml_moe_direct_test_v3_arm(int32_t reverse_children, int32_t fail_child_part,
                                          int32_t fail_after_acquire);

// ★재개봉 R3 — 다음 op 의 `part` child 만 **동기 완료 경로**로 통과시킨다(-1=해제).
//   이 저장 구성에서는 ReadFile 즉시 TRUE 가 자연 발생하지 않아 혼합 경로가 안 열린다.
GGML_API void ggml_moe_direct_test_v3_arm_sync_child(int32_t part);

// ★재개봉 R3 — **직전 op 의 child 회수 순서**(part 인덱스 열). 반환=실제 길이.
//   out-of-order 를 "일어났을 것"이 아니라 기록된 열로 단언하기 위한 창구다.
GGML_API int32_t ggml_moe_direct_test_v3_retire_order(int32_t * out_parts, int32_t max_n);

// ★재개봉 R3 — 동기 분기로 닫힌 child 수 / 이벤트 회수로 닫힌 child 수(혼합 실증).
GGML_API void ggml_moe_direct_test_v3_sync_stats(uint64_t * sync_children, uint64_t * async_children);

// ★§Z-⑥ 원자 A4 — 바운스(비4K) 경로 관측(테스트 전용): copy 완료·실패 수, staging 부족으로
//   제출을 미룬 횟수(backpressure), 현재 점유 중인 staging 칸 수와 전체 칸 수.
GGML_API void ggml_moe_direct_test_v3_bounce_stats(uint64_t * copies_done, uint64_t * copies_failed,
                                                   uint64_t * backpressure_stalls,
                                                   int32_t * staging_busy, int32_t * staging_count);

// ★§Z-⑥ R14 후속 — 직전 테스트 seal 시도의 **거부 사유 원문**(테스트 전용). 거부 진단문이
//   요구 하한과 처방을 함께 담는지를 selftest 가 직접 잠그기 위한 창구다.
//   ★R18 — **호출자 버퍼 복사형**(구 판은 내부 문자열의 포인터를 반환해 다음 seal 이 다른
//   스레드에서 문자열을 바꾸면 수명이 보호되지 않았다). 잠금 안에서 복사·NUL 종단까지 끝낸다.
GGML_API void ggml_moe_direct_test_last_seal_error(char * buf, int32_t cap);

// ★§Z-⑥ R18 — dispatcher 기동 실패 주입(테스트 전용 one-shot · 프로덕션 0). 실환경에서는
//   CreateEvent/스레드 생성 실패로만 도달하는 경로의 **되돌림 총체성**(특히 이미 열린 metrics
//   handle·header 파일 정리)을 결정론적으로 행사한다.
GGML_API void ggml_moe_direct_test_v3_arm_dispatcher_fail(int32_t on);

// ★§Z-⑥ R12 — mode=virtual IO 계약 관측(테스트 전용). 셋 다 **IO3 on/off 와 무관한 엔진 축**이다.
//     io_param_val/src : SPEC_IO_METRICS_V3 §3-1 해소값 6종과 source enum(0=default 1=manifest
//                        2=dedicated_env 3=legacy_alias) — IO3 헤더 off 104..115 와 같은 출처.
//     child_pending/immediate_demand : §2 재결속 — demand-origin 물리 child 의 pending/immediate 계수.
//     readfile_ops/digest            : §4 mode-local(virtual) 18B/child 축.
//     io3_rows_alloc                 : IO3 병렬표(op_row+chan_row) 원소 수 — "기본 off 면 0" 실증용.
GGML_API void ggml_moe_direct_test_v3_io_stats(int32_t * out_io_param_val, uint8_t * out_io_param_src,
                                               uint64_t * out_child_pending_demand,
                                               uint64_t * out_child_immediate_demand,
                                               uint64_t * out_readfile_ops,
                                               uint64_t * out_readfile_digest,
                                               int32_t * out_io3_rows_alloc);

// ★★PB virtual 배선 관측(테스트 전용 · 권위 = `bench/techdev/repack_space/
//   SPEC_PB_VIRTUAL_RUNTIME_R1.md` §2-0). **IO3·PB 사이드카 on/off 와 무관한 엔진 축**이며,
//   그것이 이 창구의 존재 이유다(A0-1 "①②는 관측 기능과 무관하게 증감"의 실증점).
//     rec_active     : A0-5 전역 gauge `rec_admitted_active_global`(reserved+inflight record 수)
//     rec_inflight   : 첫 child submit 이후 ~ terminal 전 prefetch record 수
//     child_reserved : A0-4 credit 식의 `child_reserved_prefetch`(미제출 admitted 선점분)
//     child_inflight_demand/prefetch : ②축(= SPEC_IO_METRICS_V3 §3-5 `io_child_inflight_*`)
//     deferred_credit_attempts : C3 **밖** 진단(A0-4 · retry-attempt · 중복 계수 허용)
//     skipped_drain_turns      : C3 **밖** 진단(A0-4 ④ · 실제 drain turn 당 1회)
//     admit_parked   : drain-defer park 래치(A0-4 ② wake 마스킹 중인가)
GGML_API void ggml_moe_direct_test_pbv_stats(int32_t * out_rec_active, int32_t * out_rec_inflight,
                                             int32_t * out_child_reserved,
                                             int32_t * out_child_inflight_demand,
                                             int32_t * out_child_inflight_prefetch,
                                             uint64_t * out_deferred_credit_attempts,
                                             uint64_t * out_skipped_drain_turns,
                                             int32_t * out_admit_parked);

// ★★PB virtual 배선 — 층별 정적 상한(A0-5 등식)과 wave 폭(A0-2)의 직독 창구. 게이트가
//   **엔진이 실제로 쓰는 값**을 대조하기 위해 존재한다(테스트가 등식을 재구현하면 그 사본이
//   곧 드리프트 원이다 — 값 출처는 하나다). 반환 = `N_cap(mlpos)` · `out_wave` = `W_l`.
GGML_API int32_t ggml_moe_direct_test_pbv_ncap(int32_t mlpos, int32_t n_stamp, int32_t * out_wave);

// ★★PB virtual 배선 — A0-4 wake/turn 결과 계약 ②의 **직접** 관측 창구.
//   "변화 없는 queued prefetch intent 만으로는 즉시 다음 turn 을 만들지 않는다" 는
//   **`intent_depth > 0` 인데도 `command_event` 가 non-signaled** 라는 상태로 관측된다.
//   ★이 창구가 필요한 이유: 대기 복귀 횟수(`test_channel_event_state` 의 `wait_wakeups`)로는
//   판정할 수 없다 — `test_hold_retire` 가 armed 면 회수되지 않은 채널 이벤트가 계속 signaled
//   이라서 park 여부와 무관하게 dispatcher 가 회전한다(하네스 고유 성질 · 이 라운드 실측
//   Δ=165,233/150ms). 그래서 시간 임계가 아니라 **계약 그 자체**(마스킹된 event 상태)를 본다.
//   ★`command_event` 는 manual-reset 이라 조회(`WaitForSingleObject(...,0)`)가 상태를 바꾸지
//   않고, 조회 자체는 command 를 발행하지 않으므로 관측이 event 를 세우지도 않는다.
GGML_API void ggml_moe_direct_test_pbv_wake_state(int32_t * out_command_event_signaled,
                                                  int32_t * out_intent_depth,
                                                  int32_t * out_admit_parked);

// ★§Z-⑥ R14 — IO3 children 행의 **in-memory** 사건 열(테스트 전용). 파일 flush 를 기다리지
//   못하는 경로(동기 실패 → §3-4ⓐ fatal 로 프로세스가 죽는다)에서 §5-6 dense 폐합을 판정하는
//   유일한 창구다. 반환=현재 행 수(최대 max 개를 out 배열에 복사 · terminal 미발급=전비트 1).
GGML_API int32_t ggml_moe_direct_test_io3_children(int32_t max, uint64_t * out_submit_seq,
                                                   uint64_t * out_terminal_seq, uint8_t * out_status);

// ★F1 — IO3 copies 행의 **in-memory** 내용(테스트 전용 · 위 children 창구의 짝). 행이 존재
//   하는지가 아니라 **채워졌는지**를 본다: 실행된 copy 는 `worker`·`start_qpc`·`done_qpc` 가
//   서고 `status` 가 확정되며, 미실행 의무 행은 worker=0xFF·시각 0·CANCELLED 로 남는다
//   (§5-5 상태·시각 규약). 반환=현재 행 수(최대 max 개를 out 배열에 복사).
GGML_API int32_t ggml_moe_direct_test_io3_copies(int32_t max, uint64_t * out_enqueue_qpc,
                                                 uint64_t * out_start_qpc, uint64_t * out_done_qpc,
                                                 uint8_t * out_worker, uint8_t * out_status);

// ★§Z-⑥ 원자 A6 — 프리필 coalescing 관측(테스트 전용 · SPEC_REPACK_V3 §3-6 · §5).
//     phase        : 0=off 1=prefill 2=all(해소값)
//     min/max_bytes: §5 io_coalesce_min/max 해소값
//     buffers/slots: io_coalesce_buffers · 실제 확보한 버퍼 칸 수(= buffers × max_parts)
//     groups       : 성립한 대구간 수(= coalesced child 수) · records: 참가 record 누계
//     scatters     : 완료된 scatter copy 수
//     reads_retired: **회수 완료된** 대구간 read 수(⑦ 관문은 제출이 아니라 회수 시점을 본다)
//     copy_outstanding / copy_credit_stalls : 공용 copy 크레딧 예약량·부족 횟수(S1)
//     last_reject  : 직전 계획 판정 — 0=합침 1=off 2=phase 3=동질성(§5-4 8번째 MUST)
//                    4=연속 run 5=source 6=경계 7=정렬/용량 8=claim·pin 9=min 10=자원
//     reject_mask  : 본 사유 비트 누적(`1u << 사유`) — 한 acquire 는 pump 를 여러 번 돌리므로
//                    "직전 값"은 마지막 패스(후보 1건 = 연속 run 미달)를 가리킨다. 네거티브
//                    관문의 지목은 이 마스크로 한다.
//   ★네거티브 관문은 "안 합쳐졌다"가 아니라 **어느 조건이 막았는지**를 이 값으로 주장한다.
GGML_API void ggml_moe_direct_test_v3_coalesce_stats(int32_t * out_phase, uint64_t * out_min_bytes,
                                                     uint64_t * out_max_bytes, int32_t * out_buffers,
                                                     int32_t * out_slots, uint64_t * out_groups,
                                                     uint64_t * out_records, uint64_t * out_scatters,
                                                     int32_t * out_last_reject, uint32_t * out_reject_mask,
                                                     uint64_t * out_reads_retired,
                                                     int32_t * out_copy_outstanding,
                                                     uint64_t * out_copy_credit_stalls);

// ★§Z-⑥ 원자 A6 — coalescing 가드 주입(테스트 전용 · 프로덕션 0).
//   ★**sticky** 다 — **1회성 소모형**으로 두면 한 acquire 가 pump 를 여러 번 돌리므로
//   첫 계획에서 소진되고, 두 번째 계획이 합쳐버려 "이 가드가 막았다"는 판정이
//   성립하지 않는다.
//   해제는 `arm_coalesce(0, 0)` 또는 `reset_for_test()` 가 한다.
//   실 런타임에서 도달 불가한 두 가드를 결정론적으로 행사하는 유일한 창구다:
//     overrun=1 : run 을 텐서 끝 너머로 1 expert 부풀린다(§3-6 ③ 경계 가드)
//     unpin=1   : 계획의 **head** 를 미pin 으로 취급(§3-6 ⑤ head 축)
//     unpin=2   : 계획의 **비대표 participant 전부**를 미pin 으로 취급(⑤ participant 축 —
//                 head 축만 행사하면 후보 필터를 지우는 mutant 가 살아남는다)
GGML_API void ggml_moe_direct_test_v3_arm_coalesce(int32_t overrun, int32_t unpin);

// ★§Z-⑥ 원자 A4 — 바운스 copy 결함 주입(테스트 전용 · 프로덕션 0).
//     pause           1 = copy worker 정지(staging 점유를 붙들어 backpressure 발화) · 0 = 재개 · <0 = 무시
//     fail_copy_part >=0 = 다음 그 part 의 copy 1건을 실패로 · -1 = 해제 · <-1 = 무시
GGML_API void ggml_moe_direct_test_v3_arm_copy(int32_t pause, int32_t fail_copy_part);

// ★§Z-⑥ S2 — IO3 버퍼 상한을 **바이트 단위**로 재설정(테스트 전용 · 프로덕션 0).
//   운영 키 `MOE_DIRECT_IO3_BUF_MB` 는 MB 단위라 `used < cap < used+need` 경계를 관문에서
//   밟을 수 없다 — 그 경계를 결정론적으로 행사하는 창구다(§5-1 env 표면은 무접촉).
GGML_API void ggml_moe_direct_test_v3_set_io3_cap(uint64_t bytes);

// OVERLAPPED 읽기 완료 분기 관측 — 즉시성공 완료 수 / ERROR_IO_PENDING 후 대기 완료 수.
GGML_API void ggml_moe_direct_test_read_stats(uint64_t * immediate_count, uint64_t * pending_count);

// 테스트 전용: 리스 통계(leased 슬롯 수·총 lease 카운트) — release 복귀·lease pin 검증용.
GGML_API void ggml_moe_direct_test_lease_stats(int32_t * out_leased_slots, int64_t * out_total_leases);

// ★QD 레버 검증용 테스트 전용 내부 상태 조회(reviews/codex_qd_u6_design_xcheck.md 심의 3).
// ★프로덕션 경로 무접촉 — acquire/release/ptr/seal 어디서도 호출하지 않으며, 읽기만 하고
//   모듈 상태를 바꾸지 않는다. selftest가 record→slot 매핑·LRU(last_used) 순서·slot generation
//   불변을 직접 관측하기 위한 창구다(ptr!=null 같은 간접 관측으로는 tick 순서 결함이 통과한다).
// 범위 밖/미seal이면 out에 센티널(slot=-1·owner=-1·나머지 0)을 채운다.
GGML_API void ggml_moe_direct_test_record_slot(int64_t record_idx, int32_t * out_slot,
                                               uint64_t * out_slot_generation);
GGML_API void ggml_moe_direct_test_slot_state(int32_t slot_index, int64_t * out_owner_record,
                                              uint64_t * out_last_used, uint64_t * out_generation,
                                              int32_t * out_lease);

// ---------------------------------------------------------------------------
// ★B1 영속 I/O dispatcher (PHASEB_DESIGN §2 — FROZEN) 테스트/조회 창구.
// 프로덕션 경로 무접촉(읽기 전용 조회 + 테스트 전용 제어). §7 착지 관문의 관측 창구다.
// ---------------------------------------------------------------------------

// dispatcher 활동/무손실 지표. dispatcher_demand_cmd>0 = 영속 경로가 실제로 demand를 처리했다는 증명.
GGML_API void ggml_moe_direct_test_dispatcher_stats(
    uint64_t * out_demand_cmd, uint64_t * out_release_cmd,
    uint64_t * out_query_cmd,  uint64_t * out_join_count,
    uint64_t * out_cancel_count, uint64_t * out_abort_terminal,
    uint64_t * out_q_full_waits, uint64_t * out_q_max_depth,
    int32_t  * out_io_path_persistent);

// 항목별 mem echo(§6.3) — capture/intent/predictor/snapshot은 B1에서 코드 자체가 없어 항상 0.
GGML_API void ggml_moe_direct_test_mem_echo(
    uint64_t * out_control_q_bytes, uint64_t * out_pending_bytes,
    uint64_t * out_channel_bytes,   uint64_t * out_dispatcher_stack_bytes,
    uint64_t * out_capture_bytes,   uint64_t * out_intent_bytes,
    uint64_t * out_predictor_bytes, uint64_t * out_snapshot_bytes);

// ordered ReadFile digest / record→slot·LRU digest 직접 조회(selector ABBA 완전일치 판정용).
// ★readfile digest는 metrics 활성 시에만 갱신된다(기존 게이트 불변) — 미활성이면 초기값.
GGML_API void ggml_moe_direct_test_digests(
    uint64_t * out_readfile_digest, uint64_t * out_readfile_ops,
    uint64_t * out_lru_tick_digest, uint64_t * out_tick_final);

// 채널 이벤트 상태 스냅샷 + dispatcher 대기 복귀 횟수(hot-loop/공회전 검출).
// ★불변식: terminal 처리 후 non-busy 채널의 manual-reset event는 전부 non-signaled.
GGML_API void ggml_moe_direct_test_channel_event_state(
    int32_t * out_n_channels, int32_t * out_busy,
    int32_t * out_idle_signaled, uint64_t * out_wait_wakeups,
    uint64_t * out_sync_terminal);

// 테스트 전용 — 다음 pending 제출 1건을 sync_done(즉시 완료) 분기로 강제한다(one-shot·행사 후
// 자동 disarm·0=disarm). 즉시 완료가 저장 구성상 재현되지 않는 기기에서도 그 분기의 이벤트
// 규율(ResetEvent)을 회귀로 잠그기 위한 창구. 프로덕션 경로는 항상 OFF.
GGML_API void ggml_moe_direct_test_force_sync_next(int32_t on);

// 테스트 전용 — dispatcher pass 경계 정지/재개(control_q 만석 대기 픽스처). on!=0은 정지 확인까지 블록.
GGML_API void ggml_moe_direct_test_dispatcher_pause(int32_t on);

// 테스트 전용 — 제출 op가 n_ops건 이상이 되는 순간 취소 drain(§2 abort 경로)을 행사. 0=off.
GGML_API void ggml_moe_direct_test_abort_after_submit(int32_t n_ops);

// 테스트 전용 — 종료 시퀀스 직접 행사(0=graceful drain, 1=abort). 멱등.
GGML_API void ggml_moe_direct_test_shutdown(int32_t abort_mode);
// ★★S-2(교차 r2 F-2) — **유계 종료**(테스트 전용). 반환 **1=정상 종료·정리 완료 /
//   0=timeout**(스레드 미회수 · 호출자는 반드시 **관문 실패**로 판정할 것).
//   ★존재 이유: 구 `ggml_moe_direct_test_shutdown` 은 내부가 `WaitForSingleObject(..., INFINITE)`
//   라, graceful drain 이 회귀로 안 끝나면 **관문이 FAIL 하는 게 아니라 selftest 가 교착**한다.
//   상시 도는 릴리스 관문에 교착형 게이트를 두면 순손실이므로, 종료를 기다리는 관문은
//   **반드시 이 창구를 쓴다**. `timeout_ms < 0` = INFINITE(구 거동).
//   ★timeout 시 cleanup 을 생략한다(dispatcher 가 아직 상태를 쓸 수 있음) — 그 런은 이미
//   붉은 상태라 프로세스 종료가 정상 처분이다.
GGML_API int  ggml_moe_direct_test_shutdown_bounded(int32_t abort_mode, int32_t timeout_ms);

// ---------------------------------------------------------------------------
// ★B2 프리페치(PHASEB_DESIGN §4 · SPEC_PHASEB_WIRE §4~§6) 테스트/조회 창구.
// 프로덕션 경로 무접촉(읽기 전용). 판독기 게이트를 selftest가 코드로 재계산·대조한다.
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_direct_test_pb_counts(
    uint64_t * out_n_samples, uint64_t * out_n_ios,
    uint64_t * out_publish_seq, uint64_t * out_event_seq,
    int32_t * out_prefetch_on, int32_t * out_k, int32_t * out_n);

// pb_sample / pb_io 레코드 1건 복사(bytes = 각 레코드 sizeof). 성공 0 / 범위 밖·크기 불일치 <0.
GGML_API int  ggml_moe_direct_test_pb_sample(uint64_t idx, void * out_record, uint64_t bytes);
GGML_API int  ggml_moe_direct_test_pb_io(uint64_t idx, void * out_record, uint64_t bytes);

// 슬롯 cache class(0=비상주/기타 1=READY_DEMAND 2=READY_PROBATION) + probation intent ordinal
// + ★probation 비교 키의 rank 성분(R1 #7 · 비probation = 0xFFFF).
GGML_API void ggml_moe_direct_test_slot_class(int32_t slot_index, int32_t * out_class,
                                              uint64_t * out_intent_ordinal, int32_t * out_rank);

// 테스트 전용 one-shot — 다음 prefetch 성공 terminal 1건을 부분 read/오류 terminal(§5)로 강등한다.
// 실 저장장치에서 부분 read를 재현할 수 없으므로 fail-soft unpublish·joined retry 경로의 회귀 창구.
GGML_API void ggml_moe_direct_test_pb_fault_next(int32_t on);

// prefetch in-flight 0 + intent_q 소진까지 대기(결정론 확보). 1=quiesced / 0=timeout.
GGML_API int  ggml_moe_direct_test_pb_quiesce(int32_t timeout_ms);

// 테스트 전용 — signaled 채널 회수를 보류/재개(op을 in-flight로 고정). join·pending·취소
// 픽스처가 "완료 전 창"을 결정론적으로 잡기 위한 창구. 프로덕션 경로는 항상 0.
GGML_API void ggml_moe_direct_test_hold_retire(int32_t on);
// ★★R-5(교차 r1 F3) — 테스트 전용 **제출 보류**(pump 를 통째로 멈춘다). `hold_retire` 와 다른
//   창이다: retire 보류는 이미 제출된 child 를 붙잡을 뿐이라 record 는 `submitted=1 ·
//   reserved=0` 이 된 뒤다. A0-3 이 말하는 "**미개시 예약분**"은 record 가 admission 은 했는데
//   첫 submit 을 못 한 구간에만 존재하고, 그 구간이 없으면 `pbv_rec_release_locked` 의 예약분
//   반환 갈래가 영영 미행사로 남는다 — 역변이로 실증됐다(그 반환을 지워도 V3P 32/32 통과).
//   arm 후 `test_shutdown(abort_mode=1)` 을 태우면 never-issued terminal 이 서고, 그때 반환이
//   실제로 도는지 잴 수 있다. 프로덕션 0.
GGML_API void ggml_moe_direct_test_hold_submit(int32_t on);
// ★★R-5 V3P17 — **취소 선별 관측 창구**(§3 ⑮ · A0-4 joined 보호). graceful drain 의
//   `disp_cancel_prefetch_v3_locked` 는 "waiterless prefetch 만 취소"라는 선별을 하는데, 구
//   트리에는 그 **결과를 보는 창이 없었다** — joined 보호가 실제로 발화했는지는 "demand 가
//   성공했으니 아마 취소되지 않았을 것"이라는 **간접 추론**으로만 말할 수 있었다. 두 결과를
//   직접 센다: `selected`=취소 대상으로 뽑힌 채널 · `excluded_joined`=waiter 때문에 제외된 채널.
//   프로덕션 비용 = 선별 루프 안 정수 증가 2회.
GGML_API void ggml_moe_direct_test_v3_cancel_stats(uint64_t * out_selected,
                                                   uint64_t * out_excluded_joined);
// ★★S-1(교차 r2 F-1) — credit **예약/반납 누적** 직독(테스트 전용).
//   ★왜 gauge 로는 안 되는가: `pbv_child_reserved` 는 dispatcher 정지 경로가 **무조건 0
//   으로 밀어**(`dispatcher_cleanup_locked`), "반환이 실제로 돌았는가"와 "그냥 지워졌는가"를
//   구별하지 못한다 — 반환을 통째로 지운 역변이(M-H)가 **전 관문을 통과**한 실측이 그
//   증거다. 그래서 **사건 자체를 누적**하고 teardown 은 이 두 값을 건드리지 않는다
//   (`reset_for_test` 만 지운다).
//   ★계약: 완전 drain 후 **`reserved_total == unreserved_total`** — A0-3 "현재 실제 보유한
//   reserved 만 반환 · 과반환 금지"의 관측형이다(과소=누수 · 과다=과반환).
GGML_API void ggml_moe_direct_test_pbv_credit_totals(uint64_t * out_reserved,
                                                     uint64_t * out_unreserved);
// ★★S-4(교차 r2 F-4) — **continuation credit 재검사 실패 사건** 누적(테스트 전용).
//   A0-4 **drain-before-refill** 은 "그 turn 에 continuation 재검사가 1건이라도 실패하면
//   신규 admission 을 생략"이다. 구 트리엔 그 사건을 볼 창구가 bool 하나뿐이어,
//   성공 경로만 만든 arm(V3P14b · 실측 `defer=0`)이도 녹색이었다 — 즉 **실발동 증거가
//   없었다**. 이 계수가 그 빈자리를 메운다.
GGML_API void ggml_moe_direct_test_pbv_cont_defer_events(uint64_t * out_events);
// ★★T-3(교차 r3 S-4) — **결과 계약** 축: continuation 재검사 실패 때문에 그 turn 의 **신규
//   prefetch admission 을 생략**한 횟수. 위 `cont_defer_events`(=실패 **사건**, pump 가 셈)와
//   반드시 구분할 것 — `pb_admit_locked` 가 `cont` 를 무시하도록 되돌려도 실패 사건은 그대로
//   오르고 record 도 완주하므로, 이 축이 없으면 **drain-before-refill 역변이 감도가 0** 이다.
GGML_API void ggml_moe_direct_test_pbv_admit_skipped_cont(uint64_t * out_events);
// ★★U-3(교차 r4 · **대기 고정**) — 위 계수가 "생략했다"까지라면, 이 창구는 "**막을 대상이 그
//   순간 실제로 대기 중이었다**"를 고정한다. 최초 생략 시점의 ①살아 있는 record 수
//   ②큐에 남은 intent 수를 엔진이 그 자리에서 박아 둔 값이다.
//   ★왜 엔진이 재는가: 하네스의 turn 유발 창구는 dispatcher 를 **깨울 뿐**이라, 테스트 스레드가
//     사건 계수를 읽는 시점엔 이미 record 완주·대기 intent 소비가 끝난 뒤다(실측으로 확인 —
//     사건 시점 스냅샷조차 `rec_active=0 intent_depth=0`). 관측자를 생략 지점 안에 두어야
//     경합 없이 결정론이 된다.
//   ★"막을 대상"은 **두 자리**에 설 수 있다 — 엔진은 intent 를 pop 한 뒤 rank 를 **turn 당 1건**
//     씩 소비하므로, 생략된 신규 admission 은 ①진행 중 intent 의 남은 rank 이거나 ②큐에서
//     대기 중인 다음 intent(`out_intent_depth`)다.
//   ★★V(교차 r5) — 다만 "대상이 있다"만으론 **완화**다. r5 가 실측으로 적발한 2건:
//     ①raw 잔여는 `it.ids[r] == NA` 칸까지 센다(`test_push_intent` 가 `k = pf_k` 로 두고 남는
//      칸을 NA 로 채운다) — NA rank 는 차단을 없애도 `PRE_STALE` **처분**이지 admission 이 아니다.
//     ②`N_cap` 포화면 유효 ID 라도 **cap-drop** 이지 admit 이 아니다.
//     ⇒ ⓐ`out_valid_left`(남은 rank 중 **유효 ID** 개수 · NA/범위초과 제외) ⓑ`out_ncap_room`
//       (`ncap - rec_active` · ≥1 이어야 admit 여지) ⓒ`out_credit_ok`(그 시점 credit 술어)
//       **셋이 동시에** 서야 "차단이 없었다면 그 자리에서 실제로 admit 됐다"가 성립한다.
//   ★`-1` = 생략 지점에 **한 번도 도달하지 않았다**(= `if(blocks||cont)→if(blocks)` 역변이 상태).
GGML_API void ggml_moe_direct_test_pbv_admit_skip_witness(int32_t * out_rec_active,
                                                          int32_t * out_intent_depth,
                                                          int32_t * out_valid_left,
                                                          int32_t * out_ncap_room,
                                                          int32_t * out_credit_ok);
// ★★V(교차 r5) — 생략 **그 순간**의 C3 `admitted` 기준선. 주입은 1회성이라 다음 turn 부터
//   `cont` 가 내려가므로, 막힌 것이 실 후보였다면 이 값보다 **뒤에 실제 admission 이 일어난다**.
//   ★기준선을 테스트가 나중에 읽으면 이미 재개가 끝난 뒤라 늦는다(실측 `resumed=0`) — 그래서
//     엔진이 생략 지점에서 박아 둔다. 증인 3축이 "가능했다"면 이 축은 "실제로 일어났다"이다.
GGML_API void ggml_moe_direct_test_pbv_admit_skip_admitted(uint64_t * out_admitted);
// ★★S-4 — continuation credit 재검사 **1회성 실패 주입**(테스트 전용 · 프로덕션 0).
//   ★노브만으로는 이 분기가 **도달 불가**하다(실측): part cap(`remaining+n_need>iop`)이 credit
//   재검사보다 앞서고 `W=min(P,iop)` 라 앞 wave 가 in-flight 인 동안엔 part cap 이 항상 먼저
//   막는다("wave 순차 진행"의 귀결). 앞 wave 회수 뒤라면 조건 ②는 N_cap 에 함의되어 안 깨지고
//   (§G-3①), 남는 조건 ①(demand 가 free_total 잠식)은 "회수 직후~다음 wave 제출 직전"의 한 턴
//   창을 잡아야 해 하네스로 결정론 재현이 안 된다. ⇒ 주입으로 **결과 계약**(drain-before-refill
//   이 그 turn 신규 admission 을 생략하는가)을 재는 쪽을 택한다.
GGML_API void ggml_moe_direct_test_cont_credit_fail_next(int32_t on);
// ★★T-4(교차 r3 S-5 ②) — 엔진이 **`q_stopping=true` 를 게시했는가**(테스트 전용).
//   shutdown 함수 호출 직전에 테스트가 세우는 플래그는 barrier 가 아니다 — 함수 안에서
//   predictor drain 이 먼저 돌고 그 사이에 demand 가 선완주할 수 있다. 이 창구를 대기
//   조건으로 써야 "정지 국면에 **실제로 들어간 뒤**" 를 보장한다.
GGML_API int  ggml_moe_direct_test_q_stopping(void);
// ★★T-1 — 유계 종료가 **timeout 으로 끝난** 사실. 이 래치가 서면 이후 프로덕션
//   정지 래퍼도 재진입하지 않는다(= "FAIL 기록 후 교착" 차단).
//   E-2b-0 (durable (g), carried from E-2a review r3): this used to read "the production
//   INFINITE wrapper". That stopped being true when E-2a replaced the production stop policy
//   with the finite MOE_DRAIN_DEADLINE_MS; the latch still blocks re-entry, but what it
//   blocks re-entry INTO is a bounded stop, not an unbounded one.
GGML_API int  ggml_moe_direct_test_stop_timed_out(void);

// 테스트 전용 — dispatcher 를 "stopping 스냅샷 직후" 창에 1회 정지(one-shot barrier)시켜, 그 창
// 안으로 stop 을 삽입하는 **결정적** 교착 재현을 가능하게 한다. arm: 0=창 도달/<0=실패,
// release 를 부르면 재개된다(대기는 dispatcher 쪽에서도 5s 유한 — 실패해도 영구 정지 없음).
GGML_API int  ggml_moe_direct_test_stop_race_arm(int32_t timeout_ms);
GGML_API void ggml_moe_direct_test_stop_race_release(void);

// 테스트 전용 — pending 풀 인덱스를 지정 개수만큼 소모/반납해 **재사용 경로**를 강제한다
// (R1 #1 회귀: 같은 index를 joined prefetch→일반 prefetch→retry→demand 순으로 돌린다).
// 반환 = 마지막으로 반납된 pending index(-1=불가).
GGML_API int32_t ggml_moe_direct_test_pending_reuse_probe(void);

// 테스트 전용 — CUT 잔여 검사 결과(0=잔여 없음). finalize 이후 유효.
GGML_API int32_t ggml_moe_direct_test_cut_residual(void);

// 테스트 전용 — CUT 잔여 세부 비트(0x01 intent_depth ~ 0x20 inflight) 주입. finalize 전에 설정.
// 자연 발생 잔여는 결정론적 재현이 불가하므로 echo·마스크 결속 등식은 주입으로 행사한다.
GGML_API void    ggml_moe_direct_test_force_cut_residual(int32_t bits);

// 테스트 전용 — 상태 큐 -> 보류함 드레인 1회. 반환 = 드레인 후 보류함 크기.
GGML_API int32_t ggml_moe_direct_test_drain_status(void);

// Test-only window onto one lease token. Never fatal, on purpose: a gate has to be able to
// ask about a handle it believes to be bad without dying. 0 and the outs filled for a live
// handle, <0 otherwise. out_lease_units = the token's captured sum of multiplicity.
GGML_API int ggml_moe_direct_test_xstore_token(uint64_t token, int32_t * out_mlpos,
                                               int32_t * out_n_records,
                                               int32_t * out_lease_units,
                                               uint64_t * out_epoch);

// E-2b-0 test-only - the batch part the token CAPTURED at acquire. Separate from the window
// above on purpose: that one is sealed into E-2a's checks and widening it would edit them.
// 0 and *out_part filled for a live handle, <0 otherwise. Never fatal.
GGML_API int ggml_moe_direct_test_xstore_token_part(uint64_t token, int32_t * out_part);

// E-2b-0 test-only - the handle table itself. `addr` is the storage address as an integer,
// which is what a gate needs: the claim is not "the capacity equals some constant" (that
// would only echo the constant back) but "this storage never moved", and an address is the
// only way to say that from outside. Returns 0 with the outs filled once the table exists,
// <0 before the first epoch is published. Any out pointer may be null.
GGML_API int ggml_moe_direct_test_xstore_table(uint64_t * out_addr, int32_t * out_capacity,
                                               int32_t * out_live, int32_t * out_free);

// E-2b-0 test-only - hold the module mutex from the CALLING thread so a gate can show that
// the view read path does not want it. arm() takes the lock and stays until release() is
// requested or safety_ms elapses, so a fixture defect cannot wedge a 1400-check run; it
// blocks, so the caller runs it on its own thread. wait() blocks until the hold is actually
// established (1) or the timeout expires (0). safety_fired() reports that the hold ended on
// the cap rather than on request - a window that ended that way is not the window the gate
// meant to measure. Production never calls any of these.
GGML_API void ggml_moe_direct_test_mutex_hold_arm(int32_t safety_ms);
GGML_API int32_t ggml_moe_direct_test_mutex_hold_wait(int32_t timeout_ms);
GGML_API void ggml_moe_direct_test_mutex_hold_release(void);
GGML_API int32_t ggml_moe_direct_test_mutex_hold_safety_fired(void);

// Test-only - milliseconds left of the drain's ONE deadline for a caller-supplied
// deadline_ms, computed by the same helper drain_finish uses, so a gate built on this cannot
// drift from the production path. The budget is measured from the drain_begin instant, so a
// gate can prove the anchor by burning time between begin and this call.
//   >= 0  milliseconds left (0 = exhausted, or deadline_ms <= 0, which carries no budget)
//   <  0  no drain in progress
GGML_API int32_t ggml_moe_direct_test_xstore_drain_remaining(int32_t deadline_ms);

// Test-only - the drain's zero-confirm, and the two readings it combines. Returns 1 when the
// drain would confirm zero right now. `admissions` counts acquires that have passed the
// RUNNING gate and not yet finished; those hold no lease unit yet, which is why the
// lease-unit ledger alone cannot answer this. Either out pointer may be null.
GGML_API int32_t ggml_moe_direct_test_xstore_drain_zero_ready(int64_t * out_admissions,
                                                              int64_t * out_lease_units);

// Test-only - arm the generation counter to the LAST generation the handle encoding can
// carry, so a fixture can reach the exhaustion boundary without performing 2^48 acquires.
// Takes no value: the one value it writes is the boundary itself, which is what keeps this
// seam from being a general counter setter. The table must exist and no handle may be live -
// both are fatal if violated - and the module lock is taken by this call, not by the caller.
GGML_API void ggml_moe_direct_test_xstore_arm_last_generation(void);

// Test-only (E-2a, Codex r2 blocker 2) - the LAST stage of the stop chain: the deadline the
// dispatcher join measured against, and the milliseconds it was handed at its own entry.
// Scoped to the LAST stop, which clears this on entry: returns 1 when that stop reached the
// join stage and 0 when it did not, which is what separates "the stop drained fast" from "the
// stop never got there". A process-wide tally would answer nothing - a suite run performs
// hundreds of stops through reset_for_test before any gate reads this. Pair it with
// ggml_moe_phaseb_test_drain_stages, which reports the two stages inside the predictor drain:
// three stages that all name ONE deadline is the single-budget claim, measured rather than
// assumed. Either out pointer may be null.
GGML_API int32_t ggml_moe_direct_test_stop_join_stage(int64_t * out_deadline_abs_ms,
                                                      int32_t * out_left_ms);
// Test-only - std::chrono::steady_clock in milliseconds, the epoch every deadline above uses.
// A gate that reads it before drain_begin derives `anchor + budget` independently and compares
// that with what the stages report, rather than echoing an engine-computed number back.
GGML_API int64_t ggml_moe_direct_test_steady_ms(void);

// 실 FILE_MAP_READ 매핑 뷰에 대해 VirtualProtect(PAGE_NOACCESS)→의도 터치 1회→VEH 카운터
// 확인→PAGE_READONLY 복원→재접근 성공 왕복을 모듈 내부 VEH 경로로 실행. 성공=0, 실패<0.
// 먼저 ggml_moe_direct_seal_for_test()로 VEH가 등록돼 있어야 한다.
GGML_API int ggml_moe_direct_test_mapped_view_roundtrip(void * base, size_t nbytes);

#ifdef __cplusplus
}
#endif

// R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members; lead decision D, ledger :943): the thirteen
// core definitions the test entry points in ggml-moe-direct-test-entry.cpp call.  They stay in
// ggml-moe-direct.cpp and left its anonymous namespace for ggml_moe_direct_impl; each line below
// is that definition's own declarator, declared under its ORIGINAL symbol - no wrapper, no
// facade.  C++ and Windows only, like the definitions; outside the extern "C" block above so the
// names keep C++ linkage.  This header is test-only (r3-1), so production units never see them.
#if defined(__cplusplus) && defined(_WIN32)
#include <string>
namespace ggml_moe_direct_impl {
void dfac_coal_buffers_set_locked(int32_t v);
void dfac_coal_max_bytes_set_locked(uint64_t v);
void dfac_coal_min_bytes_set_locked(uint64_t v);
void dfac_coal_phase_set_locked(uint8_t v);
void dfac_coal_pool_release_locked();
void dfac_copy_outstanding_set_locked(int32_t v);
void dfac_io_param_src_set_locked(size_t i, uint8_t v);
void dfac_io_param_val_set_locked(size_t i, int32_t v);
void dfac_staging_pool_release_locked();
void dfac_test_fail_after_acquire_set_locked(int32_t v);
void restore_prot_intervals_locked();
int seal_common_locked_v2(const std::wstring & dir_w, const ggml_moe_direct_model & model, const ggml_moe_direct_source * srcs, int32_t n_srcs, bool allow_sector_fallback, uint64_t budget_bytes, uint32_t touch_abort, const std::wstring & expects_dir_w, const char * override_profile_id, const char * override_digest, const char * override_scope, std::string & err);
void xstore_reset_locked();
}  // namespace ggml_moe_direct_impl
#endif
