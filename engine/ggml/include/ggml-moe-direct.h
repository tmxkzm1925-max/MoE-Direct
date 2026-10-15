// MoE 직독(direct-read) 모듈 — v2(manifest-driven). 사양: REPACK_V2_DESIGN.md §4 + bench/moe-direct/SPEC.md.
//
// v1(GPT-OSS 상수 검증기)을 해체하고 manifest v2(schema_version "2.0")를 유일 권위로 소비한다:
// PROD_* 상수 전폐 · 층별 stride/record_base · 멀티-shard sources[] · uint16 expert(384E) 지원 ·
// 등록 3집합 분리(L_all/M_all/R_host) · 슬롯 용량 불변식 n_slots>=n_expert.
//
// 이 헤더는 ggml-base 타깃에 포함되어 exported C ABI로 노출된다. 시공 1단계 = 이 모듈 본체
// (등록·seal·acquire/release/ptr·슬롯 풀·VEH·계측). 침습 지점(ggml-cpu.c/ops.cpp/ggml-backend.cpp/
// llama-mmap.cpp/llama-model.cpp/server.cpp)은 2단계.
//
// Windows 전용: 비Windows 빌드에서는 ggml-moe-direct.cpp가 전부 no-op/실패 스텁으로 컴파일된다.
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// VEH kill 경로(§3-5 threshold 도달 분기)가 TerminateProcess()에 넘기는 고유 exit code.
#define MOE_DIRECT_VEH_KILL_EXIT_CODE 113

// ---------------------------------------------------------------------------
// 환경 변수 게이트 (값은 최초 호출 시 1회 캐시)
// ---------------------------------------------------------------------------

// env MOE_DIRECT=1 이면 true. off(A 세션)면 이 모듈의 코드 경로는 전부 비활성.
GGML_API bool ggml_moe_direct_enabled(void);

// env MOE_NO_PREFETCH=1 이면 true. MOE_DIRECT와 독립 변수(A/B 공통으로 켤 수 있음).
GGML_API bool ggml_moe_no_prefetch(void);

// ---------------------------------------------------------------------------
// v2 seal 인자 구조체 (live GGUF 헤더에서 재도출해 manifest와 전항 대조)
// ---------------------------------------------------------------------------

// 로드 GGUF shard 하나의 실물 identity(§4-1 sources[] 대조용 — v1 files[0] 단일에서 확장).
struct ggml_moe_direct_source {
    const char * path;          // shard 경로(utf8) — manifest.sources[i].path 와 basename 대조
    uint64_t     bytes;         // 파일 크기
    int64_t      mtime_epoch;   // floor(mtime 초)
    uint64_t     data_start;    // gguf 데이터 오프셋
    uint32_t     gguf_version;
    uint32_t     alignment;
};

// live GGUF 헤더에서 재도출한 모델 메타(§4-2④ manifest.model 전항 대조용).
struct ggml_moe_direct_model {
    const char *    arch;
    int32_t         n_layer;
    int32_t         n_expert;
    int32_t         n_expert_used;
    const int32_t * moe_layers;   // 오름차순 실제 layer id 배열
    int32_t         n_moe_layers;
    int32_t         n_layer_nextn; // ★부록A: NextN 예측층 수(GGUF nextn_predict_layers·기본 0).
                                   // execution scope에서 제외 경계 layer = n_layer - n_layer_nextn.
    int32_t         n_embd;        // ★B2(R1 #5 ②): 프리페치 스냅샷 형상을 seal 시점에 확정하기 위한
                                   // hidden size. 0이면 스냅샷 형상 미확정(프리페치 신호 비활성).
    float           f_norm_rms_eps;// 〃 RMSNorm epsilon(0이면 1e-6 기본).
    int32_t         n_layer_dense_lead; // OPEN_ARCH B axis (M3): GGUF `{arch}.leading_dense_block_count`
                                   // (hparams.n_layer_dense_lead). The engine needs this value to
                                   // generate the deepseek2-family template layer set independently
                                   // from the formula `leading_dense_block_count..block_count-1`.
                                   // Non-leading-dense templates and the model-pin path never read
                                   // this field (default 0). Appended last so no existing field
                                   // offset moves.
};

// ---------------------------------------------------------------------------
// 등록 (2단계 로더가 호출) — v2: 텐서명(source_tensor)으로 등록한다. part/layer 해소는
// seal 시 manifest가 권위. is_host=0이면 비host(GPU 상주 — L_all에는 포함되나 R_host 제외:
// direct 등록·페이지 보호 대상 아님). is_host!=0이면 R_host(direct 등록·보호 대상).
// ---------------------------------------------------------------------------

// 성공=0, 실패(<0): null / seal 이후 등록 시도 / 동일 텐서 재등록 불일치.
GGML_API int ggml_moe_direct_register(const struct ggml_tensor * t, const char * source_tensor_name, int32_t is_host);

// seal: manifest v2 로드 + §4-1 전항 검증(schema_version·sources 전 shard 실물·layers 산술
// 자기일관·A_runtime 재질의·reference_lock 필드 무결) + 등록 3집합 대조(L_all==M_all 양방향·
// R_host만 direct 등록/보호) + live model 전항 대조 + 슬롯 풀 할당(불변식 n_slots>=n_expert) +
// 보호(PAGE_NOACCESS) + VEH + (env 지정 시) 계측 오픈. 실패=음수 반환(호출측 기동 중단), 성공=0.
GGML_API int ggml_moe_direct_seal(
    const struct ggml_moe_direct_model *  model,
    const struct ggml_moe_direct_source * sources,
    int32_t                               n_sources);

// sealed attestation 신형식(§4-2③): all/host/nonhost + 실제/필요 슬롯 수. seal 성공 후 유효.
// (att_all 의미: ★부록A dual-scope 이후 = managed = scope 적용 후 관리 대상 수. all scope면 raw==managed.)
GGML_API void ggml_moe_direct_attestation(
    int32_t * out_all, int32_t * out_host, int32_t * out_nonhost,
    int32_t * out_n_slots, int32_t * out_n_slots_required);

// ★부록A dual-scope attestation: raw(등록 전수)·managed(scope 적용 후 관리 대상)·excluded_nextn
// (execution 제외 수 = raw - managed). all scope면 raw==managed·excluded_nextn==0. seal 성공 후 유효.
GGML_API void ggml_moe_direct_scope_attestation(
    int32_t * out_raw, int32_t * out_managed, int32_t * out_excluded_nextn);

GGML_API bool ggml_moe_direct_is_registered(const struct ggml_tensor * t);

// acquire: t가 속한 layer 기준으로 expert_ids 각각의 레코드를 ensure(미적재시 직독)하고
// 슬롯 lease를 1씩 증가. 단일 스레드 호출 전제(내부 mutex 보호). n_tokens_hint: 1=decode >1=prefill.
// 실패(<0): 미등록/미seal/expert_id 범위 밖/슬롯 용량 초과. 읽기 I/O 실패는 즉시 abort.
GGML_API int ggml_moe_direct_acquire(
    const struct ggml_tensor * t,
    const int32_t * expert_ids,
    int32_t n_ids,
    int32_t n_tokens_hint);

// release: 같은 (t, expert_ids)로 lease 감소. 짝 맞춤은 호출측 책임.
GGML_API void ggml_moe_direct_release(
    const struct ggml_tensor * t,
    const int32_t * expert_ids,
    int32_t n_ids);

// ptr: 조회 전용, 락 없는 hot path. acquire 성공 이후에만 유효. 반환 = 슬롯 base + 해당
// (layer,part)의 슬롯 내부 offset. 미등록/미상주 시 NULL.
// ★§Z-⑥ A2 — 그 offset 은 manifest 모드로 갈린다: mode=bin 은 `part_offset`(v2 승계·불변),
//   mode=virtual 은 `data_offset`(SPEC_REPACK_V3 §3-3). 호출자 인터페이스는 양쪽에서 같다.
GGML_API const void * ggml_moe_direct_ptr(const struct ggml_tensor * t, int32_t expert_id);

// 파트 메타(복사 사이트용): t의 등록 part에 대한 슬롯 내부 offset·bytes(층별). 성공=0, 실패<0.
// ★§Z-⑥ A2 — bin=(part_offset, part_bytes) / virtual=(data_offset, slice_bytes). ptr()와 같은
//   분기를 공유하므로 두 API 의 좌표 기준은 항상 일치한다.
GGML_API int ggml_moe_direct_part_info(const struct ggml_tensor * t, uint64_t * offset, uint64_t * bytes);

// 트레이스 레코드용 원시 인덱스: t의 등록 (실제 layer, part_idx). 성공=0, 실패<0.
GGML_API int ggml_moe_direct_registered_layer_part(const struct ggml_tensor * t, int32_t * out_layer, int32_t * out_part);

// ---------------------------------------------------------------------------
// H2D 복사 방출(§4 backend split-copy) — used_ids 오름차순 "사용된" expert id 배열. 각 id에
// emit(ud, ptr(input,id), id*expert_size, expert_size), 연속 run 끝이면 min(512,잔여) zero-fill
// (src==NULL = zero-fill 신호). 반환: 성공/무동작=0, 인자 오류<0, ptr null=GGML_ABORT.
// ---------------------------------------------------------------------------
GGML_API int ggml_moe_direct_emit_expert_copies(
    const struct ggml_tensor * input,
    uint64_t     expert_size,
    uint64_t     total_bytes,
    int32_t      n_expert,
    const int32_t * used_ids,
    int32_t      n_used,
    void (*emit)(void * ud, const void * src, uint64_t dst_offset, uint64_t len),
    void * ud);

// ===========================================================================
// E-2a NEW BOUNDARY SURFACE - the logical-key store API.
//   authority : bench/techdev/repack_space/E2_DESIGN_CONTRACT.md v0.4 FROZEN, section 1
//   upstream  : E1_BOUNDARY_CONTRACT.md v0.5 FROZEN, section 1 K-0 / section 2 L-4
//
// This block is ADDITIVE. Every tensor* entry point above keeps its signature, its
// semantics and its RUNNING behaviour. The two surfaces reach the lease machinery by
// calling the SAME internal primitive DIRECTLY; the old API never routes through a token
// (contract section 1, coexistence rule - a delegating old API would mint tokens with
// nowhere to return them).
//
// Key layering (E-1 K-0):
//   mlpos    index into moe_layers[] - NOT the real layer id
//   expert   expert id in [0, n_expert)
//   record   mlpos * n_expert + expert, the dense ordinal of one (layer, expert)
//   part     the view projection key inside that record (layers[mlpos].parts ordinal)
//   epoch    the store instance epoch; a token or a part ordinal is invalid outside it
//
// HANDLE TABLE LIFETIME (E-2b-0, contract section 1 v0.5). The table behind these handles is
// built to its FULL fixed size before the first epoch is published and never grows, shrinks
// or moves after that - not even across reset_for_test, which frees entries but keeps the
// storage. An entry's payload (its record list and its captured part) is written only while
// the entry is NOT live, and publication is a release-store of the entry's generation, so a
// live entry is immutable and can be read without any lock. Publication and release still
// take the module lock; only the READ path is lock-free.
//   QUIESCENCE (caller's obligation): a token may be released only after every view of it has
//   returned. The CPU end-of-op barrier and the backend scheduler's synchronize already give
//   this. Concurrent view/release of the SAME token is out of scope - there is no reader pin.
// ===========================================================================

// Graceful-drain deadline default, in milliseconds. Frozen for E-2a. This is the finite
// default that replaces the old INFINITE stop policy, and it is what a caller passes when it
// wants the default; a caller-supplied value always wins. Wherever it applies it is ONE
// budget measured from the DRAINING entry, not a per-stage allowance. A value <= 0 means
// "check once, then fail" - it never means "wait forever" and it is never silently replaced
// by this default.
#define MOE_DRAIN_DEADLINE_MS 30000

// Store lifecycle states (contract section 2, row 7).
#define MOE_STORE_RUNNING  0
#define MOE_STORE_DRAINING 1
#define MOE_STORE_STOPPED  2

// A read-only window onto one (record, part). `bytes` is filled by the store. Today no
// consumer receives a byte count from the store at all - both pointer consumers compute it
// from nb02/nb11 themselves - so the transition atoms cross-check this number against that
// computation before the computation goes away.
typedef struct ggml_moe_buffer_view {
    const void * ptr;
    uint64_t     bytes;
} ggml_moe_buffer_view;

// acquire by logical key. On success (0) *out_token receives a non-zero opaque handle. On
// failure nothing is written to *out_token and no lease is taken at all (contract section 2
// row 6, failure atomicity). n_tokens_hint: 1 = decode, >1 = prefill.
//
// E-2b-0: `part` is the batch's OBSERVATION-CORRELATION part, and it is the selector the
// logical-key resolution validates against this layer's parts vector. It is NOT an address
// axis - a record is loaded whole and this call never offsets by it (offsetting is what
// ggml_moe_store_view's own `part` does, and no offset/bytes probe is added here: contract
// section 6, E2-U3). It exists because the demand path stamps (layer, part) onto the Phase A
// ledger and onto the lead-time B stream, while the trace stream carries the consumer's real
// part on the other side of that join - handing a fixed 0 down here would collapse the join
// key on one side only. The token CAPTURES this value, which is why the release below still
// needs nothing but the handle (contract section 1, token internals).
// Out of range (< 0, or >= this layer's parts) is refused: no token, no lease.
GGML_API int ggml_moe_store_acquire(int32_t mlpos, int32_t part,
                                    const int32_t * experts, int32_t n,
                                    int32_t n_tokens_hint, uint64_t * out_token);

// view. The token carries the batch, so a window is uniquely determined even with two
// batches outstanding. A soft failure returns a negative value and leaves *out COMPLETELY
// untouched (no partial write). A handle that is stale, already released, fabricated, or
// outside its epoch is not a soft failure - it is fatal (contract section 6, E2-U1).
//
// E-2b-0: this call takes NO module lock. `part` here stays an INDEPENDENT projection key -
// it is not required to equal the part the token captured at acquire.
GGML_API int ggml_moe_store_view(uint64_t token, int32_t expert, int32_t part,
                                 ggml_moe_buffer_view * out);

// release. Returning the token returns every lease unit it captured; pairing is no longer
// the caller's bookkeeping. Double release and stale handles are fatal.
// E-2b-0: the part handed on to the lease machinery is the one the TOKEN captured, so this
// signature is unchanged (contract section 2, row 5).
GGML_API void ggml_moe_store_release(uint64_t token);

// adapter only - resolve a registered tensor to its logical key exactly once. This is the
// bridge while both surfaces coexist; consumers move onto the logical key over E-2b.
GGML_API int ggml_moe_adapter_resolve(const struct ggml_tensor * t,
                                      int32_t * out_mlpos, int32_t * out_part);

/* XCPU-PIN-BEGIN xcpu-lease-ctx */
/* E-2b-1: the shared lease context the two CPU consumers publish through wdata.
   E2_DESIGN_CONTRACT.md v0.6 FROZEN section 5, E-2b-1 row froze these 3 fields and 16 B.
   acquire runs on ith==0 alone while view runs on EVERY worker, and a uint64 token is not a
   shared object the way a tensor pointer is - a function-local copy would leave every other
   worker looking at its own zero. The slot lives at the END of the mul_mat_id wdata layout
   and at offset 0 of the add_id one; the op's existing start barrier publishes it and the
   existing end barrier is what makes the release safe.
   Both consumers must see ONE definition (ggml-cpu.c is C, ops.cpp is C++), so it goes in
   the header both of them already include - writing the same struct into two files is
   exactly the copy drift HANDOFF_DEV.md section 2-8 is about. */
struct xcpu_lease_ctx {
    uint64_t token;      /* 0 = not issued (this op did not acquire) */
    int32_t  part;       /* -1 = unresolved. After publication, adapter_resolve's part_idx */
    int32_t  reserved;   /* contract-frozen field - this atom neither reads nor writes it (0) */
};
/* XCPU-PIN-END xcpu-lease-ctx */

/* XBE-PIN-BEGIN store-emit-decl */
// E-2b-2 (contract v0.7 section 1): H2D emit by TOKEN. Resolves the handle, epoch, part and
// capture set ONCE at entry, walks the captured records once, and drives the SAME slab /
// zero-fill loop the tensor* form uses (one helper inside the store). `part` is the projection
// key for every expert in used_ids - the backend passes the very variable it acquired with.
// used_ids ascending (as before). 0 = success / no-op (n_used == 0); <0 = argument error
// (null used_ids, n_used < 0, id out of [0,n_expert), part out of this layer's range).
// A used id the token did not capture, or a captured slot whose generation moved, is fatal
// after fallback_hit("emit-expert-copies-ptr-null"); a part whose bytes differ from
// expert_size is fatal (E2-U2 production cross-check, removed in E-2c). Lock-free like view.
GGML_API int ggml_moe_store_emit_expert_copies(
    uint64_t token, int32_t part,
    uint64_t expert_size, uint64_t total_bytes, int32_t n_expert,
    const int32_t * used_ids, int32_t n_used,
    void (*emit)(void * ud, const void * src, uint64_t dst_offset, uint64_t len),
    void * ud);
/* XBE-PIN-END store-emit-decl */

// The live store instance epoch. 0 = no valid instance (never sealed, or torn down).
GGML_API uint64_t ggml_moe_store_epoch(void);

// The current lifecycle state (one of MOE_STORE_*).
GGML_API int32_t ggml_moe_store_state(void);

// The three outstanding counts the drain contract keeps (contract section 1):
//   handles      live batch tokens - a NEW-surface diagnostic, it does not stand in for
//                old-surface leases, which mint no token at all
//   lease_units  granted minus returned lease units across BOTH surfaces; this is the
//                value the drain drives to zero
//   slot_leases  a freshly summed Sigma Slot::lease, the third value of the 3-way check
// Any argument may be null.
GGML_API void ggml_moe_store_outstanding(int64_t * out_handles, int64_t * out_lease_units,
                                         int64_t * out_slot_leases);

// RUNNING -> DRAINING. From here on new acquires are refused on BOTH surfaces, while
// releases of BOTH surfaces are still accepted. 0 on success, <0 if not RUNNING.
GGML_API int ggml_moe_store_drain_begin(void);

// DRAINING -> STOPPED, in this order and no other: wait for the lease-unit sum to reach 0,
// THEN stop the dispatcher, THEN invalidate the epoch. Reversing that order produces a
// token whose return path is already gone. The zero-confirm also requires that no acquire is
// in flight: one that has passed the RUNNING gate owes a lease unit the ledger cannot see
// yet, and confirming zero on top of it stops the store underneath a live acquire.
// The deadline is a SINGLE budget anchored at drain_begin, and every stage spends what is
// left of it - the lease wait first, the dispatcher join with the remainder. No stage starts
// a clock of its own.
//   deadline_ms  > 0  budget of deadline_ms measured FROM drain_begin
//   deadline_ms <= 0  no budget at all: check once, then fail (never the default above)
// If the sum has not reached 0 by then, the store logs the residual LEASE inventory (both
// surfaces - the old one contributes slot rows and no token rows) and aborts WITHOUT
// attempting any cleanup: contract section 2 row 7 forbids quiet progress here.
// 0 on success, <0 if the store was not DRAINING or the dispatcher join timed out.
GGML_API int ggml_moe_store_drain_finish(int32_t deadline_ms);

// ---------------------------------------------------------------------------
// 카운터 (로그·하네스 요약용 — 락 없는 원자적 조회)
// ---------------------------------------------------------------------------

GGML_API uint64_t ggml_moe_direct_touch_events(void);   // VEH가 처리한 AV 이벤트 총수
GGML_API uint64_t ggml_moe_direct_touch_pages(void);    // unique 보호 페이지 터치 수
GGML_API uint64_t ggml_moe_direct_fallback_count(void); // 구조적 폴백(1차 방어) 총수

// trace 모듈의 run 경계 read_bytes/hit/miss 델타 조회(phase: 0=decode 1=prefill).
GGML_API void ggml_moe_direct_phase_read_bytes_hit_miss(
    int32_t phase, uint64_t * out_read_bytes, uint64_t * out_hit, uint64_t * out_miss);
GGML_API void ggml_moe_direct_phase_read_bytes_hit_miss_exact(
    int32_t phase, uint64_t * out_read_bytes, uint64_t * out_hit, uint64_t * out_miss);
GGML_API void ggml_moe_direct_phase_read_bytes_hit_miss_window(
    int32_t phase, uint64_t * out_read_bytes, uint64_t * out_hit, uint64_t * out_miss);

// 원경로 인계 직전 1차 방어 계수. site는 진단용 태그.
GGML_API void ggml_moe_direct_fallback_hit(const char * site);

// ---------------------------------------------------------------------------
// ★프리페치 Phase A 계측 지원(bench/moe-direct/SPEC_PHASEA_WIRE.md) — 관측 전용.
// ---------------------------------------------------------------------------

// 계측 JSONL에 임의 레코드 1줄을 쓴다(공통 필드 type/invocation/session/leg/pid/seq/ts 자동 부착 +
// payload_json 객체 병합). Phase A 모듈이 flush 시점에 phase_a_run 레코드를 방출하는 통로다.
// metrics 미활성이면 false(무동작). 내부에서 모듈 mutex를 취득하므로 **락 밖에서** 호출할 것.
GGML_API bool ggml_moe_direct_metrics_write_record(const char * type, const char * payload_json);

// mlpos(= moe_layers 내 위치) 층에서 experts[0..n-1] 중 현재 **미상주**(캐시에 없음) 개수.
// 음수 id는 건너뛴다. 읽기 전용 — 캐시/LRU/lease를 일절 건드리지 않는다(A1 "신호 시점 absent" 산출용).
GGML_API int32_t ggml_moe_direct_absent_count(int32_t mlpos, const int32_t * experts, int32_t n);

// graph 형상 지문(ordered aggregate) — 매 graph compute 직후 llama-context가 **PA 모드와 무관하게**
// 호출한다. metrics 활성 시 (u32 n_nodes)(u32 n_splits)를 순서대로 FNV-1a 64에 먹이고 terminal
// summary에 moe_graph_count/nodes_digest/splits_digest로 산출한다 — pa_off arm이 pa_timing arm의
// graph 형상과 동일함을 증명할 off측 비교값(PREFETCH_DESIGN §3.3 · SPEC_PHASEA_WIRE §10).
GGML_API void ggml_moe_direct_graph_shape_note(int32_t n_nodes, int32_t n_splits);

// prefetch 억제 marker (llama-mmap.cpp가 호출) — stderr 고유 marker 라인 출력.
GGML_API void ggml_moe_prefetch_marker(uint64_t requested_bytes, uint32_t call_count, uint32_t suppressed_count);

// 종료 finalize(server 종료 경로가 호출) — 계측 terminal summary flush. 멱등.
GGML_API void ggml_moe_direct_finalize(void);


#ifdef __cplusplus
}
#endif
