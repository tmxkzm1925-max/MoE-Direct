#pragma once

// R-B4b (RB_DESIGN.md 2-5, RB4B_ATOM_SPEC.json): the observation/instrumentation cut.
// ggml-moe-direct-observe.cpp carries the 43 definitions this atom moved out of
// ggml-moe-direct.cpp; this header carries what the rest of the tree still has to see of
// them, plus the types and the one constant they took with them.  Every declaration below
// is the core declarator text at the span the move map names - no signature changes.
// CONTRACT (RB_DESIGN 2-2 B-1, inherited): this header neither declares nor includes
// `g_mod` / `ModuleState`.  The types it needs by reference are forward-declared; the
// state header is included by the .cpp side only.  Constants are `inline constexpr`.
// Windows only, exactly like the region it came from.

#ifdef _WIN32

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-moe-phaseb.h"
#include "ggml-moe-direct-types.h"
#include "ggml-moe-direct-manifest.h"

// `Slot`, `PbOpStamp` and `V3CoalRead` appear below by reference or by pointer.  They are
// the state header's records (R-B2) and B-1 bars this header from including it, so this
// file does NOT declare them either: both units that include it - the core TU and the
// observation TU - include `ggml-moe-direct-state.h` first, which defines all three.  A
// forward declaration here would be a new file-scope name the move map does not carry.

namespace ggml_moe_direct_impl {

inline constexpr uint32_t MOE_DIRECT_DISP_STACK    = 1u << 19; // dispatcher 스레드 스택 예약(mem echo 정확값)

enum : uint8_t {
    IO3_T_OK = 1, IO3_T_READ_ERROR = 2, IO3_T_COPY_ERROR = 3, IO3_T_CANCELLED = 4, IO3_T_SHUTDOWN = 5,
};

enum : uint8_t { IO3_C_OK = 1, IO3_C_ERROR = 2, IO3_C_CANCELLED = 3 };

// DTO rule (a) [RB4A_ATOM_SPEC.json :: seam_map.dto_rules] - 방출 시점에 g_mod 를 읽지
//   않도록 포착과 방출을 가른다. 포착은 g_mod.mutex 를 쥔 호출자 안에서만 돈다.
struct MetricsSnapshot {
    json metrics_doc;
};

// CLOSURE ADDITION (RB4B_ATOM_SPEC.json :: closure_additions).  Three moved
// bodies and 15 core sites call it and its body is pure - it names no `g_mod`,
// no core type and no other callee - so the header may legally carry the
// definition.  `inline` is KEPT: a non-inline definition in a header more than
// one TU includes is a duplicate symbol at link time.
inline uint8_t pb_clamp_u8(int32_t v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// INBOUND (25 prototypes): what the rest of the tree still calls in the moved
// cluster, less the two whose DEFINITION this header carries below.
// RB4B_ATOM_SPEC.json :: observe_denominators.derived.inbound_names.
json common_fields(const char * type);
bool io3_all_terminal_locked();
int32_t io3_child_submit_locked(int32_t oi, int32_t part, bool bounce, uint64_t read_offset, uint32_t read_len, int32_t * out_copy_row);
void io3_child_terminal_locked(int32_t row_c, uint8_t status);
void io3_close_record_locked(int32_t oi, uint8_t status, bool had_waiter, uint64_t latch_qpc);
int32_t io3_coal_submit_locked(V3CoalRead & cr, int32_t * out_copy_row0);
void io3_copy_note_enqueue_locked(int32_t row);
void io3_copy_note_terminal_locked(int32_t row, uint8_t worker, uint64_t start_qpc, uint8_t status);
void io3_finalize_locked();
int32_t io3_open_record_locked(int32_t oi);
void metrics_discard_locked();
void metrics_maybe_periodic_locked();
bool metrics_open_if_requested_locked(std::string & err);
bool metrics_write_line(const std::string & line, bool do_flush);
void metrics_write_snapshot_locked(const char * type);
void pb_collect_participants_locked(std::vector<ggml_moe_phaseb_ledger_owner> & out, bool include_ready, bool inflight_only);
void pb_led_emit_locked(ggml_moe_phaseb_ledger & r, const ggml_moe_phaseb_ledger_owner * parts, uint32_t n_parts);
ggml_moe_phaseb_ledger pb_led_new(uint8_t kind);
void pb_led_stamp_from_op(ggml_moe_phaseb_ledger & r, const PbOpStamp * st);
void pb_led_stamp_from_slot(ggml_moe_phaseb_ledger & r, const Slot & sl);
void pb_note_ready_evict_locked(Slot & s);
uint16_t pb_pred_frontier_rank_locked(uint32_t graph_seq, int32_t mlpos, int32_t expert);
void pb_pred_snap_put_locked(const ggml_moe_phaseb_intent & it);
void v3_child_submit_outcome_locked(int32_t row_c, bool pending, bool demand);
void v3_readfile_digest_add_locked(uint16_t layer, uint16_t expert, uint8_t part, uint8_t source_index, uint64_t read_offset, uint32_t read_len);

// INBOUND, HEADER-INLINE (2): the same inbound set, disposed `header-inline` by
// RB4B_ATOM_SPEC.json :: inline_dispositions.  Their bodies name nothing the state
// header owns, so the DEFINITION may sit here with its `inline` kept - which is what
// keeps the core call sites inlining them, and what makes the symbol reachable at all
// (an `inline` definition in the .cpp is emitted only where the optimiser leaves an
// out-of-line copy).  Each REPLACES the prototype it would otherwise have had.
// 참여자 없음 방출(단축형).
inline void pb_led_emit_locked(ggml_moe_phaseb_ledger & r) { pb_led_emit_locked(r, nullptr, 0); }

// owner request 의 박제 튜플로 기입. ★r3 수리 1 이후 **`BLOCKED_INTERVAL` 전용**이다 —
//   그 사건의 owner 는 대기한 demand generation 자신이고, wait/rank 술어(controller §3)가
//   그 generation 의 튜플로 집계돼야 하기 때문이다. in-flight/terminal 레코드의 demand 행은
//   부록 §14.3.3 "demand = stamp NA" 규약을 따르므로 이 함수를 쓰지 않는다.
inline void pb_led_stamp_from_request(ggml_moe_phaseb_ledger & r, uint32_t request_epoch) {
    if (request_epoch == GGML_MOE_PHASEB_NA_U32) return;
    int32_t k = 0, n = 0, qd = 0, rev = 0; uint64_t aseq = GGML_MOE_PHASEB_NA_U64;
    ggml_moe_phaseb_run_stamp(request_epoch, &k, &n, &qd, &rev, &aseq);
    r.stamp_k        = pb_clamp_u8(k);
    r.stamp_n        = pb_clamp_u8(n);
    r.stamp_qd       = pb_clamp_u8(qd);
    r.tuple_revision = pb_clamp_u8(rev);
    r.admission_seq  = aseq;
}


// CORE-RESIDENT CALLEES (2): defined in ggml-moe-direct.cpp, called from the
// observation TU.  RB4B_ATOM_SPEC.json :: seam_map.members `b4-control-q-bytes`
// (promoted out of the anonymous namespace, `inline` dropped, body unchanged) and
// `b3-io-param-snapshot` (the A3 accessor: the two scheduling-K9 members reach this
// unit BY VALUE and no pointer into `g_mod` crosses the boundary).
uint64_t control_q_bytes_locked();
void v3_io_param_snapshot_locked(int32_t (&val)[6], uint8_t (&src)[6]);

} // namespace ggml_moe_direct_impl

#endif // _WIN32
