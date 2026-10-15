#pragma once

// R-B5b (RB_DESIGN.md 2-6, RB5B_ATOM_SPEC.json): the dispatch/queue cut.
// ggml-moe-direct-dispatch.cpp carries the 90 definitions this atom moved out of
// ggml-moe-direct.cpp; this header carries what the rest of the tree still has to see
// of them, the three types they took with them, and the prototypes the moved bodies
// need for the definitions that stayed behind.  Every declaration below is the core
// declarator text at the span the move map names - no signature changes.
// CONTRACT (RB_DESIGN 2-2 B-1, inherited): this header neither declares nor includes
// `g_mod` / `ModuleState`.  The records it names by reference belong to the state
// header (R-B2) and both units that include this file include that one first.
// Windows only, exactly like the region it came from.

#ifdef _WIN32

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

// `BatchMiss`, `IoChannel`, `Slot`, `ControlCmd`, `DemandPlan`, `PbOpStamp` and `Phase`
// appear below by reference, by pointer or by value.  They are the state header's
// records (R-B2) and B-1 bars this header from including it, so this file does NOT
// declare them either: both units that include it - the core TU and the dispatch TU -
// include `ggml-moe-direct-state.h` first, which defines all of them.

namespace ggml_moe_direct_impl {

// TYPES (3): RB5B_ATOM_SPEC.json :: design_type_list.  They move with the cluster and
// name no module state, so the destination header may carry them.

// child 1건의 읽기 기하(§2-4). **aligned part 전용** — 비4K 바운스는 A4 이고, 이 원자는
// 비4K part 를 가진 프로파일을 seal 단계에서 아예 거부한다(v3_seal_shape_supported).
//   read_offset = abs_offset + e*slice − bracket_head   (A 배수 — §2-4 정렬 상수 증명)
//   read_len    = region                                (A 배수)
//   dest        = slot_base + slot_offset               (A 배수 — NO_BUFFERING 정렬 요건)
// 실데이터는 dest + bracket_head = slot_base + data_offset 에 놓인다(= `ptr()` 반환 지점).
struct V3ChildGeom {
    HANDLE   h;
    uint64_t offset;
    uint32_t len;
    char *   dest;
    // ★A4 — 바운스 child 면 dest 는 슬롯이 아니라 staging 이고, 실데이터는 그 안
    //   `src_head` 바이트 뒤에서 시작한다. copy 가 그 구간만 슬롯으로 옮긴다.
    bool     bounce   = false;
    uint64_t src_head = 0;
    char *   slot_dest = nullptr;   // 최종 착지(= slot_base + slot_offset)
    int32_t  staging  = -1;
};

// ★프리페치 admission — dispatcher **턴당 후보 1건**(§4 "턴당 1건"). 호출 위치가 demand pump
//   뒤이므로 demand 절대 우선이 구조로 보장된다. ★v1은 capacity 확보용 prefetch cancel 금지
//   (§5.2 ⓕ) — 그런 코드는 이 파일에 존재하지 않는다(pb_capacity_cancel_attempt는 상수 0).
//   cut_mode=true(PB_FINALIZE_CUT ④)에서는 신규 제출을 허가하지 않으므로 후보는 capacity
//   없음으로 처분된다(잔여 rank<k 전건 소비 완결).
// ★R1 #2 ② 반환값: 0=EMPTY(소비할 rank 없음) / 1=DISPOSED(사전 처분·admission drop —
//   progress로 인정해 남은 rank가 같은 turn 흐름에서 계속 소비된다) / 2=ISSUED(claim+제출 대기).
// ★★PB virtual A2 추가: 3=DEFERRED(§2-0 A0-4 drain-defer) — **intent·C3 를 소비하지 않고**
//   turn 을 끝낸다. progress 가 **아니며**, 같은 상태로 재진입하지 않도록 park 래치를 세운다.
enum PbAdmitResult : int {
    PB_ADMIT_EMPTY = 0, PB_ADMIT_DISPOSED = 1, PB_ADMIT_ISSUED = 2, PB_ADMIT_DEFERRED = 3,
};

// ★★R-3(교차 r1 F5 수리) — **park 사유 지문**. park 의 정확한 유효 조건은 "credit 이 안 변함"이
//   아니라 "**그 park 를 세운 사유가 그대로임**"이다(A0-4 ② "**변화 없는** queued intent").
//   지문이 같으면 재시도해도 결과가 같으므로 계속 막고, 달라지면 새 사건이므로 내린다(③).
// ★사유별로 지문을 가르는 이유는 `pbv_park_reason` 선언부 주석 참조(섞으면 A0-4 ④ 가 깨진다).
enum : uint8_t { PBV_PARK_NONE = 0, PBV_PARK_GATE = 1, PBV_PARK_CREDIT = 2 };

// DECLARE IN PLACE (3): RB5B_ATOM_SPEC.json :: declare_in_place.  The stop-join
// telemetry counters are written by two moved helpers and read by a core-resident API
// entry point, so the DEFINITION stays in the core TU and this header declares them.
extern std::atomic<int64_t> g_xs_stopjoin_deadline;
extern std::atomic<int32_t> g_xs_stopjoin_left;
extern std::atomic<int32_t> g_xs_stopjoin_n;

// INBOUND (15 prototypes): what the rest of the tree still calls in the moved cluster.
// The capture's section D-3 `DISP inbound` set; RB_DESIGN 2-6 B-2 bounds it at N_D 15.
uint64_t control_q_bytes_locked();
int32_t cq_alloc();
void cq_free(int32_t ci);
int cq_submit_and_wait(int32_t ci);
bool dispatcher_start_locked(std::string & err);
void dispatcher_stop_and_join(bool abort_mode);
int dispatcher_stop_and_join_bounded(bool abort_mode, int32_t timeout_ms);
int dispatcher_stop_and_join_until(bool abort_mode, int64_t deadline_abs_ms);
bool pb_partition_op_locked(int32_t oi, uint8_t partition);
int32_t pbv_ncap_locked(int32_t mlpos, int32_t n_stamp);
int32_t pbv_wave_width(int32_t mlpos);
bool persistent_active();
bool persistent_fail_closed();
bool v3_copy_start_locked(int32_t capacity, std::string & err);
void v3_copy_stop_and_join();

// CORE-RESIDENT CALLEES (19): defined in ggml-moe-direct.cpp, called from the dispatch
// TU.  RB5B_ATOM_SPEC.json :: seam_map.members - promoted out of the anonymous
// namespace, `inline` dropped where the definition carried it, bodies unchanged.
void io_finish_locked(BatchMiss & m, DWORD bytes, bool was_pending, int32_t mlpos, Phase phase, uint8_t origin, uint32_t want_override);
int io_submit_locked(BatchMiss & m, IoChannel & ch, int32_t mlpos, uint32_t batch_order, uint8_t origin, bool fail_soft);
void pb_drain_status_locked();
int32_t pb_find_row_locked(uint32_t graph_seq, int32_t mlpos);
void pb_finish_demand_command_locked(ControlCmd & c);
bool pb_is_served_locked(uint32_t graph_seq, int32_t mlpos);
bool pb_mark_served_locked(uint32_t graph_seq, int32_t mlpos);
int32_t pb_note_submit_locked(int64_t record_idx, int32_t mlpos, uint64_t file_offset, int32_t slot, uint64_t generation, uint8_t origin, uint16_t rank, uint64_t intent_ordinal, uint32_t req_epoch, uint32_t graph_seq, bool is_prefill, bool is_retry, uint64_t retry_of, uint64_t * out_submit_seq, const struct PbOpStamp * owner_stamp);
void pb_note_terminal_locked(int32_t io_index, uint8_t origin, uint8_t status, bool had_waiter, bool ever_joined, uint64_t publish_seq, uint32_t req_epoch, uint32_t graph_seq, uint64_t intent_ordinal, uint16_t rank, int64_t record_idx, const struct PbOpStamp * owner_stamp);
uint64_t pb_qpc_to_us(uint64_t ticks);
uint64_t pb_row_key(uint32_t graph_seq, uint16_t layer);
int32_t pb_row_locked(uint32_t graph_seq, uint32_t request_index, uint32_t decode_ordinal, int32_t mlpos, uint8_t is_prefill);
void pb_slot_stamp_clear(Slot & s);
void pb_transition_locked(uint8_t ev, int64_t record_id, int32_t slot, uint64_t generation, uint64_t intent_ordinal);
int plan_demand_locked(DemandPlan & p, int32_t mlpos, int32_t part_idx, const int32_t * ids, int32_t n_ids, Phase phase, bool pa_on, uint64_t pa_t_enter, bool persistent, uint32_t pb_epoch, uint32_t pb_gepoch, uint32_t pb_gseq, int32_t pb_cmd);
bool victim_better_locked(int32_t cand, int32_t best);
int64_t xstore_budget_deadline_ms(int32_t timeout_ms);
int32_t xstore_budget_left_ms(int64_t deadline_abs_ms);
void xstore_note_lease_returned(int32_t units);

}  // namespace ggml_moe_direct_impl

// RB5B_ATOM_SPEC.json :: header_extra_prototypes.  `dfac_q_request_stop` moved into
// the dispatch TU with `q_lock_move_set` and lost its `inline` specifier; its one
// staying caller is `ggml_moe_direct_finalize` (API) in the core TU, which needs this
// declaration to bind to the emitted symbol.  `extern "C"`, outside the named
// namespace, because the definition's linkage is unchanged by the move.
extern "C" void dfac_q_request_stop();

#endif  // _WIN32
