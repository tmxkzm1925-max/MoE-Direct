// R-B3b (RB_DESIGN.md 2-4, RB3B_ATOM_SPEC.json): the test-only entry point unit.
// The 88 definitions below are the core text of cluster TEST at the spans the move map names,
// moved verbatim - declarators, linkage (`extern "C"`) and body tokens unchanged - in source
// order, which is the order that compiled in the core TU.  The one object that travels with
// them, `g_last_seal_err`, takes the named namespace (RB3B_ATOM_SPEC.json :: design_type_list).
// The thirteen core definitions they call are declared in ggml-moe-direct-internal.h.
// This unit sits inside ggml-base and is distinct from the E-2c test-only implementation TU.
// Windows only, exactly like the region it came from: the core TU compiles its non-Windows
// half from the stub region at the bottom of that file, and this unit has no stub half.

// The head include block of ggml-moe-direct.cpp, reproduced so the moved definitions compile
// in the environment they were written in.
#include "ggml-moe-direct.h"
#include "ggml-moe-direct-internal.h"
#include "ggml-moe-trace.h"
#include "ggml-moe-phasea.h"
#include "ggml-moe-phaseb.h"
#include "ggml-moe-shadow.h"   // ★C3 캐시 view 의 선언(구현은 이 파일 — 캐시 상태 소유자)
#include "ggml-moe-leadtime.h" // ★PROBE_LEADTIME_SPEC_R1 §2 — stamp B(T1) 결선 + arm/flush(기본 OFF)
#include "ggml-moe-gpuearly.h" // ★PROBE_GPU_EARLY_SPEC_R1 §4 — C 축 arm/flush 결선(기본 OFF)
#include "ggml.h"
#include "gguf.h"   // ★§Z-⑥ A1 — mode=virtual 부팅 대조가 원본 GGUF 헤더를 재파싱한다
#include "ggml-impl.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

#include "nlohmann/json.hpp"

#include <process.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>   // E-2b-0: std::unique_ptr owns the fixed-size lease-handle table
#include <mutex>
#include <new>      // E-2b-0 r4: placement new and std::launder for the in-place payload
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

#include "ggml-moe-direct-types.h"
#include "ggml-moe-direct-manifest.h"
#include "ggml-moe-direct-state.h"
#include "ggml-moe-direct-observe.h"
#include "ggml-moe-direct-dispatch.h"

using namespace ggml_moe_direct_impl;

// The ABI literal the M5 bundle-assembly gate compares across its A/B/C axes
// (section 2-5 D2 - this is not a seal-time wire check).
#define OPEN_ARCH_TEMPLATE_ABI_STR "open-arch-template/1"

namespace ggml_moe_direct_impl {

// ★R14 후속 — 직전 **테스트 seal** 시도의 거부 사유 원문(관문이 진단문 품질을 검사한다).
std::string g_last_seal_err;

}  // namespace ggml_moe_direct_impl

extern "C" {

// section 13 of the core TU (its banner stays there): 테스트 전용 진입점

int ggml_moe_direct_seal_for_test(const char * dir,
                                  const struct ggml_moe_direct_model * model,
                                  const struct ggml_moe_direct_source * sources,
                                  int32_t n_sources,
                                  uint64_t budget_bytes,
                                  uint32_t touch_abort,
                                  const char * expects_dir,
                                  const char * test_profile_id,
                                  const char * test_digest,
                                  const char * test_scope) {
    if (!dir || !model || !sources || n_sources <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    std::wstring dir_w = utf8_to_wide(dir);
    std::wstring expects_dir_w = utf8_to_wide(expects_dir ? expects_dir : "");
    uint32_t ta = touch_abort != 0 ? touch_abort : PROD_DEFAULT_TOUCH_ABORT;
    std::string err;
    // 테스트 전용 카탈로그 주입(프로덕션 EXPECT_CATALOG 우회 아님 — override_profile_id 경유).
    // ★부록A: test_scope 를 카탈로그 등재 scope 로 주입 → manifest routed_scope 대조를 합성으로 행사.
    int rc = seal_common_locked_v2(dir_w, *model, sources, n_sources,
                                   /*allow_sector_fallback=*/true, budget_bytes, ta,
                                   expects_dir_w, test_profile_id, test_digest, test_scope, err);
    if (rc != 0) log_err("ggml_moe_direct_seal_for_test failed: %s", err.c_str());
    // ★R14 후속 — 거부 **진단문 자체**를 관문이 검사할 수 있게 남긴다(리드 조건 1: 하한값·처방
    //   동반 여부는 사람이 눈으로 보는 게 아니라 selftest 가 잠근다). 테스트 경로 전용.
    g_last_seal_err = err;
    return rc;
}

// ★R18(교차 r6 G7) — **호출자 버퍼 복사형**. 구 판은 잠금 안에서 내부 `std::string` 의
//   `c_str()` 를 얻어 잠금을 푼 뒤 반환했다 — 다음 seal 이 다른 스레드에서 문자열을 바꾸면
//   반환 포인터의 수명이 보호되지 않는다(현 순차 selftest 에서는 안전했지만 API 계약으로는
//   거짓이다). 이제 잠금 안에서 **복사까지 끝낸다**.
void ggml_moe_direct_test_last_seal_error(char * buf, int32_t cap) {
    if (!buf || cap <= 0) return;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const size_t n = g_last_seal_err.size() < (size_t)(cap - 1) ? g_last_seal_err.size() : (size_t)(cap - 1);
    std::memcpy(buf, g_last_seal_err.data(), n);
    buf[n] = '\0';
}

// R-B5a facade: the test arming flags (RB_DESIGN.md:1221 section 2-6, read as group (vi)
// "API and TEST dispatcher state" per the lead's ruling of 2026-09-19 on the sites the seven
// design groups do not name). Sixteen dispatcher-owned one-shot arming flags are written by a
// `ggml_moe_direct_test_*` entry point and again by `ggml_moe_direct_reset_for_test`; each one
// is stated here and nowhere else outside the dispatch cluster. Every member performs the store
// its call site performed, with the same value, the same cast and the same ordering - the value
// expression stays at the call site and is passed in. The seventeenth flag,
// `v3_test_fail_after_acquire`, already has `dfac_test_fail_after_acquire_set_locked` from D2a
// and is not repeated here. `_locked` names the members whose caller already holds `g_mod.mutex`
// at the site; `dfac_v3_copy_pause_set` is the one that does not, because that flag is an
// atomic written with a release store before its caller takes the lock.

// K11: v3_test_fail_dispatcher_start
inline void dfac_v3_test_fail_dispatcher_start_set_locked(int32_t v) {
    g_mod.v3_test_fail_dispatcher_start = v;
}

// K11: v3_test_reverse_children
inline void dfac_v3_test_reverse_children_set_locked(int32_t v) {
    g_mod.v3_test_reverse_children   = v;
}

// K11: v3_test_fail_child
inline void dfac_v3_test_fail_child_set_locked(int32_t v) {
    g_mod.v3_test_fail_child        = v;
}

// K6: v3_copy_pause
inline void dfac_v3_copy_pause_set(int32_t v) {
    g_mod.v3_copy_pause.store(v, std::memory_order_release);
}

// K9: v3_test_fail_copy
inline void dfac_v3_test_fail_copy_set_locked(int32_t v) {
    g_mod.v3_test_fail_copy = v;
}

// K11: v3_test_sync_child
inline void dfac_v3_test_sync_child_set_locked(int32_t v) {
    g_mod.v3_test_sync_child = v;
}

// K9: v3_test_coal_overrun
inline void dfac_v3_test_coal_overrun_set_locked(int32_t v) {
    g_mod.v3_test_coal_overrun = v;
}

// K9: v3_test_coal_unpin
inline void dfac_v3_test_coal_unpin_set_locked(int32_t v) {
    g_mod.v3_test_coal_unpin   = v;
}

// K12: test_pb_error_next
inline void dfac_test_pb_error_next_set_locked(int32_t v) {
    g_mod.test_pb_error_next = v;
}

// K12: test_cut_residual_force
inline void dfac_test_cut_residual_force_set_locked(int32_t v) {
    g_mod.test_cut_residual_force = v;
}

// K12: test_stop_race
inline void dfac_test_stop_race_set_locked(int32_t v) {
    g_mod.test_stop_race = v;
}

// K7: test_cont_credit_fail_next
inline void dfac_test_cont_credit_fail_next_set_locked(int32_t v) {
    g_mod.test_cont_credit_fail_next = v;
}

// K12: test_hold_submit
inline void dfac_test_hold_submit_set_locked(int32_t v) {
    g_mod.test_hold_submit = v;
}

// K12: test_hold_retire
inline void dfac_test_hold_retire_set_locked(int32_t v) {
    g_mod.test_hold_retire = v;
}

// K12: test_abort_after_submit
inline void dfac_test_abort_after_submit_set_locked(int32_t v) {
    g_mod.test_abort_after_submit = v;
}

// K12: test_force_sync_next
inline void dfac_test_force_sync_next_set_locked(int32_t v) {
    g_mod.test_force_sync_next = v;
}

void ggml_moe_direct_test_v3_arm_dispatcher_fail(int32_t on) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_v3_test_fail_dispatcher_start_set_locked(on);
}

// ★§Z-⑥ A1 — 직전 seal 시도가 virtual 관문을 어디까지 통과했는지(V3Stage). 프로덕션 경로는
//   이 값을 읽지 않는다(순수 관측). 네거티브 케이스가 "의도한 관문에서 거부됐다"를 주장하는
//   창구다 — rc<0 만으로는 앞 관문이 우연히 잡은 것과 구분되지 않는다.
int32_t ggml_moe_direct_test_v3_stage(void) { return g_v3_stage.load(std::memory_order_relaxed); }

int ggml_moe_direct_test_v3_probe(int32_t * n_handles, int32_t * n_slots, int32_t * n_parts,
                                  int32_t * att_managed, int32_t * att_host, int32_t * n_prot_intervals,
                                  uint64_t * pool_bytes, uint64_t * slot_stride_max,
                                  uint64_t * handle_bytes_total) {
    if (!g_v3_probe.valid) return -1;
    if (n_handles)        *n_handles        = g_v3_probe.n_handles;
    if (n_slots)          *n_slots          = g_v3_probe.n_slots;
    if (n_parts)          *n_parts          = (int32_t)g_v3_probe.part_addr.size();
    if (att_managed)      *att_managed      = g_v3_probe.att_managed;
    if (att_host)         *att_host         = g_v3_probe.att_host;
    if (n_prot_intervals) *n_prot_intervals = g_v3_probe.n_prot_intervals;
    if (pool_bytes)       *pool_bytes       = g_v3_probe.pool_bytes;
    if (slot_stride_max)  *slot_stride_max  = g_v3_probe.slot_stride_max;
    if (handle_bytes_total) *handle_bytes_total = g_v3_probe.handle_bytes_total;
    return 0;
}

// ★v0.3.1 이월 원자 ⓐ — 실패한 seal 의 rollback 이 **전건을 되돌렸는지**를 그 자리에서 읽는다.
//   위 `..._test_v3_probe` 는 확보 **직후 스냅샷**이라 "지금 무엇이 남아 있는가"를 못 본다.
//   후속 seal 이 전역을 재게시하면 stale alias 가 관측에서 지워지므로 실패 직후가 유일한 창이다.
//   축 11 = 공용 9 + virtual 전용 2. bin 은 virtual 2축을 애초에 잡지 않으므로 그 둘의 0 은
//   "되돌렸다"가 아니라 **미보유 전제**의 확인이다(그 구분은 호출자 주장이 진다).
void ggml_moe_direct_test_seal_residue(int32_t * out_data_file_valid,
                                       int32_t * out_pool_base_nonnull,
                                       uint64_t * out_pool_bytes,
                                       int32_t * out_n_slots,
                                       int32_t * out_n_slot_entries,
                                       int32_t * out_n_records,
                                       int32_t * out_n_registry,
                                       int32_t * out_n_prot_intervals,
                                       int32_t * out_n_reg_tensor_info,
                                       int32_t * out_n_source_files,
                                       int32_t * out_veh_handle_nonnull) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_data_file_valid)    *out_data_file_valid    = (g_mod.data_file != INVALID_HANDLE_VALUE) ? 1 : 0;
    if (out_pool_base_nonnull)  *out_pool_base_nonnull  = (g_mod.pool_base != nullptr) ? 1 : 0;
    if (out_pool_bytes)         *out_pool_bytes         = (uint64_t)g_mod.pool_bytes;
    if (out_n_slots)            *out_n_slots            = g_mod.n_slots;
    if (out_n_slot_entries)     *out_n_slot_entries     = (int32_t)g_mod.slots.size();
    if (out_n_records)          *out_n_records          = (int32_t)g_mod.record_table.size();
    if (out_n_registry)         *out_n_registry         = (int32_t)g_mod.registry.size();
    if (out_n_prot_intervals)   *out_n_prot_intervals   = (int32_t)g_mod.prot_intervals.size();
    if (out_n_reg_tensor_info)  *out_n_reg_tensor_info  = (int32_t)g_mod.reg_tensor_info.size();
    if (out_n_source_files)     *out_n_source_files     = (int32_t)g_mod.source_files.size();
    if (out_veh_handle_nonnull) *out_veh_handle_nonnull = (g_mod.veh_handle != nullptr) ? 1 : 0;
}

void ggml_moe_direct_test_v3_child_stats(uint64_t * submitted, uint64_t * retired,
                                         uint64_t * op_terminals, uint64_t * publishes) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (submitted)    *submitted    = g_mod.v3_children_submitted;
    if (retired)      *retired      = g_mod.v3_children_retired;
    if (op_terminals) *op_terminals = g_mod.v3_op_terminals;
    if (publishes)    *publishes    = g_mod.v3_publishes;
}

// ★★R-1 — wave 기계 발화의 직접 관측창구(정의·사유=헤더 주석).
void ggml_moe_direct_test_v3_wave_stats(uint64_t * out_waves, int32_t * out_part_parallelism) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_waves)            *out_waves            = g_mod.v3_waves_submitted;
    if (out_part_parallelism) *out_part_parallelism = g_mod.v3_io_param_val[3];
}

void ggml_moe_direct_test_v3_arm(int32_t reverse_children, int32_t fail_child_part,
                                 int32_t fail_after_acquire) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (reverse_children  >= 0) dfac_v3_test_reverse_children_set_locked(reverse_children);
    if (fail_child_part   >= -1) dfac_v3_test_fail_child_set_locked(fail_child_part);
    if (fail_after_acquire >= 0) dfac_test_fail_after_acquire_set_locked(fail_after_acquire);
}

void ggml_moe_direct_test_v3_bounce_stats(uint64_t * copies_done, uint64_t * copies_failed,
                                          uint64_t * backpressure_stalls,
                                          int32_t * staging_busy, int32_t * staging_count) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (copies_done)         *copies_done         = g_mod.v3_copies_done;
    if (copies_failed)       *copies_failed       = g_mod.v3_copies_failed;
    if (backpressure_stalls) *backpressure_stalls = g_mod.v3_backpressure_stalls;
    if (staging_count)       *staging_count       = g_mod.v3_staging_count;
    if (staging_busy) {
        int32_t n = 0;
        for (int32_t i = 0; i < g_mod.v3_staging_count; i++) if (g_mod.v3_staging_busy[(size_t)i]) n++;
        *staging_busy = n;
    }
}

void ggml_moe_direct_test_v3_io_stats(int32_t * out_io_param_val, uint8_t * out_io_param_src,
                                      uint64_t * out_child_pending_demand,
                                      uint64_t * out_child_immediate_demand,
                                      uint64_t * out_readfile_ops,
                                      uint64_t * out_readfile_digest,
                                      int32_t * out_io3_rows_alloc) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_io_param_val) for (int i = 0; i < 6; i++) out_io_param_val[i] = g_mod.v3_io_param_val[i];
    if (out_io_param_src) for (int i = 0; i < 6; i++) out_io_param_src[i] = g_mod.v3_io_param_src[i];
    if (out_child_pending_demand)   *out_child_pending_demand   = g_mod.v3_child_pending_demand;
    if (out_child_immediate_demand) *out_child_immediate_demand = g_mod.v3_child_immediate_demand;
    if (out_readfile_ops)           *out_readfile_ops           = g_mod.v3_readfile_ops;
    if (out_readfile_digest)        *out_readfile_digest        = g_mod.v3_readfile_digest;
    if (out_io3_rows_alloc)
        *out_io3_rows_alloc = (int32_t)(g_mod.io3_op_row.size() + g_mod.io3_chan_row.size());
}

// ★★PB virtual 배선 관측(헤더 주석이 축 정의의 원본 — 여기서는 되풀이하지 않는다).
void ggml_moe_direct_test_pbv_stats(int32_t * out_rec_active, int32_t * out_rec_inflight,
                                    int32_t * out_child_reserved,
                                    int32_t * out_child_inflight_demand,
                                    int32_t * out_child_inflight_prefetch,
                                    uint64_t * out_deferred_credit_attempts,
                                    uint64_t * out_skipped_drain_turns,
                                    int32_t * out_admit_parked) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_rec_active)                *out_rec_active                = g_mod.pbv_rec_active;
    if (out_rec_inflight)              *out_rec_inflight              = g_mod.pbv_rec_inflight;
    if (out_child_reserved)            *out_child_reserved            = g_mod.pbv_child_reserved;
    if (out_child_inflight_demand)     *out_child_inflight_demand     = g_mod.v3_inflight_demand;
    if (out_child_inflight_prefetch)   *out_child_inflight_prefetch   = g_mod.v3_inflight_prefetch;
    if (out_deferred_credit_attempts)  *out_deferred_credit_attempts  = g_mod.pb_admit_deferred_credit_attempts;
    if (out_skipped_drain_turns)       *out_skipped_drain_turns       = g_mod.pb_admit_skipped_drain_turns;
    if (out_admit_parked)              *out_admit_parked              = g_mod.pbv_admit_parked ? 1 : 0;
}

int32_t ggml_moe_direct_test_pbv_ncap(int32_t mlpos, int32_t n_stamp, int32_t * out_wave) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.layers.size()) { if (out_wave) *out_wave = 0; return -1; }
    if (out_wave) *out_wave = pbv_wave_width(mlpos);
    return pbv_ncap_locked(mlpos, n_stamp);
}

// ★★PB virtual — A0-4 ② wake 마스킹의 직접 관측(정의·사유는 헤더 주석이 원본).
//   `g_mod.mutex` 를 잡고 읽으므로 dispatcher 는 pass 밖에 있고, event 상태는 **직전 pass 가
//   남긴 것**으로 안정된다. 조회는 command 를 발행하지 않는다(관측이 상태를 만들지 않는다).
void ggml_moe_direct_test_pbv_wake_state(int32_t * out_command_event_signaled,
                                         int32_t * out_intent_depth,
                                         int32_t * out_admit_parked) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_command_event_signaled) {
        *out_command_event_signaled =
            (g_mod.command_event && WaitForSingleObject(g_mod.command_event, 0) == WAIT_OBJECT_0) ? 1 : 0;
    }
    if (out_intent_depth)  *out_intent_depth  = ggml_moe_phaseb_intent_depth();
    if (out_admit_parked)  *out_admit_parked  = g_mod.pbv_admit_parked ? 1 : 0;
}

int32_t ggml_moe_direct_test_io3_children(int32_t max, uint64_t * out_submit_seq,
                                          uint64_t * out_terminal_seq, uint8_t * out_status) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const int32_t n = (int32_t)g_mod.io3_children.size();
    for (int32_t i = 0; i < n && i < max; i++) {
        const Io3Child & c = g_mod.io3_children[(size_t)i];
        if (out_submit_seq)   out_submit_seq[i]   = c.submit_event_seq;
        if (out_terminal_seq) out_terminal_seq[i] = c.terminal_event_seq;
        if (out_status)       out_status[i]       = c.terminal_status;
    }
    return n;
}

// ★F1 — children 창구의 copies 짝. 파일 flush(finalize) 없이 copy obligation 행의 **내용**을
//   본다: 실행된 copy 는 worker·start·done 이 서고 status 가 확정돼야 하고, 미실행 의무 행은
//   worker=0xFF·시각 0·CANCELLED 로 남는다(§5-5). 반환=현재 행 수(최대 max 개 복사).
int32_t ggml_moe_direct_test_io3_copies(int32_t max, uint64_t * out_enqueue_qpc,
                                        uint64_t * out_start_qpc, uint64_t * out_done_qpc,
                                        uint8_t * out_worker, uint8_t * out_status) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const int32_t n = (int32_t)g_mod.io3_copies.size();
    for (int32_t i = 0; i < n && i < max; i++) {
        const Io3Copy & c = g_mod.io3_copies[(size_t)i];
        if (out_enqueue_qpc) out_enqueue_qpc[i] = c.enqueue_qpc;
        if (out_start_qpc)   out_start_qpc[i]   = c.start_qpc;
        if (out_done_qpc)    out_done_qpc[i]    = c.done_qpc;
        if (out_worker)      out_worker[i]      = c.worker;
        if (out_status)      out_status[i]      = c.status;
    }
    return n;
}

void ggml_moe_direct_test_v3_arm_copy(int32_t pause, int32_t fail_copy_part) {
    if (pause >= 0) dfac_v3_copy_pause_set(pause);
    if (fail_copy_part >= -1) {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        dfac_v3_test_fail_copy_set_locked(fail_copy_part);
    }
}

void ggml_moe_direct_test_v3_arm_sync_child(int32_t part) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_v3_test_sync_child_set_locked(part);
}

void ggml_moe_direct_test_v3_coalesce_stats(int32_t * out_phase, uint64_t * out_min_bytes,
                                            uint64_t * out_max_bytes, int32_t * out_buffers,
                                            int32_t * out_slots, uint64_t * out_groups,
                                            uint64_t * out_records, uint64_t * out_scatters,
                                            int32_t * out_last_reject, uint32_t * out_reject_mask,
                                            uint64_t * out_reads_retired,
                                            int32_t * out_copy_outstanding,
                                            uint64_t * out_copy_credit_stalls) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_phase)      *out_phase      = (int32_t)g_mod.v3_coal_phase;
    if (out_min_bytes)  *out_min_bytes  = g_mod.v3_coal_min_bytes;
    if (out_max_bytes)  *out_max_bytes  = g_mod.v3_coal_max_bytes;
    if (out_buffers)    *out_buffers    = g_mod.v3_coal_buffers;
    if (out_slots)      *out_slots      = g_mod.v3_coal_count;
    if (out_groups)     *out_groups     = g_mod.v3_coal_groups;
    if (out_records)    *out_records    = g_mod.v3_coal_records;
    if (out_scatters)   *out_scatters   = g_mod.v3_coal_scatters;
    if (out_last_reject)*out_last_reject= (int32_t)g_mod.v3_coal_last_reject;
    if (out_reject_mask) *out_reject_mask = g_mod.v3_coal_reject_mask;
    if (out_reads_retired)      *out_reads_retired      = g_mod.v3_coal_reads_retired;
    if (out_copy_outstanding)   *out_copy_outstanding   = g_mod.v3_copy_outstanding;
    if (out_copy_credit_stalls) *out_copy_credit_stalls = g_mod.v3_copy_credit_stalls;
}

void ggml_moe_direct_test_v3_set_io3_cap(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    g_mod.io3_buf_cap_bytes = bytes;
}

void ggml_moe_direct_test_v3_arm_coalesce(int32_t overrun, int32_t unpin) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_v3_test_coal_overrun_set_locked(overrun);
    dfac_v3_test_coal_unpin_set_locked(unpin);
}

int32_t ggml_moe_direct_test_v3_retire_order(int32_t * out_parts, int32_t max_n) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const int32_t n = (int32_t)g_mod.v3_retire_order.size();
    if (out_parts) {
        for (int32_t i = 0; i < n && i < max_n; i++) out_parts[i] = g_mod.v3_retire_order[(size_t)i];
    }
    return n;
}

void ggml_moe_direct_test_v3_sync_stats(uint64_t * sync_children, uint64_t * async_children) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (sync_children)  *sync_children  = g_mod.v3_children_sync;
    if (async_children) *async_children = g_mod.v3_children_async;
}

int ggml_moe_direct_test_v3_probe_part(int32_t flat_index, uint64_t * offset, uint64_t * bytes) {
    if (!g_v3_probe.valid) return -1;
    if (flat_index < 0 || (size_t)flat_index >= g_v3_probe.part_addr.size()) return -1;
    if (offset) *offset = g_v3_probe.part_addr[(size_t)flat_index].first;
    if (bytes)  *bytes  = g_v3_probe.part_addr[(size_t)flat_index].second;
    return 0;
}

// R-B5a facade (iv): command request / result lifecycle (RB_DESIGN.md:1221 section
// 2-6 (iv)). The caller states the command it wants captured; `g_mod.cmds[ci]` is
// filled here and nowhere outside the dispatch cluster. The statements below are the
// ones this member replaced, in the same order, with the caller's lock unchanged.
inline void dfac_cmd_query_lease_stats_fill_locked(int32_t ci) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.kind = CMD_TEST_QUERY; c.qkind = QK_LEASE_STATS;
}

// 테스트 전용: 리스 통계(leased 슬롯 수 · 총 lease 카운트) — F3 release 복귀·F4 lease pin 검증용.
void ggml_moe_direct_test_lease_stats(int32_t * out_leased_slots, int64_t * out_total_leases) {
    if (persistent_active()) { // ★B1 — test snapshot도 command
        const int32_t ci = cq_alloc();
        if (ci < 0) { if (out_leased_slots) *out_leased_slots = 0; if (out_total_leases) *out_total_leases = 0; return; }
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        dfac_cmd_query_lease_stats_fill_locked(ci);
        cq_submit_and_wait(ci);
        if (out_leased_slots) *out_leased_slots = c.q_out_i32_a;
        if (out_total_leases) *out_total_leases = c.q_out_i64;
        cq_free(ci);
        return;
    }
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    int32_t leased = 0; int64_t total = 0;
    for (const auto & s : g_mod.slots) {
        int32_t l = s.lease.load(std::memory_order_relaxed);
        if (l > 0) { leased++; total += l; }
    }
    if (out_leased_slots) *out_leased_slots = leased;
    if (out_total_leases) *out_total_leases = total;
}

// R-B5a facade (iv): command request / result lifecycle (RB_DESIGN.md:1221 section
// 2-6 (iv)). The caller states the command it wants captured; `g_mod.cmds[ci]` is
// filled here and nowhere outside the dispatch cluster. The statements below are the
// ones this member replaced, in the same order, with the caller's lock unchanged.
inline void dfac_cmd_query_record_slot_fill_locked(int32_t ci, int64_t record_idx) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.kind = CMD_TEST_QUERY; c.qkind = QK_RECORD_SLOT; c.q_arg_i64 = record_idx;
}

// ★QD 레버 검증용 테스트 전용 내부 상태 조회(reviews/codex_qd_u6_design_xcheck.md 심의 3).
// ★프로덕션 경로 무접촉 — acquire/release/ptr/seal 어디서도 호출하지 않고, 읽기만 하며 모듈
// 상태를 변경하지 않는다. selftest가 record→slot 매핑·LRU(last_used)·generation 불변을 직접
// 관측하기 위한 창구다(ptr!=null 같은 간접 관측으로는 tick 순서 결함이 통과하기 때문).
void ggml_moe_direct_test_record_slot(int64_t record_idx, int32_t * out_slot, uint64_t * out_slot_generation) {
    if (persistent_active()) { // ★B1 — test snapshot도 command
        const int32_t ci = cq_alloc();
        if (ci < 0) { if (out_slot) *out_slot = -1; if (out_slot_generation) *out_slot_generation = 0; return; }
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        dfac_cmd_query_record_slot_fill_locked(ci, record_idx);
        cq_submit_and_wait(ci);
        if (out_slot)            *out_slot = c.q_out_i32_a;
        if (out_slot_generation) *out_slot_generation = c.q_out_u64_a;
        cq_free(ci);
        return;
    }
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    int32_t slot = -1; uint64_t gen = 0;
    if (g_mod.sealed && record_idx >= 0 && record_idx < (int64_t)g_mod.record_table.size()) {
        const RecordEntry & re = g_mod.record_table[(size_t)record_idx];
        slot = re.slot_index.load(std::memory_order_relaxed);
        gen  = re.slot_generation.load(std::memory_order_relaxed);
    }
    if (out_slot)             *out_slot = slot;
    if (out_slot_generation)  *out_slot_generation = gen;
}

// R-B5a facade (iv): command request / result lifecycle (RB_DESIGN.md:1221 section
// 2-6 (iv)). The caller states the command it wants captured; `g_mod.cmds[ci]` is
// filled here and nowhere outside the dispatch cluster. The statements below are the
// ones this member replaced, in the same order, with the caller's lock unchanged.
inline void dfac_cmd_query_slot_state_fill_locked(int32_t ci, int32_t slot_index) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.kind = CMD_TEST_QUERY; c.qkind = QK_SLOT_STATE; c.q_arg_i32 = slot_index;
}

void ggml_moe_direct_test_slot_state(int32_t slot_index, int64_t * out_owner_record,
                                      uint64_t * out_last_used, uint64_t * out_generation,
                                      int32_t * out_lease) {
    if (persistent_active()) { // ★B1 — test snapshot도 command
        const int32_t ci = cq_alloc();
        if (ci < 0) {
            if (out_owner_record) *out_owner_record = -1;
            if (out_last_used)    *out_last_used = 0;
            if (out_generation)   *out_generation = 0;
            if (out_lease)        *out_lease = 0;
            return;
        }
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        dfac_cmd_query_slot_state_fill_locked(ci, slot_index);
        cq_submit_and_wait(ci);
        if (out_owner_record) *out_owner_record = c.q_out_i64;
        if (out_last_used)    *out_last_used    = c.q_out_u64_a;
        if (out_generation)   *out_generation   = c.q_out_u64_b;
        if (out_lease)        *out_lease        = c.q_out_i32_a;
        cq_free(ci);
        return;
    }
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    int64_t owner = -1; uint64_t last_used = 0, gen = 0; int32_t lease = 0;
    if (g_mod.sealed && slot_index >= 0 && slot_index < g_mod.n_slots) {
        const Slot & s = g_mod.slots[(size_t)slot_index];
        owner     = s.owner_record;
        last_used = s.last_used;
        gen       = s.generation.load(std::memory_order_relaxed);
        lease     = s.lease.load(std::memory_order_relaxed);
    }
    if (out_owner_record) *out_owner_record = owner;
    if (out_last_used)    *out_last_used    = last_used;
    if (out_generation)   *out_generation   = gen;
    if (out_lease)        *out_lease        = lease;
}

// OPEN_ARCH B axis section 2-5 - the machine-readable probe. Assembled once, statically
// (no lock needed, read-only, never freed).
const char * ggml_moe_direct_openarch_probe_json(void) {
    static const std::string s = []() {
        json j;
        j["abi"] = OPEN_ARCH_TEMPLATE_ABI_STR;
        json arr = json::array();
        for (const auto & e : ARCH_TEMPLATE_CATALOG) {
            json kinds = json::array();
            for (const char * w : ARCH_TEMPLATE_WEIGHT_KINDS) kinds.push_back(w);
            arr.push_back({
                { "template_id",      e.template_id },
                { "template_version", e.template_version },
                { "weight_kinds",     kinds },
                { "bias",             e.bias == TplBias::REQUIRED ? "required" : "forbidden" },
                { "layer_rule",       e.layer_rule == TplLayerRule::LEADING_DENSE ? "leading-dense" : "all" },
                { "nextn",            e.nextn == TplNextn::EXECUTION_DEFAULT ? "execution-default" : "forbidden" },
            });
        }
        j["templates"] = arr;
        return j.dump();
    }();
    return s.c_str();
}

// Test only - read back the inventory digest the last template seal recomputed (the Gate C item 4
// observation window).
int ggml_moe_direct_test_template_inventory_sha256(char * out, int32_t cap) {
    if (!out || cap <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const std::string & s = g_mod.template_inventory_sha256;
    if ((int32_t)(s.size() + 1) > cap) { out[0] = '\0'; return -1; }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return (int32_t)s.size();
}

// R-B5a facade: the reset run (RB_DESIGN.md:1221 section 2-6, the dispatcher counters the
// seven design groups do not name, read as group (vi) per the lead's ruling of 2026-09-19).
// `ggml_moe_direct_reset_for_test` writes 86 dispatcher-owned fields directly; 33 of them now
// call members D2a and D2c already landed, and the remaining 53 are these seven composites.
// Each body is the reset's own run, in its own order, at its own indentation - what changed is
// only that an interleaved site inside a run is now the call that owns that field. None of them
// reads: the reset states, it does not ask.

// K8: the sticky shutdown-regression latch. The three READS of it in this same function stay
// exactly where they are; only the store moves. It is the one new member without `_locked`,
// because its site is above the reset's `std::lock_guard`
inline void dfac_disp_stop_timed_out_set(bool v) {
    g_mod.disp_stop_timed_out = v;
}

// K6: the two K6 inflight counters. It is separate from the run below it because the line
// between them writes the K10 io3 record vectors, which are not dispatcher-owned and must not
// travel into the dispatch TU
inline void dfac_reset_inflight_counters_locked() {
    g_mod.v3_inflight_demand = g_mod.v3_inflight_prefetch = 0;
}

// K6,K11: the copy and child counters, with the `v3_copy_pause` arming store kept at the head of
// the run as the call D2c landed for it
inline void dfac_reset_v3_counters_locked() {
    dfac_v3_copy_pause_set(0);
    g_mod.v3_copies_done = g_mod.v3_copies_failed = g_mod.v3_backpressure_stalls = 0;
    g_mod.v3_children_submitted = g_mod.v3_children_retired = 0;
    g_mod.v3_waves_submitted = 0;   // ★★R-1
    g_mod.v3_op_terminals = g_mod.v3_publishes = 0;
    g_mod.v3_children_sync = g_mod.v3_children_async = 0;
}

// K6,K9: the coalesce counters and the copy credit counter, with the interleaved
// `v3_copy_outstanding` store kept at its position as the setter D2a landed
inline void dfac_reset_coal_counters_locked() {
    g_mod.v3_coal_group_seq = 0;
    g_mod.v3_coal_groups = g_mod.v3_coal_records = g_mod.v3_coal_scatters = 0;
    g_mod.v3_coal_reads_retired = 0;
    dfac_copy_outstanding_set_locked(0);
    g_mod.v3_copy_credit_stalls = 0;
    g_mod.v3_coal_last_reject = V3_COAL_R_NONE;
    g_mod.v3_coal_reject_mask = 0;
}

// K11: the retire-order witness vector
inline void dfac_retire_order_clear_locked() {
    g_mod.v3_retire_order.clear();
}

// K12: the command and queue statistics, the two cancel counters and the queue sequence, with
// the seven arming flags interleaved in the run kept at their positions as the setters D2c
// landed
inline void dfac_reset_stats_locked() {
    g_mod.stat_demand_cmd = g_mod.stat_release_cmd = g_mod.stat_query_cmd = g_mod.stat_stop_cmd = 0;
    g_mod.stat_join = g_mod.stat_cancel = g_mod.stat_abort_terminal = 0;
    g_mod.stat_q_full_waits.store(0, std::memory_order_relaxed);
    g_mod.stat_q_max_depth.store(0, std::memory_order_relaxed);
    g_mod.stat_wait_wakeups.store(0, std::memory_order_relaxed);
    dfac_test_abort_after_submit_set_locked(0);
    dfac_test_force_sync_next_set_locked(0);
    dfac_test_pb_error_next_set_locked(0);
    dfac_test_hold_retire_set_locked(0);
    dfac_test_hold_submit_set_locked(0);
    g_mod.v3_cancel_selected = 0;
    g_mod.v3_cancel_excl_joined = 0;
    dfac_test_stop_race_set_locked(0);
    dfac_test_cut_residual_force_set_locked(0);
    g_mod.stat_sync_terminal = 0;
    g_mod.q_seq_next = 1;
}

// K7: the PhaseB virtual record gauges, the reservation totals and the admit witnesses. The T-1
// comment inside the run travels with it: it explains why `disp_stop_timed_out` is NOT cleared
// among these stores, which is a statement about this run and stays attached to it
inline void dfac_reset_pbv_admit_locked() {
    g_mod.pbv_rec.clear();
    g_mod.pbv_rec_active = g_mod.pbv_rec_inflight = g_mod.pbv_child_reserved = 0;
    g_mod.pbv_credit_reserved_total = g_mod.pbv_credit_unreserved_total = 0;   // ★S-1(reset 에서만)
    g_mod.pbv_cont_defer_events = 0;   // ★S-4(reset 에서만)
    g_mod.pbv_admit_skipped_cont = 0;  // ★T-3(reset 에서만)
    g_mod.pbv_admit_skip_rec_active = g_mod.pbv_admit_skip_intent_depth = -1;  // ★U-3 증인 sentinel
    g_mod.pbv_admit_skip_valid_left = g_mod.pbv_admit_skip_ncap_room = -1;
    g_mod.pbv_admit_skip_credit_ok  = -1;
    g_mod.pbv_admit_skip_admitted   = 0;
    dfac_test_cont_credit_fail_next_set_locked(0);
    // ★★T-1 — `disp_stop_timed_out` 은 **여기서 지우지 않는다**(프로세스 sticky).
    //   timeout 은 곧 shutdown 회귀 신호이므로, 지웠다가는 다음 그룹에서 다시 8s 씩 매달려
    //   런이 끝나지 않는다. 한 번 서면 그 런은 붉은 채로 **끝까지 진행**하는 것이 옳다.
    g_mod.pb_admit_deferred_credit_attempts = 0;
    g_mod.pb_admit_skipped_drain_turns      = 0;
    g_mod.pbv_admit_parked = false;
}

void ggml_moe_direct_reset_for_test(void) {
    // ★B1 종료 시퀀스(§2) — graceful stop/join을 **락 밖에서** 먼저(dispatcher가 g_mod.mutex를
    //   필요로 하므로 락을 쥔 채 join하면 교착). 그 뒤에야 file/pool을 해제할 수 있다.
    ggml_moe_phaseb_cut_close_capture();
    // ★★T-1(교차 r3 S-2 ② — 실측으로 확인한 두 번째 길): `reset_for_test` 는 **테스트 전용
    //   경로**이므로 여기 join 이 INFINITE 면 shutdown 회귀가 "관문 FAIL" 이 아니라 **하네스
    //   교착**으로 나타난다. 실제로 graceful 종료를 막는 역변이(M-R)를 넣으니 V3P17 에 닿기도
    //   전에 **첫 그룹의 reset 에서 매달렸다**. ⇒ 두 정지 호출을 유계화한다.
    //   ★프로덕션은 무영향: `ggml_moe_direct_finalize()` 의 정지 경로는 그대로다.
    //   E-2b-0 durable (g): the sentence corrected above ended with a claim that finalize's
    //   stop path is still UNBOUNDED. E-2a's G-1 change made that false - finalize reaches
    //   dispatcher_stop_and_join, which now passes the finite MOE_DRAIN_DEADLINE_MS. What
    //   the sentence is FOR - "this 8000 ms bounding does not change the production stop
    //   policy" - is unaffected and still true.
    //   ★래치는 **프로세스 sticky** 다 — 한 번 timeout 하면 그건 회귀 신호이므로, 이후 reset 은
    //     즉시 건너뛴다(안 그러면 그룹마다 timeout 만큼 매달려 런이 끝나지 않는다).
    if (!g_mod.disp_stop_timed_out) {
        if (!ggml_moe_phaseb_cut_drain_predictor_bounded(/*timeout_ms=*/8000)) {
            dfac_disp_stop_timed_out_set(true);
            log_err("[moe-direct] reset_for_test: predictor drain timed out - shutdown regression "
                    "(subsequent resets are skipped so gates fail instead of hanging)");
        }
    }
    if (!g_mod.disp_stop_timed_out) {
        if (!dispatcher_stop_and_join_bounded(/*abort_mode=*/false, /*timeout_ms=*/8000))
            log_err("[moe-direct] reset_for_test: dispatcher stop timed out - shutdown regression "
                    "(subsequent resets are skipped so gates fail instead of hanging)");
    }
    // ★★U-1(교차 r4 ②) — 래치가 섰으면 **후속 정리를 여기서 즉시 단락**한다.
    //   구 판은 timeout 뒤에도 copy-worker join · PhaseB reset · 자원 해제를 계속했는데,
    //   그 시점 dispatcher 스레드는 **살아서 같은 상태를 쓰고 있다** ⇒ UAF 창이 열린다
    //   (`run_group` 종단 가드는 `fn()` 반환 **후**라 차단점이 아니다).
    //   "timeout 시 cleanup 생략" 원칙을 **종단까지 관철**한다 — 위험한 정리를 하느니
    //   아무것도 안 하고 가드로 직행하는 편이 안전하다.
    if (g_mod.disp_stop_timed_out) {
        log_err("[moe-direct] reset_for_test: stop timed out - skipping all subsequent teardown "
                "(a live dispatcher still uses this state; the run is already failing)");
        return;
    }
    // ★A4 — 바운스 copy worker 도 **락 밖에서** 정지·join 한다(worker 가 g_mod.mutex 를 잡으므로
    //   쥔 채 join 하면 교착). dispatcher 를 먼저 세워 신규 job 생산을 끊은 뒤 부른다.
    v3_copy_stop_and_join();
    ggml_moe_phaseb_reset_for_test();        // PB 전역 상태·env 재해소(selftest 시나리오 격리)
    // ★lead-time 도 같은 이유로 시나리오 경계에서 되돌린다 — 선할당 버퍼가 시나리오마다 쌓이면
    //   WS 가 자라고(메모리 포화 처방), 앞 시나리오의 행·complete 래치가 다음 시나리오로 샌다.
    ggml_moe_leadtime_reset_for_test();
    // ★GPU-early 도 같은 이유로 시나리오 경계에서 되돌린다(선할당 버퍼가 시나리오마다 쌓이면 WS 가
    //   자라고, 앞 시나리오의 행·카운터·pending 이 다음 시나리오로 샌다).
    ggml_moe_gpuearly_reset_for_test();

    std::lock_guard<std::mutex> lock(g_mod.mutex);
    // ★★V3HEAP — 보호 원복을 **VEH 제거보다 먼저** 한다. 순서가 계약이다: VEH 를 먼저 내리면
    //   보호된 페이지를 서비스할 핸들러가 없는 창이 생긴다. 구 판은 여기서 아무것도 안 하고
    //   아래에서 `prot_intervals.clear()` 로 기록만 버려서, 호출자 소유 텐서 페이지가
    //   PAGE_NOACCESS 인 채 남았다 — 그 텐서가 힙 메모리(V3PROD 의 gguf ggml_context 아레나 =
    //   `ggml_aligned_malloc` → `_aligned_malloc`)면 구멍 뚫린 블록이 그대로 할당자에 돌아간다.
    //   같은 원인의 얼굴 2개가 덤프로 실측됐다 — 어느 쪽이 뜨는지는 구멍이 힙의 어느 구조와
    //   겹치는가에 달렸다: `0xC0000005` AV(누수 영역 base 를 그대로 때린 WRITE) /
    //   `0xC0000374` 힙 손상(실패 힙 주소 == 누수 영역 AllocationBase).
    //   덤프 원장=`bench/techdev/repack_space/V3_HEAP_FIX_R1.md` §2-3·§5-R.
    restore_prot_intervals_locked();
    if (g_mod.veh_handle) { RemoveVectoredExceptionHandler(g_mod.veh_handle); g_mod.veh_handle = nullptr; }
    if (g_mod.pool_base) { VirtualFree(g_mod.pool_base, 0, MEM_RELEASE); g_mod.pool_base = nullptr; }
    // ★★S6(교차 r8 Z6-2) — A4 바운스 staging 풀도 여기서 반납한다. 구 판은 coalesce 풀만
    //   놓아서 시나리오마다 바운스 풀이 누적됐다(정상 teardown `virtual_release_locked` 에는
    //   똑같은 반납 코드가 이미 있다 — 빠진 쪽을 대칭으로 채운다). 테스트 전용 경로라
    //   프로덕션 무영향이지만, 순차 시나리오마다 쌓이므로 완주 관문 전에 닫는다.
    dfac_staging_pool_release_locked();
    // ★A6 — coalesce 버퍼 풀은 **시나리오 경계에서** 반납한다. 안 하면 다음 시나리오가 합침을
    //   끈 채로도 `v3_coal_count` 를 물려받아 "칸이 있다"고 보고한다(관측이 거짓이 되고, 계획이
    //   해제된 풀을 가리킬 수도 있다). 실측으로 잡힌 항목이다.
    dfac_coal_pool_release_locked();
    if (g_mod.data_file != INVALID_HANDLE_VALUE) { CloseHandle(g_mod.data_file); g_mod.data_file = INVALID_HANDLE_VALUE; }
    // ★§Z-⑥ A2 — mode=virtual 소스 핸들 집합. 경계 teardown 이 이미 비웠으면 no-op 이고,
    //   경계를 앞으로 민 뒤(A3+)엔 여기가 유일한 반납 지점이 된다.
    for (HANDLE h : g_mod.source_files) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    g_mod.source_files.clear();
    g_mod.manifest_mode = ManifestMode::BIN;
    g_mod.metrics.reset();

    g_v3_stage.store(V3_STAGE_NONE, std::memory_order_relaxed);   // ★§Z-⑥ 관문 래치도 시나리오마다 초기화
    g_v3_probe = V3ResourceProbe{};                               // ★A2 자원 스냅샷도 시나리오 격리
    // ★A3 — child 훅·계수도 시나리오마다 초기화(one-shot 이 다음 시나리오로 새면 안 된다).
    dfac_v3_test_reverse_children_set_locked(0);
    dfac_v3_test_fail_child_set_locked(-1);
    dfac_test_fail_after_acquire_set_locked(0);
    dfac_v3_test_sync_child_set_locked(-1);
    dfac_v3_test_fail_copy_set_locked(-1);
    g_mod.io3_on = false; g_mod.io3_requested = false; g_mod.io3_complete = true; g_mod.io3_out.clear();
    for (int i = 0; i < 3; i++) {
        g_mod.io3_stat_records[i] = 0; g_mod.io3_stat_written[i] = 0; g_mod.io3_stat_digest[i] = 0;
    }
    g_mod.io3_stat_buffer_bytes = 0;
    g_mod.manifest_schema_version.clear();
    g_mod.io3_event_seq = g_mod.io3_record_seq = g_mod.io3_publish_seq = g_mod.io3_dropped = 0;
    g_mod.io3_boot_id = g_mod.io3_t0_qpc = 0;
    dfac_reset_inflight_counters_locked();
    g_mod.io3_records.clear(); g_mod.io3_children.clear(); g_mod.io3_copies.clear();
    dfac_reset_v3_counters_locked();
    // ★R12 — §2 재결속 계수·§4 mode-local digest 축·§3-1 해소 파라미터도 시나리오마다 초기화한다
    //   (누수되면 다음 시나리오의 metrics 축이 앞 시나리오를 포함한다).
    g_mod.v3_child_pending_demand = g_mod.v3_child_immediate_demand = 0;
    g_mod.v3_readfile_ops = 0;
    g_mod.v3_readfile_digest = 0xcbf29ce484222325ull;
    for (int i = 0; i < 6; i++) { dfac_io_param_val_set_locked((size_t)i, 0); dfac_io_param_src_set_locked((size_t)i, 0); }
    // ★A6 — coalescing 파라미터·계수·훅도 시나리오마다 초기화한다. ★주입 훅은 **sticky**
    //   이라 여기서 내리지 않으면 다음 시나리오의 합침 판정이 앞 시나리오의 주입을
    //   그대로 물려받는다.
    dfac_coal_phase_set_locked(V3_COAL_OFF);
    dfac_coal_max_bytes_set_locked(0);
    dfac_coal_min_bytes_set_locked(0);
    dfac_coal_buffers_set_locked(0);
    dfac_reset_coal_counters_locked();
    dfac_v3_test_coal_overrun_set_locked(0);
    dfac_v3_test_coal_unpin_set_locked(0);
    g_mod.io3_op_row.clear(); g_mod.io3_chan_row.clear(); g_mod.io3_chan_copy_row.clear();
    dfac_retire_order_clear_locked();
    xstore_reset_locked();   // E-2a: epoch, adapter map and handle table
    g_mod.sealed = false;
    g_mod.pending.clear();
    g_mod.registry.clear();
    g_mod.arch.clear();
    g_mod.n_layer = g_mod.n_expert = g_mod.n_expert_used = 0;
    g_mod.moe_layers.clear();
    g_mod.layer_to_mlpos.clear();
    g_mod.layers.clear();
    g_mod.sources.clear();
    g_mod.align_bytes = g_mod.slot_stride_max = g_mod.bin_file_bytes = 0;
    g_mod.ref_profile_id.clear();
    g_mod.ref_expect_sha256.clear();
    g_mod.repack_manifest_sha256.clear();
    g_mod.template_inventory_sha256.clear();   // OPEN_ARCH B axis
    g_mod.routed_scope = "all";
    g_mod.att_all = g_mod.att_host = g_mod.att_nonhost = g_mod.att_n_slots = g_mod.att_n_slots_required = 0;
    g_mod.att_raw = g_mod.att_managed = g_mod.att_excluded_nextn = 0;
    g_mod.pool_bytes = 0;
    g_mod.n_slots = 0;
    g_mod.slots.clear();
    g_mod.record_table.clear();
    g_mod.tick = 0;
    g_mod.qd_effective = MOE_DIRECT_QD_DEFAULT; // ★QD 레버 — seal마다 env 재해소(테스트가 QD를 바꿔 재seal)
    g_mod.qd_source = "default";
    // ★B1 selector/dispatcher 상태도 seal마다 재해소(reset drain 관문).
    g_mod.io_path_persistent = true;
    g_mod.io_path_source     = "default";
    g_mod.io_path_qualified  = false;
    dfac_reset_stats_locked();
    // ★B2 dispatcher 결속 상태(seal마다 재해소).
    g_mod.pb_prefetch_on = false;
    g_mod.pb_k = g_mod.pb_n = 0;
    g_mod.pb_inflight_demand = g_mod.pb_inflight_prefetch = 0;
    // ★★PB virtual A1/A2 — ①record 운영축 gauge·예약분·진단 2종·park 래치도 여기서 0 으로.
    //   테스트는 같은 프로세스에서 seal 을 반복하므로 하나라도 남으면 다음 런의 credit 식이
    //   앞 런의 잔여를 물려받는다(고정 인덱스 재사용 누수와 같은 부류).
    dfac_reset_pbv_admit_locked();
    g_mod.pb_io_event_seq = 0;
    g_mod.pb_publish_seq  = 0;
    g_mod.pb_cur_graph_seq = GGML_MOE_PHASEB_NA_U32;
    g_mod.pb_cur_active = false;
    g_mod.pb_cur_rank = 0;
    g_mod.pb_cur_row = -1;
    g_mod.pb_served_graph.clear();
    g_mod.pb_samples.clear();
    g_mod.pb_sample_index.clear();
    g_mod.pb_ios.clear();
    g_mod.pb_finalized = false;
    g_mod.pb_cut_residual = 0;
    g_mod.pb_pending_status.clear();
    // ★5c-T 수집 축 전건 재초기화(seal 마다 재해소 — 시나리오 격리).
    g_mod.pb_collect = false;
    g_mod.pb_censused = false;
    g_mod.pb_op_stamp.clear();
    g_mod.pb_pred_snap.clear();
    g_mod.pb_pred_snap_next = 0;
    for (int32_t i = 0; i < MOE_DIRECT_CONTROL_Q_CAP; i++) {
        g_mod.pb_cmd_waited[i] = 0;
        g_mod.pb_cmd_frontier[i] = 0;
    }
    g_mod.pb_ledger_records = 0;
    g_mod.pb_ledger_digest  = 0;
    g_mod.pb_ledger_written = 0;
    // ★Phase A — 훅 게이트/V1 digest도 seal마다 재해소(모듈 상태와 함께 초기화).
    g_mod.phasea_on = false;
    g_mod.readfile_digest = 0xcbf29ce484222325ull;
    g_mod.readfile_ops = 0;
    g_mod.io_submit_ops = 0;
    g_mod.graph_nodes_digest  = 0xcbf29ce484222325ull;
    g_mod.graph_splits_digest = 0xcbf29ce484222325ull;
    g_mod.graph_count = 0;
    g_mod.pa_submit_seq_next = 0;
    g_mod.pa_retire_seq_next = 0;
    g_mod.reg_tensor_info.clear();
    // ★★V3HEAP — `prot_intervals` 는 이 함수 머리의 `restore_prot_intervals_locked()` 가
    //   **원복한 뒤** 비운다. 여기서 다시 비우면(구 판) 그게 곧 원복 없는 버림이다.
    g_mod.page_touched.clear();
    g_mod.touch_ring.clear();
    g_mod.touch_ring_next.store(0, std::memory_order_relaxed);
    g_mod.touch_abort_threshold = PROD_DEFAULT_TOUCH_ABORT;
    g_mod.touch_events.store(0, std::memory_order_relaxed);
    g_mod.touch_pages.store(0, std::memory_order_relaxed);
    g_mod.fallback_count.store(0, std::memory_order_relaxed);
    g_mod.read_immediate_count.store(0, std::memory_order_relaxed);
    g_mod.read_pending_count.store(0, std::memory_order_relaxed);
    for (int i = 0; i < 2; i++) {
        for (PhaseCounters * pc : { &g_mod.phase[i], &g_mod.phase_exact[i], &g_mod.phase_exact_window[i] }) {
            pc->read_count.store(0, std::memory_order_relaxed);
            pc->read_bytes.store(0, std::memory_order_relaxed);
            pc->hit.store(0, std::memory_order_relaxed);
            pc->miss.store(0, std::memory_order_relaxed);
            pc->ensure_wait_ms_sum.store(0, std::memory_order_relaxed);
        }
    }
    g_mod.phase_ctx_unknown_count.store(0, std::memory_order_relaxed);
}

void ggml_moe_direct_test_read_stats(uint64_t * immediate_count, uint64_t * pending_count) {
    if (immediate_count) *immediate_count = g_mod.read_immediate_count.load(std::memory_order_relaxed);
    if (pending_count) *pending_count = g_mod.read_pending_count.load(std::memory_order_relaxed);
}

int ggml_moe_direct_test_mapped_view_roundtrip(void * base, size_t nbytes) {
    if (!base || nbytes == 0) return -1;
    if (!g_mod.veh_handle) return -1;

    uintptr_t byte_start = (uintptr_t)base;
    uintptr_t byte_end = byte_start + nbytes;
    uintptr_t astart = (uintptr_t)align_up(byte_start, 4096);
    uintptr_t aend = (uintptr_t)align_down(byte_end, 4096);
    if (aend <= astart) return -1;

    int32_t tidx;
    ProtInterval iv{};
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        RegisteredTensorInfo info{};
        info.layer = -1; info.part = -1; info.addr_start = byte_start; info.nbytes = nbytes;
        tidx = (int32_t)g_mod.reg_tensor_info.size();
        g_mod.reg_tensor_info.push_back(info);

        uint32_t page_base = 0;
        for (auto & e : g_mod.prot_intervals) page_base += (uint32_t)((e.end - e.start) / 4096);
        iv.start = astart; iv.end = aend; iv.tensor_idx = tidx; iv.page_base_index = page_base;

        uint32_t add_pages = (uint32_t)((aend - astart) / 4096);
        size_t old_size = g_mod.page_touched.size();
        std::vector<std::atomic<uint8_t>> grown(old_size + add_pages);
        for (size_t i = 0; i < old_size; i++) grown[i].store(g_mod.page_touched[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        for (size_t i = old_size; i < grown.size(); i++) grown[i].store(0, std::memory_order_relaxed);
        g_mod.page_touched = std::move(grown);

        g_mod.prot_intervals.push_back(iv);
        std::sort(g_mod.prot_intervals.begin(), g_mod.prot_intervals.end(),
                  [](const ProtInterval & a, const ProtInterval & b) { return a.start < b.start; });
    }

    DWORD old_protect = 0;
    if (!VirtualProtect((LPVOID)astart, aend - astart, PAGE_NOACCESS, &old_protect)) return -1;

    uint64_t events_before = g_mod.touch_events.load(std::memory_order_relaxed);
    uint64_t pages_before = g_mod.touch_pages.load(std::memory_order_relaxed);

    volatile unsigned char probe = *(volatile unsigned char *)astart;
    (void)probe;

    uint64_t events_after = g_mod.touch_events.load(std::memory_order_relaxed);
    uint64_t pages_after = g_mod.touch_pages.load(std::memory_order_relaxed);
    if (events_after != events_before + 1) return -1;
    if (pages_after != pages_before + 1) return -1;

    uint64_t last_seq = g_mod.touch_ring_next.load(std::memory_order_acquire);
    if (last_seq == 0) return -1;
    size_t ring_slot = (size_t)((last_seq - 1) % g_mod.touch_ring.size());
    const TouchLogEntry & entry = g_mod.touch_ring[ring_slot];
    if (entry.seq.load(std::memory_order_acquire) != last_seq) return -1;
    if (entry.addr != astart) return -1;
    if (entry.tensor_idx != tidx) return -1;

    volatile unsigned char probe2 = *(volatile unsigned char *)astart;
    (void)probe2;
    return 0;
}

// section 13.1 of the core TU (its banner stays there): ★B1 영속 dispatcher 테스트/조회 진입점
// (PHASEB_DESIGN §7 착지 관문 계측 창구)

void ggml_moe_direct_test_dispatcher_stats(uint64_t * out_demand_cmd, uint64_t * out_release_cmd,
                                            uint64_t * out_query_cmd, uint64_t * out_join_count,
                                            uint64_t * out_cancel_count, uint64_t * out_abort_terminal,
                                            uint64_t * out_q_full_waits, uint64_t * out_q_max_depth,
                                            int32_t * out_io_path_persistent) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_demand_cmd)     *out_demand_cmd     = g_mod.stat_demand_cmd;
    if (out_release_cmd)    *out_release_cmd    = g_mod.stat_release_cmd;
    if (out_query_cmd)      *out_query_cmd      = g_mod.stat_query_cmd;
    if (out_join_count)     *out_join_count     = g_mod.stat_join;
    if (out_cancel_count)   *out_cancel_count   = g_mod.stat_cancel;
    if (out_abort_terminal) *out_abort_terminal = g_mod.stat_abort_terminal;
    if (out_q_full_waits)   *out_q_full_waits   = g_mod.stat_q_full_waits.load(std::memory_order_relaxed);
    if (out_q_max_depth)    *out_q_max_depth    = g_mod.stat_q_max_depth.load(std::memory_order_relaxed);
    if (out_io_path_persistent) *out_io_path_persistent = g_mod.io_path_persistent ? 1 : 0;
}

void ggml_moe_direct_test_mem_echo(uint64_t * out_control_q_bytes, uint64_t * out_pending_bytes,
                                    uint64_t * out_channel_bytes, uint64_t * out_dispatcher_stack_bytes,
                                    uint64_t * out_capture_bytes, uint64_t * out_intent_bytes,
                                    uint64_t * out_predictor_bytes, uint64_t * out_snapshot_bytes) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const bool p = g_mod.io_path_persistent && g_mod.disp_running.load(std::memory_order_acquire);
    const uint64_t n_rec = (uint64_t)g_mod.moe_layers.size() * (uint64_t)g_mod.n_expert;
    if (out_control_q_bytes) *out_control_q_bytes = p ? control_q_bytes_locked() : 0;
    if (out_pending_bytes) *out_pending_bytes = p
        ? (uint64_t)g_mod.n_slots * (uint64_t)sizeof(PendingIo) + n_rec * sizeof(int32_t) : 0;
    if (out_channel_bytes) *out_channel_bytes = p
        ? (uint64_t)g_mod.qd_effective * (uint64_t)sizeof(IoChannel) : 0;
    if (out_dispatcher_stack_bytes) *out_dispatcher_stack_bytes = p ? (uint64_t)MOE_DIRECT_DISP_STACK : 0;
    // ★B2 자원 — 프리페치 OFF 형상에서는 전부 0(§7.2 게이트 ⓐ "OFF면 B2 확장 블록 전부 0").
    uint64_t cap_slot = 0, cap_pool = 0, rt_b = 0, nm_b = 0, irec = 0, iq_b = 0, pstk = 0, co = 0, io_ = 0;
    uint32_t pth = 0;
    ggml_moe_phaseb_mem_echo(&cap_slot, &cap_pool, &rt_b, &nm_b, &irec, &iq_b, &pth, &pstk, &co, &io_);
    if (out_capture_bytes)   *out_capture_bytes   = cap_pool;
    if (out_intent_bytes)    *out_intent_bytes    = iq_b;
    if (out_predictor_bytes) *out_predictor_bytes = pstk;
    if (out_snapshot_bytes)  *out_snapshot_bytes  = rt_b + nm_b;
}

// ★B2 관측 창구(테스트 전용 · 읽기만) — pb_sample/pb_io 레코드와 5종 digest·C3 카운터를
//   selftest가 직접 재계산해 대조하기 위한 통로다.
void ggml_moe_direct_test_pb_counts(uint64_t * out_n_samples, uint64_t * out_n_ios,
                                     uint64_t * out_publish_seq, uint64_t * out_event_seq,
                                     int32_t * out_prefetch_on, int32_t * out_k, int32_t * out_n) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_n_samples)   *out_n_samples   = (uint64_t)g_mod.pb_samples.size();
    if (out_n_ios)       *out_n_ios       = (uint64_t)g_mod.pb_ios.size();
    if (out_publish_seq) *out_publish_seq = g_mod.pb_publish_seq;
    if (out_event_seq)   *out_event_seq   = g_mod.pb_io_event_seq;
    if (out_prefetch_on) *out_prefetch_on = g_mod.pb_prefetch_on ? 1 : 0;
    if (out_k)           *out_k           = g_mod.pb_k;
    if (out_n)           *out_n           = g_mod.pb_n;
}

int ggml_moe_direct_test_pb_sample(uint64_t idx, void * out_record, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!out_record || bytes != sizeof(ggml_moe_phaseb_sample)) return -1;
    if (idx >= g_mod.pb_samples.size()) return -1;
    std::memcpy(out_record, &g_mod.pb_samples[(size_t)idx], sizeof(ggml_moe_phaseb_sample));
    return 0;
}

int ggml_moe_direct_test_pb_io(uint64_t idx, void * out_record, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!out_record || bytes != sizeof(ggml_moe_phaseb_io)) return -1;
    if (idx >= g_mod.pb_ios.size()) return -1;
    std::memcpy(out_record, &g_mod.pb_ios[(size_t)idx], sizeof(ggml_moe_phaseb_io));
    return 0;
}

void ggml_moe_direct_test_pb_fault_next(int32_t on) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_test_pb_error_next_set_locked(on);
}

int32_t ggml_moe_direct_test_cut_residual(void) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    return g_mod.pb_cut_residual;
}

// 테스트 전용 — CUT 6 잔여 세부 비트 주입(0=해제). finalize 전에 설정한다.
void ggml_moe_direct_test_force_cut_residual(int32_t bits) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_test_cut_residual_force_set_locked(bits);
}

// 테스트 전용 — 상태 큐 -> 보류함 드레인 1회(dispatcher pass 없이 결정론적으로 행사).
//   반환 = 드레인 후 보류함 크기(= 보류 잔여 관측 창구).
int32_t ggml_moe_direct_test_drain_status(void) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    pb_drain_status_locked();
    return (int32_t)g_mod.pb_pending_status.size();
}

// pending 풀 인덱스 재사용 강제(R1 #1 회귀 창구) — 자유 스택 top을 1건 소모·즉시 반납해
// 다음 alloc이 **같은 인덱스**를 돌려주도록 만든다. 반환 = 그 인덱스.
int32_t ggml_moe_direct_test_pending_reuse_probe(void) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (g_mod.pending_free.empty()) return -1;
    return g_mod.pending_free.back();
}

// ★보완 ② — dispatcher 를 "stopping 스냅샷 직후" 창에 1회 정지시키고 도달을 확인한다.
//   0=창 도달 / <0=미기동·타임아웃. 도달 후 반드시 release 를 불러야 dispatcher 가 재개된다.
int ggml_moe_direct_test_stop_race_arm(int32_t timeout_ms) {
    HANDLE hit = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        if (!g_mod.disp_running.load(std::memory_order_acquire)) return -1;
        if (!g_mod.stop_race_hit || !g_mod.stop_race_go) return -1;
        ResetEvent(g_mod.stop_race_go);
        ResetEvent(g_mod.stop_race_hit);
        dfac_test_stop_race_set_locked(1);
        hit = g_mod.stop_race_hit;
    }
    SetEvent(g_mod.command_event);   // 대기 중이면 깨워 pass 로 진입시킨다
    return WaitForSingleObject(hit, (DWORD)(timeout_ms > 0 ? timeout_ms : 1000)) == WAIT_OBJECT_0 ? 0 : -1;
}

void ggml_moe_direct_test_stop_race_release(void) {
    if (g_mod.stop_race_go) SetEvent(g_mod.stop_race_go);
}

// ★★S-1 — credit 예약/반납 **누적** 직독(테스트 전용). teardown 불변 축.
void ggml_moe_direct_test_pbv_credit_totals(uint64_t * out_reserved, uint64_t * out_unreserved) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_reserved)   *out_reserved   = g_mod.pbv_credit_reserved_total;
    if (out_unreserved) *out_unreserved = g_mod.pbv_credit_unreserved_total;
}

// ★★S-4 — continuation credit 재검사 **실패 사건** 누적(= drain-before-refill 실발동 증거).
void ggml_moe_direct_test_pbv_cont_defer_events(uint64_t * out_events) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_events) *out_events = g_mod.pbv_cont_defer_events;
}

// ★★T-3 — **결과 계약** 축: continuation 실패 때문에 그 turn 신규 admission 을 생략한 횟수.
void ggml_moe_direct_test_pbv_admit_skipped_cont(uint64_t * out_events) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_events) *out_events = g_mod.pbv_admit_skipped_cont;
}

// ★★U-3(교차 r4 · 대기 고정) + ★★V(교차 r5 · **실 후보 증명**) — 생략 그 순간의 증인 직독
//   (사유=선언부 주석). `-1` = 그 지점에 한 번도 도달하지 않았다(= 역변이 상태).
void ggml_moe_direct_test_pbv_admit_skip_witness(int32_t * out_rec_active,
                                                 int32_t * out_intent_depth,
                                                 int32_t * out_valid_left,
                                                 int32_t * out_ncap_room,
                                                 int32_t * out_credit_ok) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_rec_active)   *out_rec_active   = g_mod.pbv_admit_skip_rec_active;
    if (out_intent_depth) *out_intent_depth = g_mod.pbv_admit_skip_intent_depth;
    if (out_valid_left)   *out_valid_left   = g_mod.pbv_admit_skip_valid_left;
    if (out_ncap_room)    *out_ncap_room    = g_mod.pbv_admit_skip_ncap_room;
    if (out_credit_ok)    *out_credit_ok    = g_mod.pbv_admit_skip_credit_ok;
}

// ★★V(교차 r5) — 생략 **그 순간**의 C3 `admitted` 기준선. 이후 실제 admission 이 이 값을 넘어야
//   "막힌 것이 실 후보였다"가 거동으로 증명된다(사유=선언부 주석).
void ggml_moe_direct_test_pbv_admit_skip_admitted(uint64_t * out_admitted) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_admitted) *out_admitted = g_mod.pbv_admit_skip_admitted;
}

// ★★T-1 — 유계 종료가 timeout 으로 끝난 사실(관문이 진단에 쓴다).
int ggml_moe_direct_test_stop_timed_out(void) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    return g_mod.disp_stop_timed_out ? 1 : 0;
}

// ★★S-4 — continuation credit 재검사 1회성 실패 주입(사유=선언부 주석).
void ggml_moe_direct_test_cont_credit_fail_next(int32_t on) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_test_cont_credit_fail_next_set_locked(on ? 1 : 0);
}

// ★★R-5 V3P17 — 취소 선별 결과 직독(테스트 전용).
void ggml_moe_direct_test_v3_cancel_stats(uint64_t * out_selected, uint64_t * out_excluded_joined) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_selected)        *out_selected        = g_mod.v3_cancel_selected;
    if (out_excluded_joined) *out_excluded_joined = g_mod.v3_cancel_excl_joined;
}

void ggml_moe_direct_test_hold_submit(int32_t on) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_test_hold_submit_set_locked(on);
}

void ggml_moe_direct_test_hold_retire(int32_t on) {
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        dfac_test_hold_retire_set_locked(on);
    }
    if (!on && g_mod.command_event) SetEvent(g_mod.command_event); // 보류 해제 시 dispatcher 기상
}

// prefetch in-flight 0 + intent_q 소진까지 대기(결정론 확보). 1=quiesced / 0=timeout.
int ggml_moe_direct_test_pb_quiesce(int32_t timeout_ms) {
    for (int32_t i = 0; i < (timeout_ms > 0 ? timeout_ms : 1); i++) {
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            if (g_mod.pb_inflight_prefetch == 0 && !g_mod.pb_cur_active &&
                ggml_moe_phaseb_intent_depth() == 0 &&
                g_mod.submit_head >= g_mod.submit_q.size()) return 1;
        }
        Sleep(1);
    }
    return 0;
}

void ggml_moe_direct_test_slot_class(int32_t slot_index, int32_t * out_class,
                                     uint64_t * out_intent_ordinal, int32_t * out_rank) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    int32_t cls = 0, rk = (int32_t)GGML_MOE_PHASEB_NA_U16; uint64_t ord = GGML_MOE_PHASEB_NA_U64;
    if (g_mod.sealed && slot_index >= 0 && slot_index < g_mod.n_slots) {
        cls = g_mod.slots[(size_t)slot_index].cache_class;
        ord = g_mod.slots[(size_t)slot_index].pf_intent_ordinal;
        rk  = (int32_t)g_mod.slots[(size_t)slot_index].pf_rank;
    }
    if (out_class)           *out_class           = cls;
    if (out_intent_ordinal)  *out_intent_ordinal  = ord;
    if (out_rank)            *out_rank            = rk;   // ★R1 #7 probation 비교 키 rank 성분
}

void ggml_moe_direct_test_digests(uint64_t * out_readfile_digest, uint64_t * out_readfile_ops,
                                   uint64_t * out_lru_tick_digest, uint64_t * out_tick_final) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_readfile_digest) *out_readfile_digest = g_mod.readfile_digest;
    if (out_readfile_ops)    *out_readfile_ops    = g_mod.readfile_ops;
    if (out_tick_final)      *out_tick_final      = g_mod.tick;
    if (out_lru_tick_digest) {
        uint64_t d = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < g_mod.record_table.size(); i++) {
            const uint32_t idx = (uint32_t)i;
            const int32_t  sl  = g_mod.record_table[i].slot_index.load(std::memory_order_relaxed);
            const uint64_t gen = g_mod.record_table[i].slot_generation.load(std::memory_order_relaxed);
            d = fnv1a64_bytes(d, &idx, sizeof(idx));
            d = fnv1a64_bytes(d, &sl,  sizeof(sl));
            d = fnv1a64_bytes(d, &gen, sizeof(gen));
        }
        for (size_t i = 0; i < g_mod.slots.size(); i++) {
            const uint32_t idx = (uint32_t)i;
            const int64_t  own = g_mod.slots[i].owner_record;
            const uint64_t lu  = g_mod.slots[i].last_used;
            const int32_t  ls  = g_mod.slots[i].lease.load(std::memory_order_relaxed);
            d = fnv1a64_bytes(d, &idx, sizeof(idx));
            d = fnv1a64_bytes(d, &own, sizeof(own));
            d = fnv1a64_bytes(d, &lu,  sizeof(lu));
            d = fnv1a64_bytes(d, &ls,  sizeof(ls));
        }
        *out_lru_tick_digest = d;
    }
}

// 테스트 전용 — 제출된 op가 n_ops건 이상이 되는 순간 dispatcher가 취소 drain(§2 abort 경로)을
// 실행하게 한다. 0=off. ★프로덕션 경로는 항상 0(reset에서 0으로 복귀).
void ggml_moe_direct_test_abort_after_submit(int32_t n_ops) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_test_abort_after_submit_set_locked(n_ops);
}

// 테스트 전용 — 채널 이벤트 상태 스냅샷 + dispatcher 대기 복귀 횟수.
// ★불변식: terminal 처리 후 non-busy(idle) 채널의 manual-reset event는 전부 non-signaled여야
//   한다(하나라도 signaled면 고정 wait 배열이 즉시 반환해 hot-loop가 된다).
//   g_mod.mutex를 잡고 표본하므로 dispatcher pass 중간 상태가 아니라 pass 경계 상태를 본다.
void ggml_moe_direct_test_channel_event_state(int32_t * out_n_channels, int32_t * out_busy,
                                               int32_t * out_idle_signaled, uint64_t * out_wait_wakeups,
                                               uint64_t * out_sync_terminal) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    int32_t n = 0, busy = 0, idle_sig = 0;
    for (const auto & ch : g_mod.channels) {
        n++;
        if (ch.busy) { busy++; continue; }
        if (ch.ev && WaitForSingleObject(ch.ev, 0) == WAIT_OBJECT_0) idle_sig++;
    }
    if (out_n_channels)   *out_n_channels   = n;
    if (out_busy)         *out_busy         = busy;
    if (out_idle_signaled)*out_idle_signaled= idle_sig;
    if (out_wait_wakeups) *out_wait_wakeups = g_mod.stat_wait_wakeups.load(std::memory_order_relaxed);
    if (out_sync_terminal)*out_sync_terminal= g_mod.stat_sync_terminal;
}

// 테스트 전용 — 다음 pending 제출 1건을 sync_done 분기로 강제(one-shot). 0=disarm.
void ggml_moe_direct_test_force_sync_next(int32_t on) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dfac_test_force_sync_next_set_locked(on);
}

// 테스트 전용 — 종료 시퀀스 직접 행사(graceful=0 / abort=1). 멱등.
void ggml_moe_direct_test_shutdown(int32_t abort_mode) {
    dispatcher_stop_and_join(abort_mode != 0);
}

// ★★S-2 — 유계 종료 훅(테스트 전용). 1=완료 / 0=timeout.
int ggml_moe_direct_test_shutdown_bounded(int32_t abort_mode, int32_t timeout_ms) {
    return dispatcher_stop_and_join_bounded(abort_mode != 0, timeout_ms);
}

}  // extern "C"

#endif  // _WIN32
