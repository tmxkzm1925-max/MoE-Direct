// R-B4b (RB_DESIGN.md 2-5, RB4B_ATOM_SPEC.json): the observation/instrumentation unit.
// The 43 definitions below are the core text at the span the move map names, moved
// verbatim - body tokens unchanged, with the two `declared_substitutions` the spec spells
// out (the scheduling-K9 reads of `metrics_write_header_locked` and `io3_write_file`
// become one call to the core accessor `v3_io_param_snapshot_locked`).
// Windows only, exactly like the region it came from: the core TU compiles its
// non-Windows half from the stub region at the bottom of that file, and this unit has
// no stub half of its own (the moved definitions have no stub counterparts).

// The head include block of ggml-moe-direct.cpp, reproduced so the moved definitions
// compile in the environment they were written in.
#include "ggml-moe-direct.h"
#include "ggml-moe-trace.h"
#include "ggml-moe-phasea.h"
#include "ggml-moe-phaseb.h"
#include "ggml-moe-shadow.h"
#include "ggml-moe-leadtime.h"
#include "ggml-moe-gpuearly.h"
#include "ggml.h"
#include "gguf.h"
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

#include "ggml-moe-direct-types.h"
#include "ggml-moe-direct-manifest.h"
#include "ggml-moe-direct-state.h"
#include "ggml-moe-direct-observe.h"

using namespace ggml_moe_direct_impl;

namespace ggml_moe_direct_impl {

// ledger 레코드 1건 초기화 — 무관 필드는 전부 명시 NA(부록 14.3 필드 표).
ggml_moe_phaseb_ledger pb_led_new(uint8_t kind) {
    ggml_moe_phaseb_ledger r{};
    r.event_seq       = ggml_moe_phaseb_ledger_next_event_seq();
    r.qpc             = ggml_moe_phaseb_now_qpc();
    r.io_event_seq    = GGML_MOE_PHASEB_NA_U64;
    r.intent_ordinal  = GGML_MOE_PHASEB_NA_U64;
    r.record_idx      = -1;
    r.admission_seq   = GGML_MOE_PHASEB_NA_U64;   // ★owner generation 순번(호출자가 채운다)
    r.decision_seq    = GGML_MOE_PHASEB_NA_U64;
    r.request_epoch   = GGML_MOE_PHASEB_NA_U32;
    r.graph_seq       = GGML_MOE_PHASEB_NA_U32;
    r.graph_epoch     = GGML_MOE_PHASEB_NA_U32;
    r.rank            = GGML_MOE_PHASEB_NA_U16;
    r.layer           = GGML_MOE_PHASEB_NA_U16;
    r.pred_rank       = GGML_MOE_PHASEB_NA_U16;
    r.kind            = kind;
    r.origin          = 0xFF;
    r.disposition     = 0xFF;
    r.partition       = GGML_MOE_PB_P_NA;
    // ★r1 M2 — 스탬프는 **owner 가 채운다**. 여기서 전역(ambient)을 기본값으로 넣지 않는다
    //   (controller §1: 도달 여부와 무관하게 ambient read 금지). 미기입 = 0 = NA.
    r.stamp_k         = 0;
    r.stamp_n         = 0;
    r.stamp_qd        = 0;
    r.tuple_revision  = 0;
    r.n_expert_used   = pb_clamp_u8(g_mod.n_expert_used);
    r.qd_demand       = pb_clamp_u8(g_mod.pb_inflight_demand);
    r.qd_prefetch     = pb_clamp_u8(g_mod.pb_inflight_prefetch);
    r.qd_total        = pb_clamp_u8(g_mod.pb_inflight_demand + g_mod.pb_inflight_prefetch);
    return r;
}

// 참여 집합 동봉 방출 — 레코드당 PART_MAX 명을 담고 초과분은 **같은 `event_seq` 의 연속
//   레코드**로 잇는다. `n_participants` = 이 사건의 참여자 총수(연속 레코드 합과 일치).
void pb_led_emit_locked(ggml_moe_phaseb_ledger & r,
                        const ggml_moe_phaseb_ledger_owner * parts, uint32_t n_parts) {
    if (!g_mod.pb_collect) return;
    r.n_participants = n_parts;
    // ★r1 M4 — 참여 집합은 **절단하지 않는다**. 실 유계(in-flight ≤ QD · READY ≤ n_slots)이므로
    //   필요한 만큼의 연속 레코드를 만들어 전량을 적고, 그룹을 **한 번에 원자 적재**한다.
    //   버퍼가 모자라면 한 건도 적지 않고 `complete=0` 으로 fail-close 한다(부분 방출이
    //   "완전한 참여 집합"으로 오독되는 것을 구조로 막는다 — 구 판의 4×12=48 절단 폐지).
    const uint32_t per = (uint32_t)GGML_MOE_PB_LEDGER_PART_MAX;
    const uint32_t n_recs = (n_parts + per - 1) / per == 0 ? 1u : (n_parts + per - 1) / per;
    std::vector<ggml_moe_phaseb_ledger> group;
    group.reserve(n_recs);
    uint32_t done = 0;
    for (uint32_t rec = 0; rec < n_recs; rec++) {
        ggml_moe_phaseb_ledger cur = r;
        if (rec > 0) cur.flags |= GGML_MOE_PB_L_F_CONTINUATION;
        uint32_t take = n_parts - done;
        if (take > per) take = per;
        cur.n_part_stored = (uint16_t)take;
        for (uint32_t i = 0; i < take; i++) cur.participants[i] = parts[done + i];
        group.push_back(cur);
        done += take;
    }
    ggml_moe_phaseb_ledger_push_group(group.data(), (uint32_t)group.size());
}

// ★controller §6 참여 집합 — window 에 영향을 준 **stamped prefetch 상태**를 모은다:
//   in-flight/pending prefetch op + (READY 포함 시) PROBATION 슬롯. demand-claim 슬롯은
//   스탬프가 없으므로 참여자가 아니다. 반환 = 절단 전 실 개수(out 에는 cap 까지만 채운다).
void pb_collect_participants_locked(std::vector<ggml_moe_phaseb_ledger_owner> & out,
                                    bool include_ready, bool inflight_only) {
    out.clear();
    for (size_t i = 0; i < g_mod.pending_ops.size(); i++) {
        const PendingIo & op = g_mod.pending_ops[i];
        if (!op.in_use || op.origin != PIO_ORIGIN_PREFETCH) continue;
        // blocked window 의 인과 대상은 **실제 in-flight(제출됨·미종결)** prefetch 다.
        if (inflight_only && op.pb_submit_event_seq == GGML_MOE_PHASEB_NA_U64) continue;
        ggml_moe_phaseb_ledger_owner o{};
        o.request_epoch  = op.pb_request_epoch;
        o.graph_seq      = op.pb_graph_seq;
        o.intent_ordinal = op.intent_seq;
        out.push_back(o);
    }
    if (include_ready) {
        for (size_t i = 0; i < g_mod.slots.size(); i++) {
            const Slot & s = g_mod.slots[i];
            if (s.cache_class != GGML_MOE_PB_CLASS_PROBATION) continue;
            ggml_moe_phaseb_ledger_owner o{};
            o.request_epoch  = s.pf_request_epoch;
            o.graph_seq      = s.pf_graph_seq;
            o.intent_ordinal = s.pf_intent_ordinal;
            out.push_back(o);
        }
    }
}

// ★r1 M2 — owner 스탬프 기입 3형(op side table · READY slot · owner request). 어느 경로도
//   전역 현재값을 읽지 않는다(controller §1 ambient read 금지 — 도달 여부와 무관한 계약).
void pb_led_stamp_from_op(ggml_moe_phaseb_ledger & r, const PbOpStamp * st) {
    if (st == nullptr) return;
    r.stamp_k        = st->k;
    r.stamp_n        = st->n;
    r.stamp_qd       = st->qd;
    r.tuple_revision = st->rev;
    r.admission_seq  = st->gen_admission_seq;
}

void pb_led_stamp_from_slot(ggml_moe_phaseb_ledger & r, const Slot & sl) {
    r.stamp_k        = sl.pf_stamp_k;
    r.stamp_n        = sl.pf_stamp_n;
    r.stamp_qd       = sl.pf_stamp_qd;
    r.tuple_revision = sl.pf_stamp_rev;
    r.admission_seq  = sl.pf_gen_admission_seq;
}

// ★부록 14.1 ⓑ / 14.3 ⑤ — **미소비 READY(PROBATION) 축출** = 사용 불가 확정 = resolved_wasted.
//   접점 조사: evict 전용 카운터가 없어 이 신설이 4분할의 필수 성분이다(demand claim victim ·
//   prefetch claim victim 2지점 전건). 호출자는 슬롯을 덮어쓰기 **직전**에 부른다.
//   ★배타성은 구조로 보장된다 — 축출 직후 cache_class 가 PROBATION 이 아니게 되므로 재계상 불가.
void pb_note_ready_evict_locked(Slot & s) {
    if (!g_mod.pb_collect || g_mod.pb_censused) return;
    if (s.cache_class != GGML_MOE_PB_CLASS_PROBATION) return;
    ggml_moe_phaseb_count(GGML_MOE_PB_C_RESOLVED_WASTED, s.pf_request_epoch, 1);
    ggml_moe_phaseb_ledger r = pb_led_new(GGML_MOE_PB_L_READY_EVICT);
    r.request_epoch  = s.pf_request_epoch;
    r.graph_seq      = s.pf_graph_seq;
    r.intent_ordinal = s.pf_intent_ordinal;
    r.rank           = s.pf_rank;
    r.record_idx     = s.owner_record;
    r.origin         = 1;
    r.partition      = GGML_MOE_PB_P_RESOLVED_WASTED;
    pb_led_stamp_from_slot(r, s);
    pb_led_emit_locked(r);
}

// ★부록 14.6 — (graph_seq, target_mlpos) → 예측열 ids[16] 경량 스냅샷 유지/조회(유계 링).
inline uint64_t pb_pred_key(uint32_t graph_seq, int32_t mlpos) {
    return ((uint64_t)graph_seq << 16) | (uint64_t)(uint16_t)(mlpos & 0xFFFF);
}

void pb_pred_snap_put_locked(const ggml_moe_phaseb_intent & it) {
    if (!g_mod.pb_collect) return;
    if (g_mod.pb_pred_snap.size() < ModuleState::PB_PRED_SNAP_MAX)
        g_mod.pb_pred_snap.resize(ModuleState::PB_PRED_SNAP_MAX);
    const uint64_t key = pb_pred_key(it.graph_seq, it.target_mlpos);
    for (auto & e : g_mod.pb_pred_snap) if (e.key == key) return;   // 이미 있음(intent 1개/키)
    ModuleState::PbPredSnap & e = g_mod.pb_pred_snap[g_mod.pb_pred_snap_next];
    g_mod.pb_pred_snap_next = (g_mod.pb_pred_snap_next + 1) % ModuleState::PB_PRED_SNAP_MAX;
    e.key = key;
    e.k   = pb_clamp_u8((int32_t)it.k);
    for (int i = 0; i < GGML_MOE_PHASEB_TOPK; i++) e.ids[i] = it.ids[i];
}

// required expert 가 **`pred[K]`(zero-based 첫 제외 rank) 정확 1칸**이었는가(controller §3).
//   관측 전용 — counterfactual 산출 금지. 판정 불가 = NA.
uint16_t pb_pred_frontier_rank_locked(uint32_t graph_seq, int32_t mlpos, int32_t expert) {
    if (!g_mod.pb_collect || expert < 0) return GGML_MOE_PHASEB_NA_U16;
    const uint64_t key = pb_pred_key(graph_seq, mlpos);
    for (const auto & e : g_mod.pb_pred_snap) {
        if (e.key != key || e.k == 0) continue;
        if (e.k >= GGML_MOE_PHASEB_TOPK) return GGML_MOE_PHASEB_NA_U16;   // frontier 없음(K=16)
        return (e.ids[e.k] == (uint16_t)expert) ? (uint16_t)e.k : GGML_MOE_PHASEB_NA_U16;
    }
    return GGML_MOE_PHASEB_NA_U16;
}

std::string current_iso_like_timestamp() {
    SYSTEMTIME st;
    GetSystemTime(&st);
    char buf[64];
    snprintf(buf, sizeof(buf), "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",
              st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return std::string(buf);
}

bool metrics_write_line(const std::string & line, bool do_flush) {
    if (g_mod.metrics.file == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    BOOL ok = WriteFile(g_mod.metrics.file, line.data(), (DWORD)line.size(), &written, nullptr);
    if (!ok || written != line.size()) fatal_abort("metrics JSONL write failed err=%lu", (unsigned long)GetLastError());
    if (do_flush) {
        if (!FlushFileBuffers(g_mod.metrics.file)) fatal_abort("metrics JSONL flush failed err=%lu", (unsigned long)GetLastError());
    }
    return true;
}

json phase_to_json(const PhaseCounters & p) {
    json j;
    j["read_count"] = p.read_count.load(std::memory_order_relaxed);
    j["read_bytes"] = p.read_bytes.load(std::memory_order_relaxed);
    j["hit"] = p.hit.load(std::memory_order_relaxed);
    j["miss"] = p.miss.load(std::memory_order_relaxed);
    j["ensure_wait_ms_sum"] = p.ensure_wait_ms_sum.load(std::memory_order_relaxed);
    return j;
}

json common_fields(const char * type) {
    json j;
    j["type"] = type;
    j["invocation"] = g_mod.metrics.invocation;
    j["session"] = g_mod.metrics.session;
    j["leg"] = g_mod.metrics.leg;
    j["pid"] = (uint64_t)GetCurrentProcessId();
    j["seq"] = g_mod.metrics.seq++;
    j["ts"] = current_iso_like_timestamp();
    return j;
}

uint32_t compute_generation_high_word_locked() {
    uint32_t max_high = 0;
    for (const auto & s : g_mod.slots) {
        uint64_t gen = s.generation.load(std::memory_order_relaxed);
        uint32_t high = (uint32_t)(gen >> 32);
        if (high > max_high) max_high = high;
    }
    return max_high;
}

void metrics_write_header_locked() {
    if (!g_mod.metrics.enabled) return;
    json j = common_fields("header");
    j["arch"] = g_mod.arch;
    j["n_layer"] = g_mod.n_layer;
    j["n_expert"] = g_mod.n_expert;
    j["n_moe_layers"] = (int32_t)g_mod.moe_layers.size();
    j["slot_stride_max"] = g_mod.slot_stride_max;
    j["stride_bytes"] = g_mod.slot_stride_max; // v2: 층별 — 헤더엔 max를 싣는다(하위호환 필드명 유지)
    j["slot_count"] = g_mod.n_slots;
    j["slot_count_required"] = g_mod.att_n_slots_required;
    j["touch_abort_threshold"] = g_mod.touch_abort_threshold;
    j["att_all"] = g_mod.att_all;
    j["att_host"] = g_mod.att_host;
    j["att_nonhost"] = g_mod.att_nonhost;
    // ★부록A dual-scope 신규 3필드(기존 att_all/host/nonhost 의미 불변·추가만).
    j["att_raw"] = g_mod.att_raw;
    j["att_managed"] = g_mod.att_managed;
    j["att_excluded_nextn"] = g_mod.att_excluded_nextn;
    j["routed_scope"] = g_mod.routed_scope;
    // ★QD 레버(비동기 다중 outstanding — reviews/codex_qd_u6_design_xcheck.md 심의 3) 정보성 2필드.
    // 판독기 소비 없음 — 회귀 귀속을 위해 "이 기동에 실제 적용된 상한과 그 출처"만 남긴다.
    j["qd_effective"] = g_mod.qd_effective;
    j["qd_source"] = g_mod.qd_source;
    // ★profile identity echo(3단계 R3-1) — seal 이 이미 결속해 둔 **승인 profile id + 승인 expect
    //   digest**를 metrics header 에 additive 로 노출한다. 형상·런타임 값(n_expert/slot/att_* 등)
    //   만으로는 "같은 모델의 다른 승인 profile · 다른 repack 산출물" 치환을 구분할 수 없어
    //   계열 판독기의 cross-run identity(observe 런 ↔ replay clean timing 런) 비교축이 뚫린다.
    //   ★바이너리 사이드카 헤더가 아니라 **metrics JSON** 에 넣는다: §13.1 이 off 120~127 을 이미
    //   정의해 여유가 없고, kind 1~3 바이트 불변(§5 잠금 3)을 지키려면 JSON 이 유일한 additive
    //   표면이다(a0_cpu 계약도 identity 를 JSON report 축에서 결속 — 동일 계보).
    j["ref_profile_id"]    = g_mod.ref_profile_id;
    j["ref_expect_sha256"] = g_mod.ref_expect_sha256;
    // ★R4 — 재팩물 단위 identity: 같은 profile/expect 로 정상 재팩한 **다른 manifest** 를 구분한다
    //   (게이트 ⑦이 이미 계산·대조한 값의 echo — 재해시·추가 I/O 없음).
    j["repack_manifest_sha256"] = g_mod.repack_manifest_sha256;
    // ★R13 §2 — 식별 정직성 2필드. 같은 형상·같은 profile 이라도 **어느 좌표계로 읽었는지**를
    //   판독기가 알아야 한다(mode 불일치 = 비교 거부 · §4-1 범주 1). additive.
    {
        const bool virt = (g_mod.manifest_mode == ManifestMode::VIRTUAL);
        j["mode"] = std::string(virt ? "virtual" : "bin");
        j["schema_version"] = g_mod.manifest_schema_version;
        // ★IO3 사이드카 identity(§2 header) — 조인 키(`io3_boot_id`)와 해소값 echo.
        char ib[32];
        snprintf(ib, sizeof(ib), "0x%016llx", (unsigned long long)g_mod.io3_boot_id);
        j["io3_enabled"] = g_mod.io3_on;
        j["io3_boot_id"] = std::string(ib);
        j["io3_out"]     = g_mod.io3_out;
        j["io3_buf_mb"]  = (uint64_t)(g_mod.io3_buf_cap_bytes / (1024ull * 1024ull));
        // ★§3-1 resolved IO 파라미터 6종 {value, source} — **virtual 전용 계약**이라 bin 에서는
        //   키를 만들지 않는다(0 을 실으면 "해소했다"는 거짓 주장이 된다 · §5-2 bin=N/A 와 정합).
        if (virt) {
            static const char * kNm[6] = {
                "io_qd_total", "io_qd_demand_reserve", "io_qd_prefetch_child_max",
                "io_part_parallelism", "io_bounce_copy_workers", "io_bounce_copy_qd",
            };
            static const char * kSrc[4] = { "default", "manifest", "dedicated_env", "legacy_alias" };
            json jp = json::object();
            int32_t iopv[6] = { 0, 0, 0, 0, 0, 0 };
            uint8_t iops[6] = { 0, 0, 0, 0, 0, 0 };
            v3_io_param_snapshot_locked(iopv, iops);
            for (int i = 0; i < 6; i++) {
                const uint8_t s = iops[i];
                jp[kNm[i]] = json{ { "value",  iopv[i] },
                                   { "source", std::string(kSrc[s < 4 ? s : 0]) } };
            }
            j["io_params"] = jp;
        }
    }
    // ★B1 selector echo(PHASEB_DESIGN §2 하드 구속 ⑤) — 이 기동이 어느 I/O 경로였는지 박제.
    j["io_path"]           = g_mod.io_path_persistent ? "persistent" : "legacy";
    j["io_path_source"]    = g_mod.io_path_source;
    j["io_path_qualified"] = g_mod.io_path_qualified;
    // ★B1 mem echo(§6.3 · §7 "capture/predictor/intent/snapshot allocation 0 증명") — 항목별 산술값.
    {
        json jm = json::object();
        const uint64_t n_rec = (uint64_t)g_mod.moe_layers.size() * (uint64_t)g_mod.n_expert;
        jm["slot_pool_bytes"]        = (uint64_t)g_mod.pool_bytes;
        jm["control_q_slots"]        = MOE_DIRECT_CONTROL_Q_CAP;
        jm["control_q_bytes"]        = g_mod.io_path_persistent ? control_q_bytes_locked() : (uint64_t)0;
        jm["pending_table_bytes"]    = g_mod.io_path_persistent
            ? (uint64_t)g_mod.n_slots * (uint64_t)sizeof(PendingIo) : (uint64_t)0;
        jm["pending_by_record_bytes"]= g_mod.io_path_persistent ? n_rec * sizeof(int32_t) : (uint64_t)0;
        jm["channel_bytes"]          = g_mod.io_path_persistent
            ? (uint64_t)g_mod.qd_effective * (uint64_t)sizeof(IoChannel) : (uint64_t)0;
        jm["dispatcher_stack_reserve_bytes"] = g_mod.io_path_persistent ? (uint64_t)MOE_DIRECT_DISP_STACK : (uint64_t)0;
        // ★B2 자원 — b1_mem의 B2 키는 비규범이지만 phaseb_mem과 **같은 값**으로 채운다(§7.2 ⓓ 권장).
        uint64_t cap_slot = 0, cap_pool = 0, rt_b = 0, nm_b = 0, irec = 0, iq_b = 0, pstk = 0, co = 0, io_ = 0;
        uint32_t pth = 0;
        ggml_moe_phaseb_mem_echo(&cap_slot, &cap_pool, &rt_b, &nm_b, &irec, &iq_b, &pth, &pstk, &co, &io_);
        jm["capture_pool_bytes"]     = cap_pool;
        jm["intent_q_bytes"]         = iq_b;
        jm["predictor_worker_bytes"] = pstk;
        jm["router_norm_snapshot_bytes"] = rt_b + nm_b;
        jm["predictor_threads"]      = pth;
        jm["dispatcher_threads"]     = g_mod.io_path_persistent ? 1 : 0;
        j["b1_mem"] = jm;

        // ★phaseb_mem(§7.2) — seal 시 확정되는 allocation/capacity **만**(누적 카운터 금지).
        //   키 전건 필수(형상에 따라 0일 수는 있어도 키는 항상 존재).
        json jb = json::object();
        jb["slot_pool_bytes"]        = (uint64_t)g_mod.pool_bytes;
        jb["control_q_slots"]        = MOE_DIRECT_CONTROL_Q_CAP;
        jb["control_q_bytes"]        = g_mod.io_path_persistent ? control_q_bytes_locked() : (uint64_t)0;
        jb["pending_table_bytes"]    = g_mod.io_path_persistent
            ? (uint64_t)g_mod.n_slots * (uint64_t)sizeof(PendingIo) : (uint64_t)0;
        jb["pending_by_record_bytes"]= g_mod.io_path_persistent ? n_rec * sizeof(int32_t) : (uint64_t)0;
        jb["channel_bytes"]          = g_mod.io_path_persistent
            ? (uint64_t)g_mod.qd_effective * (uint64_t)sizeof(IoChannel) : (uint64_t)0;
        jb["dispatcher_stack_reserve_bytes"] = g_mod.io_path_persistent ? (uint64_t)MOE_DIRECT_DISP_STACK : (uint64_t)0;
        jb["dispatcher_threads"]     = g_mod.io_path_persistent ? 1 : 0;
        jb["capture_pool_slots"]     = ggml_moe_phaseb_capture_pool_slots();   // ★R1 #5 ① OFF=0
        jb["capture_slot_bytes"]     = cap_slot;
        jb["capture_pool_bytes"]     = cap_pool;
        jb["router_snapshot_bytes"]  = rt_b;
        jb["norm_snapshot_bytes"]    = nm_b;
        jb["router_norm_snapshot_bytes"] = rt_b + nm_b;
        jb["intent_q_slots"]         = ggml_moe_phaseb_intent_q_slots();       // ★R1 #5 ① OFF=0
        jb["intent_record_bytes"]    = irec;
        jb["intent_q_bytes"]         = iq_b;
        jb["predictor_threads"]      = pth;
        jb["predictor_stack_reserve_bytes"] = pstk;
        jb["predictor_worker_bytes"] = pstk;
        // ★모드 종속(프리페치가 아니라 phaseb_mode 기준 — R3 #4).
        //   ★8호: 활성 모드에 observe 가 추가된다(§13.4 — off 만 0).
        jb["pb_buffer_cap_bytes"]    = (ggml_moe_phaseb_mode() == GGML_MOE_PHASEB_OFF)
            ? (uint64_t)0 : ggml_moe_phaseb_buffer_cap();
        // ★observe exact-key 5종(§13.4 · seal 시 정적) — observe off = 전건 0.
        {
            const bool obs = ggml_moe_phaseb_signal_observe_on();
            jb["observe_router_snapshot_bytes"] = obs ? rt_b : (uint64_t)0;
            jb["observe_norm_snapshot_bytes"]   = obs ? nm_b : (uint64_t)0;
            jb["observe_bias_snapshot_bytes"]   = obs ? ggml_moe_phaseb_bias_snapshot_bytes() : (uint64_t)0;
            jb["observe_capture_pool_bytes"]    = obs ? cap_pool : (uint64_t)0;
            jb["observe_buf_bytes"]             = ggml_moe_phaseb_observe_buf_bytes();
        }
        j["phaseb_mem"] = jb;
    }
    // ★Phase B echo(SPEC_PHASEB_WIRE §7.1)
    {
        char bid[32];
        snprintf(bid, sizeof(bid), "0x%016llx", (unsigned long long)ggml_moe_phaseb_boot_id());
        j["phaseb_mode"]     = ggml_moe_phaseb_mode_name();
        j["phaseb_boot_id"]  = std::string(bid);
        j["phaseb_out"]      = std::string(ggml_moe_phaseb_out_prefix());
        j["phaseb_buf_mb"]   = ggml_moe_phaseb_buf_mb();
        j["prefetch_on"]     = g_mod.pb_prefetch_on;
        j["prefetch_k"]      = g_mod.pb_prefetch_on ? g_mod.pb_k : 0;
        j["prefetch_n"]      = g_mod.pb_prefetch_on ? g_mod.pb_n : 0;
        j["prefetch_off_reason"] = std::string(g_mod.pb_prefetch_on ? "" : ggml_moe_phaseb_prefetch_off_reason());
        // ★5c-T(부록 14.4) — 수집 축 echo. **수집 ON 에서만 키가 나타난다**(OFF 런의 header
        //   바이트가 종전과 동일해야 한다는 부록 14 서두 1급 계약).
        if (ggml_moe_phaseb_collect_on()) j["phaseb_collect"] = true;
        // ★mlpos 순 층별 stride_bytes — pb_sample 논리 byte 재구성의 권위(R1 #4).
        // ★R13 §2 재결속 — virtual 에서 이 축은 **슬롯 stride**(`layer_slot_bytes` = Σ region)다.
        //   bin 의 `stride_bytes` 는 virtual manifest 에 존재하지 않는 필드라 그대로 두면 0 열이
        //   나간다(판독기가 "층 stride 0"을 읽는다).
        {
            const bool virt = (g_mod.manifest_mode == ManifestMode::VIRTUAL);
            json js = json::array();
            for (const auto & L : g_mod.layers)
                js.push_back((uint64_t)(virt ? L.layer_slot_bytes : L.stride_bytes));
            j["moe_layer_stride_bytes"] = js;
            // ★§7 D-A2 분자축(신설) — "같은 record 를 bin 으로 재팩했다면 읽었을 바이트".
            //   bin 에서는 그게 곧 `stride_bytes` 이고 virtual 에서는 `legacy_record_bytes` 다
            //   ⇒ **두 모드에서 같은 의미**라 §4-1 범주 2(legacy-equivalent) 비교축이 성립한다.
            json jl = json::array();
            for (const auto & L : g_mod.layers)
                jl.push_back((uint64_t)(virt ? L.legacy_record_bytes : L.stride_bytes));
            j["moe_layer_legacy_bytes"] = jl;
        }
        // ★8호 — observe 실행 축 + descriptor exact object(§13.4). observe off = descriptor null.
        //   값 원천은 전부 phaseb 의 adapter 표(단일 출처) — direct 가 재계산하지 않는다.
        j["signal_observe_on"] = ggml_moe_phaseb_signal_observe_on();
        if (!ggml_moe_phaseb_signal_observe_on()) {
            j["observe_descriptor"] = nullptr;
        } else {
            json ml = json::array();
            for (int32_t i = 0; i < ggml_moe_phaseb_moe_layer_count(); i++)
                ml.push_back(ggml_moe_phaseb_moe_layer_at(i));
            j["observe_descriptor"] = json{
                { "adapter",        std::string(ggml_moe_phaseb_adapter_name()) },
                { "arch",           std::string(ggml_moe_phaseb_family_arch()) },
                { "norm_tensor",    std::string(ggml_moe_phaseb_norm_tensor_name()) },
                { "selection_fn",   std::string(ggml_moe_phaseb_selection_fn_name()) },
                { "bias_snapshot",  std::string(ggml_moe_phaseb_bias_snapshot_name()) },
                { "capture_width",  ggml_moe_phaseb_capture_width() },
                { "n_expert",       ggml_moe_phaseb_family_n_expert() },
                { "n_expert_used",  ggml_moe_phaseb_family_n_expert_used() },
                { "n_moe_layers",   ggml_moe_phaseb_moe_layer_count() },
                { "moe_layer_ids",  ml },
                // ★FROZEN_PB_OBSV_V2 §7(E6) — capability echo. exact object 를 **11키**로 확장한다
                //   (openai_moe_v1=true · qwen35moe_v1=false). 판독기는 adapter_id 로 재추론하지 않고
                //   이 echo 값만 소비한다(미등재 계열의 묵시 상속 경로 차단).
                { "output_crop_before_last_moe", ggml_moe_phaseb_output_crop_before_last_moe() },
            };
            // ★DSV4_ADAPTER_DESIGN §1.5 — **조건부 필드**. 층 마스크를 선언한 계열
            //   (deepseek4_v1: learned target 3..42)에서만 `signal_target_layer_ids` 를 싣는다.
            //   기존 계열은 마스크 비활성이라 exact object 가 **11키 그대로**다(descriptor 불변).
            //   값 원천은 phaseb 표 하나(마스크 술어 = ggml_moe_phaseb_signal_target_eligible).
            if (ggml_moe_phaseb_signal_layer_mask_active()) {
                json tl = json::array();
                for (int32_t i = 1; i < ggml_moe_phaseb_moe_layer_count(); i++) {
                    const int32_t l = ggml_moe_phaseb_moe_layer_at(i);
                    if (ggml_moe_phaseb_signal_target_eligible(l)) tl.push_back(l);
                }
                j["observe_descriptor"]["signal_target_layer_ids"] = tl;
            }
        }
        // ★denominator/source norm 원소별 exact-zero inventory(FROZEN_NORM_INVENTORY §4.1
        //   exact block). descriptor(형상 identity)와 측정 결과를 **섞지 않는** 별도 블록이며,
        //   observe_descriptor 는 이 블록 때문에 확장되지 않는다.
        //   · observe off / 미측정 = `null`(observe_descriptor 와 동일 규약) — 판독기는 이를
        //     "zero 0건"이 아니라 **unmeasured** 로 fail-close 한다(기본값 0 합성 금지).
        //   · zero 0건 계열은 블록이 **존재**하고 `zero_layers: []`·총계 0 을 싣는다.
        //   값 원천은 전부 phaseb 인벤토리(단일 출처) — direct 가 재계산하지 않는다.
        if (!ggml_moe_phaseb_signal_observe_on() || !ggml_moe_phaseb_norm_inventory_measured()) {
            j["observe_norm_inventory"] = nullptr;
        } else {
            json zl = json::array();
            const int32_t nz = ggml_moe_phaseb_norm_inventory_zero_layers();
            for (int32_t i = 0; i < nz; i++) {
                int32_t  src = -1, tgt = -1;
                uint32_t cnt = 0;
                uint64_t dg  = 0;
                if (ggml_moe_phaseb_norm_inventory_at(i, &src, &tgt, &cnt, &dg) != 0) continue;
                json ix = json::array();
                for (uint32_t k = 0; k < cnt; k++)
                    ix.push_back(ggml_moe_phaseb_norm_inventory_zero_index(i, k));
                char dgh[32];
                snprintf(dgh, sizeof(dgh), "0x%016llx", (unsigned long long)dg);
                zl.push_back(json{
                    { "source_layer",    src },
                    { "affected_target", tgt },   // ★moe_layer_ids[mlpos+1](산술 source+1 아님)
                    { "zero_count",      cnt },
                    { "zero_index",      ix },    // 오름차순 0-based · 전량(절단 없음)
                    { "zero_digest",     std::string(dgh) },
                });
            }
            j["observe_norm_inventory"] = json{
                { "inventory_version",        GGML_MOE_PHASEB_NORM_INVENTORY_VERSION },
                { "norm_tensor",              std::string(ggml_moe_phaseb_norm_tensor_name()) },
                { "source_layers_scanned",    ggml_moe_phaseb_norm_inventory_scanned() },
                { "zero_layers",              zl },
                { "total_zero_elements",      ggml_moe_phaseb_norm_inventory_zero_elements() },
                { "total_zero_source_layers", nz },
            };
        }
        // ★정정 6호 — 오힌트 test 표면 echo(§7.1). 비활성 = null(비-ohint 공식 판정의 게이트 20).
        if (ggml_moe_phaseb_test_hint_rule() == GGML_MOE_PHASEB_TEST_HINT_NONE) {
            j["phaseb_test_hint"] = nullptr;
        } else {
            j["phaseb_test_hint"] = json{ {"rule", std::string(ggml_moe_phaseb_test_hint_rule_name())},
                                          {"offset", ggml_moe_phaseb_test_hint_offset()},
                                          {"n_expert", g_mod.n_expert} };
        }
    }
    // ★프리페치 Phase A echo(SPEC_PHASEA_WIRE §10) — off면 "off"/boot_id 0. 사이드카 조인 키.
    {
        char bid[32];
        snprintf(bid, sizeof(bid), "0x%016llx", (unsigned long long)ggml_moe_phasea_boot_id());
        j["phasea_mode"]          = ggml_moe_phasea_mode_name();
        j["phasea_boot_id"]       = std::string(bid);
        j["phasea_out"]           = std::string(ggml_moe_phasea_out_prefix());
        j["phasea_buf_mb"]        = ggml_moe_phasea_buf_mb();
        j["phasea_eval_callback"] = ggml_moe_phasea_eval_callback_installed();
    }
    if (ggml_moe_trace_enabled()) j["qpc_frequency"] = ggml_moe_trace_qpc_frequency();
    metrics_write_line(j.dump() + "\n", true);
    g_mod.metrics.ops_since_periodic = 0;
    g_mod.metrics.last_periodic = std::chrono::steady_clock::now();
}

void metrics_snapshot_capture_locked(MetricsSnapshot & out, const char * type) {
    json j = common_fields(type);
    j["decode"] = phase_to_json(g_mod.phase[(int)Phase::DECODE]);
    j["prefill"] = phase_to_json(g_mod.phase[(int)Phase::PREFILL]);
    j["touch_events"] = g_mod.touch_events.load(std::memory_order_relaxed);
    j["touch_pages"] = g_mod.touch_pages.load(std::memory_order_relaxed);
    j["fallback_count"] = g_mod.fallback_count.load(std::memory_order_relaxed);
    // ★B1 dispatcher 활동(§7 착지 관문: dispatcher_demand_cmd>0) + control_q 무손실 증명 지표.
    j["dispatcher_demand_cmd"]  = g_mod.stat_demand_cmd;
    j["dispatcher_release_cmd"] = g_mod.stat_release_cmd;
    j["dispatcher_query_cmd"]   = g_mod.stat_query_cmd;
    j["dispatcher_join_count"]  = g_mod.stat_join;
    j["dispatcher_cancel_count"]   = g_mod.stat_cancel;
    j["dispatcher_abort_terminal"] = g_mod.stat_abort_terminal;
    j["control_q_full_waits"]   = g_mod.stat_q_full_waits.load(std::memory_order_relaxed);
    j["control_q_max_depth"]    = g_mod.stat_q_max_depth.load(std::memory_order_relaxed);
    if (ggml_moe_trace_enabled()) {
        j["read_immediate_count"] = g_mod.read_immediate_count.load(std::memory_order_relaxed);
        j["read_pending_count"] = g_mod.read_pending_count.load(std::memory_order_relaxed);
        j["acquire_call_count"] = ggml_moe_trace_acquire_call_count();
        j["chunk_call_count"] = ggml_moe_trace_chunk_call_count();
        {
            uint64_t hist[8];
            ggml_moe_trace_n_ids_histogram(hist);
            json jh = json::object();
            for (int i = 0; i < 7; i++) jh[std::to_string(i + 1)] = hist[i];
            jh["8plus"] = hist[7];
            j["n_ids_histogram"] = jh;
        }
        j["generation_high_word"] = compute_generation_high_word_locked();
        j["timeline_drop_count"] = ggml_moe_trace_timeline_drop_count();
        json jw = json::object();
        static const std::pair<uint8_t, const char *> wall_phases[2] = {
            {MOE_TRACE_PHASE_DECODE, "decode"}, {MOE_TRACE_PHASE_PREFILL, "prefill"},
        };
        for (const auto & wp : wall_phases) {
            int64_t c = 0, lcrit = 0, other = 0, l_lock = 0, l_data = 0;
            uint64_t mismatch = 0;
            ggml_moe_trace_wall_state_totals(wp.first, &c, &lcrit, &other, &l_lock, &l_data, &mismatch);
            json jp;
            jp["c_ticks"] = c; jp["lcrit_ticks"] = lcrit; jp["other_ticks"] = other;
            jp["l_lock_ticks"] = l_lock; jp["l_data_ticks"] = l_data;
            jp["wall_state_mismatch_count"] = mismatch;
            jw[wp.second] = jp;
        }
        j["wall_state"] = jw;
        j["decode_exact"] = phase_to_json(g_mod.phase_exact[(int)Phase::DECODE]);
        j["prefill_exact"] = phase_to_json(g_mod.phase_exact[(int)Phase::PREFILL]);
        j["phase_ctx_unknown_count"] = g_mod.phase_ctx_unknown_count.load(std::memory_order_relaxed);
    }
    // ★V1 무행동 증명 4필드(SPEC_PHASEA_WIRE §10) — terminal summary 에만(periodic 은 O(n) 비용).
    if (std::strcmp(type, "summary") == 0) {
        uint64_t tick_digest = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < g_mod.record_table.size(); i++) {
            const uint32_t idx = (uint32_t)i;
            const int32_t  sl  = g_mod.record_table[i].slot_index.load(std::memory_order_relaxed);
            const uint64_t gen = g_mod.record_table[i].slot_generation.load(std::memory_order_relaxed);
            tick_digest = fnv1a64_bytes(tick_digest, &idx, sizeof(idx));
            tick_digest = fnv1a64_bytes(tick_digest, &sl,  sizeof(sl));
            tick_digest = fnv1a64_bytes(tick_digest, &gen, sizeof(gen));
        }
        for (size_t i = 0; i < g_mod.slots.size(); i++) {
            const uint32_t idx = (uint32_t)i;
            const int64_t  own = g_mod.slots[i].owner_record;
            const uint64_t lu  = g_mod.slots[i].last_used;
            const int32_t  ls  = g_mod.slots[i].lease.load(std::memory_order_relaxed);
            tick_digest = fnv1a64_bytes(tick_digest, &idx, sizeof(idx));
            tick_digest = fnv1a64_bytes(tick_digest, &own, sizeof(own));
            tick_digest = fnv1a64_bytes(tick_digest, &lu,  sizeof(lu));
            tick_digest = fnv1a64_bytes(tick_digest, &ls,  sizeof(ls));
        }
        char rd[32], td[32];
        // ★R13 §4 mode-local — virtual 은 이 두 필드가 **재정의**된다: ops=demand-origin child
        //   제출 총수 · digest=18B/child(제출 순서). bin 축(16B/op·record 미스 수)을 그대로
        //   내면 "물리 read 1건 = record 1건"이 깨진 런에서 거짓 값이 나간다. 모드 국소 비교
        //   전용이며 header `mode` 불일치 = 비교 거부(§4-1 범주 1).
        const bool virt_axis = (g_mod.manifest_mode == ManifestMode::VIRTUAL);
        snprintf(rd, sizeof(rd), "0x%016llx",
                 (unsigned long long)(virt_axis ? g_mod.v3_readfile_digest : g_mod.readfile_digest));
        snprintf(td, sizeof(td), "0x%016llx", (unsigned long long)tick_digest);
        j["moe_readfile_digest"] = std::string(rd);
        j["moe_readfile_ops"]    = virt_axis ? g_mod.v3_readfile_ops : g_mod.readfile_ops;
        j["moe_lru_tick_digest"] = std::string(td);
        j["moe_tick_final"]      = g_mod.tick;
        // ★graph 형상 지문(ordered aggregate) — PA 무관. pa_off↔pa_timing 동일성 비교값(§3.3/§10).
        char gn[32], gs[32];
        snprintf(gn, sizeof(gn), "0x%016llx", (unsigned long long)g_mod.graph_nodes_digest);
        snprintf(gs, sizeof(gs), "0x%016llx", (unsigned long long)g_mod.graph_splits_digest);
        j["moe_graph_count"]         = g_mod.graph_count;
        j["moe_graph_nodes_digest"]  = std::string(gn);
        j["moe_graph_splits_digest"] = std::string(gs);

        // ── ★Phase B summary(SPEC_PHASEB_WIRE §7.3) — C4′·C3 권위.
        //    ★metrics on이면 PB 모드 무관 **항상** 산출한다(C4′는 off arm이 digest를 내야 성립).
        {
            uint64_t d_int = 0, d_disp = 0, d_tr = 0, d_cls = 0, d_rf = 0, rf_ops = 0, n_int = 0;
            // cache-class digest는 finalize 시 슬롯 스냅샷 — 여기서 1회 확정한다(§8 endpoint 보조).
            for (size_t i = 0; i < g_mod.slots.size(); i++) {
                ggml_moe_phaseb_digest_cacheclass((uint32_t)i, g_mod.slots[i].cache_class,
                                                  g_mod.slots[i].pf_intent_ordinal);
            }
            ggml_moe_phaseb_digests(&d_int, &d_disp, &d_tr, &d_cls, &d_rf, &rf_ops, &n_int);
            auto hex = [](uint64_t v) {
                char b[32]; snprintf(b, sizeof(b), "0x%016llx", (unsigned long long)v); return std::string(b);
            };
            j["moe_pb_intent_digest"]      = hex(d_int);
            j["moe_pb_disposition_digest"] = hex(d_disp);
            j["moe_pb_transition_digest"]  = hex(d_tr);
            j["moe_pb_cache_class_digest"] = hex(d_cls);
            j["moe_pb_readfile_digest"]    = hex(d_rf);
            j["moe_pb_readfile_ops"]       = rf_ops;
            j["moe_pb_intent_count"]       = n_int;
            // ★정정 6호 — u64·프로세스 누적(§7.3). 비활성 런 = 0. overflow=0 정상 런 등식(게이트 20):
            //   = pb_intent_predicted = moe_pb_intent_count = Σ phase_b_run.counts.test_hint_applied
            j["test_hint_applied"]         = ggml_moe_phaseb_test_hint_applied();
            // C3 카운터 전건(§6 표 · summary 키와 1:1)
            static const std::pair<int, const char *> kC3[] = {
                { GGML_MOE_PB_C_INTENT_CAPTURED,          "pb_intent_captured" },
                { GGML_MOE_PB_C_INTENT_PREDICTED,         "pb_intent_predicted" },
                { GGML_MOE_PB_C_INTENT_OVERFLOW,          "pb_intent_overflow" },
                { GGML_MOE_PB_C_PRE_DROPPED_TOO_LATE,     "pb_pre_dropped_too_late" },
                { GGML_MOE_PB_C_PRE_DROPPED_STALE,        "pb_pre_dropped_stale" },
                { GGML_MOE_PB_C_PRE_SKIPPED_RESIDENT,     "pb_pre_skipped_resident" },
                { GGML_MOE_PB_C_PRE_SKIPPED_PENDING,      "pb_pre_skipped_pending" },
                { GGML_MOE_PB_C_CANDIDATE,                "pb_candidate" },
                { GGML_MOE_PB_C_DROPPED_NO_SLOT_QD,       "pb_dropped_no_slot_qd" },
                { GGML_MOE_PB_C_DROPPED_PREFETCH_CAP,     "pb_dropped_prefetch_cap" },
                { GGML_MOE_PB_C_DROPPED_RESERVE,          "pb_dropped_reserve" },
                { GGML_MOE_PB_C_ADMITTED,                 "pb_admitted" },
                { GGML_MOE_PB_C_ISSUED,                   "pb_issued" },
                { GGML_MOE_PB_C_SUBMIT_FAILED,            "pb_submit_failed" },
                { GGML_MOE_PB_C_USED_JOIN,                "pb_used_join" },
                { GGML_MOE_PB_C_USED_PROMOTED,            "pb_used_promoted" },
                { GGML_MOE_PB_C_PREFETCH_ERROR_UNPUBLISH, "pb_prefetch_error_unpublish" },
                { GGML_MOE_PB_C_PREFETCH_RETRY,           "pb_prefetch_retry" },
                { GGML_MOE_PB_C_CAPACITY_CANCEL_ATTEMPT,  "pb_capacity_cancel_attempt" },
                { GGML_MOE_PB_C_DEMAND_PREFLIGHT_REJECT,  "pb_demand_preflight_reject" },
            };
            for (const auto & kv : kC3) j[kv.second] = ggml_moe_phaseb_counter(kv.first);
            // ★pb_wasted = CUT ⑦에서 일괄 확정(§0.1 — summary 전용).
            const uint64_t issued = ggml_moe_phaseb_counter(GGML_MOE_PB_C_ISSUED);
            const uint64_t uj     = ggml_moe_phaseb_counter(GGML_MOE_PB_C_USED_JOIN);
            const uint64_t up     = ggml_moe_phaseb_counter(GGML_MOE_PB_C_USED_PROMOTED);
            j["pb_wasted"] = (issued >= uj + up) ? (issued - uj - up) : (uint64_t)0;
            // ★5c-T 신설 카운터 5종 + ledger echo(부록 14.1·14.3) — **수집 ON 에서만** 키가
            //   존재한다. 기존 `pb_wasted` 산출식·시점은 문자 그대로 불변이며, 매핑 등식
            //   `pb_wasted == pb_resolved_wasted + pb_censored` 는 판독기·oracle 이 검사한다.
            if (g_mod.pb_collect) {
                static const std::pair<int, const char *> kC3T[] = {
                    { GGML_MOE_PB_C_RESOLVED_WASTED,          "pb_resolved_wasted" },
                    { GGML_MOE_PB_C_CENSORED,                 "pb_censored" },
                    { GGML_MOE_PB_C_ELIGIBLE_DEMAND_ACQUIRE,  "pb_eligible_demand_acquire" },
                    { GGML_MOE_PB_C_WAITED_ACQUIRE,           "pb_waited_acquire" },
                    { GGML_MOE_PB_C_OBSERVED_FRONTIER_MATCH,  "pb_observed_frontier_match" },
                };
                for (const auto & kv : kC3T) j[kv.second] = ggml_moe_phaseb_counter(kv.first);
                char lb[32];
                snprintf(lb, sizeof(lb), "0x%016llx", (unsigned long long)g_mod.pb_ledger_digest);
                j["pb_ledger"] = json{ {"records", g_mod.pb_ledger_records},
                                       {"written", g_mod.pb_ledger_written},
                                       {"digest",  std::string(lb)} };
            }
            // eligibility skip reason별(§4.1 enum 1:1)
            static const char * kSkip[GGML_MOE_PB_SKIP_COUNT] = {
                "NONE", "SOURCE_RANGE", "PREFILL", "MULTI_TOKEN", "LORA_SWAP", "SRC1_TYPE",
                "NONCONTIG", "NONFINITE", "CAPTURE_QUEUE_FULL", "REQUEST_DISABLED",
                "PREFETCH_OFF", "NO_SOURCE", "NCAP_ZERO",
            };
            json jsk = json::object();
            for (int i = 1; i < GGML_MOE_PB_SKIP_COUNT; i++) jsk[kSkip[i]] = ggml_moe_phaseb_skip_counter(i);
            j["pb_elig_skip"] = jsk;
            uint32_t qmd = 0, qmp = 0, qmt = 0;
            ggml_moe_phaseb_qd_max(&qmd, &qmp, &qmt);
            j["qd_max_demand"]   = qmd;
            j["qd_max_prefetch"] = qmp;
            j["qd_max_total"]    = qmt;
            uint64_t bus = 0, bcnt = 0;
            ggml_moe_phaseb_blocked_totals(&bus, &bcnt);
            j["demand_blocked_by_prefetch_us"]    = bus;
            j["demand_blocked_by_prefetch_count"] = bcnt;
            // ★★PB virtual 배선 진단 2종 — **C3 배타 closure 밖**이다(권위 =
            //   `SPEC_PB_VIRTUAL_RUNTIME_R1` §2-0 A0-4 · PHASEB_DESIGN §1 C3 부기).
            //   `deferred_credit_attempts` = credit 부족으로 admission 을 미룬 **시도** 수
            //     (같은 intent 의 중복 계수를 명문 허용 — disposition 이 아니다).
            //   `skipped_drain_turns`      = 실제 drain turn 당 1회(동일 상태 hot-loop 는 0).
            j["pb_admit_deferred_credit_attempts"] = g_mod.pb_admit_deferred_credit_attempts;
            j["pb_admit_skipped_drain_turns"]      = g_mod.pb_admit_skipped_drain_turns;
            uint64_t cap_slot = 0, cap_pool = 0, rt_b = 0, nm_b = 0, irec = 0, iq_b = 0, pstk = 0, co = 0, io_ = 0;
            uint32_t pth = 0;
            ggml_moe_phaseb_mem_echo(&cap_slot, &cap_pool, &rt_b, &nm_b, &irec, &iq_b, &pth, &pstk, &co, &io_);
            j["phaseb_mem_final"] = json{ {"capture_q_overflow", co}, {"intent_q_overflow", io_},
                                          {"pb_buffer_used_bytes", ggml_moe_phaseb_buffer_used()} };
            // ★incomplete 사유 echo(FROZEN_INCOMPLETE_REASON_MASK §7 M4/M5) — 표면은 metrics
            //   summary **단독**이고 둘 다 u32 JSON number 다(공통 헤더 reserved 불변 · phase_b_run
            //   요청별 투영 없음). complete=1 <=> mask=0 항등식의 판독 원천이며, `pb_cut_residual`
            //   은 현행 CUT 6 산출값(0x01 intent_depth ~ 0x20 inflight)을 그대로 echo 한다(§5).
            j["pb_incomplete_reason_mask"] = ggml_moe_phaseb_incomplete_mask();
            j["pb_cut_residual"]           = (uint32_t)g_mod.pb_cut_residual;
            // ★8호 — observe 카운터 11종(§13.4 · closure 등식의 좌우변 전건).
            //   ★FROZEN_PB_OBSV_V2 §5.1 — V2 에서 `observe_no_output`(12번째·**필수 키**) 추가.
            //   observe 축이 꺼진 런에서는 전건 0이다(필드 존재는 항상 — 판독기 결손 검출용).
            static const std::pair<int, const char *> kObs[] = {
                { GGML_MOE_PB_O_EXPECTED,           "observe_expected" },
                { GGML_MOE_PB_O_ELIGIBLE,           "observe_eligible" },
                { GGML_MOE_PB_O_NOT_ELIGIBLE,       "observe_not_eligible" },
                { GGML_MOE_PB_O_CAPTURED,           "observe_captured" },
                { GGML_MOE_PB_O_CAPTURE_MISS,       "observe_capture_miss" },
                { GGML_MOE_PB_O_RECORDED,           "observe_recorded" },
                { GGML_MOE_PB_O_VALID,              "observe_valid" },
                { GGML_MOE_PB_O_INVALID_NORM_ZERO,  "observe_invalid_norm_zero" },
                { GGML_MOE_PB_O_INVALID_NONFINITE,  "observe_invalid_nonfinite" },
                { GGML_MOE_PB_O_INVALID_OVERFLOW,   "observe_invalid_overflow" },
                { GGML_MOE_PB_O_CAPTURE_Q_OVERFLOW, "observe_capture_q_overflow" },
                { GGML_MOE_PB_O_NO_OUTPUT,          "observe_no_output" },
            };
            for (const auto & kv : kObs) j[kv.second] = ggml_moe_phaseb_observe_counter(kv.first);
        }
        // ★R13 §3-5 — PB physical projection sentinel. virtual 런은 PB 물리 좌표(op 1건=record
        //   1건·stride_bytes)를 만들 수 없다: 제출부가 `pb_note_submit_locked` 를 타지 않으므로
        //   이 4필드는 **실값이 아니라 미적용**이다. 0 을 내면 "물리 read 0건"이라는 거짓 실값이
        //   되므로 도메인 밖 센티널(-1 · "na")로 낸다.
        //   ★동치 규칙(ⓓ 조건 1): 4필드 전부 센티널 ⟺ `pb_physical_applicable=false`.
        //   판독기는 불리언 하나만 보고도, 필드만 보고도 같은 결론에 도달해야 한다.
        {
            const bool virt = (g_mod.manifest_mode == ManifestMode::VIRTUAL);
            if (virt) {
                j["moe_pb_readfile_digest"] = std::string("na");
                j["moe_pb_readfile_ops"]    = -1;
                j["qd_max_demand"]          = -1;
                j["qd_max_prefetch"]        = -1;
                j["qd_max_total"]           = -1;
            }
            j["pb_physical_applicable"] = !virt;
            // ★§2 재결속 — virtual 의 immediate/pending 은 **demand-origin 물리 child** 계수다
            //   (children flags b0 기준). bin 축은 trace 활성 시에만 나가는 종전 필드 그대로다.
            if (virt) {
                j["read_immediate_count"] = g_mod.v3_child_immediate_demand;
                j["read_pending_count"]   = g_mod.v3_child_pending_demand;
            }
            // ★§2 summary `io3_sidecar` — kind별 {records, written, digest} + complete·
            //   dropped_records·buffer_bytes. OUT 미설정·버퍼 초과·shutdown 절단의 **보고처**다
            //   (arm 유효성 판정의 단일 출처 — 파일이 없을 때도 이 블록은 나간다).
            if (g_mod.io3_requested) {
                auto blk3 = [](uint64_t recs, int32_t written, uint64_t dg) {
                    char b[32];
                    snprintf(b, sizeof(b), "0x%016llx", (unsigned long long)dg);
                    return json{ { "records", recs }, { "written", written != 0 },
                                 { "digest", std::string(b) } };
                };
                j["io3_sidecar"] = json{
                    { "records",  blk3(g_mod.io3_stat_records[0], g_mod.io3_stat_written[0], g_mod.io3_stat_digest[0]) },
                    { "children", blk3(g_mod.io3_stat_records[1], g_mod.io3_stat_written[1], g_mod.io3_stat_digest[1]) },
                    { "copies",   blk3(g_mod.io3_stat_records[2], g_mod.io3_stat_written[2], g_mod.io3_stat_digest[2]) },
                    { "complete",        g_mod.io3_complete },
                    { "dropped_records", g_mod.io3_dropped },
                    { "buffer_bytes",    g_mod.io3_stat_buffer_bytes },
                };
            }
        }
    }
    out.metrics_doc.swap(j);
}

void metrics_write_snapshot_locked(const char * type) {
    if (!g_mod.metrics.enabled) return;
    MetricsSnapshot snap;
    metrics_snapshot_capture_locked(snap, type);
    metrics_write_line(snap.metrics_doc.dump() + "\n", true);
}

void metrics_maybe_periodic_locked() {
    if (!g_mod.metrics.enabled) return;
    g_mod.metrics.ops_since_periodic++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_mod.metrics.last_periodic).count();
    if (g_mod.metrics.ops_since_periodic >= 64 || elapsed_ms >= 5000) {
        metrics_write_snapshot_locked("periodic");
        g_mod.metrics.ops_since_periodic = 0;
        g_mod.metrics.last_periodic = now;
    }
}

bool metrics_open_if_requested_locked(std::string & err) {
    std::string path = getenv_str("MOE_DIRECT_METRICS");
    if (path.empty()) { g_mod.metrics.enabled = false; return true; }
    std::wstring wpath = utf8_to_wide(path);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                            CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        err = "MOE_DIRECT_METRICS exclusive-create failed(err=" + std::to_string(e) +
              (e == ERROR_FILE_EXISTS ? ", file already exists" : "") + ")";
        return false;
    }
    g_mod.metrics.file = h;
    g_mod.metrics.path_w = wpath;          // ★R18 — 실패 되돌림용
    g_mod.metrics.enabled = true;
    g_mod.metrics.invocation = getenv_str("MOE_KS_INVOCATION_ID");
    g_mod.metrics.session = getenv_str("MOE_KS_SESSION_ID");
    g_mod.metrics.leg = getenv_str("MOE_KS_LEG");
    g_mod.metrics.seq = 0;
    g_mod.metrics.finalized = false;
    metrics_write_header_locked();
    return true;
}

// ★R18(교차 r6 G7) — **실패한 seal 은 계측 아티팩트를 남기지 않는다.** metrics 를 연 뒤
//   자원 확보가 실패하면 handle 이 살아 있고 header 1줄만 든 파일이 디스크에 남는데,
//   `CREATE_NEW` 배타 생성이라 그 잔류 파일이 **같은 경로 재시도까지 막는다**(운영자는
//   "왜 두 번째 기동이 파일 존재로 죽지" 만 보게 된다). 저빈도지만 실 자원 실패 경로다.
void metrics_discard_locked() {
    if (!g_mod.metrics.enabled) return;
    const std::wstring p = g_mod.metrics.path_w;
    g_mod.metrics.reset();                 // 핸들 close + enabled=false
    g_mod.metrics.path_w.clear();
    if (!p.empty()) DeleteFileW(p.c_str());
}

inline bool io3_on() { return g_mod.io3_on; }

// 버퍼 상한(§5-1 `MOE_DIRECT_IO3_BUF_MB`) — 초과하면 기록을 멈추고 complete=0 으로 내린다.
// ★조용히 자르지 않는다: 잘린 wire 를 유효한 것처럼 읽으면 arm 판정이 거짓이 된다.
// ★★S2(교차 r8 Z3) — **예상 추가량까지 보고** 판정한다. 구 판은 `used < cap` 만 봐서
//   `used < cap < used + need` 경계에서 행 묶음이 통째로 빠지는데 `complete` 는 1 로 남았다.
//   빠진 물리 행과 그만큼 낮아진 `records.n_children` 이 **서로 자기일관**이라 §5-6 검산이
//   통과하고 판독기는 그 잘린 wire 를 유효한 것으로 읽는다(false-valid). 조용한 누락 대신
//   wire 전체를 무효로 내리는 것이 이 함수의 계약이다.
inline bool io3_budget_reserve(uint64_t need) {
    const uint64_t used = (uint64_t)g_mod.io3_records.size()  * sizeof(Io3Record)
                        + (uint64_t)g_mod.io3_children.size() * sizeof(Io3Child)
                        + (uint64_t)g_mod.io3_copies.size()   * sizeof(Io3Copy);
    if (used + need <= g_mod.io3_buf_cap_bytes) return true;
    if (g_mod.io3_complete) {
        g_mod.io3_complete = false;
        log_err("[moe-direct][io3] buffer cap reached (%llu bytes; used=%llu need=%llu) - recording "
                "stopped, complete=0", (unsigned long long)g_mod.io3_buf_cap_bytes,
                (unsigned long long)used, (unsigned long long)need);
    }
    g_mod.io3_dropped++;
    return false;
}

// records 행 개설(claim 확정 시점) — 반환 = 행 인덱스(-1 = 미기록).
int32_t io3_open_record_locked(int32_t oi) {
    if (!io3_on() || !io3_budget_reserve(sizeof(Io3Record))) return -1;   // ★S2 예상량 판정
    const PendingIo & op = g_mod.pending_ops[(size_t)oi];
    Io3Record r{};
    r.record_seq    = g_mod.io3_record_seq++;
    r.record_id     = (uint64_t)op.m.record_idx;
    r.graph_seq     = op.pb_graph_seq;
    r.request_index = op.pb_request_epoch;
    r.layer         = (uint16_t)g_mod.moe_layers[(size_t)op.mlpos];
    r.expert        = (uint16_t)(g_mod.n_expert > 0 ? (op.m.record_idx % g_mod.n_expert) : 0);
    r.origin        = (op.origin == PIO_ORIGIN_PREFETCH) ? 1 : 0;
    r.flags         = (uint8_t)((op.phase == Phase::PREFILL ? 1 : 0) | (op.pb_is_retry ? 2 : 0));
    r.terminal_status = 0;                       // terminal 에서 확정
    r.attempt       = op.pb_is_retry ? 1 : 0;
    r.n_children    = 0;                         // ★submit 도달분만 센다
    r.slot          = op.m.slot;
    r.legacy_bytes  = g_mod.layers[(size_t)op.mlpos].legacy_record_bytes;
    r.generation    = op.m.generation;
    r.first_submit_qpc = 0;
    r.terminal_qpc  = 0;
    r.publish_seq   = 0;
    g_mod.io3_records.push_back(r);
    return (int32_t)g_mod.io3_records.size() - 1;
}

// ★§5-5 submit 전환 ①~③ — **I/O 호출 직전**에 사건 seq·QPC 를 발급하고 children 행과
//   그 child 의 계획된 copies 행을 materialize 한다. 그 뒤에 ReadFile 을 부른다.
//   (현 소스 선례는 "호출 성공 후 seq 발급"이지만 IO3 는 이 새 순서가 명시 우선이다.)
int32_t io3_child_submit_locked(int32_t oi, int32_t part, bool bounce,
                                uint64_t read_offset, uint32_t read_len,
                                int32_t * out_copy_row = nullptr) {
    if (out_copy_row) *out_copy_row = -1;
    // ★S2 — 이 child 가 내는 행 전체(child 1 + bounce 면 copy 1)를 한꺼번에 판정한다.
    if (!io3_on() ||
        !io3_budget_reserve(sizeof(Io3Child) + (bounce ? sizeof(Io3Copy) : 0))) return -1;
    const PendingIo & op = g_mod.pending_ops[(size_t)oi];
    const int32_t row_r = ((size_t)oi < g_mod.io3_op_row.size()) ? g_mod.io3_op_row[(size_t)oi] : -1;
    Io3Child c{};
    c.record_seq   = (row_r >= 0) ? g_mod.io3_records[(size_t)row_r].record_seq : ~0ull;
    c.read_offset  = read_offset;
    c.read_len     = read_len;
    c.layer        = (uint16_t)g_mod.moe_layers[(size_t)op.mlpos];
    c.expert       = (uint16_t)(g_mod.n_expert > 0 ? (op.m.record_idx % g_mod.n_expert) : 0);
    c.part         = (uint8_t)part;
    c.source_index = (uint8_t)g_mod.layers[(size_t)op.mlpos].parts[(size_t)part].source_index;
    c.origin       = (op.origin == PIO_ORIGIN_PREFETCH) ? 1 : 0;
    c.flags        = (uint8_t)((op.phase == Phase::PREFILL ? 2 : 0) | (op.pb_is_retry ? 4 : 0) |
                               (bounce ? 8 : 0));
    c.coalesce_group = 0xFFFFFFFFu;                 // A6(coalescing) 전에는 항상 단독
    c.generation   = op.m.generation;
    c.submit_event_seq = g_mod.io3_event_seq++;      // ★① I/O 호출 **직전** 발급
    c.submit_qpc   = ggml_moe_phaseb_now_qpc();
    // ★A5b — QD snapshot(§5-4): submit 직후 = **이 op 포함** child in-flight.
    // ★★PB virtual A1 — 증감은 여기가 아니라 **제출 경로**(`v3_child_inflight_add_locked`)가
    //   이미 했다. IO3 는 ②축을 읽기만 한다(계측 스위치가 스케줄러 회계를 바꾸지 않는다).
    c.qd_at_submit[0] = (uint8_t)(g_mod.v3_inflight_demand   > 255 ? 255 : g_mod.v3_inflight_demand);
    c.qd_at_submit[1] = (uint8_t)(g_mod.v3_inflight_prefetch > 255 ? 255 : g_mod.v3_inflight_prefetch);
    {
        const int32_t tot = g_mod.v3_inflight_demand + g_mod.v3_inflight_prefetch;
        c.qd_at_submit[2] = (uint8_t)(tot > 255 ? 255 : tot);
    }
    c.terminal_event_seq = ~(uint64_t)0;
    c.terminal_qpc = 0;
    c.terminal_status = 0;
    c.reserved     = 0;
    c.graph_seq    = op.pb_graph_seq;
    c.request_index= op.pb_request_epoch;
    g_mod.io3_children.push_back(c);
    const int32_t row_c = (int32_t)g_mod.io3_children.size() - 1;
    if (row_r >= 0) {
        Io3Record & rr = g_mod.io3_records[(size_t)row_r];
        rr.n_children++;                              // ★submit 도달분만
        if (rr.first_submit_qpc == 0) rr.first_submit_qpc = c.submit_qpc;
    }
    // ★② 이 child 의 **계획된 copy obligation** 을 여기서 materialize 한다(실행 여부 무관).
    //    bounce child 1건당 정확 1행 — CANCELLED 로 끝나도 행은 남는다(§5-5 obligation 규약).
    if (bounce) {
        Io3Copy cp{};
        cp.record_seq = c.record_seq;
        cp.src_child_submit_event_seq = c.submit_event_seq;   // ★NA 금지 — 항상 실값
        cp.bytes      = (uint32_t)g_mod.layers[(size_t)op.mlpos].parts[(size_t)part].slice_bytes;
        cp.layer      = c.layer;
        cp.expert     = c.expert;
        cp.part       = (uint8_t)part;
        cp.worker     = 0xFF;                                  // 미착수
        cp.status     = IO3_C_CANCELLED;                       // 기본 = 미실행 취소(의무는 남는다)
        cp.flags      = (uint8_t)((op.phase == Phase::PREFILL ? 1 : 0) | (op.pb_is_retry ? 4 : 0));
        cp.coalesce_group = 0xFFFFFFFFu;
        cp.enqueue_qpc = 0; cp.start_qpc = 0; cp.done_qpc = 0;
        cp.generation = op.m.generation;
        g_mod.io3_copies.push_back(cp);
        if (out_copy_row) *out_copy_row = (int32_t)g_mod.io3_copies.size() - 1;
    }
    return row_c;
}

// ★§Z-⑥ A6 — coalesced child 1건의 wire 발급(§5-4·§5-5). 단독 child 와 갈리는 지점만 적는다:
//   · children: `record_seq`=NA(귀속은 copies 가 권위) · `generation`=NA · flags **b4=coalesced**
//     (b3=bounce 는 세우지 않는다 — 바운스는 "단독 child 의 staging 경유"이고 여기선 대구간이다)
//     · `coalesce_group`=실값 · `expert`=run 시작 expert
//   · copies: **participant record 마다 1행**(바운스의 "child 당 1행"과 갈리는 유일한 지점) ·
//     flags **b1=src_kind=1(coalesce staging)** · `coalesce_group` 실값 · record_seq/generation 은
//     그 participant 의 것.
//   · 참가 record 의 `records.n_children` 은 여기서 1씩 오른다 — §5-6 cardinality 식
//     `n_children(R) = |단독 children| + |copies: src_kind=coalesce|` 와 맞물린다.
int32_t io3_coal_submit_locked(V3CoalRead & cr, int32_t * out_copy_row0) {
    if (out_copy_row0) *out_copy_row0 = -1;
    if (!io3_on()) return -1;
    // ★행 집합(child 1 + copies n)을 **한 번에** 판정한다 — 중간에 예산이 끊기면 참가 record
    //   일부만 의무 행을 갖게 되고, 그 wire 는 §5-6 cardinality 를 거짓으로 만든다.
    {
        const uint64_t need = sizeof(Io3Child) + (uint64_t)cr.ops.size() * sizeof(Io3Copy);
        if (!io3_budget_reserve(need)) return -1;   // ★S2 — 못 담으면 wire 를 무효로 내린다
    }
    const int32_t oi0 = cr.ops.empty() ? -1 : cr.ops[0];
    if (oi0 < 0) return -1;
    const PendingIo & op0 = g_mod.pending_ops[(size_t)oi0];
    const PartInfo & pi   = g_mod.layers[(size_t)cr.mlpos].parts[(size_t)cr.part];
    Io3Child c{};
    c.record_seq   = ~(uint64_t)0;                       // ★NA — 귀속은 copies 가 권위
    c.read_offset  = cr.offset;
    c.read_len     = cr.len;
    c.layer        = (uint16_t)g_mod.moe_layers[(size_t)cr.mlpos];
    c.expert       = (uint16_t)cr.expert0;               // run 시작 expert
    c.part         = (uint8_t)cr.part;
    c.source_index = (uint8_t)pi.source_index;
    c.origin       = (op0.origin == PIO_ORIGIN_PREFETCH) ? 1 : 0;
    c.flags        = (uint8_t)((op0.phase == Phase::PREFILL ? 2 : 0) | (op0.pb_is_retry ? 4 : 0) | 16);
    c.coalesce_group = cr.group_id;
    c.generation   = ~(uint64_t)0;                       // ★NA — slot 별 generation 은 copies 가 권위
    c.submit_event_seq = g_mod.io3_event_seq++;          // ★① I/O 호출 **직전** 발급
    c.submit_qpc   = ggml_moe_phaseb_now_qpc();
    // ★★PB virtual A1 — ②축 증가는 제출 경로가 이미 했다(위 단독 child 와 같은 규율).
    c.qd_at_submit[0] = (uint8_t)(g_mod.v3_inflight_demand   > 255 ? 255 : g_mod.v3_inflight_demand);
    c.qd_at_submit[1] = (uint8_t)(g_mod.v3_inflight_prefetch > 255 ? 255 : g_mod.v3_inflight_prefetch);
    {
        const int32_t tot = g_mod.v3_inflight_demand + g_mod.v3_inflight_prefetch;
        c.qd_at_submit[2] = (uint8_t)(tot > 255 ? 255 : tot);
    }
    c.terminal_event_seq = ~(uint64_t)0;
    c.terminal_qpc = 0;
    c.terminal_status = 0;
    c.reserved     = 0;
    c.graph_seq    = op0.pb_graph_seq;
    c.request_index= op0.pb_request_epoch;
    g_mod.io3_children.push_back(c);
    const int32_t row_c = (int32_t)g_mod.io3_children.size() - 1;
    // ★② 참가 record 마다 copy obligation 1행(실행 여부 무관 — 취소도 행이 남는다).
    const int32_t row0 = (int32_t)g_mod.io3_copies.size();
    for (size_t k = 0; k < cr.ops.size(); k++) {
        const PendingIo & op = g_mod.pending_ops[(size_t)cr.ops[k]];
        const int32_t row_r = ((size_t)cr.ops[k] < g_mod.io3_op_row.size())
                              ? g_mod.io3_op_row[(size_t)cr.ops[k]] : -1;
        Io3Copy cp{};
        cp.record_seq = (row_r >= 0) ? g_mod.io3_records[(size_t)row_r].record_seq : ~(uint64_t)0;
        cp.src_child_submit_event_seq = c.submit_event_seq;   // ★1:N 조인(§5-5 "coalesced 1:N 정상")
        cp.bytes      = (uint32_t)pi.slice_bytes;             // ★exact slice(§3-6 ⑥)
        cp.layer      = c.layer;
        // ★expert 는 run 산술이 아니라 **그 record 자신의 값**에서 뽑는다(순서 가정에 기대지 않는다).
        cp.expert     = (uint16_t)(g_mod.n_expert > 0 ? (op.m.record_idx % g_mod.n_expert) : 0);
        cp.part       = (uint8_t)cr.part;
        cp.worker     = 0xFF;
        cp.status     = IO3_C_CANCELLED;                      // 기본 = 미실행(의무는 남는다)
        cp.flags      = (uint8_t)((op.phase == Phase::PREFILL ? 1 : 0) | 2 | (op.pb_is_retry ? 4 : 0));
        cp.coalesce_group = cr.group_id;
        cp.enqueue_qpc = 0; cp.start_qpc = 0; cp.done_qpc = 0;
        cp.generation = op.m.generation;
        g_mod.io3_copies.push_back(cp);
        if (row_r >= 0) {
            Io3Record & rr = g_mod.io3_records[(size_t)row_r];
            rr.n_children++;                                  // §5-6 cardinality 의 coalesce 항
            if (rr.first_submit_qpc == 0) rr.first_submit_qpc = c.submit_qpc;
        }
    }
    if (out_copy_row0) *out_copy_row0 = row0;
    return row_c;
}

// ★R12 — 제출 **결과** 반영(§5-4 flags b0 · §2 재결속 계수).
//   children 행은 ReadFile **직전**에 만들어지므로(§5-5 submit 전환 ①②) pending/immediate 는
//   그때 알 수 없다 — 호출 반환 직후 이 함수가 확정한다. b0=1 은 `ERROR_IO_PENDING`(비동기
//   접수) · b0=0 은 인라인 완료(ReadFile TRUE)와 제출 실패다.
//   ★IO3 가 꺼져 있어도 계수는 센다: `read_immediate_count`/`read_pending_count`(§2 재결속)는
//   사이드카가 아니라 metrics 축이고, IO3 는 같은 사실을 행에 담을 뿐이다.
void v3_child_submit_outcome_locked(int32_t row_c, bool pending, bool demand) {
    if (demand) {
        if (pending) g_mod.v3_child_pending_demand++;
        else         g_mod.v3_child_immediate_demand++;
    }
    if (!pending) return;                                  // b0=0 = 행 생성 시 기본값
    if (!io3_on() || row_c < 0 || (size_t)row_c >= g_mod.io3_children.size()) return;
    g_mod.io3_children[(size_t)row_c].flags |= 0x01;       // b0 = pending
}

// ★R12 — §4 mode-local `moe_readfile_digest/ops`(virtual 재정의): 제출 순서로 child 1건당
//   18B `(u16 layer)(u16 expert)(u8 part)(u8 source_index)(u64 read_offset)(u32 read_len)`.
//   bin 의 16B/op 축(:2488)과 **다른 좌표계**이며 mode 국소 비교 전용이다(§4-1 범주 1).
void v3_readfile_digest_add_locked(uint16_t layer, uint16_t expert, uint8_t part,
                                   uint8_t source_index, uint64_t read_offset, uint32_t read_len) {
    uint64_t h = g_mod.v3_readfile_digest;
    h = fnv1a64_bytes(h, &layer,        sizeof(layer));
    h = fnv1a64_bytes(h, &expert,       sizeof(expert));
    h = fnv1a64_bytes(h, &part,         sizeof(part));
    h = fnv1a64_bytes(h, &source_index, sizeof(source_index));
    h = fnv1a64_bytes(h, &read_offset,  sizeof(read_offset));
    h = fnv1a64_bytes(h, &read_len,     sizeof(read_len));
    g_mod.v3_readfile_digest = h;
    g_mod.v3_readfile_ops++;
}

// child terminal 사건 — 사건 seq 는 submit 과 **같은 카운터**에서 뽑는다(§5-6 dense 폐합).
void io3_child_terminal_locked(int32_t row_c, uint8_t status) {
    if (!io3_on() || row_c < 0 || (size_t)row_c >= g_mod.io3_children.size()) return;
    Io3Child & c = g_mod.io3_children[(size_t)row_c];
    if (c.terminal_event_seq != ~(uint64_t)0) return;      // 1회만
    c.terminal_event_seq = g_mod.io3_event_seq++;
    c.terminal_qpc       = ggml_moe_phaseb_now_qpc();
    c.terminal_status    = status;
    // ★A5b — QD snapshot(§5-4): terminal 직후 = **이 op 제외** child in-flight.
    // ★★PB virtual A1 — 감소는 여기가 아니라 **회수 경로**(`v3_child_inflight_sub_locked`)가
    //   이 호출 **직전에** 이미 했다. 그래서 아래 snapshot 은 그대로 "이 op 제외"다.
    c.qd_at_terminal[0] = (uint8_t)(g_mod.v3_inflight_demand   > 255 ? 255 : g_mod.v3_inflight_demand);
    c.qd_at_terminal[1] = (uint8_t)(g_mod.v3_inflight_prefetch > 255 ? 255 : g_mod.v3_inflight_prefetch);
    {
        const int32_t tot = g_mod.v3_inflight_demand + g_mod.v3_inflight_prefetch;
        c.qd_at_terminal[2] = (uint8_t)(tot > 255 ? 255 : tot);
    }
}

// copy obligation 행 조회 — ★A6: **행 인덱스로 직접** 잡는다. 구 판은 child submit event seq
//   역탐색이었는데(전역 무중복이라 단독 child 에선 유일 키였다) coalesced 는 같은 키를 참가
//   record 수만큼의 행이 공유하므로(§5-5 1:N) 역탐색이 마지막 행만 지목한다.
Io3Copy * io3_copy_at_locked(int32_t row) {
    if (!io3_on() || row < 0 || (size_t)row >= g_mod.io3_copies.size()) return nullptr;
    return &g_mod.io3_copies[(size_t)row];
}

// DTO rule (e) [RB4A_ATOM_SPEC.json :: seam_map.dto_rules] - 위 조회의 **결과 포인터가
//   관측 측으로 새지 않도록** 쓰기는 이름 붙은 mutator 2 로만 한다(조회 자체는 그대로).
// 시각은 **가드 안에서** 읽는다 - 행이 없으면 before 와 같이 QPC 를 한 번도 읽지 않는다
//   [RB4A_ATOM_SPEC.json :: seam_map.members[].signature].
void io3_copy_note_enqueue_locked(int32_t row) {
    if (Io3Copy * cp = io3_copy_at_locked(row)) cp->enqueue_qpc = ggml_moe_phaseb_now_qpc();
}

void io3_copy_note_terminal_locked(int32_t row, uint8_t worker, uint64_t start_qpc,
                                   uint8_t status) {
    if (Io3Copy * cp = io3_copy_at_locked(row)) {
        cp->worker    = worker;
        cp->start_qpc = start_qpc;
        cp->done_qpc  = ggml_moe_phaseb_now_qpc();
        cp->status    = status;
    }
}

// records 행 종결(op terminal) — status·publish_seq·terminal_qpc 확정.
// ★R12 `latch_qpc` — §5-3 의 `terminal_qpc` 정의를 지키는 인자다: **성공=publish 직전(=지금)**
//   이지만 **실패·취소는 latch 시각**(첫 실패/취소가 확정된 순간)이다. 0=미래치(성공 경로).
void io3_close_record_locked(int32_t oi, uint8_t status, bool had_waiter, uint64_t latch_qpc) {
    if (!io3_on() || oi < 0 || (size_t)oi >= g_mod.io3_op_row.size()) return;
    const int32_t row = g_mod.io3_op_row[(size_t)oi];
    if (row < 0 || (size_t)row >= g_mod.io3_records.size()) return;
    Io3Record & r = g_mod.io3_records[(size_t)row];
    r.terminal_status = status;
    r.terminal_qpc    = (status != IO3_T_OK && latch_qpc != 0) ? latch_qpc : ggml_moe_phaseb_now_qpc();
    if (had_waiter) r.flags |= 4;
    if (status == IO3_T_OK) r.publish_seq = ++g_mod.io3_publish_seq;
    g_mod.io3_op_row[(size_t)oi] = -1;
}

bool io3_write_file(const std::string & path, uint32_t kind, uint32_t rec_bytes,
                    const void * payload, size_t n_rows) {
    // ★R13 — 이 kind 의 보고값(§2 summary `io3_sidecar`)을 여기서 확정한다. 파일 생성에 실패해도
    //   행 수·digest 는 사실이므로 남기고 `written=false` 로 구분한다.
    const size_t ks = (kind >= 1 && kind <= 3) ? (size_t)(kind - 1) : (size_t)0;
    g_mod.io3_stat_records[ks] = (uint64_t)n_rows;
    const std::wstring wp = utf8_to_wide(path);
    HANDLE h = CreateFileW(wp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        log_err("[moe-direct][io3] cannot create %s (CREATE_NEW) err=%lu", path.c_str(),
                (unsigned long)GetLastError());
        return false;
    }
    Io3Header hd{};
    std::memcpy(hd.magic, "MOEIO3\0\0", 8);
    hd.layout_version = 1;
    hd.file_kind      = kind;
    hd.record_bytes   = rec_bytes;
    hd.engine_mode    = (g_mod.manifest_mode == ManifestMode::VIRTUAL) ? 2u : 1u;
    hd.record_count   = (uint64_t)n_rows;
    LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
    hd.qpc_frequency  = (uint64_t)qf.QuadPart;
    hd.t0_qpc         = g_mod.io3_t0_qpc;
    hd.pid            = (uint32_t)GetCurrentProcessId();
    hd.complete       = g_mod.io3_complete ? 1u : 0u;
    hd.n_expert       = (uint32_t)g_mod.n_expert;
    hd.n_moe_layers   = (uint32_t)g_mod.moe_layers.size();
    hd.n_slots        = (uint32_t)g_mod.n_slots;
    hd.n_expert_used  = (uint32_t)g_mod.n_expert_used;
    hd.boot_id        = g_mod.io3_boot_id;
    hd.total_requests = 0;
    hd.dropped_records= g_mod.io3_dropped;
    // ★off 104..115 — virtual 만 채운다(bin=전부 0 = N/A · §5-2 · §5-6 bin 분기가 0 을 확인).
    //   ★R12 — 6종 **전건**을 §3-1 해소값 그대로 싣는다(구 판은 3종만·[5]에 staging 개수를
    //   실었다 = 이름과 다른 양). 값·source 의 단일 출처는 seal 이 채운 `v3_io_param_*` 다.
    if (hd.engine_mode == 2) {
        int32_t iopv[6] = { 0, 0, 0, 0, 0, 0 };
        uint8_t iops[6] = { 0, 0, 0, 0, 0, 0 };
        v3_io_param_snapshot_locked(iopv, iops);
        for (int i = 0; i < 6; i++) {
            const int32_t v = iopv[i];
            hd.io_params[i]    = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            hd.io_param_src[i] = iops[i];
        }
    }
    uint64_t dg = 0xcbf29ce484222325ull;
    if (n_rows) dg = fnv1a64_bytes(dg, payload, (size_t)rec_bytes * n_rows);
    hd.payload_digest = dg;
    g_mod.io3_stat_digest[ks] = dg;
    DWORD wn = 0;
    bool ok = WriteFile(h, &hd, (DWORD)sizeof(hd), &wn, nullptr) && wn == sizeof(hd);
    if (ok && n_rows) {
        const DWORD nb = (DWORD)((size_t)rec_bytes * n_rows);
        ok = WriteFile(h, payload, nb, &wn, nullptr) && wn == nb;
    }
    CloseHandle(h);
    g_mod.io3_stat_written[ks] = ok ? 1 : 0;
    return ok;
}

// ★R12 — flush 시점에 **아직 terminal 이 안 선 행**이 있는가(= wire 가 잘렸는가).
//   children 은 `terminal_event_seq` 가 NA 면 미종결이고, records 는 `terminal_status` 가 0 이면
//   미종결이다. 이 술어가 거짓인 채로 `complete=1` 을 내면 판독기가 잘린 arm 을 유효한 것으로
//   읽는다 — §5-6 fail-close 의 전제가 무너진다.
bool io3_all_terminal_locked() {
    for (const auto & c : g_mod.io3_children) if (c.terminal_event_seq == ~(uint64_t)0) return false;
    for (const auto & r : g_mod.io3_records)  if (r.terminal_status == 0)                return false;
    return true;
}

void io3_finalize_locked() {
    if (!g_mod.io3_on) return;
    g_mod.io3_on = false;                 // 재진입 금지(finalize 1회)
    // ★R13 — §2 summary 보고용 버퍼 사용량(행 수는 kind 별로 io3_write_file 이 채운다).
    g_mod.io3_stat_buffer_bytes = (uint64_t)g_mod.io3_records.size()  * sizeof(Io3Record)
                                + (uint64_t)g_mod.io3_children.size() * sizeof(Io3Child)
                                + (uint64_t)g_mod.io3_copies.size()   * sizeof(Io3Copy);
    // ★R12 — 미종결 행이 남았으면 정직하게 complete=0 으로 내린다(shutdown 중 절단·copy 잔여).
    if (!io3_all_terminal_locked() && g_mod.io3_complete) {
        g_mod.io3_complete = false;
        log_err("[moe-direct][io3] rows without a terminal event are still present at flush time - "
                "the wire is truncated by shutdown and the arm is invalid (complete=0)");
    }
    if (g_mod.io3_out.empty()) {
        // OUT 미설정 = 미기록 + complete=0 보고(§5-1) — 파일을 만들지 않는다.
        // ★해소 시점(seal)에 이미 내려가 있지만, flush 가 유일한 보고 시점이므로 여기서도 못박는다.
        g_mod.io3_complete = false;
        log_err("[moe-direct][io3] MOE_DIRECT_IO3=on but MOE_DIRECT_IO3_OUT is unset - nothing "
                "was written and the arm is invalid (complete=0)");
        return;
    }
    io3_write_file(g_mod.io3_out + "_records.io3",  1, (uint32_t)sizeof(Io3Record),
                   g_mod.io3_records.data(),  g_mod.io3_records.size());
    io3_write_file(g_mod.io3_out + "_children.io3", 2, (uint32_t)sizeof(Io3Child),
                   g_mod.io3_children.data(), g_mod.io3_children.size());
    io3_write_file(g_mod.io3_out + "_copies.io3",   3, (uint32_t)sizeof(Io3Copy),
                   g_mod.io3_copies.data(),   g_mod.io3_copies.size());
}

} // namespace ggml_moe_direct_impl

// The six lock-free public readers (`move_rows_extra`).  They keep `extern-C`: a
// public C entry point that acquired the named namespace would stop being the symbol
// the rest of the tree links against.  This move changes the FILE and nothing else.
extern "C" {

uint64_t ggml_moe_direct_touch_events(void)   { return g_mod.touch_events.load(std::memory_order_relaxed); }

uint64_t ggml_moe_direct_touch_pages(void)    { return g_mod.touch_pages.load(std::memory_order_relaxed); }

uint64_t ggml_moe_direct_fallback_count(void) { return g_mod.fallback_count.load(std::memory_order_relaxed); }

void ggml_moe_direct_phase_read_bytes_hit_miss(int32_t phase, uint64_t * out_read_bytes,
                                                uint64_t * out_hit, uint64_t * out_miss) {
    uint64_t rb = 0, h = 0, m = 0;
    if (phase == (int32_t)Phase::DECODE || phase == (int32_t)Phase::PREFILL) {
        const PhaseCounters & p = g_mod.phase[phase];
        rb = p.read_bytes.load(std::memory_order_relaxed);
        h  = p.hit.load(std::memory_order_relaxed);
        m  = p.miss.load(std::memory_order_relaxed);
    }
    if (out_read_bytes) *out_read_bytes = rb;
    if (out_hit)        *out_hit        = h;
    if (out_miss)       *out_miss       = m;
}

void ggml_moe_direct_phase_read_bytes_hit_miss_exact(int32_t phase, uint64_t * out_read_bytes,
                                                      uint64_t * out_hit, uint64_t * out_miss) {
    uint64_t rb = 0, h = 0, m = 0;
    if (phase == (int32_t)Phase::DECODE || phase == (int32_t)Phase::PREFILL) {
        const PhaseCounters & p = g_mod.phase_exact[phase];
        rb = p.read_bytes.load(std::memory_order_relaxed);
        h  = p.hit.load(std::memory_order_relaxed);
        m  = p.miss.load(std::memory_order_relaxed);
    }
    if (out_read_bytes) *out_read_bytes = rb;
    if (out_hit)        *out_hit        = h;
    if (out_miss)       *out_miss       = m;
}

void ggml_moe_direct_phase_read_bytes_hit_miss_window(int32_t phase, uint64_t * out_read_bytes,
                                                        uint64_t * out_hit, uint64_t * out_miss) {
    uint64_t rb = 0, h = 0, m = 0;
    if (phase == (int32_t)Phase::DECODE || phase == (int32_t)Phase::PREFILL) {
        const PhaseCounters & p = g_mod.phase_exact_window[phase];
        rb = p.read_bytes.load(std::memory_order_relaxed);
        h  = p.hit.load(std::memory_order_relaxed);
        m  = p.miss.load(std::memory_order_relaxed);
    }
    if (out_read_bytes) *out_read_bytes = rb;
    if (out_hit)        *out_hit        = h;
    if (out_miss)       *out_miss       = m;
}

} // extern "C"

#endif // _WIN32
