// MoE GPU-side early-delivery probe (v2) - CUDA backend implementation.
// Spec: bench/techdev/repack_space/PROBE_GPU_EARLY_SPEC_R1.md (FROZEN v1.0)
//
// NOTE ON LANGUAGE: comments in this file (and moe-gpuearly.cuh) are ASCII-only on purpose.
// nvcc's host pass reads the source under the active console codepage (CP949 here); a UTF-8
// Korean comment silently swallows the FOLLOWING source line, which cost one full CUDA rebuild
// in this round (members declared right after a Korean comment vanished from the struct).
// The ggml-base side of this probe keeps Korean comments because cl.exe compiles those TUs
// directly and is not affected. Sealed in BASELINE_TREE_GPUEARLY.md.
//
// Measured quantity (spec section 1):
//   Delta_tail_gpu(i) = G_boundary(i) - G_router(i) - a pair of GPU events on the same stream.
//     G_router   = external event recorded right after the router kernel (fused topk-moe, or argsort)
//     G_boundary = external event recorded as the LAST node of the graph (= the split boundary)
//
// Why cudaEventRecordExternal (spec section 2): recording an event inside a stream capture with
// default flags makes it graph-internal - it cannot be queried or timed after launch. Recording it
// as EXTERNAL yields a timestamp that is refreshed on every replay and readable from outside. That
// is what lets this probe measure WITHOUT disabling CUDA graphs (i.e. without changing the
// performance shape). v2a (graphs OFF) stays a non-adjudicating debug fallback.
//
// Replay state machine (spec section 2, frozen wording):
//   pending=false -> at graph launch assign boundary_seq, pending=true -> right after the target
//   split boundary sync returns, query/elapsed every event -> join with A1, commit C rows ->
//   pending=false.  Re-launch / re-capture / eviction / destruction while pending = fail-close.
//
// Single-pending premise: the measurement domain is a single GPU on a single stream (spec section
// 6 gate), and in decode every GPU split that carries a router is followed by the CPU MoE split
// whose input copy consumes it. If that premise ever breaks (an anchored graph launching twice
// with no consume in between), nothing is silently mixed: the loss surfaces as
// pending_unconsumed_count and the reader fail-closes. Not being silent is the contract.

#include "moe-gpuearly.cuh"

#ifdef USE_CUDA_GRAPH

#include <cstdint>
#include <cstring>
#include <vector>

// Hard cap on timing events per graph (spec section 2: "exceeding the hard cap = fail-close").
// The pool is anchors*2, so this admits 512 anchors per graph; a real split carries one router
// (one layer). The cap exists so an unexpected graph shape cannot quietly eat device resources.
#define GGML_CUDA_GPUEARLY_MAX_EVENTS 1024

struct ggml_cuda_gpuearly_anchor {
    uint32_t layer;
    uint8_t  path;      // GGML_MOE_GPUEARLY_ROUTER_*
};

// Probe state OWNED by the graph object, so its lifetime is exactly the graph's lifetime.
// That ownership is what binds event release (and the pending fail-close) to cache eviction.
struct ggml_cuda_gpuearly_graph {
    // events[0 .. n-1] = router anchors, events[n] = boundary. Size is pool_anchors*2.
    std::vector<cudaEvent_t>               events;
    size_t                                 pool_anchors = 0;
    // Metadata frozen at capture time; replays regenerate samples from it (there is no node walk
    // on a replay, so the host hooks do not fire - spec section 2 "router-path metadata replay").
    std::vector<ggml_cuda_gpuearly_anchor> anchors;
    // Per-anchor consumption state for the CURRENT launch. Reset at every launch, because each
    // replay produces a fresh sample per anchor. An anchor still false at abandon time is the
    // only kind that counts as a loss.
    std::vector<bool>                      anchor_done;
    size_t   boundary_idx = 0;
    size_t   observed     = 0;   // anchors counted on the last node-walk pass (pool sizing input)
    size_t   walk_count   = 0;   // running count for the current pass
    bool     capturing     = false;
    bool     capture_failed = false;  // an anchor could not be planted during this capture
    bool     armed         = false;   // holds a valid event set plus metadata
    // Warmup metadata (cross-review r1, blocker 2). The capture happens in the SAME call that
    // completes graph warmup, so the launch that follows it is a warmup-transition launch and its
    // samples must not count as eligible. capture_warmup records that state on the graph metadata;
    // launch_warmup applies it to the launch actually in flight. It must NOT be permanent per
    // anchor: every graph is captured during its warmup transition, so a permanent bit would mark
    // every later replay as warmup and leave eligible = 0.
    bool     capture_warmup = false;
    bool     launch_warmup  = false;
    uint64_t launch_count   = 0;
    bool     pending      = false;
    uint64_t boundary_seq = 0;
    uint64_t uid          = 0;   // diagnostic only (record.graph_uid)
    // The backend that launched this pending graph. The split-boundary consume compares the
    // just-synchronized backend against it by POINTER IDENTITY: that is the exact question
    // ("did we just wait on the GPU that owns these events?") and it costs nothing, whereas
    // ggml_backend_dev_type() would run cudaGetDeviceProperties on every input copy.
    ggml_backend_t launch_backend = nullptr;
    // Layers that already produced an anchor in this capture (exclusivity check). Layer counts are
    // tiny, so a linear scan is cheaper than a hash set.
    std::vector<uint32_t>                  capture_layers;
};

namespace {

struct GpuEarlyCudaState {
    ggml_backend_cuda_context * ctx     = nullptr;
    ggml_backend_t              backend = nullptr;   // backend of the current graph_compute
    ggml_cuda_gpuearly_graph *  cur     = nullptr;   // probe state for the current graph_compute
    ggml_cuda_gpuearly_graph *  pending = nullptr;   // the single launch awaiting consume
    bool warmup_phase        = false;
    bool consumer_registered = false;
};

GpuEarlyCudaState g_gec;

// Sticky-error discipline: when one of MY cuda calls fails, consume the last error right here.
// Otherwise the cudaGetLastError() at the end of ggml_cuda_compute_forward (ggml-cuda.cu) would
// attribute my failure to the NEXT op and CUDA_CHECK would abort - a probe killing the engine is
// the worst possible failure mode. On success we do not touch it (never swallow engine errors).
inline cudaError_t ge_cu(cudaError_t e) {
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
    }
    return e;
}

// Layer-name parsing lives in ggml-base (ggml_moe_gpuearly_parse_layer) so that the anchor hooks,
// the consume target binding, and the offline reader's regex are ONE contract rather than three
// hand-written copies. This file only calls it.

ggml_cuda_gpuearly_graph * ge_state_of(ggml_cuda_graph * graph) {
    if (!graph) return nullptr;
    if (!graph->gpuearly) {
        graph->gpuearly = new ggml_cuda_gpuearly_graph();
    }
    return graph->gpuearly;
}

// Discard a pending launch. Ordinals are consumed FIRST (spec section 4: dropped rows still
// consume the ordinal), so the reader's per-layer cardinality against the B axis stays aligned.
void ge_abandon(ggml_cuda_gpuearly_graph * g) {
    if (!g || !g->pending) return;
    // Ordinals are consumed FIRST (spec section 4: dropped rows still consume the ordinal), and
    // ONLY for anchors that were still outstanding - the ones already committed consumed theirs
    // at commit time, so counting them again would desynchronise the per-layer ledger.
    for (size_t i = 0; i < g->anchors.size(); i++) {
        if (i < g->anchor_done.size() && g->anchor_done[i]) continue;
        (void) ggml_moe_gpuearly_next_occurrence(g->anchors[i].layer);
    }
    g->pending = false;
    if (g_gec.pending == g) g_gec.pending = nullptr;
    ggml_moe_gpuearly_abandon_pending();
}

// Lazy event pool of anchors*2. Flags are cudaEventDefault because we need cudaEventElapsedTime;
// every pre-existing event in this tree is cudaEventDisableTiming and therefore unusable here
// (COE_GPU_EARLY_R1 Q3-c).
bool ge_ensure_pool(ggml_cuda_gpuearly_graph * g, size_t n_anchors) {
    if (n_anchors == 0) return false;
    const size_t want = n_anchors * 2;
    if (want > (size_t) GGML_CUDA_GPUEARLY_MAX_EVENTS) {
        ggml_moe_gpuearly_note_fault();
        return false;
    }
    while (g->events.size() < want) {
        cudaEvent_t ev = nullptr;
        if (ge_cu(cudaEventCreateWithFlags(&ev, cudaEventDefault)) != cudaSuccess || !ev) {
            ggml_moe_gpuearly_note_fault();
            return false;
        }
        g->events.push_back(ev);
    }
    g->pool_anchors = g->events.size() / 2;
    return true;
}

// Consumer invoked by ggml-base at the split boundary (A1). Spec section 2: query/elapsed every
// event, join with A1, commit the C rows, clear pending.
void ge_consume(void * /*user*/, void * synced_backend, const char * input_name,
                uint64_t sync_return_qpc, uint64_t a1_qpc) {
    ggml_cuda_gpuearly_graph * g = g_gec.pending;
    if (!g || !g->pending) return;

    // ---- Target binding: 3 conditions AND (cross-review r1, blocker 1) -----------------
    // A CPU MoE split copies SEVERAL generic inputs - the activation first, the ids second.
    // Closing pending on the first copy stamps a1_qpc BEFORE the ids materialise (measured on
    // M1: 26,160 of 26,448 topk rows sit right after the same layer's "attn_post_norm-<L>
    // (reshaped)" copy, median 6.7us earlier). That inflates H = T1 - a1_qpc and therefore
    // SHRINKS R_UB = sum min(Delta, max(0, D - H)) - an understatement, i.e. a silent one.
    // So we consume only when all three hold:
    //   (1) the synchronized backend is the one that launched this pending graph,
    //   (2) the copied input is an ids tensor (exact "ffn_moe_topk-<il>" match),
    //   (3) its layer is a layer this pending graph actually anchored.
    if (g->launch_backend && synced_backend != (void *) g->launch_backend) return;
    uint32_t want_layer = 0;
    if (ggml_moe_gpuearly_parse_layer(input_name, "ffn_moe_topk-", &want_layer) != 0) return;
    size_t n_match = 0;
    for (size_t i = 0; i < g->anchors.size(); i++) {
        if (!g->anchor_done[i] && g->anchors[i].layer == want_layer) n_match++;
    }
    if (n_match == 0) return;

    // All anchors share the boundary event, so query it once.
    const cudaEvent_t bev = g->events[g->boundary_idx];
    const bool        bok = ge_cu(cudaEventQuery(bev)) == cudaSuccess;

    for (size_t i = 0; i < g->anchors.size(); i++) {
        const ggml_cuda_gpuearly_anchor a = g->anchors[i];
        // Only this layer's outstanding anchors are settled by this boundary. Anchors of other
        // layers wait for their own ids copy, so every row carries ITS OWN a1_qpc.
        if (g->anchor_done[i] || a.layer != want_layer) continue;
        g->anchor_done[i] = true;

        // Warmup marker reaches the REPLAY ROW here (blocker 2). Priority stays intact: these
        // rows carry VALID without DIRECT, so commit() classifies them as excluded_warmup.
        uint32_t flags = GGML_MOE_GPUEARLY_FLAG_VALID
                       | (g->launch_warmup ? GGML_MOE_GPUEARLY_FLAG_EXCLUDED_WARMUP : 0u);
        uint64_t ns    = 0;

        if (!bok || ge_cu(cudaEventQuery(g->events[i])) != cudaSuccess) {
            // cudaErrorNotReady / invalid handle = loss (spec section 2). We are standing right
            // after a synchronize of this GPU, so "not ready" here IS a state-machine violation.
            flags = 0;
        } else {
            float ms = 0.0f;
            if (ge_cu(cudaEventElapsedTime(&ms, g->events[i], bev)) != cudaSuccess) {
                flags = 0;
            } else if (ggml_moe_gpuearly_ms_to_ns(ms, &ns) != 0) {
                flags = 0;   // non-finite / negative / over 2^63 = invalid row (spec section 3)
                ns    = 0;
            }
        }

        const uint32_t occ = ggml_moe_gpuearly_next_occurrence(a.layer);
        ggml_moe_gpuearly_note_anchor(a.path);
        ggml_moe_gpuearly_commit(g->boundary_seq, a.layer, a.path, flags, occ,
                                 ns, sync_return_qpc, a1_qpc, g->uid);
    }

    // Partial consumption: only this layer's anchors were settled. ggml-base decrements the
    // outstanding count and closes pending when it reaches zero, so a mid-way loss books only
    // the anchors that were still outstanding - never the ones already committed.
    ggml_moe_gpuearly_pending_consumed((uint64_t) n_match);

    bool all_done = true;
    for (size_t i = 0; i < g->anchor_done.size(); i++) {
        if (!g->anchor_done[i]) { all_done = false; break; }
    }
    if (all_done) {
        g->pending    = false;
        g_gec.pending = nullptr;
    }
}

inline void ge_register_consumer_once() {
    if (g_gec.consumer_registered) return;
    g_gec.consumer_registered = true;
    ggml_moe_gpuearly_set_consumer(ge_consume, nullptr);
}

} // namespace

// ===========================================================================
// Public entry points
// ===========================================================================

void ggml_cuda_gpuearly_graph_destroy(ggml_cuda_gpuearly_graph * g) {
    if (!g) return;
    // Destruction while pending = fail-close (spec section 2): once the events are gone that
    // sample can never be read, so it is booked as a loss rather than dropped silently.
    ge_abandon(g);
    if (g_gec.cur     == g) g_gec.cur     = nullptr;
    if (g_gec.pending == g) g_gec.pending = nullptr;
    for (cudaEvent_t ev : g->events) {
        if (ev) (void) ge_cu(cudaEventDestroy(ev));
    }
    delete g;
}

void ggml_cuda_gpuearly_begin_compute(ggml_backend_t backend, ggml_backend_cuda_context * ctx,
                                      ggml_cuda_graph * graph,
                                      bool warmup_phase, bool node_walk, bool capturing) {
    if (!ggml_moe_gpuearly_enabled()) { g_gec.cur = nullptr; return; }
    ge_register_consumer_once();

    ggml_cuda_gpuearly_graph * g = ge_state_of(graph);
    if (!g) { g_gec.cur = nullptr; return; }

    g_gec.ctx          = ctx;
    g_gec.backend      = backend;
    g_gec.cur          = g;
    g_gec.warmup_phase = warmup_phase;
    g->uid             = graph ? graph->uid : 0;
    g->walk_count      = 0;

    // Whether we are capturing was already decided by capture_begin (it clears the flag if the
    // event pool could not be secured). A replay pass walks no nodes, so no hook fires there -
    // its samples come from the frozen metadata instead.
    (void) node_walk;
    (void) capturing;
}

void ggml_cuda_gpuearly_end_compute(ggml_cuda_graph * graph) {
    if (!ggml_moe_gpuearly_enabled()) { g_gec.cur = nullptr; return; }
    ggml_cuda_gpuearly_graph * g = graph ? graph->gpuearly : nullptr;
    if (g && !g->capturing && g->walk_count > 0) {
        // observed is refreshed on DIRECT passes only: a capture pass already used the pool it
        // was sized against, and a replay pass walks nothing so it would zero this out.
        g->observed = g->walk_count;
    }
    g_gec.cur = nullptr;
}

void ggml_cuda_gpuearly_capture_begin(ggml_backend_cuda_context * ctx, ggml_cuda_graph * graph,
                                      bool warmup_phase) {
    if (!ggml_moe_gpuearly_enabled()) return;
    ge_register_consumer_once();

    ggml_cuda_gpuearly_graph * g = ge_state_of(graph);
    if (!g) return;

    // warmup_phase is passed in rather than read from g_gec because capture_begin runs BEFORE
    // begin_compute in graph_compute - reading the global here would pick up the PREVIOUS call's
    // value. (Found while wiring blocker 2; the ordering is easy to miss.)
    g->capture_warmup = warmup_phase;
    g->launch_count   = 0;

    // Re-capture while pending = fail-close (spec section 2): the events are about to be
    // overwritten, so the previous sample can no longer be read.
    ge_abandon(g);

    g->capturing      = false;
    g->capture_failed = false;
    g->armed          = false;
    g->anchors.clear();
    g->anchor_done.clear();
    g->capture_layers.clear();

    if (g->observed == 0) {
        // No router anchor in this graph (attention-only split, etc.) - not a probe target.
        // No events are created, so non-MoE graphs pay nothing.
        return;
    }
    if (!ge_ensure_pool(g, g->observed)) {
        // Hard cap exceeded or event creation failed. capturing stays false, so this pass's
        // anchors fall through to the DIRECT path and are written as excluded rows WITH their
        // ordinals - the ledger stays exact. ge_ensure_pool therefore books a fault (complete=0)
        // rather than a drop: booking a drop would count those same anchors twice.
        return;
    }
    g->capturing = true;
    (void) ctx;
}

void ggml_cuda_gpuearly_capture_end(ggml_backend_cuda_context * ctx, ggml_cuda_graph * graph) {
    if (!ggml_moe_gpuearly_enabled()) return;
    ggml_cuda_gpuearly_graph * g = graph ? graph->gpuearly : nullptr;
    if (!g || !g->capturing) return;

    g->capturing = false;

    const size_t n = g->anchors.size();
    if (n == 0) { g->armed = false; return; }

    // The boundary event lives at slot [n]; the pool is anchors*2 so for n >= 1 it always exists.
    bool ok = !g->capture_failed && n < g->events.size();
    if (ok) {
        g->boundary_idx = n;
        ok = ge_cu(cudaEventRecordWithFlags(g->events[n], ctx->stream(),
                                            cudaEventRecordExternal)) == cudaSuccess;
    }

    if (!ok) {
        // This capture yields no samples at all. The layer executions still happened, so the
        // registered anchors must CONSUME THEIR ORDINALS before being dropped - otherwise the
        // per-layer C ledger drifts against the B axis and the reader's cardinality check would
        // fail for the wrong reason. This is the single reconciliation point for capture faults.
        for (size_t i = 0; i < g->anchors.size(); i++) {
            (void) ggml_moe_gpuearly_next_occurrence(g->anchors[i].layer);
        }
        ggml_moe_gpuearly_note_drop((uint64_t) n);
        g->anchors.clear();
        g->anchor_done.clear();
        g->armed = false;
        return;
    }
    g->armed = true;
}

void ggml_cuda_gpuearly_after_launch(ggml_cuda_graph * graph) {
    if (!ggml_moe_gpuearly_enabled()) return;
    ggml_cuda_gpuearly_graph * g = graph ? graph->gpuearly : nullptr;
    if (!g || !g->armed) return;

    // Re-launch while pending = fail-close (spec section 2). Same graph or another one, the
    // earlier sample either gets its events overwritten or loses its chance to be consumed.
    if (g_gec.pending) ge_abandon(g_gec.pending);

    g->anchor_done.assign(g->anchors.size(), false);   // every replay is a fresh sample set
    // Warmup-transition launch (blocker 2): true while the graph is still in its warmup phase,
    // and for the first launch after a capture that happened during that phase. Later replays are
    // ordinary samples.
    g->launch_warmup  = g_gec.warmup_phase || (g->capture_warmup && g->launch_count == 0);
    g->launch_count++;
    g->boundary_seq   = ggml_moe_gpuearly_pending_begin((uint64_t) g->anchors.size());
    g->pending        = true;
    g->launch_backend = g_gec.backend;
    g_gec.pending     = g;
}

void ggml_cuda_gpuearly_router_anchor(ggml_backend_cuda_context & ctx, const ggml_tensor * name_src,
                                      int router_path) {
    if (!ggml_moe_gpuearly_enabled()) return;
    if (!name_src) return;

    const char * prefix = (router_path == GGML_MOE_GPUEARLY_ROUTER_FUSED)
                              ? "ffn_moe_topk-" : "ffn_moe_argsort-";
    uint32_t layer = 0;
    if (ggml_moe_gpuearly_parse_layer(name_src->name, prefix, &layer) != 0) return;

    ggml_cuda_gpuearly_graph * g = g_gec.cur;
    if (g) g->walk_count++;

    if (g && g->capturing) {
        // Capture path: plant the event as a graph node and freeze the metadata.
        // No row is emitted here - samples are produced at consume time by replaying this
        // metadata, because a replay performs no node walk (spec section 2).
        for (uint32_t seen : g->capture_layers) {
            if (seen == layer) {
                // One layer produced two anchors inside a single capture = fused XOR argsort
                // violation (spec section 6 gate keyed on the metadata). The run is fail-closed
                // either way; the ordinal is still consumed so the ledger stays monotone.
                (void) ggml_moe_gpuearly_next_occurrence(layer);
                ggml_moe_gpuearly_note_exclusivity_violation();
                g->capture_failed = true;
                return;
            }
        }
        if (g->anchors.size() >= g->pool_anchors) {
            // More anchors than the previous pass observed - the pool is short. Book it honestly
            // (ordinal consumed) instead of quietly truncating the sample set.
            (void) ggml_moe_gpuearly_next_occurrence(layer);
            ggml_moe_gpuearly_note_drop(1);
            g->capture_failed = true;
            return;
        }
        const size_t idx = g->anchors.size();
        if (ge_cu(cudaEventRecordWithFlags(g->events[idx], ctx.stream(), cudaEventRecordExternal))
                != cudaSuccess) {
            (void) ggml_moe_gpuearly_next_occurrence(layer);
            ggml_moe_gpuearly_note_drop(1);
            g->capture_failed = true;
            return;
        }
        ggml_cuda_gpuearly_anchor a;
        a.layer = layer;
        a.path  = (uint8_t) (router_path == GGML_MOE_GPUEARLY_ROUTER_FUSED
                                 ? GGML_MOE_GPUEARLY_ROUTER_FUSED
                                 : GGML_MOE_GPUEARLY_ROUTER_ARGSORT);
        g->anchors.push_back(a);
        g->capture_layers.push_back(layer);
        return;
    }

    // Direct path (execution outside a capture): there is no GPU event pair, so Delta cannot be
    // measured. Spec section 2 disposition = EXCLUDE THE SAMPLE. The row is still written: it
    // consumes the layer ordinal and proves the exclusion through its flags, which is what lets
    // the reader reconcile ordinals against the B axis and lets the physical-layer accounting
    // decompose the file completely.
    uint32_t flags = GGML_MOE_GPUEARLY_FLAG_VALID | GGML_MOE_GPUEARLY_FLAG_EXCLUDED_DIRECT;
    if (g_gec.warmup_phase) flags |= GGML_MOE_GPUEARLY_FLAG_EXCLUDED_WARMUP;

    const uint32_t occ = ggml_moe_gpuearly_next_occurrence(layer);
    ggml_moe_gpuearly_note_anchor((uint32_t) router_path);
    ggml_moe_gpuearly_commit(/*boundary_seq=*/0, layer, (uint32_t) router_path, flags, occ,
                             /*delta_ns=*/0, /*sync_return_qpc=*/0, /*a1_qpc=*/0,
                             g ? g->uid : 0);
}

#endif // USE_CUDA_GRAPH
