// MoE GPU-side early-delivery probe (v2) - CUDA backend declarations.
// Spec: bench/techdev/repack_space/PROBE_GPU_EARLY_SPEC_R1.md (FROZEN v1.0)
//
// ASCII-only comments on purpose - see the note at the top of moe-gpuearly.cu (nvcc reads this
// source under the console codepage and a UTF-8 Korean comment swallows the next source line).
//
// Role boundary: this file and moe-gpuearly.cu own the CUDA EVENTS only - creation, capture-time
// recording (cudaEventRecordWithFlags + cudaEventRecordExternal), query, cudaEventElapsedTime,
// destruction, and the ownership/lifetime binding to the graph object. The gate, buffers, layer
// ordinals, pending accounting and the sidecar all belong to ggml/src/ggml-moe-gpuearly.cpp
// (ggml-base). ggml-base does not link CUDA, so this split is structural, not a preference.
#pragma once

#include "common.cuh"

#include "ggml-moe-gpuearly.h"

#ifdef USE_CUDA_GRAPH

// Called when the graph object dies (cache eviction or context teardown). This IS the eviction
// binding: it releases the events and honestly books any pending sample that can no longer be
// consumed (spec section 2 fail-close).
void ggml_cuda_gpuearly_graph_destroy(ggml_cuda_gpuearly_graph * g);

// Entry/exit of one graph_compute. Lets the hooks know which graph is running and in which
// phase, and refreshes the anchor count only on passes that actually walked the nodes
// (a replay pass walks nothing).
void ggml_cuda_gpuearly_begin_compute(ggml_backend_t backend, ggml_backend_cuda_context * ctx,
                                      ggml_cuda_graph * graph,
                                      bool warmup_phase, bool node_walk, bool capturing);
void ggml_cuda_gpuearly_end_compute(ggml_cuda_graph * graph);

// Immediately BEFORE cudaStreamBeginCapture - events must exist before the capture starts
// (spec section 2). Pool size = anchors observed on the previous direct pass * 2 (lazy);
// exceeding the hard cap is fail-close.
// warmup_phase is passed explicitly, not read from module state, because this runs BEFORE
// begin_compute in graph_compute (spec section 2 warmup exclusion; see implementation note).
void ggml_cuda_gpuearly_capture_begin(ggml_backend_cuda_context * ctx, ggml_cuda_graph * graph,
                                      bool warmup_phase);

// Immediately BEFORE cudaStreamEndCapture - records the boundary event as the graph's last node,
// so G_boundary is "all GPU work of this split has finished" = exactly the point the host waits
// for in cudaStreamSynchronize.
void ggml_cuda_gpuearly_capture_end(ggml_backend_cuda_context * ctx, ggml_cuda_graph * graph);

// Immediately AFTER cudaGraphLaunch - assigns boundary_seq and opens pending
// (spec section 2 replay state machine).
void ggml_cuda_gpuearly_after_launch(ggml_cuda_graph * graph);

// Router anchor hook (spec section 2 dual hook). The layer is taken from the tensor name:
//   argsort path = "ffn_moe_argsort-<il>" (dst) - fused path = "ffn_moe_topk-<il>" (ids).
// A name that does not match exactly is not an anchor (the call is a no-op).
void ggml_cuda_gpuearly_router_anchor(ggml_backend_cuda_context & ctx, const ggml_tensor * name_src,
                                      int router_path);

#endif // USE_CUDA_GRAPH
