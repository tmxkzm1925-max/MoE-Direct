# NVMoE

**Lossless MoE serving from NVMe — run 100B+ Mixture-of-Experts models on a 32 GB RAM / 16 GB VRAM Windows PC by streaming experts on demand.**

NVMoE keeps MoE expert weights **on disk**, not in memory. A repacked, 4 KiB-aligned expert store is read directly from NVMe into a fixed slot cache exactly when the router asks for each expert — with a hard guarantee that the output is **bit-identical to running the unmodified model fully in memory**. The model being ~2×–14× larger than system RAM stops being a wall and becomes a throughput dial.

> **Status: evidence preview.** Numbers, methodology and verification artifacts are published here first; buildable source and binaries ship as `v0.2` when the prefetch engine (Phase B) lands. Watch/star to follow.

---

## Headline results (measured, single box)

Test machine: Ryzen 7 7800X3D · RTX 5080 16 GB · 32 GB DDR5-6000 · PCIe 4.0 NVMe (all consumer parts, Windows 11).

| Model | Weights on disk | Fits in RAM+VRAM? | Decode speed | Integrity |
|---|---:|:---:|---:|---|
| Qwen3.5-122B-A10B (Q4_K_M) | 72.8 GB | ❌ (1.5× over) | **5.1 tok/s sustained** | token-identical vs in-memory reference |
| Kimi K2.6 1T-class (Q3_K) | **447 GB** | ❌ (9.3× over) | 1.03 tok/s (first boot, untuned) | coherent output, zero integrity faults |

Supporting measurements:

- **Read-bound by design**: at 5.1 tok/s the decode loop spends ~57 % of its time waiting on NVMe reads (113–116 ms/token measured). Compute is nearly free — the disk is the dial.
- **Async expert I/O (QD8)**: +28 % decode speed over synchronous reads, paired A/B on identical binaries.
- **Prefetch potential (measured, not yet shipped)**: a counterfactual replay over full I/O traces estimates **+46–58 ms/token recoverable** by predicting next-layer experts one layer ahead (early-signal top-8 recall = 77.3 %, top-16 = 93.8 %). That is the Phase B engine currently being built — projected ~6.8–7.3 tok/s on the 122B config. *Estimates, not promises; published as replay methodology + raw traces.*
- 1T-class locality surprise: with a cache holding only **1.8 %** of expert bytes, K2.6 still hit ~42 % byte-hit on coherent output — expert routing locality is real and exploitable at 447 GB scale.

## Why this is not "just mmap"

We measured mmap first, and it lost — that's the origin of the project. With `mmap` + page cache, expert pages are evicted and re-streamed every few tokens once the working set exceeds RAM; per-token disk traffic explodes and decode collapses. NVMoE replaces incidental caching with an **explicit, expert-granular slot cache**: 4 KiB-aligned repacked records, unbuffered overlapped reads, deterministic LRU with lease pinning, and an async QD scheduler. The comparison harness, raw run logs and the mmap control runs are part of the published evidence.

## What "lossless" means here (precisely)

- Weights are never quantized further, approximated, or re-routed. The repacked expert store is **SHA-256-verified byte-equal** to the source GGUF tensor data (verified for every expert record).
- Greedy decoding produces **identical token IDs** to the same model served fully from memory — proven with an A/B/B/A protocol on a 128-expert model, all response pairs token-identical, and re-verified after every engine change.
- Determinism is enforced, not assumed: ReadFile-sequence digests, logical LRU-tick digests and per-run token hashes must match across paired runs, or the run is void.
- We do **not** claim bitwise-identical logits (floating-point op order may differ); the contract is token-ID parity under greedy decoding plus byte-identical weights.

## How it works (one paragraph)

A one-time repacker rewrites all routed expert tensors into `experts.bin`: one 4 KiB-aligned record per (layer, expert), uniform stride per layer, manifest-described, SHA-verified. At serve time a sealed slot cache (budget-configurable, e.g. 8–10 GiB) is backed by that file; every router decision triggers direct `ReadFile` (`FILE_FLAG_NO_BUFFERING | OVERLAPPED`, queue depth 8) for missing experts, publishes them into slots under a deterministic logical-tick LRU with lease pinning, and hands the compute kernel a pointer — the dense/attention layers live on GPU as usual. An instrumentation layer (metrics JSONL + binary sidecars) records every read, cache mutation and wait with enough fidelity that the entire I/O schedule can be **replayed offline counterfactually** — which is how the prefetch design was validated before a line of it was built.

## Verification culture

This project treats measurement the way production distributed systems do:

- every claim traces to an append-only run log with raw artifacts;
- paired comparisons only ever run on identical binaries within a session;
- a degenerate-output guard (repetition collapses routing and inflates cache hits — we measured exactly that on K2.6: 1.87 "tok/s" repetitive vs 1.03 tok/s coherent) keeps the numbers honest;
- three independent review layers (implementation, maintainer review, external adversarial cross-review) gate every merge.

## Roadmap

| Milestone | Content | Status |
|---|---|---|
| Evidence preview | this README + methodology docs + demo video | **now** |
| Phase B: prefetch engine | persistent I/O dispatcher + next-layer expert prediction (deterministic, no learned components) | design frozen-track, in build |
| `v0.2` public release | full source tree (llama.cpp b10057 base + patches), repacker, verification harness, build guide, Windows binaries | after Phase B validation |
| Beyond | budget/cache levers, K2.6 (1T-class) performance ladder toward 5 tok/s | planned |

## FAQ

**Slower than X?** Compare like for like: 128 GB-RAM boxes tiering RAM→GPU are a different sport. NVMoE's regime is *model ≫ RAM+VRAM combined* on consumer hardware.

**Windows-only?** Windows is the reference testbed (overlapped unbuffered I/O, VirtualAlloc slot pool, VEH guard). The I/O layer is abstracted; no Linux/macOS numbers are promised until they exist.

**Do you redistribute model weights?** Never. You bring your own GGUF; the repacker runs locally and records source hashes.

**AI involvement?** This project is built in an AI-assisted workflow (design, implementation and review loops involving multiple AI systems), with every change gated by the verification protocol above. Posts to upstream projects are written by the author personally.

## Credit & license

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), base commit `0bd0ec6` (b10057) — upstream copyright and license preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

NVMoE additions © 2026 tmxkzm1925-max, released under the [MIT License](LICENSE). The NVMoE name identifies this project and its official builds — see [TRADEMARKS.md](TRADEMARKS.md). If you use this work, please cite it ([CITATION.cff](CITATION.cff)).
