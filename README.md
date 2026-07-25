# MoE-Direct

**Byte-preserving MoE expert streaming from NVMe — serve Mixture-of-Experts models 2.3×–13× larger than system RAM on one consumer Windows PC, with verified token-identical outputs.**

MoE-Direct keeps a model's routed expert weights **on disk**, not in memory. A repacked, 4 KiB-aligned expert store is read on demand from NVMe into a fixed slot cache exactly when the router asks for each expert. "Lossless" here means something specific and tested: **expert records are SHA-256-verified byte-equal to the source GGUF**, and under the documented deterministic greedy protocols the served outputs are **token-ID-identical** to the same engine running without direct-read. No approximation, no re-routing, no quality trade.

> **Status: project preview.** This repository currently contains this document set and the demo video below. Measurement methodology, raw run artifacts, full source (llama.cpp fork + patches), the repacker and the verification harness **will be published with `v0.2`** once the prefetch engine lands. Watch/star to follow.

## Demo — unedited single take

[![MoE-Direct demo: 1T-class MoE on a 32 GB RAM PC](https://img.youtube.com/vi/JDfrWMxwczk/maxresdefault.jpg)](https://youtu.be/JDfrWMxwczk)

**[▶ Watch on YouTube (2:49)](https://youtu.be/JDfrWMxwczk)** — cold boot to answer, no cuts: the 447 GB GGUF on disk, server load in ~19 s, live token streaming while Task Manager shows the NVMe sustaining multi-GB/s reads with system RAM staying under 32 GB, ending with the measured tok/s on screen. A second segment repeats the run on Qwen3.5-122B (5.2 tok/s decode on camera).

---

## Test environment (read this before the numbers)

Every number below comes from **one specific machine**. Results scale primarily with NVMe read speed — treat these as data points from this box, not promises for yours.

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 7800X3D (8C/16T) |
| GPU | NVIDIA RTX 5080, 16 GB VRAM |
| RAM | 32 GB DDR5-6000 (2×16 GB) |
| Model NVMe | KIOXIA EXCERIA PLUS G4 2 TB, **PCIe 4.0** (~5 GB/s effective reads observed under this workload) |
| OS | Windows 11 |

## Measured results — full model ladder

All four models were repacked with 100 % per-record verification and served with the same engine family.

| Model | Experts | Repacked expert store | Role in the ladder | Measured serving |
|---|---:|---:|---|---|
| gpt-oss-120B | 128 | 61 GB | **Integrity anchor** — OFF↔ON A/B/B/A protocol, 12 paired responses token-identical | functional gates all-pass |
| Mistral Small 4 | 256 | 70.2 GB | Format portability | 3.9 tok/s decode¹ |
| Qwen3.5-122B-A10B | 256 | 72.8 GB | Performance ladder | **5.14 tok/s decode mean²** |
| Kimi K2.6 (1T-class, Q3_K experts) | 384 | **436 GB** | Scale probe | 1.03 tok/s, coherent output³ |

¹ Measured on an earlier engine revision, before the async-I/O lever landed; not re-run since.
² Mean of two measured 511-token reps in the newest paired run (5.141 / 5.146). Full disclosure: the frozen release-gate for the async-I/O lever required ≥5.0 in every rep and passed only 2 of 4 reps (pooled mean 5.006) — 5.14 is a measured mean, **not an official gate pass**.
³ One non-gating curiosity run on then-current binaries: coherent output at 1.03 tok/s, ~42 % expert-byte cache hits (10 GiB cache ≈ 2.4 % of the model), zero touch/fallback counter events. No token-parity claim is made for this run.

**Why the jump from 122B to 1T-class?** A mundane reason: repacking needs roughly the model's size again on disk (source GGUF + expert store), and the local 2 TB NVMe already holds the four models above. Nothing in the stack is size-specific between these points — the repacker/consumer is verified across 128–384 experts, 1–2 shards and four quantization layouts. Mid-range (200–500 GB) results will follow as disk budget allows.

Sizing context (GiB, full GGUF vs this machine): Qwen3.5-122B = 72.3 GiB ≈ **1.5×** combined RAM+VRAM (48 GiB), ≈ 2.3× system RAM. Kimi K2.6 = 416.8 GiB ≈ **8.7×** combined, ≈ 13× system RAM.

## Where the time goes

At 5.14 tok/s the decode loop spends **113–116 ms of each ~194 ms token waiting on NVMe reads** (~58 %; the remainder is compute and engine overhead). That is why the lever ladder looks like this:

- **Async expert I/O (queue depth 8)** — shipped: +28.2 % decode vs QD1 in same-build paired exploration runs.
- **Next-layer expert prefetch — in design review, not built yet**: an offline counterfactual replay over complete I/O traces estimates **+46–58 ms/token** of that wait is recoverable. Honest limits, stated up front: the 57.8 ms figure is an oracle-like *perfect-prefetch proxy*; the 46.4 ms figure is an *implementable-policy proxy* in which 30.4 % of simulated prefetch reads use fallback durations; **neither is an end-to-end guarantee nor a bound**. What is directly measured is the early prediction signal itself: top-8 recall 77.3 %, top-16 recall 93.8 %. If — and only if — the implementable estimate fully realized, arithmetic gives ~6.8 tok/s on this box.

## Why this is not "just mmap"

We measured the mmap route first, on this hardware, and rejected it on data: under our conditions (expert working set ≫ RAM), page-cache eviction forced repeated re-streaming of expert pages and decode collapsed. That observation — with its control runs — will ship in the `v0.2` artifact drop. MoE-Direct replaces incidental caching with an **explicit, expert-granular slot cache**: 4 KiB-aligned repacked records, explicit positional `ReadFile` calls with overlapped I/O at queue depth, and a deterministic LRU with lease pinning. The claim is scoped: this is what we measured on this box for these models, not a universal law about mmap.

## Verification culture

- Every claim traces to an append-only run log with raw artifacts (published with `v0.2`).
- Paired comparisons only ever run on identical binaries within a session.
- Determinism is enforced: ReadFile-sequence digests, logical LRU-tick digests and per-run token hashes must match across paired runs, or the run is void.
- A manual coherence check on K2.6 exposed exactly why this matters: a degenerate (repetitive) output run showed inflated numbers — 1.87 "tok/s" and 72 % byte-hit — versus 1.03 tok/s / 42 % on coherent output. We publish the honest pair, and coherence checking is being promoted into the standard gate set.
- Changes are verified in layers: maintainer verification plus AI-assisted adversarial cross-review on qualifying changes, with raw verdicts kept.

## Roadmap

| Milestone | Content | Status |
|---|---|---|
| Project preview | this document set + demo video | **now** |
| Phase B: prefetch engine | persistent I/O dispatcher + deterministic next-layer expert prediction (no learned components) | **design under review; implementation has not started** |
| `v0.2` public release | full source tree (llama.cpp b10057 base + patches), repacker, verification harness, raw artifacts, build guide, Windows binaries | after Phase B validation |
| Beyond | cache-budget levers, 1T-class performance ladder | planned |

## Reading these numbers on your hardware

- **Decode speed scales with NVMe read throughput.** This box reads ~5 GB/s (PCIe 4.0) under this workload. A Gen3 drive will be proportionally slower; SATA SSDs are not a realistic substrate for this workload.
- **RAM sets the cache budget** (8–10 GiB slot cache here), which sets the hit rate. **VRAM only hosts the dense/attention layers** — it does not bound the expert store.
- **Compare regimes, not numbers.** A 128 GB box tiering RAM→GPU, a phone streaming from UFS, and this box streaming a disk-resident expert store from NVMe are three different sports. The regime here is *model ≫ RAM+VRAM combined* on consumer parts.

## FAQ

**Do you redistribute model weights?** Never. You bring your own GGUF; the repacker runs locally and records source hashes. Qwen, Kimi, Mistral, gpt-oss and all other model names belong to their respective owners; this project is affiliated with none of them.

**Windows-only?** Windows is the reference testbed (overlapped unbuffered I/O, VirtualAlloc slot pool, vectored exception guard). The I/O layer is abstracted; no Linux/macOS numbers are promised until they exist.

**AI involvement?** This project is built in an AI-assisted workflow (design, implementation and review loops involving multiple AI systems), with every change gated by the verification protocol above. Posts to upstream projects are written by the author personally.

## Credit & license

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), base commit `0bd0ec6` (b10057) — upstream copyright and license preserved; source releases will keep all upstream notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

MoE-Direct additions © 2026 tmxkzm1925-max, released under the [MIT License](LICENSE). The MoE-Direct name identifies this project and its official builds — see [TRADEMARKS.md](TRADEMARKS.md). If you use this work, please cite it ([CITATION.cff](CITATION.cff)).
