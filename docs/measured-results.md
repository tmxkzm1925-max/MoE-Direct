# Measurements, protocols and grades

[Back to the README](../README.md)

## The headline numbers

| What was measured | Result | The conditions it belongs to |
|---|---|---|
| Qwen3.5-122B decode | **5.65-6.26 tok/s per probe, arm average 5.96** | `PROBE` grade. Paired run on the v0.2.1 release binary (not re-run on this zip's binary), reference machine (32 GB RAM, one RTX 5080, a Gen5 NVMe). The bundle launcher in its `-Repro` release configuration: warm start and autosave off, budget autotune off with the budget fixed at 8192 MB, ctx 12288, QD 8, prefetch K=8/N=4, greedy. A normal product start on this machine sizes the cache at 12,288 MiB instead. |
| The same bundle's server started directly on plain mmap | **2.13-3.21 tok/s per probe, arm average 2.66**. Across the two matched sub-pairs (ON arm averages 5.96 and 5.86 vs OFF 2.66 and 2.48), the direct-read configuration ran a combined **2.3x** (2.2983x) faster | `PROBE` grade, same pair run. Same binary, same weights, same prompts, arms in A-B-B-A order in one sitting. Both arms ran at ctx 12288, so the two arms differ only in the read path - a single-variable pair. |
| Kimi K2.6, 1T class, out of 32 GB of RAM | 1.03 tok/s decode | `PROBE` grade, not a gate pass: budget 10240 MB, QD 8, prefetch off, coherent English output, measured on older staging binaries. |
| Output through direct-read against the same binary's plain-mmap path | token for token identical on gpt-oss-120b[^same] | The OFF/ON A-B-B-A protocol, greedy, 12 paired responses with identical token IDs. On Qwen3.5-122B the separate check is run-to-run: the official run's 12 token files were byte-identical to the parent anchor. No parity claim is made for K2.6, or for sampled decoding, which differs by construction. |
| Your weights after the one-time repack | byte for byte the source tensors[^bytes] | Every record SHA-256 compared against the source bytes, all records, no sampling, consumed fail-closed by both the launcher and the engine. |

Each row is expanded, with its full protocol and its caveats, in [Measured results](#measured-results)
and in [TECHNICAL.md](../TECHNICAL.md). Nothing above is cherry-picked, and no number in this
document appears without the conditions it was taken under.

## Measured results

Every number in this project belongs to a machine, a build, a configuration and a workload window.
A number without its conditions is a number you should not trust, so all of them are published with
theirs. Results scale primarily with NVMe read throughput; treat these as data points from this
box, not as promises for yours.

Numbers carry a grade. `OFFICIAL` means the frozen release-gate protocol on the reference machine,
with the gate verdict stated. `PROBE` means a deliberate measurement with a written protocol that
is not a gate run, and it promotes nothing. `LIVE` means something observed during ordinary use:
honest, but uncontrolled, and never to be read alongside an `OFFICIAL` number as if it were the
same kind of thing.

| Measurement | Result | Grade | Conditions |
|---|---|---|---|
| Qwen3.5-122B sustained decode | **5.59-5.69 tok/s**, `GATE1_SERVE: PASS` | `OFFICIAL` | Reference machine, budget 8192 MB, QD 8, prefetch on (K=8, N=4), ctx 12288, 4 measured reps across 2 ON arms, warmup rep excluded. Measured from a working tree that predates the release binary. |
| The same run's mmap arm | 2.4106 tok/s pooled, i.e. the direct-read arms were **2.3226x** and **2.3439x** faster | `OFFICIAL` | Same session, identical binaries. A separate ISLC-off correction run puts the adjusted decode ratio at 2.0695x instead; both figures, and why they differ by more than 5 %, are stated in full in TECHNICAL.md. |
| gpt-oss-120b output parity | 12 paired responses, token IDs identical | `OFFICIAL` | OFF/ON in A-B-B-A order within one session, greedy decoding, one request at a time. |
| Kimi K2.6, 1T class | 1.03 tok/s decode, coherent English, ~42 % expert-byte cache hits | `PROBE` | Budget 10240 MB, QD 8, prefetch off, older staging binaries. It has **not** passed the performance gate; do not plan interactive work around it. |
| Qwen3.5-122B multi-turn reuse | 314.9-316.5 context-tok/s perceived, from 8.1-8.3x prefix reuse; 9.3-9.6 tok/s of genuinely new work in the same runs | `PROBE` | Prefix-cache A/B on the reference machine. The two figures must always be read together: the large one is reuse, the small one is new work. |

**Which build these came from.** The paired figures in the box at the top of this page were
measured on the bundle in this release. The equivalent pair on the previous release, run on
2026-08-02 against the v0.2 binary, is kept in
[TECHNICAL.md](../TECHNICAL.md#the-release-pair-headline-source) as the earlier generation's
record rather than deleted, because that engine is not this one. The frozen gate record predates
both and keeps its own section.

The release-binary pair the headline comes from, the reference machine's full specification, the
paired protocol, the historical gate record, the release binary's lineage and every non-official
observation recorded so far live in **[TECHNICAL.md](../TECHNICAL.md)**. Nothing was moved there
to bury it. That file exists so this page can stay short enough to read.

## From the roadmap

| Where we are | Status |
|---|---|
| Serving a 1T-class MoE (Kimi K2.6, deepseek2 architecture) from a 32 GB machine | **Done and shown** - unedited single-take demo below. Format gate passed; performance gate not passed (1.03 tok/s, honestly labelled `PROBE`). |
| KV cache quantized to `q8_0` | **Measured and gated; the switch is still queued.** The quality gate passed on Qwen3.5-122B, with the divergence and perplexity figures in [TECHNICAL.md](../TECHNICAL.md#kv-cache-q8). What is not in this build is the opt-in switch that would let you turn it on, so the measurement is published ahead of the feature rather than the other way round. It was named for v0.2.3 and did not make it - that release turned out to be a convenience patch, and this is a feature. |

[^same]: **What was compared, and under what conditions.** The paired protocol runs the same
    engine build as four fresh-process arms in A-B-B-A order - direct-read off and on - with greedy
    decoding (temperature 0), the same prompts, the same seed and sampling parameters, and one
    request at a time. On gpt-oss-120b the OFF/ON A-B-B-A protocol produced 12 paired responses
    whose token IDs were identical. On Qwen3.5-122B the 12 token files of the official run were
    byte-identical to the parent anchor, i.e. the run is reproducible across runs as well. The
    direct-read against plain-mmap parity claim is scoped to that protocol on gpt-oss-120b; the
    Qwen3.5-122B statement is the separate run-to-run reproducibility check, not a second parity
    pair. Both are statements about greedy decoding, not about sampled decoding, where two runs
    differ by construction.
[^bytes]: **What "byte-preserving" means, precisely.** The one-time repack rewrites your expert
    tensors into a 4 KiB-aligned direct-read store. Every record written is SHA-256 compared
    against the corresponding source bytes, all records, no sampling; the report is consumed
    fail-closed by both the launcher and the engine, so a repack that did not fully verify cannot
    be served. There is no quantization step, no re-routing, no approximation, and no
    quality/space trade in this project. The general repacker test matrix covers 128-384 experts
    and 1-2 shards across four quantization layouts; separately, the shipped 512-expert, 6-shard
    Qwen3.5-397B profile has passed its model-specific format gate.
