# MoE-Direct measurements

This is the technical record of MoE-Direct: what techniques are in this project, how they behave,
and the full measured evidence that they work on consumer hardware. The [README](README.md)
publishes the headline numbers with the conditions attached to each one and then stops. Everything
underneath them is here: the machines, the protocols, the gate record, the runs that never became
gate runs, and the arithmetic you need in order not to misread any of it. Read the README to see
what this project claims; read this file to see how far each claim was actually proven.

How the techniques themselves work is described elsewhere, and deliberately not repeated here: the
[How it works](README.md#how-it-works) section of the README covers the shape of the design, and
the technical note at [10.5281/zenodo.21739367](https://doi.org/10.5281/zenodo.21739367) covers the
reasoning and the alternatives that were rejected. This file is the evidence that those techniques
ran on real hardware.

Nothing in this file is promotional. Where a measurement is uncontrolled it says so and is kept out
of the official table rather than averaged into it, and where two runs disagree by more than a few
percent both are printed. Read the grade before you read the number.

**Grades used throughout**

| Grade | Meaning |
|---|---|
| `OFFICIAL` | Produced by the frozen release-gate protocol on the reference machine, from a working tree that predates the staging source tree `f5bbfcc4` - see the provenance note under [Official gate record](#official-gate-record-historical). Gate verdict stated. |
| `PROBE` | A deliberate measurement with a written protocol, but not a gate run. Not a tier promotion. |
| `LIVE` | Observed during ordinary use. Honest, but uncontrolled - never to be read alongside the `OFFICIAL` table as if it were the same kind of number. |

## Reference machine

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 7800X3D (8C/16T) |
| GPU | NVIDIA RTX 5080, 16 GB VRAM |
| RAM | 32 GB DDR5-6000 (2x16 GB) |
| Model NVMe | KIOXIA EXCERIA PLUS G4 2 TB, PCIe Gen5. Two different numbers, kept apart on purpose: about **5.4 GB/s** was the effective read rate observed while serving K2.6, and **6.5-8.1 GB/s** is what `fio` reports for large-block reads on this drive - a device ceiling, not a serving expectation. |
| OS | Windows 11 |

Second machine used for the 35B row: Acer Aspire Lite 16, Core i7-1355U, 16 GB DDR5-4800,
HOGE H8A1 512 GB NVMe, Windows 11, CPU-only (`-ngl 0`).

## The release pair (headline source)

`PROBE` - a written-protocol paired measurement on the release binary, not a run of the frozen
gate protocol. This is where the headline number at the top of the [README](README.md) comes from,
and it is the simplest number here on purpose: one binary, one machine, one sitting.

The protocol is fixed and is the same one both generations below were measured under: ON and OFF
arms in A-B-B-A order on one binary, three fixed greedy probes per arm (`n_predict` 256), in a
canonical environment with no ISLC.

### v0.2.1 release bundle

**What the two arms actually are.** The ON arms are the bundle's own launcher started in its
`-Repro` release configuration, which is what keeps the measurement clean rather than convenient:
warm start and autosave are off, budget autotune is off and the budget is fixed at the profile's
8192 MB, with ctx 12288, QD 8 and prefetch K=8/N=4. That is deliberately **not** the default
product start, which on this machine would size the cache at 12,288 MiB instead. The OFF arms are
the same bundle's `llama-server.exe` started directly with no `MOE_DIRECT` environment at all,
which is the plain mmap path, and that command line carries **ctx 65536**. The two arms therefore
differ in context size as well as in the read path.

| Arm | Decode average | Notes |
|---|---|---|
| ON (direct-read, cold) | **5.32-6.04 tok/s per probe, arm average 5.71** | bundle launcher with `-Repro`, ctx 12288, budget 8192 MB, prefetch on |
| OFF (plain mmap, cold) | **2.15-2.78 tok/s per probe, arm average 2.49** | same binary, same weights, no direct-read, ctx 65536 |
| OFF (mmap, second pass) | **2.20-2.70 tok/s per probe, arm average 2.46** | fresh server process; OS file cache state as recorded for that arm |
| ON (second pass) | **5.40-6.03 tok/s per probe, arm average 5.77** | fresh launcher process; OS file cache state as recorded for that arm |

Ratio, stated once: **2.3x** (ON average over OFF average, cache states as listed).
Per-probe figures, for the record: `ON cold 6.04 / 5.77 / 5.32; OFF cold 2.53 / 2.78 / 2.15; OFF second 2.48 / 2.70 / 2.20; ON second 5.86 / 6.03 / 5.40 tok/s (probes 1-3)`.
Two details for honest reading: probes 2 and 3 stopped at their natural EOS before the 256-token
cap (114 and 140 tokens, the same on every arm); and all three probes returned
character-identical text across all four arms, ON and OFF alike, under greedy decoding.

### v0.2 release bundle, 2026-08-02 (previous generation)

Kept because it was really measured, not because it describes this build. **The engine in v0.2.1
is not the engine these numbers came from**, so read them as the previous generation's record.

On 2026-08-02 we ran the paired re-issue the v0.2 README had promised: ON and OFF arms in A-B-B-A
order, on the exact v0.2 release binary, three fixed greedy probes per arm (`n_predict` 256; the
second and third probes stopped on their own at 114 and 140 tokens), canonical environment (no
ISLC, current defaults).

**What the two arms actually are.** The ON arms are the bundle's own launcher, started the way a
user starts it, running at the catalog defaults it chooses for this profile: ctx 12288, budget
8192 MB, QD 8, prefetch K=8/N=4, confirmed afterwards from the launcher's EFFECTIVE diagnostics.
The OFF arms are the same bundle's `llama-server.exe` started directly with no `MOE_DIRECT`
environment at all, which is the plain mmap path, and that command line carries **ctx 65536**. The
two arms therefore differ in context size as well as in the read path.

| Arm | Decode average | Notes |
|---|---|---|
| ON (direct-read, cold) | **5.84 tok/s** | bundle launcher at its defaults, ctx 12288, prefetch on |
| OFF (plain mmap, cold) | **2.43 tok/s** | same binary, same weights, no direct-read, ctx 65536 |
| OFF (mmap, second pass) | 2.41 tok/s | fresh server process; OS file cache warmed by the first OFF arm |
| ON (second pass) | 5.77 tok/s | fresh launcher process, no persisted slot cache (v0.2 had no warm start); OS file cache carried over from the earlier arms |

Ratio, stated once: **2.4x** (ON average over OFF average, cache states as listed). Read that
figure for what it is: the ratio **between two real shipping configurations**, the launcher as it
ships against the same bundle's server driven straight at mmap. It is not a single-variable
minimal pair, and because the context sizes differ it does not isolate the read path as the sole
cause. The OFF row is an absolute speed in tokens per second; the ratio is this one bolded figure
and nothing else. We keep those apart because a reader once mistook one for the other, and that
reader was right to complain.

Per-probe figures, for the record: ON cold 5.87 / 6.09 / 5.55, ON warm 5.63 / 6.13 / 5.56, OFF
cold 2.45 / 2.76 / 2.07, OFF warm 2.40 / 2.76 / 2.08 tok/s.

## Official gate record (historical)

`OFFICIAL` - reference machine, frozen constants K=8 / N=4 / QD=8. These are the frozen
release-gate numbers. They were measured before the release binary was cut, from an earlier
working tree, and they carry their own environment note below. They remain the gate record; the
pair table above is what supersedes them as the headline.

**Provenance of this run.** The official run predates `f5bbfcc4`. Its working tree differed only by
the unrelated qwen35-35b catalog row, which was load-time-only and was judged not to affect this run
retroactively. The release binary's own lineage is stated separately:
`f5bbfcc4` -> `594d1055` -> `66c83de3`.

| Model | Result | Conditions |
|---|---|---|
| Qwen3.5-122B-A10B (`qwen35-122b-nonextn`) | **5.59-5.69 tok/s sustained decode** - `GATE1_SERVE: PASS` (all 4 measured reps >= 5.0) | budget 8192 MB, QD 8, prefetch on (K=8, N=4), ctx 12288, 4 measured reps across 2 ON arms, warmup rep excluded |
| gpt-oss-120b (`gpt-oss-120b`) | Integrity anchor: OFF/ON A-B-B-A protocol, 12 paired responses token-ID identical; functional gates all pass | paired within one session on identical binaries |

The same official run also produced the mmap comparison arm: OFF decode **2.4106 tok/s** pooled,
individual reps **2.2842-2.5185 tok/s**. Against the two ON arms that gives the official decode
ratios **2.3226x** and **2.3439x** - that is how much faster the direct-read path served this model
than the same engine reading the same weights through mmap, under these conditions. Mandatory
co-statement, carried in the body rather than a footnote because the deviation exceeds 5 %: *in a
separate, unofficial ISLC-off mmap-only correction run (no ISLC process, effective timer resolution
1 ms) the OFF arm's measured mean was 2.7178 tok/s and its measured pooled prefill 3.5954 tok/s.
Against the official ON pooled figures, the cross-run adjusted estimates are 2.0695x for decode
(ON1/ON2: 2.0601x / 2.0790x) and 12.713x for measured prefill - respectively 11.3 % below the
official decode ratio and 8.5 % below the official measured prefill ratio. That is not a matched
A/B, it does not replace the official ratio, and it is not used in any gate.*

> **This is not "just mmap".** We measured the mmap route first, on this hardware, and rejected it
> on data: with an expert working set far larger than RAM, page-cache eviction forced repeated
> re-streaming of expert pages and decode collapsed. MoE-Direct replaces incidental caching with an
> explicit, expert-granular slot cache: aligned repacked records, explicit positional reads with
> overlapped I/O at a measured queue depth, and a deterministic LRU with lease pinning. The claim
> is scoped to what we measured on this box for these models - not a universal law about mmap.

## About the binary in this release

Earlier editions of this README had to explain an awkward gap: the official gate numbers came from
a working tree that predated the shipped zip, and the only run on the exact release binary was a
single ON arm (2026-07-31, decode 6.32 tok/s across three probes, run
`launcher_20260731T045323Z_240740`), which by our own rules could not carry a ratio.

That gap was closed for the v0.2 binary. The paired re-issue promised there landed on 2026-08-02
and is the second table above: same zip, both arms, one sitting. The 6.32 single-arm run stays in
the record as the original vicinity check, and the historical gate numbers keep their own section
with their own environment notes.

v0.2.1 is a different bundle: a revised launcher, and an engine that accepts BF16 router tensors
for the `deepseek4` family - that family's router tensors only: qwen, gpt-oss and deepseek2
routers, and every norm and bias tensor, remain F32-only, and the GEMV/TopK worker body is
byte-identical to the previous build. The rule that produced the v0.2 pair applies to it unchanged, which is
why the headline for this release is its own pair on its own binary rather than the earlier
numbers carried forward.

## What v0.2.1 adds, measured

v0.2.1 adds three things that were not in v0.2, and this section records a fourth measurement
whose switch has not shipped yet. Each is written up here with the evidence behind it, and where
the evidence stops that is said plainly rather than rounded up. One label is added
for this section: `GATE` marks a feature that has its own frozen pass/fail criteria and passed
them. A feature passing its own gate says nothing about the serving performance gate, which is a
separate thing and is reported in its own section above.

### KV cache Q8

`GATE`, and measured ahead of its own release. Quantizing the KV cache to `q8_0` buys memory back,
and it is worth nothing if it quietly degrades the model, so it went behind a measured gate
instead of behind a switch. The gate passed. The switch that would let you turn it on is v0.2.2
work and is **not** in the v0.2.1 bundle, so what follows is a published measurement rather than a
feature you can use yet.

| Statistic | `q8_0` KV against the f16 arm |
|---|---|
| Median KL divergence | **0.001501** |
| P99.9 KL divergence | **0.158971** |
| Perplexity ratio | **0.999328 +/- 0.001212** (Q 4.394242 +/- 0.096000 against base 4.397198 +/- 0.095900) |
| Same top-1 token | **97.118 +/- 0.166 %** |

Conditions: Qwen3.5-122B on the reference machine, `llama-perplexity --kl-divergence`, wikitext,
20,000 tokens across 40 completed chunks of 40, two paired arms (f16 base against `-ctk q8_0
-ctv q8_0`) identical in every other argument, model and repack, each arm a fresh process.

The thresholds are frozen at **median KLD <= 0.002 and P99.9 <= 0.20**, and they are a prospective
gate: they may not be relaxed later to admit a model that fails them. Reuse requires the same
corpus SHA, the same argument and base shape, a reported completed-chunk count, and at least 40
chunks. A second criterion covers perplexity: the one-sided 95 % upper bound on the log ratio came
to about 0.001323, well under the ln(1.01) allowance of about 0.00995.

Qwen3.5-122B is the first profile to pass, and so far the only one. On any other model KV Q8 is
**unvalidated**, and the standing policy for an unvalidated lever is opt-in with the warning
attached rather than a block: the lever is offered, the fact that its quality effect has not been
measured for that model is stated next to it, and the warning is removed for a model once its
measurement exists. That policy covers performance and quality levers only. Integrity gates,
meaning repack verification and the engine seal, are never opt-in.

### Warm start

`GATE` - the A-7 acceptance run, which passed its own frozen criteria on the reference machine.
This is not the serving release gate.
Stopping the server in v0.2 threw away the prefix cache, so the next start re-prefilled everything.
v0.2.1 saves the slot on a clean stop and restores it on the next start.

| Measurement | Result | Conditions |
|---|---|---|
| Time to first token, restored against cold | **3,238 ms against 26,148 ms, 8.1x** | Strict same-prompt cold pair on the reference machine: the same prompt served once from a restored slot and once from a genuinely cold start. |
| Output after restore | character-identical, 227 characters | Greedy decoding. The restored token count closes against the saved state. |
| Saved state across configuration changes | `n_tokens` 1,397, `n_bytes` 190,670,760 and the same KV file SHA-256 on all four arms | Four arms of the same token prefix (baseline budget 8192 MB / QD 8, budget-only 10240 MB, QD-only 7, prefetch-only K=10/N=3), bundle server started directly. Cache budget and queue depth do not change the bytes that get saved. |

What this does not do: a slot file on disk carries no server checkpoint, so a request that diverges
from the restored prefix is reprocessed in full. A clean stop overwrites that profile's saved slot
each time. On K2.6 partial reuse is structurally possible but has not been measured here.

### Budget autotune

Not a measurement, a behaviour, so no grade. v0.2 asked every machine for the same 8192 MB cache
budget, which is wrong in both directions at once: too much for a 16 GB laptop, far too little for
a 64 GB desktop. v0.2.1 computes the budget at every start unless you set one.

The machine axis is installed RAM minus a fixed 20,480 MiB reserve, clamped to between 4,096 and
12,288 MiB. The model axis comes from slot geometry rather than file size, because slots are a
uniform division of the budget: full residency needs `n_expert x MoE layers x slot stride`, and
the structural floor to start at all is `n_expert x slot stride`. The chosen value is the smaller
of the two axes, and if that lands under the profile's supported minimum the launcher stops with
`fail_resource` instead of starting something that cannot serve. The input is **installed** RAM
read from the firmware, not the OS-visible figure, because on this box those differ by 868 MiB
(32,768 against 31,900) and the difference is large enough to change the answer.

On the reference machine the four profiles that have been repacked here, of the six that ship,
converge on **auto = 12,288 MiB**, limited by the machine axis in every case. The value is never silent: a startup banner states it, and a
diagnostic record carries the probe result, all three RAM readings, the constants, the slot
geometry, both candidates and which axis bound the result. If the probe or the geometry cannot be
read, autotune switches off, falls back to that profile's own default and says why.

One honest boundary on the admission check that sits next to it. A budget larger than available
RAM is a hard stop. The rest of the contract, meaning dense weights plus KV plus server plus
headroom, is still `[unmeasured]`, so clearing the check is not an approval, it is an
`[unmeasured]` label. This is not full admission control and should not be read as one.

### Crash-recovery autosave

Save-on-stop only helps if you stop on purpose. A crash, a power cut or a forced kill lost the
whole session, so v0.2.1 also saves periodically while serving.

The tick is five minutes by default and fires only when two conditions hold together: nothing is
in flight, and the reported token count has changed since the last save. Writes alternate between two
generations, so a crash during a save damages only the generation being written and the previous
complete one survives; that alternation is the part that makes it survive a power cut rather than
merely a clean crash. On the next start the most recent eligible save is restored, whether it came
from a clean stop or from a tick. Autosave follows warm start: if warm start is off, or the run is
a smoke or reproducibility run, autosave is off too and cannot be turned on by itself.

Scale, from the save path: about 190 MB per save at a 12k-context state (1,396 tokens produced
190,646,160 bytes). The four V-2 live saves took 68-73 ms for the save API round trip, and
257-280 ms including verification and sidecar publication.

`LIVE` - At a 1-minute test cadence (shipping default: 5 minutes), the token-count gate logged
seven unchanged-count skips at 1,397 tokens, then autosaved 1,402 tokens to generation A:
190,793,760 bytes, 73 ms for the save API round trip and 263 ms including verification and
sidecar publication. A forced kill targeted the sole server PID and left the KV directory
snapshot unchanged. The next launch selected that autosave over the older 1,397-token clean-stop
state and restored the same generation, saved_at and 1,402-token count. After the post-restore
request, generation B stored 1,472 tokens while A remained intact: 192,515,760 bytes, 69 ms save
round trip and 257 ms total. A later clean stop wrote the same 1,472-token state to the separate
`slot0.kv`.

Exact-prefix reuse closed on the fourth run: the same autosave generation closed
n_saved = n_restored = cache_n = 1,410, with 7 new prompt tokens evaluated - first token in
2.15 s, versus ~27 s when the prefix diverges. A third-run continuation had omitted a
23-character segment from the saved prompt, so the stored and incoming token streams diverged
after 1,334 tokens and the hybrid model correctly reprocessed all 1,409 prompt tokens
(`cache_n=0`). Reuse requires an exact token-prefix extension; after disk restore, divergent
Qwen3.5 hybrid requests are fully reprocessed because slot files do not contain server
checkpoints. The change gate compares reported token count, not a content hash.

## Non-official observations

`PROBE` / `LIVE` - real measurements, no gate. Never mixed into the official table.

| Model | Result | Grade | Conditions |
|---|---|---|---|
| Kimi K2.6, 1T-class (`kimi-k2.6-ram-447gb`) | 1.03 tok/s decode, coherent English output, ~42 % expert-byte cache hits, zero touch/fallback events | `PROBE` | curiosity run on older staging binaries, budget 10240 MB, QD 8, prefetch off. No token-parity claim for this run. It has **not** passed the performance gate. |
| Kimi K2.6, next-layer expert prefetch | off 0.96 -> K=8/N=4 **1.04 tok/s** decode (**x1.078**); K=10/N=4 1.03 tok/s; K=12/N=7 0.95 tok/s, i.e. slower than prefetch off. All four arms produced byte-identical output. | `PROBE` | single live run `k26liveA_20260731T092057` on the reference machine, four arms, each arm started cold; decode is the mean of three 256-token probes per arm; budget 10240 MB, QD 8, ctx 8192, `-ngl 99 --n-cpu-moe 61`, so the dense layers ran on CUDA (14,886 of 16,303 MiB VRAM in use, no OOM). Single run, probe tier, not the A-B-B-A protocol - not comparable with the official numbers. |
| Kimi K2.6, direct-read vs plain mmap | ON 0.851 vs OFF 0.215 tok/s decode (**3.95x**); prefill 0.775 vs 0.232 (3.35x); the two arms produced character-identical output | `PROBE` | matched minimal pair on the reference machine: same engine binary, same short prompt, greedy, 64 generated tokens, both arms cold. ON arm: budget 10240 MB, QD 8, prefetch off. OFF arm: the stock mmap path. One short run each - honest but small, and not comparable with the official table. |
| Kimi K2.6, prefetch promotion pair | pair 1: ON 991.3 vs OFF 1058.9 ms/tok (**x1.068**); pair 2: ON 997.1 vs OFF 1052.5 ms/tok (**x1.056**). Both pairs favoured ON, and all four arms produced byte-identical output. | `PROBE` | The paired run that promoted this profile's `prefetch_state` to `validated`: run `k26liveB_20260731T213127` on the reference machine, A-B-B-A arms, prefetch ON at K=8/N=4 against OFF, budget 10240 MB, QD 8, four probes of 800 predicted tokens per arm, RAMMap before every arm, ISLC off, `-ngl 99 --n-cpu-moe 61`, adjacent-pair readout frozen before the run. This is a prefetch-relative gain and **not** an official sustained-decode pass; `performance_validated` stays false. |
| Qwen3.5-397B-A17B (`qwen35-397b`) | 1.99 tok/s sustained decode | `PROBE` | function smoke, reference machine, budget 8192 MB, QD 8, prefetch off, ctx 12288, cold (page cache pre-cleared), 96 generated tokens |
| Qwen3.5-35B-A3B (`qwen35-35b`) | 4.40 tok/s decode | `PROBE` | function smoke on the laptop, CPU-only (`-ngl 0`), budget 4096 MB, QD 2, prefetch off, ctx 4096, cold, first 48 generated tokens |
| Qwen3.5-122B, first-turn prefill | 45.6-45.9 tok/s cold long-context prefill | `PROBE` | appendix observation recorded inside the official run; deliberately **not** a headline or gate number |
| Qwen3.5-122B, multi-turn reuse | 314.9-316.5 context-tok/s perceived, from 8.1-8.3x reuse; newly evaluated tokens in the same runs: 9.3-9.6 tok/s | `PROBE` | prefix-cache A/B. The two figures must always be read together - the large number is reuse, the small one is new work. |
| DeepSeek-V4-Flash-0731 (284B, `deepseek4`) | 3.13-3.28 tok/s sustained decode across four probes, coherent output, zero touch/fallback events, 51.5 % expert-byte cache hits | `PROBE` | First direct-read run of this architecture here: reference machine, staging binary, budget 8192 MB, QD 8, prefetch off, ctx 8192, `-ngl 99 --n-cpu-moe 43`, server up in 8.07 s. In the launcher catalog since v0.2.1 with prefetch disabled - see [Supported models](README.md#supported-models). |
| Qwen3.5-122B, queue depth + prefetch in ordinary use | 4.31 tok/s (QD1, prefetch off) -> 6.60 tok/s (QD8, prefetch on); prefill 31 -> 39.1 tok/s in the same sessions | `LIVE` | one user, one third-party agent client, same workload before/after. Not comparable with the official table. |

**Size context** (full GGUF vs this machine): Qwen3.5-122B = 72.3 GiB, about **1.5x** the combined
RAM+VRAM (48 GiB) and about 2.3x system RAM. Kimi K2.6 = 416.8 GiB, about **8.7x** combined and
about **13x** system RAM.

**What the K2.6 prefetch runs showed, and what they did not.** The four-arm run is one run at probe
tier and it promoted nothing by itself; the paired run above it in the table is what moved that
profile's `prefetch_state` to `validated`, and even that says nothing about the performance gate,
which K2.6 still has not passed. Two things in the four-arm run are worth writing down anyway. The
arms differed only in when expert bytes were fetched, and their outputs came out identical byte
for byte - on the largest model here, a lever that changes speed left the text alone, which is
what the boundary in the [README](README.md#what-this-is---and-what-it-is-not) says it should do.
And more speculation was not better: the K=12/N=7 arm needed the fewest demand reads of the four
and still finished slowest of all, slower even than prefetch off, because on a drive already
saturated by demand reads the speculative reads compete for the same queue. The shallow depth is
the one that helped.

**Where the time goes.** Decode on this workload is read-bound - what paces it is how fast the
expert bytes for the next token arrive from the NVMe. That is why the levers in this release are
queue depth (measured per machine at startup) and next-layer expert prefetch (enabled only on
profiles where it was validated).

**One arithmetic warning.** "Effective read speed / miss bytes per token" is an **I/O ceiling**, not
an expected tok/s - there is a real fixed cost per token on top of it (an earlier reference
calibration put it at ~79-82 ms; it has not been re-derived for this release candidate).
The launcher therefore prints the ceiling and a reference-calibrated estimate as two separate
numbers, and shows the fixed term as `[unmeasured]` on machines other than the reference one.
Anyone quoting the ceiling as an expected speed is quoting the wrong number.

**How the numbers were measured.** All published measurements were taken on a machine dedicated to
the model at that moment: most applications closed, no other heavy programs running, background
utilities kept to a minimum, and the documented canonical settings applied. That is deliberate - it
makes them reproducible. It also means that if you run a big MoE next to a browser full of tabs and
a game, you should expect less headroom than the tables show; the RAM budget in particular assumes
the machine has room to give.

**Verification culture.** Every claim traces to an append-only run log with raw artifacts. Paired
comparisons only ever run on identical binaries within one session. Determinism is enforced -
read-sequence digests, logical cache-tick digests and per-run token hashes must match across paired
runs, or the run is void. A manual coherence check on K2.6 showed exactly why that matters: a
degenerate, repetitive output run reported inflated numbers (1.87 "tok/s", 72 % byte-hit) against
1.03 tok/s / 42 % on coherent output. We publish the honest pair. Changes are verified in layers:
maintainer verification plus AI-assisted adversarial cross-review, with the raw verdicts kept.

