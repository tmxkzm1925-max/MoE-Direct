# MoE-Direct

**Run Mixture-of-Experts models far larger than your RAM on an ordinary Windows PC - by
keeping expert weights on your NVMe SSD and fetching routed expert records on demand, with
optional speculative prefetch on validated profiles.**

The output is the same. Not almost the same - the same. With greedy decoding we ran the same
prompts through an ordinary build of the engine and through MoE-Direct, and the two answers came
out token for token identical.[^same] The weights are not touched either: every expert record in
the repacked file is checked byte for byte against your original GGUF before it is ever used.[^bytes]

[^same]: **What was compared, and under what conditions.** The paired protocol runs the same
    engine build twice within one session - once with direct-read off, once on - with greedy
    decoding (temperature 0), the same prompts, the same seed and sampling parameters, and one
    request at a time. On gpt-oss-120b the OFF/ON A-B-B-A protocol produced 12 paired responses
    whose token IDs were identical. On Qwen3.5-122B the 12 token files of the official run were
    byte-identical to the parent anchor, i.e. the run is reproducible across runs as well. The
    claim is scoped to that protocol and to the model/profile rows in
    [Supported models](#supported-models): it is a statement about greedy decoding, not about
    sampled decoding, where two runs differ by construction.
[^bytes]: **What "byte-preserving" means, precisely.** The one-time repack rewrites your expert
    tensors into a 4 KiB-aligned direct-read store. Every record written is SHA-256 compared
    against the corresponding source bytes, all records, no sampling; the report is consumed
    fail-closed by both the launcher and the engine, so a repack that did not fully verify cannot
    be served. There is no quantization step, no re-routing, no approximation, and no
    quality/space trade in this project. The general repacker test matrix covers 128-384 experts
    and 1-2 shards across four quantization layouts; separately, the shipped 512-expert, 6-shard
    Qwen3.5-397B profile has passed its model-specific format gate.

> **v0.2 is a public preview aimed at hands-on users.** It runs, it is measured, and its rough
> edges are written down rather than hidden. It is not a one-click app for casual use yet - that
> is a direction, not a promise.

**Windows will warn you.** v0.2 is an *unsigned* public preview, so SmartScreen showing
"Windows protected your PC" is the expected outcome for a new unsigned file, not a sign that
something is wrong with your download - verify the SHA-256 and decide for yourself. On managed
PCs, or with Smart App Control enabled, the file may be blocked outright with no "Run anyway"
option; Smart App Control has no per-app exception. We will never ask you to turn off Defender,
SmartScreen or Smart App Control, to add antivirus exclusions, to change your machine-wide
execution policy, or to run anything as administrator. Code signing is planned but not in v0.2,
and even a signed build does not make first-release warnings disappear immediately.

---

## Table of contents

- [Who this release is for](#who-this-release-is-for)
- [What this is - and what it is not](#what-this-is---and-what-it-is-not)
- [Before you start](#before-you-start)
- [Quick start](#quick-start)
- [What success looks like](#what-success-looks-like)
- [Supported models](#supported-models)
- [How it works](#how-it-works)
- [Measured results](#measured-results)
- [Connecting a client](#connecting-a-client)
- [Manual warm start (advanced)](#manual-warm-start-advanced)
- [Model support roadmap](#model-support-roadmap)
- [What gets written to your disk](#what-gets-written-to-your-disk)
- [Update, reset, uninstall](#update-reset-uninstall)
- [Known limitations](#known-limitations)
- [Reporting problems](#reporting-problems)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Credit and license](#credit-and-license)

---

## Who this release is for

You are comfortable downloading a GGUF from Hugging Face, you have an NVMe SSD and the patience
for a one-time repack, and you want to run MoE models your RAM says you cannot. If that is you,
this release is built for you and you should be able to follow it end to end.

If you are looking for an app that finds, downloads and chats with a model for you, this is not
that, and pretending otherwise would waste your evening. MoE-Direct never downloads model weights:
you bring your own GGUF.

## What this is - and what it is not

This is **not** a "run any model on any Windows box" project. It is specifically about **MoE
models**: using their sparse activation - only a small fraction of the weights participate in any
one token - to push the size of model you can actually use on ordinary consumer hardware as far as
it will go. Dense models gain nothing here; the gain shows up on the MoE profiles this launcher
has verified, in the conditions the [Measured results](#measured-results) tables state.

This project started from a simple refusal: today, running a large local LLM means loading *all*
of its weights into VRAM or RAM, and for most people that puts the genuinely large models out of
reach - the hardware wall decides what you are allowed to run. If a Mixture-of-Experts model only
activates a small fraction of its weights per token, then only that fraction should have to live
in memory.

**What MoE-Direct does not touch.** The change is *where* expert weights are read from - not
*what* the model computes. To prevent any misunderstanding, here is the boundary, stated once:

- **Not the math.** Attention, dense layers, activations and numeric precision compute exactly
  what the stock llama.cpp build computes. No approximation is introduced anywhere in this
  project.
- **Not the routing.** The router picks the same experts it would have picked without
  MoE-Direct. Prefetch only warms the cache for experts the router is expected to ask for -
  a wrong guess costs time, never correctness, because the actually-selected expert is always
  read and used.
- **Not the weights.** No quantization step, no pruning, no fine-tuning, no value-changing
  transform of any kind: the repacked records are byte-for-byte the source tensors, verified
  record by record.
- **Not the model's own caches.** The KV cache and the prefix cache that make later turns fast
  are stock llama.cpp. The cache this project *does* build - the budgeted expert slot cache
  described in [How it works](#how-it-works) - lives strictly underneath the weight reads: it
  decides which expert bytes are resident in RAM at a given moment, never what is computed with
  them. A cache decision can change speed; it cannot change output.
- **Not your files.** The source GGUF is opened read-only; everything this project produces
  lives in its own folders (see
  [What gets written to your disk](#what-gets-written-to-your-disk)).
- **Not the sampling.** Temperature, seeds and chat templates behave as stock llama.cpp; the
  served API is the documented subset described in
  [Connecting a client](#connecting-a-client).

That boundary is why the token-identical claim at the top of this page is even possible: the
compute graph is the stock one - only the storage path underneath the expert tensors changed.

## Before you start

**You need, before anything else:**

| | Requirement | Notes |
|---|---|---|
| OS | Windows 10 or 11, x64 | Windows only in v0.2. No Linux/macOS build exists; see [FAQ](#faq). |
| Runtime | Microsoft Visual C++ Redistributable (x64) | The engine binaries are built with MSVC. Most machines already have it; if it is missing the server cannot start (see [`fail_server_start`](#fail_server_start)). Install it from Microsoft's [Latest supported Visual C++ Redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) page. |
| Storage | An NVMe SSD | This design is I/O bound by construction. Other storage - a SATA SSD, an external drive - is **not validated and not recommended**: the launcher does not block it, it measures your drive and reports the throughput it found. |
| Model | A supported GGUF, downloaded by you | Exact repos and revisions in [Supported models](#supported-models). |
| Disk | About **2x the model size**, plus reserve | The repack writes its output next to your GGUF and the original stays. Nothing is deleted. |
| Time | **Minutes to roughly 18 minutes on the recorded machine, once per model** | Recorded there: 61 GB store in 130 s, 72.8 GB in 220 s, 436 GB in 18 min. Another drive may take longer. |
| RAM | Enough for the cache budget the launcher asks for | 8 GB budget for the 122B profile, 4 GB for the 35B profile, 10 GB for K2.6. The launcher checks this before any repack output is written. |
| Python | Not needed | A pinned CPython runtime for the repacker is included in the zip. |

## Quick start

> Prefer watching first? There is a **full unedited setup walkthrough** (single take, real time,
> with chapters): [youtu.be/I0MRTEn0G6g](https://youtu.be/I0MRTEn0G6g) - it covers everything in
> this section, including the waits you should expect.

**Step 0 - get the right file.** On the Releases page there are exactly two assets:

| Asset | What it is |
|---|---|
| `moe-direct-v0.2-win-x64.zip` | The runtime bundle. This is the one you want. |
| `SHA256SUMS.txt` | The checksum of that zip. |

> GitHub also shows an automatically generated **"Source code (zip / tar.gz)"** on every release.
> That is *not* a runnable bundle - it contains no binaries. Do not download it to run
> MoE-Direct.

**Then, in this order.** The order matters: Windows marks downloaded files, and unblocking the zip
*after* extracting does not clean up the files that were already extracted.

1. **Verify the download.** In PowerShell:
   ```powershell
   Get-FileHash .\moe-direct-v0.2-win-x64.zip -Algorithm SHA256
   ```
   Compare the result with `SHA256SUMS.txt`. If it does not match, stop and download again.
2. **Unblock the zip itself** - right-click `moe-direct-v0.2-win-x64.zip` -> Properties -> tick
   **Unblock** -> OK. (Equivalent: `Unblock-File .\moe-direct-v0.2-win-x64.zip`.)
3. **Extract with Windows "Extract All"** into a folder of your choice, for example
   `C:\moe-direct\v0.2\`. Other archivers differ in how they propagate the mark-of-the-web, so
   this is the one path we document.
4. **Put your GGUF somewhere the launcher can find it.** Any path works, but if you place models
   under `<drive>:\moe-models\` (up to three levels deep, e.g.
   `D:\moe-models\qwen3.5-122b\<repo-name>\model-00001-of-00002.gguf`) the launcher lists them for
   you and you can pick one with the arrow keys. Multi-shard models must have all their shards in
   one folder.
5. **Double-click `Start-MoeDirect.cmd`** in the extracted folder.
   *Prefer PowerShell?* `Start-MoeDirect.ps1` works too - the `.cmd` delegates to that PowerShell
   launcher, it is only a double-click entry point that starts PowerShell with a bypass scoped to
   that one process (nothing machine-wide is changed, nothing is installed). If you run the `.ps1` by
   right-click -> "Run with PowerShell", the window can close on failure before you read the error;
   run it from an open console instead. The `.cmd` keeps the window open and prints the log folder.

   ![The extracted bundle folder with Start-MoeDirect.cmd selected](docs/img/01-bundle-extracted.png)
   *The extracted folder - `Start-MoeDirect.cmd` is the double-click entry point.*

   ![The model selection menu](docs/img/02-model-menu.png)
   *The first screen after the bundle integrity check: GGUF files found under
   `<drive>:\moe-models` are listed for arrow-key selection. The menu lists whatever it finds -
   whether a file is actually supported is decided later, by the catalog and the integrity gates.*

6. **Approve the one-time repack.** The launcher identifies your model, checks RAM and disk, then
   stops and shows you the exact cost - output size, free space left afterwards, expected time -
   and no model or repack output is created until you answer. This is the long step: minutes to
   roughly 18 minutes on the recorded machine, depending on model size, with
   live progress. There is no resume in v0.2; if you cancel, the next run starts the repack from
   the beginning (it will tell you and ask before deleting the partial output).

   ![The repack plan and its confirmation prompt](docs/img/03-repack-plan.png)
   *The repack plan: exact sizes, RAM and disk preflight, and the y/N prompt - nothing is
   written until you approve.*

7. **Press Enter at the status screen** to start the server, wait for `ready`, and connect a
   client (see [Connecting a client](#connecting-a-client)).

   ![The status screen](docs/img/04-status-screen.png)
   *The status screen - gate states, the measured queue-depth sweep, and the reference numbers
   with the conditions they require.*

   ![Server ready](docs/img/05-ready.png)
   *`ready` - the server is up on its loopback URL and stays in the foreground until you stop it.*

Runs after the first skip the repack entirely: the launcher re-checks the existing output against
its integrity gate, re-measures the drive if it has to, and goes straight to the status screen.

## What success looks like

The status screen tells you, in this order, what will happen when you press Enter: the model and
profile it identified, the disk and RAM it will use, and what pressing Enter does. Queue depth,
gate state and prefetch state are shown below that, as technical detail.

**Your first conversation is usually the slowest one - by design, not by accident.** On a fresh
start the expert cache is empty and fills from the NVMe as you chat, and the first turn has to
prefill your entire prompt from that cold state: seconds for a short question, several minutes
for agent-style clients that send 15k+ token system prompts. The design already softens what it
can - the server is ready in seconds because experts are never bulk-loaded up front, and the
model is usable from the very first token. In a long-lived session that keeps a stable prompt
prefix - which is how agent-style clients behave - later turns reuse the prefix cache instead of
re-reading what was already read, and that reuse is where the multi-turn numbers in
[Measured results](#measured-results) come from; a turn that changes the prefix pays for the
changed part again. So judge the speed by the later turns of a session, not its first one - and
note that stopping the server clears both caches, so the next start begins cold again.

When the server is up, the launcher prints `ready` with the base URL. When you stop it - `stop` in
the menu, or Ctrl+C - it shuts the server down, confirms the child process and the listening port
are gone, and exits with:

```
[moe-launcher] status=ok
```

That last line is the machine-readable one, on stderr, exactly once, on every run - success or
failure. Every status value is listed in [Troubleshooting](#troubleshooting).

## Supported models

MoE-Direct does not download weights. Get the GGUF from the exact repository and revision below -
a different revision may have different tensors and will be rejected by the identity check.

Every profile carries **two independent gates**, and one never implies the other:

- **Format gate** - the repack of this model was verified byte for byte and the verify report
  passed. This is what protects your output.
- **Performance gate** - a serving run of this model passed the frozen release gate on the
  reference machine. This is only about speed.

| Tier | Meaning |
|---|---|
| `reference-validated` | Format gate **and** performance gate passed on the reference machine. |
| `format-validated` | Format gate passed. Any speed number shown is an observation, not a gate pass. |
| `experimental` | Neither gate established. Your own risk. Nothing in v0.2 ships at this tier. |

| Model (profile id) | Source | Experts | Repacked expert store | Min cache budget | Tier | Prefetch |
|---|---|---:|---:|---:|---|---|
| **Qwen3.5-122B-A10B Q4_K_M** (`qwen35-122b-nonextn`) **<- start here** | [bartowski/Qwen_Qwen3.5-122B-A10B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF) rev `fec8b222a2eddc3346d6b6d7f7c85efea93cd6bf` | 256 (top-8) | 72.8 GB | 8192 MB | `reference-validated` | `validated` (K=8, N=4) |
| gpt-oss-120b MXFP4 (`gpt-oss-120b`) | [ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) rev `8d158cefb5f175c6f8842bbd8f68eca54d951ab4` | 128 (top-4) | 61 GB | 8192 MB | `format-validated` | `reference-only` |
| Qwen3.5-35B-A3B Q4_K_M (`qwen35-35b`) | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) rev `bc014a17be43adabd7066b7a86075ff935c6a4e2` | 256 (top-8) | 19.5 GB | 4096 MB | `format-validated` | `disabled` |
| Qwen3.5-397B-A17B Q4_K_M, 6 shards (`qwen35-397b`) | [unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) rev `da33c16fa4440f831149fcf53b98a22bc07785e5` | 512 (top-10) | shown by the launcher before it writes | 8192 MB | `format-validated` | `disabled` |
| Kimi K2.6 447 GB mixed-quant (`kimi-k2.6-ram-447gb`) | [baa-ai/Kimi-K2.6-RAM-447GB-GGUF](https://huggingface.co/baa-ai/Kimi-K2.6-RAM-447GB-GGUF) rev `1e8bc2c2c759db5b4bb783965129d4e1e9182bc6` | 384 (top-8) | 436 GB | 10240 MB | `format-validated` | `reference-only` |

Notes on this table:

- **First time? Take Qwen3.5-122B.** It is the only `reference-validated` profile and the one the
  published speed number belongs to. Budget about 73 GB of disk for its expert store.
- **Prefetch column.** `validated` means the next-layer expert prefetch was measured and frozen for
  that profile on the reference machine. `reference-only` means the signal exists but the
  end-to-end lever has not been qualified, and `disabled` means the family adapter is not built
  yet. The engine and the launcher both refuse a prefetch override on a profile that is not
  `validated` - this is deliberate, not a bug. Two profiles sit at `reference-only` today,
  gpt-oss-120b and K2.6; for K2.6 the evidence is the single live run recorded under
  [Non-official observations](#non-official-observations), and until a paired run promotes it the
  launcher keeps serving that profile with prefetch off.
- **Text only.** v0.2 validates text serving. Multimodal (mmproj) inputs are **not verified** -
  not blocked, just unlabelled. Reports welcome.
- **Where these values come from.** The catalog file `models.json` in the bundle is the single
  source of truth for the profile id, the repository and revision, the expert count and top-k, the
  minimum cache budget, the tier and the prefetch state; those columns are rendered from it. The
  **Repacked expert store** column is not a catalog field: it is the size recorded when that model
  was actually repacked here - on the reference machine, except the 35B row, which was repacked on
  the second machine below. The 397B row has no recorded size yet. The launcher
  computes the exact size for *your* model and shows it in the repack plan before it writes.

## How it works

1. **One-time repack.** Your GGUF's expert tensors are rewritten once into a direct-read layout
   (`experts.bin`) - large aligned blocks instead of thousands of scattered small tensors -
   with every record verified against the original. Two terms are used throughout this document:
   the **expert store** is that one file, `experts.bin`; the **repack output** is the `repack\`
   folder that holds it together with `manifest.json` and `verify_report.json`.
2. **Budgeted expert cache.** At serve time a fixed RAM budget (you choose, e.g. 8 GB) holds the
   hottest experts in model-aware slots with a deterministic LRU and lease pinning. Everything
   else stays on disk.
3. **Direct NVMe reads.** Misses are read unbuffered - no OS page-cache double-caching - with
   overlapped positional reads at a measured queue depth, straight out of the repacked file.
4. **Startup probes.** Before the status screen the launcher measures *your* drive (a short
   read-only sweep over the repacked file at queue depths 1/2/4/8) and *your* RAM, and picks the
   defaults from the measurement instead of assuming the reference machine.
5. **Dense layers on GPU.** Attention and the other dense weights run on the CUDA backend as
   usual - on the reference machine they do, with `-ngl 99` on an RTX 5080. Only the expert
   stream lives on the CPU/NVMe path. CPU fallback behaviour on machines without CUDA is
   `[unmeasured]`.
6. **OpenAI-compatible server.** The result is a local `llama-server` endpoint bound to loopback,
   speaking the implemented OpenAI-compatible subset (see
   [Connecting a client](#connecting-a-client)).

More RAM is not a threshold you have to cross - it is cache budget and headroom, which raises the
hit rate. Whether a given amount of extra RAM pays off on *your* model is `[unmeasured]` unless it
appears in the table below.

## Measured results

Every number here belongs to a machine, a build, a configuration and a workload window. A number
without its conditions is a number you should not trust, so all of them are printed with theirs.
Results scale primarily with NVMe read throughput; treat these as data points from these boxes,
not as promises for yours.

**Grades used below**

| Grade | Meaning |
|---|---|
| `OFFICIAL` | Produced by the frozen release-gate protocol on the reference machine, from a working tree that predates the staging source tree `f5bbfcc4` - see the provenance note under [Official numbers](#official-numbers). Gate verdict stated. |
| `PROBE` | A deliberate measurement with a written protocol, but not a gate run. Not a tier promotion. |
| `LIVE` | Observed during ordinary use. Honest, but uncontrolled - never to be read alongside the `OFFICIAL` table as if it were the same kind of number. |

### Reference machine

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 7800X3D (8C/16T) |
| GPU | NVIDIA RTX 5080, 16 GB VRAM |
| RAM | 32 GB DDR5-6000 (2x16 GB) |
| Model NVMe | KIOXIA EXCERIA PLUS G4 2 TB, PCIe Gen5. Two different numbers, kept apart on purpose: about **5.4 GB/s** was the effective read rate observed while serving K2.6, and **6.5-8.1 GB/s** is what `fio` reports for large-block reads on this drive - a device ceiling, not a serving expectation. |
| OS | Windows 11 |

Second machine used for the 35B row: Acer Aspire Lite 16, Core i7-1355U, 16 GB DDR5-4800,
HOGE H8A1 512 GB NVMe, Windows 11, CPU-only (`-ngl 0`).

### Official numbers

`OFFICIAL` - reference machine, frozen constants K=8 / N=4 / QD=8.

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

### About the binary in this release

The official numbers above were produced from an earlier working tree, not from the binary in this
zip. On 2026-07-31 the engine shipped in `moe-direct-v0.2-win-x64.zip` was checked against the
anchor band by a **single-arm confirmation run** (ON only), driven end to end by the bundled
launcher on the reference machine: verify-gated repack (36,864/36,864 pairs), sweep-selected QD 8,
prefetch on with the catalog defaults, cold start (standby list cleared). The run used the bundle
at zip SHA-256 `4f1c4f16...658b2f`; the published zip is `34e1713b...f28e`, and the only
difference between the two is a launcher-script revision made after the run (display-only prefill
progress tags, locked by the launcher selftest) - every engine file is byte-identical. Across the three 256-token probes, decode averaged **6.32 tok/s**
(6.08 / 6.81 / 6.07), at or above the official anchor band **5.59-5.69 tok/s**; cache integrity was
clean (0 fallback, 0 touch events). The probe prompts are shorter than the official workload, so
this is a vicinity check, not an equivalence claim - it shows the shipped binary is not degraded
relative to the anchor. Run ID `launcher_20260731T045323Z_240740` (metrics
`metrics_20260731T135350_240740.jsonl`). This was **not** an official-tier measurement, and it
supports no multiplier claim against an OFF arm.

A full paired re-issue - ON/OFF, A-B-B-A, on the exact release binary - is scheduled as a
short post-release update. When it lands, the numbers in this README are replaced by that run's
numbers and the change is noted in the release notes.

### Non-official observations

`PROBE` / `LIVE` - real measurements, no gate. Never mixed into the official table.

| Model | Result | Grade | Conditions |
|---|---|---|---|
| Kimi K2.6, 1T-class (`kimi-k2.6-ram-447gb`) | 1.03 tok/s decode, coherent English output, ~42 % expert-byte cache hits, zero touch/fallback events | `PROBE` | curiosity run on older staging binaries, budget 10240 MB, QD 8, prefetch off. No token-parity claim for this run. It has **not** passed the performance gate. |
| Kimi K2.6, next-layer expert prefetch | off 0.96 -> K=8/N=4 **1.04 tok/s** decode (**x1.078**); K=10/N=4 1.03 tok/s; K=12/N=7 0.95 tok/s, i.e. slower than prefetch off. All four arms produced byte-identical output. | `PROBE` | single live run `k26liveA_20260731T092057` on the reference machine, four arms, each arm started cold; decode is the mean of three 256-token probes per arm; budget 10240 MB, QD 8, ctx 8192, `-ngl 99 --n-cpu-moe 61`, so the dense layers ran on CUDA (14,886 of 16,303 MiB VRAM in use, no OOM). Single run, probe tier, not the A-B-B-A protocol - not comparable with the official numbers. |
| Kimi K2.6, direct-read vs plain mmap | ON 0.851 vs OFF 0.215 tok/s decode (**3.95x**); prefill 0.775 vs 0.232 (3.35x); the two arms produced character-identical output | `PROBE` | matched minimal pair on the reference machine: same engine binary, same short prompt, greedy, 64 generated tokens, both arms cold. ON arm: budget 10240 MB, QD 8, prefetch off. OFF arm: the stock mmap path. One short run each - honest but small, and not comparable with the official table. |
| Qwen3.5-397B-A17B (`qwen35-397b`) | 1.99 tok/s sustained decode | `PROBE` | function smoke, reference machine, budget 8192 MB, QD 8, prefetch off, ctx 12288, cold (page cache pre-cleared), 96 generated tokens |
| Qwen3.5-35B-A3B (`qwen35-35b`) | 4.40 tok/s decode | `PROBE` | function smoke on the laptop, CPU-only (`-ngl 0`), budget 4096 MB, QD 2, prefetch off, ctx 4096, cold, first 48 generated tokens |
| Qwen3.5-122B, first-turn prefill | 45.6-45.9 tok/s cold long-context prefill | `PROBE` | appendix observation recorded inside the official run; deliberately **not** a headline or gate number |
| Qwen3.5-122B, multi-turn reuse | 314.9-316.5 context-tok/s perceived, from 8.1-8.3x reuse; newly evaluated tokens in the same runs: 9.3-9.6 tok/s | `PROBE` | prefix-cache A/B. The two figures must always be read together - the large number is reuse, the small one is new work. |
| Qwen3.5-122B, queue depth + prefetch in ordinary use | 4.31 tok/s (QD1, prefetch off) -> 6.60 tok/s (QD8, prefetch on); prefill 31 -> 39.1 tok/s in the same sessions | `LIVE` | one user, one third-party agent client, same workload before/after. Not comparable with the official table. |

**Size context** (full GGUF vs this machine): Qwen3.5-122B = 72.3 GiB, about **1.5x** the combined
RAM+VRAM (48 GiB) and about 2.3x system RAM. Kimi K2.6 = 416.8 GiB, about **8.7x** combined and
about **13x** system RAM.

**What the K2.6 prefetch run showed, and what it did not.** It is one run at probe tier, and it
promotes nothing: K2.6 stays `reference-only` and the launcher still serves it with prefetch off.
Two things in it are worth writing down anyway. The four arms differed only in when expert bytes
were fetched, and their outputs came out identical byte for byte - on the largest model here, a
lever that changes speed left the text alone, which is what the boundary above says it should do.
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

## Connecting a client

When the launcher says `ready` you have an endpoint speaking the documented OpenAI-compatible
subset on `http://127.0.0.1:<port>/v1`, bound to loopback only. No API key is required locally;
clients that insist on one will accept any string.

```bash
curl http://127.0.0.1:8093/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8093/v1", api_key="none")
r = client.chat.completions.create(model="local",
        messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

Three settings cover most chat apps: **base URL** (`http://127.0.0.1:8093/v1`), **API key** (any
non-empty string), **model name** (whatever `/v1/models` returns).

**Read this before you file a compatibility issue:**

- This serves the **implemented OpenAI-compatible subset** - the common chat-completions surface,
  streaming, cancel and `/v1/models`. It is not full OpenAI API compatibility.
- **One request at a time** (`-np 1`). A second concurrent request waits. Restarting the server
  loses the queue.
- Long system prompts are expensive here: the first turn pays full prefill. Agent-style clients
  that ship 15k+ token system prompts will feel that on turn one, and much less afterwards, because
  the prefix cache absorbs the repeat.

**Clients verified on the v0.2 staging stack:** Hermes Agent (Qwen3.5-122B profile, context raised
to 65536 through the launcher's custom path; also driven against the K2.6 1T-class profile). Notes
from those sessions: agent-style clients tend to
require a large minimum context; "thinking" mode works, but the thinking tokens are generated at
disk-tier decode speed before the answer starts, so it carries an extra time cost on top of every
reply - whether that trade is worth it is your call, per task; and if
the app caches model metadata you may need to re-register the endpoint after a restart. Other
clients may work if they use the documented subset - they are simply not on this list until someone
has run them.

**Raise your client's timeouts before a first turn on a very large model.** During prefill the
server streams nothing back, and on the largest profile that silence is long. `LIVE` observation,
reference machine, K2.6 profile (8 GiB budget, QD 8, prefetch on K8/N4, ctx 65536,
`--no-kv-offload`): a ~15.7k-token first prompt tripped the client's default stale-stream watchdog
(900 s) repeatedly. The first attempt was interrupted three times by the watchdog, then ended in
an app error that needed a restart; the retried attempt after the restart completed in **about 44 minutes end
to end**, absorbing two more watchdog aborts on the way. Every abort resumed from the server's
prompt cache, so no server work was lost - the cost was client-side friction, not recomputation.
If your client exposes timeout settings, raise anything that watches for "no bytes received"
**above your expected first-turn time - 3600 s is a reasonable floor for 1T-class, 7200 s gives
comfortable margin**. In Hermes Agent's config the relevant keys are
`agent.local_stream_stale_timeout` (default 900) and `agent.gateway_timeout` (default 1800) - the
session above ran on those defaults, which is exactly how it collected the aborts; raising both is
our recommendation, not something the app does for you. `agent.api_max_retries: 3` can stay, since
aborted attempts resume from the prompt cache. This is a first-turn cost only: in the same session
the second turn entered generation in under a minute, because the prefix cache absorbed the
repeated system prompt.

## Manual warm start (advanced)

The first long prompt on a big model is expensive - on K2.6, the largest and slowest profile here, a
roughly 16k-token prompt took about an hour on the reference machine, an ordinary-use observation
rather than a measured protocol - and stopping the server throws that work away, because both caches
are cleared. llama.cpp upstream can save a slot's state to a file and restore it later, and this
project changes nothing about that path, so you can use it yourself today. It is a power-user route
with real sharp edges, listed below; the automated version with identity checks is v0.2.1 work (see
[Model support roadmap](#model-support-roadmap)).

**It only works when you start the server yourself**, without the launcher - the launcher fixes the
server arguments, so there is no way in through it. Add `--slot-save-path <directory>` to the server
arguments, and **create that directory first**: if it does not exist, the server refuses to start
while it is still parsing its arguments.

**Save, before you stop the server.** The slot id is a path parameter, and it is `0` because this
project serves one request at a time from a single slot:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8093/slots/0?action=save" -ContentType "application/json" -Body '{"filename":"my_session.kv"}'
```

**Restore, after you start it again.** Same shape, `action=restore`:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8093/slots/0?action=restore" -ContentType "application/json" -Body '{"filename":"my_session.kv"}'
```

**HTTP 200 is not the same as success.** Read the response body: for a save, require
`n_saved > 0` and `n_written > 0`; for a restore, require a well-formed response with
`n_restored > 0` and `n_read > 0`. If a restore is not confirmed, `POST /slots/0?action=erase`
and continue cold only after the erase succeeds - otherwise restart the server without
restoring. And restored counts alone do not prove the prompt was reused: on hybrid-attention
models the engine may still re-process the full prompt, so compare `timings.cache_n` against
`timings.prompt_n` on your next request, and check its time-to-first-token against a cold one.

Before you rely on it:

- **The engine does not check that the file matches the model.** A slot file carries no model hash,
  no vocabulary, no RoPE settings, no engine identity and no checksum; the engine validates its
  format and only part of its structure, not its identity. Restore only into **exactly the same
  model file, the same engine build/bundle, the same context size and slot layout, and the same
  state-relevant server configuration** you saved from. Anything else is undefined behaviour, not
  an error message.
- **The file contains your conversation tokens verbatim.** Treat it as you would the conversation
  itself - it is local, it is yours, and deleting it is the way to get rid of it.
- **It grows with context.** How large it gets on a given model and context size is `[unmeasured]`
  here.
- **This manual route is outside our verification matrix.** It is upstream functionality we are
  pointing at, not a feature v0.2 validates. The version with identity binding, automatic
  save/restore and cleanup is what v0.2.1 is for.

## Model support roadmap

This is an actively developed project, and v0.2 is a preview, not a finished product. The table
below is the working queue, not a wish list - the v0.2.1 items are already specified, two of
them as frozen, reviewed designs.

| Where we are | Status |
|---|---|
| Serving a 1T-class MoE (Kimi K2.6, deepseek2 architecture) from a 32 GB machine | **Done and shown** - unedited single-take demo below. Format gate passed; performance gate not passed (1.03 tok/s, honestly labelled `PROBE`). |
| Prefetch for the deepseek2 family, which is what Kimi K2.6 needs | **Adapter landed, measured once, not promoted.** The signal adapter for that family exists now, and a first live four-arm run on K2.6 selected K=8 / N=4 - the numbers are under [Non-official observations](#non-official-observations). One run is evidence, not a gate: `prefetch_state` for that profile is `reference-only`, the launcher serves it with prefetch off, and overrides stay refused on both sides. The paired A-B-B-A run that can promote it to `validated` - after which the launcher enables prefetch for it by itself - is **v0.2.1** work. |
| Warm start: saving and restoring slot state across restarts | **Specification frozen; the implementation is the next release.** Stopping the server today clears both the prefix cache and the expert slot cache, so the next start begins cold. The design that removes that cost is written and reviewed, but none of it is in the v0.2 bundle - it ships, if it passes its own verification, in **v0.2.1**. The manual upstream route you can drive by hand today is described in [Manual warm start (advanced)](#manual-warm-start-advanced). |
| Expert-cache warmer | **Targeted at v0.2.1.** Filling expert slots ahead of the first turn instead of letting them fill as you chat. Not in this build. |
| Kimi K3 (2.8T class) | **Top of the queue the moment llama.cpp upstream supports the architecture.** Its architecture is outside the base commit this release is built on, so support is gated on upstream, and the timing is upstream's, not ours. No promise is made about when. |
| Wider hardware, wider OS | Windows only today. OS expansion is genuinely hard in the current test environment and will take time. Community ports are welcome. |

Why K3 is worth naming at all: at 2.8T-class sizes nothing that resembles a consumer machine can
hold the weights in memory, so keeping experts on disk and streaming the ones each token actually
uses is the *shape* of local execution we have found practical on the hardware this project
targets. Its expert format (MXFP4) is already one
the repacker handles today. That is a structural argument about the size class, not a claim that
this project will be first or fastest there.

**Demo - unedited single take:**
[![MoE-Direct demo](https://img.youtube.com/vi/JDfrWMxwczk/maxresdefault.jpg)](https://youtu.be/JDfrWMxwczk)
[Watch on YouTube (2:49)](https://youtu.be/JDfrWMxwczk) - cold boot to answer, no cuts: the 447 GB
GGUF on disk, server load in ~19 s, live token streaming while Task Manager shows the NVMe
sustaining multi-GB/s reads with system RAM staying under 32 GB. A second segment repeats the run
on Qwen3.5-122B.

**Setup walkthrough - unedited single take:**
[Watch on YouTube](https://youtu.be/I0MRTEn0G6g) - the full v0.2 first-run experience in real
time, chapters included: download, the one-time repack, server start, connecting a chat client
(Hermes Agent), a cold first prompt with the launcher's live prefill progress lines, and the
fast second turn out of the prefix cache. Real waits are left in on purpose - skip with the
chapters.

## What gets written to your disk

Transparency about files is cheap and surprises are expensive, so here is the complete list.
Everything below is local. Nothing is uploaded, nothing touches the registry, and every file is
either plain text (JSON/JSONL/log) you can open and read, or an output you explicitly approved.
Launcher state and logs are written from the moment the launcher starts; **no model or repack
output is created before you consent to the plan** (the one exception in the repack folder is a
tiny lock file, listed below, that marks the folder as claimed - it contains no model data).

| File | Written when | What it is for |
|---|---|---|
| the bundle folder | **never** | The bundle is read-only by design. The launcher verifies every file in it against the internal SHA manifest on each start and refuses to run on a mismatch. Its own state goes below - never into the bundle. |
| `%LOCALAPPDATA%\MoE-Direct\presets.user.json` | when you save a configuration | Saved launcher configurations, offered on the next run. |
| `%LOCALAPPDATA%\MoE-Direct\probe.state.json` | after the startup queue-depth sweep | Cached drive-sweep measurements per model/profile/volume, so the sweep does not rerun on every start. Re-measured when any of those change. |
| `%LOCALAPPDATA%\MoE-Direct\probe.scratch.json` | first run, right after you approve the repack | The provisional read-speed reading from the pre-repack sanity probe, bound to the same model/profile/volume. Diagnostic only since the startup sweep took over the queue-depth decision. |
| `.moe-probe.tmp` in the repack output folder | same probe, and only while no `experts.bin` exists there yet | A 64 MiB scratch file, written so the probe measures the volume the expert store will actually be read from, and **deleted as soon as the measurement ends**. |
| `.moe-launcher.lock` in the repack output folder | when the launcher takes its locks - before the repack plan is shown | The instance lock that stops two launchers from writing the same output folder. A few bytes, no model data; the file itself remains after exit (the lock is released) and is harmless. |
| `probe.scratch.json.tmp` / `probe.state.json.tmp` / `presets.user.json.tmp` in `%LOCALAPPDATA%\MoE-Direct\` | whenever the matching state file is updated | Scratch files for atomic replacement - written first, then renamed over the real file. Removed on success; one may survive a crash, and deleting it is always safe. |
| `%LOCALAPPDATA%\MoE-Direct\recent_models.json` | after model selection | The recent-models list behind the arrow-key picker. |
| `%LOCALAPPDATA%\MoE-Direct\logs\launcher_<timestamp>_<pid>.jsonl` | every run | The launcher's decision timeline: preflight, probe results, applied arguments, gate decisions, child start, teardown. Records local file paths, which usually include your Windows user name - skim before sharing. |
| `%LOCALAPPDATA%\MoE-Direct\logs\server_<timestamp>_err.log` / `_out.log` | every server start | The engine's own output. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_log.jsonl` | every repack | Append-only record of repack attempts and their outcomes. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_plan_<timestamp>_out.log` / `_err.log` | when the repack plan is computed - **before** you approve anything | The repacker's own output while it is only costing the job out. This is a log, not repack output: no model bytes are written at this step. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_<timestamp>_out.log` / `_err.log` | every repack you approve | The repacker's own output during the repack itself, including the per-layer progress lines. |
| `%LOCALAPPDATA%\MoE-Direct\logs\metrics_<timestamp>.jsonl` | every serving session | Cache accounting: hit/miss counts, bytes read, wait times, slot and queue-depth configuration. **No prompts, no responses, no token content** - open it in a text editor and check. A few hundred KB per session. |
| `repack\` next to your GGUF (`experts.bin`, `manifest.json`, `verify_report.json`) | once per model, after you approve the plan | The direct-read expert store, its identity manifest, and the byte-level verification report that the serving gates consume - the repack output. Tens to hundreds of GB. An interrupted repack leaves `experts.bin.partial` here as well. **Never deleted automatically** - see [Update, reset, uninstall](#update-reset-uninstall). |

That is the whole list for the launcher flow. If you ever catch this project writing anywhere
else, that is a bug - please report it. (One advanced exception you create yourself: if you use
[Manual warm start](#manual-warm-start-advanced) with a self-started server, the slot files you
request are written to the directory you point `--slot-save-path` at.)

## Update, reset, uninstall

There is no installer, and nothing is written to the registry. Three things have separate
lifetimes, and you control all three:

| Thing | Where | What to do |
|---|---|---|
| The bundle | wherever you extracted the zip | **To update:** extract the new version into a *new* folder. Do not overwrite an existing bundle - the integrity manifest covers the folder as a set. Delete the old folder when you are done. |
| Launcher state and logs | `%LOCALAPPDATA%\MoE-Direct\` | **To reset settings:** delete `presets.user.json` (saved configurations), `probe.state.json` and `probe.scratch.json` (measured drive results), `recent_models.json` (the recent list). They are all rebuilt on the next run. Logs live in the `logs\` subfolder. |
| Repack output | a `repack\` folder next to each of your GGUFs | **Not deleted automatically, ever.** This is the big one - tens to hundreds of GB. Delete the `repack\` folder to reclaim the space; the next run for that model will repack from scratch. |

Uninstalling completely = delete all three. Your GGUFs are yours and are never touched.

## Known limitations

- **MoE models only.** Dense models get no benefit from this design.
- **First-run repack cost is real** - minutes to roughly 18 minutes on the recorded machine, and
  roughly the model's size again on disk. There is no resume in v0.2: an interrupted repack restarts
  from the beginning.
- **Windows only.** No Linux or macOS build, and no promised numbers for them.
- **Measured on NVIDIA CUDA only, and this zip carries the CUDA runtime only.** The expert stream -
  the direct reads and the slot cache - runs on the CPU and the NVMe, so it does not depend on
  which GPU you have; what uses the GPU is the dense-layer offload, and that is stock llama.cpp
  with its dynamic backend loading. Other backends (ROCm/HIP, unified-memory platforms) are
  therefore expected to be structurally compatible, but they are **untested** here and no build for
  them ships in v0.2. CPU-only serving (`-ngl 0`) needs no GPU at all - that is how the 35B row in
  the tables above was measured. If you run this on other hardware, the performance-report issue
  form is where we would like to see the result.
- **Unsigned preview build.** Expect SmartScreen friction; some managed machines will refuse it.
- **One request at a time** (`-np 1`), and the queue is lost on restart.
- **Prefetch only on validated profiles.** Of the five shipped profiles one is `validated`, two are
  `reference-only` and two are `disabled`; only the `validated` one serves with prefetch on, and an
  override is refused on the other four.
- **Long-context prefill on a disk tier is slow.** Multi-turn use is where it becomes comfortable,
  because the prefix cache absorbs the repeat - see the measured numbers above.
- **K2.6 has not passed the performance gate.** It runs, it is coherent, it is honest about 1.03
  tok/s. Do not plan interactive work around it.
- **Text only.** Multimodal inputs are unverified.
- **No telemetry, and no automatic uploads.** Diagnostic logs are written locally and go nowhere
  unless you attach them to an issue yourself.

## Reporting problems

Every run ends with one machine-readable line on stderr - `[moe-launcher] status=<enum>` - and, for
failures, two human-readable lines just above it saying what happened and which section of this
document to read.

Diagnostic files, all local:

| File | Contents |
|---|---|
| `%LOCALAPPDATA%\MoE-Direct\logs\launcher_<timestamp>_<pid>.jsonl` | The launcher's own timeline: preflight, probe results, applied arguments, gate decisions, child start, teardown. **This is the file to attach.** |
| `%LOCALAPPDATA%\MoE-Direct\logs\server_<timestamp>_err.log` / `_out.log` | The server's own output. Attach when the failure is `fail_server_start`, `fail_runtime_exit` or `fail_gate_engine_seal`. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_log.jsonl` | The repacker's record. Attach for repack failures. |
| `%LOCALAPPDATA%\MoE-Direct\logs\metrics_<timestamp>.jsonl` | Cache accounting for each serving session: hit/miss counts, bytes read, wait times, slot and queue-depth configuration. **No prompts, no responses, no token content - you can open it in a text editor and check.** This is the file that makes a performance report (issue form 3) actionable. A few hundred KB per session; safe to delete like any other log. |

Four issue forms exist so that the first reply is not "please send more information":

1. **Blocked before it started** - no console window, no status line, SmartScreen/Smart App
   Control/antivirus intervened. There is no log to attach in this case, which is exactly why it
   has its own form.
2. **The launcher printed a status and stopped** - pick the status from the list, attach the
   launcher JSONL.
3. **Performance report** - your hardware, the launcher's probe/sweep output and its estimate, and
   what you actually got. These build the community measurement set.
4. **Anything else**, including a request to support a GGUF that is not in the catalog.

GitHub's upload box does not accept `.jsonl` directly - put the logs in a `.zip` first. Please
skim the JSONL before attaching: it records local file paths, which usually include your Windows
user name.

**Security issues do not go in public issues** - use the private route described in `SECURITY.md`.

## Troubleshooting

Each status below is a heading, so the launcher's `see : README.md > Troubleshooting > <status>`
line corresponds to `README.md#<status>`.

Exit codes, for scripting: `0` clean stop, `2` you cancelled, `3` path/resource/repack preparation
failed, `4` a policy gate refused, `5` server start, runtime or shutdown failure, `6` smoke check
failed with a clean shutdown.

### ok

Exit `0`. The server ran and was shut down cleanly: the child process exited on its own signal, no
force-kill was needed, and the listening port is gone. Nothing to do.

### ok_smoke

Exit `0`. The `-Smoke` self-check passed end to end and the server was shut down cleanly. Nothing
to do.

### cancelled_user

Exit `2`. You cancelled before the server became ready. Two cases, and they leave different things
behind:

- **Cancelled at a prompt** - you chose `stop`, pressed Ctrl+C, or declined the repack plan or the
  deletion of a leftover partial repack. **Nothing was started and nothing was deleted.**
- **Cancelled while the repacker was running** - the same status, but the repack was already under
  way, so the interrupted output stays on disk as a `.partial` file. There is no resume: the next
  run detects the leftover, tells you, and asks before deleting it and starting over.

Not an error either way; run it again when you are ready.

### fail_model_path

Exit `3`. *The model path could not be used: the file is missing, the GGUF is not one of the
supported profiles, or the shard set is incomplete/ambiguous.*

What to do:

- Check the path, including that the file finished downloading.
- For a multi-shard model, all shards must be in the same folder and be one complete set - a
  partial download or two different quantizations mixed in one folder both land here.
- If your model genuinely is not in [Supported models](#supported-models), that is expected: the
  launcher refuses to write hundreds of GB for a layout it has never verified. Use issue form 4 to
  request the profile, and include the model's repo, revision and quantization.

### fail_resource

Exit `3`. *Preflight stopped the run: not enough RAM or disk space for this configuration.*

The message states which one and by how much. Free the space (the repack needs roughly the model's
size again, plus reserve) or lower the cache budget through the custom path - but note the profile's
minimum budget, below which the model cannot be served at all. Repacking onto the OS volume when it
is nearly full is refused on purpose.

### fail_instance_lock

Exit `3`. *Another launcher instance holds the single-instance mutex, or a profile/output/port lock
is already taken.*

This status is only about lock acquisition - the launcher-wide mutex, or the exclusive lock on this
profile / output folder / port combination. Close the other MoE-Direct window. If none is open, a
previous run may still be exiting - wait a few seconds and retry. (A server left listening on the
port from an earlier crash is a different failure: it lands in
[`fail_server_start`](#fail_server_start).)

### fail_partial_cleanup

Exit `3`. *The leftover repack outputs could not be deleted or confirmed absent.*

A previous repack was interrupted, you approved the cleanup, and the deletion failed. Usually
something else holds the files open (antivirus scan, an Explorer preview, another launcher). Close
those, or delete the `repack\` folder next to your GGUF by hand, then run again. The launcher never
deletes these without asking, and never resumes a partial repack.

### fail_repack

Exit `3`. *The repacker exited abnormally, or produced no verify report.*

Attach both `launcher_*.jsonl` and `repack_log.jsonl` (issue form 2). Common causes worth checking
first: the drive filled up mid-write, or the source GGUF is corrupt - re-verify the download's
checksum against the Hugging Face repo.

### fail_custom_args

Exit `3`. *A custom value failed the type or bounds check in non-interactive mode.*

Only reachable when the launcher is driven with arguments. The message names the offending value
and its allowed range. In interactive mode a bad value simply re-prompts instead of exiting.

### fail_gate_bundle

Exit `4`. *Bundle integrity check failed: the manifest, the schema, or the file set did not match
the sealed bundle.*

The extracted folder is not byte-identical to the released one. In order of likelihood:

- The zip was extracted over an existing folder, or files from two versions were mixed. Extract the
  release into a **new, empty** folder.
- Something was added, edited or removed inside the bundle folder - including files written there
  by another tool. Keep the bundle read-only in practice; the launcher writes its own state to
  `%LOCALAPPDATA%\MoE-Direct\`, never into the bundle.
- The download is damaged. Re-verify the zip's SHA-256 against `SHA256SUMS.txt`.

### fail_gate_catalog

Exit `4`. *`models.json` failed the catalog schema, the prefetch-state check, or the expect-digest
check.*

The catalog inside the bundle is not the released one. Re-extract from a fresh download. If you
edited `models.json` yourself, restore it - hand-edited catalogs are rejected by design, because
they are how a model would end up served under someone else's verified identity.

### fail_gate_verify

Exit `4`. *The 7-item repack gate rejected the verify report or its binding to the manifest.*

This is the gate that stands between you and serving unverified weights, so it fails closed on
anything it cannot positively confirm - a partial file present, a report that does not say `pass`,
counts that disagree, a non-empty problem list, an identity mismatch, or an unreadable file.

What to do: delete the `repack\` folder next to your GGUF and repack (the launcher will offer
this). If it fails a second time on the same model, that is worth an issue - attach both the
launcher JSONL and `verify_report.json`.

### fail_gate_engine_seal

Exit `4`. *The engine refused to start and printed its policy-gate reject line.*

The launcher's checks passed but the engine's independent checks did not - the two sides are
deliberately not allowed to trust each other. Typical cause: an argument or environment variable
that enables a lever on a profile it is not validated for (prefetch on a non-`validated` profile is
the usual one). If you set `MOE_DIRECT_*` variables yourself, clear them and retry with defaults.
Attach `server_*_err.log`.

### fail_server_start

Exit `5`. *The server never reached ready: process spawn, port, listener PID, health check, an
early exit, or CUDA out-of-memory.*

Check, in this order:

1. **`server_*_err.log`** - it usually says exactly what happened.
2. **Missing MSVC runtime.** A process that dies immediately with no useful output is the classic
   symptom; install the Microsoft Visual C++ Redistributable (x64) from Microsoft's
   [Latest supported Visual C++ Redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   page.
3. **Port already in use.** Change the port through the custom path.
4. **CUDA out of memory** - another program is holding VRAM (a game, another inference server, a
   browser doing GPU work). VRAM is checked and reported but not gated, because it can change
   between the check and the allocation.

### fail_runtime_exit

Exit `5`. *The server exited unexpectedly after it had reached ready.*

The interesting evidence is in `server_*_err.log` and the last requests you sent. Out-of-memory
under load, and a crash on an unusual request, both land here. Please report it with the log and,
if you can, the request that preceded it.

### fail_teardown

Exit `5`. *Shutdown did not complete cleanly: the signal, the grace period, or a surviving child
process or listener.*

This status takes priority over every other failure in the same run, on purpose: a run that left a
process behind is not a clean run, whatever else happened. **Check Task Manager for a surviving
`llama-server.exe` and end it** before starting again, or the next run will find the port still
taken and stop at [`fail_server_start`](#fail_server_start). Worth reporting - it means the ordinary
shutdown path did not work on your machine.

### fail_smoke

Exit `6`. *A smoke assertion failed, while the shutdown itself completed cleanly.*

Only reachable with `-Smoke`. The failing assertion is named in the output; attach the launcher
JSONL.

## FAQ

**Do you redistribute model weights?** Never. You bring your own GGUF and the repacker runs locally
on your copy. Qwen, Kimi, gpt-oss and all other model names belong to their respective owners; this
project is affiliated with none of them.

**Why PowerShell?** Because you can read it. The launcher does consequential things - it writes
hundreds of GB, spawns a server, enforces gates - and every one of those decisions is visible in
`Start-MoeDirect.ps1` in the bundle. That is the reason, not any interaction with Windows security
prompts, which a script does not avoid.

**Can I run the engine directly instead of using the launcher?** Yes, and it is not blocked. Be
clear about what changes: the engine itself still enforces model/manifest identity, the verify-pass
requirement and its own seal, whatever way it is started. Everything else - bundle manifest
checking, catalog identification, RAM/disk/SSD sizing, the repack cost confirmation, `[unmeasured]`
labelling, single-request and loopback binding, and the diagnostic log - is the launcher's, and on
the direct path it is your responsibility.

**Windows-only?** Windows is the reference testbed (overlapped unbuffered I/O, a VirtualAlloc slot
pool, a vectored exception guard). The I/O layer is abstracted, but no Linux or macOS numbers are
promised until they exist.

**AI involvement?** This project is built in an AI-assisted workflow - design, implementation and
review loops involving multiple AI systems, under human direction - with every change gated by the
verification protocol above and the raw verdicts kept. Posts to upstream projects are written by
the author personally. Parts of this document were written with AI assistance.

## Credit and license

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), base commit `0bd0ec6` (b10057) -
upstream copyright and license preserved; source releases keep all upstream notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

MoE-Direct additions (c) 2026 tmxkzm1925-max, released under the [MIT License](LICENSE). The
MoE-Direct name identifies this project and its official builds - see
[TRADEMARKS.md](TRADEMARKS.md). If you use this work, please cite it
([CITATION.cff](CITATION.cff)).
