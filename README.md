# MoE-Direct

**MoE-Direct runs Mixture-of-Experts models that are far larger than your RAM, by keeping the
expert weights on an NVMe SSD and reading only the experts each token actually routes to.** Kimi
K2.6 is a 1T-class model with 416.8 GiB of weights, roughly thirteen times the 32 GiB of RAM in
the desktop this project was built on, and the ordinary answer to that arithmetic is that you do
not get to run it. You do: the server is ready in seconds, because experts are never bulk-loaded
up front, and it answers in coherent English at 1.03 tok/s, which is slow enough that we call it a
probe rather than a product. The model this release is actually tuned for, Qwen3.5-122B, runs on
the same box at the speed in the first row below, which comes from a paired run on the v0.2.1
release binary; the pair has not been re-run on this zip's binary, and the engine delta between
the two is described in [TECHNICAL.md](TECHNICAL.md#what-v022-adds). The frozen release gate,
measured earlier on a different working tree, recorded 5.59-5.69 tok/s and passed. On gpt-oss-120b, the one profile where we ran the paired
output-parity comparison, greedy decoding through the direct-read path returned token for token the same output
as the same engine binary's plain-mmap path; on Qwen3.5-122B the separate check is run-to-run
reproducibility, with the official run's outputs byte-identical across runs. Every expert byte is
verified against your original GGUF before it is ever used.

| What was measured | Result | The conditions it belongs to |
|---|---|---|
| Qwen3.5-122B decode | **5.65-6.26 tok/s per probe, arm average 5.96** | `PROBE` grade. Paired run on the v0.2.1 release binary (not re-run on this zip's binary), reference machine (32 GB RAM, one RTX 5080, a Gen5 NVMe). The bundle launcher in its `-Repro` release configuration: warm start and autosave off, budget autotune off with the budget fixed at 8192 MB, ctx 12288, QD 8, prefetch K=8/N=4, greedy. A normal product start on this machine sizes the cache at 12,288 MiB instead. |
| The same bundle's server started directly on plain mmap | **2.13-3.21 tok/s per probe, arm average 2.66**. Across the two matched sub-pairs (ON arm averages 5.96 and 5.86 vs OFF 2.66 and 2.48), the direct-read configuration ran a combined **2.3x** (2.2983x) faster | `PROBE` grade, same pair run. Same binary, same weights, same prompts, arms in A-B-B-A order in one sitting. Both arms ran at ctx 12288, so the two arms differ only in the read path - a single-variable pair. |
| Kimi K2.6, 1T class, out of 32 GB of RAM | 1.03 tok/s decode | `PROBE` grade, not a gate pass: budget 10240 MB, QD 8, prefetch off, coherent English output, measured on older staging binaries. |
| Output through direct-read against the same binary's plain-mmap path | token for token identical on gpt-oss-120b[^same] | The OFF/ON A-B-B-A protocol, greedy, 12 paired responses with identical token IDs. On Qwen3.5-122B the separate check is run-to-run: the official run's 12 token files were byte-identical to the parent anchor. No parity claim is made for K2.6, or for sampled decoding, which differs by construction. |
| Your weights after the one-time repack | byte for byte the source tensors[^bytes] | Every record SHA-256 compared against the source bytes, all records, no sampling, consumed fail-closed by both the launcher and the engine. |

Each row is expanded, with its full protocol and its caveats, in [Measured results](#measured-results)
and in [TECHNICAL.md](TECHNICAL.md). Nothing above is cherry-picked, and no number in this
document appears without the conditions it was taken under.

Built and verified with AI assistance (Claude, GPT); [full credits below](#credit-and-license).

**New in v0.2.2:** the warm-up file precompute ([how to use it](#precomputing-a-prompt-file-at-start))
and the experimental arch-template path that serves a GGUF the catalog does not pin
([what it opens and what it does not](#running-an-unlisted-model-experimental)). The KV cache
`q8_0` opt-in switch and the DeepSeek-V4 prefetch depth search moved to the next release; the
working queue lives in [Model support roadmap](#model-support-roadmap).

> ### Where this is going
> **Four workstreams are already running, and none of them is "more catalog entries".** They ship
> one per release rather than all at once, so each lands when it is done instead of waiting for the
> slowest.
>
> - **v0.3 - the repack rework.** The one-time repack today costs your disk **the model's size
>   again**. The rework stops doing that: no second copy, the original file left untouched and read
>   in place. Space first, read performance with it. This is the one being finished now.
> - **v0.4 and the releases after it - the speed work**, ordered by what is ready rather than by a
>   schedule: a prefill path that reads each expert **once per request instead of once per token**
>   (the cheapest form of it, zero engine changes, about **3x** in an internal probe); prefetch that
>   **derives its own starting point for any model family** instead of being hand-measured one model
>   at a time; and a hunt of the same kind on the **decode** side.
>
> Dates are not promised and numbers ship only when they are measured - but know that what you are
> holding is the floor, not the ceiling.

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

---

## Table of contents

- [Who this release is for](#who-this-release-is-for)
- [What this is - and what it is not](#what-this-is---and-what-it-is-not)
- [Before you start](#before-you-start)
- [Quick start](#quick-start)
- [What success looks like](#what-success-looks-like)
- [Supported models](#supported-models)
- [Running an unlisted model (experimental)](#running-an-unlisted-model-experimental)
- [How it works](#how-it-works)
- [Measured results](#measured-results)
- [Connecting a client](#connecting-a-client)
- [Warm start](#warm-start)
- [Model support roadmap](#model-support-roadmap)
- [What gets written to your disk](#what-gets-written-to-your-disk)
- [Update, reset, uninstall](#update-reset-uninstall)
- [Source](#source)
- [Known limitations](#known-limitations)
- [Reporting problems](#reporting-problems)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Credit and license](#credit-and-license)

The long version, with every technique and every number, is in its own document:
[TECHNICAL.md](TECHNICAL.md).

---

## Who this release is for

> **v0.2.2 is a public preview aimed at hands-on users.** It runs, it is measured, and its rough
> edges are written down rather than hidden. It is not a one-click app for casual use yet - that
> is a direction, not a promise.

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

**Windows will warn you.** v0.2.2 is an *unsigned* public preview, so SmartScreen showing
"Windows protected your PC" is the expected outcome for a new unsigned file, not a sign that
something is wrong with your download - verify the SHA-256 and decide for yourself. On managed
PCs, or with Smart App Control enabled, the file may be blocked outright with no "Run anyway"
option; Smart App Control has no per-app exception. We will never ask you to turn off Defender,
SmartScreen or Smart App Control, to add antivirus exclusions, to change your machine-wide
execution policy, or to run anything as administrator. Code signing is planned but not in v0.2.2:
this project uses the SignPath Foundation's free open-source code signing program (application
in progress), and release binaries will be signed via SignPath.io once approved - the policy is
in [CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md). Even a signed build does not make
first-release warnings disappear immediately.

## Before you start

**You need, before anything else:**

| | Requirement | Notes |
|---|---|---|
| OS | Windows 10 or 11, x64 | Windows only in v0.2.2. No Linux/macOS build exists; see [FAQ](#faq). |
| Runtime | Microsoft Visual C++ Redistributable (x64) | The engine binaries are built with MSVC. Most machines already have it; if it is missing the server cannot start (see [`fail_server_start`](#fail_server_start)). Install it from Microsoft's [Latest supported Visual C++ Redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) page. |
| Storage | An NVMe SSD | This design is I/O bound by construction. Other storage - a SATA SSD, an external drive - is **not validated and not recommended**: the launcher does not block it, it measures your drive and reports the throughput it found. |
| Model | It must be GGUF, and models that have not been tested as of v0.2.3 are also acceptable |  Not all models are supported; for detailed information, Exact repos and revisions in [Supported models](#supported-models). |
| Disk | About **2x the model size**, plus reserve | The repack writes its output next to your GGUF and the original stays. Nothing is deleted. |
| Time | **Minutes to roughly 18 minutes on the recorded machine, once per model** | Recorded there: 61 GB store in 130 s, 72.8 GB in 220 s, 436 GB in 18 min. Another drive may take longer. |
| RAM | Enough for the cache budget the launcher picks | The launcher sizes the expert cache from your installed RAM and the model's slot geometry, between 4 GB and 12 GB, and you can override it. Each profile also has a minimum below which it cannot be served at all: 8 GB for the 122B, 397B, gpt-oss and DeepSeek profiles, 4 GB for the 35B, 10 GB for K2.6. On a first run the sizing needs the slot geometry that the repack produces, so it happens after the repack and its verification; the preflight that runs before any repack output is written checks the explicit or profile-minimum budget instead. |
| Python | Not needed | A pinned CPython runtime for the repacker is included in the zip. |

## Quick start

> Prefer watching first? There is a **full unedited setup walkthrough** (single take, real time,
> with chapters): [youtu.be/I0MRTEn0G6g](https://youtu.be/I0MRTEn0G6g) - it covers everything in
> this section, including the waits you should expect.

**Step 0 - get the right file.** On the Releases page there are exactly two assets:

| Asset | What it is |
|---|---|
| `moe-direct-v0.2.2-win-x64.zip` | The runtime bundle. This is the one you want. |
| `SHA256SUMS.txt` | The checksum of that zip. |

> GitHub also shows an automatically generated **"Source code (zip / tar.gz)"** on every release.
> That is *not* a runnable bundle - it contains no binaries. Do not download it to run
> MoE-Direct.

**Then, in this order.** The order matters: Windows marks downloaded files, and unblocking the zip
*after* extracting does not clean up the files that were already extracted.

1. **Verify the download.** In PowerShell:
   ```powershell
   Get-FileHash .\moe-direct-v0.2.2-win-x64.zip -Algorithm SHA256
   ```
   Compare the result with `SHA256SUMS.txt`. If it does not match, stop and download again.
2. **Unblock the zip itself** - right-click `moe-direct-v0.2.2-win-x64.zip` -> Properties -> tick
   **Unblock** -> OK. (Equivalent: `Unblock-File .\moe-direct-v0.2.2-win-x64.zip`.)
3. **Extract with Windows "Extract All"** into a folder of your choice, for example
   `C:\moe-direct\v0.2.2\`. Other archivers differ in how they propagate the mark-of-the-web, so
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

   Since v0.2.3 each entry also carries a short label read from that file's own header, so you can
   see before you pick which route the file is likely to take: `[catalog]` for a model the catalog
   describes, `[template: <arch>]` for an unlisted GGUF of an architecture there is a template for,
   `[unsupported]` for one there is not, and `[identify pending]` when the header did not give up
   all four identification fields. The screen says it too: **the labels are provisional.** They come
   from one header read per file, and the real decision at start also weighs the shard count, the
   file sizes and the source pin, so a `[catalog]` label is a good guess and not a verdict.

6. **Approve the one-time repack.** The launcher identifies your model, checks RAM and disk, then
   stops and shows you the exact cost - output size, free space left afterwards, expected time -
   and no model or repack output is created until you answer. This is the long step: minutes to
   roughly 18 minutes on the recorded machine, depending on model size, with
   live progress. There is no resume in v0.2.2; if you cancel, the next run starts the repack from
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
changed part again. So judge the speed by the later turns of a session, not its first one.

Stopping the server still empties the expert cache, which fills from the NVMe again on the next
start. The prefix cache is the part that now survives: the launcher saves the slot state when you stop
cleanly, and restores it next time, so the first turn of the next session does not re-prefill a
prompt that was already processed. [Warm start](#warm-start) describes what that covers and what it
does not.

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
| `experimental` | Neither gate established. Your own risk. No pinned catalog entry ships at this tier; a profile derived by [Running an unlisted model (experimental)](#running-an-unlisted-model-experimental) is exactly this tier. |

| Model (profile id) | Source | Experts | Repacked expert store | Min cache budget | Tier | Prefetch |
|---|---|---:|---:|---:|---|---|
| **Qwen3.5-122B-A10B Q4_K_M** (`qwen35-122b-nonextn`) **<- start here** | [bartowski/Qwen_Qwen3.5-122B-A10B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF) rev `fec8b222a2eddc3346d6b6d7f7c85efea93cd6bf` | 256 (top-8) | 72.8 GB | 8192 MB | `reference-validated` | `validated` (K=8, N=4) |
| gpt-oss-120b MXFP4 (`gpt-oss-120b`) | [ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) rev `8d158cefb5f175c6f8842bbd8f68eca54d951ab4` | 128 (top-4) | 61 GB | 8192 MB | `format-validated` | `reference-only` |
| Qwen3.5-35B-A3B Q4_K_M (`qwen35-35b`) | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) rev `bc014a17be43adabd7066b7a86075ff935c6a4e2` | 256 (top-8) | 19.5 GB | 4096 MB | `format-validated` | `disabled` |
| Qwen3.5-397B-A17B Q4_K_M, 6 shards (`qwen35-397b`) | [unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) rev `da33c16fa4440f831149fcf53b98a22bc07785e5` | 512 (top-10) | shown by the launcher before it writes | 8192 MB | `format-validated` | `disabled` |
| Kimi K2.6 447 GB mixed-quant (`kimi-k2.6-ram-447gb`) | [baa-ai/Kimi-K2.6-RAM-447GB-GGUF](https://huggingface.co/baa-ai/Kimi-K2.6-RAM-447GB-GGUF) rev `1e8bc2c2c759db5b4bb783965129d4e1e9182bc6` | 384 (top-8) | 436 GB | 10240 MB | `format-validated` | `validated` (K=8, N=4) |
| DeepSeek-V4-Flash-0731 MXFP4/Q8_0 (`deepseek-v4-flash`) **new** | [bullerwins/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/bullerwins/DeepSeek-V4-Flash-0731-GGUF) rev `ed48c7a2df419aaa01e325521cd6f93464969641` | 256 (top-6) | shown by the launcher before it writes | 8192 MB | `format-validated` | `disabled` |

Notes on this table:

- **First time? Take Qwen3.5-122B.** It is the only `reference-validated` profile and the one the
  published speed number belongs to. Budget about 73 GB of disk for its expert store.
- **Prefetch column.** `validated` means the next-layer expert prefetch was measured and frozen for
  that profile on the reference machine. `reference-only` means the signal exists but the
  end-to-end lever has not been qualified, and `disabled` means the family adapter is not built
  yet. The engine and the launcher both refuse a prefetch override on a profile that is not
  `validated` - this is deliberate, not a bug. K2.6 reached `validated` through a paired A-B-B-A
  run in which both adjacent pairs favoured the ON arm and all four arms produced byte-identical
  output. **That row describes the shipped catalog (byte-identical since v0.2.1).** The promotion travels with `models.json`, not
  with the model: a v0.2 bundle you already downloaded still carries `reference-only` for K2.6 and
  still serves it with prefetch off, and only a bundle shipping the updated catalog turns it on.
  The promotion is also about prefetch and nothing else: K2.6's performance gate is exactly where
  it was, unpassed. gpt-oss-120b is the one profile still sitting at `reference-only`. Both runs
  are recorded under [Non-official observations](TECHNICAL.md#non-official-observations).
- **DeepSeek-V4-Flash-0731 is new in v0.2.1**, and it is the newest architecture in the catalog
  rather than the most measured one. Its repack verified 33,024 of 33,024 record-part pairs, and a smoke run
  on the reference machine held 3.13-3.28 tok/s across four probes with zero fallback events
  (`PROBE`: budget 8192 MB, QD 8, prefetch off, ctx 8192). Prefetch is `disabled` for it because
  the depth constants for that family have not been searched, not because the family is
  unsupported. The launcher will accept a context up to 131072 on the custom path, but nothing
  above 8192 has been exercised here, so treat that headroom as untested rather than offered.
- **Text only.** This release validates text serving. Multimodal (mmproj) inputs are **not verified** -
  not blocked, just unlabelled. Reports welcome.
- **Where these values come from.** The catalog file `models.json` in the bundle is the single
  source of truth for the profile id, the repository and revision, the expert count and top-k, the
  minimum cache budget, the tier and the prefetch state; those columns are rendered from it. The
  **Repacked expert store** column is not a catalog field: it is the size recorded when that model
  was actually repacked here - on the reference machine, except the 35B row, which was repacked on
  the second machine described in [TECHNICAL.md](TECHNICAL.md). The 397B and DeepSeek rows
  have no recorded size yet. The launcher computes the exact size for *your* model and shows it in
  the repack plan before it writes.

## Running an unlisted model (experimental)

As of version v0.2.3, the following features are built-in and can be used automatically, while the following applies when used manually 

The table above is a list of six models, and until v0.2.2 it was also the list of models that would
run at all: a GGUF the catalog did not carry was refused even when its architecture was one the
engine already serves. That default is defensible and it is also frustrating, so v0.2.2 opened the
half of it that can be opened honestly, behind a switch. In v0.2.2 you had to know the switch
existed to find it, which meant almost nobody did, so **in v0.2.3 the path is on by default**: a
GGUF of a **known architecture** that is not in the catalog is derived, repacked, verified and
served, labelled `experimental`, with no flag at all.

Nothing about the checks changed with the default. If you want it off, for one run or for good:

```powershell
.\Start-MoeDirect.ps1 -ArchTemplate off
```

The flag applies to that run and is not written down, so it behaves like every other command-line
override here. To make the choice stick, flip it from the **model selection menu**, on the
`arch template:` row - that is stored, and later starts follow it with no flag. If you start with
`-Model` and skip the menu, the launcher asks you once, the first time a run would actually take the
template path, and remembers your answer. The custom screen has the same item, but it says
`applies from the next start` there and means it: by that point your model is already identified,
and pretending otherwise would be a lie.

The setting is stored machine-wide, next to your saved preset rather than inside it, because the
launcher has to know the answer *before* it has identified your model, and a preset is bound to a
model it has not identified yet.

`-ArchTemplate on` turns it back on the same way. The old `-ExperimentalArchTemplate` still works
and still means "on", so scripts written against v0.2.2 keep running; it is now the older spelling
of the same thing, and if you somehow pass both, the explicit `-ArchTemplate off` wins.

**Known architecture is the whole of the opening.** Templates exist for three architectures today -
`gpt-oss`, `qwen35moe` and `deepseek2` - and the end-to-end gate evidence below was run on one of
them, `gpt-oss`; a GGUF of the other two takes the same checked path, but no model of theirs has
been taken through the full gate run yet, so treat them with the stronger caution the label already
implies. A GGUF of an architecture there is no template for is still refused, before any repack
output is written, exactly as it was - the default does not turn any check off, it adds a second way
of passing the same ones. **The repack still stops and asks.** Turning this on by default did not
turn the y/N approval into a formality: nothing is derived, written or served until you answer that
prompt, and the plan you approve is the same plan v0.2.2 showed you. It
changes nothing for the six models in the table either: a model the catalog identifies takes the
route it always took, and an identified model that then fails a later check is a hard failure, not a
quiet demotion onto the experimental path.

**What is verified here, stated exactly.** Before anything is written, the launcher runs a plan that
writes nothing: it parses every shard header, decides whether your file is one the catalog knows,
closes the routed-expert inventory the architecture template selects, sizes the cache from the slot
geometry it derives, and shows you the result for approval. The repack then hashes every record it
writes against your file, all of them, as it does for any model. The engine does not take the
repacker's word for which tensors that inventory should have contained: it holds its own frozen
table of approved templates, regenerates the expected tensor set from the architecture and layer
formula it reads live, and only then runs the same seals a catalog model goes through. What this
path is allowed to claim is narrower than what a catalog entry claims, and the launcher prints it in
those words - *the template-selected routed-expert inventory was copied byte-for-byte from your
file*. Three questions are reported separately on the status screen and never folded into one badge:
`copy integrity` (did every selected routed slice verify byte for byte), `inventory authority` (who
decided which tensors the inventory contains - a model entry, or the architecture template) and
`serving validation` (has this configuration been validated for serving). They are three different
questions, and none of them is an answer to another.

**What you do not get.** No performance work. Prefetch stays off on a derived profile and the levers
that were tuned per profile are not offered here, which is a large part of why the label is
`experimental` rather than a tier. No published number covers your model, and the launcher says so
before the repack rather than after.

**What was run before this shipped.** gpt-oss-20b - 24 layers, 32 experts, MXFP4, about 12.1 GB, a
different shape from the 120B in the catalog and picked for that reason - was taken end to end
through 25 frozen gates and passed all 25. That covered the repack verifying every record part, the
engine's template seal attesting the derived inventory, zero touch and zero fallback events while
serving, and the check that matters most on a path that decides which tensors get read: five greedy
prompts served through the direct-read path returned token-identical output to the same engine
binary reading the same weights through plain mmap. Four deliberate mutations - of the derived
expectation file, the template, the manifest and the model binding - were each refused at the gate
that owns them, and the three existing profile regressions (the 512-expert 397B, the NextN layers on
Qwen3.5-122B, the 384-expert K2.6) were re-run unchanged. What that evidence does not do is make
your model a measured one. It says the path is sound on one model of one architecture, which is
exactly what `experimental` is here to mean. How the two catalog tiers work, and what each of the
three checks actually does, is in
[TECHNICAL.md](TECHNICAL.md#serving-a-model-the-catalog-does-not-pin).

## How it works

Your GGUF's expert tensors are rewritten once into a layout built for reading. After that the model
runs with only a slice of its experts in memory at a time: a budget you can see holds the ones being
used, and the rest stay on the NVMe and are read as the router asks for them, straight off the drive
rather than through the operating system's page cache. Two terms come up throughout this document:
the **expert store** is the single file the repack produces, `experts.bin`, and the **repack
output** is the `repack\` folder holding it together with `manifest.json` and `verify_report.json`.
The dense layers run on the GPU as they always did, with `-ngl 99` on the reference machine's
RTX 5080; only the expert stream lives on the CPU and NVMe path.

What that buys you is the part worth stating plainly. A model far larger than your RAM starts in
seconds rather than not starting at all, because nothing is bulk-loaded up front. It answers from
the first token. It gets faster when you keep it on the same kind of work, because the cache fills with whatever
your work actually touches - a genuinely new topic streams its own experts in first. And where that claim has been put to the test - greedy decoding on gpt-oss-120b, direct reads
against the same binary's mmap path - it produced token-for-token the same output, because none
of this changes what is computed, only where the bytes come from; the exact scope of that
comparison is stated with the headline table above.

The launcher measures your machine instead of assuming ours. A short read-only sweep picks the queue
depth for your drive, and the cache budget is sized from your installed RAM and the model's own
geometry. Both are printed before you start, and both stay overridable. What you get at the end is
an ordinary local `llama-server` endpoint on loopback, speaking the implemented OpenAI-compatible
subset (see [Connecting a client](#connecting-a-client)).

Each of those techniques is written up in full, with the problem it solves, what it does, how it
behaves and what was measured, in
[The techniques, and what they do](TECHNICAL.md#the-techniques-and-what-they-do). Why the design
looks like this, which alternatives were measured and rejected on data, and what was underneath each
decision, is in a technical note with a DOI:
[10.5281/zenodo.21739367](https://doi.org/10.5281/zenodo.21739367).

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
[TECHNICAL.md](TECHNICAL.md#the-release-pair-headline-source) as the earlier generation's
record rather than deleted, because that engine is not this one. The frozen gate record predates
both and keeps its own section.

The release-binary pair the headline comes from, the reference machine's full specification, the
paired protocol, the historical gate record, the release binary's lineage and every non-official
observation recorded so far live in **[TECHNICAL.md](TECHNICAL.md)**. Nothing was moved there
to bury it. That file exists so this page can stay short enough to read.

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

## Warm start

Stopping the server used to throw the whole session away. The prefix cache went with the process, so
the next start had to prefill your prompt again from nothing, and on a long agent-style prompt that
is minutes of work you had already paid for. The launcher saves the slot state when you stop cleanly and
restores it the next time you start.

You do not have to do anything to get it. The saved state goes under
`%LOCALAPPDATA%\MoE-Direct\kv\<profile id>\`. One thing to expect on the first clean stop with a
given model: the integrity sidecar hashes the model file once, and on a very large model that is a
few minutes of reading with no output yet; the hash is cached, so later stops skip it. The status
screen tells you what it found before you press Enter:

```
  kv               : eligible
  autosave         : on (every 5 min)
```

`eligible` means a saved state passed every check and will be restored. `cold(<reason>)` means it
will not be, and names the reason. `off(user)` and `off(mode)` mean you turned it off, or the run is
a self-check or reproducibility run where it is off by design. Once the server is up a second line
reports what actually happened, `kv=restored(<n> tokens)` or `kv=cold(<reason>)`. Verifying a large
saved state takes a moment, and the launcher prints a line while it does that rather than going
quiet.

**Restoring the wrong state would be worse than starting cold, so it is not allowed to happen.** A
slot file written by llama.cpp carries no model hash, no vocabulary, no RoPE settings, no engine
identity and no checksum of its own; the engine validates its format, not whether it belongs to this
model. The launcher supplies what the engine cannot. Next to every saved state it writes a sidecar
recording the profile, the SHA-256 of every shard of your GGUF, the repack manifest hash, the bundle
hash, a canonical hash of the server configuration that affects state, the context size, the token
count, the byte count, and the SHA-256 of the saved file itself. All of it has to match before a
restore is attempted. Anything that does not match is a cold start with the mismatching field named,
not a gamble.

**Autosave, for the stops you did not plan.** Saving on a clean stop does nothing for a crash, a
power cut or a forced kill, so the launcher also saves while it is serving. The default is a five-minute
tick that fires only when two things hold together: nothing is in flight, and the token count has
actually changed since the last save. Writes alternate between two generations, so a crash during a
save damages only the generation being written and the previous complete one survives. On the next
start the most recent state that passes the checks is restored, whether it came from a clean stop or
from a tick.

**What it does not do.** A slot file holds tokens and cache state, not the server's own prefix
checkpoints. A request that extends the restored prompt exactly reuses it; a request that diverges
from it is reprocessed in full on hybrid-attention models such as Qwen3.5. Measured figures for both
cases are in [TECHNICAL.md](TECHNICAL.md#warm-start). It also does nothing for the expert slot
cache, which still fills from the NVMe as you use the model; pre-filling that is separate work and
is not in this build.

**These files hold your conversation.** A saved state contains the session's tokens verbatim, so
treat it the way you would treat the conversation itself. It is local, it is never uploaded, and
nothing in this project collects it. At a 12k context it runs about 190 MB per state. Each profile
keeps one clean-stop state and two autosave generations, and the launcher tries to hold at most
four profiles' worth, removing the least recently used beyond that. A removal that fails is recorded
as a diagnostic rather than failing the run, so the folder can end up holding more than that.

To turn the feature off, set `warmstart` to `off` on the custom path or start the launcher with
`-Warmstart off`. To keep warm start but stop the periodic saves, set `autosave` to `off`, or to a
number of minutes between 1 and 1440 to change the interval. Turning it off stops new saves; it does
not delete what is already there. Delete `%LOCALAPPDATA%\MoE-Direct\kv\` yourself when you want it
gone.

### Precomputing a prompt file at start

Warm start only helps a start that has something to restore. The ones that do not - a first run with
a new model, a start after you deleted the saved state, a session that begins on a different prompt -
still pay the full cold prefill on the first turn, and on an agent-style system prompt that is
minutes of silence before the first token appears. Those tokens have to be evaluated once and no
setting makes that free. What you can change is *when* it happens, and whether you are the one
sitting in front of it. Point the launcher at the file your client sends as its system prompt and
that prefix is computed right after the server comes up, before you ask anything:

```powershell
.\Start-MoeDirect.ps1 -Warmup file:C:\path\to\systemprompt.txt
```

On the custom path, or in a saved preset, the same thing is the `warmup` key set to
`file:C:\path\to\systemprompt.txt`. That key takes three values: `on`, `off` and `file:<path>`.
Quote the whole value if your path contains spaces; the part after `file:` is taken exactly as you
wrote it.

**The default changed in v0.2.3: `warmup` is now `on`, where v0.2.2 and earlier had it `off`.** `on`
is the one-token warm-up request that has been in the launcher all along, sent once after the server
reports ready. It is one small request, and it means the first thing you type is not also the thing
that pays for the very first token of the run. If you preferred the old behaviour, `-Warmup off` or
the `warmup` key set to `off` restores it exactly.

That default has one honest consequence, and it is worth stating plainly because it changes what
the status screen tells you. **Every published number in this README was measured on a cold cache,
and a warmed-up run is not that condition.** So a default start now reports its performance gate as
`[unmeasured] (product warm-path baseline; official measurements are cold-cache)` rather than
claiming a measured result it is not entitled to. This is a labelling change, not a slower run -
warming up does not make anything worse, it just puts the machine in a state the official figures
did not measure. `-Repro` and `-Smoke` force `warmup` back off for exactly this reason, so a
reproduction or benchmark run stands on the same cold-cache footing as the published pairs; the
status screen names the force on the warmup line when it happens.

It is one request, sent once, and you watch it happen: the launcher reads the file as UTF-8 and
sends the text to the server exactly as the file holds it, with prompt caching on and one token of
output asked for. It is deliberately not passed through the model's chat template on the way,
because tokens computed from a re-wrapped copy would not be a prefix of what your client's first
request renders, and being that prefix is the entire point. When it finishes, the launcher prints
what the server counted (one line, wrapped here to fit):

```
[warmup] Precomputed 282 tokens. The launcher cannot observe client reuse; check the first
response timings.cache_n (expected close to 282 (tokenizer boundaries and cache checkpoints may
re-evaluate a small tail)).
```

**The verification is handed to you because the launcher cannot do it itself.** It is not a proxy
and never sees your client's request, so it states what it precomputed and stops there. Look at
`timings.cache_n` in the first response you get back: close to N means the precompute was reused, and
a shortfall of a few tokens is normal rather than a failure - the tokenizer boundary where your file
meets whatever your client appends can move a token, and a model that keeps cache checkpoints
re-evaluates a short tail. A `cache_n` near zero is the answer that means it did not work. Reuse
needs your client's rendered token sequence to begin with all N precomputed tokens, so a system
prompt differing by one character, role marker or line ending is a different prefix even when the
file looks the same.

**Warm start wins when both could apply.** If a saved state is restored, the precompute is skipped
and says so: the restored prefix is already in the slot, and overwriting it would cost you more than
the precompute buys. If you want the file precomputed for a new conversation that will diverge from
the restored one, start with `-Warmstart off`.

**Nothing here can fail your run.** A missing file, an empty file, a file that is not valid UTF-8, an
HTTP failure, a timeout: each of those prints a warning, records the reason and carries on serving
without the precompute. The request is bounded at 30 minutes, because a genuine cold prefill of a
large file takes minutes and a server that never answers must not hold the launcher for ever.

One measurement, from this machine and claimed no wider: on Qwen3.5-122B a system prompt file
precomputed to 282 tokens in 12.0 s, and the first real request after it reported `cache_n` 278 of
its 299 prompt tokens - 93 % of the precomputed prefix reused, 21 tokens of genuinely new text
evaluated. The 4-token difference is the two effects named above, one token at the tokenizer seam
and three from the checkpoint tail this model family keeps. The mechanism and the full conditions
are in [TECHNICAL.md](TECHNICAL.md#warm-up-file-precompute).

### Starting the server yourself

The save and restore calls are upstream llama.cpp, and this project changes nothing about them, so
they are available to anyone who starts `llama-server` directly instead of using the launcher. Add
`--slot-save-path <directory>` to the server arguments and **create that directory first**: if it
does not exist the server refuses to start while it is still parsing arguments. The slot id is `0`,
because this project serves one request at a time from a single slot.

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8093/slots/0?action=save" -ContentType "application/json" -Body '{"filename":"my_session.kv"}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8093/slots/0?action=restore" -ContentType "application/json" -Body '{"filename":"my_session.kv"}'
```

**HTTP 200 is not the same as success.** Read the response body: a save needs `n_saved > 0` and
`n_written > 0`, a restore needs `n_restored > 0` and `n_read > 0`. If a restore is not confirmed,
`POST /slots/0?action=erase` and continue cold only once the erase succeeds, otherwise restart the
server without restoring. Restored counts alone do not prove the prompt was reused either, so
compare `timings.cache_n` against `timings.prompt_n` on your next request and check its
time-to-first-token against a cold one.

On this path none of the identity binding above applies, because that is the launcher's work. The
engine will happily restore a file saved from a different model, and the result is undefined
behaviour rather than an error message. Restore only into exactly the same model file, the same
engine build and bundle, the same context size and slot layout, and the same state-relevant server
configuration you saved from.

## Model support roadmap

This is an actively developed project, and v0.2.3 is a preview, not a finished product. The table
below is the working queue, not a wish list. Feature work ships one workstream per release - see
[Where this is going](#where-this-is-going) for the order.

| Where we are | Status |
|---|---|
| The repack that costs your disk the model's size a second time | **Being finished now, targeted at v0.3.** Instead of writing a second copy of the expert data, the launcher reads the experts out of your original file in place, so the space cost is the model and nothing more, and onboarding stops moving hundreds of gigabytes around. Nothing of it is in this build, and it ships only if it holds read performance - the rule this project works under is that space is never bought with speed. |
| Serving a 1T-class MoE (Kimi K2.6, deepseek2 architecture) from a 32 GB machine | **Done and shown** - unedited single-take demo below. Format gate passed; performance gate not passed (1.03 tok/s, honestly labelled `PROBE`). |
| Prefetch for the deepseek2 family, which is what Kimi K2.6 needs | **Landed and promoted.** The signal adapter for that family exists, a first live four-arm run on K2.6 selected K=8 / N=4, and the paired A-B-B-A run that one run could not stand in for has since been done: both adjacent pairs favoured the ON arm and all four arms produced byte-identical output. `prefetch_state` for that profile is `validated` in the v0.2.1 catalog, so a bundle shipping that catalog enables prefetch for it by itself; a v0.2 bundle still carries `reference-only` and serves it with prefetch off. Both runs, and the exact protocol, are under [Non-official observations](TECHNICAL.md#non-official-observations). None of this touches the performance gate, which K2.6 still has not passed. |
| DeepSeek-V4-Flash-0731 (284B, `deepseek4`) | **In the catalog since v0.2.1.** The architecture is native to the base commit this release is built on, its expert tensors are MXFP4, which is a layout the repacker already handles, and the repack verified 33,024 of 33,024 record-part pairs. A direct-read smoke run held 3.13-3.28 tok/s across four probes with zero fallback events (`PROBE`: reference machine, budget 8192 MB, QD 8, prefetch off, ctx 8192). It ships with the format gate passed, the performance gate unpassed and prefetch disabled, which is where the evidence actually stands; the prefetch depth search for this family is queued for a later release. |
| Any model of an already-supported architecture (`experimental` tier) | **Shipped in v0.2.2 behind `-ExperimentalArchTemplate`; on by default since v0.2.3.** The catalog still runs its six pinned models the way it always did; this adds a second way of passing the same checks for a GGUF of a known architecture, derived, repacked, verified and served labelled `experimental` with prefetch off - an honest warning, not a block and not a promise. Templates exist for three architectures (`gpt-oss`, `qwen35moe`, `deepseek2`); the end-to-end evidence is one model of one of them (`gpt-oss`), which is what keeps the label. The default moved because behind a switch nobody found it, not because the evidence grew: an architecture with no template is still refused, and the repack still stops for your approval. `-ArchTemplate off` turns it off for a run, and the model menu's `arch template:` row stores the choice. See [Running an unlisted model (experimental)](#running-an-unlisted-model-experimental). |
| Warm start: saving and restoring slot state across restarts | **Shipped in v0.2.1.** Stopping the server in v0.2 cleared both the prefix cache and the expert slot cache, so the next start began cold. v0.2.1 saves the slot on a clean stop, saves again on a timer while serving, and restores on the next start, measured at **8.1x** faster time to first token on a strict same-prompt pair. The protocol and the caveats are in [TECHNICAL.md](TECHNICAL.md#warm-start), and the user-facing description is under [Warm start](#warm-start). |
| Warm-up file precompute (`warmup` gains a `file:<path>` mode) | **Shipped in v0.2.2.** A fresh session's first turn still pays the full cold prefill; this lets you point the launcher at your actual system-prompt file so that prefix is precomputed right after start, once, while you watch. Usage and the one measurement behind it are under [Precomputing a prompt file at start](#precomputing-a-prompt-file-at-start). |
| Expert-cache warmer | **Designed, targeted at a later release.** Filling expert slots ahead of the first turn instead of letting them fill as you chat. The design is written and reviewed; nothing of it is in this build. |
| KV cache quantized to `q8_0` | **Measured and gated; the switch is still queued.** The quality gate passed on Qwen3.5-122B, with the divergence and perplexity figures in [TECHNICAL.md](TECHNICAL.md#kv-cache-q8). What is not in this build is the opt-in switch that would let you turn it on, so the measurement is published ahead of the feature rather than the other way round. It was named for v0.2.3 and did not make it - that release turned out to be a convenience patch, and this is a feature. |
| Kimi K3 (2.8T class) | **Due to current disk space limitations, we were unable to obtain the K3 MXF-P4, but the Kimi-K3-Q2_K_XL is currently prepared on the disk, and since we are prioritizing improvements to other features, it is not yet ready We will do our best to find you in a format-validated grade or higher as soon as possible (as of 2026-08-11) |
| Wider hardware, wider OS | Windows only today. A Vulkan build of the same engine has been measured off-bundle - RTX 5080 at CUDA-parity decode, and a first AMD run (RX 9070 XT, RDNA4) at about 89 percent of that, its arms character-identical among themselves - so cross-vendor GPU support is now a hardening-and-tooling queue item rather than an open question; the numbers are under [Non-official observations](TECHNICAL.md#non-official-observations). OS expansion is genuinely hard in the current test environment and will take time. Community ports are welcome. |

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
| `%LOCALAPPDATA%\MoE-Direct\kv\<profile id>\slot0.kv` and `slot0.kv.meta.json` | when you stop the server cleanly, with warm start on | The saved slot state, and the sidecar that binds it to your model, your repack, this bundle and the server configuration it was saved under. **The state file contains the session's tokens verbatim** - see [Warm start](#warm-start). Roughly 190 MB at a 12k context on the reference machine. |
| `%LOCALAPPDATA%\MoE-Direct\kv\<profile id>\slot0.auto.a.kv` / `slot0.auto.b.kv` and their `.meta.json` sidecars | on each autosave tick that fires | The two alternating crash-recovery generations. Same contents, same sensitivity, same folder. |
| `*.tmp.<generation>` / `*.stale.<generation>` in the same `kv\` folder | while a save is being written | A save is written under a temporary name and swapped in, so an interrupted save cannot damage the state you already had. Both are cleaned up at the next start; a cleanup that fails is recorded as a diagnostic and leaves the file in place. |
| `repack\` next to your GGUF (`experts.bin`, `manifest.json`, `verify_report.json`) | once per model, after you approve the plan | The direct-read expert store, its identity manifest, and the byte-level verification report that the serving gates consume - the repack output. Tens to hundreds of GB. An interrupted repack leaves `experts.bin.partial` here as well. **Never deleted automatically** - see [Update, reset, uninstall](#update-reset-uninstall). |

That is the whole list for the launcher flow. If you ever catch this project writing anywhere
else, that is a bug - please report it. (One advanced exception you create yourself: if you use
[start the server yourself](#starting-the-server-yourself), the slot files you request are
written to the directory you point `--slot-save-path` at.)

## Update, reset, uninstall

There is no installer, and nothing is written to the registry. Four things have separate
lifetimes, and you control all four:

| Thing | Where | What to do |
|---|---|---|
| The bundle | wherever you extracted the zip | **To update:** extract the new version into a *new* folder. Do not overwrite an existing bundle - the integrity manifest covers the folder as a set. Delete the old folder when you are done. |
| Launcher state and logs | `%LOCALAPPDATA%\MoE-Direct\` | **To reset settings:** delete `presets.user.json` (saved configurations), `probe.state.json` and `probe.scratch.json` (measured drive results), `recent_models.json` (the recent list). They are all rebuilt on the next run. Logs live in the `logs\` subfolder. |
| Saved session state | `%LOCALAPPDATA%\MoE-Direct\kv\` | **To delete it:** remove that folder, or the `<profile id>` subfolder for one model. Setting `warmstart` to `off` stops new saves but does not delete what is there. The launcher itself tries to hold at most four profiles, removing the least recently used beyond that, and records a diagnostic instead of failing the run when a removal does not succeed. |
| Repack output | a `repack\` folder next to each of your GGUFs | **Not deleted automatically, ever.** This is the big one - tens to hundreds of GB. Delete the `repack\` folder to reclaim the space; the next run for that model will repack from scratch. |

Uninstalling completely = delete all four. Your GGUFs are yours and are never touched.

## Source

The launcher, the repacker and the expectation files are published in this
repository, and the shipped copies are the same bytes: `launcher/Start-MoeDirect.ps1`,
`launcher/Start-MoeDirect.cmd`, `repacker/repack_experts.py` and the nine `expects/*.expect.json`
files are byte-identical to the copies inside the release zip (verified by SHA-256 at publish
time). As of v0.2.2 the zip carries all nine, so the repacker's self-test passes from a checkout
and from the zip alike - 65/65, measured on the assembled bundle; the v0.2.1 zip's 63/64 was a
known issue of that bundle and this closes it - see [repacker/README.md](repacker/README.md). `launcher/models.json` is the catalog the launcher consumes, exactly as the zip resolves
it - it is where each profile's `expect_sha256` pin lives, so the binding between the launcher and
the expectation files can be audited end to end. What runs from a plain checkout: the repacker
does (its `repacker/expects/` catalog ships in place - see [repacker/README.md](repacker/README.md));
the launcher does not, since it needs the engine binaries and bundle manifest from the zip. The launcher's own test suite (978 checks as of
v0.2.2, all passing on the shipped launcher) is not yet published: it depends
on fixture files that need to be packaged for standalone use, and it will follow in a later
release. The engine is a patched llama.cpp build; the binaries are sealed by
the bundle's SHA manifest, and the full engine delta is published in [`patches/`](patches/) as a
single reviewed patch against the pinned upstream commit, with a mechanical proof that it
reproduces the exact source tree the shipped binaries were built from; the same tree is browsable
as a [fork branch](https://github.com/tmxkzm1925-max/llama.cpp/tree/moe-direct-v0.2.2), one branch
per release. A rebased mainline PR series is in preparation.

## Known limitations

- **MoE models only.** Dense models get no benefit from this design.
- **First-run repack cost is real** - minutes to roughly 18 minutes on the recorded machine, and
  roughly the model's size again on disk. There is no resume in v0.2.2: an interrupted repack restarts
  from the beginning.
- **Windows only.** No Linux or macOS build, and no promised numbers for them.
- **This zip carries the CUDA runtime only.** The official numbers were measured on NVIDIA CUDA.
  The expert stream - the direct reads and the slot cache - runs on the CPU and the NVMe, so it
  does not depend on which GPU you have; what uses the GPU is the dense-layer offload, and that is
  stock llama.cpp with its dynamic backend loading. A Vulkan path has in fact been measured
  off-bundle - an RTX 5080 at CUDA-parity decode, and a first cross-vendor run on an AMD
  RX 9070 XT whose arms were character-identical among themselves - see
  [Non-official observations](TECHNICAL.md#non-official-observations) - but nothing Vulkan
  ships in v0.2.2, and other backends (ROCm/HIP, unified-memory platforms) remain **untested**. CPU-only serving (`-ngl 0`) needs no GPU at all - that is how the 35B row in
  [TECHNICAL.md](TECHNICAL.md) was measured. If you run this on other hardware, the
  performance-report issue form is where we would like to see the result.
- **Unsigned preview build.** Expect SmartScreen friction; some managed machines will refuse it.
- **One request at a time** (`-np 1`), and the queue is lost on restart.
- **Prefetch only on validated profiles.** Of the six shipped profiles two are `validated`, one is
  `reference-only` and three are `disabled`; only the `validated` ones serve with prefetch on, and
  an override is refused on the other four.
- **The experimental arch-template path has been proven end to end on one architecture, and it is
  on by default since v0.2.3.** It accepts a GGUF of a known architecture the catalog does not
  carry; templates exist for `gpt-oss`, `qwen35moe` and `deepseek2`, every other architecture is
  still refused, and the full gate evidence covers a `gpt-oss` model only. A derived profile serves
  with prefetch off and none of the per-profile tuning, no published number covers it. The default
  changed, the evidence did not. `-ArchTemplate off` turns it off for one run; the menu row or the
  one-time question is what stores the choice - see
  [Running an unlisted model (experimental)](#running-an-unlisted-model-experimental).
- **A default start is labelled `[unmeasured]`, on purpose.** `warmup` defaults to `on` since
  v0.2.3 and every published figure here was measured cold, so the status screen reports
  `[unmeasured] (product warm-path baseline; official measurements are cold-cache)` instead of
  claiming a gate the run did not sit for. `-Repro` and `-Smoke` force warm-up off and restore the
  published condition on that axis; the other gate requirements are unchanged and still have to be
  met on their own.
- **Warm start reuse needs an exact prefix.** A restored session is reused by a request that
  extends the saved prompt exactly. A request that diverges from it is reprocessed in full on
  hybrid-attention models, because a slot file carries no server checkpoints. Warm start also does
  nothing for the expert slot cache, which still starts empty.
- **Long-context prefill on a disk tier is slow.** Multi-turn use is where it becomes comfortable,
  because the prefix cache absorbs the repeat - see [Measured results](#measured-results).
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

**Thanks.** This project was built by one person with a great deal of machine help, and the help
was not incidental. Anthropic's Claude and OpenAI's GPT models did design work, implementation and,
just as usefully, adversarial review of each other's output, under human direction and with every
change gated by the verification this document describes. Having a second and a third reader who
never got tired is most of the reason the checking here is as strict as it is.

Thanks are owed as well to the teams whose models this was measured against: Qwen, DeepSeek,
Moonshot AI, OpenAI for gpt-oss, and Mistral. None of them are affiliated with this project and
none of them have endorsed it. They published weights that one person with one desktop could
actually study, and without that there would have been nothing here to run.
