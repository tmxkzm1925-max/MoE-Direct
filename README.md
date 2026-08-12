# MoE-Direct

**Run Mixture-of-Experts models far larger than your RAM, on an ordinary Windows PC.**

MoE-Direct keeps a model's expert weights on your NVMe SSD and reads only the experts each
token actually routes to. That is enough to serve Kimi K2.6, a 1T-class model whose 416.8 GiB
of weights are about thirteen times the RAM of the machine this project was built on, from a
32 GB desktop. The model this release is tuned for, Qwen3.5-122B, decodes at around 6 tok/s on
that same desktop, 2.3x faster than the same engine reading the same weights through plain mmap.

The server is ready in seconds, because experts are never bulk-loaded up front. And nothing about
the model is approximated: no quantization step, no routing change, no touched weights. In the
paired comparison, the direct-read path returned token-for-token the same output as the stock
path. Every expert byte is verified against your original GGUF before it is ever used.

[![MoE-Direct demo](https://img.youtube.com/vi/JDfrWMxwczk/maxresdefault.jpg)](https://youtu.be/JDfrWMxwczk)

*Unedited single take (2:49): cold boot to answer on the 447 GB Kimi K2.6, then the same run on
Qwen3.5-122B. Task Manager stays on screen the whole time.*

**What this is, honestly:**

- It is for **MoE models only**. Dense models gain nothing here, and a model that already fits
  in your RAM does not need this.
- It is a **hands-on preview, not a one-click app**. You bring your own GGUF; MoE-Direct never
  downloads weights. The rough edges are written down in the docs rather than hidden.

## Quick start

You need Windows 10/11 x64, an NVMe SSD, disk space of about twice the model size, and a GGUF
from the [supported list](#supported-models).

1. **Download** `moe-direct-<version>-win-x64.zip` from
   [Releases](../../releases), check its SHA-256 against `SHA256SUMS.txt`, right-click,
   Properties, **Unblock**, then extract with Windows "Extract All".
2. **Place your GGUF** under `<drive>:\moe-models\<any-folder>\` and double-click
   `Start-MoeDirect.cmd`. Pick your model with the arrow keys.
3. **Approve the one-time repack** (the launcher shows the exact disk cost and time before
   writing anything), press Enter when the status screen appears, and connect any
   OpenAI-compatible client to the printed URL.

That is the whole loop. The first run repacks once (minutes to ~18 minutes here, with live
progress); every later run goes straight to serving. First conversation of a session is the
slowest by design; later turns reuse the prefix cache and are much faster.

Prefer watching first? The **[setup walkthrough](https://youtu.be/I0MRTEn0G6g)** is a single
real-time take with chapters, including the waits.

Windows SmartScreen will warn you: v0.x is an unsigned preview, so "Windows protected your PC"
is the expected message, not a verdict on your download. Verify the SHA-256 and decide for
yourself. We never ask you to disable Defender or SmartScreen. Details, including managed-PC
cases, are in [Getting started](docs/getting-started.md).

Full detail for every step, including what success looks like on screen:
**[docs/getting-started.md](docs/getting-started.md)**.

## Supported models

| Model | Experts | Expert store | Tier |
|---|---:|---:|---|
| **Qwen3.5-122B-A10B Q4_K_M** — start here | 256 (top-8) | 72.8 GB | `reference-validated` |
| gpt-oss-120b MXFP4 | 128 (top-4) | 61 GB | `format-validated` |
| Qwen3.5-35B-A3B Q4_K_M | 256 (top-8) | 19.5 GB | `format-validated` |
| Qwen3.5-397B-A17B Q4_K_M | 512 (top-10) | shown before write | `format-validated` |
| Kimi K2.6 447 GB mixed-quant | 384 (top-8) | 436 GB | `format-validated` |
| DeepSeek-V4-Flash-0731 MXFP4/Q8_0 | 256 (top-6) | shown before write | `format-validated` |

`reference-validated` means both the byte-exact format gate and the frozen performance gate
passed on the reference machine; `format-validated` means the byte-exact gate passed and any
speed number is an observation. An unlisted GGUF of a known architecture can also be served
through the experimental template path, clearly labelled.

Exact repositories, pinned revisions, minimum cache budgets, prefetch states and the template
path: **[docs/models.md](docs/models.md)**.

## Measured results

Every number ships with the conditions it was measured under; these are the headlines.

| What | Result | Grade |
|---|---|---|
| Qwen3.5-122B sustained decode | **5.59-5.69 tok/s**, release gate PASS, **2.3x** over the same binary's mmap path | `OFFICIAL` |
| Output parity, direct-read vs stock path (gpt-oss-120b, greedy) | 12 paired responses, token IDs identical | `OFFICIAL` |
| Kimi K2.6 (1T class) from 32 GB RAM | 1.03 tok/s, coherent output | `PROBE` |
| Your weights after the one-time repack | byte-for-byte the source tensors, every record SHA-256 verified | gate, fail-closed |

Results scale primarily with NVMe read throughput; treat them as data points from the reference
machine (32 GB RAM, one RTX 5080, Gen5 NVMe), not as promises for yours. Full tables, protocols
and grades: **[docs/measured-results.md](docs/measured-results.md)** and
**[TECHNICAL.md](TECHNICAL.md)**.

## Roadmap

Work ships one piece per release, when it is measured, not on a schedule.

- **v0.3, in progress:** the repack without the second copy. Today the one-time repack costs
  your disk the model's size again; the rework reads experts out of your original file in
  place. It ships only if it holds read performance — space is never bought with speed here.
- **v0.4 and after:** the speed work, in whatever order is ready first — a prefill path that
  reads each expert once per request instead of once per token (about 3x in an internal probe),
  prefetch that derives its own starting point for any model family, and the same hunt on the
  decode side.
- **Further out:** an engine-neutral expert-execution core. The parts this project actually
  owns — the expert store, cache, placement and prefetch — are being carved to a clean boundary,
  with llama.cpp as the first engine behind it. Also code signing, wider hardware and OS support.

What you are holding is the floor, not the ceiling.

## Documentation

| Read this for | File |
|---|---|
| Requirements, install, first run, what success looks like | [docs/getting-started.md](docs/getting-started.md) |
| Model list, pinned revisions, running an unlisted model | [docs/models.md](docs/models.md) |
| How the direct-read path works | [docs/how-it-works.md](docs/how-it-works.md) |
| All measurements, protocols and grades | [docs/measured-results.md](docs/measured-results.md) |
| Connecting chat clients and agents | [docs/clients.md](docs/clients.md) |
| Warm start and prompt precompute | [docs/warm-start.md](docs/warm-start.md) |
| What gets written to disk; update, reset, uninstall | [docs/disk-layout.md](docs/disk-layout.md) |
| Status codes, troubleshooting, reporting problems | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Known limitations and FAQ | [docs/faq.md](docs/faq.md) |
| The long version: every technique and every number | [TECHNICAL.md](TECHNICAL.md) |

## Credit and license

MIT License. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp); built and verified
with AI assistance (Claude, GPT). Trademarks and citation: [TRADEMARKS.md](TRADEMARKS.md),
[CITATION.cff](CITATION.cff). Archived releases carry a DOI:
[10.5281/zenodo.21739367](https://doi.org/10.5281/zenodo.21739367).
