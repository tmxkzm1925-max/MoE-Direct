# MoE-Direct

**Run Mixture-of-Experts models far larger than your RAM, on an ordinary Windows PC.**

MoE-Direct keeps a model's expert weights on your NVMe SSD and reads only the experts each
token actually routes to. That is enough to serve Kimi K2.6, a 1T-class model whose 416.8 GiB
of weights are about thirteen times the RAM of the machine this project was built on, from a
32 GB desktop. On the reference desktop, Qwen3.5-122B averaged **5.96 tok/s** (5.65-6.26 per
probe, `PROBE`) in a paired run on the v0.2.1 release binary, **2.2983x** the same binary's
plain-mmap arm. The separate historical `OFFICIAL` gate recorded **5.59-5.69 tok/s**; neither
run was repeated for v0.2.3.

The server is ready in seconds on the recorded reference machine - about 19 seconds in the Kimi
demo - because experts are never bulk-loaded up front. And nothing about the model is
approximated: no quantization step, no routing change, no touched weights. Separately, on
**gpt-oss-120b under greedy decoding**, 12 direct-read/plain-mmap response pairs had identical
token IDs - not a parity claim for sampled decoding or Kimi K2.6. On Qwen3.5-122B the separate
check is run-to-run reproducibility, not a second parity pair. Every expert byte is verified
against your original GGUF before it is ever used.

[![MoE-Direct demo](https://img.youtube.com/vi/JDfrWMxwczk/maxresdefault.jpg)](https://youtu.be/JDfrWMxwczk)

*Unedited single take (2:49): cold boot to answer on the 447 GB Kimi K2.6 - server loaded in
about 19 seconds, the NVMe sustaining multi-GB/s reads, system RAM under 32 GB the whole way -
then the same run on Qwen3.5-122B. Task Manager stays on screen the whole time.*

**What this is, honestly:**

- It is for **MoE models only**. Dense models gain nothing here, and a model that already fits
  in your RAM does not need this.
- It is a **hands-on preview, not a one-click app**. You bring your own GGUF; MoE-Direct never
  downloads weights. The rough edges are written down in the docs rather than hidden.

The exact boundary — what is changed and what is provably not touched (the math, the routing,
the weights, your files) — is stated once, in [How it works](docs/how-it-works.md). Known rough
edges live in [Limitations and FAQ](docs/faq.md).

## Quick start

You need Windows 10/11 x64, an NVMe SSD, disk space of about twice the model size, and a GGUF
from the [supported list](#supported-models).

1. **Download** `moe-direct-v0.2.3-win-x64.zip` from [Releases](../../releases), right-click
   it, Properties, tick **Unblock**, then extract with Windows "Extract All" into a new, empty
   folder. Checking the download is one paste, not a hex comparison. In Explorer, open the
   folder that holds both downloaded files, right-click empty space > Open in Terminal
   (PowerShell), then paste:

   ```powershell
   $e=((Get-Content .\SHA256SUMS.txt -Raw).Trim() -split '\s+')[0];$a=(Get-FileHash .\moe-direct-v0.2.3-win-x64.zip -Algorithm SHA256).Hash;if($a -eq $e){'OK: hash matches'}else{'MISMATCH: download again'}
   ```

   If you skip it, the launcher's own sealed-manifest check still catches files changed or
   corrupted **inside the extracted bundle** on every start - what it cannot do is authenticate
   the zip or the launcher itself; that is what this paste is for.
2. Download **every shard** of one exact tested GGUF and revision from
   [docs/models.md](docs/models.md), and keep all shards in one folder. **Place your GGUF**
   under `<drive>:\moe-models\<any-folder>\` and double-click `Start-MoeDirect.cmd`. Pick your
   model with the arrow keys.
3. **Approve the one-time repack** (the launcher shows the exact disk cost and time before
   writing anything), press Enter when the status screen appears, and connect any
   OpenAI-compatible client to the printed URL.

That is the whole loop. The first run repacks once (minutes to ~18 minutes here, with live
progress); every later run goes straight to serving. A **cold** session's first conversation is
the slowest by design; later turns reuse the prefix cache, and a restored exact prefix can skip
that first-prefill entirely.

Prefer watching first? The **[setup walkthrough](https://youtu.be/I0MRTEn0G6g)** is a single
real-time take with chapters, including the waits.

Windows SmartScreen will warn you: v0.x is an unsigned preview, so "Windows protected your PC"
is the expected message, not a verdict on your download. Verify the SHA-256 and decide for
yourself. We never ask you to disable Defender or SmartScreen. Details, including managed-PC
cases, are in [Getting started](docs/getting-started.md).

Full detail for every step, including what success looks like on screen:
**[docs/getting-started.md](docs/getting-started.md)**. Connecting a chat client or an agent:
**[docs/clients.md](docs/clients.md)**.

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

| What | Result | Evidence |
|---|---|---|
| Qwen3.5-122B sustained decode | **5.59-5.69 tok/s**, `GATE1_SERVE: PASS`; direct-read arms 2.3226x/2.3439x over mmap (separate ISLC-off cross-run estimate: 2.0695x) | `OFFICIAL` |
| Qwen3.5-122B on the v0.2.1 release binary | 5.65-6.26 per probe, arm average 5.96; combined 2.2983x over the same binary's mmap arm | `PROBE` |
| Output parity, direct-read vs stock path (gpt-oss-120b, greedy) | 12 paired responses, token IDs identical | `OFFICIAL` |
| Kimi K2.6 (1T class) from 32 GB RAM | 1.03 tok/s, coherent output (budget 10240 MB, QD 8, prefetch off, older staging binaries) | `PROBE` |
| Your weights after the one-time repack | byte-for-byte the source tensors, every record SHA-256 verified | `FORMAT GATE` (fail-closed) |

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
  reads each expert once per request instead of once per token (about 3x in an internal probe -
  an internal result, not yet a published benchmark), prefetch that derives its own starting
  point for any model family, and the same hunt on the decode side.
- **Further out:** an engine-neutral expert-execution core is **planned**: the parts this
  project owns - the expert store, cache, placement and prefetch - defined behind a clean
  boundary, with llama.cpp as the first engine behind it. Also code signing, wider hardware and
  OS support.

What you are holding is the floor, not the ceiling.

Full detail on what has already shipped: [docs/models.md](docs/models.md),
[docs/measured-results.md](docs/measured-results.md), [docs/warm-start.md](docs/warm-start.md)
and [docs/disk-layout.md](docs/disk-layout.md).

## Troubleshooting

The `status=` line the launcher prints, every status code and its fix:
[docs/troubleshooting.md](docs/troubleshooting.md).

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

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), base commit `0bd0ec6` (b10057) -
upstream copyright and license preserved; source releases keep all upstream notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

MoE-Direct additions (c) 2026 tmxkzm1925-max, released under the [MIT License](LICENSE). The
MoE-Direct name identifies this project and its official builds - see
[TRADEMARKS.md](TRADEMARKS.md). If you use this work, please cite it
([CITATION.cff](CITATION.cff)). Archived releases carry a DOI:
[10.5281/zenodo.21739367](https://doi.org/10.5281/zenodo.21739367).

**Thanks.** This project was built by one person with a great deal of machine help, and the help
was not incidental. Anthropic's Claude and OpenAI's GPT models did design work, implementation and,
just as usefully, adversarial review of each other's output, under human direction and with every
change gated by the verification this project's documents describe. Having a second and a third
reader who never got tired is most of the reason the checking here is as strict as it is.

Thanks are owed as well to the teams whose models this was measured against: Qwen, DeepSeek,
Moonshot AI, OpenAI for gpt-oss, and Mistral. None of them are affiliated with this project and
none of them have endorsed it. They published weights that one person with one desktop could
actually study, and without that there would have been nothing here to run.
