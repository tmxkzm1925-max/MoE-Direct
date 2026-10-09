# Models

[Back to the README](../README.md)

## Supported models

MoE-Direct does not download weights. The catalog reference measurements apply to the exact
repositories and revisions below. A structurally compatible file with different source bytes
may still run, but it is labelled unvalidated and receives no catalog prefetch or reference
performance claim.

Every profile carries **two independent gates**, and one never implies the other:

- **Format gate** - the repack of this model was verified byte for byte and the verify report
  passed. This is what protects your output. That wording describes the packed path, which is what
  the gate was run on. A virtual repack copies nothing, so it has no byte comparison to make: its
  equivalent evidence is the plan report and the manifest, gated by the launcher and re-derived
  independently by the engine. The tier and gate verdicts in this table describe the packed path.
  In virtual mode the screen shows copy integrity as `N/A`, format gate
  `PASS (virtual plan gate 8/8)`, serving validation as `unvalidated`, and performance
  as `[unmeasured]`; the reference measurements are labelled as packed-path results.
- **Performance gate** - a serving run of this model passed the frozen release gate on the
  reference machine. This is only about speed.

| Tier | Meaning |
|---|---|
| `reference-validated` | Format gate **and** performance gate passed on the reference machine. |
| `format-validated` | Format gate passed. Any speed number shown is an observation, not a gate pass. |
| `experimental` | Neither gate established. Your own risk. No pinned catalog entry ships at this tier; a profile derived by [Running an unlisted model (experimental)](#running-an-unlisted-model-experimental) is exactly this tier. |

| Model (profile id) | Source | Experts | Repacked expert store | Min cache budget | Tier | Launcher prefetch in v0.3.1 |
|---|---|---:|---:|---:|---|---|
| **Qwen3.5-122B-A10B Q4_K_M** (`qwen35-122b-nonextn`) **<- start here** | [bartowski/Qwen_Qwen3.5-122B-A10B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF) rev `fec8b222a2eddc3346d6b6d7f7c85efea93cd6bf` | 256 (top-8) | 72.8 GB | 8192 MB | `reference-validated` | Catalog ON (K=8, N=4; exact source pin required) |
| gpt-oss-120b MXFP4 (`gpt-oss-120b`) | [ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) rev `8d158cefb5f175c6f8842bbd8f68eca54d951ab4` | 128 (top-4) | 61 GB | 8192 MB | `format-validated` | OFF (held) |
| Qwen3.5-35B-A3B Q4_K_M (`qwen35-35b`) | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) rev `bc014a17be43adabd7066b7a86075ff935c6a4e2` | 256 (top-8) | 19.5 GB | 4096 MB | `format-validated` | OFF (held) |
| Qwen3.5-397B-A17B Q4_K_M, 6 shards (`qwen35-397b`) | [unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) rev `da33c16fa4440f831149fcf53b98a22bc07785e5` | 512 (top-10) | shown by the launcher before it writes | 8192 MB | `format-validated` | OFF (held) |
| Kimi K2.6 447 GB mixed-quant (`kimi-k2.6-ram-447gb`) | [baa-ai/Kimi-K2.6-RAM-447GB-GGUF](https://huggingface.co/baa-ai/Kimi-K2.6-RAM-447GB-GGUF) rev `1e8bc2c2c759db5b4bb783965129d4e1e9182bc6` | 384 (top-8) | 436 GB | 10240 MB | `format-validated` | OFF (no engine prefetch adapter) |
| DeepSeek-V4-Flash-0731 MXFP4/Q8_0 (`deepseek-v4-flash`) | [bullerwins/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/bullerwins/DeepSeek-V4-Flash-0731-GGUF) rev `ed48c7a2df419aaa01e325521cd6f93464969641` | 256 (top-6) | shown by the launcher before it writes | 8192 MB | `format-validated` | OFF (held) |

Notes on this table:

- **The expert-store column is the packed path's cost.** It is what the repack writes when it
  writes a store. Choose the virtual repack mode instead and that column becomes a manifest and its plan
  report next to your GGUF (measured at about 6 MB for the 122B test model and about 16 MB for
  the 397B one). Model identity is unchanged, but the packed-path tier and performance
  verdicts do not transfer to a virtual run; its status is described above.
- **First time? Take Qwen3.5-122B.** It is the only `reference-validated` profile and the one the
  published speed number belongs to. Budget about 73 GB of disk for its expert store.
- **Prefetch column.** Qwen3.5-122B is the only catalog-fixed ON profile in v0.3.1,
  using K=8 / N=4 when the source pin and startup checks pass. The other five catalog
  profiles remain OFF in the launcher, including `init` requests; `adapt` cannot activate.
  The catalog stores evidence separately from activation, so an OFF row does not by itself
  mean the engine lacks an adapter. Kimi K2.6 is the exception here: the shipped engine has
  no `deepseek2` prefetch adapter. Its earlier paired-run observations remain historical
  evidence and do not enable prefetch in this release. This restriction concerns prefetch,
  not serving the model. Historical runs are retained under
  [Non-official observations](../TECHNICAL.md#non-official-observations).
- **DeepSeek-V4-Flash-0731 is new in v0.2.1**, and it is the newest architecture in the catalog
  rather than the most measured one. Its repack verified 33,024 of 33,024 record-part pairs, and a smoke run
  on the reference machine held 3.13-3.28 tok/s across four probes with zero fallback events
  (`PROBE`: budget 8192 MB, QD 8, prefetch off, ctx 8192). This is a historical observation,
  not a measurement of the v0.3.1 binary. Prefetch remains OFF under the current catalog
  hold; observation evidence alone does not qualify a live K/N setting. The launcher will accept a context up to 131072 on the custom path, but nothing
  above 8192 has been exercised here, so treat that headroom as untested rather than offered.
- **Text only.** This release validates text serving. Multimodal (mmproj) inputs are **not verified** -
  not blocked, just unlabelled. Reports welcome.
- **Where these values come from.** The catalog file `models.json` in the bundle is the single
  source of truth for the profile id, the repository and revision, the expert count and top-k, the
  minimum cache budget, the tier and the prefetch state; those columns are rendered from it. The
  **Repacked expert store** column is not a catalog field: it is the size a packed repack of that
  model recorded here - on the reference machine, except the 35B row, which was repacked on
  the second machine described in [TECHNICAL.md](../TECHNICAL.md). The 397B and DeepSeek rows
  have no recorded size yet. The launcher computes the exact size for *your* model and shows it in
  the repack plan before it writes.

## Running an unlisted model (experimental)

Since v0.2.3 the architecture-template path is enabled by default. The controls below matter only when you want to override or persist that default.

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

The setting is stored per Windows user account, next to your saved preset rather than inside it,
because the launcher has to know the answer *before* it has identified your model, and a preset is
bound to a model it has not identified yet.

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
[TECHNICAL.md](../TECHNICAL.md#serving-a-model-the-catalog-does-not-pin).

## From the roadmap

| Where we are | Status |
|---|---|
| Prefetch for the deepseek2 family, which is what Kimi K2.6 needs | **OFF in v0.3.1.** The shipped engine has no prefetch adapter for this architecture. The catalog therefore records `unverified` evidence, `off` activation, no K/N tuple, and a promotion hold. Earlier paired-run results remain historical observations; they do not establish current engine capability or an official performance gate pass. |
| DeepSeek-V4-Flash-0731 (284B, `deepseek4`) | **In the catalog since v0.2.1.** The architecture is native to the base commit this release is built on, its expert tensors are MXFP4, which is a layout the repacker already handles, and the repack verified 33,024 of 33,024 record-part pairs. A direct-read smoke run held 3.13-3.28 tok/s across four probes with zero fallback events (`PROBE`: reference machine, budget 8192 MB, QD 8, prefetch off, ctx 8192). It ships with the format gate passed, the performance gate unpassed and prefetch disabled, which is where the evidence actually stands; the prefetch depth search for this family is queued for a later release. |
| Any model of an already-supported architecture (`experimental` tier) | **Shipped in v0.2.2 behind `-ExperimentalArchTemplate`; on by default since v0.2.3.** The catalog still runs its six pinned models the way it always did; this adds a second way of passing the same checks for a GGUF of a known architecture, derived, repacked, verified and served labelled `experimental` with prefetch off - an honest warning, not a block and not a promise. Templates exist for three architectures (`gpt-oss`, `qwen35moe`, `deepseek2`); the end-to-end evidence is one model of one of them (`gpt-oss`), which is what keeps the label. The default moved because behind a switch nobody found it, not because the evidence grew: an architecture with no template is still refused, and the repack still stops for your approval. `-ArchTemplate off` turns it off for a run, and the model menu's `arch template:` row stores the choice. See [Running an unlisted model (experimental)](#running-an-unlisted-model-experimental). |
| Kimi K3 (2.8T class) | **Not yet ready, but moving.** Disk-space limits ruled out the MXFP4 build for now; a Kimi-K3 Q2_K_XL sits prepared on disk instead. Its repacker catalog entry is registered (2026-08-13) and does ship: the entry and its expectation file are in the zip and in this repository, in both `expects/` copies. Serving is what is not registered, in the launcher or in the engine, so an attempt to serve it fail-closes. The remaining chain is the real repack verify, a smoke run, and engine-side support for the `kimi-k3` architecture. We will do our best to bring it in at a format-validated grade or higher as soon as possible. (updated 2026-08-14) |

Why K3 is worth naming at all: at 2.8T-class sizes nothing that resembles a consumer machine can
hold the weights in memory, so keeping experts on disk and streaming the ones each token actually
uses is the *shape* of local execution we have found practical on the hardware this project
targets. Its expert format (MXFP4) is already one
the repacker handles today. That is a structural argument about the size class, not a claim that
this project will be first or fastest there.

Next: [Getting started](getting-started.md)
