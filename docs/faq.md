# Limitations and FAQ

[Back to the README](../README.md)

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
  [Non-official observations](../TECHNICAL.md#non-official-observations) - but nothing Vulkan
  ships in v0.2.2, and other backends (ROCm/HIP, unified-memory platforms) remain **untested**. CPU-only serving (`-ngl 0`) needs no GPU at all - that is how the 35B row in
  [TECHNICAL.md](../TECHNICAL.md) was measured. If you run this on other hardware, the
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
  [Running an unlisted model (experimental)](models.md#running-an-unlisted-model-experimental).
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
  because the prefix cache absorbs the repeat - see [Measured results](measured-results.md#measured-results).
- **K2.6 has not passed the performance gate.** It runs, it is coherent, it is honest about 1.03
  tok/s. Do not plan interactive work around it.
- **Text only.** Multimodal inputs are unverified.
- **No telemetry, and no automatic uploads.** Diagnostic logs are written locally and go nowhere
  unless you attach them to an issue yourself.

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

## Source

The launcher, the repacker and the expectation files are published in this
repository, and the shipped copies are the same bytes: `launcher/Start-MoeDirect.ps1`,
`launcher/Start-MoeDirect.cmd`, `repacker/repack_experts.py` and the nine `expects/*.expect.json`
files are byte-identical to the copies inside the release zip (verified by SHA-256 at publish
time). As of v0.2.2 the zip carries all nine, so the repacker's self-test passes from a checkout
and from the zip alike - 65/65, measured on the assembled bundle; the v0.2.1 zip's 63/64 was a
known issue of that bundle and this closes it - see [repacker/README.md](../repacker/README.md). `launcher/models.json` is the catalog the launcher consumes, exactly as the zip resolves
it - it is where each profile's `expect_sha256` pin lives, so the binding between the launcher and
the expectation files can be audited end to end. What runs from a plain checkout: the repacker
does (its `repacker/expects/` catalog ships in place - see [repacker/README.md](../repacker/README.md));
the launcher does not, since it needs the engine binaries and bundle manifest from the zip. The launcher's own test suite (978 checks as of
v0.2.2, all passing on the shipped launcher) is not yet published: it depends
on fixture files that need to be packaged for standalone use, and it will follow in a later
release. The engine is a patched llama.cpp build; the binaries are sealed by
the bundle's SHA manifest, and the full engine delta is published in [`patches/`](../patches/) as a
single reviewed patch against the pinned upstream commit, with a mechanical proof that it
reproduces the exact source tree the shipped binaries were built from; the same tree is browsable
as a [fork branch](https://github.com/tmxkzm1925-max/llama.cpp/tree/moe-direct-v0.2.2), one branch
per release. A rebased mainline PR series is in preparation.

## From the roadmap

| Where we are | Status |
|---|---|
| Wider hardware, wider OS | Windows only today. A Vulkan build of the same engine has been measured off-bundle - RTX 5080 at CUDA-parity decode, and a first AMD run (RX 9070 XT, RDNA4) at about 89 percent of that, its arms character-identical among themselves - so cross-vendor GPU support is now a hardening-and-tooling queue item rather than an open question; the numbers are under [Non-official observations](../TECHNICAL.md#non-official-observations). OS expansion is genuinely hard in the current test environment and will take time. Community ports are welcome. |

Next: [Troubleshooting and status codes](troubleshooting.md)
