# How the direct-read path works

[Back to the README](../README.md)

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
  [What gets written to your disk](disk-layout.md#what-gets-written-to-your-disk)).
- **Not the sampling.** Temperature, seeds and chat templates behave as stock llama.cpp; the
  served API is the documented subset described in
  [Connecting a client](clients.md#connecting-a-client).

That boundary is why the scoped [token-parity result](measured-results.md) is even possible: the
compute graph is the stock one - only the storage path underneath the expert tensors changed.

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
of this changes what is computed, only where the bytes come from; the exact scope is stated in
[Measurements](measured-results.md).

The launcher measures your machine instead of assuming ours. A short read-only sweep picks the queue
depth for your drive, and the cache budget is sized from your installed RAM and the model's own
geometry. Both are printed before you start, and both stay overridable. What you get at the end is
an ordinary local `llama-server` endpoint on loopback, speaking the implemented OpenAI-compatible
subset (see [Connecting a client](clients.md#connecting-a-client)).

Each of those techniques is written up in full, with the problem it solves, what it does, how it
behaves and what was measured, in
[The techniques, and what they do](../TECHNICAL.md#the-techniques-and-what-they-do). Why the design
looks like this, which alternatives were measured and rejected on data, and what was underneath each
decision, is in a technical note with a DOI:
[10.5281/zenodo.21739367](https://doi.org/10.5281/zenodo.21739367).

Next: [TECHNICAL.md](../TECHNICAL.md) - each technique in full.
