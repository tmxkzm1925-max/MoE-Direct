# Engine patches: MoE-Direct on llama.cpp b10057

This directory publishes the engine delta behind the release zips whose patch
is out - one patch per distinct engine revision, not per release number: a
release that changes only the launcher or the docs reuses the previous engine
tree (v0.2.3 ships the v0.2.2 engine unchanged), and the initial v0.2 predates
this directory. **Not every release is covered.** The latest patch here is
v0.3.1. It reconstructs that release's source tree; the historical
v0.3-preview source state remains identified by its receipt rather than a
reconstructible patch. The launcher, repacker, catalog and expectation files are
published at the repository root; this is the remaining piece: what was changed
inside the engine. The revisions are listed newest first, and the earlier ones
are kept because their zips are still downloadable.

## v0.3.1 - source and build reference

- **Base**: llama.cpp release `b10057`, commit `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- **Patch**: [moedirect-v0.3.1-b10057.patch](moedirect-v0.3.1-b10057.patch),
  77 paths: 26 modified upstream files and 51 additions.
- **Patch SHA-256**: `236f5345da709bb151a3fb450014632e0691feec5076617b5efb5cdea85a9c8d`.
- **Reconstructed tree**: `77e6cb26213d7719db707980b17168a81ad97691`.
- **Source and build instructions**: [v0.3.1/README.md](v0.3.1/README.md),
  with the unchanged build procedure and its recorded 17-file hash reference.

The recorded CUDA-off build produced matching outputs across two clean builds
for 16 shipped runtime files plus the unshipped engine selftest. Its full
selftest reported 1484/1484. `ggml-cuda.dll` and three NVIDIA runtime DLLs were
supplied separately; their exact hashes and provenance are listed in the
instructions. Those four files and the whole ZIP are outside the two-build
reproduction claim.

The root `BUILD_RECEIPT.txt` binds this source tree and patch to the recorded
build outputs. Its catalog identifier `moe-direct-v0.3.1-src-77e6cb26` is not
an assertion that a public Git tag exists. No v0.3.1 fork URL is asserted here.

## v0.3-preview - historical source boundary

The engine in the v0.3-preview zip carries the virtual repack path, and its
exact historical delta is **not** in this directory. The v0.3.1 patch above
provides the later release tree; it does not reconstruct the earlier preview
source byte for byte.

The preview's record is narrower.
`BUILD_RECEIPT.txt` inside the v0.3-preview bundle
records the SHA-256 and the byte size of each of the seven engine source
files that make up that state, together with the toolchain and the hashes of
every shipped build output. That receipt carries per-file hashes rather than
a complete recorded tree ID, and no corresponding reconstructible source
snapshot is supplied here. The mechanical patch-to-tree proof below cannot be
given for this release. Per-file hashes identify the source state; they do
not let you reconstruct it. The current root receipt separates the v0.3.1
record from retained historical generations.

One build difference is already visible in that receipt, and it is worth
knowing before you read the **Building** section below. The v0.2.x zips carry
the stock `ggml-cuda.dll` from the upstream `b10057` release; the v0.3-preview
zip ships one built here instead, listed among its own build outputs, so this
build had the CUDA backend enabled rather than switched off. The v0.3.1
build instructions describe their own release and supplied-runtime boundary.

## v0.2.2 - what exactly this is

- **Base**: llama.cpp release `b10057`, commit `0bd0ec60998d0f71ec45471b633bf2403ac81956` -
  the same base commit v0.2.1 was built on.
- **Patch**: `moedirect-v0.2.2-b10057.patch` - one reviewed patch, 26 files,
  SHA-256 `dc4d6a31bedd13195705b02ad9942e2938f080d8d401040547408da223e8b2c3`.
- **What moved since v0.2.1**: the engine side of the arch-template path - the frozen
  table of approved architecture templates, and the independent regeneration of the
  expected tensor set that has to agree with a derived expectation file before the
  existing seals run - lands in `ggml/include/ggml-moe-direct.h`,
  `ggml/src/ggml-moe-direct.cpp` and `src/llama-model.cpp`, with the matching work in
  `tools/moe-direct-selftest/`. The pinned-catalog seal path is updated in those same
  files rather than duplicated, so a model the catalog pins takes the route it took in
  v0.2.1. One file is new - `tools/moe-direct-selftest/openarch_gate_c.py`, the gate
  script for that path - and it is what takes the count from 25 files to 26.
- **Binding to the shipped binaries**: applying this patch to the base commit
  reproduces the source tree with git tree id
  `38df4497b8dbe62528ec5d2839d4dd7e2c82a2f0`, byte for byte - the same tree id
  recorded for the source state that built the v0.2.2 engine binaries. The proof
  is mechanical and does not need our machine:

  ```bash
  # keep the patch OUTSIDE the clone - if it sits inside, `git add -A` would
  # stage the patch file itself and the tree id would not match
  curl -LO https://raw.githubusercontent.com/tmxkzm1925-max/moe-direct/main/patches/moedirect-v0.2.2-b10057.patch
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  git checkout 0bd0ec60998d0f71ec45471b633bf2403ac81956
  git apply --check ../moedirect-v0.2.2-b10057.patch   # applies cleanly
  git apply ../moedirect-v0.2.2-b10057.patch
  git add -A
  git write-tree    # prints 38df4497b8dbe62528ec5d2839d4dd7e2c82a2f0
  ```

Patch to tree is the whole of that claim, and it is worth being exact about where it
stops: nothing above binds that tree to the bytes of the shipped executables. What
seals the shipped set is the SHA manifest inside the zip. The v0.3.1 receipt
now binds those identities for its own release; it does not retroactively
change the v0.2.x evidence. The boundaries stated under **Building** below apply to this patch
unchanged.

## v0.2.1 - what exactly this is

- **Base**: llama.cpp release `b10057`, commit `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- **Patch**: `moedirect-v0.2.1-b10057.patch` - one reviewed patch, 25 files,
  SHA-256 `3568a8c22a7c9a298c77cd07c0a81d946e63af2b12139facb072e155c29c9075`.
- **Binding to the shipped binaries**: applying this patch to the base commit
  reproduces the source tree with git tree id
  `32a97db0d9941a0f302b4d6ca6200c964c41b1f6`, byte for byte - the same tree id
  recorded for the source state that built the v0.2.1 engine binaries. The
  proof is mechanical and does not need our machine:

  ```bash
  # keep the patch OUTSIDE the clone - if it sits inside, `git add -A` would
  # stage the patch file itself and the tree id would not match
  curl -LO https://raw.githubusercontent.com/tmxkzm1925-max/moe-direct/main/patches/moedirect-v0.2.1-b10057.patch
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  git checkout 0bd0ec60998d0f71ec45471b633bf2403ac81956
  git apply --check ../moedirect-v0.2.1-b10057.patch   # applies cleanly
  git apply ../moedirect-v0.2.1-b10057.patch
  git add -A
  git write-tree    # prints 32a97db0d9941a0f302b4d6ca6200c964c41b1f6
  ```

## Building v0.2.x

This section describes only the v0.2.x builds. For v0.3.1, use
[the versioned instructions](v0.3.1/README.md). The v0.3-preview build differs,
most visibly in the CUDA backend; its bundle receipt and the historical
section above remain its record.

The shipped binaries were built with MSVC 14.44 (Visual Studio 2022 Build
Tools), CMake and Ninja, in Release, with:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
ninja -C build
```

The CUDA backend DLLs in the zip (`ggml-cuda.dll`, `cublas64_13.dll`,
`cublasLt64_13.dll`, `cudart64_13.dll`) are the stock ones from the official
llama.cpp `b10057` Windows release, carried unmodified; the patch does not
touch CUDA sources.

Three honest boundaries. One more first: the delta contains a handful of absolute
development-machine paths (`D:\moe-tools\...`, `D:\moe-models\...`) - in comments,
in one test tool's `--help` string, and in one core fallback default inside
`ggml_moe_direct_seal()`. They carry no personal information. The fallback is
inert in the shipped configuration because the launcher always supplies its own
value and overrides it; only a bare-engine invocation without that environment
would ever see it. All of them are preserved because this patch must reproduce
the shipped tree byte for byte - cleaning them here would break the very proof
this file exists to give. They will be cleaned in the mainline PR
series, where the tree is new anyway. Second, bit-identical binary reproduction is not
claimed - compiler and environment differences change bytes; what seals the
shipped set is the SHA manifest inside the zip, and what this patch proves is
the source lineage. Third, the patch is published exactly as it shipped, so
the delta files carry no added license headers; the whole delta is released
under this repository's MIT license, and header cleanup will happen in the
mainline PR series.

## The same tree, browsable

The v0.2.x source revisions are also available as branches on a fork, one
branch per engine revision, one commit on top of the pinned base, carrying
exactly the tree that revision's patch reproduces (releases that reuse an
engine tree share its branch).
For the v0.2.x source revisions, the patch and branch are cross-evidence:
[`tmxkzm1925-max/llama.cpp`, branch `moe-direct-v0.2.2`](https://github.com/tmxkzm1925-max/llama.cpp/tree/moe-direct-v0.2.2)
(tree `38df4497...`), and branch
[`moe-direct-v0.2.1`](https://github.com/tmxkzm1925-max/llama.cpp/tree/moe-direct-v0.2.1)
(tree `32a97db0...`). The historical v0.3-preview tree has no branch identified
here. For v0.3.1, the patch and complete tree ID above are the source entry point.

## Status

A mainline llama.cpp PR is in preparation. It will be a rebased, split series
with per-change rationale, not this single patch. Until then, this file plus
the base pin reconstruct the source trees identified above for v0.2.x and
v0.3.1. The historical v0.3-preview limitation remains explicit.

llama.cpp is by Georgi Gerganov and contributors, and this project exists on
top of that work; see `THIRD_PARTY_NOTICES.md` at the repository root.
