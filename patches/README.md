# Engine patches: MoE-Direct on llama.cpp b10057

This directory publishes the engine delta behind the release zips whose patch
is out - one patch per distinct engine revision, not per release number: a
release that changes only the launcher or the docs reuses the previous engine
tree (v0.2.3 ships the v0.2.2 engine unchanged), and the initial v0.2 predates
this directory. **Not every release is covered.** The latest published patch is
v0.2.2, and the engine in the current v0.3-preview zip is a later revision
whose delta is not out yet; the first section below is what stands in for it
until v0.3.1. The launcher, repacker, catalog and expectation files are
published at the repository root; this is the remaining piece: what was changed
inside the engine. The revisions are listed newest first, and the earlier ones
are kept because their zips are still downloadable.

## v0.3-preview - not published yet

The engine in the v0.3-preview zip carries the virtual repack path, and its
delta is **not** in this directory. It is scheduled for v0.3.1, as a patch
against the same pinned base commit plus the matching fork branch, on the
terms every section below is held to.

What exists in the meantime is narrower and worth stating exactly.
`BUILD_RECEIPT.txt`, at the root of this repository and inside the bundle,
records the SHA-256 and the byte size of each of the seven engine source
files that make up that state, together with the toolchain and the hashes of
every shipped build output. What it does not carry is a tree id: the
v0.3-preview engine tree was never committed, so no tree id exists for it,
and the mechanical patch-to-tree proof the v0.2.x sections give cannot be
given for this release. Per-file hashes identify the source state; they do
not let you reconstruct it. That is exactly the gap v0.3.1 closes.

One build difference is already visible in that receipt, and it is worth
knowing before you read the **Building** section below. The v0.2.x zips carry
the stock `ggml-cuda.dll` from the upstream `b10057` release; the v0.3-preview
zip ships one built here instead, listed among its own build outputs, so this
build had the CUDA backend enabled rather than switched off. The exact
configure line comes with the patch.

## v0.2.2 - what exactly this is (latest published patch)

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
seals the shipped set is the SHA manifest inside the zip, and a build receipt that
machine-binds base commit, tree id and patch hash in one record is planned for a
future release. The boundaries stated under **Building** below apply to this patch
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

## Building

This section describes the v0.2.x builds, the ones the patches above
reproduce. The v0.3-preview build differs, most visibly in the CUDA backend;
`BUILD_RECEIPT.txt` and the v0.3-preview section above are its record until
its patch lands.

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

If you would rather read the source than apply a patch, the identical tree is
pushed as a branch on a fork, one branch per engine revision, one commit on
top of the pinned base, carrying exactly the tree that revision's patch
reproduces (releases that reuse an engine tree share its branch).
The patch and the branch are cross-evidence for each other:
[`tmxkzm1925-max/llama.cpp`, branch `moe-direct-v0.2.2`](https://github.com/tmxkzm1925-max/llama.cpp/tree/moe-direct-v0.2.2)
(tree `38df4497...`) for the latest published patch, and branch
[`moe-direct-v0.2.1`](https://github.com/tmxkzm1925-max/llama.cpp/tree/moe-direct-v0.2.1)
(tree `32a97db0...`) for the one before it. The v0.3-preview engine has no
branch yet; it gets one with its patch.

## Status

A mainline llama.cpp PR is in preparation. It will be a rebased, split series
with per-change rationale, not this single patch. Until then, this file plus
the base pin is the complete, verifiable statement of what the engine in the
v0.2.x releases runs, and v0.3.1 brings the v0.3-preview engine onto the same
footing.

llama.cpp is by Georgi Gerganov and contributors, and this project exists on
top of that work; see `THIRD_PARTY_NOTICES.md` at the repository root.
