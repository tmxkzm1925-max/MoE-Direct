# engine/

The MoE-Direct engine source: every llama.cpp file that the v0.3.2 patch creates or
modifies, in its patched state, at its path in the llama.cpp tree (78 files: 51 created,
27 modified).

- Upstream: [llama.cpp](https://github.com/ggml-org/llama.cpp) b10057, commit
  `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- Patch: [`patches/moedirect-v0.3.2-b10057.patch`](../patches/moedirect-v0.3.2-b10057.patch).
- License: [`LICENSE`](LICENSE) is the upstream MIT license, copied byte for byte from that
  commit; it covers the upstream code in these files. MoE-Direct's own additions are under
  the repository's [MIT License](../LICENSE).

This folder is a derived artefact. It is regenerated from the patch and never edited by
hand: a change goes into the patch first. It does not build on its own; to build, apply the
patch to a full checkout of b10057 (see [`patches/README.md`](../patches/README.md)).

Check that the folder matches the patch (POSIX shell, from the repository root):

```sh
git clone -c core.autocrlf=false https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 0bd0ec60998d0f71ec45471b633bf2403ac81956
git -C llama.cpp apply ../patches/moedirect-v0.3.2-b10057.patch
git -C llama.cpp status --short --untracked-files=all   # the 78 paths of this folder
(cd engine && find . -type f ! -name README.md -exec cmp {} ../llama.cpp/{} \;)
```

`cmp` prints nothing when every file here equals its patched upstream file. The project's
release check makes the same comparison by sha256, file for file.
