# The repacker, runnable from this checkout

`repack_experts.py` is byte-identical to the copy in the release zip, and the
`expects/` directory next to it carries the same eight expectation files the
zip ships in this exact script-relative location - the script resolves its
expectation catalog relative to its own path, so this layout is what makes a
plain checkout runnable (point it at a source GGUF with `--model`; see
`--help`). The launcher, by contrast, runs as part of the release bundle: it
needs the engine binaries and the bundle manifest from the zip, so the
checkout is an audit surface for it, not a launch surface.

One entry in the script's frozen `EXPECT_CATALOG` is ahead of its runtime:
`minimax-m27` has an approved digest and its expectation file ships **in this
repository** (both `expects/` locations), so the `OPEN_ARCH-ⓐ` catalog check
passes 9/9, and the full `--selftest` passes 64/64 from a checkout. The v0.2.1 zip carried only the other eight, which means the zip's
own `--selftest` reports 63/64 with exactly this missing-file failure - a
known issue of that bundle, listed in the release notes; serving that profile
fail-closes either way because the launcher and the engine do not register it
yet. The script is not edited here even for notes like this one, because its
bytes must stay identical to the shipped copy; the entry resolves properly in
the next bundle.
