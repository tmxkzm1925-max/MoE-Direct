# The repacker, runnable from this checkout

`repack_experts.py` is byte-identical to the copy in the release zip, and the
`expects/` directory next to it carries the same nine expectation files the
zip ships in this exact script-relative location - the script resolves its
expectation catalog relative to its own path, so this layout is what makes a
plain checkout runnable (point it at a source GGUF with `--model`; see
`--help`). The launcher, by contrast, runs as part of the release bundle: it
needs the engine binaries and the bundle manifest from the zip, so the
checkout is an audit surface for it, not a launch surface.

One entry in the script's frozen `EXPECT_CATALOG` used to be ahead of its
runtime: `minimax-m27` has an approved digest, and as of v0.2.2 its
expectation file ships in the zip as well as in this repository (both
`expects/` locations), so the `OPEN_ARCH-ⓐ` catalog check passes 9/9 and the
full `--selftest` passes from a checkout and from the zip alike - 65/65 as
the suite stands in v0.2.2, measured on the assembled bundle. (History: the
v0.2.1 zip carried only the other eight files, so that zip's own `--selftest`
reported 63/64 with exactly this missing-file failure - a known issue listed
in that release's notes. This is the resolution it promised.) Serving that
profile still fail-closes either way, because the launcher and the engine do
not register it yet; the expectation entry is repack-side only.
