# The repacker, runnable from this checkout

`repack_experts.py` is byte-identical to the copy in the release zip, and the
`expects/` directory next to it carries the same eight expectation files the
zip ships in this exact script-relative location - the script resolves its
expectation catalog relative to its own path, so this layout is what makes a
plain checkout runnable (point it at a source GGUF with `--model`; see
`--help`). The launcher, by contrast, runs as part of the release bundle: it
needs the engine binaries and the bundle manifest from the zip, so the
checkout is an audit surface for it, not a launch surface.

One entry in the script's frozen `EXPECT_CATALOG` is ahead of its artifacts:
`minimax-m27` has an approved digest but no expectation file ships yet, so
selecting it fail-closes today. It is a forward entry for a future release.
The script is not edited here even for notes like this one, because its bytes
must stay identical to the shipped copy; cleanups land with the next bundle.
