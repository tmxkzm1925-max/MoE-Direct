# The repacker, runnable from this checkout

`repack_experts.py` is byte-identical to the copy in the release zip, and the
`expects/` directory next to it carries the same ten expectation files the
zip ships in this exact script-relative location - the script resolves its
expectation catalog relative to its own path, so this layout is what makes a
plain checkout runnable (point it at a source GGUF with `--model`; see
`--help`). The launcher, by contrast, runs as part of the release bundle: it
needs the engine binaries and the bundle manifest from the zip, so the
checkout is an audit surface for it, not a launch surface.

Two entries in the script's frozen `EXPECT_CATALOG` sit ahead of their
runtime: `minimax-m27` and `kimi-k3-ud-q2kxl` have approved digests and their
expectation files ship in the zip as well as in this repository (both
`expects/` locations), but neither is served, because the launcher and the
engine do not register them yet. Those entries are repack-side only, and a
serving attempt fail-closes either way.

With the catalog and the files in agreement, the `OPEN_ARCH-ⓐ` catalog check
passes 10/10. The full `--selftest` as the suite stands in v0.3-preview passes
**90/90 on the assembled bundle**, and **89/90 from a plain checkout of this
repository**.
The one failure is `v3-⑱ launcher parser contract copy
check`, which compares the parser contract copy in the repacker against the
literal strings in `Start-MoeDirect.ps1`. It looks for that file in the two
shapes it knows, a development tree and a bundle root, and this repository
puts it at a third, `launcher/Start-MoeDirect.ps1`. Finding neither known
shape, the check fails closed instead of skipping itself. The copy it guards
is not what is wrong there; the resolver simply does not know this layout yet,
and v0.3.1 registers the third shape. (History: the v0.2.1 zip carried only
eight of the expectation files, so that zip's own `--selftest` reported 63/64
with exactly a missing-file failure - a known issue listed in that release's
notes, resolved in v0.2.2.)
