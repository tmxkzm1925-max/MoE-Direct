# Code Signing Policy - MoE-Direct

Status: the application to the SignPath Foundation's free open-source code
signing program was declined at the project's current stage (2026-08-09; the
program weighs established usage metrics, which a project this young does not
have yet). A re-application is possible once those exist. All releases to date
are unsigned; the README and the release notes say so plainly. Everything
below describes what would be signed once a signing relationship exists - and
the runtime integrity chain in "Integrity beyond the signature" holds with or
without one.

This document describes what gets signed in a MoE-Direct release, how the
artifacts are built, and how signing is controlled. It exists so that anyone
(including the signing service) can evaluate the chain from source to signed
binary.

## What is signed

Signed per release (once signing is available):

- `llama-server.exe`, `llama.exe`, `llama-cli.exe` and the other
  project-built tool executables in the bundle
- the project-built engine DLLs (`ggml-*.dll`, `llama-server-impl.dll`)
- `Start-MoeDirect.ps1` and `launcher_selftest.ps1` (PowerShell)

Explicitly not signed by this project (already signed by their vendors,
redistributed unmodified):

- NVIDIA CUDA runtime DLLs (`cudart64_*.dll`, `cublas64_*.dll`,
  `cublasLt64_*.dll`)
- the embeddable CPython runtime under `repacker/python/`

## Where the binaries come from

- The engine is built locally from a pinned upstream llama.cpp tree plus a
  published patch series. This repository publishes the complete patch and a
  mechanical proof that the patch reproduces the shipped binaries' source
  tree (see `patches/` and TECHNICAL.md). Build configuration: MSVC/Ninja,
  Release, `GGML_CUDA=OFF`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`.
- The launcher and repacker are plain-text source in this repository; the
  bundle copy is byte-identical to the repository copy.
- Because signing changes the bytes of a PE file, reproducibility statements
  in the docs refer to the pre-signature binaries; each signed release will
  publish both the pre-signature hashes (reproducibility surface) and the
  post-signature hashes (download integrity surface, `SHA256SUMS.txt`).

## Release and approval process

- Solo maintainer (`tmxkzm1925-max`). Every release is cut manually.
- A release must pass the project's gates before a signing request is
  submitted: launcher selftest, repacker selftest, engine selftest, and the
  bundle inventory check (`bundle_manifest.json`, which the launcher
  verifies in full, both directions, on every start).
- One signing request per release, submitted by the maintainer via the
  SignPath web interface or PowerShell module, and approved manually.
  Releases are infrequent (typically weeks apart).

## Key protection

The project holds no signing keys. Private keys stay in the signing
service's HSM (SignPath Foundation); the maintainer only submits artifacts
and downloads signed results. If the signing relationship ends, releases
simply return to unsigned status - no key material exists to leak.

## Integrity beyond the signature

The signature covers distribution trust. Independent of it, the bundle
carries its own integrity chain, enforced at runtime:

- `bundle_manifest.json`: every file in the bundle is hash-listed and the
  launcher refuses to start on any mismatch, extra or missing file
- repacked model data is SHA-256 verified against the source model in full
  (every record, no sampling) and served fail-closed
- `SHA256SUMS.txt` accompanies every release zip

## AI involvement

This project is developed in an AI-assisted workflow under human direction,
with every change gated by the verification described in TECHNICAL.md. The
README discloses this up front; it applies to build and release tooling as
well.
