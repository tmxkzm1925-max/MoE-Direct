# v0.3.1 engine source and build reproduction

This directory contains the build procedure and recorded reference for the
v0.3.1 engine. The patch reconstructs the complete release source tree from the
pinned llama.cpp base. The build procedure compares two clean builds of 17
named binaries: 16 runtime files used by the bundle and the unshipped
`moe-direct-selftest.exe`.

| Item | Identity |
| --- | --- |
| Upstream | llama.cpp `b10057` |
| Base commit | `0bd0ec60998d0f71ec45471b633bf2403ac81956` |
| Release source tree | `77e6cb26213d7719db707980b17168a81ad97691` |
| Patch | [moedirect-v0.3.1-b10057.patch](../moedirect-v0.3.1-b10057.patch) |
| Patch SHA-256 | `236f5345da709bb151a3fb450014632e0691feec5076617b5efb5cdea85a9c8d` |
| Catalog source identifier | `moe-direct-v0.3.1-src-77e6cb26` |

The catalog source identifier is a binding recorded in the bundle; its presence
does not establish a public Git tag. The tree above identifies v0.3.1. It does
not reconstruct the earlier v0.3-preview source state.

The release build record contains two matching clean builds, 17/17 identical
named outputs, and a full engine selftest result of 1484/1484 with exit code 0.
[BUILD_REFERENCE.txt](BUILD_REFERENCE.txt) preserves that run's output hashes.
The bundle's `BUILD_RECEIPT.txt` records the same source identity and separates
the supplied CUDA files from these build outputs.

## Files and scope

- `build_closure.ps1`: the unchanged script used for the recorded two-build
  comparison. Supply the explicit paths below; its historical defaults name
  the build machine's directories.
- `BUILD_REFERENCE.txt`: the recorded result, not an output path for your run.
- `reference/make_bundle.ps1`: an unchanged assembler snapshot used only as
  text by the closure script's `-SelfCheck` to check its runtime file list.
  It is not an independently packaged bundle-assembly entry point; do not run
  it to build this source package.

The CUDA-off build reproduces only the named set in `BUILD_REFERENCE.txt`.
Nine CPU backend DLLs are included in that set. Four additional runtime files
are supplied separately for the selftest and the release bundle:

| Supplied file | SHA-256 | Recorded provenance |
| --- | --- | --- |
| `ggml-cuda.dll` | `ab902c388552bf911e64d9b4471f62d1a2d124260f03e74b4c85180be5611838` | Project CUDA build recorded as `rbc4-clean` for the version 6 RC |
| `cublas64_13.dll` | `f1d500d0cd892f5b8c6b6cdbffd82d0c55d5f5427215668e7ceb55aeeccc1b63` | NVIDIA redistributable archive recorded as `cudart.zip` |
| `cublasLt64_13.dll` | `b592cd016d7673e9cb97716a22b27c4010ee635377a3ba28f37070a9bdb76a68` | Same NVIDIA archive |
| `cudart64_13.dll` | `b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a` | Same NVIDIA archive |

These four files are not reproduced by this procedure. Obtain the matching
bytes from the extracted v0.3.1 bundle; the script verifies every supplied
file's hash before using it. A CUDA source rebuild and a byte-identical rebuild
of the entire ZIP are outside this procedure's claim.

## Reconstruct and verify the source

Use Git and Windows PowerShell 5.1 or later. In the commands below, replace the
two initial directories with your checkout and a fresh work directory. Keep
the patch outside the engine checkout so it cannot become part of that tree.
The clone command below needs network access; an existing local checkout
containing the exact base commit can also serve as `$BaseRepo`.

```powershell
$PublicRepo = 'C:\work\moe-direct'
$Workspace = 'C:\work\moe-direct-v031-reproduction'
$ErrorActionPreference = 'Stop'
$Base = '0bd0ec60998d0f71ec45471b633bf2403ac81956'
$ExpectedTree = '77e6cb26213d7719db707980b17168a81ad97691'
$ExpectedPatch = '236f5345da709bb151a3fb450014632e0691feec5076617b5efb5cdea85a9c8d'
$Patch = Join-Path $PublicRepo 'patches\moedirect-v0.3.1-b10057.patch'
$Repro = Join-Path $PublicRepo 'patches\v0.3.1'
$BaseRepo = Join-Path $Workspace 'base'
$Source = Join-Path $Workspace 'source'
if (Test-Path -LiteralPath $Workspace) { throw 'Choose a fresh Workspace directory.' }
if ((Get-FileHash -LiteralPath $Patch -Algorithm SHA256).Hash.ToLower() -ne $ExpectedPatch) {
    throw 'Patch SHA-256 differs from the release pin.'
}
New-Item -ItemType Directory -Path $Workspace | Out-Null
git clone --depth 1 --branch b10057 --no-checkout --config core.autocrlf=false https://github.com/ggml-org/llama.cpp $BaseRepo
if ($LASTEXITCODE -ne 0) { throw 'Base clone failed.' }
$BaseHead = git -C $BaseRepo rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $BaseHead -ne $Base) { throw 'Base commit differs.' }
git clone --local --no-hardlinks --no-checkout --config core.autocrlf=false $BaseRepo $Source
if ($LASTEXITCODE -ne 0) { throw 'Source clone failed.' }
git -C $Source checkout --detach $Base
if ($LASTEXITCODE -ne 0) { throw 'Base checkout failed.' }
git -C $Source -c core.autocrlf=false apply --index --check $Patch
if ($LASTEXITCODE -ne 0) { throw 'Patch check failed.' }
git -C $Source -c core.autocrlf=false apply --index $Patch
if ($LASTEXITCODE -ne 0) { throw 'Patch application failed.' }
$Tree = git -C $Source write-tree
if ($LASTEXITCODE -ne 0 -or $Tree -ne $ExpectedTree) { throw "Source tree differs: $Tree" }
Write-Host "Source tree verified: $Tree"
```

The expected result is the full tree ID above. The patch changes 77 paths:
26 modified upstream files and 51 added files. No compilation is needed for
this source check. The resulting working tree and index intentionally contain
the applied patch; no commit is required.

## Check the build procedure without compiling

Continue in the same PowerShell session:

```powershell
$Closure = Join-Path $Repro 'build_closure.ps1'
$Assembler = Join-Path $Repro 'reference\make_bundle.ps1'
$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
& $PowerShellExe -NoProfile -File $Closure -SelfCheck -Patch $Patch -Assembler $Assembler
if ($LASTEXITCODE -ne 0) { throw 'Build closure selfcheck failed.' }
```

This uses stub files to check successful comparison, rejection of mismatches
and missing files, patch pinning, supplied-runtime validation, rejection of a
failing selftest, and agreement with the assembler's named set. It does not
compile an engine or run a model. The expected result is
`BUILD_CLOSURE SELFCHECK: PASS (17/17 cases)`.

## Run the two clean builds

The recorded toolchain was Visual Studio 2022, MSVC toolset 14.44.35207
(compiler 19.44.35228), CMake 3.31.6-msvc6, Ninja 1.12.1, and Windows SDK
10.0.26100.0. The script needs `git`, `cmake`, `ninja`, and `cl`; it can import
the installed Visual Studio environment. Select the required toolchain in an
x64 developer shell before running when multiple toolchains are installed.

Set `$Bundle` to the directory containing `Start-MoeDirect.ps1` and the runtime
DLLs in an extracted matching v0.3.1 bundle. This directory must contain the
four supplied files listed above. Keep `$BuildWork` dedicated to
this procedure: the script removes and recreates its `src`, `build`, `binA`,
and `binB` subdirectories. Do not point it at a source checkout, release
directory, or directory containing other work.

```powershell
$Bundle = 'C:\work\moe-direct-v0.3.1-win-x64'
$RuntimeDir = $Bundle
$CudaBackend = Join-Path $RuntimeDir 'ggml-cuda.dll'
$BuildWork = Join-Path $Workspace 'two-clean-builds'
$RunReference = Join-Path $Workspace 'BUILD_REFERENCE.this-run.txt'
if (Test-Path -LiteralPath $BuildWork) { throw 'Choose a fresh BuildWork directory.' }
if (Test-Path -LiteralPath $RunReference) { throw 'Choose a fresh reference output path.' }
& $PowerShellExe -NoProfile -File $Closure `
    -BaseRepo $BaseRepo -Patch $Patch -WorkDir $BuildWork `
    -Reference $RunReference -Assembler $Assembler `
    -CudaBackend $CudaBackend -NvidiaRuntimeDir $RuntimeDir
if ($LASTEXITCODE -ne 0) { throw 'Build closure failed; do not treat a partial output as a reference.' }
```

The procedure checks the pinned source tree before each build. It sets
`GGML_CUDA=OFF`, dynamic backends and all CPU variants on, native CPU selection
and LTO off, shared libraries on, and Release mode. `/Brepro` applies to EXE,
SHARED, and MODULE linker flags. It builds twice at the same paths, with
checkout line-ending conversion disabled and embedded UI disabled.

Success requires `DETERMINISTIC: 17/17 identical`, the full selftest exiting
zero, and a newly written reference file. For the recorded release run, the
full selftest reported 1484/1484. Supplied CUDA files are copied into a fresh
validation runtime and remain outside `binA`, `binB`, and the reference list.
The selftest requires no model files.

Compare `BUILD_REFERENCE.this-run.txt` with the published `BUILD_REFERENCE.txt`
separately if you want to check the release output hashes. Agreement between
your two local builds does not automatically mean agreement with the release
hashes: compiler/SDK versions, source history length (the recorded base was a
depth-one clone), and absolute build paths are relevant inputs. The script
checks the two local builds against each other; it does not silently replace
the published reference or assert cross-machine byte identity.

The source-only proof and the two-build procedure are separate checks. Source
tree agreement establishes the source identity. The recorded binary claim is
limited to the 17 named files and the stated build conditions.
