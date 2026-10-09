#requires -Version 5.1
<#
  build_closure.ps1 - G4-0 determinism re-baselining (BUILD-CLOSURE).

  Proves the moedirect binary is REPRODUCIBLE: builds the sealed source
  (base b10057 0bd0ec6 + moedirect-v0.3.1.1-b10057.patch = tree 38a1abe5) TWICE with
  pinned determinism inputs and checks the two builds produce byte-identical
  critical binaries. The sealed source tree is NOT edited - every determinism
  fix is configure-time or procedural.

  Determinism inputs pinned here:
    * BUILD_NUMBER   : both builds clone from the SAME BaseRepo, so
                       `git rev-list --count HEAD` is identical (no source edit).
    * PE timestamp   : /Brepro appended to Release EXE + SHARED + MODULE linker flags (DL backends are MODULE libraries).
    * UI gzip mtime  : -DLLAMA_UI_GZIP=OFF (removes gzip-header mtime).
    * UI provisioning: -DLLAMA_USE_PREBUILT_UI=OFF (no network/version drift);
                       default -DLLAMA_BUILD_UI=OFF (API-only; webui unused by
                       G4-1a). -KeepUI -DistDir <dir> embeds a fixed prebuilt UI.
    * Path embedding : both builds reuse the SAME src+build path (sequential).
    * Checkout EOL   : the clone is made with core.autocrlf=false, so the build
                       source bytes are the tree's blob bytes (LF) whatever the
                       host's git default is (6h HR-R2f; with the host default
                       core.autocrlf=true the clone held CRLF files and
                       `git apply --index` refused them: "does not match index").

  Named set (6h HR-R2f): only the files the release takes from this build are
  hashed and compared - the 16 runtime binaries the assembler ships that this
  build produces, plus moe-direct-selftest.exe (the assembler's ABI probe and the
  engine's self-test evidence). A named file missing from a build fails the run
  by name. Fail-closed: a hash mismatch between the two builds, a missing named
  file, or a self-test that does not exit 0 ends the run with exit 1 and no
  reference list. `-SelfCheck` exercises that logic on stub files (no compiler,
  no clone, no build) and compares the named set with the assembler's list.

  Toolchain reference (from D: CMakeCache): Ninja / Release / MSVC 14.44.35207 /
  CUDA=OFF BACKEND_DL=ON CPU_ALL_VARIANTS=ON NATIVE=OFF LTO=OFF SHARED_LIBS=ON.

  Deployment boundary: authoring this script is AI-autonomous; RUNNING it (the
  compile) is the user. It auto-bootstraps the VS 2022 build environment, so a
  plain PowerShell works; if bootstrap fails, use the "x64 Native Tools Command
  Prompt for VS 2022". Default -BaseRepo is the existing D: tree (clone takes
  only its committed b10057 state, ignoring working-tree changes), so usually:

    powershell -File build_closure.ps1

  -BaseRepo may point at any local checkout CONTAINING commit 0bd0ec6 (a fresh
  `git clone https://github.com/ggml-org/llama.cpp` works; D:'s shallow tree
  works too - only A==B matters, not the absolute build number).
#>
param(
  [string]$BaseRepo = 'D:\moe-tools\llama.cpp-src-b10057',
  [string]$Patch   = "$PSScriptRoot\moedirect-v0.3.1.1-b10057.patch",
  [string]$WorkDir = "D:\moe-tools\build-closure",
  [string]$Reference = "$PSScriptRoot\BUILD_REFERENCE.txt",
  [string]$Assembler = "$PSScriptRoot\..\launcher\packaging\make_bundle.ps1",
  [string]$CudaBackend = 'D:\moe-tools\llama.cpp-src-b10057\build-rbc4-clean\bin\ggml-cuda.dll',
  [string]$NvidiaRuntimeDir = 'D:\moe-tools\llama-moedirect-v2',
  [switch]$KeepUI,
  [string]$DistDir,
  [switch]$SelfCheck
)
$ErrorActionPreference = 'Stop'

# Native commands: PS 5.1 turns their stderr into terminating errors under
# -EA Stop, so a git/cmake progress line (e.g. "Cloning into ...") would abort
# the script. Run natives under Continue and gate on $LASTEXITCODE instead.
function Native([string]$what, [scriptblock]$block) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  # Out-Host: the tool's stdout goes to the console, not into the caller's
  # return value (Build-Once returns the hash table).
  try { & $block | Out-Host } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

$BASE      = '0bd0ec60998d0f71ec45471b633bf2403ac81956'
$TREE      = '38a1abe5cc46f7422ac421f028c9844b38f286fc'
$PATCH_SHA = '21050a201081e78729ae4e896df81f63159b5a371368c7fa7c9dbd735cf6ce32'

# The named set: what the release takes from this build. 16 = the assembler's
# runtime list (bench/moe-direct/launcher/packaging/make_bundle.ps1
# $script:RUNTIME_FILES) less the four files this CUDA-off build does not
# produce ($NOT_BUILT_HERE, copied into the staging from elsewhere); +1 = the
# assembler's ABI probe ($script:ABI_SELFTEST_EXE), which is not shipped.
# -SelfCheck case 'named-set-matches-assembler' fails when the two lists drift.
$NAMED_SET = @(
  'ggml-base.dll',
  'ggml-cpu-alderlake.dll',
  'ggml-cpu-cannonlake.dll',
  'ggml-cpu-cascadelake.dll',
  'ggml-cpu-haswell.dll',
  'ggml-cpu-icelake.dll',
  'ggml-cpu-sandybridge.dll',
  'ggml-cpu-skylakex.dll',
  'ggml-cpu-sse42.dll',
  'ggml-cpu-x64.dll',
  'ggml.dll',
  'llama-common.dll',
  'llama-server-impl.dll',
  'llama-server.exe',
  'llama.dll',
  'moe-direct-selftest.exe',
  'mtmd.dll'
)
$SELFTEST_NAME  = 'moe-direct-selftest.exe'
$NOT_BUILT_HERE = @('ggml-cuda.dll','cublas64_13.dll','cublasLt64_13.dll','cudart64_13.dll')

# Runtime-only provenance: RC_6.json produced rbc4-clean backend; NVIDIA
# source A (cudart.zip), DECISION_LEDGER.md:3051 and reassembly_w2_20260831T124803Z
# bundle_manifest.after.json. These four supplied files are NOT reproduced here.
$RUNTIME_SHA = @{
  'ggml-cuda.dll' = 'ab902c388552bf911e64d9b4471f62d1a2d124260f03e74b4c85180be5611838'
  'cublas64_13.dll' = 'f1d500d0cd892f5b8c6b6cdbffd82d0c55d5f5427215668e7ceb55aeeccc1b63'
  'cublasLt64_13.dll' = 'b592cd016d7673e9cb97716a22b27c4010ee635377a3ba28f37070a9bdb76a68'
  'cudart64_13.dll' = 'b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a'
}

function Assert-RuntimeFile([string]$path, [string]$name) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "runtime file missing: $name ($path)" }
  $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
  if ($got -ne $RUNTIME_SHA[$name]) { throw "runtime provenance mismatch: $name sha256=$got (expected $($RUNTIME_SHA[$name]))" }
}

function New-ValidationRuntime([string]$bin, [string]$runtime, [string]$backend, [string]$nvidia) {
  # Check all inputs before creating the isolated runtime; recheck copied bytes
  # to refuse a source changed between validation and copying.
  [void](Get-NamedHashes $bin 'validation')
  foreach ($name in $NOT_BUILT_HERE) {
    $source = if ($name -eq 'ggml-cuda.dll') { $backend } else { Join-Path $nvidia $name }
    Assert-RuntimeFile $source $name
  }
  if (Test-Path -LiteralPath $runtime) { throw "validation runtime already exists: $runtime" }
  New-Item -ItemType Directory -Path $runtime | Out-Null
  foreach ($name in $NAMED_SET) { Copy-Item -LiteralPath (Join-Path $bin $name) -Destination (Join-Path $runtime $name) }
  foreach ($name in $NOT_BUILT_HERE) {
    $source = if ($name -eq 'ggml-cuda.dll') { $backend } else { Join-Path $nvidia $name }
    Copy-Item -LiteralPath $source -Destination (Join-Path $runtime $name)
    Assert-RuntimeFile (Join-Path $runtime $name) $name
    Write-Host "[runtime] supplied $name sha256=$($RUNTIME_SHA[$name]) (not reproduced)"
  }
  return $runtime
}

function Invoke-ValidationSelftest([string]$runtime, [string[]]$arguments = @()) {
  $self = Join-Path $runtime $SELFTEST_NAME
  $saved = @{}
  foreach ($key in @('TEMP','TMP','GGML_BACKEND_PATH')) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
  # TEMP on the system drive: with it on D: the census of P45h loses its read-latency
  # race (reviews/_diag_7_EP-B1-P45H_v7-epb1d.md). One fresh folder per runtime.
  $temp = Join-Path ([IO.Path]::GetTempPath()) ('build_closure_selftest_' + (Split-Path -Leaf $runtime))
  if (Test-Path -LiteralPath $temp) { throw "selftest temp already exists: $temp" }
  New-Item -ItemType Directory -Path $temp | Out-Null
  Write-Host "[selftest] temp $temp"
  $prevEap = $ErrorActionPreference
  Push-Location -LiteralPath $runtime
  try {
    [Environment]::SetEnvironmentVariable('TEMP', $temp, 'Process')
    [Environment]::SetEnvironmentVariable('TMP', $temp, 'Process')
    [Environment]::SetEnvironmentVariable('GGML_BACKEND_PATH', $null, 'Process')
    Write-Host "==== moe-direct-selftest (isolated validation runtime) ===="
    $ErrorActionPreference = 'Continue'
    & $self @arguments | Out-Host
    $selfrc = $LASTEXITCODE
    if ($null -eq $selfrc) { throw 'selftest produced no exit code' }
    Write-Host "[selftest] exit $selfrc"
    if ($selfrc -eq 0) { Remove-Item -Recurse -Force -LiteralPath $temp }
    return $selfrc
  } finally {
    $ErrorActionPreference = $prevEap
    Pop-Location
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
  }
}

# Refuse a patch whose bytes differ from the pin. Called before anything else.
function Assert-PatchPin([string]$path) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "patch not found: $path" }
  $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
  if ($got -ne $PATCH_SHA) { throw "patch sha256 mismatch: $got (expected $PATCH_SHA)" }
}

# Hash the named set in one build's output folder (flat, as the staging takes
# it). A named file that is absent fails the run, naming every absent file.
function Get-NamedHashes([string]$dir, [string]$tag) {
  $h = @{}
  $missing = @()
  foreach ($name in $NAMED_SET) {
    $p = Join-Path $dir $name
    if (Test-Path -LiteralPath $p -PathType Leaf) {
      $h[$name] = (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower()
    } else { $missing += $name }
  }
  if ($missing.Count -gt 0) {
    throw "[$tag] named file(s) missing from the build output ${dir}: $($missing -join ', ')"
  }
  return $h
}

# Compare the two builds over the named set, run the self-test, and write the
# reference list only when both are clean. Returns the exit code of the run.
function Complete-Closure([hashtable]$hA, [hashtable]$hB, [string]$refPath, [scriptblock]$selftest) {
  Write-Host "==== compare A vs B (named set, $($NAMED_SET.Count) files) ===="
  $mismatch = 0
  foreach ($f in $NAMED_SET) {
    if (-not $hA.ContainsKey($f) -or -not $hB.ContainsKey($f) -or $hA[$f] -ne $hB[$f]) {
      $mismatch++; Write-Host "  MISMATCH $f  A=$($hA[$f])  B=$($hB[$f])"
    }
  }
  $n = $NAMED_SET.Count
  if ($mismatch -ne 0) {
    Write-Host "[compare] $mismatch/$n differ - determinism NOT closed; no reference list written"
    return 1
  }
  Write-Host "[compare] DETERMINISTIC: $n/$n identical"

  $selfrc = & $selftest
  if ($selfrc -ne 0) {
    Write-Host "[selftest] exit $selfrc is not 0 - no reference list written"
    return 1
  }

  Set-Content -Encoding UTF8 $refPath "# G4-0 deterministic build reference (two clean-build SHA match)"
  Add-Content -Encoding UTF8 $refPath "# base=$BASE tree=$TREE patch_sha256=$PATCH_SHA"
  Add-Content -Encoding UTF8 $refPath "# toolchain: MSVC 14.44.35207 / Ninja / Release / CUDA=OFF BACKEND_DL=ON CPU_ALL_VARIANTS=ON / /Brepro / UI_GZIP=OFF / BUILD_UI=$(if($KeepUI){'ON+fixed-dist'}else{'OFF'})"
  Add-Content -Encoding UTF8 $refPath "# named set: $n files (the assembler's runtime files built here + $SELFTEST_NAME); hashes of the first build's stash binA"
  foreach ($f in $NAMED_SET) { Add-Content -Encoding UTF8 $refPath ("{0}  {1}" -f $hA[$f], $f) }
  Write-Host "[reference] written: $refPath"
  return 0
}

function Invoke-SelfCheck {
  $root = Join-Path ([IO.Path]::GetTempPath()) ("build_closure_selfcheck_" + $PID)
  if (Test-Path -LiteralPath $root) { Remove-Item -Recurse -Force -LiteralPath $root }
  New-Item -ItemType Directory -Path $root | Out-Null
  $results = New-Object System.Collections.ArrayList
  function Case([string]$name, [bool]$ok, [string]$detail) {
    [void]$results.Add($ok)
    Write-Host ("[selfcheck] {0,2} {1} {2}: {3}" -f $results.Count, $(if ($ok) {'PASS'} else {'FAIL'}), $name, $detail)
  }
  function New-StubDir([string]$dir, [string]$salt, [string[]]$skip) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($name in $NAMED_SET) {
      if ($skip -contains $name) { continue }
      [IO.File]::WriteAllText((Join-Path $dir $name), "stub $name $salt")
    }
  }
  $ok0 = { return 0 }
  try {
    # 1. two equal stub builds -> exit 0 and a reference list at a scratch path
    New-StubDir (Join-Path $root 'A') 'same' @()
    New-StubDir (Join-Path $root 'B') 'same' @()
    $ref1 = Join-Path $root 'ref_equal.txt'
    $hA = Get-NamedHashes (Join-Path $root 'A') 'A'
    $hB = Get-NamedHashes (Join-Path $root 'B') 'B'
    $rc = Complete-Closure $hA $hB $ref1 $ok0
    $lines = @(); if (Test-Path -LiteralPath $ref1) { $lines = @(Get-Content -LiteralPath $ref1) }
    $body = @($lines | Where-Object { $_ -notmatch '^\s*(#|\uFEFF#)' })
    $hdr  = ($lines -join "`n")
    Case 'equal-tables-write-reference' ($rc -eq 0 -and $body.Count -eq $NAMED_SET.Count -and $hdr.Contains("tree=$TREE") -and $hdr.Contains("patch_sha256=$PATCH_SHA")) "rc=$rc, list lines=$($body.Count) of $($NAMED_SET.Count), header names base/tree/patch"

    # 2. two different stub builds -> nonzero and no reference list
    New-StubDir (Join-Path $root 'C') 'same' @()
    [IO.File]::WriteAllText((Join-Path (Join-Path $root 'C') 'ggml-base.dll'), "stub ggml-base.dll other")
    $ref2 = Join-Path $root 'ref_differ.txt'
    $hC = Get-NamedHashes (Join-Path $root 'C') 'C'
    $rc = Complete-Closure $hA $hC $ref2 $ok0
    Case 'different-tables-refused' ($rc -ne 0 -and -not (Test-Path -LiteralPath $ref2)) "rc=$rc, reference exists=$(Test-Path -LiteralPath $ref2)"

    # 3. a build output lacking a named file -> refused, naming it
    New-StubDir (Join-Path $root 'D') 'same' @('llama-server.exe')
    $msg = ''
    try { [void](Get-NamedHashes (Join-Path $root 'D') 'D'); $msg = '(no error)' } catch { $msg = $_.Exception.Message }
    Case 'missing-named-file-refused' ($msg -like '*missing*llama-server.exe*') $msg

    # 4. a self-test that does not exit 0 -> nonzero and no reference list
    $ref4 = Join-Path $root 'ref_selftest.txt'
    $rc = Complete-Closure $hA $hB $ref4 { return 1 }
    Case 'selftest-nonzero-refused' ($rc -ne 0 -and -not (Test-Path -LiteralPath $ref4)) "rc=$rc, reference exists=$(Test-Path -LiteralPath $ref4)"

    # 5. the pinned patch passes the pin check; a different file is refused
    $other = Join-Path $root 'other.patch'
    [IO.File]::WriteAllText($other, "not the release patch")
    $okPin = $true; try { Assert-PatchPin $Patch } catch { $okPin = $false; $pinMsg = $_.Exception.Message }
    $msg = ''
    try { Assert-PatchPin $other; $msg = '(no error)' } catch { $msg = $_.Exception.Message }
    Case 'patch-pin' ($okPin -and $msg -like 'patch sha256 mismatch*') "pinned: $(if ($okPin) {'OK'} else {$pinMsg}); other: $msg"

    # 6. the script run with a wrong patch stops before anything else: no VS
    #    environment import, no work folder (so no clone and no build)
    $wd = Join-Path $root 'wd'
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $out = & $ps -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Patch $other -WorkDir $wd -Reference (Join-Path $root 'ref_run.txt') 2>&1 | Out-String
    $childRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    $refused = $out -match 'patch sha256 mismatch'
    $before  = -not ($out -match '\[env\]|\[0\] patch sha256 OK|==== build')
    Case 'wrong-patch-refused-first' ($childRc -ne 0 -and $refused -and $before -and -not (Test-Path -LiteralPath $wd) -and -not (Test-Path -LiteralPath (Join-Path $root 'ref_run.txt'))) "child exit=$childRc, mismatch message=$refused, no VS import/build line before it=$before, work folder exists=$(Test-Path -LiteralPath $wd)"

    # Runtime fixtures exercise the production provisioning path with explicit
    # fixture hashes; release pins are restored before leaving this block.
    $releasePins = $RUNTIME_SHA
    try {
      $fixtures = Join-Path $root 'runtime-inputs'
      New-Item -ItemType Directory -Path $fixtures | Out-Null
      $script:RUNTIME_SHA = @{}
      foreach ($name in $NOT_BUILT_HERE) {
        $path = Join-Path $fixtures $name
        [IO.File]::WriteAllText($path, "runtime fixture $name")
        $script:RUNTIME_SHA[$name] = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
      }
      $backend = Join-Path $fixtures 'ggml-cuda.dll'
      $runtime = Join-Path $root 'validation-good'
      $before = Get-NamedHashes (Join-Path $root 'A') 'before'
      [void](New-ValidationRuntime (Join-Path $root 'A') $runtime $backend $fixtures)
      $refRuntime = Join-Path $root 'ref_runtime.txt'
      $probe = {
        foreach ($name in $NOT_BUILT_HERE) { Assert-RuntimeFile (Join-Path $runtime $name) $name }
        return 0
      }
      $rc = Complete-Closure $hA $hB $refRuntime $probe
      $after = Get-NamedHashes (Join-Path $root 'A') 'after'
      $unchanged = @($NAMED_SET | Where-Object { $before[$_] -ne $after[$_] }).Count -eq 0
      $body = @(Get-Content -LiteralPath $refRuntime | Where-Object { $_ -notmatch '^\s*(#|\uFEFF#)' })
      $excluded = @($body | Where-Object { ($_ -split '  ')[-1] -in $NOT_BUILT_HERE }).Count -eq 0
      Case 'runtime-provisioned-before-selftest' ($rc -eq 0 -and $unchanged -and $body.Count -eq 17 -and $excluded) "rc=$rc; binA unchanged=$unchanged; reference=17; supplied files excluded=$excluded"
      foreach ($name in $NOT_BUILT_HERE) {
        $path = Join-Path $fixtures $name
        $bytes = [IO.File]::ReadAllBytes($path)
        Remove-Item -LiteralPath $path
        $dest = Join-Path $root ("missing-" + $name)
        $refMissing = Join-Path $root ("ref-missing-" + $name)
        $msg = ''
        try { [void](Complete-Closure $hA $hB $refMissing { [void](New-ValidationRuntime (Join-Path $root 'A') $dest $backend $fixtures); return 0 }) } catch { $msg = $_.Exception.Message }
        Case ("runtime-missing-" + $name) ($msg -like "runtime file missing: $name*" -and -not (Test-Path $dest) -and -not (Test-Path $refMissing)) $msg
        [IO.File]::WriteAllText($path, 'wrong provenance')
        $dest = Join-Path $root ("mismatch-" + $name)
        $refBad = Join-Path $root ("ref-mismatch-" + $name)
        $msg = ''
        try { [void](Complete-Closure $hA $hB $refBad { [void](New-ValidationRuntime (Join-Path $root 'A') $dest $backend $fixtures); return 0 }) } catch { $msg = $_.Exception.Message }
        Case ("runtime-mismatch-" + $name) ($msg -like "runtime provenance mismatch: $name*" -and -not (Test-Path $dest) -and -not (Test-Path $refBad)) $msg
        [IO.File]::WriteAllBytes($path, $bytes)
      }
      $refFail = Join-Path $root 'ref-runtime-selftest-failed.txt'
      $rc = Complete-Closure $hA $hB $refFail { foreach ($name in $NOT_BUILT_HERE) { Assert-RuntimeFile (Join-Path $runtime $name) $name }; return 23 }
      Case 'provisioned-selftest-nonzero-refused' ($rc -ne 0 -and -not (Test-Path $refFail)) "rc=$rc; reference absent=$( -not (Test-Path $refFail))"
    } finally { $script:RUNTIME_SHA = $releasePins }

    # 7. the named set equals what the assembler takes from this build
    $mb = $Assembler
    $detail = ''
    $ok = $false
    if (Test-Path -LiteralPath $mb) {
      $txt = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $mb))
      $m = [regex]::Match($txt, '(?s)\$script:RUNTIME_FILES\s*=\s*@\((.*?)\)')
      $s = [regex]::Match($txt, '(?m)^\$script:ABI_SELFTEST_EXE\s*=\s*''([^'']+)''')
      if ($m.Success -and $s.Success) {
        $rt = @([regex]::Matches($m.Groups[1].Value, "'([^']+)'") | ForEach-Object { $_.Groups[1].Value })
        $expect = @(($rt | Where-Object { $NOT_BUILT_HERE -notcontains $_ }) + $s.Groups[1].Value | Sort-Object -Unique)
        $have = @($NAMED_SET | Sort-Object -Unique)
        $ok = (($expect -join ',') -eq ($have -join ',')) -and ($s.Groups[1].Value -eq $SELFTEST_NAME)
        $detail = "assembler runtime $($rt.Count) less not-built $($NOT_BUILT_HERE.Count) + probe = $($expect.Count); named set $($have.Count)"
        if (-not $ok) { $detail += "; only assembler: " + (($expect | Where-Object { $have -notcontains $_ }) -join ',') + "; only here: " + (($have | Where-Object { $expect -notcontains $_ }) -join ',') }
      } else { $detail = "assembler lists not found in $mb" }
    } else { $detail = "assembler not found: $mb" }
    Case 'named-set-matches-assembler' $ok $detail
  } finally {
    Remove-Item -Recurse -Force -LiteralPath $root -ErrorAction SilentlyContinue
  }
  $pass = @($results | Where-Object { $_ }).Count
  $verdict = $(if ($pass -eq $results.Count) {'PASS'} else {'FAIL'})
  Write-Host "BUILD_CLOSURE SELFCHECK: $verdict ($pass/$($results.Count) cases)"
  if ($verdict -eq 'PASS') { return 0 } else { return 1 }
}

if ($SelfCheck) { exit (Invoke-SelfCheck) }

# A patch whose bytes differ from the pin is refused before anything else.
Assert-PatchPin $Patch
Write-Host "[0] patch sha256 OK"

# Auto-bootstrap the VS 2022 build environment so this runs from a plain
# PowerShell (no need to open the x64 Native Tools prompt). No-op if cl/cmake/
# ninja are already on PATH; silently returns if VS cannot be located.
function Enter-VsDevEnv {
  if ((Get-Command cl    -ErrorAction SilentlyContinue) -and
      (Get-Command cmake -ErrorAction SilentlyContinue) -and
      (Get-Command ninja -ErrorAction SilentlyContinue)) { return }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { return }
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
  if (-not $vsPath) { return }
  $devcmd = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
  if (-not (Test-Path $devcmd)) { return }
  Write-Host "[env] importing VS dev environment (x64) from $vsPath"
  cmd /c "`"$devcmd`" -arch=x64 -host_arch=x64 >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
  }
}
Enter-VsDevEnv

function Assert-Tool([string]$name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "$name not on PATH even after VS auto-bootstrap. Open 'x64 Native Tools Command Prompt for VS 2022' (or 'Developer PowerShell for VS 2022') and re-run. Needs: cl, ninja, cmake, git."
  }
}
'git','cmake','ninja','cl' | ForEach-Object { Assert-Tool $_ }

$SRC   = Join-Path $WorkDir 'src'
$BUILD = Join-Path $WorkDir 'build'

function Build-Once([string]$tag) {
  Write-Host "==== build $tag ===="
  foreach ($d in @($SRC,$BUILD)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }

  # identical checkout from BaseRepo -> identical BUILD_NUMBER (clone ignores
  # BaseRepo working-tree changes, so applying the patch fresh stays clean).
  # core.autocrlf=false: the working files are the blob bytes (see header).
  Native "[$tag] git clone" { git clone --local --no-hardlinks --config core.autocrlf=false $BaseRepo $SRC }

  Push-Location $SRC
  try {
    Native "[$tag] git checkout $BASE (BaseRepo must contain 0bd0ec6)" { git checkout -q $BASE }
    Native "[$tag] git apply" { git -c core.autocrlf=false apply --index $Patch }
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $wt = (& git write-tree 2>$null | Select-Object -First 1)
    $ErrorActionPreference = $prevEap
    if (-not $wt) { throw "[$tag] git write-tree produced no output" }
    $wt = "$wt".Trim()
    if ($wt -ne $TREE) { throw "[$tag] write-tree $wt != sealed $TREE - source drift, abort" }
    Write-Host "[$tag] write-tree == $TREE (sealed source OK)"
  } finally { Pop-Location }

  $uiFlags = @('-DLLAMA_USE_PREBUILT_UI=OFF','-DLLAMA_UI_GZIP=OFF')
  if ($KeepUI) {
    if (-not $DistDir -or -not (Test-Path (Join-Path $DistDir 'index.html'))) {
      throw "-KeepUI requires -DistDir pointing at a prebuilt dist (with index.html)."
    }
    Copy-Item -Recurse -Force $DistDir (Join-Path $SRC 'tools\ui\dist')
    $uiFlags += '-DLLAMA_BUILD_UI=ON'
  } else {
    $uiFlags += '-DLLAMA_BUILD_UI=OFF'
  }

  $cfg = @(
    '-S', $SRC, '-B', $BUILD, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_CUDA=OFF','-DGGML_BACKEND_DL=ON','-DGGML_CPU_ALL_VARIANTS=ON',
    '-DGGML_NATIVE=OFF','-DGGML_LTO=OFF','-DBUILD_SHARED_LIBS=ON',
    '-DCMAKE_EXE_LINKER_FLAGS_RELEASE=/INCREMENTAL:NO /Brepro',
    '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=/INCREMENTAL:NO /Brepro',
    '-DCMAKE_MODULE_LINKER_FLAGS_RELEASE=/INCREMENTAL:NO /Brepro'
  ) + $uiFlags
  Native "[$tag] cmake configure" { cmake @cfg }
  Native "[$tag] cmake build" { cmake --build $BUILD --config Release }

  # stash the output folder, then hash the named set in the stash (runtime CUDA
  # DLLs are copied at staging, not built here)
  $bin = Join-Path $BUILD 'bin'
  $stash = Join-Path $WorkDir "bin$tag"
  if (Test-Path $stash) { Remove-Item -Recurse -Force $stash }
  Copy-Item -Recurse -Force $bin $stash
  return (Get-NamedHashes $stash $tag)
}

$hA = Build-Once 'A'
$hB = Build-Once 'B'

$runSelftest = {
  # A fresh sibling directory keeps supplied runtime files out of binA/binB
  # and the deterministic reference. No arguments: the full selftest is required.
  $runtime = Join-Path $WorkDir ('validation-' + [Guid]::NewGuid().ToString('N'))
  [void](New-ValidationRuntime (Join-Path $WorkDir 'binA') $runtime $CudaBackend $NvidiaRuntimeDir)
  return (Invoke-ValidationSelftest $runtime)
}

$rc = Complete-Closure $hA $hB $Reference $runSelftest
if ($rc -eq 0) {
  Write-Host "NOTE: the four provenance-checked runtime DLLs were supplied only for validation, not reproduced or listed in BUILD_REFERENCE; release staging must supply the same contract set."
}
exit $rc
