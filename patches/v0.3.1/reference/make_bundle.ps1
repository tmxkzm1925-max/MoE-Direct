#Requires -Version 5.1
<#
    make_bundle.ps1 - MoE-Direct release bundle assembler + integrity manifest generator.

    Authority (do not re-interpret):
      RELEASE_SPEC.md  v0.1 (FROZEN)  section 2 inventory, 6-2 integrity root, 10 item 11 names
      LAUNCHER_SPEC.md v0.4 (FROZEN)  section 2 "first action = full internal SHA manifest check"

    The bundle_manifest.json written here is defined by its consumer, not by this script:
    Start-MoeDirect.ps1 / Assert-BundleIntegrity requires
        bundle_manifest_version = 1 (exact)
        files[] non-empty, each { path, sha256 }, path may not contain '..'
        every listed file must exist and hash-match
        every file under the bundle root must be listed (both directions)
    bundle_manifest.json itself is the only file excluded from the reverse check.

    Output layout (staging-flat + expects/ + repacker/):
        Start-MoeDirect.cmd            double-click entry point
        Start-MoeDirect.ps1            launcher (byte-identical to the repo copy)
        models.json                    catalog (repacker_exe sentinel resolved, see below)
        bundle_manifest.json           generated here
        LICENSE, TRADEMARKS.md, ...    licence surface
        llama-server.exe + DLLs        engine runtime, flat next to the exe (GGML_BACKEND_DL)
        expects/*.expect.json          consumed by the engine via MOE_DIRECT_EXPECTS_DIR
        repacker/repack_experts.py     repacker
        repacker/expects/*.json        repacker's own EXPECTS_DIR (script-relative, see note)
        repacker/python/*              CPython embeddable runtime (pinned, hash-checked)

    All output is English ASCII. This file must stay UTF-8 with BOM (PS 5.1 CP949 hazard).
#>
[CmdletBinding()]
param(
    # Engine binaries + expects staged for release (v0.3.1 generation, HR-R1; STAGING_MANIFEST.txt inside).
    [string] $StagingDir = 'D:\moe-tools\llama-moedirect-v031',
    # Where the bundle directory / zip / SHA256SUMS.txt are written. Default: .\out
    [string] $OutRoot,
    # Public version string used in the asset names (RS 10 item 11).
    [string] $Version = 'v0.2',
    # Rehearsal build: appends "-rehearsal" to the bundle and zip names.
    [switch] $Rehearsal,
    # CPython embeddable archive. Default: .\vendor\<pinned name>
    [string] $PythonEmbedZip,
    # Value written into runtime.repacker_exe when the catalog still carries the sentinel.
    # Bundle-relative: the launcher resolves it with Join-Path <bundle root> <value>.
    [string] $RepackerExe = 'repacker/python/python.exe',
    # Optional app-local MSVC runtime source directory (see the UNRESOLVED note printed at the
    # end). Empty = do not ship the MSVC runtime; the machine must already have it.
    [string] $VcRuntimeDir = '',
    # Assemble the directory but skip zip + SHA256SUMS.txt.
    [switch] $NoZip,
    # Print the file plan and stop. No copying, no hashing.
    [switch] $PlanOnly,
    # Overwrite an existing bundle directory / zip.
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Off

# A build tool must never end with a zero exit code after a terminating error: a caller that only
# looks at the exit status would then ship a half-assembled bundle.
trap {
    [Console]::Error.WriteLine('[make_bundle] FAILED: unhandled error - ' + $_.Exception.Message)
    [Console]::Error.WriteLine($_.ScriptStackTrace)
    exit 1
}

# ============================================================================
# region PINNED INPUTS
# ============================================================================

# CPython embeddable runtime for the repacker. Decision (a), 26-07-30.
#   Why 3.11: repack_experts.py imports stdlib only (argparse ctypes hashlib json os random re
#   shutil struct subprocess sys tempfile time traceback, ctypes.wintypes, datetime) - verified by
#   reading every import statement in the file - and every recorded repacker run in this project
#   was made with CPython 3.11.x. 3.11.9 is the last 3.11 with an official Windows build.
#   MD5 is what python.org publishes on the release page; the SHA-256 is computed locally and
#   pinned here so a later re-download cannot silently change the shipped runtime.
$script:PY_EMBED_NAME   = 'python-3.11.9-embed-amd64.zip'
$script:PY_EMBED_URL    = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip'
$script:PY_EMBED_MD5    = '6d9aa08531d48fcc261ba667e2df17c4'
$script:PY_EMBED_SHA256 = '009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b'

# Consumer contract (Start-MoeDirect.ps1:146-147).
$script:BUNDLE_MANIFEST_NAME    = 'bundle_manifest.json'
$script:BUNDLE_MANIFEST_VERSION = 1
$script:CATALOG_NAME            = 'models.json'
$script:REPACKER_SENTINEL       = 'UNRESOLVED_repacker_exe'

# --- OPEN_ARCH M5 atomic activation (OPEN_ARCH_DESIGN.md v0.2 section 4) ------------------------
# "activation is atomic: repacker / engine / launcher all carry the SAME OPEN_ARCH_TEMPLATE_ABI -
#  if any one of them is absent or different, the v0.2.1 bundle assembly fails."
# This assembler is the ONLY comparator of the three axes (OPENARCH_B_SPEC_DRAFT v0.2 section 2-5
# decision D2: the ABI field is exposed by the engine probe for the M5 assembler; the probe's
# catalog rows belong to Gate G and are NOT compared here).
#
# Where each axis carries the literal:
#   A repacker  bench\repack\repack_experts.py       OPEN_ARCH_TEMPLATE_ABI = '...'
#   B engine    <staging>\moe-direct-selftest.exe    --openarch-probe-json  ->  {"abi":"..."}
#   C launcher  <launcher>\Start-MoeDirect.ps1       $script:OPEN_ARCH_TEMPLATE_ABI = '...'
# The two source-literal patterns are anchored at column 0 and must match EXACTLY ONCE: a second
# declaration would make "which one ships" a guess, and this gate never guesses.
$script:ABI_SELFTEST_EXE   = 'moe-direct-selftest.exe'
$script:ABI_PROBE_ARG      = '--openarch-probe-json'
$script:ABI_PROBE_TIMEOUT_MS = 30000
$script:ABI_RE_REPACKER    = '(?m)^OPEN_ARCH_TEMPLATE_ABI[ \t]*=[ \t]*''([^'']*)''[ \t\r]*$'
$script:ABI_RE_LAUNCHER    = '(?m)^\$script:OPEN_ARCH_TEMPLATE_ABI[ \t]*=[ \t]*''([^'']*)''[ \t\r]*$'

# Engine runtime file set. Derived, not guessed:
#   - transitive PE import closure of llama-server.exe inside the staging directory
#     (llama-server-impl, llama, llama-common, mtmd, ggml, ggml-base, ggml-cuda, cublas64_13,
#      cublasLt64_13)
#   - the ggml backends are loaded with LoadLibrary at run time (build flag GGML_BACKEND_DL=ON,
#     STAGING_MANIFEST.txt), so every ggml-cpu-* variant must ship even though no import table
#     references it. They must stay flat next to llama-server.exe.
#   - cudart64_13.dll has no importer in this closure (CUDA runtime is statically linked into
#     ggml-cuda.dll), but it is part of the official CUDA 4 set that every measured run used, so
#     it ships rather than being dropped on an inference about lazy loading. 551 KB.
$script:RUNTIME_FILES = @(
    'llama-server.exe',
    'llama-server-impl.dll',
    'llama-common.dll',
    'llama.dll',
    'mtmd.dll',
    'ggml.dll',
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
    'ggml-cuda.dll',
    'cublas64_13.dll',
    'cublasLt64_13.dll',
    'cudart64_13.dll'
)

# App-local MSVC runtime, only when -VcRuntimeDir is given (see final report).
$script:VC_RUNTIME_FILES = @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll')

# Only these characters are allowed in a manifest path, so the generated JSON needs no escaping
# and the reverse check cannot be defeated by an odd file name.
$script:REL_PATH_OK = '^[A-Za-z0-9_][A-Za-z0-9._+-]*(/[A-Za-z0-9_][A-Za-z0-9._+-]*)*$'

# endregion

# ============================================================================
# region HELPERS
# ============================================================================

function Write-Info { param([string] $Text) [Console]::Out.WriteLine($Text) }
function Write-Warn { param([string] $Text) [Console]::Out.WriteLine('[warn] ' + $Text) }
function Stop-Build {
    param([string] $Text)
    [Console]::Error.WriteLine('[make_bundle] FAILED: ' + $Text)
    exit 1
}

function Get-Sha256Lower {
    param([string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Get-Md5Lower {
    param([string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash.ToLowerInvariant()
}

function Read-TextNoBom {
    param([string] $Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $off = 0
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { $off = 3 }
    $enc = New-Object System.Text.UTF8Encoding($false, $true)
    return $enc.GetString($bytes, $off, $bytes.Length - $off)
}

function Write-TextNoBom {
    param([string] $Path, [string] $Text)
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Format-Bytes {
    param([long] $Bytes)
    if ($Bytes -ge 1073741824) { return ('{0:N2} GiB' -f ($Bytes / 1073741824)) }
    if ($Bytes -ge 1048576)    { return ('{0:N2} MiB' -f ($Bytes / 1048576)) }
    if ($Bytes -ge 1024)       { return ('{0:N2} KiB' -f ($Bytes / 1024)) }
    return ('{0} B' -f $Bytes)
}

# endregion

# ============================================================================
# region 1. RESOLVE INPUTS
# ============================================================================

$packagingDir = $PSScriptRoot
$launcherDir  = Split-Path -Parent $packagingDir                    # bench\moe-direct\launcher
$repoRoot     = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $launcherDir))
$repackDir    = Join-Path $repoRoot 'bench\repack'
$expectsSrc   = Join-Path $repackDir 'expects'
$repackerSrc  = Join-Path $repackDir 'repack_experts.py'
$publishDir   = Join-Path $repoRoot 'publish\moe-direct'

if (-not $OutRoot)        { $OutRoot        = Join-Path $packagingDir 'out' }
if (-not $PythonEmbedZip) { $PythonEmbedZip = Join-Path (Join-Path $packagingDir 'vendor') $script:PY_EMBED_NAME }

$bundleName = 'moe-direct-{0}-win-x64' -f $Version
if ($Rehearsal) { $bundleName = $bundleName + '-rehearsal' }
$bundleDir = Join-Path $OutRoot $bundleName
$zipPath   = Join-Path $OutRoot ($bundleName + '.zip')

Write-Info '=========================================================================='
Write-Info (' MoE-Direct bundle assembler   {0}' -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))
Write-Info '=========================================================================='
Write-Info (' repo root      : {0}' -f $repoRoot)
Write-Info (' staging dir    : {0}' -f $StagingDir)
Write-Info (' launcher dir   : {0}' -f $launcherDir)
Write-Info (' expects source : {0}' -f $expectsSrc)
Write-Info (' repacker       : {0}' -f $repackerSrc)
Write-Info (' python runtime : {0}' -f $PythonEmbedZip)
Write-Info (' bundle dir     : {0}' -f $bundleDir)
Write-Info ''

foreach ($p in @($StagingDir, $launcherDir, $expectsSrc, (Join-Path $StagingDir 'expects'))) {
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { Stop-Build ('input directory missing: ' + $p) }
}
foreach ($p in @($repackerSrc, (Join-Path $launcherDir 'Start-MoeDirect.ps1'),
                 (Join-Path $launcherDir $script:CATALOG_NAME),
                 (Join-Path $packagingDir 'Start-MoeDirect.cmd'))) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Stop-Build ('input file missing: ' + $p) }
}

# --- approved expect digest gate (RELEASE_SPEC.md:102-106, DECISION_LEDGER.md:3042 item 5) -----
# The catalog is the one truth for the approved expect digests; the repacker's EXPECT_CATALOG and
# the engine's EXPECT_CATALOG (read from the release patch the engine is built from) are bound to
# it here, fail-closed, before anything is copied. The comparison lives in the helper beside this
# file, which also runs alone without a staging directory.
$digestGate = Join-Path $packagingDir 'check_expect_digests.ps1'
if (-not (Test-Path -LiteralPath $digestGate -PathType Leaf)) { Stop-Build ('input file missing: ' + $digestGate) }
& $digestGate
if ($LASTEXITCODE -ne 0) { Stop-Build ('approved expect digest gate refused (exit ' + $LASTEXITCODE + '); see the [expect-digest] lines above') }

# --- pinned python runtime ---------------------------------------------------
if (-not (Test-Path -LiteralPath $PythonEmbedZip -PathType Leaf)) {
    Write-Info 'The pinned CPython embeddable runtime is not present. Download it once:'
    Write-Info ('  URL    : {0}' -f $script:PY_EMBED_URL)
    Write-Info ('  save as: {0}' -f $PythonEmbedZip)
    Write-Info ('  MD5    : {0}   (published on the python.org release page)' -f $script:PY_EMBED_MD5)
    Write-Info ('  SHA-256: {0}   (pinned by this script)' -f $script:PY_EMBED_SHA256)
    Stop-Build 'python embeddable archive missing'
}
$pyMd5 = Get-Md5Lower -Path $PythonEmbedZip
$pySha = Get-Sha256Lower -Path $PythonEmbedZip
if ($pyMd5 -ne $script:PY_EMBED_MD5) { Stop-Build ('python embeddable MD5 mismatch: got ' + $pyMd5 + ', expected ' + $script:PY_EMBED_MD5) }
if ($pySha -ne $script:PY_EMBED_SHA256) { Stop-Build ('python embeddable SHA-256 mismatch: got ' + $pySha + ', expected ' + $script:PY_EMBED_SHA256) }
Write-Info ('[python] {0} verified (MD5 + SHA-256 match the pin)' -f $script:PY_EMBED_NAME)

# --- expects set: names from staging, bytes from the repo --------------------
# The engine build in StagingDir was staged with a specific expects set; shipping exactly that set
# keeps the engine's compiled EXPECT_CATALOG closed. The bytes are taken from the repo (single
# source of truth) and asserted byte-identical to the staged copy.
$expectNames = @(Get-ChildItem -LiteralPath (Join-Path $StagingDir 'expects') -Filter '*.expect.json' -File |
                 Sort-Object Name | ForEach-Object { $_.Name })
if ($expectNames.Count -eq 0) { Stop-Build 'staging expects directory has no *.expect.json' }
foreach ($n in $expectNames) {
    $a = Join-Path $expectsSrc $n
    $b = Join-Path (Join-Path $StagingDir 'expects') $n
    if (-not (Test-Path -LiteralPath $a -PathType Leaf)) { Stop-Build ('expect staged but not in the repo: ' + $n) }
    if ((Get-Sha256Lower -Path $a) -ne (Get-Sha256Lower -Path $b)) {
        Stop-Build ('expect differs between repo and staging: ' + $n + ' (resolve before packaging)')
    }
}
Write-Info ('[expects] {0} file(s), repo bytes == staging bytes: {1}' -f $expectNames.Count, ($expectNames -join ', '))

# --- OPEN_ARCH ABI atomic gate (OPEN_ARCH_DESIGN.md v0.2 section 4) ----------
# Runs here, before anything is copied: an ABI split must stop the build while the output tree does
# not exist yet, not after a half-assembled bundle is on disk. The sources read are exactly the
# bytes the file plan copies (repack_experts.py -> repacker/repack_experts.py, Start-MoeDirect.ps1
# -> Start-MoeDirect.ps1), so what is compared is what ships.

function Get-AbiFromSource {
    # Returns @{ ok; abi; reason }. A missing file, no match, or more than one match is a failure.
    param([string] $Axis, [string] $Path, [string] $Pattern)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return @{ ok = $false; reason = ('{0}: source file missing: {1}' -f $Axis, $Path) }
    }
    $text = Read-TextNoBom -Path $Path
    $m = [regex]::Matches($text, $Pattern)
    if ($m.Count -eq 0) {
        return @{ ok = $false; reason = ('{0}: no OPEN_ARCH_TEMPLATE_ABI declaration in {1}' -f $Axis, $Path) }
    }
    if ($m.Count -gt 1) {
        return @{ ok = $false; reason = ('{0}: OPEN_ARCH_TEMPLATE_ABI is declared {1} times in {2}; refusing to guess which one ships' -f $Axis, $m.Count, $Path) }
    }
    return @{ ok = $true; abi = [string]$m[0].Groups[1].Value }
}

function Get-AbiFromEngine {
    # Returns @{ ok; abi; reason }. The engine surface is the selftest probe CLI
    # (OPENARCH_B_SPEC_DRAFT v0.2 section 2-5).
    param([string] $StagingDir)
    $exe = Join-Path $StagingDir $script:ABI_SELFTEST_EXE
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        return @{ ok = $false; reason = ('B engine: {0} not found in the staging dir: {1}' -f $script:ABI_SELFTEST_EXE, $exe) }
    }
    # Pre-scan before spawning. An engine build from before the B-axis round does not know the flag
    # and falls through to its full self-test suite (minutes of work, scratch files, and a first
    # stdout line that is not JSON). Confirming the surface in the binary keeps a stale staging
    # directory from being executed at all.
    $needle = $script:ABI_PROBE_ARG
    $enc    = [System.Text.Encoding]::GetEncoding(28591)   # latin-1: byte-for-char, no loss
    $blob   = $enc.GetString([System.IO.File]::ReadAllBytes($exe))
    if ($blob.IndexOf($needle, [System.StringComparison]::Ordinal) -lt 0) {
        return @{ ok = $false; reason = ('B engine: the staged {0} has no {1} surface (pre-scan); this engine build predates the OPEN_ARCH B axis' -f $script:ABI_SELFTEST_EXE, $script:ABI_PROBE_ARG) }
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $exe
    $psi.Arguments              = $script:ABI_PROBE_ARG
    $psi.WorkingDirectory       = $StagingDir
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    # Read both pipes asynchronously: waiting first and reading afterwards deadlocks as soon as a
    # pipe buffer fills.
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($script:ABI_PROBE_TIMEOUT_MS)) {
        try { $proc.Kill() } catch { }
        return @{ ok = $false; reason = ('B engine: {0} {1} did not exit within {2} ms' -f $script:ABI_SELFTEST_EXE, $script:ABI_PROBE_ARG, $script:ABI_PROBE_TIMEOUT_MS) }
    }
    $stdout = [string]$outTask.Result
    $stderrText = [string]$errTask.Result
    $rc = $proc.ExitCode
    if ($rc -ne 0) {
        return @{ ok = $false; reason = ('B engine: probe exited {0}: {1}' -f $rc, ($stderrText.Trim() -replace '\s+', ' ')) }
    }
    $lines = @($stdout -split "`r?`n" | Where-Object { $_.Trim().Length -gt 0 })
    if ($lines.Count -eq 0) { return @{ ok = $false; reason = 'B engine: probe produced no output' } }
    $json = $lines[$lines.Count - 1].Trim()
    try { $obj = $json | ConvertFrom-Json } catch {
        return @{ ok = $false; reason = ('B engine: probe output is not JSON: ' + $json.Substring(0, [Math]::Min(160, $json.Length))) }
    }
    $abi = [string]$obj.abi
    if ([string]::IsNullOrEmpty($abi)) { return @{ ok = $false; reason = 'B engine: probe JSON has no non-empty "abi" field' } }
    return @{ ok = $true; abi = $abi }
}

Write-Info ''
Write-Info '[abi] OPEN_ARCH atomic activation gate (repacker / engine / launcher)'
$launcherPs1 = Join-Path $launcherDir 'Start-MoeDirect.ps1'
$abiAxes = @(
    @{ Axis = 'A repacker'; Src = $repackerSrc;  Res = (Get-AbiFromSource -Axis 'A repacker' -Path $repackerSrc  -Pattern $script:ABI_RE_REPACKER) },
    @{ Axis = 'B engine';   Src = (Join-Path $StagingDir $script:ABI_SELFTEST_EXE); Res = (Get-AbiFromEngine -StagingDir $StagingDir) },
    @{ Axis = 'C launcher'; Src = $launcherPs1;  Res = (Get-AbiFromSource -Axis 'C launcher' -Path $launcherPs1 -Pattern $script:ABI_RE_LAUNCHER) }
)
$abiProblems = @()
foreach ($a in $abiAxes) {
    if ($a.Res.ok) {
        Write-Info ('  {0,-11} {1}   <- {2}' -f $a.Axis, $a.Res.abi, $a.Src)
    } else {
        Write-Info ('  {0,-11} ABSENT' -f $a.Axis)
        $abiProblems += [string]$a.Res.reason
    }
}
if ($abiProblems.Count -gt 0) {
    Stop-Build ('OPEN_ARCH_TEMPLATE_ABI is not readable on every axis, so activation cannot be atomic - ' +
                ($abiProblems -join ' | '))
}
$abiValues = @($abiAxes | ForEach-Object { [string]$_.Res.abi })
# Case-sensitive: the ABI is a wire token, not a display string.
$abiDistinct = @($abiValues | Sort-Object -Unique -CaseSensitive)
if ($abiDistinct.Count -ne 1) {
    $detail = (($abiAxes | ForEach-Object { '{0}={1}' -f $_.Axis, $_.Res.abi }) -join ' | ')
    Stop-Build ('OPEN_ARCH_TEMPLATE_ABI differs across the three axes, so activation would not be atomic - ' + $detail)
}
Write-Info ('[abi] 3/3 axes agree: {0}' -f $abiDistinct[0])

# --- licence surface (RS 2 item 7) ------------------------------------------
$licencePlan = @()
foreach ($n in @('LICENSE', 'THIRD_PARTY_NOTICES.md', 'TRADEMARKS.md', 'CITATION.cff')) {
    $p = Join-Path $publishDir $n
    if (Test-Path -LiteralPath $p -PathType Leaf) { $licencePlan += @{ Src = $p; Rel = $n } }
    else { Write-Warn ('licence file not found, not shipped: ' + $p) }
}
$upstream = Join-Path (Split-Path -Parent $StagingDir) 'llama.cpp-src-b10057\LICENSE'
if (Test-Path -LiteralPath $upstream -PathType Leaf) {
    $licencePlan += @{ Src = $upstream; Rel = 'LICENSE.llama.cpp.txt' }
} else {
    Write-Warn ('upstream llama.cpp LICENSE not found, not shipped: ' + $upstream)
}

# endregion

# ============================================================================
# region 2. FILE PLAN
# ============================================================================

$plan = New-Object System.Collections.ArrayList
function Add-Plan {
    param([string] $Src, [string] $Rel)
    if ($Rel -notmatch $script:REL_PATH_OK) { Stop-Build ('relative path not allowed in a manifest: ' + $Rel) }
    [void]$plan.Add(@{ Src = $Src; Rel = $Rel })
}

# root: entry points and the catalog (models.json is generated below, not copied)
Add-Plan -Src (Join-Path $packagingDir 'Start-MoeDirect.cmd') -Rel 'Start-MoeDirect.cmd'
Add-Plan -Src (Join-Path $launcherDir 'Start-MoeDirect.ps1')  -Rel 'Start-MoeDirect.ps1'
foreach ($l in $licencePlan) { Add-Plan -Src $l.Src -Rel $l.Rel }

# Build receipt (RS section 2 item 8: bind the shipped binaries to the source they came from).
# That item is a MUST, and models.json's source_tag alone only names the source - it does not
# state which engine sources and which build outputs the tag stands for. A missing receipt stops
# the build instead of silently shipping a bundle whose provenance cannot be checked, which is
# how every other MUST input here behaves.
$receiptSrc = Join-Path $publishDir 'BUILD_RECEIPT.txt'
if (-not (Test-Path -LiteralPath $receiptSrc -PathType Leaf)) {
    Stop-Build ('build receipt missing (RS 2 #8): ' + $receiptSrc)
}
Add-Plan -Src $receiptSrc -Rel 'BUILD_RECEIPT.txt'

# engine runtime, flat
foreach ($n in $script:RUNTIME_FILES) {
    $p = Join-Path $StagingDir $n
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Stop-Build ('runtime file missing in staging: ' + $p) }
    Add-Plan -Src $p -Rel $n
}
if ($VcRuntimeDir) {
    foreach ($n in $script:VC_RUNTIME_FILES) {
        $p = Join-Path $VcRuntimeDir $n
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Stop-Build ('MSVC runtime file missing: ' + $p) }
        Add-Plan -Src $p -Rel $n
    }
}

# expects, twice on purpose:
#   expects/            <- runtime.expects_dir, read by the engine (MOE_DIRECT_EXPECTS_DIR)
#   repacker/expects/   <- repack_experts.py:131 hardcodes EXPECTS_DIR = <script dir>\expects and
#                          has no CLI override, so the repacker cannot read the root copy.
# Both copies are byte-identical and both are re-hashed by their own consumer.
foreach ($n in $expectNames) {
    Add-Plan -Src (Join-Path $expectsSrc $n) -Rel ('expects/' + $n)
    Add-Plan -Src (Join-Path $expectsSrc $n) -Rel ('repacker/expects/' + $n)
}

# repacker script
Add-Plan -Src $repackerSrc -Rel 'repacker/repack_experts.py'

if ($PlanOnly) {
    Write-Info ''
    Write-Info '=== file plan (PlanOnly) ==='
    foreach ($e in $plan) { Write-Info ('  {0,-40} <- {1}' -f $e.Rel, $e.Src) }
    Write-Info ('  {0,-40} <- generated (sentinel resolved)' -f $script:CATALOG_NAME)
    Write-Info ('  repacker/python/*                        <- {0}' -f $script:PY_EMBED_NAME)
    Write-Info ('  {0,-40} <- generated' -f $script:BUNDLE_MANIFEST_NAME)
    Write-Info ''
    Write-Info ('planned explicit files: {0}' -f $plan.Count)
    exit 0
}

# endregion

# ============================================================================
# region 3. ASSEMBLE
# ============================================================================

if (Test-Path -LiteralPath $bundleDir) {
    if (-not $Force) { Stop-Build ('bundle directory already exists (use -Force to replace): ' + $bundleDir) }
    Remove-Item -LiteralPath $bundleDir -Recurse -Force
}
if ((Test-Path -LiteralPath $zipPath) -and -not $Force -and -not $NoZip) {
    Stop-Build ('zip already exists (use -Force to replace): ' + $zipPath)
}
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

Write-Info ''
Write-Info '[copy] engine runtime, launcher, expects, repacker script'
foreach ($e in $plan) {
    $dst = Join-Path $bundleDir ($e.Rel -replace '/', '\')
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $dstDir -PathType Container)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    Copy-Item -LiteralPath $e.Src -Destination $dst -Force
}

# --- catalog: resolve the repacker_exe sentinel, byte-scoped ----------------
# The repo catalog is a frozen, cross-reviewed artefact and is not edited. The sentinel is a
# packaging-time placeholder by design (models.provenance.md unresolved item 3), so exactly one
# JSON string token is substituted in the bundle copy and the substitution is reported. Nothing
# else in the file is touched: no JSON round-trip, no re-ordering, no re-indentation.
$catalogSrcPath = Join-Path $launcherDir $script:CATALOG_NAME
$catalogText    = Read-TextNoBom -Path $catalogSrcPath
$catalogSrcSha  = Get-Sha256Lower -Path $catalogSrcPath
$token          = '"' + $script:REPACKER_SENTINEL + '"'
$occurrences    = ([regex]::Matches($catalogText, [regex]::Escape($token))).Count
if ($occurrences -eq 1) {
    $catalogText = $catalogText.Replace($token, ('"' + $RepackerExe + '"'))
    Write-Info ('[catalog] runtime.repacker_exe: {0} -> {1}' -f $script:REPACKER_SENTINEL, $RepackerExe)
} elseif ($occurrences -eq 0) {
    Write-Info '[catalog] no repacker_exe sentinel present; catalog value used as is'
} else {
    Stop-Build ('the repacker_exe sentinel appears ' + $occurrences + ' times; refusing to guess')
}
if ($catalogText -match 'UNRESOLVED_') {
    Stop-Build 'catalog still contains an UNRESOLVED_ sentinel after substitution (packaging gate)'
}
$catalogDstPath = Join-Path $bundleDir $script:CATALOG_NAME
Write-TextNoBom -Path $catalogDstPath -Text $catalogText
Write-Info ('[catalog] source sha256 {0}' -f $catalogSrcSha)
Write-Info ('[catalog] bundle sha256 {0}' -f (Get-Sha256Lower -Path $catalogDstPath))

# --- python embeddable runtime ---------------------------------------------
Add-Type -AssemblyName System.IO.Compression.FileSystem
$pyDir = Join-Path $bundleDir 'repacker\python'
[System.IO.Compression.ZipFile]::ExtractToDirectory($PythonEmbedZip, $pyDir)
$pyCount = @(Get-ChildItem -LiteralPath $pyDir -Recurse -File).Count
Write-Info ('[python] expanded {0} file(s) into repacker\python' -f $pyCount)

# endregion

# ============================================================================
# region 4. PACKAGING GATE (independent of the launcher, before the manifest)
# ============================================================================

Write-Info ''
Write-Info '[gate] packaging checks'
$catalog = $catalogText | ConvertFrom-Json
$rt = $catalog.runtime

$serverExe = Join-Path $bundleDir ($rt.server_exe -replace '/', '\')
if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) { Stop-Build ('runtime.server_exe does not resolve inside the bundle: ' + $rt.server_exe) }
Write-Info ('  server_exe   -> {0}  OK' -f $rt.server_exe)

$repExe = Join-Path $bundleDir ($rt.repacker_exe -replace '/', '\')
if (-not (Test-Path -LiteralPath $repExe -PathType Leaf)) { Stop-Build ('runtime.repacker_exe does not resolve inside the bundle: ' + $rt.repacker_exe) }
Write-Info ('  repacker_exe -> {0}  OK' -f $rt.repacker_exe)

foreach ($a in @($rt.repacker_argv)) {
    $s = [string]$a
    if ($s.StartsWith('./') -or $s.StartsWith('.\')) {
        $p = Join-Path $bundleDir ($s.Substring(2) -replace '/', '\')
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Stop-Build ('runtime.repacker_argv entry does not resolve inside the bundle: ' + $s) }
        Write-Info ('  repacker_argv-> {0}  OK' -f $s)
    }
}

$expDir = Join-Path $bundleDir ($rt.expects_dir -replace '/', '\')
if (-not (Test-Path -LiteralPath $expDir -PathType Container)) { Stop-Build ('runtime.expects_dir does not resolve inside the bundle: ' + $rt.expects_dir) }
foreach ($p in @($catalog.profiles)) {
    $ep = Join-Path $expDir $p.expect_file
    if (-not (Test-Path -LiteralPath $ep -PathType Leaf)) { Stop-Build ($p.profile_id + ': expect file missing in the bundle: ' + $p.expect_file) }
    $h = Get-Sha256Lower -Path $ep
    if ($h -ne ([string]$p.expect_sha256).ToLowerInvariant()) { Stop-Build ($p.profile_id + ': expect sha256 mismatch in the bundle') }
    # the repacker reads its own copy, so that one is checked too
    $rp = Join-Path (Join-Path $bundleDir 'repacker\expects') $p.expect_file
    if (-not (Test-Path -LiteralPath $rp -PathType Leaf)) { Stop-Build ($p.profile_id + ': expect file missing in repacker\expects') }
    if ((Get-Sha256Lower -Path $rp) -ne $h) { Stop-Build ($p.profile_id + ': repacker expect copy differs from the root copy') }
}
Write-Info ('  expects      -> {0} profile(s) digest-matched in both copies  OK' -f @($catalog.profiles).Count)

# endregion

# ============================================================================
# region 5. MANIFEST
# ============================================================================

Write-Info ''
Write-Info '[manifest] hashing every file in the bundle'
$rootLen = $bundleDir.Length
$entries = @()
$totalBytes = [long]0
foreach ($f in (Get-ChildItem -LiteralPath $bundleDir -Recurse -File | Sort-Object FullName)) {
    $rel = $f.FullName.Substring($rootLen).TrimStart('\') -replace '\\', '/'
    if ($rel -eq $script:BUNDLE_MANIFEST_NAME) { continue }
    if ($rel -notmatch $script:REL_PATH_OK) { Stop-Build ('file name not allowed in a manifest: ' + $rel) }
    $entries += @{ path = $rel; sha256 = (Get-Sha256Lower -Path $f.FullName) }
    $totalBytes += $f.Length
}
$entries = @($entries | Sort-Object { $_.path })

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('{')
[void]$sb.AppendLine(('  "bundle_manifest_version": {0},' -f $script:BUNDLE_MANIFEST_VERSION))
[void]$sb.AppendLine('  "files": [')
for ($i = 0; $i -lt $entries.Count; $i++) {
    $comma = ','
    if ($i -eq ($entries.Count - 1)) { $comma = '' }
    [void]$sb.AppendLine(('    {{ "path": "{0}", "sha256": "{1}" }}{2}' -f $entries[$i].path, $entries[$i].sha256, $comma))
}
[void]$sb.AppendLine('  ]')
[void]$sb.AppendLine('}')
$manifestPath = Join-Path $bundleDir $script:BUNDLE_MANIFEST_NAME
Write-TextNoBom -Path $manifestPath -Text $sb.ToString()
Write-Info ('[manifest] {0} file(s), {1} of payload' -f $entries.Count, (Format-Bytes $totalBytes))

# Independent re-implementation of the consumer's both-directions check. This is deliberately not
# a call into the launcher: two implementations disagreeing is the signal we want.
$listed = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($e in $entries) { [void]$listed.Add($e.path) }
$onDisk = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($f in (Get-ChildItem -LiteralPath $bundleDir -Recurse -File)) {
    $rel = $f.FullName.Substring($rootLen).TrimStart('\') -replace '\\', '/'
    if ($rel -eq $script:BUNDLE_MANIFEST_NAME) { continue }
    [void]$onDisk.Add($rel)
}
$missing = @($listed | Where-Object { -not $onDisk.Contains($_) })
$extra   = @($onDisk | Where-Object { -not $listed.Contains($_) })
if ($missing.Count -gt 0) { Stop-Build ('manifest lists files that are not on disk: ' + ($missing -join ', ')) }
if ($extra.Count -gt 0)   { Stop-Build ('files on disk are not in the manifest: ' + ($extra -join ', ')) }
Write-Info '[manifest] both-directions self-check OK'

# endregion

# ============================================================================
# region 6. ZIP + SHA256SUMS
# ============================================================================

if ($NoZip) {
    Write-Info ''
    Write-Info ('[done] bundle directory ready (no zip requested): {0}' -f $bundleDir)
} else {
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Write-Info ''
    Write-Info ('[zip] creating {0}' -f $zipPath)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # includeBaseDirectory = false: the bundle files sit at the zip root, so Windows "Extract All"
    # produces one folder named after the zip instead of a doubled folder.
    [System.IO.Compression.ZipFile]::CreateFromDirectory($bundleDir, $zipPath,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $sw.Stop()
    $zipSha = Get-Sha256Lower -Path $zipPath
    $zipLen = (Get-Item -LiteralPath $zipPath).Length
    Write-Info ('[zip] {0}  ({1}, {2:N0} s)' -f (Split-Path -Leaf $zipPath), (Format-Bytes $zipLen), $sw.Elapsed.TotalSeconds)
    Write-Info ('[zip] sha256 {0}' -f $zipSha)

    $sumsPath = Join-Path $OutRoot 'SHA256SUMS.txt'
    Write-TextNoBom -Path $sumsPath -Text (('{0}  {1}' -f $zipSha, (Split-Path -Leaf $zipPath)) + "`r`n")
    Write-Info ('[zip] wrote {0}' -f $sumsPath)
}

# endregion

# ============================================================================
# region 7. REPORT
# ============================================================================

Write-Info ''
Write-Info '=========================================================================='
Write-Info ' assembled'
Write-Info '=========================================================================='
Write-Info (' bundle    : {0}' -f $bundleDir)
Write-Info (' files     : {0} (+ {1})' -f $entries.Count, $script:BUNDLE_MANIFEST_NAME)
Write-Info (' payload   : {0}' -f (Format-Bytes $totalBytes))
Write-Info ''
Write-Info ' next: verify with the shipped launcher, from the bundle directory:'
Write-Info ('   powershell -NoProfile -ExecutionPolicy Bypass -Command ". ''{0}\Start-MoeDirect.ps1'' -LibraryMode; Assert-BundleIntegrity -Root ''{0}''"' -f $bundleDir)
Write-Info ''
Write-Info ' NOTES:'
Write-Info '  1. (resolved) repack_log.jsonl: the launcher now passes --log-path on every repack'
Write-Info '     call, pointing outside the bundle - in-bundle writes are zero (RC-1 fix,'
Write-Info '     selftest-locked). Old rehearsal workaround text removed.'
if (-not $VcRuntimeDir) {
    Write-Info '  2. MSVC runtime is NOT in the bundle. llama-server and every ggml DLL import'
    Write-Info '     MSVCP140.dll / VCRUNTIME140.dll / VCRUNTIME140_1.dll, and the ggml-cpu-*'
    Write-Info '     variants also import VCOMP140.DLL. Documented as a prerequisite in the'
    Write-Info '     README (v0.2 decision); or re-run with -VcRuntimeDir <redist dir>.'
}
Write-Info '  3. (resolved) THIRD_PARTY_NOTICES.md recomputed for this bundle composition'
Write-Info '     (CUDA runtime DLLs, CPython 3.11.9 embeddable, llama.cpp tree DLLs) - RS 6-5.'
Write-Info ''
exit 0

# endregion
