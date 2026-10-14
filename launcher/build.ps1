#Requires -Version 5.1
<#
.SYNOPSIS
  The one build command of MoE-Direct-launcher (GL-01, version 8). It builds
  MoE-Direct-launcher.exe reproducibly and writes its build record next to it.

.DESCRIPTION
  Command (one line, from the repository root):
    powershell -NoProfile -ExecutionPolicy Bypass -File bench/moe-direct/launcher-go/build.ps1 -BundleVersion v0.3.2 -OutDir <dir>

  Steps, each fatal on failure (exit code in brackets):
    1. toolchain [1]: GOROOT = <GoTools>\go1.27.2\go; its VERSION file, `go version` and
       `go env GOVERSION` must read go1.27.2 ([A-GO]). The environment is fixed here:
       GOTOOLCHAIN=local, GOENV=off, GOWORK=off, GOFLAGS=-mod=readonly, CGO_ENABLED=0,
       GOOS=windows, GOARCH=amd64, GOAMD64=v1, GOEXPERIMENT and GODEBUG unset, and GOPATH,
       GOMODCACHE, GOCACHE inside <GoTools> unless given (-GoPath, -ModCache, -BuildCache).
    2. modules [2]: `go mod download`, then `go mod verify` must print "all modules verified".
    3. version gate [3]: TestS8LA30 of internal/version runs with MOE_BUNDLE_VERSION set to
       -BundleVersion. It fails when the launcher's version value (internal/version, the one
       place it comes from) and the bundle version differ, and the build stops there
       (S8-LA-30; F-LS edition 5.1; the Go counterpart of Start-MoeDirect.ps1:475-478).
    4. build [4]: go build -trimpath -buildvcs=false -ldflags=-buildid=
       -o <OutDir>\MoE-Direct-launcher.exe ./cmd/MoE-Direct-launcher
    5. record [5]: <OutDir>\MoE-Direct-launcher.build.json (UTF-8 without BOM, LF): the
       toolchain, the go.sum digest, the flags, the binary sha256 and the fields below.
    6. verify [7], only with -VerifyRecord <record>: every field of that record except
       built_utc must equal the same field of this build. This is the detecting gate of the
       build record; S8-RE-01 runs it on a second build from a clean module cache.

  Digests are sha256 in lower-case hex. go_sum_sha256 and go_mod_sha256 are taken over the
  file with CR LF turned into LF (the bytes git stores, so a checkout's line endings do not
  move them). source_sha256 is the sha256 of the lines "<sha256 of the file, CR LF as LF>
  <two spaces> <path>" (each LF-terminated, paths relative to the module root with /,
  sorted ordinally) over every *.go file, go.mod, go.sum and build.ps1 under the module root,
  except the subtree realhost/ of the module root; source_files counts the same set.
  realhost/ is End's real-host test harness (E-DRV, version 8): every file there carries the
  build tag realhost and the binary's package ./cmd/MoE-Direct-launcher does not import it,
  so none of its bytes reach the binary. Left in, every harness file End adds or changes would
  move the record without moving the binary, and a -VerifyRecord build against the published
  record would stop with exit 7 (R-BREC, version 8). Test files stay in the set, and no other
  path is left out.
#>
param(
    [Parameter(Mandatory = $true)] [string] $BundleVersion,
    [Parameter(Mandatory = $true)] [string] $OutDir,
    [string] $GoTools = '',
    [string] $GoPath = '',
    [string] $ModCache = '',
    [string] $BuildCache = '',
    [string] $VerifyRecord = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$Toolchain    = 'go1.27.2'
$BinaryName   = 'MoE-Direct-launcher.exe'
$RecordName   = 'MoE-Direct-launcher.build.json'
$Package      = './cmd/MoE-Direct-launcher'
$Flags        = @('-trimpath', '-buildvcs=false', '-ldflags=-buildid=')
$RecordFormat = 'moe-direct-launcher-build/1'
$HarnessDir   = 'realhost'   # module-root subtree left out of source_files / source_sha256 (see .DESCRIPTION)

function Stop-Build([int] $Code, [string] $Why) {
    [Console]::Error.WriteLine('[build] FAIL (' + $Code + '): ' + $Why)
    exit $Code
}

function Get-Sha256Hex([byte[]] $Bytes) {
    $h = [System.Security.Cryptography.SHA256]::Create()
    try { return (($h.ComputeHash($Bytes) | ForEach-Object { $_.ToString('x2') }) -join '') } finally { $h.Dispose() }
}

# The file's bytes with every CR that precedes an LF removed (CR LF -> LF).
function Get-LfBytes([string] $Path) {
    $b = [System.IO.File]::ReadAllBytes($Path)
    $out = New-Object 'System.Collections.Generic.List[byte]' ($b.Length)
    for ($i = 0; $i -lt $b.Length; $i++) {
        if ($b[$i] -eq 13 -and ($i + 1) -lt $b.Length -and $b[$i + 1] -eq 10) { continue }
        $out.Add($b[$i])
    }
    return ,$out.ToArray()
}

function Invoke-Go([string[]] $GoArgs) {
    & $script:GoExe @GoArgs | Out-Host
    return $LASTEXITCODE
}

# --- 1. toolchain and environment ---
$ModuleDir = $PSScriptRoot
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ModuleDir '..\..\..')).Path
if ($GoTools -eq '') { $GoTools = Join-Path $RepoRoot '.go_tools' }
if ($GoPath -eq '') { $GoPath = Join-Path $GoTools 'gopath' }
if ($ModCache -eq '') { $ModCache = Join-Path $GoTools 'modcache' }
if ($BuildCache -eq '') { $BuildCache = Join-Path $GoTools 'buildcache' }
$GoRoot = Join-Path $GoTools ($Toolchain + '\go')
$script:GoExe = Join-Path $GoRoot 'bin\go.exe'
if (-not (Test-Path -LiteralPath $script:GoExe)) { Stop-Build 1 ('no go.exe at ' + $script:GoExe) }
$verFile = @(Get-Content -LiteralPath (Join-Path $GoRoot 'VERSION'))
if ($verFile.Count -lt 1 -or $verFile[0] -cne $Toolchain) { Stop-Build 1 ('GOROOT VERSION file does not read ' + $Toolchain) }

$env:GOROOT = $GoRoot
$env:GOPATH = $GoPath
$env:GOMODCACHE = $ModCache
$env:GOCACHE = $BuildCache
$FixedEnv = [ordered]@{ GOTOOLCHAIN = 'local'; GOENV = 'off'; GOWORK = 'off'; GOFLAGS = '-mod=readonly';
                        CGO_ENABLED = '0'; GOOS = 'windows'; GOARCH = 'amd64'; GOAMD64 = 'v1' }
foreach ($k in $FixedEnv.Keys) { Set-Item -LiteralPath ('Env:' + $k) -Value $FixedEnv[$k] }
foreach ($n in @('GOEXPERIMENT', 'GODEBUG', 'MOE_BUNDLE_VERSION')) {
    if (Test-Path -LiteralPath ('Env:' + $n)) { Remove-Item -LiteralPath ('Env:' + $n) }
}

Set-Location -LiteralPath $ModuleDir
$goVersionLine = [string](& $script:GoExe version)
if ($LASTEXITCODE -ne 0 -or $goVersionLine -cne ('go version ' + $Toolchain + ' windows/amd64')) {
    Stop-Build 1 ('go version reads: ' + $goVersionLine)
}
$goVersion = [string](& $script:GoExe env GOVERSION)
if ($goVersion -cne $Toolchain) { Stop-Build 1 ('go env GOVERSION reads: ' + $goVersion) }
Write-Host ('[build] toolchain: ' + $goVersionLine)
Write-Host ('[build] caches: GOMODCACHE=' + $ModCache + ' GOCACHE=' + $BuildCache + ' GOPATH=' + $GoPath)

# --- 2. modules ---
if ((Invoke-Go @('mod', 'download')) -ne 0) { Stop-Build 2 'go mod download failed' }
$verify = @(& $script:GoExe mod verify)
if ($LASTEXITCODE -ne 0 -or ($verify -join "`n") -cne 'all modules verified') { Stop-Build 2 ('go mod verify: ' + ($verify -join ' | ')) }
Write-Host '[build] modules: all modules verified'

# --- 3. version gate ---
$env:MOE_BUNDLE_VERSION = $BundleVersion
$gate = Invoke-Go @('test', '-count=1', '-run', '^TestS8LA30$', '-v', './internal/version')
Remove-Item -LiteralPath 'Env:MOE_BUNDLE_VERSION'
if ($gate -ne 0) { Stop-Build 3 ('version gate: the launcher version differs from the bundle version ' + $BundleVersion) }
Write-Host ('[build] version gate: pass (bundle version ' + $BundleVersion + ')')

# --- 4. build ---
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$binary = Join-Path $OutDir $BinaryName
if ((Invoke-Go (@('build') + $Flags + @('-o', $binary, $Package))) -ne 0) { Stop-Build 4 'go build failed' }

# --- 5. record ---
$binBytes = [System.IO.File]::ReadAllBytes($binary)
$src = New-Object 'System.Collections.Generic.List[string]'
Get-ChildItem -LiteralPath $ModuleDir -Recurse -File | Where-Object {
    $_.Extension -ceq '.go' -or $_.Name -ceq 'go.mod' -or $_.Name -ceq 'go.sum' -or $_.Name -ceq 'build.ps1'
} | ForEach-Object {
    $rel = ($_.FullName.Substring($ModuleDir.Length).TrimStart('\') -replace '\\', '/')
    if (-not $rel.StartsWith($HarnessDir + '/', [System.StringComparison]::Ordinal)) { $src.Add($rel) }
}
$src.Sort([System.StringComparer]::Ordinal)
$srcText = New-Object System.Text.StringBuilder
foreach ($rel in $src) {
    $full = Join-Path $ModuleDir ($rel -replace '/', '\')
    [void]$srcText.Append((Get-Sha256Hex (Get-LfBytes $full)) + '  ' + $rel + "`n")
}

$fields = [ordered]@{
    record_format    = $RecordFormat
    binary           = $BinaryName
    binary_sha256    = (Get-Sha256Hex $binBytes)
    binary_bytes     = [string]$binBytes.Length
    toolchain        = $goVersion
    go_version       = $goVersionLine
    package          = $Package
    flags            = ($Flags -join ' ')
    env              = (@($FixedEnv.Keys | ForEach-Object { $_ + '=' + $FixedEnv[$_] }) -join ' ')
    go_sum_sha256    = (Get-Sha256Hex (Get-LfBytes (Join-Path $ModuleDir 'go.sum')))
    go_mod_sha256    = (Get-Sha256Hex (Get-LfBytes (Join-Path $ModuleDir 'go.mod')))
    source_files     = [string]$src.Count
    source_sha256    = (Get-Sha256Hex ([System.Text.Encoding]::UTF8.GetBytes($srcText.ToString())))
    modules          = 'all modules verified'
    launcher_version = $BundleVersion
    bundle_version   = $BundleVersion
    version_gate     = 'TestS8LA30 pass'
    build_command    = ('powershell -NoProfile -ExecutionPolicy Bypass -File bench/moe-direct/launcher-go/build.ps1 -BundleVersion ' + $BundleVersion + ' -OutDir <dir>')
    built_utc        = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
}
$keys = @($fields.Keys)
$json = New-Object System.Text.StringBuilder
[void]$json.Append("{`n")
for ($i = 0; $i -lt $keys.Count; $i++) {
    $v = ([string]$fields[$keys[$i]]).Replace('\', '\\').Replace('"', '\"')
    $comma = ','
    if ($i -eq $keys.Count - 1) { $comma = '' }
    [void]$json.Append('  "' + $keys[$i] + '": "' + $v + '"' + $comma + "`n")
}
[void]$json.Append("}`n")
$record = Join-Path $OutDir $RecordName
[System.IO.File]::WriteAllText($record, $json.ToString(), (New-Object System.Text.UTF8Encoding($false)))
Write-Host ('[build] binary: ' + $binary)
Write-Host ('[build] binary_sha256: ' + $fields['binary_sha256'])
Write-Host ('[build] record: ' + $record)

# --- 6. verify against an earlier record ---
if ($VerifyRecord -ne '') {
    $old = Get-Content -LiteralPath $VerifyRecord -Raw | ConvertFrom-Json
    $names = @($old.PSObject.Properties | ForEach-Object { $_.Name })
    $bad = 0
    foreach ($k in $keys) {
        if ($k -eq 'built_utc') { continue }
        $have = '<absent>'
        if ($names -contains $k) { $have = [string]$old.$k }
        if ($have -ceq [string]$fields[$k]) { Write-Host ('[verify] ' + $k + ': MATCH') }
        else { Write-Host ('[verify] ' + $k + ': DIFFERS (record ' + $have + ' / build ' + $fields[$k] + ')'); $bad++ }
    }
    foreach ($k in $names) {
        if ($keys -notcontains $k) { Write-Host ('[verify] ' + $k + ': DIFFERS (record field unknown to this build)'); $bad++ }
    }
    if ($bad -ne 0) { Stop-Build 7 ('' + $bad + ' field(s) differ from ' + $VerifyRecord) }
    Write-Host ('[verify] all fields equal ' + $VerifyRecord)
}
exit 0
