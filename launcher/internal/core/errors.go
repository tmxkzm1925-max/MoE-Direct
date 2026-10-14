package core

import (
	"errors"
	"fmt"
)

// Status is the launcher status enum of the PowerShell launcher (STATUS_EXIT,
// Start-MoeDirect.ps1:203-221). No status outside this table is produced.
type Status string

// The 17 statuses, PS:203-221.
const (
	StatusOK                 Status = "ok"
	StatusOKSmoke            Status = "ok_smoke"
	StatusCancelledUser      Status = "cancelled_user"
	StatusFailModelPath      Status = "fail_model_path"
	StatusFailResource       Status = "fail_resource"
	StatusFailInstanceLock   Status = "fail_instance_lock"
	StatusFailPartialCleanup Status = "fail_partial_cleanup"
	StatusFailRepack         Status = "fail_repack"
	StatusFailCustomArgs     Status = "fail_custom_args"
	StatusFailGateBundle     Status = "fail_gate_bundle"
	StatusFailGateCatalog    Status = "fail_gate_catalog"
	StatusFailGateVerify     Status = "fail_gate_verify"
	StatusFailGateEngineSeal Status = "fail_gate_engine_seal"
	StatusFailServerStart    Status = "fail_server_start"
	StatusFailRuntimeExit    Status = "fail_runtime_exit"
	StatusFailTeardown       Status = "fail_teardown"
	StatusFailSmoke          Status = "fail_smoke"
)

// statusExit is STATUS_EXIT, PS:203-221, entry for entry.
var statusExit = map[Status]int{
	StatusOK: 0, StatusOKSmoke: 0,
	StatusCancelledUser: 2,
	StatusFailModelPath: 3, StatusFailResource: 3, StatusFailInstanceLock: 3,
	StatusFailPartialCleanup: 3, StatusFailRepack: 3, StatusFailCustomArgs: 3,
	StatusFailGateBundle: 4, StatusFailGateCatalog: 4, StatusFailGateVerify: 4,
	StatusFailGateEngineSeal: 4,
	StatusFailServerStart:    5, StatusFailRuntimeExit: 5, StatusFailTeardown: 5,
	StatusFailSmoke: 6,
}

// ExitCode maps a status to its exit code (Get-StatusExitCode, PS:1832-1838): an unknown
// status is the most conservative failure, 5.
func ExitCode(s Status) int {
	if c, ok := statusExit[s]; ok {
		return c
	}
	return 5
}

// Known reports whether s is one of the 17 statuses.
func (s Status) Known() bool { _, ok := statusExit[s]; return ok }

// StatusHint is the one-line human verdict of each fail_* status (STATUS_HINT, PS:1806-1821).
func StatusHint(s Status) string { return statusHint[s] }

var statusHint = map[Status]string{
	StatusFailModelPath:      "the model path could not be used: file missing, unsupported GGUF, or an incomplete/ambiguous shard set",
	StatusFailResource:       "preflight stopped the run: not enough RAM or disk space for this configuration",
	StatusFailInstanceLock:   "another launcher instance holds the single-instance mutex or a profile/output/port lock",
	StatusFailPartialCleanup: "the leftover repack outputs could not be deleted or confirmed absent",
	StatusFailRepack:         "the repacker exited abnormally or produced no verify report",
	StatusFailCustomArgs:     StatusHintCustomArgs,
	StatusFailGateBundle:     "bundle integrity failed: its manifest, schema, or file set did not match the sealed bundle",
	StatusFailGateCatalog:    "models.json failed the catalog schema, the prefetch axis structure or the expect digest check",
	StatusFailGateVerify:     "the 7-item repack gate rejected the verify report or its manifest binding",
	StatusFailGateEngineSeal: "the engine refused to start and printed its policy gate reject line",
	StatusFailServerStart:    "the server never reached ready: spawn, port, listener PID, health, or an early exit",
	StatusFailRuntimeExit:    "the server exited unexpectedly after it had reached ready",
	StatusFailTeardown:       "shutdown did not complete cleanly: signal, grace period, or a surviving child/listener",
	StatusFailSmoke:          "a smoke assertion failed while the shutdown itself completed",
}

// LaunchError is the Go form of Stop-Launcher (PS:1948-1952): a terminal verdict with its
// status, its exit code and the PowerShell reason text.
type LaunchError struct {
	Status Status
	Reason string
	// Load is set when a start that ended before ready ran out of memory (conflict 8): a
	// sub-reason of fail_server_start, read from the server's own output.
	Load *LoadFailure
	// Port is set when a start that ended before ready found its port held by a process this
	// launcher did not start (GL-08; F-LS 4.3, conflict row 6 (1)): the owner is shown, never
	// ended.
	Port *PortHolder
}

// PortHolder is the process that holds a start's port. It is shown only: no action of the
// launcher ends a process the launcher did not start (rule 6; U-28 (ii)).
type PortHolder struct {
	Port int
	PID  int
	Name string // the image file name; "" when it could not be read
}

func (e *LaunchError) Error() string {
	return fmt.Sprintf("%s (exit %d): %s", e.Status, ExitCode(e.Status), e.Reason)
}

// ExitCode returns the exit code of the error's status.
func (e *LaunchError) ExitCode() int { return ExitCode(e.Status) }

func stop(s Status, format string, a ...any) *LaunchError {
	return &LaunchError{Status: s, Reason: fmt.Sprintf(format, a...)}
}

// AsLaunchError unwraps err into a *LaunchError when it is one.
func AsLaunchError(err error) (*LaunchError, bool) {
	var le *LaunchError
	if errors.As(err, &le) {
		return le, true
	}
	return nil, false
}

// ErrNotPorted is matched by errors.Is for every NotPortedError.
var ErrNotPorted = errors.New("not ported (after the user's decisions)")

// NotPortedError is returned where the slice would need a step of the PowerShell launcher
// that the draft does not port (DECISION_LEDGER.md:4197-4198) and no skipped-or-done value
// exists on this PC.
type NotPortedError struct {
	Family string // one of the Family* constants
	Detail string // why this call reached the family
}

func (e *NotPortedError) Error() string {
	if e.Detail == "" {
		return fmt.Sprintf("%s: not ported (after the user's decisions)", e.Family)
	}
	return fmt.Sprintf("%s: not ported (after the user's decisions) - %s", e.Family, e.Detail)
}

// Is makes errors.Is(err, ErrNotPorted) true.
func (e *NotPortedError) Is(target error) bool { return target == ErrNotPorted }

func notPorted(family, format string, a ...any) *NotPortedError {
	return &NotPortedError{Family: family, Detail: fmt.Sprintf(format, a...)}
}

// The families outside the slice (brief GOL-CORE, "Why" paragraph), with the PowerShell
// function each one stands for.
const (
	FamilyBundleGate     = "bundle manifest gate (Assert-BundleIntegrity PS:2187)"
	FamilyVerifyGates    = "verify, virtual plan and format gates (Assert-VerifyGate PS:5268, Assert-VirtualPlanGate PS:5453)"
	FamilyRepack         = "repack plan, confirmation and run (Invoke-Repacker PS:12353, PS:14087-14156)"
	FamilyPartialStale   = "partial and stale artifacts (Get-PartialMarkerState PS:5099, Invoke-PartialCleanup PS:5106, Remove-StaleRepackArtifacts PS:5215)"
	FamilyStartupProbe   = "startup probe and QD sweep (Invoke-StartupProbe PS:4190, Invoke-QdSweep PS:4719)"
	FamilyDriveSweep     = "drive sweep"
	FamilyManualPrefetch = "the manual prefetch pair K and N (F-PF; built by the manual prefetch surface)"
	FamilyPresets        = "user presets (Read-UserPreset PS:5653, Save-UserPreset PS:5716)"
	FamilyPrefetchArming = "prefetch bundle load and arming (Get-PrefetchLoadOnce PS:6983, New-PrefetchBundleView PS:7108)"
	FamilyCollect        = "collect runs (Resolve-CollectRequest PS:13242, Publish-CollectPendingManifest PS:8149)"
	FamilyCustomLabels   = "custom provenance and labels (Test-CustomProvenance PS:8857, Get-SurfaceAxes PS:3479)"
	FamilySmoke          = "smoke checklist (Invoke-SmokeChecklist PS:12665)"
	FamilyLauncherWarmup = "launcher warmup (Invoke-ReadyWarmup PS:14578, Invoke-LauncherWarmup PS:14593)"
	FamilyBrowserOpen    = "browser open (Open-BrowserBestEffort PS:14770)"
	FamilyServeCommand   = "typed serve command line (Read-ServeCommandLine PS:1936)"
	FamilyArchTemplate   = "arch-template setup (the template repack, PS:14153-14156)"
)

// NotPortedFamilies lists every family still outside the slice, in the brief's order. FIX2-CORE
// ported the six of PortedFamilies (conflicts 2, 9, 16, 18); their constants stay for the
// adapter's mapping.
var NotPortedFamilies = []string{
	FamilyDriveSweep, FamilyPresets, FamilyPrefetchArming, // GL-18: warm start left (GL-15 ported it); GL-10: the startup probe and the arch template left
	FamilyCollect, FamilyCustomLabels, FamilyBrowserOpen, FamilyServeCommand,
}

// PortedFamilies are the families FIX2-CORE ported: the bundle check (bundle.go), the format
// gates (gates.go), the repack plan and run with the stale and partial clearing (repack.go),
// the smoke checklist (smoke.go) and the launcher warmup (warmup.go).
// GL-10 ported the startup probe and the QD sweep (sweep.go) and the template setup (repack.go).
var PortedFamilies = []string{
	FamilyBundleGate, FamilyVerifyGates, FamilyRepack, FamilyPartialStale, FamilySmoke, FamilyLauncherWarmup,
	FamilyStartupProbe, FamilyArchTemplate,
}

// NotPorted returns the typed error of one family, for a caller (a screen action) that asks
// for a step outside the slice.
func NotPorted(family string) error { return &NotPortedError{Family: family} }
