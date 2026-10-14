package core

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Conflict 9 of reviews/_gol_conflicts_8_user_20261010.md: when a model needs its setup (the
// repack), the start shows a plan once (what is written, where, about n G written, the drive's
// free m G and what is left k G, the last repack's duration for this model and mode or «unknown»
// the first time); Enter runs it, Esc writes nothing. The numbers are the PowerShell launcher's
// (Invoke-Preflight PS:3650-3708 with Get-ExpectedRepackBytes PS:3573 and the stale reclaim
// PS:3624-3645); a drive too small is refused there (free < need, fail_resource). The run after
// consent follows the run site PS:14136-14156: the stale/partial artifacts are cleared, the
// startup probe, the repacker (Invoke-Repacker PS:12353-12436), the format gate, the speed check.
// A cancel kills the repacker and removes everything it left, so nothing half-made remains and
// the next start begins from scratch ("v1 has no resume"). The duration is recorded.
//
// GL-10 (plan cell GL-10): the run's startup probe is ported (sweep.go: Invoke-StartupProbe
// with Write-ProbeBinding, before the repacker, PS:14136-14137), and the speed check is the QD
// sweep of the start that follows the setup (Resolve-QdSweep, after the format gate of that
// start, PS (8b) :14216-14240): the setup's last stage is the verify gate, then the start runs
// gate, sweep and spawn, so the plan box's «검증 → 속도 확인 → 서버가 저절로 시작» is the order the
// user sees. The template setup of a model outside the catalog (conflict 14; the template
// repack, --experimental-arch-template without --plan, PS:14153-14156) is planned and run here
// too: its plan text is the derive plan's (PS:14095-14101 prints the captured text rather than
// running --plan a second time), its gate binds the arch-template lock id and the bytes of the
// derived.expect.json the repacker wrote (PS:14176-14185).

// Repack constants (PS:777, :1215, :59).
const (
	virtualRepackReserveMB = 128 // VIRTUAL_REPACK_RESERVE_MB
	planTimeout            = 120 * time.Second
	RepackNoResume         = "v1 has no resume: an interrupted repack restarts from the beginning."
)

// RepackNeededError is a start refused because the selected mode has no usable artifacts: the
// setup (the repack) has to run first, after the plan and the user's consent. Its status is
// the failure stage PowerShell holds there (fail_repack, exit 3).
type RepackNeededError struct {
	Status    Status
	Mode      string
	OutputDir string
	Reason    string
	Partial   bool // an interrupted setup was found
	Stale     bool // artifacts present that cannot be bound to this file (pin mismatch)
}

func (e *RepackNeededError) Error() string {
	return fmt.Sprintf("repack needed (%s) in %s: %s", e.Mode, e.OutputDir, e.Reason)
}

// RepackPlan is what the plan box shows and what RunRepack runs.
type RepackPlan struct {
	ModelPath string
	ProfileID string
	Mode      string
	OutputDir string
	Reason    string   // why the setup is needed
	Writes    []string // the files the repack writes (full paths)
	Deletes   []string // the files removed first (stale, partial, or a forced re-run), full paths
	Partial   bool
	Stale     bool
	Forced    bool // the artifacts were usable or failed their gate and the user asked for a new setup

	DiskRoot   string
	NeedKnown  bool  // false when the expect carries no expert_bytes_total (PowerShell then needs 0)
	NeedMiB    int64 // about n written
	FreeMiB    int64 // the drive's free space now
	ReclaimMiB int64 // freed by the deletions above
	LeftMiB    int64 // free + reclaim - need
	Refusal    error // the disk-shortage refusal (fail_resource), nil when it fits

	LastDuration    time.Duration // the last repack of this model and mode
	HasLastDuration bool          // false = «unknown» (first repack)

	Exe      string
	ArgvRun  []string
	ArgvPlan []string
	WorkDir  string
	LogDir   string

	PlanText     string // the repacker's --plan output, once PlanText ran
	PlanBinBytes int64  // "experts.bin expected size=N B" from that output (the box's n since GL-10)

	// GL-10: a model outside the catalog (the template setup), and what the run's probe binds.
	Template  bool
	sourceTag string
	derived   *Profile

	expectSha string
	at        time.Time
}

// RepackStage is one step of a run.
type RepackStage string

const (
	RepackStageClear  RepackStage = "clear"  // stale / partial artifacts removed
	RepackStageProbe  RepackStage = "probe"  // the startup probe (provisional, decides nothing)
	RepackStageRepack RepackStage = "repack" // the repacker
	RepackStageVerify RepackStage = "verify" // the format gate
	RepackStageSpeed  RepackStage = "speed"  // the speed check: the start's sweep step (StepSweep) since GL-10
	RepackStageDone   RepackStage = "done"
)

// RepackEvent is one progress report of a run.
type RepackEvent struct {
	Stage    RepackStage
	Line     string  // a repacker output line, or the stage's own line
	Fraction float64 // 0..1 within the stage when the line tells it; -1 when unknown
	Skipped  bool
	Elapsed  time.Duration
}

// RepackResult is a finished run.
type RepackResult struct {
	Duration time.Duration
	Gate     GateResult
	ExitCode int
	OutLog   string
	ErrLog   string
	Skipped  []string    // the stages not run, with the reason (none since GL-10)
	Probe    ProbeResult // the startup probe of this run (GL-10)
}

// RepackExecutor runs the repacker. Run writes the child's stdout to outLog and stderr to
// errLog, calls line for each complete stdout line as it appears, and returns the exit code.
// When ctx ends it must kill the child, wait for it, and return ctx's error.
type RepackExecutor interface {
	Run(ctx context.Context, exe string, argv []string, workDir string, env []string, outLog, errLog string, line func(string)) (int, error)
}

// ProcessExecutor runs the real repacker as an owned child (the job object kills it with the
// launcher). It is not used by the tests.
type ProcessExecutor struct{}

// Run implements RepackExecutor.
func (ProcessExecutor) Run(ctx context.Context, exe string, argv []string, workDir string, env []string, outLog, errLog string, line func(string)) (int, error) {
	c, err := startOwnedChild(exe, argv, env, workDir, outLog, errLog)
	if err != nil {
		return -1, err
	}
	defer c.Close()
	sent := 0
	flush := func() {
		lines := readChildStderrComplete(outLog) // complete lines of the stdout file
		for ; sent < len(lines); sent++ {
			if line != nil {
				line(lines[sent])
			}
		}
	}
	for {
		code, exited := c.WaitExit(500 * time.Millisecond)
		flush()
		if exited {
			return int(code), nil
		}
		if ctx.Err() != nil {
			c.ForceKill() // TerminateProcess, then the 10 s wait (PS:12391-12393)
			c.WaitExit(10 * time.Second)
			return -1, ctx.Err()
		}
	}
}

// repackArgv is the argument list of Invoke-Repacker (PS:12355-12376) without the arch template.
func repackArgv(cat *Catalog, root string, p *Profile, modelPath, outDir, mode, logPath string, planOnly bool) (string, []string) {
	exe := filepath.Join(root, cat.Runtime.RepackerExe)
	var argv []string
	for _, a := range cat.Runtime.RepackerArgv {
		if strings.HasPrefix(a, "./") || strings.HasPrefix(a, `.\`) {
			a = filepath.Join(root, a[2:])
		}
		argv = append(argv, a)
	}
	if planOnly {
		argv = append(argv, "--plan")
	}
	if mode == RepackVirtual {
		argv = append(argv, "--mode", RepackVirtual)
	}
	argv = append(argv, "--profile", p.ID, "--model", modelPath, "--out", outDir, "--scope", p.RoutedScope)
	argv = append(argv, "--log-path", logPath)
	return exe, argv
}

// templateRunArgv is Invoke-Repacker -ArchTemplate without -PlanOnly (PS:12355-12384,
// :14153-14156): the template plan's argv without --plan.
func templateRunArgv(cat *Catalog, root, modelPath, outDir, logPath string) (string, []string) {
	exe, argv := templatePlanArgv(cat, root, modelPath, outDir, logPath)
	var out []string
	for _, a := range argv {
		if a != "--plan" {
			out = append(out, a)
		}
	}
	return exe, out
}

// expectedRepackBytes is Get-ExpectedRepackBytes (PS:3573-3580).
func expectedRepackBytes(expectPath string) (int64, bool) {
	v, err := readJSONFileStrict(expectPath)
	if err != nil {
		return 0, false
	}
	o, _ := asObj(v)
	return jNonNegInt(o["expert_bytes_total"])
}

// psLongMiB is PowerShell's [long]($bytes / 1MB): banker's rounding of the quotient.
func psLongMiB(b int64) int64 { return int64(math.RoundToEven(float64(b) / 1048576.0)) }

func filesBytes(dir string, names []string) (int64, []string) {
	var sum int64
	var present []string
	for _, n := range names {
		p := filepath.Join(dir, n)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			sum += st.Size()
			present = append(present, p)
		}
	}
	return sum, present
}

// PlanRepack computes the plan of the setup of req's model in req's mode (the preset's mode
// when the request names none). force plans a new setup over usable or gate-failed artifacts
// (conflict 16's «Enter 다시 리팩»). The error is the disk-shortage refusal (the plan is
// returned with it, plan.Refusal set), a model the draft cannot repack (a template model, an
// unreadable file), or, without force, a *GateError for artifacts that failed their gate and a
// plain error for usable ones (no plan then).
func (l *Launcher) PlanRepack(ctx context.Context, req Request, force bool) (*RepackPlan, error) {
	cat, err := ReadCatalog(l.opt.CatalogPath, l.opt.BundleRoot)
	if err != nil {
		return nil, err
	}
	p := strings.Trim(strings.TrimSpace(req.ModelPath), `"`)
	if !fileExists(p) {
		return nil, stop(StatusFailModelPath, "model file not found: %s", p)
	}
	if abs, err := filepath.Abs(p); err == nil {
		p = abs
	}
	set, le := readModelShardSet(p)
	if le != nil {
		return nil, le
	}
	tplOn, _ := l.archTemplateOn()
	sel, err := l.resolveProfileSelection(ctx, cat, set, templateAdmissible(tplOn, set.Arch), req.ProfileID)
	if err != nil {
		return nil, err
	}
	prof := sel.Profile
	mode := strings.ToLower(strings.TrimSpace(req.RepackMode))
	if sel.Kind == SelectionTemplate {
		// GL-10 (addendum 1 of the dispatch; GL-14 landing section 4 item 1): the template
		// setup. The derive plan (--plan --experimental-arch-template, writes nothing) gives
		// the derived profile; virtual stays refused (VR1 §0, F-PF §7.6; PS:13923).
		if mode == RepackVirtual {
			return nil, stop(StatusFailCustomArgs, derivedVirtualRefusal)
		}
		planOut, err := resolveOutputDir(Request{ModelPath: p, OutDir: req.OutDir}, false)
		if err != nil {
			return nil, err
		}
		dp, err := l.derivePlan(ctx, cat, p, planOut, set, sel.Shas)
		if err != nil {
			return nil, err
		}
		prof, mode = dp, RepackPacked
	}
	if mode == "" && !req.NoPreset && prof.Derived == nil {
		if pre, ps := l.LoadPreset(ctx, p, prof.ID); ps == PresetApplies {
			mode = pre.Values["repack_mode"]
		}
	}
	if mode == "" {
		mode = RepackPacked
	}
	if mode != RepackPacked && mode != RepackVirtual {
		return nil, stop(StatusFailCustomArgs, "invalid -RepackMode '%s': expected packed or virtual", req.RepackMode)
	}
	req.ModelPath = p
	out, err := resolveOutputDir(req, mode == RepackVirtual)
	if err != nil {
		return nil, err
	}
	gateID, gateSha := prof.ID, prof.ExpectSha256
	if prof.Derived != nil {
		var gerr error
		if gateID, gateSha, gerr = derivedGateIdentity(prof); gerr != nil {
			return nil, gerr
		}
	}
	g := CheckFormat(mode, out, gateID, gateSha)
	plan := &RepackPlan{ModelPath: p, ProfileID: prof.ID, Mode: mode, OutputDir: out, Partial: g.Partial,
		expectSha: prof.ExpectSha256, at: l.os.now(), WorkDir: l.opt.BundleRoot, sourceTag: cat.SourceTag}
	deleteSet := PackedArtifactSet
	plan.Writes = []string{filepath.Join(out, "experts.bin"), filepath.Join(out, "manifest.json"), filepath.Join(out, "verify_report.json")}
	if prof.Derived != nil {
		plan.Template, plan.derived = true, prof
		plan.Writes = append(plan.Writes, prof.Derived.ExpectPath)
		if gateSha == "" && g.State != GateNeedsRepack {
			// artifacts without the derived expect the gate binds to: a new setup is needed
			g = GateResult{Mode: mode, State: GateNeedsRepack, Reason: DerivedExpectFileName + " is missing in " + out}
		}
	}
	if mode == RepackVirtual {
		deleteSet = append([]string{"manifest.json"}, VirtualArtifactSet...)
		plan.Writes = []string{filepath.Join(out, "manifest.json"), filepath.Join(out, "plan_report.json")}
	}
	stale := mode == RepackPacked && sel.Kind == SelectionMismatch && g.State != GateNeedsRepack && !g.Partial
	switch {
	case stale:
		plan.Stale, plan.Reason = true, "the artifacts in "+out+" cannot be shown to come from this file (the catalog pin does not match these bytes)"
	case g.State == GateNeedsRepack:
		plan.Reason = g.Reason
	case force:
		plan.Forced = true
		plan.Reason = "a new setup was asked for (" + g.Line() + ")"
	case g.State == GateFail:
		return nil, &GateError{LaunchError: stop(StatusFailGateVerify, "%s", g.Reason), Gate: g}
	default:
		return nil, fmt.Errorf("no repack needed: %s", g.Line())
	}
	if plan.Partial || plan.Stale || plan.Forced {
		var b int64
		b, plan.Deletes = filesBytes(out, deleteSet)
		plan.ReclaimMiB = b / 1048576 // Get-StaleArtifactReclaimMb: floor
	}
	// Invoke-Preflight's disk half.
	free, _, root, verr := l.os.volumeFree(out)
	if verr != nil {
		return nil, stop(StatusFailResource, "output volume query failed: %v", verr)
	}
	plan.FreeMiB, plan.DiskRoot = free, root
	if mode == RepackVirtual {
		plan.NeedMiB, plan.NeedKnown = virtualRepackReserveMB, true
	} else if prof.Derived != nil {
		// the derived path's resource gate runs against the plan's figure (PS:14069, :3467)
		plan.NeedMiB, plan.NeedKnown = psLongMiB(prof.Derived.ExpertBytesTotal), true
		plan.PlanText = prof.Derived.PlanText
	} else if b, ok := expectedRepackBytes(prof.ExpectPath); ok {
		plan.NeedMiB, plan.NeedKnown = psLongMiB(b), true
	}
	plan.LeftMiB = plan.FreeMiB + plan.ReclaimMiB - plan.NeedMiB
	if plan.NeedMiB > plan.FreeMiB+plan.ReclaimMiB {
		why := fmt.Sprintf("disk preflight hard stop: need %d MB, free %d MB", plan.NeedMiB, plan.FreeMiB)
		if plan.ReclaimMiB > 0 {
			why += fmt.Sprintf(" (+%d MB reclaimable by replacing the stale artifacts)", plan.ReclaimMiB)
		}
		plan.Refusal = stop(StatusFailResource, "%s", why)
	}
	if d, ok := l.LastRepackDuration(p, mode); ok {
		plan.LastDuration, plan.HasLastDuration = d, true
	}
	plan.LogDir = l.LogDir()
	logPath := filepath.Join(plan.LogDir, "repack_log.jsonl")
	if prof.Derived != nil {
		plan.Exe, plan.ArgvRun = templateRunArgv(cat, l.opt.BundleRoot, p, out, logPath)
		_, plan.ArgvPlan = templatePlanArgv(cat, l.opt.BundleRoot, p, out, logPath)
	} else {
		plan.Exe, plan.ArgvRun = repackArgv(cat, l.opt.BundleRoot, prof, p, out, mode, logPath, false)
		_, plan.ArgvPlan = repackArgv(cat, l.opt.BundleRoot, prof, p, out, mode, logPath, true)
	}
	l.diag.write("REPACK_PLAN", map[string]any{"model": p, "mode": mode, "out": out, "reason": plan.Reason,
		"need_mb": plan.NeedMiB, "free_mb": plan.FreeMiB, "reclaim_mb": plan.ReclaimMiB, "left_mb": plan.LeftMiB,
		"deletes": plan.Deletes, "refused": plan.Refusal != nil, "template": plan.Template})
	return plan, plan.Refusal
}

var planBinRe = regexp.MustCompile(`experts\.bin expected size=(\d+) B`)

func repackEnv() []string {
	return explicitEnv(map[string]string{"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}, os.Getenv)
}

// PlanText runs the repacker in --plan mode (header analysis, writes 0 bytes) with
// PLAN_TIMEOUT_S, and keeps its output in the plan (Invoke-Repacker -PlanOnly).
//
// GL-10 (F-LS 4.6: the plan box's numbers come from the repacker's --plan): a packed plan's n is
// the --plan's «experts.bin expected size», and the free space left and the disk refusal are
// computed again from it. A template plan already holds the derive plan's text (no second
// run, PS:14095-14101).
func (l *Launcher) PlanText(ctx context.Context, plan *RepackPlan, ex RepackExecutor) (string, error) {
	if plan.Template {
		return plan.PlanText, nil
	}
	if err := os.MkdirAll(plan.LogDir, 0o755); err != nil {
		return "", stop(StatusFailRepack, "repacker --plan: log directory: %v", err)
	}
	stamp := l.os.now().UTC().Format("20060102T150405Z")
	outLog := filepath.Join(plan.LogDir, "repack_plan_"+stamp+"_out.log")
	errLog := filepath.Join(plan.LogDir, "repack_plan_"+stamp+"_err.log")
	c, cancel := context.WithTimeout(ctx, planTimeout)
	defer cancel()
	code, err := ex.Run(c, plan.Exe, plan.ArgvPlan, plan.WorkDir, repackEnv(), outLog, errLog, nil)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return "", stop(StatusFailRepack, "repacker --plan exceeded the %d s deadline and was terminated; see %s", int(planTimeout.Seconds()), errLog)
		}
		if ctx.Err() != nil {
			return "", stop(StatusCancelledUser, "stop requested during the repacker --plan run")
		}
		return "", stop(StatusFailRepack, "repacker spawn failed: %v", err)
	}
	if code != 0 {
		return "", stop(StatusFailRepack, "repacker --plan exited abnormally (code=%d); see %s", code, errLog)
	}
	b, _ := os.ReadFile(outLog)
	plan.PlanText = string(b)
	if m := planBinRe.FindStringSubmatch(plan.PlanText); m != nil {
		plan.PlanBinBytes, _ = strconv.ParseInt(m[1], 10, 64)
	}
	if plan.Mode == RepackPacked && plan.PlanBinBytes > 0 {
		plan.NeedMiB, plan.NeedKnown = psLongMiB(plan.PlanBinBytes), true
		plan.LeftMiB = plan.FreeMiB + plan.ReclaimMiB - plan.NeedMiB
		plan.Refusal = nil
		if plan.NeedMiB > plan.FreeMiB+plan.ReclaimMiB {
			why := fmt.Sprintf("disk preflight hard stop: need %d MB, free %d MB", plan.NeedMiB, plan.FreeMiB)
			if plan.ReclaimMiB > 0 {
				why += fmt.Sprintf(" (+%d MB reclaimable by replacing the stale artifacts)", plan.ReclaimMiB)
			}
			plan.Refusal = stop(StatusFailResource, "%s", why)
		}
	}
	return plan.PlanText, plan.Refusal
}

var (
	repackLayerRe  = regexp.MustCompile(`^\[repack\] moe layer \d+ \((\d+)/(\d+)\) done`)
	repackVerifyRe = regexp.MustCompile(`^\[verify\] ([\d,]+)/([\d,]+) pairs`) // «13,824» (UX-R14)
)

// repackFraction reads the progress of one repacker line (repack_experts.py:1665, :2172).
func repackFraction(line string) (RepackStage, float64) {
	if m := repackLayerRe.FindStringSubmatch(line); m != nil {
		i, _ := strconv.ParseFloat(m[1], 64)
		n, _ := strconv.ParseFloat(m[2], 64)
		if n > 0 {
			return RepackStageRepack, i / n
		}
	}
	if m := repackVerifyRe.FindStringSubmatch(line); m != nil {
		i, _ := strconv.ParseFloat(strings.ReplaceAll(m[1], ",", ""), 64)
		n, _ := strconv.ParseFloat(strings.ReplaceAll(m[2], ",", ""), 64)
		if n > 0 {
			return RepackStageVerify, i / n
		}
	}
	if strings.HasPrefix(line, "[verify]") { // UX-R14: the check's start and end lines (:2134, :1705)
		return RepackStageVerify, -1
	}
	return RepackStageRepack, -1
}

// removeArtifacts is Remove-StaleRepackArtifacts / Invoke-PartialCleanup's deletion
// (PS:5215-5239): delete each listed file, then prove it absent.
func removeArtifacts(paths []string) []string {
	var failed []string
	for _, p := range paths {
		if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
			failed = append(failed, filepath.Base(p)+": "+err.Error())
			continue
		}
		if st, _ := fileAbsence(p); st != absAbsent {
			failed = append(failed, filepath.Base(p)+": still present after delete")
		}
	}
	return failed
}

// RunRepack runs a plan after the user's consent. progress gets every step and repacker line.
// ctx cancels: the repacker is killed and everything the run left in the output folder is
// removed (the next start begins from scratch), and the error is cancelled_user. Only one
// run of a plan is meant at a time; the instance and output locks of the start are taken by
// the start that follows, as PowerShell takes them before its repack.
func (l *Launcher) RunRepack(ctx context.Context, plan *RepackPlan, ex RepackExecutor, progress func(RepackEvent)) (RepackResult, error) {
	t0 := l.os.now()
	var res RepackResult
	emit := func(e RepackEvent) {
		e.Elapsed = l.os.now().Sub(t0)
		if progress != nil {
			progress(e)
		}
	}
	if plan == nil {
		return res, errors.New("no plan")
	}
	if plan.Refusal != nil {
		return res, plan.Refusal
	}
	_, statErr := os.Stat(plan.OutputDir)
	newDir := errors.Is(statErr, os.ErrNotExist) // the run made the folder: a cancel removes it (S8-LA-24)
	if err := os.MkdirAll(plan.OutputDir, 0o755); err != nil {
		return res, stop(StatusFailRepack, "output directory: %v", err)
	}
	if err := os.MkdirAll(plan.LogDir, 0o755); err != nil {
		return res, stop(StatusFailRepack, "repack log directory: %v", err)
	}
	cleanupSet := append([]string(nil), PackedArtifactSet...)
	if plan.Mode == RepackVirtual {
		cleanupSet = append([]string{"manifest.json"}, VirtualArtifactSet...)
	}
	if plan.Template {
		cleanupSet = append(cleanupSet, DerivedExpectFileName)
	}
	cleanup := func() []string {
		var paths []string
		for _, n := range cleanupSet {
			paths = append(paths, filepath.Join(plan.OutputDir, n))
		}
		return removeArtifacts(paths)
	}
	cancelled := func(stage RepackStage) (RepackResult, error) {
		failed := cleanup()
		os.Remove(filepath.Join(plan.OutputDir, probeScratchName))
		if newDir && len(failed) == 0 {
			if err := os.Remove(plan.OutputDir); err != nil && !errors.Is(err, os.ErrNotExist) {
				failed = append(failed, filepath.Base(plan.OutputDir)+": the folder the run made could not be removed: "+err.Error())
			}
		}
		l.diag.write("REPACK_CANCELLED", map[string]any{"stage": stage, "out": plan.OutputDir, "cleanup_failed": failed})
		if len(failed) > 0 {
			return res, stop(StatusFailPartialCleanup, "repack cancelled, but the partial output could not be removed: %s", strings.Join(failed, "; "))
		}
		return res, stop(StatusCancelledUser, "stop requested during the repack; the partial output was removed and the next start begins from scratch")
	}

	// 1. Clear what the plan said it would delete (after consent, before the repacker).
	if len(plan.Deletes) > 0 {
		if failed := removeArtifacts(plan.Deletes); len(failed) > 0 {
			l.diag.write("STALE_ARTIFACT_CLEANUP_FAILED", map[string]any{"failed": failed})
			return res, stop(StatusFailPartialCleanup, "stale repack artifact cleanup failed: %s", strings.Join(failed, "; "))
		}
		emit(RepackEvent{Stage: RepackStageClear, Line: fmt.Sprintf("%d earlier artifact file(s) deleted; the repack restarts from the current file", len(plan.Deletes)), Fraction: 1})
	}
	if ctx.Err() != nil {
		return cancelled(RepackStageClear)
	}
	// 2. The startup probe and its record (PS:14136-14137): provisional, it decides nothing.
	emit(RepackEvent{Stage: RepackStageProbe, Line: fmt.Sprintf("[probe] measuring SSD random read (%d samples x %d B unbuffered)...", probeSamples, probeBlockBytes(plan.OutputDir)), Fraction: 0})
	res.Probe = l.startupProbe(plan.OutputDir)
	note := l.writeProbeBinding(plan.sourceTag, plan.ProfileID, plan.expectSha, plan.OutputDir, res.Probe)
	emit(RepackEvent{Stage: RepackStageProbe, Line: res.Probe.Line, Fraction: 1})
	if note != "" {
		emit(RepackEvent{Stage: RepackStageProbe, Line: note, Fraction: 1})
	}
	if ctx.Err() != nil {
		return cancelled(RepackStageProbe)
	}

	// 3. The repacker.
	stamp := l.os.now().UTC().Format("20060102T150405Z")
	res.OutLog = filepath.Join(plan.LogDir, "repack_"+stamp+"_out.log")
	res.ErrLog = filepath.Join(plan.LogDir, "repack_"+stamp+"_err.log")
	emit(RepackEvent{Stage: RepackStageRepack, Line: "repacker started", Fraction: 0})
	code, err := ex.Run(ctx, plan.Exe, plan.ArgvRun, plan.WorkDir, repackEnv(), res.OutLog, res.ErrLog, func(line string) {
		st, f := repackFraction(line)
		emit(RepackEvent{Stage: st, Line: line, Fraction: f})
	})
	res.ExitCode = code
	if ctx.Err() != nil {
		return cancelled(RepackStageRepack)
	}
	if err != nil {
		return res, stop(StatusFailRepack, "repacker spawn failed: %v", err)
	}
	if code != 0 {
		return res, stop(StatusFailRepack, "repacker exited abnormally (code=%d); see %s", code, res.ErrLog)
	}
	post := "verify_report.json"
	if plan.Mode == RepackVirtual {
		post = "plan_report.json"
	}
	if !fileExists(filepath.Join(plan.OutputDir, post)) {
		return res, stop(StatusFailRepack, "repacker finished but %s was not produced", post)
	}

	// 4. The format gate, with the mode detected again for virtual (PS:14186-14196); a
	// template setup binds the lock id and the derived expect's bytes (PS:14176-14185).
	gateID, gateSha := plan.ProfileID, plan.expectSha
	if plan.Template {
		var gerr error
		if gateID, gateSha, gerr = derivedGateIdentity(plan.derived); gerr != nil {
			return res, gerr
		}
		if gateSha == "" {
			return res, stop(StatusFailGateVerify, "the derived expect could not be hashed - %s is missing", plan.derived.Derived.ExpectPath)
		}
	}
	g := CheckFormat(plan.Mode, plan.OutputDir, gateID, gateSha)
	g.At = l.os.now()
	res.Gate = g
	emit(RepackEvent{Stage: RepackStageVerify, Line: g.Line(), Fraction: 1})
	if g.State != GatePass {
		reason := g.Reason
		if g.State == GateNeedsRepack {
			reason = "the repack left no usable artifacts: " + g.Reason
		}
		return res, &GateError{LaunchError: stop(StatusFailGateVerify, "%s", reason), Gate: g}
	}

	res.Duration = l.os.now().Sub(t0)
	if err := l.recordRepackDuration(plan.ModelPath, plan.Mode, res.Duration); err != nil {
		l.diag.write("HISTORY_WRITE_FAILED", map[string]any{"reason": err.Error()})
	}
	l.diag.write("REPACK_DONE", map[string]any{"model": plan.ModelPath, "mode": plan.Mode, "ms": res.Duration.Milliseconds(),
		"gate": g.Line(), "out_log": res.OutLog, "err_log": res.ErrLog})
	emit(RepackEvent{Stage: RepackStageDone, Line: fmt.Sprintf("setup done in %s", res.Duration.Round(time.Second)), Fraction: 1})
	return res, nil
}

// --- the duration record (the draft's history file) ---

// RepackRecord is the last repack of one model and mode.
type RepackRecord struct {
	ModelPath  string `json:"model_path"`
	Mode       string `json:"mode"`
	DurationMs int64  `json:"duration_ms"`
	At         string `json:"at"`
}

func repackKey(modelPath, mode string) string {
	return strings.ToLower(representativePath(modelPath)) + "|" + mode
}

func (l *Launcher) recordRepackDuration(modelPath, mode string, d time.Duration) error {
	historyMu.Lock()
	defer historyMu.Unlock()
	h := readHistory(l.historyPath())
	if h.Repacks == nil {
		h.Repacks = map[string]RepackRecord{}
	}
	h.Repacks[repackKey(modelPath, mode)] = RepackRecord{ModelPath: modelPath, Mode: mode, DurationMs: d.Milliseconds(),
		At: l.os.now().UTC().Format(time.RFC3339)}
	return writeHistory(l.historyPath(), h)
}

// LastRepackDuration is the last recorded repack of a model in a mode (false = «unknown»).
func (l *Launcher) LastRepackDuration(modelPath, mode string) (time.Duration, bool) {
	historyMu.Lock()
	defer historyMu.Unlock()
	r, ok := readHistory(l.historyPath()).Repacks[repackKey(modelPath, mode)]
	if !ok || r.DurationMs <= 0 {
		return 0, false
	}
	return time.Duration(r.DurationMs) * time.Millisecond, true
}

// RemovePartialSetup is the cut-off screen's «delete half-made only» (H4; S8-LA-34; LFS Q5.11):
// Invoke-PartialCleanup's deletion (PS:5106-5137) of the four artifacts of PARTIAL_DELETE_SET in
// the model's packed output folder, under the instance, profile and output locks PowerShell takes
// before anything destructive (PS:5019-5049), and no repack after it. The action itself is the
// user's consent (PowerShell's y/N). A failed delete is fail_partial_cleanup.
func (l *Launcher) RemovePartialSetup(ctx context.Context, modelPath, profileID string) ([]string, error) {
	if l.activeSession() {
		return nil, stop(StatusFailInstanceLock, "a server started by this launcher is running; the half-made setup is left as it is")
	}
	rep := representativePath(modelPath)
	out, err := resolveOutputDir(Request{ModelPath: rep}, false)
	if err != nil {
		return nil, err
	}
	if st, _ := fileAbsence(filepath.Join(out, "experts.bin.partial")); st != absPresent {
		return nil, fmt.Errorf("no cut-off setup in %s", out)
	}
	h, err := acquireLauncherLocks(l.lockPrefix, profileID, out)
	if err != nil {
		return nil, err
	}
	defer h.release()
	var paths []string
	for _, n := range PackedArtifactSet {
		paths = append(paths, filepath.Join(out, n))
	}
	if failed := removeArtifacts(paths); len(failed) > 0 {
		l.diag.write("PARTIAL_CLEANUP_FAILED", map[string]any{"failed": failed})
		return nil, stop(StatusFailPartialCleanup, ".partial cleanup failed: %s", strings.Join(failed, "; "))
	}
	l.diag.write("PARTIAL_CLEANUP_OK", map[string]any{"out": out, "deleted": PackedArtifactSet})
	return paths, nil
}
