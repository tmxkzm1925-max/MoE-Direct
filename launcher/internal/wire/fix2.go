// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// FIX2-WIRE: the adapter's part of the user's PowerShell-conflict decisions
// (reviews/_gol_conflicts_8_user_20261010.md, the «확정» cells): the setup plan and run (row 9),
// the launcher file check (row 18-2), the «작동 확인» setting (row 18-1), the stop line in 4 Log
// (row 6), the format gates per mode on the list (row 16) and the failure reasons (rows 8, 18-1).

package wire

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// --- row 16: the format gates of both modes for the setup line ---

// modeChecks maps core's gate results of both modes (no cache across starts: the list reads
// them again whenever the model folders or a repack output change).
func modeChecks(mfs []core.ModeFormat) []ui.ModeCheck {
	var out []ui.ModeCheck
	for _, m := range mfs {
		st := ui.ModeNotPorted
		switch m.Result.State {
		case core.GatePass:
			st = ui.ModeReady
		case core.GateNeedsRepack:
			st = ui.ModeNeedsRepack
		case core.GateFail:
			st = ui.ModeCheckFailed
		}
		if m.Stale { // W7 duty 12 (GL-10): made from other model files than these
			st = ui.ModeStale
		}
		out = append(out, ui.ModeCheck{Mode: m.Mode, State: st})
	}
	return out
}

// --- row 9: the setup plan and its run ---

// planDoes says what the setup does, in the user's words.
func planDoes(p *core.RepackPlan) string {
	var parts []string
	if p.Template { // conflict 14: the template setup (TEMPLATE_COPY_SENTENCE, PS:888-889, :14120-14122)
		parts = append(parts, "outside the catalog: copies the experts the structure template selects, byte for byte, into one streaming file (experts.bin), then checks it",
			"experimental: no published measurement covers this model")
	} else if p.Mode == core.RepackVirtual {
		parts = append(parts, "writes a map of the experts inside the model file (no copy), then checks it")
	} else {
		parts = append(parts, "copies the model's experts into one streaming file (experts.bin), then checks it")
	}
	switch {
	case p.Partial:
		parts = append(parts, "an earlier setup was cut off: it starts again from the beginning")
	case p.Stale:
		parts = append(parts, "the files there do not match this model file: they are replaced")
	case p.Forced:
		parts = append(parts, "the files there are replaced")
	}
	if len(p.Deletes) > 0 {
		s := fmt.Sprintf("first removes %d earlier file(s)", len(p.Deletes))
		if p.ReclaimMiB > 0 {
			s += " (" + gText(p.ReclaimMiB) + " freed, counted as free)"
		}
		parts = append(parts, s)
	}
	return strings.Join(parts, " · ")
}

// planFrom is the plan box (conflict 9): the PowerShell preflight figures. The drive's free
// space counts what the deletions free (PowerShell's reclaim), so the box refuses exactly when
// core refuses (need > free + reclaim).
func planFrom(name string, p *core.RepackPlan) ui.RepackPlan {
	u := ui.RepackPlan{Model: name, Mode: p.Mode, WritesTo: p.OutputDir, Does: planDoes(p),
		DriveFreeGB: gb(p.FreeMiB + p.ReclaimMiB), WriteGB: ui.U, LeftGB: ui.U, LastS: ui.U}
	if p.NeedKnown {
		u.WriteGB, u.LeftGB = gb(p.NeedMiB), gb(p.LeftMiB)
	}
	if p.HasLastDuration {
		u.LastS = ui.K(p.LastDuration.Seconds())
	}
	return u
}

// repackCountRe reads «(i/n)» of a repacker layer line and «i/n pairs» of its verify line.
var repackCountRe = regexp.MustCompile(`\((\d+)/(\d+)\)|\] ([\d,]+)/([\d,]+) pairs`)

// setupStep applies one core repack event to the setup progress (H6).
func setupStep(sp *ui.SetupProgress, e core.RepackEvent, needMiB int64, packed bool) {
	sp.Elapsed = e.Elapsed
	switch e.Stage {
	case core.RepackStageClear:
		sp.Name = "clean up"
	case core.RepackStageProbe, core.RepackStageSpeed:
		return // skipped stages (not ported): the name stays
	case core.RepackStageVerify:
		if sp.Name != "check" { // UX-R14: the check's counts are its pairs, not the repack's layers
			sp.Done, sp.Total = ui.U, ui.U
		}
		sp.Name = "check"
	case core.RepackStageRepack:
		sp.Name = "repack"
	case core.RepackStageDone:
		sp.Name = "done"
	}
	if m := repackCountRe.FindStringSubmatch(e.Line); m != nil {
		i, n := m[1], m[2]
		if i == "" {
			i, n = m[3], m[4]
		}
		a, _ := strconv.ParseFloat(strings.ReplaceAll(i, ",", ""), 64)
		b, _ := strconv.ParseFloat(strings.ReplaceAll(n, ",", ""), 64)
		sp.Done, sp.Total = ui.K(a), ui.K(b)
	}
	if packed && e.Stage == core.RepackStageRepack && e.Fraction >= 0 && needMiB > 0 {
		sp.WrittenGB = ui.K(e.Fraction * float64(needMiB) / mib)
	}
}

// RepackPlan is the plan box (conflict 9): core's plan of the model's setup in the mode the
// screen holds. A mode whose files failed their check gets a new setup (the forced plan).
// Nothing is written; the plan is kept for Repack, and Esc simply leaves it unused.
func (b *Backend) RepackPlan(model string, s ui.Settings) (ui.RepackPlan, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return ui.RepackPlan{}, err
	}
	if f.ProfileID == "" && !f.Template {
		return ui.RepackPlan{Model: f.DisplayName}, fmt.Errorf("%s: no catalog profile and no structure template: %w", f.Name, ui.ErrNotPorted)
	}
	req := b.request(f.Path, s)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	p, err := b.planFn(ctx, req, false)
	var ge *core.GateError
	if errors.As(err, &ge) {
		p, err = b.planFn(ctx, req, true)
	}
	if p == nil {
		b.logf(ui.LogError, "setup plan %s: %s", f.DisplayName, refusalText(err))
		return ui.RepackPlan{Model: f.DisplayName}, err
	}
	// F-LS 4.6 (GL-10): the box's numbers come from the repacker's --plan, which writes nothing;
	// a disk shortfall is refused before it (Invoke-Preflight runs first, PS:14069-14078).
	if p.Refusal == nil && b.planTextFn != nil {
		if _, err := b.planTextFn(ctx, p, b.exec); err != nil && p.Refusal == nil {
			b.logf(ui.LogError, "setup plan %s: %s", f.DisplayName, refusalText(err))
			return ui.RepackPlan{Model: f.DisplayName}, err
		}
	}
	b.mu.Lock()
	b.plans[preKey(model, s)] = p
	b.mu.Unlock()
	if p.Refusal != nil {
		b.logf(ui.LogWarn, "setup plan %s refused: %s", f.DisplayName, refusalText(p.Refusal))
	} else {
		b.logf(ui.LogInfo, "setup plan %s (%s) into %s: need %d MB, free %d MB, left %d MB · %s", f.DisplayName, p.Mode,
			p.OutputDir, p.NeedMiB, p.FreeMiB, p.LeftMiB, p.Reason)
	}
	return planFrom(f.DisplayName, p), nil
}

// Repack runs the plan shown for this model and settings after the user's ⏎, then the start
// (conflict 9): its events are the setup progress first, then the start's. A cancel (ctrl+x)
// stops the repacker, removes what it wrote and ends cancelled_user (exit 2).
func (b *Backend) Repack(model string, s ui.Settings) (<-chan ui.StartEvent, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return nil, err
	}
	key := preKey(model, s)
	b.mu.Lock()
	p := b.plans[key]
	delete(b.plans, key)
	b.mu.Unlock()
	if p == nil {
		return nil, errors.New("no setup plan was shown for these settings")
	}
	if p.Refusal != nil {
		return nil, p.Refusal
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := b.newRun(model, f.Path, cancel)
	b.logf(ui.LogInfo, "setup %s (%s) started: %s", f.DisplayName, p.Mode, strings.Join(append([]string{p.Exe}, p.ArgvRun...), " "))
	out := make(chan ui.StartEvent, 4096)
	go func() {
		defer close(out)
		defer close(r.done)
		defer cancel()
		sp := ui.SetupProgress{Name: "repack", TotalGB: ui.U, LastS: ui.U}
		if p.NeedKnown && p.Mode == core.RepackPacked {
			sp.TotalGB = gb(p.NeedMiB)
		}
		if p.HasLastDuration {
			sp.LastS = ui.K(p.LastDuration.Seconds())
		}
		send := func(u ui.StartEvent) {
			select {
			case out <- u:
			default:
			}
		}
		res, err := b.repackFn(ctx, p, b.exec, func(e core.RepackEvent) {
			setupStep(&sp, e, p.NeedMiB, p.Mode == core.RepackPacked)
			if e.Skipped || e.Stage == core.RepackStageClear || e.Stage == core.RepackStageDone ||
				(e.Stage == core.RepackStageProbe && e.Fraction == 1) { // GL-10: the probe's reading
				b.logf(ui.LogInfo, "setup %s: %s", f.DisplayName, e.Line)
			}
			c := sp
			send(ui.StartEvent{Model: model, Setup: &c, Elapsed: e.Elapsed})
		})
		b.mu.Lock()
		b.stamp = "" // the output folder changed: the list and its gates are read again
		b.modes = nil
		b.mu.Unlock()
		b.dropPreflights(model)
		if err != nil {
			fe := core.StartEvent{Final: true, Err: err, Status: core.StatusOf(err), ExitCode: core.ExitCodeOf(err), ErrLog: res.ErrLog}
			b.mu.Lock()
			r.final = &fe
			b.mu.Unlock()
			b.setExit(fe.ExitCode)
			b.logf(ui.LogError, "setup %s ended: %s", f.DisplayName, refusalText(err))
			c := sp
			out <- ui.StartEvent{Model: model, Setup: &c, Failure: failureFrom(fe, b.now(), nil)}
			return
		}
		done := fmt.Sprintf("setup %s done in %s · %s", f.DisplayName, res.Duration.Round(time.Second), res.Gate.Line())
		if len(res.Skipped) > 0 {
			done += " · skipped: " + strings.Join(res.Skipped, " | ")
		}
		b.logf(ui.LogOK, "%s · the speed check (the QD sweep) and the start follow", done)
		ch, err := b.startFn(ctx, b.request(f.Path, s))
		if err != nil {
			fe := core.StartEvent{Final: true, Err: err, Status: core.StatusOf(err), ExitCode: core.ExitCodeOf(err)}
			b.mu.Lock()
			r.final = &fe
			b.mu.Unlock()
			b.setExit(fe.ExitCode)
			out <- ui.StartEvent{Model: model, Failure: failureFrom(fe, b.now(), nil)}
			return
		}
		b.pump(r, f, ch, out)
	}()
	return out, nil
}

// --- row 18-2: the launcher's own file check ---

// bundleFrom maps core's bundle check.
func bundleFrom(r core.BundleResult) ui.BundleCheck {
	switch r.State {
	case core.BundleOK:
		return ui.BundleCheck{State: ui.BundleOK}
	case core.BundleMismatch:
		n := r.Mismatches
		if n == 0 {
			n = 1 // the manifest itself is broken (Reason): one file differs
		}
		return ui.BundleCheck{State: ui.BundleBroken, Mismatches: n}
	case core.BundleDev:
		return ui.BundleCheck{State: ui.BundleDev}
	}
	return ui.BundleCheck{State: ui.BundleChecking}
}

// bundleLog is the 4 Log line of a finished check ("" while it runs).
func bundleLog(r core.BundleResult) (ui.LogLevel, string) {
	switch r.State {
	case core.BundleOK:
		return ui.LogOK, fmt.Sprintf("launcher file check: %d files match %s", r.Files, r.Manifest)
	case core.BundleDev:
		return ui.LogInfo, "launcher file check skipped: no bundle_manifest.json in " + r.Root + " (development build)"
	case core.BundleMismatch:
		var d []string
		for i, x := range r.Diffs {
			if i == 10 {
				d = append(d, fmt.Sprintf("… %d more", len(r.Diffs)-10))
				break
			}
			d = append(d, x.Path+" ("+x.Kind+")")
		}
		s := fmt.Sprintf("launcher file check: %d file(s) differ from %s: %s", r.Mismatches, r.Manifest, strings.Join(d, ", "))
		if r.Reason != "" {
			s = "launcher file check: the manifest cannot be used: " + r.Reason
		}
		return ui.LogError, s
	}
	return ui.LogInfo, ""
}

// Bundle is the launcher file check (conflict 18-2); its result goes to 4 Log once. The screen
// asks for it first after its first draw, and that first call starts core's background check
// (GL-13; later calls only read it).
func (b *Backend) Bundle() (ui.BundleCheck, error) {
	if b.bundleStart != nil {
		b.bundleStart()
	}
	r := b.bundleFn()
	c := bundleFrom(r)
	if lv, line := bundleLog(r); line != "" {
		b.mu.Lock()
		first := !b.bundleLogged
		b.bundleLogged = true
		b.mu.Unlock()
		if first {
			b.logf(lv, "%s", line)
		}
	}
	return c, nil
}

// --- row 18-1: the «작동 확인» setting ---

func smokeFromArg(arg string) (core.SmokeMode, bool) {
	switch strings.ToLower(strings.TrimSpace(arg)) {
	case "off":
		return core.SmokeOff, true
	case "on":
		return core.SmokeOn, true
	case "once", "first-once":
		return core.SmokeFirstOnce, true
	}
	return core.SmokeOff, false
}

func smokeUI(m core.SmokeMode) ui.SmokeMode {
	switch m {
	case core.SmokeOn:
		return ui.SmokeOn
	case core.SmokeFirstOnce:
		return ui.SmokeOnce
	}
	return ui.SmokeOff
}

var smokeWords = map[core.SmokeMode]string{core.SmokeOff: "off", core.SmokeOn: "on (after every start)",
	core.SmokeFirstOnce: "once per model and mode after the launcher opens"}

// setSmoke is ActSetSmoke: the setting holds for this launcher session (default off).
func (b *Backend) setSmoke(arg string) (ui.ActionResult, error) {
	m, ok := smokeFromArg(arg)
	if !ok {
		return ui.ActionResult{}, fmt.Errorf("check run: %q is not off, on or once", arg)
	}
	b.mu.Lock()
	b.smoke = m
	b.mu.Unlock()
	b.logf(ui.LogInfo, "check run: %s", smokeWords[m])
	return ui.ActionResult{Message: "check run: " + smokeWords[m]}, nil
}

func (b *Backend) smokeMode() core.SmokeMode {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.smoke
}

// request is requestFrom with the session's launcher settings: «작동 확인», and since GL-16
// the prompt-cache cap and the warm-start switches of 5 Settings.
func (b *Backend) request(path string, s ui.Settings) core.Request {
	req := requestFrom(path, s)
	req.Smoke = b.smokeMode()
	b.mu.Lock()
	sw := b.switches
	req.PromptCacheCap, req.Switches = b.cacheCap, &sw
	b.mu.Unlock()
	return req
}

// setLauncher sets a launcher setting of 5 Settings (GL-16): the prompt-cache cap (U-22) or a
// warm-start switch (OQ-6 (a)). A value outside its range is refused with the settings-typed
// text (F-LS 11.4) and nothing changes; a change applies at the next start (rows 6.1-6.2).
func (b *Backend) setLauncher(id ui.ActionID, arg string) (ui.ActionResult, error) {
	var name, val string
	switch id {
	case ui.ActSetCacheCap:
		c, err := core.ParseCap(arg)
		if err != nil {
			return ui.ActionResult{}, err
		}
		b.mu.Lock()
		b.cacheCap = c
		b.mu.Unlock()
		name, val = "prompt-cache cap", c.String()
	default:
		key := map[ui.ActionID]string{ui.ActSetWarmStart: "warmstart", ui.ActSetAutosave: "autosave"}[id]
		on, err := core.ParseSwitch(key, arg)
		if err != nil {
			return ui.ActionResult{}, err
		}
		b.mu.Lock()
		if key == "warmstart" {
			b.switches.WarmStart = on
		} else {
			b.switches.Autosave = on
		}
		b.mu.Unlock()
		name, val = map[string]string{"warmstart": "warm start", "autosave": "cache autosave"}[key], arg
	}
	b.mu.Lock()
	b.pre = map[string]*preEntry{} // every model's preflight takes the new value
	b.mu.Unlock()
	b.logf(ui.LogInfo, "%s: %s (applies at the next start)", name, val)
	return ui.ActionResult{Message: name + ": " + val}, nil
}

// smokeItemLogs are the check run's item lines in 4 Log, one per item with its result (GL-11,
// S8-LA-25; PowerShell's Write-Line per item): a failed item red, a passed or skipped one plain.
func smokeItemLogs(r *core.SmokeResult) []ui.LogLine {
	if r == nil {
		return nil
	}
	var out []ui.LogLine
	for _, it := range r.Items {
		lv := ui.LogInfo
		if !it.OK && !it.Skipped {
			lv = ui.LogError
		}
		out = append(out, ui.LogLine{Level: lv, Text: "check run " + it.Line()})
	}
	return out
}

// smokeLog is the 4 Log summary line of a check run ("" when none ran).
func smokeLog(r *core.SmokeResult) (ui.LogLevel, string) {
	if r == nil {
		return ui.LogInfo, ""
	}
	if r.OK {
		return ui.LogOK, "check run passed: " + r.Line()
	}
	return ui.LogError, "check run failed: " + r.Line()
}

// --- row 6: the launcher's line in 4 Log for each server stop ---

// stopLines are core's stop events as 4 Log lines (conflict 6): code 0 green, 2 and 5 yellow,
// any other red; the meaning is given for the other codes.
func stopLines(evs []core.ServerStopEvent) []ui.LogLine {
	var out []ui.LogLine
	for _, e := range evs {
		rec := &ui.StopRecord{Code: e.Code, ServerCode: ui.U}
		if e.ChildExit != nil {
			rec.ServerCode = ui.K(float64(*e.ChildExit))
		}
		lv := ui.LogError
		switch e.Code {
		case 0:
			lv = ui.LogOK
		case 2, 5:
			lv = ui.LogWarn
		default:
			rec.Meaning = e.Meaning
		}
		out = append(out, ui.LogLine{At: e.At, Level: lv, Text: e.Line(), Stop: rec})
	}
	return out
}

// --- rows 8 and 18-1: the failure reasons of a start ---

// loadFailure adds the out-of-memory reason of a load (conflict 8) to a failure.
func loadFailure(f *ui.Failure, err error) {
	le, ok := core.AsLaunchError(err)
	if !ok || le.Load == nil || le.Load.Cause != core.CauseOutOfMemory {
		return
	}
	switch le.Load.Device {
	case "vram":
		f.VRAMOut = true
		f.Why = "the model ran out of VRAM while loading"
	case "ram":
		f.Why = "the model ran out of system memory while loading"
	default:
		f.Why = "the model ran out of memory while loading"
	}
}

// templateNote is the setup note of a model outside the catalog (conflict 14).
func templateNote(f core.ModelFile) string {
	// UX-R12 addendum (finding 18): the template form of UX-R10; the hint is a literal of its own,
	// a key of the screens' table
	return "outside the catalog · " + f.Arch + " template settings · " + "⏎ shows its setup plan (the template repack)"
}

// --- H4 (GL-10): the cut-off setup ---

// partialFrom is the cut-off screen's figures: the half-made files, when the setup stopped,
// and the redo time from the last full setup (unknown before one).
func partialFrom(f core.ModelFile) ui.PartialSetup {
	pa := ui.PartialSetup{LayersDone: ui.U, LayersTotal: ui.U, FilesGB: ui.U, RedoMin: ui.U}
	if f.PartialBytes > 0 {
		pa.FilesGB = gbBytes(f.PartialBytes)
	}
	if f.LastRepack > 0 {
		pa.RedoMin = ui.K(f.LastRepack.Minutes())
	}
	if !f.PartialAt.IsZero() {
		pa.When = f.PartialAt.Local().Format("2006-01-02 15:04")
	}
	pa.Cause = "· the launcher was closed or stopped during the setup"
	return pa
}

// deletePartial is the cut-off screen's ctrl+d (H4, S8-LA-34): the half-made files are removed
// and nothing is repacked; the list is read again.
func (b *Backend) deletePartial(model string) (ui.ActionResult, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return ui.ActionResult{}, err
	}
	if b.partialRmFn == nil {
		return ui.ActionResult{}, fmt.Errorf("%s: %w", ui.ActDeletePartial, ui.ErrNotPorted)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	gone, err := b.partialRmFn(ctx, f.Path, f.ProfileID)
	if err != nil {
		b.logf(ui.LogError, "half-made setup of %s not removed: %s", f.DisplayName, refusalText(err))
		return ui.ActionResult{}, err
	}
	b.mu.Lock()
	b.stamp = "" // the output folder changed: the list and its gates are read again
	b.modes = nil
	b.mu.Unlock()
	b.dropPreflights(model)
	b.logf(ui.LogInfo, "half-made setup of %s removed (%d file(s)); nothing was repacked", f.DisplayName, len(gone))
	return ui.ActionResult{Message: "half-made files removed · ⏎ shows the setup plan"}, nil
}

// bundleStep is the H15 detail of the finished file check (live start 1 showed core's bare
// state word "dev" there).
func bundleStep(r core.BundleResult) string {
	switch r.State {
	case core.BundleOK:
		return fmt.Sprintf("%d files match", r.Files)
	case core.BundleDev:
		return "dev build · skipped"
	case core.BundleMismatch:
		n := r.Mismatches
		if n == 0 {
			n = 1
		}
		return fmt.Sprintf("%d differ", n)
	}
	return ""
}

// ownRequests keeps the launcher's own requests out of the serve page and the run summary: the
// ready warmup (PS:865, default on) and the check run finish before the run's ready time, and
// live start 1 showed the warmup as «1 requests done», TTFT 9.7 s and a 1000000 tok/s last reply
// (its one answer token in 0.00 ms). Done entries at or before readyAt are dropped and their
// counts subtracted; the subtraction freezes once an entry after readyAt is seen (the log is read
// in order), so an early entry later trimmed from Done (core keeps 64) stays subtracted.
type ownRequests struct {
	frozen       bool
	n            int
	answerTokens int64
	answerTime   time.Duration
	promptTokens int64
	promptTime   time.Duration
}

func (o *ownRequests) drop(tm core.TimingSummary, readyAt time.Time) core.TimingSummary {
	if readyAt.IsZero() {
		return tm
	}
	if !o.frozen {
		*o = ownRequests{}
		for _, d := range tm.Done {
			if d.At.After(readyAt) {
				o.frozen = true
				continue
			}
			o.n++
			o.answerTokens += d.AnswerTokens
			o.answerTime += d.AnswerTime
			o.promptTokens += d.PromptTokens
			o.promptTime += d.TTFT
		}
	}
	out := tm
	out.Done = nil
	for _, d := range tm.Done {
		if d.At.After(readyAt) {
			out.Done = append(out.Done, d)
		}
	}
	out.Requests = max(tm.Requests-o.n, 0)
	out.AnswerTokens = max(tm.AnswerTokens-o.answerTokens, 0)
	out.AnswerTime = max(tm.AnswerTime-o.answerTime, 0)
	out.PromptTokens = max(tm.PromptTokens-o.promptTokens, 0)
	out.PromptTime = max(tm.PromptTime-o.promptTime, 0)
	return out
}
