package wire

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// testBackend is a Backend over stubs only (no core.Launcher): the file list is given and the
// core calls the tests reach are replaced.
func testBackend(files ...core.ModelFile) *Backend {
	b := New(nil, "models.json", "test")
	t0 := time.Unix(2000, 0)
	b.now = func() time.Time { return t0 }
	for _, f := range files {
		b.files[f.Path] = f
	}
	return b
}

// Conflict 9: the plan box values as the adapter gives them, the refusal by the same rule as
// core, «unknown» on the first setup, and the Esc path (the plan is shown, nothing runs).
func TestPlanBoxRow9(t *testing.T) {
	p := &core.RepackPlan{Mode: "packed", OutputDir: `D:\m\repack`, NeedKnown: true, NeedMiB: 61440, FreeMiB: 102400, LeftMiB: 40960}
	u := planFrom("gpt", p)
	near(t, "write", u.WriteGB, 60)
	near(t, "free", u.DriveFreeGB, 100)
	near(t, "left", u.LeftGB, 40)
	if u.LastS.Ok() || u.WritesTo != `D:\m\repack` || u.Mode != "packed" || !strings.Contains(u.Does, "experts.bin") {
		t.Errorf("plan %+v", u)
	}
	// A cut-off setup with reclaimable files: the drive's free space counts the reclaim, so the
	// box refuses exactly when core does (need > free + reclaim).
	p = &core.RepackPlan{Mode: "packed", NeedKnown: true, NeedMiB: 61440, FreeMiB: 40960, ReclaimMiB: 30720, LeftMiB: 10240,
		Partial: true, Deletes: []string{"a", "b"}, HasLastDuration: true, LastDuration: 1500 * time.Second}
	u = planFrom("gpt", p)
	if u.LeftGB.V < 0 || u.WriteGB.V > u.DriveFreeGB.V || u.LastS.V != 1500 || !strings.Contains(u.Does, "starts again from the beginning") ||
		!strings.Contains(u.Does, "removes 2 earlier file(s)") {
		t.Errorf("reclaim plan %+v", u)
	}
	p = &core.RepackPlan{Mode: "packed", NeedKnown: true, NeedMiB: 61440, FreeMiB: 20480, LeftMiB: -40960}
	if u = planFrom("gpt", p); !(u.WriteGB.V > u.DriveFreeGB.V && u.LeftGB.V < 0) {
		t.Errorf("short plan %+v", u)
	}
	if u = planFrom("gpt", &core.RepackPlan{Mode: "virtual", NeedKnown: true, NeedMiB: 128, FreeMiB: 1024, LeftMiB: 896}); !strings.Contains(u.Does, "no copy") {
		t.Errorf("virtual plan %+v", u)
	}

	// The adapter: a mode whose files failed their check is planned again with force; the plan
	// is kept for Repack; Esc runs nothing.
	f := core.ModelFile{Path: `D:\m\gpt.gguf`, Name: "gpt.gguf", DisplayName: "gpt", ProfileID: "gpt-profile", Catalog: true}
	b := testBackend(f)
	var calls []bool
	b.planFn = func(_ context.Context, req core.Request, force bool) (*core.RepackPlan, error) {
		calls = append(calls, force)
		if req.RepackMode != "virtual" || req.ModelPath != f.Path {
			t.Errorf("plan request %+v", req)
		}
		if !force {
			return nil, &core.GateError{LaunchError: &core.LaunchError{Status: core.StatusFailGateVerify, Reason: "item 4"}}
		}
		return &core.RepackPlan{Mode: "virtual", NeedKnown: true, NeedMiB: 128, FreeMiB: 1024, LeftMiB: 896, Forced: true}, nil
	}
	ran := false
	b.repackFn = func(context.Context, *core.RepackPlan, core.RepackExecutor, func(core.RepackEvent)) (core.RepackResult, error) {
		ran = true
		return core.RepackResult{}, nil
	}
	s := ui.Settings{"mode": "virtual"}
	got, err := b.RepackPlan(f.Path, s)
	if err != nil || len(calls) != 2 || calls[0] || !calls[1] || got.Model != "gpt" || !strings.Contains(got.Does, "replaced") {
		t.Fatalf("plan: %v %v %+v", err, calls, got)
	}
	// Esc: the ui goes back to 1 Models without a call; nothing ran, no run is current, and a
	// setup for other settings than the plan shown is refused.
	if ran || b.run != nil || b.serving() {
		t.Error("the plan box ran something")
	}
	if _, err := b.Repack(f.Path, ui.Settings{"mode": "packed"}); err == nil || ran {
		t.Errorf("a setup without its plan: %v", err)
	}
	if lines, _ := logLines(b); !strings.Contains(lines, "setup plan gpt (virtual)") {
		t.Errorf("log %s", lines)
	}
	// A refused plan comes back with its values (the box shows the short line) and runs nothing.
	b.planFn = func(context.Context, core.Request, bool) (*core.RepackPlan, error) {
		p := &core.RepackPlan{Mode: "packed", NeedKnown: true, NeedMiB: 61440, FreeMiB: 20480, LeftMiB: -40960,
			Refusal: &core.LaunchError{Status: core.StatusFailResource, Reason: "disk preflight hard stop: need 61440 MB, free 20480 MB"}}
		return p, p.Refusal
	}
	if got, err = b.RepackPlan(f.Path, ui.Settings{}); err != nil || got.LeftGB.V >= 0 {
		t.Errorf("refused plan: %v %+v", err, got)
	}
	if _, err := b.Repack(f.Path, ui.Settings{}); err == nil || ran {
		t.Errorf("a refused plan ran: %v", err)
	}
}

// logLines is the launcher's own events (no server log without a core.Launcher).
func logLines(b *Backend) (string, int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	var sb strings.Builder
	for _, e := range b.events {
		sb.WriteString(e.Text + "\n")
	}
	return sb.String(), len(b.events)
}

// Conflict 9: the setup's progress events, its cancel (exit 2, nothing left), and the start that
// follows a finished setup.
func TestRepackRunRow9(t *testing.T) {
	f := core.ModelFile{Path: `D:\m\gpt.gguf`, Name: "gpt.gguf", DisplayName: "gpt", ProfileID: "gpt-profile", Catalog: true}
	b := testBackend(f)
	plan := &core.RepackPlan{Mode: "packed", NeedKnown: true, NeedMiB: 36864, FreeMiB: 102400, LeftMiB: 65536, Exe: "r.exe"}
	b.planFn = func(context.Context, core.Request, bool) (*core.RepackPlan, error) { return plan, nil }
	b.repackFn = func(ctx context.Context, p *core.RepackPlan, _ core.RepackExecutor, progress func(core.RepackEvent)) (core.RepackResult, error) {
		progress(core.RepackEvent{Stage: core.RepackStageProbe, Skipped: true, Line: "startup probe skipped: not ported", Fraction: -1})
		progress(core.RepackEvent{Stage: core.RepackStageRepack, Line: "[repack] moe layer 3 (2/36) done", Fraction: 2.0 / 36, Elapsed: 4 * time.Second})
		<-ctx.Done()
		return core.RepackResult{}, &core.LaunchError{Status: core.StatusCancelledUser, Reason: "stop requested during the repack; the partial output was removed"}
	}
	if _, err := b.RepackPlan(f.Path, ui.Settings{}); err != nil {
		t.Fatal(err)
	}
	ch, err := b.Repack(f.Path, ui.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	first := <-ch
	second := <-ch
	if first.Setup == nil || second.Setup == nil || second.Setup.Name != "repack" || second.Setup.Done.V != 2 || second.Setup.Total.V != 36 {
		t.Fatalf("progress %+v %+v", first.Setup, second.Setup)
	}
	near(t, "written", second.Setup.WrittenGB, 2.0/36*36)
	near(t, "total", second.Setup.TotalGB, 36)
	// ctrl+x while the setup runs: the start's context ends, the run ends cancelled_user (2).
	stop, err := b.Stop(ui.StopCancel)
	if err != nil {
		t.Fatal(err)
	}
	var last ui.StopEvent
	for e := range stop {
		last = e
	}
	if last.Result == nil || last.Result.ExitCode != 2 || b.ExitCode() != 2 {
		t.Errorf("cancel %+v", last.Result)
	}
	var fail ui.StartEvent
	for e := range ch {
		fail = e
	}
	if fail.Failure == nil || fail.Failure.ExitCode != 2 || fail.Failure.Stage != "cancelled_user" {
		t.Errorf("setup end %+v", fail.Failure)
	}

	// A finished setup is followed by the start: its ready event closes the channel.
	b2 := testBackend(f)
	b2.planFn = b.planFn
	b2.repackFn = func(_ context.Context, _ *core.RepackPlan, _ core.RepackExecutor, progress func(core.RepackEvent)) (core.RepackResult, error) {
		progress(core.RepackEvent{Stage: core.RepackStageDone, Line: "setup done in 25m0s", Fraction: 1})
		return core.RepackResult{Duration: 25 * time.Minute}, nil
	}
	b2.startFn = func(_ context.Context, req core.Request) (<-chan core.StartEvent, error) {
		c := make(chan core.StartEvent, 2)
		c <- core.StartEvent{Final: true, Ready: true, Address: "http://127.0.0.1:8093", PID: 7}
		close(c)
		return c, nil
	}
	b2.RepackPlan(f.Path, ui.Settings{})
	ch, err = b2.Repack(f.Path, ui.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	var evs []ui.StartEvent
	for e := range ch {
		evs = append(evs, e)
	}
	if len(evs) != 2 || evs[0].Setup == nil || evs[0].Setup.Name != "done" || !evs[1].Ready {
		t.Errorf("setup then start %+v", evs)
	}
}

// Conflict 18-2: the launcher file check as the adapter gives it, logged once.
func TestBundleRow18(t *testing.T) {
	cases := []struct {
		in   core.BundleResult
		want ui.BundleCheck
	}{
		{core.BundleResult{State: core.BundlePending}, ui.BundleCheck{State: ui.BundleChecking}},
		{core.BundleResult{State: core.BundleOK, Files: 40}, ui.BundleCheck{State: ui.BundleOK}},
		{core.BundleResult{State: core.BundleMismatch, Mismatches: 3}, ui.BundleCheck{State: ui.BundleBroken, Mismatches: 3}},
		{core.BundleResult{State: core.BundleMismatch, Reason: "manifest version 2"}, ui.BundleCheck{State: ui.BundleBroken, Mismatches: 1}},
		{core.BundleResult{State: core.BundleDev, Root: `D:\moe-tools`}, ui.BundleCheck{State: ui.BundleDev}},
	}
	for _, c := range cases {
		if got := bundleFrom(c.in); got != c.want {
			t.Errorf("%+v → %+v, want %+v", c.in, got, c.want)
		}
	}
	b := testBackend()
	res := core.BundleResult{State: core.BundlePending}
	b.bundleFn = func() core.BundleResult { return res }
	if c, _ := b.Bundle(); c.State != ui.BundleChecking {
		t.Error(c)
	}
	res = core.BundleResult{State: core.BundleMismatch, Mismatches: 2, Manifest: "bundle_manifest.json",
		Diffs: []core.BundleDiff{{Path: "llama-server.exe", Kind: "hash"}, {Path: "x.dll", Kind: "missing"}}}
	b.Bundle()
	b.Bundle()
	lines, n := logLines(b)
	if n != 1 || !strings.Contains(lines, "2 file(s) differ") || !strings.Contains(lines, "llama-server.exe (hash)") {
		t.Errorf("log %q", lines)
	}
}

// Conflict 18-1: the «작동 확인» setting through ActSetSmoke, carried by every start request, and
// a failed check as the start's failure.
func TestSmokeSettingRow18(t *testing.T) {
	b := testBackend()
	if b.smokeMode() != core.SmokeOff || smokeUI(b.smokeMode()) != ui.SmokeOff {
		t.Error("default off")
	}
	for arg, want := range map[string]core.SmokeMode{"once": core.SmokeFirstOnce, "on": core.SmokeOn, "off": core.SmokeOff} {
		if r, err := b.Action(ui.ActSetSmoke, arg); err != nil || b.smokeMode() != want || !strings.HasPrefix(r.Message, "check run: ") {
			t.Errorf("%s: %v %v", arg, err, r)
		}
		if b.request(`D:\m.gguf`, ui.Settings{}).Smoke != want {
			t.Errorf("%s: the request does not carry it", arg)
		}
	}
	b.Action(ui.ActSetSmoke, "once")
	if smokeUI(b.smokeMode()) != ui.SmokeOnce {
		t.Error("once")
	}
	if _, err := b.Action(ui.ActSetSmoke, "twice"); err == nil {
		t.Error("a bad value")
	}
	ev := core.StartEvent{Final: true, Status: core.StatusFailSmoke, ExitCode: 6,
		Err:   &core.LaunchError{Status: core.StatusFailSmoke, Reason: "smoke assertions failed: 4:cancel"},
		Smoke: &core.SmokeResult{OK: false, Failures: []string{"4:cancel"}}}
	f := failureFrom(ev, time.Unix(1, 0), nil)
	if !f.SmokeFailed || f.ExitCode != 6 || !strings.Contains(f.Why, "4:cancel") {
		t.Errorf("smoke failure %+v", f)
	}
	if lv, line := smokeLog(ev.Smoke); lv != ui.LogError || !strings.HasPrefix(line, "check run failed") {
		t.Errorf("smoke log %v %q", lv, line)
	}
	if _, line := smokeLog(nil); line != "" {
		t.Error("no check, no line")
	}
}

// Conflict 6: one 4 Log line per server stop, in time order with the other lines.
func TestStopLineRow6(t *testing.T) {
	t0 := time.Unix(5000, 0)
	zero, k := uint32(0), uint32(1)
	evs := []core.ServerStopEvent{
		{At: t0, Class: core.StopClean, Status: core.StatusOK, Code: 0, Meaning: "stopped normally", ChildExit: &zero},
		{At: t0.Add(time.Minute), Class: core.StopForced, Status: core.StatusFailTeardown, Code: 5, Meaning: "forced stop", ChildExit: &k},
		{At: t0.Add(2 * time.Minute), Class: core.StopBeforeReady, Status: core.StatusCancelledUser, Code: 2, Meaning: "stopped before ready"},
		{At: t0.Add(3 * time.Minute), Class: core.StopFailed, Status: core.StatusFailRuntimeExit, Code: 5, Meaning: "server exited"},
		{At: t0.Add(4 * time.Minute), Class: core.StopFailed, Status: core.StatusFailSmoke, Code: 6, Meaning: "the check run failed"},
	}
	lines := stopLines(evs)
	if len(lines) != 5 || lines[0].Level != ui.LogOK || lines[1].Level != ui.LogWarn || lines[2].Level != ui.LogWarn ||
		lines[4].Level != ui.LogError {
		t.Fatalf("levels %+v", lines)
	}
	if s := lines[0].Stop; s == nil || s.Code != 0 || s.ServerCode.V != 0 || !s.ServerCode.Ok() || s.Meaning != "" {
		t.Errorf("clean %+v", s)
	}
	if s := lines[2].Stop; s.ServerCode.Ok() || s.Code != 2 {
		t.Errorf("before ready %+v", s)
	}
	if s := lines[4].Stop; s.Code != 6 || s.Meaning != "the check run failed" {
		t.Errorf("failed %+v", s)
	}
	merged := mergeLogs([]ui.LogLine{{At: t0.Add(30 * time.Second), Text: "stop gpt (mode 0)"}}, []core.ServerLine{{Time: t0.Add(10 * time.Second), Text: "srv done"}},
		0)
	merged = mergeLogs(append(merged, lines[:2]...), nil, 0)
	if merged[0].Stop == nil || merged[1].Text != "srv done" || merged[2].Text != "stop gpt (mode 0)" || merged[3].Stop == nil {
		t.Errorf("order %+v", merged)
	}
}

// Conflict 8: an out-of-memory load is its own failure reason; VRAM gets the ui's sentence.
func TestOOMRow8(t *testing.T) {
	for dev, vram := range map[string]bool{"vram": true, "ram": false, "": false} {
		ev := core.StartEvent{Final: true, Status: core.StatusFailServerStart, ExitCode: 5,
			Err: &core.LaunchError{Status: core.StatusFailServerStart, Reason: "out of memory",
				Load: &core.LoadFailure{Cause: core.CauseOutOfMemory, Device: dev, Line: "cudaMalloc failed: out of memory"}}}
		f := failureFrom(ev, time.Unix(1, 0), nil)
		if f.VRAMOut != vram || !strings.Contains(f.Why, "ran out of") {
			t.Errorf("%q: %+v", dev, f)
		}
	}
	plain := failureFrom(core.StartEvent{Final: true, Status: core.StatusFailServerStart, ExitCode: 5,
		Err: &core.LaunchError{Status: core.StatusFailServerStart, Reason: "x"}}, time.Unix(1, 0), nil)
	if plain.VRAMOut {
		t.Error("no load failure")
	}
}

// Conflicts 7, 14 and 16 as the list and the preflight give them; the setup reasons of
// conflict 9 in the user's words.
func TestListAndPreflightRows(t *testing.T) {
	mfs := []core.ModeFormat{{Mode: "packed", Result: core.GateResult{State: core.GatePass}},
		{Mode: "virtual", Result: core.GateResult{State: core.GateNeedsRepack}}}
	mc := modeChecks(mfs)
	if len(mc) != 2 || mc[0] != (ui.ModeCheck{Mode: "packed", State: ui.ModeReady}) || mc[1].State != ui.ModeNeedsRepack {
		t.Errorf("modes %+v", mc)
	}
	if mc := modeChecks([]core.ModeFormat{{Mode: "packed", Result: core.GateResult{State: core.GateFail}}}); mc[0].State != ui.ModeCheckFailed {
		t.Errorf("failed %+v", mc)
	}
	ml := &core.ModelList{Files: []core.ModelFile{
		{Path: `D:\m\t.gguf`, Name: "t.gguf", DisplayName: "t", Template: true, Arch: "qwen35moe", State: core.FileUsable},
		{Path: `D:\m\c.gguf`, Name: "c.gguf", DisplayName: "c", Catalog: true, ProfileID: "c", SetupCut: true, State: core.FileUsable},
	}}
	c := catalogFrom(ml, "models.json", 0, false, core.VRAMInfo{}, nil)
	// GL-10: a template model has its setup (the template repack); a cut-off setup opens H4.
	if !c.Models[0].Template || c.Models[1].Template || c.Models[0].Setup != ui.SetupNeedsRepack ||
		!strings.Contains(c.Models[0].SetupNote, "qwen35moe template settings") {
		t.Errorf("template %+v", c.Models[0])
	}
	if c.Models[1].Setup != ui.SetupCutOff || !strings.Contains(c.Models[1].SetupNote, "starts again from the beginning") {
		t.Errorf("cut setup %+v", c.Models[1])
	}
	res := &core.PreflightResult{Check: core.StartCheck{Kind: core.CheckBlocked, BudgetMiB: 20480, FreeMiB: 12288},
		BudgetCap: core.BudgetCapInfo{GiB: 9, OK: true}}
	p, _ := preflightFrom("m", res, &core.LaunchError{Status: core.StatusFailResource, Reason: "ram"})
	near(t, "budget cap", p.BudgetCapGB, 9)
	if p, _ := preflightFrom("m", &core.PreflightResult{BudgetCap: core.BudgetCapInfo{Reason: "none fits"}}, nil); p.BudgetCapGB.Ok() {
		t.Error("no cap without a fitting budget")
	}
	for err, want := range map[error]string{
		&core.RepackNeededError{Mode: "virtual"}:               "no virtual repack files yet · ⏎ opens the setup plan",
		&core.RepackNeededError{Mode: "packed", Partial: true}: "an earlier setup was cut off · ⏎ opens the setup plan",
		&core.RepackNeededError{Mode: "packed", Stale: true}:   "the repack files do not come from this file · ⏎ opens the setup plan",
		&core.LaunchError{Status: core.StatusFailGateBundle}:   "the launcher files are damaged · install again",
		&core.GateError{LaunchError: &core.LaunchError{Status: core.StatusFailGateVerify}, Gate: core.GateResult{Mode: "packed", FailedItem: 4, Items: 7}}: "the packed repack check failed at item 4 of 7 · ⏎ opens the setup plan",
	} {
		if got, _ := userReason(err, nil); got != want {
			t.Errorf("%v: %q, want %q", err, got, want)
		}
	}
	if !errors.Is(core.NotPorted(core.FamilyRepack), core.ErrNotPorted) {
		t.Error("not ported")
	}
}

// Live start 1: the warmup request is not a user request on the serve page or in the run summary,
// and the bundle check step reads in words.
func TestOwnRequestsDropped(t *testing.T) {
	ready := time.Unix(5000, 0)
	warm := core.RequestTiming{Task: 0, At: ready.Add(-time.Second), PromptTokens: 69, TTFT: 9668 * time.Millisecond,
		AnswerTokens: 1, AnswerTime: 0, DecodeTokPerSec: 1000000}
	tm := core.TimingSummary{Done: []core.RequestTiming{warm}, Requests: 1, AnswerTokens: 1, PromptTokens: 69, PromptTime: warm.TTFT}
	var o ownRequests
	got := o.drop(tm, ready)
	if got.Requests != 0 || len(got.Done) != 0 || got.PromptTokens != 0 || got.AnswerTokens != 0 || o.frozen {
		t.Fatalf("warmup only: %+v %+v", got, o)
	}
	s := statusFrom(core.ServeStatus{Timing: got}, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, nil, ready, false)
	if s.Requests.V != 0 || s.LastDecodeTokS.Ok() || s.TTFTs.Ok() {
		t.Fatalf("serve page after the warmup: %+v", s)
	}
	user := core.RequestTiming{Task: 1, At: ready.Add(20 * time.Second), PromptTokens: 100, TTFT: time.Second,
		AnswerTokens: 200, AnswerTime: 10 * time.Second, DecodeTokPerSec: 20}
	tm.Done = append(tm.Done, user)
	tm.Requests, tm.AnswerTokens, tm.AnswerTime = 2, 201, 10*time.Second
	tm.PromptTokens, tm.PromptTime = 169, warm.TTFT+time.Second
	got = o.drop(tm, ready)
	if got.Requests != 1 || len(got.Done) != 1 || got.AnswerTokens != 200 || got.PromptTokens != 100 || !o.frozen {
		t.Fatalf("one user request: %+v %+v", got, o)
	}
	// The warmup entry trimmed from Done later stays subtracted.
	tm.Done = tm.Done[1:]
	if got = o.drop(tm, ready); got.Requests != 1 || got.PromptTime != time.Second {
		t.Fatalf("trimmed: %+v", got)
	}
	r := runSummary("gpt", ready, ready.Add(time.Minute), got)
	if r.Requests.V != 1 || r.DecodeAvg.V != 20 || r.TTFTMax.V != 1 {
		t.Fatalf("run summary: %+v", r)
	}
	// Before ready nothing is dropped.
	var o2 ownRequests
	if g := o2.drop(tm, time.Time{}); g.Requests != 2 {
		t.Fatalf("not ready: %+v", g)
	}
	for st, want := range map[core.BundleState]string{core.BundleDev: "dev build · skipped", core.BundleOK: "12 files match"} {
		tr := newStartTrack("gpt")
		ev := tr.apply(core.StartEvent{Step: core.StepBundle, State: core.StepDone, Detail: "dev",
			Bundle: &core.BundleResult{State: st, Files: 12}}, ready)
		if ev.Steps[0].Detail != want || ev.Steps[0].Name != "bundle check" {
			t.Fatalf("bundle step %v: %+v", st, ev.Steps[0])
		}
	}
}
