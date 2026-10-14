package wire

import (
	"context"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// GL-10 (plan cell GL-10): the adapter's half of the setup flow. core's halves are the core
// tests of the same names; the screen's are the ui tests of the same names.

// TestS8LA24 (S8-LA-24; rules 9 and 18; conflict row 9; F-LS 4.6): the plan box takes its numbers
// from the repacker's --plan during RepackPlan; no run starts before Repack (the confirm
// action); a --plan disk shortfall comes back as the box's refusal and runs nothing.
func TestS8LA24(t *testing.T) {
	f := core.ModelFile{Path: `D:\m\gpt.gguf`, Name: "gpt.gguf", DisplayName: "gpt", ProfileID: "gpt-profile", Catalog: true}
	b := testBackend(f)
	var log []string
	b.planFn = func(context.Context, core.Request, bool) (*core.RepackPlan, error) {
		log = append(log, "plan")
		return &core.RepackPlan{Mode: "packed", OutputDir: `D:\m\repack`, NeedKnown: true, NeedMiB: 30000, FreeMiB: 102400, LeftMiB: 72400}, nil
	}
	b.planTextFn = func(_ context.Context, p *core.RepackPlan, _ core.RepackExecutor) (string, error) {
		log = append(log, "--plan")
		p.NeedMiB, p.LeftMiB = 61440, p.FreeMiB-61440 // the --plan's experts.bin size
		return "experts.bin expected size=64424509440 B", nil
	}
	b.repackFn = func(_ context.Context, _ *core.RepackPlan, _ core.RepackExecutor, progress func(core.RepackEvent)) (core.RepackResult, error) {
		log = append(log, "run")
		progress(core.RepackEvent{Stage: core.RepackStageDone, Line: "setup done", Fraction: 1})
		return core.RepackResult{Duration: time.Minute}, nil
	}
	b.startFn = func(context.Context, core.Request) (<-chan core.StartEvent, error) {
		log = append(log, "start")
		c := make(chan core.StartEvent, 1)
		c <- core.StartEvent{Final: true, Ready: true, Address: "http://127.0.0.1:8093", PID: 7}
		close(c)
		return c, nil
	}
	u, err := b.RepackPlan(f.Path, ui.Settings{})
	if err != nil || strings.Join(log, " ") != "plan --plan" {
		t.Fatalf("plan box: %v %v", err, log)
	}
	near(t, "write (from --plan)", u.WriteGB, 60)
	near(t, "left", u.LeftGB, 40)
	if u.LastS.Ok() || u.WritesTo != `D:\m\repack` {
		t.Errorf("plan box %+v", u)
	}
	ch, err := b.Repack(f.Path, ui.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	if got := strings.Join(log, " "); got != "plan --plan run start" {
		t.Errorf("order %q", got)
	}

	// the --plan figure above the free space: the box comes back refused, nothing runs
	log = nil
	b.planTextFn = func(_ context.Context, p *core.RepackPlan, _ core.RepackExecutor) (string, error) {
		log = append(log, "--plan")
		p.NeedMiB, p.LeftMiB = 204800, p.FreeMiB-204800
		p.Refusal = &core.LaunchError{Status: core.StatusFailResource, Reason: "disk preflight hard stop: need 204800 MB, free 102400 MB"}
		return "", p.Refusal
	}
	u, err = b.RepackPlan(f.Path, ui.Settings{})
	if err != nil || u.LeftGB.V >= 0 || u.WriteGB.V <= u.DriveFreeGB.V {
		t.Errorf("shortfall box: %v %+v", err, u)
	}
	if _, err := b.Repack(f.Path, ui.Settings{}); err == nil || strings.Contains(strings.Join(log, " "), "run") {
		t.Errorf("a refused plan ran: %v %v", err, log)
	}
	if lines, _ := logLines(b); !strings.Contains(lines, "setup plan gpt refused: ") {
		t.Errorf("log %s", lines)
	}
}

// TestS8LA34 (S8-LA-34; LFS Q5.11, Q5.16; H4; W7 duty 12): the cut-off setup's row and figures,
// its «delete half-made only» action (no repack), the stale verdict in both modes' line, and the
// template model's setup row (addendum 1).
func TestS8LA34(t *testing.T) {
	at := time.Date(2026, 10, 12, 9, 30, 0, 0, time.Local)
	cut := core.ModelFile{Path: `D:\m\gpt.gguf`, Name: "gpt.gguf", DisplayName: "gpt", ProfileID: "gpt-profile", Catalog: true,
		SetupCut: true, PartialBytes: 3 * 1073741824, PartialAt: at, LastRepack: 25 * time.Minute}
	if st, line := setupOf(cut, "packed"); st != ui.SetupCutOff || !strings.Contains(line, "cut off") {
		t.Errorf("cut-off row %v %q", st, line)
	}
	pa := partialFrom(cut)
	if pa.FilesGB.V != 3 || pa.RedoMin.V != 25 || pa.When != "2026-10-12 09:30" {
		t.Errorf("cut-off figures %+v", pa)
	}
	b := testBackend(cut)
	var rm []string
	b.partialRmFn = func(_ context.Context, model, profile string) ([]string, error) {
		rm = append(rm, model+"|"+profile)
		return []string{"experts.bin.partial", "experts.bin", "manifest.json", "verify_report.json"}, nil
	}
	ran := false
	b.repackFn = func(context.Context, *core.RepackPlan, core.RepackExecutor, func(core.RepackEvent)) (core.RepackResult, error) {
		ran = true
		return core.RepackResult{}, nil
	}
	res, err := b.Action(ui.ActDeletePartial, cut.Path)
	if err != nil || len(rm) != 1 || rm[0] != `D:\m\gpt.gguf|gpt-profile` || ran || !strings.Contains(res.Message, "half-made files removed") {
		t.Errorf("delete half-made: %v %v %+v ran=%v", err, rm, res, ran)
	}
	if lines, _ := logLines(b); !strings.Contains(lines, "half-made setup of gpt removed (4 file(s)); nothing was repacked") {
		t.Errorf("log %s", lines)
	}

	// W7 duty 12: an output made from other model files reads stale in its mode's verdict
	got := modeChecks([]core.ModeFormat{
		{Mode: "packed", Result: core.GateResult{State: core.GatePass}, Stale: true},
		{Mode: "virtual", Result: core.GateResult{State: core.GateNeedsRepack}}})
	if len(got) != 2 || got[0].State != ui.ModeStale || got[1].State != ui.ModeNeedsRepack {
		t.Errorf("mode checks %+v", got)
	}

	// addendum 1: a template model's row offers the template setup
	tpl := core.ModelFile{Path: `D:\m\tpl.gguf`, Name: "tpl.gguf", Template: true, Arch: "gpt-oss"}
	if st, line := setupOf(tpl, "packed"); st != ui.SetupNeedsRepack || !strings.Contains(line, "the template repack") {
		t.Errorf("template row %v %q", st, line)
	}
	tpl.Repacked = true
	if st, _ := setupOf(tpl, "packed"); st != ui.SetupReady {
		t.Errorf("set-up template row %v", st)
	}
	if d := planDoes(&core.RepackPlan{Template: true, Mode: "packed"}); !strings.Contains(d, "structure template selects, byte for byte") {
		t.Errorf("template plan text %q", d)
	}
}
