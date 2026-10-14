package core

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Conflict 9: the plan values, the disk-shortage refusal, the run with a fake executor, the
// cancel path that leaves nothing half-made, and the duration record. No real repacker runs.

// fakeRepacker plays the repacker: it prints lines, writes what the real one writes, or waits
// for the cancel.
type fakeRepacker struct {
	t        *testing.T
	lines    []string
	write    func(outDir string)
	code     int
	block    bool // wait for ctx (the cancel test)
	started  chan struct{}
	argvSeen [][]string
}

func (f *fakeRepacker) Run(ctx context.Context, exe string, argv []string, workDir string, env []string, outLog, errLog string, line func(string)) (int, error) {
	f.argvSeen = append(f.argvSeen, argv)
	out := ""
	for i, a := range argv {
		if a == "--out" && i+1 < len(argv) {
			out = argv[i+1]
		}
	}
	os.WriteFile(errLog, nil, 0o644)
	text := strings.Join(f.lines, "\n") + "\n"
	os.WriteFile(outLog, []byte(text), 0o644)
	for _, l := range f.lines {
		if line != nil {
			line(l)
		}
	}
	if f.block {
		os.MkdirAll(out, 0o755)
		os.WriteFile(filepath.Join(out, "experts.bin.partial"), []byte{1, 2, 3}, 0o644) // a half-made output
		os.WriteFile(filepath.Join(out, "manifest.json"), []byte("{}"), 0o644)
		if f.started != nil {
			close(f.started)
		}
		<-ctx.Done()
		return -1, ctx.Err()
	}
	if f.write != nil {
		f.write(out)
	}
	return f.code, nil
}

func TestRepackPlanValuesAndShortage(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	req := Request{ModelPath: f.modelPath}
	// Usable artifacts: no plan without force.
	if p, err := f.l.PlanRepack(ctx, req, false); p != nil || err == nil || !strings.Contains(err.Error(), "no repack needed") {
		t.Fatalf("usable: %v %v", p, err)
	}
	os.Remove(filepath.Join(f.outDir, "experts.bin"))
	p, err := f.l.PlanRepack(ctx, req, false)
	if err != nil {
		t.Fatal(err)
	}
	// need = the expect's expert_bytes_total (1 GiB) in MB, free 500000, nothing to reclaim
	if !p.NeedKnown || p.NeedMiB != 1024 || p.FreeMiB != 500000 || p.ReclaimMiB != 0 || p.LeftMiB != 500000-1024 ||
		p.Refusal != nil || p.HasLastDuration || p.Mode != RepackPacked || p.OutputDir != f.outDir || p.DiskRoot != `X:\` {
		t.Errorf("plan %+v", p)
	}
	if len(p.Writes) != 3 || !strings.HasSuffix(p.Writes[0], "experts.bin") || len(p.Deletes) != 0 || !strings.Contains(p.Reason, "experts.bin missing") {
		t.Errorf("writes %v deletes %v reason %q", p.Writes, p.Deletes, p.Reason)
	}
	if !contains(p.ArgvRun, "--profile") || contains(p.ArgvRun, "--plan") || !contains(p.ArgvPlan, "--plan") ||
		!strings.HasSuffix(p.ArgvRun[len(p.ArgvRun)-1], "repack_log.jsonl") || !strings.HasPrefix(p.ArgvRun[len(p.ArgvRun)-1], f.l.LogDir()) {
		t.Errorf("argv %v / %v", p.ArgvRun, p.ArgvPlan)
	}
	// The drive too small: free < need is refused (fail_resource, exit 3), and the plan says so.
	f.l.os.volumeFree = func(string) (int64, int64, string, error) { return 500, 1000, `X:\`, nil }
	p, err = f.l.PlanRepack(ctx, req, false)
	if p == nil || p.Refusal == nil || err == nil || !strings.Contains(err.Error(), "disk preflight hard stop: need 1024 MB, free 500 MB") ||
		ExitCodeOf(err) != 3 || StatusOf(err) != StatusFailResource {
		t.Fatalf("shortage: %v %+v", err, p)
	}
	if _, err := f.l.RunRepack(ctx, p, &fakeRepacker{t: t}, nil); err == nil || !strings.Contains(err.Error(), "disk preflight hard stop") {
		t.Errorf("run of a refused plan: %v", err)
	}
	// A partial setup: the deletions reclaim their bytes (floor MB) and count for the disk.
	big := make([]byte, 3<<20+5)
	os.WriteFile(filepath.Join(f.outDir, "experts.bin.partial"), big, 0o644)
	f.l.os.volumeFree = func(string) (int64, int64, string, error) { return 1022, 1000, `X:\`, nil }
	p, err = f.l.PlanRepack(ctx, req, false)
	if err != nil || !p.Partial || p.ReclaimMiB != 3 || len(p.Deletes) != 3 || p.LeftMiB != 1022+3-1024 {
		t.Errorf("partial plan: %v %+v", err, p)
	}
	// Virtual: the reservation constant, the virtual write set.
	f.l.os.volumeFree = func(string) (int64, int64, string, error) { return 500000, 1000000, `X:\`, nil }
	p, err = f.l.PlanRepack(ctx, Request{ModelPath: f.modelPath, RepackMode: "virtual"}, false)
	if err != nil || p.NeedMiB != 128 || len(p.Writes) != 2 || !contains(p.ArgvRun, "--mode") {
		t.Errorf("virtual plan: %v %+v", err, p)
	}
}

func TestRepackRunCancelAndDuration(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	req := Request{ModelPath: f.modelPath}
	os.Remove(filepath.Join(f.outDir, "experts.bin"))
	plan, err := f.l.PlanRepack(ctx, req, false)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Cancel during the repacker: killed, everything it left removed, cancelled_user (2).
	cctx, cancel := context.WithCancel(ctx)
	fr := &fakeRepacker{t: t, block: true, started: make(chan struct{})}
	go func() { <-fr.started; cancel() }()
	_, err = f.l.RunRepack(cctx, plan, fr, nil)
	if StatusOf(err) != StatusCancelledUser || ExitCodeOf(err) != 2 || !strings.Contains(err.Error(), "next start begins from scratch") {
		t.Fatalf("cancel: %v", err)
	}
	for _, n := range PackedArtifactSet {
		if fileExists(filepath.Join(f.outDir, n)) {
			t.Errorf("%s left after the cancel", n)
		}
	}
	if _, ok := f.l.LastRepackDuration(f.modelPath, RepackPacked); ok {
		t.Error("a cancelled run must not record a duration")
	}
	// The next plan starts from scratch: nothing to delete, no partial.
	plan, err = f.l.PlanRepack(ctx, req, false)
	if err != nil || plan.Partial || len(plan.Deletes) != 0 {
		t.Fatalf("after cancel: %v %+v", err, plan)
	}

	// 2. The repacker fails: fail_repack (3).
	if _, err := f.l.RunRepack(ctx, plan, &fakeRepacker{t: t, code: 1}, nil); StatusOf(err) != StatusFailRepack || !strings.Contains(err.Error(), "exited abnormally (code=1)") {
		t.Errorf("exit 1: %v", err)
	}
	// It exits 0 without its report: fail_repack.
	if _, err := f.l.RunRepack(ctx, plan, &fakeRepacker{t: t}, nil); err == nil || !strings.Contains(err.Error(), "verify_report.json was not produced") {
		t.Errorf("no report: %v", err)
	}
	// It writes a report the gate rejects: fail_gate_verify (4) with the gate result.
	_, err = f.l.RunRepack(ctx, plan, &fakeRepacker{t: t, write: func(out string) {
		os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, out, "fixture-profile", f.expectSha)
		setReport(t, out, "pairs_pass", 1)
	}}, nil)
	var ge *GateError
	if !errors.As(err, &ge) || ge.Gate.FailedItem != 4 || ExitCodeOf(err) != 4 {
		t.Errorf("gate after repack: %v", err)
	}

	// 3. A good run: progress events in order, the gate passes, the duration is recorded.
	os.Remove(filepath.Join(f.outDir, "experts.bin"))
	plan, err = f.l.PlanRepack(ctx, req, false)
	if err != nil {
		t.Fatal(err)
	}
	clock := time.Date(2026, 10, 10, 6, 0, 0, 0, time.UTC)
	f.l.os.now = func() time.Time { clock = clock.Add(30 * time.Second); return clock }
	var evs []RepackEvent
	good := &fakeRepacker{t: t, lines: []string{"[repack] moe layer 3 (1/2) done - 1.0 MB", "[repack] moe layer 4 (2/2) done - 1.0 MB",
		"[verify] 4/8 pairs (50%)"}, write: func(out string) {
		os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, out, "fixture-profile", f.expectSha)
	}}
	res, err := f.l.RunRepack(ctx, plan, good, func(e RepackEvent) { evs = append(evs, e) })
	if err != nil || res.Gate.State != GatePass || res.Duration <= 0 {
		t.Fatalf("good run: %v %+v", err, res)
	}
	var stages []string
	for _, e := range evs {
		stages = append(stages, string(e.Stage))
	}
	// GL-10: the startup probe runs (its start line and its reading); the speed check is the
	// start's sweep step, so the run ends at the verify gate.
	want := "probe probe repack repack repack verify verify done"
	if got := strings.Join(stages, " "); got != want {
		t.Errorf("stages %q, want %q", got, want)
	}
	if evs[3].Fraction != 0.5 || evs[4].Fraction != 1 || evs[5].Fraction != 0.5 || evs[0].Skipped || evs[1].Skipped ||
		!strings.Contains(evs[1].Line, "(provisional)") {
		t.Errorf("fractions %+v", evs)
	}
	if len(res.Skipped) != 0 || !res.Probe.OK {
		t.Errorf("skipped %v probe %+v", res.Skipped, res.Probe)
	}
	d, ok := f.l.LastRepackDuration(f.modelPath, RepackPacked)
	if !ok || d != res.Duration {
		t.Errorf("duration %v %v (run %v)", d, ok, res.Duration)
	}
	if _, ok := f.l.LastRepackDuration(f.modelPath, RepackVirtual); ok {
		t.Error("the duration is per mode")
	}
	// The start now passes its gate, and a forced re-plan shows the last duration and deletes
	// the current artifacts first.
	if _, err := f.l.Preflight(ctx, req); err != nil {
		t.Errorf("preflight after the repack: %v", err)
	}
	plan, err = f.l.PlanRepack(ctx, req, true)
	if err != nil || !plan.Forced || !plan.HasLastDuration || plan.LastDuration != d || len(plan.Deletes) != 3 {
		t.Errorf("forced plan: %v %+v", err, plan)
	}
	// Running the forced plan clears first (a stage-clear event), then repacks.
	evs = nil
	if _, err := f.l.RunRepack(ctx, plan, good, func(e RepackEvent) { evs = append(evs, e) }); err != nil || evs[0].Stage != RepackStageClear {
		t.Errorf("forced run: %v %+v", err, evs)
	}
}

func TestRepackPlanText(t *testing.T) {
	f := newFixture(t, false)
	os.Remove(filepath.Join(f.outDir, "experts.bin"))
	plan, err := f.l.PlanRepack(context.Background(), Request{ModelPath: f.modelPath}, false)
	if err != nil {
		t.Fatal(err)
	}
	fr := &fakeRepacker{t: t, lines: []string{"profile=fixture-profile", "experts.bin expected size=1073807360 B (1.074 GB)"}}
	txt, err := f.l.PlanText(context.Background(), plan, fr)
	if err != nil || !strings.Contains(txt, "expected size") || plan.PlanBinBytes != 1073807360 || !contains(fr.argvSeen[0], "--plan") {
		t.Errorf("plan text: %v %q %d", err, txt, plan.PlanBinBytes)
	}
	if _, err := f.l.PlanText(context.Background(), plan, &fakeRepacker{t: t, code: 2}); StatusOf(err) != StatusFailRepack {
		t.Errorf("plan exit 2: %v", err)
	}
}
