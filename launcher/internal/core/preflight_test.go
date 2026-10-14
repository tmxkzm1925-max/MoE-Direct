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

func TestPreflightVerdicts(t *testing.T) {
	f := newFixture(t, false)
	notValidated := false // GL-05: this test covers the not-validated display (S8-LA-13)
	f.l.opt.ProjectionValidation = &notValidated
	ctx := context.Background()
	req := Request{ModelPath: f.modelPath}

	// 1. No measurement for this model and settings: «미측정», no refusal (conflict 7: the
	//    PowerShell verdict stays [unmeasured] while the budget fits).
	res, err := f.l.Preflight(ctx, req)
	if err != nil {
		t.Fatalf("preflight: %v (steps %v)", err, res.Steps)
	}
	if res.Selection != SelectionUnpinned || res.Profile.ID != "fixture-profile" {
		t.Errorf("selection %s", res.Selection)
	}
	if res.Verdict != MemUnmeasured || res.NeedKnown || res.Verdict.String() != "not measured" {
		t.Errorf("first run must be unmeasured: %+v", res.Verdict)
	}
	if res.RAM.Verdict != "unmeasured" || res.FreeMiB != 65536 {
		t.Errorf("ram %+v", res.RAM)
	}
	// budget: installed 65536 -> machine_cap 12288; model_cap = 4*2*1 GiB = 8192 MiB -> 8192 (auto)
	if res.Config.Budget.MB != 8192 || res.Config.Budget.Source != "auto" {
		t.Errorf("budget %+v", res.Config.Budget)
	}
	// no sweep binding on this fixture: the PowerShell skip value, QD 1 degraded
	if res.Sweep.OK || res.Config.QD != QDDegraded || res.QDSource != "conservative-default" {
		t.Errorf("sweep %+v", res.Sweep)
	}
	if !strings.Contains(res.Sweep.Detail, "the speed check (the QD sweep) runs at the start") { // GL-10: a preflight never sweeps
		t.Errorf("sweep detail %q", res.Sweep.Detail)
	}
	if res.HasLastReady {
		t.Error("no last duration on a first run")
	}
	if res.Config.Env[EnvDirectDir] != f.outDir {
		t.Errorf("out dir %s", res.Config.Env[EnvDirectDir])
	}

	// 2. A measurement for the same key: need, free now, left = free - need; fits.
	if err := f.l.recordRun(RunRecord{Key: res.SettingsKey, ProfileID: "fixture-profile", ModelPath: f.modelPath,
		ReadyMillis: 42000, WorkingSetBytes: 10 << 30, At: "2026-10-10T00:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	res, err = f.l.Preflight(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	// GOL-FIX1 item 13: no peak recorded yet, so need = the ready value + the budget.
	bud := res.Config.Budget.MB
	if !res.NeedKnown || res.NeedBasis != NeedReadyBudget || res.NeedMiB != 10240+bud || res.LeftMiB != 65536-10240-bud ||
		res.Verdict != MemFits || res.Check.Kind != CheckRoom || res.Check.SpareGiB != (65536-10240-bud)/1024 {
		t.Errorf("measured: need %d (%s) left %d verdict %v check %+v", res.NeedMiB, res.NeedBasis, res.LeftMiB, res.Verdict, res.Check)
	}
	if !res.HasLastReady || res.LastReady.Seconds() != 42 {
		t.Errorf("last ready %v", res.LastReady)
	}
	// a different setting is a different key: unmeasured again
	res2, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"ctx": "8192"}})
	if err != nil || res2.NeedKnown || res2.SettingsKey == res.SettingsKey {
		t.Errorf("ctx change keeps the old need: %v %v", res2.NeedKnown, err)
	}
	// the port does not change the key
	res3, _ := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"port": "8111"}})
	if res3 == nil || res3.SettingsKey != res.SettingsKey || !res3.NeedKnown {
		t.Errorf("port changed the settings key")
	}

	// 3. Need above free RAM: shown as short, not refused (PowerShell refuses on the budget only).
	f.mem.AvailPhysMiB = 9000
	res, err = f.l.Preflight(ctx, req)
	if err != nil || res.Verdict != MemShort || res.LeftMiB != 9000-10240-bud || res.Check.Kind != CheckTight {
		t.Errorf("short: %v %v %d %+v", err, res.Verdict, res.LeftMiB, res.Check)
	}
	// GOL-FIX1 item 13: a recorded peak is the need.
	if err := f.l.recordRunEnd(res.SettingsKey, f.modelPath, 12<<30, TimingSummary{}, time.Now()); err != nil {
		t.Fatal(err)
	}
	f.mem.AvailPhysMiB = 65536
	res, err = f.l.Preflight(ctx, req)
	if err != nil || res.NeedBasis != NeedPeak || res.NeedMiB != 12288 || res.FreeMiB != 65536 || res.LeftMiB != 65536-12288 {
		t.Errorf("peak: %v %s %d %d %d", err, res.NeedBasis, res.NeedMiB, res.FreeMiB, res.LeftMiB)
	}
	f.mem.AvailPhysMiB = 9000

	// 4. Budget above available RAM: the hard stop (conflict 7, Test-RamVerdict budget_only).
	f.mem.AvailPhysMiB = 8000
	res, err = f.l.Preflight(ctx, req)
	wantStatus(t, err, StatusFailResource, "RAM preflight hard stop: selected budget alone exceeds available RAM")
	if res.RAM.Verdict != "insufficient" || res.RAM.Basis != "budget_only" || res.Refusal == nil {
		t.Errorf("insufficient: %+v", res.RAM)
	}
	f.mem.AvailPhysMiB = 65536

	// 5. VRAM is display only (conflict 8): a full GPU changes nothing.
	f.l.os.vram = func() VRAMInfo { return VRAMInfo{OK: true, Source: "fixture", TotalMiB: 8192, FreeMiB: 0} }
	if _, err := f.l.Preflight(ctx, req); err != nil {
		t.Errorf("vram gated: %v", err)
	}

	// 6. The allowlist (conflict 2) and the not-ported keys.
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"mlock": "1"}})
	wantStatus(t, err, StatusFailCustomArgs, "not a changeable setting")
	// Conflict 2: warmup is open now; warm start stays locked.
	if res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"warmup": "off"}}); err != nil || res.Warmup != "off" {
		t.Errorf("warmup: %v", err)
	}
	if res, err := f.l.Preflight(ctx, req); err != nil || res.Warmup != WarmupDefault {
		t.Errorf("warmup default: %v", err)
	}
	// GL-16: warm start is a 5 Settings switch (Request.Switches), not a model setting.
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"warmstart": "on"}})
	wantStatus(t, err, StatusFailCustomArgs, "'warmstart' is not a changeable setting")
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"prefetch_n": "2"}})
	wantStatus(t, err, StatusFailCustomArgs, "K and N are set together or not at all") // GL-19 (F-PF §9 C3)
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: "zip"})
	wantStatus(t, err, StatusFailCustomArgs, "expected packed or virtual")
	_, err = f.l.Preflight(ctx, Request{ModelPath: filepath.Join(f.dir, "nope.gguf")})
	wantStatus(t, err, StatusFailModelPath, "model file not found")
}

func TestPreflightArtifactsAndVirtual(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	// A virtual run with no plan: the setup is needed (conflict 9).
	_, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: "virtual"})
	var rn *RepackNeededError
	if !errors.As(err, &rn) || rn.Mode != RepackVirtual || ExitCodeOf(err) != 3 {
		t.Fatalf("virtual without plan: %v", err)
	}
	// With a plan: QD pinned to 8, and a QD outside the pair is accepted with the notice.
	vdir := filepath.Join(filepath.Dir(f.modelPath), "repack-virtual")
	writeVirtualArtifacts(t, vdir, "fixture-profile", f.expectSha)
	res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: "virtual"})
	if err != nil || res.Config.QD != VirtualPinnedQD || res.QDSource != "virtual-pinned" || len(res.Notices) != 0 {
		t.Fatalf("virtual pinned: %v %+v", err, res)
	}
	res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: "virtual", Settings: map[string]string{"qd": "4"}})
	if err != nil || res.Config.QD != 4 || !contains(res.Notices, NoticeVirtualUnvalidated) || res.QDSource != "user-override" {
		t.Fatalf("virtual exception: %v %+v", err, res)
	}
	if res.Config.Env[EnvDirectDir] != vdir {
		t.Errorf("virtual out dir %s", res.Config.Env[EnvDirectDir])
	}
	// Packed artifacts missing: the setup is needed; a partial marker is an interrupted setup.
	os.Remove(filepath.Join(f.outDir, "experts.bin"))
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if !errors.As(err, &rn) || rn.Partial || !strings.Contains(rn.Reason, "experts.bin missing") {
		t.Errorf("missing experts.bin: %v", err)
	}
	os.WriteFile(filepath.Join(f.outDir, "experts.bin.partial"), nil, 0o644)
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if !errors.As(err, &rn) || !rn.Partial {
		t.Errorf("partial marker: %v", err)
	}
}

func TestPreflightSweepBinding(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	ms, _ := fileSha256(filepath.Join(f.outDir, "manifest.json"))
	key := sweepTargetKey("fixture-tag", "fixture-profile", f.expectSha, ms, `\\?\volume{fixture}\`)
	entry := map[string]any{"probe_algorithm": sweepAlgorithm, "measurement_method": sweepMethod,
		"source_tag": "fixture-tag", "profile_id": "fixture-profile", "expectation_digest": f.expectSha,
		"manifest_sha256": ms, "output_volume_guid": `\\?\volume{fixture}\`, "selected_qd": 4,
		"selected_reason": "io-knee", "block_bytes": 1 << 20, "window_duration_ms": 1250,
		"swept_at": "2026-10-07T02:00:00.0000000Z", "repack_checked_at": "2026-10-07T01:51:39.264165+00:00",
		"points": []any{
			map[string]any{"qd": 1, "status": "ok", "bytes": 100, "elapsed_ticks": 10},
			map[string]any{"qd": 2, "status": "ok", "bytes": 150, "elapsed_ticks": 10},
			map[string]any{"qd": 4, "status": "ok", "bytes": 199, "elapsed_ticks": 10},
			map[string]any{"qd": 8, "status": "ok", "bytes": 200, "elapsed_ticks": 10}}}
	state := filepath.Join(f.stateDir, "probe.state.json")
	writeJSON(t, state, map[string]any{"state_version": 2, "entries": map[string]any{key: entry}})
	res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if err != nil || !res.Sweep.OK || res.Config.QD != 4 || res.QDSource != "measured-sweep" || !res.Sweep.FromBinding {
		t.Fatalf("fresh binding: %v %+v", err, res.Sweep)
	}
	// A binding bound to another verify_report.checked_at is stale: back to QD 1.
	entry["repack_checked_at"] = "2026-10-01T00:00:00+00:00"
	writeJSON(t, state, map[string]any{"state_version": 2, "entries": map[string]any{key: entry}})
	res, _ = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if res.Sweep.OK || res.Config.QD != 1 || !strings.Contains(res.Sweep.Detail, "different verify_report.checked_at") {
		t.Errorf("stale binding: %+v", res.Sweep)
	}
}

func TestPreflightPinAndTemplate(t *testing.T) {
	f := newFixture(t, true) // the profile pins the fixture file's digest
	ctx := context.Background()
	res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if err != nil || res.Selection != SelectionPinned {
		t.Fatalf("pinned: %v %v", err, res.Selection)
	}
	// the miss was hashed once and kept in the launcher's own cache, never in the PowerShell cache
	if _, err := os.Stat(filepath.Join(f.stateDir, ownShardCacheFile)); err != nil {
		t.Errorf("draft cache: %v", err)
	}
	if _, err := os.Stat(filepath.Join(f.stateDir, shardCacheFile)); err == nil {
		t.Error("the PowerShell cache was written")
	}
	// Different bytes: mismatch (testarch is no template family) -> the stale-artifact family.
	b, _ := os.ReadFile(f.modelPath)
	b[len(b)-1] = 1
	os.WriteFile(f.modelPath, b, 0o644)
	later := time.Now().Add(time.Hour)
	os.Chtimes(f.modelPath, later, later) // a new identity key, so the cached digest is not reused
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	var rn *RepackNeededError
	if !errors.As(err, &rn) || !rn.Stale {
		t.Errorf("mismatch with artifacts: %v", err)
	}
}
