package core

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestModelAlias(t *testing.T) {
	cases := []struct{ id, path, want string }{
		{"gpt-oss-120b-mxfp4", `D:\moe-models\gpt-oss-120b\gpt-oss-120b-mxfp4-00001-of-00003.gguf`, "gpt-oss-120b-mxfp4"},
		{"", `D:\m\gpt-oss-120b-mxfp4-00001-of-00003.gguf`, "gpt-oss-120b-mxfp4"},
		{"", `D:\m\Qwen3-30B-A3B-Q4_K_M.gguf`, "Qwen3-30B-A3B-Q4_K_M"},
		{"", `D:\m\model.GGUF`, "model"},
	}
	for _, c := range cases {
		if got := ModelAlias(c.id, c.path); got != c.want {
			t.Errorf("ModelAlias(%q, %q) = %q want %q", c.id, c.path, got, c.want)
		}
	}
}

func TestAliasInArgvNotInKey(t *testing.T) {
	f := newFixture(t, true)
	ctx := context.Background()
	res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for i, a := range res.Config.Argv {
		if a == "--alias" {
			n++
			if i+1 >= len(res.Config.Argv) || res.Config.Argv[i+1] != "fixture-profile" {
				t.Errorf("alias value %v", res.Config.Argv)
			}
		}
	}
	if n != 1 || res.Start.Alias != "fixture-profile" {
		t.Errorf("--alias count %d, start alias %q", n, res.Start.Alias)
	}
	if strings.Contains(res.SettingsKey, "--alias") || strings.Contains(res.SettingsLine, "--alias") {
		t.Errorf("alias in the settings key: %s", res.SettingsLine)
	}
	// The key is the same with or without --alias: records written before GOL-FIX1 still match.
	cfg := *res.Config
	cfg.Argv = removeArgvPair(append([]string(nil), cfg.Argv...), "--alias")
	if k, _ := settingsKey("fixture-profile", f.modelPath, res.RepackMode, &cfg); k != res.SettingsKey {
		t.Errorf("key changed without --alias")
	}
	// What the start would use (item 2).
	st := res.Start
	if st.Ctx != 4096 || st.Threads != 8 || st.Port != 8093 || st.RepackMode != RepackPacked || st.BudgetMB != res.Config.Budget.MB ||
		st.QD != res.Config.QD || st.Prefetch != res.Config.Prefetch.Echo {
		t.Errorf("start settings %+v (budget %d qd %d echo %q)", st, res.Config.Budget.MB, res.Config.QD, res.Config.Prefetch.Echo)
	}
	m := st.SettingsMap()
	if m["ctx"] != "4096" || m["repack_mode"] != RepackPacked || m["port"] != "8093" {
		t.Errorf("settings map %v", m)
	}
	// A typed ctx shows in the start settings.
	res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"ctx": "8192"}})
	if err != nil || res.Start.Ctx != 8192 {
		t.Errorf("ctx 8192: %v %+v", err, res.Start)
	}
}

func TestLastStartAndRunEnd(t *testing.T) {
	f := newFixture(t, true)
	if _, _, ok := f.l.LastStart(f.modelPath); ok {
		t.Fatal("no start yet")
	}
	st := &StartSettings{RepackMode: RepackVirtual, Ctx: 65536, Threads: 16, Port: 8093, BudgetMB: 14848, QD: 8, Prefetch: "on", PrefetchOn: true, PrefetchK: 4, PrefetchN: 2, Alias: "fixture-profile"}
	rec := RunRecord{Key: "k1", ProfileID: "fixture-profile", ModelPath: f.modelPath, ReadyMillis: 3000,
		WorkingSetBytes: 20 << 30, At: "2026-10-09T18:00:00Z", Start: st}
	if err := f.l.recordRun(rec); err != nil {
		t.Fatal(err)
	}
	got, at, ok := f.l.LastStart(strings.ToUpper(f.modelPath))
	if !ok || got != *st || at.Format(time.RFC3339) != "2026-10-09T18:00:00Z" {
		t.Errorf("last start %+v %v %v", got, at, ok)
	}
	// Before the run ends: need = ready + budget.
	r, _ := f.l.HistoryFor("k1")
	if need, basis := NeedOf(&r); basis != NeedReadyBudget || need != 20480+14848 {
		t.Errorf("need %d %s", need, basis)
	}
	// The run ends with a peak and one finished request.
	sum := SummarizeTimings(ParseServerLines(fixtureLines(t, "server_20261009T175949Z_err.log"), fixtureSpawn))
	end := time.Date(2026, 10, 9, 18, 23, 6, 0, time.UTC)
	if err := f.l.recordRunEnd("k1", f.modelPath, 31<<30, sum, end); err != nil {
		t.Fatal(err)
	}
	r, _ = f.l.HistoryFor("k1")
	if need, basis := NeedOf(&r); basis != NeedPeak || need != 31744 || r.Runs != 1 {
		t.Errorf("need after the run %d %s runs %d", need, basis, r.Runs)
	}
	ms, ok := f.l.ModelStatsFor(strings.ToUpper(f.modelPath))
	if !ok || math.Abs(ms.DecodeTokPerSec-3000/338.47296) > 1e-9 || ms.AnswerTokens != 3000 || ms.Requests != 1 ||
		ms.At != "2026-10-09T18:23:06Z" || ms.PeakWorkingSet != 31<<30 {
		t.Errorf("model stats %+v %v", ms, ok)
	}
	// A run without a request keeps the previous stats; the history stays schema 1.
	if err := f.l.recordRunEnd("k1", f.modelPath, 0, TimingSummary{}, end.Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if ms2, _ := f.l.ModelStatsFor(f.modelPath); ms2 != ms {
		t.Errorf("stats replaced by an empty run: %+v", ms2)
	}
	if r2, _ := f.l.HistoryFor("k1"); r2.MaxWorkingSetBytes != 31<<30 {
		t.Errorf("peak lost: %d", r2.MaxWorkingSetBytes)
	}
	b, _ := os.ReadFile(f.l.historyPath())
	if !strings.Contains(string(b), `"schema_version": 1`) || !strings.Contains(string(b), `"models"`) {
		t.Errorf("history %s", b)
	}
}

func TestNeedOf(t *testing.T) {
	cases := []struct {
		rec   *RunRecord
		need  int64
		basis NeedBasis
	}{
		{nil, 0, NeedNotMeasured},
		{&RunRecord{}, 0, NeedNotMeasured},
		{&RunRecord{WorkingSetBytes: 10 << 30}, 10240, NeedReadyBudget},
		{&RunRecord{WorkingSetBytes: 10<<30 + 1, Start: &StartSettings{BudgetMB: 4096}}, 10241 + 4096, NeedReadyBudget},
		{&RunRecord{WorkingSetBytes: 10 << 30, MaxWorkingSetBytes: 12 << 30, Start: &StartSettings{BudgetMB: 4096}}, 12288, NeedPeak},
	}
	for i, c := range cases {
		if need, basis := NeedOf(c.rec); need != c.need || basis != c.basis {
			t.Errorf("case %d: %d %s want %d %s", i, need, basis, c.need, c.basis)
		}
	}
}

func TestEvaluateStart(t *testing.T) {
	c := EvaluateStart(8000, 0, NeedNotMeasured, 9000)
	if c.Kind != CheckBlocked || c.Reason != "budget" || c.BudgetMiB != 9000 || c.FreeMiB != 8000 {
		t.Errorf("blocked %+v", c)
	}
	c = EvaluateStart(8000, 0, NeedNotMeasured, 4096)
	if c.Kind != CheckUnmeasured {
		t.Errorf("unmeasured %+v", c)
	}
	c = EvaluateStart(40000, 30000, NeedPeak, 4096)
	if c.Kind != CheckRoom || c.LeftMiB != 10000 || c.SpareGiB != 9 || c.SafeLineKnown {
		t.Errorf("room %+v", c)
	}
	c = EvaluateStart(40000, 39000, NeedPeak, 4096)
	if c.Kind != CheckRoom || c.SpareGiB != 0 || c.SpareGiB >= SpareComfortGiB {
		t.Errorf("room, little spare %+v", c)
	}
	c = EvaluateStart(20000, 21000, NeedReadyBudget, 4096)
	if c.Kind != CheckTight || c.LeftMiB != -1000 || c.SpareGiB != -1 {
		t.Errorf("tight %+v", c)
	}
	if CheckTight.String() != "tight" || CheckBlocked.String() != "blocked" {
		t.Error("kind names")
	}
}

func TestPCMemoryNow(t *testing.T) {
	f := newFixture(t, true)
	m, err := f.l.PCMemoryNow()
	if err != nil || m.TotalMiB != 131072 || m.AvailMiB != 65536 || m.UsedMiB != 65536 || m.ServerMiB != 0 ||
		m.OthersMiB != 65536 || m.FreeIfStoppedMiB() != 65536 {
		t.Errorf("%+v %v", m, err)
	}
	// Measured at every call.
	f.mem.AvailPhysMiB = 1000
	if m, _ := f.l.PCMemoryNow(); m.AvailMiB != 1000 || m.UsedMiB != 130072 {
		t.Errorf("second call %+v", m)
	}
	if _, err := f.l.ServerVRAMNow(); err != ErrNoServer {
		t.Errorf("no server: %v", err)
	}
}

func TestGPUShareOf(t *testing.T) {
	inst := map[string]int64{
		"pid_10172_luid_0x00000000_0x00012568_phys_0":  6 << 30,
		"pid_10172_luid_0x00000000_0x00012568_phys_1":  1 << 20,
		"pid_101720_luid_0x00000000_0x00012568_phys_0": 5 << 30,
		"pid_4_luid_0x00000000_0x00012568_phys_0":      7,
	}
	if b, ok := gpuShareOf(inst, 10172); !ok || b != 6<<30+1<<20 {
		t.Errorf("pid 10172: %d %v", b, ok)
	}
	if _, ok := gpuShareOf(inst, 999); ok {
		t.Error("pid 999 found")
	}
}

func TestParseNvidiaSmiMemory(t *testing.T) {
	v := parseNvidiaSmiMemory("16303, 15028, 952")
	if !v.OK || v.TotalMiB != 16303 || v.FreeMiB != 15028 || v.UsedMiB != 952 {
		t.Errorf("%+v", v)
	}
	if v := parseNvidiaSmiMemory("16303, 15028"); !v.OK || v.UsedMiB != -1 {
		t.Errorf("two fields %+v", v)
	}
	if v := parseNvidiaSmiMemory("[N/A], 1"); v.OK {
		t.Errorf("garbage %+v", v)
	}
}

func TestShellActions(t *testing.T) {
	f := newFixture(t, true)
	var exe, line, clip string
	f.l.os.launch = func(e, c string) error { exe, line = e, c; return nil }
	f.l.os.clipboard = func(s string) error { clip = s; return nil }
	dir := filepath.Dir(f.modelPath)
	if err := f.l.OpenFolder(dir); err != nil || line != `explorer.exe "`+dir+`"` || !strings.EqualFold(filepath.Base(exe), "explorer.exe") {
		t.Errorf("open folder: %v %q %q", err, exe, line)
	}
	if err := f.l.RevealFile(f.modelPath); err != nil || line != `explorer.exe /select,"`+f.modelPath+`"` {
		t.Errorf("reveal: %v %q", err, line)
	}
	line = ""
	if err := f.l.OpenFolder(f.modelPath); err == nil || line != "" {
		t.Errorf("a file opened as a folder: %v %q", err, line)
	}
	if err := f.l.RevealFile(filepath.Join(dir, "missing.gguf")); err == nil || line != "" {
		t.Errorf("a missing file revealed: %v %q", err, line)
	}
	if err := f.l.CopyText("http://127.0.0.1:8093"); err != nil || clip != "http://127.0.0.1:8093" {
		t.Errorf("copy: %v %q", err, clip)
	}
	if err := f.l.CopyText("  "); err == nil {
		t.Error("empty copy accepted")
	}
}
