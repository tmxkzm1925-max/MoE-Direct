package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Decision ③: fill = elapsed / last measured; none on a first run (seconds only); overrun
// holds at 99 %, never a false 100 %.
func TestLoadProgressRule(t *testing.T) {
	p := LoadProgress(12*time.Second, 0, false, false)
	if !p.SecondsOnly || p.Fraction != 0 || p.HasLast {
		t.Errorf("first run: %+v", p)
	}
	p = LoadProgress(30*time.Second, 60*time.Second, true, false)
	if p.SecondsOnly || p.Fraction != 0.5 || p.Held {
		t.Errorf("half: %+v", p)
	}
	p = LoadProgress(59700*time.Millisecond, 60*time.Second, true, false)
	if p.Fraction != 0.99 || p.Held {
		t.Errorf("99.5 %% shows 99 %% without the overrun flag: %+v", p)
	}
	p = LoadProgress(75*time.Second, 60*time.Second, true, false)
	if p.Fraction != 0.99 || !p.Held {
		t.Errorf("overrun: %+v", p)
	}
	p = LoadProgress(75*time.Second, 60*time.Second, true, true)
	if p.Fraction != 1 || p.Held {
		t.Errorf("ready: %+v", p)
	}
	p = LoadProgress(75*time.Second, 0, false, true)
	if p.Fraction != 1 || !p.SecondsOnly {
		t.Errorf("ready on a first run: %+v", p)
	}
}

func TestHistoryRecordAndKey(t *testing.T) {
	f := newFixture(t, false)
	cfg := &Config{Argv: []string{"-c", "4096", "-t", "8", "--port", "8093", "-m", `D:\a.gguf`},
		Env:    map[string]string{EnvBudgetMB: "8192", EnvQD: "4", EnvNoPrefetch: "1", EnvMetrics: `C:\x.jsonl`},
		Budget: Budget{MB: 8192}, QD: 4, Prefetch: Prefetch{Echo: "off(reason=x)"}}
	k1, line := settingsKey("p", `D:\A.gguf`, RepackPacked, cfg)
	cfg2 := *cfg
	cfg2.Argv = []string{"-c", "4096", "-t", "8", "--port", "9000", "-m", `D:\a.gguf`}
	cfg2.Env = map[string]string{EnvBudgetMB: "8192", EnvQD: "4", EnvNoPrefetch: "1", EnvMetrics: `C:\other.jsonl`}
	k2, _ := settingsKey("p", `D:\a.gguf`, RepackPacked, &cfg2)
	if k1 != k2 {
		t.Errorf("port, metrics path and path case must not change the key:\n%s\n%s", k1, k2)
	}
	cfg2.Env[EnvQD] = "8"
	if k3, _ := settingsKey("p", `D:\a.gguf`, RepackPacked, &cfg2); k3 == k1 {
		t.Error("QD must change the key")
	}
	if k4, _ := settingsKey("p", `D:\a.gguf`, RepackVirtual, cfg); k4 == k1 {
		t.Error("the repack mode must change the key")
	}
	if !strings.Contains(line, "budget 8192 MB") || !strings.Contains(line, "QD 4") {
		t.Errorf("line %q", line)
	}
	for i := 0; i < 2; i++ {
		if err := f.l.recordRun(RunRecord{Key: k1, ModelPath: `D:\a.gguf`, ReadyMillis: int64(1000 * (i + 1)), WorkingSetBytes: 5, At: "2026-10-10T00:00:0" + string(rune('0'+i)) + "Z"}); err != nil {
			t.Fatal(err)
		}
	}
	r, ok := f.l.HistoryFor(k1)
	if !ok || r.Runs != 2 || r.ReadyDuration() != 2*time.Second {
		t.Errorf("record %+v", r)
	}
	if r, ok := f.l.LatestForModel(`d:\A.GGUF`); !ok || r.ReadyMillis != 2000 {
		t.Errorf("latest %+v", r)
	}
	// The file is the launcher's own, directly under the state root (F-LS 8.3; GL-FX2).
	b, err := os.ReadFile(filepath.Join(f.stateDir, "history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc historyDoc
	if json.Unmarshal(b, &doc) != nil || doc.SchemaVersion != 1 || len(doc.Records) != 1 {
		t.Errorf("history file %s", b)
	}
}

func TestListModelsAndSettings(t *testing.T) {
	f := newFixture(t, true)
	ctx := context.Background()
	f.l.os.modelRoots = func() []string { return []string{filepath.Join(f.dir, "models")} }
	ml, err := f.l.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ml.Candidates) != 1 || ml.Candidates[0].Label != LabelCatalog || ml.Candidates[0].Quant != "Q4_K_M" ||
		ml.Candidates[0].Source != "scan" || !ml.Candidates[0].Repacked {
		t.Fatalf("candidates %+v", ml.Candidates)
	}
	m := ml.Models[0]
	if !m.Present || m.Path != f.modelPath || m.SizeBytes != f.modelBytes || m.Quant != "Q4_K_M" {
		t.Fatalf("model %+v", m)
	}
	// Nothing is hashed at list time: the pinned profile's digest is not in any cache yet.
	if m.Identity != "not hashed" || m.Sha256[0] != "" {
		t.Errorf("identity %q %v", m.Identity, m.Sha256)
	}
	if _, err := os.Stat(filepath.Join(f.stateDir, ownShardCacheFile)); err == nil {
		t.Error("ListModels hashed a file")
	}
	if ml.FreeRAMMiB != 65536 {
		t.Errorf("free %d", ml.FreeRAMMiB)
	}
	// After a preflight hashed the shard, the list shows the digest from the cache.
	if _, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath}); err != nil {
		t.Fatal(err)
	}
	ml, _ = f.l.ListModels(ctx)
	if ml.Models[0].Identity != SelectionPinned || len(ml.Models[0].Sha256[0]) != 64 {
		t.Errorf("after hashing: %q", ml.Models[0].Identity)
	}
	// The recent list of the PowerShell launcher is read (and only read).
	recent := filepath.Join(f.stateDir, "recent_models.json")
	writeJSON(t, recent, map[string]any{"schema_version": 1, "paths": []string{f.modelPath, `Z:\gone.gguf`}})
	before, _ := os.ReadFile(recent)
	ml, _ = f.l.ListModels(ctx)
	if len(ml.Candidates) != 1 || ml.Candidates[0].Source != "recent" {
		t.Errorf("recent first, no duplicate from the scan: %+v", ml.Candidates)
	}
	if after, _ := os.ReadFile(recent); string(after) != string(before) {
		t.Error("recent_models.json was written")
	}

	rows, err := f.l.Settings("fixture-profile")
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]SettingRow{}
	for _, r := range rows {
		got[r.Key] = r
	}
	if r := got["port"]; !r.Editable || r.Default != "8093" || r.Min != 1024 || r.Max != 65535 {
		t.Errorf("port row %+v", r)
	}
	if r, ok := got["warmstart"]; ok { // GL-16: a 5 Settings switch, no model row (OQ-6 (a))
		t.Errorf("warmstart row %+v", r)
	}
	if r := got["prefetch"]; strings.Join(r.Options, "|") != "catalog|init" { // GL-19: no adapt (F-PF §2.2)
		t.Errorf("prefetch row %+v", r)
	}
	if len(rows) != len(SettingKeys)+1 {
		t.Errorf("%d rows", len(rows))
	}
}
