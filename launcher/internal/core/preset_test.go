package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPresetRoundTrip(t *testing.T) {
	f := newFixture(t, true)
	ctx := context.Background()
	if _, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile"); st != PresetNone {
		t.Fatalf("state %v before any save", st)
	}
	// Only the changed values are kept: the catalog default port and "auto" drop out.
	p, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{
		"ctx": "8192", "port": "8093", "threads": "auto", "budget_mb": "", "qd": "8", "repack_mode": "packed"})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Values) != 2 || p.Values["ctx"] != "8192" || p.Values["qd"] != "8" {
		t.Errorf("values %v", p.Values)
	}
	path := f.l.PresetPath("fixture-profile", f.modelPath)
	if path != filepath.Join(f.stateDir, "presets", "fixture-profile.json") { // GL-16: F-LS 8.2
		t.Errorf("path %s", path)
	}
	if _, err := os.Stat(filepath.Join(f.stateDir, "presets.user.json")); err == nil {
		t.Error("the PowerShell preset file was written")
	}
	got, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile")
	if st != PresetApplies || got.Model != "fixture-profile" || got.FileName != "fixture-model.gguf" || got.FileBytes != f.modelBytes ||
		got.Values["ctx"] != "8192" || len(got.Values) != 2 {
		t.Errorf("load %+v %v", got, st)
	}
	// The preflight takes the saved values; a value the request names wins.
	res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if err != nil || res.PresetState != PresetApplies || res.Start.Ctx != 8192 || res.Start.QD != 8 {
		t.Fatalf("preflight with preset: %v %v %+v", err, res.PresetState, res.Start)
	}
	if !strings.Contains(strings.Join(res.Notices, "|"), "saved settings applied") {
		t.Errorf("notices %v", res.Notices)
	}
	res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"ctx": "16384"}})
	if err != nil || res.Start.Ctx != 16384 || res.Start.QD != 8 {
		t.Errorf("request over preset: %v %+v", err, res.Start)
	}
	res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true})
	if err != nil || res.Start.Ctx != 4096 || res.PresetState != PresetNone {
		t.Errorf("no preset: %v %+v", err, res.Start)
	}
	// Invalid values are refused at the save.
	if _, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"ctx": "1"}); err == nil {
		t.Error("ctx 1 saved")
	}
	if _, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"gpu": "1"}); err == nil {
		t.Error("unknown key saved")
	}
	if _, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"repack_mode": "zip"}); err == nil {
		t.Error("bad repack mode saved")
	}
	// The repack mode is saved when it is not the default; the request's mode wins (mergePreset
	// leaves it to Preflight, which applies it only when the request names none).
	p, err = f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"repack_mode": "virtual"})
	if err != nil || p.Values["repack_mode"] != RepackVirtual {
		t.Errorf("repack mode %v %v", p, err)
	}
	m := mergePreset(map[string]string{"ctx": "2048"}, map[string]string{"ctx": "8192", "qd": "4", "repack_mode": "virtual"})
	if len(m) != 2 || m["ctx"] != "2048" || m["qd"] != "4" {
		t.Errorf("merge %v", m)
	}
	if err := f.l.DeletePreset("fixture-profile", f.modelPath); err != nil {
		t.Fatal(err)
	}
	if err := f.l.DeletePreset("fixture-profile", f.modelPath); err != nil {
		t.Errorf("second delete: %v", err)
	}
}

func TestPresetKeyRules(t *testing.T) {
	f := newFixture(t, true)
	ctx := context.Background()
	// Hash the shard so the key holds the digest (a catalog model with a cached digest).
	if _, err := f.l.shardDigests(ctx, []string{f.modelPath}, true); err != nil {
		t.Fatal(err)
	}
	p, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"ctx": "8192"})
	if err != nil || len(p.Sha256) != 1 || len(p.Sha256[0]) != 64 {
		t.Fatalf("sha key: %+v %v", p, err)
	}
	// Same name and size, other bytes: the digest decides, the preset is not applied.
	b, _ := os.ReadFile(f.modelPath)
	b[len(b)-1] ^= 0xff
	if err := os.WriteFile(f.modelPath, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := f.l.shardDigests(ctx, []string{f.modelPath}, true); err != nil {
		t.Fatal(err)
	}
	if _, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile"); st != PresetOtherFile {
		t.Errorf("other bytes: state %v", st)
	}

	// Outside the catalog (and with no digest on either side): file name + size.
	q := &Preset{FileName: "a.gguf", FileBytes: 10}
	if !q.sameFile(nil, "A.GGUF", 10) || q.sameFile(nil, "a.gguf", 11) || q.sameFile(nil, "b.gguf", 10) {
		t.Error("name+size rule")
	}
	q.Sha256 = []string{"aa"}
	if !q.sameFile([]string{"AA"}, "other.gguf", 1) || q.sameFile([]string{"bb"}, "a.gguf", 10) {
		t.Error("digest rule")
	}
	if !q.sameFile(nil, "a.gguf", 10) {
		t.Error("digest on one side only: name+size")
	}
	// The file name outside the catalog, without the split suffix; unsafe characters replaced.
	if n := PresetName("", `D:\m\Big-Model-00001-of-00004.gguf`); n != "Big-Model" {
		t.Errorf("name %q", n)
	}
	if n := PresetName("", `D:\m\a;b?.gguf`); n != "a;b_" {
		t.Errorf("unsafe name %q", n)
	}
	// An unreadable file is not applied.
	path := f.l.PresetPath("fixture-profile", f.modelPath)
	os.WriteFile(path, []byte(`{"schema_version": 1, "values": {"gpu": "1"}}`), 0o644)
	if _, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile"); st != PresetInvalid {
		t.Errorf("unknown key state %v", st)
	}
	os.WriteFile(path, []byte(`not json`), 0o644)
	if _, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile"); st != PresetUnreadable {
		t.Errorf("not json state %v", st)
	}
	// A preset saved under another profile id is not applied.
	other := Preset{SchemaVersion: presetSchema, ProfileID: "other", FileName: "fixture-model.gguf", FileBytes: f.modelBytes, Values: map[string]string{}}
	bb, _ := json.Marshal(other)
	os.WriteFile(path, bb, 0o644)
	if _, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile"); st != PresetOtherFile {
		t.Errorf("other profile state %v", st)
	}
}
