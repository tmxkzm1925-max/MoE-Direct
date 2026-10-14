package core

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeModel(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	settleFile(t, path)
}

// settleFile dates a fixture model's last write an hour back: a finished download, which the
// list shows at once (GL-14's watch holds back a file written less than GrowSettle ago).
func settleFile(t *testing.T, path string) {
	t.Helper()
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}
}

func TestModelFilesMarkAndStates(t *testing.T) {
	f := newFixture(t, true)
	ctx := context.Background()
	root := filepath.Join(f.dir, "models")
	f.l.os.modelRoots = func() []string { return []string{root} }
	gptoss := ggufBytes(10, modelKVs("gpt-oss", 36, 128, 4, 15), 64)
	writeModel(t, filepath.Join(root, "tpl", "gpt-oss-20b.gguf"), gptoss)
	writeModel(t, filepath.Join(root, "llama", "llama-moe.gguf"), ggufBytes(10, modelKVs("llama", 2, 8, 2, 15), 64))
	writeModel(t, filepath.Join(root, "split", "big-00001-of-00002.gguf"), gptoss)
	writeModel(t, filepath.Join(root, "junk", "junk.gguf"), []byte("not a gguf file"))
	writeModel(t, filepath.Join(root, "cut", "cut.gguf"), gptoss)
	writeModel(t, filepath.Join(root, "cut", "repack", "experts.bin.partial"), []byte{0})
	writeModel(t, filepath.Join(root, "notes.txt"), []byte("x"))
	for i := 0; i < 15; i++ { // more files than the old list's caps (8 recent, 12 scan)
		writeModel(t, filepath.Join(root, "many", "m"+string(rune('a'+i))+".gguf"), gptoss)
	}

	// The catalog model's digest is known (as after a preflight) and matches its pin.
	if _, err := f.l.shardDigests(ctx, []string{f.modelPath}, true); err != nil {
		t.Fatal(err)
	}
	ml, err := f.l.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ml.Files) != 21 {
		t.Fatalf("files %d (no cap expected)", len(ml.Files))
	}
	byName := map[string]ModelFile{}
	for _, mf := range ml.Files {
		byName[mf.Name] = mf
	}
	cat := ml.Files[0] // catalog files first
	if cat.Path != f.modelPath || !cat.Catalog || cat.ProfileID != "fixture-profile" || cat.DisplayName != "Fixture fixture-profile" ||
		cat.State != FileUsable || cat.Identity != SelectionPinned || cat.Alias != "fixture-profile" || !cat.Repacked ||
		cat.Quant != "Q4_K_M" || cat.NeedBasis != NeedNotMeasured {
		t.Errorf("catalog row %+v", cat)
	}
	want := map[string]struct {
		state    FileState
		catalog  bool
		template bool
	}{
		"gpt-oss-20b.gguf":        {FileUsable, false, true},
		"llama-moe.gguf":          {FileUnsupported, false, false},
		"big-00001-of-00002.gguf": {FileBroken, false, true},
		"junk.gguf":               {FileBroken, false, false},
		"cut.gguf":                {FileUsable, false, true}, // conflict 9: a cut setup restarts, see SetupCut
	}
	if !byName["cut.gguf"].SetupCut || byName["gpt-oss-20b.gguf"].SetupCut {
		t.Errorf("setup cut: %+v", byName["cut.gguf"])
	}
	for name, w := range want {
		mf, ok := byName[name]
		if !ok {
			t.Errorf("%s missing", name)
			continue
		}
		if mf.State != w.state || mf.Catalog != w.catalog || mf.Template != w.template {
			t.Errorf("%s: state %s catalog %v template %v reason %q", name, mf.State, mf.Catalog, mf.Template, mf.Reason)
		}
		if mf.State != FileUsable && mf.Reason == "" {
			t.Errorf("%s: no reason", name)
		}
	}
	if mf := byName["gpt-oss-20b.gguf"]; mf.Alias != "gpt-oss-20b" || mf.Arch != "gpt-oss" {
		t.Errorf("template row %+v", mf)
	}
	if mf := byName["big-00001-of-00002.gguf"]; mf.Shards != 1 || mf.Declared != 2 || mf.Alias != "big" {
		t.Errorf("split row %+v", mf)
	}

	// Conflict 14: the template is always on - a stored "off" preference changes nothing.
	writeJSON(t, filepath.Join(f.stateDir, "arch_template_pref.json"), map[string]any{"schema_version": 1, "arch_template": "off"})
	ml, _ = f.l.ListModels(ctx)
	for _, mf := range ml.Files {
		if mf.Name == "gpt-oss-20b.gguf" && (mf.State != FileUsable || !mf.Template) {
			t.Errorf("template must stay on: %+v", mf)
		}
	}
	if a := TemplateArchitectures(); len(a) != 3 || a[0] != "gpt-oss" || a[1] != "qwen35moe" || a[2] != "deepseek2" {
		t.Errorf("template architectures %v", a)
	}
	os.Remove(filepath.Join(f.stateDir, "arch_template_pref.json"))

	// A record gives the need; the catalog file differing from its pin is broken.
	if err := f.l.recordRun(RunRecord{Key: "k", ProfileID: "fixture-profile", ModelPath: f.modelPath, WorkingSetBytes: 2 << 30,
		MaxWorkingSetBytes: 3 << 30, At: "2026-10-09T18:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(f.modelPath)
	b[len(b)-1] ^= 0xff
	os.WriteFile(f.modelPath, b, 0o644)
	settleFile(t, f.modelPath) // a finished rewrite (GL-14's watch holds back a fresh write)
	f.l.shardDigests(ctx, []string{f.modelPath}, true)
	ml, _ = f.l.ListModels(ctx)
	cat = ml.Files[0]
	if !cat.Catalog || cat.State != FileBroken || cat.Identity != SelectionMismatch || cat.NeedMiB != 3072 || cat.NeedBasis != NeedPeak ||
		cat.LastRun == nil {
		t.Errorf("mismatch row %+v", cat)
	}
	// A catalog model absent from this PC is not a row.
	os.Remove(f.modelPath)
	ml, _ = f.l.ListModels(ctx)
	for _, mf := range ml.Files {
		if mf.Catalog {
			t.Errorf("absent catalog model listed: %+v", mf)
		}
	}
}
