package core

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The real catalog and the installed staging, read only.
const (
	realCatalog = "../../../launcher/models.json"
	stagingRoot = "D:/moe-tools/llama-moedirect-v032"
)

func needStaging(t *testing.T) {
	t.Helper()
	if st, err := os.Stat(filepath.Join(stagingRoot, "expects")); err != nil || !st.IsDir() {
		t.Skip("installed staging not present: " + stagingRoot)
	}
}

type kv struct {
	key string
	typ uint32
	val any
}

func ggufBytes(nTensors uint64, kvs []kv, pad int) []byte {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(nTensors)
	w(uint64(len(kvs)))
	for _, e := range kvs {
		str(e.key)
		w(e.typ)
		switch e.typ {
		case 4:
			w(uint32(e.val.(int)))
		case 8:
			str(e.val.(string))
		case 9: // array of strings
			arr := e.val.([]string)
			w(uint32(8))
			w(uint64(len(arr)))
			for _, s := range arr {
				str(s)
			}
		case 10:
			w(uint64(e.val.(int)))
		}
	}
	b.Write(make([]byte, pad))
	return b.Bytes()
}

func modelKVs(arch string, layers, experts, used, fileType int) []kv {
	return []kv{
		{"general.architecture", 8, arch},
		{"general.name", 8, "fixture"},
		{"tokenizer.ggml.tokens", 9, []string{"a", "bb", "ccc"}},
		{"general.file_type", 4, fileType},
		{arch + ".block_count", 4, layers},
		{arch + ".expert_count", 4, experts},
		{arch + ".expert_used_count", 4, used},
	}
}

type fixture struct {
	dir, root, catalogPath, stateDir, modelPath, outDir string
	expectSha                                           string
	modelBytes                                          int64
	l                                                   *Launcher
	mem                                                 MemStatus
	meter                                               *fakeMeter // GL-10: the speed check's mock timings
}

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func fixtureProfile(id, expectSha string, pin []string) map[string]any {
	p := map[string]any{
		"profile_id": id, "display_name": "Fixture " + id, "hf_repo": "x/y", "hf_revision": "main",
		"routed_scope": "all", "expect_file": "fixture.expect.json", "expect_sha256": expectSha,
		"identify":      map[string]any{"arch": "testarch", "n_layer": 2, "n_expert": 4, "n_expert_used": 2},
		"min_budget_mb": 4096, "prefetch_evidence": "observe", "prefetch_activation": "off",
		"prefetch_promotion_hold": true, "prefetch": nil,
		"gates":                  map[string]any{"format_validated": true, "performance_validated": false},
		"reference_measurements": []any{},
		"allowlist_bounds": map[string]any{
			"port": map[string]any{"min": 1024, "max": 65535}, "ctx": map[string]any{"min": 2048, "max": 262144},
			"threads": map[string]any{"min": 1, "max": 64}, "budget_mb": map[string]any{"min": 4096, "max": 65536},
			"qd": map[string]any{"min": 1, "max": 63}},
		"defaults": map[string]any{"argv": []string{"-ngl", "99", "--n-cpu-moe", "2", "-c", "4096", "-t", "8",
			"-np", "1", "--host", "127.0.0.1", "--port", "8093", "--no-webui", "--no-warmup"}, "env": map[string]any{}},
	}
	if pin != nil {
		p["source_shards_sha256"] = pin
	}
	return p
}

// newFixture builds a bundle root, a catalog, one model file with its packed repack output
// and a state directory, all under t.TempDir(), and a Launcher whose OS reads are fixed.
func newFixture(t *testing.T, pin bool) *fixture {
	t.Helper()
	f := &fixture{dir: t.TempDir()}
	f.root = filepath.Join(f.dir, "bundle")
	f.stateDir = filepath.Join(f.dir, "state")
	mdir := filepath.Join(f.dir, "models", "m")
	if err := os.MkdirAll(mdir, 0o755); err != nil {
		t.Fatal(err)
	}
	f.modelPath = filepath.Join(mdir, "fixture-model.gguf")
	mb := ggufBytes(10, modelKVs("testarch", 2, 4, 2, 15), 4096)
	if err := os.WriteFile(f.modelPath, mb, 0o644); err != nil {
		t.Fatal(err)
	}
	settleFile(t, f.modelPath)
	f.modelBytes = int64(len(mb))
	expectPath := filepath.Join(f.root, "expects", "fixture.expect.json")
	writeJSON(t, expectPath, map[string]any{"sources": []any{map[string]any{"file_bytes": f.modelBytes}}, "expert_bytes_total": 1 << 30})
	sha, err := fileSha256(expectPath)
	if err != nil {
		t.Fatal(err)
	}
	f.expectSha = sha
	var pins []string
	if pin {
		ms, _ := fileSha256(f.modelPath)
		pins = []string{ms}
	}
	f.catalogPath = filepath.Join(f.dir, "models.json")
	writeJSON(t, f.catalogPath, map[string]any{
		"catalog_schema_version": 2, "source_tag": "fixture-tag",
		"runtime":  map[string]any{"server_exe": fakeServerExe, "repacker_exe": "r.exe", "repacker_argv": []any{}, "expects_dir": "expects", "webui": false},
		"profiles": []any{fixtureProfile("fixture-profile", sha, pins)},
	})
	f.outDir = filepath.Join(mdir, "repack")
	os.MkdirAll(f.outDir, 0o755)
	os.WriteFile(filepath.Join(f.outDir, "experts.bin"), []byte{0}, 0o644)
	writePackedArtifacts(t, f.outDir, "fixture-profile", sha)
	l, err := New(Options{CatalogPath: f.catalogPath, BundleRoot: f.root, StateDir: f.stateDir})
	if err != nil {
		t.Fatal(err)
	}
	l.lockPrefix = fmt.Sprintf("gotest_%d_%d_", os.Getpid(), time.Now().UnixNano())
	f.mem = MemStatus{TotalPhysMiB: 131072, AvailPhysMiB: 65536}
	l.os.mem = func() (MemStatus, error) { return f.mem, nil }
	l.os.installed = func() (int64, error) { return 65536, nil }
	l.os.volumeGUID = func(string) (string, error) { return `\\?\volume{fixture}\`, nil }
	l.os.portOwners = func(int) ([]int, error) { return []int{}, nil }
	l.os.console = func() bool { return true }
	l.os.vram = func() VRAMInfo { return VRAMInfo{OK: true, Source: "fixture", TotalMiB: 1, FreeMiB: 0} }
	l.os.volumeFree = func(string) (int64, int64, string, error) { return 500000, 1000000, `X:\`, nil }
	l.os.modelRoots = func() []string { return nil }
	// GOL-FIX1: the tests never touch the real clipboard, Explorer or the GPU counter.
	l.os.clipboard = func(string) error { return errors.New("fixture: no clipboard") }
	l.os.launch = func(string, string) error { return errors.New("fixture: no launch") }
	l.os.gpuMem = func() (map[string]int64, error) { return map[string]int64{}, nil }
	// GL-10: the speed check reads through mock timings, never the disk.
	f.meter = &fakeMeter{}
	l.os.sweep = f.meter
	f.l = l
	return f
}

// writePackedArtifacts writes a manifest and a verify report that pass the 7-item verify gate
// (FIX2-CORE: the gate runs at every preflight and start).
func writePackedArtifacts(t *testing.T, outDir, profileID, expectSha string) {
	t.Helper()
	lock := map[string]any{"profile_id": profileID, "expect_sha256": expectSha}
	writeJSON(t, filepath.Join(outDir, "manifest.json"), map[string]any{
		"schema_version": "2.0", "reference_lock": lock,
		"model":  map[string]any{"n_expert": 4},
		"layout": map[string]any{"slot_stride_max": 1 << 30, "layers": []any{map[string]any{}, map[string]any{}}}})
	ms, err := fileSha256(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	rep := map[string]any{"pass": true, "checked_at": "2026-10-07T01:51:39.264165+00:00", "pairs_pass": 8, "pairs_total": 8,
		"expected_pairs": 8, "problems": []any{}, "failures": []any{}, "padding_failures": []any{},
		"reference_lock": lock, "manifest_sha256": ms}
	b, _ := json.Marshal(rep)
	if err := os.WriteFile(filepath.Join(outDir, "verify_report.json"), append(b, '\n', '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeVirtualArtifacts writes a virtual manifest and plan report that pass the 8-item gate.
func writeVirtualArtifacts(t *testing.T, outDir, profileID, expectSha string) {
	t.Helper()
	lock := map[string]any{"profile_id": profileID, "expect_sha256": expectSha}
	writeJSON(t, filepath.Join(outDir, "manifest.json"), map[string]any{
		"schema_version": "3.0", "mode": "virtual", "reference_lock": lock, "totals": map[string]any{"n_records": 16},
		"model":  map[string]any{"n_expert": 4},
		"layout": map[string]any{"slot_stride_max": 1 << 30, "layers": []any{1, 2}}})
	ms, err := fileSha256(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(outDir, "plan_report.json"), map[string]any{"mode": "virtual", "pass": true, "problems": []any{},
		"manifest_sha256": ms, "reference_lock": lock, "cardinality": map[string]any{"records": 16},
		"checked_at": "2026-10-08T00:00:00+00:00"})
}
