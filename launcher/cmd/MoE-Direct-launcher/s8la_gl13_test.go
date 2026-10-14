package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"moedirect/launcher/internal/core"
)

// la27Root writes a fixture bundle in t.TempDir() (manifest only when listed is true).
func la27Root(t *testing.T, listed bool) string {
	t.Helper()
	root := t.TempDir()
	var list []map[string]string
	for rel, body := range map[string]string{"engine/llama-server.exe": "server", "models.json": "{}"} {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256([]byte(body))
		list = append(list, map[string]string{"path": rel, "sha256": hex.EncodeToString(h[:])})
	}
	if listed {
		b, _ := json.Marshal(map[string]any{"bundle_manifest_version": 1, "files": list})
		if err := os.WriteFile(filepath.Join(root, core.BundleManifestName), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func la27Run(t *testing.T, root string) (int, string, string) {
	t.Helper()
	dir := t.TempDir()
	l, err := core.New(core.Options{CatalogPath: filepath.Join(dir, "models.json"), BundleRoot: root, StateDir: filepath.Join(dir, "state")})
	if err != nil {
		t.Fatal(err)
	}
	l.StartBundleCheck() // as run() does on this path
	var out, errw bytes.Buffer
	code := runArgs(l, argStart{model: filepath.Join(dir, "m.gguf")}, &out, &errw, nil)
	return code, out.String(), errw.String()
}

// S8-LA-27 (GL-13; conflict 18-2) on the argument-driven path: a damaged bundle refuses the
// start with the count and the reinstall request, fail_gate_bundle, exit 4; an intact bundle
// and a development build pass the check and say so (the absent catalog then ends the start).
func TestS8LA27(t *testing.T) {
	root := la27Root(t, true)
	os.WriteFile(filepath.Join(root, "extra.dll"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(root, "models.json"), []byte("{ }"), 0o644)
	code, out, errw := la27Run(t, root)
	if code != 4 || !strings.Contains(errw, "2 file(s) differ from bundle_manifest.json · install again") ||
		!strings.HasSuffix(strings.TrimSpace(errw), "status=fail_gate_bundle") || strings.Contains(out, "[start] catalog") {
		t.Fatalf("damaged: code %d\nout %q\nerr %q", code, out, errw)
	}
	for _, c := range []struct {
		listed bool
		line   string
	}{
		{true, "[start] bundle: 2 file(s) match bundle_manifest.json"},
		{false, "[start] bundle: development build · launcher file check skipped"},
	} {
		code, out, errw := la27Run(t, la27Root(t, c.listed))
		if !strings.Contains(out, c.line) || !strings.HasSuffix(strings.TrimSpace(errw), "status=fail_gate_catalog") {
			t.Fatalf("listed %v: code %d\nout %q\nerr %q", c.listed, code, out, errw)
		}
	}
}
