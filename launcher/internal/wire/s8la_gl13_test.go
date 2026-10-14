package wire

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// la27Bundle writes a fixture bundle in t.TempDir(): the files and a manifest listing them
// (the assembler's exact text is core's TestS8LA27).
func la27Bundle(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	var list []map[string]string
	for rel, body := range files {
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
	b, _ := json.Marshal(map[string]any{"bundle_manifest_version": 1, "files": list})
	if err := os.WriteFile(filepath.Join(root, core.BundleManifestName), b, 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// la27Wait asks for the check until it has answered.
func la27Wait(t *testing.T, b *Backend) ui.BundleCheck {
	t.Helper()
	for end := time.Now().Add(10 * time.Second); time.Now().Before(end); time.Sleep(5 * time.Millisecond) {
		if c, err := b.Bundle(); err != nil || c.State != ui.BundleChecking {
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
	}
	t.Fatal("the launcher file check never ended")
	return ui.BundleCheck{}
}

// S8-LA-27 (GL-13; conflict 18-2): the screen's first Bundle call starts core's check (the
// screen makes it after its first draw); nothing starts it before; its result goes to 4 Log
// once.
func TestS8LA27(t *testing.T) {
	for _, c := range []struct {
		name  string
		harm  func(root string)
		want  ui.BundleCheck
		level ui.LogLevel
		log   string
	}{
		{"intact", func(string) {}, ui.BundleCheck{State: ui.BundleOK}, ui.LogOK, "launcher file check: 2 files match"},
		{"damaged", func(root string) {
			os.WriteFile(filepath.Join(root, "extra.dll"), []byte("x"), 0o644)
			os.Remove(filepath.Join(root, "engine", "llama-server.exe"))
		}, ui.BundleCheck{State: ui.BundleBroken, Mismatches: 2}, ui.LogError, "2 file(s) differ"},
	} {
		t.Run(c.name, func(t *testing.T) {
			root := la27Bundle(t, map[string]string{"engine/llama-server.exe": "server", "models.json": "{}"})
			c.harm(root)
			dir := t.TempDir()
			l, err := core.New(core.Options{CatalogPath: filepath.Join(dir, "models.json"), BundleRoot: root, StateDir: filepath.Join(dir, "state")})
			if err != nil {
				t.Fatal(err)
			}
			b := New(l, filepath.Join(dir, "models.json"), "test")
			time.Sleep(50 * time.Millisecond)
			if r := l.BundleCheckResult(); r.State != core.BundlePending {
				t.Fatalf("the check started before the screen asked for it: %+v", r)
			}
			if got := la27Wait(t, b); got != c.want {
				t.Fatalf("got %+v, want %+v", got, c.want)
			}
			b.Bundle()
			lines, n := logLines(b)
			if n != 1 || !strings.Contains(lines, c.log) || b.events[0].Level != c.level {
				t.Fatalf("log %d %q", n, lines)
			}
		})
	}
}
