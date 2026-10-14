package core

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// S8-LA-27 (GL-13): the launcher file check against bundle_manifest.json as the assembler
// writes it (make_bundle.ps1 region 5) and as Assert-BundleIntegrity reads it (PS:2187-2238),
// with conflict 18-2's count of mismatches, the reinstall request and the development build.
// Every fixture bundle lives in t.TempDir().

// la27Files is a small bundle in the assembler's layout (names only): the launcher, an engine
// file under a folder, the catalog and a repacker expect file.
var la27Files = map[string]string{
	"MoE-Direct-launcher.exe":           "launcher",
	"engine/llama-server.exe":           "server",
	"models.json":                       `{"catalog_schema_version":1}`,
	"repacker/expects/qwen.expect.json": `{"expect":1}`,
}

// la27Assemble writes the files and then the manifest the way make_bundle.ps1 region 5 writes
// it: every file under the root but the manifest, its path relative with '/', sorted, one
// entry per line, CRLF line ends (StringBuilder.AppendLine on Windows), UTF-8 without BOM.
func la27Assemble(t *testing.T, root string, files map[string]string) {
	t.Helper()
	var rels []string
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	var sb strings.Builder
	sb.WriteString("{\r\n")
	fmt.Fprintf(&sb, "  \"bundle_manifest_version\": %d,\r\n", BundleManifestVersion)
	sb.WriteString("  \"files\": [\r\n")
	for i, rel := range rels {
		comma := ","
		if i == len(rels)-1 {
			comma = ""
		}
		fmt.Fprintf(&sb, "    { \"path\": \"%s\", \"sha256\": \"%s\" }%s\r\n", rel, sha256Hex([]byte(files[rel])), comma)
	}
	sb.WriteString("  ]\r\n}\r\n")
	if err := os.WriteFile(filepath.Join(root, BundleManifestName), []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func la27Fixture(t *testing.T) string {
	root := t.TempDir()
	la27Assemble(t, root, la27Files)
	return root
}

// la27Harm: an unlisted file, a missing listed file, one changed byte.
var la27Harm = []struct {
	name, path, kind string
	do               func(t *testing.T, root string)
}{
	{"unlisted file", `engine\extra.dll`, "unlisted", func(t *testing.T, root string) {
		if err := os.WriteFile(filepath.Join(root, "engine", "extra.dll"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}},
	{"missing listed file", "engine/llama-server.exe", "missing", func(t *testing.T, root string) {
		if err := os.Remove(filepath.Join(root, "engine", "llama-server.exe")); err != nil {
			t.Fatal(err)
		}
	}},
	{"one changed byte", "models.json", "hash", func(t *testing.T, root string) {
		p := filepath.Join(root, "models.json")
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		b[len(b)/2] ^= 0x01
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}},
}

// la27Refusal: the start refusal of a failed check names the count and asks for a reinstall,
// fail_gate_bundle, exit 4.
func la27Refusal(t *testing.T, err error, n int) {
	t.Helper()
	if err == nil || StatusOf(err) != StatusFailGateBundle || ExitCodeOf(err) != 4 {
		t.Fatalf("refusal %v (status %s, exit %d)", err, StatusOf(err), ExitCodeOf(err))
	}
	want := fmt.Sprintf("%d file(s) differ from bundle_manifest.json · install again", n)
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("refusal %q lacks %q", err.Error(), want)
	}
}

func la27Launcher(t *testing.T, root string) *Launcher {
	t.Helper()
	dir := t.TempDir()
	// the catalog is absent: a start that passes the bundle check ends at the catalog step,
	// before any lock or child
	l, err := New(Options{CatalogPath: filepath.Join(dir, "models.json"), BundleRoot: root, StateDir: filepath.Join(dir, "state")})
	if err != nil {
		t.Fatal(err)
	}
	return l
}

type la27Event struct {
	ev StartEvent
	at time.Time
}

func la27Start(t *testing.T, l *Launcher) []la27Event {
	t.Helper()
	ch, err := l.Start(context.Background(), Request{ModelPath: filepath.Join(t.TempDir(), "m.gguf")})
	if err != nil {
		t.Fatal(err)
	}
	var out []la27Event
	for ev := range ch {
		out = append(out, la27Event{ev, time.Now()})
	}
	return out
}

func TestS8LA27(t *testing.T) {
	ctx := context.Background()
	t.Run("intact bundle passes", func(t *testing.T) {
		r := CheckBundle(ctx, la27Fixture(t))
		if r.State != BundleOK || r.Files != len(la27Files) || r.Mismatches != 0 || r.Err() != nil ||
			r.Started.IsZero() || r.Finished.Before(r.Started) {
			t.Fatalf("%+v %v", r, r.Err())
		}
	})
	for _, h := range la27Harm {
		t.Run(h.name, func(t *testing.T) {
			root := la27Fixture(t)
			h.do(t, root)
			r := CheckBundle(ctx, root)
			if r.State != BundleMismatch || r.Mismatches != 1 || len(r.Diffs) != 1 ||
				r.Diffs[0] != (BundleDiff{Path: h.path, Kind: h.kind}) {
				t.Fatalf("%+v", r)
			}
			la27Refusal(t, r.Err(), 1)
		})
	}
	t.Run("all three count three", func(t *testing.T) {
		root := la27Fixture(t)
		for _, h := range la27Harm {
			h.do(t, root)
		}
		r := CheckBundle(ctx, root)
		if r.State != BundleMismatch || r.Mismatches != 3 {
			t.Fatalf("%+v", r)
		}
		la27Refusal(t, r.Err(), 3)
	})
	t.Run("development build skips", func(t *testing.T) {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "extra.dll"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		r := CheckBundle(ctx, root)
		if r.State != BundleDev || r.Err() != nil || r.Mismatches != 0 ||
			!strings.Contains(r.Reason, "development build, bundle check skipped") {
			t.Fatalf("%+v", r)
		}
	})

	// The start waits for the check's end. A refused bundle: the start began before the check
	// (the start starts it), its refusal is the check's and nothing after the bundle step runs.
	t.Run("start refused after the check ends", func(t *testing.T) {
		root := la27Fixture(t)
		la27Harm[0].do(t, root)
		la27Harm[2].do(t, root)
		l := la27Launcher(t, root)
		t0 := time.Now()
		evs := la27Start(t, l)
		last := evs[len(evs)-1].ev
		if !last.Final || last.Status != StatusFailGateBundle || last.ExitCode != 4 {
			t.Fatalf("final %+v", last)
		}
		la27Refusal(t, last.Err, 2)
		var br *BundleResult
		for _, e := range evs {
			if e.ev.Step != "" && e.ev.Step != StepBundle {
				t.Fatalf("a step after the refused bundle check ran: %+v", e.ev)
			}
			if e.ev.Bundle != nil {
				br = e.ev.Bundle
			}
		}
		if br == nil || br.State != BundleMismatch || br.Started.Before(t0) || br.Finished.Before(br.Started) ||
			br.Finished.After(evs[len(evs)-1].at) {
			t.Fatalf("bundle %+v", br)
		}
	})
	// An intact bundle (with a file large enough that hashing takes a while): the check started
	// as the screen starts it, the start right after; the start's bundle step ends with the
	// finished check and the next step begins after its end.
	t.Run("start waits for the check", func(t *testing.T) {
		files := map[string]string{"engine/ggml-big.dll": strings.Repeat("\x00", 16<<20)}
		for k, v := range la27Files {
			files[k] = v
		}
		root := t.TempDir()
		la27Assemble(t, root, files)
		l := la27Launcher(t, root)
		l.StartBundleCheck()
		evs := la27Start(t, l)
		var br *BundleResult
		next := -1
		for i, e := range evs {
			if e.ev.Step == StepBundle && e.ev.State == StepDone {
				br = e.ev.Bundle
			}
			if e.ev.Step == StepCatalog && next < 0 {
				next = i
			}
		}
		if br == nil || br.State != BundleOK || br.Files != len(files) || br.Started.IsZero() || br.Finished.Before(br.Started) {
			t.Fatalf("the start did not wait for the check: %+v", br)
		}
		if next < 0 || evs[next].at.Before(br.Finished) {
			t.Fatalf("the catalog step began before the check ended (%d)", next)
		}
		if last := evs[len(evs)-1].ev; last.Status != StatusFailGateCatalog {
			t.Fatalf("final %+v", last)
		}
	})
	t.Run("development build start", func(t *testing.T) {
		l := la27Launcher(t, t.TempDir())
		evs := la27Start(t, l)
		ok := false
		for _, e := range evs {
			if e.ev.Step == StepBundle && e.ev.State == StepDone && e.ev.Detail == "dev" && e.ev.Bundle != nil && e.ev.Bundle.State == BundleDev {
				ok = true
			}
		}
		if !ok || evs[len(evs)-1].ev.Status != StatusFailGateCatalog {
			t.Fatalf("%+v", evs)
		}
	})
}
