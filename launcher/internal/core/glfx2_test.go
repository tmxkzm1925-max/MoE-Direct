// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"bytes"
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGLFX2StatePlacement (GL-FX2; F-LS 8.3 as revised by F-SPEC-1, 505ba38fa): the run
// history, the model digest cache and the measured store of F-BU B.7 are each a direct child of
// the state root %LOCALAPPDATA%\MoE-Direct, beside logs and presets; no path of the launcher
// contains go-draft. The digest cache has a name of its own (sub-lead ruling, GL-FX2): writing
// it leaves a PowerShell cache already at the root byte for byte unchanged.
func TestGLFX2StatePlacement(t *testing.T) {
	f := newFixture(t, false)
	root := f.stateDir
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	// the PowerShell launcher's cache, present as on a machine that ran it (PS:2769), for
	// another file
	psCache := filepath.Join(root, "model_shards.cache.json")
	psBytes := []byte(`{"cache_version":2,"entries":[{"key":"v2|other|1|2","sha256":"` + strings.Repeat("ab", 32) + `"}]}` + "\n")
	if err := os.WriteFile(psCache, psBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	// the three writers
	if err := f.l.recordRun(RunRecord{Key: "k", ModelPath: f.modelPath, ReadyMillis: 1000, At: "2026-10-12T00:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	d, err := f.l.shardDigests(context.Background(), []string{f.modelPath}, true)
	if err != nil || d.Hashed != 1 || !d.Complete {
		t.Fatalf("digests %+v (%v)", d, err)
	}
	if err := f.l.residualStore().Put(ResidualKey{"id", 12288, 4096}, "testarch", AfterRequest, Residuals{CDevMiB: 1, UMiB: 2, Run: "glfx2"}); err != nil {
		t.Fatal(err)
	}

	// no path under the root contains go-draft
	var all []string
	filepath.WalkDir(root, func(p string, e fs.DirEntry, err error) error {
		if err == nil && p != root {
			rel, _ := filepath.Rel(root, p)
			all = append(all, rel)
		}
		return nil
	})
	for _, rel := range all {
		if strings.Contains(strings.ToLower(rel), "go-draft") {
			t.Errorf("a launcher path under go-draft: %s", rel)
		}
	}
	t.Logf("paths under the state root: %v", all)

	// the history and the measured store are direct children of the root
	for _, name := range []string{"history.json", "need_residuals.json"} {
		if st, err := os.Stat(filepath.Join(root, name)); err != nil || st.IsDir() {
			t.Errorf("%s is not a direct child of the state root: %v", name, err)
		}
	}
	// the digest cache: a direct child of the root other than the PowerShell file, holding the
	// digest just computed
	var own []string
	es, _ := os.ReadDir(root)
	for _, e := range es {
		p := filepath.Join(root, e.Name())
		if e.IsDir() || p == psCache || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		if c := readShardCache(p); len(c) == 1 {
			for _, v := range c {
				if v == d.Sha256[0] {
					own = append(own, e.Name())
				}
			}
		}
	}
	if len(own) != 1 {
		t.Errorf("the digest cache as a direct child of the state root: %v", own)
	}
	// the PowerShell cache is read, never written
	if b, err := os.ReadFile(psCache); err != nil || !bytes.Equal(b, psBytes) {
		t.Errorf("the PowerShell cache changed: %q (%v)", b, err)
	}
	t.Logf("digest cache %v · PowerShell cache unchanged (%d B)", own, len(psBytes))
}
