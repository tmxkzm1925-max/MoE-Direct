package core

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Review item 31: a deleted GGUF and a deleted repack output change the stamp, and the list
// that follows drops the file or shows the output missing.
func TestScanStampItem31(t *testing.T) {
	f := newFixture(t, true)
	root := filepath.Join(f.dir, "models")
	f.l.os.modelRoots = func() []string { return []string{root} }
	other := filepath.Join(root, "o", "gpt-oss-20b.gguf")
	writeModel(t, other, ggufBytes(10, modelKVs("gpt-oss", 36, 128, 4, 15), 64))

	s1 := f.l.ScanStamp()
	if s2 := f.l.ScanStamp(); s1 != s2 || len(s1) != 64 {
		t.Fatalf("stamp not stable: %s %s", s1, s2)
	}
	// Reading the list (no hashing, no writing) leaves the stamp as it was.
	if _, err := f.l.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s := f.l.ScanStamp(); s != s1 {
		t.Error("ListModels changed the stamp")
	}

	// A deleted GGUF: the stamp changes and the file is gone from the list.
	if err := os.Remove(other); err != nil {
		t.Fatal(err)
	}
	s3 := f.l.ScanStamp()
	if s3 == s1 {
		t.Error("deleted GGUF: stamp unchanged")
	}
	ml, _ := f.l.ListModels(context.Background())
	for _, mf := range ml.Files {
		if mf.Path == other {
			t.Error("deleted GGUF still listed")
		}
	}

	// A deleted repack output: the stamp changes and the row shows the output missing.
	if err := os.Remove(filepath.Join(f.outDir, "experts.bin")); err != nil {
		t.Fatal(err)
	}
	s4 := f.l.ScanStamp()
	if s4 == s3 {
		t.Error("deleted repack output: stamp unchanged")
	}
	ml, _ = f.l.ListModels(context.Background())
	if len(ml.Files) != 1 || ml.Files[0].Repacked || ml.Files[0].RepackedVirtual {
		t.Errorf("after deleting experts.bin: %+v", ml.Files)
	}

	// A rewritten file (same name, new time) changes it too.
	later := time.Now().Add(time.Hour)
	os.Chtimes(f.modelPath, later, later)
	if f.l.ScanStamp() == s4 {
		t.Error("touched GGUF: stamp unchanged")
	}
}
