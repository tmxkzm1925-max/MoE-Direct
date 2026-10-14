//go:build windows

package core

import (
	"context"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func la27Attr(t *testing.T, p string, add uint32) {
	t.Helper()
	ptr, err := syscall.UTF16PtrFromString(p)
	if err != nil {
		t.Fatal(err)
	}
	a, err := syscall.GetFileAttributes(ptr)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.SetFileAttributes(ptr, a|add); err != nil {
		t.Fatal(err)
	}
}

// GL-13: the walk for unlisted files is Get-ChildItem -Recurse -File without -Force, as the
// assembler and Assert-BundleIntegrity run it: a Hidden file and a Hidden folder (with what is
// in it) are left out, a System-only file is not, and a Hidden listed file is still checked.
func TestGL13HiddenAsGetChildItem(t *testing.T) {
	root := la27Fixture(t)
	write := func(rel string) string {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	la27Attr(t, write("desktop.ini"), syscall.FILE_ATTRIBUTE_HIDDEN)
	write(`cache\inside.bin`)
	la27Attr(t, filepath.Join(root, "cache"), syscall.FILE_ATTRIBUTE_HIDDEN)
	la27Attr(t, write(`engine\sys.bin`), syscall.FILE_ATTRIBUTE_SYSTEM)
	la27Attr(t, filepath.Join(root, "engine", "llama-server.exe"), syscall.FILE_ATTRIBUTE_HIDDEN)
	r := CheckBundle(context.Background(), root)
	if r.State != BundleMismatch || r.Mismatches != 1 || r.Diffs[0] != (BundleDiff{Path: `engine\sys.bin`, Kind: "unlisted"}) {
		t.Fatalf("%+v", r)
	}
}
