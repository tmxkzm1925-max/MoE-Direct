//go:build windows

package core

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// BND-ROOT (version 8 repair after the RC_8 seal): CheckBundle gives every walked file its
// bundle-relative path whatever spelling of the root it is given. filepath.WalkDir joins the
// children with backslashes while the root keeps the slashes it was given, so before the repair
// a root spelt C:/... left every file, the manifest included, as an absolute «unlisted» path
// (leg DB of RC_8: fail_gate_bundle, exit 4, 89 = 88 + 1).
func TestBNDROOTRootSpelling(t *testing.T) {
	root := la27Fixture(t)
	fwd, back := filepath.ToSlash(root), filepath.FromSlash(root)
	for _, c := range []struct{ name, root string }{
		{"forward slashes", fwd},
		{"backslashes", back},
		{"forward slashes, trailing separator", fwd + "/"},
		{"backslashes, trailing separator", back + `\`},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := CheckBundle(context.Background(), c.root)
			if r.State != BundleOK || r.Mismatches != 0 || len(r.Diffs) != 0 || r.Files != len(la27Files) {
				t.Fatalf("root %q: state=%v files=%d mismatches=%d diffs=%+v reason=%q",
					c.root, r.State, r.Files, r.Mismatches, r.Diffs, r.Reason)
			}
		})
	}
	t.Run("one extra file, forward slashes", func(t *testing.T) {
		if err := os.WriteFile(filepath.Join(root, "engine", "extra.dll"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		r := CheckBundle(context.Background(), fwd)
		want := BundleDiff{Path: `engine\extra.dll`, Kind: "unlisted"}
		if r.State != BundleMismatch || r.Mismatches != 1 || len(r.Diffs) != 1 || r.Diffs[0] != want {
			t.Fatalf("root %q: state=%v files=%d mismatches=%d diffs=%+v reason=%q",
				fwd, r.State, r.Files, r.Mismatches, r.Diffs, r.Reason)
		}
	})
}
