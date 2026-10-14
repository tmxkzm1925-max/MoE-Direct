// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestDump writes every screen at every test size, in both languages, as
// ANSI text into the folder named by GOLUI_DUMP (skipped when it is unset),
// for a side-by-side look with designs/launcher_design/proto/cap/go5.
func TestDump(t *testing.T) {
	dir := os.Getenv("GOLUI_DUMP")
	if dir == "" {
		t.Skip("GOLUI_DUMP not set")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	ids := append(append([]string{}, designIDs...), "S0")
	for _, id := range ids {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				hs := reach(t, id, lang, sz[0], sz[1])
				name := filepath.Join(dir, fmt.Sprintf("%s_%d_%s.ans", id, sz[0], lang))
				if err := os.WriteFile(name, []byte(hs.m.View()+"\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
