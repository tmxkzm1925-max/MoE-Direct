package core

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestUXR12bEnglish (UX-R12b; UX-R5 landing 4 items 5 and 6): core's sentences are English; the
// screens word them by their table. The memory verdict reads «not measured», the warmup's forced
// reason and the stop reason during the check run name it «check run», and no string literal of
// core's source carries the old Korean names «작동 확인» or «미측정».
func TestUXR12bEnglish(t *testing.T) {
	for v, want := range map[MemoryVerdict]string{MemUnmeasured: "not measured", MemFits: "fits", MemShort: "short"} {
		if v.String() != want {
			t.Errorf("verdict %d: %q, want %q", v, v.String(), want)
		}
	}
	if !strings.Contains(warmupForcedReason, "check run") {
		t.Errorf("warmup forced reason: %q", warmupForcedReason)
	}
	fs, err := filepath.Glob("*.go")
	if err != nil || len(fs) == 0 {
		t.Fatal(fs, err)
	}
	set := token.NewFileSet()
	stopCheckRun := false
	for _, f := range fs {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(set, f, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			b, ok := n.(*ast.BasicLit)
			if !ok || b.Kind != token.STRING {
				return true
			}
			s, _ := strconv.Unquote(b.Value)
			if strings.Contains(s, "작동 확인") || strings.Contains(s, "미측정") {
				t.Errorf("%s: a Korean name in an English sentence: %q", set.Position(b.Pos()), s)
			}
			if f == "start.go" && s == "check run" {
				stopCheckRun = true
			}
			return true
		})
	}
	if !stopCheckRun {
		t.Error("start.go: the stop during the check run is not named «check run»")
	}
}
