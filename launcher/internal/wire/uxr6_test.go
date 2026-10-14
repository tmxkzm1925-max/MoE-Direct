package wire

// UX-R6 (version 8; the user's acceptance finding 12, reviews/_user_8_LAUNCHER-REVIEW_2.md,
// screenshots 055107 and 055112): on the settings box the ctx value stepped with ←→ stopped at
// 65536 for every model, while the yellow range read «2k ~ 128k». The stop was the step list
// of settingCandidates (map.go, GOL-WIRE 9ab4e535a), which ends at 65536 with no recorded
// reason; the steps now run on to the row's own maximum, the allowlist bound.

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// uxr6Backend is a Backend over the real catalog with one model file: the catalog's
// gpt-oss-120b profile, or (template) a gpt-oss file outside the catalog.
func uxr6Backend(t *testing.T, template bool) (*Backend, string) {
	t.Helper()
	root := t.TempDir()
	folder := filepath.Join(root, "moe-models")
	model := filepath.Join(folder, "gpt-oss-120b", "gpt-oss-120b-MXFP4.gguf")
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(model, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog := filepath.Join("..", "..", "..", "launcher", "models.json")
	l, err := core.New(core.Options{CatalogPath: catalog, BundleRoot: filepath.Join("..", "..", "..", "..", "repack"),
		StateDir: filepath.Join(root, "state"), ModelsRoot: folder})
	if err != nil {
		t.Fatal(err)
	}
	b := New(l, catalog, "test")
	f := core.ModelFile{Path: model, Name: filepath.Base(model), Folder: folder, DisplayName: "gpt-oss-120b MXFP4",
		ProfileID: "gpt-oss-120b", Catalog: true, Bytes: 4, Shards: 1, Declared: 1, Identity: "unpinned"}
	if template {
		f.DisplayName, f.ProfileID, f.Catalog, f.Template, f.Arch = "gpt-oss-120b-local", "", false, true, "gpt-oss"
	}
	b.mu.Lock()
	b.ml, b.stamp = &core.ModelList{Files: []core.ModelFile{f}, Roots: []string{folder}}, l.ScanStamp()
	b.files = map[string]core.ModelFile{model: f}
	b.mu.Unlock()
	b.bundleFn = func() core.BundleResult { return core.BundleResult{State: core.BundleDev} }
	return b, model
}

// uxr6Bound is the ctx bound of the source the row stands on: the catalog profile's
// allowlist_bounds, or the structure template's (derive.go derivedBounds).
func uxr6Bound(t *testing.T, template bool) core.Bound {
	t.Helper()
	if template {
		d, ok := core.TemplateDefaults("gpt-oss")
		if !ok {
			t.Fatal("no gpt-oss template")
		}
		return d.Bounds["ctx"]
	}
	cat, err := core.ReadCatalog(filepath.Join("..", "..", "..", "launcher", "models.json"), filepath.Join("..", "..", "..", "..", "repack"))
	if err != nil {
		t.Fatal(err)
	}
	p := cat.ProfileByID("gpt-oss-120b")
	if p == nil {
		t.Fatal("no gpt-oss-120b profile")
	}
	return p.Bounds["ctx"]
}

// TestUXR6CtxSteps: on the real screen over this Backend, ←→ on the ctx row of the settings
// box steps from the default up to the row's own maximum, the allowlist bound of a 128k-bound
// catalog model and of a derived profile, and never past it (the next step wraps to the
// lowest value); the range says what bounds it.
func TestUXR6CtxSteps(t *testing.T) {
	for _, template := range []bool{false, true} {
		name := map[bool]string{false: "catalog gpt-oss-120b", true: "derived gpt-oss (outside the catalog)"}[template]
		bd := uxr6Bound(t, template)
		if bd.Max != 131072 {
			t.Fatalf("%s: the ctx bound is %d, the test is for a 128k-bound model", name, bd.Max)
		}
		b, model := uxr6Backend(t, template)
		v, err := b.ModelSettings(model)
		if err != nil {
			t.Fatal(err)
		}
		var row ui.SettingRow
		for _, r := range v.Rows {
			if r.Key == "ctx" {
				row = r
			}
		}
		wantBasis := map[bool]string{false: basisCatalog, true: basisTemplate}[template]
		if row.Note != "2k ~ 128k" || row.Basis != wantBasis {
			t.Fatalf("%s: ctx note %q basis %q, want «2k ~ 128k» and %q", name, row.Note, row.Basis, wantBasis)
		}

		s := &uxr8Screen{t: t, clock: time.Date(2026, 10, 13, 9, 0, 0, 0, time.UTC)}
		s.m = ui.New(b, ui.Options{Lang: ui.LangEnglish, Width: 160, Height: 40,
			Now: func() time.Time { return s.clock },
			Tick: func(_ time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
				s.ticks = append(s.ticks, fn)
				return nil
			}})
		s.run(s.m.Init())
		s.fire()
		s.send(tea.KeyMsg{Type: tea.KeyTab}) // into the settings box
		// UX-R10: a state mark follows ▶
		focused := regexp.MustCompile(`▶\S context +◀ (\d+)`)
		for i := 0; i < 16 && !focused.MatchString(s.plain()); i++ {
			s.send(tea.KeyMsg{Type: tea.KeyDown})
		}
		cur := func() int64 {
			p := s.plain()
			sub := focused.FindStringSubmatch(p)
			if sub == nil {
				t.Fatalf("%s: the ctx row is not focused:\n%s", name, p)
			}
			val := sub[1]
			n, err := strconv.ParseInt(val, 10, 64)
			if err != nil {
				t.Fatalf("%s: the focused ctx value %q:\n%s", name, val, p)
			}
			return n
		}
		start := cur()
		if strconv.FormatInt(start, 10) != row.Value {
			t.Fatalf("%s: the ctx row starts at %d, its value is %s", name, start, row.Value)
		}
		seq, prev, reached := []int64{start}, start, false
		for i := 0; i < 40; i++ {
			s.send(tea.KeyMsg{Type: tea.KeyRight})
			n := cur()
			seq = append(seq, n)
			if n > bd.Max || n < bd.Min {
				t.Fatalf("%s: a step left the bound %d..%d: %v", name, bd.Min, bd.Max, seq)
			}
			if n < prev { // wrapped to the lowest value: the top was reached just before
				break
			}
			if n == bd.Max {
				reached = true
			}
			prev = n
		}
		if !reached || prev != bd.Max {
			t.Fatalf("%s: stepping from the default %d does not reach the bound %d before it wraps: %v", name, start, bd.Max, seq)
		}
		t.Logf("%s: ctx %d..%d, steps from the default: %v", name, bd.Min, bd.Max, seq)
	}
}

// TestUXR6BoundedOptions: the ctx steps reach any bound, also one off the doubling steps, and
// stay inside it; a row without bounds keeps the step list as it was.
func TestUXR6BoundedOptions(t *testing.T) {
	for _, max := range []int64{131072, 200000, 262144, 32768} {
		opts := boundedOptions(core.SettingRow{Key: "ctx", Default: "12288", Editable: true, Min: 2048, Max: max, Bounded: true})
		top := opts[len(opts)-1]
		if top != strconv.FormatInt(max, 10) {
			t.Fatalf("bound %d: the steps end at %s: %v", max, top, opts)
		}
		for _, o := range opts {
			if n, _ := strconv.ParseInt(o, 10, 64); n > max {
				t.Fatalf("bound %d: step %s passes it: %v", max, o, opts)
			}
		}
		t.Logf("bound %d: %v", max, opts)
	}
	if opts := boundedOptions(core.SettingRow{Key: "ctx", Default: "12288", Editable: true}); opts[len(opts)-1] != "65536" {
		t.Fatalf("unbounded ctx: %v", opts)
	}
}
