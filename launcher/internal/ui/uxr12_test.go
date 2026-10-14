package ui

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

// UX-R12 (version 8; the user's findings 8, 10 (reviews/_user_8_LAUNCHER-REVIEW_1.md) and 20 (c)
// (reviews/_user_8_LAUNCHER-REVIEW_9.md), the remainder on 2 Server): the release build shows no
// «not ported» / «미이식» value. UX-R9 put 2 Server's r.np() places under the switch; here every
// value 2 Server draws through r.vt / r.vu (svt / svu), a step or a cache save in the not-ported
// state, and the stop alert follow it.

// uxr12NP sets every Val reachable from v (structs, pointers, slices, maps) to the NotPorted
// state, and every step's state to StepNotPorted where steps are listed, so each 2 Server place
// that can draw a not-ported value does.
func uxr12NP(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			uxr12NP(v.Elem())
		}
	case reflect.Struct:
		if v.Type() == reflect.TypeOf(Val{}) {
			if v.CanSet() {
				v.Set(reflect.ValueOf(Val{}))
			}
			return
		}
		for i := 0; i < v.NumField(); i++ {
			if v.Type().Field(i).IsExported() {
				uxr12NP(v.Field(i))
			}
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			uxr12NP(v.Index(i))
		}
	case reflect.Map:
		for _, k := range v.MapKeys() {
			e := reflect.New(v.Type().Elem()).Elem()
			e.Set(v.MapIndex(k))
			uxr12NP(e)
			v.SetMapIndex(k, e)
		}
	}
}

// uxr12States are the 2 Server states the release adapter reaches, as design screens: idle (S0),
// first setup (H6), start (H15), failure (H7), serving (H8, slow H9, q once H8Q), stopping (HQ)
// and the stopped run summary (H12); plus the unload stop in progress («UNLOADING», ctrl+f).
var uxr12States = []string{"S0", "H6", "H15", "H7", "H8", "H9", "H8Q", "HQ", "H12", "unloading"}

// uxr12Reach renders a 2 Server state with every value not ported.
func uxr12Reach(t *testing.T, id, lang string, w, h int) *harness {
	t.Helper()
	sc := scenarios[id]
	if id == "unloading" {
		sc = scenario{setup: func(f *FakeBackend) { f.StopHold = true }, keys: serving("ctrl+f")}
	}
	f := NewFake()
	if sc.setup != nil {
		sc.setup(f)
	}
	uxr12NP(reflect.ValueOf(f))
	for i := range f.StartEvs { // a step not ported in the setup's and the start's chain
		f.StartEvs[i].Steps = append(f.StartEvs[i].Steps, Step{Name: "verify", State: StepNotPorted})
	}
	hs := newH(t, f, lang, w, h)
	hs.keys(sc.keys...)
	hs.beat()
	if id != "unloading" {
		hs.want(id)
	}
	return hs
}

func TestUXR12NotPortedServer(t *testing.T) {
	words := []string{"not ported", "미지원"}
	// on: where a not-ported value is drawn with the switch on (the release adapter's places)
	on := map[string]bool{"H6": true, "H15": true, "H8": true, "H9": true, "H8Q": true, "HQ": true, "H12": true, "unloading": true}
	for _, id := range uxr12States {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range [][2]int{{160, 45}, {120, 30}, {80, 28}} {
				withNotPorted(t, false)
				p := uxr12Reach(t, id, lang, sz[0], sz[1]).plain()
				if os.Getenv("UXR12_SHOW") != "" {
					t.Logf("%s %s %dx%d off\n%s", id, lang, sz[0], sz[1], p)
				}
				for _, w := range words {
					if strings.Contains(p, w) {
						t.Fatalf("%s %s %dx%d, switch off: «%s» drawn:\n%s", id, lang, sz[0], sz[1], w, p)
					}
				}
				withNotPorted(t, true)
				p = uxr12Reach(t, id, lang, sz[0], sz[1]).plain()
				want := words[0]
				if lang == LangKorean {
					want = words[1]
				}
				if on[id] && sz[0] == 160 && !strings.Contains(p, want) {
					t.Fatalf("%s %s %dx%d, switch on: «%s» is not back:\n%s", id, lang, sz[0], sz[1], want, p)
				}
				withNotPorted(t, false)
			}
		}
	}
}

// TestUXR12HeroSeam: the latent seam UX-R9 named (its landing 4.1): a hero too narrow for the
// unit beside its big digits drew the number as text («1:12»), a word the 80x28 capture does not
// have (S8-LA-09's rule). Now the digits stay big with the unit under them, wherever they fit.
func TestUXR12HeroSeam(t *testing.T) {
	r := rc{W: 64}
	for _, w := range []int{22, 24} {
		out := r.heroL("7", "", "1:12", hiB, "h:mm", "up")(w)
		for _, wd := range words(out) {
			if wd == "1:12" {
				t.Fatalf("hero %d wide: the number as text:\n%s", w, out)
			}
		}
		if !strings.Contains(out, "h:mm") {
			t.Fatalf("hero %d wide: no unit:\n%s", w, out)
		}
	}
	// the run summary at 64x24 (hero 22 wide): every word is a word, or the prefix of one, of
	// its 80x28 capture
	for _, lang := range []string{LangEnglish, LangKorean} {
		ref := words(reach(t, "H12", lang, 80, 28).m.View())
		out := reach(t, "H12", lang, 64, 24).m.View()
		for _, wd := range words(out) {
			ok := false
			for _, rw := range ref {
				if strings.HasPrefix(rw, wd) {
					ok = true
					break
				}
			}
			if !ok {
				t.Fatalf("%s H12 64x24: word %q is not in the 80x28 capture:\n%s", lang, wd, out)
			}
		}
	}
}

// TestUXR12CacheWord: the sub-lead's addendum (UX-R7 landing 4.3). The settings word «n GB cache»
// is translated by its pattern for any budget, not by the one key «12 GB cache»: «캐시 n GB».
func TestUXR12CacheWord(t *testing.T) {
	for _, n := range []string{"20", "14.5", "12"} {
		for _, lang := range []string{LangEnglish, LangKorean} {
			f := NewFake()
			for i := range f.Statuses {
				f.Statuses[i].SettingsWords = []string{"12k context", n + " GB cache", "virtual"}
			}
			hs := newH(t, f, lang, 160, 45)
			hs.keys(serving()...)
			hs.beat()
			p := hs.plain()
			want, lack := n+" GB cache", "캐시 "+n+" GB"
			if lang == LangKorean {
				want, lack = lack, want
			}
			if !strings.Contains(p, want) || strings.Contains(p, lack) {
				t.Fatalf("%s budget %s GB: want «%s», not «%s»:\n%s", lang, n, want, lack, p)
			}
		}
	}
}

// TestUXR12TemplateWords: the sub-lead's addendum (1) (UX-R10 landing 4, finding 18). The wire's
// notes of a model outside the catalog read in UX-R10's template form, «outside the catalog ·
// <arch> template settings», and the screens draw them in the screen's language: the SETUP tile
// and the 이 모델 box of 1 Models (the setup note) and the settings summary (the preset note).
func TestUXR12TemplateWords(t *testing.T) {
	const id = "gpt-oss-120b-local"
	setup := "outside the catalog · gpt-oss template settings · ⏎ shows its setup plan (the template repack)"
	preset := "outside the catalog · gpt-oss template settings"
	for _, c := range []struct{ lang, want, hint string }{
		{LangEnglish, "outside the catalog · gpt-oss template settings", "⏎ shows its setup plan (the template repack)"},
		{LangKorean, "카탈로그 밖 · gpt-oss 템플릿 설정", "⏎ 로 준비 계획 보기 (템플릿 리팩)"},
	} {
		r := rc{W: 160, H: 45, lang: c.lang}
		if got, ok := r.outsideNote(setup); !ok || got != c.want+" · "+c.hint {
			t.Fatalf("%s setup note: %q %v", c.lang, got, ok)
		}
		if got, ok := r.outsideNote("virtual repack present"); ok || got != "virtual repack present" {
			t.Fatalf("%s another note: %q %v", c.lang, got, ok)
		}
		f := NewFake()
		for i := range f.Cat.Models {
			if f.Cat.Models[i].ID == id {
				f.Cat.Models[i].SetupNote = setup
			}
		}
		v := f.Views[id]
		v.Preset = preset
		f.Views[id] = v
		hs := newH(t, f, c.lang, 160, 45)
		hs.keys("down", "down", "down", "down")
		hs.beat()
		p := hs.plain()
		flat := strings.Join(strings.Fields(p), " ")
		if !strings.Contains(flat, strings.Join(strings.Fields(c.want), " ")) || strings.Contains(p, "structure defaults") {
			t.Fatalf("%s 1 Models: no «%s»:\n%s", c.lang, c.want, p)
		}
		if c.lang == LangKorean && strings.Contains(p, "template settings") {
			t.Fatalf("Korean 1 Models shows the English template words:\n%s", p)
		}
		row, _ := hs.m.rowByID(id)
		if s := hs.m.settingsSummary(row); !strings.HasPrefix(s, c.want) {
			t.Fatalf("%s settings summary: %q", c.lang, s)
		}
	}
}

// TestUXR12LabelSpace: the sub-lead's addendum (2) (UX-R10 landing 4). Every row label of the
// 1 Models settings box and 이 모델 box keeps at least one space before its value, in both
// languages: «repack mode» (11 cells) no longer runs into «virtual».
func TestUXR12LabelSpace(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, NewFake(), lang, 160, 45)
		hs.beat()
		m := hs.m
		row, _ := m.selRow()
		labels := []string{"quant", "file check", "folder", "setup", "decode", "prefill", "last"}
		for _, x := range m.views[row.ID].Rows {
			labels = append(labels, x.Label)
		}
		p := hs.plain()
		for _, l := range labels {
			tl := m.r.T(l)
			if !strings.Contains(p, tl) {
				continue // a row not drawn here
			}
			for _, line := range strings.Split(p, "\n") {
				for i := strings.Index(line, tl); i >= 0; {
					after := line[i+len(tl):]
					// a row label: only marks (· ● ◆ ▶) and spaces between the box edge and it
					atRow := strings.HasSuffix(strings.TrimRight(line[:i], " ·●◆▶"), "│")
					if atRow && after != "" && !strings.HasPrefix(after, " ") && !strings.HasPrefix(after, "│") {
						t.Fatalf("%s: the label %q runs into its value: %q", lang, tl, line)
					}
					j := strings.Index(after, tl)
					if j < 0 {
						break
					}
					i += len(tl) + j
				}
			}
		}
		mustHave(t, p, m.r.T("repack mode")+" ")
	}
}
