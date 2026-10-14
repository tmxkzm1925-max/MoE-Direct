package ui

// UX-R20 (version 8; the user's acceptance items 31 (a), (c) and (d),
// reviews/_user_8_LAUNCHER-REVIEW_12.md, screenshots 185129 and 185445, the user's answers of
// 19:10 KST): the 1 Models settings box title names the selected model by one rule for every
// model, saved or not, at every width: the name without its quant part (the 이 모델 box shows
// the quant), the tail leaving first where the box is narrow and the name cut last (UX-R13);
// a name taken from a file name (DeepSeek-V4.1, no spaces) is no exception. The line under the
// rows «· default» stands only where it says what the title does not.

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
)

// uxr20Title is the settings box title on the screen: the name after «SETTINGS · » and the
// rest of the title after the name.
func uxr20Title(t *testing.T, hs *harness) (name, rest string) {
	t.Helper()
	T := hs.m.r.T
	re := regexp.MustCompile("─ " + regexp.QuoteMeta(T("SETTINGS")) + " · (.*?)((?: · .*?)?) ─")
	for _, l := range strings.Split(hs.plain(), "\n") {
		if m := re.FindStringSubmatch(l); m != nil {
			return m[1], m[2]
		}
	}
	t.Fatalf("no settings title with a name:\n%s", hs.plain())
	return "", ""
}

// uxr20Name: got is base, or base cut with or without the cut mark, at least 4 cells.
func uxr20Name(t *testing.T, what, got, base, quant string) {
	t.Helper()
	g := strings.TrimSuffix(got, "…")
	if len(g) < 4 || !strings.HasPrefix(base, g) || strings.Contains(got, quant) {
		t.Fatalf("%s: the title's name is %q, want %q or its start (no %q)", what, got, base, quant)
	}
}

// uxr20Legend: a box line that holds «· default» alone.
func uxr20Legend(hs *harness) bool {
	want := "· " + hs.m.r.T("default")
	for _, l := range strings.Split(hs.plain(), "\n") {
		for _, seg := range strings.Split(l, "│") {
			if strings.TrimSpace(seg) == want {
				return true
			}
		}
	}
	return false
}

// TestUXR20Title (items 31 (b)-(d)): a catalog name with its quant and a file name without
// spaces each show their name without the quant (cut where the box is narrow, never gone) in
// the default, changed and saved states, the saved title with «my settings» (the user's
// saved marker, see below), in Korean and English at 120x40 and 80x28, and at every width from 81 to 160
// (the user saw the name come and go with the window's width).
func TestUXR20Title(t *testing.T) {
	sizes := [][2]int{{120, 40}, {80, 28}}
	for w := 81; w <= 160; w++ {
		sizes = append(sizes, [2]int{w, 30})
	}
	states := map[string]int{}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range sizes {
			what := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
			f := NewFake()
			f.Cat.Models[0].Name = "Qwen3.5-122B-A10B Q4_K_M (execution scope, NextN excluded)"
			hs := newH(t, f, lang, sz[0], sz[1])
			T := hs.m.r.T
			n0, r0 := uxr20Title(t, hs)
			uxr20Name(t, what+" default", n0, "Qwen3.5-122B-A10B", "Q4_K_M")
			hs.keys("tab", "right")
			n1, r1 := uxr20Title(t, hs)
			uxr20Name(t, what+" changed", n1, "Qwen3.5-122B-A10B", "Q4_K_M")
			hs.keys("ctrl+s")
			n2, r2 := uxr20Title(t, hs)
			uxr20Name(t, what+" saved", n2, "Qwen3.5-122B-A10B", "Q4_K_M")
			// the saved marker stays at the brief's sizes and in Korean at every width; in
			// English it leaves where it would cut the name below setNameMin cells
			if !strings.Contains(r2, T("my settings")) {
				if sz[1] != 30 || lang == LangKorean {
					t.Fatalf("%s: the saved title lacks «%s»: %q %q", what, T("my settings"), n2, r2)
				}
				if n2 != "Qwen3.5-122B-A10B" {
					t.Fatalf("%s: the saved marker left before the name was whole: %q %q", what, n2, r2)
				}
			}
			states[fmt.Sprintf("%s %q%q | %q%q | %q%q", lang, n0, r0, n1, r1, n2, r2)]++
			if sz[1] != 30 {
				t.Logf("%s: %q%q | %q%q | %q%q", what, n0, r0, n1, r1, n2, r2)
			}

			// DeepSeek-V4.1 as the user's folder shows it: a file name, outside the catalog,
			// unsupported, with no settings rows
			f = NewFake()
			f.Cat.Models = append([]ModelRow{{ID: `D:\m\DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf`,
				Name: "DeepSeek-V4.1-Flash-MXFP4-engram", Present: true, Quant: "MXFP4_MOE", Path: `D:\m\x.gguf`,
				SizeGB: K(467), NeedGB: U, FreeRAMGB: K(24.6), LeftGB: U, Fit: FitUnsupported}}, f.Cat.Models...)
			hs = newH(t, f, lang, sz[0], sz[1])
			n, r := uxr20Title(t, hs)
			uxr20Name(t, what+" DeepSeek-V4.1", n, "DeepSeek-V4.1-Flash", "MXFP4")
			states[fmt.Sprintf("%s DeepSeek-V4.1 %q%q", lang, n, r)]++
			if sz[1] != 30 {
				t.Logf("%s DeepSeek-V4.1: %q%q", what, n, r)
			}
		}
	}
	for k, n := range states { // the titles over the sizes 81-160 and their counts
		t.Logf("%3d × %s", n, k)
	}
}

// TestUXR20Legend (item 31 (a)): where the title says «default» and every row is at its
// default, the line under the rows does not repeat «· default»; where the title does not say
// it, or another mark shows, the line stands; at 120x40, 80x28 and every width from 81 to 160.
func TestUXR20Legend(t *testing.T) {
	sizes := [][2]int{{120, 40}, {80, 28}}
	for w := 81; w <= 160; w++ {
		sizes = append(sizes, [2]int{w, 30})
	}
	tails := map[string]int{}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range sizes {
			what := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
			hs := newH(t, NewFake(), lang, sz[0], sz[1])
			T := hs.m.r.T
			_, rest := uxr20Title(t, hs)
			says := strings.Contains(rest, T("default"))
			if says == uxr20Legend(hs) {
				t.Fatalf("%s: title tail %q, line «· %s» shown %v:\n%s", what, rest, T("default"), uxr20Legend(hs), hs.plain())
			}
			hs.keys("tab", "right")
			mustHave(t, hs.plain(), "◆ "+T("not saved"))
			tails[lang+" "+rest]++
			if sz[1] != 30 {
				t.Logf("%s: title tail %q", what, rest)
			}
		}
	}
	t.Logf("title tails over all sizes: %v", tails)
}
