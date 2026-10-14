// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// la09Screens: every place screen (1 Models H1, 2 Server S0, 3 Memory H14,
// 4 Log H11, 5 Settings H5), the serving screen H8 and the stopped page H12
// (plan S8-LA-09); H7, the failed start, is added because it is the screen
// that holds a log tail.
var la09Screens = []string{"H1", "S0", "H14", "H11", "H5", "H8", "H12", "H7"}

// la09Sizes: the reference size, then the three sizes below it.
var la09Sizes = [][2]int{{80, 28}, {79, 27}, {60, 20}, {40, 12}}

// words splits a capture into its words: box-drawing and block glyphs
// (U+2500-U+259F, the frames, rules and bars) separate words and are none.
func words(s string) []string {
	return strings.FieldsFunc(ansi.Strip(s), func(c rune) bool {
		return c == ' ' || c == '\n' || c == '\t' || (c >= 0x2500 && c <= 0x259F)
	})
}

// TestS8LA09 (plan S8-LA-09; decision ⑤, U-31 L:4264; edition 2.5, 10.4).
func TestS8LA09(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, id := range la09Screens {
			var ref []string
			var refParts, prev Parts
			for i, sz := range la09Sizes {
				w, h := sz[0], sz[1]
				hs := reach(t, id, lang, w, h)
				hs.want(id)
				out := hs.m.View()
				parts := hs.m.Parts()
				name := fmt.Sprintf("%s %s %dx%d", lang, id, w, h)
				if strings.Contains(out, "\x00") {
					t.Fatalf("%s: a part marker reached the screen", name)
				}
				ls := strings.Split(out, "\n")
				if len(ls) > h {
					t.Fatalf("%s: %d lines for a window of %d", name, len(ls), h)
				}
				nonEmpty := 0
				for j, l := range ls {
					if lw := lipgloss.Width(l); lw > w {
						t.Fatalf("%s: line %d is %d wide: %q", name, j, lw, ansi.Strip(l))
					}
					if strings.TrimSpace(ansi.Strip(l)) != "" {
						nonEmpty++
					}
				}
				if nonEmpty < 2 {
					t.Fatalf("%s: %d non-empty lines (a single make-larger line is not allowed)", name, nonEmpty)
				}
				if i == 0 {
					ref, refParts, prev = words(out), parts, parts
					if !parts.Has("status") {
						t.Fatalf("%s: no status line at the reference size: %v", name, parts)
					}
					continue
				}
				// every word is a word, or the prefix of a word, of the 80x28 capture
				for _, wd := range words(out) {
					ok := false
					for _, rw := range ref {
						if strings.HasPrefix(rw, wd) {
							ok = true
							break
						}
					}
					if !ok {
						t.Fatalf("%s: word %q is not in the 80x28 capture", name, wd)
					}
				}
				// the parts leave in the user's order and do not come back
				if !parts.Has("status") || (refParts.Has("verdict") && !parts.Has("verdict")) {
					t.Fatalf("%s: the status line or the start verdict left: %v (80x28: %v)", name, parts, refParts)
				}
				for _, p := range parts {
					if !prev.Has(p) {
						t.Fatalf("%s: part %q came back (bigger size: %v, here: %v)", name, p, prev, parts)
					}
				}
				order := []string{"graph", "log", "help"}
				for a := 0; a < len(order); a++ {
					for b := a + 1; b < len(order); b++ {
						if parts.Has(order[a]) && refParts.Has(order[b]) && !parts.Has(order[b]) {
							t.Fatalf("%s: %s stayed while %s left: %v", name, order[a], order[b], parts)
						}
					}
				}
				prev = parts
				if lang == LangEnglish {
					t.Logf("%s: %d lines, parts %v", name, len(ls), parts)
				}
			}
		}
	}
	// the place-row fixes the cell carries hold at every size: the mark under
	// the lit place starts after the separator, at its pill (review item 25),
	// and the pill's letters are colour 16, bold (review item 26)
	for _, sz := range append([][2]int{{120, 30}}, la09Sizes...) {
		for i, p := range places {
			hs := newH(t, NewFake(), LangEnglish, sz[0], sz[1])
			hs.keys(p[0])
			ls := strings.Split(hs.plain(), "\n")
			if strings.Trim(ls[1], "─━") != "" {
				continue // the rule has left this small window (the verdict stays longer)
			}
			pillAt := cellIdx(ls[0], p[0]+" "+p[1]) - 1
			markAt := cellIdx(ls[1], "━")
			if pillAt < 0 || pillAt != markAt {
				t.Fatalf("%dx%d place %d: pill at %d, mark at %d\n%s\n%s", sz[0], sz[1], i+1, pillAt, markAt, ls[0], ls[1])
			}
		}
	}
	ps := navPillStyle("14")
	if ps.GetForeground() != lipgloss.Color("16") || !ps.GetBold() {
		t.Fatalf("lit place: foreground %v bold %v, want colour 16 bold", ps.GetForeground(), ps.GetBold())
	}
	t.Logf("place row: the mark starts at the lit pill at 120x30, 80x28, 79x27, 60x20, 40x12; the pill is colour 16, bold")
}
