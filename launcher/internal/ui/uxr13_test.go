package ui

// UX-R13 (version 8; the user's acceptance finding 21, reviews/_user_8_LAUNCHER-REVIEW_10.md,
// screenshot 145325, with the deferred «1 Models model list has no scroll» of UX-R10): on 1
// Models the box row is drawn to the height the window leaves for it, every box whole with
// its top and bottom edge; the model list scrolls so that the selected row stays in view, the
// settings box keeps its UX-R10 scroll, the 이 모델 box cuts its last rows; where no frame fits
// (top edge, one row, bottom edge) the row leaves as one part in the U-31 order.

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// uxr13Fake is the fake backend with extra models outside the catalog after its five.
func uxr13Fake(extra int) *FakeBackend {
	f := NewFake()
	for i := 0; i < extra; i++ {
		f.Cat.Models = append(f.Cat.Models, ModelRow{ID: fmt.Sprintf("extra-%02d", i), Name: fmt.Sprintf("Extra-%02d", i),
			Present: true, Path: fmt.Sprintf(`D:\moe-models\extra-%02d.gguf`, i), SizeGB: K(10), FileCheck: FileNoReference,
			NeedGB: U, FreeRAMGB: K(24.6), LeftGB: U, Fit: FitUnmeasured, Setup: SetupReady})
	}
	return f
}

// uxr13Cells splits a plain line into its cells: a wide rune fills two, the second a 0.
func uxr13Cells(l string) []rune {
	var cs []rune
	for _, c := range l {
		cs = append(cs, c)
		for i := 1; i < lipgloss.Width(string(c)); i++ {
			cs = append(cs, 0)
		}
	}
	return cs
}

func uxr13At(cs []rune, i int) rune {
	if i < 0 || i >= len(cs) {
		return ' '
	}
	return cs[i]
}

// uxr13Box is one box of the row as found on the screen: its edge columns, its top and bottom
// lines (bottom -1: no bottom edge on the screen) and its inside lines.
type uxr13Box struct {
	c0, c1, top, bottom int
	inside              []string
}

// uxr13Boxes finds the box row under the line that holds the list's title and follows each box
// down its side edges to its bottom edge. ok is false where no box row is on the screen.
func uxr13Boxes(t *testing.T, name string, ls []string, title string) (bs []uxr13Box, ok bool) {
	t.Helper()
	top := -1
	for i, l := range ls {
		if strings.Contains(l, "╭─ "+title) {
			top = i
		}
	}
	if top < 0 {
		return nil, false
	}
	cs := uxr13Cells(ls[top])
	for c := 0; c < len(cs); c++ {
		if cs[c] != '╭' {
			continue
		}
		e := c + 1
		for e < len(cs) && cs[e] != '╮' {
			e++
		}
		if e >= len(cs) {
			t.Fatalf("%s: a box's top edge has no right corner on line %d: %q", name, top, ls[top])
		}
		b := uxr13Box{c0: c, c1: e, top: top, bottom: -1}
		for i := top + 1; i < len(ls); i++ {
			ci := uxr13Cells(ls[i])
			l, r := uxr13At(ci, c), uxr13At(ci, e)
			if l == '╰' && r == '╯' {
				b.bottom = i
				break
			}
			if l != '│' || r != '│' {
				t.Fatalf("%s: box %d (cells %d-%d) is cut on screen line %d (%q %q): %q", name, len(bs), c, e, i, l, r, ls[i])
			}
			b.inside = append(b.inside, strings.ReplaceAll(string(ci[c+1:e]), "\x00", ""))
		}
		if b.bottom < 0 {
			t.Fatalf("%s: box %d (cells %d-%d) has no bottom edge on the screen", name, len(bs), c, e)
		}
		bs = append(bs, b)
		c = e
	}
	return bs, true
}

// TestUXR13BoxesWhole (finding 21; the deferred list scroll, personal ledger M:200): 1 Models
// through the fake backend at the user's size 110x14 and at 80x28, 120x40, 60x20 and 40x12,
// with the fake's five models and with more models than the room, the selection at the top
// and at the bottom of the list, in English and Korean. Per box: the frame's top and bottom
// edge are on the screen, no row of a box is cut by the screen (both side edges on every
// line between), all boxes of the row end on one line, above the footer. The selected model's
// row is in the list box; a list that does not show all its models shows the scroll bar.
// The screen keeps to its height and width. Where no frame fits (40x12), the row has left as
// a whole: no box edge of it is on the screen.
func TestUXR13BoxesWhole(t *testing.T) {
	sizes := [][2]int{{110, 14}, {80, 28}, {120, 40}, {60, 20}, {40, 12}}
	gone := map[[2]int]bool{{40, 12}: true} // the room is below minBox: the row leaves
	cases := 0
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, extra := range []int{0, 30} {
			for _, bottom := range []bool{false, true} {
				for _, sz := range sizes {
					w, h := sz[0], sz[1]
					name := fmt.Sprintf("%s %d models %dx%d sel-%v", lang, 5+extra, w, h, map[bool]string{false: "top", true: "bottom"}[bottom])
					hs := newH(t, uxr13Fake(extra), lang, w, h)
					n := len(hs.m.cat.Models)
					if bottom {
						for i := 0; i < n; i++ {
							hs.keys("down")
						}
					}
					want := 0
					if bottom {
						want = n - 1
					}
					if hs.m.sel != want {
						t.Fatalf("%s: selection at %d, want %d", name, hs.m.sel, want)
					}
					sel, _ := hs.m.selRow()
					out := hs.plain()
					ls := strings.Split(out, "\n")
					if len(ls) > h {
						t.Fatalf("%s: %d lines for a window of %d", name, len(ls), h)
					}
					for i, l := range ls {
						if lw := lipgloss.Width(l); lw > w {
							t.Fatalf("%s: line %d is %d wide", name, i, lw)
						}
					}
					bs, ok := uxr13Boxes(t, name, ls, hs.m.r.T("MODEL"))
					cases++
					if gone[sz] {
						if ok || strings.ContainsAny(strings.Join(ls[4:], "\n"), "╭╮╰╯") {
							t.Fatalf("%s: the box row should have left as a whole:\n%s", name, out)
						}
						continue
					}
					if !ok {
						t.Fatalf("%s: no box row:\n%s", name, out)
					}
					nb := 2
					if hs.m.r.wide() {
						nb = 3
					}
					if len(bs) != nb {
						t.Fatalf("%s: %d boxes, want %d:\n%s", name, len(bs), nb, out)
					}
					foot := len(ls)
					for i, l := range ls {
						if i > bs[0].top && strings.Trim(l, "─") == "" && l != "" {
							foot = i // the footer's rule
							break
						}
					}
					for i, b := range bs {
						if b.bottom != bs[0].bottom || b.bottom >= foot {
							t.Fatalf("%s: box %d ends on line %d (box 0: %d, footer rule %d):\n%s", name, i, b.bottom, bs[0].bottom, foot, out)
						}
					}
					// the selected model's row is in the list box
					nm := []rune(sel.Name)
					nm = nm[:min(len(nm), 10)]
					found := false
					for _, l := range bs[0].inside {
						if strings.Contains(l, "▶") && strings.Contains(l, string(nm)) {
							found = true
						}
					}
					if !found {
						t.Fatalf("%s: the selected model %q is not in the list box:\n%s", name, sel.Name, out)
					}
					// a list that leaves models out shows its scroll bar in the rows' last cell
					shown := 0
					for _, x := range hs.m.cat.Models {
						p := []rune(x.Name)
						p = p[:min(len(p), 10)]
						for _, l := range bs[0].inside {
							if strings.Contains(l, string(p)) {
								shown++
								break
							}
						}
					}
					if shown < n {
						bar := false
						for _, l := range bs[0].inside {
							cs := []rune(strings.TrimRight(l, " "))
							if len(cs) > 0 && (cs[len(cs)-1] == '█' || cs[len(cs)-1] == '░') {
								bar = true
							}
						}
						if !bar {
							t.Fatalf("%s: %d of %d models shown and no scroll bar:\n%s", name, shown, n, out)
						}
					}
					if sz == [2]int{110, 14} && lang == LangEnglish && extra == 0 && !bottom {
						t.Logf("%s:\n%s", name, ansi.Strip(out))
					}
				}
			}
		}
	}
	t.Logf("%d screens: every box whole with its top and bottom edge, the selected model in view; the row gone as a whole at 40x12", cases)
}
