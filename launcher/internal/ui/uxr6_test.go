// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// uxr6Basis is the wire's words for a catalog profile's range (wire basisCatalog).
const uxr6Basis = "range = this model's catalog limits"

// uxr6Reach draws 1 Models of the fake with the wire's form of the ranges: the
// ctx and budget rows carry their range as the note and say what bounds it.
func uxr6Reach(t *testing.T, lang string, w int, saved bool) *harness {
	t.Helper()
	f := NewFake()
	v := f.Views["qwen35-122b"]
	v.Saved = saved
	v.Rows = append([]SettingRow(nil), v.Rows...)
	for i := range v.Rows {
		switch v.Rows[i].Key {
		case "ctx":
			v.Rows[i].Note, v.Rows[i].Basis = "2k ~ 128k", uxr6Basis
		case "budget":
			v.Rows[i].Note, v.Rows[i].Basis = "8 GB ~ 20 GB", uxr6Basis
		}
	}
	f.Views["qwen35-122b"] = v
	hs := newH(t, f, lang, w, 30)
	hs.want("H1")
	return hs
}

// uxr6Panel is the settings box as drawn, its lines inside the frame and, flat,
// its text with the wrapped lines joined (a wrapped sentence reads whole).
func uxr6Panel(hs *harness) (lines []string, flat string) {
	row, _ := hs.m.selRow()
	w := hs.m.r.cols(3)[1]
	if !hs.m.r.wide() {
		w = hs.m.r.cols(2)[1]
	}
	for _, l := range strings.Split(ansi.Strip(hs.m.setPanel(w, row, hs.m.focusSet, false, 0, 0)), "\n")[1:] {
		l = strings.TrimSpace(strings.Trim(strings.TrimSpace(l), "│╰╯─"))
		lines = append(lines, l)
	}
	return lines, strings.Join(strings.Fields(strings.Join(lines, " ")), " ")
}

// uxr6Closed: the frame draws the bottom edge of the boxes (the added lines do
// not push the boxes past the window).
func uxr6Closed(t *testing.T, name string, hs *harness) {
	t.Helper()
	p := hs.plain()
	for _, l := range strings.Split(p, "\n") {
		if strings.Count(l, "╰") >= 2 {
			return
		}
	}
	t.Fatalf("%s: the boxes are cut at the window's foot:\n%s", name, p)
}

func uxr6Line(lines []string, label string) string {
	for _, l := range lines {
		if strings.Contains(l, label) {
			return l
		}
	}
	return ""
}

// TestUXR6Labels (UX-R6, the user's finding 12): in the settings box of 1 Models
// (1) the yellow range says what it is («range 2k ~ 128k») and the box says what
// bounds it («range = this model's catalog limits»); (2) a changed value names
// its earlier value as the model's default or its saved value, not «was»; (3)
// «not saved» is explained: the change is for this run only and not in the
// model's settings, ctrl+s saves it. In English and Korean, in windows of 80
// (two columns), 110, 120 and 160 columns; the boxes stay whole in each.
func TestUXR6Labels(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, w := range []int{80, 110, 120, 160} {
			for _, saved := range []bool{false, true} {
				name := fmt.Sprintf("%s %d cols saved=%v", lang, w, saved)
				hs := uxr6Reach(t, lang, w, saved)
				T := hs.m.r.T
				ctx, earlier := T("context"), T("default")
				if saved {
					earlier = T("saved")
				}
				unsaved := T("not saved = this run only, not in this model's settings")

				// (1) no focus, no change: the range and its basis
				lines, flat := uxr6Panel(hs)
				// the word goes with the numbers where the row has room for it (UX-R10: the
				// row leads with a state mark, so in English it needs a box over 35 cells
				// inside): the line under the rows says what bounds it in every width
				want := T("range") + " 2k ~ 128k"
				pw := hs.m.r.cols(3)[1]
				if !hs.m.r.wide() {
					pw = hs.m.r.cols(2)[1]
				}
				if lipgloss.Width("   "+pad(ctx, 11)+"12288  "+want) > pw-4 {
					want = "12288  2k ~ 128k"
				}
				if l := uxr6Line(lines, ctx); !strings.Contains(l, want) {
					t.Fatalf("%s: the ctx row reads %q, want «%s 2k ~ 128k»\n%s", name, l, T("range"), strings.Join(lines, "\n"))
				}
				if !strings.Contains(flat, T(uxr6Basis)) {
					t.Fatalf("%s: the box lacks the basis %q:\n%s", name, T(uxr6Basis), strings.Join(lines, "\n"))
				}
				// the frame: the range (UX-R10: where the box scrolls, its rows give two cells to
				// the scroll bar and the word may go)
				if !strings.Contains(hs.plain(), "12288  2k ~ 128k") && !strings.Contains(hs.plain(), want) {
					t.Fatalf("%s: the frame lacks the ctx range", name)
				}
				uxr6Closed(t, name+", unchanged", hs)
				t.Logf("%s, unchanged:\n%s", name, strings.Join(lines, "\n"))

				// (2)+(3) the ctx value changed, the row focused
				hs.keys("tab", "down", "right")
				lines, flat = uxr6Panel(hs)
				l := uxr6Line(lines, ctx)
				if !strings.Contains(l, "◀ 16384 ▶") {
					t.Fatalf("%s: the focused ctx row reads %q", name, l)
				}
				named := strings.Contains(l, earlier+" 12288")
				arrow := strings.Contains(l, "← 12288") && strings.Contains(flat, "← "+earlier+" · "+unsaved)
				if !named && !arrow {
					t.Fatalf("%s: the changed row does not name its earlier value as %q: %q\n%s", name, earlier, l, strings.Join(lines, "\n"))
				}
				if !strings.Contains(flat, unsaved) {
					t.Fatalf("%s: the box does not say what «not saved» means:\n%s", name, strings.Join(lines, "\n"))
				}
				uxr6Closed(t, name+", ctx changed and focused", hs)
				t.Logf("%s, ctx changed and focused:\n%s", name, strings.Join(lines, "\n"))

				// back on the model list: the change stays, the box says how to save it
				hs.keys("tab")
				lines, flat = uxr6Panel(hs)
				l = uxr6Line(lines, ctx)
				named = strings.Contains(l, "16384  "+earlier+" 12288")
				arrow = strings.Contains(l, "16384  ← 12288") && strings.Contains(flat, "← "+earlier+" · "+unsaved)
				if !named && !arrow {
					t.Fatalf("%s: the changed row does not name its earlier value as %q: %q\n%s", name, earlier, l, strings.Join(lines, "\n"))
				}
				if !strings.Contains(flat, unsaved+" ctrl+s "+T("saves the changed values")) {
					t.Fatalf("%s: the box lacks the not-saved line and ctrl+s:\n%s", name, strings.Join(lines, "\n"))
				}
				p := hs.plain()
				for _, bad := range []string{"was 12288", "12288 was", "  was "} {
					if strings.Contains(p, bad) {
						t.Fatalf("%s: the frame still reads %q:\n%s", name, bad, p)
					}
				}
				uxr6Closed(t, name+", ctx changed, list focused", hs)
				t.Logf("%s, ctx changed, list focused:\n%s", name, strings.Join(lines, "\n"))
			}
		}
	}
}
