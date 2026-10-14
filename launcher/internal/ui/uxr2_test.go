// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// uxr2Widths: the window of the user's screenshot 053150 (about 120 columns)
// and the narrowest full window (80); both 30 rows high.
var uxr2Widths = []int{120, 80}

// uxr2Notices are n start notices of the backend's kind (UX-R2 test data).
func uxr2Notices(n int) []string {
	var ns []string
	for i := 0; i < n; i++ {
		ns = append(ns, fmt.Sprintf("notice %d: start settings derived for the gpt-oss structure; the performance of these settings is unverified", i+1))
	}
	return ns
}

// uxr2Reach draws 1 Models of the fake with a long preset (the screenshot's
// outside-the-catalog model) and n notices.
func uxr2Reach(t *testing.T, lang string, w, n int) *harness {
	t.Helper()
	f := NewFake()
	v := f.Views["qwen35-122b"]
	v.Preset = "outside the catalog · gpt-oss template settings" // UX-R12 addendum: the template form
	f.Views["qwen35-122b"] = v
	f.Pre.Notices = uxr2Notices(n)
	hs := newH(t, f, lang, w, 30)
	hs.beat()
	hs.want("H1")
	return hs
}

// TestUXR2StartLine (UX-R2, review items 2 and 16(a)): the start line of
// 1 Models is the start key and the model, nothing more: no preset, value or
// provenance is listed (they stand in the settings box), so none is cut at
// the window edge, and the drawn line is not wider than the window. The test
// model has a long preset (the screenshot's outside-the-catalog model), so a
// listed summary would show.
func TestUXR2StartLine(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, w := range uxr2Widths {
			for _, n := range []int{0, 2, 6} {
				name := fmt.Sprintf("%s %d cols %d notices", lang, w, n)
				hs := uxr2Reach(t, lang, w, n)
				row, _ := hs.m.selRow()
				lbl := "⏎ " + hs.m.r.T("START")
				var got string
				for _, l := range strings.Split(hs.m.View(), "\n") {
					if strings.Contains(ansi.Strip(l), lbl) {
						got = l
						break
					}
				}
				if got == "" {
					t.Fatalf("%s: no start line", name)
				}
				if lw := lipgloss.Width(got); lw > w {
					t.Fatalf("%s: the start line is %d wide in a window of %d", name, lw, w)
				}
				_, text, _ := strings.Cut(ansi.Strip(got), lbl+"  ")
				if strings.TrimRight(text, " ") != row.Name {
					t.Fatalf("%s: the start line reads %q after the start key, want the model %q alone", name, text, row.Name)
				}
				if s := hs.m.settingsSummary(row); s == "" || strings.Contains(ansi.Strip(got), s) {
					t.Fatalf("%s: the test data has no settings summary or the start line lists it (%q)", name, s)
				}
				t.Logf("%s: %s", name, strings.TrimSpace(ansi.Strip(got)))
			}
		}
	}
}

// TestUXR2NoticeArea (UX-R2, review 2 item 3): the notice area of 1 Models
// has a fixed height, so the top row of the three boxes is on the same line
// for 0, 2 and 6 notices; notices that do not fit fold into one line that
// counts them and points to 3 Memory.
func TestUXR2NoticeArea(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, w := range uxr2Widths {
			top := -1
			for _, n := range []int{0, 2, 6} {
				name := fmt.Sprintf("%s %d cols %d notices", lang, w, n)
				hs := uxr2Reach(t, lang, w, n)
				box := "╭─ " + hs.m.r.T("MODEL")
				ls := strings.Split(hs.plain(), "\n")
				at := -1
				for i, l := range ls {
					if strings.Contains(l, box) {
						at = i
						break
					}
				}
				if at < 0 {
					t.Fatalf("%s: no box row", name)
				}
				if top < 0 {
					top = at
				}
				if at != top {
					t.Fatalf("%s: the box top row is line %d, with 0 notices line %d", name, at, top)
				}
				folded := regexp.MustCompile(`▲ \+\d+ · ` + regexp.QuoteMeta(hs.m.r.T("details on 3 Memory"))).MatchString(hs.plain())
				if n == 6 && !folded {
					t.Fatalf("%s: 6 notices and no fold line", name)
				}
				t.Logf("%s: box top row at line %d, fold line %v", name, at, folded)
			}
		}
	}
}
