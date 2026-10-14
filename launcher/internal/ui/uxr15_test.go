package ui

// UX-R15 (version 8; the user's acceptance finding 25, reviews/_user_8_LAUNCHER-REVIEW_11.md,
// screenshot 153428): while a server runs, the model list of 1 Models is locked (the user's
// decision, review item 1), and the notice area above the boxes says so and how to unload,
// also when the settings differ from the running server and the not-applied line stands.

import (
	"fmt"
	"strings"
	"testing"
)

// TestUXR15LockNotice: 1 Models through the fake backend with a server up, with and without a
// changed setting, in English and Korean, at 120x40 and 80x28. The lock line stands in the
// notice area above the boxes, together with the not-applied line when a setting differs;
// the arrow keys do not move the selection; the boxes stay whole (UX-R13).
func TestUXR15LockNotice(t *testing.T) {
	want := map[string]string{
		LangEnglish: "server running · model choice locked · to unload: 2 Server, ctrl+f",
		LangKorean:  "서버 실행 중 · 모델 선택 잠김 · 모델을 내리려면 2 Server ctrl+f",
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			for _, changed := range []bool{false, true} {
				name := fmt.Sprintf("%s %dx%d changed=%v", lang, sz[0], sz[1], changed)
				hs := newH(t, NewFake(), lang, sz[0], sz[1])
				hs.keys("enter")
				hs.want("H8")
				hs.keys("1", "down")
				if hs.m.sel != 0 {
					t.Fatalf("%s: the list moved while serving: sel %d", name, hs.m.sel)
				}
				if changed {
					hs.keys("tab", "right", "tab") // a setting differs from the running server: H2
					hs.want("H2")
				} else {
					hs.want("H1")
				}
				out := hs.plain()
				ls := strings.Split(out, "\n")
				if len(ls) > sz[1] {
					t.Fatalf("%s: %d lines for a window of %d", name, len(ls), sz[1])
				}
				bs, ok := uxr13Boxes(t, name, ls, hs.m.r.T("MODEL"))
				if !ok {
					t.Fatalf("%s: no box row:\n%s", name, out)
				}
				at := func(s string) int {
					for i, l := range ls[:bs[0].top] {
						if strings.Contains(l, s) {
							return i
						}
					}
					return -1
				}
				lock := at(want[lang])
				if lock < 0 {
					t.Fatalf("%s: no lock line %q above the boxes:\n%s", name, want[lang], out)
				}
				na := at(hs.m.r.T("not applied to the running server · applies at the next start"))
				if changed && na < 0 {
					t.Fatalf("%s: the not-applied line left:\n%s", name, out)
				}
				if !changed && na >= 0 {
					t.Fatalf("%s: a not-applied line without a change:\n%s", name, out)
				}
				if lang == LangEnglish && sz == [2]int{80, 28} && changed {
					t.Logf("%s:\n%s", name, out)
				}
			}
		}
	}
}
