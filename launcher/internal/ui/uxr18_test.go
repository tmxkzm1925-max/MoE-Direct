package ui

// UX-R18 (version 8; the user's acceptance finding 28 with answer 8,
// reviews/_user_8_LAUNCHER-REVIEW_11.md, screenshots 155129 and 155318): idle, the request box of
// 2 Server showed «처리 중인 요청 없음» and «마지막 답변 7.9 tok/s» in one brightness, read as
// one; busy, it showed four rows. The user chose one layout for both states: the four rows with
// their labels always, values and marks only while a request runs, and the title by state. This
// test renders 2 Server through the fake backend idle before any request, busy (prefill and
// decode) and idle after a completed request, in English and Korean at 120x40 and 80x28, and
// reads the box by its cells. UX-R27 (review item 37) replaced the box with the DECODE and
// PREFILL statistics panels: the one layout holds there, the same titles, column names and row
// names in every state, now «—» while its phase does not run and a figure while it runs.

import (
	"strings"
	"testing"
)

func TestUXR18OneLayout(t *testing.T) {
	type state struct {
		name string
		set  func(s *ServeStatus)
		busy bool
	}
	states := []state{
		{"idle before any request", func(s *ServeStatus) {
			s.Phase, s.LastDecodeTokS, s.TTFTs, s.DecodeTokS = ReqNone, Val{}, Val{}, Val{}
		}, false},
		{"busy prefill", func(s *ServeStatus) {
			s.Phase, s.PrefillDone, s.PrefillTotal, s.PrefillTokS = ReqPrefill, K(3000), K(6000), K(300)
		}, true},
		{"busy decode", func(s *ServeStatus) { s.Phase = ReqDecode }, true},
		{"idle after a request", func(s *ServeStatus) { s.Phase = ReqNone }, false},
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			for _, st := range states {
				f := NewFake()
				st.set(&f.Statuses[0])
				hs := newH(t, f, lang, sz[0], sz[1])
				hs.keys("enter")
				hs.beat()
				p := hs.plain()
				ph := f.Statuses[0].Phase
				dec, pre := uxr27Box(p, tr(lang, "DECODE · last 60 min")), uxr27Box(p, tr(lang, "PREFILL · last 60 min"))
				if len(dec) < 2 || len(pre) < 3 || dec[0] != "now avg min max" || pre[0] != dec[0] {
					t.Fatalf("%s %dx%d %s: the two panels with their column names are not drawn:\n%s", lang, sz[0], sz[1], st.name, p)
				}
				rows := []string{dec[1], pre[1], pre[2]}
				for i, name := range []string{"decode", "prefill", "TTFT"} {
					v := strings.Fields(rows[i])
					runs := ph == ReqDecode && i == 0 || ph == ReqPrefill && i > 0
					if len(v) < 5 || v[0] != tr(lang, name) || (v[1] == "—") == runs {
						t.Fatalf("%s %dx%d %s: row %q, want %q with now «—» unless its phase runs (%v)", lang, sz[0], sz[1], st.name, rows[i], name, runs)
					}
				}
				if st.busy != (ph != ReqNone) {
					t.Fatalf("%s: state and phase disagree", st.name)
				}
				for _, gone := range []string{"no request in flight", "처리 중인 요청 없음", tr(lang, "last reply")} {
					if strings.Contains(strings.Join(append(dec, pre...), "\n"), gone) {
						t.Fatalf("%s %dx%d %s: the panels hold %q: %q %q", lang, sz[0], sz[1], st.name, gone, dec, pre)
					}
				}
			}
		}
	}
}
