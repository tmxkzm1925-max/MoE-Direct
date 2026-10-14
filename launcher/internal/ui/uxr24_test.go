package ui

// UX-R24 (version 8; the user's acceptance item 35 (a), reviews/_user_8_LAUNCHER-REVIEW_12.md,
// screenshot 190333): on 2 Server the tiles and graph titles named their boxes in capitals
// («DECODE», «PREFILL») while the prefill tile and box read «prefill · TTFT» (Korean), the request
// box «answering · this request» and the settings box «applied settings, change in Models»
// (English). Now every box name of the page reads in capitals in one form: the name in capitals,
// a unit or a qualifier after « · » (or the settings box's «, ») as it was; Korean words are
// unchanged and «decode», «prefill», «TTFT» stay English (UX-KR decision (5)). This test renders
// 2 Server through the fake backend (answering in decode and in prefill, idle, slow, and the run
// summary) in English and Korean at 120x40 and 80x28. UX-R27: the request box and the prefill box
// became the panels «DECODE · last 60 min» and «PREFILL · last 60 min», at every size.

import (
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

var uxr24Title = regexp.MustCompile(`╭─ (.*?) ─`)

// uxr24Page renders 2 Server in one state.
func uxr24Page(t *testing.T, lang, state string, w, h int) string {
	t.Helper()
	if state == "summary" {
		return uxr23Summary(t, lang, w, h, true).plain()
	}
	f := NewFake()
	switch state {
	case "prefill":
		f.Statuses[0].Phase = ReqPrefill
	case "idle":
		idleServer(f)
	case "slow":
		f.Statuses[0].Slow, f.Statuses[0].UsualTokS, f.Statuses[0].LastDecodeTokS = true, K(5.5), K(3.9)
	}
	hs := newH(t, f, lang, w, h)
	hs.keys("enter")
	hs.beat()
	return hs.plain()
}

func TestUXR24BoxNames(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			for _, state := range []string{"decode", "prefill", "idle", "slow", "summary"} {
				p := uxr24Page(t, lang, state, sz[0], sz[1])
				ls := strings.Split(p, "\n")
				fail := func(what string) {
					t.Helper()
					t.Fatalf("%s %s %dx%d: %s:\n%s", lang, state, sz[0], sz[1], what, p)
				}
				if len(ls) != sz[1] {
					fail("the page is not whole")
				}
				var names []string
				for _, l := range ls {
					if lipgloss.Width(l) > sz[0] {
						fail("a line wider than the window")
					}
					for _, m := range uxr24Title.FindAllStringSubmatch(l, -1) {
						names = append(names, m[1])
					}
				}
				if len(names) < 3 {
					fail("fewer than three box names")
				}
				// the name, up to its first qualifier, holds no lower-case letter
				for _, n := range names {
					name := n
					for _, sep := range []string{" · ", ", "} {
						if i := strings.Index(name, sep); i >= 0 {
							name = name[:i]
						}
					}
					if strings.ToUpper(name) != name {
						fail("box name «" + n + "» is not in capitals")
					}
				}
				want := map[string][]string{
					"decode":  {tr(lang, "APPLIED SETTINGS, change in Models")},
					"prefill": {},
					"idle":    {},
					"slow":    {},
				}[state]
				if state != "summary" {
					want = append(want, tr(lang, "DECODE · last 60 min"), tr(lang, "PREFILL · last 60 min"))
				}
				for _, w := range want {
					if !strings.Contains(p, "╭─ "+w+" ─") {
						fail("no box named «" + w + "» in full")
					}
				}
			}
		}
	}
}
