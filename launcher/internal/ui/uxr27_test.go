package ui

// UX-R27 (version 8; the user's decision on item 37 of 19:19 KST, reviews/_user_8_LAUNCHER-REVIEW_12.md
// section «항목 37»): the request box (UX-R18) and the prefill box (UX-R19) read as two gauges,
// one whole and one partial. Both became statistics panels of one shape, the game-FPS way: DECODE
// «now avg min max» in tok/s; PREFILL the same in tok/s and a TTFT line in s; over the graphs'
// last 60 minutes; now «—» when its phase does not run; titles that do not switch. The graph
// footer «average · lowest» went. The PREFILL panel stays where the prefill graph row has no room
// (alone and short under the DECODE panel, or beside it without graphs). This test renders 2
// Server through the fake backend, decoding, reading a prompt and idle, in English and Korean, at
// 120x40, 120x30 and 80x28.

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// uxr27Page serves, takes decode readings of 5.6, 4.0 and 6.0 tok/s five minutes apart, reports
// three finished requests (prompt 20, 30, 40 tok/s; TTFT 4, 2, 6 s) and ends in the state given:
// "decode" (live 5.2 tok/s, the prompt rate left in the status), "prefill" (a prompt read at 250 tok/s, 1000 of 4000) or "idle" (the
// last live reading left in the status).
func uxr27Page(t *testing.T, lang string, w, h int, state string) *harness {
	t.Helper()
	f := NewFake()
	hs := newH(t, f, lang, w, h)
	hs.keys("enter")
	hs.beat()
	st := *hs.m.status
	for _, v := range []float64{4.0, 6.0} {
		hs.advance(5 * time.Minute)
		st.Phase, st.DecodeTokS = ReqDecode, K(v)
		hs.send(statusMsg{st, nil})
	}
	now := hs.clock
	q := func(task int, ago time.Duration, pre, ttft float64) ReqSample {
		return ReqSample{Task: task, At: now.Add(-ago), PrefillTokS: pre, TTFTs: ttft}
	}
	st.Reqs = []ReqSample{q(3, 9*time.Minute, 20, 4), q(4, 5*time.Minute, 30, 2), q(5, time.Minute, 40, 6)}
	switch state {
	case "decode":
		st.Phase, st.DecodeTokS, st.PrefillTokS = ReqDecode, K(5.2), K(250) // the prompt rate left in the status
	case "prefill":
		st.Phase, st.DecodeTokS, st.PrefillTokS, st.PrefillDone, st.PrefillTotal = ReqPrefill, U, K(250), K(1000), K(4000)
	default:
		st.Phase, st.DecodeTokS = ReqNone, K(5.2) // a stale reading: now stays «—» while no answer runs
	}
	hs.send(statusMsg{st, nil})
	hs.want("H8")
	return hs
}

// uxr27Box returns the rows of the box whose top edge holds the title, plain and trimmed, or nil.
func uxr27Box(p, title string) []string {
	ls := strings.Split(p, "\n")
	for i, l := range ls {
		j := strings.Index(l, "╭─ "+title+" ")
		if j < 0 {
			continue
		}
		c := lipgloss.Width(l[:j])
		e := c + lipgloss.Width(l[j:j+strings.Index(l[j:], "╮")+len("╮")])
		var rows []string
		for _, x := range ls[i+1:] {
			if strings.HasPrefix(strings.TrimSpace(cutCells(x, c, c+1)), "╰") {
				return rows
			}
			rows = append(rows, strings.Join(strings.Fields(cutCells(x, c+1, e-1)), " "))
		}
	}
	return nil
}

func TestUXR27StatsPanels(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		decT, preT := tr(lang, "DECODE · last 60 min"), tr(lang, "PREFILL · last 60 min")
		for _, sz := range [][2]int{{120, 40}, {120, 30}, {80, 28}} {
			for _, state := range []string{"decode", "prefill", "idle"} {
				t.Run(fmt.Sprintf("%s_%dx%d_%s", lang, sz[0], sz[1], state), func(t *testing.T) {
					hs := uxr27Page(t, lang, sz[0], sz[1], state)
					p := hs.plain()
					ls := strings.Split(p, "\n")
					fail := func(what string) {
						t.Helper()
						t.Fatalf("%s %s %dx%d: %s:\n%s", lang, state, sz[0], sz[1], what, p)
					}
					if len(ls) != sz[1] || !strings.Contains(p, "/health ok") {
						fail("the page is not whole")
					}
					for _, l := range ls {
						if lipgloss.Width(l) > sz[0] {
							fail("a line wider than the window")
						}
					}
					dn, pn, tn := "—", "—", "—"
					switch state {
					case "decode":
						dn = "5.2"
					case "prefill":
						pn, tn = "250", "~12"
					}
					// avg of DECODE: the mean of the decode graph's samples (all inside the 60 minutes)
					sum, n := 0.0, 0
					for _, x := range hs.m.decodeSeries() {
						sum, n = sum+x.v, n+1
					}
					da := f1(sum / float64(n))
					want := map[string][][2]string{
						decT: {{"now avg min max", ""}, {fmt.Sprintf("decode %s %s 4.0 6.0", dn, da), " tok/s"}},
						preT: {{"now avg min max", ""}, {fmt.Sprintf("prefill %s 30.0 20.0 40.0", pn), " tok/s"}, {fmt.Sprintf("TTFT %s 4.0 2.0 6.0", tn), " s"}},
					}
					for title, wr := range want {
						// each row ends with its unit at these sizes
						var rows []string
						for _, x := range wr {
							rows = append(rows, x[0]+x[1])
						}
						box := uxr27Box(p, title)
						var got []string
						for _, x := range box {
							if x != "" {
								got = append(got, x)
							}
						}
						if g := strings.Join(got, " | "); g != strings.Join(rows, " | ") {
							fail(fmt.Sprintf("panel «%s» reads «%s», want «%s»", title, strings.Join(got, " | "), strings.Join(rows, " | ")))
						}
					}
					// gone: the request box with its switching title, the prefill box, the graph footer
					for _, old := range []string{tr(lang, "ANSWERING"), tr(lang, "IDLE") + " ·", "╭─ " + tr(lang, "PREFILL · TTFT") + " ─",
						fmt.Sprintf(tr(lang, "average %s · lowest %s tok/s"), "5.2", "4.0")} {
						if strings.Contains(p, old) {
							fail("«" + old + "» still shows")
						}
					}
					// the graphs where the room is: both at 120x40, the decode graph at 120x30, none at 80
					dg, pg := strings.Contains(p, tr(lang, "DECODE · tok/s · last 60 min")), strings.Contains(p, tr(lang, "PREFILL · tok/s per request"))
					if dg != (sz[0] >= 110) || pg != (sz[1] >= 40) {
						fail(fmt.Sprintf("graphs decode %v prefill %v", dg, pg))
					}
					// the reused count is in the PROCESSING tile (UX-R26)
					if tile := uxr19Tile(p, tr(lang, "PROCESSING · this request"), tr(lang, "PROCESSING")); tile == nil ||
						!strings.Contains(strings.Join(tile, " "), tr(lang, "reused")+" 3.1k") {
						fail("no reused count in the PROCESSING tile")
					}
				})
			}
		}
	}
}
