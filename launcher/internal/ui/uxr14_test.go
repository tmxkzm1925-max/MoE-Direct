package ui

// UX-R14 (version 8; the user's acceptance finding 22, reviews/_user_8_LAUNCHER-REVIEW_11.md,
// screenshots 151302 and 151431): the first-run setup page shows the whole setup until the
// server starts: one percent over the repack, the repacker's check (verify) and the start's
// speed check; a row per phase, with its bar or a turning spinner while it runs and ✓ when
// done; the STEP tile counts the phases.

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/x/ansi"
)

// uxr14Events is a setup as the wire reports it, phase by phase: the repack's layers, the
// check's start line (no count yet), its pairs, the setup's end, then the start with its sweep
// step (the speed check) running and done.
func uxr14Events() []StartEvent {
	set := func(name string, done, total Val) StartEvent {
		return StartEvent{Model: "qwen35-122b", Elapsed: 10 * time.Second,
			Setup: &SetupProgress{Name: name, Done: done, Total: total, DiskMBs: K(900), LastS: U}}
	}
	start := func(sweep StepState) StartEvent {
		next := StepTodo
		if sweep == StepDone {
			next = StepRunning
		}
		return StartEvent{Model: "qwen35-122b", Elapsed: time.Second, Steps: []Step{{Name: "bundle check", State: StepDone},
			{Name: "queue depth", State: sweep}, {Name: "start server", State: next}}}
	}
	return []StartEvent{
		set("repack", K(12), K(36)),
		set("repack", K(36), K(36)),
		set("check", U, U),
		set("check", K(6912), K(13824)),
		set("done", U, U),
		start(StepRunning),
		start(StepDone),
	}
}

// uxr14Pct reads the big digits of the percent tile, the first tile of the row under the page
// head (rows of the font in style.go, three cells a digit and a space between).
func uxr14Pct(t *testing.T, name string, ls []string) int {
	t.Helper()
	top := -1
	for i, l := range ls {
		if strings.Contains(l, "╭") {
			top = i
			break
		}
	}
	if top < 0 || top+3 >= len(ls) {
		t.Fatalf("%s: no tile row", name)
	}
	rows := [3][]rune{[]rune(ls[top+1]), []rune(ls[top+2]), []rune(ls[top+3])}
	s := ""
	for at := 3; at+3 <= len(rows[0]) && at+3 <= len(rows[1]) && at+3 <= len(rows[2]); at += 4 {
		got := [3]string{string(rows[0][at : at+3]), string(rows[1][at : at+3]), string(rows[2][at : at+3])}
		d := ""
		for c, g := range font {
			if c >= '0' && c <= '9' && g == got {
				d = string(c)
			}
		}
		if d == "" {
			break
		}
		s += d
	}
	var p int
	if _, err := fmt.Sscan(s, &p); err != nil {
		t.Fatalf("%s: the percent tile reads %q:\n%s", name, s, strings.Join(ls, "\n"))
	}
	return p
}

func uxr14Line(ls []string, s string) string {
	for _, l := range ls {
		if strings.Contains(l, s) {
			return l
		}
	}
	return ""
}

// TestUXR14SetupPhases: the fake backend's setup fed through repack 100 %, verify and the
// speed check, in English and Korean at 120x40 and 80x28. At each phase: the page is the setup
// page; the percent tile rises (never falls) and stays below 100 until the speed check ends;
// the phase rows show ✓ for a done phase, the running phase with its bar (the repack's layers,
// verify's pairs) or a spinner that turns on the screen tick, and ○ for what follows; the STEP
// tile reads «n of 3» (never «0 of 0»). When the sweep step is done the start page follows.
func TestUXR14SetupPhases(t *testing.T) {
	evs := uxr14Events()
	type want struct {
		ph               int    // the running phase
		repCount, verPct string // the repack's count, verify's percent, "" for none
		spin             bool   // the running phase has the spinner
	}
	wants := []want{
		{0, "12 / 36", "", false},
		{0, "36 / 36", "", false},
		{1, "36 / 36", "", true},
		{1, "36 / 36", " 50 %", false},
		{2, "36 / 36", "", true},
		{2, "36 / 36", "", true},
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			last := -1
			for k := range evs {
				name := fmt.Sprintf("%s %dx%d event %d", lang, sz[0], sz[1], k+1)
				f := NewFake()
				f.StartHold = true
				f.StartEvs = append([]StartEvent(nil), evs[:k+1]...)
				hs := newH(t, f, lang, sz[0], sz[1])
				T := hs.m.r.T
				hs.keys("enter")
				if k == len(evs)-1 {
					hs.want("H15") // the speed check ended: the start page
					continue
				}
				hs.want("H6")
				out := hs.plain()
				ls := strings.Split(out, "\n")
				if len(ls) > sz[1] {
					t.Fatalf("%s: %d lines for %d", name, len(ls), sz[1])
				}
				w := wants[k]
				p := uxr14Pct(t, name, ls)
				if p < last || p >= 100 {
					t.Fatalf("%s: the percent reads %d after %d (rising, below 100):\n%s", name, p, last, out)
				}
				last = p
				if !strings.Contains(ls[0], fmt.Sprintf("%d%%", p)) && !strings.Contains(ls[0], fmt.Sprintf("%d %%", p)) {
					t.Fatalf("%s: the status line does not read %d %%: %q", name, p, ls[0])
				}
				// the STEP tile counts the phases
				if !strings.Contains(out, fmt.Sprintf("%d %s 3", w.ph+1, T("of"))) || strings.Contains(out, "0 "+T("of")+" 0") {
					t.Fatalf("%s: the STEP tile does not read %d of 3:\n%s", name, w.ph+1, out)
				}
				// the phase rows
				names := []string{T("repack"), T("verify"), T("speed check")}
				doing := []string{T("repack"), T("verifying …"), T("checking the drive speed …")}
				for i := range names {
					var l string
					switch {
					case i < w.ph:
						l = uxr14Line(ls, "✓ "+names[i])
					case i > w.ph:
						l = uxr14Line(ls, "○ "+names[i])
					default:
						l = uxr14Line(ls, "▶ "+doing[i])
					}
					if l == "" {
						t.Fatalf("%s: no row for phase %d (%s):\n%s", name, i, names[i], out)
					}
				}
				rep := uxr14Line(ls, map[bool]string{true: "▶ ", false: "✓ "}[w.ph == 0]+T("repack"))
				if !strings.Contains(rep, w.repCount+" "+T("layers")) || !strings.Contains(rep, "█") {
					t.Fatalf("%s: the repack row lacks its bar and %q: %q", name, w.repCount, rep)
				}
				run := uxr14Line(ls, "▶ "+doing[w.ph])
				if w.verPct != "" && (!strings.Contains(run, w.verPct) || !strings.Contains(run, "█")) {
					t.Fatalf("%s: verify's bar and %q missing: %q", name, w.verPct, run)
				}
				if w.spin {
					a := strings.TrimRight(run, " ")
					s0 := a[len(a)-1:]
					if !strings.Contains(`|/-\`, s0) {
						t.Fatalf("%s: no spinner at the end of %q", name, run)
					}
					hs.advance(pollEvery)
					hs.beat()
					b := strings.TrimRight(uxr14Line(strings.Split(hs.plain(), "\n"), "▶ "+doing[w.ph]), " ")
					if s1 := b[len(b)-1:]; s1 == s0 || !strings.Contains(`|/-\`, s1) {
						t.Fatalf("%s: the spinner did not turn on the tick: %q then %q", name, s0, s1)
					}
				}
				if lang == LangEnglish && sz == [2]int{80, 28} && (k == 2 || k == 3 || k == 5) {
					t.Logf("%s:\n%s", name, ansi.Strip(out))
				}
			}
		}
	}
}
