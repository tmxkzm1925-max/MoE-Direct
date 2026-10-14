package ui

// GL-15 (version 8): the screens' part of warm start, S8-LA-17 of the plan
// reviews/_plan_8_STAGE1_v8-plansplit.md (d1dbe526f; F-LS edition row 2.6: the H15 clock time is
// removed, «warm start» only), and the stop page once the warm-start save is ported (a stop that
// stores nothing, a failed save).

import (
	"regexp"
	"strings"
	"testing"
)

// S8-LA-17: the warm-start line of the start steps reads «warm start» and holds no clock time or
// date: the count of \d{1,2}:\d{2} on it is 0, in both languages; a cold start has no such line.
func TestS8LA17(t *testing.T) {
	clock := regexp.MustCompile(`\d{1,2}:\d{2}`)
	date := regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)
	for _, c := range []struct {
		lang, word string
	}{{LangEnglish, "warm start"}, {LangKorean, "warm start"}} {
		for _, warm := range []bool{true, false} {
			f := NewFake()
			f.StartHold = true
			ev := f.StartEvs[1]
			ev.Warm = warm
			f.StartEvs = []StartEvent{ev}
			hs := newH(t, f, c.lang, 120, 30)
			hs.keys("enter")
			hs.want("H15")
			var lines []string
			for _, l := range strings.Split(hs.plain(), "\n") {
				if strings.Contains(l, c.word) {
					lines = append(lines, l)
				}
			}
			if !warm {
				if len(lines) != 0 {
					t.Errorf("%s cold start: %q", c.lang, lines)
				}
				continue
			}
			if len(lines) == 0 {
				t.Fatalf("%s: no warm-start line", c.lang)
			}
			for _, l := range lines {
				if n := len(clock.FindAllString(l, -1)); n != 0 || date.MatchString(l) {
					t.Errorf("%s: the warm-start line holds a clock time or date (%d): %q", c.lang, n, l)
				}
			}
		}
	}
}

// GL-15: the stop page when the stop stores nothing (warm start off, ctrl+c twice) and when the
// save fails: the cache tile says so, the save row has no bar, and the alert says the next start
// is cold; it never says the save is not ported.
func TestGL15StopSave(t *testing.T) {
	for _, c := range []struct {
		state      StepState
		tile, line string
	}{
		{StepSkipped, "save: off", "no slot save at this stop: the next start is cold"},
		{StepFailed, "save: failed", "the slot save failed: the next start is cold"},
	} {
		f := NewFake()
		idleServer(f)
		f.StopHold = true
		evs := f.StopEvs[StopQuit]
		evs[0].Save = SaveProgress{State: c.state, LastS: U, Tokens: U}
		evs[0].Steps[0] = Step{Name: "save prompt cache", State: c.state, Detail: "x"}
		evs[0].Steps[1].State = StepRunning
		hs := newH(t, f, LangEnglish, 120, 30)
		hs.keys("enter", "q", "q")
		hs.want("HQ")
		p := hs.plain()
		if !strings.Contains(p, c.tile) || !strings.Contains(p, c.line) || strings.Contains(p, "not ported") {
			t.Errorf("%v:\n%s", c.state, p)
		}
		if row := startRow(p, "save prompt cache"); strings.Contains(row, "%") || strings.Contains(row, "first time") {
			t.Errorf("%v: the save row has a bar: %q", c.state, row)
		}
	}
}
