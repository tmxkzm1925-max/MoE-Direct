package ui

// GL-07 (version 8): the screens' part of S8-LA-14 and S8-LA-15 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition rows 2.3 and 3.5; decision
// ③, DECISION_LEDGER.md:4179): the load bar and the cache-save bar per tick on the harness's
// fake clock.

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// startRow is the step list's row of a step, where the bar's text stands (the tiles above it
// share a line between panels, «││»; the step list is one panel).
func startRow(p, step string) string {
	for _, l := range strings.Split(p, "\n") {
		if strings.Contains(l, step) && !strings.Contains(l, "││") {
			return l
		}
	}
	return ""
}

// S8-LA-14: per tick, a first start shows elapsed seconds only; with a last duration D the fill
// is elapsed / D; at or past D it holds at 99 % with «slower than last time»; 100 % only at
// ready, and it stays full after ready; before the server child starts the bar has not begun.
func TestS8LA14(t *testing.T) {
	for _, c := range []struct {
		name string
		last Val
	}{{"first start", U}, {"last duration 20 s", K(20)}} {
		f := NewFake()
		f.StartHold = true
		ev := f.StartEvs[1] // «load model» running
		pre := f.StartEvs[0]
		pre.LastS, pre.Elapsed, pre.LoadRunning = c.last, 0, false
		f.StartEvs = []StartEvent{pre}
		hs := newH(t, f, LangEnglish, 120, 30)
		hs.keys("enter")
		hs.want("H15")
		hs.advance(3 * time.Second) // the steps before the child: the bar has not begun
		if row := startRow(hs.plain(), "start server"); strings.Contains(row, "100") || (c.last.Ok() && !strings.Contains(row, " 0 %")) {
			t.Errorf("%s: before the child starts: %q", c.name, row)
		}

		ev.LastS, ev.Elapsed, ev.LoadRunning = c.last, 0, true
		hs.m.startEv, hs.m.startAt = &ev, hs.clock
		for s := 0; s <= 26; s++ {
			p := hs.plain()
			row := startRow(p, "load model")
			switch {
			case !c.last.Ok():
				if !strings.Contains(p, "first time: seconds only") || strings.Contains(row, "%") ||
					!strings.Contains(row, fmt.Sprintf("%d s", s)) || !strings.Contains(row, "first time") {
					t.Errorf("%s, %d s: %q", c.name, s, row)
				}
			case s < 20:
				want := fmt.Sprintf(" %d %%", s*100/20)
				if !strings.Contains(row, want) || strings.Contains(p, "slower than last time") {
					t.Errorf("%s, %d s: want%q in %q", c.name, s, want, row)
				}
			default:
				if !strings.Contains(row, "99 % · slower than last time") {
					t.Errorf("%s, %d s: %q", c.name, s, row)
				}
			}
			if strings.Contains(row, "100 %") {
				t.Errorf("%s, %d s: 100 %% before ready: %q", c.name, s, row)
			}
			hs.advance(time.Second)
		}

		// ready: full, and full through the warmup that follows
		done := ev
		done.LoadRunning, done.LoadReady, done.LoadTook, done.Elapsed = false, true, 26*time.Second, 0
		done.Steps = append([]Step(nil), ev.Steps...)
		done.Steps[2].State, done.Steps[3].State = StepDone, StepRunning
		hs.m.startEv, hs.m.startAt = &done, hs.clock
		for i := 0; i < 3; i++ {
			if row := startRow(hs.plain(), "ready"); !strings.Contains(row, " 100 %") || !strings.Contains(row, "26 s") {
				t.Errorf("%s: after ready (+%d s): %q", c.name, i*5, row)
			}
			hs.advance(5 * time.Second)
		}
	}
	// the overrun note in Korean
	f := NewFake()
	f.StartHold = true
	f.StartEvs = f.StartEvs[1:2]
	f.StartEvs[0].Elapsed = 20 * time.Second
	hs := newH(t, f, LangKorean, 120, 30)
	hs.keys("enter")
	if !strings.Contains(hs.plain(), "99 % · 지난번보다 느림") {
		t.Errorf("Korean overrun at D:\n%s", hs.plain())
	}
}

// S8-LA-15: the cache-save bar from the quit action to the save-completion response: a first
// save shows elapsed seconds only; with an expectation the fill is elapsed / expectation; at or
// past it 99 % with «slower than last time»; 100 % only at the response.
func TestS8LA15(t *testing.T) {
	for _, c := range []struct {
		name   string
		expect Val
	}{{"first save", U}, {"expectation 12 s", K(12)}} {
		f := NewFake()
		idleServer(f)
		f.StopHold = true
		evs := f.StopEvs[StopQuit]
		evs[0].Save = SaveProgress{State: StepRunning, Elapsed: 0, LastS: c.expect, Tokens: K(2000)}
		evs[0].Steps[0] = Step{Name: "save prompt cache", State: StepRunning}
		evs[0].Steps[1].State = StepTodo
		hs := newH(t, f, LangEnglish, 120, 30)
		hs.keys("enter", "q", "q")
		hs.want("HQ")
		hs.m.stopAt = hs.clock
		for s := 0; s <= 15; s++ {
			p := hs.plain()
			row := startRow(p, "save prompt cache")
			switch {
			case !c.expect.Ok():
				if !strings.Contains(row, "first time") || strings.Contains(row, "%") {
					t.Errorf("%s, %d s: %q", c.name, s, row)
				}
			case s < 12:
				if want := fmt.Sprintf(" %d %%", s*100/12); !strings.Contains(row, want) || strings.Contains(row, "slower") {
					t.Errorf("%s, %d s: want%q in %q", c.name, s, want, row)
				}
			default:
				if !strings.Contains(row, "99 % · slower than last time") {
					t.Errorf("%s, %d s: %q", c.name, s, row)
				}
			}
			if strings.Contains(row, "100 %") || !strings.Contains(p, "2.0k") && !strings.Contains(p, "2000") && !strings.Contains(p, "2k") {
				t.Errorf("%s, %d s: %q (tokens shown?)", c.name, s, row)
			}
			hs.advance(time.Second)
		}
		// the completion response: the save step done, the bar full in the cache tile
		ev := *hs.m.stopEv
		ev.Save = SaveProgress{State: StepDone, Elapsed: 16 * time.Second, LastS: c.expect, Tokens: K(1990)}
		ev.Steps = append([]Step(nil), ev.Steps...)
		ev.Steps[0].State, ev.Steps[1].State = StepDone, StepRunning
		hs.m.stopEv, hs.m.stopAt = &ev, hs.clock
		sp := saveBar(ev.Save, 0)
		if !sp.Ready || sp.Pct != 100 || sp.Frac != 1 {
			t.Errorf("%s: the answered save's bar %+v", c.name, sp)
		}
		if p := hs.plain(); strings.Contains(p, "not ported") || strings.Contains(p, "next start is cold") {
			t.Errorf("%s: an answered save shows the not-ported state:\n%s", c.name, p)
		}
	}
	// no save reported: the not-ported state, never a bar
	if sp := saveBar(SaveProgress{State: StepNotPorted}, 5*time.Second); sp.Ready || sp.Pct != 0 {
		t.Errorf("not ported: %+v", sp)
	}
}
