package wire

// GL-07 (version 8): the adapter's part of S8-LA-14 and S8-LA-15 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition rows 2.3 and 3.5; decision
// ③, DECISION_LEDGER.md:4179): the load bar's span and the cache-save bar reach the screen.

import (
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// S8-LA-14: the load bar runs from the server child's start to /health ready, against the last
// duration of the same model and settings; after ready it stays full through the warmup.
func TestS8LA14(t *testing.T) {
	t0 := time.Unix(100, 0)
	for _, c := range []struct {
		name    string
		hasLast bool
	}{{"first start", false}, {"with a last duration", true}} {
		tr := newStartTrack("m")
		pre := &core.PreflightResult{HasLastReady: c.hasLast, LastReady: 20 * time.Second, Config: &core.Config{Host: "127.0.0.1", Port: 8093}}
		ev := tr.apply(core.StartEvent{Step: core.StepSizing, State: core.StepDone, Preflight: pre}, t0)
		if ev.LoadRunning || ev.LoadReady || ev.LastS.Ok() != c.hasLast || (c.hasLast && ev.LastS.V != 20) {
			t.Errorf("%s: before the child starts %+v", c.name, ev)
		}
		ev = tr.apply(core.StartEvent{Step: core.StepSpawn, State: core.StepDone}, t0)
		if ev.LoadRunning {
			t.Errorf("%s: the bar runs before the ready wait", c.name)
		}
		for _, el := range []time.Duration{500 * time.Millisecond, 10 * time.Second, 25 * time.Second} {
			ev = tr.apply(core.StartEvent{Step: core.StepReady, State: core.StepRunning, Elapsed: el}, t0)
			if !ev.LoadRunning || ev.LoadReady || ev.Elapsed != el {
				t.Errorf("%s: tick %v %+v", c.name, el, ev)
			}
		}
		ev = tr.apply(core.StartEvent{Step: core.StepReady, State: core.StepDone, Elapsed: 26 * time.Second}, t0)
		if ev.LoadRunning || !ev.LoadReady || ev.LoadTook != 26*time.Second {
			t.Errorf("%s: ready %+v", c.name, ev)
		}
		ev = tr.apply(core.StartEvent{Step: core.StepWarmup, State: core.StepRunning}, t0)
		if !ev.LoadReady || ev.LoadTook != 26*time.Second {
			t.Errorf("%s: the bar left full after ready: %+v", c.name, ev)
		}
		ev = tr.apply(core.StartEvent{Final: true, Ready: true, Status: core.StatusOK}, t0)
		if !ev.Ready || !ev.LoadReady {
			t.Errorf("%s: final %+v", c.name, ev)
		}
	}
}

// S8-LA-15: before the stop reports a save the bar waits (GL-15: the save is ported, so the
// interim not-ported state of GL-07 is gone); a stop that reports a save shows its bar (elapsed
// since the quit action, the expectation, the tokens), done at the completion response.
func TestS8LA15(t *testing.T) {
	tr := newStopTrack(ui.StopQuit)
	if e := tr.event(0, ui.U); e.Save.State != ui.StepTodo || e.Steps[0].State != ui.StepTodo {
		t.Fatalf("no save yet: %+v", e.Save)
	}
	tr.saveFrom(&core.CacheSaveTrack{Tokens: 1000, Elapsed: 2 * time.Second})
	e := tr.event(2*time.Second, ui.U)
	if e.Save.State != ui.StepRunning || e.Save.LastS.Ok() || e.Save.Tokens.V != 1000 || e.Save.Elapsed != 2*time.Second ||
		e.Steps[0].State != ui.StepRunning || e.Steps[0].Detail != "" {
		t.Errorf("first save running: %+v %+v", e.Save, e.Steps[0])
	}
	tr.saveFrom(&core.CacheSaveTrack{Tokens: 2000, Elapsed: 9 * time.Second, Expect: 12 * time.Second, HasExpect: true})
	e = tr.event(9*time.Second, ui.U)
	if e.Save.State != ui.StepRunning || !e.Save.LastS.Ok() || e.Save.LastS.V != 12 || e.Save.Tokens.V != 2000 {
		t.Errorf("second save running: %+v", e.Save)
	}
	tr.saveFrom(&core.CacheSaveTrack{Tokens: 2000, Elapsed: 15 * time.Second, Expect: 12 * time.Second, HasExpect: true, Done: true, Saved: 1990})
	e = tr.event(15*time.Second, ui.U)
	if e.Save.State != ui.StepDone || e.Save.Elapsed != 15*time.Second || e.Save.Tokens.V != 1990 || e.Steps[0].State != ui.StepDone {
		t.Errorf("save answered: %+v %+v", e.Save, e.Steps[0])
	}
	tr.apply(core.StopStepCtrlBreak)
	if e = tr.event(16*time.Second, ui.U); e.Steps[0].State != ui.StepDone || e.Save.State != ui.StepDone {
		t.Errorf("after the save, the stop goes on: %+v", e.Steps[0])
	}
}
