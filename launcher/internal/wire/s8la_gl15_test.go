package wire

// GL-15 (version 8): the adapter's part of warm start, S8-LA-17 of the plan
// reviews/_plan_8_STAGE1_v8-plansplit.md (d1dbe526f): the core's warm-start status text becomes
// H15's «warm start» flag (no clock time travels: the flag is a bool), and the stop's save
// reports (GL-07 finding 1: every CacheSaveTrack on StopEvent.Save; a failed save; a stop that
// stores nothing) reach the stop page.

import (
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// S8-LA-17: «warm start» shows for a start with a stored state to restore and for a restored one,
// never for a cold or off start; it stays through the later events of the start.
func TestS8LA17(t *testing.T) {
	t0 := time.Unix(100, 0)
	for _, c := range []struct {
		sizing, warmup string
		atSizing, end  bool
	}{
		{"eligible", "restored(128 tokens)", true, true},
		{"eligible", "cold(restore_failed)", true, false},
		{"cold(sidecar_missing)", "", false, false},
		{"off(user)", "", false, false},
		{"off(mode)", "", false, false},
		{"", "", false, false},
	} {
		tr := newStartTrack("m")
		pre := &core.PreflightResult{Config: &core.Config{Host: "127.0.0.1", Port: 8093}}
		ev := tr.apply(core.StartEvent{Step: core.StepSizing, State: core.StepDone, Preflight: pre, Warm: c.sizing}, t0)
		if ev.Warm != c.atSizing {
			t.Errorf("%q at sizing: warm %v", c.sizing, ev.Warm)
		}
		ev = tr.apply(core.StartEvent{Step: core.StepReady, State: core.StepDone, Elapsed: time.Second}, t0)
		if ev.Warm != c.atSizing {
			t.Errorf("%q at ready: warm %v", c.sizing, ev.Warm)
		}
		ev = tr.apply(core.StartEvent{Step: core.StepWarmup, State: core.StepDone, Warm: c.warmup}, t0)
		if ev.Warm != c.end {
			t.Errorf("%q then %q: warm %v", c.sizing, c.warmup, ev.Warm)
		}
	}
}

// GL-15: the save step of the stop page: the tracks of TimedCacheSave run the bar to done; a
// failed save fails the step (the stop goes on); a stop that stores nothing skips it with the
// core's reason; a cancelled start never saves.
func TestGL15StopSave(t *testing.T) {
	tr := newStopTrack(ui.StopQuit)
	tr.saveFrom(&core.CacheSaveTrack{Tokens: 128, Elapsed: time.Second})
	if e := tr.event(time.Second, ui.U); e.Save.State != ui.StepRunning || e.Steps[0].State != ui.StepRunning {
		t.Errorf("running save %+v", e.Save)
	}
	tr.saveFrom(&core.CacheSaveTrack{Tokens: 128, Elapsed: 2 * time.Second, Done: true, Saved: 128})
	if e := tr.event(2*time.Second, ui.U); e.Save.State != ui.StepDone || e.Save.Tokens.V != 128 || e.Steps[0].State != ui.StepDone {
		t.Errorf("done save %+v %+v", e.Save, e.Steps[0])
	}

	tr = newStopTrack(ui.StopQuit)
	tr.saveFrom(&core.CacheSaveTrack{Tokens: 128, Elapsed: time.Second})
	tr.saveEnd("http status 500", "")
	if e := tr.event(time.Second, ui.U); e.Save.State != ui.StepFailed || e.Steps[0].State != ui.StepFailed ||
		e.Steps[0].Detail != "the slot save failed: http status 500" {
		t.Errorf("failed save %+v %+v", e.Save, e.Steps[0])
	}

	tr = newStopTrack(ui.StopQuit)
	tr.saveEnd("", "warm start is switched off")
	if e := tr.event(0, ui.U); e.Save.State != ui.StepSkipped || e.Steps[0].State != ui.StepSkipped ||
		e.Steps[0].Detail != "warm start is switched off" {
		t.Errorf("no save %+v %+v", e.Save, e.Steps[0])
	}
}
