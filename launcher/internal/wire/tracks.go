// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// The H15 start-step and H6 stop-step lists (moved unchanged from map.go in FIX-WIRE).

package wire

import (
	"fmt"
	"strings"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// stepLabels name the core start steps on H15. Each fits H15's 14-cell name column
// (ui stepRows, nameW 14): the first rehearsal showed longer names pushing the row past the
// panel, merged with the detail and cut at the time column.
var stepLabels = map[string]string{
	core.StepCatalog: "read catalog", core.StepIdentify: "identify model", core.StepArtifacts: "repack files",
	core.StepLocks: "launcher locks", core.StepSweep: ui.StepSweepName, core.StepConfig: "server args",
	core.StepSizing: "check memory", core.StepSpawn: "start server", core.StepReady: "wait /health",
	core.StepBundle: "bundle check", core.StepWarmup: "warm up", core.StepSmoke: "check run", // FIX2-WIRE
}

// startNameW and stopNameW are the name columns of H15 and H6 (ui stepRows callers).
const (
	startNameW = 14
	stopNameW  = 18
)

// startTrack keeps the H15 step list across core events.
type startTrack struct {
	model   string
	steps   []ui.Step
	began   map[string]time.Time
	lastS   ui.Val
	lastRAM ui.Val
	notices []string
	// the load bar's span (GL-07, decision ③): the server child started, /health ready
	loadRunning, loadReady bool
	loadTook               time.Duration
	warm                   string // GL-15: the core's warm-start status text
}

func newStartTrack(model string) *startTrack {
	t := &startTrack{model: model, began: map[string]time.Time{}, lastS: ui.U, lastRAM: ui.U}
	for _, s := range core.AllStartSteps {
		t.steps = append(t.steps, ui.Step{Name: stepLabels[s], State: ui.StepTodo})
	}
	return t
}

func stepIndex(name string) int {
	for i, s := range core.AllStartSteps {
		if s == name {
			return i
		}
	}
	return -1
}

// apply folds one core event into the step list and gives the screen's event.
func (t *startTrack) apply(ev core.StartEvent, now time.Time) ui.StartEvent {
	if i := stepIndex(ev.Step); i >= 0 && !ev.Final {
		st := &t.steps[i]
		switch ev.State {
		case core.StepRunning:
			if st.State != ui.StepRunning {
				t.began[ev.Step] = now
			}
			st.State = ui.StepRunning
		case core.StepDone:
			st.State = ui.StepDone
			if b, ok := t.began[ev.Step]; ok {
				st.Took = now.Sub(b)
			}
		case core.StepFailed:
			st.State, st.Detail = ui.StepFailed, ev.Detail
		}
		// GL-11, conflict 18-1: a passed check run shows nothing on the screen (its result is in
		// 4 Log only), so its row keeps no detail.
		if ev.Detail != "" && ev.State != core.StepFailed && ev.Step != core.StepSmoke {
			st.Detail = ev.Detail
		}
		if ev.Step == core.StepBundle && ev.Bundle != nil && ev.State == core.StepDone {
			st.Detail = bundleStep(*ev.Bundle)
		}
	}
	if ev.Preflight != nil {
		p := ev.Preflight
		if p.HasLastReady {
			t.lastS = ui.K(p.LastReady.Seconds())
		}
		t.notices = append([]string(nil), p.Notices...)
		if p.Config != nil {
			t.steps[stepIndex(core.StepSpawn)].Detail = fmt.Sprintf("llama-server.exe on %s:%d", p.Config.Host, p.Config.Port)
		}
	}
	if ev.LastMem > 0 {
		t.lastRAM = gbBytes(ev.LastMem)
	}
	if ev.Warm != "" {
		t.warm = ev.Warm
	}
	if ev.Step == core.StepReady && !ev.Final { // the load bar spans the child's start to /health ready
		switch ev.State {
		case core.StepRunning:
			t.loadRunning = true
		case core.StepDone:
			t.loadRunning, t.loadReady, t.loadTook = false, true, ev.Elapsed
		}
	}
	out := ui.StartEvent{Model: t.model, Steps: append([]ui.Step(nil), t.steps...), Elapsed: ev.Elapsed,
		LastS: t.lastS, LastRAM: t.lastRAM, Notices: t.notices, DiskGBs: ui.Val{},
		LoadRunning: t.loadRunning, LoadReady: t.loadReady, LoadTook: t.loadTook, Warm: warmShown(t.warm)}
	if ev.ChildMem.WorkingSetBytes > 0 {
		out.RAMGB = gbBytes(ev.ChildMem.WorkingSetBytes)
	}
	if ev.Final && ev.Ready {
		for i := range out.Steps {
			out.Steps[i].State = ui.StepDone
		}
		if ev.Smoke == nil { // conflict 18-1: the check run did not run at this start
			i := stepIndex(core.StepSmoke)
			out.Steps[i].State, out.Steps[i].Detail = ui.StepSkipped, "off for this start"
		}
		out.Ready = true
		out.Address = ev.Address + "/v1"
	}
	return out
}

// failureFrom is H7 for a start that ended without ready.
func failureFrom(ev core.StartEvent, at time.Time, logTail []string) *ui.Failure {
	f := &ui.Failure{Stage: string(ev.Status), ExitCode: ev.ExitCode, At: at, LogTail: logTail, LogFile: ev.ErrLog,
		PID: ev.PID}
	f.What = core.StatusHint(ev.Status)
	if ev.Status == core.StatusFailCustomArgs { // the hint is the argument-driven mode's (F-LS 11.4)
		f.What = "a setting of this start was refused"
	}
	if f.What == "" {
		f.What = "the server did not start"
	}
	if ev.Err != nil {
		f.Why, _ = userReason(ev.Err, ev.Preflight) // item 32: the full text is in 4 Log
	}
	if le, ok := core.AsLaunchError(ev.Err); ok && le.Port != nil {
		// GL-08, conflict row 6: the owner of the busy port, shown only (no action ends it)
		f.Holder, f.PID = le.Port.Name, le.Port.PID
	}
	loadFailure(f, ev.Err)               // conflict 8
	if ev.Smoke != nil && !ev.Smoke.OK { // conflict 18-1
		f.SmokeFailed = true
		f.Why = "failed items " + strings.Join(ev.Smoke.Failures, ", ")
	}
	if ev.Teardown != nil && ev.Teardown.Reason != "" {
		f.Why += " · teardown: " + ev.Teardown.Reason
	}
	return f
}

// stopTrack keeps the HQ step list across core stop events.
type stopTrack struct {
	mode  ui.StopMode
	steps []ui.Step
	save  *core.CacheSaveTrack // the cache-save bar, once the stop reports a save (GL-07)
	// GL-15: a failed save's reason, or why this stop stores nothing
	saveErr, noSave string
}

var stopSteps = []struct{ core, label string }{
	{"", "save prompt cache"},
	{core.StopStepCtrlBreak, "send CTRL_BREAK"},
	{core.StopStepGrace, "wait exit (60 s)"},
	{core.StopStepForceKill, "force kill"},
	{core.StopStepListener, "port listener gone"},
}

func newStopTrack(mode ui.StopMode) *stopTrack {
	t := &stopTrack{mode: mode}
	for _, s := range stopSteps {
		t.steps = append(t.steps, ui.Step{Name: s.label, State: ui.StepTodo})
	}
	return t
}

// apply marks the core step running and every earlier one done (a skipped kill stays skipped).
func (t *stopTrack) apply(step string) {
	idx := -1
	for i, s := range stopSteps {
		if s.core != "" && s.core == step {
			idx = i
		}
	}
	if idx < 0 {
		return
	}
	for i := 1; i < idx; i++ {
		if t.steps[i].State == ui.StepRunning || t.steps[i].State == ui.StepTodo {
			if stopSteps[i].core == core.StopStepForceKill {
				t.steps[i].State = ui.StepSkipped
			} else {
				t.steps[i].State = ui.StepDone
			}
		}
	}
	t.steps[idx].State = ui.StepRunning
}

func (t *stopTrack) finish(td *core.TeardownResult) {
	for i := 1; i < len(t.steps); i++ {
		switch {
		case stopSteps[i].core == core.StopStepForceKill && (td == nil || !td.TaskkillUsed):
			t.steps[i].State = ui.StepSkipped
		case stopSteps[i].core == core.StopStepCtrlBreak && td != nil && td.PreExited:
			t.steps[i].State, t.steps[i].Detail = ui.StepSkipped, "the server had already exited"
		case stopSteps[i].core == core.StopStepGrace && td != nil && td.PreExited:
			t.steps[i].State = ui.StepSkipped
		case stopSteps[i].core == core.StopStepListener && td != nil && !td.ListenerGone:
			t.steps[i].State, t.steps[i].Detail = ui.StepFailed, "port listener still present"
		default:
			t.steps[i].State = ui.StepDone
		}
	}
	if td != nil && td.ExitCode != nil {
		t.steps[2].Detail = "exit code " + core.FormatExitCode(td.ExitCode)
	}
}

// saveFrom takes the cache-save bar of a core stop event (GL-07, decision ③): the save step
// runs until the completion response, then is done.
func (t *stopTrack) saveFrom(s *core.CacheSaveTrack) {
	c := *s
	t.save = &c
	st := &t.steps[0]
	st.Detail = ""
	st.State = ui.StepRunning
	if c.Done {
		st.State = ui.StepDone
	}
}

// saveEnd closes the save step without a completed save (GL-15): failed (degraded, the stop goes
// on) or skipped with the reason this stop stores nothing.
func (t *stopTrack) saveEnd(saveErr, noSave string) {
	t.saveErr, t.noSave = saveErr, noSave
	st := &t.steps[0]
	if saveErr != "" {
		st.State, st.Detail = ui.StepFailed, "the slot save failed: "+saveErr
		return
	}
	st.State, st.Detail = ui.StepSkipped, noSave
}

// saveProgress is the screen's cache-save bar: waiting until the stop reports a save, skipped
// when it stores nothing, failed when the save failed (GL-15).
func (t *stopTrack) saveProgress() ui.SaveProgress {
	switch {
	case t.noSave != "":
		return ui.SaveProgress{State: ui.StepSkipped, LastS: ui.U, Tokens: ui.U}
	case t.save == nil:
		return ui.SaveProgress{State: ui.StepTodo, LastS: ui.U, Tokens: ui.U}
	}
	p := ui.SaveProgress{State: ui.StepRunning, Elapsed: t.save.Elapsed, LastS: ui.U, Tokens: ui.K(float64(t.save.Tokens))}
	if t.save.HasExpect {
		p.LastS = ui.K(t.save.Expect.Seconds())
	}
	if t.save.Done {
		p.State = ui.StepDone
		if t.save.Saved > 0 {
			p.Tokens = ui.K(float64(t.save.Saved))
		}
	}
	if t.saveErr != "" {
		p.State = ui.StepFailed
	}
	return p
}

func (t *stopTrack) event(elapsed time.Duration, model ui.Val) ui.StopEvent {
	return ui.StopEvent{Mode: t.mode, Steps: append([]ui.Step(nil), t.steps...), Elapsed: elapsed,
		Save: t.saveProgress(), ModelRAMGB: model}
}

// warmShown is H15's «warm start» (S8-LA-17): a start with a stored slot state to restore, or
// one restored; a cold or off start shows nothing. No clock time is ever shown.
func warmShown(status string) bool {
	return status == "eligible" || strings.HasPrefix(status, "restored(")
}
