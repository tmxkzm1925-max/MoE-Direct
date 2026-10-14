package wire

// GL-08 (version 8): the adapter's part of S8-LA-18 and S8-LA-32 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition
// reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, sections 4.3, 9.2 and 11.7).

import (
	"errors"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// S8-LA-18: W7 duty 7's line, the busy port's owner shown only, no action that ends another
// process, and the exit code a quit keeps (W7 duty 34; U-32 (ii)).
func TestS8LA18(t *testing.T) {
	// Another launcher holds the locks: the line at the place of the start verdict.
	for _, reason := range []string{`single-instance: lock already held: Global\moe_direct_launcher`,
		"profile lock: lock already held", "output lock: exclusive output lock unavailable", "effective port lock: lock already held"} {
		err := &core.LaunchError{Status: core.StatusFailInstanceLock, Reason: reason}
		if r, _ := userReason(err, nil); r != LauncherLockLine {
			t.Errorf("%s: %q", reason, r)
		}
		p, _ := preflightFrom("m", &core.PreflightResult{Refusal: err}, err)
		if !p.Refused || len(p.Refusals) == 0 || p.Refusals[0] != LauncherLockLine {
			t.Errorf("%s: preflight %+v", reason, p)
		}
	}
	if LauncherLockLine != "another launcher is already running" {
		t.Errorf("line %q", LauncherLockLine)
	}

	// A busy port: the owner (name, PID) on the failure, no fix that ends it.
	busy := &core.LaunchError{Status: core.StatusFailServerStart, Reason: "port 8093 is in use by another process",
		Port: &core.PortHolder{Port: 8093, PID: 4711, Name: "other-server.exe"}}
	f := failureFrom(core.StartEvent{Final: true, Status: core.StatusFailServerStart, ExitCode: 5, Err: busy, PID: 99}, time.Unix(1, 0), nil)
	if f.Holder != "other-server.exe" || f.PID != 4711 || f.ExitCode != 5 || f.Why != "port 8093 is in use by another program" || len(f.Fixes) != 0 {
		t.Errorf("busy port failure %+v", f)
	}
	b := testBackend()
	if _, err := b.Action(ui.ActionID("close-holder"), "4711"); !errors.Is(err, ui.ErrNotPorted) {
		t.Errorf("an action ended another process: %v", err)
	}

	// A quit after the server ended on its own keeps that end's code, also unpolled.
	for _, want := range []int{5, 0, 2} {
		b := testBackend()
		r := b.newRun("m", `D:\m.gguf`, func() {})
		close(r.done)
		done := make(chan struct{})
		close(done)
		b.doneFn = func() <-chan struct{} { return done }
		b.endFn = func() (core.RunEnd, bool) { return core.RunEnd{ExitCode: want}, true }
		b.stopFn = func() (<-chan core.StopEvent, error) { t.Fatal("a stop after the end"); return nil, nil }
		if got := b.Shutdown(time.Second); got != want {
			t.Errorf("quit after an end with code %d: %d", want, got)
		}
	}
	// A server still running at the quit is stopped, and the stop's code is the exit code.
	b = testBackend()
	r := b.newRun("m", `D:\m.gguf`, func() {})
	close(r.done)
	b.doneFn = func() <-chan struct{} { return make(chan struct{}) }
	b.endFn = func() (core.RunEnd, bool) { return core.RunEnd{}, false }
	b.stopFn = func() (<-chan core.StopEvent, error) {
		ch := make(chan core.StopEvent, 1)
		ch <- core.StopEvent{Final: true, Status: core.StatusOK, ExitCode: 0}
		close(ch)
		return ch, nil
	}
	b.setExit(3)
	if got := b.Shutdown(time.Second); got != 0 {
		t.Errorf("quit while serving: %d", got)
	}
}

// S8-LA-32: a busy port shows the failure page with the owner process; other start failures
// show the failure page.
func TestS8LA32(t *testing.T) {
	cases := []struct {
		err    error
		holder string
		pid    int
		why    string
	}{
		{&core.LaunchError{Status: core.StatusFailServerStart, Reason: "port busy", Port: &core.PortHolder{Port: 8100, PID: 12, Name: "x.exe"}},
			"x.exe", 12, "port 8100 is in use by another program"},
		{&core.LaunchError{Status: core.StatusFailServerStart, Reason: "server child exited before ready (code=1 (0x00000001))"},
			"", 99, ""},
		{&core.LaunchError{Status: core.StatusFailServerStart, Reason: "ready timeout"}, "", 99, ""},
	}
	for _, c := range cases {
		f := failureFrom(core.StartEvent{Final: true, Status: core.StatusFailServerStart, ExitCode: 5, Err: c.err, PID: 99}, time.Unix(1, 0), nil)
		if f == nil || f.Stage != string(core.StatusFailServerStart) || f.ExitCode != 5 || f.Holder != c.holder || f.PID != c.pid ||
			(c.why != "" && f.Why != c.why) || f.What == "" {
			t.Errorf("%v: %+v", c.err, f)
		}
		if c.holder == "" && strings.Contains(f.Why, "in use by another program") {
			t.Errorf("%v: an owner without a busy port", c.err)
		}
	}
}
