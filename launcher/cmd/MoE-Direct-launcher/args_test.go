package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"moedirect/launcher/internal/core"
)

// fakeStarter gives one start's final event and, after ready, one stop.
type fakeStarter struct {
	final   core.StartEvent
	stop    core.StopEvent
	req     core.Request
	stopped bool
	done    chan struct{}
	// GL-08: the -stop action
	stopErr   error
	stopCalls int
	starts    int
}

func (f *fakeStarter) Start(_ context.Context, req core.Request) (<-chan core.StartEvent, error) {
	f.req = req
	f.starts++
	ch := make(chan core.StartEvent, 3)
	ch <- core.StartEvent{Step: core.StepBundle, State: core.StepDone, Detail: "dev"}
	ch <- f.final
	close(ch)
	return ch, nil
}

func (f *fakeStarter) Stop() (<-chan core.StopEvent, error) {
	f.stopped = true
	ch := make(chan core.StopEvent, 1)
	ch <- f.stop
	close(ch)
	return ch, nil
}

// StopBeforeStart is the -stop action's core call (GL-08): stopErr, or cancelled_user.
func (f *fakeStarter) StopBeforeStart(_ context.Context, req core.Request) error {
	f.req = req
	f.stopCalls++
	if f.stopErr != nil {
		return f.stopErr
	}
	return &core.LaunchError{Status: core.StatusCancelledUser, Reason: "stop requested before start"}
}

func (f *fakeStarter) Done() <-chan struct{} {
	if f.done == nil {
		f.done = make(chan struct{})
	}
	return f.done
}

func failed(st core.Status, why string) core.StartEvent {
	return core.StartEvent{Final: true, Status: st, ExitCode: core.ExitCode(st), Err: &core.LaunchError{Status: st, Reason: why}}
}

// Conflict 15: no screen without an interactive console; one guidance line and the PowerShell
// codes on the argument path (a range violation 3, a launcher file mismatch 4, a failed
// -smoke check 6, a clean run 0 with the stop's conflict 6 line).
func TestArgumentPathRow15(t *testing.T) {
	var errw bytes.Buffer
	o, _, ok := parse([]string{"-non-interactive"}, &errw)
	if !ok {
		t.Fatal("parse")
	}
	use, code, ok := argPath(o, core.TerminalInfo{Interactive: false, Reason: "-NonInteractive"}, &errw)
	lines := strings.Split(strings.TrimSpace(errw.String()), "\n")
	if !use || ok || code != 3 || len(lines) != 2 || !strings.HasPrefix(lines[0], "MoE-Direct-launcher: no interactive console (-NonInteractive)") ||
		lines[1] != "[moe-launcher] status=fail_model_path" {
		t.Errorf("guidance: %v %d %v %q", use, code, ok, errw.String())
	}
	// No console and no flag: the same guidance (output redirected).
	errw.Reset()
	o, _, _ = parse(nil, &errw)
	if use, code, _ := argPath(o, core.TerminalInfo{Reason: "output redirected"}, &errw); !use || code != 3 || !strings.Contains(errw.String(), "(output redirected)") {
		t.Errorf("redirected: %q", errw.String())
	}
	// An interactive console without a model: the screen.
	if use, _, ok := argPath(o, core.TerminalInfo{Interactive: true}, &errw); use || !ok {
		t.Error("the screen")
	}
	// A bad flag is fail_custom_args (3), not the flag package's 2.
	errw.Reset()
	if _, code, ok := parse([]string{"-no-such-flag"}, &errw); ok || code != 3 || !strings.Contains(errw.String(), "status=fail_custom_args") {
		t.Errorf("bad flag %d %q", code, errw.String())
	}
	if _, code, ok := parse([]string{"stray"}, &errw); ok || code != 3 {
		t.Errorf("stray argument %d", code)
	}
	// The flags reach the request.
	o, _, _ = parse([]string{"-model", `D:\m.gguf`, "-repack-mode", "virtual", "-ctx", "8192", "-budget-mb", "12288", "-smoke"}, &errw)
	if o.arg.model != `D:\m.gguf` || o.arg.repackMode != "virtual" || o.arg.settings["ctx"] != "8192" || o.arg.settings["budget_mb"] != "12288" ||
		len(o.arg.settings) != 2 || !o.arg.smoke {
		t.Errorf("options %+v", o.arg)
	}

	codes := []struct {
		final core.StartEvent
		want  int
		st    string
	}{
		{failed(core.StatusFailCustomArgs, "invalid ctx 999999999: expected 2048..262144"), 3, "fail_custom_args"},
		{failed(core.StatusFailGateBundle, "2 launcher file(s) differ from bundle_manifest.json"), 4, "fail_gate_bundle"},
		{failed(core.StatusFailSmoke, "smoke assertions failed: 4:cancel"), 6, "fail_smoke"},
	}
	for _, c := range codes {
		f := &fakeStarter{final: c.final}
		var out, ew bytes.Buffer
		if got := runArgs(f, argStart{model: "m.gguf", smoke: c.st == "fail_smoke"}, &out, &ew, nil); got != c.want ||
			!strings.HasSuffix(strings.TrimSpace(ew.String()), "status="+c.st) || f.stopped {
			t.Errorf("%s: code %d, stderr %q", c.st, got, ew.String())
		}
		if c.st == "fail_smoke" && f.req.Smoke != core.SmokeOn {
			t.Error("-smoke does not reach the request")
		}
		if !f.req.FromArguments { // GL-16: its refusals name the argument-driven mode (F-LS 11.4)
			t.Error("the argument path's request is not marked")
		}
	}

	// Ready, -run-seconds 0: stopped right after ready; the stop's line (conflict 6) is printed.
	zero := uint32(0)
	ev := core.ServerStopEvent{Class: core.StopClean, Status: core.StatusOK, Code: 0, Meaning: "stopped normally", ChildExit: &zero}
	f := &fakeStarter{final: core.StartEvent{Final: true, Ready: true, Status: core.StatusOK, Address: "http://127.0.0.1:8093", PID: 9},
		stop: core.StopEvent{Final: true, Status: core.StatusOK, ExitCode: 0, Event: &ev}}
	var out, ew bytes.Buffer
	if got := runArgs(f, argStart{model: "m.gguf"}, &out, &ew, nil); got != 0 || !f.stopped || !strings.Contains(out.String(), ev.Line()) ||
		strings.TrimSpace(ew.String()) != "[moe-launcher] status=ok" || !strings.Contains(out.String(), "[start] bundle: dev") {
		t.Errorf("clean run %d %q %q", got, out.String(), ew.String())
	}
	// -smoke that passed: ok_smoke.
	f.stopped = false
	ew.Reset()
	if got := runArgs(f, argStart{model: "m.gguf", smoke: true}, &out, &ew, nil); got != 0 || strings.TrimSpace(ew.String()) != "[moe-launcher] status=ok_smoke" {
		t.Errorf("smoke run %d %q", got, ew.String())
	}
	// A forced stop: 5.
	f.stop = core.StopEvent{Final: true, Status: core.StatusFailTeardown, ExitCode: 5}
	if got := runArgs(f, argStart{model: "m.gguf"}, &out, &ew, nil); got != 5 {
		t.Errorf("forced stop %d", got)
	}
	// A start refused before any event: its own code.
	if got := runArgs(errStarter{&fakeStarter{}}, argStart{model: "m.gguf"}, &out, &ew, nil); got != 2 {
		t.Errorf("busy %d", got)
	}
}

type errStarter struct{ *fakeStarter }

func (errStarter) Start(context.Context, core.Request) (<-chan core.StartEvent, error) {
	return nil, &core.LaunchError{Status: core.StatusCancelledUser, Reason: "x"}
}
