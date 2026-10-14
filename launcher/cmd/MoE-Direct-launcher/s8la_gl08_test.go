package main

// GL-08 (version 8): the command's part of S8-LA-18 and S8-LA-29 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition
// reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, sections 4.3, 9.2-9.4, 11.5 and
// 11.7). TestS8LA29 is shared with GL-16, which adds its own subtests (the custom-value
// descriptions per path); the subtests below are GL-08's.

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
)

func lines(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

// S8-LA-18: stop before start exits 2; a second launcher shows «another launcher is already
// running» and starts nothing; a stop after ready exits 0 with its stop line, a forced one 5.
func TestS8LA18(t *testing.T) {
	t.Run("stop before start", func(t *testing.T) {
		var errw bytes.Buffer
		o, _, ok := parse([]string{"-model", "m.gguf", "-stop"}, &errw)
		if !ok || !o.arg.stop || o.arg.model != "m.gguf" {
			t.Fatalf("parse -stop: %+v %q", o, errw.String())
		}
		f := &fakeStarter{}
		var out bytes.Buffer
		if got := runStop(f, o.arg, &out, &errw); got != 2 || f.stopCalls != 1 || f.starts != 0 || f.stopped ||
			strings.TrimSpace(errw.String()) != "[moe-launcher] status=cancelled_user" || !strings.Contains(out.String(), "nothing was started") {
			t.Errorf("-stop: code %d, calls %d/%d, %q %q", got, f.stopCalls, f.starts, out.String(), errw.String())
		}
		if f.req.ModelPath != "m.gguf" {
			t.Errorf("request %+v", f.req)
		}
	})
	t.Run("a second launcher starts nothing", func(t *testing.T) {
		lock := &core.LaunchError{Status: core.StatusFailInstanceLock, Reason: `single-instance: lock already held: Global\moe_direct_launcher`}
		// at the start's final event (the locks are taken in the start)
		f := &fakeStarter{final: core.StartEvent{Final: true, Status: core.StatusFailInstanceLock, ExitCode: 3, Err: lock}}
		var out, errw bytes.Buffer
		got := runArgs(f, argStart{model: "m.gguf"}, &out, &errw, nil)
		ls := lines(errw.String())
		if got != 3 || f.stopped || len(ls) != 3 || ls[0] != "MoE-Direct-launcher: another launcher is already running" ||
			ls[2] != "[moe-launcher] status=fail_instance_lock" {
			t.Errorf("start: code %d %q", got, errw.String())
		}
		// at the stop action
		f = &fakeStarter{stopErr: lock}
		errw.Reset()
		if got := runStop(f, argStart{model: "m.gguf", stop: true}, &out, &errw); got != 3 ||
			!strings.HasPrefix(errw.String(), "MoE-Direct-launcher: another launcher is already running\n") ||
			!strings.HasSuffix(errw.String(), "[moe-launcher] status=fail_instance_lock\n") {
			t.Errorf("stop action: code %d %q", got, errw.String())
		}
	})
	t.Run("stop after ready: 0 with its line; forced: 5", func(t *testing.T) {
		zero := uint32(0)
		ev := core.ServerStopEvent{At: time.Unix(1, 0), Class: core.StopClean, Status: core.StatusOK, Code: 0, Meaning: "m", ChildExit: &zero}
		f := &fakeStarter{final: core.StartEvent{Final: true, Ready: true, Status: core.StatusOK, Address: "http://127.0.0.1:8093", PID: 9},
			stop: core.StopEvent{Final: true, Status: core.StatusOK, ExitCode: 0, Event: &ev}}
		var out, errw bytes.Buffer
		if got := runArgs(f, argStart{model: "m.gguf"}, &out, &errw, nil); got != 0 || !f.stopped || !strings.Contains(out.String(), ev.Line()) ||
			!strings.Contains(out.String(), "[MoE-Direct-launcher] ready on http://127.0.0.1:8093 (pid 9)") {
			t.Errorf("clean: %d %q", got, out.String())
		}
		f.stop = core.StopEvent{Final: true, Status: core.StatusFailTeardown, ExitCode: 5}
		errw.Reset()
		if got := runArgs(f, argStart{model: "m.gguf"}, &out, &errw, nil); got != 5 || !strings.HasSuffix(errw.String(), "[moe-launcher] status=fail_teardown\n") {
			t.Errorf("forced: %d %q", got, errw.String())
		}
	})
}

// S8-LA-29, GL-08's part (W7 duties 19, 23, 40; conflict row 15).
func TestS8LA29(t *testing.T) {
	t.Run("two run modes and the arguments", func(t *testing.T) {
		want := "budget-mb catalog ctx lang model models-root non-interactive port prefetch prefetch-k prefetch-n qd repack-mode root run-seconds smoke state stop threads warmup"
		if got := strings.Join(flagNames(), " "); got != want {
			t.Errorf("flags %q, want %q", got, want)
		}
		for _, n := range flagNames() {
			for _, gone := range []string{"repro", "collect", "script", "frames"} {
				if strings.Contains(n, gone) {
					t.Errorf("flag %q of a dropped mode", n)
				}
			}
		}
		var errw bytes.Buffer
		// interactive: an interactive console with no start and no stop
		o, _, _ := parse(nil, &errw)
		if use, code, ok := argPath(o, core.TerminalInfo{Interactive: true}, &errw); use || !ok || code != 0 || errw.Len() != 0 {
			t.Errorf("interactive: %v %d %v %q", use, code, ok, errw.String())
		}
		// argument-driven: a model given, on an interactive console too
		o, _, _ = parse([]string{"-model", "m.gguf", "-run-seconds", "30", "-ctx", "8192"}, &errw)
		if use, _, ok := argPath(o, core.TerminalInfo{Interactive: true}, &errw); !use || !ok || errw.Len() != 0 ||
			o.arg.runSeconds != 30 || o.arg.settings["ctx"] != "8192" {
			t.Errorf("argument-driven on a console: %v %v %+v %q", use, ok, o.arg, errw.String())
		}
	})
	t.Run("off an interactive console: one notice line, no screen", func(t *testing.T) {
		if term := core.DetectTerminal(true); term.Interactive {
			t.Errorf("-non-interactive: %+v", term)
		}
		var errw bytes.Buffer
		o, _, _ := parse([]string{"-non-interactive", "-model", "m.gguf"}, &errw)
		use, code, ok := argPath(o, core.TerminalInfo{Reason: "-NonInteractive"}, &errw)
		ls := lines(errw.String())
		if !use || !ok || code != 0 || len(ls) != 1 || ls[0] != "MoE-Direct-launcher: no interactive console (-NonInteractive): no screen and no questions; the argument-driven mode runs" {
			t.Errorf("with a model: %v %d %v %q", use, code, ok, errw.String())
		}
		errw.Reset()
		o, _, _ = parse(nil, &errw)
		use, code, ok = argPath(o, core.TerminalInfo{Reason: "output redirected"}, &errw)
		ls = lines(errw.String())
		if !use || ok || code != 3 || len(ls) != 2 || !strings.HasPrefix(ls[0], "MoE-Direct-launcher: no interactive console (output redirected)") ||
			!strings.Contains(ls[0], "-stop") || ls[1] != "[moe-launcher] status=fail_model_path" {
			t.Errorf("without a model: %v %d %v %q", use, code, ok, errw.String())
		}
		errw.Reset()
		o, _, _ = parse([]string{"-stop"}, &errw)
		if use, code, ok := argPath(o, core.TerminalInfo{Interactive: true}, &errw); !use || ok || code != 3 ||
			!strings.HasPrefix(errw.String(), "MoE-Direct-launcher: -stop needs -model") {
			t.Errorf("-stop without a model: %v %d %v %q", use, code, ok, errw.String())
		}
	})
	t.Run("a value outside its range exits 3", func(t *testing.T) {
		var errw bytes.Buffer
		if _, code, ok := parse([]string{"-model", "m.gguf", "-run-seconds", "soon"}, &errw); ok || code != 3 ||
			!strings.HasSuffix(errw.String(), "[moe-launcher] status=fail_custom_args\n") {
			t.Errorf("bad -run-seconds: %d %q", code, errw.String())
		}
		f := &fakeStarter{final: failed(core.StatusFailCustomArgs, "invalid ctx 999999999: expected 2048..262144")}
		errw.Reset()
		var out bytes.Buffer
		if got := runArgs(f, argStart{model: "m.gguf", settings: map[string]string{"ctx": "999999999"}}, &out, &errw, nil); got != 3 ||
			f.stopped || !strings.HasSuffix(errw.String(), "[moe-launcher] status=fail_custom_args\n") || f.req.Settings["ctx"] != "999999999" {
			t.Errorf("out of range: %d %q", got, errw.String())
		}
	})
	t.Run("run-seconds serves that long, then stops", func(t *testing.T) {
		f := &fakeStarter{final: core.StartEvent{Final: true, Ready: true, Status: core.StatusOK, Address: "http://127.0.0.1:8093", PID: 9},
			stop: core.StopEvent{Final: true, Status: core.StatusOK, ExitCode: 0}}
		var out, errw bytes.Buffer
		t0 := time.Now()
		if got := runArgs(f, argStart{model: "m.gguf", runSeconds: 1}, &out, &errw, nil); got != 0 || !f.stopped || time.Since(t0) < time.Second ||
			!strings.Contains(out.String(), "[MoE-Direct-launcher] serving for 1 s") {
			t.Errorf("run-seconds: %d %v %q", got, time.Since(t0), out.String())
		}
	})
}
