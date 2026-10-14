// The argument-driven mode (conflict row 15, «확정» (1); W7 duties 23, 40; F-LS 9.3): when the
// terminal is not an interactive console the launcher draws no screen and asks nothing; it
// prints one notice line, and a start given by arguments is the regular path, with the
// PowerShell exit codes (conflict row 6's table: a range violation 3, a launcher file mismatch
// 4, a failed -smoke check 6). The flags follow the PowerShell parameters
// (Start-MoeDirect.ps1:40-110): -model (start), -stop (stop before start, PowerShell -Action
// stop; GL-08), -non-interactive, -run-seconds (0 = stop right after ready), -smoke,
// -repack-mode and the allowlist keys.

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/version"
	"moedirect/launcher/internal/wire"
)

// argStart is a start given by arguments.
type argStart struct {
	model      string
	repackMode string
	settings   map[string]string // allowlist keys (port, ctx, threads, budget_mb, qd, warmup, prefetch, prefetch_k, prefetch_n)
	smoke      bool
	runSeconds int
	stop       bool // -stop: stop before start (GL-08)
}

// starter is what the argument path needs of core.Launcher.
type starter interface {
	Start(ctx context.Context, req core.Request) (<-chan core.StartEvent, error)
	Stop() (<-chan core.StopEvent, error)
	Done() <-chan struct{}
	StopBeforeStart(ctx context.Context, req core.Request) error
}

// guidance is the one line printed when no screen can be drawn and no model is given.
func guidance(reason string) string {
	return msgPrefix + " no interactive console (" + reason + "): no screen and no questions; start a model with " +
		"-model <file.gguf> [-repack-mode packed|virtual] [-ctx n] [-budget-mb n] [-qd n] [-smoke] [-run-seconds n], " +
		"or stop before start with -model <file.gguf> -stop"
}

// notice is the one line printed off an interactive console when a model is given (conflict
// row 15; W7 duty 40): no screen, the argument-driven mode runs.
func notice(reason string) string {
	return msgPrefix + " no interactive console (" + reason + "): no screen and no questions; the argument-driven mode runs"
}

// progress is the prefix of the argument-driven mode's progress lines on stdout (U-26).
const progress = "[" + version.LauncherName + "]"

// statusLine is PowerShell's final stderr line (LS 5; F-LS 11.5, 12.2): exactly one per run,
// with its fixed prefix.
func statusLine(w io.Writer, st core.Status) {
	fmt.Fprintf(w, "[moe-launcher] status=%s\n", st)
}

// refusal prints why a start or a stop ended before ready; a launcher lock held by another
// launcher first prints W7 duty 7's line (F-LS 11.7).
func refusal(errw io.Writer, what string, err error) {
	if core.StatusOf(err) == core.StatusFailInstanceLock {
		fmt.Fprintln(errw, msgPrefix, wire.LauncherLockLine)
	}
	fmt.Fprintln(errw, msgPrefix+" "+what+":", err)
}

// runStop is the argument-driven stop action (-stop; W7 duty 23; F-LS 9.3, OQ-2 (b)), as SMD's
// -Action stop with -NonInteractive (PS:13996-13998): the model is identified and the launcher
// locks are probed, then the run stops before start, cancelled_user, exit 2 (rule 6: stop
// before ready). It starts nothing and ends no process; another launcher holding the locks
// gives its line and fail_instance_lock (3).
func runStop(l starter, a argStart, out, errw io.Writer) int {
	err := l.StopBeforeStart(context.Background(), core.Request{ModelPath: a.model, RepackMode: a.repackMode, Settings: a.settings})
	st := core.StatusOf(err)
	if st == core.StatusCancelledUser {
		fmt.Fprintln(out, "[stop] stop requested before start: nothing was started")
	} else {
		refusal(errw, "stop", err)
	}
	statusLine(errw, st)
	return core.ExitCode(st)
}

// bundleLine is the passed launcher file check on this path (conflict 18-2; GL-13): a
// development build says so (F-LS 11.7, the English wording of the screen's line).
func bundleLine(r core.BundleResult) string {
	if r.State == core.BundleDev {
		return "development build · launcher file check skipped"
	}
	return fmt.Sprintf("%d file(s) match %s", r.Files, core.BundleManifestName)
}

// runArgs runs one start by arguments: it prints the steps, serves for runSeconds after ready
// (0 = stop right after ready, PowerShell's -RunSeconds) or until a console stop request
// (interrupt), and stops by the PowerShell procedure. With smoke the «작동 확인» checklist runs
// after ready (a failure ends the start as fail_smoke, 6) and the server is then stopped
// (ok_smoke). It returns the exit code.
func runArgs(l starter, a argStart, out, errw io.Writer, interrupt <-chan struct{}) int {
	req := core.Request{ModelPath: a.model, RepackMode: a.repackMode, Settings: a.settings, FromArguments: true}
	if a.smoke {
		req.Smoke = core.SmokeOn
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := l.Start(ctx, req)
	if err != nil {
		refusal(errw, "start", err)
		statusLine(errw, core.StatusOf(err))
		return core.ExitCodeOf(err)
	}
	var final core.StartEvent
	for done := false; !done; {
		select {
		case ev, ok := <-ch:
			if !ok {
				done = true
				break
			}
			switch {
			case ev.Final:
				final, done = ev, true
			case ev.Step == core.StepBundle && ev.State == core.StepDone && ev.Bundle != nil:
				fmt.Fprintf(out, "[start] %s: %s\n", ev.Step, bundleLine(*ev.Bundle))
			case ev.State == core.StepDone && ev.Detail != "":
				fmt.Fprintf(out, "[start] %s: %s\n", ev.Step, ev.Detail)
			case ev.State == core.StepFailed:
				fmt.Fprintf(out, "[start] %s failed: %s\n", ev.Step, ev.Detail)
			}
		case <-interrupt:
			cancel() // before ready a console stop request cancels the start (cancelled_user)
			interrupt = nil
		}
	}
	if !final.Ready {
		if final.Err != nil {
			refusal(errw, "start failed", final.Err)
		}
		smokeItems(out, final.Smoke)
		if final.StopInfo != nil {
			fmt.Fprintln(out, final.StopInfo.Line())
		}
		st := final.Status
		if st == "" {
			st = core.StatusFailServerStart
		}
		statusLine(errw, st)
		return final.ExitCode
	}
	if final.Preflight != nil && final.Preflight.Sweep.Line != "" { // GL-10: the speed check (Resolve-QdSweep)
		for _, l := range strings.Split(final.Preflight.Sweep.Line, "\n") {
			fmt.Fprintln(out, progress, l)
		}
	}
	fmt.Fprintf(out, "%s ready on %s (pid %d) after %.1f s\n", progress, final.Address, final.PID, final.Elapsed.Seconds())
	if final.Engine != nil { // GL-21: the engine's prefetch values next to the candidate (F-PF §8.5)
		var cand *core.PrefetchEcho
		if final.Preflight != nil {
			cand = final.Preflight.Start.PrefetchEcho
		}
		fmt.Fprintln(out, progress, core.PostReadyLine(*final.Engine, cand))
		if l := core.NCapLine(cand); l != "" { // GL-20: N_cap per layer (F-PF §8.2)
			fmt.Fprintln(out, progress, l)
		}
	}
	if final.Warmup != nil {
		fmt.Fprintln(out, progress, "warmup:", final.Warmup.Line())
	}
	if final.Smoke != nil {
		smokeItems(out, final.Smoke)
		fmt.Fprintln(out, progress, "check run:", final.Smoke.Line())
	}
	if !a.smoke && a.runSeconds > 0 {
		fmt.Fprintf(out, "%s serving for %d s; a console stop request stops it\n", progress, a.runSeconds)
		select {
		case <-time.After(time.Duration(a.runSeconds) * time.Second):
		case <-interrupt:
			fmt.Fprintln(out, "[stop] console stop request received.")
		case <-l.Done():
		}
	}
	sch, err := l.Stop()
	if err != nil {
		fmt.Fprintln(errw, msgPrefix+" stop:", err)
		statusLine(errw, core.StatusFailTeardown)
		return core.ExitCode(core.StatusFailTeardown)
	}
	code, st := 0, core.StatusOK
	for ev := range sch {
		if !ev.Final {
			continue
		}
		code, st = ev.ExitCode, ev.Status
		if ev.Event != nil {
			fmt.Fprintln(out, ev.Event.Line()) // conflict 6
		}
	}
	if a.smoke && st == core.StatusOK {
		st = core.StatusOKSmoke
	}
	statusLine(errw, st)
	return code
}

// smokeItems prints the check run's items, one line each with its result (GL-11, S8-LA-25).
func smokeItems(out io.Writer, r *core.SmokeResult) {
	if r == nil {
		return
	}
	for _, it := range r.Items {
		fmt.Fprintln(out, progress, "check run", it.Line())
	}
}

// settingsOf collects the allowlist flags given (an empty flag is not given).
func settingsOf(kv map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range kv {
		if v = strings.TrimSpace(v); v != "" {
			out[k] = v
		}
	}
	return out
}

// closeOnWindowClose ends the process at once when the console window is closed (conflict 5,
// «확정»: X = stop at once, nothing saved, the same as ctrl+c twice): the job object ends the
// server with the launcher. Go delivers CTRL_CLOSE_EVENT as SIGTERM; without this the screen
// library would start the graceful stop, which Windows cuts off after about five seconds. The
// exit code is 5 (a forced stop in the PowerShell table) while a start or a server is up, else
// the last run's code.
func closeOnWindowClose(serving func() bool, last func() int) {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM)
	go func() {
		<-sig
		code := last()
		if serving() {
			code = core.ExitCode(core.StatusFailTeardown)
		}
		os.Exit(code)
	}()
}

// interrupts is a channel closed at the first console stop request (ctrl+c or ctrl+break).
func interrupts() <-chan struct{} {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	ch := make(chan struct{})
	go func() {
		<-sig
		close(ch)
	}()
	return ch
}
