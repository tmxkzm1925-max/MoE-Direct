package core

import (
	"fmt"
	"time"
)

// ownedChild is the server child as the stop and ready procedures see it; *winChild is the
// real one and the tests use a fake.
type ownedChild interface {
	PID() int
	NewGroup() bool
	Exited() (uint32, bool)                // Test-ChildExited
	WaitExit(time.Duration) (uint32, bool) // Wait-ChildExit
	CtrlBreak() error                      // GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)
	ForceKill()                            // taskkill /PID /T /F, then TerminateProcess(1)
}

// Stop timings (PS:1223-1224, :12159-12168).
type stopTimings struct {
	Grace        time.Duration // GRACEFUL_STOP_S
	KillWait     time.Duration // the 15000 ms wait after the force kill
	ListenerGone time.Duration // LISTENER_GONE_S
	ListenerPoll time.Duration // the 250 ms poll
}

var psStopTimings = stopTimings{Grace: 60 * time.Second, KillWait: 15 * time.Second,
	ListenerGone: 30 * time.Second, ListenerPoll: 250 * time.Millisecond}

// TeardownResult is the record of Stop-OwnedChildGraceful (PS:12112-12175).
type TeardownResult struct {
	CtrlAttempted bool
	CtrlSent      bool
	Graceful      bool
	ExitCode      *uint32
	TaskkillUsed  bool
	ChildGone     bool
	ListenerGone  bool
	PreExited     bool
	GraceExceeded bool
	StopNonzero   bool
	Reason        string
}

// FormatExitCode is Format-ExitCode (PS:11892-11896).
func FormatExitCode(c *uint32) string {
	if c == nil {
		return "n/a"
	}
	return fmt.Sprintf("%d (0x%08X)", *c, *c)
}

// Stop step names, for the stop events.
const (
	StopStepCtrlBreak = "ctrl_break"
	StopStepGrace     = "grace_wait"
	StopStepForceKill = "force_kill"
	StopStepListener  = "listener"
)

type stopEnv struct {
	console    bool
	portOwners func(int) ([]int, error)
	t          stopTimings
	now        func() time.Time
	sleep      func(time.Duration)
	step       func(string)
}

// portListenerGone is Test-PortListenerGone (PS:12077-12082): an unreadable table is "not gone".
func portListenerGone(owners func(int) ([]int, error), port int) bool {
	o, err := owners(port)
	if err != nil || o == nil {
		return false
	}
	return len(o) == 0
}

// stopOwnedChildGraceful is Stop-OwnedChildGraceful.
func stopOwnedChildGraceful(c ownedChild, port int, env stopEnv) TeardownResult {
	r := TeardownResult{}
	step := env.step
	if step == nil {
		step = func(string) {}
	}
	if code, exited := c.Exited(); exited {
		r.ChildGone, r.PreExited = true, true
		r.ExitCode = &code
		r.Reason = "child had already exited before stop was requested"
	} else {
		step(StopStepCtrlBreak)
		if c.NewGroup() && env.console {
			r.CtrlAttempted = true
			if err := c.CtrlBreak(); err != nil {
				r.Reason = "GenerateConsoleCtrlEvent failed: " + err.Error()
			} else {
				r.CtrlSent = true
			}
		} else {
			r.CtrlAttempted = true
			r.Reason = "no console or child was not created with CREATE_NEW_PROCESS_GROUP"
		}
		if r.CtrlSent {
			step(StopStepGrace)
			if code, exited := c.WaitExit(env.t.Grace); exited {
				r.ChildGone = true
				r.ExitCode = &code
				if code == 0 {
					r.Graceful = true
				} else {
					r.StopNonzero = true
					r.Reason = "child exited non-zero during stop: " + FormatExitCode(&code)
				}
			} else {
				r.GraceExceeded = true
				r.Reason = "graceful grace period exceeded"
			}
		}
	}
	if !r.ChildGone {
		step(StopStepForceKill)
		r.TaskkillUsed = true
		c.ForceKill()
		if code, exited := c.WaitExit(env.t.KillWait); exited {
			r.ChildGone = true
			r.ExitCode = &code
		}
	}
	step(StopStepListener)
	deadline := env.now().Add(env.t.ListenerGone)
	for env.now().Before(deadline) {
		if portListenerGone(env.portOwners, port) {
			r.ListenerGone = true
			break
		}
		env.sleep(env.t.ListenerPoll)
	}
	if !r.ListenerGone {
		r.ListenerGone = portListenerGone(env.portOwners, port)
	}
	if r.Reason == "" {
		switch {
		case r.TaskkillUsed:
			r.Reason = "taskkill fallback was used"
		case !r.ChildGone:
			r.Reason = "child process still alive"
		case !r.ListenerGone:
			r.Reason = "port listener still present"
		}
	}
	return r
}

// teardownStatus is the verdict half of Complete-Teardown (PS:14792-14838): the pending
// status, or fail_teardown with the first failure in PowerShell's order.
func teardownStatus(pending Status, r TeardownResult, childWasReady bool) (Status, string) {
	why := ""
	switch {
	case r.TaskkillUsed:
		why = "taskkill fallback was used"
	case r.CtrlAttempted && !r.CtrlSent:
		why = "CTRL_BREAK could not be delivered"
	case r.GraceExceeded:
		why = "graceful grace period exceeded"
	case r.StopNonzero:
		why = "child exited non-zero during stop"
	case !r.ChildGone:
		why = "child process still alive"
	}
	if why == "" {
		requested := pending == StatusOK || pending == StatusOKSmoke
		if childWasReady && !r.ListenerGone {
			why = "port listener still present"
		} else if requested && !r.Graceful {
			why = "stop did not end in a graceful exit 0"
		}
	}
	if why != "" {
		return StatusFailTeardown, why
	}
	if !pending.Known() {
		return StatusFailTeardown, "unknown pending status" // Invoke-Launcher PS:14892
	}
	return pending, ""
}
