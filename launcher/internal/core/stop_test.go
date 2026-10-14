package core

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeChild is a scripted server child.
type fakeChild struct {
	pid       int
	newGroup  bool
	exited    bool
	code      uint32
	ctrlErr   error
	onCtrl    func(c *fakeChild) // what CTRL_BREAK does to the child
	onKill    func(c *fakeChild)
	ctrlCalls int
	killCalls int
	waits     []time.Duration
}

func (c *fakeChild) PID() int       { return c.pid }
func (c *fakeChild) NewGroup() bool { return c.newGroup }
func (c *fakeChild) Exited() (uint32, bool) {
	if c.exited {
		return c.code, true
	}
	return 0, false
}
func (c *fakeChild) WaitExit(d time.Duration) (uint32, bool) {
	c.waits = append(c.waits, d)
	return c.Exited()
}
func (c *fakeChild) CtrlBreak() error {
	c.ctrlCalls++
	if c.ctrlErr != nil {
		return c.ctrlErr
	}
	if c.onCtrl != nil {
		c.onCtrl(c)
	}
	return nil
}
func (c *fakeChild) ForceKill() {
	c.killCalls++
	if c.onKill != nil {
		c.onKill(c)
	}
}

// fakeClock makes the 30 s listener wait instant.
type fakeClock struct{ t time.Time }

func (f *fakeClock) now() time.Time        { return f.t }
func (f *fakeClock) sleep(d time.Duration) { f.t = f.t.Add(d) }

func exitWith(code uint32) func(*fakeChild) {
	return func(c *fakeChild) { c.exited, c.code = true, code }
}

func runStop(c *fakeChild, console bool, owners func(int) ([]int, error)) (TeardownResult, []string) {
	clk := &fakeClock{t: time.Unix(0, 0)}
	var steps []string
	r := stopOwnedChildGraceful(c, 8093, stopEnv{console: console, portOwners: owners, t: psStopTimings,
		now: clk.now, sleep: clk.sleep, step: func(s string) { steps = append(steps, s) }})
	return r, steps
}

var noListener = func(int) ([]int, error) { return []int{}, nil }

func TestStopProcedureFakeChild(t *testing.T) {
	// 1. CTRL_BREAK, exit 0 inside the grace period, listener gone: ok, exit 0.
	c := &fakeChild{pid: 100, newGroup: true, onCtrl: exitWith(0)}
	r, steps := runStop(c, true, noListener)
	if !r.CtrlSent || !r.Graceful || !r.ChildGone || !r.ListenerGone || r.TaskkillUsed || c.killCalls != 0 {
		t.Fatalf("graceful: %+v", r)
	}
	if len(c.waits) != 1 || c.waits[0] != 60*time.Second {
		t.Errorf("grace wait %v, want one wait of 60 s (GRACEFUL_STOP_S)", c.waits)
	}
	if strings.Join(steps, ",") != "ctrl_break,grace_wait,listener" {
		t.Errorf("steps %v", steps)
	}
	if st, why := teardownStatus(StatusOK, r, true); st != StatusOK || why != "" || ExitCode(st) != 0 {
		t.Errorf("status %s %q", st, why)
	}

	// 2. Exit non-zero during the stop: fail_teardown, exit 5.
	c = &fakeChild{pid: 101, newGroup: true, onCtrl: exitWith(0xC000013A)}
	r, _ = runStop(c, true, noListener)
	if !r.StopNonzero || r.Graceful || !strings.Contains(r.Reason, "3221225786 (0xC000013A)") {
		t.Errorf("nonzero: %+v", r)
	}
	if st, why := teardownStatus(StatusOK, r, true); st != StatusFailTeardown || why != "child exited non-zero during stop" || ExitCode(st) != 5 {
		t.Errorf("nonzero status %s %q", st, why)
	}

	// 3. The child ignores CTRL_BREAK: grace exceeded, the forced kill (15 s wait), fail_teardown
	//    with the taskkill reason first (Complete-Teardown order).
	c = &fakeChild{pid: 102, newGroup: true, onKill: exitWith(1)}
	r, steps = runStop(c, true, noListener)
	if !r.GraceExceeded || !r.TaskkillUsed || !r.ChildGone || c.killCalls != 1 {
		t.Errorf("grace exceeded: %+v", r)
	}
	if len(c.waits) != 2 || c.waits[1] != 15*time.Second {
		t.Errorf("waits %v", c.waits)
	}
	if strings.Join(steps, ",") != "ctrl_break,grace_wait,force_kill,listener" {
		t.Errorf("steps %v", steps)
	}
	if st, why := teardownStatus(StatusOK, r, true); st != StatusFailTeardown || why != "taskkill fallback was used" {
		t.Errorf("status %s %q", st, why)
	}

	// 4. No console: CTRL_BREAK cannot be sent, straight to the kill.
	c = &fakeChild{pid: 103, newGroup: true, onKill: exitWith(1)}
	r, _ = runStop(c, false, noListener)
	if !r.CtrlAttempted || r.CtrlSent || c.ctrlCalls != 0 || !r.TaskkillUsed {
		t.Errorf("no console: %+v", r)
	}
	// 4b. GenerateConsoleCtrlEvent fails.
	c = &fakeChild{pid: 104, newGroup: true, ctrlErr: errors.New("gle=6"), onKill: exitWith(1)}
	r, _ = runStop(c, true, noListener)
	if r.CtrlSent || !strings.Contains(r.Reason, "GenerateConsoleCtrlEvent failed") {
		t.Errorf("ctrl failure: %+v", r)
	}

	// 5. The child had already exited (runtime exit): pre_exited, the pending status stands.
	c = &fakeChild{pid: 105, exited: true, code: 3}
	r, steps = runStop(c, true, noListener)
	if !r.PreExited || !r.ChildGone || r.CtrlAttempted || *r.ExitCode != 3 {
		t.Errorf("pre-exited: %+v", r)
	}
	if strings.Join(steps, ",") != "listener" {
		t.Errorf("steps %v", steps)
	}
	if st, _ := teardownStatus(StatusFailRuntimeExit, r, true); st != StatusFailRuntimeExit || ExitCode(st) != 5 {
		t.Errorf("runtime exit status %s", st)
	}

	// 6. A listener that stays on the port after a ready server: fail_teardown.
	c = &fakeChild{pid: 106, newGroup: true, onCtrl: exitWith(0)}
	r, _ = runStop(c, true, func(int) ([]int, error) { return []int{999}, nil })
	if r.ListenerGone || r.Reason != "port listener still present" {
		t.Errorf("listener: %+v", r)
	}
	if st, why := teardownStatus(StatusOK, r, true); st != StatusFailTeardown || why != "port listener still present" {
		t.Errorf("status %s %q", st, why)
	}
	// ... but not before ready (ChildWasReady false), where the start's own failure stands.
	if st, _ := teardownStatus(StatusFailServerStart, r, false); st != StatusFailServerStart {
		t.Errorf("pre-ready status %s", st)
	}
	// An unreadable listener table is "not gone" (fail-close).
	c = &fakeChild{pid: 107, newGroup: true, onCtrl: exitWith(0)}
	r, _ = runStop(c, true, func(int) ([]int, error) { return nil, errors.New("table") })
	if r.ListenerGone {
		t.Error("unreadable table read as gone")
	}
	// A cancelled start that stopped cleanly keeps cancelled_user (exit 2).
	if st, _ := teardownStatus(StatusCancelledUser, TeardownResult{CtrlAttempted: true, CtrlSent: true, Graceful: true, ChildGone: true, ListenerGone: true}, false); st != StatusCancelledUser || ExitCode(st) != 2 {
		t.Errorf("cancelled %s", st)
	}
}

func listenerPort(t *testing.T, srv *httptest.Server) (string, int) {
	t.Helper()
	host, p, err := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	var port int
	fmt.Sscan(p, &port)
	return host, port
}

func TestWaitForServerReady(t *testing.T) {
	healthy := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" && healthy {
			w.Write([]byte(`{"status":"ok"}`))
			return
		}
		w.WriteHeader(503)
	}))
	defer srv.Close()
	host, port := listenerPort(t, srv)
	errLog := filepath.Join(t.TempDir(), "server_err.log")
	os.WriteFile(errLog, []byte("loading\n"), 0o644)
	fast := readyTimings{Timeout: 900 * time.Second, Poll: time.Millisecond, HealthTimeout: 2 * time.Second}
	owners := func(int) ([]int, error) { return []int{4242}, nil }
	ctx := context.Background()

	// 503 twice, then ready: two ticks, then nil.
	c := &fakeChild{pid: 4242}
	ticks := 0
	err := waitForServerReady(ctx, c, host, port, errLog, owners, fast, time.Now, func(time.Duration) {},
		func(time.Duration) {
			ticks++
			if ticks == 2 {
				healthy = true
			}
		})
	if err != nil || ticks != 2 {
		t.Fatalf("ready: %v after %d ticks", err, ticks)
	}
	// Another process owns the listener: fatal fail_server_start.
	err = waitForServerReady(ctx, c, host, port, errLog, func(int) ([]int, error) { return []int{7}, nil }, fast, time.Now, func(time.Duration) {}, nil)
	wantStatus(t, err, StatusFailServerStart, "listener PID mismatch")
	// The child exits with the engine policy anchor exactly once: fail_gate_engine_seal.
	os.WriteFile(errLog, []byte("x\r\n"+enginePolicyAnchor+"\r\n"), 0o644)
	err = waitForServerReady(ctx, &fakeChild{pid: 1, exited: true, code: 1}, host, port, errLog, owners, fast, time.Now, func(time.Duration) {}, nil)
	wantStatus(t, err, StatusFailGateEngineSeal, "policy gate")
	// Twice is not the anchor: fail_server_start.
	os.WriteFile(errLog, []byte(enginePolicyAnchor+"\n"+enginePolicyAnchor+"\n"), 0o644)
	err = waitForServerReady(ctx, &fakeChild{pid: 1, exited: true, code: 1}, host, port, errLog, owners, fast, time.Now, func(time.Duration) {}, nil)
	wantStatus(t, err, StatusFailServerStart, "exited before ready (code=1 (0x00000001))")
	// 900 s without ready: "ready timeout" (a fake clock).
	healthy = false
	clk := &fakeClock{t: time.Unix(0, 0)}
	err = waitForServerReady(ctx, c, host, port, errLog, owners, readyTimings{Timeout: 900 * time.Second, Poll: 100 * time.Second, HealthTimeout: time.Second},
		clk.now, clk.sleep, nil)
	wantStatus(t, err, StatusFailServerStart, "ready timeout")
	// A cancelled start: cancelled_user.
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	err = waitForServerReady(cctx, c, host, port, errLog, owners, fast, time.Now, func(time.Duration) {}, nil)
	wantStatus(t, err, StatusCancelledUser, "")
}
