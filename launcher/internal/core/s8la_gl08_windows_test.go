//go:build windows

package core

// GL-08 (version 8): the server lifecycle's scenario tests in core, S8-LA-18, S8-LA-21,
// S8-LA-29 and S8-LA-32 of the plan reviews/_plan_8_STAGE1_v8-plansplit.md (d1dbe526f), built to
// the F-LS edition reviews/_spec_8_F-LS_launcher_go_edition.md (d3a5ab2ed) sections 4.3, 9 and
// 11. The live parts start the test binary itself as the fake server fakeServerExe (the helper
// process of stopwarmup_test.go, steered by its fake_server_mode.txt) inside the launcher's
// job object, on a free loopback port; they never touch a process the test did not start.
// TestS8LA29 is shared with GL-16, which adds its own subtests (the custom-value descriptions).

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// liveRig is a fixture whose bundle root holds the test binary as fakeServerExe.
type liveRig struct {
	f      *fixture
	port   int
	events string
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

func newLiveRig(t *testing.T, mode string) *liveRig {
	t.Helper()
	f := newFixture(t, true)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(f.root, fakeServerExe), b, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(f.root, "fake_server_mode.txt"), []byte(mode), 0o644); err != nil {
		t.Fatal(err)
	}
	// The real listener table and console (both read only): the owner of a port is whoever
	// listens on it, and a console stop request reaches the child only when a console exists.
	f.l.os.portOwners = osPortOwners
	f.l.os.console = osConsoleAvailable
	r := &liveRig{f: f, port: freePort(t), events: filepath.Join(f.root, "fake_server_events.log")}
	t.Cleanup(func() { // never leave a fake child behind (the job object ends it with the handle too)
		if f.l.activeSession() {
			if ch, err := f.l.Stop(); err == nil {
				for range ch {
				}
			}
		}
	})
	return r
}

func (r *liveRig) req() Request {
	return Request{ModelPath: r.f.modelPath, Settings: map[string]string{"port": strconv.Itoa(r.port), "warmup": "on"}}
}

// start runs one start to its final event and returns it with every event.
func (r *liveRig) start(t *testing.T, l *Launcher) (StartEvent, []StartEvent) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	ch, err := l.Start(ctx, r.req())
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	var evs []StartEvent
	var final StartEvent
	for ev := range ch {
		evs = append(evs, ev)
		if ev.Final {
			final = ev
		}
	}
	return final, evs
}

func (r *liveRig) stop(t *testing.T) StopEvent {
	t.Helper()
	ch, err := r.f.l.Stop()
	if err != nil {
		t.Fatalf("stop: %v", err)
	}
	var final StopEvent
	for ev := range ch {
		if ev.Final {
			final = ev
		}
	}
	return final
}

// eventLines is the fake child's event log.
func (r *liveRig) eventLines(t *testing.T) []string {
	t.Helper()
	b, err := os.ReadFile(r.events)
	if err != nil {
		return nil
	}
	return strings.Split(strings.TrimSpace(strings.ReplaceAll(string(b), "\r\n", "\n")), "\n")
}

func countPrefix(lines []string, p string) int {
	n := 0
	for _, l := range lines {
		if strings.HasPrefix(l, p) {
			n++
		}
	}
	return n
}

func spawned(evs []StartEvent) bool {
	for _, e := range evs {
		if e.Step == StepSpawn {
			return true
		}
	}
	return false
}

// diagRecords reads the launcher's diagnostic file (the log): kind -> records.
func diagRecords(t *testing.T, l *Launcher) map[string][]map[string]any {
	t.Helper()
	f, err := os.Open(l.diag.Path())
	if err != nil {
		t.Fatalf("log file: %v", err)
	}
	defer f.Close()
	out := map[string][]map[string]any{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var rec struct {
			TS   string         `json:"ts"`
			Kind string         `json:"kind"`
			Data map[string]any `json:"data"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(sc.Text())), &rec) == nil {
			d := rec.Data
			if d == nil {
				d = map[string]any{}
			}
			d["ts"] = rec.TS
			out[rec.Kind] = append(out[rec.Kind], d)
		}
	}
	return out
}

// secondLauncher is another launcher over the same catalog, bundle and model, with the same
// lock names (the test prefix), as a second MoE-Direct-launcher process would be.
func secondLauncher(t *testing.T, f *fixture) *Launcher {
	t.Helper()
	l2, err := New(Options{CatalogPath: f.catalogPath, BundleRoot: f.root, StateDir: filepath.Join(f.dir, "state2")})
	if err != nil {
		t.Fatal(err)
	}
	l2.lockPrefix = f.l.lockPrefix
	l2.os = f.l.os
	return l2
}

// busyPort: a process this launcher did not start (the test process, here as a stale server
// that answers /health) listens on the start's port. The start fails before ready with the
// owner named (process name and PID), exit 5, and the owner keeps running.
func busyPort(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, `{"status":"ok"}`) })}
	go srv.Serve(ln)
	defer srv.Close()
	r := newLiveRig(t, "events,break_exit0")
	r.port = ln.Addr().(*net.TCPAddr).Port
	began := time.Now()
	final, _ := r.start(t, r.f.l)
	// fail_server_start; fail_teardown when the own child had to be forced (no console)
	if final.Ready || (final.Status != StatusFailServerStart && final.Status != StatusFailTeardown) || final.ExitCode != 5 {
		t.Fatalf("busy port start: %+v", final)
	}
	le, ok := AsLaunchError(final.Err)
	if !ok || le.Port == nil {
		t.Fatalf("no port owner on the failure: %v", final.Err)
	}
	exe, _ := os.Executable()
	if le.Port.PID != os.Getpid() || le.Port.Port != r.port || !strings.EqualFold(le.Port.Name, filepath.Base(exe)) {
		t.Errorf("owner %+v, want pid %d name %s port %d", le.Port, os.Getpid(), filepath.Base(exe), r.port)
	}
	if !strings.Contains(le.Reason, "did not start; it is left running") {
		t.Errorf("reason %q", le.Reason)
	}
	if final.Teardown == nil || !final.Teardown.ChildGone || !final.Teardown.ListenerGone {
		t.Errorf("own child not gone: %+v", final.Teardown)
	}
	if d := time.Since(began); d > 20*time.Second {
		t.Errorf("the teardown waited %v for the owner's listener", d)
	}
	// The owner was shown only: it still answers.
	resp, err := http.Get("http://" + ln.Addr().String() + "/health")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("the port owner was ended or disturbed: %v", err)
	}
	resp.Body.Close()
}

// S8-LA-18 (rule 6, F-LS 4.3; conflict row 6; W7 duties 7 and 34; U-32 (ii)).
func TestS8LA18(t *testing.T) {
	t.Run("stop before start exits 2", func(t *testing.T) {
		f := newFixture(t, true)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		ch, err := f.l.Start(ctx, Request{ModelPath: f.modelPath})
		if err != nil {
			t.Fatal(err)
		}
		var final StartEvent
		var evs []StartEvent
		for ev := range ch {
			evs = append(evs, ev)
			if ev.Final {
				final = ev
			}
		}
		if final.Status != StatusCancelledUser || final.ExitCode != 2 || final.Teardown != nil || spawned(evs) {
			t.Errorf("cancelled start: %+v", final)
		}
		// The argument-driven stop action (-stop): the locks are probed, nothing is started.
		err = f.l.StopBeforeStart(context.Background(), Request{ModelPath: f.modelPath})
		if StatusOf(err) != StatusCancelledUser || ExitCodeOf(err) != 2 || !strings.Contains(err.Error(), "stop requested before start") {
			t.Errorf("stop action: %v", err)
		}
		// With no output folder yet the stop creates none.
		if err := os.RemoveAll(f.outDir); err != nil {
			t.Fatal(err)
		}
		err = f.l.StopBeforeStart(context.Background(), Request{ModelPath: f.modelPath})
		if StatusOf(err) != StatusCancelledUser {
			t.Errorf("stop action without an output folder: %v", err)
		}
		if _, err := os.Stat(f.outDir); err == nil {
			t.Error("the stop action created the output folder")
		}
	})

	t.Run("stop procedure codes", func(t *testing.T) {
		// CTRL_BREAK, the child exits 0 in the grace wait: ok, exit 0, no force.
		c := &fakeChild{pid: 200, newGroup: true, onCtrl: exitWith(0)}
		r, _ := runStop(c, true, noListener)
		st, _ := teardownStatus(StatusOK, r, true)
		if c.ctrlCalls != 1 || c.killCalls != 0 || st != StatusOK || ExitCode(st) != 0 {
			t.Errorf("clean: ctrl %d kill %d %s", c.ctrlCalls, c.killCalls, st)
		}
		if e := ClassifyStop(RunEnd{Status: st, Teardown: &r}, true, 200); e.Class != StopClean || e.Code != 0 || *e.ChildExit != 0 {
			t.Errorf("clean event %+v", e)
		}
		// The child is still running after the wait: the process tree is ended, exit 5.
		c = &fakeChild{pid: 201, newGroup: true, onKill: exitWith(1)}
		r, steps := runStop(c, true, noListener)
		st, _ = teardownStatus(StatusOK, r, true)
		if c.killCalls != 1 || !r.TaskkillUsed || st != StatusFailTeardown || ExitCode(st) != 5 ||
			strings.Join(steps, ",") != "ctrl_break,grace_wait,force_kill,listener" || c.waits[0] != 60*time.Second {
			t.Errorf("forced: kill %d %+v %s %v %v", c.killCalls, r, st, steps, c.waits)
		}
		if e := ClassifyStop(RunEnd{Status: st, Teardown: &r}, true, 201); e.Class != StopForced || e.Code != 5 {
			t.Errorf("forced event %+v", e)
		}
		// Exit 0 only without the forced end: a child that exits 0 only after the kill is still 5.
		c = &fakeChild{pid: 202, newGroup: true, onKill: exitWith(0)}
		r, _ = runStop(c, true, noListener)
		if st, _ = teardownStatus(StatusOK, r, true); st != StatusFailTeardown || ExitCode(st) != 5 {
			t.Errorf("exit 0 after a forced end: %s", st)
		}
	})

	t.Run("live child: break, wait, exit 0; one stop line; a second launcher starts nothing", func(t *testing.T) {
		r := newLiveRig(t, "events,answer,break_exit0")
		final, _ := r.start(t, r.f.l)
		if !final.Ready {
			t.Fatalf("not ready: %+v", final)
		}
		// W7 duty 7: a second launcher while this one serves is refused at the locks, starts
		// nothing (fail_instance_lock, exit 3); the wire shows «another launcher is already running».
		l2 := secondLauncher(t, r.f)
		if _, err := l2.Preflight(context.Background(), r.req()); StatusOf(err) != StatusFailInstanceLock || ExitCodeOf(err) != 3 {
			t.Errorf("second launcher's check: %v", err)
		}
		f2, evs2 := r.start(t, l2)
		if f2.Ready || f2.Status != StatusFailInstanceLock || f2.ExitCode != 3 || spawned(evs2) {
			t.Errorf("second launcher's start: %+v", f2)
		}
		if err := l2.StopBeforeStart(context.Background(), r.req()); StatusOf(err) != StatusFailInstanceLock {
			t.Errorf("second launcher's stop action: %v", err)
		}
		if n := countPrefix(r.eventLines(t), "start pid="); n != 1 {
			t.Errorf("children started: %d, want 1", n)
		}

		st := r.stop(t)
		if osConsoleAvailable() {
			if st.Status != StatusOK || st.ExitCode != 0 || !st.Teardown.CtrlSent || st.Teardown.TaskkillUsed ||
				st.Teardown.ExitCode == nil || *st.Teardown.ExitCode != 0 {
				t.Errorf("clean stop: %+v %+v", st, st.Teardown)
			}
			if countPrefix(r.eventLines(t), "ctrl_break") != 1 {
				t.Errorf("the child got no console break: %v", r.eventLines(t))
			}
		} else { // no console: the break cannot be delivered, so the forced end and 5
			t.Logf("no console: the forced path is the expected result")
			if st.Status != StatusFailTeardown || st.ExitCode != 5 || !st.Teardown.TaskkillUsed {
				t.Errorf("stop without a console: %+v", st)
			}
		}
		if !st.Teardown.ChildGone {
			t.Errorf("child not gone: %+v", st.Teardown)
		}
		evs := r.f.l.StopEvents()
		if len(evs) != 1 || st.Event == nil || evs[0].Code != st.ExitCode || evs[0].ChildExit == nil ||
			!strings.Contains(evs[0].Line(), "server stop · ") || !strings.Contains(evs[0].Line(), "launcher code "+strconv.Itoa(st.ExitCode)) {
			t.Errorf("stop events %+v", evs)
		}
		recs := diagRecords(t, r.f.l)["SERVER_STOP"]
		if len(recs) != 1 || recs[0]["ts"] == "" || recs[0]["class"] == nil || recs[0]["child_exit"] == nil ||
			int(recs[0]["code"].(float64)) != st.ExitCode {
			t.Errorf("one log line per server stop with time, result and the child's code: %v", recs)
		}
	})

	t.Run("live child ignoring the break: tree ended, exit 5", func(t *testing.T) {
		saved := psStopTimings
		psStopTimings = stopTimings{Grace: 1500 * time.Millisecond, KillWait: 15 * time.Second, ListenerGone: 10 * time.Second, ListenerPoll: 100 * time.Millisecond}
		t.Cleanup(func() { psStopTimings = saved })
		r := newLiveRig(t, "events,answer,ignore_break")
		if final, _ := r.start(t, r.f.l); !final.Ready {
			t.Fatalf("not ready: %+v", final)
		}
		st := r.stop(t)
		if st.Status != StatusFailTeardown || st.ExitCode != 5 || !st.Teardown.TaskkillUsed || !st.Teardown.ChildGone {
			t.Errorf("forced stop: %+v %+v", st, st.Teardown)
		}
		if evs := r.f.l.StopEvents(); len(evs) != 1 || evs[0].Class != StopForced || evs[0].Code != 5 {
			t.Errorf("stop event %+v", evs)
		}
	})

	t.Run("server ends while serving: its code, kept as the last stop's", func(t *testing.T) {
		r := newLiveRig(t, "events,answer,die")
		final, _ := r.start(t, r.f.l)
		if !final.Ready {
			t.Fatalf("not ready: %+v", final)
		}
		if resp, err := http.Post(final.Address+"/die", "text/plain", nil); err == nil {
			resp.Body.Close()
		}
		select {
		case <-r.f.l.Done():
		case <-time.After(30 * time.Second):
			t.Fatal("the end of the server was not noticed")
		}
		end, ok := r.f.l.LastEnd()
		if !ok || end.Status != StatusFailRuntimeExit || end.ExitCode != 5 || end.Event == nil ||
			end.Event.ChildExit == nil || *end.Event.ChildExit != 3 {
			t.Fatalf("end %+v", end)
		}
		st, err := r.f.l.Status(context.Background())
		if err != nil || st.Running || st.End == nil || st.End.ExitCode != 5 {
			t.Errorf("status after the end: %+v %v", st, err)
		}
		// A quit after it (wire's Shutdown asks Stop) keeps the last stop's code (U-32 (ii)).
		if sf := r.stop(t); sf.ExitCode != 5 || sf.Status != StatusFailRuntimeExit {
			t.Errorf("stop after the end: %+v", sf)
		}
		if evs := r.f.l.StopEvents(); len(evs) != 1 {
			t.Errorf("one stop event per server stop: %+v", evs)
		}
	})

	t.Run("busy port shows its owner and ends nothing", busyPort)
}

// S8-LA-21 (review item 1; design H12): while a server runs the start action starts nothing;
// after a stop, the start action starts the same model again with a new child.
func TestS8LA21(t *testing.T) {
	r := newLiveRig(t, "events,answer,break_exit0")
	first, _ := r.start(t, r.f.l)
	if !first.Ready {
		t.Fatalf("first start: %+v", first)
	}
	if _, err := r.f.l.Start(context.Background(), r.req()); !errors.Is(err, ErrBusy) {
		t.Errorf("start while serving: %v, want ErrBusy", err)
	}
	r.stop(t)
	second, _ := r.start(t, r.f.l)
	if !second.Ready || second.PID == first.PID || second.PID == 0 {
		t.Fatalf("second start: %+v (first pid %d)", second, first.PID)
	}
	r.stop(t)
	if n := countPrefix(r.eventLines(t), "start pid="); n != 2 {
		t.Errorf("child count %d, want 2", n)
	}
	if evs := r.f.l.StopEvents(); len(evs) != 2 {
		t.Errorf("stop events %d, want 2", len(evs))
	}
}

// jobOf is the job object of the running child and the PIDs inside it.
func jobOf(t *testing.T, l *Launcher) (uint32, []int) {
	t.Helper()
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	s.mu.Lock()
	job := s.child.job
	s.mu.Unlock()
	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	if err := windows.QueryInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)), nil); err != nil {
		t.Fatalf("job limits: %v", err)
	}
	buf := make([]byte, 8+8*64)
	if err := windows.QueryInformationJobObject(job, windows.JobObjectBasicProcessIdList,
		uintptr(unsafe.Pointer(&buf[0])), uint32(len(buf)), nil); err != nil {
		t.Fatalf("job process list: %v", err)
	}
	n := int(binary.LittleEndian.Uint32(buf[4:8]))
	var pids []int
	for i := 0; i < n && 8+8*i+8 <= len(buf); i++ {
		pids = append(pids, int(binary.LittleEndian.Uint64(buf[8+8*i:])))
	}
	return info.BasicLimitInformation.LimitFlags, pids
}

// psStartOrder is the PowerShell main flow's order of a start, as recorded in its
// coordinates: the bundle gate (Assert-BundleIntegrity PS:2187), the catalog, identify,
// artifacts and locks (PS:13840-14000), the sweep, the config and the sizing (PS:14000-14269),
// Start-OwnedChild (PS:14310-14323), Wait-ForServerReady (PS:12180-12201) and the launcher
// warmup after ready with the engine warmup off (Invoke-ReadyWarmup PS:14578; --no-warmup).
var psStartOrder = []string{StepBundle, StepCatalog, StepIdentify, StepArtifacts, StepLocks, StepSweep, StepConfig,
	StepSizing, StepSpawn, StepReady, StepWarmup}

// S8-LA-32 (LFS Q5.27-Q5.29, Q5.42, Q5.46): the child starts inside a job object, ready =
// /health ready, the warmup request follows ready with the engine warmup off; a busy port and
// the other start failures give the failure (the screen's page is wire's and ui's part).
func TestS8LA32(t *testing.T) {
	t.Run("job object, ready by /health, warmup after ready", func(t *testing.T) {
		r := newLiveRig(t, "events,answer,break_exit0")
		final, evs := r.start(t, r.f.l)
		if !final.Ready || final.Warmup == nil || !final.Warmup.OK {
			t.Fatalf("start: %+v warmup %+v", final, final.Warmup)
		}
		flags, pids := jobOf(t, r.f.l)
		if flags&jobLimitKillOnJobClose == 0 {
			t.Errorf("job limit flags %#x lack KILL_ON_JOB_CLOSE", flags)
		}
		inJob := false
		for _, p := range pids {
			inJob = inJob || p == final.PID
		}
		if !inJob {
			t.Errorf("child %d not in the job %v", final.PID, pids)
		}
		// The event sequence against the recorded one.
		var seq []string
		readyDone, warmupRun := -1, -1
		for i, e := range evs {
			if e.State == StepRunning && (len(seq) == 0 || seq[len(seq)-1] != e.Step) && !e.Final {
				seq = append(seq, e.Step)
			}
			if e.Step == StepReady && e.State == StepDone && readyDone < 0 {
				readyDone = i
			}
			if e.Step == StepWarmup && e.State == StepRunning && warmupRun < 0 {
				warmupRun = i
			}
		}
		if strings.Join(seq, ",") != strings.Join(psStartOrder, ",") {
			t.Errorf("steps %v, recorded %v", seq, psStartOrder)
		}
		if readyDone < 0 || warmupRun < readyDone {
			t.Errorf("warmup (event %d) before ready (event %d)", warmupRun, readyDone)
		}
		lines := r.eventLines(t)
		if len(lines) < 3 || !strings.HasPrefix(lines[0], "start pid=") || !strings.Contains(lines[0], " --no-warmup") {
			t.Fatalf("child events %v", lines)
		}
		firstHealth, firstChat := -1, -1
		for i, l := range lines {
			if l == "GET /health" && firstHealth < 0 {
				firstHealth = i
			}
			if l == "POST /v1/chat/completions" && firstChat < 0 {
				firstChat = i
			}
		}
		if firstHealth < 0 || firstChat < firstHealth || countPrefix(lines, "POST /v1/chat/completions") != 1 {
			t.Errorf("request order %v", lines)
		}
		r.stop(t)
	})
	t.Run("busy port: the failure with the owner process", busyPort)
	t.Run("other start failures: the failure without an owner", func(t *testing.T) {
		r := newLiveRig(t, "events,exit_early")
		final, _ := r.start(t, r.f.l)
		le, _ := AsLaunchError(final.Err)
		if final.Ready || final.Status != StatusFailServerStart || final.ExitCode != 5 || le == nil || le.Port != nil ||
			!strings.Contains(le.Reason, "exited before ready (code=1") {
			t.Errorf("early exit: %+v %v", final, final.Err)
		}
	})
}
