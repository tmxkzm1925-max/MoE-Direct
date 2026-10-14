package core

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Ready-wait timings (PS:1222, :1225, :12069).
type readyTimings struct {
	Timeout       time.Duration // READY_TIMEOUT_S
	Poll          time.Duration // HEALTH_POLL_MS
	HealthTimeout time.Duration // Invoke-WebRequest -TimeoutSec 5
}

var psReadyTimings = readyTimings{Timeout: 900 * time.Second, Poll: 500 * time.Millisecond, HealthTimeout: 5 * time.Second}

// enginePolicyAnchor is ENGINE_POLICY_ANCHOR (PS:226).
const enginePolicyAnchor = "[moe-direct] startup_reject=engine_policy_gate"

// StepState is the state of one step in an event.
type StepState uint8

const (
	StepRunning StepState = iota
	StepDone
	StepFailed
)

// StartEvent is one report of a start. Events come in order on the channel Start returns;
// the last one has Final set, and the channel is closed after it.
type StartEvent struct {
	Step    string
	State   StepState
	Detail  string
	Elapsed time.Duration // since the server child was started (decision ③); 0 before
	// Progress is decision ③ for the load bar (meaningful from StepReady on).
	Progress Progress
	PID      int
	Address  string
	ChildMem ProcessMemory // the child's memory so far (StepReady ticks)
	LastMem  int64         // working set at ready last time for this key; 0 when none
	// Preflight is set once, on the event that closes StepSizing, and on the final event.
	Preflight *PreflightResult
	Notices   []string
	Ready     bool
	Final     bool
	// On a failed start: the error (a *LaunchError or a *NotPortedError), the status after
	// teardown and its exit code; Teardown is set when a child had been started.
	Err      error
	Status   Status
	ExitCode int
	Teardown *TeardownResult
	ErrLog   string
	// FIX2-CORE: the bundle check the start waited for (on the StepBundle events and the final
	// one), the warmup and «작동 확인» results (StepWarmup / StepSmoke done events and the final
	// one), and the stop event when a server child had been started and the start failed.
	Bundle   *BundleResult
	Warmup   *WarmupResult
	Smoke    *SmokeResult
	StopInfo *ServerStopEvent
	// GL-21: the engine's prefetch values read back at ready (F-PF §8.5), on the StepReady
	// done event and the final event of a ready start.
	Engine *EnginePrefetch
	// GL-15: the warm-start status text of this start (eligible, restored(N tokens), cold(reason),
	// off(user), off(mode)), on the event that closes StepSizing and on the warmup's events after
	// the restore; empty elsewhere. H15 shows «warm start» from it, with no clock time (S8-LA-17).
	Warm string
}

// session is the one server this Launcher owns.
type session struct {
	mu        sync.Mutex
	req       Request
	pre       *PreflightResult
	child     *winChild
	locks     *heldLocks
	spawnedAt time.Time
	readyAt   time.Time
	ready     bool
	stopping  bool
	ended     bool
	end       RunEnd
	done      chan struct{}
	// GOL-FIX1: the parsed stderr (review items 5, 9, 18, 23, 29) and the largest working set
	// seen while serving (item 13).
	log  *serverLog
	peak int64
	// GL-05: the P3 reading of c_dev and U was taken or tried (once per run)
	p3Done bool
	// GL-21: the engine's prefetch values after ready (F-PF §8.5)
	engine *EnginePrefetch
}

// RunEnd is how a server's run ended.
type RunEnd struct {
	Status   Status
	ExitCode int
	Reason   string
	Teardown *TeardownResult
	At       time.Time
	// Event is conflict 6's launcher event of this stop (nil when no server child had started).
	Event *ServerStopEvent
}

func (l *Launcher) activeSession() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.sess == nil {
		return false
	}
	l.sess.mu.Lock()
	defer l.sess.mu.Unlock()
	return !l.sess.ended
}

// ErrBusy is returned by Start while this Launcher already runs a server.
var ErrBusy = errors.New("a server started by this launcher is still running")

// Start begins a start and returns at once; the steps arrive on the channel. ctx cancels the
// start before ready (the cancelled_user path); after ready the server keeps running until
// Stop, or until it exits on its own (fail_runtime_exit, see Status).
func (l *Launcher) Start(ctx context.Context, req Request) (<-chan StartEvent, error) {
	l.mu.Lock()
	if l.sess != nil {
		l.sess.mu.Lock()
		busy := !l.sess.ended
		l.sess.mu.Unlock()
		if busy {
			l.mu.Unlock()
			return nil, ErrBusy
		}
	}
	s := &session{req: req, done: make(chan struct{})}
	l.sess = s
	l.mu.Unlock()
	ch := make(chan StartEvent, 64)
	go l.runStart(ctx, s, ch)
	return ch, nil
}

func (l *Launcher) runStart(ctx context.Context, s *session, ch chan<- StartEvent) {
	defer close(ch)
	var cur string
	emit := func(e StartEvent) {
		select {
		case ch <- e:
		default:
			if e.Final || e.Step != StepReady || e.State != StepRunning {
				ch <- e // never drop a step change; ready ticks may be dropped when the reader lags
			}
		}
	}
	step := func(name string) {
		if cur != "" {
			emit(StartEvent{Step: cur, State: StepDone})
		}
		cur = name
		emit(StartEvent{Step: name, State: StepRunning})
	}
	// smokeEnd is the failed check run a fail_smoke end carries on its final event (GL-11: the
	// failure's one line at the start verdict and its items in 4 Log come from it).
	var smokeEnd *SmokeResult
	finish := func(err error, status Status, td *TeardownResult, pre *PreflightResult, errLog string) {
		code := ExitCode(status)
		if cur != "" {
			emit(StartEvent{Step: cur, State: StepFailed, Detail: err.Error()})
		}
		end := RunEnd{Status: status, ExitCode: code, Reason: err.Error(), Teardown: td, At: l.os.now()}
		if td != nil {
			e := l.recordStop(s, end)
			end.Event = &e
		}
		s.mu.Lock()
		s.ended = true
		s.end = end
		s.mu.Unlock()
		close(s.done)
		emit(StartEvent{Final: true, Err: err, Status: status, ExitCode: code, Teardown: td, Preflight: pre, ErrLog: errLog,
			StopInfo: end.Event, Smoke: smokeEnd})
	}
	if ctx.Err() != nil {
		finish(stop(StatusCancelledUser, "start cancelled before it began"), StatusCancelledUser, nil, nil, "")
		return
	}
	// Conflict 18-2: a start waits for the bundle check (started at launch, or here).
	step(StepBundle)
	br, berr := l.WaitBundleCheck(ctx)
	if berr != nil {
		finish(stop(StatusCancelledUser, "start cancelled while the bundle check ran"), StatusCancelledUser, nil, nil, "")
		return
	}
	if err := br.Err(); err != nil {
		emit(StartEvent{Step: StepBundle, State: StepFailed, Detail: err.Error(), Bundle: &br})
		cur = ""
		finish(err, StatusFailGateBundle, nil, nil, "")
		return
	}
	emit(StartEvent{Step: StepBundle, State: StepDone, Bundle: &br, Detail: br.State.String()})
	cur = ""
	pr, err := l.prepare(ctx, s.req, modeStart, step)
	if err != nil {
		st := StatusFailServerStart
		st = StatusOf(err)
		if ctx.Err() != nil {
			st, err = StatusCancelledUser, stop(StatusCancelledUser, "start cancelled: %v", err)
		}
		finish(err, st, nil, pr.res, "")
		return
	}
	pre := pr.res
	s.mu.Lock()
	s.pre, s.locks = pre, pr.locks
	s.mu.Unlock()
	emit(StartEvent{Step: StepSizing, State: StepDone, Preflight: pre, Notices: pre.Notices, Warm: pre.Warm.Text()})
	cur = ""

	// Start-OwnedChild (main flow PS:14310-14323).
	step(StepSpawn)
	cfg := pre.Config
	exe := filepath.Join(l.opt.BundleRoot, pr.cat.Runtime.ServerExe)
	logDir := l.LogDir()
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		pr.locks.release()
		finish(stop(StatusFailServerStart, "server start failed: log directory: %v", err), StatusFailServerStart, nil, pre, "")
		return
	}
	stamp := l.os.now().UTC().Format("20060102T150405Z")
	outLog, errLog := l.ServerLogPaths(stamp)
	env := explicitEnv(cfg.Env, os.Getenv)
	l.diag.write("ENV_BLOCK", map[string]any{"role": "server", "mode": "explicit", "count": len(env)})
	child, err := startOwnedChild(exe, cfg.Argv, env, l.opt.BundleRoot, outLog, errLog)
	if err != nil {
		pr.locks.release()
		finish(stop(StatusFailServerStart, "server start failed: %v", err), StatusFailServerStart, nil, pre, errLog)
		return
	}
	spawned := l.os.now()
	s.mu.Lock()
	s.child, s.spawnedAt = child, spawned
	s.log = newServerLog(errLog, spawned)
	s.mu.Unlock()
	l.diag.write("CHILD_START", map[string]any{"role": "server", "pid": child.PID(), "exe": exe, "err_log": errLog})
	emit(StartEvent{Step: StepSpawn, State: StepDone, PID: child.PID(), Address: cfg.Address()})
	cur = ""

	// Wait-ForServerReady (PS:12180-12201) with decision ③ ticks.
	step(StepReady)
	var last time.Duration
	var lastMem int64
	hasLast := false
	if rec, ok := l.HistoryFor(pre.SettingsKey); ok && rec.ReadyMillis > 0 {
		last, hasLast, lastMem = rec.ReadyDuration(), true, rec.WorkingSetBytes
	}
	// GL-07 (S8-LA-14, S8-LA-16): the load bar's expectation for this model and settings, in the
	// launcher log (none on a first start: elapsed seconds only).
	l.diag.write("LOAD_EXPECT", map[string]any{"has_last": hasLast, "expect_ms": last.Milliseconds(),
		"settings": pre.SettingsLine})
	tick := func(elapsed time.Duration) {
		m, _ := child.Memory()
		emit(StartEvent{Step: StepReady, State: StepRunning, Elapsed: elapsed, PID: child.PID(), Address: cfg.Address(),
			Progress: LoadProgress(elapsed, last, hasLast, false), ChildMem: m, LastMem: lastMem})
	}
	werr := waitForServerReady(ctx, child, cfg.Host, cfg.Port, errLog, l.os.portOwners, psReadyTimings, l.os.now, time.Sleep, tick)
	if werr != nil {
		le, _ := AsLaunchError(werr)
		if h := l.portHolder(le, cfg.Port, child.PID()); h != nil { // GL-08: a busy port names its owner
			le.Port = h
			le.Reason = fmt.Sprintf("port %d is in use by another process (%s, pid %d) that this launcher did not start; it is left running: %s",
				h.Port, h.Name, h.PID, le.Reason)
		}
		env := l.stopEnv(nil)
		if le != nil && le.Port != nil { // the teardown waits for its own child's listener, not the owner's
			env.portOwners = withoutPID(env.portOwners, le.Port.PID)
		}
		td := stopOwnedChildGraceful(child, cfg.Port, env)
		child.Close()
		l.warmTeardownCleanup(pre.Warm)
		pr.locks.release()
		st, why := teardownStatus(le.Status, td, false)
		if why != "" {
			l.diag.write("TEARDOWN_PROMOTED", map[string]any{"from": le.Status, "reason": why})
		}
		finish(werr, st, &td, pre, errLog)
		return
	}
	readyAt := l.os.now()
	took := readyAt.Sub(spawned)
	mem, memErr := child.Memory()
	if err := l.recordRun(RunRecord{Key: pre.SettingsKey, ProfileID: pre.Profile.ID, ModelPath: pre.ModelPath,
		SettingsLine: pre.SettingsLine, ReadyMillis: took.Milliseconds(), WorkingSetBytes: mem.WorkingSetBytes,
		PrivateBytes: mem.PrivateBytes, At: readyAt.UTC().Format(time.RFC3339), Start: &pre.Start}); err != nil {
		l.diag.write("HISTORY_WRITE_FAILED", map[string]any{"reason": err.Error()})
	}
	// GL-05: the P1 reading of c_dev and U (F-BU B.7), at ready, before the warmup's request
	l.recordResiduals(pre, AtLoad, child.PID(), mem, memErr, runCoord(readyAt, child.PID()))
	// GL-21: the first metrics line after ready, the engine's header (F-PF §8.5; PFP:302-309)
	eng := ReadEnginePrefetch(cfg.Env[EnvMetrics])
	s.mu.Lock()
	s.ready, s.readyAt = true, readyAt
	s.engine = &eng
	s.mu.Unlock()
	l.diag.write("READY", map[string]any{"pid": child.PID(), "port": cfg.Port, "ready_ms": took.Milliseconds()})
	l.diag.write("PREFETCH_ENGINE", map[string]any{"read": eng.Read, "prefetch_on": eng.On, "prefetch_k": eng.K,
		"prefetch_n": eng.N, "prefetch_off_reason": eng.OffReason, "why": eng.Why, "line": PostReadyLine(eng, pre.Start.PrefetchEcho)})
	emit(StartEvent{Step: StepReady, State: StepDone, Elapsed: took, PID: child.PID(), Address: cfg.Address(),
		Progress: LoadProgress(took, last, hasLast, true), ChildMem: mem, LastMem: lastMem, Engine: &eng})
	cur = ""

	// GL-15: the warm-start restore after ready and before the warmup (PS:14576-14584), its
	// ladder, and rung 4's recovery restart (Invoke-WarmstartRecoveryRestart PS:10555).
	restored := false
	if w := pre.Warm; w != nil && w.Eligible {
		ok, recovery := l.warmRestore(context.Background(), w, cfg)
		restored = ok
		s.mu.Lock()
		stopping := s.stopping
		s.mu.Unlock()
		if recovery && stopping { // a stop arrived during the restore: its teardown owns the end
			<-s.done
			s.mu.Lock()
			end := s.end
			s.mu.Unlock()
			emit(StartEvent{Final: true, Status: end.Status, ExitCode: end.ExitCode, Err: errors.New("the server was stopped during the warm-start restore"),
				Teardown: end.Teardown, StopInfo: end.Event, Preflight: pre, ErrLog: errLog, Bundle: &br})
			return
		}
		if recovery {
			nc, ncfg, nerr, td, err := l.warmRecoveryRestart(ctx, s, w, exe, outLog, errLog, tick)
			if err != nil {
				pr.locks.release()
				le, _ := AsLaunchError(err)
				st := le.Status
				if td != nil {
					if s2, why := teardownStatus(le.Status, *td, false); why != "" && le.Status != StatusFailTeardown {
						l.diag.write("TEARDOWN_PROMOTED", map[string]any{"from": le.Status, "reason": why})
						st = s2
					}
				}
				l.warmTeardownCleanup(w)
				finish(err, st, td, pre, nerr)
				return
			}
			child, cfg, errLog = nc, ncfg, nerr
		}
	}

	// The launcher warmup (Invoke-ReadyWarmup PS:14578), skipped when the «작동 확인» check runs
	// at this start (PowerShell's -Smoke forces it off) and after a restored slot (WARMSTART A-3).
	smokeDue := l.SmokeDue(s.req.Smoke, pre.ModelPath, pre.RepackMode)
	forced := ""
	if smokeDue {
		forced = warmupForcedReason
	}
	step(StepWarmup)
	cwd, _ := os.Getwd()
	wu := restoredWarmup(pre.Warmup)
	if !restored {
		wu = runWarmup(ctx, cfg.Address(), pre.Warmup, forced, cwd)
	}
	l.diag.write("WARMUP", map[string]any{"mode": wu.Mode, "ok": wu.OK, "skipped": wu.Skipped, "skip_reason": wu.SkipReason,
		"degraded": wu.Degraded, "reason": wu.Reason, "file": wu.File, "tokens": wu.Tokens, "ms": wu.Elapsed.Milliseconds()})
	emit(StartEvent{Step: StepWarmup, State: StepDone, Detail: wu.Line(), Warmup: &wu, PID: child.PID(), Address: cfg.Address(),
		Warm: pre.Warm.Text()})
	cur = ""
	// The server is ready, so Stop may have run during the warmup: then its teardown owns the
	// end, and the start reports that end instead of ready.
	stoppedDuring := func(what string) bool {
		s.mu.Lock()
		stopping := s.stopping
		s.mu.Unlock()
		if !stopping {
			return false
		}
		<-s.done
		s.mu.Lock()
		end := s.end
		s.mu.Unlock()
		emit(StartEvent{Final: true, Status: end.Status, ExitCode: end.ExitCode, Err: errors.New("the server was stopped during the " + what),
			Teardown: end.Teardown, StopInfo: end.Event, Preflight: pre, ErrLog: errLog, Warmup: &wu, Bundle: &br})
		return true
	}
	if stoppedDuring("warmup") {
		return
	}

	// Conflict 18-1: the «작동 확인» check when the setting makes it due; a failure ends the
	// start as fail_smoke (exit 6) after the normal teardown.
	var sm *SmokeResult
	if smokeDue {
		step(StepSmoke)
		r := runSmokeFull(ctx, cfg.Address(), func() []string { return readChildStderrComplete(errLog) }, psSmokeTimings,
			l.smokeContextFor(pre, cfg))
		sm = &r
		l.logSmoke(r)
		// UX-R12b (5): the English name «check run»; the screens word it by the table
		if stoppedDuring("check run") {
			return
		}
		if !r.OK {
			emit(StartEvent{Step: StepSmoke, State: StepFailed, Detail: r.Line(), Smoke: sm})
			cur = ""
			s.mu.Lock()
			if s.stopping {
				s.mu.Unlock()
				stoppedDuring("check run")
				return
			}
			s.stopping = true
			s.mu.Unlock()
			l.finishRun(s)
			td := stopOwnedChildGraceful(child, cfg.Port, l.stopEnv(nil))
			child.Close()
			l.warmTeardownCleanup(pre.Warm)
			pr.locks.release()
			st, why := teardownStatus(StatusFailSmoke, td, true)
			if why != "" {
				l.diag.write("TEARDOWN_PROMOTED", map[string]any{"from": StatusFailSmoke, "reason": why})
			}
			smokeEnd = sm
			finish(stop(StatusFailSmoke, "smoke assertions failed: %s", strings.Join(r.Failures, ", ")), st, &td, pre, errLog)
			return
		}
		l.markSmoked(pre.ModelPath, pre.RepackMode)
		emit(StartEvent{Step: StepSmoke, State: StepDone, Detail: r.Line(), Smoke: sm})
		cur = ""
	}
	emit(StartEvent{Final: true, Ready: true, Status: StatusOK, ExitCode: 0, PID: child.PID(), Address: cfg.Address(),
		Preflight: pre, Elapsed: took, ErrLog: errLog, Warmup: &wu, Smoke: sm, Bundle: &br, Engine: &eng})
	go l.watch(s)
}

// notPortedStatus is the failure stage (Set-FailureStage) the PowerShell main flow holds at
// the point where it would run the family: fail_repack before the repack (PS:14090),
// fail_partial_cleanup before the stale removal (PS:14146), fail_model_path before the derive
// plan (PS:13928), fail_custom_args around the overrides and Build-EffectiveConfig
// (PS:13963, :14293). The error itself says what is not ported.
func notPortedStatus(err error) Status {
	var np *NotPortedError
	if !errors.As(err, &np) {
		return StatusFailServerStart
	}
	switch np.Family {
	case FamilyRepack:
		return StatusFailRepack
	case FamilyPartialStale:
		return StatusFailPartialCleanup
	case FamilyArchTemplate:
		return StatusFailModelPath
	case FamilyLauncherWarmup, FamilyPresets, FamilyPrefetchArming:
		return StatusFailCustomArgs
	}
	return StatusFailServerStart
}

// portHolder is the busy port of conflict row 6 (F-LS 4.3; GL-08): a start that failed before
// ready while a process other than its own child listens on its port names that process. The
// launcher only reads the listener table and the process name; it ends nothing it did not start.
func (l *Launcher) portHolder(le *LaunchError, port, child int) *PortHolder {
	if le == nil || le.Status != StatusFailServerStart || le.Load != nil || l.os.portOwners == nil {
		return nil
	}
	owners, err := l.os.portOwners(port)
	if err != nil {
		return nil
	}
	for _, p := range owners {
		if p > 0 && p != child {
			h := &PortHolder{Port: port, PID: p}
			if l.os.procName != nil {
				h.Name = l.os.procName(p)
			}
			if h.Name == "" {
				h.Name = "unknown program"
			}
			return h
		}
	}
	return nil
}

// withoutPID is a listener table without one owner (the busy port's, which the launcher leaves
// running and does not wait for).
func withoutPID(owners func(int) ([]int, error), pid int) func(int) ([]int, error) {
	return func(port int) ([]int, error) {
		o, err := owners(port)
		if err != nil || o == nil {
			return o, err
		}
		out := []int{}
		for _, p := range o {
			if p != pid {
				out = append(out, p)
			}
		}
		return out, nil
	}
}

func (l *Launcher) stopEnv(step func(string)) stopEnv {
	return stopEnv{console: l.os.console(), portOwners: l.os.portOwners, t: psStopTimings,
		now: l.os.now, sleep: time.Sleep, step: step}
}

// readChildStderrComplete is Read-ChildStderrComplete (PS:11913-11943): the complete lines.
func readChildStderrComplete(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	raw := string(b)
	idx := strings.LastIndex(raw, "\n")
	if idx < 0 {
		return nil
	}
	parts := strings.Split(raw[:idx+1], "\n")
	out := make([]string, 0, len(parts)-1)
	for _, p := range parts[:len(parts)-1] {
		out = append(out, strings.TrimSuffix(p, "\r"))
	}
	return out
}

// testEnginePolicyAnchor is Test-EnginePolicyAnchor (PS:11945-11951).
func testEnginePolicyAnchor(errLog string) bool {
	n := 0
	for _, ln := range readChildStderrComplete(errLog) {
		if ln == enginePolicyAnchor {
			n++
		}
	}
	return n == 1
}

type healthResult struct {
	ok, fatal bool
	reason    string
}

// testHealthPidBound is Test-HealthPidBound (PS:12053-12066).
func testHealthPidBound(ctx context.Context, pid int, host string, port int, owners func(int) ([]int, error), timeout time.Duration) healthResult {
	hctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(hctx, http.MethodGet, fmt.Sprintf("http://%s:%d/health", host, port), nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return healthResult{reason: "health request failed: " + err.Error()}
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return healthResult{reason: fmt.Sprintf("health status %d", resp.StatusCode)}
	}
	o, err := owners(port)
	if err != nil || o == nil {
		return healthResult{reason: "listener owner not resolvable (fail-close)"}
	}
	for _, p := range o {
		if p == pid {
			return healthResult{ok: true}
		}
	}
	return healthResult{fatal: true, reason: fmt.Sprintf("listener PID mismatch: owners=%v child=%d", o, pid)}
}

// waitForServerReady is Wait-ForServerReady: cancel -> cancelled_user; child gone ->
// fail_gate_engine_seal (exactly one anchor line) or fail_server_start; 900 s ->
// fail_server_start "ready timeout"; /health 200 with the child owning the listener -> ready;
// a listener owned by another process -> fail_server_start.
func waitForServerReady(ctx context.Context, c ownedChild, host string, port int, errLog string,
	owners func(int) ([]int, error), t readyTimings, now func() time.Time, sleep func(time.Duration),
	tick func(time.Duration)) error {
	t0 := now()
	for {
		if ctx.Err() != nil {
			return stop(StatusCancelledUser, "stop requested before the server was ready")
		}
		if code, exited := c.Exited(); exited {
			if testEnginePolicyAnchor(errLog) {
				return stop(StatusFailGateEngineSeal, "engine refused startup by policy gate (see %s)", errLog)
			}
			// Conflict 8: an out-of-memory load is its own failure reason (still code 5).
			if f, ok := ClassifyLoadFailure(readChildStderrComplete(errLog)); ok {
				le := stop(StatusFailServerStart, "server ran out of memory while loading the model (%s; code=%s): %s; see %s",
					oomDevice(f.Device), FormatExitCode(&code), f.Line, errLog)
				le.Load = &f
				return le
			}
			return stop(StatusFailServerStart, "server child exited before ready (code=%s); see %s", FormatExitCode(&code), errLog)
		}
		el := now().Sub(t0)
		if el > t.Timeout {
			return stop(StatusFailServerStart, "ready timeout")
		}
		h := testHealthPidBound(ctx, c.PID(), host, port, owners, t.HealthTimeout)
		if h.ok {
			return nil
		}
		if h.fatal {
			return stop(StatusFailServerStart, "%s", h.reason)
		}
		if tick != nil {
			tick(now().Sub(t0))
		}
		sleep(t.Poll)
	}
}

// watch is the serve loop's exit check (PS:14551-14556): a server that ends on its own is
// fail_runtime_exit, after the teardown.
func (l *Launcher) watch(s *session) {
	for {
		s.mu.Lock()
		stopping, ended := s.stopping, s.ended
		s.mu.Unlock()
		if stopping || ended {
			return
		}
		code, exited := s.child.WaitExit(time.Second)
		if !exited {
			s.samplePeak()
			l.sampleP3(s)                                                  // GL-05: P3, one request and 60 s idle
			l.autosaveTick(context.Background(), s.pre.Warm, s.pre.Config) // GL-15: Invoke-AutosaveTick (PS:14536-14562)
			continue
		}
		s.mu.Lock()
		if s.stopping || s.ended {
			s.mu.Unlock()
			return
		}
		s.stopping = true
		s.mu.Unlock()
		reason := "server exited unexpectedly (code=" + FormatExitCode(&code) + ")"
		l.finishRun(s)
		td := stopOwnedChildGraceful(s.child, s.pre.Config.Port, l.stopEnv(nil))
		st, _ := teardownStatus(StatusFailRuntimeExit, td, true)
		s.child.Close()
		l.warmTeardownCleanup(s.pre.Warm)
		s.locks.release()
		end := RunEnd{Status: st, ExitCode: ExitCode(st), Reason: reason, Teardown: &td, At: l.os.now()}
		e := l.recordStop(s, end)
		end.Event = &e
		s.mu.Lock()
		s.ended = true
		s.end = end
		s.mu.Unlock()
		close(s.done)
		l.diag.write("RUNTIME_EXIT", map[string]any{"code": code, "status": st})
		return
	}
}

// StopEvent is one report of a stop; the last has Final set and carries the result.
type StopEvent struct {
	Step     string
	Final    bool
	Status   Status
	ExitCode int
	Reason   string
	Teardown *TeardownResult
	Event    *ServerStopEvent // conflict 6's launcher event, on the final one
	// Save is the cache-save bar while the quit's cache save runs (GL-07, decision ③); nil when
	// the stop runs no save. GL-15: the warm-start save (Invoke-WarmstartSave) runs under
	// TimedCacheSave with this save's tokens; every track arrives here.
	Save *CacheSaveTrack
	// GL-15: SaveErr is a failed save's reason (degraded: the stop goes on, the next start is
	// cold), with its last track in Save; NoSave says why this stop runs no save (sent once,
	// before the stop steps).
	SaveErr string
	NoSave  string
}

// ErrNoServer is returned by Stop and Status when this Launcher has started nothing.
var ErrNoServer = errors.New("no server started by this launcher")

// Stop stops the running server by the PowerShell procedure (CTRL_BREAK, 60 s grace, the
// forced kill, the listener check) and gives the status and exit code of Complete-Teardown
// with the pending status ok. It returns at once; the steps arrive on the channel.
func (l *Launcher) Stop() (<-chan StopEvent, error) { return l.stop(true) }

// StopNoSave is Stop without the warm-start save (GL-15): the close-now path (ctrl+c twice),
// which stores nothing.
func (l *Launcher) StopNoSave() (<-chan StopEvent, error) { return l.stop(false) }

func (l *Launcher) stop(save bool) (<-chan StopEvent, error) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return nil, ErrNoServer
	}
	s.mu.Lock()
	if s.ended || !s.ready {
		end := s.end
		ended := s.ended
		s.mu.Unlock()
		if ended {
			ch := make(chan StopEvent, 1)
			ch <- StopEvent{Final: true, Status: end.Status, ExitCode: end.ExitCode, Reason: end.Reason, Teardown: end.Teardown, Event: end.Event}
			close(ch)
			return ch, nil
		}
		return nil, errors.New("the server is still starting; cancel the start's context instead")
	}
	if s.stopping {
		s.mu.Unlock()
		return nil, errors.New("a stop is already running")
	}
	s.stopping = true
	s.mu.Unlock()
	ch := make(chan StopEvent, 16)
	go func() {
		defer close(ch)
		s.samplePeak()
		l.finishRun(s)
		if save { // GL-15: the save before the stop signal (A-2)
			l.warmStopSave(s, func(e StopEvent) { ch <- e })
		} else {
			l.diag.write("WARMSTART_SAVE_SKIP", map[string]any{"reason": noSaveClose})
			ch <- StopEvent{NoSave: noSaveClose}
		}
		td := stopOwnedChildGraceful(s.child, s.pre.Config.Port, l.stopEnv(func(st string) { ch <- StopEvent{Step: st} }))
		st, why := teardownStatus(StatusOK, td, true)
		s.child.Close()
		l.warmTeardownCleanup(s.pre.Warm)
		s.locks.release()
		reason := "stopped"
		if why != "" {
			reason = why
			l.diag.write("TEARDOWN_PROMOTED", map[string]any{"from": StatusOK, "reason": why})
		}
		l.diag.write("TEARDOWN", td)
		end := RunEnd{Status: st, ExitCode: ExitCode(st), Reason: reason, Teardown: &td, At: l.os.now()}
		e := l.recordStop(s, end)
		end.Event = &e
		s.mu.Lock()
		s.ended = true
		s.end = end
		s.mu.Unlock()
		close(s.done)
		ch <- StopEvent{Final: true, Status: st, ExitCode: ExitCode(st), Reason: reason, Teardown: &td, Event: &e}
	}()
	return ch, nil
}
