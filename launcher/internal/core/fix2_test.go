package core

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// FIX2-CORE rows 2, 3, 6, 7, 8, 10, 15, 18-1, 18-2 (rows 9 and 16: repack_test.go, gates_test.go).

// Row 7: the largest budget in G at which the start would pass.
func TestBudgetCapRow7(t *testing.T) {
	c := EvaluateStart(6000, 0, NeedNotMeasured, 8192)
	if c.Kind != CheckBlocked || c.MaxBudgetGiB != 5 {
		t.Errorf("blocked %+v", c)
	}
	if c := EvaluateStart(60000, 0, NeedNotMeasured, 8192); c.MaxBudgetGiB != 0 {
		t.Errorf("not blocked carries no cap: %+v", c)
	}
	if c := BudgetCap(6000, Bound{Min: 4096, Max: 65536}, true); !c.OK || c.GiB != 5 {
		t.Errorf("cap %+v", c)
	}
	if c := BudgetCap(200000, Bound{Min: 4096, Max: 16384}, true); !c.OK || c.GiB != 16 {
		t.Errorf("clamped to the bound's max %+v", c)
	}
	if c := BudgetCap(3000, Bound{Min: 4096, Max: 65536}, true); c.OK {
		t.Errorf("below the bound's min must not pass %+v", c)
	}
	f := newFixture(t, false)
	ctx := context.Background()
	f.mem.AvailPhysMiB = 6000 // the auto budget is 8192
	res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if !strings.Contains(fmt.Sprint(err), "RAM preflight hard stop: selected budget alone exceeds available RAM") ||
		!strings.Contains(fmt.Sprint(err), "a budget of at most 5 GB would pass") || !res.BudgetCap.OK || res.BudgetCap.GiB != 5 || ExitCodeOf(err) != 3 {
		t.Fatalf("refusal %v %+v", err, res.BudgetCap)
	}
	// The cap really passes: the same start with a 5 G budget is not refused.
	if _, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"budget_mb": "5120"}}); err != nil {
		t.Errorf("start at the cap: %v", err)
	}
	f.mem.AvailPhysMiB = 3000
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if !strings.Contains(fmt.Sprint(err), "even the smallest budget this model allows does not fit the free RAM") {
		t.Errorf("no budget fits: %v", err)
	}
}

// Row 6: the stop event's class, code, meaning and child exit code; the PS exit table.
func TestStopEventClassesRow6(t *testing.T) {
	at := time.Date(2026, 10, 10, 5, 7, 0, 0, time.Local)
	c0, c1 := uint32(0), uint32(1)
	cases := []struct {
		end   RunEnd
		class StopClass
		code  int
		child *uint32
	}{
		{RunEnd{Status: StatusOK, Teardown: &TeardownResult{Graceful: true, ExitCode: &c0}, At: at}, StopClean, 0, &c0},
		{RunEnd{Status: StatusFailTeardown, Teardown: &TeardownResult{TaskkillUsed: true, ExitCode: &c1}, At: at}, StopForced, 5, &c1},
		{RunEnd{Status: StatusCancelledUser, Teardown: &TeardownResult{ExitCode: &c1}, At: at}, StopBeforeReady, 2, &c1},
		{RunEnd{Status: StatusFailTeardown, Teardown: &TeardownResult{Graceful: true, ExitCode: &c0, ListenerGone: false}, At: at}, StopFailed, 5, &c0},
		{RunEnd{Status: StatusFailRuntimeExit, Teardown: &TeardownResult{ExitCode: &c1}, At: at}, StopFailed, 5, &c1},
		{RunEnd{Status: StatusFailSmoke, Teardown: &TeardownResult{ExitCode: &c0}, At: at}, StopFailed, 6, &c0},
		{RunEnd{Status: StatusFailServerStart, At: at}, StopFailed, 5, nil},
	}
	for _, c := range cases {
		e := ClassifyStop(c.end, true, 42)
		if e.Class != c.class || e.Code != c.code || e.ChildExit != c.child || e.Meaning == "" || e.PID != 42 {
			t.Errorf("%s: %+v", c.end.Status, e)
		}
		if !strings.HasPrefix(e.Line(), "05:07:00  server stop · "+c.class.String()) {
			t.Errorf("line %q", e.Line())
		}
	}
	// The process exit codes stay the PowerShell table.
	want := map[Status]int{StatusOK: 0, StatusOKSmoke: 0, StatusCancelledUser: 2, StatusFailCustomArgs: 3, StatusFailResource: 3,
		StatusFailRepack: 3, StatusFailGateBundle: 4, StatusFailGateVerify: 4, StatusFailServerStart: 5, StatusFailTeardown: 5,
		StatusFailRuntimeExit: 5, StatusFailSmoke: 6}
	for s, c := range want {
		if ExitCode(s) != c {
			t.Errorf("%s = %d", s, ExitCode(s))
		}
	}
	// A start that never spawned records no stop event; the launcher's list stays empty.
	f := newFixture(t, false)
	ch, _ := f.l.Start(context.Background(), Request{ModelPath: filepath.Join(f.dir, "none.gguf")})
	evs := drain(t, ch)
	if evs[len(evs)-1].StopInfo != nil || len(f.l.StopEvents()) != 0 {
		t.Errorf("no server, no stop event")
	}
}

// Row 15: the detection and the argument-path codes.
func TestTerminalAndArgCodesRow15(t *testing.T) {
	ok := func() error { return nil }
	bad := func() error { return errors.New("The handle is invalid.") }
	if r := menuModeAvailable(false, consoleProbe{stdinConsole: ok, stdoutSBufInfo: ok}); !r.Interactive {
		t.Errorf("console: %+v", r)
	}
	for _, c := range []struct {
		ni   bool
		p    consoleProbe
		want string
	}{
		{true, consoleProbe{stdinConsole: ok, stdoutSBufInfo: ok}, "-NonInteractive"},
		{false, consoleProbe{haveConsole: func() bool { return false }, stdinConsole: ok, stdoutSBufInfo: ok}, "no console"},
		{false, consoleProbe{stdinConsole: bad, stdoutSBufInfo: ok}, "input redirected"},
		{false, consoleProbe{stdinConsole: ok, stdoutSBufInfo: bad}, "output redirected"},
	} {
		if r := menuModeAvailable(c.ni, c.p); r.Interactive || !strings.HasPrefix(r.Reason, c.want) {
			t.Errorf("%s: %+v", c.want, r)
		}
	}
	// This test binary runs with redirected output: never interactive.
	if r := DetectTerminal(false); r.Interactive {
		t.Errorf("go test is not an interactive console: %+v", r)
	}
	f := newFixture(t, false)
	ctx := context.Background()
	_, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"ctx": "100"}})
	if ExitCodeOf(err) != 3 || StatusOf(err) != StatusFailCustomArgs {
		t.Errorf("range violation: %v -> %d", err, ExitCodeOf(err))
	}
	for _, c := range []struct {
		err  error
		code int
	}{
		{nil, 0},
		{stop(StatusCancelledUser, "x"), 2},
		{&RepackNeededError{Status: StatusFailRepack}, 3},
		{notPorted(FamilyPresets, "x"), 3}, // GL-18: warm start is ported (GL-15); a family still not ported
		{&GateError{LaunchError: stop(StatusFailGateVerify, "x")}, 4},
		{BundleResult{State: BundleMismatch, Mismatches: 2}.Err(), 4},
		{stop(StatusFailSmoke, "x"), 6},
		{fmt.Errorf("wrapped: %w", stop(StatusFailResource, "x")), 3},
		{errors.New("anything else"), 5},
	} {
		if got := ExitCodeOf(c.err); got != c.code {
			t.Errorf("%v -> %d, want %d", c.err, got, c.code)
		}
	}
}

// Row 18-2: the bundle manifest check with a good, a bad and a missing manifest.
func TestBundleCheckRow18(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) string {
		p := filepath.Join(root, rel)
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, []byte(body), 0o644)
		return sha256Hex([]byte(body))
	}
	// missing manifest: a development build, skipped and said so
	r := CheckBundle(context.Background(), root)
	if r.State != BundleDev || r.Err() != nil || !strings.Contains(r.Reason, "development build") {
		t.Errorf("dev %+v", r)
	}
	a := write("llama-server.exe", "server")
	b := write(`expects\x.expect.json`, "expect")
	manifest := func(files []any) {
		writeJSON(t, filepath.Join(root, BundleManifestName), map[string]any{"bundle_manifest_version": 1, "files": files})
	}
	manifest([]any{map[string]any{"path": "llama-server.exe", "sha256": a}, map[string]any{"path": "expects/x.expect.json", "sha256": strings.ToUpper(b)}})
	r = CheckBundle(context.Background(), root)
	if r.State != BundleOK || r.Files != 2 || r.Mismatches != 0 || r.Err() != nil {
		t.Errorf("good %+v", r)
	}
	// bad: one changed, one missing, one unlisted -> 3 (PowerShell stops at the first)
	write("llama-server.exe", "changed")
	os.Remove(filepath.Join(root, "expects", "x.expect.json"))
	write("extra.dll", "x")
	r = CheckBundle(context.Background(), root)
	if r.State != BundleMismatch || r.Mismatches != 3 || ExitCodeOf(r.Err()) != 4 || !strings.Contains(r.Err().Error(), "3 file(s) differ") {
		t.Errorf("bad %+v %v", r, r.Err())
	}
	kinds := map[string]string{}
	for _, d := range r.Diffs {
		kinds[d.Path] = d.Kind
	}
	if kinds["llama-server.exe"] != "hash" || kinds["expects/x.expect.json"] != "missing" || kinds["extra.dll"] != "unlisted" {
		t.Errorf("diffs %v", r.Diffs)
	}
	// a broken manifest: refused with its reason, nothing counted
	writeJSON(t, filepath.Join(root, BundleManifestName), map[string]any{"bundle_manifest_version": 2, "files": []any{}})
	if r := CheckBundle(context.Background(), root); r.State != BundleMismatch || r.Mismatches != 0 || !strings.Contains(r.Reason, "bundle_manifest_version") {
		t.Errorf("broken %+v", r)
	}
	manifest([]any{map[string]any{"path": "../x", "sha256": a}})
	if r := CheckBundle(context.Background(), root); !strings.Contains(r.Reason, "escapes root") {
		t.Errorf("escape %+v", r)
	}

	// In the background from launch; a start waits for it and refuses a mismatch with code 4.
	f := newFixture(t, false)
	if r := f.l.BundleCheckResult(); r.State != BundlePending {
		t.Errorf("before the start: %+v", r)
	}
	writeJSON(t, filepath.Join(f.root, BundleManifestName), map[string]any{"bundle_manifest_version": 1,
		"files": []any{map[string]any{"path": "llama-server.exe", "sha256": a}}})
	f.l.StartBundleCheck()
	r, err := f.l.WaitBundleCheck(context.Background())
	if err != nil || r.State != BundleMismatch || r.Mismatches != 2 { // the listed server is missing, the expect file is unlisted (the catalog lies outside the root)
		t.Errorf("fixture bundle %+v %v", r, err)
	}
	if got := f.l.BundleCheckResult(); got.State != BundleMismatch {
		t.Errorf("result after the wait: %+v", got)
	}
	ch, _ := f.l.Start(context.Background(), Request{ModelPath: f.modelPath})
	evs := drain(t, ch)
	last := evs[len(evs)-1]
	if last.Status != StatusFailGateBundle || last.ExitCode != 4 || evs[0].Step != StepBundle {
		t.Errorf("start after a bad bundle: %+v", last)
	}
}

// Row 18-1: the setting and first-once counting.
func TestSmokeSettingFirstOnceRow18(t *testing.T) {
	for in, want := range map[string]SmokeMode{"": SmokeOff, "off": SmokeOff, "ON": SmokeOn, "first-once": SmokeFirstOnce, "first_once": SmokeFirstOnce} {
		if m, err := ParseSmokeMode(in); err != nil || m != want {
			t.Errorf("%q -> %v %v", in, m, err)
		}
	}
	if _, err := ParseSmokeMode("sometimes"); err == nil {
		t.Error("unknown value accepted")
	}
	if SmokeOff.String() != "off" || SmokeFirstOnce.String() != "first-once" {
		t.Error("names")
	}
	f := newFixture(t, false)
	l, m := f.l, f.modelPath
	if l.SmokeDue(SmokeOff, m, RepackPacked) || !l.SmokeDue(SmokeOn, m, RepackPacked) || !l.SmokeDue(SmokeFirstOnce, m, RepackPacked) {
		t.Fatal("initial")
	}
	l.markSmoked(m, "") // "" is packed
	if l.SmokeDue(SmokeFirstOnce, m, RepackPacked) || !l.SmokeDue(SmokeOn, m, RepackPacked) {
		t.Error("first-once must not run again for the same model and mode")
	}
	if !l.SmokeDue(SmokeFirstOnce, m, RepackVirtual) || !l.SmokeDue(SmokeFirstOnce, filepath.Join(f.dir, "other.gguf"), RepackPacked) {
		t.Error("another mode or model is due")
	}
	// ctx and budget are not part of the count: the key is model + mode only.
	if smokeKey(m, RepackPacked) != smokeKey(strings.ToUpper(m), "packed") {
		t.Error("key")
	}
	// A new launcher session counts from zero.
	l2, _ := New(Options{CatalogPath: f.catalogPath, BundleRoot: f.root, StateDir: f.stateDir})
	if !l2.SmokeDue(SmokeFirstOnce, m, RepackPacked) {
		t.Error("first-once counts per launcher session")
	}
}

// fakeLlama is a minimal server for the checklist and the warmup: /health, /v1/models, the chat
// endpoint (non-stream and an 80-chunk stream), /completion; on a dropped stream it logs the
// engine's cancel and release lines when release is set.
type fakeLlama struct {
	mu       sync.Mutex
	log      []string
	release  bool
	task     int
	bodies   []string
	chatCode int
}

func (s *fakeLlama) lines() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.log...)
}

func (s *fakeLlama) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	s.bodies = append(s.bodies, r.URL.Path+" "+string(b))
	s.task++
	task := s.task
	s.mu.Unlock()
	switch r.URL.Path {
	case "/health":
		w.Write([]byte(`{"status":"ok"}`))
	case "/v1/models":
		w.Write([]byte(`{"object":"list","data":[{"id":"fixture"}]}`))
	case "/completion":
		w.Write([]byte(`{"content":"x","tokens_evaluated":42}`))
	case "/v1/chat/completions":
		if s.chatCode != 0 {
			w.WriteHeader(s.chatCode)
			return
		}
		if !strings.Contains(string(b), `"stream":true`) {
			w.Write([]byte(`{"choices":[{"message":{"content":"pong"},"finish_reason":"stop"}]}`))
			return
		}
		fl := w.(http.Flusher)
		for i := 1; i <= 80; i++ {
			select {
			case <-r.Context().Done():
				s.mu.Lock()
				if s.release {
					s.log = append(s.log, fmt.Sprintf("srv          stop: cancel task, id_task = %d", task),
						fmt.Sprintf("slot      release: id  0 | task %d | stop processing: n_tokens = 3, truncated = 0", task))
				}
				s.mu.Unlock()
				return
			default:
			}
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"%d \"}}]}\n\n", i)
			fl.Flush()
			time.Sleep(4 * time.Millisecond)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
	}
}

var testSmokeTimings = smokeTimings{HTTP: 10 * time.Second, Stream: 10 * time.Second, Poll: 5 * time.Millisecond, WarnDiag: 300 * time.Millisecond}

func TestSmokeChecklistRow18(t *testing.T) {
	fl := &fakeLlama{release: true}
	srv := httptest.NewServer(fl)
	defer srv.Close()
	r := runSmokeChecklist(context.Background(), srv.URL, fl.lines, testSmokeTimings)
	if !r.OK || len(r.Items) != 4 || len(r.Failures) != 0 || !strings.HasPrefix(r.Line(), "smoke PASS (4 items") {
		t.Fatalf("pass: %+v", r)
	}
	if !strings.Contains(r.Items[3].Detail, "bound release for task") {
		t.Errorf("item 4 %+v", r.Items[3])
	}
	// The bodies are PowerShell's.
	joined := strings.Join(fl.bodies, "\n")
	if !strings.Contains(joined, smokeBodyNonStream) || !strings.Contains(joined, smokeBodyStream) {
		t.Errorf("bodies %v", fl.bodies)
	}
	// A server that never releases the cancelled slot fails item 4 only.
	fl2 := &fakeLlama{}
	srv2 := httptest.NewServer(fl2)
	defer srv2.Close()
	r = runSmokeChecklist(context.Background(), srv2.URL, fl2.lines, testSmokeTimings)
	if r.OK || len(r.Failures) != 1 || r.Failures[0] != "4:cancel" || !strings.Contains(r.Items[3].Detail, `no "cancel task, id_task" warning`) {
		t.Errorf("no release: %+v", r)
	}
	// Nothing listening: items 1-4 all fail.
	r = runSmokeChecklist(context.Background(), "http://127.0.0.1:1", func() []string { return nil }, testSmokeTimings)
	if r.OK || len(r.Failures) != 4 {
		t.Errorf("down: %+v", r.Failures)
	}
	// The release finder binds the release to the cancelled task.
	b := findBoundCancelRelease([]string{"x", "cancel task, id_task = 5", "slot release: id 0 | task 4 | stop processing: a",
		"slot release: id 0 | task 5 | stop processing: b"}, 0)
	if !b.found || b.taskID != "5" || b.releaseIndex != 3 {
		t.Errorf("finder %+v", b)
	}
	if len(SmokeNotPorted) != 1 { // GL-11 ported items 5-7; item 4's producer proofs remain
		t.Error("not-ported list")
	}
}

// Row 2: warmup on / off / file / forced off / failure (degraded).
func TestWarmupRow2(t *testing.T) {
	fl := &fakeLlama{}
	srv := httptest.NewServer(fl)
	defer srv.Close()
	ctx := context.Background()
	w := runWarmup(ctx, srv.URL, "on", "", "")
	if !w.OK || w.Skipped || w.Line() != "[warmup] launcher warmup request completed." || !strings.Contains(fl.bodies[0], warmupBody) {
		t.Errorf("on: %+v %v", w, fl.bodies)
	}
	if w := runWarmup(ctx, srv.URL, "off", "", ""); !w.Skipped || w.SkipReason != WarmupSkipUserOff {
		t.Errorf("off %+v", w)
	}
	if w := runWarmup(ctx, srv.URL, "on", warmupForcedReason, ""); !w.Skipped || w.SkipReason != WarmupSkipForced {
		t.Errorf("forced %+v", w)
	}
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "w.txt"), append([]byte{0xEF, 0xBB, 0xBF}, []byte("h\u00e9\"\\ \U0001F600")...), 0o644)
	fl.bodies = nil
	w = runWarmup(ctx, srv.URL, "file:w.txt", "", dir)
	if !w.OK || w.Tokens != 42 || w.Mode != "file" || !strings.Contains(fl.bodies[0], `/completion {"prompt":"h\u00e9\"\\ \ud83d\ude00","cache_prompt":true,"n_predict":1,"stream":false}`) {
		t.Errorf("file: %+v %v", w, fl.bodies)
	}
	if w := runWarmup(ctx, srv.URL, "file:"+filepath.Join(dir, "none.txt"), "", dir); w.OK || !w.Degraded || !strings.Contains(w.Reason, "could not be read") {
		t.Errorf("missing file %+v", w)
	}
	fl.chatCode = 500
	if w := runWarmup(ctx, srv.URL, "on", "", ""); w.OK || !w.Degraded || !strings.HasPrefix(w.Line(), "[warmup] WARNING: warmup request failed") {
		t.Errorf("failure %+v", w)
	}
	if v := warmupValue(map[string]string{}); v != "on" {
		t.Errorf("default %q", v)
	}
}

// Row 2: the open set, the locked rows, the range refusals.
func TestOpenSettingsRow2(t *testing.T) {
	f := newFixture(t, false)
	rows, err := f.l.Settings("fixture-profile")
	if err != nil {
		t.Fatal(err)
	}
	byKey := map[string]SettingRow{}
	for _, r := range rows {
		byKey[r.Key] = r
	}
	for _, k := range []string{"port", "ctx", "threads", "warmup", "budget_mb", "qd", "prefetch", "repack_mode"} {
		if !byKey[k].Editable || byKey[k].Locked != "" {
			t.Errorf("%s must be open: %+v", k, byKey[k])
		}
	}
	if byKey["warmup"].Default != "on" {
		t.Errorf("warmup default %+v", byKey["warmup"])
	}
	// GL-16: warm start and autosave are 5 Settings switches (OQ-6 (a)), no model rows. GL-19:
	// K and N are open value rows (F-PF §2.2): K 1..min(16, n_expert), N from 1 (its on range).
	for _, k := range []string{"warmstart", "autosave"} {
		if _, ok := byKey[k]; ok {
			t.Errorf("%s is a model row: %+v", k, byKey[k])
		}
	}
	for _, k := range []string{"prefetch_k", "prefetch_n"} {
		if !byKey[k].Editable || byKey[k].Locked != "" || byKey[k].Default != "auto" || byKey[k].Min != 1 {
			t.Errorf("%s must be an open value row: %+v", k, byKey[k])
		}
	}
	if !byKey["prefetch_k"].Bounded || byKey["prefetch_n"].Bounded || !byKey["prefetch_n"].OnRange {
		t.Errorf("K bounded, N an on range: %+v %+v", byKey["prefetch_k"], byKey["prefetch_n"])
	}
	locked, err := f.l.LockedSettings("fixture-profile")
	if err != nil || len(locked) != 4 || locked[0].Key != "kv_type" || locked[0].Default != "f16" || locked[2].Default != "1" ||
		locked[3].Default != "127.0.0.1" || locked[1].Editable {
		t.Errorf("locked %+v %v", locked, err)
	}
	b := map[string]Bound{"ctx": {2048, 131072}}
	if _, _, err := resolveOverrides(map[string]string{"warmup": "maybe"}, b, false, PathArguments); ExitCodeOf(err) != 3 || !strings.Contains(err.Error(), "-warmup = maybe, allowed on, off or file:<path>") {
		t.Errorf("warmup range: %v", err)
	}
	if ov, _, err := resolveOverrides(map[string]string{"warmup": "file:C:\\w.txt"}, b, false, PathArguments); err != nil || ov["warmup"] != "file:C:\\w.txt" {
		t.Errorf("warmup file: %v %v", ov, err)
	}
	if _, _, err := resolveOverrides(map[string]string{"ctx": "999999"}, b, false, PathArguments); ExitCodeOf(err) != 3 || !strings.Contains(err.Error(), "-ctx = 999999, allowed 2048..131072") {
		t.Errorf("ctx range: %v", err)
	}
}

// Rows 3 and 10: provenance, custom, and the per-settings speed fact.
func TestProvenanceAndCustomRows3And10(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	l, m := f.l, f.modelPath
	pv, err := l.SettingsProvenance(ctx, m, "fixture-profile", map[string]string{"ctx": "4096", "budget_mb": "auto", "warmup": "on"})
	if err != nil || pv.State != ProvDefault || pv.Custom || len(pv.Changed) != 0 {
		t.Fatalf("defaults: %+v %v", pv, err)
	}
	pv, _ = l.SettingsProvenance(ctx, m, "fixture-profile", map[string]string{"ctx": "8192"})
	if pv.State != ProvDefaultUnsaved || !pv.Custom || strings.Join(pv.Changed, ",") != "ctx" {
		t.Errorf("changed, no preset: %+v", pv)
	}
	if _, err := l.SavePreset(ctx, m, "fixture-profile", map[string]string{"ctx": "8192"}); err != nil {
		t.Fatal(err)
	}
	pv, _ = l.SettingsProvenance(ctx, m, "fixture-profile", map[string]string{"ctx": "8192"})
	if pv.State != ProvPreset || !pv.Custom || len(pv.Changed) != 0 || pv.Preset == nil {
		t.Errorf("the preset, which differs from the defaults (custom): %+v", pv)
	}
	pv, _ = l.SettingsProvenance(ctx, m, "fixture-profile", map[string]string{"ctx": "8192", "threads": "4"})
	if pv.State != ProvPresetUnsaved || strings.Join(pv.Changed, ",") != "threads" || strings.Join(pv.CustomKeys, ",") != "ctx,threads" {
		t.Errorf("preset changed: %+v", pv)
	}
	pv, _ = l.SettingsProvenance(ctx, m, "fixture-profile", map[string]string{})
	if pv.State != ProvPresetUnsaved || strings.Join(pv.Changed, ",") != "ctx" || pv.Custom {
		t.Errorf("back to the defaults while a preset exists: %+v", pv)
	}
	l.DeletePreset("fixture-profile", m)
	pv, _ = l.SettingsProvenance(ctx, m, "fixture-profile", map[string]string{"repack_mode": "virtual"})
	if !pv.Custom || pv.CustomKeys[0] != "repack_mode" {
		t.Errorf("virtual counts as custom: %+v", pv)
	}
	// Row 10: «측정 전» per settings key until a run of that key answered a request.
	res, err := l.Preflight(ctx, Request{ModelPath: m})
	if err != nil || res.SpeedMeasured {
		t.Fatalf("first: %v %v", err, res.SpeedMeasured)
	}
	if err := l.recordRun(RunRecord{Key: res.SettingsKey, ProfileID: "fixture-profile", ModelPath: m, ReadyMillis: 1000, At: "2026-10-10T00:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	if err := l.recordRunEnd(res.SettingsKey, m, 0, TimingSummary{AnswerTokens: 62, AnswerTime: 14727 * time.Millisecond}, time.Now()); err != nil {
		t.Fatal(err)
	}
	res, _ = l.Preflight(ctx, Request{ModelPath: m})
	if !res.SpeedMeasured || res.SpeedTokPerSec < 4.2 || res.SpeedTokPerSec > 4.22 {
		t.Errorf("measured: %+v", res.SpeedTokPerSec)
	}
	res2, _ := l.Preflight(ctx, Request{ModelPath: m, Settings: map[string]string{"ctx": "8192"}})
	if res2.SpeedMeasured {
		t.Error("another settings key is not measured yet")
	}
}

// Row 8: an out-of-memory load is its own failure reason; a normal log is not one.
func TestOOMClassifyRow8(t *testing.T) {
	for _, c := range []struct {
		line, dev string
	}{
		{"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 20480.00 MiB on device 0: cudaMalloc failed: out of memory", "vram"},
		{"llama_model_load: error loading model: unable to allocate CUDA0 buffer", "vram"},
		{"CUDA error: out of memory", "vram"},
		{"llama_model_load: error loading model: unable to allocate CPU buffer", "ram"},
		{"terminate called after throwing an instance of 'std::bad_alloc'", "ram"},
		{"llama_init_from_model: failed to allocate compute buffers", ""},
	} {
		f, ok := ClassifyLoadFailure([]string{"llama_model_loader: loaded meta data", c.line})
		if !ok || f.Cause != CauseOutOfMemory || f.Device != c.dev || f.Line != c.line {
			t.Errorf("%q: %+v %v", c.line, f, ok)
		}
	}
	// a device line wins over an earlier generic one
	if f, _ := ClassifyLoadFailure([]string{"failed to allocate compute buffers", "cudaMalloc failed: out of memory"}); f.Device != "vram" {
		t.Errorf("device first: %+v", f)
	}
	for _, p := range []string{"testdata/server_20261009T171115Z_err.log", "testdata/server_20261009T175949Z_err.log"} {
		if f, ok := ClassifyLoadFailure(readChildStderrComplete(p)); ok {
			t.Errorf("%s classified as OOM: %+v", p, f)
		}
	}
	// waitForServerReady gives the sub-reason with code 5.
	dir := t.TempDir()
	errLog := filepath.Join(dir, "err.log")
	os.WriteFile(errLog, []byte("load...\ncudaMalloc failed: out of memory\n"), 0o644)
	c := &fakeChild{pid: 9, exited: true, code: 1}
	err := waitForServerReady(context.Background(), c, "127.0.0.1", 1, errLog, func(int) ([]int, error) { return nil, nil },
		psReadyTimings, time.Now, func(time.Duration) {}, nil)
	le, ok := AsLaunchError(err)
	if !ok || le.Status != StatusFailServerStart || le.Load == nil || le.Load.Device != "vram" || ExitCodeOf(err) != 5 ||
		!strings.Contains(le.Reason, "ran out of memory while loading the model (GPU memory") {
		t.Errorf("ready wait: %v %+v", err, le)
	}
}
