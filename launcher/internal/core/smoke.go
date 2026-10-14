package core

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Conflict 18-1 of reviews/_gol_conflicts_8_user_20261010.md: the «작동 확인» check is the
// PowerShell smoke checklist (Invoke-SmokeChecklist PS:12665, run site PS:14521-14526): /health,
// /v1/models, a «ping» chat and a 1-to-80 stream, a cancel after one token with the slot
// reclaimed. The setting is off / on / first-once (default off); first-once counts per model
// and repack mode within one launcher session, so a change of ctx or budget alone does not run
// it again. The result goes to the log; a failure is the start's failure reason (fail_smoke,
// exit 6, which the argument path's --smoke keeps).
//
// Ported: items 1-4 with the PowerShell bodies, timeouts and verdict arithmetic (item 4's
// timing conjunction: aborted after >= 1 token, outstanding time > 0, the cancelled task's own
// release line seen inside half the outstanding time, then a follow-up request served).
// GL-11 (smoke_items.go) adds items 5-7 and one log line per item. Not ported, named in
// SmokeNotPorted: item 4's producer-capture proofs.

// SmokeMode is the «작동 확인» setting.
type SmokeMode uint8

const (
	SmokeOff       SmokeMode = iota // default
	SmokeOn                         // every start
	SmokeFirstOnce                  // the first start of each model and repack mode in this launcher session
)

// SmokeModes are the setting's values, in the order the setting offers them.
var SmokeModes = []string{"off", "on", "first-once"}

func (m SmokeMode) String() string { return SmokeModes[m] }

// ParseSmokeMode reads the setting's value.
func ParseSmokeMode(s string) (SmokeMode, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "off":
		return SmokeOff, nil
	case "on":
		return SmokeOn, nil
	case "first-once", "first_once", "firstonce":
		return SmokeFirstOnce, nil
	}
	return SmokeOff, fmt.Errorf("smoke must be one of %s", strings.Join(SmokeModes, "|"))
}

// SmokeNotPorted names the PowerShell smoke parts this draft does not run, with the reason.
var SmokeNotPorted = []string{
	"item 4 producer proofs (Read-SmokeCapture, Test-SmokeReferenceCapture, Test-SmokeCaptureAdvance, the direct-invocation binding): they verify PowerShell's own stderr capture of a directly invoked server; the draft reads the server's stderr file itself and keeps the timing conjunction",
}

// The smoke constants (PS:252-254, :1221, Get-SmokeRequestBody PS:12650-12656).
const (
	smokeBodyNonStream = `{"model":"local","messages":[{"role":"user","content":"ping"}],"max_tokens":64,"stream":false}`
	smokeBodyStream    = `{"model":"local","messages":[{"role":"user","content":"Write the integers 1 through 80 in order, separated by single spaces. Output only the integers."}],"max_tokens":256,"stream":true}`
)

var (
	cancelTaskRe  = regexp.MustCompile(`cancel task, id_task = (\d+)`)              // CANCEL_TASK_REGEX
	taskReleaseRe = regexp.MustCompile(`\|\s*task\s+(\d+)\s*\|\s*stop processing:`) // TASK_RELEASE_REGEX
)

type smokeTimings struct {
	HTTP       time.Duration // Invoke-HttpJson -TimeoutSec 60
	Stream     time.Duration // Invoke-SmokeStream 60000 ms
	Poll       time.Duration // 50 ms
	WarnDiag   time.Duration // CANCEL_WARN_DIAG_MS
	ReferenceW time.Duration // the 60000 ms wait for the reference stream's records (diagnostic here)
}

var psSmokeTimings = smokeTimings{HTTP: 60 * time.Second, Stream: 60 * time.Second, Poll: 50 * time.Millisecond,
	WarnDiag: 3000 * time.Millisecond, ReferenceW: 60 * time.Second}

// SmokeItem is one checklist item.
type SmokeItem struct {
	N       int
	Name    string
	OK      bool
	Skipped bool // item 7 on an API-only bundle (PowerShell's SKIP)
	Detail  string
}

// SmokeResult is one run of the checklist.
type SmokeResult struct {
	OK       bool
	Items    []SmokeItem
	Failures []string // "1:/health", "2:/v1/models", "3:chat", "4:cancel" (PowerShell's labels)
	Elapsed  time.Duration
}

// Line is the result as one log line.
func (r SmokeResult) Line() string {
	if r.OK {
		return fmt.Sprintf("smoke PASS (%d items, %.1f s)", len(r.Items), r.Elapsed.Seconds())
	}
	return fmt.Sprintf("smoke FAIL: %s (%.1f s)", strings.Join(r.Failures, ", "), r.Elapsed.Seconds())
}

type httpResult struct {
	ok     bool // a response with a 2xx code (Invoke-WebRequest throws on the others)
	status int
	body   []byte
	reason string
}

func httpJSON(ctx context.Context, method, url, body string, timeout time.Duration) httpResult {
	c, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req, err := http.NewRequestWithContext(c, method, url, rd)
	if err != nil {
		return httpResult{reason: err.Error()}
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return httpResult{reason: err.Error()}
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return httpResult{reason: err.Error(), status: resp.StatusCode}
	}
	r := httpResult{status: resp.StatusCode, body: b, ok: resp.StatusCode >= 200 && resp.StatusCode < 300}
	if !r.ok {
		r.reason = fmt.Sprintf("status %d", resp.StatusCode)
	}
	return r
}

type streamResult struct {
	ok, aborted, done bool
	tokens            int
	elapsed           time.Duration
	reason            string
}

// smokeStream is Invoke-SmokeStream (PS:12955-13018): POST the stream body, count the SSE chunks
// that carry delta content, abort after abortAfter tokens (0 = read to the end).
func smokeStream(ctx context.Context, url string, abortAfter int, timeout time.Duration) streamResult {
	t0 := time.Now()
	c, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(c, http.MethodPost, url, strings.NewReader(smokeBodyStream))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return streamResult{reason: err.Error(), elapsed: time.Since(t0)}
	}
	defer resp.Body.Close()
	r := streamResult{ok: true}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[5:])
		if payload == "[DONE]" {
			r.done = true
			break
		}
		v, err := parseJSONStrict([]byte(payload))
		if err != nil {
			continue
		}
		o, _ := asObj(v)
		ch, ok := jArr(o["choices"])
		if !ok || len(ch) == 0 {
			continue
		}
		c0, _ := asObj(ch[0])
		d, _ := asObj(c0["delta"])
		content, _ := d["content"].(string)
		if content == "" {
			continue
		}
		r.tokens++
		if abortAfter > 0 && r.tokens >= abortAfter {
			r.aborted = true
			cancel() // HttpWebRequest.Abort: the connection is dropped
			r.elapsed = time.Since(t0)
			return r
		}
	}
	if err := sc.Err(); err != nil && !r.done {
		r.ok, r.reason = false, err.Error()
	}
	r.elapsed = time.Since(t0)
	return r
}

// boundCancelRelease is Find-BoundCancelRelease (PS:12017-12045): the first cancel warning
// after fromLine, then that same task's release line after it.
type boundRelease struct {
	found         bool
	taskID        string
	cancelIndex   int
	releaseIndex  int
	cancelTaskIDs []string
	reason        string
}

func findBoundCancelRelease(lines []string, fromLine int) boundRelease {
	cancelAt := map[string]int{}
	var order []string
	for i := fromLine; i < len(lines); i++ {
		if m := cancelTaskRe.FindStringSubmatch(lines[i]); m != nil {
			if _, seen := cancelAt[m[1]]; !seen {
				cancelAt[m[1]] = i
				order = append(order, m[1])
			}
		}
	}
	if len(order) == 0 {
		return boundRelease{reason: `no "cancel task, id_task" warning yet`}
	}
	for _, id := range order {
		for j := cancelAt[id] + 1; j < len(lines); j++ {
			if m := taskReleaseRe.FindStringSubmatch(lines[j]); m != nil && m[1] == id {
				return boundRelease{found: true, taskID: id, cancelIndex: cancelAt[id], releaseIndex: j, cancelTaskIDs: order}
			}
		}
	}
	return boundRelease{cancelTaskIDs: order,
		reason: "cancel warning seen for task(s) " + strings.Join(order, ",") + " but no matching release after it"}
}

// runSmokeChecklist runs items 1-4 against base (http://host:port); readLog returns the server's
// complete stderr lines so far (Read-ChildStderrComplete).
func runSmokeChecklist(ctx context.Context, base string, readLog func() []string, t smokeTimings) SmokeResult {
	t0 := time.Now()
	res := SmokeResult{}
	add := func(n int, name string, ok bool, detail string, label string) {
		res.Items = append(res.Items, SmokeItem{N: n, Name: name, OK: ok, Detail: detail})
		if !ok {
			res.Failures = append(res.Failures, label)
		}
	}
	// (1) /health 200
	r := httpJSON(ctx, http.MethodGet, base+"/health", "", t.HTTP)
	add(1, "/health 200", r.ok && r.status == 200, fmt.Sprintf("status %d %s", r.status, r.reason), "1:/health")

	// (2) /v1/models: 200 with a data array
	r = httpJSON(ctx, http.MethodGet, base+"/v1/models", "", t.HTTP)
	ok2 := false
	if r.ok && r.status == 200 {
		if v, err := parseJSONStrict(r.body); err == nil {
			o, _ := asObj(v)
			_, ok2 = jArr(o["data"])
		}
	}
	add(2, "/v1/models", ok2, fmt.Sprintf("status %d %s", r.status, r.reason), "2:/v1/models")

	// (3) the non-stream «ping» and the natural 1-to-80 stream
	chat := base + "/v1/chat/completions"
	r = httpJSON(ctx, http.MethodPost, chat, smokeBodyNonStream, t.HTTP)
	ok3a := false
	finish := ""
	if r.ok && r.status == 200 {
		if v, err := parseJSONStrict(r.body); err == nil {
			o, _ := asObj(v)
			if ch, ok := jArr(o["choices"]); ok && len(ch) > 0 {
				c0, _ := asObj(ch[0])
				msg, _ := asObj(c0["message"])
				content, _ := msg["content"].(string)
				finish, _ = c0["finish_reason"].(string)
				ok3a = content != "" && (finish == "stop" || finish == "length")
			}
		}
	}
	natural := smokeStream(ctx, chat, 0, t.Stream)
	ok3b := natural.ok && natural.tokens > 0 && natural.done
	add(3, "chat completion stream+non-stream", ok3a && ok3b,
		fmt.Sprintf("non-stream=%v finish_reason=%s stream_tokens=%d done=%v natural_ms=%d %s", ok3a, finish, natural.tokens, natural.done,
			natural.elapsed.Milliseconds(), natural.reason), "3:chat")

	// (4) cancel after one token, then the cancelled task's own release inside half the time
	// that was still outstanding, then a follow-up request served.
	lineBefore := len(readLog())
	cancelRun := smokeStream(ctx, chat, 1, t.Stream)
	cancelAt := time.Now()
	outstanding := natural.elapsed - cancelRun.elapsed
	if outstanding < 0 {
		outstanding = 0
	}
	naturalEnd := cancelAt.Add(outstanding)
	var bound boundRelease
	releaseSeen, warnSeen := false, false
	releaseAt := time.Duration(-1)
	for time.Now().Before(naturalEnd) && ctx.Err() == nil {
		bound = findBoundCancelRelease(readLog(), lineBefore)
		if len(bound.cancelTaskIDs) > 0 {
			warnSeen = true
		}
		if bound.found {
			releaseSeen, releaseAt = true, time.Since(cancelAt)
			break
		}
		time.Sleep(t.Poll)
	}
	if !warnSeen {
		deadline := time.Now().Add(t.WarnDiag)
		for time.Now().Before(deadline) && ctx.Err() == nil {
			p := findBoundCancelRelease(readLog(), lineBefore)
			if len(p.cancelTaskIDs) > 0 || p.found {
				warnSeen, bound = true, p
				if p.found {
					releaseSeen, releaseAt = true, time.Since(cancelAt)
				}
				break
			}
			time.Sleep(t.Poll)
		}
	}
	promptBudget := outstanding / 2
	promptly := releaseSeen && releaseAt >= 0 && releaseAt < promptBudget
	var why string
	switch {
	case promptly:
		why = fmt.Sprintf("bound release for task %s observed %d ms after the abort, inside the %d ms prompt budget", bound.taskID, releaseAt.Milliseconds(), promptBudget.Milliseconds())
	case releaseSeen:
		why = fmt.Sprintf("bound release for task %s arrived at %d ms, at or past the %d ms prompt budget", bound.taskID, releaseAt.Milliseconds(), promptBudget.Milliseconds())
	case warnSeen:
		why = "cancel warning seen for task(s) " + strings.Join(bound.cancelTaskIDs, ",") + " but no matching release after it"
	default:
		why = fmt.Sprintf(`no "cancel task, id_task" warning within %d ms of the abort`, t.WarnDiag.Milliseconds())
	}
	after := httpResult{}
	if promptly {
		after = httpJSON(ctx, http.MethodPost, chat, smokeBodyNonStream, t.HTTP)
	}
	ok4 := cancelRun.aborted && cancelRun.tokens >= 1 && outstanding > 0 && promptly && after.ok && after.status == 200
	add(4, "cancel + slot reclaim", ok4, fmt.Sprintf("aborted=%v tokens=%d outstanding_ms=%d next_ok=%v %s",
		cancelRun.aborted, cancelRun.tokens, outstanding.Milliseconds(), after.ok && after.status == 200, why), "4:cancel")

	res.OK = len(res.Failures) == 0
	res.Elapsed = time.Since(t0)
	return res
}

// smokeTracker is the first-once record of one launcher session.
type smokeTracker struct {
	mu   sync.Mutex
	done map[string]bool
}

func newSmokeTracker() *smokeTracker { return &smokeTracker{done: map[string]bool{}} }

func smokeKey(modelPath, repackMode string) string {
	if repackMode == "" {
		repackMode = RepackPacked
	}
	return strings.ToLower(representativePath(modelPath)) + "|" + strings.ToLower(repackMode)
}

// SmokeDue says whether a start of this model and repack mode runs the check under the
// setting: off never, on always, first-once when no check of this model and mode has passed
// in this launcher session (ctx, budget and the other values do not count).
func (l *Launcher) SmokeDue(mode SmokeMode, modelPath, repackMode string) bool {
	switch mode {
	case SmokeOn:
		return true
	case SmokeFirstOnce:
		l.smoke.mu.Lock()
		defer l.smoke.mu.Unlock()
		return !l.smoke.done[smokeKey(modelPath, repackMode)]
	}
	return false
}

func (l *Launcher) markSmoked(modelPath, repackMode string) {
	l.smoke.mu.Lock()
	l.smoke.done[smokeKey(modelPath, repackMode)] = true
	l.smoke.mu.Unlock()
}

// Smoke runs the checklist against the server this launcher serves now (the screen's own
// request, and the start's step when the setting makes it due). A pass counts for first-once.
func (l *Launcher) Smoke(ctx context.Context) (SmokeResult, error) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return SmokeResult{}, ErrNoServer
	}
	s.mu.Lock()
	ready, ended, pre, child := s.ready, s.ended, s.pre, s.child
	s.mu.Unlock()
	if !ready || ended || pre == nil || pre.Config == nil || child == nil {
		return SmokeResult{}, fmt.Errorf("no server is serving")
	}
	r := runSmokeFull(ctx, pre.Config.Address(), func() []string { return readChildStderrComplete(child.ErrLog()) }, psSmokeTimings,
		l.smokeContextFor(pre, pre.Config))
	l.logSmoke(r)
	if r.OK {
		l.markSmoked(pre.ModelPath, pre.RepackMode)
	}
	return r, nil
}
