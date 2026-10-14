package wire

// UX-R11 (version 8; the user's acceptance finding 20 (a) and (b), reviews/_user_8_LAUNCHER-REVIEW_9.md,
// screenshot 064114): on 2 Server gpt-oss-120b answered one request (64 tokens at 5.98 tok/s after a
// 10.4 s prompt), yet the decode graph stayed empty with «samples appear as the server decodes».
// The engine prints a prompt progress line once a prompt has taken 3 s and a decode progress
// line only from 100 decoded tokens on (llama-server b10057 server-context.cpp print_timings_pp,
// print_timings_tg), so after the last prompt progress line the log still reads «prefilling»
// for the whole of a short answer; the status took the phase from that stale flag over the
// slot's live decoding, and the screen keeps a live decode sample only in the decode phase.
// These tests feed the user's server log and the slot answers of each beat through the status
// path (the log parse, the own-request drop, statusFrom with the five-second window) into the
// real screen, and read the plotted columns of the graph box.

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// uxr11UserLog is the user's server log from ready to the end of the request
// (%LOCALAPPDATA%\MoE-Direct\logs\server_20261012T213739Z_err.log lines 23 to 35, verbatim).
var uxr11UserLog = []string{
	"0.02.925.389 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 102400, kv_unified = 'false'",
	"0.02.931.158 I srv  llama_server: model loaded",
	"0.02.931.164 I srv  llama_server: listening on http://127.0.0.1:8093",
	"2.41.917.023 I slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.842 (> 0.100 thold), f_keep = 0.928",
	"2.41.917.079 I slot launch_slot_: id  0 | task 161 | processing task, is_child = 0",
	"2.47.886.567 W srv          stop: cancel task, id_task = 163",
	"2.49.975.258 I slot print_timing: id  0 | task 161 | prompt processing, n_tokens =     61, progress = 0.80, t =   8.06 s / 7.57 tokens per second",
	"2.51.884.085 I slot print_timing: id  0 | task 161 | prompt processing, n_tokens =     72, progress = 0.95, t =   9.97 s / 7.22 tokens per second",
	"3.03.063.847 I slot print_timing: id  0 | task 161 | prompt eval time =   10444.98 ms /    76 tokens (  137.43 ms per token,     7.28 tokens per second)",
	"3.03.063.853 I slot print_timing: id  0 | task 161 |        eval time =   10701.77 ms /    64 tokens (  167.22 ms per token,     5.98 tokens per second)",
	"3.03.063.854 I slot print_timing: id  0 | task 161 |       total time =   21146.75 ms /   140 tokens",
	"3.03.063.855 I slot print_timing: id  0 | task 161 |    graphs reused =         62",
	"3.03.063.893 I slot      release: id  0 | task 161 | stop processing: n_tokens = 139, truncated = 0",
}

// uxr11Req is one request as /slots shows it: launched, decoding from dec (one token every
// per), done at end.
type uxr11Req struct {
	task             int
	prompt, answer   int64
	launch, dec, end time.Duration
	per              time.Duration
}

// slots is the /slots answer at t (since spawn), as the engine's slot JSON has it.
func (q uxr11Req) slots(t time.Duration) any {
	task, busy, n := -1, false, int64(0)
	switch {
	case t >= q.end:
		task, n = q.task, q.answer
	case t >= q.dec:
		task, busy, n = q.task, true, min(q.answer, int64((t-q.dec)/q.per))
	case t >= q.launch:
		task, busy = q.task, true
	}
	body := fmt.Sprintf(`[{"id":0,"id_task":%d,"n_ctx":102400,"is_processing":%t,"n_prompt_tokens":%d,"n_prompt_tokens_cache":0,`+
		`"next_token":[{"has_next_token":%t,"has_new_line":false,"n_remain":-1,"n_decoded":%d}]}]`, task, busy, q.prompt, busy, n)
	var v any
	if err := json.Unmarshal([]byte(body), &v); err != nil {
		panic(err)
	}
	return v
}

type uxr11Tick struct {
	at time.Time
	fn func(time.Time) tea.Msg
}

// uxr11Screen drives the real screen (ui.Model) over the fake backend whose serve status is the
// adapter's status of each beat; timers are kept and fired when due.
type uxr11Screen struct {
	t     *testing.T
	f     *ui.FakeBackend
	m     tea.Model
	clock time.Time
	ticks []uxr11Tick
}

func (s *uxr11Screen) run(c tea.Cmd) {
	if c == nil {
		return
	}
	got := make(chan tea.Msg, 1)
	go func() { got <- c() }()
	var msg tea.Msg
	select {
	case msg = <-got:
	case <-time.After(time.Second):
		return // a command still blocked (an open channel) is left behind
	}
	switch msg := msg.(type) {
	case nil, tea.QuitMsg:
	case tea.BatchMsg:
		for _, x := range msg {
			s.run(x)
		}
	default:
		s.send(msg)
	}
}

func (s *uxr11Screen) send(msg tea.Msg) {
	m, c := s.m.Update(msg)
	s.m = m
	s.run(c)
}

func (s *uxr11Screen) key(k string) {
	if k == "enter" {
		s.send(tea.KeyMsg{Type: tea.KeyEnter})
		return
	}
	s.send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)})
}

// fire sends the timers that are due at the screen's clock.
func (s *uxr11Screen) fire() {
	due := s.ticks
	s.ticks = nil
	var keep []uxr11Tick
	for _, p := range due {
		if p.at.After(s.clock) {
			keep = append(keep, p)
			continue
		}
		s.send(p.fn(s.clock))
	}
	s.ticks = append(keep, s.ticks...)
}

func (s *uxr11Screen) plain() string { return ansi.Strip(s.m.View()) }

// uxr11Serve starts the fake server on the screen and then beats once a second from 3 s to
// until (since spawn): at each beat the status is the adapter's, from the log lines written by
// then and the slot answer of that moment, as Backend.Status makes it (wire.go Status).
func uxr11Serve(t *testing.T, lang string, w int, log []string, q uxr11Req, until time.Duration) *uxr11Screen {
	t.Helper()
	spawn := time.Date(2026, 10, 12, 21, 37, 39, 0, time.UTC)
	readyAt := spawn.Add(2931 * time.Millisecond)
	parsed := core.ParseServerLines(log, spawn)
	var win decodeWindow
	var own ownRequests
	status := func(now time.Time) ui.ServeStatus {
		var seen []core.ServerLine
		for _, ln := range parsed {
			if !ln.Time.After(now) {
				seen = append(seen, ln)
			}
		}
		st := core.ServeStatus{Running: true, Ready: true, Alive: true, PID: 20676, Port: 8093, Address: "http://127.0.0.1:8093",
			ProfileID: "gpt-oss-120b", StartedAt: spawn, ReadyAt: readyAt, Uptime: now.Sub(readyAt),
			Health: core.HTTPRead{OK: true, Code: 200}, Slots: core.HTTPRead{OK: true, Code: 200, Body: q.slots(now.Sub(spawn))},
			Start: core.StartSettings{Alias: "gpt-oss-120b"}}
		st.Timing = own.drop(core.SummarizeTimings(seen), readyAt)
		return statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &win, now, requestInFlight(seen))
	}

	f := ui.NewFake()
	s := &uxr11Screen{t: t, f: f, clock: spawn.Add(3 * time.Second)}
	f.Statuses = []ui.ServeStatus{status(s.clock)}
	s.m = ui.New(f, ui.Options{Lang: lang, Width: w, Height: 40,
		Now: func() time.Time { return s.clock },
		Tick: func(d time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
			s.ticks = append(s.ticks, uxr11Tick{s.clock.Add(d), fn})
			return nil
		}})
	t.Cleanup(f.Close)
	s.run(s.m.Init())
	s.fire() // the first draw
	s.key("enter")
	for d := 3 * time.Second; d <= until; d += time.Second {
		s.clock = spawn.Add(d)
		f.Statuses = []ui.ServeStatus{status(s.clock)}
		s.fire()
	}
	s.key("2")
	if os.Getenv("UXR11_SHOW") != "" {
		t.Logf("\n%s", s.plain())
	}
	return s
}

// plottedAt counts the plotted columns of the k-th graph box from the top (UX-R19: decode 0,
// prefill 1): the blocks on its zero row («│   0┤»).
func plottedAt(t *testing.T, p string, k int) int {
	t.Helper()
	for _, l := range strings.Split(p, "\n") {
		loc := zeroRow.FindStringIndex(l)
		if loc == nil {
			continue
		}
		if k > 0 {
			k--
			continue
		}
		row := l[loc[1]:]
		if j := strings.Index(row, "│"); j >= 0 {
			row = row[:j]
		}
		return len([]rune(strings.TrimSpace(row))) - strings.Count(strings.TrimSpace(row), " ")
	}
	t.Fatalf("no graph zero row %d:\n%s", k, p)
	return 0
}

var zeroRow = regexp.MustCompile(`│ +0┤`)

// plotted counts the graph box's plotted columns: the blocks on its zero row («0┤»).
func plotted(t *testing.T, p string) int {
	t.Helper()
	for _, l := range strings.Split(p, "\n") {
		i := strings.Index(l, "0┤")
		if i < 0 {
			continue
		}
		row := l[i+len("0┤"):]
		if j := strings.Index(row, "│"); j >= 0 {
			row = row[:j]
		}
		return len([]rune(strings.TrimSpace(row))) - strings.Count(strings.TrimSpace(row), " ")
	}
	t.Fatalf("no graph zero row:\n%s", p)
	return 0
}

// the user's request: launched at 2:41.917, 76 prompt tokens read until 2:52.362 (10.44 s),
// 64 answer tokens one every 167.22 ms until 3:03.064
var uxr11User = uxr11Req{task: 161, prompt: 76, answer: 64,
	launch: 2*time.Minute + 41917*time.Millisecond, dec: 2*time.Minute + 52362*time.Millisecond,
	end: 3*time.Minute + 3064*time.Millisecond, per: 167220 * time.Microsecond}

func TestUXR11(t *testing.T) {
	for _, w := range []int{130, 160} {
		// before the request: the empty box and its hint
		p := uxr11Serve(t, ui.LangKorean, w, uxr11UserLog, uxr11User, 2*time.Minute+30*time.Second).plain()
		if !strings.Contains(p, "모델이 답변시 표본 표시") || plotted(t, p) != 0 { // UX-R19: the hint in the user's words
			t.Fatalf("w=%d before the request: want the empty box with its hint:\n%s", w, p)
		}

		// finding 20 (a), the user's case: the answer's live decode samples are plotted
		s := uxr11Serve(t, ui.LangKorean, w, uxr11UserLog, uxr11User, 3*time.Minute+5*time.Second)
		p = s.plain()
		if n := plotted(t, p); n != 1 { // UX-R17: the answer's live samples fill its minute's one column
			t.Fatalf("w=%d after the user's request: %d plotted columns, want the answer's minute (1):\n%s", w, n, p)
		}
		sq := strings.Join(strings.Fields(p), " ") // UX-R27: the average is the DECODE panel's avg
		if strings.Contains(p, "모델이 답변시 표본 표시") || !strings.Contains(sq, "decode — 6.") {
			t.Fatalf("w=%d after the user's request: the hint stays or the average is missing:\n%s", w, p)
		}
		// finding 20 (b), UX-R19 (answer 5): no switch; the prefill graph and its box show on the same
		// screen, under the decode graph
		if strings.Contains(p, "g prefill · TTFT 보기") {
			t.Fatalf("w=%d: the old switch is still named:\n%s", w, p)
		}
		for _, want := range []string{"PREFILL · 요청별 tok/s", "TTFT — 10.4 10.4 10.4 s", "prefill — 7.3 7.3 7.3 tok/s"} { // UX-R27: the PREFILL panel
			if !strings.Contains(sq, want) {
				t.Fatalf("w=%d: %q missing:\n%s", w, want, p)
			}
		}
		if plottedAt(t, p, 1) != 1 {
			t.Fatalf("w=%d: the request's prefill is not plotted in the prefill graph:\n%s", w, p)
		}

		// the same in English
		e := uxr11Serve(t, ui.LangEnglish, w, uxr11UserLog, uxr11User, 3*time.Minute+5*time.Second)
		p = e.plain()
		if n := plotted(t, p); n != 1 || strings.Contains(p, "samples show when the model answers") ||
			strings.Contains(p, "g prefill · TTFT view") {
			t.Fatalf("w=%d English: plotted %d, hint, or the old switch's name:\n%s", w, n, p)
		}
		if !strings.Contains(p, "PREFILL · tok/s per request") || !strings.Contains(strings.Join(strings.Fields(p), " "), "TTFT — 10.4 10.4 10.4 s") ||
			plottedAt(t, p, 1) != 1 {
			t.Fatalf("w=%d English: the prefill graph or its box:\n%s", w, p)
		}
	}

	// an answer shorter than a beat: no live sample can be taken, the request's own rate is
	// plotted at its end and the hint goes
	short := []string{
		"0.02.931.164 I srv  llama_server: listening on http://127.0.0.1:8093",
		"0.10.100.000 I slot launch_slot_: id  0 | task 5 | processing task, is_child = 0",
		"0.10.900.000 I slot print_timing: id  0 | task 5 | prompt eval time =     200.00 ms /    20 tokens (   10.00 ms per token,   100.00 tokens per second)",
		"0.10.900.000 I slot print_timing: id  0 | task 5 |        eval time =     600.00 ms /     4 tokens (  150.00 ms per token,     6.67 tokens per second)",
		"0.10.900.000 I slot print_timing: id  0 | task 5 |       total time =     800.00 ms /    24 tokens",
		"0.10.900.000 I slot      release: id  0 | task 5 | stop processing: n_tokens = 24, truncated = 0",
	}
	q := uxr11Req{task: 5, prompt: 20, answer: 4, launch: 10100 * time.Millisecond, dec: 10300 * time.Millisecond,
		end: 10900 * time.Millisecond, per: 150 * time.Millisecond}
	s := uxr11Serve(t, ui.LangEnglish, 160, short, q, 20*time.Second)
	p := s.plain()
	if plotted(t, p) != 1 || strings.Contains(p, "samples show when the model answers") ||
		!strings.Contains(strings.Join(strings.Fields(p), " "), "decode — 6.7 6.7 6.7 tok/s") { // UX-R27: the DECODE panel
		t.Fatalf("an answer shorter than a beat is not plotted:\n%s", p)
	}

	// finding 20 (a), stopped page (UX-R7 addendum): the run summary's whole-run graph plots the
	// same finished request, not «no decode samples in this run»
	s.send(tea.KeyMsg{Type: tea.KeyCtrlF})
	s.clock = s.clock.Add(time.Second)
	s.fire()
	p = s.plain()
	if os.Getenv("UXR11_SHOW") != "" {
		t.Logf("\n%s", p)
	}
	if !strings.Contains(p, "DECODE · tok/s · whole run") || strings.Contains(p, "no decode samples in this run") ||
		plotted(t, p) != 1 || !strings.Contains(p, "average 6.7 · lowest 6.7 tok/s") {
		t.Fatalf("the stopped page's whole-run graph lacks the finished request:\n%s", p)
	}
}
