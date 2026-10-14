// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// ---------- harness ----------

type pendingTick struct {
	d  time.Duration
	at time.Time
	fn func(time.Time) tea.Msg
}

type harness struct {
	t     *testing.T
	f     *FakeBackend
	m     *Model
	clock time.Time
	ticks []pendingTick
	quit  bool
}

func newH(t *testing.T, f *FakeBackend, lang string, w, h int) *harness {
	t.Helper()
	hs := &harness{t: t, f: f, clock: time.Date(2026, 10, 10, 9, 0, 0, 0, time.UTC)}
	hs.m = New(f, Options{Lang: lang, Width: w, Height: h,
		Now: func() time.Time { return hs.clock },
		Tick: func(d time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
			hs.ticks = append(hs.ticks, pendingTick{d, hs.clock.Add(d), fn})
			return nil
		}})
	t.Cleanup(f.Close)
	hs.run(hs.m.Init())
	hs.drawn()
	return hs
}

// drawn sends the first-draw message Init scheduled (GL-13): the harness has no renderer, so
// the first screen counts as drawn once Init has run; the file check is then asked for.
func (h *harness) drawn() {
	var keep []pendingTick
	var due []pendingTick
	for _, p := range h.ticks {
		if p.d == firstDrawWait {
			due = append(due, p)
		} else {
			keep = append(keep, p)
		}
	}
	h.ticks = keep
	for _, p := range due {
		h.send(p.fn(h.clock))
	}
}

// run executes a command off the update path, as Bubble Tea does, and feeds
// its message back. A command still blocked after a short wait (a channel the
// fake holds open) is left behind.
func (h *harness) run(c tea.Cmd) {
	if c == nil {
		return
	}
	got := make(chan tea.Msg, 1)
	go func() { got <- c() }()
	var msg tea.Msg
	select {
	case msg = <-got:
	case <-time.After(100 * time.Millisecond):
		return
	}
	switch msg := msg.(type) {
	case nil:
	case tea.BatchMsg:
		for _, x := range msg {
			h.run(x)
		}
	case tea.QuitMsg:
		h.quit = true
	default:
		h.send(msg)
	}
}

func (h *harness) send(msg tea.Msg) {
	_, c := h.m.Update(msg)
	h.run(c)
}

var keyTypes = map[string]tea.KeyType{
	"enter": tea.KeyEnter, "tab": tea.KeyTab, "esc": tea.KeyEsc, "up": tea.KeyUp, "down": tea.KeyDown,
	"left": tea.KeyLeft, "right": tea.KeyRight, "end": tea.KeyEnd,
	"ctrl+c": tea.KeyCtrlC, "ctrl+f": tea.KeyCtrlF, "ctrl+x": tea.KeyCtrlX, "ctrl+r": tea.KeyCtrlR,
	"ctrl+d": tea.KeyCtrlD, "ctrl+s": tea.KeyCtrlS,
}

func keyMsg(k string) tea.KeyMsg {
	if t, ok := keyTypes[k]; ok {
		return tea.KeyMsg{Type: t}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)}
}

func (h *harness) keys(ks ...string) {
	for _, k := range ks {
		h.send(keyMsg(k))
	}
}

func (h *harness) advance(d time.Duration) { h.clock = h.clock.Add(d) }

// fireDue sends the messages of the timers that are due.
func (h *harness) fireDue() {
	var keep []pendingTick
	due := h.ticks
	h.ticks = nil
	for _, p := range due {
		if !p.at.After(h.clock) {
			h.send(p.fn(h.clock))
		} else {
			keep = append(keep, p)
		}
	}
	h.ticks = append(keep, h.ticks...)
}

func (h *harness) beat() { h.send(tickMsg{h.clock}) }

func (h *harness) plain() string { return ansi.Strip(h.m.View()) }

func (h *harness) want(id string) {
	h.t.Helper()
	if got := h.m.ScreenID(); got != id {
		h.t.Fatalf("screen %s, want %s\n%s", got, id, h.plain())
	}
}

func (h *harness) calls(prefix string) int {
	n := 0
	for _, c := range h.f.Calls() {
		if strings.HasPrefix(c, prefix) {
			n++
		}
	}
	return n
}

// ---------- reaching every screen ----------

type scenario struct {
	setup func(f *FakeBackend)
	keys  []string
}

func serving(extra ...string) []string { return append([]string{"enter"}, extra...) }

var scenarios = map[string]scenario{
	"H1": {},
	"H2": {keys: serving("1", "tab", "right")},
	"H3": {setup: func(f *FakeBackend) {
		for i := range f.Cat.Models {
			f.Cat.Models[i].Present = false
		}
	}},
	"H4": {setup: func(f *FakeBackend) {
		f.Cat.Models[0].Setup = SetupCutOff
		f.Cat.Models[0].Partial = PartialSetup{LayersDone: K(25), LayersTotal: K(48), FilesGB: K(37), When: "yesterday 22:14"}
	}},
	"H5": {keys: []string{"5"}},
	"H6": {setup: func(f *FakeBackend) {
		f.StartHold = true
		f.StartEvs = []StartEvent{{Model: "qwen35-122b", Elapsed: 30 * time.Second,
			Setup: &SetupProgress{Name: "repack", Done: K(25), Total: K(48), DiskMBs: K(320)},
			Steps: []Step{{Name: "bundle", State: StepDone}, {Name: "model", State: StepDone}, {Name: "repack", State: StepRunning},
				{Name: "verify", State: StepTodo}, {Name: "start", State: StepTodo}}}}
	}, keys: []string{"enter"}},
	"H7": {setup: func(f *FakeBackend) {
		f.StartEvs = []StartEvent{{Model: "qwen35-122b", Steps: []Step{{Name: "start server", State: StepFailed, Detail: "bind failed"}},
			Failure: &Failure{Stage: "start", What: "did not start", Why: "port 8093 is busy", Holder: "python.exe", PID: 11284,
				Fixes: []Fix{{Key: "p", Label: "other port", Action: ActOtherPort}}, LogTail: []string{"[server] bind: address in use"}}}}
	}, keys: []string{"enter"}},
	"H8":  {keys: serving()},
	"H8Q": {setup: idleServer, keys: serving("q")}, // no reply running (U-27 warns otherwise)
	"H9": {setup: func(f *FakeBackend) {
		f.Statuses[0].Slow, f.Statuses[0].UsualTokS, f.Statuses[0].LastDecodeTokS = true, K(5.5), K(3.9)
		f.Statuses[0].SlowNote = "seen: chrome.exe +5.0 GB · cause not proven"
	}, keys: serving()},
	"H10": {keys: serving("3")},
	"H11": {keys: []string{"4"}},
	"H12": {keys: serving("ctrl+f")},
	"H13": {keys: []string{"?"}},
	"H14": {keys: []string{"3"}},
	"H15": {setup: func(f *FakeBackend) { f.StartHold = true; f.StartEvs = f.StartEvs[:2] }, keys: []string{"enter"}},
	"HF":  {keys: []string{"?", "tab"}},
	"HQ":  {setup: func(f *FakeBackend) { idleServer(f); f.StopHold = true }, keys: serving("q", "q")},
	"S0":  {keys: []string{"2"}},
}

// H5K, the key page, is the next version (L:4265 ②; GL-02)
var designIDs = []string{"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15", "HF", "H8Q", "HQ"}

// idleServer makes the fake's serving server answer no request: the quit is the
// idle quit sequence (edition 3.5), not the U-27 one.
func idleServer(f *FakeBackend) {
	for i := range f.Statuses {
		f.Statuses[i].Phase = ReqNone
	}
}

func reach(t *testing.T, id, lang string, w, h int) *harness {
	t.Helper()
	sc, ok := scenarios[id]
	if !ok {
		t.Fatalf("no scenario for %s", id)
	}
	f := NewFake()
	if sc.setup != nil {
		sc.setup(f)
	}
	hs := newH(t, f, lang, w, h)
	hs.keys(sc.keys...)
	hs.beat() // one heartbeat: status, memory, log polls
	return hs
}

var sizes = [][2]int{{120, 30}, {80, 28}, {64, 24}, {40, 20}}

// TestEveryScreenRenders: every design screen ID is reached through Update
// with the fake and renders without panic at 120×30, 80×28, 64×24 and 40×20,
// in English and Korean, within the window (decision ⑤).
func TestEveryScreenRenders(t *testing.T) {
	ids := append(append([]string{}, designIDs...), "S0")
	for _, id := range ids {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				name := fmt.Sprintf("%s_%dx%d_%s", id, sz[0], sz[1], lang)
				t.Run(name, func(t *testing.T) {
					hs := reach(t, id, lang, sz[0], sz[1])
					hs.want(id)
					var out string
					func() {
						defer func() {
							if p := recover(); p != nil {
								t.Fatalf("panic: %v", p)
							}
						}()
						out = hs.m.View()
					}()
					ls := strings.Split(out, "\n")
					if len(ls) > sz[1] {
						t.Fatalf("%d lines for a window of %d", len(ls), sz[1])
					}
					for i, l := range ls {
						if w := lipgloss.Width(l); w > sz[0] {
							t.Fatalf("line %d is %d wide for a window of %d: %q", i, w, sz[0], ansi.Strip(l))
						}
					}
				})
			}
		}
	}
}

// TestEveryScreenFallbackGlyphs: with the fallback set no two-cell glyph is
// left and every line keeps its width.
func TestEveryScreenFallbackGlyphs(t *testing.T) {
	for _, id := range designIDs {
		hs := reach(t, id, LangEnglish, 120, 30)
		design := hs.m.View()
		hs.m.glyphs = GlyphFallback
		fb := hs.m.View()
		for _, g := range ProbeGlyphs() {
			if strings.Contains(fb, g) {
				t.Errorf("%s: fallback output still has %q", id, g)
			}
		}
		dl, fl := strings.Split(design, "\n"), strings.Split(fb, "\n")
		if len(dl) != len(fl) {
			t.Fatalf("%s: line count %d vs %d", id, len(dl), len(fl))
		}
		for i := range dl {
			if lipgloss.Width(dl[i]) != lipgloss.Width(fl[i]) {
				t.Errorf("%s line %d: width %d vs %d", id, i, lipgloss.Width(dl[i]), lipgloss.Width(fl[i]))
			}
		}
	}
}

// ---------- the slice flow ----------

// TestFlowPickPreflightStartReadyServeStop runs the slice through Update
// with the fake: pick, preflight, start with its steps, ready, serve, stop.
func TestFlowPickPreflightStartReadyServeStop(t *testing.T) {
	f := NewFake()
	f.StartHold = true // the channel stays open after the first two events
	f.StartEvs = f.StartEvs[:2]
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.want("H1")
	if !strings.Contains(hs.plain(), "Qwen3.5-122B") || !strings.Contains(hs.plain(), "DSV4-Flash") {
		t.Fatalf("catalog rows missing:\n%s", hs.plain())
	}
	// H1's numbers (decision ⑧, review item 13) come from the backend, each
	// with its name: this model, in use (32 − 24.6 available), left after start
	for _, w := range []string{"RAM · 32 GB", "19.0 GB this model", "in use 7.4 · left", " 5.6 "} {
		if !strings.Contains(hs.plain(), w) {
			t.Fatalf("H1 RAM tile lacks %q:\n%s", w, hs.plain())
		}
	}
	// pick: down to the missing model, which refuses to start, then back
	pre := hs.calls("Preflight")
	hs.keys("down")
	hs.keys("enter")
	if hs.calls("Preflight") != pre || !strings.Contains(hs.m.notice, "missing") {
		t.Fatalf("a missing model must not be checked or started (calls %v)", f.Calls())
	}
	hs.keys("up")
	// Update itself calls nothing: the call happens when the command runs
	before := len(f.Calls())
	_, c := hs.m.Update(keyMsg("enter"))
	if len(f.Calls()) != before {
		t.Fatalf("Update called the backend: %v", f.Calls()[before:])
	}
	hs.run(c) // preflight → start → the two step events (ready has not come)
	// one check when the catalog came, one on the way back up, one for the start
	if hs.calls("Preflight qwen35-122b") != 3 || hs.calls("Start qwen35-122b") != 1 {
		t.Fatalf("calls %v", f.Calls())
	}
	hs.want("H15")
	if !strings.Contains(hs.plain(), "START STEPS") || !strings.Contains(hs.plain(), "start server") {
		t.Fatalf("start steps missing:\n%s", hs.plain())
	}
	// a second run of the flow where the start reaches /health ready
	f2 := NewFake()
	hs2 := newH(t, f2, LangEnglish, 120, 30)
	hs2.keys("enter")
	hs2.want("H8")
	if hs2.calls("Status") < 1 {
		t.Fatalf("no status read after ready: %v", f2.Calls())
	}
	hs2.beat()
	p := hs2.plain()
	for _, w := range []string{"http://127.0.0.1:8093/v1", "/health ok", "PID 4242", "9.4k", "12.3k"} {
		if !strings.Contains(p, w) {
			t.Errorf("serve page lacks %q", w)
		}
	}
	// stop: ctrl+f unloads, the launcher stays with the summary
	hs2.keys("ctrl+f")
	hs2.want("H12")
	if hs2.calls(fmt.Sprintf("Stop %d", StopUnload)) != 1 || hs2.quit {
		t.Fatalf("unload: calls %v quit %v", f2.Calls(), hs2.quit)
	}
	if !strings.Contains(hs2.plain(), "38 requests") {
		t.Fatalf("summary missing:\n%s", hs2.plain())
	}
	// ⏎ starts again with the same model and settings
	hs2.keys("enter")
	hs2.want("H8")
	if hs2.calls("Start qwen35-122b") != 2 {
		t.Fatalf("start again: %v", f2.Calls())
	}
}

// TestRefusedPreflightOffersNoStart: conflict 7 — a refused preflight shows
// why on 3 Memory and offers no ⏎.
func TestRefusedPreflightOffersNoStart(t *testing.T) {
	f := NewFake()
	f.Pre.Refused = true
	f.Pre.Verdict = VerdictCannot
	f.Pre.Refusals = []string{"budget above available RAM"}
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.want("H14")
	if hs.calls("Start") != 0 {
		t.Fatalf("started after a refusal: %v", f.Calls())
	}
	p := hs.plain()
	if !strings.Contains(p, "refused: budget above available RAM") || strings.Contains(p, "starts anyway") {
		t.Fatalf("H14 after a refusal:\n%s", p)
	}
	hs.keys("enter")
	if hs.calls("Start") != 0 {
		t.Fatalf("⏎ on a refused H14 started: %v", f.Calls())
	}
}

// TestSettingsNoticeAndLock: only editable rows change; a QD change shows the
// backend's preflight notice (the user's decision of 2026-10-10 00:44 KST).
func TestSettingsNoticeAndLock(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("tab", "down", "down", "right")
	if v := hs.m.edits["qwen35-122b"]["qd"]; v != "16" {
		t.Fatalf("qd edit %q", v)
	}
	if !strings.Contains(hs.plain(), "default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured") { // GL-20: OQ-5 (b)
		t.Fatalf("QD notice missing:\n%s", hs.plain())
	}
	// UX-R10 (finding 17): every row stands in the box, no «a»; down to the KV type, a locked
	// row (before UX-R10 «a» moved the cursor back to the top and ←→ changed the queue)
	hs.keys("down", "down", "down", "down", "right")
	if _, ok := hs.m.edits["qwen35-122b"]["kv"]; ok {
		t.Fatalf("a locked row changed")
	}
	if !strings.Contains(hs.plain(), "locked") {
		t.Fatalf("lock not shown:\n%s", hs.plain())
	}
}

// ---------- decision ③: the load bar ----------

func TestLoadBarRule(t *testing.T) {
	first := loadProgress(12*time.Second, U)
	if !first.SecondsOnly || first.Frac != 0 {
		t.Fatalf("first run: %+v", first)
	}
	if np := loadProgress(12*time.Second, Val{}); !np.SecondsOnly {
		t.Fatalf("not ported last duration: %+v", np)
	}
	within := loadProgress(10*time.Second, K(20))
	if within.SecondsOnly || within.Slower || within.Pct != 50 {
		t.Fatalf("within: %+v", within)
	}
	at := loadProgress(20*time.Second, K(20)) // S8-LA-14: at or past D, held with the note (GL-07)
	if at.Pct != 99 || !at.Slower {
		t.Fatalf("at the last duration: %+v", at)
	}
	over := loadProgress(31*time.Second, K(20))
	if over.Pct != 99 || !over.Slower || over.Frac != maxRunningFrac {
		t.Fatalf("over: %+v", over)
	}
}

func startingAt(t *testing.T, lang string, last Val, elapsed time.Duration) string {
	t.Helper()
	f := NewFake()
	f.StartHold = true
	f.StartEvs = f.StartEvs[1:2]
	f.StartEvs[0].LastS = last
	f.StartEvs[0].Elapsed = elapsed
	hs := newH(t, f, lang, 120, 30)
	hs.keys("enter")
	hs.want("H15")
	return hs.plain()
}

func TestLoadBarOnScreen(t *testing.T) {
	p := startingAt(t, LangEnglish, U, 12*time.Second)
	if !strings.Contains(p, "first time: seconds only") || strings.Contains(p, " % ") {
		t.Fatalf("first run shows a percent:\n%s", p)
	}
	p = startingAt(t, LangEnglish, K(20), 10*time.Second)
	if !strings.Contains(p, " 50 %") || strings.Contains(p, "slower than last time") {
		t.Fatalf("within:\n%s", p)
	}
	p = startingAt(t, LangEnglish, K(20), 31*time.Second)
	if !strings.Contains(p, "99 % · slower than last time") || strings.Contains(p, "100") {
		t.Fatalf("over:\n%s", p)
	}
	p = startingAt(t, LangKorean, K(20), 31*time.Second)
	if !strings.Contains(p, "지난번보다 느림") {
		t.Fatalf("over, Korean:\n%s", p)
	}
}

// The cache-save bar follows the same rule; warm-start save is not ported.
func TestCacheSaveBar(t *testing.T) {
	f := NewFake()
	idleServer(f)
	f.StopHold = true
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "q", "q")
	hs.want("HQ")
	if p := hs.plain(); !strings.Contains(p, "save: —") || !strings.Contains(p, "the next start is cold") {
		t.Fatalf("not-ported save:\n%s", p)
	}
	if strings.Contains(hs.plain(), "cut the reply") {
		t.Fatalf("decision ①: a cut step is drawn")
	}
	f = NewFake()
	idleServer(f)
	f.StopHold = true
	for _, mode := range []StopMode{StopQuit} {
		evs := f.StopEvs[mode]
		evs[0].Save = SaveProgress{State: StepRunning, Elapsed: time.Second, LastS: K(4), Tokens: K(12300)}
		evs[0].Steps[0] = Step{Name: "save prompt cache", State: StepRunning}
		evs[0].Steps[1].State = StepTodo
	}
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "q", "q")
	if p := hs.plain(); !strings.Contains(p, " 25 %") {
		t.Fatalf("save bar within its last duration:\n%s", p)
	}
}

// ---------- the two-press keys ----------

func TestQTwiceTiming(t *testing.T) {
	// within 2 s while serving: H8Q, then the quit flow, then the program ends
	// (no reply running; the U-27 case is TestS8LA03)
	f := NewFake()
	idleServer(f)
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "q")
	hs.want("H8Q")
	if !strings.Contains(hs.plain(), "q again: unload the model and exit") {
		t.Fatalf("H8Q prompt missing:\n%s", hs.plain())
	}
	hs.advance(1500 * time.Millisecond)
	hs.keys("q")
	if hs.calls(fmt.Sprintf("Stop %d", StopQuit)) != 1 || !hs.quit {
		t.Fatalf("q twice within 2 s: calls %v quit %v", f.Calls(), hs.quit)
	}

	// after 2 s the first press has lapsed: the second q arms again
	f = NewFake()
	idleServer(f)
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "q")
	hs.advance(2500 * time.Millisecond)
	hs.keys("q")
	hs.want("H8Q")
	if hs.calls("Stop") != 0 || hs.quit {
		t.Fatalf("a lapsed q quit: %v", f.Calls())
	}
	// the 2 s timer disarms by itself
	hs.advance(2100 * time.Millisecond)
	hs.fireDue()
	hs.want("H8")

	// another key cancels
	hs.keys("q", "c")
	hs.want("H8")
	hs.keys("q")
	hs.want("H8Q")
	if hs.calls("Stop") != 0 {
		t.Fatalf("q, other key, q stopped the server")
	}

	// no server: q twice closes at once
	f = NewFake()
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("q")
	hs.advance(time.Second)
	hs.keys("q")
	if !hs.quit || hs.calls("Stop") != 0 {
		t.Fatalf("q twice without a server: quit %v calls %v", hs.quit, f.Calls())
	}
	// the Hangul-mode q (ㅂ) counts too
	f = NewFake()
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("ㅂ", "ㅂ")
	if !hs.quit {
		t.Fatalf("ㅂ ㅂ did not quit")
	}
}

func TestCtrlCTwiceTiming(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "ctrl+c")
	if !strings.Contains(hs.plain(), "ctrl+c again closes now") || hs.quit {
		t.Fatalf("first ctrl+c:\n%s", hs.plain())
	}
	hs.advance(1900 * time.Millisecond)
	hs.keys("ctrl+c")
	if !hs.quit || hs.calls(fmt.Sprintf("Stop %d", StopForce)) != 1 {
		t.Fatalf("ctrl+c twice: quit %v calls %v", hs.quit, f.Calls())
	}

	f = NewFake()
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "ctrl+c")
	hs.advance(2100 * time.Millisecond)
	hs.keys("ctrl+c")
	if hs.quit || hs.calls("Stop") != 0 {
		t.Fatalf("a lapsed ctrl+c closed: %v", f.Calls())
	}
	hs.advance(2100 * time.Millisecond)
	hs.fireDue()
	if hs.m.frcArmed {
		t.Fatalf("the 2 s timer did not disarm")
	}
	// another key cancels the first press
	hs.keys("ctrl+c", "c", "ctrl+c")
	if hs.quit {
		t.Fatalf("ctrl+c, other key, ctrl+c closed")
	}
}

// TestStopKeys: ctrl+x cancels a start; decision ①: ctrl+x does nothing while
// serving (no reply cut).
func TestStopKeys(t *testing.T) {
	f := NewFake()
	f.StartHold = true
	f.StartEvs = f.StartEvs[:1]
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.want("H15")
	hs.keys("ctrl+x")
	if hs.calls(fmt.Sprintf("Stop %d", StopCancel)) != 1 {
		t.Fatalf("ctrl+x while starting: %v", f.Calls())
	}
	hs.want("S0")

	f = NewFake()
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "ctrl+x")
	hs.want("H8")
	if hs.calls("Stop") != 0 {
		t.Fatalf("ctrl+x while serving stopped something: %v", f.Calls())
	}
	if strings.Contains(hs.plain(), "cut") {
		t.Fatalf("decision ①: a cut is offered:\n%s", hs.plain())
	}
}

// TestPlaces: ←→ walks the five places and wraps, 1-5 jump, ? opens help.
func TestPlaces(t *testing.T) {
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	want := []string{"S0", "H14", "H11", "H5", "H1"}
	for _, w := range want {
		hs.keys("right")
		hs.want(w)
	}
	hs.keys("left")
	hs.want("H5")
	hs.keys("1")
	hs.want("H1")
	hs.keys("?")
	hs.want("H13")
	hs.keys("esc")
	hs.want("H1")
}

// ---------- decision ④: the glyph probe ----------
// The measurement itself is TestS8LA04 (glyph_test.go, GL-03).

func TestGlyphProbe(t *testing.T) {
	if got := parseCPR([]byte("x\x1b[12;3Ry\x1b[A\x1b[1;80R")); fmt.Sprint(got) != "[3 80]" {
		t.Fatalf("parseCPR %v", got)
	}
	if GlyphFallback.Apply("● ▲ ▶ · …") != "* ! ▸ . ~" {
		t.Fatalf("fallback %q", GlyphFallback.Apply("● ▲ ▶ · …"))
	}
}

// ---------- language ----------

func TestLanguage(t *testing.T) {
	if pickLang([]string{"ko-KR", "en-US"}) != LangKorean || pickLang([]string{"en-US", "ko-KR"}) != LangEnglish || pickLang(nil) != LangEnglish {
		t.Fatalf("pickLang")
	}
	hs := newH(t, NewFake(), LangKorean, 120, 30)
	if p := hs.plain(); !strings.Contains(p, "모델") || !strings.Contains(p, "설정") {
		t.Fatalf("Korean places missing:\n%s", p)
	}
	// 5 Settings: the language row switches for this run (the second
	// launcher row since the network row left, conflict 17)
	hs.keys("5", "tab", "down", "down", "right")
	if hs.m.r.lang != LangEnglish {
		t.Fatalf("language switch: %q", hs.m.r.lang)
	}
}

// TestTinyWindows: far below the test sizes the screens still never panic.
func TestTinyWindows(t *testing.T) {
	for _, id := range designIDs {
		for _, sz := range [][2]int{{20, 8}, {1, 1}} {
			hs := reach(t, id, LangEnglish, sz[0], sz[1])
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("%s at %dx%d: panic: %v", id, sz[0], sz[1], p)
					}
				}()
				out := hs.m.View()
				if n := len(strings.Split(out, "\n")); n > sz[1] {
					t.Fatalf("%s at %dx%d: %d lines", id, sz[0], sz[1], n)
				}
			}()
		}
	}
}
