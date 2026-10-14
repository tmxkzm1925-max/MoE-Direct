// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// The Bubble Tea program. Update never calls the Backend: every call is a
// tea.Cmd (run by Bubble Tea off the key loop) whose answer comes back as a
// message, one call per action.

package ui

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// place indexes
const (
	plModels = iota
	plServer
	plMemory
	plLog
	plSettings
)

// armWindow is how long a two-press key waits for its second press
// (q twice, ctrl+c twice; design 5.8: the second q within 2 s).
const armWindow = 2 * time.Second

// pollEvery is the heartbeat: serve status every beat, memory every two.
const pollEvery = time.Second

// memEvery is how often 1 Models and 3 Memory read the memory again (review
// item 13: the free RAM is measured at every draw, not kept from a check).
const memEvery = 2 * time.Second

// catEvery is how often 1 Models reads the model list again while it is
// shown; the backend's list call is cheap (names, sizes, modification times,
// no hashing), so a model deleted while the launcher runs leaves the list
// (review item 31).
const catEvery = 5 * time.Second

// preEvery is how often a shown preflight is run again while no server runs,
// so a verdict never stands on a stale measurement (review items 13, 22).
const preEvery = 10 * time.Second

// sample is one live decode reading with its time (review item 21: the graph
// bins columns by time).
type sample struct {
	at time.Time
	v  float64
}

type phase uint8

const (
	phNone phase = iota
	phStarting
	phServing
	phFailed
	phStopping
	phStopped
)

// Options set up a Model.
type Options struct {
	Lang   string   // LangEnglish or LangKorean; "" starts in English (UX-R1)
	Glyphs GlyphSet // from Probe
	Width  int      // first size until Bubble Tea reports the window
	Height int
	// Now and Tick are the clock; tests replace them.
	Now  func() time.Time
	Tick func(time.Duration, func(time.Time) tea.Msg) tea.Cmd
	// Keys is the action-to-key table; nil is the default layout (GL-02).
	Keys KeyTable
}

// Model is the launcher's screen state.
type Model struct {
	b      Backend
	r      rc
	glyphs GlyphSet
	now    func() time.Time
	tick   func(time.Duration, func(time.Time) tea.Msg) tea.Cmd
	auto   string // the display language
	langAt int    // 0 auto, 1 English, 2 Korean (5 Settings, this run only)

	place     int
	help      bool
	helpTiles bool // help's second page: the HF tile strip
	helpFrom  int

	// 1 Models
	cat       Catalog
	catOK     bool
	catBusy   bool
	catAt     time.Time
	catErr    string
	sel       int
	focusSet  bool
	setCur    int
	showAbout bool
	views     map[string]SettingsView
	edits     map[string]Settings
	h4Skip    map[string]bool

	// the plan box of a setup before a start (conflict 9)
	plan    *RepackPlan
	planSet Settings
	planID  string // UX-R8: the row ID the plan was asked for; its ⏎ starts this model
	// the launcher's own file check (conflict 18-2); a start waits for it
	bundle     BundleCheck
	bundleBusy bool
	drawn      bool // the first screen is drawn and the file check asked for (GL-13)
	startWait  string
	// where this run's settings came from (conflict 3): the label and the
	// changed values, kept from the moment of the start
	runLabel string
	runEdits []string

	// the typed-value box of a settings row (review item 30)
	inputOn bool
	// inputLauncher: the box types a launcher setting of 5 Settings (the
	// prompt-cache cap, GL-16), not a model setting
	inputLauncher bool
	inputKey      string
	inputBuf      string
	inputErr      string

	pre     *Preflight
	preKey  string
	preBusy bool
	preAt   time.Time

	// 2 Server
	ph         phase
	runModel   string
	runSet     Settings
	startEv    *StartEvent
	startAt    time.Time
	setup      *setupMemo // UX-R14: this start's setup so far
	stopEv     *StopEvent
	stopAt     time.Time
	stopMode   StopMode
	status     *ServeStatus
	statusBusy bool
	samples    []sample // live decode readings of this run
	// GL-18 (LFS Q4.A24, U-36 (i)): the per-request samples of this run, in memory only, and
	// the graph switch (g): false = DECODE, true = PREFILL · TTFT.
	reqSamples []ReqSample
	failure    *Failure
	result     *StopResult
	exitAfter  bool
	exiting    bool

	// 3 Memory
	mem     *MemoryReading
	memBusy bool
	memAt   time.Time
	appCur  int

	// 4 Log
	logs      []LogLine
	logBusy   bool
	logBack   int // screen lines scrolled up from the newest; 0 follows (review items 6, 7, 20)
	logNew    int // lines that arrived while the view was held
	logFilter int // tab: 0 all, 1 warnings, 2 errors (review item 18)

	// 5 Settings
	launcher  *LauncherInfo
	set5Row   int // -1: no row chosen (arrived by ←→ or 1-5)
	set5Sec   int // 0 folders, 1 launcher, 2 where the launcher writes (review items 8, 24)
	beats     int
	quitArmed bool
	// quitReply: a reply was running at the first q (U-27); quitServing: the
	// second q stopped a serving server; quitLog: the quit steps run (S8-LA-03)
	quitReply   bool
	quitServing bool
	quitLog     []QuitStep
	// keys is the action-to-key table (GL-02)
	keys     KeyTable
	quitAt   time.Time
	quitSeq  int
	frcArmed bool
	frcAt    time.Time
	frcSeq   int
	notice   string
	taskLine string // GL-09: the line shown once Task Manager has opened (U-35 (i))
}

// New makes a Model for a backend.
func New(b Backend, o Options) *Model {
	m := &Model{
		b: b, glyphs: o.Glyphs, now: o.Now, tick: o.Tick,
		views: map[string]SettingsView{}, edits: map[string]Settings{}, h4Skip: map[string]bool{},
		set5Row: -1,
		// conflict 18-2: until the file check answers, a start waits for it (GL-13)
		bundle: BundleCheck{State: BundleChecking},
	}
	if m.now == nil {
		m.now = time.Now
	}
	if m.tick == nil {
		m.tick = tea.Tick
	}
	m.keys = o.Keys
	if m.keys == nil {
		m.keys = DefaultKeyTable()
	}
	// UX-R1 (the user's acceptance finding 1): auto starts in English whatever the
	// display language; 5 Settings switches the run to Korean, and -lang ko starts in it
	m.auto = o.Lang
	if m.auto == "" {
		m.auto = LangEnglish
	}
	m.r = rc{W: o.Width, H: o.Height, lang: m.auto, shown: new(Parts)}
	if m.r.W <= 0 {
		m.r.W = 80
	}
	if m.r.H <= 0 {
		m.r.H = 28
	}
	return m
}

// Run probes the glyph width and runs the program on the console, in English
// (UX-R1), until it quits.
func Run(b Backend) error {
	p := Probe(ProbeWait)
	m := New(b, Options{Glyphs: p.Set})
	_, err := tea.NewProgram(m, tea.WithAltScreen()).Run()
	return err
}

// displayLanguage is the OS display language (lang_windows.go, lang_other.go).
// Since UX-R1 no start reads it: the launcher starts in English. It stays as the
// seam the UX-R1 test sets to a Korean display language.
var displayLanguage = DisplayLanguage

// ---------- messages ----------

type (
	catalogMsg struct {
		c   Catalog
		err error
	}
	settingsMsg struct {
		model string
		v     SettingsView
		err   error
	}
	preflightMsg struct {
		key       string
		model     string
		set       Settings
		p         Preflight
		err       error
		thenStart bool
	}
	startOpenMsg struct {
		ch  <-chan StartEvent
		err error
	}
	startEvMsg struct {
		ev StartEvent
		ok bool
		ch <-chan StartEvent
	}
	statusMsg struct {
		s   ServeStatus
		err error
	}
	memMsg struct {
		m   MemoryReading
		err error
	}
	logMsg struct {
		ls  []LogLine
		err error
	}
	launcherMsg struct {
		l   LauncherInfo
		err error
	}
	stopOpenMsg struct {
		ch  <-chan StopEvent
		err error
	}
	stopEvMsg struct {
		ev StopEvent
		ok bool
		ch <-chan StopEvent
	}
	actionMsg struct {
		id  ActionID
		res ActionResult
		err error
	}
	savedMsg struct {
		model string
		v     SettingsView
		err   error
	}
	checkedMsg struct {
		model, key, value string
		err               error
	}
	planMsg struct {
		model string
		set   Settings
		p     RepackPlan
		err   error
	}
	// drawnMsg: the first screen is drawn; the launcher file check may start (GL-13)
	drawnMsg  struct{}
	bundleMsg struct {
		b   BundleCheck
		err error
	}
	tickMsg   struct{ t time.Time }
	disarmMsg struct {
		kind armed
		seq  int
	}
)

// ---------- commands (each one Backend call) ----------

func (m *Model) cmdCatalog() tea.Cmd {
	m.catBusy = true
	m.catAt = m.now()
	b := m.b
	return func() tea.Msg { c, err := b.Catalog(); return catalogMsg{c, err} }
}

func (m *Model) cmdSettings(model string) tea.Cmd {
	b := m.b
	return func() tea.Msg { v, err := b.ModelSettings(model); return settingsMsg{model, v, err} }
}

func copySet(s Settings) Settings {
	o := Settings{}
	for k, v := range s {
		o[k] = v
	}
	return o
}

func setKey(model string, s Settings) string { return fmt.Sprintf("%s|%v", model, s) }

func (m *Model) cmdPreflight(model string, s Settings, thenStart bool) tea.Cmd {
	m.preBusy = true
	m.preAt = m.now()
	b := m.b
	s = copySet(s)
	k := setKey(model, s)
	return func() tea.Msg {
		p, err := b.Preflight(model, s)
		return preflightMsg{key: k, model: model, set: s, p: p, err: err, thenStart: thenStart}
	}
}

func (m *Model) cmdStart(model string, s Settings) tea.Cmd {
	b := m.b
	return func() tea.Msg { ch, err := b.Start(model, s); return startOpenMsg{ch, err} }
}

func waitStart(ch <-chan StartEvent) tea.Cmd {
	return func() tea.Msg { ev, ok := <-ch; return startEvMsg{ev, ok, ch} }
}

func (m *Model) cmdStatus() tea.Cmd {
	m.statusBusy = true
	b := m.b
	return func() tea.Msg { s, err := b.Status(); return statusMsg{s, err} }
}

func (m *Model) cmdMemory() tea.Cmd {
	m.memBusy = true
	m.memAt = m.now()
	b := m.b
	return func() tea.Msg { r, err := b.Memory(); return memMsg{r, err} }
}

func (m *Model) cmdLog() tea.Cmd {
	m.logBusy = true
	b := m.b
	return func() tea.Msg { ls, err := b.LogTail(500); return logMsg{ls, err} }
}

func (m *Model) cmdLauncher() tea.Cmd {
	b := m.b
	return func() tea.Msg { l, err := b.Launcher(); return launcherMsg{l, err} }
}

func (m *Model) cmdStop(mode StopMode) tea.Cmd {
	m.ph = phStopping
	m.stopMode = mode
	m.stopEv = nil
	m.stopAt = m.now()
	b := m.b
	return func() tea.Msg { ch, err := b.Stop(mode); return stopOpenMsg{ch, err} }
}

func waitStop(ch <-chan StopEvent) tea.Cmd {
	return func() tea.Msg { ev, ok := <-ch; return stopEvMsg{ev, ok, ch} }
}

func (m *Model) cmdSave(model string, s Settings) tea.Cmd {
	b := m.b
	s = copySet(s)
	return func() tea.Msg { v, err := b.SavePreset(model, s); return savedMsg{model, v, err} }
}

func (m *Model) cmdPlan(model string, s Settings) tea.Cmd {
	b := m.b
	return func() tea.Msg { p, err := b.RepackPlan(model, s); return planMsg{model, s, p, err} }
}

func (m *Model) cmdRepack(model string, s Settings) tea.Cmd {
	b := m.b
	return func() tea.Msg { ch, err := b.Repack(model, s); return startOpenMsg{ch, err} }
}

// firstDrawWait is how long after Init the screen asks for the launcher file check, whose first
// call starts it (conflict 18-2: after the first draw; GL-13). Bubble Tea hands the first view
// to its renderer before any Init command runs and writes it at its next frame (60 a second),
// so six frames after Init the first screen is on the console, with room for a busy machine.
const firstDrawWait = 100 * time.Millisecond

func (m *Model) cmdBundle() tea.Cmd {
	m.bundleBusy = true
	b := m.b
	return func() tea.Msg { c, err := b.Bundle(); return bundleMsg{c, err} }
}

func (m *Model) cmdCheck(model, key, value string) tea.Cmd {
	b := m.b
	return func() tea.Msg { v, err := b.CheckSetting(model, key, value); return checkedMsg{model, key, v, err} }
}

func (m *Model) cmdAction(id ActionID, arg string) tea.Cmd {
	b := m.b
	return func() tea.Msg { res, err := b.Action(id, arg); return actionMsg{id, res, err} }
}

// cmdForce stops the server without a save and quits; it waits at most 3 s
// for the backend, then quits anyway.
func (m *Model) cmdForce() tea.Cmd {
	b := m.b
	return func() tea.Msg {
		done := make(chan struct{})
		go func() {
			defer close(done)
			ch, err := b.Stop(StopForce)
			if err != nil || ch == nil {
				return
			}
			for range ch {
			}
		}()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
		}
		return tea.QuitMsg{}
	}
}

func (m *Model) beat() tea.Cmd {
	return m.tick(pollEvery, func(t time.Time) tea.Msg { return tickMsg{t} })
}

// ---------- tea.Model ----------

// Init loads the catalog, the launcher settings and the memory reading of
// 1 Models, and starts the heartbeat.
func (m *Model) Init() tea.Cmd {
	return tea.Batch(m.cmdCatalog(), m.cmdLauncher(), m.cmdMemory(), m.beat(),
		m.tick(firstDrawWait, func(time.Time) tea.Msg { return drawnMsg{} }))
}

// Update handles one message. It never calls the Backend.
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.r.W, m.r.H = msg.Width, msg.Height
		return m, nil
	case tea.KeyMsg:
		return m, m.key(msg.String())
	case tea.InterruptMsg:
		// ctrl+break (LAUNCH-FEAS Q1.6): same as ctrl+c twice
		return m, m.forceQuit()
	case tickMsg:
		return m, m.onBeat()
	case disarmMsg:
		if msg.kind == armedQuit && msg.seq == m.quitSeq {
			m.quitArmed = false
		}
		if msg.kind == armedForce && msg.seq == m.frcSeq {
			m.frcArmed = false
		}
		return m, nil
	case catalogMsg:
		m.catBusy = false
		if msg.err != nil {
			m.catErr = msg.err.Error()
			return m, nil
		}
		// the selection stays on the same model while it is listed (review item 31)
		old, had := m.selRow()
		m.cat, m.catOK, m.catErr = msg.c, true, ""
		m.sel = min(m.sel, max(0, len(m.cat.Models)-1))
		for i, x := range m.cat.Models {
			if had && x.ID == old.ID {
				m.sel = i
			}
		}
		if m.serverUp() {
			m.lockOnRun()
		}
		cmds := []tea.Cmd{m.needSettings()}
		now, ok := m.selRow()
		if !m.preBusy && (m.pre == nil || (ok && rowKey(now) != rowKey(old))) {
			cmds = append(cmds, m.checkSelected())
		}
		return m, tea.Batch(cmds...)
	case settingsMsg:
		if msg.err == nil {
			m.views[msg.model] = msg.v
		}
		return m, nil
	case launcherMsg:
		if msg.err == nil {
			l := msg.l
			m.launcher = &l
		}
		return m, nil
	case preflightMsg:
		return m, m.onPreflight(msg)
	case startOpenMsg:
		if msg.err != nil {
			m.fail(&Failure{Stage: "start", What: "the server did not start", Why: msg.err.Error(), At: m.now()})
			return m, nil
		}
		return m, waitStart(msg.ch)
	case startEvMsg:
		return m, m.onStartEv(msg)
	case statusMsg:
		m.statusBusy = false
		return m, m.onStatus(msg)
	case memMsg:
		m.memBusy = false
		if msg.err == nil {
			r := msg.m
			m.mem = &r
		}
		return m, nil
	case logMsg:
		m.logBusy = false
		if msg.err == nil {
			m.onLogs(msg.ls)
		}
		return m, nil
	case checkedMsg:
		return m, m.onChecked(msg)
	case planMsg:
		if msg.err != nil {
			m.notice = m.r.T("the setup plan could not be read") + ": " + msg.err.Error()
			return m, nil
		}
		p := msg.p
		m.plan, m.planSet, m.planID = &p, msg.set, msg.model
		m.place = plServer
		return m, nil
	case drawnMsg:
		m.drawn = true
		return m, m.cmdBundle()
	case bundleMsg:
		m.bundleBusy = false
		if msg.err != nil {
			m.bundle = BundleCheck{State: BundleNotPorted}
		} else {
			m.bundle = msg.b
		}
		// a start that waited for the check goes on now (conflict 18-2)
		if m.bundle.State != BundleChecking && m.startWait != "" {
			id := m.startWait
			m.startWait = ""
			if row, ok := m.rowByID(id); ok && m.bundle.State != BundleBroken {
				return m, m.startFromStopped(row)
			}
		}
		return m, nil
	case savedMsg:
		if msg.err != nil {
			m.notice = m.r.T("save") + ": " + msg.err.Error()
			return m, nil
		}
		m.views[msg.model] = msg.v
		delete(m.edits, msg.model)
		m.notice = m.r.T("saved · these values are used from the next start")
		return m, m.cmdPreflight(msg.model, Settings{}, false)
	case stopOpenMsg:
		if msg.err != nil {
			m.notice = m.r.T("stop failed") + ": " + msg.err.Error()
			m.ph = phServing
			if m.exitAfter {
				return m, tea.Quit
			}
			return m, nil
		}
		return m, waitStop(msg.ch)
	case stopEvMsg:
		return m, m.onStopEv(msg)
	case actionMsg:
		if msg.id == ActSetCacheCap && m.inputLauncher { // the typed cap: a refusal keeps the box (11.4)
			if msg.err != nil {
				m.inputErr = msg.err.Error()
				return m, nil
			}
			m.inputOn, m.inputLauncher, m.inputErr = false, false, ""
		}
		switch {
		case msg.err != nil:
			m.notice = m.actionName(msg.id) + ": " + msg.err.Error()
		case msg.id == ActTaskManager:
			m.notice = m.taskLine
		case msg.res.Message != "":
			m.notice = msg.res.Message
		}
		if (msg.id == ActSetSmoke || msg.id == ActSetCacheCap || msg.id == ActSetWarmStart || msg.id == ActSetAutosave) && msg.err == nil {
			return m, m.cmdLauncher() // the row shows the value the backend keeps (conflict 18-1)
		}
		if msg.id == ActDeletePartial && msg.err == nil {
			return m, m.cmdCatalog() // H4 (GL-10): the half-made files are gone; the list reads again
		}
		return m, nil
	}
	return m, nil
}

func (m *Model) needSettings() tea.Cmd {
	if row, ok := m.selRow(); ok {
		if _, have := m.views[row.ID]; !have {
			return m.cmdSettings(row.ID)
		}
	}
	return nil
}

func (m *Model) onBeat() tea.Cmd {
	m.beats++
	now := m.now()
	if m.quitArmed && now.Sub(m.quitAt) > armWindow {
		m.quitArmed = false
	}
	if m.frcArmed && now.Sub(m.frcAt) > armWindow {
		m.frcArmed = false
	}
	cmds := []tea.Cmd{m.beat()}
	if m.ph == phServing && !m.statusBusy {
		cmds = append(cmds, m.cmdStatus())
	}
	if (m.place == plMemory || m.place == plModels) && !m.memBusy && now.Sub(m.memAt) >= memEvery {
		cmds = append(cmds, m.cmdMemory())
	}
	if (m.place == plMemory || m.place == plModels) && !m.preBusy && now.Sub(m.preAt) >= preEvery {
		cmds = append(cmds, m.checkSelected())
	}
	if m.place == plModels && !m.catBusy && now.Sub(m.catAt) >= catEvery {
		cmds = append(cmds, m.cmdCatalog())
	}
	if m.drawn && m.bundle.State == BundleChecking && !m.bundleBusy {
		cmds = append(cmds, m.cmdBundle())
	}
	if m.place == plLog && !m.logBusy {
		cmds = append(cmds, m.cmdLog())
	}
	return tea.Batch(cmds...)
}

// onLogs takes a new log tail. While the view is held (scrolled up), the
// lines that arrived push the hold up by their screen lines, so the view
// stays where it is (review item 6), and are counted as new.
func (m *Model) onLogs(ls []LogLine) {
	old := m.logs
	m.logs = ls
	if m.logBack == 0 {
		m.logNew = 0
		return
	}
	added := ls
	if len(old) > 0 {
		last := old[len(old)-1]
		found := -1
		for i := len(ls) - 1; i >= 0; i-- {
			if ls[i] == last {
				found = i
				break
			}
		}
		if found < 0 {
			// the tail no longer holds what was shown: go back to the newest
			m.logBack, m.logNew = 0, 0
			return
		}
		added = ls[found+1:]
	}
	var shown []LogLine
	for _, x := range added {
		if m.logShows(x) {
			shown = append(shown, x)
		}
	}
	m.logBack += len(m.logScreenLines(shown, m.logTextW()))
	m.logNew += len(shown)
	m.logBack = min(m.logBack, m.logScrollMax())
}

func (m *Model) onPreflight(msg preflightMsg) tea.Cmd {
	m.preBusy = false
	if msg.err != nil {
		m.notice = m.r.T("check failed") + ": " + msg.err.Error()
		return nil
	}
	p := msg.p
	m.pre, m.preKey = &p, msg.key
	if !msg.thenStart {
		return nil
	}
	if p.Refused {
		// conflict 7: a refused start is never offered; show why on 3 Memory
		m.place = plMemory
		return nil
	}
	m.beginRun(msg.model, msg.set)
	return m.cmdStart(msg.model, msg.set)
}

// beginRun turns to 2 Server for a start (or a setup that ends in one),
// keeping where its settings came from (conflict 3).
func (m *Model) beginRun(model string, set Settings) {
	m.runModel, m.runSet = model, set
	m.runLabel, m.runEdits = m.provenance(model), m.changedValues(model)
	m.lockOnRun()
	m.ph = phStarting
	m.startEv, m.failure, m.result, m.status = nil, nil, nil, nil
	m.setup = nil
	m.startAt = m.now()
	m.samples, m.reqSamples = nil, nil
	m.plan = nil
	m.place = plServer
}

func (m *Model) onStartEv(msg startEvMsg) tea.Cmd {
	if m.ph != phStarting {
		return nil // cancelled or stopped meanwhile: drop late events
	}
	if !msg.ok {
		m.fail(&Failure{Stage: "start", What: "the server did not start", Why: "the start ended without /health ready", At: m.now()})
		return nil
	}
	ev := msg.ev
	m.startEv, m.startAt = &ev, m.now()
	m.noteSetup(ev)
	switch {
	case ev.Failure != nil:
		m.fail(ev.Failure)
		return nil
	case ev.Ready:
		m.ph = phServing
		return m.cmdStatus()
	}
	return waitStart(msg.ch)
}

func (m *Model) fail(f *Failure) {
	m.ph = phFailed
	m.failure = f
	m.place = plServer
}

func (m *Model) onStatus(msg statusMsg) tea.Cmd {
	if msg.err != nil || m.ph != phServing {
		return nil
	}
	s := msg.s
	m.status = &s
	if s.DecodeTokS.Ok() && s.Phase == ReqDecode {
		m.samples = append(m.samples, sample{m.now(), s.DecodeTokS.V})
		if len(m.samples) > 4*3600 {
			m.samples = m.samples[len(m.samples)-4*3600:]
		}
	}
	m.addReqSamples(s.Reqs)
	if s.Exited {
		m.fail(&Failure{Stage: "serving", What: "the server ended while serving",
			ExitCode: s.ExitCode, At: m.now()})
	}
	return nil
}

func (m *Model) onStopEv(msg stopEvMsg) tea.Cmd {
	if msg.ok {
		ev := msg.ev
		m.stopEv, m.stopAt = &ev, m.now()
		// the quit of a serving server begins with the cache save and its
		// bar (decision ③, edition 3.5)
		if m.stopMode == StopQuit && m.quitServing && len(m.quitLog) == 0 {
			m.quitLog = append(m.quitLog, QuitStepCacheSave)
		}
		if ev.Result == nil && !ev.Done {
			return waitStop(msg.ch)
		}
	}
	// done (a result, a done event, or the channel closed)
	if m.stopEv != nil && m.stopEv.Result != nil {
		m.result = m.stopEv.Result
	}
	if m.result == nil {
		m.result = &StopResult{Run: RunSummary{Model: m.runModel}}
	}
	if m.stopMode == StopCancel {
		m.ph = phNone
		m.notice = m.r.T("start cancelled · nothing is lost")
	} else {
		m.ph = phStopped
	}
	m.status = nil
	m.pre, m.preKey = nil, "" // the memory changed: the next check measures again (review item 22)
	if m.stopMode == StopQuit && m.quitServing {
		if len(m.quitLog) == 0 {
			m.quitLog = append(m.quitLog, QuitStepCacheSave)
		}
		m.quitLog = append(m.quitLog, QuitStepUnload)
	}
	if m.exitAfter {
		if m.stopMode == StopQuit {
			m.quitLog = append(m.quitLog, QuitStepExit)
		}
		m.exiting = true
		return tea.Quit
	}
	return nil
}

// lockOnRun points the model list at the model that runs: the list stays on
// it until the server is unloaded (review item 1).
func (m *Model) lockOnRun() {
	for i, x := range m.cat.Models {
		if x.ID == m.runModel {
			m.sel = i
		}
	}
}

func (m *Model) actionName(id ActionID) string { return string(id) }

// ---------- selection helpers ----------

// rowKey is what of a list row a new check depends on: a change here (the
// file gone, a repack output deleted, another file) runs the preflight again.
func rowKey(x ModelRow) string {
	return fmt.Sprint(x.ID, x.Present, x.Setup, x.Path, x.SizeGB, x.FileCheck)
}

func (m *Model) selRow() (ModelRow, bool) {
	if m.sel >= 0 && m.sel < len(m.cat.Models) {
		return m.cat.Models[m.sel], true
	}
	return ModelRow{}, false
}

func (m *Model) editsFor(model string) Settings {
	if s, ok := m.edits[model]; ok {
		return s
	}
	return Settings{}
}

// visibleRows are the settings rows of the box, each on its own line (UX-R10, finding 17: no
// fold): the rows the box led with first, then the rest (Folded only orders them now).
func (m *Model) visibleRows(v SettingsView) []SettingRow {
	var out, rest []SettingRow
	for _, r := range v.Rows {
		if r.Folded {
			rest = append(rest, r)
		} else {
			out = append(out, r)
		}
	}
	return append(out, rest...)
}

// appendNew appends s when it is not empty and not in xs.
func appendNew(xs []string, s string) []string {
	for _, x := range xs {
		if x == s {
			return xs
		}
	}
	if s == "" {
		return xs
	}
	return append(xs, s)
}

// value of a setting with the screen's edit applied.
func (m *Model) valueOf(model string, r SettingRow) string {
	if v, ok := m.edits[model][r.Key]; ok {
		return v
	}
	return r.Value
}

func (m *Model) serverUp() bool {
	return m.ph == phStarting || m.ph == phServing || m.ph == phStopping
}

func (m *Model) startFlow(model string) tea.Cmd {
	return m.cmdPreflight(model, m.editsFor(model), true)
}

func (m *Model) currentPreflight(model string) *Preflight {
	if m.pre != nil && m.preKey == setKey(model, m.editsFor(model)) {
		return m.pre
	}
	return nil
}

// reqSampleKeep bounds the per-request store of one run (memory only).
const reqSampleKeep = 4096

// addReqSamples keeps every finished request the status reports once, in order (GL-18): the
// server's timing lines hold the last ones only, so the store grows across polls.
func (m *Model) addReqSamples(rs []ReqSample) {
	for _, r := range rs {
		if n := len(m.reqSamples); n > 0 {
			last := m.reqSamples[n-1]
			if r.At.Before(last.At) || (r.At.Equal(last.At) && r.Task == last.Task) {
				continue // already kept
			}
		}
		m.reqSamples = append(m.reqSamples, r)
	}
	if len(m.reqSamples) > reqSampleKeep {
		m.reqSamples = m.reqSamples[len(m.reqSamples)-reqSampleKeep:]
	}
}
