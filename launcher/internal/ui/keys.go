// Keys (designs/launcher_design/README.md, mock keys.go): five places, ? help,
// ←→ between places, 1-5 to jump. Every key that stops, cancels, deletes or
// closes takes ctrl or a second press: ctrl+f unloads, ctrl+x cancels a start
// or a setup, q twice quits (the H8Q prompt while serving), ctrl+c twice
// closes now. Decision ① (DECISION_LEDGER.md:4177): there is no ctrl+x
// "cut this reply", and the q-twice flow has no cut step. GL-02: the table
// actions are reached through the action-to-key table (actions.go), with the
// confirmed default layout (L:4265 ①); while a reply runs the first q warns
// in the bottom line and the second quits at once, nothing saved (U-27).

package ui

import (
	"strings"
	"time"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
)

// isQ: the quit key. ㅂ is the q key in Hangul mode of the Korean IME
// (LAUNCH-FEAS Q1.4: whether q or ㅂ arrives is a hands-on test).
func isQ(k string) bool { return k == "q" || k == "ㅂ" }

// the launcher rows: address, language, check run, on open, logs kept, and
// since GL-16 the prompt-cache cap (U-22) and the warm-start switches (OQ-6
// (a)); the network row is gone, loopback stays fixed (conflict 17); the keys
// row and its key page H5K are the next version (L:4265 ②, GL-02)
const set5LauncherRows = 8
const set5LangRow = 1
const set5SmokeRow = 2
const set5CapRow = 5
const set5WarmRow = 6
const set5AutosaveRow = 7

// capChoices are the values ←→ cycles the prompt-cache cap through (off, the
// default of DECISION_LEDGER.md:4420, and the powers of two of the cap rule,
// F-BU E.3); Space types any value.
var capChoices = []string{"off", "256", "512", "1024", "2048", "4096", "8192"}

// 5 Settings sections: Tab goes round them; ↑↓ also cross from the folders
// into the launcher rows and back (review item 8); the writes section is
// entered by Tab only (review item 24).
const (
	secFolders = iota
	secLauncher
	secWrites
	secCount
)

func (m *Model) key(pressed string) tea.Cmd {
	now := m.now()
	m.notice = ""
	// the action-to-key table: a table action is reached only through the
	// keys the table gives it (GL-02; S8-LA-01)
	k := m.keys.canon(pressed)

	// ctrl+c twice: close now, no cache save
	if m.frcArmed && now.Sub(m.frcAt) > armWindow {
		m.frcArmed = false
	}
	if k == "ctrl+c" {
		if m.frcArmed {
			m.frcArmed = false
			return m.forceQuit()
		}
		m.frcArmed, m.frcAt = true, now
		m.quitArmed = false
		m.frcSeq++
		seq := m.frcSeq
		return m.tick(armWindow, func(time.Time) tea.Msg { return disarmMsg{armedForce, seq} })
	}
	if m.frcArmed { // any other key cancels
		m.frcArmed = false
		return nil
	}

	// the typed-value box takes every key while it is open, digits and q
	// included (review item 30)
	if m.inputOn {
		return m.keyInput(pressed)
	}

	// q twice: unload, then close (H8Q while serving)
	if m.quitArmed && now.Sub(m.quitAt) > armWindow {
		m.quitArmed = false
	}
	if m.quitArmed {
		m.quitArmed = false
		if isQ(k) {
			return m.quit()
		}
		return nil // other keys cancel
	}
	if isQ(k) {
		// U-27: a reply running at the first q is warned of in the bottom
		// line, and nothing else changes
		m.quitArmed, m.quitAt, m.quitReply = true, now, m.replyRunning()
		m.quitSeq++
		seq := m.quitSeq
		return m.tick(armWindow, func(time.Time) tea.Msg { return disarmMsg{armedQuit, seq} })
	}
	if m.exiting {
		return nil
	}

	// help
	if m.help {
		switch k {
		case "?", "esc":
			m.help, m.helpTiles = false, false
		case "tab":
			m.helpTiles = !m.helpTiles
		case "1", "2", "3", "4", "5":
			m.help, m.helpTiles = false, false
			return m.goPlace(int(k[0] - '1'))
		}
		return nil
	}
	if k == "?" {
		m.help, m.helpTiles, m.helpFrom = true, false, m.place
		return nil
	}

	// places
	switch k {
	case "1", "2", "3", "4", "5":
		return m.goPlace(int(k[0] - '1'))
	case "left", "right":
		if !m.valueMode() {
			d := 1
			if k == "left" {
				d = 4
			}
			return m.goPlace((m.place + d) % 5)
		}
	case "ctrl+f":
		if m.ph == phServing {
			m.place = plServer
			return m.cmdStop(StopUnload)
		}
		m.notice = m.r.T("no server to unload")
		return nil
	case "ctrl+x":
		if m.ph == phStarting {
			m.place = plServer
			return m.cmdStop(StopCancel)
		}
		return nil
	}

	switch m.place {
	case plModels:
		return m.keyModels(k)
	case plServer:
		return m.keyServer(k)
	case plMemory:
		return m.keyMemory(k)
	case plLog:
		return m.keyLog(k)
	case plSettings:
		return m.keySettings(k)
	}
	return nil
}

// valueMode: ←→ changes a value on this page instead of turning pages.
func (m *Model) valueMode() bool {
	switch m.place {
	case plModels:
		return m.focusSet
	case plSettings:
		return m.set5Row >= 0 && m.set5Sec != secWrites
	}
	return false
}

func (m *Model) goPlace(p int) tea.Cmd {
	m.place = p
	m.set5Row = -1 // arriving, no row is chosen yet
	var cmds []tea.Cmd
	switch p {
	case plMemory, plModels:
		// the list read again on arriving at 1 Models (review item 31)
		if p == plModels && !m.catBusy {
			cmds = append(cmds, m.cmdCatalog())
		}
		// measured again on arrival (review items 13, 22)
		if !m.memBusy {
			cmds = append(cmds, m.cmdMemory())
		}
		if !m.preBusy {
			cmds = append(cmds, m.checkSelected())
		}
	case plLog:
		if !m.logBusy {
			cmds = append(cmds, m.cmdLog())
		}
	case plSettings:
		cmds = append(cmds, m.cmdLauncher())
	}
	return tea.Batch(cmds...)
}

// replyRunning: the serving server is answering a request (prefill or
// decode) by its last status reading.
func (m *Model) replyRunning() bool {
	return m.ph == phServing && m.status != nil && m.status.Phase != ReqNone
}

// quit is the second quit action (edition 3.5). With a reply running at
// either press it ends at once, nothing saved (U-27, L:4275); otherwise a
// serving server saves the cache, unloads, and the launcher exits.
func (m *Model) quit() tea.Cmd {
	if m.quitReply || m.replyRunning() {
		m.quitReply = false
		if m.serverUp() {
			m.quitLog = append(m.quitLog, QuitStepExit)
			return m.forceQuit()
		}
	}
	switch m.ph {
	case phStarting, phServing:
		m.quitServing = m.ph == phServing
		m.exitAfter = true
		m.place = plServer
		return m.cmdStop(StopQuit)
	case phStopping:
		m.exitAfter = true
		return nil
	}
	m.quitLog = append(m.quitLog, QuitStepExit)
	m.exiting = true
	return tea.Quit
}

// QuitSteps is the quit sequence this screen ran, in order (S8-LA-03).
func (m *Model) QuitSteps() []QuitStep { return append([]QuitStep{}, m.quitLog...) }

func (m *Model) forceQuit() tea.Cmd {
	m.exiting = true
	if m.serverUp() {
		return m.cmdForce()
	}
	return tea.Quit
}

// ---------- 1 Models ----------

func (m *Model) noModels() bool {
	if !m.catOK {
		return false
	}
	for _, r := range m.cat.Models {
		if r.Present {
			return false
		}
	}
	return true
}

func (m *Model) cutOff() (ModelRow, bool) {
	row, ok := m.selRow()
	return row, ok && row.Setup == SetupCutOff && !m.h4Skip[row.ID] && !m.serverUp()
}

func (m *Model) keyModels(k string) tea.Cmd {
	if m.noModels() || (!m.catOK && m.catErr != "") {
		if k == "r" {
			return m.cmdCatalog()
		}
		return nil
	}
	if row, ok := m.cutOff(); ok {
		switch k {
		case "ctrl+r":
			// GL-10 (S8-LA-34): the redo shows the setup plan; its ⏎ removes the half-made
			// files and runs the setup, its esc writes nothing (conflict 9).
			m.h4Skip[row.ID] = true
			return m.cmdPlan(row.ID, m.editsFor(row.ID))
		case "ctrl+d":
			return m.cmdAction(ActDeletePartial, row.ID)
		case "esc":
			m.h4Skip[row.ID] = true
		}
		return nil
	}
	row, ok := m.selRow()
	if !ok {
		return nil
	}
	if !m.focusSet {
		switch k {
		case "up", "down":
			if m.serverUp() {
				// the list is locked while a server runs (review item 1)
				m.notice = m.lockWords() // UX-KR: the one lock wording
				return nil
			}
			if k == "up" && m.sel > 0 {
				m.sel--
			}
			if k == "down" && m.sel < len(m.cat.Models)-1 {
				m.sel++
			}
			return tea.Batch(m.needSettings(), m.checkSelected())
		case "tab":
			m.focusSet, m.setCur = true, 0
		case "i":
			m.showAbout = !m.showAbout
		case "enter":
			return m.startFromStopped(row)
		}
		return nil
	}
	v := m.views[row.ID]
	rows := m.visibleRows(v)
	switch k {
	case "tab":
		m.focusSet = false
	case "up":
		if m.setCur > 0 {
			m.setCur--
		}
	case "down":
		if m.setCur < len(rows)-1 {
			m.setCur++
		}
	case " ":
		// Space opens the box on a row that takes a typed value (review item 30)
		if m.setCur < len(rows) && rows[m.setCur].Typed && rows[m.setCur].Locked == "" {
			sr := rows[m.setCur]
			m.inputOn, m.inputKey, m.inputErr = true, sr.Key, ""
			m.inputBuf = strings.TrimSpace(strings.TrimSuffix(m.valueOf(row.ID, sr), sr.Unit))
		}
	case "left", "right":
		if m.setCur >= len(rows) {
			return nil
		}
		sr := rows[m.setCur]
		if !sr.Editable || len(sr.Options) == 0 {
			if sr.Locked != "" {
				m.notice = sr.Label + ": " + m.r.T("locked") + " · " + sr.Locked
			} else {
				m.notice = sr.Label + ": " + m.r.T("not changeable here")
			}
			return nil
		}
		cur := m.valueOf(row.ID, sr)
		i := 0
		for j, o := range sr.Options {
			if o == cur {
				i = j
			}
		}
		if k == "right" {
			i = (i + 1) % len(sr.Options)
		} else {
			i = (i + len(sr.Options) - 1) % len(sr.Options)
		}
		e := copySet(m.editsFor(row.ID))
		if sr.Options[i] == sr.Value {
			delete(e, sr.Key)
		} else {
			e[sr.Key] = sr.Options[i]
		}
		m.edits[row.ID] = e
		return m.cmdPreflight(row.ID, e, false)
	case "esc":
		delete(m.edits, row.ID)
		m.focusSet = false
	case "ctrl+s":
		// the model's changed settings are saved through the backend (review item 3)
		if len(m.edits[row.ID]) == 0 {
			m.notice = m.r.T("nothing changed to save")
			return nil
		}
		return m.cmdSave(row.ID, m.edits[row.ID])
	case "enter":
		return m.startFromStopped(row)
	}
	return nil
}

// keyInput edits the box: ⏎ checks the value through the backend, esc
// leaves the value as it was, backspace removes a character, any other
// single character is typed.
func (m *Model) keyInput(k string) tea.Cmd {
	row, ok := m.selRow()
	switch k {
	case "esc":
		m.inputOn, m.inputLauncher, m.inputErr = false, false, ""
	case "enter":
		if m.inputLauncher { // 5 Settings: the prompt-cache cap
			return m.cmdAction(ActSetCacheCap, strings.TrimSpace(m.inputBuf))
		}
		if ok {
			return m.cmdCheck(row.ID, m.inputKey, strings.TrimSpace(m.inputBuf))
		}
	case "backspace":
		if rs := []rune(m.inputBuf); len(rs) > 0 {
			m.inputBuf = string(rs[:len(rs)-1])
		}
		m.inputErr = ""
	default:
		if utf8.RuneCountInString(k) == 1 {
			m.inputBuf += k
			m.inputErr = ""
		}
	}
	return nil
}

// onChecked applies a checked value, or shows why it was refused and keeps
// the value as it was (review item 30).
func (m *Model) onChecked(msg checkedMsg) tea.Cmd {
	if !m.inputOn || msg.key != m.inputKey {
		return nil
	}
	if msg.err != nil {
		m.inputErr = msg.err.Error()
		return nil
	}
	m.inputOn, m.inputErr = false, ""
	e := copySet(m.editsFor(msg.model))
	for _, x := range m.views[msg.model].Rows {
		if x.Key == msg.key {
			if msg.value == x.Value {
				delete(e, x.Key)
			} else {
				e[x.Key] = msg.value
			}
		}
	}
	m.edits[msg.model] = e
	return m.cmdPreflight(msg.model, e, false)
}

// checkSelected runs the preflight of the chosen model with the screen's
// settings, when no server is up and the file is present.
func (m *Model) checkSelected() tea.Cmd {
	row, ok := m.selRow()
	if !ok || !row.Present || m.serverUp() {
		return nil
	}
	return m.cmdPreflight(row.ID, m.editsFor(row.ID), false)
}

// startFromStopped: ⏎ starts only while no server is up. While one serves,
// ⏎ never restarts it (review item 1): ctrl+f unloads first.
func (m *Model) startFromStopped(row ModelRow) tea.Cmd {
	if !row.Present {
		m.notice = row.Name + ": " + m.r.T("model file missing")
		return nil
	}
	if m.serverUp() {
		m.notice = m.r.T("a server is running · ctrl+f unloads it, then ⏎ starts")
		return nil
	}
	switch m.bundle.State { // conflict 18-2: a start waits for the file check
	case BundleBroken:
		m.notice = m.bundleBrokenText()
		return nil
	case BundleChecking:
		m.startWait = row.ID
		return nil
	}
	if m.needsSetup(row) {
		// conflict 9: the plan box first, on 2 Server; nothing is written yet
		return m.cmdPlan(row.ID, m.editsFor(row.ID))
	}
	return m.startFlow(row.ID)
}

// needsSetup: the model, or the repack mode now set, has no usable output
// (review item 31, conflicts 9 and 16), so ⏎ shows the plan box. The mode the
// screen holds decides when the backend gives its check (FIX2-WIRE rehearsal:
// the SETUP tile follows the mode the model starts with, so a model whose
// packed output is gone but whose virtual output passes its check starts in
// virtual without a plan box); the tile decides only without a mode check.
func (m *Model) needsSetup(row ModelRow) bool {
	cur := m.curMode(row)
	for _, x := range row.Modes {
		if x.Mode == cur {
			return x.State == ModeNeedsRepack || x.State == ModeCheckFailed || x.State == ModeStale
		}
	}
	return row.Setup == SetupNeeded || row.Setup == SetupNeedsRepack
}

// curMode is the repack mode the screen holds for a model ("" unknown).
func (m *Model) curMode(row ModelRow) string {
	for _, x := range m.views[row.ID].Rows {
		if x.Key == "mode" {
			return m.valueOf(row.ID, x)
		}
	}
	return ""
}

// keyPlan: ⏎ runs the planned setup, then the start; esc writes nothing and
// goes back to 1 Models (conflict 9). A disk too small offers no ⏎.
func (m *Model) keyPlan(k string) (tea.Cmd, bool) {
	p := m.plan
	switch k {
	case "enter":
		if planShort(*p) {
			return nil, true
		}
		// UX-R8: the ID the plan was asked for (a path for the real backend); the plan's own
		// Model is the backend's label for it (the display name there), which Repack does not know
		m.beginRun(m.planID, m.planSet)
		return m.cmdRepack(m.runModel, m.runSet), true
	case "esc":
		m.plan = nil
		return m.goPlace(plModels), true
	}
	return nil, false
}

func (m *Model) launcherSmoke() SmokeMode {
	if m.launcher == nil {
		return SmokeNotPorted
	}
	return m.launcher.Smoke
}

// ---------- 2 Server ----------

func (m *Model) keyServer(k string) tea.Cmd {
	if m.planShown() {
		if c, done := m.keyPlan(k); done {
			return c
		}
	}
	switch m.ph {
	case phServing:
		switch k {
		case "c":
			addr := ""
			if m.status != nil {
				addr = m.status.Address
			}
			return m.cmdAction(ActCopyAddress, addr)
		}
	case phFailed:
		switch k {
		case "r":
			if m.runModel != "" {
				return m.cmdPreflight(m.runModel, m.runSet, true)
			}
		case "y":
			return m.cmdAction(ActCopyReport, m.runModel)
		case "esc":
			return m.goPlace(plModels)
		default:
			if m.failure != nil {
				for _, f := range m.failure.Fixes {
					if f.Key == k {
						if f.Action == "" {
							return m.cmdPreflight(m.runModel, m.runSet, true)
						}
						return m.cmdAction(f.Action, m.runModel)
					}
				}
			}
		}
	case phStopped:
		switch k {
		case "enter":
			if m.runModel != "" {
				return m.cmdPreflight(m.runModel, m.runSet, true)
			}
		case "m":
			return m.goPlace(plModels)
		}
	}
	return nil
}

// ---------- 3 Memory ----------

func (m *Model) keyMemory(k string) tea.Cmd {
	switch k {
	case "o": // Task Manager opens without a selection; the line names the app (U-35 (i))
		m.taskLine = m.taskManagerLine()
		return m.cmdAction(ActTaskManager, "")
	case "up":
		if m.appCur > 0 {
			m.appCur--
		}
	case "down":
		if m.mem != nil && m.appCur < len(m.mem.Apps)-1 {
			m.appCur++
		}
	case "enter":
		if m.ph == phNone || m.ph == phStopped || m.ph == phFailed {
			row, ok := m.selRow()
			if !ok {
				return nil
			}
			p := m.currentPreflight(row.ID)
			if p == nil || p.Refused {
				return nil // conflict 7: no start offered after a refusal
			}
			return m.startFlow(row.ID)
		}
	}
	return nil
}

// ---------- 4 Log ----------

func (m *Model) keyLog(k string) tea.Cmd {
	switch k {
	case "up":
		// the hold never passes the oldest line (review item 20)
		m.logBack = min(m.logBack+1, m.logScrollMax())
	case "down":
		if m.logBack > 0 {
			m.logBack--
		}
		if m.logBack == 0 {
			m.logNew = 0
		}
	case "end":
		m.logBack, m.logNew = 0, 0
	case "tab":
		// tabs all · warnings · errors (review item 18); ←→ still turn places
		m.logFilter = (m.logFilter + 1) % 3
		m.logBack, m.logNew = 0, 0
	case "e":
		f := ""
		if m.launcher != nil {
			f = m.launcher.LogFile
		}
		return m.cmdAction(ActOpenLog, f)
	}
	return nil
}

// ---------- 5 Settings ----------

func (m *Model) keySettings(k string) tea.Cmd {
	switch k {
	case "tab":
		m.set5Sec = (m.set5Sec + 1) % secCount
		m.set5Row = -1
	case "down": // UX-R4: a hidden row (not ported) is stepped over
		switch i := m.set5Next(m.set5Row, 1); {
		case i >= 0:
			m.set5Row = i
		case m.set5Sec == secFolders:
			// from the last folder on into the launcher rows (review item 8)
			m.set5Sec = secLauncher
			m.set5Row = m.set5Next(-1, 1)
		}
	case "up":
		switch i := m.set5Next(m.set5Row, -1); {
		case m.set5Row > 0 && i >= 0:
			m.set5Row = i
		case m.set5Sec == secLauncher && m.set5Row >= 0:
			m.set5Sec = secFolders // first, so that set5Rows counts the folders
			m.set5Row = m.set5Rows() - 1
		}
	case "esc":
		m.set5Row = -1
	case "a":
		return m.cmdAction(ActAddFolder, "")
	case " ":
		// Space types the prompt-cache cap (a value row, review item 30's box)
		if m.set5Sec == secLauncher && m.set5Row == set5CapRow {
			m.inputOn, m.inputLauncher, m.inputKey, m.inputErr = true, true, "prompt_cache_cap", ""
			m.inputBuf = m.launcherInfo().CacheCap
		}
	case "x":
		if m.set5Sec == secFolders && m.set5Row >= 0 && m.set5Row < len(m.cat.Folders) {
			return m.cmdAction(ActRemoveFolder, m.cat.Folders[m.set5Row].Path)
		}
	case "enter":
		if m.set5Sec == secWrites && m.set5Row >= 0 {
			// a folder opens in Explorer, a file opens selected; a row
			// without a path does nothing (review item 24)
			ws := m.writeRows()
			if m.set5Row < len(ws) && ws[m.set5Row].path != "" {
				if ws[m.set5Row].file {
					return m.cmdAction(ActShowFile, ws[m.set5Row].path)
				}
				return m.cmdAction(ActOpenFolder, ws[m.set5Row].path)
			}
		}
	case "left", "right":
		if m.set5Sec == secFolders {
			return nil // a folder row has no value to change (review item 8)
		}
		if m.set5Sec == secLauncher && m.set5Row == set5SmokeRow { // check run: off / on / once (conflict 18-1)
			order := []string{"off", "on", "once"}
			cur := map[SmokeMode]int{SmokeOff: 0, SmokeOn: 1, SmokeOnce: 2}[m.launcherSmoke()]
			if k == "right" {
				cur = (cur + 1) % 3
			} else {
				cur = (cur + 2) % 3
			}
			return m.cmdAction(ActSetSmoke, order[cur])
		}
		if m.set5Sec == secLauncher && m.set5Row == set5CapRow { // prompt-cache cap (U-22)
			cur := 0
			for i, c := range capChoices {
				if c == m.launcherInfo().CacheCap {
					cur = i
				}
			}
			n := len(capChoices)
			if k == "right" {
				cur = (cur + 1) % n
			} else {
				cur = (cur + n - 1) % n
			}
			return m.cmdAction(ActSetCacheCap, capChoices[cur])
		}
		if m.set5Sec == secLauncher && (m.set5Row == set5WarmRow || m.set5Row == set5AutosaveRow) { // OQ-6 (a)
			id, cur := ActSetWarmStart, m.launcherInfo().WarmStart
			if m.set5Row == set5AutosaveRow {
				id, cur = ActSetAutosave, m.launcherInfo().Autosave
			}
			next := "on"
			if cur != "off" {
				next = "off"
			}
			return m.cmdAction(id, next)
		}
		if m.set5Sec == secLauncher && m.set5Row == set5LangRow { // language: this run only
			if k == "right" {
				m.langAt = (m.langAt + 1) % 3
			} else {
				m.langAt = (m.langAt + 2) % 3
			}
			m.r.lang = []string{m.auto, LangEnglish, LangKorean}[m.langAt]
			return nil
		}

		if m.set5Sec == secLauncher && m.set5Row >= 0 {
			return m.cmdAction(ActLauncherSet, "")
		}
	}
	return nil
}

// launcherInfo is the launcher settings the screen last read.
func (m *Model) launcherInfo() LauncherInfo {
	if m.launcher == nil {
		return LauncherInfo{}
	}
	return *m.launcher
}

func (m *Model) set5Rows() int {
	switch m.set5Sec {
	case secFolders:
		return max(1, len(m.cat.Folders))
	case secWrites:
		return len(m.writeRows())
	}
	return set5LauncherRows
}

// writeRow is one row of WHERE THE LAUNCHER WRITES.
type writeRow struct {
	label, path string
	file        bool
}

func (m *Model) writeRows() []writeRow {
	li := m.launcher
	if li == nil {
		li = &LauncherInfo{}
	}
	return []writeRow{
		{"setup", li.SetupDir, false},
		{"logs", li.LogDir, false},
		{"log file", li.LogFile, true},
		{"settings", li.SettingsDir, false},
	}
}
