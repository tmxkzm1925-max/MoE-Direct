// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// 4 Log (H11), 5 Settings (H5), ? help (H13) and the big-tile strip
// (HF). From the mock's h.go, keys.go, flow.go, pages.go.

package ui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"moedirect/launcher/internal/version"
)

// ---------- 4 Log ----------

// logTextCol is where a log line's text starts: "15:04:05  ✕  ".
const logTextCol = 13

// logShows: what a tab shows (review items 18, 29): all shows every line;
// warnings and errors show their own class; notices only in all.
func (m *Model) logShows(x LogLine) bool {
	switch m.logFilter {
	case 1:
		return logLevel(x) == LogWarn
	case 2:
		return logLevel(x) == LogError
	}
	return true
}

func (m *Model) logFiltered() []LogLine {
	var out []LogLine
	for _, x := range m.logs {
		if m.logShows(x) {
			out = append(out, x)
		}
	}
	return out
}

// logTextW is the width of a log line's text inside the box.
func (m *Model) logTextW() int { return max(1, m.r.cw()-4-logTextCol) }

// logRoom is the box's height in lines, the same on every tab and at every
// scroll position (review item 20).
func (m *Model) logRoom() int { return max(1, m.r.H-11) }

// logScrollMax is the scroll limit: max(0, screen lines − rows) (review
// item 20); a list that fits does not scroll.
func (m *Model) logScrollMax() int {
	return max(0, len(m.logScreenLines(m.logFiltered(), m.logTextW()))-m.logRoom())
}

// logScreenLines wraps log lines at the box width: the time and the mark on
// the first line, the rest indented to the text column (review item 7). The
// four classes (review item 29): error red ✕, warning yellow ▲, notice cyan
// i with a "notice" head, info grey ·.
func (m *Model) logScreenLines(ls []LogLine, w int) []string {
	r := m.r
	var out []string
	for _, x := range ls {
		s, sym, mst, head := dim, "·", txt, ""
		switch logLevel(x) {
		case LogOK:
			s, sym = gr, "✓"
		case LogWarn:
			s, sym, mst = yeB, "▲", ye
		case LogError:
			s, sym, mst = rdB, "✕", rd
		case LogNotice:
			s, sym, mst, head = cyB, "i", cy, r.T("notice")+"  "
		}
		t := "--:--:--"
		if !x.At.IsZero() {
			t = x.At.Format("15:04:05")
		}
		ps := wrapCells(head+r.logText(x), w)
		if r.W < fullW || r.H < fullH {
			// below 80x28 a line wraps between words, so no word is split (S8-LA-09)
			ps = wrapWords(r.logText(x), max(1, w-lipgloss.Width(head)))
			ps[0] = head + ps[0]
		}
		for i, p := range ps {
			if i > 0 {
				out = append(out, rep(" ", logTextCol)+mst.Render(p))
				continue
			}
			l := dim.Render(t) + "  " + s.Render(sym) + "  "
			if head != "" && strings.HasPrefix(p, head) {
				l += cyB.Render(head) + mst.Render(strings.TrimPrefix(p, head))
			} else {
				l += mst.Render(p)
			}
			out = append(out, l)
		}
	}
	return out
}

func (m *Model) vH11() string {
	r := m.r
	warn, errs := 0, 0
	for _, x := range m.logs {
		switch logLevel(x) {
		case LogWarn:
			warn++
		case LogError:
			errs++
		}
	}
	sl := m.logScreenLines(m.logFiltered(), m.logTextW())
	room := m.logRoom()
	back := min(m.logBack, max(0, len(sl)-room))
	end := len(sl) - back
	rs := append([]string{}, sl[max(0, end-room):end]...)
	if len(m.logs) == 0 {
		rs = []string{dim.Render(r.T("no log lines yet"))}
	}
	for len(rs) < room {
		rs = append(rs, "") // the box keeps its height (review item 20)
	}
	// tabs: all n │ warnings n │ errors n (review item 18)
	var tabs []string
	for i, x := range []struct {
		name string
		n    int
	}{{"all", len(m.logs)}, {"warnings", warn}, {"errors", errs}} {
		t := fmt.Sprintf("%s %d", r.T(x.name), x.n)
		if i == m.logFilter {
			tabs = append(tabs, hiB.Render(t))
		} else {
			tabs = append(tabs, dim.Render(t))
		}
	}
	title := strings.Join(tabs, dim.Render(" │ "))
	// held: the state itself, no "paused" (review item 6)
	status := ""
	if back > 0 {
		status = "  " + ye.Render(r.Tf("▲ %s lines up · %s new · End=newest", fmt.Sprint(back), fmt.Sprint(m.logNew)))
	}
	file := r.np()
	if m.launcher != nil && m.launcher.LogFile != "" {
		file = m.launcher.LogFile
	}
	open := "  " + cyB.Render("e") + txt.Render(" "+r.T("open"))
	where := "  " + dim.Render(r.T("file")+"  ") + txt.Render(file) + open
	if m.launcher != nil && m.launcher.LogDir != "" {
		// the log folder and its files, newest first (S8-LA-36; edition 8.1)
		head := "  " + dim.Render(r.T("logs")+"  ")
		room := max(1, r.W-lipgloss.Width(head)-lipgloss.Width(open))
		dir := midPath(m.launcher.LogDir, max(8, room/2))
		list := fmt.Sprintf("%d", len(m.launcher.LogFiles))
		if len(m.launcher.LogFiles) > 0 {
			list += " · " + strings.Join(m.launcher.LogFiles, " · ")
		}
		where = head + txt.Render(dir) + dim.Render(" · ") + txt.Render(cut(list, max(1, room-lipgloss.Width(dir)-3))) + open
	}
	body := []string{
		indent(r.panel(r.cw(), "8", title, rs...), 1),
		status,
		where,
	}
	return m.page(plLog, r.pageHead(r.T("what happened, newest last"), ""), body, foot{next: []string{
		"↑ holds the view · End jumps to the newest · Tab switches all, warnings, errors",
		"↑ holds the view · End newest · Tab switches",
	}, here: []string{"↑↓", "scroll", "tab", "lists", "e", "open file", "end", "newest"}})
}

// ---------- 5 Settings ----------

func (m *Model) vH5() string {
	r := m.r
	var rs []string
	rs = append(rs, dim.Render("  "+pad(r.T("folder"), 18)+pad(r.T("state"), 13)+pad(r.T("found"), 16)+lpad(r.T("size"), 6)))
	if len(m.cat.Folders) == 0 {
		if showNotPorted {
			rs = append(rs, "  "+dim.Render(r.T("model folders")+": "+r.np()+" · "+r.T("the catalog names the model files")))
		} else { // UX-R4: no not-ported value in the release build
			rs = append(rs, "  "+dim.Render(r.T("the catalog names the model files")))
		}
	}
	for i, x := range m.cat.Folders {
		var s string
		switch x.State {
		case FolderWatching:
			s = gr.Render(pad("● "+r.T("watching"), 13))
		case FolderMissing:
			s = rd.Render(pad("✕ "+r.T("missing"), 13))
		case FolderEmpty:
			s = ye.Render(pad("○ "+r.T("empty"), 13))
		case FolderUnsupported:
			s = ye.Render(pad("? "+r.T("unsupported"), 13))
		default:
			s = dim.Render(pad("— "+r.np(), 13)) // 13: "— not ported" filled the 12-cell column (GOL-WIRE)
			if !showNotPorted {
				s = dim.Render(pad("—", 13)) // UX-R4
			}
		}
		found := ""
		if x.Found > 0 {
			found = fmt.Sprintf("%d %s", x.Found, r.T("models"))
		}
		mark := "  "
		if m.set5Sec == secFolders && m.set5Row == i {
			mark = cyB.Render("▶ ")
		}
		size := r.vt(x.SizeGB, g0str)
		if x.SizeGB.State == NotPorted && !showNotPorted {
			size = "—" // UX-R4
		}
		l := mark + txt.Render(pad(x.Path, 18)) + s + txt.Render(pad(found, 16)) + txt.Render(lpad(size, 6))
		if r.wide() && x.Note != "" {
			l += "    " + dim.Render(x.Note)
		}
		rs = append(rs, l)
	}
	rs = append(rs, "  "+cy.Render("+ "+r.T("add a folder")))
	fEdge, fTitle := "8", dim.Render(r.T("MODEL FOLDERS")+" · "+r.T("where models are found"))
	if m.set5Sec == secFolders { // UX-R4: the whole title marks the focused section, not its first word
		fEdge, fTitle = "14", cyB.Render("▶ "+r.T("MODEL FOLDERS")+" · "+r.T("where models are found"))
	}
	folders := indent(r.panel(r.cw(), fEdge, fTitle, rs...), 1)

	li := m.launcher
	if li == nil {
		li = &LauncherInfo{}
	}
	or := func(s string) string {
		if s == "" {
			return r.np()
		}
		return s
	}
	autoName := r.T("auto · English") // UX-R1: auto is the start language, English unless -lang ko
	if m.auto == LangKorean {
		autoName = r.T("auto · Korean")
	}
	langName := map[int]string{0: autoName, 1: "English", 2: "한국어"}[m.langAt]
	smoke := map[SmokeMode]string{SmokeOff: r.T("off"), SmokeOn: r.T("on"), SmokeOnce: r.T("once per model")}[li.Smoke]
	if smoke == "" {
		smoke = r.np()
	}
	kept := r.np()
	if li.LogsKeptDays.Ok() {
		kept = fmt.Sprintf("%.0f %s", li.LogsKeptDays.V, r.T("days"))
	}
	capText := map[string]string{"": r.np(), "off": r.T("off · the default")}[li.CacheCap]
	if capText == "" {
		capText = li.CacheCap + r.T(" MiB")
	}
	if li.CacheCapMax.Ok() { // GL-06: E.3's allowed maximum beside the choice, not a default (L:4420)
		capText += " · " + r.Tf("max %s MiB at the last check", fmt.Sprintf("%.0f", li.CacheCapMax.V))
	}
	onOff := func(s string) string {
		if s == "" {
			return r.np()
		}
		return r.T(s)
	}
	items := [][2]string{
		{"address", or(li.Address)},
		{"language", langName},
		{"check run", smoke},
		{"on open", or(li.OnOpen)},
		{"logs kept", kept},
		{"prompt cache", capText},
		{"warm start", onOff(li.WarmStart)},
		{"cache autosave", onOff(li.Autosave)},
	}
	rowKey := map[int]string{set5CapRow: "prompt_cache_cap", set5WarmRow: "warmstart", set5AutosaveRow: "autosave"}
	var ls []string
	labW := 15 // UX-R4: the label column fits the longest shown label («시작 후 응답 시험»)
	for i, it := range items {
		if !m.set5Hidden(secLauncher, i) {
			labW = max(labW, lipgloss.Width(r.T(it[0]))+1)
		}
	}
	inner := r.cw() - 4
	if r.wide() {
		inner = r.cols(2)[0] - 4
	}
	for i, it := range items {
		if m.set5Hidden(secLauncher, i) { // UX-R4: a not-ported row is not shown
			continue
		}
		next := ""
		if li.NextStart[rowKey[i]] { // rows 6.1-6.2
			next = ye.Render("  ▲")
		}
		// UX-R9 addendum (2): a value wider than its room (the address «127.0.0.1 · port from the
		// catalog (8093)») folds onto the lines below, under the value column, never cut (a
		// value that cycles with ←→ keeps its one line)
		cycle := i == set5LangRow || i == set5SmokeRow || i == set5CapRow || i == set5WarmRow || i == set5AutosaveRow
		if vw := inner - 2 - labW - lipgloss.Width(next); !cycle && lipgloss.Width(it[1]) > vw {
			ws := wrapWords(it[1], max(4, vw))
			mark, lab, sty := "  ", txt.Render(pad(r.T(it[0]), labW)), txt
			if m.set5Sec == secLauncher && m.set5Row == i {
				mark, lab, sty = cyB.Render("▶ "), hiB.Render(pad(r.T(it[0]), labW)), hiB
			}
			ls = append(ls, mark+lab+sty.Render(ws[0])+next)
			for _, w := range ws[1:] {
				ls = append(ls, rep(" ", 2+labW)+sty.Render(w))
			}
			continue
		}
		if m.set5Sec == secLauncher && m.set5Row == i {
			v := hiB.Render(it[1])
			if i == set5LangRow || i == set5SmokeRow || i == set5CapRow || i == set5WarmRow || i == set5AutosaveRow {
				v = cyB.Render("◀ " + it[1] + " ▶")
			}
			if i == set5CapRow && m.inputOn && m.inputLauncher {
				v = cyB.Render("[ "+m.inputBuf+"_ ]") + dim.Render(r.T(" MiB"))
			}
			ls = append(ls, cyB.Render("▶ ")+hiB.Render(pad(r.T(it[0]), labW))+v+next)
			continue
		}
		ls = append(ls, "  "+txt.Render(pad(r.T(it[0]), labW))+txt.Render(it[1])+next)
	}
	if len(li.NextStart) > 0 {
		for _, l := range wrapWords("▲ "+r.T("not applied to the running server · applies at the next start"), inner) {
			ls = append(ls, ye.Render(l))
		}
	}
	if m.inputOn && m.inputLauncher && m.inputErr != "" {
		for _, l := range wrapWords("✕ "+typedErr(r, m.inputErr), inner) {
			ls = append(ls, rd.Render(l))
		}
	}
	// UX-R25 (item 34): the focused row's description is drawn whole in its own box under this
	// one (below), and no description names a key: the footer names the focused row's keys
	hints := []string{
		"address = where chat apps connect · next start · this PC only",
		"language = English at start · a change lasts for this run",
		"check run = a short chat test after a start · off, on, or once per model · results in 4 Log",
		"on open = the page shown first",
		"logs kept = days before old logs go",
		"prompt cache = RAM for reused prompts · off (0) by default · next start",
		"warm start = the server reloads its last cache at start · next start",
		"cache autosave = the cache is saved while the server runs · next start",
	}
	h := hints[0]
	if m.set5Sec == secLauncher && m.set5Row >= 0 && m.set5Row < len(hints) {
		h = hints[m.set5Row]
	}
	if m.bundle.State == BundleDev { // conflict 18-2
		ls = append(ls, dim.Render(r.T("development build · launcher file check skipped")))
	}
	lw := r.cw()
	if r.wide() {
		lw = r.cols(2)[0]
	}
	lEdge, lTitle := "8", dim.Render(r.T("LAUNCHER")+" · "+r.T("same for every model"))
	if m.set5Sec == secLauncher {
		lEdge, lTitle = "14", cyB.Render("▶ "+r.T("LAUNCHER")+" · "+r.T("same for every model"))
	}
	launcher := r.panel(lw, lEdge, lTitle, ls...)

	// WHERE THE LAUNCHER WRITES: entered by Tab, ↑↓ rows, ⏎ opens; each path
	// on one line, shortened in the middle (review item 24)
	ww := r.cw()
	if r.wide() {
		ww = r.cols(2)[1]
	}
	work := func(w int) []string {
		var out []string
		for i, x := range m.writeRows() {
			if m.set5Hidden(secWrites, i) { // UX-R4: no path yet, nothing to show
				continue
			}
			mark, k := "  ", dim.Render(pad(r.T(x.label), 10))
			if m.set5Sec == secWrites && m.set5Row == i {
				mark, k = cyB.Render("▶ "), hiB.Render(pad(r.T(x.label), 10))
			}
			p := r.np()
			if x.path != "" {
				p = midPath(x.path, max(4, w-4-12))
			}
			out = append(out, mark+k+txt.Render(p))
		}
		return out
	}
	wEdge, wTitle := "8", dim.Render(r.T("WHERE THE LAUNCHER WRITES"))
	if m.set5Sec == secWrites {
		wEdge, wTitle = "14", cyB.Render("▶ "+r.T("WHERE THE LAUNCHER WRITES"))
	}
	var right, writesBox string
	if r.wide() {
		c := r.cols(2)
		right = r.panel(c[1], wEdge, wTitle, work(ww)...)
		for _, f := range m.cat.Folders {
			if len(f.Skipped) > 0 {
				// UX-R4 (finding 8b): the unsupported GGUF files of the folder, each file name on one
				// line (shortened in the middle) and its reason folded below it, read in full
				var why []string
				bw := max(8, c[1]-4)
				for _, s := range f.Skipped {
					name, reason, _ := strings.Cut(s, "  ")
					why = append(why, ye.Render("? ")+txt.Render(midPath(name, max(4, bw-2))))
					for _, l := range wrapWords(strings.TrimSpace(reason), max(4, bw-4)) {
						why = append(why, "    "+dim.Render(l))
					}
				}
				t := r.T("WHY") + " " + f.Path + " " + r.T("HAS NONE")
				if f.Found > 0 {
					t = r.T("UNSUPPORTED GGUF") + " · " + r.T("SKIPPED") + " · " + f.Path
				}
				right = lines(right, r.panel(c[1], "8", dim.Render(t), why...))
				break
			}
		}
	} else {
		writesBox = r.panel(r.cw(), wEdge, wTitle, work(ww)...)
	}
	var f foot
	switch {
	case m.set5Row < 0:
		f = foot{next: []string{"↓ chooses a row · Tab moves between the sections · esc lets go of the row"},
			here: []string{"tab", "section", "↓", "choose a row", "a", "add"}}
	case m.set5Sec == secFolders:
		f = foot{next: []string{"↓ past the last folder goes on to the launcher rows · Tab skips a section"},
			here: []string{"↑↓", "row", "tab", "section", "a", "add", "x", "remove"}}
	case m.set5Sec == secWrites:
		f = foot{next: []string{"⏎ opens the folder, or shows the file in Explorer"},
			here: []string{"↑↓", "row", "⏎", "open", "tab", "section"}}
	default:
		f = foot{next: []string{
			"language changes at once (this run) · the other launcher settings: not ported · ↑ past the first row goes back to the folders",
			"language changes at once · others: not ported",
		}, here: []string{"tab", "section", "↑↓", "item", "←→", "change", "a", "add"}}
		if !showNotPorted { // UX-R4: no not-ported claim in the release build
			f.next = []string{"language changes at once (this run) · ↑ past the first row goes back to the folders",
				"language changes at once (this run)"}
		}
		if m.set5Row == set5CapRow && !m.inputOn { // UX-R25: Space, out of the description (a, add a folder, gives it room)
			f.here = []string{"space", "type a value", "tab", "section", "↑↓", "item", "←→", "change"}
		}
	}
	// UX-R25 (item 34): the description box takes the rows the page can spare, at least two,
	// and no more than the longest shown description needs, so a larger window shows more of
	// a long one and the box keeps its height while ↑↓ moves. When the page has less to spare
	// (a low window, the next-start note, the development build line), the blank line under
	// the folders box gives way first; then a whole box leaves, never a frame without its
	// bottom edge (UX-R13): the writes box while a launcher row is focused (one column), else
	// the description box
	room := r.H - len(r.nav(plSettings, m.srvState())) - 2 - len(r.footer(f, m.arm(), m.notice))
	height := func(s string) int { return strings.Count(s, "\n") + 1 }
	free := room - height(folders) - 1 - height(launcher) - 2
	if !r.wide() {
		free -= height(writesBox)
	}
	gap, showDesc := true, true
	if free < 2 {
		gap, free = false, free+1
	}
	if free < 2 {
		if !r.wide() && m.set5Sec == secLauncher {
			free += height(writesBox)
			writesBox = ""
		} else {
			showDesc = false
		}
	}
	dIn := max(4, lw-4)
	need := 2
	for i, x := range hints {
		if !m.set5Hidden(secLauncher, i) {
			need = max(need, len(wrapWords(r.T(x), dIn)))
		}
	}
	n := min(need, max(2, free))
	dl := wrapWords(r.T(h), dIn)
	if len(dl) > n { // a window too low for the whole text: the last row is cut with …
		dl = append(dl[:n-1], strings.Join(dl[n-1:], " "))
	}
	for len(dl) < n {
		dl = append(dl, "")
	}
	for i := range dl {
		dl[i] = txt.Render(dl[i])
	}
	left := []string{launcher}
	if showDesc {
		left = append(left, r.panel(lw, "8", dim.Render(r.T("DESCRIPTION")), dl...))
	}
	var bottom string
	if r.wide() {
		bottom = row(lines(left...), right)
	} else {
		if writesBox != "" {
			left = append(left, writesBox)
		}
		for i := range left {
			left[i] = indent(left[i], 1)
		}
		bottom = lines(left...)
	}
	body := []string{folders, "", bottom}
	if !gap {
		body = []string{folders, bottom}
	}
	return m.page(plSettings, r.pageHead(r.T("launcher settings and model folders"), r.T("model settings: 1 Models")), body, f)
}

// ---------- ? Help ----------

func (m *Model) vH13() string {
	r := m.r
	type pl struct{ k, name, what, keys string }
	ps := []pl{
		{"1", "Models", "pick a model, change settings, start", "↑↓ tab ⏎"},
		{"2", "Server", "steps · prefill · TTFT · decode · unload", "ctrl+f c"},
		{"3", "Memory", "can it start · what changed when slow", "⏎ o"},
		{"4", "Log", "everything that happened", "tab ↑↓ end e"},
		{"5", "Settings", "launcher settings · model folders", "tab a"},
	}
	var rs []string
	for _, p := range ps {
		rs = append(rs, cyB.Render(p.k)+"  "+hiB.Render(pad(r.T(p.name), 10))+txt.Render(pad(r.T(p.what), 44))+cy.Render(p.keys))
	}
	any := []string{
		cyB.Render(pad("←→", 8)) + txt.Render(r.T("previous · next page")),
		cyB.Render(pad("1-5", 8)) + txt.Render(r.T("go to that page")),
		cyB.Render(pad("?", 8)) + txt.Render(r.T("this help · again to close")),
		cyB.Render(pad("esc", 8)) + txt.Render(r.T("back · close · undo")),
		cyB.Render(pad("q", 8)) + txt.Render(r.T("twice: unload, then close")),
		cyB.Render(pad("ctrl+c", 8)) + txt.Render(r.T("twice: close now, no cache save")),
	}
	// conflict 5: the X line says what closing the window does, wrapped
	aw := r.cw()
	if r.wide() {
		aw = r.cols(2)[0]
	}
	for i, l := range wrapWords(r.T("closing the window stops the server at once · nothing saved (same as ctrl+c)"), max(1, aw-4-8)) {
		head := rep(" ", 8)
		if i == 0 {
			head = dim.Render(pad("X", 8))
		}
		any = append(any, head+txt.Render(l))
	}
	where := []string{
		txt.Render(pad(r.T("model files"), 19)) + cy.Render("5") + dim.Render(" "+r.T("Settings")),
		txt.Render(pad(r.T("model settings"), 19)) + cy.Render("1") + dim.Render(" "+r.T("Models")+", tab"),
		txt.Render(pad(r.T("launcher settings"), 19)) + cy.Render("5") + dim.Render(" "+r.T("Settings")),
		txt.Render(pad(r.T("log file"), 19)) + cy.Render("4") + dim.Render(" "+r.T("Log")+", e"),
		txt.Render(pad(r.T("address"), 19)) + cy.Render("2") + dim.Render(" "+r.T("Server")+", c"),
		txt.Render(pad(r.T("file check · hash"), 19)) + cy.Render("3") + dim.Render(" "+r.T("Memory")),
	}
	var bottom string
	if r.wide() {
		c := r.cols(2)
		bottom = row(r.panel(c[0], "8", dim.Render(r.T("ANYWHERE")), any...), r.panel(c[1], "8", dim.Render(r.T("WHERE IS…")), where...))
	} else {
		bottom = lines(indent(r.panel(r.cw(), "8", dim.Render(r.T("ANYWHERE")), any...), 1), indent(r.panel(r.cw(), "8", dim.Render(r.T("WHERE IS…")), where...), 1))
	}
	body := []string{indent(r.panel(r.cw(), "14", cyB.Render(r.T("PAGES")), rs...), 1), "", bottom}
	if !r.wide() {
		body = []string{indent(r.panel(r.cw(), "14", cyB.Render(r.T("PAGES")), rs...), 1), bottom}
	}
	// GL-01 (F-LS 5.1; S8-LA-30): the about text, the launcher's name and release version, from
	// internal/version, at the right of the help page's head.
	return r.screen(r.nav(m.helpFrom, m.srvState()), r.pageHead(r.T("every page and key"), version.About()), body,
		r.footer(foot{next: []string{"esc goes back to where you were · tab shows the big tile of 2 Server"}, here: []string{"esc", "close", "tab", "tile states"}}, m.arm(), m.notice))
}

// vHF: how the big tile of 2 Server moves through one request (the mock's
// flow.go strip). The numbers are the live ones when a server serves, "—"
// otherwise; the slow rule is the design's text.
func (m *Model) vHF() string {
	r := m.r
	s := m.status
	if s == nil {
		s = &ServeStatus{}
	}
	pc := ""
	if s.PrefillDone.Ok() && s.PrefillTotal.Ok() && s.PrefillTotal.V > 0 {
		pc = fmt.Sprint(int(s.PrefillDone.V / s.PrefillTotal.V * 100))
	}
	type state struct {
		edge, label, num string
		sty              lipgloss.Style
		unit, mark       string
		cap              [3]string
	}
	ss := []state{
		{"8", "DECODE", numOr(s.LastDecodeTokS), dim, "tok/s", dim.Render(r.T("last reply")),
			[3]string{r.T("no request"), r.T("shows the last reply"), r.T("grey frame")}},
		{"14", "PREFILL", pc, cyB, "%", txt.Render("TTFT ~"),
			[3]string{r.T("a request arrives"), r.T("share of the prompt read"), r.T("TTFT estimate under it")}},
		{"10", "DECODE", numOr(s.DecodeTokS), grB, "tok/s", txt.Render(r.T("live")),
			[3]string{r.T("the first token is out"), r.T("new number every 1 s"), r.T("average of the last 5 s")}},
		{"11", "DECODE", numOr(s.LastDecodeTokS), yeB, "tok/s", txt.Render(r.T("usual") + " " + num(s.UsualTokS, "%.1f")),
			[3]string{r.T("under 80 % of usual"), r.T("the number turns yellow"), r.T("reply ends slow: ▲ slow")}},
	}
	a := r.cw()
	per, w := 4, (a-3*3)/4
	if !r.wide() {
		per, w = 2, (a-3)/2
	}
	w = max(w, 8)
	arrow := lines("", "", dim.Render(" → "), "", "", "", "", "")
	block := func(x state) string {
		t := r.heroL(x.edge, x.label, x.num, x.sty, x.unit, x.mark)(w)
		c := " " + txt.Render(x.cap[0]) + "\n " + dim.Render(x.cap[1]) + "\n " + dim.Render(x.cap[2])
		return lipgloss.NewStyle().Width(w).Render(t + "\n" + c)
	}
	var rows []string
	for i := 0; i < len(ss); i += per {
		var p []string
		for j := i; j < i+per && j < len(ss); j++ {
			if j > i {
				p = append(p, arrow)
			}
			p = append(p, block(ss[j]))
		}
		rows = append(rows, row(p...), "")
	}
	type rule struct{ k, wide, narrow string }
	rs := []rule{
		{"usual", "median decode of this model's last 20 replies at a similar context (±4k)", "median decode, last 20 replies, similar context"},
		{"live", "the number turns yellow while its 5 s average stays under 80 % of usual", "number turns yellow under 80 % of usual"},
		{"slow", "a reply of 64 tokens or more ends under 80 % of usual: ▲ slow and the alert line", "a reply (64+ tokens) ends under 80 %: ▲ slow"},
		{"clears", "the next reply that ends at 90 % of usual or more", "the next reply at 90 % of usual or more"},
		{"context", "a longer context decodes slower by itself, so it is compared only with similar context", "longer context is slower by itself: compared alike"},
		{"not used", "prefill and TTFT are shown and logged, but do not set ▲ slow", "prefill and TTFT are shown, do not set ▲ slow"},
	}
	var rl []string
	for _, x := range rs {
		t := x.wide
		if !r.wide() {
			t = x.narrow
		}
		rl = append(rl, cyB.Render(pad(r.T(x.k), 10))+txt.Render(r.T(t)))
	}
	note := r.T("the slow verdict comes from the backend")
	if !s.UsualTokS.Ok() {
		note += " · " + r.T("usual") + ": " + r.vt(s.UsualTokS, f1)
	}
	rows = append(rows, indent(r.panel(a, "11", yeB.Render("▲ "+strings.ToUpper(r.T("slow")))+dim.Render(" · "+r.T("when the status line says it")), rl...), 1), r.alert("·", txt, note))
	return r.screen(r.nav(m.helpFrom, m.srvState()), r.pageHead(r.T("2 Server · the big tile through one request"), r.T("left to right")), rows,
		r.footer(foot{next: []string{"esc goes back to where you were · tab returns to the key help"}, here: []string{"esc", "close", "tab", "keys"}}, m.arm(), m.notice))
}

// typedErr is a refused typed value in the screen's language: the
// settings-typed text of F-LS 11.4 carries its range as a value. The text is
// one literal, the table's key (S8-LA-36: every key is a source word).
func typedErr(r rc, s string) string {
	f := "outside its range · allowed %s · the value is kept, nothing applied"
	i := strings.Index(f, "%s")
	head, tail := f[:i], f[i+2:]
	if len(s) >= len(head)+len(tail) && strings.HasPrefix(s, head) && strings.HasSuffix(s, tail) {
		return r.Tf(f, s[len(head):len(s)-len(tail)])
	}
	return r.T(s)
}
