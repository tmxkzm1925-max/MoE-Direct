// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// 2 Server: H15 starting, H6 first-run setup, H7 failed, H8/H9 serving,
// H8Q q pressed once, HQ quitting, H12 stopped. From the mock's h.go, s.go,
// keys.go.

package ui

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

func (m *Model) vS0() string {
	r := m.r
	body := []string{
		"  " + dim.Render("○ "+r.T("no server is running")),
		"",
		"  " + txt.Render(r.T("⏎ on 1 Models starts the chosen model; its steps show here")),
	}
	return m.page(plServer, r.pageHead(r.T("no server"), ""), body, foot{next: []string{"1 Models picks a model and starts it"}})
}

// stepRows draws the step list of a start or a stop. The running row holds
// the bar of decision ③.
// running gives the running row its tail text, its time text and its bar.
type runningRow func() (tail, right string, barFn func(w int) string)

func (m *Model) stepRows(ss []Step, nameW int, running runningRow) []string {
	r := m.r
	in := r.cw() - 4
	const timeW = 10
	detW := max(4, in-nameW-2-timeW)
	var rs []string
	for _, x := range ss {
		t := ""
		if x.Took > 0 {
			t = fmt.Sprintf("%.1f s", x.Took.Seconds())
		}
		name := r.T(x.Name)
		var l string
		switch x.State {
		case StepDone, StepSkipped:
			d := x.Detail
			if x.State == StepSkipped && d == "" {
				d = r.T("skipped")
			}
			l = gr.Render("✓ ") + txt.Render(pad(name, nameW)) + dim.Render(pad(cut(d, detW), detW)) + dim.Render(lpad(t, timeW))
		case StepRunning:
			if running != nil {
				tail, right, barFn := running()
				bw := max(0, detW-lipgloss.Width(tail)-1)
				b := ""
				if barFn != nil {
					b = barFn(bw)
				} else {
					b = rep(" ", bw)
				}
				l = cyB.Render("▶ ") + hiB.Render(pad(name, nameW)) + b + hi.Render(tail) + " " + hi.Render(lpad(right, timeW))
			} else {
				l = cyB.Render("▶ ") + hiB.Render(pad(name, nameW)) + txt.Render(pad(cut(x.Detail, detW), detW))
			}
		case StepFailed:
			l = rdB.Render("✕ ") + hiB.Render(pad(name, nameW)) + rd.Render(pad(cut(x.Detail, detW), detW))
		case StepNotPorted:
			l = dim.Render("· "+pad(name, nameW)) + dim.Render(pad(cut(r.snp(), detW), detW)) + dim.Render(lpad("—", timeW)) // UX-R12: a dash with the switch off
		default:
			l = dim.Render("○ "+pad(name, nameW)) + txt.Render(pad(cut(x.Detail, detW), detW)) + dim.Render(lpad("—", timeW))
		}
		rs = append(rs, l)
	}
	return rs
}

func curName(ss []Step) string { _, n := curStep(ss); return n }

func curStep(ss []Step) (int, string) {
	for i, s := range ss {
		if s.State == StepRunning || s.State == StepFailed {
			return i, s.Name
		}
	}
	n := 0
	for i, s := range ss {
		if s.State == StepDone || s.State == StepSkipped {
			n = i + 1
		}
	}
	if n < len(ss) {
		return n, ss[n].Name
	}
	return max(0, len(ss)-1), ""
}

// runWords are the values the server runs with, in plain words (review
// item 16): the backend's words while serving, else the settings rows.
// settingWordKeys are the backend's settings words that carry a number, translated by their
// pattern for any number: «n GB cache» (UX-R12 addendum) and «n k context» (UX-R12b)
var settingWordKeys = []string{"%s cache", "%s context"}

// settingWord is one of the backend's settings words in the screen's language.
func (r rc) settingWord(x string) string {
	if x != "" && x[0] >= '0' && x[0] <= '9' {
		for _, k := range settingWordKeys {
			if vals, ok := matchKey(k, x); ok {
				return r.Tf(k, vals...)
			}
		}
	}
	return r.T(x)
}

func (m *Model) runWords() string {
	r := m.r
	if m.status != nil && len(m.status.SettingsWords) > 0 {
		var w []string
		for _, x := range m.status.SettingsWords {
			w = append(w, r.settingWord(x))
		}
		return strings.Join(w, " · ")
	}
	if row, ok := m.rowByID(m.runModel); ok {
		if s := m.settingsSummary(row); s != "" {
			return s
		}
	}
	if m.status != nil && m.status.SettingsLine != "" {
		return m.status.SettingsLine
	}
	if !showNotPorted { // UX-R9 addendum (1): runFrame leaves the settings line out
		return ""
	}
	return r.np()
}

// runFrame holds the address and the settings of this run in one frame as
// wide as the tile row, the names in one column, the API name after the
// address (review items 15, 16, 17, 19, 27). Copying the address is the
// footer's c, not a mark on this line.
func (m *Model) runFrame(addr, alias string) string {
	r := m.r
	lw := max(lipgloss.Width(r.T("address")), lipgloss.Width(r.T("settings")), lipgloss.Width(r.T("prefetch"))) + 2
	var rs []string
	if addr != "" {
		l := dim.Render(pad(r.T("address"), lw)) + hiB.Render(addr)
		if alias != "" {
			l += dim.Render("   "+r.T("API name")+" ") + hi.Render(alias)
		}
		rs = append(rs, l)
	}
	// where the settings came from, at the line's end (conflict 3); the line
	// wraps by its pieces under the first one. UX-KR (item 24 (e)): the labels dim, the values
	// bright (as 1 Models' settings box) and the dots between them dim, so the dot does not
	// carry the reading alone
	var toks []piece
	if words := m.runWords(); words != "" {
		for _, w := range strings.Split(words, " · ") {
			toks = append(toks, piece{w, hi})
		}
		toks = append(toks, m.runProvenanceToks()...)
	}
	for i, l := range wrapToks(toks, max(1, r.cw()-4-lw)) {
		head := rep(" ", lw)
		if i == 0 {
			head = dim.Render(pad(r.T("settings"), lw))
		}
		rs = append(rs, head+l)
	}
	// GL-21: the engine's prefetch values after ready (F-PF §8.5); UX-KR (decision 4): the
	// launcher's candidate and its reasons stay in 4 Log (wire: the start's candidate line), the
	// row holds what the engine applied, one short row
	if m.status != nil && len(m.status.Prefetch) > 0 {
		var ps []piece
		for _, w := range enginePrefetch(m.status.Prefetch) {
			ps = append(ps, piece{r.T(pfWord(w)), hi}) // UX-R9 (finding 20 (d)): sentences, the codes stay in the log
		}
		for i, l := range wrapToks(ps, max(1, r.cw()-4-lw)) {
			head := rep(" ", lw)
			if i == 0 {
				head = dim.Render(pad(r.T("prefetch"), lw))
			}
			rs = append(rs, head+l)
		}
	}
	return indent(r.panel(r.cw(), "8", dim.Render(r.T("APPLIED SETTINGS, change in Models")), rs...), 1)
}

// enginePrefetch is the engine's part of the backend's prefetch words: the pieces before the
// launcher's candidate («candidate on» / «candidate off»), which the screen leaves to 4 Log
// (UX-KR, decision 4).
func enginePrefetch(ws []string) []string {
	for i, w := range ws {
		if w == "candidate on" || w == "candidate off" {
			return ws[:i]
		}
	}
	return ws
}

// PrefetchSentences is the prefetch words in the screen's English sentences, for the log line
// that keeps the candidate the screen leaves out (UX-KR, decision 4).
func PrefetchSentences(ws []string) string {
	out := make([]string, 0, len(ws))
	for _, w := range ws {
		out = append(out, pfWord(w))
	}
	return strings.Join(out, " · ")
}

// pfReasons are the codes the prefetch read-back line can show, in the user's words (UX-R9,
// finding 20 (d)): the engine's prefetch_off_reason (unset, invalid_k, invalid_n,
// invalid_range, adapter_none), the launcher candidate's off reasons (core resolvePrefetchManual
// and the preflight's derived_t_mismatch, with each prefetch_evidence value of the catalog) and
// the candidate's provenance labels. The code itself stays in the log (PREFETCH_ENGINE and the
// echo lines).
var pfReasons = map[string]string{
	"unset":                               "K·N setting unconfirmed",
	"invalid_k":                           "the K given to the engine is not a whole number",
	"invalid_n":                           "the N given to the engine is not a whole number",
	"invalid_range":                       "K or N is outside what the engine accepts",
	"adapter_none":                        "this model family has no prefetch signal",
	"catalog_semantic_invalid":            "the catalog's prefetch entries contradict each other",
	"probe_failed":                        "the drive speed check before the start failed",
	"identity_not_exact":                  "this model file is not the one the catalog measured",
	"adapt_controller_not_shipped_phase5": "the adaptive mode is not in this release",
	"not_opted_in_evidence_unverified":    "off by default: no measurement shows a gain on this model",
	"not_opted_in_evidence_observe":       "off by default: a gain on this model is only observed, not proven",
	"not_opted_in_evidence_causal_replay": "off by default: a gain on this model is shown in replay only",
	"not_opted_in_evidence_paired_live":   "off by default: not switched on for this model",
	"phase4_hold_unresolved":              "held off until a pending measurement is settled",
	"init_t_out_of_range":                 "the starting depth is out of range for this model",
	"qd_below_prefetch_depth":             "QD is too low to leave room for prefetch",
	"derived_t_mismatch":                  "the computed depth does not match the catalog",
	"catalog-validated":                   "catalog values, measured",
	"init_v1-unvalidated":                 "starting values, not measured yet",
	"env-override":                        "values you set, not measured",
}

// pfWord is one piece of the prefetch line: a code as its sentence, a code the table does not
// know as «details on 4 Log» (never the code), any other piece (engine on, K 8, N_cap 1) as is.
func pfWord(w string) string {
	if s, ok := pfReasons[w]; ok {
		return s
	}
	if w == "engine off" { // UX-KR (decision 4): what the engine did, not its name
		return "not applied"
	}
	if w == "engine on" { // UX-KR2 (2): its pair
		return "applied"
	}
	if strings.ContainsAny(w, "_-") && !strings.Contains(w, " ") {
		return "details on 4 Log"
	}
	return w
}

// vH15: the server is starting (decision ③ load bar; decision ⑧: "warm
// start" only, no clock time).
func (m *Model) vH15() string {
	r := m.r
	ev := m.startEv
	if ev == nil {
		ev = &StartEvent{}
	}
	p := startLoad(ev, m.now().Sub(m.startAt))
	n, unit, note := r.progressHero(p)
	nsty, edge := cyB, "14"
	if p.Slower {
		nsty, edge = yeB, "11"
	}
	ci, cname := curStep(ev.Steps)
	total := len(ev.Steps)
	ram := r.svt(ev.RAMGB, gstr)
	if ev.LastRAM.Ok() {
		ram += dim.Render(r.Tf(" of %s", f1(ev.LastRAM.V)))
	}
	body := []string{
		r.tiles(
			r.heroL(edge, "STARTING", n, nsty, unit, dim.Render(note)),
			r.sm("8", "STEP", hiB.Render(fmt.Sprint(min(ci+1, total)))+dim.Render(fmt.Sprintf(" %s %d", r.T("of"), total)),
				func(w int) string {
					if total == 0 {
						return ""
					}
					return bar(float64(ci)/float64(total), w, cy)
				}, cname),
			r.sm("8", "DISK", r.svu(ev.DiskGBs, f1, " GB/s"), noGauge, "reading model"),
			r.sm("8", "RAM", hiB.Render(ram), func(w int) string {
				if ev.RAMGB.Ok() && ev.LastRAM.Ok() && ev.LastRAM.V > 0 {
					return bar(ev.RAMGB.V/ev.LastRAM.V, w, cy)
				}
				return ""
			}, "model so far"),
		),
		"",
	}
	// the running row: bar, then time
	rows := m.stepRows(ev.Steps, 14, func() (string, string, func(int) string) {
		tail := fmt.Sprintf(" %d %%", p.Pct)
		right := ""
		switch {
		case p.Ready:
			right = fmt.Sprintf("%d s", int(p.Elapsed.Seconds()+0.5))
		case p.SecondsOnly:
			tail = " " + r.T("first time")
			right = fmt.Sprintf("%d s", int(p.Elapsed.Seconds()))
		case p.Slower:
			tail = " 99 % · " + r.T("slower than last time")
			right = fmt.Sprintf("%d s", int(p.Elapsed.Seconds()))
		default:
			right = fmt.Sprintf("~%d s", int((p.Last-p.Elapsed).Seconds()+0.5))
		}
		return tail, right, func(w int) string { return r.progressBarOnly(p, w) }
	})
	title := cyB.Render(r.T("START STEPS"))
	if ev.Warm {
		title += dim.Render(" · " + r.T("warm start"))
	}
	if len(rows) == 0 {
		rows = []string{dim.Render(r.T("waiting for the first step …"))}
	}
	body = append(body, indent(r.panel(r.cw(), "14", title, rows...), 1), m.runFrame(ev.Address, ""))
	for _, n := range ev.Notices {
		body = append(body, r.alert("▲", ye, r.T(n)))
	}
	body = append(body, r.alert("✓", gr, r.T("you can switch pages · the start keeps going")))
	ctx := ""
	if ev.Warm {
		ctx = r.T("warm start")
	}
	return m.page(plServer, r.pageHead(fmt.Sprintf("%s %s", m.runName(), r.T("is starting")), ctx), body, foot{next: []string{
		"turns into the serving view by itself · ctrl+x cancels, nothing is lost",
	}, here: []string{"ctrl+x", "cancel start"}})
}

// progressBarOnly is the running row's bar: the fill of decision ③, or an
// empty track when there is no earlier run to fill against.
func (r rc) progressBarOnly(p progress, w int) string {
	if p.Ready {
		return bar(1, w, cy)
	}
	if p.SecondsOnly {
		return dim.Render(rep("░", w))
	}
	s := cy
	if p.Slower {
		s = ye
	}
	return bar(p.Frac, w, s)
}

// vH6: first-run setup until the server starts (UX-R14): the repack, the repacker's check
// (verify) and the start's speed check, with one percent over the three (setup_view.go).
func (m *Model) vH6() string {
	r := m.r
	ev := m.startEv
	n := m.setupState()
	sp := n.sp
	lp := loadProgress(sp.Elapsed+m.now().Sub(n.at), sp.LastS)
	left := r.np()
	if !showNotPorted { // UX-R9 addendum (1): the first setup has no last time, not a missing port
		left = r.T("first time")
	}
	if !lp.SecondsOnly {
		left = fmt.Sprintf("%d s", int((lp.Last - lp.Elapsed).Seconds()))
	}
	chain := ""
	if ev.Setup != nil { // the setup's own steps; the start's are on H15
		chain = r.chain(ev.Steps)
	}
	total := len(setupPhases)
	body := []string{
		r.tiles(
			r.heroL("11", "", fmt.Sprint(n.pct), yeB, "%", txt.Render(r.T("of setup"))),
			r.sm("8", "LEFT", hiB.Render(left), func(w int) string { return bar(lp.Frac, w, ye) }, ""),
			r.sm("8", "DISK", r.svu(sp.DiskMBs, f0, " MB/s"), noGauge, ""),
			r.sm("8", "STEP", hiB.Render(fmt.Sprint(n.ph+1))+dim.Render(fmt.Sprintf(" %s %d", r.T("of"), total)),
				func(w int) string { return bar(float64(n.ph)/float64(total), w, ye) }, r.T(setupPhases[n.ph])),
		),
		"",
		chain,
		"",
	}
	body = append(body, m.setupRows(n)...)
	body = append(body,
		"",
		r.alert("✓", gr, r.T("setup keeps going if you switch pages")),
		r.alert("▲", ye, r.T("q twice or closing the window cuts setup off; it starts over next time")),
	)
	return m.page(plServer, r.pageHead(r.T("first-run setup")+" · "+m.runName(), ""), body, foot{next: []string{
		"then verify → speed check → the server starts by itself",
	}, here: []string{"ctrl+x", "cancel setup"}})
}

// vH7: the server did not start.
func (m *Model) vH7() string {
	r := m.r
	f := m.failure
	if f == nil {
		f = &Failure{}
	}
	a := r.cw()
	// UX-R9 (finding 15 (b)): the head names the failure once («서버 가동 실패»); the reason box
	// holds the reason whole, wrapped, never cut and never repeated; the fix box its keys; below
	// them what the page had room for: the run's settings, the start steps reached and the last
	// log lines in boxes, as the serve page holds them.
	what := f.What
	switch what {
	case "", "did not start", "the server did not start":
		what = "server start failed"
		if f.Stage == "serving" {
			what = "stopped while serving"
		}
	}
	what = r.T(what)
	why := r.reason(f.Why) // the launcher's own reasons; a reason the backend words stays its text
	if why == "" && f.Stage == "serving" {
		why = fmt.Sprintf(r.T("exit code %d"), f.ExitCode)
	}
	var fix, keys []string
	for _, x := range f.Fixes {
		fix = append(fix, cyB.Render(x.Key)+txt.Render("  "+r.T(x.Label)))
		keys = append(keys, x.Key, x.Label)
	}
	fix = append(fix, cyB.Render("r")+txt.Render("  "+r.T("try again")))
	// the reason box is as wide as the page, so the reason takes the fewest lines; the keys that
	// fix it follow in their own box, on one line when they fit
	var whyRows []string
	if why == "" {
		whyRows = append(whyRows, dim.Render(r.T("details on 4 Log")))
	}
	for _, l := range wrapWords(why, max(4, a-4)) {
		if l != "" {
			whyRows = append(whyRows, hiB.Render(l))
		}
	}
	if f.Holder != "" || f.PID > 0 {
		h := f.Holder
		if f.PID > 0 && h != "" {
			h += " · "
		}
		if f.PID > 0 {
			h += fmt.Sprintf("PID %d", f.PID)
		}
		whyRows = append(whyRows, txt.Render(h))
	}
	if one := strings.Join(fix, "     "); lipgloss.Width(one) <= a-4 {
		fix = []string{one}
	}
	top := lines(indent(r.panel(a, "9", rdB.Render("✕ "+r.T("WHY")), whyRows...), 1),
		indent(r.panel(a, "14", cyB.Render(r.T("FIX")), fix...), 1))
	body := []string{top}
	// the two failures with the user's sentence (conflicts 8 and 18-1)
	switch {
	case f.VRAMOut:
		body = append(body, "", r.alert("✕", rd, r.T("start failed: VRAM ran out · close other GPU programs or lower ctx · details on 4 Log")))
	case f.SmokeFailed:
		body = append(body, "", r.alert("✕", rd, r.T("check run failed")+" · "+r.vs(f.Why)+" · "+r.T("details on 4 Log")))
	}
	rest := []string{"", r.alert("✓", gr, r.T("nothing lost · setup and settings are kept"))}
	if r.wide() {
		rest = append(rest, r.alert("·", txt, r.T("y copies a report: model, settings, memory, last errors")))
	}
	// the log tail (U-31: it leaves after the graph), in its box
	if len(f.LogTail) > 0 || f.LogFile != "" {
		var ll []string
		for _, l := range f.LogTail {
			ll = append(ll, dim.Render(l))
		}
		if f.LogFile != "" {
			lab := r.T("full log") + "  "
			ll = append(ll, dim.Render(lab)+txt.Render(midPath(f.LogFile, max(4, a-4-lipgloss.Width(lab)))))
		}
		rest = append(rest, logPart(""), logPart(indent(r.panel(a, "8", dim.Render(r.T("LAST LOG LINES")), ll...), 1)))
	}
	// the fill: the run's settings (the serve page's «this run» frame) and the start steps, how
	// far the start came, each while the window has room for it whole; in a small window they
	// leave first, as the graph does (U-31)
	var fill []string
	if m.runWords() != "" && m.runWords() != r.np() {
		fill = append(fill, graphPart(m.runFrame("", "")))
	}
	if m.startEv != nil && len(m.startEv.Steps) > 0 {
		fill = append(fill, graphPart(indent(r.panel(a, "8", dim.Render(r.T("START STEPS")), m.stepRows(m.startEv.Steps, 18, nil)...), 1)))
	}
	height := func(xs []string) int {
		n := 0
		for _, x := range xs {
			n += strings.Count(x, "\n") + 1
		}
		return n
	}
	room := r.H - 7 // the top row and its rule, the head and a blank line, the three footer lines
	for _, x := range fill {
		if r.W < fullW || r.H < fullH || height(body)+height([]string{x})+height(rest) <= room {
			body = append(body, x)
		}
	}
	body = append(body, rest...)
	at := ""
	if !f.At.IsZero() {
		at = f.At.Format("15:04:05")
	}
	keys = append(keys, "r", "try again")
	if r.wide() {
		keys = append(keys, "y", "copy report")
	}
	head := m.runName() + " · " + what
	return m.page(plServer, r.pageHead(head, at), body, foot{next: []string{
		"after any fix it starts again · setup is not redone",
	}, here: keys})
}

// spareTile is the FREE tile of 2 Server in three lines (review item 14):
// RAM bar with what is left, VRAM bar with what is left, the colour key.
// The RAM bar is the whole PC's RAM; the VRAM bar is the card's use with
// this server's share inside it (two sources, never added; review item 4).
func (m *Model) spareTile(s *ServeStatus, edge string) tileFn {
	r := m.r
	return func(w int) string {
		in := w - 4
		left := func(v Val) string {
			if v.Ok() {
				return r.Tf("%s left", gstr(v.V))
			}
			return r.svt(v, gstr)
		}
		vfree := s.FreeVRAMGB
		if !vfree.Ok() && s.TotalVRAMGB.Ok() && s.UsedVRAMGB.Ok() {
			vfree = K(s.TotalVRAMGB.V - s.UsedVRAMGB.V)
		}
		rt, vt := left(s.FreeRAMGB), left(vfree)
		tw := max(lipgloss.Width(rt), lipgloss.Width(vt))
		if in-5-1-tw < 3 { // a narrow tile: the number alone, so a bar stays
			rt, vt = r.svt(s.FreeRAMGB, gstr), r.svt(vfree, gstr)
			tw = max(lipgloss.Width(rt), lipgloss.Width(vt))
		}
		bw := max(0, in-5-1-tw)
		ramBar := dim.Render(rep("·", bw))
		if s.TotalRAMGB.Ok() && s.TotalRAMGB.V > 0 {
			sv, o := 0.0, 0.0
			if s.ModelRAMGB.Ok() {
				sv = s.ModelRAMGB.V
			}
			if s.OtherRAMGB.Ok() {
				o = s.OtherRAMGB.V
			}
			ramBar = stack(bw, seg{sv / s.TotalRAMGB.V, "█", cy}, seg{o / s.TotalRAMGB.V, "▓", txt})
		}
		vramBar := dim.Render(rep("·", bw))
		if s.TotalVRAMGB.Ok() && s.TotalVRAMGB.V > 0 && s.UsedVRAMGB.Ok() {
			sv := 0.0
			if s.ModelVRAMGB.Ok() {
				sv = math.Min(s.ModelVRAMGB.V, s.UsedVRAMGB.V)
			}
			// UX-R16b: the VRAM bar in the VRAM colours of 3 Memory (UX-R16), never the RAM's
			vramBar = stack(bw, seg{sv / s.TotalVRAMGB.V, "█", vSrv}, seg{(s.UsedVRAMGB.V - sv) / s.TotalVRAMGB.V, "▓", vOth})
		}
		l1 := txt.Render(pad("RAM", 5)) + ramBar + " " + hi.Render(rt)
		l2 := txt.Render(pad("VRAM", 5)) + vramBar + " " + hi.Render(vt)
		// the colour key, the first form that fits the tile; each glyph twice, the RAM colour then
		// the VRAM colour (UX-R16b)
		var l3 string
		for _, k := range [][3]string{{"this server", "  ", "other programs"}, {"this server", "  ", "other"}, {"this server", " ", "other"}, {"server", " ", "other"}, {"this", " ", "other"}} {
			l3 = cy.Render("█") + vSrv.Render("█") + dim.Render(" "+r.T(k[0])+k[1]) + txt.Render("▓") + vOth.Render("▓") + dim.Render(" "+r.T(k[2]))
			if lipgloss.Width(l3) <= in {
				break
			}
		}
		if lipgloss.Width(l3) > in { // the narrowest key: each glyph pair set close to its word
			for _, k := range [][2]string{{"this", "other"}, {"server", "other"}} {
				l3 = cy.Render("█") + vSrv.Render("█") + dim.Render(r.T(k[0])+" ") + txt.Render("▓") + vOth.Render("▓") + dim.Render(r.T(k[1]))
				if lipgloss.Width(l3) <= in {
					break
				}
			}
		}
		return r.tile(w, edge, "FREE", l1, l2, l3)
	}
}

// decodeSeries is what the decode graph plots (UX-R11): the live five-second readings, and a
// finished request's whole-answer rate at its end when no live reading fell between the request
// before it and its end (an answer shorter than two status beats), so a finished request is
// never missing from the graph.
func (m *Model) decodeSeries() []sample {
	var add []sample
	i := 0
	var prev time.Time
	for _, q := range m.reqSamples {
		for i < len(m.samples) && !m.samples[i].at.After(prev) {
			i++
		}
		if q.DecodeTokS > 0 && (i >= len(m.samples) || m.samples[i].at.After(q.At)) {
			add = append(add, sample{q.At, q.DecodeTokS})
		}
		prev = q.At
	}
	if len(add) == 0 {
		return m.samples
	}
	out := append(append(make([]sample, 0, len(m.samples)+len(add)), m.samples...), add...)
	sort.SliceStable(out, func(a, b int) bool { return out[a].at.Before(out[b].at) })
	return out
}

// graphMinutes is the span of the serve page's time graphs (UX-R17, review item 27): the last 60
// wall-clock minutes, one column a minute, the current minute rightmost.
const graphMinutes = 60

// minuteValue is the value of one minute's column (UX-R17): the mean of the minute's samples.
// Whether a minute shows its mean or its highest sample is the user's open question (review
// item 27); this function is the one place that answers it.
func minuteValue(vs []float64) float64 {
	s := 0.0
	for _, v := range vs {
		s += v
	}
	return s / float64(len(vs))
}

// minuteSamples sorts the samples into the 60 wall-clock minutes that end with now's minute:
// [0] the oldest, [59] the current one (a sample of this minute counts at once). A sample of an
// older or a later minute is left out.
func minuteSamples(ss []sample, now time.Time) [][]float64 {
	last := now.Truncate(time.Minute)
	out := make([][]float64, graphMinutes)
	for _, s := range ss {
		age := int(last.Sub(s.at.Truncate(time.Minute)) / time.Minute)
		if age < 0 || age >= graphMinutes {
			continue
		}
		out[graphMinutes-1-age] = append(out[graphMinutes-1-age], s.v)
	}
	return out
}

// minuteCells lays the 60 minutes on w cells (UX-R17): from 60 cells on, each minute is w/60
// cells wide and the window stands at the right edge, the cells left of it blank; below 60
// cells minute k goes to cell k*w/60, so the current minute always holds the last cell. A cell
// is minuteValue of its minutes' samples, NaN without one (an empty column). at[k] is the first
// cell of minute k.
func minuteCells(per [][]float64, w int) (vals []float64, at []int) {
	at = make([]int, graphMinutes)
	bw := w / graphMinutes
	for k := range at {
		if bw > 0 {
			at[k] = w - graphMinutes*bw + k*bw
		} else {
			at[k] = k * w / graphMinutes
		}
	}
	cell := make([][]float64, w)
	for k, vs := range per {
		for j := 0; j < max(bw, 1); j++ {
			cell[at[k]+j] = append(cell[at[k]+j], vs...)
		}
	}
	vals = make([]float64, w)
	for c, vs := range cell {
		vals[c] = math.NaN()
		if len(vs) > 0 {
			vals[c] = minuteValue(vs)
		}
	}
	return vals, at
}

// minuteGraph draws a serve page time graph (UX-R17, review item 27): the 60 minutes that end
// with now's minute, a column a minute (minuteCells), a scale from 0 to the highest minute,
// three rows of columns, and an axis line. On the axis each clock time stands under its own
// minute's column - the oldest minute's at the left and the quarter hours' where they fit - and
// «now» at the right end (UX-R27: the average and lowest went to the panels beside); so the
// columns and their times move one column left as the minute turns, and the minute older than
// 60 leaves.
func (r rc) minuteGraph(ss []sample, now time.Time, w int, sty lipgloss.Style, right string) []string {
	per := minuteSamples(ss, now)
	top := 0.0
	for _, vs := range per {
		if len(vs) > 0 {
			top = math.Max(top, minuteValue(vs))
		}
	}
	topS := f1(top)
	sw := lipgloss.Width(topS) + 1
	cols := max(1, w-sw)
	b, at := minuteCells(per, cols)
	vals := make([]float64, cols)
	for i, v := range b {
		if !math.IsNaN(v) && top > 0 {
			vals[i] = v / top
		}
	}
	g := tall(vals, 3, sty)
	out := []string{
		dim.Render(lpad(topS, sw-1)+"┤") + g[0],
		dim.Render(rep(" ", sw-1)+"│") + g[1],
		dim.Render(lpad("0", sw-1)+"┤") + g[2],
	}
	// the axis labels: each at its cell, kept one blank cell apart, in this order of precedence
	type label struct {
		at  int
		s   string
		sty lipgloss.Style
	}
	var ls []label
	put := func(a int, s string, sty lipgloss.Style) bool {
		e := a + lipgloss.Width(s)
		if a < 0 || e > cols {
			return false
		}
		for _, l := range ls {
			if a <= l.at+lipgloss.Width(l.s) && l.at <= e {
				return false
			}
		}
		ls = append(ls, label{a, s, sty})
		return true
	}
	first := now.Truncate(time.Minute).Add(-(graphMinutes - 1) * time.Minute)
	put(cols-lipgloss.Width(right), right, dim)
	put(at[0], first.Format("15:04"), dim)
	for k := 1; k < graphMinutes; k++ {
		if t := first.Add(time.Duration(k) * time.Minute); t.Minute()%15 == 0 {
			put(at[k], t.Format("15:04"), dim)
		}
	}
	sort.Slice(ls, func(a, b int) bool { return ls[a].at < ls[b].at })
	axis, p := rep(" ", sw), 0
	for _, l := range ls {
		axis += rep(" ", l.at-p) + l.sty.Render(l.s)
		p = l.at + lipgloss.Width(l.s)
	}
	return append(out, axis)
}

// liveGraph is the decode graph of the serve page: the 60 wall-clock minutes up to now, a column
// a minute (UX-R17, review item 27). Before, the window ran from the first sample when the server
// had run less than an hour (review item 21), so every column's span grew with the run.
func (m *Model) liveGraph(ss []sample, w int, sty lipgloss.Style) []string {
	return m.r.minuteGraph(ss, m.now(), w, sty, m.r.T("now"))
}

// reqGraph is the prefill graph (GL-18, LFS Q3.64; shown with the decode graph since UX-R19):
// each finished request's prompt rate at its end time, in the same minute columns, scale and
// axis line as the decode graph (UX-R17).
func (m *Model) reqGraph(w int) []string {
	ss := make([]sample, 0, len(m.reqSamples))
	for _, q := range m.reqSamples {
		if q.PrefillTokS > 0 {
			ss = append(ss, sample{q.At, q.PrefillTokS})
		}
	}
	return m.r.minuteGraph(ss, m.now(), w, cy, m.r.T("now"))
}

// heroServe is the big tile through one request (HF): no request (grey, the
// last reply), prefill (cyan, share of the prompt read), decode (green,
// live), slow (yellow, against the usual).
func (m *Model) heroServe(s *ServeStatus) tileFn {
	r := m.r
	slow := s.Slow || (s.UsualTokS.Ok() && s.DecodeTokS.Ok() && s.Phase == ReqDecode && s.DecodeTokS.V < 0.8*s.UsualTokS.V)
	switch {
	case s.Phase == ReqPrefill:
		pc := ""
		if s.PrefillDone.Ok() && s.PrefillTotal.Ok() && s.PrefillTotal.V > 0 {
			pc = fmt.Sprint(int(s.PrefillDone.V / s.PrefillTotal.V * 100))
		}
		mark := ""
		if r.wide() && s.PrefillTokS.Ok() && s.PrefillDone.Ok() && s.PrefillTotal.Ok() && s.PrefillTokS.V > 0 {
			mark = fmt.Sprintf("TTFT ~%.0f s", (s.PrefillTotal.V-s.PrefillDone.V)/s.PrefillTokS.V)
		}
		return r.heroL("14", "PREFILL", pc, cyB, "%", txt.Render(mark))
	case slow:
		v := s.DecodeTokS
		if s.Phase != ReqDecode {
			v = s.LastDecodeTokS
		}
		return r.heroL("11", "DECODE", numOr(v), yeB, "tok/s", txt.Render(r.T("usual")+" "+num(s.UsualTokS, "%.1f")))
	case s.Phase == ReqDecode:
		mark := ""
		if r.wide() {
			mark = r.T("live · 5 s average")
		}
		return r.heroL("10", "DECODE", numOr(s.DecodeTokS), grB, "tok/s", txt.Render(mark))
	}
	return r.heroL("8", "DECODE", numOr(s.LastDecodeTokS), dim, "tok/s", dim.Render(r.T("last reply")))
}

func numOr(v Val) string {
	if !v.Ok() {
		return ""
	}
	return f1(v.V)
}

// serveBody is the body of H8, H9 and H8Q.
func (m *Model) serveBody() []string {
	r := m.r
	s := m.status
	if s == nil {
		s = &ServeStatus{}
	}
	ctxVal := hiB.Render(tok(s.CtxUsed)) + dim.Render(r.Tf(" of %s", tok(s.CtxTotal)))
	ctxNote := ""
	if s.CtxUsed.Ok() && s.CtxTotal.Ok() {
		ctxNote = r.Tf("%s left", tok(K(s.CtxTotal.V-s.CtxUsed.V)))
	} else if !s.CtxTotal.Ok() && showNotPorted { // UX-R9 addendum (1)
		ctxNote = r.np()
	}
	ctxT := r.sm("8", "CONTEXT", ctxVal, func(w int) string {
		if s.CtxUsed.Ok() && s.CtxTotal.Ok() && s.CtxTotal.V > 0 {
			return use(s.CtxUsed.V/s.CtxTotal.V, w)
		}
		return ""
	}, ctxNote)
	// UX-R26 (review item 35 (b), the user's answer of 19:13 KST and item 37): the third tile is
	// the processing tile; the prefill rate and TTFT are in the prefill box and the request box
	ttftT := m.procTile(s)
	if s.Slow {
		ttftT = r.sm("11", "LAST DECODE", yeB.Render(num(s.LastDecodeTokS, "%.1f"))+dim.Render(" tok/s"), noGauge, r.T("usual")+" "+num(s.UsualTokS, "%.1f"))
	}
	freeEdge := "8"
	if s.Slow {
		freeEdge = "9"
	}
	tiles := r.tiles(m.heroServe(s), ctxT, ttftT, m.spareTile(s, freeEdge))

	sty := gr
	if s.Slow {
		sty = ye
	}
	// UX-R19 (review item 24 (d), answer 5; sub-lead decision option C): no graph switch. From 110
	// columns the decode graph stands beside the request box and, where the height leaves room,
	// the prefill graph beside the prefill box under it; each title names what it shows.
	var mid, low, lone string
	if r.wide() {
		c := r.cols(3)
		gwid := c[0] + c[1]
		ds := m.decodeSeries()
		g := m.liveGraph(ds, max(1, gwid-4), sty)
		if len(ds) == 0 {
			g[3] = dim.Render(r.T("samples show when the model answers"))
		}
		mid = graphPart(row(r.panel(gwid, "8", dim.Render(r.T("DECODE · tok/s · last 60 min")), g...), m.decodeStats(c[2], s, true)))
		pg := m.reqGraph(max(1, gwid-4))
		if len(m.reqSamples) == 0 {
			pg[3] = dim.Render(r.T("samples show when the model answers"))
		}
		low = graphPart(row(r.panel(gwid, "8", dim.Render(r.T("PREFILL · tok/s per request")), pg...), m.prefillStats(c[2], s, true)))
		// where the prefill row has no room, the PREFILL panel stays alone, short, under the
		// DECODE panel (the sub-lead's message after UX-R26: prefill and TTFT live there)
		lone = indent(lipgloss.JoinHorizontal(lipgloss.Top, blockOf(gwid, 5), m.prefillStats(c[2], s, false)), 1)
	}
	// UX-R27: below 110 columns, or where even the short PREFILL panel has no room under the
	// decode graph, the two panels stand side by side without graphs
	h := r.cw() / 2
	pair := row(m.decodeStats(h, s, true), m.prefillStats(r.cw()-h, s, true))
	if !r.wide() {
		mid = pair
	}
	var al string
	switch {
	case s.Slow:
		n := s.SlowNote
		if n == "" {
			n = r.T("slower than usual · cause not known")
		}
		al = r.alert("▲", ye, n)
	case s.Requests.Ok():
		w := r.T("none waiting")
		if s.Waiting.Ok() && s.Waiting.V > 0 {
			w = fmt.Sprintf("%.0f %s", s.Waiting.V, r.T("waiting"))
		} else if !s.Waiting.Ok() {
			w = r.T("waiting") + ": " + r.np()
		}
		e := num(s.Errors, "%.0f")
		if s.Waiting.Ok() && s.Waiting.V == 0 {
			al = r.alert("✓", gr, r.Tf("%s requests done · %s errors · none waiting", f0(s.Requests.V), e))
		} else if !s.Waiting.Ok() && !showNotPorted { // UX-R4 (finding 20 (c)): no «waiting: not ported»
			al = r.alert("✓", gr, fmt.Sprintf("%s %s · %s %s", f0(s.Requests.V), r.T("requests done"), e, r.T("errors")))
		} else {
			al = r.alert("✓", gr, fmt.Sprintf("%s %s · %s %s · %s", f0(s.Requests.V), r.T("requests done"), e, r.T("errors"), w))
		}
	case showNotPorted: // UX-R9 addendum (1): before the first count, no line
		al = r.alert("·", txt, r.T("request and error counts")+": "+r.np())
	}
	health := s.Health
	if health == "" {
		health = "—"
	}
	pid := ""
	if s.ChildPID > 0 {
		pid = fmt.Sprintf(" · PID %d", s.ChildPID)
	}
	hl := "  " + dim.Render("/health ") + txt.Render(health) + dim.Render(pid)
	addr, alias := "", ""
	if s.Address != "" {
		addr = s.Address
	} else if showNotPorted { // UX-R9 addendum (1): without an address the line is left out
		addr = r.np()
	}
	alias = s.Alias
	body := []string{tiles, m.runFrame(addr, alias), mid, "", al, hl}
	// the prefill row only where the whole page keeps its height with it (the body's room is the
	// window less the nav, head, blank and footer lines): where it does not, it leaves first, as
	// one part, before the U-31 order takes the graph
	if low != "" {
		n := 0
		for _, x := range body {
			n += strings.Count(x, "\n") + 1
		}
		switch room := r.H - 7 - n; {
		case strings.Count(low, "\n")+1 <= room:
			body = append(body[:3], append([]string{low}, body[3:]...)...)
		case strings.Count(lone, "\n")+1 <= room:
			body = append(body[:3], append([]string{lone}, body[3:]...)...)
		default:
			body[2] = pair
		}
	}
	return body
}

// blockOf is a blank block of w cells and n lines, the room left of a lone panel.
func blockOf(w, n int) string {
	return strings.TrimSuffix(strings.Repeat(rep(" ", w)+"\n", n), "\n")
}

// UX-R27 (review item 37, the user's decision of 19:19 KST): the two statistics panels in one
// shape, the game-FPS way: now, avg, min and max over the graphs' window (the 60 wall-clock
// minutes of minuteSamples). now is this request's live reading while its phase runs, «—»
// otherwise, so the panels need no title of their own for idle. The column names now, avg,
// min, max stay English in both languages (as decode, prefill, TTFT). Sample basis: DECODE over
// the decode graph's samples (the five-second live readings and the rates of answers too short
// for one, decodeSeries); PREFILL and TTFT per finished request (GL-18's store), as the prefill
// graph plots.

// statRow is one line of a statistics panel: what it measures, now, avg, min, max, and its unit.
type statRow struct {
	name string
	v    [4]string
	unit string
}

// windowStats are avg, min and max of the samples in the graphs' window, «—» without one.
func windowStats(ss []sample, now time.Time, f func(float64) string) (avg, lo, hi string) {
	sum, n, l, h := 0.0, 0, math.Inf(1), math.Inf(-1)
	for _, vs := range minuteSamples(ss, now) {
		for _, v := range vs {
			sum, n, l, h = sum+v, n+1, math.Min(l, v), math.Max(h, v)
		}
	}
	if n == 0 {
		return "—", "—", "—"
	}
	return f(sum / float64(n)), f(l), f(h)
}

// statsPanel draws a statistics panel: a head line with the column names, then a line per row,
// its unit at the end where the row has room; the columns share the width up to 8 cells each.
// tall pads it to the height of the graph beside it.
func (m *Model) statsPanel(w int, title string, rows []statRow, tall bool) string {
	r := m.r
	nw, cw, uw := 0, 4, 0
	for _, x := range rows {
		nw = max(nw, lipgloss.Width(r.T(x.name)))
		uw = max(uw, lipgloss.Width(x.unit))
		for _, v := range x.v {
			cw = max(cw, lipgloss.Width(v)+1)
		}
	}
	in := w - 4
	cw = max(cw, min(8, (in-nw-1-uw)/4)) // wider columns, never at the units' cost
	head := rep(" ", nw+1)
	for _, k := range []string{"now", "avg", "min", "max"} {
		head += pad(k, cw)
	}
	ls := []string{dim.Render(head)}
	for _, x := range rows {
		l := txt.Render(pad(r.T(x.name), nw+1)) + hiB.Render(pad(x.v[0], cw))
		for _, v := range x.v[1:3] {
			l += hi.Render(pad(v, cw))
		}
		// the unit follows max where the row has room for it, else the row goes without
		lw, xu := lipgloss.Width(x.v[3]), lipgloss.Width(x.unit)
		switch {
		case nw+1+4*cw+xu <= in:
			l += hi.Render(pad(x.v[3], cw)) + dim.Render(x.unit)
		case nw+1+3*cw+lw+1+xu <= in:
			l += hi.Render(x.v[3]+" ") + dim.Render(x.unit)
		default:
			l += hi.Render(x.v[3])
		}
		ls = append(ls, l)
	}
	for tall && len(ls) < 4 {
		ls = append(ls, "")
	}
	return r.panel(w, "8", cyB.Render(r.T(title)), ls...)
}

// decodeStats is the DECODE panel: tok/s now, avg, min, max.
func (m *Model) decodeStats(w int, s *ServeStatus, tall bool) string {
	now := "—"
	if s.Phase == ReqDecode && s.DecodeTokS.Ok() {
		now = f1(s.DecodeTokS.V)
	}
	a, l, h := windowStats(m.decodeSeries(), m.now(), f1)
	return m.statsPanel(w, "DECODE · last 60 min", []statRow{{"decode", [4]string{now, a, l, h}, "tok/s"}}, tall)
}

// fr is a prompt rate in the PREFILL panel: one decimal below 100 tok/s, whole from 100 on, so
// the row keeps its unit at 80 columns.
func fr(f float64) string {
	if f >= 100 {
		return f0(f)
	}
	return f1(f)
}

// prefillStats is the PREFILL panel: the prompt rate and TTFT, now, avg, min, max. While a
// prompt is read, TTFT now is the estimate of the time left to the first token («~n», whole
// seconds, as the PREFILL tile shows it).
func (m *Model) prefillStats(w int, s *ServeStatus, tall bool) string {
	var pre, ttft []sample
	for _, q := range m.reqSamples {
		if q.PrefillTokS > 0 {
			pre = append(pre, sample{q.At, q.PrefillTokS})
		}
		if q.TTFTs > 0 {
			ttft = append(ttft, sample{q.At, q.TTFTs})
		}
	}
	pn, tn := "—", "—"
	if s.Phase == ReqPrefill && s.PrefillTokS.Ok() {
		pn = fr(s.PrefillTokS.V)
		if s.PrefillDone.Ok() && s.PrefillTotal.Ok() && s.PrefillTokS.V > 0 {
			tn = "~" + f0((s.PrefillTotal.V-s.PrefillDone.V)/s.PrefillTokS.V)
		}
	}
	pa, pl, ph := windowStats(pre, m.now(), fr)
	ta, tl, th := windowStats(ttft, m.now(), f1)
	return m.statsPanel(w, "PREFILL · last 60 min", []statRow{
		{"prefill", [4]string{pn, pa, pl, ph}, "tok/s"},
		{"TTFT", [4]string{tn, ta, tl, th}, "s"},
	}, tall)
}

// procTile is the top row's third tile (UX-R26): this request's input tokens with the part
// reused from the KV cache, the generated tokens (all three from /slots: n_prompt_tokens,
// n_prompt_tokens_cache, next_token n_decoded; wire/map.go statusFrom), and the batch size the
// server was started with (-b of the started argv; wire/wire.go batchOf), on a row of its own so
// a second batch figure (-ub) has room. Idle, the figures stay the last request's, as /slots
// holds them; a figure not read is «—». Where the input row does not fit, reused gets a row of
// its own and the batch goes beside the generated tokens where that fits.
func (m *Model) procTile(s *ServeStatus) tileFn {
	r := m.r
	in := U
	if s.ReusedTokens.Ok() && s.NewTokens.Ok() {
		in = K(s.ReusedTokens.V + s.NewTokens.V)
	}
	lv := func(label, v string) string { return txt.Render(r.T(label)+" ") + hiB.Render(v) }
	return func(w int) string {
		label := "PROCESSING · this request"
		if lipgloss.Width(r.T(label))+2 > w-3 {
			label = "PROCESSING"
		}
		iw := max(0, w-4)
		batch := lv("batch", num(s.Batch, "%.0f"))
		r0 := lv("in", tok(in)) + dim.Render(" (") + lv("reused", tok(s.ReusedTokens)) + dim.Render(")")
		r1, r2 := lv("out", tok(s.DecodedTokens)), batch
		if lipgloss.Width(r0) > iw {
			r0, r1, r2 = lv("in", tok(in)), lv("reused", tok(s.ReusedTokens)), lv("out", tok(s.DecodedTokens))
			if x := r2 + dim.Render(" · ") + batch; lipgloss.Width(x) <= iw {
				r2 = x
			}
		}
		return r.tile(w, "8", label, r0, r1, r2)
	}
}

// vServe is H8, H9 (slow) and H8Q (q pressed once).
func (m *Model) vServe() string {
	r := m.r
	s := m.status
	up := ""
	if s != nil && s.Uptime > 0 {
		up = r.Tf("up %s", upStr(r, s.Uptime))
	}
	head := r.Tf("%s is serving", m.runName())
	if s != nil && s.Slow {
		head = fmt.Sprintf("%s %s", m.runName(), r.T("is serving, slower than usual"))
	}
	var f foot
	switch {
	case m.quitArmed:
		f = foot{next: []string{"q again: unload the model and exit · other keys cancel"}, warn: true,
			here: []string{"q", "again to exit", "esc", "cancel"}}
	case s != nil && s.Slow:
		f = foot{next: []string{"3 Memory shows what changed around the slowdown"},
			here: []string{"ctrl+f", "unload", "c", "copy address"}}
	default:
		f = foot{next: []string{"ctrl+f unloads the model and shows a summary"},
			here: []string{"ctrl+f", "unload", "c", "copy address"}}
	}
	return m.page(plServer, r.pageHead(head, up), m.serveBody(), f)
}

func upStr(r rc, d time.Duration) string {
	h, mi := int(d.Hours()), int(d.Minutes())%60
	if r.lang == LangKorean {
		if h > 0 {
			return fmt.Sprintf("%d시간 %02d분", h, mi)
		}
		return fmt.Sprintf("%d분", mi)
	}
	if h > 0 {
		return fmt.Sprintf("%dh %02dm", h, mi)
	}
	return fmt.Sprintf("%dm", mi)
}

// vHQ: the launcher is stopping the server (q twice: then it closes;
// ctrl+f: it stays and shows the summary). Decision ①: no "cut the reply"
// step. The cache-save bar follows decision ③ (GL-15: the warm-start save; a stop that
// stores nothing shows it skipped, a failed save shows it failed).
func (m *Model) vHQ() string {
	r := m.r
	ev := m.stopEv
	if ev == nil {
		ev = &StopEvent{Mode: m.stopMode}
	}
	sp := saveBar(ev.Save, m.now().Sub(m.stopAt))
	ci, cname := curStep(ev.Steps)
	total := len(ev.Steps)
	label, title, head := "QUITTING", r.T("QUIT STEPS")+dim.Render(" · "+r.T("q twice")), r.T("closing: unload the model, then the launcher")
	switch m.stopMode {
	case StopUnload:
		label, title, head = "UNLOADING", r.T("UNLOAD STEPS")+dim.Render(" · ctrl+f"), r.T("unloading the model · the launcher stays")
	case StopCancel:
		label, title, head = "CANCELLING", r.T("CANCEL STEPS")+dim.Render(" · ctrl+x"), r.T("cancelling the start")
	case StopForce:
		label, title, head = "CLOSING", r.T("CLOSE STEPS")+dim.Render(" · "+r.T("ctrl+c twice")), r.T("closing now · no cache save")
	}
	var cacheT tileFn
	switch ev.Save.State {
	case StepNotPorted:
		cacheT = r.sm("8", "CACHE", dim.Render(r.T("save")+": "+r.snp()), noGauge, "") // UX-R12
	case StepSkipped:
		cacheT = r.sm("8", "CACHE", dim.Render(r.T("save")+": "+r.T("off")), noGauge, "")
	case StepFailed:
		cacheT = r.sm("8", "CACHE", rd.Render(r.T("save")+": "+r.T("failed")), noGauge, "")
	default:
		cacheT = r.sm("8", "CACHE", hiB.Render(tok(ev.Save.Tokens))+dim.Render(" "+r.T("tokens")), func(w int) string { return r.progressBarOnly(sp, w) }, "next start reuses it")
	}
	body := []string{
		r.tiles(
			r.heroL("11", label, fmt.Sprint(int(m.stopElapsed().Seconds())), yeB, "s", ""),
			r.sm("8", "STEP", hiB.Render(fmt.Sprint(min(ci+1, total)))+dim.Render(fmt.Sprintf(" %s %d", r.T("of"), total)),
				func(w int) string {
					if total == 0 {
						return ""
					}
					return bar(float64(ci)/float64(total), w, ye)
				}, cname),
			cacheT,
			r.sm("8", "RAM", r.svu(ev.ModelRAMGB, gstr, " "+r.T("model")), noGauge, "freed at stop"),
		),
		"",
	}
	rows := m.stepRows(ev.Steps, 18, func() (string, string, func(int) string) {
		right := fmt.Sprintf("%d s", int(m.stopElapsed().Seconds()))
		if ev.Save.State == StepNotPorted || ev.Save.State == StepSkipped || ev.Save.State == StepFailed || !strings.Contains(strings.ToLower(curName(ev.Steps)), "cache") {
			return "", right, nil
		}
		tail := fmt.Sprintf(" %d %%", sp.Pct)
		switch {
		case sp.Ready:
		case sp.SecondsOnly:
			tail = " " + r.T("first time")
		case sp.Slower:
			tail = " 99 % · " + r.T("slower than last time")
		}
		return tail, right, func(w int) string { return r.progressBarOnly(sp, w) }
	})
	if len(rows) == 0 {
		rows = []string{dim.Render(r.T("waiting for the first step …"))}
	}
	body = append(body, indent(r.panel(r.cw(), "11", yeB.Render(title), rows...), 1), "")
	switch {
	case ev.Save.State == StepNotPorted && m.stopMode != StopCancel && showNotPorted:
		body = append(body, r.alert("▲", ye, r.T("warm-start save is not ported: the next start is cold")))
	case ev.Save.State == StepNotPorted && m.stopMode != StopCancel: // UX-R12: no save runs at this stop
		body = append(body, r.alert("▲", ye, r.T("no slot save at this stop: the next start is cold")))
	case ev.Save.State == StepFailed:
		body = append(body, r.alert("▲", ye, r.T("the slot save failed: the next start is cold")))
	case ev.Save.State == StepSkipped && m.stopMode != StopCancel:
		body = append(body, r.alert("▲", ye, r.T("no slot save at this stop: the next start is cold")))
	case m.stopMode == StopCancel:
		body = append(body, r.alert("✓", gr, r.T("nothing is lost · setup and settings are kept")))
	default:
		body = append(body, r.alert("✓", gr, r.T("nothing is lost on this path · the next start is warm")))
	}
	return m.page(plServer, r.pageHead(head, ""), body, foot{next: []string{
		"stuck? ctrl+c twice closes now · the cache is not saved",
	}, warn: true, here: []string{"ctrl+c", "twice: close now"}})
}

// vH12: stopped, the run summary.
func (m *Model) vH12() string {
	r := m.r
	res := m.result
	if res == nil {
		res = &StopResult{}
	}
	ru := res.Run
	up := "—"
	if !ru.Started.IsZero() && !ru.Ended.IsZero() {
		d := ru.Ended.Sub(ru.Started)
		up = fmt.Sprintf("%d:%02d", int(d.Hours()), int(d.Minutes())%60)
	}
	nt := func(a, b string) string {
		if r.cw() >= 100 {
			return a
		}
		return b
	}
	body := []string{
		r.tiles(
			r.heroL("7", "", up, hiB, "h:mm", txt.Render(r.T("up"))),
			r.sm("8", "DECODE", r.svu(ru.DecodeAvg, f1, " tok/s "+r.T("avg")), noGauge,
				nt(r.T("low")+" "+r.svt(ru.DecodeLow, f1), r.T("low")+" "+r.svt(ru.DecodeLow, f1))),
			r.sm("8", "PREFILL", r.svu(ru.PrefillAvg, f0, " tok/s "+r.T("avg")), noGauge,
				map[bool]string{true: tok(ru.PrefillTokens) + " " + r.T("read"), false: ""}[ru.PrefillTokens.Ok()]),
			r.sm("8", "TTFT", r.svu(ru.TTFTMedian, secs, " "+r.T("median")), noGauge,
				r.T("max")+" "+r.svt(ru.TTFTMax, secs)),
		),
		"",
	}
	n := max(1, r.cw()-4)
	from, to := "", ""
	if !ru.Started.IsZero() {
		from = ru.Started.Format("15:04")
	}
	if !ru.Ended.IsZero() {
		to = ru.Ended.Format("15:04")
	}
	// the whole run: from the start to the end of the run, or the samples' span; the samples
	// are the decode graph's, finished requests too short for a live reading included (UX-R11,
	// finding 20 (a), stopped page), and the prefill graph's (UX-R23), on one time axis
	ds := m.decodeSeries()
	var ps []sample
	for _, q := range m.reqSamples {
		if q.PrefillTokS > 0 {
			ps = append(ps, sample{q.At, q.PrefillTokS})
		}
	}
	t0, t1 := ru.Started, ru.Ended
	for _, ss := range [][]sample{ds, ps} {
		if len(ss) > 0 {
			if t0.IsZero() || t0.After(ss[0].at) {
				t0 = ss[0].at
			}
			if t1.IsZero() || t1.Before(ss[len(ss)-1].at) {
				t1 = ss[len(ss)-1].at
			}
		}
	}
	g := r.timeGraph(ds, t0, t1, n, gr, t0.Format("15:04"), t1.Format("15:04"))
	if len(ds) == 0 {
		g[3] = dim.Render(r.T("no decode samples in this run"))
	}
	body = append(body, graphPart(indent(r.panel(r.cw(), "8", dim.Render(r.T("DECODE · tok/s · whole run")), g...), 1)), "")
	// UX-R23 (review item 36): the prefill graph under the decode graph, each finished request's
	// prompt rate, in the serving screen's arrangement (UX-R19); it is placed below, where the
	// page keeps its height with it
	pg := r.timeGraph(ps, t0, t1, n, cy, t0.Format("15:04"), t1.Format("15:04"))
	if len(ps) == 0 {
		pg[3] = dim.Render(r.T("no prefill samples in this run"))
	}
	low := graphPart(indent(r.panel(r.cw(), "8", dim.Render(r.T("PREFILL · tok/s · whole run")), pg...), 1))
	at := len(body)
	switch ru.CacheTokens.State {
	case Known:
		body = append(body, r.alert("✓", gr, fmt.Sprintf("%s: %s %s · %s", r.T("prompt cache saved"), tok(ru.CacheTokens), r.T("tokens"), r.T("next start reuses it"))))
	default:
		// UX-R9 addendum (1): the run summary has no save count yet; the stop's save is in 4 Log
		note := r.T("details on 4 Log")
		if showNotPorted {
			note = r.np()
		}
		body = append(body, r.alert("·", txt, r.T("prompt cache save")+": "+note))
	}
	if res.Message != "" {
		body = append(body, r.alert("·", txt, res.Message))
	}
	// the prefill row only where the whole page keeps its height with it (as on the serving
	// screen, UX-R19): where it does not, it leaves first, before the U-31 order takes the graph
	nl := strings.Count(low, "\n") + 2
	for _, x := range body {
		nl += strings.Count(x, "\n") + 1
	}
	if nl <= r.H-7 {
		body = append(body[:at], append([]string{low, ""}, body[at:]...)...)
	}
	ctx := ""
	if from != "" {
		ctx = fmt.Sprintf("%s → %s · %s %s · %s %s", from, to, num(ru.Requests, "%.0f"), r.T("requests"), num(ru.Errors, "%.0f"), r.T("errors"))
	}
	if res.ExitCode != 0 || !res.Graceful {
		ctx = strings.TrimSpace(ctx + fmt.Sprintf(" · "+r.T("exit %d"), res.ExitCode))
	}
	return m.page(plServer, r.pageHead(r.T("stopped · run summary"), ctx), body, foot{next: []string{
		"⏎ starts the same model with the same settings · 1 Models to change",
	}, here: []string{"⏎", "start again"}})
}
