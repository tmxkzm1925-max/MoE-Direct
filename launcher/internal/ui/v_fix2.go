// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// FIX2-UI: the screen parts of the PowerShell-spec conflicts the user decided
// in reviews/_gol_conflicts_8_user_20261010.md (conflict number after each).

package ui

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// ---------- conflict 3: where the settings came from ----------

// provenance is the label of a model's settings: «default» (the catalog's),
// «my settings» (its ctrl+s profile), each with «· not saved» while the
// screen holds changes.
func (m *Model) provenance(model string) string {
	r := m.r
	l := r.T("default")
	if m.views[model].Saved {
		l = r.T("my settings")
	}
	if len(m.edits[model]) > 0 {
		l += " · " + r.T("not saved")
	}
	return l
}

// provenanceView draws the label, «not saved» in yellow.
func (m *Model) provenanceView(label string) string {
	base, rest, cut := strings.Cut(label, " · ")
	if !cut {
		return dim.Render(label)
	}
	return dim.Render(base) + ye.Render(" · "+rest)
}

// changedValues are the screen's changed values of a model, «label value»,
// in the order of its rows.
func (m *Model) changedValues(model string) []string {
	e := m.edits[model]
	var out []string
	seen := map[string]bool{}
	for _, x := range m.views[model].Rows {
		if v, ok := e[x.Key]; ok {
			out = append(out, m.r.T(x.Label)+" "+v)
			seen[x.Key] = true
		}
	}
	var rest []string
	for k, v := range e {
		if !seen[k] {
			rest = append(rest, k+" "+v)
		}
	}
	sort.Strings(rest)
	return append(out, rest...)
}

// runProvenanceToks are the end of the settings line of «this run»: the
// label kept at the start, «not saved» and the changed values in yellow.
func (m *Model) runProvenanceToks() []piece {
	if m.runLabel == "" {
		return nil
	}
	var ts []piece
	base, rest, cut := strings.Cut(m.runLabel, " · ")
	ts = append(ts, piece{base, dim})
	if cut {
		t := rest
		if len(m.runEdits) > 0 {
			t += " (" + strings.Join(m.runEdits, " · ") + ")"
		}
		ts = append(ts, piece{t, ye})
	}
	return ts
}

// piece is one styled piece of a line joined by « · ».
type piece struct {
	s   string
	sty lipgloss.Style
}

// wrapToks joins pieces by « · » into lines of at most w cells, breaking
// between pieces; a piece longer than a line is broken by words.
func wrapToks(ts []piece, w int) []string {
	var out []string
	cur, n := "", 0
	sep := dim.Render(" · ")
	for _, t := range ts {
		tw := lipgloss.Width(t.s)
		switch {
		case n == 0 && tw <= w:
			cur, n = t.sty.Render(t.s), tw
		case n > 0 && n+3+tw <= w:
			cur, n = cur+sep+t.sty.Render(t.s), n+3+tw
		default:
			if n > 0 {
				out = append(out, cur)
			}
			ls := wrapWords(t.s, w)
			for _, l := range ls[:len(ls)-1] {
				out = append(out, t.sty.Render(l))
			}
			last := ls[len(ls)-1]
			cur, n = t.sty.Render(last), lipgloss.Width(last)
		}
	}
	if n > 0 {
		out = append(out, cur)
	}
	return out
}

// ---------- conflict 10: custom settings ----------

// custom: a value the screen holds differs from its default.
func (m *Model) custom(model string) bool {
	for _, x := range m.views[model].Rows {
		if x.Default != "" && m.valueOf(model, x) != x.Default {
			return true
		}
	}
	return false
}

// neverRun: the settings the screen holds were not the last run's, so the
// last run's speed does not speak for them (the speed tile «not measured»).
func (m *Model) neverRun(model string) bool {
	v := m.views[model]
	if len(v.LastRun) == 0 {
		return m.custom(model)
	}
	for _, x := range v.Rows {
		if lv, ok := v.LastRun[x.Key]; ok && lv != m.valueOf(model, x) {
			return true
		}
	}
	return false
}

// ---------- conflict 16: the two modes in THIS MODEL's setup line ----------

// modesText: «packed ✓ · virtual needs repack»; short: «packed ✓ · virtual ▲».
func (m *Model) modesText(row ModelRow, short bool) string {
	r := m.r
	cur := m.curMode(row)
	var ps []string
	for _, x := range row.Modes {
		var s string
		var sty lipgloss.Style
		switch x.State {
		case ModeReady:
			s, sty = x.Mode+" ✓", gr
		case ModeNeedsRepack:
			s, sty = x.Mode+" "+r.T("needs repack"), ye
			if short {
				s = x.Mode + " ▲"
			}
		case ModeCheckFailed:
			s, sty = x.Mode+" ✕ "+r.T("check failed"), rd
			if short {
				s = x.Mode + " ✕"
			}
		case ModeStale: // W7 duty 12 (GL-10): the short form keeps the word (a bare ✕ reads «check failed»)
			s, sty = x.Mode+" ✕ "+r.T("stale (another file)"), rd
			if short {
				s = x.Mode + " ✕ " + r.T("stale")
			}
		default:
			s, sty = x.Mode+" "+r.np(), dim
		}
		if x.Mode == cur {
			sty = sty.Bold(true) // the mode now set (conflict 16)
		}
		ps = append(ps, sty.Render(s))
	}
	return strings.Join(ps, dim.Render(" · "))
}

// ---------- conflict 18-2: the launcher's own files ----------

func (m *Model) bundleBrokenText() string {
	return m.r.Tf("✕ the launcher files are damaged · %s differ · install again", fmt.Sprint(m.bundle.Mismatches))
}

// ---------- conflict 9: the plan box ----------

func (m *Model) planShown() bool {
	return m.plan != nil && m.ph != phStarting && m.ph != phServing && m.ph != phStopping
}

// planShort: the drive cannot hold what the setup writes.
func planShort(p RepackPlan) bool {
	return (p.WriteGB.Ok() && p.DriveFreeGB.Ok() && p.WriteGB.V > p.DriveFreeGB.V) || (p.LeftGB.Ok() && p.LeftGB.V < 0)
}

// kvWrap is a label and a value wrapped at w, the value's lines under it.
func kvWrap(w, lw int, label, value string, sty lipgloss.Style) []string {
	var out []string
	for i, l := range wrapWords(value, max(1, w-lw)) {
		head := rep(" ", lw)
		if i == 0 {
			head = dim.Render(pad(label, lw))
		}
		out = append(out, head+sty.Render(l))
	}
	return out
}

// vPlan: «setup needed · model · mode» with what it does, where it writes,
// the disk, the time and what follows; a drive too small shows the refusal
// line instead of the box (conflict 9).
func (m *Model) vPlan() string {
	r := m.r
	p := m.plan
	name := p.Model
	if row, ok := m.rowByID(p.Model); ok && row.Name != "" {
		name = row.Name
	}
	var body []string
	var f foot
	if planShort(*p) {
		body = append(body, r.alert("✕", rd, r.T("cannot start")+" · "+r.Tf("free %s < needed %s", r.vt(p.DriveFreeGB, gstr), r.vt(p.WriteGB, gstr))))
		f = foot{next: []string{"free room on the drive, then ⏎ on 1 Models · esc goes back"}, here: []string{"esc", "back"}}
	} else {
		w := r.cw() - 4
		// UX-R9 (finding 14): «what to do» for «does»; in Korean the labels are the English words
		// and the backend's sentence is read in Korean
		lw := 2 + max(lipgloss.Width(r.T("what to do")), lipgloss.Width(r.T("writes to")), lipgloss.Width(r.T("disk")), lipgloss.Width(r.T("time")))
		var rs []string
		rs = append(rs, kvWrap(w, lw, r.T("what to do"), r.does(p.Does), txt)...)
		rs = append(rs, dim.Render(pad(r.T("writes to"), lw))+txt.Render(midPath(r.vs(p.WritesTo), max(4, w-lw))))
		rs = append(rs, kvWrap(w, lw, r.T("disk"), r.Tf("about %s written · drive free %s → %s left",
			r.vt(p.WriteGB, gstr), r.vt(p.DriveFreeGB, gstr), r.vt(p.LeftGB, gstr)), txt)...)
		tm := r.T("first repack: not known yet")
		if p.LastS.Ok() {
			tm = r.Tf("last time %s", (time.Duration(p.LastS.V) * time.Second).Round(time.Second).String())
		}
		rs = append(rs, dim.Render(pad(r.T("time"), lw))+txt.Render(tm), "")
		rs = append(rs, kvWrap(w, 0, "", r.T("then: check → speed check → the server starts by itself"), cy)...)
		title := yeB.Render(r.T("SETUP NEEDED")) + txt.Render(" · "+name)
		if p.Mode != "" {
			title += txt.Render(" · " + p.Mode)
		}
		body = append(body, indent(r.panel(r.cw(), "11", title, rs...), 1))
		f = foot{next: []string{"⏎ runs the setup, then the server starts · esc goes back, nothing is written"},
			here: []string{"⏎", "set up and start", "esc", "back"}}
	}
	return m.page(plServer, r.pageHead(fmt.Sprintf("%s %s", name, r.T("needs a setup")), ""), body, f)
}

// does is the setup plan's «what to do» sentence in the screen language (UX-R9, finding 14):
// the backend joins its parts with « · »; each part is a table key, the count of earlier files
// removed (with the room it frees) read through the backend's own template.
func (r rc) does(s string) string {
	if s == "" {
		return r.vs(s)
	}
	const removes, freed = "first removes %d earlier file(s)", " freed, counted as free)"
	parts := strings.Split(s, " · ")
	for i, p := range parts {
		var n int
		if k, err := fmt.Sscanf(p, removes, &n); err == nil && k == 1 {
			out := fmt.Sprintf(r.T(removes), n)
			if _, room, ok := strings.Cut(p, " ("); ok && strings.HasSuffix(room, freed) {
				out += " (" + strings.TrimSuffix(room, freed) + r.T(freed)
			}
			parts[i] = out
			continue
		}
		parts[i] = r.T(p)
	}
	return strings.Join(parts, " · ")
}

// vs is a backend text, or "not ported" when it is empty.
func (r rc) vs(s string) string {
	if s == "" {
		return r.np()
	}
	return s
}

// ---------- conflict 6: the stop line in 4 Log ----------

// logLevel is the class a line is drawn and counted in: a stop line's
// code decides it (conflict 6).
func logLevel(x LogLine) LogLevel {
	if x.Stop == nil {
		return x.Level
	}
	switch x.Stop.Code {
	case 0:
		return LogOK
	case 2, 5:
		return LogWarn
	}
	return LogError
}

// logText is the text of a line; a stop line reads «server stopped · code
// n · meaning · server exit m».
func (r rc) logText(x LogLine) string {
	if x.Stop == nil {
		return x.Text
	}
	s := x.Stop
	mean := map[int]string{0: r.T("normal stop"), 2: r.T("stopped before ready"), 5: r.T("forced stop")}[s.Code]
	if mean == "" {
		mean = s.Meaning
		if mean == "" {
			mean = r.T("failed")
		}
	}
	l := r.Tf("server stopped · code %s · %s", fmt.Sprint(s.Code), mean)
	if s.ServerCode.Ok() {
		l += " · " + r.Tf("server exit %s", fmt.Sprintf("%.0f", s.ServerCode.V))
	}
	if x.Text != "" {
		l += " · " + x.Text
	}
	return l
}
