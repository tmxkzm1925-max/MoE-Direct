// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// 1 Models: H1 (pick and start), H2 (settings changed while serving), H3 (no
// models), H4 (a setup was cut off). From the mock's h.go h1-h4.

package ui

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// h1Label is the label of the H1 need (S8-LA-13; GL-05): "measured", "predicted before run"
// or none.
func (m *Model) h1Label(row ModelRow) string {
	if p := m.currentPreflight(row.ID); p != nil {
		return p.NeedLabel
	}
	return ""
}

// h1Nums are the three H1 numbers (decision ⑧): need, free RAM now, left
// after the start, from the preflight of the screen's settings when there is
// one, else from the catalog row.
func (m *Model) h1Nums(row ModelRow) (need, free, left, cache Val) {
	if p := m.currentPreflight(row.ID); p != nil {
		return p.NeedGB, p.AvailGB, p.LeftGB, p.CacheGB
	}
	return row.NeedGB, row.FreeRAMGB, row.LeftGB, row.CacheGB
}

// liveRAM is the RAM picture of 1 Models, read again at every draw from the
// memory reading (review item 13): the PC's RAM, what is available now, this
// model's need, and what is left after the start (available − need). While
// the chosen model serves, server is its share inside "in use".
func (m *Model) liveRAM(row ModelRow) (total, avail, need, left, server Val) {
	need, avail, left, _ = m.h1Nums(row)
	if p := m.currentPreflight(row.ID); p != nil {
		total = p.RAMTotalGB
	}
	if m.mem != nil {
		if m.mem.TotalRAMGB.Ok() {
			total = m.mem.TotalRAMGB
		}
		if m.mem.ReadyNowGB.Ok() {
			avail = m.mem.ReadyNowGB
			if need.Ok() {
				left = K(avail.V - need.V)
			}
		}
		if m.ph == phServing && row.ID == m.runModel {
			server = m.mem.ModelGB
		}
	}
	if server.Ok() {
		left = avail // the model is already in
	}
	return
}

// ramTile is the 1 Models RAM tile: one bar of the whole PC's RAM (in use ·
// this model · left after the start), each number with its name (review
// item 13; decision ⑧).
func (m *Model) ramTile(row ModelRow, edge string) tileFn {
	r := m.r
	total, avail, need, left, server := m.liveRAM(row)
	label := "RAM"
	if total.Ok() {
		label = "RAM · " + g0str(total.V)
	}
	inUse := Val{}
	if total.Ok() && avail.Ok() {
		inUse = K(math.Max(0, total.V-avail.V))
	}
	use := dim.Render(r.T("in use")+" ") + hi.Render(r.vt(inUse, f1))
	var val string
	var vals []string  // with the need's label: the first that fits the tile is drawn
	var notes []string // the first that fits the tile is drawn
	var g gauge = noGauge
	if server.Ok() {
		vals = []string{hiB.Render(gstr(server.V)) + dim.Render(" "+r.T("this server"))}
		val = hiB.Render(gstr(server.V)) + dim.Render(" "+r.T("server"))
		// UX-R22 (item 33): this server / other programs / left, no «in use» total beside this
		// server; where the tile is narrow, what is left stays
		other := gstr(math.Max(0, inUse.V-server.V))
		left := dim.Render(r.Tf("%s left", gstr(avail.V)))
		notes = []string{
			dim.Render(r.T("other programs")+" ") + hi.Render(other) + dim.Render(" · ") + left,
			dim.Render(r.T("other")+" ") + hi.Render(other) + dim.Render(" · ") + left,
			left,
		}
		if total.Ok() && total.V > 0 && inUse.Ok() {
			sv := math.Min(server.V, inUse.V)
			g = func(w int) string {
				return stack(w, seg{(inUse.V - sv) / total.V, "▓", txt}, seg{sv / total.V, "█", cy})
			}
		}
	} else {
		val = hiB.Render(r.vt(need, gstr))
		if !need.Ok() {
			val = dim.Render(r.vt(need, gstr))
		}
		if lb := m.h1Label(row); lb != "" && need.Ok() {
			// whole or not at all: the name «this model» stays (review item 13); 3 Memory
			// always carries the label
			vals = []string{val + dim.Render(" "+r.T("this model")+" · "+r.T(lb))}
		}
		val += dim.Render(" " + r.T("this model"))
		lv := hi.Render(r.vt(left, f1))
		notes = []string{
			use + dim.Render(" · "+r.T("left after start")+" ") + lv,
			use + dim.Render(" · "+r.T("left")+" ") + lv,
		}
		if left.Ok() && left.V < 0 && avail.Ok() && need.Ok() {
			// UX-R10 (addendum; UX-R3's form): below zero the tile says the shortfall and its
			// baseline, the memory available now less the need (the need stands above it)
			short := rdB.Render(r.Tf("%s short", gstr(-left.V)))
			notes = []string{
				dim.Render(r.Tf("free %s − need %s", f1(avail.V), f1(need.V))+" · ") + short,
				dim.Render(r.T("free")+" ") + hi.Render(f1(avail.V)) + dim.Render(" · ") + short,
				short,
			}
		}
		if total.Ok() && total.V > 0 && inUse.Ok() {
			n := 0.0
			if need.Ok() {
				n = need.V
			}
			g = func(w int) string {
				return stack(w, seg{inUse.V / total.V, "▓", txt}, seg{n / total.V, "█", cy})
			}
		}
	}
	if left.Ok() && left.V < 0 {
		edge = "9"
	}
	return func(w int) string {
		note := notes[len(notes)-1]
		for _, n := range notes {
			if lipgloss.Width(n) <= w-4 {
				note = n
				break
			}
		}
		v := val
		for _, c := range vals {
			if lipgloss.Width(c) <= w-4 {
				v = c
				break
			}
		}
		return r.tile(w, edge, label, v, g(max(0, w-4)), note)
	}
}

// vramTile: the card's use (nvidia-smi) with this server's share inside it,
// the same source as 3 Memory and 2 Server (review items 4, 14).
func (m *Model) vramTile(row ModelRow) tileFn {
	r := m.r
	used, total, server := row.VRAMUseGB, row.VRAMTotalGB, Val{}
	if m.mem != nil {
		if m.mem.UsedVRAMGB.Ok() {
			used = m.mem.UsedVRAMGB
		}
		if m.mem.TotalVRAMGB.Ok() {
			total = m.mem.TotalVRAMGB
		}
		if m.ph == phServing && row.ID == m.runModel {
			server = m.mem.ModelVRAMGB
		}
	}
	if !used.Ok() || !total.Ok() || total.V <= 0 {
		return r.sm("8", "VRAM", dim.Render("—"), noGauge, r.vt(used, gstr))
	}
	note, short := "", ""
	sv := 0.0
	if server.Ok() {
		sv = math.Min(server.V, used.V)
		note, short = r.T("this server")+" "+gstr(server.V), r.T("server")+" "+gstr(server.V)
	}
	return func(w int) string { // UX-R22: «this server» whole or as «server», never cut
		n := note
		if lipgloss.Width(n) > w-4 {
			n = short
		}
		return r.sm("8", "VRAM", hiB.Render(f1(used.V))+dim.Render(r.Tf(" of %s", f0(total.V))),
			func(w int) string {
				return stack(w, seg{sv / total.V, "█", cy}, seg{(used.V - sv) / total.V, "▓", txt})
			}, n)(w)
	}
}

func (m *Model) setupTile(row ModelRow) tileFn {
	r := m.r
	switch row.Setup {
	case SetupReady:
		return r.sm("8", "SETUP", grB.Render("✓ "+r.T("ready")), noGauge, "starts now")
	case SetupNeeded:
		return r.sm("11", "SETUP", yeB.Render("○ "+r.T("needed")), noGauge, "first run sets up")
	case SetupCutOff:
		return r.sm("11", "SETUP", yeB.Render("▲ "+r.T("cut off")), noGauge, "starts over")
	case SetupNeedsRepack:
		n := row.SetupNote
		if n == "" {
			n = "the repack output is gone"
		}
		return r.sm("11", "SETUP", yeB.Render("▲ "+r.T("needs repack")), noGauge, n)
	}
	n := row.SetupNote
	if t, ok := r.outsideNote(n); ok { // UX-R12 addendum (finding 18)
		n = t
	}
	if n == "" {
		n = r.np()
	}
	return r.sm("8", "SETUP", dim.Render("—"), noGauge, n)
}

// decodeHero is the tok/s tile of 1 Models: the model's last run average
// and its date, live while it serves (review item 28).
func (m *Model) decodeHero(row ModelRow, edge string) tileFn {
	r := m.r
	if m.ph == phServing && row.ID == m.runModel && m.status != nil {
		s := m.status
		if s.Phase == ReqDecode && s.DecodeTokS.Ok() {
			return r.heroL("10", "", f1(s.DecodeTokS.V), grB, "tok/s", txt.Render(r.T("live")))
		}
		if s.LastDecodeTokS.Ok() {
			return r.heroL(edge, "", f1(s.LastDecodeTokS.V), cyB, "tok/s", txt.Render(r.T("last reply")))
		}
	}
	if m.neverRun(row.ID) { // conflict 10: settings never run have no speed yet
		return r.heroL("8", "", "", dim, "tok/s", dim.Render(r.T("not measured with these settings")))
	}
	note := r.T("last run average")
	if row.MeasuredOn != "" {
		note = r.T("last run") + " · " + row.MeasuredOn
	}
	if !row.DecodeTokS.Ok() {
		return r.heroL("8", "", "", dim, "tok/s", dim.Render(r.T("last run average")+" · "+r.vt(row.DecodeTokS, f1)))
	}
	return r.heroL(edge, "", f1(row.DecodeTokS.V), cyB, "tok/s", txt.Render(note))
}

// fitText is the fit column (review items 11, 12): broken, unsupported, or
// the memory verdict fits · tight · too big · ?.
func fitText(r rc, row ModelRow) (string, lipgloss.Style) {
	if !row.Present {
		return "✕ " + r.T("file"), rd
	}
	if row.Setup == SetupNeedsRepack { // review item 31: shown in the row
		return "▲ " + r.T("needs repack"), ye
	}
	switch row.Fit {
	case FitRoom:
		return "● " + r.T("fits"), gr
	case FitTight:
		return "▲ " + r.T("tight"), ye
	case FitTooBig:
		return "✕ " + r.T("too big"), rd
	case FitBroken:
		return "✕ " + r.T("broken"), rd
	case FitUnsupported:
		return "✕ " + r.T("unsupported"), rd
	case FitUnmeasured:
		return "○ ?", dim
	}
	return "· —", dim
}

func fitMark(r rc, row ModelRow, w int) string {
	t, s := fitText(r, row)
	return s.Render(pad(t, w))
}

// fileCheckText is the "file check" row (review item 10).
func fileCheckText(r rc, c FileCheck) string {
	switch c {
	case FileSame:
		return gr.Render("✓ " + r.T("same as the catalog file"))
	case FileNoReference:
		return txt.Render(r.T("no reference hash"))
	case FileDiffers:
		return rd.Render("✕ " + r.T("differs from the catalog file"))
	case FileNotChecked:
		return dim.Render(r.T("not checked yet"))
	}
	return dim.Render(r.np())
}

// runChanged: the chosen model serves and a setting on the screen differs
// from what its last actual start used (review items 1, 2).
func (m *Model) runChanged(id string) bool {
	if m.ph != phServing || id != m.runModel {
		return false
	}
	for _, x := range m.views[id].Rows {
		if m.valueOf(id, x) != m.runValue(x) {
			return true
		}
	}
	return false
}

// runValue is the value the server runs with: the backend's record of the
// last actual start, else the value this screen started it with.
func (m *Model) runValue(x SettingRow) string {
	if x.Running != "" {
		return x.Running
	}
	if v, ok := m.runSet[x.Key]; ok {
		return v
	}
	return x.Value
}

// modelPanel is the model list, h lines high with its frame (0: its natural height). A
// narrow panel drops the size column, then the decode column. A higher box puts blank rows
// above its keys; a lower one first drops the blank line above the keys, then scrolls the
// rows so that the selected one stays in view, with the settings box's bar in the rows'
// last cell (UX-R13), and below three rows cuts its keys from the bottom, then the rows,
// then the column heads.
func (m *Model) modelPanel(w int, focus, short bool, h int) string {
	r := m.r
	fw := lipgloss.Width(r.T("fit")) + 1
	for _, x := range m.cat.Models {
		t, _ := fitText(r, x)
		fw = max(fw, lipgloss.Width(t)+1)
	}
	// build draws the column heads and one line per model in the width in
	build := func(in int) (string, []string) {
		nw := 13
		if in >= 2+17+fw+7+6 {
			nw = 17 // room for the star and a longer name
		}
		showDec, showSize := in >= 2+nw+fw+7, in >= 2+nw+fw+7+6
		hdr := "  " + pad(r.T("name"), nw) + pad(r.T("fit"), fw)
		if showDec {
			hdr += lpad(r.T("decode"), 7)
		}
		if showSize {
			hdr += lpad(r.T("size"), 6)
		}
		var rows []string
		for i, x := range m.cat.Models {
			name := x.Name
			if name == "" {
				name = x.ID
			}
			// ★ marks a catalog model (review items 11, 12)
			star := "  "
			switch {
			case x.Catalog:
				star = "★ "
			case x.Template:
				star = "◇ " // conflict 14: the structure's defaults
			}
			name = star + cut(name, nw-3) // live catalog names run longer than the mock's (GOL-WIRE)
			sp := "—"
			if x.DecodeTokS.Ok() {
				sp = f1(x.DecodeTokS.V)
			}
			sz := "—"
			if x.SizeGB.Ok() {
				sz = g0str(x.SizeGB.V)
			}
			cur, nm, sps, szs := "  ", txt.Render(pad(name, nw)), txt.Render(lpad(sp, 7)), txt.Render(lpad(sz, 6))
			if !x.Present {
				nm = dim.Render(pad(name, nw))
			}
			if i == m.sel {
				c := dim
				if focus {
					c = cyB
				}
				cur, nm, sps, szs = c.Render("▶ "), hiB.Render(pad(name, nw)), hiB.Render(lpad(sp, 7)), hi.Render(lpad(sz, 6))
			}
			l := cur + nm + fitMark(r, x, fw)
			if showDec {
				l += sps
			}
			if showSize {
				l += szs
			}
			rows = append(rows, l)
		}
		return dim.Render(hdr), rows
	}
	unmeasured, template, archs := false, false, []string(nil)
	for _, x := range m.cat.Models {
		if x.Fit == FitUnmeasured && x.Present {
			unmeasured = true
		}
		if x.Template { // UX-R10 (finding 18): drawn as a catalog row; the key says what ◇ is
			template, archs = true, appendNew(archs, x.Arch)
		}
	}
	var keys []string
	if m.cat.TooBig.Ok() && m.cat.TooBig.V > 0 {
		if short {
			keys = append(keys, dim.Render(fmt.Sprintf("+ %.0f %s · ", m.cat.TooBig.V, r.T("too big")))+cy.Render("5")+dim.Render(" "+r.T("Settings adds more")))
		} else {
			keys = append(keys, dim.Render(fmt.Sprintf("+ %.0f %s", m.cat.TooBig.V, r.T("too big for this PC"))))
		}
	}
	if unmeasured {
		keys = append(keys, dim.Render("○ ? = "+r.T("need not measured yet")))
	}
	if template { // UX-R10 (finding 18): what ◇ is, the positive fact first, whole (wrapped)
		a := strings.Join(archs, ", ")
		if a == "" {
			a = r.T("its structure")
		}
		for i, l := range wrapWords(r.Tf("not in catalog · %s compatible · runs", a), w-8) {
			lead := "    "
			if i == 0 {
				lead = "◇ = "
			}
			keys = append(keys, dim.Render(lead+l))
		}
	}
	in := w - 4
	hdr, rows := build(in)
	room := h - 2 // the lines inside the frame
	var list []string
	switch n := len(rows); {
	case h <= 0 || 1+n+1+len(keys) <= room: // the blank line above the keys, blank rows to the height
		list = append(append([]string{hdr}, rows...), "")
		for len(list)+len(keys) < room {
			list = append(list, "")
		}
		list = append(list, keys...)
	case 1+n+len(keys) == room: // a low window: the keys follow the rows (UX-R10)
		list = append(append([]string{hdr}, rows...), keys...)
	default: // UX-R13: the rows scroll; the selected row stays in view
		k := max(min(3, n), room-1-len(keys))
		k = max(1, min(k, room))
		if room >= k+1 {
			list = []string{hdr}
		}
		if k < n {
			_, rows = build(in - 2) // the rows leave room for the scroll bar
			top := min(max(0, m.sel-k/2), n-k)
			bar := scrollBar(k, top, n)
			for i, l := range rows[top : top+k] {
				if lipgloss.Width(l) > in-2 {
					l = ansi.Truncate(l, in-2, r.cutMark())
				}
				list = append(list, l+strings.Repeat(" ", max(0, in-1-lipgloss.Width(l)))+bar[i])
			}
		} else {
			list = append(list, rows...)
		}
		list = append(list, keys[:min(len(keys), max(0, room-len(list)))]...)
	}
	edge, title := "8", dim.Render(r.T("MODEL"))
	if focus {
		edge, title = "14", cyB.Render(r.T("MODEL"))
	}
	return r.panel(w, edge, title, list...)
}

func (m *Model) setPanel(w int, row ModelRow, focus, running bool, minH, maxH int) string {
	r := m.r
	in := w - 4
	v, have := m.views[row.ID]
	rows := m.visibleRows(v)
	head, nsty, edge := dim, txt, "8"
	if focus {
		head, nsty, edge = cyB, hi, "14"
	}
	title, prov := m.setTitle(w, row, head, nsty, have, running)
	// build draws the rows (one line each) and the lines under them in the width in; marks
	// counts the rows with each state mark: default, saved, not saved (UX-R10)
	build := func(in int) (lines, bottom []string, marks [3]int) {
		if !have {
			lines = append(lines, dim.Render(r.T("reading settings …")))
		}
		inputTail := ""
		fits := func(s string) bool { return lipgloss.Width(s) <= in }
		arrowed := false // a changed row shows «← value»: the bottom line names it (UX-R6)
		for i, x := range rows {
			val := m.valueOf(row.ID, x)
			// UX-R10 (finding 16 (b), (d)): the state mark of the row in its own column, so the
			// value never moves: · the default, ● a saved (applied) value other than the default,
			// ◆ changed here and not saved
			st, sv, mi := dim.Render("·"), hi.Render(val), 0
			switch {
			case m.changedRow(row.ID, x):
				st, sv, mi = yeB.Render("◆"), yeB.Render(val), 2 // a changed value in yellow (conflict 3)
			case x.Default != "" && x.Value != x.Default && x.Locked == "":
				st, sv, mi = cyB.Render("●"), cyB.Render(val), 1
			}
			marks[mi]++
			mark, k, vs := " "+st+" ", txt.Render(lab(r.T(x.Label))), sv
			if focus && i == m.setCur {
				mark, k, vs = cyB.Render("▶")+st+" ", hiB.Render(lab(r.T(x.Label))), cyB.Render("◀ ")+sv+cyB.Render(" ▶")
				if m.inputOn && x.Key == m.inputKey {
					// «budget [ 14.5_ ] G  12 G ~ 20 G · Enter 적용 · Esc 취소» (review item
					// 30); what does not fit on the row goes to the panel's last line
					vs = hiB.Render("[ "+m.inputBuf+"_ ]") + txt.Render(strings.TrimRight(" "+x.Unit, " "))
					keys := "Enter " + r.T("apply") + " · Esc " + r.T("cancel")
					inputTail = keys
					if x.Range != "" {
						inputTail = x.Range + " · " + keys
					}
					if full := mark + k + vs + "  " + dim.Render(inputTail); lipgloss.Width(full) <= in {
						vs += "  " + dim.Render(inputTail)
						inputTail = ""
					} else if lipgloss.Width(inputTail) > in && x.Range != "" {
						inputTail = x.Range // the keys stand in the footer
					}
					lines = append(lines, mark+k+vs)
					continue
				}
			}
			switch {
			case x.Locked != "":
				vs = dim.Render(val) + yeB.Render("  ■ "+r.T("locked"))
			case running && val != m.runValue(x):
				// «now» is the value of the last actual start (review item 2)
				vs += "  " + ye.Render(r.T("now")+" "+m.runValue(x))
			case !running && val != x.Value:
				// UX-R6: the earlier value says what it is, the model's default or its saved
				// value; where that does not fit, «← value» and the bottom line names it
				a := m.earlier(row.ID) + " " + x.Value
				if !fits(mark + k + vs + "  " + a) {
					a, arrowed = "← "+x.Value, true
				}
				vs += "  " + ye.Render(a)
			case x.Note != "":
				n := r.T(x.Note)
				if x.Basis != "" && fits(mark+k+vs+"  "+r.T("range")+" "+n) {
					n = r.T("range") + " " + n // UX-R6: the numbers say what they are
				}
				vs += "  " + ye.Render(n)
			}
			lines = append(lines, mark+k+vs)
		}
		// bottom: the line under the rows, what the focused row or the state needs said
		var fr *SettingRow // the focused row
		if focus && m.setCur < len(rows) {
			fr = &rows[m.setCur]
		}
		switch {
		case m.inputOn && m.inputErr != "":
			bottom = append(bottom, marked(in, "✕", rd, rd, typedErr(r, m.inputErr))...)
		case m.inputOn && inputTail != "":
			bottom = append(bottom, dim.Render(inputTail))
		case fr != nil && !running && m.changedRow(row.ID, *fr):
			// UX-R6: on a changed row, what its earlier value was and what «not saved» means
			// (in place of its hint, read before the change)
			bottom = append(bottom, m.unsavedLines(row.ID, arrowed, in)...)
		case fr != nil && fr.Hint != "":
			bottom = append(bottom, dim.Render("= "+r.T(fr.Hint)))
			if fr.Basis != "" { // UX-R6: what bounds the row's range
				bottom = append(bottom, wrapIn(dim, r.T(fr.Basis), in)...)
			}
		case running && m.runChanged(row.ID):
			bottom = append(bottom, ye.Render(r.T("not applied to the running server · applies at the next start")))
		case len(m.edits[row.ID]) > 0:
			bottom = append(bottom, m.unsavedLines(row.ID, arrowed, in)...)
			bottom = append(bottom, cy.Render("ctrl+s")+dim.Render(" "+r.T("saves the changed values")))
		default:
			for _, x := range rows { // UX-R6: what bounds the yellow ranges
				if x.Basis != "" && x.Note != "" {
					bottom = append(bottom, wrapIn(dim, r.T(x.Basis), in)...)
					break
				}
			}
		}
		return lines, bottom, marks
	}
	// UX-R10 (findings 17, 16 (b)): every row stands in the box, editable in place (no fold,
	// no «a»); the line under the rows says what the marks mean. Where the window is lower
	// than the rows, the rows scroll with the cursor and a scroll bar at the rows' right edge
	// (░ the rows, █ the ones shown) marks where they are.
	compose := func(lines, bottom []string, marks [3]int) (out []string, tail int) {
		var keys []string // the marks the box shows, each with its word
		for _, k := range []struct {
			mark, word string
			n          int
		}{{dim.Render("·"), "default", marks[0]}, {cyB.Render("●"), "saved", marks[1]}, {yeB.Render("◆"), "not saved", marks[2]}} {
			if k.n > 0 {
				keys = append(keys, k.mark+dim.Render(" "+r.T(k.word)))
			}
		}
		out = append([]string{}, lines...)
		// UX-R20 (item 31 (a)): «· default» alone says what the title's «default» says;
		// the line stands where another mark shows or the title does not say it
		if have && len(keys) > 0 && (marks[1]+marks[2] > 0 || !prov || v.Saved) {
			out = append(out, strings.Join(keys, "  "))
		}
		tail = len(out)
		out = append(out, "")
		return append(out, bottom...), tail
	}
	lines, bottom, marks := build(in)
	out, tail := compose(lines, bottom, marks)
	if maxH > 0 && len(out)+2 > maxH && len(rows) > 0 {
		lines, bottom, marks = build(in - 2) // the rows leave room for the scroll bar
		out, tail = compose(lines, bottom, marks)
		k := max(3, maxH-2-len(out)+len(lines)) // the rows that fit beside the legend, blank and bottom
		k = max(1, min(k, maxH-2))              // UX-R13: never more rows than the box holds
		if k < len(rows) {
			top := min(max(0, m.setCur-k/2), len(rows)-k)
			if !focus {
				top = 0
			}
			bar := scrollBar(k, top, len(rows))
			shown := make([]string, k)
			for i, l := range lines[top : top+k] {
				if lipgloss.Width(l) > in-2 {
					l = ansi.Truncate(l, in-2, r.cutMark())
				}
				shown[i] = l + strings.Repeat(" ", max(0, in-1-lipgloss.Width(l))) + bar[i]
			}
			out = append(shown, out[len(lines):]...)
			tail = k + 1
			if len(out)+2 > maxH && tail < len(out) && out[tail] == "" {
				out = append(out[:tail], out[tail+1:]...) // the blank line goes before a row would
			}
		}
	}
	if maxH > 0 && len(out)+2 > maxH { // UX-R13: the box keeps to its height, its last lines cut
		out = out[:max(1, maxH-2)]
		tail = min(tail, len(out))
	}
	for len(out)+2 < minH { // as tall as its neighbours, blanks above the last line
		out = append(out[:tail], append([]string{""}, out[tail:]...)...)
	}
	return r.panel(w, edge, title, out...)
}

// setTitle is the settings box's title (UX-R20, items 31 (a)-(d)): the model's name by one
// rule for every model, saved or not, at every width. The name never carries its quant part
// (the 이 모델 box shows the quant; the user's answer 31 (c)) and never leaves the title. Where
// the box is narrow, the title gives way in this order: «· not saved» (the ◆ rows and the line
// under them say it), «· running», the name cut to setNameMin cells, the word for where the
// settings came from («default» or «my settings», which the user reads as the saved marker,
// answer 31 (b)), and last the name cut to the room, with the cut mark (UX-R13). prov: the
// title says where the settings came from.
func (m *Model) setTitle(w int, row ModelRow, head, nsty lipgloss.Style, have, running bool) (title string, prov bool) {
	r := m.r
	name, room := baseName(row.Name, row.Quant), w-6
	label := head.Render(r.T("SETTINGS"))
	from, run := "", ""
	if have { // where the settings came from (conflict 3)
		from = m.provenance(row.ID)
	}
	if running {
		run = ye.Render(" · " + r.T("running"))
	}
	base, _, _ := strings.Cut(from, " · ")
	cut := func(n int) string { return ansi.Truncate(name, n, r.cutMark()) }
	for _, c := range []struct {
		name, from, run string
	}{{name, from, run}, {name, base, run}, {name, base, ""}, {"", base, ""}, {"", "", ""}} {
		tail := c.run
		if c.from != "" {
			tail = dim.Render(" · ") + m.provenanceView(c.from) + tail
		}
		nm := c.name
		if nm == "" { // the name cut to the room the rest leaves
			n := room - lipgloss.Width(label+" · "+tail)
			if c.from != "" && n < min(setNameMin, lipgloss.Width(name)) {
				continue
			}
			if n < min(4, lipgloss.Width(name)) {
				break
			}
			nm = cut(n)
		}
		if t := label + nsty.Render(" · "+nm) + tail; lipgloss.Width(t) <= room {
			return t, c.from != ""
		}
	}
	return nsty.Render(cut(max(1, room))), false
}

// setNameMin is the fewest cells the settings title keeps of the name, its cut mark
// included, before the word for where the settings came from leaves (UX-R20).
const setNameMin = 6

// quantWord: a word of a catalog name that starts its quant part, a quant (Q4_K_M, IQ2_XXS,
// MXFP4, BF16 …) or a size (447GB).
func quantWord(w string) bool {
	w = strings.ToLower(w)
	digitAt := func(i int) bool { return len(w) > i && w[i] >= '0' && w[i] <= '9' }
	for _, p := range []string{"q", "iq", "mxfp", "nvfp", "bf", "f"} {
		if strings.HasPrefix(w, p) && digitAt(len(p)) {
			return true
		}
	}
	if !digitAt(0) {
		return false
	}
	n := strings.TrimLeft(w, "0123456789.")
	return n == "gb" || strings.HasPrefix(n, "gb ") || strings.HasPrefix(n, "gb,")
}

// baseName is the model's name without its quant part: the words from the first one that
// holds the file's quant, looks like a quant or opens a parenthesis (UX-R10, finding 16
// (c): «Qwen3.5-35B-A3B Q4_K_M» → «Qwen3.5-35B-A3B», «DeepSeek-V4-Flash-0731 MXFP4/Q8_0
// (284B, …)» → «DeepSeek-V4-Flash-0731»). UX-R20 (item 31 (d)): a name taken from a file
// name has no spaces, so its parts between hyphens count as words too
// («DeepSeek-V4.1-Flash-MXFP4-engram» → «DeepSeek-V4.1-Flash»).
func baseName(name, quant string) string {
	quantPart := func(w string) bool {
		return (quant != "" && strings.Contains(w, quant)) || quantWord(w) || strings.HasPrefix(w, "(")
	}
	ws := strings.Fields(name)
	for i := range ws {
		if i > 0 && quantPart(ws[i]) {
			return strings.Join(ws[:i], " ")
		}
		ps := strings.Split(ws[i], "-")
		for j := 1; j < len(ps); j++ {
			if quantPart(ps[j]) {
				return strings.Join(append(ws[:i:i], strings.Join(ps[:j], "-")), " ")
			}
		}
	}
	return name
}

// scrollBar is the bar beside k of n rows from top: █ the rows shown, ░ the rest (UX-R10).
func scrollBar(k, top, n int) []string {
	size := max(1, (k*k+n/2)/n)
	at := 0
	if n > k {
		at = (top*(k-size) + (n-k)/2) / (n - k)
	}
	bar := make([]string, k)
	for i := range bar {
		if i >= at && i < at+size {
			bar[i] = cyB.Render("█")
		} else {
			bar[i] = dim.Render("░")
		}
	}
	return bar
}

// earlier names the value a changed setting had (UX-R6): the model's default,
// or its saved value (ctrl+s) when it has a saved profile («saved 12288»).
func (m *Model) earlier(model string) string {
	if m.views[model].Saved {
		return m.r.T("saved")
	}
	return m.r.T("default")
}

// changedRow: the screen holds a value for the row other than its earlier one.
func (m *Model) changedRow(model string, x SettingRow) bool {
	v, ok := m.edits[model][x.Key]
	return ok && v != x.Value
}

// unsavedLines say what «not saved» means (UX-R6): the changed values are used
// for this run only and are not written to the model's settings; arrowed says a
// changed row shows its earlier value as «← value», which the line then names.
func (m *Model) unsavedLines(model string, arrowed bool, in int) []string {
	s := m.r.T("not saved = this run only, not in this model's settings")
	if arrowed {
		s = "← " + m.earlier(model) + " · " + s
	}
	return wrapIn(ye, s, in)
}

// wrapIn is s wrapped at w cells, each line in sty.
func wrapIn(sty lipgloss.Style, s string, w int) []string {
	ls := wrapWords(s, w)
	for i := range ls {
		ls[i] = sty.Render(ls[i])
	}
	return ls
}

// lab is a row label in the 11-cell label column of 1 Models' boxes; a label of 11 cells or more
// («repack mode») keeps one space before its value (UX-R12 addendum, UX-R10 landing 4), its
// value starting one cell later than the column's
func lab(s string) string { return pad(s, max(11, lipgloss.Width(s)+1)) }

// outsideKey is the template form of a model outside the catalog (UX-R10, finding 18).
const outsideKey = "outside the catalog · %s template settings"

// outsideNote is a wire note of a model outside the catalog, «outside the catalog · <arch>
// template settings[ · more]», in the screen's language: the arch stays, the words go through
// the table (UX-R12 addendum, finding 18); ok is false, and s comes back, for another form.
func (r rc) outsideNote(s string) (string, bool) {
	pre, suf, _ := strings.Cut(outsideKey, "%s")
	rest, ok := strings.CutPrefix(s, pre)
	if !ok {
		return s, false
	}
	arch, more, _ := strings.Cut(rest, " · ")
	if arch, ok = strings.CutSuffix(arch, suf); !ok || arch == "" {
		return s, false
	}
	out := r.Tf(outsideKey, arch)
	if more != "" {
		for _, p := range strings.Split(more, " · ") {
			out += " · " + r.T(p)
		}
	}
	return out, true
}

func (m *Model) aboutPanel(w int, row ModelRow, minH int) string {
	r := m.r
	k := func(a, b string) string { return dim.Render(lab(r.T(a))) + txt.Render(b) }
	or := func(s string) string {
		if s == "" {
			return r.np()
		}
		return s
	}
	q := or(row.Quant)
	if row.SizeGB.Ok() {
		q += " · " + gstr(row.SizeGB.V)
	}
	setup := map[SetupState]string{SetupReady: r.T("done"), SetupNeeded: r.T("needed"), SetupCutOff: r.T("cut off"), SetupNeedsRepack: r.T("needs repack")}[row.Setup]
	if setup == "" {
		setup = r.np()
	}
	if row.SetupNote != "" {
		n, _ := r.outsideNote(row.SetupNote) // UX-R12 addendum (finding 18)
		setup += " · " + n
	}
	dec := r.vt(row.DecodeTokS, func(f float64) string { return f1(f) + " tok/s" })
	pre := r.vt(row.PrefillTokS, func(f float64) string { return f0(f) + " tok/s" })
	if row.TTFTs.Ok() {
		pre += " · TTFT " + secs(row.TTFTs.V)
	}
	folder := row.Folder
	if folder == "" {
		folder = row.Path
	}
	var rows []string
	if row.Template { // conflict 14
		a := row.Arch
		if a == "" {
			a = r.T("its structure")
		}
		for _, l := range wrapWords(r.Tf("outside the catalog · %s template settings · performance untested", a), w-4) {
			rows = append(rows, ye.Render(l))
		}
	}
	rows = append(rows, k("quant", q), k("file check", fileCheckText(r, row.FileCheck)), k("folder", or(folder)))
	if len(row.Modes) > 0 {
		// conflict 16: both modes on one line, the mode now set in bold
		l := dim.Render(lab(r.T("setup"))) + m.modesText(row, false)
		if lipgloss.Width(l) > w-4 { // the short marks keep the line whole
			l = dim.Render(lab(r.T("setup"))) + m.modesText(row, true)
		}
		rows = append(rows, l)
	} else {
		rows = append(rows, k("setup", setup))
	}
	rows = append(rows, k("decode", dec))
	if row.PrefillTokS.Ok() { // nothing when the backend gives none (FIX-WIRE section 4 item 4)
		rows = append(rows, k("prefill", pre))
	}
	rows = append(rows, k("last", or(row.LastRun)))
	if minH > 0 && len(rows)+2 > minH { // UX-R13: a lower row cuts the last rows
		rows = rows[:max(1, minH-2)]
	}
	for len(rows)+2 < minH {
		rows = append(rows, "")
	}
	return r.panel(w, "8", dim.Render(r.T("THIS MODEL")), rows...)
}

// modelsRow is the box row of 1 Models, h lines high (0: its natural height, the height of
// its tallest box). Every box is drawn to that height with its whole frame (UX-R13): the
// list scrolls its rows, the settings box scrolls its rows (UX-R10), and the 이 모델 box cuts
// its last rows. The settings box may use the row's height.
func (m *Model) modelsRow(row ModelRow, running bool, h int) string {
	r := m.r
	if r.wide() {
		c := r.cols(3)
		if h <= 0 {
			h = max(lipgloss.Height(m.modelPanel(c[0], !m.focusSet, false, 0)),
				lipgloss.Height(m.setPanel(c[1], row, m.focusSet, running, 0, 0)),
				lipgloss.Height(m.aboutPanel(c[2], row, 0)))
		}
		return row3(m.modelPanel(c[0], !m.focusSet, false, h), m.setPanel(c[1], row, m.focusSet, running, h, h), m.aboutPanel(c[2], row, h))
	}
	c := r.cols(2)
	right := func(h int) string {
		if m.showAbout && !m.focusSet {
			return m.aboutPanel(c[1], row, h)
		}
		return m.setPanel(c[1], row, m.focusSet, running, h, h)
	}
	if h <= 0 {
		h = max(lipgloss.Height(m.modelPanel(c[0], !m.focusSet, true, 0)), lipgloss.Height(right(0)))
	}
	return row3(m.modelPanel(c[0], !m.focusSet, true, h), right(h))
}

func row3(ts ...string) string { return row(ts...) }

// verdictLine is the one-line verdict of 1 Models (review item 22): the
// verdict, one reason with its numbers, and where the details are. Its
// numbers are the live ones of review item 13.
func (m *Model) verdictLine(row ModelRow, p *Preflight) string {
	r := m.r
	if p == nil {
		return ""
	}
	_, avail, need, left, _ := m.liveRAM(row)
	more := " · " + r.T("details on 3 Memory")
	switch {
	case p.Refused || p.Verdict == VerdictCannot:
		why := ""
		switch {
		case p.CacheGB.Ok() && avail.Ok() && p.CacheGB.V > avail.V:
			why = r.Tf("RAM budget %s > free now %s", gstr(p.CacheGB.V), gstr(avail.V))
		case need.Ok() && avail.Ok() && need.V > avail.V:
			why = r.Tf("needs %s > free now %s", gstr(need.V), gstr(avail.V))
		case len(p.Refusals) > 0:
			why = r.reason(p.Refusals[0]) // the backend's words, whole; wrapped, not cut (review item 32); a fixed line in the screen language (W7 7)
		}
		if p.BudgetCapGB.Ok() && p.BudgetCapGB.V > 0 { // conflict 7: the cap that passes
			if why != "" {
				why += " · "
			}
			why += r.Tf("a budget of %s or less starts it", gstr(p.BudgetCapGB.V))
		}
		l := r.T("cannot start")
		if why != "" {
			l += " · " + why
		}
		return r.alert("✕", rd, l+more)
	case p.Verdict == VerdictTight:
		return r.alert("▲", ye, r.T("tight to start")+" · "+r.Tf("RAM left after the start %s", r.vt(left, gstr))+more)
	case p.Verdict == VerdictRoom:
		if !left.Ok() {
			return ""
		}
		safe := 0.0
		if p.CommitSafeGB.Ok() {
			safe = p.CommitSafeGB.V
		}
		x := math.Floor(left.V - safe)
		if x >= 4 {
			// the user's wording (04:33 KST); the threshold is 4 G
			return "  " + gr.Render(r.Tf("✓ %s GB to spare · plenty of room for other work alongside the model :>", fmt.Sprintf("%.0f", x)))
		}
		return "  " + gr.Render(r.Tf("✓ can start · %s RAM for other work", gstr(math.Max(0, left.V-safe))))
	case p.Verdict == VerdictUnmeasured:
		return "  " + dim.Render("○ "+r.T("need not measured yet")+more)
	}
	return ""
}

func (m *Model) modelsHead() string { return m.r.pageHead(m.r.T("pick a model, then start"), "") }

// settingsSummary is the "preset · budget · context" text of the start line.
func (m *Model) settingsSummary(row ModelRow) string {
	r := m.r
	v, ok := m.views[row.ID]
	if !ok {
		return ""
	}
	var p []string
	if v.Preset != "" {
		pr, ok := r.outsideNote(v.Preset) // UX-R12 addendum (finding 18)
		if !ok {
			pr = r.T(v.Preset)
		}
		p = append(p, pr)
	}
	for _, x := range v.Rows {
		if !x.Folded && len(p) < 4 {
			p = append(p, r.T(x.Label)+" "+m.valueOf(row.ID, x))
		}
	}
	return strings.Join(p, " · ")
}

// startLine is the start line of 1 Models: the start key and the model, and
// nothing more (UX-R2, review items 2 and 16(a)): the settings, their values
// and where they came from stand in the settings box, so no value is listed
// here and none is cut at the window edge.
func (m *Model) startLine(lbl string, row ModelRow) string {
	return "  " + pill(lbl, "14") + " " + txt.Render(row.Name)
}

// noticeRows is the fixed height of the notice area of 1 Models in a full
// window: the lines between the tile row and the three boxes (start line,
// verdict, notices) take this many rows whatever they hold, so the boxes
// keep their place (UX-R2, review item 3).
const noticeRows = 4

// noticeArea fits the notice area to noticeRows lines in a full window: the
// head lines (the start line and the verdict) first, then whole notices
// while they fit; what is left out folds into one line that counts the
// notices not shown and points to 3 Memory, which lists every notice. Below
// 80x28 the area keeps its natural lines, so the leaving order of a small
// window (GL-04) is the one it was.
func (m *Model) noticeArea(head []string, notices []string) []string {
	r := m.r
	var hl []string
	for _, x := range head {
		hl = append(hl, strings.Split(x, "\n")...)
	}
	nl := func(ns []string) []string {
		if len(ns) == 0 {
			return nil
		}
		return strings.Split(r.alert("▲", ye, strings.Join(ns, " · ")), "\n")
	}
	if r.W < fullW || r.H < fullH {
		return append(hl, nl(notices)...)
	}
	fold := func(k int) string { // «▲ +k · details on 3 Memory»: k notices not shown
		if k > 0 {
			return r.alert("▲", ye, "+"+fmt.Sprint(k)+" · "+r.T("details on 3 Memory"))
		}
		return "  " + dim.Render("… "+r.T("details on 3 Memory"))
	}
	out := hl
	switch {
	case len(hl) > noticeRows, len(hl) == noticeRows && len(notices) > 0:
		out = append(hl[:noticeRows-1:noticeRows-1], fold(len(notices)))
	default:
		rem := noticeRows - len(hl)
		k := len(notices)
		for k > 0 && len(nl(notices[:k])) > rem-1 && !(k == len(notices) && len(nl(notices)) <= rem) {
			k--
		}
		switch {
		case k == len(notices):
			out = append(out, nl(notices)...)
		case k > 0:
			out = append(append(out, nl(notices[:k])...), fold(len(notices)-k))
		case rem >= 2:
			// the first notice alone is higher than the room: its first lines, cut, then the fold
			first := nl(notices[:1])[:rem-1]
			first[rem-2] += r.cutMark()
			out = append(append(out, first...), fold(len(notices)-1))
		default:
			out = append(out, fold(len(notices)))
		}
	}
	for len(out) < noticeRows {
		out = append(out, "")
	}
	return out
}

// vModels is H1, and H2 while the server runs with other settings.
func (m *Model) vModels() string {
	r := m.r
	row, ok := m.selRow()
	if !ok {
		body := []string{"  " + dim.Render(r.T("reading the catalog …"))}
		if m.catErr != "" {
			body = []string{r.alert("✕", rd, r.T("the catalog could not be read")+": "+m.catErr)}
		}
		return m.page(plModels, m.modelsHead(), body, foot{next: []string{"r looks again"}, here: []string{"r", "look again"}})
	}
	running := m.ph == phServing && row.ID == m.runModel
	h2 := m.runChanged(row.ID)
	edge := "14"
	if h2 {
		edge = "11"
	}
	body := []string{
		r.tiles(m.decodeHero(row, edge), m.ramTile(row, map[bool]string{true: "11", false: "8"}[h2]), m.vramTile(row), m.setupTile(row)),
		"",
	}
	p := m.currentPreflight(row.ID)
	switch {
	case h2:
		// review item 1: the guide line, worded as confirmed; under the lock line (UX-R15)
		body = append(body, m.noticeArea([]string{m.lockLine(), r.alert("▲", ye, r.T("not applied to the running server · applies at the next start"))}, nil)...)
	case m.serverUp():
		body = append(body, m.noticeArea([]string{m.lockLine()}, nil)...)
	default:
		var head []string // the start verdict: what ⏎ starts and whether it can (U-31: it stays)
		setup := m.needsSetup(row)
		switch {
		case m.bundle.State == BundleBroken: // conflict 18-2
			head = append(head, r.alert("✕", rd, strings.TrimPrefix(m.bundleBrokenText(), "✕ ")))
		case m.startWait == row.ID:
			head = append(head, r.alert("·", txt, r.T("checking the launcher files … the start follows")))
		case setup || p == nil || !p.Refused:
			// what ⏎ starts: the model (its settings stand in the settings box; UX-R2)
			lbl := "⏎ " + r.T("START")
			if setup {
				lbl = "⏎ " + r.T("SETUP PLAN") // conflict 9: the plan box first
			}
			head = append(head, m.startLine(lbl, row))
		}
		if l := m.verdictLine(row, p); l != "" {
			head = append(head, l)
		}
		var ns []string // GL-20: in the screen language (U-37 (i))
		if p != nil {
			for _, n := range p.Notices {
				ns = append(ns, r.T(n))
			}
		}
		for _, l := range m.noticeArea(head, ns) {
			body = append(body, verdictPart(l))
		}
	}
	var below []string
	if m.custom(row.ID) { // conflict 10: under the settings, gone at the defaults
		below = append(below, r.alert("▲", ye, r.T("custom settings · performance unverified")))
	}
	f := m.modelsFoot(row, h2, p)
	boxes := ""
	switch h := m.boxRoom(body, below, f, lipgloss.Height(m.modelsRow(row, running, 0))); {
	case h >= minBox:
		boxes = m.modelsRow(row, running, h)
	case r.W < fullW || r.H < fullH: // UX-R13: no frame fits; the row leaves as one part (frame.go small)
		boxes = m.modelsRow(row, running, minBox)
	}
	body = append(body, "", boxesPart(boxes))
	body = append(body, below...)
	return m.page(plModels, m.modelsHead(), body, f)
}

// minBox is the lowest box row that still holds a frame: its top edge, one row, its bottom
// edge (UX-R13). In a lower room the row leaves as one part.
const minBox = 3

// lockLine is the first notice of 1 Models while a server runs (UX-R15, finding 25): the
// model choice is locked (review item 1) and how to unload; the page is named as the nav row
// names it.
func (m *Model) lockLine() string {
	return m.r.alert("·", txt, m.lockWords())
}

// lockWords is the one wording of the list lock (UX-KR, the addendum to UX-R15): the notice
// area, the arrow key's notice and the serving footer, the page named as the nav row names it.
func (m *Model) lockWords() string {
	r := m.r
	return r.Tf("server running · model choice locked · to unload: %s, ctrl+f", places[1][0]+" "+r.T(places[1][1]))
}

// boxRoom is the height of the box row of 1 Models (UX-R10, UX-R13): its natural height nat,
// at most what the window leaves for it, the window less the page's top (status, rule, head,
// blank), the footer, the lines above the boxes and below them. Below 80x28 (GL-04, U-31) it
// is also at most the room the 80x28 layout leaves (its notice area is noticeRows lines), so
// a smaller window never shows a row that a full one leaves out; and where the row does not
// fit beside the graph, the graph leaves first (frame.go small) and the row has its room
// too. A result below minBox means the row cannot be framed.
func (m *Model) boxRoom(above, below []string, f foot, nat int) int {
	r := m.r
	fixed := 4 + len(r.footer(f, m.arm(), m.notice)) + 1 // the blank line before the boxes
	n, graph := fixed, 0
	for _, x := range append(append([]string{}, above...), below...) {
		k, x := partOf(x)
		l := len(strings.Split(x, "\n"))
		n += l
		if k == "graph" {
			graph += l
		}
	}
	room := r.H - n
	if r.W >= fullW && r.H >= fullH {
		return min(nat, room)
	}
	h := min(nat, fullH-n+m.noticeLines(above)-noticeRows)
	if h > room {
		h = min(h, room+graph)
	}
	return h
}

// noticeLines counts the lines of the notice area in the body above the boxes: what follows
// the tile row and its blank line.
func (m *Model) noticeLines(above []string) int {
	n := 0
	for _, x := range above[min(2, len(above)):] {
		_, x = partOf(x)
		n += len(strings.Split(x, "\n"))
	}
	return n
}

// modelsFoot is the footer of 1 Models.
func (m *Model) modelsFoot(row ModelRow, h2 bool, p *Preflight) foot {
	r := m.r
	var f foot
	typed := false
	if v, ok := m.views[row.ID]; ok && m.focusSet {
		if vr := m.visibleRows(v); m.setCur < len(vr) {
			typed = vr[m.setCur].Typed && vr[m.setCur].Locked == ""
		}
	}
	switch {
	case m.inputOn:
		f = foot{next: []string{"⏎ applies after the range check · esc keeps the value as it was"},
			here: []string{"⏎", "apply", "esc", "cancel"}}
	case h2:
		f = foot{next: []string{"ctrl+s saves for the next start · ctrl+f unloads, then ⏎ starts"},
			here: []string{"←→", "change", "ctrl+s", "save", "ctrl+f", "unload", "esc", "undo"}}
	case m.serverUp() && m.focusSet:
		f = foot{next: []string{"changes apply at the next start · ctrl+f unloads first"},
			here: []string{"↑↓", "setting", "←→", "change", "tab", "models", "ctrl+s", "save", "esc", "undo"}}
	case m.serverUp():
		f = foot{next: []string{m.lockWords()},
			here: []string{"tab", "settings", "ctrl+f", "unload"}}
	case m.focusSet:
		f = foot{next: []string{"←→ changes the chosen value · ⏎ starts with these values · esc undoes"},
			here: []string{"↑↓", "setting", "←→", "change", "tab", "models", "⏎", "start", "ctrl+s", "save", "esc", "undo"}}
	}
	if typed && !m.inputOn {
		// the tail line names Space (review item 30)
		f.here = append([]string{"space", "type a value"}, f.here...)
	}
	switch {
	case m.inputOn || h2 || m.serverUp() || m.focusSet:
	default:
		next := r.T("⏎ starts") + " " + row.Name + " · " + r.T("2 Server shows each step")
		if p != nil && p.LastStartS.Ok() {
			next += fmt.Sprintf(", ~%.0f s", p.LastStartS.V)
		}
		if !row.Present {
			next = row.Name + ": " + r.T("model file missing")
		}
		here := []string{"↑↓", "model", "tab", "settings", "⏎", "start"}
		if !r.wide() {
			here = append(here, "i", "details")
		}
		f = foot{next: []string{next}, here: here}
	}
	return f
}

// vH3: no model file of the catalog is present.
func (m *Model) vH3() string {
	r := m.r
	var tl []tileFn
	for i, f := range m.cat.Folders {
		if i == 3 {
			break
		}
		f := f
		var edge, val, note string
		switch f.State {
		case FolderMissing:
			edge, val, note = "9", rdB.Render("✕ "+r.T("missing")), r.T("folder does not exist")
		case FolderEmpty:
			edge, val, note = "11", yeB.Render("○ "+r.T("empty")), r.T("no .gguf file yet")
		case FolderUnsupported:
			edge, val, note = "11", yeB.Render(fmt.Sprintf("? %d %s", f.Found, r.T("unsupported"))), r.T("not MoE models")
		case FolderWatching:
			edge, val, note = "10", grB.Render("● "+r.T("watching")), fmt.Sprintf("%d %s", f.Found, r.T("found"))
		default:
			edge, val, note = "8", dim.Render("—"), r.np()
		}
		tl = append(tl, func(w int) string { return r.tile(w, edge, f.Path, val, dim.Render(note), "") })
	}
	if len(tl) == 0 {
		tl = append(tl, r.sm("8", "FOLDERS", dim.Render("—"), noGauge, r.T("folder scan")+": "+r.np()))
	}
	present := 0
	body := []string{r.tiles(r.heroL("11", "", fmt.Sprint(present), yeB, r.T("models"), txt.Render(r.T("found"))), tl...), ""}
	watching := false
	for _, f := range m.cat.Folders {
		watching = watching || f.State == FolderWatching
	}
	where := r.T("the catalog's model folder")
	if len(m.cat.Folders) > 0 {
		where = m.cat.Folders[0].Path
	} else if len(m.cat.Models) > 0 && m.cat.Models[0].Folder != "" {
		where = m.cat.Models[0].Folder
	}
	s2 := r.T("press r to look again") + dim.Render(" · "+r.T("folder watch")+": "+r.np())
	if watching {
		s2 = r.T("it shows up here by itself, no restart needed")
	}
	steps := []string{
		cyB.Render("1") + "  " + txt.Render(r.T("put a model file (.gguf) of the catalog into")+" ") + hiB.Render(where),
		cyB.Render("2") + "  " + txt.Render(s2),
		cyB.Render("3") + "  " + txt.Render(r.T("model somewhere else? add that folder in")+" ") + cyB.Render("5") + txt.Render(" "+r.T("Settings")),
		"",
	}
	var fits []string
	for _, x := range m.cat.Models {
		if x.Fit == FitRoom {
			fits = append(fits, x.Name)
		}
	}
	if len(fits) > 0 {
		steps = append(steps, dim.Render(r.T("fits this PC")+": "+strings.Join(fits, " · ")))
	} else {
		steps = append(steps, dim.Render(r.T("fits this PC")+": "+r.nm()))
	}
	if m.catErr != "" {
		steps = append(steps, rd.Render(m.catErr))
	}
	body = append(body, indent(r.panel(r.cw(), "14", cyB.Render(r.T("HOW TO ADD A MODEL")), steps...), 1))
	return m.page(plModels, m.modelsHead(), body, foot{next: []string{
		"a .gguf of the catalog saved into a model folder appears after r · r looks now",
		"r looks again now",
	}, here: []string{"r", "look again", "5", "folders"}})
}

// vH4: the chosen model's setup was cut off.
func (m *Model) vH4() string {
	r := m.r
	row, _ := m.selRow()
	pa := row.Partial
	done := "—"
	if pa.LayersDone.Ok() {
		done = f0(pa.LayersDone.V)
	}
	unit := "/ —"
	if pa.LayersTotal.Ok() {
		unit = "/ " + f0(pa.LayersTotal.V)
	}
	files := r.vt(pa.FilesGB, g0str)
	redo := r.vt(pa.RedoMin, func(f float64) string { return fmt.Sprintf(r.T("~%.0f min"), f) })
	body := []string{
		r.tiles(r.heroL("11", "", done, yeB, unit, txt.Render(r.T("layers done"))),
			r.sm("8", "REDO", hiB.Render(redo), func(w int) string { return bar(0, w, ye) }, "from layer 1"),
			r.sm("8", "FILES", hiB.Render(files), func(w int) string {
				if pa.FilesGB.Ok() && row.SizeGB.Ok() && row.SizeGB.V > 0 {
					return bar(pa.FilesGB.V/row.SizeGB.V, w, yeD)
				}
				return ""
			}, "half-made"),
			r.sm("8", "RESUME", rdB.Render(r.T("no")), noGauge, "not possible")),
		"",
		r.alert("▲", ye, fmt.Sprintf(r.T("setup of %s was cut off")+" %s", row.Name, strings.TrimSpace(pa.When+" "+pa.Cause))),
		"",
		"  " + pill("ctrl+r "+r.T("START AGAIN"), "11") + "   " + cyB.Render("ctrl+d") + txt.Render(" "+r.T("delete half-made only")) + "   " + cyB.Render("esc") + txt.Render(" "+r.T("not now")),
		"",
	}
	for i, x := range m.cat.Models {
		if i == 3 {
			break
		}
		if i == m.sel {
			body = append(body, dim.Render("   ▶ ")+hiB.Render(pad(x.Name, 16))+ye.Render("▲ "+r.T("cut off")))
			continue
		}
		st := "● " + r.T("ready")
		if x.Setup != SetupReady {
			st = "○ " + r.T("not set up")
		}
		body = append(body, dim.Render("     "+pad(x.Name, 16))+dim.Render(st))
	}
	return m.page(plModels, r.pageHead(r.T("a setup was cut off"), ""), body, foot{next: []string{
		"ctrl+r shows the setup plan · its ⏎ deletes the half-made files and redoes setup · ctrl+d deletes them only",
		"ctrl+r setup plan · ctrl+d delete only",
	}, here: []string{"ctrl+r", "start again", "ctrl+d", "delete only", "esc", "not now"}})
}
