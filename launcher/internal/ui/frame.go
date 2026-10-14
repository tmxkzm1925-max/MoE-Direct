// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// The round-5 frame (designs/launcher_design/proto/go5/h.go): every screen is
// one of five places, and the same frame says where you are, where else you
// can go and what happens next:
//
//	row 1      places 1-5 (current one lit) · server state at the right
//	row 2      rule with a mark under the current place
//	row 3      what this page is for
//	body       tiles, panels, one picture
//	bottom 3   rule · what happens next · keys for here | keys for anywhere

package ui

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// rc is the render context: window size and language.
type rc struct {
	W, H int
	lang string
	// shown receives the parts of each drawn screen (GL-04; nil: not kept)
	shown *Parts
}

const wideCols = 110 // from here on a third column appears

func (r rc) T(s string) string { return tr(r.lang, s) }
func (r rc) wide() bool        { return r.W >= wideCols }

// cw is the content width inside the 1-space margins.
func (r rc) cw() int { return max(1, min(r.W, 160)-2) }

func (r rc) spread(l, rt string) string {
	return l + rep(" ", max(1, r.W-lipgloss.Width(l)-lipgloss.Width(rt))) + rt
}

func (r rc) Tf(tpl string, vals ...string) string {
	return fill(r.lang, tpl, func(v ...string) string { return v[0] }, func(v ...string) string { return v[0] }, vals...)
}

// assembled are the backend's sentences built around a value (a mode, a family, a date, item
// numbers): the exact sentence has no entry, so tr looks it up by its pattern (UX-KR: the rows
// of the Korean string table outside the map marked «조립 문장»; wire userReason, presetOf).
var assembled = []string{
	"no %s repack files yet · ⏎ opens the setup plan",
	"the %s repack check failed at item %d of %d · ⏎ opens the setup plan",
	"%s is not ported",
	"saved settings (%s)",
}

// not-delivered words
func (r rc) np() string { return r.T("not ported") }
func (r rc) nm() string { return r.T("not measured yet") }

// vt renders a value with f, or the word for its state.
func (r rc) vt(v Val, f func(float64) string) string {
	switch v.State {
	case Known:
		return f(v.V)
	case Unmeasured:
		return r.nm()
	}
	return r.np()
}

// vu renders a known value bright with its unit dim, or the dim word for its state.
func (r rc) vu(v Val, f func(float64) string, unit string) string {
	if v.Ok() {
		return hiB.Render(f(v.V)) + dim.Render(unit)
	}
	return dim.Render(r.vt(v, f))
}

func gstr(f float64) string  { return fmt.Sprintf("%.1f GB", f) } // UX-R7: «GB», never a bare «G»
func g0str(f float64) string { return fmt.Sprintf("%.0f GB", f) }
func f1(f float64) string    { return fmt.Sprintf("%.1f", f) }
func f0(f float64) string    { return fmt.Sprintf("%.0f", f) }
func secs(f float64) string  { return fmt.Sprintf("%.1f s", f) }

// srv is the server state at the right of the top row.
type srv struct {
	sym, word string
	sty       lipgloss.Style
	detail    string
	short     string // for narrow windows
}

// serving: the server runs (srvState gives ● serving and ▲ slow only while serving); the 2 Server
// place of the top row then carries the mark in the state's colour (UX-R9, finding 19).
func (s srv) serving() bool { return s.sym == "●" || s.sym == "▲" }

var places = [][2]string{{"1", "Models"}, {"2", "Server"}, {"3", "Memory"}, {"4", "Log"}, {"5", "Settings"}}

// navPill is the lit place: its letters are colour 16 of the 256-colour
// table, a black that a terminal does not brighten when it is bold (review
// item 26; the other pills keep colour 0).
func navPill(s, bg string) string {
	return navPillStyle(bg).Render(" " + s + " ")
}

// navPillStyle is the lit place's style (review item 26; S8-LA-09 reads it).
func navPillStyle(bg string) lipgloss.Style {
	return st().Background(lipgloss.Color(bg)).Foreground(lipgloss.Color("16")).Bold(true)
}

func (r rc) nav(cur int, s srv) []string {
	var l string
	var at, aw int
	// a long language can push the row past the window: then only the
	// current place keeps its name, the others show their number
	for _, short := range []bool{false, true} {
		l = " " + hiB.Render("MoE-Direct") + dim.Render("  │")
		x := lipgloss.Width(l)
		for i, p := range places {
			sep := " "
			if !r.wide() {
				sep = ""
			}
			// the separator comes first, so the mark under the lit place
			// starts where its pill starts (review item 25)
			l += sep
			x += len(sep)
			var it string
			live, bg := i == plServer && s.serving(), "10" // UX-R9 (finding 19): «2 Server ●» while a server runs
			if s.sym == "▲" {
				bg = "11"
			}
			switch {
			case i == cur && live:
				it = navPill(p[0]+" "+r.T(p[1])+" "+s.sym, bg)
			case i == cur:
				it = navPill(p[0]+" "+r.T(p[1]), "14")
			case short && live:
				it = " " + cyB.Render(p[0]) + s.sty.Render(s.sym) + " "
			case short:
				it = " " + cyB.Render(p[0]) + " "
			case live:
				it = " " + cyB.Render(p[0]) + " " + s.sty.Render(r.T(p[1])+" "+s.sym) + " "
			default:
				it = " " + cyB.Render(p[0]) + " " + txt.Render(r.T(p[1])) + " "
			}
			if i == cur {
				at, aw = x, lipgloss.Width(it)
			}
			l += it
			x += lipgloss.Width(it)
		}
		if lipgloss.Width(l)+lipgloss.Width(s.short)+4 <= r.W {
			break
		}
	}
	rt := s.sty.Render(s.sym+" "+s.short) + " "
	cands := []string{s.sty.Render(s.sym+" "+s.word) + dim.Render(s.detail) + " ", s.sty.Render(s.sym+" "+s.word) + " "}
	if r.W < fullW || r.H < fullH {
		// below 80x28 the state never gains the detail the full window has no
		// room for (S8-LA-09: no word that the 80x28 capture lacks)
		cands = cands[1:]
	}
	for _, c := range cands {
		if lipgloss.Width(l)+lipgloss.Width(c)+2 <= r.W {
			rt = c
			break
		}
	}
	if lipgloss.Width(l)+lipgloss.Width(rt)+1 > r.W {
		rt = s.sty.Render(s.sym) + " "
	}
	if cur < 0 { // help: no place is lit
		at, aw = 0, 0
	}
	ru := dim.Render(rep("─", at)) + cyB.Render(rep("━", aw)) + dim.Render(rep("─", r.W-at-aw))
	return []string{r.spread(l, rt), ru}
}

func (r rc) pageHead(purpose, ctx string) string {
	l := " " + hi.Render(purpose)
	if ctx == "" || lipgloss.Width(l)+lipgloss.Width(ctx)+2 > r.W {
		return l
	}
	return r.spread(l, dim.Render(ctx+" "))
}

// armed says which two-press key waits for its second press.
type armed uint8

const (
	armedNone armed = iota
	armedQuit
	armedForce
	armedQuitReply // the first q while a reply runs (U-27)
)

// foot is the bottom of a screen: what happens next, then keys.
type foot struct {
	next []string // first variant that fits; the last one is used when none fits
	warn bool     // the next line in yellow (a key waits for its second press)
	here []string // key, meaning pairs for this page
}

func (r rc) footer(f foot, arm armed, notice string) []string {
	n := ""
	if len(f.next) > 0 {
		n = r.T(f.next[len(f.next)-1])
		for _, c := range f.next {
			if lipgloss.Width(r.T(c))+4 <= r.W {
				n = r.T(c)
				break
			}
		}
	}
	nsty := txt
	if f.warn {
		nsty = yeB
	}
	switch {
	case arm == armedForce:
		n, nsty = r.T("ctrl+c again closes now · the cache is not saved · other keys cancel"), yeB
	case notice != "":
		n, nsty = notice, ye
	}
	var b strings.Builder
	b.WriteString(" ")
	for i := 0; i+1 < len(f.here); i += 2 {
		b.WriteString(" " + cyB.Render(f.here[i]) + " " + txt.Render(r.T(f.here[i+1])) + "  ")
	}
	if arm == armedQuitReply {
		// U-27 (L:4275): the first q while a reply runs puts its warning in the
		// bottom key line, in place of the page's keys; wording F-LS 11.8 (OQ-4 (b), user)
		b.Reset()
		b.WriteString("  " + yeB.Render(r.T("reply in progress · press again to force quit")) + "  ")
	}
	// ←→ turns pages unless this page's own keys use ←→; the page slot
	// shortens, then drops, when the line is full
	valueMode := false
	for i := 0; i < len(f.here); i += 2 {
		valueMode = valueMode || f.here[i] == "←→"
	}
	pgs := []string{
		cy.Render("←→ 1-5") + txt.Render(" "+r.T("pages")+"  "),
		cy.Render("←→") + txt.Render(" "+r.T("pages")+"  "),
		cy.Render("←→") + "  ",
	}
	if valueMode {
		pgs = []string{cy.Render("1-5") + txt.Render(" "+r.T("pages")+"  ")}
	}
	qk := cy.Render("q") + txt.Render(" "+r.T("exit")+" ") // UX-R9 (finding 15 (b)): «q exit», in both languages
	if arm == armedQuit || arm == armedQuitReply {
		qk = yeB.Render("q") + yeB.Render(" "+r.T("again: exit")+" ")
	}
	rest := cy.Render("?") + txt.Render(" "+r.T("help")+"  ") + qk
	rt := rest
	for _, pg := range pgs {
		if lipgloss.Width(b.String())+lipgloss.Width(pg+rest)+3 <= r.W {
			rt = pg + rest
			break
		}
	}
	return []string{
		dim.Render(rep("─", r.W)),
		"    " + nsty.Render(n),
		r.spread(b.String(), dim.Render("│ ")+rt),
	}
}

// screen assembles a page and fits it to the window: lines are cut at the
// width and body lines are dropped from the bottom when the window is low
// (decision ⑤: a smaller window shows less, never panics).
func (r rc) screen(nav []string, head string, body []string, footLines []string) string {
	if r.W < fullW || r.H < fullH {
		return r.small(nav, head, body, footLines)
	}
	top := append(append([]string{}, nav...), head, "")
	var b []string
	var kinds []string // the part of each body line
	for _, x := range body {
		k, x := partOf(x)
		for _, l := range strings.Split(x, "\n") {
			b = append(b, l)
			kinds = append(kinds, k)
		}
	}
	room := r.H - len(top) - len(footLines)
	if room < 0 {
		room = 0
	}
	if len(b) > room {
		b = b[:room]
		kinds = kinds[:room]
	}
	r.record(true, kinds, len(footLines) == 3, len(footLines) > 0)
	out := append(top, b...)
	for len(out) < r.H-len(footLines) {
		out = append(out, "")
	}
	out = append(out, footLines...)
	if len(out) > r.H && r.H > 0 {
		// very low window: keep the top rows and the key row
		keep := append(out[:max(0, r.H-1)], out[len(out)-1])
		out = keep
	}
	for i := range out {
		out[i] = cut(out[i], r.W)
	}
	return strings.Join(out, "\n")
}

// ---------- a window below 80x28 (decision ⑤, U-31; GL-04) ----------

// fullW x fullH is the smallest window that shows the whole layout; at and
// above it nothing changes (the plan cell's limit column).
const fullW, fullH = 80, 28

// cutMark ends a cut row or title: … in a full window; below 80x28 the cut
// is bare, so every word shown is a word or the start of a word of the full
// layout (S8-LA-09).
func (r rc) cutMark() string {
	if r.W < fullW || r.H < fullH {
		return ""
	}
	return "…"
}

// The parts of a page that the user named (U-31, L:4264). A body block
// carries its part as a marker prefix, which screen strips.
const (
	partGraph   = "\x00graph\x00"   // the picture: the tile row and the time graphs
	partLog     = "\x00log\x00"     // the log tail
	partVerdict = "\x00verdict\x00" // the start verdict
	partBoxes   = "\x00boxes\x00"   // a row of framed boxes, which leaves as one part (UX-R13)
)

func graphPart(s string) string   { return partGraph + s }
func logPart(s string) string     { return partLog + s }
func verdictPart(s string) string { return partVerdict + s }
func boxesPart(s string) string   { return partBoxes + s }

var partStrip = strings.NewReplacer(partGraph, "", partLog, "", partVerdict, "", partBoxes, "")

// partOf splits a body block into its part ("" for the rest of the body)
// and its text without markers.
func partOf(x string) (string, string) {
	if !strings.Contains(x, "\x00") {
		return "", x
	}
	k := ""
	for _, p := range []struct{ m, k string }{{partGraph, "graph"}, {partLog, "log"}, {partVerdict, "verdict"}, {partBoxes, "boxes"}} {
		if strings.HasPrefix(x, p.m) {
			k = p.k
			break
		}
	}
	return k, partStrip.Replace(x)
}

// Parts names the parts on the last drawn screen, in the order status,
// verdict, help, log, graph, keys (S8-LA-09 reads it).
type Parts []string

func (p Parts) Has(name string) bool {
	for _, x := range p {
		if x == name {
			return true
		}
	}
	return false
}

// record notes which parts the drawn screen holds.
func (r rc) record(status bool, kinds []string, help, keys bool) {
	if r.shown == nil {
		return
	}
	has := map[string]bool{"status": status, "help": help, "keys": keys}
	for _, k := range kinds {
		if k != "" {
			has[k] = true
		}
	}
	var p Parts
	for _, n := range []string{"status", "verdict", "help", "log", "graph", "keys"} {
		if has[n] {
			p = append(p, n)
		}
	}
	*r.shown = p
}

// small draws a page in a window below 80x28: the screen shows less, never a
// single make-larger line. While the page is higher than the window, parts
// leave in this order: the graph, the log tail, the rest of the body line by
// line from the bottom up (a row of framed boxes as one part, never a frame without its
// bottom edge: UX-R13), the help line (the line under the bottom rule), the blank line,
// the page head, the two rules, the key line; the status line (the top row
// with the server state) and the start verdict stay to the end, the verdict
// losing lines from its bottom only when nothing else is left. Lines wider
// than the window are cut at its edge.
func (r rc) small(nav []string, head string, body []string, footLines []string) string {
	type block struct {
		kind  string
		lines []string
		on    bool
	}
	var bs []*block
	for _, x := range body {
		k, x := partOf(x)
		bs = append(bs, &block{kind: k, lines: strings.Split(x, "\n"), on: true})
	}
	status, navRule := "", ""
	if len(nav) > 0 {
		status = nav[0]
	}
	if len(nav) > 1 {
		navRule = nav[1]
	}
	fRule, help, keys := "", "", ""
	fOn := [3]bool{}
	switch len(footLines) {
	case 3:
		fRule, help, keys = footLines[0], footLines[1], footLines[2]
		fOn = [3]bool{true, true, true}
	case 0:
	default:
		keys = footLines[len(footLines)-1]
		fOn[2] = true
	}
	navRuleOn, headOn, blankOn := len(nav) > 1, true, true
	count := func() int {
		n := 1
		for _, on := range []bool{navRuleOn, headOn, blankOn, fOn[0], fOn[1], fOn[2]} {
			if on {
				n++
			}
		}
		for _, b := range bs {
			if b.on {
				n += len(b.lines)
			}
		}
		return n
	}
	over := func() bool { return count() > r.H }
	// 1 the graph, 2 the log tail: each leaves as a whole
	for _, k := range []string{"graph", "log"} {
		if over() {
			for _, b := range bs {
				if b.kind == k {
					b.on = false
				}
			}
		}
	}
	// 3 the rest of the body, line by line from the bottom up; a box row as one part
	for i := len(bs) - 1; i >= 0 && over(); i-- {
		b := bs[i]
		if b.kind == "boxes" {
			b.on = false
			continue
		}
		for b.kind == "" && b.on && over() {
			b.lines = b.lines[:len(b.lines)-1]
			b.on = len(b.lines) > 0
		}
	}
	// 4 the help line, 5 the frame lines, 6 the key line
	for _, off := range []*bool{&fOn[1], &blankOn, &headOn, &navRuleOn, &fOn[0], &fOn[2]} {
		if over() {
			*off = false
		}
	}
	// 7 the start verdict loses lines from its bottom
	for i := len(bs) - 1; i >= 0 && over(); i-- {
		b := bs[i]
		for b.on && len(b.lines) > 0 && over() {
			b.lines = b.lines[:len(b.lines)-1]
		}
	}
	out := []string{status}
	if navRuleOn {
		out = append(out, navRule)
	}
	if headOn {
		out = append(out, head)
	}
	if blankOn {
		out = append(out, "")
	}
	var kinds []string
	for _, b := range bs {
		if b.on && len(b.lines) > 0 {
			out = append(out, b.lines...)
			for range b.lines {
				kinds = append(kinds, b.kind)
			}
		}
	}
	var foot []string
	for i, l := range []string{fRule, help, keys} {
		if fOn[i] {
			foot = append(foot, l)
		}
	}
	for len(out)+len(foot) < r.H {
		out = append(out, "")
	}
	out = append(out, foot...)
	if len(out) > r.H {
		out = out[:max(1, r.H)]
	}
	r.record(true, kinds, fOn[1], fOn[2])
	for i := range out {
		out[i] = cut(out[i], r.W)
	}
	return strings.Join(out, "\n")
}

// ---------- tiles and panels ----------

// tile draws a framed box whose label sits in the top edge.
func (r rc) tile(w int, edge, label string, r0, r1, r2 string) string {
	w = max(w, 6)
	in := w - 4
	rows := []string{r0, r1, r2}
	for i := range rows {
		if lipgloss.Width(rows[i]) > in {
			rows[i] = ansi.Truncate(rows[i], in, "")
		}
	}
	b := st().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color(edge)).
		Padding(0, 1).Width(w - 2).Render(lines(rows...))
	if label == "" {
		return b
	}
	ls := strings.Split(b, "\n")
	ec := fg(edge)
	t := " " + r.T(label) + " "
	// a translated label longer than the edge is cut, never allowed to break the frame
	if room := w - 3; lipgloss.Width(t) > room {
		t = ansi.Truncate(t, max(0, room-1), r.cutMark())
	}
	ls[0] = ec.Render("╭─") + ec.Bold(true).Render(t) + ec.Render(rep("─", w-3-lipgloss.Width(t))+"╮")
	return strings.Join(ls, "\n")
}

// panel is a framed area whose title sits in its own top edge.
func (r rc) panel(w int, edge, title string, rows ...string) string {
	w = max(w, 8)
	rows = append([]string{}, rows...)
	// a row longer than the inside is cut with …, so a long translation never wraps the box
	for i, x := range rows {
		if lipgloss.Width(strings.TrimRight(ansi.Strip(x), " ")) > w-4 {
			rows[i] = ansi.Truncate(x, max(0, w-5), r.cutMark())
		}
	}
	if lipgloss.Width(title) > w-6 {
		title = ansi.Truncate(title, max(0, w-7), r.cutMark())
	}
	b := st().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color(edge)).
		Padding(0, 1).Width(w - 2).Render(lines(rows...))
	ls := strings.Split(b, "\n")
	ec := fg(edge)
	ls[0] = ec.Render("╭─") + " " + title + " " + ec.Render(rep("─", w-5-lipgloss.Width(title))+"╮")
	return strings.Join(ls, "\n")
}

func row(ts ...string) string { return indent(lipgloss.JoinHorizontal(lipgloss.Top, ts...), 1) }

// cols splits the content width into n panel widths.
func (r rc) cols(n int) []int {
	a := r.cw()
	w := make([]int, n)
	for i := range w {
		w[i] = a / n
	}
	w[n-1] = a - (n-1)*(a/n)
	return w
}

// tileFn draws a tile at a given width.
type tileFn func(w int) string

// gauge draws a bar at a given width.
type gauge func(w int) string

func noGauge(int) string { return "" }

// sm is a standard tile: value, gauge, one note.
func (r rc) sm(edge, label, value string, g gauge, note string) tileFn {
	return func(w int) string {
		return r.tile(w, edge, label, value, g(max(0, w-4)), txt.Render(r.T(note)))
	}
}

// heroL is the hero tile with its name in the top edge and a big number.
func (r rc) heroL(edge, label, n string, sty lipgloss.Style, unit, mark string) tileFn {
	return func(w int) string {
		b := big(n, sty)
		if bw := lipgloss.Width(b[0]); n != "" && bw+2+lipgloss.Width(unit) > w-4 && bw+1+lipgloss.Width(unit) <= w-4 {
			// UX-R12 (UX-R9 landing 4.1, the latent seam): too narrow for the unit beside the
			// big digits but not for the digits: the digits stay big and the unit goes under
			// them, so a narrow window shows the number in the form the wider one does
			return r.tile(w, edge, label, b[0], b[1], b[2]+" "+hiB.Render(unit))
		}
		if n == "" || lipgloss.Width(b[0])+2+lipgloss.Width(unit) > w-4 {
			// too narrow for the big digits, or no number: plain text
			v := n
			if v == "" {
				v = "—"
			}
			return r.tile(w, edge, label, sty.Render(v)+" "+hiB.Render(unit), "", mark)
		}
		return r.tile(w, edge, label, b[0]+"  "+hiB.Render(unit), b[1], b[2]+"  "+mark)
	}
}

// tiles lays out a hero tile and up to three small tiles across the content
// width. The design's split is 3/10 for the hero; when the window is too
// narrow for a small tile of 16 cells, small tiles drop from the right.
func (r rc) tiles(hero tileFn, smalls ...tileFn) string {
	a := r.cw()
	n := len(smalls)
	const minSmall, minHero = 16, 22
	for n > 0 && (a-minHero)/n < minSmall {
		n--
	}
	if n == 0 {
		return graphPart(row(hero(a)))
	}
	var hw, tw int
	if n == len(smalls) {
		hw = a * 3 / 10
		tw = (a - hw) / n
	} else {
		hw = max(minHero, a*3/10)
		tw = (a - hw) / n
	}
	hw = a - n*tw
	parts := []string{hero(hw)}
	for _, s := range smalls[:n] {
		parts = append(parts, s(tw))
	}
	return graphPart(row(parts...)) // the tile row is the page's picture (U-31: the graph)
}

// alert is a marked line under the boxes; a long one wraps to the next line,
// the text continuing under its first word (review item 32).
func (r rc) alert(sym string, sty lipgloss.Style, msg string) string {
	ls := marked(r.W-3, sym, sty.Bold(true), sty, msg)
	for i := range ls {
		ls[i] = "  " + ls[i]
	}
	return strings.Join(ls, "\n")
}

// marked is a mark and a text wrapped at w cells; continuation lines start
// under the text (review item 32: warning and refusal lines wrap, not cut).
func marked(w int, mark string, msty, tsty lipgloss.Style, msg string) []string {
	mw := lipgloss.Width(mark) + 1
	ls := wrapWords(msg, w-mw)
	out := make([]string, len(ls))
	for i, l := range ls {
		head := strings.Repeat(" ", mw)
		if i == 0 {
			head = msty.Render(mark) + " "
		}
		out[i] = head + tsty.Render(l)
	}
	return out
}

// wrapWords breaks s at spaces into lines of at most w cells; a word longer
// than w (a path) is broken by cells.
func wrapWords(s string, w int) []string {
	if w <= 0 {
		return []string{s}
	}
	var out []string
	cur := ""
	for _, word := range strings.Fields(s) {
		for lipgloss.Width(word) > w {
			if cur != "" {
				out = append(out, cur)
				cur = ""
			}
			cs := wrapCells(word, w)
			out = append(out, cs[:len(cs)-1]...)
			word = cs[len(cs)-1]
		}
		switch {
		case cur == "":
			cur = word
		case lipgloss.Width(cur)+1+lipgloss.Width(word) <= w:
			cur += " " + word
		default:
			out = append(out, cur)
			cur = word
		}
	}
	return append(out, cur)
}

// bins averages the samples in w equal time spans from from to to; a span
// without a sample is NaN (review item 21: a column covers its share of the
// time between the two end labels).
func bins(ss []sample, from, to time.Time, w int) []float64 {
	out := make([]float64, max(w, 0))
	if w <= 0 {
		return out
	}
	sum := make([]float64, w)
	n := make([]int, w)
	span := to.Sub(from)
	for _, s := range ss {
		if s.at.Before(from) || s.at.After(to) {
			continue
		}
		i := w - 1
		if span > 0 {
			i = min(w-1, int(float64(s.at.Sub(from))/float64(span)*float64(w)))
		}
		sum[i] += s.v
		n[i]++
	}
	for i := range out {
		out[i] = math.NaN()
		if n[i] > 0 {
			out[i] = sum[i] / float64(n[i])
		}
	}
	return out
}

// timeGraph draws decode samples between from and to across w cells: a
// vertical scale from 0 to the highest column, three rows of columns, and
// an axis line with the end labels and «average n · lowest n tok/s» (review
// item 21).
func (r rc) timeGraph(ss []sample, from, to time.Time, w int, sty lipgloss.Style, left, right string) []string {
	var in []float64
	for _, s := range ss {
		if !s.at.Before(from) && !s.at.After(to) {
			in = append(in, s.v)
		}
	}
	top, low, sum := 0.0, math.Inf(1), 0.0
	for _, v := range in {
		top, low, sum = math.Max(top, v), math.Min(low, v), sum+v
	}
	topS := f1(top)
	sw := lipgloss.Width(topS) + 1
	cols := max(1, w-sw)
	b := bins(ss, from, to, cols)
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
	mid := ""
	if len(in) > 0 {
		mid = r.Tf("average %s · lowest %s tok/s", f1(sum/float64(len(in))), f1(low))
	}
	lw, rw, mw := lipgloss.Width(left), lipgloss.Width(right), lipgloss.Width(mid)
	gap := cols - lw - rw
	axis := rep(" ", sw) + dim.Render(left)
	if mid != "" && gap >= mw+2 {
		a := (gap - mw) / 2
		axis += rep(" ", a) + txt.Render(mid) + rep(" ", gap-mw-a)
	} else {
		axis += rep(" ", max(1, gap))
	}
	return append(out, axis+dim.Render(right))
}

// midPath shortens a path in the middle to w cells, keeping the start and
// as much of the end as fits: C:\Users\me\…\MoE-Direct\logs (review item 24).
func midPath(p string, w int) string {
	if lipgloss.Width(p) <= w || w <= 0 {
		return p
	}
	sep := `\`
	if !strings.Contains(p, sep) {
		sep = "/"
	}
	parts := strings.Split(p, sep)
	for head := min(3, len(parts)-1); head >= 1; head-- {
		for tail := len(parts) - head; tail >= 1; tail-- {
			s := strings.Join(parts[:head], sep) + sep + "…" + sep + strings.Join(parts[len(parts)-tail:], sep)
			if lipgloss.Width(s) <= w {
				return s
			}
		}
	}
	// even the last part does not fit: keep its end
	rs := []rune(p)
	for i := range rs {
		if s := "…" + string(rs[i:]); lipgloss.Width(s) <= w {
			return s
		}
	}
	return ""
}

// wrapCells breaks plain text into lines of at most w cells (a hard wrap,
// as a console does; review item 7).
func wrapCells(s string, w int) []string {
	if w <= 0 {
		return []string{""}
	}
	var out []string
	var b strings.Builder
	n := 0
	for _, c := range s {
		cw := lipgloss.Width(string(c))
		if n+cw > w && n > 0 {
			out = append(out, b.String())
			b.Reset()
			n = 0
		}
		b.WriteRune(c)
		n += cw
	}
	return append(out, b.String())
}

// steps draws a step chain: ✓ done  ▶ CURRENT  ○ todo.
func (r rc) chain(ss []Step) string {
	var p []string
	for _, s := range ss {
		name := r.T(s.Name)
		switch s.State {
		case StepDone, StepSkipped:
			p = append(p, gr.Render("✓ "+name))
		case StepRunning:
			p = append(p, pill("▶ "+strings.ToUpper(name), "11"))
		case StepFailed:
			p = append(p, pill("✕ "+strings.ToUpper(name), "9"))
		case StepNotPorted:
			if showNotPorted {
				p = append(p, dim.Render("· "+name+" ("+r.np()+")"))
			} else { // UX-R12: the step without the not-ported word
				p = append(p, dim.Render("· "+name))
			}
		default:
			p = append(p, dim.Render("○ "+name))
		}
	}
	return "  " + strings.Join(p, "  ")
}
