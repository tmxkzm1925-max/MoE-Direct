// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// 3 Memory: H10 while serving, H14 before a start. From the mock's m.go.

package ui

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

var (
	pModel = func(g float64) seg { return seg{g, "█", cy} }
	pApps  = func(g float64) seg { return seg{g, "▓", txt} }
	pMod   = func(g float64) seg { return seg{g, "▒", ye} }
	pStby  = func(g float64) seg { return seg{g, "░", bl} }
)

// UX-R16 (finding 26): the VRAM bar has colours of its own, magenta, apart from every RAM segment
// colour (cyan █, grey ▓, yellow ▒, blue ░), so a VRAM bar is never read as the model's RAM:
// this server's share bright magenta (13), the rest of the card's use magenta (5).
var (
	vSrv  = fg("13")
	vOth  = fg("5")
	pVSrv = func(g float64) seg { return seg{g, "█", vSrv} }
	pVOth = func(g float64) seg { return seg{g, "▓", vOth} }
)

// compactNote is a term note with the spaces around − and + and before «GB» dropped (UX-R16b).
func compactNote(s string) string {
	return strings.NewReplacer(" − ", "−", " + ", "+", " GB", "GB").Replace(s)
}

// fitNote is the first of the forms that fits w cells; where none does, the last, cut to w back
// to its last whole term, so a cut never leaves part of an amount or a bare «G» (UX-R7, UX-R16b).
func fitNote(w int, forms ...string) string {
	for _, f := range forms {
		if lipgloss.Width(f) <= w {
			return f
		}
	}
	x := ansi.Truncate(forms[len(forms)-1], max(0, w), "")
	if i := strings.LastIndexAny(x, " +−"); i > 0 {
		x = x[:i]
	}
	return strings.TrimRight(x, " ·")
}

// gbv is an amount with its unit «GB» (UX-R7), or the dash when it is not known (UX-R16).
func gbv(v Val) string {
	if v.Ok() {
		return gstr(v.V)
	}
	return "—"
}

// packed joins items with sep on as few lines as the width w allows; a line after the first
// starts with ind. An item wider than a line stands alone (the panel cuts it with …).
func packed(w int, sep, ind string, items ...string) []string {
	var out []string
	cur := ""
	for _, it := range items {
		switch {
		case cur == "":
			cur = it
		case lipgloss.Width(cur+sep+it) <= w:
			cur += sep + it
		default:
			out = append(out, cur)
			cur = ind + it
		}
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

// keyLines is the key of a bar, under the bar (UX-R16): one line, three spaces between the keys,
// where it fits the panel, else two spaces and as many lines as the keys need («GB» on every
// amount makes the RAM key wider than 80 columns).
func (r rc) keyLines(items ...string) []string {
	room, ind := r.cw()-4-barLabel, rep(" ", barLabel)
	if one := strings.Join(items, "   "); lipgloss.Width(one) <= room {
		return []string{ind + one}
	}
	ls := packed(room, "  ", "", items...)
	for i := range ls {
		ls[i] = ind + ls[i]
	}
	return ls
}

const barLabel = 7 // "  RAM  " in front of a bar

func (r rc) memBarW() int { return max(4, r.cw()-4-barLabel-10) }

func (r rc) memLine(name string, total float64, tail string, parts ...seg) string {
	return txt.Render(pad(r.T(name), barLabel)) + mbar(r.memBarW(), total, parts...) + "  " + tail
}

func memKey(glyph string, s lipgloss.Style, name, v string) string {
	return s.Render(glyph) + " " + dim.Render(name+" ") + hi.Render(v)
}

// vramKeys is the VRAM bar's own key (UX-R16): this server's share (GPU process counter), the
// rest of the card's use (nvidia-smi) and what is free, in the bar's colours; the two counters
// stay apart, never added (review item 4). UX-R22 (item 33): three parts, this server / other
// programs / free; the card's use is no part of its own beside them, only the label of their sum
// («this server + other programs = in use»), where the line has room. Narrower, the sum label
// leaves first, then «other programs» shortens to «other», then what is free leaves (the bar's
// dots still show it), so the panel keeps its height at 80x28.
func (r rc) vramKeys(inUse, srv, oth, free Val) []string {
	room := r.cw() - 4 - barLabel
	s := memKey("█", vSrv, r.T("this server"), gbv(srv))
	o := func(name string) string { return memKey("▓", vOth, r.T(name), gbv(oth)) }
	f := memKey("·", dim, r.T("free"), gbv(free))
	forms := [][]string{{s, o("other programs"), f}, {s, o("other"), f}, {s, o("other")}}
	if inUse.Ok() {
		sum := s + dim.Render(" + ") + o("other programs") + dim.Render(" = "+r.T("in use")+" ") + hi.Render(gstr(inUse.V))
		forms = append([][]string{{sum, f}}, forms...)
	}
	for _, ks := range forms {
		if lipgloss.Width(strings.Join(ks, "   ")) <= room {
			return r.keyLines(ks...)
		}
	}
	return r.keyLines(s, o("other"), f)
}

// parts keeps the known values only.
func parts(ps ...struct {
	v Val
	f func(float64) seg
}) []seg {
	var out []seg
	for _, p := range ps {
		if p.v.Ok() {
			out = append(out, p.f(p.v.V))
		}
	}
	return out
}

type pv = struct {
	v Val
	f func(float64) seg
}

func (m *Model) launcherCost() string {
	r := m.r
	if m.mem == nil || !m.mem.LauncherGB.Ok() {
		return ""
	}
	l := m.mem
	every := ""
	if l.EveryS.Ok() {
		every = fmt.Sprintf("%s %.0f s · ", r.T("every"), l.EveryS.V)
	}
	cpu := ""
	if l.LauncherCPU.Ok() {
		cpu = fmt.Sprintf(", %.0f %% CPU", l.LauncherCPU.V)
	}
	return every + r.T("this launcher") + " " + gstr(l.LauncherGB.V) + cpu
}

// vH10: memory while serving.
func (m *Model) vH10() string {
	r := m.r
	a := r.cw()
	mm := m.mem
	if mm == nil {
		return m.page(plMemory, r.pageHead(r.T("memory while serving"), ""), []string{"  " + dim.Render(r.T("reading memory …"))},
			foot{next: []string{"apps are not closed here · o opens Task Manager"}, here: []string{"↑↓", "app", "o", "Task Manager"}})
	}
	commitLeft := Val{State: mm.CommitLimitGB.State}
	if mm.CommitLimitGB.Ok() && mm.CommitUsedGB.Ok() {
		commitLeft = K(mm.CommitLimitGB.V - mm.CommitUsedGB.V)
	}
	clEdge, clSty := "8", hiB
	if commitLeft.Ok() && mm.CommitSafeGB.Ok() && commitLeft.V < mm.CommitSafeGB.V {
		clEdge, clSty = "11", yeB
	}
	safe, safeS := "", "" // UX-R16: the safe line with its unit, and its short form for a narrow tile
	if mm.CommitSafeGB.Ok() {
		safe, safeS = r.Tf("safe ≥ %s", g0str(mm.CommitSafeGB.V)), "≥ "+g0str(mm.CommitSafeGB.V)
	}
	pfNote := mm.PagefileNote
	commitVal := dim.Render(r.svt(mm.CommitUsedGB, gstr))
	commitShort, commitNote := commitVal, r.T("RAM+pagefile")
	if mm.CommitUsedGB.Ok() && mm.CommitLimitGB.Ok() {
		commitVal = hiB.Render(gstr(mm.CommitUsedGB.V)) + dim.Render(r.Tf(" of %s", gstr(mm.CommitLimitGB.V)))
		commitShort = hiB.Render(gstr(mm.CommitUsedGB.V))
	}
	// UX-R3 (findings 4, 6): commit left below zero is the excess over the limit, its baseline
	// named, the safe line beside it
	clN := numOr(commitLeft)
	clMarks := []string{dim.Render(safe), dim.Render(safeS)}
	if !commitLeft.Ok() {
		clMarks = []string{dim.Render(r.svt(commitLeft, f1))}
	}
	if commitLeft.Ok() && commitLeft.V < 0 {
		clEdge, clSty = "9", rdB
		clN = numOr(K(-commitLeft.V))
		over := rdB.Render(r.T("over the limit"))
		clMarks = []string{over + dim.Render(" · "+safe), over + dim.Render(" · "+safeS), over}
	}
	tiles := r.tiles(
		func(w int) string { // UX-R16: the first mark that fits the tile, so no amount loses its unit
			in, bw := w-4, lipgloss.Width(big(clN, clSty)[0])
			mk := clMarks[len(clMarks)-1]
			for _, x := range clMarks {
				need := bw + 2 + lipgloss.Width(x)
				if clN == "" || bw+2+lipgloss.Width("GB") > in { // the plain form: the mark on its own row
					need = lipgloss.Width(x)
				}
				if need <= in {
					mk = x
					break
				}
			}
			return r.heroL(clEdge, "COMMIT LEFT", clN, clSty, "GB", mk)(w)
		},
		func(w int) string { // UX-R16: used and limit each with «GB»; where both do not fit, the limit in the note
			v, note := commitVal, commitNote
			if lipgloss.Width(v) > w-4 && mm.CommitLimitGB.Ok() {
				v, note = commitShort, strings.TrimSpace(r.Tf(" of %s", gstr(mm.CommitLimitGB.V)))
			}
			g := ""
			if mm.CommitUsedGB.Ok() && mm.CommitLimitGB.Ok() && mm.CommitLimitGB.V > 0 {
				g = use(mm.CommitUsedGB.V/mm.CommitLimitGB.V, max(0, w-4))
			}
			return r.tile(w, "8", "COMMIT", v, g, txt.Render(note))
		},
		r.sm("8", "PAGEFILE", r.svu(mm.PagefileMBs, f0, " MB/s"), noGauge, pfNote),
		r.ramFreeTile(mm),
	)
	// UX-R16 (finding 26): each bar with its own key under it, the VRAM bar in its own colours;
	// the model's breakdown is the bottom line; every amount with «GB»
	var mem []string
	if mm.TotalRAMGB.Ok() {
		mem = append(mem, r.memLine("RAM", mm.TotalRAMGB.V, dim.Render(g0str(mm.TotalRAMGB.V)),
			parts(pv{mm.ModelGB, pModel}, pv{mm.AppsGB, pApps}, pv{mm.ModifiedGB, pMod}, pv{mm.StandbyGB, pStby})...))
	} else {
		mem = append(mem, txt.Render(pad("RAM", barLabel))+dim.Render(r.snp()))
	}
	mem = append(mem, r.keyLines(memKey("█", cy, r.T("model"), gbv(mm.ModelGB)), memKey("▓", txt, r.T("apps"), gbv(mm.AppsGB)),
		memKey("▒", ye, r.T("modified"), gbv(mm.ModifiedGB)), memKey("░", bl, r.T("standby"), gbv(mm.StandbyGB)),
		memKey("·", dim, r.T("free"), gbv(mm.FreeGB)))...)
	switch {
	case mm.TotalVRAMGB.Ok() && mm.UsedVRAMGB.Ok():
		// the card's use (nvidia-smi) with this server's share (GPU process
		// counter) drawn inside it: two counters, never added (review item 4)
		sv, srv := 0.0, mm.ModelVRAMGB
		if mm.ModelVRAMGB.Ok() {
			sv = math.Min(mm.ModelVRAMGB.V, mm.UsedVRAMGB.V)
			srv = K(sv)
		}
		mem = append(mem, r.memLine("VRAM", mm.TotalVRAMGB.V, dim.Render(g0str(mm.TotalVRAMGB.V)),
			pVSrv(sv), pVOth(mm.UsedVRAMGB.V-sv)))
		mem = append(mem, r.vramKeys(mm.UsedVRAMGB, srv, K(mm.UsedVRAMGB.V-sv), K(math.Max(0, mm.TotalVRAMGB.V-mm.UsedVRAMGB.V)))...)
	case mm.TotalVRAMGB.Ok():
		mem = append(mem, r.memLine("VRAM", mm.TotalVRAMGB.V, dim.Render(g0str(mm.TotalVRAMGB.V)),
			parts(pv{mm.ModelVRAMGB, pVSrv}, pv{mm.OtherVRAMGB, pVOth})...))
		free := Val{State: mm.OtherVRAMGB.State}
		if mm.ModelVRAMGB.Ok() && mm.OtherVRAMGB.Ok() {
			free = K(math.Max(0, mm.TotalVRAMGB.V-mm.ModelVRAMGB.V-mm.OtherVRAMGB.V))
		}
		mem = append(mem, r.vramKeys(mm.UsedVRAMGB, mm.ModelVRAMGB, mm.OtherVRAMGB, free)...)
	default:
		mem = append(mem, txt.Render(pad("VRAM", barLabel))+dim.Render(r.snp()))
	}
	if mm.ModelGB.Ok() {
		head := dim.Render(r.T("model") + " " + gstr(mm.ModelGB.V))
		var ps []string
		if mm.ExpertCacheGB.Ok() {
			// «expert cache 12.0 GB (budget 12 GB)», in Korean «전문가 캐시 12.0 GB (예산 할당 12 GB)»
			h := head + dim.Render(" = ") + txt.Render(r.T("expert cache")+" "+gstr(mm.ExpertCacheGB.V))
			if mm.BudgetGB.Ok() {
				h += dim.Render(" " + r.Tf("(budget %s)", g0str(mm.BudgetGB.V)))
			}
			ps = append(ps, h)
			if mm.FileMapGB.State != NotPorted { // GL-06: «not measured yet» (U-20 (i))
				ps = append(ps, txt.Render(r.T("file mapping")+" "+r.vt(mm.FileMapGB, gstr)))
			}
			if mm.OtherGB.Ok() {
				ps = append(ps, txt.Render(r.T("other")+" "+gstr(mm.OtherGB.V)))
			}
		} else {
			ps = append(ps, head)
			if mm.ExpertCacheGB.State != NotPorted || showNotPorted { // GL-06: «not measured yet» once ported
				ps = append(ps, dim.Render(r.T("expert cache split")+": "+r.vt(mm.ExpertCacheGB, gstr)))
			}
		}
		mem = append(mem, packed(a-4, dim.Render(" · "), "  ", ps...)...)
	}
	hdr := "  " + pad(r.T("app"), 16) + lpad(r.T("memory"), 8) + "  " + pad("", 13) + pad(r.T("last 10 min"), 13)
	rs := []string{dim.Render(hdr)}
	if len(mm.Apps) == 0 {
		rs = append(rs, dim.Render("  "+r.T("app list")+": "+r.snp()))
	}
	for i, x := range mm.Apps {
		mk, nm := "  ", txt.Render(pad(x.Name, 16))
		if i == m.appCur {
			mk, nm = cyB.Render("▶ "), hiB.Render(pad(x.Name, 16))
		}
		ch := dim.Render(pad("—", 13))
		if x.ChangeGB.Ok() && x.ChangeGB.V > 0.05 {
			ch = yeB.Render(pad(fmt.Sprintf("▲ +%.1f GB", x.ChangeGB.V), 13))
		}
		g := 0.0
		if x.GB.Ok() && mm.TotalRAMGB.Ok() && mm.TotalRAMGB.V > 0 {
			g = x.GB.V / mm.TotalRAMGB.V
		}
		l := mk + nm + hi.Render(lpad(r.svt(x.GB, gstr), 8)) + "  " + bar(g, 12, txt) + " " + ch
		if x.Tag != "" {
			l += dim.Render(r.T(x.Tag))
		}
		rs = append(rs, l)
	}
	if mm.AppsMore > 0 {
		rs = append(rs, dim.Render(fmt.Sprintf("  + %d %s", mm.AppsMore, r.T("more"))))
	}
	var bottom []string
	causes := mm.Causes
	why := []string{}
	for i, x := range causes {
		why = append(why, yeB.Render(fmt.Sprintf("%d ", i+1))+txt.Render(x))
	}
	if len(why) == 0 && showNotPorted {
		why = []string{dim.Render(r.T("cause rules") + ": " + r.np())}
	}
	// UX-R16 (findings 8, 10, 26): with the switch off the cause box holding only the not-ported
	// cause rules is not drawn, and the apps box takes the row
	if r.wide() && len(why) > 0 {
		c := r.cols(3)
		bottom = []string{row(r.panel(c[0]+c[1], "8", dim.Render(r.T("APPS · by memory")), rs...),
			r.panel(c[2], "11", yeB.Render(r.T("LIKELY CAUSE"))+dim.Render(" · "+r.T("not proven")), why...))}
	} else {
		bottom = []string{indent(r.panel(a, "8", dim.Render(r.T("APPS · by memory")), rs...), 1)}
		if len(causes) > 0 {
			bottom = append(bottom, r.alert("▲", ye, r.T("likely, not proven")+": "+causes[0]))
		}
	}
	body := append([]string{tiles, "", indent(r.panel(a, "8", dim.Render(r.T("MEMORY · now")), mem...), 1), ""}, bottom...)
	head := r.T("memory while serving")
	if m.status != nil && m.status.Slow {
		head = r.T("what changed when it got slow")
	}
	return m.page(plMemory, r.pageHead(head, m.launcherCost()), body, foot{next: []string{
		"apps are not closed here · o opens Task Manager",
	}, here: []string{"↑↓", "app", "o", "Task Manager"}})
}

// ramFreeTile is «RAM free (now)» / «RAM 여유(지금)» (UX-R16b; finding 26, the user's answer ⑥):
// the RAM available now, its total figure kept, with the free list and the standby list apart
// (their bar segments in the RAM key's colours) and standby named in a few words where the tile
// has room. Free and standby apart need the memory lists: without them FreeGB is the whole
// available figure, so both read as their state (not measured), never as a figure or zero.
func (r rc) ramFreeTile(mm *MemoryReading) tileFn {
	return func(w int) string {
		in := max(0, w-4)
		label := "RAM free (now)"
		if lipgloss.Width(" "+r.T(label)+" ") > w-3 {
			label = "RAM free"
		}
		total := hiB.Render(r.svt(mm.ReadyNowGB, gstr))
		first := func(xs ...string) string {
			for _, x := range xs {
				if lipgloss.Width(x) <= in {
					return x
				}
			}
			return xs[len(xs)-1]
		}
		if mm.ReadyNowGB.Ok() { // UX-R22 (item 33): the sum is the RAM usable now
			total = first(total+dim.Render(" "+r.T("usable now")), total)
		}
		if !mm.StandbyGB.Ok() || !mm.FreeGB.Ok() {
			// without the lists FreeGB is the whole available figure: free and standby are named
			// with the state of the missing one (not measured, or the dash), never a figure or zero
			st := Val{State: mm.StandbyGB.State}
			if mm.StandbyGB.Ok() {
				st = Val{State: mm.FreeGB.State}
			}
			word := r.svt(st, gstr)
			if st.State == Unmeasured {
				word = first(word, r.T("not measured"))
			}
			names := first(ye.Render(r.T("free"))+dim.Render(" · ")+bl.Render(r.T("standby")), bl.Render(r.T("standby")))
			return r.tile(w, "8", label, total, names, hi.Render(word))
		}
		// UX-R22 (item 33): one row per figure, each with what it is in plain words where the tile
		// has room: free = the RAM empty now, standby = the cache Windows gives up on demand; the
		// bar is the MEMORY panel's (its ░ standby and · free segments)
		free := ye.Render(r.T("free")) + " " + hi.Render(gbv(mm.FreeGB))
		stby := bl.Render(r.T("standby")) + " " + hi.Render(gbv(mm.StandbyGB))
		stbyC := bl.Render(r.T("standby")) + " " + hi.Render(strings.ReplaceAll(gbv(mm.StandbyGB), " GB", "GB"))
		eq, plus := dim.Render("= "), dim.Render("+ ")
		why := func(s string) string { return dim.Render(" · " + r.T(s)) }
		r1 := first(eq+free+why("RAM empty now"), eq+free+why("empty RAM"), eq+free+why("empty"), eq+free, free)
		r2 := first(plus+stby+why("cache Windows gives up on demand"), plus+stby+why("given up on demand"), plus+stby+why("cache"),
			stby+why("cache"), plus+stby, stby, stbyC)
		return r.tile(w, "8", label, total, r1, r2)
	}
}

// leftTile is «LEFT AFTER» (UX-R3 finding 5): what is left after the start, of what and after
// what. The value is «n G left» (below zero «n G short»), with the room line beside it where the
// tile is wide enough; the bar is the memory available now split into the need and what is
// left; the term line is «free F − need N» (Left = available now − the need, GL-05).
func (r rc) leftTile(edge string, sty lipgloss.Style, p *Preflight) tileFn {
	return func(w int) string {
		in := max(0, w-4)
		room := ""
		if p.RoomLineGB.Ok() {
			room = r.T("room line") + " " + g0str(p.RoomLineGB.V)
		}
		val := sty.Render(r.svt(p.LeftGB, gstr))
		if p.LeftGB.Ok() {
			if p.LeftGB.V < 0 {
				val = sty.Render(r.Tf("%s short", gstr(-p.LeftGB.V)))
			} else {
				val = sty.Render(r.Tf("%s left", gstr(p.LeftGB.V)))
				if room != "" && lipgloss.Width(val+" · "+room) <= in {
					val += dim.Render(" · " + room)
				}
			}
		}
		gauge, note := "", room
		if p.AvailGB.Ok() && p.NeedGB.Ok() && p.AvailGB.V > 0 {
			switch {
			case p.LeftGB.Ok() && p.LeftGB.V < 0:
				gauge = bar(1, in, rd) // the need is more than the whole bar
			default:
				lc := gr
				if p.LeftGB.Ok() && p.RoomLineGB.Ok() && p.LeftGB.V < p.RoomLineGB.V {
					lc = ye
				}
				gauge = stack(in, seg{p.NeedGB.V / p.AvailGB.V, "▓", txt}, seg{1 - p.NeedGB.V/p.AvailGB.V, "█", lc})
			}
			// UX-R16b: «GB» on every amount; a narrow tile takes a closer-set form before any cut
			full := r.Tf("free %s − need %s", gstr(p.AvailGB.V), gstr(p.NeedGB.V))
			note = fitNote(in, full, compactNote(full), strings.ReplaceAll(compactNote(full), " ", ""))
		}
		return r.tile(w, edge, "LEFT AFTER", val, gauge, txt.Render(note))
	}
}

// vH14: can the chosen model start now? Available = free + standby; the
// need and the verdict come from the backend's preflight (decision ②, made in
// GOL-CORE). A refused preflight offers no start (conflict 7).
func (m *Model) vH14() string {
	r := m.r
	a := r.cw()
	row, ok := m.selRow()
	name := row.Name
	if !ok {
		name = "—"
	}
	p := m.currentPreflight(row.ID)
	headT := r.Tf("%s start checklist", name) // UX-R3 (finding 7): the box is the start checklist
	if p == nil {
		msg := r.T("checking …")
		if !m.preBusy {
			msg = r.T("no check yet · 1 Models picks a model")
		}
		if m.notice != "" {
			msg = m.notice
		}
		return m.page(plMemory, r.pageHead(headT, ""), []string{"  " + dim.Render(msg)}, foot{next: []string{"the check runs when this page opens"}})
	}
	need := p.NeedGB
	cache := p.CacheGB
	leftEdge, leftSty := "8", hiB
	if p.LeftGB.Ok() && p.RoomLineGB.Ok() && p.LeftGB.V < p.RoomLineGB.V {
		leftEdge, leftSty = "11", yeB
	}
	if p.LeftGB.Ok() && p.LeftGB.V < 0 {
		leftEdge, leftSty = "9", rdB
	}
	room := ""
	if p.RoomLineGB.Ok() {
		room = fmt.Sprintf("%s %.0f GB", r.T("room line"), p.RoomLineGB.V)
	}
	safe := ""
	if p.CommitSafeGB.Ok() {
		safe = fmt.Sprintf("%s %.0f GB", r.T("safe line"), p.CommitSafeGB.V)
	}
	needNote := ""
	if p.ModelGB.Ok() && p.ReplyGB.Ok() { // UX-R16b: «GB» on every amount
		needNote = fmt.Sprintf("%s %s + %s %s", r.T("model"), gstr(p.ModelGB.V), gstr(p.ReplyGB.V), r.T("reply"))
	} else if need.State == Unmeasured {
		needNote = r.T("first run of these settings")
	}
	needAlone := needNote // the note without the need's label, for a narrow tile
	// GL-05: the need's label stands first in the note, whole on this page when 1 Models'
	// RAM tile is too narrow for it (S8-LA-13; F-BU B.4 Label)
	if p.NeedLabel != "" && need.Ok() {
		if needNote != "" {
			needNote = r.T(p.NeedLabel) + " · " + needNote
		} else {
			needNote = r.T(p.NeedLabel)
		}
	}
	// UX-R3 (finding 4): commit left below zero is shown as the excess over the limit, its
	// baseline named, with the safe line beside it on the tile
	clEdge, clVal := "8", r.svu(p.CommitLeftAfterGB, gstr, " "+r.T("after"))
	if p.CommitLeftAfterGB.Ok() && p.CommitLeftAfterGB.V < 0 {
		clEdge, clVal = "9", rdB.Render(r.Tf("%s over the limit", gstr(-p.CommitLeftAfterGB.V)))
	}
	tiles := r.tiles(
		r.heroL("11", "AVAILABLE NOW", numOr(p.AvailGB), yeB, "GB", ""),
		func(w int) string { // UX-R16b: the note in the first form that fits, never a cut «GB»
			return r.sm("8", "NEEDS", hiB.Render(r.svt(need, gstr)), func(w int) string {
				if need.Ok() && p.AvailGB.Ok() && p.AvailGB.V > 0 {
					c := 0.0
					if cache.Ok() {
						c = cache.V
					}
					return stack(w, seg{c / p.AvailGB.V, "█", cy}, seg{(need.V - c) / p.AvailGB.V, "▓", txt})
				}
				return ""
			}, fitNote(w-4, needNote, compactNote(needNote), needAlone, compactNote(needAlone)))(w)
		},
		r.leftTile(leftEdge, leftSty, p),
		r.sm(clEdge, "COMMIT LEFT", clVal, func(w int) string {
			if p.CommitLeftAfterGB.Ok() && p.CommitLimitGB.Ok() && p.CommitLimitGB.V > 0 {
				return use(1-p.CommitLeftAfterGB.V/p.CommitLimitGB.V, w)
			}
			return ""
		}, safe),
	)
	var mem []string
	if p.RAMTotalGB.Ok() {
		bw := r.memBarW()
		mem = append(mem, r.memLine("RAM", p.RAMTotalGB.V, dim.Render(g0str(p.RAMTotalGB.V)),
			parts(pv{p.AppsGB, pApps}, pv{p.ModifiedGB, pMod}, pv{p.StandbyGB, pStby})...))
		if p.AppsGB.Ok() && p.ModifiedGB.Ok() && p.AvailGB.Ok() && p.RAMTotalGB.V > 0 {
			from := int((p.AppsGB.V+p.ModifiedGB.V)/p.RAMTotalGB.V*float64(bw) + 0.5)
			span := bw - from
			lab := " " + r.T("available") + " " + gstr(p.AvailGB.V) + " = " + r.T("standby + free") + " "
			if span-2 < lipgloss.Width(lab)+2 {
				lab = " " + gstr(p.AvailGB.V) + " " + r.T("available") + " "
			}
			fl := span - 2 - lipgloss.Width(lab)
			if fl >= 0 {
				mem = append(mem, rep(" ", barLabel+from)+yeD.Render("└"+rep("─", fl/2))+ye.Render(lab)+yeD.Render(rep("─", fl-fl/2)+"┘"))
			}
		}
	} else {
		mem = append(mem, txt.Render(pad("RAM", barLabel))+dim.Render(r.T("memory lists")+": "+r.snp()))
	}
	if p.VRAMTotalGB.Ok() { // UX-R16: the card's use in the VRAM bar's own colour
		mem = append(mem, r.memLine("VRAM", p.VRAMTotalGB.V, dim.Render(g0str(p.VRAMTotalGB.V)), parts(pv{p.VRAMUseGB, pVOth})...))
	} else {
		mem = append(mem, txt.Render(pad("VRAM", barLabel))+dim.Render(r.snp()))
	}
	mem = append(mem, r.keyLines(memKey("▓", txt, r.T("apps"), gbv(p.AppsGB)), memKey("▒", ye, r.T("modified"), gbv(p.ModifiedGB)),
		memKey("░", bl, r.T("standby"), gbv(p.StandbyGB)), memKey("·", dim, r.T("free"), gbv(p.FreeGB)))...)

	roomWhy := r.T("enough left after the start")
	if room != "" {
		roomWhy = room + " " + r.T("or more left after the start")
	}
	type vr struct {
		v         Verdict
		what, why string
	}
	vs := []vr{
		{VerdictRoom, r.T("starts with room"), roomWhy},
		{VerdictTight, r.T("starts, but tight"), r.T("apps may be paged out to disk")},
		{VerdictCannot, r.T("cannot start"), r.T("less available than it needs")},
	}
	var rs []string
	switch p.Verdict {
	case VerdictUnmeasured:
		rs = append(rs, cyB.Render("▶ ")+yeB.Render("○ "+pad(r.T("need not measured yet"), 22))+txt.Render(r.T("no earlier run of these settings")))
	case VerdictNotPorted:
		if showNotPorted { // UX-R16: with the switch off the not-ported verdict row is not drawn
			rs = append(rs, "  "+dim.Render(r.T("verdict")+": "+r.np()))
		}
	default:
		for _, x := range vs {
			why := x.why
			if !r.wide() {
				why = strings.SplitN(why, " · ", 2)[0]
			}
			if x.v == p.Verdict {
				rs = append(rs, cyB.Render("▶ ")+yeB.Render("● "+pad(x.what, 20))+txt.Render(why))
			} else {
				rs = append(rs, "  "+dim.Render("○ "+pad(x.what, 20)+why))
			}
		}
	}
	// refusals and notices wrap inside the box, never cut (review item 32)
	// the RAM refusal is 11.3's one line «verdict · reason · budget part» (CF 7; OQ-7 (b));
	// this page is where its details are
	capDone := false
	for i, x := range p.Refusals {
		l := r.T("refused") + ": " + r.reason(x)
		if i == 0 && p.Verdict == VerdictCannot && p.CacheGB.Ok() && p.AvailGB.Ok() && p.CacheGB.V > p.AvailGB.V {
			l = r.T("cannot start") + " · " + r.Tf("RAM budget %s > free now %s", gstr(p.CacheGB.V), gstr(p.AvailGB.V))
			if p.BudgetCapGB.Ok() && p.BudgetCapGB.V > 0 {
				l += " · " + r.Tf("a budget of %s or less starts it", gstr(p.BudgetCapGB.V))
				capDone = true
			}
		}
		rs = append(rs, marked(a-4, "✕", rdB, rd, l)...)
	}
	if p.Refused && !capDone && p.BudgetCapGB.Ok() && p.BudgetCapGB.V > 0 {
		rs = append(rs, marked(a-4, "→", rdB, rd, r.Tf("a budget of %s or less starts it", gstr(p.BudgetCapGB.V)))...)
	}
	for _, x := range p.Notices {
		rs = append(rs, marked(a-4, "▲", yeB, ye, r.T(x))...)
	}
	if p.CommitLeftAfterGB.Ok() {
		mk, msty := "✓ ", gr // GL-06: near the limit the mark turns, and nothing else changes (U-13)
		if p.Near != nil {
			mk, msty = "▲ ", yeB
		}
		left := gstr(p.CommitLeftAfterGB.V) + " " + r.T("left after the start")
		if p.CommitLeftAfterGB.V < 0 { // UX-R3 (finding 6): a shortfall against the limit, not a negative «left»
			left = r.Tf("%s short after the start · against the limit", gstr(-p.CommitLeftAfterGB.V))
		}
		cm := "  " + msty.Render(mk) + txt.Render(pad(r.T("commit"), 20)) + txt.Render(left)
		if r.wide() {
			cm += dim.Render(" · " + r.T("judged on its own, freeing RAM does not add commit"))
		}
		rs = append(rs, "", cm)
	}
	if p.PoolGB.Ok() { // GL-06: the pool term on its own line (F-BU F.1; FL:14)
		pl := "    " + txt.Render(pad(r.T("expert pool"), 20)) + txt.Render(gstr(p.PoolGB.V))
		if p.PredictedGB.Ok() {
			pl += dim.Render(" · " + r.Tf("of this start's commit %s", gstr(p.PredictedGB.V)))
		}
		rs = append(rs, pl)
	}
	if n := p.Near; n != nil { // GL-06: F-BU F.3 ([A-NEAR]; U-13): the numbers, no action
		nl := r.Tf("commit near its limit: %s left after the start is below the safe line %s · limit %s · in use now %s · this start %s, of it the expert pool %s · nothing is changed",
			gstr(n.RoomGB), gstr(n.SafeGB), gstr(n.LimitGB), gstr(n.NowGB), gstr(n.PredictedGB), gstr(n.PoolGB))
		if n.RoomGB < 0 { // UX-R3 (finding 6): the shortfall against the limit, the safe line named apart
			nl = r.Tf("commit near its limit: %s short after the start, against the limit · safe line %s · limit %s · in use now %s · this start %s, of it the expert pool %s · nothing is changed",
				gstr(-n.RoomGB), gstr(n.SafeGB), gstr(n.LimitGB), gstr(n.NowGB), gstr(n.PredictedGB), gstr(n.PoolGB))
		}
		rs = append(rs, marked(a-4, "▲", yeB, ye, nl)...)
	}
	// the budget's effect and its price (review item 22)
	rs = append(rs, "")
	for _, l := range wrapWords(r.T("budget↑ = more cache hits · but less room for other programs · no effect once the whole model fits"), a-4) {
		rs = append(rs, dim.Render(l))
	}
	// the file check with the hash ends (review item 10: details live here)
	if ok {
		fc := "  " + dim.Render(pad(r.T("file check"), 12)) + fileCheckText(r, row.FileCheck)
		if row.Identity != "" {
			fc += dim.Render(" · " + row.Identity)
		}
		rs = append(rs, fc)
	}
	edge := "11"
	if p.Refused {
		edge = "9"
	}
	body := []string{tiles, ""}
	// U-21 (L:4277): the memory figures, the modified memory and the «left» figures among them,
	// only when the start verdict is tight or cannot start (F-BU G.4)
	if p.Refused || p.Verdict == VerdictTight || p.Verdict == VerdictCannot {
		// UX-R3 (finding 6): a figure below zero is a shortfall, never a negative amount left
		ll := dim.Render(r.T("left after start")+" ") + hiB.Render(r.svt(p.LeftGB, gstr))
		if p.LeftGB.Ok() && p.LeftGB.V < 0 {
			ll = rdB.Render(r.Tf("%s short after the start", gstr(-p.LeftGB.V)))
		}
		lw := dim.Render(" · "+r.T("if written to disk first")+" ") + hi.Render(r.svt(p.LeftIfWrittenGB, gstr))
		if p.LeftIfWrittenGB.Ok() && p.LeftIfWrittenGB.V < 0 {
			lw = dim.Render(" · ") + rd.Render(r.Tf("%s short if written to disk first", gstr(-p.LeftIfWrittenGB.V)))
		}
		mem = append(mem, ll+lw)
		body = append(body, indent(r.panel(a, "8", dim.Render(r.T("MEMORY · now")), mem...), 1), "")
	}
	body = append(body, verdictPart(indent(r.panel(a, edge, yeB.Render(strings.ToUpper(headT)), rs...), 1))) // the start verdict (U-31: it stays)
	var f foot
	switch {
	case p.Refused:
		f = foot{next: []string{"the launcher refuses this start · change the settings in 1 Models"}, here: []string{"o", "Task Manager"}}
	case p.Verdict == VerdictTight:
		f = foot{next: []string{"⏎ starts · tight: close an app first for more room"}, here: []string{"⏎", "start", "o", "Task Manager"}}
	default:
		f = foot{next: []string{r.T("⏎ starts") + " " + name}, here: []string{"⏎", "start", "o", "Task Manager"}}
	}
	return m.page(plMemory, r.pageHead(headT, m.launcherCost()), body, f)
}
