package ui

// UX-R14 (version 8; the user's acceptance finding 22, reviews/_user_8_LAUNCHER-REVIEW_11.md,
// screenshots 151302 and 151431): the first-run setup page (H6) shows the whole setup until
// the server starts: the repack, the repacker's own check (verify) and the start's speed
// check. The big percent covers the three phases; the repack keeps its bar; verify has its
// bar where the repacker reports its pairs, else a spinner; the speed check (the start's
// sweep step, which reports no fraction) has a spinner; a done phase shows ✓; the STEP tile
// counts the phases.

import (
	"fmt"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// StepSweepName is the name of the start's sweep step (core.StepSweep), the setup's speed
// check; the wire names the step so (wire/tracks.go stepLabels).
const StepSweepName = "queue depth"

// setupPhases are the setup's phases in their order; setupWeights their shares of the setup
// page's percent. The repack and verify each move the model's expert bytes twice (the repack
// reads the source and writes the output; verify reads both again, repack_experts.py
// verify_repack), and writing is the slower half, so the repack weighs more; the speed check is
// a short sweep (10.6 s in the user's screenshot 151431) and reports no fraction, so it holds
// the last share and the percent stays below 100 until it ends.
var (
	setupPhases  = [3]string{"repack", "verify", "speed check"}
	setupWeights = [3]float64{55, 40, 5}
)

// spinFrames turn one frame per screen tick (pollEvery).
var spinFrames = [4]string{"|", "/", "-", `\`}

// setupMemo is what this start's setup events told so far.
type setupMemo struct {
	last              SetupProgress // the last setup progress
	at                time.Time     // when it came
	repDone, repTotal Val           // the repack's layer counts
	phase             int           // the phase reached (index of setupPhases); it never goes back
}

// noteSetup keeps a start event's setup progress; the first event without one after a setup
// means the setup ended and the start's speed check follows.
func (m *Model) noteSetup(ev StartEvent) {
	if ev.Setup == nil {
		if m.setup != nil {
			m.setup.phase = len(setupPhases) - 1
		}
		return
	}
	if m.setup == nil {
		m.setup = &setupMemo{}
	}
	s := m.setup
	s.last, s.at = *ev.Setup, m.now()
	switch ev.Setup.Name { // the wire's names (wire/fix2.go setupStep)
	case "check":
		s.phase = max(s.phase, 1)
	case "done":
		s.phase = max(s.phase, 2)
	}
	if s.phase == 0 && ev.Setup.Done.Ok() && ev.Setup.Total.Ok() {
		s.repDone, s.repTotal = ev.Setup.Done, ev.Setup.Total
	}
}

// setupPage: the start shows the setup page (H6) while a setup event is the last one, and
// after a setup until the start's speed check has ended.
func (m *Model) setupPage() bool {
	ev := m.startEv
	return ev != nil && (ev.Setup != nil || m.setup != nil && !speedDone(ev))
}

// speedDone: the start's sweep step has ended (or the start reports none, or is ready).
func speedDone(ev *StartEvent) bool {
	if ev.Ready {
		return true
	}
	for _, s := range ev.Steps {
		if s.Name == StepSweepName {
			return s.State != StepTodo && s.State != StepRunning
		}
	}
	return len(ev.Steps) > 0
}

// spinner is the spinner's frame at this screen tick.
func (m *Model) spinner() string {
	return spinFrames[int(m.now().UnixNano()/int64(pollEvery))%len(spinFrames)]
}

// setupNow is the setup page's state.
type setupNow struct {
	sp                 SetupProgress
	at                 time.Time
	ph                 int
	rep, ver           float64 // the phases' fractions
	repKnown, verKnown bool
	repDone, repTotal  Val
	pct                int
}

func (m *Model) setupState() setupNow {
	s := m.setup
	if s == nil {
		s = &setupMemo{}
	}
	n := setupNow{sp: s.last, at: s.at, ph: s.phase, repDone: s.repDone, repTotal: s.repTotal}
	if s.repDone.Ok() && s.repTotal.Ok() && s.repTotal.V > 0 {
		n.rep, n.repKnown = min(1, s.repDone.V/s.repTotal.V), true
	}
	if n.ph > 0 {
		n.rep = 1
	}
	if n.ph == 1 && n.sp.Done.Ok() && n.sp.Total.Ok() && n.sp.Total.V > 0 {
		n.ver, n.verKnown = min(1, n.sp.Done.V/n.sp.Total.V), true
	}
	if n.ph > 1 {
		n.ver = 1
	}
	n.pct = min(99, int(setupWeights[0]*n.rep+setupWeights[1]*n.ver))
	return n
}

// setupRows are the phase rows: the repack's bar and count with its line, then verify and
// the speed check, each «… -ing …» with its bar or the spinner while it runs; ✓ when done.
func (m *Model) setupRows(n setupNow) []string {
	r := m.r
	mark := func(i int) string {
		switch {
		case i < n.ph:
			return gr.Render("✓ ")
		case i == n.ph:
			return yeB.Render("▶ ")
		}
		return dim.Render("○ ")
	}
	l := "  " + mark(0) + txt.Render(pad(r.T("repack"), 8))
	if n.repKnown || n.ph > 0 {
		l += bar(n.rep, max(0, r.cw()-32), ye)
	}
	if n.repKnown {
		l += "  " + hiB.Render(fmt.Sprintf("%.0f", n.repDone.V)) + dim.Render(fmt.Sprintf(" / %.0f %s", n.repTotal.V, r.T("layers")))
	}
	out := []string{l, "    " + dim.Render(r.T("rewrites the model file once so experts load fast · first run only"))}
	doings := [3]string{"", "verifying …", "checking the drive speed …"}
	for i := 1; i < len(doings); i++ {
		doing := doings[i]
		switch {
		case i < n.ph:
			out = append(out, "  "+mark(i)+txt.Render(r.T(setupPhases[i])))
		case i > n.ph:
			out = append(out, "  "+mark(i)+dim.Render(r.T(setupPhases[i])))
		case i == 1 && n.verKnown:
			d := r.T(doing)
			out = append(out, "  "+mark(i)+hiB.Render(d)+" "+bar(n.ver, max(0, r.cw()-12-lipgloss.Width(d)), ye)+
				hiB.Render(fmt.Sprintf(" %3d %%", int(n.ver*100))))
		default:
			out = append(out, "  "+mark(i)+hiB.Render(r.T(doing))+" "+cyB.Render(m.spinner()))
		}
	}
	return out
}
