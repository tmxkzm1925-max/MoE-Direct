// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"time"
)

// ScreenID names the design screen on display (designs/launcher_design
// README: H1-H15, HF, H8Q, HQ; the key page H5K is the next version, L:4265
// ②). "S0" is the 2 Server page with no server, which the design does not draw.
func (m *Model) ScreenID() string {
	if m.help {
		if m.helpTiles {
			return "HF"
		}
		return "H13"
	}
	switch m.place {
	case plModels:
		switch {
		case m.noModels():
			return "H3"
		}
		if _, ok := m.cutOff(); ok {
			return "H4"
		}
		if row, ok := m.selRow(); ok && m.runChanged(row.ID) {
			return "H2"
		}
		return "H1"
	case plServer:
		if m.planShown() {
			return "HP" // the plan box before a setup (conflict 9)
		}
		switch m.ph {
		case phStarting:
			if m.setupPage() { // UX-R14: also through the start's speed check
				return "H6"
			}
			return "H15"
		case phFailed:
			return "H7"
		case phServing:
			if m.quitArmed {
				return "H8Q"
			}
			if m.status != nil && m.status.Slow {
				return "H9"
			}
			return "H8"
		case phStopping:
			return "HQ"
		case phStopped:
			return "H12"
		}
		return "S0"
	case plMemory:
		if m.ph == phServing {
			return "H10"
		}
		return "H14"
	case plLog:
		return "H11"
	case plSettings:
		return "H5"
	}
	return "H1"
}

// View draws the screen on display, in the glyph set the probe chose.
func (m *Model) View() string {
	return m.glyphs.Apply(partStrip.Replace(m.render()))
}

// Parts names the parts on the last drawn screen (GL-04; S8-LA-09).
func (m *Model) Parts() Parts {
	if m.r.shown == nil {
		return nil
	}
	return append(Parts{}, *m.r.shown...)
}

func (m *Model) render() string {
	switch m.ScreenID() {
	case "HF":
		return m.vHF()
	case "H13":
		return m.vH13()
	case "H3":
		return m.vH3()
	case "H4":
		return m.vH4()
	case "H1", "H2":
		return m.vModels()
	case "H6":
		return m.vH6()
	case "HP":
		return m.vPlan()
	case "H15":
		return m.vH15()
	case "H7":
		return m.vH7()
	case "H8", "H9", "H8Q":
		return m.vServe()
	case "HQ":
		return m.vHQ()
	case "H12":
		return m.vH12()
	case "S0":
		return m.vS0()
	case "H10":
		return m.vH10()
	case "H14":
		return m.vH14()
	case "H11":
		return m.vH11()
	case "H5":
		return m.vH5()
	}
	return ""
}

// srvState is the server state at the right of the top row.
func (m *Model) srvState() srv {
	r := m.r
	name := m.runModel
	if row, ok := m.rowByID(m.runModel); ok {
		name = row.Name
	}
	det := "  " + name
	if m.status != nil && m.status.Port > 0 {
		det += fmt.Sprintf(" · :%d", m.status.Port)
	}
	switch m.ph {
	case phStarting:
		if m.setupPage() { // UX-R14: the setup page's percent over its three phases
			pc := m.setupState().pct
			return srv{"◐", fmt.Sprintf("%s %d %%", r.T("preparing"), pc), yeB, det, fmt.Sprintf("%s %d%%", r.T("setup"), pc)}
		}
		return srv{"◐", fmt.Sprintf("%s %d s", r.T("starting"), int(m.startElapsed().Seconds())), cyB, det, r.T("starting")}
	case phServing:
		if m.status != nil && m.status.Slow {
			return srv{"▲", r.T("slow") + " · " + r.T("decode") + " " + num(m.status.LastDecodeTokS, "%.1f") + " tok/s", yeB, det, r.T("slow")}
		}
		if m.status != nil && m.status.DecodeTokS.Ok() {
			return srv{"●", r.T("decode") + " " + num(m.status.DecodeTokS, "%.1f") + " tok/s", grB, det, r.T("serving")}
		}
		return srv{"●", r.T("serving"), grB, det, r.T("serving")}
	case phFailed:
		return srv{"✕", r.T("failed"), rdB, det, r.T("failed")}
	case phStopping:
		w := map[StopMode]string{StopQuit: "quitting", StopUnload: "unloading", StopForce: "closing", StopCancel: "cancelling"}[m.stopMode]
		return srv{"◐", r.T(w), yeB, det, r.T(w)}
	case phStopped:
		return srv{"■", r.T("stopped"), txt, det, r.T("stopped")}
	}
	return srv{"○", r.T("no server"), dim, "", r.T("no server")}
}

func (m *Model) rowByID(id string) (ModelRow, bool) {
	for _, r := range m.cat.Models {
		if r.ID == id {
			return r, true
		}
	}
	return ModelRow{}, false
}

func (m *Model) runName() string {
	if row, ok := m.rowByID(m.runModel); ok && row.Name != "" {
		return row.Name
	}
	return m.runModel
}

func (m *Model) arm() armed {
	switch {
	case m.frcArmed:
		return armedForce
	case m.quitArmed && m.quitReply:
		return armedQuitReply
	case m.quitArmed:
		return armedQuit
	}
	return armedNone
}

// page assembles one place's page.
func (m *Model) page(cur int, head string, body []string, f foot) string {
	r := m.r
	return r.screen(r.nav(cur, m.srvState()), head, body, r.footer(f, m.arm(), m.notice))
}

func (m *Model) startElapsed() time.Duration {
	if m.startEv == nil {
		return m.now().Sub(m.startAt)
	}
	return m.startEv.Elapsed + m.now().Sub(m.startAt)
}

func (m *Model) stopElapsed() time.Duration {
	if m.stopEv == nil {
		return m.now().Sub(m.stopAt)
	}
	return m.stopEv.Elapsed + m.now().Sub(m.stopAt)
}
