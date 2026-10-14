// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

// showNotPorted is the one switch of UX-R4 (the user's acceptance findings 8, 10 and 20 (c);
// the main's reading of letter 5 items 8 and 10; L:4198, U-43): false, the release setting,
// hides every 5 Settings row whose value would read «not ported», the not-ported claim of its
// foot line and the 2 Server «waiting: not ported» part, so the screens show only what works;
// true brings them back as before. The rows, their keys and their table entries stay; a row
// returns by itself once its value is ported.
var showNotPorted = false

// set5Hidden: a 5 Settings row that is hidden, its value not ported (showNotPorted off).
func (m *Model) set5Hidden(sec, i int) bool {
	if showNotPorted {
		return false
	}
	li := m.launcherInfo()
	switch sec {
	case secLauncher:
		switch i {
		case 0: // address
			return li.Address == ""
		case set5SmokeRow:
			return li.Smoke != SmokeOff && li.Smoke != SmokeOn && li.Smoke != SmokeOnce
		case 3: // on open
			return li.OnOpen == "" || li.OnOpen == "not ported"
		case 4: // logs kept
			return !li.LogsKeptDays.Ok()
		case set5CapRow:
			return li.CacheCap == ""
		case set5WarmRow:
			return li.WarmStart == ""
		case set5AutosaveRow:
			return li.Autosave == ""
		}
	case secWrites:
		ws := m.writeRows()
		return i >= 0 && i < len(ws) && ws[i].path == ""
	}
	return false
}

// set5Next is the next shown row of the section from row from in direction dir (+1, -1), or -1.
func (m *Model) set5Next(from, dir int) int {
	n := m.set5Rows()
	for i := from + dir; i >= 0 && i < n; i += dir {
		if !m.set5Hidden(m.set5Sec, i) {
			return i
		}
	}
	return -1
}

// UX-R12 (findings 8, 10 and 20 (c), the remainder on 2 Server): with the switch off, a value
// that reaches 2 Server in the NotPorted state draws a dash, never «not ported» / «미이식»: the
// stop summary's PREFILL and TTFT tiles and the TTFT «max», the DISK tiles of the setup and the
// start, the RAM tiles and the free-memory tile. Unmeasured keeps «not measured yet».
func (r rc) svt(v Val, f func(float64) string) string {
	if v.State == NotPorted && !showNotPorted {
		return "—"
	}
	return r.vt(v, f)
}

// svu is vu for 2 Server, with svt's dash.
func (r rc) svu(v Val, f func(float64) string, unit string) string {
	if v.State == NotPorted && !showNotPorted {
		return dim.Render("—")
	}
	return r.vu(v, f, unit)
}

// snp is the not-ported word of a 2 Server place with no value of its own (a step, the cache
// save), or, with the switch off, the dash.
func (r rc) snp() string {
	if showNotPorted {
		return r.np()
	}
	return "—"
}
