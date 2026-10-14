// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"strconv"
	"strings"
)

// reasonKeys are the backend reasons with numbers the screens show in the screen language
// (GL-09: 11.3's RAM reason; GL-08's busy port, its finding 3).
var reasonKeys = []string{
	"RAM budget %s > free now %s",
	"port %s is in use by another program",
}

// reason is a backend reason in the screen language: a reason of reasonKeys with its numbers,
// else the text by the table (a reason the backend words stays its text).
func (r rc) reason(s string) string {
	for _, k := range reasonKeys {
		if vals, ok := matchKey(k, s); ok {
			return r.Tf(k, vals...)
		}
	}
	return r.T(s)
}

// matchKey reads the %s values of a key's English text back out of s.
func matchKey(key, s string) ([]string, bool) {
	parts := strings.Split(key, "%s")
	if len(parts) < 2 || !strings.HasPrefix(s, parts[0]) {
		return nil, false
	}
	rest := s[len(parts[0]):]
	var vals []string
	for _, p := range parts[1:] {
		if p == "" {
			vals, rest = append(vals, rest), ""
			continue
		}
		i := strings.Index(rest, p)
		if i <= 0 {
			return nil, false
		}
		vals, rest = append(vals, rest[:i]), rest[i+len(p):]
	}
	return vals, rest == ""
}

// taskManagerLine is the line once Task Manager has opened (U-35 (i), L:4275): it opens
// without a selection, so the line names the app chosen on 3 Memory and its PID.
func (m *Model) taskManagerLine() string {
	r := m.r
	if m.mem != nil && m.appCur >= 0 && m.appCur < len(m.mem.Apps) {
		if a := m.mem.Apps[m.appCur]; a.PID > 0 {
			return r.Tf("Task Manager opened · find %s, PID %s", a.Name, strconv.Itoa(a.PID))
		}
	}
	return r.T("Task Manager opened · nothing is selected in it")
}
