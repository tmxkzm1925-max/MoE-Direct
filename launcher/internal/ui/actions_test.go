package ui

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// newHK is newH with an action-to-key table of the test's own.
func newHK(t *testing.T, f *FakeBackend, keys KeyTable) *harness {
	t.Helper()
	hs := &harness{t: t, f: f, clock: time.Date(2026, 10, 10, 9, 0, 0, 0, time.UTC)}
	hs.m = New(f, Options{Lang: LangEnglish, Width: 120, Height: 30, Keys: keys,
		Now: func() time.Time { return hs.clock },
		Tick: func(d time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
			hs.ticks = append(hs.ticks, pendingTick{d, hs.clock.Add(d), fn})
			return nil
		}})
	t.Cleanup(f.Close)
	hs.run(hs.m.Init())
	hs.drawn()
	return hs
}

// permuted gives every table action the binding of the next action in TableActions (the last
// gets the first's): no action keeps a key it held.
func permuted() KeyTable {
	d := DefaultKeyTable()
	p := KeyTable{}
	for i, a := range TableActions {
		p[a] = d[TableActions[(i+1)%len(TableActions)]]
	}
	return p
}

func sortedActions(as []Action) []string {
	var s []string
	for _, a := range as {
		s = append(s, string(a))
	}
	sort.Strings(s)
	return s
}

// stops lists the Stop calls made so far.
func stops(h *harness) []string {
	var s []string
	for _, c := range h.f.Calls() {
		if strings.HasPrefix(c, "Stop ") {
			s = append(s, c)
		}
	}
	return s
}

// TestS8LA01 (S8-LA-01; F-LS edition 3.1-3.3; GL-02).
func TestS8LA01(t *testing.T) {
	// (1) the action list the table holds = the round-5 design's key table
	// (designs/launcher_design/proto/go5/keys.go:79-87, in the edition's ids) minus the
	// reply cut (L:4177 ①) and closing another app (CF 6; U-28 (ii)), plus the redo of a
	// cut-off setup the confirmed base layout holds (L:4265 ①) (edition 3.1, OQ-8 (a))
	design := []Action{"cut-this-reply", DoUnloadModel, DoCancelSetupOrStart, DoQuit, DoForceQuit,
		DoDeleteCutOffSetup, "close-the-other-app", DoCopyAddress, DoNavigate}
	var want []Action
	for _, a := range design {
		if a != "cut-this-reply" && a != "close-the-other-app" {
			want = append(want, a)
		}
	}
	want = append(want, DoRedoCutOffSetup)
	if got := sortedActions(TableActions); !reflect.DeepEqual(got, sortedActions(want)) {
		t.Fatalf("(1) table actions %v, want %v", got, sortedActions(want))
	}
	var inTable []Action
	for a := range DefaultKeyTable() {
		inTable = append(inTable, a)
	}
	if !reflect.DeepEqual(sortedActions(inTable), sortedActions(want)) {
		t.Fatalf("(1) the default table binds %v, want %v", sortedActions(inTable), sortedActions(want))
	}
	for a := range RemovedActions {
		if _, ok := DefaultKeyTable()[a]; ok {
			t.Fatalf("(1) removed action %q is in the table", a)
		}
	}
	t.Logf("(1) table actions = %v", sortedActions(TableActions))

	// (4) the entries equal the confirmed default layout of the base (L:4265 ①; the GOL
	// layout at 85e432ec8: keys.go:28-38 key list, :61-75, :87-103, :108-136, :257-260,
	// :515-520; the Log filter on its own key, keys.go:605-607, is a place action)
	base := KeyTable{
		DoUnloadModel:        {Keys: []string{"ctrl+f"}},
		DoCancelSetupOrStart: {Keys: []string{"ctrl+x"}},
		DoQuit:               {Keys: []string{"q", "ㅂ"}},
		DoForceQuit:          {Keys: []string{"ctrl+c"}},
		DoDeleteCutOffSetup:  {Keys: []string{"ctrl+d"}},
		DoRedoCutOffSetup:    {Keys: []string{"ctrl+r"}},
		DoCopyAddress:        {Keys: []string{"c"}},
		DoNavigate:           {Keys: []string{"left", "right", "1", "2", "3", "4", "5", "?", "esc"}},
	}
	if !reflect.DeepEqual(DefaultKeyTable(), base) {
		t.Fatalf("(4) default table %v, want the base layout %v", DefaultKeyTable(), base)
	}
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	hs.keys("4", "tab")
	if hs.m.logFilter != 1 {
		t.Fatalf("(4) the Log filter is not on tab (keys.go:605-607)")
	}
	t.Logf("(4) default table = base layout (8 entries)")

	// (2) under the permuted table every action is reached through its new entry and
	// through no entry it no longer holds
	p := permuted()
	d := DefaultKeyTable()
	for _, a := range TableActions {
		if reflect.DeepEqual(p[a], d[a]) {
			t.Fatalf("(2) the permutation leaves %q its binding", a)
		}
	}
	reached := func(a Action, keys []string) bool { return reach2(t, p, a, keys) }
	for _, a := range TableActions {
		nk := p[a].Keys
		if !reached(a, nk[:1]) {
			t.Fatalf("(2) %s not reached through its new entry %q", a, nk[0])
		}
		for _, k := range d[a].Keys {
			if reached(a, []string{k}) {
				t.Fatalf("(2) %s still reached through %q, which it no longer holds", a, k)
			}
		}
	}
	// every key of a multi-key binding reaches the action that holds it now
	for _, k := range p[DoCopyAddress].Keys {
		if !reached(DoCopyAddress, []string{k}) {
			t.Fatalf("(2) copy-address not reached through %q", k)
		}
	}
	t.Logf("(2) permuted table: 8 of 8 actions reached through the new entry, none through an old one")

	// (3) while a reply runs (VERSION_8.md:262 as corrected by the main, Astra 8_20):
	// an action that keeps the server serving never cuts the reply (no stop, no cut call);
	// unload-model starts the normal stop ladder (StopUnload) at the press, without
	// waiting for the reply; the first quit is a warning only (no stop, the U-27 line);
	// the second quit and the confirmed force-quit end the server at once, nothing
	// saved (StopForce); the server's end ends the reply with it
	for _, tbl := range []KeyTable{d, p} {
		for _, a := range TableActions {
			for _, k := range tbl[a].Keys {
				f := NewFake() // the fake serves with a reply in decode (Phase ReqDecode)
				h := newHK(t, f, tbl)
				h.keys("enter")
				if !h.m.replyRunning() {
					t.Fatalf("(3) no reply running in the fixture")
				}
				h.keys(k)
				if a == DoQuit || a == DoForceQuit {
					// the first press: a warning only
					if s := stops(h); len(s) != 0 {
						t.Fatalf("(3) %s via %q: the first press stopped %v", a, k, s)
					}
					want := armedForce
					if a == DoQuit {
						want = armedQuitReply
					}
					if got := h.m.arm(); got != want {
						t.Fatalf("(3) %s via %q: first press arm %v, want %v", a, k, got, want)
					}
					h.keys(k) // the second press
				}
				got := stops(h)
				var allowed []string
				switch a {
				case DoUnloadModel:
					allowed = []string{fmt.Sprintf("Stop %d", StopUnload)}
				case DoQuit, DoForceQuit:
					allowed = []string{fmt.Sprintf("Stop %d", StopForce)}
				}
				if !reflect.DeepEqual(got, allowed) && !(len(got) == 0 && len(allowed) == 0) {
					t.Fatalf("(3) %s via %q during a reply: stops %v, want %v (calls %v)", a, k, got, allowed, f.Calls())
				}
				for _, c := range f.Calls() {
					if strings.Contains(c, "cut") {
						t.Fatalf("(3) %s via %q: a cut call %q", a, k, c)
					}
				}
			}
		}
	}
	t.Logf("(3) during a reply: the serving actions never cut it; unload = the stop ladder at the press (StopUnload); first quit = warning only; second quit and confirmed force-quit = at once, nothing saved (StopForce)")
}

// reach2 runs one action's probe on a fresh fake and screen with table p and the keys given:
// true when the action's own effect happened.
func reach2(t *testing.T, p KeyTable, a Action, keys []string) bool {
	t.Helper()
	f := NewFake()
	idleServer(f)
	press := func(h *harness) {
		for _, k := range keys {
			h.keys(k)
			if a == DoQuit || a == DoForceQuit {
				h.keys(k)
			}
		}
	}
	switch a {
	case DoUnloadModel:
		h := newHK(t, f, p)
		h.keys("enter")
		press(h)
		return h.calls(fmt.Sprintf("Stop %d", StopUnload)) == 1
	case DoCancelSetupOrStart:
		f.StartHold = true
		f.StartEvs = f.StartEvs[:2]
		h := newHK(t, f, p)
		h.keys("enter")
		press(h)
		return h.calls(fmt.Sprintf("Stop %d", StopCancel)) == 1
	case DoQuit:
		h := newHK(t, f, p)
		press(h)
		return h.quit && h.calls("Stop") == 0
	case DoForceQuit:
		h := newHK(t, f, p)
		h.keys("enter")
		press(h)
		return h.calls(fmt.Sprintf("Stop %d", StopForce)) == 1
	case DoDeleteCutOffSetup, DoRedoCutOffSetup:
		f.Cat.Models[0].Setup = SetupCutOff
		f.Cat.Models[0].Partial = PartialSetup{LayersDone: K(25), LayersTotal: K(48), FilesGB: K(37), When: "yesterday 22:14"}
		h := newHK(t, f, p)
		press(h)
		if a == DoRedoCutOffSetup {
			// GL-10 (S8-LA-34): the redo shows the setup plan (its ⏎ redoes the setup)
			return h.calls("RepackPlan "+f.Cat.Models[0].ID+" ") == 1 && h.calls("Action ") == 0
		}
		return h.calls("Action "+string(ActDeletePartial)+" ") == 1
	case DoCopyAddress:
		h := newHK(t, f, p)
		h.keys("enter")
		h.m.place = plServer
		press(h)
		return h.calls("Action "+string(ActCopyAddress)+" ") == 1
	case DoNavigate:
		// from 1 Models: previous place (5 Settings), next place (2 Server), place n, help
		h := newHK(t, f, p)
		before := h.m.place
		press(h)
		return h.m.place != before || h.m.help
	}
	t.Fatalf("no probe for %s", a)
	return false
}

// TestS8LA03 (S8-LA-03; edition 3.5; U-27, L:4275; GL-02): the quit sequence with a fake
// server holding a model, first with no reply running, then with one running.
func TestS8LA03(t *testing.T) {
	// no reply running: the second q runs cache save (with the item-3 bar), unload, exit;
	// no step cuts a reply; the stop begins at unload (StopQuit, the stop ladder)
	f := NewFake()
	idleServer(f)
	f.StopHold = true
	h := newH(t, f, LangEnglish, 120, 30)
	h.keys("enter", "q")
	h.want("H8Q")
	if !strings.Contains(h.plain(), "q again: unload the model and exit") {
		t.Fatalf("idle first q: bottom line\n%s", h.plain())
	}
	h.keys("q")
	h.want("HQ")
	if p := h.plain(); !strings.Contains(p, "save: —") {
		t.Fatalf("idle quit: the cache-save bar (item 3) is not drawn:\n%s", p)
	}
	// the same quit run to its end (the fake's stop events, then the exit)
	f = NewFake()
	idleServer(f)
	h = newH(t, f, LangEnglish, 120, 30)
	h.keys("enter", "q", "q")
	if got := h.m.QuitSteps(); !reflect.DeepEqual(got, QuitIdle) {
		t.Fatalf("idle quit steps %v, want %v (calls %v)", got, QuitIdle, f.Calls())
	}
	if !reflect.DeepEqual(stops(h), []string{fmt.Sprintf("Stop %d", StopQuit)}) || !h.quit {
		t.Fatalf("idle quit: stops %v quit %v", stops(h), h.quit)
	}
	for _, c := range f.Calls() {
		if strings.Contains(c, "cut") {
			t.Fatalf("idle quit: a cut call %q", c)
		}
	}
	t.Logf("no reply: steps %v; stop %v; exit %v", h.m.QuitSteps(), stops(h), h.quit)

	// a reply running: the first q shows the warning in the bottom line and changes nothing
	// else; the second ends at once with no cache save
	f = NewFake() // Phase ReqDecode: a reply in decode
	h = newH(t, f, LangEnglish, 120, 30)
	h.keys("enter")
	if !h.m.replyRunning() {
		t.Fatalf("no reply running in the fixture")
	}
	place, ph, n := h.m.place, h.m.ph, len(f.Calls())
	h.keys("q")
	lines := strings.Split(strings.TrimRight(h.plain(), "\n"), "\n")
	bottom := lines[len(lines)-1]
	if !strings.Contains(bottom, "reply in progress · press again to force quit") {
		t.Fatalf("reply first q: bottom line %q", bottom)
	}
	if h.m.place != place || h.m.ph != ph || len(f.Calls()) != n || h.quit {
		t.Fatalf("reply first q changed more than the bottom line: place %d→%d ph %d→%d calls %v",
			place, h.m.place, ph, h.m.ph, f.Calls()[n:])
	}
	k := newH(t, NewFake(), LangKorean, 120, 30)
	k.keys("enter", "q")
	if !strings.Contains(k.plain(), "출력 중 · 한 번 더 누르면 강제 종료") {
		t.Fatalf("reply first q, Korean:\n%s", k.plain())
	}
	h.keys("q")
	if got := h.m.QuitSteps(); !reflect.DeepEqual(got, QuitDuringReply) {
		t.Fatalf("reply quit steps %v, want %v", got, QuitDuringReply)
	}
	if !reflect.DeepEqual(stops(h), []string{fmt.Sprintf("Stop %d", StopForce)}) || !h.quit {
		t.Fatalf("reply quit: stops %v (want StopForce only, no save) quit %v", stops(h), h.quit)
	}
	t.Logf("reply running: bottom line %q; steps %v; stop %v; exit %v", strings.TrimSpace(bottom), h.m.QuitSteps(), stops(h), h.quit)
}
