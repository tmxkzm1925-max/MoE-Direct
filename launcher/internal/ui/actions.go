// Actions and the action-to-key table (GL-02; F-LS edition 3.1-3.6 and 12.1; S8-LA-01,
// S8-LA-03). The table holds the actions of edition 3.1: the round-5 design's key table minus
// the reply cut (DECISION_LEDGER.md:4177 ①) and closing another app (conflict row 6; U-28
// (ii)), plus the cut-off setup's redo of the confirmed base layout (:4265 ①). Its default
// binding is that confirmed layout (:4265 ①). Changing keys and the key page H5K are the next
// version (:4265 ②), so the table is fixed in code; tests give the screen another one.
//
// The table is the only way a key reaches one of these actions: Model.key maps the key pressed
// to the action's own key of the default layout (canon), and the rest of the dispatch reads
// that. A key the table no longer holds reaches none of these actions.

package ui

// Action is one action of the action-to-key table (edition 12.1; its ids).
type Action string

// The table actions (edition 3.1). The edition's Go names (ActUnloadModel, ...) are spelled
// Do... here because the ActionID constants of Backend.Action already use Act...
const (
	DoUnloadModel        Action = "unload-model"
	DoCancelSetupOrStart Action = "cancel-setup-or-start"
	DoQuit               Action = "quit"
	DoForceQuit          Action = "force-quit"
	DoDeleteCutOffSetup  Action = "delete-cut-off-setup"
	DoRedoCutOffSetup    Action = "redo-cut-off-setup"
	DoCopyAddress        Action = "copy-address"
	DoNavigate           Action = "navigate" // previous · next place, place 1-5, help, back
)

// TableActions is the action list the table holds, in the edition's order.
var TableActions = []Action{
	DoUnloadModel, DoCancelSetupOrStart, DoQuit, DoForceQuit,
	DoDeleteCutOffSetup, DoRedoCutOffSetup, DoCopyAddress, DoNavigate,
}

// RemovedActions are the actions the table no longer holds, with their records (edition 3.2).
var RemovedActions = map[Action]string{
	"cut-this-reply":      "L:4177 ① (reply cut is the client platform's)",
	"close-the-other-app": "conflict row 6 (1) at c900f9cd0; U-28 (ii); rule 6",
	"change-key":          "L:4265 ② (H5K and the stored key table: next version)",
	"default-key":         "L:4265 ② (H5K and the stored key table: next version)",
}

// Binding is the keys an action is reached through, as Bubble Tea names them. Navigate's keys
// are read by position: previous place, next place, place 1 to 5, help, back. The other
// actions do the same thing through every key they hold.
type Binding struct{ Keys []string }

// KeyTable binds every table action.
type KeyTable map[Action]Binding

// navigate's positions in its binding.
const (
	navPrev   = 0
	navNext   = 1
	navPlace1 = 2 // places 1-5 are navPlace1 .. navPlace1+4
	navHelp   = 7
	navBack   = 8
)

// DefaultKeyTable is the confirmed default layout of the base (L:4265 ①; the GOL draft's keys).
// The quit key also answers to ㅂ, the q key in Hangul mode of the Korean IME (LAUNCH-FEAS
// Q1.4). force-quit and quit act on their second press (edition 3.5, 3.6).
func DefaultKeyTable() KeyTable {
	return KeyTable{
		DoUnloadModel:        {Keys: []string{"ctrl+f"}},
		DoCancelSetupOrStart: {Keys: []string{"ctrl+x"}},
		DoQuit:               {Keys: []string{"q", "ㅂ"}},
		DoForceQuit:          {Keys: []string{"ctrl+c"}},
		DoDeleteCutOffSetup:  {Keys: []string{"ctrl+d"}},
		DoRedoCutOffSetup:    {Keys: []string{"ctrl+r"}},
		DoCopyAddress:        {Keys: []string{"c"}},
		DoNavigate:           {Keys: []string{"left", "right", "1", "2", "3", "4", "5", "?", "esc"}},
	}
}

// Lookup gives the table action that holds key k and k's position in its binding.
func (t KeyTable) Lookup(k string) (Action, int, bool) {
	for _, a := range TableActions {
		for i, x := range t[a].Keys {
			if x == k {
				return a, i, true
			}
		}
	}
	return "", 0, false
}

// defaultKeys is every key the default layout gives a table action.
var defaultKeys = func() map[string]bool {
	m := map[string]bool{}
	for _, b := range DefaultKeyTable() {
		for _, k := range b.Keys {
			m[k] = true
		}
	}
	return m
}()

// canon maps a key pressed to the key the dispatch reads: a key the table holds becomes the
// default layout's key of the same action and position (an action's last default key past
// the end of its default binding); a default-layout key the table no longer holds becomes ""
// (it reaches no table action); any other key stays itself (the places' own keys).
func (t KeyTable) canon(k string) string {
	if a, i, ok := t.Lookup(k); ok {
		d := DefaultKeyTable()[a].Keys
		return d[min(i, len(d)-1)]
	}
	if defaultKeys[k] {
		return ""
	}
	return k
}

// QuitStep is one step of the quit sequence (edition 3.5, 12.1).
type QuitStep string

const (
	QuitStepCacheSave QuitStep = "cache-save"
	QuitStepUnload    QuitStep = "unload"
	QuitStepExit      QuitStep = "exit"
)

// The quit sequences: with no reply running, the second quit action saves the cache (with the
// item-3 bar), unloads and exits; with a reply running it exits at once, nothing saved (U-27).
var (
	QuitIdle        = []QuitStep{QuitStepCacheSave, QuitStepUnload, QuitStepExit}
	QuitDuringReply = []QuitStep{QuitStepExit}
)
