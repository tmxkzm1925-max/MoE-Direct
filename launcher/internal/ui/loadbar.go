// Decision ③ (DECISION_LEDGER.md:4179): the load bar runs from the launcher's
// start of the server to /health ready. It is filled by the elapsed time over
// the last measured duration for the same model and settings (the backend
// gives it). With no measurement it shows elapsed seconds only. On an overrun
// it holds at 99 % and says «지난번보다 느림» (slower than last time); it never
// shows a false 100 %: 100 % only at ready. The cache-save bar follows the same rule
// from the quit action to the save-completion response (GL-07).

package ui

import (
	"fmt"
	"time"
)

// progress is the bar state at one moment.
type progress struct {
	Frac        float64 // 0 .. 0.99 while running
	Pct         int     // what the screen prints
	SecondsOnly bool    // no last duration: elapsed seconds only, no fill
	Slower      bool    // elapsed reached the last duration: held at 99 %
	Ready       bool    // the span ended (ready, or the save answered): 100 %
	Elapsed     time.Duration
	Last        time.Duration
}

// maxRunningFrac is the most a running bar shows.
const maxRunningFrac = 0.99

// loadProgress applies the rule to an elapsed time and the last duration.
func loadProgress(elapsed time.Duration, lastS Val) progress {
	p := progress{Elapsed: elapsed}
	if !lastS.Ok() || lastS.V <= 0 {
		p.SecondsOnly = true
		return p
	}
	p.Last = time.Duration(lastS.V * float64(time.Second))
	f := elapsed.Seconds() / lastS.V
	if elapsed >= p.Last {
		p.Slower = true
	}
	if f > maxRunningFrac {
		f = maxRunningFrac
	}
	if f < 0 {
		f = 0
	}
	p.Frac = f
	p.Pct = int(f*100 + 1e-9)
	return p
}

// doneProgress is a span that has ended: 100 %, whatever the last duration was.
func doneProgress(took time.Duration, lastS Val) progress {
	p := progress{Elapsed: took, Frac: 1, Pct: 100, Ready: true}
	if lastS.Ok() && lastS.V > 0 {
		p.Last = time.Duration(lastS.V * float64(time.Second))
	}
	return p
}

// startLoad is the load bar of a start event at now (GL-07): not begun before the server child
// starts, running from the child's start to /health ready, full at ready.
func startLoad(ev *StartEvent, sinceEvent time.Duration) progress {
	switch {
	case ev == nil:
		return loadProgress(0, U)
	case ev.LoadReady:
		return doneProgress(ev.LoadTook, ev.LastS)
	case ev.LoadRunning:
		return loadProgress(ev.Elapsed+sinceEvent, ev.LastS)
	}
	return loadProgress(0, ev.LastS)
}

// saveBar is the cache-save bar of a stop event at now (GL-07).
func saveBar(s SaveProgress, sinceEvent time.Duration) progress {
	if s.State == StepDone {
		return doneProgress(s.Elapsed, s.LastS)
	}
	return loadProgress(s.Elapsed+sinceEvent, s.LastS)
}

// big number and note of a hero tile for a progress.
func (r rc) progressHero(p progress) (n, unit, note string) {
	if p.Ready {
		return "100", "%", fmt.Sprintf("%d s", int(p.Elapsed.Seconds()+0.5))
	}
	if p.SecondsOnly {
		return fmt.Sprintf("%d", int(p.Elapsed.Seconds())), "s", r.T("first time: seconds only")
	}
	if p.Slower {
		return fmt.Sprintf("%d", p.Pct), "%", r.T("slower than last time")
	}
	return fmt.Sprintf("%d", p.Pct), "%", fmt.Sprintf("%d s ", int(p.Elapsed.Seconds())) + r.Tf("of ~%s", fmt.Sprintf("%d s", int(p.Last.Seconds()+0.5)))
}

// progressBar draws the bar, or the elapsed seconds when there is no
// measurement to fill it against.
func (r rc) progressBar(p progress, w int) string {
	if p.Ready {
		return bar(1, w, cy)
	}
	if p.SecondsOnly {
		return dim.Render(fmt.Sprintf("%d s ", int(p.Elapsed.Seconds())) + r.T("elapsed · no earlier run to compare"))
	}
	s := cy
	if p.Slower {
		s = ye
	}
	return bar(p.Frac, w, s)
}
