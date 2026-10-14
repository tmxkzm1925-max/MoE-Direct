// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// The adapter's own computations (FIX-WIRE): the five-second decode window (review item 23),
// the preflight reuse rule (items 13, 20, 22) and the stopped run's summary (H12, item 23).
// Pure: the clock is passed in.

package wire

import (
	"sort"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// decodeSpan is the live decode window: «the last 5 s» as the screen labels it (item 23).
const decodeSpan = 5 * time.Second

type decodeSample struct {
	at   time.Time
	task int64
	n    int64
}

// decodeWindow keeps the /slots n_decoded samples of the request in flight.
type decodeWindow struct {
	s []decodeSample
}

// add records one sample; a sample of another task, or an idle server, starts over.
func (w *decodeWindow) add(at time.Time, task, n int64, decoding bool) {
	if !decoding {
		w.s = w.s[:0]
		return
	}
	if len(w.s) > 0 {
		last := w.s[len(w.s)-1]
		if last.task != task || n < last.n || !at.After(last.at) {
			w.s = w.s[:0]
		}
	}
	w.s = append(w.s, decodeSample{at: at, task: task, n: n})
	// keep one sample at or before the window's start, so a full window always has a base
	cut := 0
	for i := range w.s {
		if at.Sub(w.s[i].at) >= decodeSpan {
			cut = i
		}
	}
	w.s = w.s[cut:]
}

// rate is the tokens per second over the last five seconds: from the newest sample at or before
// the window's start (or the oldest kept) to the newest sample.
func (w *decodeWindow) rate(now time.Time) (float64, bool) {
	if len(w.s) < 2 {
		return 0, false
	}
	first, last := w.s[0], w.s[len(w.s)-1]
	span := last.at.Sub(first.at).Seconds()
	if span <= 0 || now.Sub(last.at) > decodeSpan {
		return 0, false
	}
	return float64(last.n-first.n) / span, true
}

// preflightReuse is how long a preflight result stands for the same model and settings while
// nothing else changed; the RAM numbers are re-read at every call regardless (item 13).
const preflightReuse = 10 * time.Second

// preEntry is one remembered preflight.
type preEntry struct {
	res     *core.PreflightResult
	err     error
	at      time.Time
	serving bool // taken while a server of this launcher was up
}

// stale is the re-run rule: a result taken while a server was up is never reused once it has
// stopped (row 20: those results carried the serving free RAM), nor the other way round, nor
// after preflightReuse.
func (e *preEntry) stale(now time.Time, serving bool) bool {
	return e == nil || e.serving != serving || now.Sub(e.at) >= preflightReuse
}

// freshen re-reads the RAM side of a remembered result: free RAM now (plus this server's share
// while it serves) against the same need and budget gives item 22's verdict again.
func freshen(res *core.PreflightResult, availMiB, serverMiB int64) *core.PreflightResult {
	if res == nil {
		return nil
	}
	c := *res
	c.FreeMiB, c.ServerMiB, c.FreeForStartMiB = availMiB, serverMiB, availMiB+serverMiB
	c.RAM.AvailMiB = c.FreeForStartMiB
	budget := int64(0)
	if c.Config != nil {
		budget = c.Config.Budget.MB
	}
	c.Check = core.EvaluateStart(c.FreeForStartMiB, c.NeedMiB, c.NeedBasis, budget)
	if c.NeedKnown {
		c.LeftMiB = c.Check.LeftMiB
		c.Verdict = core.MemFits
		if c.Check.Kind == core.CheckTight {
			c.Verdict = core.MemShort
		}
	}
	return &c
}

// runSummary is H12 from the run's print_timing lines: the decode average is the server's
// whole-answer average (tokens over time of every reply, item 23), the lowest is the slowest
// reply's own average.
func runSummary(model string, started, ended time.Time, tm core.TimingSummary) ui.RunSummary {
	r := ui.RunSummary{Model: model, Started: started, Ended: ended, Requests: ui.K(float64(tm.Requests)),
		Errors: ui.Val{}, DecodeAvg: ui.U, DecodeLow: ui.U, PrefillAvg: ui.U, PrefillTokens: ui.U,
		TTFTMedian: ui.U, TTFTMax: ui.U, CacheTokens: ui.Val{}}
	if avg, ok := tm.RunDecodeAvg(); ok {
		r.DecodeAvg = ui.K(avg)
	}
	if tm.PromptTime > 0 {
		r.PrefillAvg = ui.K(float64(tm.PromptTokens) / tm.PromptTime.Seconds())
		r.PrefillTokens = ui.K(float64(tm.PromptTokens))
	}
	if len(tm.Done) == 0 {
		return r
	}
	low := tm.Done[0].DecodeTokPerSec
	ttft := make([]float64, 0, len(tm.Done))
	for _, d := range tm.Done {
		if d.DecodeTokPerSec < low {
			low = d.DecodeTokPerSec
		}
		ttft = append(ttft, d.TTFT.Seconds())
	}
	r.DecodeLow = ui.K(low)
	sort.Float64s(ttft)
	med := ttft[len(ttft)/2]
	if len(ttft)%2 == 0 {
		med = (ttft[len(ttft)/2-1] + ttft[len(ttft)/2]) / 2
	}
	r.TTFTMedian, r.TTFTMax = ui.K(med), ui.K(ttft[len(ttft)-1])
	return r
}
