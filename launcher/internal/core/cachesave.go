package core

import (
	"context"
	"time"
)

// GL-07 (version 8): the cache-save bar of decision ③ (DECISION_LEDGER.md:4179; F-LS edition
// 2.3 and 3.5; plan scenario S8-LA-15). The bar spans the quit action to the save-completion
// response. Its expectation is the last save's time per token times this save's tokens, kept in
// the launcher's duration history (history.go, CacheSaveRecord). A first save shows elapsed
// seconds only; an overrun holds at 99 % («slower than last time»); 100 % only at the
// completion response. No engine progress signal is read: the save answers once, at its end
// (LAUNCH-FEAS Q3.133).
//
// The save itself (the slot save request of warm start) is GL-15's: it calls TimedCacheSave
// with its request as save, and puts each CacheSaveTrack on its stop events (StopEvent.Save).

// CacheSaveTrack is the cache-save bar at one moment.
type CacheSaveTrack struct {
	Tokens    int           // this save's tokens (the expectation's multiplier)
	Elapsed   time.Duration // since the quit action
	Expect    time.Duration // last time per token × Tokens; zero when HasExpect is false
	HasExpect bool
	Done      bool // the save-completion response arrived
	Saved     int  // the tokens the response reports saved (Done only)
	Progress  Progress
}

// cacheSavePoll is how often TimedCacheSave reports the bar while the save runs.
var cacheSavePoll = 250 * time.Millisecond

// CacheSaveExpectation is the expected duration of a save of tokens tokens from the last save's
// record: its time per token times tokens. ok is false without a usable record (a first save).
func CacheSaveExpectation(rec CacheSaveRecord, tokens int) (time.Duration, bool) {
	if rec.Tokens <= 0 || rec.Millis <= 0 || tokens <= 0 {
		return 0, false
	}
	perToken := float64(rec.Millis) / float64(rec.Tokens) // ms per token
	return time.Duration(perToken * float64(tokens) * float64(time.Millisecond)), true
}

// TimedCacheSave runs one cache save under the bar. key names the model and settings (the load
// bar's key, PreflightResult.SettingsKey); tokens is this save's token count as known before the
// request; quitAt is the quit action. save performs the request and returns when its completion
// response arrives, with the tokens it reports saved. emit receives the bar at every poll and
// once at the end. A completed save records its duration over the tokens saved for the next
// save of the key; a failed or cancelled save records nothing.
func (l *Launcher) TimedCacheSave(ctx context.Context, key string, tokens int, quitAt time.Time,
	save func(context.Context) (int, error), emit func(CacheSaveTrack)) (CacheSaveTrack, error) {
	tr := CacheSaveTrack{Tokens: tokens}
	if rec, ok := l.LastCacheSave(key); ok {
		tr.Expect, tr.HasExpect = CacheSaveExpectation(rec, tokens)
	}
	l.diag.write("CACHE_SAVE_EXPECT", map[string]any{"tokens": tokens, "has_expect": tr.HasExpect,
		"expect_ms": tr.Expect.Milliseconds()})
	at := func(done bool) CacheSaveTrack {
		t := tr
		t.Elapsed = l.os.now().Sub(quitAt)
		t.Done = done
		t.Progress = LoadProgress(t.Elapsed, t.Expect, t.HasExpect, done)
		return t
	}
	type result struct {
		n   int
		err error
	}
	ch := make(chan result, 1)
	go func() {
		n, err := save(ctx)
		ch <- result{n, err}
	}()
	tk := time.NewTicker(cacheSavePoll)
	defer tk.Stop()
	for {
		select {
		case r := <-ch:
			if r.err != nil {
				t := at(false)
				l.diag.write("CACHE_SAVE_FAILED", map[string]any{"reason": r.err.Error(), "ms": t.Elapsed.Milliseconds()})
				return t, r.err
			}
			t := at(true)
			t.Saved = r.n
			if emit != nil {
				emit(t)
			}
			if err := l.recordCacheSave(CacheSaveRecord{Key: key, Tokens: r.n, Millis: t.Elapsed.Milliseconds(),
				At: l.os.now().UTC().Format(time.RFC3339)}); err != nil {
				l.diag.write("HISTORY_WRITE_FAILED", map[string]any{"reason": err.Error()})
			}
			l.diag.write("CACHE_SAVE_DONE", map[string]any{"tokens": r.n, "ms": t.Elapsed.Milliseconds()})
			return t, nil
		case <-tk.C:
			if emit != nil {
				emit(at(false))
			}
		}
	}
}
