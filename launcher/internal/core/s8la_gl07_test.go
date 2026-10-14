package core

// GL-07 (version 8): core's part of S8-LA-14 and S8-LA-15 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f, cell GL-07 and scenario rows; F-LS
// edition reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, rows 2.3 and 3.5; decision ③,
// DECISION_LEDGER.md:4179), on fake clocks. The live part of S8-LA-14 (two starts of the fake
// child) is in s8la_gl07_windows_test.go.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"
)

// lockedClock is a fake clock shared between a test and the goroutines it drives.
type lockedClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *lockedClock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *lockedClock) add(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// pct is the percent a fill shows (the screen prints the whole percent).
func pct(f float64) int { return int(f*100 + 1e-9) }

// S8-LA-14: the load bar per tick on a fake clock, the ready loop's elapsed time measured from
// the child's start, and (live, Windows) the duration recorded for the combination at ready.
func TestS8LA14(t *testing.T) {
	t.Run("percent and text per tick", func(t *testing.T) {
		const D = 20 * time.Second
		for s := 0; s <= 30; s++ {
			el := time.Duration(s) * time.Second
			first := LoadProgress(el, 0, false, false)
			if !first.SecondsOnly || first.Fraction != 0 || first.Elapsed != el {
				t.Errorf("first start, %d s: %+v", s, first)
			}
			p := LoadProgress(el, D, true, false)
			switch {
			case el < D:
				want := float64(el) / float64(D)
				if want > 0.99 {
					want = 0.99
				}
				if p.SecondsOnly || p.Held || p.Fraction != want || pct(p.Fraction) != pct(want) {
					t.Errorf("%d s of %v: %+v", s, D, p)
				}
			default: // at or past D before ready: held at 99 % with the overrun flag
				if p.Fraction != 0.99 || pct(p.Fraction) != 99 || !p.Held {
					t.Errorf("%d s, at or past %v: %+v", s, D, p)
				}
			}
			if p.Fraction >= 1 || first.Fraction >= 1 {
				t.Errorf("%d s: 100 %% before ready", s)
			}
		}
		if r := LoadProgress(31*time.Second, D, true, true); r.Fraction != 1 || r.Held {
			t.Errorf("ready: %+v", r)
		}
		if r := LoadProgress(9*time.Second, 0, false, true); r.Fraction != 1 || !r.SecondsOnly {
			t.Errorf("ready on a first start: %+v", r)
		}
	})

	t.Run("ready loop ticks on a fake clock", func(t *testing.T) {
		// /health answers 503 «Loading model» for five polls, then 200; the listener is the child's.
		polls := 0
		var mu sync.Mutex
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			polls++
			n := polls
			mu.Unlock()
			if n <= 5 {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			fmt.Fprint(w, `{"status":"ok"}`)
		}))
		defer srv.Close()
		host, ps, _ := net.SplitHostPort(srv.Listener.Addr().String())
		port, _ := strconv.Atoi(ps)
		child := &fakeChild{pid: 4242}
		owners := func(int) ([]int, error) { return []int{4242}, nil }
		clk := &lockedClock{t: time.Unix(1000, 0)}
		var ticks []time.Duration
		tm := readyTimings{Timeout: time.Hour, Poll: 4 * time.Second, HealthTimeout: 5 * time.Second}
		err := waitForServerReady(context.Background(), child, host, port, "", owners, tm, clk.now, clk.add,
			func(el time.Duration) { ticks = append(ticks, el) })
		if err != nil {
			t.Fatal(err)
		}
		if len(ticks) != 5 {
			t.Fatalf("ticks %v", ticks)
		}
		const D = 12 * time.Second
		wantPct := []int{0, 33, 66, 99, 99}
		for i, el := range ticks {
			if el != time.Duration(i)*tm.Poll {
				t.Errorf("tick %d at %v, want %v from the child's start", i, el, time.Duration(i)*tm.Poll)
			}
			p := LoadProgress(el, D, true, false)
			if pct(p.Fraction) != wantPct[i] || p.Held != (el >= D) {
				t.Errorf("tick %d (%v of %v): %d %% held %v", i, el, D, pct(p.Fraction), p.Held)
			}
		}
	})

	t.Run("live: two starts of the same combination", la14Live)
}

// S8-LA-15: the cache-save bar spans the quit action to the save-completion response; its
// expectation is the last save's time per token times this save's tokens; a first save shows
// elapsed seconds only; an overrun holds at 99 % with the overrun flag; 100 % only at the
// response; the duration is recorded per key for the next save.
func TestS8LA15(t *testing.T) {
	old := cacheSavePoll
	cacheSavePoll = time.Millisecond
	defer func() { cacheSavePoll = old }()

	f := newFixture(t, false)
	clk := &lockedClock{t: time.Unix(5000, 0)}
	f.l.os.now = clk.now
	const key = "v1|fixture|m.gguf|packed|-c 4096|"

	// run drives one save: the quit action at quitAt, the request answering after the clock has
	// moved by steps; it gives every bar the save reported.
	run := func(tokens, saved int, quitAt time.Time, steps []time.Duration, fail error) ([]CacheSaveTrack, CacheSaveTrack, error) {
		var mu sync.Mutex
		var seen []CacheSaveTrack
		emit := func(tr CacheSaveTrack) { mu.Lock(); seen = append(seen, tr); mu.Unlock() }
		count := func() int { mu.Lock(); defer mu.Unlock(); return len(seen) }
		save := func(context.Context) (int, error) {
			for _, d := range steps {
				clk.add(d)
				n := count()
				for deadline := time.Now().Add(2 * time.Second); count() < n+2 && time.Now().Before(deadline); {
					time.Sleep(time.Millisecond) // let the bar report at this clock
				}
			}
			return saved, fail
		}
		final, err := f.l.TimedCacheSave(context.Background(), key, tokens, quitAt, save, emit)
		mu.Lock()
		defer mu.Unlock()
		return append([]CacheSaveTrack(nil), seen...), final, err
	}
	check := func(name string, seen []CacheSaveTrack, final CacheSaveTrack) {
		t.Helper()
		if len(seen) < 3 {
			t.Fatalf("%s: %d bars", name, len(seen))
		}
		for i, tr := range seen[:len(seen)-1] { // every bar before the response
			p := LoadProgress(tr.Elapsed, tr.Expect, tr.HasExpect, false)
			if tr.Done || tr.Progress != p || tr.Progress.Fraction >= 1 {
				t.Errorf("%s: bar %d before the response: %+v", name, i, tr)
			}
		}
		last := seen[len(seen)-1]
		if !last.Done || last.Progress.Fraction != 1 || last != final {
			t.Errorf("%s: the response's bar %+v (final %+v)", name, last, final)
		}
	}

	// A first save: elapsed seconds only, the span from the quit action (3 s before the request).
	quit := clk.now()
	clk.add(3 * time.Second)
	seen, final, err := run(1000, 1000, quit, []time.Duration{time.Second, time.Second, time.Second}, nil)
	if err != nil {
		t.Fatal(err)
	}
	check("first save", seen, final)
	for _, tr := range seen {
		if tr.HasExpect || !tr.Progress.SecondsOnly {
			t.Errorf("first save shows an expectation: %+v", tr)
		}
	}
	if final.Elapsed != 6*time.Second || final.Saved != 1000 {
		t.Errorf("first save's span %v saved %d, want 6 s from the quit action", final.Elapsed, final.Saved)
	}
	rec, ok := f.l.LastCacheSave(key)
	if !ok || rec.Tokens != 1000 || rec.Millis != 6000 || rec.Saves != 1 {
		t.Fatalf("record after the first save: %+v %v", rec, ok)
	}
	if n := diagKinds(t, f.l, "CACHE_SAVE_EXPECT"); n != 1 {
		t.Errorf("launcher log: %d CACHE_SAVE_EXPECT records", n)
	}

	// A second save of twice the tokens: expectation = 6 ms per token × 2000 = 12 s; ticks at
	// 3, 6, 9, 12 and 15 s: 25, 50, 75, then 99 % held (at and past 12 s); 100 % at the response.
	quit = clk.now()
	seen, final, err = run(2000, 2000, quit, []time.Duration{3 * time.Second, 3 * time.Second, 3 * time.Second, 3 * time.Second, 3 * time.Second}, nil)
	if err != nil {
		t.Fatal(err)
	}
	check("second save", seen, final)
	want := map[time.Duration]struct {
		pct  int
		held bool
	}{3 * time.Second: {25, false}, 6 * time.Second: {50, false}, 9 * time.Second: {75, false},
		12 * time.Second: {99, true}, 15 * time.Second: {99, true}}
	got := map[time.Duration]bool{}
	for _, tr := range seen[:len(seen)-1] {
		if !tr.HasExpect || tr.Expect != 12*time.Second || tr.Tokens != 2000 {
			t.Errorf("expectation %+v", tr)
		}
		if w, ok := want[tr.Elapsed]; ok {
			got[tr.Elapsed] = true
			if pct(tr.Progress.Fraction) != w.pct || tr.Progress.Held != w.held {
				t.Errorf("at %v: %d %% held %v, want %d %% held %v", tr.Elapsed, pct(tr.Progress.Fraction), tr.Progress.Held, w.pct, w.held)
			}
		}
	}
	if len(got) != len(want) {
		t.Errorf("bars seen at %v, want each of 3..15 s", got)
	}
	if rec, _ := f.l.LastCacheSave(key); rec.Tokens != 2000 || rec.Millis != 15000 || rec.Saves != 2 {
		t.Errorf("record after the second save: %+v", rec)
	}
	if e, ok := CacheSaveExpectation(CacheSaveRecord{Tokens: 2000, Millis: 15000}, 400); !ok || e != 3*time.Second {
		t.Errorf("expectation for 400 tokens: %v %v", e, ok)
	}
	if _, ok := CacheSaveExpectation(CacheSaveRecord{}, 400); ok {
		t.Error("an expectation without a record")
	}

	// A failed save records nothing and never shows 100 %.
	quit = clk.now()
	seen, final, err = run(500, 0, quit, []time.Duration{time.Second, time.Second}, errors.New("slot save refused"))
	if err == nil || final.Done || final.Progress.Fraction >= 1 {
		t.Errorf("failed save: %v %+v", err, final)
	}
	for _, tr := range seen {
		if tr.Done || tr.Progress.Fraction >= 1 {
			t.Errorf("failed save showed done: %+v", tr)
		}
	}
	if rec, _ := f.l.LastCacheSave(key); rec.Saves != 2 || rec.Millis != 15000 {
		t.Errorf("a failed save changed the record: %+v", rec)
	}
}
