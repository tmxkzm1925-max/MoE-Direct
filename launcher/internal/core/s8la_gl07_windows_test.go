//go:build windows

package core

// GL-07 (version 8): the live part of S8-LA-14. Two starts of the same model and settings run
// the fake child (stopwarmup_test.go, fakeServerExe; «slow_health» keeps /health at 503 for
// 1.2 s so the ready loop ticks). It is the seconds-layer counterpart of the exclusive leg D1
// row S8-LA-16, which runs the smoke model in the sealed bundle and is not run here.

import (
	"testing"
	"time"
)

func la14Live(t *testing.T) {
	r := newLiveRig(t, "events answer break_exit0 slow_health")
	type startSeen struct {
		key   string
		ticks []StartEvent
		ready StartEvent
	}
	one := func(name string) startSeen {
		t.Helper()
		final, evs := r.start(t, r.f.l)
		if !final.Ready || final.Status != StatusOK {
			t.Fatalf("%s: %+v", name, final)
		}
		var s startSeen
		for _, ev := range evs {
			switch {
			case ev.Preflight != nil && s.key == "":
				s.key = ev.Preflight.SettingsKey
			case ev.Step == StepReady && ev.State == StepRunning && ev.Elapsed > 0:
				s.ticks = append(s.ticks, ev)
			case ev.Step == StepReady && ev.State == StepDone:
				s.ready = ev
			}
		}
		if s.key == "" || len(s.ticks) < 2 || s.ready.Step != StepReady {
			t.Fatalf("%s: key %q, %d ticks, ready %+v", name, s.key, len(s.ticks), s.ready)
		}
		for _, ev := range s.ticks { // 100 % only at ready
			if ev.Progress.Fraction >= 1 || ev.Progress.Elapsed != ev.Elapsed {
				t.Errorf("%s: tick before ready %+v", name, ev.Progress)
			}
		}
		if s.ready.Progress.Fraction != 1 || s.ready.Elapsed < 1200*time.Millisecond {
			t.Errorf("%s: ready %+v after %v", name, s.ready.Progress, s.ready.Elapsed)
		}
		if st := r.stop(t); st.ExitCode != 0 && st.ExitCode != 5 {
			t.Fatalf("%s: stop %+v", name, st)
		}
		return s
	}

	first := one("first start")
	for _, ev := range first.ticks { // a first start of the combination: elapsed seconds only
		if !ev.Progress.SecondsOnly || ev.Progress.HasLast || ev.Progress.Fraction != 0 {
			t.Errorf("first start shows a fill: %+v", ev.Progress)
		}
	}
	rec, ok := r.f.l.HistoryFor(first.key)
	if !ok || rec.ReadyMillis != first.ready.Elapsed.Milliseconds() || rec.Runs != 1 {
		t.Fatalf("record after the first start: %+v %v (took %v)", rec, ok, first.ready.Elapsed)
	}

	second := one("second start")
	if second.key != first.key {
		t.Fatalf("the same model and settings gave two keys:\n%s\n%s", first.key, second.key)
	}
	D := time.Duration(rec.ReadyMillis) * time.Millisecond
	for _, ev := range second.ticks { // fill = elapsed / the first start's recorded duration
		p := ev.Progress
		if p.SecondsOnly || !p.HasLast || p.Last != D || p != LoadProgress(ev.Elapsed, D, true, false) {
			t.Errorf("second start's tick %+v, want the fill against %v", p, D)
		}
	}
	rec2, _ := r.f.l.HistoryFor(first.key)
	if rec2.Runs != 2 || rec2.ReadyMillis != second.ready.Elapsed.Milliseconds() {
		t.Errorf("record after the second start: %+v (took %v)", rec2, second.ready.Elapsed)
	}

	// The launcher log: no expectation at the first start, the first start's duration at the
	// second (S8-LA-16's log lines, here from the fake child).
	exp := diagRecords(t, r.f.l)["LOAD_EXPECT"]
	if len(exp) != 2 || exp[0]["has_last"] != false || exp[1]["has_last"] != true ||
		exp[1]["expect_ms"] != float64(rec.ReadyMillis) {
		t.Errorf("LOAD_EXPECT records %v, want none then %d ms", exp, rec.ReadyMillis)
	}
}
