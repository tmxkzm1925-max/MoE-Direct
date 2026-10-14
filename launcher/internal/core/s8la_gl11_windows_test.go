//go:build windows

package core

// GL-11 (version 8): the live part of TestS8LA25 (s8la_gl11_test.go). Each start runs the test
// binary as the fake server child (fakeServerExe, mode «smoke»: the check run's endpoints, the
// engine's raw seal line and its cancel and release lines on stderr) inside the launcher's job
// object, on a free loopback port; item 6 reads the real listener table of that child's port.

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"
)

// la25Start runs one start of the rig with the check setting and extra settings.
func la25Start(t *testing.T, r *liveRig, mode SmokeMode, extra map[string]string) (StartEvent, []StartEvent) {
	t.Helper()
	req := r.req()
	req.Smoke = mode
	for k, v := range extra {
		req.Settings[k] = v
	}
	return la25Run(t, r, req)
}

func la25Run(t *testing.T, r *liveRig, req Request) (StartEvent, []StartEvent) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	ch, err := r.f.l.Start(ctx, req)
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	var evs []StartEvent
	var final StartEvent
	for ev := range ch {
		evs = append(evs, ev)
		if ev.Final {
			final = ev
		}
	}
	return final, evs
}

// smokeSteps counts the start's «작동 확인» step events by state.
func smokeSteps(evs []StartEvent) map[StepState]int {
	n := map[StepState]int{}
	for _, ev := range evs {
		if ev.Step == StepSmoke && !ev.Final {
			n[ev.State]++
		}
	}
	return n
}

func la25Live(t *testing.T) {
	t.Run("off: no check runs", func(t *testing.T) {
		r := newLiveRig(t, "smoke break_exit0")
		final, evs := la25Start(t, r, SmokeOff, nil)
		if !final.Ready || final.Smoke != nil || len(smokeSteps(evs)) != 0 {
			t.Fatalf("off: ready=%v smoke=%v steps=%v err=%v", final.Ready, final.Smoke, smokeSteps(evs), final.Err)
		}
		d := diagRecords(t, r.f.l)
		if len(d["SMOKE"]) != 0 || len(d["SMOKE_ITEM"]) != 0 {
			t.Errorf("off wrote %d SMOKE and %d SMOKE_ITEM records", len(d["SMOKE"]), len(d["SMOKE_ITEM"]))
		}
		r.stop(t)
	})

	t.Run("on: the checklist after ready, one log line per item", func(t *testing.T) {
		r := newLiveRig(t, "smoke break_exit0")
		final, evs := la25Start(t, r, SmokeOn, nil)
		if !final.Ready || final.Smoke == nil || !final.Smoke.OK || len(final.Smoke.Items) != 7 {
			t.Fatalf("on: ready=%v smoke=%+v err=%v", final.Ready, final.Smoke, final.Err)
		}
		if st := smokeSteps(evs); st[StepRunning] != 1 || st[StepDone] != 1 || st[StepFailed] != 0 {
			t.Errorf("steps %v", st)
		}
		d := diagRecords(t, r.f.l)
		items := d["SMOKE_ITEM"]
		if len(items) != 7 || len(d["SMOKE"]) != 1 {
			t.Fatalf("records: %d SMOKE_ITEM, %d SMOKE", len(items), len(d["SMOKE"]))
		}
		for i, rec := range items {
			it := final.Smoke.Items[i]
			want := map[bool]string{true: "PASS", false: "FAIL"}[it.OK]
			if it.Skipped {
				want = "SKIP"
			}
			if rec["item"] != float64(i+1) || rec["result"] != want || rec["line"] != it.Line() || rec["name"] != it.Name {
				t.Errorf("record %d %v, item %+v", i, rec, it)
			}
		}
		if it := final.Smoke.Items[5]; it.Detail != "127.0.0.1" { // the real listener table of the child's port
			t.Errorf("item 6 %+v", it)
		}
		if it := final.Smoke.Items[4]; !strings.Contains(it.Detail, "slots 6/6; launcher gate manifest ") {
			t.Errorf("item 5 %+v", it)
		}
		if it := final.Smoke.Items[6]; !it.Skipped || it.Line() != "[7] built-in web UI                  SKIP (API-only bundle)" {
			t.Errorf("item 7 %+v", it)
		}
		if final.Smoke.Elapsed <= 0 || d["SMOKE"][0]["ok"] != true {
			t.Errorf("summary %v", d["SMOKE"][0])
		}
		if st := r.stop(t); st.Status != StatusOK {
			t.Errorf("stop after a passed check: %s", st.Status)
		}
		// On runs at every start.
		final, _ = la25Start(t, r, SmokeOn, nil)
		if !final.Ready || final.Smoke == nil || len(diagRecords(t, r.f.l)["SMOKE"]) != 2 {
			t.Errorf("second on start: ready=%v smoke=%v", final.Ready, final.Smoke)
		}
		r.stop(t)
	})

	t.Run("first once: not again when only ctx or the budget changes", func(t *testing.T) {
		r := newLiveRig(t, "smoke break_exit0")
		count := func() int { return len(diagRecords(t, r.f.l)["SMOKE"]) }
		final, _ := la25Start(t, r, SmokeFirstOnce, nil)
		if !final.Ready || final.Smoke == nil || !final.Smoke.OK || count() != 1 {
			t.Fatalf("first: ready=%v smoke=%+v count=%d err=%v", final.Ready, final.Smoke, count(), final.Err)
		}
		r.stop(t)
		for _, extra := range []map[string]string{{"ctx": "8192"}, {"budget_mb": strconv.Itoa(20000)}} {
			final, evs := la25Start(t, r, SmokeFirstOnce, extra)
			if !final.Ready || final.Smoke != nil || len(smokeSteps(evs)) != 0 || count() != 1 {
				t.Errorf("%v: ready=%v smoke=%v steps=%v count=%d err=%v", extra, final.Ready, final.Smoke, smokeSteps(evs), count(), final.Err)
			}
			r.stop(t)
		}
		// The setting on still runs it (first once counts per model and mode, on counts nothing).
		final, _ = la25Start(t, r, SmokeOn, map[string]string{"ctx": "8192"})
		if final.Smoke == nil || count() != 2 {
			t.Errorf("on after first once: smoke=%v count=%d", final.Smoke, count())
		}
		r.stop(t)
	})

	t.Run("failure: fail_smoke, exit 6, the result on the final event", func(t *testing.T) {
		r := newLiveRig(t, "smoke no_seal break_exit0")
		final, evs := la25Start(t, r, SmokeOn, nil)
		if final.Ready || final.Status != StatusFailSmoke || final.ExitCode != 6 {
			t.Fatalf("status %s code %d err %v", final.Status, final.ExitCode, final.Err)
		}
		if final.Smoke == nil || final.Smoke.OK || strings.Join(final.Smoke.Failures, ",") != "5:verify-consume" {
			t.Fatalf("final smoke %+v", final.Smoke)
		}
		if st := smokeSteps(evs); st[StepFailed] != 1 || st[StepDone] != 0 {
			t.Errorf("steps %v", st)
		}
		d := diagRecords(t, r.f.l)
		if len(d["SMOKE_ITEM"]) != 7 || d["SMOKE_ITEM"][4]["result"] != "FAIL" || d["SMOKE_ITEM"][4]["detail"] != "no seal success line" {
			t.Errorf("records %v", d["SMOKE_ITEM"])
		}
		if r.f.l.activeSession() {
			t.Error("the failed start left a session")
		}
		// A first-once failure does not count: the next first-once start checks again.
		final, _ = la25Start(t, r, SmokeFirstOnce, nil)
		if final.Smoke == nil || final.Status != StatusFailSmoke {
			t.Errorf("first once after a failure: %s %v", final.Status, final.Smoke)
		}
	})
}
