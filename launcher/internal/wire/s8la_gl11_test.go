package wire

// GL-11 (version 8): the adapter's part of S8-LA-25 (plan reviews/_plan_8_STAGE1_v8-plansplit.md
// at d1dbe526f; F-LS edition reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, conflict
// row 18-1 [A-SMK]): the check run's result goes to 4 Log, one line per item; a pass shows
// nothing on the screen; a failure is one line at the place of the start verdict (H7).

import (
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

func la25Result(ok bool) *core.SmokeResult {
	r := &core.SmokeResult{OK: ok, Elapsed: 2 * time.Second, Items: []core.SmokeItem{
		{N: 1, Name: "/health 200", OK: true}, {N: 2, Name: "/v1/models", OK: true},
		{N: 3, Name: "chat completion stream+non-stream", OK: true}, {N: 4, Name: "cancel + slot reclaim", OK: true, Detail: "aborted=true"},
		{N: 5, Name: "verify PASS consumption", OK: true, Detail: "engine seal line x1, slots 6/6; launcher gate manifest abababababab"},
		{N: 6, Name: "loopback-only binding", OK: true, Detail: "127.0.0.1"},
		{N: 7, Name: "built-in web UI", OK: true, Skipped: true, Detail: "API-only bundle"},
	}}
	if !ok {
		r.Items[4].OK, r.Items[4].Detail = false, "no seal success line"
		r.Failures = []string{"5:verify-consume"}
	}
	return r
}

func TestS8LA25(t *testing.T) {
	t.Run("4 Log: one line per item, then the summary", func(t *testing.T) {
		for _, ok := range []bool{true, false} {
			b := testBackend()
			r := la25Result(ok)
			b.logChecks(core.StartEvent{Final: true, Ready: ok, Smoke: r})
			if len(b.events) != 8 {
				t.Fatalf("ok=%v: %d lines", ok, len(b.events))
			}
			for i, it := range r.Items {
				ln := b.events[i]
				want := ui.LogInfo
				if !it.OK && !it.Skipped {
					want = ui.LogError
				}
				if ln.Text != "check run "+it.Line() || ln.Level != want {
					t.Errorf("ok=%v line %d: %v %q", ok, i, ln.Level, ln.Text)
				}
			}
			sum := b.events[7]
			if ok && (sum.Level != ui.LogOK || sum.Text != "check run passed: smoke PASS (7 items, 2.0 s)") ||
				!ok && (sum.Level != ui.LogError || sum.Text != "check run failed: smoke FAIL: 5:verify-consume (2.0 s)") {
				t.Errorf("ok=%v summary %v %q", ok, sum.Level, sum.Text)
			}
		}
		// Off: no check ran, no line.
		b := testBackend()
		b.logChecks(core.StartEvent{Final: true, Ready: true})
		if len(b.events) != 0 {
			t.Errorf("off wrote %v", b.events)
		}
	})

	t.Run("a pass shows nothing on the screen", func(t *testing.T) {
		tr := newStartTrack("m")
		t0 := time.Unix(10, 0)
		r := la25Result(true)
		tr.apply(core.StartEvent{Step: core.StepSmoke, State: core.StepRunning}, t0)
		u := tr.apply(core.StartEvent{Step: core.StepSmoke, State: core.StepDone, Detail: r.Line(), Smoke: r}, t0.Add(2*time.Second))
		st := u.Steps[stepIndex(core.StepSmoke)]
		if st.Detail != "" || st.State != ui.StepDone {
			t.Errorf("passed check row %+v", st)
		}
		u = tr.apply(core.StartEvent{Final: true, Ready: true, Status: core.StatusOK, Smoke: r, Address: "http://127.0.0.1:8093"}, t0.Add(3*time.Second))
		for _, s := range u.Steps {
			if strings.Contains(s.Detail, "PASS") || strings.Contains(s.Detail, "smoke") {
				t.Errorf("a step shows the pass: %+v", s)
			}
		}
		for _, n := range u.Notices {
			if strings.Contains(n, "check run") {
				t.Errorf("notice %q", n)
			}
		}
	})

	t.Run("a failure is one line at the start verdict", func(t *testing.T) {
		r := la25Result(false)
		ev := core.StartEvent{Final: true, Status: core.StatusFailSmoke, ExitCode: 6, Smoke: r,
			Err: &core.LaunchError{Status: core.StatusFailSmoke, Reason: "smoke assertions failed: 5:verify-consume"}}
		f := failureFrom(ev, time.Unix(1, 0), nil)
		if !f.SmokeFailed || f.ExitCode != 6 || f.Why != "failed items 5:verify-consume" {
			t.Errorf("failure %+v", f)
		}
		tr := newStartTrack("m")
		u := tr.apply(core.StartEvent{Step: core.StepSmoke, State: core.StepFailed, Detail: r.Line(), Smoke: r}, time.Unix(1, 0))
		if st := u.Steps[stepIndex(core.StepSmoke)]; st.State != ui.StepFailed || st.Detail != r.Line() {
			t.Errorf("failed check row %+v", st)
		}
	})
}
