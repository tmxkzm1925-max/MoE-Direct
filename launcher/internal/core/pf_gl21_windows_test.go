//go:build windows

package core

// GL-21: the read-back through a start (F-PF §8.5). The fake child (stopwarmup_test.go, mode
// «metrics») writes the engine's header to the metrics path the launcher chose, from the env
// the launcher built, before it listens, as the engine does at the seal.

import (
	"context"
	"strconv"
	"strings"
	"testing"
)

func pf16Live(t *testing.T) {
	for _, c := range []struct {
		name   string
		k, n   string
		want   EnginePrefetch
		suffix string
	}{
		// QD 4 (the sweep) at least N+1: the candidate is on and the engine echoes the pair.
		{"on", "2", "1", EnginePrefetch{Read: true, On: true, K: 2, N: 1}, "candidate on requested_k=2 requested_n=1"},
		// QD 4 below N+1 (F-PF §6.4): off before start, «unset» after ready; the screen follows
		// the engine (TestS8PF02's after-ready half, here through a start).
		{"below QD N+1", "2", "4", EnginePrefetch{Read: true, OffReason: "unset"}, "candidate off(reason=qd_below_prefetch_depth)"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := newLiveRig(t, "metrics answer")
			bindSweep(t, r.f)
			if c.name == "below QD N+1" {
				// GL-10 [A-SWEEP]: with the pair set the sweep prefers a QD of its 90 % set that
				// clears N+1; a record whose 90 % set is QD 4 alone keeps the sweep's QD at 4.
				bindSweepQD4Only(t, r.f)
			}
			req := r.req()
			req.Settings["prefetch_k"], req.Settings["prefetch_n"] = c.k, c.n
			ctx := context.Background()
			ch, err := r.f.l.Start(ctx, req)
			if err != nil {
				t.Fatal(err)
			}
			var final StartEvent
			var atReady *EnginePrefetch
			for ev := range ch {
				if ev.Step == StepReady && ev.State == StepDone && !ev.Final {
					atReady = ev.Engine
				}
				if ev.Final {
					final = ev
				}
			}
			if !final.Ready {
				t.Fatalf("not ready: %v", final.Err)
			}
			if st := final.Preflight.Config.QD; st != 4 {
				t.Fatalf("QD %d, want the sweep's 4", st)
			}
			if final.Engine == nil || *final.Engine != c.want || atReady == nil || *atReady != c.want {
				t.Fatalf("engine %+v at ready %+v", final.Engine, atReady)
			}
			line := PostReadyLine(*final.Engine, final.Preflight.Start.PrefetchEcho)
			if !strings.HasSuffix(line, c.suffix) || !strings.Contains(line, "prefetch_on="+strconv.FormatBool(c.want.On)+" ") {
				t.Errorf("line %s", line)
			}
			st, err := r.f.l.Status(ctx)
			if err != nil || st.Engine == nil || *st.Engine != c.want {
				t.Errorf("status engine %+v %v", st.Engine, err)
			}
			if diagKinds(t, r.f.l, "PREFETCH_ENGINE") != 1 {
				t.Error("no PREFETCH_ENGINE diagnostic line")
			}
			r.stop(t)
		})
	}
}
