package wire

import (
	"testing"

	"moedirect/launcher/internal/core"
)

// TestUXKRCandidateLog: UX-KR decision 4. 2 Server's prefetch row holds what the engine applied;
// the launcher's candidate and its reasons, which the row no longer shows, are one 4 Log line
// after ready, in the screen's English sentences (never the codes).
func TestUXKRCandidateLog(t *testing.T) {
	k, n, qd := int64(8), int64(4), int64(8)
	for _, c := range []struct {
		cand *core.PrefetchEcho
		want string
	}{
		{nil, ""},
		{&core.PrefetchEcho{EffectivePrefetch: "off(reason=not_opted_in_evidence_observe)", OffReason: "not_opted_in_evidence_observe", RequestedQD: &qd},
			"prefetch candidate (not shown on 2 Server): candidate off · off by default: a gain on this model is only observed, not proven"},
		{&core.PrefetchEcho{EffectivePrefetch: "on", RequestedK: &k, RequestedN: &n, RequestedQD: &qd, PrefetchProvenance: core.ProvCatalog},
			"prefetch candidate (not shown on 2 Server): candidate on · K 8 · N 4 · catalog values, measured"},
	} {
		if got := candidateLogLine(c.cand); got != c.want {
			t.Errorf("candidate %+v:\n got %q\nwant %q", c.cand, got, c.want)
		}
	}
}
