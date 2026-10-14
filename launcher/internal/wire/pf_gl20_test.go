package wire

import (
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
)

// S8-PF-08 (the screen part; GL-20, F-PF §7.2, §8.2-§8.3): the virtual notices reach the screen
// as their T18 texts exactly, and the prefetch row shows N beside the smallest N_cap.
func TestS8PF08(t *testing.T) {
	if virtualNoticeWords != core.NoticeVirtualUnvalidated || nCapZeroWords != core.NoticeNCapZero {
		t.Fatalf("the screen words differ from core's texts")
	}
	ns := userNotices([]string{core.NoticeVirtualUnvalidated, core.NoticeNCapZero})
	if len(ns) != 2 || ns[0] != core.NoticeVirtualUnvalidated || ns[1] != core.NoticeNCapZero {
		t.Errorf("notices %q", ns)
	}
	k, n, qd, nc := int64(8), int64(4), int64(8), int64(1)
	on := &core.PrefetchEcho{EffectivePrefetch: "on", RequestedK: &k, RequestedN: &n, RequestedQD: &qd, PrefetchProvenance: core.ProvCatalog,
		PrefetchNCap: &nc, PrefetchNCapPerLayer: []int64{1, 1, 1}}
	s := statusFrom(core.ServeStatus{Start: core.StartSettings{Prefetch: "catalog", PrefetchOn: true, PrefetchEcho: on},
		Engine: &core.EnginePrefetch{Read: true, On: true, K: 8, N: 4}}, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, nil, time.Time{}, false)
	if got := strings.Join(s.Prefetch, "|"); got != "engine on|K 8|N 4|candidate on|K 8|N 4|N_cap 1|catalog-validated" {
		t.Errorf("row %s", got)
	}
	if l := core.PostReadyLine(core.EnginePrefetch{Read: true, On: true, K: 8, N: 4}, on); !strings.HasSuffix(l, "candidate on requested_k=8 requested_n=4 prefetch_n_cap=1") {
		t.Errorf("line %s", l)
	}
	if l := core.NCapLine(on); l != "prefetch_n_cap_per_layer=1,1,1" {
		t.Errorf("log line %s", l)
	}
}
