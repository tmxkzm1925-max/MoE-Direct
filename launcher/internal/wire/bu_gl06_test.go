package wire

import (
	"path/filepath"
	"testing"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// TestS8BU08 (S8-BU-08; GL-06), the adapter side: the check's commit room after the start
// beside free RAM, the safe line at 4 GiB, the pool term on its own, the near-limit line's
// numbers (U-13: shown only), the serving page's expert-cache split (U-20 (i): T_pool, file
// mapping «not measured yet», other = the rest of the fit's host terms) and E.3's maximum
// beside the cap on 5 Settings. The core side is internal/core TestS8BU08; the screens are
// internal/ui TestS8BU08.
func TestS8BU08(t *testing.T) {
	pred := core.CommitPrediction{PoolMiB: 12285.9, MirrorMiB: 4537.0, HostRemainderMiB: 3656.6}
	base := func() *core.PreflightResult {
		return &core.PreflightResult{FreeMiB: 20480, FreeForStartMiB: 20480, RAM: core.RAMVerdict{TotalMiB: 32768, AvailMiB: 20480},
			Commit: core.CommitReading{LimitMiB: 39323.7, TotalMiB: 13877.0}, CommitOK: true, Prediction: pred, PredictionOK: true,
			RoomAfterMiB: 4967.2, Check: core.StartCheck{Kind: core.CheckRoom}}
	}
	// (1) room: commit left after beside free RAM, the safe line, the pool apart, no near line
	p, _ := preflightFrom("m", base(), nil)
	near(t, "commit left after", p.CommitLeftAfterGB, 4967.2/1024)
	near(t, "free now", p.AvailGB, 20)
	near(t, "commit limit", p.CommitLimitGB, 39323.7/1024)
	near(t, "commit now", p.CommitNowGB, 13877.0/1024)
	near(t, "safe line", p.CommitSafeGB, 4)
	near(t, "pool", p.PoolGB, 12285.9/1024)
	near(t, "predicted", p.PredictedGB, pred.Total()/1024)
	if p.Near != nil || p.Refused {
		t.Fatalf("room: near %+v refused %v", p.Near, p.Refused)
	}
	// (2) near the limit: the numbers, and the verdict and the start stay as they are
	res := base()
	res.RoomAfterMiB = 1000
	res.Near, res.NearOK = core.NearLimit{Room: 1000, SafeLine: 4096, Limit: 35356.5, Now: 13877.0, Predicted: pred.TotalNoCache(), Pool: pred.PoolMiB}, true
	p, full := preflightFrom("m", res, nil)
	if p.Near == nil || p.Refused || full != "" || p.Verdict != ui.VerdictRoom {
		t.Fatalf("near: %+v", p)
	}
	n := *p.Near
	for _, c := range []struct {
		what      string
		got, want float64
	}{{"room", n.RoomGB, 1000.0 / 1024}, {"safe", n.SafeGB, 4}, {"limit", n.LimitGB, 35356.5 / 1024}, {"now", n.NowGB, 13877.0 / 1024},
		{"predicted", n.PredictedGB, pred.TotalNoCache() / 1024}, {"pool", n.PoolGB, pred.PoolMiB / 1024}} {
		near(t, "near "+c.what, ui.K(c.got), c.want)
	}
	// (3) no reading, no prediction: «not measured yet», the safe line still 4 GiB
	u := base()
	u.CommitOK, u.PredictionOK = false, false
	if p, _ := preflightFrom("m", u, nil); p.CommitLeftAfterGB.State != ui.Unmeasured || p.CommitLimitGB.State != ui.Unmeasured ||
		p.PoolGB.State != ui.Unmeasured || p.Near != nil || !p.CommitSafeGB.Ok() {
		t.Fatalf("unmeasured: %+v", p)
	}
	// (4) the serving page's split
	r := ui.MemoryReading{}
	splitInto(&r, pred, true, true)
	near(t, "expert cache", r.ExpertCacheGB, pred.PoolMiB/1024)
	near(t, "other", r.OtherGB, pred.HostRemainderMiB/1024)
	near(t, "safe line", r.CommitSafeGB, 4)
	if r.FileMapGB.State != ui.Unmeasured {
		t.Fatalf("file mapping %+v", r.FileMapGB)
	}
	r = ui.MemoryReading{}
	if splitInto(&r, core.CommitPrediction{}, false, true); r.ExpertCacheGB.State != ui.Unmeasured || r.OtherGB.Ok() {
		t.Fatalf("no prediction: %+v", r)
	}
	// (5) 5 Settings: E.3's maximum beside the cap, U before a check gave one
	root := t.TempDir()
	l, err := core.New(core.Options{CatalogPath: filepath.Join(root, "models.json"), BundleRoot: t.TempDir(), StateDir: filepath.Join(root, "state")})
	if err != nil {
		t.Fatal(err)
	}
	b := New(l, filepath.Join(root, "models.json"), "test")
	if li, _ := b.Launcher(); li.CacheCapMax.State != ui.Unmeasured || li.CacheCap != "off" {
		t.Fatalf("before a check: %+v %q", li.CacheCapMax, li.CacheCap)
	}
	b.capMax, b.capMaxOK = 4096, true
	if li, _ := b.Launcher(); !li.CacheCapMax.Ok() || li.CacheCapMax.V != 4096 || li.CacheCap != "off" {
		t.Fatalf("after a check: %+v %q (the maximum is not the default)", li.CacheCapMax, li.CacheCap)
	}
}
