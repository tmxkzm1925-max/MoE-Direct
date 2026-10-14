package ui

import (
	"strings"
	"testing"
)

// commitFake is a check with the commit prediction (GL-06): 4.9 G of commit left after the
// start, the safe line 4 G, this start's commit 20.0 G of which the expert pool 12.0 G.
func commitFake() *FakeBackend {
	f := NewFake()
	f.Pre.CommitLeftAfterGB, f.Pre.CommitLimitGB, f.Pre.CommitNowGB, f.Pre.CommitSafeGB = K(4.9), K(38.4), K(13.6), K(4)
	f.Pre.PoolGB, f.Pre.PredictedGB = K(12.0), K(20.0)
	return f
}

// TestS8BU08 (S8-BU-08; GL-06), the screens: 3 Memory shows the commit room beside free
// physical RAM, the pool term on its own line and the commit safe line at 4 GiB; near the limit
// a warning line with the numbers, and nothing else: the verdict and the start stay as they
// are, no action is sent (U-13 [A-NEAR]). The serving page splits the model's memory as U-20
// (i) has it; 5 Settings shows E.3's maximum beside the cap. Capture rows are logged.
func TestS8BU08(t *testing.T) {
	// (1) room: both figures, the safe line, the pool line, no warning
	h := newH(t, commitFake(), LangEnglish, 200, 40)
	h.keys("3")
	p := h.plain()
	mustHave(t, p, "AVAILABLE NOW", "COMMIT LEFT", "4.9 GB after", "safe line 4 GB",
		"✓ commit", "4.9 GB left after the start", "expert pool", "12.0 GB · of this start's commit 20.0 GB")
	mustLack(t, p, "near its limit")
	for _, l := range strings.Split(p, "\n") {
		if strings.Contains(l, "COMMIT LEFT") || strings.Contains(l, "expert pool") || strings.Contains(l, "✓ commit") {
			t.Logf("capture: %s", strings.TrimRight(l, " "))
		}
	}

	// (2) near: the warning line with its numbers; no action, the verdict unchanged, ⏎ starts
	nf := commitFake()
	nf.Pre.CommitLeftAfterGB = K(1.0)
	nf.Pre.Near = &NearLine{RoomGB: 1.0, SafeGB: 4, LimitGB: 34.5, NowGB: 13.6, PredictedGB: 20.0, PoolGB: 12.0}
	hn := newH(t, nf, LangEnglish, 300, 40)
	hn.keys("3")
	pn := hn.plain()
	want := "commit near its limit: 1.0 GB left after the start is below the safe line 4.0 GB · limit 34.5 GB · in use now 13.6 GB · this start 20.0 GB, of it the expert pool 12.0 GB · nothing is changed"
	mustHave(t, pn, "▲ commit", "▶ ● starts with room")
	mustHave(t, flat(pn), flat("▲ "+want)) // the line wraps inside the box (review item 32)
	mustLack(t, pn, "refused", "✕")
	for _, c := range nf.Calls() {
		if strings.HasPrefix(c, "Action") || strings.HasPrefix(c, "Start") || strings.HasPrefix(c, "Set") {
			t.Fatalf("the near line sent %q (%v)", c, nf.Calls())
		}
	}
	for _, l := range strings.Split(pn, "\n") {
		if strings.Contains(l, "near its limit") {
			t.Logf("capture: %s", strings.TrimSpace(l))
		}
	}
	hn.keys("enter")
	if hn.calls("Start") != 1 {
		t.Fatalf("a start near the limit did not start: %v", nf.Calls())
	}
	hk := newH(t, nf, LangKorean, 300, 40)
	hk.keys("3")
	mustHave(t, flat(hk.plain()), flat("커밋이 한도에 가까움: 시작 뒤 남는 1.0 GB 가 안전선 4.0 GB 아래 · 한도 34.5 GB · 지금 사용 13.6 GB · 이번 시작 20.0 GB, 그중 전문가 메모리 예산 12.0 GB · 아무것도 바꾸지 않음"))

	// (3) the serving page: expert cache T_pool, file mapping not measured yet, other, safe line
	sf := NewFake()
	sf.Mem.ExpertCacheGB, sf.Mem.BudgetGB, sf.Mem.FileMapGB, sf.Mem.OtherGB = K(12.0), K(12), U, K(1.1)
	sf.Mem.CommitUsedGB, sf.Mem.CommitLimitGB, sf.Mem.CommitSafeGB = K(30.0), K(38.4), K(4)
	hs := newH(t, sf, LangEnglish, 200, 40)
	hs.keys(serving("3")...)
	ps := hs.plain()
	mustHave(t, ps, "expert cache 12.0", "file mapping not measured yet", "other 1.1")
	mustLack(t, ps, "expert cache split")
	uf := NewFake()
	uf.Mem.ExpertCacheGB = U
	hu := newH(t, uf, LangEnglish, 200, 40)
	hu.keys(serving("3")...)
	mustHave(t, hu.plain(), "expert cache split: not measured yet")

	// (4) 5 Settings: the maximum beside the cap, the default still off
	cf := NewFake()
	cf.Info.CacheCap, cf.Info.CacheCapMax = "off", K(4096)
	hc := newH(t, cf, LangEnglish, 200, 40)
	hc.keys("5")
	mustHave(t, hc.plain(), "off · the default · max 4096 MiB at the last check")
}
