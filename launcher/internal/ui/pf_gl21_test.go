package ui

import "testing"

// S8-PF-16 (the drawn part; GL-21, F-PF §8.5): the serve page's «this run» frame holds the
// prefetch row, the engine's values after ready and then the candidate, in the screen language.
func TestS8PF16(t *testing.T) {
	for _, c := range []struct {
		lang string
		want []string
	}{
		// UX-KR (decision 4): the engine's values only; the candidate stays in 4 Log
		{LangEnglish, []string{"prefetch", "applied · K 8 · N 4"}}, // UX-R9: the label in words
		{LangKorean, []string{"프리페치", "적용됨 · K 8 · N 4"}},
	} {
		f := NewFake()
		f.Statuses[0].Prefetch = []string{"engine on", "K 8", "N 4", "candidate on", "K 8", "N 4", "catalog-validated"}
		hs := newH(t, f, c.lang, 120, 34)
		hs.keys("enter")
		hs.beat()
		mustHave(t, hs.plain(), c.want...)
		mustLack(t, hs.plain(), tr(c.lang, "candidate on"), tr(c.lang, "catalog values, measured"))
	}
	// Without the row (an older backend) the frame is as before.
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	hs.keys("enter")
	hs.beat()
	mustLack(t, hs.plain(), "engine on", "candidate on")
}
