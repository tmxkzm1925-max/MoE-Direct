package ui

// GL-18 (version 8): the screens' part of S8-LA-37 (plan reviews/_plan_8_STAGE1_v8-plansplit.md,
// d1dbe526f; F-LS edition rows 6.19, 6.25; review items 21, 27; LFS Q4.A24; U-36 (i), L:4275):
// the decode graph's columns against its time buckets, the graph switch over the per-request
// samples, and no file written for the samples.

import (
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/x/ansi"
)

// S8-LA-37 (screens).
func TestS8LA37(t *testing.T) {
	// review item 21: every column is the mean of its time bucket; the end labels span the
	// buckets; the scale runs from the maximum to 0; the axis line holds the average and minimum
	t0 := time.Date(2026, 10, 10, 9, 0, 0, 0, time.UTC)
	to := t0.Add(30 * time.Minute)
	var ss []sample
	for i := 0; i < 1800; i += 5 { // every 5 s: 12, then 6, then 9 tok/s in three ten-minute thirds
		v := []float64{12, 6, 9}[i/600]
		ss = append(ss, sample{t0.Add(time.Duration(i) * time.Second), v})
	}
	r := rc{W: 120, H: 30, lang: LangEnglish}
	const w = 70
	g := r.timeGraph(ss, t0, to, w, gr, "09:00", "now")
	cols := w - 5 // "12.0┤" is five cells
	want := make([]float64, cols)
	for c := range want { // the bucket means, computed here apart from bins
		lo, hi := t0.Add(time.Duration(float64(c)/float64(cols)*float64(to.Sub(t0)))), t0.Add(time.Duration(float64(c+1)/float64(cols)*float64(to.Sub(t0))))
		sum, n := 0.0, 0
		for _, s := range ss {
			if !s.at.Before(lo) && (s.at.Before(hi) || c == cols-1) {
				sum, n = sum+s.v, n+1
			}
		}
		if n > 0 {
			want[c] = sum / float64(n) / 12
		}
	}
	exp := tall(want, 3, gr)
	for i := 0; i < 3; i++ {
		got := ansi.Strip(g[i])
		got = got[strings.Index(got, "┤")+len("┤"):]
		if i == 1 {
			got = ansi.Strip(g[i])[strings.Index(ansi.Strip(g[i]), "│")+len("│"):]
		}
		if got != ansi.Strip(exp[i]) {
			t.Errorf("graph row %d:\n got  %q\n want %q", i, got, ansi.Strip(exp[i]))
		}
	}
	if !strings.HasPrefix(ansi.Strip(g[0]), "12.0┤") || !strings.HasPrefix(ansi.Strip(g[2]), "   0┤") {
		t.Errorf("scale %q %q", ansi.Strip(g[0]), ansi.Strip(g[2]))
	}
	axis := ansi.Strip(g[3])
	avg := 0.0
	for _, s := range ss {
		avg += s.v
	}
	avg /= float64(len(ss))
	if !strings.HasPrefix(strings.TrimSpace(axis), "09:00") || !strings.HasSuffix(axis, "now") ||
		!strings.Contains(axis, "average "+f1(avg)+" · lowest 6.0 tok/s") {
		t.Errorf("axis %q", axis)
	}

	// LFS Q4.A24: the per-request prefill rate and TTFT show on the screen only (since UX-R19 beside
	// the decode graph, not behind a switch); the per-request store grows across polls without doubles
	dir := t.TempDir()
	wd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 40)
	hs.keys("enter")
	hs.beat()
	hs.want("H8")
	now := hs.clock
	q := func(task int, ago time.Duration, pre, ttft float64) ReqSample {
		return ReqSample{Task: task, At: now.Add(-ago), PrefillTokS: pre, TTFTs: ttft, DecodeTokS: 9}
	}
	st := *hs.m.status
	st.Reqs = []ReqSample{q(3, 20*time.Minute, 20, 4), q(4, 10*time.Minute, 30, 2)}
	hs.send(statusMsg{st, nil})
	st.Reqs = []ReqSample{q(4, 10*time.Minute, 30, 2), q(5, time.Minute, 40, 6)}
	hs.send(statusMsg{st, nil})
	if n := len(hs.m.reqSamples); n != 3 || hs.m.reqSamples[2].Task != 5 {
		t.Fatalf("per-request store %+v", hs.m.reqSamples)
	}
	before := len(f.Calls())
	p := hs.plain()
	mustHave(t, p, "DECODE · tok/s · last 60 min", "PREFILL · tok/s per request")
	// UX-R27: the rate and TTFT of the window in the PREFILL panel (was «TTFT median 4.0 s · max
	// 6.0 s» and «average 30.0 · lowest 20.0 tok/s»)
	mustHave(t, strings.Join(strings.Fields(p), " "), "prefill — 30.0 20.0 40.0 tok/s", "TTFT — 4.0 2.0 6.0 s")
	hs.keys("g") // no switch any more: the screen stays as it is
	if hs.plain() != p {
		t.Errorf("g changed the screen")
	}
	for _, c := range f.Calls()[before:] {
		if strings.HasPrefix(c, "Action") {
			t.Errorf("the graph switch called the backend: %s", c)
		}
	}
	// U-36 (i): the samples are held in memory only; nothing is written for them
	if es, _ := os.ReadDir(dir); len(es) != 0 {
		t.Errorf("files written: %v", es)
	}
	if math.IsNaN(avg) {
		t.Fatal("no samples")
	}
}
