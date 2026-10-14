package ui

import (
	"regexp"
	"strings"
	"testing"
	"time"
)

// ramCell is the 1 Models RAM tile of a plain screen: its value, bar and note lines
// (the three lines under the tile's top edge, cut at its column).
func ramCell(t *testing.T, plain string) []string {
	t.Helper()
	lines := strings.Split(plain, "\n")
	for i, l := range lines {
		rs := []rune(l)
		at := strings.Index(l, "╭─ RAM")
		if at < 0 {
			continue
		}
		col := len([]rune(l[:at]))
		end := col + 1
		for end < len(rs) && rs[end] != '╮' {
			end++
		}
		var out []string
		for _, x := range lines[i+1 : min(i+4, len(lines))] {
			xr := []rune(x)
			out = append(out, strings.TrimSpace(string(xr[min(col+1, len(xr)):min(end, len(xr))])))
		}
		return out
	}
	t.Fatalf("no RAM tile:\n%s", plain)
	return nil
}

// setIdleReady changes the OS reader's free-RAM value the fake reports while no server is up.
func setIdleReady(f *FakeBackend, ready, apps float64) {
	f.mu.Lock()
	f.MemIdle.ReadyNowGB, f.MemIdle.AppsGB = K(ready), K(apps)
	f.mu.Unlock()
}

// readAgain is one beat later: the timers that are due fire and the screen draws again.
func (h *harness) readAgain() {
	h.advance(2 * time.Second)
	h.fireDue()
	h.beat()
}

var leftNumber = regexp.MustCompile(`left( after start)? -?[0-9]`)

// TestS8LA10 is S8-LA-10 (GL-05; review item 13; design H1:6-8) with the fake backend: H1
// shows the need P of the check, the free value F of the OS reader and left F-P in the
// design's number form; a failed projection shows the need not measured and no left value;
// a new OS reading at a draw changes only free and left, with no new check; the bar spans
// the whole PC RAM with in use, this model and left, each with its name. The need's labels
// of S8-LA-13 are drawn where they fit and always on 3 Memory.
func TestS8LA10(t *testing.T) {
	// (1) P = 19.0 (the check's need), F = 24.6 (the OS reader), left 5.6; total 32 → in use 7.4
	f := NewFake()
	f.Pre.NeedLabel = "measured"
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.want("H1")
	c := ramCell(t, hs.plain())
	if c[0] != "19.0 GB this model" || c[2] != "in use 7.4 · left 5.6" {
		t.Fatalf("H1 RAM tile at 120: %q", c)
	}
	if !strings.Contains(hs.plain(), "RAM · 32 GB") {
		t.Fatalf("the tile's title lacks the PC's RAM:\n%s", hs.plain())
	}
	// the bar spans the whole RAM: in use ▓ (7.4 of 32), this model █ (19.0 of 32), left ░
	used, model := strings.Count(c[1], "▓"), strings.Count(c[1], "█")
	cells := len([]rune(c[1]))
	if used == 0 || model == 0 || !strings.Contains(c[1], "░") || used >= model ||
		abs(float64(used)/float64(cells)-7.4/32) > 1.0/float64(cells) ||
		abs(float64(model)/float64(cells)-19.0/32) > 1.0/float64(cells) {
		t.Fatalf("bar %q: in use %d, this model %d of %d", c[1], used, model, cells)
	}
	// the label is drawn whole or not at all: at 200 it fits beside the name
	wide := newH(t, f, LangEnglish, 200, 30)
	if c := ramCell(t, wide.plain()); c[0] != "19.0 GB this model · measured" || c[2] != "in use 7.4 · left after start 5.6" {
		t.Fatalf("H1 RAM tile at 200: %q", c)
	}

	// (2) a new OS reading: free 20.6 → in use 11.4, left 1.6; the need stays; no new check
	preN, memN := hs.calls("Preflight"), hs.calls("Memory")
	setIdleReady(f, 20.6, 11.4)
	hs.readAgain()
	c2 := ramCell(t, hs.plain())
	if c2[0] != "19.0 GB this model" || c2[2] != "in use 11.4 · left 1.6" {
		t.Fatalf("after a new OS reading: %q", c2)
	}
	if hs.calls("Preflight") != preN || hs.calls("Memory") <= memN {
		t.Fatalf("a draw re-checked or did not read: %v", f.Calls())
	}
	if u := strings.Count(c2[1], "▓"); u <= used || strings.Count(c2[1], "█") != model {
		t.Fatalf("bar after the reading %q (in use was %d, this model %d)", c2[1], used, model)
	}

	// (3) a failed projection: the need is not measured and no left value is computed,
	// also after a new reading
	g := NewFake()
	g.Pre.NeedGB, g.Pre.LeftGB, g.Pre.Verdict, g.Pre.NeedLabel = U, U, VerdictUnmeasured, ""
	hu := newH(t, g, LangEnglish, 120, 30)
	cu := ramCell(t, hu.plain())
	if !strings.HasPrefix(cu[0], "not measured yet") || leftNumber.MatchString(cu[2]) || strings.ContainsRune(cu[1], '█') {
		t.Fatalf("failed projection, H1 RAM tile: %q", cu)
	}
	if !strings.Contains(hu.plain(), "need not measured yet") {
		t.Fatalf("failed projection, verdict:\n%s", hu.plain())
	}
	setIdleReady(g, 20.6, 11.4)
	hu.readAgain()
	if cu := ramCell(t, hu.plain()); leftNumber.MatchString(cu[2]) || !strings.HasPrefix(cu[2], "in use 11.4") {
		t.Fatalf("failed projection after a reading: %q", cu)
	}

	// (4) the labels on 3 Memory, whole, in both languages; a projection of a measured
	// setting carries none
	for _, k := range []struct{ label, lang, want string }{
		{"measured", LangEnglish, "measured"},
		{"predicted before run", LangEnglish, "predicted before run"}, // UX-R10 (finding 16 (e))
		{"measured", LangKorean, "측정값"},
		{"predicted before run", LangKorean, "실측 전 예측값"},
	} {
		f := NewFake()
		f.Pre.NeedLabel = k.label
		h := newH(t, f, k.lang, 120, 30)
		h.keys("3")
		if !strings.Contains(h.plain(), k.want) {
			t.Fatalf("3 Memory (%s) lacks the label %q:\n%s", k.lang, k.want, h.plain())
		}
	}
	n := NewFake()
	hn := newH(t, n, LangEnglish, 200, 30)
	pn := hn.plain()
	hn.keys("3")
	pn += hn.plain()
	if strings.Contains(pn, "measured ") && strings.Contains(pn, "· measured") || strings.Contains(pn, "predicted before run") {
		t.Fatalf("a need without a label drew one:\n%s", pn)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
