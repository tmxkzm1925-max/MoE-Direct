package ui

// UX-R17 (version 8; the user's acceptance finding 27, reviews/_user_8_LAUNCHER-REVIEW_11.md item
// 27, screenshot 154623): on 2 Server the decode graph read as bars arriving from the right and
// piling up, because its window ran from the first sample to now and every column's span grew
// with the run. The user expects the last 60 minutes in 60 columns, one column a minute holding
// that minute's output, the whole window moving one column left as the minute turns, the 61st
// minute leaving, and the clock times under the graph moving with it. These tests feed 75
// minutes of samples on an injected clock and read the drawn rows and the axis line after each
// minute turn against columns computed here, apart from minuteCells.

import (
	"math"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/x/ansi"
)

// uxr17Run is the fed run: minute i (0..75) of t0 has a sample every 10 s, alternately 0.5
// under and over 2+i%11 (its mean), except the minutes i%13 == 6, which have none.
func uxr17Run(t0 time.Time) []sample {
	var ss []sample
	for i := 0; i <= 75; i++ {
		if i%13 == 6 {
			continue
		}
		for s := 0; s < 60; s += 10 {
			d := -0.5
			if s/10%2 == 1 {
				d = 0.5
			}
			ss = append(ss, sample{t0.Add(time.Duration(i)*time.Minute + time.Duration(s)*time.Second), float64(2+i%11) + d})
		}
	}
	return ss
}

// uxr17Want computes, for the samples fed by now and the window of the 60 minutes that end with
// minute end of t0: the three graph rows, the scale's cells, each window minute's first cell,
// the window minutes' samples (UX-R27: the axis line has no «average · lowest» text any more).
func uxr17Want(ss []sample, t0, now time.Time, end, w int) (rows []string, sw int, first []int, mins [][]float64) {
	mins = make([][]float64, 60) // window minute k = run minute end-59+k
	for _, s := range ss {
		if s.at.After(now) {
			continue
		}
		i := int(s.at.Sub(t0) / time.Minute)
		if k := i - (end - 59); k >= 0 && k < 60 {
			mins[k] = append(mins[k], s.v)
		}
	}
	mean := func(vs []float64) float64 {
		t := 0.0
		for _, v := range vs {
			t += v
		}
		return t / float64(len(vs))
	}
	top := 0.0
	for _, vs := range mins {
		if len(vs) > 0 {
			top = math.Max(top, mean(vs))
		}
	}
	sw = len(f1(top)) + 1
	cols := w - sw
	vals := make([]float64, cols)
	first = make([]int, 60)
	if cols >= 60 { // a minute is cols/60 cells wide, the window at the right edge
		bw := cols / 60
		for k := range mins {
			first[k] = cols - 60*bw + k*bw
			for j := 0; j < bw && len(mins[k]) > 0; j++ {
				vals[first[k]+j] = mean(mins[k]) / top
			}
		}
	} else { // several minutes to a cell: all their samples
		for c := 0; c < cols; c++ {
			var vs []float64
			for k := range mins {
				if k*cols/60 == c {
					vs = append(vs, mins[k]...)
				}
			}
			if len(vs) > 0 {
				vals[c] = mean(vs) / top
			}
		}
		for k := range first {
			first[k] = k * cols / 60
		}
	}
	for _, l := range tall(vals, 3, gr) {
		rows = append(rows, ansi.Strip(l))
	}
	return rows, sw, first, mins
}

// uxr17Rows strips the scale off the three drawn rows.
func uxr17Rows(g []string, sw int) [][]rune {
	var out [][]rune
	for _, l := range g[:3] {
		out = append(out, []rune(ansi.Strip(l))[sw:])
	}
	return out
}

var uxr17Clock = regexp.MustCompile(`\d\d:\d\d`)

func TestUXR17MinuteWindow(t *testing.T) {
	t0 := time.Date(2026, 10, 13, 15, 0, 0, 0, time.UTC)
	all := uxr17Run(t0)
	var now time.Time
	m := &Model{r: rc{W: 160, H: 40, lang: LangEnglish}, now: func() time.Time { return now }}
	for _, w := range []int{140, 70, 40} { // two cells a minute, one, and fewer cells than minutes
		moved, prevSW := 0, 0
		prev := map[string]int{}
		for turn := 1; turn <= 75; turn++ {
			now = t0.Add(time.Duration(turn)*time.Minute + 20*time.Second) // the current minute has 3 samples
			var fed []sample
			for _, s := range all {
				if !s.at.After(now) {
					fed = append(fed, s)
				}
			}
			g := m.liveGraph(fed, w, gr)
			want, sw, first, mins := uxr17Want(fed, t0, now, turn, w)
			got := uxr17Rows(g, sw)
			cols := w - sw
			// each minute's column at its place: the window ends with this minute
			for i := range want {
				if string(got[i]) != want[i] {
					t.Fatalf("w=%d minute %d (now %s) row %d:\n got  %q\n want %q", w, turn, now.Format("15:04:05"), i, string(got[i]), want[i])
				}
			}
			// shifted: the rows of the window one minute earlier are not what is drawn (where a
			// minute has its own cells; fewer cells than minutes can coincide)
			if old, _, _, _ := uxr17Want(fed, t0, now, turn-1, w); cols >= 60 && strings.Join(old, "\n") == string(got[0])+"\n"+string(got[1])+"\n"+string(got[2]) {
				t.Fatalf("w=%d minute %d: the window did not move on", w, turn)
			}
			// an empty minute is an empty column, a minute with samples is drawn; with fewer
			// cells than minutes the current minute keeps the last cell
			if cols >= 60 {
				for k, vs := range mins {
					for j := 0; j < cols/60; j++ {
						c := first[k] + j
						blank := got[0][c] == ' ' && got[1][c] == ' ' && got[2][c] == ' '
						if len(vs) == 0 && !blank {
							t.Fatalf("w=%d minute %d: the empty window minute %d is drawn: %q", w, turn, k, string(got[2]))
						}
						if len(vs) > 0 && got[2][c] == ' ' {
							t.Fatalf("w=%d minute %d: window minute %d with %d samples is not drawn: %q", w, turn, k, len(vs), string(got[2]))
						}
					}
				}
			} else if len(mins[59]) > 0 && got[2][cols-1] == ' ' {
				t.Fatalf("w=%d minute %d: the current minute is not in the last cell: %q", w, turn, string(got[2]))
			}
			// the axis: «now» at the end and no average or lowest (UX-R27: they are the DECODE
			// panel's); every clock time under its own minute's column, the oldest minute's
			// always, the others quarter hours
			axis := ansi.Strip(g[3])
			if n := len([]rune(axis)); n != w || !strings.HasSuffix(axis, "now") {
				t.Fatalf("w=%d minute %d: axis %q (want %d cells, ending «now»)", w, turn, axis, w)
			}
			if strings.Contains(axis, "average") || strings.Contains(axis, "lowest") {
				t.Fatalf("w=%d minute %d: axis %q still has the average or the lowest", w, turn, axis)
			}
			oldest := t0.Add(time.Duration(turn-59) * time.Minute)
			seen := map[string]int{}
			for _, loc := range uxr17Clock.FindAllStringIndex(axis, -1) {
				s := axis[loc[0]:loc[1]]
				at := len([]rune(axis[:loc[0]])) - sw
				tm, _ := time.Parse("15:04", s)
				k := ((tm.Hour()*60 + tm.Minute()) - (oldest.Hour()*60 + oldest.Minute()) + 24*60) % (24 * 60)
				if k >= 60 || (k != 0 && tm.Minute()%15 != 0) || at != first[k] {
					t.Fatalf("w=%d minute %d: clock time %s at cell %d is not under its own minute's column: %q", w, turn, s, at, axis)
				}
				seen[s] = at
			}
			if _, ok := seen[oldest.Format("15:04")]; !ok {
				t.Fatalf("w=%d minute %d: the oldest minute's time %s is not on the axis: %q", w, turn, oldest.Format("15:04"), axis)
			}
			if cols >= 60 && sw == prevSW { // (a new scale width moves the whole graph)
				for s, at := range seen {
					if p, ok := prev[s]; ok {
						if p-at != cols/60 {
							t.Fatalf("w=%d minute %d: %s moved from cell %d to %d, not one column left", w, turn, s, p, at)
						}
						moved++
					}
				}
			}
			prev, prevSW = seen, sw
		}
		if w >= 70 && moved == 0 {
			t.Fatalf("w=%d: no clock time was seen moving with the window", w)
		}
	}
}

// a request completed in the current minute is in the rightmost column at once, in the decode
// graph (an answer shorter than a beat: its whole-answer rate, UX-R11) and in the prefill view;
// one 61 minutes old is gone
func TestUXR17ShortRequest(t *testing.T) {
	now := time.Date(2026, 10, 13, 15, 42, 7, 0, time.UTC)
	for _, w := range []int{70, 140, 40} {
		m := &Model{r: rc{W: 160, H: 40, lang: LangEnglish}, now: func() time.Time { return now }}
		m.reqSamples = []ReqSample{
			{Task: 1, At: now.Add(-61 * time.Minute), PrefillTokS: 90, DecodeTokS: 3},
			{Task: 2, At: now.Add(-5 * time.Second), PrefillTokS: 50, DecodeTokS: 7},
		}
		for _, v := range []struct {
			name string
			g    []string
		}{{"decode", m.liveGraph(m.decodeSeries(), w, gr)}, {"prefill", m.reqGraph(w)}} {
			zero := ansi.Strip(v.g[2])
			body := strings.TrimLeft(zero[strings.Index(zero, "┤")+len("┤"):], " ")
			cells := 1
			if c := w - len(strings.TrimSpace(zero[:strings.Index(zero, "┤")])) - 1; c >= 60 {
				cells = c / 60
			}
			if !strings.HasSuffix(zero, "█") || strings.Count(body, "█") != cells || strings.Trim(body, "█") != "" {
				t.Fatalf("w=%d %s: the request of this minute is not alone in the rightmost column: %q", w, v.name, zero)
			}
			if ax := ansi.Strip(v.g[3]); strings.Contains(ax, "lowest 3.0") || strings.Contains(ax, "lowest 90.0") {
				t.Fatalf("w=%d %s: the request 61 minutes old still counts: %q", w, v.name, ax)
			}
		}
	}
}
