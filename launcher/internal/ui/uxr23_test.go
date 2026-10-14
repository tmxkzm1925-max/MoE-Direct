package ui

// UX-R23 (version 8; the user's acceptance item 36, reviews/_user_8_LAUNCHER-REVIEW_12.md, section
// «추가 메시지 19:05 KST»): after a stop the 2 Server run summary showed only the decode graph
// while the serving screen shows decode and prefill (UX-R19). Now the summary shows the prefill
// graph (each finished request's prompt rate over the whole run) under the decode graph, in the
// serving screen's arrangement: both where the height allows; the prefill row leaves first when
// short. This test renders the summary through the fake backend in English and Korean at 120x40
// (both graphs) and 80x28 (the prefill row has left; the page keeps its height).

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// uxr23Summary serves, reports four finished requests eight minutes apart (with their prompt
// rates when reqs is set), stops with ctrl+f and returns the harness on the run summary.
func uxr23Summary(t *testing.T, lang string, w, h int, reqs bool) *harness {
	t.Helper()
	f := NewFake()
	hs := newH(t, f, lang, w, h)
	start := hs.clock
	for _, evs := range f.StopEvs {
		res := *evs[len(evs)-1].Result
		res.Run.Started, res.Run.Ended = start, start.Add(40*time.Minute)
		evs[len(evs)-1].Result = &res
	}
	hs.keys("enter")
	hs.beat()
	var rs []ReqSample
	for i := 1; i <= 4; i++ {
		hs.advance(8 * time.Minute)
		st := *hs.m.status
		if reqs {
			rs = append(rs, ReqSample{Task: i, At: hs.clock, PrefillTokS: float64(10 + 10*i), TTFTs: 2, DecodeTokS: 6})
			st.Reqs = append([]ReqSample{}, rs...)
		}
		hs.send(statusMsg{st, nil})
	}
	hs.keys("ctrl+f")
	hs.want("H12")
	return hs
}

// uxr23Line is the index of the first line holding the text, or -1.
func uxr23Line(ls []string, text string) int {
	for i, l := range ls {
		if strings.Contains(l, text) {
			return i
		}
	}
	return -1
}

func TestUXR23SummaryGraphs(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		decT, preT := tr(lang, "DECODE · tok/s · whole run"), tr(lang, "PREFILL · tok/s · whole run")
		noPre := tr(lang, "no prefill samples in this run")
		axis := fmt.Sprintf(tr(lang, "average %s · lowest %s tok/s"), "35.0", "20.0")
		for _, c := range []struct {
			w, h      int
			reqs, pre bool // requests with prompt rates; the prefill graph is expected
		}{
			{120, 40, true, true}, {80, 28, true, true}, // both graphs
			{120, 40, false, true}, // no prompt rate in the run: the graph says so
			{120, 26, true, false}, // short: the prefill row has left first, the decode graph stays
		} {
			p := uxr23Summary(t, lang, c.w, c.h, c.reqs).plain()
			ls := strings.Split(p, "\n")
			fail := func(what string) {
				t.Helper()
				t.Fatalf("%s %dx%d reqs=%v: %s:\n%s", lang, c.w, c.h, c.reqs, what, p)
			}
			if len(ls) != c.h {
				fail(fmt.Sprintf("%d lines", len(ls)))
			}
			d, q := uxr23Line(ls, "─ "+decT+" ─"), uxr23Line(ls, "─ "+preT+" ─")
			if d < 0 {
				fail("no decode graph")
			}
			if !c.pre {
				if q >= 0 {
					fail("the prefill graph stayed on a page too short for it")
				}
				continue
			}
			if q < d+6 {
				fail("no prefill graph under the decode graph")
			}
			if c.reqs {
				if uxr23Line(ls[q:q+6], axis) < 0 || !strings.Contains(ls[q+1], "█") {
					fail("the prefill graph does not plot the requests' prompt rates")
				}
			} else if uxr23Line(ls[q:q+6], noPre) < 0 {
				fail("no empty-graph note")
			}
		}
	}
}
