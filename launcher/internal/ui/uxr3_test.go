package ui

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// UX-R3 (user acceptance, findings 4 to 7; screenshot 053506): on 3 Memory a «left» below zero
// reads as a shortfall against its named baseline, the left tile carries its terms and a bar,
// the expert pool reads «전문가 메모리 예산» and the check box is the model's start checklist.
// The numbers and the verdicts are the memory gate's, unchanged.

// uxr3Fake is the check of the screenshot: 17.2 G available now, the room line 4 G; the commit
// limit 38.4 G, in use 20.3 G, this start 20.9 G of which the expert pool 12.0 G, the safe line
// 4 G. short=false: the need 12.0 G leaves 5.2 G, and 4.9 G of commit is left. short=true: the
// need 19.8 G is 2.6 G more than available (1.1 G more if written to disk first), and the
// commit is 2.8 G over its limit after the start, near-limit line included.
func uxr3Fake(short bool) *FakeBackend {
	f := NewFake()
	p := &f.Pre
	p.AvailGB, p.NeedGB, p.LeftGB, p.RoomLineGB = K(17.2), K(12.0), K(5.2), K(4)
	p.CommitLeftAfterGB, p.CommitLimitGB, p.CommitNowGB, p.CommitSafeGB = K(4.9), K(38.4), K(20.3), K(4)
	p.PoolGB, p.PredictedGB = K(12.0), K(20.9)
	p.Verdict = VerdictRoom
	if short {
		p.NeedGB, p.LeftGB, p.LeftIfWrittenGB = K(19.8), K(-2.6), K(-1.1)
		p.CommitLeftAfterGB = K(-2.8)
		p.Near = &NearLine{RoomGB: -2.8, SafeGB: 4, LimitGB: 38.4, NowGB: 20.3, PredictedGB: 20.9, PoolGB: 12.0}
		p.Verdict = VerdictCannot
		p.RAMTotalGB, p.AppsGB, p.ModifiedGB, p.StandbyGB, p.FreeGB = K(64), K(45.3), K(1.5), K(12.0), K(5.2)
	}
	return f
}

// lone is a number with a minus sign of its own (not a hyphen inside a name like «-V4»).
var lone = regexp.MustCompile(`(^|[\s(·:])[-−]\s?\d`)

func uxr3Page(t *testing.T, f *FakeBackend, lang string, w, h int) (string, string) {
	t.Helper()
	hs := newH(t, f, lang, w, h)
	hs.keys("3")
	p := hs.plain()
	if os.Getenv("UXR3_SHOW") != "" {
		t.Logf("\n%s", p)
	}
	row, _ := hs.m.selRow()
	return p, row.Name
}

func TestUXR3(t *testing.T) {
	for _, w := range []int{110, 160} {
		// a «left» below zero, Korean: the shortfall with its baseline everywhere
		p, name := uxr3Page(t, uxr3Fake(true), LangKorean, w, 40)
		fp := flat(p)
		mustHave(t, p, "커밋 여유", "한도 기준 2.8 GB 초과", "안전선 4 GB", // finding 4: the tile names its baseline
			"시작 뒤 남음", "2.6 GB 부족", // finding 5: the terms (below, with «GB», UX-R16b)
			"시작 뒤 2.8 GB 부족 · 한도 기준 · 따로 판정",         // finding 6: the commit row
			"시작 뒤 2.6 GB 부족", "디스크에 먼저 쓰면 1.1 GB 부족", // the memory panel's left figures
			"전문가 메모리 예산", name+" 점검표", strings.ToUpper(name)+" 점검표") // findings 6, 7
		mustHave(t, fp, flat("여유 17.2 GB − 필요 19.8 GB")) // UX-R16b: «GB» on every amount, closer set where narrow
		mustHave(t, fp, flat("커밋이 한도에 가까움: 시작 뒤 2.8 GB 부족 · 한도 기준 · 안전선 4.0 GB · 한도 38.4 GB · 지금 사용 20.3 GB · 이번 시작 20.9 GB, 그중 전문가 메모리 예산 12.0 GB · 아무것도 바꾸지 않음"))
		mustLack(t, p, "익스퍼트 풀", "지금 시작?", "가능 "+name, "남는 -", "남음 -", "GB 뒤")
		if m := lone.FindString(p); m != "" {
			t.Fatalf("w=%d: a negative amount is shown (%q):\n%s", w, m, p)
		}
		// the same in English
		pe, _ := uxr3Page(t, uxr3Fake(true), LangEnglish, w, 40)
		mustHave(t, pe, "COMMIT LEFT", "2.8 GB over the limit", "safe line 4 GB",
			"LEFT AFTER", "2.6 GB short",
			"2.8 GB short after the start · against the limit · judged on its own",
			"2.6 GB short after the start · 1.1 GB short if written to disk first",
			"expert pool", name+" start checklist", strings.ToUpper(name+" start checklist"))
		mustHave(t, flat(pe), flat("free 17.2 GB − need 19.8 GB")) // UX-R16b
		mustHave(t, flat(pe), flat("commit near its limit: 2.8 GB short after the start, against the limit · safe line 4.0 GB · limit 38.4 GB · in use now 20.3 GB · this start 20.9 GB, of it the expert pool 12.0 GB · nothing is changed"))
		mustLack(t, pe, "start now?", "GB after")
		if m := lone.FindString(pe); m != "" {
			t.Fatalf("w=%d: a negative amount is shown (%q):\n%s", w, m, pe)
		}

		// a positive «left»: the amounts left, the terms, the bar, no shortfall words
		pk, _ := uxr3Page(t, uxr3Fake(false), LangKorean, w, 40)
		mustHave(t, flat(pk), flat("여유 17.2 GB − 필요 12.0 GB")) // UX-R16b
		mustHave(t, pk, "5.2 GB 남음", "4.9 GB 뒤", "안전선 4 GB", "4.9 GB 시작 뒤 남음", name+" 점검표")
		mustLack(t, pk, "부족", "초과", "익스퍼트 풀")
		pp, _ := uxr3Page(t, uxr3Fake(false), LangEnglish, w, 40)
		mustHave(t, flat(pp), flat("free 17.2 GB − need 12.0 GB")) // UX-R16b
		mustHave(t, pp, "5.2 GB left", "4.9 GB after", "4.9 GB left after the start")
		mustLack(t, pp, "short", "over the limit")
		for _, x := range []string{pk, pp} {
			var tile string
			for _, l := range strings.Split(x, "\n") {
				if strings.Contains(l, "▓") && strings.Contains(l, "█") && strings.Count(l, "│") >= 6 {
					tile = l
				}
			}
			if tile == "" {
				t.Fatalf("w=%d: no need/left bar in the tile row:\n%s", w, x)
			}
		}
		if w == 160 { // the room line joins the amount where the tile is wide enough
			mustHave(t, pk, "5.2 GB 남음 · 여유선 4 GB")
			mustHave(t, pp, "5.2 GB left · room line 4 GB")
		}
	}

	// the serving page: commit used above the limit is shown as the excess, its baseline named
	sf := NewFake()
	sf.Mem.CommitUsedGB, sf.Mem.CommitLimitGB, sf.Mem.CommitSafeGB = K(40.0), K(38.4), K(4)
	hs := newH(t, sf, LangKorean, 160, 40)
	hs.keys(serving("3")...)
	ps := hs.plain()
	mustHave(t, ps, "커밋 여유", "한도 기준 초과 · 안전 ≥ 4")
	if m := lone.FindString(ps); m != "" {
		t.Fatalf("serving: a negative amount is shown (%q):\n%s", m, ps)
	}
}
