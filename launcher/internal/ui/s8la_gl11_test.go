package ui

// GL-11 (version 8): the screen's part of S8-LA-25 (plan reviews/_plan_8_STAGE1_v8-plansplit.md
// at d1dbe526f; F-LS edition reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, conflict
// row 18-1 [A-SMK]): a failed check run is one line at the place of the start verdict (H7),
// «✕ 시작 후 응답 시험 실패 · … · 자세히 4 Log» (UX-R4: «작동 확인» renamed); the adapter keeps a pass off the screen (wire TestS8LA25).

import (
	"strings"
	"testing"
)

func TestS8LA25(t *testing.T) {
	for _, c := range []struct {
		lang       string
		line, word string
	}{
		{LangEnglish, "✕ check run failed · failed items 5:verify-consume · details on 4 Log", "check run failed"},
		{LangKorean, "✕ 시작 후 응답 시험 실패 · failed items 5:verify-consume · 자세히 4 Log", "시작 후 응답 시험 실패"},
	} {
		hs := newH(t, NewFake(), c.lang, 120, 30)
		hs.m.failure = &Failure{Stage: "fail_smoke", ExitCode: 6, Why: "failed items 5:verify-consume", SmokeFailed: true}
		hs.m.ph = phFailed
		hs.keys("2")
		p := hs.plain()
		if n := strings.Count(p, c.word); n != 1 {
			t.Errorf("%v: %q appears %d times:\n%s", c.lang, c.word, n, p)
		}
		lines := 0
		for _, ln := range strings.Split(p, "\n") {
			if strings.Contains(ln, c.line) {
				lines++
			}
		}
		if lines != 1 {
			t.Errorf("%v: the verdict line %q on %d lines:\n%s", c.lang, c.line, lines, p)
		}
	}
	// Without a failed check there is no such line.
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	hs.m.failure = &Failure{Stage: "fail_server_start", ExitCode: 5, Why: "the server exited"}
	hs.m.ph = phFailed
	hs.keys("2")
	if strings.Contains(hs.plain(), "check run") {
		t.Errorf("a start failure without a check shows it:\n%s", hs.plain())
	}
}
