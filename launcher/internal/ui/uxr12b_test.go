package ui

import (
	"strings"
	"testing"
)

// UX-R12b (version 8; the sub-lead's addenda from UX-R5's landing 4 items 5 and 11 and UX-R12's
// landing 4 item 4): the check run's stop reason in the screen's language, the user-confirmed
// tail «이 설정의 성능은 미확인» of a model outside the catalog, and «n k context» by its pattern.

func TestUXR12bWords(t *testing.T) {
	// (5) core's stop reason during the check run: «시작 후 응답 시험» on the Korean screen
	const stopped = "the server was stopped during the check run"
	for _, c := range []struct{ lang, want string }{{LangEnglish, stopped}, {LangKorean, "시작 후 응답 시험 중 서버가 멈춤"}} {
		if got := (rc{lang: c.lang}).reason(stopped); got != c.want {
			t.Fatalf("%s stop reason: %q", c.lang, got)
		}
		f := NewFake()
		f.StartEvs = []StartEvent{{Model: "qwen35-122b", Steps: []Step{{Name: "check run", State: StepFailed}},
			Failure: &Failure{Stage: "start", What: "did not start", Why: stopped}}}
		hs := newH(t, f, c.lang, 160, 45)
		hs.keys("enter")
		hs.want("H7")
		p := strings.Join(strings.Fields(hs.plain()), " ")
		if !strings.Contains(p, strings.Join(strings.Fields(c.want), " ")) || strings.Contains(p, "작동 확인") {
			t.Fatalf("%s H7: no «%s»:\n%s", c.lang, c.want, hs.plain())
		}
	}

	// (11) the confirmed tail on the 이 모델 box of a model outside the catalog
	hs := newH(t, NewFake(), LangKorean, 160, 45)
	hs.keys("down", "down", "down", "down")
	tpl := hs.m.r.Tf("outside the catalog · %s template settings · performance untested", "gpt-oss")
	p := hs.plain() // the box wraps the line: its two parts
	if tpl != "카탈로그 밖 · gpt-oss 템플릿 설정 · 이 설정의 성능은 미확인" || !strings.Contains(p, "이 설정의") ||
		!strings.Contains(p, "성능은 미확인") || strings.Contains(p, "이 설정 성능 미확인") {
		t.Fatalf("Korean template model: the tail is not «이 설정의 성능은 미확인»:\n%s", hs.plain())
	}

	// (4) «n k context» for any context, with «n GB cache»
	for _, n := range []string{"32k", "8k", "12k", "40000"} {
		for _, lang := range []string{LangEnglish, LangKorean} {
			f := NewFake()
			for i := range f.Statuses {
				f.Statuses[i].SettingsWords = []string{n + " context", "20 GB cache", "virtual"}
			}
			hs := newH(t, f, lang, 160, 45)
			hs.keys(serving()...)
			hs.beat()
			p := hs.plain()
			want, lack := n+" context · 20 GB cache", "컨텍스트 "+n
			if lang == LangKorean {
				want, lack = "컨텍스트 "+n+" · 캐시 20 GB", n+" context"
			}
			if !strings.Contains(p, want) || strings.Contains(p, lack) {
				t.Fatalf("%s context %s: want «%s», not «%s»:\n%s", lang, n, want, lack, p)
			}
		}
	}
}
