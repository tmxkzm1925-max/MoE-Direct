// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"strings"
	"testing"
)

// TestUXR1EnglishStart: the user's acceptance finding 1 (UX-R1). With a Korean display
// language the launcher still starts in English in auto mode; the language row of 5 Settings
// switches the run to Korean and back, and its value no longer claims that auto follows the
// display language; -lang ko still starts in Korean.
func TestUXR1EnglishStart(t *testing.T) {
	old := displayLanguage
	displayLanguage = func() string { return LangKorean }
	defer func() { displayLanguage = old }()

	// (1) auto mode (no -lang): the first frame is English
	hs := newH(t, NewFake(), "", 120, 30)
	if hs.m.r.lang != LangEnglish || hs.m.auto != LangEnglish {
		t.Fatalf("auto start language %q (auto %q), want English", hs.m.r.lang, hs.m.auto)
	}
	if p := hs.plain(); hangul.MatchString(p) {
		t.Fatalf("first frame shows Korean under a Korean display language:\n%s", p)
	}

	// (2) 5 Settings, the language row: auto names English, no display-language claim
	hs.keys("5", "tab", "down", "down")
	p := hs.plain()
	if !strings.Contains(p, "auto · English") {
		t.Fatalf("language row lacks «auto · English»:\n%s", p)
	}
	if strings.Contains(p, "display language") || strings.Contains(p, "Korean, else English") {
		t.Fatalf("5 Settings still says auto follows the display language:\n%s", p)
	}

	// (3) the toggle: → English → Korean; every place follows at once
	hs.keys("right", "right")
	if hs.m.r.lang != LangKorean || hs.m.langAt != 2 {
		t.Fatalf("language row toggle: %q at %d, want Korean", hs.m.r.lang, hs.m.langAt)
	}
	if p := hs.plain(); !strings.Contains(p, "한국어") || !strings.Contains(p, "언어 = 시작은 영어") {
		t.Fatalf("5 Settings after the toggle:\n%s", p)
	}
	for _, place := range []string{"1", "2", "3", "4", "5"} {
		hs.keys(place)
		if p := hs.plain(); !hangul.MatchString(strings.ReplaceAll(p, "한국어", "")) {
			t.Fatalf("place %s after the toggle not in Korean:\n%s", place, p)
		}
	}
	hs.keys("down", "down", "right") // back to auto: English again
	if hs.m.r.lang != LangEnglish || hs.m.langAt != 0 {
		t.Fatalf("language row back to auto: %q at %d", hs.m.r.lang, hs.m.langAt)
	}
	if p := hs.plain(); hangul.MatchString(p) {
		t.Fatalf("auto after the toggle shows Korean:\n%s", p)
	}

	// (4) -lang ko still starts in Korean, and its auto value names Korean
	hk := newH(t, NewFake(), LangKorean, 120, 30)
	if hk.m.r.lang != LangKorean {
		t.Fatalf("-lang ko start language %q", hk.m.r.lang)
	}
	hk.keys("5", "tab", "down", "down")
	if p := hk.plain(); !strings.Contains(p, "자동 · 한국어") {
		t.Fatalf("-lang ko language row lacks «자동 · 한국어»:\n%s", p)
	}
}
