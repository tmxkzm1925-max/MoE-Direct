package ui

import "testing"

// S8-PF-08 (the drawn part; GL-20, F-PF §7.2, OQ-5 (b)): the virtual notice is drawn in the
// screen language, the user's words in Korean.
func TestS8PF08(t *testing.T) {
	for lang, want := range map[string]string{
		LangEnglish: "default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured",
		LangKorean:  "기본 값=QD 8+카탈로그 프리페치뿐 · 그 밖=동작 미측정",
	} {
		f := NewFake()
		hs := newH(t, f, lang, 120, 30)
		hs.keys("tab", "down", "down", "right") // QD 16: off the default
		mustHave(t, hs.plain(), want)
	}
}
