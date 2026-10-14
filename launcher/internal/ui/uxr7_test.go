package ui

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// UX-R7 (version 8; the user's acceptance finding 13, reviews/_user_8_LAUNCHER-REVIEW_3.md): the
// bare unit «G» («RAM · 31 G», «9.5 G», «안전선 4 G») reads as gravity; every memory figure the
// screens show carries «GB» in both languages. The numbers are unchanged: they stay the
// launcher's binary GiB values; «GiB» stays only where exactness matters (5 Settings help, the
// log files).

// bareG is a figure, an input box or a big-digit hero followed by the unit «G» alone: «12 G»,
// «12G», «] G», «█▀▀  G». No exception is needed on these screens: no other word ends in a lone
// «G» after a digit, a bracket or a hero glyph (model names read «Qwen3.5-122B», «gpt-oss-120b»).
var bareG = regexp.MustCompile(`(\d|\]) ?G\b|[▀▄█] +G\b`)

// heroGB is the big-digit hero with its unit «GB» (3 Memory, AVAILABLE NOW).
var heroGB = regexp.MustCompile(`[▀▄█] +GB\b`)

// uxr7Memory are the screens that show memory figures, each with «GB» figures it must carry in
// English at 160 columns (the fake's values).
var uxr7Memory = map[string][]string{
	"H1":  {"19.0 GB this model", "RAM · 32 GB", "RAM budget 12 GB"},
	"H2":  {"18.2 GB this server", "4.7 GB left"},
	"H8":  {"4.7 GB left", "2.6 GB left", "12 GB cache"},
	"H10": {"4.7 GB", "32 GB", "16 GB", "█ this server 12.9 GB + ▓ other programs 0.5 GB = in use 13.4 GB"}, // UX-R22: the VRAM key
	"H14": {"19.0 GB", "5.6 GB left"},
}

func TestUXR7(t *testing.T) {
	ids := append(append([]string{}, designIDs...), "S0")
	for _, id := range ids {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range [][2]int{{160, 45}, {120, 30}} {
				hs := reach(t, id, lang, sz[0], sz[1])
				p := hs.plain()
				if os.Getenv("UXR7_SHOW") != "" {
					t.Logf("%s %s %dx%d\n%s", id, lang, sz[0], sz[1], p)
				}
				if m := bareG.FindString(p); m != "" {
					t.Fatalf("%s %s %dx%d: a memory figure with a bare «G» (%q):\n%s", id, lang, sz[0], sz[1], m, p)
				}
				if want, ok := uxr7Memory[id]; ok && lang == LangEnglish && sz[0] == 160 {
					mustHave(t, p, want...)
				}
				if id == "H14" && !heroGB.MatchString(p) {
					t.Fatalf("H14 %s %dx%d: the AVAILABLE NOW hero lacks «GB»:\n%s", lang, sz[0], sz[1], p)
				}
			}
		}
	}

	// 3 Memory with the check's lines all shown (UX-R3's case: the room line, the safe line, the
	// commit near its limit, the shortfalls), both languages
	for _, short := range []bool{false, true} {
		for _, lang := range []string{LangEnglish, LangKorean} {
			hs := newH(t, uxr3Fake(short), lang, 160, 40)
			hs.keys("3")
			p := hs.plain()
			if m := bareG.FindString(p); m != "" || !strings.Contains(p, " GB") {
				t.Fatalf("3 Memory (short %v, %s): a bare «G» (%q) or no «GB»:\n%s", short, lang, m, p)
			}
		}
	}

	// the typed budget box (1 Models settings): the unit and the range in «GB», Korean and English
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, NewFake(), lang, 120, 30)
		hs.keys("tab", " ")
		p := hs.plain()
		mustHave(t, p, "[ 12_ ] GB", "8 GB ~ 20 GB")
		if m := bareG.FindString(p); m != "" {
			t.Fatalf("%s typed box: a bare «G» (%q):\n%s", lang, m, p)
		}
	}
	// the Korean settings word and the 3 Memory check figures
	k := reach(t, "H8", LangKorean, 160, 45)
	mustHave(t, k.plain(), "캐시 12 GB")
	m := reach(t, "H14", LangKorean, 160, 45)
	if !regexp.MustCompile(`\d GB`).MatchString(m.plain()) || strings.Contains(m.plain(), " G ") {
		t.Fatalf("3 Memory (Korean) without «GB» figures:\n%s", m.plain())
	}
}
