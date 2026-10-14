package ui

import (
	"testing"

	"moedirect/launcher/internal/version"
)

// TestS8LA30 (S8-LA-30, the about text): the help page shows the launcher's name and the
// release version v0.3.2 in both screen languages.
func TestS8LA30(t *testing.T) {
	const about = "MoE-Direct-launcher v0.3.2"
	if version.About() != about {
		t.Fatalf("version.About() = %q, want %q", version.About(), about)
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, NewFake(), lang, 120, 30)
		hs.keys("?")
		hs.want("H13")
		mustHave(t, hs.plain(), about)
	}
}
