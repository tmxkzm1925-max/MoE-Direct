package ui

// GL-12 (version 8): the screen's part of S8-LA-26 (plan reviews/_plan_8_STAGE1_v8-plansplit.md
// at d1dbe526f; F-LS edition 4.7; CF 17): 5 Settings offers no network option, in either
// language; the design's network choice is removed (network deferred, README:29).

import (
	"strings"
	"testing"
)

func TestS8LA26(t *testing.T) {
	for _, c := range []struct {
		lang  string
		words []string
	}{
		{LangEnglish, []string{"network", "LAN", "bind", "0.0.0.0"}},
		{LangKorean, []string{"네트워크", "LAN", "network", "0.0.0.0"}},
	} {
		hs := newH(t, NewFake(), c.lang, 120, 40)
		hs.keys("5")
		p := hs.plain()
		if !strings.Contains(p, "127.0.0.1") {
			t.Errorf("%s: no address on 5 Settings:\n%s", c.lang, p)
		}
		for _, w := range c.words {
			if strings.Contains(p, w) {
				t.Errorf("%s: 5 Settings shows %q:\n%s", c.lang, w, p)
			}
		}
	}
}
