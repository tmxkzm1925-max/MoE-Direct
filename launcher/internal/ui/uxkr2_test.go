package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

// UX-KR2 (version 8): the user's word choices of 18:43 and 18:48 KST
// (reviews/_brief_8_UX-KR2_v8-uxkr.md; reviews/_user_8_LAUNCHER-REVIEW_11.md): (1) the 4 Log
// header «이벤트 기록 (최신=아래)»; (2) the prefetch on-word «적용됨» / «applied», the pair of
// «미적용» / «not applied»; (3) one not-ported word «미지원» (switch on only); (4) the idle title
// of the 2 Server request box «대기 중 · 요청 없음» / «IDLE · no request» went with the box in
// UX-R27 (the statistics panels' titles do not switch; idle is «—» in their now column).
// Each page is drawn in both languages at 120x40 and 80x28: the new words are there, the old
// ones are not, and no line is wider than the window.
func TestUXKR2Words(t *testing.T) {
	type page struct {
		name     string
		notPorte bool
		h        func(lang string, w, h int) *harness
		want     map[string][]string
	}
	engineOn := func() *FakeBackend {
		f := NewFake()
		for i := range f.Statuses {
			f.Statuses[i].Prefetch = []string{"engine on", "K 8", "N 4", "candidate on", "K 8", "N 4", "catalog-validated"}
			f.Statuses[i].Phase = ReqNone
		}
		return f
	}
	pages := []page{
		{"4 Log", false, func(lang string, w, h int) *harness {
			hs := newH(t, uxkrFake(), lang, w, h)
			hs.keys(serving("4")...)
			return hs
		}, map[string][]string{LangEnglish: {"what happened, newest last"}, LangKorean: {"이벤트 기록 (최신=아래)"}}},
		{"2 Server, engine on, idle", false, func(lang string, w, h int) *harness {
			hs := newH(t, engineOn(), lang, w, h)
			hs.keys(serving()...)
			hs.beat()
			return hs
		}, map[string][]string{
			LangEnglish: {"applied · K 8 · N 4"},
			LangKorean:  {"적용됨 · K 8 · N 4"},
		}},
		{"5 Settings, switch on", true, func(lang string, w, h int) *harness {
			hs := newH(t, uxr4Fake(), lang, w, h)
			hs.keys("5")
			return hs
		}, map[string][]string{LangEnglish: {"not ported"}, LangKorean: {"미지원"}}},
		{"2 Server, waiting not measured, switch on", true, func(lang string, w, h int) *harness {
			f := uxr4Fake()
			for i := range f.Statuses {
				f.Statuses[i].Requests, f.Statuses[i].Errors, f.Statuses[i].Waiting = K(1), K(0), U
				f.Statuses[i].Slow = false
			}
			hs := newH(t, f, lang, w, h)
			hs.keys(serving()...)
			hs.beat()
			return hs
		}, map[string][]string{LangEnglish: {"waiting: not ported"}, LangKorean: {"대기: 미지원"}}},
	}
	old := map[string][]string{
		LangEnglish: {"engine on", "speedometer", " · idle"},
		LangKorean:  {"기록 · 아래가 최신", "엔진 켜짐", "아직 지원", "속도계", "대기 상태"},
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			for _, pg := range pages {
				withNotPorted(t, pg.notPorte)
				hs := pg.h(lang, sz[0], sz[1])
				p := hs.plain()
				for _, w := range pg.want[lang] {
					if !strings.Contains(p, w) {
						t.Errorf("%s %s %dx%d: %q is not drawn", pg.name, lang, sz[0], sz[1], w)
					}
				}
				for _, w := range old[lang] {
					if strings.Contains(p, w) {
						t.Errorf("%s %s %dx%d: the old %q is drawn", pg.name, lang, sz[0], sz[1], w)
					}
				}
				ls := strings.Split(p, "\n")
				if len(ls) > sz[1] {
					t.Fatalf("%s %s %dx%d: %d lines:\n%s", pg.name, lang, sz[0], sz[1], len(ls), p)
				}
				for i, l := range ls {
					if w := lipgloss.Width(l); w > sz[0] {
						t.Fatalf("%s %s %dx%d: line %d is %d wide:\n%s", pg.name, lang, sz[0], sz[1], i, w, p)
					}
				}
			}
		}
	}
	withNotPorted(t, false)
}
