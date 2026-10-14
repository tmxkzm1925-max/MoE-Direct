package ui

// UX-R26 (version 8; the user's acceptance item 35 (b) with the answer of 19:13 KST and item 37,
// reviews/_user_8_LAUNCHER-REVIEW_12.md, screenshot 190333): the top row's third tile on 2 Server
// repeated the prefill · TTFT box beside the prefill graph. Now it is the processing tile of this
// request: the input tokens with the part reused from the KV cache, the generated tokens and the
// batch size the server was started with (-b, on a row of its own so a second batch figure has
// room); a figure not read shows «—», never 0. The prefill box and the request box are unchanged.
// This test renders 2 Server through the fake backend, answering and idle, in English and Korean,
// at 120x40 and 80x28, and once at 120x40 before any request.

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestUXR26ProcessingTile(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		L := func(k, v string) string { return tr(lang, k) + " " + v }
		for _, c := range []struct {
			w, h  int
			state string // "busy", "idle", "none" (no request read yet)
			rows  []string
		}{
			{120, 40, "busy", []string{L("in", "9.2k") + " (" + L("reused", "3.1k") + ")", L("out", "214"), L("batch", "2048")}},
			{120, 40, "idle", []string{L("in", "9.2k") + " (" + L("reused", "3.1k") + ")", L("out", "214"), L("batch", "2048")}},
			{120, 40, "none", []string{L("in", "—") + " (" + L("reused", "—") + ")", L("out", "—"), L("batch", "—")}},
			// 80x28: the tile is 14 cells inside; reused gets its own row and the batch, a setting
			// of the start, gives way
			{80, 28, "busy", []string{L("in", "9.2k"), L("reused", "3.1k"), L("out", "214")}},
			{80, 28, "idle", []string{L("in", "9.2k"), L("reused", "3.1k"), L("out", "214")}},
		} {
			f := NewFake()
			switch c.state {
			case "idle":
				idleServer(f)
			case "none":
				idleServer(f)
				for i := range f.Statuses {
					f.Statuses[i].ReusedTokens, f.Statuses[i].NewTokens, f.Statuses[i].DecodedTokens, f.Statuses[i].Batch = U, U, U, U
				}
			}
			hs := newH(t, f, lang, c.w, c.h)
			hs.keys("enter")
			hs.beat()
			hs.want("H8")
			p := hs.plain()
			ls := strings.Split(p, "\n")
			fail := func(what string) {
				t.Helper()
				t.Fatalf("%s %s %dx%d: %s:\n%s", lang, c.state, c.w, c.h, what, p)
			}
			if len(ls) != c.h {
				fail("the page is not whole")
			}
			for _, l := range ls {
				if lipgloss.Width(l) > c.w {
					fail("a line wider than the window")
				}
			}
			tile := uxr19Tile(p, tr(lang, "PROCESSING · this request"), tr(lang, "PROCESSING"))
			if tile == nil {
				fail("no processing tile in the tile row")
			}
			if strings.Join(tile, " | ") != strings.Join(c.rows, " | ") {
				fail("the processing tile reads «" + strings.Join(tile, " | ") + "», want «" + strings.Join(c.rows, " | ") + "»")
			}
			if strings.Contains(strings.Join(tile, " "), " 0") {
				fail("a figure not read shows 0")
			}
			if uxr19Tile(p, "PREFILL · TTFT", tr(lang, "PREFILL")) != nil {
				fail("the prefill tile is still in the tile row")
			}
			// UX-R27: the request box and the prefill box became the two statistics panels, both at
			// every size
			for _, box := range []string{tr(lang, "DECODE · last 60 min"), tr(lang, "PREFILL · last 60 min")} {
				if !strings.Contains(p, "╭─ "+box+" ─") {
					fail("no panel «" + box + "»")
				}
			}
		}
	}
}
