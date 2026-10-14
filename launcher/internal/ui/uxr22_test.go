package ui

// UX-R22 (version 8; the user's acceptance item 33, reviews/_user_8_LAUNCHER-REVIEW_12.md,
// screenshot 185755): (a) the 3 Memory tile «RAM free (now)» says in plain words what each
// figure is: the sum is the RAM usable now, free is the RAM empty now, standby is the cache
// Windows gives up on demand (each gloss where the tile has room, never cut); (b) the VRAM key is
// three parts, this server / other programs / free, with the card's use no part of its own, only
// the label of their sum where the line has room; 1 Models' RAM tile follows the same rule (this
// server / other programs / left), and no tile line there is wider than its box at 80x28.

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

// uxr22Fake is the fake backend serving with the reading of the user's screenshot 185755
// (RAM 31 GB: model 4.4, apps 14.7, modified 0.2, standby 11.8, free 0.1, available 11.9;
// VRAM 16 GB: in use 3.8, this server 2.7, other programs 1.1; commit 40.9 of 41.7).
func uxr22Fake() *FakeBackend {
	f := NewFake()
	f.Mem = MemoryReading{TotalRAMGB: K(31), ModelGB: K(4.4), AppsGB: K(14.7), ModifiedGB: K(0.2), StandbyGB: K(11.8),
		FreeGB: K(0.1), ReadyNowGB: K(11.9), TotalVRAMGB: K(16), UsedVRAMGB: K(3.8), ModelVRAMGB: K(2.7), OtherVRAMGB: K(1.1),
		CommitUsedGB: K(40.9), CommitLimitGB: K(41.7), CommitSafeGB: K(4),
		ExpertCacheGB: K(12.0), BudgetGB: K(12), FileMapGB: U, OtherGB: K(0.0),
		Apps: []AppMem{{Name: "llama-server.exe", GB: K(4.4), Tag: "this model"}}}
	return f
}

func TestUXR22MemoryWords(t *testing.T) {
	withNotPorted(t, false)
	oneOf := func(s string, xs ...string) bool {
		for _, x := range xs {
			if s == x {
				return true
			}
		}
		return false
	}
	endsOne := func(s string, xs ...string) bool {
		for _, x := range xs {
			if strings.HasSuffix(s, x) {
				return true
			}
		}
		return false
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		T := func(s string) string { return tr(lang, s) }
		for _, sz := range [][2]int{{160, 45}, {120, 40}, {80, 28}} {
			name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
			hs := newH(t, uxr22Fake(), lang, sz[0], sz[1])
			hs.keys(serving("3")...)
			hs.want("H10")
			p := hs.plain()
			ls := strings.Split(p, "\n")
			if len(ls) > sz[1] {
				t.Fatalf("%s: %d lines in a %d-line window", name, len(ls), sz[1])
			}
			for _, l := range ls {
				if lipgloss.Width(l) > sz[0] {
					t.Fatalf("%s: a line wider than the window: %q", name, l)
				}
			}

			// (a) the RAM-free tile: each figure whole, with its gloss where the tile has room
			rows, _, ok := uxr16bTile(ls, T("RAM free (now)"), T("RAM free"))
			if !ok {
				t.Fatalf("%s: no RAM free tile:\n%s", name, p)
			}
			in := strings.Join(rows, " | ")
			freeWhy := []string{T("RAM empty now"), T("empty RAM"), T("empty")}
			stbyWhy := []string{T("cache Windows gives up on demand"), T("given up on demand"), T("cache")}
			if !strings.HasPrefix(rows[0], "11.9 GB") || !endsOne(rows[0], "GB", T("usable now")) {
				t.Fatalf("%s: the sum row %q (want «11.9 GB», its gloss whole or none)", name, rows[0])
			}
			if !strings.Contains(rows[1], T("free")+" 0.1 GB") || !endsOne(rows[1], append([]string{"GB"}, freeWhy...)...) {
				t.Fatalf("%s: the free row %q (want «%s 0.1 GB», its gloss whole or none)", name, rows[1], T("free"))
			}
			if !(strings.Contains(rows[2], T("standby")+" 11.8 GB") || strings.Contains(rows[2], T("standby")+" 11.8GB")) ||
				!endsOne(rows[2], append([]string{"GB"}, stbyWhy...)...) {
				t.Fatalf("%s: the standby row %q (want «%s 11.8 GB», its gloss whole or none)", name, rows[2], T("standby"))
			}
			if sz[0] >= 120 {
				if !strings.Contains(rows[0], T("usable now")) || !endsOne(rows[1], freeWhy...) || !endsOne(rows[2], stbyWhy...) {
					t.Fatalf("%s: with room each figure says what it is: %q", name, in)
				}
			}
			if sz[0] >= 160 && (!strings.HasSuffix(rows[1], T("RAM empty now")) || (lang == LangKorean && !strings.HasSuffix(rows[2], T("given up on demand")))) {
				t.Fatalf("%s: the wide tile's longer glosses: %q", name, in)
			}

			// (b) the VRAM key: this server / other programs / free; in use only as the sum's label
			vk := ""
			for i, l := range ls {
				if strings.HasPrefix(strings.TrimLeft(l, " │"), "VRAM ") && i+1 < len(ls) {
					vk = strings.TrimSpace(strings.Trim(strings.TrimSpace(ls[i+1]), "│"))
					break
				}
			}
			for _, w := range []string{"█ " + T("this server") + " 2.7 GB", "▓ " + T("other programs") + " 1.1 GB", "· " + T("free") + " 12.2 GB"} {
				if !strings.Contains(vk, w) {
					t.Fatalf("%s: the VRAM key lacks %q: %q", name, w, vk)
				}
			}
			if strings.Contains(vk, T("in use")) && !strings.Contains(vk, "2.7 GB + ▓ "+T("other programs")+" 1.1 GB = "+T("in use")+" 3.8 GB") {
				t.Fatalf("%s: «%s» beside this server, not as the sum's label: %q", name, T("in use"), vk)
			}
			if sz[0] >= 120 && !strings.Contains(vk, " = "+T("in use")+" 3.8 GB") {
				t.Fatalf("%s: the card's use as the sum's label: %q", name, vk)
			}

			// 1 Models, serving: the RAM tile without «in use», each tile line whole
			if sz[0] == 160 {
				continue
			}
			hm := newH(t, uxr22Fake(), lang, sz[0], sz[1])
			hm.keys(serving("1")...)
			mp := hm.plain()
			mls := strings.Split(mp, "\n")
			ram, _, ok := uxr16bTile(mls, "RAM · 31 GB")
			if !ok {
				t.Fatalf("%s: no 1 Models RAM tile:\n%s", name, mp)
			}
			left := strings.ReplaceAll(T("%s left"), "%s", "11.9 GB")
			if strings.Contains(strings.Join(ram, " "), T("in use")) ||
				!oneOf(ram[0], "4.4 GB "+T("this server"), "4.4 GB "+T("server")) ||
				!oneOf(ram[2], T("other programs")+" 14.7 GB · "+left, T("other")+" 14.7 GB · "+left, left) {
				t.Fatalf("%s: 1 Models RAM tile, want this server / other programs / left, whole, no «%s»: %q", name, T("in use"), ram)
			}
			vr, _, ok := uxr16bTile(mls, "VRAM")
			if !ok || !oneOf(vr[2], T("this server")+" 2.7 GB", T("server")+" 2.7 GB") {
				t.Fatalf("%s: 1 Models VRAM tile, want «%s 2.7 GB» whole: %q", name, T("this server"), vr)
			}
		}
	}
}
