package ui

import (
	"strconv"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// UX-R25 (version 8, third test batch item 34; reviews/_brief_8_UX-R25_v8-uxkr.md): on 5 Settings
// the focused launcher row's description is drawn whole, wrapped by words, in a box of its own
// under the launcher box, in both languages at 120x40 and 80x28; no description names a key (the
// prompt cache's Space is in the footer while its row is focused); every frame keeps its bottom
// edge and no line is wider than the window, also with the next-start note and the development
// build line in the launcher box.

// uxr25Desc is each launcher row's description (row order of vH5).
var uxr25Desc = []string{
	"address = where chat apps connect · next start · this PC only",
	"language = English at start · a change lasts for this run",
	"check run = a short chat test after a start · off, on, or once per model · results in 4 Log",
	"on open = the page shown first",
	"logs kept = days before old logs go",
	"prompt cache = RAM for reused prompts · off (0) by default · next start",
	"warm start = the server reloads its last cache at start · next start",
	"cache autosave = the cache is saved while the server runs · next start",
}

// uxr25Box returns the inner rows of the description box, or nil when there is none.
func uxr25Box(hs *harness, lang string) []string {
	p := hs.plain()
	ls := strings.Split(p, "\n")
	lw := hs.m.r.cw()
	if hs.m.r.wide() {
		lw = hs.m.r.cols(2)[0]
	}
	title := "╭─ " + tr(lang, "DESCRIPTION") + " ─"
	for i, l := range ls {
		if !strings.HasPrefix(ansi.Cut(l, 1, 1+lw), title) {
			continue
		}
		var rows []string
		for _, x := range ls[i+1:] {
			seg := ansi.Cut(x, 1, 1+lw)
			if strings.HasPrefix(seg, "╰") {
				return rows
			}
			rows = append(rows, strings.TrimSpace(strings.Trim(strings.TrimSpace(seg), "│")))
		}
		return nil // no bottom edge
	}
	return nil
}

func uxr25Frames(t *testing.T, name string, p string, w, h int) {
	t.Helper()
	ls := strings.Split(p, "\n")
	if len(ls) > h {
		t.Fatalf("%s: %d lines:\n%s", name, len(ls), p)
	}
	for i, l := range ls {
		if lw := lipgloss.Width(l); lw > w {
			t.Fatalf("%s: line %d is %d wide:\n%s", name, i, lw, p)
		}
	}
	if o, c := strings.Count(p, "╭"), strings.Count(p, "╰"); o != c {
		t.Fatalf("%s: %d frames open, %d closed:\n%s", name, o, c, p)
	}
}

func uxr25Focus(t *testing.T, hs *harness, row int) {
	t.Helper()
	for i := 0; i < 30 && !(hs.m.set5Sec == secLauncher && hs.m.set5Row == row); i++ {
		hs.keys("down")
	}
	if hs.m.set5Sec != secLauncher || hs.m.set5Row != row {
		t.Fatalf("launcher row %d not reached (sec %d row %d)", row, hs.m.set5Sec, hs.m.set5Row)
	}
}

func TestUXR25DescriptionBox(t *testing.T) {
	withNotPorted(t, false)
	// no description names a key, in either language
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, d := range uxr25Desc {
			for _, k := range []string{"←→", "↑↓", "⏎", "Space", "space", "Tab", "tab ", "esc"} {
				if strings.Contains(tr(lang, d), k) {
					t.Errorf("%s: the description %q names the key %q", lang, tr(lang, d), k)
				}
			}
		}
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			shown := 0
			for row := range uxr25Desc {
				hs := newH(t, uxr4Fake(), lang, sz[0], sz[1])
				hs.keys("5")
				if hs.m.set5Hidden(secLauncher, row) {
					continue
				}
				shown++
				uxr25Focus(t, hs, row)
				p := hs.plain()
				name := lang + " " + strconv.Itoa(sz[0]) + "x" + strconv.Itoa(sz[1]) + " " + uxr25Desc[row][:strings.Index(uxr25Desc[row], " =")]
				uxr25Frames(t, name, p, sz[0], sz[1])
				box := uxr25Box(hs, lang)
				if len(box) < 2 {
					t.Fatalf("%s: no description box of two rows or more under the launcher box: %q\n%s", name, box, p)
				}
				got := strings.TrimSpace(strings.Join(box, " "))
				if want := tr(lang, uxr25Desc[row]); got != want {
					t.Fatalf("%s: the description box holds %q, want the whole %q\n%s", name, got, want, p)
				}
				// the launcher box no longer carries the description line
				if strings.Count(p, tr(lang, uxr25Desc[row])[:len(tr(lang, uxr25Desc[row]))/2]) > 1 {
					t.Fatalf("%s: the description is drawn twice:\n%s", name, p)
				}
				ls := strings.Split(p, "\n")
				keys := ls[len(ls)-1]
				typed := strings.Contains(keys, "space") && strings.Contains(keys, tr(lang, "type a value"))
				if typed != (row == set5CapRow) {
					t.Fatalf("%s: the footer names Space %v, want %v: %q", name, typed, row == set5CapRow, keys)
				}
				if strings.Contains(keys, "…") || !strings.Contains(keys, "q "+tr(lang, "exit")) {
					t.Fatalf("%s: the key line is cut: %q", name, keys)
				}
			}
			if shown < 6 {
				t.Fatalf("%s %dx%d: %d launcher rows shown", lang, sz[0], sz[1], shown)
			}
		}
	}
	// a larger window: the box is not smaller, the text stays whole
	for _, lang := range []string{LangEnglish, LangKorean} {
		var n []int
		for _, sz := range [][2]int{{80, 28}, {80, 40}, {120, 28}, {120, 40}} {
			hs := newH(t, uxr4Fake(), lang, sz[0], sz[1])
			hs.keys("5")
			uxr25Focus(t, hs, set5SmokeRow)
			n = append(n, len(uxr25Box(hs, lang)))
		}
		if n[1] < n[0] || n[3] < n[2] || n[0] < 2 {
			t.Fatalf("%s: description rows at 80x28, 80x40, 120x28, 120x40: %v", lang, n)
		}
	}
	// the launcher box with its extra lines: the next-start note while a server runs, the
	// development build line; the frames stay whole
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			f := uxr4Fake()
			f.Info.NextStart = map[string]bool{"warmstart": true, "prompt_cache_cap": true}
			f.Bundles = []BundleCheck{{State: BundleDev}}
			hs := newH(t, f, lang, sz[0], sz[1])
			hs.keys(serving("5")...)
			uxr25Focus(t, hs, set5CapRow)
			p := hs.plain()
			name := lang + " " + strconv.Itoa(sz[0]) + "x" + strconv.Itoa(sz[1]) + " next start + development build"
			uxr25Frames(t, name, p, sz[0], sz[1])
			if box := uxr25Box(hs, lang); strings.TrimSpace(strings.Join(box, " ")) != tr(lang, uxr25Desc[set5CapRow]) {
				t.Fatalf("%s: the description box %q:\n%s", name, box, p)
			}
		}
	}
}
