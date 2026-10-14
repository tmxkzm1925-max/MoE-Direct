package ui

// UX-R16 (version 8; the user's acceptance finding 26, reviews/_user_8_LAUNCHER-REVIEW_11.md item
// 26, screenshots 153733 and 154234): on 3 Memory the VRAM bar has colours of its own, apart from
// every RAM segment colour, with a key of its own under it; the VRAM counter line is gone, its
// values in that key; the model's breakdown is the bottom line, in the user's words «전문가 캐시 …
// (예산 할당 …)»; every amount carries «GB»; and with showNotPorted off no not-ported value shows
// (the cause box holding only «cause rules: not ported» is not drawn, the PAGEFILE tile reads —).
// The READY NOW tile is not this atom's.

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/termenv"
)

// uxr16Fake is the fake backend serving, with the memory reading of the user's screenshots
// (153733: RAM 31 GB, model 12.2, apps 11.8, modified 0.1, standby 6.5, free 0.5; VRAM 16 GB, in
// use 3.9, this server 2.9; expert cache 12.0, budget 12, other 1.1; 154234: commit 37.4 of 39.4,
// safe line 4, ready now 6.9).
func uxr16Fake() *FakeBackend {
	f := NewFake()
	f.Mem = MemoryReading{TotalRAMGB: K(31), ModelGB: K(12.2), AppsGB: K(11.8), ModifiedGB: K(0.1), StandbyGB: K(6.5),
		FreeGB: K(0.5), ReadyNowGB: K(6.9), TotalVRAMGB: K(16), UsedVRAMGB: K(3.9), ModelVRAMGB: K(2.9), OtherVRAMGB: K(1.0),
		CommitUsedGB: K(37.4), CommitLimitGB: K(39.4), CommitSafeGB: K(4),
		ExpertCacheGB: K(12.0), BudgetGB: K(12), FileMapGB: U, OtherGB: K(1.1),
		Apps: []AppMem{{Name: "llama-server.exe", GB: K(12.2), Tag: "this model"}}}
	return f
}

// uxr16Cell is one drawn character with the foreground SGR it was drawn in ("" = default).
type uxr16Cell struct {
	r  rune
	fg string
}

// uxr16Cells reads a rendered line into its characters and their foreground colour.
func uxr16Cells(l string) []uxr16Cell {
	var out []uxr16Cell
	fgc := ""
	for i := 0; i < len(l); {
		if strings.HasPrefix(l[i:], "\x1b[") {
			j := i + 2
			for j < len(l) && (l[j] < 0x40 || l[j] > 0x7e) {
				j++
			}
			if j < len(l) && l[j] == 'm' {
				ps := strings.Split(l[i+2:j], ";")
				for k := 0; k < len(ps); k++ {
					n, _ := strconv.Atoi(ps[k])
					switch {
					case ps[k] == "" || n == 0 || n == 39:
						fgc = ""
					case (n >= 30 && n <= 37) || (n >= 90 && n <= 97):
						fgc = ps[k]
					case (n == 38) && k+2 < len(ps) && ps[k+1] == "5":
						fgc = "38;5;" + ps[k+2]
						k += 2
					case (n == 38) && k+4 < len(ps) && ps[k+1] == "2":
						fgc = "38;2;" + strings.Join(ps[k+2:k+5], ";")
						k += 4
					}
				}
			}
			i = j + 1
			continue
		}
		r, w := []rune(l[i:])[0], len(string([]rune(l[i:])[0]))
		out = append(out, uxr16Cell{r, fgc})
		i += w
	}
	return out
}

// uxr16Colours is the set of foreground colours of the bar glyphs (█ ▓ ▒ ░) of a line.
func uxr16Colours(l string) map[string]bool {
	cs := map[string]bool{}
	for _, c := range uxr16Cells(l) {
		if strings.ContainsRune("█▓▒░", c.r) {
			cs[c.fg] = true
		}
	}
	return cs
}

// uxr16Schemes are a dark and a light terminal theme: the 16 ANSI colours and the background of
// the Windows Terminal built-in schemes Campbell (dark, the default) and One Half Light (light).
var uxr16Schemes = map[string]struct {
	bg  [3]int
	pal [16][3]int
}{
	"Campbell (dark)": {[3]int{0x0C, 0x0C, 0x0C}, [16][3]int{
		{0x0C, 0x0C, 0x0C}, {0xC5, 0x0F, 0x1F}, {0x13, 0xA1, 0x0E}, {0xC1, 0x9C, 0x00}, {0x00, 0x37, 0xDA}, {0x88, 0x17, 0x98}, {0x3A, 0x96, 0xDD}, {0xCC, 0xCC, 0xCC},
		{0x76, 0x76, 0x76}, {0xE7, 0x48, 0x56}, {0x16, 0xC6, 0x0C}, {0xF9, 0xF1, 0xA5}, {0x3B, 0x78, 0xFF}, {0xB4, 0x00, 0x9E}, {0x61, 0xD6, 0xD6}, {0xF2, 0xF2, 0xF2}}},
	"One Half Light (light)": {[3]int{0xFA, 0xFA, 0xFA}, [16][3]int{
		{0x38, 0x3A, 0x42}, {0xE4, 0x56, 0x49}, {0x50, 0xA1, 0x4F}, {0xC1, 0x83, 0x01}, {0x01, 0x84, 0xBC}, {0xA6, 0x26, 0xA4}, {0x09, 0x97, 0xB3}, {0xFA, 0xFA, 0xFA},
		{0x4F, 0x52, 0x5D}, {0xDF, 0x6C, 0x75}, {0x98, 0xC3, 0x79}, {0xE4, 0xC0, 0x7A}, {0x61, 0xAF, 0xEF}, {0xC5, 0x77, 0xDD}, {0x56, 0xB5, 0xC1}, {0xFF, 0xFF, 0xFF}}},
}

// uxr16RGB is an SGR foreground code of the 16-colour profile in a scheme.
func uxr16RGB(t *testing.T, code string, pal [16][3]int) [3]int {
	t.Helper()
	n, err := strconv.Atoi(code)
	switch {
	case err == nil && n >= 30 && n <= 37:
		return pal[n-30]
	case err == nil && n >= 90 && n <= 97:
		return pal[n-90+8]
	}
	t.Fatalf("colour %q is not one of the 16 ANSI colours", code)
	return [3]int{}
}

func uxr16Dist(a, b [3]int) float64 {
	d := 0.0
	for i := range a {
		d += float64((a[i] - b[i]) * (a[i] - b[i]))
	}
	return math.Sqrt(d)
}

// uxr16Panel is the MEMORY · now panel: the screen line of its top edge and its inside rows.
func uxr16Panel(t *testing.T, name string, ls []string, title string) (top int, rows []string) {
	t.Helper()
	top = -1
	for i, l := range ls {
		if strings.Contains(l, "╭─ "+title) {
			top = i
		}
	}
	if top < 0 {
		t.Fatalf("%s: no %q panel:\n%s", name, title, strings.Join(ls, "\n"))
	}
	for _, l := range ls[top+1:] {
		x := strings.TrimSpace(l)
		if strings.HasPrefix(x, "╰") {
			return top, rows
		}
		x = strings.TrimSuffix(strings.TrimPrefix(x, "│"), "│")
		rows = append(rows, strings.TrimRight(strings.TrimPrefix(x, " "), " "))
	}
	t.Fatalf("%s: the %q panel has no bottom edge on the screen:\n%s", name, title, strings.Join(ls, "\n"))
	return
}

var (
	uxr16Decimal = regexp.MustCompile(`(^|[^\w.])(\d+\.\d+)( GB| tok/s| MB/s)?`)
	// the whole numbers that are amounts here: the safe line and the budget
	uxr16Whole = regexp.MustCompile(`(≥|\(budget|\(예산 할당) (\d+)( GB)?`)
)

func TestUXR16Memory(t *testing.T) {
	old := lipgloss.ColorProfile()
	lipgloss.SetColorProfile(termenv.ANSI)
	t.Cleanup(func() { lipgloss.SetColorProfile(old); lipgloss.SetHasDarkBackground(true) })

	want := map[string]struct {
		title, ramKey, inUse, server, other, breakdown, notPorted, cause string
		lack                                                             []string
	}{
		LangEnglish: {"MEMORY · now", "model 12.2 GB", "in use 3.9 GB", "this server 2.9 GB", "other programs 1.0 GB",
			"model 12.2 GB = expert cache 12.0 GB (budget 12 GB) · file mapping not measured yet · other 1.1 GB",
			"not ported", "LIKELY CAUSE", []string{"two counters", "(budget 12)"}},
		LangKorean: {"메모리 · 지금", "모델 12.2 GB", "사용 중 3.9 GB", "이 서버 2.9 GB", "다른 프로그램 1.0 GB",
			"모델 12.2 GB = 전문가 캐시 12.0 GB (예산 할당 12 GB) · 파일 매핑 측정 전 · 기타 1.1 GB",
			"미지원", "예상 원인", []string{"카운터 둘", "익스퍼트", "budget", "(예산 할당 12)"}},
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		wl := want[lang]
		for _, sz := range [][2]int{{120, 40}, {80, 28}} {
			for _, dark := range []bool{true, false} {
				for _, gs := range []GlyphSet{GlyphDesign, GlyphFallback} {
					name := fmt.Sprintf("%s %dx%d dark=%v glyphs=%s", lang, sz[0], sz[1], dark, gs)
					lipgloss.SetHasDarkBackground(dark)
					withNotPorted(t, false)
					hs := newH(t, uxr16Fake(), lang, sz[0], sz[1])
					hs.keys(serving("3")...)
					hs.want("H10")
					designPage := hs.plain() // the whole-number check reads «≥», which the fallback set writes «>»
					hs.m.glyphs = gs
					view := hs.m.View()
					cl := strings.Split(view, "\n")
					ls := strings.Split(ansi.Strip(view), "\n")
					page := strings.Join(ls, "\n")
					if len(ls) > sz[1] {
						t.Fatalf("%s: %d lines in a %d-line window", name, len(ls), sz[1])
					}

					// (1) the panel: RAM bar, its key, VRAM bar (once), its key, the breakdown last
					top, rows := uxr16Panel(t, name, ls, gs.Apply(wl.title))
					if len(rows) == 0 || !strings.HasPrefix(rows[0], "RAM ") {
						t.Fatalf("%s: the panel does not start with the RAM bar: %q", name, rows)
					}
					vi, bi := -1, -1
					for i, x := range rows {
						if strings.HasPrefix(x, "VRAM") {
							if vi >= 0 {
								t.Fatalf("%s: a second VRAM line (the counter line is gone): %q", name, x)
							}
							vi = i
						}
						if strings.HasPrefix(x, strings.SplitN(gs.Apply(wl.breakdown), " = ", 2)[0]+" = ") {
							bi = i
						}
					}
					if vi < 2 || bi < vi+2 {
						t.Fatalf("%s: want RAM bar, RAM key, VRAM bar, VRAM key, breakdown; VRAM at %d, breakdown at %d:\n%s", name, vi, bi, strings.Join(rows, "\n"))
					}
					ramKey := strings.Join(rows[1:vi], " ")
					if !strings.Contains(ramKey, wl.ramKey) || strings.Contains(ramKey, "VRAM") {
						t.Fatalf("%s: RAM key %q", name, ramKey)
					}
					vk := strings.Join(rows[vi+1:bi], " ")
					ws := []string{wl.server, wl.other}
					if sz[0] >= 120 { // UX-R22: in use only as the label of the sum, where the line has room
						ws = append(ws, wl.inUse)
					}
					for _, w := range ws {
						if !strings.Contains(vk, w) {
							t.Fatalf("%s: the VRAM key under the VRAM bar lacks %q: %q", name, w, vk)
						}
					}
					bd := rows[bi]
					for _, x := range rows[bi+1:] {
						if !strings.HasPrefix(x, "  ") {
							t.Fatalf("%s: a row after the breakdown that is not its continuation: %q", name, x)
						}
						bd += gs.Apply(" · ") + strings.TrimSpace(x)
					}
					if bd != gs.Apply(wl.breakdown) {
						t.Fatalf("%s: the bottom line\n got %q\nwant %q", name, bd, gs.Apply(wl.breakdown))
					}
					mustLack(t, page, wl.lack...)

					// (2) colours: the VRAM bar's glyphs in colours no RAM segment has, its key in them
					ram, vram := uxr16Colours(cl[top+1]), uxr16Colours(cl[top+1+vi])
					if len(ram) < 3 || len(vram) == 0 {
						t.Fatalf("%s: bar colours RAM %v VRAM %v", name, ram, vram)
					}
					for c := range vram {
						if ram[c] {
							t.Fatalf("%s: the VRAM bar shares colour %q with a RAM segment (RAM %v, VRAM %v)", name, c, ram, vram)
						}
					}
					keyCols := map[string]bool{}
					for _, l := range cl[top+1+vi+1 : top+1+bi] {
						for c := range uxr16Colours(l) {
							keyCols[c] = true
						}
					}
					if fmt.Sprint(keyCols) != fmt.Sprint(vram) {
						t.Fatalf("%s: the VRAM key's glyph colours %v are not the VRAM bar's %v", name, keyCols, vram)
					}
					for sn, sc := range uxr16Schemes {
						for v := range vram {
							rv := uxr16RGB(t, v, sc.pal)
							if d := uxr16Dist(rv, sc.bg); d < 60 {
								t.Fatalf("%s %s: VRAM colour %s is %.0f from the background", name, sn, v, d)
							}
							for c := range ram {
								if d := uxr16Dist(rv, uxr16RGB(t, c, sc.pal)); d < 60 {
									t.Fatalf("%s %s: VRAM colour %s is %.0f from RAM colour %s", name, sn, v, d, c)
								}
							}
						}
					}

					// (3) «GB» on every amount
					for _, m := range uxr16Decimal.FindAllStringSubmatch(page, -1) {
						if m[3] == "" {
							t.Fatalf("%s: the amount %q has no unit:\n%s", name, m[2], page)
						}
					}
					for _, m := range uxr16Whole.FindAllStringSubmatch(designPage, -1) {
						if m[3] == "" {
							t.Fatalf("%s: the amount %q has no unit:\n%s", name, m[0], page)
						}
					}
					mustHave(t, page, "31 GB", "16 GB", "37.4 GB", "39.4 GB", gs.Apply("≥ 4 GB"))

					// (4) no not-ported value with the switch off; the box with nothing else is not drawn
					mustLack(t, page, "not ported", "미지원", wl.cause)
				}
			}

			// (4) with the switch on the not-ported values are back; a real cause keeps its box
			withNotPorted(t, true)
			on := newH(t, uxr16Fake(), lang, sz[0], sz[1])
			on.keys(serving("3")...)
			mustHave(t, on.plain(), wl.notPorted)
			if sz[0] >= wideCols {
				mustHave(t, on.plain(), wl.cause)
			}
			withNotPorted(t, false)
			f := uxr16Fake()
			f.Mem.Causes = []string{"chrome.exe +5.0 GB while it got slow"}
			cs := newH(t, f, lang, sz[0], sz[1])
			cs.keys(serving("3")...)
			if sz[0] >= wideCols {
				mustHave(t, cs.plain(), wl.cause, "chrome.exe +5.0 GB")
			}
			mustLack(t, cs.plain(), "not ported", "미지원")
			// the reading with the values the wire never delivers (no commit, no expert cache
			// split, no pagefile rate, no causes): no not-ported value either
			bare := NewFake()
			bare.Mem.Causes, bare.Mem.PagefileMBs = nil, Val{}
			bh := newH(t, bare, lang, sz[0], sz[1])
			bh.keys(serving("3")...)
			mustLack(t, bh.plain(), "not ported", "미지원", wl.cause)
		}
	}
}
