package ui

// UX-R19 (version 8; the user's acceptance finding 24 (d) with answer 5,
// reviews/_user_8_LAUNCHER-REVIEW_11.md, screenshots 152007 and 153217; sub-lead decision option C):
// 2 Server showed one time graph that «g» switched between decode and prefill/TTFT; the user
// misread the key word, found switching to compare inconvenient, and found no prefill cell as
// clear as decode's. Now there is no switch: from 110 columns the decode graph stands beside the
// request box and, where the height leaves room, the prefill graph beside a prefill box under
// it; the TTFT tile became a prefill tile (prompt rate and TTFT) at every size. This test renders
// 2 Server through the fake backend serving and idle, in English and Korean, at 160x45, 120x40,
// 120x30, 100x30 and 80x28. UX-R27: the request box and the prefill box became the DECODE and
// PREFILL statistics panels; the PREFILL panel (prompt rate and TTFT) shows at every size, beside
// the prefill graph where its row has room, else without it.

import (
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

// uxr19Tile returns the three rows of the tile row's box whose top edge starts with one of the
// titles, or nil.
func uxr19Tile(p string, titles ...string) []string {
	ls := strings.Split(p, "\n")
	for i, l := range ls {
		if !strings.Contains(l, "╭─") || i+3 >= len(ls) {
			continue
		}
		for _, title := range titles {
			j := strings.Index(l, "╭─ "+title+" ")
			if j < 0 {
				continue
			}
			c := lipgloss.Width(l[:j])
			e := c + lipgloss.Width(l[j:j+strings.Index(l[j:], "╮")+len("╮")])
			var rows []string
			for _, x := range ls[i+1 : i+4] {
				rows = append(rows, strings.TrimSpace(cutCells(x, c+1, e-1)))
			}
			return rows
		}
		return nil // the first box row is the tile row
	}
	return nil
}

// cutCells is the plain text between two cell columns.
func cutCells(s string, from, to int) string {
	var b strings.Builder
	n := 0
	for _, r := range s {
		w := lipgloss.Width(string(r))
		if n >= from && n+w <= to {
			b.WriteRune(r)
		}
		n += w
	}
	return b.String()
}

var uxr19Zero = regexp.MustCompile(`│ +0┤`)

func TestUXR19BothGraphs(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		decT, preT := tr(lang, "DECODE · tok/s · last 60 min"), tr(lang, "PREFILL · tok/s per request")
		boxT, hint := tr(lang, "PREFILL · last 60 min"), tr(lang, "samples show when the model answers")
		for _, sz := range [][2]int{{160, 45}, {120, 40}, {120, 30}, {100, 30}, {80, 28}} {
			for _, idle := range []bool{false, true} {
				name := lang + " " + map[bool]string{false: "serving", true: "idle"}[idle]
				f := NewFake()
				if idle {
					idleServer(f)
				}
				hs := newH(t, f, lang, sz[0], sz[1])
				hs.keys("enter")
				hs.beat()
				p := hs.plain()
				ls := strings.Split(p, "\n")
				fail := func(what string) {
					t.Helper()
					t.Fatalf("%s %dx%d: %s:\n%s", name, sz[0], sz[1], what, p)
				}
				// the page whole: within the window, the health line and the footer keys on screen
				if len(ls) > sz[1] || !strings.Contains(p, "/health ok") || !strings.Contains(ls[len(ls)-1], "ctrl+f") {
					fail("the page is not whole")
				}
				// UX-R26 (review item 35 (b)): the third tile is the processing tile at every size; the
				// prefill rate and TTFT are in the prefill box beside the prefill graph
				tile := uxr19Tile(p, tr(lang, "PROCESSING · this request"), tr(lang, "PROCESSING"))
				if tile == nil || !strings.HasPrefix(tile[0], tr(lang, "in")+" ") || uxr19Tile(p, "PREFILL · TTFT", tr(lang, "PREFILL")) != nil {
					fail("no processing tile in the tile row, or a prefill tile there (" + strings.Join(tile, " | ") + ")")
				}
				// no graph switch: no key word on a title, no «g» in the footer
				for _, old := range []string{"prefill · TTFT view", "decode view", "프리필 · TTFT 보기", "디코드 보기", "samples appear", "표본이 쌓임"} {
					if strings.Contains(p, old) {
						fail("the old switch or hint «" + old + "» shows")
					}
				}
				if strings.Contains(ls[len(ls)-1], " g ") {
					fail("the footer still offers g")
				}
				// UX-R27: the PREFILL panel, once, with its prefill and TTFT rows, at every size
				if box := uxr27Box(p, boxT); strings.Count(p, "╭─ "+boxT+" ─") != 1 || len(box) < 3 ||
					!strings.HasPrefix(box[1], "prefill ") || !strings.HasPrefix(box[2], "TTFT ") {
					fail("want the PREFILL panel once, with its prefill and TTFT rows")
				}
				zero := len(uxr19Zero.FindAllString(p, -1)) // the graphs' zero rows
				switch {
				case sz[0] >= 110 && sz[1] >= 40: // both graphs, each beside its panel
					if !strings.Contains(p, "╭─ "+decT+" ─") || !strings.Contains(p, "╭─ "+preT+" ─") || zero != 2 {
						fail("want both graphs with their titles")
					}
					if strings.Index(p, decT) > strings.Index(p, preT) {
						fail("the prefill graph is not under the decode graph")
					}
					if !strings.Contains(p, hint) {
						fail("the empty prefill graph lacks the hint «" + hint + "»")
					}
					if lang == LangKorean && !strings.Contains(p, "╭─ PREFILL · ") {
						fail("prefill is not English in the Korean title")
					}
				case sz[0] >= 110: // the height has no room for the prefill row: it left first
					if !strings.Contains(p, "╭─ "+decT+" ─") || strings.Contains(p, preT) || zero != 1 {
						fail("want the decode graph alone, the prefill row gone")
					}
				default: // below 110 columns: no graph
					if strings.Contains(p, decT) || strings.Contains(p, preT) || zero != 0 {
						fail("want no graph below 110 columns")
					}
				}
				if idle && sz[0] >= 110 && !strings.Contains(p, hint) {
					fail("the empty decode graph lacks the hint «" + hint + "»")
				}
			}
		}
	}
}
