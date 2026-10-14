package ui

// UX-R16b (version 8; the user's acceptance finding 26 with answer ⑥,
// reviews/_user_8_LAUNCHER-REVIEW_11.md, screenshot 154234): the 3 Memory tile READY NOW is
// «RAM free (now)» / «RAM 여유(지금)»; it keeps its total and shows the free and the standby
// lists apart (their sum the total), standby named in a few words where the tile has room, and a
// value that is not measured reads as such, never as a figure or zero. The sub-lead's addenda:
// (1) with no reading the tile reads «—» while showNotPorted is off, never «not ported»;
// (2) the H14 tile notes carry «GB» on every amount (LEFT AFTER «free … − need …», NEEDS
// «model … + … reply»); (3) the 2 Server FREE tile draws its VRAM bar in the VRAM colours of
// UX-R16.

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

// uxr16bTile is the inside text of the tile whose top edge holds one of the labels: its three
// rows, each trimmed. ok is false where no such tile is on the screen.
func uxr16bTile(ls []string, labels ...string) (rows []string, label string, ok bool) {
	for i, l := range ls {
		cs := uxr13Cells(l)
		for _, lb := range labels {
			at := strings.Index(l, "╭─ "+lb+" ")
			if at < 0 {
				continue
			}
			c0 := len(uxr13Cells(l[:at]))
			c1 := c0 + 1
			for c1 < len(cs) && cs[c1] != '╮' {
				c1++
			}
			for _, x := range ls[i+1 : min(len(ls), i+4)] {
				xc := uxr13Cells(x)
				if c1 >= len(xc) {
					return nil, lb, false
				}
				var b strings.Builder
				for _, r := range xc[c0+1 : c1] {
					if r != 0 {
						b.WriteRune(r)
					}
				}
				rows = append(rows, strings.TrimSpace(b.String()))
			}
			return rows, lb, true
		}
	}
	return nil, "", false
}

var uxr16bAmount = regexp.MustCompile(`(\d+\.\d) GB`)

func uxr16bFake() *FakeBackend {
	f := uxr16Fake()
	f.Mem.FreeGB, f.Mem.StandbyGB, f.Mem.ReadyNowGB = K(0.4), K(6.5), K(6.9)
	return f
}

func uxr16bMemory(t *testing.T, f *FakeBackend, lang string, w, h int) string {
	t.Helper()
	hs := newH(t, f, lang, w, h)
	hs.keys(serving("3")...)
	hs.want("H10")
	return hs.plain()
}

func TestUXR16bRAMFree(t *testing.T) {
	withNotPorted(t, false)
	sizes := [][2]int{{120, 40}, {80, 28}}
	labels := func(lang string) []string { return []string{tr(lang, "RAM free (now)"), tr(lang, "RAM free")} }

	t.Run("name, total, free and standby apart", func(t *testing.T) {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
				p := uxr16bMemory(t, uxr16bFake(), lang, sz[0], sz[1])
				mustLack(t, p, "READY NOW", "지금 준비됨", "free+standby", "여유+대기")
				rows, lb, ok := uxr16bTile(strings.Split(p, "\n"), labels(lang)...)
				if !ok {
					t.Fatalf("%s: no tile named %q:\n%s", name, labels(lang), p)
				}
				if sz[0] >= wideCols && lb != tr(lang, "RAM free (now)") {
					t.Fatalf("%s: the tile is named %q, want %q", name, lb, tr(lang, "RAM free (now)"))
				}
				in := strings.Join(rows, " | ")
				total := uxr16bAmount.FindStringSubmatch(rows[0])
				fm := regexp.MustCompile(regexp.QuoteMeta(tr(lang, "free")) + ` (\d+\.\d) GB`).FindStringSubmatch(in)
				sm := regexp.MustCompile(regexp.QuoteMeta(tr(lang, "standby")) + ` (\d+\.\d) GB`).FindStringSubmatch(in)
				if total == nil || fm == nil || sm == nil {
					t.Fatalf("%s: want the total, «%s x.x GB» and «%s y.y GB» apart in the tile: %q", name, tr(lang, "free"), tr(lang, "standby"), in)
				}
				tv, _ := strconv.ParseFloat(total[1], 64)
				fv, _ := strconv.ParseFloat(fm[1], 64)
				sv, _ := strconv.ParseFloat(sm[1], 64)
				if tv != 6.9 || fv != 0.4 || sv != 6.5 || math.Abs(fv+sv-tv) > 0.051 {
					t.Fatalf("%s: total %v, free %v, standby %v: want 6.9 = 0.4 + 6.5 (%q)", name, tv, fv, sv, in)
				}
				if sz[0] >= wideCols && !strings.Contains(in, tr(lang, "cache")) {
					t.Fatalf("%s: standby is not named in the tile with room: %q", name, in)
				}
			}
		}
	})

	t.Run("not measured reads as such", func(t *testing.T) {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
				f := uxr16bFake()
				f.Mem.StandbyGB = U // the memory lists were not read: free is not apart either
				p := uxr16bMemory(t, f, lang, sz[0], sz[1])
				rows, _, ok := uxr16bTile(strings.Split(p, "\n"), labels(lang)...)
				if !ok {
					t.Fatalf("%s: no tile named %q:\n%s", name, labels(lang), p)
				}
				in := strings.Join(rows, " | ")
				if !strings.HasPrefix(rows[0], "6.9 GB") {
					t.Fatalf("%s: the total is not kept: %q", name, in)
				}
				for _, w := range []string{tr(lang, "free"), tr(lang, "standby")} {
					if regexp.MustCompile(regexp.QuoteMeta(w) + ` \d`).MatchString(in) {
						t.Fatalf("%s: a figure for %q without the lists: %q", name, w, in)
					}
				}
				if strings.Contains(in, "0.0 GB") || strings.Contains(in, "0.4 GB") {
					t.Fatalf("%s: a zero or the whole-available figure as «free» without the lists: %q", name, in)
				}
				if w := rows[2]; w != tr(lang, "not measured yet") && !(sz[0] < wideCols && w == tr(lang, "not measured")) {
					t.Fatalf("%s: want the whole word «%s» under the names: %q", name, tr(lang, "not measured yet"), in)
				}
				if !strings.Contains(in, tr(lang, "standby")) {
					t.Fatalf("%s: the standby row is gone: %q", name, in)
				}
			}
		}
	})

	t.Run("no reading: a dash, never not ported", func(t *testing.T) {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
				f := uxr16bFake()
				f.Mem.ReadyNowGB, f.Mem.FreeGB, f.Mem.StandbyGB = Val{}, Val{}, Val{}
				p := uxr16bMemory(t, f, lang, sz[0], sz[1])
				rows, _, ok := uxr16bTile(strings.Split(p, "\n"), labels(lang)...)
				if !ok {
					t.Fatalf("%s: no tile named %q:\n%s", name, labels(lang), p)
				}
				in := strings.Join(rows, " | ")
				if !strings.HasPrefix(rows[0], "—") || strings.Contains(in, "not ported") || strings.Contains(in, "미지원") {
					t.Fatalf("%s: switch off, no reading: want «—», no not-ported word: %q", name, in)
				}
				withNotPorted(t, true)
				on := uxr16bMemory(t, f, lang, sz[0], sz[1])
				withNotPorted(t, false)
				rows, _, _ = uxr16bTile(strings.Split(on, "\n"), labels(lang)...)
				if !strings.Contains(strings.Join(rows, " "), tr(lang, "not ported")) {
					t.Fatalf("%s: switch on: the not-ported word is not back: %q", name, rows)
				}
			}
		}
	})

	t.Run("H14 tile notes carry GB", func(t *testing.T) {
		decimal := regexp.MustCompile(`(^|[^\w.])(\d+\.\d+)( ?GB| tok/s| MB/s)?`)
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range [][2]int{{160, 45}, {120, 40}, {110, 40}, {80, 28}} {
				name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
				f := NewFake()
				f.Pre.ModelGB, f.Pre.ReplyGB = K(12.0), K(7.0)
				hs := newH(t, f, lang, sz[0], sz[1])
				hs.keys("3")
				hs.want("H14")
				p := hs.plain()
				if sz[0] == 160 {
					mustHave(t, p, tr(lang, "free")+" 24.6 GB − "+strings.TrimSuffix(strings.SplitN(tr(lang, "free %s − need %s"), "− ", 2)[1], " %s")+" 19.0 GB",
						tr(lang, "model")+" 12.0 GB + 7.0 GB "+tr(lang, "reply"))
				}
				if sz[0] >= 110 {
					mustHave(t, flat(p), flat(tr(lang, "free")+" 24.6 GB − "))
				}
				for _, m := range decimal.FindAllStringSubmatch(p, -1) {
					if m[3] == "" {
						t.Fatalf("%s: the amount %q has no unit:\n%s", name, m[2], p)
					}
				}
			}
		}
	})

	t.Run("2 Server FREE tile: VRAM bar in the VRAM colours", func(t *testing.T) {
		old := lipgloss.ColorProfile()
		lipgloss.SetColorProfile(termenv.ANSI)
		t.Cleanup(func() { lipgloss.SetColorProfile(old) })
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range [][2]int{{120, 30}, {80, 28}} {
				name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
				hs := newH(t, NewFake(), lang, sz[0], sz[1])
				hs.keys(serving("2")...)
				hs.beat()
				view := hs.m.View()
				bar := func(label string) map[string]bool {
					for _, l := range strings.Split(view, "\n") {
						cs := uxr16Cells(l)
						for i := range cs {
							if i+len(label) < len(cs) && string(runesOf(cs[i:i+len(label)])) == label && (i == 0 || cs[i-1].r == ' ' || cs[i-1].r == '│') {
								got := map[string]bool{}
								for _, c := range cs[i+len(label):] {
									if c.r == '│' {
										break
									}
									if strings.ContainsRune("█▓", c.r) {
										got[c.fg] = true
									}
								}
								if len(got) > 0 {
									return got
								}
							}
						}
					}
					return nil
				}
				ram, vram := bar("RAM  "), bar("VRAM ")
				if len(ram) == 0 || len(vram) == 0 {
					t.Fatalf("%s: no FREE tile bars (RAM %v, VRAM %v):\n%s", name, ram, vram, ansi.Strip(view))
				}
				for c := range vram {
					if c != "95" && c != "35" {
						t.Fatalf("%s: the FREE tile's VRAM bar in colour %q, want the VRAM colours 95/35 (RAM %v, VRAM %v)", name, c, ram, vram)
					}
				}
				for c := range ram {
					if c == "95" || c == "35" {
						t.Fatalf("%s: the FREE tile's RAM bar in a VRAM colour %q", name, c)
					}
				}
				// the key names both glyph pairs with whole words at every width
				rows, _, ok := uxr16bTile(strings.Split(ansi.Strip(view), "\n"), tr(lang, "FREE"))
				if !ok || !strings.Contains(rows[2], "██") || !strings.Contains(rows[2], "▓▓") || !strings.HasSuffix(rows[2], tr(lang, "other")) {
					t.Fatalf("%s: the FREE key, want both glyph pairs and the whole word «%s» last: %q", name, tr(lang, "other"), rows)
				}
			}
		}
	})
}

func runesOf(cs []uxr16Cell) []rune {
	rs := make([]rune, len(cs))
	for i, c := range cs {
		rs[i] = c.r
	}
	return rs
}
