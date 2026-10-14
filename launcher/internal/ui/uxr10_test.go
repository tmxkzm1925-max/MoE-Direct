package ui

// UX-R10 (version 8; the user's acceptance findings 16 (b)-(e), 17 and 18,
// reviews/_user_8_LAUNCHER-REVIEW_5.md, _6.md and _7.md, screenshots 061107, 061348 and
// 061352): the settings box of 1 Models marks each row's state in a column of its own (·
// the default, ● a saved value other than the default, ◆ changed and not saved) without
// moving the value; every row stands in the box, editable in place (no «▸ n more» fold, no
// «a»), scrolling with a bar where the window is low; the title keeps the model's name and
// leaves out only its quant part; a model outside the catalog with a structure template
// reads as what it is; the RAM tile names a shortfall as UX-R3 does.

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// uxr10Fake: the settings rows carry their defaults as the wire gives them, and the queue
// is saved at 16 against its default 8.
func uxr10Fake() *FakeBackend {
	f := NewFake()
	v := f.Views["qwen35-122b"]
	v.Saved = true
	v.Rows = append([]SettingRow(nil), v.Rows...)
	for i := range v.Rows {
		switch v.Rows[i].Key {
		case "budget":
			v.Rows[i].Default = "12 GB"
		case "ctx":
			v.Rows[i].Default = "12288"
		case "qd":
			v.Rows[i].Value, v.Rows[i].Default = "16", "8"
		}
	}
	f.Views["qwen35-122b"] = v
	return f
}

// uxr10Box is the settings box as drawn (no height limit), each line inside its frame with
// its leading cells kept, so the columns can be compared.
func uxr10Box(hs *harness) []string {
	row, _ := hs.m.selRow()
	w := hs.m.r.cols(3)[1]
	if !hs.m.r.wide() {
		w = hs.m.r.cols(2)[1]
	}
	ls := strings.Split(ansi.Strip(hs.m.setPanel(w, row, hs.m.focusSet, false, 0, 0)), "\n")
	var out []string
	for _, l := range ls[1 : len(ls)-1] {
		rs := []rune(l)
		out = append(out, string(rs[1:len(rs)-1]))
	}
	return out
}

// uxr10Col is the column where val starts on the box line that holds label, -1 if none.
func uxr10Col(lines []string, label, val string) int {
	for _, l := range lines {
		if i := strings.Index(l, label); i >= 0 {
			if j := strings.Index(l[i+len(label):], val); j >= 0 {
				return lipgloss.Width(l[:i+len(label)+j])
			}
		}
	}
	return -1
}

func uxr10Line(lines []string, label string) string {
	for _, l := range lines {
		if strings.Contains(l, label) {
			return l
		}
	}
	return ""
}

// TestUXR10States (finding 16 (b), (d)): the default, saved and changed states are told
// apart on each row by a mark of its own column, and the value stays in its column in every
// state; the line under the rows names the marks shown. Korean and English, 120 and 80
// columns.
func TestUXR10States(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, w := range []int{120, 80} {
			name := fmt.Sprintf("%s %d", lang, w)
			hs := newH(t, uxr10Fake(), lang, w, 40)
			T := hs.m.r.T
			bud, ctx, qd := T("RAM budget"), T("context"), T("queue")
			lines := uxr10Box(hs)
			if l := uxr10Line(lines, bud); !strings.Contains(l, "· "+bud) {
				t.Fatalf("%s: the default row lacks its mark ·: %q\n%s", name, l, strings.Join(lines, "\n"))
			}
			if l := uxr10Line(lines, qd); !strings.Contains(l, "● "+qd) {
				t.Fatalf("%s: the saved row lacks its mark ●: %q\n%s", name, l, strings.Join(lines, "\n"))
			}
			keys := "· " + T("default") + "  ● " + T("saved")
			if l := uxr10Line(lines, keys); l == "" || strings.Contains(l, "◆") {
				t.Fatalf("%s: the marks' line is not «%s» alone:\n%s", name, keys, strings.Join(lines, "\n"))
			}
			col := uxr10Col(lines, bud, "12 GB")
			if c := uxr10Col(lines, ctx, "12288"); c != col || col < 0 {
				t.Fatalf("%s: the values stand in columns %d and %d:\n%s", name, col, c, strings.Join(lines, "\n"))
			}
			if c := uxr10Col(lines, qd, "16"); c != col {
				t.Fatalf("%s: the saved value moved to column %d from %d:\n%s", name, c, col, strings.Join(lines, "\n"))
			}
			// changed and not saved: ◆, the value in the same column
			hs.keys("tab", "right", "down")
			lines = uxr10Box(hs)
			if l := uxr10Line(lines, bud); !strings.Contains(l, "◆ "+bud) {
				t.Fatalf("%s: the changed row lacks its mark ◆: %q\n%s", name, l, strings.Join(lines, "\n"))
			}
			if c := uxr10Col(lines, bud, "16 GB"); c != col {
				t.Fatalf("%s: the changed value moved to column %d from %d:\n%s", name, c, col, strings.Join(lines, "\n"))
			}
			if uxr10Line(lines, "◆ "+T("not saved")) == "" {
				t.Fatalf("%s: the marks' line lacks «◆ %s»:\n%s", name, T("not saved"), strings.Join(lines, "\n"))
			}
			mustHave(t, hs.plain(), "◆ "+bud, "● "+qd)
			t.Logf("%s:\n%s", name, strings.Join(lines, "\n"))
		}
	}
}

// TestUXR10NoFold (finding 17): no «▸ n more» line and no «a» entry; every row stands in
// the box and changes in place, the formerly folded ones too; «a» changes nothing.
func TestUXR10NoFold(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, NewFake(), lang, 120, 40)
		T := hs.m.r.T
		p := hs.plain()
		for _, l := range []string{"RAM budget", "context", "queue", "threads", "prefetch", "repack mode", "KV type", "parallel"} {
			mustHave(t, p, T(l))
		}
		mustLack(t, p, "▸", "a opens", "a 열기", T("more"))
		hs.keys("tab")
		before := hs.plain()
		hs.keys("a")
		if hs.plain() != before || hs.m.setCur != 0 {
			t.Fatalf("%s: «a» changed the box:\n%s", lang, hs.plain())
		}
		hs.keys("down", "down", "down", "right") // threads, a folded row before UX-R10
		if got := hs.m.edits["qwen35-122b"]["threads"]; got != "12" {
			t.Fatalf("%s: the threads row did not change in place: %q", lang, got)
		}
		mustHave(t, hs.plain(), "▶◆ "+T("threads"))
		hs.keys("down", "down", "down", "down")
		mustHave(t, hs.plain(), "▶· "+T("parallel"))
	}
}

// TestUXR10Fits (finding 17, the sub-lead's order): the box row stays inside the window:
// the boxes' foot and the conflict-10 line under them are drawn at 120x30 and 80x24; where
// the rows do not fit they scroll with the cursor, a bar beside them, and each row is
// reached.
func TestUXR10Fits(t *testing.T) {
	labels := []string{"RAM budget", "context", "queue", "threads", "prefetch", "repack mode", "KV type", "parallel"}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, sz := range [][2]int{{120, 30}, {80, 24}, {120, 24}} {
			name := fmt.Sprintf("%s %dx%d", lang, sz[0], sz[1])
			hs := newH(t, NewFake(), lang, sz[0], sz[1])
			T := hs.m.r.T
			hs.keys("tab", "right") // a custom value: the conflict-10 line
			for i, l := range labels {
				if i > 0 {
					hs.keys("down")
				}
				p := hs.plain()
				mark := "▶· "
				if i == 0 {
					mark = "▶◆ "
				}
				mustHave(t, p, mark+T(l), "▲ "+T("custom settings · performance unverified"))
				uxr6Closed(t, fmt.Sprintf("%s row %d", name, i), hs)
				if len(strings.Split(p, "\n")) > sz[1] {
					t.Fatalf("%s: the frame is taller than the window", name)
				}
			}
			if sz[1] == 24 { // the low windows scroll: the bar stands beside the shown rows
				var line string
				for _, l := range strings.Split(hs.plain(), "\n") {
					if strings.Contains(l, "▶· "+T("parallel")) {
						line = l
					}
				}
				if !strings.Contains(line, "█ │") {
					t.Fatalf("%s: no scroll bar beside the focused row: %q", name, line)
				}
				hs.keys("tab")
				mustHave(t, hs.plain(), "◆ "+T("RAM budget")) // unfocused, the box shows its first rows
			}
		}
	}
}

// TestUXR10Title (finding 16 (c)): the title keeps the model's name where the box has room
// for it, and on a shorter box leaves out only the quant part of the name.
func TestUXR10Title(t *testing.T) {
	for _, c := range [][2]string{
		{"Qwen3.5-122B-A10B Q4_K_M (execution scope, NextN excluded)", "Qwen3.5-122B-A10B"},
		{"gpt-oss-120b MXFP4", "gpt-oss-120b"},
		{"Qwen3.5-35B-A3B Q4_K_M", "Qwen3.5-35B-A3B"},
		{"Qwen3.5-397B-A17B Q4_K_M (6-shard)", "Qwen3.5-397B-A17B"},
		{"Kimi K2.6 447GB mixed-quant (deepseek2)", "Kimi K2.6"},
		{"DeepSeek-V4-Flash-0731 MXFP4/Q8_0 (284B, 43L x 256E top-6)", "DeepSeek-V4-Flash-0731"},
		{"gpt-oss-20b", "gpt-oss-20b"},
	} {
		if got := baseName(c[0], ""); got != c[1] {
			t.Fatalf("baseName(%q) = %q, want %q", c[0], got, c[1])
		}
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, c := range []struct{ name, quant, want string }{
			{"Qwen3.5-35B-A3B Q4_K_M", "Q4_K_M", "Qwen3.5-35B-A3B"}, // UX-R20: never the quant (answer 31 (c))
			{"DeepSeek-V4-Flash-0731 MXFP4/Q8_0 (284B, 43L x 256E top-6)", "MXFP4", "DeepSeek-V4-Flash-0731"},
		} {
			f := NewFake()
			f.Cat.Models[0].Name, f.Cat.Models[0].Quant = c.name, c.quant
			hs := newH(t, f, lang, 160, 40)
			T := hs.m.r.T
			mustHave(t, hs.plain(), "─ "+T("SETTINGS")+" · "+c.want+" · "+T("default")+" ─")
		}
	}
}

// TestUXR10Template (finding 18): a ◇ row is drawn as a catalog row, the key says what ◇
// is, the positive fact first; the 이 모델 box keeps the performance caveat; ✕ unsupported
// stays.
func TestUXR10Template(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		f := NewFake()
		f.Cat.Models[2].Fit = FitUnsupported
		hs := newH(t, f, lang, 160, 40)
		T := hs.m.r.T
		p := hs.plain()
		mustHave(t, p, "◇ = "+hs.m.r.Tf("not in catalog · %s compatible · runs", "gpt-oss"), "✕ "+T("unsupported"))
		if !regexp.MustCompile(`◇ gpt-oss-120b-l ○ \? +— +61 GB`).MatchString(p) { // the columns of a catalog row
			t.Fatalf("%s: the ◇ row lacks the decode and size columns:\n%s", lang, p)
		}
		mustLack(t, p, "arch defaults", "구조 기본값")
		hs.keys("down", "down", "down", "down")
		// the 이 모델 box, its line wrapped at the box's width: the first two pieces whole, then
		// each word of the caveat
		pieces := strings.Split(hs.m.r.Tf("outside the catalog · %s template settings · performance untested", "gpt-oss"), " · ")
		mustHave(t, hs.plain(), pieces[0]+" · "+pieces[1])
		for _, wd := range strings.Fields(pieces[2]) {
			mustHave(t, hs.plain(), wd)
		}
	}
}

// TestUXR10RAMShort (addendum): a need over the memory available now reads as a shortfall
// in UX-R3's words, with its baseline, not as a negative «left».
func TestUXR10RAMShort(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, w := range []int{160, 120} {
			f := uxr3Fake(true)
			f.MemIdle.ReadyNowGB = K(17.2) // the memory available now, as the preflight read it
			hs := newH(t, f, lang, w, 40)
			T := hs.m.r.T
			p := hs.plain()
			mustHave(t, p, hs.m.r.Tf("%s short", "2.6 GB"), "19.8 GB")
			if w == 160 || lang == LangKorean { // the baseline where the tile has room (English at 120: 24 cells in 23)
				mustHave(t, p, T("free")+" 17.2 · "+hs.m.r.Tf("%s short", "2.6 GB"))
			}
			mustLack(t, p, "-2.6", "−2.6", T("left after start")+" -")
		}
	}
}
