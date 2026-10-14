// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// FIX-UI: tests of the review items of reviews/_gol_review_8_user_20261010.md.

package ui

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

func init() { keyTypes["backspace"] = tea.KeyBackspace }

// cellIdx is the cell column where sub starts in a plain line (-1: absent).
func cellIdx(s, sub string) int {
	i := strings.Index(s, sub)
	if i < 0 {
		return -1
	}
	return lipgloss.Width(s[:i])
}

func mustHave(t *testing.T, p string, ws ...string) {
	t.Helper()
	for _, w := range ws {
		if !strings.Contains(p, w) {
			t.Fatalf("lacks %q:\n%s", w, p)
		}
	}
}

func mustLack(t *testing.T, p string, ws ...string) {
	t.Helper()
	for _, w := range ws {
		if strings.Contains(p, w) {
			t.Fatalf("has %q:\n%s", w, p)
		}
	}
}

// logBox is the log box of 4 Log: its rows between the top and bottom edge
// (the title edge left out, since its counts change).
func logBox(p string) []string {
	ls := strings.Split(p, "\n")
	top, bot := -1, -1
	for i, l := range ls {
		if top < 0 && strings.Contains(l, "╭") {
			top = i
		}
		if top >= 0 && strings.Contains(l, "╰") {
			bot = i
			break
		}
	}
	if top < 0 || bot < 0 {
		return nil
	}
	return ls[top+1 : bot]
}

func manyLogs(n int) []LogLine {
	t0 := time.Date(2026, 10, 10, 9, 0, 0, 0, time.Local)
	var ls []LogLine
	for i := 0; i < n; i++ {
		ls = append(ls, LogLine{At: t0.Add(time.Duration(i) * time.Second), Level: LogInfo, Text: fmt.Sprintf("line %02d", i)})
	}
	return ls
}

// item 1: while serving the list is locked, ⏎ never restarts, the guide
// line has the confirmed wording, ctrl+f unloads.
func TestItem1ListLockedNoRestart(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.want("H8")
	hs.keys("1", "down")
	if hs.m.sel != 0 || !strings.Contains(hs.m.notice, "locked") {
		t.Fatalf("the list moved while serving: sel %d notice %q", hs.m.sel, hs.m.notice)
	}
	hs.keys("enter")
	if hs.calls("Stop") != 0 || hs.calls("Start") != 1 {
		t.Fatalf("⏎ while serving: %v", f.Calls())
	}
	hs.keys("tab", "right") // a setting changes while serving: H2
	hs.want("H2")
	mustHave(t, hs.plain(), "not applied to the running server · applies at the next start")
	hs.keys("enter")
	if hs.calls("Stop") != 0 || hs.calls("Start") != 1 {
		t.Fatalf("⏎ on H2 restarted: %v", f.Calls())
	}
	hs.keys("ctrl+f")
	if hs.calls(fmt.Sprintf("Stop %d", StopUnload)) != 1 {
		t.Fatalf("ctrl+f: %v", f.Calls())
	}
	// Korean wording of the guide line
	f = NewFake()
	hs = newH(t, f, LangKorean, 120, 30)
	hs.keys("enter", "1", "tab", "right")
	mustHave(t, hs.plain(), "지금 서버에는 적용 안 됨 · 다음 시작 때 적용")
}

// item 2: «now» compares with the last actual start's values and the guide
// line shows only when a value differs from them.
func TestItem2NowMarksAgainstRun(t *testing.T) {
	f := NewFake()
	v := f.Views["qwen35-122b"]
	v.Rows[0].Value = "16 GB" // the saved default differs from what runs
	f.Views["qwen35-122b"] = v
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter", "1")
	hs.want("H2") // the run used 12 G, the screen holds 16 G
	mustHave(t, hs.plain(), "now 12 GB")
	hs.keys("tab", "left") // back to 12 G: the value the server runs with
	hs.want("H1")
	mustLack(t, hs.plain(), "now 12 GB", "not applied to the running server")
}

// item 3: ctrl+s saves the changed values through the backend.
func TestItem3Save(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("tab", "right")
	hs.keys("ctrl+s")
	if hs.calls("SavePreset qwen35-122b map[budget:16 GB]") != 1 {
		t.Fatalf("save: %v", f.Calls())
	}
	if len(hs.m.edits["qwen35-122b"]) != 0 || hs.m.views["qwen35-122b"].Rows[0].Value != "16 GB" {
		t.Fatalf("after save: edits %v view %v", hs.m.edits, hs.m.views["qwen35-122b"].Rows[0])
	}
	mustLack(t, hs.plain(), "was 12 GB")
}

// item 6: the held log view does not move when lines arrive; its state line
// says how far up and how many are new; no «following» line.
func TestItem6HeldViewStays(t *testing.T) {
	f := NewFake()
	f.Logs = manyLogs(40)
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("4")
	mustLack(t, hs.plain(), "following", "paused")
	hs.keys("up", "up", "up")
	before := logBox(hs.plain())
	f.Logs = append(f.Logs, manyLogs(43)[40:]...)
	hs.beat()
	after := logBox(hs.plain())
	if strings.Join(before, "\n") != strings.Join(after, "\n") {
		t.Fatalf("the held view moved:\n%s\n----\n%s", strings.Join(before, "\n"), strings.Join(after, "\n"))
	}
	mustHave(t, hs.plain(), "▲ 6 lines up · 3 new · End=newest")
	hs.keys("end")
	mustHave(t, hs.plain(), "line 42")
	mustLack(t, hs.plain(), "lines up")
	// Korean wording
	f = NewFake()
	f.Logs = manyLogs(40)
	hs = newH(t, f, LangKorean, 120, 30)
	hs.keys("4", "up")
	mustHave(t, hs.plain(), "▲ 최신에서 1줄 위 · 새 줄 0 · End=최신으로")
}

// item 7: a long line wraps at the window width; the time and the mark stay
// on the first line, the rest starts at the text column.
func TestItem7Wrap(t *testing.T) {
	f := NewFake()
	f.Logs = []LogLine{{At: time.Date(2026, 10, 10, 9, 0, 0, 0, time.Local), Level: LogWarn, Text: strings.Repeat("abcdefghij", 20) + "END"}}
	hs := newH(t, f, LangEnglish, 80, 28)
	hs.keys("4")
	box := logBox(hs.plain())
	if !strings.Contains(box[0], "09:00:00  ▲  abcdefghij") {
		t.Fatalf("first line: %q", box[0])
	}
	if !strings.HasPrefix(box[1], " │ "+strings.Repeat(" ", logTextCol)) || strings.HasPrefix(box[1], " │ "+strings.Repeat(" ", logTextCol+1)) {
		t.Fatalf("continuation not at the text column: %q", box[1])
	}
	mustHave(t, strings.Join(box, "\n"), "END")
	for _, l := range strings.Split(hs.plain(), "\n") {
		if lipgloss.Width(l) > 80 {
			t.Fatalf("line wider than the window: %q", l)
		}
	}
}

// items 18 and 29: tabs «all n │ warnings n │ errors n», Tab switches them,
// ←→ still turn places, f does nothing; notices (cyan i) only in all.
func TestItem18Tabs(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("4")
	mustHave(t, hs.plain(), "all 4 │ warnings 1 │ errors 0", "i  notice  [moe-direct]")
	hs.keys("f")
	if hs.m.logFilter != 0 {
		t.Fatalf("f still filters")
	}
	hs.keys("tab")
	if hs.m.logFilter != 1 {
		t.Fatalf("tab: filter %d", hs.m.logFilter)
	}
	p := hs.plain()
	mustHave(t, p, "RAM low")
	mustLack(t, p, "notice", "catalog read")
	hs.keys("tab")
	mustLack(t, hs.plain(), "RAM low")
	hs.keys("tab")
	if hs.m.logFilter != 0 {
		t.Fatalf("tab does not go round")
	}
	hs.keys("right")
	hs.want("H5")
	hs.keys("left")
	hs.want("H11")
}

// item 20: the box keeps its height on every tab and while scrolling a
// short list; the scroll stops at max(0, lines − rows).
func TestItem20BoxHeightAndScrollLimit(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("4")
	h0 := len(logBox(hs.plain()))
	hs.keys("tab") // warnings: one line
	for i := 0; i < 5; i++ {
		hs.keys("up")
		if n := len(logBox(hs.plain())); n != h0 {
			t.Fatalf("box height %d, was %d", n, h0)
		}
	}
	if hs.m.logBack != 0 {
		t.Fatalf("a list that fits scrolled: %d", hs.m.logBack)
	}
	f = NewFake()
	f.Logs = manyLogs(40)
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("4")
	for i := 0; i < 60; i++ {
		hs.keys("up")
	}
	if want := 40 - hs.m.logRoom(); hs.m.logBack != want {
		t.Fatalf("scroll limit %d, want %d", hs.m.logBack, want)
	}
	mustHave(t, hs.plain(), "line 00")
}

// item 21: the graph's columns are time spans between the end labels, with
// a scale from 0 to the top and «average · lowest».
func TestItem21GraphBins(t *testing.T) {
	t0 := time.Date(2026, 10, 10, 9, 0, 0, 0, time.UTC)
	ss := []sample{{t0.Add(30 * time.Second), 2}, {t0.Add(45 * time.Second), 4}, {t0.Add(150 * time.Second), 6}}
	b := bins(ss, t0, t0.Add(4*time.Minute), 4)
	if b[0] != 3 || !math.IsNaN(b[1]) || b[2] != 6 || !math.IsNaN(b[3]) {
		t.Fatalf("bins %v", b)
	}
	// ten minutes: five at 10 tok/s, then five at 2; the left half of the
	// columns must be the high ones (the old graph showed the last w samples)
	var long []sample
	for i := 0; i < 600; i++ {
		v := 10.0
		if i >= 300 {
			v = 2
		}
		long = append(long, sample{t0.Add(time.Duration(i) * time.Second), v})
	}
	r := rc{W: 120, H: 30, lang: LangEnglish}
	g := r.timeGraph(long, t0, t0.Add(599*time.Second), 54, gr, "09:00", "now")
	top := ansi.Strip(g[0])
	if !strings.HasPrefix(top, "10.0┤") {
		t.Fatalf("scale top %q", top)
	}
	cols := []rune(strings.TrimPrefix(top, "10.0┤"))
	if len(cols) != 54-5 {
		t.Fatalf("%d columns", len(cols))
	}
	if cols[0] != '█' || cols[len(cols)/2-2] != '█' || cols[len(cols)-1] == '█' {
		t.Fatalf("columns do not follow time: %q", string(cols))
	}
	if !strings.HasPrefix(ansi.Strip(g[2]), "   0┤") {
		t.Fatalf("scale bottom %q", ansi.Strip(g[2]))
	}
	mustHave(t, ansi.Strip(g[3]), "09:00", "average 6.0 · lowest 2.0 tok/s", "now")
}

// item 25: the mark under the lit place starts at its pill, wide and narrow.
func TestItem25Underline(t *testing.T) {
	for _, w := range []int{120, 80} {
		for i, p := range places {
			hs := newH(t, NewFake(), LangEnglish, w, 30)
			hs.keys(p[0])
			ls := strings.Split(hs.plain(), "\n")
			pillAt := cellIdx(ls[0], p[0]+" "+p[1]) - 1
			markAt := cellIdx(ls[1], "━")
			if pillAt < 0 || pillAt != markAt {
				t.Fatalf("width %d place %d: pill at %d, mark at %d\n%s\n%s", w, i+1, pillAt, markAt, ls[0], ls[1])
			}
		}
	}
}

// item 8: ↓ from the last folder goes on into the launcher rows and ↑ back;
// ←→ on a folder row does nothing; item 24: the writes section by Tab, ⏎
// opens a folder or shows a file, a row without a path does nothing.
func TestItem8And24Settings(t *testing.T) {
	defer func(v bool) { showNotPorted = v }(showNotPorted)
	showNotPorted = true // UX-R4: every row shown, as this walk was written; the hidden rows: TestUXR4NotPortedHidden
	f := NewFake()
	f.Cat.Folders = []Folder{{Path: `D:\moe-models`, State: FolderWatching, Found: 3}, {Path: `E:\models`, State: FolderEmpty}}
	f.Info.LogDir = `C:\Users\tmxkz\AppData\Local\MoE-Direct\logs`
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("5", "down", "down")
	if hs.m.set5Sec != secFolders || hs.m.set5Row != 1 {
		t.Fatalf("folders: sec %d row %d", hs.m.set5Sec, hs.m.set5Row)
	}
	hs.keys("left", "right")
	hs.want("H5")
	if hs.calls("Action") != 0 {
		t.Fatalf("←→ on a folder row acted: %v", f.Calls())
	}
	hs.keys("down")
	if hs.m.set5Sec != secLauncher || hs.m.set5Row != 0 {
		t.Fatalf("↓ past the last folder: sec %d row %d", hs.m.set5Sec, hs.m.set5Row)
	}
	hs.keys("up")
	if hs.m.set5Sec != secFolders || hs.m.set5Row != 1 {
		t.Fatalf("↑ from the first launcher row: sec %d row %d", hs.m.set5Sec, hs.m.set5Row)
	}
	hs.keys("tab", "tab") // folders → launcher → writes
	if hs.m.set5Sec != secWrites {
		t.Fatalf("tab: sec %d", hs.m.set5Sec)
	}
	hs.keys("down", "enter") // setup: no path
	if hs.calls("Action") != 0 {
		t.Fatalf("a row without a path acted: %v", f.Calls())
	}
	hs.keys("down", "enter")
	if hs.calls(`Action open-folder C:\Users\tmxkz\AppData\Local\MoE-Direct\logs`) != 1 {
		t.Fatalf("logs folder: %v", f.Calls())
	}
	hs.keys("down", "enter")
	if hs.calls(`Action show-file C:\state\logs\launcher.log`) != 1 {
		t.Fatalf("log file: %v", f.Calls())
	}
	if got := midPath(`C:\Users\tmxkz\AppData\Local\MoE-Direct\logs`, 30); got != `C:\Users\tmxkz\…\logs` {
		t.Fatalf("midPath %q", got)
	}
	// a log folder longer than the narrow column (the state root under a longer account name),
	// as the go-draft path was before GL-FX2
	f.Info.LogDir = `C:\Users\launcher-test-account\AppData\Local\MoE-Direct\logs`
	hs = newH(t, f, LangEnglish, 64, 40) // narrow, and tall enough to show the section
	hs.keys("5")
	mustHave(t, hs.plain(), `\…\`)
}

// item 30: Space opens a box; while it is open digits and q are typed, not
// place jumps or a quit; Esc keeps the value; ⏎ checks the range through
// the backend and refuses with its reason; «64k» reads as 65536.
func TestItem30InputBox(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("tab")
	mustHave(t, hs.plain(), "space type a value")
	hs.keys(" ")
	if !hs.m.inputOn || hs.m.inputBuf != "12" {
		t.Fatalf("box: on %v buf %q", hs.m.inputOn, hs.m.inputBuf)
	}
	// a narrow panel: the range on the panel's last line, the keys in the footer
	mustHave(t, hs.plain(), "▶· RAM budget [ 12_ ] GB", "│ 8 GB ~ 20 GB ", "⏎ apply   esc cancel") // UX-R10: the state mark after ▶
	hs.keys("backspace", "backspace", "2", "5", "3", "q")
	hs.want("H1")
	if hs.m.inputBuf != "253q" || hs.m.quitArmed {
		t.Fatalf("typed: buf %q quit armed %v", hs.m.inputBuf, hs.m.quitArmed)
	}
	hs.keys("backspace", "backspace", "enter")
	if !hs.m.inputOn || !strings.Contains(hs.m.inputErr, "outside") {
		t.Fatalf("out of range: on %v err %q", hs.m.inputOn, hs.m.inputErr)
	}
	mustHave(t, hs.plain(), "✕ 25 is outside 8 ~ 20")
	if _, ok := hs.m.edits["qwen35-122b"]["budget"]; ok {
		t.Fatalf("a refused value was applied")
	}
	hs.keys("esc")
	if hs.m.inputOn || len(hs.m.edits["qwen35-122b"]) != 0 {
		t.Fatalf("esc: on %v edits %v", hs.m.inputOn, hs.m.edits)
	}
	pre := hs.calls("Preflight")
	hs.keys(" ", "backspace", "backspace", "1", "4", ".", "5", "enter")
	if v := hs.m.edits["qwen35-122b"]["budget"]; v != "14.5 GB" || hs.m.inputOn {
		t.Fatalf("in range: edit %q on %v", v, hs.m.inputOn)
	}
	if hs.calls("Preflight") != pre+1 {
		t.Fatalf("no preflight after a typed value: %v", f.Calls())
	}
	hs.keys("down", " ")
	for range "12288" {
		hs.keys("backspace")
	}
	hs.keys("6", "4", "k", "enter")
	if v := hs.m.edits["qwen35-122b"]["ctx"]; v != "65536" {
		t.Fatalf("64k: %q", v)
	}
	// a locked or untyped row opens no box
	hs.keys("a")
	for hs.m.setCur < 4 {
		hs.keys("down")
	}
	hs.keys(" ")
	if hs.m.inputOn {
		t.Fatalf("a locked row opened the box")
	}
	// Korean tail line
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.keys("tab")
	mustHave(t, hs.plain(), "space 직접 입력")
}

// item 22: the one-line verdict of 1 Models, with live numbers.
func TestItem22VerdictLine(t *testing.T) {
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	mustHave(t, hs.plain(), "✓ 5 GB to spare · plenty of room for other work alongside the model :>")
	f := NewFake()
	f.MemIdle.ReadyNowGB = K(21) // 21 − 19 = 2 G left: under the 4 G threshold
	hs = newH(t, f, LangEnglish, 120, 30)
	mustHave(t, hs.plain(), "✓ can start · 2.0 GB RAM for other work")
	f = NewFake()
	f.Pre.Refused, f.Pre.Verdict, f.Pre.Refusals = true, VerdictCannot, []string{"fail_resource (exit 3)"}
	f.MemIdle.ReadyNowGB = K(6.9)
	hs = newH(t, f, LangEnglish, 120, 30)
	p := hs.plain()
	mustHave(t, p, "✕ cannot start · RAM budget 12.0 GB > free now 6.9 GB · details on 3 Memory")
	mustLack(t, p, "⏎ START", "fail_resource")
	f = NewFake()
	f.Pre.Verdict = VerdictTight
	hs = newH(t, f, LangKorean, 120, 30)
	mustHave(t, hs.plain(), "▲ 빠듯함 · 시작 뒤 남는 RAM 5.6 GB · 자세히 3 Memory")
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	mustHave(t, hs.plain(), "✓ 5 GB만큼 여유롭네요 · 모델과 함께 다양한 작업을 하기에 충분하겠군요 :>")
	hs.keys("3")
	mustHave(t, hs.plain(), "예산↑=캐시 적중↑")
}

// items 4, 9, 14, 15, 16, 17, 19, 27, 28: the serve page and the tiles.
func TestServePageItems(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.beat()
	p := hs.plain()
	// 16, 17, 27: one frame, the API name after the address, plain words
	mustHave(t, p, "APPLIED SETTINGS, change in Models", "API name qwen3.5-122b", "12k context · 12 GB cache · virtual · QD 8")
	if strings.Count(p, "change in Models") != 1 {
		t.Fatalf("a floating «change in Models» is left:\n%s", p)
	}
	// 15: the footer's c only
	if strings.Count(p, "c copy") != 1 {
		t.Fatalf("«c copy» twice:\n%s", p)
	}
	hs.keys("c")
	if hs.calls("Action copy-address http://127.0.0.1:8093/v1") != 1 {
		t.Fatalf("copy: %v", f.Calls())
	}
	// 14: three lines, RAM and VRAM each with what is left, the colour key
	mustHave(t, p, "4.7 GB left", "2.6 GB left", "this server ▓▓ other") // UX-R16b: RAM and VRAM colour in the key
	// 4: the card's use and this server apart
	hs.keys("3")
	mustHave(t, hs.plain(), "█ this server 12.9 GB + ▓ other programs 0.5 GB = in use 13.4 GB") // UX-R22: the VRAM key, in use as the sum's label
	// 28: 1 Models tok/s is live while the model serves
	hs.keys("1")
	mustHave(t, hs.plain(), "live")
	// 9: prefill rate and TTFT from the backend (UX-R27: in the PREFILL panel, now column; the
	// progress count went with the request box, the hero tile keeps the percentage)
	f = NewFake()
	f.Statuses[0].Phase, f.Statuses[0].PrefillDone, f.Statuses[0].PrefillTotal, f.Statuses[0].PrefillTokS = ReqPrefill, K(3000), K(6000), K(300)
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.beat()
	mustHave(t, strings.Join(strings.Fields(hs.plain()), " "), "PREFILL · last 60 min", "prefill 300 — — — tok/s", "TTFT ~10 — — — s", "TTFT ~10 s")
}

// items 10, 11, 12, 19, 28: the list and THIS MODEL.
func TestModelListItems(t *testing.T) {
	f := NewFake()
	f.Cat.Models[2].Fit = FitBroken
	hs := newH(t, f, LangEnglish, 120, 30)
	p := hs.plain()
	mustHave(t, p, "★ Qwen3.5-12", "    gpt-oss-20", "✕ broken", "file check ✓ same as the catalog", "last run · 10-09")
	mustLack(t, p, "the folder is set in", " id ")
	f.Cat.Models[2].Fit = FitUnsupported
	hs = newH(t, f, LangKorean, 120, 30)
	mustHave(t, hs.plain(), "✕ unsupported", "파일 확인")
	hs.keys("3")
	mustHave(t, hs.plain(), "sha256 3f9a…c21e")
}

// the states this unit adds render at every test size in both languages,
// never wider than the window: the input box with and without a refusal,
// the held log on each tab, the writes section, the serve page with samples.
func TestNewStatesRender(t *testing.T) {
	seqs := map[string][]string{
		"input":    {"tab", " ", "9"},
		"refused":  {"tab", " ", "backspace", "backspace", "9", "9", "enter"},
		"logheld":  {"4", "up", "tab"},
		"logerr":   {"4", "tab", "tab"},
		"writes":   {"5", "tab", "tab", "down"},
		"launcher": {"5", "down", "down"},
		"serving":  {"enter"},
		"h2":       {"enter", "1", "tab", "right"},
	}
	for name, seq := range seqs {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				f := NewFake()
				f.Logs = append(f.Logs, LogLine{At: time.Now(), Level: LogError, Text: strings.Repeat("a long error line ", 12)})
				hs := newH(t, f, lang, sz[0], sz[1])
				hs.keys(seq...)
				hs.beat()
				for _, l := range strings.Split(hs.plain(), "\n") {
					if lipgloss.Width(l) > sz[0] {
						t.Fatalf("%s %s %dx%d: line wider than the window: %q", name, lang, sz[0], sz[1], l)
					}
				}
			}
		}
	}
}
