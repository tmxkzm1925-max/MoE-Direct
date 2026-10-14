// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// FIX2-UI: tests of the conflicts of reviews/_gol_conflicts_8_user_20261010.md.

package ui

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// conflict 7: the refusal line carries the budget cap that passes, on
// 1 Models and 3 Memory; no «start anyway» anywhere.
func TestConflict7BudgetCap(t *testing.T) {
	f := NewFake()
	f.Pre.Refused, f.Pre.Verdict, f.Pre.BudgetCapGB = true, VerdictCannot, K(5)
	f.MemIdle.ReadyNowGB = K(6.9)
	hs := newH(t, f, LangEnglish, 120, 30)
	mustHave(t, hs.plain(), "✕ cannot start · RAM budget 12.0 GB > free now 6.9 GB · a budget of 5.0 GB or less starts it · details on 3 Memory")
	hs.keys("3")
	mustHave(t, hs.plain(), "→ a budget of 5.0 GB or less starts it")
	hs = newH(t, f, LangKorean, 120, 30)
	mustHave(t, hs.plain(), "예산을 5.0 GB 이하로 내리면 시작 가능")
	f = NewFake()
	f.Pre.Verdict = VerdictTight
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("3")
	mustLack(t, hs.plain(), "anyway")
}

// conflict 6: the launcher's stop line in 4 Log, in three looks by its code.
func TestConflict6StopLines(t *testing.T) {
	f := NewFake()
	t0 := time.Date(2026, 10, 10, 11, 0, 0, 0, time.Local)
	f.Logs = []LogLine{
		{At: t0, Stop: &StopRecord{Code: 0, ServerCode: K(0)}},
		{At: t0.Add(time.Minute), Stop: &StopRecord{Code: 5, ServerCode: K(1)}},
		{At: t0.Add(2 * time.Minute), Stop: &StopRecord{Code: 2}},
		{At: t0.Add(3 * time.Minute), Stop: &StopRecord{Code: 3, Meaning: "the server did not start", ServerCode: K(3221225477)}},
	}
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("4")
	p := hs.plain()
	mustHave(t, p,
		"11:00:00  ✓  server stopped · code 0 · normal stop · server exit 0",
		"11:01:00  ▲  server stopped · code 5 · forced stop · server exit 1",
		"11:02:00  ▲  server stopped · code 2 · stopped before ready",
		"11:03:00  ✕  server stopped · code 3 · the server did not start · server exit 3221225477",
		"all 4 │ warnings 2 │ errors 1")
	for i, want := range []LogLevel{LogOK, LogWarn, LogWarn, LogError} {
		if got := logLevel(f.Logs[i]); got != want {
			t.Fatalf("line %d: class %d, want %d", i, got, want)
		}
	}
	hs.keys("tab", "tab")
	mustHave(t, hs.plain(), "code 3")
	mustLack(t, hs.plain(), "code 5")
	// a stop through the fake writes its line
	f = NewFake()
	hs = newH(t, f, LangKorean, 120, 30)
	hs.keys("enter", "ctrl+f", "4")
	mustHave(t, hs.plain(), "서버 끔 · 코드 0 · 정상 · 서버 종료 코드 0")
}

// conflict 9: ⏎ on a model that needs a setup opens the plan box on
// 2 Server; esc goes back with nothing started; ⏎ goes on to the setup
// progress; a drive too small shows the refusal line instead of the box.
func TestConflict9PlanBox(t *testing.T) {
	f := NewFake()
	f.StartHold = true // the setup goes on while the test looks
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("down", "down", "down") // Qwen3-30B-A3B: virtual needs repack
	mustHave(t, hs.plain(), "⏎ SETUP PLAN")
	hs.keys("enter")
	hs.want("HP")
	p := hs.plain()
	mustHave(t, p, "SETUP NEEDED · Qwen3-30B-A3B · virtual", "what to do", "writes to", // UX-R9: «what to do» for «does» "D:\\moe-models\\repack\\qwen3-30b\\virtual",
		"about 17.5 GB written · drive free 412.0 GB → 394.5 GB left", "first repack: not known yet",
		"then: check → speed check → the server starts by itself")
	if hs.calls("RepackPlan qwen3-30b") != 1 || hs.calls("Repack ") != 0 || hs.calls("Start") != 0 {
		t.Fatalf("calls: %v", f.Calls())
	}
	hs.keys("esc")
	hs.want("H1")
	if hs.m.plan != nil || hs.calls("Repack ") != 0 || hs.calls("Start") != 0 {
		t.Fatalf("esc started something: %v", f.Calls())
	}
	hs.keys("enter", "enter")
	if hs.calls("Repack qwen3-30b") != 1 {
		t.Fatalf("⏎ in the box: %v", f.Calls())
	}
	hs.want("H6") // the first-run setup progress
	// the last time, when known
	f = NewFake()
	f.Plan.LastS = K(754)
	hs = newH(t, f, LangKorean, 120, 30)
	hs.keys("down", "down", "down", "enter")
	mustHave(t, hs.plain(), "준비 필요 · Qwen3-30B-A3B · virtual", "지난번 12m34s", "이후 단계: 자동 확인 검증 → 디스크 읽기 속도 확인 → 서버 자동 시작") // UX-R9 (finding 14)
	// a drive too small: the refusal line, no box, ⏎ does nothing
	f = NewFake()
	f.Plan.WriteGB, f.Plan.LeftGB = K(500), K(-88)
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("down", "down", "down", "enter")
	p = hs.plain()
	mustHave(t, p, "✕ cannot start · free 412.0 GB < needed 500.0 GB")
	mustLack(t, p, "SETUP NEEDED")
	hs.keys("enter")
	if hs.calls("Repack ") != 0 {
		t.Fatalf("a short drive started the setup: %v", f.Calls())
	}
	// conflict 16: the mode now set failed its check: ⏎ shows the plan too
	f = NewFake()
	f.Cat.Models[0].Modes = []ModeCheck{{"packed", ModeReady}, {"virtual", ModeCheckFailed}}
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	if hs.calls("RepackPlan qwen35-122b") != 1 || hs.calls("Start") != 0 {
		t.Fatalf("a failed check did not lead to the plan: %v", f.Calls())
	}
}

// conflict 16: both modes in THIS MODEL's setup line, the box height kept.
func TestConflict16SetupLine(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 160, 30)
	mustHave(t, hs.plain(), "setup      packed ✓ · virtual ✓")
	h0 := strings.Count(hs.plain(), "\n")
	hs.keys("down", "down", "down")
	mustHave(t, hs.plain(), "setup      packed ✓ · virtual needs repack")
	if strings.Count(hs.plain(), "\n") != h0 {
		t.Fatalf("the screen height changed")
	}
	f.Cat.Models[3].Modes = []ModeCheck{{"packed", ModeCheckFailed}, {"virtual", ModeReady}}
	hs = newH(t, f, LangEnglish, 160, 30)
	hs.keys("down", "down", "down")
	mustHave(t, hs.plain(), "setup      packed ✕ check failed · virtual ✓")
	f.Cat.Models[3].Modes = []ModeCheck{{"packed", ModeCheckFailed}, {"virtual", ModeNeedsRepack}}
	hs = newH(t, f, LangEnglish, 120, 30) // narrow: the short marks
	hs.keys("down", "down", "down")
	mustHave(t, hs.plain(), "setup      packed ✕ · virtual ▲")
	hs = newH(t, f, LangKorean, 160, 30)
	hs.keys("down", "down", "down")
	mustHave(t, hs.plain(), "packed ✕ 점검 실패 · virtual 리팩 필요")
}

// conflict 18-1: the one «check run» row in 5 Settings, ←→ through the
// backend; its failure on the start result with the user's sentence.
func TestConflict18CheckRun(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("5")
	mustHave(t, hs.plain(), "check run      off")
	mustLack(t, hs.plain(), "network")
	hs.keys("tab", "down", "down", "down")
	if hs.m.set5Sec != secLauncher || hs.m.set5Row != set5SmokeRow {
		t.Fatalf("row: sec %d row %d", hs.m.set5Sec, hs.m.set5Row)
	}
	hs.keys("right")
	if hs.calls("Action set-smoke on") != 1 {
		t.Fatalf("set: %v", f.Calls())
	}
	mustHave(t, hs.plain(), "◀ on ▶")
	hs.keys("right")
	mustHave(t, hs.plain(), "◀ once per model ▶")
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.keys("5")
	mustHave(t, hs.plain(), "시작 후 응답 시험", "끔") // UX-R4: «작동 확인» renamed
	// the failure, only where the start ended
	hs = newH(t, NewFake(), LangEnglish, 120, 30)
	hs.m.failure = &Failure{What: "the check run failed", Why: "no answer to «ping» within 30 s", SmokeFailed: true}
	hs.m.ph = phFailed
	hs.keys("2")
	mustHave(t, hs.plain(), "✕ check run failed · no answer to «ping» within 30 s · details on 4 Log")
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.m.failure = &Failure{Why: "응답 없음", SmokeFailed: true}
	hs.m.ph = phFailed
	hs.keys("2")
	mustHave(t, hs.plain(), "✕ 시작 후 응답 시험 실패 · 응답 없음 · 자세히 4 Log")
}

// conflict 8: the out-of-memory load failure in the user's sentence.
func TestConflict8VRAMOut(t *testing.T) {
	hs := newH(t, NewFake(), LangKorean, 120, 30)
	hs.m.failure = &Failure{What: "load failed", Why: "cudaMalloc failed: out of memory", VRAMOut: true}
	hs.m.ph = phFailed
	hs.keys("2")
	mustHave(t, hs.plain(), "✕ VRAM 부족으로 시작 실패 · 다른 GPU 프로그램을 끄거나 ctx 를 줄여 보세요 · 자세히 4 Log")
}

// conflict 18-2: a start waits for the launcher's file check; a damaged
// bundle refuses with the user's line; a development build says so.
func TestConflict18Bundle(t *testing.T) {
	f := NewFake()
	f.Bundles = []BundleCheck{{State: BundleChecking}, {State: BundleChecking}, {State: BundleOK}}
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	if hs.calls("Start") != 0 || hs.m.startWait != "qwen35-122b" {
		t.Fatalf("started before the check: %v", f.Calls())
	}
	mustHave(t, hs.plain(), "checking the launcher files … the start follows")
	hs.advance(time.Second)
	hs.beat() // still checking
	hs.advance(time.Second)
	hs.beat() // done: the start goes on
	if hs.calls("Start") != 1 {
		t.Fatalf("the start did not follow the check: %v", f.Calls())
	}
	f = NewFake()
	f.Bundles = []BundleCheck{{State: BundleBroken, Mismatches: 3}}
	hs = newH(t, f, LangEnglish, 120, 30)
	p := hs.plain()
	mustHave(t, p, "✕ the launcher files are damaged · 3 differ · install again")
	mustLack(t, p, "⏎ START")
	hs.keys("enter")
	if hs.calls("Start") != 0 || hs.calls("Preflight qwen35-122b map[] true") != 0 {
		t.Fatalf("a damaged bundle started: %v", f.Calls())
	}
	hs = newH(t, f, LangKorean, 120, 30)
	mustHave(t, hs.plain(), "✕ 배포 파일 손상 · 3개 불일치 · 다시 설치")
	f = NewFake()
	f.Bundles = []BundleCheck{{State: BundleDev}}
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("5")
	mustHave(t, hs.plain(), "development build · launcher file check skipped")
	hs.keys("1")
	mustLack(t, hs.plain(), "development build")
}

// conflict 3: the provenance label in the settings panel's title and at the
// end of «this run»'s settings line; no label in the list, and since UX-R2
// (review item 16(a)) none on the start line, which names the model alone.
func TestConflict3Labels(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	p := hs.plain()
	mustHave(t, p, "SETTINGS · Qwen3.5-122B · default", "⏎ START  Qwen3.5-122B")
	mustLack(t, p, "queue 8 · default")
	// UX-R20: the title keeps the model's name; where it is narrow «· not saved» leaves the
	// title first (the ◆ line in the box says it), then the name is cut before the label
	hs.keys("tab", "right")
	mustHave(t, hs.plain(), "─ SETTINGS · Qwen3.5-122B · default ─", "◆ not saved")
	hs.keys("ctrl+s")
	mustHave(t, hs.plain(), "─ SETTINGS · Qwen3.5… · my settings ─")
	mustLack(t, hs.plain(), "not saved")
	hs.keys("right")
	mustHave(t, hs.plain(), "─ SETTINGS · Qwen3.5… · my settings ─", "◆ not saved") // a narrow title keeps the label whole
	if strings.Count(hs.plain(), "my settings") != 1 {                              // the title, not the list or the start line (UX-R2)
		t.Fatalf("label count:\n%s", hs.plain())
	}
	// the run keeps the label of its start, the changed value after it
	f = NewFake()
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("tab", "right", "enter")
	hs.beat()
	hs.want("H8")
	mustHave(t, hs.plain(), "prefetch catalog · default · not saved (RAM budget 16 GB)")
	// at 80 the line wraps by its pieces and the label stays whole
	hs = newH(t, NewFake(), LangEnglish, 80, 28)
	hs.keys("tab", "right", "enter")
	hs.beat()
	mustHave(t, hs.plain(), "│           default · not saved (RAM budget 16 GB)")
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.keys("tab", "right")
	mustHave(t, hs.plain(), "기본값 · 저장 안 됨")
}

// conflict 10: the yellow line under the settings while a value differs
// from its default, also for a saved profile; the speed tile «not
// measured» for settings never run.
func TestConflict10Custom(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	mustLack(t, hs.plain(), "custom settings")
	mustHave(t, hs.plain(), "last run · 10-09")
	hs.keys("tab", "right")
	mustHave(t, hs.plain(), "▲ custom settings · performance unverified", "not measured with these settings")
	hs.keys("left")
	mustLack(t, hs.plain(), "custom settings", "not measured with these settings")
	hs.keys("right", "ctrl+s")
	mustHave(t, hs.plain(), "▲ custom settings · performance unverified")
	// the last run's values speak for themselves
	f = NewFake()
	v := f.Views["qwen35-122b"]
	v.LastRun = Settings{"budget": "16 GB"}
	f.Views["qwen35-122b"] = v
	hs = newH(t, f, LangEnglish, 120, 30)
	mustHave(t, hs.plain(), "not measured with these settings")
	hs.keys("tab", "right")
	mustHave(t, hs.plain(), "last run · 10-09", "▲ custom settings")
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.keys("tab", "right")
	mustHave(t, hs.plain(), "▲ 커스텀 설정 · 이 설정의 성능은 미확인", "측정 전")
}

// conflict 14: a model outside the catalog with its structure's defaults; since UX-R10
// (finding 18) its row is drawn as a catalog row, the key says what ◇ is, the positive fact
// first, and the 이 모델 box keeps the performance caveat.
func TestConflict14Template(t *testing.T) {
	hs := newH(t, NewFake(), LangEnglish, 160, 30)
	mustHave(t, hs.plain(), "◇ gpt-oss-120b-l ○ ?", "◇ = not in catalog · gpt-oss compatible · runs")
	mustLack(t, hs.plain(), "arch defaults")
	hs = newH(t, NewFake(), LangEnglish, 120, 30) // the key wraps under its sign
	mustHave(t, hs.plain(), "◇ gpt-oss-12", "◇ = not in catalog · gpt-oss", "    compatible · runs")
	hs.keys("down", "down", "down", "down")
	mustHave(t, hs.plain(), "outside the catalog · gpt-oss", "template settings · performance", "untested")

	hs = newH(t, NewFake(), LangKorean, 160, 30)
	mustHave(t, hs.plain(), "◇ = 카탈로그 밖 · gpt-oss 호환 · 실행 가능")
}

// conflict 2: prefetch and the repack mode change with ←→; KV type and
// -np 1 show locked; no network row anywhere (conflict 17).
func TestConflict2Rows(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 40)
	hs.keys("tab", "a")
	for hs.m.setCur < 4 {
		hs.keys("down")
	}
	hs.keys("right") // prefetch
	// GL-19: the options are catalog and init (F-PF §2.2)
	if v := hs.m.edits["qwen35-122b"]["prefetch"]; v != "init" {
		t.Fatalf("prefetch: %q", v)
	}
	hs.keys("down", "left") // repack mode
	if v := hs.m.edits["qwen35-122b"]["mode"]; v != "packed" {
		t.Fatalf("mode: %q", v)
	}
	p := hs.plain()
	mustHave(t, p, "KV type    f16  ■ locked", "parallel   1  ■ locked")
	mustLack(t, p, "network")
}

// conflict 5: the help's X line; FIX-WIRE section 4 item 4: no «START LAST
// TIME», no «prefill not ported», the skipped box titled by what it lists.
func TestConflict5AndWireTexts(t *testing.T) {
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	hs.keys("?")
	if !strings.Contains(flat(hs.plain()), flat("X closing the window stops the server at once · nothing saved (same as ctrl+c)")) {
		t.Fatalf("the X line:\n%s", hs.plain())
	}
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.keys("?")
	if !strings.Contains(flat(hs.plain()), flat("창 닫기=서버도 바로 꺼짐 · 저장 없음(ctrl+c 와 같음)")) {
		t.Fatalf("the Korean X line:\n%s", hs.plain())
	}
	f := NewFake()
	f.Cat.Models[0].LastRun = "10-09 21:14 · 5.5 tok/s"
	hs = newH(t, f, LangEnglish, 120, 30)
	p := hs.plain()
	mustLack(t, p, "LAST TIME", "prefill    not ported")
	mustHave(t, p, "⏎ START  Qwen3.5-122B")
	f = NewFake()
	f.Cat.Folders = []Folder{{Path: `D:\moe-models`, State: FolderWatching, Found: 3, Skipped: []string{"notes.gguf  not a model file"}}}
	hs = newH(t, f, LangEnglish, 120, 40)
	hs.keys("5")
	mustHave(t, hs.plain(), `UNSUPPORTED GGUF · SKIPPED · D:\moe-models`) // UX-R4 (finding 8b)
	mustLack(t, hs.plain(), "HAS NONE")
	f.Cat.Folders[0].Found = 0
	hs = newH(t, f, LangEnglish, 120, 40)
	hs.keys("5")
	mustHave(t, hs.plain(), "HAS NONE")
}

// the states this unit adds render at every test size in both languages,
// never wider than the window.
func TestFix2StatesRender(t *testing.T) {
	seqs := map[string]func(*FakeBackend) []string{
		"plan": func(f *FakeBackend) []string { return []string{"down", "down", "down", "enter"} },
		"short": func(f *FakeBackend) []string {
			f.Plan.WriteGB = K(900)
			return []string{"down", "down", "down", "enter"}
		},
		"setup":   func(f *FakeBackend) []string { return []string{"down", "down", "down", "enter", "enter"} },
		"stoplog": func(f *FakeBackend) []string { return []string{"enter", "ctrl+f", "4"} },
		"custom":  func(f *FakeBackend) []string { return []string{"tab", "right"} },
		"bundle": func(f *FakeBackend) []string {
			f.Bundles = []BundleCheck{{State: BundleBroken, Mismatches: 12}}
			return nil
		},
		"cap": func(f *FakeBackend) []string {
			f.Pre.Refused, f.Pre.Verdict, f.Pre.BudgetCapGB = true, VerdictCannot, K(5)
			return []string{"3"}
		},
		"template": func(f *FakeBackend) []string { return []string{"down", "down", "down", "down"} },
		"settings": func(f *FakeBackend) []string {
			f.Bundles = []BundleCheck{{State: BundleDev}}
			return []string{"5", "tab", "down", "down", "down"}
		},
	}
	for name, seq := range seqs {
		for _, lang := range []string{LangEnglish, LangKorean} {
			for _, sz := range sizes {
				f := NewFake()
				ks := seq(f)
				hs := newH(t, f, lang, sz[0], sz[1])
				hs.keys(ks...)
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
