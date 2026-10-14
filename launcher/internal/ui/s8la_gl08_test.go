package ui

// GL-08 (version 8): the screens' part of S8-LA-18, S8-LA-19 and S8-LA-21 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition
// reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, sections 4.2, 4.3, 6.1, 9.2, 11.1,
// 11.2 and 11.7), over the fake backend.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// S8-LA-18: W7 duty 7's line at the place of the start verdict, W7 duty 34's page, and no
// action that ends another process.
func TestS8LA18(t *testing.T) {
	for lang, line := range map[string]string{LangEnglish: "cannot start · another launcher is already running",
		LangKorean: "다른 런처가 이미 실행 중"} {
		f := NewFake()
		f.Pre.Refused = true
		f.Pre.Verdict = VerdictCannot
		f.Pre.Refusals = []string{"another launcher is already running"} // wire.LauncherLockLine
		hs := newH(t, f, lang, 120, 30)
		hs.want("H1")
		mustHave(t, flat(hs.plain()), flat(line))
		hs.keys("enter", "enter")
		if hs.calls("Start") != 0 {
			t.Fatalf("%s: a second launcher started a server: %v", lang, f.Calls())
		}
	}

	// A server that ends while serving: the page with its exit code (W7 duty 34).
	f := NewFake()
	serving := f.Statuses[0]
	ended := serving
	ended.Exited, ended.ExitCode = true, 5
	f.Statuses = []ServeStatus{serving, ended}
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.want("H8")
	for i := 0; i < 3 && hs.m.ph == phServing; i++ {
		hs.beat()
	}
	if hs.m.ph != phFailed || hs.m.failure == nil {
		t.Fatalf("the end while serving was not shown: phase %v", hs.m.ph)
	}
	mustHave(t, hs.plain(), "the server ended while serving", "exit code 5")
	if hs.calls("Stop") != 0 {
		t.Errorf("the launcher stopped something after the server ended: %v", f.Calls())
	}

	// No action ends a process the launcher did not start (conflict row 6 (1); U-28 (ii)).
	// The table's actions (GL-02's table, F-LS 3.1) hold none; close-the-other-app is removed
	// (F-LS 3.2); the screen's controls hold none (ActCloseHolder removed in GL-08).
	for _, a := range TableActions {
		if strings.Contains(string(a), "close") || strings.Contains(string(a), "kill") || strings.Contains(string(a), "other-app") {
			t.Errorf("table action %q ends another process", a)
		}
	}
	if _, ok := RemovedActions["close-the-other-app"]; !ok {
		t.Error("close-the-other-app is not among the removed actions")
	}
	for _, id := range []ActionID{ActCopyAddress, ActCopyReport, ActOtherPort, ActRedoSetup, ActDeletePartial, ActTaskManager,
		ActOpenLog, ActSavePreset, ActAddFolder, ActRemoveFolder, ActLauncherSet, ActGraph, ActOpenFolder, ActShowFile, ActSetSmoke} {
		if strings.Contains(string(id), "close") || strings.Contains(string(id), "kill") || strings.Contains(string(id), "holder") {
			t.Errorf("control %q ends another process", id)
		}
	}
	// No failure page offers a fix that ends the port's owner.
	f2 := NewFake()
	f2.StartEvs = []StartEvent{{Model: "qwen35-122b", Failure: &Failure{Stage: "fail_server_start", What: "the server did not start",
		Why: "port 8093 is in use by another program", Holder: "other-server.exe", PID: 4711, ExitCode: 5}}}
	hs = newH(t, f2, LangEnglish, 120, 30)
	hs.keys("enter")
	mustHave(t, hs.plain(), "port 8093 is in use by another program", "other-server.exe", "PID 4711")
	before := len(f2.Calls())
	for _, k := range []string{"k", "ctrl+k", "x", "d"} {
		hs.keys(k)
	}
	for _, c := range f2.Calls()[before:] {
		if strings.HasPrefix(c, "Action") || strings.HasPrefix(c, "Stop") {
			t.Errorf("a key on the failure page acted on another process: %s", c)
		}
	}
}

// closeClaims is F-LS 11.2's phrase list (LS:138; conflict row 5).
var closeClaims = []string{"closing the window saves", "saved when the window closes", "saved on close",
	"cleans up on close", "cleanup on close", "the window closes cleanly", "closing the window stops the server cleanly",
	"창을 닫으면 저장", "창 닫을 때 저장", "창을 닫아도 저장", "창을 닫으면 정리", "창 닫기=정상 종료"}

// S8-LA-19 (rule 5, F-LS 4.2, 11.1, 11.2; conflict row 5): the help line for closing the window
// says the decided fact in both languages, and no screen, help or log string of the module
// claims a save or an orderly stop on closing the window.
func TestS8LA19(t *testing.T) {
	for lang, want := range map[string]string{
		LangEnglish: "X closing the window stops the server at once · nothing saved (same as ctrl+c)",
		LangKorean:  "X 창 닫기=서버도 바로 꺼짐 · 저장 없음(ctrl+c 와 같음)"} {
		hs := newH(t, NewFake(), lang, 160, 40)
		hs.keys("?")
		if !strings.Contains(flat(hs.plain()), flat(want)) {
			t.Errorf("%s help line, want %q:\n%s", lang, want, hs.plain())
		}
	}
	if got := tr(LangKorean, "closing the window stops the server at once · nothing saved (same as ctrl+c)"); got != "창 닫기=서버도 바로 꺼짐 · 저장 없음(ctrl+c 와 같음)" {
		t.Errorf("Korean help text %q", got)
	}

	// The string tables: every string literal of every source file of the module (the screen,
	// help and log strings of both languages, tr.go's table among them).
	root := filepath.Join("..", "..")
	files, lits := 0, 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		fset := token.NewFileSet()
		af, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			return err
		}
		files++
		ast.Inspect(af, func(n ast.Node) bool {
			bl, ok := n.(*ast.BasicLit)
			if !ok || bl.Kind != token.STRING {
				return true
			}
			s, err := strconv.Unquote(bl.Value)
			if err != nil {
				return true
			}
			lits++
			low := strings.ToLower(s)
			for _, ph := range closeClaims {
				if strings.Contains(low, strings.ToLower(ph)) {
					t.Errorf("%s: %q claims %q", fset.Position(bl.Pos()), s, ph)
				}
			}
			return true
		})
		return nil
	})
	if err != nil || files < 50 || lits < 1000 {
		t.Fatalf("source walk: %v, %d files, %d literals", err, files, lits)
	}
	for _, m := range table {
		for k, v := range m {
			for _, ph := range closeClaims {
				if strings.Contains(strings.ToLower(k+"\x00"+v), strings.ToLower(ph)) {
					t.Errorf("table entry %q = %q claims %q", k, v, ph)
				}
			}
		}
	}
}

// S8-LA-21 (review item 1; design H12): while a server runs the model list is locked, the
// start action starts nothing and a changed setting says it applies at the next start; after a
// stop the launcher stays on the stopped page, and the start action starts the same model again.
func TestS8LA21(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	hs.want("H8")
	hs.keys("1", "down")
	if hs.m.sel != 0 || !strings.Contains(hs.m.notice, "locked") {
		t.Fatalf("the list moved while serving: sel %d notice %q", hs.m.sel, hs.m.notice)
	}
	hs.keys("enter")
	hs.keys("tab", "right")
	mustHave(t, hs.plain(), "not applied to the running server · applies at the next start")
	hs.keys("enter")
	if hs.calls("Start") != 1 || hs.calls("Stop") != 0 {
		t.Fatalf("the start action while serving: %v", f.Calls())
	}
	hs.keys("2", "ctrl+f")
	hs.want("H12")
	hs.advance(10 * 60 * 1e9)
	hs.beat()
	hs.want("H12") // the launcher stays on the stopped page
	if hs.quit {
		t.Fatal("the launcher quit after the stop")
	}
	hs.keys("enter")
	hs.want("H8")
	if hs.calls("Start qwen35-122b") != 2 {
		t.Fatalf("start again: %v", f.Calls())
	}
}
