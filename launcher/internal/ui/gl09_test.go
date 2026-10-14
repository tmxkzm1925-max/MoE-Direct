package ui

import (
	"regexp"
	"strings"
	"testing"
)

// refusedFake is a check refused by rule 7: the budget 12 G above the 8 G free now; a
// budget of 7 G or less starts it (conflict row 7).
func refusedFake() *FakeBackend {
	f := NewFake()
	f.Pre.Refused, f.Pre.Verdict = true, VerdictCannot
	f.Pre.CacheGB, f.Pre.AvailGB, f.Pre.BudgetCapGB = K(12), K(8), K(7)
	f.Pre.NeedGB, f.Pre.LeftGB = K(19), K(-11)
	f.Pre.RAMTotalGB, f.Pre.AppsGB, f.Pre.ModifiedGB, f.Pre.StandbyGB, f.Pre.FreeGB = K(32), K(22.5), K(1.5), K(6.0), K(2.0)
	f.Pre.LeftIfWrittenGB = K(-9.5)
	f.Pre.Refusals = []string{"RAM budget 12.0 GB > free now 8.0 GB"}
	return f
}

var approval = regexp.MustCompile(`(?i)approve|anyway|confirm|승인|그래도`)

// TestS8LA22 is S8-LA-22 (GL-09; rule 7, LS:175, :230-231; review item 22; conflict row 7;
// U-21, L:4277; U-35, L:4275) on the screens with the fake backend.
func TestS8LA22(t *testing.T) {
	// (1) a budget above available RAM: the red line of 11.3, verdict · reason · budget part ·
	// details, and no start, no start-anyway action
	f := refusedFake()
	hs := newH(t, f, LangEnglish, 160, 34)
	mustHave(t, hs.plain(), "✕ cannot start · RAM budget 12.0 GB > free now 8.0 GB · a budget of 7.0 GB or less starts it · details on 3 Memory")
	hs.keys("enter")
	if hs.calls("Start") != 0 {
		t.Fatalf("a refused start started: %v", f.Calls())
	}
	if approval.MatchString(hs.plain()) {
		t.Fatalf("approval or start-anyway wording:\n%s", hs.plain())
	}
	k := newH(t, refusedFake(), LangKorean, 160, 34)
	mustHave(t, k.plain(), "✕ 시작 불가 · RAM 예산 12.0 GB > 지금 여유 8.0 GB · 예산을 7.0 GB 이하로 내리면 시작 가능 · 자세히 3 Memory")

	// (2) 3 Memory: the same line without its details part (this is the details page), the
	// budget part on it and no line of its own; the memory figures shown filled, the modified
	// memory and the «left» figures among them (U-21; F-BU G)
	hs.keys("3")
	p := hs.plain()
	mustHave(t, p, "cannot start · RAM budget 12.0 GB > free now 8.0 GB · a budget of 7.0 GB or less starts it",
		"MEMORY · now", "modified", "1.5", "standby", "6.0", "11.0 GB short after the start · 9.5 GB short if written to disk first")
	if strings.Contains(p, "→ a budget of") || approval.MatchString(p) {
		t.Fatalf("3 Memory, refused:\n%s", p)
	}
	k.keys("3")
	mustHave(t, k.plain(), "시작 불가 · RAM 예산 12.0 GB > 지금 여유 8.0 GB · 예산을 7.0 GB 이하로 내리면 시작 가능", "디스크에 먼저 쓰면")

	// (3) tight: the figures shown; an unmeasured need gives «not measured yet» for both left
	// figures (G.3)
	tf := NewFake()
	tf.Pre.Verdict, tf.Pre.LeftGB, tf.Pre.ModifiedGB, tf.Pre.LeftIfWrittenGB = VerdictTight, K(1.2), K(0.8), K(2.0)
	ht := newH(t, tf, LangEnglish, 160, 34)
	ht.keys("3")
	mustHave(t, ht.plain(), "MEMORY · now", "left after start 1.2 GB · if written to disk first 2.0 GB")
	uf := refusedFake()
	uf.Pre.NeedGB, uf.Pre.LeftGB, uf.Pre.LeftIfWrittenGB = U, U, U
	hu := newH(t, uf, LangEnglish, 160, 34)
	hu.keys("3")
	mustHave(t, hu.plain(), "left after start not measured yet · if written to disk first not measured yet")

	// (4) a budget within available memory: no figures panel, no approval wording, ⏎ starts
	rf := NewFake()
	hr := newH(t, rf, LangEnglish, 160, 34)
	hr.keys("3")
	if p := hr.plain(); strings.Contains(p, "MEMORY · now") || strings.Contains(p, "if written to disk first") || approval.MatchString(p) {
		t.Fatalf("clear verdict, 3 Memory:\n%s", p)
	}
	hr.keys("enter")
	if hr.calls("Start") != 1 {
		t.Fatalf("a start within available memory did not start: %v", rf.Calls())
	}

	// (5) Task Manager opens without a selection; the line names the app and its PID
	sf := NewFake()
	sf.Mem.Apps[0].PID = 4242
	hsv := newH(t, sf, LangEnglish, 160, 34)
	hsv.keys(serving("3")...)
	hsv.keys("o")
	if hsv.calls("Action task-manager ") != 1 || hsv.calls("Action task-manager x") != 0 {
		t.Fatalf("task manager call: %v", sf.Calls())
	}
	if hsv.m.notice != "Task Manager opened · find llama-server.exe, PID 4242" {
		t.Fatalf("line %q", hsv.m.notice)
	}
	idle := newH(t, NewFake(), LangKorean, 160, 34)
	idle.keys("3", "o")
	if idle.m.notice != "작업 관리자 열림 · 선택된 항목 없음" {
		t.Fatalf("idle line %q", idle.m.notice)
	}

	// (6) the busy-port reason of GL-08 in the screen language (its finding 3)
	pf := NewFake()
	pf.Pre.Refused, pf.Pre.Refusals = true, []string{"port 8093 is in use by another program"}
	hp := newH(t, pf, LangKorean, 160, 34)
	mustHave(t, hp.plain(), "포트 8093 를 다른 프로그램이 쓰는 중")
	if got := (rc{lang: LangKorean}).reason("port 8093 is in use by another program"); got != "포트 8093 를 다른 프로그램이 쓰는 중" {
		t.Fatalf("reason %q", got)
	}
}

// vramVerdict is a VRAM verdict word (rule 8: VRAM is never a gate).
var vramVerdict = regexp.MustCompile(`(?i)VRAM[^\n]{0,20}(cannot|tight|short|insufficient|not enough|부족|불가|빠듯)|(cannot|tight|short|insufficient|부족|불가|빠듯)[^\n]{0,20}VRAM`)

// TestS8LA23 is S8-LA-23 (GL-09; rule 8, LS:234-238; review item 4; conflict row 8): a full
// card shows its figures and no VRAM verdict word, the start is offered; the device's use
// and this server's share are shown apart; a load that ran out of video memory shows one
// cause line in the form of review item 22.
func TestS8LA23(t *testing.T) {
	f := NewFake()
	f.Pre.VRAMTotalGB, f.Pre.VRAMUseGB = K(16), K(15.9)
	for i := range f.Cat.Models {
		f.Cat.Models[i].VRAMTotalGB, f.Cat.Models[i].VRAMUseGB = K(16), K(15.9)
	}
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, f, lang, 160, 34)
		if vramVerdict.MatchString(hs.plain()) {
			t.Fatalf("a VRAM verdict word (%s):\n%s", lang, hs.plain())
		}
		hs.keys("3")
		if vramVerdict.MatchString(hs.plain()) {
			t.Fatalf("a VRAM verdict word on 3 Memory (%s):\n%s", lang, hs.plain())
		}
	}
	hs := newH(t, f, LangEnglish, 160, 34)
	hs.keys("enter")
	if hs.calls("Start") != 1 {
		t.Fatalf("a full card refused the start: %v", f.Calls())
	}
	// the device's use and this server's share apart (review item 4): 13.4 of 16, this server 12.9
	sv := newH(t, NewFake(), LangEnglish, 160, 34)
	sv.keys(serving("3")...)
	mustHave(t, sv.plain(), "13.4", "12.9")
	// one cause line after a load that ran out of video memory (conflict row 8)
	for lang, want := range map[string]string{
		LangEnglish: "✕ start failed: VRAM ran out · close other GPU programs or lower ctx · details on 4 Log",
		LangKorean:  "✕ VRAM 부족으로 시작 실패 · 다른 GPU 프로그램을 끄거나 ctx 를 줄여 보세요 · 자세히 4 Log",
	} {
		h := newH(t, NewFake(), lang, 120, 30)
		h.m.failure = &Failure{What: "load failed", Why: "the model ran out of VRAM while loading", VRAMOut: true}
		h.m.ph = phFailed
		h.keys("2")
		if p := h.plain(); strings.Count(p, want) != 1 {
			t.Fatalf("cause line (%s) %d times:\n%s", lang, strings.Count(p, want), p)
		}
	}
}
