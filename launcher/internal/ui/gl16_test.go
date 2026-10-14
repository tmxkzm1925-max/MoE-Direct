package ui

import "testing"

// GL-16: the launcher settings of 5 Settings: the prompt-cache cap (U-22; ←→ cycles, Space
// types MiB, 0 = off, a refused value keeps the box with the settings-typed text of F-LS 11.4)
// and the warm-start switches (OQ-6 (a)); a value that differs from the running server's shows
// the next-start line (rows 6.1-6.2).
func TestGL16LauncherSettings(t *testing.T) {
	f := NewFake()
	f.Info.CacheCap, f.Info.WarmStart, f.Info.Autosave = "off", "on", "on"
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("5", "tab")
	for hs.m.set5Row < set5CapRow {
		hs.keys("down")
	}
	if hs.m.set5Sec != secLauncher || hs.m.set5Row != set5CapRow {
		t.Fatalf("cap row: sec %d row %d", hs.m.set5Sec, hs.m.set5Row)
	}
	// The default is the off form (DECISION_LEDGER.md:4420); ←→ cycles to a positive cap.
	mustHave(t, hs.plain(), "prompt cache", "off · the default", "prompt cache = RAM for reused prompts")
	hs.keys("right")
	if hs.calls("Action set-cache-cap 256") != 1 || f.Info.CacheCap != "256" {
		t.Fatalf("cycle: %v %q", f.Calls(), f.Info.CacheCap)
	}
	hs.keys("left")
	if hs.calls("Action set-cache-cap off") != 1 || f.Info.CacheCap != "off" {
		t.Fatalf("cycle back: %v %q", f.Calls(), f.Info.CacheCap)
	}
	hs.keys(" ", "backspace", "backspace", "backspace", "1", "0", "0", "enter")
	if !hs.m.inputOn || hs.m.inputErr == "" {
		t.Fatalf("a refused cap must keep the box: on %v err %q", hs.m.inputOn, hs.m.inputErr)
	}
	mustHave(t, hs.plain(), "✕ outside its range · allowed 0 (off) or 256..8192 MiB", "· the value is kept, nothing applied")
	if f.Info.CacheCap != "off" {
		t.Fatalf("a refused value was applied: %q", f.Info.CacheCap)
	}
	hs.keys("backspace", "backspace", "backspace", "2", "5", "6", "enter")
	if hs.m.inputOn || f.Info.CacheCap != "256" {
		t.Fatalf("typed cap: on %v cap %q", hs.m.inputOn, f.Info.CacheCap)
	}
	mustHave(t, hs.plain(), "256 MiB")
	hs.keys("down", "right")
	if hs.calls("Action set-warmstart off") != 1 || f.Info.WarmStart != "off" {
		t.Fatalf("warm start: %v", f.Calls())
	}
	hs.keys("down", "left")
	if hs.calls("Action set-autosave off") != 1 {
		t.Fatalf("autosave: %v", f.Calls())
	}
	// While a server runs with other values: the next-start line.
	f.Info.NextStart = map[string]bool{"warmstart": true}
	hs.keys("right") // the screen re-reads the launcher settings after an action
	mustHave(t, hs.plain(), "warm start     off  ▲", "▲ not applied to the running server · applies at the", "next start")
}

// The settings-typed refusal in Korean (F-LS 11.4).
func TestGL16TypedRefusalKorean(t *testing.T) {
	f := NewFake()
	f.Info.CacheCap = "off"
	hs := newH(t, f, LangKorean, 120, 30)
	hs.keys("5", "tab")
	for hs.m.set5Row < set5CapRow {
		hs.keys("down")
	}
	hs.keys(" ", "backspace", "backspace", "backspace", "backspace", "1", "0", "0", "enter")
	if got := typedErr(hs.m.r, hs.m.inputErr); got != "범위 밖 · 허용 0 (off) or 256..8192 MiB · 값은 그대로, 적용 안 됨" {
		t.Fatalf("korean refusal %q", got)
	}
	mustHave(t, hs.plain(), "✕ 범위 밖 · 허용 0 (off) or")
}
