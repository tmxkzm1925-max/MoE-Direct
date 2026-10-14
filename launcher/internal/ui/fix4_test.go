package ui

import "testing"

// FIX2-WIRE rehearsal repair: the SETUP tile follows the mode a model starts with (its saved
// preset, else the catalog default), but the mode the screen holds decides ⏎ when the backend
// gives that mode's check: a model whose packed output is gone and whose virtual output passes
// starts in virtual without the plan box; the other way round the plan box opens.
func TestHeldModeDecidesSetup(t *testing.T) {
	f := NewFake()
	f.Cat.Models[0].Setup = SetupNeedsRepack // packed, the start mode, has no output
	f.Cat.Models[0].Modes = []ModeCheck{{"packed", ModeNeedsRepack}, {"virtual", ModeReady}}
	hs := newH(t, f, LangEnglish, 120, 30)
	if cur := hs.m.curMode(f.Cat.Models[0]); cur != "virtual" {
		t.Fatalf("the fake's held mode is %q", cur)
	}
	hs.keys("enter")
	if hs.calls("RepackPlan") != 0 || hs.calls("Start") != 1 {
		t.Fatalf("held virtual with its check passed: %v", f.Calls())
	}
	f = NewFake()
	f.Cat.Models[0].Setup = SetupReady
	f.Cat.Models[0].Modes = []ModeCheck{{"packed", ModeReady}, {"virtual", ModeNeedsRepack}}
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	if hs.calls("RepackPlan") != 1 || hs.calls("Start") != 0 {
		t.Fatalf("held virtual without its output: %v", f.Calls())
	}
	// Without a mode check the tile decides, as before.
	f = NewFake()
	f.Cat.Models[0].Setup = SetupNeedsRepack
	f.Cat.Models[0].Modes = nil
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("enter")
	if hs.calls("RepackPlan") != 1 {
		t.Fatalf("no mode check: %v", f.Calls())
	}
}
