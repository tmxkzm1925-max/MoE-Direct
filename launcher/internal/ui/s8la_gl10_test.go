package ui

import (
	"testing"

	"github.com/charmbracelet/x/ansi"
)

// GL-10 (plan cell GL-10): the screen's half of the setup flow; core's and the adapter's halves
// are the core and wire tests of the same names.

// TestS8LA24 (S8-LA-24; rules 9 and 18; conflict row 9): a model that needs a setup is marked;
// ⏎ shows the plan box (what, where, disk against the drive, time unknown the first time) and
// runs nothing; esc writes nothing; the box's ⏎ (the confirm action) runs the setup; the next
// line says verify, the speed check and the start follow; a disk shortfall is the refusal line.
func TestS8LA24(t *testing.T) {
	f := NewFake()
	f.StartHold = true
	hs := newH(t, f, LangEnglish, 160, 30)
	hs.keys("down", "down", "down") // Qwen3-30B-A3B: virtual needs its setup
	mustHave(t, hs.plain(), "setup      packed ✓ · virtual needs repack", "⏎ SETUP PLAN")
	hs.keys("enter")
	mustHave(t, hs.plain(), "SETUP NEEDED · Qwen3-30B-A3B · virtual", "writes to", `D:\moe-models\repack\qwen3-30b\virtual`,
		"about 17.5 GB written · drive free 412.0 GB → 394.5 GB left", "first repack: not known yet",
		"then: check → speed check → the server starts by itself")
	if hs.calls("RepackPlan qwen3-30b") != 1 || hs.calls("Repack ") != 0 || hs.calls("Start") != 0 {
		t.Fatalf("the plan box ran something: %v", f.Calls())
	}
	hs.keys("esc")
	if hs.calls("Repack ") != 0 || hs.calls("Start") != 0 {
		t.Fatalf("esc ran something: %v", f.Calls())
	}
	hs.keys("enter", "enter")
	if hs.calls("Repack qwen3-30b") != 1 || hs.calls("Start") != 0 {
		t.Fatalf("the confirm action: %v", f.Calls())
	}
	hs.want("H6")
	f = NewFake()
	f.Plan.WriteGB, f.Plan.LeftGB = K(500), K(-88)
	hs = newH(t, f, LangEnglish, 160, 30)
	hs.keys("down", "down", "down", "enter")
	mustHave(t, hs.plain(), "✕ cannot start · free 412.0 GB < needed 500.0 GB")
	mustLack(t, hs.plain(), "SETUP NEEDED")
	hs.keys("enter")
	if hs.calls("Repack ") != 0 {
		t.Fatalf("a short drive ran the setup: %v", f.Calls())
	}
}

// TestS8LA34 (S8-LA-34; LFS Q5.11, Q5.16; H4; W7 duty 12): the cut-off screen's ctrl+r shows the
// setup plan (no run before its ⏎), its ctrl+d removes the half-made files only and reads the
// list again; an output made from other model files reads stale in the setup line (EN and KO).
func TestS8LA34(t *testing.T) {
	cutOff := func() *FakeBackend {
		f := NewFake()
		f.Cat.Models[0].Setup = SetupCutOff
		f.Cat.Models[0].Partial = PartialSetup{LayersDone: U, LayersTotal: U, FilesGB: K(37), RedoMin: K(25), When: "2026-10-12 09:30",
			Cause: "· the launcher was closed or stopped during the setup"}
		return f
	}
	f := cutOff()
	hs := newH(t, f, LangEnglish, 160, 30)
	mustHave(t, hs.plain(), "setup of Qwen3.5-122B was cut off 2026-10-12 09:30 · the launcher was closed or stopped during the setup",
		"START AGAIN", "ctrl+d delete half-made only", "esc not now")
	hs.keys("ctrl+r")
	if hs.calls("RepackPlan qwen35-122b ") != 1 || hs.calls("Repack ") != 0 || hs.calls("Action ") != 0 {
		t.Fatalf("ctrl+r: %v", f.Calls())
	}
	mustHave(t, hs.plain(), "SETUP NEEDED")
	hs.keys("enter")
	if hs.calls("Repack qwen35-122b") != 1 {
		t.Fatalf("the plan's ⏎: %v", f.Calls())
	}

	f = cutOff()
	hs = newH(t, f, LangEnglish, 160, 30)
	n0 := hs.calls("Catalog")
	hs.keys("ctrl+d")
	if hs.calls("Action "+string(ActDeletePartial)+" qwen35-122b") != 1 || hs.calls("Repack") != 0 || hs.calls("Catalog") <= n0 {
		t.Fatalf("ctrl+d: %v", f.Calls())
	}
	mustHave(t, hs.plain(), "half-made files removed")
	mustLack(t, hs.plain(), "START AGAIN")

	// the setup line in its box: the long form where it fits (KO), the short form keeps the word (EN)
	for _, c := range []struct{ lang, want, long string }{
		{LangEnglish, "setup      packed ✓ · virtual ✕ stale", "packed ✓ · virtual ✕ stale (another file)"},
		{LangKorean, "준비       packed ✓ · virtual ✕ 다른 파일용", "packed ✓ · virtual ✕ 다른 파일용"},
	} {
		f = NewFake()
		f.Cat.Models[0].Modes = []ModeCheck{{"packed", ModeReady}, {"virtual", ModeStale}} // virtual: the mode now set
		hs = newH(t, f, c.lang, 160, 30)
		mustHave(t, hs.plain(), c.want)
		if got := ansi.Strip(hs.m.modesText(f.Cat.Models[0], false)); got != c.long {
			t.Errorf("%s long form %q", c.lang, got)
		}
		hs.keys("enter") // the stale mode now set needs its setup: the plan, not the start
		if hs.calls("RepackPlan qwen35-122b") != 1 || hs.calls("Start") != 0 {
			t.Errorf("%s: ⏎ on a stale mode: %v", c.lang, f.Calls())
		}
	}
}
