// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// FIX-UI2: tests of review items 31 and 32 of reviews/_gol_review_8_user_20261010.md.

package ui

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// item 31: on 1 Models the list is read again every five seconds; a model
// removed from the backend's list is gone within one tick and the
// selection stays on the same model; on other places the list is not read.
func TestItem31Refresh(t *testing.T) {
	f := NewFake()
	hs := newH(t, f, LangEnglish, 120, 30)
	hs.keys("down", "down", "down") // Qwen3-30B-A3B, the fourth row
	if row, _ := hs.m.selRow(); row.ID != "qwen3-30b" {
		t.Fatalf("selected %q", row.ID)
	}
	mustHave(t, hs.plain(), "gpt-oss-20")
	n := hs.calls("Catalog")
	f.RemoveModel("gpt-oss-20b")
	hs.advance(4 * time.Second)
	hs.beat()
	if hs.calls("Catalog") != n {
		t.Fatalf("read again before five seconds")
	}
	hs.advance(time.Second)
	hs.beat()
	if hs.calls("Catalog") != n+1 {
		t.Fatalf("not read again after five seconds: %d", hs.calls("Catalog")-n)
	}
	mustLack(t, hs.plain(), "gpt-oss-20")
	if row, _ := hs.m.selRow(); row.ID != "qwen3-30b" || hs.m.sel != 2 {
		t.Fatalf("selection moved: %q at %d", row.ID, hs.m.sel)
	}
	// other places: no reading
	hs.keys("3")
	n = hs.calls("Catalog")
	for i := 0; i < 4; i++ {
		hs.advance(5 * time.Second)
		hs.beat()
	}
	hs.keys("4")
	hs.advance(5 * time.Second)
	hs.beat()
	if hs.calls("Catalog") != n {
		t.Fatalf("the list was read on another place: %d", hs.calls("Catalog")-n)
	}
	// arriving at 1 Models reads it at once
	f.RemoveModel("dsv4-flash")
	hs.keys("1")
	if hs.calls("Catalog") != n+1 {
		t.Fatalf("not read on arrival")
	}
	mustLack(t, hs.plain(), "DSV4-Flash")
}

// item 31: a model whose repack output was deleted shows «needs repack» in
// its row, in the SETUP tile and in THIS MODEL's setup line.
func TestItem31NeedsRepack(t *testing.T) {
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	hs.keys("down", "down", "down")
	mustHave(t, hs.plain(), "Qwen3-30B- ▲ needs repack", "SETUP", "▲ needs repack  ", "repack output deleted", "setup      packed ✓ · virtual ▲")
	hs = newH(t, NewFake(), LangKorean, 120, 30)
	hs.keys("down", "down", "down")
	mustHave(t, hs.plain(), "▲ 리팩 필요")
}

// flat drops spaces and box edges, so a wrapped text reads as one.
func flat(s string) string {
	return strings.Map(func(r rune) rune {
		if strings.ContainsRune(" \n│╭╮╰╯─", r) {
			return -1
		}
		return r
	}, s)
}

// textCol is the cell column of the first character after a line's left box
// edge and its spaces (-1: no edge).
func textCol(l string) int {
	i := strings.Index(l, "│")
	if i < 0 {
		return -1
	}
	rest := l[i+len("│"):]
	t := strings.TrimLeft(rest, " ")
	return lipgloss.Width(l[:i]) + 1 + (len(rest) - len(t))
}

// item 32: a long refusal reason wraps inside the 3 Memory box and under the
// 1 Models tiles, whole, at 80 and 120 columns; the box keeps its edges.
func TestItem32Wrap(t *testing.T) {
	for _, w := range []int{120, 80} {
		f := NewFake()
		f.Pre.Refused, f.Pre.Verdict, f.Pre.Refusals = true, VerdictCannot, []string{FakeLongRefusal}
		f.MemIdle.ReadyNowGB = K(30) // budget and need fit: the reason is the backend's
		hs := newH(t, f, LangEnglish, w, 40)
		p := hs.plain()
		if !strings.Contains(flat(p), flat("✕ cannot start · "+FakeLongRefusal+" · details on 3 Memory")) {
			t.Fatalf("%d: 1 Models verdict not whole:\n%s", w, p)
		}
		hs.keys("3")
		p = hs.plain()
		if !strings.Contains(flat(p), flat("✕ refused: "+FakeLongRefusal)) {
			t.Fatalf("%d: 3 Memory refusal not whole:\n%s", w, p)
		}
		ls := strings.Split(p, "\n")
		first := -1
		for i, l := range ls {
			if lipgloss.Width(l) > w {
				t.Fatalf("%d: line wider than the window: %q", w, l)
			}
			if first < 0 && strings.Contains(l, "✕ refused:") {
				first = i
			}
		}
		if first < 0 || strings.Contains(ls[first], "…") {
			t.Fatalf("%d: refusal missing or cut:\n%s", w, p)
		}
		// the next line continues under the text, inside the same box edges
		next := ls[first+1]
		if cellIdx(next, "│") != cellIdx(ls[first], "│") || lipgloss.Width(next) != lipgloss.Width(ls[first]) {
			t.Fatalf("%d: box edges moved:\n%s\n%s", w, ls[first], next)
		}
		if at := cellIdx(ls[first], "refused:"); textCol(next) != at {
			t.Fatalf("%d: continuation at %d, text at %d:\n%s\n%s", w, textCol(next), at, ls[first], next)
		}
	}
}

// item 32: a failure reason too long for its tile follows whole on 2 Server.
func TestItem32FailureReason(t *testing.T) {
	hs := newH(t, NewFake(), LangEnglish, 80, 30)
	hs.m.failure = &Failure{What: "repack output missing", Why: FakeLongRefusal}
	hs.m.ph = phFailed
	hs.keys("2")
	hs.want("H7")
	if !strings.Contains(flat(hs.plain()), flat(FakeLongRefusal)) {
		t.Fatalf("failure reason not whole:\n%s", hs.plain())
	}
}
