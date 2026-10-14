package script

import (
	"bytes"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"moedirect/launcher/internal/ui"
)

func TestParseAndKeys(t *testing.T) {
	steps, err := Parse(strings.NewReader("# c\n\nkey ctrl+f\nkey ?\nwait 2.5 Server .*ready\npause 300\nframe H8\nquit 10\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 6 || steps[0].Key != "ctrl+f" || steps[2].Timeout != 2500*time.Millisecond || steps[3].Timeout != 300*time.Millisecond ||
		steps[4].Label != "H8" || steps[5].Timeout != 10*time.Second || steps[2].Line != 5 {
		t.Fatalf("steps %+v", steps)
	}
	for _, k := range []string{"ctrl+f", "ctrl+x", "ctrl+c", "enter", "esc", "tab", "left", "right", "up", "down", "q", "?", "1", "ㅂ"} {
		m, ok := keyMsg(k)
		if !ok || m.String() != k {
			t.Errorf("key %q -> %q", k, m.String())
		}
	}
	for _, bad := range []string{"key nosuchkey", "wait x y", "wait 3", "frob 1"} {
		if _, err := Parse(strings.NewReader(bad)); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// The whole flow on the fake backend: catalog, start, serving, the five places, help, ctrl+f,
// q twice. The fake's start and stop end at once, so this runs in seconds.
func TestRunOnFake(t *testing.T) {
	f := ui.NewFake()
	m := ui.New(f, ui.Options{Lang: ui.LangEnglish, Glyphs: ui.GlyphDesign})
	src := `
wait 5 Qwen3\.5-122B
key enter
wait 5 (?i)serv
key 3
key 4
key 5
key 1
key 2
key ?
key ?
key ctrl+f
wait 5 (?i)stopped|summary|run
key q
key q
quit 5
`
	steps, err := Parse(strings.NewReader(src))
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := Run(m, steps, &out, 100, 32); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	s := out.String()
	if n := strings.Count(s, "=== frame "); n < 14 {
		t.Errorf("%d frames", n)
	}
	if strings.Contains(s, "\x1b[") {
		t.Error("ANSI left in the frames")
	}
	if !strings.Contains(s, "program quit") {
		t.Error("no quit")
	}
	calls := strings.Join(f.Calls(), "|")
	for _, c := range []string{"Catalog", "Preflight qwen35-122b", "Start qwen35-122b", "Status", "Memory", "LogTail", "Launcher", "Stop 0"} {
		if !strings.Contains(calls, c) {
			t.Errorf("no %s call: %s", c, calls)
		}
	}
}

func TestWaitFailsWithFrame(t *testing.T) {
	m := ui.New(ui.NewFake(), ui.Options{Lang: ui.LangEnglish})
	steps, _ := Parse(strings.NewReader("wait 0.3 no-such-text-anywhere\n"))
	var out bytes.Buffer
	if err := Run(m, steps, &out, 80, 28); err == nil || !strings.Contains(out.String(), "FAILED") {
		t.Errorf("err %v\n%s", err, out.String())
	}
}

var _ tea.Model = (*ui.Model)(nil)
