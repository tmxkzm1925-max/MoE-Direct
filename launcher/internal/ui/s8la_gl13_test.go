package ui

import (
	"strings"
	"sync"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

// S8-LA-27 (GL-13; conflict 18-2): the launcher file check starts after the first screen is
// drawn (the backend's first Bundle call starts it), only the start waits for its end, a
// damaged bundle refuses with the count and the reinstall line, a development build says so.

// la27Out records whether a frame (any printable text) has reached the console.
type la27Out struct {
	mu    sync.Mutex
	drawn bool
}

func (w *la27Out) Write(b []byte) (int, error) {
	w.mu.Lock()
	if strings.TrimSpace(ansi.Strip(string(b))) != "" {
		w.drawn = true
	}
	w.mu.Unlock()
	return len(b), nil
}

func (w *la27Out) isDrawn() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.drawn
}

// la27Backend is the fake whose Bundle notes whether the first screen was on the console when
// the check was first asked for (that call starts the check in the real backend).
type la27Backend struct {
	*FakeBackend
	out          *la27Out
	once         sync.Once
	first        chan struct{}
	drawnAtFirst bool
}

func (b *la27Backend) Bundle() (BundleCheck, error) {
	b.once.Do(func() {
		b.drawnAtFirst = b.out.isDrawn()
		close(b.first)
	})
	return b.FakeBackend.Bundle()
}

// la27H is newH without the first draw: Init has run, the check is not asked for yet.
func la27H(t *testing.T, f *FakeBackend, lang string) *harness {
	t.Helper()
	hs := &harness{t: t, f: f, clock: time.Date(2026, 10, 10, 9, 0, 0, 0, time.UTC)}
	hs.m = New(f, Options{Lang: lang, Width: 120, Height: 30,
		Now: func() time.Time { return hs.clock },
		Tick: func(d time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
			hs.ticks = append(hs.ticks, pendingTick{d, hs.clock.Add(d), fn})
			return nil
		}})
	t.Cleanup(f.Close)
	hs.run(hs.m.Init())
	return hs
}

func TestS8LA27(t *testing.T) {
	t.Run("the check is asked for after the first draw", func(t *testing.T) {
		out := &la27Out{}
		b := &la27Backend{FakeBackend: NewFake(), out: out, first: make(chan struct{})}
		t.Cleanup(b.FakeBackend.Close)
		p := tea.NewProgram(New(b, Options{Lang: LangEnglish, Width: 100, Height: 30}),
			tea.WithOutput(out), tea.WithInput(nil), tea.WithoutSignalHandler())
		done := make(chan error, 1)
		go func() { _, err := p.Run(); done <- err }()
		select {
		case <-b.first:
		case <-time.After(10 * time.Second):
			p.Kill()
			t.Fatal("the file check was never asked for")
		}
		p.Quit()
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(10 * time.Second):
			p.Kill()
			t.Fatal("the program did not end")
		}
		if !b.drawnAtFirst {
			t.Fatal("the file check was asked for before the first screen reached the console")
		}
	})

	t.Run("only the start waits for the check", func(t *testing.T) {
		f := NewFake()
		hs := la27H(t, f, LangEnglish)
		if n := hs.calls("Bundle"); n != 0 {
			t.Fatalf("the check was asked for before the first draw: %v", f.Calls())
		}
		hs.keys("enter")
		if hs.calls("Start") != 0 || hs.calls("Preflight qwen35-122b map[] true") != 0 || hs.m.startWait == "" {
			t.Fatalf("a start went on before the check answered: %v", f.Calls())
		}
		mustHave(t, hs.plain(), "checking the launcher files … the start follows")
		hs.keys("5") // the other places do not wait
		mustHave(t, hs.plain(), "LAUNCHER")
		hs.keys("1")
		hs.beat() // no poll before the first draw
		if n := hs.calls("Bundle"); n != 0 {
			t.Fatalf("polled before the first draw: %v", f.Calls())
		}
		hs.drawn() // the first screen is up: the check is asked for, answers OK, the start goes on
		if hs.calls("Bundle") != 1 || hs.calls("Start") != 1 {
			t.Fatalf("the start did not follow the check: %v", f.Calls())
		}
	})

	t.Run("a damaged bundle refuses with the count", func(t *testing.T) {
		for lang, want := range map[string]string{
			LangEnglish: "✕ the launcher files are damaged · 2 differ · install again",
			LangKorean:  "✕ 배포 파일 손상 · 2개 불일치 · 다시 설치",
		} {
			f := NewFake()
			f.Bundles = []BundleCheck{{State: BundleBroken, Mismatches: 2}}
			hs := newH(t, f, lang, 120, 30)
			mustHave(t, hs.plain(), want)
			hs.keys("enter")
			if hs.calls("Start") != 0 || hs.calls("Preflight qwen35-122b map[] true") != 0 {
				t.Fatalf("%s: a damaged bundle started: %v", lang, f.Calls())
			}
			mustHave(t, hs.plain(), want)
		}
	})

	t.Run("a development build says so", func(t *testing.T) {
		f := NewFake()
		f.Bundles = []BundleCheck{{State: BundleDev}}
		hs := newH(t, f, LangKorean, 120, 30)
		hs.keys("5")
		mustHave(t, hs.plain(), "개발 빌드 · 배포 점검 생략")
		hs.keys("1", "enter")
		if hs.calls("Start") != 1 {
			t.Fatalf("a development build did not start: %v", f.Calls())
		}
	})
}
