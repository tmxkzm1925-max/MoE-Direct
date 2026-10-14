package wire

// UX-R8 (version 8; the user's acceptance finding 15 (a), reviews/_user_8_LAUNCHER-REVIEW_4.md):
// on 2 Server the user pressed ⏎ in the setup plan box of gpt-oss-120b MXFP4 (packed) and the
// start failed with «gpt-oss-120b MXFP4: the model file is no longer in the model folders»,
// the model being present under D:\moe-models. The plan box's model is what the screen's ⏎
// hands to Repack (ui keys.go keyPlan: beginRun(p.Model)), so it must be the row ID (the
// path), not the display name; fileOf only knows paths.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// uxr8Screen drives the real screen (ui.Model) over this Backend: commands run off the update
// path as Bubble Tea runs them and their messages are fed back; timers are kept and fired on
// demand.
type uxr8Screen struct {
	t     *testing.T
	m     tea.Model
	clock time.Time
	ticks []func(time.Time) tea.Msg
}

func (s *uxr8Screen) run(c tea.Cmd) {
	if c == nil {
		return
	}
	got := make(chan tea.Msg, 1)
	go func() { got <- c() }()
	var msg tea.Msg
	select {
	case msg = <-got:
	case <-time.After(3 * time.Second):
		return // a command still blocked (an open channel) is left behind
	}
	switch msg := msg.(type) {
	case nil, tea.QuitMsg:
	case tea.BatchMsg:
		for _, x := range msg {
			s.run(x)
		}
	default:
		s.send(msg)
	}
}

func (s *uxr8Screen) send(msg tea.Msg) {
	m, c := s.m.Update(msg)
	s.m = m
	s.run(c)
}

func (s *uxr8Screen) key(k string) {
	if k == "enter" {
		s.send(tea.KeyMsg{Type: tea.KeyEnter})
		return
	}
	s.send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)})
}

// fire sends the kept timers once (the first draw, a beat).
func (s *uxr8Screen) fire() {
	due := s.ticks
	s.ticks = nil
	for _, fn := range due {
		s.send(fn(s.clock))
	}
}

func (s *uxr8Screen) plain() string { return ansi.Strip(s.m.View()) }

func TestUXR8SetupEntry(t *testing.T) {
	// the user's case: one gpt-oss-120b MXFP4 file in a watched model folder, its packed setup
	// missing (launcher_20261012T205651Z_1432.jsonl lines 13 and 25)
	root := t.TempDir()
	folder := filepath.Join(root, "moe-models")
	model := filepath.Join(folder, "gpt-oss-120b", "gpt-oss-120b-MXFP4.gguf")
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(model, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog := filepath.Join("..", "..", "..", "launcher", "models.json")
	l, err := core.New(core.Options{CatalogPath: catalog, BundleRoot: filepath.Join("..", "..", "..", "..", "repack"),
		StateDir: filepath.Join(root, "state"), ModelsRoot: folder})
	if err != nil {
		t.Fatal(err)
	}
	b := New(l, catalog, "test")
	// the list as core gives it for the user's file (catalog match: the profile's display name);
	// the folder stamp is the current one, so the screen's 5 s re-read keeps this list
	f := core.ModelFile{Path: model, Name: filepath.Base(model), Folder: folder, DisplayName: "gpt-oss-120b MXFP4",
		ProfileID: "gpt-oss-120b", Catalog: true, Bytes: 4, Shards: 1, Declared: 1, Identity: "unpinned"}
	b.mu.Lock()
	b.ml, b.stamp = &core.ModelList{Files: []core.ModelFile{f}, Roots: []string{folder}}, l.ScanStamp()
	b.files = map[string]core.ModelFile{model: f}
	b.mu.Unlock()
	b.bundleFn = func() core.BundleResult { return core.BundleResult{State: core.BundleDev} }
	b.planTextFn = nil
	var mu sync.Mutex
	var planned, repacked, started []string
	b.planFn = func(_ context.Context, req core.Request, _ bool) (*core.RepackPlan, error) {
		mu.Lock()
		planned = append(planned, req.ModelPath)
		mu.Unlock()
		return &core.RepackPlan{Mode: "packed", OutputDir: filepath.Join(filepath.Dir(model), "repack"), NeedKnown: true,
			NeedMiB: 58244, FreeMiB: 112321, LeftMiB: 54077, Exe: "repacker.exe"}, nil
	}
	b.repackFn = func(_ context.Context, p *core.RepackPlan, _ core.RepackExecutor, progress func(core.RepackEvent)) (core.RepackResult, error) {
		mu.Lock()
		repacked = append(repacked, p.OutputDir)
		mu.Unlock()
		progress(core.RepackEvent{Stage: core.RepackStageVerify, Line: "[verify] 36/36 pairs", Fraction: 1})
		progress(core.RepackEvent{Stage: core.RepackStageDone, Line: "setup done", Fraction: 1})
		return core.RepackResult{Duration: 30 * time.Minute}, nil
	}
	b.startFn = func(_ context.Context, req core.Request) (<-chan core.StartEvent, error) {
		mu.Lock()
		started = append(started, req.ModelPath)
		mu.Unlock()
		c := make(chan core.StartEvent, 1)
		c <- core.StartEvent{Final: true, Ready: true, Address: "http://127.0.0.1:8093", PID: 7}
		close(c)
		return c, nil
	}

	s := &uxr8Screen{t: t, clock: time.Date(2026, 10, 13, 6, 1, 0, 0, time.UTC)}
	s.m = ui.New(b, ui.Options{Lang: ui.LangEnglish, Width: 120, Height: 30,
		Now: func() time.Time { return s.clock },
		Tick: func(_ time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
			s.ticks = append(s.ticks, fn)
			return nil
		}})
	s.run(s.m.Init())
	s.fire() // the first draw: the launcher file check answers (development build)
	if p := s.plain(); !strings.Contains(p, "gpt-oss-120b MXFP4") {
		t.Fatalf("1 Models lacks the model:\n%s", p)
	}

	// ⏎ on 1 Models: the setup plan box on 2 Server (conflict 9)
	s.key("enter")
	if p := s.plain(); !strings.Contains(p, "SETUP NEEDED") || !strings.Contains(p, "gpt-oss-120b MXFP4") {
		t.Fatalf("no plan box after ⏎:\n%s", p)
	}
	s.fire() // a 5 s beat between the plan and the ⏎: the folder re-read keeps the list
	// ⏎ in the plan box: the repack, the check, then the start, all on the model's path
	s.key("enter")
	for i := 0; i < 50; i++ {
		mu.Lock()
		n := len(started)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	p := s.plain()
	mu.Lock()
	defer mu.Unlock()
	if strings.Contains(p, "no longer in the model folders") || len(repacked) != 1 || len(started) != 1 ||
		started[0] != model || len(planned) != 1 || planned[0] != model {
		t.Fatalf("⏎ in the plan box: planned %q, repacked %q, started %q\n%s", planned, repacked, started, p)
	}

	// a model really gone from the folders still gets the line
	b.mu.Lock()
	delete(b.files, model)
	b.ml, b.stamp = &core.ModelList{Roots: []string{folder}}, l.ScanStamp() // the last list no longer has it
	b.mu.Unlock()
	if _, err := b.Repack(model, ui.Settings{}); err == nil || !strings.Contains(err.Error(), "gpt-oss-120b-MXFP4.gguf: the model file is no longer in the model folders") {
		t.Fatalf("gone model: %v", err)
	}
}
