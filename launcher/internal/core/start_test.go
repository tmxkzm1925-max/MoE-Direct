package core

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func drain(t *testing.T, ch <-chan StartEvent) []StartEvent {
	t.Helper()
	var evs []StartEvent
	timeout := time.After(30 * time.Second)
	for {
		select {
		case e, ok := <-ch:
			if !ok {
				return evs
			}
			evs = append(evs, e)
		case <-timeout:
			t.Fatal("start events did not end")
		}
	}
}

// The start API without a spawn: every run here ends before the server step, so no process is
// created (the spawn and ready path is exercised in the GOL-WIRE rehearsal).
func TestStartEndsAtPSFailureStage(t *testing.T) {
	f := newFixture(t, false)
	if _, err := f.l.Stop(); !errors.Is(err, ErrNoServer) {
		t.Errorf("stop before start: %v", err)
	}
	if st, err := f.l.Status(context.Background()); !errors.Is(err, ErrNoServer) || st.Running {
		t.Errorf("status before start: %+v %v", st, err)
	}
	os.Remove(filepath.Join(f.outDir, "experts.bin"))
	ch, err := f.l.Start(context.Background(), Request{ModelPath: f.modelPath})
	if err != nil {
		t.Fatal(err)
	}
	evs := drain(t, ch)
	last := evs[len(evs)-1]
	var rn *RepackNeededError
	if !last.Final || last.Ready || last.Status != StatusFailRepack || last.ExitCode != 3 || !errors.As(last.Err, &rn) {
		t.Fatalf("final %+v", last)
	}
	var steps []string
	for _, e := range evs {
		if !e.Final && e.State == StepRunning {
			steps = append(steps, e.Step)
		}
	}
	if len(steps) != 4 || steps[0] != StepBundle || steps[1] != StepCatalog || steps[3] != StepArtifacts {
		t.Errorf("steps %v", steps)
	}
	if evs[len(evs)-2].State != StepFailed || evs[len(evs)-2].Step != StepArtifacts {
		t.Errorf("failed step %+v", evs[len(evs)-2])
	}
	// The launcher is free again: a second start runs, and a cancelled one ends cancelled_user.
	cctx, cancel := context.WithCancel(context.Background())
	cancel()
	ch, err = f.l.Start(cctx, Request{ModelPath: f.modelPath})
	if err != nil {
		t.Fatal(err)
	}
	evs = drain(t, ch)
	if last := evs[len(evs)-1]; last.Status != StatusCancelledUser || last.ExitCode != 2 {
		t.Errorf("cancelled %+v", last)
	}
	// A refusal before the steps: the exit table of the stage.
	ch, _ = f.l.Start(context.Background(), Request{ModelPath: filepath.Join(f.dir, "none.gguf")})
	evs = drain(t, ch)
	if last := evs[len(evs)-1]; last.Status != StatusFailModelPath || last.ExitCode != 3 {
		t.Errorf("missing model %+v", last)
	}
}
