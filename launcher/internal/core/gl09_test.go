package core

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestS8LA22 is S8-LA-22's core side (GL-09; rule 7, LS:175, :228-233; conflict row 7; U-21,
// L:4277; U-35, L:4275): a budget above available RAM refuses the start before any child; the
// memory lists are read with the check; the «left» figures; Task Manager without a selection.
func TestS8LA22(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	req := Request{ModelPath: f.modelPath}
	f.l.os.memLists = func() (MemoryLists, error) { return MemoryLists{ModifiedMiB: 1536, FreeZeroMiB: 2048}, nil }

	// the hard stop: no child, fail_resource exit 3, the budget that passes named
	f.mem.AvailPhysMiB = 8000
	ch, err := f.l.Start(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	evs := drain(t, ch)
	last := evs[len(evs)-1]
	if !last.Final || last.Ready || last.Status != StatusFailResource || last.ExitCode != 3 ||
		!strings.Contains(last.Err.Error(), "selected budget alone exceeds available RAM") {
		t.Fatalf("final %+v", last)
	}
	for _, e := range evs {
		if e.Step == StepSpawn || e.Step == StepReady || e.PID != 0 {
			t.Fatalf("a refused start reached the server: %+v", e)
		}
	}
	res, perr := f.l.Preflight(ctx, req)
	if perr == nil || res.Check.Kind != CheckBlocked || !res.BudgetCap.OK || res.BudgetCap.GiB <= 0 ||
		!strings.Contains(perr.Error(), "a budget of at most") {
		t.Fatalf("refusal: %v %+v %+v", perr, res.Check, res.BudgetCap)
	}
	// the lists come with the check, also with a refusal (U-21: shown when it cannot start)
	if !res.ListsOK || res.Lists.ModifiedMiB != 1536 || res.Lists.FreeZeroMiB != 2048 {
		t.Fatalf("lists %+v %v", res.Lists, res.ListsOK)
	}
	f.l.os.memLists = func() (MemoryLists, error) { return MemoryLists{}, errors.New("PdhCollectQueryData failed") }
	f.mem.AvailPhysMiB = 65536
	if res, err := f.l.Preflight(ctx, req); err != nil || res.ListsOK {
		t.Fatalf("a failed list reading: %v %v", err, res.ListsOK)
	}
	f.l.os.memLists = nil
	if _, ok := f.l.readMemoryLists(); ok {
		t.Fatal("no reader read lists")
	}

	// the «left» figures (F-BU G): free now − need; + modified; «unmeasured» with the need
	ml := MemoryLine{FreeNow: 8192, Modified: 1536, ModifiedOK: true, Need: 6144, NeedLabel: NeedProjected}
	if v, ok := ml.Left(); !ok || v != 2048 {
		t.Fatalf("left %d %v", v, ok)
	}
	if v, ok := ml.LeftIfWrittenFirst(); !ok || v != 3584 {
		t.Fatalf("left if written first %d %v", v, ok)
	}
	ml.ModifiedOK = false
	if _, ok := ml.LeftIfWrittenFirst(); ok {
		t.Fatal("no modified reading gave a value")
	}
	ml.ModifiedOK, ml.NeedLabel = true, NeedUnmeasured
	if _, ok := ml.Left(); ok {
		t.Fatal("an unmeasured need gave a left value")
	}
	if _, ok := ml.LeftIfWrittenFirst(); ok {
		t.Fatal("an unmeasured need gave a left-if-written value")
	}

	// Task Manager: one call, no target; an error comes back
	calls := 0
	f.l.os.taskmgr = func() error { calls++; return nil }
	if err := f.l.OpenTaskManager(); err != nil || calls != 1 {
		t.Fatalf("task manager: %v %d", err, calls)
	}
	f.l.os.taskmgr = func() error { return errors.New("shell refused") }
	if err := f.l.OpenTaskManager(); err == nil {
		t.Fatal("an error was lost")
	}
	f.l.os.taskmgr = nil
	if err := f.l.OpenTaskManager(); err == nil {
		t.Fatal("no opener opened")
	}
}

// TestS8LA23 is S8-LA-23's core side (GL-09; rule 8, LS:234-238; conflict row 8): a full card
// refuses nothing and names no VRAM verdict; a load that runs out of video memory is its own
// failure cause, exit 5.
func TestS8LA23(t *testing.T) {
	f := newFixture(t, false)
	f.l.os.vram = func() VRAMInfo {
		return VRAMInfo{OK: true, Source: "fixture", TotalMiB: 16384, UsedMiB: 16300, FreeMiB: 84}
	}
	res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath})
	if err != nil || res.Check.Kind == CheckBlocked || res.Refusal != nil {
		t.Fatalf("a full card gated the start: %v %+v", err, res.Check)
	}
	for _, s := range []string{res.Check.Reason, res.RAM.Reason, res.RAM.Verdict} {
		if strings.Contains(strings.ToLower(s), "vram") {
			t.Fatalf("a VRAM word in the verdict: %q", s)
		}
	}
	lf, ok := ClassifyLoadFailure([]string{"llama_model_load: loading", "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory"})
	if !ok || lf.Cause != CauseOutOfMemory || lf.Device != "vram" {
		t.Fatalf("load failure %+v %v", lf, ok)
	}
	if ExitCode(StatusFailServerStart) != 5 {
		t.Fatalf("exit %d", ExitCode(StatusFailServerStart))
	}
}
