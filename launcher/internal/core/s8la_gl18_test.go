package core

// GL-18 (version 8): the serving values' core part of S8-LA-37 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md, d1dbe526f; F-LS edition rows 6.9, 6.21, 6.25; review
// items 9, 23, 27; U-36 (i), L:4275). A mock server's stderr (print_timing lines in the engine's
// format, written step by step to a log file) is read through the same serverLog the launcher
// polls while serving: prefill progress, rate and time to first token come from the timing
// lines, the last reply is the eval-time line's whole-answer average, and nothing is written
// beside the log for the samples.

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// mockTimingLines is one request of the mock server, in llama-server b10057's print_timing form
// (the stamp is minutes.seconds.ms.us since the start, as the real log of testdata).
func mockTimingLines(task int, at time.Duration) []string {
	st := func(d time.Duration) string {
		d += at
		m := int(d / time.Minute)
		s := int(d % time.Minute / time.Second)
		ms := int(d % time.Second / time.Millisecond)
		us := int(d % time.Millisecond / time.Microsecond)
		return fmt.Sprintf("%d.%02d.%03d.%03d", m, s, ms, us)
	}
	return []string{
		fmt.Sprintf("%s I slot launch_slot_: id  0 | task %d | processing task, is_child = 0", st(0), task),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d | prompt processing, n_tokens =    100, progress = 0.50, t =   4.00 s / 25.00 tokens per second", st(4*time.Second), task),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d | n_decoded =     30, tg =  10.00 t/s, tg_3s =  10.00 t/s", st(12*time.Second), task),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d | prompt eval time =    8000.00 ms /   200 tokens (   40.00 ms per token,    25.00 tokens per second)", st(20*time.Second), task),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d |        eval time =   12000.00 ms /   120 tokens (  100.00 ms per token,    10.00 tokens per second)", st(20*time.Second), task),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d |       total time =   20000.00 ms /   320 tokens", st(20*time.Second), task),
		fmt.Sprintf("%s I slot      release: id  0 | task %d | stop processing: n_tokens = 319, truncated = 0", st(20*time.Second), task),
	}
}

func appendLines(t *testing.T, path string, lines ...string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, l := range lines {
		if _, err := f.WriteString(l + "\n"); err != nil {
			t.Fatal(err)
		}
	}
}

func dirNames(t *testing.T, dir string) string {
	t.Helper()
	es, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var n []string
	for _, e := range es {
		n = append(n, e.Name())
	}
	return strings.Join(n, ",")
}

// S8-LA-37 (core): --alias carries the short API name; the timing lines give prefill progress,
// rate and TTFT and the last reply's whole-answer average; the samples stay in memory.
func TestS8LA37(t *testing.T) {
	// review item 27: the catalog id, else the file name without its extension; once in the argv
	f := newFixture(t, true)
	res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := argvValue(res.Config.Argv, "--alias"); !ok || v != "fixture-profile" || strings.Count(strings.Join(res.Config.Argv, " "), "--alias") != 1 {
		t.Errorf("--alias in the argv: %v", res.Config.Argv)
	}
	if a := ModelAlias("", `D:\models\Qwen3-30B-A3B-Q4_K_M.gguf`); a != "Qwen3-30B-A3B-Q4_K_M" {
		t.Errorf("outside the catalog: %q", a)
	}

	// review items 9, 23: the mock server's timing lines, read as the launcher reads them
	dir := t.TempDir()
	errLog := filepath.Join(dir, "server_err.log")
	start := time.Date(2026, 10, 12, 10, 0, 0, 0, time.UTC)
	lines := mockTimingLines(7, 0)
	appendLines(t, errLog, lines[:2]...)
	sl := newServerLog(errLog, start)
	s := sl.summary()
	if !s.Prefilling || s.PrefillTokens != 100 || s.PrefillProgress != 0.5 || s.PrefillTokPerSec != 25 || s.Requests != 0 {
		t.Errorf("prefill: %+v", s)
	}
	appendLines(t, errLog, lines[2])
	if s = sl.summary(); s.Prefilling || !s.Decoding || s.DecodeTokens != 30 {
		t.Errorf("decode: %+v", s)
	}
	appendLines(t, errLog, lines[3:]...)
	s = sl.summary()
	last, ok := s.Last()
	if !ok || s.Requests != 1 || last.Task != 7 || last.TTFT != 8*time.Second || last.PromptTokPerSec != 25 ||
		last.PromptTokens != 200 || last.DecodeTokPerSec != 10 || last.AnswerTokens != 120 || !last.At.Equal(start.Add(20*time.Second)) {
		t.Errorf("finished request: %+v %v", last, ok)
	}
	// a second request at 4 tok/s: the last reply is its own average, the run's is token weighted
	second := mockTimingLines(8, time.Minute)
	second[4] = strings.Replace(strings.Replace(second[4], "12000.00 ms", "30000.00 ms", 1), "10.00 tokens per second", " 4.00 tokens per second", 1)
	appendLines(t, errLog, second...)
	s = sl.summary()
	last, _ = s.Last()
	avg, aok := s.RunDecodeAvg()
	if s.Requests != 2 || len(s.Done) != 2 || last.DecodeTokPerSec != 4 || !aok || math.Abs(avg-240.0/42) > 1e-9 {
		t.Errorf("two requests: last %+v, run average %v %v", last, avg, aok)
	}
	// U-36 (i): the samples are held in memory; the only file is the server's own log
	if got := dirNames(t, dir); got != "server_err.log" {
		t.Errorf("files beside the server log: %s", got)
	}
}
