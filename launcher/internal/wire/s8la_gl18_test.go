package wire

// GL-18 (version 8): the adapter's part of S8-LA-37 (plan reviews/_plan_8_STAGE1_v8-plansplit.md,
// d1dbe526f; F-LS edition rows 6.9, 6.21, 6.25, 6.26; review items 9, 23, 27, 28; U-36 (i)): the
// serve page's tiles against a mock server's timing lines, the live decode as a 5 s window, the
// per-request samples of the graph switch, the API name, and the models page speed tile.

import (
	"fmt"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// mockServerLines is the mock server's stderr for one request (llama-server b10057's print_timing
// form): prompt progress at 50 %, a decode progress line, then the prompt-eval, eval and total lines.
// The engine prints a decode progress line only from 100 decoded tokens on (server-context.cpp
// print_timings_tg; UX-R7 addendum, finding 20 (a)): the 120-token answer, decoding from 8 s,
// reaches 100 tokens at 8 s + 100/tg; the line is written then when that is before the answer's
// end at 20 s (10 tok/s: at 18 s), and not at all otherwise (4 tok/s).
func mockServerLines(task int, at time.Duration, evalMs float64) []string {
	st := func(d time.Duration) string {
		d += at
		return fmt.Sprintf("%d.%02d.%03d.%03d", int(d/time.Minute), int(d%time.Minute/time.Second), int(d%time.Second/time.Millisecond), 0)
	}
	tg := 120 / (evalMs / 1000)
	ls := []string{
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d | prompt processing, n_tokens =    100, progress = 0.50, t =   4.00 s / 25.00 tokens per second", st(4*time.Second), task),
	}
	if at100 := 8*time.Second + time.Duration(100/tg*float64(time.Second)); at100 <= 20*time.Second {
		ls = append(ls, fmt.Sprintf("%s I slot print_timing: id  0 | task %d | n_decoded =    100, tg = %6.2f t/s, tg_3s = %6.2f t/s", st(at100), task, tg, tg))
	}
	return append(ls,
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d | prompt eval time =    8000.00 ms /   200 tokens (   40.00 ms per token,    25.00 tokens per second)", st(20*time.Second), task),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d |        eval time =   %.2f ms /   120 tokens (  100.00 ms per token,    %.2f tokens per second)", st(20*time.Second), task, evalMs, tg),
		fmt.Sprintf("%s I slot print_timing: id  0 | task %d |       total time =   20000.00 ms /   320 tokens", st(20*time.Second), task),
	)
}

// S8-LA-37 (adapter).
func TestS8LA37(t *testing.T) {
	spawn := time.Date(2026, 10, 12, 10, 0, 0, 0, time.UTC)
	now := spawn.Add(30 * time.Second)
	lines := append(mockServerLines(7, 0, 12000), mockServerLines(8, time.Minute, 30000)...)
	sum := func(n int) core.TimingSummary { return core.SummarizeTimings(core.ParseServerLines(lines[:n], spawn)) }
	st := core.ServeStatus{Running: true, Ready: true, Alive: true, Port: 8093, Address: "http://127.0.0.1:8093",
		Start: core.StartSettings{Alias: "fixture-profile"}}

	// review item 9: while the prompt is read, progress, rate and the TTFT estimate's inputs
	st.Timing = sum(1)
	s := statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, nil, now, true)
	if s.Phase != ui.ReqPrefill || s.PrefillDone.V != 100 || s.PrefillTotal.V != 200 || s.PrefillTokS.V != 25 {
		t.Errorf("prefill tiles: phase %v done %v total %v rate %v", s.Phase, s.PrefillDone, s.PrefillTotal, s.PrefillTokS)
	}
	// review item 23 with finding 20 (a) (UX-R11; UX-R7 addendum): below 100 decoded tokens the
	// engine prints no decode progress line, so after the last prompt progress line the log still
	// reads «prefilling» while the answer decodes; the slot that is decoding wins: the phase is
	// decode and the live window gives its rate (10 tokens a second)
	slot := func(n int) any {
		return []any{map[string]any{"id": 0.0, "id_task": 7.0, "is_processing": true, "n_prompt_tokens": 200.0,
			"next_token": []any{map[string]any{"n_decoded": float64(n)}}}}
	}
	var dw decodeWindow
	st.Slots = core.HTTPRead{OK: true, Code: 200, Body: slot(10)}
	statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &dw, spawn.Add(9*time.Second), true)
	st.Slots.Body = slot(20)
	s = statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &dw, spawn.Add(10*time.Second), true)
	if s.Phase != ui.ReqDecode || !s.DecodeTokS.Ok() || s.DecodeTokS.V != 10 {
		t.Errorf("a decoding slot under the log's last prompt progress line: phase %v decode %v", s.Phase, s.DecodeTokS)
	}
	st.Slots = core.HTTPRead{}
	// review items 9, 23, 27: after the reply, TTFT from the prompt-eval line, the last reply from
	// the eval-time line, the API name once
	st.Timing = sum(5)
	s = statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, nil, now, false)
	near(t, "TTFT", s.TTFTs, 8)
	near(t, "last reply", s.LastDecodeTokS, 10)
	near(t, "prefill rate", s.PrefillTokS, 25)
	near(t, "requests", s.Requests, 1)
	if s.Alias != "fixture-profile" {
		t.Errorf("API name %q", s.Alias)
	}
	// the graph switch's samples: one per finished request, in order, with its own figures
	st.Timing = sum(len(lines))
	s = statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, nil, now, false)
	if len(s.Reqs) != 2 || s.Reqs[0].Task != 7 || s.Reqs[1].Task != 8 || s.Reqs[0].TTFTs != 8 || s.Reqs[0].PrefillTokS != 25 ||
		s.Reqs[0].DecodeTokS != 10 || s.Reqs[1].DecodeTokS != 4 || !s.Reqs[1].At.After(s.Reqs[0].At) {
		t.Errorf("per-request samples %+v", s.Reqs)
	}
	near(t, "last reply (second)", s.LastDecodeTokS, 4)
	near(t, "usual = the run's whole-answer average", s.UsualTokS, 240.0/42)

	// review item 23: live decode = the last 5 s of the slot's decoded count
	var w decodeWindow
	n := int64(0)
	for i := 0; i <= 10; i++ {
		rate := int64(10)
		if i > 5 {
			rate = 2
		}
		if i > 0 {
			n += rate
		}
		w.add(spawn.Add(time.Duration(i)*time.Second), 9, n, true)
	}
	if r, ok := w.rate(spawn.Add(10 * time.Second)); !ok || r != 2 {
		t.Errorf("5 s window %v %v", r, ok)
	}

	// review item 28: the models page tile is the last run's average with its date
	ml := &core.ModelList{TotalRAMMiB: 65536, Files: []core.ModelFile{{Path: `D:\m\a\a.gguf`, Name: "a.gguf", Folder: `D:\m`,
		Catalog: true, ProfileID: "a", DisplayName: "A", Stats: &core.ModelStats{DecodeTokPerSec: 8.86, At: "2026-10-09T12:23:06Z"}}}}
	c := catalogFrom(ml, "models.json", 20480, true, core.VRAMInfo{}, nil)
	if len(c.Models) != 1 || c.Models[0].DecodeTokS.V != 8.86 || c.Models[0].MeasuredOn != dateOf("2026-10-09T12:23:06Z") || c.Models[0].MeasuredOn == "" {
		t.Errorf("speed tile %+v %q", c.Models[0].DecodeTokS, c.Models[0].MeasuredOn)
	}
}
