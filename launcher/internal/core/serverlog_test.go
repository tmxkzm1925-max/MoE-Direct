package core

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The fixtures are real stderr logs of llama-server b10057 on this PC, copied unchanged from
// C:\Users\tmxkz\AppData\Local\MoE-Direct\go-draft\logs\ (GOL-WIRE live starts):
//   - server_20261009T175949Z_err.log: gpt-oss-120b, virtual repack, one request of 92 prompt
//     tokens and 3000 answer tokens, then the stop; spawned at 2026-10-09T17:59:49.7389571Z
//     (CHILD_START of launcher_20261009T174905Z_19160.jsonl).
//   - server_20261009T171115Z_err.log: a start and a stop with no request.

var fixtureSpawn = time.Date(2026, 10, 9, 17, 59, 49, 738957100, time.UTC)

func fixtureLines(t *testing.T, name string) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimRight(string(b), "\n"), "\n")
}

func TestParseServerLinesRealLog(t *testing.T) {
	raw := fixtureLines(t, "server_20261009T175949Z_err.log")
	lines := ParseServerLines(raw, fixtureSpawn)
	if len(lines) != 141 {
		t.Fatalf("lines %d", len(lines))
	}
	at := func(i int) ServerLine { return lines[i-1] } // 1-based like the file
	// Clock time = spawn + the stamp "minutes.seconds.ms.us".
	if got, want := at(1).Time, fixtureSpawn.Add(89177*time.Microsecond); !got.Equal(want) || !at(1).Stamped || at(1).Letter != "I" {
		t.Errorf("line 1: %v want %v", got, want)
	}
	if got, want := at(141).Time, fixtureSpawn.Add(23*time.Minute+16*time.Second+190184*time.Microsecond); !got.Equal(want) {
		t.Errorf("line 141: %v want %v", got, want)
	}
	if at(141).Time.Format("15:04:05") != "18:23:05" {
		t.Errorf("line 141 clock %s", at(141).Time.Format("15:04:05.000"))
	}
	// Unstamped lines take the previous line's time.
	for _, i := range []int{16, 17, 18, 19, 20} {
		if at(i).Stamped || !at(i).Time.Equal(at(15).Time) || at(i).Letter != "" {
			t.Errorf("line %d: stamped %v time %v (line 15 %v)", i, at(i).Stamped, at(i).Time, at(15).Time)
		}
	}
	// The level from the engine's letter; the adjustment line is a notice despite its ERROR prefix.
	want := map[int]LogLevel{1: LevelInfo, 14: LevelWarn, 15: LevelWarn, 16: LevelInfo, 18: LevelNotice, 20: LevelInfo,
		26: LevelWarn, 28: LevelInfo, 136: LevelInfo, 141: LevelInfo}
	for i, lv := range want {
		if at(i).Level != lv {
			t.Errorf("line %d level %s want %s: %q", i, at(i).Level, lv, at(i).Raw)
		}
	}
	count := map[LogLevel]int{}
	for _, l := range lines {
		count[l.Level]++
	}
	if count[LevelNotice] != 1 || count[LevelError] != 0 || count[LevelWarn] != 14 {
		t.Errorf("levels %v", count)
	}
	if !strings.HasPrefix(at(26).Text, "srv          stop: cancel task") {
		t.Errorf("text without the stamp: %q", at(26).Text)
	}
}

func TestClassifyNoticeMarker(t *testing.T) {
	cases := []struct {
		line string
		want LogLevel
	}{
		{"[moe-direct] NOTICE: mode=virtual raises io_bounce_copy_qd 2 -> 14 (record fan-out shape floor)", LevelNotice},
		{"0.00.001.000 E [moe-direct] NOTICE: budget lowered", LevelNotice},
		{"[moe-direct] ERROR: [moe-direct] mode=virtual raises io_bounce_slots 9 -> 24 (record fan-out shape floor)", LevelNotice},
		{"[moe-direct] ERROR: manifest unreadable", LevelError},
		{"0.00.001.000 E srv  load_model: failed to load", LevelError},
		{"0.00.001.000 W srv  something", LevelWarn},
		{"0.00.001.000 D srv  debug", LevelInfo},
		{"plain warning: x", LevelWarn},
		{"\x1b[31m0.00.001.000 E red\x1b[0m", LevelError},
	}
	for _, c := range cases {
		got := ParseServerLines([]string{c.line}, fixtureSpawn)[0]
		if got.Level != c.want {
			t.Errorf("%q: %s want %s", c.line, got.Level, c.want)
		}
	}
}

func TestParseTimingRealLines(t *testing.T) {
	raw := fixtureLines(t, "server_20261009T175949Z_err.log")
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	tm, ok := ParseTiming(raw[27]) // line 28
	if !ok || tm.Kind != TimingPromptProgress || tm.Slot != 0 || tm.Task != 64 || tm.Tokens != 61 ||
		!near(tm.Progress, 0.66) || tm.Duration != 8180*time.Millisecond || !near(tm.TokPerSec, 7.46) {
		t.Errorf("line 28: %+v %v", tm, ok)
	}
	tm, ok = ParseTiming(raw[30]) // line 31
	if !ok || tm.Kind != TimingDecodeProgress || tm.Tokens != 100 || !near(tm.TokPerSec, 5.97) || !near(tm.Recent, 5.97) {
		t.Errorf("line 31: %+v %v", tm, ok)
	}
	tm, ok = ParseTiming(raw[135]) // line 136
	if !ok || tm.Kind != TimingPromptEval || tm.Tokens != 92 || tm.Duration != 14333530*time.Microsecond || !near(tm.TokPerSec, 6.42) {
		t.Errorf("line 136: %+v %v", tm, ok)
	}
	tm, ok = ParseTiming(raw[136]) // line 137
	if !ok || tm.Kind != TimingEval || tm.Tokens != 3000 || tm.Duration != 338472960*time.Microsecond || !near(tm.TokPerSec, 8.86) {
		t.Errorf("line 137: %+v %v", tm, ok)
	}
	tm, ok = ParseTiming(raw[137]) // line 138
	if !ok || tm.Kind != TimingTotal || tm.Tokens != 3092 || tm.Duration != 352806490*time.Microsecond {
		t.Errorf("line 138: %+v %v", tm, ok)
	}
	for _, i := range []int{139, 140, 21} { // graphs reused, release, a non-timing line
		if tm, ok := ParseTiming(raw[i-1]); ok {
			t.Errorf("line %d parsed as timing: %+v", i, tm)
		}
	}
}

func TestSummarizeTimingsRealLog(t *testing.T) {
	lines := ParseServerLines(fixtureLines(t, "server_20261009T175949Z_err.log"), fixtureSpawn)
	// While the prompt is processed (after line 30).
	s := SummarizeTimings(lines[:30])
	if !s.Prefilling || s.Decoding || s.PrefillTokens != 88 || s.PrefillProgress != 0.96 || s.PrefillTokPerSec != 6.51 ||
		s.PrefillFor != 13520*time.Millisecond || s.Requests != 0 {
		t.Errorf("prefill: %+v", s)
	}
	// While the answer is generated (after line 34).
	s = SummarizeTimings(lines[:34])
	if s.Prefilling || !s.Decoding || s.DecodeTokens != 179 || s.DecodeTokPerSec != 6.93 || s.DecodeRecent != 10.43 {
		t.Errorf("decode: %+v", s)
	}
	// The whole log: one finished request.
	s = SummarizeTimings(lines)
	if s.Prefilling || s.Decoding || s.Requests != 1 || len(s.Done) != 1 {
		t.Fatalf("done: %+v", s)
	}
	r := s.Done[0]
	if r.Task != 64 || r.PromptTokens != 92 || r.TTFT != 14333530*time.Microsecond || r.PromptTokPerSec != 6.42 ||
		r.AnswerTokens != 3000 || r.DecodeTokPerSec != 8.86 || !r.At.Equal(lines[136].Time) {
		t.Errorf("request: %+v", r)
	}
	avg, ok := s.RunDecodeAvg()
	if !ok || math.Abs(avg-3000/338.47296) > 1e-9 {
		t.Errorf("run average %v %v", avg, ok)
	}
	if last, ok := s.Last(); !ok || last.Task != 64 {
		t.Errorf("last %+v", last)
	}
	// A run without a request has no average.
	s = SummarizeTimings(ParseServerLines(fixtureLines(t, "server_20261009T171115Z_err.log"), fixtureSpawn))
	if _, ok := s.RunDecodeAvg(); ok || s.Requests != 0 || s.Prefilling || s.Decoding {
		t.Errorf("no request: %+v", s)
	}
}

func TestServerLogIncremental(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "server_20261009T175949Z_err.log"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "server_err.log")
	cut := strings.Index(string(src), "prompt eval time") // inside line 136
	if err := os.WriteFile(path, src[:cut], 0o644); err != nil {
		t.Fatal(err)
	}
	r := newServerLog(path, fixtureSpawn)
	if got := r.tail(0); len(got) != 135 {
		t.Fatalf("first read %d lines (the cut line must wait)", len(got))
	}
	if s := r.summary(); !s.Decoding || s.DecodeTokens != 2967 {
		t.Errorf("mid summary %+v", s)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	f.Write(src[cut:])
	f.Close()
	got := r.tail(0)
	want := ParseServerLines(fixtureLines(t, "server_20261009T175949Z_err.log"), fixtureSpawn)
	if len(got) != len(want) {
		t.Fatalf("after append %d lines", len(got))
	}
	for i := range want {
		if got[i].Raw != want[i].Raw || !got[i].Time.Equal(want[i].Time) || got[i].Level != want[i].Level {
			t.Fatalf("line %d differs: %+v vs %+v", i+1, got[i], want[i])
		}
	}
	if tail := r.tail(3); len(tail) != 3 || tail[2].Raw != want[140].Raw {
		t.Errorf("tail 3: %+v", tail)
	}
	if s := r.summary(); s.Requests != 1 || s.Decoding {
		t.Errorf("final summary %+v", s)
	}
	// A missing file yields nothing and no panic.
	if got := newServerLog(filepath.Join(t.TempDir(), "none.log"), fixtureSpawn).tail(5); len(got) != 0 {
		t.Errorf("missing file %d lines", len(got))
	}
}
