package core

import (
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// The server's stderr log read as data (review items 5, 18, 29, 9, 23, 28 of
// reviews/_gol_review_8_user_20261010.md).
//
// llama-server (common/log.cpp:96-115, b10057) prefixes each line it logs through common_log
// with "<minutes>.<seconds 2>.<ms 3>.<us 3> " (time since the log started, i.e. since the
// process started) and a level letter "I ", "W ", "E " or "D ". Lines the engine writes with
// a plain fprintf (the MoE-Direct patch's "[moe-direct] ..." lines among them) have neither.

// LogLevel is the class of one log line (item 29: four classes).
type LogLevel uint8

const (
	LevelInfo   LogLevel = iota // engine "I"/"D", and unstamped lines with no sign of trouble
	LevelNotice                 // the engine's automatic-adjustment lines ([moe-direct] ... raises x a -> b)
	LevelWarn                   // engine "W"
	LevelError                  // engine "E", and unstamped error lines
)

func (v LogLevel) String() string {
	switch v {
	case LevelNotice:
		return "notice"
	case LevelWarn:
		return "warning"
	case LevelError:
		return "error"
	}
	return "info"
}

// EngineNoticeMarker is the line head this launcher proposes the engine prints for an
// automatic adjustment (item 29; the engine wording is First's engine atom): the launcher keys
// on it first. Until the engine adopts it, the current wording is recognised by
// engineAdjustRe.
const EngineNoticeMarker = "[moe-direct] NOTICE: "

// engineAdjustRe is the engine's adjustment wording today (ggml-moe-direct.cpp:2957, :2990):
// "[moe-direct] ERROR: [moe-direct] mode=virtual raises io_bounce_copy_qd 2 -> 14 (...)".
var engineAdjustRe = regexp.MustCompile(`\[moe-direct\] mode=\S+ raises \S+ -?\d+ -> -?\d+`)

// ServerLine is one parsed stderr line.
type ServerLine struct {
	Time    time.Time     // server start + the engine's stamp; the previous line's time when unstamped
	Elapsed time.Duration // the engine's stamp, or the previous line's
	Stamped bool
	Letter  string // "I", "W", "E", "D" or ""
	Level   LogLevel
	Text    string // the line without the stamp and the letter
	Raw     string
}

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)

// stampRe is "%d.%02d.%03d.%03d " followed by an optional level letter.
var stampRe = regexp.MustCompile(`^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) (?:([IWED]) )?`)

// parseStamp splits the engine's stamp and letter from a line.
func parseStamp(s string) (el time.Duration, letter, rest string, ok bool) {
	m := stampRe.FindStringSubmatch(s)
	if m == nil {
		return 0, "", s, false
	}
	mi, _ := strconv.ParseInt(m[1], 10, 64)
	se, _ := strconv.ParseInt(m[2], 10, 64)
	ms, _ := strconv.ParseInt(m[3], 10, 64)
	us, _ := strconv.ParseInt(m[4], 10, 64)
	el = time.Duration(mi)*time.Minute + time.Duration(se)*time.Second + time.Duration(ms)*time.Millisecond +
		time.Duration(us)*time.Microsecond
	return el, m[5], s[len(m[0]):], true
}

// classify gives a line's level: the engine's own letter first; the notice marker and the
// adjustment wording make a notice whatever the letter; unstamped lines by their wording.
func classify(letter, text string) LogLevel {
	if strings.Contains(text, EngineNoticeMarker) || engineAdjustRe.MatchString(text) {
		return LevelNotice
	}
	switch letter {
	case "W":
		return LevelWarn
	case "E":
		return LevelError
	case "I", "D":
		return LevelInfo
	}
	t := strings.ToLower(text)
	switch {
	case strings.Contains(t, "error") || strings.Contains(t, "failed") || strings.Contains(t, "fatal"):
		return LevelError
	case strings.Contains(t, "warn"):
		return LevelWarn
	}
	return LevelInfo
}

// ParseServerLines parses stderr lines of a server started at start (item 5): clock time =
// start + the engine's stamp; a line without a stamp takes the previous line's time (start for
// the lines before the first stamp).
func ParseServerLines(lines []string, start time.Time) []ServerLine {
	out := make([]ServerLine, 0, len(lines))
	var prev time.Duration
	for _, raw := range lines {
		out = append(out, parseServerLine(raw, start, &prev))
	}
	return out
}

func parseServerLine(raw string, start time.Time, prev *time.Duration) ServerLine {
	s := ansiRe.ReplaceAllString(strings.TrimRight(raw, "\r"), "")
	el, letter, rest, ok := parseStamp(s)
	if ok {
		*prev = el
	} else {
		el = *prev
	}
	return ServerLine{Time: start.Add(el), Elapsed: el, Stamped: ok, Letter: letter, Level: classify(letter, rest),
		Text: rest, Raw: s}
}

// The print_timing lines (item 9, 23, 28) as llama-server b10057 prints them:
//
//	slot print_timing: id  0 | task 64 | prompt processing, n_tokens =     61, progress = 0.66, t =   8.18 s / 7.46 tokens per second
//	slot print_timing: id  0 | task 64 | n_decoded =    100, tg =   5.97 t/s, tg_3s =   5.97 t/s
//	slot print_timing: id  0 | task 64 | prompt eval time =   14333.53 ms /    92 tokens (  155.80 ms per token,     6.42 tokens per second)
//	slot print_timing: id  0 | task 64 |        eval time =  338472.96 ms /  3000 tokens (  112.82 ms per token,     8.86 tokens per second)
var (
	timingHeadRe   = regexp.MustCompile(`print_timing: id +(\d+) \| task +(-?\d+) \| (.*)$`)
	promptProgRe   = regexp.MustCompile(`^prompt processing, n_tokens = +(\d+), progress = +([0-9.]+), t = +([0-9.]+) s / +([0-9.]+) tokens per second`)
	decodeProgRe   = regexp.MustCompile(`^n_decoded = +(\d+), tg = +([0-9.]+) t/s(?:, tg_3s = +([0-9.]+) t/s)?`)
	promptEvalRe   = regexp.MustCompile(`^prompt eval time = +([0-9.]+) ms / +(\d+) tokens \( *[0-9.]+ ms per token, +([0-9.]+) tokens per second\)`)
	evalTimeRe     = regexp.MustCompile(`^eval time = +([0-9.]+) ms / +(\d+) tokens \( *[0-9.]+ ms per token, +([0-9.]+) tokens per second\)`)
	timingTotalRe  = regexp.MustCompile(`^total time = +([0-9.]+) ms / +(\d+) tokens`)
	timingKindName = map[TimingKind]string{TimingPromptProgress: "prompt progress", TimingDecodeProgress: "decode progress",
		TimingPromptEval: "prompt eval", TimingEval: "eval", TimingTotal: "total"}
)

// TimingKind is the kind of a print_timing line.
type TimingKind uint8

const (
	TimingPromptProgress TimingKind = iota + 1 // a prompt is being processed
	TimingDecodeProgress                       // tokens are being generated
	TimingPromptEval                           // the request's whole prompt: TTFT and the prefill rate
	TimingEval                                 // the request's whole answer: the decode average
	TimingTotal
)

func (k TimingKind) String() string { return timingKindName[k] }

// Timing is one print_timing line.
type Timing struct {
	Kind      TimingKind
	Slot      int
	Task      int
	Tokens    int64         // n_tokens so far, n_decoded, or the tokens of the eval line
	Progress  float64       // prompt progress 0..1
	Duration  time.Duration // t (prompt progress), or the eval time
	TokPerSec float64       // the rate on the line (tg for decode progress)
	Recent    float64       // tg_3s
}

// ParseTiming reads a print_timing line's text (with or without the stamp).
func ParseTiming(text string) (Timing, bool) {
	m := timingHeadRe.FindStringSubmatch(text)
	if m == nil {
		return Timing{}, false
	}
	t := Timing{}
	t.Slot, _ = strconv.Atoi(m[1])
	t.Task, _ = strconv.Atoi(m[2])
	body := strings.TrimSpace(m[3])
	f := func(s string) float64 { v, _ := strconv.ParseFloat(s, 64); return v }
	n := func(s string) int64 { v, _ := strconv.ParseInt(s, 10, 64); return v }
	ms := func(s string) time.Duration { return time.Duration(f(s) * float64(time.Millisecond)) }
	if x := promptProgRe.FindStringSubmatch(body); x != nil {
		t.Kind, t.Tokens, t.Progress = TimingPromptProgress, n(x[1]), f(x[2])
		t.Duration, t.TokPerSec = time.Duration(f(x[3])*float64(time.Second)), f(x[4])
		return t, true
	}
	if x := decodeProgRe.FindStringSubmatch(body); x != nil {
		t.Kind, t.Tokens, t.TokPerSec = TimingDecodeProgress, n(x[1]), f(x[2])
		if x[3] != "" {
			t.Recent = f(x[3])
		}
		return t, true
	}
	if x := promptEvalRe.FindStringSubmatch(body); x != nil {
		t.Kind, t.Duration, t.Tokens, t.TokPerSec = TimingPromptEval, ms(x[1]), n(x[2]), f(x[3])
		return t, true
	}
	if x := evalTimeRe.FindStringSubmatch(body); x != nil {
		t.Kind, t.Duration, t.Tokens, t.TokPerSec = TimingEval, ms(x[1]), n(x[2]), f(x[3])
		return t, true
	}
	if x := timingTotalRe.FindStringSubmatch(body); x != nil {
		t.Kind, t.Duration, t.Tokens = TimingTotal, ms(x[1]), n(x[2])
		return t, true
	}
	return Timing{}, false
}

// RequestTiming is one finished request (items 9, 23).
type RequestTiming struct {
	Task            int
	At              time.Time
	PromptTokens    int64
	TTFT            time.Duration // the prompt eval time (prompt processed = first token ready)
	PromptTokPerSec float64
	AnswerTokens    int64
	AnswerTime      time.Duration
	DecodeTokPerSec float64 // the whole answer's average (the eval time line)
}

// TimingSummary is what the print_timing lines tell so far.
type TimingSummary struct {
	// The request in flight, if any.
	Prefilling       bool
	PrefillTask      int
	PrefillProgress  float64
	PrefillTokens    int64
	PrefillTokPerSec float64
	PrefillFor       time.Duration
	Decoding         bool
	DecodeTask       int
	DecodeTokens     int64
	DecodeTokPerSec  float64 // tg: the request's running average
	DecodeRecent     float64 // tg_3s
	// Finished requests, oldest first (at most timingKeep).
	Done []RequestTiming
	// Sums over every finished request of the run (item 28: the run's decode average).
	Requests     int
	AnswerTokens int64
	AnswerTime   time.Duration
	PromptTokens int64
	PromptTime   time.Duration
	LastActivity time.Time
}

// RunDecodeAvg is the run's decode average, answer tokens over answer time (token weighted).
func (s TimingSummary) RunDecodeAvg() (float64, bool) {
	if s.AnswerTime <= 0 || s.AnswerTokens <= 0 {
		return 0, false
	}
	return float64(s.AnswerTokens) / s.AnswerTime.Seconds(), true
}

// Last is the newest finished request.
func (s TimingSummary) Last() (RequestTiming, bool) {
	if len(s.Done) == 0 {
		return RequestTiming{}, false
	}
	return s.Done[len(s.Done)-1], true
}

const timingKeep = 64

// timingTracker folds parsed lines into a TimingSummary.
type timingTracker struct {
	sum     TimingSummary
	pending map[int]*RequestTiming // by task, between its prompt eval and eval lines
}

func (t *timingTracker) add(ln ServerLine) {
	tm, ok := ParseTiming(ln.Text)
	if !ok {
		if strings.Contains(ln.Text, "stop processing:") || strings.Contains(ln.Text, "release:") {
			t.sum.Prefilling, t.sum.Decoding = false, false
		}
		return
	}
	if t.pending == nil {
		t.pending = map[int]*RequestTiming{}
	}
	s := &t.sum
	s.LastActivity = ln.Time
	switch tm.Kind {
	case TimingPromptProgress:
		s.Prefilling, s.Decoding = true, false
		s.PrefillTask, s.PrefillProgress, s.PrefillTokens = tm.Task, tm.Progress, tm.Tokens
		s.PrefillTokPerSec, s.PrefillFor = tm.TokPerSec, tm.Duration
	case TimingDecodeProgress:
		s.Prefilling, s.Decoding = false, true
		s.DecodeTask, s.DecodeTokens, s.DecodeTokPerSec, s.DecodeRecent = tm.Task, tm.Tokens, tm.TokPerSec, tm.Recent
	case TimingPromptEval:
		t.pending[tm.Task] = &RequestTiming{Task: tm.Task, At: ln.Time, PromptTokens: tm.Tokens, TTFT: tm.Duration,
			PromptTokPerSec: tm.TokPerSec}
	case TimingEval:
		r := t.pending[tm.Task]
		if r == nil {
			r = &RequestTiming{Task: tm.Task}
		}
		delete(t.pending, tm.Task)
		r.At, r.AnswerTokens, r.AnswerTime, r.DecodeTokPerSec = ln.Time, tm.Tokens, tm.Duration, tm.TokPerSec
		s.Done = append(s.Done, *r)
		if len(s.Done) > timingKeep {
			s.Done = s.Done[len(s.Done)-timingKeep:]
		}
		s.Requests++
		s.AnswerTokens += r.AnswerTokens
		s.AnswerTime += r.AnswerTime
		s.PromptTokens += r.PromptTokens
		s.PromptTime += r.TTFT
		s.Prefilling, s.Decoding = false, false
	}
}

// SummarizeTimings folds a whole parsed log.
func SummarizeTimings(lines []ServerLine) TimingSummary {
	var t timingTracker
	for _, ln := range lines {
		t.add(ln)
	}
	return t.sum
}

// serverLogKeep is how many parsed lines a reader keeps.
const serverLogKeep = 5000

// serverLog reads a growing stderr log incrementally: complete lines only, parsed once, the
// newest serverLogKeep kept, the timing summary folded as they come.
type serverLog struct {
	mu      sync.Mutex
	path    string
	start   time.Time
	off     int64
	partial string
	prev    time.Duration
	lines   []ServerLine
	timing  timingTracker
}

func newServerLog(path string, start time.Time) *serverLog {
	return &serverLog{path: path, start: start}
}

// poll reads what was appended since the last call.
func (r *serverLog) poll() {
	r.mu.Lock()
	defer r.mu.Unlock()
	f, err := os.Open(r.path)
	if err != nil {
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || st.Size() <= r.off {
		return
	}
	b := make([]byte, st.Size()-r.off)
	n, err := f.ReadAt(b, r.off)
	if err != nil && err != io.EOF {
		return
	}
	r.off += int64(n)
	text := r.partial + string(b[:n])
	idx := strings.LastIndex(text, "\n")
	if idx < 0 {
		r.partial = text
		return
	}
	r.partial = text[idx+1:]
	for _, raw := range strings.Split(text[:idx], "\n") {
		ln := parseServerLine(raw, r.start, &r.prev)
		r.timing.add(ln)
		r.lines = append(r.lines, ln)
	}
	if len(r.lines) > serverLogKeep {
		r.lines = append([]ServerLine(nil), r.lines[len(r.lines)-serverLogKeep:]...)
	}
}

// tail is the newest n parsed lines, oldest first.
func (r *serverLog) tail(n int) []ServerLine {
	r.poll()
	r.mu.Lock()
	defer r.mu.Unlock()
	if n <= 0 || n > len(r.lines) {
		n = len(r.lines)
	}
	return append([]ServerLine(nil), r.lines[len(r.lines)-n:]...)
}

// summary is the timing summary so far.
func (r *serverLog) summary() TimingSummary {
	r.poll()
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.timing.sum
	s.Done = append([]RequestTiming(nil), s.Done...)
	return s
}
