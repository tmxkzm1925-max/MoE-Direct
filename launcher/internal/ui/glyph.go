// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// Decision ④ (DECISION_LEDGER.md:4181): glyph width is measured at every
// start. The launcher prints each glyph that a console font may draw two cells
// wide, asks the cursor position after it (CSI 6n), and uses the fallback set
// when any of them took two cells or when no answer came within a short wait.
// Two sets are fixed in code; no per-PC value is stored. GL-03 builds the
// measurement to GLP steps 1-7 (below) and writes its start-log record.

package ui

import (
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// GlyphSet is one of the two glyph sets.
type GlyphSet uint8

const (
	GlyphDesign   GlyphSet = iota // the design's glyphs: the primary set
	GlyphFallback                 // one-cell replacements
)

func (g GlyphSet) String() string {
	if g == GlyphFallback {
		return "fallback"
	}
	return "primary" // the scenarios' name (S8-LA-04 to -06)
}

// fallbackPairs: design glyph, its one-cell replacement. The left side is
// every glyph the feasibility readout measured at two cells in this PC's
// console font 굴림체 (· … ← → ● ○ ■ ▲ ▶ ◐; LAUNCH-FEAS Q2.3, C029) plus the
// unprobed arrows and signs the screens use (◀ ↑ ↓ ± ≥; LAUNCH-FEAS §10 item
// 15). The right side is ASCII or a glyph measured at one cell in both fonts
// (▸). Each replacement is one cell, so a screen keeps its alignment.
var fallbackPairs = []string{
	"·", ".", "…", "~", "←", "<", "→", ">", "●", "*", "○", "o", "■", "#",
	"▲", "!", "▶", "▸", "◐", "+", "◀", "<", "↑", "^", "↓", "v", "±", "+", "≥", ">",
	"★", "*", // the catalog mark (FIX-UI, review items 11, 12): East Asian ambiguous width
	"◇", "o", // the structure-defaults mark (FIX2-UI, conflict 14)
}

var fallbackReplacer = strings.NewReplacer(fallbackPairs...)

// ProbeGlyphs is the list of glyphs the probe prints.
func ProbeGlyphs() []string {
	var gs []string
	for i := 0; i < len(fallbackPairs); i += 2 {
		gs = append(gs, fallbackPairs[i])
	}
	return gs
}

// Apply rewrites a rendered screen into this set.
func (g GlyphSet) Apply(s string) string {
	if g == GlyphFallback {
		return fallbackReplacer.Replace(s)
	}
	return s
}

// ---------- the glyph self-measurement (GL-03; GLP steps 1-7) ----------
//
// The measurement runs in the launcher's own code before the UI library's
// Run (GLP §4: Bubble Tea v1 has no cursor-report decoding, and its reader
// would take a late reply as keys). Steps, by GLP:124-155:
//  1. read the console's input and output modes and keep them;
//  2. output mode |= VT processing and DISABLE_NEWLINE_AUTO_RETURN; a failure
//     means no reply can come: the fallback set at once, no wait;
//  3. input mode = EXTENDED_FLAGS|WINDOW_INPUT|VT input (0x0288), so that a
//     Windows Terminal reply arrives as characters, not as F3 (G010 A vs B),
//     and processed input is off (no ctrl+c signal during the probe);
//  4. one batched write of every glyph query (CR, erase line, glyph, CSI 6n),
//     then reads under one deadline, ProbeWait (M-13 (c)); width = column-1;
//  5. the user's key records that arrive meanwhile are kept: a reply record is
//     a key record with virtual key 0 (G009 A1, G010 A1: firstVK=0x0), a key
//     the user typed carries its virtual key;
//  6. when not every reply came before the deadline, the input is drained
//     until it stays quiet for ProbeQuiet, so a late reply never reaches the
//     UI as keys;
//  7. the probe line is erased, both modes are restored, and only then the
//     kept key records are written back to the input, once, in order.

// Console mode bits the measurement sets (Windows console API values).
const (
	modeVTProcessing  uint32 = 0x0004                                            // ENABLE_VIRTUAL_TERMINAL_PROCESSING (output)
	modeNoAutoReturn  uint32 = 0x0008                                            // DISABLE_NEWLINE_AUTO_RETURN (output)
	modeWindowInput   uint32 = 0x0008                                            // ENABLE_WINDOW_INPUT (input)
	modeExtendedFlags uint32 = 0x0080                                            // ENABLE_EXTENDED_FLAGS (input)
	modeVTInput       uint32 = 0x0200                                            // ENABLE_VIRTUAL_TERMINAL_INPUT (input)
	probeInMode              = modeExtendedFlags | modeWindowInput | modeVTInput // 0x0288 (GLP step 3)
	probeOutBits             = modeVTProcessing | modeNoAutoReturn
)

// ConRecord is one console input record as the measurement sees it.
type ConRecord struct {
	Key  bool   // a key record (other records: window, focus, menu, mouse)
	Down bool   // key pressed (false: released)
	VK   uint16 // virtual key; 0 in the console's reply records
	Char rune
	Raw  any // the record as the console gave it, written back unchanged
}

// probeConsole is the console seen by the measurement: its modes, the query
// write, the record read under a deadline, and the write-back of kept keys.
type probeConsole interface {
	Modes() (in, out uint32, err error)
	SetIn(mode uint32) error
	SetOut(mode uint32) error
	Write(p []byte) (int, error)
	// Read returns the records that arrived before the deadline; none means the
	// deadline passed.
	Read(deadline time.Time) ([]ConRecord, error)
	// Forward writes key records back to the console input.
	Forward(recs []ConRecord) error
}

// ProbeResult is what the measurement saw; it is the start-log record.
type ProbeResult struct {
	Set       GlyphSet
	Widths    []int // cells per probed glyph, in ProbeGlyphs order; -1 = no answer
	Answers   int   // replies read before the deadline
	Reason    string
	Wait      time.Duration // the configured wait (ProbeWait)
	Elapsed   time.Duration // from the start of the measurement to the restored modes
	InMode    uint32        // the input mode set during the measurement (0 when none was set)
	Forwarded int           // user key records written back
	Dropped   int           // reply records drained after the deadline
}

// ProbeWait is the one deadline for all replies (M-13 (c): the base value;
// GLP:168-169, :259: not verified on a remote or nested host).
const ProbeWait = 400 * time.Millisecond

// ProbeQuiet is the quiet period that ends the drain after a missed deadline
// (GLP step 6); the drain never runs longer than the wait.
const ProbeQuiet = 50 * time.Millisecond

// runProbe is the measurement over a console (GLP steps 1-7, see above).
func runProbe(c probeConsole, glyphs []string, wait, quiet time.Duration, now func() time.Time) ProbeResult {
	t0 := now()
	res := ProbeResult{Set: GlyphFallback, Widths: make([]int, len(glyphs)), Wait: wait}
	for i := range res.Widths {
		res.Widths[i] = -1
	}
	done := func(r ProbeResult) ProbeResult { r.Elapsed = now().Sub(t0); return r }
	// 1
	in0, out0, err := c.Modes()
	if err != nil {
		res.Reason = "no console: " + err.Error()
		return done(res)
	}
	// 2
	if err := c.SetOut(out0 | probeOutBits); err != nil {
		_ = c.SetOut(out0)
		res.Reason = "no VT output: " + err.Error()
		return done(res)
	}
	// 3
	if err := c.SetIn(probeInMode); err != nil {
		_ = c.SetIn(in0)
		_ = c.SetOut(out0)
		res.Reason = "no VT input: " + err.Error()
		return done(res)
	}
	res.InMode = probeInMode
	restore := func() {
		_ = c.SetIn(in0)
		_ = c.SetOut(out0)
	}
	// 4
	var q strings.Builder
	for _, g := range glyphs {
		q.WriteString("\r\x1b[2K" + g + "\x1b[6n")
	}
	q.WriteString("\r\x1b[2K")
	if _, err := c.Write([]byte(q.String())); err != nil {
		restore()
		res.Reason = "write failed: " + err.Error()
		return done(res)
	}
	deadline := t0.Add(wait)
	var inTime []ConRecord
	for len(replyCols(inTime)) < len(glyphs) && now().Before(deadline) {
		rs, err := c.Read(deadline)
		inTime = append(inTime, rs...)
		if err != nil {
			break
		}
	}
	// 6
	var late []ConRecord
	if len(replyCols(inTime)) < len(glyphs) {
		stop := now().Add(wait)
		for {
			until := now().Add(quiet)
			if until.After(stop) {
				until = stop
			}
			rs, err := c.Read(until)
			late = append(late, rs...)
			if len(rs) == 0 || err != nil || !now().Before(stop) {
				break
			}
		}
	}
	// 7
	_, _ = c.Write([]byte("\r\x1b[2K"))
	restore()
	keep := append(userKeys(inTime), userKeys(late)...)
	if len(keep) > 0 && c.Forward(keep) == nil {
		res.Forwarded = len(keep)
	}
	res.Dropped = len(late) - len(userKeys(late))
	cols := replyCols(inTime)
	res.Answers = len(cols)
	for i := 0; i < len(glyphs) && i < len(cols); i++ {
		res.Widths[i] = cols[i] - 1
	}
	return done(decide(res, glyphs))
}

// isReply: a record of the console's own reply (a key record with virtual key 0).
func isReply(r ConRecord) bool { return r.Key && r.VK == 0 }

// replyCols parses the cursor replies out of the reply records, in order.
func replyCols(recs []ConRecord) []int {
	var b []byte
	for _, r := range recs {
		if isReply(r) && r.Down && r.Char != 0 {
			b = utf8.AppendRune(b, r.Char)
		}
	}
	return parseCPR(b)
}

// userKeys: the key records the user typed (virtual key set), in order.
func userKeys(recs []ConRecord) []ConRecord {
	var out []ConRecord
	for _, r := range recs {
		if r.Key && !isReply(r) {
			out = append(out, r)
		}
	}
	return out
}

func decide(res ProbeResult, glyphs []string) ProbeResult {
	if res.Answers < len(glyphs) {
		res.Set = GlyphFallback
		res.Reason = "no answer within the wait (" + strconv.Itoa(res.Answers) + " of " + strconv.Itoa(len(glyphs)) + ")"
		return res
	}
	for i, w := range res.Widths {
		if w != 1 {
			res.Set = GlyphFallback
			res.Reason = glyphs[i] + " took " + strconv.Itoa(w) + " cells"
			return res
		}
	}
	res.Set = GlyphDesign
	res.Reason = "every glyph took one cell"
	return res
}

// parseCPR returns the column of every "ESC[<row>;<col>R" in b, in order.
func parseCPR(b []byte) []int {
	var cols []int
	s := string(b)
	for {
		i := strings.Index(s, "\x1b[")
		if i < 0 {
			return cols
		}
		s = s[i+2:]
		j := strings.IndexByte(s, 'R')
		if j < 0 {
			return cols
		}
		body := s[:j]
		semi := strings.IndexByte(body, ';')
		if semi > 0 {
			if _, err := strconv.Atoi(body[:semi]); err == nil {
				if c, err := strconv.Atoi(body[semi+1:]); err == nil {
					cols = append(cols, c)
					s = s[j+1:]
					continue
				}
			}
		}
	}
}

// LogData is the start-log record of the measurement: the set, the reason,
// the width of each glyph, the wait and the time it took (S8-LA-05 to -07
// read it against the GLP §6 columns).
func (r ProbeResult) LogData() map[string]any {
	gs := ProbeGlyphs()
	w := make(map[string]int, len(gs))
	for i, g := range gs {
		if i < len(r.Widths) {
			w[g] = r.Widths[i]
		}
	}
	return map[string]any{
		"glyph_set": r.Set.String(), "reason": r.Reason, "widths": w, "answers": r.Answers,
		"glyphs": len(gs), "wait_ms": r.Wait.Milliseconds(), "elapsed_ms": r.Elapsed.Milliseconds(),
		"input_mode": r.InMode, "forwarded_keys": r.Forwarded, "dropped_replies": r.Dropped,
	}
}

// LogLine is the measurement as one text line.
func (r ProbeResult) LogLine() string {
	return "glyph set = " + r.Set.String() + " (" + r.Reason + ") · wait " + r.Wait.String() + " · took " + r.Elapsed.String()
}

// NoProbe is the glyph record of a run that does not measure (F-LS 9.4, conflict row 7.10;
// S8-LA-07; GL-FX3): the argument-driven mode draws no screen, so it writes no probe glyph and
// no cursor query to any console and has no probe wait: the fallback set, the reason given,
// wait 0, every width unanswered.
func NoProbe(reason string) ProbeResult {
	res := ProbeResult{Set: GlyphFallback, Widths: make([]int, len(ProbeGlyphs())), Reason: reason}
	for i := range res.Widths {
		res.Widths[i] = -1
	}
	return res
}

// Probe measures the glyphs on the process console before the UI starts
// (GL-03). With no console it gives the fallback set at once.
func Probe(wait time.Duration) ProbeResult {
	return runProbe(openProbeConsole(), ProbeGlyphs(), wait, ProbeQuiet, time.Now)
}
