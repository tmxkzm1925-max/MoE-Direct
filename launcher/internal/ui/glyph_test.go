// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

// laConsole is a scripted console for the glyph measurement (S8-LA-04): a fake
// clock, the console modes, and records that arrive at given times after the
// measurement starts. Read moves the clock to the next arrival or to the
// deadline, so every elapsed time is exact.
type laConsole struct {
	t0, clock time.Time
	in0, out0 uint32
	in, out   uint32
	inSet     []uint32 // every input mode set, in order
	failOut   bool     // SetOut with the VT bits fails
	steps     []laStep
	wrote     strings.Builder
	forwarded []ConRecord
	reads     int
}

type laStep struct {
	at   time.Duration
	recs []ConRecord
}

func newLA(steps ...laStep) *laConsole {
	t0 := time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC)
	return &laConsole{t0: t0, clock: t0, in0: 0x01F7, out0: 0x0003, in: 0x01F7, out: 0x0003, steps: steps}
}

func (c *laConsole) now() time.Time { return c.clock }

func (c *laConsole) Modes() (uint32, uint32, error) { return c.in, c.out, nil }

func (c *laConsole) SetIn(m uint32) error { c.in = m; c.inSet = append(c.inSet, m); return nil }

func (c *laConsole) SetOut(m uint32) error {
	if c.failOut && m&modeVTProcessing != 0 && c.out0&modeVTProcessing == 0 {
		return errors.New("the parameter is incorrect")
	}
	c.out = m
	return nil
}

func (c *laConsole) Write(p []byte) (int, error) { c.wrote.Write(p); return len(p), nil }

func (c *laConsole) Read(deadline time.Time) ([]ConRecord, error) {
	c.reads++
	if len(c.steps) > 0 {
		at := c.t0.Add(c.steps[0].at)
		if !at.After(deadline) {
			if at.After(c.clock) {
				c.clock = at
			}
			r := c.steps[0].recs
			c.steps = c.steps[1:]
			return r, nil
		}
	}
	if deadline.After(c.clock) {
		c.clock = deadline
	}
	return nil, nil
}

func (c *laConsole) Forward(recs []ConRecord) error {
	if c.in != c.in0 || c.out != c.out0 {
		return errors.New("forward before the modes were restored")
	}
	c.forwarded = append(c.forwarded, recs...)
	return nil
}

// reply is the console's cursor reply as key records with virtual key 0 (G009 A1).
func reply(row, col int) []ConRecord {
	var out []ConRecord
	for _, ch := range fmt.Sprintf("\x1b[%d;%dR", row, col) {
		out = append(out, ConRecord{Key: true, Down: true, VK: 0, Char: ch})
	}
	return out
}

// replies: one reply per width, in order.
func replies(widths []int) []ConRecord {
	var out []ConRecord
	for _, w := range widths {
		out = append(out, reply(1, w+1)...)
	}
	return out
}

// userKey is a letter the user typed: a press and a release with its virtual key.
func userKey(ch rune) []ConRecord {
	vk := uint16(ch)
	if ch >= 'a' && ch <= 'z' {
		vk = uint16(ch - 'a' + 'A')
	}
	return []ConRecord{{Key: true, Down: true, VK: vk, Char: ch}, {Key: true, Down: false, VK: vk, Char: ch}}
}

func cat(rs ...[]ConRecord) []ConRecord {
	var out []ConRecord
	for _, r := range rs {
		out = append(out, r...)
	}
	return out
}

func letters(recs []ConRecord) string {
	var b strings.Builder
	for _, r := range recs {
		if r.Down {
			b.WriteRune(r.Char)
		}
	}
	return b.String()
}

// TestS8LA04 (plan S8-LA-04; GLP:211-229 cases 1-10; GLP:124-155 steps 1-7): the
// glyph self-measurement on a scripted console with the configured wait ProbeWait.
func TestS8LA04(t *testing.T) {
	gs := ProbeGlyphs()
	n := len(gs)
	ones := make([]int, n)
	for i := range ones {
		ones[i] = 1
	}
	wait := ProbeWait
	// tolerance of the no-reply case: the drain's quiet period (GLP step 6)
	tol := ProbeQuiet
	run := func(c *laConsole) ProbeResult { return runProbe(c, gs, wait, ProbeQuiet, c.now) }
	restored := func(t *testing.T, c *laConsole) {
		t.Helper()
		if c.in != c.in0 || c.out != c.out0 {
			t.Fatalf("modes after the measurement: in %#x out %#x, want %#x %#x", c.in, c.out, c.in0, c.out0)
		}
	}
	var scenarios []func() *laConsole // cases 1-7, for case08

	// case01: all-narrow replies give the primary set; no key is forwarded
	s1 := func() *laConsole { return newLA(laStep{time.Millisecond, replies(ones)}) }
	scenarios = append(scenarios, s1)
	t.Run("case01", func(t *testing.T) {
		c := s1()
		r := run(c)
		if r.Set != GlyphDesign || r.Set.String() != "primary" || r.Answers != n || len(c.forwarded) != 0 {
			t.Fatalf("all narrow: set %v answers %d forwarded %d (%s)", r.Set, r.Answers, len(c.forwarded), r.Reason)
		}
		if !strings.Contains(c.wrote.String(), "\r\x1b[2K·\x1b[6n") || !strings.HasSuffix(c.wrote.String(), "\r\x1b[2K") {
			t.Fatalf("probe writes %q", c.wrote.String())
		}
		if r.Elapsed != time.Millisecond {
			t.Fatalf("elapsed %v, want the reply's arrival (1ms): no wait after the last reply", r.Elapsed)
		}
		t.Logf("case01: set %s, %d of %d answers, elapsed %v", r.Set, r.Answers, n, r.Elapsed)
	})

	// case02: one wide glyph gives the fallback set
	wide := append([]int{}, ones...)
	wide[4] = 2 // ●
	s2 := func() *laConsole { return newLA(laStep{time.Millisecond, replies(wide)}) }
	scenarios = append(scenarios, s2)
	t.Run("case02", func(t *testing.T) {
		r := run(s2())
		if r.Set != GlyphFallback || r.Set.String() != "fallback" || r.Widths[4] != 2 || !strings.Contains(r.Reason, "● took 2 cells") {
			t.Fatalf("one wide: %+v", r)
		}
		t.Logf("case02: set %s (%s)", r.Set, r.Reason)
	})

	// case03: no reply gives the fallback set within the configured wait plus tolerance
	s3 := func() *laConsole { return newLA() }
	scenarios = append(scenarios, s3)
	t.Run("case03", func(t *testing.T) {
		r := run(s3())
		if r.Set != GlyphFallback || r.Answers != 0 || r.Elapsed > wait+tol || r.Elapsed < wait {
			t.Fatalf("no reply: set %v answers %d elapsed %v (wait %v, tolerance %v)", r.Set, r.Answers, r.Elapsed, wait, tol)
		}
		t.Logf("case03: set %s, elapsed %v <= wait %v + tolerance %v", r.Set, r.Elapsed, wait, tol)
	})

	// case04: a late reply is drained; only the user's key is forwarded, once
	s4 := func() *laConsole {
		return newLA(laStep{wait + 10*time.Millisecond, replies(ones)}, laStep{wait + 20*time.Millisecond, userKey('x')})
	}
	scenarios = append(scenarios, s4)
	t.Run("case04", func(t *testing.T) {
		c := s4()
		r := run(c)
		if r.Set != GlyphFallback || r.Answers != 0 {
			t.Fatalf("late reply: set %v answers %d", r.Set, r.Answers)
		}
		if !reflect.DeepEqual(c.forwarded, userKey('x')) {
			t.Fatalf("late reply: forwarded %+v, want the key x once", c.forwarded)
		}
		if r.Dropped != len(replies(ones)) || len(c.steps) != 0 {
			t.Fatalf("late reply: dropped %d of %d, steps left %d", r.Dropped, len(replies(ones)), len(c.steps))
		}
		t.Logf("case04: set %s, %d late reply records drained, forwarded %q once", r.Set, r.Dropped, letters(c.forwarded))
	})

	// case05: a letter that precedes the reply is forwarded once, in order
	s5 := func() *laConsole {
		half := len(replies(ones[:n/2]))
		all := replies(ones)
		return newLA(laStep{time.Millisecond / 2, userKey('q')},
			laStep{time.Millisecond, cat(all[:half], userKey('r'), all[half:])})
	}
	scenarios = append(scenarios, s5)
	t.Run("case05", func(t *testing.T) {
		c := s5()
		r := run(c)
		if r.Set != GlyphDesign || r.Answers != n {
			t.Fatalf("key before reply: set %v answers %d (%s)", r.Set, r.Answers, r.Reason)
		}
		if !reflect.DeepEqual(c.forwarded, cat(userKey('q'), userKey('r'))) {
			t.Fatalf("key before reply: forwarded %+v, want q then r, once each", c.forwarded)
		}
		t.Logf("case05: set %s, forwarded %q in order", r.Set, letters(c.forwarded))
	})

	// case06: a split reply reads as one
	s6 := func() *laConsole {
		all := replies(ones)
		return newLA(laStep{time.Millisecond, all[:3]}, laStep{2 * time.Millisecond, all[3:]}) // ESC [ 1 / ; 2 R ...
	}
	scenarios = append(scenarios, s6)
	t.Run("case06", func(t *testing.T) {
		c := s6()
		r := run(c)
		if r.Set != GlyphDesign || r.Answers != n || c.reads != 2 {
			t.Fatalf("split reply: set %v answers %d reads %d", r.Set, r.Answers, c.reads)
		}
		t.Logf("case06: set %s from %d reads", r.Set, c.reads)
	})

	// case07: a failed VT-mode set gives the fallback without waiting
	s7 := func() *laConsole { c := newLA(laStep{time.Millisecond, replies(ones)}); c.failOut = true; return c }
	scenarios = append(scenarios, s7)
	t.Run("case07", func(t *testing.T) {
		c := s7()
		r := run(c)
		if r.Set != GlyphFallback || r.Elapsed != 0 || c.reads != 0 || c.wrote.Len() != 0 || !strings.Contains(r.Reason, "no VT output") {
			t.Fatalf("VT set fails: set %v elapsed %v reads %d wrote %q (%s)", r.Set, r.Elapsed, c.reads, c.wrote.String(), r.Reason)
		}
		restored(t, c)
		t.Logf("case07: set %s at once (%s), elapsed %v", r.Set, r.Reason, r.Elapsed)
	})

	// case08: the modes after every case equal the originals
	t.Run("case08", func(t *testing.T) {
		for i, s := range scenarios {
			c := s()
			run(c)
			if c.in != c.in0 || c.out != c.out0 {
				t.Fatalf("case%02d: modes after: in %#x out %#x, want %#x %#x", i+1, c.in, c.out, c.in0, c.out0)
			}
		}
		t.Logf("case08: modes restored after cases 01-07 (in %#x, out %#x)", uint32(0x01F7), uint32(0x0003))
	})

	// case09: the input mode during measurement includes 0x0200 (VT input)
	t.Run("case09", func(t *testing.T) {
		c := s1()
		r := run(c)
		if len(c.inSet) == 0 || c.inSet[0]&0x0200 == 0 || c.inSet[0] != 0x0288 || r.InMode != 0x0288 {
			t.Fatalf("input modes set %#x, result %#x", c.inSet, r.InMode)
		}
		t.Logf("case09: input mode during measurement %#x (includes 0x0200), then %#x", c.inSet[0], c.inSet[len(c.inSet)-1])
	})

	// case10: reply records (VK 0) are told from key records: typed digits and
	// brackets inside a reply do not change it, and no reply record is forwarded
	t.Run("case10", func(t *testing.T) {
		all := replies(ones)
		typed := cat(userKey('1'), userKey('['), userKey(';'))
		c := newLA(laStep{time.Millisecond, cat(all[:2], typed, all[2:])})
		r := run(c)
		if r.Set != GlyphDesign || r.Answers != n {
			t.Fatalf("mixed records: set %v answers %d (%s)", r.Set, r.Answers, r.Reason)
		}
		if !reflect.DeepEqual(c.forwarded, typed) {
			t.Fatalf("mixed records: forwarded %+v, want the typed keys only", c.forwarded)
		}
		for _, f := range c.forwarded {
			if f.VK == 0 {
				t.Fatalf("a reply record was forwarded: %+v", f)
			}
		}
		t.Logf("case10: set %s; forwarded %q (VK != 0 only)", r.Set, letters(c.forwarded))
	})
}

// TestGlyphProbeLog: the start-log record and line of the measurement (GL-03).
func TestGlyphProbeLog(t *testing.T) {
	c := newLA()
	r := runProbe(c, ProbeGlyphs(), ProbeWait, ProbeQuiet, c.now)
	d := r.LogData()
	if d["glyph_set"] != "fallback" || d["wait_ms"] != int64(400) || len(d["widths"].(map[string]int)) != len(ProbeGlyphs()) {
		t.Fatalf("log data %v", d)
	}
	if l := r.LogLine(); !strings.HasPrefix(l, "glyph set = fallback (no answer within the wait (0 of ") {
		t.Fatalf("log line %q", l)
	}
	nc := runProbe(noConsole{}, ProbeGlyphs(), ProbeWait, ProbeQuiet, time.Now)
	if nc.Set != GlyphFallback || !strings.HasPrefix(nc.Reason, "no console: ") || nc.Elapsed > 50*time.Millisecond {
		t.Fatalf("no console: %+v", nc)
	}
}

type noConsole struct{}

func (noConsole) Modes() (uint32, uint32, error)      { return 0, 0, errors.New("the handle is invalid") }
func (noConsole) SetIn(uint32) error                  { return errors.New("unexpected") }
func (noConsole) SetOut(uint32) error                 { return errors.New("unexpected") }
func (noConsole) Write(p []byte) (int, error)         { return 0, errors.New("unexpected") }
func (noConsole) Read(time.Time) ([]ConRecord, error) { return nil, errors.New("unexpected") }
func (noConsole) Forward([]ConRecord) error           { return errors.New("unexpected") }
