package core

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
)

// GL-10: the speed check's mock timings. fakeMeter stands for the disk: it records every call in
// order (S8-LA-38's «probe sequence») and answers each window from timing, a table of bytes and
// 100 ns ticks per queue depth and per round of that depth (nil = every window fails).
type meterCall struct {
	Kind string // scratch, probe, point
	QD   int
	Path string
}

type fakeMeter struct {
	mu       sync.Mutex
	calls    []meterCall
	timing   func(qd, round int) (bytes, ticks int64)
	phys     int64
	probeErr error
}

func (m *fakeMeter) record(c meterCall) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = append(m.calls, c)
}

// sequence is the call log as one line: «scratch probe 1 2 4 8 8 4 2 1».
func (m *fakeMeter) sequence() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var parts []string
	for _, c := range m.calls {
		if c.Kind == "point" {
			parts = append(parts, fmt.Sprint(c.QD))
		} else {
			parts = append(parts, c.Kind)
		}
	}
	return strings.Join(parts, " ")
}

func (m *fakeMeter) PhysicalSectorBytes(string) int64 { return m.phys }

func (m *fakeMeter) RunPoint(path string, qd int, block int64, offs []int64, win int, align int64) sweepRaw {
	m.mu.Lock()
	round := 0
	for _, c := range m.calls {
		if c.Kind == "point" && c.QD == qd {
			round++
		}
	}
	m.calls = append(m.calls, meterCall{"point", qd, path})
	timing := m.timing
	m.mu.Unlock()
	if timing == nil {
		return sweepRaw{QD: qd, Err: "fixture: no sweep timing"}
	}
	b, t := timing(qd, round)
	if b <= 0 || t <= 0 {
		return sweepRaw{QD: qd, Bytes: b, Ticks: t, Err: "fixture: failed window"}
	}
	return sweepRaw{QD: qd, OK: true, Bytes: b, Ticks: t, Reads: int(b / block)}
}

func (m *fakeMeter) RandomRead(path string, block int64, samples int) (probeReading, error) {
	m.record(meterCall{Kind: "probe", Path: path})
	if m.probeErr != nil {
		return probeReading{}, m.probeErr
	}
	b := max(block/probeSectorAlign*probeSectorAlign, probeSectorAlign)
	return probeReading{MiBps: 2048.5, BlockBytes: b, Samples: samples, Bytes: b * int64(samples)}, nil
}

func (m *fakeMeter) WriteScratch(path string, size int64) error {
	m.record(meterCall{Kind: "scratch", Path: path})
	if size <= 0 {
		return errors.New("fixture: no size")
	}
	return os.WriteFile(path, []byte("scratch"), 0o644)
}
