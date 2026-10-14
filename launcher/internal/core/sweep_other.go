//go:build !windows

package core

import "errors"

// osIOMeter has no unbuffered overlapped reader off Windows: the probe and the sweep fail and
// the start degrades to QD 1 (RS 5), as a failed sweep does on Windows.
type osIOMeter struct{}

var errNoMeter = errors.New("the speed check reads are Windows only")

func (osIOMeter) PhysicalSectorBytes(string) int64 { return 0 }

func (osIOMeter) RunPoint(_ string, qd int, _ int64, _ []int64, _ int, _ int64) sweepRaw {
	return sweepRaw{QD: qd, Err: errNoMeter.Error()}
}

func (osIOMeter) RandomRead(string, int64, int) (probeReading, error) {
	return probeReading{}, errNoMeter
}

func (osIOMeter) WriteScratch(string, int64) error { return errNoMeter }
