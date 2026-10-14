//go:build windows

package core

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// osIOMeter is the real speed-check reader: MoeLauncher.Sweep (PS:1449-1714) and
// Measure-SsdRandomRead (PS:3979-4038) over CreateFileW with FILE_FLAG_NO_BUFFERING. The read
// buffers are Go memory pinned for the life of the reads (runtime.Pinner) and aligned by hand
// to the sector, as PowerShell aligns its unmanaged buffer.
type osIOMeter struct{}

const (
	sweepReadWaitMs  = 30000 // READ_WAIT_MS
	sweepDrainWaitMs = 60000 // DRAIN_WAIT_MS
)

func openUnbuffered(path string, flags uint32) (windows.Handle, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return windows.InvalidHandle, err
	}
	return windows.CreateFile(p, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil,
		windows.OPEN_EXISTING, flags, 0)
}

// alignedBuf returns n bytes of raw starting at an address that is a multiple of align.
func alignedBuf(raw []byte, n, align int64) []byte {
	addr := uintptr(unsafe.Pointer(&raw[0]))
	off := int64(0)
	if r := int64(addr % uintptr(align)); r != 0 {
		off = align - r
	}
	return raw[off : off+n]
}

func saneSector(v int64) bool { return v >= 512 && v <= 1048576 && v&(v-1) == 0 }

// PhysicalSectorBytes is QueryPhysicalSectorBytes (PS:1515-1536): FILE_STORAGE_INFO's logical
// and physical sizes, sanity-checked; 0 when unknown.
func (osIOMeter) PhysicalSectorBytes(path string) int64 {
	h, err := openUnbuffered(path, 0)
	if err != nil {
		return 0
	}
	defer windows.CloseHandle(h)
	var buf [64]byte
	if windows.GetFileInformationByHandleEx(h, windows.FileStorageInfo, &buf[0], uint32(len(buf))) != nil {
		return 0
	}
	logical := int64(uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2])<<16 | uint32(buf[3])<<24)
	physical := int64(uint32(buf[4]) | uint32(buf[5])<<8 | uint32(buf[6])<<16 | uint32(buf[7])<<24)
	if !saneSector(logical) || !saneSector(physical) || physical < logical {
		return 0
	}
	return physical
}

// RunPoint is MoeLauncher.Sweep.RunPoint (PS:1543-1653): qd overlapped reads kept in flight
// over the offset sequence; the clock starts when the first read of the full queue completes
// (the ramp is not measured) and the window ends at the first completion past windowMs.
func (osIOMeter) RunPoint(path string, qd int, blockBytes int64, offsets []int64, windowMs int, alignBytes int64) sweepRaw {
	r := sweepRaw{QD: qd}
	switch {
	case qd < 1 || qd > 64:
		r.Err = "queue depth out of range"
		return r
	case len(offsets) == 0:
		r.Err = "empty offset sequence"
		return r
	case blockBytes <= 0 || blockBytes > 268435456:
		r.Err = "block size out of range"
		return r
	case windowMs <= 0:
		r.Err = "window duration out of range"
		return r
	}
	h, err := openUnbuffered(path, windows.FILE_FLAG_NO_BUFFERING|windows.FILE_FLAG_OVERLAPPED)
	if err != nil {
		r.Err = "CreateFile(NO_BUFFERING|OVERLAPPED) failed: " + err.Error()
		return r
	}
	align := max(alignBytes, 4096)
	raw := make([]byte, blockBytes*int64(qd)+align)
	ovs := make([]windows.Overlapped, qd)
	events := make([]windows.Handle, qd)
	pending := make([]bool, qd)
	var pin runtime.Pinner
	pin.Pin(&raw[0])
	pin.Pin(&ovs[0])
	buf := alignedBuf(raw, blockBytes*int64(qd), align)
	var bytes, ticks int64
	reads := 0
	drained := true
	defer func() {
		windows.CloseHandle(h)
		for _, e := range events {
			if e != 0 {
				windows.CloseHandle(e)
			}
		}
		if drained {
			pin.Unpin()
		} // else the pinned buffers are deliberately left pinned (a read may still write into them)
	}()
	for i := range events {
		e, err := windows.CreateEvent(nil, 1, 0, nil)
		if err != nil {
			r.Err = "CreateEvent failed: " + err.Error()
			return r
		}
		events[i] = e
	}
	issue := func(slot int, off int64) string {
		if err := windows.ResetEvent(events[slot]); err != nil {
			return "ResetEvent failed: " + err.Error()
		}
		ovs[slot] = windows.Overlapped{Offset: uint32(off & 0xFFFFFFFF), OffsetHigh: uint32(off >> 32), HEvent: events[slot]}
		b := buf[int64(slot)*blockBytes : int64(slot+1)*blockBytes]
		if err := windows.ReadFile(h, b, nil, &ovs[slot]); err != nil && !errors.Is(err, windows.ERROR_IO_PENDING) {
			return "ReadFile failed: " + err.Error()
		}
		pending[slot] = true
		return ""
	}
	waitOne := func() (int, uint32, string) {
		var idx []int
		var hs []windows.Handle
		for i, p := range pending {
			if p {
				idx = append(idx, i)
				hs = append(hs, events[i])
			}
		}
		if len(hs) == 0 {
			return -1, 0, "no outstanding read to wait for"
		}
		w, err := windows.WaitForMultipleObjects(hs, false, sweepReadWaitMs)
		if err != nil {
			return -1, 0, "WaitForMultipleObjects failed: " + err.Error()
		}
		if w == uint32(windows.WAIT_TIMEOUT) {
			return -1, 0, "read did not complete within the wait limit"
		}
		if w >= uint32(len(hs)) {
			return -1, 0, fmt.Sprintf("WaitForMultipleObjects returned %d", w)
		}
		slot := idx[w]
		pending[slot] = false
		var got uint32
		if err := windows.GetOverlappedResult(h, &ovs[slot], &got, false); err != nil {
			return -1, 0, "GetOverlappedResult failed: " + err.Error()
		}
		if got == 0 {
			return -1, 0, "read returned zero bytes"
		}
		return slot, got, ""
	}
	idx := 0
	errs := ""
	for i := 0; i < qd && errs == ""; i++ {
		errs = issue(i, offsets[idx%len(offsets)])
		idx++
	}
	if errs == "" {
		window := time.Duration(windowMs) * time.Millisecond
		var t0, tLast time.Time
		started := false
		for {
			slot, got, e := waitOne()
			if e != "" {
				errs = e
				break
			}
			now := time.Now()
			if !started {
				started, t0, tLast = true, now, now
			} else {
				bytes += int64(got)
				reads++
				tLast = now
				if now.Sub(t0) >= window {
					break
				}
			}
			if e := issue(slot, offsets[idx%len(offsets)]); e != "" {
				errs = e
				break
			}
			idx++
		}
		if started {
			ticks = tLast.Sub(t0).Nanoseconds() / 100
		}
	}
	r.Err = errs
	if drained = drainAll(events, pending); !drained {
		if r.Err == "" {
			r.Err = "outstanding reads did not complete"
		}
	}
	r.Bytes, r.Reads, r.Ticks = bytes, reads, ticks
	r.OK = r.Err == "" && bytes > 0 && ticks > 0
	return r
}

// drainAll is DrainAll (PS:1698-1711): wait for every outstanding read.
func drainAll(events []windows.Handle, pending []bool) bool {
	var hs []windows.Handle
	for i, p := range pending {
		if p && events[i] != 0 {
			hs = append(hs, events[i])
		}
	}
	if len(hs) == 0 {
		return true
	}
	w, err := windows.WaitForMultipleObjects(hs, true, sweepDrainWaitMs)
	if err != nil || w != windows.WAIT_OBJECT_0 {
		return false
	}
	for i := range pending {
		pending[i] = false
	}
	return true
}

// RandomRead is Measure-SsdRandomRead (PS:3979-4038): samples synchronous unbuffered reads of
// one aligned block at random block offsets. The offsets come from a seeded generator with
// PowerShell's seed (20260730); PowerShell's is System.Random, Go's is PCG, so the offsets
// differ (the reading is provisional and decides nothing, LS 12-1).
func (osIOMeter) RandomRead(path string, blockBytes int64, samples int) (probeReading, error) {
	align := int64(probeSectorAlign)
	block := blockBytes / align * align
	if block < align {
		block = align
	}
	fi, err := os.Stat(path)
	if err != nil {
		return probeReading{}, errors.New("probe target missing")
	}
	size := fi.Size()
	if size < block*4 {
		return probeReading{}, errors.New("probe target smaller than 4 blocks")
	}
	h, err := openUnbuffered(path, windows.FILE_FLAG_NO_BUFFERING|windows.FILE_FLAG_RANDOM_ACCESS)
	if err != nil {
		return probeReading{}, fmt.Errorf("CreateFile(NO_BUFFERING) failed: %v", err)
	}
	defer windows.CloseHandle(h)
	raw := make([]byte, block+align)
	buf := alignedBuf(raw, block, align)
	maxBlocks := size/block - 1
	if maxBlocks < 1 {
		return probeReading{}, errors.New("probe target has no full block")
	}
	rng := rand.New(rand.NewPCG(20260730, 0))
	var total int64
	t0 := time.Now()
	for i := 0; i < samples; i++ {
		off := rng.Int64N(min(maxBlocks, math.MaxInt32)) * block
		ov := windows.Overlapped{Offset: uint32(off & 0xFFFFFFFF), OffsetHigh: uint32(off >> 32)}
		var got uint32
		if err := windows.ReadFile(h, buf, &got, &ov); err != nil {
			return probeReading{}, fmt.Errorf("ReadFile failed: %v", err)
		}
		if got == 0 {
			return probeReading{}, errors.New("ReadFile returned 0 bytes")
		}
		total += int64(got)
	}
	secs := time.Since(t0).Seconds()
	if secs <= 0 {
		return probeReading{}, errors.New("probe elapsed time not measurable")
	}
	m := (float64(total) / 1048576.0) / secs
	return probeReading{MiBps: math.RoundToEven(m*100) / 100, BlockBytes: block, Samples: samples, Bytes: total}, nil
}

// WriteScratch writes the probe's scratch file in 1 MiB blocks of seeded random bytes and
// flushes it (Get-ProbeTarget PS:4069-4078).
func (osIOMeter) WriteScratch(path string, size int64) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	blk := make([]byte, 1048576)
	rng := rand.New(rand.NewPCG(20260730, 0))
	for i := range blk {
		blk[i] = byte(rng.Uint32())
	}
	for written := int64(0); written < size; written += int64(len(blk)) {
		if _, err := f.Write(blk); err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
