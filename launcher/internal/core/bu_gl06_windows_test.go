// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"math"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

// piRaw is PERFORMANCE_INFORMATION read by the test on its own (psapi.dll, not the reader's
// kernel32 entry).
type piRaw struct {
	Cb                                     uint32
	CommitTotal, CommitLimit, CommitPeak   uintptr
	PhysicalTotal, PhysicalAvailable       uintptr
	SystemCache, KernelTotal, KernelPaged  uintptr
	KernelNonpaged, PageSize               uintptr
	HandleCount, ProcessCount, ThreadCount uint32
}

var procPsapiGetPerformanceInfo = windows.NewLazySystemDLL("psapi.dll").NewProc("GetPerformanceInfo")

func rawPerf(t *testing.T) (limit, avail, page float64) {
	t.Helper()
	var pi piRaw
	pi.Cb = uint32(unsafe.Sizeof(pi))
	if r, _, err := procPsapiGetPerformanceInfo.Call(uintptr(unsafe.Pointer(&pi)), uintptr(pi.Cb)); r == 0 {
		t.Fatalf("GetPerformanceInfo: %v", err)
	}
	pg := float64(pi.PageSize)
	return float64(pi.CommitLimit) * pg / bytesPerMiB, float64(pi.CommitLimit-pi.CommitTotal) * pg / bytesPerMiB, pg / bytesPerMiB
}

// TestS8BU01 (S8-BU-01; GL-06), on the host: the launcher's commit reader and
// GetPerformanceInfo, read at the same moment, give the same commit limit and available commit,
// equal within the change between two consecutive GetPerformanceInfo calls (one page for the
// MiB conversion). The commit moves with every process on the host, so a pass on one of up to
// 20 bracketed readings is the row's «same moment».
func TestS8BU01(t *testing.T) {
	l := &Launcher{os: defaultHooks()}
	for try := 1; try <= 20; try++ {
		l1, a1, page := rawPerf(t)
		c, err := l.CommitNow()
		if err != nil {
			t.Fatal(err)
		}
		l2, a2, _ := rawPerf(t)
		okL := math.Abs(c.LimitMiB-l1) <= math.Abs(l2-l1)+page
		okA := math.Abs(c.RoomMiB()-a1) <= math.Abs(a2-a1)+page
		if okL && okA {
			t.Logf("try %d: limit %.3f MiB (GetPerformanceInfo %.3f, %.3f) · available %.3f MiB (%.3f, %.3f)", try, c.LimitMiB, l1, l2, c.RoomMiB(), a1, a2)
			return
		}
		t.Logf("try %d: limit %.3f vs %.3f..%.3f · available %.3f vs %.3f..%.3f", try, c.LimitMiB, l1, l2, c.RoomMiB(), a1, a2)
	}
	t.Fatal("the reader and GetPerformanceInfo disagree beyond the change between two calls")
}
