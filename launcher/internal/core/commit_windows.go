// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"fmt"
	"unsafe"
)

var procK32GetPerformanceInfo = modkernel32.NewProc("K32GetPerformanceInfo")

// performanceInformation is PERFORMANCE_INFORMATION (psapi.h) on 64-bit Windows.
type performanceInformation struct {
	Cb                                     uint32
	CommitTotal, CommitLimit, CommitPeak   uintptr
	PhysicalTotal, PhysicalAvailable       uintptr
	SystemCache, KernelTotal, KernelPaged  uintptr
	KernelNonpaged, PageSize               uintptr
	HandleCount, ProcessCount, ThreadCount uint32
}

// osPerformanceInfo reads the commit limit and total (pages × page size) by GetPerformanceInfo.
func osPerformanceInfo() (CommitReading, error) {
	var pi performanceInformation
	pi.Cb = uint32(unsafe.Sizeof(pi))
	if r, _, err := procK32GetPerformanceInfo.Call(uintptr(unsafe.Pointer(&pi)), uintptr(pi.Cb)); r == 0 {
		return CommitReading{}, fmt.Errorf("GetPerformanceInfo failed: %v", err)
	}
	page := float64(pi.PageSize)
	return CommitReading{LimitMiB: float64(pi.CommitLimit) * page / bytesPerMiB, TotalMiB: float64(pi.CommitTotal) * page / bytesPerMiB}, nil
}
