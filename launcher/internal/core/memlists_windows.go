// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

var procPdhGetFormattedCounterValue = modpdh.NewProc("PdhGetFormattedCounterValue")

// pdhFmtCounterValue is PDH_FMT_COUNTERVALUE: CStatus, padding, the 8-byte union.
type pdhFmtCounterValue struct {
	CStatus uint32
	_       uint32
	Large   int64
}

// osMemoryLists reads \Memory\Modified Page List Bytes and \Memory\Free & Zero Page List Bytes
// in one PDH query (LFS:302).
func osMemoryLists() (MemoryLists, error) {
	var q windows.Handle
	if r, _, _ := procPdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&q))); r != 0 {
		return MemoryLists{}, fmt.Errorf("PdhOpenQuery failed: 0x%08x", r)
	}
	defer procPdhCloseQuery.Call(uintptr(q))
	paths := []string{`\Memory\Modified Page List Bytes`, `\Memory\Free & Zero Page List Bytes`}
	cs := make([]windows.Handle, len(paths))
	for i, p := range paths {
		u, _ := windows.UTF16PtrFromString(p)
		if r, _, _ := procPdhAddEnglishCounter.Call(uintptr(q), uintptr(unsafe.Pointer(u)), 0, uintptr(unsafe.Pointer(&cs[i]))); r != 0 {
			return MemoryLists{}, fmt.Errorf("PdhAddEnglishCounter %s failed: 0x%08x", p, r)
		}
	}
	if r, _, _ := procPdhCollectQueryData.Call(uintptr(q)); r != 0 {
		return MemoryLists{}, fmt.Errorf("PdhCollectQueryData failed: 0x%08x", r)
	}
	var vals [2]int64
	for i, c := range cs {
		var v pdhFmtCounterValue
		r, _, _ := procPdhGetFormattedCounterValue.Call(uintptr(c), pdhFmtLarge, 0, uintptr(unsafe.Pointer(&v)))
		if r != 0 || v.CStatus > 1 { // PDH_CSTATUS_VALID_DATA 0, PDH_CSTATUS_NEW_DATA 1
			return MemoryLists{}, fmt.Errorf("PdhGetFormattedCounterValue %s failed: 0x%08x status %d", paths[i], r, v.CStatus)
		}
		vals[i] = v.Large
	}
	return MemoryLists{ModifiedMiB: vals[0] >> 20, FreeZeroMiB: vals[1] >> 20}, nil
}

// osTaskManager opens Task Manager through the shell (it asks for elevation itself where the
// account needs it), with no arguments: no process is selected (U-35 (i)).
func osTaskManager() error {
	verb, _ := windows.UTF16PtrFromString("open")
	file, _ := windows.UTF16PtrFromString("taskmgr.exe")
	return windows.ShellExecute(0, verb, file, nil, nil, windows.SW_SHOWNORMAL)
}
