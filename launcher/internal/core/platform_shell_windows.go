//go:build windows

package core

import (
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Review items 4/14 (the GPU Process Memory counter), 15 (the clipboard) and 24 (Explorer).

var (
	moduser32            = windows.NewLazySystemDLL("user32.dll")
	procOpenClipboard    = moduser32.NewProc("OpenClipboard")
	procCloseClipboard   = moduser32.NewProc("CloseClipboard")
	procEmptyClipboard   = moduser32.NewProc("EmptyClipboard")
	procSetClipboardData = moduser32.NewProc("SetClipboardData")
	procGlobalAlloc      = modkernel32.NewProc("GlobalAlloc")
	procGlobalLock       = modkernel32.NewProc("GlobalLock")
	procGlobalUnlock     = modkernel32.NewProc("GlobalUnlock")
	procGlobalFree       = modkernel32.NewProc("GlobalFree")
	procRtlMoveMemory    = modkernel32.NewProc("RtlMoveMemory")

	modpdh                          = windows.NewLazySystemDLL("pdh.dll")
	procPdhOpenQuery                = modpdh.NewProc("PdhOpenQueryW")
	procPdhAddEnglishCounter        = modpdh.NewProc("PdhAddEnglishCounterW")
	procPdhCollectQueryData         = modpdh.NewProc("PdhCollectQueryData")
	procPdhGetFormattedCounterArray = modpdh.NewProc("PdhGetFormattedCounterArrayW")
	procPdhCloseQuery               = modpdh.NewProc("PdhCloseQuery")
)

const (
	cfUnicodeText = 13
	gmemMoveable  = 0x0002
	pdhFmtLarge   = 0x00000400
	pdhMoreData   = 0x800007D2
)

// osClipboard puts text on the clipboard as CF_UNICODETEXT.
func osClipboard(text string) error {
	u, err := windows.UTF16FromString(text)
	if err != nil {
		return err
	}
	if r, _, e := procOpenClipboard.Call(0); r == 0 {
		return fmt.Errorf("OpenClipboard failed: %v", e)
	}
	defer procCloseClipboard.Call()
	if r, _, e := procEmptyClipboard.Call(); r == 0 {
		return fmt.Errorf("EmptyClipboard failed: %v", e)
	}
	size := uintptr(len(u) * 2)
	h, _, e := procGlobalAlloc.Call(gmemMoveable, size)
	if h == 0 {
		return fmt.Errorf("GlobalAlloc failed: %v", e)
	}
	p, _, e := procGlobalLock.Call(h)
	if p == 0 {
		procGlobalFree.Call(h)
		return fmt.Errorf("GlobalLock failed: %v", e)
	}
	procRtlMoveMemory.Call(p, uintptr(unsafe.Pointer(&u[0])), size)
	procGlobalUnlock.Call(h)
	if r, _, e := procSetClipboardData.Call(cfUnicodeText, h); r == 0 {
		procGlobalFree.Call(h)
		return fmt.Errorf("SetClipboardData failed: %v", e)
	}
	return nil // the clipboard owns h now
}

// osLaunch starts a program with a literal command line and does not wait for it (an outside
// process: Explorer outlives the launcher). Explorer's exit code means nothing, so only the
// creation can fail.
func osLaunch(exe, cmdLine string) error {
	c := exec.Command(exe)
	c.SysProcAttr = &syscall.SysProcAttr{CmdLine: cmdLine}
	if err := c.Start(); err != nil {
		return err
	}
	go c.Wait()
	return nil
}

// pdhFmtCounterValueItem is PDH_FMT_COUNTERVALUE_ITEM_W on 64-bit Windows: the instance
// name pointer, then PDH_FMT_COUNTERVALUE (CStatus, padding, the 8-byte union).
type pdhFmtCounterValueItem struct {
	Name    *uint16
	CStatus uint32
	_       uint32
	Large   int64
}

// osGPUProcessMemory reads \GPU Process Memory(*)\Dedicated Usage: bytes per instance
// ("pid_<PID>_luid_<..>_phys_<n>").
func osGPUProcessMemory() (map[string]int64, error) {
	var q windows.Handle
	if r, _, _ := procPdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&q))); r != 0 {
		return nil, fmt.Errorf("PdhOpenQuery failed: 0x%08x", r)
	}
	defer procPdhCloseQuery.Call(uintptr(q))
	path, _ := windows.UTF16PtrFromString(`\GPU Process Memory(*)\Dedicated Usage`)
	var c windows.Handle
	if r, _, _ := procPdhAddEnglishCounter.Call(uintptr(q), uintptr(unsafe.Pointer(path)), 0, uintptr(unsafe.Pointer(&c))); r != 0 {
		return nil, fmt.Errorf("PdhAddEnglishCounter failed: 0x%08x", r)
	}
	if r, _, _ := procPdhCollectQueryData.Call(uintptr(q)); r != 0 {
		return nil, fmt.Errorf("PdhCollectQueryData failed: 0x%08x", r)
	}
	var size, count uint32
	r, _, _ := procPdhGetFormattedCounterArray.Call(uintptr(c), pdhFmtLarge, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), 0)
	if r != pdhMoreData {
		if r == 0 {
			return map[string]int64{}, nil
		}
		return nil, fmt.Errorf("PdhGetFormattedCounterArray failed: 0x%08x", r)
	}
	// The items, then the instance names they point to, in one buffer of size bytes; a slice
	// of items keeps it aligned.
	itemSize := uint32(unsafe.Sizeof(pdhFmtCounterValueItem{}))
	buf := make([]pdhFmtCounterValueItem, size/itemSize+2)
	r, _, _ = procPdhGetFormattedCounterArray.Call(uintptr(c), pdhFmtLarge, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&buf[0])))
	if r != 0 {
		return nil, fmt.Errorf("PdhGetFormattedCounterArray failed: 0x%08x", r)
	}
	out := make(map[string]int64, count)
	for _, it := range buf[:min(int(count), len(buf))] {
		if it.Name == nil || it.CStatus != 0 && it.CStatus != 1 { // PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA
			continue
		}
		name := windows.UTF16PtrToString(it.Name)
		out[strings.ToLower(name)] += it.Large
	}
	runtime.KeepAlive(buf)
	if len(out) == 0 && count > 0 {
		return nil, errors.New("GPU Process Memory: no valid instance")
	}
	return out, nil
}
