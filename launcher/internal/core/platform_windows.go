//go:build windows

package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// The Win32 calls of the slice. Each one names the PowerShell function whose
// [MoeLauncher.Native] call it repeats.

var (
	modkernel32  = windows.NewLazySystemDLL("kernel32.dll")
	modiphlpapi  = windows.NewLazySystemDLL("iphlpapi.dll")
	procGMSEx    = modkernel32.NewProc("GlobalMemoryStatusEx")
	procInstMem  = modkernel32.NewProc("GetPhysicallyInstalledSystemMemory")
	procProcMem  = modkernel32.NewProc("K32GetProcessMemoryInfo")
	procTcpTable = modiphlpapi.NewProc("GetExtendedTcpTable")
)

const (
	stillActive            = 259    // STILL_ACTIVE
	ctrlBreakEvent         = 1      // CTRL_BREAK_EVENT (PS CTRL_BREAK_EVENT)
	createNewProcessGroup  = 0x200  // CREATE_NEW_PROCESS_GROUP
	jobLimitKillOnJobClose = 0x2000 // JOB_LIMIT_KILL_ON_JOB_CLOSE
	tcpTableOwnerPidListen = 3      // TCP_TABLE_OWNER_PID_LISTENER
	errInsufficientBuffer  = 122
	fileReadAttributes     = 0x80
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

// roundMiB is PowerShell's [long]($bytes / 1MB): a double division cast with
// round-half-to-even, not a floor (Get-MemStatus PS:3517-3523).
func roundMiB(b uint64) int64 { return int64(math.RoundToEven(float64(b) / 1048576.0)) }

// osMemStatus is Get-MemStatus (PS:3517-3524), plus the commit pair for the screen.
func osMemStatus() (MemStatus, error) {
	var m memoryStatusEx
	m.Length = uint32(unsafe.Sizeof(m))
	r, _, e := procGMSEx.Call(uintptr(unsafe.Pointer(&m)))
	if r == 0 {
		return MemStatus{}, fmt.Errorf("GlobalMemoryStatusEx failed: %v", e)
	}
	return MemStatus{
		TotalPhysMiB: roundMiB(m.TotalPhys), AvailPhysMiB: roundMiB(m.AvailPhys),
		CommitLimitMiB: roundMiB(m.TotalPageFile), CommitAvailMiB: roundMiB(m.AvailPageFile),
	}, nil
}

// osInstalledMiB is Get-InstalledMemoryMib (PS:3664-3682): floor(KB / 1024).
func osInstalledMiB() (int64, error) {
	var kb uint64
	r, _, e := procInstMem.Call(uintptr(unsafe.Pointer(&kb)))
	if r == 0 {
		return 0, fmt.Errorf("GetPhysicallyInstalledSystemMemory failed (%v)", e)
	}
	if kb == 0 {
		return 0, errors.New("GetPhysicallyInstalledSystemMemory reported 0 KB")
	}
	return int64(kb / 1024), nil
}

// osFileIdentity is Get-FileIdentity (PS:2135-2167) with QueryFileId128 (PS:1342-1361):
// "<volume x16>:<16 id bytes hex>", the size and the last-write FILETIME as int64.
func osFileIdentity(path string) (FileIdentity, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return FileIdentity{}, err
	}
	h, err := windows.CreateFile(p, fileReadAttributes,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, 0, 0)
	if err != nil {
		return FileIdentity{}, fmt.Errorf("open for identity failed (%v)", err)
	}
	defer windows.CloseHandle(h)
	var info0 struct { // FILE_ID_INFO; a typed struct keeps the 8-byte alignment the call requires
		VolumeSerialNumber uint64
		FileID             [16]byte
	}
	if err := windows.GetFileInformationByHandleEx(h, windows.FileIdInfo, (*byte)(unsafe.Pointer(&info0)), uint32(unsafe.Sizeof(info0))); err != nil {
		return FileIdentity{}, fmt.Errorf("FILE_ID_INFO unavailable (%v); this filesystem supplies no 128 bit file id", err)
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%016x:", info0.VolumeSerialNumber))
	any := false
	for i := 0; i < 16; i++ {
		b := info0.FileID[i]
		if b != 0 {
			any = true
		}
		sb.WriteString(fmt.Sprintf("%02x", b))
	}
	if !any {
		return FileIdentity{}, errors.New("FILE_ID_INFO unavailable; this filesystem supplies no 128 bit file id")
	}
	var info windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(h, &info); err != nil {
		return FileIdentity{}, fmt.Errorf("GetFileInformationByHandle failed (%v)", err)
	}
	size := int64(info.FileSizeHigh)<<32 | int64(info.FileSizeLow)
	mtime := int64(info.LastWriteTime.HighDateTime)<<32 | int64(info.LastWriteTime.LowDateTime)
	return FileIdentity{ID: sb.String(), Size: size, MTime: mtime}, nil
}

// osVolumeGUID is Get-VolumeIdentity (PS:4095-4110), lower case.
func osVolumeGUID(path string) (string, error) {
	full, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	p, err := windows.UTF16PtrFromString(full)
	if err != nil {
		return "", err
	}
	mount := make([]uint16, 260)
	if err := windows.GetVolumePathName(p, &mount[0], 260); err != nil {
		return "", fmt.Errorf("GetVolumePathNameW failed (%v)", err)
	}
	guid := make([]uint16, 260)
	if err := windows.GetVolumeNameForVolumeMountPoint(&mount[0], &guid[0], 260); err != nil {
		return "", fmt.Errorf("GetVolumeNameForVolumeMountPointW failed (%v)", err)
	}
	return strings.ToLower(windows.UTF16ToString(guid)), nil
}

// osPortOwners is Get-PortOwnerPids (PS:12067-12075): the owning PIDs of every TCP
// listener (IPv4 and IPv6) on the port; an empty slice when there is none, an error when
// the table cannot be read (PowerShell returns $null then: fail-close).
func osPortOwners(port int) ([]int, error) {
	seen := map[int]bool{}
	for _, af := range []uint32{windows.AF_INET, windows.AF_INET6} {
		var size uint32
		var buf []byte
		for i := 0; i < 8; i++ {
			var ptr uintptr
			if len(buf) > 0 {
				ptr = uintptr(unsafe.Pointer(&buf[0]))
			}
			r, _, _ := procTcpTable.Call(ptr, uintptr(unsafe.Pointer(&size)), 0, uintptr(af), tcpTableOwnerPidListen, 0)
			if r == 0 {
				break
			}
			if r != errInsufficientBuffer {
				return nil, fmt.Errorf("GetExtendedTcpTable failed (%d)", r)
			}
			buf = make([]byte, size+1024)
			size = uint32(len(buf))
		}
		if len(buf) < 4 {
			continue
		}
		n := binary.LittleEndian.Uint32(buf[0:4])
		rowSize, portOff, pidOff, base := 24, 8, 20, 4
		if af == windows.AF_INET6 {
			rowSize, portOff, pidOff = 56, 20, 52
		}
		for i := 0; i < int(n); i++ {
			off := base + i*rowSize
			if off+rowSize > len(buf) {
				break
			}
			raw := binary.LittleEndian.Uint32(buf[off+portOff : off+portOff+4])
			p := int(uint16(raw>>8&0xff) | uint16(raw&0xff)<<8)
			if p != port {
				continue
			}
			seen[int(binary.LittleEndian.Uint32(buf[off+pidOff:off+pidOff+4]))] = true
		}
	}
	out := make([]int, 0, len(seen))
	for pid := range seen {
		out = append(out, pid)
	}
	sort.Ints(out)
	return out, nil
}

// osPortListeners is the address part of Get-NetTCPConnection -State Listen -LocalPort (smoke
// item 6, Test-LoopbackOnlyBinding PS:12092-12106; GL-11): the local address of every TCP
// listener (IPv4 and IPv6) on the port, from the same listener table as osPortOwners.
func osPortListeners(port int) ([]string, error) {
	var out []string
	for _, af := range []uint32{windows.AF_INET, windows.AF_INET6} {
		var size uint32
		var buf []byte
		for i := 0; i < 8; i++ {
			var ptr uintptr
			if len(buf) > 0 {
				ptr = uintptr(unsafe.Pointer(&buf[0]))
			}
			r, _, _ := procTcpTable.Call(ptr, uintptr(unsafe.Pointer(&size)), 0, uintptr(af), tcpTableOwnerPidListen, 0)
			if r == 0 {
				break
			}
			if r != errInsufficientBuffer {
				return nil, fmt.Errorf("GetExtendedTcpTable failed (%d)", r)
			}
			buf = make([]byte, size+1024)
			size = uint32(len(buf))
		}
		if len(buf) < 4 {
			continue
		}
		n := binary.LittleEndian.Uint32(buf[0:4])
		// MIB_TCPROW_OWNER_PID: state, local address at 4, local port at 8;
		// MIB_TCP6ROW_OWNER_PID: local address (16 bytes) at 0, local port at 20
		rowSize, portOff, addrOff, addrLen, base := 24, 8, 4, 4, 4
		if af == windows.AF_INET6 {
			rowSize, portOff, addrOff, addrLen = 56, 20, 0, 16
		}
		for i := 0; i < int(n); i++ {
			off := base + i*rowSize
			if off+rowSize > len(buf) {
				break
			}
			raw := binary.LittleEndian.Uint32(buf[off+portOff : off+portOff+4])
			if int(uint16(raw>>8&0xff)|uint16(raw&0xff)<<8) != port {
				continue
			}
			ip := net.IP(append([]byte(nil), buf[off+addrOff:off+addrOff+addrLen]...))
			out = append(out, ip.String())
		}
	}
	return out, nil
}

type processMemoryCountersEx struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
	PrivateUsage               uintptr
}

func processMemory(h windows.Handle) (ProcessMemory, error) {
	var c processMemoryCountersEx
	c.CB = uint32(unsafe.Sizeof(c))
	r, _, e := procProcMem.Call(uintptr(h), uintptr(unsafe.Pointer(&c)), uintptr(c.CB))
	if r == 0 {
		return ProcessMemory{}, fmt.Errorf("K32GetProcessMemoryInfo failed: %v", e)
	}
	return ProcessMemory{WorkingSetBytes: int64(c.WorkingSetSize), PrivateBytes: int64(c.PrivateUsage),
		PeakWorkingSetBytes: int64(c.PeakWorkingSetSize)}, nil
}

// osConsoleAvailable is Test-ConsoleAvailable (PS:1872-1875).
func osConsoleAvailable() bool {
	h, err := windows.GetStdHandle(windows.STD_INPUT_HANDLE)
	return err == nil && h != 0
}

// --- locks (Acquire-NamedMutex PS:4970-4991, Acquire-FileLock PS:4992-5002) ---

type heldMutex struct {
	h    windows.Handle
	name string
}

func (m *heldMutex) release() {
	if m != nil && m.h != 0 {
		windows.ReleaseMutex(m.h)
		windows.CloseHandle(m.h)
		m.h = 0
	}
}

// acquireNamedMutex tries Global\ then Local\ (Local only on access denied). An existing
// mutex is "lock already held". The handle keeps the name alive; ownership is the thread's
// that created it, and only existence is what another launcher tests (landing, deviation 7).
func acquireNamedMutex(name string) (*heldMutex, error) {
	for _, prefix := range []string{`Global\`, `Local\`} {
		p, err := windows.UTF16PtrFromString(prefix + name)
		if err != nil {
			return nil, err
		}
		h, err := windows.CreateMutex(nil, true, p)
		if err == windows.ERROR_ALREADY_EXISTS {
			windows.CloseHandle(h)
			return nil, fmt.Errorf("lock already held: %s%s", prefix, name)
		}
		if err == windows.ERROR_ACCESS_DENIED {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("mutex error: %v", err)
		}
		return &heldMutex{h: h, name: prefix + name}, nil
	}
	return nil, errors.New("mutex could not be created in either scope")
}

type heldFile struct{ h windows.Handle }

func (f *heldFile) release() {
	if f != nil && f.h != 0 {
		windows.CloseHandle(f.h)
		f.h = 0
	}
}

// acquireFileLock is Acquire-FileLock: OpenOrCreate, ReadWrite, FileShare None.
func acquireFileLock(path string) (*heldFile, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	h, err := windows.CreateFile(p, windows.GENERIC_READ|windows.GENERIC_WRITE, 0, nil,
		windows.OPEN_ALWAYS, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, fmt.Errorf("exclusive output lock unavailable: %v", err)
	}
	return &heldFile{h: h}, nil
}

// --- the owned child (Start-OwnedChild PS:11747-11873) ---

type winChild struct {
	pid      int
	h        windows.Handle
	job      windows.Handle
	newGroup bool
	outLog   string
	errLog   string
	cmd      *exec.Cmd
}

func (c *winChild) PID() int       { return c.pid }
func (c *winChild) NewGroup() bool { return c.newGroup }
func (c *winChild) ErrLog() string { return c.errLog }
func (c *winChild) OutLog() string { return c.outLog }

// Exited is Test-ChildExited (PS:11898-11904).
func (c *winChild) Exited() (uint32, bool) {
	var code uint32
	if err := windows.GetExitCodeProcess(c.h, &code); err != nil {
		return 0, false
	}
	if code == stillActive {
		return 0, false
	}
	return code, true
}

// WaitExit is Wait-ChildExit (PS:11906-11911).
func (c *winChild) WaitExit(d time.Duration) (uint32, bool) {
	r, _ := windows.WaitForSingleObject(c.h, uint32(d/time.Millisecond))
	if r != windows.WAIT_OBJECT_0 {
		return 0, false
	}
	return c.Exited()
}

// osProcessName is the image file name of a process (GL-08: the owner of a busy port). It opens
// the process for PROCESS_QUERY_LIMITED_INFORMATION only, so it can read and never end it.
func osProcessName(pid int) string {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ""
	}
	defer windows.CloseHandle(h)
	buf := make([]uint16, 1024)
	n := uint32(len(buf))
	if err := windows.QueryFullProcessImageName(h, 0, &buf[0], &n); err != nil {
		return ""
	}
	return filepath.Base(windows.UTF16ToString(buf[:n]))
}

// CtrlBreak is GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) (PS:12133).
func (c *winChild) CtrlBreak() error {
	return windows.GenerateConsoleCtrlEvent(ctrlBreakEvent, uint32(c.pid))
}

// ForceKill is "taskkill.exe /PID <pid> /T /F" then TerminateProcess(handle, 1) (PS:12158-12159).
func (c *winChild) ForceKill() {
	tk := filepath.Join(os.Getenv("SystemRoot"), "System32", "taskkill.exe")
	cmd := exec.Command(tk, "/PID", strconv.Itoa(c.pid), "/T", "/F")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	_ = cmd.Run()
	_ = windows.TerminateProcess(c.h, 1)
}

func (c *winChild) Memory() (ProcessMemory, error) { return processMemory(c.h) }

// Close is Close-OwnedChildHandles plus the job handle: closing the job kills whatever is
// still inside it (KILL_ON_JOB_CLOSE).
func (c *winChild) Close() {
	if c.h != 0 {
		windows.CloseHandle(c.h)
		c.h = 0
	}
	if c.job != 0 {
		windows.CloseHandle(c.job)
		c.job = 0
	}
}

// startOwnedChild is Start-OwnedChild for the server role: stdout/stderr to the two log
// files (CREATE_ALWAYS), stdin from NUL, CREATE_NEW_PROCESS_GROUP, the explicit environment
// block, the bundle root as working directory, the CRT-quoted command line, and the job
// object with KILL_ON_JOB_CLOSE; a job bind failure kills the child.
func startOwnedChild(exe string, argv []string, env []string, workDir, outLog, errLog string) (*winChild, error) {
	if st, err := os.Stat(exe); err != nil || st.IsDir() {
		return nil, fmt.Errorf("executable missing: %s", exe)
	}
	fo, err := os.OpenFile(outLog, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, errors.New("could not create redirected stdin/stdout/stderr handles")
	}
	fe, err := os.OpenFile(errLog, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		fo.Close()
		return nil, errors.New("could not create redirected stdin/stdout/stderr handles")
	}
	cmd := &exec.Cmd{Path: exe, Dir: workDir, Env: env, Stdout: fo, Stderr: fe}
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CmdLine:       commandLine(exe, argv),
		CreationFlags: createNewProcessGroup,
	}
	if err := cmd.Start(); err != nil {
		fo.Close()
		fe.Close()
		return nil, fmt.Errorf("CreateProcess failed: %v", err)
	}
	fo.Close()
	fe.Close()
	c := &winChild{pid: cmd.Process.Pid, newGroup: true, outLog: outLog, errLog: errLog, cmd: cmd}
	h, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE|
		windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.PROCESS_QUERY_INFORMATION|
		windows.PROCESS_VM_READ, false, uint32(c.pid))
	if err != nil {
		_ = cmd.Process.Kill()
		go cmd.Wait()
		return nil, fmt.Errorf("job object bind failed (process handle: %v); child killed to avoid an orphan", err)
	}
	c.h = h
	go cmd.Wait() // reaps Go's own handle; c.h stays valid for the exit code
	job, err := windows.CreateJobObject(nil, nil)
	bound := false
	if err == nil {
		var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
		info.BasicLimitInformation.LimitFlags = jobLimitKillOnJobClose
		if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
			uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err == nil {
			bound = windows.AssignProcessToJobObject(job, h) == nil
		}
	}
	if !bound {
		_ = windows.TerminateProcess(h, 1)
		if job != 0 {
			windows.CloseHandle(job)
		}
		c.Close()
		return nil, errors.New("job object bind failed (KILL_ON_JOB_CLOSE unset or assign failed); child killed to avoid an orphan")
	}
	c.job = job
	return c, nil
}

// fixedDriveModelRoots is the drive half of Get-ModelScanRoots (PS:11177-11184).
func fixedDriveModelRoots() []string {
	mask, err := windows.GetLogicalDrives()
	if err != nil {
		return nil
	}
	var out []string
	for i := 0; i < 26; i++ {
		if mask&(1<<uint(i)) == 0 {
			continue
		}
		root := string(rune('A'+i)) + `:\`
		p, _ := windows.UTF16PtrFromString(root)
		if windows.GetDriveType(p) != windows.DRIVE_FIXED {
			continue
		}
		conv := filepath.Join(root, modelsDirName)
		if st, err := os.Stat(conv); err == nil && st.IsDir() {
			out = append(out, conv)
		}
	}
	return out
}

// isHidden reports the hidden attribute: Get-ChildItem without -Force skips such entries.
func isHidden(path string) bool {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return false
	}
	a, err := windows.GetFileAttributes(p)
	return err == nil && a&windows.FILE_ATTRIBUTE_HIDDEN != 0
}

// osVolumeFree is Get-VolumeFreeMb (PS:3525-3534): the free (available to the caller) and
// total bytes of the volume holding path, rounded like PowerShell, with the volume root.
func osVolumeFree(path string) (freeMiB, totalMiB int64, root string, err error) {
	full, err := filepath.Abs(path)
	if err != nil {
		return 0, 0, "", err
	}
	root = filepath.VolumeName(full) + `\`
	p, err := windows.UTF16PtrFromString(root)
	if err != nil {
		return 0, 0, "", err
	}
	var avail, total, free uint64
	if err := windows.GetDiskFreeSpaceEx(p, &avail, &total, &free); err != nil {
		return 0, 0, root, err
	}
	return roundMiB(avail), roundMiB(total), root, nil
}
