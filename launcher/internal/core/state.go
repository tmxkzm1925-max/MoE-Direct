package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"moedirect/launcher/internal/version"
)

// Options configure a Launcher. CatalogPath and BundleRoot are separate because the installed
// staging has no models.json (the PowerShell launcher reads both from one root, PS:2290).
type Options struct {
	CatalogPath string // models.json
	BundleRoot  string // engine bundle: runtime.server_exe and runtime.expects_dir live under it
	StateDir    string // "" = Get-LauncherStateDir (PS:1777-1781)
	ModelsRoot  string // -ModelsRoot (PS:11169-11176), optional
	// Residuals is the store of the last measured c_dev and U (GL-05; nil: the file
	// need_residuals.json under the state root, F-BU B.7).
	Residuals ResidualStore
	// ProjectionValidation pins the «validated» state (nil: ProjectionMethodValidated); a
	// test of one display path sets it (S8-LA-13).
	ProjectionValidation *bool
}

// MemStatus is GlobalMemoryStatusEx in MiB (Get-MemStatus PS:3517-3524 for the two
// physical values; the commit pair is read for the screen only).
type MemStatus struct {
	TotalPhysMiB, AvailPhysMiB     int64
	CommitLimitMiB, CommitAvailMiB int64
}

// ProcessMemory is K32GetProcessMemoryInfo of the server child.
type ProcessMemory struct {
	WorkingSetBytes, PrivateBytes, PeakWorkingSetBytes int64
}

// FileIdentity is Get-FileIdentity's answer.
type FileIdentity struct {
	ID    string // "<volume x16>:<file id x32>"
	Size  int64
	MTime int64 // FILETIME
}

// osHooks are the OS reads the tests replace.
type osHooks struct {
	mem        func() (MemStatus, error)
	installed  func() (int64, error)
	identity   func(string) (FileIdentity, error)
	volumeGUID func(string) (string, error)
	portOwners func(int) ([]int, error)
	console    func() bool
	vram       func() VRAMInfo
	now        func() time.Time
	modelRoots func() []string
	volumeFree func(string) (int64, int64, string, error)
	// GOL-FIX1 (review items 4/14, 15, 24): the GPU Process Memory counter, the clipboard and
	// an outside process start.
	gpuMem    func() (map[string]int64, error)
	clipboard func(string) error
	launch    func(exe, cmdLine string) error
	fit       fitRunner // llama-fit-params (GL-05)
	// GL-08: the image name of a process that holds a start's port (read only, conflict row 6).
	procName func(pid int) string
	// listeners gives the local addresses listening on a port (smoke item 6; GL-11)
	listeners func(port int) ([]string, error)
	memLists  func() (MemoryLists, error)   // GL-09: the modified and free lists (PDH, LFS:302)
	taskmgr   func() error                  // GL-09: Task Manager, no selection (U-35 (i))
	perf      func() (CommitReading, error) // GL-06: the commit reader (GetPerformanceInfo, S57)
	sweep     ioMeter                       // GL-10: the timed reads of the speed check (sweep.go)
}

func defaultHooks() osHooks {
	return osHooks{mem: osMemStatus, installed: osInstalledMiB, identity: osFileIdentity,
		volumeGUID: osVolumeGUID, portOwners: osPortOwners, console: osConsoleAvailable,
		vram: osVRAM, now: time.Now, modelRoots: fixedDriveModelRoots, volumeFree: osVolumeFree,
		gpuMem: osGPUProcessMemory, clipboard: osClipboard, launch: osLaunch, procName: osProcessName, listeners: osPortListeners,
		memLists: osMemoryLists, taskmgr: osTaskManager,
		fit: runFitProcess, perf: osPerformanceInfo, sweep: osIOMeter{}}
}

// Launcher is the core of the Go launcher draft. One Launcher runs at most one server.
type Launcher struct {
	opt  Options
	os   osHooks
	mu   sync.Mutex
	sess *session // the running (or last) server, nil before the first start
	// lockPrefix is prepended to the named mutexes; "" in use (the PowerShell names).
	lockPrefix string
	diag       *diagLog
	// FIX2-CORE: the background bundle check (conflict 18-2), the first-once record of the
	// «작동 확인» setting (18-1) and the stop events of this launcher's servers (6).
	bundle *bundleCheck
	smoke  *smokeTracker
	stops  *stopLog
	// GL-14: the derived profiles of models outside the catalog (derive.go) and the repacker
	// runner of their plan (nil = ProcessExecutor; the tests set a fake).
	derive     deriveCache
	deriveExec RepackExecutor
	// GL-14: the sizes of the last look at the model folders (watch.go).
	grow growWatch
	// GL-16: the preset files whose discard was logged (preset.go).
	presets presetLog
}

// New makes a Launcher. Nothing is read or written until a call needs it.
func New(opt Options) (*Launcher, error) {
	if opt.CatalogPath == "" || opt.BundleRoot == "" {
		return nil, fmt.Errorf("catalog path and bundle root are both required")
	}
	if opt.StateDir == "" {
		opt.StateDir = LauncherStateDir()
	}
	l := &Launcher{opt: opt, os: defaultHooks(), bundle: newBundleCheck(), smoke: newSmokeTracker(), stops: &stopLog{}}
	l.diag = newDiagLog(l.LogDir(), l.os.now)
	return l, nil
}

// LauncherStateDir is Get-LauncherStateDir (PS:1777-1781): %LOCALAPPDATA%\MoE-Direct, or
// the temp directory when LOCALAPPDATA is unset.
func LauncherStateDir() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "MoE-Direct")
}

// StateDir is the state root %LOCALAPPDATA%\MoE-Direct, which the PowerShell launcher shares:
// its files there are read only for this package (the named lock objects are shared on
// purpose). This package's own files sit directly under it (F-LS 8.2-8.3; GL-FX2): the logs
// (LogDir) and the presets folders, the run history, its own digest cache and the measured
// store of F-BU B.7. No file goes under a go-draft folder, and a draft build's go-draft files
// are neither read nor migrated (the draft was never released).
func (l *Launcher) StateDir() string { return l.opt.StateDir }

// The folders under the state root %LOCALAPPDATA%\MoE-Direct (U-03, L:4264; edition 8.2 and
// 12.6, OQ-3 (a), First): the logs and the saved settings sit directly under the root.
const (
	StateLogsDir    = "logs"
	StatePresetsDir = "presets"
)

// LogDir is the log folder, <state>\logs: the diagnostic log, the server logs, the setup
// (repacker) logs and the server's metrics file (GL-17; S8-LA-36).
func (l *Launcher) LogDir() string { return filepath.Join(l.opt.StateDir, StateLogsDir) }

// LogFiles are the names of the files in the log folder, newest first (the log page lists
// them; S8-LA-36). A folder not made yet holds none.
func (l *Launcher) LogFiles() []string {
	es, err := os.ReadDir(l.LogDir())
	if err != nil {
		return nil
	}
	type file struct {
		name string
		at   time.Time
	}
	var fs []file
	for _, e := range es {
		if e.IsDir() {
			continue
		}
		var at time.Time
		if fi, err := e.Info(); err == nil {
			at = fi.ModTime()
		}
		fs = append(fs, file{e.Name(), at})
	}
	sort.SliceStable(fs, func(i, j int) bool {
		if !fs[i].at.Equal(fs[j].at) {
			return fs[i].at.After(fs[j].at)
		}
		return fs[i].name > fs[j].name
	})
	names := make([]string, len(fs))
	for i, f := range fs {
		names[i] = f.name
	}
	return names
}

// ServerLogPaths are the two log files of a server started at stamp, in the log folder.
func (l *Launcher) ServerLogPaths(stamp string) (outLog, errLog string) {
	return filepath.Join(l.LogDir(), fmt.Sprintf("server_%s_out.log", stamp)),
		filepath.Join(l.LogDir(), fmt.Sprintf("server_%s_err.log", stamp))
}

// --- the diagnostic log (Write-Diag PS:1765-1775, Initialize-DiagLog PS:1782-1791) ---

type diagLog struct {
	mu   sync.Mutex
	dir  string
	path string
	now  func() time.Time
}

func newDiagLog(dir string, now func() time.Time) *diagLog { return &diagLog{dir: dir, now: now} }

// Path is the diagnostic file of this Launcher ("" until the first record).
func (d *diagLog) Path() string { d.mu.Lock(); defer d.mu.Unlock(); return d.path }

func (d *diagLog) write(kind string, data any) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.path == "" {
		if err := os.MkdirAll(d.dir, 0o755); err != nil {
			return
		}
		d.path = filepath.Join(d.dir, fmt.Sprintf("launcher_%s_%d.jsonl",
			d.now().UTC().Format("20060102T150405Z"), os.Getpid()))
		d.append(diagHeaderKind, diagHeader())
	}
	d.append(kind, data)
}

// diagHeaderKind is the kind of the log header, the first record of every diagnostic file
// (the PowerShell launcher's first record is LAUNCH too, Initialize-DiagLog PS:1789).
const diagHeaderKind = "LAUNCH"

// diagHeader is the log header's data: the process, the launcher's name and its version
// (GL-01; F-LS 5.1: the version value in the log header; S8-LA-30).
func diagHeader() map[string]any {
	return map[string]any{"pid": os.Getpid(), "launcher": version.LauncherName, "version": version.LauncherVersion}
}

// append writes one record; the caller holds d.mu and has set d.path.
func (d *diagLog) append(kind string, data any) {
	rec := struct {
		TS   string `json:"ts"`
		Kind string `json:"kind"`
		Data any    `json:"data,omitempty"`
	}{d.now().UTC().Format(time.RFC3339Nano), kind, data}
	b, err := json.Marshal(rec)
	if err != nil {
		return
	}
	f, err := os.OpenFile(d.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	f.Write(append(b, '\r', '\n'))
	f.Close()
}

// writeFileAtomic writes via a temp file and a rename (Move-FileAtomic PS:2094).
func writeFileAtomic(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// MemNow reads the physical memory and the commit now (GlobalMemoryStatusEx), for the
// screens' live free-RAM values (decision ⑧ "free RAM now"; H8, H10). Added in GOL-WIRE: the
// adapter had no way to read the value core already reads inside ListModels and Preflight.
func (l *Launcher) MemNow() (MemStatus, error) { return l.os.mem() }

// VRAMNow is Get-VramInfo (PS:3550) now: display only, never a gate (conflict 8). Added in
// GOL-WIRE for the same reason as MemNow.
func (l *Launcher) VRAMNow() VRAMInfo { return l.os.vram() }
