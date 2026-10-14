// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"errors"
	"time"
)

// Backend is everything the screens show and do. The screens never compute a
// launcher value themselves: each value comes from one call below, and each
// call is made from a tea.Cmd, never from Update, so a slow call cannot block
// the key loop. A backend that has not ported a function returns ErrNotPorted
// (or leaves a value in the NotPorted state), and the screen draws the design's
// place with "not ported" instead of an example number.
type Backend interface {
	// Catalog lists the catalog models with presence, size, quant, identity
	// and the three H1 numbers (need, free RAM now, left after the start).
	Catalog() (Catalog, error) // cheap: 1 Models calls it every 5 s (names, sizes, modification times; no hashing; review item 31)
	// ModelSettings gives the current settings of one model; only rows with
	// Editable set can be changed on the screen (the backend's allowlist).
	ModelSettings(model string) (SettingsView, error)
	// Preflight checks one model with the settings the screen holds and
	// returns values, the verdict, refusals and notices. A refused preflight
	// is never offered a start.
	Preflight(model string, s Settings) (Preflight, error)
	// Start starts the model and streams its steps until /health is ready
	// (an event with Ready set) or the start fails (an event with Failure).
	// The channel is closed after the last event.
	Start(model string, s Settings) (<-chan StartEvent, error)
	// Status reads the serving server: /health, /props, /slots, the child
	// process and the address.
	Status() (ServeStatus, error)
	// Memory reads the memory values of the 3 Memory place.
	Memory() (MemoryReading, error)
	// LogTail gives the newest server and launcher log lines, oldest first.
	LogTail(max int) ([]LogLine, error)
	// Launcher gives the launcher settings of the 5 Settings place.
	Launcher() (LauncherInfo, error)
	// Stop stops the server by the given mode and streams its steps; the
	// last event carries the result. The channel is closed after it.
	Stop(mode StopMode) (<-chan StopEvent, error)
	// SavePreset saves the settings the screen changed for one model (only
	// the changed keys; review item 3) and gives the model's settings as they
	// read after the save, so the saved values become the defaults.
	SavePreset(model string, s Settings) (SettingsView, error)
	// CheckSetting checks one typed value against the setting's range (the
	// draft's PowerShell ranges) and gives it in the setting's own form, e.g.
	// "64k" → "65536"; an error's text is the reason shown in red (review
	// item 30).
	CheckSetting(model, key, value string) (string, error)
	// Action runs one control whose function lies outside the slice or in
	// the backend (copy address, redo setup, open Task Manager, ...).
	Action(id ActionID, arg string) (ActionResult, error)
	// RepackPlan gives the plan of the setup a model needs before it can
	// start: what it does, where it writes, the disk and the time (the
	// numbers of the PowerShell --plan). Nothing is written (conflict 9).
	RepackPlan(model string, s Settings) (RepackPlan, error)
	// Repack runs the planned setup after the user's ⏎, then the check, the
	// speed check and the server start; its events read as Start's, the
	// setup progress first (conflict 9).
	Repack(model string, s Settings) (<-chan StartEvent, error)
	// Bundle reads the check of the launcher's own files against its
	// manifest, which the backend runs from the launcher's start; it is read
	// again until it is done (conflict 18-2).
	Bundle() (BundleCheck, error)
}

// ErrNotPorted is what a backend returns for a function not ported yet
// (after the user's decisions).
var ErrNotPorted = errors.New("not ported (after the user's decisions)")

// ValState says where a number stands. The zero value is NotPorted, so a
// field the backend does not fill renders as "not ported", never as 0.
type ValState uint8

const (
	NotPorted  ValState = iota // the backend does not deliver this value yet
	Known                      // V holds the value
	Unmeasured                 // the backend knows it has no measurement
)

// Val is a number with its state.
type Val struct {
	V     float64
	State ValState
}

// K makes a known value.
func K(v float64) Val { return Val{V: v, State: Known} }

// U is the unmeasured value.
var U = Val{State: Unmeasured}

// Ok reports a known value.
func (v Val) Ok() bool { return v.State == Known }

// Fit is the model list's fit mark.
type Fit uint8

const (
	FitNotPorted Fit = iota
	FitUnmeasured
	FitRoom
	FitTight
	FitTooBig
	FitBroken      // the file differs from its reference, a setup was cut off, ... (review item 11)
	FitUnsupported // the model's structure is not supported (review item 11)
)

// FileCheck is the "file check" row of THIS MODEL (review item 10): whether
// the catalog's tuned settings are meant for this very file.
type FileCheck uint8

const (
	FileCheckNotPorted FileCheck = iota
	FileSame                     // the file hash equals the catalog's pinned hash
	FileNoReference              // the catalog pins no hash for this model
	FileDiffers                  // the file hash differs from the pinned hash
	FileNotChecked               // the hash is not computed yet
)

// SetupState is the SETUP tile of 1 Models.
type SetupState uint8

const (
	SetupNotPorted SetupState = iota
	SetupReady
	SetupNeeded
	SetupCutOff
	SetupNeedsRepack // the repack output was deleted after the setup: «needs repack» (review item 31)
)

// ModeState is the state of one repack mode's output (conflict 16).
type ModeState uint8

const (
	ModeNotPorted   ModeState = iota
	ModeReady                 // the output is there and its check passed: «packed ✓»
	ModeNeedsRepack           // no output for this mode: «virtual needs repack»
	ModeCheckFailed           // the output's check failed: «packed ✕ check failed»
	ModeStale                 // W7 duty 12 (GL-10): made from other model files: «packed ✕ stale (another file)»
)

// ModeCheck is one repack mode of a model with its state (conflict 16).
type ModeCheck struct {
	Mode  string // "packed", "virtual"
	State ModeState
}

// RepackPlan is the plan box of 2 Server before a setup (conflict 9).
type RepackPlan struct {
	Model, Mode string
	Does        string // what the setup does, in the user's words
	WritesTo    string // the folder it writes into
	WriteGB     Val    // about how much it writes
	DriveFreeGB Val    // free on that drive now
	LeftGB      Val    // free after it; below zero (or WriteGB > DriveFreeGB) refuses
	LastS       Val    // the last setup's time in seconds; not Ok on the first one
}

// BundleState is the state of the launcher's own file check (conflict 18-2).
type BundleState uint8

const (
	BundleNotPorted BundleState = iota
	BundleChecking              // still running: a start waits for it
	BundleOK
	BundleBroken // files differ from the manifest: «✕ the launcher files are damaged · n differ · install again»
	BundleDev    // no manifest (a development build): «development build · file check skipped»
)

// BundleCheck is the result of the launcher's own file check.
type BundleCheck struct {
	State      BundleState
	Mismatches int
}

// SmokeMode is the 5 Settings row «check run: off / on / once» (conflict 18-1).
type SmokeMode uint8

const (
	SmokeNotPorted SmokeMode = iota
	SmokeOff                 // the default
	SmokeOn                  // after every start
	SmokeOnce                // once per model and mode after the launcher opens
)

// StopRecord is the launcher's line in 4 Log for one server stop
// (conflict 6): the launcher's code and the server child's exit code.
type StopRecord struct {
	Code       int    // 0 normal, 2 stopped before ready, 5 forced; other: failed
	ServerCode Val    // the server child's exit code, when known
	Meaning    string // the code's meaning from the PowerShell table, for codes other than 0, 2 and 5
}

// PartialSetup is a setup that was cut off (H4).
type PartialSetup struct {
	LayersDone, LayersTotal Val
	FilesGB                 Val
	RedoMin                 Val
	When, Cause             string
}

// ModelRow is one catalog model.
type ModelRow struct {
	ID, Name     string
	Present      bool
	Catalog      bool // a catalog model (tuned settings), drawn with a star (review items 11, 12)
	Path, Folder string
	SizeGB       Val
	Quant        string // "" when not delivered
	Identity     string // e.g. "sha256 3f9a…c21e"; "" when not delivered (shown in the detail views)
	FileCheck    FileCheck
	// The three H1 numbers (decision ⑧): need, free RAM now, left after start.
	NeedGB, FreeRAMGB, LeftGB Val
	CacheGB                   Val // the expert-cache budget part of the need
	VRAMUseGB, VRAMTotalGB    Val // VRAM in use on the whole card (nvidia-smi) and its size
	Fit                       Fit
	DecodeTokS, PrefillTokS   Val // DecodeTokS: the last run's average reply decode (review item 28)
	TTFTs                     Val
	MeasuredOn                string // the date of that last run
	Setup                     SetupState
	SetupNote                 string
	Partial                   PartialSetup
	LastRun                   string
	Template                  bool        // outside the catalog, started with the structure's defaults (conflict 14)
	Arch                      string      // the GGUF's architecture, for a Template row's words (UX-R10)
	Modes                     []ModeCheck // each repack mode's output and check (conflict 16)
}

// FolderState is one watched folder's state (H3, H5).
type FolderState uint8

const (
	FolderNotPorted FolderState = iota
	FolderWatching
	FolderMissing
	FolderEmpty
	FolderUnsupported
)

// Folder is one model folder.
type Folder struct {
	Path    string
	State   FolderState
	Found   int
	SizeGB  Val
	Note    string
	Skipped []string // "name  why" lines
}

// Catalog is the model list.
type Catalog struct {
	Models  []ModelRow
	Folders []Folder
	TooBig  Val // models this PC cannot run
	Source  string
}

// Settings holds the values the screen changed, by key.
type Settings map[string]string

// SettingRow is one model setting.
type SettingRow struct {
	Key, Label, Value string
	Options           []string // values ←→ cycles through, when Editable
	Editable          bool
	Locked            string // why it is locked; "" when it is not
	Note, Hint        string
	Folded            bool   // drawn after the leading rows of the box (UX-R10: no fold any more)
	Running           string // the value the last actual start used ("" unknown); the "now" marks compare with it (review item 2)
	Typed             bool   // Space opens a box for a typed value (review item 30)
	Default           string // the catalog's (or the structure's) default; "" unknown (conflict 10)
	Unit              string // the unit drawn after the box, e.g. "G"
	Range             string // the range drawn after the box, e.g. "12 G ~ 20 G"
	Basis             string // what bounds Note's range, a table key («range = …»); "" none (UX-R6)
}

// SettingsView is one model's settings.
type SettingsView struct {
	Model   string
	Preset  string
	Rows    []SettingRow
	Saved   bool     // the model has a saved profile (ctrl+s): «my settings», else «default» (conflict 3)
	LastRun Settings // the values the model's last run used; its decode speed was measured with them (conflict 10)
}

// Verdict is the pre-start memory verdict (H14).
type Verdict uint8

const (
	VerdictNotPorted Verdict = iota
	VerdictUnmeasured
	VerdictRoom
	VerdictTight
	VerdictCannot
)

// Preflight is the check before a start.
type Preflight struct {
	Model                   string
	AvailGB, NeedGB, LeftGB Val
	NeedLabel               string // "measured", "predicted before run" or "" (S8-LA-13; GL-05; UX-R10)
	// LeftIfWrittenGB is free now + the modified memory − the need (F-BU G.2; GL-09), U while
	// the need or the modified memory is not measured
	LeftIfWrittenGB                  Val
	ModelGB, ReplyGB, CacheGB        Val
	CommitLeftAfterGB, CommitLimitGB Val
	RAMTotalGB, AppsGB, ModifiedGB   Val
	StandbyGB, FreeGB                Val
	VRAMTotalGB, VRAMUseGB           Val
	RoomLineGB, CommitSafeGB         Val
	// GL-06 (F-BU F): the commit total now, this start's predicted commit, the expert pool on its
	// own line (FL:14) and the near-limit line's numbers (U-13 [A-NEAR]: shown; nothing is
	// changed and nothing refused); Near is nil while the room after the start is at or above
	// the safe line or not measured
	CommitNowGB, PredictedGB, PoolGB Val
	Near                             *NearLine
	Verdict                          Verdict
	Refused                          bool
	Refusals, Notices                []string
	LastStartS                       Val
	DecodeTokS                       Val
	SettingsLine                     string
	BudgetCapGB                      Val // a refused start passes with a budget of this much or less (conflict 7)
}

// NearLine is F-BU F.3's warning line (core.NearLimit in G): the room after the start counted
// without the prompt-cache term (Q-5 (c)), the safe line, the commit limit, the total now, this
// start's predicted commit without the cache term and its expert pool.
type NearLine struct {
	RoomGB, SafeGB, LimitGB, NowGB, PredictedGB, PoolGB float64
}

// StepState is the state of one start, setup or stop step.
type StepState uint8

const (
	StepTodo StepState = iota
	StepRunning
	StepDone
	StepFailed
	StepSkipped
	StepNotPorted
)

// Step is one step of a start or a stop.
type Step struct {
	Name, Detail string
	State        StepState
	Took         time.Duration
}

// SetupProgress is a first-run setup in progress (H6).
type SetupProgress struct {
	Name               string
	Done, Total        Val
	Elapsed            time.Duration
	LastS              Val
	DiskMBs            Val
	WrittenGB, TotalGB Val
}

// Fix is one way out of a failed start.
type Fix struct {
	Key, Label string
	Action     ActionID // "" means: start again
}

// Failure is a failed start (H7) or a server that ended while serving.
type Failure struct {
	Stage, What, Why string
	Holder           string
	PID              int
	At               time.Time
	Fixes            []Fix
	LogTail          []string
	LogFile          string
	ExitCode         int
	VRAMOut          bool // the load ran out of VRAM (conflict 8)
	SmokeFailed      bool // the check run after the start failed (conflict 18-1); Why says how
}

// StartEvent is one report of a start in progress.
type StartEvent struct {
	Model   string
	Steps   []Step
	Elapsed time.Duration // since the launcher started the server (decision ③)
	LastS   Val           // last measured start → ready for this model and settings
	// The load bar's span (GL-07, decision ③): it runs from the server child's start
	// (LoadRunning) to /health ready (LoadReady, LoadTook long); before the child starts it has
	// not begun.
	LoadRunning bool
	LoadReady   bool
	LoadTook    time.Duration
	Warm        bool // a warm start (H15 shows "warm start", no clock time)
	DiskGBs     Val
	RAMGB       Val // the child's memory so far
	LastRAM     Val // the same model's memory at ready last time
	Setup       *SetupProgress
	Ready       bool
	Failure     *Failure
	Notices     []string
	Address     string
}

// ReqPhase is where the request in flight stands.
type ReqPhase uint8

const (
	ReqNone ReqPhase = iota
	ReqPrefill
	ReqDecode
)

// ReqSample is one finished request from the server's timing lines (GL-18, LFS Q3.64 and Q4.A24):
// when it ended, its prompt rate, its time to first token and its whole-answer decode rate. The
// screen keeps them in memory only (U-36 (i), L:4275); no file is written for them.
type ReqSample struct {
	Task        int
	At          time.Time
	PrefillTokS float64
	TTFTs       float64
	DecodeTokS  float64
}

// ServeStatus is the serving server.
type ServeStatus struct {
	// Reqs are the finished requests the server's timing lines still hold, oldest first (GL-18).
	Reqs                  []ReqSample
	Model, Alias, Address string
	Port, ChildPID        int
	ChildAlive            bool
	Health                string // /health answer: "ok", "loading", "error", ...
	Uptime                time.Duration
	CtxTotal, CtxUsed     Val
	Phase                 ReqPhase
	DecodeTokS            Val // live, the last seconds' average
	LastDecodeTokS        Val
	UsualTokS             Val
	PrefillDone           Val
	PrefillTotal          Val
	PrefillTokS           Val
	ReusedTokens          Val
	NewTokens             Val
	DecodedTokens         Val
	Batch                 Val // UX-R26: the -b the server was started with (the started argv)
	TTFTs                 Val
	Requests              Val
	Errors                Val
	Waiting               Val
	FreeRAMGB, FreeVRAMGB Val
	ModelRAMGB            Val // this server's RAM
	OtherRAMGB            Val
	TotalRAMGB            Val
	ModelVRAMGB           Val // this server's VRAM (Windows GPU process counter)
	OtherVRAMGB           Val
	TotalVRAMGB           Val
	UsedVRAMGB            Val // VRAM in use on the whole card (nvidia-smi); never added to ModelVRAMGB (review item 4)
	Slow                  bool
	SlowNote              string
	SettingsLine          string   // the machine form (kept for the log)
	SettingsWords         []string // the same settings in plain words: "64k context", "12 G cache", "virtual" (review item 16)
	// GL-21: the prefetch row, the engine's values after ready then the launcher's candidate
	// (F-PF §8.5); each piece is translated on its own
	Prefetch []string
	Exited   bool
	ExitCode int
}

// AppMem is one app on the Memory page.
type AppMem struct {
	PID      int // GL-09: named on the Task Manager line (U-35 (i))
	Name     string
	GB       Val
	ChangeGB Val
	Tag      string
}

// MemoryReading is the 3 Memory place while a server runs (H10).
type MemoryReading struct {
	TotalRAMGB, ModelGB, AppsGB     Val
	ModifiedGB, StandbyGB, FreeGB   Val
	TotalVRAMGB, ModelVRAMGB        Val // ModelVRAMGB: this server (Windows GPU process counter)
	OtherVRAMGB                     Val
	UsedVRAMGB                      Val // the whole card in use (nvidia-smi); shown apart from ModelVRAMGB, never added (review item 4)
	CommitUsedGB, CommitLimitGB     Val
	CommitSafeGB                    Val
	PagefileMBs                     Val
	PagefileNote                    string
	ReadyNowGB                      Val // available now (free + standby); 1 Models reads it at every draw (review item 13)
	ExpertCacheGB, BudgetGB         Val
	FileMapGB, OtherGB              Val
	Apps                            []AppMem
	AppsMore                        int
	Causes                          []string
	LauncherGB, LauncherCPU, EveryS Val
}

// LogLevel marks a log line.
type LogLevel uint8

const (
	LogInfo LogLevel = iota
	LogOK
	LogWarn
	LogError
	LogNotice // an automatic adjustment worth seeing, e.g. the engine raising a queue depth (review item 29)
)

// LogLine is one line of the 4 Log place.
type LogLine struct {
	At    time.Time
	Level LogLevel
	Text  string
	Stop  *StopRecord // set on the launcher's line for a server stop (conflict 6)
}

// LauncherInfo is the 5 Settings place.
type LauncherInfo struct {
	Address      string // this PC only: no network option exists, not even a locked one (rule 17, GL-12)
	OnOpen       string
	LogsKeptDays Val
	SetupDir     string
	SetupUsedGB  Val
	SetupFreeGB  Val
	LogDir       string
	LogFiles     []string // the files in LogDir, newest first (S8-LA-36)
	LogFile      string
	SettingsDir  string
	Version      string
	Smoke        SmokeMode // «check run» (conflict 18-1)
	// GL-16: the launcher settings the same for every model (5 Settings): the prompt-cache cap
	// («off», the default of DECISION_LEDGER.md:4420, or MiB; U-22) and the warm-start switches («on»/«off»; OQ-6 (a)).
	CacheCap string
	// CacheCapMax is F-BU E.3's allowed maximum for a positive cap (MiB) at the last check, shown
	// beside the choice and never used as the default (L:4420; GL-06); U before a check gave one
	CacheCapMax Val
	WarmStart   string
	Autosave    string
	// NextStart names the launcher settings that differ from the running server's (rows
	// 6.1-6.2: «not applied to the running server · applies at the next start»).
	NextStart map[string]bool
}

// StopMode is the way a server is stopped.
type StopMode uint8

const (
	StopUnload StopMode = iota // ctrl+f: unload, the launcher stays
	StopQuit                   // q twice: unload, then the launcher closes
	StopForce                  // ctrl+c twice: close now, no cache save
	StopCancel                 // ctrl+x while starting or in setup
)

// SaveProgress is the cache-save bar (decision ③, same rule as the load bar; GL-07). It spans
// the quit action to the save-completion response: State StepRunning while the save runs,
// StepDone at the response (the bar shows 100 % only then), StepNotPorted while the stop runs no
// save. Elapsed is since the quit action; LastS is the expectation (the last save's time per
// token × Tokens), unmeasured on a first save (elapsed seconds only).
type SaveProgress struct {
	State   StepState
	Elapsed time.Duration
	LastS   Val
	Tokens  Val
}

// StopEvent is one report of a stop in progress.
type StopEvent struct {
	Mode       StopMode
	Steps      []Step
	Elapsed    time.Duration
	Save       SaveProgress
	ModelRAMGB Val
	Done       bool
	Result     *StopResult
}

// RunSummary is the run summary after a stop (H12).
type RunSummary struct {
	Model                     string
	Started, Ended            time.Time
	Requests, Errors          Val
	DecodeAvg, DecodeLow      Val
	PrefillAvg, PrefillTokens Val
	TTFTMedian, TTFTMax       Val
	CacheTokens               Val
}

// StopResult is the end of a stop.
type StopResult struct {
	ExitCode int
	Graceful bool
	Message  string
	Run      RunSummary
}

// ActionID names one control handled by Backend.Action.
type ActionID string

const (
	ActCopyAddress   ActionID = "copy-address"
	ActCopyReport    ActionID = "copy-report"
	ActOtherPort     ActionID = "other-port"
	ActRedoSetup     ActionID = "redo-setup"
	ActDeletePartial ActionID = "delete-partial"
	ActTaskManager   ActionID = "task-manager"
	ActOpenLog       ActionID = "open-log"
	ActSavePreset    ActionID = "save-preset"
	ActAddFolder     ActionID = "add-folder"
	ActRemoveFolder  ActionID = "remove-folder"
	ActLauncherSet   ActionID = "launcher-setting"
	ActGraph         ActionID = "graph"
	ActOpenFolder    ActionID = "open-folder" // Explorer opens the folder (review item 24)
	ActShowFile      ActionID = "show-file"   // Explorer opens with the file selected (review item 24)
	ActSetSmoke      ActionID = "set-smoke"   // arg "off", "on" or "once" (conflict 18-1)
	// GL-16: the launcher settings of 5 Settings.
	ActSetCacheCap  ActionID = "set-cache-cap" // arg «off» (the default) or the cap in MiB (U-22)
	ActSetWarmStart ActionID = "set-warmstart" // arg "on" or "off" (OQ-6 (a))
	ActSetAutosave  ActionID = "set-autosave"  // arg "on" or "off" (OQ-6 (a))
)

// ActionResult is what an action says back.
type ActionResult struct {
	Message string
}
