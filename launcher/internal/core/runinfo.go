package core

import (
	"errors"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Review items 2, 13, 22, 27 and 28 of reviews/_gol_review_8_user_20261010.md: what the last
// actual start used, the need rule, the whole-PC memory, the start verdict line's values and
// the server's API name.

// StartSettings are the settings a start actually used, read back from its argv and env
// (item 2: the screen compares against these, not against saved defaults).
type StartSettings struct {
	RepackMode string `json:"repack_mode"`
	Ctx        int64  `json:"ctx"`
	Threads    int64  `json:"threads"`
	Port       int    `json:"port"`
	BudgetMB   int64  `json:"budget_mb"`
	QD         int    `json:"qd"`
	PrefetchOn bool   `json:"prefetch_on"`
	PrefetchK  int64  `json:"prefetch_k,omitempty"`
	PrefetchN  int64  `json:"prefetch_n,omitempty"`
	Prefetch   string `json:"prefetch"` // "on" or "off(reason=...)"
	Alias      string `json:"alias"`
	// GL-16: the launcher settings this start took (5 Settings; rows 6.1-6.2): the cap as
	// «off» (the default, DECISION_LEDGER.md:4420) or MiB, the warm-start switches «on» or
	// «off». Empty in older records.
	PromptCacheCap string `json:"prompt_cache_cap,omitempty"`
	WarmStart      string `json:"warmstart,omitempty"`
	Autosave       string `json:"autosave,omitempty"`
	// GL-19: the prefetch echo of F-PF §4.4 (P4:289-294), the effective record the status
	// lines print the same values from (S8-PF-04). Absent in older records.
	PrefetchEcho *PrefetchEcho `json:"prefetch_echo,omitempty"`
}

// startSettingsOf reads a Config back.
func startSettingsOf(cfg *Config, repackMode string) StartSettings {
	s := StartSettings{RepackMode: repackMode, Port: cfg.Port, BudgetMB: cfg.Budget.MB, QD: cfg.QD,
		PrefetchOn: cfg.Prefetch.On, Prefetch: cfg.Prefetch.Echo}
	if cfg.Prefetch.On {
		s.PrefetchK, s.PrefetchN, s.Prefetch = cfg.Prefetch.K, cfg.Prefetch.N, "on"
	}
	if v, ok := argvValue(cfg.Argv, "-c"); ok {
		s.Ctx, _ = strconv.ParseInt(v, 10, 64)
	}
	if v, ok := argvValue(cfg.Argv, "-t"); ok {
		s.Threads, _ = strconv.ParseInt(v, 10, 64)
	}
	s.Alias, _ = argvValue(cfg.Argv, "--alias")
	if v, ok := cfg.Env[EnvQD]; ok {
		s.QD, _ = strconv.Atoi(v)
	}
	if v, ok := cfg.Env[EnvBudgetMB]; ok {
		s.BudgetMB, _ = strconv.ParseInt(v, 10, 64)
	}
	e := echoOf(cfg.Prefetch, s.QD)
	s.PrefetchEcho = &e
	return s
}

// SettingsMap is the start settings as the request's setting keys (SettingKeys plus
// repack_mode), for a comparison with what the screen holds.
func (s StartSettings) SettingsMap() map[string]string {
	m := map[string]string{
		"ctx": strconv.FormatInt(s.Ctx, 10), "threads": strconv.FormatInt(s.Threads, 10),
		"port": strconv.Itoa(s.Port), "budget_mb": strconv.FormatInt(s.BudgetMB, 10),
		"qd": strconv.Itoa(s.QD), "repack_mode": s.RepackMode,
	}
	for k, v := range map[string]string{"prompt_cache_cap": s.PromptCacheCap, "warmstart": s.WarmStart, "autosave": s.Autosave} {
		if v != "" {
			m[k] = v
		}
	}
	return m
}

// LastStart is the settings of the newest start of a model file, and when it was ready.
func (l *Launcher) LastStart(modelPath string) (StartSettings, time.Time, bool) {
	rec, ok := l.LatestForModel(modelPath)
	if !ok || rec.Start == nil {
		return StartSettings{}, time.Time{}, false
	}
	at, _ := time.Parse(time.RFC3339, rec.At)
	return *rec.Start, at, true
}

// ModelAlias is the short API name passed as --alias (item 27): the catalog profile id, or
// outside the catalog the file name without its extension (and without the split suffix).
func ModelAlias(profileID, modelPath string) string {
	if profileID != "" {
		return profileID
	}
	base := filepath.Base(modelPath)
	if m := splitRegex.FindStringSubmatch(base); m != nil {
		return m[1]
	}
	return strings.TrimSuffix(base, filepath.Ext(base))
}

// NeedBasis says where a need value comes from (item 13).
type NeedBasis string

const (
	NeedPeak        NeedBasis = "peak"         // the last run's maximum memory of the server
	NeedReadyBudget NeedBasis = "ready+budget" // no peak recorded: the ready value + the RAM budget
	NeedNotMeasured NeedBasis = "not measured" // no run recorded
	// GL-05, once the projection is validated (S8-LA-13)
	NeedFromProjection NeedBasis = "projection"            // the projection, a measured structure and setting
	NeedFromUnverified NeedBasis = "unverified prediction" // the projection of a structure not measured
)

// NeedOf applies item 13's rule to a run record: the last run's maximum working set; with no
// maximum recorded, the working set at ready plus the RAM budget of that run; with no
// record, not measured.
func NeedOf(rec *RunRecord) (int64, NeedBasis) {
	if rec == nil {
		return 0, NeedNotMeasured
	}
	if rec.MaxWorkingSetBytes > 0 {
		return ceilMiB(rec.MaxWorkingSetBytes), NeedPeak
	}
	if rec.WorkingSetBytes > 0 {
		budget := int64(0)
		if rec.Start != nil {
			budget = rec.Start.BudgetMB
		}
		return ceilMiB(rec.WorkingSetBytes) + budget, NeedReadyBudget
	}
	return 0, NeedNotMeasured
}

// PCMemory is the whole PC's RAM now (item 13, 14): measured at every call.
type PCMemory struct {
	TotalMiB                       int64
	AvailMiB                       int64 // available physical memory (GlobalMemoryStatusEx)
	UsedMiB                        int64 // total - available
	ServerMiB                      int64 // this launcher's server working set; 0 when none serves
	OthersMiB                      int64 // used - server
	ServerPID                      int   // this launcher's serving server; 0 when none serves (GL-09, U-35)
	Lists                          MemoryLists
	ListsOK                        bool // GL-09: the modified and free lists were read (U-21)
	CommitLimitMiB, CommitAvailMiB int64
	At                             time.Time
}

// FreeIfStoppedMiB is the RAM available once this launcher's server is stopped.
func (m PCMemory) FreeIfStoppedMiB() int64 { return m.AvailMiB + m.ServerMiB }

// PCMemoryNow reads the RAM now, with the serving server's share apart.
func (l *Launcher) PCMemoryNow() (PCMemory, error) {
	ms, err := l.os.mem()
	if err != nil {
		return PCMemory{}, err
	}
	m := PCMemory{TotalMiB: ms.TotalPhysMiB, AvailMiB: ms.AvailPhysMiB, CommitLimitMiB: ms.CommitLimitMiB,
		CommitAvailMiB: ms.CommitAvailMiB, At: l.os.now()}
	m.UsedMiB = max(0, m.TotalMiB-m.AvailMiB)
	if pm, ok := l.serverMemory(); ok {
		m.ServerMiB = ceilMiB(pm.WorkingSetBytes)
	}
	m.OthersMiB = max(0, m.UsedMiB-m.ServerMiB)
	if m.ServerMiB > 0 {
		m.ServerPID = l.serverPID()
	}
	m.Lists, m.ListsOK = l.readMemoryLists()
	return m, nil
}

// serverMemory is the running server's memory, when one runs.
func (l *Launcher) serverMemory() (ProcessMemory, bool) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return ProcessMemory{}, false
	}
	s.mu.Lock()
	child, ended := s.child, s.ended
	s.mu.Unlock()
	if child == nil || ended {
		return ProcessMemory{}, false
	}
	if _, exited := child.Exited(); exited {
		return ProcessMemory{}, false
	}
	pm, err := child.Memory()
	if err != nil {
		return ProcessMemory{}, false
	}
	return pm, true
}

// ServerVRAMNow is this launcher's server's dedicated GPU memory now (item 4, 14), from the
// Windows counter \GPU Process Memory(pid_<PID>_*)\Dedicated Usage (nvidia-smi gives [N/A] for
// per-process memory on this PC). It is reported apart from nvidia-smi's used value: the two
// sources count differently and are never added.
func (l *Launcher) ServerVRAMNow() (int64, error) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return 0, ErrNoServer
	}
	s.mu.Lock()
	child, ended := s.child, s.ended
	s.mu.Unlock()
	if child == nil || ended {
		return 0, ErrNoServer
	}
	inst, err := l.os.gpuMem()
	if err != nil {
		return 0, err
	}
	b, ok := gpuShareOf(inst, child.PID())
	if !ok {
		return 0, errors.New("no GPU Process Memory instance for the server's PID")
	}
	return b, nil
}

// gpuShareOf sums the counter instances of one PID ("pid_<PID>_luid_..._phys_<n>").
func gpuShareOf(inst map[string]int64, pid int) (int64, bool) {
	prefix := "pid_" + strconv.Itoa(pid) + "_"
	var sum int64
	found := false
	for name, v := range inst {
		if strings.HasPrefix(name, prefix) {
			sum += v
			found = true
		}
	}
	return sum, found
}

// StartCheckKind is the verdict of item 22's line.
type StartCheckKind uint8

const (
	CheckUnmeasured StartCheckKind = iota // need not measured yet
	CheckRoom                             // starts; Spare is what is left for other work
	CheckTight                            // starts, but the RAM left after the start is below the safe line
	CheckBlocked                          // refused: the budget alone exceeds the free RAM (PowerShell's hard stop)
)

func (k StartCheckKind) String() string {
	return [...]string{"unmeasured", "room", "tight", "blocked"}[k]
}

// SpareComfortGiB is item 22's threshold: with at least this much to spare the screen says so
// in its friendly sentence; below it, "startable · RAM for other work n G".
const SpareComfortGiB = 4

// StartCheck carries the values of item 22's verdict line.
type StartCheck struct {
	Kind      StartCheckKind
	Reason    string // blocked: "budget"; else ""
	BudgetMiB int64
	FreeMiB   int64 // the RAM the check used: available now (+ this server's share when it serves)
	NeedMiB   int64
	NeedBasis NeedBasis
	LeftMiB   int64 // free - need
	// SafeLineMiB is the preflight safe line. PowerShell's RAM_HEADROOM_MB is [unmeasured]
	// (Start-MoeDirect.ps1:1174), so it is 0 with SafeLineKnown false until it is measured.
	SafeLineMiB   int64
	SafeLineKnown bool
	SpareGiB      int64 // x = floor((left - safe line) / 1 GiB), item 22
	// MaxBudgetGiB is conflict 7: on a budget refusal, the largest budget in G (1024 MB) at which
	// the start would pass, floor(free / 1024) - the refusal is budget > free (items 13, 22), so
	// any budget up to the free RAM passes. Preflight clamps it to the profile's budget bounds
	// (BudgetCap); 0 on the other kinds.
	MaxBudgetGiB int64
}

// EvaluateStart computes item 22's values: blocked when the budget alone exceeds the free RAM
// (PowerShell's only refusal on RAM, conflict 7); otherwise, with a known need, tight when the
// RAM left after the start is below the safe line, else room with x to spare.
func EvaluateStart(freeMiB, needMiB int64, basis NeedBasis, budgetMiB int64) StartCheck {
	c := StartCheck{BudgetMiB: budgetMiB, FreeMiB: freeMiB, NeedMiB: needMiB, NeedBasis: basis}
	if budgetMiB > freeMiB {
		c.Kind, c.Reason = CheckBlocked, "budget"
		c.MaxBudgetGiB = max(0, freeMiB) / 1024
		return c
	}
	if basis == NeedNotMeasured {
		return c
	}
	c.LeftMiB = freeMiB - needMiB
	spare := c.LeftMiB - c.SafeLineMiB
	if spare < 0 {
		c.Kind = CheckTight
		c.SpareGiB = -((-spare + 1023) / 1024)
		return c
	}
	c.Kind, c.SpareGiB = CheckRoom, spare/1024
	return c
}

// ModelStats is a model's last run, per model file (item 28: the 1 Models tok/s tile).
type ModelStats struct {
	ModelPath       string  `json:"model_path"`
	DecodeTokPerSec float64 `json:"decode_tok_per_sec"` // the run's answers: tokens over time
	AnswerTokens    int64   `json:"answer_tokens"`
	Requests        int     `json:"requests"`
	PeakWorkingSet  int64   `json:"peak_working_set_bytes"`
	At              string  `json:"at"` // when the run ended
}

// ModelStatsFor is the last run's stats of a model file.
func (l *Launcher) ModelStatsFor(modelPath string) (ModelStats, bool) {
	historyMu.Lock()
	defer historyMu.Unlock()
	st, ok := readHistory(l.historyPath()).Models[strings.ToLower(modelPath)]
	return st, ok
}

// recordRunEnd writes what a run learned at its end: the server's peak memory into the
// settings key's record (item 13) and, when at least one request finished, the decode average
// and date per model (item 28).
func (l *Launcher) recordRunEnd(key, modelPath string, peak int64, sum TimingSummary, at time.Time) error {
	historyMu.Lock()
	defer historyMu.Unlock()
	d := readHistory(l.historyPath())
	if r, ok := d.Records[key]; ok && peak > 0 {
		r.MaxWorkingSetBytes = peak
		d.Records[key] = r
	}
	if avg, ok := sum.RunDecodeAvg(); ok {
		if r, ok := d.Records[key]; ok {
			r.DecodeTokPerSec = avg
			d.Records[key] = r
		}
		if d.Models == nil {
			d.Models = map[string]ModelStats{}
		}
		d.Models[strings.ToLower(modelPath)] = ModelStats{ModelPath: modelPath, DecodeTokPerSec: avg,
			AnswerTokens: sum.AnswerTokens, Requests: sum.Requests, PeakWorkingSet: peak, At: at.UTC().Format(time.RFC3339)}
	}
	return writeHistory(l.historyPath(), d)
}

// ServerLog is the running (or last) server's stderr, parsed (items 5, 18, 29): the newest n
// lines with their clock time and level.
func (l *Launcher) ServerLog(n int) ([]ServerLine, error) {
	r := l.sessionLog()
	if r == nil {
		return nil, ErrNoServer
	}
	return r.tail(n), nil
}

// Timings is the running (or last) server's print_timing summary (items 9, 23).
func (l *Launcher) Timings() (TimingSummary, error) {
	r := l.sessionLog()
	if r == nil {
		return TimingSummary{}, ErrNoServer
	}
	return r.summary(), nil
}

func (l *Launcher) sessionLog() *serverLog {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.log
}

// samplePeak reads the server's memory and keeps the session's maximum (item 13).
func (s *session) samplePeak() {
	s.mu.Lock()
	child := s.child
	s.mu.Unlock()
	if child == nil {
		return
	}
	pm, err := child.Memory()
	if err != nil {
		return
	}
	v := max(pm.WorkingSetBytes, pm.PeakWorkingSetBytes)
	s.mu.Lock()
	if v > s.peak {
		s.peak = v
	}
	s.mu.Unlock()
}

// finishRun records the run's end (peak memory, decode average); called once after a stop
// or a runtime exit of a server that had been ready.
func (l *Launcher) finishRun(s *session) {
	s.mu.Lock()
	pre, peak, lg, ready := s.pre, s.peak, s.log, s.ready
	s.mu.Unlock()
	if !ready || pre == nil {
		return
	}
	var sum TimingSummary
	if lg != nil {
		sum = lg.summary()
	}
	if err := l.recordRunEnd(pre.SettingsKey, pre.ModelPath, peak, sum, l.os.now()); err != nil {
		l.diag.write("HISTORY_WRITE_FAILED", map[string]any{"reason": err.Error()})
	}
	avg, _ := sum.RunDecodeAvg()
	l.diag.write("RUN_END", map[string]any{"peak_working_set_bytes": peak, "requests": sum.Requests, "decode_tok_per_sec": avg})
}
