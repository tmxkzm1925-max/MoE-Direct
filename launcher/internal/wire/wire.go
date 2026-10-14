// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

// Package wire is the adapter from internal/core to ui.Backend. It holds no launcher logic of
// its own beyond what the review round of 2026-10-10 gave it (the five-second decode window,
// the time-ordered log merge, the preflight reuse rule and the user wording of reasons): every
// value comes from core (or from the OS reads core exposes), and every function outside the
// slice answers ui.ErrNotPorted or leaves its value NotPorted.
package wire

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// statusTimeout bounds one Status call (three endpoint reads of 5 s each run in parallel).
const statusTimeout = 8 * time.Second

// vramEvery limits the nvidia-smi and GPU-counter reads on the serve page.
const vramEvery = 5 * time.Second

// LauncherLockLine is W7 duty 7's line (F-LS 9.2, 11.7): a start refused because another
// launcher holds the launcher locks (fail_instance_lock) shows it at the place of the start
// verdict, and the argument-driven mode prints it; nothing is started.
const LauncherLockLine = "another launcher is already running"

// Backend implements ui.Backend over a core.Launcher.
type Backend struct {
	l           *core.Launcher
	catalogPath string
	version     string
	now         func() time.Time

	mu      sync.Mutex
	files   map[string]core.ModelFile // row ID (the file path) → the file of the last list
	ml      *core.ModelList
	stamp   string // core.ScanStamp of the last list (item 31)
	run     *run
	events  []ui.LogLine
	exit    int
	exitSet bool
	win     decodeWindow
	vram    core.VRAMInfo
	vramAt  time.Time
	gpu     serverGPU
	gpuAt   time.Time
	pre     map[string]*preEntry      // model + settings → the last preflight (items 13, 20, 22)
	refused map[string]string         // model → the last refusal logged (row 20: no repeated warnings)
	noticed map[string]string         // model → the last full notices logged (item 32)
	modes   map[string]string         // file → the repack mode it starts with (item 31)
	gates   map[string][]ui.ModeCheck // file → both modes' format gates, read with the list (conflict 16)

	// FIX2-WIRE
	smoke        core.SmokeMode  // «작동 확인» for this launcher session (conflict 18-1, default off)
	cacheCap     core.CapSetting // GL-16: the prompt-cache cap of 5 Settings (U-22; default: auto)
	capMax       int             // GL-06: E.3's allowed maximum at the last check (MiB)
	capMaxOK     bool
	switches     core.Switches               // GL-16: the warm-start switches of 5 Settings (OQ-6 (a))
	bundleLogged bool                        // the launcher file check's result is in 4 Log (conflict 18-2)
	plans        map[string]*core.RepackPlan // model + settings → the plan the box showed (conflict 9)
	exec         core.RepackExecutor         // the repacker runner (core.ProcessExecutor; a fake in tests)
	// core's calls the setup, the start and the log go through (the tests replace them).
	planFn func(ctx context.Context, req core.Request, force bool) (*core.RepackPlan, error)
	// GL-10: the repacker's --plan run whose numbers the box shows (F-LS 4.6), and the
	// cut-off screen's «delete half-made only» (H4). nil = not called (the stub tests).
	planTextFn  func(ctx context.Context, p *core.RepackPlan, ex core.RepackExecutor) (string, error)
	partialRmFn func(ctx context.Context, modelPath, profileID string) ([]string, error)
	repackFn    func(ctx context.Context, p *core.RepackPlan, ex core.RepackExecutor, progress func(core.RepackEvent)) (core.RepackResult, error)
	startFn     func(ctx context.Context, req core.Request) (<-chan core.StartEvent, error)
	stopsFn     func() []core.ServerStopEvent
	bundleFn    func() core.BundleResult
	// GL-13: the first Bundle call starts the check (conflict 18-2: after the first draw)
	bundleStart func()
	// GL-08: the end of the run the shutdown reads (U-32 (ii): a quit keeps the last stop's code).
	doneFn func() <-chan struct{}
	stopFn func() (<-chan core.StopEvent, error)
	endFn  func() (core.RunEnd, bool)
}

// run is the one start this backend drives.
type run struct {
	model     string
	path      string
	cancel    context.CancelFunc
	done      chan struct{} // closed after the start's final event
	final     *core.StartEvent
	pre       *core.PreflightResult
	readyAt   time.Time
	own       ownRequests // the warmup and check-run requests before readyAt (FIX2-WIRE)
	stopping  bool
	stopped   bool
	lastModel ui.Val
}

// New makes the adapter. catalogPath is shown as the catalog source; version on 5 Settings.
func New(l *core.Launcher, catalogPath, version string) *Backend {
	b := &Backend{l: l, catalogPath: catalogPath, version: version, now: time.Now,
		files: map[string]core.ModelFile{}, pre: map[string]*preEntry{}, refused: map[string]string{},
		noticed: map[string]string{}, plans: map[string]*core.RepackPlan{}, exec: core.ProcessExecutor{},
		switches: core.DefaultSwitches}
	if l != nil {
		b.planFn, b.repackFn, b.startFn, b.stopsFn, b.bundleFn = l.PlanRepack, l.RunRepack, l.Start, l.StopEvents, l.BundleCheckResult
		b.doneFn, b.stopFn, b.endFn = l.Done, l.Stop, l.LastEnd
		b.planTextFn, b.partialRmFn = l.PlanText, l.RemovePartialSetup
		b.bundleStart = l.StartBundleCheck
	}
	return b
}

var _ ui.Backend = (*Backend)(nil)

// logChecks writes the post-ready checks of a start's final event to 4 Log: the warmup line,
// then the «작동 확인» check run's items, one line each, and its summary (GL-11, S8-LA-25).
func (b *Backend) logChecks(ev core.StartEvent) {
	if ev.Warmup != nil {
		b.logf(map[bool]ui.LogLevel{true: ui.LogInfo, false: ui.LogWarn}[ev.Warmup.OK || ev.Warmup.Skipped], "warmup: %s", ev.Warmup.Line())
	}
	for _, ln := range smokeItemLogs(ev.Smoke) {
		b.logf(ln.Level, "%s", ln.Text)
	}
	if lv, line := smokeLog(ev.Smoke); line != "" {
		b.logf(lv, "%s", line)
	}
}

// candidateLogLine is the launcher's prefetch candidate in words for 4 Log, once after ready
// beside the engine's values ("" without a candidate): 2 Server shows what the engine applied
// only, and the candidate with its reasons stays here (UX-KR, decision 4 of the user's Korean
// string table).
func candidateLogLine(cand *core.PrefetchEcho) string {
	ws := prefetchWords(cand, nil)
	if len(ws) == 0 {
		return ""
	}
	return "prefetch candidate (not shown on 2 Server): " + ui.PrefetchSentences(ws)
}

func (b *Backend) logf(level ui.LogLevel, format string, a ...any) {
	b.mu.Lock()
	b.events = append(b.events, ui.LogLine{At: b.now(), Level: level, Text: fmt.Sprintf(format, a...)})
	if len(b.events) > 500 {
		b.events = b.events[len(b.events)-500:]
	}
	b.mu.Unlock()
}

func (b *Backend) setExit(code int) {
	b.mu.Lock()
	b.exit, b.exitSet = code, true
	b.mu.Unlock()
}

// ExitCode is the PowerShell exit code of the last run (0 when nothing was started).
func (b *Backend) ExitCode() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.exit
}

// Serving reports a start or a server of this launcher that has not ended (the window-close
// exit code of row 5 rests on it).
func (b *Backend) Serving() bool { return b.serving() }

// serving reports a server of this launcher that is started and not yet stopped.
func (b *Backend) serving() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.run != nil && !b.run.stopped
}

// Catalog is the model list (items 11, 12, 31). The screen asks every five seconds on 1 Models:
// the folders are stat'ed (core.ScanStamp: names, sizes, times; no hashing), and the list is
// read again only when they changed; the RAM numbers are measured at every call (item 13).
func (b *Backend) Catalog() (ui.Catalog, error) {
	stamp := b.l.ScanStamp()
	b.mu.Lock()
	ml, same, modes := b.ml, b.ml != nil && stamp == b.stamp, b.modes
	b.mu.Unlock()
	if !same {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		var err error
		ml, err = b.l.ListModels(ctx)
		if err != nil {
			b.logf(ui.LogError, "catalog: %v", err)
			return ui.Catalog{}, err
		}
		b.mu.Lock()
		gone := 0
		for id := range b.files {
			found := false
			for _, f := range ml.Files {
				if f.Path == id {
					found = true
				}
			}
			if !found {
				gone++
			}
		}
		b.files = map[string]core.ModelFile{}
		for _, f := range ml.Files {
			b.files[f.Path] = f
		}
		hadList := b.ml != nil
		b.ml, b.stamp, b.modes = ml, stamp, nil
		b.pre = map[string]*preEntry{} // a file or a repack output changed: no preflight stands
		b.mu.Unlock()
		if hadList {
			b.logf(ui.LogInfo, "model folders changed: list read again (%d files, %d gone)", len(ml.Files), gone)
		} else {
			b.logf(ui.LogInfo, "model list read: %d files in %d folder(s)", len(ml.Files), len(ml.Roots))
		}
	}
	var gates map[string][]ui.ModeCheck
	if modes == nil || !same {
		modes, gates = b.startModes(ml)
		b.mu.Lock()
		b.modes, b.gates = modes, gates
		b.mu.Unlock()
	}
	b.mu.Lock()
	gates = b.gates
	b.mu.Unlock()
	mem, merr := b.l.MemNow()
	c := catalogFrom(ml, b.catalogPath, mem.AvailPhysMiB, merr == nil, b.vramNow(), modes)
	for i := range c.Models {
		c.Models[i].Modes = gates[c.Models[i].ID]
	}
	return c, nil
}

// startModes is the repack mode each catalog file starts with: its saved preset's, else the
// catalog default (item 31's setup state rests on it), and the format gates of both modes
// (conflict 16: small JSON reads and the manifest hash, never experts.bin). They are read again
// only when the list changes, a preset is saved or a setup ran.
func (b *Backend) startModes(ml *core.ModelList) (map[string]string, map[string][]ui.ModeCheck) {
	modes := map[string]string{}
	gates := map[string][]ui.ModeCheck{}
	for _, f := range ml.Files {
		if f.ProfileID == "" {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if mfs, err := b.l.FormatGates(ctx, f.Path, f.ProfileID); err == nil {
			gates[f.Path] = modeChecks(mfs)
		}
		cancel()
		mode := "packed"
		if rows, err := b.l.Settings(f.ProfileID); err == nil {
			for _, r := range rows {
				if r.Key == "repack_mode" && r.Default != "" {
					mode = r.Default
				}
			}
		}
		if p, ps := b.l.LoadPreset(context.Background(), f.Path, f.ProfileID); ps == core.PresetApplies {
			if v, ok := p.Values["repack_mode"]; ok {
				mode = v
			}
		}
		modes[f.Path] = mode
	}
	return modes, gates
}

// fileOf is the model file of a row ID (the path), from the last list or, failing that, a
// fresh list.
func (b *Backend) fileOf(model string) (core.ModelFile, error) {
	b.mu.Lock()
	f, ok := b.files[model]
	b.mu.Unlock()
	if ok {
		return f, nil
	}
	if _, err := b.Catalog(); err != nil {
		return core.ModelFile{}, err
	}
	b.mu.Lock()
	f, ok = b.files[model]
	b.mu.Unlock()
	if !ok {
		return core.ModelFile{}, fmt.Errorf("%s: the model file is no longer in the model folders", filepath.Base(model))
	}
	return f, nil
}

// ModelSettings is core.Settings with the last actual start (item 2) and the saved preset
// (item 3); a file outside the catalog with a template structure shows its structure's
// defaults (GL-14, conflict 14), another file outside the catalog has none.
func (b *Backend) ModelSettings(model string) (ui.SettingsView, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return ui.SettingsView{}, err
	}
	if f.ProfileID == "" && f.Template {
		rows, _ := core.StructureSettings(f.Arch)
		var running map[string]string
		if st, _, ok := b.l.LastStart(f.Path); ok {
			running = st.SettingsMap()
		}
		// GL-16: a model outside the catalog saves and reads its settings by file name and size.
		saved, note := b.presetOf(f.Path, "", "outside the catalog · "+f.Arch+" template settings") // UX-R12 addendum (finding 18)
		v := settingsFrom(model, rows, running, saved, note)
		for i := range v.Rows { // UX-R6: these ranges are the structure template's
			if v.Rows[i].Basis != "" {
				v.Rows[i].Basis = basisTemplate
			}
		}
		return v, nil
	}
	if f.ProfileID == "" {
		return ui.SettingsView{Model: model, Preset: "no catalog profile: settings not ported"}, nil
	}
	rows, err := b.l.Settings(f.ProfileID)
	if err != nil {
		return ui.SettingsView{}, err
	}
	if locked, err := b.l.LockedSettings(f.ProfileID); err == nil {
		for _, r := range locked {
			if r.Key != "host" { // conflict 17: no network option at all, not even a locked row
				rows = append(rows, r)
			}
		}
	}
	var running map[string]string
	if st, _, ok := b.l.LastStart(f.Path); ok {
		running = st.SettingsMap()
	}
	saved, note := b.presetOf(f.Path, f.ProfileID, "")
	return settingsFrom(model, rows, running, saved, note), nil
}

// savedSettingsNotUsed is core.NoticePresetNotUsed (F-LS 11.4, saved-settings path) as the
// screens look it up: a backend word of the Korean table (S8-LA-36); TestGL16LauncherSettings
// checks the two agree.
const savedSettingsNotUsed = "the saved settings of this model hold a value that failed its check · not used · the defaults apply"

// presetOf is a model's saved settings and the note the settings show (dflt with none).
func (b *Backend) presetOf(path, profileID, dflt string) (map[string]string, string) {
	pre, ps := b.l.LoadPreset(context.Background(), path, profileID)
	switch ps {
	case core.PresetApplies:
		return pre.Values, "saved settings (" + pre.SavedAt + ")"
	case core.PresetOtherFile:
		return nil, "defaults · the saved settings belong to another file"
	case core.PresetUnreadable, core.PresetInvalid: // F-LS 11.4, saved-settings path
		return nil, savedSettingsNotUsed
	}
	return nil, dflt
}

// SavePreset is ctrl+s (item 3): the changed values saved under <state>\presets (GL-16;
// a model outside the catalog with a template structure too, keyed by file name and size).
func (b *Backend) SavePreset(model string, s ui.Settings) (ui.SettingsView, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return ui.SettingsView{}, err
	}
	if f.ProfileID == "" && !f.Template {
		return ui.SettingsView{}, fmt.Errorf("no catalog profile and no template structure: no settings to save")
	}
	p, err := b.l.SavePreset(context.Background(), f.Path, f.ProfileID, presetValues(s))
	if err != nil {
		return ui.SettingsView{}, err
	}
	b.logf(ui.LogOK, "saved settings of %s: %v → %s", f.DisplayName, p.Values, b.l.PresetPath(f.ProfileID, f.Path))
	b.dropPreflights(model)
	b.mu.Lock()
	b.modes = nil // the saved repack mode decides the setup state (item 31)
	b.mu.Unlock()
	return b.ModelSettings(model)
}

// CheckSetting is item 30's typed value: parsed and range-checked by core with the PowerShell
// bounds, given back in the screen's form (the budget in G).
func (b *Backend) CheckSetting(model, key, value string) (string, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return "", err
	}
	if f.ProfileID == "" && !f.Template {
		return "", fmt.Errorf("no catalog profile")
	}
	v, note, err := b.l.ParseModelSetting(f.ProfileID, f.Path, key, value)
	if err != nil {
		return "", err
	}
	if note != "" {
		b.logf(ui.LogNotice, "%s %s: %s", settingLabels[key], v, note)
	}
	return toScreen(key, v), nil
}

// preKey names one preflight: the model and its settings.
func preKey(model string, s ui.Settings) string {
	keys := make([]string, 0, len(s))
	for k := range s {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	sb.WriteString(model)
	for _, k := range keys {
		sb.WriteString("|" + k + "=" + s[k])
	}
	return sb.String()
}

func (b *Backend) dropPreflights(model string) {
	b.mu.Lock()
	for k := range b.pre {
		if strings.HasPrefix(k, model+"|") || k == model {
			delete(b.pre, k)
		}
	}
	b.mu.Unlock()
}

// Preflight is core.Preflight with item 13's live RAM: a result for the same model and settings
// is reused for a few seconds with the RAM re-read now, and re-run when it is stale (taken
// while a server was up that has stopped since, row 20). A refusal comes back as a refused
// result in user words (item 32); its full text goes to 4 Log once per change.
func (b *Backend) Preflight(model string, s ui.Settings) (ui.Preflight, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return ui.Preflight{}, err
	}
	key, now, serving := preKey(model, s), b.now(), b.serving()
	b.mu.Lock()
	e := b.pre[key]
	b.mu.Unlock()
	var res *core.PreflightResult
	var perr error
	if e.stale(now, serving) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		res, perr = b.l.Preflight(ctx, b.request(f.Path, s))
		cancel()
		b.mu.Lock()
		b.pre[key] = &preEntry{res: res, err: perr, at: now, serving: serving}
		b.mu.Unlock()
	} else {
		res, perr = e.res, e.err
		if pc, err := b.l.PCMemoryNow(); err == nil {
			res = freshen(res, pc.AvailMiB, pc.ServerMiB)
		}
	}
	p, full := preflightFrom(model, res, perr)
	b.mu.Lock()
	if res != nil && res.CapMaxOK { // GL-06: shown beside the cap on 5 Settings (F-BU E.3)
		b.capMax, b.capMaxOK = res.CapMaxMiB, true
	}
	last := b.refused[model]
	b.refused[model] = full
	b.mu.Unlock()
	if full != "" && full != last && !serving {
		b.logf(ui.LogWarn, "preflight %s refused: %s", f.DisplayName, full)
	}
	if n := noticeLog(res); n != "" {
		b.mu.Lock()
		lastN := b.noticed[model]
		b.noticed[model] = n
		b.mu.Unlock()
		if n != lastN {
			b.logf(ui.LogInfo, "preflight %s notices: %s", f.DisplayName, n)
		}
	}
	return p, nil
}

// Start is core.Start; the core events are folded into the H15 step list.
func (b *Backend) Start(model string, s ui.Settings) (<-chan ui.StartEvent, error) {
	f, err := b.fileOf(model)
	if err != nil {
		return nil, err
	}
	path := f.Path
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := b.startFn(ctx, b.request(path, s))
	if err != nil {
		cancel()
		return nil, err
	}
	r := b.newRun(model, path, cancel)
	// The out channel never blocks the reader of core's channel: it holds every event of a
	// start (ticks every 500 ms for at most 900 s), and the reader drops one only if it is full.
	out := make(chan ui.StartEvent, 4096)
	go func() {
		defer close(out)
		defer close(r.done)
		defer cancel()
		b.pump(r, f, ch, out)
	}()
	return out, nil
}

// newRun makes the run of a start or a setup and makes it the backend's current one.
func (b *Backend) newRun(model, path string, cancel context.CancelFunc) *run {
	r := &run{model: model, path: path, cancel: cancel, done: make(chan struct{})}
	b.mu.Lock()
	b.run = r
	b.win = decodeWindow{}
	b.gpuAt = time.Time{}
	b.mu.Unlock()
	b.logf(ui.LogInfo, "start %s: %s", filepath.Base(path), path)
	return r
}

// pump folds core's start events into the screen's (H15, H7) and records the end.
func (b *Backend) pump(r *run, f core.ModelFile, ch <-chan core.StartEvent, out chan<- ui.StartEvent) {
	model := r.model
	{
		t := newStartTrack(model)
		for ev := range ch {
			now := b.now()
			if ev.Preflight != nil {
				b.mu.Lock()
				first := r.pre == nil
				r.pre = ev.Preflight
				b.mu.Unlock()
				if first && ev.Preflight.Sweep.Line != "" { // GL-10: the speed check's lines (Resolve-QdSweep)
					for _, l := range strings.Split(ev.Preflight.Sweep.Line, "\n") {
						b.logf(map[bool]ui.LogLevel{true: ui.LogInfo, false: ui.LogWarn}[ev.Preflight.Sweep.OK], "%s", l)
					}
				}
			}
			u := t.apply(ev, now)
			u.Notices = userNotices(u.Notices)
			if ev.Final {
				e := ev
				b.mu.Lock()
				r.final = &e
				if ev.Ready {
					r.readyAt = now
				}
				b.mu.Unlock()
				b.logChecks(ev)
				if ev.Ready {
					b.logf(ui.LogOK, "ready: %s on %s after %.1f s (pid %d)", f.DisplayName, ev.Address, ev.Elapsed.Seconds(), ev.PID)
					if ev.Engine != nil { // GL-21 (F-PF §8.5): the machine form in 4 Log
						var cand *core.PrefetchEcho
						if ev.Preflight != nil {
							cand = ev.Preflight.Start.PrefetchEcho
						}
						b.logf(map[bool]ui.LogLevel{true: ui.LogInfo, false: ui.LogWarn}[ev.Engine.Read], "%s", core.PostReadyLine(*ev.Engine, cand))
						if l := core.NCapLine(cand); l != "" { // GL-20: the list per layer (F-PF §8.2)
							b.logf(ui.LogInfo, "%s", l)
						}
						if l := candidateLogLine(cand); l != "" { // UX-KR (decision 4)
							b.logf(ui.LogInfo, "%s", l)
						}
					}
				} else {
					b.setExit(ev.ExitCode)
					tail, _ := b.l.LogTail(12)
					u.Failure = failureFrom(ev, now, tail)
					b.logf(ui.LogError, "start %s ended: %s (exit %d): %s", f.DisplayName, ev.Status, ev.ExitCode, refusalText(ev.Err))
				}
			}
			select {
			case out <- u:
			default:
				if ev.Final { // never lose the last event
					out <- u
				}
			}
		}
	}
}

// batchOf is the -b value of the argv the server was started with (UX-R26), or not known.
func batchOf(argv []string) ui.Val {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == "-b" {
			if n, err := strconv.Atoi(argv[i+1]); err == nil && n > 0 {
				return ui.K(float64(n))
			}
		}
	}
	return ui.U
}

// Status is core.Status with the OS memory, the GPU reads and the five-second decode window.
func (b *Backend) Status() (ui.ServeStatus, error) {
	ctx, cancel := context.WithTimeout(context.Background(), statusTimeout)
	defer cancel()
	st, err := b.l.Status(ctx)
	if err != nil {
		return ui.ServeStatus{}, err
	}
	mem, merr := b.l.MemNow()
	vr, gpu := b.vramNow(), b.gpuNow()
	lines, _ := b.l.ServerLog(0)
	inflight := requestInFlight(lines)
	b.mu.Lock()
	if b.run != nil {
		st.Timing = b.run.own.drop(st.Timing, b.run.readyAt)
	}
	s := statusFrom(st, mem, merr == nil, vr, gpu, &b.win, b.now(), inflight)
	if b.run != nil && b.run.pre != nil && b.run.pre.Config != nil {
		s.Batch = batchOf(b.run.pre.Config.Argv) // UX-R26: the processing tile's batch size
	}
	if b.run != nil {
		if s.ModelRAMGB.Ok() {
			b.run.lastModel = s.ModelRAMGB
		}
		if f, ok := b.files[b.run.model]; ok {
			s.Model = f.DisplayName
		}
		if b.run.stopping || b.run.stopped {
			s.Exited = false // our own stop, not a server that ended while serving
		}
	}
	b.mu.Unlock()
	if s.Exited {
		b.setExit(s.ExitCode)
	}
	return s, nil
}

func (b *Backend) vramNow() core.VRAMInfo {
	b.mu.Lock()
	if !b.vramAt.IsZero() && b.now().Sub(b.vramAt) < vramEvery {
		v := b.vram
		b.mu.Unlock()
		return v
	}
	b.mu.Unlock()
	v := b.l.VRAMNow()
	b.mu.Lock()
	b.vram, b.vramAt = v, b.now()
	b.mu.Unlock()
	return v
}

// gpuNow is this server's dedicated GPU memory (the Windows GPU process counter, item 4).
func (b *Backend) gpuNow() serverGPU {
	b.mu.Lock()
	if !b.gpuAt.IsZero() && b.now().Sub(b.gpuAt) < vramEvery {
		g := b.gpu
		b.mu.Unlock()
		return g
	}
	b.mu.Unlock()
	g := serverGPU{}
	if v, err := b.l.ServerVRAMNow(); err == nil {
		g = serverGPU{bytes: v, ok: true}
	}
	b.mu.Lock()
	b.gpu, b.gpuAt = g, b.now()
	b.mu.Unlock()
	return g
}

// Memory is H10: the whole PC's RAM now with this server's share apart, and the VRAM.
func (b *Backend) Memory() (ui.MemoryReading, error) {
	pc, merr := b.l.PCMemoryNow()
	var budget int64
	var pred core.CommitPrediction
	var predOK bool
	b.mu.Lock()
	if b.run != nil && b.run.pre != nil && b.run.pre.Config != nil && !b.run.stopped {
		budget = b.run.pre.Config.Budget.MB
		pred, predOK = b.run.pre.Prediction, b.run.pre.PredictionOK
	}
	b.mu.Unlock()
	r := memoryFrom(pc, merr == nil, budget, b.vramNow(), b.gpuNow())
	splitInto(&r, pred, predOK, budget > 0) // GL-06: U-20 (i)
	return r, nil
}

// LogTail is items 5 and 18: the launcher's events and the server's parsed lines in one
// time-ordered list, with the server's own levels and the notice class of item 29.
func (b *Backend) LogTail(max int) ([]ui.LogLine, error) {
	b.mu.Lock()
	events := append([]ui.LogLine(nil), b.events...)
	b.mu.Unlock()
	if b.stopsFn != nil {
		events = append(events, stopLines(b.stopsFn())...) // conflict 6: one line per server stop
	}
	server, _ := b.l.ServerLog(0)
	return mergeLogs(events, server, max), nil
}

// Launcher is 5 Settings: the fixed address rule, the loopback lock and where the draft writes
// (item 24: setup = the repack output of the model last started, logs, the log file, settings).
func (b *Backend) Launcher() (ui.LauncherInfo, error) {
	info := ui.LauncherInfo{Address: "127.0.0.1 · port from the catalog (8093)",
		OnOpen: "not ported", SettingsDir: b.l.PresetDir(),
		LogDir: b.l.LogDir(), LogFiles: b.l.LogFiles(), Version: b.version, Smoke: smokeUI(b.smokeMode())}
	b.mu.Lock()
	held := core.LauncherSettingsMap(b.cacheCap, b.switches)
	info.CacheCap, info.WarmStart, info.Autosave = held["prompt_cache_cap"], held["warmstart"], held["autosave"]
	info.CacheCapMax = ui.U
	if b.capMaxOK {
		info.CacheCapMax = ui.K(float64(b.capMax))
	}
	if b.run != nil && b.run.pre != nil && !b.run.stopped { // rows 6.1-6.2: the running server's values
		ran := b.run.pre.Start.SettingsMap()
		info.NextStart = map[string]bool{}
		for k, v := range held {
			if rv, ok := ran[k]; ok && rv != v {
				info.NextStart[k] = true
			}
		}
	}
	if b.run != nil {
		mode := "repack"
		if b.run.pre != nil && b.run.pre.RepackMode == core.RepackVirtual {
			mode = "repack-virtual"
		}
		info.SetupDir = filepath.Join(filepath.Dir(b.run.path), mode)
		if b.run.pre != nil && b.run.pre.OutputDir != "" {
			info.SetupDir = b.run.pre.OutputDir
		}
	}
	b.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), statusTimeout)
	defer cancel()
	if st, err := b.l.Status(ctx); err == nil {
		info.LogFile = st.ErrLog
		if st.Address != "" {
			info.Address = st.Address
		}
	}
	return info, nil
}

// Stop runs the PowerShell stop procedure, or cancels a start that is not ready yet.
func (b *Backend) Stop(mode ui.StopMode) (<-chan ui.StopEvent, error) {
	b.mu.Lock()
	r := b.run
	b.mu.Unlock()
	if r == nil {
		return nil, core.ErrNoServer
	}
	b.mu.Lock()
	ready := r.final != nil && r.final.Ready
	starting := r.final == nil
	b.mu.Unlock()
	if starting || mode == ui.StopCancel && !ready {
		return b.cancelStart(r, mode), nil
	}
	stopFn := b.l.Stop
	if mode == ui.StopForce { // ctrl+c twice: close now, no cache save (GL-15)
		stopFn = b.l.StopNoSave
	}
	ch, err := stopFn()
	if err != nil {
		return nil, err
	}
	b.mu.Lock()
	r.stopping = true
	model := r.lastModel
	name := r.model
	if f, ok := b.files[r.model]; ok {
		name = f.DisplayName
	}
	b.mu.Unlock()
	b.logf(ui.LogInfo, "stop %s (mode %d)", name, mode)
	out := make(chan ui.StopEvent, 64)
	began := b.now()
	go func() {
		defer close(out)
		t := newStopTrack(mode)
		out <- t.event(0, model)
		for ev := range ch {
			if ev.Save != nil {
				t.saveFrom(ev.Save)
			}
			if ev.SaveErr != "" || ev.NoSave != "" { // GL-15
				t.saveEnd(ev.SaveErr, ev.NoSave)
			}
			if !ev.Final {
				t.apply(ev.Step)
				out <- t.event(b.now().Sub(began), model)
				continue
			}
			t.finish(ev.Teardown)
			b.setExit(ev.ExitCode)
			b.mu.Lock()
			r.stopped, r.stopping = true, false
			b.mu.Unlock()
			graceful := ev.Status == core.StatusOK
			b.logf(map[bool]ui.LogLevel{true: ui.LogOK, false: ui.LogError}[graceful],
				"stopped %s: %s (exit %d) %s", name, ev.Status, ev.ExitCode, ev.Reason)
			tm, _ := b.l.Timings()
			b.mu.Lock()
			tm = r.own.drop(tm, r.readyAt)
			b.mu.Unlock()
			e := t.event(b.now().Sub(began), model)
			e.Done = true
			e.Result = &ui.StopResult{ExitCode: ev.ExitCode, Graceful: graceful,
				Message: fmt.Sprintf("%s · exit %d · %s", ev.Status, ev.ExitCode, ev.Reason),
				Run:     runSummary(name, r.readyAt, b.now(), tm)}
			out <- e
		}
	}()
	return out, nil
}

// cancelStart is ctrl+x (or a quit) while starting: the start's context is cancelled and the
// start ends cancelled_user (exit 2) after its own teardown.
func (b *Backend) cancelStart(r *run, mode ui.StopMode) <-chan ui.StopEvent {
	r.cancel()
	out := make(chan ui.StopEvent, 4)
	began := b.now()
	go func() {
		defer close(out)
		t := newStopTrack(mode)
		t.saveEnd("", "no cache save while the server starts")
		t.steps[1] = ui.Step{Name: "cancel the start", State: ui.StepRunning}
		out <- t.event(0, ui.Val{})
		<-r.done
		b.mu.Lock()
		f := r.final
		r.stopped = true
		b.mu.Unlock()
		res := &ui.StopResult{ExitCode: core.ExitCode(core.StatusCancelledUser), Message: "start cancelled"}
		if f != nil {
			res.ExitCode = f.ExitCode
			res.Graceful = f.Teardown == nil || f.Teardown.Graceful || f.Teardown.PreExited
			res.Message = fmt.Sprintf("%s · exit %d", f.Status, f.ExitCode)
			t.finish(f.Teardown)
		}
		b.setExit(res.ExitCode)
		t.steps[1].State = ui.StepDone
		e := t.event(b.now().Sub(began), ui.Val{})
		e.Done, e.Result = true, res
		out <- e
	}()
	return out
}

// Action runs the controls core provides (items 15 and 24: copy the address, open a folder,
// show a file in Explorer); every other control answers ui.ErrNotPorted.
func (b *Backend) Action(id ui.ActionID, arg string) (ui.ActionResult, error) {
	switch id {
	case ui.ActCopyAddress:
		addr := strings.TrimSpace(arg)
		if addr == "" && b.l != nil {
			ctx, cancel := context.WithTimeout(context.Background(), statusTimeout)
			if st, err := b.l.Status(ctx); err == nil && st.Address != "" {
				addr = st.Address + "/v1"
			}
			cancel()
		}
		if addr == "" {
			return ui.ActionResult{}, errors.New("no server address to copy")
		}
		if err := b.l.CopyText(addr); err != nil {
			return ui.ActionResult{}, err
		}
		return ui.ActionResult{Message: "copied " + addr}, nil
	case ui.ActOpenFolder:
		if err := b.l.OpenFolder(arg); err != nil {
			return ui.ActionResult{}, err
		}
		return ui.ActionResult{Message: "opened in Explorer"}, nil
	case ui.ActShowFile:
		if err := b.l.RevealFile(arg); err != nil {
			return ui.ActionResult{}, err
		}
		return ui.ActionResult{Message: "shown in Explorer"}, nil
	case ui.ActSetSmoke:
		return b.setSmoke(arg)
	case ui.ActTaskManager: // GL-09: no selection (U-35 (i)); the screen names the app
		if err := b.l.OpenTaskManager(); err != nil {
			return ui.ActionResult{}, err
		}
		return ui.ActionResult{Message: "Task Manager opened"}, nil
	case ui.ActSetCacheCap, ui.ActSetWarmStart, ui.ActSetAutosave:
		return b.setLauncher(id, arg)
	case ui.ActDeletePartial:
		return b.deletePartial(arg)
	}
	return ui.ActionResult{}, fmt.Errorf("%s: %w", id, ui.ErrNotPorted)
}

// Shutdown stops a server still running when the screens have closed (the q-twice and
// ctrl+c-twice paths end the program before or while the stop runs) and waits for the end.
// It returns the exit code to exit with: the code of the last stop (U-32 (ii), L:4264), also
// when the server ended on its own and the screen never polled that end (W7 duty 34).
func (b *Backend) Shutdown(timeout time.Duration) int {
	b.mu.Lock()
	r := b.run
	b.mu.Unlock()
	if r == nil {
		return b.ExitCode()
	}
	deadline := time.After(timeout)
	select {
	case <-r.done:
	case <-deadline:
		r.cancel()
		<-r.done
	}
	if b.doneFn == nil {
		return b.ExitCode()
	}
	if done := b.doneFn(); done != nil {
		select {
		case <-done:
			if b.endFn != nil {
				if end, ok := b.endFn(); ok {
					b.setExit(end.ExitCode)
				}
			}
			return b.ExitCode()
		default:
		}
		if ch, err := b.stopFn(); err == nil {
			for ev := range ch {
				if ev.Final {
					b.setExit(ev.ExitCode)
				}
			}
		} else if !errors.Is(err, core.ErrNoServer) {
			select {
			case <-done:
				if b.endFn != nil {
					if end, ok := b.endFn(); ok {
						b.setExit(end.ExitCode)
					}
				}
			case <-deadline:
			}
		}
	}
	return b.ExitCode()
}
