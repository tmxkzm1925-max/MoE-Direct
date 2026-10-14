package core

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// derivedVirtualRefusal is PS:13923: a derived (template) profile takes no virtual repack.
const derivedVirtualRefusal = "derived/template profiles do not support -RepackMode virtual in this preview"

// Repack modes (REPACK_MODE_* PS:742-756).
const (
	RepackPacked  = "packed"
	RepackVirtual = "virtual"
	modelsDirName = "moe-models" // MODELS_DIR_NAME (PS:10992)
)

// Request is what a screen asks to preflight or start.
type Request struct {
	ModelPath  string            // -Model: any shard of a split set, or the one file
	RepackMode string            // -RepackMode: "" = packed (REPACK_MODE_DEFAULT PS:745)
	ProfileID  string            // the answer to Select-Profile's question when several profiles match; usually ""
	OutDir     string            // -OutDir: "" = <model dir>\repack or \repack-virtual
	Settings   map[string]string // the allowlist keys (SettingKeys)
	// NoPreset leaves the model's saved preset out (review item 3: by default the preset's
	// values fill the keys the request does not name, and its repack mode applies when
	// RepackMode is "").
	NoPreset bool
	// Smoke is the «작동 확인» setting of conflict 18-1 (a launcher setting, not a server one):
	// the start runs the checklist after ready when SmokeDue says so.
	Smoke SmokeMode
	// GL-16: the launcher settings of 5 Settings, the same for every model: the prompt-cache
	// cap (U-22; its argv and its prediction term are GL-06's) and the warm-start switches
	// (OQ-6 (a); their function is warm start's, GL-15). The start records them (6.1-6.2).
	PromptCacheCap CapSetting
	Switches       *Switches // nil = DefaultSwitches
	// FromArguments marks a request of the argument-driven mode: its refusals name that
	// surface (F-LS 5.3, 11.4; S8-LA-29).
	FromArguments bool
}

// customPath is the surface of a request's values.
func (r Request) customPath() CustomValuePath {
	if r.FromArguments {
		return PathArguments
	}
	return PathSettingsTyped
}

// switches are the request's warm-start switches.
func (r Request) switches() Switches {
	if r.Switches == nil {
		return DefaultSwitches
	}
	return *r.Switches
}

// Step names of a preflight and a start, in their order (the main flow PS:13840-14327).
const (
	StepCatalog   = "catalog"
	StepIdentify  = "identify"
	StepArtifacts = "artifacts"
	StepLocks     = "locks"
	StepSweep     = "sweep"
	StepConfig    = "config"
	StepSizing    = "sizing"
	StepSpawn     = "spawn"
	StepReady     = "ready"
)

// StartSteps lists every step of a start up to ready (the adapter's step list).
var StartSteps = []string{StepCatalog, StepIdentify, StepArtifacts, StepLocks, StepSweep, StepConfig, StepSizing, StepSpawn, StepReady}

// FIX2-CORE steps: the wait for the bundle check before the catalog step (conflict 18-2), and
// the launcher warmup (conflict 2) and the «작동 확인» check (18-1) after ready.
const (
	StepBundle = "bundle"
	StepWarmup = "warmup"
	StepSmoke  = "smoke"
)

// AllStartSteps is StartSteps with the FIX2-CORE steps in their places.
var AllStartSteps = []string{StepBundle, StepCatalog, StepIdentify, StepArtifacts, StepLocks, StepSweep, StepConfig, StepSizing, StepSpawn, StepReady, StepWarmup, StepSmoke}

// RAMVerdict is Test-RamVerdict (PS:3587-3618) as it runs today: the fixed terms are
// [unmeasured] (PS:1170-1175), so the verdict is "insufficient" (budget above available RAM,
// a hard stop) or "unmeasured" (conflict 7).
type RAMVerdict struct {
	Verdict  string // insufficient | unmeasured
	Basis    string
	Reason   string
	AvailMiB int64
	TotalMiB int64
	BudgetMB int64
	Ctx      int64
}

// MemoryVerdict is the H1/H14 verdict of decision ②.
type MemoryVerdict uint8

const (
	MemUnmeasured MemoryVerdict = iota // no measurement for this model and settings («미측정»)
	MemFits                            // need <= free RAM now
	MemShort                           // need > free RAM now (shown, not a refusal: PowerShell refuses only on the budget)
)

func (v MemoryVerdict) String() string {
	switch v {
	case MemFits:
		return "fits"
	case MemShort:
		return "short"
	}
	return "not measured" // UX-R12b (6): English, as its neighbours; the screens word it by the table
}

// VRAMInfo is Get-VramInfo (PS:3535-3556): display only, never a gate (conflict 8).
type VRAMInfo struct {
	OK       bool
	Source   string
	TotalMiB int64
	FreeMiB  int64 // -1 when the source gives no free value
	UsedMiB  int64 // -1 when the source gives no used value (review item 4: the whole GPU's use)
}

// osVRAM asks nvidia-smi as PowerShell does; the Win32_VideoController fallback is not
// ported (it only fills a display line).
func osVRAM() VRAMInfo {
	exe, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return VRAMInfo{}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, exe, "--query-gpu=memory.total,memory.free,memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return VRAMInfo{}
	}
	line := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0]
	return parseNvidiaSmiMemory(line)
}

// parseNvidiaSmiMemory reads "total, free[, used]" (MiB, no units).
func parseNvidiaSmiMemory(line string) VRAMInfo {
	parts := strings.Split(line, ",")
	if len(parts) < 2 {
		return VRAMInfo{}
	}
	t, e1 := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
	f, e2 := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
	if e1 != nil || e2 != nil {
		return VRAMInfo{}
	}
	v := VRAMInfo{OK: true, Source: "nvidia-smi", TotalMiB: t, FreeMiB: f, UsedMiB: -1}
	if len(parts) >= 3 {
		if u, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64); err == nil {
			v.UsedMiB = u
		}
	}
	return v
}

// PreflightResult is everything a preflight found. Fields after a failing step stay zero.
type PreflightResult struct {
	ModelPath    string
	RepackMode   string
	Arch         string
	Quant        string
	Shards       int
	TotalBytes   int64
	Selection    string // pinned | unpinned | mismatch
	Profile      *Profile
	Shas         []string
	SourceTag    string
	OutputDir    string
	ManifestSha  string // sha256 of <out>\manifest.json (the value the gate returns when it passes)
	WebUI        bool   // the catalog's runtime.webui (smoke item 7; GL-11)
	CheckedAt    string // verify_report checked_at (packed)
	Overrides    map[string]string
	Sweep        Sweep
	SweepArm     string // GL-10: the arm the sweep selected with (none, catalog-fixed, init, manual)
	QDSource     string
	Config       *Config
	RAM          RAMVerdict
	DiskFreeMiB  int64
	DiskRoot     string
	VRAM         VRAMInfo
	SettingsKey  string
	SettingsLine string
	// GL-15: this start's warm start (init, bindings, eligibility; the restore, save and
	// autosave use it); nil on a preflight, which never touches the kv tree.
	Warm *WarmStart

	// Decision ② / ⑧ (H1): need, free RAM now, left after the start.
	NeedKnown bool
	NeedMiB   int64
	FreeMiB   int64 // available RAM measured by this call (review item 13: never a cached value)
	LeftMiB   int64 // FreeForStartMiB - NeedMiB
	Verdict   MemoryVerdict
	// Review items 13 and 22.
	NeedBasis       NeedBasis
	ServerMiB       int64 // this launcher's serving server's working set (0 when none serves)
	FreeForStartMiB int64 // FreeMiB + ServerMiB: a start comes after that server's stop (item 1)
	Check           StartCheck
	// Review item 3: the model's preset and whether it was applied.
	Preset      *Preset
	PresetState PresetState
	// Review items 2 and 27: what a start with this preflight would use.
	Start StartSettings
	// Decision ③: the last start-to-ready time for this key.
	LastReady    time.Duration
	HasLastReady bool
	// FIX2-CORE: the format gate of this start (conflict 16, run every time); the budget cap of a
	// RAM refusal (conflict 7); the speed fact of these settings (conflict 10); the warmup value
	// the start uses (conflict 2).
	Gate           *GateResult
	BudgetCap      BudgetCapInfo
	SpeedMeasured  bool
	SpeedTokPerSec float64
	Warmup         string

	Notices []string
	Refusal error // the error the preflight ended with, also returned
	Steps   []string
	// GL-05: the projection of this start (F-BU B.5, B.6) and the label of the shown need.
	Fit        FitResult
	Projection NeedProjection
	NeedLabel  NeedLabel
	// GL-06: the commit prediction (F-BU B-F): the reading, the prediction after one request,
	// the room after start, the near-limit line and the cap's allowed maximum (E.3)
	Commit       CommitReading
	CommitOK     bool
	Prediction   CommitPrediction
	PredictionOK bool
	RoomAfterMiB float64
	Near         NearLimit
	NearOK       bool
	CapMaxMiB    int
	CapMaxOK     bool
	// GL-09: the memory lists of U-21 (the modified memory among them; PDH, LFS:302)
	Lists   MemoryLists
	ListsOK bool
}

type heldLocks struct {
	prefix  string // "" in use; the tests set one so they never touch a real launcher's names
	mutexes []*heldMutex
	file    *heldFile
	port    *heldMutex
}

func (h *heldLocks) release() {
	if h == nil {
		return
	}
	if h.file != nil {
		h.file.release()
	}
	for _, m := range h.mutexes {
		m.release()
	}
	if h.port != nil {
		h.port.release()
	}
	h.mutexes, h.file, h.port = nil, nil, nil
}

// safeToken is Get-SafeToken (PS:5009-5013).
func safeToken(s string) string { return sha256Hex([]byte(s))[:16] }

// acquireLauncherLocks is Acquire-LauncherLocks (PS:5036-5049) without the directory
// creation (the output directory exists: the artifacts were found in it).
func acquireLauncherLocks(prefix, profileID, outputDir string) (*heldLocks, error) {
	h := &heldLocks{prefix: prefix}
	m, err := acquireNamedMutex(prefix + "moe_direct_launcher")
	if err != nil {
		return nil, stop(StatusFailInstanceLock, "single-instance: %v", err)
	}
	h.mutexes = append(h.mutexes, m)
	m, err = acquireNamedMutex(prefix + "moe_direct_launcher_profile_" + safeToken(profileID))
	if err != nil {
		h.release()
		return nil, stop(StatusFailInstanceLock, "profile lock: %v", err)
	}
	h.mutexes = append(h.mutexes, m)
	f, err := acquireFileLock(filepath.Join(outputDir, ".moe-launcher.lock"))
	if err != nil {
		h.release()
		return nil, stop(StatusFailInstanceLock, "output lock: %v", err)
	}
	h.file = f
	return h, nil
}

// probeLocksForStop probes the launcher locks for the argument-driven stop and keeps none.
// Without an output folder yet no launcher can hold its output lock, and the stop creates no
// folder, so then only the two named mutexes are probed.
func probeLocksForStop(prefix, profileID, outputDir string) error {
	if st, err := os.Stat(outputDir); err == nil && st.IsDir() {
		h, err := acquireLauncherLocks(prefix, profileID, outputDir)
		if err != nil {
			return err
		}
		h.release()
		return nil
	}
	h := &heldLocks{prefix: prefix}
	defer h.release()
	for _, name := range []string{"moe_direct_launcher", "moe_direct_launcher_profile_" + safeToken(profileID)} {
		m, err := acquireNamedMutex(prefix + name)
		if err != nil {
			return stop(StatusFailInstanceLock, "single-instance: %v", err)
		}
		h.mutexes = append(h.mutexes, m)
	}
	return nil
}

// setPortLock is Set-EffectivePortLock (PS:5051-5064).
func (h *heldLocks) setPortLock(port int) error {
	m, err := acquireNamedMutex(h.prefix + "moe_direct_launcher_port_" + strconv.Itoa(port))
	if err != nil {
		return stop(StatusFailInstanceLock, "effective port lock: %v", err)
	}
	h.port = m
	return nil
}

// resolveOutputDir is Resolve-OutputDirectory (PS:13642-13658).
func resolveOutputDir(req Request, virtual bool) (string, error) {
	out := req.OutDir
	if out == "" {
		leaf := "repack"
		if virtual {
			leaf = "repack-virtual"
		}
		out = filepath.Join(filepath.Dir(req.ModelPath), leaf)
	}
	abs, err := filepath.Abs(out)
	if err != nil {
		return "", stop(StatusFailModelPath, "output directory path is not valid: %s", out)
	}
	if len(abs) > 3 {
		abs = strings.TrimRight(abs, `\`)
	}
	return abs, nil
}

func fileExists(p string) bool { st, err := os.Stat(p); return err == nil && !st.IsDir() }

// lastNonEmptyRecord is Get-LastNonEmptyRecordText (PS:5241-5255).
func lastNonEmptyRecord(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	last := ""
	for _, ln := range strings.Split(string(b), "\n") {
		t := strings.TrimRight(ln, "\r\n \t")
		if t != "" {
			last = t
		}
	}
	if last == "" {
		return "", fmt.Errorf("verify_report.json has no non-empty record")
	}
	return last, nil
}

type prepareMode int

const (
	modePreflight prepareMode = iota // probe the locks, keep nothing
	modeStart                        // hold the locks for the server's life
	// modeStopBeforeStart is the argument-driven stop (GL-08): probe the locks, then stop as
	// cancelled_user before anything is checked or started further.
	modeStopBeforeStart
)

type prepared struct {
	res   *PreflightResult
	cat   *Catalog
	locks *heldLocks
	stamp time.Time
}

// prepare runs the slice of the main flow up to the effective sizing (PS:13840-14269):
// catalog, identify, artifacts, locks, sweep, config and sizing. step is called as each step
// starts. On a refusal the result so far comes back with the error.
func (l *Launcher) prepare(ctx context.Context, req Request, mode prepareMode, step func(string)) (*prepared, error) {
	res := &PreflightResult{RepackMode: RepackPacked}
	pr := &prepared{res: res, stamp: l.os.now()}
	fail := func(err error) (*prepared, error) {
		res.Refusal = err
		if pr.locks != nil && mode == modeStart {
			pr.locks.release()
			pr.locks = nil
		}
		l.diag.write("PREFLIGHT_STOP", map[string]any{"error": err.Error()})
		return pr, err
	}
	enter := func(s string) {
		res.Steps = append(res.Steps, s)
		if step != nil {
			step(s)
		}
	}

	// Resolve-RepackMode (PS:13088-13101).
	if m := strings.ToLower(strings.TrimSpace(req.RepackMode)); m != "" {
		if m != RepackPacked && m != RepackVirtual {
			return fail(stop(StatusFailCustomArgs, "invalid -RepackMode '%s': expected packed or virtual", req.RepackMode))
		}
		res.RepackMode = m
	}
	virtual := res.RepackMode == RepackVirtual

	enter(StepCatalog)
	cat, err := ReadCatalog(l.opt.CatalogPath, l.opt.BundleRoot)
	if err != nil {
		return fail(err)
	}
	pr.cat = cat
	res.SourceTag = cat.SourceTag
	res.WebUI = cat.Runtime.WebUI

	enter(StepIdentify)
	// Resolve-ModelPath (PS:13660-13674).
	p := strings.Trim(strings.TrimSpace(req.ModelPath), `"`)
	if p == "" {
		return fail(stop(StatusFailModelPath, "empty model path"))
	}
	if !fileExists(p) {
		return fail(stop(StatusFailModelPath, "model file not found: %s", p))
	}
	if abs, err := filepath.Abs(p); err == nil {
		p = abs
	}
	req.ModelPath = p
	res.ModelPath = p
	set, le := readModelShardSet(p)
	if le != nil {
		return fail(le)
	}
	res.Arch, res.Shards, res.TotalBytes = set.Arch, len(set.Shards), set.TotalBytes
	tplOn, _ := l.archTemplateOn()
	sel, err := l.resolveProfileSelection(ctx, cat, set, templateAdmissible(tplOn, set.Arch), req.ProfileID)
	if err != nil {
		return fail(err)
	}
	res.Shas = sel.Shas
	prof := sel.Profile
	if sel.Kind == SelectionTemplate {
		// GL-14 (W7 duties 5 and 6, conflict 14): a model outside the catalog with a template
		// structure is accepted; its profile is derived by the template plan (PS:13914-13937).
		// Virtual stays refused before the plan runs (VR1 §0, kept by F-PF §7.6; PS:13923).
		if virtual {
			return fail(stop(StatusFailCustomArgs, derivedVirtualRefusal))
		}
		planOut, err := resolveOutputDir(req, false)
		if err != nil {
			return fail(err)
		}
		dp, err := l.derivePlan(ctx, cat, p, planOut, set, sel.Shas)
		if err != nil {
			return fail(err)
		}
		prof = dp
		res.Notices = append(res.Notices, "outside the catalog: start settings derived for the "+set.Arch+
			" structure ("+dp.Derived.DerivedFrom+", "+dp.ID+"); the performance of these settings is unverified")
	}
	res.Selection, res.Profile = sel.Kind, prof
	settings := req.Settings
	var presetValues map[string]string // the saved values applied (F-PF §4.2's source)
	if !req.NoPreset {
		presetID := prof.ID
		if prof.Derived != nil { // a model outside the catalog keys its preset by file (GL-16)
			presetID = ""
		}
		pre, ps := l.LoadPreset(ctx, p, presetID)
		res.Preset, res.PresetState = pre, ps
		switch ps {
		case PresetApplies:
			settings = mergePreset(req.Settings, pre.Values)
			presetValues = pre.Values
			if m, ok := pre.Values["repack_mode"]; ok && strings.TrimSpace(req.RepackMode) == "" {
				res.RepackMode = m
				virtual = m == RepackVirtual
			}
			res.Notices = append(res.Notices, "saved settings applied ("+l.PresetPath(presetID, p)+")")
		case PresetOtherFile:
			res.Notices = append(res.Notices, "saved settings not applied: they were saved for another file")
		case PresetUnreadable, PresetInvalid:
			res.Notices = append(res.Notices, NoticePresetNotUsed)
		}
	}
	if prof.Derived != nil && virtual { // a saved repack mode cannot open what PS:13923 closes
		return fail(stop(StatusFailCustomArgs, derivedVirtualRefusal))
	}
	ov, notices, err := resolveOverrides(settings, prof.Bounds, virtual, req.customPath())
	if err != nil {
		return fail(err)
	}
	res.Overrides = ov
	res.Notices = append(res.Notices, notices...)
	// F-PF §9 C6 (S8-PF-07): K at most the model's n_expert, read from the GGUF; refused
	// before start, so no K or N reaches an env.
	pair := manualPairOf(ov)
	if pair != nil {
		e, ok := set.ArchLong(".expert_count")
		if !ok {
			e = prof.Identify.NExpert
		}
		if e > 0 && pair.K > e {
			return fail(stop(StatusFailCustomArgs, "%s · %s", CustomValueText(req.customPath(), "prefetch_k", ov["prefetch_k"],
				fmt.Sprintf("1..%d", min(int64(PrefetchKMax), e))), refuseKAboveText(pair.K, e)))
		}
	}
	// GL-20 (F-PF §7.3, S8-PF-19): on a virtual repack a QD below the model's wave width cannot
	// work; refused before start, a text distinct from the notice.
	if v, ok := ov["qd"]; ok && virtual {
		if w, known := waveWidthsFor(prof.ID); known {
			if q, _ := strconv.ParseInt(v, 10, 64); q < w.Max() {
				return fail(stop(StatusFailCustomArgs, "%s · %s", CustomValueText(req.customPath(), "qd", v,
					fmt.Sprintf("%d..%d", w.Max(), EngineQDMax)), refuseWaveText(q, w.Max())))
			}
		}
	}
	if sel.Kind == SelectionMismatch {
		res.Notices = append(res.Notices, "the catalog pin for this profile does NOT match this file - continuing with the catalog geometry, prefetch off, and every reference claim withheld")
	}

	enter(StepArtifacts)
	out, err := resolveOutputDir(req, virtual)
	if err != nil {
		return fail(err)
	}
	res.OutputDir = out
	if mode == modeStopBeforeStart {
		// SMD -Action stop with -NonInteractive (PS:13996-13998): after the exclusive locks and
		// before any eligibility decision the run stops as cancelled_user (exit 2). The locks are
		// probed, so a second launcher still sees «another launcher is already running» (W7 7).
		enter(StepLocks)
		if err := probeLocksForStop(l.lockPrefix, prof.ID, out); err != nil {
			return fail(err)
		}
		return fail(stop(StatusCancelledUser, "stop requested before start"))
	}
	// Conflict 16: the artifact disposition and the format gate of the selected mode, read
	// again at every start (CheckFormat; PS:14028-14080, :14186-14196, Resolve-VirtualArtifactState).
	// A derived profile's gate binds the arch-template lock id and the bytes of the derived
	// expect the template setup wrote (PS:14176-14185).
	gateID, gateSha := prof.ID, prof.ExpectSha256
	if prof.Derived != nil {
		var gerr error
		if gateID, gateSha, gerr = derivedGateIdentity(prof); gerr != nil {
			return fail(gerr)
		}
	}
	gate := CheckFormat(res.RepackMode, out, gateID, gateSha)
	gate.At = l.os.now()
	res.Gate = &gate
	if prof.Derived != nil && gateSha == "" && gate.State != GateNeedsRepack {
		return fail(stop(StatusFailGateVerify, "the derived expect could not be hashed - %s is missing", prof.Derived.ExpectPath))
	}
	l.diag.write("FORMAT_GATE", map[string]any{"mode": gate.Mode, "out": out, "state": gate.State.String(),
		"items": gate.Items, "passed": gate.Passed, "failed_item": gate.FailedItem, "reason": gate.Reason})
	switch {
	case !virtual && sel.Kind == SelectionMismatch && gate.State != GateNeedsRepack && !gate.Partial:
		// P4 2.5 (b): artifacts that nothing can bind to these bytes are stale.
		return fail(&RepackNeededError{Status: StatusFailRepack, Mode: res.RepackMode, OutputDir: out, Stale: true,
			Reason: "stale artifacts: this file is not the bytes the catalog profile pins and " + out + " cannot be shown to come from it"})
	case gate.State == GateNeedsRepack:
		return fail(&RepackNeededError{Status: StatusFailRepack, Mode: res.RepackMode, OutputDir: out, Partial: gate.Partial, Reason: gate.Reason})
	case gate.State == GateFail:
		return fail(&GateError{LaunchError: stop(StatusFailGateVerify, "%s", gate.Reason), Gate: gate})
	}
	res.ManifestSha, res.CheckedAt = gate.ManifestSha, gate.CheckedAt
	if st, err := os.Stat(filepath.Join(l.opt.StateDir, "presets.user.json")); err == nil && !st.IsDir() {
		res.Notices = append(res.Notices, "the stored user preset is not read (user presets are not ported); the request's settings are the only overrides")
	}

	enter(StepLocks)
	if mode == modeStart {
		h, err := acquireLauncherLocks(l.lockPrefix, prof.ID, out)
		if err != nil {
			return fail(err)
		}
		pr.locks = h
	} else if l.activeSession() {
		res.Notices = append(res.Notices, "a server started by this launcher is running; the instance locks were not probed")
	} else {
		h, err := acquireLauncherLocks(l.lockPrefix, prof.ID, out)
		if err != nil {
			return fail(err)
		}
		h.release()
	}

	enter(StepSweep)
	request := prefetchRequestDflt
	if v, ok := ov["prefetch"]; ok {
		request = v
	}
	optIn := optInOf(request)
	arm := prefetchArm(prof, optIn, sel.Kind)
	var armN *int64
	if pair != nil && !derivedTMismatch(prof, set) {
		// GL-10 [A-SWEEP] (F-PF §3.5, M-12 (i)): a manual pair is the manual arm, and the sweep
		// prefers the smallest QD of its set that clears the set N+1. Step 2 (a derived row
		// whose t disagrees) comes before step 3, so that row keeps its own arm.
		arm, armN = armManual, &pair.N
	}
	res.SweepArm = arm
	var sw Sweep
	if virtual {
		sw = newVirtualPinnedQD()
	} else {
		sw = l.resolveSweep(ctx, out, cat.SourceTag, prof.ID, prof.ExpectSha256, res.ManifestSha, res.CheckedAt, prof, arm, armN, mode == modeStart)
	}
	qd := sw.QD
	if !sw.OK {
		qd = QDDegraded
	}
	res.Sweep = sw
	res.QDSource = qdSource(ov, sw)

	enter(StepConfig)
	effQD := qd
	if v, ok := ov["qd"]; ok {
		effQD, _ = strconv.Atoi(v)
	}
	decision := resolvePrefetchManual(prof, sw.OK, optIn, effQD, sel.Kind, request,
		manualInput{pair: pair, virtual: virtual, source: prefetchSource(req, presetValues, ov)})
	if prof.Derived != nil {
		// P4 3 (PS:7514-7521): the plan's t against the header's, read apart (F-PF §3.2 step 2).
		if derivedTMismatch(prof, set) {
			d := decision
			decision = Prefetch{Echo: offEcho("derived_t_mismatch"), OffReason: "derived_t_mismatch", CandidateActivation: "off",
				Request: request, Evidence: d.Evidence, Activation: d.Activation, Identity: d.Identity, Source: d.Source,
				InitVersion: PrefetchInitVersion}
		}
	}
	if decision.Warning != "" {
		res.Notices = append(res.Notices, decision.Warning)
	}
	installed, instErr := l.os.installed()
	geo, geoErr := readSlotGeometry(out)
	cfg, err := buildEffectiveConfig(configInput{cat: cat, profile: prof, root: l.opt.BundleRoot, outputDir: out,
		modelPath: p, ov: ov, decision: decision, qd: qd, installed: installed, installedErr: instErr,
		geo: geo, geoErr: geoErr, metricsDir: l.LogDir(), stamp: pr.stamp,
		alias: ModelAlias(prof.ID, p), cache: ResolveCap(req.PromptCacheCap), virtual: virtual})
	if err != nil {
		if le, ok := AsLaunchError(err); ok && le.Status == StatusFailResource && strings.HasPrefix(le.Reason, autotuneStopPrefix) {
			l.diag.write("BUDGET_AUTOTUNE", map[string]any{"source": "fail_resource", // PS :3958's banner (F-BU H.2)
				"banner": BannerAutotuneImpossible(strings.TrimPrefix(le.Reason, autotuneStopPrefix))})
		}
		return fail(err)
	}
	res.Config = cfg
	if cfg.Prefetch.Notice != "" { // F-PF §3.2 3c, shown while the candidate is on (§6.1)
		res.Notices = append(res.Notices, cfg.Prefetch.Notice)
	}
	if virtual && virtualOffCatalogPolicy(cfg.QD, request, pair, prof) { // GL-20: F-PF §7.2 (and §3.2 3c on virtual)
		res.Notices = append(res.Notices, NoticeVirtualUnvalidated)
	}
	if w, known := waveWidthsFor(prof.ID); virtual && known && cfg.Prefetch.On { // GL-20: N beside N_cap (F-PF §8.2-§8.3)
		cfg.Prefetch.NCapPerLayer = nCapPerLayer(cfg.Prefetch.N, int64(cfg.QD), w)
		if nCapZeroEverywhere(cfg.Prefetch.NCapPerLayer) {
			res.Notices = append(res.Notices, NoticeNCapZero)
		}
		l.diag.write("PREFETCH_N_CAP", map[string]any{"n": cfg.Prefetch.N, "qd": cfg.QD, "line": nCapLogLine(cfg.Prefetch.NCapPerLayer)})
	}
	if mode == modeStart { // GL-15: Initialize-Warmstart and Build-EffectiveConfig's warm part (PS:8578-8598)
		res.Warm = l.newWarmStart(prof.ID, prof.Derived != nil, l.SmokeDue(req.Smoke, p, res.RepackMode), req.switches())
		l.warmConfigure(ctx, res.Warm, cfg, l.opt.BundleRoot, res.ManifestSha, siblingPaths(p))
	}
	res.SettingsKey, res.SettingsLine = settingsKey(prof.ID, p, res.RepackMode, cfg)
	res.Start = startSettingsOf(cfg, res.RepackMode)
	res.Start.PromptCacheCap, res.Start.WarmStart, res.Start.Autosave = req.PromptCacheCap.String(),
		onOff(req.switches().WarmStart), onOff(req.switches().Autosave)
	res.Warmup = warmupValue(ov)
	res.SpeedTokPerSec, res.SpeedMeasured = l.SpeedFor(res.SettingsKey)

	enter(StepSizing)
	// Confirm-EffectiveSizing (PS:13702-13716) -> Invoke-Preflight (PS:3650-3740), no repack.
	free, total, root, verr := l.os.volumeFree(out)
	if verr != nil {
		return fail(stop(StatusFailResource, "output volume query failed: %v", verr))
	}
	res.DiskFreeMiB, res.DiskRoot = free, root
	_ = total
	ctxTok := int64(0)
	if v, ok := argvValue(cfg.Argv, "-c"); ok {
		ctxTok, _ = strconv.ParseInt(v, 10, 64)
	}
	mem, merr := l.os.mem()
	res.RAM = RAMVerdict{Verdict: "unmeasured", BudgetMB: cfg.Budget.MB, Ctx: ctxTok}
	if merr != nil {
		res.RAM.Reason = "GlobalMemoryStatusEx failed"
	} else {
		// Review item 13: measured now; while this launcher's server runs, a start can only
		// follow its stop, so its share counts as free for the start and is reported apart.
		if pm, ok := l.serverMemory(); ok && mode == modePreflight {
			res.ServerMiB = ceilMiB(pm.WorkingSetBytes)
		}
		res.FreeMiB = mem.AvailPhysMiB
		res.Lists, res.ListsOK = l.readMemoryLists() // GL-09 (U-21)
		res.FreeForStartMiB = mem.AvailPhysMiB + res.ServerMiB
		res.RAM.AvailMiB, res.RAM.TotalMiB = res.FreeForStartMiB, mem.TotalPhysMiB
		if cfg.Budget.MB > res.FreeForStartMiB {
			res.RAM.Verdict, res.RAM.Basis = "insufficient", "budget_only"
			res.RAM.Reason = ReasonBudgetOverRAM // PS :3601 (F-BU H.2)
		} else {
			res.RAM.Reason = "dense/KV/server/headroom terms not yet measured (LS 9 item 1)"
		}
	}
	// Decision ② / ⑧ and ③: the last measurement of this key; the need by review item 13
	// (the last run's peak, else the ready value + that run's budget).
	res.NeedBasis = NeedNotMeasured
	if rec, ok := l.HistoryFor(res.SettingsKey); ok {
		res.LastReady, res.HasLastReady = rec.ReadyDuration(), rec.ReadyMillis > 0
		if rec.Start == nil {
			rec.Start = &StartSettings{BudgetMB: cfg.Budget.MB} // a record older than GOL-FIX1: the same key has this budget
		}
		res.NeedMiB, res.NeedBasis = NeedOf(&rec)
		res.NeedKnown = res.NeedBasis != NeedNotMeasured
	}
	// GL-05: the projection; the method's «validated» flag picks it or the last measured value
	// (S8-LA-13; F-BU B.6, D.3)
	res.Fit, res.Projection = l.projection(ctx, cat, set, cfg)
	res.NeedKnown, res.NeedMiB, res.NeedBasis, res.NeedLabel = ShownNeed(res.NeedMiB, res.NeedBasis, res.Projection,
		l.projectionValidated())
	l.commitPrediction(res, cfg)
	if merr == nil {
		res.Check = EvaluateStart(res.FreeForStartMiB, res.NeedMiB, res.NeedBasis, cfg.Budget.MB)
		if res.NeedKnown {
			res.LeftMiB = res.Check.LeftMiB
			res.Verdict = MemFits
			if res.Check.Kind == CheckTight {
				res.Verdict = MemShort
			}
		}
	}
	res.VRAM = l.os.vram()
	l.diag.write("PREFLIGHT", map[string]any{"ram": res.RAM, "vram": res.VRAM, "budget_mb": cfg.Budget.MB,
		"need_known": res.NeedKnown, "need_mib": res.NeedMiB, "need_basis": res.NeedBasis, "free_mib": res.FreeMiB,
		"server_mib": res.ServerMiB, "check": res.Check.Kind.String(), "key": res.SettingsKey})
	if res.RAM.Verdict == "insufficient" {
		// Conflict 7: the refusal carries the largest budget at which the start would pass.
		b, bounded := prof.Bounds["budget_mb"]
		res.BudgetCap = BudgetCap(res.FreeForStartMiB, b, bounded)
		why := res.RAM.Reason
		ctxTok, _ := strconv.ParseInt(func() string { v, _ := argvValue(cfg.Argv, "-c"); return v }(), 10, 64)
		l.diag.write("RAM_PREFLIGHT", map[string]any{"verdict": res.RAM.Verdict, // PS :3695's ram line (F-BU H.2)
			"line": LineRAMInsufficient(cfg.Budget.MB, res.FreeForStartMiB, ctxTok, res.RAM.Basis)})
		if res.BudgetCap.OK {
			why += fmt.Sprintf("; a budget of at most %d GB would pass", res.BudgetCap.GiB)
		} else {
			why += "; " + res.BudgetCap.Reason
		}
		return fail(stop(StatusFailResource, "%s", StopRAMPreflight(why))) // PS :3696
	}
	if pr.locks != nil {
		if err := pr.locks.setPortLock(cfg.Port); err != nil {
			return fail(err)
		}
	} else if !l.activeSession() {
		tmp := heldLocks{prefix: l.lockPrefix}
		if err := tmp.setPortLock(cfg.Port); err != nil {
			return fail(err)
		}
		tmp.release()
	}
	return pr, nil
}

// Preflight checks one model with its settings: the values, the memory verdict of decision
// ②, and the PowerShell refusals (conflicts 2, 7, 8, 17). It holds no lock afterwards and
// starts nothing. The error is the refusal (a *LaunchError or a *NotPortedError); the result
// carries whatever was found before it.
func (l *Launcher) Preflight(ctx context.Context, req Request) (*PreflightResult, error) {
	pr, err := l.prepare(ctx, req, modePreflight, nil)
	return pr.res, err
}

// StopBeforeStart is the argument-driven mode's stop action (W7 duty 23; F-LS 9.3, OQ-2 (b),
// argument -stop), as SMD's -Action stop with -NonInteractive does it (PS:13996-13998): the
// model is identified and the launcher locks are probed, then the run stops before start. It
// starts nothing and ends nothing. The error is the run's verdict: cancelled_user (exit 2), or
// what refused first (another launcher holding a lock: fail_instance_lock, exit 3).
func (l *Launcher) StopBeforeStart(ctx context.Context, req Request) error {
	_, err := l.prepare(ctx, req, modeStopBeforeStart, nil)
	if err == nil {
		err = stop(StatusCancelledUser, "stop requested before start")
	}
	return err
}

// derivedTMismatch is P4 3's test (PS:7514-7521): a derived profile whose plan t is not the
// header's t (F-PF §3.2 step 2).
func derivedTMismatch(prof *Profile, set *ShardSet) bool {
	if prof.Derived == nil {
		return false
	}
	t, ok := set.ArchLong(".expert_used_count")
	return ok && t != prof.Identify.NExpertUsed
}
