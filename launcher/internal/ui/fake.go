// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// FakeBackend drives the screens without a server: the tests use it, and a
// command may use it to show the screens. Its numbers are test data, not
// measurements; fields it leaves empty show the not-ported state.

package ui

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// the fake implements the whole interface
var _ Backend = (*FakeBackend)(nil)

// FakeBackend is a scripted Backend.
type FakeBackend struct {
	mu sync.Mutex

	Cat       Catalog
	Views     map[string]SettingsView
	Pre       Preflight
	StartEvs  []StartEvent
	StartHold bool // keep the start channel open after StartEvs
	Statuses  []ServeStatus
	Mem       MemoryReading // while a server is up
	MemIdle   MemoryReading // while none is
	Logs      []LogLine
	Info      LauncherInfo
	StopEvs   map[StopMode][]StopEvent
	StopHold  bool // keep the stop channel open after the events
	ActionErr error
	Plan      RepackPlan    // the plan box of a model that needs a setup (conflict 9)
	RepackEvs []StartEvent  // the setup's events, then the start's
	Bundles   []BundleCheck // the launcher's file check, read in turn; the last stays
	nBundle   int

	calls   []string
	nStatus int
	open    []chan struct{}
	up      bool
}

// NewFake gives a fake with one present model, one missing model, a slice
// start that reaches ready, a serving status and a stop with a summary.
func NewFake() *FakeBackend {
	t0 := time.Date(2026, 10, 10, 9, 0, 0, 0, time.Local)
	f := &FakeBackend{
		Cat: Catalog{
			Models: []ModelRow{
				{ID: "qwen35-122b", Name: "Qwen3.5-122B", Present: true, Catalog: true, Path: `D:\moe-models\qwen.gguf`, Folder: `D:\moe-models`,
					SizeGB: K(70.2), Quant: "Q4_K_M", Identity: "sha256 3f9a…c21e", FileCheck: FileSame,
					Modes:  []ModeCheck{{"packed", ModeReady}, {"virtual", ModeReady}},
					NeedGB: K(19.0), FreeRAMGB: K(24.6), LeftGB: K(5.6), CacheGB: K(12.0),
					VRAMUseGB: K(0.6), VRAMTotalGB: K(16),
					DecodeTokS: K(5.5), MeasuredOn: "10-09",
					Fit: FitRoom, Setup: SetupReady, SetupNote: "virtual repack present"},
				{ID: "dsv4-flash", Name: "DSV4-Flash", Present: false, Catalog: true, Path: `D:\moe-models\dsv4.gguf`,
					NeedGB: U, FreeRAMGB: K(24.6), LeftGB: U, Fit: FitUnmeasured, Setup: SetupNotPorted},
				{ID: "gpt-oss-20b", Name: "gpt-oss-20b", Present: true, Path: `D:\moe-models\gpt-oss-20b.gguf`,
					SizeGB: K(12.1), FileCheck: FileNoReference, NeedGB: U, FreeRAMGB: K(24.6), LeftGB: U,
					Fit: FitUnmeasured, Setup: SetupReady},
				// its repack output was deleted while the launcher ran (review item 31)
				{ID: "qwen3-30b", Name: "Qwen3-30B-A3B", Present: true, Catalog: true, Path: `D:\moe-models\qwen3-30b.gguf`,
					SizeGB: K(17.3), FileCheck: FileSame, NeedGB: K(6.0), FreeRAMGB: K(24.6), LeftGB: K(18.6),
					Fit: FitRoom, Setup: SetupNeedsRepack, SetupNote: "repack output deleted",
					Modes: []ModeCheck{{"packed", ModeReady}, {"virtual", ModeNeedsRepack}}},
				// outside the catalog, started with its structure's defaults (conflict 14)
				{ID: "gpt-oss-120b-local", Name: "gpt-oss-120b-local", Present: true, Template: true, Arch: "gpt-oss", Path: `E:\models\gpt-oss-120b.gguf`,
					SizeGB: K(60.9), FileCheck: FileNoReference, NeedGB: U, FreeRAMGB: K(24.6), LeftGB: U,
					Fit: FitUnmeasured, Setup: SetupReady},
			},
			Source: "models.json",
		},
		Views: map[string]SettingsView{
			"qwen35-122b": {Model: "qwen35-122b", Rows: []SettingRow{
				{Key: "budget", Label: "RAM budget", Value: "12 GB", Options: []string{"8 GB", "12 GB", "16 GB"}, Editable: true, Hint: "expert cache only · the whole model uses more",
					Running: "12 GB", Typed: true, Unit: "GB", Range: "8 GB ~ 20 GB", Default: "12 GB"},
				{Key: "ctx", Label: "context", Value: "12288", Options: []string{"8192", "12288", "16384"}, Editable: true, Hint: "how much text one chat can hold",
					Running: "12288", Typed: true, Range: "2048 ~ 131072 (64k)", Default: "12288"},
				{Key: "qd", Label: "queue", Value: "8", Options: []string{"4", "8", "16"}, Editable: true, Hint: "QD 8 with the catalog prefetch is the validated pair",
					Running: "8", Typed: true, Range: "1 ~ 32", Default: "8"},
				{Key: "threads", Label: "threads", Value: "8", Options: []string{"8", "12"}, Editable: true, Folded: true, Running: "8", Typed: true, Range: "1 ~ 64", Default: "8"},
				// conflict 2: prefetch and the repack mode change with ←→; KV type, the
				// check gate and -np 1 stay locked; the network is gone (conflict 17)
				{Key: "prefetch", Label: "prefetch", Value: "catalog", Options: []string{"catalog", "init"}, Editable: true, Folded: true, Running: "catalog", Default: "catalog"},
				{Key: "mode", Label: "repack mode", Value: "virtual", Options: []string{"packed", "virtual"}, Editable: true, Folded: true, Running: "virtual", Default: "virtual"},
				{Key: "kv", Label: "KV type", Value: "f16", Locked: "fixed for now", Folded: true, Running: "f16", Default: "f16"},
				{Key: "np", Label: "parallel", Value: "1", Locked: "one request at a time", Folded: true, Running: "1", Default: "1"},
			}},
		},
		Pre: Preflight{Model: "qwen35-122b", AvailGB: K(24.6), NeedGB: K(19.0), LeftGB: K(5.6), CacheGB: K(12.0),
			Verdict: VerdictRoom, LastStartS: K(20), SettingsLine: "12 GB · ctx 12288 · QD 8 · prefetch catalog"},
		StartEvs: []StartEvent{
			{Model: "qwen35-122b", Elapsed: 1 * time.Second, LastS: K(20), Steps: []Step{
				{Name: "check memory", State: StepDone, Detail: "needs 19.0 GB · 24.6 GB free", Took: 400 * time.Millisecond},
				{Name: "start server", State: StepRunning, Detail: "llama-server.exe on :8093"},
				{Name: "load model", State: StepTodo},
				{Name: "ready", State: StepTodo, Detail: "/health answers ok"},
			}},
			{Model: "qwen35-122b", Elapsed: 10 * time.Second, LastS: K(20), LoadRunning: true, Steps: []Step{
				{Name: "check memory", State: StepDone, Took: 400 * time.Millisecond},
				{Name: "start server", State: StepDone, Took: 800 * time.Millisecond},
				{Name: "load model", State: StepRunning},
				{Name: "ready", State: StepTodo},
			}},
			{Model: "qwen35-122b", Elapsed: 19 * time.Second, LastS: K(20), LoadReady: true, LoadTook: 19 * time.Second, Ready: true, Address: "http://127.0.0.1:8093/v1", Steps: []Step{
				{Name: "check memory", State: StepDone}, {Name: "start server", State: StepDone},
				{Name: "load model", State: StepDone}, {Name: "ready", State: StepDone},
			}},
		},
		Statuses: []ServeStatus{{
			Model: "qwen35-122b", Alias: "qwen3.5-122b", Address: "http://127.0.0.1:8093/v1", Port: 8093, ChildPID: 4242, ChildAlive: true,
			Health: "ok", Uptime: 62 * time.Minute, CtxTotal: K(12288), CtxUsed: K(9400), Phase: ReqDecode,
			DecodeTokS: K(5.6), LastDecodeTokS: K(5.5), ReusedTokens: K(3100), NewTokens: K(6100), PrefillTokS: K(312), Batch: K(2048),
			DecodedTokens: K(214), TTFTs: K(19.8), FreeRAMGB: K(4.7), FreeVRAMGB: K(2.6),
			ModelRAMGB: K(18.2), OtherRAMGB: K(9.1), TotalRAMGB: K(32),
			ModelVRAMGB: K(12.9), UsedVRAMGB: K(13.4), TotalVRAMGB: K(16),
			SettingsLine:  "virtual, ctx 12288, budget 12288, qd 8, prefetch catalog",
			SettingsWords: []string{"12k context", "12 GB cache", "virtual", "QD 8", "prefetch catalog"},
		}},
		Mem: MemoryReading{TotalRAMGB: K(32), ModelGB: K(18.2), AppsGB: K(10.1), FreeGB: K(3.7), ReadyNowGB: K(4.7),
			TotalVRAMGB: K(16), ModelVRAMGB: K(12.9), OtherVRAMGB: K(0.3), UsedVRAMGB: K(13.4),
			Apps: []AppMem{{Name: "llama-server.exe", GB: K(18.2), Tag: "this model"}, {Name: "chrome.exe", GB: K(7.4)}}},
		MemIdle: MemoryReading{TotalRAMGB: K(32), AppsGB: K(7.4), FreeGB: K(20.1), ReadyNowGB: K(24.6),
			TotalVRAMGB: K(16), UsedVRAMGB: K(0.6)},
		Logs: []LogLine{
			{At: t0, Level: LogInfo, Text: "catalog read: 6 profiles"},
			{At: t0.Add(19 * time.Second), Level: LogOK, Text: "server ready on :8093"},
			{At: t0.Add(20 * time.Second), Level: LogNotice, Text: "[moe-direct] mode=virtual raises io_bounce_copy_qd 2 -> 14 (record fan-out shape floor)"},
			{At: t0.Add(30 * time.Minute), Level: LogWarn, Text: "RAM low: 1.2 GB free"},
		},
		Info:    LauncherInfo{Address: "127.0.0.1:8093", LogFile: `C:\state\logs\launcher.log`, Smoke: SmokeOff},
		StopEvs: map[StopMode][]StopEvent{},
		Plan: RepackPlan{Mode: "virtual", Does: "builds the virtual layout of the experts from the GGUF (the GGUF is not changed)",
			WritesTo: `D:\moe-models\repack\qwen3-30b\virtual`, WriteGB: K(17.5), DriveFreeGB: K(412), LeftGB: K(394.5)},
		RepackEvs: []StartEvent{
			{Model: "qwen3-30b", Elapsed: 2 * time.Second, Setup: &SetupProgress{Name: "repack", Done: K(12), Total: K(48),
				Elapsed: 40 * time.Second, WrittenGB: K(4.4), TotalGB: K(17.5), DiskMBs: K(900)}},
		},
	}
	run := RunSummary{Model: "qwen35-122b", Started: t0, Ended: t0.Add(72 * time.Minute), Requests: K(38), Errors: K(0),
		DecodeAvg: K(5.5), DecodeLow: K(3.6)}
	for _, mode := range []StopMode{StopUnload, StopQuit, StopForce, StopCancel} {
		f.StopEvs[mode] = []StopEvent{
			{Mode: mode, Elapsed: 100 * time.Millisecond, Save: SaveProgress{State: StepNotPorted}, ModelRAMGB: K(18.2), Steps: []Step{
				{Name: "save prompt cache", State: StepNotPorted},
				{Name: "stop the server", State: StepRunning, Detail: "CTRL_BREAK to the child"},
				{Name: "close launcher", State: StepTodo},
			}},
			{Mode: mode, Done: true, Result: &StopResult{ExitCode: 0, Graceful: true, Run: run}},
		}
	}
	return f
}

func (f *FakeBackend) call(s string) {
	f.mu.Lock()
	f.calls = append(f.calls, s)
	f.mu.Unlock()
}

// Calls lists the calls made so far, in order.
func (f *FakeBackend) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string{}, f.calls...)
}

// Close releases held channels.
func (f *FakeBackend) Close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, c := range f.open {
		close(c)
	}
	f.open = nil
}

func (f *FakeBackend) Catalog() (Catalog, error) {
	f.call("Catalog")
	f.mu.Lock()
	defer f.mu.Unlock()
	c := f.Cat
	c.Models = append([]ModelRow{}, f.Cat.Models...)
	return c, nil
}

func (f *FakeBackend) ModelSettings(model string) (SettingsView, error) {
	f.call("ModelSettings " + model)
	if v, ok := f.Views[model]; ok {
		return v, nil
	}
	return SettingsView{Model: model}, nil
}

func (f *FakeBackend) Preflight(model string, s Settings) (Preflight, error) {
	f.call(fmt.Sprintf("Preflight %s %v", model, s))
	p := f.Pre
	p.Model = model
	// the user's decision of 2026-10-10 00:44 KST and OQ-5 (b) of 2026-10-12: QD
	// outside the default is accepted with the notice (here as test data)
	if q, ok := s["qd"]; ok && q != "8" {
		p.Notices = append(append([]string{}, p.Notices...), "default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured")
	}
	return p, nil
}

func (f *FakeBackend) Start(model string, s Settings) (<-chan StartEvent, error) {
	f.call(fmt.Sprintf("Start %s %v", model, s))
	f.mu.Lock()
	f.up = true
	f.mu.Unlock()
	ch := make(chan StartEvent, len(f.StartEvs))
	for _, e := range f.StartEvs {
		ch <- e
	}
	if !f.StartHold {
		close(ch)
		return ch, nil
	}
	return f.held(ch), nil
}

// held forwards a buffered channel and keeps it open until Close.
func (f *FakeBackend) held(in chan StartEvent) <-chan StartEvent {
	out := make(chan StartEvent)
	stop := make(chan struct{})
	f.mu.Lock()
	f.open = append(f.open, stop)
	f.mu.Unlock()
	go func() {
		defer close(out)
		for {
			select {
			case e := <-in:
				select {
				case out <- e:
				case <-stop:
					return
				}
			case <-stop:
				return
			}
		}
	}()
	return out
}

func (f *FakeBackend) Status() (ServeStatus, error) {
	f.call("Status")
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.Statuses) == 0 {
		return ServeStatus{}, ErrNotPorted
	}
	i := min(f.nStatus, len(f.Statuses)-1)
	f.nStatus++
	return f.Statuses[i], nil
}

func (f *FakeBackend) Memory() (MemoryReading, error) {
	f.call("Memory")
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.up {
		return f.MemIdle, nil
	}
	return f.Mem, nil
}

// FakeLongRefusal is a refusal reason longer than any box, with a path in it
// (review item 32: it wraps, never cut).
const FakeLongRefusal = `the repack output of this model is missing: D:\moe-models\repack\qwen35-122b\experts-virtual-layout.pack was not found · press Enter to repack it again`

// RemoveModel takes a model off the list, as if its file was deleted while
// the launcher runs (review item 31).
func (f *FakeBackend) RemoveModel(id string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var ms []ModelRow
	for _, x := range f.Cat.Models {
		if x.ID != id {
			ms = append(ms, x)
		}
	}
	f.Cat.Models = ms
}

// RepackPlan gives the scripted plan (conflict 9).
func (f *FakeBackend) RepackPlan(model string, s Settings) (RepackPlan, error) {
	f.call(fmt.Sprintf("RepackPlan %s %v", model, s))
	p := f.Plan
	p.Model = model
	return p, nil
}

// Repack streams the scripted setup events, then serves (conflict 9).
func (f *FakeBackend) Repack(model string, s Settings) (<-chan StartEvent, error) {
	f.call(fmt.Sprintf("Repack %s %v", model, s))
	f.mu.Lock()
	f.up = true
	f.mu.Unlock()
	ch := make(chan StartEvent, len(f.RepackEvs))
	for _, e := range f.RepackEvs {
		ch <- e
	}
	if !f.StartHold {
		close(ch)
		return ch, nil
	}
	return f.held(ch), nil
}

// Bundle gives the scripted file checks in turn (conflict 18-2).
func (f *FakeBackend) Bundle() (BundleCheck, error) {
	f.call("Bundle")
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.Bundles) == 0 {
		return BundleCheck{State: BundleOK}, nil
	}
	b := f.Bundles[min(f.nBundle, len(f.Bundles)-1)]
	f.nBundle++
	return b, nil
}

// SavePreset keeps the changed values as the model's new defaults.
func (f *FakeBackend) SavePreset(model string, s Settings) (SettingsView, error) {
	f.call(fmt.Sprintf("SavePreset %s %v", model, s))
	f.mu.Lock()
	defer f.mu.Unlock()
	v := f.Views[model]
	v.Saved = true
	rows := append([]SettingRow{}, v.Rows...)
	for i := range rows {
		if nv, ok := s[rows[i].Key]; ok {
			rows[i].Value = nv
		}
	}
	v.Rows = rows
	if f.Views == nil {
		f.Views = map[string]SettingsView{}
	}
	f.Views[model] = v
	return v, nil
}

// CheckSetting: test ranges; budget 8 to 20 GB, context 2048 to 131072
// ("64k" reads as 65536), queue 1 to 32, threads 1 to 64.
func (f *FakeBackend) CheckSetting(model, key, value string) (string, error) {
	f.call(fmt.Sprintf("CheckSetting %s %s %s", model, key, value))
	v := strings.TrimSpace(strings.TrimSuffix(strings.TrimSuffix(strings.TrimSpace(value), "B"), "G")) // «12 GB» or «12 G»
	mult := 1.0
	if strings.HasSuffix(strings.ToLower(v), "k") {
		v, mult = v[:len(v)-1], 1024
	}
	n, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return "", fmt.Errorf("%q is not a number", value)
	}
	n *= mult
	lim := map[string][2]float64{"budget": {8, 20}, "ctx": {2048, 131072}, "qd": {1, 32}, "threads": {1, 64}}[key]
	if lim[1] > 0 && (n < lim[0] || n > lim[1]) {
		return "", fmt.Errorf("%s is outside %g ~ %g", value, lim[0], lim[1])
	}
	if key == "budget" {
		return strconv.FormatFloat(n, 'f', -1, 64) + " GB", nil
	}
	return strconv.FormatFloat(n, 'f', -1, 64), nil
}

func (f *FakeBackend) LogTail(max int) ([]LogLine, error) {
	f.call("LogTail")
	ls := f.Logs
	if len(ls) > max {
		ls = ls[len(ls)-max:]
	}
	return ls, nil
}

func (f *FakeBackend) Launcher() (LauncherInfo, error) { f.call("Launcher"); return f.Info, nil }

func (f *FakeBackend) Stop(mode StopMode) (<-chan StopEvent, error) {
	f.call(fmt.Sprintf("Stop %d", mode))
	f.mu.Lock()
	f.up = false
	// the launcher's stop line in 4 Log (conflict 6)
	code := map[StopMode]int{StopForce: 5, StopCancel: 2}[mode]
	at := time.Date(2026, 10, 10, 10, 12, 0, 0, time.Local)
	if n := len(f.Logs); n > 0 {
		at = f.Logs[n-1].At.Add(time.Minute)
	}
	f.Logs = append(f.Logs, LogLine{At: at, Level: LogInfo, Stop: &StopRecord{Code: code, ServerCode: K(0)}})
	f.mu.Unlock()
	evs := f.StopEvs[mode]
	if f.StopHold && len(evs) > 0 {
		evs = evs[:len(evs)-1]
	}
	ch := make(chan StopEvent, len(evs)+1)
	for _, e := range evs {
		ch <- e
	}
	if !f.StopHold {
		close(ch)
		return ch, nil
	}
	out := make(chan StopEvent)
	stop := make(chan struct{})
	f.mu.Lock()
	f.open = append(f.open, stop)
	f.mu.Unlock()
	go func() {
		defer close(out)
		for {
			select {
			case e := <-ch:
				select {
				case out <- e:
				case <-stop:
					return
				}
			case <-stop:
				return
			}
		}
	}()
	return out, nil
}

func (f *FakeBackend) Action(id ActionID, arg string) (ActionResult, error) {
	f.call("Action " + string(id) + " " + arg)
	if f.ActionErr != nil {
		return ActionResult{}, f.ActionErr
	}
	switch id {
	case ActCopyAddress, ActOpenFolder, ActShowFile:
		return ActionResult{Message: string(id) + ": " + arg}, nil
	case ActSetSmoke:
		f.mu.Lock()
		f.Info.Smoke = map[string]SmokeMode{"off": SmokeOff, "on": SmokeOn, "once": SmokeOnce}[arg]
		f.mu.Unlock()
		return ActionResult{}, nil
	case ActSetCacheCap:
		v := strings.ToLower(strings.TrimSpace(arg))
		n, err := strconv.Atoi(v)
		if v != "off" && (err != nil || n != 0 && (n < 256 || n > 8192)) {
			return ActionResult{}, fmt.Errorf("outside its range · allowed 0 (off) or 256..8192 MiB · the value is kept, nothing applied")
		}
		if v == "0" {
			v = "off"
		}
		f.mu.Lock()
		f.Info.CacheCap = v
		f.mu.Unlock()
		return ActionResult{}, nil
	case ActTaskManager: // GL-09: opened without a selection (U-35 (i))
		return ActionResult{Message: "Task Manager opened"}, nil
	case ActDeletePartial: // GL-10 (H4): the half-made files go; the model needs its setup again
		f.mu.Lock()
		for i := range f.Cat.Models {
			if f.Cat.Models[i].ID == arg && f.Cat.Models[i].Setup == SetupCutOff {
				f.Cat.Models[i].Setup, f.Cat.Models[i].Partial = SetupNeedsRepack, PartialSetup{}
			}
		}
		f.mu.Unlock()
		return ActionResult{Message: "half-made files removed · ⏎ shows the setup plan"}, nil
	case ActSetWarmStart, ActSetAutosave:
		f.mu.Lock()
		if id == ActSetWarmStart {
			f.Info.WarmStart = arg
		} else {
			f.Info.Autosave = arg
		}
		f.mu.Unlock()
		return ActionResult{}, nil
	}
	return ActionResult{}, ErrNotPorted
}
