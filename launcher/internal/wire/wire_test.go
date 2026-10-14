package wire

import (
	"errors"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

func near(t *testing.T, what string, v ui.Val, want float64) {
	t.Helper()
	if !v.Ok() || v.V < want-0.01 || v.V > want+0.01 {
		t.Errorf("%s = %+v, want %.2f", what, v, want)
	}
}

// Items 10, 11, 12, 13, 28 and 31: one row per file, the mark, the file check, the states,
// the need against the RAM free now, the last run's decode average and «needs repack».
func TestCatalogMapping(t *testing.T) {
	ml := &core.ModelList{TotalRAMMiB: 65536, Roots: []string{`D:\moe-models`, `E:\empty`},
		Files: []core.ModelFile{
			{Path: `D:\moe-models\a\a.gguf`, Name: "a.gguf", Folder: `D:\moe-models`, Bytes: 2 << 30, Catalog: true,
				ProfileID: "a", DisplayName: "A", Quant: "Q4_K_M", Identity: core.SelectionPinned,
				Sha256: []string{"3f9a0000000000000000000000000000000000000000000000000000000c21e"}, Repacked: true,
				NeedMiB: 8192, NeedBasis: core.NeedPeak,
				LastRun: &core.RunRecord{ReadyMillis: 12400, At: "2026-10-10T01:00:00Z", Start: &core.StartSettings{BudgetMB: 4096}},
				Stats:   &core.ModelStats{DecodeTokPerSec: 8.86, At: "2026-10-09T18:23:06Z"}},
			{Path: `D:\moe-models\b\b.gguf`, Name: "b.gguf", Folder: `D:\moe-models`, Bytes: 1 << 30, Catalog: true,
				ProfileID: "b", DisplayName: "B", Identity: "not hashed", NeedMiB: 30000, NeedBasis: core.NeedReadyBudget},
			{Path: `D:\moe-models\c\c.gguf`, Name: "c.gguf", Folder: `D:\moe-models`, Template: true, DisplayName: "c",
				NeedBasis: core.NeedNotMeasured},
			{Path: `D:\moe-models\d\d.gguf`, Name: "d.gguf", Folder: `D:\moe-models`, State: core.FileUnsupported,
				Reason: "architecture llama has no catalog profile and no template", DisplayName: "d"},
			{Path: `D:\moe-models\e\e.gguf`, Name: "e.gguf", Folder: `D:\moe-models`, Catalog: true, ProfileID: "e",
				Identity: core.SelectionMismatch, State: core.FileBroken, Reason: "the file differs from the catalog's pinned bytes",
				Repacked: true, DisplayName: "E"},
			{Path: `D:\moe-models\f\f.gguf`, Name: "f.gguf", Folder: `D:\moe-models`, Catalog: true, ProfileID: "f",
				Identity: "unpinned", NeedMiB: 70000, NeedBasis: core.NeedPeak, Repacked: true, DisplayName: "F"},
		}}
	c := catalogFrom(ml, "models.json", 20480, true, core.VRAMInfo{OK: true, TotalMiB: 16303, FreeMiB: 15028, UsedMiB: 952}, nil)
	if len(c.Models) != 6 || c.Source != "models.json" {
		t.Fatalf("rows %d", len(c.Models))
	}
	a, bb, cc, d, e, f := c.Models[0], c.Models[1], c.Models[2], c.Models[3], c.Models[4], c.Models[5]
	if a.ID != `D:\moe-models\a\a.gguf` || !a.Catalog || a.FileCheck != ui.FileSame || a.Identity != "pinned · sha256 3f9a…c21e" ||
		a.Fit != ui.FitRoom || a.Setup != ui.SetupReady || a.MeasuredOn == "" || a.Folder != `D:\moe-models\a` {
		t.Errorf("a %+v", a)
	}
	near(t, "a need", a.NeedGB, 8)
	near(t, "a left", a.LeftGB, 12)
	near(t, "a free", a.FreeRAMGB, 20)
	near(t, "a decode", a.DecodeTokS, 8.86)
	near(t, "a cache", a.CacheGB, 4)
	near(t, "vram use", a.VRAMUseGB, 952.0/1024)
	// Item 31: no repack output on a catalog file reads «needs repack».
	if bb.Fit != ui.FitTight || bb.FileCheck != ui.FileNotChecked || bb.Setup != ui.SetupNeedsRepack || bb.SetupNote != "repack output missing" {
		t.Errorf("b %+v", bb)
	}
	if cc.Catalog || cc.Fit != ui.FitUnmeasured || cc.FileCheck != ui.FileNoReference || cc.Setup != ui.SetupNeedsRepack || // GL-10: the template setup
		cc.DecodeTokS.State != ui.Unmeasured {
		t.Errorf("c %+v", cc)
	}
	if d.Fit != ui.FitUnsupported || !strings.Contains(d.SetupNote, "llama") {
		t.Errorf("d %+v", d)
	}
	if e.Fit != ui.FitBroken || e.FileCheck != ui.FileDiffers || !strings.Contains(e.SetupNote, "pinned") {
		t.Errorf("e %+v", e)
	}
	if f.Fit != ui.FitTooBig || f.FileCheck != ui.FileNoReference {
		t.Errorf("f %+v", f)
	}
	near(t, "too big", c.TooBig, 1)
	if len(c.Folders) != 2 || c.Folders[0].Found != 6 || c.Folders[0].State != ui.FolderWatching || len(c.Folders[0].Skipped) != 1 ||
		c.Folders[1].State != ui.FolderEmpty {
		t.Errorf("folders %+v", c.Folders)
	}
	// Item 31 as the user met it (05:12 KST): the packed output deleted, the virtual one kept.
	g := core.ModelFile{Catalog: true, ProfileID: "g", RepackedVirtual: true}
	if s, n := setupOf(g, ""); s != ui.SetupNeedsRepack || n != "packed repack output missing · virtual present" {
		t.Errorf("packed gone: %v %q", s, n)
	}
	if s, n := setupOf(g, "virtual"); s != ui.SetupReady || n != "virtual repack present" {
		t.Errorf("virtual start: %v %q", s, n)
	}
	g.Repacked = true
	if s, _ := setupOf(g, "virtual"); s != ui.SetupReady {
		t.Error("both present")
	}
	g.Path = "g.gguf"
	g.Repacked = false
	if c := catalogFrom(&core.ModelList{Files: []core.ModelFile{g}}, "x", 0, false, core.VRAMInfo{}, map[string]string{"g.gguf": "virtual"}); c.Models[0].Setup != ui.SetupReady {
		t.Errorf("modes not used: %+v", c.Models[0])
	}
	// RAM not measured: no verdict from memory.
	c = catalogFrom(ml, "x", 0, false, core.VRAMInfo{}, nil)
	if c.Models[0].Fit != ui.FitUnmeasured || c.Models[0].FreeRAMGB.State != ui.NotPorted {
		t.Errorf("no mem %+v", c.Models[0])
	}
}

// Items 2, 3 and 30: the running values, the saved values as defaults, the typed rows, the
// budget in G both ways.
func TestSettingsMapping(t *testing.T) {
	rows := []core.SettingRow{
		{Key: "port", Default: "8093", Editable: true, Min: 1024, Max: 65535, Bounded: true},
		{Key: "ctx", Default: "12288", Editable: true, Min: 2048, Max: 262144, Bounded: true},
		{Key: "budget_mb", Default: "auto", Editable: true, Min: 4096, Max: 65536, Bounded: true},
		{Key: "qd", Default: "auto", Editable: true, Min: 1, Max: 63, Bounded: true},
		{Key: "warmup", Editable: false, Locked: "not ported (after the user's decisions): launcher warmup"},
		{Key: "prefetch", Default: "catalog", Editable: true, Options: core.PrefetchRequests},
		{Key: "repack_mode", Default: "packed", Editable: true, Options: []string{"packed", "virtual"}},
	}
	running := map[string]string{"ctx": "65536", "budget_mb": "12288", "repack_mode": "virtual", "port": "8093", "qd": "8"}
	saved := map[string]string{"budget_mb": "14848"}
	v := settingsFrom("m", rows, running, saved, "saved settings (2026-10-10T05:00:00Z)")
	by := map[string]ui.SettingRow{}
	for _, r := range v.Rows {
		by[r.Key] = r
	}
	if v.Preset != "saved settings (2026-10-10T05:00:00Z)" {
		t.Errorf("preset %q", v.Preset)
	}
	b := by["budget_mb"]
	if b.Value != "14.5 GB" || b.Running != "12 GB" || !b.Typed || b.Unit != "GB" || b.Range != "4 GB ~ 64 GB" ||
		b.Options[0] != "14.5 GB" || !contains(b.Options, "auto") || !contains(b.Options, "12 GB") {
		t.Errorf("budget %+v", b)
	}
	if b.Note != "4 GB ~ 64 GB" {
		t.Errorf("budget note %q", b.Note)
	}
	if c := by["ctx"]; c.Running != "65536" || !c.Typed || c.Range != "2k ~ 256k" || c.Unit != "" || c.Value != "12288" {
		t.Errorf("ctx %+v", c)
	}
	if p := by["prefetch"]; p.Typed || len(p.Options) != 2 { // GL-19: catalog, init (F-PF §2.2)
		t.Errorf("prefetch %+v", p)
	}
	// FIX2-WIRE: the repack mode is the screen's row "mode" (the FIX2-UI contract), with its
	// default (conflict 10), the saved state (conflict 3) and the last run's values.
	if r := by["mode"]; r.Typed || r.Running != "virtual" || r.Default != "packed" || r.Label != "repack" {
		t.Errorf("repack %+v", r)
	}
	if b.Default != "auto" || !v.Saved || v.LastRun["mode"] != "virtual" || v.LastRun["budget_mb"] != "12 GB" {
		t.Errorf("defaults and last run: %+v %v %v", b, v.Saved, v.LastRun)
	}
	if v2 := settingsFrom("m", rows, nil, nil, ""); v2.Saved || v2.LastRun != nil {
		t.Errorf("no preset, no run: %+v", v2)
	}
	if req := requestFrom(`D:\m.gguf`, ui.Settings{"mode": "virtual"}); req.RepackMode != "virtual" || len(req.Settings) != 0 {
		t.Errorf("mode request %+v", req)
	}
	if pv := presetValues(ui.Settings{"mode": "virtual"}); pv["repack_mode"] != "virtual" || len(pv) != 1 {
		t.Errorf("mode preset %v", pv)
	}
	if w := by["warmup"]; w.Typed || w.Editable || w.Locked == "" {
		t.Errorf("warmup %+v", w)
	}
	if q := by["qd"]; q.Range != "1 ~ 63" || q.Running != "8" {
		t.Errorf("qd %+v", q)
	}
	// The screen's G form goes back to MB in a request and in a preset.
	req := requestFrom(`D:\m.gguf`, ui.Settings{"budget_mb": "14.5 GB", "ctx": "65536", "qd": "auto", "repack_mode": "virtual"})
	if req.Settings["budget_mb"] != "14848" || req.Settings["ctx"] != "65536" || len(req.Settings) != 2 || req.RepackMode != "virtual" {
		t.Errorf("request %+v", req)
	}
	pv := presetValues(ui.Settings{"budget_mb": "12 GB", "repack_mode": "virtual", "qd": "auto"})
	if pv["budget_mb"] != "12288" || pv["repack_mode"] != "virtual" || pv["qd"] != "auto" {
		t.Errorf("preset values %v", pv)
	}
	if toRequest("budget_mb", "12288") != "12288" || toScreen("budget_mb", "auto") != "auto" || toScreen("ctx", "8192") != "8192" {
		t.Error("forms")
	}
	w := settingWords(core.StartSettings{Ctx: 65536, BudgetMB: 12288, RepackMode: "virtual", QD: 8, PrefetchOn: true, Threads: 16})
	if strings.Join(w, ", ") != "64k context, 12 GB cache, virtual, QD 8, prefetch on, 16 threads" {
		t.Errorf("words %v", w)
	}
}

// Items 13, 22 and 32: the verdict from core's check, the live free RAM and the reasons in
// user words with the full text kept for 4 Log.
func TestPreflightMapping(t *testing.T) {
	res := &core.PreflightResult{FreeMiB: 20480, FreeForStartMiB: 20480, RAM: core.RAMVerdict{TotalMiB: 65536, AvailMiB: 20480},
		TotalBytes: 60 << 30, Config: &core.Config{Budget: core.Budget{MB: 12288}}, SettingsLine: "line",
		VRAM:      core.VRAMInfo{OK: true, TotalMiB: 8192, FreeMiB: 2048, UsedMiB: 6000},
		Notices:   []string{`saved settings applied (C:\Users\x\AppData\Local\MoE-Direct\presets\a.json)`},
		NeedBasis: core.NeedNotMeasured, Check: core.StartCheck{Kind: core.CheckUnmeasured}}
	p, full := preflightFrom("m", res, nil)
	if p.Verdict != ui.VerdictUnmeasured || p.NeedGB.State != ui.Unmeasured || p.Refused || full != "" ||
		p.Notices[0] != "saved settings applied" || !p.CommitSafeGB.Ok() || p.CommitSafeGB.V != 4 { // GL-06: F.1's 4 GiB
		t.Errorf("unmeasured %+v", p)
	}
	near(t, "vram use", p.VRAMUseGB, 6000.0/1024)
	near(t, "apps", p.AppsGB, 44)
	res.NeedKnown, res.NeedMiB, res.LeftMiB, res.NeedBasis = true, 8192, 12288, core.NeedPeak
	res.Check = core.EvaluateStart(20480, 8192, core.NeedPeak, 12288)
	if p, _ = preflightFrom("m", res, nil); p.Verdict != ui.VerdictRoom {
		t.Errorf("room %+v", p)
	}
	near(t, "left", p.LeftGB, 12)
	// The hard stop in user words, with the numbers.
	res.Check = core.EvaluateStart(8000, 8192, core.NeedPeak, 12288)
	err := &core.LaunchError{Status: core.StatusFailResource, Reason: "RAM preflight hard stop: selected budget alone exceeds available RAM (the unmeasured fixed ...)"}
	p, full = preflightFrom("m", res, err)
	if !p.Refused || p.Verdict != ui.VerdictCannot || p.Refusals[0] != "RAM budget 12 GB > free now 7.8 GB" ||
		!strings.Contains(full, "fail_resource (exit 3): RAM preflight hard stop") {
		t.Errorf("hard stop %+v / %q", p, full)
	}
	// A missing repack output: the plain reason; the coordinates go to the log only.
	p, full = preflightFrom("m", nil, core.NotPorted(core.FamilyRepack))
	if p.Refusals[0] != "repack output missing" || strings.Contains(p.Refusals[0], "PS:") ||
		!strings.Contains(full, "Invoke-Repacker PS:12353") {
		t.Errorf("repack %+v / %q", p, full)
	}
	if p, _ := preflightFrom("m", &core.PreflightResult{Notices: []string{"the stored user preset is not read (x)", "plain"}}, nil); len(p.Notices) != 1 || p.Notices[0] != "plain" {
		t.Errorf("log-only notice shown: %v", p.Notices)
	}
	if userNotice(core.NoticeVirtualUnvalidated) != core.NoticeVirtualUnvalidated || // GL-20: the T18 text exactly (S8-PF-08)
		!strings.HasPrefix(userNotice("the catalog pin for this profile does NOT match this file - x"), "this file differs") ||
		noticeLog(res) != `saved settings applied (C:\Users\x\AppData\Local\MoE-Direct\presets\a.json)` {
		t.Errorf("notices %q", noticeLog(res))
	}
	cases := map[error]string{
		core.NotPorted(core.FamilyArchTemplate):                                          "outside the catalog: the template setup is not ported yet",
		core.NotPorted(core.FamilyStartupProbe):                                          "startup probe and QD sweep is not ported",
		&core.LaunchError{Status: core.StatusFailModelPath, Reason: `no file D:\m.gguf`}: "the model file is missing or cannot be used",
		&core.LaunchError{Status: core.StatusFailInstanceLock, Reason: "port 8093 held"}: LauncherLockLine, // GL-08: W7 duty 7
		errors.New(`open D:\x\y.gguf: gone`):                                             "open gone",
	}
	for e, want := range cases {
		if got, _ := userReason(e, nil); got != want {
			t.Errorf("%v: %q want %q", e, got, want)
		}
	}
}

// Item 23: the live decode is the average over the last five seconds of /slots samples.
func TestDecodeWindow(t *testing.T) {
	var w decodeWindow
	t0 := time.Unix(1000, 0)
	if _, ok := w.rate(t0); ok {
		t.Error("rate from nothing")
	}
	// 10 tok/s for 4 s, then 2 tok/s: the window follows the recent seconds.
	n := int64(0)
	for i := 0; i <= 4; i++ {
		w.add(t0.Add(time.Duration(i)*time.Second), 7, n, true)
		n += 10
	}
	if r, ok := w.rate(t0.Add(4 * time.Second)); !ok || r != 10 {
		t.Errorf("first window %v %v", r, ok)
	}
	n -= 10
	for i := 5; i <= 12; i++ {
		n += 2
		w.add(t0.Add(time.Duration(i)*time.Second), 7, n, true)
	}
	if r, ok := w.rate(t0.Add(12 * time.Second)); !ok || r != 2 {
		t.Errorf("after the slowdown %v %v (samples %d)", r, ok, len(w.s))
	}
	// A new task starts over; an idle server has no live rate.
	w.add(t0.Add(13*time.Second), 8, 5, true)
	if _, ok := w.rate(t0.Add(13 * time.Second)); ok {
		t.Error("rate across tasks")
	}
	w.add(t0.Add(14*time.Second), 8, 15, true)
	if r, ok := w.rate(t0.Add(14 * time.Second)); !ok || r != 10 {
		t.Errorf("new task %v %v", r, ok)
	}
	w.add(t0.Add(15*time.Second), 0, 0, false)
	if _, ok := w.rate(t0.Add(15 * time.Second)); ok {
		t.Error("rate while idle")
	}
	// Through statusFrom: the live value while decoding, the last reply's average from print_timing.
	var sw decodeWindow
	st := core.ServeStatus{Alive: true, Slots: core.HTTPRead{OK: true, Body: []any{map[string]any{"is_processing": true,
		"id_task": float64(64), "n_prompt_tokens": float64(92), "next_token": []any{map[string]any{"n_decoded": float64(100)}}}}},
		Timing: core.TimingSummary{Requests: 1, AnswerTokens: 3000, AnswerTime: 338472960 * time.Microsecond,
			Done: []core.RequestTiming{{Task: 63, PromptTokens: 92, TTFT: 14333530 * time.Microsecond, PromptTokPerSec: 6.42, DecodeTokPerSec: 8.86}}},
		Start: core.StartSettings{Alias: "gpt-oss-120b", Ctx: 65536}}
	s := statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &sw, t0, false)
	if s.DecodeTokS.Ok() || s.Phase != ui.ReqDecode || s.Alias != "gpt-oss-120b" || s.SettingsWords[0] != "64k context" {
		t.Errorf("first status %+v", s)
	}
	near(t, "last reply", s.LastDecodeTokS, 8.86)
	near(t, "ttft", s.TTFTs, 14.33353)
	near(t, "run avg", s.UsualTokS, 3000/338.47296)
	near(t, "requests", s.Requests, 1)
	st.Slots.Body.([]any)[0].(map[string]any)["next_token"] = []any{map[string]any{"n_decoded": float64(130)}}
	s = statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &sw, t0.Add(3*time.Second), false)
	near(t, "live", s.DecodeTokS, 10)
}

// Items 5, 18, 29: one time-ordered list; server levels and notices kept.
func TestMergeLogs(t *testing.T) {
	spawn := time.Date(2026, 10, 9, 17, 59, 49, 738957100, time.UTC)
	server := core.ParseServerLines([]string{
		"0.00.089.177 I cmn  common_param: verbosity = 3",
		"[moe-direct] ERROR: [moe-direct] mode=virtual raises io_bounce_copy_qd 2 -> 14 (record fan-out shape floor)",
		"0.02.578.274 I srv  llama_server: listening on http://127.0.0.1:8093",
		"1.04.811.654 W srv          stop: cancel task, id_task = 66",
		"1.05.000.000 E srv  something failed",
		"",
	}, spawn)
	events := []ui.LogLine{
		{At: spawn.Add(-time.Second), Level: ui.LogInfo, Text: "start gpt-oss-120b"},
		{At: spawn.Add(3 * time.Second), Level: ui.LogOK, Text: "ready"},
		{At: spawn.Add(2 * time.Minute), Level: ui.LogInfo, Text: "stop"},
	}
	got := mergeLogs(events, server, 0)
	want := []struct {
		text  string
		level ui.LogLevel
	}{
		{"start gpt-oss-120b", ui.LogInfo},
		{"cmn  common_param: verbosity = 3", ui.LogInfo},
		{"[moe-direct] ERROR: [moe-direct] mode=virtual raises io_bounce_copy_qd 2 -> 14 (record fan-out shape floor)", ui.LogNotice},
		{"srv  llama_server: listening on http://127.0.0.1:8093", ui.LogOK},
		{"ready", ui.LogOK},
		{"srv          stop: cancel task, id_task = 66", ui.LogWarn},
		{"srv  something failed", ui.LogError},
		{"stop", ui.LogInfo},
	}
	if len(got) != len(want) {
		t.Fatalf("merged %d lines: %+v", len(got), got)
	}
	for i, w := range want {
		if got[i].Text != w.text || got[i].Level != w.level {
			t.Errorf("line %d: %q %d want %q %d", i, got[i].Text, got[i].Level, w.text, w.level)
		}
		if i > 0 && got[i].At.Before(got[i-1].At) {
			t.Errorf("line %d out of order", i)
		}
	}
	if !got[2].At.Equal(got[1].At) {
		t.Error("an unstamped line takes the previous line's time")
	}
	if tail := mergeLogs(events, server, 3); len(tail) != 3 || tail[2].Text != "stop" {
		t.Errorf("tail %+v", tail)
	}
}

// Items 13, 20 and 22: a preflight taken while serving is re-run after the stop; within the
// reuse span the RAM side is re-read.
func TestPreflightReuseAndStale(t *testing.T) {
	t0 := time.Unix(5000, 0)
	var none *preEntry
	if !none.stale(t0, false) {
		t.Error("no entry must run")
	}
	e := &preEntry{at: t0, serving: true}
	if e.stale(t0.Add(time.Second), true) {
		t.Error("fresh, same state: reuse")
	}
	if !e.stale(t0.Add(time.Second), false) {
		t.Error("taken while serving, server stopped since: re-run")
	}
	if !(&preEntry{at: t0}).stale(t0.Add(time.Second), true) {
		t.Error("taken while stopped, server up since: re-run")
	}
	if !(&preEntry{at: t0}).stale(t0.Add(preflightReuse), false) {
		t.Error("old: re-run")
	}
	// freshen: the remembered need and budget against the RAM now.
	res := &core.PreflightResult{NeedKnown: true, NeedMiB: 30000, NeedBasis: core.NeedPeak, FreeMiB: 7089,
		Config: &core.Config{Budget: core.Budget{MB: 12288}}}
	res.Check = core.EvaluateStart(7089, 30000, core.NeedPeak, 12288)
	f := freshen(res, 20000, 0)
	if f.Check.Kind != core.CheckTight || f.FreeMiB != 20000 || f.LeftMiB != -10000 || res.FreeMiB != 7089 {
		t.Errorf("freshen %+v", f)
	}
	f = freshen(res, 20000, 15000) // while serving, the server's share counts for a start after its stop
	if f.Check.Kind != core.CheckRoom || f.FreeForStartMiB != 35000 || f.Verdict != core.MemFits {
		t.Errorf("serving freshen %+v", f.Check)
	}
	if freshen(nil, 1, 1) != nil {
		t.Error("nil")
	}
	if preKey("m", ui.Settings{"b": "2", "a": "1"}) != "m|a=1|b=2" {
		t.Error("key order")
	}
}

// H12 after a stop: the server's whole-answer average, the slowest reply, TTFT median and max.
func TestRunSummary(t *testing.T) {
	tm := core.TimingSummary{Requests: 3, AnswerTokens: 300, AnswerTime: 30 * time.Second, PromptTokens: 200, PromptTime: 20 * time.Second,
		Done: []core.RequestTiming{{DecodeTokPerSec: 12, TTFT: 2 * time.Second}, {DecodeTokPerSec: 8, TTFT: 6 * time.Second},
			{DecodeTokPerSec: 10, TTFT: 4 * time.Second}}}
	r := runSummary("A", time.Unix(0, 0), time.Unix(60, 0), tm)
	near(t, "avg", r.DecodeAvg, 10)
	near(t, "low", r.DecodeLow, 8)
	near(t, "prefill", r.PrefillAvg, 10)
	near(t, "ttft median", r.TTFTMedian, 4)
	near(t, "ttft max", r.TTFTMax, 6)
	near(t, "requests", r.Requests, 3)
	r = runSummary("A", time.Unix(0, 0), time.Unix(60, 0), core.TimingSummary{})
	if r.DecodeAvg.State != ui.Unmeasured || r.TTFTMedian.State != ui.Unmeasured {
		t.Errorf("empty %+v", r)
	}
}

// Every control outside the slice answers not ported (copy, open folder and show file are core's).
func TestActionsNotPorted(t *testing.T) {
	b := &Backend{}
	for _, id := range []ui.ActionID{ui.ActRedoSetup, ui.ActSavePreset, ui.ActOtherPort} {
		if _, err := b.Action(id, ""); !errors.Is(err, ui.ErrNotPorted) {
			t.Errorf("%s: %v", id, err)
		}
	}
}

// 3 Memory: the PC's RAM with this server inside «in use», VRAM use and this server apart.
func TestMemoryMapping(t *testing.T) {
	pc := core.PCMemory{TotalMiB: 32768, AvailMiB: 7168, UsedMiB: 25600, ServerMiB: 19456, OthersMiB: 6144, CommitLimitMiB: 40000, CommitAvailMiB: 10000}
	r := memoryFrom(pc, true, 12288, core.VRAMInfo{OK: true, TotalMiB: 16303, UsedMiB: 13722}, serverGPU{bytes: 13 << 30, ok: true})
	near(t, "ready now", r.ReadyNowGB, 7)
	near(t, "model", r.ModelGB, 19)
	near(t, "apps", r.AppsGB, 6)
	near(t, "used vram", r.UsedVRAMGB, 13722.0/1024)
	near(t, "server vram", r.ModelVRAMGB, 13)
	near(t, "other vram", r.OtherVRAMGB, 13722.0/1024-13)
	near(t, "budget", r.BudgetGB, 12)
	if r.CommitSafeGB.State != ui.Unmeasured || len(r.Apps) != 1 {
		t.Errorf("%+v", r)
	}
}

func TestStartMapping(t *testing.T) {
	tr := newStartTrack("gpt")
	t0 := time.Unix(1000, 0)
	ev := tr.apply(core.StartEvent{Step: core.StepCatalog, State: core.StepRunning}, t0)
	// FIX2-WIRE: the launcher file check first, the warmup and the check run after ready.
	if ev.Steps[1].State != ui.StepRunning || ev.Steps[2].State != ui.StepTodo || len(ev.Steps) != 12 || ev.LastS.State != ui.Unmeasured ||
		ev.Steps[0].Name != "bundle check" || ev.Steps[10].Name != "warm up" || ev.Steps[11].Name != "check run" {
		t.Fatalf("first %+v", ev)
	}
	tr.apply(core.StartEvent{Step: core.StepCatalog, State: core.StepDone}, t0.Add(300*time.Millisecond))
	pre := &core.PreflightResult{HasLastReady: true, LastReady: 12 * time.Second, Notices: []string{"virtual notice"},
		Config: &core.Config{Host: "127.0.0.1", Port: 8093}}
	ev = tr.apply(core.StartEvent{Step: core.StepSizing, State: core.StepDone, Preflight: pre}, t0.Add(time.Second))
	if ev.Steps[1].Took != 300*time.Millisecond || ev.LastS.V != 12 || ev.Notices[0] != "virtual notice" ||
		ev.Steps[8].Detail != "llama-server.exe on 127.0.0.1:8093" {
		t.Errorf("after sizing %+v", ev)
	}
	ev = tr.apply(core.StartEvent{Step: core.StepReady, State: core.StepRunning, Elapsed: 5 * time.Second,
		ChildMem: core.ProcessMemory{WorkingSetBytes: 3 << 30}, LastMem: 9 << 30}, t0.Add(6*time.Second))
	if ev.Elapsed != 5*time.Second || ev.Ready {
		t.Errorf("tick %+v", ev)
	}
	near(t, "ram", ev.RAMGB, 3)
	near(t, "last ram", ev.LastRAM, 9)
	ev = tr.apply(core.StartEvent{Final: true, Ready: true, Address: "http://127.0.0.1:8093", Elapsed: 11 * time.Second}, t0.Add(12*time.Second))
	if !ev.Ready || ev.Address != "http://127.0.0.1:8093/v1" || ev.Steps[9].State != ui.StepDone || ev.Steps[11].State != ui.StepSkipped {
		t.Errorf("ready %+v", ev)
	}
	// A failed start: H7 with the PowerShell stage and exit code.
	tr = newStartTrack("gpt")
	tr.apply(core.StartEvent{Step: core.StepArtifacts, State: core.StepRunning}, t0)
	ev = tr.apply(core.StartEvent{Step: core.StepArtifacts, State: core.StepFailed, Detail: "x"}, t0)
	if ev.Steps[3].State != ui.StepFailed {
		t.Errorf("failed step %+v", ev.Steps[3])
	}
	fev := core.StartEvent{Final: true, Status: core.StatusFailRepack, ExitCode: 3, Err: core.NotPorted(core.FamilyRepack), ErrLog: "e.log"}
	f := failureFrom(fev, t0, []string{"tail"})
	if f.Stage != "fail_repack" || f.ExitCode != 3 || f.Why != "repack output missing" || f.LogFile != "e.log" || f.What == "" {
		t.Errorf("failure %+v", f)
	}
}

func TestStopMapping(t *testing.T) {
	tr := newStopTrack(ui.StopUnload)
	if tr.steps[0].State != ui.StepTodo { // GL-15: the save is ported; it waits for the stop's report
		t.Error("the cache save step must wait for the stop's report")
	}
	tr.apply(core.StopStepCtrlBreak)
	tr.apply(core.StopStepGrace)
	if tr.steps[1].State != ui.StepDone || tr.steps[2].State != ui.StepRunning {
		t.Errorf("steps %+v", tr.steps)
	}
	tr.apply(core.StopStepListener)
	code := uint32(0)
	tr.finish(&core.TeardownResult{Graceful: true, ListenerGone: true, ExitCode: &code})
	if tr.steps[3].State != ui.StepSkipped || tr.steps[4].State != ui.StepDone || !strings.Contains(tr.steps[2].Detail, "0 (0x00000000)") {
		t.Errorf("graceful %+v", tr.steps)
	}
	tr = newStopTrack(ui.StopQuit)
	tr.finish(&core.TeardownResult{TaskkillUsed: true, ListenerGone: false})
	if tr.steps[3].State != ui.StepDone || tr.steps[4].State != ui.StepFailed {
		t.Errorf("forced %+v", tr.steps)
	}
}

// Every step name fits its ui name column (the first rehearsal found H15 rows overrunning).
func TestStepNamesFit(t *testing.T) {
	for k, v := range stepLabels {
		if n := len([]rune(v)); n > startNameW {
			t.Errorf("start step %s: %q is %d cells, column %d", k, v, n, startNameW)
		}
	}
	for _, x := range stopSteps {
		if n := len([]rune(x.label)); n > stopNameW {
			t.Errorf("stop step %q is %d cells, column %d", x.label, n, stopNameW)
		}
	}
}

// Live start 1 (2026-10-10 05:50): /slots did not answer during a cold prefill; the phase comes
// from the server log, and the request ends with its eval-time line or the slot's release.
func TestBusyServerPhase(t *testing.T) {
	spawn := time.Date(2026, 10, 9, 20, 49, 54, 0, time.UTC)
	lines := core.ParseServerLines([]string{
		"0.02.947.778 I srv  llama_server: listening on http://127.0.0.1:8093",
		"0.10.029.663 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1",
		"0.10.030.079 I slot launch_slot_: id  0 | task 8 | processing task, is_child = 0",
		"0.16.062.558 W srv          stop: cancel task, id_task = 10",
	}, spawn)
	if !requestInFlight(lines) {
		t.Fatal("task 8 is in flight")
	}
	st := core.ServeStatus{Alive: true, Health: core.HTTPRead{OK: true, Code: 200}, Slots: core.HTTPRead{Err: "context deadline exceeded"}}
	var w decodeWindow
	if s := statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &w, spawn, true); s.Phase != ui.ReqPrefill {
		t.Errorf("busy, in flight: phase %v", s.Phase)
	}
	if s := statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &w, spawn, false); s.Phase != ui.ReqNone {
		t.Errorf("busy, nothing in flight: phase %v", s.Phase)
	}
	st.Timing = core.TimingSummary{Decoding: true, DecodeTokens: 12}
	if s := statusFrom(st, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, &w, spawn, true); s.Phase != ui.ReqDecode || s.DecodedTokens.V != 12 {
		t.Errorf("decoding from the log: %+v", s)
	}
	done := core.ParseServerLines([]string{
		"0.10.030.079 I slot launch_slot_: id  0 | task 8 | processing task, is_child = 0",
		"0.40.000.000 I slot print_timing: id  0 | task 8 | prompt eval time =   14333.53 ms /    92 tokens (  155.80 ms per token,     6.42 tokens per second)",
		"0.40.000.001 I slot print_timing: id  0 | task 8 |        eval time =   9000.00 ms /    64 tokens (  140.62 ms per token,     7.11 tokens per second)",
	}, spawn)
	if requestInFlight(done) {
		t.Error("task 8 finished")
	}
	if requestInFlight(append(done, core.ParseServerLines([]string{"0.41.000.000 I slot launch_slot_: id  0 | task 9 | processing task, is_child = 0"}, spawn)...)) == false {
		t.Error("task 9 in flight")
	}
	released := core.ParseServerLines([]string{
		"0.10.030.079 I slot launch_slot_: id  0 | task 8 | processing task, is_child = 0",
		"0.12.000.000 I slot      release: id  0 | task 8 | stop processing: n_tokens = 130, truncated = 0",
	}, spawn)
	if requestInFlight(released) {
		t.Error("task 8 released")
	}
}
