// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// The mapping from internal/core values to the ui.Backend types. Every function here is pure
// (no OS, no clock but the one passed in), so the tests cover the mapping without a server.
// Review items (reviews/_gol_review_8_user_20261010.md) are named where a mapping serves one.

package wire

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

const mib = 1024.0 // MiB per GiB: the screens show G

func gb(miB int64) ui.Val { return ui.K(float64(miB) / mib) }

func gbBytes(b int64) ui.Val { return ui.K(float64(b) / (1 << 30)) }

// shortSha is the identity text of the design's H1 "id sha256 3f9a…c21e".
func shortSha(s string) string {
	if len(s) < 8 {
		return s
	}
	return s[:4] + "…" + s[len(s)-4:]
}

// identityText joins the pin verdict and the first shard's digest from the caches.
func identityText(f core.ModelFile) string {
	if !f.Catalog {
		return ""
	}
	if len(f.Sha256) > 0 && f.Sha256[0] != "" {
		return f.Identity + " · sha256 " + shortSha(f.Sha256[0])
	}
	return f.Identity
}

// fileCheckOf is the «file check» row (item 10).
func fileCheckOf(f core.ModelFile) ui.FileCheck {
	if !f.Catalog {
		return ui.FileNoReference
	}
	switch f.Identity {
	case core.SelectionPinned:
		return ui.FileSame
	case core.SelectionMismatch:
		return ui.FileDiffers
	case "unpinned":
		return ui.FileNoReference
	case "not hashed":
		return ui.FileNotChecked
	}
	return ui.FileCheckNotPorted
}

// needLabel is the shown need's label (S8-LA-13; F-BU B.4 Label): the last measured value is
// «measured», a structure not measured «predicted before run» (UX-R10, the user's finding
// 16 (e): core's «unverified prediction» read as «not to be trusted»; the words only, the value
// and its rule are core's); the projection of a measured
// setting carries none, and «unmeasured» is the value's own word.
func needLabel(l core.NeedLabel) string {
	switch l {
	case core.NeedMeasured:
		return "measured"
	case core.NeedUnverified:
		return "predicted before run"
	}
	return ""
}

// listsInto fills the memory figures of U-21 (L:4277; F-BU G): the modified, standby and free
// lists (PDH, LFS:302), apps without the modified memory, and the «left» figures by design
// decision ⑧ (free now − need; free now + modified − need), U when the need is unmeasured.
// The screen shows them only when the verdict is tight or cannot start.
func listsInto(p *ui.Preflight, res *core.PreflightResult) {
	avail := res.FreeForStartMiB
	if avail == 0 {
		avail = res.RAM.AvailMiB
	}
	if res.ListsOK && res.RAM.TotalMiB > 0 && res.FreeMiB > 0 {
		p.ModifiedGB, p.FreeGB = gb(res.Lists.ModifiedMiB), gb(res.Lists.FreeZeroMiB)
		p.StandbyGB = gb(max(0, res.FreeMiB-res.Lists.FreeZeroMiB))
		p.AppsGB = gb(max(0, res.RAM.TotalMiB-res.FreeMiB-res.ServerMiB-res.Lists.ModifiedMiB))
	}
	label := core.NeedUnmeasured
	if res.NeedKnown {
		label = core.NeedMeasured
	}
	ml := core.MemoryLine{FreeNow: avail, Modified: res.Lists.ModifiedMiB, ModifiedOK: res.ListsOK, Need: res.NeedMiB,
		NeedLabel: label}
	p.LeftIfWrittenGB = ui.U
	if v, ok := ml.LeftIfWrittenFirst(); ok && avail > 0 {
		p.LeftIfWrittenGB = gb(v)
	}
}

// needVal is item 13's need as a screen value.
func needVal(miB int64, basis core.NeedBasis) (ui.Val, bool) {
	if basis == core.NeedNotMeasured || miB <= 0 {
		return ui.U, false
	}
	return gb(miB), true
}

// fitOf is the list mark (items 11, 12, 22): broken and unsupported first; then the memory
// verdict of the need against the RAM free now. A need above the whole PC's RAM is too big; a
// need above the free RAM is tight, never refused (PowerShell refuses on the budget only).
func fitOf(f core.ModelFile, need ui.Val, known bool, freeMiB, totalMiB int64, memOK bool) (ui.Fit, ui.Val) {
	switch f.State {
	case core.FileBroken:
		return ui.FitBroken, ui.U
	case core.FileUnsupported:
		return ui.FitUnsupported, ui.U
	}
	if !known || !memOK {
		return ui.FitUnmeasured, ui.U
	}
	left := ui.K(float64(freeMiB)/mib - need.V)
	switch {
	case totalMiB > 0 && f.NeedMiB > totalMiB:
		return ui.FitTooBig, left
	case left.V >= 0:
		return ui.FitRoom, left
	}
	return ui.FitTight, left
}

// setupOf is the SETUP tile (items 31 and conflict 9): a catalog file needs its repack output;
// a deleted output reads «needs repack», and ⏎ opens the setup plan (FIX2-WIRE). A setup that
// was cut off reads «cut off» and opens the cut-off screen (H4; GL-10, S8-LA-34): ctrl+r shows
// the setup plan again (the half-made files go after its ⏎), ctrl+d removes them only. A model
// outside the catalog with a structure template has the template setup (GL-10, conflict 14).
//
// mode is the repack mode the model starts with (its saved preset, else the catalog default;
// "" = packed): the setup is ready when that mode's output is present (item 31: a deleted
// repack output reads «needs repack» even while the other mode's output is still there).
func setupOf(f core.ModelFile, mode string) (ui.SetupState, string) {
	switch {
	case f.SetupCut:
		return ui.SetupCutOff, "the setup was cut off · it starts again from the beginning"
	case !f.Catalog && f.Template && f.Repacked:
		return ui.SetupReady, templateNote(f)
	case !f.Catalog && f.Template:
		return ui.SetupNeedsRepack, templateNote(f)
	case !f.Catalog:
		return ui.SetupNotPorted, ""
	}
	if mode != "virtual" {
		mode = "packed"
	}
	other := map[string]string{"packed": "virtual", "virtual": "packed"}[mode]
	has := map[string]bool{"packed": f.Repacked, "virtual": f.RepackedVirtual}
	switch {
	case has[mode] && has[other]:
		return ui.SetupReady, "packed and virtual repack present"
	case has[mode]:
		return ui.SetupReady, mode + " repack present"
	case has[other]:
		return ui.SetupNeedsRepack, mode + " repack output missing · " + other + " present"
	}
	return ui.SetupNeedsRepack, "repack output missing"
}

// dateOf is the date of an RFC 3339 time ("" when it does not parse).
func dateOf(at string) string {
	t, err := time.Parse(time.RFC3339, at)
	if err != nil {
		return ""
	}
	return t.Local().Format("2006-01-02")
}

// catalogFrom is 1 Models (H1, H3) from ListModels: one row per GGUF in the model folders
// (items 11, 12; the ID is the file path, unique per row), the catalog mark, the file check,
// the need and its verdict against the RAM free now (item 13), the last run's decode average
// and date (item 28) and the setup state (item 31).
func catalogFrom(ml *core.ModelList, catalogPath string, freeMiB int64, memOK bool, vram core.VRAMInfo, modes map[string]string) ui.Catalog {
	c := ui.Catalog{Source: catalogPath}
	free := ui.Val{}
	if memOK {
		free = gb(freeMiB)
	}
	tooBig := 0
	for _, f := range ml.Files {
		row := ui.ModelRow{ID: f.Path, Name: f.DisplayName, Present: true, Catalog: f.Catalog, Path: f.Path,
			Folder: filepath.Dir(f.Path), Quant: f.Quant, Identity: identityText(f), FileCheck: fileCheckOf(f),
			FreeRAMGB: free, NeedGB: ui.U, LeftGB: ui.U, DecodeTokS: ui.U,
			Template: f.Template && !f.Catalog} // conflict 14: the template is always on
		if row.Template {
			row.Arch = f.Arch // UX-R10 (finding 18): «outside the catalog · gpt-oss compatible · runs»
		}
		if f.Bytes > 0 {
			row.SizeGB = gbBytes(f.Bytes)
		}
		need, known := needVal(f.NeedMiB, f.NeedBasis)
		row.NeedGB = need
		row.Fit, row.LeftGB = fitOf(f, need, known, freeMiB, ml.TotalRAMMiB, memOK)
		if row.Fit == ui.FitTooBig {
			tooBig++
		}
		if vram.OK {
			row.VRAMTotalGB = gb(vram.TotalMiB)
			if vram.UsedMiB >= 0 {
				row.VRAMUseGB = gb(vram.UsedMiB)
			} else {
				row.VRAMUseGB = gb(vram.TotalMiB - vram.FreeMiB)
			}
		}
		row.Setup, row.SetupNote = setupOf(f, modes[f.Path])
		if f.SetupCut {
			row.Partial = partialFrom(f)
		}
		if f.State != core.FileUsable && f.Reason != "" && !f.SetupCut {
			row.SetupNote = f.Reason
		}
		if f.LastRun != nil {
			row.LastRun = fmt.Sprintf("%s · ready in %.1f s", dateOf(f.LastRun.At), f.LastRun.ReadyDuration().Seconds())
			if f.LastRun.Start != nil && f.LastRun.Start.BudgetMB > 0 {
				row.CacheGB = gb(f.LastRun.Start.BudgetMB)
			}
		}
		if f.Stats != nil && f.Stats.DecodeTokPerSec > 0 {
			row.DecodeTokS, row.MeasuredOn = ui.K(f.Stats.DecodeTokPerSec), dateOf(f.Stats.At)
		}
		c.Models = append(c.Models, row)
	}
	c.TooBig = ui.K(float64(tooBig))
	for _, root := range ml.Roots {
		fo := ui.Folder{Path: root, State: ui.FolderWatching, Note: "re-read every 5 s on 1 Models (names, sizes, times)"}
		var bytes int64
		for _, f := range ml.Files {
			if strings.EqualFold(f.Folder, root) {
				fo.Found++
				if f.Bytes > 0 {
					bytes += f.Bytes
				}
				if f.State == core.FileUnsupported {
					fo.Skipped = append(fo.Skipped, f.Name+"  "+f.Reason)
				}
			}
		}
		if fo.Found > 0 {
			fo.SizeGB = gbBytes(bytes)
		} else {
			fo.State = ui.FolderEmpty
		}
		c.Folders = append(c.Folders, fo)
	}
	return c
}

// --- settings (items 2, 3, 30) ---

// settingLabels name the allowlist keys on the screen.
// (FIX2-WIRE: the ui table has each of them in Korean.)
var settingLabels = map[string]string{
	"budget_mb": "RAM budget", "ctx": "context", "qd": "queue (QD)", "repack_mode": "repack",
	"threads": "threads", "port": "port", "prefetch": "prefetch",
	"prefetch_k": "prefetch K", "prefetch_n": "prefetch N", // GL-19 (F-PF §2.2)
	"warmup": "warmup", "warmstart": "warm start", "autosave": "cache autosave",
	"kv_type": "KV type", "verify_gates": "format check", "parallel": "parallel",
}

// uiKey is the screen's key of a setting: the repack mode is the row "mode" (FIX2-UI contract).
func uiKey(k string) string {
	if k == "repack_mode" {
		return "mode"
	}
	return k
}

// coreKey is the request's key of a screen key.
func coreKey(k string) string {
	if k == "mode" {
		return "repack_mode"
	}
	return k
}

var settingHints = map[string]string{
	"budget_mb":   "expert cache only · auto = the PowerShell autotune",
	"ctx":         "how much text one chat can hold",
	"qd":          "auto = the sweep binding, else QD 1 · virtual: default QD 8, below the wave width cannot work",
	"repack_mode": "packed or virtual · a mode without its files opens the setup plan",
	"warmup":      "a short warmup request right after ready · on by default",
	"prefetch":    "catalog or init · a set K and N pair is the run's pair",
	"prefetch_k":  "manual prefetch K, set with N · auto = no pair · not a validated pair: unmeasured",
	"prefetch_n":  "manual prefetch N, set with K · on while QD is at least N+1, else off with its reason",
}

// UX-R6: what bounds a row's range (its yellow numbers), in the screen's words: a catalog
// profile's allowlist_bounds, or the structure template of a model outside the catalog.
const (
	basisCatalog  = "range = this model's catalog limits"
	basisTemplate = "range = structure template limits"
)

// shown on their own row; the rest fold into "more"
var settingVisible = map[string]bool{"budget_mb": true, "ctx": true, "qd": true, "repack_mode": true}

// settingCandidates are the values ←→ cycles through, cut to the profile's bounds.
var settingCandidates = map[string][]int64{
	"budget_mb": {4096, 6144, 8192, 10240, 12288, 16384, 24576},
	"ctx":       {4096, 8192, 12288, 16384, 32768, 65536},
	"qd":        {1, 2, 4, 8, 16},
	"threads":   {4, 6, 8, 12, 16},
	// GL-19: the manual pair (F-PF §2.2); any value in the on range can also be typed
	"prefetch_k": {1, 2, 4, 8, 16},
	"prefetch_n": {1, 2, 3, 4},
}

// gText is a MiB amount in the screen's GB form: "12 GB", "14.5 GB" (UX-R7).
func gText(miB int64) string {
	return strconv.FormatFloat(math.Round(float64(miB)/mib*10)/10, 'f', -1, 64) + " GB" // UX-R7
}

// toScreen is a request value in the screen's form: the budget in G (item 30's box reads
// «budget [ 14.5_ ] G»), every other value as it is.
func toScreen(key, v string) string {
	if key != "budget_mb" {
		return v
	}
	if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil {
		return gText(n)
	}
	return v
}

// toRequest is a screen value in the request's form: the budget back to MB.
func toRequest(key, v string) string {
	v = strings.TrimSpace(v)
	if key != "budget_mb" || v == "" || strings.EqualFold(v, "auto") {
		return v
	}
	t := strings.ToLower(v) // UX-R7: «14.5 GB», «14.5 G» and «14.5» alike
	t = strings.TrimSpace(strings.TrimSuffix(strings.TrimSuffix(t, "b"), "g"))
	if f, err := strconv.ParseFloat(t, 64); err == nil && t != v {
		return strconv.FormatInt(int64(math.Round(f*mib)), 10)
	}
	return v
}

func boundedOptions(r core.SettingRow) []string {
	var opts []string
	seen := map[string]bool{}
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			opts = append(opts, s)
		}
	}
	if r.Default == "auto" {
		add("auto")
	}
	var nums []int64
	if n, err := strconv.ParseInt(r.Default, 10, 64); err == nil {
		nums = append(nums, n)
	}
	switch r.Key {
	case "port":
		if n, err := strconv.ParseInt(r.Default, 10, 64); err == nil {
			nums = append(nums, n+1, n+2)
		}
	default:
		nums = append(nums, settingCandidates[r.Key]...)
	}
	if r.Key == "ctx" && r.Bounded {
		// UX-R6: the steps run on to the row's own maximum (the allowlist bound, e.g. 128k);
		// the step list above ends at 64k, a value with no recorded reason (GOL-WIRE 9ab4e535a)
		for n := int64(131072); n < r.Max; n *= 2 {
			nums = append(nums, n)
		}
		nums = append(nums, r.Max)
	}
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })
	for _, n := range nums {
		if r.Bounded && (n < r.Min || n > r.Max) {
			continue
		}
		add(toScreen(r.Key, strconv.FormatInt(n, 10)))
	}
	return opts
}

// onRangeText is the on range of N: Min ~ QD-1 with the QD the screen holds, «QD-1» while
// the launcher still decides the QD (auto).
func onRangeText(r core.SettingRow, qd string) string {
	if q, err := strconv.ParseInt(strings.TrimSpace(qd), 10, 64); err == nil {
		if q-1 < r.Min {
			return fmt.Sprintf("off at QD %d", q)
		}
		return fmt.Sprintf("%d ~ %d", r.Min, q-1)
	}
	return fmt.Sprintf("%d ~ QD-1", r.Min)
}

// rangeText is the range drawn after the typed-value box (item 30), in the screen's unit.
func rangeText(r core.SettingRow) string {
	if !r.Bounded {
		return ""
	}
	switch r.Key {
	case "budget_mb":
		return strings.TrimSuffix(gText(r.Min), " GB") + " GB ~ " + gText(r.Max)
	case "ctx":
		k := func(n int64) string {
			if n%1024 == 0 {
				return strconv.FormatInt(n/1024, 10) + "k"
			}
			return strconv.FormatInt(n, 10)
		}
		return k(r.Min) + " ~ " + k(r.Max)
	}
	return fmt.Sprintf("%d ~ %d", r.Min, r.Max)
}

// settingsFrom is the settings panel of 1 Models from core.Settings (the allowlist, conflict 2):
// running is the last actual start's values (item 2), saved the model's preset values when
// they apply to this file (item 3; they read as the defaults).
func settingsFrom(model string, rows []core.SettingRow, running, saved map[string]string, presetNote string) ui.SettingsView {
	v := ui.SettingsView{Model: model, Preset: "catalog defaults", Saved: saved != nil}
	if presetNote != "" {
		v.Preset = presetNote
	}
	if running != nil {
		v.LastRun = ui.Settings{}
		for k, x := range running {
			v.LastRun[uiKey(k)] = toScreen(k, x)
		}
	}
	qdNow := "" // the QD the screen holds, for N's on range (F-PF §6; S8-PF-16)
	for _, r := range rows {
		value := r.Default
		if s, ok := saved[r.Key]; ok {
			value = s
		}
		if r.Key == "qd" {
			qdNow = value
		}
		sr := ui.SettingRow{Key: uiKey(r.Key), Label: settingLabels[r.Key], Value: toScreen(r.Key, value), Editable: r.Editable,
			Locked: r.Locked, Hint: settingHints[r.Key], Folded: !settingVisible[r.Key], Default: toScreen(r.Key, r.Default)}
		if sr.Label == "" {
			sr.Label = r.Key
		}
		if run, ok := running[r.Key]; ok {
			sr.Running = toScreen(r.Key, run)
		}
		if r.Editable {
			if len(r.Options) > 0 {
				sr.Options = append([]string(nil), r.Options...)
			} else {
				sr.Options = boundedOptions(r)
			}
			if !contains(sr.Options, sr.Value) && sr.Value != "" {
				sr.Options = append([]string{sr.Value}, sr.Options...)
			}
			sr.Typed = contains(core.TypedKeys, r.Key)
			if r.Key == "budget_mb" {
				sr.Unit = "GB"
			}
			sr.Range = rangeText(r)
		}
		if r.Bounded {
			sr.Note = rangeText(r)     // in the screen's units (the budget in G), dry run 2026-10-10
			if r.Key != "prefetch_k" { // K's range is F-PF's 1..min(16, n_expert), not the profile's
				sr.Basis = basisCatalog
			}
		}
		if r.OnRange { // N: on in 1..QD-1, accepted above it and off (F-PF §6; S8-PF-16)
			sr.Note = onRangeText(r, qdNow)
			if r.Editable {
				sr.Range = sr.Note
			}
		}
		v.Rows = append(v.Rows, sr)
	}
	return v
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// requestFrom turns the screen's settings into a core request: "auto" leaves the launcher's
// own decision, repack_mode is the launch option, the rest are allowlist keys (the budget back
// in MB).
func requestFrom(path string, s ui.Settings) core.Request {
	req := core.Request{ModelPath: path, Settings: map[string]string{}}
	for k, v := range s {
		k = coreKey(k)
		switch {
		case k == "repack_mode":
			req.RepackMode = v
		case v == "auto" || v == "":
		default:
			req.Settings[k] = toRequest(k, v)
		}
	}
	return req
}

// presetValues is what ctrl+s saves (item 3): every key the screen changed, in request form.
func presetValues(s ui.Settings) map[string]string {
	out := map[string]string{}
	for k, v := range s {
		out[coreKey(k)] = toRequest(coreKey(k), v)
	}
	return out
}

// settingWords is the start's settings in plain words (item 16): "64k context", "12 GB cache"…
func settingWords(st core.StartSettings) []string {
	var w []string
	if st.Ctx > 0 {
		if st.Ctx%1024 == 0 {
			w = append(w, fmt.Sprintf("%dk context", st.Ctx/1024))
		} else {
			w = append(w, fmt.Sprintf("%d context", st.Ctx))
		}
	}
	if st.BudgetMB > 0 {
		w = append(w, gText(st.BudgetMB)+" cache")
	}
	if st.RepackMode != "" {
		w = append(w, st.RepackMode)
	}
	if st.QD > 0 {
		w = append(w, fmt.Sprintf("QD %d", st.QD))
	}
	if st.PrefetchOn {
		w = append(w, "prefetch on")
	} else if st.Prefetch != "" {
		w = append(w, "prefetch off")
	}
	if st.Threads > 0 {
		w = append(w, fmt.Sprintf("%d threads", st.Threads))
	}
	return w
}

// prefetchWords is the server screen's prefetch row (GL-19 echo, GL-21 read-back; F-PF §4.4,
// §6.4, §8.5): after ready the engine's values first, the launcher's candidate after them;
// before ready the candidate alone. Each piece is one word group the screen translates.
func prefetchWords(c *core.PrefetchEcho, e *core.EnginePrefetch) []string {
	var w []string
	if e != nil {
		switch {
		case !e.Read:
			w = append(w, "engine values not read")
		case e.On:
			w = append(w, "engine on", fmt.Sprintf("K %d", e.K), fmt.Sprintf("N %d", e.N))
		default:
			w = append(w, "engine off", e.OffReason)
		}
	}
	if c != nil {
		cand := "candidate off"
		if c.EffectivePrefetch == "on" {
			cand = "candidate on"
		}
		w = append(w, cand)
		switch {
		case c.RequestedK != nil && c.RequestedN != nil:
			w = append(w, fmt.Sprintf("K %d", *c.RequestedK), fmt.Sprintf("N %d", *c.RequestedN))
			if c.PrefetchNCap != nil { // GL-20: N beside the smallest N_cap (F-PF §8.2; U-39 (iii))
				w = append(w, fmt.Sprintf("N_cap %d", *c.PrefetchNCap))
			}
		case c.OffReason != "":
			w = append(w, c.OffReason)
		}
		if c.PrefetchProvenance != "" {
			w = append(w, c.PrefetchProvenance)
		}
	}
	return w
}

// --- preflight and the user wording of reasons (items 13, 22, 32) ---

// pathRe finds a Windows path in a reason (item 32: full paths go to 4 Log, not the screen).
var pathRe = regexp.MustCompile(`\s*\(?[A-Za-z]:\\[^\s)]*\)?`)

// noticeWords gives core's preflight notices in the user's words (item 32); the full text goes
// to 4 Log. A notice not listed shows without its paths; an empty one shows only in 4 Log (it
// asks nothing of the user).
var noticeWords = []struct{ head, words string }{
	{"the catalog pin for this profile does NOT match", "this file differs from the catalog file · catalog settings, prefetch off, no reference speed"},
	{"the stored user preset is not read", ""},
	{"a server started by this launcher is running", ""},
	// GL-20: the virtual notices show their T18 text exactly (S8-PF-08); the literals are the
	// screen's string-table keys (TestS8LA36), held equal to core's (TestS8PF08)
	{core.NoticeVirtualUnvalidated, virtualNoticeWords},
	{core.NoticeNCapZero, nCapZeroWords},
	// UX-R12b (5): core's «the server was stopped during the check run» (start.go stoppedDuring),
	// a key of the screens' table (Korean «시작 후 응답 시험»)
	{stoppedCheckRun, stoppedCheckRun},
	{"prefetch=init ignored", "prefetch «init» ignored: this model's prefetch is fixed by the catalog"},
	{"prefetch_init_v1 is an unvalidated", "prefetch «init» is an unvalidated starting point · no speed claim"},
}

// stoppedCheckRun is core's reason when the server is stopped during the check run.
const stoppedCheckRun = "the server was stopped during the check run"

// The virtual notices' screen words: core's T18 texts (F-PF §7.2, §8.3; OQ-5 (b)).
const (
	virtualNoticeWords = "default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured"
	nCapZeroWords      = "prefetch admits nothing: N_cap is 0 on every layer at this QD (the run starts; raise QD to let prefetch work)"
)

// userNotice is a notice in the user's words, without paths ("" = 4 Log only).
func userNotice(s string) string {
	for _, w := range noticeWords {
		if strings.HasPrefix(s, w.head) {
			return w.words
		}
	}
	return strings.TrimSpace(pathRe.ReplaceAllString(s, ""))
}

// userNotices is the notices the screen shows.
func userNotices(ns []string) []string {
	var out []string
	for _, n := range ns {
		if u := userNotice(n); u != "" {
			out = append(out, u)
		}
	}
	return out
}

// noticeLog is the full text of the notices whose screen form differs ("" when none does).
func noticeLog(res *core.PreflightResult) string {
	if res == nil {
		return ""
	}
	var out []string
	for _, n := range res.Notices {
		if userNotice(n) != n {
			out = append(out, n)
		}
	}
	return strings.Join(out, " | ")
}

// userReason is item 32's form of a refusal: the one reason with its numbers; the screens put
// the verdict before it (1 Models «✕ cannot start · <reason> · details on 3 Memory», item 22;
// 3 Memory «refused: <reason>»; H7 the WHY tile). The full text with its internal coordinates
// is the second value, for 4 Log.
func userReason(err error, res *core.PreflightResult) (string, string) {
	full := refusalText(err)
	const head = ""
	var rn *core.RepackNeededError
	if errors.As(err, &rn) { // conflict 9: ⏎ on 1 Models opens the setup plan
		switch {
		case rn.Partial:
			return "an earlier setup was cut off · ⏎ opens the setup plan", full
		case rn.Stale:
			return "the repack files do not come from this file · ⏎ opens the setup plan", full
		}
		return "no " + rn.Mode + " repack files yet · ⏎ opens the setup plan", full
	}
	var ge *core.GateError
	if errors.As(err, &ge) { // conflict 16: the failed item of the format gate
		return fmt.Sprintf("the %s repack check failed at item %d of %d · ⏎ opens the setup plan", ge.Gate.Mode, ge.Gate.FailedItem, ge.Gate.Items), full
	}
	var np *core.NotPortedError
	if errors.As(err, &np) {
		switch np.Family {
		case core.FamilyRepack:
			return head + "repack output missing", full
		case core.FamilyArchTemplate:
			return head + "outside the catalog: the template setup is not ported yet", full
		}
		name := np.Family
		if i := strings.Index(name, " ("); i > 0 {
			name = name[:i]
		}
		return head + name + " is not ported", full
	}
	le, ok := core.AsLaunchError(err)
	if !ok {
		return head + userNotice(err.Error()), full
	}
	if le.Port != nil { // GL-08, conflict row 6: a busy port shows its owner only (failureFrom)
		return fmt.Sprintf("%sport %d is in use by another program", head, le.Port.Port), full
	}
	switch le.Status {
	case core.StatusFailResource:
		if res != nil && res.Check.Kind == core.CheckBlocked {
			return fmt.Sprintf("%sRAM budget %s > free now %s", head, gText(res.Check.BudgetMiB), gText(res.Check.FreeMiB)), full
		}
		if strings.Contains(strings.ToLower(le.Reason), "disk") {
			return head + "not enough free disk space", full
		}
		return head + "not enough RAM for this budget", full
	case core.StatusFailModelPath:
		return head + modelPathReason(le.Reason), full
	case core.StatusFailInstanceLock:
		return head + LauncherLockLine, full // W7 duty 7 at the place of the start verdict
	case core.StatusFailCustomArgs:
		return head + "a setting is outside its range", full
	case core.StatusFailGateBundle:
		return head + "the launcher files are damaged · install again", full
	case core.StatusFailGateCatalog:
		return head + "the catalog check failed", full
	case core.StatusFailGateVerify:
		return head + "the repack check failed (redo the repack)", full
	}
	if h := core.StatusHint(le.Status); h != "" {
		return head + h, full
	}
	return head + string(le.Status), full
}

// modelPathReason tells the causes of a fail_model_path refusal apart by core's reason (UX-R21,
// item 32; the status and the exit code stay): a structure the launcher does not support (a
// readable header that no catalog profile and no template matches, core select.go
// selectProfile), a missing file (core preflight.go, repack.go), and a file that cannot be
// read (a header that does not parse or names no or conflicting architectures, core gguf.go;
// an incomplete shard set; the shard hashing). Any other model-path refusal keeps the general
// text.
func modelPathReason(reason string) string {
	has := func(ps ...string) bool {
		for _, p := range ps {
			if strings.HasPrefix(reason, p) {
				return true
			}
		}
		return false
	}
	switch {
	case has("unsupported GGUF: no catalog profile matches"):
		return "unsupported model structure"
	case has("model file not found", "empty model path"):
		return "the model file is missing"
	case has("unsupported GGUF: ", "architecture metadata conflicts", "incomplete shard set", "split count 0",
		"source shard hashing failed"):
		return "the model file cannot be read"
	}
	return "the model file is missing or cannot be used"
}

// verdictOf is item 22's verdict.
func verdictOf(k core.StartCheckKind) ui.Verdict {
	switch k {
	case core.CheckRoom:
		return ui.VerdictRoom
	case core.CheckTight:
		return ui.VerdictTight
	case core.CheckBlocked:
		return ui.VerdictCannot
	}
	return ui.VerdictUnmeasured
}

// preflightFrom is H14 (and the H1 numbers once a preflight exists). A refusal is a result,
// not an error: the screen shows it in user words and offers no start (conflict 7, item 32).
// The second value is the full refusal for 4 Log ("" without one).
func preflightFrom(model string, res *core.PreflightResult, err error) (ui.Preflight, string) {
	p := ui.Preflight{Model: model, RoomLineGB: ui.U, CommitSafeGB: ui.U}
	if res != nil {
		p.Notices = userNotices(res.Notices)
		if res.FreeForStartMiB > 0 || res.RAM.AvailMiB > 0 {
			avail := res.FreeForStartMiB
			if avail == 0 {
				avail = res.RAM.AvailMiB
			}
			p.AvailGB, p.FreeGB = gb(avail), gb(res.FreeMiB)
		}
		if res.RAM.TotalMiB > 0 {
			p.RAMTotalGB = gb(res.RAM.TotalMiB)
			if res.FreeMiB > 0 {
				p.AppsGB = gb(res.RAM.TotalMiB - res.FreeMiB - res.ServerMiB)
			}
		}
		if res.TotalBytes > 0 {
			p.ModelGB = gbBytes(res.TotalBytes)
		}
		if res.Config != nil {
			p.CacheGB = gb(res.Config.Budget.MB)
		}
		if res.VRAM.OK {
			p.VRAMTotalGB = gb(res.VRAM.TotalMiB)
			if res.VRAM.UsedMiB >= 0 {
				p.VRAMUseGB = gb(res.VRAM.UsedMiB)
			} else {
				p.VRAMUseGB = gb(res.VRAM.TotalMiB - res.VRAM.FreeMiB)
			}
		}
		p.NeedGB, p.LeftGB = ui.U, ui.U
		if res.NeedKnown {
			p.NeedGB, p.LeftGB = gb(res.NeedMiB), gb(res.LeftMiB)
		}
		p.NeedLabel = needLabel(res.NeedLabel)
		listsInto(&p, res)
		commitInto(&p, res)
		p.Verdict = verdictOf(res.Check.Kind)
		if res.BudgetCap.OK && res.BudgetCap.GiB > 0 { // conflict 7
			p.BudgetCapGB = ui.K(float64(res.BudgetCap.GiB))
		}
		p.LastStartS = ui.U
		if res.HasLastReady {
			p.LastStartS = ui.K(res.LastReady.Seconds())
		}
		p.SettingsLine = res.SettingsLine
	}
	if err == nil {
		return p, ""
	}
	p.Refused = true
	short, full := userReason(err, res)
	p.Refusals = append(p.Refusals, short)
	if le, ok := core.AsLaunchError(err); ok && le.Status == core.StatusFailResource {
		p.Verdict = ui.VerdictCannot
	}
	return p, full
}

// gbf is gb for a float MiB figure.
func gbf(miB float64) ui.Val { return ui.K(miB / mib) }

// commitInto is F-BU F (GL-06): the commit limit and total now, the room after the start, the
// safe line of 4 GiB, this start's predicted commit with the expert pool on its own line, and
// the near-limit line with its numbers (U-13: shown; nothing is changed and nothing refused).
func commitInto(p *ui.Preflight, res *core.PreflightResult) {
	p.CommitSafeGB = gbf(core.SafeLineMiB)
	p.CommitLimitGB, p.CommitNowGB, p.CommitLeftAfterGB, p.PredictedGB, p.PoolGB = ui.U, ui.U, ui.U, ui.U, ui.U
	if res.CommitOK {
		p.CommitLimitGB, p.CommitNowGB = gbf(res.Commit.LimitMiB), gbf(res.Commit.TotalMiB)
	}
	if !res.PredictionOK {
		return
	}
	p.PredictedGB, p.PoolGB = gbf(res.Prediction.Total()), gbf(res.Prediction.PoolMiB)
	if res.CommitOK {
		p.CommitLeftAfterGB = gbf(res.RoomAfterMiB)
	}
	if res.NearOK {
		n := res.Near
		p.Near = &ui.NearLine{RoomGB: n.Room / mib, SafeGB: n.SafeLine / mib, LimitGB: n.Limit / mib, NowGB: n.Now / mib,
			PredictedGB: n.Predicted / mib, PoolGB: n.Pool / mib}
	}
}

// splitInto is U-20 (i) on the serving page (GL-06; F-BU F.2): the expert cache is T_pool, file
// mapping is not measured, other is the rest of the fit's host terms; the commit safe line is
// 4 GiB.
func splitInto(r *ui.MemoryReading, pred core.CommitPrediction, ok, budgetSet bool) {
	r.CommitSafeGB = gbf(core.SafeLineMiB)
	if ok {
		r.ExpertCacheGB, r.FileMapGB, r.OtherGB = gbf(pred.PoolMiB), ui.U, gbf(pred.HostRemainderMiB)
	} else if budgetSet {
		r.ExpertCacheGB = ui.U // a server runs and its prediction is not measured
	}
}

// refusalText is the PowerShell status and reason, or the not-ported family (4 Log's form).
func refusalText(err error) string {
	if le, ok := core.AsLaunchError(err); ok {
		return fmt.Sprintf("%s (exit %d): %s", le.Status, le.ExitCode(), le.Reason)
	}
	var np *core.NotPortedError
	if errors.As(err, &np) {
		return np.Error()
	}
	return err.Error()
}

// --- the serve page and 3 Memory (items 4, 9, 13, 14, 16, 23, 27) ---

func num(v any) (int64, bool) {
	switch x := v.(type) {
	case float64:
		return int64(x), true
	case int64:
		return x, true
	case int:
		return int64(x), true
	}
	return 0, false
}

func obj(v any) map[string]any { m, _ := v.(map[string]any); return m }

// serverGPU is the server's dedicated GPU memory (the Windows counter), when read.
type serverGPU struct {
	bytes int64
	ok    bool
}

// statusFrom is H8 from core.Status, the OS memory, the GPU reads and the decode window.
//
// inflight is requestInFlight of the server log: while the server is busy its /slots answer can
// take longer than the read allows (live start 1 of FIX-WIRE: a cold prefill), and the phase then
// comes from the log instead of reading «no request in flight».
func statusFrom(st core.ServeStatus, mem core.MemStatus, memOK bool, vram core.VRAMInfo, gpu serverGPU, w *decodeWindow, now time.Time, inflight bool) ui.ServeStatus {
	s := ui.ServeStatus{Model: st.ProfileID, Address: st.Address, Port: st.Port, ChildPID: st.PID, ChildAlive: st.Alive,
		Uptime: st.Uptime, SettingsLine: st.SettingsLine, SettingsWords: settingWords(st.Start), Alias: st.Start.Alias,
		Prefetch: prefetchWords(st.Start.PrefetchEcho, st.Engine)}
	if e := st.Engine; e != nil && e.Read { // after ready the words follow the engine (F-PF §6.4)
		for i, w := range s.SettingsWords {
			if w == "prefetch on" || w == "prefetch off" {
				s.SettingsWords[i] = map[bool]string{true: "prefetch on", false: "prefetch off"}[e.On]
			}
		}
	}
	if st.Address != "" {
		s.Address = st.Address + "/v1"
	}
	switch {
	case st.Health.OK:
		s.Health = "ok"
		if b := obj(st.Health.Body); b != nil {
			if v, ok := b["status"].(string); ok && v != "" {
				s.Health = v
			}
		}
	case st.Health.Code == 503:
		s.Health = "loading"
	case st.Health.Err != "":
		s.Health = "error: " + st.Health.Err
	case st.Health.Code != 0:
		s.Health = fmt.Sprintf("http %d", st.Health.Code)
	case !st.Alive:
		s.Health = "no server"
	}
	if p := obj(st.Props.Body); p != nil {
		if a, ok := p["model_alias"].(string); ok && s.Alias == "" {
			s.Alias = a
		}
		if n, ok := num(obj(p["default_generation_settings"])["n_ctx"]); ok {
			s.CtxTotal = ui.K(float64(n))
		}
	}
	s.Phase = ui.ReqNone
	s.DecodeTokS = ui.U
	slotDecoding := false
	if arr, ok := st.Slots.Body.([]any); ok && len(arr) > 0 {
		sl := obj(arr[0])
		if n, ok := num(sl["n_ctx"]); ok && !s.CtxTotal.Ok() {
			s.CtxTotal = ui.K(float64(n))
		}
		busy, _ := sl["is_processing"].(bool)
		task, hasTask := num(sl["id_task"])
		nt := obj(nil)
		if a, ok := sl["next_token"].([]any); ok && len(a) > 0 {
			nt = obj(a[0])
		}
		dec, hasDec := num(nt["n_decoded"])
		if hasTask {
			pt, _ := num(sl["n_prompt_tokens"])
			pc, _ := num(sl["n_prompt_tokens_cache"])
			s.CtxUsed = ui.K(float64(pt + dec))
			s.ReusedTokens = ui.K(float64(pc))
			s.NewTokens = ui.K(float64(pt - pc))
			s.DecodedTokens = ui.K(float64(dec))
		} else {
			s.CtxUsed = ui.K(0)
		}
		if busy {
			s.Phase = ui.ReqPrefill
			if dec > 0 {
				s.Phase, slotDecoding = ui.ReqDecode, true
			}
		}
		if w != nil {
			// UX-R11: the window starts at the first decoded token, not in the prompt's read
			w.add(now, task, dec, busy && hasTask && hasDec && dec > 0)
			if r, ok := w.rate(now); ok && busy {
				s.DecodeTokS = ui.K(r)
			}
		}
	}
	// print_timing (items 9, 23): prefill progress and rate, TTFT, the last reply's average.
	// UX-R11: the engine prints a prompt progress line once a prompt has taken 3 s but a decode
	// progress line only from 100 decoded tokens on, so after the last prompt progress line the
	// log reads «prefilling» through a short answer; a slot that is decoding wins over it.
	tm := st.Timing
	prefilling := tm.Prefilling && !slotDecoding
	if prefilling {
		s.Phase = ui.ReqPrefill
		s.PrefillDone, s.PrefillTokS = ui.K(float64(tm.PrefillTokens)), ui.K(tm.PrefillTokPerSec)
		if tm.PrefillProgress > 0 {
			s.PrefillTotal = ui.K(math.Round(float64(tm.PrefillTokens) / tm.PrefillProgress))
		}
	}
	s.Requests = ui.K(float64(tm.Requests))
	for _, d := range tm.Done { // GL-18: the per-request samples of the graph switch (memory only)
		s.Reqs = append(s.Reqs, ui.ReqSample{Task: d.Task, At: d.At, PrefillTokS: d.PromptTokPerSec,
			TTFTs: d.TTFT.Seconds(), DecodeTokS: d.DecodeTokPerSec})
	}
	s.LastDecodeTokS, s.TTFTs = ui.U, ui.U
	if last, ok := tm.Last(); ok {
		s.LastDecodeTokS = ui.K(last.DecodeTokPerSec)
		s.TTFTs = ui.K(last.TTFT.Seconds())
		if !prefilling {
			s.PrefillTokS = ui.K(last.PromptTokPerSec)
			s.PrefillDone, s.PrefillTotal = ui.K(float64(last.PromptTokens)), ui.K(float64(last.PromptTokens))
		}
	}
	if avg, ok := tm.RunDecodeAvg(); ok {
		s.UsualTokS = ui.K(avg)
	}
	if _, slotsOK := st.Slots.Body.([]any); !slotsOK && s.Phase == ui.ReqNone {
		switch {
		case tm.Decoding:
			s.Phase, s.DecodedTokens = ui.ReqDecode, ui.K(float64(tm.DecodeTokens))
		case inflight:
			s.Phase = ui.ReqPrefill
		}
	}
	// RAM (items 13, 14): measured at every status; this server's share apart.
	if memOK {
		s.FreeRAMGB = gb(mem.AvailPhysMiB)
		s.TotalRAMGB = gb(mem.TotalPhysMiB)
		if st.Memory.WorkingSetBytes > 0 {
			s.ModelRAMGB = gbBytes(st.Memory.WorkingSetBytes)
			s.OtherRAMGB = ui.K(math.Max(0, s.TotalRAMGB.V-s.FreeRAMGB.V-s.ModelRAMGB.V))
		}
	}
	// VRAM (item 4): the card's use (nvidia-smi) and this server (the GPU process counter), apart.
	if vram.OK {
		s.TotalVRAMGB = gb(vram.TotalMiB)
		s.FreeVRAMGB = gb(vram.FreeMiB)
		if vram.UsedMiB >= 0 {
			s.UsedVRAMGB = gb(vram.UsedMiB)
		}
	}
	if gpu.ok {
		s.ModelVRAMGB = gbBytes(gpu.bytes)
		if s.UsedVRAMGB.Ok() {
			s.OtherVRAMGB = ui.K(math.Max(0, s.UsedVRAMGB.V-s.ModelVRAMGB.V))
		}
	}
	if st.End != nil {
		s.Exited, s.ExitCode = true, st.End.ExitCode
	}
	return s
}

// memoryFrom is H10: the whole PC's RAM (item 13: used, this server inside it, available now)
// and the card's VRAM with this server's share apart (item 4).
func memoryFrom(pc core.PCMemory, memOK bool, budgetMB int64, vram core.VRAMInfo, gpu serverGPU) ui.MemoryReading {
	r := ui.MemoryReading{CommitSafeGB: ui.U}
	if memOK {
		r.TotalRAMGB, r.FreeGB, r.ReadyNowGB = gb(pc.TotalMiB), gb(pc.AvailMiB), gb(pc.AvailMiB)
		r.ModelGB = gb(pc.ServerMiB)
		r.AppsGB = gb(pc.OthersMiB)
		if pc.CommitLimitMiB > 0 {
			r.CommitLimitGB = gb(pc.CommitLimitMiB)
			r.CommitUsedGB = gb(pc.CommitLimitMiB - pc.CommitAvailMiB)
		}
		if pc.ServerMiB > 0 {
			r.Apps = []ui.AppMem{{PID: pc.ServerPID, Name: "llama-server.exe", GB: gb(pc.ServerMiB), Tag: "this model"}}
		}
		if pc.ListsOK { // GL-09: the modified and free lists (PDH; U-21)
			r.ModifiedGB, r.FreeGB = gb(pc.Lists.ModifiedMiB), gb(pc.Lists.FreeZeroMiB)
			r.StandbyGB = gb(max(0, pc.AvailMiB-pc.Lists.FreeZeroMiB))
			r.AppsGB = gb(max(0, pc.OthersMiB-pc.Lists.ModifiedMiB))
		} else { // UX-R16b: without the lists standby is not measured (never zero; FreeGB is all available)
			r.StandbyGB = ui.U
		}
	}
	if budgetMB > 0 {
		r.BudgetGB = gb(budgetMB)
	}
	if vram.OK {
		r.TotalVRAMGB = gb(vram.TotalMiB)
		if vram.UsedMiB >= 0 {
			r.UsedVRAMGB = gb(vram.UsedMiB)
		}
	}
	if gpu.ok {
		r.ModelVRAMGB = gbBytes(gpu.bytes)
		if r.UsedVRAMGB.Ok() {
			r.OtherVRAMGB = ui.K(math.Max(0, r.UsedVRAMGB.V-r.ModelVRAMGB.V))
		}
	}
	return r
}

// --- 4 Log (items 5, 18, 29) ---

// evalDoneRe is the print_timing line that ends a request (its whole answer).
var evalDoneRe = regexp.MustCompile(`print_timing: .*\|\s+eval time =`)

// requestInFlight reads the server log from the end: a slot that started a task after the last
// finished one is a request in flight.
func requestInFlight(lines []core.ServerLine) bool {
	for i := len(lines) - 1; i >= 0; i-- {
		t := lines[i].Text
		switch {
		case strings.Contains(t, "stop processing") || evalDoneRe.MatchString(t):
			return false
		case strings.Contains(t, "launch_slot_") && strings.Contains(t, "processing task"):
			return true
		}
	}
	return false
}

// levelOf maps the core parser's level; the server's «listening» line reads as ready (LogOK).
func levelOf(l core.ServerLine) ui.LogLevel {
	switch l.Level {
	case core.LevelNotice:
		return ui.LogNotice
	case core.LevelWarn:
		return ui.LogWarn
	case core.LevelError:
		return ui.LogError
	}
	if strings.Contains(l.Text, "server is listening") || strings.Contains(l.Text, "listening on http") {
		return ui.LogOK
	}
	return ui.LogInfo
}

// mergeLogs is items 5 and 18's one time-ordered list: the launcher's events and the server's
// lines with the clock times of the core parser (spawn time + the engine's stamp). The order
// is stable, so lines of the same time keep their order, launcher events before server lines.
func mergeLogs(events []ui.LogLine, server []core.ServerLine, max int) []ui.LogLine {
	out := make([]ui.LogLine, 0, len(events)+len(server))
	out = append(out, events...)
	for _, l := range server {
		if strings.TrimSpace(l.Text) == "" {
			continue
		}
		out = append(out, ui.LogLine{At: l.Time, Level: levelOf(l), Text: strings.TrimSpace(l.Text)})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].At.Before(out[j].At) })
	if max > 0 && len(out) > max {
		out = out[len(out)-max:]
	}
	return out
}
