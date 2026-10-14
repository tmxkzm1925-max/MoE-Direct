package core

import (
	"context"
	"fmt"
	"math"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// --- the changeable settings (conflict 2: the allowlist) ---

// GL-16 (F-LS 4.1, 12.4; S8-LA-28): rule 2's allowlist, deny by default. OpenSettings is the
// whole open set in the edition's order: the six settings, prefetch, the pair K and N, the
// repack mode and the prompt-cache cap. Everything else of the catalog argv and env is locked
// (LockedItems). Prefetch and the repack mode only cycle (CycleOnly).
var OpenSettings = []string{"port", "ctx", "threads", "warmup", "budget_mb", "qd",
	"prefetch", "prefetch_k", "prefetch_n", "repack_mode", "prompt_cache_cap"}

// SettingKeys are the open settings a model's request carries as overrides (the per-model
// settings, the argument path, the per-model saved settings): OpenSettings without the repack
// mode (a launch option of its own, Request.RepackMode) and the prompt-cache cap (a launcher
// setting of 5 Settings, the same for every model: Request.PromptCacheCap).
var SettingKeys = []string{"port", "ctx", "threads", "warmup", "budget_mb", "qd", "prefetch", "prefetch_k", "prefetch_n"}

// CycleOnly are the open settings that take no typed value (review item 30).
var CycleOnly = []string{"prefetch", "repack_mode"}

// LockedItems are what rule 2 keeps locked (F-LS 4.1, 12.4 LockedSettings): the KV cache type
// (deferred by the user), the verify and expects gates, -np 1 and the loopback bind.
var LockedItems = []string{"kv-type", "verify-expects", "np-1", "loopback"}

// WarmStartSwitches are the warm-start feature's own switches (F-LS 4.1, 12.4; OQ-6 (a), user):
// outside rule 2's allowlist, set on 5 Settings, applied at the next server start.
var WarmStartSwitches = []string{"warmstart", "autosave"}

// settingsNotPorted are the open keys whose function lies in an atom not built yet, refused
// with their family. Empty since GL-19 built the manual prefetch pair K and N (F-PF).
var settingsNotPorted = map[string]string{}

// PrefetchRequests are the request values a Go surface offers (F-PF §1 T1, §2.2): catalog and
// init. The PowerShell value «adapt» (PS:318) is offered by no Go path and refused as a value
// outside the range (§2.3; S8-PF-09).
var PrefetchRequests = []string{"catalog", "init"}

const (
	portMin             = 1024 // PORT_MIN (PS:1152)
	portMax             = 65535
	warmupFilePrefix    = "file:"
	autosaveMinMinutes  = 1    // KV_AUTOSAVE_MIN_MINUTES (PS:1101)
	autosaveMaxMinutes  = 1440 // KV_AUTOSAVE_MAX_MINUTES (PS:1102)
	prefetchRequestDflt = "catalog"
)

// checkSetting is Test-OverrideValue (PS:6032-6104): the type and bounds check of one value.
func checkSetting(key, value string, bounds map[string]Bound) (string, error) {
	switch key {
	case "warmup":
		raw := strings.TrimLeft(value, " \t\r\n\v\f")
		if len(raw) > len(warmupFilePrefix) && strings.ToLower(raw[:len(warmupFilePrefix)]) == warmupFilePrefix {
			return warmupFilePrefix + raw[len(warmupFilePrefix):], nil
		}
	case "prefetch":
		v := strings.ToLower(strings.TrimSpace(value))
		if contains(PrefetchRequests, v) {
			return v, nil
		}
		return "", fmt.Errorf("prefetch must be one of %s", strings.Join(PrefetchRequests, "|"))
	case "prefetch_k", "prefetch_n": // F-PF §9 C1-C2: the value as the surface delivered it
		return checkPairValue(key, value)
	}
	if key == "warmup" || key == "warmstart" || key == "autosave" {
		v := strings.ToLower(strings.TrimSpace(value))
		if v == "on" || v == "true" || v == "1" {
			return "on", nil
		}
		if v == "off" || v == "false" || v == "0" {
			return "off", nil
		}
		if key == "warmup" {
			return "", fmt.Errorf("warmup must be 'on', 'off' or 'file:<path>'")
		}
		return "", fmt.Errorf("%s must be 'on' or 'off'", key)
	}
	n, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
	if err != nil {
		return "", fmt.Errorf("%s", ReasonNotInteger(key)) // BUP 2.7, PS :6088 (F-BU H.2)
	}
	b, bounded := bounds[key]
	lo, hi := b.Min, b.Max
	if !bounded { // GL-16: a row without bounds (a structure's budget before its plan) takes any count
		lo, hi = 1, math.MaxInt64
	}
	if key == "port" {
		lo, hi = max(lo, portMin), min(hi, portMax)
	}
	if key == "qd" {
		lo, hi = max(lo, EngineQDMin), min(hi, EngineQDMax)
	}
	if n < lo || n > hi {
		return "", fmt.Errorf("%s", ReasonOutOfBounds(key, lo, hi)) // PS :6101
	}
	return strconv.FormatInt(n, 10), nil
}

// settingAllowed is the allowed range of a setting as the refusal texts name it (F-LS 11.4
// <range>): numbers «lo..hi» after the port and QD clamps, else the accepted words.
func settingAllowed(key string, bounds map[string]Bound) string {
	switch key {
	case "warmup":
		return "on, off or file:<path>"
	case "prefetch":
		return strings.Join(PrefetchRequests, ", ")
	case "warmstart", "autosave":
		return "on or off"
	case "prefetch_k":
		return allowedK
	case "prefetch_n":
		return allowedN
	}
	b, ok := bounds[key]
	switch key {
	case "port":
		if !ok {
			b = Bound{Min: portMin, Max: portMax}
		}
		b.Min, b.Max, ok = max(b.Min, portMin), min(b.Max, portMax), true
	case "qd":
		if !ok {
			b = Bound{Min: EngineQDMin, Max: EngineQDMax}
		}
		b.Min, b.Max, ok = max(b.Min, EngineQDMin), min(b.Max, EngineQDMax), true
	}
	if !ok {
		return "a whole number"
	}
	return fmt.Sprintf("%d..%d", b.Min, b.Max)
}

// Notice texts.
const (
	// NoticeVirtualUnvalidated is T18 (F-PF §7.2): on a virtual repack QD, the prefetch request
	// and the manual pair are accepted, and a run off QD 8 with the catalog prefetch shows this
	// notice, not a refusal. The wording is the user's OQ-5 (b) of 2026-10-12 19:48 KST (F-LS
	// 11.7; Korean «기본 값=QD 8+카탈로그 프리페치뿐 · 그 밖=동작 미측정», the screen's string
	// table); it replaces L:4199's «검증된 조합=…» wording (GL-20).
	NoticeVirtualUnvalidated = "default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured"
)

// resolveOverrides checks the request's settings the way Get-CliOverrides does with
// -NonInteractive (an invalid value is fail_custom_args), refuses a key outside the allowlist,
// returns the not-ported error for the keys of an atom not built yet, and applies the virtual
// exception in place of Assert-VirtualCliPins / Assert-VirtualOverridePins (PS:13441-13470).
// path is the surface the values came from (F-LS 5.3, 11.4; S8-LA-29): the refusal names it.
func resolveOverrides(settings map[string]string, bounds map[string]Bound, virtual bool, path CustomValuePath) (map[string]string, []string, error) {
	ov := map[string]string{}
	var notices []string
	keys := make([]string, 0, len(settings))
	for k := range settings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !contains(SettingKeys, k) {
			if path == PathArguments {
				return nil, nil, stop(StatusFailCustomArgs, "%s", CustomValueText(path, k, settings[k],
					"one of the settings "+strings.Join(SettingKeys, ", ")))
			}
			return nil, nil, stop(StatusFailCustomArgs, "'%s' is not a changeable setting (allowlist: %s)", k, strings.Join(SettingKeys, ", "))
		}
	}
	for _, k := range SettingKeys {
		v, ok := settings[k]
		// An empty value is no value, except for K and N: a surface that delivers the key
		// delivers its value, and an empty one is refused (F-PF §9 C1; S8-PF-01).
		if !ok || strings.TrimSpace(v) == "" && !isPairKey(k) {
			continue
		}
		if fam, np := settingsNotPorted[k]; np {
			return nil, nil, notPorted(fam, "setting '%s'", k)
		}
		val, err := checkSetting(k, v, bounds)
		if err != nil {
			text := CustomValueText(path, k, v, settingAllowed(k, bounds))
			if isPairKey(k) { // the bound named in F-PF §2.5's form
				text += " · " + err.Error()
			} else if k == "qd" {
				if n, perr := strconv.ParseInt(strings.TrimSpace(v), 10, 64); perr != nil || n < EngineQDMin || n > EngineQDMax {
					text += " · " + refuseQDText(v) // F-PF §9 C11
				}
			}
			return nil, nil, stop(StatusFailCustomArgs, "%s", text)
		}
		ov[k] = val
	}
	// F-PF §9 C3: K and N are set together or not at all.
	_, hasK := ov["prefetch_k"]
	_, hasN := ov["prefetch_n"]
	if hasK != hasN {
		key := "prefetch_k"
		if hasN {
			key = "prefetch_n"
		}
		return nil, nil, stop(StatusFailCustomArgs, "%s · %s", CustomValueText(path, key, settings[key], allowedPair), refusePairText)
	}
	// GL-20: the virtual notice is decided after the config, on the run's outcome (F-PF §7.2;
	// preflight), not on the values given here.
	_ = virtual
	return ov, notices, nil
}

// --- argv helpers (PS:6106-6145) ---

// argvValue is Get-ArgvValue.
func argvValue(argv []string, flag string) (string, bool) {
	for i := 0; i < len(argv)-1; i++ {
		if argv[i] == flag {
			return argv[i+1], true
		}
	}
	return "", false
}

// setArgvValue is Set-ArgvValue: replace every flag+value pair, or append one.
func setArgvValue(argv []string, flag, value string) []string {
	out := make([]string, 0, len(argv)+2)
	replaced := false
	for i := 0; i < len(argv); i++ {
		if argv[i] == flag && i < len(argv)-1 {
			out = append(out, flag, value)
			i++
			replaced = true
		} else {
			out = append(out, argv[i])
		}
	}
	if !replaced {
		out = append(out, flag, value)
	}
	return out
}

// removeArgvPair is Remove-ArgvPair.
func removeArgvPair(argv []string, flag string) []string {
	out := make([]string, 0, len(argv))
	for i := 0; i < len(argv); {
		if argv[i] == flag {
			i += 2
			continue
		}
		out = append(out, argv[i])
		i++
	}
	return out
}

// envArgHost is llama-server's environment form of --host; the command line wins over it, and
// the launcher never passes it (rule 17; GL-12).
const envArgHost = "LLAMA_ARG_HOST"

// bindLoopback is rule 17 (F-LS 4.7, CF 17; GL-12): the server binds BindHost, whatever the
// catalog row holds and whatever the settings path. Every --host value of the catalog argv
// (the pair form and the --host=value form) must be a loopback address, as PowerShell's
// Test-LoopbackAddress requires (a non-loopback one is the catalog gate's refusal); the first
// becomes the one `--host 127.0.0.1` in its place and the others go, so no later occurrence
// can win; without one the pair is appended.
func bindLoopback(argv []string) ([]string, *LaunchError) {
	out := make([]string, 0, len(argv)+2)
	placed := false
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		v, isHost := "", false
		switch {
		case a == "--host":
			if i+1 >= len(argv) {
				return nil, stop(StatusFailGateCatalog, "catalog defaults.argv ends with --host and no value; loopback-only is locked (RS 7-3)")
			}
			v, isHost = argv[i+1], true
			i++
		case strings.HasPrefix(a, "--host="):
			v, isHost = strings.TrimPrefix(a, "--host="), true
		}
		if !isHost {
			out = append(out, a)
			continue
		}
		if !isLoopback(v) {
			return nil, stop(StatusFailGateCatalog, "catalog defaults.argv binds a non-loopback host (%s); loopback-only is locked (RS 7-3)", v)
		}
		if !placed {
			out = append(out, "--host", BindHost)
			placed = true
		}
	}
	if !placed {
		out = append(out, "--host", BindHost)
	}
	return out, nil
}

// isLoopback is Test-LoopbackAddress (PS:8893-8899); conflict 17.
func isLoopback(addr string) bool {
	if strings.EqualFold(addr, "localhost") {
		return true
	}
	ip := net.ParseIP(addr)
	return ip != nil && ip.IsLoopback()
}

// --- the budget (Resolve-BudgetAutotune PS:3743-3830) ---

const (
	budgetReserveMiB   = 20480 // BUDGET_RESERVE_MIB (PS:3661)
	budgetFloorMiB     = 4096
	budgetLadderMaxMiB = 12288
)

// SlotGeometry is Get-BudgetSlotGeometry (PS:3684-3705).
type SlotGeometry struct {
	NExpert, SlotStrideMax, Layers int64
}

func readSlotGeometry(outputDir string) (SlotGeometry, error) {
	path := filepath.Join(outputDir, "manifest.json")
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		return SlotGeometry{}, fmt.Errorf("manifest.json not present")
	}
	v, err := readJSONFileStrict(path)
	if err != nil {
		return SlotGeometry{}, fmt.Errorf("manifest.json unreadable - %v", err)
	}
	o, _ := asObj(v)
	layout, ok := asObj(o["layout"])
	if !ok {
		return SlotGeometry{}, fmt.Errorf("manifest.json has no layout object")
	}
	mo, _ := asObj(o["model"])
	ne, ok := jNonNegInt(mo["n_expert"])
	if !ok || ne <= 0 {
		return SlotGeometry{}, fmt.Errorf("manifest model.n_expert is missing or not a positive integer")
	}
	stride, ok := jNonNegInt(layout["slot_stride_max"])
	if !ok || stride <= 0 {
		return SlotGeometry{}, fmt.Errorf("manifest layout.slot_stride_max is missing or not a positive integer")
	}
	layers, _ := jArr(layout["layers"])
	if len(layers) == 0 {
		return SlotGeometry{}, fmt.Errorf("manifest layout.layers is absent or empty")
	}
	return SlotGeometry{NExpert: ne, SlotStrideMax: stride, Layers: int64(len(layers))}, nil
}

func ceilMiB(b int64) int64 { return int64(math.Ceil(float64(b) / 1048576.0)) }

// Budget is the autotune's answer.
type Budget struct {
	MB           int64
	Source       string // explicit | probe_failed_fallback | geometry_unavailable_fallback | auto
	Reason       string
	Unmeasured   bool
	MachineCapMB int64
	ModelCapMB   int64
	SupportedMin int64
	InstalledMiB int64
}

// resolveBudget is Resolve-BudgetAutotune without -Repro (the draft has no repro mode).
func resolveBudget(p *Profile, ov map[string]string, installed int64, instErr error, geo SlotGeometry, geoErr error) (Budget, *LaunchError) {
	b := Budget{MB: p.MinBudgetMB, InstalledMiB: installed}
	var calcOK bool
	if instErr == nil && geoErr == nil {
		mc := installed - budgetReserveMiB
		if mc < budgetFloorMiB {
			mc = budgetFloorMiB
		}
		if mc > budgetLadderMaxMiB {
			mc = budgetLadderMaxMiB
		}
		structural := ceilMiB(geo.NExpert * geo.SlotStrideMax)
		supported := max(structural, p.MinBudgetMB)
		modelCap := ceilMiB(geo.NExpert * geo.Layers * geo.SlotStrideMax)
		cand, axis := mc, "machine_cap"
		if modelCap < mc {
			cand, axis = modelCap, "model_cap"
		}
		b.MachineCapMB, b.ModelCapMB, b.SupportedMin = mc, modelCap, supported
		calcOK = cand >= supported
		if v, ok := ov["budget_mb"]; ok {
			n, _ := strconv.ParseInt(v, 10, 64)
			b.MB, b.Source = n, "explicit"
			b.Reason = "explicit budget (CLI / stored preset / interactive custom) outranks the autotune"
		} else if !calcOK {
			return b, stop(StatusFailResource, "%s", StopAutotune(ReasonAutotuneBelowMin(AutotuneBelowMin{ // PS :3881-3884, :3968
				Candidate: cand, SupportedMin: supported, StructuralMin: structural, ProfileMin: p.MinBudgetMB, MachineCap: mc,
				InstalledMiB: installed, ModelCap: modelCap})))
		} else {
			b.MB, b.Source = cand, "auto"
			b.Reason = fmt.Sprintf("installed %d MiB - reserve %d -> machine_cap %d MB, model_cap %d MB, supported_min %d MB; limiting axis %s",
				installed, budgetReserveMiB, mc, modelCap, supported, axis)
		}
	} else if v, ok := ov["budget_mb"]; ok {
		n, _ := strconv.ParseInt(v, 10, 64)
		b.MB, b.Source = n, "explicit"
		b.Reason = "explicit budget (CLI / stored preset / interactive custom) outranks the autotune"
	} else if instErr != nil {
		b.Source, b.Reason = "probe_failed_fallback", "autotune off, installed-RAM probe failed: "+instErr.Error()
	} else {
		b.Source, b.Reason = "geometry_unavailable_fallback", "autotune off, slot geometry unavailable: "+geoErr.Error()
	}
	b.Unmeasured = b.Source == "auto" && b.MB != p.MinBudgetMB
	return b, nil
}

// --- prefetch (Get-PrefetchCatalogAxes PS:7322-7368, Resolve-EffectivePrefetch PS:7393-7510) ---

const (
	prefetchInitNCap = 4  // PREFETCH_INIT_N_CAP (PS:447)
	prefetchInitTMin = 1  // PREFETCH_INIT_T_MIN (PS:448)
	prefetchInitTMax = 16 // PREFETCH_INIT_T_MAX (PS:450)

	armNone    = "none"
	armCatalog = "catalog-fixed"
	armInit    = "init"
)

var prefetchEvidence = []string{"unverified", "observe", "causal-replay", "paired-live"}

type prefetchAxes struct {
	evidence, activation string
	hold                 bool
	k, n                 int64
	tupleOK              bool
	semanticInvalid      bool
	detail               string
}

func catalogPrefetchAxes(p *Profile) prefetchAxes {
	a := prefetchAxes{}
	a.evidence, _ = p.Raw["prefetch_evidence"].(string)
	a.activation, _ = p.Raw["prefetch_activation"].(string)
	if h, ok := p.Raw["prefetch_promotion_hold"].(bool); ok {
		a.hold = h
	}
	pf, _ := asObj(p.Raw["prefetch"])
	if pf != nil {
		k, ok1 := jNonNegInt(pf["k"])
		n, ok2 := jNonNegInt(pf["n"])
		if ok1 && ok2 {
			a.k, a.n, a.tupleOK = k, n, true
		}
	}
	invalid := func(why string) prefetchAxes { a.semanticInvalid, a.detail = true, why; return a }
	if !contains(prefetchEvidence, a.evidence) {
		return invalid("unknown prefetch_evidence: " + a.evidence)
	}
	if a.activation == "opt-in-fixed" || a.activation == "opt-in-adaptive" {
		return invalid("prefetch_activation " + a.activation + " is a runtime value and may not be stored")
	}
	if a.activation != "off" && a.activation != "catalog-fixed" {
		return invalid("unknown prefetch_activation: " + a.activation)
	}
	if a.activation == "catalog-fixed" {
		if a.evidence != "paired-live" {
			return invalid("catalog-fixed requires prefetch_evidence=paired-live, not " + a.evidence)
		}
		if !a.tupleOK {
			return invalid("catalog-fixed requires a valid prefetch {k,n} tuple")
		}
	} else {
		if p.Raw["prefetch"] != nil {
			return invalid("prefetch_activation=off requires prefetch=null (stray tuple)")
		}
		if a.evidence == "paired-live" {
			return invalid("prefetch_evidence=paired-live with prefetch_activation=off is not a closed state")
		}
	}
	return a
}

// Prefetch is the effective prefetch decision (F-PF §1 T14; the echo fields of P4 §3 and the
// edition's §4: the row's evidence and activation, the identity verdict, the init version, the
// source and the notice).
type Prefetch struct {
	On                  bool
	K, N                int64
	Echo                string // "on" or "off(reason=...)"
	OffReason           string
	Provenance          string
	CandidateActivation string
	Warning             string
	Request             string
	QDDemoted           bool
	Evidence            string // the row's prefetch_evidence
	Activation          string // the row's stored prefetch_activation
	Identity            string // the identity verdict (the selection kind)
	InitVersion         string // prefetch_init_v1, "" for a manual pair (PS:8706)
	Source              string // F-PF §4.2: settings_screen, profile, arguments or ""
	Notice              string // F-PF §3.2 3c: NoticeManualUnmeasured, or ""
	// GL-20: N_cap per layer on a virtual repack, an ON candidate only (F-PF §8.2)
	NCapPerLayer []int64
}

func offEcho(reason string) string { return "off(reason=" + reason + ")" }

// optInOf is ConvertTo-PrefetchOptIn (PS:7369-7377).
func optInOf(request string) string {
	switch request {
	case "init":
		return armInit
	case "adapt":
		return "adapt"
	}
	return armNone
}

// resolvePrefetch is Resolve-EffectivePrefetch for the product run mode (no repro, no bench,
// no collect: the draft has none of them, so the adapt demotion never applies), without a
// manual pair.
func resolvePrefetch(p *Profile, probeOK bool, optIn string, effQD int, identity, request string) Prefetch {
	return resolvePrefetchManual(p, probeOK, optIn, effQD, identity, request, manualInput{})
}

// manualInput is what F-PF's manual arm reads beside P4's inputs: the pair, the repack mode
// and the surface the prefetch value came from.
type manualInput struct {
	pair    *ManualPair
	virtual bool
	source  string
}

// resolvePrefetchManual is F-PF §3.2: P4 §2's order with the manual arm as step 3 (first
// match wins). Every decision carries the echo fields of §4.
func resolvePrefetchManual(p *Profile, probeOK bool, optIn string, effQD int, identity, request string, m manualInput) Prefetch {
	if request == "" {
		request = prefetchRequestDflt
	}
	a := catalogPrefetchAxes(p)
	base := func(pf Prefetch) Prefetch {
		pf.Request, pf.Evidence, pf.Activation, pf.Identity, pf.Source = request, a.evidence, a.activation, identity, m.source
		if pf.InitVersion == "" && m.pair == nil {
			pf.InitVersion = PrefetchInitVersion
		}
		return pf
	}
	off := func(r string) Prefetch {
		return base(Prefetch{Echo: offEcho(r), OffReason: r, CandidateActivation: "off"})
	}
	if a.semanticInvalid {
		return off("catalog_semantic_invalid")
	}
	// Step 2 (derived_t_mismatch) is the preflight's, read apart (PS:7514-7521). Step 3: the
	// manual arm, before the catalog-fixed branch (§3.2; §3.3; §5.2).
	if m.pair != nil {
		if !m.virtual && !probeOK { // 3a: the probe is not an input on a virtual repack (§7.5)
			return off("probe_failed")
		}
		pf := base(Prefetch{On: true, Echo: "on", K: m.pair.K, N: m.pair.N, Provenance: ProvEnvOverride,
			CandidateActivation: "opt-in-fixed"})
		if optIn == armInit { // 3b: the pair is the run's pair (OQ-3 (a), D15)
			pf.Warning = WarnInitIgnoredPair
		}
		// 3c: the unmeasured notice, unless it is the validated combination; on a virtual
		// repack §7.2's notice replaces it (the preflight adds that one).
		validated := a.activation == "catalog-fixed" && identity == SelectionPinned && a.tupleOK &&
			a.k == m.pair.K && a.n == m.pair.N
		if !validated && !m.virtual {
			pf.Notice = NoticeManualUnmeasured
		}
		return pf
	}
	if a.activation == "catalog-fixed" {
		if identity != SelectionPinned {
			return off("identity_not_exact")
		}
		if optIn == "adapt" {
			return off("adapt_controller_not_shipped_phase5")
		}
		if !probeOK {
			return off("probe_failed")
		}
		pf := base(Prefetch{On: true, Echo: "on", K: a.k, N: a.n, Provenance: ProvCatalog,
			CandidateActivation: "catalog-fixed"})
		if optIn == armInit {
			pf.Warning = WarnInitIgnoredCatalogFixed
		}
		return pf
	}
	switch optIn {
	case armNone:
		return off("not_opted_in_evidence_" + strings.ReplaceAll(a.evidence, "-", "_"))
	case armInit:
		if a.hold {
			return off("phase4_hold_unresolved")
		}
		t := p.Identify.NExpertUsed
		if t < prefetchInitTMin || t > prefetchInitTMax {
			return off("init_t_out_of_range")
		}
		if !probeOK {
			return off("probe_failed")
		}
		if effQD < 2 {
			return off("qd_below_prefetch_depth")
		}
		n := int64(min(prefetchInitNCap, effQD-1, max(1, effQD/2)))
		return base(Prefetch{On: true, Echo: "on", K: t, N: n, Provenance: ProvInitV1,
			CandidateActivation: "opt-in-fixed", Warning: WarnInitV1})
	case "adapt":
		return off("adapt_controller_not_shipped_phase5")
	}
	return off("catalog_semantic_invalid")
}

// prefetchArm is Resolve-PrefetchArm (PS:7621-7630).
func prefetchArm(p *Profile, optIn, identity string) string {
	d := resolvePrefetch(p, true, optIn, 2, identity, "")
	if !d.On {
		return armNone
	}
	if d.CandidateActivation == "catalog-fixed" {
		return armCatalog
	}
	return armInit
}

// prefetchForQD is Resolve-PrefetchForQd (PS:7643-7667).
func prefetchForQD(d Prefetch, effQD int) Prefetch {
	if !d.On || int64(effQD) >= d.N+1 {
		return d
	}
	// F-PF §6.1 [A-QDN]: accepted, prefetch off with its reason; the echo keeps the row's
	// fields and the source, drops the pair, the label and the notice.
	return Prefetch{Echo: offEcho("qd_below_prefetch_depth"), OffReason: "qd_below_prefetch_depth",
		CandidateActivation: "off", Request: d.Request, QDDemoted: true, Evidence: d.Evidence,
		Activation: d.Activation, Identity: d.Identity, InitVersion: d.InitVersion, Source: d.Source,
		Warning: d.Warning}
}

// --- the QD (Resolve-QdSweep PS:4778-4912, binding read only) ---

const (
	sweepAlgorithm  = "qd-sweep-v1"          // SWEEP_PROBE_ALGORITHM (PS:1191)
	sweepMethod     = "engine-overlapped-v1" // SWEEP_MEASUREMENT_METHOD (PS:1192)
	sweepStateFile  = "probe.state.json"     // SWEEP_STATE_FILE (PS:1148)
	sweepStateVer   = 2                      // SWEEP_STATE_VERSION (PS:1149)
	QDDegraded      = 1                      // QD_DEGRADED (PS:1166)
	VirtualPinnedQD = 8                      // VIRTUAL_PINNED_QD (PS:765)
)

var sweepPoints = []int{1, 2, 4, 8} // SWEEP_QD_POINTS (PS:1193)

// Sweep is the QD source of a start.
type Sweep struct {
	OK          bool
	QD          int
	QDSource    string // measured-sweep | conservative-default | virtual-pinned
	Reason      string
	Detail      string
	FromBinding bool
	// GL-10: the measurement of this start (empty on a binding hit or a skip).
	Points        []SweepPoint
	Rounds        int
	PersistFailed bool
	Order         []int  // the QD of every measured window, in order
	Line          string // the [sweep] line(s) PowerShell writes
}

type sweepPoint struct {
	qd           int
	status       string
	bytes, ticks int64
}

// sweepSelection is Select-SweepQd's answer.
type sweepSelection struct {
	OK     bool
	QD     int
	Reason string
	S90    []int
	QBase  int
	MaxBq  float64
}

// selectSweepQD is Select-SweepQd (PS:4446-4489) in its three-value form.
func selectSweepQD(points []sweepPoint, arm string, n *int64) (int, string, bool) {
	s := selectSweep(points, arm, n)
	return s.QD, s.Reason, s.OK
}

// selectSweep is Select-SweepQd (PS:4446-4489). n is the arm's N: the catalog N on the
// catalog-fixed arm, the set N on the manual arm (GL-10 [A-SWEEP], F-PF §3.5: the smallest QD of
// the 90 % set that clears N+1, the catalog arm's formula with the effective N).
func selectSweep(points []sweepPoint, arm string, n *int64) sweepSelection {
	type v struct {
		qd int
		bq float64
	}
	var valid []v
	for _, p := range points {
		if p.status != "ok" || p.bytes <= 0 || p.ticks <= 0 {
			continue
		}
		bq := float64(p.bytes) / float64(p.ticks)
		if math.IsNaN(bq) || math.IsInf(bq, 0) || bq <= 0 {
			continue
		}
		valid = append(valid, v{p.qd, bq})
	}
	if len(valid) == 0 {
		return sweepSelection{QD: QDDegraded}
	}
	m := 0.0
	for _, x := range valid {
		m = math.Max(m, x.bq)
	}
	var s90 []int
	for _, x := range valid {
		if 10*x.bq >= 9*m {
			s90 = append(s90, x.qd)
		}
	}
	sort.Ints(s90)
	qBase := s90[0]
	qd, reason := qBase, "io-knee"
	if (arm == armCatalog || arm == armManual) && n != nil {
		for _, q := range s90 {
			if int64(q) >= *n+1 {
				if q != qBase {
					qd, reason = q, "prefetch-preferred"
					if arm == armManual {
						reason = manualPreferred
					}
				}
				break
			}
		}
	}
	if arm == armInit {
		for _, q := range s90 {
			if q >= 2 {
				if q != qBase {
					qd, reason = q, "init-prefetch-preferred"
				}
				break
			}
		}
	}
	return sweepSelection{OK: true, QD: qd, Reason: reason, S90: s90, QBase: qBase, MaxBq: m}
}

// sweepTargetKey is Get-SweepTargetKey (PS:4511-4528).
func sweepTargetKey(sourceTag, profileID, expectDigest, manifestSha, volumeGUID string) string {
	canon := sweepAlgorithm + "|" + sweepMethod + "|" + sourceTag + "|" + profileID + "|" +
		strings.ToLower(expectDigest) + "|" + strings.ToLower(manifestSha) + "|" + strings.ToLower(volumeGUID)
	return sha256Hex([]byte(canon))
}

func parseUTC(s string) (time.Time, bool) {
	t, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(s))
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}

// readSweepBinding is Read-SweepBinding + Test-SweepEntry + Test-SweepFreshness
// (PS:4550-4680): the stored entry for the key, fresh against checked_at.
func readSweepBinding(path, key, checkedAt string) ([]sweepPoint, string) {
	st, err := os.Stat(path)
	if err != nil || st.IsDir() {
		return nil, "no stored sweep binding"
	}
	v, err := readJSONFileStrict(path)
	if err != nil {
		return nil, "sweep binding parse failure - " + err.Error()
	}
	o, _ := asObj(v)
	if ver, ok := jNonNegInt(o["state_version"]); !ok || ver != sweepStateVer {
		return nil, "sweep binding is not state_version 2 (v1 or unknown schema)"
	}
	es, _ := asObj(o["entries"])
	e, ok := asObj(es[key])
	if !ok {
		return nil, "no entry for this target key"
	}
	bad := func(r string) ([]sweepPoint, string) { return nil, "stored entry is schema-invalid - " + r }
	if s, _ := e["probe_algorithm"].(string); s != sweepAlgorithm {
		return bad("probe_algorithm mismatch")
	}
	if s, _ := e["measurement_method"].(string); s != sweepMethod {
		return bad("measurement_method mismatch")
	}
	for _, k := range []string{"profile_id", "expectation_digest", "manifest_sha256", "output_volume_guid", "selected_reason", "swept_at", "repack_checked_at"} {
		if _, ok := jNonEmptyStr(e[k]); !ok {
			return bad(k + " missing or not a non-empty string")
		}
	}
	swept, ok1 := parseUTC(e["swept_at"].(string))
	if !ok1 {
		return bad("swept_at is not a parsable UTC timestamp")
	}
	repackAt, ok2 := parseUTC(e["repack_checked_at"].(string))
	if !ok2 {
		return bad("repack_checked_at is not a parsable UTC timestamp")
	}
	for _, k := range []string{"selected_qd", "block_bytes", "window_duration_ms"} {
		if n, ok := jNonNegInt(e[k]); !ok || n <= 0 {
			return bad(k + " missing or not a positive integer")
		}
	}
	pts, _ := jArr(e["points"])
	if len(pts) != len(sweepPoints) {
		return bad("points array does not carry every swept queue depth")
	}
	var out []sweepPoint
	for _, pv := range pts {
		po, _ := asObj(pv)
		q, ok := jNonNegInt(po["qd"])
		if !ok || q <= 0 {
			return bad("point qd is not a positive integer")
		}
		stt, _ := po["status"].(string)
		if stt != "ok" && stt != "failed" {
			return bad("point status is not ok/failed")
		}
		sp := sweepPoint{qd: int(q), status: stt}
		if stt == "ok" {
			b, okb := jNonNegInt(po["bytes"])
			t, okt := jNonNegInt(po["elapsed_ticks"])
			if !okb || b <= 0 || !okt || t <= 0 {
				return bad("ok point without a positive bytes/elapsed pair")
			}
			sp.bytes, sp.ticks = b, t
		}
		out = append(out, sp)
	}
	cur, ok := parseUTC(checkedAt)
	if !ok {
		return nil, "stale sweep binding - current verify_report.checked_at unusable - not a parsable UTC timestamp"
	}
	if !repackAt.Equal(cur) {
		return nil, "stale sweep binding - stored sweep is bound to a different verify_report.checked_at"
	}
	if swept.Before(cur) {
		return nil, "stale sweep binding - stored sweep predates the current verify_report.checked_at"
	}
	return out, ""
}

// sweptAtOf is the stored entry's swept_at as written (Resolve-QdSweep's «stored binding from»).
func sweptAtOf(path, key string) string {
	v, err := readJSONFileStrict(path)
	if err != nil {
		return ""
	}
	o, _ := asObj(v)
	es, _ := asObj(o["entries"])
	e, _ := asObj(es[key])
	s, _ := e["swept_at"].(string)
	return s
}

// resolveSweep is Resolve-QdSweep (PS:4778-4912): a stored, fresh binding gives the measured
// QD; without one the sweep measures (GL-10), its success is bound to the volume and to
// verify_report.checked_at and stored, and every failure degrades to QD_DEGRADED with probe_ok
// false (RS 5). measure is false on a preflight of the screen: the reads of a sweep (about 10 s
// of disk) belong to a start, so a miss there reports the degraded value and says so.
// n is the arm's N (selectSweep): nil = the profile's catalog N.
func (l *Launcher) resolveSweep(ctx context.Context, outputDir, sourceTag, profileID, expectDigest, manifestSha, checkedAt string, p *Profile, arm string, n *int64, measure bool) Sweep {
	out := Sweep{QD: QDDegraded, QDSource: "conservative-default", Detail: "sweep not run"}
	if n == nil {
		if pf, ok := asObj(p.Raw["prefetch"]); ok {
			if v, ok := jNonNegInt(pf["n"]); ok {
				n = &v
			}
		}
	}
	key := ""
	guid, err := l.os.volumeGUID(outputDir)
	if err == nil {
		key = sweepTargetKey(sourceTag, profileID, expectDigest, manifestSha, guid)
		pts, why := readSweepBinding(filepath.Join(l.opt.StateDir, sweepStateFile), key, checkedAt)
		if pts != nil {
			out.FromBinding = true
			sel := selectSweep(pts, arm, n)
			if sel.OK {
				out.OK, out.QD, out.Reason, out.QDSource = true, sel.QD, sel.Reason, "measured-sweep"
				out.Detail = "stored binding from " + sweptAtOf(filepath.Join(l.opt.StateDir, sweepStateFile), key)
			} else {
				out.Detail = "stored binding carries no valid point"
			}
			out.Line = fmt.Sprintf("[sweep] reusing the bound sweep record (QD%d, %s)", out.QD, out.Detail)
			l.diag.write("SWEEP_BINDING_READ", map[string]any{"key": key, "hit": true, "reason": ""})
			l.diag.write("SWEEP_BINDING_HIT", map[string]any{"key": key, "qd": out.QD, "reason": out.Reason})
			return out
		}
		l.diag.write("SWEEP_BINDING_READ", map[string]any{"key": key, "hit": false, "reason": why})
		out.Detail = why
	} else {
		out.Detail = "output volume identity unavailable - " + err.Error()
	}
	if !measure {
		out.Detail += "; the speed check (the QD sweep) runs at the start"
		return out
	}
	run := l.qdSweep(ctx, outputDir, manifestSha, arm, n)
	out.Order = run.Order
	pre := ""
	if len(run.Lines) > 0 {
		pre = strings.Join(run.Lines, "\n") + "\n"
	}
	if !run.OK {
		out.Detail = run.Reason
		out.Line = pre + fmt.Sprintf("[sweep] QD sweep could not run (%s) -> degraded: QD%d, conservative default (RS 5)", run.Reason, QDDegraded)
		l.diag.write("SWEEP_FAILED", map[string]any{"reason": run.Reason})
		return out
	}
	out.Points, out.Rounds = run.Points, run.Rounds
	if !run.Sel.OK {
		out.Detail = "every swept point failed"
		out.Line = pre + fmt.Sprintf("[sweep] every swept point failed -> degraded: QD%d, prefetch OFF (RS 5)", QDDegraded)
		l.diag.write("SWEEP_ALL_POINTS_FAILED", map[string]any{"points": run.Points})
		return out
	}
	out.OK, out.QD, out.Reason, out.QDSource = true, run.Sel.QD, run.Sel.Reason, "measured-sweep"
	out.Detail = fmt.Sprintf("measured in this run (%d round(s))", run.Rounds)
	out.Line = pre + fmt.Sprintf("[sweep] %s -> QD%d (%s)", formatSweepPoints(run.Points), out.QD, out.Reason)
	l.diag.write("SWEEP_SELECTED", map[string]any{"qd": out.QD, "reason": out.Reason, "s90": run.Sel.S90,
		"probe_algorithm": sweepAlgorithm, "measurement_method": sweepMethod, "q_base": run.Sel.QBase,
		"points": run.Points, "block_bytes": run.BlockBytes, "rounds": run.Rounds, "confirm_round": run.Confirm,
		"align_bytes": run.AlignBytes, "physical_sector": run.Physical, "manifest_align": run.ManifestAlign})
	// A success is stored only when bound to a volume identity and to a usable checked_at.
	if key == "" {
		out.PersistFailed = true
		out.Line += "\n[sweep] binding_persist=failed (no output volume identity); this run uses the measurement in memory."
		return out
	}
	if _, ok := parseUTC(checkedAt); !ok {
		out.PersistFailed = true
		out.Line += "\n[sweep] binding_persist=failed (verify_report.checked_at is not a parsable UTC timestamp); this run uses the measurement in memory."
		return out
	}
	e := sweepEntry{Algorithm: sweepAlgorithm, Method: sweepMethod, SourceTag: sourceTag, ProfileID: profileID,
		ExpectDigest: strings.ToLower(expectDigest), ManifestSha: strings.ToLower(manifestSha), VolumeGUID: guid,
		SelectedQD: out.QD, SelectedReason: out.Reason, BlockBytes: run.BlockBytes, WindowMs: run.WindowMs,
		WindowsPerPoint: run.Rounds, Points: run.Points, SweptAt: l.os.now().UTC().Format("2006-01-02T15:04:05.0000000Z"),
		RepackCheckedAt: checkedAt}
	if err := l.writeSweepBinding(key, e); err != nil {
		out.PersistFailed = true
		out.Line += "\n[sweep] binding_persist=failed (" + err.Error() + "); this run uses the measurement in memory and the next start re-probes."
		l.diag.write("sweep_binding_save_failed", map[string]any{"key": key, "reason": err.Error()})
	}
	return out
}

// newVirtualPinnedQD is New-VirtualPinnedQd (PS:4948-4958).
func newVirtualPinnedQD() Sweep {
	return Sweep{OK: true, QD: VirtualPinnedQD, QDSource: "virtual-pinned", Reason: "mode=virtual pinned shape",
		Detail: "sweep not run (mode=virtual)"}
}

// qdSource is Get-QdSource (PS:4931-4936).
func qdSource(ov map[string]string, s Sweep) string {
	if _, ok := ov["qd"]; ok {
		return "user-override"
	}
	return s.QDSource
}

// --- the effective configuration (Build-EffectiveConfig PS:8519-8673) ---

// Environment names (PS:939-947).
const (
	EnvDirect     = "MOE_DIRECT"
	EnvDirectDir  = "MOE_DIRECT_DIR"
	EnvBudgetMB   = "MOE_DIRECT_BUDGET_MB"
	EnvExpectsDir = "MOE_DIRECT_EXPECTS_DIR"
	EnvQD         = "MOE_DIRECT_QD"
	EnvPrefetchK  = "MOE_DIRECT_PREFETCH_K"
	EnvPrefetchN  = "MOE_DIRECT_PREFETCH_N"
	EnvNoPrefetch = "MOE_NO_PREFETCH"
	EnvMetrics    = "MOE_DIRECT_METRICS"
	argSlotSave   = "--slot-save-path"
)

// envBootstrapAllowlist is ENV_OS_BOOTSTRAP_ALLOWLIST (PS:1024-1029).
var envBootstrapAllowlist = []string{
	"SystemRoot", "windir", "SystemDrive", "ComSpec", "PATHEXT", "PATH", "TEMP", "TMP",
	"USERPROFILE", "LOCALAPPDATA", "APPDATA", "ProgramData", "PUBLIC",
	"NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
	"PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS", "COMPUTERNAME", "USERNAME",
	"USERDOMAIN", "SESSIONNAME", "LOGONSERVER", "HOMEDRIVE", "HOMEPATH"}

// Config is the effective server configuration.
type Config struct {
	Argv     []string
	Env      map[string]string // the configuration half of the environment block
	Port     int
	Host     string
	Budget   Budget
	QD       int
	Prefetch Prefetch
}

// Address is http://host:port.
func (c *Config) Address() string { return fmt.Sprintf("http://%s:%d", c.Host, c.Port) }

type configInput struct {
	cat          *Catalog
	profile      *Profile
	root         string
	outputDir    string
	modelPath    string
	ov           map[string]string
	decision     Prefetch
	qd           int
	installed    int64
	installedErr error
	geo          SlotGeometry
	geoErr       error
	metricsDir   string
	stamp        time.Time
	alias        string    // --alias, the short API name (review item 27); "" = none
	cache        CacheArgs // GL-06: the prompt-cache options (F-BU E; the off form by default, L:4420)
	virtual      bool      // GL-20: a virtual repack (F-PF §7.4)
}

// buildEffectiveConfig is Build-EffectiveConfig. Warm start is not ported, so the slot save
// path is removed and never re-added (the state PowerShell is in when warm start is hard-off,
// Get-WarmstartState PS:8993-8999); the collect run and the prefetch bundle view are outside
// the slice (an existing bundle is a not-ported error, none is PowerShell's "rejected").
func buildEffectiveConfig(in configInput) (*Config, error) {
	argv := append([]string(nil), in.profile.DefaultArgv...)
	argv = setArgvValue(argv, "-m", in.modelPath)
	port, ok := in.ov["port"]
	if !ok {
		port, ok = argvValue(argv, "--port")
	}
	if !ok {
		return nil, stop(StatusFailGateCatalog, "catalog defaults.argv has no --port and no override supplied")
	}
	argv = setArgvValue(argv, "--port", port)
	if v, ok := in.ov["ctx"]; ok {
		argv = setArgvValue(argv, "-c", v)
	}
	if v, ok := in.ov["threads"]; ok {
		argv = setArgvValue(argv, "-t", v)
	}
	if !contains(argv, "--no-warmup") {
		argv = append(argv, "--no-warmup")
	}
	if !contains(argv, "-np") {
		argv = append(argv, "-np", "1")
	} else {
		argv = setArgvValue(argv, "-np", "1")
	}
	argv, le := bindLoopback(argv)
	if le != nil {
		return nil, le
	}
	host := BindHost
	argv = removeArgvPair(argv, argSlotSave) // GL-15: warmConfigure sets it at a start, never the catalog
	if _, has := argvValue(argv, "--alias"); !has && in.alias != "" {
		argv = append(argv, "--alias", in.alias) // review item 27; a catalog --alias is kept
	}
	// GL-06: the prompt-cache cap the launcher passes (F-BU E.2, E.6; U-22, L:4275; L:4420)
	argv = append(StripCacheArgs(argv), in.cache.ServerArgv()...)

	env := map[string]string{}
	for k, v := range in.profile.DefaultEnv {
		if strings.EqualFold(k, envArgHost) { // rule 17: no second way to the bind (the argv wins anyway)
			continue
		}
		env[k] = v
	}
	env[EnvDirect] = "1"
	env[EnvDirectDir] = in.outputDir
	env[EnvExpectsDir] = filepath.Join(in.root, in.cat.Runtime.ExpectsDir)
	bt, le := resolveBudget(in.profile, in.ov, in.installed, in.installedErr, in.geo, in.geoErr)
	if le != nil {
		return nil, le
	}
	env[EnvBudgetMB] = strconv.FormatInt(bt.MB, 10)
	qdEff := in.qd
	if v, ok := in.ov["qd"]; ok {
		qdEff, _ = strconv.Atoi(v)
	}
	env[EnvQD] = strconv.Itoa(qdEff)
	if in.virtual { // GL-20 (F-PF §7.4): the alias and the dedicated key equal; the other knobs unset
		env[EnvIOQDTotal] = strconv.Itoa(qdEff)
		for k := range env {
			for _, knob := range virtualUnexposedKnobs {
				if strings.EqualFold(k, knob) {
					delete(env, k)
				}
			}
		}
	}
	if _, ok := env[EnvMetrics]; !ok {
		if ext := os.Getenv(EnvMetrics); ext != "" {
			env[EnvMetrics] = ext
		} else {
			env[EnvMetrics] = filepath.Join(in.metricsDir, fmt.Sprintf("metrics_%s_%d.jsonl", in.stamp.Format("20060102T150405"), os.Getpid()))
		}
	}
	mraw := env[EnvMetrics]
	if !filepath.IsAbs(mraw) {
		return nil, stop(StatusFailGateCatalog, "MOE_DIRECT_METRICS must be an absolute path (relative): %s", mraw)
	}
	mcanon := filepath.Clean(mraw)
	rootSep := strings.TrimRight(in.root, `\`) + `\`
	if strings.EqualFold(mcanon, in.root) || strings.HasPrefix(strings.ToLower(mcanon), strings.ToLower(rootSep)) {
		return nil, stop(StatusFailGateCatalog, "MOE_DIRECT_METRICS must be outside the bundle root (inside-bundle): %s", mraw)
	}
	env[EnvMetrics] = mcanon
	pf := prefetchForQD(in.decision, qdEff)
	delete(env, EnvPrefetchK)
	delete(env, EnvPrefetchN)
	delete(env, EnvNoPrefetch)
	delete(env, EnvActivationGrant) // F-PF §3.4: never set, whatever the catalog row holds
	if pf.On {
		env[EnvPrefetchK] = strconv.FormatInt(pf.K, 10)
		env[EnvPrefetchN] = strconv.FormatInt(pf.N, 10)
	} else {
		env[EnvNoPrefetch] = "1"
	}
	if st, err := os.Stat(filepath.Join(in.root, `prefetch\bundle_v1\MANIFEST.json`)); err == nil && !st.IsDir() {
		return nil, notPorted(FamilyPrefetchArming, "a prefetch bundle is present under %s; its acceptance check is not ported", in.root)
	}
	p, _ := strconv.Atoi(port)
	return &Config{Argv: argv, Env: env, Port: p, Host: host, Budget: bt, QD: qdEff, Prefetch: pf}, nil
}

// explicitEnv is New-ExplicitEnvironmentPairs (PS:11713-11729): the bootstrap allowlist from
// this process, then the configuration (which wins on a case-insensitive clash), sorted
// ordinal-ignore-case, as NAME=value lines.
func explicitEnv(cfgEnv map[string]string, getenv func(string) string) []string {
	merged := map[string][2]string{} // upper -> {name, value}
	for _, k := range envBootstrapAllowlist {
		if v := getenv(k); v != "" {
			merged[strings.ToUpper(k)] = [2]string{k, v}
		}
	}
	for k, v := range cfgEnv {
		merged[strings.ToUpper(k)] = [2]string{k, v}
	}
	keys := make([]string, 0, len(merged))
	for k := range merged {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, merged[k][0]+"="+merged[k][1])
	}
	return out
}

// crtArg is ConvertTo-CrtArgument (PS:11671-11690).
func crtArg(a string) string {
	if a != "" && !strings.ContainsAny(a, " \t\n\r\v\f\"") {
		return a
	}
	var sb strings.Builder
	sb.WriteByte('"')
	slashes := 0
	for _, c := range a {
		if c == '\\' {
			slashes++
			continue
		}
		if c == '"' {
			sb.WriteString(strings.Repeat(`\`, slashes*2+1))
			sb.WriteByte('"')
			slashes = 0
			continue
		}
		if slashes > 0 {
			sb.WriteString(strings.Repeat(`\`, slashes))
			slashes = 0
		}
		sb.WriteRune(c)
	}
	if slashes > 0 {
		sb.WriteString(strings.Repeat(`\`, slashes*2))
	}
	sb.WriteByte('"')
	return sb.String()
}

// commandLine is ConvertTo-CommandLine (PS:11692-11700).
func commandLine(exe string, argv []string) string {
	parts := []string{crtArg(exe)}
	for _, a := range argv {
		parts = append(parts, crtArg(a))
	}
	return strings.Join(parts, " ")
}
