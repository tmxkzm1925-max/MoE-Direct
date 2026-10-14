package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// GL-14 (version 8; W7 duties 5 and 6, L:4277; conflict 14): a GGUF outside the catalog whose
// structure has an architecture template (gpt-oss, qwen35moe, deepseek2) is accepted, and its
// start settings are derived per structure as the PowerShell launcher derives them: region 7b
// of Start-MoeDirect.ps1, the derive plan (Invoke-DerivePlan PS:3430-3469). The repacker runs
// once in «--plan --experimental-arch-template» mode (it writes nothing); its text is read by
// parseTemplatePlan (ConvertFrom-TemplatePlanText PS:3139-3255); newDerivedProfile builds the
// derived profile (New-DerivedProfile PS:3262-3326) and validateDerivedProfile checks it
// (Test-DerivedProfile PS:3338-3424). The values and the refusal texts are PowerShell's; the
// recorded PowerShell outputs over the fixtures of testdata/la31 lock them (S8-LA-31).
//
// The derived-profile clause of F-PF §10 (K188; KNOWN row 188, RRB:90) is the one place where
// this file leaves the PowerShell port: the writer adds `prefetch_promotion_hold: true` to every
// derived profile and the validator requires it, so an «init» request on a derived profile ends
// as OFF phase4_hold_unresolved (P4 §2 order 4-ⓐ) and a «catalog» request as OFF
// not_opted_in_evidence_unverified (F-PF §10.1, §10.4). A derived profile is never validated
// (its evidence is fixed at unverified), so the hold key is the carrier of «init refused unless
// validated»; it never lets init through (F-PF §10.2; Astra 8_19, the S8-PF-17 literal).

// Region 7b constants (PS:795-833, :918-935).
const (
	OpenArchTemplateABI         = "open-arch-template/1" // OPEN_ARCH_TEMPLATE_ABI
	DerivedProfileSchemaVersion = 1                      // DERIVED_PROFILE_SCHEMA_VERSION
	DerivedExpectFileName       = "derived.expect.json"  // DERIVED_EXPECT_FILE_NAME
	derivedLockIDPrefix         = "arch-template:"       // DERIVED_LOCK_ID_PREFIX
	derivedDigestChars          = 16                     // DERIVED_DIGEST_CHARS
	derivedArgvNCPUMoESlot      = "<n_layer>"            // DERIVED_ARGV_NCPUMOE_SLOT
	derivedBoundsBudgetMax      = 65536                  // DERIVED_BOUNDS_BUDGET_MAX
)

var (
	archTokenRe        = regexp.MustCompile(`^[a-z0-9._-]{1,32}$`)                      // ARCH_TOKEN_REGEX
	derivedProfileIDRe = regexp.MustCompile(`^derived-[a-z0-9._-]{1,32}-[0-9a-f]{16}$`) // DERIVED_PROFILE_ID_REGEX

	// derivedArgvSkeleton is DERIVED_ARGV_SKELETON (PS:924-927): the argument list every shipped
	// catalog profile shares, with the one model-dependent slot (--n-cpu-moe) filled from the plan.
	derivedArgvSkeleton = []string{"-ngl", "99", "--n-cpu-moe", derivedArgvNCPUMoESlot, "-c", "4096", "-t", "8",
		"-b", "2048", "-ub", "512", "-fa", "on", "-np", "1",
		"--host", "127.0.0.1", "--port", "8093", "--no-webui", "--no-warmup"}

	// derivedBounds are DERIVED_BOUNDS_* (PS:931-935); budget_mb.min is the plan's structural
	// minimum and is not here.
	derivedBounds = map[string]Bound{"port": {1024, 65535}, "ctx": {2048, 131072}, "threads": {1, 64}, "qd": {1, 63}}

	// derivedProfileKeys and derivedProfileForbiddenKeys are DERIVED_PROFILE_KEYS and
	// DERIVED_PROFILE_FORBIDDEN_KEYS (PS:3328-3336).
	derivedProfileKeys = []string{"derived_profile_schema_version", "profile_id", "display_name", "routed_scope", "identify",
		"min_budget_mb", "prefetch_evidence", "prefetch_activation", "prefetch", "prefetch_promotion_hold", "gates",
		"reference_measurements", "allowlist_bounds", "defaults", "derivation"}
	derivedProfileForbiddenKeys = []string{"hf_repo", "hf_revision", "expect_file", "expect_sha256", "source_shards_sha256"}
)

// isPathSafeToken is Test-PathSafeToken (PS:2841-2846): lower case and the whole pattern.
func isPathSafeToken(v string, re *regexp.Regexp) bool {
	return v == strings.ToLower(v) && re.MatchString(v)
}

// derivedLockID is Get-DerivedLockId (PS:3123-3126): what the repacker writes into the
// reference lock in template mode.
func derivedLockID(derivedFrom string) string { return derivedLockIDPrefix + derivedFrom }

// derivedExpectPath is Get-DerivedExpectPath (PS:3118-3121).
func derivedExpectPath(outDir string) string { return filepath.Join(outDir, DerivedExpectFileName) }

// TemplatePlan is ConvertFrom-TemplatePlanText's answer.
type TemplatePlan struct {
	DerivedFrom, TemplateID, TemplateVersion, RoutedScope, InventorySha256, Arch string
	NLayer, NExpert, NExpertUsed, MoELayers, TemplateLayers, RoutedTensors       int64
	SlotStrideMax, ExpertBytesTotal                                              int64
	ExpectText                                                                   string
}

// The plan lines (repack_experts.py _print_plan_summary and cmd_plan; PS:3143-3150).
var (
	planReDerive = regexp.MustCompile(`^\[EXPERIMENTAL arch-template\] derived_from=(\S+) routed_scope=(\S+) \(template default=(\S+)\) inventory_sha256=([0-9a-f]{64})$`)
	planReTpl    = regexp.MustCompile(`^\[EXPERIMENTAL arch-template\] template layers=(\d+)\.\.(\d+) \((\d+)\) routed_tensors=(\d+) `)
	planReArch   = regexp.MustCompile(`^arch=(\S+) n_layer=(\d+) n_expert=(\d+) n_expert_used=(\d+) schema=(\S+) bias=(\S+)$`)
	planReMoe    = regexp.MustCompile(`^moe_layers: (\d+) entries \[(\d+)\.\.(\d+)\]`)
	planReStride = regexp.MustCompile(`^output alignment A=(\d+), stride\[l\] .* \(min=(\d+) max=(\d+)\), slot_stride_max=(\d+)$`)
	planReBytes  = regexp.MustCompile(`^expert_payload_total\(=expert_bytes\)=(\d+)$`)
)

const (
	planExpectHead = "--- derived expect (" + DerivedExpectFileName + ", not written in --plan) ---"
	planDone       = "--plan done (0 bytes written, no GPU used)"
)

// planLong is PowerShell's [long] of a \d+ group.
func planLong(s string) int64 { v, _ := strconv.ParseInt(s, 10, 64); return v }

// parseTemplatePlan is ConvertFrom-TemplatePlanText (PS:3139-3255): every keyed line exactly
// once, numbers as numbers, the completion line present, and the derived expect body read a
// second time and checked field by field against the summary lines. The reason strings are
// PowerShell's.
func parseTemplatePlan(text string) (*TemplatePlan, string) {
	if text == "" {
		return nil, "the plan produced no output"
	}
	type pat struct {
		key string
		re  *regexp.Regexp
	}
	pats := []pat{{"derive", planReDerive}, {"tpl", planReTpl}, {"arch", planReArch}, {"moe", planReMoe}, {"stride", planReStride}, {"bytes", planReBytes}}
	found := map[string][]string{}
	expectHead := false
	inExpect, sawDone := false, false
	var expectLines []string
	for _, raw := range strings.Split(text, "\n") {
		ln := strings.TrimRight(raw, "\r")
		if ln == planDone {
			sawDone, inExpect = true, false
			continue
		}
		if ln == planExpectHead {
			if expectHead {
				return nil, "the derived expect header appears more than once"
			}
			expectHead, inExpect = true, true
			continue
		}
		if inExpect {
			expectLines = append(expectLines, ln)
			continue
		}
		for _, p := range pats {
			if m := p.re.FindStringSubmatch(ln); m != nil {
				if _, dup := found[p.key]; dup {
					return nil, `the plan carries more than one "` + p.key + `" line`
				}
				found[p.key] = m
			}
		}
	}
	for _, k := range []string{"derive", "tpl", "arch", "moe", "stride", "bytes"} {
		if _, ok := found[k]; !ok {
			return nil, `the plan has no "` + k + `" line`
		}
	}
	if !expectHead {
		return nil, "the plan printed no derived expect body"
	}
	if !sawDone {
		return nil, "the plan output is not terminated by its completion line (truncated capture)"
	}
	derivedFrom := found["derive"][1]
	parts := strings.Split(derivedFrom, "@")
	if len(parts) != 2 {
		return nil, "derived_from is not <template_id>@<version>: " + derivedFrom
	}
	if !isPathSafeToken(parts[0], archTokenRe) {
		return nil, "template id is not a safe token: " + parts[0]
	}
	if !regexp.MustCompile(`^\d{1,8}$`).MatchString(parts[1]) {
		return nil, "template version is not a number: " + parts[1]
	}
	scope := found["derive"][2]
	if !strings.EqualFold(scope, "all") && !strings.EqualFold(scope, "execution") { // -notcontains is case-insensitive
		return nil, "unknown routed scope: " + scope
	}
	arch := found["arch"][1]
	if !isPathSafeToken(arch, archTokenRe) {
		return nil, "arch is not a safe token: " + arch
	}
	if arch != parts[0] {
		return nil, "arch (" + arch + ") and template id (" + parts[0] + ") disagree"
	}
	tp := &TemplatePlan{DerivedFrom: derivedFrom, TemplateID: parts[0], TemplateVersion: parts[1], RoutedScope: scope,
		InventorySha256: strings.ToLower(found["derive"][4]), Arch: arch,
		NLayer: planLong(found["arch"][2]), NExpert: planLong(found["arch"][3]), NExpertUsed: planLong(found["arch"][4]),
		MoELayers: planLong(found["moe"][1]), TemplateLayers: planLong(found["tpl"][3]), RoutedTensors: planLong(found["tpl"][4]),
		SlotStrideMax: planLong(found["stride"][4]), ExpertBytesTotal: planLong(found["bytes"][1]),
		ExpectText: strings.Join(expectLines, "\n")}
	switch {
	case tp.NExpert <= 0:
		return nil, "n_expert is not positive"
	case tp.SlotStrideMax <= 0:
		return nil, "slot_stride_max is not positive"
	case tp.MoELayers <= 0:
		return nil, "the plan reports no MoE layers"
	case tp.MoELayers != tp.TemplateLayers:
		return nil, fmt.Sprintf("the template layer count (%d) and the layout layer count (%d) disagree", tp.TemplateLayers, tp.MoELayers)
	}
	v, err := parseJSONStrict([]byte(tp.ExpectText))
	if err != nil {
		return nil, "the derived expect body is not strict JSON - " + err.Error()
	}
	ex, _ := asObj(v)
	for _, c := range [][2]string{{"derived_from", tp.DerivedFrom}, {"template_id", tp.TemplateID},
		{"template_version", tp.TemplateVersion}, {"routed_scope", tp.RoutedScope}, {"arch", tp.Arch},
		{"inventory_sha256", tp.InventorySha256}} {
		if s := psString(ex[c[0]]); s != c[1] {
			return nil, "derived expect " + c[0] + "=" + s + " disagrees with the plan summary (" + c[1] + ")"
		}
	}
	for _, c := range []struct {
		k string
		v int64
	}{{"n_layer", tp.NLayer}, {"n_expert", tp.NExpert}, {"n_expert_used", tp.NExpertUsed},
		{"routed_tensors", tp.RoutedTensors}, {"expert_bytes_total", tp.ExpertBytesTotal}} {
		if n, ok := jNonNegInt(ex[c.k]); !ok || n != c.v {
			return nil, fmt.Sprintf("derived expect %s=%s disagrees with the plan summary (%d)", c.k, psString(ex[c.k]), c.v)
		}
	}
	return tp, ""
}

// psString is PowerShell's [string] of a decoded JSON value ($null is "").
func psString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case json.Number:
		return x.String()
	case bool:
		if x {
			return "True"
		}
		return "False"
	}
	return fmt.Sprint(v)
}

// derivedProfileDoc is the derived profile in the key order of New-DerivedProfile (PS:3272-3317).
type derivedProfileDoc struct {
	SchemaVersion         int               `json:"derived_profile_schema_version"`
	ProfileID             string            `json:"profile_id"`
	DisplayName           string            `json:"display_name"`
	RoutedScope           string            `json:"routed_scope"`
	Identify              derivedIdentify   `json:"identify"`
	MinBudgetMB           int64             `json:"min_budget_mb"`
	PrefetchEvidence      string            `json:"prefetch_evidence"`
	PrefetchActivation    string            `json:"prefetch_activation"`
	Prefetch              any               `json:"prefetch"`
	PrefetchPromotionHold bool              `json:"prefetch_promotion_hold"` // F-PF §10.1 (K188), not PowerShell's
	Gates                 derivedGates      `json:"gates"`
	ReferenceMeasurements []any             `json:"reference_measurements"`
	AllowlistBounds       derivedBoundsDoc  `json:"allowlist_bounds"`
	Defaults              derivedDefaults   `json:"defaults"`
	Derivation            derivedDerivation `json:"derivation"`
}

type derivedIdentify struct {
	Arch        string `json:"arch"`
	NLayer      int64  `json:"n_layer"`
	NExpert     int64  `json:"n_expert"`
	NExpertUsed int64  `json:"n_expert_used"`
}

type derivedGates struct {
	FormatValidated      bool `json:"format_validated"`
	PerformanceValidated bool `json:"performance_validated"`
}

type minMax struct {
	Min int64 `json:"min"`
	Max int64 `json:"max"`
}

type derivedBoundsDoc struct {
	Port     minMax `json:"port"`
	Ctx      minMax `json:"ctx"`
	Threads  minMax `json:"threads"`
	BudgetMB minMax `json:"budget_mb"`
	QD       minMax `json:"qd"`
}

type derivedDefaults struct {
	Argv []string          `json:"argv"`
	Env  map[string]string `json:"env"`
}

type derivedDerivation struct {
	ABI                string   `json:"abi"`
	TemplateID         string   `json:"template_id"`
	TemplateVersion    string   `json:"template_version"`
	DerivedFrom        string   `json:"derived_from"`
	InventorySha256    string   `json:"inventory_sha256"`
	DerivationDigest   string   `json:"derivation_digest"`
	SlotStrideMax      int64    `json:"slot_stride_max"`
	MoELayers          int64    `json:"moe_layers"`
	RoutedTensors      int64    `json:"routed_tensors"`
	ExpertBytesTotal   int64    `json:"expert_bytes_total"`
	SourceShardsSha256 []string `json:"source_shards_sha256"`
	ExpectPath         string   `json:"expect_path"`
	LockID             string   `json:"lock_id"`
}

func mm(b Bound) minMax { return minMax{b.Min, b.Max} }

// derivedArgv fills the skeleton's one slot with the layer count.
func derivedArgv(nLayer int64) []string {
	argv := make([]string, len(derivedArgvSkeleton))
	for i, a := range derivedArgvSkeleton {
		if a == derivedArgvNCPUMoESlot {
			a = strconv.FormatInt(nLayer, 10)
		}
		argv[i] = a
	}
	return argv
}

// newDerivedProfile is New-DerivedProfile (PS:3262-3326): the derived profile as its JSON object
// (round-tripped through JSON, as PowerShell does, so it reads like a catalog row).
func newDerivedProfile(tp *TemplatePlan, outDir string, shas []string) (jobj, error) {
	digest := tp.InventorySha256[:derivedDigestChars]
	minBudget := ceilMiB(tp.NExpert * tp.SlotStrideMax)
	if shas == nil {
		shas = []string{}
	}
	doc := derivedProfileDoc{
		SchemaVersion: DerivedProfileSchemaVersion,
		ProfileID:     "derived-" + tp.Arch + "-" + digest,
		DisplayName:   tp.Arch + " via arch-template " + tp.DerivedFrom + " (experimental)",
		RoutedScope:   tp.RoutedScope,
		Identify:      derivedIdentify{tp.Arch, tp.NLayer, tp.NExpert, tp.NExpertUsed},
		MinBudgetMB:   minBudget,
		// P4 1: nothing measured for a model derived on the spot.
		PrefetchEvidence: "unverified", PrefetchActivation: "off", Prefetch: nil,
		// F-PF §10.1 (K188): the hold key refuses init on every derived profile.
		PrefetchPromotionHold: true,
		Gates:                 derivedGates{FormatValidated: true, PerformanceValidated: false},
		ReferenceMeasurements: []any{},
		AllowlistBounds: derivedBoundsDoc{Port: mm(derivedBounds["port"]), Ctx: mm(derivedBounds["ctx"]),
			Threads: mm(derivedBounds["threads"]), BudgetMB: minMax{minBudget, derivedBoundsBudgetMax}, QD: mm(derivedBounds["qd"])},
		Defaults: derivedDefaults{Argv: derivedArgv(tp.NLayer), Env: map[string]string{}},
		Derivation: derivedDerivation{ABI: OpenArchTemplateABI, TemplateID: tp.TemplateID, TemplateVersion: tp.TemplateVersion,
			DerivedFrom: tp.DerivedFrom, InventorySha256: tp.InventorySha256, DerivationDigest: digest,
			SlotStrideMax: tp.SlotStrideMax, MoELayers: tp.MoELayers, RoutedTensors: tp.RoutedTensors,
			ExpertBytesTotal: tp.ExpertBytesTotal, SourceShardsSha256: shas,
			ExpectPath: derivedExpectPath(outDir), LockID: derivedLockID(tp.DerivedFrom)},
	}
	b, err := json.Marshal(doc)
	if err != nil {
		return nil, stop(StatusFailModelPath, "the derived profile could not be materialised - %v", err)
	}
	v, err := parseJSONStrict(b)
	if err != nil {
		return nil, stop(StatusFailModelPath, "the derived profile could not be materialised - %v", err)
	}
	o, _ := asObj(v)
	return o, nil
}

// validateDerivedProfile is Test-DerivedProfile (PS:3338-3424); "" = valid.
func validateDerivedProfile(o jobj) string {
	if o == nil {
		return "no derived profile"
	}
	for _, k := range derivedProfileForbiddenKeys {
		if has(o, k) {
			return "a derived profile must not carry the catalog key '" + k + "'"
		}
	}
	for k := range o {
		if !contains(derivedProfileKeys, k) {
			return "unknown key '" + k + "' in the derived profile (deny-by-default)"
		}
	}
	for _, k := range derivedProfileKeys {
		if !has(o, k) {
			return "required key '" + k + "' missing in the derived profile"
		}
	}
	if sv, ok := jNonNegInt(o["derived_profile_schema_version"]); !ok || sv != DerivedProfileSchemaVersion {
		return "derived_profile_schema_version is not an exact match"
	}
	d, _ := asObj(o["derivation"])
	idf, _ := asObj(o["identify"])
	arch := psString(idf["arch"])
	if !isPathSafeToken(arch, archTokenRe) {
		return "identify.arch is not a path-safe token: " + arch
	}
	inv := psString(d["inventory_sha256"])
	if !isSha256Hex(inv) || inv != strings.ToLower(inv) {
		return "derivation.inventory_sha256 is not a lowercase 64 hex digest"
	}
	pid := psString(o["profile_id"])
	if !isPathSafeToken(pid, derivedProfileIDRe) {
		return "profile_id is not a path-safe derived id: " + pid
	}
	if want := "derived-" + arch + "-" + inv[:derivedDigestChars]; pid != want {
		return "profile_id is not the deterministic id for this derivation (expected " + want + ")"
	}
	if psString(d["abi"]) != OpenArchTemplateABI {
		return "derivation.abi does not match this launcher OPEN_ARCH_TEMPLATE_ABI"
	}
	if psString(d["derived_from"]) != psString(d["template_id"])+"@"+psString(d["template_version"]) {
		return "derivation.derived_from is not template_id@template_version"
	}
	if psString(o["prefetch_evidence"]) != "unverified" || psString(o["prefetch_activation"]) != "off" {
		return "a derived profile is prefetch_evidence=unverified / prefetch_activation=off by contract"
	}
	if o["prefetch"] != nil {
		return "a derived profile carries no prefetch tuple"
	}
	if h, ok := o["prefetch_promotion_hold"].(bool); !ok || !h {
		return "a derived profile holds prefetch_promotion_hold = true by contract (F-PF §10.1)"
	}
	if g, _ := asObj(o["gates"]); g != nil {
		if b, ok := g["performance_validated"].(bool); ok && b {
			return "a derived profile can never claim performance_validated"
		}
	}
	nExp, _ := jInt(idf["n_expert"])
	nLayer, _ := jInt(idf["n_layer"])
	stride, _ := jInt(d["slot_stride_max"])
	layers, _ := jInt(d["moe_layers"])
	minB, _ := jInt(o["min_budget_mb"])
	if nExp <= 0 || stride <= 0 || layers <= 0 {
		return "the slot geometry is not positive"
	}
	if minB != ceilMiB(nExp*stride) {
		return "min_budget_mb is not ceil(n_expert * slot_stride_max / MiB)"
	}
	if minB > ceilMiB(nExp*layers*stride) {
		return "min_budget_mb exceeds the full slot residency (model_cap)"
	}
	ab, _ := asObj(o["allowlist_bounds"])
	bb, _ := asObj(ab["budget_mb"])
	if bmin, _ := jInt(bb["min"]); bmin != minB {
		return "allowlist_bounds.budget_mb.min is not the derived minimum"
	}
	def, _ := asObj(o["defaults"])
	arr, _ := jArr(def["argv"])
	argv := make([]string, 0, len(arr))
	for _, a := range arr {
		argv = append(argv, psString(a))
	}
	ncm, _ := argvValue(argv, "--n-cpu-moe")
	if ncm != strconv.FormatInt(nLayer, 10) {
		return "defaults.argv --n-cpu-moe is not the derived layer count"
	}
	host, _ := argvValue(argv, "--host")
	if !isLoopback(host) {
		return "defaults.argv binds a non-loopback host"
	}
	return ""
}

// Derivation is what a derived profile carries beyond the catalog fields: the plan's facts and
// the two identities the gate and the state bind to (PS:13950-13962).
type Derivation struct {
	DerivedFrom      string
	InventorySha256  string // the state binding (preset, sweep): PowerShell's $expectDigest
	LockID           string // the reference lock the repacker writes: arch-template:<template>@<version>
	ExpectPath       string // <out>\derived.expect.json, written by the template repack
	SlotStrideMax    int64
	MoELayers        int64
	RoutedTensors    int64
	ExpertBytesTotal int64
	PlanText         string
}

// profileFromDerived types a validated derived profile object as a Profile: the defaults, the
// bounds and the prefetch axes are read from it as from a catalog row (Raw), and Derived holds
// the derivation.
func profileFromDerived(o jobj, tp *TemplatePlan, planText string) *Profile {
	p := &Profile{Raw: o, Bounds: map[string]Bound{}, DefaultEnv: map[string]string{}}
	p.ID = psString(o["profile_id"])
	p.DisplayName = psString(o["display_name"])
	p.RoutedScope = psString(o["routed_scope"])
	p.Identify = Identify{Arch: tp.Arch, NLayer: tp.NLayer, NExpert: tp.NExpert, NExpertUsed: tp.NExpertUsed}
	p.MinBudgetMB, _ = jInt(o["min_budget_mb"])
	p.FormatValidated, p.PerformanceValidated = true, false
	ab, _ := asObj(o["allowlist_bounds"])
	for _, k := range catalogBoundKeys {
		b, _ := asObj(ab[k])
		lo, _ := jInt(b["min"])
		hi, _ := jInt(b["max"])
		p.Bounds[k] = Bound{lo, hi}
	}
	def, _ := asObj(o["defaults"])
	arr, _ := jArr(def["argv"])
	for _, a := range arr {
		p.DefaultArgv = append(p.DefaultArgv, psString(a))
	}
	d, _ := asObj(o["derivation"])
	p.ExpectPath = psString(d["expect_path"])
	p.ExpectSha256 = tp.InventorySha256
	p.Derived = &Derivation{DerivedFrom: tp.DerivedFrom, InventorySha256: tp.InventorySha256,
		LockID: psString(d["lock_id"]), ExpectPath: p.ExpectPath, SlotStrideMax: tp.SlotStrideMax,
		MoELayers: tp.MoELayers, RoutedTensors: tp.RoutedTensors, ExpertBytesTotal: tp.ExpertBytesTotal,
		PlanText: planText}
	return p
}

// deriveCache keeps the derived profile of a model file for this launcher's life, keyed by the
// shard identities, the output folder and the catalog's repacker: the screen preflights a model
// at every change of its settings, and PowerShell runs the plan once per start.
type deriveCache struct {
	mu sync.Mutex
	m  map[string]*Profile
}

func (c *deriveCache) get(k string) (*Profile, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.m[k]
	return p, ok
}

func (c *deriveCache) put(k string, p *Profile) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[string]*Profile{}
	}
	c.m[k] = p
}

// templatePlanArgv is Invoke-Repacker -PlanOnly -ArchTemplate (PS:12355-12384): no --profile
// and no --scope (the repacker refuses --profile with the template, and the template decides
// the scope).
func templatePlanArgv(cat *Catalog, root, modelPath, outDir, logPath string) (string, []string) {
	exe := filepath.Join(root, cat.Runtime.RepackerExe)
	var argv []string
	for _, a := range cat.Runtime.RepackerArgv {
		if strings.HasPrefix(a, "./") || strings.HasPrefix(a, `.\`) {
			a = filepath.Join(root, a[2:])
		}
		argv = append(argv, a)
	}
	argv = append(argv, "--plan", "--experimental-arch-template", "--model", modelPath, "--out", outDir, "--log-path", logPath)
	return exe, argv
}

// derivePlan is Invoke-DerivePlan (PS:3430-3469) steps 2 to 5: the repacker's template plan
// (writes nothing), its text read, the derived profile completed and validated. Every refusal is
// fail_model_path, as PowerShell's. The result is cached for the launcher's life.
func (l *Launcher) derivePlan(ctx context.Context, cat *Catalog, modelPath, outDir string, set *ShardSet, shas []string) (*Profile, error) {
	key := outDir + "|" + cat.Runtime.RepackerExe + "|" + strings.Join(cat.Runtime.RepackerArgv, " ")
	for _, s := range set.Shards {
		id, err := l.os.identity(s.Path)
		if err != nil {
			key = ""
			break
		}
		key += "|" + shardCacheKey(id)
	}
	if key != "" {
		if p, ok := l.derive.get(key); ok {
			return p, nil
		}
	}
	logDir := l.LogDir()
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, stop(StatusFailModelPath, "repacker --plan: log directory: %v", err)
	}
	stamp := l.os.now().UTC().Format("20060102T150405Z")
	outLog := filepath.Join(logDir, "repack_plan_"+stamp+"_out.log")
	errLog := filepath.Join(logDir, "repack_plan_"+stamp+"_err.log")
	exe, argv := templatePlanArgv(cat, l.opt.BundleRoot, modelPath, outDir, filepath.Join(logDir, "repack_log.jsonl"))
	ex := l.deriveExec
	if ex == nil {
		ex = ProcessExecutor{}
	}
	c, cancel := context.WithTimeout(ctx, planTimeout)
	defer cancel()
	code, err := ex.Run(c, exe, argv, l.opt.BundleRoot, repackEnv(), outLog, errLog, nil)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, stop(StatusFailModelPath, "repacker --plan exceeded the %d s deadline and was terminated; see %s", int(planTimeout.Seconds()), errLog)
		}
		if ctx.Err() != nil {
			return nil, stop(StatusCancelledUser, "stop requested during the repacker --plan run")
		}
		return nil, stop(StatusFailModelPath, "repacker spawn failed: %v", err)
	}
	if code != 0 {
		return nil, stop(StatusFailModelPath, "repacker --plan exited abnormally (code=%d); see %s", code, errLog)
	}
	b, _ := os.ReadFile(outLog)
	text := string(b)
	tp, why := parseTemplatePlan(text)
	if tp == nil {
		return nil, stop(StatusFailModelPath, "the arch-template plan could not be read - %s", why)
	}
	o, err := newDerivedProfile(tp, outDir, shas)
	if err != nil {
		return nil, err
	}
	if why := validateDerivedProfile(o); why != "" {
		return nil, stop(StatusFailModelPath, "the derived profile failed its own validator - %s", why)
	}
	p := profileFromDerived(o, tp, text)
	l.diag.write("DERIVE_PLAN", map[string]any{"profile_id": p.ID, "derived_from": tp.DerivedFrom,
		"inventory_sha256": tp.InventorySha256, "routed_scope": tp.RoutedScope, "arch": tp.Arch, "n_layer": tp.NLayer,
		"n_expert": tp.NExpert, "moe_layers": tp.MoELayers, "routed_tensors": tp.RoutedTensors,
		"slot_stride_max": tp.SlotStrideMax, "expert_bytes_total": tp.ExpertBytesTotal, "min_budget_mb": p.MinBudgetMB,
		"source_shards": len(shas), "abi": OpenArchTemplateABI})
	if key != "" {
		l.derive.put(key, p)
	}
	return p, nil
}

// derivedGateIdentity is the gate's binding for a derived profile (PS:14176-14185): the lock id
// is the arch-template marker and the expect hash is the real bytes of derived.expect.json in
// the output folder; "" when that file is absent (the template setup has not run yet).
func derivedGateIdentity(p *Profile) (lockID, expectSha string, err error) {
	if _, e := os.Stat(p.Derived.ExpectPath); e != nil {
		return p.Derived.LockID, "", nil
	}
	h, e := fileSha256(p.Derived.ExpectPath)
	if e != nil {
		return p.Derived.LockID, "", stop(StatusFailGateVerify, "the derived expect could not be hashed - %v", e)
	}
	return p.Derived.LockID, h, nil
}

// StructureDefaults are the start settings a model outside the catalog of the named structure
// shows before its plan has run (conflict 14, review item 12): the derived skeleton's values and
// bounds; the budget's lower bound is the plan's structural minimum and is shown as unknown
// (MinBudgetMB 0) until the plan ran. ok is false for a structure without a template.
type StructureDefaults struct {
	Arch   string
	Argv   []string // --n-cpu-moe holds the slot «<n_layer>»
	Bounds map[string]Bound
}

// TemplateDefaults gives StructureDefaults for arch (gpt-oss, qwen35moe, deepseek2).
func TemplateDefaults(arch string) (StructureDefaults, bool) {
	if !contains(archTemplateFamilies, arch) {
		return StructureDefaults{}, false
	}
	b := map[string]Bound{"budget_mb": {0, derivedBoundsBudgetMax}}
	for k, v := range derivedBounds {
		b[k] = v
	}
	return StructureDefaults{Arch: arch, Argv: append([]string(nil), derivedArgvSkeleton...), Bounds: b}, true
}
