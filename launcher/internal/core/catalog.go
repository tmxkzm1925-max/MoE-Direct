package core

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// CatalogSchemaVersion is CATALOG_SCHEMA_VERSION (PS:783).
const CatalogSchemaVersion = 2

// Key lists of region 6 (PS:2247-2272).
var (
	catalogTopKeys         = []string{"catalog_schema_version", "source_tag", "runtime", "profiles"}
	catalogRuntimeKeys     = []string{"server_exe", "repacker_exe", "repacker_argv", "expects_dir", "webui"}
	catalogProfileKeys     = []string{"profile_id", "display_name", "hf_repo", "hf_revision", "routed_scope", "expect_file", "expect_sha256", "identify", "min_budget_mb", "prefetch_evidence", "prefetch_activation", "prefetch", "gates", "reference_measurements", "allowlist_bounds", "defaults"}
	catalogProfileOptional = []string{"source_shards_sha256", "prefetch_promotion_hold"}
	catalogIdentifyKeys    = []string{"arch", "n_layer", "n_expert", "n_expert_used"}
	catalogBoundKeys       = []string{"port", "ctx", "threads", "budget_mb", "qd"}
	measurementColumns     = []string{"model", "tier", "machine_storage", "budget_qd_prefetch", "workload_window", "observed_tok_s"}
)

// Engine QD caps, ENGINE_QD_MIN/MAX (PS:998-999).
const (
	EngineQDMin = 1
	EngineQDMax = 63
)

// Catalog is models.json after Read-Catalog (PS:2288-2351).
type Catalog struct {
	Path      string
	SourceTag string
	Runtime   Runtime
	Profiles  []*Profile
}

// Runtime is the catalog's runtime object.
type Runtime struct {
	ServerExe    string
	RepackerExe  string
	RepackerArgv []string
	ExpectsDir   string
	WebUI        bool
}

// Bound is one allowlist_bounds entry.
type Bound struct{ Min, Max int64 }

// Identify is a profile's header fingerprint.
type Identify struct {
	Arch                         string
	NLayer, NExpert, NExpertUsed int64
}

// Measurement is one reference_measurements row; Complete is false when a condition column
// is missing (LS 1-9, Format-ReferenceMeasurements PS:2529-2549), and the screen then shows
// [unmeasured] instead of the number.
type Measurement struct {
	Model, Tier, MachineStorage, BudgetQDPrefetch, WorkloadWindow, ObservedTokS string
	Complete                                                                    bool
}

// Profile is one catalog profile, typed. Raw keeps the decoded object for the prefetch axis
// reader, which must see the row exactly as written (Get-PrefetchCatalogAxes PS:7322).
type Profile struct {
	ID, DisplayName, HFRepo, HFRevision, RoutedScope string
	ExpectFile, ExpectSha256                         string
	Identify                                         Identify
	MinBudgetMB                                      int64
	SourceShardsSha256                               []string // lower case; empty = unpinned
	FormatValidated, PerformanceValidated            bool
	Measurements                                     []Measurement
	Bounds                                           map[string]Bound
	DefaultArgv                                      []string
	DefaultEnv                                       map[string]string
	ExpectPath                                       string // bundle root + expects_dir + expect_file
	Raw                                              jobj
	// Derived is set for a derived profile (a model outside the catalog, GL-14; derive.go) and
	// nil for a catalog row. ExpectPath is then the derived expect in the output folder and
	// ExpectSha256 the inventory digest that binds the state.
	Derived *Derivation
}

// ReadCatalog is Read-Catalog + Test-CatalogProfile (PS:2288-2511). catalogPath is the
// models.json file; bundleRoot is the engine bundle whose expects directory the expect files
// are re-hashed from. The PowerShell launcher reads both from one root (PS:2290, :2476); the
// installed staging has no models.json, so the draft takes the two apart (landing, deviation 1).
func ReadCatalog(catalogPath, bundleRoot string) (*Catalog, error) {
	st, err := os.Stat(catalogPath)
	if err != nil || st.IsDir() {
		return nil, stop(StatusFailGateCatalog, "catalog missing: %s", catalogPath)
	}
	v, err := readJSONFileStrict(catalogPath)
	if err != nil {
		return nil, stop(StatusFailGateCatalog, "catalog unreadable - %v", err)
	}
	c, ok := asObj(v)
	if !ok {
		return nil, stop(StatusFailGateCatalog, "catalog root is not a JSON object")
	}
	if le := denyUnknownKeys(c, catalogTopKeys, nil, "models.json"); le != nil {
		return nil, le
	}
	if sv, ok := jNonNegInt(c["catalog_schema_version"]); !ok || sv != CatalogSchemaVersion {
		return nil, stop(StatusFailGateCatalog, "catalog_schema_version is not an exact match")
	}
	tag, ok := jNonEmptyStr(c["source_tag"])
	if !ok {
		return nil, stop(StatusFailGateCatalog, "source_tag missing")
	}
	rt, _ := asObj(c["runtime"])
	if le := denyUnknownKeys(rt, catalogRuntimeKeys, nil, "runtime"); le != nil {
		return nil, le
	}
	cat := &Catalog{Path: catalogPath, SourceTag: tag}
	for _, k := range []string{"server_exe", "repacker_exe", "expects_dir"} {
		if _, ok := jNonEmptyStr(rt[k]); !ok {
			return nil, stop(StatusFailGateCatalog, "runtime.%s must be a non-empty string", k)
		}
	}
	cat.Runtime.ServerExe, _ = jStr(rt["server_exe"])
	cat.Runtime.RepackerExe, _ = jStr(rt["repacker_exe"])
	cat.Runtime.ExpectsDir, _ = jStr(rt["expects_dir"])
	ra, ok := jArr(rt["repacker_argv"])
	if !ok {
		return nil, stop(StatusFailGateCatalog, "runtime.repacker_argv must be an array")
	}
	for _, a := range ra {
		s, _ := a.(string)
		cat.Runtime.RepackerArgv = append(cat.Runtime.RepackerArgv, s)
	}
	wb, ok := jBool(rt["webui"])
	if !ok {
		return nil, stop(StatusFailGateCatalog, "runtime.webui must be a JSON boolean")
	}
	cat.Runtime.WebUI = wb

	profiles, ok := jArr(c["profiles"])
	if !ok || len(profiles) == 0 {
		return nil, stop(StatusFailGateCatalog, "profiles[] missing or empty")
	}
	seen := map[string]bool{}
	pinSeen := map[string]string{}
	for _, pv := range profiles {
		po, ok := asObj(pv)
		if !ok {
			return nil, stop(StatusFailGateCatalog, "profiles[] entry is not a JSON object")
		}
		p, le := testCatalogProfile(po, bundleRoot, cat.Runtime)
		if le != nil {
			return nil, le
		}
		if seen[p.ID] {
			return nil, stop(StatusFailGateCatalog, "duplicate profile_id: %s", p.ID)
		}
		seen[p.ID] = true
		if len(p.SourceShardsSha256) > 0 {
			vec := strings.Join(p.SourceShardsSha256, "|")
			if other, dup := pinSeen[vec]; dup {
				return nil, stop(StatusFailGateCatalog, "profiles '%s' and '%s' are pinned to the same source digests", other, p.ID)
			}
			pinSeen[vec] = p.ID
		}
		cat.Profiles = append(cat.Profiles, p)
	}
	return cat, nil
}

// ProfileByID is Get-CatalogProfileById (PS:2513-2519).
func (c *Catalog) ProfileByID(id string) *Profile {
	for _, p := range c.Profiles {
		if p.ID == id {
			return p
		}
	}
	return nil
}

func testCatalogProfile(o jobj, root string, rt Runtime) (*Profile, *LaunchError) {
	if le := denyUnknownKeys(o, catalogProfileKeys, catalogProfileOptional, "profiles[]"); le != nil {
		return nil, le
	}
	p := &Profile{Raw: o, Bounds: map[string]Bound{}, DefaultEnv: map[string]string{}}
	var ok bool
	if p.ID, ok = jNonEmptyStr(o["profile_id"]); !ok {
		return nil, stop(StatusFailGateCatalog, "profile_id must be a non-empty string")
	}
	for _, k := range []string{"display_name", "hf_repo", "hf_revision", "expect_file"} {
		if _, ok := jNonEmptyStr(o[k]); !ok {
			return nil, stop(StatusFailGateCatalog, "%s: %s must be a non-empty string", p.ID, k)
		}
	}
	p.DisplayName, _ = jStr(o["display_name"])
	p.HFRepo, _ = jStr(o["hf_repo"])
	p.HFRevision, _ = jStr(o["hf_revision"])
	p.ExpectFile, _ = jStr(o["expect_file"])
	scope, _ := o["routed_scope"].(string)
	if !strings.EqualFold(scope, "all") && !strings.EqualFold(scope, "execution") { // -notcontains is case-insensitive
		return nil, stop(StatusFailGateCatalog, "%s: routed_scope must be 'all' or 'execution'", p.ID)
	}
	p.RoutedScope = scope
	if !isSha256Hex(o["expect_sha256"]) {
		return nil, stop(StatusFailGateCatalog, "%s: expect_sha256 missing or not 64 hex (expect digest absent)", p.ID)
	}
	p.ExpectSha256 = strings.ToLower(o["expect_sha256"].(string))

	id, _ := asObj(o["identify"])
	if le := denyUnknownKeys(id, catalogIdentifyKeys, nil, p.ID+".identify"); le != nil {
		return nil, le
	}
	if p.Identify.Arch, ok = jNonEmptyStr(id["arch"]); !ok {
		return nil, stop(StatusFailGateCatalog, "%s: identify.arch must be a non-empty string", p.ID)
	}
	for _, k := range []string{"n_layer", "n_expert", "n_expert_used"} {
		n, ok := jNonNegInt(id[k])
		if !ok {
			return nil, stop(StatusFailGateCatalog, "%s: identify.%s must be a non-negative integer", p.ID, k)
		}
		switch k {
		case "n_layer":
			p.Identify.NLayer = n
		case "n_expert":
			p.Identify.NExpert = n
		default:
			p.Identify.NExpertUsed = n
		}
	}
	if p.MinBudgetMB, ok = jNonNegInt(o["min_budget_mb"]); !ok {
		return nil, stop(StatusFailGateCatalog, "%s: min_budget_mb must be a non-negative integer", p.ID)
	}
	if has(o, "source_shards_sha256") {
		ss, ok := jArr(o["source_shards_sha256"])
		if !ok {
			return nil, stop(StatusFailGateCatalog, "%s: source_shards_sha256 must be a JSON array (absent or [] = unpinned)", p.ID)
		}
		for _, h := range ss {
			if !isSha256Hex(h) {
				return nil, stop(StatusFailGateCatalog, "%s: source_shards_sha256 entries must all be 64 hex digests", p.ID)
			}
			p.SourceShardsSha256 = append(p.SourceShardsSha256, strings.ToLower(h.(string)))
		}
	}
	for _, k := range []string{"prefetch_evidence", "prefetch_activation"} {
		if _, ok := jNonEmptyStr(o[k]); !ok {
			return nil, stop(StatusFailGateCatalog, "%s: %s must be a non-empty string", p.ID, k)
		}
	}
	if has(o, "prefetch_promotion_hold") {
		if _, ok := jBool(o["prefetch_promotion_hold"]); !ok {
			return nil, stop(StatusFailGateCatalog, "%s: prefetch_promotion_hold must be a JSON boolean (absent = false)", p.ID)
		}
	}
	if pf := o["prefetch"]; pf != nil {
		pfo, _ := asObj(pf)
		if le := denyUnknownKeys(pfo, []string{"k", "n"}, nil, p.ID+".prefetch"); le != nil {
			return nil, le
		}
	}
	g, _ := asObj(o["gates"])
	if le := denyUnknownKeys(g, []string{"format_validated", "performance_validated"}, nil, p.ID+".gates"); le != nil {
		return nil, le
	}
	if p.FormatValidated, ok = jBool(g["format_validated"]); !ok {
		return nil, stop(StatusFailGateCatalog, "%s: gates.format_validated must be a JSON boolean", p.ID)
	}
	if p.PerformanceValidated, ok = jBool(g["performance_validated"]); !ok {
		return nil, stop(StatusFailGateCatalog, "%s: gates.performance_validated must be a JSON boolean", p.ID)
	}
	rm, ok := jArr(o["reference_measurements"])
	if !ok {
		return nil, stop(StatusFailGateCatalog, "%s: reference_measurements must be an array", p.ID)
	}
	for _, mv := range rm {
		mo, _ := asObj(mv)
		m := Measurement{Complete: true}
		vals := make([]string, len(measurementColumns))
		for i, col := range measurementColumns {
			v, present := mo[col]
			if !present || v == nil {
				m.Complete = false
				continue
			}
			vals[i] = fmt.Sprint(v)
		}
		m.Model, m.Tier, m.MachineStorage, m.BudgetQDPrefetch, m.WorkloadWindow, m.ObservedTokS =
			vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
		p.Measurements = append(p.Measurements, m)
	}

	ab, _ := asObj(o["allowlist_bounds"])
	if le := denyUnknownKeys(ab, catalogBoundKeys, nil, p.ID+".allowlist_bounds"); le != nil {
		return nil, le
	}
	for _, k := range catalogBoundKeys {
		b, _ := asObj(ab[k])
		if le := denyUnknownKeys(b, []string{"min", "max"}, nil, p.ID+".allowlist_bounds."+k); le != nil {
			return nil, le
		}
		lo, ok1 := jNonNegInt(b["min"])
		hi, ok2 := jNonNegInt(b["max"])
		if !ok1 {
			return nil, stop(StatusFailGateCatalog, "%s: allowlist_bounds.%s.min must be a non-negative integer", p.ID, k)
		}
		if !ok2 {
			return nil, stop(StatusFailGateCatalog, "%s: allowlist_bounds.%s.max must be a non-negative integer", p.ID, k)
		}
		if lo > hi {
			return nil, stop(StatusFailGateCatalog, "%s: allowlist_bounds.%s min > max", p.ID, k)
		}
		p.Bounds[k] = Bound{lo, hi}
	}
	if p.Bounds["qd"].Min < EngineQDMin || p.Bounds["qd"].Max > EngineQDMax {
		return nil, stop(StatusFailGateCatalog, "%s: allowlist_bounds.qd outside engine range 1..63", p.ID)
	}
	if p.Bounds["budget_mb"].Min < p.MinBudgetMB {
		return nil, stop(StatusFailGateCatalog, "%s: allowlist_bounds.budget_mb.min below min_budget_mb", p.ID)
	}

	d, _ := asObj(o["defaults"])
	if le := denyUnknownKeys(d, []string{"argv", "env"}, nil, p.ID+".defaults"); le != nil {
		return nil, le
	}
	argv, ok := jArr(d["argv"])
	if !ok {
		return nil, stop(StatusFailGateCatalog, "%s: defaults.argv must be an array", p.ID)
	}
	for _, a := range argv {
		s, ok := a.(string)
		if !ok {
			return nil, stop(StatusFailGateCatalog, "%s: defaults.argv entries must be strings", p.ID)
		}
		p.DefaultArgv = append(p.DefaultArgv, s)
	}
	if de, ok := asObj(d["env"]); ok {
		for k, v := range de {
			p.DefaultEnv[k] = fmt.Sprint(v)
		}
	}

	// Expect re-hash (PS:2475-2484).
	p.ExpectPath = filepath.Join(root, rt.ExpectsDir, p.ExpectFile)
	if st, err := os.Stat(p.ExpectPath); err != nil || st.IsDir() {
		return nil, stop(StatusFailGateCatalog, "%s: expect file missing: %s", p.ID, p.ExpectPath)
	}
	h, err := fileSha256(p.ExpectPath)
	if err != nil {
		return nil, stop(StatusFailGateCatalog, "%s: expect hash failed - %v", p.ID, err)
	}
	if h != p.ExpectSha256 {
		return nil, stop(StatusFailGateCatalog, "%s: expect file re-hash != catalog expect_sha256", p.ID)
	}
	// Pin arity against expect.sources[] (PS:2496-2510).
	if len(p.SourceShardsSha256) > 0 {
		srcs, err := expectSources(p.ExpectPath)
		if err != nil {
			return nil, stop(StatusFailGateCatalog, "%s: expect unreadable while checking the source pin - %v", p.ID, err)
		}
		if srcs == nil {
			return nil, stop(StatusFailGateCatalog, "%s: expect sources[] missing while checking the source pin", p.ID)
		}
		if len(srcs) != len(p.SourceShardsSha256) {
			return nil, stop(StatusFailGateCatalog, "%s: source_shards_sha256 has %d digest(s) but the expect describes %d source shard(s)", p.ID, len(p.SourceShardsSha256), len(srcs))
		}
	}
	return p, nil
}

// expectSources returns expect.sources[] (nil when absent or not an array).
func expectSources(path string) ([]any, error) {
	v, err := readJSONFileStrict(path)
	if err != nil {
		return nil, err
	}
	o, _ := asObj(v)
	s, ok := jArr(o["sources"])
	if !ok {
		return nil, nil
	}
	return s, nil
}

// fileSha256 is Get-FileSha256Lower (PS:2116-2124).
func fileSha256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
