package core

import (
	"fmt"
	"strconv"
)

// GL-19: the manual prefetch surface of the manual-prefetch edition (F-PF,
// reviews/_edition_8_F-PF_manual_prefetch.md): the pair K and N on the three surfaces (the
// settings screen, the per-model saved settings, the argument-driven mode; §2), its checks
// before start (§9 C1-C3, C6, C11), the manual arm of the resolution order (§3.2 step 3), the
// held rows (§5), QD below N+1 (§6), the source field and the pre-seal echo (§4). The engine and
// the adaptive controller are untouched (§0.2 B2, B3); no new off-reason literal (B4).

// The bounds of §1 T7 (PB:1309-1316): K 1..16, N at least 1; N's upper bound QD-1 is an
// activation rule (§6), not a refusal.
const (
	PrefetchKMin = 1
	PrefetchKMax = 16
	PrefetchNMin = 1
)

// The surfaces of §1 T2 (U-15 (iii), L:4264): the surface whose prefetch value the run uses.
const (
	SourceSettingsScreen = "settings_screen"
	SourceProfile        = "profile"
	SourceArguments      = "arguments"
)

// The labels of §1 T3 (INIT inv. 7, unchanged) and the init version of P4's echo.
const (
	ProvInitV1          = "init_v1-unvalidated"
	ProvCatalog         = "catalog-validated"
	ProvEnvOverride     = "env-override" // a manual pair's ON candidate (§4.3; OQ-1 (a), D13)
	PrefetchInitVersion = "prefetch_init_v1"
)

// EnvActivationGrant is never set (§3.4; §0.2 B3; S8-PF-09).
const EnvActivationGrant = "MOE_DIRECT_PREFETCH_ACTIVATION_GRANT"

// The T18 texts of §1 and §2.5 this atom shows (English; the Korean entries are the screen's
// string table). The virtual-repack texts (NoticeVirtualUnvalidated, N_cap, the wave refusal)
// are GL-20's.
const (
	NoticeManualUnmeasured      = "manual prefetch: this pair is not a validated pair for this model; its behaviour is unmeasured"
	WarnInitIgnoredPair         = "prefetch=init ignored: a manual K/N pair is set"
	WarnInitIgnoredCatalogFixed = "prefetch=init ignored: this profile is catalog-fixed (validated K/N); the measured pair is kept"
	WarnInitV1                  = "prefetch_init_v1 is an unvalidated starting point (K=n_expert_used, N=min(4,QD-1,floor(QD/2))): no performance claim is made for this pair"
	// EchoCandidateOnly is the line an ON candidate adds to the echo (SMD:10782-10796; §4.4).
	EchoCandidateOnly = "candidate only - the engine seal decides the final activation."
)

// The refusal sentences of §2.5 (S8-PF-01, S8-PF-07; the form of the collect-tuple messages,
// PFP:78): they name the key, the bound and the value.
func refuseKText(v string) string { return fmt.Sprintf("K must be a whole number 1..16 (got: %s)", v) }
func refuseNText(v string) string {
	return fmt.Sprintf("N must be a whole number of at least 1 (got: %s)", v)
}
func refuseQDText(v string) string {
	return fmt.Sprintf("QD must be a whole number 1..63 (got: %s)", v)
}

const refusePairText = "K and N are set together or not at all"

func refuseKAboveText(k, e int64) string {
	return fmt.Sprintf("K %d is above this model's n_expert %d", k, e)
}

// The allowed ranges as the per-path refusal texts of F-LS 11.4 name them.
const (
	allowedK    = "1..16"
	allowedN    = "1 or more"
	allowedPair = "K and N together, or neither"
)

// digitsOnly is §9 C1's shape: decimal digits only, on the value as the surface delivered it
// (no trim, no sign: «8x», «+8», « 8» and the empty value are refused).
func digitsOnly(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// checkPairValue is §9 C1 and C2 for one value of the pair: the canonical decimal string, or
// the §2.5 sentence.
func checkPairValue(key, v string) (string, error) {
	n, err := strconv.ParseInt(v, 10, 64)
	switch key {
	case "prefetch_k":
		if !digitsOnly(v) || err != nil || n < PrefetchKMin || n > PrefetchKMax {
			return "", fmt.Errorf("%s", refuseKText(v))
		}
	default:
		if !digitsOnly(v) || err != nil || n < PrefetchNMin {
			return "", fmt.Errorf("%s", refuseNText(v))
		}
	}
	return strconv.FormatInt(n, 10), nil
}

// isPairKey is true for the two keys of the manual pair.
func isPairKey(k string) bool { return k == "prefetch_k" || k == "prefetch_n" }

// ManualPair is §1 T10: the pair a surface set, after the checks of §9.
type ManualPair struct {
	K, N int64
}

// manualPairOf is the pair of checked overrides (nil = no pair; resolveOverrides has already
// refused a pair with one value).
func manualPairOf(ov map[string]string) *ManualPair {
	k, okK := ov["prefetch_k"]
	n, okN := ov["prefetch_n"]
	if !okK || !okN {
		return nil
	}
	kv, _ := strconv.ParseInt(k, 10, 64)
	nv, _ := strconv.ParseInt(n, 10, 64)
	return &ManualPair{K: kv, N: nv}
}

// prefetchSource is §4.2: the surface whose prefetch value the run uses, under the settings
// order (the per-model profile loaded, then the screen's changes; or the arguments). The keys
// are the pair when it is set, else the request; "" when no surface set either.
func prefetchSource(req Request, preset map[string]string, ov map[string]string) string {
	keys := []string{"prefetch"}
	if manualPairOf(ov) != nil {
		keys = []string{"prefetch_k", "prefetch_n"}
	}
	src := ""
	for _, k := range keys {
		if _, ok := ov[k]; !ok {
			continue
		}
		v, given := req.Settings[k]
		switch {
		case given && req.FromArguments:
			return SourceArguments
		case given && (preset == nil || preset[k] != v):
			return SourceSettingsScreen
		default:
			src = SourceProfile
		}
	}
	return src
}

// PrefetchEcho is §1 T15: the mandatory pre-seal echo of P4:289-294 with this edition's fields
// (§4.4). The status lines and the effective record print the same values (S8-PF-04). A nil
// pointer or an empty string prints «(null)». The virtual fields (prefetch_n_cap) are GL-20's.
type PrefetchEcho struct {
	EffectivePrefetch           string `json:"effective_prefetch"`
	PrefetchRequest             string `json:"prefetch_request"`
	CatalogEvidence             string `json:"catalog_evidence"`
	CatalogActivation           string `json:"catalog_activation"`
	LauncherCandidateActivation string `json:"launcher_candidate_activation"`
	PrefetchIdentity            string `json:"prefetch_identity"`
	RequestedK                  *int64 `json:"requested_k"`
	RequestedN                  *int64 `json:"requested_n"`
	RequestedQD                 *int64 `json:"requested_qd"`
	PrefetchProvenance          string `json:"prefetch_provenance"`
	PrefetchSource              string `json:"prefetch_source"`
	PrefetchInitVersion         string `json:"prefetch_init_version"`
	OffReason                   string `json:"off_reason"`
	Warning                     string `json:"warning"`
	PrefetchNotice              string `json:"prefetch_notice"`
	// GL-20 (F-PF §8.2, T15): on a virtual repack with an ON candidate, the smallest N_cap over
	// the layers (the screen's value) and the list per layer (the log's line)
	PrefetchNCap         *int64  `json:"prefetch_n_cap"`
	PrefetchNCapPerLayer []int64 `json:"prefetch_n_cap_per_layer,omitempty"`
}

// echoOf is the echo of an effective config: the decision after the QD check (§6) and the QD
// the run passes.
func echoOf(pf Prefetch, qd int) PrefetchEcho {
	e := PrefetchEcho{EffectivePrefetch: pf.Echo, PrefetchRequest: pf.Request, CatalogEvidence: pf.Evidence,
		CatalogActivation: pf.Activation, LauncherCandidateActivation: pf.CandidateActivation,
		PrefetchIdentity: pf.Identity, PrefetchProvenance: pf.Provenance, PrefetchSource: pf.Source,
		PrefetchInitVersion: pf.InitVersion, OffReason: pf.OffReason, Warning: pf.Warning, PrefetchNotice: pf.Notice}
	if pf.On {
		k, n := pf.K, pf.N
		e.RequestedK, e.RequestedN = &k, &n
	}
	q := int64(qd)
	e.RequestedQD = &q
	if m, ok := nCapMin(pf.NCapPerLayer); ok {
		e.PrefetchNCap, e.PrefetchNCapPerLayer = &m, append([]int64(nil), pf.NCapPerLayer...)
	}
	return e
}

func nullStr(s string) string {
	if s == "" {
		return "(null)"
	}
	return s
}

func nullInt(p *int64) string {
	if p == nil {
		return "(null)"
	}
	return strconv.FormatInt(*p, 10)
}

// StatusLines is the echo as the status block prints it, in the PowerShell launcher's order
// and field names (SMD:10780-10796), plus the edition's prefetch_source and prefetch_notice.
func (e PrefetchEcho) StatusLines() []string {
	const pad = "                     "
	ls := []string{
		fmt.Sprintf("  effective_prefetch: %s   [catalog evidence=%s activation=%s]", e.EffectivePrefetch, e.CatalogEvidence, e.CatalogActivation),
		fmt.Sprintf("%sprefetch_request=%s catalog_evidence=%s catalog_activation=%s", pad, nullStr(e.PrefetchRequest), nullStr(e.CatalogEvidence), nullStr(e.CatalogActivation)),
		fmt.Sprintf("%slauncher_candidate_activation=%s prefetch_identity=%s", pad, nullStr(e.LauncherCandidateActivation), nullStr(e.PrefetchIdentity)),
		fmt.Sprintf("%srequested_k=%s requested_n=%s requested_qd=%s prefetch_n_cap=%s", pad, nullInt(e.RequestedK), nullInt(e.RequestedN), nullInt(e.RequestedQD), nullInt(e.PrefetchNCap)),
		fmt.Sprintf("%sprefetch_provenance=%s prefetch_source=%s prefetch_init_version=%s", pad, nullStr(e.PrefetchProvenance), nullStr(e.PrefetchSource), nullStr(e.PrefetchInitVersion)),
		fmt.Sprintf("%soff_reason=%s", pad, nullStr(e.OffReason)),
		fmt.Sprintf("%swarning=%s", pad, nullStr(e.Warning)),
		fmt.Sprintf("%sprefetch_notice=%s", pad, nullStr(e.PrefetchNotice)),
	}
	if e.EffectivePrefetch == "on" {
		ls = append(ls, pad+EchoCandidateOnly)
	}
	return ls
}
