package core

import (
	"context"
	"os"
	"path/filepath"
	"strings"
)

// Selection kinds of Resolve-ProfileSelection (PS:3009-3092).
const (
	SelectionPinned   = "pinned"
	SelectionUnpinned = "unpinned"
	SelectionMismatch = "mismatch"
	SelectionTemplate = "template"
)

// archTemplateFamilies is ARCH_TEMPLATE_FAMILIES (PS:823).
var archTemplateFamilies = []string{"gpt-oss", "qwen35moe", "deepseek2"}

// Selection is the identified profile.
type Selection struct {
	Kind       string
	Profile    *Profile
	Shas       []string
	Candidates []*Profile
}

// structuralCandidates is Get-StructuralProfileCandidates (PS:2856-2885).
func structuralCandidates(cat *Catalog, set *ShardSet) ([]*Profile, *LaunchError) {
	nLayer, okL := set.ArchLong(".block_count")
	nExp, okE := set.ArchLong(".expert_count")
	nExpU, okU := set.ArchLong(".expert_used_count")
	var out []*Profile
	for _, p := range cat.Profiles {
		if p.Identify.Arch != set.Arch {
			continue
		}
		if !okL || p.Identify.NLayer != nLayer || !okE || p.Identify.NExpert != nExp || !okU || p.Identify.NExpertUsed != nExpU {
			continue
		}
		srcs, err := expectSources(p.ExpectPath)
		if err != nil {
			return nil, stop(StatusFailGateCatalog, "expect unreadable: %s - %v", p.ExpectPath, err)
		}
		if srcs == nil {
			return nil, stop(StatusFailGateCatalog, "expect sources[] missing: %s", p.ExpectPath)
		}
		if len(srcs) != len(set.Shards) {
			continue
		}
		ok := true
		var sum int64
		for i, s := range srcs {
			so, _ := asObj(s)
			fb, isInt := jNonNegInt(so["file_bytes"])
			if !isInt || fb != set.Shards[i].FileBytes {
				ok = false
				break
			}
			sum += fb
		}
		if !ok || sum != set.TotalBytes {
			continue
		}
		out = append(out, p)
	}
	return out, nil
}

// pinMatch is Test-ProfilePinMatch (PS:2896-2903).
func pinMatch(pin, actual []string) bool {
	if len(pin) == 0 || len(pin) != len(actual) {
		return false
	}
	for i := range pin {
		if !strings.EqualFold(pin[i], actual[i]) {
			return false
		}
	}
	return true
}

// selectProfile is Select-Profile (PS:2905-2939) without its console question: with more than
// one match the caller's choice decides (wantID), and without one it is the non-interactive
// refusal.
func selectProfile(matches []*Profile, wantID string) (*Profile, *LaunchError) {
	if len(matches) == 0 {
		return nil, stop(StatusFailModelPath, "unsupported GGUF: no catalog profile matches the header fingerprint")
	}
	if len(matches) > 1 {
		for _, p := range matches {
			if wantID != "" && p.ID == wantID {
				return p, nil
			}
		}
		return nil, stop(StatusFailModelPath, "ambiguous shard set: multiple catalog profiles match (non-interactive)")
	}
	return matches[0], nil
}

// resolveProfileSelection is Resolve-ProfileSelection. Source digests come from the caches
// first; a miss is hashed here, as at the PowerShell identify step (never at list time).
// wantID stands in for the console question of Select-Profile when several unpinned
// profiles match.
func (l *Launcher) resolveProfileSelection(ctx context.Context, cat *Catalog, set *ShardSet,
	templateAllowed bool, wantID string) (*Selection, error) {
	cands, le := structuralCandidates(cat, set)
	if le != nil {
		return nil, le
	}
	var pinned, unpinned []*Profile
	for _, p := range cands {
		if len(p.SourceShardsSha256) > 0 {
			pinned = append(pinned, p)
		} else {
			unpinned = append(unpinned, p)
		}
	}
	needShas := len(pinned) > 0 || (templateAllowed && len(unpinned) == 0)
	var shas []string
	if needShas {
		paths := make([]string, len(set.Shards))
		for i, h := range set.Shards {
			paths[i] = h.Path
		}
		d, err := l.shardDigests(ctx, paths, true)
		if err != nil {
			return nil, stop(StatusFailModelPath, "source shard hashing failed, the profile pin cannot be decided: %v", err)
		}
		shas = d.Sha256
	}
	var hits []*Profile
	for _, p := range pinned {
		if pinMatch(p.SourceShardsSha256, shas) {
			hits = append(hits, p)
		}
	}
	if len(hits) > 1 {
		ids := make([]string, len(hits))
		for i, p := range hits {
			ids[i] = p.ID
		}
		return nil, stop(StatusFailGateCatalog, "two catalog profiles are pinned to the same source bytes: %s", strings.Join(ids, ", "))
	}
	if len(hits) == 1 {
		return &Selection{Kind: SelectionPinned, Profile: hits[0], Shas: shas, Candidates: cands}, nil
	}
	if len(unpinned) > 0 {
		p, le := selectProfile(unpinned, wantID)
		if le != nil {
			return nil, le
		}
		return &Selection{Kind: SelectionUnpinned, Profile: p, Shas: shas, Candidates: cands}, nil
	}
	if len(pinned) > 0 && !templateAllowed {
		p, le := selectProfile(pinned, wantID)
		if le != nil {
			return nil, le
		}
		return &Selection{Kind: SelectionMismatch, Profile: p, Shas: shas, Candidates: cands}, nil
	}
	if !templateAllowed {
		_, le := selectProfile(nil, wantID)
		return nil, le
	}
	return &Selection{Kind: SelectionTemplate, Shas: shas, Candidates: cands}, nil
}

// archTemplateOn is Resolve-ArchTemplate (PS:5904-5928) without the CLI switches: the stored
// preference (read only) when valid, OFF when it is discarded, else ARCH_TEMPLATE_DEFAULT 'on'.
//
// Conflict 14 (2026-10-10 05:32 KST): the template stays always on, with no toggle; the stored
// preference is no longer read (archTemplatePrefOn keeps the old reader for reference).
func (l *Launcher) archTemplateOn() (bool, string) { return true, "always on (conflict 14)" }

// TemplateArchitectures are the architectures the template covers (ARCH_TEMPLATE_FAMILIES
// PS:823): a model of one of them outside the catalog is listed as a template model.
func TemplateArchitectures() []string { return append([]string(nil), archTemplateFamilies...) }

func (l *Launcher) archTemplatePrefOn() (bool, string) {
	path := filepath.Join(l.opt.StateDir, "arch_template_pref.json")
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		return true, "default"
	}
	v, err := readJSONFileStrict(path)
	if err != nil {
		return false, "pref_discarded"
	}
	o, ok := asObj(v)
	if !ok || denyUnknownKeys(o, []string{"schema_version", "arch_template"}, nil, "pref") != nil {
		return false, "pref_discarded"
	}
	if sv, ok := jNonNegInt(o["schema_version"]); !ok || sv != 1 {
		return false, "pref_discarded"
	}
	s, _ := o["arch_template"].(string)
	switch s {
	case "on":
		return true, "preference"
	case "off":
		return false, "preference"
	}
	return false, "pref_discarded"
}

// templateAdmissible is Test-TemplateAdmissible (PS:5930-5934).
func templateAdmissible(on bool, arch string) bool {
	return on && contains(archTemplateFamilies, arch)
}
