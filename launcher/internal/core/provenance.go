package core

import (
	"context"
	"sort"
)

// Conflicts 3, 7 and 10 of reviews/_gol_conflicts_8_user_20261010.md.

// --- conflict 7: the budget cap of a RAM refusal ---

// BudgetCapInfo is the largest budget in G at which a start would pass.
type BudgetCapInfo struct {
	GiB    int64
	OK     bool   // false: no budget inside the profile's bounds fits the free RAM
	Reason string // why not
}

// BudgetCap is conflict 7's «예산을 n G 이하로 내리면 시작 가능»: the refusal is budget > free
// RAM for the start (items 13 and 22: the RAM available now plus this launcher's serving
// server's share), so the cap is floor(free / 1024) G, clamped to the profile's budget_mb
// bounds; below the bounds' minimum no budget passes.
func BudgetCap(freeMiB int64, b Bound, bounded bool) BudgetCapInfo {
	c := BudgetCapInfo{GiB: max(0, freeMiB) / 1024, OK: true}
	if bounded && c.GiB*1024 > b.Max {
		c.GiB = b.Max / 1024
	}
	if bounded && c.GiB*1024 < b.Min {
		c.OK = false
		c.Reason = "even the smallest budget this model allows does not fit the free RAM"
	}
	if c.GiB <= 0 {
		c.OK = false
		if c.Reason == "" {
			c.Reason = "less than 1 GB of RAM is free"
		}
	}
	return c
}

// --- conflict 3: where a model's settings come from; conflict 10: custom ---

// ProvenanceState is the settings provenance of conflict 3.
type ProvenanceState uint8

const (
	ProvDefault        ProvenanceState = iota // «기본값»: the catalog defaults, unchanged
	ProvPreset                                // «내 설정»: the model's saved preset, unchanged
	ProvDefaultUnsaved                        // «기본값 · 저장 안 됨»: no preset, values changed
	ProvPresetUnsaved                         // «내 설정 · 저장 안 됨»: the preset, values changed
)

func (s ProvenanceState) String() string {
	return [...]string{"default", "preset", "default (unsaved)", "preset (unsaved)"}[s]
}

// Provenance is the provenance of the values a screen holds for a model.
type Provenance struct {
	State ProvenanceState
	// Changed are the keys whose value differs from the basis (the preset, or the defaults
	// without one): the values the screen marks as unsaved.
	Changed []string
	// Custom is conflict 10: at least one value differs from the catalog default (a saved
	// preset that differs counts too); CustomKeys are those keys.
	Custom     bool
	CustomKeys []string
	Preset     *Preset
}

// SettingsProvenance compares the screen's values (SettingKeys and repack_mode; "auto" or ""
// mean the default) with the catalog defaults and the model's saved preset.
func (l *Launcher) SettingsProvenance(ctx context.Context, modelPath, profileID string, values map[string]string) (Provenance, error) {
	cur, err := l.changedValues(profileID, modelPath, values)
	if err != nil {
		return Provenance{}, err
	}
	pv := Provenance{}
	basis := map[string]string{}
	pre, ps := l.LoadPreset(ctx, modelPath, profileID)
	if ps == PresetApplies {
		pv.Preset = pre
		for k, v := range pre.Values {
			basis[k] = v
		}
	}
	keys := map[string]bool{}
	for k := range cur {
		keys[k] = true
	}
	for k := range basis {
		keys[k] = true
	}
	for k := range keys {
		if cur[k] != basis[k] {
			pv.Changed = append(pv.Changed, k)
		}
	}
	for k := range cur {
		pv.CustomKeys = append(pv.CustomKeys, k)
	}
	sort.Strings(pv.Changed)
	sort.Strings(pv.CustomKeys)
	pv.Custom = len(cur) > 0
	switch {
	case pv.Preset != nil && len(pv.Changed) > 0:
		pv.State = ProvPresetUnsaved
	case pv.Preset != nil:
		pv.State = ProvPreset
	case len(pv.Changed) > 0:
		pv.State = ProvDefaultUnsaved
	default:
		pv.State = ProvDefault
	}
	return pv, nil
}

// SpeedFor is conflict 10's speed fact per model and settings: the decode average of the last
// ended run of this settings key (PreflightResult.SettingsKey); false = «측정 전» (no run of
// these settings has answered a request yet).
func (l *Launcher) SpeedFor(key string) (float64, bool) {
	r, ok := l.HistoryFor(key)
	if !ok || r.DecodeTokPerSec <= 0 {
		return 0, false
	}
	return r.DecodeTokPerSec, true
}
