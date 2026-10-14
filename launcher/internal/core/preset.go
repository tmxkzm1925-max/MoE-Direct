package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Model settings presets (review item 3 of reviews/_gol_review_8_user_20261010.md, ctrl+s;
// U-19: the per-model saved settings are the presets, with no separate preset path): one file
// per model, <state>\presets\<model>.json (F-LS 6.3, 8.2: «presets» directly under
// %LOCALAPPDATA%\MoE-Direct since GL-16, no draft subfolder), one profile per
// model. The key is the model name and the file's identity: for a catalog model
// the shard sha256 set (from the digest caches), outside the catalog, or when a digest is not
// cached, the file name and size. The values are the changed items only, the repack mode
// among them. The PowerShell preset file presets.user.json is never touched.
//
// GL-16 (S8-LA-35; F-LS 5.3): a stored value that fails its check (or a key outside the open
// settings) means the whole file is not used: the defaults apply, the settings say so with the
// saved-settings text of 11.4, and the launcher log gets one line per file. A model outside
// the catalog (a template structure, GL-14) keys its file by file name and size and checks it
// against its structure's defaults, the same before and after the derive plan ran.

const presetSchema = 1

// Preset is one model's saved settings.
type Preset struct {
	SchemaVersion int               `json:"schema_version"`
	Model         string            `json:"model"`                // the catalog profile id, or the file name without extension
	ProfileID     string            `json:"profile_id,omitempty"` // "" outside the catalog
	Sha256        []string          `json:"sha256,omitempty"`     // the shard digests when all were known at the save
	FileName      string            `json:"file_name"`
	FileBytes     int64             `json:"file_bytes"`
	Values        map[string]string `json:"values"` // changed items only: SettingKeys and repack_mode (the stored key list)
	SavedAt       string            `json:"saved_at"`
}

// presetKeys are the keys a preset may hold.
func presetKey(k string) bool { return k == "repack_mode" || contains(SettingKeys, k) }

// PresetName is the file name stem of a model's preset.
func PresetName(profileID, modelPath string) string {
	n := ModelAlias(profileID, modelPath)
	return strings.Map(func(r rune) rune {
		if strings.ContainsRune(`<>:"/\|?*`, r) || r < 32 {
			return '_'
		}
		return r
	}, n)
}

// PresetDir is the folder of the per-model saved settings, <state>\presets.
func (l *Launcher) PresetDir() string { return filepath.Join(l.opt.StateDir, StatePresetsDir) }

// PresetPath is where a model's preset lives.
func (l *Launcher) PresetPath(profileID, modelPath string) string {
	return filepath.Join(l.PresetDir(), PresetName(profileID, modelPath)+".json")
}

// presetRows are the settings a model's preset is checked against: a catalog profile's rows,
// else, for a model outside the catalog, its template structure's defaults (GL-14).
func (l *Launcher) presetRows(profileID, modelPath string) ([]SettingRow, error) {
	if profileID != "" {
		return l.Settings(profileID)
	}
	arch := ""
	if h, err := readGGUFHeader(representativePath(modelPath), wantLabel()); err == nil {
		arch, _ = h.Meta["general.architecture"].(string)
	}
	rows, ok := StructureSettings(arch)
	if !ok {
		return nil, fmt.Errorf("no catalog profile and no template structure: no settings to save")
	}
	return rows, nil
}

// rowBounds are the bounds of the bounded rows.
func rowBounds(rows []SettingRow) map[string]Bound {
	b := map[string]Bound{}
	for _, r := range rows {
		if r.Bounded {
			b[r.Key] = Bound{Min: r.Min, Max: r.Max}
		}
	}
	return b
}

// checkPresetValue is the check of one stored value against the model's rows.
func checkPresetValue(rows []SettingRow, k, v string) (string, error) {
	if k == "repack_mode" {
		v = strings.ToLower(strings.TrimSpace(v))
		for _, r := range rows {
			if r.Key == "repack_mode" && contains(r.Options, v) {
				return v, nil
			}
		}
		return "", fmt.Errorf("repack_mode must be one of the modes this model takes")
	}
	if !presetKey(k) {
		return "", fmt.Errorf("unknown setting %q", k)
	}
	if fam, np := settingsNotPorted[k]; np {
		return "", fmt.Errorf("%s: not ported (%s)", k, fam)
	}
	v2, err := checkSetting(k, v, rowBounds(rows))
	if err == nil && isPairKey(k) {
		for _, r := range rows { // K within its row's 1..min(16, n_expert) (F-PF §9 C6)
			if r.Key == k && r.Bounded {
				if n, _ := strconv.ParseInt(v2, 10, 64); n < r.Min || n > r.Max {
					return "", fmt.Errorf("%s", refuseKAboveText(n, r.Max))
				}
			}
		}
	}
	return v2, err
}

// fileKey is a model file's identity for a preset.
func (l *Launcher) fileKey(ctx context.Context, modelPath string) (shas []string, name string, bytes int64) {
	rep := representativePath(modelPath)
	name = filepath.Base(rep)
	bytes, _, _ = shardDisplayAggregate(rep)
	if d, err := l.shardDigests(ctx, siblingPaths(rep), false); err == nil && d.Complete {
		shas = d.Sha256
	}
	return shas, name, bytes
}

// sameFile is the key rule: the digests when both sides have them, else name and size.
func (p *Preset) sameFile(shas []string, name string, bytes int64) bool {
	if len(p.Sha256) > 0 && len(shas) > 0 {
		return pinMatch(p.Sha256, shas)
	}
	return strings.EqualFold(p.FileName, name) && p.FileBytes == bytes
}

// changedValues keeps the values that differ from the defaults; "auto" and "" mean the
// default. Each value is checked against the model's rows (presetRows).
func (l *Launcher) changedValues(profileID, modelPath string, values map[string]string) (map[string]string, error) {
	defaults := map[string]string{"repack_mode": RepackPacked}
	rows, err := l.presetRows(profileID, modelPath)
	if err != nil {
		return nil, err
	}
	for _, r := range rows {
		defaults[r.Key] = r.Default
	}
	out := map[string]string{}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := strings.TrimSpace(values[k])
		if !presetKey(k) {
			return nil, fmt.Errorf("preset: unknown setting %q", k)
		}
		if v == "" || strings.EqualFold(v, "auto") || v == defaults[k] {
			continue
		}
		nv, err := checkPresetValue(rows, k, v)
		if err != nil {
			return nil, fmt.Errorf("preset: %s: %v", k, err)
		}
		if nv == defaults[k] {
			continue
		}
		out[k] = nv
	}
	return out, nil
}

// SavePreset saves a model's settings (the changed items only) and returns what was written.
func (l *Launcher) SavePreset(ctx context.Context, modelPath, profileID string, values map[string]string) (*Preset, error) {
	ch, err := l.changedValues(profileID, modelPath, values)
	if err != nil {
		return nil, err
	}
	shas, name, bytes := l.fileKey(ctx, modelPath)
	p := &Preset{SchemaVersion: presetSchema, Model: ModelAlias(profileID, modelPath), ProfileID: profileID,
		Sha256: shas, FileName: name, FileBytes: bytes, Values: ch, SavedAt: l.os.now().UTC().Format(time.RFC3339)}
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return nil, err
	}
	path := l.PresetPath(profileID, modelPath)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := writeFileAtomic(path, b); err != nil {
		return nil, err
	}
	l.diag.write("PRESET_SAVED", map[string]any{"path": path, "values": ch})
	return p, nil
}

// PresetState says what LoadPreset found.
type PresetState uint8

const (
	PresetNone       PresetState = iota // no file
	PresetApplies                       // saved for this very file
	PresetOtherFile                     // saved for another file of the same name (not applied)
	PresetUnreadable                    // the file is not a preset of this schema (not applied)
	PresetInvalid                       // a stored value or key failed its check (not applied; S8-LA-35)
)

// NoticePresetNotUsed is the saved-settings path's text (F-LS 11.4) for a preset not applied
// because it cannot be read or a value failed its check.
var NoticePresetNotUsed = CustomValueTemplates[PathSavedSettings].EN

// presetLog keeps the discard log to one line per preset file (its path, size and time).
type presetLog struct {
	mu   sync.Mutex
	seen map[string]bool
}

// discarded writes the one log line of a preset file not used.
func (l *Launcher) discarded(path, key, value, why string) {
	st, err := os.Stat(path)
	id := path
	if err == nil {
		id = fmt.Sprintf("%s|%d|%d", strings.ToLower(path), st.Size(), st.ModTime().UnixNano())
	}
	l.presets.mu.Lock()
	if l.presets.seen == nil {
		l.presets.seen = map[string]bool{}
	}
	first := !l.presets.seen[id]
	l.presets.seen[id] = true
	l.presets.mu.Unlock()
	if first {
		l.diag.write("PRESET_DISCARDED", map[string]any{"path": path, "key": key, "value": value, "why": why,
			"text": NoticePresetNotUsed})
	}
}

// LoadPreset reads a model's preset and says whether it belongs to this file.
func (l *Launcher) LoadPreset(ctx context.Context, modelPath, profileID string) (*Preset, PresetState) {
	path := l.PresetPath(profileID, modelPath)
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, PresetNone
	}
	var p Preset
	if json.Unmarshal(b, &p) != nil || p.SchemaVersion != presetSchema || p.Values == nil {
		l.discarded(path, "", "", "not a preset of this schema")
		return nil, PresetUnreadable
	}
	for k, v := range p.Values {
		if !presetKey(k) {
			l.discarded(path, k, v, "not a changeable setting")
			return &p, PresetInvalid
		}
	}
	shas, name, bytes := l.fileKey(ctx, modelPath)
	if p.ProfileID != profileID || !p.sameFile(shas, name, bytes) {
		return &p, PresetOtherFile
	}
	rows, err := l.presetRows(profileID, modelPath)
	if err != nil {
		l.discarded(path, "", "", err.Error())
		return &p, PresetInvalid
	}
	keys := make([]string, 0, len(p.Values))
	for k := range p.Values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, err := checkPresetValue(rows, k, p.Values[k]); err != nil {
			l.discarded(path, k, p.Values[k], err.Error())
			return &p, PresetInvalid
		}
	}
	// F-PF §9 C3 for a stored pair: K and N together or not at all (S8-PF-10).
	if _, k := p.Values["prefetch_k"]; k {
		if _, n := p.Values["prefetch_n"]; !n {
			l.discarded(path, "prefetch_k", p.Values["prefetch_k"], refusePairText)
			return &p, PresetInvalid
		}
	} else if _, n := p.Values["prefetch_n"]; n {
		l.discarded(path, "prefetch_n", p.Values["prefetch_n"], refusePairText)
		return &p, PresetInvalid
	}
	return &p, PresetApplies
}

// DeletePreset removes a model's preset (no error when there is none).
func (l *Launcher) DeletePreset(profileID, modelPath string) error {
	err := os.Remove(l.PresetPath(profileID, modelPath))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// mergePreset lays the request over the preset: a key the request names wins; the preset
// fills the rest (review item 3: preflight and start take the saved values as overrides).
func mergePreset(req map[string]string, preset map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range preset {
		if k != "repack_mode" {
			out[k] = v
		}
	}
	for k, v := range req {
		out[k] = v
	}
	return out
}
