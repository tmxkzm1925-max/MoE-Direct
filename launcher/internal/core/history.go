package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// The run history of decisions ② and ③ (DECISION_LEDGER.md:4178-4179): per model and
// settings, the last start-to-/health-ready duration and the memory measured at ready. It is
// the launcher's own file, <state>\history.json directly under the state root (F-LS 8.3;
// GL-FX2); the PowerShell launcher has no counterpart.

const historyFile = "history.json"
const historySchema = 1

// RunRecord is one key's last measurement.
type RunRecord struct {
	Key             string `json:"key"`
	ProfileID       string `json:"profile_id"`
	ModelPath       string `json:"model_path"`
	SettingsLine    string `json:"settings_line"`
	ReadyMillis     int64  `json:"ready_ms"`
	WorkingSetBytes int64  `json:"working_set_bytes"`
	PrivateBytes    int64  `json:"private_bytes"`
	At              string `json:"at"`
	Runs            int    `json:"runs"`
	// MaxWorkingSetBytes is the server's largest working set over the run, recorded at its end
	// (review item 13: the need); 0 until a run of this key has ended after GOL-FIX1.
	MaxWorkingSetBytes int64 `json:"max_working_set_bytes,omitempty"`
	// Start is what the start actually used (review item 2).
	Start *StartSettings `json:"start,omitempty"`
	// DecodeTokPerSec is the decode average of the last ended run of this key with at least one
	// answered request (conflict 10: the speed tile's «측정 전» is per model and settings); 0
	// until then.
	DecodeTokPerSec float64 `json:"decode_tok_per_sec,omitempty"`
}

// ReadyDuration is the recorded start-to-ready time.
func (r RunRecord) ReadyDuration() time.Duration {
	return time.Duration(r.ReadyMillis) * time.Millisecond
}

type historyDoc struct {
	SchemaVersion int                  `json:"schema_version"`
	Records       map[string]RunRecord `json:"records"`
	// Models is the last run per model file (review item 28), keyed by the lower-case path.
	Models map[string]ModelStats `json:"models,omitempty"`
	// Repacks is the last repack per model file and mode (conflict 9: the plan's duration).
	Repacks map[string]RepackRecord `json:"repacks,omitempty"`
	// CacheSaves is the last cache save per model and settings key (GL-07: the cache-save
	// bar's expectation, decision ③).
	CacheSaves map[string]CacheSaveRecord `json:"cache_saves,omitempty"`
}

// CacheSaveRecord is one key's last completed cache save: the span from the quit action to the
// save-completion response, and the tokens the response reported saved (GL-07).
type CacheSaveRecord struct {
	Key    string `json:"key"`
	Tokens int    `json:"tokens"`
	Millis int64  `json:"ms"`
	At     string `json:"at"`
	Saves  int    `json:"saves"`
}

// LastCacheSave is the last completed cache save of a key, if any.
func (l *Launcher) LastCacheSave(key string) (CacheSaveRecord, bool) {
	historyMu.Lock()
	defer historyMu.Unlock()
	r, ok := readHistory(l.historyPath()).CacheSaves[key]
	return r, ok
}

func (l *Launcher) recordCacheSave(r CacheSaveRecord) error {
	historyMu.Lock()
	defer historyMu.Unlock()
	d := readHistory(l.historyPath())
	if d.CacheSaves == nil {
		d.CacheSaves = map[string]CacheSaveRecord{}
	}
	r.Saves = d.CacheSaves[r.Key].Saves + 1
	d.CacheSaves[r.Key] = r
	return writeHistory(l.historyPath(), d)
}

func writeHistory(path string, d historyDoc) error {
	b, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(path, b)
}

var historyMu sync.Mutex

func (l *Launcher) historyPath() string { return filepath.Join(l.opt.StateDir, historyFile) }

func readHistory(path string) historyDoc {
	d := historyDoc{SchemaVersion: historySchema, Records: map[string]RunRecord{}}
	b, err := os.ReadFile(path)
	if err != nil {
		return d
	}
	var in historyDoc
	if json.Unmarshal(b, &in) != nil || in.SchemaVersion != historySchema || in.Records == nil {
		return d
	}
	return in
}

// HistoryFor is the record of one key, if any.
func (l *Launcher) HistoryFor(key string) (RunRecord, bool) {
	historyMu.Lock()
	defer historyMu.Unlock()
	r, ok := readHistory(l.historyPath()).Records[key]
	return r, ok
}

// LatestForModel is the newest record of a model file under any settings (the model list's
// "last measured" column; the exact need needs the settings, see Preflight).
func (l *Launcher) LatestForModel(modelPath string) (RunRecord, bool) {
	historyMu.Lock()
	defer historyMu.Unlock()
	var best RunRecord
	found := false
	for _, r := range readHistory(l.historyPath()).Records {
		if !strings.EqualFold(r.ModelPath, modelPath) {
			continue
		}
		if !found || r.At > best.At {
			best, found = r, true
		}
	}
	return best, found
}

func (l *Launcher) recordRun(r RunRecord) error {
	historyMu.Lock()
	defer historyMu.Unlock()
	d := readHistory(l.historyPath())
	if old, ok := d.Records[r.Key]; ok {
		r.Runs = old.Runs + 1
	} else {
		r.Runs = 1
	}
	d.Records[r.Key] = r
	return writeHistory(l.historyPath(), d)
}

// settingsKey names a model and its effective settings: the profile, the model file, the
// repack mode, the server argv without the port, the model path and the API name, and the
// engine variables that shape the load (budget, QD, prefetch). The port, the address, the
// --alias (review item 27), the slot save path (GL-15: a start has it, a preflight does not) and
// the metrics path do not change what the load costs, so they are left out (records written
// before --alias existed keep matching).
func settingsKey(profileID, modelPath, repackMode string, cfg *Config) (string, string) {
	var args []string
	for i := 0; i < len(cfg.Argv); i++ {
		if (cfg.Argv[i] == "--port" || cfg.Argv[i] == "-m" || cfg.Argv[i] == "--alias" || cfg.Argv[i] == argSlotSave) && i+1 < len(cfg.Argv) {
			i++
			continue
		}
		args = append(args, cfg.Argv[i])
	}
	var envs []string
	for _, k := range []string{EnvBudgetMB, EnvQD, EnvPrefetchK, EnvPrefetchN, EnvNoPrefetch} {
		if v, ok := cfg.Env[k]; ok {
			envs = append(envs, k+"="+v)
		}
	}
	sort.Strings(envs)
	key := fmt.Sprintf("v1|%s|%s|%s|%s|%s", profileID, strings.ToLower(modelPath), repackMode,
		strings.Join(args, " "), strings.Join(envs, " "))
	ctx, _ := argvValue(cfg.Argv, "-c")
	th, _ := argvValue(cfg.Argv, "-t")
	line := fmt.Sprintf("%s, ctx %s, threads %s, budget %d MB, QD %d, prefetch %s", repackMode, ctx, th,
		cfg.Budget.MB, cfg.QD, cfg.Prefetch.Echo)
	if cfg.Prefetch.On {
		line = fmt.Sprintf("%s, ctx %s, threads %s, budget %d MB, QD %d, prefetch K%d N%d", repackMode, ctx, th,
			cfg.Budget.MB, cfg.QD, cfg.Prefetch.K, cfg.Prefetch.N)
	}
	return key, line
}

// Progress is the load bar of decision ③.
type Progress struct {
	Elapsed     time.Duration
	Last        time.Duration // zero when HasLast is false
	HasLast     bool
	Fraction    float64 // elapsed / last, held at 0.99 while not ready; 1 at ready
	Held        bool    // the start runs longer than the last one (the bar holds at 99 %)
	SecondsOnly bool    // a first run of this key: show elapsed seconds only
}

// LoadProgress applies decision ③: fill = elapsed over the last measured duration of the
// same key; no record = seconds only; overrun = hold at 99 % (never a false 100 %).
func LoadProgress(elapsed time.Duration, last time.Duration, hasLast, ready bool) Progress {
	p := Progress{Elapsed: elapsed, Last: last, HasLast: hasLast}
	if ready {
		p.Fraction = 1
		p.SecondsOnly = !hasLast
		return p
	}
	if !hasLast || last <= 0 {
		p.SecondsOnly = true
		return p
	}
	f := float64(elapsed) / float64(last)
	if f >= 0.99 {
		f = 0.99
		p.Held = elapsed >= last
	}
	p.Fraction = f
	return p
}
