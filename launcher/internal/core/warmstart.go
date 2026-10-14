package core

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf16"
)

// GL-15 (version 8): warm start, the PowerShell launcher's region 14b (LS 13 = WS-1; value
// authority WARMSTART_SPEC v0.5, AUTOSAVE_SPEC v0.1; PS:8919-10645): the mode and the warm-start
// switches, the kv directory and its two-phase housekeeping, the bindings, the eligibility of
// every stored generation, the --slot-save-path argument, the restore ladder after ready (with
// the recovery restart), the save before a requested stop, and the autosave while idle. The
// slot requests and file names are PowerShell's, byte for byte (S8-LA-33; the PowerShell
// selftest's records, launcher_selftest.ps1:5912-7362). No status and no exit code is new:
// every failure is degraded (LS 13-3), except the recovery restart's own failures, which reuse
// fail_server_start and fail_teardown.
//
// The PowerShell launcher is one start per process; the Go launcher holds many starts in one
// process, so every per-process latch of WarmstartCtx is per start here (one WarmStart per start).
// PowerShell's console lines ([kv] ...) are the launcher log's records (the diag kinds below).

// The constants of PS:1036-1139.
const (
	kvDirName           = "kv"
	kvCanonicalData     = "slot0.kv"
	kvMetaSuffix        = ".meta.json"
	kvCanonicalMeta     = kvCanonicalData + kvMetaSuffix
	kvGCWarnMatches     = 16
	kvProfileRetention  = 4
	kvMetaSchemaVersion = 1
	kvSlotID            = 0
	kvAutoGenA          = "slot0.auto.a.kv"
	kvAutoGenB          = "slot0.auto.b.kv"
	kvOriginAutosave    = "autosave"
	kvOriginStop        = "stop"

	kvReasonSidecarMissing  = "sidecar_missing"
	kvReasonKVFileMissing   = "kv_file_missing"
	kvReasonMetaParseFailed = "meta_parse_failed"
	kvReasonFileIntegrity   = "file_integrity_broken"
	kvReasonOffUser         = "off_user"
	kvReasonOffMode         = "off_mode"
	kvReasonRecoveryCold    = "recovery_cold"
	kvReasonUnavailable     = "eligibility_unavailable"
	kvReasonRestoreFailed   = "restore_failed"

	kvDiagSaveFailed    = "warmstart_save_failed"
	kvDiagRestoreFailed = "warmstart_restore_failed"
	kvDiagGCFailed      = "warmstart_gc_failed"

	// The warm-start states of the LS 13-2 truth table (Get-WarmstartState PS:8993).
	WarmHardOff    = "hard_off"
	WarmSoftOff    = "soft_off"
	WarmLatchedOff = "latched_off"
	WarmOn         = "on"
)

var (
	kvGCPatterns      = []string{"slot0.kv.tmp.*", "slot0.kv.stale.*", "slot0.kv.meta.json.tmp.*", "slot0.kv.meta.json.stale.*"}
	kvAutoGenerations = []string{kvAutoGenA, kvAutoGenB}
	// Structural deadlines (PS:1064-1066, :1106); variables so a test can shorten them.
	kvSaveTimeout    = 900 * time.Second
	kvRestoreTimeout = 900 * time.Second
	kvEraseTimeout   = 120 * time.Second
	kvSlotsTimeout   = 30 * time.Second
	// KV_AUTOSAVE_DEFAULT_MIN (PS:1100): the switch is on/off, so «on» is the default period.
	kvAutosavePeriod = 5 * time.Minute
	// A-4c exclusion tables (PS:1117-1139), frozen.
	kvSemanticsArgvDrop = []struct {
		flag  string
		arity int
	}{{"--port", 1}, {"--host", 1}, {"-m", 1}, {"-t", 1}, {argSlotSave, 1}, {"--no-webui", 0}}
	kvSemanticsEnvDrop = []string{"MOE_DIRECT_METRICS", "MOE_DIRECT_DIR", "MOE_DIRECT_EXPECTS_DIR",
		"MOE_DIRECT_PREFETCH_K", "MOE_DIRECT_PREFETCH_N", "MOE_NO_PREFETCH", "MOE_DIRECT_BUDGET_MB", "MOE_DIRECT_QD"}
)

// WarmStart is one start's warm-start context (New-WarmstartState PS:8925) and its verdict.
type WarmStart struct {
	Mode        string // product | hard_off | arch_template (A-6; OA-1's derived gate)
	Override    string // on | off: the warm-start switch (OQ-6 (a))
	LatchedOff  bool   // A-1 directory latch
	LatchReason string
	ProfileID   string
	Dir         string // <state>\kv\<profile>
	Root        string // the bundle root (bindings)
	ManifestSha string // the repack gate's manifest sha256 (bindings)
	ModelPaths  []string
	Eligible    bool
	Reason      string
	StatusText  string // off(mode) | off(user) | eligible | cold(<reason>) | restored(<n> tokens)
	Detail      string
	Selected    string // the generation the restore asks for
	Origin      string
	SavedAt     string
	Restored    bool
	NRestored   int64
	Autosave    bool // the autosave switch (OQ-6 (a)); the warm-start state gates it again

	mu                sync.Mutex
	saveMu            sync.Mutex // one slot save at a time (the stop's save waits for an autosave)
	bundleSha         string
	modelShas         []string
	modelShasTried    bool
	facts             map[string]kvFacts
	restoreLatched    bool
	restoreDone       bool
	unavailableWarned bool
	autoNext          string
	autoTokens        int64
	autoHasTokens     bool
	autoClock         time.Time
	autoCount         int
	autoStopped       bool
	autoWarned        bool
	tmpAfterJoin      []string
	staleAfterStop    []string
	recoveries        int
	seenTokens        int64 // the slot's tokens last seen (restore, /slots): the stop save's bar
}

// warmupRestoredReason is the warmup's skip after a restored slot (WARMSTART A-3, PS:14580).
const warmupRestoredReason = "slot state restored (WARMSTART A-3): a launcher warmup would overwrite the restored prefix"

// Text is the status text (Get-WarmstartStatusText); empty for a start without warm start.
func (w *WarmStart) Text() string {
	if w == nil {
		return ""
	}
	return w.StatusText
}

// State is Get-WarmstartState.
func (w *WarmStart) State() string {
	switch {
	case w == nil || w.Mode != "product":
		return WarmHardOff
	case w.Override == "off":
		return WarmSoftOff
	case w.LatchedOff:
		return WarmLatchedOff
	}
	return WarmOn
}

func (w *WarmStart) active() bool { return w.State() == WarmOn }

func kvStatusText(eligible bool, reason string) string {
	switch {
	case eligible:
		return "eligible"
	case reason == kvReasonOffUser:
		return "off(user)"
	case reason == kvReasonOffMode:
		return "off(mode)"
	}
	return "cold(" + reason + ")"
}

func (w *WarmStart) verdict(eligible bool, reason, detail string) {
	w.Eligible, w.Reason, w.Detail, w.StatusText = eligible, reason, detail, kvStatusText(eligible, reason)
}

func (l *Launcher) kvRootDir() string { return filepath.Join(l.opt.StateDir, kvDirName) }

// newWarmStart is Initialize-Warmstart (PS:9086) with the switches bound as
// Build-EffectiveConfig binds them (PS:8578-8590): hardOff is PowerShell's -Smoke (the check run
// due at this start), derived a profile from the structure templates (Test-DerivedWarmstartGate
// answers false, PS:9068). Only a product run touches the kv tree: the generation sweep here.
func (l *Launcher) newWarmStart(profileID string, derived, hardOff bool, sw Switches) *WarmStart {
	w := &WarmStart{Mode: "product", Override: onOff(sw.WarmStart), Autosave: sw.Autosave, ProfileID: profileID,
		Dir: filepath.Join(l.kvRootDir(), profileID), facts: map[string]kvFacts{}, Selected: kvCanonicalData, Origin: kvOriginStop}
	if hardOff {
		w.Mode = WarmHardOff
	}
	if derived && w.Mode == "product" {
		w.Mode = "arch_template"
		l.diag.write("WARMSTART_DERIVED_GATE", map[string]any{"eligible": false,
			"unmet": []string{"manifest_binding_carries_template", "e2e_cold_and_autosave_restore_passed"}})
	}
	l.diag.write("WARMSTART_MODE", map[string]any{"mode": w.Mode, "smoke": hardOff, "profile": profileID, "derived": derived,
		"dir": w.Dir, "warmstart": w.Override, "autosave": onOff(sw.Autosave)})
	if w.Mode != "product" {
		w.verdict(false, kvReasonOffMode, "")
		return w
	}
	l.kvGenerationSweep()
	return w
}

// --- A-4b housekeeping (PS:9134-9365) ---

func (l *Launcher) kvGCDegraded(reason string, data map[string]any) {
	d := map[string]any{"reason": reason}
	for k, v := range data {
		d[k] = v
	}
	l.diag.write(kvDiagGCFailed, d)
}

// kvGenerationSweep is Invoke-KvGenerationSweep: every profile's tmp and stale generations go.
func (l *Launcher) kvGenerationSweep() {
	root := l.kvRootDir()
	if st, err := os.Stat(root); err != nil || !st.IsDir() {
		l.diag.write("WARMSTART_GC", map[string]any{"root": root, "phase": "generations", "state": "absent", "matched": 0, "removed_generations": 0})
		return
	}
	dirs, err := os.ReadDir(root)
	if err != nil {
		l.kvGCDegraded("profile enumeration failed: "+err.Error(), map[string]any{"root": root, "phase": "generations"})
		return
	}
	var failures []string
	matched, removed := 0, 0
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		m, r, f := kvRemoveGenerationFiles(filepath.Join(root, d.Name()))
		matched, removed, failures = matched+m, removed+r, append(failures, f...)
	}
	if matched > kvGCWarnMatches {
		l.diag.write("WARMSTART_GC_WARN", map[string]any{"root": root, "matched": matched, "threshold": kvGCWarnMatches,
			"note": "unusually many leftover generations (crash loop signal)"})
	}
	l.diag.write("WARMSTART_GC", map[string]any{"root": root, "phase": "generations", "matched": matched, "removed_generations": removed, "failures": failures})
	if len(failures) > 0 {
		l.kvGCDegraded("some leftover generations could not be removed", map[string]any{"root": root, "phase": "generations", "failures": failures})
	}
}

// kvRemoveGenerationFiles is Remove-KvGenerationFiles: the four patterns, ordinal name order.
func kvRemoveGenerationFiles(dir string) (matched, removed int, failures []string) {
	es, err := os.ReadDir(dir)
	if err != nil {
		return 0, 0, []string{"enumerate " + dir + ": " + err.Error()}
	}
	var names []string
	for _, e := range es {
		if e.IsDir() {
			continue
		}
		for _, pat := range kvGCPatterns {
			if ok, _ := filepath.Match(pat, e.Name()); ok {
				names = append(names, e.Name())
				break
			}
		}
	}
	sort.Strings(names)
	for _, n := range names {
		if err := os.Remove(filepath.Join(dir, n)); err != nil {
			failures = append(failures, n+": "+err.Error())
			continue
		}
		removed++
	}
	return len(names), removed, failures
}

// kvProfileTicks is Get-KvProfileCanonicalTicks: the newest complete pair of a profile.
func kvProfileTicks(dir string) int64 {
	best := int64(math.MinInt64)
	for _, n := range append([]string{kvCanonicalData}, kvAutoGenerations...) {
		d, e1 := os.Stat(filepath.Join(dir, n))
		m, e2 := os.Stat(filepath.Join(dir, n+kvMetaSuffix))
		if e1 != nil || e2 != nil || d.IsDir() || m.IsDir() {
			continue
		}
		t := d.ModTime().UnixNano()
		if mt := m.ModTime().UnixNano(); mt > t {
			t = mt
		}
		if t > best {
			best = t
		}
	}
	return best
}

// kvProfileCap is Invoke-KvProfileCapAdjust: at most four profile directories, the current one
// pinned, the rest recovered oldest first (ordinal key, a profile without a pair first).
func (l *Launcher) kvProfileCap(current string) {
	root := l.kvRootDir()
	if st, err := os.Stat(root); err != nil || !st.IsDir() {
		l.diag.write("WARMSTART_GC", map[string]any{"root": root, "phase": "profile_cap", "state": "absent", "removed_profiles": []string{}, "profiles_after": 0})
		return
	}
	es, err := os.ReadDir(root)
	if err != nil {
		l.kvGCDegraded("profile enumeration failed: "+err.Error(), map[string]any{"root": root, "phase": "profile_cap"})
		return
	}
	type cand struct{ name, key string }
	var cands []cand
	n := 0
	for _, e := range es {
		if !e.IsDir() {
			continue
		}
		n++
		if e.Name() == current {
			continue
		}
		t := kvProfileTicks(filepath.Join(root, e.Name()))
		rank := "0" + strings.Repeat("0", 19)
		if t > 0 {
			rank = "1" + fmt.Sprintf("%019d", t)
		}
		cands = append(cands, cand{e.Name(), rank + "|" + e.Name()})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].key < cands[j].key })
	var removedProfiles, failures []string
	excess := n - kvProfileRetention
	for _, c := range cands {
		if excess <= 0 {
			break
		}
		if err := os.RemoveAll(filepath.Join(root, c.name)); err != nil {
			failures = append(failures, "profile "+c.name+": "+err.Error())
			continue
		}
		removedProfiles = append(removedProfiles, c.name)
		excess--
	}
	final := 0
	if es, err := os.ReadDir(root); err == nil {
		for _, e := range es {
			if e.IsDir() {
				final++
			}
		}
	} else {
		final = -1
		failures = append(failures, "final profile count failed: "+err.Error())
	}
	l.diag.write("WARMSTART_GC", map[string]any{"root": root, "phase": "profile_cap", "removed_profiles": removedProfiles,
		"profiles_after": final, "retention": kvProfileRetention, "failures": failures})
	if final > kvProfileRetention || len(failures) > 0 {
		l.kvGCDegraded(fmt.Sprintf("the profile cap could not be fully enforced (profiles_after=%d)", final),
			map[string]any{"root": root, "phase": "profile_cap", "profiles_after": final, "failures": failures})
	}
}

// confirmDir is Confirm-WarmstartDirectory with the A-1 latch (Set-WarmstartOffLatch).
func (l *Launcher) warmConfirmDir(w *WarmStart, stage string) bool {
	if w.State() != WarmOn {
		return false
	}
	if err := os.MkdirAll(w.Dir, 0o755); err != nil {
		w.LatchedOff, w.LatchReason = true, "kv directory could not be created: "+err.Error()
		l.diag.write("WARMSTART_OFF_LATCH", map[string]any{"stage": stage, "reason": w.LatchReason,
			"effect": "no --slot-save-path, no restore, no save"})
		return false
	}
	l.diag.write("WARMSTART_DIR", map[string]any{"stage": stage, "dir": w.Dir})
	return true
}

// warmConfigure is the warm-start part of Build-EffectiveConfig (PS:8578-8598, :8819): the slot
// save path when the state is on and the directory exists, the profile cap, then the
// eligibility of the stored generations against this configuration.
func (l *Launcher) warmConfigure(ctx context.Context, w *WarmStart, cfg *Config, root, manifestSha string, modelPaths []string) {
	cfg.Argv = removeArgvPair(cfg.Argv, argSlotSave)
	dirOK := l.warmConfirmDir(w, "config")
	if w.Mode == "product" {
		l.kvProfileCap(w.ProfileID)
	}
	if dirOK {
		cfg.Argv = setArgvValue(cfg.Argv, argSlotSave, strings.TrimRight(w.Dir, `\`)+`\`)
	}
	// Set-WarmstartBindings (PS:9122)
	w.Root, w.ManifestSha, w.ModelPaths = root, strings.ToLower(manifestSha), modelPaths
	l.warmEligibility(ctx, w, cfg)
}

// --- A-4c semantics hash (PS:9367-9455) ---

// kvJSONString is ConvertTo-KvJsonString: \" and \\, every control or non-ASCII UTF-16 unit
// as \uXXXX in lower-case hex, no short escapes.
func kvJSONString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, u := range utf16.Encode([]rune(s)) {
		switch {
		case u == '"':
			b.WriteString(`\"`)
		case u == '\\':
			b.WriteString(`\\`)
		case u < 32 || u > 126:
			fmt.Fprintf(&b, `\u%04x`, u)
		default:
			b.WriteByte(byte(u))
		}
	}
	b.WriteByte('"')
	return b.String()
}

// kvSemanticsProjection is Get-KvSemanticsProjection: the frozen argv and env exclusions and the
// OS bootstrap allowlist out.
func kvSemanticsProjection(argv []string, env map[string]string) ([]string, map[string]string) {
	var out []string
	for i := 0; i < len(argv); {
		a := argv[i]
		dropped := false
		for _, d := range kvSemanticsArgvDrop {
			if a == d.flag {
				i += 1 + d.arity
				dropped = true
				break
			}
		}
		if !dropped {
			out = append(out, a)
			i++
		}
	}
	oe := map[string]string{}
	for k, v := range env {
		skip := false
		for _, d := range append(append([]string(nil), kvSemanticsEnvDrop...), envBootstrapAllowlist...) {
			if strings.EqualFold(k, d) {
				skip = true
				break
			}
		}
		if !skip {
			oe[k] = v
		}
	}
	return out, oe
}

// kvSemanticsSha256 is Get-KvSemanticsSha256 over ConvertTo-KvCanonicalJson.
func kvSemanticsSha256(argv []string, env map[string]string) string {
	pa, pe := kvSemanticsProjection(argv, env)
	var b strings.Builder
	b.WriteString(`{"schema":1,"argv":[`)
	for i, a := range pa {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(kvJSONString(a))
	}
	b.WriteString(`],"env":{`)
	keys := make([]string, 0, len(pe))
	for k := range pe {
		keys = append(keys, k)
	}
	sort.Strings(keys) // ordinal
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(kvJSONString(k) + ":" + kvJSONString(pe[k]))
	}
	b.WriteString(`}}`)
	return sha256Hex([]byte(b.String()))
}

// --- binding inputs (PS:9467-9535) ---

func (l *Launcher) kvBundleSha(w *WarmStart) (string, bool) {
	if w.bundleSha != "" {
		return w.bundleSha, true
	}
	if w.Root == "" {
		return "", false
	}
	h, err := fileSha256(filepath.Join(w.Root, BundleManifestName))
	if err != nil {
		return "", false
	}
	w.bundleSha = strings.ToLower(h)
	return w.bundleSha, true
}

// kvModelShas is Get-KvModelShardShas: the shard digests (the identify caches first), once per
// start, with the failure latched.
func (l *Launcher) kvModelShas(ctx context.Context, w *WarmStart) ([]string, bool) {
	if w.modelShas != nil {
		return w.modelShas, true
	}
	if w.modelShasTried || len(w.ModelPaths) == 0 {
		return nil, false
	}
	w.modelShasTried = true
	d, err := l.shardDigests(ctx, w.ModelPaths, true)
	if err != nil || !d.Complete {
		return nil, false
	}
	w.modelShas = d.Sha256
	return w.modelShas, true
}

// kvFileHeader is Get-KvFileHeaderFields: the first 8 bytes, magic hex and the LE version.
func kvFileHeader(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, fmt.Errorf("state file header read failed: %v", err)
	}
	defer f.Close()
	buf := make([]byte, 8)
	n, _ := f.Read(buf)
	if n != 8 {
		return "", 0, fmt.Errorf("state file is shorter than its 8 byte header")
	}
	return fmt.Sprintf("%02x%02x%02x%02x", buf[0], buf[1], buf[2], buf[3]), int64(binary.LittleEndian.Uint32(buf[4:])), nil
}

// kvArgvLong is Get-KvArgvLong: 0 when the flag is absent or not a non-negative integer.
func kvArgvLong(argv []string, flag string) int64 {
	v, ok := argvValue(argv, flag)
	if !ok {
		return 0
	}
	n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// --- A-4 sidecar (PS:9538-9770) ---

// kvMeta is a sidecar, in PowerShell's field order.
type kvMeta struct {
	MetaSchemaVersion             int64    `json:"meta_schema_version"`
	ProfileID                     string   `json:"profile_id"`
	ModelShardsSha256             []string `json:"model_shards_sha256"`
	RepackManifestSha256          string   `json:"repack_manifest_sha256"`
	EngineBundleSha256            string   `json:"engine_bundle_sha256"`
	EffectiveStateSemanticsSha256 string   `json:"effective_state_semantics_sha256"`
	NCtx                          int64    `json:"n_ctx"`
	NParallel                     int64    `json:"n_parallel"`
	LlamaStateSeqMagic            string   `json:"llama_state_seq_magic"`
	LlamaStateSeqVersion          int64    `json:"llama_state_seq_version"`
	NTokens                       int64    `json:"n_tokens"`
	NBytes                        int64    `json:"n_bytes"`
	KVFileSha256                  string   `json:"kv_file_sha256"`
	GenerationID                  string   `json:"generation_id"`
	SavedAt                       string   `json:"saved_at"`
	Origin                        string   `json:"origin,omitempty"`
}

// readKvSidecar is Read-KvSidecar: strict, every field checked.
func readKvSidecar(path string) (*kvMeta, string) {
	v, err := readJSONFileStrict(path)
	if err != nil {
		return nil, err.Error()
	}
	o, ok := asObj(v)
	if !ok {
		return nil, "the sidecar is not a JSON object"
	}
	m := &kvMeta{}
	ints := map[string]*int64{"meta_schema_version": &m.MetaSchemaVersion, "n_ctx": &m.NCtx, "n_parallel": &m.NParallel,
		"llama_state_seq_version": &m.LlamaStateSeqVersion, "n_tokens": &m.NTokens, "n_bytes": &m.NBytes}
	for _, k := range []string{"meta_schema_version", "n_ctx", "n_parallel", "llama_state_seq_version", "n_tokens", "n_bytes"} {
		n, ok := jNonNegInt(o[k])
		if !ok {
			return nil, k + " missing or not a non-negative integer"
		}
		*ints[k] = n
	}
	strs := map[string]*string{"profile_id": &m.ProfileID, "llama_state_seq_magic": &m.LlamaStateSeqMagic,
		"generation_id": &m.GenerationID, "saved_at": &m.SavedAt}
	for _, k := range []string{"profile_id", "llama_state_seq_magic", "generation_id", "saved_at"} {
		s, ok := jNonEmptyStr(o[k])
		if !ok {
			return nil, k + " missing or not a non-empty string"
		}
		*strs[k] = s
	}
	shas := map[string]*string{"repack_manifest_sha256": &m.RepackManifestSha256, "engine_bundle_sha256": &m.EngineBundleSha256,
		"effective_state_semantics_sha256": &m.EffectiveStateSemanticsSha256, "kv_file_sha256": &m.KVFileSha256}
	for _, k := range []string{"repack_manifest_sha256", "engine_bundle_sha256", "effective_state_semantics_sha256", "kv_file_sha256"} {
		if !isSha256Hex(o[k]) {
			return nil, k + " missing or not a 64 hex digest"
		}
		*shas[k] = strings.ToLower(o[k].(string))
	}
	arr, ok := jArr(o["model_shards_sha256"])
	if !ok || len(arr) == 0 {
		return nil, "model_shards_sha256 missing or not a non-empty array"
	}
	for _, s := range arr {
		if !isSha256Hex(s) {
			return nil, "model_shards_sha256 entry is not a 64 hex digest"
		}
		m.ModelShardsSha256 = append(m.ModelShardsSha256, strings.ToLower(s.(string)))
	}
	if s, ok := jNonEmptyStr(o["origin"]); ok {
		m.Origin = s
	}
	return m, ""
}

// savedAtNanos is Get-KvSavedAtTicks: an unusable value sorts as the oldest possible.
func (m *kvMeta) savedAtNanos() int64 {
	t, err := time.Parse(time.RFC3339Nano, m.SavedAt)
	if err != nil {
		return math.MinInt64
	}
	return t.UTC().UnixNano()
}

func (m *kvMeta) origin() string {
	if m.Origin != "" {
		return m.Origin
	}
	return kvOriginStop
}

func (w *WarmStart) dataPath(name string) string { return filepath.Join(w.Dir, name) }
func (w *WarmStart) metaPath(name string) string { return filepath.Join(w.Dir, name+kvMetaSuffix) }

// kvCandidateOrder is Get-KvCandidateOrder: newest saved_at first, ties in the fixed order
// (canonical, a, b).
func (w *WarmStart) kvCandidateOrder() []string {
	names := append([]string{kvCanonicalData}, kvAutoGenerations...)
	ticks := map[string]int64{}
	for _, n := range names {
		ticks[n] = math.MinInt64
		if m, _ := readKvSidecar(w.metaPath(n)); m != nil {
			ticks[n] = m.savedAtNanos()
		}
	}
	sort.SliceStable(names, func(i, j int) bool { return ticks[names[i]] > ticks[names[j]] })
	return names
}

// kvRemovePair is Remove-KvCanonicalPair: class (a), both halves go.
func (l *Launcher) kvRemovePair(w *WarmStart, why, name string) {
	var removed, failed []string
	for _, p := range []string{w.dataPath(name), w.metaPath(name)} {
		if !fileExists(p) {
			continue
		}
		if err := os.Remove(p); err != nil {
			failed = append(failed, p+": "+err.Error())
			continue
		}
		removed = append(removed, p)
	}
	l.diag.write("WARMSTART_CANONICAL_DISCARDED", map[string]any{"why": why, "generation": name, "removed": removed, "failed": failed})
}

type kvFacts struct {
	ok     bool
	reason string
	detail string
	meta   *kvMeta
	length int64
	sha    string
}

// kvFileFacts is Measure-KvFileFacts with Get-KvFileFacts' cache (once per start and generation).
func (l *Launcher) kvFileFacts(ctx context.Context, w *WarmStart, name string) kvFacts {
	if f, ok := w.facts[name]; ok {
		return f
	}
	f := l.measureKvFileFacts(ctx, w, name)
	w.facts[name] = f
	return f
}

func (l *Launcher) measureKvFileFacts(ctx context.Context, w *WarmStart, name string) kvFacts {
	dp, mp := w.dataPath(name), w.metaPath(name)
	if !fileExists(mp) {
		return kvFacts{reason: kvReasonSidecarMissing}
	}
	if !fileExists(dp) {
		return kvFacts{reason: kvReasonKVFileMissing}
	}
	meta, why := readKvSidecar(mp)
	if meta == nil {
		l.kvRemovePair(w, "meta_parse_failed: "+why, name)
		return kvFacts{reason: kvReasonMetaParseFailed, detail: why}
	}
	st, err := os.Stat(dp)
	if err != nil {
		return kvFacts{reason: kvReasonUnavailable, detail: "length query failed: " + err.Error()}
	}
	if st.Size() != meta.NBytes {
		l.kvRemovePair(w, "file_integrity_broken: n_bytes != real length", name)
		return kvFacts{reason: kvReasonFileIntegrity, detail: fmt.Sprintf("n_bytes=%d real=%d", meta.NBytes, st.Size())}
	}
	l.diag.write("WARMSTART_HASH_NOTICE", map[string]any{"what": "verifying stored slot state", "mb": int64(math.Round(float64(st.Size()) / (1 << 20)))})
	h, err := hashFileCtx(ctx, dp)
	if err != nil {
		return kvFacts{reason: kvReasonUnavailable, detail: err.Error()}
	}
	if h != meta.KVFileSha256 {
		l.kvRemovePair(w, "file_integrity_broken: kv_file_sha256 != real file hash", name)
		return kvFacts{reason: kvReasonFileIntegrity, detail: "kv_file_sha256 mismatch"}
	}
	magic, ver, err := kvFileHeader(dp)
	if err != nil {
		l.kvRemovePair(w, "file_integrity_broken: "+err.Error(), name)
		return kvFacts{reason: kvReasonFileIntegrity, detail: err.Error()}
	}
	if magic != strings.ToLower(meta.LlamaStateSeqMagic) || ver != meta.LlamaStateSeqVersion {
		l.kvRemovePair(w, "file_integrity_broken: state header disagrees with the sidecar", name)
		return kvFacts{reason: kvReasonFileIntegrity, detail: "state header mismatch"}
	}
	return kvFacts{ok: true, meta: meta, length: st.Size(), sha: h}
}

// kvCompareBinding is Compare-KvConfigBinding: the cheap fields first, the shard hashes last.
func (l *Launcher) kvCompareBinding(ctx context.Context, w *WarmStart, m *kvMeta, cfg *Config) (bool, string, string, string) {
	switch {
	case m.MetaSchemaVersion != kvMetaSchemaVersion:
		return false, "meta_schema_version_mismatch", "", ""
	case m.ProfileID != w.ProfileID:
		return false, "profile_id_mismatch", "", ""
	case m.RepackManifestSha256 != w.ManifestSha:
		return false, "repack_manifest_sha256_mismatch", "", ""
	}
	bundle, ok := l.kvBundleSha(w)
	if !ok {
		return false, kvReasonUnavailable, "bundle manifest hash unavailable", ""
	}
	if m.EngineBundleSha256 != bundle {
		return false, "engine_bundle_sha256_mismatch", "", ""
	}
	if m.NCtx != kvArgvLong(cfg.Argv, "-c") {
		return false, "n_ctx_mismatch", "", ""
	}
	if m.NParallel != kvArgvLong(cfg.Argv, "-np") {
		return false, "n_parallel_mismatch", "", ""
	}
	sem := kvSemanticsSha256(cfg.Argv, cfg.Env)
	if m.EffectiveStateSemanticsSha256 != sem {
		return false, "effective_state_semantics_sha256_mismatch", "now=" + sem, ""
	}
	shas, ok := l.kvModelShas(ctx, w)
	if !ok {
		return false, kvReasonUnavailable, "model shard hashes unavailable", ""
	}
	if len(shas) != len(m.ModelShardsSha256) {
		return false, "model_shards_sha256_mismatch", "", ""
	}
	for i := range shas {
		if strings.ToLower(shas[i]) != m.ModelShardsSha256[i] {
			return false, "model_shards_sha256_mismatch", "", ""
		}
	}
	return true, "", "", sem
}

// warmEligibility is Update-WarmstartEligibility: every generation judged by the same machine,
// the newest eligible one elected, the damage check exhaustive.
func (l *Launcher) warmEligibility(ctx context.Context, w *WarmStart, cfg *Config) {
	switch w.State() {
	case WarmHardOff:
		w.verdict(false, kvReasonOffMode, "")
		return
	case WarmSoftOff:
		w.verdict(false, kvReasonOffUser, "")
		return
	case WarmLatchedOff:
		w.verdict(false, kvReasonUnavailable, w.LatchReason)
		return
	}
	if w.restoreDone {
		return
	}
	var summary []map[string]any
	var canon map[string]any
	type best struct {
		name    string
		facts   kvFacts
		sem     string
		origin  string
		savedAt string
	}
	var win *best
	warn := func(detail string) {
		if !w.unavailableWarned {
			w.unavailableWarned = true
			l.diag.write("WARMSTART_UNAVAILABLE", map[string]any{"line": "the stored slot state could not be evaluated (" + detail + "); this run starts cold."})
		}
	}
	for _, name := range w.kvCandidateOrder() {
		isCanon := name == kvCanonicalData
		f := l.kvFileFacts(ctx, w, name)
		if !f.ok {
			v := map[string]any{"generation": name, "eligible": false, "reason": f.reason, "detail": f.detail, "stage": "file"}
			if isCanon {
				canon = v
			} else if f.reason != kvReasonSidecarMissing && f.reason != kvReasonKVFileMissing {
				summary = append(summary, v)
			}
			if f.reason == kvReasonUnavailable {
				warn(f.detail)
			}
			continue
		}
		ok, reason, detail, sem := l.kvCompareBinding(ctx, w, f.meta, cfg)
		if !ok {
			v := map[string]any{"generation": name, "eligible": false, "reason": reason, "detail": detail, "stage": "config"}
			if isCanon {
				canon = v
			} else {
				summary = append(summary, v)
			}
			if reason == kvReasonUnavailable {
				warn(detail)
			}
			continue
		}
		if isCanon {
			canon = map[string]any{"generation": name, "eligible": true, "reason": "eligible"}
		} else {
			summary = append(summary, map[string]any{"generation": name, "eligible": true, "saved_at": f.meta.SavedAt})
		}
		if win == nil {
			win = &best{name, f, sem, f.meta.origin(), f.meta.SavedAt}
		}
	}
	if win == nil {
		if canon == nil {
			canon = map[string]any{"reason": kvReasonSidecarMissing, "detail": "", "stage": "file"}
		}
		r, _ := canon["reason"].(string)
		d, _ := canon["detail"].(string)
		w.verdict(false, r, d)
		l.diag.write("WARMSTART_ELIGIBILITY", map[string]any{"eligible": false, "reason": r, "detail": d, "stage": canon["stage"], "candidates": summary})
		return
	}
	w.verdict(true, "eligible", "")
	w.Selected, w.Origin, w.SavedAt = win.name, win.origin, win.savedAt
	l.diag.write("WARMSTART_ELIGIBILITY", map[string]any{"eligible": true, "reason": "eligible", "n_bytes": win.facts.length,
		"kv_file_sha256": win.facts.sha, "effective_state_semantics_sha256": win.sem, "generation_id": win.facts.meta.GenerationID,
		"selected": win.name, "origin": win.origin, "saved_at": win.savedAt, "candidates": summary})
}

// --- A-2 / A-3 wire (PS:9873-9918) ---

// kvSlotAction is Invoke-KvSlotAction: POST /slots/0?action=<a>, the body {"filename":...}.
func kvSlotAction(ctx context.Context, cfg *Config, action, filename string, timeout time.Duration) httpResult {
	url := fmt.Sprintf("http://%s:%d/slots/%d?action=%s", cfg.Host, cfg.Port, kvSlotID, action)
	body := ""
	if filename != "" {
		body = `{"filename":` + kvJSONString(filename) + `}`
	}
	return httpJSON(ctx, "POST", url, body, timeout)
}

// kvSlotVerdict is Test-KvSlotResponse: a 200 alone proves nothing (A-2 (4)).
func kvSlotVerdict(r httpResult, filename string, positive ...string) (ok, delivered bool, reason string, vals map[string]int64) {
	if !r.ok {
		why := r.reason
		if why == "" {
			why = "no response"
		}
		return false, r.status != 0, why, nil
	}
	if r.status != 200 {
		return false, true, fmt.Sprintf("http status %d", r.status), nil
	}
	v, err := parseJSONStrict(r.body)
	if err != nil {
		return false, true, "malformed response body - " + err.Error(), nil
	}
	o, _ := asObj(v)
	if id, ok := jNonNegInt(o["id_slot"]); !ok || id != kvSlotID {
		return false, true, "id_slot missing or not slot 0", nil
	}
	if echo, ok := jNonEmptyStr(o["filename"]); !ok || echo != filename {
		return false, true, "filename echo does not match the requested name", nil
	}
	vals = map[string]int64{}
	for _, f := range positive {
		n, ok := jNonNegInt(o[f])
		if !ok {
			return false, true, f + " missing or not an integer", nil
		}
		if n <= 0 {
			return false, true, f + " is 0", nil
		}
		vals[f] = n
	}
	return true, true, "", vals
}

// warmRestore is Invoke-WarmstartRestore and its ladder; recovery reports rung 4 (the erase
// failed too), which the caller answers with the recovery restart.
func (l *Launcher) warmRestore(ctx context.Context, w *WarmStart, cfg *Config) (restored, recovery bool) {
	if w == nil || !w.active() || w.restoreLatched || !w.Eligible {
		return false, false
	}
	w.saveMu.Lock() // a stop that arrives during the restore saves after it
	defer w.saveMu.Unlock()
	w.restoreLatched = true
	name := w.Selected
	if name == "" {
		name = kvCanonicalData
	}
	ok, delivered, reason, vals := kvSlotVerdict(kvSlotAction(ctx, cfg, "restore", name, kvRestoreTimeout), name, "n_restored", "n_read")
	if ok {
		w.mu.Lock()
		w.Restored, w.NRestored, w.restoreDone = true, vals["n_restored"], true
		w.autoTokens, w.autoHasTokens, w.seenTokens = vals["n_restored"], true, vals["n_restored"]
		if contains(kvAutoGenerations, name) {
			w.autoNext = kvOtherGeneration(name)
		}
		w.mu.Unlock()
		w.verdict(true, "restored", "")
		w.StatusText = fmt.Sprintf("restored(%d tokens)", vals["n_restored"])
		l.diag.write("WARMSTART_RESTORE_OK", map[string]any{"n_restored": vals["n_restored"], "n_read": vals["n_read"],
			"generation": name, "origin": w.Origin, "saved_at": w.SavedAt})
		return true, false
	}
	l.diag.write(kvDiagRestoreFailed, map[string]any{"reason": reason, "delivered": delivered})
	e := kvSlotAction(ctx, cfg, "erase", "", kvEraseTimeout)
	if e.ok && e.status == 200 {
		w.restoreDone = true
		w.verdict(false, kvReasonRestoreFailed, reason)
		l.diag.write("WARMSTART_RESTORE_COLD", map[string]any{"reason": kvReasonRestoreFailed, "erase": "ok"})
		return false, false
	}
	why := "erase request failed"
	if e.reason != "" {
		why = e.reason
	}
	l.diag.write(kvDiagRestoreFailed, map[string]any{"reason": why, "stage": "erase", "next": "recovery restart"})
	return false, true
}

// warmRecovered is the verdict after the recovery restart (Invoke-WarmstartRecoveryRestart (5)).
func (l *Launcher) warmRecovered(w *WarmStart, n int) {
	w.restoreDone = true
	w.verdict(false, kvReasonRecoveryCold, "")
	l.diag.write("WARMSTART_RECOVERY", map[string]any{"stage": "ready", "incarnation": n, "reason": kvReasonRecoveryCold})
}

// kvRecoveryPath is New-KvRecoveryPath: <base>.recovery<n><ext>, the first absent one.
func kvRecoveryPath(path string, n int) (string, error) {
	dir, ext := filepath.Dir(path), filepath.Ext(path)
	base := strings.TrimSuffix(filepath.Base(path), ext)
	for i := 0; i < 64; i++ {
		c := filepath.Join(dir, fmt.Sprintf("%s.recovery%d%s", base, n+i, ext))
		if _, err := os.Stat(c); os.IsNotExist(err) {
			return c, nil
		}
	}
	return "", fmt.Errorf("no free recovery path derived from %s", path)
}

// --- A-2 save (PS:9993-10208) ---

// warmSaveAllowed is Test-WarmstartSaveAllowed: a requested normal stop of a live, ready child.
func warmSaveAllowed(w *WarmStart, ready bool, pending Status, alive bool) bool {
	return w != nil && w.active() && ready && alive && (pending == StatusOK || pending == StatusOKSmoke)
}

// kvGenerationID is Get-KvGenerationId: UTC yyyyMMddTHHmmssfff, «Z_», the launcher PID.
func kvGenerationID(t time.Time) string {
	u := t.UTC()
	return u.Format("20060102T150405") + fmt.Sprintf("%03d", u.Nanosecond()/1e6) + "Z_" + strconv.Itoa(os.Getpid())
}

func kvSavedAt(t time.Time) string { return t.UTC().Format("2006-01-02T15:04:05.0000000Z") }

func (l *Launcher) kvSaveDegraded(reason string, data map[string]any) {
	d := map[string]any{"reason": reason}
	for k, v := range data {
		d[k] = v
	}
	l.diag.write(kvDiagSaveFailed, d)
}

func (l *Launcher) kvRemoveBestEffort(path, why string) {
	if !fileExists(path) {
		return
	}
	if err := os.Remove(path); err != nil {
		l.diag.write(kvDiagSaveFailed, map[string]any{"reason": "leftover cleanup failed: " + err.Error(), "path": path, "why": why})
	}
}

// kvFreeOK is A-2 (7): the previous size plus 10 % must fit the kv volume.
func (l *Launcher) kvFreeOK(w *WarmStart, need int64, degraded func(string)) bool {
	if need <= 0 {
		return true
	}
	free, _, _, err := l.os.volumeFree(w.Dir)
	if err != nil {
		degraded("kv volume free space query failed: " + err.Error())
		return false
	}
	needMb := int64(math.Ceil(float64(need) * 1.1 / (1 << 20)))
	if free < needMb {
		degraded(fmt.Sprintf("kv volume free space %d MB is below the %d MB this save needs", free, needMb))
		return false
	}
	return true
}

// kvBuildMeta is the sidecar of a verified generation (the save and the autosave share it).
func (l *Launcher) kvBuildMeta(ctx context.Context, w *WarmStart, cfg *Config, path string, saved int64, gen string) (*kvMeta, string) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, "saved file length query failed: " + err.Error()
	}
	if st.Size() <= 0 {
		return nil, "the saved file is empty on disk"
	}
	l.diag.write("WARMSTART_HASH_NOTICE", map[string]any{"what": "checksumming the saved slot state", "mb": int64(math.Round(float64(st.Size()) / (1 << 20)))})
	h, err := hashFileCtx(ctx, path)
	if err != nil {
		return nil, err.Error()
	}
	magic, ver, err := kvFileHeader(path)
	if err != nil {
		return nil, err.Error()
	}
	shas, ok1 := l.kvModelShas(ctx, w)
	bundle, ok2 := l.kvBundleSha(w)
	if !ok1 || !ok2 {
		return nil, "model shard or bundle binding inputs are unavailable"
	}
	return &kvMeta{MetaSchemaVersion: kvMetaSchemaVersion, ProfileID: w.ProfileID, ModelShardsSha256: append([]string(nil), shas...),
		RepackManifestSha256: w.ManifestSha, EngineBundleSha256: bundle, EffectiveStateSemanticsSha256: kvSemanticsSha256(cfg.Argv, cfg.Env),
		NCtx: kvArgvLong(cfg.Argv, "-c"), NParallel: kvArgvLong(cfg.Argv, "-np"), LlamaStateSeqMagic: magic, LlamaStateSeqVersion: ver,
		NTokens: saved, NBytes: st.Size(), KVFileSha256: h, GenerationID: gen, SavedAt: kvSavedAt(l.os.now())}, ""
}

// kvPublish moves tmp onto final, the previous final kept as stale for the end of teardown
// ([System.IO.File]::Replace in PowerShell; Move-FileAtomic without a previous one).
func (w *WarmStart) kvPublish(tmp, final, stale string) error {
	if fileExists(final) {
		if err := os.Rename(final, stale); err != nil {
			return err
		}
		w.staleAfterStop = append(w.staleAfterStop, stale)
	}
	if err := os.Rename(tmp, final); err != nil {
		if fileExists(stale) && !fileExists(final) {
			_ = os.Rename(stale, final)
		}
		return err
	}
	return nil
}

// warmSave is Invoke-WarmstartSave: the save before the stop signal (A-2), data first, then the
// sidecar that carries its hash. It returns the saved tokens for the cache-save bar (GL-07);
// an error is a degraded save that the stop never turns into a status.
func (l *Launcher) warmSave(ctx context.Context, w *WarmStart, cfg *Config) (int, error) {
	w.saveMu.Lock()
	defer w.saveMu.Unlock()
	gen := kvGenerationID(l.os.now())
	tmpName := kvCanonicalData + ".tmp." + gen
	tmpPath := w.dataPath(tmpName)
	metaTmp := filepath.Join(w.Dir, kvCanonicalMeta+".tmp."+gen)
	dataStale := filepath.Join(w.Dir, kvCanonicalData+".stale."+gen)
	metaStale := filepath.Join(w.Dir, kvCanonicalMeta+".stale."+gen)
	fail := func(reason string, data map[string]any) (int, error) {
		l.kvSaveDegraded(reason, data)
		return 0, fmt.Errorf("%s", reason)
	}
	if prev, _ := readKvSidecar(w.metaPath(kvCanonicalData)); prev != nil {
		why := ""
		if !l.kvFreeOK(w, prev.NBytes, func(s string) { why = s }) {
			return fail(why, nil)
		}
	}
	l.diag.write("WARMSTART_SAVE", map[string]any{"line": "saving the slot state before shutdown...", "filename": tmpName})
	ok, delivered, reason, vals := kvSlotVerdict(kvSlotAction(ctx, cfg, "save", tmpName, kvSaveTimeout), tmpName, "n_saved", "n_written")
	if !ok {
		if delivered {
			l.kvRemoveBestEffort(tmpPath, "save failed (response received)")
		} else {
			w.tmpAfterJoin = append(w.tmpAfterJoin, tmpPath)
		}
		return fail(reason, map[string]any{"filename": tmpName, "delivered": delivered})
	}
	meta, why := l.kvBuildMeta(ctx, w, cfg, tmpPath, vals["n_saved"], gen)
	if meta == nil {
		l.kvRemoveBestEffort(tmpPath, "save verification failed")
		return fail(why, nil)
	}
	if err := w.kvPublish(tmpPath, w.dataPath(kvCanonicalData), dataStale); err != nil {
		l.kvRemoveBestEffort(tmpPath, "data publish failed")
		return fail("publishing the state file failed: "+err.Error(), nil)
	}
	b, _ := json.MarshalIndent(meta, "", "    ")
	if err := os.WriteFile(metaTmp, b, 0o644); err != nil {
		l.kvRemoveBestEffort(metaTmp, "meta publish failed")
		return fail("publishing the sidecar failed: "+err.Error(), nil)
	}
	if m, why := readKvSidecar(metaTmp); m == nil {
		l.kvRemoveBestEffort(metaTmp, "meta publish failed")
		return fail("publishing the sidecar failed: read-back parse failed: "+why, nil)
	}
	if err := w.kvPublish(metaTmp, w.metaPath(kvCanonicalData), metaStale); err != nil {
		l.kvRemoveBestEffort(metaTmp, "meta publish failed")
		return fail("publishing the sidecar failed: "+err.Error(), nil)
	}
	l.diag.write("WARMSTART_SAVE_OK", map[string]any{"generation_id": gen, "n_tokens": vals["n_saved"], "n_bytes": meta.NBytes, "kv_file_sha256": meta.KVFileSha256})
	return int(vals["n_saved"]), nil
}

// warmTeardownCleanup is Complete-WarmstartTeardownCleanup: the deferred tmp files once the
// child is joined, and this transaction's stale pair.
func (l *Launcher) warmTeardownCleanup(w *WarmStart) {
	if w == nil {
		return
	}
	w.mu.Lock()
	tmp, stale := w.tmpAfterJoin, w.staleAfterStop
	w.tmpAfterJoin, w.staleAfterStop = nil, nil
	w.mu.Unlock()
	for _, p := range tmp {
		l.kvRemoveBestEffort(p, "save timed out")
	}
	for _, p := range stale {
		l.kvRemoveBestEffort(p, "superseded generation")
	}
}

// --- LS 13-8 autosave (PS:10210-10518) ---

type kvSlots struct {
	ok        bool
	reason    string
	idle      bool
	hasTokens bool
	tokens    int64
}

// kvSlotsSnapshot is Get-KvSlotsSnapshot: GET /slots, the idle and change gates' input.
func kvSlotsSnapshot(ctx context.Context, cfg *Config) kvSlots {
	r := httpJSON(ctx, "GET", fmt.Sprintf("http://%s:%d/slots", cfg.Host, cfg.Port), "", kvSlotsTimeout)
	if !r.ok {
		why := r.reason
		if why == "" {
			why = "no response"
		}
		return kvSlots{reason: why}
	}
	if r.status != 200 {
		return kvSlots{reason: fmt.Sprintf("http status %d", r.status)}
	}
	v, err := parseJSONStrict(r.body)
	if err != nil {
		return kvSlots{reason: "malformed /slots body - " + err.Error()}
	}
	items, ok := jArr(v)
	if !ok {
		items = []any{v}
	}
	busy, seen := false, 0
	var slot jobj
	for _, it := range items {
		o, _ := asObj(it)
		p, ok := jBool(o["is_processing"])
		if !ok {
			continue
		}
		seen++
		if p {
			busy = true
		}
		if id, ok := jNonNegInt(o["id"]); ok && id == kvSlotID {
			slot = o
		}
	}
	if seen == 0 {
		return kvSlots{reason: "the /slots answer carries no slot state"}
	}
	if slot == nil {
		return kvSlots{reason: fmt.Sprintf("the /slots answer carries no slot %d", kvSlotID)}
	}
	n, has := jNonNegInt(slot["n_prompt_tokens"])
	return kvSlots{ok: true, idle: !busy, hasTokens: has, tokens: n}
}

func kvOtherGeneration(name string) string {
	for _, n := range kvAutoGenerations {
		if n != name {
			return n
		}
	}
	return name
}

// selectAutoGeneration is Select-AutosaveGeneration: the older generation (an absent or unreadable
// sidecar is the oldest), unless the next one is already pinned.
func (w *WarmStart) selectAutoGeneration() string {
	if w.autoNext != "" {
		return w.autoNext
	}
	pick, pickT := "", int64(0)
	for _, n := range kvAutoGenerations {
		t := int64(math.MinInt64)
		if m, _ := readKvSidecar(w.metaPath(n)); m != nil {
			t = m.savedAtNanos()
		}
		if pick == "" || t < pickT {
			pick, pickT = n, t
		}
	}
	w.autoNext = pick
	return pick
}

// kvAutosaveSizeHint is Get-KvAutosaveSizeHint: the target's size, the other's, the stop save's.
func (w *WarmStart) kvAutosaveSizeHint(name string) int64 {
	for _, n := range []string{name, kvOtherGeneration(name), kvCanonicalData} {
		if m, _ := readKvSidecar(w.metaPath(n)); m != nil && m.NBytes > 0 {
			return m.NBytes
		}
	}
	return 0
}

func (l *Launcher) kvAutosaveDegraded(w *WarmStart, reason string, data map[string]any) {
	d := map[string]any{"reason": reason, "stage": "autosave"}
	for k, v := range data {
		d[k] = v
	}
	l.diag.write(kvDiagSaveFailed, d)
	w.autoWarned = true
}

// autosaveWrite is Invoke-AutosaveWrite: save API, validated success, checksum, sidecar; the
// generation under its final name (the alternation is the atomicity).
func (l *Launcher) autosaveWrite(ctx context.Context, w *WarmStart, cfg *Config, tokens int64) bool {
	w.saveMu.Lock()
	defer w.saveMu.Unlock()
	name := w.selectAutoGeneration()
	dp, mp := w.dataPath(name), w.metaPath(name)
	why := ""
	if !l.kvFreeOK(w, w.kvAutosaveSizeHint(name), func(s string) { why = s }) {
		l.kvAutosaveDegraded(w, strings.Replace(why, "this save needs", "this autosave needs", 1), nil)
		return false
	}
	l.diag.write("WARMSTART_AUTOSAVE", map[string]any{"line": fmt.Sprintf("autosave: the server is idle, storing the current %d token state...", tokens), "generation": name})
	t0 := l.os.now()
	ok, delivered, reason, vals := kvSlotVerdict(kvSlotAction(ctx, cfg, "save", name, kvSaveTimeout), name, "n_saved", "n_written")
	saveMs := l.os.now().Sub(t0).Milliseconds()
	if !ok {
		if delivered {
			l.kvRemoveBestEffort(dp, "autosave failed (response received)")
			l.kvRemoveBestEffort(mp, "autosave failed (response received)")
		} else {
			w.tmpAfterJoin = append(w.tmpAfterJoin, dp, mp)
			w.autoStopped = true
		}
		l.kvAutosaveDegraded(w, reason, map[string]any{"generation": name, "delivered": delivered})
		return false
	}
	gen := kvGenerationID(l.os.now())
	meta, why := l.kvBuildMeta(ctx, w, cfg, dp, vals["n_saved"], gen)
	if meta == nil {
		l.kvRemoveBestEffort(dp, "autosave verification failed")
		l.kvRemoveBestEffort(mp, "autosave verification failed")
		l.kvAutosaveDegraded(w, why, map[string]any{"generation": name})
		return false
	}
	meta.Origin = kvOriginAutosave
	b, _ := json.MarshalIndent(meta, "", "    ")
	err := os.WriteFile(mp, b, 0o644)
	if err == nil {
		if m, why := readKvSidecar(mp); m == nil {
			err = fmt.Errorf("read-back parse failed: %s", why)
		}
	}
	if err != nil {
		l.kvRemoveBestEffort(dp, "autosave sidecar write failed")
		l.kvRemoveBestEffort(mp, "autosave sidecar write failed")
		l.kvAutosaveDegraded(w, "writing the autosave sidecar failed: "+err.Error(), map[string]any{"generation": name})
		return false
	}
	w.mu.Lock()
	w.autoTokens, w.autoHasTokens = vals["n_saved"], true
	w.autoNext = kvOtherGeneration(name)
	w.autoCount++
	count := w.autoCount
	w.mu.Unlock()
	l.diag.write("WARMSTART_AUTOSAVE_OK", map[string]any{"generation": name, "generation_id": gen, "n_tokens": vals["n_saved"],
		"n_bytes": meta.NBytes, "kv_file_sha256": meta.KVFileSha256, "origin": kvOriginAutosave, "autosave_count": count,
		"started_at": t0.UTC().Format(time.RFC3339Nano), "elapsed_ms": saveMs, "total_ms": l.os.now().Sub(t0).Milliseconds()})
	return true
}

// autosaveTick is Invoke-AutosaveTick: a full period after ready, then once per period, only
// when the server is idle and the sequence changed since the last save or restore.
func (l *Launcher) autosaveTick(ctx context.Context, w *WarmStart, cfg *Config) {
	defer func() {
		if r := recover(); r != nil {
			w.autoStopped = true
			l.diag.write(kvDiagSaveFailed, map[string]any{"stage": "autosave", "reason": fmt.Sprintf("autosave tick fault: %v", r),
				"effect": "autosave is off for the rest of this run"})
		}
	}()
	if w == nil || !w.active() || !w.Autosave || w.autoStopped || cfg == nil {
		return
	}
	now := l.os.now()
	if w.autoClock.IsZero() {
		w.autoClock = now
		return
	}
	if now.Sub(w.autoClock) < kvAutosavePeriod {
		return
	}
	w.autoClock = now
	snap := kvSlotsSnapshot(ctx, cfg)
	switch {
	case !snap.ok:
		l.kvAutosaveDegraded(w, "the idle probe failed: "+snap.reason, nil)
		return
	case !snap.idle:
		l.diag.write("WARMSTART_AUTOSAVE_SKIP", map[string]any{"reason": "a request is in flight"})
		return
	case !snap.hasTokens:
		l.diag.write("WARMSTART_AUTOSAVE_SKIP", map[string]any{"reason": "the slot holds no sequence yet"})
		return
	}
	w.mu.Lock()
	same := w.autoHasTokens && snap.tokens == w.autoTokens
	w.seenTokens = snap.tokens
	w.mu.Unlock()
	if same {
		l.diag.write("WARMSTART_AUTOSAVE_SKIP", map[string]any{"reason": "unchanged since the last save", "n_tokens": snap.tokens})
		return
	}
	l.autosaveWrite(ctx, w, cfg, snap.tokens)
}

// --- the start and stop sites (GL-15) ---

// noSaveClose is the close-now path's reason (StopNoSave).
const noSaveClose = "closing now: no cache save"

// warmNoSave says why a requested stop stores nothing (Test-WarmstartSaveAllowed false).
func warmNoSave(w *WarmStart, alive bool) string {
	switch {
	case w == nil || w.State() == WarmHardOff:
		return "warm start is off for this start (a check run or a template profile)"
	case w.State() == WarmSoftOff:
		return "warm start is switched off"
	case w.State() == WarmLatchedOff:
		return "the kv directory could not be created"
	case !alive:
		return "the server had already exited"
	}
	return ""
}

// warmStopSave is the save of a requested stop, immediately before the stop signal (PS:14795-14815),
// under the cache-save bar (GL-07 finding 1): TimedCacheSave with this save's tokens as last seen
// (no request is added to PowerShell's), every track on a StopEvent. A failure is degraded.
func (l *Launcher) warmStopSave(s *session, send func(StopEvent)) {
	w := s.pre.Warm
	_, exited := s.child.Exited()
	s.mu.Lock()
	ready := s.ready
	s.mu.Unlock()
	if !warmSaveAllowed(w, ready, StatusOK, !exited) {
		why := warmNoSave(w, !exited)
		if why == "" {
			why = "the server is not ready"
		}
		l.diag.write("WARMSTART_SAVE_SKIP", map[string]any{"reason": why})
		send(StopEvent{NoSave: why})
		return
	}
	w.mu.Lock()
	tokens := int(w.seenTokens)
	w.mu.Unlock()
	cfg := s.pre.Config
	tr, err := l.TimedCacheSave(context.Background(), s.pre.SettingsKey, tokens, l.os.now(),
		func(ctx context.Context) (int, error) { return l.warmSave(ctx, w, cfg) },
		func(t CacheSaveTrack) { send(StopEvent{Save: &t}) })
	if err != nil {
		send(StopEvent{Save: &tr, SaveErr: err.Error()})
	}
}

// warmRecoveryRestart is Invoke-WarmstartRecoveryRestart (PS:10555): the old child stopped without
// a save and without releasing a lock (Stop-OwnedChildForRecovery), a new child on fresh out, err
// and metrics paths (New-KvRecoveryPath), ready again, the verdict recovery_cold. Only its own
// stop failure (fail_teardown) and the new child's spawn or health failure (fail_server_start)
// end the start; the returned teardown is the child that was stopped last.
func (l *Launcher) warmRecoveryRestart(ctx context.Context, s *session, w *WarmStart, exe, outLog, errLog string,
	tick func(time.Duration)) (*winChild, *Config, string, *TeardownResult, error) {
	s.mu.Lock()
	old, cfg := s.child, s.pre.Config
	s.ready = false // a stop now is a cancel of the start (the replacement is not ready yet)
	s.mu.Unlock()
	w.recoveries++
	n := w.recoveries
	l.diag.write("WARMSTART_RECOVERY", map[string]any{"stage": "begin", "incarnation": n,
		"line": "the slot erase failed as well; restarting the server so the slot is known clean."})
	td := stopOwnedChildGraceful(old, cfg.Port, l.stopEnv(nil))
	old.Close()
	_, why := teardownStatus(StatusFailServerStart, td, true)
	l.diag.write("WARMSTART_RECOVERY", map[string]any{"stage": "stop_old_child", "failed": why})
	if why != "" {
		return nil, nil, errLog, &td, stop(StatusFailTeardown, "warmstart recovery could not stop the previous server: %s", why)
	}
	newOut, err := kvRecoveryPath(outLog, n)
	if err != nil {
		return nil, nil, errLog, &td, stop(StatusFailServerStart, "warmstart recovery server start failed: %v", err)
	}
	newErr, err := kvRecoveryPath(errLog, n)
	if err != nil {
		return nil, nil, errLog, &td, stop(StatusFailServerStart, "warmstart recovery server start failed: %v", err)
	}
	c2 := *cfg
	c2.Env = make(map[string]string, len(cfg.Env))
	for k, v := range cfg.Env {
		c2.Env[k] = v
	}
	if m, ok := c2.Env[EnvMetrics]; ok && m != "" { // the engine opens it CREATE_NEW
		p, err := kvRecoveryPath(m, n)
		if err != nil {
			return nil, nil, errLog, &td, stop(StatusFailServerStart, "warmstart recovery server start failed: %v", err)
		}
		c2.Env[EnvMetrics] = p
		l.diag.write("METRICS_ENV", map[string]any{"injected": true, "path": p, "recovery_incarnation": n})
	}
	child, err := startOwnedChild(exe, c2.Argv, explicitEnv(c2.Env, os.Getenv), l.opt.BundleRoot, newOut, newErr)
	if err != nil {
		return nil, nil, newErr, &td, stop(StatusFailServerStart, "warmstart recovery server start failed: %v", err)
	}
	spawned := l.os.now()
	s.mu.Lock()
	s.child, s.log = child, newServerLog(newErr, spawned)
	s.mu.Unlock()
	l.diag.write("WARMSTART_RECOVERY", map[string]any{"stage": "respawn", "incarnation": n, "pid": child.PID(), "out_log": newOut, "err_log": newErr})
	if werr := waitForServerReady(ctx, child, c2.Host, c2.Port, newErr, l.os.portOwners, psReadyTimings, l.os.now, time.Sleep, tick); werr != nil {
		td2 := stopOwnedChildGraceful(child, c2.Port, l.stopEnv(nil))
		child.Close()
		return nil, nil, newErr, &td2, stop(StatusFailServerStart, "warmstart recovery server start failed: %v", werr)
	}
	s.mu.Lock()
	s.pre.Config, s.ready = &c2, true
	s.mu.Unlock()
	l.warmRecovered(w, n)
	return child, &c2, newErr, nil, nil
}
