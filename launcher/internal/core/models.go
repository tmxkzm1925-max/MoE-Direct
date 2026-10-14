package core

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// The model list (the menu sources of UI-1 3 and UX 1-3, PS:10987-11288).

const (
	recentModelsShow = 8  // RECENT_MODELS_SHOW (PS:10989)
	scanModelsShow   = 12 // SCAN_MODELS_SHOW (PS:10990)
	scanMaxDepth     = 3  // SCAN_MAX_DEPTH (PS:10991)

	LabelCatalog         = "[catalog]"                                                     // LABEL_CATALOG (PS:855)
	LabelTemplatePrefix  = "[template: "                                                   // LABEL_TEMPLATE_PREFIX (PS:856)
	LabelUnsupported     = "[unsupported]"                                                 // LABEL_UNSUPPORTED (PS:857)
	LabelIdentifyPending = "[identify pending]"                                            // LABEL_IDENTIFY_PENDING (PS:858)
	LabelProvisionalNote = "labels are provisional; final identification happens at start" // PS:859
)

// Candidate is one offered model file (Build-ModelCandidates PS:11256-11288).
type Candidate struct {
	Path     string
	Name     string
	Bytes    int64 // summed over the existing siblings; -1 when unavailable
	Shards   int
	Declared int  // the split count in the name (1 when not split)
	Repacked bool // the three artifacts in <model dir>\repack (existence only)
	Label    string
	Source   string // recent | scan
	MTime    time.Time
	Arch     string
	Quant    string
	// ProfileIDs are the catalog profiles whose identify block, shard count and per-shard
	// sizes match (a list-time association; the pin is decided at start).
	ProfileIDs []string
}

// CatalogModel is one catalog profile with the file found for it on this PC.
type CatalogModel struct {
	ProfileID, DisplayName, HFRepo string
	Arch                           string
	MinBudgetMB                    int64
	FormatValidated                bool
	PerformanceValidated           bool
	Measurements                   []Measurement
	Present                        bool
	Path                           string // the file the start would be given (shard 1 of a split set)
	Folder                         string
	SizeBytes                      int64
	Shards                         int
	Quant                          string
	Sha256                         []string // from the shard digest caches only; "" for a shard not in a cache
	// Identity is the pin verdict as far as the caches tell: "pinned", "mismatch",
	// "unpinned" (the profile has no pin) or "not hashed" (a shard is in no cache).
	Identity        string
	Repacked        bool // packed artifacts present
	RepackedVirtual bool // a virtual plan present
	LastRun         *RunRecord
}

// ModelList is the answer of ListModels.
type ModelList struct {
	Models      []CatalogModel
	Candidates  []Candidate
	Truncated   bool
	Roots       []string
	RecentPath  string
	FreeRAMMiB  int64
	TotalRAMMiB int64
	MemErr      string
	Note        string
	// Files is every GGUF in the model folders (review items 11, 12): the list the screen shows.
	Files []ModelFile
}

// readRecentModels is Read-RecentModels (PS:11019-11038): read only, anything wrong is empty.
func readRecentModels(path string) []string {
	v, err := readJSONFileStrict(path)
	if err != nil {
		return nil
	}
	o, _ := asObj(v)
	arr, ok := jArr(o["paths"])
	if !ok {
		return nil
	}
	var out []string
	for _, p := range arr {
		if s, ok := jNonEmptyStr(p); ok {
			out = append(out, s)
		}
	}
	return out
}

// normalizeScanRoot is Get-NormalizedScanRoot (PS:11081-11089).
func normalizeScanRoot(r string) string {
	if r == "" {
		return r
	}
	t := strings.TrimRight(r, `\`)
	if t == "" {
		return r
	}
	if regexp.MustCompile(`^[A-Za-z]:$`).MatchString(t) {
		return t + `\`
	}
	return t
}

// isShardRepresentative is Test-ShardRepresentative (PS:11065-11070).
func isShardRepresentative(name string) bool {
	m := splitRegex.FindStringSubmatch(name)
	if m == nil {
		return true
	}
	i, _ := strconv.Atoi(m[2])
	return i == 1
}

type scanHit struct {
	path, name string
	bytes      int64
	mtime      time.Time
}

// scanCandidates is Get-GgufScanCandidates (PS:11091-11110).
func scanCandidates(roots []string) []scanHit {
	var out []scanHit
	for _, root := range roots {
		r := normalizeScanRoot(root)
		if st, err := os.Stat(r); err != nil || !st.IsDir() {
			continue
		}
		_ = filepath.WalkDir(r, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			rel := strings.TrimLeft(strings.TrimPrefix(p, r), `\`)
			depth := len(strings.Split(rel, `\`))
			if d.IsDir() {
				if p != r && (depth >= scanMaxDepth || isHidden(p)) {
					return fs.SkipDir
				}
				return nil
			}
			if depth > scanMaxDepth || !strings.EqualFold(filepath.Ext(p), ".gguf") || isHidden(p) {
				return nil
			}
			if !isShardRepresentative(d.Name()) {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return nil
			}
			out = append(out, scanHit{path: p, name: d.Name(), bytes: info.Size(), mtime: info.ModTime().UTC()})
			return nil
		})
	}
	return out
}

// shardDisplayAggregate is Get-ShardDisplayAggregate (PS:11118-11146).
func shardDisplayAggregate(path string) (bytes int64, shards, declared int) {
	bytes, shards, declared = -1, 1, 1
	name := filepath.Base(path)
	m := splitRegex.FindStringSubmatch(name)
	if m == nil {
		if st, err := os.Stat(path); err == nil {
			bytes = st.Size()
		}
		return
	}
	cnt, _ := strconv.Atoi(m[3])
	declared = cnt
	var sum int64
	found := 0
	for i := 1; i <= cnt; i++ {
		sib := filepath.Join(filepath.Dir(path), m[1]+"-"+pad5(i)+"-of-"+pad5(cnt)+".gguf")
		if st, err := os.Stat(sib); err == nil && !st.IsDir() {
			sum += st.Size()
			found++
		}
	}
	if found == 0 {
		return
	}
	return sum, found, declared
}

func pad5(i int) string { s := strconv.Itoa(i); return strings.Repeat("0", max(0, 5-len(s))) + s }

// siblingPaths lists the existing siblings in order (display use; the start uses shardPaths).
func siblingPaths(path string) []string {
	m := splitRegex.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return []string{path}
	}
	cnt, _ := strconv.Atoi(m[3])
	var out []string
	for i := 1; i <= cnt; i++ {
		sib := filepath.Join(filepath.Dir(path), m[1]+"-"+pad5(i)+"-of-"+pad5(cnt)+".gguf")
		if fileExists(sib) {
			out = append(out, sib)
		}
	}
	return out
}

// representativePath is Get-ShardRepresentativePath (PS:11193-11202).
func representativePath(path string) string {
	m := splitRegex.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return path
	}
	cnt, _ := strconv.Atoi(m[3])
	first := filepath.Join(filepath.Dir(path), m[1]+"-00001-of-"+pad5(cnt)+".gguf")
	if fileExists(first) {
		return first
	}
	return path
}

func repackArtifactsPresent(modelPath, leaf string, names ...string) bool {
	dir := filepath.Join(filepath.Dir(modelPath), leaf)
	for _, n := range names {
		if !fileExists(filepath.Join(dir, n)) {
			return false
		}
	}
	return true
}

// candidateLabel is Get-ModelCandidateLabel (PS:11230-11248), returning the header read too.
func candidateLabel(path string, cat *Catalog) (string, *GGUFHeader) {
	h, err := readGGUFHeader(representativePath(path), wantLabel())
	if err != nil {
		return LabelIdentifyPending, nil
	}
	archV, ok := h.Meta["general.architecture"]
	if !ok {
		return LabelIdentifyPending, h
	}
	arch, _ := archV.(string)
	var vals [3]int64
	for i, s := range ggufSuffixes {
		v, ok := h.Meta[arch+s]
		if !ok {
			return LabelIdentifyPending, h
		}
		vals[i], _ = metaLong(v)
	}
	if cat != nil {
		for _, p := range cat.Profiles {
			if p.Identify.Arch == arch && p.Identify.NLayer == vals[0] && p.Identify.NExpert == vals[1] && p.Identify.NExpertUsed == vals[2] {
				return LabelCatalog, h
			}
		}
	}
	if contains(archTemplateFamilies, arch) {
		return LabelTemplatePrefix + arch + "]", h
	}
	return LabelUnsupported, h
}

// profileFileMatch is the list-time association: identify block, shard count and per-shard
// sizes against expect.sources[] (the structural half of Get-StructuralProfileCandidates
// without reading every shard header).
func profileFileMatch(p *Profile, h *GGUFHeader, shardFiles []string) bool {
	if h == nil {
		return false
	}
	arch, _ := h.Meta["general.architecture"].(string)
	if arch != p.Identify.Arch {
		return false
	}
	var vals [3]int64
	for i, s := range ggufSuffixes {
		vals[i], _ = metaLong(h.Meta[arch+s])
	}
	if p.Identify.NLayer != vals[0] || p.Identify.NExpert != vals[1] || p.Identify.NExpertUsed != vals[2] {
		return false
	}
	srcs, err := expectSources(p.ExpectPath)
	if err != nil || len(srcs) != len(shardFiles) {
		return false
	}
	for i, s := range srcs {
		so, _ := asObj(s)
		fb, ok := jNonNegInt(so["file_bytes"])
		st, err := os.Stat(shardFiles[i])
		if !ok || err != nil || st.Size() != fb {
			return false
		}
	}
	return true
}

// ListModels lists the catalog models and the offered files: recent first (existing only,
// newest first, at most 8), then the scan of -ModelsRoot and <X>:\moe-models (newest first, at
// most 12), each with size, shard count, repack presence, the provisional label and the quant;
// for each catalog profile the first matching file, its digests from the caches (nothing is
// hashed here) and the free RAM now (decision ⑧).
func (l *Launcher) ListModels(ctx context.Context) (*ModelList, error) {
	cat, err := ReadCatalog(l.opt.CatalogPath, l.opt.BundleRoot)
	if err != nil {
		return nil, err
	}
	ml := &ModelList{Note: LabelProvisionalNote, RecentPath: filepath.Join(l.opt.StateDir, "recent_models.json")}
	if l.opt.ModelsRoot != "" {
		r := strings.Trim(strings.TrimSpace(l.opt.ModelsRoot), `"`)
		if st, err := os.Stat(r); err == nil && st.IsDir() {
			if abs, err := filepath.Abs(r); err == nil {
				ml.Roots = append(ml.Roots, normalizeScanRoot(abs))
			}
		}
	}
	ml.Roots = append(ml.Roots, l.os.modelRoots()...)

	seen := map[string]bool{}
	add := func(path, name, source string, mtime time.Time) {
		c := Candidate{Path: path, Name: name, Source: source, MTime: mtime}
		c.Bytes, c.Shards, c.Declared = shardDisplayAggregate(path)
		c.Repacked = repackArtifactsPresent(path, "repack", "experts.bin", "manifest.json", "verify_report.json")
		var h *GGUFHeader
		c.Label, h = candidateLabel(path, cat)
		if h != nil {
			c.Arch, _ = h.Meta["general.architecture"].(string)
			c.Quant = QuantName(h.Meta)
			sibs := siblingPaths(path)
			for _, p := range cat.Profiles {
				if profileFileMatch(p, h, sibs) {
					c.ProfileIDs = append(c.ProfileIDs, p.ID)
				}
			}
		}
		ml.Candidates = append(ml.Candidates, c)
	}
	recentAdded := 0
	for _, p := range readRecentModels(ml.RecentPath) {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if recentAdded >= recentModelsShow {
			break
		}
		if !fileExists(p) || seen[strings.ToLower(p)] {
			continue
		}
		seen[strings.ToLower(p)] = true
		var mt time.Time
		if st, err := os.Stat(p); err == nil {
			mt = st.ModTime().UTC()
		}
		add(p, filepath.Base(p), "recent", mt)
		recentAdded++
	}
	hits := scanCandidates(ml.Roots)
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].mtime.After(hits[j].mtime) })
	scanAdded := 0
	for _, h := range hits {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if seen[strings.ToLower(h.path)] {
			continue
		}
		seen[strings.ToLower(h.path)] = true
		if !l.settledSet(h.path) { // GL-14: a file still being written is not offered yet (watch.go)
			continue
		}
		if scanAdded >= scanModelsShow {
			ml.Truncated = true
			break
		}
		add(h.path, h.name, "scan", h.mtime)
		scanAdded++
	}

	for _, p := range cat.Profiles {
		m := CatalogModel{ProfileID: p.ID, DisplayName: p.DisplayName, HFRepo: p.HFRepo, Arch: p.Identify.Arch,
			MinBudgetMB: p.MinBudgetMB, FormatValidated: p.FormatValidated, PerformanceValidated: p.PerformanceValidated,
			Measurements: p.Measurements}
		for _, c := range ml.Candidates {
			if !contains(c.ProfileIDs, p.ID) {
				continue
			}
			m.Present, m.Path, m.Folder = true, representativePath(c.Path), filepath.Dir(c.Path)
			m.SizeBytes, m.Shards, m.Quant = c.Bytes, c.Shards, c.Quant
			m.Repacked = c.Repacked
			m.RepackedVirtual = repackArtifactsPresent(c.Path, "repack-virtual", "manifest.json")
			d, _ := l.shardDigests(ctx, siblingPaths(c.Path), false)
			m.Sha256 = d.Sha256
			switch {
			case len(p.SourceShardsSha256) == 0:
				m.Identity = "unpinned"
			case !d.Complete:
				m.Identity = "not hashed"
			case pinMatch(p.SourceShardsSha256, d.Sha256):
				m.Identity = SelectionPinned
			default:
				m.Identity = SelectionMismatch
			}
			if rec, ok := l.LatestForModel(m.Path); ok {
				m.LastRun = &rec
			}
			break
		}
		ml.Models = append(ml.Models, m)
	}
	ml.Files = l.modelFiles(ctx, cat, ml.Roots)
	if mem, err := l.os.mem(); err == nil {
		ml.FreeRAMMiB, ml.TotalRAMMiB = mem.AvailPhysMiB, mem.TotalPhysMiB
	} else {
		ml.MemErr = err.Error()
	}
	return ml, nil
}

// SettingRow is one changeable setting of a model.
type SettingRow struct {
	Key      string
	Default  string // the catalog default ("auto" where the launcher decides at start)
	Editable bool
	Options  []string // for enumerated values
	Min, Max int64    // for numbers (after the port and QD clamps of Test-OverrideValue)
	Bounded  bool
	Locked   string // why it cannot be changed here
	// OnRange is a row whose upper bound is an activation rule, not a refusal (F-PF §6: N is
	// on in 1..QD-1 and accepted above it, off with its reason): Min is checked, the screen
	// shows Min..QD-1 with the QD the run uses.
	OnRange bool
}

// Settings lists the changeable settings of a catalog profile: the allowlist (conflict 2),
// the bounds of allowlist_bounds with the port and QD clamps, the three keys whose families
// are not ported locked, and the repack mode (a launch option, not an allowlist key).
func (l *Launcher) Settings(profileID string) ([]SettingRow, error) {
	cat, err := ReadCatalog(l.opt.CatalogPath, l.opt.BundleRoot)
	if err != nil {
		return nil, err
	}
	p := cat.ProfileByID(profileID)
	if p == nil {
		return nil, stop(StatusFailGateCatalog, "no catalog profile %s", profileID)
	}
	return settingRows(p.DefaultArgv, p.Bounds, p.Identify.NExpert), nil
}

// StructureSettings are the settings of a model outside the catalog whose structure has a
// template (GL-14, conflict 14): the structure's defaults (the derived skeleton, PS:924-935);
// the budget's lower bound is the plan's structural minimum, known once the start's derive
// plan ran, so the budget row is unbounded here. ok is false for another structure.
func StructureSettings(arch string) ([]SettingRow, bool) {
	d, ok := TemplateDefaults(arch)
	if !ok {
		return nil, false
	}
	b := map[string]Bound{}
	for k, v := range d.Bounds {
		if k != "budget_mb" {
			b[k] = v
		}
	}
	rows := settingRows(d.Argv, b, 0)
	for i := range rows {
		if rows[i].Key == "repack_mode" { // PS:13923: a derived profile takes no virtual repack
			rows[i].Options, rows[i].Editable = []string{RepackPacked}, false
			rows[i].Locked = "locked: a model outside the catalog is set up packed only"
		}
	}
	return rows, true
}

// settingRows are the rows of Settings from a profile's default argv and bounds; nExpert is
// the profile's n_expert (0 = not known before the plan: K's bound stays 16, and the start
// checks the GGUF's value, F-PF §9 C6).
func settingRows(argv []string, bounds map[string]Bound, nExpert int64) []SettingRow {
	var rows []SettingRow
	for _, k := range SettingKeys {
		r := SettingRow{Key: k, Editable: true}
		switch k {
		case "port":
			r.Default, _ = argvValue(argv, "--port")
		case "ctx":
			r.Default, _ = argvValue(argv, "-c")
		case "threads":
			r.Default, _ = argvValue(argv, "-t")
		case "budget_mb":
			r.Default = "auto"
		case "qd":
			r.Default = "auto"
		case "prefetch":
			r.Default, r.Options = prefetchRequestDflt, PrefetchRequests
		case "prefetch_k": // F-PF §2.2, S8-PF-16: the on range 1..min(16, n_expert); auto = no pair
			r.Default, r.Min, r.Max, r.Bounded = "auto", PrefetchKMin, PrefetchKMax, true
			if nExpert > 0 {
				r.Max = min(r.Max, nExpert)
			}
		case "prefetch_n": // N 1..QD-1 is the on range, not a refusal (§6)
			r.Default, r.Min, r.OnRange = "auto", PrefetchNMin, true
		case "warmup":
			// conflict 2: open; "file:<path>" is typed (ParseTyped)
			r.Default, r.Options = WarmupDefault, []string{"on", "off"}
		}
		if b, ok := bounds[k]; ok {
			r.Min, r.Max, r.Bounded = b.Min, b.Max, true
			if k == "port" {
				r.Min, r.Max = max(r.Min, portMin), min(r.Max, portMax)
			}
			if k == "qd" {
				r.Min, r.Max = max(r.Min, EngineQDMin), min(r.Max, EngineQDMax)
			}
		}
		if fam, np := settingsNotPorted[k]; np {
			r.Editable, r.Locked = false, "locked: not in the open set of conflict 2 (not ported: "+fam+")"
		}
		rows = append(rows, r)
	}
	return append(rows, SettingRow{Key: "repack_mode", Default: RepackPacked, Editable: true,
		Options: []string{RepackPacked, RepackVirtual}})
}

// LockedSettings are the values conflict 2 keeps locked, with their value and why: the KV
// cache type (deferred by the user, 04:49 KST), the verify gates (LAUNCHER_SPEC.md:208-226),
// one parallel slot (-np 1, RELEASE_SPEC.md 7-3) and this PC only (loopback; conflict 17: there
// is no network option at all, so a screen need not show this row).
func (l *Launcher) LockedSettings(profileID string) ([]SettingRow, error) {
	cat, err := ReadCatalog(l.opt.CatalogPath, l.opt.BundleRoot)
	if err != nil {
		return nil, err
	}
	p := cat.ProfileByID(profileID)
	if p == nil {
		return nil, stop(StatusFailGateCatalog, "no catalog profile %s", profileID)
	}
	kv := "f16"
	if v, ok := argvValue(p.DefaultArgv, "-ctk"); ok {
		kv = v
	} else if v, ok := argvValue(p.DefaultArgv, "--cache-type-k"); ok {
		kv = v
	}
	return []SettingRow{
		{Key: "kv_type", Default: kv, Locked: "locked: the KV cache type is deferred (conflict 2)"},
		{Key: "verify_gates", Default: "on", Locked: "locked: the format gates run at every start (conflict 2, 16)"},
		{Key: "parallel", Default: "1", Locked: "locked: one request at a time (-np 1, conflict 2)"},
		{Key: "host", Default: BindHost, Locked: "locked: this PC only (loopback, conflict 17)"},
	}, nil
}
