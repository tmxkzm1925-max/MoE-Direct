package core

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// The model list of review items 11 and 12 (reviews/_gol_review_8_user_20261010.md): every
// GGUF found in the model folders (a catalog model absent from this PC is not in it), each
// with its catalog mark (★: a catalog profile matches the file) and one of three states:
// broken, unsupported, or usable (the screen then shows the memory verdict). Outside the
// catalog the architecture-template path (select.go archTemplateOn, templateAdmissible;
// ARCH_TEMPLATE_FAMILIES) decides "supported"; starting such a model keeps the PowerShell
// default of the template toggle (conflict 14 is not decided here).

// FileState is the state of one model file.
type FileState uint8

const (
	FileUsable      FileState = iota // the memory verdict decides what the screen shows
	FileBroken                       // the file differs from its pin, the split set is incomplete, a cut setup, an unreadable header
	FileUnsupported                  // no catalog profile and the architecture is not admissible for the template path
)

func (s FileState) String() string {
	return [...]string{"usable", "broken", "unsupported"}[s]
}

// ModelFile is one row of the model list.
type ModelFile struct {
	Path     string // the start path (shard 1 of a split set)
	Name     string
	Folder   string // the model folder (scan root) it was found under
	Bytes    int64
	Shards   int
	Declared int
	Arch     string
	Quant    string

	ProfileID   string // the matching catalog profile; "" outside the catalog
	DisplayName string // the catalog display name, or the file name without extension
	Catalog     bool   // ★
	Template    bool   // supported through the architecture template (outside the catalog)

	State  FileState
	Reason string // why broken or unsupported

	// Identity is the pin verdict for a catalog file: "pinned", "mismatch", "unpinned",
	// "not hashed"; "" outside the catalog.
	Identity        string
	Sha256          []string
	Repacked        bool
	RepackedVirtual bool
	// SetupCut is conflict 9: an interrupted setup (repack\experts.bin.partial) is no longer a
	// broken file - the next setup removes it and starts from the beginning.
	SetupCut bool
	// GL-10 (H4): what the cut-off setup left - the bytes of its half-made files, the time the
	// marker was last written, and the last full setup's duration of this file (0 = unknown).
	PartialBytes int64
	PartialAt    time.Time
	LastRepack   time.Duration
	Alias        string // the --alias a start passes (item 27)

	// Item 13: the need of the newest run of this file, and item 28: its last run's stats.
	LastRun   *RunRecord
	NeedMiB   int64
	NeedBasis NeedBasis
	Stats     *ModelStats
}

// modelFiles lists every GGUF under the roots (shard representatives, the scan depth of
// Get-GgufScanCandidates, no count cap).
func (l *Launcher) modelFiles(ctx context.Context, cat *Catalog, roots []string) []ModelFile {
	tplOn, _ := l.archTemplateOn()
	var out []ModelFile
	seen := map[string]bool{}
	for _, root := range roots {
		for _, h := range scanCandidates([]string{root}) {
			if ctx.Err() != nil {
				return out
			}
			key := strings.ToLower(h.path)
			if seen[key] {
				continue
			}
			seen[key] = true
			if !l.settledSet(h.path) { // GL-14: a file still being written is not listed yet (watch.go)
				continue
			}
			out = append(out, l.modelFile(ctx, cat, tplOn, h.path, root))
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Catalog != out[j].Catalog {
			return out[i].Catalog
		}
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out
}

func (l *Launcher) modelFile(ctx context.Context, cat *Catalog, tplOn bool, path, root string) ModelFile {
	rep := representativePath(path)
	f := ModelFile{Path: rep, Name: filepath.Base(rep), Folder: root}
	f.Bytes, f.Shards, f.Declared = shardDisplayAggregate(rep)
	f.DisplayName = strings.TrimSuffix(f.Name, filepath.Ext(f.Name))
	if m := splitRegex.FindStringSubmatch(f.Name); m != nil {
		f.DisplayName = m[1]
	}
	f.Repacked = repackArtifactsPresent(rep, "repack", "experts.bin", "manifest.json", "verify_report.json")
	f.RepackedVirtual = repackArtifactsPresent(rep, "repack-virtual", "manifest.json")
	label, h := candidateLabel(rep, cat)
	if h != nil {
		f.Arch, _ = h.Meta["general.architecture"].(string)
		f.Quant = QuantName(h.Meta)
	}
	sibs := siblingPaths(rep)
	if h != nil && cat != nil {
		for _, p := range cat.Profiles {
			if profileFileMatch(p, h, sibs) {
				f.ProfileID, f.DisplayName, f.Catalog = p.ID, p.DisplayName, true
				d, _ := l.shardDigests(ctx, sibs, false)
				f.Sha256 = d.Sha256
				switch {
				case len(p.SourceShardsSha256) == 0:
					f.Identity = "unpinned"
				case !d.Complete:
					f.Identity = "not hashed"
				case pinMatch(p.SourceShardsSha256, d.Sha256):
					f.Identity = SelectionPinned
				default:
					f.Identity = SelectionMismatch
				}
				break
			}
		}
	}
	if !f.Catalog && label != LabelIdentifyPending && templateAdmissible(tplOn, f.Arch) {
		f.Template = true
	}
	f.Alias = ModelAlias(f.ProfileID, rep)
	f.SetupCut = fileExists(filepath.Join(filepath.Dir(rep), "repack", "experts.bin.partial"))
	if f.SetupCut {
		dir := filepath.Join(filepath.Dir(rep), "repack")
		f.PartialBytes, _ = filesBytes(dir, PackedArtifactSet)
		if st, err := os.Stat(filepath.Join(dir, "experts.bin.partial")); err == nil {
			f.PartialAt = st.ModTime()
		}
		f.LastRepack, _ = l.LastRepackDuration(rep, RepackPacked)
	}

	switch {
	case h == nil || label == LabelIdentifyPending:
		f.State, f.Reason = FileBroken, "the GGUF header cannot be read as a MoE model"
	case f.Declared > 1 && f.Shards < f.Declared:
		f.State, f.Reason = FileBroken, "split set incomplete: "+strconv.Itoa(f.Shards)+" of "+strconv.Itoa(f.Declared)+" files"
	case f.Identity == SelectionMismatch:
		f.State, f.Reason = FileBroken, "the file differs from the catalog's pinned bytes"
	case !f.Catalog && !f.Template:
		f.State = FileUnsupported
		if !contains(archTemplateFamilies, f.Arch) {
			f.Reason = "architecture " + f.Arch + " has no catalog profile and no template"
		} else {
			f.Reason = "the architecture template is off"
		}
	}

	if rec, ok := l.LatestForModel(rep); ok {
		r := rec
		f.LastRun = &r
		f.NeedMiB, f.NeedBasis = NeedOf(&r)
	} else {
		f.NeedBasis = NeedNotMeasured
	}
	if st, ok := l.ModelStatsFor(rep); ok {
		s := st
		f.Stats = &s
	}
	return f
}
