package core

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// S8-LA-31 (GL-14): the catalog read, the GGUF scan, the shard sets, the model identity with its
// digest cache, and the derived start settings of a model outside the catalog equal the
// PowerShell launcher's outputs recorded over the same fixtures (testdata/la31: gen_fixtures.py
// made the files, committed as gguf/*.gguf.fixture; record_ps.ps1 recorded ps_recorded.json and
// ps_shard_cache.json); a missing model folder is polled and a .gguf still growing appears only
// after its size stops changing.

const (
	la31Dir     = "testdata/la31"
	la31Expects = "../../../../repack" // bench/repack: its expects/ equal the catalog's expect digests
	la31PlanOut = `C:\la31\repack`     // the output folder record_ps.ps1 gave New-DerivedProfile
)

type la31Bound struct{ Min, Max int64 }

type la31Record struct {
	Catalog struct {
		SourceTag string `json:"source_tag"`
		Runtime   struct {
			ServerExe    string   `json:"server_exe"`
			RepackerExe  string   `json:"repacker_exe"`
			RepackerArgv []string `json:"repacker_argv"`
			ExpectsDir   string   `json:"expects_dir"`
			WebUI        bool     `json:"webui"`
		} `json:"runtime"`
		Profiles []struct {
			ID           string `json:"profile_id"`
			DisplayName  string `json:"display_name"`
			HFRepo       string `json:"hf_repo"`
			HFRevision   string `json:"hf_revision"`
			RoutedScope  string `json:"routed_scope"`
			ExpectFile   string `json:"expect_file"`
			ExpectSha256 string `json:"expect_sha256"`
			Identify     struct {
				Arch        string `json:"arch"`
				NLayer      int64  `json:"n_layer"`
				NExpert     int64  `json:"n_expert"`
				NExpertUsed int64  `json:"n_expert_used"`
			} `json:"identify"`
			MinBudgetMB  int64                `json:"min_budget_mb"`
			Pins         []string             `json:"source_shards_sha256"`
			Format       bool                 `json:"format_validated"`
			Performance  bool                 `json:"performance_validated"`
			Measurements int                  `json:"reference_measurements"`
			Bounds       map[string]la31Bound `json:"allowlist_bounds"`
			Argv         []string             `json:"argv"`
			Env          map[string]string    `json:"env"`
		} `json:"profiles"`
	} `json:"catalog"`
	Scan []struct {
		Rel   string `json:"rel"`
		Name  string `json:"name"`
		Bytes int64  `json:"bytes"`
	} `json:"scan"`
	Files []struct {
		Rel       string `json:"rel"`
		Label     string `json:"label"`
		Aggregate struct {
			Bytes    int64 `json:"bytes"`
			Shards   int   `json:"shards"`
			Split    bool  `json:"split"`
			Declared int   `json:"declared"`
		} `json:"aggregate"`
		ShardSet struct {
			Arch         string `json:"arch"`
			TotalBytes   int64  `json:"total_bytes"`
			TotalTensors int64  `json:"total_tensors"`
			IsSplit      bool   `json:"is_split"`
			Shards       []struct {
				Rel       string `json:"rel"`
				FileBytes int64  `json:"file_bytes"`
				NTensors  int64  `json:"n_tensors"`
			} `json:"shards"`
			BlockCount      int64 `json:"block_count"`
			ExpertCount     int64 `json:"expert_count"`
			ExpertUsedCount int64 `json:"expert_used_count"`
		} `json:"shard_set"`
		Sha256     []string `json:"sha256"`
		Admissible bool     `json:"template_admissible"`
		Selection  *string  `json:"selection"`
	} `json:"files"`
	Identities []struct {
		Rel      string `json:"rel"`
		Identity string `json:"identity"`
		Size     int64  `json:"size"`
		MTime    int64  `json:"mtime"`
		Key      string `json:"key"`
	} `json:"identities"`
	Derived map[string]struct {
		Parsed struct {
			DerivedFrom      string `json:"derived_from"`
			TemplateID       string `json:"template_id"`
			TemplateVersion  string `json:"template_version"`
			RoutedScope      string `json:"routed_scope"`
			InventorySha256  string `json:"inventory_sha256"`
			Arch             string `json:"arch"`
			NLayer           int64  `json:"n_layer"`
			NExpert          int64  `json:"n_expert"`
			NExpertUsed      int64  `json:"n_expert_used"`
			MoELayers        int64  `json:"moe_layers"`
			TemplateLayers   int64  `json:"template_layers"`
			RoutedTensors    int64  `json:"routed_tensors"`
			SlotStrideMax    int64  `json:"slot_stride_max"`
			ExpertBytesTotal int64  `json:"expert_bytes_total"`
		} `json:"parsed"`
		ProfileJSON string `json:"profile_json"`
		Valid       bool   `json:"valid"`
		LockID      string `json:"lock_id"`
	} `json:"derived"`
	PlanRefusals map[string]struct {
		OK     bool   `json:"ok"`
		Reason string `json:"reason"`
	} `json:"plan_refusals"`
}

func la31Load(t *testing.T) (*la31Record, string) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(la31Dir, "ps_recorded.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r la31Record
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return &r, la31Models(t)
}

// la31Models copies the committed fixtures (testdata/la31/gguf/<path>.gguf.fixture: the
// repository ignores *.gguf and models/) to their .gguf names under a scratch models folder, as
// record_ps.ps1 does before it records, and dates them an hour back (finished files, so the
// folder watch lists them at once).
func la31Models(t *testing.T) string {
	t.Helper()
	src, err := filepath.Abs(filepath.Join(la31Dir, "gguf"))
	if err != nil {
		t.Fatal(err)
	}
	models := filepath.Join(t.TempDir(), "models")
	old := time.Now().Add(-time.Hour)
	n := 0
	err = filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".gguf.fixture") {
			return err
		}
		rel, err := filepath.Rel(src, strings.TrimSuffix(p, ".fixture"))
		if err != nil {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		dst := filepath.Join(models, rel)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			return err
		}
		n++
		return os.Chtimes(dst, old, old)
	})
	if err != nil || n != 6 {
		t.Fatalf("fixtures: %d copied (%v)", n, err)
	}
	return models
}

func la31Plan(t *testing.T, arch string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(la31Dir, "plans", arch+".plan.txt"))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func la31Rel(models, p string) string {
	return strings.ReplaceAll(strings.TrimLeft(strings.TrimPrefix(p, models), `\`), `\`, "/")
}

// la31PlanExec is the repacker of the derive plan: it writes the recorded plan text of the
// model's structure and counts its calls.
type la31PlanExec struct {
	text  map[string]string // model file name -> plan text
	calls int
	argv  []string
}

func (e *la31PlanExec) Run(_ context.Context, _ string, argv []string, _ string, _ []string, outLog, _ string, _ func(string)) (int, error) {
	e.calls++
	e.argv = argv
	model, _ := argvValue(argv, "--model")
	text, ok := e.text[filepath.Base(model)]
	if !ok {
		return 2, nil
	}
	return 0, os.WriteFile(outLog, []byte(text), 0o644)
}

func TestS8LA31(t *testing.T) {
	rec, models := la31Load(t)
	ctx := context.Background()

	t.Run("catalog", func(t *testing.T) {
		cat, err := ReadCatalog(realCatalog, la31Expects)
		if err != nil {
			t.Fatal(err)
		}
		rc := rec.Catalog
		if cat.SourceTag != rc.SourceTag || len(cat.Profiles) != 6 || len(rc.Profiles) != 6 {
			t.Fatalf("source_tag %q/%q, profiles %d/%d", cat.SourceTag, rc.SourceTag, len(cat.Profiles), len(rc.Profiles))
		}
		rt := cat.Runtime
		if rt.ServerExe != rc.Runtime.ServerExe || rt.RepackerExe != rc.Runtime.RepackerExe || !reflect.DeepEqual(rt.RepackerArgv, rc.Runtime.RepackerArgv) ||
			rt.ExpectsDir != rc.Runtime.ExpectsDir || rt.WebUI != rc.Runtime.WebUI {
			t.Errorf("runtime %+v, recorded %+v", rt, rc.Runtime)
		}
		for i, w := range rc.Profiles {
			p := cat.Profiles[i]
			pins := p.SourceShardsSha256
			if pins == nil {
				pins = []string{}
			}
			bounds := map[string]la31Bound{}
			for k, b := range p.Bounds {
				bounds[k] = la31Bound{b.Min, b.Max}
			}
			got := []any{p.ID, p.DisplayName, p.HFRepo, p.HFRevision, p.RoutedScope, p.ExpectFile, p.ExpectSha256,
				p.Identify.Arch, p.Identify.NLayer, p.Identify.NExpert, p.Identify.NExpertUsed, p.MinBudgetMB, pins,
				p.FormatValidated, p.PerformanceValidated, len(p.Measurements), bounds, p.DefaultArgv, p.DefaultEnv}
			want := []any{w.ID, w.DisplayName, w.HFRepo, w.HFRevision, w.RoutedScope, w.ExpectFile, w.ExpectSha256,
				w.Identify.Arch, w.Identify.NLayer, w.Identify.NExpert, w.Identify.NExpertUsed, w.MinBudgetMB, w.Pins,
				w.Format, w.Performance, w.Measurements, w.Bounds, w.Argv, w.Env}
			for j := range want {
				if !reflect.DeepEqual(got[j], want[j]) {
					t.Errorf("profile %s field %d: %v, recorded %v", w.ID, j, got[j], want[j])
				}
			}
		}
	})

	t.Run("scan", func(t *testing.T) {
		type row struct {
			rel, name string
			bytes     int64
		}
		var got, want []row
		for _, h := range scanCandidates([]string{models}) {
			got = append(got, row{la31Rel(models, h.path), h.name, h.bytes})
		}
		for _, s := range rec.Scan {
			want = append(want, row{s.Rel, s.Name, s.Bytes})
		}
		sort.Slice(got, func(i, j int) bool { return got[i].rel < got[j].rel })
		if !reflect.DeepEqual(got, want) || len(want) != 5 {
			t.Errorf("scan %v, recorded %v", got, want)
		}
	})

	t.Run("files", func(t *testing.T) {
		f := newFixture(t, false)
		cat, err := ReadCatalog(realCatalog, la31Expects)
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range rec.Files {
			path := filepath.Join(models, filepath.FromSlash(w.Rel))
			if label, _ := candidateLabel(path, cat); label != w.Label {
				t.Errorf("%s: label %q, recorded %q", w.Rel, label, w.Label)
			}
			b, s, d := shardDisplayAggregate(path)
			if b != w.Aggregate.Bytes || s != w.Aggregate.Shards || d != w.Aggregate.Declared || (d > 1) != w.Aggregate.Split {
				t.Errorf("%s: aggregate %d %d %d, recorded %+v", w.Rel, b, s, d, w.Aggregate)
			}
			set, le := readModelShardSet(path)
			if le != nil {
				t.Fatalf("%s: %v", w.Rel, le)
			}
			ws := w.ShardSet
			if set.Arch != ws.Arch || set.TotalBytes != ws.TotalBytes || set.TotalTensors != ws.TotalTensors || set.IsSplit != ws.IsSplit ||
				len(set.Shards) != len(ws.Shards) {
				t.Errorf("%s: shard set %s %d %d %v %d, recorded %+v", w.Rel, set.Arch, set.TotalBytes, set.TotalTensors, set.IsSplit, len(set.Shards), ws)
				continue
			}
			for i, sh := range set.Shards {
				if la31Rel(models, sh.Path) != ws.Shards[i].Rel || sh.FileBytes != ws.Shards[i].FileBytes || sh.NTensors != ws.Shards[i].NTensors {
					t.Errorf("%s: shard %d %s %d %d, recorded %+v", w.Rel, i, sh.Path, sh.FileBytes, sh.NTensors, ws.Shards[i])
				}
			}
			for _, c := range []struct {
				suffix string
				want   int64
			}{{".block_count", ws.BlockCount}, {".expert_count", ws.ExpertCount}, {".expert_used_count", ws.ExpertUsedCount}} {
				if v, _ := set.ArchLong(c.suffix); v != c.want {
					t.Errorf("%s: %s %d, recorded %d", w.Rel, c.suffix, v, c.want)
				}
			}
			paths := make([]string, len(set.Shards))
			for i, sh := range set.Shards {
				paths[i] = sh.Path
			}
			dg, err := f.l.shardDigests(ctx, paths, true)
			if err != nil || !reflect.DeepEqual(dg.Sha256, w.Sha256) {
				t.Errorf("%s: digests %v (%v), recorded %v", w.Rel, dg.Sha256, err, w.Sha256)
			}
			adm := templateAdmissible(true, set.Arch)
			if adm != w.Admissible {
				t.Errorf("%s: template admissible %v, recorded %v", w.Rel, adm, w.Admissible)
			}
			if w.Selection == nil {
				if _, err := f.l.resolveProfileSelection(ctx, cat, set, adm, ""); err == nil {
					t.Errorf("%s: a structure without a template must not be identified", w.Rel)
				}
				continue
			}
			sel, err := f.l.resolveProfileSelection(ctx, cat, set, adm, "")
			if err != nil || sel.Kind != *w.Selection {
				t.Errorf("%s: selection %v (%v), recorded %s", w.Rel, sel, err, *w.Selection)
			}
		}
	})

	t.Run("identity-cache", func(t *testing.T) {
		// The key of the cache is PowerShell's (volume serial and 128-bit file id, size, mtime).
		ids := map[string]FileIdentity{}
		for _, w := range rec.Identities {
			id := FileIdentity{ID: w.Identity, Size: w.Size, MTime: w.MTime}
			if k := shardCacheKey(id); k != w.Key {
				t.Errorf("%s: key %q, recorded %q", w.Rel, k, w.Key)
			}
			ids[w.Rel] = id
		}
		want := readShardCache(filepath.Join(la31Dir, "ps_shard_cache.json"))
		if len(want) != 6 {
			t.Fatalf("recorded cache has %d entries", len(want))
		}
		identity := func(p string) (FileIdentity, error) {
			if id, ok := ids[la31Rel(models, p)]; ok {
				return id, nil
			}
			return FileIdentity{}, errors.New("not a fixture")
		}
		// (1) the PowerShell cache file is read: every digest is a hit, nothing is hashed.
		f := newFixture(t, false)
		f.l.os.identity = identity
		b, _ := os.ReadFile(filepath.Join(la31Dir, "ps_shard_cache.json"))
		if err := os.MkdirAll(f.stateDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(f.stateDir, shardCacheFile), b, 0o644); err != nil {
			t.Fatal(err)
		}
		for _, w := range rec.Files {
			var paths []string
			for _, sh := range w.ShardSet.Shards {
				paths = append(paths, filepath.Join(models, filepath.FromSlash(sh.Rel)))
			}
			d, err := f.l.shardDigests(ctx, paths, false)
			if err != nil || !d.Complete || d.Hashed != 0 || !reflect.DeepEqual(d.Sha256, w.Sha256) {
				t.Errorf("%s: cached digests %+v (%v), recorded %v", w.Rel, d, err, w.Sha256)
			}
		}
		// (2) without a cache the digests are computed and written in PowerShell's form: the
		// launcher's own cache then holds the entries PowerShell's holds.
		g := newFixture(t, false)
		g.l.os.identity = identity
		hashed := 0
		for _, w := range rec.Files {
			var paths []string
			for _, sh := range w.ShardSet.Shards {
				paths = append(paths, filepath.Join(models, filepath.FromSlash(sh.Rel)))
			}
			d, err := g.l.shardDigests(ctx, paths, true)
			if err != nil || !reflect.DeepEqual(d.Sha256, w.Sha256) {
				t.Errorf("%s: digests %v (%v)", w.Rel, d.Sha256, err)
			}
			hashed += d.Hashed
		}
		if got := readShardCache(g.l.ownShardCachePath()); hashed != 6 || !reflect.DeepEqual(got, want) {
			t.Errorf("written cache %v (hashed %d), recorded %v", got, hashed, want)
		}
	})

	t.Run("derived", func(t *testing.T) {
		for _, arch := range []string{"gpt-oss", "qwen35moe", "deepseek2"} {
			w, ok := rec.Derived[arch]
			if !ok {
				t.Fatalf("%s: no record", arch)
			}
			tp, why := parseTemplatePlan(la31Plan(t, arch))
			if tp == nil {
				t.Fatalf("%s: %s", arch, why)
			}
			wp := w.Parsed
			got := []any{tp.DerivedFrom, tp.TemplateID, tp.TemplateVersion, tp.RoutedScope, tp.InventorySha256, tp.Arch, tp.NLayer, tp.NExpert,
				tp.NExpertUsed, tp.MoELayers, tp.TemplateLayers, tp.RoutedTensors, tp.SlotStrideMax, tp.ExpertBytesTotal}
			want := []any{wp.DerivedFrom, wp.TemplateID, wp.TemplateVersion, wp.RoutedScope, wp.InventorySha256, wp.Arch, wp.NLayer, wp.NExpert,
				wp.NExpertUsed, wp.MoELayers, wp.TemplateLayers, wp.RoutedTensors, wp.SlotStrideMax, wp.ExpertBytesTotal}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s: parsed %v, recorded %v", arch, got, want)
			}
			var shas []string
			for _, f := range rec.Files {
				if f.Rel == arch+"-tpl.gguf" {
					shas = f.Sha256
				}
			}
			o, err := newDerivedProfile(tp, la31PlanOut, shas)
			if err != nil {
				t.Fatal(err)
			}
			psv, err := parseJSONStrict([]byte(w.ProfileJSON))
			if err != nil {
				t.Fatal(err)
			}
			// The one key the Go writer adds to PowerShell's profile: the hold key of F-PF §10.1
			// (K188; the PowerShell part of KNOWN row 188 is dropped with the retiring launcher).
			// Everything else equals the recorded PowerShell output.
			if h, ok := o["prefetch_promotion_hold"].(bool); !ok || !h {
				t.Errorf("%s: the derived profile lacks prefetch_promotion_hold = true", arch)
			}
			ps := jobj{}
			for k, v := range o {
				if k != "prefetch_promotion_hold" {
					ps[k] = v
				}
			}
			if !reflect.DeepEqual(any(ps), psv) {
				gb, _ := json.Marshal(ps)
				t.Errorf("%s: derived profile\n go %s\n ps %s", arch, gb, w.ProfileJSON)
			}
			if v := validateDerivedProfile(o); (v == "") != w.Valid {
				t.Errorf("%s: validator %q, recorded valid=%v", arch, v, w.Valid)
			}
			if derivedLockID(tp.DerivedFrom) != w.LockID {
				t.Errorf("%s: lock id %s, recorded %s", arch, derivedLockID(tp.DerivedFrom), w.LockID)
			}
		}
	})

	t.Run("plan-refusals", func(t *testing.T) {
		text := la31Plan(t, "gpt-oss")
		p0, _ := parseTemplatePlan(text)
		filter := func(drop string) string {
			var keep []string
			for _, ln := range strings.Split(text, "\n") {
				if !strings.Contains(ln, drop) {
					keep = append(keep, ln)
				}
			}
			return strings.Join(keep, "\n")
		}
		mut := map[string]string{
			"no stride line":           filter("slot_stride_max="),
			"duplicated arch line":     text + "\narch=gpt-oss n_layer=3 n_expert=4 n_expert_used=2 schema=separate bias=True",
			"non-numeric stride":       strings.ReplaceAll(text, "slot_stride_max=4096", "slot_stride_max=lots"),
			"no template version":      strings.ReplaceAll(text, "derived_from=gpt-oss@1", "derived_from=gpt-oss"),
			"short inventory":          strings.ReplaceAll(text, "inventory_sha256="+p0.InventorySha256, "inventory_sha256="+p0.InventorySha256[:63]),
			"truncated":                filter("--plan done"),
			"expect disagrees":         strings.ReplaceAll(text, `"n_expert": 4`, `"n_expert": 5`),
			"unsafe arch":              strings.ReplaceAll(text, "arch=gpt-oss n_layer", `arch=..\evil n_layer`),
			"empty":                    "",
			"arch and template differ": strings.ReplaceAll(text, "arch=gpt-oss n_layer", "arch=qwen35moe n_layer"),
			"unknown scope":            strings.ReplaceAll(text, "routed_scope=all (", "routed_scope=some ("),
		}
		if len(rec.PlanRefusals) != len(mut) {
			t.Fatalf("recorded %d refusals, %d here", len(rec.PlanRefusals), len(mut))
		}
		for name, w := range rec.PlanRefusals {
			tp, why := parseTemplatePlan(mut[name])
			if (tp != nil) != w.OK || why != w.Reason {
				t.Errorf("%s: %q, recorded %q", name, why, w.Reason)
			}
		}
	})

	t.Run("accepted-and-derived-start-settings", func(t *testing.T) {
		f := newFixture(t, false)
		mdir := filepath.Join(f.dir, "models", "tpl")
		model := filepath.Join(mdir, "gpt-oss-tpl.gguf")
		b, err := os.ReadFile(filepath.Join(models, "gpt-oss-tpl.gguf"))
		if err != nil {
			t.Fatal(err)
		}
		writeModel(t, model, b)
		ex := &la31PlanExec{text: map[string]string{"gpt-oss-tpl.gguf": la31Plan(t, "gpt-oss")}}
		f.l.deriveExec = ex
		var w = rec.Derived["gpt-oss"]
		psv, _ := parseJSONStrict([]byte(w.ProfileJSON))
		psProf, _ := asObj(psv)
		psDef, _ := asObj(psProf["defaults"])
		psArgv := []string{}
		arr, _ := jArr(psDef["argv"])
		for _, a := range arr {
			psArgv = append(psArgv, psString(a))
		}

		// Virtual is refused before the plan runs (PS:13923).
		if _, err := f.l.Preflight(ctx, Request{ModelPath: model, RepackMode: RepackVirtual, NoPreset: true}); err == nil {
			t.Fatal("virtual must be refused for a model outside the catalog")
		} else {
			wantStatus(t, err, StatusFailCustomArgs, "do not support -RepackMode virtual")
		}
		if ex.calls != 0 {
			t.Fatalf("the plan ran for a refused virtual request")
		}

		// Accepted: the plan runs, the derived profile is the start's profile, and the start
		// needs its setup (the template repack) first.
		res, err := f.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true})
		var rn *RepackNeededError
		if !errors.As(err, &rn) || rn.Mode != RepackPacked {
			t.Fatalf("want the setup first, got %v", err)
		}
		if ex.calls != 1 || !contains(ex.argv, "--experimental-arch-template") || !contains(ex.argv, "--plan") || contains(ex.argv, "--profile") || contains(ex.argv, "--scope") {
			t.Fatalf("plan argv %v (calls %d)", ex.argv, ex.calls)
		}
		p := res.Profile
		if p == nil || p.Derived == nil || p.ID != psString(psProf["profile_id"]) || res.Selection != SelectionTemplate {
			t.Fatalf("derived profile %+v", p)
		}
		if !reflect.DeepEqual(p.DefaultArgv, psArgv) || p.MinBudgetMB != 1 || p.Bounds["budget_mb"] != (Bound{1, 65536}) ||
			p.Bounds["ctx"] != (Bound{2048, 131072}) || p.Derived.LockID != w.LockID {
			t.Errorf("derived start settings %v %d %v, recorded %v", p.DefaultArgv, p.MinBudgetMB, p.Bounds, psArgv)
		}

		// After the template setup (its derived expect and packed artifacts bound to the lock id)
		// the preflight passes with the derived settings; the plan is not run again.
		out := filepath.Join(mdir, "repack")
		writeJSON(t, filepath.Join(out, DerivedExpectFileName), map[string]any{"derived_from": "gpt-oss@1"})
		dsha, _ := fileSha256(filepath.Join(out, DerivedExpectFileName))
		os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, out, w.LockID, dsha)
		res, err = f.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true})
		if err != nil {
			t.Fatalf("preflight after the setup: %v", err)
		}
		if ex.calls != 1 {
			t.Errorf("the plan ran again (%d calls)", ex.calls)
		}
		st := res.Start
		if st.Ctx != 4096 || st.Threads != 8 || st.Port != 8093 || st.PrefetchOn {
			t.Errorf("start settings %+v", st)
		}
		if v, _ := argvValue(res.Config.Argv, "--n-cpu-moe"); v != "3" {
			t.Errorf("--n-cpu-moe %q in %v", v, res.Config.Argv)
		}
		if res.Config.Prefetch.OffReason != "not_opted_in_evidence_unverified" {
			t.Errorf("prefetch %+v", res.Config.Prefetch)
		}
		// A changed derived expect no longer passes the gate.
		writeJSON(t, filepath.Join(out, DerivedExpectFileName), map[string]any{"derived_from": "gpt-oss@2"})
		if _, err := f.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true}); err == nil {
			t.Errorf("a changed derived expect must fail the gate")
		}
	})

	t.Run("list-and-structure-defaults", func(t *testing.T) {
		f := newFixture(t, false)
		f.l.opt.ModelsRoot = models
		ml, err := f.l.ListModels(ctx)
		if err != nil {
			t.Fatal(err)
		}
		states := map[string]string{}
		for _, mf := range ml.Files {
			states[mf.Name] = mf.State.String()
			if mf.Template != (mf.Arch != "moetest") {
				t.Errorf("%s: template %v", mf.Name, mf.Template)
			}
		}
		want := map[string]string{"gpt-oss-tpl.gguf": "usable", "qwen35moe-tpl.gguf": "usable", "deepseek2-tpl.gguf": "usable",
			"gptoss-split-00001-of-00002.gguf": "usable", "moetest-unsupported.gguf": "unsupported"}
		for k, v := range want {
			if states[k] != v {
				t.Errorf("%s: %q, want %q (%v)", k, states[k], v, states)
			}
		}
		rows, ok := StructureSettings("gpt-oss")
		if !ok {
			t.Fatal("no structure settings for gpt-oss")
		}
		byKey := map[string]SettingRow{}
		for _, r := range rows {
			byKey[r.Key] = r
		}
		if byKey["port"].Default != "8093" || byKey["ctx"].Default != "4096" || byKey["threads"].Default != "8" ||
			byKey["ctx"].Min != 2048 || byKey["ctx"].Max != 131072 || byKey["threads"].Max != 64 || byKey["budget_mb"].Bounded ||
			byKey["repack_mode"].Editable || !reflect.DeepEqual(byKey["repack_mode"].Options, []string{RepackPacked}) {
			t.Errorf("structure settings %+v", rows)
		}
		if _, ok := StructureSettings("moetest"); ok {
			t.Error("a structure without a template has no structure settings")
		}
	})

	t.Run("watch", func(t *testing.T) {
		f := newFixture(t, false)
		root := filepath.Join(f.dir, "later")
		f.l.opt.ModelsRoot = root
		listed := func() bool {
			ml, err := f.l.ListModels(ctx)
			if err != nil {
				t.Fatal(err)
			}
			for _, mf := range ml.Files {
				if mf.Name == "grow.gguf" {
					return true
				}
			}
			return false
		}
		t0 := time.Now().Add(-time.Hour)
		clock := t0
		f.l.os.now = func() time.Time { return clock }
		s0 := f.l.ScanStamp()
		// The missing folder appears: the next look sees it.
		path := filepath.Join(root, "g", "grow.gguf")
		b, _ := os.ReadFile(filepath.Join(models, "gpt-oss-tpl.gguf"))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, b[:len(b)/2], 0o644); err != nil {
			t.Fatal(err)
		}
		os.Chtimes(path, t0, t0) // written «now»
		s1 := f.l.ScanStamp()
		if s1 == s0 {
			t.Error("a folder created later is not polled")
		}
		// Still being written: not listed.
		if listed() {
			t.Error("a file written just now is listed")
		}
		// Its size changes between two looks: not listed, even with an old write time.
		clock = t0.Add(GrowSettle)
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		old := t0.Add(-time.Hour)
		os.Chtimes(path, old, old)
		s2 := f.l.ScanStamp()
		if listed() || f.l.ScanStamp() != s2 {
			t.Error("a file whose size changed since the last look is listed")
		}
		// The size stayed the same for GrowSettle: the stamp changes and the file is listed.
		clock = clock.Add(GrowSettle)
		if s3 := f.l.ScanStamp(); s3 == s2 || !listed() {
			t.Errorf("a settled file is not listed (stamp changed %v)", s3 != s2)
		}
		// A fresh write of the same size holds it back again for GrowSettle.
		os.Chtimes(path, clock, clock)
		if listed() {
			t.Error("a file rewritten just now is listed")
		}
		clock = clock.Add(GrowSettle)
		if !listed() {
			t.Error("a file GrowSettle after its last write is not listed")
		}
	})
}
