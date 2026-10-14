// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// GL-06: the commit prediction (F-BU B-F, H; plan rows S8-BU-02, -08, -09, -10). S8-BU-01 is
// in bu_gl06_windows_test.go (the host's GetPerformanceInfo).

func closeTo(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

// diagData is the data of the records of one kind in l's diagnostic log, in order.
func diagData(t *testing.T, l *Launcher, kind string) []map[string]any {
	t.Helper()
	fh, err := os.Open(l.diag.Path())
	if err != nil {
		return nil
	}
	defer fh.Close()
	var out []map[string]any
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var rec struct {
			Kind string
			Data map[string]any
		}
		if json.Unmarshal(sc.Bytes(), &rec) == nil && rec.Kind == kind {
			out = append(out, rec.Data)
		}
	}
	return out
}

// TestS8BU02 (S8-BU-02; GL-06), on the host, GGUF headers only: T_pool at budget 12288 from the
// GGUF tensor table: Qwen 12,882,726,912 B (S 6,131,712 B; 2,101 slots), DSV4 12,873,129,984 B
// (S 13,381,632 B; 962 slots); integer equality.
func TestS8BU02(t *testing.T) {
	for _, c := range []struct {
		name, path       string
		stride, slots, b int64
	}{
		{"Qwen", `D:\moe-models\qwen35-122b\Qwen_Qwen3.5-122B-A10B-Q4_K_M\Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf`, 6131712, 2101, 12882726912},
		{"DSV4", `D:\moe-models\deepseek-v4-flash\DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf`, 13381632, 962, 12873129984},
	} {
		if _, err := os.Stat(c.path); err != nil {
			t.Skipf("host model file not present (%s): %v", c.name, err)
		}
		shape, err := readModelShape(c.path)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		terms, ok := TermsOf(shape, 12288, false)
		if !ok {
			t.Fatalf("%s: no terms from %s", c.name, shape)
		}
		if terms.StrideBytes != c.stride || terms.Slots != c.slots || terms.Slots*terms.StrideBytes != c.b {
			t.Errorf("%s: S %d, slots %d, T_pool %d B; want S %d, slots %d, T_pool %d B", c.name, terms.StrideBytes, terms.Slots,
				terms.Slots*terms.StrideBytes, c.stride, c.slots, c.b)
		}
		if !closeTo(terms.PoolMiB, float64(c.b)/bytesPerMiB) {
			t.Errorf("%s: PoolMiB %v, want %v", c.name, terms.PoolMiB, float64(c.b)/bytesPerMiB)
		}
		t.Logf("%s: S %d B · %d slots · T_pool %d B (%.4f MiB)", c.name, terms.StrideBytes, terms.Slots, terms.Slots*terms.StrideBytes, terms.PoolMiB)
	}
}

// bu08Projection is TestS8LA13's projection of a measured structure (X-FIT f01, residuals of
// one run) with the cache cap passed.
func bu08Projection(capMiB float64) (FitResult, NeedProjection) {
	shape := &ModelShape{Arch: "testmoe", NLayer: 2, NEmbd: 64, NExpert: 4, NVocab: 1000, Identity: "id1",
		Experts: map[int64][]ExpertPart{0: {{Type: 8, Dims: []int64{64, 64, 4}, AbsOffset: 100}}, 1: {{Type: 8, Dims: []int64{64, 64, 4}, AbsOffset: 200}}}}
	fit := RunFitParams(context.Background(), fakeFit(xfitStdout[0].out, 0, nil), "fit.exe", qwenServerArgv, "m.gguf", nil)
	key := ResidualKey{"id1", 12288, 64}
	store := memStore{arch: "testmoe", m: map[memKey]Residuals{{key, AfterRequest}: {CDevMiB: 341, UMiB: 1041.25, Run: "X-U s01"}}}
	return fit, ProjectNeed(fit, shape, nil, "id1", 12288, 64, true, capMiB, store)
}

// TestS8BU08 (S8-BU-08; GL-06), the core side: the check reads the commit beside free RAM,
// predicts the commit with the pool term apart, gives the room after the start, the safe line
// at 4 GiB and, near the limit, the warning line's numbers; it changes nothing and refuses
// nothing (U-13 [A-NEAR]). The wire and screen sides are internal/wire and internal/ui
// TestS8BU08.
func TestS8BU08(t *testing.T) {
	if SafeLineMiB != 4096 {
		t.Fatalf("safe line %v MiB, want 4 GiB", SafeLineMiB)
	}
	// (1) the preflight reads the commit through the reader (beside free RAM)
	f := newFixture(t, false)
	reading := CommitReading{LimitMiB: 39323.7, TotalMiB: 13877.0}
	f.l.os.perf = func() (CommitReading, error) { return reading, nil }
	res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.CommitOK || res.Commit.LimitMiB != reading.LimitMiB || res.Commit.TotalMiB != reading.TotalMiB || res.FreeMiB != f.mem.AvailPhysMiB {
		t.Fatalf("commit %+v (ok %v) beside free %d", res.Commit, res.CommitOK, res.FreeMiB)
	}
	if res.PredictionOK || res.NearOK || res.CapMaxOK { // the fixture's need is «unmeasured»: no prediction
		t.Fatalf("a prediction from an unmeasured need: %+v", res.Prediction)
	}

	// (2) the prediction from the projection: four terms, the pool apart, the cache term = cap
	fit, prj := bu08Projection(0)
	pred, ok := CommitFrom(fit, prj, 0)
	if !ok || pred.PoolMiB != prj.Terms.PoolMiB || pred.PromptCacheMiB != 0 {
		t.Fatalf("prediction %+v", pred)
	}
	if !closeTo(pred.MirrorMiB, float64(fit.Print.DevSelf())+341) || !closeTo(pred.HostRemainderMiB, prj.HostCMiB+prj.Terms.OutMiB+prj.Terms.SnapMiB+prj.Terms.TabMiB+CHostMiB+1041.25) {
		t.Fatalf("mirror %v, host remainder %v", pred.MirrorMiB, pred.HostRemainderMiB)
	}
	if !closeTo(pred.Total(), pred.PoolMiB+pred.MirrorMiB+pred.HostRemainderMiB) || !closeTo(pred.RAMNeed(), prj.MiB) {
		t.Fatalf("total %v, RAM need %v (projection %v)", pred.Total(), pred.RAMNeed(), prj.MiB)
	}
	if _, ok := CommitFrom(fit, NeedProjection{Label: NeedUnmeasured}, 0); ok {
		t.Fatal("a prediction from an unmeasured projection")
	}

	// (3) room after the start, the near line, no action
	for _, c := range []struct {
		name  string
		limit float64
		near  bool
	}{{"room", 13877.0 + pred.Total() + 6000, false}, {"near", 13877.0 + pred.Total() + 1000, true}} {
		l := newFixture(t, false).l
		cr := CommitReading{LimitMiB: c.limit, TotalMiB: 13877.0}
		l.os.perf = func() (CommitReading, error) { return cr, nil }
		cfg := &Config{Argv: append([]string(nil), qwenServerArgv...)}
		before := append([]string(nil), cfg.Argv...)
		r := &PreflightResult{Fit: fit, Projection: prj}
		l.commitPrediction(r, cfg)
		if !r.CommitOK || !r.PredictionOK || !closeTo(r.RoomAfterMiB, c.limit-13877.0-pred.Total()) {
			t.Fatalf("%s: %+v", c.name, r)
		}
		if r.NearOK != c.near {
			t.Fatalf("%s: near %v, want %v (room %v)", c.name, r.NearOK, c.near, r.RoomAfterMiB)
		}
		n := r.Near
		if !closeTo(n.Room, r.RoomAfterMiB) || n.SafeLine != 4096 || n.Limit != c.limit || n.Now != 13877.0 ||
			!closeTo(n.Predicted, pred.TotalNoCache()) || n.Pool != pred.PoolMiB {
			t.Fatalf("%s: near numbers %+v", c.name, n)
		}
		if !reflect.DeepEqual(cfg.Argv, before) || r.Refusal != nil || r.Config != nil {
			t.Fatalf("%s: the check changed or refused something: %v %v", c.name, cfg.Argv, r.Refusal)
		}
		recs := diagData(t, l, "COMMIT_PREDICTION")
		if len(recs) != 1 || recs[0]["near"] != c.near || recs[0]["pool_mib"] != pred.PoolMiB {
			t.Fatalf("%s: diagnostic record %v", c.name, recs)
		}
		t.Logf("%s: limit %.1f · now %.1f · predicted %.1f (pool %.1f) · room after start %.1f · near %v · cap max %d",
			c.name, c.limit, 13877.0, pred.Total(), pred.PoolMiB, r.RoomAfterMiB, r.NearOK, r.CapMaxMiB)
	}
	// (4) not validated: the reading alone
	off := false
	l := newFixture(t, false).l
	l.opt.ProjectionValidation = &off
	l.os.perf = func() (CommitReading, error) { return reading, nil }
	r := &PreflightResult{Fit: fit, Projection: prj}
	l.commitPrediction(r, &Config{Argv: qwenServerArgv})
	if !r.CommitOK || r.PredictionOK || r.NearOK {
		t.Fatalf("not validated: %+v", r)
	}
}

// TestS8BU09 (S8-BU-09; GL-06): each refusal of BUP section 2.7 gives F-BU H.2's text byte for
// byte: the functions from their inputs, and the launcher's own paths triggered by their input
// (the setting check's reasons, the autotune stop and its banner, the RAM stop and its ram line).
func TestS8BU09(t *testing.T) {
	// (1) H.2, every row, from its inputs
	for _, c := range []struct{ got, want string }{
		{ReasonNotInteger("ctx"), "ctx must be an integer"},
		{ReasonOutOfBounds("ctx", 2048, 262144), "ctx out of bounds [2048..262144]"},
		{StopInvalidCLI("ctx", "ctx must be an integer"), "invalid -ctx: ctx must be an integer"},
		{LineIgnoredCLI("ctx", "ctx must be an integer"), "[custom] ignoring invalid -ctx: ctx must be an integer"},
		{LineEditorRejected("ctx out of bounds [2048..262144]"), "    rejected: ctx out of bounds [2048..262144]"},
		{LinePresetDiscarded("qd", "qd out of bounds [1..63]"), "[preset] stored preset discarded (override qd rejected: qd out of bounds [1..63]); falling back to catalog defaults."},
		{ReasonAutotuneBelowMin(AutotuneBelowMin{Candidate: 8, SupportedMin: 8192, StructuralMin: 4, ProfileMin: 8192, MachineCap: 4096, InstalledMiB: 16384, ModelCap: 8}),
			"candidate 8 MB is below the supported minimum 8192 MB (structural_min 4 MB, profile min_budget_mb 8192 MB; machine_cap 4096 MB from installed 16384 MiB, model_cap 8 MB)"},
		{BannerAutotuneImpossible("r"), "[budget] autotune cannot serve this model on this machine - r"},
		{StopAutotune("r"), "budget autotune: r"},
		{LineRAMInsufficient(12288, 8000, 4096, "budget_only"), "  ram    : required 12288 MB > available 8000 MB (ctx 4096, basis budget_only)"},
		{ReasonBudgetOverRAM, "selected budget alone exceeds available RAM (the unmeasured fixed terms can only add to it)"},
		{StopRAMPreflight("r"), "RAM preflight hard stop: r"},
	} {
		if c.got != c.want {
			t.Errorf("text\n got %q\nwant %q", c.got, c.want)
		}
	}

	// (2) the setting check's reasons (PS :6088, :6101), triggered by their input
	bounds := map[string]Bound{"ctx": {2048, 262144}}
	if _, err := checkSetting("ctx", "12k", bounds); err == nil || err.Error() != "ctx must be an integer" {
		t.Errorf("not an integer: %v", err)
	}
	if _, err := checkSetting("ctx", "1024", bounds); err == nil || err.Error() != "ctx out of bounds [2048..262144]" {
		t.Errorf("out of bounds: %v", err)
	}

	// (3) the autotune stop (PS :3881-3884, :3968), directly and through a preflight with its
	// banner (PS :3958) in the diagnostic log
	_, le := resolveBudget(&Profile{MinBudgetMB: 8192}, map[string]string{}, 16384, nil, SlotGeometry{NExpert: 4, SlotStrideMax: 1 << 20, Layers: 2}, nil)
	if le == nil || le.Status != StatusFailResource || le.Reason != "budget autotune: candidate 8 MB is below the supported minimum 8192 MB (structural_min 4 MB, profile min_budget_mb 8192 MB; machine_cap 4096 MB from installed 16384 MiB, model_cap 8 MB)" {
		t.Errorf("autotune stop: %v", le)
	}
	f := newFixture(t, false)
	writeStrideArtifacts(t, f.outDir, "fixture-profile", f.expectSha, 4<<30) // structural_min 16384 MB > machine_cap 12288 MB
	_, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true})
	reason := "candidate 12288 MB is below the supported minimum 16384 MB (structural_min 16384 MB, profile min_budget_mb 4096 MB; machine_cap 12288 MB from installed 65536 MiB, model_cap 32768 MB)"
	if got, ok := AsLaunchError(err); !ok || got.Status != StatusFailResource || got.Reason != "budget autotune: "+reason {
		t.Fatalf("preflight autotune stop: %v", err)
	}
	if recs := diagData(t, f.l, "BUDGET_AUTOTUNE"); len(recs) != 1 || recs[0]["banner"] != "[budget] autotune cannot serve this model on this machine - "+reason {
		t.Errorf("banner records %v", recs)
	}

	// (4) the RAM stop (PS :3601, :3696) with its ram line (PS :3695); the Go stop adds
	// conflict 7's budget part after the PS text
	g := newFixture(t, false)
	g.mem.AvailPhysMiB = 8000
	res, err := g.l.Preflight(context.Background(), Request{ModelPath: g.modelPath, NoPreset: true})
	got, ok := AsLaunchError(err)
	head := "RAM preflight hard stop: selected budget alone exceeds available RAM (the unmeasured fixed terms can only add to it)"
	if !ok || got.Status != StatusFailResource || !strings.HasPrefix(got.Reason, head+"; ") {
		t.Fatalf("RAM stop: %v", err)
	}
	ctxArg, _ := argvValue(res.Config.Argv, "-c")
	line := "  ram    : required " + itoa(int(res.Config.Budget.MB)) + " MB > available 8000 MB (ctx " + ctxArg + ", basis budget_only)"
	if recs := diagData(t, g.l, "RAM_PREFLIGHT"); len(recs) != 1 || recs[0]["line"] != line {
		t.Errorf("ram line records %v, want %q", recs, line)
	}
	t.Logf("RAM stop: %s", got.Reason)
}

// writeStrideArtifacts is writePackedArtifacts with the slot stride given.
func writeStrideArtifacts(t *testing.T, outDir, profileID, expectSha string, stride int64) {
	t.Helper()
	lock := map[string]any{"profile_id": profileID, "expect_sha256": expectSha}
	writeJSON(t, filepath.Join(outDir, "manifest.json"), map[string]any{
		"schema_version": "2.0", "reference_lock": lock,
		"model":  map[string]any{"n_expert": 4},
		"layout": map[string]any{"slot_stride_max": stride, "layers": []any{map[string]any{}, map[string]any{}}}})
	ms, err := fileSha256(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	rep := map[string]any{"pass": true, "checked_at": "2026-10-07T01:51:39.264165+00:00", "pairs_pass": 8, "pairs_total": 8,
		"expected_pairs": 8, "problems": []any{}, "failures": []any{}, "padding_failures": []any{},
		"reference_lock": lock, "manifest_sha256": ms}
	b, _ := json.Marshal(rep)
	if err := os.WriteFile(filepath.Join(outDir, "verify_report.json"), append(b, '\n', '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	settleFile(t, filepath.Join(outDir, "verify_report.json"))
}

// TestS8BU10 (S8-BU-10; GL-06), with a fixture of the reference machine and a 16 GB fixture:
// the server argv holds --cache-ram with the cap F-BU fixes and --ctx-checkpoints 4; E.3's
// maximum fits the commit room left (4,096 on the reference machine, 0 on the 16 GB fixture);
// the default is the off form (E.2, L:4420; the edition over the cell's «defaults to X-CB's
// value»); a cap set on the settings page replaces it and 0 turns the cache off; the commit
// prediction's cache term equals the cap passed; the llama-fit-params argv holds neither option.
func TestS8BU10(t *testing.T) {
	// (1) E.3 on the fixtures
	for _, c := range []struct {
		name string
		h    float64
		want int
	}{
		{"reference machine, X-CB measured H", 5052.10, 4096},
		{"reference machine, live reading (F.4)", 39323.7 - 13877.0 - 20479.5, 4096},
		{"16 GB fixture (X-CB scaling)", -10897.75, 0},
		{"smallest step", 256 + 210.5, 256},
		{"below the smallest step", 256 + 210.4, 0},
		{"largest step", 1e6, 8192},
	} {
		if got := DefaultCap(c.h); got != c.want {
			t.Errorf("%s: H %.2f -> %d, want %d", c.name, c.h, got, c.want)
		}
	}
	// (2) F.4: the reference reading; the near room counts the prediction without the cache
	ref := CommitReading{LimitMiB: 39323.7, TotalMiB: 13877.0}
	p0 := CommitPrediction{PoolMiB: 12285.9, MirrorMiB: 4537.0, HostRemainderMiB: 3656.6}
	if h := RoomAfterStart(ref, p0); math.Abs(h-4967.2) > 1e-6 || DefaultCap(h) != 4096 {
		t.Errorf("reference room %v, max %d", h, DefaultCap(h))
	}
	p4 := p0
	p4.PromptCacheMiB = 4096
	if n, near := CheckNear(ref, p4); near || math.Abs(n.Room-4967.2) > 1e-6 || math.Abs(p4.Total()-24575.5) > 1e-6 ||
		math.Abs(RoomAfterStart(ref, p4)-871.2) > 1e-6 {
		t.Errorf("cap 4096: near %v room %v, total %v, room with the cache %v", near, n.Room, p4.Total(), RoomAfterStart(ref, p4))
	}
	// (3) ResolveCap: the default and 0 are the off form; a positive cap replaces it
	for _, c := range []struct {
		set  CapSetting
		want []string
	}{
		{CapSetting{}, []string{"--cache-ram", "0", "--ctx-checkpoints", "4"}},
		{CapSetting{Set: true, Value: 0}, []string{"--cache-ram", "0", "--ctx-checkpoints", "4"}},
		{CapSetting{Set: true, Value: 2048}, []string{"--cache-ram", "2048", "--ctx-checkpoints", "4"}},
	} {
		if got := ResolveCap(c.set).ServerArgv(); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%+v: %q, want %q", c.set, got, c.want)
		}
	}
	// (4) the server argv of a preflight: the options once each, the cap given; the fit argv
	// holds neither; the cache term of the projection and of the prediction is the cap
	f := newFixture(t, false)
	for _, c := range []struct {
		set CapSetting
		cap string
	}{{CapSetting{}, "0"}, {CapSetting{Set: true, Value: 0}, "0"}, {CapSetting{Set: true, Value: 2048}, "2048"}} {
		res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true, PromptCacheCap: c.set})
		if err != nil {
			t.Fatal(err)
		}
		argv := res.Config.Argv
		n := 0
		for _, a := range argv {
			if a == "--cache-ram" || a == "--ctx-checkpoints" || a == "-cram" || a == "--swa-checkpoints" {
				n++
			}
		}
		v, _ := argvValue(argv, "--cache-ram")
		k, _ := argvValue(argv, "--ctx-checkpoints")
		if n != 2 || v != c.cap || k != "4" || !reflect.DeepEqual(argv[len(argv)-4:], []string{"--cache-ram", c.cap, "--ctx-checkpoints", "4"}) {
			t.Errorf("%+v: server argv %q", c.set, argv)
		}
		if cacheCapOf(argv) != float64(ResolveCap(c.set).CacheRAMMiB) {
			t.Errorf("%+v: cache term %v", c.set, cacheCapOf(argv))
		}
		for _, a := range FitArgv(argv, f.modelPath) {
			if a == "--cache-ram" || a == "--ctx-checkpoints" || a == "-cram" || a == "--swa-checkpoints" {
				t.Errorf("%+v: the fit argv holds %s", c.set, a)
			}
		}
	}
	// a catalog argv's own cache options give way to the launcher's
	in := []string{"-c", "1", "--cache-ram", "8192", "-cram", "5", "--ctx-checkpoints", "32", "--swa-checkpoints", "3", "-m", "x"}
	if got := StripCacheArgs(in); !reflect.DeepEqual(got, []string{"-c", "1", "-m", "x"}) {
		t.Errorf("strip %q", got)
	}
	_, prj0 := bu08Projection(0)
	fit, prj := bu08Projection(2048)
	pred, ok := CommitFrom(fit, prj, 2048)
	if !ok || pred.PromptCacheMiB != 2048 || !closeTo(prj.MiB-prj0.MiB, 2048) || !closeTo(pred.RAMNeed(), prj.MiB) ||
		!closeTo(pred.Total()-pred.TotalNoCache(), 2048) {
		t.Errorf("cache term: prediction %+v, projection %v vs %v", pred, prj.MiB, prj0.MiB)
	}
	// (5) the check's maximum is E.3's rule on the room with the cache off, whatever the cap
	l := newFixture(t, false).l
	l.os.perf = func() (CommitReading, error) {
		return CommitReading{LimitMiB: 13877.0 + pred.TotalNoCache() + 5052.1, TotalMiB: 13877.0}, nil
	}
	r := &PreflightResult{Fit: fit, Projection: prj}
	l.commitPrediction(r, &Config{Argv: append(StripCacheArgs(qwenServerArgv), "--cache-ram", "2048", "--ctx-checkpoints", "4")})
	if !r.CapMaxOK || r.CapMaxMiB != 4096 || r.Prediction.PromptCacheMiB != 2048 {
		t.Errorf("cap max %d (ok %v), cache term %v", r.CapMaxMiB, r.CapMaxOK, r.Prediction.PromptCacheMiB)
	}
}
