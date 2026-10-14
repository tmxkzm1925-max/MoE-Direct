package core

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"
)

var testStamp = time.Date(2026, 10, 10, 1, 2, 3, 0, time.Local)

func realProfile(t *testing.T, id string) (*Catalog, *Profile) {
	t.Helper()
	needStaging(t)
	cat, err := ReadCatalog(realCatalog, stagingRoot)
	if err != nil {
		t.Fatal(err)
	}
	p := cat.ProfileByID(id)
	if p == nil {
		t.Fatalf("no profile %s", id)
	}
	return cat, p
}

func build(t *testing.T, cat *Catalog, p *Profile, ov map[string]string, decision Prefetch, qd int) (*Config, error) {
	t.Helper()
	return buildEffectiveConfig(configInput{cat: cat, profile: p, root: filepath.Clean(stagingRoot),
		outputDir: `D:\moe-models\m\repack`, modelPath: `D:\moe-models\m\model-00001-of-00002.gguf`, ov: ov,
		decision: decision, qd: qd, installed: 131072, geo: SlotGeometry{NExpert: 256, SlotStrideMax: 5308416, Layers: 48},
		metricsDir: `C:\state\logs`, stamp: testStamp})
}

// The PowerShell rules for the argv (Build-EffectiveConfig PS:8526-8558): defaults kept in
// order, -m appended, --port kept in place, -np forced to 1, --host kept (loopback), no
// --slot-save-path from the catalog (GL-15: the start's warm-start step adds its own, warmConfigure).
func TestBuildConfigQwen122BPinned(t *testing.T) {
	cat, p := realProfile(t, "qwen35-122b-nonextn")
	d := resolvePrefetch(p, true, optInOf("catalog"), 8, SelectionPinned, "catalog")
	if !d.On || d.K != 8 || d.N != 4 || d.Provenance != "catalog-validated" {
		t.Fatalf("decision %+v", d)
	}
	cfg, err := build(t, cat, p, map[string]string{}, d, 8)
	if err != nil {
		t.Fatal(err)
	}
	wantArgv := []string{"-ngl", "99", "--n-cpu-moe", "49", "-c", "12288", "-t", "8", "-b", "2048", "-ub", "512",
		"-fa", "on", "-np", "1", "--host", "127.0.0.1", "--port", "8093", "--no-webui", "--no-warmup",
		"-m", `D:\moe-models\m\model-00001-of-00002.gguf`, "--cache-ram", "0", "--ctx-checkpoints", "4"} // GL-06: E.2's off form
	if !reflect.DeepEqual(cfg.Argv, wantArgv) {
		t.Fatalf("argv\n got %q\nwant %q", cfg.Argv, wantArgv)
	}
	// budget: installed 131072 - 20480 -> 110592 clamped to 12288; model_cap = ceil(256*48*5308416/MiB) = 62208;
	// structural ceil(256*5308416/MiB) = 1296 < min 8192 -> auto 12288 (machine_cap).
	if cfg.Budget.MB != 12288 || cfg.Budget.Source != "auto" || !cfg.Budget.Unmeasured {
		t.Errorf("budget %+v", cfg.Budget)
	}
	want := map[string]string{
		EnvDirect: "1", EnvDirectDir: `D:\moe-models\m\repack`, EnvExpectsDir: filepath.Join(filepath.Clean(stagingRoot), "expects"),
		EnvBudgetMB: "12288", EnvQD: "8", EnvPrefetchK: "8", EnvPrefetchN: "4",
		EnvMetrics: `C:\state\logs\metrics_20261010T010203_` + strconv.Itoa(os.Getpid()) + `.jsonl`,
	}
	if !reflect.DeepEqual(cfg.Env, want) {
		t.Fatalf("env\n got %v\nwant %v", cfg.Env, want)
	}
	if cfg.Port != 8093 || cfg.Host != "127.0.0.1" || cfg.Address() != "http://127.0.0.1:8093" {
		t.Errorf("port/host %d %s", cfg.Port, cfg.Host)
	}

	// Overrides: port, ctx, threads replace in place; QD 3 is below N+1 = 5, so the catalog
	// pair is demoted (Resolve-PrefetchForQd) and MOE_NO_PREFETCH=1 replaces K/N.
	cfg, err = build(t, cat, p, map[string]string{"port": "8100", "ctx": "32768", "threads": "16", "qd": "3", "budget_mb": "20000"}, d, 8)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := argvValue(cfg.Argv, "--port"); v != "8100" {
		t.Errorf("port %s", v)
	}
	if v, _ := argvValue(cfg.Argv, "-c"); v != "32768" {
		t.Errorf("ctx %s", v)
	}
	if v, _ := argvValue(cfg.Argv, "-t"); v != "16" {
		t.Errorf("threads %s", v)
	}
	if cfg.Env[EnvQD] != "3" || cfg.Env[EnvNoPrefetch] != "1" || cfg.Env[EnvPrefetchK] != "" || !cfg.Prefetch.QDDemoted {
		t.Errorf("qd demotion env %v", cfg.Env)
	}
	if cfg.Prefetch.Echo != "off(reason=qd_below_prefetch_depth)" {
		t.Errorf("echo %s", cfg.Prefetch.Echo)
	}
	if cfg.Env[EnvBudgetMB] != "20000" || cfg.Budget.Source != "explicit" {
		t.Errorf("explicit budget %+v", cfg.Budget)
	}
}

func TestBuildConfigGptOssUnpinned(t *testing.T) {
	cat, p := realProfile(t, "gpt-oss-120b")
	// observe/off profile, request catalog -> not opted in; the QD from a degraded sweep is 1.
	d := resolvePrefetch(p, false, optInOf("catalog"), 1, SelectionUnpinned, "catalog")
	if d.On || d.OffReason != "not_opted_in_evidence_observe" {
		t.Fatalf("decision %+v", d)
	}
	cfg, err := build(t, cat, p, nil, d, QDDegraded)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := argvValue(cfg.Argv, "--n-cpu-moe"); v != "36" {
		t.Errorf("n-cpu-moe %s", v)
	}
	if cfg.Env[EnvQD] != "1" || cfg.Env[EnvNoPrefetch] != "1" {
		t.Errorf("env %v", cfg.Env)
	}
	if _, ok := cfg.Env[EnvPrefetchK]; ok {
		t.Error("K must be absent when prefetch is off")
	}
	if contains(cfg.Argv, argSlotSave) {
		t.Error("--slot-save-path must not come from the catalog (the warm-start step sets it at a start)")
	}
	// init opt-in on an observe profile with hold=true -> phase4_hold_unresolved.
	if d := resolvePrefetch(p, true, armInit, 8, SelectionUnpinned, "init"); d.On || d.OffReason != "phase4_hold_unresolved" {
		t.Errorf("init with hold: %+v", d)
	}
}

func TestBuildConfigRefusesNonLoopback(t *testing.T) {
	cat, p := realProfile(t, "qwen35-35b")
	q := *p
	q.DefaultArgv = setArgvValue(append([]string(nil), p.DefaultArgv...), "--host", "0.0.0.0")
	_, err := build(t, cat, &q, nil, Prefetch{Echo: "off(reason=x)"}, 1)
	wantStatus(t, err, StatusFailGateCatalog, "non-loopback host (0.0.0.0)")
	q.DefaultArgv = setArgvValue(append([]string(nil), p.DefaultArgv...), "--host", "::1")
	if _, err := build(t, cat, &q, nil, Prefetch{}, 1); err != nil {
		t.Errorf("::1 is loopback: %v", err)
	}
	q.DefaultArgv = removeArgvPair(append([]string(nil), p.DefaultArgv...), "--host")
	cfg, err := build(t, cat, &q, nil, Prefetch{}, 1)
	if err != nil || cfg.Host != "127.0.0.1" {
		t.Errorf("absent --host defaults to 127.0.0.1: %v %v", err, cfg)
	}
	q.DefaultArgv = removeArgvPair(append([]string(nil), p.DefaultArgv...), "--port")
	_, err = build(t, cat, &q, nil, Prefetch{}, 1)
	wantStatus(t, err, StatusFailGateCatalog, "no --port")
}

func TestResolveOverrides(t *testing.T) {
	b := map[string]Bound{"port": {0, 70000}, "ctx": {2048, 131072}, "threads": {1, 64}, "budget_mb": {8192, 65536}, "qd": {0, 100}}
	ov, notes, err := resolveOverrides(map[string]string{"port": " 8100 ", "prefetch": "INIT", "qd": "8"}, b, false, PathArguments)
	if err != nil || ov["port"] != "8100" || ov["prefetch"] != "init" || len(notes) != 0 {
		t.Fatalf("%v %v %v", ov, notes, err)
	}
	_, _, err = resolveOverrides(map[string]string{"n_gpu_layers": "10"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "command-line argument failed its type or range check: -n-gpu-layers = 10, allowed one of the settings")
	_, _, err = resolveOverrides(map[string]string{"port": "80"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-port = 80, allowed 1024..65535")
	_, _, err = resolveOverrides(map[string]string{"qd": "64"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-qd = 64, allowed 1..63")
	_, _, err = resolveOverrides(map[string]string{"ctx": "12k"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-ctx = 12k, allowed 2048..131072")
	_, _, err = resolveOverrides(map[string]string{"prefetch": "auto"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-prefetch = auto, allowed catalog, init") // GL-19: no adapt (F-PF §2.2)
	// GL-16: the warm-start switches left the model settings (OQ-6 (a)). GL-19: K alone is half
	// a pair (F-PF §9 C3).
	_, _, err = resolveOverrides(map[string]string{"warmstart": "on"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-warmstart = on, allowed one of the settings")
	_, _, err = resolveOverrides(map[string]string{"prefetch_k": "4"}, b, false, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-prefetch-k = 4, allowed K and N together, or neither · K and N are set together or not at all")
	// The user exception: on a virtual repack QD and prefetch outside the pair are accepted; since
	// GL-20 the notice is decided on the run's outcome after the config (F-PF §7.2; TestS8PF08).
	ov, notes, err = resolveOverrides(map[string]string{"qd": "4", "prefetch": "init"}, b, true, PathArguments)
	if err != nil || ov["qd"] != "4" || len(notes) != 0 {
		t.Errorf("virtual exception: %v %v %v", ov, notes, err)
	}
	_, notes, _ = resolveOverrides(map[string]string{"qd": "8", "prefetch": "catalog"}, b, true, PathArguments)
	if len(notes) != 0 {
		t.Errorf("validated pair noticed: %v", notes)
	}
	_, _, err = resolveOverrides(map[string]string{"qd": "0"}, b, true, PathArguments)
	wantStatus(t, err, StatusFailCustomArgs, "-qd = 0, allowed 1..63")
}

func TestResolveBudget(t *testing.T) {
	p := &Profile{MinBudgetMB: 8192}
	geo := SlotGeometry{NExpert: 256, SlotStrideMax: 5308416, Layers: 48}
	b, le := resolveBudget(p, nil, 32768, nil, geo, nil)
	if le != nil || b.MB != 12288 || b.MachineCapMB != 12288 {
		t.Errorf("32 GiB: machine_cap = max(4096, 32768-20480) = 12288: %+v %v", b, le)
	}
	b, le = resolveBudget(p, nil, 16384, nil, geo, nil)
	// machine_cap = clamp(-4096) = 4096 < supported 8192 -> fail_resource
	wantStatus(t, le, StatusFailResource, "below the supported minimum 8192")
	b, le = resolveBudget(p, map[string]string{"budget_mb": "9000"}, 16384, nil, geo, nil)
	if le != nil || b.MB != 9000 || b.Source != "explicit" {
		t.Errorf("explicit outranks: %+v %v", b, le)
	}
	b, _ = resolveBudget(p, nil, 0, errors.New("x"), geo, nil)
	if b.MB != 8192 || b.Source != "probe_failed_fallback" {
		t.Errorf("probe failed: %+v", b)
	}
	b, _ = resolveBudget(p, nil, 65536, nil, SlotGeometry{}, errors.New("manifest.json not present"))
	if b.MB != 8192 || b.Source != "geometry_unavailable_fallback" {
		t.Errorf("geometry: %+v", b)
	}
}

func TestSelectSweepQD(t *testing.T) {
	pts := []sweepPoint{{1, "ok", 100, 10}, {2, "ok", 170, 10}, {4, "ok", 195, 10}, {8, "ok", 200, 10}}
	n := int64(4)
	if q, r, ok := selectSweepQD(pts, armNone, nil); !ok || q != 4 || r != "io-knee" {
		t.Errorf("knee: %d %s", q, r)
	}
	if q, r, _ := selectSweepQD(pts, armCatalog, &n); q != 8 || r != "prefetch-preferred" {
		t.Errorf("catalog arm N=4 needs QD>=5: %d %s", q, r)
	}
	if q, r, _ := selectSweepQD([]sweepPoint{{1, "ok", 100, 10}, {2, "ok", 100, 10}, {4, "failed", 0, 0}, {8, "failed", 0, 0}}, armInit, nil); q != 2 || r != "init-prefetch-preferred" {
		t.Errorf("init arm: %d %s", q, r)
	}
	if q, _, ok := selectSweepQD([]sweepPoint{{1, "failed", 0, 0}}, armNone, nil); ok || q != QDDegraded {
		t.Errorf("no valid point -> degraded")
	}
}

func TestCommandLineAndEnvBlock(t *testing.T) {
	cases := map[string]string{"": `""`, "plain": "plain", "a b": `"a b"`, `x"y`: `"x\"y"`,
		`C:\dir name\`: `"C:\dir name\\"`, `C:\no\space`: `C:\no\space`, `q\"`: `"q\\\""`}
	for in, want := range cases {
		if got := crtArg(in); got != want {
			t.Errorf("crtArg(%q) = %s, want %s", in, got, want)
		}
	}
	if got := commandLine(`D:\t\llama-server.exe`, []string{"-m", `D:\m x\a.gguf`}); got != `D:\t\llama-server.exe -m "D:\m x\a.gguf"` {
		t.Errorf("command line %s", got)
	}
	env := explicitEnv(map[string]string{"path": "cfg", "MOE_DIRECT": "1"}, func(k string) string {
		return map[string]string{"PATH": "os", "SystemRoot": `C:\Windows`, "TEMP": `C:\t`}[k]
	})
	want := []string{"MOE_DIRECT=1", "path=cfg", `SystemRoot=C:\Windows`, `TEMP=C:\t`}
	if !reflect.DeepEqual(env, want) {
		t.Errorf("env block %q", env)
	}
}
