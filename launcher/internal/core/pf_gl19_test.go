package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// GL-19: the manual prefetch surface of the manual-prefetch edition (F-PF,
// reviews/_edition_8_F-PF_manual_prefetch.md), checked by the GT rows of its §12.

// pfFixture is the core fixture with a fresh sweep binding (QD 4), so the probe is ok and a
// manual pair can be an ON candidate (§3.2 3a). The fixture's n_expert is 4 (GGUF and catalog).
func pfFixture(t *testing.T) *fixture {
	t.Helper()
	f := newFixture(t, false)
	bindSweep(t, f)
	return f
}

// bindSweep writes the fixture's sweep binding, selected QD 4 (GL-21 uses it on the live rig).
func bindSweep(t *testing.T, f *fixture) {
	t.Helper()
	ms, _ := fileSha256(filepath.Join(f.outDir, "manifest.json"))
	key := sweepTargetKey("fixture-tag", "fixture-profile", f.expectSha, ms, `\\?\volume{fixture}\`)
	entry := map[string]any{"probe_algorithm": sweepAlgorithm, "measurement_method": sweepMethod,
		"source_tag": "fixture-tag", "profile_id": "fixture-profile", "expectation_digest": f.expectSha,
		"manifest_sha256": ms, "output_volume_guid": `\\?\volume{fixture}\`, "selected_qd": 4,
		"selected_reason": "io-knee", "block_bytes": 1 << 20, "window_duration_ms": 1250,
		"swept_at": "2026-10-07T02:00:00.0000000Z", "repack_checked_at": "2026-10-07T01:51:39.264165+00:00",
		"points": []any{
			map[string]any{"qd": 1, "status": "ok", "bytes": 100, "elapsed_ticks": 10},
			map[string]any{"qd": 2, "status": "ok", "bytes": 150, "elapsed_ticks": 10},
			map[string]any{"qd": 4, "status": "ok", "bytes": 199, "elapsed_ticks": 10},
			map[string]any{"qd": 8, "status": "ok", "bytes": 200, "elapsed_ticks": 10}}}
	writeJSON(t, filepath.Join(f.stateDir, "probe.state.json"), map[string]any{"state_version": 2, "entries": map[string]any{key: entry}})
}

// bindSweepQD4Only is bindSweep with QD 8 below 90 % of QD 4 (GL-10): the 90 % set is {4}, so
// no arm moves the sweep's QD off 4.
func bindSweepQD4Only(t *testing.T, f *fixture) {
	t.Helper()
	bindSweep(t, f)
	path := filepath.Join(f.stateDir, "probe.state.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b = []byte(strings.Replace(string(b), `"bytes": 200`, `"bytes": 150`, 1))
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// engineRangeOff restates the engine's range check (PB:1309-1316): K outside 1..16, N below
// 1, or N at least the QD it receives gives invalid_range.
func engineRangeOff(k, n, qd int64) bool { return k < 1 || k > 16 || n < 1 || n >= qd }

// statusField reads one «name=value» field back from the status lines (S8-PF-04).
func statusField(lines []string, name string) (string, bool) {
	for _, l := range lines {
		for _, f := range strings.Fields(l) {
			if strings.HasPrefix(f, name+"=") {
				return strings.TrimPrefix(f, name+"="), true
			}
		}
	}
	return "", false
}

// S8-PF-01: each listed value is refused before start with a message naming the bound (K 1..16,
// N at least 1, both or neither); no accepted value is turned off by the engine for a range
// reason at QD at least N+1 (§9 C1-C3).
func TestS8PF01(t *testing.T) {
	cases := []struct {
		name string
		set  map[string]string
		want string
	}{
		{"K 0", map[string]string{"prefetch_k": "0", "prefetch_n": "2"}, "K must be a whole number 1..16 (got: 0)"},
		{"K 17", map[string]string{"prefetch_k": "17", "prefetch_n": "2"}, "K must be a whole number 1..16 (got: 17)"},
		{"K 8x", map[string]string{"prefetch_k": "8x", "prefetch_n": "2"}, "K must be a whole number 1..16 (got: 8x)"},
		{"K +8", map[string]string{"prefetch_k": "+8", "prefetch_n": "2"}, "K must be a whole number 1..16 (got: +8)"},
		{"K blank 8", map[string]string{"prefetch_k": " 8", "prefetch_n": "2"}, "K must be a whole number 1..16 (got:  8)"},
		{"K empty", map[string]string{"prefetch_k": "", "prefetch_n": "2"}, "K must be a whole number 1..16 (got: )"},
		{"N 0", map[string]string{"prefetch_k": "8", "prefetch_n": "0"}, "N must be a whole number of at least 1 (got: 0)"},
		{"N letter", map[string]string{"prefetch_k": "8", "prefetch_n": "a"}, "N must be a whole number of at least 1 (got: a)"},
		{"K alone", map[string]string{"prefetch_k": "8"}, "K and N are set together or not at all"},
		{"N alone", map[string]string{"prefetch_n": "2"}, "K and N are set together or not at all"},
	}
	t.Run("refused-before-start", func(t *testing.T) {
		for _, path := range CustomValuePaths[:2] { // the arguments and the screen's start
			for _, c := range cases {
				_, _, err := resolveOverrides(c.set, nil, false, path)
				if StatusOf(err) != StatusFailCustomArgs || !strings.Contains(err.Error(), c.want) {
					t.Errorf("%s %s: %v", path, c.name, err)
				}
				if path == PathArguments && ExitCodeOf(err) != 3 {
					t.Errorf("%s: exit %d", c.name, ExitCodeOf(err))
				}
			}
		}
		// Through a start: refused before any config or env exists.
		f := pfFixture(t)
		for _, c := range cases {
			res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true, FromArguments: true, Settings: c.set})
			if StatusOf(err) != StatusFailCustomArgs || !strings.Contains(err.Error(), c.want) || (res != nil && res.Config != nil) {
				t.Errorf("start %s: %v", c.name, err)
			}
		}
	})
	t.Run("screen-typed", func(t *testing.T) {
		// The screen checks the text as typed (ParseTyped otherwise trims; First's note).
		k := SettingRow{Key: "prefetch_k", Min: 1, Max: 16, Bounded: true}
		n := SettingRow{Key: "prefetch_n", Min: 1, OnRange: true}
		for _, v := range []string{"0", "17", "8x", "+8", " 8", ""} {
			if _, _, err := ParseTyped(k, v); err == nil {
				t.Errorf("typed K %q accepted", v)
			}
		}
		for _, v := range []string{"0", "a"} {
			if _, _, err := ParseTyped(n, v); err == nil {
				t.Errorf("typed N %q accepted", v)
			}
		}
		if v, _, err := ParseTyped(k, "8"); err != nil || v != "8" {
			t.Errorf("typed K 8: %q %v", v, err)
		}
		if v, _, err := ParseTyped(n, "auto"); err != nil || v != "auto" {
			t.Errorf("typed N auto: %q %v", v, err)
		}
	})
	t.Run("accepted-set-against-the-engine", func(t *testing.T) {
		// Every accepted pair at a QD of at least N+1 is an ON candidate and passes the engine's
		// range check (E1, E2: PB:1309-1316).
		_, p := realProfile(t, "qwen35-35b")
		for k := int64(1); k <= 16; k++ {
			for _, qd := range []int64{2, 5, 8, 63} {
				for n := int64(1); n < qd; n++ {
					ov, _, err := resolveOverrides(map[string]string{"prefetch_k": strconv.FormatInt(k, 10), "prefetch_n": strconv.FormatInt(n, 10)}, nil, false, PathArguments)
					if err != nil {
						t.Fatalf("K %d N %d refused: %v", k, n, err)
					}
					d := prefetchForQD(resolvePrefetchManual(p, true, armNone, int(qd), SelectionUnpinned, "", manualInput{pair: manualPairOf(ov)}), int(qd))
					if !d.On || engineRangeOff(d.K, d.N, qd) {
						t.Fatalf("K %d N %d QD %d: %+v", k, n, qd, d)
					}
				}
			}
		}
	})
}

// S8-PF-02: K 8, N 4 at QD 4 is off with qd_below_prefetch_depth shown, K and N absent from the
// child env and MOE_NO_PREFETCH=1 [A-QDN]; at QD 5 an ON candidate (§6).
func TestS8PF02(t *testing.T) {
	cat, p := realProfile(t, "qwen35-35b")
	pair := &ManualPair{K: 8, N: 4}
	for _, c := range []struct {
		qd  int
		on  bool
		env map[string]string
	}{
		{4, false, map[string]string{EnvNoPrefetch: "1"}},
		{5, true, map[string]string{EnvPrefetchK: "8", EnvPrefetchN: "4"}},
	} {
		d := resolvePrefetchManual(p, true, armNone, c.qd, SelectionUnpinned, "", manualInput{pair: pair, source: SourceSettingsScreen})
		cfg, err := build(t, cat, p, map[string]string{"qd": strconv.Itoa(c.qd), "prefetch_k": "8", "prefetch_n": "4"}, d, 8)
		if err != nil {
			t.Fatal(err)
		}
		got := map[string]string{}
		for _, k := range []string{EnvPrefetchK, EnvPrefetchN, EnvNoPrefetch} {
			if v, ok := cfg.Env[k]; ok {
				got[k] = v
			}
		}
		if len(got) != len(c.env) {
			t.Errorf("QD %d env %v, want %v", c.qd, got, c.env)
		}
		for k, v := range c.env {
			if got[k] != v {
				t.Errorf("QD %d env %v, want %v", c.qd, got, c.env)
			}
		}
		e := echoOf(cfg.Prefetch, cfg.QD)
		if !c.on { // GL-21: the engine's half after ready, over a fake header line (F-PF §6.4)
			eng := parseEngineHeader([]byte(`{"type":"header","prefetch_on":false,"prefetch_k":0,"prefetch_n":0,"prefetch_off_reason":"unset"}`))
			if l := PostReadyLine(eng, &e); !strings.Contains(l, "engine prefetch_on=false prefetch_k=0 prefetch_n=0 prefetch_off_reason=unset") ||
				!strings.HasSuffix(l, "candidate off(reason=qd_below_prefetch_depth)") {
				t.Errorf("after ready: %s", l)
			}
		}
		if c.on != cfg.Prefetch.On {
			t.Errorf("QD %d on %v", c.qd, cfg.Prefetch.On)
		}
		if !c.on {
			if cfg.Prefetch.Echo != "off(reason=qd_below_prefetch_depth)" || !cfg.Prefetch.QDDemoted || e.RequestedK != nil ||
				!strings.Contains(strings.Join(e.StatusLines(), "\n"), "off_reason=qd_below_prefetch_depth") {
				t.Errorf("QD %d shown %+v", c.qd, e)
			}
		}
	}
}

// S8-PF-03: shell MOE_DIRECT_PREFETCH_K=3 and _N=1 with settings 5 and 2 give the child K=5,
// N=2; without settings both keys are absent (§2.4).
func TestS8PF03(t *testing.T) {
	t.Setenv(EnvPrefetchK, "3")
	t.Setenv(EnvPrefetchN, "1")
	cat, p := realProfile(t, "qwen35-35b")
	lines := func(cfg *Config) map[string]string {
		m := map[string]string{}
		for _, l := range explicitEnv(cfg.Env, os.Getenv) {
			if i := strings.IndexByte(l, '='); i > 0 {
				m[l[:i]] = l[i+1:]
			}
		}
		return m
	}
	d := resolvePrefetchManual(p, true, armNone, 8, SelectionUnpinned, "", manualInput{pair: &ManualPair{K: 5, N: 2}, source: SourceArguments})
	cfg, err := build(t, cat, p, map[string]string{"prefetch_k": "5", "prefetch_n": "2"}, d, 8)
	if err != nil {
		t.Fatal(err)
	}
	if m := lines(cfg); m[EnvPrefetchK] != "5" || m[EnvPrefetchN] != "2" {
		t.Errorf("with settings: %v %v", m[EnvPrefetchK], m[EnvPrefetchN])
	}
	d = resolvePrefetch(p, true, armNone, 8, SelectionUnpinned, "")
	cfg, err = build(t, cat, p, map[string]string{}, d, 8)
	if err != nil {
		t.Fatal(err)
	}
	if m := lines(cfg); hasKey(m, EnvPrefetchK) || hasKey(m, EnvPrefetchN) {
		t.Errorf("without settings: %v", m)
	}
}

func hasKey(m map[string]string, k string) bool { _, ok := m[k]; return ok }

// S8-PF-04: across a manual run, a catalog run and an init run (derived profile), the status
// lines requested_k, requested_n and the provenance label equal the effective record; the
// three labels are kept and the source field names the surface (§4).
func TestS8PF04(t *testing.T) {
	ctx := context.Background()
	compare := func(t *testing.T, name string, st StartSettings, wantProv, wantSource string) {
		t.Helper()
		if st.PrefetchEcho == nil {
			t.Fatalf("%s: no echo in the record", name)
		}
		b, _ := json.Marshal(st) // the effective record as the history stores it
		var rec struct {
			Echo map[string]any `json:"prefetch_echo"`
		}
		if err := json.Unmarshal(b, &rec); err != nil {
			t.Fatal(err)
		}
		lines := st.PrefetchEcho.StatusLines()
		for _, f := range []string{"requested_k", "requested_n", "prefetch_provenance", "prefetch_source"} {
			got, ok := statusField(lines, f)
			want := "(null)"
			switch v := rec.Echo[f].(type) {
			case float64:
				want = strconv.FormatInt(int64(v), 10)
			case string:
				if v != "" {
					want = v
				}
			}
			if !ok || got != want {
				t.Errorf("%s: %s status %q, record %q", name, f, got, want)
			}
		}
		if p, _ := statusField(lines, "prefetch_provenance"); p != wantProv {
			t.Errorf("%s: provenance %q, want %q", name, p, wantProv)
		}
		if s, _ := statusField(lines, "prefetch_source"); s != wantSource {
			t.Errorf("%s: source %q, want %q", name, s, wantSource)
		}
	}

	f := pfFixture(t)
	t.Run("manual-per-surface", func(t *testing.T) {
		pair := map[string]string{"prefetch_k": "2", "prefetch_n": "1"}
		res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, Settings: pair})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "screen", res.Start, ProvEnvOverride, SourceSettingsScreen)
		if res.Config.Prefetch.CandidateActivation != "opt-in-fixed" || !contains(res.Notices, NoticeManualUnmeasured) {
			t.Errorf("manual decision %+v notices %v", res.Config.Prefetch, res.Notices)
		}
		if !contains(res.Start.PrefetchEcho.StatusLines(), "                     "+EchoCandidateOnly) {
			t.Errorf("no candidate-only line")
		}
		res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, FromArguments: true, Settings: pair})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "arguments", res.Start, ProvEnvOverride, SourceArguments)
		writeLA35Preset(t, f, pair)
		res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
		if err != nil || res.PresetState != PresetApplies {
			t.Fatal(err, res.PresetState)
		}
		compare(t, "profile", res.Start, ProvEnvOverride, SourceProfile)
		os.Remove(f.l.PresetPath("fixture-profile", f.modelPath))
	})
	t.Run("catalog", func(t *testing.T) {
		res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "catalog off", res.Start, "(null)", "(null)")
		if pe := res.Start.PrefetchEcho; pe.PrefetchInitVersion != PrefetchInitVersion || pe.OffReason != "not_opted_in_evidence_observe" {
			t.Errorf("catalog echo %+v", pe)
		}
		cat, p := realProfile(t, "qwen35-122b-nonextn")
		d := resolvePrefetch(p, true, armNone, 8, SelectionPinned, "catalog")
		cfg, err := build(t, cat, p, map[string]string{}, d, 8)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "catalog on", startSettingsOf(cfg, RepackPacked), ProvCatalog, "(null)")
	})
	t.Run("init-derived", func(t *testing.T) {
		rec, models := la31Load(t)
		g := newFixture(t, false)
		mdir := filepath.Join(g.dir, "models", "tpl")
		model := filepath.Join(mdir, "gpt-oss-tpl.gguf")
		b, err := os.ReadFile(filepath.Join(models, "gpt-oss-tpl.gguf"))
		if err != nil {
			t.Fatal(err)
		}
		writeModel(t, model, b)
		g.l.deriveExec = &la31PlanExec{text: map[string]string{"gpt-oss-tpl.gguf": la31Plan(t, "gpt-oss")}}
		g.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true})
		out := filepath.Join(mdir, "repack")
		writeJSON(t, filepath.Join(out, DerivedExpectFileName), map[string]any{"derived_from": "gpt-oss@1"})
		dsha, _ := fileSha256(filepath.Join(out, DerivedExpectFileName))
		os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, out, rec.Derived["gpt-oss"].LockID, dsha)
		res, err := g.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true, Settings: map[string]string{"prefetch": "init"}})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "init derived", res.Start, "(null)", SourceSettingsScreen)
		if res.Start.PrefetchEcho.OffReason != "phase4_hold_unresolved" || res.Start.PrefetchEcho.PrefetchRequest != "init" {
			t.Errorf("init echo %+v", res.Start.PrefetchEcho)
		}
	})
}

// S8-PF-05: a manual pair on each of the five held rows passes with the unmeasured notice; the
// catalog, init and adapt cases are unchanged (§5.2's table).
func TestS8PF05(t *testing.T) {
	table := []struct{ id, catalog string }{
		{"gpt-oss-120b", "not_opted_in_evidence_observe"},
		{"qwen35-35b", "not_opted_in_evidence_observe"},
		{"qwen35-397b", "not_opted_in_evidence_unverified"},
		{"kimi-k2.6-ram-447gb", "not_opted_in_evidence_unverified"},
		{"deepseek-v4-flash", "not_opted_in_evidence_observe"},
	}
	for _, r := range table {
		_, p := realProfile(t, r.id)
		if a := catalogPrefetchAxes(p); !a.hold || a.activation != "off" {
			t.Fatalf("%s is not a held row: %+v", r.id, a)
		}
		if d := resolvePrefetch(p, true, armNone, 8, SelectionPinned, "catalog"); d.On || d.OffReason != r.catalog {
			t.Errorf("%s catalog: %+v", r.id, d)
		}
		if d := resolvePrefetch(p, true, armInit, 8, SelectionPinned, "init"); d.On || d.OffReason != "phase4_hold_unresolved" {
			t.Errorf("%s init: %+v", r.id, d)
		}
		d := resolvePrefetchManual(p, true, armNone, 8, SelectionPinned, "", manualInput{pair: &ManualPair{K: 4, N: 2}, source: SourceSettingsScreen})
		if !d.On || d.Provenance != ProvEnvOverride || d.Notice != NoticeManualUnmeasured || d.K != 4 || d.N != 2 {
			t.Errorf("%s manual: %+v", r.id, d)
		}
	}
	// adapt: no Go surface offers it (§2.2), so its case stays the PowerShell one, unreachable.
	if _, err := checkSetting("prefetch", "adapt", nil); err == nil {
		t.Error("adapt offered")
	}
}

// S8-PF-06: the qwen122 precedence (§3.3; OQ-2 (a), OQ-3 (a)): a set pair is the run's pair,
// an init request beside it is ignored with its warning; init alone keeps K8/N4 and its warning.
func TestS8PF06(t *testing.T) {
	_, p := realProfile(t, "qwen35-122b-nonextn")
	for _, c := range []struct {
		name, request string
		pair          *ManualPair
		k, n          int64
		prov, warn    string
		notice        string
	}{
		{"pair", "", &ManualPair{K: 6, N: 2}, 6, 2, ProvEnvOverride, "", NoticeManualUnmeasured},
		{"catalog pair set", "", &ManualPair{K: 8, N: 4}, 8, 4, ProvEnvOverride, "", ""},
		{"init with pair", "init", &ManualPair{K: 6, N: 2}, 6, 2, ProvEnvOverride, WarnInitIgnoredPair, NoticeManualUnmeasured},
		{"init alone", "init", nil, 8, 4, ProvCatalog, WarnInitIgnoredCatalogFixed, ""},
		{"catalog", "", nil, 8, 4, ProvCatalog, "", ""},
	} {
		d := resolvePrefetchManual(p, true, optInOf(c.request), 8, SelectionPinned, c.request, manualInput{pair: c.pair, source: SourceArguments})
		if !d.On || d.K != c.k || d.N != c.n || d.Provenance != c.prov || d.Warning != c.warn || d.Notice != c.notice {
			t.Errorf("%s: %+v", c.name, d)
		}
		if c.pair != nil && (d.CandidateActivation != "opt-in-fixed" || d.InitVersion != "") {
			t.Errorf("%s: a manual pair is neither init nor catalog: %+v", c.name, d)
		}
	}
	if d := resolvePrefetchManual(p, true, armNone, 8, SelectionUnpinned, "", manualInput{}); d.On || d.OffReason != "identity_not_exact" {
		t.Errorf("unpinned without a pair: %+v", d)
	}
}

// S8-PF-07: a fixture whose n_expert (read from the GGUF) is below K is refused before start,
// with no K or N in an env (§9 C6).
func TestS8PF07(t *testing.T) {
	f := pfFixture(t)
	res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true, FromArguments: true,
		Settings: map[string]string{"prefetch_k": "8", "prefetch_n": "1"}})
	if StatusOf(err) != StatusFailCustomArgs || ExitCodeOf(err) != 3 || !strings.Contains(err.Error(), "K 8 is above this model's n_expert 4") ||
		!strings.Contains(err.Error(), "-prefetch-k = 8, allowed 1..4") {
		t.Fatalf("refusal %v", err)
	}
	if res != nil && res.Config != nil {
		t.Errorf("an env was built: %v", res.Config.Env)
	}
	res, err = f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true, Settings: map[string]string{"prefetch_k": "4", "prefetch_n": "1"}})
	if err != nil || res.Config.Env[EnvPrefetchK] != "4" {
		t.Errorf("K = n_expert: %v", err)
	}
}

// S8-PF-09: no Go settings path yields the adapt request, and the child env never holds
// MOE_DIRECT_PREFETCH_ACTIVATION_GRANT (§0.2 B3, §2.2, §3.4).
func TestS8PF09(t *testing.T) {
	if contains(PrefetchRequests, "adapt") {
		t.Errorf("offered %v", PrefetchRequests)
	}
	for _, path := range CustomValuePaths[:2] {
		if _, _, err := resolveOverrides(map[string]string{"prefetch": "adapt"}, nil, false, path); StatusOf(err) != StatusFailCustomArgs {
			t.Errorf("%s: adapt %v", path, err)
		}
	}
	f := pfFixture(t)
	rows, _ := f.l.Settings("fixture-profile")
	for _, r := range rows {
		if r.Key == "prefetch" && strings.Join(r.Options, "|") != "catalog|init" {
			t.Errorf("screen offers %v", r.Options)
		}
	}
	writeLA35Preset(t, f, map[string]string{"prefetch": "adapt"})
	if _, ps := f.l.LoadPreset(context.Background(), f.modelPath, "fixture-profile"); ps != PresetInvalid {
		t.Errorf("a stored adapt is used: %v", ps)
	}
	t.Setenv(EnvActivationGrant, "1")
	cat, p := realProfile(t, "qwen35-122b-nonextn")
	q := *p
	q.DefaultEnv = map[string]string{EnvActivationGrant: "1"}
	cfg, err := build(t, cat, &q, map[string]string{}, resolvePrefetch(&q, true, armNone, 8, SelectionPinned, ""), 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range explicitEnv(cfg.Env, os.Getenv) {
		if strings.HasPrefix(strings.ToUpper(l), EnvActivationGrant+"=") {
			t.Errorf("the child env holds %s", l)
		}
	}
}

// S8-PF-10 (with GL-16's finding 3): per-model settings with K and N survive save and reload;
// an invalid stored K discards the stored settings with one log line; the stored key list
// holds the new keys.
func TestS8PF10(t *testing.T) {
	ctx := context.Background()
	f := pfFixture(t)
	t.Run("save-and-reload", func(t *testing.T) {
		if _, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"prefetch_k": "3", "prefetch_n": "2", "ctx": "4096"}); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(f.l.PresetPath("fixture-profile", f.modelPath))
		if err != nil {
			t.Fatal(err)
		}
		var stored Preset
		json.Unmarshal(b, &stored)
		if stored.Values["prefetch_k"] != "3" || stored.Values["prefetch_n"] != "2" || len(stored.Values) != 2 {
			t.Errorf("stored values %v", stored.Values)
		}
		g, err := New(Options{CatalogPath: f.catalogPath, BundleRoot: f.root, StateDir: f.stateDir})
		if err != nil {
			t.Fatal(err)
		}
		g.os, g.lockPrefix = f.l.os, f.l.lockPrefix
		res, err := g.Preflight(ctx, Request{ModelPath: f.modelPath})
		if err != nil || res.PresetState != PresetApplies || res.Config.Prefetch.K != 3 || res.Config.Prefetch.N != 2 ||
			res.Config.Prefetch.Source != SourceProfile {
			t.Fatalf("reload: %v %v %+v", err, res.PresetState, res.Config.Prefetch)
		}
	})
	t.Run("invalid-stored-K-discarded", func(t *testing.T) {
		for name, vals := range map[string]map[string]string{
			"K 17":    {"prefetch_k": "17", "prefetch_n": "2"},
			"K 8x":    {"prefetch_k": "8x", "prefetch_n": "2"},
			"K alone": {"prefetch_k": "3"},
			"K above": {"prefetch_k": "8", "prefetch_n": "2"}, // n_expert 4
		} {
			before := diagKinds(t, f.l, "PRESET_DISCARDED")
			writeLA35Preset(t, f, vals)
			res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
			if err != nil || res.PresetState != PresetInvalid || res.Config.Prefetch.Provenance == ProvEnvOverride {
				t.Errorf("%s: %v %v", name, err, res.PresetState)
			}
			f.l.Preflight(ctx, Request{ModelPath: f.modelPath}) // the same file again: no second line
			if n := diagKinds(t, f.l, "PRESET_DISCARDED") - before; n != 1 {
				t.Errorf("%s: %d log lines", name, n)
			}
		}
	})
	t.Run("stored-key-list", func(t *testing.T) {
		for _, k := range []string{"prefetch_k", "prefetch_n"} {
			if !presetKey(k) || !contains(SettingKeys, k) {
				t.Errorf("%s is not a stored key", k)
			}
		}
	})
}

// S8-PF-16 (core part; the screen text is wire's TestS8PF16, the engine's values after ready
// GL-21's): the rows give K 1..min(16, n_expert) and N 1..QD-1 as an on range, not a refusal.
func TestS8PF16(t *testing.T) {
	f := pfFixture(t)
	rows, err := f.l.Settings("fixture-profile")
	if err != nil {
		t.Fatal(err)
	}
	by := map[string]SettingRow{}
	for _, r := range rows {
		by[r.Key] = r
	}
	if k := by["prefetch_k"]; !k.Bounded || k.Min != 1 || k.Max != 4 { // n_expert 4
		t.Errorf("K row %+v", k)
	}
	if n := by["prefetch_n"]; n.Bounded || !n.OnRange || n.Min != 1 {
		t.Errorf("N row %+v", n)
	}
	if _, _, err := ParseTyped(by["prefetch_k"], "5"); err == nil || !strings.Contains(err.Error(), "allowed 1 ~ 4") {
		t.Errorf("typed K above n_expert: %v", err)
	}
	if v, _, err := ParseTyped(by["prefetch_n"], "40"); err != nil || v != "40" {
		t.Errorf("typed N above QD-1 is accepted (off at start): %q %v", v, err)
	}
	_, p := realProfile(t, "qwen35-122b-nonextn")
	for _, r := range settingRows(p.DefaultArgv, p.Bounds, p.Identify.NExpert) {
		if r.Key == "prefetch_k" && r.Max != 16 {
			t.Errorf("K max on a 256-expert model: %+v", r)
		}
	}
	// GL-21: the engine's values after ready (F-PF §8.5; PLAN-FIX4 A2-1), over a fake metrics
	// line and through a start whose fake child writes the header.
	t.Run("read-back-fake-metrics-line", pf16ReadBack)
	t.Run("read-back-at-ready", pf16Live)
}
