package core

// GL-20: the virtual lift of the manual-prefetch edition (F-PF §7, §8.2-§8.4), checked by the
// GT rows of its §12: TestS8PF08 (§7.1-§7.2 the notice on each path, §8.2 N_cap) and TestS8PF19
// (§7.3 the refusal below the wave width, its text distinct from the notice).

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// withWidths records wave widths for a profile for one test (the fixture has no X-VQD record).
func withWidths(t *testing.T, id string, per ...int64) {
	t.Helper()
	old, had := virtualWaveWidths[id]
	virtualWaveWidths[id] = WaveWidths{ProfileID: id, PerLayer: per}
	t.Cleanup(func() {
		if had {
			virtualWaveWidths[id] = old
		} else {
			delete(virtualWaveWidths, id)
		}
	})
}

// virtualFixture is the core fixture with a virtual plan that passes the 8-item gate.
func virtualFixture(t *testing.T) *fixture {
	t.Helper()
	f := newFixture(t, false)
	writeVirtualArtifacts(t, filepath.Join(filepath.Dir(f.modelPath), "repack-virtual"), "fixture-profile", f.expectSha)
	return f
}

func countOf(ns []string, s string) int {
	n := 0
	for _, x := range ns {
		if x == s {
			n++
		}
	}
	return n
}

// S8-PF-08: on a virtual repack QD and the pair are accepted on each Go path that carries them
// (the settings screen, the per-model saved settings, the argument-driven mode), each with the
// notice fixed by the user (OQ-5 (b)); the shown N_cap is the formula; N_cap 0 on every layer
// shows its notice; where N_cap differs by layer the screen shows the smallest and the log the
// list per layer.
func TestS8PF08(t *testing.T) {
	ctx := context.Background()
	if NoticeVirtualUnvalidated != "default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured" {
		t.Fatalf("notice text %q", NoticeVirtualUnvalidated)
	}
	t.Run("each-path", func(t *testing.T) {
		f := virtualFixture(t)
		withWidths(t, "fixture-profile", 3, 2)
		set := map[string]string{"qd": "8", "prefetch_k": "2", "prefetch_n": "2"}
		check := func(t *testing.T, name string, res *PreflightResult, err error, source string) {
			t.Helper()
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			cfg := res.Config
			if cfg.QD != 8 || cfg.Env[EnvQD] != "8" || cfg.Env[EnvIOQDTotal] != "8" || !cfg.Prefetch.On || cfg.Prefetch.K != 2 || cfg.Prefetch.N != 2 ||
				cfg.Env[EnvPrefetchK] != "2" || cfg.Env[EnvPrefetchN] != "2" || cfg.Prefetch.Source != source {
				t.Errorf("%s: config %+v env %v", name, cfg.Prefetch, cfg.Env)
			}
			if countOf(res.Notices, NoticeVirtualUnvalidated) != 1 || contains(res.Notices, NoticeManualUnmeasured) || contains(res.Notices, NoticeNCapZero) {
				t.Errorf("%s: notices %v", name, res.Notices)
			}
			// N_cap per layer: W 3 -> min(2, 4/3, (8-4)/3) = 1; W 2 -> min(2, 4/2, (8-4)/2) = 2
			e := res.Start.PrefetchEcho
			if nCapLogLine(cfg.Prefetch.NCapPerLayer) != "prefetch_n_cap_per_layer=1,2" || e.PrefetchNCap == nil || *e.PrefetchNCap != 1 ||
				NCapLine(e) != "prefetch_n_cap_per_layer=1,2" {
				t.Errorf("%s: N_cap %v echo %v", name, cfg.Prefetch.NCapPerLayer, e.PrefetchNCap)
			}
			if v, _ := statusField(e.StatusLines(), "prefetch_n_cap"); v != "1" {
				t.Errorf("%s: status prefetch_n_cap=%s", name, v)
			}
		}
		res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true, Settings: set})
		check(t, "screen", res, err, SourceSettingsScreen)
		res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true, FromArguments: true, Settings: set})
		check(t, "arguments", res, err, SourceArguments)
		writeLA35Preset(t, f, set)
		res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual})
		if err == nil && res.PresetState != PresetApplies {
			t.Fatalf("saved settings %v", res.PresetState)
		}
		check(t, "saved settings", res, err, SourceProfile)
		if diagKinds(t, f.l, "PREFETCH_N_CAP") < 3 {
			t.Error("no PREFETCH_N_CAP log line")
		}
		os.Remove(f.l.PresetPath("fixture-profile", f.modelPath))
		// The notice follows the run's outcome (§7.2): QD alone, init alone; none for the default.
		for name, c := range map[string]struct {
			set    map[string]string
			notice bool
		}{
			"default":     {nil, false},
			"QD 5 alone":  {map[string]string{"qd": "5"}, true},
			"QD 8 given":  {map[string]string{"qd": "8"}, false},
			"init":        {map[string]string{"prefetch": "init"}, true},
			"catalog":     {map[string]string{"prefetch": "catalog"}, false},
			"pair, QD 16": {map[string]string{"qd": "16", "prefetch_k": "4", "prefetch_n": "1"}, true},
		} {
			res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true, Settings: c.set})
			if err != nil || contains(res.Notices, NoticeVirtualUnvalidated) != c.notice {
				t.Errorf("%s: %v %v", name, err, res.Notices)
			}
		}
	})
	t.Run("n-cap-formula", func(t *testing.T) {
		// T17 against the formula of S8-PF-08, restated: min(N, floor(child_max/W_l), floor((qd_total-reserve)/W_l)).
		for w := int64(1); w <= 3; w++ {
			for qd := int64(1); qd <= 63; qd++ {
				for n := int64(1); n <= 16; n++ {
					want := min(n, 4/w)
					if qd > 4 {
						want = min(want, (qd-4)/w)
					} else {
						want = 0
					}
					if got := NCap(n, qd, EngineDemandReserve, EngineChildMax, w); got != want {
						t.Fatalf("NCap(%d, %d, 4, 4, %d) = %d, want %d", n, qd, w, got, want)
					}
				}
			}
		}
		// The worked values of S8-PF-18 (PLAN:479) on qwen35-122b's recorded widths.
		w, ok := waveWidthsFor("qwen35-122b-nonextn")
		if !ok {
			t.Fatal("no record for qwen35-122b")
		}
		for _, c := range []struct{ n, qd, want int64 }{{4, 8, 1}, {2, 6, 0}, {2, 3, 0}, {1, 7, 1}, {4, 16, 1}} {
			per := nCapPerLayer(c.n, c.qd, w)
			if m, _ := nCapMin(per); m != c.want || len(per) != 48 || nCapZeroEverywhere(per) != (c.want == 0) {
				t.Errorf("N %d QD %d: %v", c.n, c.qd, per)
			}
		}
	})
	t.Run("n-cap-zero-every-layer", func(t *testing.T) {
		f := virtualFixture(t)
		withWidths(t, "fixture-profile", 3, 3)
		res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true,
			Settings: map[string]string{"qd": "6", "prefetch_k": "2", "prefetch_n": "2"}})
		if err != nil || !res.Config.Prefetch.On || !contains(res.Notices, NoticeNCapZero) || !contains(res.Notices, NoticeVirtualUnvalidated) {
			t.Fatalf("N_cap 0: %v %v", err, res)
		}
		// differing layers with one zero: the smallest is 0, but not every layer: no notice
		withWidths(t, "fixture-profile", 3, 2)
		res, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true,
			Settings: map[string]string{"qd": "6", "prefetch_k": "2", "prefetch_n": "2"}})
		if err != nil || contains(res.Notices, NoticeNCapZero) || *res.Start.PrefetchEcho.PrefetchNCap != 0 ||
			NCapLine(res.Start.PrefetchEcho) != "prefetch_n_cap_per_layer=0,1" {
			t.Errorf("one layer zero: %v %v", err, res.Notices)
		}
	})
	t.Run("io-knobs", func(t *testing.T) {
		// §7.4: the alias and the dedicated key equal; the three other knobs neither set nor
		// passed, whatever the catalog's env holds; on packed no dedicated key.
		cat, p := realProfile(t, "qwen35-122b-nonextn")
		q := *p
		q.DefaultEnv = map[string]string{"moe_direct_io_qd_demand_reserve": "2", EnvIOQDPrefetchChildMax: "8", EnvIOPartParallelism: "1"}
		t.Setenv(EnvIOQDDemandReserve, "1")
		d := resolvePrefetch(&q, true, armNone, 8, SelectionPinned, "catalog")
		for _, virtual := range []bool{true, false} {
			cfg, err := buildEffectiveConfig(configInput{cat: cat, profile: &q, root: filepath.Clean(stagingRoot),
				outputDir: `D:\moe-models\m\repack-virtual`, modelPath: `D:\moe-models\m\model-00001-of-00002.gguf`,
				ov: map[string]string{"qd": "6"}, decision: d, qd: 8, installed: 131072,
				geo: SlotGeometry{NExpert: 256, SlotStrideMax: 5308416, Layers: 48}, metricsDir: `C:\state\go-draft\logs`,
				stamp: testStamp, virtual: virtual})
			if err != nil {
				t.Fatal(err)
			}
			env := explicitEnv(cfg.Env, os.Getenv)
			joined := strings.ToUpper(strings.Join(env, "\n"))
			if strings.Contains(joined, "IO_QD_DEMAND_RESERVE") || strings.Contains(joined, "IO_QD_PREFETCH_CHILD_MAX") || strings.Contains(joined, "IO_PART_PARALLELISM") {
				if virtual {
					t.Errorf("virtual: a knob is passed: %v", env)
				}
			}
			if virtual && (cfg.Env[EnvQD] != "6" || cfg.Env[EnvIOQDTotal] != "6") {
				t.Errorf("virtual: QD %s / %s", cfg.Env[EnvQD], cfg.Env[EnvIOQDTotal])
			}
			if !virtual && hasKey(cfg.Env, EnvIOQDTotal) {
				t.Errorf("packed: %s set", EnvIOQDTotal)
			}
		}
	})
	t.Run("packed-no-n-cap", func(t *testing.T) {
		// §8.4: N_cap is a virtual rule.
		f := pfFixture(t)
		withWidths(t, "fixture-profile", 3, 2)
		res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, Settings: map[string]string{"prefetch_k": "2", "prefetch_n": "1"}})
		if err != nil || !res.Config.Prefetch.On || res.Config.Prefetch.NCapPerLayer != nil || res.Start.PrefetchEcho.PrefetchNCap != nil ||
			contains(res.Notices, NoticeVirtualUnvalidated) {
			t.Errorf("packed: %v %+v", err, res.Config.Prefetch)
		}
	})
}

// S8-PF-19: per model the wave width X-VQD records; a virtual QD below it is refused before start
// as «cannot work», a text distinct from the unmeasured notice; QD equal to the width is accepted
// with the notice.
func TestS8PF19(t *testing.T) {
	ctx := context.Background()
	record := map[string]struct {
		layers int
		w      int64
	}{ // End's X-VQD landing, section 1 (2f0012a07)
		"qwen35-122b-nonextn": {48, 3}, "deepseek-v4-flash": {43, 3}, "gpt-oss-120b": {36, 3},
	}
	if len(virtualWaveWidths) != len(record) {
		t.Errorf("widths %d, X-VQD recorded %d", len(virtualWaveWidths), len(record))
	}
	for id, r := range record {
		w, ok := waveWidthsFor(id)
		if !ok || len(w.PerLayer) != r.layers || w.Max() != r.w {
			t.Fatalf("%s: %+v", id, w)
		}
		t.Run(id, func(t *testing.T) {
			f := virtualFixture(t)
			withWidths(t, "fixture-profile", w.PerLayer[:2]...)
			for q := int64(1); q < r.w; q++ {
				qs := strconv.FormatInt(q, 10)
				want := "cannot work: QD " + qs + " is below this model's virtual wave width " + strconv.FormatInt(r.w, 10)
				for _, args := range []bool{true, false} {
					res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true, FromArguments: args,
						Settings: map[string]string{"qd": qs}})
					if StatusOf(err) != StatusFailCustomArgs || ExitCodeOf(err) != 3 || !strings.Contains(err.Error(), want) ||
						strings.Contains(err.Error(), NoticeVirtualUnvalidated) || (res != nil && res.Config != nil) {
						t.Errorf("QD %d (arguments %v): %v", q, args, err)
					}
				}
				writeLA35Preset(t, f, map[string]string{"qd": qs}) // a stored QD below the width: the start refuses it
				if _, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual}); !strings.Contains(fmtErr(err), want) {
					t.Errorf("stored QD %d: %v", q, err)
				}
				os.Remove(f.l.PresetPath("fixture-profile", f.modelPath))
			}
			res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, RepackMode: RepackVirtual, NoPreset: true,
				Settings: map[string]string{"qd": strconv.FormatInt(r.w, 10)}})
			if err != nil || res.Config.QD != int(r.w) || !contains(res.Notices, NoticeVirtualUnvalidated) {
				t.Errorf("QD = width: %v %v", err, res)
			}
		})
	}
	if strings.Contains(refuseWaveText(2, 3), "unmeasured") || refuseWaveText(2, 3) == NoticeVirtualUnvalidated {
		t.Error("the refusal reads as the notice")
	}
	// Packed has no wave rule; a profile without a record has no launcher-side check (the engine
	// still refuses, MD:2995-3001).
	g := virtualFixture(t)
	if _, err := g.l.Preflight(ctx, Request{ModelPath: g.modelPath, RepackMode: RepackVirtual, NoPreset: true, Settings: map[string]string{"qd": "2"}}); err != nil {
		t.Errorf("no record, QD 2: %v", err)
	}
	f := pfFixture(t)
	withWidths(t, "fixture-profile", 3, 3)
	if _, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, Settings: map[string]string{"qd": "2"}}); err != nil {
		t.Errorf("packed QD 2: %v", err)
	}
}

func fmtErr(err error) string {
	if err == nil {
		return "<nil>"
	}
	return err.Error()
}
