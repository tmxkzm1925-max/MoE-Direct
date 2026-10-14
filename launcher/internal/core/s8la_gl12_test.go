package core

// GL-12 (version 8): scenario S8-LA-26 of the plan reviews/_plan_8_STAGE1_v8-plansplit.md
// (d1dbe526f), built to the F-LS edition reviews/_spec_8_F-LS_launcher_go_edition.md (d3a5ab2ed)
// 4.7 and 12.4 (rule 17, CF 17): the server argv binds 127.0.0.1 for every catalog model and
// every settings path; no setting changes the bind address and no network option exists.

import (
	"context"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// la26Hosts is every bind value the argv carries, in either form.
func la26Hosts(argv []string) []string {
	var out []string
	for i, a := range argv {
		if a == "--host" && i+1 < len(argv) {
			out = append(out, argv[i+1])
		}
		if strings.HasPrefix(a, "--host=") {
			out = append(out, strings.TrimPrefix(a, "--host="))
		}
	}
	return out
}

func la26Bound(t *testing.T, what string, cfg *Config) {
	t.Helper()
	if h := la26Hosts(cfg.Argv); !reflect.DeepEqual(h, []string{BindHost}) || cfg.Host != BindHost ||
		!strings.HasPrefix(cfg.Address(), "http://127.0.0.1:") {
		t.Errorf("%s: hosts %v, Host %q, address %s, argv %q", what, h, cfg.Host, cfg.Address(), cfg.Argv)
	}
	for k := range cfg.Env {
		if strings.EqualFold(k, envArgHost) {
			t.Errorf("%s: env carries %s", what, k)
		}
	}
}

func TestS8LA26(t *testing.T) {
	t.Run("the bind rule on a catalog argv", func(t *testing.T) {
		ok := []struct {
			name string
			in   []string
			want []string
		}{
			{"kept in place", []string{"-np", "1", "--host", "127.0.0.1", "--port", "8093"}, []string{"-np", "1", "--host", "127.0.0.1", "--port", "8093"}},
			{"none: appended", []string{"-np", "1", "--port", "8093"}, []string{"-np", "1", "--port", "8093", "--host", "127.0.0.1"}},
			{"localhost becomes 127.0.0.1", []string{"--host", "localhost", "--port", "8093"}, []string{"--host", "127.0.0.1", "--port", "8093"}},
			{"::1 becomes 127.0.0.1", []string{"--host", "::1"}, []string{"--host", "127.0.0.1"}},
			{"127.0.0.2 becomes 127.0.0.1", []string{"--host", "127.0.0.2"}, []string{"--host", "127.0.0.1"}},
			{"the = form", []string{"--host=127.0.0.1", "--port", "8093"}, []string{"--host", "127.0.0.1", "--port", "8093"}},
			{"a second pair goes", []string{"--host", "127.0.0.1", "-c", "4096", "--host", "localhost"}, []string{"--host", "127.0.0.1", "-c", "4096"}},
		}
		for _, c := range ok {
			got, le := bindLoopback(c.in)
			if le != nil || !reflect.DeepEqual(got, c.want) {
				t.Errorf("%s: %q %v", c.name, got, le)
			}
		}
		bad := []struct {
			name string
			in   []string
			why  string
		}{
			{"any address", []string{"--host", "0.0.0.0"}, "non-loopback host (0.0.0.0)"},
			{"lan address", []string{"--host", "192.168.0.7"}, "non-loopback host (192.168.0.7)"},
			{"ipv6 any", []string{"--host", "::"}, "non-loopback host (::)"},
			{"a later pair would win in the engine", []string{"--host", "127.0.0.1", "--host", "0.0.0.0"}, "non-loopback host (0.0.0.0)"},
			{"the = form", []string{"--host=0.0.0.0"}, "non-loopback host (0.0.0.0)"},
			{"no value", []string{"-np", "1", "--host"}, "ends with --host and no value"},
		}
		for _, c := range bad {
			_, le := bindLoopback(c.in)
			if le == nil || le.Status != StatusFailGateCatalog || !strings.Contains(le.Reason, c.why) {
				t.Errorf("%s: %v", c.name, le)
			}
		}
	})

	t.Run("every catalog model", func(t *testing.T) {
		cat, err := ReadCatalog(realCatalog, la31Expects)
		if err != nil {
			t.Fatal(err)
		}
		if len(cat.Profiles) == 0 {
			t.Fatal("no catalog profile")
		}
		tmp := t.TempDir()
		root := filepath.Join(tmp, "bundle")
		for _, p := range cat.Profiles {
			full := map[string]string{"port": "8100", "ctx": "8192", "threads": "4", "qd": "4", "warmup": "on"}
			if b := p.Bounds["budget_mb"]; b.Min > 0 {
				full["budget_mb"] = fmt.Sprint(b.Min)
			}
			for i, ov := range []map[string]string{{}, full} {
				p := *p
				// A catalog row that also carried the engine's environment form of the bind.
				p.DefaultEnv = map[string]string{"llama_arg_host": "0.0.0.0"}
				cfg, err := buildEffectiveConfig(configInput{cat: cat, profile: &p, root: root,
					outputDir: filepath.Join(tmp, "repack"), modelPath: filepath.Join(tmp, "m.gguf"), ov: ov,
					qd: 8, installed: 131072, geo: SlotGeometry{NExpert: 256, SlotStrideMax: 5308416, Layers: 48},
					metricsDir: filepath.Join(tmp, "logs"), stamp: testStamp})
				if err != nil {
					t.Errorf("%s overrides %d: %v", p.ID, i, err)
					continue
				}
				la26Bound(t, fmt.Sprintf("%s overrides %d", p.ID, i), cfg)
			}
		}
	})

	t.Run("no setting is the bind and no network option exists", func(t *testing.T) {
		for _, k := range append(append([]string(nil), SettingKeys...), CycleOnly...) {
			for _, w := range []string{"host", "bind", "network", "listen", "lan", "addr"} {
				if strings.Contains(strings.ToLower(k), w) {
					t.Errorf("setting key %q", k)
				}
			}
		}
		if !contains(LockedItems, "loopback") || BindHost != "127.0.0.1" {
			t.Errorf("locked %v bind %q", LockedItems, BindHost)
		}
		for _, path := range []CustomValuePath{PathArguments, PathSettingsTyped, PathSavedSettings} {
			for _, k := range []string{"host", "bind", "network"} {
				_, _, err := resolveOverrides(map[string]string{k: "0.0.0.0"}, nil, false, path)
				if ExitCodeOf(err) != 3 || StatusOf(err) != StatusFailCustomArgs {
					t.Errorf("%s %s: %v", path, k, err)
				}
			}
		}
		if presetKey("host") || presetKey("bind") {
			t.Error("a saved setting can hold the bind")
		}
	})

	t.Run("every settings path of a start", func(t *testing.T) {
		f := newFixture(t, false)
		ctx := context.Background()
		full := map[string]string{"port": "8100", "ctx": "8192", "threads": "4", "qd": "4", "warmup": "on", "budget_mb": "5120"}
		for _, req := range []Request{
			{ModelPath: f.modelPath},
			{ModelPath: f.modelPath, Settings: full},
			{ModelPath: f.modelPath, Settings: full, FromArguments: true},
			{ModelPath: f.modelPath, Settings: map[string]string{"ctx": "8192"}, Smoke: SmokeOn},
		} {
			res, err := f.l.Preflight(ctx, req)
			if err != nil || res.Config == nil {
				t.Errorf("%v: %v", req.Settings, err)
				continue
			}
			la26Bound(t, fmt.Sprint(req.Settings), res.Config)
		}
		for _, args := range []bool{false, true} {
			_, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"host": "0.0.0.0"}, FromArguments: args})
			if StatusOf(err) != StatusFailCustomArgs || !strings.Contains(fmt.Sprint(err), "host") {
				t.Errorf("arguments=%v: a host setting %v", args, err)
			}
		}
		// The locked row core keeps (the adapter shows none, wire TestS8LA26) holds BindHost.
		locked, err := f.l.LockedSettings("fixture-profile")
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range locked {
			if r.Key == "host" && (r.Default != BindHost || r.Editable) {
				t.Errorf("host row %+v", r)
			}
		}
		rows, err := f.l.Settings("fixture-profile")
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rows {
			if r.Key == "host" || r.Editable && strings.Contains(r.Key, "host") {
				t.Errorf("an open host row %+v", r)
			}
		}
	})
}
