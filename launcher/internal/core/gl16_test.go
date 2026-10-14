package core

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// GL-16: the per-model settings, rule 2's allowlist and the custom-value texts per path
// (F-LS 4.1, 5.3, 6.3, 7.1-7.2, 11.4, 12.4-12.6).

// S8-LA-28: the changeable settings are exactly rule 2's allowlist plus the user's decision;
// the KV type, the gates, -np 1 and the loopback bind stay locked; a value row takes a typed
// value checked against its range; prefetch and the repack mode only cycle.
func TestS8LA28(t *testing.T) {
	t.Run("open-and-locked", func(t *testing.T) {
		// F-LS 12.4 OpenSettings, LockedSettings, CycleOnly, WarmStartSwitches, in order.
		want := []string{"port", "ctx", "threads", "warmup", "budget_mb", "qd", "prefetch", "prefetch_k", "prefetch_n", "repack_mode", "prompt_cache_cap"}
		if !reflect.DeepEqual(OpenSettings, want) {
			t.Errorf("open %v, want %v", OpenSettings, want)
		}
		if !reflect.DeepEqual(LockedItems, []string{"kv-type", "verify-expects", "np-1", "loopback"}) {
			t.Errorf("locked %v", LockedItems)
		}
		if !reflect.DeepEqual(CycleOnly, []string{"prefetch", "repack_mode"}) {
			t.Errorf("cycle only %v", CycleOnly)
		}
		if !reflect.DeepEqual(WarmStartSwitches, []string{"warmstart", "autosave"}) {
			t.Errorf("switches %v", WarmStartSwitches)
		}
		// The request's model keys are the open set without the repack mode (its own field) and
		// the cap (a launcher setting); the switches are outside the allowlist (OQ-6 (a)).
		var model []string
		for _, k := range OpenSettings {
			if k != "repack_mode" && k != "prompt_cache_cap" {
				model = append(model, k)
			}
		}
		if !reflect.DeepEqual(SettingKeys, model) {
			t.Errorf("model keys %v, want %v", SettingKeys, model)
		}
		for _, k := range WarmStartSwitches {
			if contains(OpenSettings, k) || contains(SettingKeys, k) {
				t.Errorf("switch %s in the allowlist", k)
			}
		}
		f := newFixture(t, false)
		rows, err := f.l.Settings("fixture-profile")
		if err != nil {
			t.Fatal(err)
		}
		var keys []string
		for _, r := range rows {
			keys = append(keys, r.Key)
			typed := contains(TypedKeys, r.Key)
			if contains(CycleOnly, r.Key) && typed {
				t.Errorf("%s cycles only but takes a typed value", r.Key)
			}
		}
		if !reflect.DeepEqual(keys, append(append([]string{}, SettingKeys...), "repack_mode")) {
			t.Errorf("rows %v", keys)
		}
		locked, err := f.l.LockedSettings("fixture-profile")
		if err != nil {
			t.Fatal(err)
		}
		var lk []string
		for _, r := range locked {
			if r.Editable || r.Locked == "" {
				t.Errorf("locked row open: %+v", r)
			}
			lk = append(lk, r.Key)
		}
		if !reflect.DeepEqual(lk, []string{"kv_type", "verify_gates", "parallel", "host"}) {
			t.Errorf("locked rows %v", lk)
		}
	})

	t.Run("effective-argv-and-env", func(t *testing.T) {
		f := newFixture(t, false)
		ctx := context.Background()
		base, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true})
		if err != nil {
			t.Fatal(err)
		}
		// Each open key changes only its own flag or variable; every other token of the
		// catalog argv and env, the locked ones among them, stays the catalog's.
		cases := []struct {
			key, value string
			flags      []string // argv flags whose value may change
			env        []string // env variables that may change
		}{
			{"port", "8101", []string{"--port"}, nil},
			{"ctx", "8192", []string{"-c"}, nil},
			{"threads", "7", []string{"-t"}, nil},
			{"budget_mb", "4096", nil, []string{EnvBudgetMB}},
			{"qd", "4", nil, []string{EnvQD, EnvPrefetchK, EnvPrefetchN, EnvNoPrefetch}},
			{"warmup", "off", nil, nil},
			{"prefetch", "init", nil, []string{EnvPrefetchK, EnvPrefetchN, EnvNoPrefetch}},
		}
		for _, c := range cases {
			res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, Settings: map[string]string{c.key: c.value}})
			if err != nil {
				t.Errorf("%s=%s: %v", c.key, c.value, err)
				continue
			}
			a, b := base.Config.Argv, res.Config.Argv
			if len(a) != len(b) {
				t.Errorf("%s: argv %v -> %v", c.key, a, b)
				continue
			}
			changed := false
			for i := range a {
				if a[i] == b[i] {
					continue
				}
				if i == 0 || !contains(c.flags, a[i-1]) {
					t.Errorf("%s: argv[%d] %q -> %q outside its flag", c.key, i, a[i], b[i])
				}
				changed = true
			}
			if len(c.flags) > 0 && !changed {
				t.Errorf("%s: the value did not reach the argv", c.key)
			}
			for k, v := range res.Config.Env {
				if k == EnvMetrics { // GL-06: per start, stamped to the second
					continue
				}
				if base.Config.Env[k] != v && !contains(c.env, k) {
					t.Errorf("%s: env %s %q -> %q", c.key, k, base.Config.Env[k], v)
				}
			}
			for k := range base.Config.Env {
				if _, ok := res.Config.Env[k]; !ok && !contains(c.env, k) {
					t.Errorf("%s: env %s dropped", c.key, k)
				}
			}
		}
		// The locked items keep the catalog's values: one slot, loopback.
		if v, _ := argvValue(base.Config.Argv, "-np"); v != "1" {
			t.Errorf("-np %q", v)
		}
		if base.Config.Host != BindHost {
			t.Errorf("host %q", base.Config.Host)
		}
		// Deny by default: a key outside the allowlist and a locked item are refused (3). K and N
		// are open since GL-19 (F-PF): one alone is refused as half a pair (§9 C3).
		for _, k := range []string{"n_gpu_layers", "kv_type", "host", "np"} {
			_, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, Settings: map[string]string{k: "1"}})
			wantStatus(t, err, StatusFailCustomArgs, "not a changeable setting")
		}
		for _, k := range []string{"prefetch_k", "prefetch_n"} {
			_, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath, NoPreset: true, Settings: map[string]string{k: "2"}})
			wantStatus(t, err, StatusFailCustomArgs, "K and N are set together or not at all")
		}
	})

	t.Run("typed-range-verdicts", func(t *testing.T) {
		f := newFixture(t, false)
		rows, _ := f.l.Settings("fixture-profile")
		row := map[string]SettingRow{}
		for _, r := range rows {
			row[r.Key] = r
		}
		for _, c := range []struct{ key, in, want string }{
			{"ctx", "64k", "65536"}, {"ctx", "65536", "65536"}, {"port", "8101", "8101"}, {"threads", "8", "8"},
			{"qd", "8", "8"}, {"budget_mb", "8", "8192"}, {"warmup", `C:\w.txt`, `file:C:\w.txt`},
		} {
			if v, _, err := ParseTyped(row[c.key], c.in); err != nil || v != c.want {
				t.Errorf("%s %q: %q %v", c.key, c.in, v, err)
			}
		}
		for _, c := range []struct{ key, in, rng string }{
			{"ctx", "1k", "2k ~ 256k"}, {"port", "80", "1024 ~ 65535"}, {"qd", "64", "1 ~ 63"},
		} {
			_, _, err := ParseTyped(row[c.key], c.in)
			want := "outside its range · allowed " + c.rng + " · the value is kept, nothing applied"
			if err == nil || err.Error() != want {
				t.Errorf("%s %q: %v, want %q", c.key, c.in, err, want)
			}
		}
		for _, k := range CycleOnly {
			if _, _, err := ParseTyped(SettingRow{Key: k}, "x"); err == nil || !strings.Contains(err.Error(), "not typed") {
				t.Errorf("%s typed: %v", k, err)
			}
		}
		// The prompt-cache cap, 0 = off (U-22): no choice, off, 0, the cap rule's bounds; no
		// choice is the off form (DECISION_LEDGER.md:4420), so «auto» is no value.
		for in, want := range map[string]CapSetting{"": {}, "off": {true, 0}, "0": {true, 0}, "256": {true, 256}, "8192": {true, 8192}, "1000": {true, 1000}} {
			if c, err := ParseCap(in); err != nil || c != want {
				t.Errorf("cap %q: %+v %v", in, c, err)
			}
		}
		for c, want := range map[CapSetting]string{{}: "off", {true, 0}: "off", {true, 4096}: "4096"} {
			if c.String() != want {
				t.Errorf("cap %+v shows %q", c, c.String())
			}
		}
		if CapChoices[0] != "0" {
			t.Errorf("the cycle starts at the default off form: %v", CapChoices)
		}
		for _, in := range []string{"100", "8193", "-1", "lots", "auto"} {
			_, err := ParseCap(in)
			want := "outside its range · allowed 0 (off) or 256..8192 MiB · the value is kept, nothing applied"
			if err == nil || err.Error() != want {
				t.Errorf("cap %q: %v", in, err)
			}
		}
		for in, want := range map[string]bool{"on": true, "off": false, "1": true, "false": false} {
			if v, err := ParseSwitch("warmstart", in); err != nil || v != want {
				t.Errorf("switch %q: %v %v", in, v, err)
			}
		}
		if _, err := ParseSwitch("autosave", "5"); err == nil {
			t.Error("autosave takes on or off")
		}
	})

	t.Run("launcher-settings-at-start", func(t *testing.T) {
		// The cap and the switches go with the start and are recorded with it (rows 6.1-6.2);
		// they change no model setting.
		f := newFixture(t, false)
		res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true,
			PromptCacheCap: CapSetting{Set: true, Value: 1024}, Switches: &Switches{WarmStart: false, Autosave: true}})
		if err != nil {
			t.Fatal(err)
		}
		m := res.Start.SettingsMap()
		if m["prompt_cache_cap"] != "1024" || m["warmstart"] != "off" || m["autosave"] != "on" {
			t.Errorf("recorded %v", m)
		}
		def, _ := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true})
		if dm := def.Start.SettingsMap(); dm["prompt_cache_cap"] != "off" || dm["warmstart"] != "on" || dm["autosave"] != "on" {
			t.Errorf("defaults %v", dm)
		}
		if v, _ := argvValue(res.Config.Argv, "--cache-ram"); v != "1024" {
			t.Errorf("the cap is not in the argv: %v", res.Config.Argv)
		}
		if !reflect.DeepEqual(StripCacheArgs(def.Config.Argv), StripCacheArgs(res.Config.Argv)) {
			t.Errorf("a launcher setting changed the model argv")
		}
	})
}

// S8-LA-29 (GL-16's part; GL-08 adds the run modes and arguments): every path that refuses a
// custom value describes its own surface; no path says non-interactive.
func TestS8LA29(t *testing.T) {
	// F-LS 11.4 as written.
	texts := map[CustomValuePath][2]string{
		PathArguments:     {"a value given as a command-line argument failed its type or range check: <name> = <value>, allowed <range>", ""},
		PathSettingsTyped: {"outside its range · allowed <range> · the value is kept, nothing applied", "범위 밖 · 허용 <range> · 값은 그대로, 적용 안 됨"},
		PathSavedSettings: {"the saved settings of this model hold a value that failed its check · not used · the defaults apply", "이 모델의 저장된 설정에 검사를 통과하지 못한 값이 있음 · 쓰지 않음 · 기본값 적용"},
	}
	t.Run("text-table", func(t *testing.T) {
		if !reflect.DeepEqual(CustomValuePaths, []CustomValuePath{"arguments", "settings-typed", "saved-settings"}) {
			t.Errorf("paths %v", CustomValuePaths)
		}
		for p, w := range texts {
			if g := CustomValueTemplates[p]; g.EN != w[0] || g.KO != w[1] {
				t.Errorf("%s: %q / %q", p, g.EN, g.KO)
			}
		}
		if StatusHint(StatusFailCustomArgs) != "a value given to the argument-driven mode failed its type or range check" {
			t.Errorf("hint %q", StatusHint(StatusFailCustomArgs))
		}
		for p, w := range texts {
			for _, s := range w {
				if strings.Contains(strings.ToLower(s), "non-interactive") {
					t.Errorf("%s says non-interactive: %q", p, s)
				}
			}
			if p != PathArguments && (strings.Contains(w[0], "command-line") || strings.Contains(w[0], "argument")) {
				t.Errorf("%s names the argument surface: %q", p, w[0])
			}
		}
	})
	t.Run("arguments", func(t *testing.T) {
		f := newFixture(t, false)
		_, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true, FromArguments: true,
			Settings: map[string]string{"budget_mb": "999999"}})
		want := "a value given as a command-line argument failed its type or range check: -budget-mb = 999999, allowed 4096..65536"
		if StatusOf(err) != StatusFailCustomArgs || ExitCodeOf(err) != 3 || !strings.Contains(err.Error(), want) {
			t.Errorf("arguments: %v", err)
		}
	})
	t.Run("settings-typed", func(t *testing.T) {
		_, _, err := ParseTyped(SettingRow{Key: "threads", Min: 1, Max: 64, Bounded: true}, "65")
		if err == nil || err.Error() != "outside its range · allowed 1 ~ 64 · the value is kept, nothing applied" {
			t.Errorf("typed: %v", err)
		}
		// A start from the screen that still meets a bad value names the settings, not the
		// argument-driven mode.
		f := newFixture(t, false)
		_, err = f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath, NoPreset: true,
			Settings: map[string]string{"threads": "65"}})
		if StatusOf(err) != StatusFailCustomArgs || !strings.Contains(err.Error(), "outside its range · allowed 1..64") ||
			strings.Contains(err.Error(), "argument") {
			t.Errorf("screen start: %v", err)
		}
	})
	t.Run("saved-settings", func(t *testing.T) {
		f := newFixture(t, false)
		writeLA35Preset(t, f, map[string]string{"ctx": "999999"})
		res, err := f.l.Preflight(context.Background(), Request{ModelPath: f.modelPath})
		if err != nil {
			t.Fatalf("an invalid saved value must not end the start: %v", err)
		}
		if !contains(res.Notices, texts[PathSavedSettings][0]) || res.PresetState != PresetInvalid {
			t.Errorf("notices %v state %v", res.Notices, res.PresetState)
		}
	})
	// GL-08's part: the argument-driven mode's range refusal and the stop action (s8la_gl08_test.go).
	t.Run("gl08-arguments-and-stop", la29GL08)
}

// writeLA35Preset writes a preset for the fixture model as SavePreset would, with the values
// given as they are (no check).
func writeLA35Preset(t *testing.T, f *fixture, values map[string]string) string {
	t.Helper()
	p := Preset{SchemaVersion: presetSchema, Model: "fixture-profile", ProfileID: "fixture-profile",
		FileName: "fixture-model.gguf", FileBytes: f.modelBytes, Values: values, SavedAt: "2026-10-12T00:00:00Z"}
	path := f.l.PresetPath("fixture-profile", f.modelPath)
	b, _ := json.Marshal(p)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// diagKinds counts the records of a kind in the launcher's diagnostic log.
func diagKinds(t *testing.T, l *Launcher, kind string) int {
	t.Helper()
	fh, err := os.Open(l.diag.Path())
	if err != nil {
		return 0
	}
	defer fh.Close()
	n := 0
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		var rec struct{ Kind string }
		if json.Unmarshal(sc.Bytes(), &rec) == nil && rec.Kind == kind {
			n++
		}
	}
	return n
}

// S8-LA-35: one saved profile per model under the user data folder, keyed by model name and
// file identity, holding only the changed values with the repack mode; the state labels; a
// reload restores them; an invalid stored value discards the model's stored settings with one
// log line.
func TestS8LA35(t *testing.T) {
	ctx := context.Background()
	t.Run("save-and-reload", func(t *testing.T) {
		f := newFixture(t, false)
		p, err := f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{
			"ctx": "8192", "port": "8093", "threads": "auto", "qd": "8", "repack_mode": "virtual", "prefetch": "catalog"})
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(f.stateDir, StatePresetsDir, "fixture-profile.json") // F-LS 6.3, 8.2
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var stored map[string]any
		json.Unmarshal(b, &stored)
		if !reflect.DeepEqual(stored["values"], map[string]any{"ctx": "8192", "qd": "8", "repack_mode": "virtual"}) {
			t.Errorf("stored values %v", stored["values"])
		}
		if stored["model"] != "fixture-profile" || stored["file_name"] != "fixture-model.gguf" ||
			int64(stored["file_bytes"].(float64)) != f.modelBytes {
			t.Errorf("key %v %v %v", stored["model"], stored["file_name"], stored["file_bytes"])
		}
		if _, err := os.Stat(filepath.Join(f.stateDir, "go-draft", "presets")); err == nil {
			t.Error("the draft folder was written")
		}
		// A reload (a new launcher over the same folders) restores them.
		l2, err := New(Options{CatalogPath: f.catalogPath, BundleRoot: f.root, StateDir: f.stateDir})
		if err != nil {
			t.Fatal(err)
		}
		q, st := l2.LoadPreset(ctx, f.modelPath, "fixture-profile")
		if st != PresetApplies || !reflect.DeepEqual(q.Values, p.Values) {
			t.Errorf("reload %v %v", st, q)
		}
	})
	t.Run("state-labels", func(t *testing.T) {
		f := newFixture(t, false)
		pv, _ := f.l.SettingsProvenance(ctx, f.modelPath, "fixture-profile", map[string]string{})
		if pv.State != ProvDefault || pv.Custom {
			t.Errorf("default %+v", pv)
		}
		pv, _ = f.l.SettingsProvenance(ctx, f.modelPath, "fixture-profile", map[string]string{"ctx": "8192"})
		if pv.State != ProvDefaultUnsaved || !reflect.DeepEqual(pv.Changed, []string{"ctx"}) || !pv.Custom {
			t.Errorf("default · not saved %+v", pv)
		}
		f.l.SavePreset(ctx, f.modelPath, "fixture-profile", map[string]string{"ctx": "8192"})
		pv, _ = f.l.SettingsProvenance(ctx, f.modelPath, "fixture-profile", map[string]string{"ctx": "8192"})
		if pv.State != ProvPreset || len(pv.Changed) != 0 || !pv.Custom {
			t.Errorf("my settings %+v", pv)
		}
		pv, _ = f.l.SettingsProvenance(ctx, f.modelPath, "fixture-profile", map[string]string{"ctx": "8192", "qd": "4"})
		if pv.State != ProvPresetUnsaved || !reflect.DeepEqual(pv.Changed, []string{"qd"}) {
			t.Errorf("my settings · not saved %+v", pv)
		}
	})
	t.Run("invalid-value-discarded", func(t *testing.T) {
		for name, values := range map[string]map[string]string{
			"out-of-range": {"ctx": "999999"},
			"bad-mode":     {"repack_mode": "zip"},
			"unknown-key":  {"warmstart": "on"}, // a switch since GL-16, no model setting
			"k-not-built":  {"prefetch_k": "4"},
		} {
			f := newFixture(t, false)
			writeLA35Preset(t, f, values)
			for i := 0; i < 3; i++ {
				if _, st := f.l.LoadPreset(ctx, f.modelPath, "fixture-profile"); st != PresetInvalid {
					t.Errorf("%s: state %v", name, st)
				}
			}
			res, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
			if err != nil || res.Start.Ctx != 4096 || res.RepackMode == "zip" {
				t.Errorf("%s: the defaults must apply: %v %+v", name, err, res.Start)
			}
			if n := diagKinds(t, f.l, "PRESET_DISCARDED"); n != 1 {
				t.Errorf("%s: %d log lines, want 1", name, n)
			}
		}
	})
	t.Run("stored-key-list", func(t *testing.T) {
		for _, k := range append(append([]string{}, SettingKeys...), "repack_mode") {
			if !presetKey(k) {
				t.Errorf("%s not storable", k)
			}
		}
		for _, k := range []string{"warmstart", "autosave", "prompt_cache_cap", "kv_type"} {
			if presetKey(k) {
				t.Errorf("%s storable per model", k)
			}
		}
	})
	t.Run("outside-the-catalog", func(t *testing.T) {
		// A template model (GL-14) saves and reads by file name and size, checked against its
		// structure's defaults; a virtual repack is not one of its modes (PS:13923).
		models := la31Models(t)
		f := newFixture(t, false)
		path := filepath.Join(models, "gpt-oss-tpl.gguf")
		p, err := f.l.SavePreset(ctx, path, "", map[string]string{"ctx": "8192"})
		if err != nil || p.ProfileID != "" || p.FileName != "gpt-oss-tpl.gguf" || p.Values["ctx"] != "8192" {
			t.Fatalf("save %+v %v", p, err)
		}
		if got := f.l.PresetPath("", path); got != filepath.Join(f.stateDir, StatePresetsDir, "gpt-oss-tpl.json") {
			t.Errorf("path %s", got)
		}
		if _, st := f.l.LoadPreset(ctx, path, ""); st != PresetApplies {
			t.Errorf("load %v", st)
		}
		if _, err := f.l.SavePreset(ctx, path, "", map[string]string{"repack_mode": "virtual"}); err == nil {
			t.Error("a virtual repack saved for a model outside the catalog")
		}
		if v, _, err := f.l.ParseModelSetting("", path, "ctx", "64k"); err != nil || v != "65536" {
			t.Errorf("typed %q %v", v, err)
		}
		if _, err := f.l.SavePreset(ctx, filepath.Join(models, "moetest-unsupported.gguf"), "", map[string]string{"ctx": "8192"}); err == nil {
			t.Error("saved settings for a structure without a template")
		}
	})
}
