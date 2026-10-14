package wire

import (
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// S8-PF-16 (screen part; the rows are core's TestS8PF16, the engine's values after ready
// GL-21's): the screen shows K's range 1..min(16, n_expert) and N's on range 1..QD-1 with the
// QD the screen holds, «QD-1» while the QD is auto, and the off state at QD 1.
func TestS8PF16(t *testing.T) {
	rows := func(qd string) []core.SettingRow {
		return []core.SettingRow{
			{Key: "qd", Default: qd, Editable: true, Min: 1, Max: 63, Bounded: true},
			{Key: "prefetch_k", Default: "auto", Editable: true, Min: 1, Max: 4, Bounded: true},
			{Key: "prefetch_n", Default: "auto", Editable: true, Min: 1, OnRange: true},
		}
	}
	for _, c := range []struct{ qd, k, n string }{
		{"8", "1 ~ 4", "1 ~ 7"},
		{"auto", "1 ~ 4", "1 ~ QD-1"},
		{"1", "1 ~ 4", "off at QD 1"},
	} {
		v := settingsFrom("m", rows(c.qd), nil, nil, "")
		by := map[string]ui.SettingRow{}
		for _, r := range v.Rows {
			by[r.Key] = r
		}
		if r := by["prefetch_k"]; r.Range != c.k || r.Label != "prefetch K" {
			t.Errorf("QD %s: K row %+v", c.qd, r)
		}
		if r := by["prefetch_n"]; r.Range != c.n || r.Note != c.n || r.Label != "prefetch N" {
			t.Errorf("QD %s: N row %+v", c.qd, r)
		}
	}
	// A saved QD is the screen's QD.
	v := settingsFrom("m", rows("auto"), nil, map[string]string{"qd": "5"}, "")
	for _, r := range v.Rows {
		if r.Key == "prefetch_n" && r.Range != "1 ~ 4" {
			t.Errorf("saved QD 5: %+v", r)
		}
	}
	t.Run("engine-values-after-ready", pf16Screen)
}

// GL-21 (F-PF §8.5, §6.4): the server screen's prefetch row shows the engine's values after ready
// and the candidate after them; the settings words follow the engine, not the candidate.
func pf16Screen(t *testing.T) {
	k, n, qd := int64(8), int64(4), int64(8)
	on := &core.PrefetchEcho{EffectivePrefetch: "on", RequestedK: &k, RequestedN: &n, RequestedQD: &qd, PrefetchProvenance: core.ProvCatalog}
	off := &core.PrefetchEcho{EffectivePrefetch: "off(reason=qd_below_prefetch_depth)", OffReason: "qd_below_prefetch_depth", RequestedQD: &qd}
	for _, c := range []struct {
		name      string
		start     core.StartSettings
		eng       *core.EnginePrefetch
		words     string
		settingPF string
	}{
		{"before ready", core.StartSettings{Prefetch: "catalog", PrefetchOn: true, PrefetchEcho: on}, nil,
			"candidate on|K 8|N 4|catalog-validated", "prefetch on"},
		{"engine on", core.StartSettings{Prefetch: "catalog", PrefetchOn: true, PrefetchEcho: on}, &core.EnginePrefetch{Read: true, On: true, K: 8, N: 4},
			"engine on|K 8|N 4|candidate on|K 8|N 4|catalog-validated", "prefetch on"},
		{"engine turned it off", core.StartSettings{Prefetch: "catalog", PrefetchOn: true, PrefetchEcho: on}, &core.EnginePrefetch{Read: true, OffReason: "adapter_none"},
			"engine off|adapter_none|candidate on|K 8|N 4|catalog-validated", "prefetch off"},
		{"unset below QD N+1", core.StartSettings{Prefetch: "catalog", PrefetchEcho: off}, &core.EnginePrefetch{Read: true, OffReason: "unset"},
			"engine off|unset|candidate off|qd_below_prefetch_depth", "prefetch off"},
		{"not read", core.StartSettings{Prefetch: "catalog", PrefetchOn: true, PrefetchEcho: on}, &core.EnginePrefetch{Why: "no metrics path"},
			"engine values not read|candidate on|K 8|N 4|catalog-validated", "prefetch on"},
	} {
		s := statusFrom(core.ServeStatus{Start: c.start, Engine: c.eng}, core.MemStatus{}, false, core.VRAMInfo{}, serverGPU{}, nil, time.Time{}, false)
		if got := strings.Join(s.Prefetch, "|"); got != c.words {
			t.Errorf("%s: %s", c.name, got)
		}
		if !strings.Contains(strings.Join(s.SettingsWords, "|"), c.settingPF) {
			t.Errorf("%s: settings words %v", c.name, s.SettingsWords)
		}
	}
}
