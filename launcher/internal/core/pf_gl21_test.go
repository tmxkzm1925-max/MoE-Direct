package core

// GL-21: the post-ready read-back of the engine's prefetch values (F-PF §8.5, §6.4), the part of
// GT(TestS8PF16) over a fake metrics line (PLAN-FIX4 A2-1). The header lines below are the
// engine's form: the key set of a real header (X-VQD checks :84, :106) with «type» header first.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const (
	// the X-VQD v23 header (checks :106) with the common fields the engine writes first
	hdrUnset = `{"type": "header", "invocation": "", "session": "", "leg": "", "pid": 13876, "seq": 0, "ts": "2026-10-12T08:40:00.000Z", "qd_effective": 3, "qd_source": "legacy_alias", "io_qd_total": {"source": "legacy_alias", "value": 3}, "io_qd_demand_reserve": {"source": "default", "value": 4}, "io_qd_prefetch_child_max": {"source": "default", "value": 4}, "io_part_parallelism": {"source": "default", "value": 3}, "prefetch_on": false, "prefetch_k": 0, "prefetch_n": 0, "prefetch_off_reason": "unset", "io_path": "persistent", "mode": "virtual"}`
	// S8-PF-11's expected header: qwen35-35b, QD 2, K 8, N 1
	hdrOn = `{"type": "header", "pid": 1, "seq": 0, "qd_effective": 2, "prefetch_on": true, "prefetch_k": 8, "prefetch_n": 1, "prefetch_off_reason": "", "io_path": "persistent", "mode": "bin", "moe_layer_stride_bytes": [1, 2, 3]}`
	// S8-PF-12's: the seal turned prefetch off (PB:2362-2367)
	hdrAdapter = `{"type": "header", "pid": 1, "seq": 0, "prefetch_on": false, "prefetch_k": 0, "prefetch_n": 0, "prefetch_off_reason": "adapter_none"}`
	// the X-VQD v14 header (checks :84): the pair reached the engine directly
	hdrRange = `{"type": "header", "pid": 1, "seq": 0, "qd_effective": 3, "prefetch_on": false, "prefetch_k": 0, "prefetch_n": 0, "prefetch_off_reason": "invalid_range", "mode": "virtual"}`
)

func writeMetrics(t *testing.T, lines ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "metrics_20261012T120000_1.jsonl")
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func pf16ReadBack(t *testing.T) {
	k8, n1 := int64(8), int64(1)
	onCand := &PrefetchEcho{EffectivePrefetch: "on", RequestedK: &k8, RequestedN: &n1, PrefetchProvenance: ProvEnvOverride}
	offCand := &PrefetchEcho{EffectivePrefetch: "off(reason=qd_below_prefetch_depth)", OffReason: "qd_below_prefetch_depth"}
	tick := `{"type": "tick", "seq": 1}`
	for _, c := range []struct {
		name, header string
		cand         *PrefetchEcho
		want         EnginePrefetch
		line         string
	}{
		{"on (S8-PF-11)", hdrOn, onCand, EnginePrefetch{Read: true, On: true, K: 8, N: 1},
			"prefetch after ready: engine prefetch_on=true prefetch_k=8 prefetch_n=1 prefetch_off_reason= · candidate on requested_k=8 requested_n=1"},
		{"adapter_none (S8-PF-12)", hdrAdapter, onCand, EnginePrefetch{Read: true, OffReason: "adapter_none"},
			"prefetch after ready: engine prefetch_on=false prefetch_k=0 prefetch_n=0 prefetch_off_reason=adapter_none · candidate on requested_k=8 requested_n=1"},
		{"unset below QD N+1 (§6.4)", hdrUnset, offCand, EnginePrefetch{Read: true, OffReason: "unset"},
			"prefetch after ready: engine prefetch_on=false prefetch_k=0 prefetch_n=0 prefetch_off_reason=unset · candidate off(reason=qd_below_prefetch_depth)"},
		{"invalid_range (§6.3)", hdrRange, nil, EnginePrefetch{Read: true, OffReason: "invalid_range"},
			"prefetch after ready: engine prefetch_on=false prefetch_k=0 prefetch_n=0 prefetch_off_reason=invalid_range · candidate (none)"},
	} {
		got := ReadEnginePrefetch(writeMetrics(t, c.header, tick))
		if got != c.want {
			t.Errorf("%s: %+v", c.name, got)
		}
		if l := PostReadyLine(got, c.cand); l != c.line {
			t.Errorf("%s: line\n %s\nwant\n %s", c.name, l, c.line)
		}
		// header fields against the launcher line, exactly (S8-PF-11's comparison)
		fields := map[string]string{}
		for _, f := range strings.Fields(PostReadyLine(got, c.cand)) {
			if i := strings.IndexByte(f, '='); i > 0 && strings.HasPrefix(f, "prefetch_") {
				if _, seen := fields[f[:i]]; !seen {
					fields[f[:i]] = f[i+1:]
				}
			}
		}
		var hdr map[string]any
		jsonUnmarshalString(t, c.header, &hdr)
		for _, key := range []string{"prefetch_on", "prefetch_k", "prefetch_n", "prefetch_off_reason"} {
			if fields[key] != jsonScalar(hdr[key]) {
				t.Errorf("%s: %s line %q header %q", c.name, key, fields[key], jsonScalar(hdr[key]))
			}
		}
	}
	// Not read: never a guess, the reason instead.
	dir := t.TempDir()
	for name, c := range map[string]struct {
		path string
		why  string
	}{
		"no path":          {"", "no metrics path"},
		"no file":          {filepath.Join(dir, "none.jsonl"), "metrics file not readable"},
		"empty":            {writeMetrics(t, ""), "no header line"},
		"not the header":   {writeMetrics(t, `{"type": "tick", "seq": 1}`, hdrOn), "not the header"},
		"not JSON":         {writeMetrics(t, `{"type": "header", "prefetch_on": tru`), "not JSON"},
		"a key missing":    {writeMetrics(t, `{"type": "header", "prefetch_on": true, "prefetch_k": 8, "prefetch_n": 1}`), "without prefetch_off_reason"},
		"a key's type off": {writeMetrics(t, `{"type": "header", "prefetch_on": "true", "prefetch_k": 8, "prefetch_n": 1, "prefetch_off_reason": ""}`), "without prefetch_on"},
	} {
		got := ReadEnginePrefetch(c.path)
		if got.Read || !strings.Contains(got.Why, c.why) || got.On || got.K != 0 {
			t.Errorf("%s: %+v", name, got)
		}
		if l := PostReadyLine(got, onCand); !strings.Contains(l, "engine not read (") {
			t.Errorf("%s: %s", name, l)
		}
	}
}

func jsonUnmarshalString(t *testing.T, s string, v any) {
	t.Helper()
	if err := json.Unmarshal([]byte(s), v); err != nil {
		t.Fatal(err)
	}
}

// jsonScalar prints a header value as the launcher line prints it.
func jsonScalar(v any) string {
	switch x := v.(type) {
	case bool:
		return strconv.FormatBool(x)
	case float64:
		return strconv.FormatInt(int64(x), 10)
	case string:
		return x
	}
	return "?"
}
