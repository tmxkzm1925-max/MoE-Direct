package core

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Conflict 16: the gates per mode against fixture folders, every item, no cache.

func readObj(t *testing.T, path string) map[string]any {
	t.Helper()
	v, err := readJSONFileStrict(path)
	if err != nil {
		t.Fatal(err)
	}
	o, _ := asObj(v)
	return o
}

// rebind rewrites the report's manifest_sha256 to the manifest's real bytes (so a manifest
// mutation reaches the item under test, not gate 7 / vgate 6).
func rebind(t *testing.T, dir, report string) {
	t.Helper()
	ms, _ := fileSha256(filepath.Join(dir, "manifest.json"))
	o := readObj(t, filepath.Join(dir, report))
	o["manifest_sha256"] = ms
	bs, _ := json.Marshal(o) // one record: the verify report is read by its last non-empty line
	os.WriteFile(filepath.Join(dir, report), append(bs, '\n'), 0o644)
}

func TestVerifyGatePackedItems(t *testing.T) {
	const pid = "p1"
	sha := strings.Repeat("ab", 32)
	cases := []struct {
		name string
		item int
		mut  func(t *testing.T, dir string)
		want string
	}{
		{"pass", 0, func(*testing.T, string) {}, ""},
		{"partial marker", 1, func(t *testing.T, d string) { os.WriteFile(filepath.Join(d, "experts.bin.partial"), nil, 0o644) }, "gate 1: experts.bin.partial present"},
		{"no report", 2, func(t *testing.T, d string) { os.Remove(filepath.Join(d, "verify_report.json")) }, "gate 2: verify_report.json missing"},
		{"bad last record", 2, func(t *testing.T, d string) {
			os.WriteFile(filepath.Join(d, "verify_report.json"), []byte("{\"pass\":true}\n{oops\n"), 0o644)
		}, "gate 2: last record strict parse failed"},
		{"pass false", 3, func(t *testing.T, d string) { setReport(t, d, "pass", false) }, "gate 3: pass is false"},
		{"pass string", 3, func(t *testing.T, d string) { setReport(t, d, "pass", "true") }, "gate 3: pass is not a JSON boolean"},
		{"counts", 4, func(t *testing.T, d string) { setReport(t, d, "pairs_pass", 7) }, "gate 4: count equality broken pairs_pass=7"},
		{"count float", 4, func(t *testing.T, d string) { setReport(t, d, "expected_pairs", 8.5) }, "gate 4: expected_pairs is not a non-negative integer"},
		{"problems", 5, func(t *testing.T, d string) { setReport(t, d, "failures", []any{"x"}) }, "gate 5: failures is not empty"},
		{"problems missing", 5, func(t *testing.T, d string) { setReport(t, d, "padding_failures", nil) }, "gate 5: padding_failures is not a JSON array"},
		{"lock profile", 6, func(t *testing.T, d string) {
			setReport(t, d, "reference_lock", map[string]any{"profile_id": "other", "expect_sha256": sha})
		}, "gate 6: reference_lock.profile_id three-way mismatch (manifest=p1 verify_report=other selected=p1)"},
		{"lock sha", 6, func(t *testing.T, d string) {
			o := readObj(t, filepath.Join(d, "manifest.json"))
			o["reference_lock"] = map[string]any{"profile_id": pid, "expect_sha256": strings.Repeat("cd", 32)}
			writeJSON(t, filepath.Join(d, "manifest.json"), o)
			rebind(t, d, "verify_report.json")
		}, "gate 6: reference_lock.expect_sha256 three-way mismatch"},
		{"manifest changed", 7, func(t *testing.T, d string) {
			os.WriteFile(filepath.Join(d, "manifest.json"), []byte(`{"schema_version":"2.0","reference_lock":{"profile_id":"p1","expect_sha256":"`+sha+`"}}`), 0o644)
		}, "gate 7: manifest_sha256 != real manifest.json bytes"},
	}
	for _, c := range cases {
		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, dir, pid, sha)
		c.mut(t, dir)
		g := CheckFormat(RepackPacked, dir, pid, sha)
		if c.item == 0 {
			if g.State != GatePass || g.Passed != 7 || g.Items != 7 || g.Count != 8 || !strings.HasPrefix(g.Line(), "verify gate PASS (7/7) - pairs 8, manifest ") {
				t.Errorf("%s: %+v %s", c.name, g, g.Line())
			}
			continue
		}
		if c.name == "partial marker" || c.name == "no report" {
			// the disposition comes before the gate: an interrupted setup or a missing report
			// needs the repack (PS:14046-14057), the gate never sees them
			if g.State != GateNeedsRepack || (c.name == "partial marker") != g.Partial {
				t.Errorf("%s: %+v", c.name, g)
			}
			r, s := verifyGate(dir, pid, sha) // the gate itself, as Assert-VerifyGate
			if s == nil || s.item != c.item || !strings.Contains(s.reason, c.want) || r.State == GatePass {
				t.Errorf("%s: gate %+v", c.name, s)
			}
			continue
		}
		if g.State != GateFail || g.FailedItem != c.item || g.Passed != c.item-1 || !strings.Contains(g.Reason, c.want) {
			t.Errorf("%s: item %d %q (want %d %q)", c.name, g.FailedItem, g.Reason, c.item, c.want)
		}
	}
}

func setReport(t *testing.T, dir, key string, v any) {
	t.Helper()
	p := filepath.Join(dir, "verify_report.json")
	o := readObj(t, p)
	o[key] = v
	b, _ := json.Marshal(o)
	os.WriteFile(p, append(b, '\n'), 0o644)
}

func TestVirtualPlanGateItems(t *testing.T) {
	const pid = "p1"
	sha := strings.Repeat("ab", 32)
	setPlan := func(t *testing.T, d, k string, v any) {
		o := readObj(t, filepath.Join(d, "plan_report.json"))
		if v == nil {
			delete(o, k)
		} else {
			o[k] = v
		}
		writeJSON(t, filepath.Join(d, "plan_report.json"), o)
	}
	cases := []struct {
		name string
		item int
		mut  func(t *testing.T, dir string)
		want string
	}{
		{"pass", 0, func(*testing.T, string) {}, ""},
		{"bin here", 1, func(t *testing.T, d string) { os.WriteFile(filepath.Join(d, "experts.bin"), nil, 0o644) }, "vgate 1: experts.bin present in a virtual output directory"},
		{"report unreadable", 2, func(t *testing.T, d string) { os.WriteFile(filepath.Join(d, "plan_report.json"), []byte("[1"), 0o644) }, "vgate 2: plan_report.json - json parse failed"},
		{"report array", 2, func(t *testing.T, d string) { os.WriteFile(filepath.Join(d, "plan_report.json"), []byte("[1]"), 0o644) }, "vgate 2: plan_report.json root is not a JSON object"},
		{"mode", 3, func(t *testing.T, d string) { setPlan(t, d, "mode", "bin") }, "vgate 3: plan_report.mode is 'bin', not 'virtual'"},
		{"pass", 4, func(t *testing.T, d string) { setPlan(t, d, "pass", nil) }, "vgate 4: pass key missing"},
		{"problems", 5, func(t *testing.T, d string) { setPlan(t, d, "problems", []any{1}) }, "vgate 5: problems is not empty"},
		{"sha", 6, func(t *testing.T, d string) { setPlan(t, d, "manifest_sha256", strings.Repeat("0", 64)) }, "vgate 6: manifest_sha256 != real manifest.json bytes"},
		{"lock", 7, func(t *testing.T, d string) {
			setPlan(t, d, "reference_lock", map[string]any{"profile_id": pid, "expect_sha256": "zz"})
		}, "vgate 7: plan_report.reference_lock.expect_sha256 missing or not 64 hex"},
		{"cardinality", 8, func(t *testing.T, d string) { setPlan(t, d, "cardinality", map[string]any{"records": 15}) }, "vgate 8: cardinality.records(15) != manifest totals.n_records(16)"},
	}
	for _, c := range cases {
		dir := t.TempDir()
		writeVirtualArtifacts(t, dir, pid, sha)
		c.mut(t, dir)
		g := CheckFormat(RepackVirtual, dir, pid, sha)
		if c.item == 0 {
			if g.State != GatePass || g.Passed != 8 || g.Count != 16 || !strings.HasPrefix(g.Line(), "virtual plan gate PASS (8/8) - mode=virtual, records 16") {
				t.Errorf("%s: %+v", c.name, g)
			}
			continue
		}
		if g.State != GateFail || g.FailedItem != c.item || !strings.Contains(g.Reason, c.want) {
			t.Errorf("%s: item %d %q (want %d %q)", c.name, g.FailedItem, g.Reason, c.item, c.want)
		}
	}
	// No manifest: a repack is needed; a leftover plan report marks an interrupted setup.
	dir := t.TempDir()
	if g := CheckFormat(RepackVirtual, dir, pid, sha); g.State != GateNeedsRepack || g.Partial {
		t.Errorf("empty virtual dir %+v", g)
	}
	os.WriteFile(filepath.Join(dir, "plan_report.json"), []byte("{}"), 0o644)
	if g := CheckFormat(RepackVirtual, dir, pid, sha); g.State != GateNeedsRepack || !g.Partial {
		t.Errorf("partial virtual dir %+v", g)
	}
	// A packed manifest in a virtual folder, and a virtual manifest in a packed one: mode mismatch.
	pk := t.TempDir()
	writePackedArtifacts(t, pk, pid, sha)
	if g := CheckFormat(RepackVirtual, pk, pid, sha); g.State != GateFail || !strings.Contains(g.Reason, "existing artifacts are bin") {
		t.Errorf("bin in virtual %+v", g)
	}
	vt := t.TempDir()
	writeVirtualArtifacts(t, vt, pid, sha)
	if g := CheckFormat(RepackPacked, vt, pid, sha); g.State != GateFail || !strings.Contains(g.Reason, "existing artifacts are virtual") {
		t.Errorf("virtual in packed %+v", g)
	}
}

func TestManifestMode(t *testing.T) {
	dir := t.TempDir()
	for _, c := range []struct {
		body, mode, reason string
	}{
		{`{"schema_version":"2.0"}`, ManifestModeBin, ""},
		{`{"schema_version":"3.0","mode":"virtual"}`, ManifestModeVirtual, ""},
		{`{"schema_version":"2.0","mode":"virtual"}`, ManifestModeUnrecognized, "schema_version 2.0 carries a mode key"},
		{`{"schema_version":"3.0"}`, ManifestModeUnrecognized, "schema_version 3.0 has no mode key"},
		{`{"schema_version":3}`, ManifestModeUnrecognized, "schema_version is not a JSON string"},
		{`{"schema_version":"4.0"}`, ManifestModeUnrecognized, "schema_version is '4.0'"},
	} {
		p := filepath.Join(dir, "m.json")
		os.WriteFile(p, []byte(c.body), 0o644)
		if m, r := ManifestMode(p); m != c.mode || r != c.reason {
			t.Errorf("%s: %s %q", c.body, m, r)
		}
	}
}

// The launcher's per-model call gives both modes, re-reads on every call (no cache), and a
// start runs the gate of its mode each time.
func TestFormatGatesBothModesNoCache(t *testing.T) {
	f := newFixture(t, false)
	ctx := context.Background()
	res, err := f.l.FormatGates(ctx, f.modelPath, "fixture-profile")
	if err != nil || len(res) != 2 || res[0].Mode != RepackPacked || res[0].Result.State != GatePass ||
		res[1].Mode != RepackVirtual || res[1].Result.State != GateNeedsRepack {
		t.Fatalf("both modes: %v %+v", err, res)
	}
	pre, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	if err != nil || pre.Gate == nil || pre.Gate.State != GatePass || pre.ManifestSha != pre.Gate.ManifestSha {
		t.Fatalf("preflight gate: %v %+v", err, pre.Gate)
	}
	// Break the report between two calls: the next call sees it (no cache across calls).
	setReport(t, f.outDir, "pass", false)
	res, _ = f.l.FormatGates(ctx, f.modelPath, "fixture-profile")
	if res[0].Result.State != GateFail || res[0].Result.FailedItem != 3 {
		t.Errorf("after break: %+v", res[0].Result)
	}
	_, err = f.l.Preflight(ctx, Request{ModelPath: f.modelPath})
	var ge *GateError
	if !errors.As(err, &ge) || ge.Gate.FailedItem != 3 || ExitCodeOf(err) != 4 || StatusOf(err) != StatusFailGateVerify {
		t.Errorf("start-time gate failure: %v", err)
	}
	// A virtual plan appears: the virtual mode passes too.
	writeVirtualArtifacts(t, filepath.Join(filepath.Dir(f.modelPath), "repack-virtual"), "fixture-profile", f.expectSha)
	res, _ = f.l.FormatGates(ctx, f.modelPath, "fixture-profile")
	if res[1].Result.State != GatePass || res[1].Result.Items != 8 {
		t.Errorf("virtual: %+v", res[1].Result)
	}
}
