package core

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func wantStatus(t *testing.T, err error, st Status, substr string) {
	t.Helper()
	le, ok := AsLaunchError(err)
	if !ok || le == nil {
		t.Fatalf("want %s error, got %v", st, err)
	}
	if le.Status != st {
		t.Fatalf("want status %s, got %s (%s)", st, le.Status, le.Reason)
	}
	if substr != "" && !strings.Contains(le.Reason, substr) {
		t.Fatalf("reason %q does not contain %q", le.Reason, substr)
	}
}

func TestExitCodeTable(t *testing.T) {
	want := map[Status]int{
		StatusOK: 0, StatusOKSmoke: 0, StatusCancelledUser: 2,
		StatusFailModelPath: 3, StatusFailResource: 3, StatusFailInstanceLock: 3, StatusFailPartialCleanup: 3,
		StatusFailRepack: 3, StatusFailCustomArgs: 3,
		StatusFailGateBundle: 4, StatusFailGateCatalog: 4, StatusFailGateVerify: 4, StatusFailGateEngineSeal: 4,
		StatusFailServerStart: 5, StatusFailRuntimeExit: 5, StatusFailTeardown: 5, StatusFailSmoke: 6,
	}
	if len(want) != len(statusExit) {
		t.Fatalf("table has %d statuses, PS STATUS_EXIT has 17", len(statusExit))
	}
	for s, c := range want {
		if got := ExitCode(s); got != c {
			t.Errorf("ExitCode(%s) = %d, want %d", s, got, c)
		}
	}
	if ExitCode("no_such_status") != 5 {
		t.Error("an unknown status must map to 5 (Get-StatusExitCode)")
	}
}

func TestNotPortedErrors(t *testing.T) {
	if len(NotPortedFamilies) != 7 || len(PortedFamilies) != 8 { // GL-18: warm start left the list (GL-15); GL-10: the probe and sweep, the template setup
		t.Fatalf("families = %d", len(NotPortedFamilies))
	}
	for _, f := range NotPortedFamilies {
		err := NotPorted(f)
		if !errors.Is(err, ErrNotPorted) {
			t.Errorf("%s: not matched by ErrNotPorted", f)
		}
		if !strings.Contains(err.Error(), "not ported (after the user's decisions)") {
			t.Errorf("%s: text %q", f, err.Error())
		}
		if _, ok := AsLaunchError(err); ok {
			t.Errorf("%s: a not-ported error is not a launcher verdict", f)
		}
	}
}

func TestReadCatalogReal(t *testing.T) {
	needStaging(t)
	cat, err := ReadCatalog(realCatalog, stagingRoot)
	if err != nil {
		t.Fatal(err)
	}
	if cat.SourceTag == "" || len(cat.Profiles) != 6 {
		t.Fatalf("source_tag %q, %d profiles", cat.SourceTag, len(cat.Profiles))
	}
	ids := []string{"qwen35-122b-nonextn", "gpt-oss-120b", "qwen35-35b", "qwen35-397b", "kimi-k2.6-ram-447gb", "deepseek-v4-flash"}
	for i, id := range ids {
		if cat.Profiles[i].ID != id {
			t.Errorf("profile %d = %s, want %s", i, cat.Profiles[i].ID, id)
		}
	}
	q := cat.ProfileByID("qwen35-122b-nonextn")
	if len(q.SourceShardsSha256) != 2 || q.Identify.Arch != "qwen35moe" || q.Identify.NLayer != 49 || q.MinBudgetMB != 8192 {
		t.Errorf("qwen35-122b-nonextn read wrong: %+v", q.Identify)
	}
	a := catalogPrefetchAxes(q)
	if a.semanticInvalid || a.activation != "catalog-fixed" || a.k != 8 || a.n != 4 {
		t.Errorf("qwen prefetch axes %+v", a)
	}
	if cat.Runtime.ServerExe != "llama-server.exe" || cat.Runtime.ExpectsDir != "expects" || cat.Runtime.WebUI {
		t.Errorf("runtime %+v", cat.Runtime)
	}
	if b := q.Bounds["qd"]; b.Min != 1 || b.Max != 63 {
		t.Errorf("qd bounds %+v", b)
	}
}

func TestReadCatalogRefusals(t *testing.T) {
	f := newFixture(t, false)
	if _, err := ReadCatalog(f.catalogPath, f.root); err != nil {
		t.Fatalf("fixture catalog: %v", err)
	}
	b, _ := os.ReadFile(f.catalogPath)
	mutate := func(name string, edit func(c map[string]any)) string {
		var c map[string]any
		json.Unmarshal(b, &c)
		edit(c)
		p := filepath.Join(f.dir, name)
		writeJSON(t, p, c)
		return p
	}
	prof := func(c map[string]any) map[string]any { return c["profiles"].([]any)[0].(map[string]any) }
	cases := []struct {
		name, substr string
		edit         func(c map[string]any)
	}{
		{"unknown_top.json", "unknown key 'extra' in models.json", func(c map[string]any) { c["extra"] = 1 }},
		{"schema.json", "catalog_schema_version is not an exact match", func(c map[string]any) { c["catalog_schema_version"] = 3 }},
		{"unknown_profile_key.json", "unknown key 'tuning'", func(c map[string]any) { prof(c)["tuning"] = true }},
		{"expect_sha.json", "re-hash != catalog expect_sha256", func(c map[string]any) {
			prof(c)["expect_sha256"] = strings.Repeat("0", 64)
		}},
		{"pin_arity.json", "source_shards_sha256 has 2 digest(s) but the expect describes 1", func(c map[string]any) {
			prof(c)["source_shards_sha256"] = []string{strings.Repeat("a", 64), strings.Repeat("b", 64)}
		}},
		{"qd_bounds.json", "allowlist_bounds.qd outside engine range", func(c map[string]any) {
			prof(c)["allowlist_bounds"].(map[string]any)["qd"] = map[string]any{"min": 1, "max": 64}
		}},
		{"budget_min.json", "budget_mb.min below min_budget_mb", func(c map[string]any) { prof(c)["min_budget_mb"] = 8192 }},
		{"identify.json", "required key 'arch' missing", func(c map[string]any) { delete(prof(c), "identify"); prof(c)["identify"] = map[string]any{} }},
	}
	for _, tc := range cases {
		p := mutate(tc.name, tc.edit)
		_, err := ReadCatalog(p, f.root)
		wantStatus(t, err, StatusFailGateCatalog, tc.substr)
	}
	_, err := ReadCatalog(filepath.Join(f.dir, "absent.json"), f.root)
	wantStatus(t, err, StatusFailGateCatalog, "catalog missing")
}
