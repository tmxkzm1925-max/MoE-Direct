package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// GL-10 (plan cell GL-10; F-LS 4.6, 7.7, 7.11 and W7 duty 12): the setup flow. No real repacker,
// server or disk timing runs: the repacker is a fake executor that logs its calls, the speed
// check's reads are fakeMeter's mock timings (meter_gl10_test.go), and the PowerShell side is the
// record of testdata/la38 (record_ps.ps1 over Start-MoeDirect.ps1's own functions).

// la38Record is testdata/la38/ps_recorded.json.
type la38Record struct {
	ManifestSha string `json:"manifest_sha256"`
	Strides     struct {
		OK      bool    `json:"ok"`
		Strides []int64 `json:"strides"`
		Align   int64   `json:"align_bytes"`
	} `json:"strides"`
	Blocks []struct {
		Strides []int64 `json:"strides"`
		Align   int64   `json:"align"`
		OK      bool    `json:"ok"`
		Block   *int64  `json:"block_bytes"`
		Median  int64   `json:"median"`
		Reason  *string `json:"reason"`
	} `json:"block_bytes"`
	Offsets struct {
		OK     bool    `json:"ok"`
		Count  int     `json:"count"`
		First  []int64 `json:"first16"`
		Sha256 string  `json:"sha256"`
	} `json:"offsets"`
	ProbeBlock int64  `json:"probe_block_bytes"`
	TargetKey  string `json:"target_key"`
	Scenarios  []struct {
		Name      string          `json:"name"`
		Arm       string          `json:"arm"`
		N         *int64          `json:"n"`
		OK        bool            `json:"ok"`
		Order     []int           `json:"order"`
		Rounds    int             `json:"rounds"`
		Confirm   bool            `json:"confirm_round"`
		Block     int64           `json:"block_bytes"`
		Align     int64           `json:"align_bytes"`
		Count     int64           `json:"block_count"`
		Median    int64           `json:"stride_median"`
		Points    []psPoint       `json:"points"`
		Selection psSelection     `json:"selection"`
		Format    string          `json:"format"`
		Lines     psStrings       `json:"lines"`
		Raw       json.RawMessage `json:"-"`
	} `json:"scenarios"`
	Resolve struct {
		OK            bool      `json:"ok"`
		QD            int       `json:"qd"`
		QDSource      string    `json:"qd_source"`
		Reason        string    `json:"reason"`
		Detail        string    `json:"detail"`
		PersistFailed bool      `json:"persist_failed"`
		Order         []int     `json:"order"`
		Lines         psStrings `json:"lines"`
		StateVersion  int       `json:"state_version"`
		Key           string    `json:"key"`
		Entry         struct {
			Algorithm       string `json:"probe_algorithm"`
			Method          string `json:"measurement_method"`
			SourceTag       string `json:"source_tag"`
			ProfileID       string `json:"profile_id"`
			ExpectDigest    string `json:"expectation_digest"`
			ManifestSha     string `json:"manifest_sha256"`
			VolumeGUID      string `json:"output_volume_guid"`
			SelectedQD      int    `json:"selected_qd"`
			SelectedReason  string `json:"selected_reason"`
			BlockBytes      int64  `json:"block_bytes"`
			WindowMs        int    `json:"window_duration_ms"`
			WindowsPerPoint int    `json:"windows_per_point"`
			RepackCheckedAt string `json:"repack_checked_at"`
			PointCount      int    `json:"point_count"`
		} `json:"entry"`
	} `json:"resolve"`
	ResolveHit struct {
		OK          bool      `json:"ok"`
		QD          int       `json:"qd"`
		Reason      string    `json:"reason"`
		FromBinding bool      `json:"from_binding"`
		Windows     int       `json:"windows"`
		Lines       psStrings `json:"lines"`
	} `json:"resolve_hit"`
	Probe struct {
		OK       bool      `json:"ok"`
		QD       int       `json:"qd"`
		MiBps    float64   `json:"mibps"`
		QDSource string    `json:"qd_source"`
		Lines    psStrings `json:"lines"`
		Record   struct {
			StateVersion int     `json:"state_version"`
			SourceTag    string  `json:"source_tag"`
			ProfileID    string  `json:"profile_id"`
			ExpectDigest string  `json:"expect_digest"`
			VolumeID     string  `json:"output_volume_id"`
			ProbeOK      bool    `json:"probe_ok"`
			QD           int     `json:"qd"`
			MiBps        float64 `json:"mibps"`
		} `json:"record"`
	} `json:"probe"`
}

// psStrings reads ConvertTo-Json's array, or its bare string for a one-element array.
type psStrings []string

func (p *psStrings) UnmarshalJSON(b []byte) error {
	var one string
	if json.Unmarshal(b, &one) == nil {
		*p = []string{one}
		return nil
	}
	var many []string
	err := json.Unmarshal(b, &many)
	*p = many
	return err
}

type psPoint struct {
	QD      int      `json:"qd"`
	Status  string   `json:"status"`
	Bytes   int64    `json:"bytes"`
	Ticks   int64    `json:"elapsed_ticks"`
	Reads   int      `json:"reads"`
	Windows *int     `json:"windows"`
	MiBps   *float64 `json:"mibps"`
	Error   *string  `json:"error"`
}

type psSelection struct {
	OK     bool   `json:"ok"`
	QD     int    `json:"qd"`
	Reason string `json:"reason"`
	S90    []int  `json:"s90"`
	QBase  *int   `json:"q_base"`
}

func readLA38(t *testing.T) *la38Record {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "la38", "ps_recorded.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r la38Record
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

// la38Timings is testdata/la38/timings.json: per scenario, per QD, the windows in order.
type la38Timings struct {
	Scenarios []struct {
		Name   string               `json:"name"`
		Arm    string               `json:"arm"`
		N      *int64               `json:"n"`
		Timing map[string][][]int64 `json:"timing"`
	} `json:"scenarios"`
}

func readTimings(t *testing.T) *la38Timings {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "la38", "timings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r la38Timings
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

// timingOf gives fakeMeter the scenario's table: the round-th window of a QD, the last repeated.
func timingOf(tab map[string][][]int64) func(qd, round int) (int64, int64) {
	return func(qd, round int) (int64, int64) {
		ws := tab[fmt.Sprint(qd)]
		if len(ws) == 0 {
			return 0, 0
		}
		w := ws[min(round, len(ws)-1)]
		return w[0], w[1]
	}
}

// la38Out is the record's output folder: the committed manifest and a 64 MiB experts.bin.
func la38Out(t *testing.T) string {
	t.Helper()
	out := filepath.Join(t.TempDir(), "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join("testdata", "la38", "out", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "manifest.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(out, "experts.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(67108864); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return out
}

func la38Launcher(t *testing.T, m *fakeMeter) *Launcher {
	t.Helper()
	l, err := New(Options{CatalogPath: "unused.json", BundleRoot: t.TempDir(), StateDir: filepath.Join(t.TempDir(), "state")})
	if err != nil {
		t.Fatal(err)
	}
	l.os.sweep = m
	l.os.volumeGUID = func(string) (string, error) { return `\\?\volume{fixture}\`, nil }
	return l
}

func pointsEqual(t *testing.T, what string, got []SweepPoint, want []psPoint) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d points, PowerShell %d", what, len(got), len(want))
	}
	for i, g := range got {
		w := want[i]
		ge, we := "", ""
		if g.Error != nil {
			ge = *g.Error
		}
		if w.Error != nil {
			we = *w.Error
		}
		gm, wm := -1.0, -1.0
		if g.MiBps != nil {
			gm = *g.MiBps
		}
		if w.MiBps != nil {
			wm = *w.MiBps
		}
		gw, ww := -1, -1
		if g.Windows != nil {
			gw = *g.Windows
		}
		if w.Windows != nil {
			ww = *w.Windows
		}
		if g.QD != w.QD || g.Status != w.Status || g.Bytes != w.Bytes || g.Ticks != w.Ticks || g.Reads != w.Reads ||
			gm != wm || ge != we || gw != ww {
			t.Errorf("%s point %d: Go %+v (mibps %v err %q windows %d), PowerShell %+v (mibps %v err %q windows %d)",
				what, i, g, gm, ge, gw, w, wm, we, ww)
		}
	}
}

// TestS8LA38 (S8-LA-38; LFS Q5.15; PFP:360-368): the speed check runs the startup probe and the
// QD sweep in the recorded order, and with a manual pair the sweep uses the effective N
// [A-SWEEP]; probe sequence and chosen QD against the recorded result.
func TestS8LA38(t *testing.T) {
	rec := readLA38(t)
	tim := readTimings(t)

	t.Run("pure parts against the record", func(t *testing.T) {
		out := la38Out(t)
		ms, _ := fileSha256(filepath.Join(out, "manifest.json"))
		if ms != rec.ManifestSha {
			t.Fatalf("manifest sha %s, PowerShell %s", ms, rec.ManifestSha)
		}
		st, align, why := manifestStrides(filepath.Join(out, "manifest.json"))
		if why != "" || !reflect.DeepEqual(st, rec.Strides.Strides) || align != rec.Strides.Align {
			t.Errorf("strides %v %d %q, PowerShell %+v", st, align, why, rec.Strides)
		}
		for _, c := range rec.Blocks {
			b := sweepBlockBytes(c.Strides, c.Align)
			ok := b.OK == c.OK && b.Median == c.Median
			if c.Block != nil {
				ok = ok && b.BlockBytes == *c.Block
			}
			if c.Reason != nil {
				ok = ok && b.Reason == *c.Reason
			}
			if !ok {
				t.Errorf("block %v/%d: Go %+v, PowerShell %+v", c.Strides, c.Align, b, c)
			}
		}
		offs, why := sweepOffsets(rec.ManifestSha, 32, 2097152, sweepOffsetCount)
		var parts []string
		for _, o := range offs {
			parts = append(parts, fmt.Sprint(o))
		}
		if why != "" || len(offs) != rec.Offsets.Count || !reflect.DeepEqual(offs[:16], rec.Offsets.First) ||
			sha256Hex([]byte(strings.Join(parts, ","))) != rec.Offsets.Sha256 {
			t.Errorf("offsets %v (%q), PowerShell %+v", offs[:16], why, rec.Offsets)
		}
		if b := probeBlockBytes(out); b != rec.ProbeBlock {
			t.Errorf("probe block %d, PowerShell %d", b, rec.ProbeBlock)
		}
		if k := sweepTargetKey("fixture-tag", "fixture-profile", strings.Repeat("AB", 32), rec.ManifestSha, `\\?\Volume{Fixture}\`); k != rec.TargetKey {
			t.Errorf("target key %s, PowerShell %s", k, rec.TargetKey)
		}
	})

	t.Run("sweep per scenario against the record", func(t *testing.T) {
		if len(rec.Scenarios) != len(tim.Scenarios) || len(rec.Scenarios) < 7 {
			t.Fatalf("%d recorded scenarios, %d timing scenarios", len(rec.Scenarios), len(tim.Scenarios))
		}
		for i, ps := range rec.Scenarios {
			m := &fakeMeter{timing: timingOf(tim.Scenarios[i].Timing), phys: 4096}
			l := la38Launcher(t, m)
			out := la38Out(t)
			run := l.qdSweep(context.Background(), out, rec.ManifestSha, ps.Arm, ps.N)
			if run.OK != ps.OK || !reflect.DeepEqual(run.Order, ps.Order) || run.Rounds != ps.Rounds || run.Confirm != ps.Confirm ||
				run.BlockBytes != ps.Block || run.AlignBytes != ps.Align || run.BlockCount != ps.Count || run.Median != ps.Median {
				t.Errorf("%s: Go ok=%v order=%v rounds=%d confirm=%v block=%d align=%d count=%d median=%d; PowerShell %+v",
					ps.Name, run.OK, run.Order, run.Rounds, run.Confirm, run.BlockBytes, run.AlignBytes, run.BlockCount, run.Median, ps)
			}
			if got := m.sequence(); got != strings.Trim(strings.Join(strings.Fields(fmt.Sprint(ps.Order)), " "), "[]") {
				t.Errorf("%s: meter sequence %q, PowerShell order %v", ps.Name, got, ps.Order)
			}
			pointsEqual(t, ps.Name, run.Points, ps.Points)
			s := run.Sel
			qb := 0
			if ps.Selection.QBase != nil {
				qb = *ps.Selection.QBase
			}
			if s.OK != ps.Selection.OK || s.QD != ps.Selection.QD || s.Reason != ps.Selection.Reason || s.QBase != qb ||
				!reflect.DeepEqual(append([]int{}, s.S90...), append([]int{}, ps.Selection.S90...)) {
				t.Errorf("%s: selection %+v, PowerShell %+v", ps.Name, s, ps.Selection)
			}
			if f := formatSweepPoints(run.Points); f != ps.Format {
				t.Errorf("%s: format %q, PowerShell %q", ps.Name, f, ps.Format)
			}
			if !reflect.DeepEqual(run.Lines, []string(ps.Lines)) {
				t.Errorf("%s: lines %q, PowerShell %q", ps.Name, run.Lines, ps.Lines)
			}
			t.Logf("%s: order %v -> QD%d (%s), PowerShell QD%d (%s)", ps.Name, run.Order, s.QD, s.Reason, ps.Selection.QD, ps.Selection.Reason)
		}
	})

	t.Run("manual pair uses the effective N [A-SWEEP]", func(t *testing.T) {
		// The manual arm is the catalog-fixed arm's formula with the set N (F-PF §3.5): the
		// same windows, the same QD as PowerShell's catalog-fixed run with that N; the reason
		// word is the manual arm's own when the preference moved the QD.
		for i, ps := range rec.Scenarios {
			if ps.Arm != armCatalog {
				continue
			}
			m := &fakeMeter{timing: timingOf(tim.Scenarios[i].Timing), phys: 4096}
			run := la38Launcher(t, m).qdSweep(context.Background(), la38Out(t), rec.ManifestSha, armManual, ps.N)
			want := ps.Selection.Reason
			if want == "prefetch-preferred" {
				want = manualPreferred
			}
			if !reflect.DeepEqual(run.Order, ps.Order) || run.Sel.QD != ps.Selection.QD || run.Sel.Reason != want {
				t.Errorf("%s with the manual arm: order %v QD%d (%s), want the record's %v QD%d (%s)", ps.Name, run.Order,
					run.Sel.QD, run.Sel.Reason, ps.Order, ps.Selection.QD, want)
			}
			t.Logf("manual N=%d: QD%d (%s); PowerShell catalog-fixed N=%d: QD%d (%s)", *ps.N, run.Sel.QD, run.Sel.Reason,
				*ps.N, ps.Selection.QD, ps.Selection.Reason)
		}
		// Without the arm the same windows keep the knee: the effective N is what moves it.
		m := &fakeMeter{timing: timingOf(tim.Scenarios[2].Timing), phys: 4096}
		if run := la38Launcher(t, m).qdSweep(context.Background(), la38Out(t), rec.ManifestSha, armNone, nil); run.Sel.QD != 4 || run.Sel.Reason != "io-knee" {
			t.Errorf("no arm: QD%d (%s)", run.Sel.QD, run.Sel.Reason)
		}
	})

	t.Run("Resolve-QdSweep: miss, sweep, binding, hit", func(t *testing.T) {
		m := &fakeMeter{timing: timingOf(tim.Scenarios[0].Timing), phys: 4096}
		l := la38Launcher(t, m)
		out := la38Out(t)
		prof := &Profile{ID: "fixture-profile", Raw: jobj{"prefetch": map[string]any{"k": json.Number("8"), "n": json.Number("4")}}}
		checked := "2026-10-07T01:51:39.264165+00:00"
		ab := strings.Repeat("ab", 32)
		// a preflight of the screen does not measure (no window)
		pre := l.resolveSweep(context.Background(), out, "fixture-tag", "fixture-profile", ab, rec.ManifestSha, checked, prof, armNone, nil, false)
		if pre.OK || pre.QD != QDDegraded || m.sequence() != "" || !strings.Contains(pre.Detail, "runs at the start") {
			t.Fatalf("preflight swept: %+v %q", pre, m.sequence())
		}
		sw := l.resolveSweep(context.Background(), out, "fixture-tag", "fixture-profile", ab, rec.ManifestSha, checked, prof, armNone, nil, true)
		r := rec.Resolve
		if sw.OK != r.OK || sw.QD != r.QD || sw.QDSource != r.QDSource || sw.Reason != r.Reason || sw.Detail != r.Detail ||
			sw.PersistFailed != r.PersistFailed || !reflect.DeepEqual(sw.Order, r.Order) ||
			!reflect.DeepEqual(strings.Split(sw.Line, "\n"), []string(r.Lines)) {
			t.Errorf("resolve: Go %+v\nPowerShell %+v", sw, r)
		}
		b, err := os.ReadFile(filepath.Join(l.opt.StateDir, sweepStateFile))
		if err != nil {
			t.Fatal(err)
		}
		var doc struct {
			Ver     int                        `json:"state_version"`
			Entries map[string]json.RawMessage `json:"entries"`
		}
		if err := json.Unmarshal(b, &doc); err != nil || doc.Ver != r.StateVersion || len(doc.Entries) != 1 || doc.Entries[r.Key] == nil {
			t.Fatalf("state %v %s", err, b)
		}
		var e struct {
			sweepEntry
			Points []json.RawMessage `json:"points"`
		}
		json.Unmarshal(doc.Entries[r.Key], &e)
		w := r.Entry
		if e.Algorithm != w.Algorithm || e.Method != w.Method || e.SourceTag != w.SourceTag || e.ProfileID != w.ProfileID ||
			e.ExpectDigest != w.ExpectDigest || e.ManifestSha != w.ManifestSha || e.VolumeGUID != w.VolumeGUID ||
			e.SelectedQD != w.SelectedQD || e.SelectedReason != w.SelectedReason || e.BlockBytes != w.BlockBytes ||
			e.WindowMs != w.WindowMs || e.WindowsPerPoint != w.WindowsPerPoint || e.RepackCheckedAt != w.RepackCheckedAt ||
			len(e.Points) != w.PointCount {
			t.Errorf("entry %+v\nPowerShell %+v", e.sweepEntry, w)
		}
		// the same key again, on the catalog arm: a binding hit, no window measured
		n0 := len(m.calls)
		hit := l.resolveSweep(context.Background(), out, "fixture-tag", "fixture-profile", ab, rec.ManifestSha, checked, prof, armCatalog, nil, true)
		h := rec.ResolveHit
		if hit.OK != h.OK || hit.QD != h.QD || hit.Reason != h.Reason || hit.FromBinding != h.FromBinding || len(m.calls)-n0 != h.Windows ||
			len(h.Lines) != 1 || !strings.HasPrefix(hit.Line, "[sweep] reusing the bound sweep record (QD8, stored binding from 20") ||
			!strings.HasPrefix(h.Lines[0], "[sweep] reusing the bound sweep record (QD8, stored binding from 20") {
			t.Errorf("hit %+v, PowerShell %+v", hit, h)
		}
	})

	t.Run("startup probe and its record", func(t *testing.T) {
		m := &fakeMeter{phys: 4096}
		l := la38Launcher(t, m)
		out := la38Out(t)
		p := l.startupProbe(out)
		pr := rec.Probe
		lines := strings.Split(p.Line, "\n")
		if p.OK != pr.OK || p.QD != pr.QD || p.MiBps != pr.MiBps || p.QDSource != pr.QDSource || len(pr.Lines) != 3 ||
			!reflect.DeepEqual(lines, []string(pr.Lines[1:])) || m.sequence() != "probe" {
			t.Errorf("probe %+v %q, PowerShell %+v", p, m.sequence(), pr)
		}
		if first := fmt.Sprintf("[probe] measuring SSD random read (%d samples x %d B unbuffered)...", probeSamples, probeBlockBytes(out)); first != pr.Lines[0] {
			t.Errorf("start line %q, PowerShell %q", first, pr.Lines[0])
		}
		l.writeProbeBinding("fixture-tag", "fixture-profile", strings.Repeat("ab", 32), out, p)
		var got probeRecord
		b, _ := os.ReadFile(filepath.Join(l.opt.StateDir, probeStateFile))
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatal(err)
		}
		w := pr.Record
		if got.StateVersion != w.StateVersion || got.SourceTag != w.SourceTag || got.ProfileID != w.ProfileID ||
			got.ExpectDigest != w.ExpectDigest || got.VolumeID != w.VolumeID || got.ProbeOK != w.ProbeOK || got.QD != w.QD ||
			got.MiBps == nil || *got.MiBps != w.MiBps {
			t.Errorf("record %+v, PowerShell %+v", got, w)
		}
		// no experts.bin yet (the first setup): the scratch file on the output volume, removed after
		os.Remove(filepath.Join(out, "experts.bin"))
		m2 := &fakeMeter{}
		p2 := la38Launcher(t, m2).startupProbe(out)
		if !p2.OK || !p2.Scratch || m2.sequence() != "scratch probe" || fileExists(filepath.Join(out, probeScratchName)) {
			t.Errorf("scratch probe %+v %q", p2, m2.sequence())
		}
	})

	t.Run("setup then start: probe, repacker, verify, sweep with the manual pair", func(t *testing.T) {
		// The recorded order through the setup flow (PS:14136-14156 then (8)-(8b)): the probe
		// before the repacker, the verify gate after it, the sweep at the start's sweep step,
		// and with K 2 / N 4 set the sweep selects with N 4 (catalog-n4's windows: QD 8).
		f := newFixture(t, false)
		f.meter.timing = timingOf(tim.Scenarios[2].Timing)
		f.meter.phys = 4096
		os.RemoveAll(f.outDir)
		req := Request{ModelPath: f.modelPath, Settings: map[string]string{"prefetch_k": "2", "prefetch_n": "4", "budget_mb": "5120"}}
		plan, err := f.l.PlanRepack(context.Background(), req, false)
		if err != nil {
			t.Fatal(err)
		}
		ex := &loggedRepacker{m: f.meter, write: func(out string) { writeSweepableArtifacts(t, out, "fixture-profile", f.expectSha) }}
		if _, err := f.l.RunRepack(context.Background(), plan, ex, nil); err != nil {
			t.Fatal(err)
		}
		pr, err := f.l.prepare(context.Background(), req, modeStart, nil)
		if pr != nil && pr.locks != nil {
			pr.locks.release()
		}
		if err != nil {
			t.Fatal(err)
		}
		want := "scratch probe repack " + strings.Trim(fmt.Sprint(rec.Scenarios[2].Order), "[]")
		if got := f.meter.sequence(); got != want {
			t.Errorf("sequence %q, want %q", got, want)
		}
		res := pr.res
		if res.SweepArm != armManual || res.Sweep.QD != rec.Scenarios[2].Selection.QD || res.Sweep.Reason != manualPreferred ||
			res.Config.QD != 8 || !res.Config.Prefetch.On || res.Config.Prefetch.N != 4 {
			t.Errorf("start: arm %s sweep %+v QD %d prefetch %+v", res.SweepArm, res.Sweep, res.Config.QD, res.Config.Prefetch)
		}
		t.Logf("sequence %q · arm %s · QD%d (%s), PowerShell catalog-fixed N=4: QD%d", f.meter.sequence(), res.SweepArm,
			res.Sweep.QD, res.Sweep.Reason, rec.Scenarios[2].Selection.QD)
	})
}

// writeSweepableArtifacts is writePackedArtifacts with the layout the sweep reads (the record's
// strides and alignment) and a 64 MiB experts.bin (32 blocks of 2 MiB).
func writeSweepableArtifacts(t *testing.T, outDir, profileID, expectSha string) {
	t.Helper()
	lock := map[string]any{"profile_id": profileID, "expect_sha256": expectSha}
	writeJSON(t, filepath.Join(outDir, "manifest.json"), map[string]any{
		"schema_version": "2.0", "reference_lock": lock, "model": map[string]any{"n_expert": 4},
		"layout": map[string]any{"slot_stride_max": 3145728, "align_bytes": 4096, "layers": []any{
			map[string]any{"stride_bytes": 1048576}, map[string]any{"stride_bytes": 3145728},
			map[string]any{"stride_bytes": 2097152}, map[string]any{"stride_bytes": 2097152}}}})
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
	f, err := os.Create(filepath.Join(outDir, "experts.bin"))
	if err != nil {
		t.Fatal(err)
	}
	f.Truncate(67108864)
	f.Close()
}

// loggedRepacker is the mock repacker of the setup flow: it logs every call (into the meter's
// sequence as «plan» or «repack», and with its argument list), writes the output a real run
// writes, or waits for the cancel; exited is set when its Run has returned (no child left).
type loggedRepacker struct {
	m       *fakeMeter
	mu      sync.Mutex
	argv    [][]string
	write   func(out string)
	planOut string // the --plan text
	block   bool
	started chan struct{}
	exited  bool
}

func (r *loggedRepacker) Run(ctx context.Context, exe string, argv []string, workDir string, env []string, outLog, errLog string, line func(string)) (int, error) {
	defer func() { r.mu.Lock(); r.exited = true; r.mu.Unlock() }()
	r.mu.Lock()
	r.argv = append(r.argv, append([]string{exe}, argv...))
	r.exited = false
	r.mu.Unlock()
	plan := contains(argv, "--plan")
	if r.m != nil {
		r.m.record(meterCall{Kind: map[bool]string{true: "plan", false: "repack"}[plan]})
	}
	os.WriteFile(errLog, nil, 0o644)
	if plan {
		os.WriteFile(outLog, []byte(r.planOut), 0o644)
		return 0, nil
	}
	out := ""
	for i, a := range argv {
		if a == "--out" && i+1 < len(argv) {
			out = argv[i+1]
		}
	}
	os.WriteFile(outLog, []byte("[repack] moe layer 3 (1/2) done\n[repack] moe layer 4 (2/2) done\n"), 0o644)
	if line != nil {
		line("[repack] moe layer 3 (1/2) done")
	}
	if r.block {
		os.MkdirAll(out, 0o755)
		os.WriteFile(filepath.Join(out, "experts.bin.partial"), []byte{1, 2, 3}, 0o644)
		if r.started != nil {
			close(r.started)
		}
		<-ctx.Done()
		return -1, ctx.Err()
	}
	if r.write != nil {
		r.write(out)
	}
	return 0, nil
}

func (r *loggedRepacker) calls() [][]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([][]string(nil), r.argv...)
}

// TestS8LA24 (S8-LA-24; L:4196 rules 9 and 18; LFS Q5.14; conflict row 9): a model that needs a
// repack is marked so; the plan comes from the repacker's --plan call (what, where, disk against
// the drive's free space, time unknown the first time then the last run's); a disk shortfall is
// a refusal; the run starts only after the confirm action, and verify, the speed check and the
// start follow; the cancel leaves no repacker process and no new output folder.
func TestS8LA24(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t, false)
	os.RemoveAll(f.outDir)                                                                   // the first setup: no output folder yet
	req := Request{ModelPath: f.modelPath, Settings: map[string]string{"budget_mb": "5120"}} // the 64 MiB fixture bin is below the profile's minimum

	// (1) marked: the start refuses with «setup needed», both modes read «needs repack»
	if _, err := f.l.Preflight(ctx, req); err == nil {
		t.Fatal("a model without its setup started")
	} else if rn, ok := err.(*RepackNeededError); !ok || rn.Mode != RepackPacked {
		t.Fatalf("refusal %v", err)
	}
	mfs, err := f.l.FormatGates(ctx, f.modelPath, "fixture-profile")
	if err != nil || len(mfs) != 2 || mfs[0].Result.State != GateNeedsRepack || mfs[1].Result.State != GateNeedsRepack {
		t.Fatalf("gates %v %+v", err, mfs)
	}

	// (2) the plan: core's figures, then the repacker's --plan (writes nothing); the box's n is
	// the --plan's experts.bin size; the time is unknown on the first setup
	plan, err := f.l.PlanRepack(ctx, req, false)
	if err != nil || plan.HasLastDuration || plan.OutputDir != f.outDir {
		t.Fatalf("plan %v %+v", err, plan)
	}
	ex := &loggedRepacker{m: f.meter, planOut: "profile=fixture-profile\nexperts.bin expected size=2147483648 B (2.147 GB)\n",
		write: func(out string) { writeSweepableArtifacts(t, out, "fixture-profile", f.expectSha) }}
	if _, err := f.l.PlanText(ctx, plan, ex); err != nil {
		t.Fatal(err)
	}
	if plan.NeedMiB != 2048 || plan.LeftMiB != plan.FreeMiB-2048 || !plan.NeedKnown || plan.Refusal != nil {
		t.Errorf("--plan figures: need %d left %d free %d", plan.NeedMiB, plan.LeftMiB, plan.FreeMiB)
	}
	if c := ex.calls(); len(c) != 1 || !contains(c[0], "--plan") || fileExists(f.outDir) || f.meter.sequence() != "plan" {
		t.Fatalf("the plan wrote or ran something: %v %q", c, f.meter.sequence())
	}
	t.Logf("plan box: writes %s · need %d MB · free %d MB · left %d MB · last: unknown", plan.OutputDir, plan.NeedMiB, plan.FreeMiB, plan.LeftMiB)

	// (3) a disk shortfall: the --plan figure above the free space is refused (fail_resource)
	f.l.os.volumeFree = func(string) (int64, int64, string, error) { return 1024, 1000000, `X:\`, nil }
	short, _ := f.l.PlanRepack(ctx, req, false)
	if _, err := f.l.PlanText(ctx, short, ex); StatusOf(err) != StatusFailResource || short.Refusal == nil ||
		!strings.Contains(err.Error(), "need 2048 MB, free 1024 MB") {
		t.Errorf("shortfall %v", err)
	}
	if _, err := f.l.RunRepack(ctx, short, ex, nil); StatusOf(err) != StatusFailResource || len(ex.calls()) != 2 {
		t.Errorf("a refused plan ran: %v %v", err, ex.calls())
	}
	f.l.os.volumeFree = func(string) (int64, int64, string, error) { return 500000, 1000000, `X:\`, nil }

	// (4) the cancel during the run: the repacker is ended, the output folder the run made is
	// gone, the next start begins from scratch (cancelled_user)
	blk := &loggedRepacker{block: true, started: make(chan struct{})}
	cctx, cancel := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { _, err := f.l.RunRepack(cctx, plan, blk, nil); done <- err }()
	<-blk.started
	cancel()
	if err := <-done; StatusOf(err) != StatusCancelledUser {
		t.Errorf("cancel %v", err)
	}
	if !blk.exited || fileExists(f.outDir) {
		t.Errorf("after the cancel: repacker ended %v, output folder left %v", blk.exited, fileExists(f.outDir))
	}

	// (5) after the confirm action: the run (probe, repacker, verify), then the start's sweep
	f.meter.calls = nil
	f.meter.timing = timingOf(readTimings(t).Scenarios[0].Timing)
	plan, _ = f.l.PlanRepack(ctx, req, false)
	clock := time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC) // each reading 30 s on: the run has a duration
	f.l.os.now = func() time.Time { clock = clock.Add(30 * time.Second); return clock }
	var stages []string
	res, err := f.l.RunRepack(ctx, plan, ex, func(e RepackEvent) { stages = append(stages, string(e.Stage)) })
	if err != nil || res.Gate.State != GatePass {
		t.Fatalf("run %v %+v", err, res.Gate)
	}
	pr, err := f.l.prepare(ctx, req, modeStart, nil)
	if pr != nil && pr.locks != nil {
		pr.locks.release()
	}
	if err != nil || pr.res.Sweep.QDSource != "measured-sweep" || pr.res.Config.QD != 4 {
		t.Fatalf("start after the setup: %v %+v", err, pr.res.Sweep)
	}
	if got, want := strings.Join(stages, " "), "probe probe repack repack verify done"; got != want {
		t.Errorf("stages %q, want %q", got, want)
	}
	if got, want := f.meter.sequence(), "scratch probe repack 1 2 4 8 8 4 2 1"; got != want {
		t.Errorf("sequence %q, want %q", got, want)
	}
	// (6) the next plan shows the last setup's time
	if p, err := f.l.PlanRepack(ctx, req, true); err != nil || !p.HasLastDuration || p.LastDuration != res.Duration {
		t.Errorf("last time %v %+v", err, p)
	}
	t.Logf("run stages %v · sequence %q · start QD %d (%s)", stages, f.meter.sequence(), pr.res.Config.QD, pr.res.QDSource)
}

// la34Record is testdata/la34/ps_recorded.json: Invoke-Repacker's argument lists.
type la34Record struct {
	Cases []struct {
		Name     string    `json:"name"`
		Mode     string    `json:"mode"`
		PlanOnly bool      `json:"plan_only"`
		Template bool      `json:"template"`
		Exe      string    `json:"exe"`
		Args     psStrings `json:"args"`
		WorkDir  string    `json:"workdir"`
		Env      struct {
			IOEnc string `json:"PYTHONIOENCODING"`
			Unbuf string `json:"PYTHONUNBUFFERED"`
		} `json:"env"`
	} `json:"cases"`
	Root, Model, Out, State string
	Catalog                 struct {
		Exe  string    `json:"repacker_exe"`
		Argv psStrings `json:"repacker_argv"`
	} `json:"catalog"`
	ProfileID   string `json:"profile_id"`
	RoutedScope string `json:"routed_scope"`
}

// TestS8LA34 (S8-LA-34; LFS Q5.11, Q5.16; Q4.A14, A15; W7 duty 12): the repacker gets the
// recorded arguments; a partial output is handled through the redo (the plan, then the run that
// clears it first) and the remove-half-made actions; an output made from other model files is
// reported stale in the setup line's verdict.
func TestS8LA34(t *testing.T) {
	ctx := context.Background()
	t.Run("recorded arguments", func(t *testing.T) {
		b, err := os.ReadFile(filepath.Join("testdata", "la34", "ps_recorded.json"))
		if err != nil {
			t.Fatal(err)
		}
		var rec la34Record
		if err := json.Unmarshal(b, &rec); err != nil {
			t.Fatal(err)
		}
		cat := &Catalog{Runtime: Runtime{RepackerExe: rec.Catalog.Exe, RepackerArgv: rec.Catalog.Argv}}
		prof := &Profile{ID: rec.ProfileID, RoutedScope: rec.RoutedScope}
		logPath := filepath.Join(rec.State, StateLogsDir, "repack_log.jsonl")
		if len(rec.Cases) != 6 {
			t.Fatalf("%d cases", len(rec.Cases))
		}
		for _, c := range rec.Cases {
			var exe string
			var argv []string
			switch {
			case c.Template && c.PlanOnly:
				exe, argv = templatePlanArgv(cat, rec.Root, rec.Model, rec.Out, logPath)
			case c.Template:
				exe, argv = templateRunArgv(cat, rec.Root, rec.Model, rec.Out, logPath)
			default:
				exe, argv = repackArgv(cat, rec.Root, prof, rec.Model, rec.Out, c.Mode, logPath, c.PlanOnly)
			}
			if exe != c.Exe || !reflect.DeepEqual(argv, []string(c.Args)) {
				t.Errorf("%s:\n Go         %s %q\n PowerShell %s %q", c.Name, exe, argv, c.Exe, c.Args)
			}
			env := strings.Join(repackEnv(), "\n")
			if c.WorkDir != rec.Root || !strings.Contains(env, "PYTHONIOENCODING="+c.Env.IOEnc) || !strings.Contains(env, "PYTHONUNBUFFERED="+c.Env.Unbuf) {
				t.Errorf("%s: workdir %s env %+v", c.Name, c.WorkDir, c.Env)
			}
			t.Logf("%s: %s %s", c.Name, filepath.Base(exe), strings.Join(argv, " "))
		}
	})

	t.Run("a partial output: remove-half-made, then redo", func(t *testing.T) {
		f := newFixture(t, false)
		// a cut-off setup: the marker and what the repacker had written
		os.WriteFile(filepath.Join(f.outDir, "experts.bin.partial"), make([]byte, 4096), 0o644)
		mf := f.l.modelFile(ctx, mustCatalog(t, f), true, f.modelPath, filepath.Dir(f.modelPath))
		if !mf.SetupCut || mf.PartialBytes <= 4096 || mf.PartialAt.IsZero() {
			t.Fatalf("cut-off %+v", mf)
		}
		if _, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath}); err == nil {
			t.Fatal("a cut-off setup started")
		} else if rn, ok := err.(*RepackNeededError); !ok || !rn.Partial {
			t.Fatalf("refusal %v", err)
		}
		// remove-half-made: the four files go, nothing is repacked
		gone, err := f.l.RemovePartialSetup(ctx, f.modelPath, "fixture-profile")
		if err != nil || len(gone) != 4 {
			t.Fatalf("remove %v %v", err, gone)
		}
		for _, n := range PackedArtifactSet {
			if fileExists(filepath.Join(f.outDir, n)) {
				t.Errorf("%s left", n)
			}
		}
		if _, err := f.l.RemovePartialSetup(ctx, f.modelPath, "fixture-profile"); err == nil {
			t.Error("a second remove found a cut-off setup")
		}
		// redo: a new cut-off; the plan lists the half-made files, its run clears them first
		os.WriteFile(filepath.Join(f.outDir, "experts.bin.partial"), []byte{1}, 0o644)
		os.WriteFile(filepath.Join(f.outDir, "manifest.json"), []byte("{}"), 0o644)
		plan, err := f.l.PlanRepack(ctx, Request{ModelPath: f.modelPath}, false)
		if err != nil || !plan.Partial || len(plan.Deletes) != 2 {
			t.Fatalf("redo plan %v %+v", err, plan)
		}
		ex := &loggedRepacker{write: func(out string) {
			writePackedArtifacts(t, out, "fixture-profile", f.expectSha)
			os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		}}
		var evs []RepackEvent
		if _, err := f.l.RunRepack(ctx, plan, ex, func(e RepackEvent) { evs = append(evs, e) }); err != nil || evs[0].Stage != RepackStageClear {
			t.Fatalf("redo run %v %+v", err, evs)
		}
		if fileExists(filepath.Join(f.outDir, "experts.bin.partial")) || len(ex.calls()) != 1 {
			t.Errorf("after the redo: partial %v calls %v", fileExists(filepath.Join(f.outDir, "experts.bin.partial")), ex.calls())
		}
	})

	t.Run("stale output in the setup line's verdict", func(t *testing.T) {
		f := newFixture(t, true) // the catalog pins the fixture bytes
		ms, _ := fileSha256(f.modelPath)
		// another file under the same name: the output in its folder came from the pinned bytes
		b, _ := os.ReadFile(f.modelPath)
		b[len(b)-1] ^= 0xFF
		os.WriteFile(f.modelPath, b, 0o644)
		settleFile(t, f.modelPath)
		if _, err := f.l.shardDigests(ctx, siblingPaths(f.modelPath), true); err != nil {
			t.Fatal(err)
		}
		mfs, err := f.l.FormatGates(ctx, f.modelPath, "fixture-profile")
		if err != nil || len(mfs) != 2 || !mfs[0].Stale || mfs[0].Result.State != GatePass || mfs[1].Stale {
			t.Fatalf("gates %v %+v (pin %s)", err, mfs, ms)
		}
		if _, err := f.l.Preflight(ctx, Request{ModelPath: f.modelPath}); err == nil {
			t.Fatal("a stale output started")
		} else if rn, ok := err.(*RepackNeededError); !ok || !rn.Stale {
			t.Fatalf("refusal %v", err)
		}
		plan, err := f.l.PlanRepack(ctx, Request{ModelPath: f.modelPath}, false)
		if err != nil || !plan.Stale || len(plan.Deletes) != 3 {
			t.Errorf("stale plan %v %+v", err, plan)
		}
		// the pinned bytes back: no stale verdict
		b[len(b)-1] ^= 0xFF
		os.WriteFile(f.modelPath, b, 0o644)
		settleFile(t, f.modelPath)
		f.l.shardDigests(ctx, siblingPaths(f.modelPath), true)
		if mfs, _ := f.l.FormatGates(ctx, f.modelPath, "fixture-profile"); mfs[0].Stale {
			t.Error("the pinned file reads stale")
		}
	})

	t.Run("template setup (addendum 1)", func(t *testing.T) {
		la34Template(t)
	})
}

func mustCatalog(t *testing.T, f *fixture) *Catalog {
	t.Helper()
	cat, err := ReadCatalog(f.catalogPath, f.root)
	if err != nil {
		t.Fatal(err)
	}
	return cat
}

// la34Template is addendum 1 (step 6 of PowerShell's template path, PS:14153-14156): a model
// outside the catalog sets up by the template repack - virtual refused, the plan box from the
// derive plan (no second --plan), the run with the template argv without --plan, the gate bound
// to the derived expect, and the start afterwards. The model is the GL-14 fixture
// testdata/la31/gguf/gpt-oss-tpl.gguf.fixture.
func la34Template(t *testing.T) {
	ctx := context.Background()
	rec, models := la31Load(t)
	f := newFixture(t, false)
	mdir := filepath.Join(f.dir, "models", "tpl")
	model := filepath.Join(mdir, "gpt-oss-tpl.gguf")
	b, err := os.ReadFile(filepath.Join(models, "gpt-oss-tpl.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	writeModel(t, model, b)
	derive := &la31PlanExec{text: map[string]string{"gpt-oss-tpl.gguf": la31Plan(t, "gpt-oss")}}
	f.l.deriveExec = derive
	lockID := rec.Derived["gpt-oss"].LockID

	if _, err := f.l.PlanRepack(ctx, Request{ModelPath: model, RepackMode: RepackVirtual, NoPreset: true}, false); StatusOf(err) != StatusFailCustomArgs || derive.calls != 0 {
		t.Fatalf("virtual template setup: %v (derive calls %d)", err, derive.calls)
	}
	req := Request{ModelPath: model, NoPreset: true}
	plan, err := f.l.PlanRepack(ctx, req, false)
	if err != nil || !plan.Template || plan.Mode != RepackPacked || derive.calls != 1 {
		t.Fatalf("template plan %v %+v", err, plan)
	}
	tp, _ := parseTemplatePlan(la31Plan(t, "gpt-oss"))
	if plan.NeedMiB != psLongMiB(tp.ExpertBytesTotal) || !plan.NeedKnown || plan.PlanText == "" {
		t.Errorf("plan box need %d, the derive plan's %d B", plan.NeedMiB, tp.ExpertBytesTotal)
	}
	ex := &loggedRepacker{write: func(out string) {
		writeJSON(t, filepath.Join(out, DerivedExpectFileName), map[string]any{"derived_from": "gpt-oss@1"})
		dsha, _ := fileSha256(filepath.Join(out, DerivedExpectFileName))
		os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, out, lockID, dsha)
	}}
	if txt, err := f.l.PlanText(ctx, plan, ex); err != nil || txt != plan.PlanText || len(ex.calls()) != 0 {
		t.Errorf("template plan text: %v, a second --plan ran: %v", err, ex.calls())
	}
	if !contains(plan.Writes, filepath.Join(plan.OutputDir, DerivedExpectFileName)) {
		t.Errorf("writes %v", plan.Writes)
	}
	res, err := f.l.RunRepack(ctx, plan, ex, nil)
	if err != nil || res.Gate.State != GatePass {
		t.Fatalf("template run %v %+v", err, res.Gate)
	}
	c := ex.calls()
	if len(c) != 1 || !contains(c[0], "--experimental-arch-template") || contains(c[0], "--plan") || contains(c[0], "--profile") || contains(c[0], "--scope") {
		t.Errorf("template run argv %v", c)
	}
	if _, err := f.l.Preflight(ctx, req); err != nil {
		t.Errorf("preflight after the template setup: %v", err)
	}
	if derive.calls != 1 {
		t.Errorf("the derive plan ran %d times", derive.calls)
	}
	t.Logf("template setup: need %d MB (%d B) from the derive plan, run argv %s", plan.NeedMiB, tp.ExpertBytesTotal, strings.Join(c[0][len(c[0])-7:], " "))
}
