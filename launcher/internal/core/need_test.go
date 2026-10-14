// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"
)

// X-FIT's recorded stdout of llama-fit-params (run 20261012T065637Z; sha256 as listed in
// reviews/_landing_8_X-FIT_v8-xrun_records.txt, commit 3ab805fd8), byte for byte.
var xfitStdout = []struct {
	name, sha, out string
	dev, host      FitDevice
}{
	{"f01_qwen_c12288_noenv", "9b0163871fc3a90ae27979b14bfba490d8dd1bd0c0da37c682b0f7e3f60ed19e",
		"CUDA0 4193 437 1301 \r\nHost 69817 0 24 \r\n", FitDevice{"CUDA0", 4193, 437, 1301}, FitDevice{"Host", 69817, 0, 24}},
	{"f03_qwen_c240000_noenv", "9e69d6d3e054fbaa405a51c29343436ee427a3d0b177bb7e3e5ae6c0a6771006",
		"CUDA0 4193 5777 1523 \r\nHost 69817 0 246 \r\n", FitDevice{"CUDA0", 4193, 5777, 1523}, FitDevice{"Host", 69817, 0, 246}},
	{"f07_dsv4_c8192_noenv", "7c71e1eee7f6577a5664a62e103876c766e8e5315ff5ffaeace13027886278e7",
		"CUDA0 7766 101 1251 \r\nHost 141362 0 21 \r\n", FitDevice{"CUDA0", 7766, 101, 1251}, FitDevice{"Host", 141362, 0, 21}},
	{"f09_dsv4_c131072_noenv", "e2bf89ff1658f616b7bf2f5afcf0467f954998d1c46f719ef9ab21d67b1608c8",
		"CUDA0 7766 903 1299 \r\nHost 141362 0 81 \r\n", FitDevice{"CUDA0", 7766, 903, 1299}, FitDevice{"Host", 141362, 0, 81}},
}

// the catalog's Qwen argv (models.json profile qwen35-122b-nonextn, defaults.argv) as the Go
// server argv carries it, with the launcher's own additions
var qwenServerArgv = []string{"-ngl", "99", "--n-cpu-moe", "49", "-c", "12288", "-t", "8", "-b", "2048", "-ub", "512",
	"-fa", "on", "-np", "1", "--host", "127.0.0.1", "--port", "8093", "--no-webui", "--no-warmup",
	"-m", `D:\m\Qwen-00001-of-00002.gguf`, "--cache-ram", "0", "--ctx-checkpoints", "4", "--alias", "qwen"}

func fakeFit(out string, code int, err error) fitRunner {
	return func(context.Context, string, []string, []string, string) ([]byte, []byte, int, error) {
		return []byte(out), []byte("log line\n"), code, err
	}
}

// TestS8LA11 (S8-LA-11; GL-05): ① the fit-params argv repeats the model-affecting part of the
// server argv and holds no server-only option; ② X-FIT's recorded stdout parses to its
// integers; ③ exit 1, an abnormal end or an unparsable line gives «unmeasured».
func TestS8LA11(t *testing.T) {
	// ① X-FIT's f01 call (call.json argv less the exe), from the server argv
	want := []string{"-ngl", "99", "--n-cpu-moe", "49", "-c", "12288", "-t", "8", "-b", "2048", "-ub", "512", "-fa", "on",
		"-np", "1", "--fit-print", "on", "-m", `D:\m\Qwen-00001-of-00002.gguf`}
	got := FitArgv(qwenServerArgv, `D:\m\Qwen-00001-of-00002.gguf`)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("fit argv\n got %q\nwant %q", got, want)
	}
	for _, a := range got {
		for _, s := range []string{"--host", "--port", "--no-webui", "--no-warmup", "--cache-ram", "--ctx-checkpoints", "--alias"} {
			if a == s {
				t.Fatalf("server-only %s in %q", s, got)
			}
		}
	}
	// ② the recorded outputs, byte for byte, through the runner
	for _, x := range xfitStdout {
		h := sha256.Sum256([]byte(x.out))
		if hex.EncodeToString(h[:]) != x.sha {
			t.Fatalf("%s: the test's copy is not the recorded bytes", x.name)
		}
		r := RunFitParams(context.Background(), fakeFit(x.out, 0, nil), "fit.exe", qwenServerArgv, "m.gguf", nil)
		if !r.OK || len(r.Print.Devices) != 1 || r.Print.Devices[0] != x.dev || r.Print.Host != x.host {
			t.Fatalf("%s: %+v", x.name, r)
		}
	}
	// ③ every failure is «unmeasured»
	for name, run := range map[string]fitRunner{
		"exit 1 (argument error)":            fakeFit("", 1, nil),
		"exit 0xC0000409 (load failure f13)": fakeFit("", 0xC0000409, nil),
		"did not run":                        fakeFit("", 0, errors.New("file not found")),
		"unparsable line":                    fakeFit("CUDA0 4193 437 1301 \r\nHost 69817 0 x24 \r\n", 0, nil),
		"extra text":                         fakeFit("CUDA0 4193 437 1301 \r\nwarning: x\r\nHost 69817 0 24 \r\n", 0, nil),
		"no Host line":                       fakeFit("CUDA0 4193 437 1301 \r\n", 0, nil),
		"empty stdout":                       fakeFit("", 0, nil),
	} {
		r := RunFitParams(context.Background(), run, "fit.exe", qwenServerArgv, "m.gguf", nil)
		p := ProjectNeed(r, &ModelShape{}, nil, "id", 12288, 12288, true, 0, nil)
		if r.OK || r.Reason == "" || p.Label != NeedUnmeasured {
			t.Fatalf("%s: %+v %+v", name, r, p)
		}
		if _, _, _, l := ShownNeed(0, NeedNotMeasured, p, true); l != NeedUnmeasured {
			t.Fatalf("%s: shown %v", name, l)
		}
	}
	t.Logf("argv %d items = X-FIT f01; %d recorded outputs parsed; 7 failures unmeasured", len(got), len(xfitStdout))
}

// ggufWithExperts builds a small MoE GGUF: numeric metadata, a token list, general.name and
// one expert tensor of each part in each layer (Q8_0), with the data section after them.
func ggufWithExperts(name string, layers, embd, ff, experts int) []byte {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	ws := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	type tens struct {
		name string
		dims []uint64
	}
	var ts []tens
	for l := 0; l < layers; l++ {
		for _, p := range []string{"gate", "up"} {
			ts = append(ts, tens{"blk." + itoa(l) + ".ffn_" + p + "_exps.weight", []uint64{uint64(embd), uint64(ff), uint64(experts)}})
		}
		ts = append(ts, tens{"blk." + itoa(l) + ".ffn_down_exps.weight", []uint64{uint64(ff), uint64(embd), uint64(experts)}})
	}
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(ts)))
	w(uint64(7))
	ws("general.architecture")
	w(uint32(8))
	ws("testmoe")
	ws("general.name")
	w(uint32(8))
	ws(name)
	for _, kv := range []struct {
		k string
		v uint32
	}{{"testmoe.block_count", uint32(layers)}, {"testmoe.embedding_length", uint32(embd)}, {"testmoe.expert_count", uint32(experts)}} {
		ws(kv.k)
		w(uint32(4))
		w(kv.v)
	}
	ws("tokenizer.ggml.tokens")
	w(uint32(9))
	w(uint32(8))
	w(uint64(3))
	for _, s := range []string{"a", "b", "c"} {
		ws(s)
	}
	ws("general.alignment")
	w(uint32(4))
	w(uint32(32))
	off := uint64(0)
	for _, x := range ts {
		ws(x.name)
		w(uint32(len(x.dims)))
		for _, d := range x.dims {
			w(d)
		}
		w(uint32(8)) // Q8_0
		w(off)
		n := x.dims[0] / 32 * 34 * x.dims[1] * x.dims[2]
		off += (n + 31) / 32 * 32
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, off))
	return b.Bytes()
}

func itoa(i int) string { return strconv.Itoa(i) }

// TestS8LA12 (S8-LA-12; GL-05): a GGUF fixture and its copy under another file name and
// general.name get the same need; the need with an empty catalog equals the need with the
// shipped catalog; a model absent from the catalog gets a need or «unmeasured», never a
// refusal for being unknown.
func TestS8LA12(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "fixture-moe.gguf")
	b := filepath.Join(dir, "renamed copy of something else.gguf")
	if err := os.WriteFile(a, ggufWithExperts("Fixture MoE", 3, 256, 128, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, ggufWithExperts("A wholly different general.name for the same weights", 3, 256, 128, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	shipped, err := ReadCatalog(realCatalog, la31Expects) // the shipped catalog (S8-LA-31's reading)
	if err != nil {
		t.Fatal(err)
	}
	empty := &Catalog{Runtime: shipped.Runtime}
	cfg := &Config{Argv: qwenServerArgv, Budget: Budget{MB: 64}, Prefetch: Prefetch{On: true}}
	need := func(path string, cat *Catalog, store ResidualStore) NeedProjection {
		set, le := readModelShardSet(path)
		if le != nil {
			t.Fatal(le)
		}
		l := &Launcher{opt: Options{BundleRoot: dir, Residuals: store}, os: osHooks{fit: fakeFit(xfitStdout[0].out, 0, nil)},
			diag: newDiagLog(filepath.Join(dir, "logs"), time.Now)}
		_, p := l.projection(context.Background(), cat, set, cfg)
		return p
	}
	pa, pb := need(a, shipped, nil), need(b, shipped, nil)
	pe := need(a, empty, nil)
	if pa.Label != NeedUnverified || pa.MiB <= 0 {
		t.Fatalf("fixture need: %+v", pa)
	}
	if ceilMiBf(pa.MiB) != ceilMiBf(pb.MiB) || pa.Label != pb.Label {
		t.Fatalf("copy: %v %v, fixture %v %v", pb.MiB, pb.Label, pa.MiB, pa.Label)
	}
	if ceilMiBf(pa.MiB) != ceilMiBf(pe.MiB) {
		t.Fatalf("empty catalog %v, shipped %v", pe.MiB, pa.MiB)
	}
	sa, _ := readModelShape(a)
	sb, _ := readModelShape(b)
	if sa.Identity != sb.Identity {
		t.Fatalf("identity differs: %s %s", sa.Identity, sb.Identity)
	}
	// absent from the catalog and of a structure with recorded residuals but not these
	// settings: «unmeasured», still no refusal
	store := memStore{arch: "testmoe"}
	if p := need(a, shipped, store); p.Label != NeedUnmeasured {
		t.Fatalf("measured structure, no residuals: %+v", p)
	}
	for _, p := range []NeedProjection{pa, need(a, shipped, store)} {
		known, miB, basis, _ := ShownNeed(0, NeedNotMeasured, p, true)
		c := EvaluateStart(65536, miB, basis, 64)
		if c.Kind == CheckBlocked || (known && c.Kind == CheckUnmeasured) {
			t.Fatalf("an unknown model was refused or lost its need: %+v", c)
		}
	}
	t.Logf("fixture, renamed copy and empty catalog: %d MiB (%v); identity %s", ceilMiBf(pa.MiB), pa.Label, sa.Identity)
}

// memStore is a residual store for the tests.
type memStore struct {
	arch string
	m    map[memKey]Residuals
}

type memKey struct {
	ResidualKey
	MeasureState
}

func (s memStore) Lookup(k ResidualKey, st MeasureState) (Residuals, bool) {
	r, ok := s.m[memKey{k, st}]
	return r, ok
}
func (s memStore) Measured(arch string) bool { return arch == s.arch }
func (s memStore) Put(k ResidualKey, arch string, st MeasureState, r Residuals) error {
	s.m[memKey{k, st}] = r
	return nil
}

// TestS8LA13 (S8-LA-13; GL-05): not validated, the need shown is the last measured value for
// the same model and settings, labelled measured (the last run's maximum memory, or its ready
// value plus the RAM budget); none recorded: «unmeasured». Validated: the projection; a model
// of a structure not measured shows the projection labelled «unverified prediction».
func TestS8LA13(t *testing.T) {
	shape := &ModelShape{Arch: "testmoe", NLayer: 2, NEmbd: 64, NExpert: 4, NVocab: 1000, Identity: "id1",
		Experts: map[int64][]ExpertPart{0: {{Type: 8, Dims: []int64{64, 64, 4}, AbsOffset: 100}}, 1: {{Type: 8, Dims: []int64{64, 64, 4}, AbsOffset: 200}}}}
	fit := RunFitParams(context.Background(), fakeFit(xfitStdout[0].out, 0, nil), "fit.exe", qwenServerArgv, "m.gguf", nil)
	key := ResidualKey{"id1", 12288, 64}
	store := memStore{arch: "testmoe", m: map[memKey]Residuals{
		{key, AfterRequest}:                    {CDevMiB: 341, UMiB: 1041.25, Run: "X-U s01"},
		{ResidualKey{"id1", 8192, 64}, AtLoad}: {CDevMiB: 341, UMiB: 1000, Run: "P1 only"}}}
	unv := ProjectNeed(fit, shape, nil, "id1", 12288, 64, true, 0, nil)
	prj := ProjectNeed(fit, shape, nil, "id1", 12288, 64, true, 0, store)
	none := ProjectNeed(fit, shape, nil, "id1", 8192, 64, true, 0, store)
	if prj.Label != NeedProjected || unv.Label != NeedUnverified || none.Label != NeedUnmeasured {
		t.Fatalf("labels %v %v %v", prj.Label, unv.Label, none.Label)
	}
	if d := prj.MiB - unv.MiB; d < CHostMiB+1041.25-1e-6 || d > CHostMiB+1041.25+1e-6 {
		t.Fatalf("projection − unverified = %v, want c_host + U", d)
	}
	t1 := unv.Terms
	if base := t1.PoolMiB + 24 + t1.OutMiB + t1.SnapMiB + t1.TabMiB; base != unv.MiB {
		t.Fatalf("unverified %v, want pool + P_Hc + T_out + T_snap + T_tab = %v", unv.MiB, base)
	}
	peak := &RunRecord{MaxWorkingSetBytes: 20000 << 20, WorkingSetBytes: 9000 << 20, Start: &StartSettings{BudgetMB: 12288}}
	ready := &RunRecord{WorkingSetBytes: 9000 << 20, Start: &StartSettings{BudgetMB: 12288}}
	for _, c := range []struct {
		name      string
		rec       *RunRecord
		p         NeedProjection
		validated bool
		known     bool
		miB       int64
		label     NeedLabel
	}{
		{"not validated, peak recorded", peak, prj, false, true, 20000, NeedMeasured},
		{"not validated, ready only", ready, prj, false, true, 9000 + 12288, NeedMeasured},
		{"not validated, none recorded", nil, prj, false, false, 0, NeedUnmeasured},
		{"validated, measured structure", peak, prj, true, true, ceilMiBf(prj.MiB), NeedProjected},
		{"validated, structure not measured", peak, unv, true, true, ceilMiBf(unv.MiB), NeedUnverified},
		{"validated, setting never run", peak, none, true, false, 0, NeedUnmeasured},
	} {
		m, b := NeedOf(c.rec)
		known, miB, _, label := ShownNeed(m, b, c.p, c.validated)
		if known != c.known || miB != c.miB || label != c.label {
			t.Fatalf("%s: %v %d %v, want %v %d %v", c.name, known, miB, label, c.known, c.miB, c.label)
		}
	}
	t.Logf("labels: measured · unmeasured · projection %d MiB · unverified prediction %d MiB", ceilMiBf(prj.MiB), ceilMiBf(unv.MiB))

	// the shipped flag is the method's, set by D.3's reading (F-BU B.7 (c))
	if !ProjectionMethodValidated {
		t.Fatal("the projection method's flag is not the one D.3 sets")
	}
	t.Run("residuals from a run", func(t *testing.T) { testResidualsFromRun(t, fit, shape) })
}

// testResidualsFromRun is the store side of S8-LA-13 under F-BU B.7 (a):
// c_dev and U derived from a run's PrivateUsage and GPU dedicated memory with the X-FIT
// formula, stored per key and state, the latest run replacing the entry, a missing reading
// storing nothing; P3 after one request and 60 s idle.
func testResidualsFromRun(t *testing.T, fit FitResult, shape *ModelShape) {
	mib := func(v float64) int64 { return int64(math.Round(v * bytesPerMiB)) }
	// X-FIT s01 (Qwen ctx 12288, budget 12288, P1): private 20,034.1, mirror 6,195.6; FIT's
	// terms pool 12,285.9375, T_out 0.947, T_snap 141.56, T_tab 0.654 (no phase-B pools)
	qt := EngineTerms{PoolMiB: 12285.9375, OutMiB: 993280.0 / bytesPerMiB, SnapMiB: 141.56, TabMiB: 0.654}
	r := DeriveResiduals(mib(20034.1), mib(6195.6), fit.Print, qt)
	if math.Abs(r.CDevMiB-264.6) > 0.05 || math.Abs(r.UMiB-1044.82) > 1.0 {
		t.Fatalf("X-FIT s01: c_dev %.2f (264.6), U %.2f (1,044.82 with the run's T_tab)", r.CDevMiB, r.UMiB)
	}
	// the projection of a measured key gives back the reading's RAM need: private − mirror
	p := NeedProjection{Terms: qt, HostCMiB: float64(fit.Print.Host.Compute)}
	if got := p.Terms.PoolMiB + p.HostCMiB + qt.OutMiB + qt.SnapMiB + qt.TabMiB + CHostMiB + r.UMiB; math.Abs(got-(20034.1-6195.6)) > 1e-6 {
		t.Fatalf("projection from the reading %.4f, want private − mirror %.4f", got, 20034.1-6195.6)
	}

	dir := t.TempDir()
	fs := FileResidualStore{Path: filepath.Join(dir, "MoE-Direct", residualFile)}
	pid := 4242
	var gpu map[string]int64
	var gerr error
	l := &Launcher{opt: Options{Residuals: fs}, diag: newDiagLog(filepath.Join(dir, "logs"), time.Now),
		os: osHooks{gpuMem: func() (map[string]int64, error) { return gpu, gerr }}}
	pre := &PreflightResult{Fit: fit, Projection: ProjectNeed(fit, shape, nil, "id1", 12288, 64, true, 0, nil)}
	key := pre.Projection.Key
	pm := ProcessMemory{PrivateBytes: mib(20034.1)}
	if fs.Measured("testmoe") {
		t.Fatal("an empty store measured a structure")
	}
	// a missing reading stores nothing: no GPU reading, no instance for the PID, no private
	// reading, no fit-params terms
	gerr = errors.New("PdhCollectQueryData failed")
	l.recordResiduals(pre, AtLoad, pid, pm, nil, "r0")
	gerr, gpu = nil, map[string]int64{"pid_77_luid_0x0_phys_0": mib(6195.6)}
	l.recordResiduals(pre, AtLoad, pid, pm, nil, "r0")
	gpu["pid_4242_luid_0x0_phys_0"] = mib(6195.6)
	l.recordResiduals(pre, AtLoad, pid, ProcessMemory{}, errors.New("access denied"), "r0")
	l.recordResiduals(&PreflightResult{Fit: FitResult{Reason: "exit 0xC0000409"}, Projection: pre.Projection}, AtLoad, pid, pm, nil, "r0")
	if _, ok := fs.Lookup(key, AtLoad); ok || fs.Measured("testmoe") {
		t.Fatal("a missing reading stored residuals")
	}
	// P1, then P3; the key is exact (identity, ctx, budget) and per state
	l.recordResiduals(pre, AtLoad, pid, pm, nil, "run 1")
	if _, ok := fs.Lookup(key, AfterRequest); ok {
		t.Fatal("a P1 reading answered for P3")
	}
	l.recordResiduals(pre, AfterRequest, pid, ProcessMemory{PrivateBytes: mib(20479.5)}, nil, "run 1")
	p1, ok1 := fs.Lookup(key, AtLoad)
	p3, ok3 := fs.Lookup(key, AfterRequest)
	if !ok1 || !ok3 || p1.Run != "run 1" || math.Abs(p3.UMiB-p1.UMiB-445.4) > 1e-6 || !fs.Measured("testmoe") {
		t.Fatalf("P1 %+v %v, P3 %+v %v", p1, ok1, p3, ok3)
	}
	for _, k := range []ResidualKey{{"id1", 8192, 64}, {"id1", 12288, 128}, {"id2", 12288, 64}} {
		if _, ok := fs.Lookup(k, AfterRequest); ok {
			t.Fatalf("key %+v found by %+v", k, key)
		}
	}
	// the latest run replaces the entry
	l.recordResiduals(pre, AfterRequest, pid, ProcessMemory{PrivateBytes: mib(20500)}, nil, "run 2")
	if p3b, _ := fs.Lookup(key, AfterRequest); p3b.Run != "run 2" || math.Abs(p3b.UMiB-p3.UMiB-20.5) > 1e-6 {
		t.Fatalf("latest run: %+v (was %+v)", p3b, p3)
	}
	if n := len(fs.read().Entries); n != 2 {
		t.Fatalf("%d entries, want 2 (P1 and P3 of one key)", n)
	}
	// the stored P3 entry is what the projection of the next check reads
	if pr := ProjectNeed(fit, shape, nil, "id1", 12288, 64, true, 0, fs); pr.Label != NeedProjected || pr.Res.Run != "run 2" {
		t.Fatalf("projection from the store: %+v", pr)
	}
	// an unreadable or other-schema file reads as empty
	if err := os.WriteFile(fs.Path, []byte(`{"schema_version":99,"entries":[{"identity":"id1"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if fs.Measured("testmoe") {
		t.Fatal("another schema was read")
	}
	// P3 is due after one request and 60 s idle, with none in flight
	t0 := time.Date(2026, 10, 12, 12, 0, 0, 0, time.UTC)
	for _, c := range []struct {
		name string
		sum  TimingSummary
		now  time.Time
		due  bool
	}{
		{"no request", TimingSummary{LastActivity: t0}, t0.Add(2 * time.Minute), false},
		{"59 s idle", TimingSummary{Requests: 1, LastActivity: t0}, t0.Add(59 * time.Second), false},
		{"60 s idle", TimingSummary{Requests: 1, LastActivity: t0}, t0.Add(60 * time.Second), true},
		{"a request in flight", TimingSummary{Requests: 1, Decoding: true, LastActivity: t0}, t0.Add(5 * time.Minute), false},
		{"prefilling", TimingSummary{Requests: 2, Prefilling: true, LastActivity: t0}, t0.Add(5 * time.Minute), false},
		{"no activity time", TimingSummary{Requests: 1}, t0, false},
	} {
		if got := p3Due(c.sum, c.now); got != c.due {
			t.Fatalf("p3Due %s: %v", c.name, got)
		}
	}
}
