package core

// GL-15 (version 8): warm start, S8-LA-33 of the plan reviews/_plan_8_STAGE1_v8-plansplit.md
// (d1dbe526f), built to the F-LS edition (OQ-6 (a): the warm-start switches) and the PowerShell
// launcher's region 14b (WARMSTART_SPEC v0.5, AUTOSAVE_SPEC v0.1). The recorded PowerShell log
// is the slot request record of the PowerShell selftest bench/moe-direct/launcher/launcher_selftest.ps1:
// the save stub's request and answer (:5912-5926), the restore request and its body (:6188-6189),
// the erase after a failed restore (:6209-6215), the autosave's GET /slots and its save (:7156-7177)
// and the generation alternation a -> b -> a (:7370-7382), and the restore of an elected autosave
// generation (:7531). A mock llama-server (httptest) records every request this package sends;
// the request log is compared, request by request, with that record.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// slotMock is the mock llama-server of the slot API: it writes a state file as llama-server
// does (New-KvDataFile's GGSQ header and payload) and answers as the selftest's stubs answer.
type slotMock struct {
	mu          sync.Mutex
	log         []string // "METHOD URI BODY"
	dir         string   // where a save writes (the server's --slot-save-path)
	tokens      int64    // n_prompt_tokens of slot 0 (GET /slots)
	saved       int64    // n_saved of a save
	restoreFail bool     // a 200 answer with n_restored 0 (selftest :6209)
	eraseFail   bool     // erase answers 500
}

func kvDataBytes(payload int) []byte { // New-KvDataFile (selftest :5225-5233)
	b := make([]byte, 8+payload)
	copy(b, "GGSQ")
	b[4] = 2
	for i := 8; i < len(b); i++ {
		b[i] = byte(i % 251)
	}
	return b
}

func (m *slotMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.log = append(m.log, strings.TrimSpace(r.Method+" "+r.URL.RequestURI()+" "+string(body)))
	name := ""
	var req struct {
		Filename string `json:"filename"`
	}
	if json.Unmarshal(body, &req) == nil {
		name = req.Filename
	}
	switch {
	case r.Method == "GET" && r.URL.Path == "/slots":
		fmt.Fprintf(w, `[{"id":0,"n_ctx":4096,"speculative":false,"is_processing":false,"id_task":7,"n_prompt_tokens":%d}]`, m.tokens)
	case r.URL.Query().Get("action") == "save":
		data := kvDataBytes(64)
		if err := os.WriteFile(filepath.Join(m.dir, name), data, 0o644); err != nil {
			w.WriteHeader(500)
			return
		}
		fmt.Fprintf(w, `{"id_slot":0,"filename":"%s","n_saved":%d,"n_written":%d}`, name, m.saved, len(data))
	case r.URL.Query().Get("action") == "restore":
		if m.restoreFail {
			fmt.Fprintf(w, `{"id_slot":0,"filename":"%s","n_restored":0,"n_read":0}`, name)
			return
		}
		st, err := os.Stat(filepath.Join(m.dir, name))
		if err != nil {
			w.WriteHeader(400)
			return
		}
		fmt.Fprintf(w, `{"id_slot":0,"filename":"%s","n_restored":%d,"n_read":%d}`, name, m.saved, st.Size())
	case r.URL.Query().Get("action") == "erase":
		if m.eraseFail {
			w.WriteHeader(500)
			return
		}
		io.WriteString(w, `{"id_slot":0,"n_erased":0}`)
	default:
		w.WriteHeader(404)
	}
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

func (m *slotMock) take() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := m.log
	m.log = nil
	return out
}

// warmRig is one Launcher with a bundle root, a model file and a state directory, and the mock.
type warmRig struct {
	l        *Launcher
	root     string
	model    string
	manifest string
	mock     *slotMock
	host     string
	port     int
}

func newWarmRig(t *testing.T) *warmRig {
	t.Helper()
	dir := t.TempDir()
	root := filepath.Join(dir, "bundle")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, BundleManifestName), []byte(`{"bundle_manifest_version":1,"files":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(model, []byte("GGUF fixture model bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	l, err := New(Options{CatalogPath: filepath.Join(dir, "models.json"), BundleRoot: root, StateDir: filepath.Join(dir, "state")})
	if err != nil {
		t.Fatal(err)
	}
	m := &slotMock{tokens: 200, saved: 128}
	srv := httptest.NewServer(m)
	t.Cleanup(srv.Close)
	host, ps, _ := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	port, _ := strconv.Atoi(ps)
	return &warmRig{l: l, root: root, model: model, manifest: strings.Repeat("ab", 32), mock: m, host: host, port: port}
}

// config is one start's configuration (a fresh copy: warmConfigure edits the argv).
func (r *warmRig) config() *Config {
	return &Config{Argv: []string{"-m", r.model, "-c", "4096", "-np", "1", "--host", r.host, "--port", strconv.Itoa(r.port), "--alias", "fixture"},
		Env: map[string]string{"LLAMA_ARG_FLASH_ATTN": "on", EnvMetrics: `C:\logs\metrics.csv`}, Host: r.host, Port: r.port}
}

// begin is a start's init and config (Initialize-Warmstart, Build-EffectiveConfig's warm part).
func (r *warmRig) begin(t *testing.T, sw Switches) (*WarmStart, *Config) {
	t.Helper()
	w := r.l.newWarmStart("fixture-profile", false, false, sw)
	cfg := r.config()
	r.l.warmConfigure(context.Background(), w, cfg, r.root, r.manifest, []string{r.model})
	r.mock.dir = w.Dir
	return w, cfg
}

// S8-LA-33: warm start sends the same slot requests with the same file names as the recorded log:
// init and bindings, restore on start, save on unload, autosave while idle.
func TestS8LA33(t *testing.T) {
	r := newWarmRig(t)
	var got []string // the whole request log, in order

	// init and bindings: the kv directory, --slot-save-path, the generation sweep, the verdict
	other := filepath.Join(r.l.kvRootDir(), "other-profile")
	if err := os.MkdirAll(other, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"slot0.kv.tmp.20260101T000000000Z_1", "slot0.kv.stale.x", "slot0.kv.meta.json.tmp.x", "slot0.kv.meta.json.stale.x", "keep.txt"} {
		os.WriteFile(filepath.Join(other, n), []byte("x"), 0o644)
	}
	w, cfg := r.begin(t, DefaultSwitches)
	if es, _ := os.ReadDir(other); len(es) != 1 || es[0].Name() != "keep.txt" {
		t.Errorf("the generation sweep left %v", es)
	}
	if w.Dir != filepath.Join(r.l.StateDir(), "kv", "fixture-profile") || !isDir(w.Dir) {
		t.Errorf("kv dir %q", w.Dir)
	}
	if v, ok := argvValue(cfg.Argv, argSlotSave); !ok || v != w.Dir+`\` {
		t.Errorf("--slot-save-path = %q, %v (want the kv dir with a trailing backslash)", v, ok)
	}
	if w.State() != WarmOn || w.Eligible || w.Text() != "cold(sidecar_missing)" {
		t.Errorf("first start: %s %v %q", w.State(), w.Eligible, w.Text())
	}
	k1, _ := settingsKey("p", r.model, "packed", cfg)
	k0, _ := settingsKey("p", r.model, "packed", r.config())
	if k1 != k0 {
		t.Error("the slot save path must not change the settings key (a preflight has none)")
	}
	// the off states: no --slot-save-path and no request (LS 13-2)
	for _, c := range []struct {
		name           string
		derived, smoke bool
		sw             Switches
		state, text    string
	}{
		{"switch off", false, false, Switches{WarmStart: false, Autosave: true}, WarmSoftOff, "off(user)"},
		{"check run", false, true, DefaultSwitches, WarmHardOff, "off(mode)"},
		{"template profile", true, false, DefaultSwitches, WarmHardOff, "off(mode)"},
	} {
		ow := r.l.newWarmStart("fixture-profile", c.derived, c.smoke, c.sw)
		oc := r.config()
		r.l.warmConfigure(context.Background(), ow, oc, r.root, r.manifest, []string{r.model})
		if _, ok := argvValue(oc.Argv, argSlotSave); ok || ow.State() != c.state || ow.Text() != c.text {
			t.Errorf("%s: slot save path %v, %s, %q", c.name, ok, ow.State(), ow.Text())
		}
		if ok, rec := r.l.warmRestore(context.Background(), ow, oc); ok || rec {
			t.Errorf("%s: an off start restored", c.name)
		}
		r.l.autosaveTick(context.Background(), ow, oc)
		r.l.autosaveTick(context.Background(), ow, oc)
	}
	if lg := r.mock.take(); len(lg) != 0 {
		t.Errorf("init, bindings and the off states sent requests: %q", lg)
	}

	// save on unload: the tmp name, then slot0.kv and its sidecar
	n, err := r.l.warmSave(context.Background(), w, cfg)
	if err != nil || n != 128 {
		t.Fatalf("save: %d %v", n, err)
	}
	r.l.warmTeardownCleanup(w)
	lg := r.mock.take()
	saveRe := regexp.MustCompile(`^POST /slots/0\?action=save \{"filename":"slot0\.kv\.tmp\.\d{8}T\d{9}Z_\d+"\}$`)
	if len(lg) != 1 || !saveRe.MatchString(lg[0]) {
		t.Errorf("save on unload: %q", lg)
	}
	got = append(got, lg...)
	es, _ := os.ReadDir(w.Dir)
	var names []string
	for _, e := range es {
		names = append(names, e.Name())
	}
	if strings.Join(names, ",") != "slot0.kv,slot0.kv.meta.json" {
		t.Errorf("after the save the kv dir holds %v", names)
	}
	mb, _ := os.ReadFile(w.metaPath(kvCanonicalData))
	order := []string{"meta_schema_version", "profile_id", "model_shards_sha256", "repack_manifest_sha256", "engine_bundle_sha256",
		"effective_state_semantics_sha256", "n_ctx", "n_parallel", "llama_state_seq_magic", "llama_state_seq_version", "n_tokens",
		"n_bytes", "kv_file_sha256", "generation_id", "saved_at"}
	last := -1
	for _, k := range order {
		i := strings.Index(string(mb), `"`+k+`"`)
		if i <= last {
			t.Errorf("sidecar field %s out of the PowerShell order", k)
		}
		last = i
	}
	if m, why := readKvSidecar(w.metaPath(kvCanonicalData)); m == nil || m.NTokens != 128 || m.NCtx != 4096 || m.NParallel != 1 ||
		m.LlamaStateSeqMagic != "47475351" || m.LlamaStateSeqVersion != 2 || m.Origin != "" {
		t.Errorf("sidecar %+v %s", m, why)
	}

	// restore on start: eligible, then the restore of slot0.kv after ready
	w2, cfg2 := r.begin(t, DefaultSwitches)
	if !w2.Eligible || w2.Selected != kvCanonicalData || w2.Text() != "eligible" {
		t.Fatalf("second start: %v %q %q %q", w2.Eligible, w2.Selected, w2.Text(), w2.Detail)
	}
	if ok, rec := r.l.warmRestore(context.Background(), w2, cfg2); !ok || rec || w2.Text() != "restored(128 tokens)" {
		t.Errorf("restore: %v %v %q", ok, rec, w2.Text())
	}
	if ok, _ := r.l.warmRestore(context.Background(), w2, cfg2); ok {
		t.Error("a second restore in one start")
	}
	lg = r.mock.take()
	if strings.Join(lg, "|") != `POST /slots/0?action=restore {"filename":"slot0.kv"}` {
		t.Errorf("restore on start: %q", lg)
	}
	got = append(got, lg...)

	// a failed restore: the erase, then cold(restore_failed); a failed erase too: the recovery restart
	r.mock.restoreFail = true
	w3, cfg3 := r.begin(t, DefaultSwitches)
	if ok, rec := r.l.warmRestore(context.Background(), w3, cfg3); ok || rec || w3.Text() != "cold(restore_failed)" {
		t.Errorf("failed restore: %v %v %q", ok, rec, w3.Text())
	}
	r.mock.eraseFail = true
	w4, cfg4 := r.begin(t, DefaultSwitches)
	if ok, rec := r.l.warmRestore(context.Background(), w4, cfg4); ok || !rec {
		t.Errorf("failed restore and erase: %v %v (want the recovery restart)", ok, rec)
	}
	r.l.warmRecovered(w4, 1)
	if w4.Text() != "cold(recovery_cold)" {
		t.Errorf("after the recovery restart: %q", w4.Text())
	}
	r.mock.restoreFail, r.mock.eraseFail = false, false
	lg = r.mock.take()
	want := []string{`POST /slots/0?action=restore {"filename":"slot0.kv"}`, `POST /slots/0?action=erase`,
		`POST /slots/0?action=restore {"filename":"slot0.kv"}`, `POST /slots/0?action=erase`}
	if strings.Join(lg, "|") != strings.Join(want, "|") {
		t.Errorf("restore failures: %q, want %q", lg, want)
	}
	got = append(got, lg...)

	// autosave while idle: a full period first, then GET /slots and a save when the slot changed;
	// the generations alternate a -> b -> a; no change, no save; the switch off, nothing
	defer func(p time.Duration) { kvAutosavePeriod = p }(kvAutosavePeriod)
	kvAutosavePeriod = time.Nanosecond
	off, offCfg := r.begin(t, Switches{WarmStart: true, Autosave: false})
	r.l.autosaveTick(context.Background(), off, offCfg)
	r.l.autosaveTick(context.Background(), off, offCfg)
	if lg := r.mock.take(); len(lg) != 0 {
		t.Errorf("autosave switched off sent %q", lg)
	}
	w5, cfg5 := r.begin(t, DefaultSwitches)
	r.l.autosaveTick(context.Background(), w5, cfg5) // the first tick starts the clock
	for _, tok := range []int64{300, 300, 400, 500} {
		r.mock.tokens, r.mock.saved = tok, tok
		time.Sleep(time.Millisecond)
		r.l.autosaveTick(context.Background(), w5, cfg5)
	}
	lg = r.mock.take()
	want = []string{`GET /slots`, `POST /slots/0?action=save {"filename":"slot0.auto.a.kv"}`, `GET /slots`,
		`GET /slots`, `POST /slots/0?action=save {"filename":"slot0.auto.b.kv"}`,
		`GET /slots`, `POST /slots/0?action=save {"filename":"slot0.auto.a.kv"}`}
	if strings.Join(lg, "|") != strings.Join(want, "|") {
		t.Errorf("autosave: %q, want %q", lg, want)
	}
	got = append(got, lg...)
	if m, _ := readKvSidecar(w5.metaPath(kvAutoGenA)); m == nil || m.Origin != kvOriginAutosave || m.NTokens != 500 {
		t.Errorf("autosave sidecar %+v", m)
	}

	// the newest generation wins at the next start (selftest :7522, :7531), and the next autosave
	// goes to the other one
	w6, cfg6 := r.begin(t, DefaultSwitches)
	if !w6.Eligible || w6.Selected != kvAutoGenA || w6.Origin != kvOriginAutosave {
		t.Fatalf("election: %v %q %q", w6.Eligible, w6.Selected, w6.Origin)
	}
	if ok, _ := r.l.warmRestore(context.Background(), w6, cfg6); !ok || w6.selectAutoGeneration() != kvAutoGenB {
		t.Errorf("restore of the autosave generation: %v, next %q", ok, w6.autoNext)
	}
	lg = r.mock.take()
	if strings.Join(lg, "|") != `POST /slots/0?action=restore {"filename":"slot0.auto.a.kv"}` {
		t.Errorf("restore of the elected generation: %q", lg)
	}
	got = append(got, lg...)

	// the whole log against the recorded one (the tmp save name carries its generation id)
	recorded := []string{
		`POST /slots/0?action=save {"filename":"slot0.kv.tmp.<gen>"}`, // :5912-5926, A-2 (5)
		`POST /slots/0?action=restore {"filename":"slot0.kv"}`,        // :6188-6189
		`POST /slots/0?action=restore {"filename":"slot0.kv"}`,        // :6209
		`POST /slots/0?action=erase`,                                  // :6215
		`POST /slots/0?action=restore {"filename":"slot0.kv"}`,        // :6227
		`POST /slots/0?action=erase`,                                  // the erase that fails (rung 4)
		`GET /slots`,                                                  // :7159
		`POST /slots/0?action=save {"filename":"slot0.auto.a.kv"}`,    // :7169, :7382
		`GET /slots`, // :7320 (no change, no save)
		`GET /slots`, // :7159
		`POST /slots/0?action=save {"filename":"slot0.auto.b.kv"}`, // :7382
		`GET /slots`, // :7159
		`POST /slots/0?action=save {"filename":"slot0.auto.a.kv"}`,    // :7382
		`POST /slots/0?action=restore {"filename":"slot0.auto.a.kv"}`, // :7531
	}
	gen := regexp.MustCompile(`slot0\.kv\.tmp\.\d{8}T\d{9}Z_\d+`)
	var norm []string
	for _, g := range got {
		norm = append(norm, gen.ReplaceAllString(g, "slot0.kv.tmp.<gen>"))
	}
	if strings.Join(norm, "\n") != strings.Join(recorded, "\n") {
		t.Errorf("request log:\n%s\nrecorded:\n%s", strings.Join(norm, "\n"), strings.Join(recorded, "\n"))
	}
	t.Logf("request log (%d requests) equals the recorded log", len(norm))
}
