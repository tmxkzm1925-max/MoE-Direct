// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// The child mode of this test binary: run() as the launcher would, with its arguments from
// GLFX3_ARGS, GL-03's measurement replaced by a marker (the argument-driven path must never
// call it), and the child's own console detection written beside the run for the parent.
func TestMain(m *testing.M) {
	if os.Getenv("GLFX3_CHILD") == "1" {
		var args []string
		if err := json.Unmarshal([]byte(os.Getenv("GLFX3_ARGS")), &args); err != nil {
			os.Exit(90)
		}
		dir := os.Getenv("GLFX3_DIR")
		glyphProbe = func(time.Duration) ui.ProbeResult {
			os.WriteFile(filepath.Join(dir, "probe_called"), []byte("1"), 0o644)
			return ui.NoProbe("test: the measurement was called")
		}
		term := core.DetectTerminal(false)
		b, _ := json.Marshal(term)
		os.WriteFile(filepath.Join(dir, "terminal.json"), b, 0o644)
		os.Args = append([]string{"MoE-Direct-launcher"}, args...)
		os.Exit(run())
	}
	os.Exit(m.Run())
}

type childRun struct {
	code          int
	stdout, errs  string
	term          core.TerminalInfo
	probeCalled   bool
	records       []map[string]any
	logFiles      []string
	took          time.Duration
	dir, stateDir string
}

// runChild starts this test binary as the launcher with its output (and input) redirected,
// so its console detection is the one a redirected run sees.
func runChild(t *testing.T, args ...string) childRun {
	t.Helper()
	dir := t.TempDir()
	state := filepath.Join(dir, "state")
	full := append([]string{"-catalog", filepath.Join(dir, "models.json"), "-root", filepath.Join(dir, "root"), "-state", state,
		"-models-root", filepath.Join(dir, "models")}, args...)
	a, _ := json.Marshal(full)
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(), "GLFX3_CHILD=1", "GLFX3_ARGS="+string(a), "GLFX3_DIR="+dir, "LOCALAPPDATA="+dir)
	var out, errw bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errw
	t0 := time.Now()
	err := cmd.Run()
	r := childRun{stdout: out.String(), errs: errw.String(), took: time.Since(t0), dir: dir, stateDir: state}
	if ee, ok := err.(*exec.ExitError); ok {
		r.code = ee.ExitCode()
	} else if err != nil {
		t.Fatalf("child: %v", err)
	}
	if b, err := os.ReadFile(filepath.Join(dir, "terminal.json")); err == nil {
		json.Unmarshal(b, &r.term)
	}
	_, perr := os.Stat(filepath.Join(dir, "probe_called"))
	r.probeCalled = perr == nil
	r.logFiles, _ = filepath.Glob(filepath.Join(state, "logs", "launcher_*.jsonl"))
	for _, f := range r.logFiles {
		fh, err := os.Open(f)
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var rec struct {
				Kind string
				Data map[string]any
			}
			if json.Unmarshal(sc.Bytes(), &rec) == nil && rec.Kind == core.StartLogGlyphProbe {
				r.records = append(r.records, rec.Data)
			}
		}
		fh.Close()
	}
	return r
}

// TestS8LA07ArgPath (S8-LA-07 family; F-LS 9.4, conflict row 7.10; GL-FX3): the argument-driven
// path with -model and its output not a console writes exactly one GLYPH_PROBE start-log record
// without measuring - glyph set fallback, reason «no console: » and the detection's reason,
// wait 0 - and writes no probe glyph and no cursor query; a stop with -model does the same;
// without -model the run ends at the guidance line, exit 3, before the start log opens.
func TestS8LA07ArgPath(t *testing.T) {
	queries := []string{"\x1b[6n", "\x1b[2K"}
	for _, c := range []struct {
		name string
		args []string
	}{
		{"start with -model", []string{"-model", "m.gguf"}},
		{"stop with -model", []string{"-stop", "-model", "m.gguf"}},
	} {
		r := runChild(t, append(c.args[:len(c.args)-1:len(c.args)-1], filepath.Join(t.TempDir(), c.args[len(c.args)-1]))...)
		if r.term.Interactive {
			t.Fatalf("%s: the child's console detection says interactive (%+v)", c.name, r.term)
		}
		if len(r.records) != 1 {
			t.Fatalf("%s: %d GLYPH_PROBE records in %v (exit %d)\nstderr: %s", c.name, len(r.records), r.logFiles, r.code, r.errs)
		}
		g := r.records[0]
		if g["glyph_set"] != "fallback" || g["reason"] != "no console: "+r.term.Reason || g["wait_ms"] != float64(0) ||
			g["elapsed_ms"] != float64(0) || g["answers"] != float64(0) || g["input_mode"] != float64(0) {
			t.Errorf("%s: record %v (detection %+v)", c.name, g, r.term)
		}
		if r.probeCalled {
			t.Errorf("%s: the argument-driven path called the glyph measurement", c.name)
		}
		for _, q := range queries { // the probe writes each glyph between ESC [2K and ESC [6n (GLP step 4)
			if strings.Contains(r.stdout, q) || strings.Contains(r.errs, q) {
				t.Errorf("%s: the output holds the probe text %q", c.name, q)
			}
		}
		t.Logf("%s: exit %d in %v · record glyph_set=%v reason=%q wait_ms=%v · stderr %q", c.name, r.code, r.took.Round(time.Millisecond),
			g["glyph_set"], g["reason"], g["wait_ms"], strings.TrimSpace(r.errs))
	}

	// without -model: the guidance line, fail_model_path, exit 3, no start log
	r := runChild(t)
	if r.code != 3 || !strings.Contains(r.errs, "status=fail_model_path") || len(r.logFiles) != 0 || len(r.records) != 0 || r.probeCalled {
		t.Errorf("no model: exit %d, logs %v, records %d, probe %v\nstderr: %s", r.code, r.logFiles, len(r.records), r.probeCalled, r.errs)
	}
	if _, err := os.Stat(r.stateDir); err == nil {
		t.Errorf("no model: the state folder was created")
	}
	t.Logf("no model: exit %d · stderr %q", r.code, strings.TrimSpace(r.errs))
}
