//go:build windows

package core

// GL-15 (version 8): the live part of warm start. The start and stop paths of this package
// against the fake server (the test binary itself, «slots» mode of stopwarmup_test.go) inside
// the launcher's job object on a free loopback port: a first start is cold and its stop saves
// slot0.kv under the cache-save bar (TimedCacheSave, every track on StopEvent.Save, GL-07
// finding 1); the next start of a new launcher on the same state is eligible, restores the slot
// after ready and skips the warmup (WARMSTART A-3), and its stop saves again. The fake server's
// request log (its stderr) is compared with the recorded PowerShell log as in TestS8LA33.

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// writeBundleManifest lists every file of the bundle root (a release bundle: the engine bundle
// hash is one of the bindings, WARMSTART A-4).
func writeBundleManifest(t *testing.T, root string) {
	t.Helper()
	var files []map[string]any
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		h, herr := fileSha256(p)
		if herr != nil {
			t.Fatal(herr)
		}
		files = append(files, map[string]any{"path": rel, "sha256": h})
		return nil
	})
	writeJSON(t, filepath.Join(root, BundleManifestName), map[string]any{"bundle_manifest_version": 1, "files": files})
}

// slotLines is the fake server's request log in its stderr, the generation id masked.
func slotLines(t *testing.T, errLog string) []string {
	t.Helper()
	b, _ := os.ReadFile(errLog)
	gen := regexp.MustCompile(`slot0\.kv\.tmp\.\d{8}T\d{9}Z_\d+`)
	var out []string
	for _, l := range strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n") {
		if strings.HasPrefix(l, "[slots] ") {
			out = append(out, gen.ReplaceAllString(strings.TrimPrefix(l, "[slots] "), "slot0.kv.tmp.<gen>"))
		}
	}
	return out
}

func warmTexts(evs []StartEvent) []string {
	var out []string
	for _, e := range evs {
		if e.Warm != "" && (len(out) == 0 || out[len(out)-1] != e.Warm) {
			out = append(out, e.Warm)
		}
	}
	return out
}

func stopAll(t *testing.T, l *Launcher) (StopEvent, []StopEvent) {
	t.Helper()
	ch, err := l.Stop()
	if err != nil {
		t.Fatalf("stop: %v", err)
	}
	var all []StopEvent
	var final StopEvent
	for ev := range ch {
		all = append(all, ev)
		if ev.Final {
			final = ev
		}
	}
	return final, all
}

func lastSave(evs []StopEvent) *CacheSaveTrack {
	var s *CacheSaveTrack
	for _, e := range evs {
		if e.Save != nil {
			s = e.Save
		}
	}
	return s
}

func TestGL15WarmStartLive(t *testing.T) {
	r := newLiveRig(t, "slots answer break_exit0")
	writeBundleManifest(t, r.f.root)

	// the first start: cold, the slot save path passed, then the save at the stop
	final, evs := r.start(t, r.f.l)
	if !final.Ready {
		t.Fatalf("first start: %v %s", final.Err, final.Status)
	}
	if w := warmTexts(evs); strings.Join(w, ",") != "cold(sidecar_missing)" {
		t.Errorf("first start warm texts %q", w)
	}
	dir := filepath.Join(r.f.stateDir, "kv", final.Preflight.Profile.ID)
	if v, ok := argvValue(final.Preflight.Config.Argv, argSlotSave); !ok || v != dir+`\` {
		t.Errorf("--slot-save-path %q %v", v, ok)
	}
	sf, sevs := stopAll(t, r.f.l)
	if sf.Status != StatusOK {
		t.Errorf("first stop: %s %s", sf.Status, sf.Reason)
	}
	if s := lastSave(sevs); s == nil || !s.Done || s.Saved != 128 {
		t.Errorf("first stop's cache-save bar %+v", s)
	}
	for _, e := range sevs {
		if e.SaveErr != "" || e.NoSave != "" {
			t.Errorf("first stop: save error %q, no save %q", e.SaveErr, e.NoSave)
		}
	}
	if !fileExists(filepath.Join(dir, kvCanonicalData)) || !fileExists(filepath.Join(dir, kvCanonicalMeta)) {
		t.Fatal("no slot0.kv pair after the first stop")
	}
	log1 := slotLines(t, final.ErrLog)

	// the next start, a new launcher on the same state: eligible, restored, no warmup
	l2, err := New(Options{CatalogPath: r.f.catalogPath, BundleRoot: r.f.root, StateDir: r.f.stateDir})
	if err != nil {
		t.Fatal(err)
	}
	l2.os.portOwners, l2.os.console = osPortOwners, osConsoleAvailable
	t.Cleanup(func() {
		if l2.activeSession() {
			if ch, err := l2.Stop(); err == nil {
				for range ch {
				}
			}
		}
	})
	final2, evs2 := r.start(t, l2)
	if !final2.Ready {
		t.Fatalf("second start: %v %s", final2.Err, final2.Status)
	}
	if w := warmTexts(evs2); strings.Join(w, ",") != "eligible,restored(128 tokens)" {
		t.Errorf("second start warm texts %q", w)
	}
	if final2.Warmup == nil || !final2.Warmup.Skipped || final2.Warmup.SkipReason != WarmupSkipRestored || final2.Warmup.Reason != warmupRestoredReason {
		t.Errorf("the warmup after a restore %+v", final2.Warmup)
	}
	sf2, sevs2 := stopAll(t, l2)
	if s := lastSave(sevs2); sf2.Status != StatusOK || s == nil || !s.Done || s.Saved != 128 || s.Tokens != 128 {
		t.Errorf("second stop %s, its bar %+v (the restored tokens are this save's tokens)", sf2.Status, s)
	}
	log2 := slotLines(t, final2.ErrLog)

	want1 := []string{`POST /slots/0?action=save {"filename":"slot0.kv.tmp.<gen>"}`}
	want2 := []string{`POST /slots/0?action=restore {"filename":"slot0.kv"}`, `POST /slots/0?action=save {"filename":"slot0.kv.tmp.<gen>"}`}
	if strings.Join(log1, "|") != strings.Join(want1, "|") || strings.Join(log2, "|") != strings.Join(want2, "|") {
		t.Errorf("request logs\n%q\n%q\nrecorded\n%q\n%q", log1, log2, want1, want2)
	}

	// the switch off (OQ-6 (a)): no slot save path, no request, the stop stores nothing
	req := r.req()
	req.Switches = &Switches{WarmStart: false, Autosave: true}
	ctx3, cancel3 := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel3()
	ch3, err := l2.Start(ctx3, req)
	if err != nil {
		t.Fatal(err)
	}
	var final3 StartEvent
	var evs3 []StartEvent
	for ev := range ch3 {
		evs3 = append(evs3, ev)
		if ev.Final {
			final3 = ev
		}
	}
	if !final3.Ready || strings.Join(warmTexts(evs3), ",") != "off(user)" {
		t.Fatalf("switched-off start: %v %q", final3.Ready, warmTexts(evs3))
	}
	if _, ok := argvValue(final3.Preflight.Config.Argv, argSlotSave); ok {
		t.Error("a switched-off start passed --slot-save-path")
	}
	sf3, sevs3 := stopAll(t, l2)
	if sf3.Status != StatusOK || lastSave(sevs3) != nil || len(sevs3) == 0 || sevs3[0].NoSave != "warm start is switched off" {
		t.Errorf("switched-off stop: %s, first event %+v", sf3.Status, sevs3[0])
	}
	if lg := slotLines(t, final3.ErrLog); len(lg) != 0 {
		t.Errorf("switched-off run sent %q", lg)
	}

	// close now (ctrl+c twice): the restore runs, the stop stores nothing
	final4, _ := r.start(t, l2)
	ch4, err := l2.StopNoSave()
	if err != nil {
		t.Fatal(err)
	}
	var sevs4 []StopEvent
	for ev := range ch4 {
		sevs4 = append(sevs4, ev)
	}
	if lastSave(sevs4) != nil || sevs4[0].NoSave != noSaveClose {
		t.Errorf("close now: %+v", sevs4[0])
	}
	if lg := slotLines(t, final4.ErrLog); strings.Join(lg, "|") != `POST /slots/0?action=restore {"filename":"slot0.kv"}` {
		t.Errorf("close-now run sent %q", lg)
	}

	es, _ := os.ReadDir(dir)
	var names []string
	for _, e := range es {
		names = append(names, e.Name())
	}
	if strings.Join(names, ",") != "slot0.kv,slot0.kv.meta.json" {
		t.Errorf("the kv dir after the stops holds %v (the stale pair and the tmp go at teardown)", names)
	}
}
