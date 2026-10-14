// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package wire

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// TestS8LA36 (plan S8-LA-36; GL-17), the log side: the diagnostic and server logs are in
// the log folder under %LOCALAPPDATA%\MoE-Direct\ (U-03, L:4264; edition 8.1, 8.2, 12.6);
// the log page's list merges the launcher's events and the server's lines in clock order, a
// server line's clock time being the server's start plus the engine's stamp (review item
// 5); a line's class comes from the engine's letter first, and the notice marker or the
// automatic-adjustment wording makes a notice whatever the letter (items 18, 29 (2),
// L:4261). The lists and the language are the UI side (internal/ui TestS8LA36).
func TestS8LA36(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALAPPDATA", root)
	state := core.LauncherStateDir()
	if state != filepath.Join(root, "MoE-Direct") {
		t.Fatalf("state root %q, want %q", state, filepath.Join(root, "MoE-Direct"))
	}
	l, err := core.New(core.Options{CatalogPath: filepath.Join(root, "models.json"), BundleRoot: t.TempDir(), StateDir: state})
	if err != nil {
		t.Fatal(err)
	}
	logs := filepath.Join(state, "logs")
	if l.LogDir() != logs || core.StateLogsDir != "logs" {
		t.Fatalf("log folder %q, want %q", l.LogDir(), logs)
	}

	// the diagnostic log: its file appears in the log folder with the LAUNCH header first
	l.StartLog("LA36_PROBE", map[string]any{"x": 1})
	diag, _ := filepath.Glob(filepath.Join(logs, "launcher_*.jsonl"))
	if len(diag) != 1 {
		t.Fatalf("diagnostic log files in %s: %v", logs, diag)
	}
	f, err := os.Open(diag[0])
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(f)
	var kinds []string
	for sc.Scan() {
		for _, k := range []string{`"kind":"LAUNCH"`, `"kind":"LA36_PROBE"`} {
			if strings.Contains(sc.Text(), k) {
				kinds = append(kinds, k)
			}
		}
	}
	f.Close()
	if strings.Join(kinds, ",") != `"kind":"LAUNCH","kind":"LA36_PROBE"` {
		t.Fatalf("diagnostic records %v", kinds)
	}

	// the server logs: both files of a start are in the log folder
	outLog, errLog := l.ServerLogPaths("20261012T000000Z")
	if filepath.Dir(outLog) != logs || filepath.Dir(errLog) != logs {
		t.Fatalf("server logs %q %q not in %q", outLog, errLog, logs)
	}
	lines := []string{
		"0.00.089.177 I cmn  common_param: verbosity = 3",
		"[moe-direct] NOTICE: budget lowered to fit",
		"0.01.000.000 I [moe-direct] mode=virtual raises io_bounce_copy_qd 2 -> 14 (record fan-out shape floor)",
		"0.01.500.000 W [moe-direct] mode=virtual raises io_bounce_slots 9 -> 24 (record fan-out shape floor)",
		"[moe-direct] ERROR: [moe-direct] mode=virtual raises io_bounce_slots 9 -> 24 (record fan-out shape floor)",
		"0.02.578.274 I srv  llama_server: listening on http://127.0.0.1:8093",
		"1.04.811.654 W srv          stop: cancel task, id_task = 66",
		"1.05.000.000 E srv  something failed",
	}
	if err := os.MkdirAll(logs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(errLog, []byte(strings.Join(lines, "\r\n")+"\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// the log page's list: the folder's files, the server's log and the diagnostic log among them
	names := strings.Join(l.LogFiles(), "|")
	if !strings.Contains(names, filepath.Base(errLog)) || !strings.Contains(names, filepath.Base(diag[0])) {
		t.Fatalf("log files %q", names)
	}
	b, err := os.ReadFile(errLog)
	if err != nil {
		t.Fatal(err)
	}
	spawn := time.Date(2026, 10, 12, 9, 0, 0, 0, time.UTC)
	server := core.ParseServerLines(strings.Split(strings.TrimRight(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n"), "\n"), spawn)
	events := []ui.LogLine{
		{At: spawn.Add(-time.Second), Level: ui.LogInfo, Text: "start"},
		{At: spawn.Add(2 * time.Second), Level: ui.LogOK, Text: "ready"},
		{At: spawn.Add(64 * time.Second), Level: ui.LogInfo, Text: "request"},
		{At: spawn.Add(2 * time.Minute), Level: ui.LogInfo, Text: "stop"},
	}
	got := mergeLogs(events, server, 0)
	want := []struct {
		text  string
		level ui.LogLevel
		at    time.Duration // after the spawn; a line without a stamp takes the previous line's time (edition 6.5)
	}{
		{"start", ui.LogInfo, -time.Second},
		{"cmn  common_param: verbosity = 3", ui.LogInfo, 89177 * time.Microsecond},
		{"[moe-direct] NOTICE: budget lowered to fit", ui.LogNotice, 89177 * time.Microsecond},
		{"[moe-direct] mode=virtual raises io_bounce_copy_qd 2 -> 14 (record fan-out shape floor)", ui.LogNotice, time.Second},
		{"[moe-direct] mode=virtual raises io_bounce_slots 9 -> 24 (record fan-out shape floor)", ui.LogNotice, 1500 * time.Millisecond},
		{"[moe-direct] ERROR: [moe-direct] mode=virtual raises io_bounce_slots 9 -> 24 (record fan-out shape floor)", ui.LogNotice, 1500 * time.Millisecond},
		{"ready", ui.LogOK, 2 * time.Second},
		{"srv  llama_server: listening on http://127.0.0.1:8093", ui.LogOK, 2578274 * time.Microsecond},
		{"request", ui.LogInfo, 64 * time.Second},
		{"srv          stop: cancel task, id_task = 66", ui.LogWarn, 64811654 * time.Microsecond},
		{"srv  something failed", ui.LogError, 65 * time.Second},
		{"stop", ui.LogInfo, 2 * time.Minute},
	}
	if len(got) != len(want) {
		for _, g := range got {
			t.Logf("%v %d %q", g.At.Sub(spawn), g.Level, g.Text)
		}
		t.Fatalf("merged %d lines, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i].Text != w.text || got[i].Level != w.level || !got[i].At.Equal(spawn.Add(w.at)) {
			t.Fatalf("line %d: %v %d %q, want %v %d %q", i, got[i].At.Sub(spawn), got[i].Level, got[i].Text, w.at, w.level, w.text)
		}
	}
	t.Logf("log folder %s holds the diagnostic log and the server logs; %d lines merged in clock order; 4 notices whatever the letter", logs, len(got))
}
