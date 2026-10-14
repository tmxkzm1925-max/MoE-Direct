package core

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// TestS8LA30 (S8-LA-30, the log header): the first record of a diagnostic file is the
// LAUNCH header and carries the launcher's name and the release version v0.3.2; the
// record the caller wrote follows it, and a later record adds no second header.
func TestS8LA30(t *testing.T) {
	dir := t.TempDir()
	d := newDiagLog(dir, func() time.Time { return time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC) })
	d.write("FIRST", map[string]any{"n": 1})
	d.write("SECOND", nil)
	f, err := os.Open(d.Path())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	type rec struct {
		Kind string         `json:"kind"`
		Data map[string]any `json:"data"`
	}
	var recs []rec
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r rec
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatalf("record %d: %v", len(recs)+1, err)
		}
		recs = append(recs, r)
	}
	if len(recs) != 3 || recs[0].Kind != "LAUNCH" || recs[1].Kind != "FIRST" || recs[2].Kind != "SECOND" {
		t.Fatalf("records = %+v, want LAUNCH, FIRST, SECOND", recs)
	}
	h := recs[0].Data
	if h["version"] != "v0.3.2" { // the release version; internal/version's TestS8LA30 ties it to LauncherVersion
		t.Fatalf("log header version = %v, want the release version v0.3.2", h["version"])
	}
	if h["launcher"] != "MoE-Direct-launcher" {
		t.Fatalf("log header launcher = %v, want MoE-Direct-launcher", h["launcher"])
	}
	if pid, ok := h["pid"].(float64); !ok || int(pid) != os.Getpid() {
		t.Fatalf("log header pid = %v, want %d", h["pid"], os.Getpid())
	}
}
