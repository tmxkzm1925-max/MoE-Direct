package core

import (
	"strings"
	"testing"
)

func TestParseTyped(t *testing.T) {
	f := newFixture(t, true)
	rows, err := f.l.Settings("fixture-profile")
	if err != nil {
		t.Fatal(err)
	}
	row := map[string]SettingRow{}
	for _, r := range rows {
		row[r.Key] = r
	}
	ok := []struct{ key, in, want string }{
		{"ctx", "64k", "65536"},
		{"ctx", "65536", "65536"},
		{"ctx", " 64K ", "65536"},
		{"ctx", "2048", "2048"},
		{"port", "8111", "8111"},
		{"threads", "16", "16"},
		{"qd", "8", "8"},
		{"qd", "auto", "auto"},
		{"budget_mb", "14.5", "14848"},
		{"budget_mb", "14.5G", "14848"},
		{"budget_mb", "16 gb", "16384"},
		{"budget_mb", "20000m", "20000"},
		{"budget_mb", "20000MB", "20000"},
		{"budget_mb", "Auto", "auto"},
	}
	for _, c := range ok {
		v, note, err := ParseTyped(row[c.key], c.in)
		if err != nil || v != c.want || note != "" {
			t.Errorf("%s %q: %q %q %v want %q", c.key, c.in, v, note, err, c.want)
		}
	}
	bad := []struct{ key, in, reason string }{
		{"ctx", "1k", "outside its range · allowed 2k ~ 256k"},
		{"ctx", "64kk", "not a whole number"},
		{"ctx", "1.5k", "not a whole number"},
		{"port", "80", "outside its range · allowed 1024 ~ 65535"},
		{"port", "auto", "port has no auto value"},
		{"threads", "65", "outside its range · allowed 1 ~ 64"},
		{"threads", "8.5", "not a whole number"},
		{"qd", "64", "outside its range · allowed 1 ~ 63"},
		{"budget_mb", "3", "outside its range · allowed 4 GB ~ 64 GB"},
		{"budget_mb", "70", "outside its range · allowed 4 GB ~ 64 GB"},
		{"budget_mb", "100.5m", "MB must be a whole number"},
		{"budget_mb", "lots", "not a number"},
		{"prefetch", "init", "prefetch is chosen with ←→, not typed"},
		{"ctx", "  ", "empty"},
	}
	for _, c := range bad {
		v, _, err := ParseTyped(row[c.key], c.in)
		if err == nil || v != "" || !strings.HasPrefix(err.Error(), c.reason) {
			t.Errorf("%s %q: %q %v want %q", c.key, c.in, v, err, c.reason)
		}
	}
	// The warmup file path is kept as text; the note says what it does in the draft.
	v, note, err := ParseTyped(row["warmup"], `file:"C:\prompts\warm up.txt"`)
	if err != nil || v != `file:C:\prompts\warm up.txt` || note != NoteWarmupNotPorted {
		t.Errorf("warmup %q %q %v", v, note, err)
	}
	if v, _, err := ParseTyped(row["warmup"], `D:\w.txt`); err != nil || v != `file:D:\w.txt` {
		t.Errorf("warmup without prefix %q %v", v, err)
	}
	if _, _, err := ParseTyped(row["warmup"], "file:"); err == nil {
		t.Error("empty warmup path accepted")
	}
	// An unbounded port or QD row still gets the PowerShell clamps.
	if _, _, err := ParseTyped(SettingRow{Key: "port"}, "80"); err == nil {
		t.Error("port 80 accepted without bounds")
	}
	if _, _, err := ParseTyped(SettingRow{Key: "qd"}, "0"); err == nil {
		t.Error("qd 0 accepted without bounds")
	}
	// Through the launcher, and a typed value goes through the preflight as a request value.
	if v, _, err := f.l.ParseSetting("fixture-profile", "ctx", "8k"); err != nil || v != "8192" {
		t.Errorf("ParseSetting %q %v", v, err)
	}
}
