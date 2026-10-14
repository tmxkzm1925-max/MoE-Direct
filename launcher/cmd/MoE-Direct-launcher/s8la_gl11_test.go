package main

// GL-11 (version 8): the argument-driven part of S8-LA-25 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition
// reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, conflict row 18-1 and 9.3): `-smoke`
// prints the check run's items, one line each with its result, and a failure keeps exit code 6.

import (
	"bytes"
	"strings"
	"testing"

	"moedirect/launcher/internal/core"
)

func TestS8LA25(t *testing.T) {
	items := []core.SmokeItem{
		{N: 1, Name: "/health 200", OK: true}, {N: 2, Name: "/v1/models", OK: true},
		{N: 3, Name: "chat completion stream+non-stream", OK: true}, {N: 4, Name: "cancel + slot reclaim", OK: true},
		{N: 5, Name: "verify PASS consumption", OK: true, Detail: "engine seal line x1, slots 6/6; launcher gate manifest abababababab"},
		{N: 6, Name: "loopback-only binding", OK: true, Detail: "127.0.0.1"},
		{N: 7, Name: "built-in web UI", OK: true, Skipped: true, Detail: "API-only bundle"},
	}
	itemLines := func(out string) []string {
		var got []string
		for _, ln := range strings.Split(out, "\n") {
			if strings.HasPrefix(ln, progress+" check run [") {
				got = append(got, strings.TrimPrefix(ln, progress+" check run "))
			}
		}
		return got
	}

	// A pass: seven item lines, the summary, status ok_smoke after the stop.
	pass := &core.SmokeResult{OK: true, Items: items}
	f := &fakeStarter{final: core.StartEvent{Final: true, Ready: true, Status: core.StatusOK, Address: "http://127.0.0.1:8093", PID: 9, Smoke: pass},
		stop: core.StopEvent{Final: true, Status: core.StatusOK}}
	var out, ew bytes.Buffer
	if got := runArgs(f, argStart{model: "m.gguf", smoke: true}, &out, &ew, nil); got != 0 ||
		!strings.HasSuffix(strings.TrimSpace(ew.String()), "status=ok_smoke") {
		t.Errorf("pass: code %d, stderr %q", got, ew.String())
	}
	got := itemLines(out.String())
	if len(got) != 7 {
		t.Fatalf("pass: %d item lines:\n%s", len(got), out.String())
	}
	for i, it := range items {
		if got[i] != it.Line() {
			t.Errorf("line %d %q, want %q", i, got[i], it.Line())
		}
	}
	if got[6] != "[7] built-in web UI                  SKIP (API-only bundle)" || !strings.Contains(out.String(), "check run: smoke PASS (7 items") {
		t.Errorf("pass output:\n%s", out.String())
	}

	// A failure: the items printed, exit code 6, status fail_smoke.
	failItems := append([]core.SmokeItem(nil), items...)
	failItems[4].OK, failItems[4].Detail = false, "no seal success line"
	fail := failed(core.StatusFailSmoke, "smoke assertions failed: 5:verify-consume")
	fail.Smoke = &core.SmokeResult{Items: failItems, Failures: []string{"5:verify-consume"}}
	f = &fakeStarter{final: fail}
	out.Reset()
	ew.Reset()
	if code := runArgs(f, argStart{model: "m.gguf", smoke: true}, &out, &ew, nil); code != 6 ||
		!strings.HasSuffix(strings.TrimSpace(ew.String()), "status=fail_smoke") || f.stopped {
		t.Errorf("fail: code %d, stderr %q", code, ew.String())
	}
	got = itemLines(out.String())
	if len(got) != 7 || got[4] != "[5] verify PASS consumption          FAIL (no seal success line)" {
		t.Errorf("fail output:\n%s", out.String())
	}
}
