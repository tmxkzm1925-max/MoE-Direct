package main

import (
	"bytes"
	"strings"
	"testing"

	"moedirect/launcher/internal/core"
)

// S8-PF-01 on the arguments path (F-PF §9 C1): a given -prefetch-k or -prefetch-n reaches the
// launcher's check as given, untrimmed and empty included, so « 8» and «» are refused there
// (core's TestS8PF01) instead of being trimmed or dropped here.
func TestS8PF01ArgumentsRaw(t *testing.T) {
	for _, c := range []struct {
		args []string
		k, n string
		set  bool
	}{
		{[]string{"-prefetch-k", " 8", "-prefetch-n", "2"}, " 8", "2", true},
		{[]string{"-prefetch-k=", "-prefetch-n", "2"}, "", "2", true},
		{[]string{"-prefetch-k", "8", "-prefetch-n", "3 "}, "8", "3 ", true},
		{[]string{"-ctx", "8192"}, "", "", false},
	} {
		var errw bytes.Buffer
		o, _, ok := parse(c.args, &errw)
		if !ok {
			t.Fatalf("%v: %s", c.args, errw.String())
		}
		k, hk := o.arg.settings["prefetch_k"]
		n, hn := o.arg.settings["prefetch_n"]
		if hk != c.set || hn != c.set || k != c.k || n != c.n {
			t.Errorf("%v: K %q (%v) N %q (%v)", c.args, k, hk, n, hn)
		}
	}
}

// S8-PF-16 / S8-PF-11 on the arguments path (GL-21, F-PF §8.5): after the ready line the
// launcher prints the engine's values from its metrics header next to the candidate, the
// header's own names and values (S8-PF-11 compares them exactly).
func TestS8PF16ArgumentsLine(t *testing.T) {
	k, n := int64(8), int64(1)
	cand := &core.PrefetchEcho{EffectivePrefetch: "on", RequestedK: &k, RequestedN: &n}
	eng := core.EnginePrefetch{Read: true, On: true, K: 8, N: 1}
	pre := &core.PreflightResult{Start: core.StartSettings{PrefetchEcho: cand}}
	f := &fakeStarter{final: core.StartEvent{Final: true, Ready: true, Status: core.StatusOK, Address: "http://127.0.0.1:8093", PID: 9,
		Preflight: pre, Engine: &eng}, stop: core.StopEvent{Final: true, Status: core.StatusOK}}
	var out, ew bytes.Buffer
	runArgs(f, argStart{model: "m.gguf"}, &out, &ew, nil)
	want := "prefetch after ready: engine prefetch_on=true prefetch_k=8 prefetch_n=1 prefetch_off_reason= · candidate on requested_k=8 requested_n=1"
	lines := strings.Split(out.String(), "\n")
	at := -1
	for i, l := range lines {
		if strings.Contains(l, " ready on ") {
			at = i
		}
	}
	if at < 0 || at+1 >= len(lines) || !strings.HasSuffix(lines[at+1], want) {
		t.Errorf("output:\n%s", out.String())
	}
	// A start without the read-back (an older core event) prints no such line.
	f = &fakeStarter{final: core.StartEvent{Final: true, Ready: true, Status: core.StatusOK}, stop: core.StopEvent{Final: true, Status: core.StatusOK}}
	out.Reset()
	runArgs(f, argStart{model: "m.gguf"}, &out, &ew, nil)
	if strings.Contains(out.String(), "prefetch after ready") {
		t.Errorf("output:\n%s", out.String())
	}
}
