package main

// GL-12 (version 8): the argument-driven part of S8-LA-26 (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition 4.7, 9.3; CF 17): no argument
// sets the bind address or a network option; one given is refused as a usage error (exit 3).

import (
	"bytes"
	"strings"
	"testing"
)

func TestS8LA26(t *testing.T) {
	for _, n := range flagNames() {
		for _, w := range []string{"host", "network", "bind", "listen", "addr"} {
			if strings.Contains(n, w) || n == "lan" {
				t.Errorf("flag %q", n)
			}
		}
	}
	for _, args := range [][]string{
		{"-model", "m.gguf", "-host", "0.0.0.0"},
		{"-model", "m.gguf", "-network", "lan"},
		{"-model", "m.gguf", "-bind", "0.0.0.0"},
	} {
		var ew bytes.Buffer
		o, code, run := parse(args, &ew)
		if o != nil || run || code != 3 || !strings.Contains(ew.String(), "status=fail_custom_args") {
			t.Errorf("%v: options %v, code %d, run %v, stderr %q", args, o, code, run, ew.String())
		}
	}
}
