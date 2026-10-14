package core

// GL-08 (version 8): core's part of S8-LA-29 that needs no Windows API (plan
// reviews/_plan_8_STAGE1_v8-plansplit.md at d1dbe526f; F-LS edition
// reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, sections 9.3 and 9.4).

import (
	"context"
	"testing"
)

// S8-LA-29, GL-08's part (W7 duty 23 and conflict row 15): a value outside its range given to
// the argument-driven mode ends the run fail_custom_args, exit 3, before anything starts; the
// stop action checks the same values. It runs as a subtest of TestS8LA29, which GL-16 created (gl16_test.go).
func la29GL08(t *testing.T) {
	f := newFixture(t, true)
	for k, v := range map[string]string{"ctx": "999999999", "port": "80", "budget_mb": "x", "qd": "0"} {
		req := Request{ModelPath: f.modelPath, Settings: map[string]string{k: v}}
		if _, err := f.l.Preflight(context.Background(), req); StatusOf(err) != StatusFailCustomArgs || ExitCodeOf(err) != 3 {
			t.Errorf("%s=%s: %v", k, v, err)
		}
		if err := f.l.StopBeforeStart(context.Background(), req); StatusOf(err) != StatusFailCustomArgs {
			t.Errorf("stop action %s=%s: %v", k, v, err)
		}
	}
}
