package wire

// UX-R14 (version 8; finding 22): the setup progress follows the repacker's check: its name
// turns to «check» at the check's first line, its counts are the check's pairs (thousands
// separators read), never the repack's layers.

import (
	"testing"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

func TestUXR14SetupStep(t *testing.T) {
	sp := ui.SetupProgress{}
	for _, c := range []struct {
		e           core.RepackEvent
		name        string
		done, total ui.Val
	}{
		{core.RepackEvent{Stage: core.RepackStageRepack, Line: "[repack] moe layer 35 (36/36) done - 1.0 MB", Fraction: 1}, "repack", ui.K(36), ui.K(36)},
		{core.RepackEvent{Stage: core.RepackStageVerify, Line: "[verify] checking 13,824 record pairs (re-reading source shards and experts.bin)...", Fraction: -1}, "check", ui.U, ui.U},
		{core.RepackEvent{Stage: core.RepackStageVerify, Line: "[verify] 6,912/13,824 pairs (50%)", Fraction: 0.5}, "check", ui.K(6912), ui.K(13824)},
		{core.RepackEvent{Stage: core.RepackStageVerify, Line: "[verify] 13,824/13,824 PASS", Fraction: -1}, "check", ui.K(6912), ui.K(13824)},
		{core.RepackEvent{Stage: core.RepackStageDone, Line: "setup done in 2m0s", Fraction: 1}, "done", ui.K(6912), ui.K(13824)},
	} {
		setupStep(&sp, c.e, 0, true)
		if sp.Name != c.name || sp.Done != c.done || sp.Total != c.total {
			t.Fatalf("after %q: %q %v/%v, want %q %v/%v", c.e.Line, sp.Name, sp.Done, sp.Total, c.name, c.done, c.total)
		}
	}
}
