package core

// UX-R14 (version 8; finding 22): the repacker's check prints its counts with thousands
// separators (bench/repack/repack_experts.py:2174, format(n, ',')); every «[verify]» line is
// the check's, so the setup page sees the check begin, move and end.

import "testing"

func TestUXR14VerifyLines(t *testing.T) {
	for _, c := range []struct {
		line  string
		stage RepackStage
		frac  float64
	}{
		{"[repack] moe layer 35 (36/36) done - 1.0 MB (this layer), cumulative 1.0 MB/1.0s, average 1.0 MB/s", RepackStageRepack, 1},
		{"[manifest] sealing manifest.json (4608 records, reference_lock=gpt-oss-120b)...", RepackStageRepack, -1},
		{"[verify] checking 13,824 record pairs (re-reading source shards and experts.bin)...", RepackStageVerify, -1},
		{"[verify] 6,912/13,824 pairs (50%)", RepackStageVerify, 0.5},
		{"[verify] 4/8 pairs (50%)", RepackStageVerify, 0.5},
		{"[verify] 13,824/13,824 PASS", RepackStageVerify, -1},
	} {
		if st, f := repackFraction(c.line); st != c.stage || f != c.frac {
			t.Fatalf("repackFraction(%q) = %s %v, want %s %v", c.line, st, f, c.stage, c.frac)
		}
	}
}
