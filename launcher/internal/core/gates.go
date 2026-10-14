package core

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Conflict 16 of reviews/_gol_conflicts_8_user_20261010.md: the format gates run at every start,
// per repack mode, as the PowerShell gates do - Assert-VerifyGate (PS:5268-5356, 7 items) for a
// packed output and Assert-VirtualPlanGate (PS:5453-5588, 8 items) for a virtual one. Both read
// small JSON files and hash manifest.json; neither reads experts.bin. Nothing is cached across
// calls (PS:14190-14199 runs the gate on every start).

// Manifest modes (MANIFEST_MODE_* PS:749-752).
const (
	ManifestModeBin          = "bin"
	ManifestModeVirtual      = "virtual"
	ManifestModeUnrecognized = "unrecognized"

	manifestModeUnrecognizedReason = "mode: unrecognized manifest schema/mode"
)

// The item counts of the two gates.
const (
	VerifyGateItems      = 7
	VirtualPlanGateItems = 8
)

// The artifact sets (PARTIAL_DELETE_SET PS:736, VIRTUAL_PARTIAL_DELETE_SET PS:761).
var (
	PackedArtifactSet  = []string{"experts.bin.partial", "experts.bin", "manifest.json", "verify_report.json"}
	VirtualArtifactSet = []string{"plan_report.json", "manifest.json.partial"}
)

// GateState is what a format check of one mode found.
type GateState uint8

const (
	GateNeedsRepack GateState = iota // no usable artifacts: the setup (repack) has to run
	GatePass                         // the artifacts passed every item
	GateFail                         // the artifacts are present and an item failed
)

func (s GateState) String() string {
	return [...]string{"needs repack", "pass", "fail"}[s]
}

// GateResult is one gate run (or the finding that no artifacts exist for the mode).
type GateResult struct {
	Mode        string // packed | virtual
	OutputDir   string
	State       GateState
	Items       int    // 7 (packed) or 8 (virtual)
	Passed      int    // items passed (Items on a pass; the items before the failed one on a fail)
	FailedItem  int    // 0 unless State is GateFail
	Reason      string // the PowerShell reason text ("gate N: ..." / "vgate N: ...", or why a repack is needed)
	Partial     bool   // an interrupted setup was found (experts.bin.partial, or a virtual partial file)
	ManifestSha string // the real manifest.json sha256 (on a pass)
	CheckedAt   string // the report's checked_at (on a pass)
	Count       int64  // pairs (packed) or records (virtual), on a pass
	At          time.Time
}

// Line is the PowerShell pass line, or the failure reason (the log line of conflict 16).
func (g GateResult) Line() string {
	switch g.State {
	case GatePass:
		if g.Mode == RepackVirtual {
			return fmt.Sprintf("virtual plan gate PASS (8/8) - mode=virtual, records %d, manifest %s", g.Count, short12(g.ManifestSha))
		}
		return fmt.Sprintf("verify gate PASS (7/7) - pairs %d, manifest %s", g.Count, short12(g.ManifestSha))
	case GateFail:
		return fmt.Sprintf("%s gate FAIL (item %d of %d): %s", g.Mode, g.FailedItem, g.Items, g.Reason)
	}
	return fmt.Sprintf("%s: repack needed (%s)", g.Mode, g.Reason)
}

func short12(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// GateError is a start refused by a format gate (fail_gate_verify, exit 4); the gate result
// rides along for the log line and the «Enter 다시 리팩» offer of conflict 16.
type GateError struct {
	*LaunchError
	Gate GateResult
}

// Unwrap lets AsLaunchError find the status.
func (e *GateError) Unwrap() error { return e.LaunchError }

// absence is Get-FileAbsenceState (PS:5076-5098): present, absent (provably) or unknown.
type absence uint8

const (
	absPresent absence = iota
	absAbsent
	absUnknown
)

func fileAbsence(p string) (absence, string) {
	_, err := os.Lstat(p)
	if err == nil {
		return absPresent, ""
	}
	if errors.Is(err, os.ErrNotExist) {
		return absAbsent, ""
	}
	return absUnknown, "absence not provable - " + err.Error()
}

// gateStop is one failed item: Stop-Launcher 'fail_gate_verify' with the item's text.
type gateStop struct {
	item   int
	reason string
}

// ManifestMode is Get-ManifestMode (PS:5381-5420): "bin" for schema 2.0 without a mode key,
// "virtual" for schema 3.0 with mode "virtual", else "unrecognized" with the reason.
func ManifestMode(path string) (mode, reason string) {
	v, err := readJSONFileStrict(path)
	if err != nil {
		return ManifestModeUnrecognized, err.Error()
	}
	o, ok := asObj(v)
	if !ok {
		return ManifestModeUnrecognized, "manifest root is not a JSON object"
	}
	if !has(o, "schema_version") {
		return ManifestModeUnrecognized, "schema_version key missing"
	}
	sv, ok := jNonEmptyStr(o["schema_version"])
	if !ok {
		return ManifestModeUnrecognized, "schema_version is not a JSON string"
	}
	hasMode := has(o, "mode")
	switch sv {
	case "2.0":
		if hasMode {
			return ManifestModeUnrecognized, "schema_version 2.0 carries a mode key"
		}
		return ManifestModeBin, ""
	case "3.0":
		if !hasMode {
			return ManifestModeUnrecognized, "schema_version 3.0 has no mode key"
		}
		mv, ok := jNonEmptyStr(o["mode"])
		if !ok {
			return ManifestModeUnrecognized, "mode is not a JSON string"
		}
		if mv == ManifestModeVirtual {
			return ManifestModeVirtual, ""
		}
		return ManifestModeUnrecognized, "mode is '" + mv + "'"
	}
	return ManifestModeUnrecognized, "schema_version is '" + sv + "'"
}

// checkReferenceLocks is gate 6 / vgate 7: both locks present, well formed, and equal to the
// selected profile id and the catalog expect hash.
func checkReferenceLocks(prefix string, item int, other string, mLock, rLock any, profileID, expectSha string) *gateStop {
	for _, pair := range []struct {
		name string
		lock any
	}{{"manifest", mLock}, {other, rLock}} {
		lo, ok := asObj(pair.lock)
		if pair.lock == nil || !ok {
			return &gateStop{item, fmt.Sprintf("%s %d: %s.reference_lock missing", prefix, item, pair.name)}
		}
		if _, ok := jNonEmptyStr(lo["profile_id"]); !ok {
			return &gateStop{item, fmt.Sprintf("%s %d: %s.reference_lock.profile_id missing or not a string", prefix, item, pair.name)}
		}
		if !isSha256Hex(lo["expect_sha256"]) {
			return &gateStop{item, fmt.Sprintf("%s %d: %s.reference_lock.expect_sha256 missing or not 64 hex", prefix, item, pair.name)}
		}
	}
	mo, _ := asObj(mLock)
	ro, _ := asObj(rLock)
	mPid, _ := mo["profile_id"].(string)
	rPid, _ := ro["profile_id"].(string)
	mSha := strings.ToLower(mo["expect_sha256"].(string))
	rSha := strings.ToLower(ro["expect_sha256"].(string))
	cSha := strings.ToLower(expectSha)
	if mPid != profileID || rPid != profileID {
		return &gateStop{item, fmt.Sprintf("%s %d: reference_lock.profile_id three-way mismatch (manifest=%s %s=%s selected=%s)",
			prefix, item, mPid, other, rPid, profileID)}
	}
	if mSha != cSha || rSha != cSha {
		return &gateStop{item, fmt.Sprintf("%s %d: reference_lock.expect_sha256 three-way mismatch against the catalog expect hash", prefix, item)}
	}
	return nil
}

// verifyGate is Assert-VerifyGate (PS:5268-5356) item for item.
func verifyGate(outDir, profileID, expectSha string) (GateResult, *gateStop) {
	g := GateResult{Mode: RepackPacked, OutputDir: outDir, Items: VerifyGateItems}
	switch st, why := fileAbsence(filepath.Join(outDir, "experts.bin.partial")); st {
	case absPresent:
		return g, &gateStop{1, "gate 1: experts.bin.partial present (interrupted or unverified artifact)"}
	case absUnknown:
		return g, &gateStop{1, "gate 1: experts.bin.partial " + why}
	}
	reportPath := filepath.Join(outDir, "verify_report.json")
	manifestPath := filepath.Join(outDir, "manifest.json")
	if !fileExists(reportPath) {
		return g, &gateStop{2, "gate 2: verify_report.json missing"}
	}
	if !fileExists(manifestPath) {
		return g, &gateStop{6, "gate 6: manifest.json missing"}
	}
	last, err := lastNonEmptyRecord(reportPath)
	if err != nil {
		return g, &gateStop{2, "gate 2: " + err.Error()}
	}
	v, err := parseJSONStrict([]byte(last))
	if err != nil {
		return g, &gateStop{2, "gate 2: last record strict parse failed - " + err.Error()}
	}
	rec, _ := asObj(v)
	if !has(rec, "pass") {
		return g, &gateStop{3, "gate 3: pass key missing"}
	}
	pv, isBool := jBool(rec["pass"])
	if !isBool {
		return g, &gateStop{3, "gate 3: pass is not a JSON boolean"}
	}
	if !pv {
		return g, &gateStop{3, "gate 3: pass is false"}
	}
	counts := map[string]int64{}
	for _, k := range []string{"pairs_pass", "pairs_total", "expected_pairs"} {
		if !has(rec, k) {
			return g, &gateStop{4, "gate 4: " + k + " missing"}
		}
		n, ok := jNonNegInt(rec[k])
		if !ok {
			return g, &gateStop{4, "gate 4: " + k + " is not a non-negative integer"}
		}
		counts[k] = n
	}
	if counts["pairs_pass"] != counts["pairs_total"] || counts["pairs_total"] != counts["expected_pairs"] {
		return g, &gateStop{4, fmt.Sprintf("gate 4: count equality broken pairs_pass=%d pairs_total=%d expected_pairs=%d",
			counts["pairs_pass"], counts["pairs_total"], counts["expected_pairs"])}
	}
	for _, k := range []string{"problems", "failures", "padding_failures"} {
		if !has(rec, k) {
			return g, &gateStop{5, "gate 5: " + k + " missing (missing key is not an empty array)"}
		}
		a, ok := jArr(rec[k])
		if !ok {
			return g, &gateStop{5, "gate 5: " + k + " is not a JSON array"}
		}
		if len(a) != 0 {
			return g, &gateStop{5, "gate 5: " + k + " is not empty"}
		}
	}
	mv, err := readJSONFileStrict(manifestPath)
	if err != nil {
		return g, &gateStop{6, "gate 6: manifest.json unreadable - " + err.Error()}
	}
	mo, _ := asObj(mv)
	if s := checkReferenceLocks("gate", 6, "verify_report", mo["reference_lock"], rec["reference_lock"], profileID, expectSha); s != nil {
		return g, s
	}
	if !has(rec, "manifest_sha256") {
		return g, &gateStop{7, "gate 7: manifest_sha256 missing (report not bound to a cache key)"}
	}
	if !isSha256Hex(rec["manifest_sha256"]) {
		return g, &gateStop{7, "gate 7: manifest_sha256 is not a 64 hex string"}
	}
	real, err := fileSha256(manifestPath)
	if err != nil {
		return g, &gateStop{7, "gate 7: manifest.json re-hash failed - " + err.Error()}
	}
	if real != strings.ToLower(rec["manifest_sha256"].(string)) {
		return g, &gateStop{7, "gate 7: manifest_sha256 != real manifest.json bytes (report belongs to a different artifact)"}
	}
	g.State, g.Passed, g.ManifestSha, g.Count = GatePass, VerifyGateItems, real, counts["pairs_total"]
	g.CheckedAt, _ = rec["checked_at"].(string)
	return g, nil
}

// virtualPlanGate is Assert-VirtualPlanGate (PS:5453-5588) item for item.
func virtualPlanGate(outDir, profileID, expectSha string) (GateResult, *gateStop) {
	g := GateResult{Mode: RepackVirtual, OutputDir: outDir, Items: VirtualPlanGateItems}
	for _, n := range []string{"experts.bin", "experts.bin.partial"} {
		switch st, why := fileAbsence(filepath.Join(outDir, n)); st {
		case absPresent:
			return g, &gateStop{1, "vgate 1: " + n + " present in a virtual output directory"}
		case absUnknown:
			return g, &gateStop{1, "vgate 1: " + n + " " + why}
		}
	}
	manifestPath := filepath.Join(outDir, "manifest.json")
	v, err := readJSONFileStrict(filepath.Join(outDir, "plan_report.json"))
	if err != nil {
		return g, &gateStop{2, "vgate 2: plan_report.json - " + err.Error()}
	}
	rec, ok := asObj(v)
	if !ok {
		return g, &gateStop{2, "vgate 2: plan_report.json root is not a JSON object"}
	}
	if !has(rec, "mode") {
		return g, &gateStop{3, "vgate 3: mode key missing"}
	}
	mode, ok := jNonEmptyStr(rec["mode"])
	if !ok {
		return g, &gateStop{3, "vgate 3: mode is not a JSON string"}
	}
	if mode != ManifestModeVirtual {
		return g, &gateStop{3, "vgate 3: plan_report.mode is '" + mode + "', not 'virtual'"}
	}
	if !has(rec, "pass") {
		return g, &gateStop{4, "vgate 4: pass key missing"}
	}
	pv, isBool := jBool(rec["pass"])
	if !isBool {
		return g, &gateStop{4, "vgate 4: pass is not a JSON boolean"}
	}
	if !pv {
		return g, &gateStop{4, "vgate 4: pass is false"}
	}
	if !has(rec, "problems") {
		return g, &gateStop{5, "vgate 5: problems missing (missing key is not an empty array)"}
	}
	probs, ok := jArr(rec["problems"])
	if !ok {
		return g, &gateStop{5, "vgate 5: problems is not a JSON array"}
	}
	if len(probs) != 0 {
		return g, &gateStop{5, "vgate 5: problems is not empty"}
	}
	if !has(rec, "manifest_sha256") {
		return g, &gateStop{6, "vgate 6: manifest_sha256 missing (report not bound to a manifest)"}
	}
	if !isSha256Hex(rec["manifest_sha256"]) {
		return g, &gateStop{6, "vgate 6: manifest_sha256 is not a 64 hex string"}
	}
	real, err := fileSha256(manifestPath)
	if err != nil {
		return g, &gateStop{6, "vgate 6: manifest.json re-hash failed - " + err.Error()}
	}
	if real != strings.ToLower(rec["manifest_sha256"].(string)) {
		return g, &gateStop{6, "vgate 6: manifest_sha256 != real manifest.json bytes (report belongs to a different artifact)"}
	}
	mv, err := readJSONFileStrict(manifestPath)
	if err != nil {
		return g, &gateStop{7, "vgate 7: manifest.json unreadable - " + err.Error()}
	}
	mo, _ := asObj(mv)
	if s := checkReferenceLocks("vgate", 7, "plan_report", mo["reference_lock"], rec["reference_lock"], profileID, expectSha); s != nil {
		return g, s
	}
	card, ok := asObj(rec["cardinality"])
	if rec["cardinality"] == nil || !ok {
		return g, &gateStop{8, "vgate 8: cardinality missing"}
	}
	if !has(card, "records") {
		return g, &gateStop{8, "vgate 8: cardinality.records missing"}
	}
	recs, ok := jNonNegInt(card["records"])
	if !ok {
		return g, &gateStop{8, "vgate 8: cardinality.records is not a non-negative integer"}
	}
	totals, ok := asObj(mo["totals"])
	if mo["totals"] == nil || !ok {
		return g, &gateStop{8, "vgate 8: manifest totals missing"}
	}
	if !has(totals, "n_records") {
		return g, &gateStop{8, "vgate 8: manifest totals.n_records missing"}
	}
	nrec, ok := jNonNegInt(totals["n_records"])
	if !ok {
		return g, &gateStop{8, "vgate 8: manifest totals.n_records is not a non-negative integer"}
	}
	if recs != nrec {
		return g, &gateStop{8, fmt.Sprintf("vgate 8: cardinality.records(%d) != manifest totals.n_records(%d)", recs, nrec)}
	}
	g.State, g.Passed, g.ManifestSha, g.Count = GatePass, VirtualPlanGateItems, real, recs
	g.CheckedAt, _ = rec["checked_at"].(string)
	return g, nil
}

func gateFailed(g GateResult, s *gateStop) GateResult {
	g.State, g.FailedItem, g.Passed, g.Reason = GateFail, s.item, s.item-1, s.reason
	return g
}

// CheckFormat runs the artifact disposition and the gate of one mode in outDir, the way the
// PowerShell main flow does before a start (PS:14028-14080 for packed, Resolve-VirtualArtifactState
// PS:5590-5631 for virtual):
//   - packed: experts.bin.partial present = an interrupted setup (repack needed, Partial set);
//     any of experts.bin, manifest.json, verify_report.json missing = repack needed; a virtual
//     manifest here = fail (mode mismatch); else the 7-item verify gate.
//   - virtual: manifest.json present = mode detection (bin or unrecognized = fail) then the
//     8-item plan gate; absent = bin leftovers fail vgate 1, else repack needed (Partial set
//     when plan_report.json or manifest.json.partial is left).
//
// Nothing is cached: every call reads the files again.
func CheckFormat(mode, outDir, profileID, expectSha string) GateResult {
	g := GateResult{Mode: mode, OutputDir: outDir, Items: VerifyGateItems}
	if mode == RepackVirtual {
		g.Items = VirtualPlanGateItems
		mp := filepath.Join(outDir, "manifest.json")
		switch st, why := fileAbsence(mp); st {
		case absUnknown:
			return gateFailed(g, &gateStop{1, "manifest.json " + why})
		case absPresent:
			m, reason := ManifestMode(mp)
			if m == ManifestModeUnrecognized {
				return gateFailed(g, &gateStop{1, manifestModeUnrecognizedReason + " (" + reason + ")"})
			}
			if m != ManifestModeVirtual {
				return gateFailed(g, &gateStop{1, "mode: existing artifacts are " + m + ", requested " + RepackVirtual})
			}
			r, s := virtualPlanGate(outDir, profileID, expectSha)
			if s != nil {
				return gateFailed(r, s)
			}
			return r
		}
		for _, n := range []string{"experts.bin", "experts.bin.partial"} {
			switch st, why := fileAbsence(filepath.Join(outDir, n)); st {
			case absPresent:
				return gateFailed(g, &gateStop{1, "vgate 1: " + n + " present in a virtual output directory"})
			case absUnknown:
				return gateFailed(g, &gateStop{1, "vgate 1: " + n + " " + why})
			}
		}
		for _, n := range VirtualArtifactSet {
			if st, _ := fileAbsence(filepath.Join(outDir, n)); st == absPresent {
				g.Partial = true
			}
		}
		g.Reason = "no virtual plan in " + outDir
		return g
	}
	switch st, why := fileAbsence(filepath.Join(outDir, "experts.bin.partial")); st {
	case absUnknown:
		return gateFailed(g, &gateStop{1, "experts.bin.partial " + why})
	case absPresent:
		g.Partial, g.Reason = true, "an interrupted setup was found (experts.bin.partial); the repack restarts from the beginning"
		return g
	}
	var missing []string
	for _, n := range []string{"experts.bin", "manifest.json", "verify_report.json"} {
		if !fileExists(filepath.Join(outDir, n)) {
			missing = append(missing, n)
		}
	}
	if len(missing) > 0 {
		mp := filepath.Join(outDir, "manifest.json")
		if fileExists(mp) {
			if m, _ := ManifestMode(mp); m == ManifestModeVirtual {
				return gateFailed(g, &gateStop{1, "mode: existing artifacts are virtual, requested packed"})
			}
		}
		g.Reason = strings.Join(missing, ", ") + " missing in " + outDir
		return g
	}
	r, s := verifyGate(outDir, profileID, expectSha)
	if s != nil {
		return gateFailed(r, s)
	}
	return r
}

// ModeFormat is the format state of one model in one mode, for the THIS MODEL setup line.
type ModeFormat struct {
	Mode   string
	Result GateResult
	// Stale is W7 duty 12 (GL-10; SMD:14049-14058, P4 2.5 (b)): packed artifacts that pass or
	// fail their gate while the file is not the bytes the catalog pins, so nothing binds them
	// to it - the setup line's verdict reads «stale».
	Stale bool
}

// FormatGates checks both modes of a catalog model (conflict 16: «packed ✓ · virtual ✓» /
// «virtual 리팩 필요» / «packed ✕ 점검 실패»), each in its default output directory, with the
// profile's lock id and expect hash. Every call reads the files again.
func (l *Launcher) FormatGates(ctx context.Context, modelPath, profileID string) ([]ModeFormat, error) {
	cat, err := ReadCatalog(l.opt.CatalogPath, l.opt.BundleRoot)
	if err != nil {
		return nil, err
	}
	p := cat.ProfileByID(profileID)
	if p == nil {
		return nil, stop(StatusFailGateCatalog, "no catalog profile %s", profileID)
	}
	// The pin verdict from the digest cache only (no hashing here; the list's own check hashes).
	mismatch := false
	if len(p.SourceShardsSha256) > 0 {
		if d, err := l.shardDigests(ctx, siblingPaths(representativePath(modelPath)), false); err == nil && d.Complete {
			mismatch = !pinMatch(p.SourceShardsSha256, d.Sha256)
		}
	}
	var out []ModeFormat
	for _, m := range []string{RepackPacked, RepackVirtual} {
		if ctx.Err() != nil {
			return out, ctx.Err()
		}
		dir, err := resolveOutputDir(Request{ModelPath: representativePath(modelPath)}, m == RepackVirtual)
		if err != nil {
			return out, err
		}
		g := CheckFormat(m, dir, p.ID, p.ExpectSha256)
		g.At = l.os.now()
		out = append(out, ModeFormat{Mode: m, Result: g,
			Stale: m == RepackPacked && mismatch && g.State != GateNeedsRepack && !g.Partial})
	}
	l.diag.write("FORMAT_GATES", map[string]any{"model": modelPath, "results": out})
	return out, nil
}
