package core

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// GL-10 (plan cell GL-10, F-LS 4.6 «검증 → 속도 확인 → 서버가 저절로 시작», S8-LA-38): the speed
// check of the setup flow, ported from the PowerShell launcher: the startup probe (region 9,
// Invoke-StartupProbe PS:4190-4237 with Write-ProbeBinding PS:4152-4187) and the QD sweep
// (region 9-B, Invoke-QdSweep PS:4719-4768 and Resolve-QdSweep PS:4778-4912 with its pure
// helpers PS:4248-4502). The timed reads go through the ioMeter hook (osHooks.sweep): the real
// one is MoeLauncher.Sweep's overlapped unbuffered reads (sweep_windows.go); the tests give
// mock timings. The pure parts are locked against the PowerShell functions' recorded outputs
// (testdata/la38, record_ps.ps1).
//
// One addition is Go's own: the manual arm. With a manual prefetch pair the sweep prefers the
// smallest QD of its 90 % set that clears the pair's N + 1 [A-SWEEP] (F-PF §3.5; M-12 (i)); its
// reason word is manualPreferred, apart from PowerShell's two (PI 2: a number from an
// unvalidated prior does not wear a validated label).

// Region 9 and 9-B constants (PS:1146-1207).
const (
	probeStateFile     = "probe.scratch.json" // PROBE_STATE_FILE
	probeStateVersion  = 1                    // PROBE_STATE_VERSION
	probeSamples       = 64                   // PROBE_SAMPLES
	probeBlockFallback = 1048576              // PROBE_BLOCK_BYTES_FALLBACK
	probeSectorAlign   = 4096                 // PROBE_SECTOR_ALIGN
	probeScratchName   = ".moe-probe.tmp"     // PROBE_SCRATCH_NAME
	sweepWindowMs      = 1250                 // SWEEP_WINDOW_MS
	sweepBlockMin      = 1048576              // SWEEP_BLOCK_MIN_BYTES
	sweepBlockMax      = 16777216             // SWEEP_BLOCK_MAX_BYTES
	sweepMinBlockSpan  = 32                   // SWEEP_MIN_BLOCK_SPAN
	sweepOffsetCount   = 8192                 // SWEEP_OFFSET_COUNT
	sweepFallbackAlign = 4096                 // SWEEP_FALLBACK_ALIGN
	armManual          = "manual"             // Go: the manual pair's arm (F-PF §3.5)
	manualPreferred    = "manual-prefetch-preferred"
)

// probeScratchBytes is PROBE_SCRATCH_BYTES (64 MiB); a variable so the tests can keep the
// scratch file small.
var probeScratchBytes int64 = 67108864

var (
	sweepOrderForward = []int{1, 2, 4, 8} // SWEEP_ORDER_FORWARD
	sweepOrderReverse = []int{8, 4, 2, 1} // SWEEP_ORDER_REVERSE
)

// sweepRaw is one measured window (MoeLauncher.SweepPoint): Ticks in 100 ns units.
type sweepRaw struct {
	QD    int
	OK    bool
	Bytes int64
	Ticks int64
	Reads int
	Err   string
}

// probeReading is Measure-SsdRandomRead's success answer.
type probeReading struct {
	MiBps      float64
	BlockBytes int64
	Samples    int
	Bytes      int64
}

// ioMeter does the timed reads of the speed check. The real one reads the disk; the tests'
// one gives mock timings and records the order of the calls.
type ioMeter interface {
	// PhysicalSectorBytes is QueryPhysicalSectorBytes: 0 when unknown.
	PhysicalSectorBytes(path string) int64
	// RunPoint is one measured window at one queue depth (MoeLauncher.Sweep.RunPoint).
	RunPoint(path string, qd int, blockBytes int64, offsets []int64, windowMs int, alignBytes int64) sweepRaw
	// RandomRead is Measure-SsdRandomRead (PS:3979-4038).
	RandomRead(path string, blockBytes int64, samples int) (probeReading, error)
	// WriteScratch writes the probe's scratch file (Get-ProbeTarget PS:4057-4084).
	WriteScratch(path string, size int64) error
}

// --- the startup probe (region 9) ---

// ProbeResult is Invoke-StartupProbe's answer: a provisional sanity reading that decides
// nothing (PROBE_QD_MAP is [unmeasured], LS 12-1).
type ProbeResult struct {
	OK         bool
	QD         int
	QDSource   string
	MiBps      float64
	BlockBytes int64
	Samples    int
	Target     string
	Scratch    bool
	Reason     string
	Line       string // the [probe] line PowerShell writes
}

// probeBlockBytes is Get-ProbeBlockBytes (PS:4040-4052): manifest.layout.slot_stride_max when it
// is a positive integer, else 1 MiB.
func probeBlockBytes(outDir string) int64 {
	if v, err := readJSONFileStrict(filepath.Join(outDir, "manifest.json")); err == nil {
		o, _ := asObj(v)
		lay, _ := asObj(o["layout"])
		if s, ok := jNonNegInt(lay["slot_stride_max"]); ok && s > 0 {
			return s
		}
	}
	return probeBlockFallback
}

// probeTarget is Get-ProbeTarget (PS:4057-4084): experts.bin when it holds four fallback
// blocks, else a scratch file on the output volume.
func (l *Launcher) probeTarget(outDir string) (path string, scratch bool, reason string) {
	bin := filepath.Join(outDir, "experts.bin")
	if st, _ := fileAbsence(bin); st == absPresent {
		if fi, err := os.Stat(bin); err == nil && fi.Size() >= probeBlockFallback*4 {
			return bin, false, ""
		}
	}
	sp := filepath.Join(outDir, probeScratchName)
	if err := l.os.sweep.WriteScratch(sp, probeScratchBytes); err != nil {
		os.Remove(sp)
		return "", false, "probe scratch file could not be written on the output volume: " + err.Error()
	}
	return sp, true, ""
}

// volumeKey is Get-VolumeKey (PS:4113-4117): the path's root in lower case, display only.
func volumeKey(p string) string {
	if v := filepath.VolumeName(p); v != "" {
		return strings.ToLower(v + `\`)
	}
	return "?"
}

// startupProbe is Invoke-StartupProbe (PS:4190-4237).
func (l *Launcher) startupProbe(outDir string) ProbeResult {
	block := probeBlockBytes(outDir)
	target, scratch, why := l.probeTarget(outDir)
	if why != "" {
		l.diag.write("PROBE_FAILED", map[string]any{"reason": why})
		return ProbeResult{QD: QDDegraded, Reason: why,
			Line: fmt.Sprintf("[probe] scratch sanity measurement could not start (%s); provisional only - the QD sweep after the verify gate decides QD (RS 5)", why)}
	}
	if scratch {
		defer os.Remove(target)
	}
	res, err := l.os.sweep.RandomRead(target, block, probeSamples)
	if err != nil {
		l.diag.write("PROBE_FAILED", map[string]any{"reason": err.Error()})
		return ProbeResult{QD: QDDegraded, Reason: err.Error(), Target: target, Scratch: scratch,
			Line: fmt.Sprintf("[probe] scratch sanity read failed (%s); provisional only - the QD sweep after the verify gate decides QD (RS 5)", err.Error())}
	}
	out := ProbeResult{OK: true, QD: QDDegraded, QDSource: "conservative-default", MiBps: res.MiBps, BlockBytes: res.BlockBytes,
		Samples: res.Samples, Target: target, Scratch: scratch,
		Line: fmt.Sprintf("[probe] scratch sanity read %s MiB/s @ %d B block x%d on %s (provisional)", psNum(res.MiBps), res.BlockBytes, res.Samples, volumeKey(target)) +
			"\n        the scratch probe->QD threshold table is [unmeasured] (LS 9 item 1); QD comes from the LS 12 sweep."}
	l.diag.write("PROBE_OK", map[string]any{"mibps": res.MiBps, "block_bytes": res.BlockBytes, "qd": out.QD,
		"qd_source": out.QDSource, "target": target, "scratch": scratch})
	return out
}

// psNum prints a double as PowerShell's -f {0} does for these values (no trailing zeros).
func psNum(v float64) string {
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.2f", v), "0"), ".")
}

// probeRecord is the probe.scratch.json document in Write-ProbeBinding's key order.
type probeRecord struct {
	StateVersion int      `json:"state_version"`
	SourceTag    string   `json:"source_tag"`
	ProfileID    string   `json:"profile_id"`
	ExpectDigest string   `json:"expect_digest"`
	VolumeID     string   `json:"output_volume_id"`
	VolumeMount  string   `json:"output_volume_mount"`
	ProbeOK      bool     `json:"probe_ok"`
	QD           int      `json:"qd"`
	MiBps        *float64 `json:"mibps"`
	TS           string   `json:"ts"`
}

// writeProbeBinding is Write-ProbeBinding (PS:4152-4187): a success is not stored without a
// volume identity; a save failure is a diagnostic only.
func (l *Launcher) writeProbeBinding(sourceTag, profileID, expectDigest, outDir string, r ProbeResult) string {
	guid, err := l.os.volumeGUID(outDir)
	if err != nil && r.OK {
		l.diag.write("probe_binding_not_stored", map[string]any{"reason": "volume identity unavailable - " + err.Error()})
		return "[probe] NOTE: probe result not stored (output volume identity unavailable: " + err.Error() + "); the next run will re-probe."
	}
	rec := probeRecord{StateVersion: probeStateVersion, SourceTag: sourceTag, ProfileID: profileID, ExpectDigest: expectDigest,
		VolumeID: strings.ToLower(guid), VolumeMount: volumeKey(outDir), ProbeOK: r.OK, QD: r.QD,
		TS: l.os.now().UTC().Format("2006-01-02T15:04:05.0000000Z")}
	if r.OK {
		m := r.MiBps
		rec.MiBps = &m
	}
	b, _ := json.MarshalIndent(rec, "", "    ")
	if err := os.MkdirAll(l.opt.StateDir, 0o755); err != nil {
		l.diag.write("probe_binding_save_failed", map[string]any{"reason": err.Error()})
		return ""
	}
	if err := writeFileAtomic(filepath.Join(l.opt.StateDir, probeStateFile), b); err != nil {
		l.diag.write("probe_binding_save_failed", map[string]any{"reason": err.Error()})
		return ""
	}
	l.diag.write("PROBE_BINDING_SAVED", map[string]any{"profile_id": profileID, "probe_ok": r.OK, "qd": r.QD})
	return ""
}

// --- the QD sweep (region 9-B) ---

// manifestStrides is Get-ManifestStrides (PS:4248-4269).
func manifestStrides(path string) ([]int64, int64, string) {
	v, err := readJSONFileStrict(path)
	if err != nil {
		return nil, 0, "manifest.json unreadable - " + err.Error()
	}
	o, _ := asObj(v)
	lay, ok := asObj(o["layout"])
	if !ok {
		return nil, 0, "manifest.json has no layout object"
	}
	ls, _ := jArr(lay["layers"])
	var vals []int64
	for _, L := range ls {
		lo, _ := asObj(L)
		s, ok := jNonNegInt(lo["stride_bytes"])
		if !ok || s <= 0 {
			return nil, 0, "layout.layers[*].stride_bytes is missing or not a positive integer"
		}
		vals = append(vals, s)
	}
	if len(vals) == 0 {
		return nil, 0, "manifest.json carries no layout.layers[*].stride_bytes population"
	}
	var align int64
	if ab, ok := jNonNegInt(lay["align_bytes"]); ok && ab > 0 {
		align = ab
	}
	return vals, align, ""
}

// sweepBlock is Get-SweepBlockBytes's answer.
type sweepBlock struct {
	OK         bool
	BlockBytes int64
	Median     int64
	Reason     string
}

// sweepBlockBytes is Get-SweepBlockBytes (PS:4275-4305): the lower median of the strides, then
// the nearest aligned size inside [1 MiB, 16 MiB]; ties take the smaller.
func sweepBlockBytes(strides []int64, align int64) sweepBlock {
	if len(strides) == 0 {
		return sweepBlock{Reason: "empty stride population"}
	}
	if align <= 0 {
		return sweepBlock{Reason: "alignment is not a positive number"}
	}
	s := append([]int64(nil), strides...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	median := s[(len(s)-1)/2]
	target := min(max(median, sweepBlockMin), sweepBlockMax)
	var cands []int64
	for _, c := range []int64{target / align * align, (target + align - 1) / align * align} {
		if c >= sweepBlockMin && c <= sweepBlockMax {
			cands = append(cands, c)
		}
	}
	if len(cands) == 0 {
		return sweepBlock{Median: median, Reason: fmt.Sprintf("no block size in [1 MiB, 16 MiB] satisfies the %d B alignment", align)}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i] < cands[j] })
	best, dist := int64(0), int64(-1)
	for _, c := range cands {
		d := c - target
		if d < 0 {
			d = -d
		}
		if dist < 0 || d < dist {
			best, dist = c, d
		}
	}
	return sweepBlock{OK: true, BlockBytes: best, Median: median}
}

// sweepAlignment is Get-SweepAlignment (PS:4311-4318): the largest of the page floor, the
// physical sector and the manifest alignment.
func sweepAlignment(phys, manifestAlign int64) int64 {
	return max(int64(sweepFallbackAlign), phys, manifestAlign)
}

// sweepOffsets is New-SweepOffsetSequence (PS:4323-4346): SHA-256 over the manifest digest's 32
// bytes and the little-endian index, the first eight bytes as an unsigned little-endian number,
// modulo the block count, times the block size.
func sweepOffsets(manifestSha string, blockCount, blockBytes int64, count int) ([]int64, string) {
	if blockCount < 1 {
		return nil, "target holds no full block"
	}
	if !isSha256Hex(manifestSha) {
		return nil, "manifest_sha256 is not a 64 hex string"
	}
	seed, err := hex.DecodeString(manifestSha)
	if err != nil {
		return nil, "offset sequence derivation failed: " + err.Error()
	}
	buf := make([]byte, 36)
	copy(buf, seed)
	out := make([]int64, count)
	for i := 0; i < count; i++ {
		binary.LittleEndian.PutUint32(buf[32:], uint32(int32(i)))
		d := sha256.Sum256(buf)
		v := binary.LittleEndian.Uint64(d[:8])
		out[i] = int64(v%uint64(blockCount)) * blockBytes
	}
	return out, ""
}

// SweepPoint is one queue depth's record (ConvertTo-SweepPointRecord PS:4351-4369 and
// Merge-SweepRounds PS:4387-4436); MiBps is a display value.
type SweepPoint struct {
	QD      int      `json:"qd"`
	Status  string   `json:"status"`
	Bytes   int64    `json:"bytes"`
	Ticks   int64    `json:"elapsed_ticks"`
	Reads   int      `json:"reads"`
	Windows *int     `json:"windows,omitempty"`
	MiBps   *float64 `json:"mibps"`
	Error   *string  `json:"error"`
}

func strPtr(s string) *string { return &s }

// pointMiBps is the display value: MiB over seconds, rounded to two places (banker's
// rounding, as [math]::Round); false when not finite and positive.
func pointMiBps(bytes, ticks int64) (float64, bool) {
	m := (float64(bytes) / 1048576.0) / (float64(ticks) / 10000000.0)
	if math.IsNaN(m) || math.IsInf(m, 0) || m <= 0 {
		return 0, false
	}
	return math.RoundToEven(m*100) / 100, true
}

// sweepPointRecord is ConvertTo-SweepPointRecord (PS:4351-4369).
func sweepPointRecord(r sweepRaw) SweepPoint {
	rec := SweepPoint{QD: r.QD, Status: "failed", Bytes: r.Bytes, Ticks: r.Ticks, Reads: r.Reads}
	if r.Err != "" {
		rec.Error = strPtr(r.Err)
	}
	if r.OK && r.Bytes > 0 && r.Ticks > 0 {
		if m, ok := pointMiBps(r.Bytes, r.Ticks); ok {
			rec.Status, rec.MiBps = "ok", &m
		} else if rec.Error == nil {
			rec.Error = strPtr("throughput is not a finite positive number")
		}
	} else if rec.Error == nil {
		rec.Error = strPtr("no measurable window")
	}
	return rec
}

// mergeSweepRounds is Merge-SweepRounds (PS:4387-4436): one record per swept QD, the rounds'
// bytes and time summed; one failed window fails the point.
func mergeSweepRounds(rounds [][]SweepPoint) []SweepPoint {
	type slot struct {
		bytes, ticks int64
		reads, wins  int
		failed       bool
		err          string
	}
	acc := map[int]*slot{}
	for _, q := range sweepPoints {
		acc[q] = &slot{}
	}
	for _, round := range rounds {
		for _, p := range round {
			s, ok := acc[p.QD]
			if !ok {
				continue
			}
			s.wins++
			if p.Status != "ok" {
				s.failed = true
				if s.err == "" && p.Error != nil {
					s.err = *p.Error
				}
				continue
			}
			s.bytes += p.Bytes
			s.ticks += p.Ticks
			s.reads += p.Reads
		}
	}
	var out []SweepPoint
	for _, q := range sweepPoints {
		s := acc[q]
		w := s.wins
		rec := SweepPoint{QD: q, Status: "failed", Bytes: s.bytes, Ticks: s.ticks, Reads: s.reads, Windows: &w}
		if s.err != "" {
			rec.Error = strPtr(s.err)
		}
		if !s.failed && s.wins > 0 && s.bytes > 0 && s.ticks > 0 {
			if m, ok := pointMiBps(s.bytes, s.ticks); ok {
				rec.Status, rec.MiBps = "ok", &m
			} else if rec.Error == nil {
				rec.Error = strPtr("throughput is not a finite positive number")
			}
		} else if rec.Error == nil {
			rec.Error = strPtr("no measured window for this queue depth")
		}
		out = append(out, rec)
	}
	return out
}

func toSelectPoints(ps []SweepPoint) []sweepPoint {
	out := make([]sweepPoint, 0, len(ps))
	for _, p := range ps {
		out = append(out, sweepPoint{qd: p.QD, status: p.Status, bytes: p.Bytes, ticks: p.Ticks})
	}
	return out
}

// formatSweepPoints is Format-SweepPoints (PS:4914-4926).
func formatSweepPoints(ps []SweepPoint) string {
	var parts []string
	for _, q := range sweepPoints {
		txt := "n/a"
		for _, p := range ps {
			if p.QD != q {
				continue
			}
			if p.Status == "ok" && p.MiBps != nil {
				txt = psNum(*p.MiBps) + " MiB/s"
			} else {
				txt = "failed"
			}
		}
		parts = append(parts, fmt.Sprintf("QD%d %s", q, txt))
	}
	return strings.Join(parts, " | ")
}

// sweepRun is Invoke-QdSweep's answer.
type sweepRun struct {
	OK            bool
	Reason        string
	Points        []SweepPoint
	Sel           sweepSelection
	BlockBytes    int64
	WindowMs      int
	Rounds        int
	Confirm       bool
	AlignBytes    int64
	Physical      int64
	ManifestAlign int64
	BlockCount    int64
	Median        int64
	Order         []int    // the QD of every measured window, in order (S8-LA-38)
	Lines         []string // the [sweep] lines PowerShell writes while it measures
}

// qdSweep is Invoke-QdSweep (PS:4719-4768): two crossed rounds over the same offsets, one
// confirmation round when they disagree, the rounds merged, the selection with the arm. A
// stop between two windows ends the sweep (Go: PowerShell polls no stop inside it).
func (l *Launcher) qdSweep(ctx context.Context, outDir, manifestSha, arm string, n *int64) sweepRun {
	bin := filepath.Join(outDir, "experts.bin")
	if st, _ := fileAbsence(bin); st != absPresent {
		return sweepRun{Reason: "experts.bin is not available for the sweep"}
	}
	fi, err := os.Stat(bin)
	if err != nil {
		return sweepRun{Reason: "experts.bin size query failed: " + err.Error()}
	}
	strides, mAlign, why := manifestStrides(filepath.Join(outDir, "manifest.json"))
	if why != "" {
		return sweepRun{Reason: why}
	}
	phys := l.os.sweep.PhysicalSectorBytes(bin)
	align := sweepAlignment(phys, mAlign)
	blk := sweepBlockBytes(strides, align)
	if !blk.OK {
		return sweepRun{Reason: blk.Reason}
	}
	count := fi.Size() / blk.BlockBytes
	if count < sweepMinBlockSpan {
		return sweepRun{Reason: fmt.Sprintf("experts.bin holds %d full %d B blocks, below the %d block span the sweep needs", count, blk.BlockBytes, sweepMinBlockSpan)}
	}
	offs, why := sweepOffsets(manifestSha, count, blk.BlockBytes, sweepOffsetCount)
	if why != "" {
		return sweepRun{Reason: why}
	}
	run := sweepRun{OK: true, BlockBytes: blk.BlockBytes, WindowMs: sweepWindowMs, AlignBytes: align, Physical: phys,
		ManifestAlign: mAlign, BlockCount: count, Median: blk.Median}
	run.Lines = append(run.Lines, fmt.Sprintf("[sweep] measuring queue depth 1/2/4/8 on experts.bin (%d B block, 2 x %d ms per point, about %d s)...",
		blk.BlockBytes, sweepWindowMs, sweepWindowMs*2*len(sweepPoints)/1000))
	l.diag.write("SWEEP_START", map[string]any{"line": run.Lines[0]})
	round := func(order []int) ([]SweepPoint, bool) {
		var pts []SweepPoint
		for _, q := range order {
			if ctx.Err() != nil {
				return pts, false
			}
			rec := sweepPointRecord(l.os.sweep.RunPoint(bin, q, blk.BlockBytes, offs, sweepWindowMs, align))
			run.Order = append(run.Order, q)
			l.diag.write("SWEEP_POINT", rec)
			pts = append(pts, rec)
		}
		return pts, true
	}
	var rounds [][]SweepPoint
	for _, order := range [][]int{sweepOrderForward, sweepOrderReverse} {
		pts, ok := round(order)
		if !ok {
			return sweepRun{Reason: "stop requested during the QD sweep", Order: run.Order, Lines: run.Lines}
		}
		rounds = append(rounds, pts)
	}
	selA := selectSweep(toSelectPoints(rounds[0]), arm, n)
	selB := selectSweep(toSelectPoints(rounds[1]), arm, n)
	if selA.QD != selB.QD {
		run.Confirm = true
		run.Lines = append(run.Lines, fmt.Sprintf("[sweep] the two crossed rounds selected QD%d and QD%d; running one confirmation round...", selA.QD, selB.QD))
		l.diag.write("SWEEP_CONFIRM", map[string]any{"line": run.Lines[len(run.Lines)-1]})
		pts, ok := round(sweepOrderForward)
		if !ok {
			return sweepRun{Reason: "stop requested during the QD sweep", Order: run.Order, Lines: run.Lines}
		}
		rounds = append(rounds, pts)
	}
	run.Points = mergeSweepRounds(rounds)
	run.Sel = selectSweep(toSelectPoints(run.Points), arm, n)
	run.Rounds = len(rounds)
	return run
}

// sweepEntry is one probe.state.json entry in Resolve-QdSweep's key order (PS:4877-4893).
type sweepEntry struct {
	Algorithm       string       `json:"probe_algorithm"`
	Method          string       `json:"measurement_method"`
	SourceTag       string       `json:"source_tag"`
	ProfileID       string       `json:"profile_id"`
	ExpectDigest    string       `json:"expectation_digest"`
	ManifestSha     string       `json:"manifest_sha256"`
	VolumeGUID      string       `json:"output_volume_guid"`
	SelectedQD      int          `json:"selected_qd"`
	SelectedReason  string       `json:"selected_reason"`
	BlockBytes      int64        `json:"block_bytes"`
	WindowMs        int          `json:"window_duration_ms"`
	WindowsPerPoint int          `json:"windows_per_point"`
	Points          []SweepPoint `json:"points"`
	SweptAt         string       `json:"swept_at"`
	RepackCheckedAt string       `json:"repack_checked_at"`
}

// writeSweepBinding is Write-SweepBinding (PS:4682-4716): the other keys survive, a v1 or
// unreadable document is replaced, the file is published atomically.
func (l *Launcher) writeSweepBinding(key string, e sweepEntry) error {
	if err := os.MkdirAll(l.opt.StateDir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(l.opt.StateDir, sweepStateFile)
	entries := map[string]json.RawMessage{}
	var order []string
	if b, err := os.ReadFile(path); err == nil {
		var doc struct {
			Ver     json.RawMessage `json:"state_version"`
			Entries json.RawMessage `json:"entries"`
		}
		if json.Unmarshal(b, &doc) == nil && strings.TrimSpace(string(doc.Ver)) == fmt.Sprint(sweepStateVer) {
			var m map[string]json.RawMessage
			if json.Unmarshal(doc.Entries, &m) == nil {
				for k, v := range m {
					if k != key {
						entries[k] = v
						order = append(order, k)
					}
				}
			}
		}
	}
	sort.Strings(order)
	eb, err := json.Marshal(e)
	if err != nil {
		return err
	}
	entries[key] = eb
	order = append(order, key)
	var sb strings.Builder
	fmt.Fprintf(&sb, "{\n    \"state_version\": %d,\n    \"entries\": {", sweepStateVer)
	for i, k := range order {
		kb, _ := json.Marshal(k)
		if i > 0 {
			sb.WriteString(",")
		}
		fmt.Fprintf(&sb, "\n        %s: %s", kb, entries[k])
	}
	sb.WriteString("\n    }\n}")
	if err := writeFileAtomic(path, []byte(sb.String())); err != nil {
		return err
	}
	l.diag.write("SWEEP_BINDING_SAVED", map[string]any{"key": key, "selected_qd": e.SelectedQD})
	return nil
}
