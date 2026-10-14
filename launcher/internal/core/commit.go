// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"errors"
	"math"
	"strconv"
	"time"
)

// GL-06: the commit prediction (F-BU B-F; FL:14). Units are MiB.

// CommitReading is the commit limit and the commit total now (GetPerformanceInfo; S57).
type CommitReading struct {
	LimitMiB, TotalMiB float64
	At                 time.Time
}

// RoomMiB is the commit room now: limit − total.
func (c CommitReading) RoomMiB() float64 { return c.LimitMiB - c.TotalMiB }

// CommitNow reads the commit through GetPerformanceInfo.
func (l *Launcher) CommitNow() (CommitReading, error) {
	if l.os.perf == nil {
		return CommitReading{}, errors.New("no commit reader on this system")
	}
	c, err := l.os.perf()
	if err != nil {
		return CommitReading{}, err
	}
	c.At = l.os.now()
	return c, nil
}

// SafeLineMiB is the commit safe line (F-BU F.1; IDEA.md:100-107, L:4178): 4 GiB.
const SafeLineMiB = 4096.0

// ToleranceTermMiB and ToleranceTotalMiB are D.2's tolerance of projection against actual.
const (
	ToleranceTermMiB  = 48.0
	ToleranceTotalMiB = 48.0
)

// Within says |diff| ≤ tol.
func Within(diff, tol float64) bool { return math.Abs(diff) <= tol }

// --- the prompt-cache cap (F-BU E; U-22, L:4275; the user's decision L:4420) ---

// CapOverheadMiB is the overhead above the cap X-CB measured on Qwen (C.6, Q-10 (a)).
const CapOverheadMiB = 210.5

// DefaultCheckpoints is E.4's --ctx-checkpoints: reduced from 32, never 0.
const DefaultCheckpoints = 4

// DefaultCap is E.3's rule, since L:4420 the allowed maximum shown for a positive choice (not
// the default; the declared name is kept): the largest power of two P from 256 to 8,192 with
// P + 210.5 ≤ H, else 0; H is the commit room left after the start with the cache off.
func DefaultCap(hMiB float64) int {
	best := 0
	for p := CapMinMiB; p <= CapMaxMiB; p *= 2 {
		if float64(p)+CapOverheadMiB <= hMiB {
			best = p
		}
	}
	return best
}

// CacheArgs are the server's prompt-cache options (E.6).
type CacheArgs struct {
	CacheRAMMiB    int
	CtxCheckpoints int
}

// ServerArgv is `--cache-ram <cap> --ctx-checkpoints 4` (checkpoints never 0, E.4).
func (c CacheArgs) ServerArgv() []string {
	cp := c.CtxCheckpoints
	if cp <= 0 {
		cp = DefaultCheckpoints
	}
	return []string{"--cache-ram", strconv.Itoa(max(0, c.CacheRAMMiB)), "--ctx-checkpoints", strconv.Itoa(cp)}
}

// ResolveCap is E.2 and E.5: no setting or 0 is the off form (cap 0, checkpoints 4); a
// positive cap the user chose is passed as it is, with checkpoints 4.
func ResolveCap(c CapSetting) CacheArgs {
	if !c.Set || c.Value <= 0 {
		return CacheArgs{CacheRAMMiB: CapOffMiB, CtxCheckpoints: DefaultCheckpoints}
	}
	return CacheArgs{CacheRAMMiB: c.Value, CtxCheckpoints: DefaultCheckpoints}
}

// StripCacheArgs removes --cache-ram (-cram) and --ctx-checkpoints with their values (E.6:
// the launcher sets them; llama-fit-params holds neither).
func StripCacheArgs(argv []string) []string {
	out := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		switch argv[i] {
		case "--cache-ram", "-cram", "--ctx-checkpoints", "--swa-checkpoints":
			i++
			continue
		}
		out = append(out, argv[i])
	}
	return out
}

// cacheCapOf is the cap a server argv passes (0 when none).
func cacheCapOf(argv []string) float64 {
	if v, ok := argvValue(argv, "--cache-ram"); ok {
		if n, err := strconv.ParseFloat(v, 64); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

// --- the prediction (B.3-B.5) ---

// CommitPrediction is one state's prediction: the pool on its own (FL:14), the VRAM mirror,
// the host remainder and the prompt-cache cap passed (E.6).
type CommitPrediction struct {
	PoolMiB, MirrorMiB, HostRemainderMiB, PromptCacheMiB float64
	State                                                MeasureState
	Label                                                NeedLabel
}

// Total is the commit prediction: pool + mirror + host remainder + cap.
func (p CommitPrediction) Total() float64 {
	return p.PoolMiB + p.MirrorMiB + p.HostRemainderMiB + p.PromptCacheMiB
}

// TotalNoCache is the prediction without the cache term (F.1, Q-5 (c)).
func (p CommitPrediction) TotalNoCache() float64 { return p.Total() - p.PromptCacheMiB }

// RAMNeed is pool + host remainder + cap (B.5, Q-7 (a)): the mirror is a commit charge only.
func (p CommitPrediction) RAMNeed() float64 { return p.PoolMiB + p.HostRemainderMiB + p.PromptCacheMiB }

// CommitFrom builds the prediction from the need path's projection (B.5: one set of terms for
// the RAM need and the commit): a measured key carries c_dev, c_host and U; a structure not
// measured leaves them out (B.6, «unverified prediction»); anything else is «unmeasured».
func CommitFrom(fit FitResult, p NeedProjection, capMiB float64) (CommitPrediction, bool) {
	t := p.Terms
	host := p.HostCMiB + t.OutMiB + t.SnapMiB + t.TabMiB
	mirror := float64(fit.Print.DevSelf())
	switch p.Label {
	case NeedProjected:
		mirror += p.Res.CDevMiB
		host += CHostMiB + p.Res.UMiB
	case NeedUnverified:
	default:
		return CommitPrediction{}, false
	}
	return CommitPrediction{PoolMiB: t.PoolMiB, MirrorMiB: mirror, HostRemainderMiB: host, PromptCacheMiB: capMiB,
		State: AfterRequest, Label: p.Label}, true
}

// RoomAfterStart is F.1: commit limit − commit total now − this run's predicted commit.
func RoomAfterStart(c CommitReading, p CommitPrediction) float64 {
	return c.LimitMiB - c.TotalMiB - p.Total()
}

// NearLimit is F.3's warning line: its numbers.
type NearLimit struct {
	Room, SafeLine, Limit, Now, Predicted, Pool float64
}

// CheckNear is F.3 ([A-NEAR]; U-13): the room after start counted without the cache term
// (Q-5 (c)); the line shows when it lies below the safe line. It changes nothing and refuses
// nothing.
func CheckNear(c CommitReading, p CommitPrediction) (NearLimit, bool) {
	n := NearLimit{Room: c.LimitMiB - c.TotalMiB - p.TotalNoCache(), SafeLine: SafeLineMiB, Limit: c.LimitMiB, Now: c.TotalMiB,
		Predicted: p.TotalNoCache(), Pool: p.PoolMiB}
	return n, n.Room < SafeLineMiB
}

// commitPrediction fills the check's commit part (F-BU F; GL-06): the reading; with the
// projection as the shown method, the prediction after one request, the room after start,
// the near-limit line and the cap's allowed maximum (E.3: H with the cache off).
func (l *Launcher) commitPrediction(res *PreflightResult, cfg *Config) {
	if c, err := l.CommitNow(); err == nil {
		res.Commit, res.CommitOK = c, true
	}
	if cfg == nil || !l.projectionValidated() {
		return
	}
	pred, ok := CommitFrom(res.Fit, res.Projection, cacheCapOf(cfg.Argv))
	res.Prediction, res.PredictionOK = pred, ok
	if !ok || !res.CommitOK {
		return
	}
	res.RoomAfterMiB = RoomAfterStart(res.Commit, pred)
	res.Near, res.NearOK = CheckNear(res.Commit, pred)
	off := pred
	off.PromptCacheMiB = 0
	res.CapMaxMiB, res.CapMaxOK = DefaultCap(RoomAfterStart(res.Commit, off)), true
	l.diag.write("COMMIT_PREDICTION", map[string]any{"limit_mib": res.Commit.LimitMiB, "total_mib": res.Commit.TotalMiB,
		"pool_mib": pred.PoolMiB, "mirror_mib": pred.MirrorMiB, "host_remainder_mib": pred.HostRemainderMiB,
		"prompt_cache_mib": pred.PromptCacheMiB, "room_after_mib": res.RoomAfterMiB, "near": res.NearOK,
		"cap_max_mib": res.CapMaxMiB, "label": pred.Label.String()})
}
