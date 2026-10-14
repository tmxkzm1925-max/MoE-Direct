// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// The need path (GL-05; F-BU edition B.3-B.6, FIT-PRE sections 5 and 6, X-FIT): the RAM need of
// a start from the shipped llama-fit-params at the model-affecting part of the server argv,
// the GGUF and the settings, with no per-model constant and no list of known models.

// FitParamsExe is the projection tool shipped beside llama-server.exe (U-25 (i), L:4264).
const FitParamsExe = "llama-fit-params.exe"

// fitTimeout bounds one call: X-FIT measured 0.46-0.71 s (F-BU C.2).
const fitTimeout = 60 * time.Second

// fitServerOnly are the server argv options llama-fit-params does not register (FIT-PRE 6.1;
// X-FIT section 4 item 3: --host, --port, --no-webui, --no-warmup for the catalog profiles;
// S8-LA-11 ①: --cache-ram, --ctx-checkpoints; the launcher's own --alias and slot save path),
// with the number of values each takes. -m is set again at the end.
var fitServerOnly = map[string]int{
	"--host": 1, "--port": 1, "--no-webui": 0, "--no-warmup": 0, "--cache-ram": 1, "-cram": 1,
	"--ctx-checkpoints": 1, "--swa-checkpoints": 1, "--alias": 1, "-a": 1, argSlotSave: 1,
	"-m": 1, "--model": 1,
}

// FitArgv is the llama-fit-params argv of a start: the server argv less the server-only
// options, then --fit-print on -m <first shard> (X-FIT P-ARGV gates; S8-LA-11 ①).
func FitArgv(serverArgv []string, modelPath string) []string {
	out := make([]string, 0, len(serverArgv)+4)
	for i := 0; i < len(serverArgv); i++ {
		if n, ok := fitServerOnly[serverArgv[i]]; ok {
			i += n
			continue
		}
		out = append(out, serverArgv[i])
	}
	return append(out, "--fit-print", "on", "-m", modelPath)
}

// FitDevice is one line of --fit-print on: device, model, context and compute in MiB.
type FitDevice struct {
	Name                    string
	Model, Context, Compute int64
}

// FitPrint is a parsed llama-fit-params output: the devices, then Host (FIT-PRE 6.2).
type FitPrint struct {
	Devices []FitDevice
	Host    FitDevice
}

// DevSelf is Σ_d P_d, the devices' model + context + compute.
func (f FitPrint) DevSelf() int64 {
	var s int64
	for _, d := range f.Devices {
		s += d.Model + d.Context + d.Compute
	}
	return s
}

var fitLineRe = regexp.MustCompile(`^(\S+) (\d+) (\d+) (\d+)\s*$`)

// ParseFitPrint reads stdout: every non-empty line is «<device> <model> <context> <compute>»
// and exactly one is Host; anything else is unparsable (S8-LA-11 ②, ③).
func ParseFitPrint(stdout string) (FitPrint, error) {
	var f FitPrint
	host := 0
	sc := bufio.NewScanner(strings.NewReader(stdout))
	for sc.Scan() {
		ln := strings.TrimRight(sc.Text(), "\r")
		if strings.TrimSpace(ln) == "" {
			continue
		}
		m := fitLineRe.FindStringSubmatch(ln)
		if m == nil {
			return FitPrint{}, fmt.Errorf("unparsable fit-params line %q", ln)
		}
		d := FitDevice{Name: m[1]}
		var err error
		for i, p := range []*int64{&d.Model, &d.Context, &d.Compute} {
			if *p, err = strconv.ParseInt(m[2+i], 10, 64); err != nil {
				return FitPrint{}, fmt.Errorf("unparsable fit-params value %q", m[2+i])
			}
		}
		if d.Name == "Host" {
			host++
			f.Host = d
		} else {
			f.Devices = append(f.Devices, d)
		}
	}
	if host != 1 {
		return FitPrint{}, fmt.Errorf("fit-params printed %d Host lines", host)
	}
	return f, nil
}

// FitResult is one call: the output, or why there is none.
type FitResult struct {
	Argv     []string
	Print    FitPrint
	OK       bool
	ExitCode int
	Reason   string // why the projection is «unmeasured»
	Took     time.Duration
}

// fitRunner runs the tool and returns its stdout, stderr and exit code (a test fakes it).
type fitRunner func(ctx context.Context, exe string, argv, env []string, dir string) (stdout, stderr []byte, code int, err error)

// runFitProcess runs the tool with stdout and stderr on separate handles: one merged handle
// cut the output in X-FIT's run (X-FIT section 4 item 2; End's 2-A, EL:17).
func runFitProcess(ctx context.Context, exe string, argv, env []string, dir string) ([]byte, []byte, int, error) {
	var out, errb bytes.Buffer
	cmd := exec.CommandContext(ctx, exe, argv...)
	cmd.Env, cmd.Dir = env, dir
	cmd.Stdout, cmd.Stderr = &out, &errb
	hideWindow(cmd)
	err := cmd.Run()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		err = nil
	}
	return out.Bytes(), errb.Bytes(), code, err
}

// RunFitParams calls the tool for a start's argv. Any failure gives OK false with the
// reason: a missing tool, a start failure, a non-zero exit (1 for an argument error,
// 0xC0000409 for a load failure, F-BU C.2), a timeout or an unparsable output (F-BU B.6;
// L:4178: only a model whose prediction fails is «unmeasured»).
func RunFitParams(ctx context.Context, run fitRunner, exe string, serverArgv []string, modelPath string, env []string) FitResult {
	r := FitResult{Argv: FitArgv(serverArgv, modelPath)}
	if run == nil {
		run = runFitProcess
	}
	t0 := time.Now()
	ctx, cancel := context.WithTimeout(ctx, fitTimeout)
	defer cancel()
	stdout, _, code, err := run(ctx, exe, r.Argv, env, filepath.Dir(exe))
	r.Took, r.ExitCode = time.Since(t0), code
	switch {
	case err != nil:
		r.Reason = "fit-params did not run: " + err.Error()
	case ctx.Err() != nil:
		r.Reason = "fit-params timed out"
	case code != 0:
		r.Reason = fmt.Sprintf("fit-params exit 0x%08X", uint32(code))
	default:
		p, perr := ParseFitPrint(string(stdout))
		if perr != nil {
			r.Reason = perr.Error()
		} else {
			r.Print, r.OK = p, true
		}
	}
	return r
}

// NeedLabel says what the shown need is (S8-LA-13; F-BU B.4 Label).
type NeedLabel int

const (
	NeedUnmeasured NeedLabel = iota // «unmeasured»
	NeedMeasured                    // «measured»: the last measured value (not validated)
	NeedProjected                   // the projection (validated, a measured structure and setting)
	NeedUnverified                  // «unverified prediction»: a structure not measured
)

func (n NeedLabel) String() string {
	return [...]string{"unmeasured", "measured", "projection", "unverified prediction"}[n]
}

// CHostMiB is c_host, one probe value (F-BU B.3, Q-12 (c)): used for a measured setting only.
const CHostMiB = 340.0

// PageBytes is A of FIT section 5: max(4096, the volume's sector size); the sector size is
// not read here, and NVMe sectors are 512 or 4096 B [R].
const PageBytes = 4096

// EngineTerms are the FIT section 5 terms the GGUF and the settings give.
type EngineTerms struct {
	StrideBytes int64   // S
	Slots       int64   // n_slots = floor(B × 2^20 / S)
	PoolMiB     float64 // T_pool = n_slots × S
	OutMiB      float64 // T_out = n_vocab × 4 B (one output row)
	SnapMiB     float64 // T_snap, when prefetch is on
	TabMiB      float64 // T_tab without the phase-B pools (no formula, FIT section 5)
}

const bytesPerMiB = 1 << 20

// TermsOf computes the engine terms for a budget (MiB) and the prefetch state.
func TermsOf(s *ModelShape, budgetMiB int64, prefetch bool) (EngineTerms, bool) {
	stride, ok := s.SlotStride(PageBytes)
	if !ok || budgetMiB <= 0 {
		return EngineTerms{}, false
	}
	t := EngineTerms{StrideBytes: stride, Slots: budgetMiB * bytesPerMiB / stride}
	t.PoolMiB = float64(t.Slots*stride) / bytesPerMiB
	t.OutMiB = float64(s.NVocab*4) / bytesPerMiB
	if prefetch && s.NLayer > 0 {
		t.SnapMiB = float64(4*((s.NLayer-1)*s.NEmbd*s.NExpert+s.NLayer*s.NEmbd)) / bytesPerMiB
	}
	t.TabMiB = float64(t.Slots*232+s.MoELayers()*s.NExpert*4+148992+448) / bytesPerMiB
	return t, true
}

// MeasureState is the state a reading stands for (F-BU B.2): at load (P1: the server has
// loaded and answers) or after one request (P3: one request and 60 s idle).
type MeasureState int

const (
	AtLoad       MeasureState = iota // P1
	AfterRequest                     // P3
)

func (s MeasureState) String() string { return [...]string{"P1", "P3"}[s] }

// Residuals are the last measured c_dev and U of one model, settings and state (F-BU B.6; Q-1
// (a): the exact key only), with the run they come from. The launcher's own runs are their
// only source (F-BU B.7 (a)).
type Residuals struct {
	CDevMiB, UMiB float64
	Run           string
}

// ResidualKey is the key of the last measured residuals: the model's content identity (never
// its file name or general.name, S8-LA-12), the context and the budget.
type ResidualKey struct {
	Identity  string
	Ctx       int64
	BudgetMiB int64
}

// ResidualStore holds the last measured residuals per key and state; Measured says whether
// any setting of a structure (the GGUF architecture) has them (F-BU B.6: a structure measured
// or not); Put replaces the entry of a key and state with the latest run's.
type ResidualStore interface {
	Lookup(ResidualKey, MeasureState) (Residuals, bool)
	Measured(arch string) bool
	Put(key ResidualKey, arch string, st MeasureState, r Residuals) error
}

// DeriveResiduals is one reading's c_dev and U (F-BU B.7 (a); X-FIT :84):
// c_dev = mirror − Σ_d P_d and U = PrivateUsage − T_pool − mirror − P_Hc − T_out − T_snap −
// T_tab − c_host, with the same run's fit-params terms and engine terms. privateBytes is the
// server's PrivateUsage, mirrorBytes its GPU dedicated memory.
func DeriveResiduals(privateBytes, mirrorBytes int64, fit FitPrint, t EngineTerms) Residuals {
	mirror := float64(mirrorBytes) / bytesPerMiB
	return Residuals{
		CDevMiB: mirror - float64(fit.DevSelf()),
		UMiB: float64(privateBytes)/bytesPerMiB - t.PoolMiB - mirror - float64(fit.Host.Compute) - t.OutMiB - t.SnapMiB -
			t.TabMiB - CHostMiB,
	}
}

// NeedProjection is the projection's RAM need (F-BU B.5, Q-7 (a)): pool + host remainder +
// the cap. The host remainder is P_Hc + T_out + T_snap + T_tab + c_host + U for a measured
// setting; for a structure not measured it leaves out c_host and U, which are named
// «unmeasured» (Q-2 (a), Q-12 (c)).
type NeedProjection struct {
	Label    NeedLabel
	MiB      float64
	Terms    EngineTerms
	HostCMiB float64 // P_Hc
	Res      Residuals
	Reason   string // why «unmeasured»
	// Key and Arch name the start for the residual store; TermsOK says the terms were computed
	// (a run of this start can then record its residuals).
	Key     ResidualKey
	Arch    string
	TermsOK bool
}

// ProjectNeed puts the terms together. capMiB is the prompt-cache cap the start passes (0 at
// the off form, F-BU E.2; GL-06 sets it).
func ProjectNeed(fit FitResult, shape *ModelShape, shapeErr error, identity string, ctx, budgetMiB int64, prefetch bool,
	capMiB float64, store ResidualStore) NeedProjection {
	switch {
	case !fit.OK:
		return NeedProjection{Reason: fit.Reason}
	case shapeErr != nil || shape == nil:
		return NeedProjection{Reason: fmt.Sprintf("model shape: %v", shapeErr)}
	}
	t, ok := TermsOf(shape, budgetMiB, prefetch)
	if !ok {
		return NeedProjection{Reason: "slot stride: a tensor type or shape is not known"}
	}
	p := NeedProjection{Terms: t, HostCMiB: float64(fit.Print.Host.Compute), Key: ResidualKey{identity, ctx, budgetMiB},
		Arch: shape.Arch, TermsOK: true}
	base := t.PoolMiB + p.HostCMiB + t.OutMiB + t.SnapMiB + t.TabMiB + capMiB
	if store == nil || !store.Measured(shape.Arch) {
		p.Label, p.MiB = NeedUnverified, base
		return p
	}
	// the displayed state is after one request (F-BU B.2, Q-3 (a))
	res, ok := store.Lookup(p.Key, AfterRequest)
	if !ok {
		p.Reason = "no measured residuals for these settings"
		return p
	}
	p.Label, p.Res, p.MiB = NeedProjected, res, base+CHostMiB+res.UMiB
	return p
}

// ceilMiBf rounds a MiB figure up to a whole MiB.
func ceilMiBf(v float64) int64 { return int64(math.Ceil(v - 1e-9)) }

// projection runs the need path for a prepared start: llama-fit-params beside the server
// exe at the start's argv, with the bootstrap environment only (X-FIT: the MOE_* settings
// leave the output unchanged, f01 = f02), and the model's shape for the engine terms.
func (l *Launcher) projection(ctx context.Context, cat *Catalog, set *ShardSet, cfg *Config) (FitResult, NeedProjection) {
	if cfg == nil || set == nil || len(set.Shards) == 0 {
		return FitResult{Reason: "no configuration"}, NeedProjection{Reason: "no configuration"}
	}
	first := set.Shards[0].Path
	exe := filepath.Join(l.opt.BundleRoot, filepath.Dir(cat.Runtime.ServerExe), FitParamsExe)
	fit := RunFitParams(ctx, l.os.fit, exe, cfg.Argv, first, explicitEnv(nil, os.Getenv))
	shape, serr := readModelShape(first)
	ctxTok := int64(0)
	if v, ok := argvValue(cfg.Argv, "-c"); ok {
		ctxTok, _ = strconv.ParseInt(v, 10, 64)
	}
	ident := ""
	if shape != nil {
		ident = shape.Identity
	}
	// the cache term is the cap the server argv passes (F-BU E.6; GL-06)
	pr := ProjectNeed(fit, shape, serr, ident, ctxTok, cfg.Budget.MB, cfg.Prefetch.On, cacheCapOf(cfg.Argv), l.residualStore())
	l.diag.write("NEED_PROJECTION", map[string]any{"fit_argv": fit.Argv, "fit_ok": fit.OK, "fit_exit": fit.ExitCode,
		"fit_ms": fit.Took.Milliseconds(), "reason": pr.Reason, "label": pr.Label.String(), "need_mib": pr.MiB,
		"terms": pr.Terms, "host_compute_mib": pr.HostCMiB})
	return fit, pr
}

// ProjectionMethodValidated is the one «validated» flag of the projection method ([A-VAL];
// F-BU B.6, D.3): set by D.3's reading (the method passes the 48 MiB tolerance on the measured
// models Qwen and DSV4), to be confirmed or reversed by the legs S59-S62; a reversal changes
// only this line (F-BU B.7 (c)).
const ProjectionMethodValidated = true

// projectionValidated is the flag this launcher uses: the method's, unless pinned.
func (l *Launcher) projectionValidated() bool {
	if l.opt.ProjectionValidation != nil {
		return *l.opt.ProjectionValidation
	}
	return ProjectionMethodValidated
}

// ShownNeed is the need H1 shows (S8-LA-13; F-BU B.6). Not validated: the last measured
// value (review item 13, O1: the last run's peak, else its ready value + its budget), labelled
// «measured»; none recorded: «unmeasured». Validated: the projection; a structure not measured
// shows it labelled «unverified prediction»; a projection that failed, or a measured
// structure's setting with no residuals, is «unmeasured».
func ShownNeed(lastMiB int64, lastBasis NeedBasis, p NeedProjection, validated bool) (known bool, miB int64, basis NeedBasis, label NeedLabel) {
	if !validated {
		if lastBasis == NeedNotMeasured || lastMiB <= 0 {
			return false, 0, NeedNotMeasured, NeedUnmeasured
		}
		return true, lastMiB, lastBasis, NeedMeasured
	}
	switch p.Label {
	case NeedProjected:
		return true, ceilMiBf(p.MiB), NeedFromProjection, NeedProjected
	case NeedUnverified:
		return true, ceilMiBf(p.MiB), NeedFromUnverified, NeedUnverified
	}
	return false, 0, NeedNotMeasured, NeedUnmeasured
}
