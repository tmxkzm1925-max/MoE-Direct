// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// The store of the last measured c_dev and U (GL-05; F-BU B.7 (a), 54505642f): filled from this
// launcher's own runs only, at P1 and at P3 when reached; no seed data (B.7 (b)). It sits
// directly under the user data root %LOCALAPPDATA%\MoE-Direct, beside logs and presets
// (B.7's StoreDir).
const (
	residualFile   = "need_residuals.json"
	residualSchema = 1
)

// P3Idle is the idle time after a request before the P3 reading (F-BU B.2: one request and
// 60 s idle, as X-FIT measured it).
const P3Idle = 60 * time.Second

type residualEntry struct {
	Identity  string  `json:"identity"`
	Arch      string  `json:"arch"`
	Ctx       int64   `json:"ctx"`
	BudgetMiB int64   `json:"budget_mib"`
	State     string  `json:"state"` // "P1" or "P3"
	CDevMiB   float64 `json:"c_dev_mib"`
	UMiB      float64 `json:"u_mib"`
	Run       string  `json:"run"` // the run coordinate of the reading
}

type residualDoc struct {
	SchemaVersion int             `json:"schema_version"`
	Entries       []residualEntry `json:"entries"`
}

// FileResidualStore is the ResidualStore of one JSON file. A file that is missing, unreadable
// or of another schema reads as empty: every key is then «unmeasured».
type FileResidualStore struct{ Path string }

var residualMu sync.Mutex

func (f FileResidualStore) read() residualDoc {
	d := residualDoc{SchemaVersion: residualSchema}
	b, err := os.ReadFile(f.Path)
	if err != nil {
		return d
	}
	var in residualDoc
	if json.Unmarshal(b, &in) != nil || in.SchemaVersion != residualSchema {
		return d
	}
	return in
}

func (e residualEntry) is(k ResidualKey, st MeasureState) bool {
	return e.Identity == k.Identity && e.Ctx == k.Ctx && e.BudgetMiB == k.BudgetMiB && e.State == st.String()
}

// Lookup is the exact key and state only (Q-1 (a)).
func (f FileResidualStore) Lookup(k ResidualKey, st MeasureState) (Residuals, bool) {
	residualMu.Lock()
	defer residualMu.Unlock()
	for _, e := range f.read().Entries {
		if e.is(k, st) {
			return Residuals{CDevMiB: e.CDevMiB, UMiB: e.UMiB, Run: e.Run}, true
		}
	}
	return Residuals{}, false
}

// Measured says whether any entry is of this architecture.
func (f FileResidualStore) Measured(arch string) bool {
	residualMu.Lock()
	defer residualMu.Unlock()
	for _, e := range f.read().Entries {
		if e.Arch == arch {
			return true
		}
	}
	return false
}

// Put replaces the entry of the key and state with the latest run's.
func (f FileResidualStore) Put(k ResidualKey, arch string, st MeasureState, r Residuals) error {
	if k.Identity == "" {
		return fmt.Errorf("residuals without a model identity")
	}
	residualMu.Lock()
	defer residualMu.Unlock()
	d := f.read()
	e := residualEntry{Identity: k.Identity, Arch: arch, Ctx: k.Ctx, BudgetMiB: k.BudgetMiB, State: st.String(),
		CDevMiB: r.CDevMiB, UMiB: r.UMiB, Run: r.Run}
	out := d.Entries[:0:0]
	for _, x := range d.Entries {
		if !x.is(k, st) {
			out = append(out, x)
		}
	}
	out = append(out, e)
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Identity != b.Identity {
			return a.Identity < b.Identity
		}
		if a.Ctx != b.Ctx {
			return a.Ctx < b.Ctx
		}
		if a.BudgetMiB != b.BudgetMiB {
			return a.BudgetMiB < b.BudgetMiB
		}
		return a.State < b.State
	})
	d.SchemaVersion, d.Entries = residualSchema, out
	b, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(f.Path, b)
}

// residualStore is the store of this launcher: the options' (tests), else the file.
func (l *Launcher) residualStore() ResidualStore {
	if l.opt.Residuals != nil {
		return l.opt.Residuals
	}
	return FileResidualStore{Path: filepath.Join(l.opt.StateDir, residualFile)}
}

// runCoord is a run coordinate of a reading: the ready time and the server's PID.
func runCoord(readyAt time.Time, pid int) string {
	return fmt.Sprintf("%s pid %d", readyAt.UTC().Format(time.RFC3339), pid)
}

// recordResiduals derives one reading's c_dev and U and stores them (F-BU B.7): pre is the
// start's check, whose fit-params and engine terms are the same run's; pm is the server's
// PrivateUsage reading and the GPU dedicated memory is read here. A missing reading stores
// nothing, and the key stays «unmeasured».
func (l *Launcher) recordResiduals(pre *PreflightResult, st MeasureState, pid int, pm ProcessMemory, memErr error, run string) {
	why := ""
	switch {
	case pre == nil || !pre.Fit.OK || !pre.Projection.TermsOK || pre.Projection.Key.Identity == "":
		why = "no projection terms for this start"
	case memErr != nil || pm.PrivateBytes <= 0:
		why = "no PrivateUsage reading"
	}
	var mirror int64
	if why == "" {
		inst, err := l.os.gpuMem()
		b, ok := gpuShareOf(inst, pid)
		switch {
		case err != nil:
			why = "no GPU dedicated memory reading: " + err.Error()
		case !ok:
			why = "no GPU Process Memory instance for the server's PID"
		default:
			mirror = b
		}
	}
	if why != "" {
		l.diag.write("RESIDUALS_NOT_RECORDED", map[string]any{"state": st.String(), "reason": why})
		return
	}
	r := DeriveResiduals(pm.PrivateBytes, mirror, pre.Fit.Print, pre.Projection.Terms)
	r.Run = run
	err := l.residualStore().Put(pre.Projection.Key, pre.Projection.Arch, st, r)
	rec := map[string]any{"state": st.String(), "identity": pre.Projection.Key.Identity, "ctx": pre.Projection.Key.Ctx,
		"budget_mib": pre.Projection.Key.BudgetMiB, "private_bytes": pm.PrivateBytes, "mirror_bytes": mirror,
		"c_dev_mib": r.CDevMiB, "u_mib": r.UMiB, "run": run}
	if err != nil {
		rec["write_error"] = err.Error()
	}
	l.diag.write("RESIDUALS", rec)
}

// p3Due says the P3 reading is due: at least one request finished, none in flight, and the
// server idle for P3Idle since its last activity.
func p3Due(sum TimingSummary, now time.Time) bool {
	return sum.Requests >= 1 && !sum.Prefilling && !sum.Decoding && !sum.LastActivity.IsZero() &&
		now.Sub(sum.LastActivity) >= P3Idle
}

// sampleP3 takes the P3 reading once per run when it is due (the serve loop calls it with
// the peak sample).
func (l *Launcher) sampleP3(s *session) {
	s.mu.Lock()
	done, lg, pre, child, readyAt := s.p3Done, s.log, s.pre, s.child, s.readyAt
	s.mu.Unlock()
	if done || lg == nil || child == nil {
		return
	}
	lg.poll()
	if !p3Due(lg.summary(), l.os.now()) {
		return
	}
	s.mu.Lock()
	s.p3Done = true
	s.mu.Unlock()
	pm, err := child.Memory()
	l.recordResiduals(pre, AfterRequest, child.PID(), pm, err, runCoord(readyAt, child.PID()))
}
