package core

import (
	"context"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Conflict 18-2 of reviews/_gol_conflicts_8_user_20261010.md: the bundle manifest check of
// Assert-BundleIntegrity (PS:2187-2238) runs at every launch, in the background from the
// moment the screen is up; a start waits for it to finish; a mismatch gives the count of
// differing files (exit 4 on the argument path); with no manifest the build is a development
// build and the check is skipped and said so.
//
// PowerShell stops at the first difference; the user asked for the count, so every listed
// file is hashed and every difference counted: a listed file that is missing, a listed file
// whose sha256 differs, and a file inside the root that the manifest does not list.
//
// GL-13: the walk for unlisted files is PowerShell's Get-ChildItem -Recurse -File without
// -Force, as both the assembler (make_bundle.ps1 region 5) and Assert-BundleIntegrity run it:
// a hidden file, and everything under a hidden folder, is neither listed nor counted
// (bundleHidden). The manifest format is the assembler's and stays unchanged.

// Bundle manifest constants (BUNDLE_MANIFEST_NAME, BUNDLE_MANIFEST_VERSION).
const (
	BundleManifestName    = "bundle_manifest.json"
	BundleManifestVersion = 1
)

// BundleState is the verdict of one bundle check.
type BundleState uint8

const (
	BundlePending  BundleState = iota // not finished yet
	BundleOK                          // every listed file present with its sha256, nothing unlisted
	BundleMismatch                    // n files differ (Mismatches), or the manifest itself is broken
	BundleDev                         // no manifest: a development build, the check is skipped
)

func (s BundleState) String() string {
	return [...]string{"pending", "ok", "mismatch", "dev"}[s]
}

// BundleDiff is one differing file.
type BundleDiff struct {
	Path string
	Kind string // missing | hash | unlisted | hash-failed
}

// BundleResult is one bundle check.
type BundleResult struct {
	State      BundleState
	Root       string
	Manifest   string // the manifest path
	Files      int    // files listed in the manifest
	Mismatches int    // the count of differing files
	Diffs      []BundleDiff
	// Reason is the manifest's own defect (unreadable, wrong version, no files, a bad entry):
	// then nothing can be compared and Mismatches stays 0.
	Reason   string
	Started  time.Time
	Finished time.Time
}

// Err is the start refusal of a failed check (fail_gate_bundle, exit 4); nil otherwise.
func (r BundleResult) Err() error {
	if r.State != BundleMismatch {
		return nil
	}
	if r.Reason != "" {
		return stop(StatusFailGateBundle, "bundle manifest check failed: %s · install again", r.Reason)
	}
	return stop(StatusFailGateBundle, "bundle manifest check failed: %d file(s) differ from %s · install again", r.Mismatches, BundleManifestName)
}

// CheckBundle is the check itself, synchronous; ctx stops the hashing.
func CheckBundle(ctx context.Context, root string) BundleResult {
	// every return sets Finished itself (GL-13: a deferred assignment to the local did not reach
	// the returned copy, so the passed check's end and its diagnostic «ms» were lost)
	r := BundleResult{Root: root, Manifest: filepath.Join(root, BundleManifestName), Started: time.Now()}
	broken := func(format string, a ...any) BundleResult {
		r.State, r.Reason, r.Finished = BundleMismatch, fmt.Sprintf(format, a...), time.Now()
		return r
	}
	if !fileExists(r.Manifest) {
		r.State, r.Reason, r.Finished = BundleDev, "no "+BundleManifestName+" in "+root+": development build, bundle check skipped", time.Now()
		return r
	}
	v, err := readJSONFileStrict(r.Manifest)
	if err != nil {
		return broken("bundle manifest unreadable - %v", err)
	}
	m, _ := asObj(v)
	if ver, ok := jNonNegInt(m["bundle_manifest_version"]); !ok || ver != BundleManifestVersion {
		return broken("bundle_manifest_version missing or not an exact match")
	}
	files, ok := jArr(m["files"])
	if !ok || len(files) == 0 {
		return broken("bundle manifest files[] missing or empty")
	}
	type entry struct{ rel, sha string }
	var entries []entry
	for _, e := range files {
		eo, _ := asObj(e)
		rel, ok := jNonEmptyStr(eo["path"])
		if !ok {
			return broken("bundle manifest entry without path")
		}
		if !isSha256Hex(eo["sha256"]) {
			return broken("bundle manifest entry sha256 invalid: %s", rel)
		}
		if strings.Contains(rel, "..") {
			return broken("bundle manifest path escapes root: %s", rel)
		}
		entries = append(entries, entry{rel, strings.ToLower(eo["sha256"].(string))})
	}
	r.Files = len(entries)
	listed := map[string]bool{}
	for _, e := range entries {
		if ctx.Err() != nil {
			return broken("bundle check cancelled")
		}
		norm := strings.ReplaceAll(e.rel, "/", `\`)
		listed[strings.ToLower(norm)] = true
		abs := filepath.Join(root, norm)
		if !fileExists(abs) {
			r.Diffs = append(r.Diffs, BundleDiff{Path: e.rel, Kind: "missing"})
			continue
		}
		h, err := hashFileCtx(ctx, abs)
		if err != nil {
			if ctx.Err() != nil {
				return broken("bundle check cancelled")
			}
			r.Diffs = append(r.Diffs, BundleDiff{Path: e.rel, Kind: "hash-failed"})
			continue
		}
		if h != e.sha {
			r.Diffs = append(r.Diffs, BundleDiff{Path: e.rel, Kind: "hash"})
		}
	}
	var extra []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if p != root && bundleHidden(d) { // Get-ChildItem -Recurse -File without -Force (GL-13)
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		// BND-ROOT: WalkDir joins the children cleaned (backslashes on Windows) while root keeps
		// the spelling it was given, so a prefix cut left C:/... roots absolute; Rel cleans both.
		rel, err := filepath.Rel(root, p)
		if err != nil {
			rel = p
		}
		rel = strings.ReplaceAll(rel, "/", `\`)
		if strings.EqualFold(rel, BundleManifestName) {
			return nil
		}
		if !listed[strings.ToLower(rel)] {
			extra = append(extra, rel)
		}
		return nil
	})
	sort.Strings(extra)
	for _, x := range extra {
		r.Diffs = append(r.Diffs, BundleDiff{Path: x, Kind: "unlisted"})
	}
	r.Mismatches = len(r.Diffs)
	r.State = BundleOK
	if r.Mismatches > 0 {
		r.State = BundleMismatch
	}
	r.Finished = time.Now()
	return r
}

// bundleCheck is the one background check of a Launcher.
type bundleCheck struct {
	once    sync.Once
	started atomic.Bool
	done    chan struct{} // made in New
	res     BundleResult
}

func newBundleCheck() *bundleCheck { return &bundleCheck{done: make(chan struct{})} }

// StartBundleCheck starts the bundle check in the background (once per Launcher; a second
// call does nothing). Call it as soon as the screen is up.
func (l *Launcher) StartBundleCheck() {
	l.bundle.once.Do(func() {
		l.bundle.started.Store(true)
		go func() {
			r := CheckBundle(context.Background(), l.opt.BundleRoot)
			l.bundle.res = r
			l.diag.write("BUNDLE_CHECK", map[string]any{"state": r.State.String(), "files": r.Files,
				"mismatches": r.Mismatches, "diffs": r.Diffs, "reason": r.Reason,
				"ms": r.Finished.Sub(r.Started).Milliseconds()})
			close(l.bundle.done)
		}()
	})
}

// BundleCheckResult is the check's result without waiting (State BundlePending while it runs,
// and before StartBundleCheck).
func (l *Launcher) BundleCheckResult() BundleResult {
	if !l.bundle.started.Load() {
		return BundleResult{State: BundlePending, Root: l.opt.BundleRoot}
	}
	select {
	case <-l.bundle.done:
		return l.bundle.res
	default:
		return BundleResult{State: BundlePending, Root: l.opt.BundleRoot}
	}
}

// WaitBundleCheck starts the check when it has not started and waits for it (a start waits
// for it to finish, conflict 18-2); ctx ends the wait, not the check.
func (l *Launcher) WaitBundleCheck(ctx context.Context) (BundleResult, error) {
	l.StartBundleCheck()
	select {
	case <-l.bundle.done:
		return l.bundle.res, nil
	case <-ctx.Done():
		return BundleResult{State: BundlePending, Root: l.opt.BundleRoot}, ctx.Err()
	}
}
