package core

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Review item 31 (reviews/_gol_review_8_user_20261010.md): the screen re-reads the model list
// every five seconds while 1 Models is shown. ScanStamp is the cheap half of that re-read: it
// stats the files the list depends on (names, sizes and modification times only, no header
// read, no hashing) and returns a fingerprint. When the fingerprint is unchanged the caller may
// keep its last ListModels result; when a GGUF or a repack output was added, deleted or
// rewritten, the fingerprint changes and the caller lists again.

// repackStampFiles are the setup outputs whose presence decides a row's setup state.
var repackStampFiles = []string{
	filepath.Join("repack", "experts.bin"), filepath.Join("repack", "manifest.json"),
	filepath.Join("repack", "verify_report.json"), filepath.Join("repack", "experts.bin.partial"),
	filepath.Join("repack-virtual", "manifest.json"),
}

// listRoots are the folders ListModels scans: -ModelsRoot, then the fixed drives' model folders.
func (l *Launcher) listRoots() []string {
	var roots []string
	if l.opt.ModelsRoot != "" {
		r := strings.Trim(strings.TrimSpace(l.opt.ModelsRoot), `"`)
		if st, err := os.Stat(r); err == nil && st.IsDir() {
			if abs, err := filepath.Abs(r); err == nil {
				roots = append(roots, normalizeScanRoot(abs))
			}
		}
	}
	return append(roots, l.os.modelRoots()...)
}

func stampOf(path string) string {
	st, err := os.Stat(path)
	if err != nil {
		return "-"
	}
	return fmt.Sprintf("%d|%d", st.Size(), st.ModTime().UnixNano())
}

// ScanStamp fingerprints what the model list is built from: every .gguf under the model folders
// (each shard, to the scan depth), the setup outputs beside them, the catalog, the history and
// the digest caches.
func (l *Launcher) ScanStamp() string {
	var lines []string
	dirs := map[string]bool{}
	for _, root := range l.listRoots() {
		r := normalizeScanRoot(root)
		lines = append(lines, "root "+strings.ToLower(r))
		if st, err := os.Stat(r); err != nil || !st.IsDir() {
			continue
		}
		_ = filepath.WalkDir(r, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			rel := strings.TrimLeft(strings.TrimPrefix(p, r), `\`)
			depth := len(strings.Split(rel, `\`))
			if d.IsDir() {
				if p != r && (depth >= scanMaxDepth || isHidden(p)) {
					return fs.SkipDir
				}
				return nil
			}
			if depth > scanMaxDepth || !strings.EqualFold(filepath.Ext(p), ".gguf") || isHidden(p) {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return nil
			}
			lines = append(lines, fmt.Sprintf("gguf %s|%d|%d", strings.ToLower(p), info.Size(), info.ModTime().UnixNano()))
			dirs[filepath.Dir(p)] = true
			// GL-14 (watch.go): a file still being written is held back from the list; the line
			// goes when it settles, so the stamp changes and the list is read again.
			if st, err := os.Stat(p); err == nil && !l.settled(p, st.Size(), st.ModTime()) {
				lines = append(lines, "pending "+strings.ToLower(p))
			}
			return nil
		})
	}
	for d := range dirs {
		for _, f := range repackStampFiles {
			lines = append(lines, "out "+strings.ToLower(filepath.Join(d, f))+" "+stampOf(filepath.Join(d, f)))
		}
	}
	for _, p := range []string{l.opt.CatalogPath, l.historyPath(), l.psShardCachePath(), l.ownShardCachePath(),
		filepath.Join(l.opt.StateDir, "arch_template_pref.json")} {
		lines = append(lines, "file "+strings.ToLower(p)+" "+stampOf(p))
	}
	sort.Strings(lines)
	h := sha256.Sum256([]byte(strings.Join(lines, "\n")))
	return hex.EncodeToString(h[:])
}
