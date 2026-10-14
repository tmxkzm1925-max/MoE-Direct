package core

import (
	"os"
	"strings"
	"sync"
	"time"
)

// GL-14 (S8-LA-31; F-LS LFS Q5.3): the folder watch of the model list. The screen re-reads the
// list every five seconds while 1 Models is shown (review item 31, ScanStamp). Two rules make
// that re-read a watch:
//   - a model folder that does not exist yet is polled: the roots are resolved again at every
//     ScanStamp and ListModels call, so a folder created later (a -models-root given before it
//     exists, or a drive's moe-models folder) enters the list at the next re-read;
//   - a .gguf still being written (a download, a copy) is not listed until its size stops
//     changing: a file is held back until GrowSettle has passed since the later of its last
//     write and the first look that saw its present size. Every look (ScanStamp, ListModels)
//     reads the same record, so two looks in one re-read agree. ScanStamp carries a «pending»
//     line for each held file, so the stamp changes when the file settles and the list is read
//     again.

// GrowSettle is how long a .gguf's size must stay the same, and its last write lie back, before
// the list shows it.
const GrowSettle = 10 * time.Second

type growSeen struct {
	size  int64
	since time.Time // the first look at this size (a first look takes the file's last write)
}

// growWatch remembers the size of every .gguf the looks saw.
type growWatch struct {
	mu   sync.Mutex
	seen map[string]growSeen
}

// since records a look at path and gives the time its present size was first seen.
func (g *growWatch) since(path string, size int64, mtime, now time.Time) time.Time {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.seen == nil {
		g.seen = map[string]growSeen{}
	}
	k := strings.ToLower(path)
	s, ok := g.seen[k]
	switch {
	case !ok:
		s = growSeen{size, mtime}
	case s.size != size:
		s = growSeen{size, now}
	}
	g.seen[k] = s
	return s.since
}

// settled reports whether one .gguf file has stopped growing.
func (l *Launcher) settled(path string, size int64, mtime time.Time) bool {
	now := l.os.now()
	since := l.grow.since(path, size, mtime, now)
	if mtime.After(since) {
		since = mtime
	}
	return now.Sub(since) >= GrowSettle
}

// settledSet reports whether every existing file of path's shard set has stopped growing.
func (l *Launcher) settledSet(path string) bool {
	for _, p := range siblingPaths(path) {
		st, err := os.Stat(p)
		if err != nil {
			continue
		}
		if !l.settled(p, st.Size(), st.ModTime()) {
			return false
		}
	}
	return true
}
