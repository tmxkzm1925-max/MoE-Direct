package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// HTTPRead is one engine endpoint read.
type HTTPRead struct {
	OK   bool
	Code int
	Body any // decoded JSON, nil when the body was not JSON
	Err  string
}

// ServeStatus is the serving server as the launcher sees it.
type ServeStatus struct {
	Running      bool // started, not ended
	Ready        bool
	PID          int
	Alive        bool
	Host         string
	Port         int
	Address      string
	ModelPath    string
	ProfileID    string
	SettingsLine string
	StartedAt    time.Time
	ReadyAt      time.Time
	Uptime       time.Duration // since ready
	ReadyIn      time.Duration // spawn to ready
	Health       HTTPRead      // GET /health
	Props        HTTPRead      // GET /props
	Slots        HTTPRead      // GET /slots
	Memory       ProcessMemory
	ErrLog       string
	OutLog       string
	LogTail      []string
	End          *RunEnd // how it ended, when it has
	// GOL-FIX1: what this start actually used (review item 2; Alias is item 27) and the
	// print_timing summary so far (items 9, 23).
	Start  StartSettings
	Timing TimingSummary
	// GL-21: the engine's prefetch values after ready (F-PF §8.5); nil before ready
	Engine *EnginePrefetch
}

// statusLogLines is how many server log lines Status carries.
const statusLogLines = 20

func getJSON(ctx context.Context, url string, timeout time.Duration) HTTPRead {
	c, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(c, http.MethodGet, url, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return HTTPRead{Err: err.Error()}
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	r := HTTPRead{Code: resp.StatusCode, OK: resp.StatusCode == 200}
	var v any
	if json.Unmarshal(b, &v) == nil {
		r.Body = v
	}
	return r
}

// Status reads the running server: the child, the address, /health, /props, /slots and the
// server log tail. Every endpoint read has the 5 s timeout of Test-HealthPidBound.
func (l *Launcher) Status(ctx context.Context) (ServeStatus, error) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return ServeStatus{}, ErrNoServer
	}
	s.mu.Lock()
	st := ServeStatus{Running: !s.ended, Ready: s.ready, StartedAt: s.spawnedAt, ReadyAt: s.readyAt}
	if s.ended {
		e := s.end
		st.End = &e
	}
	if s.pre != nil && s.pre.Config != nil {
		st.Host, st.Port, st.Address = s.pre.Config.Host, s.pre.Config.Port, s.pre.Config.Address()
		st.ModelPath, st.SettingsLine, st.Start = s.pre.ModelPath, s.pre.SettingsLine, s.pre.Start
		st.Engine = s.engine
		if s.pre.Profile != nil {
			st.ProfileID = s.pre.Profile.ID
		}
	}
	child := s.child
	ended, ready := s.ended, s.ready
	lg := s.log
	s.mu.Unlock()
	if lg != nil {
		st.Timing = lg.summary()
	}
	if ready {
		st.ReadyIn = st.ReadyAt.Sub(st.StartedAt)
		if !ended {
			st.Uptime = l.os.now().Sub(st.ReadyAt)
		}
	}
	if child != nil {
		st.PID, st.ErrLog, st.OutLog = child.PID(), child.ErrLog(), child.OutLog()
		if !ended {
			if _, exited := child.Exited(); !exited {
				st.Alive = true
				st.Memory, _ = child.Memory()
			}
		}
		st.LogTail = tailLines(st.ErrLog, statusLogLines)
	}
	if st.Alive && st.Port > 0 {
		base := fmt.Sprintf("http://%s:%d", st.Host, st.Port)
		var wg sync.WaitGroup
		for _, ep := range []struct {
			path string
			dst  *HTTPRead
		}{{"/health", &st.Health}, {"/props", &st.Props}, {"/slots", &st.Slots}} {
			wg.Add(1)
			go func(path string, dst *HTTPRead) {
				defer wg.Done()
				*dst = getJSON(ctx, base+path, psReadyTimings.HealthTimeout)
			}(ep.path, ep.dst)
		}
		wg.Wait()
	}
	return st, nil
}

// LogTail is the last n lines of the server's stderr log (where llama-server logs), oldest
// first; complete lines only, like Read-ChildStderrComplete.
func (l *Launcher) LogTail(n int) ([]string, error) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return nil, ErrNoServer
	}
	s.mu.Lock()
	child := s.child
	s.mu.Unlock()
	if child == nil {
		return nil, nil
	}
	return tailLines(child.ErrLog(), n), nil
}

func tailLines(path string, n int) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil
	}
	const window = 256 << 10
	off := int64(0)
	if st.Size() > window {
		off = st.Size() - window
	}
	b := make([]byte, st.Size()-off)
	if _, err := f.ReadAt(b, off); err != nil && err != io.EOF {
		return nil
	}
	raw := string(b)
	idx := strings.LastIndex(raw, "\n")
	if idx < 0 {
		return nil
	}
	lines := strings.Split(raw[:idx], "\n")
	if off > 0 && len(lines) > 0 {
		lines = lines[1:] // the first line may be cut
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	for i := range lines {
		lines[i] = strings.TrimSuffix(lines[i], "\r")
	}
	return lines
}

// LastEnd is how the current (or last) server's run ended (GL-08: the exit code a quit from the
// stopped page keeps, U-32 (ii)); false while it runs and before the first start.
func (l *Launcher) LastEnd() (RunEnd, bool) {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return RunEnd{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.end, s.ended
}

// Done is closed when the current (or last) server has ended; nil before the first start.
func (l *Launcher) Done() <-chan struct{} {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.sess == nil {
		return nil
	}
	return l.sess.done
}
