package core

import (
	"fmt"
	"sync"
	"time"
)

// Conflict 6 of reviews/_gol_conflicts_8_user_20261010.md: one launcher event per server stop
// for the 4 Log page - the clock time, the result (clean stop 0, forced stop 5, stopped before
// ready 2, any other failure), the launcher's code and its meaning, and the server child's own
// exit code. The code is the PowerShell STATUS_EXIT code of the run's status (PS:203-221); the
// launcher process's own exit code stays that table too (the argument path, ExitCodeOf).

// StopClass is the result class of one server stop.
type StopClass uint8

const (
	StopClean       StopClass = iota // ✓ green: stopped without force, code 0
	StopForced                       // ▲ yellow: the forced kill was used, code 5
	StopBeforeReady                  // stopped before the server was ready, code 2
	StopFailed                       // ✕ red: any other failure (its own code)
)

func (c StopClass) String() string {
	return [...]string{"clean stop", "forced stop", "stopped before ready", "failed"}[c]
}

// ServerStopEvent is the launcher event of one server stop.
type ServerStopEvent struct {
	At        time.Time
	Class     StopClass
	Status    Status
	Code      int     // the launcher's code (PowerShell table)
	Meaning   string  // what the code means
	ChildExit *uint32 // the server child's own exit code; nil when it could not be read
	Reason    string  // the run's end reason
	WasReady  bool    // the server had reached ready
	PID       int
}

// Line is the event as one log line (English; the screen words are the UI's).
func (e ServerStopEvent) Line() string {
	return fmt.Sprintf("%s  server stop · %s · launcher code %d (%s: %s) · server exit %s",
		e.At.Local().Format("15:04:05"), e.Class, e.Code, e.Status, e.Meaning, FormatExitCode(e.ChildExit))
}

// stopMeaning is the meaning of a run's end status.
func stopMeaning(s Status) string {
	switch s {
	case StatusOK, StatusOKSmoke:
		return "the server stopped cleanly (CTRL_BREAK, exit 0)"
	case StatusCancelledUser:
		return "stopped before the server was ready"
	}
	if h := StatusHint(s); h != "" {
		return h
	}
	return "unknown status"
}

// ClassifyStop makes the event of a run's end.
func ClassifyStop(end RunEnd, wasReady bool, pid int) ServerStopEvent {
	e := ServerStopEvent{At: end.At, Status: end.Status, Code: ExitCode(end.Status), Meaning: stopMeaning(end.Status),
		Reason: end.Reason, WasReady: wasReady, PID: pid}
	if end.Teardown != nil {
		e.ChildExit = end.Teardown.ExitCode
	}
	switch {
	case end.Status == StatusOK || end.Status == StatusOKSmoke:
		e.Class = StopClean
	case end.Status == StatusCancelledUser:
		e.Class = StopBeforeReady
	case end.Status == StatusFailTeardown && end.Teardown != nil && end.Teardown.TaskkillUsed:
		e.Class = StopForced
		e.Meaning = "the server did not stop within the grace period and was killed (taskkill); " + StatusHint(end.Status)
	default:
		e.Class = StopFailed
	}
	return e
}

// stopLog keeps this launcher's stop events, oldest first.
type stopLog struct {
	mu     sync.Mutex
	events []ServerStopEvent
}

func (l *Launcher) recordStop(s *session, end RunEnd) ServerStopEvent {
	s.mu.Lock()
	ready := s.ready
	pid := 0
	if s.child != nil {
		pid = s.child.PID()
	}
	s.mu.Unlock()
	e := ClassifyStop(end, ready, pid)
	l.stops.mu.Lock()
	l.stops.events = append(l.stops.events, e)
	l.stops.mu.Unlock()
	l.diag.write("SERVER_STOP", map[string]any{"class": e.Class.String(), "status": e.Status, "code": e.Code,
		"meaning": e.Meaning, "child_exit": e.ChildExit, "reason": e.Reason, "was_ready": e.WasReady, "pid": e.PID})
	return e
}

// StopEvents are the stop events of this launcher's servers, oldest first.
func (l *Launcher) StopEvents() []ServerStopEvent {
	l.stops.mu.Lock()
	defer l.stops.mu.Unlock()
	return append([]ServerStopEvent(nil), l.stops.events...)
}
