// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import "errors"

// MemoryLists are two of Windows' memory lists, read from PDH (LFS:302; U-21, L:4277): the
// modified page list (written to disk before its pages can be reused) and the free and zero
// page list. Standby is the available memory less the free and zero list.
type MemoryLists struct {
	ModifiedMiB int64 // \Memory\Modified Page List Bytes
	FreeZeroMiB int64 // \Memory\Free & Zero Page List Bytes
}

// readMemoryLists reads the lists; ok is false when the reader is missing or fails.
func (l *Launcher) readMemoryLists() (MemoryLists, bool) {
	if l.os.memLists == nil {
		return MemoryLists{}, false
	}
	ml, err := l.os.memLists()
	if err != nil || ml.ModifiedMiB < 0 || ml.FreeZeroMiB < 0 {
		return MemoryLists{}, false
	}
	return ml, true
}

// MemoryLine is the «left» part of the memory line (F-BU B.4 and G; design decision ⑧,
// L:4183; decision ②, L:4178): free RAM now, the modified memory and the need GL-05 computes,
// all in MiB.
type MemoryLine struct {
	FreeNow, Modified, Need int64
	ModifiedOK              bool
	NeedLabel               NeedLabel
}

// Left is free RAM now − the need (G.1); false when the need is «unmeasured» (G.3).
func (m MemoryLine) Left() (int64, bool) {
	if m.NeedLabel == NeedUnmeasured {
		return 0, false
	}
	return m.FreeNow - m.Need, true
}

// LeftIfWrittenFirst is free RAM now + the modified memory − the need (G.2): what is left
// once the modified pages are written to disk first; false when the need is «unmeasured» or
// the modified memory was not read.
func (m MemoryLine) LeftIfWrittenFirst() (int64, bool) {
	if m.NeedLabel == NeedUnmeasured || !m.ModifiedOK {
		return 0, false
	}
	return m.FreeNow + m.Modified - m.Need, true
}

// OpenTaskManager opens Windows Task Manager without a selection (U-35 (i), L:4275): Task
// Manager takes no process to select, so the screen's line names the app and its PID.
func (l *Launcher) OpenTaskManager() error {
	if l.os.taskmgr == nil {
		return errors.New("no Task Manager opener on this system")
	}
	err := l.os.taskmgr()
	rec := map[string]any{"selection": "none"}
	if err != nil {
		rec["error"] = err.Error()
	}
	l.diag.write("TASK_MANAGER", rec)
	return err
}

// serverPID is the PID of this launcher's serving server (0 when none serves).
func (l *Launcher) serverPID() int {
	l.mu.Lock()
	s := l.sess
	l.mu.Unlock()
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.child == nil || s.ended {
		return 0
	}
	return s.child.PID()
}
