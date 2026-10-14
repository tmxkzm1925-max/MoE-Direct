// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

//go:build !windows

package ui

import (
	"os"
	"time"

	"github.com/charmbracelet/x/term"
)

// ttyConsole: the launcher is a Windows program; elsewhere the measurement
// reads the raw terminal bytes as reply records and keeps no user keys.
type ttyConsole struct{ st *term.State }

func openProbeConsole() probeConsole { return &ttyConsole{} }

func (c *ttyConsole) Modes() (uint32, uint32, error) {
	if _, err := term.GetState(os.Stdin.Fd()); err != nil {
		return 0, 0, err
	}
	return 0, 0, nil
}

func (c *ttyConsole) SetIn(mode uint32) error {
	if mode == probeInMode {
		st, err := term.MakeRaw(os.Stdin.Fd())
		c.st = st
		return err
	}
	if c.st != nil {
		_ = os.Stdin.SetReadDeadline(time.Time{})
		err := term.Restore(os.Stdin.Fd(), c.st)
		c.st = nil
		return err
	}
	return nil
}

func (c *ttyConsole) SetOut(uint32) error { return nil }

func (c *ttyConsole) Write(b []byte) (int, error) { return os.Stdout.Write(b) }

func (c *ttyConsole) Read(deadline time.Time) ([]ConRecord, error) {
	if err := os.Stdin.SetReadDeadline(deadline); err != nil {
		return nil, err
	}
	b := make([]byte, 256)
	n, err := os.Stdin.Read(b)
	var out []ConRecord
	for _, ch := range string(b[:n]) {
		out = append(out, ConRecord{Key: true, Down: true, Char: ch})
	}
	return out, err
}

func (c *ttyConsole) Forward([]ConRecord) error { return nil }
