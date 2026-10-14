// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"os/exec"
	"syscall"
)

// hideWindow runs a tool with no console window of its own (CREATE_NO_WINDOW).
func hideWindow(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
}
