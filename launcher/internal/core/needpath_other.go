//go:build !windows

// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import "os/exec"

func hideWindow(*exec.Cmd) {}
