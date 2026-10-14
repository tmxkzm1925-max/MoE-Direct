//go:build !windows

package core

import (
	"io/fs"
	"strings"
)

// bundleHidden: a dot name, which PowerShell's Get-ChildItem without -Force leaves out off
// Windows.
func bundleHidden(d fs.DirEntry) bool { return strings.HasPrefix(d.Name(), ".") }
