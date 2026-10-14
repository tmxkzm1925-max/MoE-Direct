//go:build windows

package core

import (
	"io/fs"
	"syscall"
)

// bundleHidden: the entry has the Hidden attribute, which Get-ChildItem without -Force leaves
// out (a hidden folder with everything under it; a System-only file is still listed).
func bundleHidden(d fs.DirEntry) bool {
	fi, err := d.Info()
	if err != nil {
		return false
	}
	a, ok := fi.Sys().(*syscall.Win32FileAttributeData)
	return ok && a.FileAttributes&syscall.FILE_ATTRIBUTE_HIDDEN != 0
}
