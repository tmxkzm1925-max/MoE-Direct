//go:build !windows

package core

import "testing"

// la14Live needs the Windows job object and console of the live start (s8la_gl07_windows_test.go).
func la14Live(t *testing.T) { t.Skip("the live start is Windows-only") }
