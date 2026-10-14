//go:build !windows

package core

import "testing"

// pf16Live needs the Windows job object of the live start (pf_gl21_windows_test.go).
func pf16Live(t *testing.T) { t.Skip("the live start is Windows-only") }
