//go:build !windows

package core

import "testing"

func la25Live(t *testing.T) { t.Skip("the live start is Windows-only") }
