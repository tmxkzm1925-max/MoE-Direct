//go:build windows

package core

import (
	"os"
	"strings"
	"testing"
)

// TestOSGPUProcessMemoryLive reads the real counter (read only; no process is touched). It
// runs only with MOE_LIVE_PDH=1 so the unit run does not depend on the GPU driver.
func TestOSGPUProcessMemoryLive(t *testing.T) {
	if os.Getenv("MOE_LIVE_PDH") != "1" {
		t.Skip(`set MOE_LIVE_PDH=1 to read \GPU Process Memory(*)\Dedicated Usage`)
	}
	inst, err := osGPUProcessMemory()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for name, v := range inst {
		if !strings.HasPrefix(name, "pid_") || v < 0 {
			t.Errorf("instance %q = %d", name, v)
		}
		if v > 0 {
			n++
		}
	}
	t.Logf("%d instances, %d with dedicated memory in use", len(inst), n)
	if len(inst) == 0 {
		t.Error("no instance")
	}
}
