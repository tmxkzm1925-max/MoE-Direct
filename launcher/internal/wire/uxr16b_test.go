package wire

import (
	"math"
	"testing"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// TestUXR16bWireFreeStandby is UX-R16b's wire side: «RAM free (now)» is the available figure,
// free the free-and-zero list, standby the rest (their sum the available figure); without the
// lists standby is not measured, never zero (map.go memoryFrom).
func TestUXR16bWireFreeStandby(t *testing.T) {
	pc := core.PCMemory{TotalMiB: 32768, AvailMiB: 7168, UsedMiB: 25600, ServerMiB: 19456, OthersMiB: 6144,
		ListsOK: true, Lists: core.MemoryLists{ModifiedMiB: 1024, FreeZeroMiB: 1024}}
	r := memoryFrom(pc, true, 12288, core.VRAMInfo{}, serverGPU{})
	if !r.ReadyNowGB.Ok() || !r.FreeGB.Ok() || !r.StandbyGB.Ok() || math.Abs(r.ReadyNowGB.V-7.0) > 0.05 ||
		math.Abs(r.FreeGB.V-1.0) > 0.05 || math.Abs(r.FreeGB.V+r.StandbyGB.V-r.ReadyNowGB.V) > 0.05 {
		t.Fatalf("lists read: ready %+v free %+v standby %+v, want 7.0 = 1.0 + 6.0", r.ReadyNowGB, r.FreeGB, r.StandbyGB)
	}
	pc.ListsOK = false
	r = memoryFrom(pc, true, 12288, core.VRAMInfo{}, serverGPU{})
	if !r.ReadyNowGB.Ok() || r.StandbyGB.State != ui.Unmeasured {
		t.Fatalf("no lists: ready %+v standby %+v, want 7.0 and standby not measured", r.ReadyNowGB, r.StandbyGB)
	}
}
