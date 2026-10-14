package wire

import (
	"math"
	"testing"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// TestS8LA22 is S8-LA-22's wire side (GL-09; U-21, L:4277; F-BU G; U-35, L:4275): the memory
// figures from the PDH lists, apps without the modified memory, the «left» figures by design
// decision ⑧ and «unmeasured» with the need; the serving app with its PID.
func TestS8LA22(t *testing.T) {
	res := &core.PreflightResult{FreeMiB: 8192, FreeForStartMiB: 8192, NeedKnown: true, NeedMiB: 6144,
		NeedBasis: core.NeedFromProjection, NeedLabel: core.NeedProjected, ListsOK: true,
		Lists: core.MemoryLists{ModifiedMiB: 1536, FreeZeroMiB: 2048}}
	res.RAM.TotalMiB = 32768
	p, _ := preflightFrom("m", res, nil)
	for _, c := range []struct {
		name string
		v    ui.Val
		want float64
	}{
		{"modified", p.ModifiedGB, 1.5}, {"standby", p.StandbyGB, 6.0}, {"free", p.FreeGB, 2.0},
		{"apps", p.AppsGB, 22.5}, {"left", p.LeftGB, 0}, {"left if written first", p.LeftIfWrittenGB, 3.5},
	} {
		if c.name == "left" {
			continue // LeftGB is the check's (GL-05)
		}
		if !c.v.Ok() || math.Abs(c.v.V-c.want) > 0.05 {
			t.Errorf("%s %+v, want %v", c.name, c.v, c.want)
		}
	}
	// an unmeasured need: no left figure; no lists read: the figures stay unmeasured
	res.NeedKnown, res.NeedMiB, res.NeedLabel = false, 0, core.NeedUnmeasured
	if p, _ := preflightFrom("m", res, nil); p.LeftIfWrittenGB.Ok() || p.LeftGB.Ok() {
		t.Errorf("unmeasured need: %+v %+v", p.LeftIfWrittenGB, p.LeftGB)
	}
	res.ListsOK = false
	if p, _ := preflightFrom("m", res, nil); p.ModifiedGB.Ok() || p.LeftIfWrittenGB.Ok() {
		t.Errorf("no lists: %+v", p.ModifiedGB)
	}
	// while serving: the server's PID on its app row, the lists in the reading
	pc := core.PCMemory{TotalMiB: 32768, AvailMiB: 7168, UsedMiB: 25600, ServerMiB: 19456, OthersMiB: 6144, ServerPID: 4242,
		ListsOK: true, Lists: core.MemoryLists{ModifiedMiB: 1024, FreeZeroMiB: 1024}}
	r := memoryFrom(pc, true, 12288, core.VRAMInfo{}, serverGPU{})
	if len(r.Apps) != 1 || r.Apps[0].PID != 4242 || r.Apps[0].Name != "llama-server.exe" {
		t.Fatalf("apps %+v", r.Apps)
	}
	if math.Abs(r.ModifiedGB.V-1.0) > 0.05 || math.Abs(r.StandbyGB.V-6.0) > 0.05 || math.Abs(r.AppsGB.V-5.0) > 0.05 {
		t.Fatalf("serving lists: %+v %+v %+v", r.ModifiedGB, r.StandbyGB, r.AppsGB)
	}
}
