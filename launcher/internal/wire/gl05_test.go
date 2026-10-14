package wire

import (
	"testing"

	"moedirect/launcher/internal/core"
)

// TestGL05NeedLabel is the wire side of S8-LA-13 (GL-05): the core label of the shown need
// reaches the screen as its word; the projection of a measured key and «unmeasured» carry
// none (the value itself says «not measured yet»).
func TestGL05NeedLabel(t *testing.T) {
	for l, want := range map[core.NeedLabel]string{
		core.NeedMeasured: "measured", core.NeedUnverified: "predicted before run", // UX-R10 (finding 16 (e)): the softened words
		core.NeedProjected: "", core.NeedUnmeasured: "",
	} {
		if got := needLabel(l); got != want {
			t.Fatalf("%v: %q, want %q", l, got, want)
		}
	}
}
