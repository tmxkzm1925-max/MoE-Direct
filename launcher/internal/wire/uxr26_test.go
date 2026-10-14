package wire

// UX-R26: the processing tile's batch size is the -b of the argv the server was started with
// (Backend.Status, batchOf). Every catalog profile's argv yields its -b; an argv without one, or
// with no number after it, yields «not read» (the tile then shows «—»).

import (
	"encoding/json"
	"os"
	"strconv"
	"testing"

	"moedirect/launcher/internal/ui"
)

func TestUXR26Batch(t *testing.T) {
	raw, err := os.ReadFile("../../../launcher/models.json")
	if err != nil {
		t.Fatal(err)
	}
	var cat struct {
		Profiles []struct {
			ID       string `json:"profile_id"`
			Defaults struct {
				Argv []string `json:"argv"`
			} `json:"defaults"`
		} `json:"profiles"`
	}
	if err := json.Unmarshal(raw, &cat); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, p := range cat.Profiles {
		want := ui.U
		for i := 0; i+1 < len(p.Defaults.Argv); i++ {
			if p.Defaults.Argv[i] == "-b" {
				v, _ := strconv.Atoi(p.Defaults.Argv[i+1])
				want = ui.K(float64(v))
				n++
			}
		}
		if got := batchOf(p.Defaults.Argv); got != want {
			t.Errorf("%s: batch %+v, want %+v", p.ID, got, want)
		}
	}
	if n == 0 {
		t.Fatalf("no catalog profile has -b")
	}
	for _, c := range []struct {
		argv []string
		want ui.Val
	}{
		{[]string{"-c", "4096", "-b", "512", "-ub", "256"}, ui.K(512)},
		{[]string{"-c", "4096", "-ub", "256"}, ui.U},
		{[]string{"-c", "4096", "-b"}, ui.U},
		{[]string{"-b", "x"}, ui.U},
		{nil, ui.U},
	} {
		if got := batchOf(c.argv); got != c.want {
			t.Errorf("%v: batch %+v, want %+v", c.argv, got, c.want)
		}
	}
}
