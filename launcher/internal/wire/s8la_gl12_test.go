package wire

// GL-12 (version 8): the adapter's part of S8-LA-26 (plan reviews/_plan_8_STAGE1_v8-plansplit.md
// at d1dbe526f; F-LS edition reviews/_spec_8_F-LS_launcher_go_edition.md at d3a5ab2ed, 4.7 and
// CF 17): no network option exists, not even a locked row or a launcher-settings field; the
// one address shown is 127.0.0.1.

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

func TestS8LA26(t *testing.T) {
	catalog := filepath.Join("..", "..", "..", "launcher", "models.json")
	l, err := core.New(core.Options{CatalogPath: catalog, BundleRoot: filepath.Join("..", "..", "..", "..", "repack"), StateDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	b := New(l, catalog, "test")
	cat, err := core.ReadCatalog(catalog, filepath.Join("..", "..", "..", "..", "repack"))
	if err != nil {
		t.Fatal(err)
	}
	words := []string{"host", "network", "bind", "loopback", "lan", "listen"}
	for _, p := range cat.Profiles {
		path := filepath.Join(t.TempDir(), p.ID+".gguf")
		b.files[path] = core.ModelFile{Path: path, Name: p.ID + ".gguf", ProfileID: p.ID}
		v, err := b.ModelSettings(path)
		if err != nil {
			t.Fatalf("%s: %v", p.ID, err)
		}
		if len(v.Rows) == 0 {
			t.Fatalf("%s: no rows", p.ID)
		}
		for _, r := range v.Rows {
			for _, w := range words {
				if strings.Contains(strings.ToLower(r.Key+" "+r.Label), w) {
					t.Errorf("%s: row %q %q", p.ID, r.Key, r.Label)
				}
			}
		}
	}
	// The launcher settings carry no network field; the address is this PC's.
	ty := reflect.TypeOf(ui.LauncherInfo{})
	for i := 0; i < ty.NumField(); i++ {
		if n := strings.ToLower(ty.Field(i).Name); strings.Contains(n, "network") || strings.Contains(n, "bind") || strings.Contains(n, "host") {
			t.Errorf("LauncherInfo field %s", ty.Field(i).Name)
		}
	}
	info, err := b.Launcher()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(info.Address, core.BindHost+" ") {
		t.Errorf("address %q", info.Address)
	}
}
