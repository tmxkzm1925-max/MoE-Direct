package main

import (
	"io"
	"os"
	"path/filepath"
	"testing"
)

// TestS8LA31CatalogDefault (GL-14): a release binary reads the catalog beside itself at the
// bundle root (make_bundle.ps1:17-20; Read-Catalog PS:2288-2291), not a path of this PC.
func TestS8LA31CatalogDefault(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(filepath.Dir(exe), "models.json")
	if got := defaultCatalog(); got != want {
		t.Fatalf("default catalog %q, want %q", got, want)
	}
	o, _, ok := parse(nil, io.Discard)
	if !ok || o.catalog != want {
		t.Fatalf("-catalog default %q, want %q", o.catalog, want)
	}
	o, _, ok = parse([]string{"-catalog", `X:\other\models.json`}, io.Discard)
	if !ok || o.catalog != `X:\other\models.json` {
		t.Fatalf("-catalog given: %q", o.catalog)
	}
}
