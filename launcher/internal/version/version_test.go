package version

import (
	"os"
	"testing"
)

// TestS8LA30 (S8-LA-30, this package's part): the version value is the release version
// v0.3.2, the gate refuses any other bundle version, and when the build command hands a
// bundle version in BundleEnv the gate holds for it (the build fails otherwise). The log
// header and the about text are checked by TestS8LA30 in internal/core and internal/ui.
func TestS8LA30(t *testing.T) {
	const release = "v0.3.2" // S8-LA-30: string equality with the release version
	if LauncherVersion != release {
		t.Fatalf("LauncherVersion = %q, want the release version %q", LauncherVersion, release)
	}
	if LauncherName != "MoE-Direct-launcher" {
		t.Fatalf("LauncherName = %q, want MoE-Direct-launcher ([A-BIN])", LauncherName)
	}
	if got, want := About(), "MoE-Direct-launcher v0.3.2"; got != want {
		t.Fatalf("About() = %q, want %q", got, want)
	}
	if err := CheckBundle(release); err != nil {
		t.Fatalf("CheckBundle(%q) = %v, want nil", release, err)
	}
	for _, other := range []string{"", "v0.3.1-dev", "v0.3.1", "v0.3.3", "0.3.2", "v0.3.2 ", "V0.3.2", "v0.2"} {
		if CheckBundle(other) == nil {
			t.Errorf("CheckBundle(%q) = nil, want the version gate to refuse it", other)
		}
	}
	if bundle, ok := os.LookupEnv(BundleEnv); ok {
		if err := CheckBundle(bundle); err != nil {
			t.Fatalf("version gate (%s=%q): %v", BundleEnv, bundle, err)
		}
		t.Logf("version gate: %s=%q equals LauncherVersion %q", BundleEnv, bundle, LauncherVersion)
	}
}
