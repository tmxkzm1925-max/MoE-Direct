// Package version is the one place the launcher's name and release version come from
// (GL-01; F-LS edition 5.1 and 12.6; S8-LA-30). Every other part of the launcher (the
// log header in internal/core, the about text on the help page in internal/ui, the flag
// set in cmd/MoE-Direct-launcher) reads these constants, and the build command
// (build.ps1) refuses to build when LauncherVersion and the bundle version it is given
// differ: it runs TestS8LA30 of this package with the bundle version in BundleEnv before
// it compiles (the Go counterpart of the PowerShell literal Start-MoeDirect.ps1:475-478,
// which stays in the retired file).
package version

import "fmt"

const (
	// LauncherName is the binary and entry name (U-26, DECISION_LEDGER.md:4264; [A-BIN]).
	LauncherName = "MoE-Direct-launcher"
	// LauncherVersion is the release version (DECISION_LEDGER.md:3967, :4102; S8-LA-30).
	LauncherVersion = "v0.3.2"
)

// BundleEnv names the environment variable through which the build command hands the
// bundle version to TestS8LA30. Unset, the test checks the value and the comparison only.
const BundleEnv = "MOE_BUNDLE_VERSION"

// About is the about text: the launcher's name and version, as the help page shows it.
func About() string { return LauncherName + " " + LauncherVersion }

// CheckBundle is the version gate: nil when the bundle version equals LauncherVersion
// exactly (string equality), an error naming both values otherwise.
func CheckBundle(bundle string) error {
	if bundle != LauncherVersion {
		return fmt.Errorf("launcher version %q differs from the bundle version %q", LauncherVersion, bundle)
	}
	return nil
}
