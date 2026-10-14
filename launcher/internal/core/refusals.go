// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import "fmt"

// The BUP 2.7 refusal texts in Go (F-BU H.2; GL-06): message text only, byte for byte against
// the PowerShell source lines named in each comment; the stop routine that prints one is the
// launcher's.

// ReasonNotInteger is PS :6088.
func ReasonNotInteger(key string) string { return fmt.Sprintf("%s must be an integer", key) }

// ReasonOutOfBounds is PS :6101.
func ReasonOutOfBounds(key string, lo, hi int64) string {
	return fmt.Sprintf("%s out of bounds [%d..%d]", key, lo, hi)
}

// StopInvalidCLI is PS :13642 (fail_custom_args, non-interactive).
func StopInvalidCLI(key, reason string) string { return fmt.Sprintf("invalid -%s: %s", key, reason) }

// LineIgnoredCLI is PS :13643 (interactive: the value is dropped).
func LineIgnoredCLI(key, reason string) string {
	return fmt.Sprintf("[custom] ignoring invalid -%s: %s", key, reason)
}

// LineEditorRejected is PS :10917 (the custom editor asks again; four spaces).
func LineEditorRejected(reason string) string { return "    rejected: " + reason }

// LinePresetDiscarded is PS :5700, :5711.
func LinePresetDiscarded(key, reason string) string {
	return fmt.Sprintf("[preset] stored preset discarded (override %s rejected: %s); falling back to catalog defaults.", key, reason)
}

// AutotuneBelowMin carries the numbers of PS :3881-3884.
type AutotuneBelowMin struct {
	Candidate, SupportedMin, StructuralMin, ProfileMin, MachineCap, InstalledMiB, ModelCap int64
}

// ReasonAutotuneBelowMin is PS :3881-3884 (the source's full numbers and semicolon, H.3).
func ReasonAutotuneBelowMin(c AutotuneBelowMin) string {
	return fmt.Sprintf("candidate %d MB is below the supported minimum %d MB (structural_min %d MB, profile min_budget_mb %d MB; machine_cap %d MB from installed %d MiB, model_cap %d MB)",
		c.Candidate, c.SupportedMin, c.StructuralMin, c.ProfileMin, c.MachineCap, c.InstalledMiB, c.ModelCap)
}

// BannerAutotuneImpossible is PS :3958.
func BannerAutotuneImpossible(reason string) string {
	return "[budget] autotune cannot serve this model on this machine - " + reason
}

// StopAutotune is PS :3968 (fail_resource).
func StopAutotune(reason string) string { return autotuneStopPrefix + reason }

const autotuneStopPrefix = "budget autotune: "

// LineRAMInsufficient is PS :3695.
func LineRAMInsufficient(req, avail, ctx int64, basis string) string {
	return fmt.Sprintf("  ram    : required %d MB > available %d MB (ctx %d, basis %s)", req, avail, ctx, basis)
}

// ReasonBudgetOverRAM is PS :3601.
const ReasonBudgetOverRAM = "selected budget alone exceeds available RAM (the unmeasured fixed terms can only add to it)"

// StopRAMPreflight is PS :3696 (fail_resource).
func StopRAMPreflight(reason string) string { return "RAM preflight hard stop: " + reason }
