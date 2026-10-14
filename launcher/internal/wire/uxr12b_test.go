package wire

import (
	"errors"
	"testing"
)

// TestUXR12bStopReason (UX-R12b; UX-R5 landing 4 item 5): core's reason when the server is
// stopped during the check run reaches the screen whole, as the key of the screens' table.
func TestUXR12bStopReason(t *testing.T) {
	err := errors.New("the server was stopped during the " + "check run") // core start.go stoppedDuring
	if why, _ := userReason(err, nil); why != stoppedCheckRun {
		t.Errorf("stop reason: %q", why)
	}
}
