package core

import (
	"errors"
)

// Conflict 15 of reviews/_gol_conflicts_8_user_20261010.md: when the launcher is not in an
// interactive console it draws no screen, prints one line and runs the argument-driven start as
// the official path, whose exit codes are the PowerShell table (a range violation = 3).
//
// Detection is Test-MenuModeAvailable (PS:11325-11335): -NonInteractive, then
// [Console]::IsInputRedirected, then the RawUI probes (KeyAvailable reads the console input,
// CursorPosition reads the console screen buffer) - any of them failing means no menu. The Go
// probes are the Win32 calls under those .NET members: GetConsoleMode on the standard input
// (fails when the input is a file or a pipe, which is also what a terminal that is not a real
// console, such as Git Bash mintty, gives a child) and GetConsoleScreenBufferInfo on the standard
// output (fails when the output is redirected or there is no console at all).

// TerminalInfo is the detection's answer.
type TerminalInfo struct {
	Interactive bool
	Reason      string // why not: "-NonInteractive", "input redirected", "output redirected", "no console"
}

// consoleProbe is the three reads; the tests replace them.
type consoleProbe struct {
	haveConsole    func() bool  // a console is attached to this process
	stdinConsole   func() error // GetConsoleMode(STD_INPUT_HANDLE)
	stdoutSBufInfo func() error // GetConsoleScreenBufferInfo(STD_OUTPUT_HANDLE)
}

// menuModeAvailable is Test-MenuModeAvailable with the three probes.
func menuModeAvailable(nonInteractive bool, p consoleProbe) TerminalInfo {
	if nonInteractive {
		return TerminalInfo{Reason: "-NonInteractive"}
	}
	if p.haveConsole != nil && !p.haveConsole() {
		return TerminalInfo{Reason: "no console"}
	}
	if err := p.stdinConsole(); err != nil {
		return TerminalInfo{Reason: "input redirected or not a real console (" + err.Error() + ")"}
	}
	if err := p.stdoutSBufInfo(); err != nil {
		return TerminalInfo{Reason: "output redirected or not a real console (" + err.Error() + ")"}
	}
	return TerminalInfo{Interactive: true}
}

// DetectTerminal answers for this process. nonInteractive is the -NonInteractive switch.
func DetectTerminal(nonInteractive bool) TerminalInfo {
	return menuModeAvailable(nonInteractive, osConsoleProbe())
}

// StatusOf is the PowerShell status an argument-driven start ends with for err (nil = ok):
// a *LaunchError's own status (a *GateError and a *RepackNeededError carry one), a not-ported
// family's failure stage, and the most conservative failure (fail_server_start, 5) for
// anything else, as Get-StatusExitCode does for an unknown status.
func StatusOf(err error) Status {
	if err == nil {
		return StatusOK
	}
	var rn *RepackNeededError
	if errors.As(err, &rn) {
		return rn.Status
	}
	if le, ok := AsLaunchError(err); ok {
		return le.Status
	}
	if errors.Is(err, ErrNotPorted) {
		return notPortedStatus(err)
	}
	return StatusFailServerStart
}

// ExitCodeOf is the process exit code of the argument path for err: the PowerShell table
// (STATUS_EXIT PS:203-221) - 0 ok, 2 cancelled, 3 refused input or resources (a value out of
// range is fail_custom_args = 3), 4 a gate (bundle, catalog, verify, engine seal), 5 the
// server, 6 the smoke check.
func ExitCodeOf(err error) int { return ExitCode(StatusOf(err)) }
