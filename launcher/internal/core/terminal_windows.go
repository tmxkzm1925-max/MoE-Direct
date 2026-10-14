//go:build windows

package core

import "golang.org/x/sys/windows"

// osConsoleProbe reads the console state of this process (Test-MenuModeAvailable's probes).
// With no console at all the standard handles are not console handles, so the input probe
// fails first; haveConsole is left unset.
func osConsoleProbe() consoleProbe {
	return consoleProbe{
		stdinConsole: func() error {
			h, err := windows.GetStdHandle(windows.STD_INPUT_HANDLE)
			if err != nil {
				return err
			}
			var mode uint32
			return windows.GetConsoleMode(h, &mode)
		},
		stdoutSBufInfo: func() error {
			h, err := windows.GetStdHandle(windows.STD_OUTPUT_HANDLE)
			if err != nil {
				return err
			}
			var info windows.ConsoleScreenBufferInfo
			return windows.GetConsoleScreenBufferInfo(h, &info)
		},
	}
}
