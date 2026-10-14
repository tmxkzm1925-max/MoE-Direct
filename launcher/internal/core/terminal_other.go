//go:build !windows

package core

// osConsoleProbe: the launcher is a Windows program; elsewhere there is no menu mode.
func osConsoleProbe() consoleProbe {
	return consoleProbe{
		haveConsole:    func() bool { return false },
		stdinConsole:   func() error { return errWindowsOnly },
		stdoutSBufInfo: func() error { return errWindowsOnly },
	}
}
