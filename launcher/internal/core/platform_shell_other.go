//go:build !windows

package core

func osClipboard(string) error                      { return errWindowsOnly }
func osLaunch(string, string) error                 { return errWindowsOnly }
func osGPUProcessMemory() (map[string]int64, error) { return nil, errWindowsOnly }
