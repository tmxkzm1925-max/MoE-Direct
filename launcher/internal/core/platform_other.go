//go:build !windows

package core

import (
	"errors"
	"time"
)

// The launcher is a Windows program (the PowerShell launcher calls Win32 throughout);
// these stubs only keep the package compiling elsewhere, and every call fails.

var errWindowsOnly = errors.New("windows only")

func osMemStatus() (MemStatus, error)              { return MemStatus{}, errWindowsOnly }
func osInstalledMiB() (int64, error)               { return 0, errWindowsOnly }
func osFileIdentity(string) (FileIdentity, error)  { return FileIdentity{}, errWindowsOnly }
func osVolumeGUID(string) (string, error)          { return "", errWindowsOnly }
func osPortOwners(int) ([]int, error)              { return nil, errWindowsOnly }
func osConsoleAvailable() bool                     { return false }
func osProcessName(int) string                     { return "" }
func osPortListeners(int) ([]string, error)        { return nil, errWindowsOnly }
func fixedDriveModelRoots() []string               { return nil }
func isHidden(string) bool                         { return false }
func acquireNamedMutex(string) (*heldMutex, error) { return nil, errWindowsOnly }
func acquireFileLock(string) (*heldFile, error)    { return nil, errWindowsOnly }

type heldMutex struct{}

func (m *heldMutex) release() {}

type heldFile struct{}

func (f *heldFile) release() {}

type winChild struct{}

func (c *winChild) PID() int                              { return 0 }
func (c *winChild) NewGroup() bool                        { return false }
func (c *winChild) ErrLog() string                        { return "" }
func (c *winChild) OutLog() string                        { return "" }
func (c *winChild) Exited() (uint32, bool)                { return 0, true }
func (c *winChild) WaitExit(time.Duration) (uint32, bool) { return 0, true }
func (c *winChild) CtrlBreak() error                      { return errWindowsOnly }
func (c *winChild) ForceKill()                            {}
func (c *winChild) Memory() (ProcessMemory, error)        { return ProcessMemory{}, errWindowsOnly }
func (c *winChild) Close()                                {}

func startOwnedChild(string, []string, []string, string, string, string) (*winChild, error) {
	return nil, errWindowsOnly
}

func osVolumeFree(string) (int64, int64, string, error) { return 0, 0, "", errWindowsOnly }
