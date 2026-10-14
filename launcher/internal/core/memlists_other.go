// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

//go:build !windows

package core

func osMemoryLists() (MemoryLists, error) { return MemoryLists{}, errWindowsOnly }

func osTaskManager() error { return errWindowsOnly }
