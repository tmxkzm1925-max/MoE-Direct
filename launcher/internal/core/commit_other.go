// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

//go:build !windows

package core

func osPerformanceInfo() (CommitReading, error) { return CommitReading{}, errWindowsOnly }
