package core

import (
	"regexp"
	"strings"
)

// Conflict 8 of reviews/_gol_conflicts_8_user_20261010.md: VRAM stays no gate (display only,
// PS:3727-3734); an out-of-memory failure while the model loads is recognised from the
// server's own output and classified as its own failure reason - a sub-reason of
// fail_server_start (code 5), the status PowerShell gives a server that never reached ready.
// The user's wording is the UI's and the adapter's.

// CauseOutOfMemory is the LaunchError cause of a load that ran out of memory.
const CauseOutOfMemory = "out_of_memory"

// LoadFailure is the classified failure of a load.
type LoadFailure struct {
	Cause  string // CauseOutOfMemory
	Device string // "vram" (a CUDA / device allocation), "ram" (a host allocation) or "" (not said)
	Line   string // the server line that said it
}

var oomPatterns = []struct {
	re     *regexp.Regexp
	device string
}{
	{regexp.MustCompile(`(?i)cudaMalloc failed: out of memory`), "vram"},
	{regexp.MustCompile(`(?i)CUDA error: out of memory`), "vram"},
	{regexp.MustCompile(`(?i)(unable|failed) to allocate (CUDA|cuda)\d*(_Host)? buffer`), "vram"},
	{regexp.MustCompile(`(?i)ggml_backend_cuda_buffer_type_alloc_buffer: allocating .* failed`), "vram"},
	{regexp.MustCompile(`(?i)ggml_gallocr_reserve_n: failed to allocate (CUDA|cuda)\d* buffer`), "vram"},
	{regexp.MustCompile(`(?i)(unable|failed) to allocate CPU buffer`), "ram"},
	{regexp.MustCompile(`(?i)std::bad_alloc`), "ram"},
	{regexp.MustCompile(`(?i)failed to allocate compute buffers`), ""},
	{regexp.MustCompile(`(?i)\bout of memory\b`), ""},
}

// ClassifyLoadFailure reads the server's stderr lines of a start that ended before ready and
// says whether it ran out of memory. The first device-specific line wins over a generic one.
func ClassifyLoadFailure(lines []string) (LoadFailure, bool) {
	var generic *LoadFailure
	for _, ln := range lines {
		for _, p := range oomPatterns {
			if !p.re.MatchString(ln) {
				continue
			}
			f := LoadFailure{Cause: CauseOutOfMemory, Device: p.device, Line: strings.TrimSpace(ln)}
			if p.device != "" {
				return f, true
			}
			if generic == nil {
				generic = &f
			}
			break
		}
	}
	if generic != nil {
		return *generic, true
	}
	return LoadFailure{}, false
}

func oomDevice(d string) string {
	switch d {
	case "vram":
		return "GPU memory"
	case "ram":
		return "system memory"
	}
	return "memory"
}
