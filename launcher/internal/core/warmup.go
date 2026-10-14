package core

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

// Conflict 2 of reviews/_gol_conflicts_8_user_20261010.md opens warmup (one of the six
// PowerShell settings). This is Invoke-ReadyWarmup / Invoke-LauncherWarmup /
// Invoke-LauncherWarmfile (PS:14578-14768): after ready, "on" sends one chat request with
// max_tokens 1 (300 s), "file:<path>" precomputes the file's text as a cached prompt prefix
// (1800 s), "off" sends nothing; every failure is degraded and non-terminal. The default is
// "on" (PS:865). A start that runs the «작동 확인» check skips it, as PowerShell's -Smoke forces
// warmup off (forced_bench). GL-15: a restored slot skips it in every mode (Invoke-ReadyWarmup
// PS:14579-14585, WARMSTART A-3), with the skip reason warmstart_restored.

const (
	WarmupDefault      = "on" // PS:865
	warmupTimeout      = 300 * time.Second
	warmfileTimeout    = 1800 * time.Second // WARMFILE_TIMEOUT_SEC
	warmupBody         = `{"model":"local","messages":[{"role":"user","content":"warmup"}],"max_tokens":1,"stream":false}`
	WarmupSkipUserOff  = "user_off"           // WARMUP_SKIP_USER_OFF
	WarmupSkipForced   = "forced_bench"       // WARMUP_SKIP_FORCED_BENCH
	WarmupSkipRestored = "warmstart_restored" // GL-15: PowerShell's WARMUP_SKIPPED carries the text only
	// UX-R12b (5): the check run by its English name
	warmupForcedReason = "the check run is due at this start (PowerShell -Smoke forces warmup off)"
)

// WarmupResult is what the warmup did.
type WarmupResult struct {
	Mode       string // on | off | file
	Skipped    bool
	SkipReason string // user_off | forced_bench | warmstart_restored
	OK         bool
	Degraded   bool   // a failure: the server serves without warmup
	Reason     string // the failure, or the skip text
	File       string
	Tokens     int64 // the file mode's precomputed token count
	Elapsed    time.Duration
}

// Line is the result as one log line (the PowerShell [warmup] lines).
func (w WarmupResult) Line() string {
	switch {
	case w.Skipped:
		return "[warmup] skipped: " + w.Reason + " [reason=" + w.SkipReason + "]"
	case w.OK && w.Mode == "file":
		return fmt.Sprintf("[warmup] Precomputed %d tokens.", w.Tokens)
	case w.OK:
		return "[warmup] launcher warmup request completed."
	}
	return "[warmup] WARNING: " + w.Reason + "; continuing without warmup (degraded, non-terminal)."
}

// restoredWarmup is Invoke-ReadyWarmup's first branch: a restored slot owns the prefix, so no
// warmup of any mode runs (GL-15).
func restoredWarmup(value string) WarmupResult {
	mode := "on"
	if _, ok := warmupFilePath(value); ok {
		mode = "file"
	} else if value != "on" {
		mode = "off"
	}
	return WarmupResult{Mode: mode, Skipped: true, SkipReason: WarmupSkipRestored, Reason: warmupRestoredReason}
}

// warmupFilePath is Get-WarmupFilePath (PS:6023-6030).
func warmupFilePath(v string) (string, bool) {
	if len(v) <= len(warmupFilePrefix) || strings.ToLower(v[:len(warmupFilePrefix)]) != warmupFilePrefix {
		return "", false
	}
	return v[len(warmupFilePrefix):], true
}

// warmfileJSONString is ConvertTo-WarmfileJsonString (PS:14651-14663): '"', '\' escaped,
// every UTF-16 code unit outside 32..126 as \uXXXX.
func warmfileJSONString(s string) string {
	var sb strings.Builder
	sb.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			sb.WriteString(`\"`)
		case r == '\\':
			sb.WriteString(`\\`)
		case r < 32 || r > 126:
			if r > 0xFFFF {
				r -= 0x10000
				fmt.Fprintf(&sb, `\u%04x\u%04x`, 0xD800+(r>>10), 0xDC00+(r&0x3FF))
			} else {
				fmt.Fprintf(&sb, `\u%04x`, r)
			}
		default:
			sb.WriteRune(r)
		}
	}
	sb.WriteByte('"')
	return sb.String()
}

// warmfileTokenCount is Get-WarmfileTokenCount (PS:14665-14677).
func warmfileTokenCount(body []byte) (int64, bool) {
	v, err := parseJSONStrict(body)
	if err != nil {
		return 0, false
	}
	o, _ := asObj(v)
	if n, ok := jNonNegInt(o["tokens_evaluated"]); ok {
		return n, true
	}
	tm, _ := asObj(o["timings"])
	pn, ok1 := jNonNegInt(tm["prompt_n"])
	cn, ok2 := jNonNegInt(tm["cache_n"])
	if ok1 && ok2 {
		return pn + cn, true
	}
	return 0, false
}

// runWarmup is Invoke-LauncherWarmup with its file branch, against base (http://host:port).
func runWarmup(ctx context.Context, base, value, forcedReason string, cwd string) WarmupResult {
	t0 := time.Now()
	if raw, ok := warmupFilePath(value); ok {
		w := runWarmfile(ctx, base, raw, cwd)
		w.Elapsed = time.Since(t0)
		return w
	}
	if value != "on" {
		w := WarmupResult{Mode: "off", Skipped: true, SkipReason: WarmupSkipUserOff, Reason: "warmup off (user or stored preset)"}
		if forcedReason != "" {
			w.SkipReason, w.Reason = WarmupSkipForced, forcedReason
		}
		return w
	}
	if forcedReason != "" {
		return WarmupResult{Mode: "on", Skipped: true, SkipReason: WarmupSkipForced, Reason: forcedReason}
	}
	r := httpJSON(ctx, http.MethodPost, base+"/v1/chat/completions", warmupBody, warmupTimeout)
	w := WarmupResult{Mode: "on", Elapsed: time.Since(t0)}
	if r.ok && r.status == 200 {
		w.OK = true
		return w
	}
	w.Degraded = true
	w.Reason = "warmup request failed (" + r.reason + ")"
	return w
}

func runWarmfile(ctx context.Context, base, raw, cwd string) WarmupResult {
	w := WarmupResult{Mode: "file", Degraded: true}
	file := raw
	if !filepath.IsAbs(file) {
		file = filepath.Join(cwd, file)
	}
	file = filepath.Clean(file)
	w.File = file
	b, err := os.ReadFile(file)
	if err != nil {
		w.Reason = "warmup file precompute failed (warmup file could not be read: " + err.Error() + ")"
		return w
	}
	b = bytes.TrimPrefix(b, []byte{0xEF, 0xBB, 0xBF})
	if !utf8.Valid(b) {
		w.Reason = "warmup file precompute failed (warmup file is not valid UTF-8: utf-8 decode failed)"
		return w
	}
	if len(b) == 0 {
		w.Reason = "warmup file precompute failed (warmup file has no text to precompute)"
		return w
	}
	body := `{"prompt":` + warmfileJSONString(string(b)) + `,"cache_prompt":true,"n_predict":1,"stream":false}`
	r := httpJSON(ctx, http.MethodPost, base+"/completion", body, warmfileTimeout)
	if !(r.ok && r.status == 200) {
		w.Reason = "warmup file precompute failed (warmup file request failed: " + r.reason + ")"
		return w
	}
	n, ok := warmfileTokenCount(r.body)
	if !ok {
		w.Reason = "warmup file precompute failed (warmup file response carried no server token count (no value is estimated))"
		return w
	}
	w.OK, w.Degraded, w.Tokens = true, false, n
	return w
}

// warmupValue is the start's warmup value: the request's (after the preset), else the default.
func warmupValue(ov map[string]string) string {
	if v, ok := ov["warmup"]; ok && v != "" {
		return v
	}
	return WarmupDefault
}
