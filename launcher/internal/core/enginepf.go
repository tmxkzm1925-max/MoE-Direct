package core

// GL-21: the post-ready read-back of the engine's prefetch values (F-PF §8.5, §6.4; PF-PRE
// 5.3, PFP:302-309). Once the server is ready the launcher reads the first line of the metrics
// file whose path it chose (MOE_DIRECT_METRICS, config.go; SMD:8626-8641 form), the engine's
// header (OBS:272-274 «type» header, :418-421 the four prefetch keys), and shows prefetch_on,
// prefetch_k, prefetch_n and prefetch_off_reason next to the launcher's candidate. After ready
// the screen follows the engine, not the candidate (§6.4).

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// engineHeaderMax bounds the header line read (the header carries per-layer arrays).
const engineHeaderMax = 8 << 20

// EnginePrefetch is the engine's prefetch state after ready, as its metrics header gives it.
// Read is false when the header could not be read; Why then says why.
type EnginePrefetch struct {
	Read      bool   `json:"read"`
	On        bool   `json:"prefetch_on"`
	K         int64  `json:"prefetch_k"`
	N         int64  `json:"prefetch_n"`
	OffReason string `json:"prefetch_off_reason"`
	Why       string `json:"why,omitempty"`
}

// ReadEnginePrefetch reads the first line of the metrics file at path. The engine keeps the
// file open for writing and shares it for reading (OBS CreateFileW FILE_SHARE_READ), so the
// read does not disturb it.
func ReadEnginePrefetch(path string) EnginePrefetch {
	if path == "" {
		return EnginePrefetch{Why: "no metrics path"}
	}
	f, err := os.Open(path)
	if err != nil {
		return EnginePrefetch{Why: "metrics file not readable: " + err.Error()}
	}
	defer f.Close()
	line, err := bufio.NewReaderSize(io.LimitReader(f, engineHeaderMax), 64<<10).ReadBytes('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return EnginePrefetch{Why: "metrics file not readable: " + err.Error()}
	}
	return parseEngineHeader(line)
}

// parseEngineHeader reads the four prefetch keys of one header line; every key must be
// present with its type, or the values are not shown (never a guess).
func parseEngineHeader(line []byte) EnginePrefetch {
	if len(strings.TrimSpace(string(line))) == 0 {
		return EnginePrefetch{Why: "metrics file has no header line"}
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(line, &h); err != nil {
		return EnginePrefetch{Why: "metrics header is not JSON"}
	}
	var typ string
	if json.Unmarshal(h["type"], &typ) != nil || typ != "header" {
		return EnginePrefetch{Why: "the first metrics line is not the header"}
	}
	e := EnginePrefetch{Read: true}
	for key, dst := range map[string]any{"prefetch_on": &e.On, "prefetch_k": &e.K, "prefetch_n": &e.N, "prefetch_off_reason": &e.OffReason} {
		raw, ok := h[key]
		if !ok || json.Unmarshal(raw, dst) != nil {
			return EnginePrefetch{Why: "metrics header without " + key}
		}
	}
	return e
}

// Fields is the engine's values in the header's own names, exactly as the header gives them
// (S8-PF-11: header fields against the launcher line); an empty off reason prints as nothing.
func (e EnginePrefetch) Fields() string {
	if !e.Read {
		return "not read (" + e.Why + ")"
	}
	return fmt.Sprintf("prefetch_on=%t prefetch_k=%d prefetch_n=%d prefetch_off_reason=%s", e.On, e.K, e.N, e.OffReason)
}

// candidateText is the launcher's candidate in the echo's words (§4.4).
func candidateText(c *PrefetchEcho) string {
	if c == nil {
		return "(none)"
	}
	if c.EffectivePrefetch == "on" {
		s := fmt.Sprintf("on requested_k=%s requested_n=%s", nullInt(c.RequestedK), nullInt(c.RequestedN))
		if c.PrefetchNCap != nil { // GL-20: N beside N_cap on a virtual repack (F-PF §8.2)
			s += fmt.Sprintf(" prefetch_n_cap=%d", *c.PrefetchNCap)
		}
		return s
	}
	return c.EffectivePrefetch
}

// NCapLine is the log's N_cap line per layer of the candidate ("" when it has none; F-PF §8.2).
func NCapLine(c *PrefetchEcho) string {
	if c == nil || len(c.PrefetchNCapPerLayer) == 0 {
		return ""
	}
	return nCapLogLine(c.PrefetchNCapPerLayer)
}

// PostReadyLine is the launcher's post-ready line: the engine's values, then the candidate.
func PostReadyLine(e EnginePrefetch, cand *PrefetchEcho) string {
	return "prefetch after ready: engine " + e.Fields() + " · candidate " + candidateText(cand)
}
