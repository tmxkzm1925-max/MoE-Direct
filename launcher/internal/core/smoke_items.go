package core

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// GL-11 (version 8): the rest of the PowerShell smoke checklist (Invoke-SmokeChecklist
// PS:12665-12950; LS:291-299 «API/UI smoke»; conflict row 18-1 [A-SMK]), and one log line per
// item. Items 1-4 are runSmokeChecklist (FIX2). This file adds:
//   - item 5, verify PASS consumption: the engine's post-seal attestation in the server's
//     complete stderr lines (Get-EngineSealAttestation PS:11954-11985) and the launcher's own
//     gate manifest (the verify or virtual plan gate's manifest sha256);
//   - item 6, loopback-only binding: every listener on the server's port has a loopback
//     address (Test-LoopbackOnlyBinding PS:12092-12106);
//   - item 7, the built-in web UI: a page load and a completion through the same origin when
//     the catalog's runtime.webui is true, else «SKIP (API-only bundle)» as PowerShell prints.

// The engine seal lines (PS:231-234).
const engineSealRawPrefix = "[moe-direct] sealed:"

var (
	engineSealRawRe        = regexp.MustCompile(`^\[moe-direct\] sealed: all=(?P<all>[0-9]+) host=(?P<host>[0-9]+) nonhost=(?P<nonhost>[0-9]+) slots=(?P<have>[0-9]+)/(?P<need>[0-9]+) moe_layers=(?P<layers>[0-9]+)$`)
	engineSealInfoPrefixRe = regexp.MustCompile(`^(?:[0-9]+(?:\.[0-9]+){3} I )?load_tensors: moe-direct: sealed `)
	engineSealInfoRe       = regexp.MustCompile(`^(?:[0-9]+(?:\.[0-9]+){3} I )?load_tensors: moe-direct: sealed all=(?P<all>[0-9]+) host=(?P<host>[0-9]+) nonhost=(?P<nonhost>[0-9]+) slots=(?P<have>[0-9]+)/(?P<need>[0-9]+) moe_layers=(?P<layers>[0-9]+) \(matched live=[0-9]+ host=[0-9]+ nonhost=[0-9]+\)$`)
	sealFields             = []string{"all", "host", "nonhost", "have", "need", "layers"}
)

// sealAttest is Get-EngineSealAttestation's result.
type sealAttest struct {
	OK        bool
	Reason    string
	Lines     int // the attestation records seen (raw and legacy INFO)
	SlotsHave int64
	SlotsNeed int64
	Line      string
}

// engineSealAttestation reads only complete lines; no substring evidence is accepted. One raw
// record, one legacy INFO record, or one of each that agree on every field passes; a duplicate
// of either kind, a malformed line or disagreeing records fail.
func engineSealAttestation(lines []string) sealAttest {
	type rec struct {
		kind, line string
		f          map[string]int64
	}
	var recs []rec
	for _, ln := range lines {
		var re *regexp.Regexp
		kind := ""
		switch {
		case strings.HasPrefix(ln, engineSealRawPrefix):
			kind, re = "raw", engineSealRawRe
		case engineSealInfoPrefixRe.MatchString(ln):
			kind, re = "info", engineSealInfoRe
		default:
			continue
		}
		m := re.FindStringSubmatch(ln)
		if m == nil {
			return sealAttest{Reason: "malformed complete seal success line", Line: ln}
		}
		f := map[string]int64{}
		for _, name := range sealFields {
			v, err := strconv.ParseInt(m[re.SubexpIndex(name)], 10, 64)
			if err != nil {
				return sealAttest{Reason: "seal numeric field is outside Int64", Line: ln}
			}
			f[name] = v
		}
		recs = append(recs, rec{kind, ln, f})
	}
	if len(recs) == 0 {
		return sealAttest{Reason: "no seal success line"}
	}
	for _, kind := range []string{"raw", "info"} {
		n := 0
		for _, r := range recs {
			if r.kind == kind {
				n++
			}
		}
		if n > 1 {
			return sealAttest{Lines: len(recs), Reason: fmt.Sprintf("seal success line seen %d times (duplicate %s attestation)", len(recs), kind)}
		}
	}
	if len(recs) == 2 {
		for _, name := range sealFields {
			if recs[0].f[name] != recs[1].f[name] {
				return sealAttest{Lines: 2, Reason: "seal representations disagree on " + name}
			}
		}
	}
	one := recs[0]
	return sealAttest{OK: true, Lines: len(recs), SlotsHave: one.f["have"], SlotsNeed: one.f["need"], Line: one.line}
}

// loopbackOnlyBinding is Test-LoopbackOnlyBinding: at least one listener on the port, and every
// listener's local address a loopback address.
func loopbackOnlyBinding(port int, listeners func(int) ([]string, error)) (bool, []string, string) {
	if listeners == nil {
		return false, nil, "listener enumeration failed: no listener table"
	}
	addrs, err := listeners(port)
	if err != nil {
		return false, nil, "listener enumeration failed: " + err.Error()
	}
	if len(addrs) == 0 {
		return false, nil, "no listener found"
	}
	for _, a := range addrs {
		if ip := net.ParseIP(a); ip == nil || !ip.IsLoopback() {
			return false, addrs, "non-loopback listener: " + a
		}
	}
	return true, addrs, ""
}

// smokeContext is what items 5-7 need beyond the server's address.
type smokeContext struct {
	ManifestSha string // the launcher gate's manifest sha256 (item 5)
	Port        int    // the server's port (item 6)
	WebUI       bool   // the catalog's runtime.webui (item 7)
	Listeners   func(int) ([]string, error)
}

// runSmokeFull is the whole checklist: items 1-4 (runSmokeChecklist), then 5-7.
func runSmokeFull(ctx context.Context, base string, readLog func() []string, t smokeTimings, x smokeContext) SmokeResult {
	t0 := time.Now()
	res := runSmokeChecklist(ctx, base, readLog, t)
	add := func(it SmokeItem, label string) {
		res.Items = append(res.Items, it)
		if !it.OK && !it.Skipped {
			res.Failures = append(res.Failures, label)
		}
	}

	// (5) the engine's post-seal attestation and the launcher's gate manifest
	seal := engineSealAttestation(readLog())
	ok5 := seal.OK && x.ManifestSha != ""
	d5 := seal.Reason
	switch {
	case ok5:
		d5 = fmt.Sprintf("engine seal line x1, slots %d/%d; launcher gate manifest %s", seal.SlotsHave, seal.SlotsNeed, short12(x.ManifestSha))
	case seal.OK:
		d5 = "no launcher gate manifest"
	}
	add(SmokeItem{N: 5, Name: "verify PASS consumption", OK: ok5, Detail: d5}, "5:verify-consume")

	// (6) loopback-only binding
	ok6, addrs, why6 := loopbackOnlyBinding(x.Port, x.Listeners)
	d6 := why6
	if ok6 {
		d6 = strings.Join(addrs, ",")
	}
	add(SmokeItem{N: 6, Name: "loopback-only binding", OK: ok6, Detail: d6}, "6:loopback")

	// (7) the built-in web UI: load and send, or skipped on an API-only bundle
	if x.WebUI {
		g := httpJSON(ctx, http.MethodGet, base+"/", "", t.HTTP)
		okLoad := g.ok && g.status == 200 && len(g.body) > 0
		s := httpJSON(ctx, http.MethodPost, base+"/v1/chat/completions", smokeBodyNonStream, t.HTTP)
		okSend := false
		if s.ok && s.status == 200 {
			if v, err := parseJSONStrict(s.body); err == nil {
				o, _ := asObj(v)
				if ch, ok := jArr(o["choices"]); ok && len(ch) > 0 {
					okSend = true
				}
			}
		}
		it := SmokeItem{N: 7, Name: "built-in web UI load + send", OK: okLoad && okSend}
		if !it.OK {
			it.Detail = fmt.Sprintf("load=%v send=%v", okLoad, okSend)
		}
		add(it, "7:webui")
	} else {
		add(SmokeItem{N: 7, Name: "built-in web UI", OK: true, Skipped: true, Detail: "API-only bundle"}, "7:webui")
	}

	res.OK = len(res.Failures) == 0
	res.Elapsed = time.Since(t0)
	return res
}

// Line is one item as PowerShell writes it (Write-Line «  [n] name   PASS (detail)», without the
// two leading spaces; items 1-4 keep the draft's own details, FIX2).
func (it SmokeItem) Line() string {
	v := "FAIL"
	switch {
	case it.Skipped:
		v = "SKIP"
	case it.OK:
		v = "PASS"
	}
	if it.Detail != "" {
		v += " (" + it.Detail + ")"
	}
	return fmt.Sprintf("[%d] %-32s %s", it.N, it.Name, v) // PowerShell's column: the verdict at 37, item 3's longer name one space after
}

// smokeContextFor is items 5-7's context for a start of pre on cfg.
func (l *Launcher) smokeContextFor(pre *PreflightResult, cfg *Config) smokeContext {
	return smokeContext{ManifestSha: pre.ManifestSha, Port: cfg.Port, WebUI: pre.WebUI, Listeners: l.os.listeners}
}

// logSmoke writes the checklist to the launcher log: one SMOKE_ITEM record per item, then the
// SMOKE summary (PowerShell: a Write-Line per item, then Write-Diag SMOKE).
func (l *Launcher) logSmoke(r SmokeResult) {
	for _, it := range r.Items {
		res := "FAIL"
		switch {
		case it.Skipped:
			res = "SKIP"
		case it.OK:
			res = "PASS"
		}
		l.diag.write("SMOKE_ITEM", map[string]any{"item": it.N, "name": it.Name, "result": res, "detail": it.Detail, "line": it.Line()})
	}
	l.diag.write("SMOKE", map[string]any{"ok": r.OK, "failures": r.Failures, "items": r.Items, "ms": r.Elapsed.Milliseconds(),
		"not_ported": SmokeNotPorted})
}
