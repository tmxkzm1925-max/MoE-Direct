package core

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The strict readers of region 4 of the PowerShell launcher (PS:1956-2091): bytes, explicit
// UTF-8 with an optional BOM, then a JSON parse that keeps integers apart from other numbers.

type jobj = map[string]any

// readJSONFileStrict is Read-JsonFileStrict (PS:1984-1991).
func readJSONFileStrict(path string) (any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read failed: %v", err)
	}
	return parseJSONStrict(b)
}

// parseJSONStrict is ConvertFrom-Utf8Strict + ConvertFrom-JsonStrict (PS:1970-2003).
func parseJSONStrict(b []byte) (any, error) {
	b = bytes.TrimPrefix(b, []byte{0xEF, 0xBB, 0xBF})
	if !utf8.Valid(b) {
		return nil, fmt.Errorf("utf-8 decode failed: invalid byte sequence")
	}
	if len(bytes.TrimSpace(b)) == 0 {
		return nil, fmt.Errorf("empty json document")
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("json parse failed: %v", err)
	}
	if dec.More() {
		return nil, fmt.Errorf("json parse failed: trailing data after the document")
	}
	if v == nil {
		return nil, fmt.Errorf("json parsed to null")
	}
	return v, nil
}

func asObj(v any) (jobj, bool) { o, ok := v.(map[string]any); return o, ok }

func has(o jobj, k string) bool { _, ok := o[k]; return ok }

// jInt is Test-JsonNonNegativeInteger / Test-JsonInteger (PS:2050-2067): a JSON integer
// (no fraction, no exponent), never a boolean.
func jInt(v any) (int64, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	s := string(n)
	if strings.ContainsAny(s, ".eE") {
		return 0, false
	}
	i, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	return i, true
}

func jNonNegInt(v any) (int64, bool) {
	i, ok := jInt(v)
	if !ok || i < 0 {
		return 0, false
	}
	return i, true
}

func jStr(v any) (string, bool) { s, ok := v.(string); return s, ok }

// jNonEmptyStr is Test-JsonNonEmptyString (PS:2080-2084).
func jNonEmptyStr(v any) (string, bool) {
	s, ok := v.(string)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

func jBool(v any) (bool, bool) { b, ok := v.(bool); return b, ok }

func jArr(v any) ([]any, bool) { a, ok := v.([]any); return a, ok }

var sha256HexRe = regexp.MustCompile(`^[0-9a-fA-F]{64}$`)

// isSha256Hex is Test-Sha256Hex (PS:2086-2091).
func isSha256Hex(v any) bool {
	s, ok := v.(string)
	return ok && sha256HexRe.MatchString(s)
}

// denyUnknownKeys is Deny-UnknownKeys (PS:2274-2286): every key outside allowed+optional is
// refused, and every allowed key must be present.
func denyUnknownKeys(o jobj, allowed, optional []string, where string) *LaunchError {
	if o == nil {
		o = jobj{} // a missing object has no keys: the first required key is reported (PS Get-JsonKeys on $null)
	}
	keys := make([]string, 0, len(o))
	for k := range o {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !contains(allowed, k) && !contains(optional, k) {
			return stop(StatusFailGateCatalog, "unknown key '%s' in %s (deny-by-default)", k, where)
		}
	}
	for _, k := range allowed {
		if !has(o, k) {
			return stop(StatusFailGateCatalog, "required key '%s' missing in %s", k, where)
		}
	}
	return nil
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

// sha256Hex is the lower-case hex SHA-256 of b.
func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
