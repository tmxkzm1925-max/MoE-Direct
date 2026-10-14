package core

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Review item 30 (reviews/_gol_review_8_user_20261010.md, amendment 1 of the FIX-CORE brief):
// a value typed into a setting's input box. The screen opens the box with Space on a value
// row; on Enter it calls ParseTyped and either applies the value (the preflight is then
// recomputed) or shows the reason in red and keeps the typed text.

// TypedKeys are the settings that take a typed value (prefetch and the repack mode stay ←→).
var TypedKeys = []string{"port", "ctx", "threads", "budget_mb", "qd", "warmup", "prefetch_k", "prefetch_n"}

// NoteWarmupNotPorted says what a typed warmup file does (the name is kept for the adapter):
// since FIX2-CORE the launcher warmup is ported, so the file's text is precomputed once after
// ready (Invoke-LauncherWarmfile PS:14710); a failure is degraded, never a start failure.
const NoteWarmupNotPorted = "warmup file: its text is precomputed once after ready (a failure only skips the warmup)"

// ParseTyped parses and range-checks a typed value for the setting row r (Settings gives the
// rows with the PowerShell bounds, the port and QD clamps applied). It returns the value in
// the request's form (SettingKeys values: whole numbers, budget in MB, warmup "file:<path>"),
// a note when the value has a consequence the screen should say, or a short reason.
//
//   - port, threads, qd: a whole number.
//   - ctx: a whole number of tokens, or with a k suffix (1k = 1024): «64k» and «65536» are the
//     same value.
//   - budget_mb: the screen shows the budget in G, so a plain number is GiB (decimals allowed,
//     «14.5» = 14848 MB); a «g»/«gb» suffix says the same, an «m»/«mb» suffix gives MB.
//   - budget_mb and qd also take «auto» (the launcher decides at start).
//   - warmup: a file path as text (with or without the «file:» prefix).
//   - prefetch_k, prefetch_n (GL-19, F-PF §9 C1-C2): decimal digits only, checked on the text
//     as typed (« 8» is refused, S8-PF-01); «auto» clears the pair. K is refused outside its row's
//     1..min(16, n_expert); N only below 1, since N above QD-1 is accepted and off (§6).
func ParseTyped(r SettingRow, text string) (value, note string, err error) {
	if isPairKey(r.Key) && strings.ToLower(strings.TrimSpace(text)) != "auto" {
		v, perr := checkPairValue(r.Key, text)
		if perr == nil && r.Bounded {
			n, _ := strconv.ParseInt(v, 10, 64)
			if n < r.Min || n > r.Max {
				perr = fmt.Errorf("out of range")
			}
		}
		if perr != nil {
			rng := typedRange(r)
			if r.OnRange {
				rng = fmt.Sprintf("%d or more", r.Min)
			}
			return "", "", fmt.Errorf("%s", CustomValueText(PathSettingsTyped, r.Key, text, rng))
		}
		return v, "", nil
	}
	t := strings.TrimSpace(text)
	if t == "" {
		return "", "", fmt.Errorf("empty")
	}
	if !contains(TypedKeys, r.Key) {
		return "", "", fmt.Errorf("%s is chosen with ←→, not typed", r.Key)
	}
	if r.Key == "warmup" {
		p := t
		if len(p) >= len(warmupFilePrefix) && strings.EqualFold(p[:len(warmupFilePrefix)], warmupFilePrefix) {
			p = strings.TrimSpace(p[len(warmupFilePrefix):])
		}
		p = strings.Trim(p, `"`)
		if p == "" {
			return "", "", fmt.Errorf("no file path")
		}
		return warmupFilePrefix + p, NoteWarmupNotPorted, nil
	}
	lt := strings.ToLower(t)
	if lt == "auto" {
		if r.Key == "budget_mb" || r.Key == "qd" || isPairKey(r.Key) {
			return "auto", "", nil
		}
		return "", "", fmt.Errorf("%s has no auto value", r.Key)
	}
	var n int64
	switch r.Key {
	case "ctx":
		mult := int64(1)
		if strings.HasSuffix(lt, "k") {
			mult, lt = 1024, strings.TrimSpace(strings.TrimSuffix(lt, "k"))
		}
		v, perr := strconv.ParseInt(lt, 10, 64)
		if perr != nil || v > math.MaxInt64/mult {
			return "", "", fmt.Errorf("not a whole number (e.g. 64k or 65536)")
		}
		n = v * mult
	case "budget_mb":
		mib := 1024.0
		switch {
		case strings.HasSuffix(lt, "mb"):
			mib, lt = 1, strings.TrimSuffix(lt, "mb")
		case strings.HasSuffix(lt, "m"):
			mib, lt = 1, strings.TrimSuffix(lt, "m")
		case strings.HasSuffix(lt, "gb"):
			lt = strings.TrimSuffix(lt, "gb")
		case strings.HasSuffix(lt, "g"):
			lt = strings.TrimSuffix(lt, "g")
		}
		f, perr := strconv.ParseFloat(strings.TrimSpace(lt), 64)
		if perr != nil || math.IsNaN(f) || math.IsInf(f, 0) || f > 1<<40 {
			return "", "", fmt.Errorf("not a number (e.g. 14.5 for 14.5 GB)")
		}
		if mib == 1 && f != math.Trunc(f) {
			return "", "", fmt.Errorf("MB must be a whole number")
		}
		n = int64(math.Round(f * mib))
	default:
		v, perr := strconv.ParseInt(lt, 10, 64)
		if perr != nil {
			return "", "", fmt.Errorf("not a whole number")
		}
		n = v
	}
	if !r.Bounded && r.Key == "port" {
		r.Min, r.Max, r.Bounded = portMin, portMax, true
	}
	if !r.Bounded && r.Key == "qd" {
		r.Min, r.Max, r.Bounded = EngineQDMin, EngineQDMax, true
	}
	if r.Bounded && (n < r.Min || n > r.Max) {
		return "", "", fmt.Errorf("%s", CustomValueText(PathSettingsTyped, r.Key, t, typedRange(r))) // F-LS 11.4
	}
	if n <= 0 {
		return "", "", fmt.Errorf("must be above 0")
	}
	return strconv.FormatInt(n, 10), "", nil
}

// typedRange is the range in the unit the screen shows.
func typedRange(r SettingRow) string {
	switch r.Key {
	case "budget_mb":
		return fmt.Sprintf("%s GB ~ %s GB", gib1(r.Min), gib1(r.Max))
	case "ctx":
		return fmt.Sprintf("%s ~ %s", ctxK(r.Min), ctxK(r.Max))
	}
	return fmt.Sprintf("%d ~ %d", r.Min, r.Max)
}

func gib1(mb int64) string {
	return strconv.FormatFloat(math.Round(float64(mb)/1024*10)/10, 'f', -1, 64)
}

func ctxK(n int64) string {
	if n > 0 && n%1024 == 0 {
		return strconv.FormatInt(n/1024, 10) + "k"
	}
	return strconv.FormatInt(n, 10)
}

// ParseModelSetting is ParseTyped for a model's setting: a catalog profile's rows, or, for a
// model outside the catalog (profileID ""), its template structure's rows (GL-16).
func (l *Launcher) ParseModelSetting(profileID, modelPath, key, text string) (value, note string, err error) {
	rows, err := l.presetRows(profileID, modelPath)
	if err != nil {
		return "", "", err
	}
	for _, r := range rows {
		if r.Key == key {
			return ParseTyped(r, text)
		}
	}
	return "", "", fmt.Errorf("%s is not a setting", key)
}

// ParseSetting is ParseTyped for a catalog profile's setting.
func (l *Launcher) ParseSetting(profileID, key, text string) (value, note string, err error) {
	rows, err := l.Settings(profileID)
	if err != nil {
		return "", "", err
	}
	for _, r := range rows {
		if r.Key == key {
			return ParseTyped(r, text)
		}
	}
	return "", "", fmt.Errorf("%s is not a setting", key)
}
