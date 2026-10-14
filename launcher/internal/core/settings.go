package core

import (
	"fmt"
	"strconv"
	"strings"
)

// GL-16: the per-model settings and the launcher settings of 5 Settings (F-LS 4.1, 5.3, 6.2,
// 6.3, 6.28, 7.1, 7.2, 7.8, 11.4, 12.4-12.6; S8-LA-28, S8-LA-29, S8-LA-35).

// --- the custom-value refusals per path (F-LS 5.3, 11.4, 12.5; S8-LA-29) ---

// CustomValuePath is the surface a refused value came from (F-LS 12.5).
type CustomValuePath string

const (
	PathArguments     CustomValuePath = "arguments"      // the argument-driven mode
	PathSettingsTyped CustomValuePath = "settings-typed" // a typed value on the settings page
	PathSavedSettings CustomValuePath = "saved-settings" // the per-model saved settings
)

// CustomValuePaths lists the paths that refuse a custom value, in the edition's order.
var CustomValuePaths = []CustomValuePath{PathArguments, PathSettingsTyped, PathSavedSettings}

// CustomValueTemplates are F-LS 11.4's texts (draft wording, First's; the user confirms through
// the main): <name>, <value> and <range> are filled by CustomValueText. The argument-driven
// mode has no screen, so it has no Korean text.
var CustomValueTemplates = map[CustomValuePath]struct{ EN, KO string }{
	PathArguments:     {"a value given as a command-line argument failed its type or range check: <name> = <value>, allowed <range>", ""},
	PathSettingsTyped: {"outside its range · allowed <range> · the value is kept, nothing applied", "범위 밖 · 허용 <range> · 값은 그대로, 적용 안 됨"},
	PathSavedSettings: {"the saved settings of this model hold a value that failed its check · not used · the defaults apply", "이 모델의 저장된 설정에 검사를 통과하지 못한 값이 있음 · 쓰지 않음 · 기본값 적용"},
}

// StatusHintCustomArgs is 11.4's «hint» line: the status hint of fail_custom_args, which only
// the argument-driven mode ends with (the settings page asks again, the saved settings are not
// used; F-LS 5.3).
const StatusHintCustomArgs = "a value given to the argument-driven mode failed its type or range check"

// argFlag is the command-line name of a setting key (cmd: -budget-mb for budget_mb).
func argFlag(key string) string { return "-" + strings.ReplaceAll(key, "_", "-") }

// CustomValueText is the English refusal of path for setting key with the given value.
func CustomValueText(path CustomValuePath, key, value, allowed string) string {
	name := key
	if path == PathArguments {
		name = argFlag(key)
	}
	return strings.NewReplacer("<name>", name, "<value>", strings.TrimSpace(value), "<range>", allowed).
		Replace(CustomValueTemplates[path].EN)
}

// --- the prompt-cache cap (U-22, L:4275; F-BU E; F-LS 4.1, 10.14) ---

// CapSetting is the cap set on 5 Settings (F-BU's declared type): Set false = no choice, the
// default, which is the off form (the user's decision of DECISION_LEDGER.md:4420, F-BU E.2:
// `--cache-ram 0 --ctx-checkpoints 4`); Value 0 is the same off form chosen; a positive Value
// is the user's choice, passed as `--cache-ram <cap> --ctx-checkpoints 4` (F-BU E.5-E.6). The
// argv, the commit prediction's cache term (the cap passed, 0 by default) and the allowed
// maximum of E.3 shown beside the choice are F-BU's builder's (GL-06, `ResolveCap`).
type CapSetting struct {
	Set   bool
	Value int
}

// The values a set cap may take (MiB): 0 = off, else the bounds of F-BU's cap rule
// (CapMinMiB, CapMaxMiB; F-BU E.3).
const (
	CapOffMiB = 0
	CapMinMiB = 256
	CapMaxMiB = 8192
)

// CapChoices are the values the cap row cycles through (0 = off, the default, and the powers
// of two of F-BU E.3); any whole number in the bounds can also be typed.
var CapChoices = []string{"0", "256", "512", "1024", "2048", "4096", "8192"}

// CapAllowed is the cap's range as the refusal texts name it.
const CapAllowed = "0 (off) or 256..8192 MiB"

// ParseCap reads a cap value: empty = no choice (the default, off), «off» = 0, else a whole
// number of MiB, 0 or in [CapMinMiB, CapMaxMiB]; a refusal is the settings-typed text. «auto»
// is no value since the default became the off form (DECISION_LEDGER.md:4420).
func ParseCap(text string) (CapSetting, error) {
	t := strings.ToLower(strings.TrimSpace(text))
	t = strings.TrimSpace(strings.TrimSuffix(strings.TrimSuffix(t, "mib"), "mb"))
	switch t {
	case "":
		return CapSetting{}, nil
	case "off":
		return CapSetting{Set: true, Value: CapOffMiB}, nil
	}
	n, err := strconv.Atoi(t)
	if err != nil || n != CapOffMiB && (n < CapMinMiB || n > CapMaxMiB) {
		return CapSetting{}, fmt.Errorf("%s", CustomValueText(PathSettingsTyped, "prompt_cache_cap", text, CapAllowed))
	}
	return CapSetting{Set: true, Value: n}, nil
}

// String is the cap as the settings show and record it: «off» (no choice or 0; the same off
// form, DECISION_LEDGER.md:4420) or the MiB.
func (c CapSetting) String() string {
	if !c.Set || c.Value == CapOffMiB {
		return "off"
	}
	return strconv.Itoa(c.Value)
}

// --- the warm-start switches (OQ-6 (a), user; F-LS 4.1, 12.4) ---

// Switches are the warm-start feature's own switches as 5 Settings holds them. Both default
// to on, as the PowerShell launcher's (warmstart absent = on, PS:704; autosave every
// KV_AUTOSAVE_DEFAULT_MIN minutes, PS:1100). The feature itself is warm start's (GL-15); until
// it is built the server runs with warm start off whatever the switches say (Q5.10).
type Switches struct {
	WarmStart bool
	Autosave  bool
}

// DefaultSwitches are the switches before the user changes them.
var DefaultSwitches = Switches{WarmStart: true, Autosave: true}

// ParseSwitch reads one switch value (on/off and the PowerShell spellings of PS:6077).
func ParseSwitch(key, text string) (bool, error) {
	if !contains(WarmStartSwitches, key) {
		return false, fmt.Errorf("%s is not a warm-start switch", key)
	}
	v, err := checkSetting(key, text, nil)
	if err != nil {
		return false, fmt.Errorf("%s", CustomValueText(PathSettingsTyped, key, text, "on or off"))
	}
	return v == "on", nil
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// LauncherSettingsMap is the launcher settings of a start as their keys, for the comparison
// with the last actual start (review item 2, rows 6.1-6.2).
func LauncherSettingsMap(c CapSetting, s Switches) map[string]string {
	return map[string]string{"prompt_cache_cap": c.String(), "warmstart": onOff(s.WarmStart), "autosave": onOff(s.Autosave)}
}

// NoticeNextStart is the line of review item 1 (F-LS 6.1, 11.7) while a server runs and a
// setting differs from the one it started with.
const NoticeNextStart = "not applied to the running server · applies at the next start"

// BindHost is the one bind address (rule 17, F-LS 4.7, 12.4): this PC only.
const BindHost = "127.0.0.1"
