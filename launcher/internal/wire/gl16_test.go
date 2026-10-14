package wire

import (
	"strings"
	"testing"
	"time"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// GL-16: the launcher settings of 5 Settings go with every start request; a refused value
// changes nothing and gives the settings-typed text (F-LS 11.4); a screen start refused for a
// setting does not borrow the argument-driven mode's hint.
func TestGL16LauncherSettings(t *testing.T) {
	b := testBackend()
	req := b.request(`D:\m.gguf`, ui.Settings{})
	if req.PromptCacheCap != (core.CapSetting{}) || req.Switches == nil || *req.Switches != core.DefaultSwitches || req.FromArguments {
		t.Fatalf("defaults %+v", req)
	}
	for _, c := range []struct {
		id        ui.ActionID
		arg, want string
	}{{ui.ActSetCacheCap, "1024", "1024"}, {ui.ActSetCacheCap, "off", "off"}, {ui.ActSetCacheCap, "", "off"},
		{ui.ActSetCacheCap, "2048", "2048"}, {ui.ActSetWarmStart, "off", "off"}, {ui.ActSetAutosave, "off", "off"}} {
		if _, err := b.Action(c.id, c.arg); err != nil {
			t.Errorf("%s %s: %v", c.id, c.arg, err)
		}
		m := core.LauncherSettingsMap(b.cacheCap, b.switches)
		key := map[ui.ActionID]string{ui.ActSetCacheCap: "prompt_cache_cap", ui.ActSetWarmStart: "warmstart", ui.ActSetAutosave: "autosave"}[c.id]
		if m[key] != c.want {
			t.Errorf("%s %s: held %q", c.id, c.arg, m[key])
		}
	}
	req = b.request(`D:\m.gguf`, ui.Settings{})
	if req.PromptCacheCap != (core.CapSetting{Set: true, Value: 2048}) || req.Switches.WarmStart || req.Switches.Autosave {
		t.Errorf("the request does not carry them: %+v %+v", req.PromptCacheCap, *req.Switches)
	}
	if _, err := b.Action(ui.ActSetCacheCap, "100"); err == nil ||
		err.Error() != "outside its range · allowed 0 (off) or 256..8192 MiB · the value is kept, nothing applied" {
		t.Errorf("refusal %v", err)
	}
	if b.cacheCap.Value != 2048 {
		t.Errorf("a refused value was applied: %+v", b.cacheCap)
	}
	if _, err := b.Action(ui.ActSetWarmStart, "maybe"); err == nil {
		t.Error("a bad switch value")
	}
	if savedSettingsNotUsed != core.NoticePresetNotUsed {
		t.Errorf("the screens' saved-settings text %q differs from core's %q", savedSettingsNotUsed, core.NoticePresetNotUsed)
	}
	ev := core.StartEvent{Final: true, Status: core.StatusFailCustomArgs, ExitCode: 3,
		Err: &core.LaunchError{Status: core.StatusFailCustomArgs, Reason: "x"}}
	if f := failureFrom(ev, time.Unix(1, 0), nil); strings.Contains(f.What, "argument") || f.What == "" {
		t.Errorf("screen failure %q", f.What)
	}
}
