package wire

// UX-R21 (version 8; the user's acceptance item 32, reviews/_user_8_LAUNCHER-REVIEW_12.md,
// screenshots 185445 and 185529): DeepSeek-V4.1 (architecture deepseek4, «✕ unsupported» in the
// 1 Models list) was refused with «the model file is missing or cannot be used» on the 1 Models
// verdict line and the 3 Memory checklist «refused:» line, a file problem it does not have. The
// refusal now names its cause: an unsupported model structure, a missing model file, or a model
// file that cannot be read. core's reasons for each cause are pinned by core
// TestUXR21ModelPathCauses; here the screen text this package makes of them is drawn through
// the fake backend, in English and Korean at 120x40 and 80x28.

import (
	"fmt"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
)

// uxr21Flat is the screen with each run of spaces and box edges made one space, so a wrapped
// line reads on.
func uxr21Flat(p string) string {
	p = strings.NewReplacer("│", " ", "╭", " ", "╮", " ", "╰", " ", "╯", " ", "─", " ").Replace(p)
	return strings.Join(strings.Fields(p), " ")
}

func TestUXR21RefusalCause(t *testing.T) {
	const old, oldKo = "the model file is missing or cannot be used", "모델 파일이 없거나 쓸 수 없음"
	for _, c := range []struct{ what, reason, en, ko string }{
		{"DeepSeek-V4.1", "unsupported GGUF: no catalog profile matches the header fingerprint",
			"unsupported model structure", "지원하지 않는 모델 구조"},
		{"missing", `model file not found: D:\m\gone.gguf`, "the model file is missing", "모델 파일이 없음"},
		{"unreadable", `unsupported GGUF: GGUF magic mismatch: D:\m\junk.gguf`,
			"the model file cannot be read", "모델 파일을 읽을 수 없음"},
	} {
		le := &core.LaunchError{Status: core.StatusFailModelPath, Reason: c.reason}
		if got, _ := userReason(le, nil); got != c.en {
			t.Fatalf("%s: userReason %q, want %q", c.what, got, c.en)
		}
		for _, lang := range []string{ui.LangEnglish, ui.LangKorean} {
			for _, sz := range [][2]int{{120, 40}, {80, 28}} {
				what := fmt.Sprintf("%s %s %dx%d", c.what, lang, sz[0], sz[1])
				f := ui.NewFake()
				f.Cat.Models = append([]ui.ModelRow{{ID: `D:\m\DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf`,
					Name: "DeepSeek-V4.1-Flash-MXFP4-engram", Present: true, Quant: "MXFP4_MOE", Path: `D:\m\x.gguf`,
					SizeGB: ui.K(467), NeedGB: ui.U, FreeRAMGB: ui.K(24.6), LeftGB: ui.U, Fit: ui.FitUnsupported}}, f.Cat.Models...)
				f.Pre, _ = preflightFrom(f.Cat.Models[0].ID, nil, le)
				s := &uxr8Screen{t: t, clock: time.Date(2026, 10, 13, 18, 54, 0, 0, time.UTC)}
				s.m = ui.New(f, ui.Options{Lang: lang, Width: sz[0], Height: sz[1],
					Now: func() time.Time { return s.clock },
					Tick: func(_ time.Duration, fn func(time.Time) tea.Msg) tea.Cmd {
						s.ticks = append(s.ticks, fn)
						return nil
					}})
				s.run(s.m.Init())
				s.fire()
				want, cannot, refused, stale := c.en, "cannot start", "refused", old
				if lang == ui.LangKorean {
					want, cannot, refused, stale = c.ko, "시작 불가", "거절", oldKo
				}
				p1 := uxr21Flat(s.plain())
				if !strings.Contains(p1, "✕ "+cannot+" · "+want+" · ") || strings.Contains(p1, stale) {
					t.Fatalf("%s: 1 Models verdict lacks «✕ %s · %s · …» or keeps «%s»:\n%s", what, cannot, want, stale, s.plain())
				}
				s.key("3")
				p3 := uxr21Flat(s.plain())
				if !strings.Contains(p3, refused+": "+want) || strings.Contains(p3, stale) {
					t.Fatalf("%s: 3 Memory lacks «%s: %s» or keeps «%s»:\n%s", what, refused, want, stale, s.plain())
				}
				t.Logf("%s: «✕ %s · %s» · «%s: %s»", what, cannot, want, refused, want)
			}
		}
	}
}
