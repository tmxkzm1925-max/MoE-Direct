// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"regexp"
	"strings"
	"testing"
)

// uxr4Fake is the fake with the release adapter's launcher values (wire Launcher): the address,
// the check run, the cap and the two switches are ported; «on open» comes as «not ported», the
// days logs are kept are not measured, and before a start there is no setup folder and no
// server log file.
func uxr4Fake() *FakeBackend {
	f := NewFake()
	f.Info = LauncherInfo{Address: "127.0.0.1 · port from the catalog (8093)", OnOpen: "not ported", Smoke: SmokeOff,
		CacheCap: "off", WarmStart: "on", Autosave: "on", LogDir: `C:\Users\u\AppData\Local\MoE-Direct\logs`,
		SettingsDir: `C:\Users\u\AppData\Local\MoE-Direct\presets`}
	f.Cat.Folders = []Folder{{Path: `D:\moe-models`, State: FolderWatching, Found: 9, Skipped: []string{
		"DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf  architecture deepseek4 has no catalog profile and no template",
		"Kimi-K3-dense-Q3_K-pure-fallback93.gguf  the GGUF header cannot be read as a MoE model"}}}
	return f
}

func withNotPorted(t *testing.T, on bool) {
	t.Helper()
	old := showNotPorted
	showNotPorted = on
	t.Cleanup(func() { showNotPorted = old })
}

// TestUXR4NotPortedHidden: findings 8 and 10 (and 20 (c) on 2 Server). With the switch off, as
// released, no 5 Settings row shows «not ported»/«미이식», ↑↓ never stops on a hidden row, and
// the 2 Server status line has no «waiting: not ported»; with it on, the rows come back.
func TestUXR4NotPortedHidden(t *testing.T) {
	withNotPorted(t, false)
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, uxr4Fake(), lang, 120, 30)
		hs.keys("5")
		p := hs.plain()
		for _, w := range []string{"not ported", "미지원", "on open", "logs kept", "로그 보관 기간", "로그 파일", "log file"} {
			if strings.Contains(p, w) {
				t.Fatalf("%s: switch off, 5 Settings shows %q:\n%s", lang, w, p)
			}
		}
		for _, w := range []string{hs.m.r.T("language"), hs.m.r.T("warm start"), hs.m.r.T("settings")} {
			if !strings.Contains(p, w) {
				t.Fatalf("%s: switch off, 5 Settings lacks %q:\n%s", lang, w, p)
			}
		}
	}
	// ↑↓ over the launcher rows: rows 3 (on open) and 4 (logs kept) are stepped over, both ways
	hs := newH(t, uxr4Fake(), LangEnglish, 120, 30)
	hs.keys("5", "tab")
	var seen []int
	for i := 0; i < 8; i++ {
		hs.keys("down")
		seen = append(seen, hs.m.set5Row)
	}
	for i := 0; i < 8; i++ {
		hs.keys("up")
		if hs.m.set5Sec == secLauncher {
			seen = append(seen, hs.m.set5Row)
		}
	}
	for _, r := range seen {
		if r == 3 || r == 4 {
			t.Fatalf("a hidden launcher row was chosen: %v", seen)
		}
	}
	// the writes section: the setup folder and the log file have no path yet
	hs.keys("tab", "tab")
	hs.keys("down")
	if hs.m.set5Sec != secWrites || hs.m.set5Row != 1 {
		t.Fatalf("writes: first shown row sec %d row %d, want logs (1)", hs.m.set5Sec, hs.m.set5Row)
	}
	hs.keys("down")
	if hs.m.set5Row != 3 {
		t.Fatalf("writes: ↓ from logs went to %d, want settings (3)", hs.m.set5Row)
	}

	// 2 Server serving, the waiting count not measured (finding 20 (c))
	serve := func() string {
		f := uxr4Fake()
		for i := range f.Statuses {
			f.Statuses[i].Requests, f.Statuses[i].Errors, f.Statuses[i].Waiting = K(1), K(0), U
			f.Statuses[i].Slow = false
		}
		hs := newH(t, f, LangKorean, 120, 30)
		hs.keys(serving()...)
		hs.beat()
		return hs.plain()
	}
	if p := serve(); strings.Contains(p, "미지원") || !strings.Contains(p, "1 요청 완료") {
		t.Fatalf("switch off, 2 Server status line:\n%s", p)
	}

	// the switch on: the rows and the waiting part come back
	withNotPorted(t, true)
	hs = newH(t, uxr4Fake(), LangEnglish, 120, 30)
	hs.keys("5")
	p := hs.plain()
	for _, w := range []string{"on open", "logs kept", "not ported", "log file"} {
		if !strings.Contains(p, w) {
			t.Fatalf("switch on, 5 Settings lacks %q:\n%s", w, p)
		}
	}
	if p := serve(); !strings.Contains(p, "대기: 미지원") {
		t.Fatalf("switch on, 2 Server lacks the waiting part:\n%s", p)
	}
}

// TestUXR4SectionFocus: finding 9. The focused section is marked on its whole title, in both
// languages; an unfocused title carries no marker.
func TestUXR4SectionFocus(t *testing.T) {
	withNotPorted(t, false)
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, uxr4Fake(), lang, 120, 30)
		hs.keys("5")
		r := hs.m.r
		titles := []string{r.T("MODEL FOLDERS") + " · " + r.T("where models are found"),
			r.T("LAUNCHER") + " · " + r.T("same for every model"), r.T("WHERE THE LAUNCHER WRITES")}
		for sec := 0; sec < 3; sec++ {
			p := hs.plain()
			for i, tt := range titles {
				marked := strings.Contains(p, "▶ "+tt)
				if marked != (i == hs.m.set5Sec) || !strings.Contains(p, tt) {
					t.Fatalf("%s: section %d focused, title %q marked %v:\n%s", lang, hs.m.set5Sec, tt, marked, p)
				}
			}
			hs.keys("tab")
		}
	}
}

// TestUXR4Wording: finding 9's rename and finding 8b's box at about 120 columns.
func TestUXR4Wording(t *testing.T) {
	withNotPorted(t, false)
	hs := newH(t, uxr4Fake(), LangKorean, 120, 30)
	hs.keys("5")
	p := hs.plain()
	if !regexp.MustCompile(`시작 후 응답 시험 +끔`).MatchString(p) || strings.Contains(p, "작동 확인") { // label, a gap, the value
		t.Fatalf("the check run row in Korean:\n%s", p)
	}
	if !strings.Contains(p, "지원 안 되는 GGUF · 건너뜀 · D:\\moe-models") {
		t.Fatalf("the Korean box title:\n%s", p)
	}
	hs = newH(t, uxr4Fake(), LangEnglish, 120, 30)
	hs.keys("5")
	p = hs.plain()
	if !strings.Contains(p, "UNSUPPORTED GGUF · SKIPPED · D:\\moe-models") {
		t.Fatalf("the English box title:\n%s", p)
	}
	// each reason is read in full: the box's text, its lines joined, holds the whole reason
	var box []string
	for _, l := range strings.Split(p, "\n") {
		if i := strings.LastIndex(l, "││"); i >= 0 {
			l = l[i+len("││"):]
		} else if i := strings.Index(l, "│"); i >= 0 && strings.Count(l, "│") >= 2 {
			parts := strings.Split(l, "│")
			l = parts[len(parts)-2]
		}
		box = append(box, strings.TrimSpace(strings.Trim(l, "│ ")))
	}
	flat := strings.Join(strings.Fields(strings.Join(box, " ")), " ")
	for _, why := range []string{"architecture deepseek4 has no catalog profile and no template",
		"the GGUF header cannot be read as a MoE model"} {
		if !strings.Contains(flat, why) {
			t.Fatalf("reason %q not read in full:\n%s", why, p)
		}
	}
}
