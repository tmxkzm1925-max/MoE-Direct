// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestUXR9SetupBox: finding 14. The 2 Server setup box: in Korean the labels are the English
// words «what to do», «where», «disk», «time», the backend's sentence reads in Korean, the
// first time reads «첫 시도라 확인 불가» and the step line «이후 단계: …»; English keeps its
// text with «what to do» for «does».
func TestUXR9SetupBox(t *testing.T) {
	does := "copies the model's experts into one streaming file (experts.bin), then checks it · first removes 2 earlier file(s) (3.0 GB freed, counted as free)"
	for _, c := range []struct {
		lang string
		want []string // regexps over the plain frame
		lack []string
	}{
		{LangKorean, []string{
			`│ what to do +모델의 전문가 가중치를 스트리밍용 파일 하나\(experts\.bin\)로 복사한 뒤 확인 · 먼저 이전 파일 2개 지움 \(3\.0 GB 확보, 여유로 셈\)`,
			`│ where +D:\\moe-models\\repack\\qwen3-30b\\virtual`, `│ disk +약 17\.5 GB 씀`, `│ time +첫 시도라 확인 불가`,
			`│ 이후 단계: 자동 확인 검증 → 디스크 읽기 속도 확인 → 서버 자동 시작`},
			[]string{"하는 일", "쓰는 곳", "첫 리팩이라 모름", "그 뒤 검증", "copies the model"}},
		{LangEnglish, []string{
			`│ what to do +copies the model's experts into one streaming file \(experts\.bin\), then checks it · first`,
			`│ writes to +D:\\moe-models`, `│ disk +about 17\.5 GB written`, `│ time +first repack: not known yet`,
			`│ then: check → speed check → the server starts by itself`},
			[]string{"│ does "}},
	} {
		f := NewFake()
		f.Plan.Does = does
		hs := newH(t, f, c.lang, 160, 30)
		hs.keys("down", "down", "down", "enter")
		hs.want("HP")
		p := hs.plain()
		for _, w := range c.want {
			if !regexp.MustCompile(w).MatchString(p) {
				t.Fatalf("%s: the setup box lacks %s:\n%s", c.lang, w, p)
			}
		}
		mustLack(t, p, c.lack...)
	}
}

// uxr9Fail is a failed start with a reason longer than any box, the steps reached and a log tail.
func uxr9Fail() *FakeBackend {
	f := NewFake()
	f.StartEvs = []StartEvent{{Model: "qwen35-122b", Steps: []Step{{Name: "bundle", State: StepDone}, {Name: "model", State: StepDone},
		{Name: "start server", State: StepFailed, Detail: "the model file is gone"}},
		Failure: &Failure{Stage: "start", What: "the server did not start", Why: FakeLongRefusal, At: time.Date(2026, 10, 13, 6, 1, 13, 0, time.UTC),
			LogTail: []string{"[server] llama_model_load: failed to open the model file"}, LogFile: `C:\Users\u\AppData\Local\MoE-Direct\logs\launcher.log`}}}
	return f
}

// TestUXR9FailurePage: finding 15 (b). The failed start is titled «서버 가동 실패» / «server start
// failed»; its reason shows once and whole, wrapped (at 120 columns and at a narrow 60); at 120
// the page holds the run's settings, the start steps and the last log lines.
func TestUXR9FailurePage(t *testing.T) {
	for _, lang := range []string{LangKorean, LangEnglish} {
		for _, w := range []int{120, 60} {
			hs := newH(t, uxr9Fail(), lang, w, 44)
			hs.keys("enter")
			hs.want("H7")
			p := hs.plain()
			r := hs.m.r
			head := strings.Split(p, "\n")[2]
			if !strings.Contains(head, "Qwen3.5-122B · "+r.T("server start failed")) {
				t.Fatalf("%s %d: the head is %q:\n%s", lang, w, head, p)
			}
			mustLack(t, p, "서버가 시작되지 않음", "did not start", "시작 안 됨")
			if n := strings.Count(flat(p), flat(FakeLongRefusal)); n != 1 {
				t.Fatalf("%s %d: the whole reason is shown %d times:\n%s", lang, w, n, p)
			}
			for _, l := range strings.Split(p, "\n") {
				if strings.Contains(l, "…") && !strings.Contains(l, r.T("full log")) { // the log path alone is shortened
					t.Fatalf("%s %d: a line is cut: %q\n%s", lang, w, l, p)
				}
			}
			if w == 120 {
				mustHave(t, p, r.T("APPLIED SETTINGS, change in Models"), r.T("START STEPS"), r.T("the model file is gone"),
					r.T("LAST LOG LINES"), "failed to open the model file", r.T("full log"), r.T("FIX"), r.T("try again"))
			}
		}
	}
	// at 120x30 the fill takes only the room it has: every box closes above the footer
	for _, lang := range []string{LangKorean, LangEnglish} {
		hs := newH(t, uxr9Fail(), lang, 120, 30)
		hs.keys("enter")
		if p := hs.plain(); strings.Count(p, "╭") != strings.Count(p, "╰") || strings.Count(flat(p), flat(FakeLongRefusal)) != 1 {
			t.Fatalf("%s 120x30: a box is cut by the window:\n%s", lang, p)
		}
	}
	if r := (rc{lang: LangKorean}); r.T("server start failed") != "서버 가동 실패" || r.T("the server did not start") != "서버 가동 실패" {
		t.Fatalf("the Korean title: %q %q", r.T("server start failed"), r.T("the server did not start"))
	}
}

// TestUXR9Footer: finding 15 (b). The footer keys read «1-5 pages · ? help · q exit» in both
// languages; «장소» names a page nowhere in Korean.
func TestUXR9Footer(t *testing.T) {
	for _, lang := range []string{LangKorean, LangEnglish} {
		for _, keys := range [][]string{{"1"}, {"5"}, {"2"}} {
			hs := newH(t, uxr4Fake(), lang, 120, 30)
			hs.keys(keys...)
			ls := strings.Split(hs.plain(), "\n")
			last := ls[len(ls)-1]
			for _, w := range []string{"1-5 pages", "? help", "q exit"} {
				if !strings.Contains(last, w) {
					t.Fatalf("%s %v: the key line lacks %q: %q", lang, keys, w, last)
				}
			}
			mustLack(t, last, "quit", "장소", "도움말", "끝내기")
		}
	}
	for k, v := range table[LangKorean] {
		if strings.Contains(v, "장소") {
			t.Fatalf("the Korean text of %q still says 장소: %q", k, v)
		}
	}
}

// TestUXR9ServingCue: finding 19. While a server runs, the 2 Server place of the top row carries
// «●» (lit or not, on any page) beside the state at the right; with no server and after a stop
// it is plain.
func TestUXR9ServingCue(t *testing.T) {
	for _, lang := range []string{LangKorean, LangEnglish} {
		hs := newH(t, NewFake(), lang, 120, 30)
		place := "2 " + hs.m.r.T("Server")
		nav := func() string { return strings.Split(hs.plain(), "\n")[0] }
		if strings.Contains(nav(), place+" ●") {
			t.Fatalf("%s: no server, yet %q: %q", lang, place+" ●", nav())
		}
		hs.keys(serving()...)
		hs.beat()
		for _, page := range []string{"2", "1", "3", "5"} {
			hs.keys(page)
			if n := nav(); !strings.Contains(n, place+" ●") || !strings.Contains(n, "● ") {
				t.Fatalf("%s page %s: serving, the top row is %q", lang, page, n)
			}
		}
		hs.keys("2", "ctrl+f")
		hs.want("H12")
		if n := nav(); strings.Contains(n, place+" ●") || !strings.Contains(n, place) {
			t.Fatalf("%s: stopped, the top row is %q", lang, n)
		}
	}
}

// pfCodes are every code the prefetch read-back line can show (source coordinates of the
// launcher tree and of the released engine source): the engine's prefetch_off_reason
// (ggml-moe-phaseb.cpp: unset :1307, invalid_k :1311, invalid_range :1312, invalid_n :1315,
// adapter_none :2366), the candidate's off reasons (core config.go resolvePrefetchManual
// :551-638 with each prefetch_evidence of config.go:431; preflight.go:645) and its provenance
// labels (core manualpf.go:32-34).
var pfCodes = []string{"unset", "invalid_k", "invalid_n", "invalid_range", "adapter_none",
	"catalog_semantic_invalid", "probe_failed", "identity_not_exact", "adapt_controller_not_shipped_phase5",
	"not_opted_in_evidence_unverified", "not_opted_in_evidence_observe", "not_opted_in_evidence_causal_replay",
	"not_opted_in_evidence_paired_live", "phase4_hold_unresolved", "init_t_out_of_range", "qd_below_prefetch_depth",
	"derived_t_mismatch", "catalog-validated", "init_v1-unvalidated", "env-override"}

// TestUXR9PrefetchWords: finding 20 (d). The prefetch line shows each code as a sentence in both
// languages and never the code; a code the table does not know reads «details on 4 Log».
func TestUXR9PrefetchWords(t *testing.T) {
	var keys []string
	for k := range pfReasons {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	want := append([]string{}, pfCodes...)
	sort.Strings(want)
	if strings.Join(keys, " ") != strings.Join(want, " ") {
		t.Fatalf("mapped codes %v, the line can show %v", keys, want)
	}
	line := func(lang string, words ...string) string {
		f := NewFake()
		f.Statuses[0].Prefetch = words
		hs := newH(t, f, lang, 160, 34)
		hs.keys(serving()...)
		hs.beat()
		for _, l := range strings.Split(hs.plain(), "\n") {
			if strings.HasPrefix(strings.TrimLeft(l, " │"), hs.m.r.T("prefetch")+" ") {
				return l
			}
		}
		t.Fatalf("%s: no prefetch line:\n%s", lang, hs.plain())
		return ""
	}
	// the user's screenshot (064114)
	// UX-KR (decision 4): the engine's part only, the candidate in 4 Log
	if l := line(LangKorean, "engine off", "unset", "candidate off", "not_opted_in_evidence_observe"); !strings.Contains(l,
		"미적용 · K·N 값 설정 미확인") || strings.Contains(l, "런처 요청") || strings.Contains(l, "기본은 꺼짐") {
		t.Fatalf("the Korean line: %q", l)
	}
	for _, lang := range []string{LangKorean, LangEnglish} {
		for _, c := range pfCodes {
			l := line(lang, "engine off", c)
			if strings.Contains(l, c) || !strings.Contains(l, tr(lang, pfReasons[c])) {
				t.Fatalf("%s %s: %q", lang, c, l)
			}
		}
		if l := line(lang, "engine off", "zz_new_reason"); strings.Contains(l, "zz_new_reason") || !strings.Contains(l, tr(lang, "details on 4 Log")) {
			t.Fatalf("%s: an unknown code: %q", lang, l)
		}
	}
}

// TestUXR9NotPortedServer: Middle's addendum (1). With the switch off, as released, the 2 Server
// places the release adapter reaches show no «not ported»: the first setup's time left, the
// request counts before the first count, the run summary's cache save; with it on they return.
func TestUXR9NotPortedServer(t *testing.T) {
	render := func(on bool) (setup, serve, stopped string) {
		withNotPorted(t, on)
		f := NewFake()
		f.StartHold = true
		hs := newH(t, f, LangEnglish, 120, 30)
		hs.keys("down", "down", "down", "enter", "enter")
		hs.want("H6")
		setup = hs.plain()
		f = NewFake()
		for i := range f.Statuses {
			f.Statuses[i].Requests, f.Statuses[i].CtxTotal = U, U
		}
		hs = newH(t, f, LangEnglish, 120, 30)
		hs.keys(serving()...)
		hs.beat()
		serve = hs.plain()
		hs.keys("ctrl+f")
		hs.want("H12")
		return setup, serve, hs.plain()
	}
	setup, serve, stopped := render(false)
	if !regexp.MustCompile(`LEFT[^\n]*\n[^\n]*first time`).MatchString(setup) || strings.Contains(setup, "not ported") {
		t.Fatalf("switch off, the first setup:\n%s", setup)
	}
	mustLack(t, serve, "request and error counts", "not ported")
	mustHave(t, stopped, "prompt cache save: details on 4 Log")
	mustLack(t, stopped, "prompt cache save: not ported")
	setup, serve, stopped = render(true)
	mustHave(t, setup, "not ported")
	mustHave(t, serve, "request and error counts: not ported")
	mustHave(t, stopped, "prompt cache save: not ported")
}

// TestUXR9AddressFold: Middle's addendum (2). In the 5 Settings launcher box the address
// «127.0.0.1 · port from the catalog (8093)» folds onto a second line at 120 columns, read
// whole, in both languages.
func TestUXR9AddressFold(t *testing.T) {
	withNotPorted(t, false)
	for _, lang := range []string{LangEnglish, LangKorean} {
		hs := newH(t, uxr4Fake(), lang, 120, 30)
		hs.keys("5")
		p := hs.plain()
		var box []string // the launcher box: the text between the first two │ of each line
		for _, l := range strings.Split(p, "\n") {
			if parts := strings.Split(l, "│"); len(parts) >= 3 {
				box = append(box, parts[1])
			}
		}
		flatBox := strings.Join(strings.Fields(strings.Join(box, " ")), " ")
		if !strings.Contains(flatBox, hs.m.r.T("address")+" 127.0.0.1 · port from the catalog (8093)") {
			t.Fatalf("%s: the address is not read whole:\n%s", lang, p)
		}
		for _, l := range box {
			if strings.Contains(l, "127.0.0.1") && strings.Contains(l, "…") {
				t.Fatalf("%s: the address row is cut: %q", lang, l)
			}
		}
		if !regexp.MustCompile(`\n │ +\(8093\) +│`).MatchString(p) {
			t.Fatalf("%s: the address does not fold onto a second line:\n%s", lang, p)
		}
	}
}

// TestUXR9TTFTLabel: Middle's addendum (UX-R11 landing 4.4). The TTFT figure is the last
// finished request's: while a new request is read or answered it was labelled «last request» /
// «직전 요청», with none running «this request» / «이번 요청». Since UX-R27 (review item 37) TTFT
// is a row of the PREFILL panel, now · avg · min · max over the finished requests of the last 60
// minutes: now is the estimate «~n» while a prompt is read and «—» otherwise, so the label went.
func TestUXR9TTFTLabel(t *testing.T) {
	for _, lang := range []string{LangEnglish, LangKorean} {
		for _, c := range []struct {
			ph  ReqPhase
			now string
		}{{ReqPrefill, "~10"}, {ReqDecode, "—"}, {ReqNone, "—"}} {
			f := NewFake()
			for i := range f.Statuses {
				f.Statuses[i].Phase, f.Statuses[i].TTFTs, f.Statuses[i].Slow = c.ph, K(10.4), false
				f.Statuses[i].PrefillDone, f.Statuses[i].PrefillTotal, f.Statuses[i].PrefillTokS = K(3000), K(6000), K(300)
			}
			hs := newH(t, f, lang, 120, 40)
			hs.keys(serving()...)
			hs.beat()
			p := hs.plain()
			box := uxr27Box(p, tr(lang, "PREFILL · last 60 min"))
			if len(box) < 3 || box[2] != "TTFT "+c.now+" — — — s" {
				t.Fatalf("%s phase %d: the PREFILL panel's TTFT row is not «TTFT %s — — — s» (%q):\n%s", lang, c.ph, c.now, box, p)
			}
			for _, old := range []string{"last request", "직전 요청"} {
				if strings.Contains(p, old) {
					t.Fatalf("%s phase %d: the old label %q shows:\n%s", lang, c.ph, old, p)
				}
			}
		}
	}
}
