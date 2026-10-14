package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// UX-KR (version 8): the Korean mode as the user chose it from the Korean string table
// (reviews/_landing_8_UX-R5_v8-krtab.md, 661 rows) with the decisions of the brief
// (reviews/_brief_8_UX-KR_v8-uxkr.md): (1) the tab names 1-5 stay English, also inside Korean
// sentences; (2) «fits» stays English; (3) the 2 Server box title «적용된 설정값, 변경은 Models»;
// (4) the prefetch row holds the engine's part only, «미적용 · K·N 값 설정 미확인»; (5) decode,
// prefill and TTFT stay English inside Korean strings; (6) the 4 Log header replaced; (7) the
// labels of the 2 Server box dim and its values bright; (8) one wording of the list lock.

// uxkrWant is the Korean the map holds after UX-KR, per English key: every E and K row of the
// table applied by key (the K candidates outside the map added), the decisions' entries, and
// each entry that carried «디코드» or «프리필». The pairs were extracted from the table file by
// the department's script (krparse.py, plan.py; landing section 3) and are embedded here.
var uxkrWant = map[string]string{
	"%s is not ported":                                "%s 미지원",                   // K-assembled 오류/wire/map.go:681
	"1 Models picks a model and starts it":            "1 Models 에서 모델을 골라 시작",    // K 2/153
	"2 Server shows each step":                        "2 Server 에서 단계별 확인",       // K 1/156
	"2 Server · the big tile through one request":     "2 Server · 요청 하나 동안의 큰 칸", // K 도움말·알림/157
	"3 Memory shows what changed around the slowdown": "3 Memory 에서 느려진 때의 변화 확인", // K 2/158
	"DECODE":                              "DECODE",                  // D5 title (upper case as the English tile titles, as DISK)
	"DECODE · tok/s · last 60 min":        "DECODE · tok/s · 최근 60분", // D5 title (upper case as the English tile titles, as DISK)
	"DECODE · tok/s · whole run":          "DECODE · tok/s · 전체 실행",  // D5 title (upper case as the English tile titles, as DISK)
	"DISK":                                "DISK",                    // E 2/168
	"K·N setting unconfirmed":             "K·N 값 설정 미확인",            // decision 4
	"LAST DECODE":                         "최근 DECODE",               // D5 title (upper case as the English tile titles, as DISK)
	"Log":                                 "Log",                     // E 탭·머리줄·바닥줄/4
	"Memory":                              "Memory",                  // E 탭·머리줄·바닥줄/3
	"Models":                              "Models",                  // E 탭·머리줄·바닥줄/1
	"PAGEFILE":                            "페이징 파일",                  // K 3/183
	"PREFILL":                             "PREFILL",                 // D5 title (upper case as the English tile titles, as DISK)
	"PREFILL · tok/s per request":         "PREFILL · 요청별 tok/s",     // D5 title (upper case as the English tile titles, as DISK)
	"RAM+pagefile":                        "RAM+페이징 파일",              // K 3/190
	"Server":                              "Server",                  // E 탭·머리줄·바닥줄/2
	"Settings":                            "Settings",                // E 탭·머리줄·바닥줄/5
	"Settings adds more":                  "Settings 에서 더 추가",        // K 1/450
	"WHERE THE LAUNCHER WRITES":           "런처 저장 위치",                // K 5/208
	"a setting is outside its range":      "설정 값이 범위 밖",              // K-out 오류/wire/map.go:704
	"a setting of this start was refused": "이번 시작의 설정 하나가 거부됨",       // K-out 오류/wire/tracks.go:139
	"a short warmup request right after ready · on by default":      "준비 직후 짧은 예열 요청 · 기본은 켬",               // K-out 1/wire/map.go:290
	"address = where chat apps connect · next start · this PC only": "주소 = 채팅 앱이 연결하는 곳 · 다음 시작부터 · 이 PC 만", // K 5/109
	"again: exit": "한 번 더: exit", // K 탭·머리줄·바닥줄/25
	"an earlier setup was cut off · ⏎ opens the setup plan":                                         "이전 준비가 중간에 끊김 · ⏎ 준비 계획 열기",                                     // K-out 오류/wire/map.go:659
	"APPLIED SETTINGS, change in Models":                                                            "적용된 설정값, 변경은 Models",                                            // decision 3
	"auto = the sweep binding, else QD 1 · virtual: default QD 8, below the wave width cannot work": "auto = QD 스윕 결과, 없으면 QD 1 · virtual: 기본 QD 8, wave 폭 미만은 동작 불가", // K-out 1/wire/map.go:288
	"available":               "사용 가능",    // K-out 3/ui/v_memory.go:350
	"average of the last 5 s": "최근 5초 평균", // K 도움말·알림/228
	"broken":                  "broken",   // E-out 1/ui/v_models.go:236
	"budget":                  "예산",       // K-out 3/ui/v_memory.go:149
	"candidate off":           "런처 요청 꺼짐", // K 2/132
	"candidate on":            "런처 요청 켜짐", // K 2/131
	"catalog defaults":        "카탈로그 기본값", // K-out 1/wire/map.go:425
	"catalog or init · a set K and N pair is the run's pair": "catalog 또는 init · K·N 을 정하면 그 쌍으로 실행", // K-out 1/wire/map.go:291
	"check": "확인", // K-out 2/wire/fix2.go:104
	"check run = a short chat test after a start · off, on, or once per model · results in 4 Log": "시작 후 응답 시험 = 시작 뒤 짧은 대화로 응답 확인 · 끔·켬·최초 1회 · 결과는 4 Log", // decision 1 (tab name in a decided string)
	"clean up": "정리", // K-out 2/wire/fix2.go:100
	"copies the model's experts into one streaming file (experts.bin), then checks it": "모델의 전문가 가중치를 스트리밍용 파일 하나(experts.bin)로 복사한 뒤 확인", // K 2/560
	"days":   "일",      // K-out 5/ui/v_other.go:247
	"decode": "decode", // D5 2/245
	"defaults · the saved settings belong to another file": "기본값 · 저장된 설정은 다른 파일의 것", // K-out 1/wire/wire.go:342
	"details on 3 Memory": "자세히 3 Memory", // decision 1 (tab name in a decided string)
	"details on 4 Log":    "자세히 4 Log",    // decision 1 (tab name in a decided string)
	"empty":               "비어 있음",        // K-out 1/ui/v_models.go:937
	"esc goes back to where you were · tab shows the big tile of 2 Server": "esc 이전 화면으로 · tab 2 Server 의 큰 칸 보기", // K 도움말·알림/254
	"every":        "주기",     // K-out 3/ui/v_memory.go:60
	"expert cache": "전문가 캐시", // K(already) 3/464
	"expert cache only · auto = the PowerShell autotune": "전문가 캐시만 · auto = PowerShell 자동 조정", // K-out 1/wire/map.go:286
	"expert cache only · the whole model uses more":      "전문가 캐시만 · 모델 전체는 더 씀",              // K 1/257
	"expert cache split": "전문가 캐시 나눔", // K(already) 3/258
	"fit":                "fit",       // E 1/262
	"fits":               "fits",      // E 1/263
	"force kill":         "강제 종료",     // K-out 2/wire/tracks.go:175
	"free room on the drive, then ⏎ on 1 Models · esc goes back": "드라이브 공간을 비운 뒤 1 Models 에서 ⏎ · esc 돌아가기", // K 2/92
	"how much text one chat can hold":                            "대화 하나가 담을 수 있는 글의 양",                    // K-out 1/wire/map.go:287
	"identify model":                                             "모델 확인",                                  // K-out 2/wire/tracks.go:19
	"it shows up here by itself, no restart needed":              "자동으로 여기 나타남, 다시 시작 필요 없음",               // K 1/472
	"language changes at once (this run) · the other launcher settings: not ported · ↑ past the first row goes back to the folders": "언어는 바로 바뀜 (이번 실행) · 다른 런처 설정: 미지원 · 첫 행에서 ↑ 하면 폴더로", // K 5/477
	"language changes at once · others: not ported": "언어는 바로 바뀜 · 나머지: 미지원", // K 5/478
	"launcher locks":                      "런처 잠금",                         // K-out 2/wire/tracks.go:20
	"live · 5 s average":                  "실시간 · 5초 평균",                   // K 2/294
	"logs kept":                           "로그 보관 기간",                      // K 5/298
	"logs kept = days before old logs go": "로그 보관 기간 = 오래된 로그를 지우기까지의 날 수", // K 5/481
	"manual prefetch K, set with N · auto = no pair · not a validated pair: unmeasured":     "수동 프리페치 K, N 과 함께 설정 · auto = 쌍 없음 · 검증된 쌍이 아니면 미측정",     // K-out 1/wire/map.go:292
	"manual prefetch N, set with K · on while QD is at least N+1, else off with its reason": "수동 프리페치 N, K 와 함께 설정 · QD 가 N+1 이상일 때 켜짐, 아니면 이유와 함께 꺼짐", // K-out 1/wire/map.go:293
	"median decode of this model's last 20 replies at a similar context (±4k)":              "비슷한 컨텍스트(±4k)에서 이 모델의 최근 답변 20개 decode 중앙값",              // D5 도움말·알림/304
	"median decode, last 20 replies, similar context":                                       "decode 중앙값, 최근 답변 20개, 비슷한 컨텍스트",                         // D5 도움말·알림/305
	"missing":                  "없음",              // K-out 1/ui/v_models.go:935
	"model settings: 1 Models": "모델 설정: 1 Models", // K 5/313
	"model somewhere else? add that folder in": "모델이 다른 폴더에 있다면 그 폴더를 추가", // K 1/315
	"more":                 "개 더",       // K 3/317
	"new number every 1 s": "1초마다 새 숫자", // K 도움말·알림/321
	"no %s repack files yet · ⏎ opens the setup plan":   "아직 %s 리팩 파일 없음 · ⏎ 준비 계획 열기",   // K-assembled 오류/wire/map.go:663
	"no catalog profile: settings not ported":           "카탈로그 프로필 없음: 설정 미지원",           // K-out 1/wire/wire.go:309
	"no check yet · 1 Models picks a model":             "아직 확인 없음 · 1 Models 에서 모델 고르기", // K 3/486
	"no decode samples in this run":                     "이번 실행에 decode 표본 없음",           // D5 2/487
	"no slot save at this stop: the next start is cold": "이번 정지는 캐시 저장 없음: 다음 시작은 캐시 없이", // K 2/421
	"not applied":                    "미적용",          // decision 4
	"not enough RAM for this budget": "이 예산에 RAM 부족", // K-out 오류/wire/map.go:698
	"not enough free disk space":     "디스크 여유 공간 부족", // K-out 오류/wire/map.go:696
	"not ported":                     "미지원",          // K 5/329
	"nothing is lost on this path · the next start is warm": "이 경로에서 잃는 것 없음 · 다음 시작은 캐시를 이어 씀", // K 2/496
	"on open":                        "on open",              // E 5/340
	"on open = the page shown first": "on open = 처음 보이는 페이지", // K 5/498
	"outside the catalog · %s template settings · performance untested":                                                                            "카탈로그 밖 · %s 템플릿 설정 · 이 설정의 성능은 미확인",                                     // K(already) 1/96
	"outside the catalog: copies the experts the structure template selects, byte for byte, into one streaming file (experts.bin), then checks it": "카탈로그 밖: 구조 템플릿이 고른 전문가 가중치를 바이트 그대로 스트리밍용 파일 하나(experts.bin)로 복사한 뒤 확인", // K 2/562
	"outside the catalog: the template setup is not ported yet":                                                                                    "카탈로그 밖: 템플릿 준비 미지원",                                                     // K-out 오류/wire/map.go:675
	"packed or virtual · a mode without its files opens the setup plan":                                                                            "packed 또는 virtual · 파일이 없는 방식을 고르면 준비 계획이 열림",                           // K-out 1/wire/map.go:289
	"port listener gone": "포트 닫힘 확인", // K-out 2/wire/tracks.go:176
	"prefetch admits nothing: N_cap is 0 on every layer at this QD (the run starts; raise QD to let prefetch work)": "프리페치가 아무것도 받지 않음: 이 QD 에서 모든 레이어의 N_cap=0 (시작은 함 · 프리페치를 쓰려면 QD 를 올림)", // K 도움말·알림/127
	"prefetch off": "프리페치 끔", // K-out 2/wire/map.go:542
	"prefetch on":  "프리페치 켬", // K-out 2/wire/map.go:540
	"prefetch «init» ignored: this model's prefetch is fixed by the catalog": "프리페치 «init» 무시: 이 모델의 프리페치는 카탈로그가 정함", // K-out 도움말·알림/wire/map.go:603
	"prefetch «init» is an unvalidated starting point · no speed claim":      "프리페치 «init» 는 검증 전 시작값 · 속도 주장 없음",    // K-out 도움말·알림/wire/map.go:604
	"prefill": "prefill", // D5 2/346
	"prefill and TTFT are shown and logged, but do not set ▲ slow": "prefill 과 TTFT 는 표시·기록만 하고 ▲ 느림에는 안 씀", // D5 (particle after the English word)
	"prefill and TTFT are shown, do not set ▲ slow":                "prefill 과 TTFT 는 표시만, ▲ 느림에는 안 씀",      // D5 (particle after the English word)
	"put a model file (.gguf) of the catalog into":                 "카탈로그의 모델 파일(.gguf)을 다음 폴더에 넣기",         // K 1/353
	"queue depth": "큐 깊이(QD)", // K-out 2/wire/tracks.go:20
	"re-read every 5 s on 1 Models (names, sizes, times)": "1 Models 에서 5초마다 다시 읽음 (이름, 크기, 시간)", // K 1/539
	"read catalog":          "카탈로그 읽기",   // K-out 2/wire/tracks.go:19
	"refused":               "거절",        // K-out 3/ui/v_memory.go:407
	"repack files":          "리팩 파일",     // K-out 2/wire/tracks.go:19
	"repack output missing": "리팩 결과물 없음", // K-out 오류/wire/map.go:673
	"rewrites the model file once so experts load fast · first run only": "전문가 가중치가 빨리 올라오게 모델 파일을 한 번 다시 씀 · 첫 실행만", // K 2/366
	"saved settings (%s)":                 "저장된 설정 (%s)",        // K-assembled 1/wire/wire.go:340
	"send CTRL_BREAK":                     "CTRL_BREAK 보내기",     // K-out 2/wire/tracks.go:173
	"server args":                         "서버 인자",              // K-out 2/wire/tracks.go:20
	"skipped":                             "건너뜀",                // K-out 2/ui/v_server.go:50
	"slower than usual · cause not known": "평소보다 느림 · 원인 확인 불가", // K 2/513
	"stale":                               "안 맞음",               // K 1/99
	"stale (another file)":                "다른 파일용",             // K 1/98
	"start failed: VRAM ran out · close other GPU programs or lower ctx · details on 4 Log":        "VRAM 부족으로 시작 실패 · 다른 GPU 프로그램을 끄거나 ctx 를 줄여 보세요 · 자세히 4 Log", // K 오류/79
	"steps · prefill · TTFT · decode · unload":                                                     "단계 · prefill · TTFT · decode · 내리기",                          // D5 도움말·알림/392
	"stuck? ctrl+c twice closes now · the cache is not saved":                                      "응답이 없나요? ctrl+c 두 번이면 바로 닫기 · 캐시 저장 안 됨",                     // K 2/396
	"the %s repack check failed at item %d of %d · ⏎ opens the setup plan":                         "%s 리팩 점검 실패(%d/%d 항목) · ⏎ 준비 계획 열기",                          // K-assembled 오류/wire/map.go:667
	"the adaptive mode is not in this release":                                                     "적응 모드는 이번 버전에 없음",                                            // K 2/577
	"the catalog check failed":                                                                     "카탈로그 점검 실패",                                                  // K-out 오류/wire/map.go:708
	"the check runs when this page opens":                                                          "이 페이지를 열 때 확인 실행",                                            // K 3/521
	"the launcher files are damaged · install again":                                               "배포 파일 손상 · 다시 설치",                                            // K-out 오류/wire/map.go:706
	"the launcher refuses this start · change the settings in 1 Models":                            "런처가 이 시작을 거부 · 1 Models 에서 설정 바꾸기",                           // K 3/522
	"the model file is missing or cannot be used":                                                  "모델 파일이 없거나 쓸 수 없음",                                           // K-out 오류/wire/map.go:700
	"the number turns yellow while its 5 s average stays under 80 % of usual":                      "5초 평균이 평소의 80 % 미만인 동안 숫자가 노란색",                              // K 도움말·알림/403
	"the repack check failed (redo the repack)":                                                    "리팩 점검 실패(리팩 다시 하기)",                                          // K-out 오류/wire/map.go:710
	"the repack files do not come from this file · ⏎ opens the setup plan":                         "리팩 파일이 이 모델 파일에서 만든 것이 아님 · ⏎ 준비 계획 열기",                      // K-out 오류/wire/map.go:661
	"the server ended while serving":                                                               "서빙 중 서버가 꺼짐",                                                 // K 도움말·알림/525
	"the slot save failed: the next start is cold":                                                 "캐시 저장 실패: 다음 시작은 캐시 없이",                                      // K 2/420
	"this file differs from the catalog file · catalog settings, prefetch off, no reference speed": "카탈로그 파일과 다른 파일 · 카탈로그 설정, 프리페치 끔, 기준 속도 없음",                  // K-out 도움말·알림/wire/map.go:596
	"tile states": "칸 상태", // K 도움말·알림/407
	"tokens":      "토큰",   // K-out 2/ui/v_server.go:902
	"turns into the serving view by itself · ctrl+x cancels, nothing is lost": "자동으로 서빙 화면으로 바뀜 · ctrl+x 취소, 잃는 것 없음", // K 2/409
	"unsupported":      "unsupported",  // E-out 1/ui/v_models.go:238
	"verdict":          "판정",           // K-out 3/ui/v_memory.go:388
	"virtual":          "virtual",      // E 1/418
	"wait /health":     "/health 기다림",  // K-out 2/wire/tracks.go:21
	"wait exit (60 s)": "종료 기다림 (60초)", // K-out 2/wire/tracks.go:174
	"waiting":          "대기",           // K-out 2/ui/v_server.go:793
	"warm start":       "warm start",   // E 2/135
	"warm start = the server reloads its last cache at start · next start": "warm start = 시작 때 지난 캐시를 다시 불러옴 · 다음 시작부터", // K 5/140
	"warm-start save is not ported: the next start is cold":                "warm start 저장 미지원: 다음 시작은 캐시 없이",           // K 2/419
	"watching":                   "감시 중",           // K-out 1/ui/v_models.go:941
	"what happened, newest last": "이벤트 기록 (최신=아래)", // decision 6
	"writes a map of the experts inside the model file (no copy), then checks it": "모델 파일 안 전문가 가중치 위치표를 쓴 뒤 확인(복사 없음)",    // K 2/561
	"⏎ on 1 Models starts the chosen model; its steps show here":                  "1 Models 에서 ⏎ 하면 고른 모델 시작; 단계는 여기에 표시", // K 2/431
	"⏎ starts the same model with the same settings · 1 Models to change":         "⏎ 같은 모델을 같은 설정으로 시작 · 바꾸려면 1 Models",   // K 2/433
}

// uxkrGone are the keys UX-KR replaced: decision 3 (the box title), decision 4 (the engine's
// unset reason) and decision 8 (the old lock wording).
var uxkrGone = []string{"no K and N were set for the engine", "the list is locked while the server runs · ctrl+f unloads", "this run · change in 1 Models"}

func TestUXKRTable(t *testing.T) {
	ko := table[LangKorean]
	for k, want := range uxkrWant {
		got, ok := ko[k]
		if !ok {
			t.Errorf("no Korean entry for %q (want %q)", k, want)
		} else if got != want {
			t.Errorf("%q: Korean %q, want %q", k, got, want)
		}
	}
	for _, k := range uxkrGone {
		if _, ok := ko[k]; ok {
			t.Errorf("the replaced key %q is still in the map", k)
		}
	}
	for k, v := range ko {
		if strings.Contains(v, "디코드") || strings.Contains(v, "프리필") {
			t.Errorf("decision 5: %q reads %q", k, v)
		}
	}
	// the backend's assembled sentences, looked up by their pattern (rows «조립 문장»)
	for in, want := range map[string]string{
		"no packed repack files yet · ⏎ opens the setup plan":                      "아직 packed 리팩 파일 없음 · ⏎ 준비 계획 열기",
		"the virtual repack check failed at item 3 of 12 · ⏎ opens the setup plan": "virtual 리팩 점검 실패(3/12 항목) · ⏎ 준비 계획 열기",
		"prefetch arming is not ported":                                            "prefetch arming 미지원",
		"saved settings (2026-10-13 16:00)":                                        "저장된 설정 (2026-10-13 16:00)",
	} {
		if got := tr(LangKorean, in); got != want {
			t.Errorf("assembled %q: %q, want %q", in, got, want)
		}
		if got := tr(LangEnglish, in); got != in {
			t.Errorf("assembled %q in English: %q", in, got)
		}
	}
}

// uxkrPrefetch is the user's case (screenshot 153217): the engine off with no K and N, the
// launcher's candidate off with its evidence reason.
var uxkrPrefetch = []string{"engine off", "unset", "candidate off", "not_opted_in_evidence_observe"}

func uxkrFake() *FakeBackend {
	f := NewFake()
	for i := range f.Statuses {
		f.Statuses[i].Prefetch = uxkrPrefetch
	}
	return f
}

func TestUXKRScreens(t *testing.T) {
	type page struct {
		name string
		h    func(lang string, w, h int) *harness
	}
	pages := []page{
		{"1 Models", func(lang string, w, h int) *harness { return reach(t, "H1", lang, w, h) }},
		{"1 Models serving", func(lang string, w, h int) *harness {
			hs := newH(t, uxkrFake(), lang, w, h)
			hs.keys(serving("1", "down")...)
			return hs
		}},
		{"2 Server", func(lang string, w, h int) *harness {
			hs := newH(t, uxkrFake(), lang, w, h)
			hs.keys(serving()...)
			hs.beat()
			return hs
		}},
		{"3 Memory", func(lang string, w, h int) *harness {
			hs := newH(t, uxkrFake(), lang, w, h)
			hs.keys(serving("3")...)
			return hs
		}},
		{"4 Log", func(lang string, w, h int) *harness {
			hs := newH(t, uxkrFake(), lang, w, h)
			hs.keys(serving("4")...)
			return hs
		}},
		{"5 Settings", func(lang string, w, h int) *harness {
			hs := newH(t, uxkrFake(), lang, w, h)
			hs.keys("5")
			return hs
		}},
		{"setup", func(lang string, w, h int) *harness { return reach(t, "H6", lang, w, h) }},
	}
	lock := map[string]string{
		LangEnglish: "server running · model choice locked · to unload: 2 Server, ctrl+f",
		LangKorean:  "서버 실행 중 · 모델 선택 잠김 · 모델을 내리려면 2 Server ctrl+f",
	}
	for _, sz := range [][2]int{{120, 40}, {80, 28}} {
		for _, pg := range pages {
			hs := pg.h(LangKorean, sz[0], sz[1])
			p := hs.plain()
			ls := strings.Split(p, "\n")
			// the nav row: the tab names in English (decision 1)
			for _, tab := range []string{"Models", "Server", "Memory", "Log", "Settings"} {
				if !strings.Contains(ls[0], tab) {
					t.Fatalf("%s %dx%d: the nav lacks %q: %q", pg.name, sz[0], sz[1], tab, ls[0])
				}
			}
			for _, old := range []string{"모델", "서버", "메모리", "로그", "설정"} {
				for _, n := range []string{"1 ", "2 ", "3 ", "4 ", "5 "} {
					if strings.Contains(p, n+old) {
						t.Fatalf("%s %dx%d: a tab named in Korean %q:\n%s", pg.name, sz[0], sz[1], n+old, p)
					}
				}
			}
			// decision 5
			for _, w := range []string{"디코드", "프리필"} {
				if strings.Contains(p, w) {
					t.Fatalf("%s %dx%d: %q on the Korean screen:\n%s", pg.name, sz[0], sz[1], w, p)
				}
			}
			// no line wider than the window, no more lines than it has
			if len(ls) > sz[1] {
				t.Fatalf("%s %dx%d: %d lines:\n%s", pg.name, sz[0], sz[1], len(ls), p)
			}
			for i, l := range ls {
				if w := lipgloss.Width(l); w > sz[0] {
					t.Fatalf("%s %dx%d: line %d is %d wide:\n%s", pg.name, sz[0], sz[1], i, w, p)
				}
			}
			switch pg.name {
			case "1 Models serving": // decision 8: the arrow key's notice in the one lock wording
				if !strings.Contains(hs.m.notice, lock[LangKorean]) || strings.Contains(p, "목록 잠김") {
					t.Fatalf("%s %dx%d: the arrow key's notice %q:\n%s", pg.name, sz[0], sz[1], hs.m.notice, p)
				}
			case "2 Server":
				uxkrServerBox(t, hs, LangKorean, sz)
			case "4 Log": // decision 6
				if !strings.Contains(p, "이벤트 기록 (최신=아래)") || strings.Contains(p, "일어난 일") {
					t.Fatalf("%s %dx%d: the 4 Log header:\n%s", pg.name, sz[0], sz[1], p)
				}
			}
		}
	}
	// the English mode: the counterparts of decisions 3 and 4, and the one lock wording
	for _, sz := range [][2]int{{120, 40}, {80, 28}} {
		hs := newH(t, uxkrFake(), LangEnglish, sz[0], sz[1])
		hs.keys(serving()...)
		hs.beat()
		uxkrServerBox(t, hs, LangEnglish, sz)
		hs.keys("1", "down")
		if hs.m.notice != lock[LangEnglish] {
			t.Fatalf("en %dx%d: the arrow key's notice %q", sz[0], sz[1], hs.m.notice)
		}
		hs.m.notice = ""
		if p := hs.plain(); strings.Count(p, lock[LangEnglish]) < 2 || strings.Contains(p, "the list is locked") {
			t.Fatalf("en %dx%d: the lock line and the serving footer:\n%s", sz[0], sz[1], p)
		}
	}
}

// uxkrServerBox checks the 2 Server «applied settings» box: its title (decision 3), the prefetch
// row of the engine's part only in one row (decision 4), and the labels dim, the values bright
// (decision 7), drawn with colours.
func uxkrServerBox(t *testing.T, hs *harness, lang string, sz [2]int) {
	t.Helper()
	title, pf, gone := "적용된 설정값, 변경은 Models", "미적용 · K·N 값 설정 미확인", []string{"런처 요청", "후보", "기본은 꺼짐", "입증 전", "이번 실행"}
	if lang == LangEnglish {
		title, pf, gone = "APPLIED SETTINGS, change in Models", "not applied · K·N setting unconfirmed", []string{"candidate", "off by default", "this run"}
	}
	p := hs.plain()
	if !strings.Contains(p, title) {
		t.Fatalf("%s %dx%d: no box title %q:\n%s", lang, sz[0], sz[1], title, p)
	}
	rows := 0
	for _, l := range strings.Split(p, "\n") {
		if strings.HasPrefix(strings.TrimLeft(l, " │"), hs.m.r.T("prefetch")+" ") {
			rows++
			if !strings.Contains(l, pf) {
				t.Fatalf("%s %dx%d: the prefetch row %q, want %q", lang, sz[0], sz[1], l, pf)
			}
		}
	}
	if rows != 1 {
		t.Fatalf("%s %dx%d: %d prefetch rows:\n%s", lang, sz[0], sz[1], rows, p)
	}
	for _, g := range gone {
		if strings.Contains(p, g) {
			t.Fatalf("%s %dx%d: %q on 2 Server:\n%s", lang, sz[0], sz[1], g, p)
		}
	}
	// the colours: a label drawn dim, its value bright, and the two differ
	prev := lipgloss.ColorProfile()
	lipgloss.SetColorProfile(1) // ANSI 256: the harness has no terminal
	defer lipgloss.SetColorProfile(prev)
	raw, r := hs.m.View(), hs.m.r
	dimOn, _, _ := strings.Cut(dim.Render("x"), "x")
	hiOn, _, _ := strings.Cut(hi.Render("x"), "x")
	if dimOn == "" || hiOn == "" || dimOn == hiOn {
		t.Fatalf("dim %q and bright %q", dimOn, hiOn)
	}
	vals := map[string]string{r.T("settings"): "12k context", r.T("prefetch"): "not applied"}
	if lang == LangKorean {
		vals = map[string]string{r.T("settings"): "컨텍스트 12k", r.T("prefetch"): "미적용"}
	}
	for label, v := range vals {
		found := false
		for _, l := range strings.Split(raw, "\n") {
			if !strings.Contains(l, dimOn+label) {
				continue
			}
			found = true
			if !strings.Contains(l, hiOn+v) {
				t.Fatalf("%s %dx%d: the row of %q does not draw %q bright:\n%q", lang, sz[0], sz[1], label, v, l)
			}
		}
		if !found {
			t.Fatalf("%s %dx%d: no dim label %q:\n%s", lang, sz[0], sz[1], label, ansi.Strip(raw))
		}
	}
}
