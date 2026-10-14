// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// The string table of the round-5 mock (designs/launcher_design/proto/go5/
// tr.go), English and Korean only (designs/launcher_design/README.md: English
// and 한국어 only). Every visible string goes through T, keyed by its English
// text; a key the table lacks falls back to English.

package ui

import "strings"

// Language codes the launcher knows.
const (
	LangEnglish = "en"
	LangKorean  = "ko"
)

// pickLang maps a list of display-language names (most preferred first,
// e.g. "ko-KR") to a launcher language: Korean if the first one is Korean,
// else English.
func pickLang(names []string) string {
	if len(names) > 0 && strings.HasPrefix(strings.ToLower(names[0]), "ko") {
		return LangKorean
	}
	return LangEnglish
}

// tr looks a key up in the language's table. English is the key itself: the
// English table is the set of keys the screens ask for, and the Korean table
// holds the same set (U-37 (i); S8-LA-36).
func tr(lang, s string) string {
	v, ok := s, lang == LangEnglish
	if !ok {
		if m, has := table[lang]; has {
			v, ok = m[s]
			if !ok {
				v, ok = assembledTr(m, s)
			}
			if !ok {
				v = s
			}
		}
	}
	if trSeen != nil {
		trSeen(lang, s, ok)
	}
	return v
}

// assembledTr looks a sentence of assembled (frame.go) up by its pattern: %s and %d match any
// value, and the values go into the entry's own pattern in the same order.
func assembledTr(m map[string]string, s string) (string, bool) {
	for _, k := range assembled {
		v, has := m[k]
		if !has {
			continue
		}
		vals, ok := matchKey(strings.ReplaceAll(k, "%d", "%s"), s)
		if !ok {
			continue
		}
		parts := strings.Split(strings.ReplaceAll(v, "%d", "%s"), "%s")
		var b strings.Builder
		for i, p := range parts {
			b.WriteString(p)
			if i < len(vals) && i < len(parts)-1 {
				b.WriteString(vals[i])
			}
		}
		return b.String(), true
	}
	return "", false
}

// trSeen, when set (tests), hears every lookup and whether the table held it.
var trSeen func(lang, key string, found bool)

// fill renders a template: text parts with ts, each %s with the next value in vs.
func fill(lang, tpl string, ts, vs func(...string) string, vals ...string) string {
	parts := strings.Split(tr(lang, tpl), "%s")
	var b strings.Builder
	for i, p := range parts {
		if p != "" {
			b.WriteString(ts(p))
		}
		if i < len(vals) && i < len(parts)-1 {
			b.WriteString(vs(vals[i]))
		}
	}
	return b.String()
}

var table = map[string]map[string]string{
	LangKorean: {
		// from the mock's table (proto/go5/tr.go "ko")
		"Models": "Models", "Server": "Server", "Memory": "Memory", "Log": "Log", "Settings": "Settings",
		"serving": "서빙 중",
		"CONTEXT": "컨텍스트", "FREE": "여유",
		"address":              "주소",
		"settings":             "설정",
		"now":                  "지금",
		"DECODE · last 60 min": "DECODE · 최근 60분", "PREFILL · last 60 min": "PREFILL · 최근 60분", // UX-R27 (the request box and its idle title went)
		"reused":                    "재사용",
		"PROCESSING · this request": "처리 · 이번 요청", "PROCESSING": "처리", "in": "입력", "out": "생성", "batch": "묶음", // UX-R26
		"copy address": "주소 복사", "pages": "pages", "help": "help", "exit": "exit",
		"unload":      "모델 내리기",
		"again: exit": "한 번 더: exit", "again to exit": "한 번 더", "cancel": "취소",
		"q again: unload the model and exit · other keys cancel": "q 한 번 더: 모델을 내리고 끝내기 · 다른 키는 취소",
		// U-27; wording F-LS 11.8 (OQ-4 (b), user)
		"reply in progress · press again to force quit": "출력 중 · 한 번 더 누르면 강제 종료",
		// the mock's example-bound entries turned into templates for live
		// values (same Korean words, the example value replaced by %s)
		"%s is serving": "%s 서빙 중", // "Qwen3.5-122B is serving"
		"up %s":         "가동 %s",   // "up 1h 02m"
		" of %s":        " / %s",   // " of 16k"
		"%s left":       "%s 남음",   // "6.6k left"
		"%s requests done · %s errors · none waiting": "요청 %s건 완료 · 오류 %s · 대기 없음",
		// the ctrl+x part removed by decision ① (DECISION_LEDGER.md:4177)
		"ctrl+f unloads the model and shows a summary": "ctrl+f 모델을 내리고 요약 보기",
		// given by the brief (decision ③, DECISION_LEDGER.md:4179)
		"slower than last time": "지난번보다 느림",
		// FIX-UI: the Korean wording of the confirmed cells of
		// reviews/_gol_review_8_user_20261010.md (item number after each)
		"not applied to the running server · applies at the next start": "지금 서버에는 적용 안 됨 · 다음 시작 때 적용",      // 1
		"▲ %s lines up · %s new · End=newest":                           "▲ 최신에서 %s줄 위 · 새 줄 %s · End=최신으로", // 6
		"file check":                    "파일 확인",       // 10
		"same as the catalog file":      "카탈로그 파일과 같음", // 10
		"no reference hash":             "기준 해시 없음",    // 10
		"differs from the catalog file": "카탈로그 파일과 다름", // 10
		"not checked yet":               "아직 확인 전",     // 10
		"tight":                         "빠듯",          // 11, 12
		"too big":                       "안 들어감",       // 11, 12
		"in use":                        "사용 중",        // 13
		"this model":                    "이 모델",        // 13
		"left after start":              "시작 뒤 남음",     // 13
		"this server":                   "이 서버",        // 13, 14
		"other programs":                "다른 프로그램",     // 14
		// 16, 17
		"API name":                     "API 이름",               // 27
		"average %s · lowest %s tok/s": "평균 %s · 최저 %s tok/s",  // 21
		"cannot start":                 "시작 불가",                // 22
		"RAM budget %s > free now %s":  "RAM 예산 %s > 지금 여유 %s", // 22
		"details on 3 Memory":          "자세히 3 Memory",         // 22
		"tight to start":               "빠듯함",                  // 22
		"RAM left after the start %s":  "시작 뒤 남는 RAM %s",       // 22
		"✓ %s GB to spare · plenty of room for other work alongside the model :>":                            "✓ %s GB만큼 여유롭네요 · 모델과 함께 다양한 작업을 하기에 충분하겠군요 :>",   // 22
		"✓ can start · %s RAM for other work":                                                                "✓ 시작 가능 · 다른 작업에 쓸 RAM %s",                        // 22
		"budget↑ = more cache hits · but less room for other programs · no effect once the whole model fits": "예산↑=캐시 적중↑ · 대신 다른 프로그램 몫↓ · 모델 전체가 이미 들어가면 효과 0", // 22
		"type a value": "직접 입력", // 30
		"apply":        "적용",    // 30
		"other":        "기타",    // 14, the short form of "other programs"
		"server":       "서버",    // 14, the shortest colour key
		// FIX-UI2 (review items 31, 32)
		"needs repack":              "리팩 필요",
		"the repack output is gone": "리팩 결과물 없음",
		// FIX2-UI: the wording of reviews/_gol_conflicts_8_user_20261010.md (conflict after each)
		"default":     "기본값",    // 3
		"my settings": "내 설정",   // 3
		"not saved":   "저장 안 됨", // 3
		"closing the window stops the server at once · nothing saved (same as ctrl+c)": "창 닫기=서버도 바로 꺼짐 · 저장 없음(ctrl+c 와 같음)", // 5
		"another launcher is already running":                                          "다른 런처가 이미 실행 중",                      // W7 7 (F-LS 11.7; GL-08)
		"server stopped · code %s · %s":                                                "서버 끔 · 코드 %s · %s",                   // 6
		"normal stop":                                                                  "정상",                                  // 6
		"stopped before ready":                                                         "준비 전 중지",                             // 6
		"forced stop":                                                                  "강제 종료",                               // 6
		"server exit %s":                                                               "서버 종료 코드 %s",                         // 6
		"a budget of %s or less starts it":                                             "예산을 %s 이하로 내리면 시작 가능",                // 7
		"start failed: VRAM ran out · close other GPU programs or lower ctx · details on 4 Log": "VRAM 부족으로 시작 실패 · 다른 GPU 프로그램을 끄거나 ctx 를 줄여 보세요 · 자세히 4 Log", // 8
		"SETUP NEEDED":  "준비 필요", // 9
		"needs a setup": "준비 필요", // 9
		"SETUP PLAN":    "준비 계획", // 9
		"writes to":     "where", // 9; UX-R9 (finding 14)
		"about %s written · drive free %s → %s left":              "약 %s 씀 · 드라이브 여유 %s → 남음 %s",               // 9
		"first repack: not known yet":                             "첫 시도라 확인 불가",                               // 9; UX-R9
		"last time %s":                                            "지난번 %s",                                    // 9
		"then: check → speed check → the server starts by itself": "이후 단계: 자동 확인 검증 → 디스크 읽기 속도 확인 → 서버 자동 시작", // 9; UX-R9
		"free %s < needed %s":                                     "여유 %s < 필요 %s",                             // 9
		"the setup plan could not be read":                        "준비 계획을 읽지 못함",                              // 9
		"⏎ runs the setup, then the server starts · esc goes back, nothing is written": "⏎ 준비 뒤 서버 시작 · esc 돌아가기, 아무것도 안 씀", // 9
		"set up and start": "준비 후 시작", // 9
		"free room on the drive, then ⏎ on 1 Models · esc goes back":        "드라이브 공간을 비운 뒤 1 Models 에서 ⏎ · esc 돌아가기", // 9
		"custom settings · performance unverified":                          "커스텀 설정 · 이 설정의 성능은 미확인",                 // 10
		"not in catalog · %s compatible · runs":                             "카탈로그 밖 · %s 호환 · 실행 가능",                 // UX-R10 (finding 18)
		"not measured with these settings":                                  "측정 전",                                   // 10
		"outside the catalog · %s template settings · performance untested": "카탈로그 밖 · %s 템플릿 설정 · 이 설정의 성능은 미확인",     // 14; UX-R10
		"outside the catalog · %s template settings":                        "카탈로그 밖 · %s 템플릿 설정",                     // UX-R12 addendum (finding 18)
		"⏎ shows its setup plan (the template repack)":                      "⏎ 로 준비 계획 보기 (템플릿 리팩)",                  // UX-R12 addendum (finding 18)
		"check failed":         "점검 실패",      // 16
		"stale (another file)": "다른 파일용",     // W7 12 (GL-10)
		"stale":                "안 맞음",       // W7 12 (GL-10): the short form
		"check run":            "시작 후 응답 시험", // 18-1; UX-R4 (finding 9)
		"once per model":       "최초 1회",      // 18-1
		"check run = a short chat test after a start · off, on, or once per model · results in 4 Log": "시작 후 응답 시험 = 시작 뒤 짧은 대화로 응답 확인 · 끔·켬·최초 1회 · 결과는 4 Log", // 18-1; UX-R4
		"check run failed": "시작 후 응답 시험 실패", // 18-1; UX-R4
		"✕ the launcher files are damaged · %s differ · install again": "✕ 배포 파일 손상 · %s개 불일치 · 다시 설치", // 18-2
		"development build · launcher file check skipped":              "개발 빌드 · 배포 점검 생략",             // 18-2
		"checking the launcher files … the start follows":              "배포 파일 확인 중 … 끝나면 시작",          // 18-2
		"SKIPPED": "건너뜀", // FIX-WIRE 4; UX-R4 (finding 8b)
		"⏎ starts · tight: close an app first for more room":            "⏎ 시작 · 빠듯함: 앱을 먼저 닫으면 여유가 늘어남",        // 7
		"address = where chat apps connect · next start · this PC only": "주소 = 채팅 앱이 연결하는 곳 · 다음 시작부터 · 이 PC 만", // 17
		"off":              "끔",         // 18-1
		"on":               "켬",         // 18-1
		"details on 4 Log": "자세히 4 Log", // 8, 18-1
		"disk":             "disk",      // 9; UX-R9 (finding 14)
		"time":             "time",      // 9; UX-R9 (finding 14)
		"failed":           "실패",        // 6
		"back":             "돌아가기",      // 9
		"left":             "남음",        // 13, the short form of "left after start"
		// FIX2-WIRE: the adapter's setting labels and start steps (FIX2-UI finding 2)
		"RAM budget": "RAM 예산",
		"context":    "컨텍스트",
		"%s context": "컨텍스트 %s",
		"queue (QD)": "큐 (QD)",
		"repack":     "리팩",
		"threads":    "스레드",
		"port":       "포트",
		"prefetch":   "프리페치",
		"prefetch K": "프리페치 K",
		// GL-20: the virtual notices (F-PF T18; OQ-5 (b), the user's wording of 2026-10-12)
		"default = QD 8 + catalog prefetch only · anything else: behaviour unmeasured":                                  "기본 값=QD 8+카탈로그 프리페치뿐 · 그 밖=동작 미측정",
		"prefetch admits nothing: N_cap is 0 on every layer at this QD (the run starts; raise QD to let prefetch work)": "프리페치가 아무것도 받지 않음: 이 QD 에서 모든 레이어의 N_cap=0 (시작은 함 · 프리페치를 쓰려면 QD 를 올림)",
		// GL-21: the server screen's prefetch row (F-PF §8.5)
		"engine on":              "엔진 켜짐",
		"applied":                "적용됨", // UX-KR2 (2): the engine on, paired with «미적용»
		"engine off":             "엔진 꺼짐",
		"engine values not read": "엔진 값 읽지 못함",
		"candidate on":           "런처 요청 켜짐",
		"candidate off":          "런처 요청 꺼짐",
		"prefetch N":             "프리페치 N",
		"warmup":                 "예열",
		"warm start":             "warm start",
		// GL-16: the launcher settings of 5 Settings and the custom-value texts (F-LS 11.4)
		"prompt cache":      "프롬프트 캐시",
		" MiB":              " MiB",
		"off · the default": "끔 · 기본",
		"DESCRIPTION":       "설명", // UX-R25: the 5 Settings description box
		"prompt cache = RAM for reused prompts · off (0) by default · next start":                             "프롬프트 캐시 = 다시 쓰는 프롬프트용 RAM · 기본은 끔(0) · 다음 시작부터",
		"warm start = the server reloads its last cache at start · next start":                                "warm start = 시작 때 지난 캐시를 다시 불러옴 · 다음 시작부터",
		"cache autosave = the cache is saved while the server runs · next start":                              "캐시 자동 저장 = 서버가 도는 동안 캐시를 저장 · 다음 시작부터",
		"outside its range · allowed %s · the value is kept, nothing applied":                                 "범위 밖 · 허용 %s · 값은 그대로, 적용 안 됨",
		"range = this model's catalog limits":                                                                 "범위 = 이 모델 카탈로그의 허용값",           // UX-R6
		"range = structure template limits":                                                                   "범위 = 구조 템플릿의 허용값",              // UX-R6
		"not saved = this run only, not in this model's settings":                                             "저장 안 됨 = 이번 실행에만, 이 모델 설정엔 없음", // UX-R6
		"the saved settings of this model hold a value that failed its check · not used · the defaults apply": "이 모델의 저장된 설정에 검사를 통과하지 못한 값이 있음 · 쓰지 않음 · 기본값 적용",
		"cache autosave": "캐시 자동 저장",
		"KV type":        "KV 형식",
		"format check":   "형식 점검",
		"parallel":       "동시 요청",
		"bundle check":   "배포 확인",
		"warm up":        "예열",
		// GL-17: the keys every screen asks for (U-37 (i); S8-LA-36)
		"1 Models picks a model and starts it":            "1 Models 에서 모델을 골라 시작",
		"%s cache":                                        "캐시 %s", // UX-R12 addendum: any budget, «n GB cache»
		"2 Server shows each step":                        "2 Server 에서 단계별 확인",
		"2 Server · the big tile through one request":     "2 Server · 요청 하나 동안의 큰 칸",
		"3 Memory shows what changed around the slowdown": "3 Memory 에서 느려진 때의 변화 확인",
		"ANYWHERE":                     "어디서나",
		"APPS · by memory":             "앱 · 메모리순",
		"AVAILABLE NOW":                "지금 여유",
		"CACHE":                        "캐시",
		"COMMIT":                       "커밋",
		"COMMIT LEFT":                  "커밋 여유",
		"DECODE":                       "DECODE",
		"DECODE · tok/s · last 60 min": "DECODE · tok/s · 최근 60분",
		"DECODE · tok/s · whole run":   "DECODE · tok/s · 전체 실행",
		"DISK":                         "DISK",
		"FILES":                        "파일",
		"FIX":                          "해결",
		"FOLDERS":                      "폴더",
		"HOW TO ADD A MODEL":           "모델 추가 방법",
		"LAST DECODE":                  "최근 DECODE",
		"LAST LOG LINES":               "최근 로그 줄",
		"LAUNCHER":                     "런처",
		"LEFT":                         "남음",
		"LEFT AFTER":                   "시작 뒤 남음",
		"LIKELY CAUSE":                 "예상 원인",
		"MEMORY · now":                 "메모리 · 지금",
		"MODEL":                        "모델",
		"MODEL FOLDERS":                "모델 폴더",
		"NEEDS":                        "필요",
		"PAGEFILE":                     "페이징 파일",
		"PAGES":                        "페이지",
		"PREFILL":                      "PREFILL",
		"PREFILL · tok/s · whole run":  "PREFILL · tok/s · 전체 실행",
		"QD 8":                         "QD 8",
		"QUIT STEPS":                   "끝내기 단계",
		"QUITTING":                     "끝내는 중",
		"RAM":                          "RAM",
		"RAM+pagefile":                 "RAM+페이징 파일",
		"RAM free (now)":               "RAM 여유(지금)", // UX-R16b: the user's name (answer ⑥)
		"RAM free":                     "RAM 여유",
		"REDO":                         "다시 하기",
		"RESUME":                       "이어서",
		"SETTINGS":                     "설정",
		"SETUP":                        "준비",
		"START":                        "시작",
		"START AGAIN":                  "다시 시작",
		"START STEPS":                  "시작 단계",
		"STARTING":                     "시작 중",
		"STEP":                         "단계",
		"THIS MODEL":                   "이 모델",
		"TTFT":                         "TTFT",
		"TTFT estimate under it":       "아래 TTFT 추정",
		"Task Manager":                 "작업 관리자",
		"UNSUPPORTED GGUF":             "지원 안 되는 GGUF",
		"VRAM":                         "VRAM",
		"WHERE IS…":                    "어디에…",
		"WHERE THE LAUNCHER WRITES":    "런처 저장 위치",
		"WHY":                          "이유",
		"a .gguf of the catalog saved into a model folder appears after r · r looks now":         "카탈로그 .gguf 를 모델 폴더에 넣고 r 누르면 나타남 · r 지금 확인",
		"a longer context decodes slower by itself, so it is compared only with similar context": "긴 컨텍스트는 원래 더 느려서 비슷한 컨텍스트끼리만 비교",
		"its structure": "이 구조", // UX-R10 (finding 18)
		"a reply (64+ tokens) ends under 80 %: ▲ slow":                                     "답변(64+ 토큰)이 80 % 미만으로 끝남: ▲ 느림",
		"a reply of 64 tokens or more ends under 80 % of usual: ▲ slow and the alert line": "64 토큰 이상 답변이 평소의 80 % 미만으로 끝나면 ▲ 느림과 알림 줄",
		"a request arrives":   "요청이 들어옴",
		"a setup was cut off": "준비가 중간에 끊김",
		"add":                 "추가",
		"add a folder":        "폴더 추가",
		"after":               "뒤",
		"after any fix it starts again · setup is not redone": "고친 뒤 다시 시작 · 준비는 다시 안 함",
		"all":  "전체",
		"app":  "앱",
		"apps": "앱",
		"apps are not closed here · o opens Task Manager": "여기서 앱을 닫지 않음 · o 작업 관리자 열기",
		"apps may be paged out to disk":                   "앱이 디스크로 밀려날 수 있음",
		"auto · English":                                  "자동 · 영어",
		"auto · Korean":                                   "자동 · 한국어",
		"average of the last 5 s":                         "최근 5초 평균",
		"avg":                                             "평균",
		"back · close · undo":                             "돌아가기 · 닫기 · 되돌리기",
		"can it start · what changed when slow":           "시작 가능한지 · 느릴 때 달라진 점",
		"cancel setup":                                    "준비 취소",
		"cancel start":                                    "시작 취소",
		"cause rules":                                     "원인 규칙",
		"change":                                          "바꾸기",
		"check memory":                                    "메모리 확인",
		"choose a row":                                    "행 고르기",
		"clears":                                          "지워짐",
		"close":                                           "닫기",
		"close launcher":                                  "런처 닫기",
		"closing: unload the model, then the launcher": "닫는 중: 모델을 내린 뒤 런처 닫기",
		"copy report": "보고서 복사",
		"ctrl+s saves for the next start · ctrl+f unloads, then ⏎ starts": "ctrl+s 다음 시작용 저장 · ctrl+f 내린 뒤 ⏎ 시작",
		"cut off":                     "끊김",
		"decode":                      "decode",
		"delete half-made only":       "만들다 만 것만 지우기",
		"delete only":                 "지우기만 하기",
		"details":                     "자세히",
		"did not start":               "시작 안 됨",
		"done":                        "완료",
		"enough left after the start": "시작 뒤 충분히 남음",
		"errors":                      "오류",
		"esc goes back to where you were · tab returns to the key help":        "esc 이전 화면으로 · tab 키 도움말로",
		"esc goes back to where you were · tab shows the big tile of 2 Server": "esc 이전 화면으로 · tab 2 Server 의 큰 칸 보기",
		"every page and key":                            "모든 페이지와 키",
		"everything that happened":                      "일어난 모든 일",
		"expert cache only · the whole model uses more": "전문가 캐시만 · 모델 전체는 더 씀",
		"expert cache split":                            "전문가 캐시 나눔", // UX-R16
		"file":                                          "파일",
		"file check · hash":                             "파일 확인 · 해시",
		"first-run setup":                               "첫 실행 준비",
		"fit":                                           "fit",
		"fits":                                          "fits",
		"fits this PC":                                  "이 PC 에 들어감",
		"folder":                                        "폴더",
		"folder scan":                                   "폴더 검색",
		"folder watch":                                  "폴더 감시",
		"folders":                                       "폴더",
		"found":                                         "찾음",
		"free":                                          "여유",
		"cache Windows gives up on demand":              "Windows가 요청 즉시 내주는 캐시", // UX-R22 (item 33)
		"given up on demand":                            "요청 즉시 내줌",
		"usable now":                                    "지금 쓸 수 있음",
		"RAM empty now":                                 "지금 빈 RAM",
		"empty RAM":                                     "빈 RAM",
		"cache":                                         "캐시",
		"reply":                                         "답변",   // UX-R16b: the NEEDS note «model … + … reply»
		"not measured":                                  "측정 전", // UX-R16b: «not measured yet» in a narrow tile
		"freed at stop":                                 "멈추면 비워짐",
		"from layer 1":                                  "레이어 1부터",
		"go to that page":                               "그 페이지로 이동",
		"grey frame":                                    "회색 틀",
		"half-made":                                     "만들다 만 것",
		"is serving, slower than usual":                 "서빙 중, 평소보다 느림",
		"is starting":                                   "시작 중",
		"keys":                                          "키",
		"language":                                      "언어",
		"last":                                          "최근",
		"last 10 min":                                   "최근 10분",
		"last reply":                                    "마지막 답변",
		"last run":                                      "지난 실행",
		"last run average":                              "지난 실행 평균",
		"launcher settings":                             "런처 설정",
		"launcher settings and model folders":           "런처 설정과 모델 폴더",
		"launcher settings · model folders":             "런처 설정 · 모델 폴더",
		"layers":                                        "레이어",
		"layers done":                                   "레이어 완료",
		"left to right":                                 "왼쪽에서 오른쪽으로",
		"less available than it needs":                  "필요한 것보다 여유가 적음",
		"live":                                          "실시간",
		"live · 5 s average":                            "실시간 · 5초 평균",
		"load model":                                    "모델 불러오기",
		"log file":                                      "로그 파일",
		"logs":                                          "로그",
		"logs kept":                                     "로그 보관 기간",
		"longer context is slower by itself: compared alike": "긴 컨텍스트는 원래 느림: 비슷한 것끼리 비교",
		"look again": "다시 보기",
		"low":        "낮음",
		"max":        "최대",
		"median":     "중앙값",
		"median decode of this model's last 20 replies at a similar context (±4k)": "비슷한 컨텍스트(±4k)에서 이 모델의 최근 답변 20개 decode 중앙값",
		"median decode, last 20 replies, similar context":                          "decode 중앙값, 최근 답변 20개, 비슷한 컨텍스트",
		"memory":                   "메모리",
		"memory lists":             "메모리 목록",
		"memory while serving":     "서빙 중 메모리",
		"model":                    "모델",
		"model files":              "모델 파일",
		"model folders":            "모델 폴더",
		"model settings":           "모델 설정",
		"model settings: 1 Models": "모델 설정: 1 Models",
		"model so far":             "모델 적재분",
		"model somewhere else? add that folder in": "모델이 다른 폴더에 있다면 그 폴더를 추가",
		"modified":              "수정됨",
		"more":                  "개 더",
		"name":                  "이름",
		"need not measured yet": "필요량 측정 전",
		"needed":                "필요",
		"new number every 1 s":  "1초마다 새 숫자",
		"newest":                "최신",
		"no":                    "아니요",
		"no request":            "요청 없음",
		"no server":             "서버 없음",
		"no server is running":  "실행 중인 서버 없음",
		"not now":               "나중에",
		"not ported":            "미지원",
		"not possible":          "불가능",
		"not proven":            "미확인",
		"not set up":            "준비 안 됨",
		"not used":              "안 씀",
		"nothing lost · setup and settings are kept": "잃는 것 없음 · 준비와 설정은 유지",
		"notice": "알림",
		"number turns yellow under 80 % of usual": "평소의 80 % 미만이면 숫자가 노란색",
		"of":                                   "/",
		"of setup":                             "준비 중",
		"of ~%s":                               "/ ~%s",
		"on open":                              "on open",
		"open":                                 "열기",
		"open file":                            "파일 열기",
		"pick a model, change settings, start": "모델 고르고, 설정 바꾸고, 시작",
		"pick a model, then start":             "모델 고른 뒤 시작",
		"prefetch catalog":                     "프리페치 카탈로그",
		"prefill":                              "prefill",
		"prefill and TTFT are shown and logged, but do not set ▲ slow": "prefill 과 TTFT 는 표시·기록만 하고 ▲ 느림에는 안 씀",
		"prefill and TTFT are shown, do not set ▲ slow":                "prefill 과 TTFT 는 표시만, ▲ 느림에는 안 씀",
		"preparing":             "준비 중",
		"press r to look again": "r 누르면 다시 봄",
		"previous · next page":  "이전 · 다음 페이지",
		"prompt cache save":     "프롬프트 캐시 저장",
		"put a model file (.gguf) of the catalog into": "카탈로그의 모델 파일(.gguf)을 다음 폴더에 넣기",
		"q twice": "q 두 번",
		"q twice or closing the window cuts setup off; it starts over next time": "q 두 번이나 창 닫기는 준비를 끊음; 다음에 처음부터",
		"quant":                    "양자화",
		"queue":                    "큐",
		"quitting":                 "끝내는 중",
		"r looks again now":        "r 지금 다시 봄",
		"read":                     "읽음",
		"reading model":            "모델 읽는 중",
		"ready":                    "준비됨",
		"reply ends slow: ▲ slow":  "답변이 느리게 끝남: ▲ 느림",
		"request and error counts": "요청·오류 수",
		"requests":                 "요청",
		"rewrites the model file once so experts load fast · first run only": "전문가 가중치가 빨리 올라오게 모델 파일을 한 번 다시 씀 · 첫 실행만",
		"running":                               "실행 중",
		"same for every model":                  "모든 모델 공통",
		"samples show when the model answers":   "모델이 답변시 표본 표시",
		"PREFILL · tok/s per request":           "PREFILL · 요청별 tok/s",
		"save":                                  "저장",
		"save prompt cache":                     "프롬프트 캐시 저장",
		"scroll":                                "스크롤",
		"section":                               "구역",
		"setup":                                 "준비",
		"setup keeps going if you switch pages": "페이지를 바꿔도 준비는 계속됨",
		"share of the prompt read":              "읽은 프롬프트 비율",
		"shows the last reply":                  "마지막 답변 표시",
		"size":                                  "크기",
		"slow":                                  "느림",
		"standby":                               "대기",
		"start":                                 "시작",
		"start again":                           "다시 시작",
		"start server":                          "서버 시작",
		"starting":                              "시작 중",
		"starts now":                            "지금 시작",
		"starts with room":                      "여유 있게 시작",
		"starts, but tight":                     "시작은 되지만 빠듯함",
		"state":                                 "상태",
		"steps · prefill · TTFT · decode · unload": "단계 · prefill · TTFT · decode · 내리기",
		"stop the server":       "서버 멈추기",
		"stopped":               "멈춤",
		"stopped · run summary": "멈춤 · 실행 요약",
		"stuck? ctrl+c twice closes now · the cache is not saved":                 "응답이 없나요? ctrl+c 두 번이면 바로 닫기 · 캐시 저장 안 됨",
		"the catalog names the model files":                                       "카탈로그가 모델 파일 이름을 정함",
		"the catalog's model folder":                                              "카탈로그의 모델 폴더",
		"the first token is out":                                                  "첫 토큰 나옴",
		"the next reply at 90 % of usual or more":                                 "다음 답변이 평소의 90 % 이상",
		"the next reply that ends at 90 % of usual or more":                       "평소의 90 % 이상으로 끝난 다음 답변",
		"the number turns yellow":                                                 "숫자가 노란색으로 바뀜",
		"the number turns yellow while its 5 s average stays under 80 % of usual": "5초 평균이 평소의 80 % 미만인 동안 숫자가 노란색",
		"the slow verdict comes from the backend":                                 "느림 판정은 백엔드에서 옴",
		"then verify → speed check → the server starts by itself":                 "이후 단계: 자동 확인 검증 → 디스크 읽기 속도 확인 → 서버 자동 시작",
		"this help · again to close":                                              "이 도움말 · 한 번 더 누르면 닫기",
		"tile states":                                                             "칸 상태",
		"try again":                                                               "다시 시도",
		"turns into the serving view by itself · ctrl+x cancels, nothing is lost": "자동으로 서빙 화면으로 바뀜 · ctrl+x 취소, 잃는 것 없음",
		"twice: close now":                                                        "두 번: 바로 닫기",
		"twice: close now, no cache save":                                         "두 번: 바로 닫기, 캐시 저장 안 함",
		"twice: unload, then close":                                               "두 번: 내린 뒤 닫기",
		"under 80 % of usual":                                                     "평소의 80 % 미만",
		"undo":                                                                    "되돌리기",
		"up":                                                                      "가동",
		"usual":                                                                   "평소",
		"virtual":                                                                 "virtual",
		"warm-start save is not ported: the next start is cold":                   "warm start 저장 미지원: 다음 시작은 캐시 없이",
		"the slot save failed: the next start is cold":                            "캐시 저장 실패: 다음 시작은 캐시 없이",
		"no slot save at this stop: the next start is cold":                       "이번 정지는 캐시 저장 없음: 다음 시작은 캐시 없이",
		"warnings":                     "경고",
		"what happened, newest last":   "이벤트 기록 (최신=아래)",
		"when the status line says it": "상태 줄에 나올 때",
		"where models are found":       "모델을 찾는 곳",
		"y copies a report: model, settings, memory, last errors":                         "y 보고서 복사: 모델, 설정, 메모리, 최근 오류",
		"you can switch pages · the start keeps going":                                    "페이지를 바꿔도 됨 · 시작은 계속됨",
		"↑ holds the view · End jumps to the newest · Tab switches all, warnings, errors": "↑ 화면 고정 · End 최신으로 · Tab 전체, 경고, 오류 전환",
		"↑ holds the view · End newest · Tab switches":                                    "↑ 화면 고정 · End 최신 · Tab 전환",
		"↓ chooses a row · Tab moves between the sections · esc lets go of the row":       "↓ 행 고르기 · Tab 구역 이동 · esc 행 놓기",
		"⏎ on 1 Models starts the chosen model; its steps show here":                      "1 Models 에서 ⏎ 하면 고른 모델 시작; 단계는 여기에 표시",
		"⏎ starts": "⏎ 시작",
		"⏎ starts the same model with the same settings · 1 Models to change": "⏎ 같은 모델을 같은 설정으로 시작 · 바꾸려면 1 Models",
		"models":                  "모델",
		"setup of %s was cut off": "%s 준비가 끊김",
		"lists":                   "목록",
		"locked":                  "잠김", // UX-R10
		// GL-05: the need's labels (S8-LA-13; U-23 (i): «unverified prediction» = 미검증 예측; UX-R10 (finding 16 (e)): the words softened)
		"measured":             "측정값",
		"predicted before run": "실측 전 예측값",
		// GL-09: Task Manager (U-35 (i)), the «left» figures (F-BU G), the busy port (GL-08)
		"Task Manager opened · find %s, PID %s":           "작업 관리자 열림 · %s, PID %s 를 찾으세요",
		"Task Manager opened · nothing is selected in it": "작업 관리자 열림 · 선택된 항목 없음",
		"if written to disk first":                        "디스크에 먼저 쓰면",
		"port %s is in use by another program":            "포트 %s 를 다른 프로그램이 쓰는 중",
		// GL-17: the words of the states the design screens do not reach (source literals)
		"CANCEL STEPS":       "취소 단계",
		"CANCELLING":         "취소 중",
		"CLOSE STEPS":        "닫기 단계",
		"CLOSING":            "닫는 중",
		"UNLOAD STEPS":       "내리기 단계",
		"UNLOADING":          "내리는 중",
		"HAS NONE":           "없음",
		"Settings adds more": "Settings 에서 더 추가",
		"a server is running · ctrl+f unloads it, then ⏎ starts": "서버 실행 중 · ctrl+f 로 내린 뒤 ⏎ 시작",
		"app list":             "앱 목록",
		"cancelling the start": "시작 취소 중",
		"changes apply at the next start · ctrl+f unloads first": "바꾼 값은 다음 시작부터 · 먼저 ctrl+f 로 내리기",
		"checking …":                  "확인 중 …",
		"closing now · no cache save": "바로 닫는 중 · 캐시 저장 안 함",
		"ctrl+c again closes now · the cache is not saved · other keys cancel": "ctrl+c 한 번 더: 바로 닫기 · 캐시 저장 안 됨 · 다른 키: 취소",
		"ctrl+c twice": "ctrl+c 두 번",
		"ctrl+r shows the setup plan · its ⏎ deletes the half-made files and redoes setup · ctrl+d deletes them only": "ctrl+r 준비 계획 · 그 ⏎ 로 만들다 만 파일을 지우고 준비 다시 · ctrl+d 지우기만",
		"ctrl+r setup plan · ctrl+d delete only": "ctrl+r 준비 계획 · ctrl+d 지우기만",
		"elapsed · no earlier run to compare":    "경과 · 비교할 지난 실행 없음",
		"exit %d":                                "종료 %d",
		"exit code %d":                           "종료 코드 %d",
		"expert cache":                           "전문가 캐시",     // UX-R16: the user's word
		"(budget %s)":                            "(예산 할당 %s)", // UX-R16: the user's words
		"file mapping":                           "파일 매핑",
		"first run of these settings":            "이 설정의 첫 실행",
		"first run sets up":                      "첫 실행 때 준비",
		"first time":                             "처음",
		"first time: seconds only":               "처음: 초만 표시",
		"folder does not exist":                  "폴더 없음",
		"full log":                               "전체 로그",
		"it shows up here by itself, no restart needed":                                                                                 "자동으로 여기 나타남, 다시 시작 필요 없음",
		"judged on its own, freeing RAM does not add commit":                                                                            "따로 판정, RAM 을 비워도 커밋은 늘지 않음",
		"language = English at start · a change lasts for this run":                                                                     "언어 = 시작은 영어 · 바꾸면 이번 실행에만",
		"language changes at once (this run)":                                                                                           "언어는 바로 바뀜 (이번 실행)",
		"language changes at once (this run) · ↑ past the first row goes back to the folders":                                           "언어는 바로 바뀜 (이번 실행) · 첫 행에서 ↑ 하면 폴더로",
		"language changes at once (this run) · the other launcher settings: not ported · ↑ past the first row goes back to the folders": "언어는 바로 바뀜 (이번 실행) · 다른 런처 설정: 미지원 · 첫 행에서 ↑ 하면 폴더로",
		"language changes at once · others: not ported":                                                                                 "언어는 바로 바뀜 · 나머지: 미지원",
		"left after the start":                                  "시작 뒤 남음",
		"likely, not proven":                                    "추정, 미확인",
		"logs kept = days before old logs go":                   "로그 보관 기간 = 오래된 로그를 지우기까지의 날 수",
		"model file missing":                                    "모델 파일 없음",
		"needs %s > free now %s":                                "필요 %s > 지금 여유 %s",
		"next start reuses it":                                  "다음 시작에 다시 씀",
		"no .gguf file yet":                                     "아직 .gguf 파일 없음",
		"no check yet · 1 Models picks a model":                 "아직 확인 없음 · 1 Models 에서 모델 고르기",
		"no decode samples in this run":                         "이번 실행에 decode 표본 없음",
		"no earlier run of these settings":                      "이 설정의 지난 실행 없음",
		"no log lines yet":                                      "아직 로그 줄 없음",
		"no prefill samples in this run":                        "이번 실행에 prefill 표본 없음",
		"no server to unload":                                   "내릴 서버 없음",
		"none waiting":                                          "대기 없음",
		"not MoE models":                                        "MoE 모델 아님",
		"not changeable here":                                   "여기서 못 바꿈",
		"not measured yet":                                      "측정 전",
		"nothing changed to save":                               "저장할 변경 없음",
		"nothing is lost on this path · the next start is warm": "이 경로에서 잃는 것 없음 · 다음 시작은 캐시를 이어 씀",
		"nothing is lost · setup and settings are kept":         "잃는 것 없음 · 준비와 설정은 유지",
		"on open = the page shown first":                        "on open = 처음 보이는 페이지",
		"or more left after the start":                          "이상 시작 뒤 남음",
		"prompt cache saved":                                    "프롬프트 캐시 저장됨",
		"r looks again":                                         "r 다시 보기",
		"range":                                                 "범위",  // UX-R6
		"saved":                                                 "저장값", // UX-R6
		"reading memory …":                                      "메모리 읽는 중 …",
		"reading settings …":                                    "설정 읽는 중 …",
		"reading the catalog …":                                 "카탈로그 읽는 중 …",
		"requests done":                                         "요청 완료",
		"room line":                                             "여유선",
		"safe line":                                             "안전선",
		"safe ≥ %s":                                             "안전 ≥ %s", // UX-R16: with «GB»
		"saved · these values are used from the next start":     "저장됨 · 다음 시작부터 이 값 사용",
		"saves the changed values":                              "바꾼 값 저장",
		"slower than usual · cause not known":                   "평소보다 느림 · 원인 확인 불가",
		"standby + free":                                        "대기 + 여유",
		"start cancelled · nothing is lost":                     "시작 취소됨 · 잃는 것 없음",
		"starts over":                                           "처음부터 다시",
		"stop failed":                                           "멈추기 실패",
		"stopped while serving":                                 "서빙 중 멈춤",
		"the catalog could not be read":                         "카탈로그를 읽지 못함",
		"the check runs when this page opens":                   "이 페이지를 열 때 확인 실행",
		"the launcher refuses this start · change the settings in 1 Models": "런처가 이 시작을 거부 · 1 Models 에서 설정 바꾸기",
		"the server did not start":                    "서버 가동 실패",
		"the server ended while serving":              "서빙 중 서버가 꺼짐",
		"the server was stopped during the check run": "시작 후 응답 시험 중 서버가 멈춤",
		"the start ended without /health ready":       "/health 준비 없이 시작이 끝남",
		"this launcher":                               "이 런처",
		"too big for this PC":                         "이 PC 에 너무 큼",
		"unloading the model · the launcher stays":    "모델 내리는 중 · 런처는 남음",
		"waiting for the first step …":                "첫 단계 기다리는 중 …",
		"what changed when it got slow":               "느려질 때 달라진 점",
		"~%.0f min":                                   "~%.0f분",
		"←→ changes the chosen value · ⏎ starts with these values · esc undoes":     "←→ 고른 값 바꾸기 · ⏎ 이 값으로 시작 · esc 되돌리기",
		"↓ past the last folder goes on to the launcher rows · Tab skips a section": "마지막 폴더에서 ↓ 하면 런처 행으로 · Tab 구역 건너뛰기",
		"⏎ applies after the range check · esc keeps the value as it was":           "⏎ 범위 확인 뒤 적용 · esc 원래 값 유지",
		"⏎ opens the folder, or shows the file in Explorer":                         "⏎ 폴더를 열거나 탐색기에서 파일 보기",
		"repack mode": "리팩 방식",
		"this":        "이 서버",
		"re-read every 5 s on 1 Models (names, sizes, times)": "1 Models 에서 5초마다 다시 읽음 (이름, 크기, 시간)",
		"cancel the start": "시작 취소",
		// GL-06 (F-BU E.3, F)
		"commit":                       "커밋",
		"expert pool":                  "전문가 메모리 예산",
		"of this start's commit %s":    "이번 시작의 커밋 %s 중",
		"max %s MiB at the last check": "마지막 점검 기준 최대 %s MiB",
		"commit near its limit: %s left after the start is below the safe line %s · limit %s · in use now %s · this start %s, of it the expert pool %s · nothing is changed": "커밋이 한도에 가까움: 시작 뒤 남는 %s 가 안전선 %s 아래 · 한도 %s · 지금 사용 %s · 이번 시작 %s, 그중 전문가 메모리 예산 %s · 아무것도 바꾸지 않음",

		// UX-R3 (3 Memory, user findings 4 to 7): a «left» below zero is a shortfall against its
		// named baseline; the left tile's terms; the start checklist title
		"commit near its limit: %s short after the start, against the limit · safe line %s · limit %s · in use now %s · this start %s, of it the expert pool %s · nothing is changed": "커밋이 한도에 가까움: 시작 뒤 %s 부족 · 한도 기준 · 안전선 %s · 한도 %s · 지금 사용 %s · 이번 시작 %s, 그중 전문가 메모리 예산 %s · 아무것도 바꾸지 않음",
		"%s start checklist": "%s 점검표",
		"%s short":           "%s 부족",
		"free %s − need %s":  "여유 %s − 필요 %s",
		"%s over the limit":  "한도 기준 %s 초과",
		"over the limit":     "한도 기준 초과",
		"%s short after the start · against the limit": "시작 뒤 %s 부족 · 한도 기준",
		"%s short after the start":                     "시작 뒤 %s 부족",
		"%s short if written to disk first":            "디스크에 먼저 쓰면 %s 부족",

		// UX-R9 (findings 14, 15 (b), 20 (d)): the setup plan's sentences, the failure title and
		// the prefetch read-back codes in words
		"server start failed": "서버 가동 실패",
		"what to do":          "what to do",
		"copies the model's experts into one streaming file (experts.bin), then checks it":                                                             "모델의 전문가 가중치를 스트리밍용 파일 하나(experts.bin)로 복사한 뒤 확인",
		"writes a map of the experts inside the model file (no copy), then checks it":                                                                  "모델 파일 안 전문가 가중치 위치표를 쓴 뒤 확인(복사 없음)",
		"outside the catalog: copies the experts the structure template selects, byte for byte, into one streaming file (experts.bin), then checks it": "카탈로그 밖: 구조 템플릿이 고른 전문가 가중치를 바이트 그대로 스트리밍용 파일 하나(experts.bin)로 복사한 뒤 확인",
		"experimental: no published measurement covers this model":                                                                                     "실험적: 이 모델의 공개 측정 없음",
		"an earlier setup was cut off: it starts again from the beginning":                                                                             "이전 준비가 중간에 끊김: 처음부터 다시 함",
		"the files there do not match this model file: they are replaced":                                                                              "그곳 파일이 이 모델 파일과 맞지 않음: 새로 바꿈",
		"the files there are replaced":                                      "그곳 파일을 새로 바꿈",
		"first removes %d earlier file(s)":                                  "먼저 이전 파일 %d개 지움",
		" freed, counted as free)":                                          " 확보, 여유로 셈)",
		"the K given to the engine is not a whole number":                   "엔진에 준 K 값이 정수가 아님",
		"the N given to the engine is not a whole number":                   "엔진에 준 N 값이 정수가 아님",
		"K or N is outside what the engine accepts":                         "K 또는 N 이 엔진이 받는 범위 밖",
		"this model family has no prefetch signal":                          "이 모델 계열엔 프리페치 신호가 없음",
		"the catalog's prefetch entries contradict each other":              "카탈로그의 프리페치 항목이 서로 어긋남",
		"the drive speed check before the start failed":                     "시작 전 드라이브 속도 측정이 실패함",
		"this model file is not the one the catalog measured":               "카탈로그가 측정한 모델 파일과 다름",
		"the adaptive mode is not in this release":                          "적응 모드는 이번 버전에 없음",
		"off by default: no measurement shows a gain on this model":         "기본은 꺼짐: 이 모델에서 이득이 측정되지 않음",
		"off by default: a gain on this model is only observed, not proven": "기본은 꺼짐: 이 모델에서 이득은 관찰만 됐고 입증 전",
		"off by default: a gain on this model is shown in replay only":      "기본은 꺼짐: 이 모델에서 이득은 재현 시험에서만 확인됨",
		"off by default: not switched on for this model":                    "기본은 꺼짐: 이 모델에는 켜지 않음",
		"held off until a pending measurement is settled":                   "남은 측정이 정리될 때까지 보류",
		"the starting depth is out of range for this model":                 "시작 깊이가 이 모델의 범위 밖",
		"QD is too low to leave room for prefetch":                          "QD 가 낮아 프리페치 자리가 없음",
		"the computed depth does not match the catalog":                     "계산한 깊이가 카탈로그와 다름",
		"catalog values, measured":                                          "카탈로그 값, 측정됨",
		"starting values, not measured yet":                                 "시작 값, 아직 측정 전",
		"values you set, not measured":                                      "직접 정한 값, 측정 전",
		"server running · model choice locked · to unload: %s, ctrl+f":      "서버 실행 중 · 모델 선택 잠김 · 모델을 내리려면 %s ctrl+f",
		"verify":                     "검증",
		"speed check":                "디스크 속도 확인",
		"verifying …":                "검증하는 중 …",
		"checking the drive speed …": "디스크 읽기 속도 확인하는 중 …",

		// UX-KR (the Korean string table, reviews/_landing_8_UX-R5_v8-krtab.md, as the user chose it:
		// decisions (1)-(8) of the brief and the addendum): entries new to the map
		"expert cache only · auto = the PowerShell autotune":                                            "전문가 캐시만 · auto = PowerShell 자동 조정",
		"how much text one chat can hold":                                                               "대화 하나가 담을 수 있는 글의 양",
		"auto = the sweep binding, else QD 1 · virtual: default QD 8, below the wave width cannot work": "auto = QD 스윕 결과, 없으면 QD 1 · virtual: 기본 QD 8, wave 폭 미만은 동작 불가",
		"packed or virtual · a mode without its files opens the setup plan":                             "packed 또는 virtual · 파일이 없는 방식을 고르면 준비 계획이 열림",
		"a short warmup request right after ready · on by default":                                      "준비 직후 짧은 예열 요청 · 기본은 켬",
		"catalog or init · a set K and N pair is the run's pair":                                        "catalog 또는 init · K·N 을 정하면 그 쌍으로 실행",
		"manual prefetch K, set with N · auto = no pair · not a validated pair: unmeasured":             "수동 프리페치 K, N 과 함께 설정 · auto = 쌍 없음 · 검증된 쌍이 아니면 미측정",
		"manual prefetch N, set with K · on while QD is at least N+1, else off with its reason":         "수동 프리페치 N, K 와 함께 설정 · QD 가 N+1 이상일 때 켜짐, 아니면 이유와 함께 꺼짐",
		"catalog defaults": "카탈로그 기본값",
		"defaults · the saved settings belong to another file": "기본값 · 저장된 설정은 다른 파일의 것",
		"no catalog profile: settings not ported":              "카탈로그 프로필 없음: 설정 미지원",
		"saved settings (%s)":                                  "저장된 설정 (%s)",
		"broken":                                               "broken",
		"unsupported":                                          "unsupported",
		"missing":                                              "없음",
		"empty":                                                "비어 있음",
		"watching":                                             "감시 중",
		"read catalog":                                         "카탈로그 읽기",
		"identify model":                                       "모델 확인",
		"repack files":                                         "리팩 파일",
		"launcher locks":                                       "런처 잠금",
		"queue depth":                                          "큐 깊이(QD)",
		"server args":                                          "서버 인자",
		"wait /health":                                         "/health 기다림",
		"send CTRL_BREAK":                                      "CTRL_BREAK 보내기",
		"wait exit (60 s)":                                     "종료 기다림 (60초)",
		"force kill":                                           "강제 종료",
		"port listener gone":                                   "포트 닫힘 확인",
		"clean up":                                             "정리",
		"check":                                                "확인",
		"prefetch on":                                          "프리페치 켬",
		"prefetch off":                                         "프리페치 끔",
		"skipped":                                              "건너뜀",
		"waiting":                                              "대기",
		"tokens":                                               "토큰",
		"every":                                                "주기",
		"budget":                                               "예산",
		"available":                                            "사용 가능",
		"verdict":                                              "판정",
		"refused":                                              "거절",
		"days":                                                 "일",
		"this file differs from the catalog file · catalog settings, prefetch off, no reference speed": "카탈로그 파일과 다른 파일 · 카탈로그 설정, 프리페치 끔, 기준 속도 없음",
		"prefetch «init» ignored: this model's prefetch is fixed by the catalog":                       "프리페치 «init» 무시: 이 모델의 프리페치는 카탈로그가 정함",
		"prefetch «init» is an unvalidated starting point · no speed claim":                            "프리페치 «init» 는 검증 전 시작값 · 속도 주장 없음",
		"an earlier setup was cut off · ⏎ opens the setup plan":                                        "이전 준비가 중간에 끊김 · ⏎ 준비 계획 열기",
		"the repack files do not come from this file · ⏎ opens the setup plan":                         "리팩 파일이 이 모델 파일에서 만든 것이 아님 · ⏎ 준비 계획 열기",
		"no %s repack files yet · ⏎ opens the setup plan":                                              "아직 %s 리팩 파일 없음 · ⏎ 준비 계획 열기",
		"the %s repack check failed at item %d of %d · ⏎ opens the setup plan":                         "%s 리팩 점검 실패(%d/%d 항목) · ⏎ 준비 계획 열기",
		"repack output missing": "리팩 결과물 없음",
		"outside the catalog: the template setup is not ported yet": "카탈로그 밖: 템플릿 준비 미지원",
		"%s is not ported":                               "%s 미지원",
		"not enough free disk space":                     "디스크 여유 공간 부족",
		"not enough RAM for this budget":                 "이 예산에 RAM 부족",
		"the model file is missing or cannot be used":    "모델 파일이 없거나 쓸 수 없음",
		"unsupported model structure":                    "지원하지 않는 모델 구조", // UX-R21 (item 32)
		"the model file is missing":                      "모델 파일이 없음",
		"the model file cannot be read":                  "모델 파일을 읽을 수 없음",
		"a setting is outside its range":                 "설정 값이 범위 밖",
		"the launcher files are damaged · install again": "배포 파일 손상 · 다시 설치",
		"the catalog check failed":                       "카탈로그 점검 실패",
		"the repack check failed (redo the repack)":      "리팩 점검 실패(리팩 다시 하기)",
		"a setting of this start was refused":            "이번 시작의 설정 하나가 거부됨",
		"APPLIED SETTINGS, change in Models":             "적용된 설정값, 변경은 Models",
		"K·N setting unconfirmed":                        "K·N 값 설정 미확인",
		"not applied":                                    "미적용",
	},
}
