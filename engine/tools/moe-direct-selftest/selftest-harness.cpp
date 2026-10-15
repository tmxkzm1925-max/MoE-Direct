#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "selftest-harness.h"
#include "ggml-moe-direct.h"
#include "ggml-moe-direct-internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <string>

// ===========================================================================
// 최소 PASS/FAIL 하네스
// ===========================================================================
int g_total = 0;
int g_failed = 0;

// ★r3 — `--filter <접두사>` 선택 실행(조립기 phase-0 명령이 가리키는 실존 CLI).
//   ① 그룹 단위: 그룹 태그와 접두사가 **서로 접두 관계**면 그 그룹만 돌고 나머지는 통째로 스킵.
//   ② 그룹 안: 접두사가 그룹 태그보다 **길 때만** 검사 이름 접두사로 한 번 더 거른다
//      (`--filter P43` = P43 그룹 전건 · `--filter P43g` = 그 안의 P43g* 만).
//   필터가 비어 있으면(=평시) 아래 두 분기 모두 거짓이라 동작이 완전히 동일하다.
std::string g_filter;         // 빈 문자열 = 전건 실행
static std::string g_group_tag;      // 현재 그룹 태그
int         g_groups_run     = 0;
int         g_groups_skipped = 0;

bool check(bool cond, const std::string & name) {
    if (!g_filter.empty() && g_filter.size() > g_group_tag.size() &&
        name.compare(0, g_filter.size(), g_filter) != 0) {
        return cond;                 // 필터 밖 — 계수도 출력도 하지 않는다
    }
    g_total++;
    if (!cond) g_failed++;
    printf("[%s] %s\n", cond ? "PASS" : "FAIL", name.c_str());
    fflush(stdout);
    return cond;
}

// 두 문자열이 서로 접두 관계인가(짧은 쪽 길이만큼 같으면 참).
static bool prefix_compatible(const std::string & a, const std::string & b) {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    return a.compare(0, n, b, 0, n) == 0;
}

void run_group(const char * tag, void (*fn)()) {
    if (!g_filter.empty() && !prefix_compatible(g_filter, tag)) {
        g_groups_skipped++;
        return;
    }
    g_groups_run++;
    g_group_tag = tag;
    fn();
    g_group_tag.clear();
    // ★★T-1(교차 r3 S-2) — **종단 유계의 마지막 칸**. 유계 종료가 timeout 으로 끝난 것은
    //   **shutdown 회귀** 이고, 그 순간 dispatcher 스레드가 살아서 하네스 상태를 계속
    //   만진다 ⇒ 그대로 진행하면 use-after-free 로 **세그폴트**한다(실측함).
    //   교착도 크래시도 관문의 올바른 출력이 아니므로, **수치를 적고 즉시 종료**한다
    //   (O6b lost-wake 워치도그 선례와 같은 처분 · exit 1).
    if (ggml_moe_direct_test_stop_timed_out()) {
        check(false, std::string("RUN. ★★shutdown 유계 초과 — 그룹 [") + tag +
                     "] 이후 dispatcher 정지가 timeout 했다(= shutdown 회귀). 하네스 상태를 "
                     "쓰는 스레드가 살아 있어 이후 진행은 안전하지 않으므로 즉시 종료한다");
        printf("\n==== 총계 ====\n%d/%d PASS (%d FAIL) — shutdown 유계 초과로 즉시 종료\n",
               g_total - g_failed, g_total, g_failed);
        fflush(stdout);
        _exit(1);
    }
}

void section(const std::string & title) {
    printf("\n==== %s ====\n", title.c_str());
    fflush(stdout);
}

std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
std::string wide_to_utf8(const std::wstring & w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}
void set_env(const char * name, const std::string & value) { _putenv_s(name, value.c_str()); }
void clear_env(const char * name) { _putenv_s(name, ""); }

std::wstring g_scratch_root;
static std::wstring g_scratch_parent;      // ★§2-10 위치 격리 상위 폴더
static int g_case_counter = 0;

// ===========================================================================
// ★★§2-10 셀프테스트 임시 산출물 규약 (권위 = `HANDOFF_DEV.md` §2-10 · 경위 = L:teardown)
//   실사고: `%TEMP%\moe-direct-selftest-*` 996개·**267.16GiB** 누적으로 C: 압박.
//   원인 코드는 "삭제 코드 없음" 하나였다(런당 scratch 최대 3개 — 부모·VEH child·trace child).
//   계약 3항을 여기서 시공한다:
//     1. **자기 정리 기본**: 성공 시 자기 워크디렉토리 삭제 · **실패 시에만 잔존**(경로 1줄 고지)
//     2. **위치 격리**: `%TEMP%\moe-selftest\` 단일 상위 폴더 아래
//     3. **시작 시 청소 게이트**: 이전 잔존물(실행 중 제외) 자동 정리 + 잔존 총량 출력
//   ★4항(잔존물은 물증이 아니다 — 판정 권위는 N/N 수치+환경 줄)은 문서 규약이라 코드 없음.
//   ★"실행 중 제외" 판정 = 폴더 이름의 pid 로 `OpenProcess` — 살아 있으면 건드리지 않는다.
//     이름 규약과 무관한 폴더는 **절대 지우지 않는다**(남의 것을 지우지 않는다).
// ===========================================================================
static const wchar_t * const kScratchPrefix = L"moe-direct-selftest-";

// 재귀 삭제 — 삭제한 바이트를 누적한다(청소 게이트가 총량을 보고한다).
bool remove_tree_w(const std::wstring & dir, uint64_t * out_bytes) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    bool ok = true;
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (wcscmp(fd.cFileName, L".") == 0 || wcscmp(fd.cFileName, L"..") == 0) continue;
            const std::wstring p = dir + L"\\" + fd.cFileName;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                ok = remove_tree_w(p, out_bytes) && ok;
            } else {
                if (out_bytes)
                    *out_bytes += ((uint64_t)fd.nFileSizeHigh << 32) | (uint64_t)fd.nFileSizeLow;
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_READONLY)
                    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
                if (!DeleteFileW(p.c_str())) ok = false;
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    if (!RemoveDirectoryW(dir.c_str())) ok = false;
    return ok;
}

// ★★R-6(교차 r1 F6) — **unknown 은 보존한다**(구 판은 조회 실패를 전부 dead 로 처분했다).
//   이 판정의 유일한 소비자는 `sweep_stale_scratch` 의 **삭제** 결정이라 오판의 두 방향이
//   대칭이 아니다: 죽은 것을 살았다고 하면 폴더 하나가 한 라운드 더 남을 뿐이지만, **살아
//   있는 것을 죽었다고 하면 지금 돌고 있는 다른 selftest 의 워크디렉토리를 지운다**. 그래서
//   "모르면 살아 있다고 친다"가 이 함수의 옳은 기본값이다.
//   ★`OpenProcess` 실패를 dead 로 뭉개면 안 되는 이유: 권한 차이(상승 프로세스·다른 사용자·
//     보호 프로세스)로 **조회만** 못 할 뿐 그 PID 는 실행 중일 수 있다. 진짜 부재는
//     `ERROR_INVALID_PARAMETER`(존재하지 않는 PID 에 대한 Win32 규약)로만 구별된다.
bool pid_is_alive(unsigned long pid) {
    if (pid == 0) return false;
    SetLastError(0);
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (h == nullptr) {
        const DWORD e = GetLastError();
        // 존재하지 않는 PID = 확정 dead. 그 외(ACCESS_DENIED 등) = unknown ⇒ 보존.
        return e != ERROR_INVALID_PARAMETER;
    }
    DWORD code = 0;
    const bool queried = (GetExitCodeProcess(h, &code) != 0);
    CloseHandle(h);
    if (!queried) return true;                // 조회 실패도 unknown ⇒ 보존
    return code == STILL_ACTIVE;
}

// §2-10 ③ 시작 시 청소 게이트 / ①의 사후 정리 겸용. 반환 = 지운 디렉토리 수.
int sweep_stale_scratch(uint64_t * out_bytes, bool quiet) {
    if (g_scratch_parent.empty()) return 0;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((g_scratch_parent + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int removed = 0, skipped_live = 0;
    uint64_t bytes = 0;
    const unsigned long self = (unsigned long)GetCurrentProcessId();
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (wcscmp(fd.cFileName, L".") == 0 || wcscmp(fd.cFileName, L"..") == 0) continue;
        const size_t plen = wcslen(kScratchPrefix);
        if (wcsncmp(fd.cFileName, kScratchPrefix, plen) != 0) continue;   // 남의 것은 건드리지 않는다
        const wchar_t * digits = fd.cFileName + plen;
        if (*digits == L'\0') continue;
        wchar_t * endp = nullptr;
        const unsigned long pid = wcstoul(digits, &endp, 10);
        if (endp == digits || (endp && *endp != L'\0')) continue;         // 이름 규약 밖
        if (pid == self) continue;
        if (pid_is_alive(pid)) { skipped_live++; continue; }              // 실행 중 제외
        if (remove_tree_w(g_scratch_parent + L"\\" + fd.cFileName, &bytes)) removed++;
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    if (out_bytes) *out_bytes += bytes;
    if (!quiet && (removed > 0 || skipped_live > 0)) {
        printf("scratch sweep: %d dir 정리(%.2f MiB) · %d dir 실행 중이라 보존\n",
               removed, (double)bytes / (1024.0 * 1024.0), skipped_live);
    }
    return removed;
}

// §2-10 ① 자기 정리 — 성공이면 지우고, 실패면 **남기고 경로를 고지**한다(진단용).
// ★★R-6(교차 r1 F6) — 반환 = **정리가 실제로 끝났는가**. 구 판은 `remove_tree_w` 의 반환을
//   버리고 경로만 비워서, 열린 핸들·AV 공유 잠금·ACL 로 삭제가 실패해도 **"성공 exit + 잔존"**
//   이 성립했다(§2-10 의 자기 정리 계약이 조용히 깨진 채로 녹색). 실패는 전파한다:
//   ①stderr 에 경로와 함께 명시 ②`g_scratch_root` 를 **비우지 않는다**(잔존 사실을 남긴다)
//   ③호출자가 exit code 에 반영한다. 실패 런의 "보존"(위 분기)과 달리 이건 **사고**다.
bool teardown_scratch_root(bool success) {
    if (g_scratch_root.empty()) return true;
    if (!success) {
        fprintf(stderr, "scratch kept for diagnosis: %ls\n", g_scratch_root.c_str());
        fflush(stderr);
        return true;   // 의도된 보존 — 정리 실패가 아니다
    }
    uint64_t bytes = 0;
    if (!remove_tree_w(g_scratch_root, &bytes)) {
        fprintf(stderr, "scratch teardown FAILED (residue left behind): %ls\n"
                        "  - 성공 런이 자기 워크디렉토리를 못 지웠다(열린 핸들·AV 잠금·ACL 의심).\n"
                        "  - §2-10 ① 자기 정리 계약 위반이므로 exit code 로 전파한다.\n",
                g_scratch_root.c_str());
        fflush(stderr);
        return false;
    }
    g_scratch_root.clear();
    return true;
}

// The default lives on the DECLARATION (selftest-harness.h); repeating it here would not
// compile. See that header for why the parameter exists at all.
void init_scratch_root(bool announce) {
    wchar_t tmp[MAX_PATH];
    DWORD n = GetTempPathW(MAX_PATH, tmp);
    if (n == 0 || n > MAX_PATH) {
        fprintf(stderr, "FATAL: GetTempPathW 실패(반환=%lu, err=%lu)\n", (unsigned long)n, (unsigned long)GetLastError());
        fflush(stderr); exit(2);
    }
    // ★§2-10 ② 위치 격리 — 워크디렉토리는 `%TEMP%\moe-selftest\` 단일 상위 폴더 아래로.
    //   청소·측정이 한 경로에서 끝나야 "그때 그 폴더가 뭐였나"를 다시 조사하지 않는다.
    g_scratch_parent = std::wstring(tmp) + L"moe-selftest";
    if (!CreateDirectoryW(g_scratch_parent.c_str(), nullptr)) {
        const DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            fprintf(stderr, "FATAL: 스크래치 상위 폴더 생성 실패 err=%lu\n", (unsigned long)err);
            fflush(stderr); exit(2);
        }
    }
    wchar_t buf[64];
    swprintf(buf, 64, L"%ls%lu", kScratchPrefix, (unsigned long)GetCurrentProcessId());
    g_scratch_root = g_scratch_parent + L"\\" + buf;
    if (!CreateDirectoryW(g_scratch_root.c_str(), nullptr)) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            fprintf(stderr, "FATAL: 스크래치 루트 생성 실패 err=%lu\n", (unsigned long)err); fflush(stderr); exit(2);
        }
    }
    // The ONLY stdout this function has. A caller whose stdout is a machine-read
    // protocol passes announce=false; everyone else keeps the line.
    if (announce) {
        printf("scratch root: (temp)\\moe-selftest\\%ls\n", buf);
    }
}

std::wstring make_case_dir(const std::string & tag) {
    wchar_t buf[64];
    swprintf(buf, 64, L"\\case_%03d_", g_case_counter++);
    std::wstring dir = g_scratch_root + buf + utf8_to_wide(tag);
    if (!CreateDirectoryW(dir.c_str(), nullptr)) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            fprintf(stderr, "FATAL: case 디렉토리 생성 실패 err=%lu tag=%s\n", (unsigned long)err, tag.c_str());
            fflush(stderr); exit(2);
        }
    }
    return dir;
}

// ===========================================================================
// HT — ★★R-6: 하네스 자기 정리(§2-10)의 **실패 전파** (교차 r1 F6)
//
// ★이 그룹이 존재하는 이유: 두 결함 다 "조용히 틀리는" 종류라 검사 총계로는 영영 안 드러난다.
//   ①삭제 실패를 성공으로 보고하면 **녹색 exit 뒤에 잔존물이 쌓이고**(계약이 깨진 줄 아무도
//   모른다) ②살아 있는 PID 를 dead 로 처분하면 **지금 돌고 있는 다른 런의 워크디렉토리를
//   지운다**(남의 것을 지우지 않는다는 §2-10 의 대전제 위반). 관문이 없으면 사고가 난 뒤에야
//   안다 — 그것도 "왜 파일이 없지"로.
// ===========================================================================
void test_harness_teardown() {
    section("HT. ★★R-6 하네스 자기 정리(§2-10) 실패 전파 — 삭제 실패 보고 · PID unknown 보존");

    // ── ① `remove_tree_w` 는 삭제 실패를 **false 로 보고**한다 ──
    //    실패는 **열린 핸들(공유 전면 거부)**로 만든다. 손편집·위조가 아니라 실 런타임에서
    //    도달 가능한 사건(AV 스캔의 열린 핸들·다른 프로세스 점유·ACL)과 **같은 기제**다.
    {
        const std::wstring d = make_case_dir("ht_locked");
        const std::wstring f = d + L"\\locked.bin";
        HANDLE h = CreateFileW(f.c_str(), GENERIC_WRITE, /*dwShareMode=*/0, nullptr,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (check(h != INVALID_HANDLE_VALUE, "HT1a. 잠긴 파일 픽스처 생성(공유 전면 거부)")) {
            uint64_t bytes = 0;
            const bool ok = remove_tree_w(d, &bytes);
            check(!ok,
                  "HT1. ★삭제 실패가 **false 로 보고**된다 — 구 판은 `teardown_scratch_root` 가 이 "
                  "반환을 버리고 경로만 비워서 \"성공 exit + 잔존\" 이 성립했다(§2-10 ① 자기 정리 "
                  "계약이 조용히 깨진 채 녹색). 이제 호출자가 exit code 로 전파한다");
            CloseHandle(h);
            uint64_t b2 = 0;
            check(remove_tree_w(d, &b2),
                  "HT2. ★대조 — 잠금을 풀면 **같은 트리가 정상 삭제**된다(HT1 의 false 가 픽스처 "
                  "결함이 아니라 잠금 때문이라는 증거 · 이 정리가 없으면 내 관문이 만든 잔존물이 "
                  "런 끝의 teardown 을 붉힌다)");
        }
    }

    // ── ①′ `teardown_scratch_root` 자체가 그 실패를 **전파**한다 ──
    //    ★HT1 은 `remove_tree_w` 의 반환을 볼 뿐이고, 그 반환은 구 판에서도 옳았다. F6 의
    //      실체는 **호출자가 그걸 버린 것**이므로 관문도 호출자를 직접 쳐야 한다. `g_scratch_root`
    //      를 잠긴 트리로 잠시 바꿔 실 호출 경로를 그대로 행사하고 원복한다.
    {
        const std::wstring saved = g_scratch_root;
        const std::wstring d = make_case_dir("ht_teardown_fail");
        HANDLE h = CreateFileW((d + L"\\locked.bin").c_str(), GENERIC_WRITE, /*share=*/0,
                               nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        g_scratch_root = d;
        const bool rc   = teardown_scratch_root(/*success=*/true);
        const bool kept = !g_scratch_root.empty();     // 실패면 경로를 **비우지 않는다**
        g_scratch_root = saved;
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        check(!rc && kept,
              "HT7. ★★F6 본체 — 성공 런의 정리가 실패하면 `teardown_scratch_root` 가 **false 를 "
              "돌려주고 경로도 안 비운다**(구 판은 반환을 버리고 무조건 clear 해서 `main` 이 "
              "exit 0 을 냈다 = 성공 위장). 호출자 2곳이 이 값을 exit code 에 반영한다 [rc=" +
              std::to_string((int)rc) + " path_kept=" + std::to_string((int)kept) + "]");
        uint64_t b = 0; remove_tree_w(d, &b);          // 관문이 만든 잔존물은 관문이 치운다
    }
    // ── ①″ 대조 — 정상 트리는 true 를 돌려주고 경로를 비운다(위 false 가 상시값이 아니다) ──
    {
        const std::wstring saved = g_scratch_root;
        const std::wstring d = make_case_dir("ht_teardown_ok");
        g_scratch_root = d;
        const bool rc      = teardown_scratch_root(/*success=*/true);
        const bool cleared = g_scratch_root.empty();
        g_scratch_root = saved;
        check(rc && cleared,
              "HT8. ★대조 — 잠금이 없으면 같은 호출이 true·경로 clear 로 끝난다(HT7 의 false 가 "
              "'항상 실패'가 아니라 **실패를 실패로 보고한 것**이라는 증거) [rc=" +
              std::to_string((int)rc) + " cleared=" + std::to_string((int)cleared) + "]");
    }

    // ── ② `pid_is_alive` — **unknown 은 보존**(살아 있다고 친다) ──
    {
        check(!pid_is_alive(0), "HT3. pid 0 = dead(스윕 대상)");
        check(pid_is_alive((unsigned long)GetCurrentProcessId()), "HT4. 자기 자신 = alive(보존)");
        // 존재할 수 없는 PID — Windows PID 는 4의 배수라 아래 값은 영영 유효하지 않다
        // (`OpenProcess` → ERROR_INVALID_PARAMETER = **확정 부재**). 이건 dead 로 처분해야
        // 스윕이 제 일을 한다(모든 걸 보존하면 그건 그냥 청소를 안 하는 것이다).
        check(!pid_is_alive(0xFFFFFFF1ul),
              "HT5. ★존재할 수 없는 PID = dead — `ERROR_INVALID_PARAMETER` 만이 '확정 부재' 다"
              "(보존 규칙이 스윕 기능 자체를 죽이지 않는다는 반대편 증거)");
        // 조회가 막히는 PID(System=4)로 unknown 팔을 행사한다. 호스트 권한에 따라 조회가
        // 될 수도 있으므로 **기대값을 OpenProcess 결과에서 직접 정해** 환경 의존을 없앤다
        // (검사 건수는 항상 1 — 총계 결정론 유지).
        SetLastError(0);
        HANDLE sys = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, 4);
        const DWORD e = sys ? 0ul : GetLastError();
        if (sys) CloseHandle(sys);
        const bool unknown_arm = (sys == nullptr && e != ERROR_INVALID_PARAMETER);
        check(pid_is_alive(4) == true,
              std::string("HT6. ★System PID(4) 는 **보존**된다 — ") +
              (unknown_arm
                 ? ("조회가 막힌 unknown(err=" + std::to_string((unsigned long)e) +
                    ") 인데도 alive 로 처분했다. **구 판은 여기서 dead 로 처분해** 실행 중인 "
                    "프로세스의 워크디렉토리를 스윕 대상으로 삼았다 — 이 팔이 F6 의 본체다")
                 : "이 호스트는 System 조회가 허용돼 unknown 팔이 생성되지 않았다"
                   "(정상 조회 경로로 alive 확인 · unknown 처분은 HT5 와 코드상 같은 분기)"));
    }
}

