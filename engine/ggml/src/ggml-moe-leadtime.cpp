// MoE lead-time 계측 모듈 본체 — 사양: bench/techdev/repack_space/PROBE_LEADTIME_SPEC_R1.md ★FROZEN v1.0
//
// 이 파일이 소유하는 것: env 게이트 · A/B 분리 POD 링 버퍼(arm 시 전량 선할당) · 원자 인덱스 예약
// · 사이드카 2종 기록 · 정직 강하(drop → complete=0).
// 이 파일이 소유하지 않는 것: 스탬프를 부르는 지점(ggml-backend.cpp 2곳 · ggml-moe-direct.cpp
// acquire 1곳) · T2/T3(기존 IO3) · 판독(오프라인 python) · PhaseA/PhaseB 의 어떤 것도.
//
// ★동시성 모델: 두 스탬프 지점은 모두 graph 스케줄/실행 경로(단일 스케줄러 스레드 · acquire 는
//   ith==0 단독)에서 불린다. 그래도 이 모듈은 **락 없이 다중 생산자 안전**하게 만든다 —
//   칸 예약은 `fetch_add` 1회이고 그 뒤의 memcpy 는 자기 칸에만 쓴다. 예약 순번(`seq`)과 QPC 의
//   순서는 단일 생산자에서 일치하고, 다중 생산자에서 어긋날 수 있으므로 **순서 검증은 판독기의
//   일**이다(사양 §4b "ordering 전건"). 엔진은 원자료를 있는 그대로 싣는다.
//
// Windows 전용 — 비Windows 는 파일 하단 스텁.

#include "ggml-moe-leadtime.h"
#include "ggml-moe-leadtime-internal.h"   // ★테스트 전용 진입점(공개 헤더 표면 밖 — 교차 r2 R3-4)

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstddef>      // offsetof — wire 오프셋 동결 static_assert
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// ===========================================================================
// §1. wire 구조체 (리틀엔디언·고정폭 — 판독기와의 유일한 계약)
// ===========================================================================

#pragma pack(push, 1)

// 공통 128B 헤더. 필드 집합은 사양 §2 가 열거한 9종(magic/version/qpc_frequency/t0_qpc/
// boot_id/pid/complete/dropped/count)을 전부 담고, IO3/PhaseA 계열 128B 헤더 관례와 자리를
// 맞춘다(같은 판독 습관으로 읽히게). `record_capacity` 는 헤드룸 감사용 추가 칸이다 —
// 실 런에서 drop 이 나기 **전에** 여유가 얼마였는지 사후 확인할 수 있어야 한다.
struct LtHeader {
    char     magic[8];          // "MOELT1\0\0"                        @0
    uint32_t layout_version;    // 1                                   @8
    uint32_t file_kind;         // 1=A stamps · 2=B stamps             @12
    uint32_t record_bytes;      // 64 / 32                             @16
    uint32_t engine_mode;       // 0=미기재 1=bin 2=virtual (echo)     @20
    uint64_t record_count;      //                                     @24
    uint64_t payload_digest;    // FNV-1a-64 over payload bytes         @32
    uint64_t qpc_frequency;     //                                     @40
    uint64_t t0_qpc;            // arm 시각(같은 프로세스 단일 시계)    @48
    uint32_t pid;               //                                     @56
    uint32_t complete;          // 1=결손 없음 0=fail-close            @60
    uint32_t n_expert;          // echo(0=미기재)                      @64
    uint32_t n_moe_layers;      // echo(0=미기재)                      @68
    uint32_t reserved0;         //                                     @72
    uint32_t reserved1;         //                                     @76
    uint64_t boot_id;           // ★IO3 조인 키                        @80
    uint64_t dropped_records;   // A+B 합계(모듈 단일 계수)             @88
    uint64_t record_capacity;   // 이 축의 선할당 칸 수                 @96
    uint8_t  reserved[24];      //                                     @104..127
};

// kind=1 · 64B — T0 스탬프
struct LtStampA {
    uint32_t seq;                                  // @0   이 축의 예약 순번(0-based)
    uint8_t  site;                                 // @4   1=A1 2=A2
    uint8_t  flags;                                // @5   b0=NAME_TRUNCATED b1=NAME_NULL
    uint16_t name_len;                             // @6   ★절단 **전** 원 이름 길이
    uint64_t qpc;                                  // @8
    char     name[GGML_MOE_LEADTIME_NAME_BYTES];   // @16..63  남는 칸 = 0
};

// kind=2 · 32B — T1 스탬프
struct LtStampB {
    uint32_t seq;               // @0
    int16_t  mlpos;             // @4
    int16_t  part_idx;          // @6
    uint64_t qpc;               // @8   pre-lock 진입 시각
    int32_t  n_ids;             // @16
    uint8_t  flags;             // @20  b0=PREFILL
    uint8_t  reserved0;         // @21
    uint16_t reserved1;         // @22
    uint64_t reserved2;         // @24
};

#pragma pack(pop)

static_assert(sizeof(LtHeader) == GGML_MOE_LEADTIME_HDR_BYTES, "lead header must be exactly 128B");
static_assert(sizeof(LtStampA) == GGML_MOE_LEADTIME_A_BYTES,   "lead stamp A row must be exactly 64B");
static_assert(sizeof(LtStampB) == GGML_MOE_LEADTIME_B_BYTES,   "lead stamp B row must be exactly 32B");

// 오프셋 동결 — 판독기가 바이트 배치를 하드코딩하므로 폭만 맞고 자리가 밀리면 조용히 오독한다.
static_assert(offsetof(LtHeader, layout_version)  ==   8, "LtHeader.layout_version @8");
static_assert(offsetof(LtHeader, file_kind)       ==  12, "LtHeader.file_kind @12");
static_assert(offsetof(LtHeader, record_bytes)    ==  16, "LtHeader.record_bytes @16");
static_assert(offsetof(LtHeader, engine_mode)     ==  20, "LtHeader.engine_mode @20");
static_assert(offsetof(LtHeader, record_count)    ==  24, "LtHeader.record_count @24");
static_assert(offsetof(LtHeader, payload_digest)  ==  32, "LtHeader.payload_digest @32");
static_assert(offsetof(LtHeader, qpc_frequency)   ==  40, "LtHeader.qpc_frequency @40");
static_assert(offsetof(LtHeader, t0_qpc)          ==  48, "LtHeader.t0_qpc @48");
static_assert(offsetof(LtHeader, pid)             ==  56, "LtHeader.pid @56");
static_assert(offsetof(LtHeader, complete)        ==  60, "LtHeader.complete @60");
static_assert(offsetof(LtHeader, n_expert)        ==  64, "LtHeader.n_expert @64");
static_assert(offsetof(LtHeader, n_moe_layers)    ==  68, "LtHeader.n_moe_layers @68");
static_assert(offsetof(LtHeader, boot_id)         ==  80, "LtHeader.boot_id @80");
static_assert(offsetof(LtHeader, dropped_records) ==  88, "LtHeader.dropped_records @88");
static_assert(offsetof(LtHeader, record_capacity) ==  96, "LtHeader.record_capacity @96");
static_assert(offsetof(LtHeader, reserved)        == 104, "LtHeader.reserved @104");

static_assert(offsetof(LtStampA, seq)      ==  0, "LtStampA.seq @0");
static_assert(offsetof(LtStampA, site)     ==  4, "LtStampA.site @4");
static_assert(offsetof(LtStampA, flags)    ==  5, "LtStampA.flags @5");
static_assert(offsetof(LtStampA, name_len) ==  6, "LtStampA.name_len @6");
static_assert(offsetof(LtStampA, qpc)      ==  8, "LtStampA.qpc @8");
static_assert(offsetof(LtStampA, name)     == 16, "LtStampA.name @16");

static_assert(offsetof(LtStampB, seq)       ==  0, "LtStampB.seq @0");
static_assert(offsetof(LtStampB, mlpos)     ==  4, "LtStampB.mlpos @4");
static_assert(offsetof(LtStampB, part_idx)  ==  6, "LtStampB.part_idx @6");
static_assert(offsetof(LtStampB, qpc)       ==  8, "LtStampB.qpc @8");
static_assert(offsetof(LtStampB, n_ids)     == 16, "LtStampB.n_ids @16");
static_assert(offsetof(LtStampB, flags)     == 20, "LtStampB.flags @20");
static_assert(offsetof(LtStampB, reserved0) == 21, "LtStampB.reserved0 @21");
static_assert(offsetof(LtStampB, reserved1) == 22, "LtStampB.reserved1 @22");
static_assert(offsetof(LtStampB, reserved2) == 24, "LtStampB.reserved2 @24");

// ===========================================================================
// §2. 상태
// ===========================================================================

constexpr uint64_t LT_FNV_SEED = 0xcbf29ce484222325ull;
constexpr uint32_t LT_DEFAULT_MAX_MB = 64;      // 사양 §2 기본값
constexpr uint32_t LT_MAX_MAX_MB     = 4096;    // 상한(오타로 수 GB 를 잡는 것을 막는다)
constexpr size_t   LT_NAME_SCAN_MAX  = 255;     // 이름 길이 스캔 상한(무경계 strlen 금지)

// 게이트 상태. 미해소(-1)는 hot-path 에서 **false 로 취급**되며 해소를 유발하지 않는다.
// ★CLOSED(2) = flush 가 닫은 상태(publication 계약 ①) — OFF 와 **구분해야** 한다: OFF 는 애초에
//   무장 안 된 런이라 스탬프가 손실이 아니고, CLOSED 는 무장된 런에서 기록 창이 닫힌 뒤 도착한
//   것이라 **손실**이므로 dropped + complete=0 로 정직 처분해야 한다.
constexpr int8_t LT_GATE_UNRESOLVED = -1;
constexpr int8_t LT_GATE_OFF        =  0;
constexpr int8_t LT_GATE_ON         =  1;
constexpr int8_t LT_GATE_CLOSED     =  2;

// flush 의 in-flight writer drain 상한 — ★**반복 횟수가 아니라 경과 시간(ms)** 이다.
//   구 판은 `Sleep(1)` 200회로 셌는데, Windows 의 `Sleep(1)` 은 타이머 해상도에 따라 실제로
//   1~15ms 를 잔다 ⇒ 실 상한이 200ms~3s 로 흔들리고 로그의 "within 200 ms" 문면이 거짓이었다
//   (자기 관측이 자기 로그와 안 맞는 상태). 벽시계로 재면 상한이 예측 가능해지고 문면도 참이 된다.
// 스탬프는 마이크로초급이라 실 런에서 첫 확인에 끝난다 — 이 상한은 "쓰던 스레드가 죽었다" 같은
// 병리에서 finalize 가 영구 대기하지 않게 하는 안전장치이고, 초과 시 정직 강하(complete=0)다.
constexpr uint32_t LT_DRAIN_BUDGET_MS = 200;

// 원자 인덱스 예약 링. `next` 는 상한을 넘어서도 계속 증가한다(넘친 예약은 곧 drop 이며,
// 그 수가 "얼마나 모자랐나"를 그대로 알려준다 — 넘칠 때 되감으면 그 정보가 사라진다).
struct LtRing {
    uint8_t *             base = nullptr;
    uint64_t              cap  = 0;     // 칸 수
    uint32_t              rec  = 0;     // 칸 바이트
    std::atomic<uint64_t> next{0};      // 예약 카운터

    uint64_t used() const {
        const uint64_t n = next.load(std::memory_order_acquire);
        return n < cap ? n : cap;
    }
    void release() {
        if (base) { VirtualFree(base, 0, MEM_RELEASE); base = nullptr; }
        cap = 0; rec = 0;
        next.store(0, std::memory_order_release);
    }
};

struct LtState {
    // ── 게이트/식별 ──
    std::atomic<int8_t> gate{LT_GATE_UNRESOLVED};
    std::mutex          init_mu;        // 해소·arm·reset·flush 직렬화(hot path 는 안 잡는다)
    bool                resolved = false;
    std::string         out_prefix;     // 비면 카운터만
    uint32_t            max_mb   = LT_DEFAULT_MAX_MB;
    uint64_t            boot_id  = 0;
    uint64_t            qpc_freq = 0;
    uint64_t            t0_qpc   = 0;
    uint32_t            engine_mode   = 0;
    uint32_t            n_expert      = 0;
    uint32_t            n_moe_layers  = 0;
    bool                boot_id_joined = false;   // IO3 boot_id 채택 여부

    // ── 버퍼 ──
    LtRing a, b;

    // ── 결손 회계 ──
    std::atomic<uint64_t> dropped{0};
    std::atomic<bool>     complete{true};

    // ── ★publication 계약(교차 r1 BLOCKER): 진입/완료 카운터 쌍 ──
    //   불변식: `w_enter == w_exit` 인 순간, 예약된 전 칸의 본문 memcpy 가 완료돼 있고 그 쓰기가
    //   관측 스레드에 가시적이다(w_exit=release ↔ flush=acquire). flush 는 게이트를 닫은 뒤에만
    //   이 등식을 기다리므로 대기 중 새 예약이 끼어들 수 없다.
    std::atomic<uint64_t> w_enter{0};
    std::atomic<uint64_t> w_exit{0};

    // ── ★테스트 시임(프로덕션 0 — 내부 헤더 `ggml-moe-leadtime-internal.h` 전용) ──
    //   payload 기록 창 안에 late stamp 를 주입할 횟수. `lt_write_file` 이 1회 소비한다.
    std::atomic<int32_t> test_late_stamp{0};

    // ── 진단 계수(테스트 창구) ──
    std::atomic<uint64_t> resolve_count{0};
    std::atomic<uint64_t> qpc_calls{0};

    bool flushed = false;
};

LtState g_lt;

// ===========================================================================
// §3. 유틸
// ===========================================================================

inline uint64_t fnv1a64(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

void lt_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[moe-lead] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

std::string lt_env(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// 내부 raw QPC — 공개 `now_qpc()` 의 진단 카운터를 건드리지 않는다(그 카운터는 "훅이 몇 번
// 시각을 집었나"를 재는 것이라 arm 의 t0 채취가 섞이면 OFF 계약 검사가 무의미해진다).
inline uint64_t lt_qpc_raw() {
    LARGE_INTEGER c;
    if (!QueryPerformanceCounter(&c)) return 0;
    return (uint64_t) c.QuadPart;
}

std::wstring lt_utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), &w[0], n);
    return w;
}

// ---------------------------------------------------------------------------
// 칸 예약. ★hot path 전량: fetch_add 1회 + 비교 1회(+ 호출측 memcpy 1회).
//   상한 초과·미무장(cap==0)은 drop 이고 **latch 로 complete=0** 이다(사양 §4b — lead 는 한 행
//   결손도 지표 전체를 무효화하므로 부분 기록을 유효한 것처럼 두면 안 된다).
//   ★drop 경로에서 로그를 남기지 않는다 — 사양 §2 "hot hook 내 문자열 포맷 금지"를 문자 그대로
//   지킨다. 결손 사실은 flush 로그(`dropped=N`)와 헤더 `complete=0` 이 전달하므로 정보 손실이
//   없다(flush 는 finalize 에서 항상 돌고, flush 전에 죽은 런은 사이드카 자체가 없다).
//   ★예약과 기록 사이의 창(교차 r1 BLOCKER): 예약은 memcpy 보다 먼저 공개되므로 **이 함수만으로는
//   칸의 유효성을 보장하지 못한다**. 유효성은 `LtWriterScope`(w_enter/w_exit)와 flush 의 drain 이
//   함께 보장한다 — 상세 = 헤더 "publication 계약". 구 판 주석("flush 는 단일 스레드 시점이라
//   in-flight 가 없다")은 **가정이었고 계약이 아니었다**: 그 가정을 코드로 강제하는 것이 이 수리다.
// ---------------------------------------------------------------------------
inline uint8_t * lt_reserve(LtRing & r, uint32_t * out_seq) {
    const uint64_t idx = r.next.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= r.cap || !r.base) {
        g_lt.dropped.fetch_add(1, std::memory_order_relaxed);
        g_lt.complete.store(false, std::memory_order_release);
        return nullptr;
    }
    *out_seq = (uint32_t) idx;
    return r.base + (size_t) idx * (size_t) r.rec;
}

// ---------------------------------------------------------------------------
// ★publication 스코프 — 스탬프 1회의 "진입/완료"를 감싼다(헤더 publication 계약).
//   ①생성자가 게이트 확인 **이전에** w_enter 를 올린다: 그래야 "게이트는 ON 으로 읽었지만 아직
//     예약 전"인 스레드가 flush 의 drain 창에 반드시 포함된다(순서가 계약이다 — 반대로 두면
//     정확히 그 스레드가 창을 빠져나가 late 행이 된다).
//   ②소멸자가 **전 반환 경로에서** w_exit 를 release 로 올린다: 본문 memcpy 가 이 store 보다
//     앞서므로, flush 가 acquire 로 등식을 확인한 순간 그 본문이 가시적이다.
// ---------------------------------------------------------------------------
struct LtWriterScope {
    LtWriterScope()  { g_lt.w_enter.fetch_add(1, std::memory_order_acq_rel); }
    ~LtWriterScope() { g_lt.w_exit.fetch_add(1, std::memory_order_release); }
    LtWriterScope(const LtWriterScope &)             = delete;
    LtWriterScope & operator=(const LtWriterScope &) = delete;
};

// 게이트가 스탬프를 받지 않는 상태일 때의 처분. CLOSED(무장된 런의 기록 창 폐쇄 후 도착)는
// **손실**이므로 정직 처분하고, OFF/미해소는 애초에 관측하지 않는 런이라 상태를 건드리지 않는다.
inline void lt_reject_stamp(int8_t gate_state) {
    if (gate_state == LT_GATE_CLOSED) {
        g_lt.dropped.fetch_add(1, std::memory_order_relaxed);
        g_lt.complete.store(false, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// env 해소 + 전량 선할당. ★`init_mu` 를 **이미 쥔 상태**에서만 부른다.
// ---------------------------------------------------------------------------
void lt_resolve_locked() {
    if (g_lt.resolved) return;
    g_lt.resolved = true;
    g_lt.resolve_count.fetch_add(1, std::memory_order_relaxed);

    // ── 게이트: exact "1" 만 on. 미설정/빈값 = 조용한 off. 그 외 = off + 경고 1회 ──
    //   (사양 §2 "`MOE_DIRECT_LEADTIME=1`(그 외=off·경고)". 오타로 조용히 켜지거나 꺼지는 것을
    //    금하는 것이 이 엄격 파싱의 목적이다 — `on`/`true` 도 받지 않는다.)
    const std::string raw = lt_env("MOE_DIRECT_LEADTIME");
    bool on = false;
    if (raw == "1") {
        on = true;
    } else if (!raw.empty()) {
        lt_log("MOE_DIRECT_LEADTIME=\"%s\" is not the only accepted value (\"1\") - lead-time "
               "telemetry stays OFF", raw.c_str());
    }
    if (!on) {
        g_lt.gate.store(LT_GATE_OFF, std::memory_order_release);
        return;   // ★OFF — 어떤 버퍼도 할당하지 않는다(흔적 0 계약)
    }

    // ── 버퍼 상한 ──
    g_lt.max_mb = LT_DEFAULT_MAX_MB;
    {
        const std::string s = lt_env("MOE_DIRECT_LEADTIME_MAX_MB");
        if (!s.empty()) {
            char * end = nullptr;
            const long v = std::strtol(s.c_str(), &end, 10);
            if (end && *end == '\0' && v > 0 && v <= (long) LT_MAX_MAX_MB) {
                g_lt.max_mb = (uint32_t) v;
            } else {
                lt_log("MOE_DIRECT_LEADTIME_MAX_MB=\"%s\" is invalid (1..%u) - using the default %u",
                       s.c_str(), (unsigned) LT_MAX_MAX_MB, (unsigned) LT_DEFAULT_MAX_MB);
            }
        }
    }

    g_lt.out_prefix = lt_env("MOE_DIRECT_LEADTIME_OUT");

    g_lt.qpc_freq = 0;
    {
        LARGE_INTEGER f;
        if (QueryPerformanceFrequency(&f)) g_lt.qpc_freq = (uint64_t) f.QuadPart;
    }
    g_lt.t0_qpc = lt_qpc_raw();

    // 조인 키 기본값 = 자체 생성(note_seal 이 IO3 값을 주면 그것으로 덮는다).
    {
        const uint32_t pid = (uint32_t) GetCurrentProcessId();
        uint64_t h = LT_FNV_SEED;
        h = fnv1a64(h, &pid, sizeof(pid));
        h = fnv1a64(h, &g_lt.t0_qpc, sizeof(g_lt.t0_qpc));
        g_lt.boot_id = h;
        g_lt.boot_id_joined = false;
    }

    // ── ★전량 선할당(사양 §2) — A 3/4 · B 1/4 ──
    //   근거(칸 수 산정): B 는 층 실행당 3행(up/gate/down)으로 상한이 뚜렷하고(≈8만/런),
    //   A 는 split 의 **모든** generic 동기 복사에 서므로(대상 이름만 거르지 않는 것이 사양
    //   §2 NIT) 훨씬 많다. 그래서 여유를 A 쪽에 싣는다. 기본 64MiB → A 786,432칸 · B 524,288칸.
    const uint64_t cap_bytes = (uint64_t) g_lt.max_mb * 1024ull * 1024ull;
    const uint64_t b_bytes   = cap_bytes / 4;
    const uint64_t a_bytes   = cap_bytes - b_bytes;

    g_lt.a.rec = (uint32_t) sizeof(LtStampA);
    g_lt.b.rec = (uint32_t) sizeof(LtStampB);
    const uint64_t cap_a = a_bytes / g_lt.a.rec;
    const uint64_t cap_b = b_bytes / g_lt.b.rec;

    void * pa = (cap_a > 0) ? VirtualAlloc(nullptr, (size_t) (cap_a * g_lt.a.rec),
                                           MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) : nullptr;
    void * pb = (cap_b > 0) ? VirtualAlloc(nullptr, (size_t) (cap_b * g_lt.b.rec),
                                           MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) : nullptr;

    if (!pa || !pb) {
        // ★조용히 OFF 로 접지 않는다. 게이트는 켜 둔 채 cap=0 으로 두면 모든 스탬프가 drop 되고
        //   flush 가 `complete=0` 인 0행 사이드카를 남긴다 ⇒ 판독기가 fail-close 한다.
        //   OFF 로 접으면 "계측 안 한 런"과 구분되지 않아 조용한 무데이터가 된다.
        if (pa) VirtualFree(pa, 0, MEM_RELEASE);
        if (pb) VirtualFree(pb, 0, MEM_RELEASE);
        g_lt.a.base = g_lt.b.base = nullptr;
        g_lt.a.cap  = g_lt.b.cap  = 0;
        g_lt.complete.store(false, std::memory_order_release);
        g_lt.gate.store(LT_GATE_ON, std::memory_order_release);
        lt_log("cannot pre-allocate the lead-time buffers (%u MiB, err=%lu) - the arm is invalid "
               "(complete=0) and every stamp will be dropped",
               (unsigned) g_lt.max_mb, (unsigned long) GetLastError());
        return;
    }

    g_lt.a.base = (uint8_t *) pa; g_lt.a.cap = cap_a;
    g_lt.b.base = (uint8_t *) pb; g_lt.b.cap = cap_b;
    g_lt.gate.store(LT_GATE_ON, std::memory_order_release);

    lt_log("armed: max_mb=%u -> A %llu rows x %uB + B %llu rows x %uB · qpc_freq=%llu · out='%s'",
           (unsigned) g_lt.max_mb,
           (unsigned long long) cap_a, (unsigned) g_lt.a.rec,
           (unsigned long long) cap_b, (unsigned) g_lt.b.rec,
           (unsigned long long) g_lt.qpc_freq, g_lt.out_prefix.c_str());
    if (g_lt.out_prefix.empty()) {
        // ★`complete` 를 내리지 않는다 — "기록한 레코드에 결손이 있는가"와 "쓸 곳이 있는가"는
        //   다른 사실이다. 내리면 명시 접두사로 정상 기록한 다른 런의 판독 규약까지 흐려진다.
        //   파일이 없으면 판독기는 그 부재로 이미 fail-close 한다(shadow 선례와 동일 판단).
        lt_log("MOE_DIRECT_LEADTIME_OUT is unset - counters only, no sidecar will be written");
    }
}

// ---------------------------------------------------------------------------
// 사이드카 1종 기록. IO3 `io3_write_file` / shadow `write_sidecar` 관례를 따르되(CREATE_NEW ·
// 128B 헤더 · 32MiB 청크 · payload FNV-1a-64 · FlushFileBuffers) ★기록 순서가 다르다.
//
// ★★2단계 기록(교차 r2 R3-1 폐합) — **헤더를 마지막에 확정한다**:
//   구 판은 헤더를 **먼저** 써서 `complete=1` 을 박은 뒤 payload 를 썼다. 그 사이(또는 그 뒤)에
//   late stamp 가 도착하면 메모리 `dropped/complete` 만 갱신되고 **파일은 `complete=1` 로 남는다**
//   — 재flush 는 once 래치로 즉시 반환하므로 그 파일을 영원히 고칠 기회가 없다.
//   이제: ①헤더 자리(128B)를 0 으로 예약 ②payload 기록 ③**최종 재검**(게이트 CLOSED ∧
//   `w_enter==w_exit` ∧ `dropped` 무변) ④재검을 통과하고 `complete` 도 참일 때만 `complete=1` 인
//   헤더를 offset 0 에 확정 기록. 재검 실패 = `complete=0` 으로 기록한다.
//   ⇒ payload 기록 창 안에 끼어든 late stamp 가 **파일 헤더에 반영**된다.
//
// `drained=false`(drain 타임아웃)면 ★**payload 를 아예 읽지 않는다**(교차 r2 R3-2): 그 순간
//   버퍼에는 본문을 쓰고 있는 live writer 가 있을 수 있고, 해시·WriteFile 은 그 칸을 비원자로
//   동시 읽는 것이 된다. 대신 `record_count=0` + `complete=0` + `dropped=미커밋 수` 인 **헤더만**
//   기록한다(크기 등식 `128 + 0`이 성립하므로 판독기가 손상 파일로 오독하지 않고, `complete=0`
//   으로 fail-close 한다).
// ---------------------------------------------------------------------------
bool lt_write_file(const std::string & path, uint32_t kind, const LtRing & r,
                   bool drained, uint64_t dropped_at_entry) {
    // ★payload 를 읽어도 되는가 = drain 이 성립했는가. 아니면 행 수 0(헤더 전용).
    const uint64_t n_rows  = drained ? r.used() : 0;
    const size_t   payload = (size_t) n_rows * (size_t) r.rec;

    const std::wstring wp = lt_utf8_to_wide(path);
    HANDLE f = CreateFileW(wp.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        lt_log("cannot create sidecar (CREATE_NEW) path=%s err=%lu", path.c_str(),
               (unsigned long) GetLastError());
        return false;
    }

    // ── ① 헤더 자리 예약(0 채움) ──
    LtHeader zero{};
    DWORD wr = 0;
    bool ok = WriteFile(f, &zero, (DWORD) sizeof(zero), &wr, nullptr) != 0 && wr == sizeof(zero);

    // ── ② payload 기록 + digest(같은 바이트를 한 번만 읽는다) ──
    uint64_t digest = LT_FNV_SEED;
    if (ok && payload) {
        size_t off = 0;
        while (off < payload) {
            const DWORD chunk = (DWORD) std::min<size_t>(payload - off, (size_t) 32 * 1024 * 1024);
            digest = fnv1a64(digest, r.base + off, (size_t) chunk);
            if (!WriteFile(f, r.base + off, chunk, &wr, nullptr) || wr != chunk) { ok = false; break; }
            off += chunk;
        }
    }

    // ★시임 ③(테스트 전용 · 교차 r2 R3-1 회귀) — **payload 기록 창 안**의 late stamp 주입.
    //   그 창은 실 코드 경로에서 외부가 건드릴 수 없으므로, 최종 재검이 정말로 헤더를 내리는지
    //   확인하는 유일한 수단이다. `exchange(0)` 이라 첫 축(A)에서 1회만 소비되고, 재검 실패가
    //   `complete` 래치를 내리므로 둘째 축(B)도 같은 판정을 받는다(런 단위 fail-close).
    //   프로덕션은 이 값이 0 이라 루프가 0회 돈다.
    {
        const int32_t n_late = g_lt.test_late_stamp.exchange(0, std::memory_order_acq_rel);
        for (int32_t i = 0; i < n_late; i++) {
            ggml_moe_leadtime_stamp_a(GGML_MOE_LEADTIME_SITE_A1, 0, "late-stamp-seam");
        }
    }

    // ── ③ ★최종 재검 — payload 기록이 끝난 **뒤**의 상태로 판정한다 ──
    const bool gate_closed = g_lt.gate.load(std::memory_order_acquire) == LT_GATE_CLOSED;
    const uint64_t en_now  = g_lt.w_enter.load(std::memory_order_acquire);
    const uint64_t ex_now  = g_lt.w_exit.load(std::memory_order_acquire);
    const uint64_t drop_now = g_lt.dropped.load(std::memory_order_acquire);
    const bool recheck_ok  = drained && gate_closed && (en_now == ex_now) &&
                             (drop_now == dropped_at_entry);
    if (!recheck_ok) {
        // 정직 강하 — 메모리 래치도 함께 내려 다음 축 파일도 같은 판정을 받는다.
        g_lt.complete.store(false, std::memory_order_release);
        if (drained) {
            lt_log("final recheck failed while writing %s (gate_closed=%d enter=%llu exit=%llu "
                   "dropped %llu->%llu) - header is written with complete=0",
                   path.c_str(), gate_closed ? 1 : 0,
                   (unsigned long long) en_now, (unsigned long long) ex_now,
                   (unsigned long long) dropped_at_entry, (unsigned long long) drop_now);
        }
    }

    // ── ④ 헤더 확정 기록(offset 0) ──
    LtHeader h{};
    std::memcpy(h.magic, "MOELT1\0\0", 8);
    h.layout_version  = 1;
    h.file_kind       = kind;
    h.record_bytes    = r.rec;
    h.engine_mode     = g_lt.engine_mode;
    h.record_count    = n_rows;
    h.payload_digest  = digest;
    h.qpc_frequency   = g_lt.qpc_freq;
    h.t0_qpc          = g_lt.t0_qpc;
    h.pid             = (uint32_t) GetCurrentProcessId();
    h.complete        = (recheck_ok && g_lt.complete.load(std::memory_order_acquire)) ? 1u : 0u;
    h.n_expert        = g_lt.n_expert;
    h.n_moe_layers    = g_lt.n_moe_layers;
    h.boot_id         = g_lt.boot_id;
    h.dropped_records = g_lt.dropped.load(std::memory_order_acquire);
    h.record_capacity = r.cap;

    if (ok) {
        LARGE_INTEGER zpos; zpos.QuadPart = 0;
        ok = SetFilePointerEx(f, zpos, nullptr, FILE_BEGIN) != 0;
    }
    if (ok) ok = WriteFile(f, &h, (DWORD) sizeof(h), &wr, nullptr) != 0 && wr == sizeof(h);
    if (ok) ok = FlushFileBuffers(f) != 0;
    CloseHandle(f);
    if (!ok) lt_log("cannot write sidecar path=%s err=%lu", path.c_str(), (unsigned long) GetLastError());
    return ok;
}

} // namespace

// ===========================================================================
// §4. 공개 API — 게이트
// ===========================================================================

bool ggml_moe_leadtime_enabled(void) {
    // ★hot-path fast-gate. 원자 로드 1회 + 비교 1회. 미해소(-1)는 false 이며 **해소하지 않는다**
    //   (해소는 getenv+VirtualAlloc 을 하므로 hot hook 에서 절대 일어나면 안 된다 — 헤더 ★).
    return g_lt.gate.load(std::memory_order_acquire) == LT_GATE_ON;
}

uint64_t ggml_moe_leadtime_now_qpc(void) {
    g_lt.qpc_calls.fetch_add(1, std::memory_order_relaxed);
    return lt_qpc_raw();
}

uint64_t ggml_moe_leadtime_boot_id(void) {
    // ★게이트 상태로 가리지 않는다 — `boot_id` 는 OFF/미해소에서 애초에 0 이고(해소가 ON 분기에서만
    //   채운다·reset 이 0 으로 되돌린다), flush 가 게이트를 CLOSED 로 닫은 **뒤에도** 사후 대조에
    //   쓸 수 있어야 한다(파일 헤더 ↔ 모듈 조회값 일치 검사).
    return g_lt.boot_id;
}

// ===========================================================================
// §5. 공개 API — arm(seal)
// ===========================================================================

void ggml_moe_leadtime_note_seal(uint64_t join_boot_id, uint32_t engine_mode,
                                 uint32_t n_expert, uint32_t n_moe_layers) {
    std::lock_guard<std::mutex> lk(g_lt.init_mu);
    lt_resolve_locked();
    if (g_lt.gate.load(std::memory_order_acquire) != LT_GATE_ON) return;

    // 멱등 — 이미 스탬프가 흐르기 시작했으면 조인 키를 바꾸지 않는다(앞 행과 뒤 행의 도메인이
    // 갈리면 판독기가 한 파일 안에서 두 런을 보게 된다).
    const bool started = g_lt.a.next.load(std::memory_order_acquire) != 0 ||
                         g_lt.b.next.load(std::memory_order_acquire) != 0;
    if (started) {
        lt_log("note_seal called again after stamping had begun - the join key is kept as is");
        return;
    }

    g_lt.engine_mode  = engine_mode;
    g_lt.n_expert     = n_expert;
    g_lt.n_moe_layers = n_moe_layers;

    if (join_boot_id != 0) {
        g_lt.boot_id        = join_boot_id;
        g_lt.boot_id_joined = true;
    } else if (!g_lt.boot_id_joined) {
        lt_log("no IO3 boot_id to join (IO3 is off or unarmed) - a self-generated boot_id is used "
               "and the IO3 join will fail-close in the reader (this is intended, not a silent pass)");
    }
}

// ===========================================================================
// §6. 공개 API — hot hook
// ===========================================================================

void ggml_moe_leadtime_stamp_a(int32_t site, uint64_t qpc, const char * tensor_name) {
    // ★publication 계약 — 게이트 확인보다 **먼저** 진입을 공개한다(순서가 계약).
    LtWriterScope ws;
    const int8_t g = g_lt.gate.load(std::memory_order_acquire);
    if (g != LT_GATE_ON) { lt_reject_stamp(g); return; }

    uint32_t  seq  = 0;
    uint8_t * slot = lt_reserve(g_lt.a, &seq);
    if (!slot) return;

    LtStampA r{};
    r.seq  = seq;
    r.site = (site == GGML_MOE_LEADTIME_SITE_A1 || site == GGML_MOE_LEADTIME_SITE_A2)
                 ? (uint8_t) site : (uint8_t) 0;   // 범위 밖 = 0(판독기가 배제) — 조용히 접지 않는다
    r.qpc  = qpc;

    if (!tensor_name) {
        r.flags = (uint8_t) (r.flags | GGML_MOE_LEADTIME_A_FLAG_NAME_NULL);
    } else {
        // 무경계 strlen 금지(상한 스캔). 원 길이를 그대로 싣고 48B 초과분만 절단한다 —
        // 판독기는 `name_len` 으로 "정확 복사인가"를 알 수 있다.
        size_t len = 0;
        while (len < LT_NAME_SCAN_MAX && tensor_name[len] != '\0') len++;
        r.name_len = (uint16_t) len;
        const size_t cp = len < (size_t) GGML_MOE_LEADTIME_NAME_BYTES
                              ? len : (size_t) GGML_MOE_LEADTIME_NAME_BYTES;
        std::memcpy(r.name, tensor_name, cp);
        if (len > (size_t) GGML_MOE_LEADTIME_NAME_BYTES) {
            r.flags = (uint8_t) (r.flags | GGML_MOE_LEADTIME_A_FLAG_NAME_TRUNCATED);
        }
    }

    std::memcpy(slot, &r, sizeof(r));
}

void ggml_moe_leadtime_stamp_b(uint64_t qpc, int32_t mlpos, int32_t part_idx,
                               int32_t n_ids, int32_t is_prefill) {
    // ★publication 계약 — stamp_a 와 동형(게이트 확인보다 먼저 진입 공개).
    LtWriterScope ws;
    const int8_t g = g_lt.gate.load(std::memory_order_acquire);
    if (g != LT_GATE_ON) { lt_reject_stamp(g); return; }

    uint32_t  seq  = 0;
    uint8_t * slot = lt_reserve(g_lt.b, &seq);
    if (!slot) return;

    LtStampB r{};
    r.seq      = seq;
    r.qpc      = qpc;
    // mlpos/part_idx 는 registry 값이라 실제로는 0..수백이다. 폭을 좁힌 대신 **접기 전에 포화**
    // 시켜 랩어라운드로 다른 층을 가리키는 값이 실리는 것을 막는다(판독기가 오귀속하면 그 층의
    // lead 가 통째로 거짓이 된다).
    r.mlpos    = (int16_t) (mlpos    < -32768 ? -32768 : (mlpos    > 32767 ? 32767 : mlpos));
    r.part_idx = (int16_t) (part_idx < -32768 ? -32768 : (part_idx > 32767 ? 32767 : part_idx));
    r.n_ids    = n_ids;
    r.flags    = is_prefill ? (uint8_t) GGML_MOE_LEADTIME_B_FLAG_PREFILL : (uint8_t) 0;

    std::memcpy(slot, &r, sizeof(r));
}

// ===========================================================================
// §7. 공개 API — flush
// ===========================================================================

void ggml_moe_leadtime_flush(void) {
    std::lock_guard<std::mutex> lk(g_lt.init_mu);

    if (g_lt.gate.load(std::memory_order_acquire) != LT_GATE_ON) {
        // ★무장되지 않은 런의 침묵 방지. finalize 는 hot path 가 아니므로 여기서만 env 를 진단
        //   목적으로 들여다본다. `=1` 이었는데 무장이 안 됐다 = seal(virtual) 에 도달하지 않은
        //   런이라는 뜻이고, 사용자의 실측 슬롯을 헛돌게 하는 대표적 실패라 반드시 크게 남긴다.
        if (!g_lt.resolved && lt_env("MOE_DIRECT_LEADTIME") == "1") {
            lt_log("MOE_DIRECT_LEADTIME=1 but the probe was never armed (the virtual seal path was "
                   "not reached) - no lead-time record exists for this run");
        }
        return;
    }

    if (g_lt.flushed) return;      // once 래치
    g_lt.flushed = true;

    // ── ★① 게이트 폐쇄(publication 계약) — **drain·record_count 확정보다 먼저** ────────────
    //   이 store 이후 `enabled()` 는 false 이므로 훅이 새 스탬프를 시작하지 않는다. 폐쇄 전에
    //   record_count 를 확정하면 그 사이 도착한 스탬프가 버퍼에만 남아 조용히 사라진다.
    g_lt.gate.store(LT_GATE_CLOSED, std::memory_order_seq_cst);

    // ── ★② in-flight writer drain — `w_enter == w_exit` 까지 ─────────────────────────────
    //   게이트가 이미 닫혀 있으므로 새 진입은 없고, 남은 것은 "게이트를 ON 으로 읽고 들어온"
    //   유한 개의 writer 뿐이다. 등식이 서면 (a)`next` 가 안정되고 (b)기록된 전 본문 memcpy 가
    //   이 스레드에 가시적이다(w_exit release ↔ 여기 acquire).
    bool     drained  = false;
    uint64_t inflight = 0;
    {
        // ★`w_exit` 를 먼저, `w_enter` 를 나중에 읽는다: 그 순서라면 두 값이 같게 보이는 순간은
        //   실제로 모든 진입이 완료된 순간뿐이다(그 사이에 새 진입이 있었다면 en > ex 로 보인다).
        const ULONGLONG t0 = GetTickCount64();
        for (;;) {
            const uint64_t ex = g_lt.w_exit.load(std::memory_order_acquire);
            const uint64_t en = g_lt.w_enter.load(std::memory_order_acquire);
            if (en == ex) { drained = true; inflight = 0; break; }
            inflight = en - ex;
            if (GetTickCount64() - t0 >= (ULONGLONG) LT_DRAIN_BUDGET_MS) break;
            Sleep(1);
        }
    }
    if (!drained) {
        const uint64_t en = g_lt.w_enter.load(std::memory_order_acquire);
        const uint64_t ex = g_lt.w_exit.load(std::memory_order_acquire);
        inflight = (en > ex) ? (en - ex) : 0;
        // ── ★③ 미완 예약이 남았다 = 커밋되지 않은 칸이 있을 수 있다 ⇒ 정직 강하 ──
        g_lt.dropped.fetch_add(inflight, std::memory_order_relaxed);
        g_lt.complete.store(false, std::memory_order_release);
        lt_log("in-flight lead writers did not drain within %u ms (%llu outstanding) - some reserved "
               "slots may be uncommitted, so the payload is NOT read and the arm is invalid (complete=0)",
               (unsigned) LT_DRAIN_BUDGET_MS, (unsigned long long) inflight);
    }

    // drop 이 있었으면 정직 강하(append 시점에 이미 내려가 있지만 flush 가 유일한 보고 시점이라
    // 여기서 다시 못박는다 — IO3 `io3_finalize_locked` 와 동형 계약).
    const uint64_t dropped = g_lt.dropped.load(std::memory_order_acquire);
    if (dropped > 0) g_lt.complete.store(false, std::memory_order_release);

    // ★record_count 는 **drain 뒤에** 확정한다 — drain 이 성립했으면 아래 `used()` 의 전 행이
    //   커밋 완료 행이다(불변식: torn 행은 complete=1 파일에 실릴 수 없다).
    const uint64_t used_a = g_lt.a.used();
    const uint64_t used_b = g_lt.b.used();
    const uint64_t want_a = g_lt.a.next.load(std::memory_order_acquire);
    const uint64_t want_b = g_lt.b.next.load(std::memory_order_acquire);

    lt_log("flush: A %llu/%llu rows · B %llu/%llu rows · dropped=%llu · complete=%d · "
           "drain=%s(inflight=%llu) (requested A=%llu B=%llu)",
           (unsigned long long) used_a, (unsigned long long) g_lt.a.cap,
           (unsigned long long) used_b, (unsigned long long) g_lt.b.cap,
           (unsigned long long) dropped,
           g_lt.complete.load(std::memory_order_acquire) ? 1 : 0,
           drained ? "ok" : "TIMEOUT", (unsigned long long) inflight,
           (unsigned long long) want_a, (unsigned long long) want_b);

    if (g_lt.out_prefix.empty()) {
        lt_log("MOE_DIRECT_LEADTIME_OUT is unset - nothing was written (counters only)");
        return;
    }

    // ★`dropped_at_entry` = 재검의 기준값. 이 시점 이후 늘어난 drop 은 "기록 창 안에서 발생한
    //   손실"이므로 헤더가 complete=0 으로 그것을 반영해야 한다(시임 ③ 이 그 창에 주입한다).
    const uint64_t dropped_at_entry = g_lt.dropped.load(std::memory_order_acquire);
    lt_write_file(g_lt.out_prefix + "_lead_a.lt", GGML_MOE_LEADTIME_KIND_A, g_lt.a,
                  drained, dropped_at_entry);
    lt_write_file(g_lt.out_prefix + "_lead_b.lt", GGML_MOE_LEADTIME_KIND_B, g_lt.b,
                  drained, dropped_at_entry);
}

// ===========================================================================
// §8. 테스트 전용 진입점
// ===========================================================================

void ggml_moe_leadtime_reset_for_test(void) {
    std::lock_guard<std::mutex> lk(g_lt.init_mu);
    g_lt.gate.store(LT_GATE_UNRESOLVED, std::memory_order_release);
    g_lt.resolved = false;
    g_lt.flushed  = false;
    g_lt.a.release();
    g_lt.b.release();
    g_lt.out_prefix.clear();
    g_lt.max_mb        = LT_DEFAULT_MAX_MB;
    g_lt.boot_id       = 0;
    g_lt.qpc_freq      = 0;
    g_lt.t0_qpc        = 0;
    g_lt.engine_mode   = 0;
    g_lt.n_expert      = 0;
    g_lt.n_moe_layers  = 0;
    g_lt.boot_id_joined = false;
    g_lt.dropped.store(0, std::memory_order_release);
    g_lt.complete.store(true, std::memory_order_release);
    g_lt.qpc_calls.store(0, std::memory_order_release);
    // publication 카운터 쌍도 시나리오마다 되돌린다(앞 시나리오의 in-flight 잔여가 다음 flush 의
    // drain 을 거짓 실패시키면 안 된다 — 시임으로 세운 정지 writer 포함).
    g_lt.w_enter.store(0, std::memory_order_release);
    g_lt.w_exit.store(0, std::memory_order_release);
    g_lt.test_late_stamp.store(0, std::memory_order_release);
    // resolve_count 는 **누적 유지**한다 — "해소가 몇 번 일어났나"는 시나리오 경계를 넘어
    // 확인해야 의미가 있는 계수다(리셋하면 반복 해소를 못 잡는다).
}

void ggml_moe_leadtime_test_counts(uint64_t * out_a, uint64_t * out_b,
                                   uint64_t * out_dropped, int32_t * out_complete,
                                   uint64_t * out_cap_a, uint64_t * out_cap_b) {
    if (out_a)        *out_a        = g_lt.a.used();
    if (out_b)        *out_b        = g_lt.b.used();
    if (out_dropped)  *out_dropped  = g_lt.dropped.load(std::memory_order_acquire);
    if (out_complete) *out_complete = g_lt.complete.load(std::memory_order_acquire) ? 1 : 0;
    if (out_cap_a)    *out_cap_a    = g_lt.a.cap;
    if (out_cap_b)    *out_cap_b    = g_lt.b.cap;
}

int ggml_moe_leadtime_test_a_bytes(uint64_t idx, void * out, uint64_t bytes) {
    if (!out || bytes != sizeof(LtStampA)) return -1;
    if (!g_lt.a.base || idx >= g_lt.a.used()) return -2;
    std::memcpy(out, g_lt.a.base + (size_t) idx * sizeof(LtStampA), sizeof(LtStampA));
    return 0;
}

int ggml_moe_leadtime_test_b_bytes(uint64_t idx, void * out, uint64_t bytes) {
    if (!out || bytes != sizeof(LtStampB)) return -1;
    if (!g_lt.b.base || idx >= g_lt.b.used()) return -2;
    std::memcpy(out, g_lt.b.base + (size_t) idx * sizeof(LtStampB), sizeof(LtStampB));
    return 0;
}

uint64_t ggml_moe_leadtime_test_resolve_count(void) {
    return g_lt.resolve_count.load(std::memory_order_acquire);
}

uint64_t ggml_moe_leadtime_test_qpc_calls(void) {
    return g_lt.qpc_calls.load(std::memory_order_acquire);
}

void ggml_moe_leadtime_test_publication(uint64_t * out_enter, uint64_t * out_exit,
                                        int32_t * out_gate) {
    if (out_enter) *out_enter = g_lt.w_enter.load(std::memory_order_acquire);
    if (out_exit)  *out_exit  = g_lt.w_exit.load(std::memory_order_acquire);
    if (out_gate)  *out_gate  = (int32_t) g_lt.gate.load(std::memory_order_acquire);
}

// ★시임 — `w_enter` 만 올려 "커밋되지 않은 예약이 남은 상태"를 결정론적으로 만든다.
//   실 스레드를 멈춰 세우지 않고도 flush 의 drain 실패 경로를 정확히 태울 수 있다(스레드 경합
//   기반 재현은 비결정적이라 회귀 관문으로 못 쓴다).
void ggml_moe_leadtime_test_stall_writer(int32_t n) {
    if (n > 0) g_lt.w_enter.fetch_add((uint64_t) n, std::memory_order_acq_rel);
}
void ggml_moe_leadtime_test_release_stalled_writer(int32_t n) {
    if (n > 0) g_lt.w_exit.fetch_add((uint64_t) n, std::memory_order_release);
}

// ★시임 ② — **진짜 in-flight writer**(교차 r2 R3-2). 프로덕션 `stamp_b` 와 같은 순서를 밟되
//   예약과 본문 memcpy 사이에서 `delay_ms` 머문다: 즉 "칸이 예약돼 공개됐고 그 칸이 아직
//   쓰이는 중"인 상태를 실제로 만든다(시임 ① 은 예약 자체를 하지 않아 이 상태를 못 만든다).
//   호출측이 별 스레드에서 부르고 본체가 그 사이에 flush 를 호출하면, flush 가 그 칸을
//   해시·기록하지 않는지(비원자 동시 읽기 회피)를 검사할 수 있다.
void ggml_moe_leadtime_test_stamp_b_delayed(uint64_t qpc, int32_t mlpos, int32_t part_idx,
                                            int32_t n_ids, int32_t delay_ms) {
    LtWriterScope ws;
    const int8_t g = g_lt.gate.load(std::memory_order_acquire);
    if (g != LT_GATE_ON) { lt_reject_stamp(g); return; }

    uint32_t  seq  = 0;
    uint8_t * slot = lt_reserve(g_lt.b, &seq);
    if (!slot) return;

    if (delay_ms > 0) Sleep((DWORD) delay_ms);   // ★본문 기록 지연 = in-flight 창

    LtStampB r{};
    r.seq      = seq;
    r.qpc      = qpc;
    r.mlpos    = (int16_t) (mlpos    < -32768 ? -32768 : (mlpos    > 32767 ? 32767 : mlpos));
    r.part_idx = (int16_t) (part_idx < -32768 ? -32768 : (part_idx > 32767 ? 32767 : part_idx));
    r.n_ids    = n_ids;
    r.flags    = 0;
    std::memcpy(slot, &r, sizeof(r));
}

// ★시임 ③ — flush 의 payload 기록 창 안에서 late stamp 를 n 회 주입하도록 무장한다(1회 소비).
void ggml_moe_leadtime_test_arm_late_stamp(int32_t n) {
    g_lt.test_late_stamp.store(n > 0 ? n : 0, std::memory_order_release);
}

#else  // !_WIN32 — 비Windows 스텁(Phase A·shadow 관례 동일: 심볼은 존재하고 전부 no-op)

bool     ggml_moe_leadtime_enabled(void)  { return false; }
uint64_t ggml_moe_leadtime_now_qpc(void)  { return 0; }
uint64_t ggml_moe_leadtime_boot_id(void)  { return 0; }
void     ggml_moe_leadtime_note_seal(uint64_t, uint32_t, uint32_t, uint32_t) {}
void     ggml_moe_leadtime_stamp_a(int32_t, uint64_t, const char *) {}
void     ggml_moe_leadtime_stamp_b(uint64_t, int32_t, int32_t, int32_t, int32_t) {}
void     ggml_moe_leadtime_flush(void) {}
void     ggml_moe_leadtime_reset_for_test(void) {}
void     ggml_moe_leadtime_test_counts(uint64_t * a, uint64_t * b, uint64_t * d, int32_t * c,
                                       uint64_t * ca, uint64_t * cb) {
    if (a) *a = 0; if (b) *b = 0; if (d) *d = 0; if (c) *c = 0; if (ca) *ca = 0; if (cb) *cb = 0;
}
int      ggml_moe_leadtime_test_a_bytes(uint64_t, void *, uint64_t) { return -1; }
int      ggml_moe_leadtime_test_b_bytes(uint64_t, void *, uint64_t) { return -1; }
uint64_t ggml_moe_leadtime_test_resolve_count(void) { return 0; }
uint64_t ggml_moe_leadtime_test_qpc_calls(void)     { return 0; }
void     ggml_moe_leadtime_test_publication(uint64_t * e, uint64_t * x, int32_t * g) {
    if (e) *e = 0; if (x) *x = 0; if (g) *g = -1;
}
void     ggml_moe_leadtime_test_stall_writer(int32_t) {}
void     ggml_moe_leadtime_test_release_stalled_writer(int32_t) {}
void     ggml_moe_leadtime_test_stamp_b_delayed(uint64_t, int32_t, int32_t, int32_t, int32_t) {}
void     ggml_moe_leadtime_test_arm_late_stamp(int32_t) {}

#endif // _WIN32
