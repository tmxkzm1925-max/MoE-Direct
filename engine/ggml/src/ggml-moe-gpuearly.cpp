// MoE GPU측 조기 전달 계측(v2) 모듈 본체 — 사양: bench/techdev/repack_space/PROBE_GPU_EARLY_SPEC_R1.md
// ★FROZEN v1.0
//
// 이 파일이 소유하는 것: env 게이트 · POD 링 버퍼(arm 시 전량 선할당) · 원자 인덱스 예약 ·
// 층별 서수(occurrence) 테이블 · boundary_seq 발번 · pending 회계 · 물리층 결산 카운터 ·
// 사이드카 1종 기록 · ms→ns 변환 규칙(사양 §3 유일 구현 지점).
// 이 파일이 소유하지 않는 것: CUDA 이벤트(생성·기록·질의 — `ggml-cuda/moe-gpuearly.cu`) ·
// 훅 지점(ggml-cuda.cu · topk-moe.cu · ggml-backend.cpp) · 판독(오프라인 python).
//
// ★동시성 모델: 커밋 지점은 graph 스케줄/실행 경로(단일 스케줄러 스레드)에서 불린다. 그래도 이
//   모듈은 lead-time 과 동형으로 **락 없이 다중 생산자 안전**하다 — 칸 예약은 `fetch_add` 1회이고
//   그 뒤의 memcpy 는 자기 칸에만 쓴다. 순서 검증은 판독기의 일이다(엔진은 원자료를 그대로 싣는다).
//
// ★lead-time 모듈과의 관계: **완전 독립**이다(심볼·상태·락 공유 0). 같은 QPC 도메인을 쓰는 것만이
//   유일한 접점이고, 그건 두 모듈이 각자 `QueryPerformanceCounter` 를 부르기 때문이지 결합이 아니다.
//
// Windows 전용 — 비Windows 는 파일 하단 스텁.

#include "ggml-moe-gpuearly.h"
#include "ggml-moe-gpuearly-internal.h"   // ★테스트 전용 진입점(공개 헤더 표면 밖)

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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
// §1. wire 구조체 (사양 §4 · 리틀엔디언·고정폭 — 판독기와의 유일한 계약)
// ===========================================================================

#pragma pack(push, 1)

// 128B 헤더. 가변 필드(record_count·dropped·complete·digest·excluded_*·pending_unconsumed·
// invalid)는 CREATE_NEW 직후 0 으로 선기록되고 finalize 에서 backpatch 된다(사양 §4 publication).
struct GeHeader {
    char     magic[8];                  // "MOEGPUC1"                      @0
    uint32_t version;                   // 1                               @8
    uint32_t record_size;               // 64                              @12
    uint32_t header_size;               // 128                             @16
    uint32_t pid;                       //                                 @20
    uint64_t qpc_frequency;             //                                 @24
    uint64_t boot_id;                   // ★IO3/lead-time 공통 조인 키      @32
    uint64_t t0_qpc;                    // arm 시각(같은 프로세스 단일 시계) @40
    uint64_t record_count;              // backpatch                       @48
    uint32_t dropped;                   // backpatch                       @56
    uint8_t  complete;                  // backpatch                       @60
    uint8_t  reserved_a[3];             // 0                               @61
    uint64_t digest;                    // FNV-1a64 over payload · backpatch @64
    uint64_t excluded_direct_count;     // backpatch                       @72
    uint64_t excluded_warmup_count;     // backpatch                       @80
    uint64_t pending_unconsumed_count;  // backpatch · 유효 조건 = 0        @88
    uint64_t invalid_count;             // backpatch · flags b0=0 행 수와 일치 @96
    uint8_t  reserved_b[24];            // 0                               @104..127
};

// 64B 레코드(문자열 없음).
struct GeRecord {
    uint64_t boundary_seq;        // @0   A1 과 직접 결속 키(launch 시 할당·단조)
    uint16_t layer;               // @8
    uint8_t  router_path;         // @10  0=argsort 1=fused
    uint8_t  flags;               // @11  b0=valid b1=excluded_direct b2=excluded_warmup
    uint32_t occurrence;          // @12
    uint64_t delta_tail_gpu_ns;   // @16
    uint64_t sync_return_qpc;     // @24
    uint64_t a1_qpc;              // @32
    uint64_t graph_uid;           // @40  진단 전용(조인 키 아님)
    uint8_t  reserved[16];        // @48..63
};

#pragma pack(pop)

static_assert(sizeof(GeHeader) == GGML_MOE_GPUEARLY_HDR_BYTES, "gpuearly header must be exactly 128B");
static_assert(sizeof(GeRecord) == GGML_MOE_GPUEARLY_REC_BYTES, "gpuearly record must be exactly 64B");

// ★오프셋 동결(사양 §4 "엔진 코드 offset static_assert 의무") — 폭만 맞고 자리가 밀리면
//   판독기가 조용히 오독한다. 이 assert 들이 그 조용함을 컴파일 에러로 바꾼다.
static_assert(offsetof(GeHeader, magic)                    ==   0, "GeHeader.magic @0");
static_assert(offsetof(GeHeader, version)                  ==   8, "GeHeader.version @8");
static_assert(offsetof(GeHeader, record_size)              ==  12, "GeHeader.record_size @12");
static_assert(offsetof(GeHeader, header_size)              ==  16, "GeHeader.header_size @16");
static_assert(offsetof(GeHeader, pid)                      ==  20, "GeHeader.pid @20");
static_assert(offsetof(GeHeader, qpc_frequency)            ==  24, "GeHeader.qpc_frequency @24");
static_assert(offsetof(GeHeader, boot_id)                  ==  32, "GeHeader.boot_id @32");
static_assert(offsetof(GeHeader, t0_qpc)                   ==  40, "GeHeader.t0_qpc @40");
static_assert(offsetof(GeHeader, record_count)             ==  48, "GeHeader.record_count @48");
static_assert(offsetof(GeHeader, dropped)                  ==  56, "GeHeader.dropped @56");
static_assert(offsetof(GeHeader, complete)                 ==  60, "GeHeader.complete @60");
static_assert(offsetof(GeHeader, reserved_a)               ==  61, "GeHeader.reserved_a @61");
static_assert(offsetof(GeHeader, digest)                   ==  64, "GeHeader.digest @64");
static_assert(offsetof(GeHeader, excluded_direct_count)    ==  72, "GeHeader.excluded_direct_count @72");
static_assert(offsetof(GeHeader, excluded_warmup_count)    ==  80, "GeHeader.excluded_warmup_count @80");
static_assert(offsetof(GeHeader, pending_unconsumed_count) ==  88, "GeHeader.pending_unconsumed_count @88");
static_assert(offsetof(GeHeader, invalid_count)            ==  96, "GeHeader.invalid_count @96");
static_assert(offsetof(GeHeader, reserved_b)               == 104, "GeHeader.reserved_b @104");

static_assert(offsetof(GeRecord, boundary_seq)      ==  0, "GeRecord.boundary_seq @0");
static_assert(offsetof(GeRecord, layer)             ==  8, "GeRecord.layer @8");
static_assert(offsetof(GeRecord, router_path)       == 10, "GeRecord.router_path @10");
static_assert(offsetof(GeRecord, flags)             == 11, "GeRecord.flags @11");
static_assert(offsetof(GeRecord, occurrence)        == 12, "GeRecord.occurrence @12");
static_assert(offsetof(GeRecord, delta_tail_gpu_ns) == 16, "GeRecord.delta_tail_gpu_ns @16");
static_assert(offsetof(GeRecord, sync_return_qpc)   == 24, "GeRecord.sync_return_qpc @24");
static_assert(offsetof(GeRecord, a1_qpc)            == 32, "GeRecord.a1_qpc @32");
static_assert(offsetof(GeRecord, graph_uid)         == 40, "GeRecord.graph_uid @40");
static_assert(offsetof(GeRecord, reserved)          == 48, "GeRecord.reserved @48");

// ===========================================================================
// §2. 상태
// ===========================================================================

constexpr uint64_t GE_FNV_SEED       = 0xcbf29ce484222325ull;
constexpr uint32_t GE_DEFAULT_MAX_MB = 64;      // 사양 §4 기본값
constexpr uint32_t GE_MAX_MAX_MB     = 4096;    // 상한(오타로 수 GB 를 잡는 것을 막는다)

// 게이트 상태 — lead-time 과 동형. CLOSED 는 OFF 와 **구분해야** 한다(OFF 는 애초에 무장 안 된
// 런이라 커밋이 손실이 아니고, CLOSED 는 무장된 런의 기록 창이 닫힌 뒤라 손실이다).
constexpr int8_t GE_GATE_UNRESOLVED = -1;
constexpr int8_t GE_GATE_OFF        =  0;
constexpr int8_t GE_GATE_ON         =  1;
constexpr int8_t GE_GATE_CLOSED     =  2;

// flush 의 in-flight writer drain 상한 — ★벽시계 예산(ms). 반복 횟수로 세면 Windows `Sleep(1)`
// 의 실제 수면(1~15ms)에 따라 상한이 흔들리고 로그 문면이 거짓이 된다(lead-time R3 M10 실사고).
constexpr uint32_t GE_DRAIN_BUDGET_MS = 200;

// 원자 인덱스 예약 링. `next` 는 상한을 넘어서도 계속 증가한다(넘친 예약은 drop 이며 그 수가
// "얼마나 모자랐나"를 그대로 알려준다 — 되감으면 그 정보가 사라진다).
struct GeRing {
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

struct GeState {
    // ── 게이트/식별 ──
    std::atomic<int8_t> gate{GE_GATE_UNRESOLVED};
    std::mutex          init_mu;        // 해소·arm·reset·flush 직렬화(hot path 는 안 잡는다)
    bool                resolved = false;
    std::string         out_prefix;     // 비면 카운터만
    uint32_t            max_mb   = GE_DEFAULT_MAX_MB;
    uint64_t            boot_id  = 0;
    uint64_t            qpc_freq = 0;
    uint64_t            t0_qpc   = 0;
    bool                boot_id_joined = false;

    // ── 버퍼 ──
    GeRing ring;

    // ── 물리층 결산 카운터(사양 §3 — 분류 우선순위 invalid > direct > warmup 로 **하나만** 올린다) ──
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> invalid_count{0};
    std::atomic<uint64_t> excl_direct{0};
    std::atomic<uint64_t> excl_warmup{0};
    std::atomic<uint64_t> pending_unconsumed{0};
    std::atomic<bool>     complete{true};

    // ── 진단 카운터(상호배타 검증 창구 — 사양 §7 원자⑤) ──
    std::atomic<uint64_t> anchor_argsort{0};
    std::atomic<uint64_t> anchor_fused{0};
    std::atomic<uint64_t> exclusivity_violations{0};
    // 행 수를 늘리지 않는 무결성 강하 계수(`note_fault`) — 헤더엔 `complete=0` 으로만 나간다.
    std::atomic<uint64_t> faults{0};

    // ── replay 상태기계(사양 §2) ──
    std::atomic<uint64_t> boundary_seq_next{1};   // ★1부터 — 0 은 "boundary 없음"(제외 표본) 표식
    std::atomic<int32_t>  pending_open{0};
    std::atomic<uint64_t> pending_seq{0};
    std::atomic<uint64_t> pending_anchors{0};

    // ── 층별 FIFO 서수(사양 §4 occurrence · run 전체 reset 없음) ──
    std::atomic<uint32_t> occ[GGML_MOE_GPUEARLY_MAX_LAYERS];

    // ── publication 계약(lead-time R3 승계): 진입/완료 카운터 쌍 ──
    std::atomic<uint64_t> w_enter{0};
    std::atomic<uint64_t> w_exit{0};

    // ── 소비자 시임(CUDA 백엔드가 등록) ──
    std::atomic<ggml_moe_gpuearly_consume_fn> consumer{nullptr};
    std::atomic<void *>                       consumer_user{nullptr};

    // ── 테스트 시임/진단 ──
    std::atomic<int32_t>  test_late_commit{0};
    std::atomic<uint64_t> resolve_count{0};
    std::atomic<uint64_t> qpc_calls{0};

    bool flushed = false;
};

GeState g_ge;

// ===========================================================================
// §3. 유틸
// ===========================================================================

inline uint64_t fnv1a64(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

void ge_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[moe-gpuearly] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

std::string ge_env(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// 내부 raw QPC — 공개 `now_qpc()` 의 진단 카운터를 건드리지 않는다(그 카운터는 "훅이 몇 번
// 시각을 집었나"를 재는 것이라 arm 의 t0 채취가 섞이면 OFF 계약 검사가 무의미해진다).
inline uint64_t ge_qpc_raw() {
    LARGE_INTEGER c;
    if (!QueryPerformanceCounter(&c)) return 0;
    return (uint64_t) c.QuadPart;
}

std::wstring ge_utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), &w[0], n);
    return w;
}

// ---------------------------------------------------------------------------
// 칸 예약. ★hot path 전량: fetch_add 1회 + 비교 1회(+ 호출측 memcpy 1회).
//   상한 초과·미무장(cap==0)은 drop 이고 latch 로 complete=0 이다(사양 §4 "초과=drop 계수 +
//   complete=0"). ★drop 경로에 로그를 넣지 않는다 — hot hook 내 문자열 포맷 금지.
// ---------------------------------------------------------------------------
inline uint8_t * ge_reserve(GeRing & r) {
    const uint64_t idx = r.next.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= r.cap || !r.base) {
        g_ge.dropped.fetch_add(1, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
        return nullptr;
    }
    return r.base + (size_t) idx * (size_t) r.rec;
}

// ---------------------------------------------------------------------------
// ★publication 스코프 — 커밋 1회의 "진입/완료"를 감싼다(lead-time R3 와 동형).
//   ①생성자가 게이트 확인 **이전에** w_enter 를 올린다(순서가 계약 — 그래야 "게이트는 ON 으로
//     읽었지만 아직 예약 전"인 스레드가 flush 의 drain 창에 반드시 포함된다).
//   ②소멸자가 **전 반환 경로에서** w_exit 를 release 로 올린다.
// ---------------------------------------------------------------------------
struct GeWriterScope {
    GeWriterScope()  { g_ge.w_enter.fetch_add(1, std::memory_order_acq_rel); }
    ~GeWriterScope() { g_ge.w_exit.fetch_add(1, std::memory_order_release); }
    GeWriterScope(const GeWriterScope &)             = delete;
    GeWriterScope & operator=(const GeWriterScope &) = delete;
};

// 게이트가 커밋을 받지 않는 상태일 때의 처분. CLOSED = 손실이므로 정직 처분한다.
inline void ge_reject_commit(int8_t gate_state) {
    if (gate_state == GE_GATE_CLOSED) {
        g_ge.dropped.fetch_add(1, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
    }
}

// pending 유실 처분(공통 본체). ★호출측이 서수 소비를 먼저 끝낸 뒤 부른다.
inline void ge_pending_lost_locked_free() {
    if (g_ge.pending_open.exchange(0, std::memory_order_acq_rel) != 0) {
        const uint64_t n = g_ge.pending_anchors.exchange(0, std::memory_order_acq_rel);
        g_ge.pending_unconsumed.fetch_add(1, std::memory_order_relaxed);
        g_ge.dropped.fetch_add(n, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// env 해소 + 전량 선할당. ★`init_mu` 를 **이미 쥔 상태**에서만 부른다.
// ---------------------------------------------------------------------------
void ge_resolve_locked() {
    if (g_ge.resolved) return;
    g_ge.resolved = true;
    g_ge.resolve_count.fetch_add(1, std::memory_order_relaxed);

    // ── 게이트: exact "1" 만 on. 미설정/빈값 = 조용한 off. 그 외 = off + 경고 1회 ──
    const std::string raw = ge_env("MOE_DIRECT_GPUEARLY");
    bool on = false;
    if (raw == "1") {
        on = true;
    } else if (!raw.empty()) {
        ge_log("MOE_DIRECT_GPUEARLY=\"%s\" is not the only accepted value (\"1\") - the GPU-early "
               "probe stays OFF", raw.c_str());
    }
    if (!on) {
        g_ge.gate.store(GE_GATE_OFF, std::memory_order_release);
        return;   // ★OFF — 어떤 버퍼도 할당하지 않는다(흔적 0 계약)
    }

    // ── 버퍼 상한 ──
    g_ge.max_mb = GE_DEFAULT_MAX_MB;
    {
        const std::string s = ge_env("MOE_DIRECT_GPUEARLY_MAX_MB");
        if (!s.empty()) {
            char * end = nullptr;
            const long v = std::strtol(s.c_str(), &end, 10);
            if (end && *end == '\0' && v > 0 && v <= (long) GE_MAX_MAX_MB) {
                g_ge.max_mb = (uint32_t) v;
            } else {
                ge_log("MOE_DIRECT_GPUEARLY_MAX_MB=\"%s\" is invalid (1..%u) - using the default %u",
                       s.c_str(), (unsigned) GE_MAX_MAX_MB, (unsigned) GE_DEFAULT_MAX_MB);
            }
        }
    }

    g_ge.out_prefix = ge_env("MOE_DIRECT_GPUEARLY_OUT");

    g_ge.qpc_freq = 0;
    {
        LARGE_INTEGER f;
        if (QueryPerformanceFrequency(&f)) g_ge.qpc_freq = (uint64_t) f.QuadPart;
    }
    g_ge.t0_qpc = ge_qpc_raw();

    // 조인 키 기본값 = 자체 생성(note_seal 이 IO3 값을 주면 그것으로 덮는다).
    {
        const uint32_t pid = (uint32_t) GetCurrentProcessId();
        uint64_t h = GE_FNV_SEED;
        h = fnv1a64(h, &pid, sizeof(pid));
        h = fnv1a64(h, &g_ge.t0_qpc, sizeof(g_ge.t0_qpc));
        g_ge.boot_id = h;
        g_ge.boot_id_joined = false;
    }

    // 층별 서수 테이블 초기화(정적 저장기간이라 이미 0 이지만, reset 후 재무장 경로에서도 같은
    // 상태여야 하므로 명시적으로 되돌린다).
    for (uint32_t i = 0; i < GGML_MOE_GPUEARLY_MAX_LAYERS; i++) {
        g_ge.occ[i].store(0, std::memory_order_relaxed);
    }

    // ── ★전량 선할당(사양 §4 publication 의 전제 — hot hook 에서 할당하지 않기 위함) ──
    const uint64_t cap_bytes = (uint64_t) g_ge.max_mb * 1024ull * 1024ull;
    g_ge.ring.rec = (uint32_t) sizeof(GeRecord);
    const uint64_t cap_rows = cap_bytes / g_ge.ring.rec;

    void * p = (cap_rows > 0) ? VirtualAlloc(nullptr, (size_t) (cap_rows * g_ge.ring.rec),
                                             MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) : nullptr;
    if (!p) {
        // ★조용히 OFF 로 접지 않는다. 게이트는 켜 둔 채 cap=0 으로 두면 모든 커밋이 drop 되고
        //   flush 가 `complete=0` 인 0행 사이드카를 남긴다 ⇒ 판독기가 fail-close 한다.
        //   OFF 로 접으면 "계측 안 한 런"과 구분되지 않아 조용한 무데이터가 된다.
        g_ge.ring.base = nullptr;
        g_ge.ring.cap  = 0;
        g_ge.complete.store(false, std::memory_order_release);
        g_ge.gate.store(GE_GATE_ON, std::memory_order_release);
        ge_log("cannot pre-allocate the GPU-early buffer (%u MiB, err=%lu) - the arm is invalid "
               "(complete=0) and every record will be dropped",
               (unsigned) g_ge.max_mb, (unsigned long) GetLastError());
        return;
    }

    g_ge.ring.base = (uint8_t *) p;
    g_ge.ring.cap  = cap_rows;
    g_ge.gate.store(GE_GATE_ON, std::memory_order_release);

    ge_log("armed: max_mb=%u -> %llu rows x %uB · qpc_freq=%llu · out='%s'",
           (unsigned) g_ge.max_mb, (unsigned long long) cap_rows, (unsigned) g_ge.ring.rec,
           (unsigned long long) g_ge.qpc_freq, g_ge.out_prefix.c_str());
    if (g_ge.out_prefix.empty()) {
        // ★`complete` 를 내리지 않는다 — "기록한 레코드에 결손이 있는가"와 "쓸 곳이 있는가"는
        //   다른 사실이다. 파일이 없으면 판독기는 그 부재로 이미 fail-close 한다
        //   (lead-time §E 3번·shadow 선례와 같은 판단).
        ge_log("MOE_DIRECT_GPUEARLY_OUT is unset - counters only, no sidecar will be written");
    }
}

// ---------------------------------------------------------------------------
// 사이드카 기록 — ★2단계(헤더 자리 예약 → payload → 최종 재검 → 헤더 확정).
//   lead-time R3 의 `lt_write_file` 과 동형이며, 그 라운드의 교차 r2 R3-1/R3-2 폐합분을
//   그대로 승계한다: ①헤더를 마지막에 확정해 payload 기록 창 안의 late 커밋을 반영한다
//   ②drain 실패면 payload 를 **아예 읽지 않는다**(live writer 와의 비원자 동시 읽기 회피).
// ---------------------------------------------------------------------------
bool ge_write_file(const std::string & path, bool drained, uint64_t dropped_at_entry) {
    const uint64_t n_rows  = drained ? g_ge.ring.used() : 0;
    const size_t   payload = (size_t) n_rows * (size_t) g_ge.ring.rec;

    const std::wstring wp = ge_utf8_to_wide(path);
    HANDLE f = CreateFileW(wp.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        ge_log("cannot create sidecar (CREATE_NEW) path=%s err=%lu", path.c_str(),
               (unsigned long) GetLastError());
        return false;
    }

    // ── ① 헤더 선기록 — 사양 §4 문면 그대로 **"가변 필드만 0"** ──
    //   ★전체 0 으로 쓰면 기록 도중 죽은 파일이 magic 도 없는 쓰레기로 남아 판독기가 "손상"과
    //   "미완결"을 구분하지 못한다. 고정 필드(magic·version·폭·pid·시계·조인 키)를 먼저 박아
    //   두면 그 파일은 **자기를 설명할 수 있는 미완결 파일**이 되고, backpatch 되지 않은
    //   가변 필드가 0 인 것 자체가 "완결되지 않았다"는 정직한 표시가 된다(교차 r1 권고).
    GeHeader pre{};
    std::memcpy(pre.magic, "MOEGPUC1", 8);
    pre.version       = GGML_MOE_GPUEARLY_VERSION;
    pre.record_size   = (uint32_t) sizeof(GeRecord);
    pre.header_size   = (uint32_t) sizeof(GeHeader);
    pre.pid           = (uint32_t) GetCurrentProcessId();
    pre.qpc_frequency = g_ge.qpc_freq;
    pre.boot_id       = g_ge.boot_id;
    pre.t0_qpc        = g_ge.t0_qpc;
    // record_count·dropped·complete·digest·excluded_*·pending_unconsumed·invalid = 0(가변 필드)
    DWORD wr = 0;
    bool ok = WriteFile(f, &pre, (DWORD) sizeof(pre), &wr, nullptr) != 0 && wr == sizeof(pre);

    // ── ② payload append-only + digest(같은 바이트를 한 번만 읽는다) ──
    uint64_t digest = GE_FNV_SEED;
    if (ok && payload) {
        size_t off = 0;
        while (off < payload) {
            const DWORD chunk = (DWORD) std::min<size_t>(payload - off, (size_t) 32 * 1024 * 1024);
            digest = fnv1a64(digest, g_ge.ring.base + off, (size_t) chunk);
            if (!WriteFile(f, g_ge.ring.base + off, chunk, &wr, nullptr) || wr != chunk) { ok = false; break; }
            off += chunk;
        }
    }

    // ★시임(테스트 전용) — **payload 기록 창 안**의 late 커밋 주입. 그 창은 실 경로에서 외부가
    //   건드릴 수 없으므로, 최종 재검이 정말로 헤더를 내리는지 확인하는 유일한 수단이다.
    {
        const int32_t n_late = g_ge.test_late_commit.exchange(0, std::memory_order_acq_rel);
        for (int32_t i = 0; i < n_late; i++) {
            ggml_moe_gpuearly_commit(0, 0, GGML_MOE_GPUEARLY_ROUTER_ARGSORT,
                                     GGML_MOE_GPUEARLY_FLAG_VALID, 0, 0, 0, 0, 0);
        }
    }

    // ── ③ ★최종 재검 — payload 기록이 끝난 **뒤**의 상태로 판정한다 ──
    const bool     gate_closed = g_ge.gate.load(std::memory_order_acquire) == GE_GATE_CLOSED;
    const uint64_t en_now      = g_ge.w_enter.load(std::memory_order_acquire);
    const uint64_t ex_now      = g_ge.w_exit.load(std::memory_order_acquire);
    const uint64_t drop_now    = g_ge.dropped.load(std::memory_order_acquire);
    const bool     recheck_ok  = drained && gate_closed && (en_now == ex_now) &&
                                 (drop_now == dropped_at_entry);
    if (!recheck_ok) {
        g_ge.complete.store(false, std::memory_order_release);
        if (drained) {
            ge_log("final recheck failed while writing %s (gate_closed=%d enter=%llu exit=%llu "
                   "dropped %llu->%llu) - header is written with complete=0",
                   path.c_str(), gate_closed ? 1 : 0,
                   (unsigned long long) en_now, (unsigned long long) ex_now,
                   (unsigned long long) dropped_at_entry, (unsigned long long) drop_now);
        }
    }

    // ── ④ 헤더 확정 기록(offset 0 · backpatch) ──
    const uint64_t dropped_u64 = g_ge.dropped.load(std::memory_order_acquire);

    GeHeader h{};
    std::memcpy(h.magic, "MOEGPUC1", 8);
    h.version                  = GGML_MOE_GPUEARLY_VERSION;
    h.record_size              = (uint32_t) sizeof(GeRecord);
    h.header_size              = (uint32_t) sizeof(GeHeader);
    h.pid                      = (uint32_t) GetCurrentProcessId();
    h.qpc_frequency            = g_ge.qpc_freq;
    h.boot_id                  = g_ge.boot_id;
    h.t0_qpc                   = g_ge.t0_qpc;
    h.record_count             = n_rows;
    // ★`dropped` 는 wire 상 u32 다(사양 §4 off 56). u32 를 넘는 결손은 포화시키되 그 사실을
    //   `complete=0` 이 이미 전달하므로 정보가 사라지지 않는다(포화 자체가 이상 신호다).
    h.dropped                  = (uint32_t) std::min<uint64_t>(dropped_u64, 0xFFFFFFFFull);
    h.complete                 = (recheck_ok && g_ge.complete.load(std::memory_order_acquire)) ? 1u : 0u;
    h.digest                   = digest;
    h.excluded_direct_count    = g_ge.excl_direct.load(std::memory_order_acquire);
    h.excluded_warmup_count    = g_ge.excl_warmup.load(std::memory_order_acquire);
    h.pending_unconsumed_count = g_ge.pending_unconsumed.load(std::memory_order_acquire);
    h.invalid_count            = g_ge.invalid_count.load(std::memory_order_acquire);

    if (ok) {
        LARGE_INTEGER zpos; zpos.QuadPart = 0;
        ok = SetFilePointerEx(f, zpos, nullptr, FILE_BEGIN) != 0;
    }
    if (ok) ok = WriteFile(f, &h, (DWORD) sizeof(h), &wr, nullptr) != 0 && wr == sizeof(h);
    if (ok) ok = FlushFileBuffers(f) != 0;
    CloseHandle(f);
    if (!ok) ge_log("cannot write sidecar path=%s err=%lu", path.c_str(), (unsigned long) GetLastError());
    return ok;
}

} // namespace

// ===========================================================================
// §4. 공개 API — 게이트
// ===========================================================================

bool ggml_moe_gpuearly_enabled(void) {
    // ★hot-path fast-gate. 원자 로드 1회 + 비교 1회. 미해소(-1)는 false 이며 **해소하지 않는다**.
    return g_ge.gate.load(std::memory_order_acquire) == GE_GATE_ON;
}

uint64_t ggml_moe_gpuearly_now_qpc(void) {
    g_ge.qpc_calls.fetch_add(1, std::memory_order_relaxed);
    return ge_qpc_raw();
}

uint64_t ggml_moe_gpuearly_boot_id(void) {
    if (g_ge.gate.load(std::memory_order_acquire) == GE_GATE_UNRESOLVED) return 0;
    if (g_ge.gate.load(std::memory_order_acquire) == GE_GATE_OFF)        return 0;
    return g_ge.boot_id;
}

// ===========================================================================
// §5. 공개 API — arm(seal)
// ===========================================================================

void ggml_moe_gpuearly_note_seal(uint64_t join_boot_id) {
    std::lock_guard<std::mutex> lk(g_ge.init_mu);
    ge_resolve_locked();
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;

    // 멱등 — 이미 커밋이 흐르기 시작했으면 조인 키를 바꾸지 않는다(앞 행과 뒤 행의 도메인이
    // 갈리면 판독기가 한 파일 안에서 두 런을 보게 된다).
    if (g_ge.ring.next.load(std::memory_order_acquire) != 0) {
        ge_log("note_seal called again after records had begun - the join key is kept as is");
        return;
    }

    if (join_boot_id != 0) {
        g_ge.boot_id        = join_boot_id;
        g_ge.boot_id_joined = true;
    } else if (!g_ge.boot_id_joined) {
        ge_log("no IO3 boot_id to join (IO3 is off or unarmed) - a self-generated boot_id is used "
               "and the IO3/lead-time join will fail-close in the reader (intended, not a silent pass)");
    }
}

// ===========================================================================
// §6. 공개 API — hot hook
// ===========================================================================

uint32_t ggml_moe_gpuearly_next_occurrence(uint32_t layer) {
    // ★OFF 무흔적 계약 — 호출측이 이미 `enabled()` 를 확인하지만, 게이트가 ON 이 아닐 때 서수
    //   테이블을 건드리지 않는 것을 **여기서도** 잠근다(방어가 아니라 계약의 총체성: OFF 런의
    //   어떤 경로도 이 모듈의 상태를 바꾸지 못한다는 것이 selftest 로 증명 가능해야 한다).
    //   ★CLOSED 에서 조용히 반환해도 손실이 사라지지 않는다 — 뒤따르는 `commit()` 이
    //   `ge_reject_commit(CLOSED)` 로 그 한 건을 정직 계상한다(이중 계상 방지가 이 배치의 이유).
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return UINT32_MAX;
    if (layer >= GGML_MOE_GPUEARLY_MAX_LAYERS) {
        // 서수를 발급할 수 없는 층 = 장부가 성립하지 않는다. 조용히 접지 않고 정직 처분한다.
        g_ge.dropped.fetch_add(1, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
        return UINT32_MAX;
    }
    const uint32_t v = g_ge.occ[layer].fetch_add(1, std::memory_order_acq_rel);
    if (v == UINT32_MAX) {
        // 사양 §4 "u32 초과=fail-close(도달 불가 설계)".
        g_ge.dropped.fetch_add(1, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
    }
    return v;
}

uint64_t ggml_moe_gpuearly_pending_begin(uint64_t n_anchors) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return 0;   // OFF 무흔적
    // 이미 열려 있으면 그 pending 은 소비되지 못한 것이다 — 사양 §2 fail-close.
    // (정상 경로에서는 호출측이 `abandon_pending()` 으로 먼저 처분하므로 여기 걸리지 않는다.
    //  이 자리는 그 규율이 깨졌을 때의 **정직한 안전망**이다.)
    ge_pending_lost_locked_free();

    const uint64_t seq = g_ge.boundary_seq_next.fetch_add(1, std::memory_order_acq_rel);
    g_ge.pending_seq.store(seq, std::memory_order_release);
    g_ge.pending_anchors.store(n_anchors, std::memory_order_release);
    g_ge.pending_open.store(1, std::memory_order_release);
    return seq;
}

void ggml_moe_gpuearly_pending_consumed(uint64_t n_consumed) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    if (g_ge.pending_open.load(std::memory_order_acquire) == 0) return;
    const uint64_t left = g_ge.pending_anchors.load(std::memory_order_acquire);
    const uint64_t rem  = (n_consumed >= left) ? 0 : (left - n_consumed);
    g_ge.pending_anchors.store(rem, std::memory_order_release);
    if (rem == 0) {
        // 전건 소비 = 정상 종료. 남은 수를 0 으로 만든 뒤 닫으므로, 이후 유실 처분이 일어나도
        // `dropped` 에 **이미 커밋된 앵커가 섞이지 않는다**(과다 계상 차단).
        g_ge.pending_open.store(0, std::memory_order_release);
    }
}

void ggml_moe_gpuearly_pending_end(void) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    g_ge.pending_open.store(0, std::memory_order_release);
    g_ge.pending_anchors.store(0, std::memory_order_release);
}

void ggml_moe_gpuearly_abandon_pending(void) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    ge_pending_lost_locked_free();
}

void ggml_moe_gpuearly_commit(uint64_t boundary_seq,
                              uint32_t layer,
                              uint32_t router_path,
                              uint32_t flags,
                              uint32_t occurrence,
                              uint64_t delta_tail_gpu_ns,
                              uint64_t sync_return_qpc,
                              uint64_t a1_qpc,
                              uint64_t graph_uid) {
    // ★publication 계약 — 게이트 확인보다 **먼저** 진입을 공개한다(순서가 계약).
    GeWriterScope ws;
    const int8_t g = g_ge.gate.load(std::memory_order_acquire);
    if (g != GE_GATE_ON) { ge_reject_commit(g); return; }

    uint8_t * slot = ge_reserve(g_ge.ring);
    if (!slot) return;

    GeRecord r{};
    r.boundary_seq      = boundary_seq;
    r.layer             = (uint16_t) (layer > 0xFFFFu ? 0xFFFFu : layer);
    r.router_path       = (uint8_t)  (router_path == GGML_MOE_GPUEARLY_ROUTER_FUSED
                                          ? GGML_MOE_GPUEARLY_ROUTER_FUSED
                                          : GGML_MOE_GPUEARLY_ROUTER_ARGSORT);
    r.flags             = (uint8_t)  (flags & 0x07u);
    r.occurrence        = occurrence;
    r.delta_tail_gpu_ns = delta_tail_gpu_ns;
    r.sync_return_qpc   = sync_return_qpc;
    r.a1_qpc            = a1_qpc;
    r.graph_uid         = graph_uid;

    std::memcpy(slot, &r, sizeof(r));

    // ★분류 카운터 — 사양 §3 물리층의 **우선순위 그대로** 정확히 하나만 올린다
    //   (invalid > direct > warmup > eligible). 비트별로 세면 두 비트가 선 행이 이중 계상되어
    //   `eligible = record_count − invalid − direct − warmup` 파생식이 깨진다.
    if ((r.flags & GGML_MOE_GPUEARLY_FLAG_VALID) == 0) {
        g_ge.invalid_count.fetch_add(1, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
    } else if (r.flags & GGML_MOE_GPUEARLY_FLAG_EXCLUDED_DIRECT) {
        g_ge.excl_direct.fetch_add(1, std::memory_order_relaxed);
    } else if (r.flags & GGML_MOE_GPUEARLY_FLAG_EXCLUDED_WARMUP) {
        g_ge.excl_warmup.fetch_add(1, std::memory_order_relaxed);
    }
}

int ggml_moe_gpuearly_ms_to_ns(float elapsed_ms, uint64_t * out_ns) {
    if (!out_ns) return -1;
    const double ms = (double) elapsed_ms;
    // ★`!(ms >= 0.0)` 이 NaN 도 함께 거른다(NaN 은 모든 비교가 거짓이다).
    if (!(ms >= 0.0) || !std::isfinite(ms)) return -1;
    const double ns = ms * 1e6;
    // 2^63 = 9223372036854775808.0 — 이 이상은 u64 ns 로 표현할 수 없다(사양 §3).
    if (!(ns < 9223372036854775808.0)) return -1;
    const long long v = std::llround(ns);
    if (v < 0) return -1;
    *out_ns = (uint64_t) v;
    return 0;
}

void ggml_moe_gpuearly_note_exclusivity_violation(void) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    g_ge.exclusivity_violations.fetch_add(1, std::memory_order_relaxed);
    g_ge.complete.store(false, std::memory_order_release);
}

void ggml_moe_gpuearly_note_anchor(uint32_t router_path) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    if (router_path == GGML_MOE_GPUEARLY_ROUTER_FUSED) {
        g_ge.anchor_fused.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_ge.anchor_argsort.fetch_add(1, std::memory_order_relaxed);
    }
}

void ggml_moe_gpuearly_note_drop(uint64_t n) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    if (n == 0) return;
    g_ge.dropped.fetch_add(n, std::memory_order_relaxed);
    g_ge.complete.store(false, std::memory_order_release);
}

void ggml_moe_gpuearly_note_fault(void) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;     // OFF 무흔적
    g_ge.faults.fetch_add(1, std::memory_order_relaxed);
    g_ge.complete.store(false, std::memory_order_release);
}

int ggml_moe_gpuearly_parse_layer(const char * name, const char * prefix, uint32_t * out_layer) {
    if (!name || !prefix || !out_layer) return -1;
    size_t i = 0;
    for (; prefix[i]; i++) {
        if (name[i] != prefix[i]) return -1;
    }
    if (name[i] < '0' || name[i] > '9') return -1;
    uint64_t v = 0;
    size_t   d = 0;
    while (name[i] >= '0' && name[i] <= '9') {
        v = v * 10u + (uint64_t) (name[i] - '0');
        if (v >= GGML_MOE_GPUEARLY_MAX_LAYERS) return -1;
        i++;
        if (++d > 6) return -1;
    }
    // ★정확 매칭 — 접미사가 붙은 이름(` (reshaped)` 등)은 **대상이 아니다**. 이 한 줄이
    //   "같은 층의 활성화 복사"를 ids 복사로 오인하는 경로를 구조적으로 막는다(교차 r1 차단①).
    if (name[i] != '\0') return -1;
    *out_layer = (uint32_t) v;
    return 0;
}

// ===========================================================================
// §7. 공개 API — 소비자 시임
// ===========================================================================

void ggml_moe_gpuearly_set_consumer(ggml_moe_gpuearly_consume_fn fn, void * user) {
    g_ge.consumer_user.store(user, std::memory_order_release);
    g_ge.consumer.store(fn, std::memory_order_release);
}

void ggml_moe_gpuearly_consume(void * synced_backend, const char * input_name,
                               uint64_t sync_return_qpc, uint64_t a1_qpc) {
    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) return;
    // pending 이 없으면 간접 호출조차 하지 않는다(같은 split 의 다른 입력 복사에서 무동작).
    if (g_ge.pending_open.load(std::memory_order_acquire) == 0) return;
    // ★대상 결속 1단계(교차 r1 차단①) — ids 이름이 아니면 여기서 끝난다. 활성화 복사가
    //   pending 을 닫아 `a1_qpc` 를 이르게 잡는 경로를 ggml-base 층에서 먼저 차단한다.
    //   나머지 2조건(층 일치·backend 동일)은 graph 메타데이터를 쥔 CUDA 소비자가 본다.
    uint32_t layer_probe = 0;
    if (ggml_moe_gpuearly_parse_layer(input_name, "ffn_moe_topk-", &layer_probe) != 0) return;
    const ggml_moe_gpuearly_consume_fn fn = g_ge.consumer.load(std::memory_order_acquire);
    if (!fn) return;
    fn(g_ge.consumer_user.load(std::memory_order_acquire), synced_backend, input_name,
       sync_return_qpc, a1_qpc);
}

// ===========================================================================
// §8. 공개 API — flush
// ===========================================================================

void ggml_moe_gpuearly_flush(void) {
    std::lock_guard<std::mutex> lk(g_ge.init_mu);

    if (g_ge.gate.load(std::memory_order_acquire) != GE_GATE_ON) {
        // ★무장되지 않은 런의 침묵 방지(finalize 는 hot path 가 아니므로 여기서만 env 를 본다).
        if (!g_ge.resolved && ge_env("MOE_DIRECT_GPUEARLY") == "1") {
            ge_log("MOE_DIRECT_GPUEARLY=1 but the probe was never armed (the virtual seal path was "
                   "not reached) - no GPU-early record exists for this run");
        }
        return;
    }

    if (g_ge.flushed) return;      // once 래치
    g_ge.flushed = true;

    // ── ★① 게이트 폐쇄(publication 계약) — drain·record_count 확정보다 먼저 ──
    g_ge.gate.store(GE_GATE_CLOSED, std::memory_order_seq_cst);

    // ── ★② 남은 pending 처분(사양 §3 "pending_unconsumed_count = 0 유효 조건") ──
    //   정상 종료면 마지막 launch 도 그 split 경계에서 소비됐으므로 여기 걸리지 않는다.
    //   걸렸다면 관측이 한 launch 분 비었다는 뜻이라 그 사실을 헤더로 내보낸다.
    if (g_ge.pending_open.load(std::memory_order_acquire) != 0) {
        const uint64_t n = g_ge.pending_anchors.load(std::memory_order_acquire);
        ge_log("a pending graph launch was never consumed at flush time (seq=%llu anchors=%llu) - "
               "pending_unconsumed_count is raised and the run is fail-closed",
               (unsigned long long) g_ge.pending_seq.load(std::memory_order_acquire),
               (unsigned long long) n);
        ge_pending_lost_locked_free();
    }

    // ── ★③ in-flight writer drain — `w_enter == w_exit` 까지 ──
    bool     drained  = false;
    uint64_t inflight = 0;
    {
        // ★`w_exit` 를 먼저, `w_enter` 를 나중에 읽는다: 그 순서라면 두 값이 같게 보이는 순간은
        //   실제로 모든 진입이 완료된 순간뿐이다.
        const ULONGLONG t0 = GetTickCount64();
        for (;;) {
            const uint64_t ex = g_ge.w_exit.load(std::memory_order_acquire);
            const uint64_t en = g_ge.w_enter.load(std::memory_order_acquire);
            if (en == ex) { drained = true; inflight = 0; break; }
            inflight = en - ex;
            if (GetTickCount64() - t0 >= (ULONGLONG) GE_DRAIN_BUDGET_MS) break;
            Sleep(1);
        }
    }
    if (!drained) {
        const uint64_t en = g_ge.w_enter.load(std::memory_order_acquire);
        const uint64_t ex = g_ge.w_exit.load(std::memory_order_acquire);
        inflight = (en > ex) ? (en - ex) : 0;
        g_ge.dropped.fetch_add(inflight, std::memory_order_relaxed);
        g_ge.complete.store(false, std::memory_order_release);
        ge_log("in-flight GPU-early writers did not drain within %u ms (%llu outstanding) - some "
               "reserved slots may be uncommitted, so the payload is NOT read and the arm is "
               "invalid (complete=0)", (unsigned) GE_DRAIN_BUDGET_MS, (unsigned long long) inflight);
    }

    const uint64_t dropped = g_ge.dropped.load(std::memory_order_acquire);
    if (dropped > 0) g_ge.complete.store(false, std::memory_order_release);

    const uint64_t used = g_ge.ring.used();
    const uint64_t want = g_ge.ring.next.load(std::memory_order_acquire);

    ge_log("flush: %llu/%llu rows · dropped=%llu · invalid=%llu · direct=%llu · warmup=%llu · "
           "pending_unconsumed=%llu · anchors(argsort=%llu fused=%llu excl_viol=%llu) · "
           "complete=%d · drain=%s(inflight=%llu) (requested=%llu)",
           (unsigned long long) used, (unsigned long long) g_ge.ring.cap,
           (unsigned long long) dropped,
           (unsigned long long) g_ge.invalid_count.load(std::memory_order_acquire),
           (unsigned long long) g_ge.excl_direct.load(std::memory_order_acquire),
           (unsigned long long) g_ge.excl_warmup.load(std::memory_order_acquire),
           (unsigned long long) g_ge.pending_unconsumed.load(std::memory_order_acquire),
           (unsigned long long) g_ge.anchor_argsort.load(std::memory_order_acquire),
           (unsigned long long) g_ge.anchor_fused.load(std::memory_order_acquire),
           (unsigned long long) g_ge.exclusivity_violations.load(std::memory_order_acquire),
           g_ge.complete.load(std::memory_order_acquire) ? 1 : 0,
           drained ? "ok" : "TIMEOUT", (unsigned long long) inflight,
           (unsigned long long) want);

    if (g_ge.out_prefix.empty()) {
        ge_log("MOE_DIRECT_GPUEARLY_OUT is unset - nothing was written (counters only)");
        return;
    }

    const uint64_t dropped_at_entry = g_ge.dropped.load(std::memory_order_acquire);
    ge_write_file(g_ge.out_prefix + "_gpuearly_c.gc", drained, dropped_at_entry);
}

// ===========================================================================
// §9. 테스트 전용 진입점
// ===========================================================================

void ggml_moe_gpuearly_reset_for_test(void) {
    std::lock_guard<std::mutex> lk(g_ge.init_mu);
    g_ge.gate.store(GE_GATE_UNRESOLVED, std::memory_order_release);
    g_ge.resolved = false;
    g_ge.flushed  = false;
    g_ge.ring.release();
    g_ge.out_prefix.clear();
    g_ge.max_mb         = GE_DEFAULT_MAX_MB;
    g_ge.boot_id        = 0;
    g_ge.qpc_freq       = 0;
    g_ge.t0_qpc         = 0;
    g_ge.boot_id_joined = false;
    g_ge.dropped.store(0, std::memory_order_release);
    g_ge.invalid_count.store(0, std::memory_order_release);
    g_ge.excl_direct.store(0, std::memory_order_release);
    g_ge.excl_warmup.store(0, std::memory_order_release);
    g_ge.pending_unconsumed.store(0, std::memory_order_release);
    g_ge.complete.store(true, std::memory_order_release);
    g_ge.anchor_argsort.store(0, std::memory_order_release);
    g_ge.anchor_fused.store(0, std::memory_order_release);
    g_ge.exclusivity_violations.store(0, std::memory_order_release);
    g_ge.faults.store(0, std::memory_order_release);
    g_ge.boundary_seq_next.store(1, std::memory_order_release);
    g_ge.pending_open.store(0, std::memory_order_release);
    g_ge.pending_seq.store(0, std::memory_order_release);
    g_ge.pending_anchors.store(0, std::memory_order_release);
    for (uint32_t i = 0; i < GGML_MOE_GPUEARLY_MAX_LAYERS; i++) {
        g_ge.occ[i].store(0, std::memory_order_release);
    }
    g_ge.w_enter.store(0, std::memory_order_release);
    g_ge.w_exit.store(0, std::memory_order_release);
    g_ge.test_late_commit.store(0, std::memory_order_release);
    g_ge.qpc_calls.store(0, std::memory_order_release);
    // ★소비자 등록은 되돌리지 않는다 — 백엔드 초기화 1회 결선이라 시나리오 경계와 무관하고,
    //   지우면 selftest 뒤 실 런의 소비자가 사라진다(테스트가 프로덕션을 망가뜨리는 형태).
    // resolve_count 도 **누적 유지**한다(반복 해소를 잡으려면 시나리오 경계를 넘어야 의미가 있다).
}

void ggml_moe_gpuearly_test_counts(uint64_t * out_rows, uint64_t * out_dropped,
                                   int32_t * out_complete, uint64_t * out_capacity,
                                   uint64_t * out_invalid, uint64_t * out_excl_direct,
                                   uint64_t * out_excl_warmup, uint64_t * out_pending_unconsumed) {
    if (out_rows)     *out_rows     = g_ge.ring.used();
    if (out_dropped)  *out_dropped  = g_ge.dropped.load(std::memory_order_acquire);
    if (out_complete) *out_complete = g_ge.complete.load(std::memory_order_acquire) ? 1 : 0;
    if (out_capacity) *out_capacity = g_ge.ring.cap;
    if (out_invalid)  *out_invalid  = g_ge.invalid_count.load(std::memory_order_acquire);
    if (out_excl_direct)  *out_excl_direct  = g_ge.excl_direct.load(std::memory_order_acquire);
    if (out_excl_warmup)  *out_excl_warmup  = g_ge.excl_warmup.load(std::memory_order_acquire);
    if (out_pending_unconsumed) {
        *out_pending_unconsumed = g_ge.pending_unconsumed.load(std::memory_order_acquire);
    }
}

int ggml_moe_gpuearly_test_row_bytes(uint64_t idx, void * out, uint64_t bytes) {
    if (!out || bytes != sizeof(GeRecord)) return -1;
    if (!g_ge.ring.base || idx >= g_ge.ring.used()) return -2;
    std::memcpy(out, g_ge.ring.base + (size_t) idx * (size_t) g_ge.ring.rec, sizeof(GeRecord));
    return 0;
}

uint64_t ggml_moe_gpuearly_test_resolve_count(void) {
    return g_ge.resolve_count.load(std::memory_order_acquire);
}

uint64_t ggml_moe_gpuearly_test_qpc_calls(void) {
    return g_ge.qpc_calls.load(std::memory_order_acquire);
}

void ggml_moe_gpuearly_test_publication(uint64_t * out_enter, uint64_t * out_exit,
                                        int32_t * out_gate) {
    if (out_enter) *out_enter = g_ge.w_enter.load(std::memory_order_acquire);
    if (out_exit)  *out_exit  = g_ge.w_exit.load(std::memory_order_acquire);
    if (out_gate)  *out_gate  = (int32_t) g_ge.gate.load(std::memory_order_acquire);
}

void ggml_moe_gpuearly_test_anchor_counts(uint64_t * out_argsort, uint64_t * out_fused,
                                          uint64_t * out_exclusivity_violations) {
    if (out_argsort) *out_argsort = g_ge.anchor_argsort.load(std::memory_order_acquire);
    if (out_fused)   *out_fused   = g_ge.anchor_fused.load(std::memory_order_acquire);
    if (out_exclusivity_violations) {
        *out_exclusivity_violations = g_ge.exclusivity_violations.load(std::memory_order_acquire);
    }
}

uint64_t ggml_moe_gpuearly_test_faults(void) {
    return g_ge.faults.load(std::memory_order_acquire);
}

void ggml_moe_gpuearly_test_get_consumer(ggml_moe_gpuearly_consume_fn * out_fn, void ** out_user) {
    if (out_fn)   *out_fn   = g_ge.consumer.load(std::memory_order_acquire);
    if (out_user) *out_user = g_ge.consumer_user.load(std::memory_order_acquire);
}

void ggml_moe_gpuearly_test_pending(int32_t * out_open, uint64_t * out_seq, uint64_t * out_anchors) {
    if (out_open)    *out_open    = g_ge.pending_open.load(std::memory_order_acquire);
    if (out_seq)     *out_seq     = g_ge.pending_seq.load(std::memory_order_acquire);
    if (out_anchors) *out_anchors = g_ge.pending_anchors.load(std::memory_order_acquire);
}

uint32_t ggml_moe_gpuearly_test_occurrence_peek(uint32_t layer) {
    if (layer >= GGML_MOE_GPUEARLY_MAX_LAYERS) return UINT32_MAX;
    return g_ge.occ[layer].load(std::memory_order_acquire);
}

void ggml_moe_gpuearly_test_stall_writer(int32_t n) {
    if (n > 0) g_ge.w_enter.fetch_add((uint64_t) n, std::memory_order_acq_rel);
}

void ggml_moe_gpuearly_test_release_stalled_writer(int32_t n) {
    if (n > 0) g_ge.w_exit.fetch_add((uint64_t) n, std::memory_order_release);
}

void ggml_moe_gpuearly_test_arm_late_commit(int32_t n) {
    g_ge.test_late_commit.store(n, std::memory_order_release);
}

#else   // !_WIN32 — 비Windows 스텁(PhaseA·shadow·lead-time 관례)

bool     ggml_moe_gpuearly_enabled(void)   { return false; }
uint64_t ggml_moe_gpuearly_now_qpc(void)   { return 0; }
uint64_t ggml_moe_gpuearly_boot_id(void)   { return 0; }
void     ggml_moe_gpuearly_note_seal(uint64_t) {}
uint32_t ggml_moe_gpuearly_next_occurrence(uint32_t) { return 0; }
uint64_t ggml_moe_gpuearly_pending_begin(uint64_t)   { return 0; }
void     ggml_moe_gpuearly_pending_consumed(uint64_t) {}
void     ggml_moe_gpuearly_pending_end(void)         {}
void     ggml_moe_gpuearly_abandon_pending(void)     {}
void     ggml_moe_gpuearly_commit(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint64_t, uint64_t, uint64_t, uint64_t) {}
int      ggml_moe_gpuearly_ms_to_ns(float, uint64_t * out_ns) { if (out_ns) *out_ns = 0; return -1; }
void     ggml_moe_gpuearly_note_exclusivity_violation(void) {}
void     ggml_moe_gpuearly_note_anchor(uint32_t) {}
void     ggml_moe_gpuearly_note_drop(uint64_t) {}
void     ggml_moe_gpuearly_note_fault(void) {}
int      ggml_moe_gpuearly_parse_layer(const char *, const char *, uint32_t *) { return -1; }
void     ggml_moe_gpuearly_set_consumer(ggml_moe_gpuearly_consume_fn, void *) {}
void     ggml_moe_gpuearly_consume(void *, const char *, uint64_t, uint64_t) {}
void     ggml_moe_gpuearly_flush(void) {}
void     ggml_moe_gpuearly_reset_for_test(void) {}

void ggml_moe_gpuearly_test_counts(uint64_t * a, uint64_t * b, int32_t * c, uint64_t * d,
                                   uint64_t * e, uint64_t * f, uint64_t * g, uint64_t * h) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
    if (e) *e = 0; if (f) *f = 0; if (g) *g = 0; if (h) *h = 0;
}
int      ggml_moe_gpuearly_test_row_bytes(uint64_t, void *, uint64_t) { return -1; }
uint64_t ggml_moe_gpuearly_test_resolve_count(void) { return 0; }
uint64_t ggml_moe_gpuearly_test_qpc_calls(void)     { return 0; }
void     ggml_moe_gpuearly_test_publication(uint64_t * a, uint64_t * b, int32_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
void     ggml_moe_gpuearly_test_anchor_counts(uint64_t * a, uint64_t * b, uint64_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
uint64_t ggml_moe_gpuearly_test_faults(void) { return 0; }
void     ggml_moe_gpuearly_test_get_consumer(ggml_moe_gpuearly_consume_fn * f, void ** u) {
    if (f) *f = nullptr; if (u) *u = nullptr;
}
void     ggml_moe_gpuearly_test_pending(int32_t * a, uint64_t * b, uint64_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
uint32_t ggml_moe_gpuearly_test_occurrence_peek(uint32_t) { return 0; }
void     ggml_moe_gpuearly_test_stall_writer(int32_t) {}
void     ggml_moe_gpuearly_test_release_stalled_writer(int32_t) {}
void     ggml_moe_gpuearly_test_arm_late_commit(int32_t) {}

#endif  // _WIN32
