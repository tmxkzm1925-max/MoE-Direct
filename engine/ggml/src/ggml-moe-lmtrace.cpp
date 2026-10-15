// MoE-Direct 레버 A 단계 1 — layer-major 관측 모듈(LMTRACE) 본체. **관측 전용·기본 OFF.**
//
// 사양·설계 근거·환경 변수·CUDA 관측 한계는 ggml-moe-lmtrace.h 상단 주석 참조.
//
// ★이 모듈은 생산 경로를 바꾸지 않는다: 스케줄러 eval callback 을 설치하지 않고(그것은 split 을
//   graph view 로 재분할해 실행 형상 자체를 바꾼다), 이미 확정된 sched 상태를 읽기만 한다.
//
// ★락 순서: 이 모듈의 뮤텍스는 **말단**이다. moe-direct 모듈 카운터 조회
//   (ggml_moe_direct_phase_read_bytes_hit_miss — 내부에서 g_mod.mutex 취득)는 **이 모듈 락을
//   잡기 전에** 끝낸다. 따라서 lm.mu ↔ g_mod.mutex 사이에 순환이 생기지 않는다.
//
// Windows 전용 — 비Windows 는 파일 하단 스텁.

#include "ggml-moe-lmtrace.h"
#include "ggml-moe-direct.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

#include "ggml-impl.h"

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ===========================================================================
// §1. 소도구 — 로그 · SHA-256 · env · 시계
// ===========================================================================

void lm_log(const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[moe-lmtrace] %s\n", buf);
    fflush(stderr);
}

// SHA-256(BCrypt one-shot pseudo-handle) → 소문자 hex. 실패=false.
// ★ggml-moe-direct.cpp:109 sha256_bytes 와 동일 알고리즘. 그 구현은 익명 네임스페이스라 링크
//   불가하므로(모듈 간 원시 프리미티브는 공유하지 않는 것이 이 트리의 관행 — selftest 도 같은
//   이유로 자체 복제본을 둔다) 여기에 복제한다.
bool sha256_bytes(const uint8_t * data, size_t len, std::string & hex_out) {
    unsigned char digest[32];
    NTSTATUS st = BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                             (PUCHAR)data, (ULONG)len, digest, sizeof(digest));
    if (st != 0) return false;
    static const char * hx = "0123456789abcdef";
    hex_out.clear();
    hex_out.reserve(64);
    for (int i = 0; i < 32; i++) { hex_out.push_back(hx[digest[i] >> 4]); hex_out.push_back(hx[digest[i] & 0xF]); }
    return true;
}

std::string getenv_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

bool getenv_is_1(const char * name) {
    return getenv_str(name) == "1";
}

double qpc_freq() {
    static double f = []() {
        LARGE_INTEGER li;
        QueryPerformanceFrequency(&li);
        return (double) li.QuadPart;
    }();
    return f;
}

int64_t qpc_now() {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (int64_t) li.QuadPart;
}

double qpc_ms(int64_t from, int64_t to) {
    const double f = qpc_freq();
    if (f <= 0.0) return 0.0;
    return ((double)(to - from)) * 1000.0 / f;
}

std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), &w[0], n);
    return w;
}

// ===========================================================================
// §2. 결정론 직렬화 프리미티브 (리틀엔디언 고정폭 · 가변 길이는 전부 길이 프레이밍)
// ===========================================================================

void put_u32(std::vector<uint8_t> & b, uint32_t v) {
    b.push_back((uint8_t)( v        & 0xFF));
    b.push_back((uint8_t)((v >>  8) & 0xFF));
    b.push_back((uint8_t)((v >> 16) & 0xFF));
    b.push_back((uint8_t)((v >> 24) & 0xFF));
}

void put_u64(std::vector<uint8_t> & b, uint64_t v) {
    for (int i = 0; i < 8; i++) b.push_back((uint8_t)((v >> (8*i)) & 0xFF));
}

void put_i32(std::vector<uint8_t> & b, int32_t v) { put_u32(b, (uint32_t) v); }
void put_i64(std::vector<uint8_t> & b, int64_t v) { put_u64(b, (uint64_t) v); }

// 고정 위치의 u32 를 사후 기입(레코드 전체 길이 패치용).
void patch_u32(std::vector<uint8_t> & b, size_t off, uint32_t v) {
    b[off + 0] = (uint8_t)( v        & 0xFF);
    b[off + 1] = (uint8_t)((v >>  8) & 0xFF);
    b[off + 2] = (uint8_t)((v >> 16) & 0xFF);
    b[off + 3] = (uint8_t)((v >> 24) & 0xFF);
}

// 레코드 지역 문자열 테이블 — 최초 등장 순서로 인턴(결정론적).
struct StrTable {
    std::vector<std::string>                     items;
    std::unordered_map<std::string, uint32_t>    index;

    StrTable() { intern(""); } // 0번은 항상 빈 문자열

    uint32_t intern(const char * s) {
        return intern(std::string(s ? s : ""));
    }
    uint32_t intern(const std::string & s) {
        auto it = index.find(s);
        if (it != index.end()) return it->second;
        const uint32_t id = (uint32_t) items.size();
        items.push_back(s);
        index.emplace(s, id);
        return id;
    }
    void serialize(std::vector<uint8_t> & b) const {
        put_u32(b, (uint32_t) items.size());
        for (const auto & s : items) {
            put_u32(b, (uint32_t) s.size());   // 길이 프레이밍
            b.insert(b.end(), s.begin(), s.end());
        }
    }
};

// 텐서 참조 — kind: 0=없음 1=node 2=leaf 3=extern(그래프 밖).
struct TRef { uint32_t kind; uint32_t index; };

// ===========================================================================
// §3. 모듈 상태
// ===========================================================================

struct LmState {
    std::mutex mu;

    // env (뮤텍스 아래에서만 갱신 · enabled 는 원자 캐시로 빠른 경로 제공)
    std::atomic<int> enabled_cache{-1};   // -1=미해석 0=off 1=on
    bool             resolved = false;
    std::string      out_prefix;
    uint64_t         max_bytes = 512ull * 1024ull * 1024ull;

    // sidecar
    bool   files_tried = false;
    bool   files_ok    = false;
    HANDLE h_bin = INVALID_HANDLE_VALUE;
    HANDLE h_idx = INVALID_HANDLE_VALUE;
    HANDLE h_req = INVALID_HANDLE_VALUE;

    // 카운터(도달성 앵커 포함)
    std::atomic<uint64_t> probe_calls{0};
    std::atomic<uint64_t> records{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> requests{0};

    uint64_t    capture_seq = 0;
    std::string last_sha;
    std::string chain_sha;

    // (B) 요청 상태기계
    int      phase_state    = 0;   // 0=idle 1=prefill 2=decode
    bool     req_open       = false;
    uint64_t req_seq        = 0;
    uint32_t req_segments   = 0;
    uint64_t req_tokens     = 0;
    double   req_stage_ms[GGML_MOE_LMTRACE_STAGE_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0};
    uint64_t req_base_read_bytes = 0;
    uint64_t req_base_hit        = 0;
    uint64_t req_base_miss       = 0;
    uint64_t req_records_at_open = 0;

    // ★S2 — 경계 CPY 회계(요청 단위 누적)
    uint64_t req_boundary_bytes  = 0;
    uint64_t req_boundary_count  = 0;

    // ★S2 — 다음 capture 에 실릴 range 태그(armed=false 면 tag_flags=0 = 구 레이아웃)
    bool     tag_armed    = false;
    uint64_t tag_decode_call = 0;
    int32_t  tag_range_idx  = 0;
    int32_t  tag_n_ranges   = 0;
    int32_t  tag_ub_idx     = 0;
    int32_t  tag_n_ubatch   = 0;
    int32_t  tag_il_start   = 0;
    int32_t  tag_il_end     = 0;
    int32_t  tag_role       = 0;

    // 스테이지 진행 중 시작 시각(-1 = 열려 있지 않음)
    int64_t  stage_t0[GGML_MOE_LMTRACE_STAGE_COUNT] = {-1, -1, -1, -1, -1};
};

LmState g_lm;

// --- env 해석 -------------------------------------------------------------

void resolve_locked() {
    if (g_lm.resolved) return;
    g_lm.resolved   = true;
    const bool on   = getenv_is_1("MOE_DIRECT_LMTRACE");
    g_lm.out_prefix = on ? getenv_str("MOE_DIRECT_LMTRACE_OUT") : std::string();

    g_lm.max_bytes = 512ull * 1024ull * 1024ull;
    const std::string mb = on ? getenv_str("MOE_DIRECT_LMTRACE_MAX_MB") : std::string();
    if (!mb.empty()) {
        char * endp = nullptr;
        const long v = strtol(mb.c_str(), &endp, 10);
        if (endp && *endp == '\0' && v > 0) {
            g_lm.max_bytes = (uint64_t) v * 1024ull * 1024ull;
        } else {
            lm_log("MOE_DIRECT_LMTRACE_MAX_MB='%s' 해석 불가 — 기본 512MB 유지", mb.c_str());
        }
    }
    g_lm.enabled_cache.store(on ? 1 : 0, std::memory_order_release);
}

bool lm_on() {
    const int c = g_lm.enabled_cache.load(std::memory_order_acquire);
    if (c >= 0) return c == 1;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    resolve_locked();
    return g_lm.enabled_cache.load(std::memory_order_acquire) == 1;
}

// --- sidecar 파일 ---------------------------------------------------------

HANDLE open_new_file(const std::wstring & path) {
    // CREATE_NEW — 기존 산출물을 덮어쓰지 않는다(재런은 새 접두사로).
    return CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                       CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
}

void open_files_locked() {
    if (g_lm.files_tried) return;
    g_lm.files_tried = true;
    if (g_lm.out_prefix.empty()) {
        // 파일 출력 없음 — 카운터·digest 만 유지(사양: OUT 미설정이면 파일을 만들지 않는다).
        return;
    }
    const std::wstring wp = utf8_to_wide(g_lm.out_prefix);
    if (wp.empty()) { lm_log("MOE_DIRECT_LMTRACE_OUT 경로 변환 실패 — 파일 출력 비활성"); return; }

    g_lm.h_bin = open_new_file(wp + L".lmt");
    g_lm.h_idx = open_new_file(wp + L".lmtidx");
    g_lm.h_req = open_new_file(wp + L".lmtreq");

    if (g_lm.h_bin == INVALID_HANDLE_VALUE || g_lm.h_idx == INVALID_HANDLE_VALUE ||
        g_lm.h_req == INVALID_HANDLE_VALUE) {
        lm_log("sidecar 생성 실패(이미 존재하거나 경로 불가) — 파일 출력 비활성, 카운터만 유지");
        if (g_lm.h_bin != INVALID_HANDLE_VALUE) { CloseHandle(g_lm.h_bin); g_lm.h_bin = INVALID_HANDLE_VALUE; }
        if (g_lm.h_idx != INVALID_HANDLE_VALUE) { CloseHandle(g_lm.h_idx); g_lm.h_idx = INVALID_HANDLE_VALUE; }
        if (g_lm.h_req != INVALID_HANDLE_VALUE) { CloseHandle(g_lm.h_req); g_lm.h_req = INVALID_HANDLE_VALUE; }
        return;
    }
    g_lm.files_ok = true;
}

// 실패해도 관측만 잃는다(생산 경로 불변이 최우선) — abort 하지 않고 파일 출력만 끈다.
bool write_all(HANDLE h, const void * data, size_t n) {
    if (h == INVALID_HANDLE_VALUE || n == 0) return h != INVALID_HANDLE_VALUE;
    const uint8_t * p = (const uint8_t *) data;
    size_t off = 0;
    while (off < n) {
        const DWORD chunk = (DWORD) ((n - off) > (32u << 20) ? (32u << 20) : (n - off));
        DWORD wrote = 0;
        if (!WriteFile(h, p + off, chunk, &wrote, nullptr) || wrote == 0) return false;
        off += wrote;
    }
    return true;
}

void close_files_locked() {
    if (g_lm.h_bin != INVALID_HANDLE_VALUE) { FlushFileBuffers(g_lm.h_bin); CloseHandle(g_lm.h_bin); g_lm.h_bin = INVALID_HANDLE_VALUE; }
    if (g_lm.h_idx != INVALID_HANDLE_VALUE) { FlushFileBuffers(g_lm.h_idx); CloseHandle(g_lm.h_idx); g_lm.h_idx = INVALID_HANDLE_VALUE; }
    if (g_lm.h_req != INVALID_HANDLE_VALUE) { FlushFileBuffers(g_lm.h_req); CloseHandle(g_lm.h_req); g_lm.h_req = INVALID_HANDLE_VALUE; }
    g_lm.files_ok    = false;
    g_lm.files_tried = false;
}

// ===========================================================================
// §4. 텐서 기술자 직렬화
// ===========================================================================

struct RefMap {
    std::unordered_map<const ggml_tensor *, TRef> m;
    std::vector<const ggml_tensor *>              externs;

    TRef resolve(const ggml_tensor * t) {
        if (t == nullptr) return TRef{0u, 0u};
        auto it = m.find(t);
        if (it != m.end()) return it->second;
        const TRef r{3u, (uint32_t) externs.size()};
        externs.push_back(t);
        m.emplace(t, r);
        return r;
    }
};

// 텐서 1개의 기술자. 형상·스트라이드·op descriptor·버퍼 사용까지 담고, **할당 주소(data/extra)는
// 담지 않는다** — 정체성 계약의 대상은 "무엇을 어떤 형상·backend 로 계산하는가"이지 할당자가
// 어디에 놓았는가가 아니다.
// with_refs=false(extern) 이면 view_src/src 참조는 전부 {0,0} 으로 채워 레이아웃을 균일하게 둔다.
void put_tensor_desc(std::vector<uint8_t> & b, StrTable & st, RefMap & refs,
                     const ggml_tensor * t, bool with_refs) {
    put_u32(b, (uint32_t) t->op);
    put_u32(b, st.intern(ggml_op_name(t->op)));
    put_u32(b, (uint32_t) t->type);
    put_u32(b, st.intern(ggml_type_name(t->type)));

    for (int i = 0; i < GGML_MAX_DIMS; i++) put_i64(b, (int64_t) t->ne[i]);
    for (int i = 0; i < GGML_MAX_DIMS; i++) put_u64(b, (uint64_t) t->nb[i]);

    put_i32(b, (int32_t) t->flags);

    const int n_op_params = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t);
    put_u32(b, (uint32_t) n_op_params);
    for (int i = 0; i < n_op_params; i++) put_i32(b, t->op_params[i]);

    put_u64(b, (uint64_t) t->view_offs);

    const TRef vr = with_refs ? refs.resolve(t->view_src) : TRef{0u, 0u};
    put_u32(b, vr.kind);
    put_u32(b, vr.index);

    put_u32(b, (uint32_t) GGML_MAX_SRC);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const TRef sr = with_refs ? refs.resolve(t->src[i]) : TRef{0u, 0u};
        put_u32(b, sr.kind);
        put_u32(b, sr.index);
    }

    put_u32(b, st.intern(t->name));

    // 버퍼 사용 — 이름만 담는다(주소·오프셋은 담지 않음).
    const bool has_buf = (t->buffer != nullptr);
    put_u32(b, has_buf ? 1u : 0u);
    put_u32(b, has_buf ? st.intern(ggml_backend_buffer_name(t->buffer)) : 0u);
    put_u32(b, has_buf ? st.intern(ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer))) : 0u);
}

} // namespace

// ===========================================================================
// §5. 공개 API — 게이트
// ===========================================================================

bool ggml_moe_lmtrace_enabled(void) {
    return lm_on();
}

// ===========================================================================
// §6. 공개 API — (A) graph/sched 시그니처
// ===========================================================================

void ggml_moe_lmtrace_graph_capture(
        const struct ggml_cgraph                 * graph,
        const int32_t                            * node_backend_ids,
        const int32_t                            * leaf_backend_ids,
        int32_t                                    n_backends,
        ggml_backend_t                           * backends,
        int32_t                                    n_splits,
        const struct ggml_moe_lmtrace_split_desc * splits) {
    // ★도달성 앵커: 게이트 **이전**에 증가한다. 프로덕션에서는 호출자가
    //   ggml_moe_lmtrace_enabled() 로 막으므로 OFF 면 이 함수 자체가 불린 적 없어 0 이다.
    //   selftest 는 이 함수를 직접 불러 "호출은 도달했으나 본문은 미진입"을 구분한다.
    g_lm.probe_calls.fetch_add(1, std::memory_order_relaxed);

    if (!lm_on())      return;
    if (graph == nullptr) return;

    // 인자 정규화 — 음수 개수/NULL 배열은 "없음"으로 읽는다(호출자는 신뢰원이지만 selftest 가
    // backends 없이 부르는 경로가 정상 사용례라 진입점에서 한 번 정리한다).
    if (splits   == nullptr || n_splits   < 0) n_splits   = 0;
    if (backends == nullptr || n_backends < 0) n_backends = 0;

    // ★캡처 전 구간을 단일 락으로 감싼다 — seq 발급·직렬화·SHA·기록·연쇄 지문이 한 원자 단위여야
    //   레코드 순서와 chain 값이 결정론적이다(seq 만 락 안에서 받고 나머지를 밖에서 하면 동시
    //   캡처 시 기록 순서와 seq 순서가 어긋난다).
    std::lock_guard<std::mutex> lock(g_lm.mu);

    // --- 참조 지도: nodes → leafs 순으로 등록(결정론) ---
    RefMap refs;
    refs.m.reserve((size_t) graph->n_nodes + (size_t) graph->n_leafs + 16);
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]) refs.m.emplace(graph->nodes[i], TRef{1u, (uint32_t) i});
    }
    for (int i = 0; i < graph->n_leafs; i++) {
        if (graph->leafs[i]) refs.m.emplace(graph->leafs[i], TRef{2u, (uint32_t) i});
    }

    StrTable st;
    std::vector<uint8_t> b;
    b.reserve((size_t) (graph->n_nodes + graph->n_leafs) * 320 + 4096);

    // --- 헤더 ---
    const char * magic = MOE_LMTRACE_MAGIC;      // "MOELMT1" (7자)
    b.insert(b.end(), magic, magic + 7);
    b.push_back(0x00);                            // 8바이트 정렬 패딩
    const size_t off_rec_bytes = b.size();
    put_u32(b, 0u);                               // rec_bytes — 말미에 패치
    put_u32(b, MOE_LMTRACE_LAYOUT_VERSION);
    put_u32(b, (uint32_t) MOE_LMTRACE_KIND_GRAPH_SIGNATURE);
    // ★구 `reserved` = S2 tag_flags. 태그가 무장돼 있지 않으면 0 을 그대로 써서 **구 레이아웃과
    //   바이트 동일**을 유지한다(S0 baseline 판독·legacy 레코드 1:1 비교의 전제).
    put_u32(b, g_lm.tag_armed ? MOE_LMTRACE_TAG_FLAG_RANGE : MOE_LMTRACE_TAG_FLAG_NONE);

    const uint64_t seq = g_lm.capture_seq++;
    put_u64(b, seq);
    put_u64(b, (uint64_t) graph->uid);
    put_u32(b, (uint32_t) graph->n_nodes);
    put_u32(b, (uint32_t) graph->n_leafs);
    put_u32(b, (uint32_t) (n_splits   > 0 ? n_splits   : 0));
    put_u32(b, (uint32_t) (n_backends > 0 ? n_backends : 0));

    // --- ★S2 range 태그 블록(tag_flags==1 일 때만) ---
    if (g_lm.tag_armed) {
        put_u32(b, MOE_LMTRACE_TAG_BYTES);
        put_u32(b, MOE_LMTRACE_TAG_VERSION);
        put_u64(b, g_lm.tag_decode_call);
        put_u32(b, (uint32_t) g_lm.tag_range_idx);
        put_u32(b, (uint32_t) g_lm.tag_n_ranges);
        put_u32(b, (uint32_t) g_lm.tag_ub_idx);
        put_u32(b, (uint32_t) g_lm.tag_n_ubatch);
        put_i32(b, g_lm.tag_il_start);
        put_i32(b, g_lm.tag_il_end);
        put_u32(b, (uint32_t) g_lm.tag_role);
        put_u32(b, (uint32_t) GGML_MOE_LMTRACE_MODE_RANGE);
    }

    // externs/strings 개수는 본문을 다 만든 뒤에야 확정되므로 별도 꼬리 블록에 기록한다
    // (헤더에 자리를 비워두고 패치하는 방식보다 파서가 단순하다).

    // --- backends ---
    {
        std::vector<uint32_t> refs_names;
        refs_names.reserve((size_t) (n_backends > 0 ? n_backends : 0));
        for (int32_t i = 0; i < n_backends; i++) {
            const char * nm = (backends && backends[i]) ? ggml_backend_name(backends[i]) : "";
            refs_names.push_back(st.intern(nm));
        }
        for (uint32_t r : refs_names) put_u32(b, r);
    }

    // --- splits ---
    for (int32_t i = 0; i < n_splits; i++) {
        const struct ggml_moe_lmtrace_split_desc & s = splits[i];
        put_i32(b, s.backend_id);
        put_i32(b, s.i_start);
        put_i32(b, s.i_end);
        put_i32(b, s.n_inputs);
    }

    // --- nodes ---
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * t = graph->nodes[i];
        if (t == nullptr) { put_u32(b, 0xFFFFFFFFu); continue; }  // null 노드 표식(정상 그래프엔 없음)
        put_u32(b, 0u);
        put_tensor_desc(b, st, refs, t, /*with_refs=*/true);
        put_i32(b, node_backend_ids ? node_backend_ids[i] : -1);
    }

    // --- leafs ---
    for (int i = 0; i < graph->n_leafs; i++) {
        const ggml_tensor * t = graph->leafs[i];
        if (t == nullptr) { put_u32(b, 0xFFFFFFFFu); continue; }
        put_u32(b, 0u);
        put_tensor_desc(b, st, refs, t, /*with_refs=*/true);
        put_i32(b, leaf_backend_ids ? leaf_backend_ids[i] : -1);
    }

    // --- externs(그래프 밖에서 참조된 텐서 — 정상 그래프에서는 0건) ---
    // refs.externs 는 위 순회 중에만 늘어난다. shallow 기술자(참조는 전부 {0,0})라 재귀가 없다.
    // ★개수는 .lmtidx 에도 노출한다 — G-L0 는 대상 프리필 레코드 전건 externs=0 을 fail-close
    //   조건으로 요구하므로(R13 ③8·④7) 판독기가 바이너리를 다 풀지 않고도 볼 수 있어야 한다.
    const size_t n_ext = refs.externs.size();
    {
        put_u32(b, (uint32_t) n_ext);
        for (size_t i = 0; i < n_ext; i++) {
            put_tensor_desc(b, st, refs, refs.externs[i], /*with_refs=*/false);
        }
    }

    // --- 문자열 테이블(꼬리 — 인턴이 모두 끝난 뒤) ---
    st.serialize(b);

    // --- 길이 패치 + 지문 ---
    patch_u32(b, off_rec_bytes, (uint32_t) b.size());

    std::string rec_sha;
    if (!sha256_bytes(b.data(), b.size(), rec_sha)) {
        lm_log("SHA-256 계산 실패 — 레코드 %llu 드롭", (unsigned long long) seq);
        g_lm.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // --- 기록 (진입부에서 잡은 락을 그대로 유지) ---
    {
        open_files_locked();

        const uint64_t prev = g_lm.bytes.load(std::memory_order_relaxed);
        if (prev + (uint64_t) b.size() > g_lm.max_bytes) {
            g_lm.dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (g_lm.files_ok) {
            if (!write_all(g_lm.h_bin, b.data(), b.size())) {
                lm_log("sidecar 바이너리 기록 실패 — 파일 출력 비활성(카운터는 유지)");
                close_files_locked();
            } else {
                char line[256];
                const int n = snprintf(line, sizeof(line),
                                       "seq=%llu uid=%llu nodes=%d leafs=%d splits=%d externs=%zu "
                                       "bytes=%zu sha256=%s\n",
                                       (unsigned long long) seq, (unsigned long long) graph->uid,
                                       graph->n_nodes, graph->n_leafs, (int) n_splits, n_ext,
                                       b.size(), rec_sha.c_str());
                if (n > 0 && !write_all(g_lm.h_idx, line, (size_t) n)) {
                    lm_log("sidecar 인덱스 기록 실패 — 파일 출력 비활성(카운터는 유지)");
                    close_files_locked();
                }
            }
        }

        // 연쇄 지문 chain = SHA256(prev_hex || rec_hex) — 런 전체를 값 하나로 비교하기 위함.
        const std::string joined = g_lm.chain_sha + rec_sha;
        std::string next_chain;
        if (sha256_bytes((const uint8_t *) joined.data(), joined.size(), next_chain)) {
            g_lm.chain_sha = next_chain;
        }
        g_lm.last_sha = rec_sha;

        g_lm.bytes.fetch_add((uint64_t) b.size(), std::memory_order_relaxed);
        g_lm.records.fetch_add(1, std::memory_order_relaxed);
    }
}

// ===========================================================================
// §7. 공개 API — (B) request-level 카운터
// ===========================================================================

namespace {

// 요청 폐합 — 뮤텍스를 쥔 상태에서 호출한다. 여기서 experts 읽기 델타를 확정해 한 줄 기록한다.
// status = MOE_LMTRACE_STATUS_* (decode/reopen/finish/abort) — 판독기가 정상 폐합과 실패 폐합을
// 구분해야 G-L0 대상 요청을 안전하게 고를 수 있다.
void close_request_locked(uint64_t now_read_bytes, uint64_t now_hit, uint64_t now_miss,
                          uint64_t now_records, const char * status) {
    if (!g_lm.req_open) return;
    g_lm.req_open = false;

    const uint64_t d_bytes   = now_read_bytes >= g_lm.req_base_read_bytes ? now_read_bytes - g_lm.req_base_read_bytes : 0;
    const uint64_t d_hit     = now_hit        >= g_lm.req_base_hit        ? now_hit        - g_lm.req_base_hit        : 0;
    const uint64_t d_miss    = now_miss       >= g_lm.req_base_miss       ? now_miss       - g_lm.req_base_miss       : 0;
    const uint64_t d_records = now_records    >= g_lm.req_records_at_open ? now_records    - g_lm.req_records_at_open : 0;

    g_lm.requests.fetch_add(1, std::memory_order_relaxed);

    if (g_lm.files_ok) {
        char line[1024];
        const int n = snprintf(line, sizeof(line),
            "{\"kind\":\"lmtrace_request\",\"req_seq\":%llu,\"status\":\"%s\",\"segments\":%u,\"prefill_tokens\":%llu,"
            "\"graph_build_ms\":%.3f,\"sched_alloc_ms\":%.3f,\"submit_host_ms\":%.3f,\"sync_ms\":%.3f,"
            "\"boundary_copy_ms\":%.3f,"
            "\"boundary_copy_bytes\":%llu,\"boundary_copy_count\":%llu,"
            "\"experts_read_bytes\":%llu,\"experts_hit\":%llu,\"experts_miss\":%llu,"
            "\"graph_records\":%llu,\"graph_chain_sha256\":\"%s\"}\n",
            (unsigned long long) g_lm.req_seq,
            status ? status : MOE_LMTRACE_STATUS_FINISH,
            g_lm.req_segments,
            (unsigned long long) g_lm.req_tokens,
            g_lm.req_stage_ms[GGML_MOE_LMTRACE_STAGE_GRAPH_BUILD],
            g_lm.req_stage_ms[GGML_MOE_LMTRACE_STAGE_SCHED_ALLOC],
            g_lm.req_stage_ms[GGML_MOE_LMTRACE_STAGE_SUBMIT_HOST],
            g_lm.req_stage_ms[GGML_MOE_LMTRACE_STAGE_SYNC],
            g_lm.req_stage_ms[GGML_MOE_LMTRACE_STAGE_BOUNDARY_COPY],
            (unsigned long long) g_lm.req_boundary_bytes,
            (unsigned long long) g_lm.req_boundary_count,
            (unsigned long long) d_bytes,
            (unsigned long long) d_hit,
            (unsigned long long) d_miss,
            (unsigned long long) d_records,
            g_lm.chain_sha.c_str());
        if (n > 0 && !write_all(g_lm.h_req, line, (size_t) n)) {
            lm_log("요청 레코드 기록 실패 — 파일 출력 비활성(카운터는 유지)");
            close_files_locked();
        }
    }

    g_lm.req_seq++;
    g_lm.req_segments = 0;
    g_lm.req_tokens   = 0;
    g_lm.req_boundary_bytes = 0;
    g_lm.req_boundary_count = 0;
    for (int i = 0; i < GGML_MOE_LMTRACE_STAGE_COUNT; i++) g_lm.req_stage_ms[i] = 0.0;
}

// prefill 요청 개시(락 보유 상태). graph_boundary 의 prefill 분기와 동일한 초기화를 공유한다.
void open_prefill_locked(uint64_t rb, uint64_t hit, uint64_t miss, uint64_t recs) {
    close_request_locked(rb, hit, miss, recs, MOE_LMTRACE_STATUS_REOPEN);
    g_lm.req_open             = true;
    g_lm.req_segments         = 0;
    g_lm.req_tokens           = 0;
    g_lm.req_base_read_bytes  = rb;
    g_lm.req_base_hit         = hit;
    g_lm.req_base_miss        = miss;
    g_lm.req_records_at_open  = recs;
    g_lm.req_boundary_bytes   = 0;
    g_lm.req_boundary_count   = 0;
    for (int i = 0; i < GGML_MOE_LMTRACE_STAGE_COUNT; i++) g_lm.req_stage_ms[i] = 0.0;
}

} // namespace

void ggml_moe_lmtrace_graph_boundary(int32_t is_prefill, int32_t n_tokens) {
    if (!lm_on()) return;

    // ★락 순서 — moe-direct 카운터 조회는 이 모듈 락을 잡기 전에 끝낸다(헤더 상단 주석 참조).
    //   phase 1 = PREFILL(ggml-moe-direct 의 Phase 열거와 동일).
    uint64_t rb = 0, hit = 0, miss = 0;
    ggml_moe_direct_phase_read_bytes_hit_miss(1, &rb, &hit, &miss);
    const uint64_t recs = g_lm.records.load(std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(g_lm.mu);
    open_files_locked();

    if (is_prefill) {
        if (g_lm.phase_state != 1) {
            // 새 요청 개시 — 이전에 열린 요청이 있으면 먼저 폐합한다(순수 프리필 요청의 자연 폐합).
            open_prefill_locked(rb, hit, miss, recs);
        }
        g_lm.phase_state = 1;
        g_lm.req_segments++;
        g_lm.req_tokens += (uint64_t) (n_tokens > 0 ? n_tokens : 0);
    } else {
        if (g_lm.phase_state == 1) {
            close_request_locked(rb, hit, miss, recs, MOE_LMTRACE_STATUS_DECODE);
        }
        g_lm.phase_state = 2;
    }
}

void ggml_moe_lmtrace_stage_enter(int32_t stage) {
    if (!lm_on()) return;
    if (stage < 0 || stage >= GGML_MOE_LMTRACE_STAGE_COUNT) return;
    const int64_t t0 = qpc_now();
    std::lock_guard<std::mutex> lock(g_lm.mu);
    g_lm.stage_t0[stage] = t0;
}

void ggml_moe_lmtrace_stage_leave(int32_t stage) {
    if (!lm_on()) return;
    if (stage < 0 || stage >= GGML_MOE_LMTRACE_STAGE_COUNT) return;
    const int64_t t1 = qpc_now();
    std::lock_guard<std::mutex> lock(g_lm.mu);
    const int64_t t0 = g_lm.stage_t0[stage];
    g_lm.stage_t0[stage] = -1;
    if (t0 < 0) return;                 // 짝이 맞지 않는 leave 는 무시
    if (!g_lm.req_open) return;         // 요청 밖(decode 등)의 구간은 집계하지 않는다
    g_lm.req_stage_ms[stage] += qpc_ms(t0, t1);
}

namespace {

// finish/abort 공통 — 열린 요청을 status 로 폐합하고 상태기계를 idle 로 되돌린다.
// ★idle 복귀가 핵심이다: 이것이 없으면 실패 후 다음 프리필이 phase_state==prefill 인 채로 와서
//   이전 요청에 합쳐진다(R13 ③4 REACHABLE 결함).
void close_and_reset(const char * status) {
    if (!lm_on()) return;
    uint64_t rb = 0, hit = 0, miss = 0;
    ggml_moe_direct_phase_read_bytes_hit_miss(1, &rb, &hit, &miss);
    const uint64_t recs = g_lm.records.load(std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(g_lm.mu);
    open_files_locked();
    close_request_locked(rb, hit, miss, recs, status);
    g_lm.phase_state = 0;
    for (int i = 0; i < GGML_MOE_LMTRACE_STAGE_COUNT; i++) g_lm.stage_t0[i] = -1;
}

} // namespace

void ggml_moe_lmtrace_request_finish(void) { close_and_reset(MOE_LMTRACE_STATUS_FINISH); }
void ggml_moe_lmtrace_request_abort (void) { close_and_reset(MOE_LMTRACE_STATUS_ABORT);  }

// ===========================================================================
// §7-b. 공개 API — ★S2 층-우선(range) 확장
// ===========================================================================

void ggml_moe_lmtrace_range_tag_set(uint64_t decode_call,
                                    int32_t  range_idx, int32_t n_ranges,
                                    int32_t  ub_idx,    int32_t n_ubatch,
                                    int32_t  il_start,  int32_t il_end,
                                    int32_t  boundary_role) {
    if (!lm_on()) return;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    g_lm.tag_armed       = true;
    g_lm.tag_decode_call = decode_call;
    g_lm.tag_range_idx   = range_idx;
    g_lm.tag_n_ranges    = n_ranges;
    g_lm.tag_ub_idx      = ub_idx;
    g_lm.tag_n_ubatch    = n_ubatch;
    g_lm.tag_il_start    = il_start;
    g_lm.tag_il_end      = il_end;
    g_lm.tag_role        = boundary_role;
}

void ggml_moe_lmtrace_range_tag_clear(void) {
    if (!lm_on()) return;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    g_lm.tag_armed = false;
}

void ggml_moe_lmtrace_request_prefill_enter(int32_t n_tokens) {
    if (!lm_on()) return;

    // ★락 순서 — moe-direct 카운터 조회는 이 모듈 락을 잡기 전에 끝낸다(graph_boundary 와 동일).
    uint64_t rb = 0, hit = 0, miss = 0;
    ggml_moe_direct_phase_read_bytes_hit_miss(1, &rb, &hit, &miss);
    const uint64_t recs = g_lm.records.load(std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(g_lm.mu);
    open_files_locked();

    if (g_lm.phase_state != 1) {
        open_prefill_locked(rb, hit, miss, recs);
    }
    g_lm.phase_state = 1;
    // ★tokens 는 여기서 **decode 진입당 1회만** 적산한다(SPEC §5 ⓑ). segment 는 늘리지 않는다 —
    //   그것은 각 (range,ub) compute 가 range_segment() 로 한다.
    g_lm.req_tokens += (uint64_t) (n_tokens > 0 ? n_tokens : 0);
}

void ggml_moe_lmtrace_range_segment(void) {
    if (!lm_on()) return;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    if (!g_lm.req_open) return;
    g_lm.req_segments++;
}

void ggml_moe_lmtrace_boundary_copy_note(uint64_t bytes, uint64_t count) {
    if (!lm_on()) return;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    if (!g_lm.req_open) return;
    g_lm.req_boundary_bytes += bytes;
    g_lm.req_boundary_count += count;
}

void ggml_moe_lmtrace_boundary_copy_counters(uint64_t * bytes, uint64_t * count) {
    std::lock_guard<std::mutex> lock(g_lm.mu);
    if (bytes) *bytes = g_lm.req_boundary_bytes;
    if (count) *count = g_lm.req_boundary_count;
}

// ===========================================================================
// §8. 공개 API — 조회
// ===========================================================================

void ggml_moe_lmtrace_counters(uint64_t * probe_calls, uint64_t * records, uint64_t * dropped,
                               uint64_t * bytes, uint64_t * requests) {
    if (probe_calls) *probe_calls = g_lm.probe_calls.load(std::memory_order_relaxed);
    if (records)     *records     = g_lm.records.load(std::memory_order_relaxed);
    if (dropped)     *dropped     = g_lm.dropped.load(std::memory_order_relaxed);
    if (bytes)       *bytes       = g_lm.bytes.load(std::memory_order_relaxed);
    if (requests)    *requests    = g_lm.requests.load(std::memory_order_relaxed);
}

int ggml_moe_lmtrace_last_digest(char * out, int32_t cap) {
    if (!out || cap < 65) return -1;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    const std::string & s = g_lm.last_sha;
    if ((int32_t) s.size() + 1 > cap) return -1;
    memcpy(out, s.c_str(), s.size() + 1);
    return 0;
}

int ggml_moe_lmtrace_chain_digest(char * out, int32_t cap) {
    if (!out || cap < 65) return -1;
    std::lock_guard<std::mutex> lock(g_lm.mu);
    const std::string & s = g_lm.chain_sha;
    if ((int32_t) s.size() + 1 > cap) return -1;
    memcpy(out, s.c_str(), s.size() + 1);
    return 0;
}

void ggml_moe_lmtrace_test_reset(void) {
    std::lock_guard<std::mutex> lock(g_lm.mu);
    close_files_locked();
    g_lm.resolved = false;
    g_lm.enabled_cache.store(-1, std::memory_order_release);
    g_lm.out_prefix.clear();
    g_lm.max_bytes = 512ull * 1024ull * 1024ull;

    g_lm.probe_calls.store(0, std::memory_order_relaxed);
    g_lm.records.store(0, std::memory_order_relaxed);
    g_lm.dropped.store(0, std::memory_order_relaxed);
    g_lm.bytes.store(0, std::memory_order_relaxed);
    g_lm.requests.store(0, std::memory_order_relaxed);

    g_lm.capture_seq = 0;
    g_lm.last_sha.clear();
    g_lm.chain_sha.clear();

    g_lm.phase_state  = 0;
    g_lm.req_open     = false;
    g_lm.req_seq      = 0;
    g_lm.req_segments = 0;
    g_lm.req_tokens   = 0;
    for (int i = 0; i < GGML_MOE_LMTRACE_STAGE_COUNT; i++) { g_lm.req_stage_ms[i] = 0.0; g_lm.stage_t0[i] = -1; }
    g_lm.files_ok = false;
    g_lm.req_base_read_bytes = 0;
    g_lm.req_base_hit        = 0;
    g_lm.req_base_miss       = 0;
    g_lm.req_records_at_open = 0;

    g_lm.req_boundary_bytes  = 0;
    g_lm.req_boundary_count  = 0;
    g_lm.tag_armed           = false;
}

#else // !_WIN32

// ===========================================================================
// 비Windows 스텁 — 전부 무동작(모듈은 Windows 전용)
// ===========================================================================

bool ggml_moe_lmtrace_enabled(void) { return false; }

void ggml_moe_lmtrace_graph_capture(const struct ggml_cgraph *, const int32_t *, const int32_t *,
                                    int32_t, ggml_backend_t *, int32_t,
                                    const struct ggml_moe_lmtrace_split_desc *) {}

void ggml_moe_lmtrace_graph_boundary(int32_t, int32_t) {}
void ggml_moe_lmtrace_stage_enter(int32_t) {}
void ggml_moe_lmtrace_stage_leave(int32_t) {}
void ggml_moe_lmtrace_request_finish(void) {}
void ggml_moe_lmtrace_request_abort(void) {}

void ggml_moe_lmtrace_counters(uint64_t * probe_calls, uint64_t * records, uint64_t * dropped,
                               uint64_t * bytes, uint64_t * requests) {
    if (probe_calls) *probe_calls = 0;
    if (records)     *records     = 0;
    if (dropped)     *dropped     = 0;
    if (bytes)       *bytes       = 0;
    if (requests)    *requests    = 0;
}

int  ggml_moe_lmtrace_last_digest(char *, int32_t)  { return -1; }
int  ggml_moe_lmtrace_chain_digest(char *, int32_t) { return -1; }
void ggml_moe_lmtrace_test_reset(void) {}

// ★S2 확장 — 비Windows 스텁
void ggml_moe_lmtrace_range_tag_set(uint64_t, int32_t, int32_t, int32_t, int32_t,
                                    int32_t, int32_t, int32_t) {}
void ggml_moe_lmtrace_range_tag_clear(void) {}
void ggml_moe_lmtrace_request_prefill_enter(int32_t) {}
void ggml_moe_lmtrace_range_segment(void) {}
void ggml_moe_lmtrace_boundary_copy_note(uint64_t, uint64_t) {}
void ggml_moe_lmtrace_boundary_copy_counters(uint64_t * bytes, uint64_t * count) {
    if (bytes) *bytes = 0;
    if (count) *count = 0;
}

#endif // _WIN32
