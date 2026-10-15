#pragma once

// ===========================================================================
// MoE-Direct 레버 A 단계 1 · S1 — immutable ubatch memory plan
//
// 권위 사양: bench/techdev/superbatch_s1/SPEC_S1_MEMPLAN.md v1.0(동결 · r23 [ACCEPT])
//
// plan = memplan_prepare(memory, balloc, n_ubatch)  전 ubatch 배치 선계산 + ub별 동결 뷰 캡처
// view = plan->select(ub_idx)                       동결 뷰 조회 · 임의 순서 · 무변이·멱등
//        plan->apply(ub_idx)                        변이 지점 · canonical 순서 0→K-1 강제
//        plan->commit()                             전 ubatch apply 후 1회 — 종결·불변식 확인
//        plan->abort()                              모든 비종결 상태에서 허용 — prepare 이전 복원
//
// ★S1 은 라이브 decode 경로를 바꾸지 않는다(배선은 S2). llama-context 무접촉.
// ===========================================================================

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// S1 selfcheck 콜백. ok != 0 이면 PASS. name 은 콜백 호출 중에만 유효하다.
typedef void (*llama_memplan_selfcheck_cb)(void * ud, int ok, const char * name);

// SPEC §4 픽스처 ⓐ~ⓛ 를 3계열(standard KV · hybrid · DSV4)에서 독립 실행한다.
// 반환값 = 실패한 check 의 수(0 = 전건 PASS). cb 가 null 이면 집계만 한다.
// ★엔진 내부 C++ 타입(llama_kv_cache 등)은 DLL 로 내보내지지 않으므로, 픽스처 본체는
//   이 라이브러리 안에서 돌고 결과만 콜백으로 넘긴다(호출자가 PASS/FAIL 회계를 소유).
LLAMA_API int llama_memplan_selfcheck(llama_memplan_selfcheck_cb cb, void * ud);

#ifdef __cplusplus
}
#endif

// ---------------------------------------------------------------------------
// 이하 C++ 엔진 표면. 내부 헤더에 의존하므로, C 진입점만 필요한 소비자(selftest)는
// LLAMA_MEMPLAN_C_ONLY 를 정의하고 include 한다.
// ---------------------------------------------------------------------------
#if defined(__cplusplus) && !defined(LLAMA_MEMPLAN_C_ONLY)

#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// SPEC §1 상태기계
enum llama_memplan_state {
    LLAMA_MEMPLAN_STATE_PREPARED = 0,
    LLAMA_MEMPLAN_STATE_APPLYING,
    LLAMA_MEMPLAN_STATE_APPLIED,
    LLAMA_MEMPLAN_STATE_COMMITTED,   // terminal
    LLAMA_MEMPLAN_STATE_ABORTED,     // terminal
    LLAMA_MEMPLAN_STATE_FAILED_PREPARE,
};

enum llama_memplan_kind {
    LLAMA_MEMPLAN_KIND_NONE = 0,
    LLAMA_MEMPLAN_KIND_KV,       // ① standard KV (deepseek2 류)
    LLAMA_MEMPLAN_KIND_HYBRID,   // ② hybrid (qwen35moe — attn KV + recurrent 하위 kind)
    LLAMA_MEMPLAN_KIND_DSV4,     // ③ DSV4 (deepseek4 전용 복합)
};

// fail-close 사유 (관측 가능해야 한다 — 픽스처 ⓒⓗ 판정축)
enum llama_memplan_fail {
    LLAMA_MEMPLAN_FAIL_NONE = 0,
    LLAMA_MEMPLAN_FAIL_PREPARE,      // 배치 불가 / 하위 prepare 실패
    LLAMA_MEMPLAN_FAIL_ALLOC,        // allocation preflight (§6 Q-B)
    LLAMA_MEMPLAN_FAIL_ORDER,        // canonical 0→K-1 순서 위반
    LLAMA_MEMPLAN_FAIL_REAPPLY,      // 같은 ub_idx 재-apply
    LLAMA_MEMPLAN_FAIL_TERMINAL,     // terminal 상태에서 apply/commit 재호출
    LLAMA_MEMPLAN_FAIL_INCOMPLETE,   // 미적용 ub 잔존 commit
    LLAMA_MEMPLAN_FAIL_APPLY,        // 하위 apply 실패 (composite 보상 복원 수반)
    LLAMA_MEMPLAN_FAIL_DESTRUCT,     // 비종결 destructor 자동 abort
    LLAMA_MEMPLAN_FAIL_RESTORE,      // 복원 불가 → 해당 seq fail-close clear
    LLAMA_MEMPLAN_FAIL_PREPARE_DRIFT,// prepare 무변이 대조 실패 → 실복원 후 FAILED_PREPARE
    LLAMA_MEMPLAN_FAIL_STALE,        // 폐기(FAILED_PREPARE) plan 재사용 거부
};

// test-only 호출 순서 trace 이벤트 (r24 수리 3 — "순서를 뒤집으면 반드시 FAIL" 의 증명 장치)
enum llama_memplan_event {
    LLAMA_MEMPLAN_EV_NONE = 0,
    LLAMA_MEMPLAN_EV_KV_APPLY,            // ① standard KV
    LLAMA_MEMPLAN_EV_ATTN_APPLY,          // ② hybrid 첫 번째 하위
    LLAMA_MEMPLAN_EV_RECURRENT_REPLAY,    // ② hybrid 두 번째 하위(캡처 리플레이)
    LLAMA_MEMPLAN_EV_DSV4_RAW_BASE,       // ③ DSV4 첫 번째 하위
    LLAMA_MEMPLAN_EV_DSV4_RAW_SWA,        // ③ DSV4 두 번째 하위
    LLAMA_MEMPLAN_EV_DSV4_COMP_BOUNDARY,  // ③ DSV4 압축 경계 통과(graph output — apply 변이 없음)
};

// ---------------------------------------------------------------------------
// 동결 뷰 (tagged immutable view — SPEC §1·§3)
//   그래프 입력이 소비하는 값 전부가 "값으로" 들어 있어야 한다. live 재계산 금지.
// ---------------------------------------------------------------------------

// ① standard KV / ② hybrid 의 attn 하위 / (DSV4 는 별도)
struct llama_memplan_kv_view {
    llama_kv_cache::slot_info sinfo;      // 셀 인덱스

    uint32_t n_kv = 0;                    // 동결값 (live v_cells 계산 결과)

    std::vector<int64_t> k_idxs;          // set_input_k_idxs 동결본
    std::vector<int64_t> v_idxs;          // set_input_v_idxs 동결본

    std::vector<float>   kq_mask;         // set_input_kq_mask 동결본 (visibility)
    int64_t              kq_ne[4] = { 0, 0, 0, 0 };
};

// ② hybrid 의 recurrent 하위 kind — find_slot 캡처-리플레이 (SPEC §3 ② · §6 Q-A′)
struct llama_memplan_rs_view {
    // 사후 상태 스냅샷 전건
    std::vector<llama_memory_recurrent::mem_cell> cells;
    uint32_t used = 0;
    uint32_t head = 0;
    uint32_t n    = 0;
    int32_t  rs_z = -1;

    // ★rs_idx 1회 소비 의미론(r22 수정 1)
    std::vector<uint32_t> rs_idx_pre;   // s_copy 소비 직전 (이 ubatch 진입 상태)
    std::vector<uint32_t> rs_idx_post;  // s_copy 소비 후 — 다음 ubatch·최종 APPLIED 로 승계

    std::vector<int32_t>  s_copy;       // immutable 복사본 (실 set_input 순서로 뽑은 값)
};

// ③ DSV4 — batch 뷰 전건 (SPEC §3 ③)
struct llama_memplan_dsv4_view {
    llama_kv_cache::slot_info sinfo_raw_base_write;
    llama_kv_cache::slot_info sinfo_raw_swa_write;
    llama_kv_cache::slot_info sinfo_raw_swa_read;

    uint32_t raw_n_kv = 0;
    std::vector<float> raw_kq_mask;             // raw visibility
    int64_t            raw_kq_ne[4] = { 0, 0, 0, 0 };

    llama_kv_cache::slot_info sinfo_csa;
    llama_kv_cache::slot_info sinfo_hca;
    llama_kv_cache::slot_info sinfo_lid;

    llama_kv_cache_dsv4_context::comp_plan plan_csa;
    llama_kv_cache_dsv4_context::comp_plan plan_hca;
    llama_kv_cache_dsv4_context::comp_plan plan_lid;

    // compressor-state identity 3 (압축 캐시는 apply 대상이 아니라 graph output —
    //   plan 은 identity 동결과 abort 경계만 계약한다)
    struct comp_identity {
        const void * state = nullptr;
        uint32_t ratio      = 0;
        uint32_t state_size = 0;
        uint32_t n_stream   = 0;

        bool operator==(const comp_identity & o) const {
            return state == o.state && ratio == o.ratio && state_size == o.state_size && n_stream == o.n_stream;
        }
    };

    comp_identity id_csa;
    comp_identity id_hca;
    comp_identity id_lid;
};

struct llama_memplan_view {
    uint32_t ub_idx = 0;

    llama_ubatch ubatch     {};   // 계획된 ubatch (data 는 shared_ptr 로 살아 있다)
    llama_ubatch ubatch_raw {};   // DSV4 전용 raw-write ubatch

    bool has_kv   = false;
    bool has_rs   = false;
    bool has_dsv4 = false;

    llama_memplan_kv_view   kv;
    llama_memplan_rs_view   rs;
    llama_memplan_dsv4_view dsv4;
};

// ---------------------------------------------------------------------------
// plan
// ---------------------------------------------------------------------------
class llama_memplan {
public:
    ~llama_memplan();

    llama_memplan(const llama_memplan &)             = delete;
    llama_memplan & operator=(const llama_memplan &) = delete;

    // 실패 = 무변이 FAILED_PREPARE (§6 Q-B) — 항상 non-null plan 을 돌려주고
    // 상태/사유는 get_state()/get_fail() 로 관측한다.
    static std::unique_ptr<llama_memplan> prepare(
            llama_memory_i     * mem,
            llama_batch_allocr & balloc,
            uint32_t             n_ubatch,
            bool                 embd_all,
            bool                 causal_attn);

    llama_memplan_kind  get_kind()  const { return kind;  }
    llama_memplan_state get_state() const { return state; }
    llama_memplan_fail  get_fail()  const { return fail;  }

    const std::string & get_fail_note() const { return fail_note; }

    uint32_t get_n_ubatch()  const { return (uint32_t) views.size(); }
    uint32_t get_n_applied() const { return n_applied; }

    // I1: 읽기 전용·멱등. 부작용 있는 현행 reader 직접 호출 금지.
    const llama_memplan_view * select(uint32_t ub_idx) const;

    // I2: apply 정확히 1회 · canonical 순서 0→K-1
    bool apply(uint32_t ub_idx);

    bool commit();
    bool abort();

    // 관측 상태 다이제스트 (계열 전체 metadata) — 등가성/무변이 판정축
    uint64_t observe_digest() const;

    //
    // test-only 결함 주입 (SPEC §4 ⓙⓚ — composite 원자성 검증용)
    //   sub: 0 = 없음, 1 = 첫 번째 하위, 2 = 두 번째 하위
    //
    void test_inject_apply_fail(int32_t ub_idx, int32_t sub) { inj_ub = ub_idx; inj_sub = sub; }

    // 복원 불가 형상(I3 후단 — seq fail-close clear) 경로 개봉용
    void test_inject_restore_fail(bool on) { inj_restore = on; }

    // shadow 캡처 도중(= 이미 부분 변이가 일어난 뒤) 실패 경로 개봉용 — ⓖ 무변이 증명.
    // prepare() 가 static 팩토리이므로 훅도 static 이다(-1 = 비활성).
    static void test_set_prepare_fail_at(int32_t ub_idx);

    // 결정론적 std::bad_alloc 주입(-1 = 비활성).
    //   0 = origin snapshot · 1 = phase 1.5 선할당 · 2 = shadow 뒤 복원 경로
    static void test_set_alloc_fail_at(int32_t point);

    // 지속 OOM 모사 — 여러 지점을 동시에 무장한다(bit N = 지점 N).
    //   지점 3 = set_fail 의 실패 문자열 대입(이중 실패 경로 개봉용)
    static void test_set_alloc_fail_points(uint32_t mask);

    // allocation preflight(§6 Q-B) 실패 경로 개봉용 — plan metadata 상한(바이트).
    // ★production 기본값은 무제한(UINT64_MAX) — SPEC §6 Q-B "별도 정적 상한 불요".
    static void test_set_alloc_budget(uint64_t bytes);

    // 호출 순서 trace(test-only). enable 시 apply 경로가 하위 변이 순서를 기록한다.
    void test_trace_enable(bool on) { trace_on = on; trace.clear(); }

    const std::vector<int32_t> & test_trace() const { return trace; }

    // preflight 가 산출한 plan metadata 상계(바이트) — 관측용
    uint64_t get_preflight_bytes() const { return preflight_bytes; }

    // ★shadow(변이 구간)가 새 할당 없이 달렸는가 — 선할당 직후와 shadow 직후의
    // 용량 합계가 같으면 재할당이 없었다는 뜻이다(r25 수리 2b-1 의 직접 오라클).
    bool     shadow_was_realloc_free() const { return shadow_cap_sig == prealloc_cap_sig; }
    uint64_t get_prealloc_cap_sig()    const { return prealloc_cap_sig; }

    // 비종결 소멸에 의한 자동 abort 누적 횟수 (fail-close 기록 관측용)
    static uint64_t test_destruct_abort_count();
    static void     test_reset_counters();

private:
    llama_memplan() = default;

    struct kv_snapshot {
        llama_kv_cache * kv = nullptr;
        llama_kv_cache::memplan_state_t st;
    };

    struct rs_snapshot {
        llama_memory_recurrent * rs = nullptr;
        std::vector<llama_memory_recurrent::mem_cell> cells;
        uint32_t used = 0;
        uint32_t head = 0;
        uint32_t n    = 0;
        int32_t  rs_z = -1;
        std::vector<uint32_t> rs_idx;
    };

    // prepare 직전 상태 전건 (I3 복원 완전성의 기준)
    struct origin_t {
        std::vector<kv_snapshot> kv;
        std::vector<rs_snapshot> rs;
    };

    void     snapshot_take(origin_t & dst) const;
    void     snapshot_restore(const origin_t & src);
    bool     seq_fail_close_clear();

    // allocation preflight (§6 Q-B · r24 수리 2) — shadow apply 시작 전에
    // 전 capture 표면의 크기를 계산·overflow 검사하고 상한과 대조한다.
    bool     preflight(const std::vector<llama_ubatch> & ubs,
                       const std::vector<llama_ubatch> & ubs_raw);

    // phase 1.5 (r25 수리 2b) — shadow 시작 전에 전 캡처 표면의 저장소를 **실제로** 확보한다.
    // 이후 shadow 는 확보된 용량 안에서 쓰기만 하므로 변이 뒤 할당 실패가 생기지 않는다.
    bool     prealloc(const std::vector<llama_ubatch> & ubs);

    uint64_t origin_bytes() const;

    // 선할당된 전 벡터의 capacity 합계(재할당 감지용)
    uint64_t cap_signature() const;

    void     trace_push(llama_memplan_event ev) {
        if (trace_on) {
            trace.push_back((int32_t) ev);
        }
    }

    bool     capture_kv_view(llama_kv_cache * kv, const llama_kv_cache::slot_info & sinfo,
                             const llama_ubatch & ub, uint32_t n_kv, llama_memplan_kv_view & out) const;

    bool     apply_kv   (uint32_t ub_idx);
    bool     apply_hyb  (uint32_t ub_idx);
    bool     apply_dsv4 (uint32_t ub_idx);

    // ★실질 noexcept (r26 수리 2c-1): enum 보존이 우선이고 문자열 기록은 부가 정보다.
    //   지속 OOM 에서 fail_note 대입이 다시 throw 하면 복원 경로에 도달하지 못하므로
    //   안에서 삼킨다.
    void     set_fail(llama_memplan_fail f, const char * note) noexcept;

    // 대상 memory
    llama_memory_i * mem = nullptr;

    llama_kv_cache         * m_kv   = nullptr;   // ① 또는 ② 의 attn 하위
    llama_memory_recurrent * m_rs   = nullptr;   // ② 의 recurrent 하위
    llama_kv_cache_dsv4    * m_dsv4 = nullptr;   // ③
    llama_kv_cache         * m_raw_base = nullptr;
    llama_kv_cache         * m_raw_swa  = nullptr;

    llama_memplan_kind  kind  = LLAMA_MEMPLAN_KIND_NONE;
    llama_memplan_state state = LLAMA_MEMPLAN_STATE_FAILED_PREPARE;
    llama_memplan_fail  fail  = LLAMA_MEMPLAN_FAIL_NONE;

    std::string fail_note;

    std::vector<llama_memplan_view> views;

    uint32_t n_applied = 0;
    bool     causal    = true;

    origin_t origin;

    uint64_t preflight_bytes = 0;
    uint64_t prealloc_cap_sig = 0;
    uint64_t shadow_cap_sig   = 0;

    int32_t inj_ub      = -1;
    int32_t inj_sub     = 0;
    bool    inj_restore = false;

    bool                 trace_on = false;
    std::vector<int32_t> trace;
};

using llama_memplan_ptr = std::unique_ptr<llama_memplan>;

#endif // __cplusplus && !LLAMA_MEMPLAN_C_ONLY
