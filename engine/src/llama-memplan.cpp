// ===========================================================================
// MoE-Direct 레버 A 단계 1 · S1 — immutable ubatch memory plan (engine + selfcheck)
//
// 권위 사양: bench/techdev/superbatch_s1/SPEC_S1_MEMPLAN.md v1.0(동결 · r23 [ACCEPT])
// 1차 소스 지도: bench/techdev/superbatch_s1/SURVEY_MEMORY_SURFACES.md
//
// 이 파일은 두 부분으로 나뉜다.
//   [A] plan 엔진 — prepare/select/apply/commit/abort + 상태기계 + 동결 뷰 캡처
//   [B] selfcheck 픽스처 — SPEC §4 ⓐ~ⓛ × 3계열(standard KV · hybrid · DSV4)
//
// [B] 가 이 파일에 있는 이유(재량 선택): 이 빌드는 BUILD_SHARED_LIBS=ON 이고
//   llama_kv_cache / llama_memory_recurrent / llama_kv_cache_dsv4 등 내부 C++ 심볼은
//   llama.dll 에서 내보내지지 않는다(llama.lib 확인: `llama_kv_cache@@` 심볼 0건).
//   따라서 픽스처가 실 memory 객체를 세우려면 라이브러리 안에서 돌아야 한다.
//   PASS/FAIL 회계·섹션 명명·총계는 호출자(moe-direct-selftest)가 그대로 소유한다.
// ===========================================================================

#include "llama-memplan.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-kv-cache-iswa.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// ===========================================================================
// [A] plan 엔진
// ===========================================================================

namespace {

uint64_t   g_destruct_abort = 0;
int32_t    g_inj_prep_ub    = -1;   // test-only: shadow 캡처 중간 실패 주입 지점
uint32_t   g_inj_alloc_mask  = 0;   // bit N = 지점 N (0=origin 1=prealloc 2=restore 3=set_fail note)
                                    // 여러 지점을 동시에 무장하면 "allocator 가 계속 고갈된" 지속 OOM 이 된다.

// 지정 지점에서 실제 할당 실패를 흉내 낸다(fail-close 경로 개봉용).
void maybe_throw_alloc(int32_t point) {
    if (point >= 0 && point < 32 && ((g_inj_alloc_mask >> point) & 1u) != 0u) {
        throw std::bad_alloc();
    }
}

bool comp_plan_equal(const llama_kv_cache_dsv4_context::comp_plan & a,
                     const llama_kv_cache_dsv4_context::comp_plan & b) {
    return a.state_pos               == b.state_pos               &&
           a.state_persist_src_idxs  == b.state_persist_src_idxs  &&
           a.state_persist_dst_idxs  == b.state_persist_dst_idxs  &&
           a.state_read_idxs         == b.state_read_idxs         &&
           a.state_write_idxs        == b.state_write_idxs        &&
           a.state_write_pos         == b.state_write_pos         &&
           a.n_visible               == b.n_visible               &&
           a.n_stream                == b.n_stream                &&
           a.n_kv                    == b.n_kv;
}

// ★SPEC §6 Q-B: "별도 정적 상한은 불요하되 allocation preflight + 실패 시 무변이 FAILED_PREPARE 는 필수".
//   따라서 production 기본값은 무제한이고, 상한은 test 훅으로만 주입한다(r24 수리 2).
const uint64_t MEMPLAN_ALLOC_BUDGET_UNLIMITED = UINT64_MAX;
uint64_t   g_alloc_budget = MEMPLAN_ALLOC_BUDGET_UNLIMITED;

// 체크드 덧셈 — 크기 계산 overflow 를 fail-close 로 만든다.
bool add_checked(uint64_t & acc, uint64_t term) {
    if (acc > UINT64_MAX - term) {
        return false;
    }
    acc += term;
    return true;
}

bool mul_checked(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > UINT64_MAX/a) {
        return false;
    }
    out = a*b;
    return true;
}

// 셀 1개가 실제로 쓰는 힙까지 감안한 보수적 상수(mem_cell 은 std::set 을 품는다)
const uint64_t MEMPLAN_CELL_OVERHEAD = 64;

// FNV-1a 64
struct fnv64 {
    uint64_t h = 0xcbf29ce484222325ull;

    void u64(uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (uint64_t) ((v >> (8*b)) & 0xff);
            h *= 0x100000001b3ull;
        }
    }

    void i32(int32_t v) { u64((uint64_t) (uint32_t) v); }
    void u32(uint32_t v) { u64((uint64_t) v); }

    void bytes(const void * p, size_t n) {
        const uint8_t * q = (const uint8_t *) p;
        for (size_t i = 0; i < n; ++i) {
            h ^= (uint64_t) q[i];
            h *= 0x100000001b3ull;
        }
    }
};

// 호스트 버퍼 위의 임시 텐서 — set_input_* 를 실제로 돌려 동결본을 값으로 뽑는다.
// (재계산 구현을 새로 쓰지 않으므로 그래프-가시 값과 byte 동일이 구조적으로 보장된다)
struct host_scratch {
    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ~host_scratch() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }

    bool init(size_t n_tensors) {
        ggml_init_params p = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(n_tensors + 4) + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(p);

        return ctx != nullptr;
    }

    bool alloc() {
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());

        return buf != nullptr;
    }
};

void grab_i64(const ggml_tensor * t, std::vector<int64_t> & out) {
    if (!t) {
        out.clear();
        return;
    }

    const int64_t n = ggml_nelements(t);
    out.resize((size_t) n);
    std::memcpy(out.data(), t->data, (size_t) n*sizeof(int64_t));
}

void grab_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) {
        out.clear();
        return;
    }

    const int64_t n = ggml_nelements(t);
    out.resize((size_t) n);
    std::memcpy(out.data(), t->data, (size_t) n*sizeof(float));
}

void digest_rs(fnv64 & f, const llama_memory_recurrent * rs) {
    f.u32(rs->size);
    f.u32(rs->used);
    f.u32(rs->head);
    f.u32(rs->n);
    f.i32(rs->rs_z);

    for (const auto & c : rs->cells) {
        f.i32(c.pos);
        f.i32(c.src);
        f.i32(c.src0);
        f.i32(c.tail);
        for (const llama_seq_id s : c.seq_id) {
            f.i32(s);
        }
        f.u64(0xC311EDull);
    }

    for (const uint32_t v : rs->rs_idx) {
        f.u32(v);
    }
}

void digest_sinfo(fnv64 & f, const llama_kv_cache::slot_info & si) {
    f.u32(si.s0);
    f.u32(si.s1);
    for (const auto v : si.strm) {
        f.i32(v);
    }
    for (const auto & row : si.idxs) {
        f.u64(row.size());
        for (const auto v : row) {
            f.u32(v);
        }
    }
}

bool sinfo_equal(const llama_kv_cache::slot_info & a, const llama_kv_cache::slot_info & b) {
    fnv64 fa; digest_sinfo(fa, a);
    fnv64 fb; digest_sinfo(fb, b);

    return fa.h == fb.h;
}

} // namespace

void llama_memplan::test_set_alloc_budget(uint64_t bytes) {
    g_alloc_budget = bytes;
}

uint64_t llama_memplan::test_destruct_abort_count() {
    return g_destruct_abort;
}

void llama_memplan::test_set_prepare_fail_at(int32_t ub_idx) {
    g_inj_prep_ub = ub_idx;
}

void llama_memplan::test_set_alloc_fail_at(int32_t point) {
    g_inj_alloc_mask = point < 0 ? 0u : (1u << point);
}

void llama_memplan::test_set_alloc_fail_points(uint32_t mask) {
    g_inj_alloc_mask = mask;
}

void llama_memplan::test_reset_counters() {
    g_destruct_abort = 0;
    g_alloc_budget    = MEMPLAN_ALLOC_BUDGET_UNLIMITED;
    g_inj_prep_ub     = -1;
    g_inj_alloc_mask = 0;
}

void llama_memplan::set_fail(llama_memplan_fail f, const char * note) noexcept {
    fail = f;   // ★enum 은 무조건 보존된다(무할당)

    if (!note) {
        return;
    }

    // ★r26 수리 2c-1: MSVC 의 SSO 는 15자이고 실패 문구는 그보다 길다 — 지속 OOM 에서는
    //   이 대입이 다시 throw 해 복원 경로까지 도달하지 못할 수 있다. 문자열은 부가 정보이므로
    //   안에서 삼키고 빈 값으로 둔다(std::string::clear() 는 noexcept · 무할당).
    try {
        maybe_throw_alloc(3);
        fail_note = note;
    } catch (...) {
        fail_note.clear();
    }
}

// ---------------------------------------------------------------------------
// snapshot / restore (I3 복원 완전성)
// ---------------------------------------------------------------------------

void llama_memplan::snapshot_take(origin_t & dst) const {
    dst.kv.clear();
    dst.rs.clear();

    for (llama_kv_cache * kv : { m_kv, m_raw_base, m_raw_swa }) {
        if (!kv) {
            continue;
        }

        kv_snapshot s;
        s.kv = kv;
        s.st = kv->memplan_capture();
        dst.kv.push_back(std::move(s));
    }

    if (m_dsv4) {
        for (llama_kv_cache * kv : { m_dsv4->get_csa(), m_dsv4->get_hca(), m_dsv4->get_lid() }) {
            if (!kv) {
                continue;
            }

            kv_snapshot s;
            s.kv = kv;
            s.st = kv->memplan_capture();
            dst.kv.push_back(std::move(s));
        }
    }

    if (m_rs) {
        rs_snapshot s;
        s.rs     = m_rs;
        s.cells  = m_rs->cells;
        s.used   = m_rs->used;
        s.head   = m_rs->head;
        s.n      = m_rs->n;
        s.rs_z   = m_rs->rs_z;
        s.rs_idx = m_rs->rs_idx;
        dst.rs.push_back(std::move(s));
    }
}

void llama_memplan::snapshot_restore(const origin_t & src) {
    for (const auto & s : src.kv) {
        s.kv->memplan_restore(s.st);
    }

    for (const auto & s : src.rs) {
        s.rs->cells  = s.cells;
        s.rs->used   = s.used;
        s.rs->head   = s.head;
        s.rs->n      = s.n;
        s.rs->rs_z   = s.rs_z;
        s.rs->rs_idx = s.rs_idx;
    }
}

// I3 후단: 복원 불가 형상에서는 절반 상태를 남기지 않고 해당 memory 를 명시 clear 한다.
// ★r24 수리 1: `data = true` 로 부른다. compute 는 recurrent `r_l/s_l`(models/delta-net-base.cpp:496,
//   555-558,603)과 DSV4 압축 캐시·compressor state(models/deepseek4.cpp:930-945,982-997,1025-1046)에
//   **물리적으로** 쓴다. metadata 만 지우는 clear 는 stale 물리 행을 남겨(특히 DSV4 는 그래프가
//   덮지 않는 압축 행을 읽는다) 절반 상태가 된다.
bool llama_memplan::seq_fail_close_clear() {
    if (mem) {
        mem->clear(true);
    }

    // ★현행 `llama_memory_recurrent::clear()` 는 cells/head/used/rs_idx 만 되돌리고 `n`·`rs_z` 는
    //   그대로 남긴다(SURVEY Q2 의 "prepare 복원 범위 = cells/used/head 만"과 같은 결).
    //   셀의 `src0` 도 clear() 대상이 아니다. 그대로 두면 "명시 clear" 가 절반 상태로 남아
    //   I3 후단 계약을 어기므로 plan 이 recurrent metadata 를 pristine 형상으로 마저 되돌린다.
    //   (엔진의 clear 의미론 자체는 건드리지 않는다 — S1 은 라이브 경로 무변경)
    if (m_rs) {
        m_rs->cells.assign(m_rs->size, llama_memory_recurrent::mem_cell{});
        m_rs->head = 0;
        m_rs->used = 0;
        m_rs->n    = 0;
        m_rs->rs_z = -1;

        std::fill(m_rs->rs_idx.begin(), m_rs->rs_idx.end(), 0u);
    }

    set_fail(LLAMA_MEMPLAN_FAIL_RESTORE, "restore impossible -> fail-close clear");

    return false;
}

// ---------------------------------------------------------------------------
// allocation preflight (§6 Q-B · r24 수리 2)
//   shadow apply 를 시작하기 전에 전 capture 표면의 크기를 계산하고 overflow·상한을 검사한다.
//   실패 = 아직 어떤 변이도 하지 않은 시점이므로 그대로 무변이 FAILED_PREPARE 가 된다.
// ---------------------------------------------------------------------------

uint64_t llama_memplan::origin_bytes() const {
    uint64_t bytes = 0;

    for (const auto & s : origin.kv) {
        for (const auto & c : s.st.v_cells) {
            bytes += (uint64_t) c.size()*MEMPLAN_CELL_OVERHEAD;
        }
        bytes += (uint64_t) s.st.v_heads.size()*sizeof(uint32_t);
    }

    for (const auto & s : origin.rs) {
        bytes += (uint64_t) s.cells.size()*(sizeof(llama_memory_recurrent::mem_cell) + MEMPLAN_CELL_OVERHEAD);
        bytes += (uint64_t) s.rs_idx.size()*sizeof(uint32_t);
    }

    return bytes;
}

bool llama_memplan::preflight(
        const std::vector<llama_ubatch> & ubs,
        const std::vector<llama_ubatch> & ubs_raw) {
    uint64_t bytes = origin_bytes();

    const auto fail_alloc = [this](const char * why) {
        set_fail(LLAMA_MEMPLAN_FAIL_ALLOC, why);
        return false;
    };

    // n_kv 는 apply 이후에만 확정되지만 상계는 셀 수다(get_n_kv 의 std::min(cells.size(), ...)).
    const uint64_t n_kv_max     = m_kv       ? m_kv->get_size()       : 0;
    const uint64_t n_kv_raw_max = m_raw_swa  ? m_raw_swa->get_size()  : 0;

    for (size_t i = 0; i < ubs.size(); ++i) {
        const llama_ubatch & ub = ubs[i];

        uint64_t term = 0;

        if (!add_checked(bytes, sizeof(llama_memplan_view))) {
            return fail_alloc("preflight overflow (view)");
        }

        if (m_kv && kind != LLAMA_MEMPLAN_KIND_DSV4) {
            // kq mask 상계
            if (!mul_checked(n_kv_max, (uint64_t) ub.n_tokens, term) ||
                !mul_checked(term, sizeof(float), term) ||
                !add_checked(bytes, term)) {
                return fail_alloc("preflight overflow (kq mask)");
            }

            // k_idxs / v_idxs — 실제 빌더가 만드는 정확한 원소 수를 no_alloc 컨텍스트에서 읽는다
            host_scratch hs;
            if (!hs.init(2)) {
                return fail_alloc("preflight scratch init failed");
            }

            ggml_tensor * k_idxs = m_kv->build_input_k_idxs(hs.ctx, ub);
            ggml_tensor * v_idxs = m_kv->build_input_v_idxs(hs.ctx, ub);

            const uint64_t n_idx =
                (uint64_t) (k_idxs ? ggml_nelements(k_idxs) : 0) +
                (uint64_t) (v_idxs ? ggml_nelements(v_idxs) : 0);

            if (!mul_checked(n_idx, sizeof(int64_t), term) || !add_checked(bytes, term)) {
                return fail_alloc("preflight overflow (idxs)");
            }

            // sinfo(idxs+strm)
            if (!mul_checked((uint64_t) ub.n_tokens, sizeof(uint32_t) + sizeof(llama_seq_id), term) ||
                !add_checked(bytes, term) || !add_checked(bytes, 128)) {
                return fail_alloc("preflight overflow (sinfo)");
            }
        }

        if (m_rs) {
            const uint64_t n_cells = m_rs->size;

            if (!mul_checked(n_cells, sizeof(llama_memory_recurrent::mem_cell) + MEMPLAN_CELL_OVERHEAD, term) ||
                !add_checked(bytes, term)) {
                return fail_alloc("preflight overflow (rs cells)");
            }

            if (!mul_checked(n_cells, sizeof(int32_t), term) || !add_checked(bytes, term)) {
                return fail_alloc("preflight overflow (s_copy)");
            }

            // rs_idx_pre + rs_idx_post
            if (!mul_checked((uint64_t) m_rs->rs_idx.size(), 2*sizeof(uint32_t), term) ||
                !add_checked(bytes, term)) {
                return fail_alloc("preflight overflow (rs_idx)");
            }
        }

        if (kind == LLAMA_MEMPLAN_KIND_DSV4) {
            const uint64_t n_tok_raw = i < ubs_raw.size() ? ubs_raw[i].n_tokens : ub.n_tokens;

            if (!mul_checked(n_kv_raw_max, (uint64_t) ub.n_tokens, term) ||
                !mul_checked(term, sizeof(float), term) ||
                !add_checked(bytes, term)) {
                return fail_alloc("preflight overflow (dsv4 raw mask)");
            }

            // raw base/swa write + swa read + csa/hca/lid sinfo
            if (!mul_checked(n_tok_raw, 6*(sizeof(uint32_t) + sizeof(llama_seq_id)), term) ||
                !add_checked(bytes, term) || !add_checked(bytes, 768)) {
                return fail_alloc("preflight overflow (dsv4 sinfos)");
            }

            // comp_plan 3종 상계 — 각 plan 의 벡터 총량은 n_tokens 선형(중첩 압축의 prev/cur 2배 포함)
            if (!mul_checked((uint64_t) ub.n_tokens, 3*8*sizeof(int64_t), term) ||
                !add_checked(bytes, term) || !add_checked(bytes, 3*512)) {
                return fail_alloc("preflight overflow (dsv4 comp plan)");
            }
        }
    }

    preflight_bytes = bytes;

    if (bytes > g_alloc_budget) {
        return fail_alloc("plan metadata exceeds budget");
    }

    return true;
}

// ---------------------------------------------------------------------------
// phase 1.5 — 전 캡처 표면 저장소 실선할당 (r25 수리 2b)
//   ★shadow(=변이 구간) 중에는 새 할당이 없어야 한다. 여기서 상계 용량으로 실제 확보해 두고,
//     shadow 는 그 안에서 resize(축소)·동일 크기 대입만 한다.
//   ★잔여 할당 1건(불가피): `mem_cell::seq_id` 는 std::set 이라 셀 복사 시 노드 할당이 생긴다.
//     벡터 저장소 자체는 여기서 확보하므로 남는 것은 셀당 seq 집합 노드뿐이다.
// ---------------------------------------------------------------------------
uint64_t llama_memplan::cap_signature() const {
    uint64_t acc = views.capacity();

    for (const auto & v : views) {
        acc += v.kv.k_idxs .capacity();
        acc += v.kv.v_idxs .capacity();
        acc += v.kv.kq_mask.capacity();

        acc += v.kv.sinfo.strm.capacity();
        acc += v.kv.sinfo.idxs.capacity();
        for (const auto & row : v.kv.sinfo.idxs) {
            acc += row.capacity();
        }

        acc += v.rs.cells       .capacity();
        acc += v.rs.s_copy      .capacity();
        acc += v.rs.rs_idx_pre  .capacity();
        acc += v.rs.rs_idx_post .capacity();

        acc += v.dsv4.raw_kq_mask.capacity();
    }

    return acc;
}

bool llama_memplan::prealloc(const std::vector<llama_ubatch> & ubs) {
    maybe_throw_alloc(1);

    const size_t n_kv_ub     = m_kv      ? m_kv->get_size()      : 0;
    const size_t n_kv_raw_ub = m_raw_swa ? m_raw_swa->get_size() : 0;

    for (size_t i = 0; i < views.size() && i < ubs.size(); ++i) {
        llama_memplan_view & v  = views[i];
        const llama_ubatch & ub = ubs[i];

        if (v.has_kv && m_kv) {
            host_scratch hs;
            if (!hs.init(2)) {
                set_fail(LLAMA_MEMPLAN_FAIL_ALLOC, "prealloc scratch init failed");
                return false;
            }

            ggml_tensor * k_idxs = m_kv->build_input_k_idxs(hs.ctx, ub);
            ggml_tensor * v_idxs = m_kv->build_input_v_idxs(hs.ctx, ub);

            v.kv.k_idxs .resize(k_idxs ? (size_t) ggml_nelements(k_idxs) : 0);
            v.kv.v_idxs .resize(v_idxs ? (size_t) ggml_nelements(v_idxs) : 0);
            v.kv.kq_mask.resize(n_kv_ub*(size_t) ub.n_tokens);
        }

        if (v.has_rs && m_rs) {
            v.rs.cells        = m_rs->cells;                   // 셀 벡터 저장소 확보(크기 동일)
            v.rs.s_copy       .resize(m_rs->size);
            v.rs.rs_idx_pre   .resize(m_rs->rs_idx.size());
            v.rs.rs_idx_post  .resize(m_rs->rs_idx.size());
        }

        if (v.has_dsv4) {
            v.dsv4.raw_kq_mask.resize(n_kv_raw_ub*(size_t) ub.n_tokens);
        }
    }

    prealloc_cap_sig = cap_signature();
    shadow_cap_sig   = prealloc_cap_sig;

    return true;
}

// ---------------------------------------------------------------------------
// observe digest
// ---------------------------------------------------------------------------

uint64_t llama_memplan::observe_digest() const {
    fnv64 f;

    f.u32((uint32_t) kind);

    if (m_kv) {
        f.u64(m_kv->memplan_digest());
    }

    if (m_rs) {
        digest_rs(f, m_rs);
    }

    if (m_dsv4) {
        f.u64(m_raw_base ? m_raw_base->memplan_digest() : 0);
        f.u64(m_raw_swa  ? m_raw_swa->memplan_digest()  : 0);
        f.u64(m_dsv4->get_csa()->memplan_digest());
        f.u64(m_dsv4->get_hca()->memplan_digest());
        f.u64(m_dsv4->get_lid()->memplan_digest());
    }

    return f.h;
}

// ---------------------------------------------------------------------------
// 동결 뷰 캡처
// ---------------------------------------------------------------------------

bool llama_memplan::capture_kv_view(
        llama_kv_cache * kv,
        const llama_kv_cache::slot_info & sinfo,
        const llama_ubatch & ub,
        uint32_t n_kv,
        llama_memplan_kv_view & out) const {
    // ★out.sinfo 는 phase 1 에서 확정된다. 호출부가 out.sinfo 를 그대로 넘기므로 여기서의
    //   대입은 자기대입이었다 — 삭제해 증명을 단순화한다(r26 B).
    out.n_kv = n_kv;

    const uint32_t n_tokens  = ub.n_tokens;
    const uint32_t n_stream_g = kv->get_n_stream() == 1 ? 1 : std::max<uint32_t>(1, ub.n_seqs_unq);

    if (n_tokens == 0 || n_kv == 0 || (n_tokens % n_stream_g) != 0) {
        return false;
    }

    host_scratch hs;
    if (!hs.init(3)) {
        return false;
    }

    ggml_tensor * k_idxs = kv->build_input_k_idxs(hs.ctx, ub);
    ggml_tensor * v_idxs = kv->build_input_v_idxs(hs.ctx, ub);
    ggml_tensor * mask   = ggml_new_tensor_4d(hs.ctx, GGML_TYPE_F32, n_kv, n_tokens/n_stream_g, 1, n_stream_g);

    if (!k_idxs || !v_idxs || !mask || !hs.alloc()) {
        return false;
    }

    kv->set_input_k_idxs (k_idxs, &ub, sinfo);
    kv->set_input_v_idxs (v_idxs, &ub, sinfo);
    kv->set_input_kq_mask(mask,   &ub, causal);

    grab_i64(k_idxs, out.k_idxs);
    grab_i64(v_idxs, out.v_idxs);
    grab_f32(mask,   out.kq_mask);

    for (int d = 0; d < 4; ++d) {
        out.kq_ne[d] = mask->ne[d];
    }

    return true;
}

// ---------------------------------------------------------------------------
// prepare
// ---------------------------------------------------------------------------

std::unique_ptr<llama_memplan> llama_memplan::prepare(
        llama_memory_i     * mem,
        llama_batch_allocr & balloc,
        uint32_t             n_ubatch,
        bool                 embd_all,
        bool                 causal_attn) {
    std::unique_ptr<llama_memplan> plan(new llama_memplan());

    plan->mem    = mem;
    plan->causal = causal_attn;
    plan->state  = LLAMA_MEMPLAN_STATE_FAILED_PREPARE;

    if (!mem) {
        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "null memory");
        return plan;
    }

    // --- 계열 판별 -------------------------------------------------------
    if (auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem)) {
        plan->kind = LLAMA_MEMPLAN_KIND_HYBRID;
        plan->m_kv = hyb->get_mem_attn();
        plan->m_rs = hyb->get_mem_recr();
    } else if (auto * d4 = dynamic_cast<llama_kv_cache_dsv4 *>(mem)) {
        plan->kind       = LLAMA_MEMPLAN_KIND_DSV4;
        plan->m_dsv4     = d4;
        plan->m_raw_base = d4->get_raw()->get_base();
        plan->m_raw_swa  = d4->get_raw()->get_swa();
    } else if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        plan->kind = LLAMA_MEMPLAN_KIND_KV;
        plan->m_kv = kv;
    } else {
        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "unsupported memory kind");
        return plan;
    }

    // --- phase 0: prepare 직전 전체 관측 스냅샷 (§3 ① r22 수정 2) ----------
    //   ★r24 수리 2: origin snapshot 자체가 힙 할당이므로 예외 경계 안에 둔다.
    //   여기서 실패해도 아직 어떤 변이도 없으므로 그대로 무변이 FAILED_PREPARE 다.
    try {
        maybe_throw_alloc(0);
        plan->snapshot_take(plan->origin);
    } catch (const std::bad_alloc &) {
        plan->set_fail(LLAMA_MEMPLAN_FAIL_ALLOC, "bad_alloc during origin snapshot");
        plan->state = LLAMA_MEMPLAN_STATE_FAILED_PREPARE;
        return plan;
    } catch (const std::exception & e) {
        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, e.what());
        plan->state = LLAMA_MEMPLAN_STATE_FAILED_PREPARE;
        return plan;
    }

    const uint64_t digest_pre = plan->observe_digest();

    bool ok = true;

    try {
        // --- phase 1: ubatch 열거(무변이) + allocation preflight -------------
        //   apply 를 부르지 않으므로 memory 는 변하지 않는다(init_batch 내부 드라이런은
        //   스스로 되감는다). 여기서 전 capture 표면 크기를 먼저 확정한다.
        {
            std::vector<llama_ubatch> ubs;
            std::vector<llama_ubatch> ubs_raw;

            llama_memory_context_ptr mctx0 = mem->init_batch(balloc, n_ubatch, embd_all);

            if (!mctx0 || llama_memory_status_is_fail(mctx0->get_status()) ||
                    mctx0->get_status() == LLAMA_MEMORY_STATUS_NO_UPDATE) {
                plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "init_batch failed");
                ok = false;
            } else {
                // ★apply 를 부르지 않으므로 아래 전건은 무변이 관측이다. 여기서 views 를 미리
                //   구성해 두면(=저장소 확보 + apply 무관 표면 확정) shadow 는 apply 의존 필드만
                //   기확보 용량 안에 채우면 된다(r25 수리 2b).
                uint32_t i = 0;

                while (ok) {
                    ubs.push_back(mctx0->get_ubatch());

                    plan->views.emplace_back();

                    llama_memplan_view & v = plan->views.back();

                    v.ub_idx = i;
                    v.ubatch = ubs.back();

                    if (plan->kind == LLAMA_MEMPLAN_KIND_KV || plan->kind == LLAMA_MEMPLAN_KIND_HYBRID) {
                        const llama_kv_cache_context * kctx =
                            plan->kind == LLAMA_MEMPLAN_KIND_HYBRID
                                ? static_cast<const llama_memory_hybrid_context *>(mctx0.get())->get_attn()
                                : static_cast<const llama_kv_cache_context *>(mctx0.get());

                        if (i >= kctx->get_sinfos().size()) {
                            plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "sinfo index out of range");
                            ok = false;
                            break;
                        }

                        v.kv.sinfo = kctx->get_sinfos()[i];
                        v.has_kv   = true;
                    }

                    if (plan->kind == LLAMA_MEMPLAN_KIND_HYBRID) {
                        v.has_rs = true;
                    }

                    if (plan->kind == LLAMA_MEMPLAN_KIND_DSV4) {
                        const auto * dctx = static_cast<const llama_kv_cache_dsv4_context *>(mctx0.get());
                        const auto * raw  = dctx->get_raw();

                        if (i >= raw->get_sinfos_write().size() || i >= raw->get_sinfos_read().size() ||
                            i >= raw->get_ubatches_write().size() ||
                            !raw->get_base_ctx() || i >= raw->get_base_ctx()->get_sinfos().size()) {
                            plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "dsv4 sinfo index out of range");
                            ok = false;
                            break;
                        }

                        auto & d = v.dsv4;

                        d.sinfo_raw_base_write = raw->get_base_ctx()->get_sinfos()[i];
                        d.sinfo_raw_swa_write  = raw->get_sinfos_write()[i];
                        d.sinfo_raw_swa_read   = raw->get_sinfos_read()[i];

                        v.ubatch_raw = raw->get_ubatches_write()[i];

                        if (i < dctx->get_csa()->get_sinfos().size()) { d.sinfo_csa = dctx->get_csa()->get_sinfos()[i]; }
                        if (i < dctx->get_hca()->get_sinfos().size()) { d.sinfo_hca = dctx->get_hca()->get_sinfos()[i]; }
                        if (i < dctx->get_lid()->get_sinfos().size()) { d.sinfo_lid = dctx->get_lid()->get_sinfos()[i]; }

                        d.plan_csa = dctx->get_csa_plan();
                        d.plan_hca = dctx->get_hca_plan();
                        d.plan_lid = dctx->get_lid_plan();

                        const auto identity = [](const llama_dsv4_comp_state * st) {
                            llama_memplan_dsv4_view::comp_identity id;
                            if (st) {
                                id.state      = (const void *) st;
                                id.ratio      = st->get_ratio();
                                id.state_size = st->get_state_size();
                                id.n_stream   = st->get_n_stream();
                            }
                            return id;
                        };

                        d.id_csa = identity(dctx->get_csa_state());
                        d.id_hca = identity(dctx->get_hca_state());
                        d.id_lid = identity(dctx->get_lid_state());

                        v.has_dsv4 = true;
                    }

                    if (!mctx0->next()) {
                        break;
                    }

                    ++i;
                }

                if (plan->kind == LLAMA_MEMPLAN_KIND_DSV4) {
                    ubs_raw = static_cast<const llama_kv_cache_dsv4_context *>(mctx0.get())
                                ->get_raw()->get_ubatches_write();
                }
            }

            // phase 1 잔여(예: recurrent n/rs_z 드리프트) 제거 — phase 2 는 prepare 직전에서 다시 시작
            plan->snapshot_restore(plan->origin);

            if (ok && !plan->preflight(ubs, ubs_raw)) {
                ok = false;   // FAIL_ALLOC — 아직 무변이
            }

            // --- phase 1.5: 전 캡처 표면 저장소 실선할당 (shadow 진입 전) ---
            if (ok && !plan->prealloc(ubs)) {
                ok = false;   // FAIL_ALLOC — 아직 무변이
            }
        }

        // --- phase 2: 배치 전체 선계산 + shadow 캡처 --------------------------
        // 현행 init_batch 가 "적용 후 되감기" 드라이런을 내부에서 수행하므로,
        // S1 은 그 위에 소유·수명·1회성·순서 강제를 얹는다.
        llama_memory_context_ptr mctx = ok ? mem->init_batch(balloc, n_ubatch, embd_all)
                                           : llama_memory_context_ptr();

        if (!ok) {
            // preflight 단계에서 이미 결정됨
        } else if (!mctx || llama_memory_status_is_fail(mctx->get_status()) ||
                mctx->get_status() == LLAMA_MEMORY_STATUS_NO_UPDATE) {
            plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "init_batch failed");
            ok = false;
        } else {
            // --- shadow pass: apply → 동결 뷰 캡처 → next ---------------
            // 그래프-가시 값(n_kv·kq mask·s_copy 등)은 해당 ubatch 의 apply 직후에만
            // 관측 가능하다. 그림자 실행으로 값을 떠 두고, 뒤에서 전량 복원한다.
            uint32_t i = 0;

            while (i < plan->views.size()) {
                llama_memplan_view & v = plan->views[i];

                if (!mctx->apply()) {
                    plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "shadow apply failed");
                    ok = false;
                    break;
                }

                if (plan->kind == LLAMA_MEMPLAN_KIND_KV || plan->kind == LLAMA_MEMPLAN_KIND_HYBRID) {
                    const llama_kv_cache_context * kctx =
                        plan->kind == LLAMA_MEMPLAN_KIND_HYBRID
                            ? static_cast<const llama_memory_hybrid_context *>(mctx.get())->get_attn()
                            : static_cast<const llama_kv_cache_context *>(mctx.get());

                    // phase 1 에서 뜬 계획 좌표와 동일해야 한다(선할당 근거의 전제 — 어긋나면 fail-close)
                    if (i >= kctx->get_sinfos().size() || !sinfo_equal(v.kv.sinfo, kctx->get_sinfos()[i])) {
                        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "sinfo drift between enumeration and shadow");
                        ok = false;
                        break;
                    }

                    if (!plan->capture_kv_view(plan->m_kv, v.kv.sinfo, v.ubatch,
                                kctx->get_n_kv(), v.kv)) {
                        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "kv view capture failed");
                        ok = false;
                        break;
                    }
                }

                if (plan->kind == LLAMA_MEMPLAN_KIND_HYBRID) {
                    llama_memory_recurrent * rs = plan->m_rs;

                    v.rs.cells = rs->cells;
                    v.rs.used  = rs->used;
                    v.rs.head  = rs->head;
                    v.rs.n     = rs->n;
                    v.rs.rs_z  = rs->rs_z;

                    v.rs.rs_idx_pre = rs->rs_idx;

                    // ★s_copy 는 소비형 읽기 — shadow 가 실제 set_input 순서 그대로 소비하고
                    //   post-consumption 상태를 다음 ubatch·최종 APPLIED 로 승계한다.
                    {
                        llama_memory_recurrent_context rctx(rs, std::vector<llama_ubatch>{});

                        v.rs.s_copy.resize(rs->n);
                        for (uint32_t j = 0; j < rs->n; ++j) {
                            v.rs.s_copy[j] = rctx.s_copy((int) j);
                        }
                    }

                    v.rs.rs_idx_post = rs->rs_idx;
                }

                if (plan->kind == LLAMA_MEMPLAN_KIND_DSV4) {
                    const auto * dctx = static_cast<const llama_kv_cache_dsv4_context *>(mctx.get());
                    const auto * raw  = dctx->get_raw();

                    if (i >= raw->get_sinfos_write().size() || i >= raw->get_sinfos_read().size() ||
                        i >= raw->get_ubatches_write().size() ||
                        !raw->get_base_ctx() || i >= raw->get_base_ctx()->get_sinfos().size()) {
                        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "dsv4 sinfo index out of range");
                        ok = false;
                        break;
                    }

                    auto & d = v.dsv4;

                    // phase 1 에서 뜬 값과 동일해야 한다(선할당 근거의 전제).
                    // ★검사 범위 = raw sinfo 3종 + comp_plan 3종. CSA/HCA/LID sinfo 3종은 phase 1
                    //   전용이고 shadow 에서 쓰지 않으므로 할당 차단점이 아니다(r26 B 정정).
                    if (!sinfo_equal(d.sinfo_raw_base_write, raw->get_base_ctx()->get_sinfos()[i]) ||
                        !sinfo_equal(d.sinfo_raw_swa_write,  raw->get_sinfos_write()[i])          ||
                        !sinfo_equal(d.sinfo_raw_swa_read,   raw->get_sinfos_read()[i])           ||
                        !comp_plan_equal(d.plan_csa, dctx->get_csa_plan())                        ||
                        !comp_plan_equal(d.plan_hca, dctx->get_hca_plan())                        ||
                        !comp_plan_equal(d.plan_lid, dctx->get_lid_plan())) {
                        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "dsv4 plan drift between enumeration and shadow");
                        ok = false;
                        break;
                    }

                    d.raw_n_kv = raw->get_n_kv();

                    // raw visibility 동결본 (DSV4 raw 는 K-only — v_idxs 는 그래프 입력이 아니다)
                    {
                        const llama_kv_cache * kv_swa = raw->get_kv_swa();

                        const uint32_t n_tokens   = v.ubatch.n_tokens;
                        const uint32_t n_stream_g = kv_swa->get_n_stream() == 1
                            ? 1 : std::max<uint32_t>(1, v.ubatch.n_seqs_unq);

                        host_scratch hs;

                        if (!hs.init(1) || n_tokens == 0 || d.raw_n_kv == 0 || (n_tokens % n_stream_g) != 0) {
                            plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "dsv4 raw mask shape invalid");
                            ok = false;
                            break;
                        }

                        ggml_tensor * mask = ggml_new_tensor_4d(hs.ctx, GGML_TYPE_F32,
                                d.raw_n_kv, n_tokens/n_stream_g, 1, n_stream_g);

                        if (!mask || !hs.alloc()) {
                            plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "dsv4 raw mask alloc failed");
                            ok = false;
                            break;
                        }

                        raw->set_input_kq_mask(mask, &v.ubatch, plan->causal);

                        grab_f32(mask, d.raw_kq_mask);

                        for (int dd = 0; dd < 4; ++dd) {
                            d.raw_kq_ne[dd] = mask->ne[dd];
                        }
                    }

                }

                // ★shadow 도중 실패 주입(test-only) — 이 시점엔 ubatch 0..i 의 변이가 이미
                //   memory 에 얹혀 있으므로, 뒤따르는 실복원이 "실패 = 무변이"를 실제로 증명한다.
                if (g_inj_prep_ub >= 0 && (int32_t) i >= g_inj_prep_ub) {
                    plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "injected mid-shadow capture failure");
                    ok = false;
                    break;
                }

                if (!mctx->next()) {
                    break;
                }

                ++i;
            }

            // ★shadow 가 기확보 용량 안에서만 돌았는가를 실측한다(r25 수리 2b-1)
            plan->shadow_cap_sig = plan->cap_signature();
        }
    } catch (const std::bad_alloc &) {
        plan->set_fail(LLAMA_MEMPLAN_FAIL_ALLOC, "bad_alloc during plan capture");
        ok = false;
    } catch (const std::exception & e) {
        plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, e.what());
        ok = false;
    }

    // --- 실복원 + 대조: 하나의 no-throw `restore-or-clear` 경로 (r26 수리 2c-2) -------
    // ★1차 복원·재대조·2차 복원·clear 후단을 한 경로로 묶고, 그 안에서 예외가 절대 탈출하지
    //   않게 한다. 복원은 vector/set 대입(=할당)이라 지속 OOM 에서 실패할 수 있고, 예외가
    //   새어 나가면 부분 변이가 그대로 남는다. clear 경로는 추가 힙 할당 없이 돌며
    //   (kv_cells::reset / recurrent clear / cells.assign(동일 크기) / buffer memset),
    //   유일한 잠재 할당이던 set_fail 의 문자열 대입은 2c-1 로 실질 noexcept 화됐다.
    //
    //   반환값: true = 복원 성공·무변이 확인(정상 진행) / false = FAILED_PREPARE 로 종결
    const auto restore_or_clear = [&plan, digest_pre]() noexcept -> bool {
        bool restored = true;

        try {
            maybe_throw_alloc(2);
            plan->snapshot_restore(plan->origin);
        } catch (...) {
            restored = false;
        }

        if (restored) {
            uint64_t digest_post = 0;

            try {
                digest_post = plan->observe_digest();
            } catch (...) {
                restored = false;
            }

            if (restored && digest_post == digest_pre) {
                return true;
            }

            if (restored) {
                // 검출만 하고 반환하면 "실패 = 무변이" 계약 위반 — 실제 복원을 한 번 더 하고 재대조
                try {
                    plan->snapshot_restore(plan->origin);
                    digest_post = plan->observe_digest();
                } catch (...) {
                    restored = false;
                }

                if (restored && digest_post == digest_pre) {
                    plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE_DRIFT, "prepare mutated observable state");
                    return false;
                }
            }
        }

        // 복원 불가(또는 재대조 불일치 지속) → I3 후단: 절반 상태 금지, 명시 clear
        try {
            plan->seq_fail_close_clear();
        } catch (...) {
            plan->fail = LLAMA_MEMPLAN_FAIL_RESTORE;   // 문자열 없이 사유 enum 만
        }

        return false;
    };

    if (!restore_or_clear()) {
        plan->views.clear();
        plan->state = LLAMA_MEMPLAN_STATE_FAILED_PREPARE;

        return plan;
    }

    if (!ok || plan->views.empty()) {
        if (plan->fail == LLAMA_MEMPLAN_FAIL_NONE) {
            plan->set_fail(LLAMA_MEMPLAN_FAIL_PREPARE, "empty plan");
        }

        plan->views.clear();
        plan->state = LLAMA_MEMPLAN_STATE_FAILED_PREPARE;

        return plan;
    }

    plan->state = LLAMA_MEMPLAN_STATE_PREPARED;

    return plan;
}

// ---------------------------------------------------------------------------
// select — I1: 읽기 전용·멱등·임의 순서
// ---------------------------------------------------------------------------

const llama_memplan_view * llama_memplan::select(uint32_t ub_idx) const {
    if (state == LLAMA_MEMPLAN_STATE_FAILED_PREPARE) {
        return nullptr;
    }

    if (ub_idx >= views.size()) {
        return nullptr;
    }

    return &views[ub_idx];
}

// ---------------------------------------------------------------------------
// apply — I2: 정확히 1회 · canonical 순서 0→K-1
// ---------------------------------------------------------------------------

bool llama_memplan::apply(uint32_t ub_idx) {
    if (state == LLAMA_MEMPLAN_STATE_COMMITTED || state == LLAMA_MEMPLAN_STATE_ABORTED) {
        set_fail(LLAMA_MEMPLAN_FAIL_TERMINAL, "apply on terminal plan");
        return false;
    }

    if (state == LLAMA_MEMPLAN_STATE_FAILED_PREPARE) {
        set_fail(LLAMA_MEMPLAN_FAIL_STALE, "apply on discarded plan");
        return false;
    }

    if (ub_idx >= views.size()) {
        set_fail(LLAMA_MEMPLAN_FAIL_ORDER, "apply index out of range");
        return false;
    }

    if (ub_idx < n_applied) {
        set_fail(LLAMA_MEMPLAN_FAIL_REAPPLY, "ubatch already applied");
        return false;
    }

    if (ub_idx != n_applied) {
        set_fail(LLAMA_MEMPLAN_FAIL_ORDER, "canonical order violation");
        return false;
    }

    state = LLAMA_MEMPLAN_STATE_APPLYING;

    bool res = false;

    switch (kind) {
        case LLAMA_MEMPLAN_KIND_KV:     res = apply_kv  (ub_idx); break;
        case LLAMA_MEMPLAN_KIND_HYBRID: res = apply_hyb (ub_idx); break;
        case LLAMA_MEMPLAN_KIND_DSV4:   res = apply_dsv4(ub_idx); break;
        default:                        res = false;              break;
    }

    if (!res) {
        return false;
    }

    ++n_applied;

    if (n_applied == views.size()) {
        state = LLAMA_MEMPLAN_STATE_APPLIED;
    }

    return true;
}

bool llama_memplan::apply_kv(uint32_t ub_idx) {
    if (inj_ub == (int32_t) ub_idx && inj_sub != 0) {
        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, "injected kv apply failure");
        return false;
    }

    const auto & v = views[ub_idx];

    m_kv->apply_ubatch(v.kv.sinfo, v.ubatch);
    trace_push(LLAMA_MEMPLAN_EV_KV_APPLY);

    return true;
}

bool llama_memplan::apply_hyb(uint32_t ub_idx) {
    const auto & v = views[ub_idx];

    // composite 원자성(I3): 첫 번째 하위(attn) 변이 전 보상 지점을 잡는다.
    llama_kv_cache::memplan_state_t attn_before = m_kv->memplan_capture();

    if (inj_ub == (int32_t) ub_idx && inj_sub == 1) {
        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, "injected hybrid attn apply failure");
        return false;   // 첫 번째 하위 변이 전이므로 보상 불요
    }

    // 순서 계약: apply 는 attn → recurrent (현행 그대로 · 변경 금지)
    m_kv->apply_ubatch(v.kv.sinfo, v.ubatch);
    trace_push(LLAMA_MEMPLAN_EV_ATTN_APPLY);

    if (inj_ub == (int32_t) ub_idx && inj_sub == 2) {
        // 두 번째 하위 실패 → 첫 번째 하위 보상 복원 후 실패 전파
        m_kv->memplan_restore(attn_before);
        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, "injected hybrid recurrent apply failure (attn compensated)");
        return false;
    }

    // recurrent 하위 kind: 재유도 없이 캡처 리플레이 (find_slot 재호출 금지)
    if (v.rs.cells.size() != m_rs->cells.size()) {
        m_kv->memplan_restore(attn_before);
        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, "recurrent snapshot shape mismatch (attn compensated)");
        return false;
    }

    m_rs->cells  = v.rs.cells;
    m_rs->used   = v.rs.used;
    m_rs->head   = v.rs.head;
    m_rs->n      = v.rs.n;
    m_rs->rs_z   = v.rs.rs_z;
    m_rs->rs_idx = v.rs.rs_idx_post;   // post-consumption 상태 승계

    trace_push(LLAMA_MEMPLAN_EV_RECURRENT_REPLAY);

    return true;
}

bool llama_memplan::apply_dsv4(uint32_t ub_idx) {
    const auto & v = views[ub_idx];

    llama_kv_cache::memplan_state_t base_before = m_raw_base->memplan_capture();

    if (inj_ub == (int32_t) ub_idx && inj_sub == 1) {
        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, "injected dsv4 base apply failure");
        return false;
    }

    m_raw_base->apply_ubatch(v.dsv4.sinfo_raw_base_write, v.ubatch_raw);
    trace_push(LLAMA_MEMPLAN_EV_DSV4_RAW_BASE);

    if (inj_ub == (int32_t) ub_idx && inj_sub == 2) {
        m_raw_base->memplan_restore(base_before);
        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, "injected dsv4 swa apply failure (base compensated)");
        return false;
    }

    llama_kv_cache::memplan_state_t swa_before = m_raw_swa->memplan_capture();

    m_raw_swa->apply_ubatch(v.dsv4.sinfo_raw_swa_write, v.ubatch_raw);
    trace_push(LLAMA_MEMPLAN_EV_DSV4_RAW_SWA);

    // 압축 캐시(csa/hca/lid)와 compressor state 는 **graph output** 이라 plan 의 apply 변이가 없다.
    // 따라서 이 경계들의 "실패 주입"은 apply 단계에서는 사유 구분만 하고 raw 측 composite 를
    // 전량 보상한다. 물리 상태(실변이) 축의 경계별 검증은 APPLIED-abort 의 명시 clear 가 진다
    // (r24 수리 4 — P41 ⓚ arm B 의 sentinel 오라클).
    if (inj_ub == (int32_t) ub_idx && inj_sub >= 3 && inj_sub <= 6) {
        static const char * const bnd[] = { "csa", "hca", "lid", "compressor-state" };

        m_raw_swa->memplan_restore(swa_before);
        m_raw_base->memplan_restore(base_before);

        const std::string why =
            std::string("injected dsv4 ") + bnd[inj_sub - 3] + " boundary failure (raw compensated)";

        set_fail(LLAMA_MEMPLAN_FAIL_APPLY, why.c_str());
        return false;
    }

    trace_push(LLAMA_MEMPLAN_EV_DSV4_COMP_BOUNDARY);

    return true;
}

// ---------------------------------------------------------------------------
// commit / abort
// ---------------------------------------------------------------------------

bool llama_memplan::commit() {
    if (state == LLAMA_MEMPLAN_STATE_COMMITTED || state == LLAMA_MEMPLAN_STATE_ABORTED) {
        set_fail(LLAMA_MEMPLAN_FAIL_TERMINAL, "commit on terminal plan");
        return false;
    }

    if (state == LLAMA_MEMPLAN_STATE_FAILED_PREPARE) {
        set_fail(LLAMA_MEMPLAN_FAIL_TERMINAL, "commit on failed plan");
        return false;
    }

    if (n_applied != views.size()) {
        set_fail(LLAMA_MEMPLAN_FAIL_INCOMPLETE, "commit with unapplied ubatches");
        return false;
    }

    state = LLAMA_MEMPLAN_STATE_COMMITTED;

    return true;
}

bool llama_memplan::abort() {
    if (state == LLAMA_MEMPLAN_STATE_COMMITTED || state == LLAMA_MEMPLAN_STATE_ABORTED) {
        set_fail(LLAMA_MEMPLAN_FAIL_TERMINAL, "abort on terminal plan");
        return false;
    }

    if (state == LLAMA_MEMPLAN_STATE_FAILED_PREPARE) {
        // 이미 무변이 — 종결만 표시
        state = LLAMA_MEMPLAN_STATE_ABORTED;
        return true;
    }

    // ★r24 수리 1 — APPLIED 는 "apply_all 이 끝난" 상태이고, SPEC §6 Q-A 가 승인한 순서가
    //   `apply_all → compute들 → commit` 이므로 **compute 가 이미 물리 캐시에 썼을 수 있는
    //   유일한 구간**이다(recurrent r_l/s_l · DSV4 압축 캐시/compressor state — 1차 소스
    //   models/delta-net-base.cpp:496,555-558,603 / models/deepseek4.cpp:930-945,982-997,1025-1046).
    //   S1 에는 물리 rollback 수단이 없다 → metadata-only 복원은 I3 위반이므로 보수적으로
    //   명시 clear(data 포함)로 처리한다. PREPARED/APPLYING 은 compute 이전이라 metadata 복원이 옳다.
    if (inj_restore || state == LLAMA_MEMPLAN_STATE_APPLIED) {
        state = LLAMA_MEMPLAN_STATE_ABORTED;
        seq_fail_close_clear();
        return false;
    }

    snapshot_restore(origin);

    state = LLAMA_MEMPLAN_STATE_ABORTED;

    return true;
}

llama_memplan::~llama_memplan() {
    if (state == LLAMA_MEMPLAN_STATE_PREPARED ||
        state == LLAMA_MEMPLAN_STATE_APPLYING ||
        state == LLAMA_MEMPLAN_STATE_APPLIED) {
        // 종결 없이 소멸 = 자동 abort + fail-close 기록
        ++g_destruct_abort;

        LLAMA_LOG_WARN("%s: memplan destroyed in non-terminal state %d - auto abort\n", __func__, (int) state);

        // abort() 와 같은 분기 규율(r24 수리 1): APPLIED 는 물리 상태 보수적 clear.
        if (state == LLAMA_MEMPLAN_STATE_APPLIED) {
            seq_fail_close_clear();
        } else {
            snapshot_restore(origin);
        }

        state = LLAMA_MEMPLAN_STATE_ABORTED;
        fail  = LLAMA_MEMPLAN_FAIL_DESTRUCT;
    }
}

// ===========================================================================
// [B] selfcheck 픽스처 — SPEC §4 ⓐ~ⓛ × 3계열(standard KV · hybrid · DSV4)
// ===========================================================================

namespace {

//
// PASS/FAIL 회계는 호출자 소유. 여기서는 콜백으로 넘기고 실패 수만 센다.
//
struct sc_ctx {
    llama_memplan_selfcheck_cb cb = nullptr;
    void *                     ud = nullptr;

    int total  = 0;
    int failed = 0;

    bool check(bool ok, const std::string & name) {
        ++total;

        if (!ok) {
            ++failed;
        }

        if (cb) {
            cb(ud, ok ? 1 : 0, name.c_str());
        }

        return ok;
    }
};

//
// 관측 다이제스트 (plan 밖에서 같은 축으로 재계산 — 등가성 판정 오라클)
//
uint64_t fx_digest(llama_memory_i * mem) {
    fnv64 f;

    if (auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem)) {
        f.u32((uint32_t) LLAMA_MEMPLAN_KIND_HYBRID);
        f.u64(hyb->get_mem_attn()->memplan_digest());
        digest_rs(f, hyb->get_mem_recr());
    } else if (auto * d4 = dynamic_cast<llama_kv_cache_dsv4 *>(mem)) {
        f.u32((uint32_t) LLAMA_MEMPLAN_KIND_DSV4);
        f.u64(d4->get_raw()->get_base()->memplan_digest());
        f.u64(d4->get_raw()->get_swa()->memplan_digest());
        f.u64(d4->get_csa()->memplan_digest());
        f.u64(d4->get_hca()->memplan_digest());
        f.u64(d4->get_lid()->memplan_digest());
    } else if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        f.u32((uint32_t) LLAMA_MEMPLAN_KIND_KV);
        f.u64(kv->memplan_digest());
    }

    return f.h;
}

void digest_comp_plan(fnv64 & f, const llama_kv_cache_dsv4_context::comp_plan & p) {
    const auto vec_i32 = [&f](const std::vector<int32_t> & v) {
        f.u64(v.size());
        for (const auto x : v) { f.i32(x); }
    };

    vec_i32(p.state_pos);
    vec_i32(p.state_persist_src_idxs);
    vec_i32(p.state_persist_dst_idxs);
    vec_i32(p.state_read_idxs);
    vec_i32(p.state_write_pos);
    vec_i32(p.n_visible);

    f.u64(p.state_write_idxs.size());
    for (const auto x : p.state_write_idxs) { f.u64((uint64_t) x); }

    f.u64((uint64_t) p.n_stream);
    f.u64((uint64_t) p.n_kv);
}

// 동결 뷰 다이제스트 (select 멱등·순서 무관 판정 + legacy 대조)
uint64_t view_digest(const llama_memplan_view & v) {
    fnv64 f;

    f.u32(v.ub_idx);
    f.u32(v.ubatch.n_tokens);

    if (v.has_kv) {
        f.u64(0xA11u);
        digest_sinfo(f, v.kv.sinfo);
        f.u32(v.kv.n_kv);
        f.u64(v.kv.k_idxs.size());  f.bytes(v.kv.k_idxs.data(),  v.kv.k_idxs.size()*sizeof(int64_t));
        f.u64(v.kv.v_idxs.size());  f.bytes(v.kv.v_idxs.data(),  v.kv.v_idxs.size()*sizeof(int64_t));
        f.u64(v.kv.kq_mask.size()); f.bytes(v.kv.kq_mask.data(), v.kv.kq_mask.size()*sizeof(float));
        for (int d = 0; d < 4; ++d) { f.u64((uint64_t) v.kv.kq_ne[d]); }
    }

    if (v.has_rs) {
        f.u64(0xB22u);
        f.u32(v.rs.used); f.u32(v.rs.head); f.u32(v.rs.n); f.i32(v.rs.rs_z);
        f.u64(v.rs.s_copy.size()); f.bytes(v.rs.s_copy.data(), v.rs.s_copy.size()*sizeof(int32_t));
        for (const auto x : v.rs.rs_idx_pre)  { f.u32(x); }
        f.u64(0xB23u);
        for (const auto x : v.rs.rs_idx_post) { f.u32(x); }
        for (const auto & c : v.rs.cells) {
            f.i32(c.pos); f.i32(c.src); f.i32(c.src0); f.i32(c.tail);
            for (const auto sq : c.seq_id) { f.i32(sq); }
            f.u64(0xB24u);
        }
    }

    if (v.has_dsv4) {
        f.u64(0xC33u);
        digest_sinfo(f, v.dsv4.sinfo_raw_base_write);
        digest_sinfo(f, v.dsv4.sinfo_raw_swa_write);
        digest_sinfo(f, v.dsv4.sinfo_raw_swa_read);
        f.u32(v.dsv4.raw_n_kv);
        f.u64(v.dsv4.raw_kq_mask.size());
        f.bytes(v.dsv4.raw_kq_mask.data(), v.dsv4.raw_kq_mask.size()*sizeof(float));
        for (int d = 0; d < 4; ++d) { f.u64((uint64_t) v.dsv4.raw_kq_ne[d]); }
        digest_sinfo(f, v.dsv4.sinfo_csa);
        digest_sinfo(f, v.dsv4.sinfo_hca);
        digest_sinfo(f, v.dsv4.sinfo_lid);
        digest_comp_plan(f, v.dsv4.plan_csa);
        digest_comp_plan(f, v.dsv4.plan_hca);
        digest_comp_plan(f, v.dsv4.plan_lid);
        f.u64((uint64_t) (uintptr_t) v.dsv4.id_csa.state);
        f.u32(v.dsv4.id_csa.ratio); f.u32(v.dsv4.id_csa.state_size); f.u32(v.dsv4.id_csa.n_stream);
        f.u64((uint64_t) (uintptr_t) v.dsv4.id_hca.state);
        f.u32(v.dsv4.id_hca.ratio); f.u32(v.dsv4.id_hca.state_size); f.u32(v.dsv4.id_hca.n_stream);
        f.u64((uint64_t) (uintptr_t) v.dsv4.id_lid.state);
        f.u32(v.dsv4.id_lid.ratio); f.u32(v.dsv4.id_lid.state_size); f.u32(v.dsv4.id_lid.n_stream);
    }

    return f.h;
}

// compressor-state identity 는 인스턴스 포인터를 포함하므로, 서로 다른 memory 인스턴스끼리
// 비교할 때는 포인터 축을 제외한다(계획 내용 자체는 그대로 비교).
uint64_t view_digest_xinstance(const llama_memplan_view & v) {
    llama_memplan_view t = v;

    t.dsv4.id_csa.state = nullptr;
    t.dsv4.id_hca.state = nullptr;
    t.dsv4.id_lid.state = nullptr;

    return view_digest(t);
}

//
// 물리 sentinel 프로브 (r24 수리 4) — memplan_digest 는 metadata 전용이라 물리 축의 오라클이
// 못 된다(r24 D). 각 경계의 **실제 텐서 바이트**를 직접 쓰고 되읽는다.
//
struct sentinel_probe {
    const char  * name = "";
    ggml_tensor * t    = nullptr;
    size_t        off  = 0;
    size_t        n    = 16;

    bool valid() const { return t != nullptr && t->buffer != nullptr && t->data != nullptr; }
};

void sentinel_write(const sentinel_probe & p, uint8_t pat) {
    if (!p.valid()) {
        return;
    }

    std::vector<uint8_t> buf(p.n, pat);
    ggml_backend_tensor_set(p.t, buf.data(), p.off, p.n);
}

bool sentinel_is(const sentinel_probe & p, uint8_t pat) {
    if (!p.valid()) {
        return false;
    }

    std::vector<uint8_t> buf(p.n, (uint8_t) (pat ^ 0xff));
    ggml_backend_tensor_get(p.t, buf.data(), p.off, p.n);

    for (const uint8_t b : buf) {
        if (b != pat) {
            return false;
        }
    }

    return true;
}

bool sentinel_is_zero(const sentinel_probe & p) {
    return sentinel_is(p, 0x00);
}

//
// 배치 픽스처
//
struct fx_batch {
    std::vector<float>          embd;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   sid;
    std::vector<llama_seq_id *> sid_ptr;

    llama_batch b {};

    void build(uint32_t n_embd, const std::vector<llama_seq_id> & s, const std::vector<llama_pos> & p) {
        const size_t n = s.size();

        embd.resize(n*n_embd);
        for (size_t i = 0; i < embd.size(); ++i) {
            embd[i] = (float) ((i*2654435761ull) % 97)/97.0f;
        }

        pos = p;
        n_seq_id.assign(n, 1);
        sid = s;

        sid_ptr.resize(n + 1);
        for (size_t i = 0; i < n; ++i) {
            sid_ptr[i] = &sid[i];
        }
        sid_ptr[n] = nullptr;

        b.n_tokens = (int32_t) n;
        b.token    = nullptr;
        b.embd     = embd.data();
        b.pos      = pos.data();
        b.n_seq_id = n_seq_id.data();
        b.seq_id   = sid_ptr.data();
        b.logits   = nullptr;
    }
};

//
// legacy(현행) 경로 실행 오라클 — 각 ubatch 의 apply → set_inputs 직후 값을 저장한다.
//
struct legacy_ub {
    uint32_t n_kv = 0;

    std::vector<int64_t> k_idxs;
    std::vector<int64_t> v_idxs;
    std::vector<float>   kq_mask;
    int64_t              kq_ne[4] = { 0, 0, 0, 0 };

    uint32_t rs_head = 0;
    uint32_t rs_n    = 0;
    int32_t  rs_z    = -1;
    std::vector<int32_t> s_copy;

    uint32_t           raw_n_kv  = 0;
    std::vector<float> raw_mask;
    int64_t            raw_ne[4] = { 0, 0, 0, 0 };

    llama_kv_cache::slot_info si_kv, si_base, si_write, si_read, si_csa, si_hca, si_lid;

    llama_kv_cache_dsv4_context::comp_plan p_csa, p_hca, p_lid;
};

struct legacy_run {
    bool ok = false;

    std::vector<legacy_ub> ubs;
    std::vector<uint32_t>  rs_idx_final;

    uint64_t digest = 0;
};

legacy_run legacy_execute(
        llama_memory_i     * mem,
        llama_batch_allocr & balloc,
        uint32_t             n_ubatch,
        bool                 embd_all,
        bool                 causal) {
    legacy_run run;

    llama_memory_context_ptr mctx = mem->init_batch(balloc, n_ubatch, embd_all);

    if (!mctx || llama_memory_status_is_fail(mctx->get_status())) {
        return run;
    }

    auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem);
    auto * d4  = dynamic_cast<llama_kv_cache_dsv4 *>(mem);
    auto * kv  = dynamic_cast<llama_kv_cache *>(mem);

    uint32_t i = 0;

    while (true) {
        const llama_ubatch ub = mctx->get_ubatch();

        if (!mctx->apply()) {
            return run;
        }

        legacy_ub lu;

        if (kv || hyb) {
            const llama_kv_cache_context * kctx = hyb
                ? static_cast<const llama_memory_hybrid_context *>(mctx.get())->get_attn()
                : static_cast<const llama_kv_cache_context *>(mctx.get());

            llama_kv_cache * cache = hyb ? hyb->get_mem_attn() : kv;

            lu.n_kv  = kctx->get_n_kv();
            lu.si_kv = kctx->get_sinfos()[i];

            const uint32_t n_stream_g = cache->get_n_stream() == 1 ? 1 : std::max<uint32_t>(1, ub.n_seqs_unq);

            host_scratch hs;
            if (!hs.init(3)) {
                return run;
            }

            ggml_tensor * k_idxs = kctx->build_input_k_idxs(hs.ctx, ub);
            ggml_tensor * v_idxs = kctx->build_input_v_idxs(hs.ctx, ub);
            ggml_tensor * mask   = ggml_new_tensor_4d(hs.ctx, GGML_TYPE_F32,
                                        lu.n_kv, ub.n_tokens/n_stream_g, 1, n_stream_g);

            if (!hs.alloc()) {
                return run;
            }

            kctx->set_input_k_idxs (k_idxs, &ub);
            kctx->set_input_v_idxs (v_idxs, &ub);
            kctx->set_input_kq_mask(mask,   &ub, causal);

            grab_i64(k_idxs, lu.k_idxs);
            grab_i64(v_idxs, lu.v_idxs);
            grab_f32(mask,   lu.kq_mask);

            for (int d = 0; d < 4; ++d) {
                lu.kq_ne[d] = mask->ne[d];
            }
        }

        if (hyb) {
            const auto * rctx = static_cast<const llama_memory_hybrid_context *>(mctx.get())->get_recr();

            lu.rs_head = rctx->get_head();
            lu.rs_n    = rctx->get_n_rs();
            lu.rs_z    = rctx->get_rs_z();

            lu.s_copy.resize(lu.rs_n);
            for (uint32_t j = 0; j < lu.rs_n; ++j) {
                lu.s_copy[j] = rctx->s_copy((int) j);
            }
        }

        if (d4) {
            const auto * dctx = static_cast<const llama_kv_cache_dsv4_context *>(mctx.get());
            const auto * raw  = dctx->get_raw();

            lu.raw_n_kv = raw->get_n_kv();
            lu.si_base  = raw->get_base_ctx()->get_sinfos()[i];
            lu.si_write = raw->get_sinfos_write()[i];
            lu.si_read  = raw->get_sinfos_read()[i];
            lu.si_csa   = dctx->get_csa()->get_sinfos()[i];
            lu.si_hca   = dctx->get_hca()->get_sinfos()[i];
            lu.si_lid   = dctx->get_lid()->get_sinfos()[i];
            lu.p_csa    = dctx->get_csa_plan();
            lu.p_hca    = dctx->get_hca_plan();
            lu.p_lid    = dctx->get_lid_plan();

            const llama_kv_cache * kv_swa = raw->get_kv_swa();

            const uint32_t n_stream_g = kv_swa->get_n_stream() == 1 ? 1 : std::max<uint32_t>(1, ub.n_seqs_unq);

            host_scratch hs;
            if (!hs.init(1)) {
                return run;
            }

            ggml_tensor * mask = ggml_new_tensor_4d(hs.ctx, GGML_TYPE_F32,
                                    lu.raw_n_kv, ub.n_tokens/n_stream_g, 1, n_stream_g);

            if (!hs.alloc()) {
                return run;
            }

            raw->set_input_kq_mask(mask, &ub, causal);

            grab_f32(mask, lu.raw_mask);

            for (int d = 0; d < 4; ++d) {
                lu.raw_ne[d] = mask->ne[d];
            }
        }

        run.ubs.push_back(std::move(lu));

        if (!mctx->next()) {
            break;
        }

        ++i;
    }

    if (hyb) {
        run.rs_idx_final = hyb->get_mem_recr()->rs_idx;
    }

    run.digest = fx_digest(mem);
    run.ok     = true;

    return run;
}

//
// legacy 캡처값 ↔ plan 동결 뷰 1:1 대조 (SPEC §4 ⓘ · byte-equivalent)
//
bool legacy_view_match(const legacy_ub & l, const llama_memplan_view & v) {
    if (v.has_kv) {
        if (l.n_kv    != v.kv.n_kv)    { return false; }
        if (l.k_idxs  != v.kv.k_idxs)  { return false; }
        if (l.v_idxs  != v.kv.v_idxs)  { return false; }
        if (l.kq_mask != v.kv.kq_mask) { return false; }

        for (int d = 0; d < 4; ++d) {
            if (l.kq_ne[d] != v.kv.kq_ne[d]) { return false; }
        }

        if (!sinfo_equal(l.si_kv, v.kv.sinfo)) { return false; }
    }

    if (v.has_rs) {
        if (l.rs_head != v.rs.head)   { return false; }
        if (l.rs_n    != v.rs.n)      { return false; }
        if (l.rs_z    != v.rs.rs_z)   { return false; }
        if (l.s_copy  != v.rs.s_copy) { return false; }
    }

    if (v.has_dsv4) {
        if (l.raw_n_kv != v.dsv4.raw_n_kv)    { return false; }
        if (l.raw_mask != v.dsv4.raw_kq_mask) { return false; }

        for (int d = 0; d < 4; ++d) {
            if (l.raw_ne[d] != v.dsv4.raw_kq_ne[d]) { return false; }
        }

        if (!sinfo_equal(l.si_base,  v.dsv4.sinfo_raw_base_write)) { return false; }
        if (!sinfo_equal(l.si_write, v.dsv4.sinfo_raw_swa_write))  { return false; }
        if (!sinfo_equal(l.si_read,  v.dsv4.sinfo_raw_swa_read))   { return false; }
        if (!sinfo_equal(l.si_csa,   v.dsv4.sinfo_csa))            { return false; }
        if (!sinfo_equal(l.si_hca,   v.dsv4.sinfo_hca))            { return false; }
        if (!sinfo_equal(l.si_lid,   v.dsv4.sinfo_lid))            { return false; }

        fnv64 a;
        digest_comp_plan(a, l.p_csa); digest_comp_plan(a, l.p_hca); digest_comp_plan(a, l.p_lid);

        fnv64 b;
        digest_comp_plan(b, v.dsv4.plan_csa); digest_comp_plan(b, v.dsv4.plan_hca); digest_comp_plan(b, v.dsv4.plan_lid);

        if (a.h != b.h) { return false; }
    }

    return true;
}

//
// 모델·hparams 픽스처
//
void fx_hparams(llama_hparams & hp, uint32_t n_layer, uint32_t n_embd_head, uint32_t n_head_kv) {
    std::memset((void *) &hp, 0, sizeof(hp));

    hp.n_ctx_train           = 8192;
    hp.n_embd                = n_embd_head*n_head_kv;
    hp.n_layer_all           = n_layer;
    hp.n_layer_nextn         = 0;
    hp.n_layer_kv_from_start = -1;

    hp.n_embd_head_k_full = n_embd_head;
    hp.n_embd_head_v_full = n_embd_head;
    hp.n_embd_head_k_swa  = n_embd_head;
    hp.n_embd_head_v_swa  = n_embd_head;

    hp.causal_attn = true;
    hp.swa_type    = LLAMA_SWA_TYPE_NONE;
    hp.n_swa       = 0;
    hp.rope_type   = LLAMA_ROPE_TYPE_NORM;
    hp.no_alloc    = false;

    for (uint32_t i = 0; i < LLAMA_MAX_LAYERS; ++i) {
        hp.n_head_arr[i]            = n_head_kv;
        hp.n_head_kv_arr[i]         = n_head_kv;
        hp.n_ff_arr[i]              = 4*hp.n_embd;
        hp.is_swa_impl[i]           = 0;
        hp.is_recr_impl[i]          = 0;
        hp.dsv4_compress_ratios[i]  = 0;
        hp.deepstack_mapping_arr[i] = -1;
    }
}

// llama_model 은 추상 클래스(load_*/build_arch_graph 순수 가상)이므로 직접 인스턴스화할 수 없다.
// 아키별 구현체를 만드는 팩토리(llama_model_create)를 그대로 쓰고 hparams 만 픽스처 값으로 덮는다.
struct fx_model {
    llama_model_params            mp;
    std::unique_ptr<llama_model>  model;

    fx_model() : mp(llama_model_default_params()) {
        model.reset(llama_model_create(LLM_ARCH_LLAMA, mp));

        if (model) {
            model->name = "memplan-fixture";
        }
    }
};

//
// 계열 정의 — 같은 픽스처 몸통을 3계열에 반복 적용한다.
//
enum fx_kind { FX_KV = 0, FX_HYBRID = 1, FX_DSV4 = 2 };

struct fx_series {
    fx_kind      kind = FX_KV;
    const char * tag  = "KV";

    uint32_t n_layer     = 4;
    uint32_t n_embd_head = 32;
    uint32_t n_head_kv   = 2;

    uint32_t kv_size   = 1024;
    uint32_t n_pad     = 32;
    uint32_t n_seq_max = 1;
    uint32_t n_ubatch  = 96;
    uint32_t n_seq     = 1;
    uint32_t n_tok_seq = 384;
    uint32_t n_rs_seq  = 0;
    uint32_t rs_size   = 0;
};

fx_series fx_series_of(fx_kind k) {
    fx_series s;

    s.kind = k;

    switch (k) {
        case FX_KV:
            s.tag = "KV";
            break;
        case FX_HYBRID:
            s.tag       = "HYB";
            s.n_seq_max = 4;
            s.n_seq     = 2;
            s.n_tok_seq = 64;
            s.n_ubatch  = 32;
            s.n_rs_seq  = 2;
            s.rs_size   = 16;
            break;
        case FX_DSV4:
            s.tag       = "DSV4";
            s.n_seq_max = 1;
            s.n_seq     = 1;
            s.n_tok_seq = 384;
            s.n_ubatch  = 96;
            break;
    }

    return s;
}

void fx_apply_hparams(const fx_series & s, llama_hparams & hp) {
    fx_hparams(hp, s.n_layer, s.n_embd_head, s.n_head_kv);

    if (s.kind == FX_HYBRID) {
        // qwen35moe 형상: attn/recurrent 층 혼재 + rs 롤백 평면
        hp.ssm_d_conv  = 4;
        hp.ssm_d_inner = 32;
        hp.ssm_d_state = 8;
        hp.ssm_n_group = 1;

        for (uint32_t i = 0; i < s.n_layer; ++i) {
            hp.is_recr_impl[i] = (i % 2) ? 1 : 0;
        }
    }

    if (s.kind == FX_DSV4) {
        // deepseek4 형상: raw ISWA(base 1층 + swa 3층) + CSA/HCA/LID 압축
        hp.swa_type          = LLAMA_SWA_TYPE_STANDARD;
        hp.n_swa             = 64;
        hp.indexer_head_size = 16;
        hp.indexer_n_head    = 1;

        for (uint32_t i = 0; i < s.n_layer; ++i) {
            hp.is_swa_impl[i]          = (i == 0) ? 0u : 1u;
            hp.dsv4_compress_ratios[i] = (i == 2) ? 128u : 4u;
        }
    }
}

std::unique_ptr<llama_memory_i> fx_make_mem(const fx_series & s, const llama_model & model) {
    switch (s.kind) {
        case FX_KV:
            return std::unique_ptr<llama_memory_i>(new llama_kv_cache(
                        model, model.hparams,
                        GGML_TYPE_F16, GGML_TYPE_F16,
                        /*v_trans =*/ false, /*offload =*/ false, /*unified =*/ true,
                        s.kv_size, s.n_seq_max, s.n_pad,
                        /*n_swa =*/ 0, LLAMA_SWA_TYPE_NONE, /*mem_other =*/ nullptr,
                        nullptr, nullptr, nullptr));

        case FX_HYBRID:
            return std::unique_ptr<llama_memory_i>(new llama_memory_hybrid(
                        model,
                        GGML_TYPE_F16, GGML_TYPE_F16, /*v_trans =*/ false,
                        s.kv_size, s.n_pad, /*n_swa =*/ 0, LLAMA_SWA_TYPE_NONE,
                        GGML_TYPE_F32, GGML_TYPE_F32, s.rs_size,
                        s.n_seq_max, s.n_rs_seq,
                        /*offload =*/ false, /*unified =*/ true,
                        nullptr, nullptr));

        case FX_DSV4:
            return std::unique_ptr<llama_memory_i>(new llama_kv_cache_dsv4(
                        model,
                        GGML_TYPE_F16, GGML_TYPE_F16, /*v_trans =*/ false,
                        /*offload =*/ false, /*swa_full =*/ true, /*unified =*/ true,
                        s.kv_size, s.n_seq_max, s.n_ubatch, s.n_pad,
                        nullptr, nullptr));
    }

    return nullptr;
}

void fx_make_batch(const fx_series & s, fx_batch & fb, uint32_t n_tok_seq, llama_pos pos0) {
    std::vector<llama_seq_id> sid;
    std::vector<llama_pos>    pos;

    for (uint32_t q = 0; q < s.n_seq; ++q) {
        for (uint32_t t = 0; t < n_tok_seq; ++t) {
            sid.push_back((llama_seq_id) q);
            pos.push_back(pos0 + (llama_pos) t);
        }
    }

    fb.build(s.n_embd_head*s.n_head_kv, sid, pos);
}

// 한 세트(memory + batch + balloc) — 픽스처 인스턴스
struct fx_inst {
    std::unique_ptr<llama_memory_i>     mem;
    fx_batch                            fb;
    std::unique_ptr<llama_batch_allocr> balloc;

    bool init(const fx_series & s, const llama_model & model, uint32_t n_tok_seq, llama_pos pos0) {
        mem = fx_make_mem(s, model);
        if (!mem) {
            return false;
        }

        return rebatch(s, model, n_tok_seq, pos0);
    }

    bool rebatch(const fx_series & s, const llama_model & model, uint32_t n_tok_seq, llama_pos pos0) {
        fx_make_batch(s, fb, n_tok_seq, pos0);

        balloc.reset(new llama_batch_allocr(model.hparams.n_pos_per_embd()));

        return balloc->init(fb.b, model.vocab, mem.get(), model.hparams.n_embd, s.n_seq_max, /*output_all =*/ false);
    }
};

// 계열별 "경계" 물리 프로브 — 각 경계에 구별되는 sentinel 을 심고 독립 확인한다.
struct boundary_probes {
    ggml_context *              ctx = nullptr;
    std::vector<sentinel_probe> v;

    ~boundary_probes() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    bool build(fx_kind kind, llama_memory_i * mem) {
        ggml_init_params p = {
            /*.mem_size   =*/ ggml_tensor_overhead()*16 + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(p);
        if (!ctx) {
            return false;
        }

        // compressor state 는 view(reshape)로만 노출된다. 새로 만든 view 는 buffer 가 NULL 이라
        // (ggml.c:1810 — ggml_new_tensor_impl 은 view 에도 buffer=NULL 을 넣는다) 원본 버퍼에 결속해야
        // ggml_backend_tensor_set/get 이 선다.
        const auto bind_view = [](ggml_tensor * t) -> ggml_tensor * {
            if (t && t->buffer == nullptr && t->view_src != nullptr) {
                ggml_backend_view_init(t);
            }
            return t;
        };

        if (kind == FX_KV) {
            auto * kv = dynamic_cast<llama_kv_cache *>(mem);
            v.push_back({ "kv-K", kv->get_k_storage(0), 0, 16 });
        } else if (kind == FX_HYBRID) {
            auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem);
            v.push_back({ "attn-K",        hyb->get_mem_attn()->get_k_storage(0), 0, 16 });
            v.push_back({ "recurrent-s_l", hyb->get_mem_recr()->s_l[1],           0, 16 });
        } else {
            auto * d4 = dynamic_cast<llama_kv_cache_dsv4 *>(mem);
            v.push_back({ "raw-base-K",  d4->get_raw()->get_base()->get_k_storage(0), 0, 16 });
            v.push_back({ "raw-swa-K",   d4->get_raw()->get_swa ()->get_k_storage(1), 0, 16 });
            v.push_back({ "csa-K",       d4->get_csa()->get_k_storage(0),             0, 16 });
            v.push_back({ "hca-K",       d4->get_hca()->get_k_storage(2),             0, 16 });
            v.push_back({ "lid-K",       d4->get_lid()->get_k_storage(0),             0, 16 });
            v.push_back({ "comp-state",  bind_view(d4->get_csa_state()->get_kv(ctx, 0)), 0, 16 });
        }

        for (const auto & pr : v) {
            if (!pr.valid()) {
                return false;
            }
        }

        return true;
    }
};

std::string nm(const fx_series & s, const char * item, const char * text) {
    return std::string("P41-") + s.tag + "-" + item + ". " + text;
}

//
// 한 계열에 대한 ⓐ~ⓛ 전건
//
void fx_run_series(sc_ctx & sc, fx_kind kind) {
    const fx_series s = fx_series_of(kind);

    fx_model fm;

    if (!fm.model) {
        sc.check(false, nm(s, "a0", "픽스처 모델 생성(llama_model_create)"));
        return;
    }

    fx_apply_hparams(s, fm.model->hparams);

    llama_memplan::test_reset_counters();

    const llama_model & model = *fm.model;

    // ---------------------------------------------------------------
    // ⓐ prepare 무변이
    // ---------------------------------------------------------------
    uint32_t K = 0;

    {
        fx_inst a;
        if (!sc.check(a.init(s, model, s.n_tok_seq, 0), nm(s, "a0", "픽스처 memory/batch 구성"))) {
            return;
        }

        const uint64_t d0 = fx_digest(a.mem.get());

        auto plan = llama_memplan::prepare(a.mem.get(), *a.balloc, s.n_ubatch, false, true);

        const uint64_t d1 = fx_digest(a.mem.get());

        K = plan->get_n_ubatch();

        sc.check(plan->get_state() == LLAMA_MEMPLAN_STATE_PREPARED && K >= 2 && d1 == d0,
                 nm(s, "a", "ⓐprepare 무변이 — 전 ubatch 선계산 + 동결 뷰 캡처 뒤에도 memory 관측 상태가 전후 동일"));

        plan->abort();
    }

    if (K < 2) {
        sc.check(false, nm(s, "a2", "ⓐ 전제 — K>=2 인 계획이 서지 않았다"));
        return;
    }

    // ---------------------------------------------------------------
    // ⓑ select 멱등·임의 순서
    // ---------------------------------------------------------------
    {
        fx_inst b;
        b.init(s, model, s.n_tok_seq, 0);

        auto plan = llama_memplan::prepare(b.mem.get(), *b.balloc, s.n_ubatch, false, true);

        std::vector<uint64_t> in_order(K);
        for (uint32_t i = 0; i < K; ++i) {
            in_order[i] = view_digest(*plan->select(i));
        }

        const uint64_t d_before_select = fx_digest(b.mem.get());

        // 전 인덱스를 덮는 비순차 순열 [K-1, 0, K-2, 1, ...] + 반복 조회(멱등)
        std::vector<uint32_t> shuffled;
        {
            uint32_t lo = 0;
            uint32_t hi = K - 1;

            while (lo <= hi) {
                shuffled.push_back(hi);
                if (lo != hi) {
                    shuffled.push_back(lo);
                }
                if (hi == 0) {
                    break;
                }
                ++lo;
                --hi;
            }
        }

        std::vector<bool> seen(K, false);
        for (const uint32_t i : shuffled) {
            seen[i] = true;
        }

        shuffled.push_back(K - 1);
        shuffled.push_back(0);
        shuffled.push_back(K - 1);

        bool ok = true;
        for (const uint32_t i : shuffled) {
            if (view_digest(*plan->select(i)) != in_order[i]) {
                ok = false;
            }
        }

        // 순열이 전 인덱스를 실제로 덮었는가(픽스처 공회전 방지)
        for (uint32_t i = 0; i < K; ++i) {
            ok = ok && seen[i];
        }

        ok = ok && (plan->select(K) == nullptr);

        // select 는 무변이여야 한다 — 조회 전후 memory 관측 상태 동일
        ok = ok && (fx_digest(b.mem.get()) == d_before_select);

        sc.check(ok, nm(s, "b", "ⓑselect 멱등·임의 순서 — 임의 순서/반복 조회가 순차 조회와 동일 view(부작용 있는 현행 reader 직접 호출 0)"));

        plan->abort();
    }

    // ---------------------------------------------------------------
    // ⓒ apply 1회 · canonical 순서 강제
    // ---------------------------------------------------------------
    {
        fx_inst c;
        c.init(s, model, s.n_tok_seq, 0);

        auto plan = llama_memplan::prepare(c.mem.get(), *c.balloc, s.n_ubatch, false, true);

        const bool e_order  = !plan->apply(1) && plan->get_fail() == LLAMA_MEMPLAN_FAIL_ORDER;
        const bool a_ok     =  plan->apply(0);
        const bool e_re     = !plan->apply(0) && plan->get_fail() == LLAMA_MEMPLAN_FAIL_REAPPLY;
        const bool e_skip   = !plan->apply(2) && plan->get_fail() == LLAMA_MEMPLAN_FAIL_ORDER;
        const bool e_commit = !plan->commit() && plan->get_fail() == LLAMA_MEMPLAN_FAIL_INCOMPLETE;

        sc.check(e_order && a_ok && e_re && e_skip && e_commit,
                 nm(s, "c", "ⓒapply 1회·canonical 순서 — 0 이전 진입/재-apply/건너뛰기/미적용 commit 이 전부 fail-close"));

        plan->abort();
    }

    // ---------------------------------------------------------------
    // ⓓ commit 후 상태 = 현행 경로 등가 · ⓘ 뷰 등가성(byte-equivalent) · ⓗ terminal
    // ---------------------------------------------------------------
    {
        fx_inst p;
        fx_inst l;

        p.init(s, model, s.n_tok_seq, 0);
        l.init(s, model, s.n_tok_seq, 0);

        auto plan = llama_memplan::prepare(p.mem.get(), *p.balloc, s.n_ubatch, false, true);

        bool applied = true;
        for (uint32_t i = 0; i < K; ++i) {
            applied = applied && plan->apply(i);
        }

        const bool     committed = plan->commit();
        const uint64_t d_plan    = fx_digest(p.mem.get());

        const legacy_run lr = legacy_execute(l.mem.get(), *l.balloc, s.n_ubatch, false, true);

        sc.check(applied && committed && lr.ok && lr.digest == d_plan &&
                 plan->get_state() == LLAMA_MEMPLAN_STATE_COMMITTED,
                 nm(s, "d", "ⓓcommit 후 상태 = 현행 순차 경로와 관측 동일(셀 점유·pos 범위·상태 digest)"));

        bool views_match = lr.ok && lr.ubs.size() == K;
        for (uint32_t i = 0; views_match && i < K; ++i) {
            views_match = legacy_view_match(lr.ubs[i], *plan->select(i));
        }

        bool rs_final_ok = true;
        if (s.kind == FX_HYBRID) {
            auto * hyb_p = dynamic_cast<llama_memory_hybrid *>(p.mem.get());
            rs_final_ok = (hyb_p->get_mem_recr()->rs_idx == lr.rs_idx_final) &&
                          (hyb_p->get_mem_recr()->rs_idx == plan->select(K - 1)->rs.rs_idx_post);
        }

        sc.check(views_match && rs_final_ok,
                 nm(s, "i", "ⓘ뷰 등가성 — 각 ubatch 의 k_idxs/n_kv/mask/head/n/rs_z/s_copy/DSV4 comp inputs 가 "
                            "legacy 순차 경로의 apply→set_inputs 직후 값과 1:1 byte 동일 + 최종 live rs_idx 등가"));

        const bool t_apply  = !plan->apply(0) && plan->get_fail() == LLAMA_MEMPLAN_FAIL_TERMINAL;
        const bool t_commit = !plan->commit() && plan->get_fail() == LLAMA_MEMPLAN_FAIL_TERMINAL;
        const bool t_abort  = !plan->abort()  && plan->get_fail() == LLAMA_MEMPLAN_FAIL_TERMINAL;

        sc.check(t_apply && t_commit && t_abort,
                 nm(s, "h1", "ⓗterminal(COMMITTED) 재호출 — apply/commit/abort 전부 하드 실패"));
    }

    // ---------------------------------------------------------------
    // ⓘ 보강 — rs_idx 1회 소비 의미론(r22 수정 1)을 "무장된" 상태에서 폐합한다.
    //   무장 값은 엔진의 seq_rm 부분 롤백 경로가 부르는 바로 그 setter(set_rs_idx)로 세운다.
    //   (무장하지 않으면 rs_idx 가 전부 0 이라 소비 의미가 공회전한다)
    // ---------------------------------------------------------------
    if (s.kind == FX_HYBRID) {
        fx_inst p;
        fx_inst l;

        p.init(s, model, s.n_tok_seq, 0);
        l.init(s, model, s.n_tok_seq, 0);

        auto * rp = dynamic_cast<llama_memory_hybrid *>(p.mem.get())->get_mem_recr();
        auto * rl = dynamic_cast<llama_memory_hybrid *>(l.mem.get())->get_mem_recr();

        for (uint32_t q = 0; q < s.n_seq; ++q) {
            rp->set_rs_idx((llama_seq_id) q, q + 1);
            rl->set_rs_idx((llama_seq_id) q, q + 1);
        }

        const std::vector<uint32_t> armed = rp->rs_idx;

        auto plan = llama_memplan::prepare(p.mem.get(), *p.balloc, s.n_ubatch, false, true);

        const bool armed_kept = (rp->rs_idx == armed);

        bool applied = true;
        for (uint32_t i = 0; i < K; ++i) {
            applied = applied && plan->apply(i);
        }
        applied = applied && plan->commit();

        const legacy_run lr = legacy_execute(l.mem.get(), *l.balloc, s.n_ubatch, false, true);

        bool s_copy_match = lr.ok && lr.ubs.size() == K;
        for (uint32_t i = 0; s_copy_match && i < K; ++i) {
            s_copy_match = (lr.ubs[i].s_copy == plan->select(i)->rs.s_copy);
        }

        // 무장 신호가 실제로 살아 있었는가 — rollback 평면 주소(idx*size + src0)가 나와야 한다
        bool live_signal = false;
        for (const int32_t v : plan->select(0)->rs.s_copy) {
            if (v >= (int32_t) rp->size) {
                live_signal = true;
            }
        }

        const bool consumed = (plan->select(0)->rs.rs_idx_pre  == armed) &&
                              (plan->select(0)->rs.rs_idx_post != armed);

        const bool final_eq = (rp->rs_idx == rl->rs_idx) &&
                              (rp->rs_idx == plan->select(K - 1)->rs.rs_idx_post);

        sc.check(applied && armed_kept && s_copy_match && live_signal && consumed && final_eq,
                 nm(s, "i2", "ⓘ보강 — rs 롤백 평면을 무장한 상태에서 s_copy 1회 소비 의미론 폐합"
                             "(prepare 는 무장 상태를 소모하지 않고, 동결 s_copy 가 rollback 평면 주소를 실제로 담으며, "
                             "ubatch별 값과 최종 live rs_idx 가 legacy 와 동일)"));
    }

    // ---------------------------------------------------------------
    // ⓔ abort 복원 (부분 apply)
    // ---------------------------------------------------------------
    {
        fx_inst e;
        e.init(s, model, s.n_tok_seq, 0);

        // ★origin 을 비어 있지 않게 만든다. 그래야 "부분 apply(APPLYING) = metadata 복원" 이
        //   "명시 clear"(APPLIED 후단) 와 관측상 구별된다 — 빈 memory 로는 둘이 같아 공회전한다.
        const legacy_run pre  = legacy_execute(e.mem.get(), *e.balloc, s.n_ubatch, false, true);
        const bool       reba = e.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq);

        fx_inst empty_ref;
        empty_ref.init(s, model, s.n_tok_seq, 0);

        const uint64_t d0 = fx_digest(e.mem.get());

        auto plan = llama_memplan::prepare(e.mem.get(), *e.balloc, s.n_ubatch, false, true);

        // ★부분 apply 로 유지해야 한다(APPLIED 가 되면 r24 수리 1 에 따라 abort 의미론이 clear 로 바뀐다)
        const uint32_t n_part = std::min<uint32_t>(K - 1, K/2 + 1);

        bool applied = true;
        for (uint32_t i = 0; i < n_part; ++i) {
            applied = applied && plan->apply(i);
        }

        const uint64_t d_mid   = fx_digest(e.mem.get());
        const bool     aborted = plan->abort();
        const uint64_t d1      = fx_digest(e.mem.get());

        sc.check(pre.ok && reba && applied && aborted && d_mid != d0 && d1 == d0 &&
                 d0 != fx_digest(empty_ref.mem.get()) &&   // 전제: origin 이 빈 memory 와 다르다(판정 공회전 방지)
                 plan->get_state() == LLAMA_MEMPLAN_STATE_ABORTED,
                 nm(s, "e", "ⓔabort 복원(I3) — 비어 있지 않은 origin 위에서 부분 apply(APPLYING) 로 상태가 실제로 "
                            "변한 뒤 abort 하면 prepare 직전과 구별 불가(= clear 가 아니라 metadata 복원)"));
    }

    // ---------------------------------------------------------------
    // ⓕ K=1 / decode 형상 = plan 경로 비진입(신규 부작용 0)
    // ---------------------------------------------------------------
    {
        fx_inst f1;
        fx_inst f2;

        f1.init(s, model, 1, 0);
        f2.init(s, model, 1, 0);

        const uint64_t before = llama_memplan::test_destruct_abort_count();

        const legacy_run lr = legacy_execute(f1.mem.get(), *f1.balloc, s.n_ubatch, false, true);

        const bool no_side_effect = (llama_memplan::test_destruct_abort_count() == before);

        auto plan = llama_memplan::prepare(f2.mem.get(), *f2.balloc, s.n_ubatch, false, true);

        const bool k1 = plan->get_n_ubatch() == 1;
        const bool ap = plan->apply(0) && plan->commit();

        sc.check(lr.ok && no_side_effect && k1 && ap && fx_digest(f2.mem.get()) == lr.digest,
                 nm(s, "f", "ⓕK=1/decode 형상 — plan 미진입 경로에 신규 부작용 0이고 K=1 plan 커밋 상태도 legacy 와 동일"));
    }

    // ---------------------------------------------------------------
    // ⓖ fragmented 상태 + prepare 중간 실패 → 전체 metadata 무변이
    // ---------------------------------------------------------------
    {
        fx_inst g;
        g.init(s, model, s.n_tok_seq, 0);

        const legacy_run lr = legacy_execute(g.mem.get(), *g.balloc, s.n_ubatch, false, true);

        g.mem->seq_rm(0, (llama_pos) (s.n_tok_seq/4), (llama_pos) (s.n_tok_seq/2));

        const bool rebatched = g.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq);

        const uint64_t d0 = fx_digest(g.mem.get());

        // ★shadow 캡처 "도중" 실패 — ub0/ub1 의 그림자 변이가 이미 memory 에 얹힌 뒤 실패한다.
        //   따라서 이 케이스가 실복원(무변이)을 실제로 증명한다.
        llama_memplan::test_set_prepare_fail_at(1);

        auto plan = llama_memplan::prepare(g.mem.get(), *g.balloc, s.n_ubatch, false, true);

        llama_memplan::test_set_prepare_fail_at(-1);

        const uint64_t d1 = fx_digest(g.mem.get());

        sc.check(lr.ok && rebatched &&
                 plan->get_state() == LLAMA_MEMPLAN_STATE_FAILED_PREPARE &&
                 plan->get_fail()  == LLAMA_MEMPLAN_FAIL_PREPARE &&
                 d1 == d0 && plan->select(0) == nullptr,
                 nm(s, "g", "ⓖfragmented/SWA 상태 + prepare **중간**(shadow 캡처 도중) 실패 — "
                            "이미 얹힌 부분 그림자 변이를 실복원해 전체 metadata 무변이 FAILED_PREPARE"));

        sc.check(!plan->apply(0) && plan->get_fail() == LLAMA_MEMPLAN_FAIL_STALE,
                 nm(s, "l2", "ⓛ폐기(FAILED_PREPARE) plan 재사용 거부"));

        // ⓖ2 — allocation preflight(§6 Q-B)는 shadow apply **이전**에 선다
        const uint64_t d2 = fx_digest(g.mem.get());

        llama_memplan::test_set_alloc_budget(16);

        auto plan2 = llama_memplan::prepare(g.mem.get(), *g.balloc, s.n_ubatch, false, true);

        llama_memplan::test_set_alloc_budget(UINT64_MAX);

        sc.check(plan2->get_state() == LLAMA_MEMPLAN_STATE_FAILED_PREPARE &&
                 plan2->get_fail()  == LLAMA_MEMPLAN_FAIL_ALLOC &&
                 plan2->get_preflight_bytes() > 16 &&
                 fx_digest(g.mem.get()) == d2,
                 nm(s, "g2", "ⓖ2allocation preflight(§6 Q-B) — 전 capture 표면 크기를 shadow apply 시작 전에 "
                             "계산·overflow 검사하고 상한 초과 시 무변이 FAILED_PREPARE/FAIL_ALLOC"
                             "(production 기본 상한은 무제한 · 상한은 test 훅으로만 주입)"));
    }

    // ---------------------------------------------------------------
    // ⓖ3 / ⓖ4 — **실할당 실패**(bad_alloc) 결정론 주입 (r25 수리 2b-3)
    //   ⓖ2 는 "상한 초과로 거부"만 보지만, 여기는 실제 할당이 터지는 경로를 연다.
    // ---------------------------------------------------------------
    {
        fx_inst ga;
        fx_inst clean;

        ga.init(s, model, s.n_tok_seq, 0);
        clean.init(s, model, s.n_tok_seq, 0);

        // origin 을 비우지 않는다(무변이 · clear 가 관측상 구별되게)
        const legacy_run pre  = legacy_execute(ga.mem.get(), *ga.balloc, s.n_ubatch, false, true);
        const bool       reba = ga.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq);

        const uint64_t d_origin = fx_digest(ga.mem.get());
        const uint64_t d_clean  = fx_digest(clean.mem.get());

        // ⓖ3: phase 1.5 선할당에서 bad_alloc → shadow 진입 전 실패 = 무변이
        llama_memplan::test_set_alloc_fail_at(1);

        auto p3 = llama_memplan::prepare(ga.mem.get(), *ga.balloc, s.n_ubatch, false, true);

        llama_memplan::test_set_alloc_fail_at(-1);

        sc.check(pre.ok && reba && d_origin != d_clean &&
                 p3->get_state() == LLAMA_MEMPLAN_STATE_FAILED_PREPARE &&
                 p3->get_fail()  == LLAMA_MEMPLAN_FAIL_ALLOC &&
                 p3->select(0)   == nullptr &&
                 fx_digest(ga.mem.get()) == d_origin,
                 nm(s, "g3", "ⓖ3실할당 실패(bad_alloc) — 전 캐처 표면 선할당(phase 1.5)에서 터져도 "
                             "shadow 진입 전이므로 예외 탈출 0 · memory 무변이 FAILED_PREPARE/FAIL_ALLOC"));

        // ⓖ4: shadow 부분 변이 후 **복원 경로**에서 bad_alloc → fail-close clear 종결
        fx_inst gr;
        gr.init(s, model, s.n_tok_seq, 0);

        const legacy_run pre2  = legacy_execute(gr.mem.get(), *gr.balloc, s.n_ubatch, false, true);
        const bool       reba2 = gr.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq);

        const uint64_t d_origin2 = fx_digest(gr.mem.get());

        llama_memplan::test_set_prepare_fail_at(1);   // shadow 가 ub0/ub1 을 실제로 얹고 실패
        llama_memplan::test_set_alloc_fail_at(2);     // 그 뒤 복원에서 할당 실패

        auto p4 = llama_memplan::prepare(gr.mem.get(), *gr.balloc, s.n_ubatch, false, true);

        llama_memplan::test_set_prepare_fail_at(-1);
        llama_memplan::test_set_alloc_fail_at(-1);

        sc.check(pre2.ok && reba2 && d_origin2 != d_clean &&
                 p4->get_state() == LLAMA_MEMPLAN_STATE_FAILED_PREPARE &&
                 p4->get_fail()  == LLAMA_MEMPLAN_FAIL_RESTORE &&
                 p4->select(0)   == nullptr &&
                 fx_digest(gr.mem.get()) == d_clean,
                 nm(s, "g4", "ⓖ4복원 경로 할당 실패 — shadow 부분 변이 뒤 복원이 bad_alloc 으로 죽어도 "
                             "예외가 탈출하지 않고(plan 을 정상 반환) 절반 상태 대신 fail-close clear 로 종결"));
    }

    // ---------------------------------------------------------------
    // ⓖ5 — shadow 가 실제로 재할당 없이 돌았는가(수리 2b-1 의 직접 오라클)
    // ---------------------------------------------------------------
    {
        fx_inst gc;
        gc.init(s, model, s.n_tok_seq, 0);

        auto plan = llama_memplan::prepare(gc.mem.get(), *gc.balloc, s.n_ubatch, false, true);

        const bool prepared = plan->get_state() == LLAMA_MEMPLAN_STATE_PREPARED;
        const bool sized    = plan->get_prealloc_cap_sig() > 0;
        const bool no_re    = plan->shadow_was_realloc_free();

        plan->abort();

        sc.check(prepared && sized && no_re,
                 nm(s, "g5", "ⓖ5shadow 재할당 0 — phase 1.5 선할당 직후와 shadow 종료 직후의 전 벡터 capacity "
                             "합계가 같다(= 변이 구간에서 새 할당이 일어나지 않았다는 실측). "
                             "★잔여 할당 1건은 사양상 불가피: mem_cell::seq_id 가 std::set 이라 셀 복사 시 "
                             "노드 할당이 생긴다(벡터 저장소 자체는 선확보됨)"));
    }

    // ---------------------------------------------------------------
    // ⓖ6 — **이중 실패**(지속 OOM): 복원도 실패하고 실패 문자열 대입까지 실패한다
    //   r26 수리 2c-3. set_fail 이 실질 noexcept 가 아니면 여기서 예외가 탈출해
    //   복원/clear 에 도달하지 못한다 = 부분 변이 잔존.
    // ---------------------------------------------------------------
    {
        fx_inst gd;
        fx_inst clean;

        gd.init(s, model, s.n_tok_seq, 0);
        clean.init(s, model, s.n_tok_seq, 0);

        const legacy_run pre  = legacy_execute(gd.mem.get(), *gd.balloc, s.n_ubatch, false, true);
        const bool       reba = gd.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq);

        const uint64_t d_origin = fx_digest(gd.mem.get());
        const uint64_t d_clean  = fx_digest(clean.mem.get());

        // shadow 가 ub0/ub1 을 실제로 얹은 뒤 실패 → 복원(지점 2)도 실패 → 그 사유 문자열
        // 대입(지점 3)까지 실패하는 "allocator 지속 고갈" 형상
        llama_memplan::test_set_prepare_fail_at(1);
        llama_memplan::test_set_alloc_fail_points((1u << 2) | (1u << 3));

        auto p6 = llama_memplan::prepare(gd.mem.get(), *gd.balloc, s.n_ubatch, false, true);

        llama_memplan::test_set_prepare_fail_at(-1);
        llama_memplan::test_set_alloc_fail_points(0);

        sc.check(pre.ok && reba && d_origin != d_clean &&
                 p6->get_state()     == LLAMA_MEMPLAN_STATE_FAILED_PREPARE &&
                 p6->get_fail()      == LLAMA_MEMPLAN_FAIL_RESTORE &&
                 p6->get_fail_note().empty() &&
                 p6->select(0)       == nullptr &&
                 fx_digest(gd.mem.get()) == d_clean,
                 nm(s, "g6", "ⓖ6이중 실패(지속 OOM) — 복원이 bad_alloc 으로 죽고 그 실패 사유 문자열 대입까지 "
                             "다시 bad_alloc 인 형상에서도 ①예외 탈출 0(plan 정상 반환) ②사유 enum 보존"
                             "(FAIL_RESTORE · note 는 빈 값) ③절반 상태 대신 fail-close clear 로 종결"));
    }

    // ---------------------------------------------------------------
    // ⓗ 비종결 destructor → 자동 abort + fail-close 기록
    // ---------------------------------------------------------------
    {
        fx_inst h;
        h.init(s, model, s.n_tok_seq, 0);

        const uint64_t d0     = fx_digest(h.mem.get());
        const uint64_t before = llama_memplan::test_destruct_abort_count();

        {
            auto plan = llama_memplan::prepare(h.mem.get(), *h.balloc, s.n_ubatch, false, true);
            plan->apply(0);
        }

        const uint64_t after = llama_memplan::test_destruct_abort_count();

        sc.check(after == before + 1 && fx_digest(h.mem.get()) == d0,
                 nm(s, "h2", "ⓗ비종결 destructor — 자동 abort 로 prepare 직전 상태 복원 + fail-close 기록 1건"));
    }

    // ---------------------------------------------------------------
    // ⓙ composite 원자성(하위 apply 실패 주입)
    // ---------------------------------------------------------------
    {
        fx_inst j1;
        j1.init(s, model, s.n_tok_seq, 0);

        auto p1 = llama_memplan::prepare(j1.mem.get(), *j1.balloc, s.n_ubatch, false, true);

        // ★호출 순서 trace(r24 수리 3) — 하위 변이 순서를 직접 단언한다.
        //   순서를 뒤집은 구현은 이 단언에서 반드시 FAIL 한다.
        std::vector<int32_t> exp_ok;
        std::vector<int32_t> exp_fail2;

        switch (s.kind) {
            case FX_KV:
                exp_ok    = { LLAMA_MEMPLAN_EV_KV_APPLY };
                exp_fail2 = { };                                   // 단일 하위 — 변이 전 실패
                break;
            case FX_HYBRID:
                exp_ok    = { LLAMA_MEMPLAN_EV_ATTN_APPLY, LLAMA_MEMPLAN_EV_RECURRENT_REPLAY };
                exp_fail2 = { LLAMA_MEMPLAN_EV_ATTN_APPLY };       // 두 번째 이벤트 직전 실패
                break;
            case FX_DSV4:
                exp_ok    = { LLAMA_MEMPLAN_EV_DSV4_RAW_BASE, LLAMA_MEMPLAN_EV_DSV4_RAW_SWA,
                              LLAMA_MEMPLAN_EV_DSV4_COMP_BOUNDARY };
                exp_fail2 = { LLAMA_MEMPLAN_EV_DSV4_RAW_BASE };
                break;
        }

        p1->test_trace_enable(true);

        const bool     a0        = p1->apply(0);
        const bool     order_ok  = (p1->test_trace() == exp_ok);
        const uint64_t d_after0  = fx_digest(j1.mem.get());

        p1->test_trace_enable(true);   // trace 초기화
        p1->test_inject_apply_fail(1, 1);

        const bool     f1   = !p1->apply(1) && p1->get_fail() == LLAMA_MEMPLAN_FAIL_APPLY;
        const bool     tr1  = p1->test_trace().empty();   // 첫 번째 하위 변이 전 실패 → 이벤트 0
        const uint64_t d_f1 = fx_digest(j1.mem.get());

        p1->abort();

        fx_inst j2;
        j2.init(s, model, s.n_tok_seq, 0);

        auto p2 = llama_memplan::prepare(j2.mem.get(), *j2.balloc, s.n_ubatch, false, true);

        const bool     b0        = p2->apply(0);
        const uint64_t d2_after0 = fx_digest(j2.mem.get());

        p2->test_trace_enable(true);
        p2->test_inject_apply_fail(1, 2);

        const bool     f2     = !p2->apply(1) && p2->get_fail() == LLAMA_MEMPLAN_FAIL_APPLY;
        const bool     tr2    = (p2->test_trace() == exp_fail2);
        const uint64_t d2_f2  = fx_digest(j2.mem.get());

        const bool n_applied_frozen = p2->get_n_applied() == 1;

        p2->abort();

        sc.check(a0 && order_ok && f1 && tr1 && d_f1 == d_after0 &&
                 b0 && f2 && tr2 && d2_f2 == d2_after0 && n_applied_frozen,
                 nm(s, "j", "ⓙ호출 순서 trace + composite 원자성(I3) — 성공 apply 의 하위 변이 순서가 계약 순서와 "
                            "정확히 일치(hybrid=[ATTN_APPLY,RECURRENT_REPLAY]) · 두 번째 이벤트 직전 실패 주입 시 "
                            "trace 가 첫 이벤트에서 멈추고 첫 번째 하위 변이는 보상 복원 · n_applied 미증가"));
    }

    // ---------------------------------------------------------------
    // ⓚ 하위 경계별 부분 실패 주입 — 전 composite 복원
    // ---------------------------------------------------------------
    // --- arm A: apply 단계 경계별 실패 주입(metadata composite 보상) ---
    {
        // ★계열이 실제로 갖는 하위 경계 수만 돈다. standard KV 는 하위가 1개라 경계 구분이 없다
        //   (r24 C 판정: 단일 하위 계열의 ⓙⓚ vacuous 는 비차단 — 허구의 경계를 만들지 않는다).
        const int subs_max = (s.kind == FX_DSV4) ? 6 : (s.kind == FX_HYBRID ? 2 : 1);

        bool all_ok = true;

        std::vector<std::string> notes;

        for (int sub = 1; sub <= subs_max; ++sub) {
            fx_inst k;
            k.init(s, model, s.n_tok_seq, 0);

            const uint64_t d_origin = fx_digest(k.mem.get());

            auto plan = llama_memplan::prepare(k.mem.get(), *k.balloc, s.n_ubatch, false, true);

            all_ok = all_ok && plan->apply(0);

            const uint64_t d_ref = fx_digest(k.mem.get());

            plan->test_inject_apply_fail(1, sub);

            all_ok = all_ok && !plan->apply(1);
            all_ok = all_ok && plan->get_fail() == LLAMA_MEMPLAN_FAIL_APPLY;
            all_ok = all_ok && fx_digest(k.mem.get()) == d_ref;

            notes.push_back(plan->get_fail_note());

            // 부분 apply 상태(APPLYING) 의 abort = metadata 복원
            all_ok = all_ok && plan->abort();
            all_ok = all_ok && fx_digest(k.mem.get()) == d_origin;
        }

        // 경계별 사유가 서로 구별되는가(같은 분기로 뭉뚱그려지지 않았는가)
        for (size_t i = 0; i < notes.size(); ++i) {
            for (size_t j = i + 1; j < notes.size(); ++j) {
                all_ok = all_ok && (notes[i] != notes[j]);
            }
        }

        sc.check(all_ok,
                 nm(s, "k", "ⓚarm A — 하위 경계별 실패 주입(DSV4 = raw base/raw swa/CSA/HCA/LID/compressor-state 6종 · "
                            "hybrid = attn/recurrent 2종 · standard KV = 단일 하위 1종): 매 경계에서 전 composite metadata 복원 후 실패 전파 · 사유가 경계별로 구별 · "
                            "이어지는 abort 로 prepare 직전 완전 복원"));
    }

    // --- arm B: 경계별 **실변이 sentinel** + APPLIED-abort 명시 clear (r24 수리 1·4) ---
    //   ★memplan_digest 는 metadata 전용이라 이 축의 오라클이 못 된다 — 텐서 바이트를 직접 본다.
    {
        boundary_probes bp0;
        fx_inst probe_ref;

        std::string why;   // 실패 지점을 이름에 드러낸다(단일 메가-AND 금지)

        if (!probe_ref.init(s, model, s.n_tok_seq, 0) || !bp0.build(s.kind, probe_ref.mem.get())) {
            why = "probe build failed";
        }

        const size_t n_bnd = bp0.v.size();

        for (size_t b = 0; b < n_bnd && why.empty(); ++b) {
            const std::string tag = std::string("[") + bp0.v[b].name + "] ";

            fx_inst k;
            fx_inst clean;

            if (!k.init(s, model, s.n_tok_seq, 0) || !clean.init(s, model, s.n_tok_seq, 0)) {
                why = tag + "fixture init failed";
                break;
            }

            // ★origin ≠ 빈 memory (metadata 축도 metadata-only 복원과 구별되게 한다).
            //   legacy 경로는 compute 를 돌리지 않으므로 물리 바이트는 여전히 0 이다.
            const legacy_run pre = legacy_execute(k.mem.get(), *k.balloc, s.n_ubatch, false, true);

            if (!pre.ok || !k.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq)) {
                why = tag + "pre-populate failed";
                break;
            }

            if (fx_digest(k.mem.get()) == fx_digest(clean.mem.get())) {
                why = tag + "origin equals clean (판정 공회전)";
                break;
            }

            boundary_probes bp;
            if (!bp.build(s.kind, k.mem.get())) {
                why = tag + "probe build failed";
                break;
            }

            auto plan = llama_memplan::prepare(k.mem.get(), *k.balloc, s.n_ubatch, false, true);

            for (uint32_t i = 0; i < K; ++i) {
                if (!plan->apply(i)) {
                    why = tag + "apply failed";
                    break;
                }
            }
            if (!why.empty()) { break; }

            // S1 의 apply 는 metadata 만 만진다 — 전 경계의 물리 바이트가 아직 0 이어야 한다
            for (size_t q = 0; q < n_bnd; ++q) {
                if (!sentinel_is_zero(bp.v[q])) {
                    why = tag + "pre-sentinel not zero at " + bp.v[q].name;
                    break;
                }
            }
            if (!why.empty()) { break; }

            // compute 가 이 경계에 물리 기록을 했다고 가정하고 구별되는 sentinel 을 심는다
            const uint8_t pat = (uint8_t) (0xA1 + b);
            sentinel_write(bp.v[b], pat);

            // ① 변이가 실제 발생했다
            if (!sentinel_is(bp.v[b], pat)) {
                why = tag + "sentinel write not observable";
                break;
            }

            // ③ 다른 경계에는 번지지 않았다(경계 독립 oracle)
            for (size_t q = 0; q < n_bnd; ++q) {
                if (q != b && sentinel_is(bp.v[q], pat)) {
                    why = tag + "sentinel leaked into " + bp.v[q].name;
                    break;
                }
            }
            if (!why.empty()) { break; }

            // ★APPLIED 에서의 abort = 물리 rollback 수단 없음 → 명시 clear(절반 상태 금지)
            const bool aborted = plan->abort();

            if (aborted || plan->get_fail() != LLAMA_MEMPLAN_FAIL_RESTORE ||
                    plan->get_state() != LLAMA_MEMPLAN_STATE_ABORTED) {
                why = tag + "APPLIED abort did not take the explicit-clear path";
                break;
            }

            // ② 물리까지 clear 되었다
            for (size_t q = 0; q < n_bnd; ++q) {
                if (!sentinel_is_zero(bp.v[q])) {
                    why = tag + "physical bytes survived clear at " + bp.v[q].name;
                    break;
                }
            }
            if (!why.empty()) { break; }

            // metadata 도 갓 생성한 memory 와 관측 동일
            if (fx_digest(k.mem.get()) != fx_digest(clean.mem.get())) {
                why = tag + "metadata not pristine after clear";
                break;
            }
        }

        if (n_bnd == 0 && why.empty()) {
            why = "no boundary probes";
        }

        sc.check(why.empty(),
                 nm(s, "k2", ("ⓚarm B — 경계별 **실변이 sentinel**(DSV4 = raw base/raw swa/CSA/HCA/LID/compressor-state 6종): "
                              "①심은 변이가 실제 관측되고 ③다른 경계로 번지지 않으며 ②APPLIED-abort 의 명시 clear 가 "
                              "물리 바이트까지 0 으로 되돌린다(metadata digest 가 아니라 텐서 바이트를 직접 판독)" +
                              (why.empty() ? std::string() : std::string(" — 실패: ") + why)).c_str()));
    }

    // ---------------------------------------------------------------
    // ⓛ abort 후 재-prepare = clean control 과 동일 view/digest
    // ---------------------------------------------------------------
    {
        fx_inst l1;
        fx_inst l2;

        l1.init(s, model, s.n_tok_seq, 0);
        l2.init(s, model, s.n_tok_seq, 0);

        auto p1 = llama_memplan::prepare(l1.mem.get(), *l1.balloc, s.n_ubatch, false, true);

        bool applied = true;
        for (uint32_t i = 0; i < K; ++i) {
            applied = applied && p1->apply(i);
        }

        // ★r24 수리 1: APPLIED(=apply_all 완료) 에서의 abort 는 metadata 복원이 아니라 명시 clear 다.
        const bool aborted        = p1->abort();
        const bool clear_semantic = !aborted &&
                                    p1->get_fail()  == LLAMA_MEMPLAN_FAIL_RESTORE &&
                                    p1->get_state() == LLAMA_MEMPLAN_STATE_ABORTED;

        auto p2 = llama_memplan::prepare(l1.mem.get(), *l1.balloc, s.n_ubatch, false, true);
        auto pc = llama_memplan::prepare(l2.mem.get(), *l2.balloc, s.n_ubatch, false, true);

        bool same = p2->get_n_ubatch() == pc->get_n_ubatch() &&
                    fx_digest(l1.mem.get()) == fx_digest(l2.mem.get());

        for (uint32_t i = 0; same && i < p2->get_n_ubatch(); ++i) {
            same = view_digest_xinstance(*p2->select(i)) == view_digest_xinstance(*pc->select(i));
        }

        const bool reuse_rejected = !p1->apply(0) && p1->get_fail() == LLAMA_MEMPLAN_FAIL_TERMINAL;

        p2->abort();
        pc->abort();

        sc.check(applied && clear_semantic && same && reuse_rejected,
                 nm(s, "l", "ⓛapply_all 후 abort → (APPLIED 이므로) 명시 clear 후단으로 종결하고, 같은 memory 재-prepare 가 "
                            "clean control 과 동일 view/digest + 폐기(ABORTED) plan 재사용 거부"));
    }

    // ---------------------------------------------------------------
    // ⓗ3 비종결 destructor(APPLIED) — abort() 와 같은 보수적 clear 규율
    // ---------------------------------------------------------------
    {
        fx_inst h3;
        fx_inst clean;

        h3.init(s, model, s.n_tok_seq, 0);
        clean.init(s, model, s.n_tok_seq, 0);

        // ★origin ≠ 빈 memory 로 만든다 — metadata-only 복원이면 origin 으로 돌아가므로 판정이 갈린다
        const legacy_run pre  = legacy_execute(h3.mem.get(), *h3.balloc, s.n_ubatch, false, true);
        const bool       reba = h3.rebatch(s, model, s.n_tok_seq, (llama_pos) s.n_tok_seq);

        const uint64_t d_origin = fx_digest(h3.mem.get());
        const uint64_t d_clean  = fx_digest(clean.mem.get());

        const uint64_t before = llama_memplan::test_destruct_abort_count();

        bool applied = true;
        {
            auto plan = llama_memplan::prepare(h3.mem.get(), *h3.balloc, s.n_ubatch, false, true);

            for (uint32_t i = 0; i < K; ++i) {
                applied = applied && plan->apply(i);
            }
            // APPLIED 인 채 종결 없이 소멸
        }

        const uint64_t after = llama_memplan::test_destruct_abort_count();

        sc.check(pre.ok && reba && applied && after == before + 1 &&
                 d_origin != d_clean &&                              // 전제: 두 결과가 실제로 갈린다
                 fx_digest(h3.mem.get()) == d_clean,
                 nm(s, "h3", "ⓗAPPLIED 상태 비종결 destructor — 자동 abort 가 metadata-only 복원(=origin 복귀)이 아니라 "
                             "명시 clear 로 종결(compute 가 물리 캐시에 썼을 수 있는 구간이므로 보수적)"));
    }

    // ---------------------------------------------------------------
    // I3 후단 — 복원 불가 형상은 절반 상태가 아니라 명시 clear
    // ---------------------------------------------------------------
    {
        fx_inst r;
        r.init(s, model, s.n_tok_seq, 0);

        auto plan = llama_memplan::prepare(r.mem.get(), *r.balloc, s.n_ubatch, false, true);

        plan->apply(0);
        plan->test_inject_restore_fail(true);

        const bool res = plan->abort();

        fx_inst clean;
        clean.init(s, model, s.n_tok_seq, 0);

        sc.check(!res && plan->get_fail() == LLAMA_MEMPLAN_FAIL_RESTORE &&
                 plan->get_state() == LLAMA_MEMPLAN_STATE_ABORTED &&
                 fx_digest(r.mem.get()) == fx_digest(clean.mem.get()),
                 nm(s, "e2", "ⓔI3 후단 — 복원 불가 형상에서는 절반 상태를 남기지 않고 해당 memory 를 명시 clear(빈 memory 와 관측 동일)"));
    }

    llama_memplan::test_reset_counters();
}

} // namespace

int llama_memplan_selfcheck(llama_memplan_selfcheck_cb cb, void * ud) {
    sc_ctx sc;

    sc.cb = cb;
    sc.ud = ud;

    fx_run_series(sc, FX_KV);
    fx_run_series(sc, FX_HYBRID);
    fx_run_series(sc, FX_DSV4);

    return sc.failed;
}
