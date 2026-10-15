#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// [MoE-Direct 레버 A 단계 1 / S2] S1 동결 뷰(실 정의는 llama-memplan.h — 여기서는 포인터만).
struct llama_memplan_view;
class  llama_memplan;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    // =======================================================================
    // ★C0 그래프 역할 격리 (SPEC_SHADOW_DRAFT §3 C0 — 섀도 드래프트 관측)
    //   이 컨텍스트가 만드는 모든 graph 의 역할. 기본 MAIN 이라 기존 경로는 무변이며,
    //   섀도 드라이버가 만드는 **사설 컨텍스트**만 SHADOW 로 설정한다. 판정값이
    //   컨텍스트-로컬이므로 main 과 섀도가 동시에 존재해도 서로의 관측을 끄지 않는다
    //   (전역 임시 OFF 플래그 금지 조항의 이행).
    // =======================================================================
    void     set_observer_role(uint32_t role);
    uint32_t get_observer_role() const { return observer_role; }
    bool     is_shadow_role()    const { return observer_role == LLM_GRAPH_ROLE_SHADOW; }

    // =======================================================================
    // ★섀도 드래프트 드라이버 (SPEC_SHADOW_DRAFT §1 D1~D6 · §3 C1/C2/C4)
    //   토큰이 확정된 **직후** 동기로 1회 호출한다. 게이트(MOE_SHADOW_DRAFT)가 OFF 면
    //   어떤 상태도 만들지 않고 즉시 0 을 돌려준다.
    //   반환: 실행한 섀도 스텝 수(0 = 미실행/미트리거) · 음수 = 거부 또는 실패 코드.
    // =======================================================================
    enum shadow_rc {
        SHADOW_RC_GATE_OFF      = -1,   // 게이트 OFF
        SHADOW_RC_SCOPE_HYBRID  = -2,   // hybrid/recurrent 계열 = v0.2 스코프 밖(명시 거부)
        SHADOW_RC_NO_CACHE      = -3,   // direct 미seal / view 진입 실패
        SHADOW_RC_NO_CTX        = -4,   // 사설 컨텍스트 생성 실패
        SHADOW_RC_CHECKPOINT    = -5,   // KV full sequence checkpoint 실패
        SHADOW_RC_RESTORE_DIGEST= -6,   // 복원 digest 등식 위반
        SHADOW_RC_VIEW_END      = -7,   // C3 진입/탈출 등식 위반(fail-close)
        SHADOW_RC_DECODE        = -8,   // 섀도 decode 실패
    };

    // ★r1 차단 1 수리 — confirmed_token 추가(j=1 의 실제 입력) · 트리거는 요청별 PA ordinal 로.
    int shadow_draft_run(llama_seq_id seq_id,
                         llama_token  confirmed_token,
                         uint32_t     request_index,
                         uint32_t     origin_graph_seq,
                         uint32_t     origin_decode_ordinal);

    // P43 엔진측 픽스처 진입점 — 하네스는 PASS/FAIL 회계만 소유한다(P41/P42 관행).
    static int shadow_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud);

    // =======================================================================
    // [MoE-Direct 레버 A 단계 1 / S2] 층-우선(layer-major) 프리필
    //   권위: bench/techdev/superbatch_s2/SPEC_S2_RANGEGRAPH.md v1.0
    //   ★기본 OFF. `MOE_DIRECT_PREFILL_LAYERMAJOR=1` 옵트인 + admission allowlist 를 전부
    //     통과한 `llama_decode` 배치에서만 진입하며, 미달은 **전부 현행 경로로 fail-close** 한다.
    // =======================================================================

    // admission 판별 결과 — 거부 사유를 관측 가능하게 남긴다(G-L2-S2 판정축).
    enum moe_s2_admit_code {
        MOE_S2_ADMIT_OK = 0,
        MOE_S2_ADMIT_NO_ENV,          // 옵트인 미설정
        MOE_S2_ADMIT_NO_ARCH,         // arch != qwen35moe
        MOE_S2_ADMIT_NO_DECODER,      // text decoder 그래프가 아님
        MOE_S2_ADMIT_NO_SINGLE_SEQ,   // 배치에 여러 seq 가 섞임
        MOE_S2_ADMIT_NO_K,            // K < 2 (K=1 꼬리·decode)
        MOE_S2_ADMIT_NO_CUDA,         // CUDA 기준 경로가 아님
        MOE_S2_ADMIT_NO_ADAPTER,      // cvec / lora 가 붙어 있음
        MOE_S2_ADMIT_NO_MEMORY,       // hybrid memory 가 아님
        MOE_S2_ADMIT_NO_EMBD,         // embeddings/nextn/sampler 등 비표준 출력 형상
        MOE_S2_ADMIT_NO_OBSERVER,     // PA/PB(프리페치 포함) 관측 모듈이 켜져 있음 — 상호 배타
    };

    static const char * moe_s2_admit_name(moe_s2_admit_code c);

    // ★admission 판별기 — **순수 판정**이라 selftest 가 형상별로 직접 검증할 수 있다.
    //   (실 게이트는 decode() 안에서 이 결과를 그대로 쓴다.)
    moe_s2_admit_code moe_s2_admit(const llama_batch & batch_inp, uint32_t n_tokens_all,
                                   uint32_t n_ubatch_planned) const;

    // 계측 — 잠금 형상 오라클(`range_build=720 · range_reuse=0`).
    uint64_t moe_s2_range_build_count() const { return moe_s2_range_build; }
    uint64_t moe_s2_range_reuse_count() const { return moe_s2_range_reuse; }

    // 실패가 발생한 단계 — 처분표(§3)의 입력축.
    enum moe_s2_stage {
        MOE_S2_STAGE_ADMISSION = 0,   // pre-APPLIED
        MOE_S2_STAGE_BANK,            // pre-APPLIED
        MOE_S2_STAGE_PREPARE,         // pre-APPLIED
        MOE_S2_STAGE_PREPARED_GUARD,  // pre-APPLIED (K 불일치·output_reserve 등 — 아직 무변이)
        MOE_S2_STAGE_APPLY,           // APPLIED 이후
        MOE_S2_STAGE_RANGE_BUILD,     // APPLIED 이후
        MOE_S2_STAGE_RANGE_ALLOC,     // APPLIED 이후
        MOE_S2_STAGE_RANGE_COMPUTE,   // APPLIED 이후
        MOE_S2_STAGE_COMMIT,          // APPLIED 이후
    };

    struct moe_s2_disposition {
        bool handled;      // false = 호출자가 현행 경로로 처리(legacy fallback)
        int  rc;           // handled 일 때의 decode() 반환값 — ★절대 2 가 아니다
        bool abort_plan;   // plan->abort() 후 ABORTED+clear 확인이 필요한가
        bool free_bank;    // bank 정리가 필요한가(항상 참 — 명시적으로 둔다)
    };

    // ★§3 실패 의미론의 **유일 원천**. 드라이버가 이 표를 그대로 쓰고, P42 픽스처가 이 표를
    //   직접 검증한다("ret=2 는 어떤 입력에서도 나오지 않는다"를 기계적으로 잠근다).
    static moe_s2_disposition moe_s2_dispose(moe_s2_stage stage, ggml_status st);

    // 실패 정리의 **단계** — 무엇을 하느냐(처분표)와 별개로 **어느 순서로 하느냐**가 계약이다.
    enum moe_s2_fail_step {
        MOE_S2_FAIL_STEP_SYNC = 0,   // ★항상 첫 단계 — in-flight range 커널 drain
        MOE_S2_FAIL_STEP_ABORT,      // plan->abort() + ABORTED 확인(= memplan 의 clear)
        MOE_S2_FAIL_STEP_FREE,       // 2-bank 해제
        MOE_S2_FAIL_STEP_TRACE,      // lmtrace range 태그 clear + request abort
    };

    // ★실패 정리 **순서**의 유일 원천. 드라이버는 실제 동작을, P42 픽스처는 기록만 하는 `run` 을
    //   넘겨 **같은 코드**를 돌린다 — 순서가 여기서만 정의되므로 픽스처가 실경로 순서를 잠근다
    //   (§3 처분을 `moe_s2_dispose` 가 단독 소유하는 것과 같은 구조).
    //   ★SYNC 가 맨 앞인 이유: range 그래프는 `graph_compute()` 가 비동기로 제출하므로 실패
    //     시점에 in-flight 커널이 남아 있을 수 있다. abort 의 clear(물리 zero-fill)나 bank 해제가
    //     그 커널보다 먼저 끝나면 늦게 도착한 쓰기가 "비었어야 할" 상태를 덮어 재시도 안전이
    //     깨진다. 조건 분기 없이 **전 단계 일괄** 1회 — 실패 경로의 sync 비용은 무시 가능하다.
    template <typename Run>
    static void moe_s2_fail_cleanup(const moe_s2_disposition & d, bool have_plan, bool trace_on, Run run) {
        run(MOE_S2_FAIL_STEP_SYNC);

        if (d.abort_plan && have_plan) {
            run(MOE_S2_FAIL_STEP_ABORT);
        }

        if (d.free_bank) {
            run(MOE_S2_FAIL_STEP_FREE);
        }

        if (d.handled && trace_on) {
            run(MOE_S2_FAIL_STEP_TRACE);
        }
    }

    // ★예외 unwind 경로의 처분(r32 차단 1 수리). 반환값 실패는 전부 `fail_at` 를 통과하지만,
    //   APPLIED 구간에서 C++ 예외(`model.build_graph()` 의 std::bad_alloc 등 host 메모리 압박)가
    //   던져지면 그 지점을 **건너뛴다**. 그대로 두면 `llama_memplan` 소멸자가 APPLIED 를
    //   `seq_fail_close_clear()` 로 물리 clear 하는데(llama-memplan.cpp) scheduler sync 가 선행되지
    //   않아, 앞선 range 의 in-flight 커널이 clear 된 KV/R/S 를 나중에 덮어쓴다.
    //   ★단계 선택 근거: APPLIED 구간의 예외는 "APPLIED 이후 · 비할당 실패"라는 축이 post-APPLIED
    //     실패와 완전히 같다 — §3 표의 RANGE_COMPUTE/FAILED 행(abort_plan·free_bank·handled 전부
    //     참)을 **그대로 재사용**하고 새 행을 만들지 않는다. rc 는 이 경로에서 쓰이지 않는다
    //     (예외는 정리 후 그대로 재전파되며 rc 로 변환되지 않는다 — legacy 경로와 같은 의미).
    static moe_s2_disposition moe_s2_unwind_dispose() {
        return moe_s2_dispose(MOE_S2_STAGE_RANGE_COMPUTE, GGML_STATUS_FAILED);
    }

    // 예외 unwind 정리의 **유일 진입점** — 실경로(가드 소멸자)와 P42h 픽스처가 같은 함수를 돈다.
    //   순서는 `moe_s2_fail_cleanup` 이 단독 소유하므로 SYNC 선행이 여기서도 자동으로 성립한다.
    template <typename Run>
    static void moe_s2_unwind_cleanup(bool have_plan, bool trace_on, Run run) {
        moe_s2_fail_cleanup(moe_s2_unwind_dispose(), have_plan, trace_on, run);
    }

    // ★scope guard — 소멸 시 armed 이면 `fn()` 을 부른다. **예외를 흡수하지 않는다**(정리만 하고
    //   원 예외는 그대로 전파된다 — 예외→rc 변환이라는 새 의미를 만들지 않는다).
    //   ★`fn` 은 **no-throw 계약**이다: unwind 중 소멸자 문맥에서 도는 코드라 예외가 새면
    //     std::terminate 다(S1 수리 2c 의 noexcept 전례). 실경로 fn 이 부르는
    //     sched_synchronize / plan->abort / bank free / lmtrace 는 전부 no-throw 다.
    //   ★armed 창 = **plan 상태 APPLIED 확정 ~ commit 성공**(드라이버가 arm/dismiss 로 연다 —
    //     r33 차단 1 수리 3). APPLYING 까지 넓히면 그 구간 `abort()` 가 `snapshot_restore`(복사
    //     할당 경로)라 위 no-throw 계약이 깨진다 — 배치 근거는 llama-context.cpp 의 arm 지점 주석.
    template <typename Fn>
    class moe_s2_scope_guard {
    public:
        explicit moe_s2_scope_guard(Fn fn, bool armed = true) : fn_(fn), armed_(armed) {}

        ~moe_s2_scope_guard() {
            if (armed_) {
                fn_();
            }
        }

        void arm()     { armed_ = true;  }
        void dismiss() { armed_ = false; }

        moe_s2_scope_guard(const moe_s2_scope_guard &)             = delete;
        moe_s2_scope_guard & operator=(const moe_s2_scope_guard &) = delete;

    private:
        Fn   fn_;
        bool armed_;
    };

    // 2-bank(교대) — sched 아레나 **밖**의 동일 buft 영속 텐서 2개(KV 캐시 관용구).
    struct moe_s2_bank_t {
        ggml_context_ptr           ctx;
        ggml_backend_buffer_ptr    buf;
        ggml_tensor *              base = nullptr;   // F32 [n_embd, n_ubatch*K]
    };

    // ★핵심부는 buft 를 인자로 받는 static — 실 모델 없이도 픽스처가 검증할 수 있다.
    //   fail_at: -1 = 정상 · 0 = bank #1 할당 실패 주입 · 1 = bank #2 할당 실패 주입
    static bool     moe_s2_bank_alloc_into(moe_s2_bank_t bank[2], ggml_backend_buffer_type_t buft,
                                           int64_t n_embd, int64_t n_ubatch, uint32_t k, int fail_at);
    static void     moe_s2_bank_free_of  (moe_s2_bank_t bank[2]);
    static uint64_t moe_s2_bank_bytes_of (const moe_s2_bank_t bank[2]);

    static ggml_tensor * moe_s2_bank_slice_of(const moe_s2_bank_t bank[2], int idx,
                                              int64_t n_embd, int64_t n_ubatch, uint32_t k,
                                              ggml_context * ctx, uint32_t ub_idx, int64_t n_tokens);

    // -----------------------------------------------------------------------
    // ★G-L4-S2 라이브 실패 주입 레버 (게이트 런 전용 · 기본 **완전 무동작**)
    //   env `MOE_DIRECT_S2_FAIL_INJECT` = `"<stage>:<r>:<ub>"` · stage ∈ build|alloc|compute.
    //   ★신규 실패 경로를 만들지 않는다 — 층-우선 순회의 **기존 세 관문**(build null /
    //     sched_alloc 실패 / compute status)의 결과값만 뒤집어 `fail_at` 를 그대로 태운다.
    //   ★1회 발화 후 자동 해제(프로세스 수명 내 재발화 없음) — G-L4-S2 의 "실패 후 동일
    //     프롬프트 재시도가 정상 완주" 를 같은 프로세스에서 증명할 수 있어야 하기 때문이다.
    //   ★미설정·파싱 실패 = 무장 자체가 없다(발화 지점의 비용 = 무장 bool 1개 검사).
    // -----------------------------------------------------------------------
    enum moe_s2_inject_stage {
        MOE_S2_INJECT_NONE = 0,
        MOE_S2_INJECT_BUILD,     // model.build_graph() 결과를 null 취급 → GGML_STATUS_FAILED
        MOE_S2_INJECT_ALLOC,     // ggml_backend_sched_alloc_graph() 실패 취급 → ALLOC_FAILED
        MOE_S2_INJECT_COMPUTE,   // graph_compute() status 를 FAILED 취급
    };

    struct moe_s2_inject_t {
        moe_s2_inject_stage stage = MOE_S2_INJECT_NONE;
        int32_t             r     = -1;   // range 인덱스(= il_start · Q-A 로 range 폭 1층)
        int32_t             ub    = -1;   // ubatch 슬롯 인덱스
        bool                armed = false;
    };

    // 파서 — **순수 함수**라 P42i 픽스처가 정상/이상 입력을 직접 검증한다.
    //   실패 시 `out` 은 항상 기본값(NONE · 무장 해제)으로 남는다(부분 파싱 잔재 금지).
    static bool moe_s2_inject_parse(const char * spec, moe_s2_inject_t & out);

    // 1회 발화 상태기계 — (stage,r,ub) 전건 일치일 때만 true 이며 **그 자리에서 무장 해제**한다.
    static bool moe_s2_inject_fire(moe_s2_inject_t & st, moe_s2_inject_stage stage,
                                   int32_t r, int32_t ub);

    static const char * moe_s2_inject_stage_name(moe_s2_inject_stage s);

    // ★P42 selfcheck 진입점 — 내부 C++ 타입이 DLL 밖으로 나오지 않으므로 픽스처는 여기서 돌고
    //   결과만 콜백으로 나간다(S1 의 llama_memplan_selfcheck 관행 승계).
public:
    static int s2_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud);

private:
    // admission 시 1회 전량 할당. 부분 할당(#1 성공·#2 실패)은 정리하고 false 를 돌려준다.
    bool moe_s2_bank_alloc(int64_t n_embd, int64_t n_ubatch, uint32_t k);
    void moe_s2_bank_free();

    // 현재 살아 있는 bank 바이트(G-L2-S2 "live bank bytes=0" 판정축).
    uint64_t moe_s2_bank_bytes() const;

    // ub 슬롯의 슬라이스 뷰 — 서로 겹치지 않는다(고정 stride = n_ubatch 행).
    ggml_tensor * moe_s2_bank_slice(ggml_context * ctx, int bank, uint32_t ub_idx, int64_t n_tokens) const;

    // CUDA 기준 경로 판별기(기성 헬퍼 부재 — 신규). 전 층 device 가 GPU 이고 backend 이름이
    // CUDA 계열이어야 한다.
    bool moe_s2_is_cuda_path() const;

    // process_ubatch 의 range 형제 — can_reuse 를 **강제 bypass** 하고 (range,ub) 마다 빌드한다.
    llm_graph_result * process_ubatch_range(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
       const llama_memplan_view    * pv,
                           int32_t   il_start,
                           int32_t   il_end,
                          uint32_t   boundary_flags,
                       ggml_tensor * bank_in,
                       ggml_tensor * bank_out,
                       ggml_status & ret);

    // 층-우선 드라이버. 반환값은 decode() 의 반환 규약과 동일하다.
    //   ★pre-APPLIED 실패는 `handled=false` 로 나가 호출자가 현행 경로를 타게 하고,
    //     APPLIED 이후 실패는 plan abort 후 **fatal 음수**로 나간다(ret=2 절대 금지 — §3).
    int decode_layermajor(const llama_batch & batch_inp, uint32_t n_tokens_all,
                          uint32_t n_outputs_all, bool output_all, bool & handled);

public:

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    // [MoE-Direct S2] range 축을 실은 params — legacy 판을 만든 뒤 range 필드만 덧씌운다.
    llm_graph_params graph_params_range(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype,
            const llama_memplan_view     * pv,
                                 int32_t   il_start,
                                 int32_t   il_end,
                                uint32_t   boundary_flags,
                             ggml_tensor * bank_in,
                             ggml_tensor * bank_out) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // ★C0 그래프 역할(SPEC_SHADOW_DRAFT §3 C0). 기본 MAIN — 기존 컨텍스트는 전부 MAIN 이다.
    uint32_t observer_role = LLM_GRAPH_ROLE_MAIN;

    // [R-BC3] the Shadow state (PIMPL): llama_ctx_shadow_state holds the member that was here
    std::unique_ptr<struct llama_ctx_shadow_state> moe_shadow;
    // ★r1 차단 1 — 컨텍스트 수명 전역 카운터는 요청별 ordinal 계약과 어긋나 폐지했다.
    //   트리거 판정은 PA 가 준 요청별 origin_decode_ordinal 로만 한다.

    int  shadow_admission() const;         // 0 = 허용, 음수 = 거부 코드
    int  shadow_ensure_ctx();              // 사설 컨텍스트 지연 생성

    // [R-BC2] the S2 state (PIMPL): llama_ctx_s2_state holds the members that were here
    std::unique_ptr<struct llama_ctx_s2_state> moe_s2;

    // 계측 — SPEC §2 잠금 형상 오라클(range_build=720 · range_reuse=0)
    uint64_t moe_s2_range_build = 0;
    uint64_t moe_s2_range_reuse = 0;


    void moe_s2_inject_init();


    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
