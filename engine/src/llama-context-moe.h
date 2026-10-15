#pragma once

// [R-BC1] llama-context-moe.h - the seam between llama-context.cpp and its two selftest TUs
//   (RB_DESIGN 2-8 link closure (a) and (c), and the section H closure universe). The six seam
//   TYPES are defined here: the three moe types, and the three state save/load io classes the
//   core and the shadow selftest both construct. The seam functions and g_moe_bench, whose
//   definitions stay in llama-context.cpp, are declared here. Everything is in namespace
//   llama_ctx_moe. Internal to src/, not a public header.

#include "llama-context.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include "ggml-moe-shadow.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace llama_ctx_moe {

struct moe_bench_cycle {
    bool     open   = false;
    uint32_t req    = 0;
    uint32_t ord    = 0;
    uint64_t t0_qpc = 0;
    ggml_moe_shadow_cpu_times   cpu0 = {};
    ggml_moe_shadow_backend_acc acc  = {};
};

struct moe_bench_state {
    bool     staging     = false;
    uint64_t stage_t_qpc = 0;
    ggml_moe_shadow_cpu_times   stage_cpu = {};
    ggml_moe_shadow_backend_acc stage_acc = {};

    moe_bench_cycle cycle;

    // SHADOW 역할 전용 누산기 — 섀도 view 창(view_begin~view_end)마다 리셋한다.
    ggml_moe_shadow_backend_acc shadow_acc = {};

    uint64_t emitted = 0;   // 발행한 TARGET 사이클 수
    uint64_t dropped = 0;   // 미종결로 버린 사이클 수(요청 tail 등)
};

extern moe_bench_state g_moe_bench;

void moe_bench_acc_reset(ggml_moe_shadow_backend_acc & a);
void moe_bench_note_submit(bool is_shadow, uint64_t t_qpc);
void moe_bench_note_sync(bool is_shadow, uint64_t t0_qpc, uint64_t t1_qpc);
// [R-BC4] the core-reference declarations (RB_DESIGN 7-54): the core hooks call these after
//   their definitions moved to llama-context-observe.cpp.
void moe_bench_stage_begin();
void moe_bench_rotate();
extern std::atomic<uint32_t> s_moe_trace_graph_seq_next;

struct moe_s2_admit_inputs {
    bool     env_on            = false;
    bool     arch_is_qwen35moe = false;
    bool     is_text_decoder   = false;
    bool     single_seq        = false;
    uint32_t k_planned         = 0;
    bool     cuda_path         = false;
    bool     has_adapter       = true;
    bool     memory_is_hybrid  = false;
    bool     plain_output      = false;   // embeddings/nextn/layer-inp/backend sampler 없음
    bool     observers_off     = false;   // PA/PB(프리페치 포함) 전부 OFF
};

llama_context::moe_s2_admit_code moe_s2_admit_decide(const moe_s2_admit_inputs & in);

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};


// [R-BC2] the S2 seam: defined in llama-context.cpp, called by llama-context-s2.cpp
llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type);

} // namespace llama_ctx_moe

// [R-BC2] the S2 adapter (RB_DESIGN 2-8, the R-BC2 row): the tri-state answer, the view,
//   the host operation the run entry calls back through, and the two entries.
enum llama_ctx_s2_result { LLAMA_CTX_S2_NOT_ADMITTED = 0, LLAMA_CTX_S2_FALLTHROUGH = 1, LLAMA_CTX_S2_HANDLED = 2 };

struct llama_ctx_view {
    ggml_backend_sched_t sched;
    uint32_t             n_ubatch;
    uint32_t             n_outputs_all;
    bool                 output_all;
};

struct llama_ctx_host_ops {
    void * self;
    int (*layermajor)(void * self, const llama_batch & batch_inp, uint32_t n_tokens_all,
                      uint32_t n_outputs_all, bool output_all, bool & handled);
};

struct llama_ctx_s2_state {
    using moe_s2_bank_t   = llama_context::moe_s2_bank_t;
    using moe_s2_inject_t = llama_context::moe_s2_inject_t;
    // 사설 컨텍스트 지연 생성
    // -----------------------------------------------------------------------
    // [MoE-Direct 레버 A 단계 1 / S2] 층-우선 상태
    // -----------------------------------------------------------------------
    // env: MOE_DIRECT_PREFILL_LAYERMAJOR (기본 OFF — 최초 조회 시 1회 캐시)
    mutable int moe_s2_env_cache = -1;   // -1=미해석 0=off 1=on

    moe_s2_bank_t moe_s2_bank[2];
    uint32_t      moe_s2_bank_k        = 0;   // 할당 당시의 K
    int64_t       moe_s2_bank_n_ubatch = 0;   // 슬롯 stride(행)
    int64_t       moe_s2_bank_n_embd   = 0;
    // env: MOE_DIRECT_S2_FAIL_INJECT (기본 무동작 — 드라이버 최초 진입 시 1회 해석 후 캐시.
    //   admission env 캐시와 같은 관행이며, 발화 지점 비용은 `armed` bool 1개 검사다)
    moe_s2_inject_t moe_s2_inject;
    bool            moe_s2_inject_parsed = false;
    // lmtrace 태그의 decode_call 번호(admission 통과 호출마다 1 증가)
    uint64_t moe_s2_decode_call = 0;
    bool s2_allowed = true;
};

bool llama_ctx_s2_admit(llama_context & ctx, const llama_ctx_s2_state & st, const llama_batch & batch_inp,
                        uint32_t n_tokens_all, uint32_t n_ubatch, llama_ctx_s2_result & verdict);
llama_ctx_s2_result llama_ctx_s2_run(const llama_ctx_view & view, const llama_ctx_host_ops & ops, llama_context & ctx,
                                     const llama_batch & batch_inp, uint32_t n_tokens_all, int & rc);

// [R-BC3] the Shadow adapter (RB_DESIGN 2-8, the R-BC3 row): the host operation the Shadow runtime
//   calls back through on its own instance, and the Shadow state the core holds through moe_shadow.
struct llama_ctx_shadow_host_ops {
    void (*main_sync)(llama_context & ctx);
};

struct llama_ctx_shadow_state {
    // ★섀도 드래프트 사설 상태(C1) — main 과 공유하는 것은 **불변 모델 가중치뿐**이다.
    //   컨텍스트가 자기 자신의 타입을 unique_ptr 로 갖는 것은 불완전 타입 문제가 없다
    //   (llama_context 정의 안이라 완전 타입이다). 게이트 OFF 면 끝까지 nullptr 로 남는다.
    std::unique_ptr<llama_context> shadow_ctx;
    llama_ctx_shadow_host_ops host_ops = { [](llama_context & ctx) { ctx.synchronize(); } };
};
