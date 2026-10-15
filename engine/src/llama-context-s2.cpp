#include "llama-context.h"

#include "ggml.h"
#include "ggml-moe-trace.h" // G3-2c 계측(SPEC_G32C.md v4 §4-1, 배선 2단계) — graph_begin/end 훅
#include "ggml-moe-phasea.h" // 프리페치 Phase A 계측 — graph 경계·가중치 스냅샷·eval callback 설치
#include "ggml-moe-phaseb.h" // 프리페치 Phase B — graph 경계·router/post-norm 스냅샷·a0_ref tap
#include "ggml-moe-direct.h" // graph 형상 지문(PA 무관 metrics 채널 — SPEC_PHASEA_WIRE §10)
#include "ggml-moe-lmtrace.h" // 레버 A 단계 1 S0 — 요청 단위 고정비 분해 계측(기본 OFF)
#include "ggml-moe-shadow.h"  // ★섀도 드래프트 관측 — C0 역할 격리·C3 캐시 view·kind5/6 wire
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"   // [MoE-Direct S2] admission 의 hybrid 판별
#include "llama-memory-hybrid-iswa.h" // ★섀도 admission — qwen3.5 계열은 이 타입을 받는다
#include "llama-memory-recurrent.h"   // ★섀도 admission — recurrent 계열도 스코프 밖
#include "llama-memplan.h"         // [MoE-Direct S2] S1 plan/동결 뷰 — 층-우선 드라이버의 유일 경로
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>       // [MoE-Direct S2 r32 수리 2] P42h 예외 주입 심의 std::bad_alloc
#include <stdexcept>
#include <string>

#include "llama-context-moe.h" // [R-BC2] the seam and the adapter types

using namespace llama_ctx_moe; // [R-BC2] the promoted seam keeps its unqualified spelling

// [R-BC2] llama-context-s2.cpp - the S2 layer-major prefill runtime, moved out of
//   llama-context.cpp (RB_DESIGN 2-8, the R-BC2 row), and its adapter.

enum llama_ctx_s2_mode { LLAMA_CTX_S2_MODE_OFF = 0, LLAMA_CTX_S2_MODE_LAYERMAJOR = 1 };

struct llama_ctx_s2_admission {
    llama_context::moe_s2_admit_code code = llama_context::MOE_S2_ADMIT_NO_ENV;
    llama_ctx_s2_mode                mode = LLAMA_CTX_S2_MODE_OFF;
};

// ===========================================================================
// [MoE-Direct 레버 A 단계 1 / S2] 층-우선(layer-major) 프리필
//   권위: bench/techdev/superbatch_s2/SPEC_S2_RANGEGRAPH.md v1.0 (동결 · Codex r30 [ACCEPT])
//
//   현행: `for ub: for L`  →  S2: `for R(층 구간): for ub`
//   경계를 넘는 것은 잔차 스트림 **하나**(F32 [n_embd, n_tokens])뿐이며, sched 아레나 밖의
//   영속 GPU bank 텐서 2개를 교대로 써서 device 안에서만 옮긴다(host 왕복 금지).
//
//   ★기본 OFF. `MOE_DIRECT_PREFILL_LAYERMAJOR=1` 옵트인 + admission allowlist 전건 통과에서만
//     진입하고, 미달은 전부 현행 경로로 fail-close 한다(옵트인 OFF 시 행동 불변 계약).
// ===========================================================================

const char * llama_context::moe_s2_admit_name(moe_s2_admit_code c) {
    switch (c) {
        case MOE_S2_ADMIT_OK:            return "ok";
        case MOE_S2_ADMIT_NO_ENV:        return "no_env";
        case MOE_S2_ADMIT_NO_ARCH:       return "no_arch";
        case MOE_S2_ADMIT_NO_DECODER:    return "no_decoder";
        case MOE_S2_ADMIT_NO_SINGLE_SEQ: return "no_single_seq";
        case MOE_S2_ADMIT_NO_K:          return "no_k";
        case MOE_S2_ADMIT_NO_CUDA:       return "no_cuda";
        case MOE_S2_ADMIT_NO_ADAPTER:    return "no_adapter";
        case MOE_S2_ADMIT_NO_MEMORY:     return "no_memory";
        case MOE_S2_ADMIT_NO_EMBD:       return "no_embd";
        case MOE_S2_ADMIT_NO_OBSERVER:   return "no_observer";
    }
    return "unknown";
}

// ★순수 판정부 — 실 컨텍스트 없이도 형상별로 검증할 수 있게 값만 받는다(P42 픽스처의 오라클).
//   순서는 SPEC §1 의 allowlist 나열 순서를 따른다.
// [R-BC1] moe_s2_admit_inputs is defined in llama-context-moe.h. The decider leaves `static`
// for llama_ctx_moe, so moe_s2_admit and llama-context-s2-selftest.cpp call one definition.
namespace llama_ctx_moe {

llama_context::moe_s2_admit_code moe_s2_admit_decide(const moe_s2_admit_inputs & in) {
    if (!in.env_on)            return llama_context::MOE_S2_ADMIT_NO_ENV;
    if (!in.arch_is_qwen35moe) return llama_context::MOE_S2_ADMIT_NO_ARCH;
    if (!in.is_text_decoder)   return llama_context::MOE_S2_ADMIT_NO_DECODER;
    if (!in.memory_is_hybrid)  return llama_context::MOE_S2_ADMIT_NO_MEMORY;
    if (!in.single_seq)        return llama_context::MOE_S2_ADMIT_NO_SINGLE_SEQ;
    if (in.k_planned < 2)      return llama_context::MOE_S2_ADMIT_NO_K;
    if (!in.cuda_path)         return llama_context::MOE_S2_ADMIT_NO_CUDA;
    if (in.has_adapter)        return llama_context::MOE_S2_ADMIT_NO_ADAPTER;
    if (!in.plain_output)      return llama_context::MOE_S2_ADMIT_NO_EMBD;
    // ★SPEC §1 "PB/prefetch 는 layer-major 경로에서 OFF" 의 이행 — 방향만 뒤집었다.
    //   그 모듈들(ggml-moe-phasea/phaseb)은 §6 접점 목록 밖이고 외부에서 사유를 주입할 API 도 없다.
    //   그래서 **관측 모듈을 끄는 대신 layer-major 진입을 거부**한다: 새로 들어오는 옵트인 쪽이
    //   양보하는 것이 §1 의 "미등재 전부 현행 경로 fail-close" 규율과 같은 방향이고, 기존 관측
    //   런의 의미를 조용히 바꾸지 않는다. 사유는 admission 코드 `no_observer` 로 관측된다.
    if (!in.observers_off)     return llama_context::MOE_S2_ADMIT_NO_OBSERVER;
    return llama_context::MOE_S2_ADMIT_OK;
}

} // namespace llama_ctx_moe

bool llama_context::moe_s2_is_cuda_path() const {
    // ★기성 헬퍼가 없어 신규 작성(SURVEY Q5). "전 층 + output 이 GPU 이고 backend device 이름이
    //   CUDA 계열" 을 기준 경로로 본다 — 부분 오프로드·CPU 혼합은 전부 거부(fail-close).
    const int32_t n_layer_x = (int32_t) model.hparams.n_layer();
    if (n_layer_x <= 0) {
        return false;
    }

    const auto dev_ok = [](ggml_backend_dev_t dev) {
        if (!dev) {
            return false;
        }
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            return false;
        }
        const char * nm = ggml_backend_dev_name(dev);
        return nm != nullptr && std::strncmp(nm, "CUDA", 4) == 0;
    };

    for (int32_t il = 0; il < n_layer_x; ++il) {
        if (!dev_ok(model.dev_layer(il))) {
            return false;
        }
    }

    return dev_ok(model.dev_output());
}

llama_context::moe_s2_admit_code llama_context::moe_s2_admit(
        const llama_batch & batch_inp, uint32_t n_tokens_all, uint32_t n_ubatch_planned) const {
    moe_s2_admit_inputs in;

    // --- 옵트인 게이트(최초 1회 해석 후 캐시) ---
    if (moe_s2->moe_s2_env_cache < 0) {
        const char * v = getenv("MOE_DIRECT_PREFILL_LAYERMAJOR");
        moe_s2->moe_s2_env_cache = (v && v[0] == '1' && v[1] == '\0') ? 1 : 0;
    }
    in.env_on = (moe_s2->moe_s2_env_cache == 1);

    // ★기본 OFF 경로의 비용은 **캐시된 int 하나**로 끝난다 — 아래 수집(전 층 device 조회·
    //   adapter 스캔·배치 seq 스캔)은 옵트인이 켜졌을 때만 지불한다. 이렇게 하지 않으면
    //   "옵트인 OFF 시 라이브 경로 불변" 이 행동에서는 성립해도 비용에서 깨진다.
    if (!in.env_on) {
        return MOE_S2_ADMIT_NO_ENV;
    }

    in.arch_is_qwen35moe = (model.arch == LLM_ARCH_QWEN35MOE);
    // ★수리 4(게이트 런 1차 실측 적발 no_decoder — 26-08-07): 이 코드베이스에서 일반 텍스트
    //   디코드 **메인 컨텍스트**의 graph type 은 LLM_GRAPH_TYPE_DEFAULT 다 —
    //   `ctx_type_to_graph_type` 은 DEFAULT→DEFAULT · MTP→DECODER_MTP 만 반환하며(:54),
    //   LLM_GRAPH_TYPE_DECODER 는 enc-dec 디코더 전용이라 서버 메인 컨텍스트에서 도달
    //   불가(구 검사는 항상 거짓 = 층-우선 영구 비진입). 배제 의도는 MTP 드래프트 컨텍스트
    //   (server-context 가 cparams_dft.ctx_type=MTP 로만 생성)이므로 DEFAULT 등가가 SPEC §1
    //   "text decoder" 의 정확한 이행이다.
    in.is_text_decoder   = (ctx_type_to_graph_type(cparams.ctx_type) == LLM_GRAPH_TYPE_DEFAULT);
    in.memory_is_hybrid  = (dynamic_cast<llama_memory_hybrid *>(memory.get()) != nullptr);

    // --- 배치 단일 seq — balloc 에 고유 seq 접근자가 없어 원시 배치를 직접 훑는다(SURVEY Q5).
    //   ★"컨텍스트에 동시 다른 요청이 없다"는 엔진 단독으로 증명할 수 없다. 그것은 **서버
    //     스케줄링 불변식**이며 v1 은 `-np 1` 운용을 전제한다(SPEC §1). 여기서 검사하는 것은
    //     "이 배치 안에 여러 seq 가 섞이지 않았다" 까지다.
    {
        bool         ok  = (batch_inp.n_tokens > 0);
        llama_seq_id sid = -1;
        for (int32_t i = 0; ok && i < batch_inp.n_tokens; ++i) {
            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;
            if (ns != 1) {
                ok = false;
                break;
            }
            const llama_seq_id cur = batch_inp.seq_id ? batch_inp.seq_id[i][0] : 0;
            if (sid < 0) {
                sid = cur;
            } else if (sid != cur) {
                ok = false;
            }
        }
        in.single_seq = ok;
    }

    in.k_planned = (n_ubatch_planned > 0) ? ((n_tokens_all + n_ubatch_planned - 1) / n_ubatch_planned) : 0;

    in.cuda_path = moe_s2_is_cuda_path();

    // --- adapter(cvec/lora) 부재 ---
    {
        bool has = (loras && !loras->empty());
        if (!has && cvec) {
            const int32_t n_layer_x = (int32_t) model.hparams.n_layer();
            for (int32_t il = 0; il < n_layer_x; ++il) {
                if (cvec->tensor_for(il) != nullptr) {
                    has = true;
                    break;
                }
            }
        }
        in.has_adapter = has;
    }

    // --- 출력 형상이 평범한가(logits 만) ---
    {
        bool plain = !cparams.embeddings && !cparams.embeddings_nextn && sampling.samplers.empty();
        for (bool enabled : cparams.embeddings_layer_inp) {
            if (enabled) {
                plain = false;
                break;
            }
        }
        in.plain_output = plain;
    }

    // --- PA/PB(프리페치 포함) 관측 모듈이 전부 꺼져 있는가 ---
    //   PA 는 sched eval callback 을 설치해 split 을 graph view 로 재분할하고, PB 는 graph 경계
    //   회계를 ubatch-major 로 전제한다 — 둘 다 층-우선과 상호 배타다.
    in.observers_off = !ggml_moe_phasea_enabled() &&
                       !ggml_moe_phaseb_enabled() &&
                       !ggml_moe_phaseb_prefetch_on();

    return moe_s2_admit_decide(in);
}

// ---------------------------------------------------------------------------
// 2-bank — sched 컴퓨트 아레나 **밖**의 동일 buft 영속 텐서 2개(KV 캐시 관용구).
//   각 bank = F32 [n_embd, n_ubatch*K]. ub 슬롯 stride 는 n_ubatch 행 고정이라 슬라이스가
//   서로 겹치지 않는다. 잠금 형상: 3072×512×4×4B = 24MiB/bank · 2-bank = 48MiB.
// ---------------------------------------------------------------------------

// ★§3 실패 처분표 — 드라이버와 픽스처가 **같은 표**를 본다.
llama_context::moe_s2_disposition llama_context::moe_s2_dispose(moe_s2_stage stage, ggml_status st) {
    switch (stage) {
        // --- pre-APPLIED: 무변이 확인 후 legacy fallback (그 배치는 현행 경로가 정확히 1회 처리) ---
        case MOE_S2_STAGE_ADMISSION:
        case MOE_S2_STAGE_BANK:
        case MOE_S2_STAGE_PREPARE:
            return { /*handled=*/false, /*rc=*/0, /*abort_plan=*/false, /*free_bank=*/true };

        case MOE_S2_STAGE_PREPARED_GUARD:
            // plan 은 만들어졌지만 아직 apply 전 = 무변이. PREPARED 에서의 abort 로 종결만 한다.
            return { false, 0, true, true };

        // --- APPLIED 이후: 자동 내부 fallback 없음. abort 후 **fatal 음수**로 나간다 ---
        case MOE_S2_STAGE_APPLY:
        case MOE_S2_STAGE_COMMIT:
            return { true, -3, true, true };

        case MOE_S2_STAGE_RANGE_BUILD:
        case MOE_S2_STAGE_RANGE_ALLOC:
        case MOE_S2_STAGE_RANGE_COMPUTE:
            // ★ret=2(ABORTED) 금지: 현 서버는 그것을 빈 오류로 읽어 배치를 반감해 재시도하는데,
            //   clear 된 seq 위의 재시도는 치명 오염이다. GGML_STATUS_ABORTED 도 -3 으로 정규화.
            return { true, (st == GGML_STATUS_ALLOC_FAILED) ? -2 : -3, true, true };
    }

    return { true, -3, true, true };
}

uint64_t llama_context::moe_s2_bank_bytes_of(const moe_s2_bank_t bank[2]) {
    uint64_t total = 0;
    for (int i = 0; i < 2; ++i) {
        if (bank[i].buf) {
            total += (uint64_t) ggml_backend_buffer_get_size(bank[i].buf.get());
        }
    }
    return total;
}

void llama_context::moe_s2_bank_free_of(moe_s2_bank_t bank[2]) {
    for (int i = 0; i < 2; ++i) {
        bank[i].base = nullptr;
        bank[i].buf.reset();
        bank[i].ctx.reset();
    }
}

bool llama_context::moe_s2_bank_alloc_into(moe_s2_bank_t bank[2], ggml_backend_buffer_type_t buft,
                                           int64_t n_embd_bank, int64_t n_ubatch_bank, uint32_t k,
                                           int fail_at) {
    moe_s2_bank_free_of(bank);

    if (!buft || n_embd_bank <= 0 || n_ubatch_bank <= 0 || k == 0) {
        return false;
    }

    for (int i = 0; i < 2; ++i) {
        // ★부분 할당 정리(G-L2-S2 ③) — #2 에서 실패해도 #1 을 남기지 않는다.
        if (fail_at == i) {
            moe_s2_bank_free_of(bank);
            return false;
        }

        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*2,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        bank[i].ctx.reset(ggml_init(ip));
        if (!bank[i].ctx) {
            moe_s2_bank_free_of(bank);
            return false;
        }

        bank[i].base = ggml_new_tensor_2d(bank[i].ctx.get(), GGML_TYPE_F32,
                                          n_embd_bank, n_ubatch_bank*(int64_t) k);
        if (!bank[i].base) {
            moe_s2_bank_free_of(bank);
            return false;
        }
        ggml_format_name(bank[i].base, "s2_bank_%d", i);

        bank[i].buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(bank[i].ctx.get(), buft));
        if (!bank[i].buf) {
            moe_s2_bank_free_of(bank);
            return false;
        }
    }

    return true;
}

ggml_tensor * llama_context::moe_s2_bank_slice_of(const moe_s2_bank_t bank[2], int idx,
                                                  int64_t n_embd_bank, int64_t n_ubatch_bank, uint32_t k,
                                                  ggml_context * ctx, uint32_t ub_idx, int64_t n_tokens) {
    if (!ctx || idx < 0 || idx > 1 || !bank[idx].base) {
        return nullptr;
    }
    if (ub_idx >= k || n_tokens <= 0 || n_tokens > n_ubatch_bank) {
        return nullptr;
    }

    ggml_tensor * base = bank[idx].base;

    // 슬롯 stride 는 n_ubatch 행 고정 — 서로 다른 ub 의 슬라이스가 절대 겹치지 않는다.
    return ggml_view_2d(ctx, base, n_embd_bank, n_tokens, base->nb[1],
                        (size_t) ub_idx*(size_t) n_ubatch_bank*base->nb[1]);
}

// fail_at: -1 = 정상 (G-L2-S2 ③ 주입은 static 판을 통해서만 — 라이브 경로엔 주입점이 없다)
uint64_t llama_context::moe_s2_bank_bytes() const {
    return moe_s2_bank_bytes_of(moe_s2->moe_s2_bank);
}

void llama_context::moe_s2_bank_free() {
    // ★device 대기 필수 — `graph_compute()` 는 `ggml_backend_sched_graph_compute_async()` 만
    //   호출하므로 마지막 range 의 커널이 **아직 bank 를 읽고 있을 수 있다**. buffer 해제는
    //   host 측 free 라 스트림 순서를 따르지 않으므로, 여기서 동기화하지 않으면 device
    //   use-after-free 가 된다. bank 가 살아 있을 때만 지불한다(decode 호출당 최대 1회).
    if (moe_s2_bank_bytes() > 0) {
        ggml_backend_sched_synchronize(sched.get());
    }

    moe_s2_bank_free_of(moe_s2->moe_s2_bank);
    moe_s2->moe_s2_bank_k        = 0;
    moe_s2->moe_s2_bank_n_ubatch = 0;
    moe_s2->moe_s2_bank_n_embd   = 0;
}

bool llama_context::moe_s2_bank_alloc(int64_t n_embd_bank, int64_t n_ubatch_bank, uint32_t k) {
    moe_s2_bank_free();

    // ★buft 선택 = **층 0 device 의 기본 buffer type**.
    //   `model.select_buft(0)`(가중치 buft)이 아니라 이쪽을 쓰는 이유: llama_context 가
    //   sched 에 넘기는 compute arena buft 가 정확히 `ggml_backend_get_default_buffer_type(backend)`
    //   = 이 값이다(llama-context.cpp 의 backend_buft 구성). 그래야 경계 CPY 의 src(아레나 텐서)와
    //   dst(bank slice)가 **같은 buft** 로 잡혀 byte-preserving·동일 backend 계약이 성립한다.
    //   admission 이 전 층 CUDA GPU 를 요구하므로 층 0 device 가 곧 그 device 다.
    ggml_backend_dev_t bank_dev = model.dev_layer(0);
    ggml_backend_buffer_type_t buft = bank_dev ? ggml_backend_dev_buffer_type(bank_dev) : nullptr;

    if (!moe_s2_bank_alloc_into(moe_s2->moe_s2_bank, buft, n_embd_bank, n_ubatch_bank, k, /*fail_at=*/-1)) {
        return false;
    }

    moe_s2->moe_s2_bank_k        = k;
    moe_s2->moe_s2_bank_n_ubatch = n_ubatch_bank;
    moe_s2->moe_s2_bank_n_embd   = n_embd_bank;

    return true;
}

ggml_tensor * llama_context::moe_s2_bank_slice(ggml_context * ctx, int bank, uint32_t ub_idx,
                                               int64_t n_tokens) const {
    return moe_s2_bank_slice_of(moe_s2->moe_s2_bank, bank, moe_s2->moe_s2_bank_n_embd, moe_s2->moe_s2_bank_n_ubatch,
                                moe_s2->moe_s2_bank_k, ctx, ub_idx, n_tokens);
}

llm_graph_params llama_context::graph_params_range(
                    llm_graph_result * res,
                  const llama_ubatch & ubatch,
        const llama_memory_context_i * mctx,
                      llm_graph_type   gtype,
        const llama_memplan_view     * pv,
                             int32_t   il_start,
                             int32_t   il_end,
                            uint32_t   boundary_flags,
                         ggml_tensor * bank_in,
                         ggml_tensor * bank_out) const {
    llm_graph_params gparams = graph_params(res, ubatch, mctx, gtype);

    gparams.il_start       = il_start;
    gparams.il_end         = il_end;
    gparams.boundary_flags = boundary_flags;
    gparams.range_mode     = LLM_GRAPH_RANGE_MODE_RANGE;
    gparams.bank_in        = bank_in;
    gparams.bank_out       = bank_out;
    gparams.plan_view      = pv;

    return gparams;
}

llm_graph_result * llama_context::process_ubatch_range(
            const llama_ubatch & ubatch,
                llm_graph_type   gtype,
        llama_memory_context_i * mctx,
   const llama_memplan_view    * pv,
                       int32_t   il_start,
                       int32_t   il_end,
                      uint32_t   boundary_flags,
                   ggml_tensor * bank_in_base,
                   ggml_tensor * bank_out_base,
                   ggml_status & ret) {
    // ★process_ubatch 의 형제. 결정적 차이 셋:
    //   ① `mctx->apply()` 를 부르지 않는다 — 변이는 이미 plan 의 apply(0..K-1) 이 끝냈고,
    //      이 함수는 동결 뷰 `pv` 만 소비한다(SPEC §2).
    //   ② `can_reuse()` 를 **강제 bypass** 한다 — 같은 range 의 다음 ub 는 topology 가 같아도
    //      bank slice identity 가 달라 거짓 재사용이 된다(r29 수정 1 불변식).
    //   ③ 요청 신호는 진입당 1회로 이미 냈고 여기서는 segment 만 늘린다(§5 ⓑ).
    //   ★`mctx` 는 full-cache 뷰다(영속 캐시 텐서 접근 전용) — 배치 좌표는 전부 `pv` 에서 온다.

    const bool moe_lmtrace_on = ggml_moe_lmtrace_enabled();
    if (moe_lmtrace_on) {
        ggml_moe_lmtrace_range_segment();
    }

    auto * res = gf_res_prev.get();

    // ★reuse 강제 bypass — 계측 카운터로 잠금 형상 오라클(range_build=720 · range_reuse=0)을 고정.
    moe_s2_range_build++;

    res->reset();

    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

    // ★bank 슬라이스 뷰는 **reset 뒤** 그래프 ctx0 안에서 만든다(reset 이 ctx_compute 를 새로
    //   만들기 때문). 뷰의 원본은 아레나 밖 영속 텐서라 gallocr 가 손대지 않는다.
    ggml_context * ctx0 = res->get_ctx();

    ggml_tensor * bank_in  = nullptr;
    ggml_tensor * bank_out = nullptr;

    // base 포인터 → bank 인덱스. 어느 쪽과도 일치하지 않으면 -1 이고, slice 헬퍼가 nullptr 로
    // 떨어뜨려 fail-close 된다(누가 잘못된 base 를 넘겨도 조용히 다른 슬롯을 쓰지 않는다).
    const auto bank_index_of = [&](const ggml_tensor * base) {
        if (base && base == moe_s2->moe_s2_bank[0].base) return 0;
        if (base && base == moe_s2->moe_s2_bank[1].base) return 1;
        return -1;
    };

    if (boundary_flags & LLM_GRAPH_BOUNDARY_IN) {
        bank_in = moe_s2_bank_slice(ctx0, bank_index_of(bank_in_base), pv->ub_idx, ubatch.n_tokens);
        if (!bank_in) {
            LLAMA_LOG_ERROR("%s: failed to bind S2 boundary input slice\n", __func__);
            if (moe_lmtrace_on) ggml_moe_lmtrace_request_abort();
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    if (boundary_flags & LLM_GRAPH_BOUNDARY_OUT) {
        bank_out = moe_s2_bank_slice(ctx0, bank_index_of(bank_out_base), pv->ub_idx, ubatch.n_tokens);
        if (!bank_out) {
            LLAMA_LOG_ERROR("%s: failed to bind S2 boundary output slice\n", __func__);
            if (moe_lmtrace_on) ggml_moe_lmtrace_request_abort();
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    const auto gparams = graph_params_range(res, ubatch, mctx, gtype, pv,
                                            il_start, il_end, boundary_flags, bank_in, bank_out);

    if (moe_lmtrace_on) ggml_moe_lmtrace_stage_enter(GGML_MOE_LMTRACE_STAGE_GRAPH_BUILD);
    ggml_cgraph * gf = model.build_graph(gparams);
    if (moe_lmtrace_on) ggml_moe_lmtrace_stage_leave(GGML_MOE_LMTRACE_STAGE_GRAPH_BUILD);

    // ★G-L4-S2 주입점 ①build — 아래 기존 `!gf` 관문을 그대로 태운다(신규 경로 없음).
    //   range 인덱스는 `il_start` 다(Q-A: range 폭 1층 ⇒ il_start == r · 드라이버 :2274 호출부).
    if (moe_s2->moe_s2_inject.armed &&
        moe_s2_inject_fire(moe_s2->moe_s2_inject, MOE_S2_INJECT_BUILD, il_start, (int32_t) pv->ub_idx)) {
        LLAMA_LOG_WARN("%s: G-L4-S2 injected build failure at (range=%d, ub=%u)\n",
                       __func__, (int) il_start, (unsigned) pv->ub_idx);
        gf = nullptr;
    }

    if (!gf) {
        LLAMA_LOG_ERROR("%s: failed to initialize range graph\n", __func__);
        if (moe_lmtrace_on) ggml_moe_lmtrace_request_abort();
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    if (moe_lmtrace_on) ggml_moe_lmtrace_stage_enter(GGML_MOE_LMTRACE_STAGE_SCHED_ALLOC);
    bool sched_alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
    if (moe_lmtrace_on) ggml_moe_lmtrace_stage_leave(GGML_MOE_LMTRACE_STAGE_SCHED_ALLOC);

    // ★G-L4-S2 주입점 ②alloc — 아래 기존 `!sched_alloc_ok` 관문(ALLOC_FAILED)을 그대로 태운다.
    if (moe_s2->moe_s2_inject.armed &&
        moe_s2_inject_fire(moe_s2->moe_s2_inject, MOE_S2_INJECT_ALLOC, il_start, (int32_t) pv->ub_idx)) {
        LLAMA_LOG_WARN("%s: G-L4-S2 injected alloc failure at (range=%d, ub=%u)\n",
                       __func__, (int) il_start, (unsigned) pv->ub_idx);
        sched_alloc_ok = false;
    }

    if (!sched_alloc_ok) {
        LLAMA_LOG_ERROR("%s: failed to allocate range graph\n", __func__);
        if (moe_lmtrace_on) ggml_moe_lmtrace_request_abort();
        ret = GGML_STATUS_ALLOC_FAILED;
        return nullptr;
    }

    res->set_inputs(&ubatch);

    // ★경계 CPY 회계 — 정확값은 bytes/count 다. `boundary_copy_ms` 는 경계 CPY 를 **포함한**
    //   range 그래프의 host-submit 시간 상계일 뿐 CPY 단독 시간이 아니다(헤더 주석·SPEC §5 ⓑ).
    const bool has_bnd = (boundary_flags & LLM_GRAPH_BOUNDARY_OUT) != 0;

    if (moe_lmtrace_on) {
        ggml_moe_lmtrace_stage_enter(GGML_MOE_LMTRACE_STAGE_SUBMIT_HOST);
        if (has_bnd) ggml_moe_lmtrace_stage_enter(GGML_MOE_LMTRACE_STAGE_BOUNDARY_COPY);
    }

    auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);

    if (moe_lmtrace_on) {
        if (has_bnd) {
            ggml_moe_lmtrace_stage_leave(GGML_MOE_LMTRACE_STAGE_BOUNDARY_COPY);
            ggml_moe_lmtrace_boundary_copy_note((uint64_t) ggml_nbytes(bank_out), 1);
        }
        ggml_moe_lmtrace_stage_leave(GGML_MOE_LMTRACE_STAGE_SUBMIT_HOST);
    }

    ggml_moe_direct_graph_shape_note(ggml_graph_n_nodes(res->get_gf()),
                                     ggml_backend_sched_get_n_splits(sched.get()));

    // ★G-L4-S2 주입점 ③compute — 아래 기존 `status != SUCCESS` 관문을 그대로 태운다.
    //   ★배치 근거: 실제 compute 실패도 lmtrace stage leave·shape note 를 거친 뒤 이 관문에
    //     도달한다(비동기 제출이라 커널은 이미 in-flight 일 수 있다) — 주입이 그 상태를 그대로
    //     재현해야 `fail_at` 의 sync-first 정리(r31 수리 1)가 실제로 검증된다.
    if (moe_s2->moe_s2_inject.armed &&
        moe_s2_inject_fire(moe_s2->moe_s2_inject, MOE_S2_INJECT_COMPUTE, il_start, (int32_t) pv->ub_idx)) {
        LLAMA_LOG_WARN("%s: G-L4-S2 injected compute failure at (range=%d, ub=%u)\n",
                       __func__, (int) il_start, (unsigned) pv->ub_idx);
        status = GGML_STATUS_FAILED;
    }

    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute range graph, compute status: %d\n", __func__, status);
        if (moe_lmtrace_on) ggml_moe_lmtrace_request_abort();
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

// ---------------------------------------------------------------------------
// ★G-L4-S2 라이브 실패 주입 레버 — 게이트 런 전용(기본 완전 무동작)
//   `MOE_DIRECT_S2_FAIL_INJECT="<stage>:<r>:<ub>"` · stage ∈ build|alloc|compute.
//   신규 실패 경로를 만들지 않는다: 층-우선 순회의 **기존 세 관문**의 결과값만 뒤집어
//   `process_ubatch_range` 의 기존 실패 반환 → 드라이버의 `fail_at` 를 그대로 태운다.
// ---------------------------------------------------------------------------

const char * llama_context::moe_s2_inject_stage_name(moe_s2_inject_stage s) {
    switch (s) {
        case MOE_S2_INJECT_NONE:    return "none";
        case MOE_S2_INJECT_BUILD:   return "build";
        case MOE_S2_INJECT_ALLOC:   return "alloc";
        case MOE_S2_INJECT_COMPUTE: return "compute";
    }
    return "unknown";
}

// [begin,end) 구간이 **10진 숫자만으로 이루어진 비어 있지 않은 문자열**일 때만 수용한다.
//   부호·공백·접두 '+'·후행 쓰레기는 전부 거부(관대한 strtol 대신 엄격 파서 — 오설정을
//   조용히 흡수하면 게이트 런에서 "주입했는데 안 터졌다" 가 미스터리가 된다).
static bool moe_s2_inject_parse_idx(const char * begin, const char * end, int32_t & out) {
    if (!begin || !end || begin >= end || (end - begin) > 9) {
        return false;
    }

    int32_t v = 0;
    for (const char * p = begin; p < end; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        v = v*10 + (*p - '0');
    }

    out = v;
    return true;
}

bool llama_context::moe_s2_inject_parse(const char * spec, moe_s2_inject_t & out) {
    out = moe_s2_inject_t();   // ★부분 파싱 잔재 금지 — 실패는 항상 "무장 없음" 으로 끝난다

    if (!spec || spec[0] == '\0') {
        return false;
    }

    const char * c1 = std::strchr(spec, ':');
    if (!c1) {
        return false;
    }
    const char * c2 = std::strchr(c1 + 1, ':');
    if (!c2) {
        return false;
    }
    if (std::strchr(c2 + 1, ':')) {
        return false;   // 필드 4개 이상 = 형식 위반
    }

    const size_t n_stage = (size_t) (c1 - spec);

    moe_s2_inject_stage stage = MOE_S2_INJECT_NONE;
    if      (n_stage == 5 && std::strncmp(spec, "build",   5) == 0) stage = MOE_S2_INJECT_BUILD;
    else if (n_stage == 5 && std::strncmp(spec, "alloc",   5) == 0) stage = MOE_S2_INJECT_ALLOC;
    else if (n_stage == 7 && std::strncmp(spec, "compute", 7) == 0) stage = MOE_S2_INJECT_COMPUTE;
    else return false;

    int32_t r = 0;
    int32_t u = 0;
    if (!moe_s2_inject_parse_idx(c1 + 1, c2, r)) {
        return false;
    }
    if (!moe_s2_inject_parse_idx(c2 + 1, spec + std::strlen(spec), u)) {
        return false;
    }

    out.stage = stage;
    out.r     = r;
    out.ub    = u;
    out.armed = true;

    return true;
}

bool llama_context::moe_s2_inject_fire(moe_s2_inject_t & st, moe_s2_inject_stage stage,
                                       int32_t r, int32_t ub) {
    if (!st.armed || st.stage != stage || st.r != r || st.ub != ub) {
        return false;
    }

    // ★1회 발화 후 자동 해제 — 뒤따르는 재시도 요청이 정상 주행해야 G-L4-S2 의
    //   "실패 후 재시도 정상"(상태 무오염의 종단 증거)을 같은 프로세스에서 얻을 수 있다.
    st.armed = false;

    return true;
}

void llama_context::moe_s2_inject_init() {
    if (moe_s2->moe_s2_inject_parsed) {
        return;
    }
    moe_s2->moe_s2_inject_parsed = true;

    const char * v = getenv("MOE_DIRECT_S2_FAIL_INJECT");
    if (!v || v[0] == '\0') {
        return;   // 미설정 = 완전 무동작(무장 없음)
    }

    if (!moe_s2_inject_parse(v, moe_s2->moe_s2_inject)) {
        LLAMA_LOG_WARN("%s: MOE_DIRECT_S2_FAIL_INJECT=\"%s\" is invalid "
                       "(expected \"<build|alloc|compute>:<range>:<ubatch>\") - injection disabled\n",
                       __func__, v);
        return;   // 파싱 실패 = 완전 무동작(경고 1줄)
    }

    LLAMA_LOG_WARN("%s: G-L4-S2 fail injection armed: stage=%s range=%d ubatch=%d "
                   "(one-shot - auto-disarms after firing)\n",
                   __func__, moe_s2_inject_stage_name(moe_s2->moe_s2_inject.stage),
                   (int) moe_s2->moe_s2_inject.r, (int) moe_s2->moe_s2_inject.ub);
}

int llama_context::decode_layermajor(const llama_batch & batch_inp, uint32_t n_tokens_all,
                                     uint32_t n_outputs_all, bool output_all, bool & handled) {
    GGML_UNUSED(batch_inp);

    handled = false;

    // ★주입 레버 해석은 드라이버 진입 시 1회(admission env 캐시와 같은 관행). 미설정이면
    //   `moe_s2_inject.armed` 가 false 로 남아 발화 지점 비용이 bool 검사 하나로 끝난다.
    moe_s2_inject_init();

    const auto & hparams = model.hparams;

    const int64_t  n_embd_bank = hparams.n_embd;
    const int32_t  n_ranges    = (int32_t) hparams.n_layer();   // Q-A: range 폭 = 1층
    const uint32_t n_ub        = cparams.n_ubatch;
    const uint32_t k_planned   = (n_ub > 0) ? ((n_tokens_all + n_ub - 1) / n_ub) : 0;
    const int64_t  n_vocab     = model.vocab.n_tokens();

    // ===== 실행 순서 계약(G-L2-S2 ②): admission → 2-bank 전량 할당 → prepare → apply =====

    llama_memplan_ptr plan;

    // ★모든 실패는 이 한 지점을 통과한다 — 처분은 §3 표(`moe_s2_dispose`)가, 정리 **순서**는
    //   `moe_s2_fail_cleanup`(헤더)이 단독으로 정한다.
    //   ★순서 제약(r31 차단 1): device 동기화가 **가장 먼저** 끝나야 한다. range 그래프는
    //     `graph_compute()` 가 비동기 제출하므로 실패 시점에 앞선 range 의 커널이 아직 KV/R/S
    //     상태에 쓰고 있을 수 있고, 그 전에 abort 의 clear(zero-fill)나 bank 해제가 끝나면 늦게
    //     도착한 쓰기가 "비었어야 할" 상태를 오염시킨다 — drain 뒤에만 clear/free 가 유효하다.
    //   ★정리 **동작**은 이 한 실행부가 소유한다 — 반환값 실패(`fail_at`)와 예외 unwind 가드가
    //     같은 코드를 돌리므로 두 경로의 정리 의미가 갈라질 수 없다.
    //     `noexcept`: unwind 중 소멸자 문맥에서도 돈다(헤더 `moe_s2_scope_guard` 의 no-throw 계약 —
    //     sync/abort/bank_free/lmtrace 는 전부 no-throw 다. S1 수리 2c 의 noexcept 전례).
    const auto run_step = [&](moe_s2_fail_step step) noexcept {
        switch (step) {
            case MOE_S2_FAIL_STEP_SYNC:
                // 실패 경로 1회 · 조건 분기 없음(단순성 우선 — 비용은 무시 가능).
                ggml_backend_sched_synchronize(sched.get());
                break;

            case MOE_S2_FAIL_STEP_ABORT:
                // abort() 의 반환값과 무관하게 ABORTED + clear 의미론을 **확인**한다(SPEC §3).
                plan->abort();
                if (plan->get_state() != LLAMA_MEMPLAN_STATE_ABORTED) {
                    LLAMA_LOG_ERROR("%s: memplan abort did not reach ABORTED (state=%d)\n",
                                    __func__, (int) plan->get_state());
                }
                break;

            case MOE_S2_FAIL_STEP_FREE:
                // bank 내용 = 비커밋 휘발(복원 의무 없음 — 정리만).
                moe_s2_bank_free();
                break;

            case MOE_S2_FAIL_STEP_TRACE:
                ggml_moe_lmtrace_range_tag_clear();
                ggml_moe_lmtrace_request_abort();
                break;
        }
    };

    // ★예외 unwind 가드(r32 차단 1 수리): APPLIED 구간에서 예외가 던져지면 `fail_at` 를 통과하지
    //   못한 채 plan 소멸자가 물리 clear 를 돌린다 — sync 선행 없이. 이 가드가 **소멸 전에**
    //   같은 정리(SYNC 선행)를 돌린다. 예외는 흡수하지 않고 그대로 전파된다.
    //   ★초기 disarmed: pre-APPLIED(bank_alloc ~ apply 루프 전건 완료 전 — **APPLYING 포함**)는
    //     제출된 range 가 0이라 in-flight 커널과의 경합 무대상이고, 그 구간의 소멸자 경로는
    //     metadata 복원(snapshot_restore)이 정답이다 — arm 은 apply **전건 성공 직후**(= plan 상태
    //     APPLIED 확정) 한 지점뿐이다(r33 차단 1 수리 3 — 배치 근거는 그 지점 주석).
    const auto unwind_run = [&]() noexcept {
        moe_s2_unwind_cleanup(plan != nullptr, ggml_moe_lmtrace_enabled(), run_step);
    };

    moe_s2_scope_guard<decltype(unwind_run)> unwind(unwind_run, /*armed=*/false);

    const auto fail_at = [&](moe_s2_stage stage, ggml_status st) {
        const moe_s2_disposition d = moe_s2_dispose(stage, st);

        moe_s2_fail_cleanup(d, plan != nullptr, ggml_moe_lmtrace_enabled(), run_step);

        // 이 경로는 정리를 이미 마쳤다 — 되돌아가는 길에 가드가 한 번 더 돌지 않게 한다.
        unwind.dismiss();

        handled = d.handled;
        return d.rc;
    };

    // --- ② 2-bank 전량 할당 (pre-APPLIED — 실패는 무변이 확인 후 legacy fallback) ---
    if (!moe_s2_bank_alloc(n_embd_bank, (int64_t) n_ub, k_planned)) {
        return fail_at(MOE_S2_STAGE_BANK, GGML_STATUS_ALLOC_FAILED);
    }

    // --- ③ plan prepare (pre-APPLIED) ---
    plan = llama_memplan::prepare(memory.get(), *balloc, n_ub, output_all, cparams.causal_attn);

    if (!plan || plan->get_state() != LLAMA_MEMPLAN_STATE_PREPARED) {
        // FAILED_PREPARE = 무변이 계약(S1 §1). plan 은 이미 종결 상태이므로 abort 하지 않는다.
        plan.reset();
        return fail_at(MOE_S2_STAGE_PREPARE, GGML_STATUS_FAILED);
    }

    const uint32_t k = plan->get_n_ubatch();

    // ★계획 K 를 넘으면 bank 슬롯이 모자란다 — 아직 apply 전이므로 무변이 abort 후 legacy 로.
    if (k < 2 || k > k_planned) {
        return fail_at(MOE_S2_STAGE_PREPARED_GUARD, GGML_STATUS_FAILED);
    }

    if (output_reserve(n_outputs_all) < n_outputs_all) {
        return fail_at(MOE_S2_STAGE_PREPARED_GUARD, GGML_STATUS_ALLOC_FAILED);
    }

    // ===== 여기서부터 APPLIED 구간 — 실패는 전부 fatal 음수다(ret=2 절대 금지 · SPEC §3) =====

    // --- ④ apply 0 → K-1 (canonical 순서 강제 — S1 I2) ---
    for (uint32_t i = 0; i < k; ++i) {
        if (!plan->apply(i)) {
            LLAMA_LOG_ERROR("%s: memplan apply(%u) failed (fail=%d)\n", __func__, i, (int) plan->get_fail());
            return fail_at(MOE_S2_STAGE_APPLY, GGML_STATUS_FAILED);
        }
    }

    // ★arm 지점 = apply **전건 성공 직후**(= plan 상태 APPLIED 확정 · llama-memplan.cpp 는 마지막
    //   ubatch 가 적용될 때만 APPLIED 로 올린다). 여기서부터 예외로 빠져나가도 plan 소멸 **전에**
    //   정리(SYNC 선행)가 돈다(r32 차단 1).
    //   ★APPLYING 구간(apply 루프 안)을 덮지 않는 이유 2가지(r33 차단 1 수리 3 — 재량 확장 롤백):
    //     ① 제출된 range 가 0이라 늦게 도착하는 커널과의 경합이 **무대상**이다 — 가드가 지키려는
    //        "clear 보다 sync 가 먼저"라는 위험 자체가 그 구간엔 없다.
    //     ② APPLYING 상태의 `abort()` 는 `snapshot_restore(origin)`(llama-memplan.cpp) — vector/set/map
    //        복사 **할당** 경로라 host 메모리 압박에서 스스로 던질 수 있다. no-throw 계약인 가드 정리
    //        사슬(`run_step` noexcept)에 넣으면 연쇄 OOM 의 2차 예외가 noexcept 경계를 넘어
    //        **std::terminate** 가 된다 — "원 예외 재전파" 계약이 그 구간에서 깨진다.
    //     APPLIED 의 `abort()` 는 `seq_fail_close_clear()`(S1 수리 2c 의 no-throw 계열 — `set_fail` 이
    //     문자열 대입을 삼키는 noexcept)라 가드 정리 사슬이 **전 구간 no-throw** 로 성립한다.
    //     APPLYING 구간의 예외는 기존대로 plan 소멸자의 auto-abort(metadata 복원)가 처리한다 —
    //     pre-APPLIED 와 같은 의미이고, 반환값 실패는 그대로 `fail_at` 경로다.
    unwind.arm();

    // --- ⑤ prefill 신호는 decode 진입당 1회 (각 range compute 는 segment 만 증가 · §5 ⓑ) ---
    const bool moe_lmtrace_on = ggml_moe_lmtrace_enabled();
    if (moe_lmtrace_on) {
        ggml_moe_lmtrace_request_prefill_enter((int32_t) n_tokens_all);
    }

    const uint64_t decode_call = moe_s2->moe_s2_decode_call++;

    const llm_graph_type gtype = ctx_type_to_graph_type(cparams.ctx_type);

    // ★그래프 빌더는 **영속 캐시 텐서**(k/v·r_l/s_l)에 닿을 통로가 필요하다. 배치 좌표는 전부
    //   동결 뷰에서 오므로, 여기 mctx 는 "plan 밖 불변 표면"에만 쓰이는 full-cache 컨텍스트다
    //   (SPEC §2 — live mctx 는 영속 cache tensor 소유 등 plan 밖 표면에만 사용).
    //   ★init_full() 은 컨텍스트 객체만 만든다(변이 없음 — llama-kv-cache.cpp / -recurrent.cpp).
    //     apply() 는 절대 부르지 않는다.
    llama_memory_context_ptr mctx_view = memory->init_full();
    if (!mctx_view || llama_memory_status_is_fail(mctx_view->get_status())) {
        LLAMA_LOG_ERROR("%s: failed to create full memory context for range graphs\n", __func__);
        return fail_at(MOE_S2_STAGE_RANGE_BUILD, GGML_STATUS_FAILED);
    }

    // --- ⑥ 층-우선 순회: for R: for ub (ub 는 0→K-1 고정 — 인과성·recurrent 전이 순서) ---
    for (int32_t r = 0; r < n_ranges; ++r) {
        const bool first = (r == 0);
        const bool last  = (r == n_ranges - 1);

        uint32_t bflags = LLM_GRAPH_BOUNDARY_NONE;
        if (!first) bflags |= LLM_GRAPH_BOUNDARY_IN;
        if (!last)  bflags |= LLM_GRAPH_BOUNDARY_OUT;

        // 2-bank 교대: range r 은 bank[(r-1)&1] 을 읽고 bank[r&1] 에 쓴다.
        ggml_tensor * bank_in_base  = first ? nullptr : moe_s2->moe_s2_bank[(r - 1) & 1].base;
        ggml_tensor * bank_out_base = last  ? nullptr : moe_s2->moe_s2_bank[ r      & 1].base;

        for (uint32_t ub = 0; ub < k; ++ub) {
            const llama_memplan_view * pv = plan->select(ub);
            if (!pv) {
                LLAMA_LOG_ERROR("%s: memplan select(%u) returned null\n", __func__, ub);
                return fail_at(MOE_S2_STAGE_RANGE_BUILD, GGML_STATUS_FAILED);
            }

            const llama_ubatch & ubatch = pv->ubatch;

            // count the outputs in this ubatch — needs to happen before the graph is built
            {
                int32_t n_outputs_new = 0;

                if (last) {
                    if (n_outputs_all == n_tokens_all) {
                        n_outputs_new = ubatch.n_tokens;
                    } else {
                        for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                            n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                        }
                    }
                }

                n_outputs = n_outputs_new;
            }

            if (moe_lmtrace_on) {
                ggml_moe_lmtrace_range_tag_set(decode_call, r, n_ranges, (int32_t) ub, (int32_t) k,
                                               r, r + 1, (int32_t) bflags);
            }

            ggml_status status = GGML_STATUS_SUCCESS;

            const auto * res = process_ubatch_range(ubatch, gtype, mctx_view.get(), pv,
                                                    r, r + 1, bflags,
                                                    bank_in_base, bank_out_base, status);

            if (moe_lmtrace_on) {
                ggml_moe_lmtrace_range_tag_clear();
            }

            if (!res) {
                // ★현행 실패 되감기(seq_rm)는 "이 ubatch 만 apply 됐다"는 ubatch-major 불변식
                //   전제라 여기서는 쓸 수 없다 — plan abort 로 대체한다(SURVEY Q6).
                const moe_s2_stage st = (status == GGML_STATUS_ALLOC_FAILED)
                                            ? MOE_S2_STAGE_RANGE_ALLOC
                                            : MOE_S2_STAGE_RANGE_COMPUTE;
                return fail_at(st, status);
            }

            // --- 출력 추출은 마지막 range 에서만 일어난다(그 앞 range 에는 logits 노드가 없다) ---
            if (last) {
                auto * t_logits = res->get_logits();

                if (logits.data && t_logits && n_outputs > 0) {
                    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
                    GGML_ASSERT(backend_res != nullptr);
                    GGML_ASSERT(logits.data != nullptr);

                    // ★admission 이 단일 seq·평범한 출력 형상만 통과시키므로 출력 행은 ubatch
                    //   순서대로 이어 붙는다. 앞 ubatch 들의 출력 행 수를 다시 세어 오프셋을 잡는다.
                    int64_t n_outputs_prev = 0;
                    if (n_outputs_all == n_tokens_all) {
                        for (uint32_t j = 0; j < ub; ++j) {
                            n_outputs_prev += plan->select(j)->ubatch.n_tokens;
                        }
                    } else {
                        for (uint32_t j = 0; j < ub; ++j) {
                            const llama_ubatch & uj = plan->select(j)->ubatch;
                            for (uint32_t i = 0; i < uj.n_tokens; ++i) {
                                n_outputs_prev += (int64_t) (uj.output[i] != 0);
                            }
                        }
                    }

                    float * logits_out = logits.data + n_outputs_prev*n_vocab;

                    GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                    GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                    ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0,
                                                  n_outputs*n_vocab*sizeof(float));
                }
            }
        }
    }

    // --- ⑦ 전 range 완주 후 commit (변이 없음 — 불변식 검증·종결) ---
    if (!plan->commit()) {
        LLAMA_LOG_ERROR("%s: memplan commit failed (fail=%d)\n", __func__, (int) plan->get_fail());
        return fail_at(MOE_S2_STAGE_COMMIT, GGML_STATUS_FAILED);
    }

    // commit 성공 = plan 종결(COMMITTED) — 소멸자가 손댈 상태가 아니고, bank 는 아래에서 정상
    // 해제된다. 정상 경로에서 정리가 도는 일은 없어야 하므로 여기서 가드를 내린다.
    unwind.dismiss();

    moe_s2_bank_free();

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings (legacy 경로와 동일 규약)
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    handled = true;

    return 0;
}

bool llama_ctx_s2_admit(llama_context & ctx, const llama_ctx_s2_state & st, const llama_batch & batch_inp,
                        uint32_t n_tokens_all, uint32_t n_ubatch, llama_ctx_s2_result & verdict) {
    llama_ctx_s2_admission adm;
    if (!st.s2_allowed) {
        verdict = LLAMA_CTX_S2_NOT_ADMITTED;   // the SHADOW instance: the NO_ENV answer, no log
        return false;
    }
    adm.code = ctx.moe_s2_admit(batch_inp, n_tokens_all, n_ubatch);
    adm.mode = adm.code == llama_context::MOE_S2_ADMIT_OK ? LLAMA_CTX_S2_MODE_LAYERMAJOR : LLAMA_CTX_S2_MODE_OFF;
    if (adm.code != llama_context::MOE_S2_ADMIT_OK && adm.code != llama_context::MOE_S2_ADMIT_NO_ENV) {
        LLAMA_LOG_DEBUG("%s: layer-major prefill not admitted: %s\n", "decode", llama_context::moe_s2_admit_name(adm.code));
    }
    if (adm.mode != LLAMA_CTX_S2_MODE_LAYERMAJOR) {
        verdict = LLAMA_CTX_S2_NOT_ADMITTED;
        return false;
    }
    return true;
}

llama_ctx_s2_result llama_ctx_s2_run(const llama_ctx_view & view, const llama_ctx_host_ops & ops, llama_context & ctx,
                                     const llama_batch & batch_inp, uint32_t n_tokens_all, int & rc) {
    bool handled = false;
    GGML_UNUSED(ctx);
    rc = ops.layermajor(ops.self, batch_inp, n_tokens_all, view.n_outputs_all, view.output_all, handled);
    if (handled) {
        return LLAMA_CTX_S2_HANDLED;
    }
    return LLAMA_CTX_S2_FALLTHROUGH;
}
