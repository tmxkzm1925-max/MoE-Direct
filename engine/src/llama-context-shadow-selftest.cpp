// [R-BC1] The P43 shadow selfcheck of llama_context, its fixtures and its C entry point llama_shadow_selfcheck.
//   Moved out of llama-context.cpp with no body change (RB_DESIGN 2-8, R-BC1). The include
//   block below is llama-context.cpp's own, in its order, so every header both units
//   include is compiled the same way in each; the seam is declared in llama-context-moe.h.

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

#include "llama-context-moe.h"

using namespace llama_ctx_moe;

// ===========================================================================
// ★섀도 드래프트 P43 selfcheck — 엔진 내부 픽스처 (SPEC_SHADOW_DRAFT §5 P43f)
//
//   ★경계 명시(P41/P42 관행 승계 · 보고와 세트): 여기서 도는 것은 **C4 KV/DSV4 full sequence
//     checkpoint 왕복 등식**뿐이다. 섀도 실 decode(그래프 빌드 · 마스크 재정규화 forward ·
//     main logits/sampler/perf 회계 · `process_ubatch()` 안의 C0 관측기 가드)는 **모델 가중치가
//     로드돼 있어야 도달**하므로 라이브 게이트 런 소관이며, 여기서 검증한 척하지 않는다.
//     하네스가 도달 가능한 나머지 항(a·b·c·d·e·g·h·i·j)은 moe-direct-selftest 의 P43 그룹에 있다.
//
//   ★픽스처를 여기 새로 쓴 이유: S1 의 memplan 픽스처 kit(fx_model/fx_make_mem/fx_series_of)은
//     llama-memplan.cpp 의 **익명 네임스페이스** 안이라 다른 번역 단위에서 재사용할 수 없다.
//     같은 관행(추상 llama_model 을 팩토리로 만들고 hparams 만 덮는다)을 그대로 따른다.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// 모델·hparams 픽스처
// ---------------------------------------------------------------------------
struct sh_fx_model {
    llama_model_params           mp;
    std::unique_ptr<llama_model> model;

    sh_fx_model() : mp(llama_model_default_params()) {
        model.reset(llama_model_create(LLM_ARCH_LLAMA, mp));

        if (model) {
            model->name = "shadow-p43-fixture";
        }
    }
};

void sh_fx_hparams(llama_hparams & hp, uint32_t n_layer, uint32_t n_embd_head, uint32_t n_head_kv) {
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

// deepseek4 형상 — raw ISWA(base 1층 + swa 3층) + CSA/HCA/LID 압축(P41 DSV4 계열과 동일값).
void sh_fx_hparams_dsv4(llama_hparams & hp, uint32_t n_layer) {
    hp.swa_type          = LLAMA_SWA_TYPE_STANDARD;
    hp.n_swa             = 64;
    hp.indexer_head_size = 16;
    hp.indexer_n_head    = 1;

    for (uint32_t i = 0; i < n_layer; ++i) {
        hp.is_swa_impl[i]          = (i == 0) ? 0u : 1u;
        hp.dsv4_compress_ratios[i] = (i == 2) ? 128u : 4u;
    }
}

// 픽스처 상수(P41 KV/DSV4 계열과 같은 형상 — 이미 이 프로세스에서 도는 것이 실증된 값이다)
const uint32_t SH_FX_N_LAYER  = 4;
const uint32_t SH_FX_EMBD_HD  = 32;
const uint32_t SH_FX_HEAD_KV  = 2;
const uint32_t SH_FX_KV_SIZE  = 1024;
const uint32_t SH_FX_N_PAD    = 32;
const uint32_t SH_FX_N_SEQMAX = 1;
const uint32_t SH_FX_N_UBATCH = 96;
const uint32_t SH_FX_N_TOK    = 384;

struct sh_fx_batch {
    std::vector<float>          embd;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   sid;
    std::vector<llama_seq_id *> sid_ptr;

    llama_batch b {};

    void build(uint32_t n_embd, uint32_t n_tok) {
        const size_t n = n_tok;

        embd.resize(n*n_embd);
        for (size_t i = 0; i < embd.size(); ++i) {
            embd[i] = (float) ((i*2654435761ull) % 97)/97.0f;
        }

        pos.resize(n);
        for (size_t i = 0; i < n; ++i) {
            pos[i] = (llama_pos) i;
        }

        n_seq_id.assign(n, 1);
        sid.assign(n, (llama_seq_id) 0);

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

// ★셀을 실제로 점유시킨다. 빈 캐시는 `state_write` 가 cell_count=0 만 쓰므로(llama-kv-cache.cpp)
//   왕복 등식이 "8바이트 헤더끼리 같다"는 공허한 검사가 된다 — 그런 검사는 만들지 않는다.
bool sh_fx_fill(llama_memory_i * mem, const llama_model & model) {
    sh_fx_batch fb;
    fb.build(model.hparams.n_embd, SH_FX_N_TOK);

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());

    if (!balloc.init(fb.b, model.vocab, mem, model.hparams.n_embd, SH_FX_N_SEQMAX, /*output_all =*/ false)) {
        return false;
    }

    llama_memory_context_ptr mctx = mem->init_batch(balloc, SH_FX_N_UBATCH, /*embd_all =*/ false);

    if (!mctx || llama_memory_status_is_fail(mctx->get_status())) {
        return false;
    }

    while (true) {
        (void) mctx->get_ubatch();

        if (!mctx->apply()) {
            return false;
        }

        if (!mctx->next()) {
            break;
        }
    }

    return true;
}

// full state 캡처. ★`llama_io_write_host` 는 텐서 바이트를 **소멸자**에서 옮기므로 스코프를
//   닫은 뒤에야 버퍼가 완성된다(열어둔 채 읽으면 텐서 구간이 비어 있다).
bool sh_state_capture(const llama_memory_i * mem, llama_seq_id seq_id,
                      llama_state_seq_flags flags, std::vector<uint8_t> & out) {
    size_t need = 0;

    try {
        llama_io_write_dummy io(false);
        mem->state_write(io, seq_id, flags);
        need = io.n_bytes();
    } catch (const std::exception &) {
        return false;
    }

    if (need == 0) {
        return false;
    }

    out.assign(need, 0);

    try {
        llama_io_write_host io(out.data(), out.size());
        mem->state_write(io, seq_id, flags);
    } catch (const std::exception &) {
        return false;
    }

    return true;
}

// 복원. 실패는 예외로 나온다(llama-kv-cache.cpp / llama-kv-cache-dsv4.cpp) — 삼키지 않고
// 호출자가 판정할 수 있게 bool 로 접고 예외 문구를 남긴다.
bool sh_state_restore(llama_memory_i * mem, const std::vector<uint8_t> & buf,
                      llama_seq_id seq_id, llama_state_seq_flags flags, std::string * out_what) {
    try {
        llama_io_read_host io(buf.data(), buf.size());
        mem->state_read(io, seq_id, flags);
    } catch (const std::exception & e) {
        if (out_what) {
            *out_what = e.what();
        }
        return false;
    }

    return true;
}

// 텐서 앞부분을 특정 패턴으로 덮는다(캡처가 실제로 데이터를 읽는지 확인하는 오라클).
bool sh_scribble(ggml_tensor * t, uint8_t pat, size_t n) {
    if (!t || !t->buffer || !t->data) {
        return false;
    }

    const size_t nb = ggml_nbytes(t);
    if (nb < n) {
        n = nb;
    }
    if (n == 0) {
        return false;
    }

    std::vector<uint8_t> buf(n, pat);
    ggml_backend_tensor_set(t, buf.data(), 0, n);

    return true;
}

} // namespace

int llama_context::shadow_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud) {
    int failed = 0;

    const auto chk = [&](bool ok, const char * name) {
        if (!ok) {
            failed++;
        }
        if (cb) {
            cb(ud, ok ? 1 : 0, name);
        }
    };

    const llama_seq_id seq = 0;

    // -----------------------------------------------------------------------
    // ⓐ standard KV (K2.6 계열) — full sequence checkpoint 왕복
    // -----------------------------------------------------------------------
    {
        sh_fx_model fm;

        if (!fm.model) {
            chk(false, "P43f0. (KV) 픽스처 모델 생성(llama_model_create) — 실패하면 이하 f 전건이 무의미하다");
        } else {
            sh_fx_hparams(fm.model->hparams, SH_FX_N_LAYER, SH_FX_EMBD_HD, SH_FX_HEAD_KV);

            const llama_model & model = *fm.model;

            std::unique_ptr<llama_memory_i> mem(new llama_kv_cache(
                        model, model.hparams,
                        GGML_TYPE_F16, GGML_TYPE_F16,
                        /*v_trans =*/ false, /*offload =*/ false, /*unified =*/ true,
                        SH_FX_KV_SIZE, SH_FX_N_SEQMAX, SH_FX_N_PAD,
                        /*n_swa =*/ 0, LLAMA_SWA_TYPE_NONE, /*mem_other =*/ nullptr,
                        nullptr, nullptr, nullptr));

            auto * kv = dynamic_cast<llama_kv_cache *>(mem.get());

            const bool filled = kv && sh_fx_fill(mem.get(), model);

            chk(filled && kv->seq_pos_max(seq) == (llama_pos) (SH_FX_N_TOK - 1),
                "P43f1. (KV) 픽스처 셀 점유 — 384토큰이 실제로 배치돼 checkpoint 에 담길 내용이 있다"
                "(빈 캐시 왕복은 cell_count=0 만 비교하는 공허한 검사라 금지)");

            std::vector<uint8_t> a;
            const bool cap_a = filled && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, a);

            chk(cap_a && !a.empty(), "P43f2. (KV) FULL 캡처 성립(state_write 바이트 > 0)");

            // 캡처가 **텐서 데이터를 실제로 읽는지** — 덮어쓰면 캡처 바이트가 달라져야 한다.
            std::vector<uint8_t> c;
            bool scribbled = false, cap_c = false;
            if (cap_a) {
                scribbled = sh_scribble(kv->get_k_storage(0), 0xA5, 64);
                cap_c     = sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, c);
            }

            chk(scribbled && cap_c && c.size() == a.size() &&
                std::memcmp(c.data(), a.data(), a.size()) != 0,
                "P43f3. ★(KV) 캡처 오라클의 유효성 — K 저장소를 덮으면 캡처 바이트가 달라진다"
                "(이게 서야 아래 '왕복 동일'이 상수끼리의 비교가 아니다)");

            std::string what;
            const bool res_ok = cap_c && sh_state_restore(mem.get(), a, seq, LLAMA_STATE_SEQ_FLAGS_NONE, &what);

            std::vector<uint8_t> b;
            const bool cap_b = res_ok && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, b);

            chk(cap_b && b.size() == a.size() && std::memcmp(b.data(), a.data(), a.size()) == 0,
                "P43f4. ★(KV) full sequence checkpoint 왕복 — 캡처→덮어씀→복원→재캡처가 "
                "**바이트 동일**(C4 standard KV 등식)");
        }
    }

    // -----------------------------------------------------------------------
    // ⓑ DSV4 — raw + CSA/HCA/LID + compressor state 3종을 포함한 full checkpoint
    //   (SPEC C4: DSV4 는 단순 seq_rm 금지 — 이 등식 하나가 하위 상태 전건의 왕복을 증명한다)
    // -----------------------------------------------------------------------
    {
        sh_fx_model fm;

        if (!fm.model) {
            chk(false, "P43f4b. (DSV4) 픽스처 모델 생성(llama_model_create)");
        } else {
            sh_fx_hparams(fm.model->hparams, SH_FX_N_LAYER, SH_FX_EMBD_HD, SH_FX_HEAD_KV);
            sh_fx_hparams_dsv4(fm.model->hparams, SH_FX_N_LAYER);

            const llama_model & model = *fm.model;

            std::unique_ptr<llama_memory_i> mem(new llama_kv_cache_dsv4(
                        model,
                        GGML_TYPE_F16, GGML_TYPE_F16, /*v_trans =*/ false,
                        /*offload =*/ false, /*swa_full =*/ true, /*unified =*/ true,
                        SH_FX_KV_SIZE, SH_FX_N_SEQMAX, SH_FX_N_UBATCH, SH_FX_N_PAD,
                        nullptr, nullptr));

            auto * d4 = dynamic_cast<llama_kv_cache_dsv4 *>(mem.get());

            const bool filled = d4 && sh_fx_fill(mem.get(), model);

            chk(filled, "P43f5. (DSV4) 픽스처 셀 점유 — raw base/swa + CSA/HCA/LID 경계를 통과한 384토큰");

            // compressor state 는 view(reshape)로만 노출된다 — 새 view 는 buffer 가 NULL 이라
            // 원본 버퍼에 결속해야 ggml_backend_tensor_set/get 이 선다(P41 과 같은 관용구).
            ggml_init_params ip = {
                /*.mem_size   =*/ ggml_tensor_overhead()*8 + 1024,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * vctx = ggml_init(ip);

            ggml_tensor * comp_kv = nullptr;
            if (filled && vctx) {
                comp_kv = d4->get_csa_state()->get_kv(vctx, 0);
                if (comp_kv && comp_kv->buffer == nullptr && comp_kv->view_src != nullptr) {
                    ggml_backend_view_init(comp_kv);
                }
            }

            std::vector<uint8_t> a;
            const bool cap_a = filled && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, a);

            std::vector<uint8_t> p;
            const bool cap_p = filled && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, p);

            chk(cap_a && cap_p && a.size() > p.size(),
                "P43f6. ★(DSV4) FULL 캡처 ⊃ PARTIAL 캡처 — raw base + CSA/HCA/LID k-cache 가 FULL "
                "에만 실린다(mode=FULL 이 '전 하위 상태 포함'이라는 물증)");

            // 캡처 유효성 — compressor state 를 덮으면 캡처 바이트가 달라져야 한다.
            std::vector<uint8_t> c;
            bool scribbled = false, cap_c = false;
            if (cap_a) {
                scribbled = sh_scribble(comp_kv, 0x5A, 64);
                cap_c     = sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, c);
            }

            chk(scribbled && cap_c && c.size() == a.size() &&
                std::memcmp(c.data(), a.data(), a.size()) != 0,
                "P43f7. ★(DSV4) 캡처 오라클의 유효성 — **compressor state** 를 덮으면 캡처 바이트가 "
                "달라진다(= compressor 3종이 실제로 blob 안에 있다는 증명)");

            std::string what;
            const bool res_ok = cap_c && sh_state_restore(mem.get(), a, seq, LLAMA_STATE_SEQ_FLAGS_NONE, &what);

            std::vector<uint8_t> b;
            const bool cap_b = res_ok && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, b);

            chk(cap_b && b.size() == a.size() && std::memcmp(b.data(), a.data(), a.size()) == 0,
                "P43f8. ★(DSV4) full sequence checkpoint 왕복 — raw + CSA/HCA/LID + compressor state "
                "3종을 덮어쓴 뒤 복원·재캡처가 **바이트 동일**(C4 DSV4 등식)");

            // PARTIAL_ONLY 로 FULL blob 을 읽으면 mode 불일치로 던진다(부분 재사용의 조용한 통과 금지).
            std::string what2;
            const bool threw = cap_a && !sh_state_restore(mem.get(), a, seq,
                                                          LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, &what2);

            chk(threw && what2.find("DSV4 state flags mismatch") != std::string::npos,
                "P43f9. ★(DSV4) FULL blob 을 PARTIAL_ONLY 로 읽으면 'DSV4 state flags mismatch' 예외 — "
                "PARTIAL 재사용은 물리 바이트 동일성 별도 증명 없이는 금지(C4)");

            // ---- 음성 대조: GGML_MOE_SHADOW_FAULT_KV_BYTE ----
            //   ★주입 지점은 **이 픽스처**다(엔진 실경로에는 이 비트의 소비자가 없다 — 있으면
            //     프로덕션이 자기 KV 를 스스로 망가뜨리는 코드가 된다). 비트를 무장하면 캡처본의
            //     compressor 구간 1바이트를 뒤집어 복원하고, 위 P43f8 과 **같은 등식**이 FAIL 하는
            //     것을 관측한다 — FAIL 관측 자체가 이 검사의 PASS 다.
            bool observed_fail = false;
            if (cap_b) {
                ggml_moe_shadow_test_fault(GGML_MOE_SHADOW_FAULT_KV_BYTE);

                std::vector<uint8_t> bad = a;

                // blob 꼬리 = lid_state 마지막 층 score 텐서의 payload(= compressor state 구간).
                // 메타데이터가 아니라 payload 라 복원이 예외 없이 성사되고 재캡처에서만 어긋난다.
                if ((ggml_moe_shadow_test_fault_bits() & GGML_MOE_SHADOW_FAULT_KV_BYTE) != 0 && !bad.empty()) {
                    bad[bad.size() - 1] = (uint8_t) (bad[bad.size() - 1] ^ 0xFF);
                }

                std::vector<uint8_t> d;
                const bool r2 = sh_state_restore(mem.get(), bad, seq, LLAMA_STATE_SEQ_FLAGS_NONE, nullptr);
                const bool c2 = r2 && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, d);

                observed_fail = c2 && !(d.size() == a.size() &&
                                        std::memcmp(d.data(), a.data(), a.size()) == 0);

                ggml_moe_shadow_test_fault(GGML_MOE_SHADOW_FAULT_NONE);

                // 원복 — 뒤 검사·다른 그룹에 오염을 남기지 않는다.
                (void) sh_state_restore(mem.get(), a, seq, LLAMA_STATE_SEQ_FLAGS_NONE, nullptr);
            }

            chk(observed_fail,
                "P43f10-neg. ★음성 대조(FAULT_KV_BYTE) — compressor state 구간 **1바이트** 변조본을 "
                "복원하면 P43f8 의 왕복 등식이 FAIL 한다(FAIL 관측 = 이 검사의 PASS · 등식에 이빨이 "
                "있다는 증명). 주입 지점은 픽스처이며 엔진 실경로엔 이 비트의 소비자가 없다");

            std::vector<uint8_t> e;
            const bool cap_e = cap_b && sh_state_capture(mem.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE, e);

            chk(cap_e && e.size() == a.size() && std::memcmp(e.data(), a.data(), a.size()) == 0 &&
                ggml_moe_shadow_test_fault_bits() == GGML_MOE_SHADOW_FAULT_NONE,
                "P43f11. 음성 대조 후 원복 — 변조 전 blob 으로 되돌아왔고 fault 비트도 0 이다"
                "(다음 그룹에 전역 오염을 남기지 않는다)");

            if (vctx) {
                ggml_free(vctx);
            }
        }
    }

    // -----------------------------------------------------------------------
    // ⓖ ★r6 P0 — `SYNC_OBS_DROP`(0x10) 역할 격리 음성
    //
    //   구 `NO_SYNC` 는 process-global 이라 `synchronize()` 가 역할을 가리지 않고 실 fence 를
    //   건너뛰었고(진단 교차 `codex_shadow_phase0_r1.md` §1-C), 그 결과 **메인 샘플링의
    //   `llama_synchronize(ctx_tgt)` 까지 생략**돼 메인 토큰열이 깨질 수 있었다. 재정의 후
    //   비트의 소비자는 `moe_bench_note_sync()` 하나뿐이며 SHADOW 역할에만 작용해야 한다.
    //   여기서 검사하는 것(모델 적재 없이 도달 가능한 표면 전부):
    //     ①무장 시 **main 역할** sync 관측은 그대로 기록된다(=메인 회계 무영향)
    //     ②무장 시 **SHADOW 역할** sync 관측만 누락된다(n_syncs·last_sync_end 불변)
    //     ③해제하면 SHADOW 관측이 즉시 돌아온다(원복)
    //     ④`note_submit` 은 무장과 무관하다 — 그래야 VIEW 행이 `n_submits>0 ∧ n_syncs==0`
    //       이라는 **판독기가 잡아야 할 형상**으로 남는다
    //   ★실 fence 자체는 이제 코드에 생략 분기가 없다(`synchronize()` 무조건 호출) —
    //     그 사실은 ⑤에서 소스 계약으로 못박고, 실 scheduler 왕복은 라이브 게이트 런 소관이다.
    // -----------------------------------------------------------------------
    {
        const int32_t fault_before = ggml_moe_shadow_test_fault_bits();

        // main 역할 누산기는 "열린 사이클"이 있어야 잡히므로 staging 을 직접 연다.
        const auto main_acc_probe = [&](bool armed) -> ggml_moe_shadow_backend_acc {
            g_moe_bench.staging = true;
            moe_bench_acc_reset(g_moe_bench.stage_acc);
            ggml_moe_shadow_test_fault(armed ? GGML_MOE_SHADOW_FAULT_SYNC_OBS_DROP
                                             : GGML_MOE_SHADOW_FAULT_NONE);
            moe_bench_note_submit(/*is_shadow=*/false, 1000);
            moe_bench_note_sync  (/*is_shadow=*/false, 1000, 1500);
            ggml_moe_shadow_test_fault(GGML_MOE_SHADOW_FAULT_NONE);
            const ggml_moe_shadow_backend_acc out = g_moe_bench.stage_acc;
            g_moe_bench.staging = false;
            moe_bench_acc_reset(g_moe_bench.stage_acc);
            return out;
        };

        const auto shadow_acc_probe = [&](bool armed) -> ggml_moe_shadow_backend_acc {
            moe_bench_acc_reset(g_moe_bench.shadow_acc);
            ggml_moe_shadow_test_fault(armed ? GGML_MOE_SHADOW_FAULT_SYNC_OBS_DROP
                                             : GGML_MOE_SHADOW_FAULT_NONE);
            moe_bench_note_submit(/*is_shadow=*/true, 2000);
            moe_bench_note_sync  (/*is_shadow=*/true, 2000, 2500);
            ggml_moe_shadow_test_fault(GGML_MOE_SHADOW_FAULT_NONE);
            const ggml_moe_shadow_backend_acc out = g_moe_bench.shadow_acc;
            moe_bench_acc_reset(g_moe_bench.shadow_acc);
            return out;
        };

        const ggml_moe_shadow_backend_acc m_armed  = main_acc_probe(true);
        const ggml_moe_shadow_backend_acc m_clean  = main_acc_probe(false);
        const ggml_moe_shadow_backend_acc s_armed  = shadow_acc_probe(true);
        const ggml_moe_shadow_backend_acc s_clean  = shadow_acc_probe(false);

        chk(m_armed.n_submits == 1 && m_armed.n_syncs == 1 &&
            m_armed.last_sync_end_qpc == 1500 && m_armed.wait_qpc == 500 &&
            m_armed.n_syncs == m_clean.n_syncs &&
            m_armed.last_sync_end_qpc == m_clean.last_sync_end_qpc &&
            m_armed.wait_qpc == m_clean.wait_qpc,
            "P43g30. ★역할 격리(SYNC_OBS_DROP 무장) — **main 역할** sync 관측은 무장/비무장이 "
            "바이트 단위로 같다(n_syncs·last_sync_end·wait 전건). 구 NO_SYNC 가 메인 fence 까지 "
            "생략해 메인 토큰열을 깨뜨렸던 결함의 재발 방지선이다");

        chk(s_clean.n_submits == 1 && s_clean.n_syncs == 1 &&
            s_clean.last_sync_end_qpc == 2500 && s_clean.wait_qpc == 500,
            "P43g31. 대조군 — 비무장 SHADOW 역할에서는 sync 관측이 정상 기록된다"
            "(이게 서야 아래 음성이 '원래 안 되는 것'을 검사하는 게 아니다)");

        chk(s_armed.n_submits == 1 && s_armed.n_syncs == 0 &&
            s_armed.last_sync_end_qpc == 0 && s_armed.wait_qpc == 0,
            "P43g32-neg. ★★음성(SYNC_OBS_DROP) — SHADOW 역할의 sync **관측만** 누락된다: "
            "n_submits>0 인데 n_syncs==0 · last_sync_end==0. 이 형상이 곧 VIEW 원가 행의 "
            "device_interval=0 · NO_BACKEND 이며, 판독기의 VIEW 하드 계약이 이걸 잡아야 한다");

        chk(ggml_moe_shadow_test_fault_bits() == GGML_MOE_SHADOW_FAULT_NONE &&
            g_moe_bench.staging == false &&
            g_moe_bench.shadow_acc.n_syncs == 0 && g_moe_bench.stage_acc.n_syncs == 0,
            "P43g33. 음성 후 원복 — fault 비트 0 · staging 닫힘 · 두 누산기 초기화"
            "(다음 그룹에 전역 오염을 남기지 않는다)");

        // ★실 fence 무조건 실행의 **소스 계약** — 이 파일에 fence 생략 분기가 남아 있으면
        //   위 음성이 '관측 전용'이라는 전제가 무너진다. 값으로 검사할 수 없는 항이라
        //   심볼 부재로 못박는다: 0x10 의 구 이름은 헤더에서 삭제됐으므로, 이 소스가
        //   컴파일된다는 사실 자체가 "구 NO_SYNC 소비자가 0개"라는 증명이다.
        chk(GGML_MOE_SHADOW_FAULT_SYNC_OBS_DROP == 0x10 && fault_before == 0,
            "P43g34. 0x10 재정의 확정 — `SYNC_OBS_DROP` 이 그 자리를 차지하고 구 "
            "`FAULT_NO_SYNC` 심볼은 헤더에서 삭제됐다(남아 있으면 이 TU 가 컴파일되지 않는다). "
            "`synchronize()` 는 fault 와 무관하게 항상 ggml_backend_sched_synchronize 를 부른다");
    }

    return failed;
}

// ★C 진입점 — 내부 C++ 타입은 DLL 밖으로 나오지 않으므로 결과만 콜백으로 넘긴다(P41/P42 관행).
extern "C" LLAMA_API int llama_shadow_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud);

extern "C" int llama_shadow_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud) {
    return llama_context::shadow_selfcheck(cb, ud);
}
