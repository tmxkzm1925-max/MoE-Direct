// [R-BC1] The S2 P42 selfcheck of llama_context and its C entry point llama_s2_selfcheck.
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
// [MoE-Direct 레버 A 단계 1 / S2] P42 selfcheck — 픽스처 가능 범위만
//
//   ★경계 명시(보고서·SPEC §4 와 세트): 여기서 검증하는 것은 **판정 로직·수명주기·계약 표**
//     까지다. 실 그래프 빌드/compute·lmtrace 회계 오라클(tokens=8194·segments=723·
//     boundary CPY=705·range_build=720/reuse=0)은 모델 로드가 필요하므로 **라이브 게이트 런**
//     소관이며, 여기서 검증한 척하지 않는다.
// ===========================================================================

int llama_context::s2_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud) {
    int failed = 0;

    const auto chk = [&](bool ok, const char * name) {
        if (!ok) {
            failed++;
        }
        if (cb) {
            cb(ud, ok ? 1 : 0, name);
        }
    };

    // -----------------------------------------------------------------------
    // ⓐ admission 판별기 — 형상별 수용/거부
    // -----------------------------------------------------------------------
    const auto base_in = []() {
        moe_s2_admit_inputs in;
        in.env_on            = true;
        in.arch_is_qwen35moe = true;
        in.is_text_decoder   = true;
        in.single_seq        = true;
        in.k_planned         = 4;
        in.cuda_path         = true;
        in.has_adapter       = false;
        in.memory_is_hybrid  = true;
        in.plain_output      = true;
        in.observers_off     = true;
        return in;
    };

    chk(moe_s2_admit_decide(base_in()) == MOE_S2_ADMIT_OK,
        "P42a1. admission — 잠금 형상(qwen35moe·decoder·hybrid·단일 seq·K=4·CUDA·무어댑터·평범 출력) 수용");

    {
        auto in = base_in(); in.env_on = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_ENV,
            "P42a2. ★옵트인 미설정이면 거부 — 기본 OFF 가 계약(다른 조건이 전부 맞아도 진입 금지)");
    }
    {
        auto in = base_in(); in.arch_is_qwen35moe = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_ARCH, "P42a3. arch != qwen35moe 거부");
    }
    {
        auto in = base_in(); in.is_text_decoder = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_DECODER, "P42a4. text decoder 아님 거부");
    }
    {
        auto in = base_in(); in.memory_is_hybrid = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_MEMORY, "P42a5. hybrid memory 아님 거부");
    }
    {
        auto in = base_in(); in.single_seq = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_SINGLE_SEQ, "P42a6. 배치에 여러 seq 혼재 거부");
    }
    {
        auto in = base_in(); in.k_planned = 1;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_K,
            "P42a7. ★K=1 거부 — 512/4 꼬리 배치는 현행 경로(정상·미세 꼬리)");
    }
    {
        auto in = base_in(); in.k_planned = 0;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_K, "P42a8. K=0(빈 배치) 거부");
    }
    {
        auto in = base_in(); in.k_planned = 2;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_OK, "P42a9. K=2 는 경계 수용(K>=2 계약)");
    }
    {
        auto in = base_in(); in.cuda_path = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_CUDA, "P42a10. CUDA 기준 경로 아님 거부");
    }
    {
        auto in = base_in(); in.has_adapter = true;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_ADAPTER, "P42a11. cvec/lora 존재 거부");
    }
    {
        auto in = base_in(); in.plain_output = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_EMBD,
            "P42a12. embeddings/nextn/layer-inp/backend sampler 형상 거부");
    }
    {
        auto in = base_in(); in.observers_off = false;
        chk(moe_s2_admit_decide(in) == MOE_S2_ADMIT_NO_OBSERVER,
            "P42a13. ★PA/PB(프리페치 포함)가 켜져 있으면 거부 — SPEC §1 'PB/prefetch 는 layer-major "
            "경로에서 OFF' 를 상호 배타 admission 으로 이행(관측 런의 의미를 조용히 바꾸지 않는다)");
    }
    {
        // 전 조건 동시 위반에서도 결과는 항상 "거부"다(OK 로 새는 조합이 없다).
        moe_s2_admit_inputs in;   // 전부 기본값(env_on=false·has_adapter=true …)
        chk(moe_s2_admit_decide(in) != MOE_S2_ADMIT_OK, "P42a14. 전 조건 미달 조합은 어떤 경우도 수용되지 않는다");
    }

    // -----------------------------------------------------------------------
    // ⓑ bank lifecycle — 할당·비중첩·동일 buft·부분 할당 정리·해제
    // -----------------------------------------------------------------------
    ggml_backend_dev_t     dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : nullptr;

    chk(buft != nullptr, "P42b0. 픽스처용 buft 확보(CPU device)");

    if (buft) {
        const int64_t  n_embd_f   = 64;
        const int64_t  n_ubatch_f = 8;
        const uint32_t k_f        = 4;
        const uint64_t want_bytes = (uint64_t) n_embd_f*n_ubatch_f*k_f*sizeof(float)*2;

        moe_s2_bank_t bank[2];

        const bool ok_alloc = moe_s2_bank_alloc_into(bank, buft, n_embd_f, n_ubatch_f, k_f, -1);
        chk(ok_alloc && bank[0].base && bank[1].base, "P42b1. 2-bank 전량 할당 성공");

        chk(ok_alloc && moe_s2_bank_bytes_of(bank) >= want_bytes,
            "P42b2. 할당 바이트 = 2 × n_embd × n_ubatch × K × 4B 이상(padding 허용)");

        chk(ok_alloc && bank[0].buf && bank[1].buf &&
            ggml_backend_buffer_get_type(bank[0].buf.get()) ==
            ggml_backend_buffer_get_type(bank[1].buf.get()),
            "P42b3. ★두 bank 가 **동일 buft** (경계 CPY 가 backend 를 넘지 않는다는 전제)");

        chk(ok_alloc && bank[0].base != bank[1].base,
            "P42b4. 두 bank 는 서로 다른 영속 텐서(교대 결선의 전제)");

        if (ok_alloc) {
            ggml_init_params ip = { ggml_tensor_overhead()*16, nullptr, true };
            ggml_context * vctx = ggml_init(ip);

            ggml_tensor * s0 = moe_s2_bank_slice_of(bank, 0, n_embd_f, n_ubatch_f, k_f, vctx, 0, n_ubatch_f);
            ggml_tensor * s1 = moe_s2_bank_slice_of(bank, 0, n_embd_f, n_ubatch_f, k_f, vctx, 1, n_ubatch_f);
            ggml_tensor * s3 = moe_s2_bank_slice_of(bank, 0, n_embd_f, n_ubatch_f, k_f, vctx, 3, 3);

            const bool have = (s0 && s1 && s3);

            // 슬라이스 비중첩 — 슬롯 stride 는 n_ubatch 행 고정.
            bool disjoint = false;
            if (have) {
                const size_t sz0 = ggml_nbytes(s0);
                const size_t o0  = s0->view_offs;
                const size_t o1  = s1->view_offs;
                disjoint = (o1 >= o0 + sz0);
            }
            chk(have && disjoint, "P42b5. ★두 slice 비중첩 — ub 슬롯 stride = n_ubatch 행 고정");

            chk(have && s0->ne[0] == n_embd_f && s0->ne[1] == n_ubatch_f && s3->ne[1] == 3,
                "P42b6. slice 형상 = [n_embd, n_tokens] (마지막 ub 의 짧은 토큰 수도 그대로)");

            chk(moe_s2_bank_slice_of(bank, 0, n_embd_f, n_ubatch_f, k_f, vctx, k_f, n_ubatch_f) == nullptr,
                "P42b7. ub_idx >= K 는 slice 거부(fail-close)");
            chk(moe_s2_bank_slice_of(bank, 0, n_embd_f, n_ubatch_f, k_f, vctx, 0, n_ubatch_f + 1) == nullptr,
                "P42b8. n_tokens > n_ubatch 는 slice 거부(슬롯 침범 방지)");
            chk(moe_s2_bank_slice_of(bank, 2, n_embd_f, n_ubatch_f, k_f, vctx, 0, n_ubatch_f) == nullptr,
                "P42b9. bank 인덱스 범위 밖 거부");

            if (vctx) {
                ggml_free(vctx);
            }
        }

        moe_s2_bank_free_of(bank);
        chk(moe_s2_bank_bytes_of(bank) == 0, "P42b10. 해제 후 live bank bytes = 0");

        // 부분 할당 정리 — #1 실패 / #2 실패 각각에서 잔재 0
        moe_s2_bank_t bank_f0[2];
        const bool r0 = moe_s2_bank_alloc_into(bank_f0, buft, n_embd_f, n_ubatch_f, k_f, /*fail_at=*/0);
        chk(!r0 && moe_s2_bank_bytes_of(bank_f0) == 0 && !bank_f0[0].base && !bank_f0[1].base,
            "P42b11. ★bank #1 할당 실패 → 잔재 0(G-L2-S2 ③)");

        moe_s2_bank_t bank_f1[2];
        const bool r1 = moe_s2_bank_alloc_into(bank_f1, buft, n_embd_f, n_ubatch_f, k_f, /*fail_at=*/1);
        chk(!r1 && moe_s2_bank_bytes_of(bank_f1) == 0 && !bank_f1[0].base && !bank_f1[1].base,
            "P42b12. ★bank #2 실패(부분 할당) → **#1 도 정리**해 잔재 0(G-L2-S2 ③)");

        moe_s2_bank_t bank_bad[2];
        chk(!moe_s2_bank_alloc_into(bank_bad, buft, 0, n_ubatch_f, k_f, -1) &&
            !moe_s2_bank_alloc_into(bank_bad, buft, n_embd_f, 0, k_f, -1) &&
            !moe_s2_bank_alloc_into(bank_bad, buft, n_embd_f, n_ubatch_f, 0, -1) &&
            !moe_s2_bank_alloc_into(bank_bad, nullptr, n_embd_f, n_ubatch_f, k_f, -1) &&
            moe_s2_bank_bytes_of(bank_bad) == 0,
            "P42b13. 비정상 인자(0 차원·K=0·buft null) 전건 거부 + 잔재 0");
    }

    // -----------------------------------------------------------------------
    // ⓒ reuse 전환 경계 key — `allow_reuse` 가 range 축을 실제로 구분하는가
    //     ★같은 range 안 ub 간 거짓 재사용 차단은 드라이버의 강제 bypass 담당이고,
    //       이 key 는 range↔legacy 전환 누출(역방향 오염)을 막는 축이다(SPEC §2).
    // -----------------------------------------------------------------------
    {
        llm_graph_params a{};
        llm_graph_params b{};

        chk(a.allow_reuse(b) && b.allow_reuse(a),
            "P42c1. legacy 기본값끼리는 재사용 허용(기존 동작 무변 — range 축이 잡음이 되지 않는다)");

        chk(a.il_start == 0 && a.il_end == -1 &&
            a.boundary_flags == LLM_GRAPH_BOUNDARY_NONE &&
            a.range_mode == LLM_GRAPH_RANGE_MODE_LEGACY &&
            a.bank_in == nullptr && a.bank_out == nullptr && a.plan_view == nullptr,
            "P42c2. ★legacy 기본값 = full-range·경계 없음·plan_view null (옵트인 OFF 행동 불변의 근거)");

        { llm_graph_params r = a; r.range_mode = LLM_GRAPH_RANGE_MODE_RANGE;
          chk(!a.allow_reuse(r) && !r.allow_reuse(a),
              "P42c3. ★legacy ↔ range 전환은 재사용 거부(컨텍스트 수명 캐시 누출 차단)"); }

        { llm_graph_params x = a; x.range_mode = LLM_GRAPH_RANGE_MODE_RANGE; x.il_start = 0; x.il_end = 1;
          llm_graph_params y = x; y.il_start = 1; y.il_end = 2;
          chk(!x.allow_reuse(y), "P42c4. 서로 다른 층 구간은 재사용 거부"); }

        { llm_graph_params x = a; x.boundary_flags = LLM_GRAPH_BOUNDARY_OUT;
          llm_graph_params y = a; y.boundary_flags = LLM_GRAPH_BOUNDARY_IN | LLM_GRAPH_BOUNDARY_OUT;
          chk(!x.allow_reuse(y), "P42c5. 경계 역할(R0 vs R_mid)이 다르면 재사용 거부"); }

        { ggml_tensor t0{}; ggml_tensor t1{};
          llm_graph_params x = a; x.bank_in = &t0;
          llm_graph_params y = a; y.bank_in = &t1;
          chk(!x.allow_reuse(y), "P42c6. ★bank slice identity 가 다르면 재사용 거부(topology 로는 구분 불가)");
          llm_graph_params z = a; z.bank_out = &t0;
          chk(!a.allow_reuse(z), "P42c7. bank 출력 결선 유무가 다르면 재사용 거부"); }
    }

    // -----------------------------------------------------------------------
    // ⓓ 실패 처분표(§3) — pre/post-APPLIED 행렬
    // -----------------------------------------------------------------------
    {
        const moe_s2_stage pre[]  = { MOE_S2_STAGE_ADMISSION, MOE_S2_STAGE_BANK,
                                      MOE_S2_STAGE_PREPARE,   MOE_S2_STAGE_PREPARED_GUARD };
        const moe_s2_stage post[] = { MOE_S2_STAGE_APPLY,       MOE_S2_STAGE_RANGE_BUILD,
                                      MOE_S2_STAGE_RANGE_ALLOC, MOE_S2_STAGE_RANGE_COMPUTE,
                                      MOE_S2_STAGE_COMMIT };
        const ggml_status sts[]   = { GGML_STATUS_SUCCESS, GGML_STATUS_FAILED,
                                      GGML_STATUS_ALLOC_FAILED, GGML_STATUS_ABORTED };

        bool pre_ok = true;
        for (auto s : pre) {
            for (auto st : sts) {
                const auto d = moe_s2_dispose(s, st);
                pre_ok &= (d.handled == false) && (d.rc == 0) && d.free_bank;
            }
        }
        chk(pre_ok, "P42d1. ★pre-APPLIED(admission·bank·prepare·PREPARED guard) = 무변이 → legacy fallback"
                    "(handled=false·rc=0·bank 정리)");

        bool post_ok = true;
        for (auto s : post) {
            for (auto st : sts) {
                const auto d = moe_s2_dispose(s, st);
                post_ok &= d.handled && (d.rc < 0) && d.abort_plan && d.free_bank;
            }
        }
        chk(post_ok, "P42d2. ★APPLIED 이후 실패 = plan abort + **fatal 음수** 반환(내부 fallback 없음)");

        bool never2 = true;
        for (auto s : { MOE_S2_STAGE_ADMISSION, MOE_S2_STAGE_BANK, MOE_S2_STAGE_PREPARE,
                        MOE_S2_STAGE_PREPARED_GUARD, MOE_S2_STAGE_APPLY, MOE_S2_STAGE_RANGE_BUILD,
                        MOE_S2_STAGE_RANGE_ALLOC, MOE_S2_STAGE_RANGE_COMPUTE, MOE_S2_STAGE_COMMIT }) {
            for (auto st : sts) {
                never2 &= (moe_s2_dispose(s, st).rc != 2);
            }
        }
        chk(never2, "P42d3. ★★ret=2 는 전 조합에서 나오지 않는다 — 서버의 배치 반감 재시도가 "
                    "clear 된 seq 를 덮치는 경로를 원천 차단(SPEC §3)");

        chk(moe_s2_dispose(MOE_S2_STAGE_RANGE_ALLOC, GGML_STATUS_ALLOC_FAILED).rc == -2 &&
            moe_s2_dispose(MOE_S2_STAGE_RANGE_COMPUTE, GGML_STATUS_ABORTED).rc == -3,
            "P42d4. ALLOC 실패는 -2, ABORTED 포함 그 외는 -3 으로 정규화");

        chk(moe_s2_dispose(MOE_S2_STAGE_PREPARE, GGML_STATUS_FAILED).abort_plan == false &&
            moe_s2_dispose(MOE_S2_STAGE_PREPARED_GUARD, GGML_STATUS_FAILED).abort_plan == true,
            "P42d5. FAILED_PREPARE 는 이미 종결이라 abort 하지 않고, PREPARED guard 는 abort 로 종결");
    }

    // -----------------------------------------------------------------------
    // ⓔ 동결 뷰 관통 — 접근자가 live 가 아니라 **뷰의 값**을 돌려주는가
    // -----------------------------------------------------------------------
    {
        llama_memplan_view v1{};
        v1.has_rs   = true;
        v1.rs.head  = 11;
        v1.rs.n     = 5;
        v1.rs.rs_z  = 2;

        llama_memplan_view v2{};
        v2.has_rs   = true;
        v2.rs.head  = 77;
        v2.rs.n     = 9;
        v2.rs.rs_z  = -1;

        // mctx 는 null 이다 — 동결 뷰가 붙어 있으면 live 를 **건드리지 않는다**는 것 자체가 증명이다
        // (live 를 읽는 구현이었다면 여기서 즉시 죽는다 = 음성 대조).
        llm_graph_input_rs inp(nullptr);

        inp.plan_view = &v1;
        const bool ok1 = (inp.get_rs_head() == 11) && (inp.get_rs_n() == 5) && (inp.get_rs_zero() == 2);

        inp.plan_view = &v2;
        const bool ok2 = (inp.get_rs_head() == 77) && (inp.get_rs_n() == 9) && (inp.get_rs_zero() == -1);

        chk(ok1 && ok2,
            "P42e1. ★recurrent 좌표(head/n/rs_z)는 동결 뷰에서만 온다 — 서로 다른 두 뷰가 서로 다른 "
            "값을 내고, live mctx(null)는 한 번도 참조되지 않는다(참조했다면 크래시)");

        llama_memplan_view vk{};
        vk.has_kv     = true;
        vk.kv.n_kv    = 1024;
        vk.kv.k_idxs.assign(4, 7);
        vk.kv.v_idxs.assign(4, 9);
        vk.kv.kq_mask.assign(8, 0.0f);
        chk(vk.kv.n_kv == 1024 && vk.kv.k_idxs.size() == 4 && vk.kv.kq_mask.size() == 8,
            "P42e2. attn 동결 뷰 필드(n_kv·k_idxs·v_idxs·kq_mask)가 값으로 존재 — build/set_input 이 "
            "이 값만 소비한다(라이브 관통 증명은 게이트 런 소관)");
    }

    // -----------------------------------------------------------------------
    // ⓕ range/boundary 역할 파생 — W=1 · N 층에서 경계 CPY 는 정확히 N-1 회
    // -----------------------------------------------------------------------
    {
        const int32_t n_ranges = 48;   // 잠금 형상(qwen35moe 실행 trunk N=48)

        int n_in = 0, n_out = 0;
        bool roles_ok = true;

        for (int32_t r = 0; r < n_ranges; ++r) {
            const bool first = (r == 0);
            const bool last  = (r == n_ranges - 1);

            uint32_t f = LLM_GRAPH_BOUNDARY_NONE;
            if (!first) f |= LLM_GRAPH_BOUNDARY_IN;
            if (!last)  f |= LLM_GRAPH_BOUNDARY_OUT;

            if (f & LLM_GRAPH_BOUNDARY_IN)  n_in++;
            if (f & LLM_GRAPH_BOUNDARY_OUT) n_out++;

            if (first) roles_ok &= (f == LLM_GRAPH_BOUNDARY_OUT);
            else if (last) roles_ok &= (f == LLM_GRAPH_BOUNDARY_IN);
            else roles_ok &= (f == (LLM_GRAPH_BOUNDARY_IN | LLM_GRAPH_BOUNDARY_OUT));

            // 2-bank 교대 — 읽는 bank 와 쓰는 bank 는 절대 같지 않다.
            if (!first && !last) {
                roles_ok &= (((r - 1) & 1) != (r & 1));
            }
        }

        chk(roles_ok, "P42f1. ★역할 파생 — R0=OUT만 · R_mid=IN|OUT · R_last=IN만, 2-bank 교대에서 "
                      "읽기/쓰기 bank 가 매 range 다르다");
        chk(n_out == n_ranges - 1 && n_in == n_ranges - 1,
            "P42f2. ★경계 CPY 횟수 = N-1 (N=48 → 47) — SPEC §0 산술과 일치");

        // 잠금 형상 회계의 **산식**만 확인한다(실측은 라이브 게이트 런 소관).
        const uint64_t n_embd_lock = 3072, tokens_2048 = 2048;
        const uint64_t move_2048   = (uint64_t)(n_ranges - 1)*tokens_2048*n_embd_lock*4ull;
        chk(move_2048 == 1182793728ull,
            "P42f3. 배치당 최대 이동 = (N-1)×2048×3072×4B = 1128MiB (SPEC §0 산술 잠금)");

        const uint64_t slice_bytes = n_embd_lock*512ull*4ull;
        chk(slice_bytes == 6ull*1024*1024 && slice_bytes*4*2 == 48ull*1024*1024,
            "P42f4. slice = 3072×512×4B = 6MiB · K=4 2-bank = 48MiB (SPEC §0 산술 잠금)");
    }

    // -----------------------------------------------------------------------
    // ⓖ 실패 정리 **순서**(r31 차단 1 수리) — device sync 가 언제나 첫 단계인가
    //     ★경계: "늦게 도착한 커널이 clear 를 덮지 않는다"는 라이브 CUDA 오라클은 게이트 런
    //       소관이다. 여기서는 실경로가 쓰는 `moe_s2_fail_cleanup` 을 그대로 돌려 **호출 순서
    //       사실**만 잠근다(순서가 뒤집힌 구현은 여기서 즉시 FAIL 한다).
    // -----------------------------------------------------------------------
    {
        const moe_s2_stage all[] = { MOE_S2_STAGE_ADMISSION,   MOE_S2_STAGE_BANK,
                                     MOE_S2_STAGE_PREPARE,     MOE_S2_STAGE_PREPARED_GUARD,
                                     MOE_S2_STAGE_APPLY,       MOE_S2_STAGE_RANGE_BUILD,
                                     MOE_S2_STAGE_RANGE_ALLOC, MOE_S2_STAGE_RANGE_COMPUTE,
                                     MOE_S2_STAGE_COMMIT };
        const ggml_status sts[]  = { GGML_STATUS_SUCCESS, GGML_STATUS_FAILED,
                                     GGML_STATUS_ALLOC_FAILED, GGML_STATUS_ABORTED };

        int steps[8] = { 0 };
        int n_steps  = 0;

        // 실경로와 **같은 함수**를 돌리되 부작용 대신 순서만 적는다.
        const auto run_case = [&](const moe_s2_disposition & d, bool have_plan, bool trace_on) {
            n_steps = 0;
            moe_s2_fail_cleanup(d, have_plan, trace_on, [&](moe_s2_fail_step s) {
                if (n_steps < 8) {
                    steps[n_steps++] = (int) s;
                }
            });
        };

        // g1: 전 조합에서 첫 단계는 언제나 sync 이며 정확히 1회다(pre/post-APPLIED 분기 없음).
        bool sync_first = true;
        for (auto s : all) {
            for (auto st : sts) {
                for (int hp = 0; hp < 2; ++hp) {
                    for (int tr = 0; tr < 2; ++tr) {
                        run_case(moe_s2_dispose(s, st), hp != 0, tr != 0);

                        int n_sync = 0;
                        for (int i = 0; i < n_steps; ++i) {
                            n_sync += (steps[i] == MOE_S2_FAIL_STEP_SYNC);
                        }
                        sync_first &= (n_steps > 0) && (steps[0] == MOE_S2_FAIL_STEP_SYNC) && (n_sync == 1);
                    }
                }
            }
        }
        chk(sync_first, "P42g1. ★실패 정리의 첫 단계는 전 조합(9단계 × 4상태 × plan 유무 × trace 유무 = 144)"
                        "에서 언제나 device sync **정확히 1회** — 조건부 sync 로 새는 경로가 없다");

        // g2: APPLIED 이후(plan 존재·trace ON) 전체 순서 잠금.
        {
            run_case(moe_s2_dispose(MOE_S2_STAGE_RANGE_COMPUTE, GGML_STATUS_FAILED), true, true);
            const bool ok = (n_steps == 4) &&
                            steps[0] == MOE_S2_FAIL_STEP_SYNC  && steps[1] == MOE_S2_FAIL_STEP_ABORT &&
                            steps[2] == MOE_S2_FAIL_STEP_FREE  && steps[3] == MOE_S2_FAIL_STEP_TRACE;
            chk(ok, "P42g2. ★★APPLIED 실패 순서 = sync → abort(clear) → bank free → lmtrace abort — "
                    "비동기 제출된 range 커널을 drain 한 **뒤에만** clear/free 가 유효하다(r31 차단 1)");
        }

        // g3: sync < abort < free 가 전 단계에서 엄격 성립.
        bool strict = true;
        for (auto s : all) {
            for (auto st : sts) {
                run_case(moe_s2_dispose(s, st), true, true);

                int i_sync = -1, i_abort = -1, i_free = -1;
                for (int i = 0; i < n_steps; ++i) {
                    if (steps[i] == MOE_S2_FAIL_STEP_SYNC)  i_sync  = i;
                    if (steps[i] == MOE_S2_FAIL_STEP_ABORT) i_abort = i;
                    if (steps[i] == MOE_S2_FAIL_STEP_FREE)  i_free  = i;
                }

                strict &= (i_sync == 0);
                if (i_abort >= 0)                 strict &= (i_sync  < i_abort);
                if (i_free  >= 0)                 strict &= (i_sync  < i_free);
                if (i_abort >= 0 && i_free >= 0)  strict &= (i_abort < i_free);
            }
        }
        chk(strict, "P42g3. ★sync < abort < free 엄격 성립 — clear 의 zero-fill 이나 buffer 해제가 "
                    "in-flight 커널보다 먼저 끝나는 배치가 어떤 단계에도 없다");

        // g4: 조건부 단계는 여전히 조건부지만 sync 는 어떤 경우에도 빠지지 않는다.
        {
            run_case(moe_s2_dispose(MOE_S2_STAGE_APPLY, GGML_STATUS_FAILED), /*have_plan=*/false, /*trace_on=*/false);
            const bool no_plan_ok = (n_steps == 2) &&
                                    steps[0] == MOE_S2_FAIL_STEP_SYNC && steps[1] == MOE_S2_FAIL_STEP_FREE;

            run_case(moe_s2_dispose(MOE_S2_STAGE_BANK, GGML_STATUS_ALLOC_FAILED), /*have_plan=*/true, /*trace_on=*/true);
            const bool pre_ok = (n_steps == 2) &&
                                steps[0] == MOE_S2_FAIL_STEP_SYNC && steps[1] == MOE_S2_FAIL_STEP_FREE;

            chk(no_plan_ok && pre_ok,
                "P42g4. plan 부재(abort 생략)·pre-APPLIED(handled=false 라 lmtrace 생략) 어느 쪽에서도 "
                "정리는 sync → bank free 2단계 — sync 는 생략 대상이 아니다");
        }
    }

    // -----------------------------------------------------------------------
    // ⓗ APPLIED 구간 **예외** unwind 정리(r32 차단 1 수리) — 반환값 실패가 아니라 C++ 예외로
    //     빠져나갈 때도 같은 정리가 SYNC 선행 순서로 돌고, 예외는 흡수되지 않는가
    //     ★경계: 실제 throw 지점(`model.build_graph()` 의 host 메모리 압박)은 모델 로드가 필요해
    //       라이브 게이트 런 소관이다. 여기서는 실경로 가드가 쓰는 **같은 타입·같은 진입점**
    //       (`moe_s2_scope_guard` / `moe_s2_unwind_cleanup`)을 그대로 인스턴스화해 unwind 사실
    //       (정리 실행·순서·예외 전파·dismiss 면제)만 잠근다.
    //     ★픽스처 전제(r33 차단 1 수리 3): 실경로 가드는 **plan 상태 APPLIED 확정 이후에만** armed
    //       이다(APPLYING 은 가드 밖 — 그 구간 abort 가 할당 경로라 no-throw 사슬에 못 넣는다).
    //       아래 인스턴스는 기본 armed=true 로 그 **armed 창 안**을 모사한다. arm/dismiss 의
    //       **배치 자체**는 픽스처가 잠글 수 없다 — 정적 검증과 G-L4-S2 게이트 런 소관이다.
    // -----------------------------------------------------------------------
    {
        int steps[8] = { 0 };
        int n_steps  = 0;

        const auto rec = [&](moe_s2_fail_step s) noexcept {
            if (n_steps < 8) {
                steps[n_steps++] = (int) s;
            }
        };

        // 예외 주입 심(S1 `maybe_throw_alloc` 전례) — 주입점을 **런타임 인자**로 받는다. 그래야
        // 같은 코드가 "예외" 분기와 "정상 종결(dismiss)" 분기를 모두 돈다.
        const auto inject = [](int point, int arm_at) {
            if (point == arm_at) {
                throw std::bad_alloc();
            }
        };

        bool unwind_order_ok = false, unwind_rethrown = false, unwind_reached_dismiss = false;
        bool normal_clean    = false, normal_ok       = false;

        for (int arm_at = 0; arm_at < 2; ++arm_at) {
            n_steps = 0;

            bool threw = false, dismissed = false;

            try {
                const auto run = [&]() noexcept {
                    moe_s2_unwind_cleanup(/*have_plan=*/true, /*trace_on=*/true, rec);
                };

                moe_s2_scope_guard<decltype(run)> g(run);

                inject(0, arm_at);   // arm_at == 0 → APPLIED 구간 예외 대역 / 1 → 예외 없음

                g.dismiss();
                dismissed = true;
            } catch (const std::bad_alloc &) {
                threw = true;
            }

            if (arm_at == 0) {
                unwind_rethrown        = threw;
                unwind_reached_dismiss = dismissed;
                unwind_order_ok        = (n_steps == 4) &&
                    steps[0] == MOE_S2_FAIL_STEP_SYNC && steps[1] == MOE_S2_FAIL_STEP_ABORT &&
                    steps[2] == MOE_S2_FAIL_STEP_FREE && steps[3] == MOE_S2_FAIL_STEP_TRACE;
            } else {
                normal_ok    = !threw && dismissed;
                normal_clean = (n_steps == 0);
            }
        }

        chk(unwind_order_ok && unwind_rethrown && !unwind_reached_dismiss,
            "P42h1. ★★APPLIED 구간 예외 unwind — 가드가 plan 소멸 **전에** sync → abort(clear) → "
            "bank free → lmtrace abort 순으로 정리하고 원 예외는 흡수 없이 그대로 전파된다"
            "(r32 차단 1: 예외 경로엔 fail_at 가 없어 sync 없는 물리 clear 로 직행하던 구멍)");

        chk(normal_clean && normal_ok,
            "P42h2. 정상 종결(dismiss)에서는 정리가 **한 단계도** 돌지 않는다 — 성공 경로의 "
            "commit → bank free 의미가 그대로다");

        // h3: plan 유무·trace 유무 어느 조합에서도 sync 는 첫 단계 1회이며 생략되지 않는다.
        //   ★`have_plan=false` 조합은 armed 창(APPLIED 확정 이후) 안의 실경로 unwind 에는 나타나지
        //     않는다 — plan 이 살아 있어야 APPLIED 일 수 있기 때문이다(r33 차단 1 수리 3). 이 축은
        //     반환값 경로(`fail_at` 의 PREPARE 실패 등 plan 부재)와 **공유하는**
        //     `moe_s2_fail_cleanup` 의 조건부 단계 계약을 함께 잠그는 용도로 유지한다.
        {
            bool ok = true;

            for (int hp = 0; hp < 2; ++hp) {
                for (int tr = 0; tr < 2; ++tr) {
                    n_steps = 0;

                    try {
                        const auto run = [&]() noexcept {
                            moe_s2_unwind_cleanup(hp != 0, tr != 0, rec);
                        };

                        moe_s2_scope_guard<decltype(run)> g(run);

                        inject(0, 0);

                        g.dismiss();
                    } catch (const std::bad_alloc &) {
                        // 전파 확인은 h1 소관 — 여기서는 정리 순서만 본다
                    }

                    int n_sync = 0;
                    for (int i = 0; i < n_steps; ++i) {
                        n_sync += (steps[i] == MOE_S2_FAIL_STEP_SYNC);
                    }

                    ok &= (n_steps > 0) && (steps[0] == MOE_S2_FAIL_STEP_SYNC) && (n_sync == 1);
                    // sync + free 는 무조건 · abort 는 plan 있을 때 · trace 는 lmtrace ON 일 때
                    ok &= (n_steps == 2 + (hp != 0 ? 1 : 0) + (tr != 0 ? 1 : 0));
                }
            }

            chk(ok, "P42h3. 예외 unwind 정리도 plan 유무 × trace 유무 4조합에서 sync 가 언제나 첫 단계 "
                    "정확히 1회 — 조건부 단계(abort/trace)만 조건부다");
        }

        // h4: 예외 처분은 §3 표의 post-APPLIED 행 재사용 — 새 의미를 발명하지 않는다.
        {
            const moe_s2_disposition du = moe_s2_unwind_dispose();
            const moe_s2_disposition dp = moe_s2_dispose(MOE_S2_STAGE_RANGE_COMPUTE, GGML_STATUS_FAILED);

            chk(du.abort_plan && du.free_bank && du.handled &&
                du.handled    == dp.handled    && du.rc == dp.rc &&
                du.abort_plan == dp.abort_plan && du.free_bank == dp.free_bank,
                "P42h4. ★APPLIED 구간 예외의 처분 = post-APPLIED 실패와 **같은 §3 표 행** — "
                "abort_plan·free_bank 참이라 태그·bank 잔존이 없다(예외는 rc 로 변환되지 않는다)");
        }
    }

    // -----------------------------------------------------------------------
    // ⓘ G-L4-S2 실패 주입 레버 — 파서(정상/이상 입력) + 1회 발화 상태기계
    //   ★검증 경계(과대 주장 금지): 여기서 잠그는 것은 **파싱 계약과 발화 상태기계**까지다.
    //     실제 (range,ub) 에서의 라이브 발화와 그 뒤의 clear/재시도 무오염은 모델 로드가
    //     필요하므로 **G-L4-S2 게이트 런**(RUNLINE_S2_GATES [6]) 소관이다.
    // -----------------------------------------------------------------------
    {
        const auto parse_ok = [](const char * s, moe_s2_inject_stage st, int32_t r, int32_t ub) {
            moe_s2_inject_t o;
            return moe_s2_inject_parse(s, o) && o.armed && o.stage == st && o.r == r && o.ub == ub;
        };

        // 이상 입력은 false 를 돌려주고 **무장 잔재가 0** 이어야 한다(부분 파싱 금지).
        const auto parse_rejects = [](const char * s) {
            moe_s2_inject_t o;
            const bool ok = moe_s2_inject_parse(s, o);
            return !ok && !o.armed && o.stage == MOE_S2_INJECT_NONE && o.r == -1 && o.ub == -1;
        };

        chk(parse_ok("build:24:2",   MOE_S2_INJECT_BUILD,   24, 2) &&
            parse_ok("alloc:24:2",   MOE_S2_INJECT_ALLOC,   24, 2) &&
            parse_ok("compute:24:2", MOE_S2_INJECT_COMPUTE, 24, 2),
            "P42i1. 정상 입력 3종(build/alloc/compute × range 24 · ub 2) 파싱 — 게이트 런 [6] 케이스");

        chk(parse_ok("build:0:0", MOE_S2_INJECT_BUILD, 0, 0) &&
            parse_ok("compute:47:15", MOE_S2_INJECT_COMPUTE, 47, 15),
            "P42i2. 경계 인덱스 수용 — 첫 range/첫 ub(0:0) 와 마지막 range(47)·K=16 상한 ub(15)");

        chk(parse_rejects(nullptr) && parse_rejects("") && parse_rejects("1"),
            "P42i3. 미설정·빈 문자열·비형식 = 무동작(기본 OFF 계약)");

        chk(parse_rejects("build:24") && parse_rejects("build") &&
            parse_rejects("build:24:2:1") && parse_rejects("build:24:2:"),
            "P42i4. 필드 수 위반(2개·1개·4개 이상) 거부");

        chk(parse_rejects("BUILD:24:2") && parse_rejects("Build:24:2") &&
            parse_rejects("builds:24:2") && parse_rejects("buil:24:2") &&
            parse_rejects("comput:24:2") && parse_rejects("computee:24:2"),
            "P42i5. ★stage 토큰은 exact 일치만 — 대소문자·접두/접미 변형 전부 거부"
            "(오설정을 조용히 흡수하면 '주입했는데 안 터졌다' 가 미스터리가 된다)");

        chk(parse_rejects("build:-1:2") && parse_rejects("build:+1:2") &&
            parse_rejects("build: 24:2") && parse_rejects("build:24: 2") &&
            parse_rejects("build:24:2 ") && parse_rejects(" build:24:2"),
            "P42i6. 부호·공백(선·후행) 거부 — 관대한 strtol 대신 엄격 10진 파서");

        chk(parse_rejects("build::2") && parse_rejects("build:24:") &&
            parse_rejects("build:2a:2") && parse_rejects("build:24:0x2") &&
            parse_rejects("build:1234567890:2"),
            "P42i7. 빈 인덱스·비숫자 혼입·자릿수 초과 거부");

        chk(parse_rejects(":24:2") && parse_rejects("::"),
            "P42i8. stage 필드가 비어도 거부");

        // --- 1회 발화 상태기계 ---
        {
            moe_s2_inject_t st;
            const bool never = !moe_s2_inject_fire(st, MOE_S2_INJECT_BUILD,   0, 0) &&
                               !moe_s2_inject_fire(st, MOE_S2_INJECT_ALLOC,   0, 0) &&
                               !moe_s2_inject_fire(st, MOE_S2_INJECT_COMPUTE, 0, 0) &&
                               !moe_s2_inject_fire(st, MOE_S2_INJECT_NONE,    0, 0);
            chk(never, "P42i9. ★무장 없음(기본값)에서는 어떤 (stage,r,ub) 조합도 발화하지 않는다 — "
                       "OFF 경로 무동작의 상태기계 근거");
        }

        {
            moe_s2_inject_t st;
            bool ok = moe_s2_inject_parse("compute:24:2", st);
            // 축이 하나라도 다르면 발화하지 않고 **무장은 유지**된다.
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_BUILD,   24, 2) && st.armed;
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_ALLOC,   24, 2) && st.armed;
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_COMPUTE, 23, 2) && st.armed;
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_COMPUTE, 25, 2) && st.armed;
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_COMPUTE, 24, 1) && st.armed;
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_COMPUTE, 24, 3) && st.armed;
            chk(ok, "P42i10. stage·range·ub 세 축 **전건 일치**에서만 발화 — 하나라도 다르면 "
                    "무발화이고 무장은 그대로 남는다(지나친 range 에서 조기 소모되지 않는다)");
        }

        {
            moe_s2_inject_t st;
            bool ok = moe_s2_inject_parse("alloc:24:2", st) && st.armed;
            ok = ok && moe_s2_inject_fire(st, MOE_S2_INJECT_ALLOC, 24, 2);   // 1회차 = 발화
            ok = ok && !st.armed;                                            // 자동 해제
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_ALLOC, 24, 2);  // 2회차 = 무발화
            ok = ok && !moe_s2_inject_fire(st, MOE_S2_INJECT_ALLOC, 24, 2);
            chk(ok, "P42i11. ★★1회 발화 후 자동 해제 — 같은 (stage,r,ub) 재도달에서 다시 터지지 "
                    "않는다. 이것이 G-L4-S2 의 '실패 후 **동일 프롬프트 재시도 정상 완주**'를 "
                    "같은 프로세스에서 얻는 전제다");
        }

        {
            moe_s2_inject_t st;
            bool ok = moe_s2_inject_parse("build:24:2", st) &&
                      moe_s2_inject_fire(st, MOE_S2_INJECT_BUILD, 24, 2);
            // 해제 후에는 **어떤 조합도** 다시 발화하지 않는다(프로세스 수명 내 재발화 없음).
            for (int s = MOE_S2_INJECT_BUILD; ok && s <= MOE_S2_INJECT_COMPUTE; ++s) {
                for (int32_t r = 0; ok && r < 48; ++r) {
                    for (int32_t u = 0; ok && u < 4; ++u) {
                        ok = !moe_s2_inject_fire(st, (moe_s2_inject_stage) s, r, u);
                    }
                }
            }
            chk(ok, "P42i12. 해제 후 3×48×4=576 조합 전건 무발화 — 재무장 경로가 존재하지 않는다");
        }

        // --- 처분표 결속: 주입이 태우는 것은 **기존 §3 행**이지 신규 의미가 아니다 ---
        {
            // build 주입 = gf null = GGML_STATUS_FAILED · alloc 주입 = ALLOC_FAILED ·
            // compute 주입 = FAILED. 드라이버(:2330)는 ALLOC_FAILED 만 RANGE_ALLOC 로,
            // 나머지는 RANGE_COMPUTE 로 분류한다 — 세 행 모두 handled·abort·free 참이다.
            const moe_s2_disposition d_build   = moe_s2_dispose(MOE_S2_STAGE_RANGE_COMPUTE, GGML_STATUS_FAILED);
            const moe_s2_disposition d_alloc   = moe_s2_dispose(MOE_S2_STAGE_RANGE_ALLOC,   GGML_STATUS_ALLOC_FAILED);
            const moe_s2_disposition d_compute = moe_s2_dispose(MOE_S2_STAGE_RANGE_COMPUTE, GGML_STATUS_FAILED);

            const bool ok = d_build.handled   && d_build.abort_plan   && d_build.free_bank   && d_build.rc   == -3 &&
                            d_alloc.handled   && d_alloc.abort_plan   && d_alloc.free_bank   && d_alloc.rc   == -2 &&
                            d_compute.handled && d_compute.abort_plan && d_compute.free_bank && d_compute.rc == -3 &&
                            d_build.rc != 2 && d_alloc.rc != 2 && d_compute.rc != 2;

            chk(ok, "P42i13. ★주입 3종의 처분 = 기존 §3 표 행 그대로(build/compute→-3 · alloc→-2 · "
                    "전건 plan abort + bank free · ret=2 없음) — 신규 실패 경로를 만들지 않았다는 "
                    "계약이 처분축에서 성립한다");
        }

        chk(std::strcmp(moe_s2_inject_stage_name(MOE_S2_INJECT_NONE),    "none")    == 0 &&
            std::strcmp(moe_s2_inject_stage_name(MOE_S2_INJECT_BUILD),   "build")   == 0 &&
            std::strcmp(moe_s2_inject_stage_name(MOE_S2_INJECT_ALLOC),   "alloc")   == 0 &&
            std::strcmp(moe_s2_inject_stage_name(MOE_S2_INJECT_COMPUTE), "compute") == 0,
            "P42i14. stage 이름 표기 = env 문법과 **같은 토큰** — 로그를 그대로 재입력할 수 있다");
    }

    return failed;
}

// ★C 진입점 — 내부 C++ 타입은 DLL 밖으로 나오지 않으므로 결과만 콜백으로 넘긴다.
extern "C" LLAMA_API int llama_s2_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud);

extern "C" int llama_s2_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud) {
    return llama_context::s2_selfcheck(cb, ud);
}

// ===========================================================================
// [T-CHAR] S2 decode characterization - the admit -> prelude -> run path of decode() driven
//   through the real llama_decode on a synthetic qwen35moe model (groups S2CHAR-DECODE and
//   S2CHAR-ATTR of moe-direct-selftest, run in a child process whose MOE_DIRECT is cleared,
//   because a synthetic model cannot load under the suite parent's MOE_DIRECT=1).
//
//   The model and the context are built the way tests/test-llama-archs.cpp builds them: GGUF
//   metadata in memory through llama_model_saver, llama_model_init_from_user with a device list,
//   llama_init_from_model with n_ubatch 64, and a batch of 128 tokens added by hand, so K = 2.
//   Every arm gets a fresh context; every environment variable and lever an arm sets is restored
//   before the next arm. The observables are public: the llama_decode rc, the logits, the
//   memory position, the log lines (a llama_log_set callback that receives every level), the
//   node order seen by the context's eval callback, and for S2CHAR-ATTR the bench state declared
//   in llama-context-moe.h. No member of llama_context is named here.
// ===========================================================================

#include "llama-model-saver.h"
#include "gguf.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace {

void s2c_env_set(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

bool s2c_env_is_set(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
}

// Saves a variable on entry and puts it back on exit (absent stays absent).
struct s2c_env_keep {
    std::string name;
    bool        had = false;
    std::string val;

    explicit s2c_env_keep(const char * n) : name(n) {
        const char * v = std::getenv(n);
        had = (v != nullptr);
        if (had) {
            val = v;
        }
    }
    ~s2c_env_keep() {
        s2c_env_set(name.c_str(), had ? val.c_str() : nullptr);
    }
};

std::string s2c_fmt(const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

// --- log capture: every level into the arm's buffer; WARN and ERROR also reach the previous logger
struct s2c_log {
    ggml_log_callback prev_cb = nullptr;
    void *            prev_ud = nullptr;
    std::string       text;
};

void s2c_log_cb(ggml_log_level level, const char * text, void * ud) {
    s2c_log * L = (s2c_log *) ud;
    if (text) {
        L->text += text;
    }
    if (L->prev_cb && (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR)) {
        L->prev_cb(level, text, L->prev_ud);
    }
}

struct s2c_log_scope {
    s2c_log & L;
    explicit s2c_log_scope(s2c_log & l) : L(l) {
        llama_log_get(&L.prev_cb, &L.prev_ud);
        L.text.clear();
        llama_log_set(s2c_log_cb, &L);
    }
    ~s2c_log_scope() {
        llama_log_set(L.prev_cb, L.prev_ud);
    }
};

std::vector<std::string> s2c_lines_with(const std::string & text, const char * needle) {
    std::vector<std::string> out;
    size_t s = 0;
    while (s < text.size()) {
        size_t e = text.find('\n', s);
        if (e == std::string::npos) {
            e = text.size();
        }
        std::string line = text.substr(s, e - s);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }
        if (line.find(needle) != std::string::npos) {
            out.push_back(line);
        }
        s = e + 1;
    }
    return out;
}

const char * const S2C_REFUSAL = "layer-major prefill not admitted: ";

// The admission refusal lines of one arm (llama-context.cpp decode(): LLAMA_LOG_DEBUG).
std::vector<std::string> s2c_refusals(const std::string & log) {
    return s2c_lines_with(log, S2C_REFUSAL);
}

std::string s2c_refusal_code(const std::string & line) {
    const size_t p = line.find(S2C_REFUSAL);
    return p == std::string::npos ? std::string() : line.substr(p + std::strlen(S2C_REFUSAL));
}

// --- node order: the eval callback sees every node of every computed graph (ask == true) and
//     never asks for data (returns false), so the compute itself is unchanged. A node named
//     "<name>-<il>" belongs to layer il. A decrease of il between consecutive layer nodes is a
//     restart: ubatch-major execution (legacy, K ubatches) restarts at layer 0 for every ubatch
//     after the first; layer-major execution (one range graph per layer, S2) never restarts.
struct s2c_order {
    int       n_layer  = 0;
    int       last     = -1;
    int       restarts = 0;
    long long seen     = 0;
};

bool s2c_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    if (!ask) {
        return true;
    }
    s2c_order * o = (s2c_order *) ud;
    if (!o || !t) {
        return false;
    }
    const char * nm = t->name;
    const size_t n  = std::strlen(nm);
    size_t i = n;
    while (i > 0 && nm[i - 1] >= '0' && nm[i - 1] <= '9') {
        --i;
    }
    if (i == n || i == 0 || nm[i - 1] != '-' || n - i > 4) {
        return false;
    }
    const int il = std::atoi(nm + i);
    if (il < 0 || il >= o->n_layer) {
        return false;
    }
    o->seen++;
    if (o->last >= 0 && il < o->last) {
        o->restarts++;
    }
    o->last = il;
    return false;
}

// --- the synthetic model, as tests/test-llama-archs.cpp builds it (get_gguf_ctx, set_tensor_data)
double s2c_nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return 1.0e30;
    }
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        mse_a_b += d*d;
        mse_a_0 += (double) a[i]*(double) a[i];
    }
    return mse_a_0 > 0.0 ? mse_a_b/mse_a_0 : 1.0e30;
}

void s2c_set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

std::vector<llama_token> s2c_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// get_gguf_ctx(arch, moe=true) of test-llama-archs for the two architectures used here
// (qwen35moe, qwen3moe); n_layer_nextn > 0 appends that many MTP blocks to the 2-layer trunk.
gguf_context * s2c_gguf(const llm_arch arch, const uint32_t n_layer_nextn) {
    gguf_context * g = gguf_init_empty();
    llama_model_saver ms(arch, g);
    const uint32_t n_ctx = 128;

    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_layer = 2 + n_layer_nextn;

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    if (n_layer_nextn > 0) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,  n_layer_nextn);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,         8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");

    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2)); // sigmoid
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return g;
}

bool s2c_silent_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

// One synthetic model on one device (the metadata lives as long as the model).
struct s2c_model {
    gguf_context * g = nullptr;
    llama_model  * m = nullptr;

    s2c_model(const llm_arch arch, const uint32_t n_layer_nextn, ggml_backend_dev_t dev, size_t seed) {
        g = s2c_gguf(arch, n_layer_nextn);
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = s2c_silent_progress;
        ggml_backend_dev_t devs[2] = { dev, nullptr };
        mp.devices    = devs;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        size_t tmp = seed;
        s2c_log L;
        s2c_log_scope scope(L);
        m = llama_model_init_from_user(g, s2c_set_tensor_data, &tmp, mp);
    }
    ~s2c_model() {
        if (m) {
            llama_model_free(m);
        }
        if (g) {
            gguf_free(g);
        }
    }
    s2c_model(const s2c_model &) = delete;
    s2c_model & operator=(const s2c_model &) = delete;
};

// get_model_and_ctx of test-llama-archs: n_ctx 0 (the model's 128), 4 threads, n_ubatch 64.
llama_context_params s2c_cparams() {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 0;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.n_ubatch        = 64;
    return cp;
}

struct s2c_batch {
    std::vector<llama_token>  tok;
    std::vector<llama_pos>    pos;
    std::vector<llama_seq_id> seq;
    std::vector<float>        embd;   // optional per-token rows (the MTP hidden-state input)
};

s2c_batch s2c_one_seq(const std::vector<llama_token> & toks, size_t n) {
    s2c_batch b;
    for (size_t i = 0; i < n && i < toks.size(); ++i) {
        b.tok.push_back(toks[i]);
        b.pos.push_back((llama_pos) i);
        b.seq.push_back(0);
    }
    return b;
}

struct s2c_run {
    bool               ctx_ok   = false;
    int                rc       = 999;
    std::vector<float> logits;          // the first n_rows outputs, n_vocab each
    std::string        log;
    int                restarts = -1;
    long long          nodes    = 0;
    llama_pos          pos_max0 = -2;
};

// One arm: a fresh context, one llama_decode of the batch, the first n_rows logits, the memory's
// last position of seq 0, the captured log and the node order. The context is freed before return.
s2c_run s2c_decode(llama_model * model, llama_context_params cp, const s2c_batch & b, int n_rows,
                   const std::function<bool(llama_context *)> & after_ctx = nullptr) {
    s2c_run   r;
    s2c_order ord;
    ord.n_layer = 64;   // the synthetic models have at most 3 blocks; any "-<il>" below 64 counts
    cp.cb_eval           = s2c_eval_cb;
    cp.cb_eval_user_data = &ord;

    s2c_log L;
    {
        s2c_log_scope scope(L);
        llama_context * ctx = llama_init_from_model(model, cp);
        if (ctx) {
            r.ctx_ok = true;
            bool pre_ok = true;
            if (after_ctx) {
                pre_ok = after_ctx(ctx);
            }
            const int n = (int) b.tok.size();
            llama_batch batch = llama_batch_init(n, 0, 1);
            if (!b.embd.empty()) {
                batch.embd = (float *) malloc(sizeof(float) * b.embd.size());
                memcpy(batch.embd, b.embd.data(), sizeof(float) * b.embd.size());
            }
            for (int i = 0; i < n; ++i) {
                batch.token[i]     = b.tok[i];
                batch.pos[i]       = b.pos[i];
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = b.seq[i];
                batch.logits[i]    = 1;
            }
            batch.n_tokens = n;
            if (pre_ok) {
                r.rc = llama_decode(ctx, batch);
                if (r.rc == 0) {
                    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                    r.logits.reserve((size_t) n_rows * n_vocab);
                    for (int i = 0; i < n_rows; ++i) {
                        const float * li = llama_get_logits_ith(ctx, i);
                        if (!li) {
                            r.logits.clear();
                            break;
                        }
                        r.logits.insert(r.logits.end(), li, li + n_vocab);
                    }
                    r.pos_max0 = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                }
            }
            llama_batch_free(batch);
            llama_free(ctx);
        }
    }
    r.log      = L.text;
    r.restarts = ord.restarts;
    r.nodes    = ord.seen;
    return r;
}

std::vector<float> s2c_rows(const std::vector<float> & v, size_t n_rows, size_t n_vocab) {
    const size_t n = n_rows * n_vocab;
    return v.size() >= n ? std::vector<float>(v.begin(), v.begin() + n) : std::vector<float>();
}

bool s2c_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return !v.empty();
}

// Snapshot of the bench fields the S2CHAR-ATTR cases compare (every field of each accumulator).
struct s2c_bench {
    ggml_moe_shadow_backend_acc stage_acc  = {};
    ggml_moe_shadow_backend_acc cycle_acc  = {};
    ggml_moe_shadow_backend_acc shadow_acc = {};
    bool     staging    = false;
    bool     cycle_open = false;
    uint64_t emitted    = 0;
    uint64_t dropped    = 0;
};

s2c_bench s2c_bench_now() {
    s2c_bench s;
    s.stage_acc  = g_moe_bench.stage_acc;
    s.cycle_acc  = g_moe_bench.cycle.acc;
    s.shadow_acc = g_moe_bench.shadow_acc;
    s.staging    = g_moe_bench.staging;
    s.cycle_open = g_moe_bench.cycle.open;
    s.emitted    = g_moe_bench.emitted;
    s.dropped    = g_moe_bench.dropped;
    return s;
}

bool s2c_acc_eq(const ggml_moe_shadow_backend_acc & a, const ggml_moe_shadow_backend_acc & b) {
    return a.wait_qpc == b.wait_qpc && a.first_submit_qpc == b.first_submit_qpc &&
           a.last_sync_end_qpc == b.last_sync_end_qpc && a.n_submits == b.n_submits && a.n_syncs == b.n_syncs;
}

bool s2c_acc_zero(const ggml_moe_shadow_backend_acc & a) {
    return a.wait_qpc == 0 && a.first_submit_qpc == 0 && a.last_sync_end_qpc == 0 && a.n_submits == 0 && a.n_syncs == 0;
}

bool s2c_bench_eq(const s2c_bench & a, const s2c_bench & b) {
    return s2c_acc_eq(a.stage_acc, b.stage_acc) && s2c_acc_eq(a.cycle_acc, b.cycle_acc) &&
           s2c_acc_eq(a.shadow_acc, b.shadow_acc) && a.staging == b.staging && a.cycle_open == b.cycle_open &&
           a.emitted == b.emitted && a.dropped == b.dropped;
}

std::string s2c_bench_str(const s2c_bench & s) {
    return s2c_fmt("staging=%d cycle.open=%d stage_acc{submits=%u syncs=%u first_submit=%s} "
                   "cycle.acc{submits=%u syncs=%u} shadow_acc{submits=%u syncs=%u} emitted=%llu dropped=%llu",
                   (int) s.staging, (int) s.cycle_open, s.stage_acc.n_submits, s.stage_acc.n_syncs,
                   s.stage_acc.first_submit_qpc ? "set" : "0", s.cycle_acc.n_submits, s.cycle_acc.n_syncs,
                   s.shadow_acc.n_submits, s.shadow_acc.n_syncs,
                   (unsigned long long) s.emitted, (unsigned long long) s.dropped);
}

// The bench fields reset field by field in P43g33's form (staging closed, each accumulator through
// moe_bench_acc_reset, the cycle fields one by one, the counters at zero), naming no bench type.
void s2c_bench_zero() {
    g_moe_bench.staging     = false;
    g_moe_bench.stage_t_qpc = 0;
    g_moe_bench.stage_cpu   = {};
    moe_bench_acc_reset(g_moe_bench.stage_acc);
    g_moe_bench.cycle.open   = false;
    g_moe_bench.cycle.req    = 0;
    g_moe_bench.cycle.ord    = 0;
    g_moe_bench.cycle.t0_qpc = 0;
    g_moe_bench.cycle.cpu0   = {};
    moe_bench_acc_reset(g_moe_bench.cycle.acc);
    moe_bench_acc_reset(g_moe_bench.shadow_acc);
    g_moe_bench.emitted = 0;
    g_moe_bench.dropped = 0;
}

// The bench and Phase A state restored as P43g33 restores it: gate variables absent and
// re-resolved, staging closed, every accumulator and counter at its initial value.
void s2c_bench_restore() {
    s2c_env_set("MOE_SHADOW_BENCH", nullptr);
    s2c_env_set("MOE_DIRECT_PHASEA", nullptr);
    ggml_moe_shadow_reset_for_test();
    ggml_moe_phasea_reset_for_test();
    s2c_bench_zero();
}

const char * const S2C_ENV_LM     = "MOE_DIRECT_PREFILL_LAYERMAJOR";
const char * const S2C_ENV_INJECT = "MOE_DIRECT_S2_FAIL_INJECT";
const char * const S2C_ENV_PHASEA = "MOE_DIRECT_PHASEA";
const char * const S2C_ENV_BENCH  = "MOE_SHADOW_BENCH";

// Scoped value of one variable for one arm (absent afterwards).
struct s2c_env_arm {
    const char * name;
    s2c_env_arm(const char * n, const char * v) : name(n) { s2c_env_set(n, v); }
    ~s2c_env_arm() { s2c_env_set(name, nullptr); }
};

struct s2c_devs {
    ggml_backend_dev_t cuda = nullptr;
    ggml_backend_dev_t cpu  = nullptr;
};

s2c_devs s2c_find_devs() {
    s2c_devs d;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * nm = ggml_backend_dev_name(dev);
        if (!d.cuda && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && nm && std::strncmp(nm, "CUDA", 4) == 0) {
            d.cuda = dev;
        }
    }
    d.cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return d;
}

const double S2C_NMSE_TOL = 1e-4;   // tests/test-llama-archs.cpp: nmse_val > 1e-4 is FAIL
const size_t S2C_SEED     = 1;
const int    S2C_N_TOK    = 128;    // test-llama-archs get_tokens(128, ...) -> K = 128/64 = 2
const int    S2C_N_VOCAB  = 128;

using s2c_chk_t = std::function<void(bool, const std::string &)>;

// ---------------------------------------------------------------------------
// S2CHAR-DECODE - admission codes, the dispose table's handled/rc mapping, and the host prelude
// re-run after FALLTHROUGH, each through one llama_decode.
// ---------------------------------------------------------------------------
void s2c_group_decode(const s2c_chk_t & chk) {
    const s2c_devs d = s2c_find_devs();
    chk(d.cuda != nullptr && d.cpu != nullptr,
        s2c_fmt("S2CHAR-D00. devices after ggml_backend_load_all(): a CUDA device and the CPU device are registered [cuda=%s cpu=%s]",
                d.cuda ? ggml_backend_dev_name(d.cuda) : "none", d.cpu ? ggml_backend_dev_name(d.cpu) : "none"));
    if (!d.cuda || !d.cpu) {
        return;
    }

    const std::vector<llama_token> toks = s2c_tokens(S2C_N_TOK, S2C_N_VOCAB, S2C_SEED);
    const s2c_batch main_b = s2c_one_seq(toks, S2C_N_TOK);

    s2c_model M(LLM_ARCH_QWEN35MOE, 0, d.cuda, S2C_SEED);
    chk(M.m != nullptr && llama_model_n_layer(M.m) == 2,
        s2c_fmt("S2CHAR-D01. synthetic qwen35moe model (test-llama-archs shape: 2 layers, n_embd 256, 2 experts) built on %s",
                ggml_backend_dev_name(d.cuda)));
    if (!M.m) {
        return;
    }

    // (1) OFF arm - no S2 variable. The reference.
    const s2c_run off = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
    const auto off_ref = s2c_refusals(off.log);
    chk(off.ctx_ok && off.rc == 0 && off.logits.size() == (size_t) S2C_N_TOK * S2C_N_VOCAB && s2c_finite(off.logits) &&
        off.pos_max0 == S2C_N_TOK - 1,
        s2c_fmt("S2CHAR-D02. OFF arm (no S2 variable): llama_decode of 128 tokens returns rc 0 with 128 finite logit rows "
                "(the reference) [rc=%d rows=%zu pos_max=%d]", off.rc, off.logits.size() / S2C_N_VOCAB, (int) off.pos_max0));
    chk(off_ref.empty(),
        s2c_fmt("S2CHAR-D03. no_env is silent: the OFF arm's log has no refusal line (decode() logs every refusal except no_env) "
                "[refusal_lines=%zu]", off_ref.size()));
    chk(off.restarts >= 1,
        s2c_fmt("S2CHAR-D04. the OFF arm runs ubatch-major: the layer order restarts at layer 0 for the second ubatch "
                "[restarts=%d layer_nodes=%lld]", off.restarts, off.nodes));

    // (2) ON arm - admitted, range path to its success return.
    s2c_run on;
    {
        s2c_env_arm e(S2C_ENV_LM, "1");
        on = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
    }
    const auto on_ref = s2c_refusals(on.log);
    const double on_nmse = s2c_nmse(off.logits, on.logits);
    chk(on.ctx_ok && on_ref.empty(),
        s2c_fmt("S2CHAR-D05. ON arm (MOE_DIRECT_PREFILL_LAYERMAJOR=1 before the context): admitted - no refusal line [refusal_lines=%zu]",
                on_ref.size()));
    chk(on.rc == 0 && on.restarts == 0 && on.nodes > 0 && on.pos_max0 == S2C_N_TOK - 1,
        s2c_fmt("S2CHAR-D06. ON arm ran layer-major and reached its success return: rc 0, no layer-order restart over %lld layer nodes, "
                "the batch committed once (seq 0 last position 127) [rc=%d restarts=%d pos_max=%d]",
                on.nodes, on.rc, on.restarts, (int) on.pos_max0));
    chk(on_nmse <= S2C_NMSE_TOL,
        s2c_fmt("S2CHAR-D07. ON arm logits equal the OFF reference within the test-llama-archs tolerance [nmse=%.3e tol=1e-4]", on_nmse));

    // (3) one arm per refused code. Each: exactly one refusal line naming that code, the decode on
    //     the legacy path returns rc 0, and its logits equal the OFF decode of the same shape.
    const auto refused = [&](const char * id_line, const char * id_legacy, const char * code, const char * how,
                             const s2c_run & r, const std::vector<float> & ref, size_t n_rows) {
        const auto lines = s2c_refusals(r.log);
        const bool one   = lines.size() == 1 && s2c_refusal_code(lines[0]) == code;
        chk(r.ctx_ok && one,
            s2c_fmt("%s. %s arm (%s): exactly one refusal line and it names %s [line=\"%s\" lines=%zu]", id_line, code, how, code,
                    lines.empty() ? "" : lines[0].c_str(), lines.size()));
        const std::vector<float> got = s2c_rows(r.logits, n_rows, S2C_N_VOCAB);
        const double nm = s2c_nmse(ref, got);
        chk(r.rc == 0 && !got.empty() && nm <= S2C_NMSE_TOL,
            s2c_fmt("%s. %s arm: the legacy path decodes the batch, rc 0, logits equal the OFF decode of the same shape "
                    "[rc=%d rows=%zu nmse=%.3e restarts=%d]", id_legacy, code, r.rc, n_rows, nm, r.restarts));
    };

    // no_arch - a second architecture test-llama-archs saves (qwen3moe, MoE)
    {
        s2c_model Q(LLM_ARCH_QWEN3MOE, 0, d.cuda, S2C_SEED);
        chk(Q.m != nullptr, "S2CHAR-D08. synthetic qwen3moe model (the no_arch shape) built");
        if (Q.m) {
            const s2c_run q_off = s2c_decode(Q.m, s2c_cparams(), main_b, S2C_N_TOK);
            s2c_run q_on;
            {
                s2c_env_arm e(S2C_ENV_LM, "1");
                q_on = s2c_decode(Q.m, s2c_cparams(), main_b, S2C_N_TOK);
            }
            chk(q_off.rc == 0 && s2c_refusals(q_off.log).empty() && q_off.logits.size() == (size_t) S2C_N_TOK * S2C_N_VOCAB,
                s2c_fmt("S2CHAR-D09. no_arch shape OFF decode (its reference): rc 0, no refusal line [rc=%d]", q_off.rc));
            refused("S2CHAR-D10", "S2CHAR-D11", "no_arch", "qwen3moe model", q_on, q_off.logits, S2C_N_TOK);
        }
    }

    // no_decoder - ctx_type MTP (a qwen35moe model with one MTP block; the batch carries the tokens
    //              and the per-token hidden-state rows the MTP head reads)
    {
        s2c_model T(LLM_ARCH_QWEN35MOE, 1, d.cuda, S2C_SEED);
        chk(T.m != nullptr && llama_model_n_layer(T.m) == 2,
            "S2CHAR-D12. synthetic qwen35moe model with one MTP block (the no_decoder shape) built");
        if (T.m) {
            s2c_batch mb = main_b;
            {
                const int n_embd = llama_model_n_embd(T.m);
                std::mt19937 gen(S2C_SEED);
                std::normal_distribution<float> dis(0.0f, 1.0f);
                mb.embd.resize((size_t) S2C_N_TOK * n_embd);
                for (auto & x : mb.embd) {
                    x = dis(gen);
                }
            }
            llama_context_params cp = s2c_cparams();
            cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            const s2c_run t_off = s2c_decode(T.m, cp, mb, S2C_N_TOK);
            s2c_run t_on;
            {
                s2c_env_arm e(S2C_ENV_LM, "1");
                t_on = s2c_decode(T.m, cp, mb, S2C_N_TOK);
            }
            chk(t_off.ctx_ok && t_off.rc == 0 && s2c_refusals(t_off.log).empty() && s2c_finite(t_off.logits) &&
                t_off.logits.size() == (size_t) S2C_N_TOK * S2C_N_VOCAB,
                s2c_fmt("S2CHAR-D13. no_decoder shape OFF decode on the MTP context (its reference): rc 0, no refusal line [ctx=%d rc=%d]",
                        (int) t_off.ctx_ok, t_off.rc));
            refused("S2CHAR-D14", "S2CHAR-D15", "no_decoder", "ctx_type MTP", t_on, t_off.logits, S2C_N_TOK);
        }
    }

    // no_single_seq - the same 128 tokens as two sequence ids (seq 0 and seq 1) in one batch
    {
        s2c_batch b2 = main_b;
        for (int i = 0; i < S2C_N_TOK; ++i) {
            b2.tok.push_back(toks[i]);
            b2.pos.push_back((llama_pos) i);
            b2.seq.push_back(1);
        }
        llama_context_params cp = s2c_cparams();
        cp.n_ctx     = 512;
        cp.n_seq_max = 2;
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            r = s2c_decode(M.m, cp, b2, S2C_N_TOK);
        }
        refused("S2CHAR-D16", "S2CHAR-D17", "no_single_seq", "seq 0 and seq 1 in one batch, seq 0 rows compared", r, off.logits, S2C_N_TOK);
    }

    // no_k - n_tokens not above n_ubatch (64 tokens, n_ubatch 64 -> K = 1)
    {
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            r = s2c_decode(M.m, s2c_cparams(), s2c_one_seq(toks, 64), 64);
        }
        refused("S2CHAR-D18", "S2CHAR-D19", "no_k", "64 tokens with n_ubatch 64, rows 0..63 compared", r,
                s2c_rows(off.logits, 64, S2C_N_VOCAB), 64);
    }

    // no_cuda - the model on the CPU device (compared with the CUDA reference, as test-llama-archs does)
    {
        s2c_model C(LLM_ARCH_QWEN35MOE, 0, d.cpu, S2C_SEED);
        chk(C.m != nullptr, s2c_fmt("S2CHAR-D20. synthetic qwen35moe model on %s (the no_cuda shape) built", ggml_backend_dev_name(d.cpu)));
        if (C.m) {
            s2c_run r;
            {
                s2c_env_arm e(S2C_ENV_LM, "1");
                r = s2c_decode(C.m, s2c_cparams(), main_b, S2C_N_TOK);
            }
            refused("S2CHAR-D21", "S2CHAR-D22", "no_cuda", "CPU device", r, off.logits, S2C_N_TOK);
        }
    }

    // no_adapter - a control vector applied with llama_set_adapter_cvec (all zeros, so the sum is unchanged)
    {
        int cvec_rc = 999;
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK, [&](llama_context * ctx) {
                const int n_embd  = llama_model_n_embd(M.m);
                const int n_layer = llama_model_n_layer(M.m);
                std::vector<float> zeros((size_t) n_embd * (n_layer - 1), 0.0f);
                cvec_rc = llama_set_adapter_cvec(ctx, zeros.data(), zeros.size(), n_embd, 1, n_layer - 1);
                return cvec_rc == 0;
            });
        }
        refused("S2CHAR-D23", "S2CHAR-D24", "no_adapter",
                s2c_fmt("llama_set_adapter_cvec zeros layers 1..n_layer-1, rc %d", cvec_rc).c_str(), r, off.logits, S2C_N_TOK);
    }

    // no_embd - embeddings
    {
        llama_context_params cp = s2c_cparams();
        cp.embeddings = true;
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            r = s2c_decode(M.m, cp, main_b, S2C_N_TOK);
        }
        refused("S2CHAR-D25", "S2CHAR-D26", "no_embd", "embeddings = true", r, off.logits, S2C_N_TOK);
    }

    // no_observer - MOE_DIRECT_PHASEA=timing plus ggml_moe_phasea_reset_for_test
    {
        s2c_run r;
        bool pa_on = false;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            s2c_env_arm p(S2C_ENV_PHASEA, "timing");
            ggml_moe_phasea_reset_for_test();
            pa_on = ggml_moe_phasea_enabled();
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
        }
        ggml_moe_phasea_reset_for_test();
        const bool pa_off_after = !ggml_moe_phasea_enabled();
        refused("S2CHAR-D27", "S2CHAR-D28", "no_observer", pa_on ? "Phase A timing on" : "Phase A NOT on", r, off.logits, S2C_N_TOK);
        chk(pa_on && pa_off_after,
            s2c_fmt("S2CHAR-D29. no_observer arm: Phase A was on during the arm and is off again after the variable is cleared "
                    "and the state reset [on=%d off_after=%d]", (int) pa_on, (int) pa_off_after));
    }

    // (4) FALLTHROUGH - prepare fails on the first plan (llama_memplan::test_set_prepare_fail_at(0)):
    //     pre-APPLIED, handled = false, the host prelude re-runs and the legacy path decodes once.
    {
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            llama_memplan::test_set_prepare_fail_at(0);
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
            llama_memplan::test_set_prepare_fail_at(-1);
        }
        const auto lines = s2c_refusals(r.log);
        const double nm  = s2c_nmse(off.logits, r.logits);
        chk(r.ctx_ok && lines.empty(),
            s2c_fmt("S2CHAR-D30. FALLTHROUGH arm (prepare fails on the first plan): admitted - no refusal line [refusal_lines=%zu]",
                    lines.size()));
        chk(r.rc == 0 && nm <= S2C_NMSE_TOL && r.pos_max0 == S2C_N_TOK - 1 && r.restarts == off.restarts,
            s2c_fmt("S2CHAR-D31. FALLTHROUGH: handled = false, the legacy path decodes the batch exactly once after the prelude "
                    "re-run - rc 0, logits equal the reference, seq 0 last position 127, and the OFF arm's ubatch-major node order "
                    "[rc=%d nmse=%.3e pos_max=%d restarts=%d off_restarts=%d]",
                    r.rc, nm, (int) r.pos_max0, r.restarts, off.restarts));
    }

    // (5)(6) RANGE_ALLOC and RANGE_COMPUTE through MOE_DIRECT_S2_FAIL_INJECT: post-APPLIED, handled,
    //        rc -2 for alloc and -3 for compute and build. The injected line at range 0 also shows
    //        prepare succeeded again, so the FALLTHROUGH lever above was cleared.
    const auto inject = [&](const char * id_lines, const char * id_rc, const char * stage, const char * dispose, int want_rc) {
        const std::string spec = std::string(stage) + ":0:0";
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            s2c_env_arm i(S2C_ENV_INJECT, spec.c_str());
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
        }
        const std::string armed_want = s2c_fmt("G-L4-S2 fail injection armed: stage=%s range=0 ubatch=0", stage);
        const std::string fired_want = s2c_fmt("G-L4-S2 injected %s failure at (range=0, ub=0)", stage);
        const auto armed = s2c_lines_with(r.log, armed_want.c_str());
        const auto fired = s2c_lines_with(r.log, fired_want.c_str());
        const auto lines = s2c_refusals(r.log);
        chk(r.ctx_ok && lines.empty() && armed.size() == 1 && fired.size() == 1,
            s2c_fmt("%s. %s arm (%s=%s): admitted, the armed line and the injected line each once [armed=\"%s\" injected=\"%s\" refusal_lines=%zu]",
                    id_lines, dispose, S2C_ENV_INJECT, spec.c_str(), armed.empty() ? "" : armed[0].c_str(),
                    fired.empty() ? "" : fired[0].c_str(), lines.size()));
        chk(r.rc == want_rc,
            s2c_fmt("%s. %s arm: post-APPLIED failure is handled and llama_decode returns rc %d [rc=%d]", id_rc, dispose, want_rc, r.rc));
    };
    inject("S2CHAR-D32", "S2CHAR-D33", "alloc",   "RANGE_ALLOC",   -2);
    inject("S2CHAR-D34", "S2CHAR-D35", "compute", "RANGE_COMPUTE", -3);
    inject("S2CHAR-D36", "S2CHAR-D37", "build",   "RANGE_COMPUTE", -3);
}

// ---------------------------------------------------------------------------
// S2CHAR-ATTR - the bench attribution of an S2 decode (keys A1-A3).
// ---------------------------------------------------------------------------
void s2c_group_attr(const s2c_chk_t & chk) {
    const s2c_devs d = s2c_find_devs();
    chk(d.cuda != nullptr, s2c_fmt("S2CHAR-A00. a CUDA device is registered [cuda=%s]", d.cuda ? ggml_backend_dev_name(d.cuda) : "none"));
    if (!d.cuda) {
        return;
    }
    const std::vector<llama_token> toks = s2c_tokens(S2C_N_TOK, S2C_N_VOCAB, S2C_SEED);
    const s2c_batch main_b = s2c_one_seq(toks, S2C_N_TOK);

    s2c_model M(LLM_ARCH_QWEN35MOE, 0, d.cuda, S2C_SEED);
    chk(M.m != nullptr, "S2CHAR-A01. synthetic qwen35moe model built");
    if (!M.m) {
        return;
    }

    // The bench gate on: MOE_SHADOW_BENCH=1 re-read after ggml_moe_shadow_reset_for_test.
    s2c_bench_restore();
    s2c_env_set(S2C_ENV_BENCH, "1");
    ggml_moe_shadow_reset_for_test();
    const bool gate_on = ggml_moe_shadow_bench_enabled();
    const bool pa_off  = !ggml_moe_phasea_enabled();
    chk(gate_on && pa_off,
        s2c_fmt("S2CHAR-A02. bench gate on (MOE_SHADOW_BENCH=1 re-read after ggml_moe_shadow_reset_for_test), Phase A off "
                "[bench=%d phasea=%d]", (int) gate_on, (int) !pa_off));

    // A1 - an admitted ON decode leaves every bench field as it found it.
    {
        s2c_bench_zero();
        const s2c_bench before = s2c_bench_now();
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
        }
        const s2c_bench after = s2c_bench_now();
        chk(r.ctx_ok && r.rc == 0 && s2c_refusals(r.log).empty() && r.restarts == 0 && r.nodes > 0,
            s2c_fmt("S2CHAR-A03. A1 arm is an admitted layer-major decode under the bench gate: rc 0, no refusal line, "
                    "no layer-order restart [rc=%d restarts=%d]", r.rc, r.restarts));
        chk(s2c_bench_eq(before, after),
            s2c_fmt("S2CHAR-A04. A1: the admitted S2 decode leaves stage_acc, cycle.acc, shadow_acc, staging, cycle.open, "
                    "emitted and dropped equal field by field before and after (no staging opens, its submits find no "
                    "accumulator) [before: %s | after: %s]", s2c_bench_str(before).c_str(), s2c_bench_str(after).c_str()));
    }

    // A3 - the OFF control on the same batch: the last graph's submits stay in stage_acc.
    s2c_bench a3_before;
    s2c_bench a3_after;
    {
        s2c_bench_zero();
        a3_before = s2c_bench_now();
        const s2c_run r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
        a3_after = s2c_bench_now();
        chk(r.ctx_ok && r.rc == 0 && s2c_refusals(r.log).empty() && r.restarts >= 1,
            s2c_fmt("S2CHAR-A05. A3 OFF control decodes the same batch ubatch-major: rc 0 [rc=%d restarts=%d]", r.rc, r.restarts));
        chk(!a3_after.staging && a3_after.stage_acc.n_submits >= 1 && a3_after.stage_acc.first_submit_qpc != 0 &&
            !a3_after.cycle_open && s2c_acc_eq(a3_before.cycle_acc, a3_after.cycle_acc) &&
            s2c_acc_eq(a3_before.shadow_acc, a3_after.shadow_acc) &&
            a3_before.emitted == a3_after.emitted && a3_before.dropped == a3_after.dropped,
            s2c_fmt("S2CHAR-A06. A3: the OFF control leaves the last graph's submits in stage_acc (staging closed by the rotate), "
                    "and cycle.acc, cycle.open, shadow_acc, emitted, dropped unchanged [before: %s | after: %s]",
                    s2c_bench_str(a3_before).c_str(), s2c_bench_str(a3_after).c_str()));
    }

    // A2 - ON with Phase A timing is refused no_observer, and its attribution equals the OFF arm's.
    {
        s2c_bench_zero();
        const s2c_bench before = s2c_bench_now();
        s2c_run r;
        bool pa_on = false;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            s2c_env_arm p(S2C_ENV_PHASEA, "timing");
            ggml_moe_phasea_reset_for_test();
            pa_on = ggml_moe_phasea_enabled();
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
        }
        ggml_moe_phasea_reset_for_test();
        const s2c_bench after = s2c_bench_now();
        const auto lines = s2c_refusals(r.log);
        chk(pa_on && r.ctx_ok && r.rc == 0 && lines.size() == 1 && s2c_refusal_code(lines[0]) == "no_observer",
            s2c_fmt("S2CHAR-A07. A2 arm (ON, MOE_DIRECT_PHASEA=timing after ggml_moe_phasea_reset_for_test) is refused no_observer "
                    "and decodes on the legacy path [phasea=%d rc=%d line=\"%s\"]", (int) pa_on, r.rc,
                    lines.empty() ? "" : lines[0].c_str()));
        const bool same_shape =
            before.staging == a3_before.staging && after.staging == a3_after.staging &&
            after.cycle_open == a3_after.cycle_open &&
            after.stage_acc.n_submits == a3_after.stage_acc.n_submits &&
            after.stage_acc.n_syncs == a3_after.stage_acc.n_syncs &&
            (after.stage_acc.first_submit_qpc != 0) == (a3_after.stage_acc.first_submit_qpc != 0) &&
            s2c_acc_eq(before.cycle_acc, after.cycle_acc) && s2c_acc_eq(before.shadow_acc, after.shadow_acc) &&
            after.emitted == a3_after.emitted && after.dropped == a3_after.dropped;
        chk(same_shape,
            s2c_fmt("S2CHAR-A08. A2: the refused arm's attribution equals the A3 OFF arm's for the same batch (staging, cycle.open, "
                    "stage_acc counts, cycle.acc and shadow_acc unchanged, emitted, dropped; clock values excepted) "
                    "[A2 after: %s | A3 after: %s]", s2c_bench_str(after).c_str(), s2c_bench_str(a3_after).c_str()));
    }

    // Restore as P43g33: gate variables absent, both states reset, staging closed, accumulators zero.
    s2c_bench_restore();
    const s2c_bench z = s2c_bench_now();
    chk(!ggml_moe_shadow_bench_enabled() && !ggml_moe_phasea_enabled() && !s2c_env_is_set(S2C_ENV_BENCH) &&
        !s2c_env_is_set(S2C_ENV_PHASEA) && ggml_moe_shadow_test_fault_bits() == GGML_MOE_SHADOW_FAULT_NONE &&
        !z.staging && !z.cycle_open && s2c_acc_zero(z.stage_acc) && s2c_acc_zero(z.cycle_acc) && s2c_acc_zero(z.shadow_acc) &&
        z.emitted == 0 && z.dropped == 0,
        "S2CHAR-A09. restored as P43g33: MOE_SHADOW_BENCH and MOE_DIRECT_PHASEA absent, bench gate and Phase A off after "
        "their resets, fault bits 0, staging closed, every accumulator and counter zero");
}


// ---------------------------------------------------------------------------
// OBSERVE-BENCH - the token-cycle counters of the bench observer (stage_begin -> rotate ->
//   cycle open, emitted, dropped) driven through real one-token llama_decode calls on the
//   synthetic qwen35moe model, with the range path not admitted: every decode asks for S2
//   (MOE_DIRECT_PREFILL_LAYERMAJOR=1) and its refusal line is asserted - no_observer for the
//   K = 2 batch of the ON-refused control, no_k for each one-token batch (the decider tests K
//   before the observers). Arms: C = ON-refused control under Phase A timing with a run begun,
//   W = the witness (one run, consecutive one-token decodes), R = a new run begun between
//   decodes, F = the same decodes with Phase A off. Every arm starts from and returns to the
//   P43g33 zero state; the readings go through g_moe_bench as S2CHAR-ATTR reads it.
// ---------------------------------------------------------------------------

// One reading of the bench fields plus the open cycle's request index and decode ordinal.
struct s2c_obs {
    s2c_bench b;
    uint32_t  req = 0;
    uint32_t  ord = 0;
};

s2c_obs s2c_obs_now() {
    s2c_obs o;
    o.b   = s2c_bench_now();
    o.req = g_moe_bench.cycle.req;
    o.ord = g_moe_bench.cycle.ord;
    return o;
}

std::string s2c_obs_str(const s2c_obs & o) {
    return s2c_fmt("stage_acc{submits=%u syncs=%u first_submit=%s} staging=%d cycle.open=%d cycle{req=%u ord=%u} "
                   "cycle.acc{submits=%u syncs=%u} shadow_acc{submits=%u syncs=%u} emitted=%llu dropped=%llu",
                   o.b.stage_acc.n_submits, o.b.stage_acc.n_syncs, o.b.stage_acc.first_submit_qpc ? "set" : "0",
                   (int) o.b.staging, (int) o.b.cycle_open, o.req, o.ord, o.b.cycle_acc.n_submits, o.b.cycle_acc.n_syncs,
                   o.b.shadow_acc.n_submits, o.b.shadow_acc.n_syncs,
                   (unsigned long long) o.b.emitted, (unsigned long long) o.b.dropped);
}

// One context held across the one-token decodes of one arm; its load and free log lines are captured.
struct s2c_hold_ctx {
    s2c_log         L;
    llama_context * ctx = nullptr;

    explicit s2c_hold_ctx(llama_model * m) {
        s2c_log_scope scope(L);
        ctx = llama_init_from_model(m, s2c_cparams());
    }
    ~s2c_hold_ctx() {
        if (ctx) {
            s2c_log_scope scope(L);
            llama_free(ctx);
        }
    }
    s2c_hold_ctx(const s2c_hold_ctx &) = delete;
    s2c_hold_ctx & operator=(const s2c_hold_ctx &) = delete;
};

struct s2c_step {
    int         rc = 999;
    std::string log;
    s2c_obs     before;
    s2c_obs     after;
};

// One llama_decode of one token at pos with one output row, then llama_synchronize, so every submit
// and sync of the graph is accounted before the bench state is read.
s2c_step s2c_decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    s2c_step st;
    st.before = s2c_obs_now();
    if (!ctx) {
        st.after = st.before;
        return st;
    }
    s2c_log L;
    {
        s2c_log_scope scope(L);
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0]     = tok;
        batch.pos[0]       = pos;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;
        st.rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_batch_free(batch);
    }
    st.after = s2c_obs_now();
    st.log   = L.text;
    return st;
}

// The one-token decode's refusal: exactly one line, code no_k, and llama_decode rc 0.
bool s2c_step_refused_no_k(const s2c_step & st, std::string & line) {
    const auto lines = s2c_refusals(st.log);
    line = lines.empty() ? std::string() : lines[0];
    return st.rc == 0 && lines.size() == 1 && s2c_refusal_code(lines[0]) == "no_k";
}

// Arms the bench gate, and Phase A timing when pa, from the P43g33 zero state: each variable is
// re-read after its reset. Returns whether the gate and Phase A read as asked.
bool s2c_ob_arm(bool pa) {
    s2c_bench_restore();
    s2c_env_set(S2C_ENV_BENCH, "1");
    ggml_moe_shadow_reset_for_test();
    if (pa) {
        s2c_env_set(S2C_ENV_PHASEA, "timing");
    }
    ggml_moe_phasea_reset_for_test();
    s2c_bench_zero();
    return ggml_moe_shadow_bench_enabled() && ggml_moe_phasea_enabled() == pa;
}

// The state an arm leaves after s2c_bench_restore(): the four variables absent, bench gate and
// Phase A off after their resets, fault bits 0, staging closed, every accumulator, the cycle's
// identity and both counters at zero.
bool s2c_ob_restored() {
    const s2c_obs z = s2c_obs_now();
    return !ggml_moe_shadow_bench_enabled() && !ggml_moe_phasea_enabled() && !s2c_env_is_set(S2C_ENV_BENCH) &&
           !s2c_env_is_set(S2C_ENV_PHASEA) && !s2c_env_is_set(S2C_ENV_LM) && !s2c_env_is_set(S2C_ENV_INJECT) &&
           ggml_moe_shadow_test_fault_bits() == GGML_MOE_SHADOW_FAULT_NONE &&
           !z.b.staging && !z.b.cycle_open && s2c_acc_zero(z.b.stage_acc) && s2c_acc_zero(z.b.cycle_acc) &&
           s2c_acc_zero(z.b.shadow_acc) && z.req == 0 && z.ord == 0 && z.b.emitted == 0 && z.b.dropped == 0;
}

const int S2C_OB_N = 4;   // one-token decodes per arm (W, R, F)

void s2c_group_obench(const s2c_chk_t & chk) {
    const s2c_devs d = s2c_find_devs();
    chk(d.cuda != nullptr, s2c_fmt("OBSERVE-BENCH-B00. a CUDA device is registered [cuda=%s]", d.cuda ? ggml_backend_dev_name(d.cuda) : "none"));
    if (!d.cuda) {
        return;
    }
    const std::vector<llama_token> toks = s2c_tokens(S2C_N_TOK, S2C_N_VOCAB, S2C_SEED);
    const s2c_batch main_b = s2c_one_seq(toks, S2C_N_TOK);

    s2c_model M(LLM_ARCH_QWEN35MOE, 0, d.cuda, S2C_SEED);
    chk(M.m != nullptr, "OBSERVE-BENCH-B01. synthetic qwen35moe model built");
    if (!M.m) {
        return;
    }
    const char * restored_txt =
        "restored as P43g33: MOE_SHADOW_BENCH, MOE_DIRECT_PHASEA, MOE_DIRECT_PREFILL_LAYERMAJOR and "
        "MOE_DIRECT_S2_FAIL_INJECT absent, bench gate and Phase A off after their resets, fault bits 0, staging closed, "
        "every accumulator, cycle{req,ord}, emitted and dropped zero";

    // C - the ON-refused control: S2 asked for a K = 2 batch under Phase A timing with a run begun.
    //     The decider refuses no_observer, the legacy path decodes two prefill graphs, the Phase A
    //     identity stays invalid for them, so no cycle opens and emitted/dropped stay.
    {
        const bool armed = s2c_ob_arm(true);
        ggml_moe_phasea_run_begin();
        const s2c_obs before = s2c_obs_now();
        chk(armed && s2c_acc_zero(before.b.stage_acc) && !before.b.cycle_open && before.b.emitted == 0 && before.b.dropped == 0,
            s2c_fmt("OBSERVE-BENCH-C01. C arm armed: bench gate on (MOE_SHADOW_BENCH=1 after ggml_moe_shadow_reset_for_test), "
                    "Phase A timing (after ggml_moe_phasea_reset_for_test), ggml_moe_phasea_run_begin called, bench zero "
                    "[armed=%d | before: %s]", (int) armed, s2c_obs_str(before).c_str()));
        s2c_run r;
        {
            s2c_env_arm e(S2C_ENV_LM, "1");
            r = s2c_decode(M.m, s2c_cparams(), main_b, S2C_N_TOK);
        }
        const s2c_obs after = s2c_obs_now();
        const auto lines = s2c_refusals(r.log);
        chk(r.ctx_ok && r.rc == 0 && lines.size() == 1 && s2c_refusal_code(lines[0]) == "no_observer" && r.restarts >= 1,
            s2c_fmt("OBSERVE-BENCH-C02. C arm (S2 asked, %d tokens, n_ubatch 64, K = 2) is refused no_observer and decodes "
                    "ubatch-major on the legacy path [rc=%d restarts=%d line=\"%s\"]", S2C_N_TOK, r.rc, r.restarts,
                    lines.empty() ? "" : lines[0].c_str()));
        chk(!after.b.staging && after.b.stage_acc.n_submits >= 1 && after.b.stage_acc.first_submit_qpc != 0 &&
            !after.b.cycle_open && s2c_acc_zero(after.b.cycle_acc) && after.req == 0 && after.ord == 0 &&
            s2c_acc_zero(after.b.shadow_acc) && after.b.emitted == before.b.emitted && after.b.dropped == before.b.dropped,
            s2c_fmt("OBSERVE-BENCH-C03. C: prefill graphs leave the Phase A identity invalid - the last graph's submits stay "
                    "in stage_acc with staging closed, no cycle opens (cycle.acc zero), emitted and dropped unchanged "
                    "[before: %s | after: %s]", s2c_obs_str(before).c_str(), s2c_obs_str(after).c_str()));
        s2c_bench_restore();
        chk(s2c_ob_restored(), std::string("OBSERVE-BENCH-C04. C arm ") + restored_txt);
    }

    // W - the witness (draft M-PA): one run, consecutive one-token decodes on one context.
    //     Per graph: stage_begin resets stage_acc (a fresh first submit, the same per-graph count)
    //     and rotate closes staging; the first graph opens the cycle (req, ord 0); every following
    //     graph is the consecutive ordinal, so emitted grows by one and dropped stays.
    {
        const bool armed = s2c_ob_arm(true);
        ggml_moe_phasea_run_begin();
        const s2c_obs z = s2c_obs_now();
        chk(armed && s2c_acc_zero(z.b.stage_acc) && s2c_acc_zero(z.b.cycle_acc) && !z.b.staging && !z.b.cycle_open &&
            z.b.emitted == 0 && z.b.dropped == 0,
            s2c_fmt("OBSERVE-BENCH-W01. W arm armed: bench gate on, Phase A timing, ggml_moe_phasea_run_begin called "
                    "(request 0), no cycle open, bench zero [armed=%d | before: %s]", (int) armed, s2c_obs_str(z).c_str()));
        s2c_env_arm e(S2C_ENV_LM, "1");
        s2c_hold_ctx H(M.m);
        uint32_t s1 = 0;
        int id = 2;
        for (int g = 1; g <= S2C_OB_N; ++g) {
            const s2c_step st = s2c_decode_one(H.ctx, toks[g - 1], (llama_pos) (g - 1));
            std::string line;
            const bool refused = H.ctx && s2c_step_refused_no_k(st, line);   // before the name is formatted
            chk(refused,
                s2c_fmt("OBSERVE-BENCH-W%02d. W graph %d: the one-token decode (pos %d) is refused no_k and returns rc 0 "
                        "on the legacy path [rc=%d line=\"%s\"]", id++, g, g - 1, st.rc, line.c_str()));
            const s2c_obs & p = st.before;
            const s2c_obs & a = st.after;
            if (g == 1) {
                s1 = a.b.stage_acc.n_submits;
            }
            const bool reset_ok = g == 1 ? a.b.stage_acc.n_submits >= 1
                                         : (a.b.stage_acc.n_submits == s1 &&
                                            a.b.stage_acc.first_submit_qpc > p.b.stage_acc.first_submit_qpc);
            const bool ok = reset_ok && a.b.stage_acc.first_submit_qpc != 0 && !a.b.staging && a.b.cycle_open &&
                            a.req == 0 && a.ord == (uint32_t) (g - 1) &&
                            a.b.cycle_acc.n_submits == a.b.stage_acc.n_submits &&
                            a.b.cycle_acc.first_submit_qpc == a.b.stage_acc.first_submit_qpc &&
                            s2c_acc_zero(a.b.shadow_acc) &&
                            a.b.emitted == p.b.emitted + (g >= 2 ? 1u : 0u) && a.b.dropped == p.b.dropped;
            chk(ok,
                s2c_fmt("OBSERVE-BENCH-W%02d. W graph %d: stage_acc reset by stage_begin (%s), staging closed by rotate, "
                        "cycle open at (req 0, ord %d) holding this graph's submits, emitted %s, dropped unchanged "
                        "[before: %s | after: %s]", id++, g,
                        g == 1 ? "this graph's submits only" : "a fresh first submit, the same per-graph count as graph 1",
                        g - 1, g >= 2 ? "+1 (consecutive ordinal)" : "unchanged (no cycle was open)",
                        s2c_obs_str(p).c_str(), s2c_obs_str(a).c_str()));
        }
        const s2c_obs fin = s2c_obs_now();
        chk(fin.b.emitted == (uint64_t) (S2C_OB_N - 1) && fin.b.dropped == 0,
            s2c_fmt("OBSERVE-BENCH-W%02d. W: %d consecutive one-token graphs of one run emit %d cycles and drop none "
                    "[emitted=%llu dropped=%llu]", id++, S2C_OB_N, S2C_OB_N - 1,
                    (unsigned long long) fin.b.emitted, (unsigned long long) fin.b.dropped));
    }
    s2c_bench_restore();
    chk(s2c_ob_restored(), std::string("OBSERVE-BENCH-W11. W arm ") + restored_txt);

    // R - a new ggml_moe_phasea_run_begin between decodes: the open cycle (request 0) and the next
    //     identity (request 1, ordinal 0) are not consecutive, so dropped moves by one and emitted not.
    {
        const bool armed = s2c_ob_arm(true);
        ggml_moe_phasea_run_begin();
        const s2c_obs z = s2c_obs_now();
        chk(armed && s2c_acc_zero(z.b.stage_acc) && s2c_acc_zero(z.b.cycle_acc) && !z.b.staging && !z.b.cycle_open &&
            z.b.emitted == 0 && z.b.dropped == 0,
            s2c_fmt("OBSERVE-BENCH-R01. R arm armed: bench gate on, Phase A timing, ggml_moe_phasea_run_begin called "
                    "(request 0), bench zero [armed=%d | before: %s]", (int) armed, s2c_obs_str(z).c_str()));
        s2c_env_arm e(S2C_ENV_LM, "1");
        s2c_hold_ctx H(M.m);
        // graph g: expected cycle identity after it, and the emitted/dropped deltas it causes
        struct { uint32_t req, ord, d_emit, d_drop; } want[S2C_OB_N] = {
            { 0, 0, 0, 0 },   // first graph of request 0 opens the cycle
            { 0, 1, 1, 0 },   // consecutive ordinal of request 0
            { 1, 0, 0, 1 },   // after the second run_begin: request 1, ordinal 0 - not consecutive
            { 1, 1, 1, 0 },   // consecutive ordinal of request 1
        };
        int id = 2;
        for (int g = 1; g <= S2C_OB_N; ++g) {
            if (g == 3) {
                ggml_moe_phasea_run_begin();
            }
            const s2c_step st = s2c_decode_one(H.ctx, toks[g - 1], (llama_pos) (g - 1));
            std::string line;
            const bool refused = H.ctx && s2c_step_refused_no_k(st, line);   // before the name is formatted
            chk(refused,
                s2c_fmt("OBSERVE-BENCH-R%02d. R graph %d%s: the one-token decode (pos %d) is refused no_k and returns rc 0 "
                        "on the legacy path [rc=%d line=\"%s\"]", id++, g, g == 3 ? " (after the second run_begin)" : "",
                        g - 1, st.rc, line.c_str()));
            const s2c_obs & p = st.before;
            const s2c_obs & a = st.after;
            const auto & w = want[g - 1];
            chk(!a.b.staging && a.b.cycle_open && a.req == w.req && a.ord == w.ord &&
                a.b.cycle_acc.n_submits == a.b.stage_acc.n_submits && a.b.stage_acc.n_submits >= 1 &&
                a.b.emitted == p.b.emitted + w.d_emit && a.b.dropped == p.b.dropped + w.d_drop,
                s2c_fmt("OBSERVE-BENCH-R%02d. R graph %d: cycle at (req %u, ord %u), emitted %+d, dropped %+d "
                        "[before: %s | after: %s]", id++, g, w.req, w.ord, (int) w.d_emit, (int) w.d_drop,
                        s2c_obs_str(p).c_str(), s2c_obs_str(a).c_str()));
        }
        const s2c_obs fin = s2c_obs_now();
        chk(fin.b.emitted == 2 && fin.b.dropped == 1,
            s2c_fmt("OBSERVE-BENCH-R%02d. R: two runs of two one-token graphs emit 2 cycles and drop 1 (the cycle left open "
                    "at the request boundary) [emitted=%llu dropped=%llu]", id++,
                    (unsigned long long) fin.b.emitted, (unsigned long long) fin.b.dropped));
    }
    s2c_bench_restore();
    chk(s2c_ob_restored(), std::string("OBSERVE-BENCH-R11. R arm ") + restored_txt);

    // F - the control: the same one-token decodes with Phase A off (no run begun, the identity is
    //     invalid): stage_begin still resets stage_acc per graph and rotate closes staging, but no
    //     cycle opens, so the last graph's submits stay in stage_acc and emitted/dropped do not move.
    {
        const bool armed = s2c_ob_arm(false);
        const s2c_obs z = s2c_obs_now();
        chk(armed && !s2c_env_is_set(S2C_ENV_PHASEA) && s2c_acc_zero(z.b.stage_acc) && !z.b.cycle_open &&
            z.b.emitted == 0 && z.b.dropped == 0,
            s2c_fmt("OBSERVE-BENCH-F01. F arm armed: bench gate on, Phase A off (MOE_DIRECT_PHASEA absent after "
                    "ggml_moe_phasea_reset_for_test), no run begun, bench zero [armed=%d | before: %s]", (int) armed,
                    s2c_obs_str(z).c_str()));
        s2c_env_arm e(S2C_ENV_LM, "1");
        s2c_hold_ctx H(M.m);
        uint32_t s1 = 0;
        int id = 2;
        for (int g = 1; g <= S2C_OB_N; ++g) {
            const s2c_step st = s2c_decode_one(H.ctx, toks[g - 1], (llama_pos) (g - 1));
            std::string line;
            const bool refused = H.ctx && s2c_step_refused_no_k(st, line);   // before the name is formatted
            chk(refused,
                s2c_fmt("OBSERVE-BENCH-F%02d. F graph %d: the one-token decode (pos %d) is refused no_k and returns rc 0 "
                        "on the legacy path [rc=%d line=\"%s\"]", id++, g, g - 1, st.rc, line.c_str()));
            const s2c_obs & p = st.before;
            const s2c_obs & a = st.after;
            if (g == 1) {
                s1 = a.b.stage_acc.n_submits;
            }
            const bool reset_ok = g == 1 ? a.b.stage_acc.n_submits >= 1
                                         : (a.b.stage_acc.n_submits == s1 &&
                                            a.b.stage_acc.first_submit_qpc > p.b.stage_acc.first_submit_qpc);
            chk(reset_ok && a.b.stage_acc.first_submit_qpc != 0 && !a.b.staging && !a.b.cycle_open &&
                s2c_acc_zero(a.b.cycle_acc) && a.req == 0 && a.ord == 0 && s2c_acc_zero(a.b.shadow_acc) &&
                a.b.emitted == p.b.emitted && a.b.dropped == p.b.dropped,
                s2c_fmt("OBSERVE-BENCH-F%02d. F graph %d: stage_acc holds this graph's submits (%s), staging closed, "
                        "no cycle opens (cycle.acc zero), emitted and dropped unchanged [before: %s | after: %s]", id++, g,
                        g == 1 ? "reset by stage_begin" : "a fresh first submit, the same per-graph count as graph 1",
                        s2c_obs_str(p).c_str(), s2c_obs_str(a).c_str()));
        }
        const s2c_obs fin = s2c_obs_now();
        chk(!fin.b.cycle_open && fin.b.emitted == 0 && fin.b.dropped == 0 && fin.b.stage_acc.n_submits == s1,
            s2c_fmt("OBSERVE-BENCH-F%02d. F: with Phase A off %d one-token graphs open no cycle, emit none and drop none, "
                    "and stage_acc ends with the last graph's submits [%s]", id++, S2C_OB_N, s2c_obs_str(fin).c_str()));
    }
    s2c_bench_restore();
    chk(s2c_ob_restored(), std::string("OBSERVE-BENCH-F11. F arm ") + restored_txt);
}

} // namespace

// The entry the moe-direct-selftest child calls (mode 1 = S2CHAR-DECODE, mode 2 = S2CHAR-ATTR,
// mode 3 = OBSERVE-BENCH). Returns the number of failed cases, or -1 for an unknown mode. The S2 variables and levers this
// code sets are cleared on entry and put back to their previous values on exit.
extern "C" LLAMA_API int llama_s2_char_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud, int mode);

extern "C" int llama_s2_char_selfcheck(void (*cb)(void * ud, int ok, const char * name), void * ud, int mode) {
    if (mode < 1 || mode > 3) {
        return -1;
    }
    int failed = 0;
    const s2c_chk_t chk = [&](bool ok, const std::string & name) {
        if (!ok) {
            failed++;
        }
        if (cb) {
            cb(ud, ok ? 1 : 0, name.c_str());
        }
    };

    s2c_env_keep k_lm(S2C_ENV_LM);
    s2c_env_keep k_inject(S2C_ENV_INJECT);
    s2c_env_keep k_phasea(S2C_ENV_PHASEA);
    s2c_env_keep k_bench(S2C_ENV_BENCH);
    s2c_env_set(S2C_ENV_LM, nullptr);
    s2c_env_set(S2C_ENV_INJECT, nullptr);
    s2c_env_set(S2C_ENV_PHASEA, nullptr);
    s2c_env_set(S2C_ENV_BENCH, nullptr);
    ggml_moe_phasea_reset_for_test();
    ggml_moe_shadow_reset_for_test();

    llama_backend_init();

    if (mode == 1) {
        s2c_group_decode(chk);
    } else if (mode == 2) {
        s2c_group_attr(chk);
    } else {
        s2c_group_obench(chk);
    }

    llama_memplan::test_set_prepare_fail_at(-1);
    chk(!s2c_env_is_set(S2C_ENV_LM) && !s2c_env_is_set(S2C_ENV_INJECT) && !s2c_env_is_set(S2C_ENV_PHASEA) &&
        !s2c_env_is_set(S2C_ENV_BENCH),
        mode == 1 ? "S2CHAR-D99. every arm restored its variables: MOE_DIRECT_PREFILL_LAYERMAJOR, MOE_DIRECT_S2_FAIL_INJECT, "
                    "MOE_DIRECT_PHASEA and MOE_SHADOW_BENCH are absent at the end of the group"
        : mode == 2 ? "S2CHAR-A99. every arm restored its variables: MOE_DIRECT_PREFILL_LAYERMAJOR, MOE_DIRECT_S2_FAIL_INJECT, "
                      "MOE_DIRECT_PHASEA and MOE_SHADOW_BENCH are absent at the end of the group"
                    : "OBSERVE-BENCH-B99. every arm restored its variables: MOE_DIRECT_PREFILL_LAYERMAJOR, MOE_DIRECT_S2_FAIL_INJECT, "
                      "MOE_DIRECT_PHASEA and MOE_SHADOW_BENCH are absent at the end of the group");
    ggml_moe_phasea_reset_for_test();
    ggml_moe_shadow_reset_for_test();
    return failed;
}
