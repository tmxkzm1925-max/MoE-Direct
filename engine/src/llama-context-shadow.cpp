// [R-BC3] llama-context-shadow.cpp - the Shadow runtime (RB_DESIGN 2-8, the R-BC3 row): the four
//   llama_context Shadow members and the extern-C wrapper, moved out of llama-context.cpp. The
//   Shadow private context lives in llama_ctx_shadow_state (PIMPL, llama-context-moe.h); the own-
//   instance synchronize goes through llama_ctx_shadow_host_ops. Internal to src/.

#include "llama-context.h"
#include "llama-context-moe.h"

#include "ggml-moe-shadow.h"
#include "llama-impl.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <exception>
#include <vector>

using namespace llama_ctx_moe;

// ★C0 — 역할 설정. 컨텍스트가 소유한 scheduler 에도 같은 역할을 심어, ggml 내부(lmtrace
//   capture 훅)에서도 graph-local 판정이 성립하게 한다. 섀도 컨텍스트 생성 직후 1회 호출.
void llama_context::set_observer_role(uint32_t role) {
    observer_role = role;
    if (sched) {
        ggml_backend_sched_set_observer_role(sched.get(), (int32_t) role);
    }
}

// ===========================================================================
// ★섀도 드래프트 드라이버 (SPEC_SHADOW_DRAFT §1 D1~D6 · §3 C1/C2/C4)
// ===========================================================================

// admission — 스코프 밖 형상은 **명시 거부**한다(조용한 통과 금지 · SPEC C4).
int llama_context::shadow_admission() const {
    if (!ggml_moe_shadow_enabled()) {
        return SHADOW_RC_GATE_OFF;
    }
    llama_memory_i * mem = memory.get();
    if (!mem) {
        return SHADOW_RC_SCOPE_HYBRID;
    }
    // hybrid(qwen 계열)·recurrent = v0.2 스코프 밖. 이월하려면 cells·used/head/n·rs_z·rs_idx·
    // s_copy 전체의 별도 계약이 필요하다(SPEC C4). 지금은 거부가 정답이다.
    //   ※ qwen3.5 계열은 llama_memory_hybrid 가 아니라 llama_memory_hybrid_iswa 를 받는다 —
    //     둘은 상속 관계가 아니므로 **양쪽 다** 검사해야 한다.
    if (dynamic_cast<llama_memory_hybrid *>(mem)      != nullptr ||
        dynamic_cast<llama_memory_hybrid_iswa *>(mem) != nullptr ||
        dynamic_cast<llama_memory_recurrent *>(mem)   != nullptr) {
        return SHADOW_RC_SCOPE_HYBRID;
    }
    return 0;
}

// 사설 컨텍스트 지연 생성 (C1) — main 과 공유하는 것은 불변 모델 가중치뿐이다.
//   memory · scheduler · batch allocator · output/logits 버퍼 · 샘플러가 전부 별개가 된다
//   (llama_context 생성자가 그 전부를 자기 것으로 만든다).
int llama_context::shadow_ensure_ctx() {
    if (moe_shadow->shadow_ctx) {
        return 0;
    }
    llama_context_params cp = llama_context_default_params();
    // ★KV 기하는 main 과 같아야 한다 — full sequence checkpoint 를 그대로 복원하기 때문이다
    //   (셀 수가 모자라면 state_seq_set_data 가 실패한다).
    cp.n_ctx           = cparams.n_ctx;
    cp.n_seq_max       = cparams.n_seq_max;
    cp.n_batch         = 1;          // D1 — 섀도는 1토큰 greedy 스텝만 돈다
    cp.n_ubatch        = 1;
    cp.n_threads       = cparams.n_threads;
    cp.n_threads_batch = cparams.n_threads_batch;
    cp.offload_kqv     = cparams.offload_kqv;
    // flash_attn 종류는 cparams 가 아니라 컨텍스트 파라미터 기본값을 그대로 쓴다
    // (llama_cparams 에는 그 필드가 없다 — 기본값 = AUTO).
    cp.embeddings      = false;
    cp.no_perf         = true;       // C1 — main 의 perf 회계와 섞이지 않게

    try {
        moe_shadow->shadow_ctx.reset(new llama_context(model, cp));
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: failed to create the private shadow context: %s\n", __func__, e.what());
        moe_shadow->shadow_ctx.reset();
        return SHADOW_RC_NO_CTX;
    }
    if (!moe_shadow->shadow_ctx) {
        return SHADOW_RC_NO_CTX;
    }
    // ★C0 — 이 컨텍스트가 만드는 모든 graph 는 SHADOW 다(scheduler 에도 같은 값이 심긴다).
    moe_shadow->shadow_ctx->set_observer_role(LLM_GRAPH_ROLE_SHADOW);
    // ★S2/memplan 비진입 assertion(SPEC C4) — 층-우선 프리필 경로는 섀도의 스코프가 아니다.
    moe_shadow->shadow_ctx->moe_s2->s2_allowed = false;
    return 0;
}

int llama_context::shadow_draft_run(llama_seq_id seq_id,
                                    llama_token  confirmed_token,
                                    uint32_t     request_index,
                                    uint32_t     origin_graph_seq,
                                    uint32_t     origin_decode_ordinal) {
    // ── ⓪ 게이트 — OFF 면 어떤 상태도 만들지 않는다(기본 완전 OFF) ──
    const int adm = shadow_admission();
    if (adm != 0) {
        return adm;
    }

    // ── ① 트리거 (D4: D=8 · N=8 단일) ──
    //   ★r1 차단 1 수리 — 컨텍스트 전역 카운터를 폐지하고 **요청별 PA decode ordinal** 로 판정한다.
    //   `(origin_decode_ordinal + 1) % every == 0` → origin = 7, 15, 23 … 이 트리거다.
    //   전역 카운터는 요청이 바뀌어도 리셋되지 않아 요청별 ordinal 계약과 어긋났다.
    const int32_t every = ggml_moe_shadow_every();
    const int32_t depth = ggml_moe_shadow_depth();
    if (every <= 0 || ((origin_decode_ordinal + 1u) % (uint32_t) every) != 0u) {
        return 0;
    }

    // ── ② 사설 컨텍스트 ──
    // ★r2 M2 — ⑥ CLONE 스팬 실 emitter(r2 교차 ⑥ "CLONE 은 enum 만 있고 emitter 가 없다").
    //   사설 컨텍스트는 **지연 생성**이라 프로세스당 1회만 실제로 만들어진다. 재사용(이미 존재)
    //   때는 스팬을 내지 않는다 — 0폭 스팬을 매 트리거마다 깔면 ⑥ 비용 분해의 CLONE 분포가
    //   0 으로 희석돼 "clone 은 공짜"라는 거짓 결론이 나오기 때문이다.
    //   aux 는 WRITER 전용 계약이라 0 을 싣고, 성공/실패는 rc 와 로그로 구분한다.
    const bool     sh_ctx_existed = (moe_shadow->shadow_ctx != nullptr);
    const uint64_t sh_t_clone0    = ggml_moe_shadow_now_qpc();
    const int      crc            = shadow_ensure_ctx();
    if (!sh_ctx_existed) {
        ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_CLONE, request_index, origin_decode_ordinal,
                                  0, sh_t_clone0, ggml_moe_shadow_now_qpc(), 0);
        LLAMA_LOG_DEBUG("%s: shadow private context clone req=%u ord=%u rc=%d\n", __func__,
                        request_index, origin_decode_ordinal, crc);
    }
    if (crc != 0) {
        return crc;
    }

    // ── ③ main 의 비동기 backend 작업을 전부 완료시킨 뒤 관측 스냅샷을 잡는다(C1) ──
    moe_shadow->host_ops.main_sync(*this);

    uint64_t d_cache_0 = 0, d_tick_0 = 0, d_rf_0 = 0, d_pend_0 = 0;
    ggml_moe_shadow_state_digest(&d_cache_0, &d_tick_0, &d_rf_0, &d_pend_0);

    // ── ④ C3 캐시 view 진입(dispatcher drain + mutation epoch 동결) ──
    // ★r1 차단 6 — ⑥ dispatcher drain 비용은 view_begin 내부에서 발생한다.
    const uint64_t sh_t_drain0 = ggml_moe_shadow_now_qpc();
    const int sh_vb = ggml_moe_shadow_view_begin();
    ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_DRAIN, request_index, origin_decode_ordinal,
                              0, sh_t_drain0, ggml_moe_shadow_now_qpc(), 0);
    if (sh_vb != 0) {
        return SHADOW_RC_NO_CACHE;
    }
    const uint64_t sh_t_view0 = ggml_moe_shadow_now_qpc();
    // ★r2 M2(SPEC §4 G-SD3 ⑦) — 섀도(SHADOW 역할) 전용 backend 누산기를 view 창 기준으로 연다.
    //   VIEW 창은 섀도가 만드는 모든 제출과 그 완료 동기(teardown)를 **전부 포함**하므로, 이
    //   창에서 계산한 device interval 은 창 밖으로 삐져나가지 않는다(구간 정합성).
    ggml_moe_shadow_cpu_times sh_cpu_view0 = {};
    ggml_moe_shadow_cpu_now(&sh_cpu_view0);
    moe_bench_acc_reset(g_moe_bench.shadow_acc);

    int rc = 0;
    int steps_done = 0;

    // ★r1 차단 3 수리 — view 를 연 **직후** 정리 가드를 무장한다.
    //   기존 코드는 정리를 직선 정상 반환부에만 두어, checkpoint/back-copy 벡터·custom op
    //   내부 vector·writer 의 insert 가 bad_alloc 을 던지면 **view 가 열린 채** 함수를
    //   빠져나갔다. 그러면 캐시 mutation 이 영구 동결되고 사이드카는 complete=1 인 채
    //   끝나 거짓 완전 기록이 된다. 이 가드가 정상·오류·예외 전 경로에서
    //   step_end → shadow sync → private seq discard → view_end 순서를 보장한다.
    struct ShadowScopeGuard {
        llama_context * self;
        llama_seq_id    seq;
        bool            released = false;
        int             vrc      = 0;
        bool            ran      = false;
        ~ShadowScopeGuard() { run(); }
        void run() {
            if (ran) return;
            ran = true;
            // ① 열린 스텝이 있으면 닫는다(kind=6 행 결손 방지).
            ggml_moe_shadow_step_end();
            // ② 섀도 backend 를 반드시 동기화한 뒤에야 상태를 폐기할 수 있다(C1).
            if (self->moe_shadow->shadow_ctx) {
                try {
                    const uint64_t t0 = ggml_moe_shadow_now_qpc();
                    self->moe_shadow->shadow_ctx->synchronize();
                    ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_DEVICE_SYNC, req, ord, 0,
                                              t0, ggml_moe_shadow_now_qpc(), 0);
                    self->moe_shadow->shadow_ctx->memory->seq_rm(seq, -1, -1);
                } catch (...) {
                    ggml_moe_shadow_run_error("shadow teardown threw");
                }
            }
            // ③ view 는 무슨 일이 있어도 닫는다(영구 동결 금지).
            vrc = ggml_moe_shadow_view_end();
            const uint64_t view_t1 = ggml_moe_shadow_now_qpc();
            ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_VIEW, req, ord, 0,
                                      view_t0, view_t1, 0);
            // ★r2 M2(G-SD3 ⑦) — 섀도 쪽 CPU/backend 원가의 **완결 행**. ② 의 device sync 까지
            //   끝난 시점이라 SHADOW 누산기의 device interval 이 여기서 비로소 닫힌다
            //   (TRIGGER 행은 j-루프 창만 보므로 그 시점엔 sync 가 없어 NO_BACKEND 다 —
            //    미관측을 0 으로 위장하지 않기 위해 두 행을 따로 남긴다).
            ggml_moe_shadow_cpu_times cpu1 = {};
            ggml_moe_shadow_cpu_now(&cpu1);
            ggml_moe_shadow_cost_note(GGML_MOE_SHADOW_SPAN_VIEW, req, ord, 0, view_t0, view_t1,
                                      &cpu_view0, &cpu1, &g_moe_bench.shadow_acc);
            if (!released) {
                ggml_moe_shadow_run_error("shadow run left the scope abnormally");
            }
        }
        uint32_t req = 0, ord = 0;
        uint64_t view_t0 = 0;
        ggml_moe_shadow_cpu_times cpu_view0 = {};
    } guard{ this, seq_id };
    guard.req       = request_index;
    guard.ord       = origin_decode_ordinal;
    guard.view_t0   = sh_t_view0;
    guard.cpu_view0 = sh_cpu_view0;

    // ★r1 차단 3 수리 — 이 구간의 할당(checkpoint blob·back-copy·custom op 내부 vector·
    //   writer insert)은 OOM 시 예외로 함수를 탈출하는 대신 **명시 rc + incomplete latch**
    //   로 바꾼다. 정리 자체는 위 가드가 책임진다(예외가 나도 view 는 닫힌다).
    try {
    // ── ⑤ KV full sequence checkpoint (C4) ──
    //     DSV4 는 단순 seq_rm 이 금지다 — raw + CSA/HCA/LID + compressor state 3종이 전부
    //     들어가는 full checkpoint 를 뜨고, 복원본을 되읽어 **바이트 동일**임을 확인한다.
    //     이 등식 하나가 8개 하위 상태 전부의 왕복을 증명한다(부분 증명 금지).
    std::vector<uint8_t> ckpt;
    const uint64_t sh_t_ckpt0 = ggml_moe_shadow_now_qpc();
    {
        const size_t need = state_seq_get_size(seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (need == 0) {
            rc = SHADOW_RC_CHECKPOINT;
        } else {
            ckpt.resize(need);
            const size_t got = state_seq_get_data(seq_id, ckpt.data(), ckpt.size(),
                                                  LLAMA_STATE_SEQ_FLAGS_NONE);
            if (got == 0) {
                rc = SHADOW_RC_CHECKPOINT;
            } else {
                ckpt.resize(got);
            }
        }
    }

    // ★r2 M2 — aux 는 WRITER 전용이라는 소비자 하드 계약에 맞춰 0 을 싣는다(비-WRITER 에 값을
    //   실으면 판독기가 그 런 전체를 fail-close 한다). checkpoint blob 크기는 게이트 지표가
    //   아니라 진단값이므로 로그로 남긴다 — wire 에서 조용히 사라지지 않게.
    LLAMA_LOG_DEBUG("%s: shadow checkpoint blob req=%u ord=%u bytes=%zu\n", __func__,
                    request_index, origin_decode_ordinal, ckpt.size());
    ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_CHECKPOINT, request_index, origin_decode_ordinal,
                              0, sh_t_ckpt0, ggml_moe_shadow_now_qpc(), 0);
    const uint64_t sh_t_rest0 = ggml_moe_shadow_now_qpc();
    if (rc == 0) {
        // 섀도 쪽 시퀀스를 비우고 checkpoint 를 심는다(전용 시퀀스 — main 은 손대지 않는다).
        moe_shadow->shadow_ctx->memory->seq_rm(seq_id, -1, -1);
        const size_t set = moe_shadow->shadow_ctx->state_seq_set_data(seq_id, ckpt.data(), ckpt.size(),
                                                          LLAMA_STATE_SEQ_FLAGS_NONE);
        if (set == 0) {
            rc = SHADOW_RC_CHECKPOINT;
        }
    }

    if (rc == 0) {
        // 복원 digest 등식 — 되뜬 blob 이 원본과 바이트 동일해야 한다.
        std::vector<uint8_t> back(moe_shadow->shadow_ctx->state_seq_get_size(seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
        const size_t got = back.empty() ? 0
                         : moe_shadow->shadow_ctx->state_seq_get_data(seq_id, back.data(), back.size(),
                                                          LLAMA_STATE_SEQ_FLAGS_NONE);
        back.resize(got);
        if (got == 0 || back.size() != ckpt.size() ||
            std::memcmp(back.data(), ckpt.data(), ckpt.size()) != 0) {
            rc = SHADOW_RC_RESTORE_DIGEST;
        }
    }

    ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_RESTORE, request_index, origin_decode_ordinal,
                              0, sh_t_rest0, ggml_moe_shadow_now_qpc(), 0);

    // ★r2 M2(SPEC §4 G-SD3 ⑦) — 트리거 1회의 CPU/backend 원가. 섀도 드라이버는 메인 디코드와
    //   **같은 스레드의 동기 실행**(D6)이므로 여기서 잰 thread CPU 가 곧 섀도의 호스트측 비용이고,
    //   같은 값이 TARGET 사이클의 thread CPU 안에도 들어 있다(부분↔전체 관계가 성립해야 겹침
    //   투영이 성립한다). 섀도의 device 제출/동기는 SHADOW 전용 누산기(view 창에서 열었다)에
    //   쌓이며, 이 창에서는 아직 완료 동기가 없어 device interval 은 NO_BACKEND 로 남는다 —
    //   닫힌 값은 가드가 내는 VIEW 원가 행에 있다.
    const uint64_t sh_t_trig0 = ggml_moe_shadow_now_qpc();
    ggml_moe_shadow_cpu_times sh_cpu_trig0 = {};
    ggml_moe_shadow_cpu_now(&sh_cpu_trig0);
    // ── ⑥ j = 1..D greedy 섀도 스텝 (D1/D5) ──
    if (rc == 0) {
        const int32_t  n_vocab = (int32_t) model.vocab.n_tokens();
        llama_pos      pos     = moe_shadow->shadow_ctx->memory->seq_pos_max(seq_id) + 1;
        llama_token    tok     = -1;

        // 시작 토큰 = main 이 방금 확정한 토큰(confirmed_token). checkpoint 는 그 토큰이
        // KV 에 들어가기 **전** 상태이므로, j=1 이 그 토큰을 먹고 j=2 부터는 섀도 자신의
        // greedy 결과를 먹는다(§D5 결정론).
        for (int32_t j = 1; j <= depth; j++) {
            ggml_moe_shadow_step_begin(request_index, origin_graph_seq, origin_decode_ordinal,
                                       origin_decode_ordinal + (uint32_t) j, j);

            // ★r1 차단 1 수리 — j=1 의 입력은 **메인이 방금 확정한 토큰**이다.
            //   기존 `token 0` 은 첫 섀도 입력부터 실제와 달라 trajectory·recall 전체를 오염시켰다.
            llama_token  b_tok   = (j == 1) ? confirmed_token : tok;
            llama_pos    b_pos   = pos;
            llama_seq_id b_seq   = seq_id;
            llama_seq_id * b_seq_p = &b_seq;
            int32_t      b_nseq  = 1;
            int8_t       b_logit = 1;

            llama_batch b = {};
            b.n_tokens  = 1;
            b.token     = &b_tok;
            b.pos       = &b_pos;
            b.n_seq_id  = &b_nseq;
            b.seq_id    = &b_seq_p;
            b.logits    = &b_logit;

            const int drc = moe_shadow->shadow_ctx->decode(b);
            if (drc != 0) {
                ggml_moe_shadow_step_end();
                rc = SHADOW_RC_DECODE;
                break;
            }

            const float * lg = moe_shadow->shadow_ctx->get_logits_ith(0);
            if (lg && n_vocab > 0) {
                int32_t best = 0;
                float   bv   = lg[0];
                for (int32_t v = 1; v < n_vocab; v++) {
                    if (lg[v] > bv) { bv = lg[v]; best = v; }
                }
                tok = (llama_token) best;      // C2 — 섀도 자체 greedy(메인 샘플러 무접촉)
            }
            pos++;
            steps_done++;

            ggml_moe_shadow_step_end();
        }
    }

    {
        const uint64_t sh_t_trig1 = ggml_moe_shadow_now_qpc();
        ggml_moe_shadow_cpu_times sh_cpu_trig1 = {};
        ggml_moe_shadow_cpu_now(&sh_cpu_trig1);
        // ★r2 M2 — aux=0(WRITER 전용 계약). steps_done 은 kind=6 의 같은 origin ordinal 행 수로
        //   그대로 복원되므로 wire 정보 손실이 없다.
        ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_TRIGGER, request_index, origin_decode_ordinal,
                                  0, sh_t_trig0, sh_t_trig1, 0);
        ggml_moe_shadow_cost_note(GGML_MOE_SHADOW_SPAN_TRIGGER, request_index, origin_decode_ordinal,
                                  0, sh_t_trig0, sh_t_trig1,
                                  &sh_cpu_trig0, &sh_cpu_trig1, &g_moe_bench.shadow_acc);
    }

    } catch (const std::bad_alloc &) {
        ggml_moe_shadow_run_error("out of memory inside the shadow run (checkpoint/op/writer)");
        if (rc == 0) rc = SHADOW_RC_CHECKPOINT;
    } catch (const std::exception & e) {
        ggml_moe_shadow_run_error(e.what());
        if (rc == 0) rc = SHADOW_RC_DECODE;
    } catch (...) {
        ggml_moe_shadow_run_error("unknown exception inside the shadow run");
        if (rc == 0) rc = SHADOW_RC_DECODE;
    }

    // ── ⑦ 정상 종료 — 가드를 "정상"으로 표시하고 그 자리에서 정리를 돌린다.
    //     (순서 sync → restore → view end 는 가드가 소유한다 — SPEC §5 P43j)
    guard.released = true;
    guard.run();
    const int vrc = guard.vrc;
    if (rc == 0 && vrc != 0) {
        rc = SHADOW_RC_VIEW_END;
    }

    // ── ⑧ C1/C3 탈출 등식 — main 의 캐시 관측 상태가 진입 시와 같아야 한다 ──
    {
        uint64_t d_cache_1 = 0, d_tick_1 = 0, d_rf_1 = 0, d_pend_1 = 0;
        ggml_moe_shadow_state_digest(&d_cache_1, &d_tick_1, &d_rf_1, &d_pend_1);
        if (d_cache_1 != d_cache_0 || d_tick_1 != d_tick_0 ||
            d_rf_1 != d_rf_0 || d_pend_1 != d_pend_0) {
            if (rc == 0) rc = SHADOW_RC_VIEW_END;
        }
    }

    return rc != 0 ? rc : steps_done;
}

// ★r1 차단 1 수리 — 서버 배선용 **private wrapper**.
//   공개 안정 API(`include/llama.h`)는 넓히지 않는다(교차 지시). 심볼만 llama.dll 에서
//   내보내고, 호출측(server-context.cpp)이 로컬 extern 선언으로 집어간다.
extern "C" LLAMA_API int llama_shadow_draft_run_private(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_token            confirmed_token,
        uint32_t               request_index,
        uint32_t               origin_graph_seq,
        uint32_t               origin_decode_ordinal);

extern "C" int llama_shadow_draft_run_private(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_token            confirmed_token,
        uint32_t               request_index,
        uint32_t               origin_graph_seq,
        uint32_t               origin_decode_ordinal) {
    if (!ctx) return -1;
    return ctx->shadow_draft_run(seq_id, confirmed_token, request_index,
                                 origin_graph_seq, origin_decode_ordinal);
}
