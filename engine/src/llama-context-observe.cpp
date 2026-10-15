// [R-BC4] llama-context-observe.cpp - the observe facade (RB_DESIGN 2-8, the R-BC4 row): the
//   benchmark observe state g_moe_bench, its seven definitions and the trace graph sequence
//   counter, moved out of llama-context.cpp. The core keeps one-line hooks that call
//   moe_bench_stage_begin, moe_bench_rotate, moe_bench_note_submit and moe_bench_note_sync and
//   read s_moe_trace_graph_seq_next through llama-context-moe.h. Internal to src/.

#include "llama-context.h"
#include "llama-context-moe.h"

#include "ggml-moe-phasea.h"
#include "ggml-moe-shadow.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdint>
#include <cassert>  // [R-BC4] the core reaches <cassert> twice (the second time through
                     //   llama-kv-cells.h), so assert.h expands identically in both units

using namespace llama_ctx_moe;

// ===========================================================================
// ★r2 M2 — benchmark 전용 TARGET(토큰 사이클) 계측  [SPEC_SHADOW_DRAFT §4 G-SD3 ①⑦⑨]
// ===========================================================================
// 무엇을 고쳤나(r2 교차 ⑥ "신규 계측 모순" 2건):
//   ① 옛 TARGET 은 **섀도 실행 게이트(MOE_SHADOW_DRAFT)가 ON 일 때만** 발행됐다. 그러면 짝런
//      OFF 팔에서 엔진이 분모를 만들 수 없어, 판독기가 요구하는 "동일 ordinal OFF/ON 짝"이
//      원천적으로 성립하지 않는다(생산자↔소비자 비양립).
//      → 계측을 실행에서 떼어낸 **MOE_SHADOW_BENCH** 게이트로 옮겼다. 게이트 행렬과 제품 OFF
//        "섀도 흔적 0" 계약은 ggml-moe-shadow.h 머리주석의 표가 권위다.
//   ② 옛 TARGET 은 비동기 `graph_compute` **제출 둘레**만 쟀다. 실제 섀도 실행은 sampler accept
//      뒤 서버에서 일어나므로 그 구간 밖이고, 따라서 ON/OFF 비율이 섀도 비용을 담지 못했다.
//      → 구간을 **토큰 사이클 전체**로 바꿨다.
//
// 토큰 사이클의 정의(닫는 일은 다음 열림이 한다):
//   cycle(k) = [ ordinal k 의 graph 진입 시각 , ordinal k+1 의 graph 진입 시각 )
//   이 창은 build/alloc/set_inputs/제출/device 대기/샘플링/서버 왕복/**섀도 트리거 실행**을
//   전부 포함한다. ordinal 은 Phase A identity(`decode ∧ n_tokens=1 ∧ n_outputs=1`)에서 오며,
//   따라서 이 축은 **Phase A 가 돌고 있어야** 생긴다(안 돌면 0행 → 판독기가 fail-close).
//   요청 경계·비연속 ordinal 에서 열려 있던 사이클은 **버린다** — 요청 tail 은 SPEC §4 조인
//   계약상 censored 라 분모 밖이므로, 끝을 모르는 구간을 날조하지 않는다.
//
// 이벤트 귀속(staging): process_ubatch 진입(step1)부터 Phase A graph_end 직후의 회전(step5)
//   까지 일어난 backend 이벤트는 **다음 사이클의 것**이다(그 사이클이 step1 시각에 시작하므로).
//   그래서 step1 에서 stage 를 비우고, 회전 때 stage 를 새 사이클의 초기값으로 넘긴다. 회전
//   이후~다음 step1 사이의 이벤트는 열린 사이클에 직접 쌓인다.
//
// C0 역할 격리: SHADOW 역할의 제출/동기는 **별도 누산기**로 간다(main 사이클의 device 구간을
//   오염시키지 않는다). 섀도 자신의 backend 비용은 VIEW 원가 행(kind=8)에 실린다 — VIEW 창이
//   섀도의 모든 제출과 그 완료 동기를 포함하는 유일한 창이기 때문이다.
//
// 스레딩: 이 상태는 Phase A(g_pa)와 같은 전제를 쓴다 — 디코드 슬롯 스레드 1개(`-np 1`)에서만
//   전이한다. 섀도 드라이버도 같은 스레드의 동기 실행이다(SPEC D6). begin/end 스냅샷의
//   thread_id 가 달라지면 kind=8 이 CROSS_THREAD 로 사실을 표시한다(조용한 오측정 금지).
// [R-BC1] moe_bench_cycle and moe_bench_state are defined in llama-context-moe.h. g_moe_bench
// and moe_bench_acc_reset stay in this unit and leave the anonymous namespace for
// llama_ctx_moe, so the shadow selftest TU reaches them through that header.
namespace llama_ctx_moe {

moe_bench_state g_moe_bench;

void moe_bench_acc_reset(ggml_moe_shadow_backend_acc & a) {
    a.wait_qpc          = 0;
    a.first_submit_qpc  = 0;
    a.last_sync_end_qpc = 0;
    a.n_submits         = 0;
    a.n_syncs           = 0;
}

} // namespace llama_ctx_moe

namespace { // the anonymous namespace resumes

void moe_bench_acc_merge(ggml_moe_shadow_backend_acc & dst,
                         const ggml_moe_shadow_backend_acc & src) {
    dst.wait_qpc += src.wait_qpc;
    if (src.first_submit_qpc != 0 &&
        (dst.first_submit_qpc == 0 || src.first_submit_qpc < dst.first_submit_qpc)) {
        dst.first_submit_qpc = src.first_submit_qpc;
    }
    if (src.last_sync_end_qpc > dst.last_sync_end_qpc) {
        dst.last_sync_end_qpc = src.last_sync_end_qpc;
    }
    dst.n_submits += src.n_submits;
    dst.n_syncs   += src.n_syncs;
}

// 지금 backend 이벤트가 쌓일 누산기. 열린 사이클도 staging 도 없으면 nullptr —
// 그 이벤트는 **버린다**(존재하지 않는 사이클에 비용을 붙이지 않는다).
ggml_moe_shadow_backend_acc * moe_bench_acc_for(bool is_shadow) {
    if (is_shadow)            return &g_moe_bench.shadow_acc;
    if (g_moe_bench.staging)  return &g_moe_bench.stage_acc;
    return g_moe_bench.cycle.open ? &g_moe_bench.cycle.acc : nullptr;
}

} // namespace

// [R-BC1] moe_bench_note_submit and moe_bench_note_sync leave the anonymous namespace for
// llama_ctx_moe for the same reason; moe_bench_acc_for, which they call, stays core-private.
namespace llama_ctx_moe {

void moe_bench_note_submit(bool is_shadow, uint64_t t_qpc) {
    ggml_moe_shadow_backend_acc * a = moe_bench_acc_for(is_shadow);
    if (!a) return;
    if (a->first_submit_qpc == 0) a->first_submit_qpc = t_qpc;
    a->n_submits++;
}

void moe_bench_note_sync(bool is_shadow, uint64_t t0_qpc, uint64_t t1_qpc) {
    // ★r6 P0 — `SYNC_OBS_DROP`(0x10) 음성의 **유일한 소비 지점**. 실 fence 는 이미 끝났고
    //   (`synchronize()` 는 fault 와 무관하게 항상 동기화한다) 여기서는 **관측 기록만** 건너뛴다.
    //   ★SHADOW 역할에 한정 — main 역할 누산기는 절대 건드리지 않는다(상위 계약 C1 보존).
    //   무장 결과: SHADOW 누산기가 `n_submits>0 ∧ n_syncs==0 ∧ last_sync_end==0` 이 되어
    //   VIEW 원가 행이 device interval 0 · NO_BACKEND 로 남고, 판독기의 VIEW 하드 계약이
    //   그걸 잡아야 한다(안 잡으면 판독 경로에 구멍이 있다는 뜻 = 음성의 목적).
    if (is_shadow && (ggml_moe_shadow_test_fault_bits() & GGML_MOE_SHADOW_FAULT_SYNC_OBS_DROP)) {
        return;
    }
    ggml_moe_shadow_backend_acc * a = moe_bench_acc_for(is_shadow);
    if (!a) return;
    if (t1_qpc > t0_qpc)              a->wait_qpc += (t1_qpc - t0_qpc);
    if (t1_qpc > a->last_sync_end_qpc) a->last_sync_end_qpc = t1_qpc;
    a->n_syncs++;
}

} // namespace llama_ctx_moe

namespace llama_ctx_moe { // [R-BC4] promoted from the anonymous namespace: the core hooks call it

// step1 — 이 graph 의 진입 시각·CPU 시계를 집고 staging 을 연다(MAIN 역할에서만 호출).
void moe_bench_stage_begin() {
    if (!ggml_moe_shadow_bench_enabled()) return;   // 게이트 OFF = QPC 호출조차 없다
    g_moe_bench.stage_t_qpc = ggml_moe_shadow_now_qpc();
    ggml_moe_shadow_cpu_now(&g_moe_bench.stage_cpu);
    moe_bench_acc_reset(g_moe_bench.stage_acc);
    g_moe_bench.staging = true;
}

// step5 — Phase A graph_end 직후. 열린 사이클을 닫아 kind=7 TARGET + kind=8 원가를 내고,
//          이 graph 의 ordinal 로 새 사이클을 연다.
void moe_bench_rotate() {
    if (!ggml_moe_shadow_bench_enabled() || !g_moe_bench.staging) return;
    g_moe_bench.staging = false;

    struct ggml_moe_phasea_ident id;
    ggml_moe_phasea_identity(&id);
    if (!id.valid) {
        // 적격 decode graph 가 아니다(프리필·서버 기동 prelude·다중 출력 batch). 이 graph 의
        // backend 이벤트는 여전히 "열려 있는 사이클" 안에서 일어난 일이므로 그쪽으로 합친다.
        if (g_moe_bench.cycle.open) {
            moe_bench_acc_merge(g_moe_bench.cycle.acc, g_moe_bench.stage_acc);
        }
        return;
    }

    if (g_moe_bench.cycle.open) {
        const bool consecutive = (g_moe_bench.cycle.req == id.request_index) &&
                                 (g_moe_bench.cycle.ord + 1u == id.decode_ordinal);
        if (consecutive) {
            ggml_moe_shadow_span_note(GGML_MOE_SHADOW_SPAN_TARGET,
                                      g_moe_bench.cycle.req, g_moe_bench.cycle.ord, /*j=*/0,
                                      g_moe_bench.cycle.t0_qpc, g_moe_bench.stage_t_qpc,
                                      /*aux=*/0);
            ggml_moe_shadow_cost_note(GGML_MOE_SHADOW_SPAN_TARGET,
                                      g_moe_bench.cycle.req, g_moe_bench.cycle.ord, /*j=*/0,
                                      g_moe_bench.cycle.t0_qpc, g_moe_bench.stage_t_qpc,
                                      &g_moe_bench.cycle.cpu0, &g_moe_bench.stage_cpu,
                                      &g_moe_bench.cycle.acc);
            g_moe_bench.emitted++;
        } else {
            // 요청 경계·비연속 ordinal = 미종결 사이클. 끝을 모르므로 폐기한다(SPEC 조인 계약
            // 상 요청 tail 은 censored 이니 분모 결손이 아니다).
            g_moe_bench.dropped++;
            LLAMA_LOG_DEBUG("%s: bench target cycle dropped (unterminated) "
                            "open=(req=%u, ord=%u) next=(req=%u, ord=%u) emitted=%llu dropped=%llu\n",
                            __func__, g_moe_bench.cycle.req, g_moe_bench.cycle.ord,
                            id.request_index, id.decode_ordinal,
                            (unsigned long long) g_moe_bench.emitted,
                            (unsigned long long) g_moe_bench.dropped);
        }
    }

    g_moe_bench.cycle.open   = true;
    g_moe_bench.cycle.req    = id.request_index;
    g_moe_bench.cycle.ord    = id.decode_ordinal;
    g_moe_bench.cycle.t0_qpc = g_moe_bench.stage_t_qpc;
    g_moe_bench.cycle.cpu0   = g_moe_bench.stage_cpu;
    g_moe_bench.cycle.acc    = g_moe_bench.stage_acc;   // 이 graph 의 제출/동기는 새 사이클 것
}

} // namespace llama_ctx_moe

namespace llama_ctx_moe {

// [R-BC4] the trace graph sequence counter, a function-local static of
//   llama_context::process_ubatch before the move (declared owner move, constant-initialised).
std::atomic<uint32_t> s_moe_trace_graph_seq_next{0};

} // namespace llama_ctx_moe
