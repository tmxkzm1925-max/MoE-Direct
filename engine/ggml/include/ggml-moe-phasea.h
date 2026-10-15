// MoE 프리페치 Phase A — 무행동 계측 모듈(관측 전용: 출력·캐시·읽기 무변경).
//
// 사양 권위: bench/moe-direct/PREFETCH_DESIGN.md §3(★FROZEN v1 · 하드 동결 §3.1~3.5)
// wire 계약: bench/moe-direct/SPEC_PHASEA_WIRE.md (필드 단위 확정 — 후속 러너/replay 인터페이스)
//
// 이 모듈은 ggml-base에 포함되어 exported C ABI로 노출된다. 소비자는 셋:
//   ① ggml/src/ggml-moe-direct.cpp  — acquire/release/pump 훅(캐시 mutation ledger·A2·A3·ops)
//   ② src/llama-context.cpp          — graph 경계·가중치 스냅샷·eval callback 설치(accuracy)
//   ③ tools/server/server-context.cpp— 요청(run) 경계
//
// ★★ 동시성 계약(중요) ★★
//   이 모듈은 **자체 뮤텍스를 두지 않는다.** 전 진입점이 아래 둘 중 하나이기 때문이다:
//     (a) ggml_moe_direct_acquire/release 내부 — 호출자가 g_mod.mutex를 보유하며, acquire 자체가
//         "단일 스레드 호출 전제"(ggml-moe-direct.h) 이므로 직렬화된다.
//     (b) graph/run 경계·flush — llama_decode를 부르는 단일 스레드에서만 실행되며, 그 시점에는
//         (a)가 진행 중일 수 없다(graph_begin < graph_compute < graph_end 순서).
//   자체 뮤텍스를 두면 (a)에서 direct→pa, flush에서 pa→direct 순으로 락 순서가 역전된다.
//   ★새 호출부를 추가할 때 이 전제를 반드시 재확인할 것.
//
// Windows 전용 — 비Windows는 전부 no-op 스텁으로 컴파일된다.
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 모드 (env MOE_DIRECT_PHASEA — off[기본] / accuracy / timing)
// ---------------------------------------------------------------------------
#define GGML_MOE_PHASEA_OFF      0
#define GGML_MOE_PHASEA_ACCURACY 1
#define GGML_MOE_PHASEA_TIMING   2

// 최초 호출 시 env를 해소해 캐시한다(테스트는 ggml_moe_phasea_reset_for_test()로 재해소).
GGML_API int32_t      ggml_moe_phasea_mode(void);
GGML_API bool         ggml_moe_phasea_enabled(void);      // mode != off
GGML_API const char * ggml_moe_phasea_mode_name(void);    // "off"/"accuracy"/"timing"
GGML_API uint64_t     ggml_moe_phasea_boot_id(void);      // 사이드카↔metrics 조인 키
GGML_API const char * ggml_moe_phasea_out_prefix(void);   // MOE_DIRECT_PHASEA_OUT 원값("" 가능)
GGML_API uint32_t     ggml_moe_phasea_buf_mb(void);       // 버퍼 상한(MiB)
GGML_API bool         ggml_moe_phasea_eval_callback_installed(void);
GGML_API uint64_t     ggml_moe_phasea_now_qpc(void);      // QueryPerformanceCounter 원값(off여도 유효)

// ---------------------------------------------------------------------------
// seal 통지 (ggml-moe-direct.cpp seal 성공 말미) — 모델 상수 + arm 시작 슬롯 상태 확정
//   layer_stride_bytes[i] = moe_layers[i] 층의 record stride(=1 record 읽기 바이트)
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_phasea_note_seal(
    int32_t n_expert, int32_t n_expert_used, int32_t n_moe_layers, int32_t n_slots,
    const int32_t * moe_layers, const uint64_t * layer_stride_bytes);

// ---------------------------------------------------------------------------
// 요청(run) 경계 — tools/server/server-context.cpp
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_phasea_run_begin(void);
GGML_API void ggml_moe_phasea_run_end(uint32_t predicted_n, uint32_t prompt_n);

// ---------------------------------------------------------------------------
// graph 경계 — src/llama-context.cpp (phase: 0=decode 1=prefill, ubatch.n_tokens 원값 기준)
//   graph_end는 무행동 증명용 형상(n_nodes/n_splits)을 함께 넘긴다.
// ---------------------------------------------------------------------------
// ★r1 차단 1 수리 — 섬도 트리거가 쉼 수 있는 **coherent identity 스냅샷**.
//   두 개의 독립 getter 로 나누면 그 사이에 graph 경계가 끊길 수 있어 한 번에 읽는다.
//   갱신 조건 = `phase=decode ∧ n_tokens=1 ∧ n_outputs=1` graph 가 **끝났을 때**만.
//   run begin/end 에서 무효화된다(요청 경계를 넘는 재사용 금지).
struct ggml_moe_phasea_ident {
    int32_t  valid;            // 0 = 쓸 수 없음(트리거 금지)
    uint32_t request_index;
    uint32_t graph_seq;
    uint32_t decode_ordinal;   // 이 요청 안에서의 0-based decode 순번
};
GGML_API void ggml_moe_phasea_identity(struct ggml_moe_phasea_ident * out);

// 이 graph 의 확정 출력 행 수를 PA 에 알린다(graph_begin 직후 호출).
GGML_API void ggml_moe_phasea_note_outputs(int32_t n_outputs);

GGML_API void ggml_moe_phasea_graph_begin(uint8_t phase, uint32_t n_tokens);
GGML_API void ggml_moe_phasea_graph_end  (int32_t n_nodes, int32_t n_splits);

// ---------------------------------------------------------------------------
// accuracy 모드 가중치 host 스냅샷 — src/llama-context.cpp (모델 로드 후 1회·멱등)
//   router      = blk.<l>.ffn_gate_inp.weight        [n_embd, n_expert] F32
//   post_norm   = blk.<l>.post_attention_norm.weight [n_embd]           F32
//   ★target layer 1..n_layer_exec-1 만 필요(전이 l→l+1 · NextN 층 제외).
//   반환 0=성공(또는 무동작), <0=실패(타입/차원 불일치 등 — 호출측은 로그만).
// ---------------------------------------------------------------------------
GGML_API int  ggml_moe_phasea_snapshot_configure(int32_t n_layer_exec, int32_t n_embd,
                                                 int32_t n_expert, int32_t n_expert_used,
                                                 float rms_eps);
GGML_API bool ggml_moe_phasea_snapshot_needed(void);   // accuracy && 아직 미완료
GGML_API int  ggml_moe_phasea_snapshot_layer(int32_t target_layer,
                                             const struct ggml_tensor * router,
                                             const struct ggml_tensor * post_norm);
GGML_API int  ggml_moe_phasea_snapshot_seal(void);     // 전 층 채워졌는지 검사 + 확정
GGML_API uint64_t ggml_moe_phasea_snapshot_bytes(void);

// eval callback — accuracy 모드에서 llama_context가 sched에 설치한다(timing/off는 설치 금지).
GGML_API bool ggml_moe_phasea_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);
GGML_API void ggml_moe_phasea_note_eval_callback_installed(void);

// ---------------------------------------------------------------------------
// ggml-moe-direct.cpp 전용 훅 (전부 g_mod.mutex 보유 상태에서 호출)
// ---------------------------------------------------------------------------

// acquire 진입. t_enter_qpc는 **락 취득 이전**에 채취한 값(D_j).
// out_is_canonical: 이 (graph, layer) 조합의 첫 acquire면 1.
GGML_API void ggml_moe_phasea_acquire_enter(int32_t mlpos, int32_t part, int32_t n_ids,
                                            const int32_t * expert_ids, int32_t is_prefill,
                                            uint64_t t_enter_qpc, int32_t * out_is_canonical);

// claim에서 victim이 기존 owner를 갖고 있었을 때(축출 발생) 1건.
GGML_API void ggml_moe_phasea_acquire_evict(int32_t slot, int64_t evicted_record_idx,
                                            int32_t new_expert);

// 요청 인덱스 순서대로 n_ids회 호출(hit/miss·slot·논리 tick).
GGML_API void ggml_moe_phasea_acquire_expert(int32_t expert, int32_t is_hit, int32_t slot,
                                             uint64_t logical_tick);

// 배치 pump 중 outstanding 수가 바뀔 때마다(제출 버스트 후 / 회수 후) 호출 — Q_j 적분.
GGML_API void ggml_moe_phasea_io_qd(int32_t n_outstanding, uint64_t t_qpc);

// demand read op 1건 회수 확정 시(publish 직전).
GGML_API void ggml_moe_phasea_io_op(int64_t record_idx, uint64_t file_offset, uint32_t bytes,
                                    int32_t slot, uint32_t batch_order,
                                    uint64_t submit_seq, uint64_t submit_qpc,
                                    uint64_t retire_seq, uint64_t retire_qpc,
                                    int32_t was_pending);

// acquire 종료(R_j) — n_miss/miss_bytes는 이 배치의 유니크 miss 집계.
GGML_API void ggml_moe_phasea_acquire_exit(int32_t n_miss, uint64_t miss_bytes, uint64_t t_exit_qpc);

// release 경계.
GGML_API void ggml_moe_phasea_release_enter(int32_t mlpos, int32_t part, int32_t n_ids,
                                            uint64_t t_qpc);
GGML_API void ggml_moe_phasea_release_expert(int32_t expert, int32_t slot,
                                             int32_t lease_before, int32_t lease_after);

// ---------------------------------------------------------------------------
// flush — ggml_moe_direct_finalize()가 **락을 잡기 전에** 1회 호출한다(멱등).
//   사이드카 4파일 기록 + 버퍼된 phase_a_run 레코드를 metrics JSONL에 일괄 방출.
// ---------------------------------------------------------------------------
GGML_API void ggml_moe_phasea_flush(void);

// selftest 전용: 전역 상태 완전 초기화 + env 재해소.
GGML_API void ggml_moe_phasea_reset_for_test(void);

// selftest 전용 관측 창구(프로덕션 경로 무접촉).
GGML_API void ggml_moe_phasea_test_counts(uint64_t * out_events, uint64_t * out_ops,
                                          uint64_t * out_ledger, uint64_t * out_sets,
                                          uint64_t * out_dropped, int32_t * out_complete);

#ifdef __cplusplus
}
#endif
