// R-B5b (RB_DESIGN.md 2-6, RB5B_ATOM_SPEC.json): the dispatch/queue unit.
// The 90 definitions below are the core text at the span the move map names, moved
// verbatim - body tokens unchanged, with the declarator changes the spec approves
// (`promote-anon-namespace-to-named`, `remove-inline` on the three cross-cluster
// `inline` helpers of RB5B_ATOM_SPEC.json :: inline_dispositions).  They keep their
// source order, which is the order that compiled in the core TU.
// Windows only, exactly like the region it came from: the core TU compiles its
// non-Windows half from the stub region at the bottom of that file, and this unit has
// no stub half of its own.

// The head include block of ggml-moe-direct.cpp, reproduced so the moved definitions
// compile in the environment they were written in.
#include "ggml-moe-direct.h"
#include "ggml-moe-direct-internal.h"
#include "ggml-moe-trace.h"
#include "ggml-moe-phasea.h"
#include "ggml-moe-phaseb.h"
#include "ggml-moe-shadow.h"
#include "ggml-moe-leadtime.h"
#include "ggml-moe-gpuearly.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-impl.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

#include "ggml-moe-direct-types.h"
#include "ggml-moe-direct-manifest.h"
#include "ggml-moe-direct-state.h"
#include "ggml-moe-direct-observe.h"
#include "ggml-moe-direct-dispatch.h"

using namespace ggml_moe_direct_impl;

namespace ggml_moe_direct_impl {

// FORWARD DECLARATIONS, carried over with the bodies they serve: the core TU wrote
// them because a definition further down is called further up, and the same order
// holds here.  Declarations, so the move map does not carry them.
// ★rank 처분 기록의 정의는 §10.6 앞(`pb_dispose_rank_locked`)이다. PB virtual A3 의 ③축
//   확정은 그보다 앞 절(제출 pump·record terminal)에서 일어나므로 전방 선언한다.
void pb_dispose_rank_locked(int32_t row, uint64_t intent_ordinal, uint16_t rank, uint8_t disp,
                            uint32_t request_epoch);
bool v3_maybe_terminal_locked(int32_t oi);                                   // 아래 정의
bool v3_copy_complete_locked(const ModuleState::V3CopyJob & job, bool ok, uint64_t start_qpc);  // 〃

// ★★R-1 — A0-4 이중 credit 식의 정의는 §10.6(admission)에 있다. continuation wave 재검사가
//   그보다 앞 절(pump)에서 **같은 식**을 써야 하므로 전방 선언한다(식을 두 벌로 두지 않는다).
bool pbv_credit_ok_locked(int32_t w);


// command 1건이 보유하는 id 배열 capacity(고정) — 정상 운전 중 재할당 0이 되도록 상한을 잡는다.
// n_expert(demand 상한)와 PA absent query의 K(16) 중 큰 값. mem echo도 같은 식을 쓴다.
inline size_t cq_ids_capacity() {
    return (size_t)(g_mod.n_expert > 16 ? g_mod.n_expert : 16);
}

// ── control_q(생산자 측) ───────────────────────────────────────────────────
// 자유 슬롯 1개 확보. STOPPING이면 -1(신규 command 거절). 만석이면 대기(무손실 — drop 없음).
int32_t cq_alloc() {
    // ★R1 #3 ① / R2 #1 — capture 훅과 **동일한 단일 원자 래치**를 확인한다(둘이 같은 순간에 닫힘).
    if (ggml_moe_phaseb_capture_closed()) return -1;
    std::unique_lock<std::mutex> lk(g_mod.q_mutex);
    if (g_mod.q_stopping) return -1;
    if (g_mod.q_free_top == 0) {
        g_mod.stat_q_full_waits.fetch_add(1, std::memory_order_relaxed);
        g_mod.q_free_cv.wait(lk, [] { return g_mod.q_free_top > 0 || g_mod.q_stopping; });
    }
    if (g_mod.q_stopping) return -1;
    const int32_t ci = g_mod.q_free[--g_mod.q_free_top];
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.in_use = true;
    c.done   = false;
    c.result = 0;
    c.deps   = 0;
    c.pa_open = false;
    c.seq    = g_mod.q_seq_next++;
    return ci;
}

// enqueue + per-command CV 대기(구간 중 g_mod.mutex 무보유). 반환 = command 결과.
int cq_submit_and_wait(int32_t ci) {
    std::unique_lock<std::mutex> lk(g_mod.q_mutex);
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    if (g_mod.q_stopping) { c.done = true; return -1; }
    g_mod.q_ring[(size_t)((g_mod.q_head + g_mod.q_count) % MOE_DIRECT_CONTROL_Q_CAP)] = ci;
    g_mod.q_count++;
    if ((uint64_t)g_mod.q_count > g_mod.stat_q_max_depth.load(std::memory_order_relaxed))
        g_mod.stat_q_max_depth.store((uint64_t)g_mod.q_count, std::memory_order_relaxed);
    SetEvent(g_mod.command_event); // ★enqueue와 같은 mutex 아래(lost-wake 방지)
    c.cv.wait(lk, [&c] { return c.done; });
    return c.result;
}

void cq_free(int32_t ci) {
    std::lock_guard<std::mutex> lk(g_mod.q_mutex);
    g_mod.cmds[(size_t)ci].in_use = false;
    g_mod.q_free[g_mod.q_free_top++] = ci;
    g_mod.q_free_cv.notify_one();
}

// ── control_q(dispatcher 측) ───────────────────────────────────────────────
// 다음 command 1건 조회(pop 아님). 큐가 비면 같은 mutex 아래에서 ResetEvent 후 -1.
int32_t disp_peek() {
    std::lock_guard<std::mutex> lk(g_mod.q_mutex);
    if (g_mod.q_count == 0) {
        // ★R1 #2 ① lost-wake — command_event 는 intent enqueue(worker)도 신호한다. 그래서
        //   command 큐가 비었더라도 **intent_q 공백까지 같은 임계 구간에서 확인**한 뒤에만
        //   내린다(락 순서 q_mutex → int_mu 단방향 — PB는 q_mutex를 잡지 않는다).
        // ★★정지 신호 보존(R2 후속 근인 수리) — `dispatcher_stop_and_join`은 q_stopping을 세우고
        //   SetEvent(command_event)를 **단 1회** 보낸 뒤 무한 join 한다. 그 신호가 이 자리에서
        //   지워지면(디스패처가 pass 진입 시 읽은 stopping 은 아직 false) 그 pass 는 graceful 평가를
        //   건너뛴 채 INFINITE 대기로 들어가고, 깨울 신호원이 영영 없어 **전 스레드 idle 교착**이
        //   된다(실측 16% 간헐·정지 지점 무작위). manual-reset 이벤트를 서 있는 채로 두면 다음
        //   pass 가 즉시 재개돼 stopping 을 재평가한다.
        // ★★보완 ①(Codex R1-후속 CONFIRMED) — intent 생산자 lost-wake. `intent_depth()==0`
        //   확인과 `ResetEvent`가 **int_mu 단일 임계구간이 아니다**: 그 사이에 worker 가
        //   enqueue+SetEvent 하면 여기서 신호를 지워버리고, admission 은 pass 0 에서만 돌므로
        //   dispatcher 가 정상 운전 INFINITE 대기로 들어가 intent 가 다음 demand command 까지
        //   정체된다(프리페치 ON 실기동 liveness). reset 직후 **재확인해 재무장**하면 창이 닫힌다:
        //     · reset 이전 enqueue  → 재확인이 depth!=0 을 보고 SetEvent 로 복구
        //     · 재확인 이후 enqueue → worker 자신의 SetEvent 가 살아 있다
        // ★★PB virtual A2(§2-0 A0-4 wake/turn 결과 계약 ②) — **drain-defer park**.
        //   admission 이 credit 부족(또는 demand predicate)으로 이 turn 을 접었다면, 큐에
        //   남은 **변화 없는** prefetch intent 만으로는 command_event 를 다시 세우지 않는다.
        //   구 규칙(intent 가 있으면 재무장)을 그대로 두면 credit 이 변하지 않는 동안 pass 가
        //   무한 재진입해 CPU 를 태우고 진단 카운터를 오염시킨다(r7 반례).
        //   ★park 는 credit 변화(child terminal·record terminal)와 command 처리가 내린다 —
        //   그래서 ③ "child terminal/credit 변화에는 재시도 · demand/control 은 즉시 처리"가
        //   성립하고, 채널 이벤트(handle 1..QD)는 이 마스킹과 무관하게 계속 dispatcher 를 깨운다.
        // ★★**pause 신호 보존**(26-08-17 PB virtual 라운드 자기 적발 — 위 정지 신호 보존과
        //   **같은 부류의 pre-existing 경합**이다). `ggml_moe_direct_test_dispatcher_pause(1)` 도
        //   `q_pause_req` 를 세우고 `SetEvent(command_event)` 를 **단 1회** 보낸 뒤 `q_paused` 를
        //   무한 대기한다. 그 신호가 이 자리에서 지워지면 dispatcher 는 pause 지점에 못 가고
        //   INFINITE 대기로 들어가며, 테스트 스레드는 영영 깨지 않는다 — P25-⑥ 이 그 반례이고
        //   이 라운드에서 **2/4 회 재현**됐다(정지 신호 쪽은 R2 후속에서 이미 같은 처방을 받았는데
        //   pause 축만 빠져 있었다). q_stopping 과 동일하게 reset 대상에서 제외한다.
        if (!g_mod.q_stopping && !g_mod.q_pause_req &&
            (g_mod.pbv_admit_parked || ggml_moe_phaseb_intent_depth() == 0)) {
            ResetEvent(g_mod.command_event);
            if (!g_mod.pbv_admit_parked && ggml_moe_phaseb_intent_depth() != 0)
                SetEvent(g_mod.command_event);
        }
        return -1;
    }
    return g_mod.q_ring[(size_t)g_mod.q_head];
}

void disp_pop() {
    std::lock_guard<std::mutex> lk(g_mod.q_mutex);
    g_mod.q_head = (g_mod.q_head + 1) % MOE_DIRECT_CONTROL_Q_CAP;
    g_mod.q_count--;
}

// command terminal — 결과 확정 + 대기 생산자 기상(dispatcher가 g_mod.mutex 보유 상태로 호출).
void disp_complete(int32_t ci, int rc) {
    std::lock_guard<std::mutex> lk(g_mod.q_mutex);
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    if (c.done) return;
    c.result = rc;
    c.done   = true;
    c.cv.notify_one();
}

// ── pending 풀 ─────────────────────────────────────────────────────────────
void pio_reset(PendingIo & op) {
    op.m = BatchMiss();
    op.mlpos = -1;
    op.phase = Phase::DECODE;
    op.origin = PIO_ORIGIN_DEMAND;
    op.state = PIO_CLAIMED;
    op.channel = -1;
    op.request_identity = 0;
    op.intent_seq = GGML_MOE_PHASEB_NA_U64;
    op.retry_of = -1;
    op.in_use = false;
    op.waiters.clear(); // capacity 유지(정상 운전 중 재할당 0)
    // ★R1 #1 — B2 확장 필드 전건 초기화. pending 풀은 고정 인덱스 재사용이므로 하나라도
    //   남으면 앞 op(joined prefetch·retry)의 래치·epoch·io 인덱스가 다음 op으로 새어
    //   pb_io/digest/used 삼분할이 오염된다(같은 index를 joined→prefetch→retry→demand 순으로
    //   돌리면 즉시 재현 — 픽스처 P12).
    op.pb_request_epoch      = GGML_MOE_PHASEB_NA_U32;
    op.pb_graph_epoch        = GGML_MOE_PHASEB_NA_U32;
    op.pb_graph_seq          = GGML_MOE_PHASEB_NA_U32;
    op.pb_join_request_epoch = GGML_MOE_PHASEB_NA_U32;
    op.pb_join_graph_epoch   = GGML_MOE_PHASEB_NA_U32;
    op.pb_join_graph_seq     = GGML_MOE_PHASEB_NA_U32;
    op.pb_join_cmd           = -1;
    op.pb_rank               = GGML_MOE_PHASEB_NA_U16;
    op.pb_io_index           = -1;
    op.pb_submit_event_seq   = GGML_MOE_PHASEB_NA_U64;
    op.pb_retry_of_event_seq = GGML_MOE_PHASEB_NA_U64;
    op.pb_is_retry           = false;
    op.pb_ever_joined        = false;
    op.pb_blocked_start      = 0;
}

// ★5c-T §1 — pending op 병렬 스탬프 side table 접근/리셋(수집 OFF 면 미할당·no-op).
inline PbOpStamp * pb_op_stamp_at(int32_t oi) {
    if (!g_mod.pb_collect || oi < 0 || oi >= (int32_t)g_mod.pb_op_stamp.size()) return nullptr;
    return &g_mod.pb_op_stamp[(size_t)oi];
}

inline void pb_op_stamp_reset(int32_t oi) {
    if (PbOpStamp * st = pb_op_stamp_at(oi)) *st = PbOpStamp{};
}

// A0-2 동시 물리 폭 `W_l = min(P_l, io_part_parallelism)`.
// ★★문면 정정(S-6 · 교차 r2 비차단 지적) — 구 주석은 "현 엔진에서 `W_l == P_l` 이고
//   continuation 은 발생하지 않는다"고 적어 있었는데, **R-1 이후 거짓**이다:
//   `kFloor[3]`(=`max_parts` 형상 하한)은 R-1 에서 **철거**됐고, `disp_pump_locked` 는
//   이제 wave 단위(`W_l` 개씩)로 제출한다. ⇒ **`W_l < P_l` 은 정상 형상**이고 record 당
//   wave 는 `ceil(P_l / W_l)` 개이며 A0-4 continuation 은 **실제로 발동한다**
//   (증거: V3W narrow arm · V3P14b `waves=4` · V3P14c `cont_defer_events 0→1` ·
//    V3IO3RT 의 판독기 왕복). 이 주석이 반대로 남아 있었던 것이 비거동 드리프트다.
int32_t pbv_wave_width(int32_t mlpos) {
    // ★intent 의 `target_mlpos` 는 capture 훅이 채운 값이라 경계를 여기서 본다(0=폭 없음 ⇒
    //   `pbv_ncap_locked` 가 0 을 돌려 그 층은 프리페치 OFF 로 처분된다).
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.layers.size()) return 0;
    const int32_t p  = v3_parts_of(mlpos);
    const int32_t pp = g_mod.v3_io_param_val[3];   // io_part_parallelism(bin=0=미해소)
    return (pp > 0 && pp < p) ? pp : p;
}

// ②축 — child 물리 in-flight. **제출/회수 경로가** 부른다(IO3 방출부가 아니다 — A0-1).
//   IO3 children 의 QD snapshot 6필드는 이 값을 읽기만 한다.
inline void v3_child_inflight_add_locked(bool prefetch) {
    if (prefetch) g_mod.v3_inflight_prefetch++; else g_mod.v3_inflight_demand++;
}

inline void v3_child_inflight_sub_locked(bool prefetch) {
    if (prefetch) { if (g_mod.v3_inflight_prefetch > 0) g_mod.v3_inflight_prefetch--; }
    else          { if (g_mod.v3_inflight_demand   > 0) g_mod.v3_inflight_demand--;   }
    // ★A0-4 wake 결과 계약 ③ — credit 이 실제로 변했다. drain-defer park 를 내린다.
    g_mod.pbv_admit_parked = false;
}

inline PbvRec * pbv_rec_at(int32_t oi) {
    if (oi < 0 || oi >= (int32_t)g_mod.pbv_rec.size()) return nullptr;
    return &g_mod.pbv_rec[(size_t)oi];
}

// ③축(A3) latch 조회 — `pb_submit_event_seq` 대신 쓰는 mode-neutral 판정원.
inline bool pbv_rec_issued(int32_t oi) {
    const PbvRec * r = pbv_rec_at(oi);
    return r != nullptr && r->issued != 0;
}

// A0-3 **admission commit** — `rec_admitted+1` · 첫 wave 의 `W_l` child credit 원자 선점
//   (②축 `reserved` 로 즉시 가시) · reservation 생성. 호출자는 claim 을 마친 뒤에 부른다.
void pbv_rec_admit_locked(int32_t oi, int32_t mlpos) {
    PbvRec * r = pbv_rec_at(oi);
    if (r == nullptr) return;
    // ★★R-1 — 구 판은 여기서 `W != P` 를 fatal 로 막았다(wave 미구현의 자기 가드).
    //   A0-2 wave 기계가 서면서 그 가드는 **철거**된다: `W < P` 는 이제 정상 형상이고
    //   admission 은 사양대로 **첫 wave 의 `W_l` 만** 선점한다(후속 wave 는 개시 시 확보).
    const int32_t w = pbv_wave_width(mlpos);
    *r = PbvRec{};
    r->active   = 1;
    r->reserved = w;
    r->wave     = w;
    g_mod.pbv_rec_active++;
    g_mod.pbv_child_reserved += w;
    g_mod.pbv_credit_reserved_total += (uint64_t)w;   // ★S-1
}

// A0-3 **첫 child submit** — reservation→inflight 전환(child 단위 reserved−1·submitted+1 은
//   ②축 증가가 담당하고, 여기서는 record 축의 예약 해제·`rec_inflight+1` 을 한다).
//   ★한 record 의 child 는 같은 자리에서 한꺼번에 나가므로 이 전환도 1회다(A0-2 wave).
void pbv_rec_note_first_submit_locked(int32_t oi) {
    PbvRec * r = pbv_rec_at(oi);
    if (r == nullptr || !r->active || r->submitted) return;
    r->submitted = 1;
    g_mod.pbv_child_reserved -= r->reserved;
    if (g_mod.pbv_child_reserved < 0) g_mod.pbv_child_reserved = 0;
    g_mod.pbv_credit_unreserved_total += (uint64_t)r->reserved;   // ★S-1(전환도 예약 해소다)
    r->reserved = 0;
    g_mod.pbv_rec_inflight++;
}

// A0-3 **단일 record terminal** — `rec_inflight−1`(never-submitted 면 reserved 해제만) ·
//   ★credit 반환 = **현재 실제 보유한 reserved 만**(미개시 후속 wave 분은 애초 미획득 —
//   과반환 금지[r3 정정]). op 이 pending 풀로 돌아가는 **모든** 경로가 여기를 통과한다
//   (`pio_release_locked` 단일 초크포인트)라 취소·abort·retry 이전에서도 누수가 없다.
void pbv_rec_release_locked(int32_t oi) {
    PbvRec * r = pbv_rec_at(oi);
    if (r == nullptr || !r->active) return;
    if (r->reserved > 0) {
        g_mod.pbv_child_reserved -= r->reserved;
        if (g_mod.pbv_child_reserved < 0) g_mod.pbv_child_reserved = 0;
        g_mod.pbv_credit_unreserved_total += (uint64_t)r->reserved;   // ★S-1(never-issued 반환)
    }
    if (r->submitted && g_mod.pbv_rec_inflight > 0) g_mod.pbv_rec_inflight--;
    if (g_mod.pbv_rec_active > 0) g_mod.pbv_rec_active--;
    *r = PbvRec{};
}

// ★★PB virtual A3 — ③ 논리 disposition 축의 mode-neutral latch(§2-0 A0-1 ③ · A0-3).
//   A0-3 `issued ⇔ 성공 제출 child ≥ 1`. bin 은 `pb_submit_event_seq != NA` 로 같은 사실을
//   읽었지만 그 필드는 **PB physical 계측**(`pb_note_submit_locked`)이 발급하는 것이고,
//   §3-5 는 virtual 에서 그 호출을 금지한다 ⇒ 판정원을 이 래치로 옮긴다.
//   "성공 제출" = `ReadFile` 이 TRUE 또는 `ERROR_IO_PENDING` 을 돌려준 child 가 1건 이상.
inline void pbv_note_child_issued_locked(int32_t oi) {
    PbvRec * r = pbv_rec_at(oi);
    if (r == nullptr || !r->active) return;
    r->issued = 1;
}

// ★r1 M3 ② — 보류된 ADMISSION_DECISION 을 **최종 disposition 확정 시점**에 1건만 방출한다.
//   `d==0xFF` 는 제출에 도달하지 못한 채 종결된 경우(abort cut 등)이며 UNSUBMITTED 비트와 세트다.
void pb_flush_admit_rec_locked(int32_t oi, uint8_t d, uint64_t io_event_seq) {
    PbOpStamp * st = pb_op_stamp_at(oi);
    if (st == nullptr || !st->admit_rec_valid) return;
    st->admit_rec_valid = 0;
    st->admit_rec.disposition  = d;
    st->admit_rec.io_event_seq = io_event_seq;
    if (d == 0xFF) st->admit_rec.flags |= GGML_MOE_PB_L_F_UNSUBMITTED;
    pb_led_emit_locked(st->admit_rec);
}

// ★★PB virtual A3 — `issued` 확정 1회 방출. **첫 성공 child submit 자리**에서 부른다:
//   bin 의 같은 처분(제출 성공 직후 `pb_flush_admit_rec_locked(ISSUED)` +
//   `pb_dispose_rank_locked(ISSUED)`)과 **같은 위치·같은 의미**여서 mode 간 동형이 유지된다.
//   never-issued(`submit_failed`)는 record terminal 이 닫는다 — `disposed` 래치가 둘 사이의
//   배타를 보장한다(A0-3 C3 배타 `admitted = issued + submit_failed`).
//   ★`io_event_seq` 는 NA 다: 그 좌표는 ④PB physical 축의 것이고 virtual 은 §3-5 로
//   `pb_note_submit_locked` 를 호출하지 않는다(없는 값을 지어내지 않는다).
void pbv_flush_issued_locked(int32_t oi) {
    PbvRec * r = pbv_rec_at(oi);
    if (r == nullptr || !r->active || !r->issued || r->disposed) return;
    PendingIo & op = g_mod.pending_ops[(size_t)oi];
    if (op.origin != PIO_ORIGIN_PREFETCH) return;
    r->disposed = 1;
    pb_flush_admit_rec_locked(oi, GGML_MOE_PB_D_ISSUED, GGML_MOE_PHASEB_NA_U64);
    pb_dispose_rank_locked(pb_find_row_locked(op.pb_graph_seq, op.mlpos), op.intent_seq,
                           op.pb_rank, GGML_MOE_PB_D_ISSUED, op.pb_request_epoch);
}


int32_t pio_alloc_locked() {
    if (g_mod.pending_free.empty()) return -1;
    const int32_t i = g_mod.pending_free.back();
    g_mod.pending_free.pop_back();
    pio_reset(g_mod.pending_ops[(size_t)i]);
    pb_op_stamp_reset(i);   // ★5c-T — 고정 인덱스 재사용이므로 스탬프도 함께 초기화(누수 금지)
    // ★★PB virtual A1 — 같은 사유의 ①축 초기화. gauge 는 건드리지 않는다(반납은 release 가 했다).
    if (PbvRec * pr = pbv_rec_at(i)) *pr = PbvRec{};
    g_mod.pending_ops[(size_t)i].in_use = true;
    return i;
}

void pio_release_locked(int32_t i) {
    // ★r1 M3 ② — 제출에 도달하지 못한 채 op 이 사라지는 경로(abort cut 의 미제출 claimed op)
    //   에서도 보류 레코드를 잃지 않는다. disposition 미확정(0xFF) + UNSUBMITTED 비트로 남겨
    //   offline 이 "admit 됐으나 제출 전 종결"을 판별할 수 있게 한다.
    pb_flush_admit_rec_locked(i, 0xFF, GGML_MOE_PHASEB_NA_U64);
    // ★★PB virtual A1 — ①record 운영축의 **단일 초크포인트**. op 이 풀로 돌아오는 모든 경로
    //   (성공/오류/취소/abort/joined retry 이전)가 여기를 지나므로 gauge·credit 누수가 0 이다.
    pbv_rec_release_locked(i);
    // ★A0-4 wake 결과 계약 ③ — op 이 풀로 돌아오는 것은 **credit·pending 풀·waiter 집합이
    //   전부 변하는** 사건이다(prefetch record 의 예약 반환 · demand 로 막혀 있던 predicate 의
    //   해제 · `disp_can_plan_demand` 여유 회복). drain-defer park 를 내리는 두 지점 중 하나.
    g_mod.pbv_admit_parked = false;
    pio_reset(g_mod.pending_ops[(size_t)i]);
    pb_op_stamp_reset(i);
    g_mod.pending_free.push_back(i);
}

// ★5c-T(부록 14.1·14.3 ④) — issued op 1건의 **배타** 분할 확정. 같은 op 에 두 번 전이하면
//   래치가 막고(oracle ⓒ "중복 전이=fail"의 실장) 카운터도 늘지 않는다. census cut 이후에는
//   어떤 재계상도 하지 않는다(controller §2 · 부록 14.5).
//   반환 true = 이 호출이 분할을 확정했다.
bool pb_partition_op_locked(int32_t oi, uint8_t partition) {
    if (!g_mod.pb_collect) return false;
    PbOpStamp * st = pb_op_stamp_at(oi);
    if (st == nullptr || st->partitioned) return false;
    st->partitioned = 1;
    st->partition   = partition;
    return true;
}

// terminal 1건(SPEC_PHASEB_WIRE §5.2 닫힌 enum).
//   1 SUCCESS_PUBLISHED   = lease 반영→publish→dependency 감소→signal
//   2 ERROR_UNPUBLISHED   = 오류/부분 read(§5 — speculative=fail-soft unpublish / joined=단일 retry)
//   3 CANCELLED_UNPUBLISHED / 4 CANCEL_SUCCESS_UNPUBLISHED = shutdown·abort 경로(★publish 금지)
// R-B5a facade (i): K2 metadata claim / publish / release / terminal.
// The dispatch path states its intent through these members instead of mutating the slot and
// record tables directly. Each performs exactly the statements it replaced, in the same order,
// with the caller's `g_mod.mutex` held. `pb_slot_stamp_clear` is storage-owned and is CALLED
// here, never moved, which section 2-6's forbidden lines require.
inline void k2fac_slot_touch_tick_locked(int32_t slot, uint64_t tick) {
    g_mod.slots[(size_t)slot].last_used = tick;
}

inline uint64_t k2fac_slot_generation_bump_locked(int32_t slot) {
    return g_mod.slots[(size_t)slot].generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline int32_t k2fac_slot_lease_return_locked(int32_t slot) {
    const int32_t prev = g_mod.slots[(size_t)slot].lease.fetch_sub(1, std::memory_order_relaxed);
    if (prev <= 0) g_mod.slots[(size_t)slot].lease.fetch_add(1, std::memory_order_relaxed);
    return prev;
}

inline void k2fac_record_slot_detach_locked(int64_t record) {
    g_mod.record_table[(size_t)record].slot_index.store(-1, std::memory_order_release);
}

inline void k2fac_slot_unpublish_locked(int32_t slot) {
    Slot & t = g_mod.slots[(size_t)slot];
    t.cache_class       = GGML_MOE_PB_CLASS_NONE;
    t.pf_intent_ordinal = GGML_MOE_PHASEB_NA_U64;
    t.pf_rank           = GGML_MOE_PHASEB_NA_U16;
    t.pf_io_index       = -1;
    pb_slot_stamp_clear(t);
}

inline void k2fac_slot_release_owner_locked(int32_t slot, int64_t record_idx) {
    Slot & t = g_mod.slots[(size_t)slot];
    if (t.owner_record == record_idx) t.owner_record = -1;
    k2fac_slot_unpublish_locked(slot);
}

inline void k2fac_slot_claim_victim_locked(int32_t slot, int64_t record, uint64_t tick) {
    Slot & t = g_mod.slots[(size_t)slot];
    t.owner_record       = record;
    t.last_used          = tick;
    k2fac_slot_unpublish_locked(slot);
}

inline void k2fac_slot_publish_probation_locked(int32_t slot, const PendingIo & op, int32_t oi) {
    Slot & t = g_mod.slots[(size_t)slot];
    t.cache_class       = GGML_MOE_PB_CLASS_PROBATION;
    t.pf_intent_ordinal = op.intent_seq;
    t.pf_rank           = op.pb_rank;
    t.pf_io_index       = op.pb_io_index;
    t.pf_request_epoch  = op.pb_request_epoch;
    t.pf_graph_seq      = op.pb_graph_seq;
    if (const PbOpStamp * st = pb_op_stamp_at(oi)) {
        t.pf_stamp_k = st->k; t.pf_stamp_n = st->n;
        t.pf_stamp_qd = st->qd; t.pf_stamp_rev = st->rev;
        t.pf_gen_admission_seq = st->gen_admission_seq;
    }
}

void disp_op_terminal_locked(int32_t oi, DWORD bytes, bool was_pending, uint8_t status,
                             uint32_t want_override = 0) {
    PendingIo & op = g_mod.pending_ops[(size_t)oi];
    op.state = PIO_TERMINAL;
    // ★테스트 전용 one-shot — prefetch 성공 terminal 1건을 부분 read/오류로 강등(§5 픽스처 ④).
    if (g_mod.test_pb_error_next && op.origin == PIO_ORIGIN_PREFETCH &&
        status == GGML_MOE_PB_T_SUCCESS_PUBLISHED) {
        g_mod.test_pb_error_next = 0;
        status = GGML_MOE_PB_T_ERROR_UNPUBLISHED;
    }
    const bool ok         = (status == GGML_MOE_PB_T_SUCCESS_PUBLISHED);
    const bool had_waiter = !op.waiters.empty();
    const bool prefetch   = (op.origin == PIO_ORIGIN_PREFETCH);
    // ★★PB virtual A3 — 축 분리(§2-0 A0-1). 구 판은 `submitted` 하나가 ④PB physical 계측과
    //   ③논리 disposition 을 겸했는데, virtual 은 §3-5 로 `pb_note_*` 를 호출하지 않으므로
    //   `pb_submit_event_seq` 가 영영 NA 다 ⇒ 그 하나로 ③축을 판정하면 virtual 의 issued /
    //   4분할 / OP_TERMINAL ledger 가 통째로 사라진다(계측 축이 논리 축을 삼키는 결함).
    //     `pb_submitted` = ④축(PB physical wire 가 이 op 을 기록했나 — **bin 전용**)
    //     `issued`       = ③축(A0-3 `성공 제출 child ≥ 1` — mode-neutral)
    const bool pb_submitted = (op.pb_submit_event_seq != GGML_MOE_PHASEB_NA_U64);
    const bool issued       = pb_submitted || pbv_rec_issued(oi);

    // ── in-flight 회계 + terminal 사건 순번(§5.1 공용 namespace) ──
    //   ★publish_seq 는 성공 terminal에서만 발급되므로 여기서 미리 정한다(core에 전달).
    const uint64_t pseq = ok ? (++g_mod.pb_publish_seq) : (uint64_t)0;
    if (pb_submitted) {
        // ★§3-5 동결 처분 — virtual child 는 `pb_note_submit/terminal` 을 **둘 다** 호출하지
        //   않는다. 그 계약이 `pb_submitted` 게이트로 그대로 유지된다(virtual = 항상 false).
        pb_note_terminal_locked(op.pb_io_index, op.origin, status, had_waiter,
                                op.pb_ever_joined, pseq,
                                op.pb_request_epoch, op.pb_graph_seq, op.intent_seq,
                                op.pb_rank, op.m.record_idx, pb_op_stamp_at(oi));
    }

    // ── ★★PB virtual A3 — ③축 rank disposition 의 terminal 확정(§2-0 A0-3) ──
    //   terminal 은 `issued` / `submit_failed` **상태만** 확정한다(used_join·used_promoted·
    //   wasted 는 각자의 사건 시점·finalize 잔차 — WIRE :436 계약 그대로).
    //   ISSUED 는 첫 성공 child submit 자리에서 이미 방출됐다(bin 과 동형) — 여기서는
    //   **never-issued** 만 닫는다. 사유 서브코드(fail/cancel/shutdown)는 바로 아래
    //   OP_TERMINAL ledger 의 `terminal_status` 가 같은 op 에 병기한다.
    if (PbvRec * pvr = pbv_rec_at(oi)) {
        if (pvr->active && prefetch && !pvr->disposed && !pvr->issued) {
            pvr->disposed = 1;
            pb_flush_admit_rec_locked(oi, GGML_MOE_PB_D_SUBMIT_FAILED, GGML_MOE_PHASEB_NA_U64);
            pb_dispose_rank_locked(pb_find_row_locked(op.pb_graph_seq, op.mlpos), op.intent_seq,
                                   op.pb_rank, GGML_MOE_PB_D_SUBMIT_FAILED, op.pb_request_epoch);
        }
    }

    // ── ★5c-T 종결 4분할(부록 14.1 · controller §2 분류 원리) ─────────────────────────────
    //   issued(= prefetch ∧ submitted) op 1건은 정확히 한 번만 전이한다.
    //     ever_joined      → used_join      (계수는 join 시점 · 여기서는 분할 래치만)
    //     ¬joined ∧ 성공    → READY_PROBATION 로 **유예**(승격/축출/census 가 확정)
    //     ¬joined ∧ 비성공  → resolved_wasted(오류 unpublish · 취소 계열 전건 = 사용 불가 확정)
    //   ★census cut 이후에는 어떤 재계상도 하지 않는다(부록 14.5).
    //   ★★PB virtual A3 — 게이트를 ④축(`pb_submitted`)에서 ③축(`issued`)으로 옮겼다.
    //     bin 에서는 두 값이 항상 같으므로 **거동 불변**이고, virtual 에서는 이 전환이
    //     4분할·OP_TERMINAL ledger 를 살린다.
    if (g_mod.pb_collect && prefetch && issued && !g_mod.pb_censused) {
        const uint8_t part = op.pb_ever_joined ? (uint8_t)GGML_MOE_PB_P_USED_JOIN
                           : (ok ? (uint8_t)GGML_MOE_PB_P_NA
                                 : (uint8_t)GGML_MOE_PB_P_RESOLVED_WASTED);
        const bool latched = (part != GGML_MOE_PB_P_NA) && pb_partition_op_locked(oi, part);
        if (latched && part == GGML_MOE_PB_P_RESOLVED_WASTED)
            ggml_moe_phaseb_count(GGML_MOE_PB_C_RESOLVED_WASTED, op.pb_request_epoch, 1);
        ggml_moe_phaseb_ledger lr = pb_led_new(GGML_MOE_PB_L_OP_TERMINAL);
        lr.io_event_seq    = op.pb_submit_event_seq;
        lr.request_epoch   = op.pb_request_epoch;
        lr.graph_seq       = op.pb_graph_seq;
        lr.graph_epoch     = op.pb_graph_epoch;
        lr.intent_ordinal  = op.intent_seq;
        lr.rank            = op.pb_rank;
        lr.record_idx      = op.m.record_idx;
        lr.origin          = 1;
        lr.terminal_status = status;
        lr.partition       = part;
        if (op.pb_ever_joined) lr.flags |= GGML_MOE_PB_L_F_EVER_JOINED;
        if (had_waiter)        lr.flags |= GGML_MOE_PB_L_F_HAD_WAITER;
        pb_led_stamp_from_op(lr, pb_op_stamp_at(oi));
        pb_led_emit_locked(lr);

        // ★r2 M7-1 — **중복 terminal 전이 프로브**(테스트 전용 one-shot · 프로덕션 0).
        //   같은 live op 에 대해 4분할 전이를 **한 번 더** 시도하고, 그 2차 시도가 카운터·QD·
        //   ledger 를 하나도 늘리지 않았음을 실측해 남긴다. finalize 2차 호출 같은 우회가
        //   아니라 배타 래치(`pb_partition_op_locked`)가 방어하는 바로 그 경로를 친다.
        if (ggml_moe_phaseb_test_dup_terminal_take()) {
            const uint64_t c0 = ggml_moe_phaseb_counter(GGML_MOE_PB_C_RESOLVED_WASTED) +
                                ggml_moe_phaseb_counter(GGML_MOE_PB_C_CENSORED) +
                                ggml_moe_phaseb_counter(GGML_MOE_PB_C_USED_JOIN) +
                                ggml_moe_phaseb_counter(GGML_MOE_PB_C_USED_PROMOTED);
            const uint64_t l0 = ggml_moe_phaseb_ledger_records();
            const int32_t  q0 = g_mod.pb_inflight_demand + g_mod.pb_inflight_prefetch;
            // 2차 전이 시도 — 래치가 서 있으면 분할도 카운터도 레코드도 늘지 않아야 한다.
            const bool latched2 = (part != GGML_MOE_PB_P_NA) && pb_partition_op_locked(oi, part);
            if (latched2 && part == GGML_MOE_PB_P_RESOLVED_WASTED)
                ggml_moe_phaseb_count(GGML_MOE_PB_C_RESOLVED_WASTED, op.pb_request_epoch, 1);
            if (latched2) {
                ggml_moe_phaseb_ledger dr = pb_led_new(GGML_MOE_PB_L_OP_TERMINAL);
                dr.request_epoch  = op.pb_request_epoch;
                dr.intent_ordinal = op.intent_seq;
                dr.rank           = op.pb_rank;
                dr.partition      = part;
                pb_led_stamp_from_op(dr, pb_op_stamp_at(oi));
                pb_led_emit_locked(dr);
            }
            const uint64_t c1 = ggml_moe_phaseb_counter(GGML_MOE_PB_C_RESOLVED_WASTED) +
                                ggml_moe_phaseb_counter(GGML_MOE_PB_C_CENSORED) +
                                ggml_moe_phaseb_counter(GGML_MOE_PB_C_USED_JOIN) +
                                ggml_moe_phaseb_counter(GGML_MOE_PB_C_USED_PROMOTED);
            const uint64_t l1 = ggml_moe_phaseb_ledger_records();
            const int32_t  q1 = g_mod.pb_inflight_demand + g_mod.pb_inflight_prefetch;
            ggml_moe_phaseb_test_dup_terminal_note((int32_t)status, c1 - c0, l1 - l0, q1 - q0);
        }
    }

    if (ok) {
        io_finish_locked(op.m, bytes, was_pending, op.mlpos, op.phase, op.origin, want_override);
        // ★R1 (a) 선행 의무(HANDOFF durable) — persistent join이 갱신한 op tick을 슬롯에 재반영한다.
        //   join은 C2에 따라 slot lease/last_used를 건드리지 않으므로, 여기서 반영하지 않으면
        //   join된 record의 LRU가 claim 시점 tick에 고정돼 victim 순서가 뒤집힌다.
        k2fac_slot_touch_tick_locked(op.m.slot, op.m.logical_last_tick);
        if (had_waiter) {
            // waiter 有 publish → READY_DEMAND(io_finish_locked 기본값 그대로).
            pb_transition_locked(GGML_MOE_PB_EV_PUBLISH_DEMAND, op.m.record_idx, op.m.slot,
                                 op.m.generation, op.intent_seq);
        } else {
            // waiter 無 publish → READY_PROBATION(§4 상태기계).
            // ★5c-T §1 — owner 스탬프를 READY slot 까지 전파(§6 참여 집합 귀속·census owner ref).
            k2fac_slot_publish_probation_locked(op.m.slot, op, oi);
            pb_transition_locked(GGML_MOE_PB_EV_PUBLISH_PROBATION, op.m.record_idx, op.m.slot,
                                 op.m.generation, op.intent_seq);
        }
    } else {
        if (status != GGML_MOE_PB_T_ERROR_UNPUBLISHED) g_mod.stat_abort_terminal++;
        if (op.origin == PIO_ORIGIN_DEMAND && ggml_moe_trace_enabled()) {
            ggml_moe_trace_io_complete(op.m.trace_op_id, MOE_TRACE_IO_STATUS_ERROR, (uint32_t)bytes,
                                       was_pending, (uint32_t)op.m.slot);
            ggml_moe_trace_data_wait_end(op.m.trace_wait_episode);
        }
        // ★joined **prefetch**(waiter 있음) 오류/부분 read = **단 하나의 새 generation retry**
        //   (§5·C2). waiter·multiplicity·마지막 demand_tick을 이전하고 구 슬롯 lease는 0을 유지한다
        //   (lease는 성공 terminal에서만 반영되므로 구조적으로 0이다). retry의 retry는 없다.
        //
        // ★★origin 구분(§Z-⑥ A3 재개봉 R1 — SPEC_REPACK_V3 §3-4 3분류): 이 게이트는
        //   **joined prefetch 전용**이다. 사양은 실패를 origin 으로 3분한다 —
        //     ⓐ direct demand      = 부분 publish 0 + **fail-closed(fatal)**
        //     ⓑ waiterless prefetch = 부분 publish 0 + ABSENT/fail-soft
        //     ⓒ joined prefetch     = 실패 op publish 0 + 새 generation retry 1회(재실패 fail-stop)
        //   구 조건은 `had_waiter && !pb_is_retry` 뿐이라 ⓐ까지 retry 로 삼켰다.
        // ★**bin 거동 불변 증명**: bin 에서 `ERROR_UNPUBLISHED` 생산자는 **전부 prefetch 게이트
        //   아래**다(테스트 훅 `op.origin==PREFETCH` · pump sync `(bytes_ok||!prefetch)` ·
        //   retire `(prefetch && !bytes_ok)`). bin 의 demand 실패는 그 앞에서 이미 fatal 이다
        //   (`GetOverlappedResult` 실패 시 `!gok && !prefetch` → fatal_abort · short read 는
        //   `io_finish_locked` 의 want_bytes 대조 → fatal_abort). ⇒ bin 에서는 이 조건이 **한 번도
        //   demand 로 도달하지 않으므로 origin 절 추가는 no-op** 이다. demand-origin ERROR 를
        //   실제로 만들기 시작한 것은 A3 의 virtual child 경로(`v3_child_retire_locked`)다.
        if (status == GGML_MOE_PB_T_ERROR_UNPUBLISHED && had_waiter && !op.pb_is_retry &&
            op.origin == PIO_ORIGIN_PREFETCH) {
            const int32_t ri = pio_alloc_locked();
            // ★R1 #9 ① — retry 슬롯 확보 실패는 fail-soft가 아니다(§5 "retry 실패 시에만 기존
            //   fail-stop"): waiter가 붙은 오류를 조용히 error 종결하면 계약 위반이다.
            if (ri < 0) fatal_abort("cannot reserve a pending slot to retry a joined error (fail-stop) record=%lld",
                                    (long long)op.m.record_idx);
            {
                PendingIo & r = g_mod.pending_ops[(size_t)ri];
                r.m       = op.m;
                r.m.generation = k2fac_slot_generation_bump_locked(op.m.slot);
                r.mlpos   = op.mlpos;
                r.phase   = op.phase;
                r.origin  = PIO_ORIGIN_DEMAND;   // ★최종 3분할상 demand-read(R1 #7)
                r.state   = PIO_CLAIMED;
                r.channel = -1;
                r.request_identity = op.request_identity;
                // ★R1 #9 ③ — retry는 demand-origin 사건이므로 intent 상관자는 NA다.
                r.intent_seq       = GGML_MOE_PHASEB_NA_U64;
                r.retry_of         = oi;
                r.waiters          = op.waiters;
                // ★R1 #9 ② — owner epoch = **첫 waiter의 demand epoch**(원 prefetch 값 복사 금지).
                r.pb_request_epoch = op.pb_join_request_epoch != GGML_MOE_PHASEB_NA_U32
                                     ? op.pb_join_request_epoch : op.pb_request_epoch;
                r.pb_graph_epoch   = op.pb_join_graph_epoch != GGML_MOE_PHASEB_NA_U32
                                     ? op.pb_join_graph_epoch : op.pb_graph_epoch;
                r.pb_graph_seq     = op.pb_join_graph_seq != GGML_MOE_PHASEB_NA_U32
                                     ? op.pb_join_graph_seq : op.pb_graph_seq;
                r.pb_rank          = GGML_MOE_PHASEB_NA_U16;
                r.pb_is_retry      = true;
                r.pb_retry_of_event_seq = op.pb_submit_event_seq;
                r.pb_ever_joined   = op.pb_ever_joined;
                k2fac_slot_unpublish_locked(op.m.slot);   // ★5c-T §1
                g_mod.pending_by_record[(size_t)op.m.record_idx] = ri;
                g_mod.submit_q.push_back(ri);
                // 3분할 최종 처분 이동: join 실패 후 retry = demand_read(§6.1).
                for (int32_t ci : op.waiters) {
                    ControlCmd & c = g_mod.cmds[(size_t)ci];
                    uint32_t moved = 0;
                    for (size_t k = 0; k < c.req_join_op.size(); k++)
                        if (c.req_join_op[k] == oi) { c.req_join_op[k] = -1; moved++; }
                    if (moved > c.sp_join) moved = c.sp_join;
                    c.sp_join -= moved;
                    c.sp_demand_read += moved;
                }
                // ★R1 #9 ④ — PA batch_order 는 첫 waiter command 의 별도 순번(원 op 값 재사용 금지).
                if (op.pb_join_cmd >= 0 && op.pb_join_cmd < MOE_DIRECT_CONTROL_Q_CAP) {
                    ControlCmd & jc = g_mod.cmds[(size_t)op.pb_join_cmd];
                    r.m.pa_batch_order = jc.pa_batch_next++;
                    // ★R1 #9 ⑤ — PA events 의 n_miss/miss_bytes 는 "이 acquire 가 유발한 demand
                    //   read op 수"여야 PA ops 레코드와 1:1이 유지된다(SPEC_PHASEB_WIRE §11-a가
                    //   retry 포함 전건 1:1을 요구하고, retry 는 io_finish 에서 PA ops 를 낸다).
                    //   따라서 retry 1건을 n_miss/bytes 에 가산한다.
                    // ★PA n_miss 의미(R2 권고 2): "이 acquire 가 유발한 demand read op 수".
                    //   joined-success = 0 가산(물리 제출이 없다) · 첫 waiter 소유 retry = 1 가산.
                    jc.pa_n_miss += 1;
                    jc.pa_bytes  += g_mod.layers[(size_t)op.mlpos].stride_bytes;
                }
                ggml_moe_phaseb_count(GGML_MOE_PB_C_PREFETCH_RETRY, r.pb_request_epoch, 1);
                pb_transition_locked(GGML_MOE_PB_EV_RETRY_ISSUE, r.m.record_idx, r.m.slot,
                                     r.m.generation, GGML_MOE_PHASEB_NA_U64);
                pio_release_locked(oi);
                return;   // ★waiter는 깨우지 않는다(retry가 결과를 공유)
            }
        }
        // ★★§3-4 ⓐ·ⓒ 후단(A3 재개봉 R1) — **demand 실패는 fail-closed(fatal)**.
        //   ⓐ direct demand 오류가 여기 닿았다는 것은 요청자가 기다리는 데이터를 못 준다는 뜻이고,
        //     사양은 이를 조용한 -1 종결이 아니라 fatal 로 규정한다.
        //   ⓒ joined-prefetch **retry 의 재실패**도 여기로 온다 — retry op 은 origin 을
        //     `PIO_ORIGIN_DEMAND` 로 승격해 만들어지고(위 retry 블록) `pb_is_retry` 라 게이트를
        //     건너뛰므로, "재실패 = fail-stop" 이 이 한 줄로 함께 성립한다.
        //   ★취소 계열(CANCELLED/CANCEL_SUCCESS)은 정상 종료 경로라 해당 없다 — ERROR 만 본다.
        //   ★bin 무영향: 위 게이트 주석의 열거대로 bin 은 demand-origin ERROR 를 생산하지 않는다.
        if (status == GGML_MOE_PB_T_ERROR_UNPUBLISHED && op.origin == PIO_ORIGIN_DEMAND) {
            fatal_abort("demand read failed (fail-closed per spec 3-4): record=%lld slot=%d bytes=%lu%s",
                        (long long)op.m.record_idx, (int)op.m.slot, (unsigned long)bytes,
                        op.pb_is_retry ? " [retry re-failure = fail-stop]" : "");
        }
        // 슬롯을 ABSENT로 되돌린다(구 owner는 claim에서 이미 unpublish·새 record는 미publish).
        k2fac_slot_release_owner_locked(op.m.slot, op.m.record_idx);   // ★5c-T §1 (terminal)
        if (status == GGML_MOE_PB_T_ERROR_UNPUBLISHED) {
            if (prefetch) ggml_moe_phaseb_count(GGML_MOE_PB_C_PREFETCH_ERROR_UNPUBLISH, op.pb_request_epoch, 1);
            pb_transition_locked(GGML_MOE_PB_EV_UNPUBLISH_ERROR, op.m.record_idx, op.m.slot,
                                 op.m.generation, op.intent_seq);
        } else {
            pb_transition_locked(GGML_MOE_PB_EV_CANCEL, op.m.record_idx, op.m.slot,
                                 op.m.generation, op.intent_seq);
        }
    }
    // ★cancel 완료 확인 후에만 pin 해제(slot 재사용 금지 구간의 끝).
    if (op.m.slot >= 0 && op.m.slot < (int32_t)g_mod.slot_pending.size()) g_mod.slot_pending[(size_t)op.m.slot] = 0;
    if (op.m.record_idx >= 0 && op.m.record_idx < (int64_t)g_mod.pending_by_record.size() &&
        g_mod.pending_by_record[(size_t)op.m.record_idx] == oi) {
        g_mod.pending_by_record[(size_t)op.m.record_idx] = -1;
    }
    // dependency 감소 → 전 required가 준비된 command만 signal.
    for (int32_t ci : op.waiters) {
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        if (!ok) {
            if (c.deps == 0) continue; // 이미 종결된 command(다른 op의 abort로 error 처리됨)
            if (c.pa_open) { ggml_moe_phasea_acquire_exit(c.pa_n_miss, c.pa_bytes, ggml_moe_phasea_now_qpc()); c.pa_open = false; }
            c.deps = 0;
            pb_finish_demand_command_locked(c);
            g_mod.demand_outstanding--;
            disp_complete(ci, -1);
            continue;
        }
        if (--c.deps == 0) {
            // ★Phase A R_j — demand record 전부 host 가용.
            if (c.pa_open) { ggml_moe_phasea_acquire_exit(c.pa_n_miss, c.pa_bytes, ggml_moe_phasea_now_qpc()); c.pa_open = false; }
            // ★§7.5 predictor 노출 조인 재료 — 이 층의 demand R 시각.
            ggml_moe_phaseb_note_demand_ready(c.pb_graph_seq,
                                              c.mlpos >= 0 && c.mlpos < (int32_t)g_mod.moe_layers.size()
                                                  ? g_mod.moe_layers[(size_t)c.mlpos] : -1,
                                              ggml_moe_phaseb_now_qpc());
            pb_finish_demand_command_locked(c);
            g_mod.demand_outstanding--;
            metrics_maybe_periodic_locked();
            disp_complete(ci, 0);
        }
    }
    pio_release_locked(oi);
}

// ★B2 — 성공 제출 1건의 계측(§5·§5.1·§6·§8). io_submit_locked가 성공(pending/sync 무관)한
//   직후에만 호출된다. SUBMIT_FAILED는 여기 오지 않는다(pb_io 미기록 계약).
void pb_on_submit_locked(PendingIo & op, int32_t oi) {
    // ★R1 #6 ① — 계측 본문은 공용 core(pb_note_submit_locked) 하나뿐이다.
    op.pb_io_index = pb_note_submit_locked(
        op.m.record_idx, op.mlpos, op.m.file_offset, op.m.slot, op.m.generation, op.origin,
        op.pb_rank, op.intent_seq, op.pb_request_epoch, op.pb_graph_seq,
        op.phase == Phase::PREFILL, op.pb_is_retry, op.pb_retry_of_event_seq,
        &op.pb_submit_event_seq, pb_op_stamp_at(oi));   // ★r1 M2 owner 스탬프
}

// rank 처분 기록(§4.2 · §8 disposition digest · §6 카운터) — 사전 처분/admission/issued 공통.
void pb_dispose_rank_locked(int32_t row, uint64_t intent_ordinal, uint16_t rank, uint8_t disp,
                            uint32_t request_epoch) {
    if (row >= 0 && row < (int32_t)g_mod.pb_samples.size() && rank < GGML_MOE_PHASEB_TOPK)
        g_mod.pb_samples[(size_t)row].rank_disposition[rank] = disp;
    ggml_moe_phaseb_digest_disposition(intent_ordinal, rank, disp);
    static const int32_t kCtr[10] = {
        -1,
        GGML_MOE_PB_C_PRE_DROPPED_TOO_LATE, GGML_MOE_PB_C_PRE_DROPPED_STALE,
        GGML_MOE_PB_C_PRE_SKIPPED_RESIDENT, GGML_MOE_PB_C_PRE_SKIPPED_PENDING,
        GGML_MOE_PB_C_DROPPED_NO_SLOT_QD,   GGML_MOE_PB_C_DROPPED_PREFETCH_CAP,
        GGML_MOE_PB_C_DROPPED_RESERVE,      GGML_MOE_PB_C_ISSUED, GGML_MOE_PB_C_SUBMIT_FAILED,
    };
    if (disp >= 1 && disp <= 9 && kCtr[disp] >= 0)
        ggml_moe_phaseb_count(kCtr[disp], request_epoch, 1);
}

// R-B5a facade (ii): K2 source and payload handle capability (RB_DESIGN.md:1221 section 2-6
// (ii)). The dispatch cluster reaches a VIRTUAL source handle, the handle COUNT and the BIN
// payload handle through these three members instead of touching `g_mod.source_files` and
// `g_mod.data_file` directly. Each hands out the VALUE - never a reference and never an address
// of module state - so the capability is a read capability and cannot become a write channel.
// The caller's `g_mod.mutex` is already held, as it was at every site these replace.
inline size_t k2fac_source_count_locked() {
    return g_mod.source_files.size();
}

inline HANDLE k2fac_source_handle_locked(size_t i) {
    return g_mod.source_files[i];
}

inline HANDLE k2fac_data_handle_locked() {
    return g_mod.data_file;
}

// aligned(§2-4 정렬 상수 증명):
//   read_offset = abs + e*slice − h · read_len = region · dest = slot_base + slot_offset
//     (실데이터는 dest + bracket_head = slot_base + data_offset 에 놓인다)
// 비4K(A4 · §3-5 옵션 B):
//   src_head(e) = (abs + e*slice) mod A · read_offset = align_down(abs + e*slice, A)
//   read_len    = align_up(src_head + slice, A)   (≤ staging_bytes — §2-4 가 보장)
//   dest        = staging · 최종 착지는 copy 가 slot_base + slot_offset(= data_offset)로
bool v3_child_geom_locked(int32_t mlpos, int32_t expert, int32_t pidx, int32_t slot,
                          int32_t staging_idx, V3ChildGeom & out) {
    const PartInfo & p = g_mod.layers[(size_t)mlpos].parts[(size_t)pidx];
    if (p.source_index < 0 || (size_t)p.source_index >= k2fac_source_count_locked()) return false;
    out.h = k2fac_source_handle_locked((size_t)p.source_index);
    char * const slot_dest = (char *)g_mod.slots[(size_t)slot].base + p.slot_offset;
    if (p.aligned) {
        out.offset = p.abs_offset + (uint64_t)expert * p.slice_bytes - p.bracket_head;
        out.len    = (uint32_t)p.region;
        out.dest   = slot_dest;
        out.bounce = false;
        return true;
    }
    if (staging_idx < 0 || staging_idx >= g_mod.v3_staging_count || !g_mod.v3_staging_base) return false;
    const uint64_t A   = g_mod.align_bytes;
    const uint64_t src = p.abs_offset + (uint64_t)expert * p.slice_bytes;
    out.src_head  = src % A;
    out.offset    = src - out.src_head;                       // = align_down(src, A)
    out.len       = (uint32_t)align_up(out.src_head + p.slice_bytes, A);
    out.staging   = staging_idx;
    out.dest      = (char *)g_mod.v3_staging_base + (size_t)staging_idx * g_mod.v3_staging_stride;
    out.slot_dest = slot_dest;
    out.bounce    = true;
    if ((uint64_t)out.len > (uint64_t)g_mod.v3_staging_stride) return false;   // §2-4 staging_bytes 계약
    return true;
}

// staging 1칸 확보(없으면 -1 = backpressure). ★copy terminal 까지 반납하지 않는다(§3-5).
int32_t v3_staging_take_locked() {
    for (int32_t i = 0; i < g_mod.v3_staging_count; i++) {
        if (!g_mod.v3_staging_busy[(size_t)i]) { g_mod.v3_staging_busy[(size_t)i] = 1; return i; }
    }
    return -1;
}

inline void v3_staging_give_locked(int32_t idx) {
    if (idx >= 0 && idx < g_mod.v3_staging_count) g_mod.v3_staging_busy[(size_t)idx] = 0;
}

// 채널 i 에 걸린 child 가 읽고 있는 shard 핸들. virtual 은 핸들이 여럿이라 CancelIoEx·
// GetOverlappedResult 가 **그 child 의 핸들**을 받아야 한다(bin 의 단일 data_file 과 다른 점).
HANDLE v3_chan_handle_locked(size_t chan) {
    const int32_t oi = g_mod.v3_chan_op[chan];
    const int32_t p  = g_mod.v3_chan_part[chan];
    if (oi < 0 || p < 0) return INVALID_HANDLE_VALUE;
    const PendingIo & op = g_mod.pending_ops[(size_t)oi];
    if (op.mlpos < 0 || p >= v3_parts_of(op.mlpos)) return INVALID_HANDLE_VALUE;
    const int32_t sidx = g_mod.layers[(size_t)op.mlpos].parts[(size_t)p].source_index;
    if (sidx < 0 || (size_t)sidx >= k2fac_source_count_locked()) return INVALID_HANDLE_VALUE;
    return k2fac_source_handle_locked((size_t)sidx);
}

// 이 op 이 아직 커널에 걸어둔 형제 child 를 전부 취소한다(첫 실패 직후 1회).
// ★취소는 "회수"가 아니다 — CancelIoEx 된 child 도 완료 통지가 오고, 그 통지가 remaining 을
//   내린다. 그래서 slot unpin 은 여기가 아니라 remaining==0 시점의 op terminal 이 한다.
void v3_cancel_children_locked(int32_t oi) {
    V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
    if (cs.cancel_issued) return;
    cs.cancel_issued = 1;
    const PendingIo & op = g_mod.pending_ops[(size_t)oi];
    for (size_t i = 0; i < g_mod.channels.size(); i++) {
        if (!g_mod.channels[i].busy) continue;
        // ★A6 — 대구간 채널은 `v3_chan_op` 에 **대표(첫 참가) op** 만 담는다. 그러나 실패한 것이
        //   비대표 참가 record 여도 그 대구간은 이 record 의 형제 I/O 다 — 소유 관계는 참가
        //   집합이 권위다. 대표만 보면 형제 취소가 조용히 빠진다.
        bool serves = (g_mod.v3_chan_op[i] == oi);
        if (!serves && g_mod.v3_chan_coal[i] >= 0) {
            const V3CoalRead & cr = g_mod.v3_coal[(size_t)g_mod.v3_chan_coal[i]];
            for (int32_t o : cr.ops) if (o == oi) { serves = true; break; }
        }
        if (!serves) continue;
        const int32_t pidx = g_mod.v3_chan_part[i];
        if (pidx < 0 || pidx >= v3_parts_of(op.mlpos)) continue;
        // ★CancelIoEx 의 핸들은 **그 child 가 실제로 읽는 shard 핸들**이어야 한다 —
        //   virtual 은 shard 마다 핸들이 다르므로 (op, part) → source_index 로 되짚는다.
        const int32_t sidx = g_mod.layers[(size_t)op.mlpos].parts[(size_t)pidx].source_index;
        if (sidx < 0 || (size_t)sidx >= k2fac_source_count_locked()) continue;
        CancelIoEx(k2fac_source_handle_locked((size_t)sidx), &g_mod.channels[i].ov);
    }
}

bool v3_copy_credit_take_locked(int32_t n) {
    if (n <= 0) return true;
    if (g_mod.v3_copy_cap <= 0) return false;                       // copy worker 자체가 없다
    if (g_mod.v3_copy_outstanding + n > g_mod.v3_copy_cap) return false;
    g_mod.v3_copy_outstanding += n;
    return true;
}

inline void v3_copy_credit_give_locked(int32_t n) {
    g_mod.v3_copy_outstanding -= n;
    if (g_mod.v3_copy_outstanding < 0) g_mod.v3_copy_outstanding = 0;
}

bool v3_copy_enqueue(const ModuleState::V3CopyJob & job) {
    {
        std::lock_guard<std::mutex> lk(g_mod.v3_copy_mutex);
        if (g_mod.v3_copy_cap <= 0 || g_mod.v3_copy_count >= g_mod.v3_copy_cap) return false;
        const int32_t tail = (g_mod.v3_copy_head + g_mod.v3_copy_count) % g_mod.v3_copy_cap;
        g_mod.v3_copy_q[(size_t)tail] = job;
        g_mod.v3_copy_count++;
    }
    if (g_mod.v3_copy_event) SetEvent(g_mod.v3_copy_event);
    return true;
}

unsigned __stdcall v3_copy_worker_main(void *) {
    for (;;) {
        if (!g_mod.v3_copy_running.load(std::memory_order_acquire)) break;
        ModuleState::V3CopyJob job;
        bool have = false;
        {
            std::lock_guard<std::mutex> lk(g_mod.v3_copy_mutex);
            if (g_mod.v3_copy_count > 0 && g_mod.v3_copy_pause.load(std::memory_order_acquire) == 0) {
                job = g_mod.v3_copy_q[(size_t)g_mod.v3_copy_head];
                g_mod.v3_copy_head = (g_mod.v3_copy_head + 1) % g_mod.v3_copy_cap;
                g_mod.v3_copy_count--;
                have = true;
            }
        }
        if (!have) {
            // pause 중에도 정지 신호를 놓치지 않도록 짧게 깬다.
            WaitForSingleObject(g_mod.v3_copy_event, 5);
            continue;
        }
        // ---- ★락 밖에서 memcpy(§3-5 — 이 구간이 dispatcher 를 막지 않는 것이 옵션 B 의 요지) ----
        const uint64_t start_qpc = ggml_moe_phaseb_now_qpc();   // ★A5 — worker 착수 시각
        bool ok = (job.ok != 0);
        // ★A6 — 원본 포인터는 enqueue 시점에 확정돼 있다(바운스 staging · coalesce 버퍼 공통).
        if (ok && job.src) memcpy(job.dest, job.src, (size_t)job.bytes);
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            if (ok) g_mod.v3_copies_done++; else g_mod.v3_copies_failed++;
            v3_copy_complete_locked(job, ok, start_qpc);
        }
        // 대기 중인 demand 가 있으면 dispatcher 가 command 를 닫아야 하므로 깨운다.
        if (g_mod.command_event) SetEvent(g_mod.command_event);
    }
    return 0;
}

bool v3_copy_start_locked(int32_t capacity, std::string & err) {
    // ★★R-2(FROZEN §2-2 "실패 후 자원 잔존 0" · 교차 r1 F1) — **자기 unwind**.
    //   구 판은 queue·cap 을 먼저 만든 뒤 event/thread 생성에 실패하면 그 큐와 `v3_copy_cap`
    //   을 **남긴 채** false 를 돌려줬다. `v3_copy_stop_and_join` 은 `running==false` 면 즉시
    //   반환하고 `virtual_release_locked` 도 큐를 비우지 않으므로 그 잔존은 아무도 안 치운다.
    //   ⇒ 이 함수가 잡은 것은 이 함수가 되돌린다(트랜잭션 경계 = 이 함수).
    auto unwind = [&]() {
        g_mod.v3_copy_running.store(false, std::memory_order_release);
        if (g_mod.v3_copy_event) { CloseHandle(g_mod.v3_copy_event); g_mod.v3_copy_event = nullptr; }
        g_mod.v3_copy_q.clear();
        g_mod.v3_copy_q.shrink_to_fit();
        g_mod.v3_copy_head = g_mod.v3_copy_count = g_mod.v3_copy_cap = 0;
        g_mod.v3_copy_outstanding = 0;
    };
    g_mod.v3_copy_cap   = capacity;
    g_mod.v3_copy_q.assign((size_t)capacity, ModuleState::V3CopyJob{});
    g_mod.v3_copy_head  = 0;
    g_mod.v3_copy_count = 0;
    g_mod.v3_copy_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);   // auto-reset
    if (!g_mod.v3_copy_event) { unwind(); err = "CreateEventW(bounce copy) failed"; return false; }
    g_mod.v3_copy_running.store(true, std::memory_order_release);
    g_mod.v3_copy_thread = _beginthreadex(nullptr, 1u << 16, v3_copy_worker_main, nullptr, 0, nullptr);
    if (g_mod.v3_copy_thread == 0) {
        unwind();
        err = "cannot create the bounce copy worker thread";
        return false;
    }
    return true;
}

// ★락 밖에서 부른다(worker 가 g_mod.mutex 를 잡으므로 쥔 채 join 하면 교착).
void v3_copy_stop_and_join() {
    if (!g_mod.v3_copy_running.load(std::memory_order_acquire)) return;
    g_mod.v3_copy_running.store(false, std::memory_order_release);
    g_mod.v3_copy_pause.store(0, std::memory_order_release);   // pause 중이어도 빠져나오게
    if (g_mod.v3_copy_event) SetEvent(g_mod.v3_copy_event);
    if (g_mod.v3_copy_thread) {
        WaitForSingleObject((HANDLE)g_mod.v3_copy_thread, INFINITE);
        CloseHandle((HANDLE)g_mod.v3_copy_thread);
        g_mod.v3_copy_thread = 0;
    }
    if (g_mod.v3_copy_event) { CloseHandle(g_mod.v3_copy_event); g_mod.v3_copy_event = nullptr; }
    std::lock_guard<std::mutex> lk(g_mod.v3_copy_mutex);
    g_mod.v3_copy_q.clear(); g_mod.v3_copy_head = g_mod.v3_copy_count = g_mod.v3_copy_cap = 0;
    g_mod.v3_copy_outstanding = 0;   // ★S1 — 큰를 비웠으므로 미결 의무가 남을 수 없다
}

// op terminal 판정 1지점 — **reads·copies 가 둘 다 0** 일 때만 선다(§3-4).
// 반환 true = 이 호출로 op terminal 이 섰다(호출자는 이후 op/cs 를 만지면 안 된다).
bool v3_maybe_terminal_locked(int32_t oi) {
    V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
    if (cs.remaining > 0 || cs.remaining_copies > 0) return false;   // 아직 drain 중
    // ★★R-1(A0-2 부분 제출) — **아직 안 나간 wave 가 있으면 terminal 이 아니다.** wave k 가
    //   전부 회수돼 `remaining==0` 이 되어도 record 는 미완이다(구 판은 전 child 가 한 번에
    //   나갔으므로 이 구분이 필요 없었다). 여기서 막지 않으면 첫 wave 회수만으로 publish 가
    //   서서 **부분 데이터를 완성본으로 내보낸다**.
    //   ★단 실패/취소 래치가 섰으면 남은 wave 는 발행하지 않고 그대로 닫는다 — 그 경우
    //     `bytes_ok < want_total` 이라 `clean` 이 거짓이고, IO3 §5-3 의 "미도달 membership 이
    //     남으면 terminal_status=OK 금지"와도 정합한다(status 는 실패/취소 계열로 확정된다).
    if (cs.active && cs.wave_next < cs.n_children && !cs.failure_latch && !cs.cancel_mode) return false;
    PendingIo & op = g_mod.pending_ops[(size_t)oi];
    // ★기대 바이트는 층별로 미리 확정돼 있다: aligned=region(브라켓 전량이 슬롯에 앉는다) ·
    //   비4K=slice_bytes(브라켓은 staging 으로 가고 슬롯엔 copy 가 슬라이스만 옮긴다).
    // ★A6 — 기대 바이트는 **제출 시점에 이 op 이 확정한 값**이다(coalesced part 는 브라켓이
    //   아니라 slice 만 슬롯에 남으므로 층 상수와 갈린다). 0=미설정(있을 수 없는 값)은 층
    //   상수로 되돌려 A5 이전 거동을 그대로 둔다.
    const uint64_t want_total = cs.bytes_expected != 0 ? cs.bytes_expected
                                                       : g_mod.layers[(size_t)op.mlpos].child_bytes_expected;
    const bool clean = !cs.failure_latch && cs.bytes_ok == want_total;
    uint8_t status;
    if (cs.cancel_mode) {
        // ★취소 drain — 전 child 가 성공했더라도 publish 금지(성공은 status 로만 구분 기록).
        status = clean ? (uint8_t)GGML_MOE_PB_T_CANCEL_SUCCESS_UNPUB
                       : (uint8_t)GGML_MOE_PB_T_CANCELLED_UNPUBLISHED;
    } else if (clean) {
        status = (uint8_t)GGML_MOE_PB_T_SUCCESS_PUBLISHED;
    } else {
        status = cs.status ? cs.status : (uint8_t)GGML_MOE_PB_T_ERROR_UNPUBLISHED;
    }
    const bool all_ok = (status == GGML_MOE_PB_T_SUCCESS_PUBLISHED);
    const bool wp = (cs.any_pending != 0);
    const DWORD agg = (DWORD)(all_ok ? want_total : cs.bytes_ok);
    const bool was_copy_error = (cs.copy_error != 0);
    const uint64_t latch_qpc  = cs.terminal_latch_qpc;   // ★R12 — 초기화 전에 집는다(§5-3)
    cs = V3ChildIo{};   // 재사용 전 초기화(다음 op 이 이 인덱스를 받는다)
    g_mod.v3_op_terminals++;
    if (all_ok) g_mod.v3_publishes++;
    // ★A5 — IO3 records 행 종결(§5-3). 상태는 실패 원인을 따른다(READ vs COPY 구분).
    {
        uint8_t io3st;
        if (status == GGML_MOE_PB_T_SUCCESS_PUBLISHED)      io3st = IO3_T_OK;
        else if (status == GGML_MOE_PB_T_ERROR_UNPUBLISHED) io3st = was_copy_error ? IO3_T_COPY_ERROR : IO3_T_READ_ERROR;
        else                                                io3st = IO3_T_CANCELLED;
        io3_close_record_locked(oi, io3st, !g_mod.pending_ops[(size_t)oi].waiters.empty(), latch_qpc);
    }
    disp_op_terminal_locked(oi, agg, wp, status, (uint32_t)want_total);
    return true;
}

// 실패 래치 1지점 — 첫 실패가 형제를 취소하고 상태를 고정한다.
inline void v3_latch_failure_locked(int32_t oi) {
    V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
    if (cs.failure_latch) return;
    cs.failure_latch = 1;
    // ★R12 — §5-3: 실패 record 의 `terminal_qpc` 는 **이 순간**이다(형제 취소 회수가 끝난
    //   시각이 아니다). 첫 래치만 세운다.
    if (cs.terminal_latch_qpc == 0) cs.terminal_latch_qpc = ggml_moe_phaseb_now_qpc();
    cs.status = GGML_MOE_PB_T_ERROR_UNPUBLISHED;
    v3_cancel_children_locked(oi);
}

// ★A4 — 바운스 copy 1건의 종결(copy worker 가 memcpy 를 마치고 부른다).
//   staging 은 **여기서** 반납한다(= copy terminal · §3-5 "copy terminal 까지 재사용 금지").
bool v3_copy_complete_locked(const ModuleState::V3CopyJob & job, bool ok, uint64_t start_qpc) {
    const int32_t oi = job.oi;
    V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
    // ★A5 — copy obligation 행 종결(§5-5): 착수 행은 start/done 이 서고 status 가 확정된다.
    io3_copy_note_terminal_locked(job.io3_copy_row, 0, start_qpc,
                                  ok ? (uint8_t)IO3_C_OK : (uint8_t)IO3_C_ERROR);
    v3_copy_credit_give_locked(1);   // ★S1 — 예약~완료 구간의 끝
    if (job.coal >= 0) {
        // ★A6 — coalesce 버퍼는 **참가 scatter 전건이 끝나야** 반납된다(§3-6 ⑤·§3-5 소유권 사상:
        //   copy terminal 전 재사용 금지). 참가 record 하나가 먼저 끝났다고 놓으면 남은 scatter 가
        //   덮어써진 버퍼를 읽는다.
        V3CoalRead & cr = g_mod.v3_coal[(size_t)job.coal];
        if (cr.pending_copies > 0 && --cr.pending_copies == 0) cr.active = 0;
        g_mod.v3_coal_scatters++;
    } else {
        v3_staging_give_locked(job.staging);
    }
    if (ok) cs.bytes_ok += job.bytes;
    else  { cs.copy_error = 1; v3_latch_failure_locked(oi); }
    if (cs.remaining_copies > 0) cs.remaining_copies--;
    return v3_maybe_terminal_locked(oi);
}

// child read 1건의 회수. 반환 true = 이 완료로 op terminal 이 섰다.
// ★out-of-order 안전: 어떤 순서로 와도 마지막 회수만 terminal 을 부른다. immediate(동기)·
//   pending(비동기) 혼합도 같은 경로로 합류한다.
// ★A4 — 바운스 child 는 여기서 **끝나지 않는다**: 채널만 반환하고 copy 를 큐에 넘긴 뒤
//   `remaining_copies` 로 넘어간다(§3-5 "완료 시 채널 즉시 반환").
bool v3_child_retire_locked(int32_t oi, int32_t chan, bool ok, DWORD bytes, bool was_pending,
                            bool cancel = false) {
    V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
    PendingIo & op = g_mod.pending_ops[(size_t)oi];
    int32_t rp = -1, staging = -1;
    if (chan >= 0) {
        // ★R3 — 회수 **순서**를 part 인덱스 열로 남긴다(단언 근거). 채널 매핑을 지우기 전에 읽는다.
        rp = g_mod.v3_chan_part[(size_t)chan];
        if (rp >= 0) g_mod.v3_retire_order.push_back(rp);
        staging = g_mod.v3_chan_staging[(size_t)chan];
        g_mod.v3_chan_op[(size_t)chan]      = -1;
        g_mod.v3_chan_part[(size_t)chan]    = -1;
        g_mod.v3_chan_staging[(size_t)chan] = -1;
    }
    // ★A5 — child terminal 사건(§5-6 공용 카운터). 채널 매핑을 지우기 전에 행을 집는다.
    int32_t io3_row = -1, io3_copy_row = -1;
    if (chan >= 0 && (size_t)chan < g_mod.io3_chan_row.size()) {
        io3_row = g_mod.io3_chan_row[(size_t)chan];
        g_mod.io3_chan_row[(size_t)chan] = -1;
        io3_copy_row = g_mod.io3_chan_copy_row[(size_t)chan];
        g_mod.io3_chan_copy_row[(size_t)chan] = -1;
    }
    // ★★PB virtual A1 — ②축 감소는 IO3 방출 **직전**에 경로가 한다(snapshot 의미 "이 op 제외"
    //   보존). credit 이 실제로 풀리는 지점이므로 drain-defer park 도 여기서 내려간다.
    v3_child_inflight_sub_locked(op.origin == PIO_ORIGIN_PREFETCH);
    io3_child_terminal_locked(io3_row, ok ? (uint8_t)IO3_C_OK
                                          : (cancel ? (uint8_t)IO3_C_CANCELLED : (uint8_t)IO3_C_ERROR));
    if (was_pending) g_mod.v3_children_async++; else g_mod.v3_children_sync++;
    if (cancel) {
        cs.cancel_mode = 1;
        // ★R12 — 취소도 "실패 계열 terminal"이다(§5-3 latch 시각 규약 — 성공만 publish 직전).
        if (cs.terminal_latch_qpc == 0) cs.terminal_latch_qpc = ggml_moe_phaseb_now_qpc();
    }
    if (was_pending) cs.any_pending = 1;
    g_mod.v3_children_retired++;
    if (cs.remaining > 0) cs.remaining--;

    if (!ok) {
        v3_latch_failure_locked(oi);
        v3_staging_give_locked(staging);        // 읽기가 실패했으면 copy 는 없다
        if (staging >= 0) v3_copy_credit_give_locked(1);   // ★S1 — 예약해둔 몲 반납
        return v3_maybe_terminal_locked(oi);
    }
    if (staging < 0 || rp < 0) {                // aligned child — 슬롯에 바로 앉았다
        cs.bytes_ok += (uint64_t)bytes;
        return v3_maybe_terminal_locked(oi);
    }
    // ---- 바운스 child: 채널은 이미 반환됐고, 이제 copy 가 남았다 ----
    const PartInfo & pi = g_mod.layers[(size_t)op.mlpos].parts[(size_t)rp];
    const uint64_t A   = g_mod.align_bytes;
    const uint64_t src = pi.abs_offset + (uint64_t)(op.m.record_idx % g_mod.n_expert) * pi.slice_bytes;
    ModuleState::V3CopyJob job;
    job.oi       = oi;
    job.part     = rp;
    job.staging  = staging;
    job.src      = (const char *)g_mod.v3_staging_base +
                   (size_t)staging * g_mod.v3_staging_stride + (size_t)(src % A);
    job.bytes    = pi.slice_bytes;
    job.dest     = (char *)g_mod.slots[(size_t)op.m.slot].base + pi.slot_offset;
    job.ok       = 1;
    job.coal     = -1;
    job.io3_copy_row = io3_copy_row;
    if (g_mod.v3_test_fail_copy == rp) { g_mod.v3_test_fail_copy = -1; job.ok = 0; }
    // ★취소 중이면 copy 를 태우지 않는다 — 취소된 op 의 슬롯에 쓰는 것은 무의미하고,
    //   drain 만 정확히 닫으면 된다.
    if (cs.cancel_mode || cs.failure_latch) {
        v3_staging_give_locked(staging);
        v3_copy_credit_give_locked(1);          // ★S1
        return v3_maybe_terminal_locked(oi);
    }
    cs.remaining_copies++;
    // ★A5 — 이 child 의 copy obligation 행에 enqueue 시각을 찍는다(§5-5 상태·시각 규약).
    io3_copy_note_enqueue_locked(job.io3_copy_row);
    if (!v3_copy_enqueue(job)) {
        // 큐 만석 = 제출 측 backpressure 가 깨진 것이다(용량은 in-flight 상한 이상으로 잡는다).
        fatal_abort("bounce copy queue overflow (capacity=%d) - submission backpressure is broken",
                    (int)g_mod.v3_copy_cap);
    }
    return false;   // copy 가 끝나야 terminal 이다
}

inline char * v3_coal_buf(int32_t ci) {
    return (char *)g_mod.v3_coal_base + (size_t)ci * g_mod.v3_coal_stride;
}

inline void v3_coal_reject(uint8_t why) {
    g_mod.v3_coal_last_reject = why;
    g_mod.v3_coal_reject_mask |= (1u << why);
}

// 대구간 계획. 반환 true = 합칠 수 있다(`g_mod.v3_coal_plan` 이 그 스냅샷).
// ★7조건 대응: ①동일 source/tensor=같은 (mlpos, part) 로만 후보를 모은다 + source_index 재확인
//   ②expert 완전 연속(gap 0B — 건너뛴 expert 를 over-read 하지 않는다) ③텐서 경계 불월
//   ④브라켓 시작·길이·staging 정렬/용량 ⑤전 참가 record 사전 claim·pin ⑥⑦은 실행 규칙이라
//   제출·회수 쪽에서 지킨다(scatter=exact slice · READY=전 part scatter 후).
bool v3_plan_coalesce_locked(int32_t oi_head) {
    V3CoalPlan & pl = g_mod.v3_coal_plan;
    pl.ops.clear();
    if (g_mod.v3_coal_phase == V3_COAL_OFF || g_mod.v3_coal_count <= 0) {
        v3_coal_reject(V3_COAL_R_DISABLED); return false;
    }
    const PendingIo & h = g_mod.pending_ops[(size_t)oi_head];
    if (g_mod.v3_coal_phase == V3_COAL_PREFILL && h.phase != Phase::PREFILL) {
        v3_coal_reject(V3_COAL_R_PHASE); return false;      // §3-6 프리필 한정(디코드 기본 OFF)
    }
    const int32_t mlpos = h.mlpos;
    const int32_t np    = v3_parts_of(mlpos);
    const int32_t nE    = g_mod.n_expert;
    if (np <= 0 || np > 64 || nE <= 1) { v3_coal_reject(V3_COAL_R_RUN); return false; }

    // ⑤ 사전 claim·pin — head 부터 본다. 큐에 든 op 은 claim 이 끝나 있는 것이 A3 계약이지만,
    //    합침은 **남의 슬롯에 쓰는** 일이라 그 계약에 기대지 않고 매번 확인한다.
    auto claimed_and_pinned = [&](const PendingIo & op) {
        const int32_t s = op.m.slot;
        if (s < 0 || (size_t)s >= g_mod.slots.size()) return false;
        if ((size_t)s >= g_mod.slot_pending.size() || g_mod.slot_pending[(size_t)s] == 0) return false;
        return g_mod.slots[(size_t)s].owner_record == op.m.record_idx;
    };
    // ★주입은 **sticky** 다 — **1회성 소모형**으로 두면 한 acquire 가 pump 를 여러 번
    //   돌리므로 첫 계획에서 소진되고, 두 번째 계획이 합쳐버려 "이 가드가 막았다"는
    //   판정이 성립하지 않는다.
    if (g_mod.v3_test_coal_unpin == 1 || !claimed_and_pinned(h)) { v3_coal_reject(V3_COAL_R_CLAIM); return false; }

    // 후보 수집 — 큐의 **미제출 잔여 전체**에서 같은 층·동질 조건을 만족하는 record 를 모은다
    //   (앞뒤 인접 위치만 보지 않는다: 큐 순서는 유니크 miss 첫 출현 순이라 expert 순이 아니다).
    if ((int32_t)pl.cand.size() < nE) pl.cand.assign((size_t)nE, -1);
    else std::fill(pl.cand.begin(), pl.cand.begin() + nE, -1);
    // §5-4 8번째 MUST — 같은 대구간의 record 는 origin·phase·graph_seq·request_index·is_retry 동일.
    auto homogeneous = [&](const PendingIo & op) {
        return op.origin == h.origin && op.phase == h.phase &&
               op.pb_graph_seq == h.pb_graph_seq && op.pb_request_epoch == h.pb_request_epoch &&
               op.pb_is_retry == h.pb_is_retry;
    };
    bool saw_unpinned = false;  // ⑤ 미충족으로 제외된 후보가 있었나(거부 사유 지목)
    bool saw_foreign = false;   // 같은 층인데 동질 조건이 갈린 후보가 있었나(관측용)
    for (size_t q = g_mod.submit_head; q < g_mod.submit_q.size(); q++) {
        const int32_t oi = g_mod.submit_q[q];
        if (oi < 0) continue;
        const PendingIo & op = g_mod.pending_ops[(size_t)oi];
        if (op.mlpos != mlpos) continue;                       // ① 다른 층 = 다른 tensor
        if (!homogeneous(op))  { saw_foreign = true; continue; }
        // ★★S3(r8 Z1-1) — 주입 축이 둘이다: 1=head · 2=**participant**. 구 판은 head 만
        //   행사해서 비대표 후보의 claim·pin 필터를 지워도 관문이 통과했다 — ⑤ 는
        //   "대상 record **전부**"를 요구하므로 그 축을 따로 잠그지 않으면 관문이 공허하다.
        if (!claimed_and_pinned(op) || (g_mod.v3_test_coal_unpin == 2 && oi != oi_head)) {
            saw_unpinned = true;
            continue;
        }
        const int32_t e = (int32_t)(op.m.record_idx % nE);
        if (e >= 0 && e < nE && pl.cand[(size_t)e] < 0) pl.cand[(size_t)e] = oi;
    }
    const int32_t e_head = (int32_t)(h.m.record_idx % nE);

    // ② 완전 연속 run — head 를 포함해 아래·위로 **빈틈 없이** 확장한다(gap 0B).
    int32_t e0 = e_head, e1 = e_head;
    while (e0 - 1 >= 0  && pl.cand[(size_t)(e0 - 1)] >= 0) e0--;
    while (e1 + 1 <  nE && pl.cand[(size_t)(e1 + 1)] >= 0) e1++;
    int32_t n = e1 - e0 + 1;
    // run 이 안 서는 사유를 지목한다 — 후보가 ⑤ 로 빠졌으면 그것이 진짜 원인이다.
    if (n < 2) {
        v3_coal_reject(saw_unpinned ? V3_COAL_R_CLAIM
                     : saw_foreign  ? V3_COAL_R_HOMOGEN : V3_COAL_R_RUN);
        return false;
    }

    // ★테스트 전용 — run 을 텐서 끝 너머로 1 expert 부풀려 ③ 경계 가드를 행사한다.
    //   실 런타임에서는 도달 불가다: expert 는 `record_idx % n_expert` 라 [0, n_expert) 를 못 벗어나고,
    //   §2-4 가 `align_up(abs + n_expert*slice, A) <= source.bytes` 를 **플랜 단계에서** 강제한다.
    const int32_t overrun = g_mod.v3_test_coal_overrun ? 1 : 0;   // ★sticky(위 unpin 과 같은 사유)

    // ④ 용량 — 대구간 1건은 `min(io_coalesce_max, 버퍼 1칸)` 을 넘을 수 없다. 전 part 가
    //    함께 들어가야 하므로(참가 집합은 part 마다 같다) 가장 빡빡한 part 가 run 을 정한다.
    uint64_t fit = g_mod.v3_coal_max_bytes;
    if (fit > (uint64_t)g_mod.v3_coal_stride) fit = (uint64_t)g_mod.v3_coal_stride;
    const uint64_t A = g_mod.align_bytes;
    int32_t n_max = n;
    for (int32_t p = 0; p < np; p++) {
        const PartInfo & pi = g_mod.layers[(size_t)mlpos].parts[(size_t)p];
        if (pi.slice_bytes == 0) { v3_coal_reject(V3_COAL_R_ALIGN); return false; }
        const uint64_t head0 = (pi.abs_offset + (uint64_t)e0 * pi.slice_bytes) % A;
        // align_up(head + k*slice, A) <= fit ⟺ head + k*slice <= fit (fit 은 A 배수)
        const int64_t room = (int64_t)fit - (int64_t)head0;
        const int32_t k = (room <= 0) ? 0 : (int32_t)((uint64_t)room / pi.slice_bytes);
        if (k < n_max) n_max = k;
    }
    if (n_max < 2) { v3_coal_reject(V3_COAL_R_ALIGN); return false; }
    // ★copy 큐 하한 — 대구간 1건이 완료되면 참가 record 수만큼의 scatter 의무가 **동시에** 큐로
    //   간다. 그룹의 part 전건이 그렇게 되므로 최악 동시 의무는 `np × run` 이다. 이 값이
    //   `io_bounce_copy_qd`(§5-6 copy queue cap 이 대조하는 바로 그 값)를 넘으면 wire 가 자기
    //   선언을 어기게 되므로, 큐 깊이가 run 길이의 상한이다 — 더 깊게 합치려면 그 노브를 올린다.
    {
        const int32_t qd_copies = g_mod.v3_io_param_val[5];
        const int32_t by_queue  = (np > 0) ? qd_copies / np : 0;
        if (by_queue < n_max) n_max = by_queue;
        if (n_max < 2) { v3_coal_reject(V3_COAL_R_RESOURCE); return false; }
    }
    if (n > n_max) {
        // 창을 줄이되 **head 를 반드시 포함**한다 — head 가 빠지면 큐 head 가 소비되지 않아
        // pump 가 같은 자리에서 다시 계획하고 진행하지 않는다.
        int32_t s = e_head - (n_max - 1);
        if (s < e0) s = e0;
        int32_t l = e1 - s + 1;
        if (l > n_max) l = n_max;
        e0 = s; n = l;
    }
    if (n < 2) { v3_coal_reject(V3_COAL_R_ALIGN); return false; }

    // ③ 텐서 경계 불월 — run 이 expert 축을 벗어나지 않는다(+ 대구간 끝이 source EOF 안).
    const int32_t e_last = e0 + n - 1 + overrun;
    if (e0 < 0 || e_last >= nE) { v3_coal_reject(V3_COAL_R_BOUNDARY); return false; }

    // ①④ 기하 확정 — part 마다 (offset, len, head). 여기서 A 정렬을 **산출이 아니라 검사**한다.
    uint64_t total = 0;
    const int32_t sidx0 = g_mod.layers[(size_t)mlpos].parts[0].source_index;
    (void)sidx0;
    for (int32_t p = 0; p < np; p++) {
        const PartInfo & pi = g_mod.layers[(size_t)mlpos].parts[(size_t)p];
        if (pi.source_index < 0 || (size_t)pi.source_index >= k2fac_source_count_locked()) {
            v3_coal_reject(V3_COAL_R_SOURCE); return false;    // ① 소스 핸들이 없는 part
        }
        const uint64_t src   = pi.abs_offset + (uint64_t)e0 * pi.slice_bytes;
        const uint64_t head0 = src % A;
        const uint64_t off   = src - head0;
        uint64_t len64 = 0;
        if (!ck_add(head0, (uint64_t)n * pi.slice_bytes, len64)) { v3_coal_reject(V3_COAL_R_ALIGN); return false; }
        len64 = align_up(len64, A);
        if ((off % A) != 0 || (len64 % A) != 0 || len64 > fit || len64 > 0xFFFFFFFFull) {
            v3_coal_reject(V3_COAL_R_ALIGN); return false;     // ④
        }
        // ③ source EOF — §2-4 가 플랜 단계에서 이미 보장하지만 소비 시점에도 확인한다.
        uint64_t end = 0;
        if (!ck_add(off, len64, end) ||
            (size_t)pi.source_index >= g_mod.sources.size() ||
            end > g_mod.sources[(size_t)pi.source_index].bytes) {
            v3_coal_reject(V3_COAL_R_BOUNDARY); return false;
        }
        pl.off[p]  = off;
        pl.len[p]  = (uint32_t)len64;
        pl.head[p] = head0;
        total     += len64;
    }
    // §5 io_coalesce_min — 합쳐서 얻는 것이 없으면 합치지 않는다(합침은 staging+scatter 비용이다).
    if (total < g_mod.v3_coal_min_bytes) { v3_coal_reject(V3_COAL_R_MIN); return false; }

    pl.mlpos   = mlpos;
    pl.expert0 = e0;
    pl.n_parts = np;
    for (int32_t e = e0; e < e0 + n; e++) pl.ops.push_back(pl.cand[(size_t)e]);
    v3_coal_reject(V3_COAL_R_NONE);
    return true;
}

// 대구간 1건의 회수 — 참가 record **전건**을 처분한다(단독 child 의 op 1건 처분과 갈리는 지점).
void v3_coal_child_retire_locked(int32_t ci, int32_t chan, bool ok, DWORD bytes,
                                 bool was_pending, bool cancel) {
    V3CoalRead & cr = g_mod.v3_coal[(size_t)ci];
    if (chan >= 0) {
        g_mod.v3_chan_op[(size_t)chan]      = -1;
        g_mod.v3_chan_part[(size_t)chan]    = -1;
        g_mod.v3_chan_staging[(size_t)chan] = -1;
        g_mod.v3_chan_coal[(size_t)chan]    = -1;
    }
    int32_t io3_row = -1;
    if (chan >= 0 && (size_t)chan < g_mod.io3_chan_row.size()) {
        io3_row = g_mod.io3_chan_row[(size_t)chan];
        g_mod.io3_chan_row[(size_t)chan] = -1;
        g_mod.io3_chan_copy_row[(size_t)chan] = -1;   // 대구간의 copies 는 cr 이 소유한다
    }
    // ★★PB virtual A1 — ②축 감소(단독 child 와 같은 규율). 대구간의 origin 은 참가 집합이
    //   동질이므로(§5-4 8번째 MUST) 대표 op 하나로 판정해도 된다.
    {
        const int32_t oi0 = cr.ops.empty() ? -1 : cr.ops[0];
        v3_child_inflight_sub_locked(oi0 >= 0 &&
                                     g_mod.pending_ops[(size_t)oi0].origin == PIO_ORIGIN_PREFETCH);
    }
    io3_child_terminal_locked(io3_row, ok ? (uint8_t)IO3_C_OK
                                          : (cancel ? (uint8_t)IO3_C_CANCELLED : (uint8_t)IO3_C_ERROR));
    if (was_pending) g_mod.v3_children_async++; else g_mod.v3_children_sync++;
    g_mod.v3_children_retired++;
    g_mod.v3_coal_reads_retired++;   // ★S3 — ⑦ 관문은 제출이 아니라 **회수** 시점을 봐야 한다
    (void)bytes;

    const PartInfo & pi = g_mod.layers[(size_t)cr.mlpos].parts[(size_t)cr.part];
    cr.pending_copies = 0;
    // ★1단계: 태울 copy 를 먼저 **전부 계상**한다. 계상 전에 enqueue 하면 worker 가 먼저 완료해
    //   `pending_copies` 를 0 으로 만들고 버퍼를 놓아버릴 수 있다(우리는 잠금을 쥐고 있어 실제로는
    //   막히지만, 순서 자체를 규약으로 둔다).
    const size_t n_part_ops = cr.ops.size();
    std::vector<ModuleState::V3CopyJob> jobs;
    std::vector<uint8_t> carried((size_t)n_part_ops, 0);
    jobs.reserve(n_part_ops);
    for (size_t k = 0; k < n_part_ops; k++) {
        const int32_t oi = cr.ops[k];
        V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
        PendingIo & op = g_mod.pending_ops[(size_t)oi];
        const int32_t copy_row = (cr.io3_copy_row0 >= 0) ? cr.io3_copy_row0 + (int32_t)k : -1;
        if (!ok) {
            if (cancel) {
                cs.cancel_mode = 1;
                if (cs.terminal_latch_qpc == 0) cs.terminal_latch_qpc = ggml_moe_phaseb_now_qpc();
            } else {
                v3_latch_failure_locked(oi);
            }
            continue;   // 아래 2단계에서 remaining_copies 를 내린다
        }
        if (cs.cancel_mode || cs.failure_latch) continue;   // 이미 진 record 의 슬롯엔 쓰지 않는다
        ModuleState::V3CopyJob job;
        job.oi    = oi;
        job.part  = cr.part;
        job.staging = -1;
        job.coal  = ci;
        // ⑥ scatter 는 exact slice 단위 — 대구간 안에서 그 expert 의 슬라이스 시작 위치로만 간다.
        const int32_t e = (int32_t)(op.m.record_idx % g_mod.n_expert);
        job.src   = v3_coal_buf(ci) + (size_t)(cr.src_head + (uint64_t)(e - cr.expert0) * pi.slice_bytes);
        job.bytes = pi.slice_bytes;
        // ★착지는 `slot_offset` 이 아니라 **`data_offset`** 이다 — `ptr()` 이 돌려주는 지점이
        //   그것이고 aligned part 에서는 둘이 `bracket_head` 만큼 다르다(§2-4). 바운스는 비4K
        //   전용이라 둘이 늘 같았고, 그래서 그 코드를 그대로 베끼면 aligned 합침에서 어긋난다.
        job.dest  = (char *)g_mod.slots[(size_t)op.m.slot].base + pi.data_offset;
        job.ok    = 1;
        job.io3_copy_row = copy_row;
        if (g_mod.v3_test_fail_copy == cr.part) { g_mod.v3_test_fail_copy = -1; job.ok = 0; }
        io3_copy_note_enqueue_locked(copy_row);
        cr.pending_copies++;
        carried[k] = 1;
        jobs.push_back(job);
    }
    // ★2단계: copy 가 안 붙은 참가 record 는 여기서 의무를 내린다(제출 때 올려둔 몫).
    //   ★op terminal 이 여기서 설 수 있으므로(그 op 은 해제된다) 이 뒤로 그 op 을 만지지 않는다.
    for (size_t k = 0; k < n_part_ops; k++) {
        if (carried[k]) continue;
        const int32_t oi = cr.ops[k];
        v3_copy_credit_give_locked(1);          // ★S1 — 이 participant 몲은 안 태운다
        V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
        if (cs.remaining_copies > 0) cs.remaining_copies--;
        v3_maybe_terminal_locked(oi);
    }
    if (cr.pending_copies == 0) cr.active = 0;    // 아무도 안 태웠으면 버퍼는 지금 논다
    for (const auto & j : jobs) {
        if (!v3_copy_enqueue(j))
            fatal_abort("coalesce scatter queue overflow (capacity=%d) - submission backpressure is broken",
                        (int)g_mod.v3_copy_cap);
    }
}

// blocked 구간 마감 — 이 demand op 이 대기했다면 그 µs 를 자기 행에 귀속한다.
// ★A6 에서 함수로 분리(거동 불변): 합침 제출은 record 를 여러 건 한꺼번에 내보내므로 이 마감을
//   참가 record **각각**에 대해 돌려야 한다. 인라인 블록이면 그 자리에 둘 수 없다.
void disp_close_blocked_locked(int32_t oi) {
    PendingIo & op = g_mod.pending_ops[(size_t)oi];
    if (op.pb_blocked_start == 0) return;
    const uint64_t d = ggml_moe_phaseb_now_qpc() - op.pb_blocked_start;
    const uint64_t us = pb_qpc_to_us(d);
    const uint64_t key = pb_row_key(op.pb_graph_seq, (uint16_t)g_mod.moe_layers[(size_t)op.mlpos]);
    auto it = g_mod.pb_sample_index.find(key);
    if (it != g_mod.pb_sample_index.end()) {
        ggml_moe_phaseb_sample & s = g_mod.pb_samples[(size_t)it->second];
        s.blocked_us += (uint32_t)us;
        s.blocked_count++;
    }
    ggml_moe_phaseb_count_blocked(op.pb_request_epoch, us, 1);
    // ★5c-T(부록 14.1·14.3 ③·14.6) — blocked interval 1건 원시 기록 + rank 주석.
    //   ①`waited_acquire`/`observed_frontier_match` 는 **1 acquire 1계수**라 여기서는
    //     소유 command 에 래치만 걸고, 계수는 canonical acquire 종결(§pb_finish_demand_
    //     command_locked) 1지점에서 한다. ②rank 는 경량 예측열 스냅샷으로만 판정하며
    //     counterfactual 을 만들지 않는다(controller §3 C3 경계).
    if (g_mod.pb_collect) {
        const int32_t expert = (op.m.record_idx >= 0 && g_mod.n_expert > 0)
                               ? (int32_t)(op.m.record_idx % g_mod.n_expert) : -1;
        const uint16_t pr = pb_pred_frontier_rank_locked(op.pb_graph_seq, op.mlpos, expert);
        for (int32_t ci : op.waiters) {
            if (ci < 0 || ci >= MOE_DIRECT_CONTROL_Q_CAP) continue;
            g_mod.pb_cmd_waited[(size_t)ci] = 1;
            if (pr != GGML_MOE_PHASEB_NA_U16) g_mod.pb_cmd_frontier[(size_t)ci] = 1;
        }
        ggml_moe_phaseb_note_blocked_interval(op.pb_request_epoch);
        ggml_moe_phaseb_ledger r = pb_led_new(GGML_MOE_PB_L_BLOCKED_INTERVAL);
        r.request_epoch = op.pb_request_epoch;
        r.graph_seq     = op.pb_graph_seq;
        r.graph_epoch   = op.pb_graph_epoch;
        r.record_idx    = op.m.record_idx;
        r.layer         = (uint16_t)g_mod.moe_layers[(size_t)op.mlpos];
        r.origin        = 0;   // blocked 주체는 demand
        r.pred_rank     = pr;
        r.start_qpc     = op.pb_blocked_start;
        r.end_qpc       = op.pb_blocked_start + d;
        r.qpc           = op.pb_blocked_start;   // 사건 시각 = 대기 시작
        pb_led_stamp_from_request(r, op.pb_request_epoch);
        // ★r1 M4 — 참여 집합은 대기 **시작 시점** 스냅샷(controller §6 window 참여자).
        PbOpStamp * ost = pb_op_stamp_at(oi);
        if (ost != nullptr) {
            pb_led_emit_locked(r, ost->blocked_parts.data(), (uint32_t)ost->blocked_parts.size());
            ost->blocked_parts.clear();
        } else {
            pb_led_emit_locked(r);
        }
    }
    op.pb_blocked_start = 0;
}

// 제출 pump — 자유 채널이 있는 동안 submit_q FIFO 순으로 제출(= 유니크 miss 첫 출현 순).
// 반환 true = 이 pass에서 제출/회수 활동이 있었다.
bool disp_pump_locked() {
    // ★★R-5 — 제출 보류(테스트 전용). 큐에 남긴 채 돌려보내므로 record 는 admitted 상태로
    //   `reserved` 를 든 채 머문다 = never-issued terminal 을 만들 수 있는 유일한 창이다.
    if (g_mod.test_hold_submit) return false;
    bool progressed = false;
    const bool pa_on = g_mod.phasea_on;

    // ★A6 — 합침이 큐 중간의 record 를 함께 소비하면 그 자리가 -1 로 남는다. head 는 실 엔트리
    //   위에만 서야 한다(그래야 `submit_head >= size` 라는 기존 "큐 비었다" 판정이 그대로 산다).
    auto skip_consumed = [&]() {
        while (g_mod.submit_head < g_mod.submit_q.size() && g_mod.submit_q[g_mod.submit_head] < 0)
            g_mod.submit_head++;
    };
    skip_consumed();
    // ★★R-1(FROZEN §2-0 A0-2) — 큐 **스캔 커서**. 구 판은 `submit_head` 하나로 순회했지만
    //   wave 기계에서는 ①한 record 가 여러 pass 에 걸쳐 남아 있고 ②continuation credit 이
    //   막힌 prefetch record 는 **건너뛰어야** 뒤의 demand 가 안 막힌다. 그래서 소비 위치
    //   (`submit_head`)와 스캔 위치(`qi`)를 분리한다. `qi` 는 매 반복 단조 증가라 종료가 자명하다.
    //   ★소비된 자리는 기존 A6 의 `-1` 홀 기제를 그대로 쓴다(신규 발명 0).
    size_t qi = g_mod.submit_head;
    // 이 pass 에서 continuation 재검사가 1건이라도 실패했는가(A0-4 drain-before-refill 입력).
    bool continuation_deferred = false;
    while (qi < g_mod.submit_q.size()) {
        if (g_mod.submit_q[qi] < 0) { qi++; continue; }   // ★A6 합침이 걷어간 자리
        // ★★R-1 — 이 record 가 **이번 wave 에 낼 child 수**를 정한다:
        //     `W_l = min(P_l, io_part_parallelism)` · `n_need = min(W_l, 남은 part 수)`.
        //   부분 제출은 **정상 상태**다(사양 A0-2 · IO3 가 허용·검증).
        int32_t n_need = 1, n_bounce_need = 0;
        bool coalesce = false;
        int32_t coal_slot[64];
        int32_t n_coal = 0;
        int32_t wave_from = 0;              // 이번 wave 가 제출할 첫 part 인덱스
        if (v3_active()) {
            const int32_t oi_peek = g_mod.submit_q[qi];
            const int32_t ml = g_mod.pending_ops[(size_t)oi_peek].mlpos;
            const int32_t np_total = v3_parts_of(ml);
            V3ChildIo & cs_peek = g_mod.v3_child[(size_t)oi_peek];
            wave_from = cs_peek.active ? cs_peek.wave_next : 0;
            const int32_t w = pbv_wave_width(ml);
            const int32_t rem_parts = np_total - wave_from;
            n_need = (w < rem_parts) ? w : rem_parts;
            if (n_need <= 0) { g_mod.submit_q[qi] = -1; qi++; skip_consumed(); continue; }  // 방어
            // ★part cap(IO3 §5-6) — **record 별 동시 child membership** ≤ `io_part_parallelism`.
            //   (§5-6 문면: "**part cap**=record별 동시 child membership ≤ io_part_parallelism".
            //    전역 QD cap 이 아니라 **per-record** 단위라는 점이 이 조건식의 근거다.)
            //   wave k 가 아직 in-flight 인데 k+1 을 띄우면 membership 이 2W 가 되어 그 cap 을
            //   깬다 ⇒ 다음 wave 는 회수로 자리가 날 때까지 기다린다("wave 순차 진행"의 문자 그대로).
            //   ★★막혔을 때의 처분이 origin 별로 갈린다 — **§4 mode-local ordered ReadFile
            //     digest 의 결정론이 걸린 자리**다:
            //       · **demand = `break`**. 건너뛰면 뒤의 demand record 가 먼저 나가고, 어느
            //         record 의 wave 가 먼저 회수됐는지(=I/O 타이밍)에 따라 **제출 열이 런마다
            //         달라진다** ⇒ digest 가 비결정이 된다. `break` 하면 demand 는 record 단위로
            //         엄격 순차라 제출 열이 all-at-once 판과 **동일**하고 결정론이 보존된다.
            //       · **prefetch = skip**(아래 continuation 절과 같은 처분). digest 는
            //         demand-origin child 만 담으므로(§4) prefetch 열은 digest 에 영향이 없고,
            //         건너뛰어야 뒤의 demand 가 막히지 않는다.
            const int32_t iop = g_mod.v3_io_param_val[3] > 0 ? g_mod.v3_io_param_val[3] : n_need;
            if (wave_from > 0 && cs_peek.remaining + n_need > iop) {
                if (g_mod.pending_ops[(size_t)oi_peek].origin == PIO_ORIGIN_PREFETCH) { qi++; continue; }
                break;
            }
            // ★continuation wave 의 credit 재검사(A0-4 r5 폐합) — **prefetch 만** 해당한다
            //   (credit 식은 prefetch 축이다). 실패하면 record 를 active 로 유지한 채 제출·C3
            //   계상 없이 건너뛰고, 이 pass 를 drain-before-refill 입력으로 표시한다.
            // ★S-4 — 1회성 실패 주입(프로덕션은 항상 0 · relaxed 비교 1회).
            bool cont_inj = false;
            if (wave_from > 0 &&
                g_mod.pending_ops[(size_t)oi_peek].origin == PIO_ORIGIN_PREFETCH &&
                g_mod.test_cont_credit_fail_next != 0) {
                g_mod.test_cont_credit_fail_next = 0;   // one-shot
                cont_inj = true;
            }
            if (wave_from > 0 &&
                g_mod.pending_ops[(size_t)oi_peek].origin == PIO_ORIGIN_PREFETCH &&
                (cont_inj || !pbv_credit_ok_locked(n_need))) {
                continuation_deferred = true;
                qi++;
                continue;
            }
            // ★A6 — 합침을 먼저 계획한다(§3-6). 서면 이 record 는 이웃 expert 들과 **한 묶음**으로
            //   나가고, 대구간은 슬롯이 아니라 coalesce 버퍼에 착지하므로 바운스 staging 은 쓰지
            //   않는다. 조건이 하나라도 안 서면 계획이 서지 않고 아래 단독 경로가 그대로 돈다.
            //   ★★R-1 **wave × coalescing 합성 규칙(명시 확정 — 리드 요구)**:
            //     ⓐ **합침은 단일 wave record 에만 성립한다**(`wave_from == 0 ∧ n_need == P_l`,
            //        즉 `W_l ≥ P_l`). 다중 wave 형상에서는 계획 자체를 세우지 않고 단독 경로로
            //        돈다 — 부분 제출과 섞으면 아래 ⓑ~ⓓ 가 전부 갈라진다.
            //     ⓑ **membership 계상**: 대구간 child 1건은 참가 record **전건**의 membership
            //        1건이다(§5-6 cardinality 가 `copies: src_kind=coalesce` 를 record 별로 세는
            //        것과 같은 단위). ⓐ 때문에 참가 record 의 membership 은 정확히 `P_l`(=1 wave)
            //        이고, 따라서 part cap(`≤ io_part_parallelism`)은 `W_l ≥ P_l` 조건이 이미
            //        보장한다 — 대구간 경로에는 별도 wave 개시 조건이 필요 없다.
            //     ⓒ **wave 상태**: 참가 record 는 제출 시점에 `wave_next = P_l`(완주)로 세운다
            //        (아래 ① 블록) ⇒ terminal 가드의 "미제출 wave 잔여" 검사가 서지 않는다.
            //     ⓓ **credit 축**: 대구간은 참가 record 수만큼의 예약을 물리 child 하나로 소화해
            //        결과가 **보수적**이다(실 in-flight ≤ 예약 · 상한 위반 방향이 아니다).
            //     ★현 동결 조합은 coalesce OFF 라 이 경로는 미발화지만, 코드 정합은 위 4항으로
            //       고정한다(위협모델 밖 2건과 상신 5 가 전부 이 접면에 있다).
            coalesce = (wave_from == 0) && (n_need == np_total) && v3_plan_coalesce_locked(oi_peek);
            // ★방어 — 계획에 head 가 없으면 큐 head 가 소비되지 않아 pump 가 같은 자리를 맴돈다.
            //   계획 산술이 그것을 보장하지만(창 클램프), 진행성은 감으로 두지 않는다.
            if (coalesce) {
                bool has_head = false;
                for (int32_t o : g_mod.v3_coal_plan.ops) if (o == oi_peek) { has_head = true; break; }
                if (!has_head) { v3_coal_reject(V3_COAL_R_RUN); coalesce = false; }
            }
            if (coalesce) {
                for (int32_t i = 0; i < g_mod.v3_coal_count && n_coal < n_need; i++)
                    if (!g_mod.v3_coal[(size_t)i].active) coal_slot[n_coal++] = i;
                if (n_coal < n_need) {           // 순간 자원 부족 — 이번 패스는 단독으로 낸다
                    v3_coal_reject(V3_COAL_R_RESOURCE);
                    coalesce = false;
                    n_coal = 0;
                }
            }
        }
        int32_t free_ch[MOE_DIRECT_QD_MAX];
        int32_t n_free = 0;
        for (size_t i = 0; i < g_mod.channels.size() && n_free < n_need; i++)
            if (!g_mod.channels[i].busy) free_ch[n_free++] = (int32_t)i;
        int32_t ci_free = (n_free >= n_need) ? free_ch[0] : -1;
        // ★★S1 — 자원 확정 단계(채널 확보 뒤). copy 의무는 **제출 전에 예약**한다 — 회수
        //   시점에 큐가 모자란 것을 알면 그때는 물러날 곳이 없다(구 판은 거기서 `fatal_abort`).
        //   계획의 `copy_qd/parts` run 상한은 **한 그룹이 혼자 들어갈 수 있는가**(필요조건)이고,
        //   이 크레딧은 **다른 생산자와 동시에 들어갈 수 있는가**(전역 안전조건)다 — 둘은 다른 조건이다.
        if (ci_free >= 0 && v3_active()) {
            const int32_t ml2 = g_mod.pending_ops[(size_t)g_mod.submit_q[qi]].mlpos;
            if (coalesce) {
                // 대구간 그룹의 의무 = 참가 record 수 × part 수(회수되면 동시에 큐로 간다).
                const int32_t need = (int32_t)g_mod.v3_coal_plan.ops.size() * n_need;
                if (!v3_copy_credit_take_locked(need)) {
                    g_mod.v3_copy_credit_stalls++;
                    v3_coal_reject(V3_COAL_R_RESOURCE);   // 합침만 포기 — 단독 경로는 그대로 돌다
                    coalesce = false;
                    n_coal   = 0;
                }
            }
            // ★A4 backpressure — staging 이 모자라면 **제출하지 않는다**(§6-4 ⑪). 억지로 밀면
            //   staging 을 copy terminal 전에 재사용하게 되고, 그게 §3-5 가 금지하는 바로 그것이다.
            // ★★R-1 — 바운스 staging 수요는 **이번 wave 가 낼 part 만** 센다(구 판은 층의 전
            //   part 를 셌다 — 그때는 전 part 가 한 번에 나갔으니 같은 값이었다).
            if (!coalesce) {
                const auto & pl2 = g_mod.layers[(size_t)ml2].parts;
                for (int32_t p = wave_from; p < wave_from + n_need && (size_t)p < pl2.size(); p++)
                    if (!pl2[(size_t)p].aligned) n_bounce_need++;
            }
            if (n_bounce_need > 0) {
                int32_t n_stg_free = 0;
                for (int32_t i = 0; i < g_mod.v3_staging_count; i++)
                    if (!g_mod.v3_staging_busy[(size_t)i]) n_stg_free++;
                if (n_stg_free < n_bounce_need) { g_mod.v3_backpressure_stalls++; ci_free = -1; }
                else if (!v3_copy_credit_take_locked(n_bounce_need)) {
                    // 바운스도 같은 큐를 쓴다 — 크레딧이 없으면 같이 물러선다.
                    g_mod.v3_copy_credit_stalls++;
                    g_mod.v3_backpressure_stalls++;
                    ci_free = -1;
                }
            }
        }
        if (ci_free < 0) {
            // ★demand_blocked_by_prefetch(§6) — "총 QD 만석 ∧ prefetch_inflight>0"으로 대기 시작.
            const int32_t oi_head = g_mod.submit_q[qi];
            PendingIo & head = g_mod.pending_ops[(size_t)oi_head];
            // ★★PB virtual A2 — 이 카운터의 **정의는 불변**이다(총 QD 만석 ∧ prefetch
            //   inflight>0 의 실대기 구간). virtual 에서 그 "prefetch inflight" 를 읽는 축이
            //   ②child 물리축일 뿐이다 — A0-4 의 admission predicate 와는 무관하며 predicate
            //   가 이 카운터를 자동으로 올리지도 않는다(r3 분리).
            const bool pf_busy = (g_mod.pb_inflight_prefetch > 0) ||
                                 (v3_active() && g_mod.v3_inflight_prefetch > 0);
            if (head.origin == PIO_ORIGIN_DEMAND && pf_busy &&
                head.pb_blocked_start == 0) {
                head.pb_blocked_start = ggml_moe_phaseb_now_qpc();
                // ★r1 M4 — 참여 집합을 **대기 시작 순간**에 고정한다. 종료 시 재스캔하면,
                //   깨어난 pass 가 signaled prefetch 를 먼저 retire 한 뒤 pump 하므로 실제
                //   blocker 가 이미 사라져 빈 집합이 기록된다(인과 누락 — r1 적발).
                if (PbOpStamp * hst = pb_op_stamp_at(oi_head))
                    pb_collect_participants_locked(hst->blocked_parts, /*include_ready=*/false,
                                                   /*inflight_only=*/true);
            }
            break;
        }
        // ---- ★A6 mode=virtual 대구간 제출(프리필 coalescing · §3-6) ----
        if (coalesce) {
            const V3CoalPlan & pl = g_mod.v3_coal_plan;
            const int32_t np    = pl.n_parts;
            const int32_t mlpos = pl.mlpos;
            progressed = true;
            // ① 참가 record 를 큐에서 걷어내고(-1) 각자의 상태·IO3 논리 행을 연다.
            //    ★copies 행이 record_seq 를 참조하므로 records 행이 **먼저** 서야 한다.
            uint64_t slice_total = 0;
            for (int32_t p = 0; p < np; p++)
                slice_total += g_mod.layers[(size_t)mlpos].parts[(size_t)p].slice_bytes;
            for (size_t k = 0; k < pl.ops.size(); k++) {
                const int32_t oi_p = pl.ops[k];
                for (size_t q = g_mod.submit_head; q < g_mod.submit_q.size(); q++)
                    if (g_mod.submit_q[q] == oi_p) { g_mod.submit_q[q] = -1; break; }
                PendingIo & op_p = g_mod.pending_ops[(size_t)oi_p];
                op_p.channel = -1;                  // 채널은 대구간이 소유한다(record 단위가 아니다)
                op_p.state   = PIO_SUBMITTED;
                disp_close_blocked_locked(oi_p);
                V3ChildIo & cs = g_mod.v3_child[(size_t)oi_p];
                cs = V3ChildIo{};
                cs.active = 1;
                cs.n_children = np;
                cs.remaining  = 0;                  // 물리 read 는 대구간이 소유 — 이 op 은 안 센다
                // ⑦ record READY = 그 record 의 **모든 part** scatter 완료 후. 의무를 제출
                //    시점에 올려 둔다(대구간 회수 전에는 remaining_copies 가 0 이 될 수 없다).
                cs.remaining_copies = np;
                // ★★R-1 — 합침은 **단일 wave 형상에서만** 성립하므로(위 계획 조건) 참가 record 는
                //   여기서 전 part 제출을 마친 것이다. `wave_next` 를 완주로 세워 두지 않으면
                //   `v3_maybe_terminal_locked` 의 "미제출 wave 잔여" 가드가 영원히 걸린다.
                cs.wave_next        = np;
                cs.bytes_expected   = slice_total;  // ⑥ scatter 는 exact slice — 브라켓은 안 온다
                // ★★PB virtual A1 — 이 참가 record 의 wave 개시(A0-3 "첫 child submit"):
                //   reservation→inflight 전환. 대구간은 참가 record 수만큼의 예약을 물리
                //   child 하나로 소화하므로 결과는 **보수적**(실 in-flight ≤ 예약)이다.
                pbv_rec_note_first_submit_locked(oi_p);
                if ((size_t)oi_p < g_mod.io3_op_row.size())
                    g_mod.io3_op_row[(size_t)oi_p] = io3_open_record_locked(oi_p);
                g_mod.v3_coal_records++;
            }
            const int32_t leader = pl.ops[0];
            // ② part 마다 대구간 1건을 건다.
            uint8_t outcome_c[64];
            DWORD   got_c[64];
            int32_t chan_c[64];
            for (int32_t p = 0; p < np; p++) {
                const int32_t ci = coal_slot[p];
                V3CoalRead & cr = g_mod.v3_coal[(size_t)ci];
                cr.active   = 1;
                cr.mlpos    = mlpos;
                cr.part     = p;
                cr.expert0  = pl.expert0;
                cr.offset   = pl.off[p];
                cr.len      = pl.len[p];
                cr.src_head = pl.head[p];
                cr.pending_copies = 0;
                cr.io3_copy_row0  = -1;
                cr.ops.assign(pl.ops.begin(), pl.ops.end());
                if (g_mod.v3_coal_group_seq == 0xFFFFFFFFu) g_mod.v3_coal_group_seq = 0;   // NA 회피
                cr.group_id = g_mod.v3_coal_group_seq++;
                g_mod.v3_coal_groups++;

                const int32_t chan = free_ch[p];
                chan_c[p] = chan;
                IoChannel & cch = g_mod.channels[(size_t)chan];
                const PartInfo & pi = g_mod.layers[(size_t)mlpos].parts[(size_t)p];
                g_mod.v3_chan_staging[(size_t)chan] = -1;
                g_mod.v3_chan_coal[(size_t)chan]    = ci;
                g_mod.v3_chan_len[(size_t)chan]     = cr.len;
                g_mod.v3_chan_op[(size_t)chan]      = leader;   // 핸들 되짚기·취소 스캔의 기준
                g_mod.v3_chan_part[(size_t)chan]    = p;
                ZeroMemory(&cch.ov, sizeof(cch.ov));
                cch.ov.hEvent     = cch.ev;
                cch.ov.Offset     = (DWORD)(cr.offset & 0xFFFFFFFFu);
                cch.ov.OffsetHigh = (DWORD)(cr.offset >> 32);
                ResetEvent(cch.ev);
                cch.op_index = leader;
                g_mod.io_submit_ops++;
                g_mod.v3_children_submitted++;
                outcome_c[p] = 0;   // 0=pending 1=sync 2=fail
                got_c[p]     = 0;
                // ★§5-5 submit 전환 ①② — 사건 seq·QPC 발급 + children/copies 행 materialize 를
                //   ReadFile **앞에서** 끝낸다.
                const bool coal_prefetch =
                    (g_mod.pending_ops[(size_t)leader].origin == PIO_ORIGIN_PREFETCH);
                // ★★PB virtual A1 — ②축(child 물리축) 증가는 IO3 방출부가 아니라 **여기**다.
                v3_child_inflight_add_locked(coal_prefetch);
                int32_t copy_row0 = -1;
                const int32_t io3_row = io3_coal_submit_locked(cr, &copy_row0);
                cr.io3_copy_row0 = copy_row0;
                if ((size_t)chan < g_mod.io3_chan_row.size()) {
                    g_mod.io3_chan_row[(size_t)chan]      = io3_row;
                    g_mod.io3_chan_copy_row[(size_t)chan] = -1;   // 대구간의 copies 는 cr 이 소유
                }
                const bool demand_child =
                    (g_mod.pending_ops[(size_t)leader].origin != PIO_ORIGIN_PREFETCH);
                if (demand_child)
                    v3_readfile_digest_add_locked((uint16_t)g_mod.moe_layers[(size_t)mlpos],
                                                  (uint16_t)cr.expert0, (uint8_t)p,
                                                  (uint8_t)pi.source_index, cr.offset, cr.len);
                if (ReadFile(k2fac_source_handle_locked((size_t)pi.source_index), v3_coal_buf(ci),
                             cr.len, nullptr, &cch.ov)) {
                    cch.busy = true;
                    outcome_c[p] = 1;
                    // ★★A3 — 대구간 child 1건은 **참가 record 전건**의 성공 제출이다(A0-3).
                    for (int32_t o : cr.ops) pbv_note_child_issued_locked(o);
                    v3_child_submit_outcome_locked(io3_row, /*pending=*/false, demand_child);
                } else if (GetLastError() == ERROR_IO_PENDING) {
                    cch.busy = true;
                    for (int32_t o : cr.ops) pbv_note_child_issued_locked(o);
                    v3_child_submit_outcome_locked(io3_row, /*pending=*/true, demand_child);
                } else {
                    outcome_c[p] = 2;
                    v3_child_submit_outcome_locked(io3_row, /*pending=*/false, demand_child);
                }
            }
            // ★★A3 — ③축 확정을 회수(③단계)보다 앞에 둔다(단독 경로와 같은 사유).
            for (int32_t o : pl.ops) pbv_flush_issued_locked(o);
            // ③ 동기 완료·제출 실패 회수. 대구간 회수는 참가 record **전건**을 처분하므로
            //    op 이 여기서 해제될 수 있다 — 아래에서 op 을 만지지 않는다.
            for (int32_t p = 0; p < np; p++) {
                if (outcome_c[p] == 0) continue;
                IoChannel & cch = g_mod.channels[(size_t)chan_c[p]];
                bool ok = false;
                if (outcome_c[p] == 1) {
                    const int32_t sidx = g_mod.layers[(size_t)mlpos].parts[(size_t)p].source_index;
                    const DWORD want = (DWORD)g_mod.v3_chan_len[(size_t)chan_c[p]];
                    ok = GetOverlappedResult(k2fac_source_handle_locked((size_t)sidx), &cch.ov, &got_c[p], TRUE) &&
                         got_c[p] == want;
                    g_mod.stat_sync_terminal++;
                }
                ResetEvent(cch.ev);
                cch.busy = false;
                cch.op_index = -1;
                v3_coal_child_retire_locked(coal_slot[p], chan_c[p], ok, got_c[p],
                                            /*was_pending=*/false, /*cancel=*/false);
            }
            qi++;
            skip_consumed();
            continue;
        }
        const int32_t oi = g_mod.submit_q[qi];
        PendingIo & op = g_mod.pending_ops[(size_t)oi];
        const bool prefetch = (op.origin == PIO_ORIGIN_PREFETCH);
        IoChannel & ch = g_mod.channels[(size_t)ci_free];
        ch.op_index = oi;
        op.channel  = ci_free;
        op.state    = PIO_SUBMITTED;
        progressed  = true;
        // ★blocked 구간 마감 — 이 demand op이 대기했다면 그 µs를 자기 행에 귀속한다.
        disp_close_blocked_locked(oi);
        // ---- ★A3 mode=virtual child 팬아웃 제출 ----
        if (v3_active()) {
            const int32_t np_total = v3_parts_of(op.mlpos);
            const int32_t expert   = (int32_t)(op.m.record_idx % g_mod.n_expert);
            V3ChildIo & cs = g_mod.v3_child[(size_t)oi];
            // ★★R-1 — **이번 wave** 가 낼 child 수와 그 part 구간. `np` 는 이제 "이 record 의 전
            //   part 수"가 아니라 **이번 wave 의 폭**이다(아래 루프는 전부 wave 로컬 인덱스로 돈다).
            const int32_t np    = n_need;
            const int32_t p_beg = wave_from;
            const bool first_wave = (p_beg == 0);
            if (first_wave) {
                cs = V3ChildIo{};
                cs.active = 1; cs.n_children = np_total; cs.remaining = 0;
                // ★★PB virtual A1 — wave 개시(A0-3 "첫 child submit"): 이 record 가 admission 에서
                //   선점해 둔 `W_l` child credit 예약을 물리 in-flight 로 전환한다. 아래 child 루프가
                //   ②축을 child 단위로 올리므로 post-state 는 **불변**(reserved−W · inflight+W)이다.
                pbv_rec_note_first_submit_locked(oi);
                // ★A5 — 이 record 채움 시도의 IO3 논리 행을 열다(§5-3 · 한 generation = 1행).
                if ((size_t)oi < g_mod.io3_op_row.size()) g_mod.io3_op_row[(size_t)oi] = io3_open_record_locked(oi);
            }
            // ★후속 wave 도 in-flight child 를 늘린다 — records 행·①축은 첫 wave 가 이미 열었다.
            cs.remaining  += np;
            cs.wave_next   = p_beg + np;
            g_mod.v3_waves_submitted++;   // ★★R-1 — wave 기계 발화의 직접 증거
            // ── 1단계: 전 child 를 건다. 완료 처분은 2단계로 미룬다 — 제출 도중에 terminal 이
            //    나면(op 해제) 남은 child 를 못 걸기 때문이다.
            enum : uint8_t { CH_PENDING = 0, CH_SYNC = 1, CH_FAIL = 2 };
            uint8_t  outcome[MOE_DIRECT_QD_MAX];
            DWORD    got[MOE_DIRECT_QD_MAX];
            // ★테스트 전용 one-shot — part→채널 배정을 역순으로 돌린다. 회수 스캔은 채널 인덱스
            //   오름차순이므로, 이 배정이 곧 **part 순서와 어긋난 완료 순서**(§6-4 ⑩ out-of-order)다.
            int32_t chan_of[MOE_DIRECT_QD_MAX];
            const bool rev = (g_mod.v3_test_reverse_children != 0);
            if (rev) g_mod.v3_test_reverse_children = 0;
            for (int32_t p = 0; p < np; p++) chan_of[p] = rev ? free_ch[np - 1 - p] : free_ch[p];
            // ★테스트 전용 one-shot — 지정 part 의 child 를 실패로 강등(§6-4 ⑩ short/error).
            const int32_t fail_part = g_mod.v3_test_fail_child;
            if (fail_part >= 0) g_mod.v3_test_fail_child = -1;
            // ★R3 one-shot — 지정 part 만 동기 완료 경로로(혼합 발화). 프로덕션 0.
            const int32_t sync_part = g_mod.v3_test_sync_child;
            if (sync_part >= 0) g_mod.v3_test_sync_child = -1;
            // ★R-1 — 회수 열은 **record 단위**다. 첫 wave 에서만 비우고 후속 wave 는 이어 붙인다
            //   (그래야 다중 wave record 의 전체 회수 순서를 한 열로 단언할 수 있다).
            if (first_wave) g_mod.v3_retire_order.clear();
            for (int32_t p = 0; p < np; p++) {
                // ★★R-1 — `p` 는 **wave 로컬** 인덱스이고 실제 part 인덱스는 `pi_abs` 다.
                //   채널·outcome 배열은 wave 로컬, 기하·wire·테스트 훅은 절대 part 로 간다.
                const int32_t pi_abs = p_beg + p;
                IoChannel & cch = g_mod.channels[(size_t)chan_of[p]];
                V3ChildGeom gm{};
                // ★A4 — 비4K part 는 staging 1칸을 잡고 거기로 읽는다(§3-5). 확보 실패는
                //   제출 측 backpressure 로 처리해야 하므로 여기까지 오면 안 된다(아래 사전 확인).
                const int32_t stg = g_mod.layers[(size_t)op.mlpos].parts[(size_t)pi_abs].aligned
                                    ? -1 : v3_staging_take_locked();
                if (!v3_child_geom_locked(op.mlpos, expert, pi_abs, op.m.slot, stg, gm))
                    fatal_abort("virtual child geometry is unresolvable (mlpos=%d part=%d staging=%d)",
                                (int)op.mlpos, (int)pi_abs, (int)stg);
                g_mod.v3_chan_staging[(size_t)chan_of[p]] = gm.bounce ? gm.staging : -1;
                g_mod.v3_chan_len[(size_t)chan_of[p]]     = gm.len;   // ★R10 — 실제 제출 길이
                ZeroMemory(&cch.ov, sizeof(cch.ov));
                cch.ov.hEvent     = cch.ev;
                cch.ov.Offset     = (DWORD)(gm.offset & 0xFFFFFFFFu);
                cch.ov.OffsetHigh = (DWORD)(gm.offset >> 32);
                ResetEvent(cch.ev);
                cch.op_index = oi;
                g_mod.v3_chan_op[(size_t)chan_of[p]]   = oi;
                g_mod.v3_chan_part[(size_t)chan_of[p]] = pi_abs;
                g_mod.io_submit_ops++;
                g_mod.v3_children_submitted++;
                outcome[p] = CH_PENDING;
                got[p]     = 0;
                // ★A5 §5-5 submit 전환 ①② — 사건 seq·QPC 발급 + children/copies 행
                //   materialize 를 **ReadFile 호출 전에** 끝낸다(호출 성공 후 발급 금지).
                // ★R12 — 표는 IO3 on 에서만 할당된다(기본 off = 자료구조 0). 그래서 여기서
                //   경계를 본다 — 행 인덱스는 아래 결과 반영에도 쓰므로 지역 변수로 든다.
                // ★F1 — copies 행도 **여기서** 채널에 건다. 이 결속이 없으면 회수 측
                //   (`v3_child_retire_locked`)이 -1 을 집어 `job.io3_copy_row` 가 비고,
                //   enqueue/start/done qpc·worker·status 가 한 줄도 안 찍힌다(행은 남지만
                //   내용이 없어 wire 는 자기일관해 보인다). 바로 위 `io3_chan_row` 결속과 대칭.
                // ★★PB virtual A1 — ②축(child 물리축) 증가는 IO3 방출부가 아니라 **여기**다.
                //   IO3 off 런에서도 admission credit 식이 실제 in-flight 를 본다.
                v3_child_inflight_add_locked(op.origin == PIO_ORIGIN_PREFETCH);
                int32_t io3_copy_row_p = -1;
                const int32_t io3_row_p = io3_child_submit_locked(oi, pi_abs, gm.bounce, gm.offset, gm.len,
                                                                  &io3_copy_row_p);
                if ((size_t)chan_of[p] < g_mod.io3_chan_row.size()) {
                    g_mod.io3_chan_row[(size_t)chan_of[p]]      = io3_row_p;
                    g_mod.io3_chan_copy_row[(size_t)chan_of[p]] = io3_copy_row_p;
                }
                const bool demand_child = (op.origin != PIO_ORIGIN_PREFETCH);
                // ★R12 — §4 mode-local digest/ops(제출 순서·demand-origin). IO3 on/off 무관.
                if (demand_child)
                    v3_readfile_digest_add_locked((uint16_t)g_mod.moe_layers[(size_t)op.mlpos],
                                                  (uint16_t)expert, (uint8_t)pi_abs,
                                                  (uint8_t)g_mod.layers[(size_t)op.mlpos].parts[(size_t)pi_abs].source_index,
                                                  gm.offset, gm.len);
                if (pi_abs == fail_part) {
                    // 강등 child — 제출 자체를 하지 않고 실패 회수 대상으로 둔다(실 저장장치에서
                    // 부분 read 를 재현할 수 없기 때문의 창구 · 프로덕션 0).
                    outcome[p] = CH_FAIL;
                    continue;
                }
                if (ReadFile(gm.h, gm.dest, gm.len, nullptr, &cch.ov)) {
                    cch.busy = true;   // 동기 완료도 아래 2단계가 GetOverlappedResult 로 닫는다
                    outcome[p] = CH_SYNC;
                    pbv_note_child_issued_locked(oi);   // ★★A3 — 성공 제출 child ≥ 1(A0-3)
                    // ★R12 — 인라인 완료 = §5-4 b0 immediate(0). 계수만 세운다.
                    v3_child_submit_outcome_locked(io3_row_p, /*pending=*/false, demand_child);
                } else if (GetLastError() == ERROR_IO_PENDING) {
                    cch.busy = true;
                    pbv_note_child_issued_locked(oi);   // ★★A3 — 커널 접수도 성공 제출이다
                    // ★R12 — 커널이 비동기로 접수했다 = §5-4 b0 pending(1). ★아래 `sync_part`
                    //   훅이 그 자리에서 완료를 기다려 **회수 경로**를 동기로 밀더라도 이 child 의
                    //   I/O 접수 사실은 pending 이다(회수 경로 라벨과 제출 접수 라벨은 다른 축).
                    v3_child_submit_outcome_locked(io3_row_p, /*pending=*/true, demand_child);
                    // ★R3 one-shot — 이 child 만 그 자리에서 완료를 기다려 **동기 분기**로 민다.
                    //   그래야 한 op 안에서 sync·async 가 실제로 섞인다(bin `test_force_sync_next`
                    //   와 같은 기제 — 이 저장 구성에서는 즉시 완료가 자연 발생하지 않는다).
                    if (pi_abs == sync_part) {
                        WaitForSingleObject(cch.ev, INFINITE);
                        outcome[p] = CH_SYNC;
                    }
                } else {
                    // 제출 실패 child — 완료 통지가 오지 않으므로 여기서 실패 회수 대상이 된다.
                    outcome[p] = CH_FAIL;
                    // ★R12 — 접수되지 않았으므로 pending 이 아니다(b0=0 · §5-4).
                    v3_child_submit_outcome_locked(io3_row_p, /*pending=*/false, demand_child);
                }
            }
            // ★★PB virtual A3 — ③축 `issued` 확정은 **2단계(회수)보다 앞**이다: 2단계에서
            //   op terminal 이 설 수 있고, terminal 은 never-issued 만 `submit_failed` 로
            //   닫아야 하기 때문이다(둘의 배타는 `disposed` 래치가 잡는다).
            pbv_flush_issued_locked(oi);
            // ── 2단계: 동기 완료·제출 실패를 회수한다. remaining 이 0 이 되는 마지막 1건만
            //    op terminal 로 간다(v3_child_retire_locked 가 그 판정을 소유).
            bool terminated = false;
            for (int32_t p = 0; p < np && !terminated; p++) {
                if (outcome[p] == CH_PENDING) continue;
                IoChannel & cch = g_mod.channels[(size_t)chan_of[p]];
                bool ok = false;
                if (outcome[p] == CH_SYNC) {
                    const PartInfo & pi = g_mod.layers[(size_t)op.mlpos].parts[(size_t)(p_beg + p)];
                    const int32_t sidx  = pi.source_index;
                    // ★★R10 — **그 child 가 실제로 제출한 길이**와 대조한다(region 이 아니다).
                    const DWORD want = (DWORD)g_mod.v3_chan_len[(size_t)chan_of[p]];
                    ok = GetOverlappedResult(k2fac_source_handle_locked((size_t)sidx), &cch.ov, &got[p], TRUE) &&
                         got[p] == want;
                    g_mod.stat_sync_terminal++;
                }
                ResetEvent(cch.ev);
                cch.busy = false;
                cch.op_index = -1;
                terminated = v3_child_retire_locked(oi, chan_of[p], ok, got[p], /*was_pending=*/false);
            }
            // ★★R-1 — 큐 처분: 전 part 를 내보냈으면 자리를 비우고(-1), 아직 남았으면 **그 자리에
            //   둔 채** 다음 pass 로 넘긴다(부분 제출 = 정상 상태). 다음 wave 는 part cap 이
            //   허락할 때(= 이번 wave 가 회수될 때) 열리므로, 그 회수가 만드는 turn 이 곧
            //   다음 wave 의 개시 신호다 — 진행성은 채널 완료가 보장한다.
            //   ★`terminated` 면 op 은 이미 해제됐다(실패/취소로 남은 wave 를 버린 경우) —
            //     그 자리도 비워야 abort drain 이 죽은 인덱스를 다시 종결하지 않는다.
            if (terminated || cs.wave_next >= np_total) g_mod.submit_q[qi] = -1;
            qi++;
            skip_consumed();
            continue;   // pb_on_submit_locked 등 bin 전용 후속은 virtual 에 해당 없음(계측=A5)
        }
        // ★★R-1 — bin 에는 wave 가 없다(record 1건 = 물리 read 1건) ⇒ 이 자리에서 큐를 소비한다.
        //   구 판은 head 를 `submit_head++` 로 집으며 소비했는데, 스캔 커서 도입으로 소비를 명시
        //   지점으로 옮겼다(bin 거동 불변 — 같은 순서로 정확히 1칸씩 소비한다).
        g_mod.submit_q[qi] = -1;
        qi++;
        skip_consumed();
        const int rc = io_submit_locked(op.m, ch, op.mlpos, op.m.pa_batch_order, op.origin,
                                        /*fail_soft=*/prefetch);
        if (rc < 0) {
            // ★prefetch fail-soft(§5·§6 pb_submit_failed) — slot 원복·unclaim·pb_io 미기록.
            ch.op_index = -1;
            op.channel  = -1;
            k2fac_slot_release_owner_locked(op.m.slot, op.m.record_idx);   // ★5c-T §1
            g_mod.slot_pending[(size_t)op.m.slot] = 0;
            if (op.m.record_idx >= 0 && g_mod.pending_by_record[(size_t)op.m.record_idx] == oi)
                g_mod.pending_by_record[(size_t)op.m.record_idx] = -1;
            // ★r1 M3 ② — 실 제출 실패가 확정된 지점. 보류 레코드를 SUBMIT_FAILED 로 닫는다.
            pb_flush_admit_rec_locked(oi, GGML_MOE_PB_D_SUBMIT_FAILED, GGML_MOE_PHASEB_NA_U64);
            pb_dispose_rank_locked(pb_find_row_locked(op.pb_graph_seq, op.mlpos), op.intent_seq,
                                   op.pb_rank, GGML_MOE_PB_D_SUBMIT_FAILED, op.pb_request_epoch);
            pio_release_locked(oi);
            continue;
        }
        pb_on_submit_locked(op, oi);
        if (rc == 0 && op.pb_io_index >= 0) g_mod.pb_ios[(size_t)op.pb_io_index].flags |= 0x01; // pending
        if (prefetch) {
            // ★r1 M3 ② — 제출 성공 확정. 보류 레코드를 ISSUED 로 닫고 submit event 와 조인한다.
            pb_flush_admit_rec_locked(oi, GGML_MOE_PB_D_ISSUED, op.pb_submit_event_seq);
            pb_dispose_rank_locked(pb_find_row_locked(op.pb_graph_seq, op.mlpos), op.intent_seq,
                                   op.pb_rank, GGML_MOE_PB_D_ISSUED, op.pb_request_epoch);
        }
        bool sync_done = (rc == 1);
        // ★selftest 전용 one-shot 강제 경로 — ReadFile TRUE(즉시 완료)는 ReadFile API 계약상
        //   실도달 경로지만 저장 구성에 따라 재현되지 않는다(이 기기: NVMe+NO_BUFFERING에서
        //   immediate=0). armed면 완료를 그 자리에서 기다린 뒤 **아래 sync_done 분기 코드를
        //   그대로** 통과시켜 그 분기의 이벤트 규율을 회귀로 잠근다. 프로덕션은 항상 OFF.
        if (!sync_done && g_mod.test_force_sync_next) {
            g_mod.test_force_sync_next = 0; // one-shot
            WaitForSingleObject(ch.ev, INFINITE);
            sync_done = true;
        }
        if (sync_done) {
            // ★동기 완료도 submit→terminal 계측 전량(§2) — 그 자리에서 회수.
            DWORD n = 0;
            if (!GetOverlappedResult(k2fac_data_handle_locked(), &ch.ov, &n, TRUE))
                fatal_abort("GetOverlappedResult failed err=%lu offset=%llu",
                            (unsigned long)GetLastError(), (unsigned long long)op.m.file_offset);
            // ★동기 완료도 커널이 event를 signaled로 올린다. idle 채널의 ev도 고정 wait 배열
            //   (handle 1..QD)에 있으므로 여기서 내리지 않으면 WaitForMultipleObjects가 즉시
            //   반환하는 무한 hot-loop가 된다 — async retire와 대칭으로 idle 전환 전에 내린다.
            ResetEvent(ch.ev);
            ch.busy = false;
            ch.op_index = -1;
            g_mod.stat_sync_terminal++;
            const bool bytes_ok = (n == (DWORD)g_mod.layers[(size_t)op.mlpos].stride_bytes);
            disp_op_terminal_locked(oi, n, /*was_pending=*/false,
                                    (bytes_ok || !prefetch) ? GGML_MOE_PB_T_SUCCESS_PUBLISHED
                                                            : GGML_MOE_PB_T_ERROR_UNPUBLISHED);
        }
    }
    if (g_mod.submit_head >= g_mod.submit_q.size() && !g_mod.submit_q.empty()) {
        g_mod.submit_q.clear();
        g_mod.submit_head = 0;
    }
    // ★★R-1 — 이 pass 의 continuation 재검사 결과를 admission 이 읽을 자리에 넘긴다
    //   (A0-4 drain-before-refill: "그 turn 에 continuation 재검사가 1건이라도 실패했으면
    //   신규 prefetch admission 을 생략"). pump 는 admission 보다 **앞**에 돌므로 같은 turn 이다.
    if (continuation_deferred) g_mod.pbv_cont_defer_events++;   // ★S-4
    g_mod.pbv_cont_deferred = continuation_deferred;
    // ★PA io_qd 피드는 demand-origin outstanding만 계수한다(§2.3 스코프 보존).
    if (progressed && pa_on) ggml_moe_phasea_io_qd(g_mod.pb_inflight_demand, ggml_moe_phasea_now_qpc());
    return progressed;
}

// signaled 채널을 **channel index 순**으로 회수(완료 순서 무관 — 최종 상태 결정론 보존).
bool disp_retire_signaled_locked() {
    bool progressed = false;
    const bool pa_on = g_mod.phasea_on;
    if (g_mod.test_hold_retire) return false;   // ★테스트 전용 — in-flight 고정(픽스처 창구)
    for (size_t i = 0; i < g_mod.channels.size(); i++) {
        IoChannel & ch = g_mod.channels[i];
        if (!ch.busy) continue;
        if (WaitForSingleObject(ch.ev, 0) != WAIT_OBJECT_0) continue;
        const int32_t oi = ch.op_index;
        PendingIo & op = g_mod.pending_ops[(size_t)oi];
        const bool prefetch = (op.origin == PIO_ORIGIN_PREFETCH);
        DWORD n = 0;
        // ---- ★A3 mode=virtual child 회수 ----
        if (v3_active()) {
            const int32_t pidx = g_mod.v3_chan_part[i];
            const PartInfo & pi = g_mod.layers[(size_t)op.mlpos].parts[(size_t)pidx];
            // ★A6 — 대구간 child 는 op 1건이 아니라 참가 record 전건을 처분한다.
            const int32_t coal_i = g_mod.v3_chan_coal[i];
            if (coal_i >= 0) {
                const BOOL cok2 = GetOverlappedResult(k2fac_source_handle_locked((size_t)pi.source_index),
                                                      &ch.ov, &n, TRUE);
                const bool ok2 = (cok2 != FALSE) && (n == (DWORD)g_mod.v3_chan_len[i]);
                ch.busy = false;
                ch.op_index = -1;
                ResetEvent(ch.ev);
                v3_coal_child_retire_locked(coal_i, (int32_t)i, ok2, n, /*was_pending=*/true,
                                            /*cancel=*/false);
                progressed = true;
                continue;
            }
            // ★핸들은 그 child 가 읽은 shard 의 것이어야 한다(virtual 은 핸들이 여럿).
            const BOOL cok = GetOverlappedResult(k2fac_source_handle_locked((size_t)pi.source_index),
                                                 &ch.ov, &n, TRUE);
            // ★★R10 — 기대 바이트 = 그 child 의 **제출 길이**(비4K 는 region 과 다를 수 있다).
            const DWORD want_child = (DWORD)g_mod.v3_chan_len[i];
            // 취소된 child 는 ERROR_OPERATION_ABORTED 로 돌아온다 — 실패로 접수해 drain 을
            // 완주시킨다(래치는 이미 서 있으므로 상태를 덮지 않는다).
            const bool ok = (cok != FALSE) && (n == want_child);
            ch.busy = false;
            ch.op_index = -1;
            ResetEvent(ch.ev);
            v3_child_retire_locked(oi, (int32_t)i, ok, n, /*was_pending=*/true);
            progressed = true;
            continue;
        }
        const BOOL gok = GetOverlappedResult(k2fac_data_handle_locked(), &ch.ov, &n, TRUE);
        if (!gok && !prefetch)
            fatal_abort("GetOverlappedResult failed err=%lu", (unsigned long)GetLastError());
        // ★prefetch 오류/부분 read = fail-soft(§5) — terminal 판정은 GetOverlappedResult 결과로만.
        const bool bytes_ok = gok && (n == (DWORD)g_mod.layers[(size_t)op.mlpos].stride_bytes);
        ch.busy = false;
        ch.op_index = -1;
        ResetEvent(ch.ev);
        disp_op_terminal_locked(oi, n, /*was_pending=*/true,
                                (prefetch && !bytes_ok) ? GGML_MOE_PB_T_ERROR_UNPUBLISHED
                                                        : GGML_MOE_PB_T_SUCCESS_PUBLISHED);
        progressed = true;
        // ★PA io_qd 피드 = demand-origin outstanding만(§2.3).
        if (pa_on) ggml_moe_phasea_io_qd(g_mod.pb_inflight_demand, ggml_moe_phasea_now_qpc());
    }
    return progressed;
}

// ★R1 #3 ③ — graceful 경로: demand drain 완료 후 잔여 **prefetch-origin** in-flight만
//   CancelIoEx → terminal 회수한다(§0.1 CUT ⑤ "잔여 speculative CancelIoEx → 전 채널 terminal
//   회수"). demand는 여기 오지 않는다(③에서 이미 종결).
// ★★PB virtual A4 — 위 graceful 취소의 mode=virtual 판(§2-0 A0-4 joined 보호 · A0-3 단일
//   terminal). bin 판과 갈리는 지점만 적는다:
//     ①핸들이 shard 마다 다르다 ⇒ CancelIoEx 는 `v3_cancel_children_locked` 가 (op,part)→
//       source_index 로 되짚어 건다(그 함수가 이미 소유한 계약 — 여기서 재구현하지 않는다).
//     ②한 record 가 채널을 여러 개 문다 ⇒ 회수는 `v3_child_retire_locked` 로 합류시키고,
//       **마지막 child 가 회수되는 순간에만** op terminal 이 선다(채널마다 terminal 을 부르면
//       같은 op 을 N 번 종결해 pending 을 이중 반납한다 — abort 경로와 같은 사유).
//     ③대구간(coalesced)은 참가 record 전건이 speculative 일 때만 취소 대상이다 — 참가자
//       하나라도 waiter 를 달고 있으면 그 그룹은 정상 요청의 데이터를 나르는 중이다.
//   ★credit·pin 의 exactly-once 는 `pio_release_locked` 단일 초크포인트가 잡는다(A1).
bool disp_cancel_prefetch_v3_locked() {
    const size_t nch = g_mod.channels.size();
    std::vector<uint8_t> sel(nch, 0);
    // ── ①선별: 채널이 서비스하는 record 집합이 전부 speculative prefetch 인가 ──
    for (size_t i = 0; i < nch; i++) {
        const IoChannel & ch = g_mod.channels[i];
        if (!ch.busy) continue;
        bool all_spec = true;
        const int32_t coal_i = g_mod.v3_chan_coal[i];
        if (coal_i >= 0) {
            const V3CoalRead & cr = g_mod.v3_coal[(size_t)coal_i];
            if (cr.ops.empty()) all_spec = false;
            for (int32_t o : cr.ops) {
                const PendingIo & op = g_mod.pending_ops[(size_t)o];
                if (op.origin != PIO_ORIGIN_PREFETCH || !op.waiters.empty()) { all_spec = false; break; }
            }
        } else {
            const int32_t oi = ch.op_index;
            if (oi < 0) continue;
            const PendingIo & op = g_mod.pending_ops[(size_t)oi];
            all_spec = (op.origin == PIO_ORIGIN_PREFETCH) && op.waiters.empty();
            // ★V3P17 — prefetch 인데 waiter 때문에 빠진 건이 곧 **joined 보호가 발화한 자리**다.
            if (op.origin == PIO_ORIGIN_PREFETCH && !op.waiters.empty())
                g_mod.v3_cancel_excl_joined++;
        }
        if (all_spec) { sel[i] = 1; g_mod.v3_cancel_selected++; }
    }
    bool any = false;
    for (size_t i = 0; i < nch; i++) if (sel[i]) { any = true; break; }
    if (!any) return false;
    // ── ②취소 발행: 선별된 채널이 속한 record 의 **형제 전건**을 한 번에 끊는다 ──
    for (size_t i = 0; i < nch; i++) {
        if (!sel[i]) continue;
        const int32_t coal_i = g_mod.v3_chan_coal[i];
        if (coal_i >= 0) {
            for (int32_t o : g_mod.v3_coal[(size_t)coal_i].ops) {
                g_mod.stat_cancel++;
                v3_cancel_children_locked(o);   // 1회성 래치 — 참가자 반복 호출은 no-op
            }
        } else {
            g_mod.stat_cancel++;
            v3_cancel_children_locked(g_mod.channels[i].op_index);
        }
    }
    // ── ③회수 drain: CancelIoEx 성공 반환 ≠ terminal — ERROR_OPERATION_ABORTED 까지 회수 ──
    //   ★`v3_cancel_children_locked` 는 형제까지 끊으므로 선별 밖 채널도 abort 로 돌아올 수
    //   있다. 형제는 같은 record 소속이라 선별 결과와 집합이 같다(취소 대상 record 의 전
    //   채널) — 그래서 여기서는 선별 집합만 돌면 된다.
    for (size_t i = 0; i < nch; i++) {
        if (!sel[i]) continue;
        IoChannel & ch = g_mod.channels[i];
        if (!ch.busy) continue;                       // 형제 회수가 이미 닫았다
        const HANDLE h = v3_chan_handle_locked(i);
        DWORD n = 0;
        const BOOL ok = GetOverlappedResult(h, &ch.ov, &n, TRUE);
        const DWORD e = ok ? 0 : GetLastError();
        if (!ok && e != ERROR_OPERATION_ABORTED)
            fatal_abort("prefetch cancel drain GetOverlappedResult failed err=%lu", (unsigned long)e);
        const int32_t oi   = ch.op_index;
        const int32_t chan = (int32_t)i;
        const int32_t coal_i = g_mod.v3_chan_coal[i];
        ch.busy = false;
        ch.op_index = -1;
        ResetEvent(ch.ev);
        // ★success-after-cancel 도 publish 금지 — `cancel=true` 가 `cancel_mode` 를 세운다.
        if (coal_i >= 0) v3_coal_child_retire_locked(coal_i, chan, ok != FALSE, n,
                                                     /*was_pending=*/true, /*cancel=*/true);
        else             v3_child_retire_locked(oi, chan, ok != FALSE, n,
                                                /*was_pending=*/true, /*cancel=*/true);
    }
    return true;
}

bool disp_cancel_prefetch_inflight_locked() {
    if (v3_active()) return disp_cancel_prefetch_v3_locked();   // ★★PB virtual A4
    bool any = false;
    // ★R2 #2 — waiter(joined demand) 가 붙은 op 은 **취소 금지**다: 그 demand 는 이 op 의
    //   결과를 기다리는 정상 요청이므로 완료까지 대기해야 한다(speculative = waiter 無 만 취소 대상).
    auto cancelable = [](const IoChannel & ch) {
        if (!ch.busy || ch.op_index < 0) return false;
        const PendingIo & op = g_mod.pending_ops[(size_t)ch.op_index];
        return op.origin == PIO_ORIGIN_PREFETCH && op.waiters.empty();
    };
    for (auto & ch : g_mod.channels) {
        if (!cancelable(ch)) continue;
        g_mod.stat_cancel++;
        CancelIoEx(k2fac_data_handle_locked(), &ch.ov);
        any = true;
    }
    if (!any) return false;
    for (auto & ch : g_mod.channels) {
        if (!cancelable(ch)) continue;
        DWORD n = 0;
        const BOOL ok = GetOverlappedResult(k2fac_data_handle_locked(), &ch.ov, &n, TRUE);
        const DWORD e = ok ? 0 : GetLastError();
        if (!ok && e != ERROR_OPERATION_ABORTED)
            fatal_abort("prefetch cancel drain GetOverlappedResult failed err=%lu", (unsigned long)e);
        const int32_t oi = ch.op_index;
        ch.busy = false;
        ch.op_index = -1;
        ResetEvent(ch.ev);
        // ★success-after-cancel 도 publish 금지(§2 종료 시퀀스).
        disp_op_terminal_locked(oi, n, /*was_pending=*/true,
                                ok ? GGML_MOE_PB_T_CANCEL_SUCCESS_UNPUB
                                   : GGML_MOE_PB_T_CANCELLED_UNPUBLISHED);
    }
    return true;
}

// abort 종료 — 잔여 in-flight를 CancelIoEx하고 **terminal까지 회수**한 뒤(CancelIoEx 성공 반환
// ≠ terminal — GetOverlappedResult의 ERROR_OPERATION_ABORTED까지 drain) 전 waiter를 error로 깨운다.
// ★shutdown/abort 중 success-after-cancel도 publish하지 않고 ABSENT로 종결한다(§2).
void disp_abort_inflight_locked() {
    // ★§Z-⑥ A3 — virtual 은 ①채널마다 shard 핸들이 다르고 ②한 op 이 여러 채널을 문다.
    //   그래서 취소 핸들을 child 별로 되짚고, 완료는 **op 단위로 합류**시킨다. 채널마다
    //   `disp_op_terminal_locked` 를 부르면 같은 op 을 N 번 종결해 pending 을 이중 반납한다.
    const bool virt = v3_active();
    for (size_t i = 0; i < g_mod.channels.size(); i++) {
        if (!g_mod.channels[i].busy) continue;
        g_mod.stat_cancel++;
        CancelIoEx(virt ? v3_chan_handle_locked(i) : k2fac_data_handle_locked(), &g_mod.channels[i].ov);
    }
    for (size_t i = 0; i < g_mod.channels.size(); i++) {
        IoChannel & ch = g_mod.channels[i];
        if (!ch.busy) continue;
        const HANDLE h = virt ? v3_chan_handle_locked(i) : k2fac_data_handle_locked();
        DWORD n = 0;
        const BOOL ok = GetOverlappedResult(h, &ch.ov, &n, TRUE);
        const DWORD e = ok ? 0 : GetLastError();
        if (!ok && e != ERROR_OPERATION_ABORTED)
            fatal_abort("cancel drain GetOverlappedResult failed err=%lu", (unsigned long)e);
        const int32_t oi = ch.op_index;
        const int32_t chan = (int32_t)i;
        ch.busy = false;
        ch.op_index = -1;
        ResetEvent(ch.ev);
        // ★success-after-cancel도 publish 금지(§2 종료 시퀀스) — status 4로 구분 기록한다.
        if (virt && g_mod.v3_chan_coal[i] >= 0) {
            // ★A6 — 대구간은 참가 record 전건을 취소로 처분한다(어느 하나도 publish 로 가지 않는다).
            v3_coal_child_retire_locked(g_mod.v3_chan_coal[i], chan, ok != FALSE, n,
                                        /*was_pending=*/true, /*cancel=*/true);
        } else if (virt) {
            // 마지막 child 가 회수되는 순간에만 op terminal 이 선다(cancel_mode = publish 금지).
            v3_child_retire_locked(oi, chan, ok != FALSE, n, /*was_pending=*/true, /*cancel=*/true);
        } else {
            disp_op_terminal_locked(oi, n, /*was_pending=*/true,
                                    ok ? GGML_MOE_PB_T_CANCEL_SUCCESS_UNPUB
                                       : GGML_MOE_PB_T_CANCELLED_UNPUBLISHED);
        }
    }
    // 미제출(claimed) op도 전부 abort terminal로 종결.
    while (g_mod.submit_head < g_mod.submit_q.size()) {
        const int32_t oi = g_mod.submit_q[g_mod.submit_head++];
        if (oi < 0) continue;   // ★A6 — 합침이 이미 걷어간 자리
        // ★★R-1 — wave 기계에서는 **부분 제출 record 가 큐와 채널에 동시에 존재**한다. 위
        //   채널 drain 이 그 record 를 이미 종결(=`pio_release_locked`)했을 수 있으므로, 죽은
        //   인덱스를 다시 종결하면 pending 을 이중 반납한다. 살아 있는 것만 닫는다.
        if (!g_mod.pending_ops[(size_t)oi].in_use) continue;
        disp_op_terminal_locked(oi, 0, /*was_pending=*/false, GGML_MOE_PB_T_CANCELLED_UNPUBLISHED);
    }
    g_mod.submit_q.clear();
    g_mod.submit_head = 0;
}

// demand 계획 가능 여부 — ★PA on일 때는 acquire_enter/exit 문맥이 1개뿐이라(ggml-moe-phasea.h
// 동시성 계약) 동시 계획을 금지한다. pending 풀이 모자라도 다음 terminal까지 미룬다.
bool disp_can_plan_demand(int32_t n_ids) {
    if (g_mod.phasea_on && g_mod.demand_outstanding > 0) return false;
    if ((int32_t)g_mod.pending_free.size() < n_ids)      return false;
    return true;
}

void disp_handle_demand_locked(int32_t ci) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    g_mod.stat_demand_cmd++;
    const bool pa_on = g_mod.phasea_on;
    // ★B2 — dispatcher가 인지하는 "현재 epoch"(사전 처분 stale/too_late 판정 기준).
    g_mod.pb_cur_graph_seq = c.pb_graph_seq;
    DemandPlan & p = g_mod.disp_plan;
    const int rc = plan_demand_locked(p, c.mlpos, c.part_idx, c.ids.data(), c.n_ids, c.phase,
                                      pa_on, c.pa_t_enter, /*persistent=*/true,
                                      c.pb_request_epoch, c.pb_graph_epoch, c.pb_graph_seq, ci);
    if (rc != 0) {
        // ★admission reserve ③이 지키는 불변식 — 이 값은 0이어야 한다(§6 pb_demand_preflight_reject).
        ggml_moe_phaseb_count(GGML_MOE_PB_C_DEMAND_PREFLIGHT_REJECT, c.pb_request_epoch, 1);
        disp_complete(ci, -1);
        return;
    }

    const LayerInfo & L = g_mod.layers[(size_t)c.mlpos];
    c.pa_n_miss = (int32_t)p.miss.size();
    c.pa_bytes  = (uint64_t)p.miss.size() * L.stride_bytes;
    c.pa_batch_next = (uint32_t)p.miss.size();   // ★R1 #9 ④ retry 순번은 여기서 이어진다
    c.pa_open   = pa_on;
    c.deps      = 0;
    g_mod.demand_outstanding++;

    // ── ★B2 배타 3분할(§4·§6.1) — 요청 인덱스별로 ready|join|demand_read 중 정확히 하나 ──
    c.sp_ready = c.sp_join = c.sp_demand_read = c.sp_join_attempt = 0;
    c.req_join_op.assign((size_t)c.n_ids, -1);
    for (int32_t i = 0; i < c.n_ids; i++) {
        if (p.hit_slot[(size_t)i] >= 0)        c.sp_ready++;
        else if (p.req_join[(size_t)i] >= 0) { c.sp_join++; c.sp_join_attempt++;
                                               c.req_join_op[(size_t)i] = p.req_join[(size_t)i]; }
        else                                   c.sp_demand_read++;
    }
    // pb_sample 행(§4) — per (graph, layer) 1개. 이 (graph,layer)의 **첫**(canonical) acquire만
    // 3분할을 채운다(PA ledger canonical ACQ_CALL과 1:1 — §9 게이트 9-b).
    // ★canonical 판정(= too_late 표식)은 **PB 모드와 무관**하게 먼저 확정한다(C4′).
    c.pb_canonical = pb_mark_served_locked(c.pb_graph_seq, c.mlpos);
    c.pb_row = -1;
    // ★5c-T(부록 14.1) — 이 acquire 의 wait/frontier 래치 초기화. command 슬롯은 고정 64칸
    //   재사용이라 여기서 지우지 않으면 앞 acquire 의 래치가 다음 acquire 로 샌다(1 acquire
    //   1계수 위반). 위치 구속: `g_mod.mutex` 아래여야 한다(cq_alloc 은 q_mutex 라 경합).
    if (g_mod.pb_collect && ci >= 0 && ci < MOE_DIRECT_CONTROL_Q_CAP) {
        g_mod.pb_cmd_waited[(size_t)ci]   = 0;
        g_mod.pb_cmd_frontier[(size_t)ci] = 0;
    }
    if (c.pb_canonical) {
        // ★NO_SOURCE(layer 0 demand 행)는 capture 훅이 낼 수 없는 사유라 여기서 계수한다.
        if (c.phase != Phase::PREFILL && c.mlpos == 0)
            ggml_moe_phaseb_count_skip(GGML_MOE_PB_SKIP_NO_SOURCE, c.pb_request_epoch);
        const int32_t row = pb_row_locked(c.pb_graph_seq, c.pb_request_epoch, c.pb_graph_epoch,
                                          c.mlpos, c.phase == Phase::PREFILL ? 1 : 0);
        if (row >= 0) {
            ggml_moe_phaseb_sample & s = g_mod.pb_samples[(size_t)row];
            s.n_required     = (uint16_t)c.n_ids;
            s.n_ready        = (uint16_t)c.sp_ready;
            s.n_join         = (uint16_t)c.sp_join;
            s.n_demand_read  = (uint16_t)c.sp_demand_read;
            s.n_join_attempt = (uint16_t)c.sp_join_attempt;
            c.pb_row = row;
        }
    }

    // 새 유니크 miss → PendingIo 생성(첫 출현 순 = 제출 순 = ordered ReadFile digest 순).
    for (size_t k = 0; k < p.miss.size(); k++) {
        const int32_t oi = pio_alloc_locked();
        if (oi < 0) fatal_abort("pending pool exhausted (disp_can_plan_demand invariant violated)");
        PendingIo & op = g_mod.pending_ops[(size_t)oi];
        op.m = p.miss[k];
        op.m.pa_batch_order = (uint32_t)k;
        op.mlpos  = c.mlpos;
        op.phase  = c.phase;
        op.origin = PIO_ORIGIN_DEMAND;
        op.state  = PIO_CLAIMED;
        op.request_identity = c.seq;
        op.intent_seq = GGML_MOE_PHASEB_NA_U64;   // demand-origin = intent NA
        op.retry_of   = -1;
        op.pb_request_epoch = c.pb_request_epoch;
        op.pb_graph_epoch   = c.pb_graph_epoch;
        op.pb_graph_seq     = c.pb_graph_seq;
        op.pb_rank          = GGML_MOE_PHASEB_NA_U16;
        op.waiters.push_back(ci);
        g_mod.pending_by_record[(size_t)op.m.record_idx] = oi;
        g_mod.slot_pending[(size_t)op.m.slot] = 1;
        g_mod.submit_q.push_back(oi);
        c.deps++;
    }
    // in-flight join(C2) — 새 물리 제출 없음.
    for (int32_t oi : p.join_ops) {
        g_mod.pending_ops[(size_t)oi].waiters.push_back(ci);
        g_mod.stat_join++;
        c.deps++;
    }

    if (c.deps == 0) {
        // 전건 hit — 반환 직후 ptr() 전건 유효(lease는 ④에서 이미 반영됨).
        if (c.pa_open) { ggml_moe_phasea_acquire_exit(c.pa_n_miss, c.pa_bytes, ggml_moe_phasea_now_qpc()); c.pa_open = false; }
        ggml_moe_phaseb_note_demand_ready(c.pb_graph_seq,
                                          c.mlpos >= 0 && c.mlpos < (int32_t)g_mod.moe_layers.size()
                                              ? g_mod.moe_layers[(size_t)c.mlpos] : -1,
                                          ggml_moe_phaseb_now_qpc());
        pb_finish_demand_command_locked(c);
        g_mod.demand_outstanding--;
        metrics_maybe_periodic_locked();
        disp_complete(ci, 0);
    }
}

/* XBE-PIN-BEGIN disp-release-wrap */
void disp_handle_release_locked(int32_t ci) {
    // E-2b-2 (D12 catch 2): this runs on the DISPATCHER thread, so the caller-side catch in
    // ggml_moe_store_release cannot see an exception raised here. The PhaseA release hooks
    // below append to a vector, so bad_alloc is reachable with MOE_DIRECT_PHASEA on. Left
    // unsealed it would be std::terminate - the process still dies, but without the contract
    // terminal, the [moe-direct] FATAL line and the exit class that goes with it.
    try {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    g_mod.stat_release_cmd++;
    const bool pa_on = g_mod.phasea_on;
    if (pa_on) ggml_moe_phasea_release_enter(c.mlpos, c.part_idx, c.n_ids, ggml_moe_phasea_now_qpc());
    for (int32_t i = 0; i < c.n_ids; i++) {
        const int32_t eid = c.ids[(size_t)i];
        if (eid < 0 || eid >= g_mod.n_expert) continue;
        const int64_t record_idx = (int64_t)c.mlpos * g_mod.n_expert + eid;
        RecordEntry & re = g_mod.record_table[(size_t)record_idx];
        const int32_t si = re.slot_index.load(std::memory_order_relaxed);
        if (si < 0) continue;
        const int32_t prev = k2fac_slot_lease_return_locked(si);
        if (prev > 0)  xstore_note_lease_returned(1);   // E-2a lease-unit ledger (b)
        if (pa_on) ggml_moe_phasea_release_expert(eid, si, prev, prev > 0 ? prev - 1 : prev);
    }
    disp_complete(ci, 0); // ★동기 ack — 반환 전 lease 감소 완결
    } catch (const std::bad_alloc &) {
        fatal_abort("xstore: allocation failed in dispatcher release for cmd %d - fail-stop (contract section 2 row 5)",
                    (int) ci);
    }
}
/* XBE-PIN-END disp-release-wrap */

void disp_handle_query_locked(int32_t ci) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    g_mod.stat_query_cmd++;
    switch (c.qkind) {
        case QK_ABSENT_COUNT: {
            int32_t absent = 0;
            if (g_mod.sealed && c.mlpos >= 0 && c.mlpos < (int32_t)g_mod.moe_layers.size()) {
                for (int32_t i = 0; i < c.n_ids; i++) {
                    const int32_t eid = c.ids[(size_t)i];
                    if (eid < 0 || eid >= g_mod.n_expert) continue;
                    const int64_t record_idx = (int64_t)c.mlpos * g_mod.n_expert + eid;
                    const RecordEntry & re = g_mod.record_table[(size_t)record_idx];
                    const int32_t si = re.slot_index.load(std::memory_order_relaxed);
                    bool resident = false;
                    if (si >= 0) {
                        const Slot & s = g_mod.slots[(size_t)si];
                        resident = s.generation.load(std::memory_order_relaxed) == re.slot_generation.load(std::memory_order_relaxed)
                                && s.owner_record == record_idx;
                    }
                    if (!resident) absent++;
                }
            }
            c.q_out_i32_a = absent;
            break;
        }
        case QK_RECORD_SLOT: {
            int32_t slot = -1; uint64_t gen = 0;
            if (g_mod.sealed && c.q_arg_i64 >= 0 && c.q_arg_i64 < (int64_t)g_mod.record_table.size()) {
                const RecordEntry & re = g_mod.record_table[(size_t)c.q_arg_i64];
                slot = re.slot_index.load(std::memory_order_relaxed);
                gen  = re.slot_generation.load(std::memory_order_relaxed);
            }
            c.q_out_i32_a = slot;
            c.q_out_u64_a = gen;
            break;
        }
        case QK_SLOT_STATE: {
            int64_t owner = -1; uint64_t last_used = 0, gen = 0; int32_t lease = 0;
            if (g_mod.sealed && c.q_arg_i32 >= 0 && c.q_arg_i32 < g_mod.n_slots) {
                const Slot & s = g_mod.slots[(size_t)c.q_arg_i32];
                owner = s.owner_record;
                last_used = s.last_used;
                gen = s.generation.load(std::memory_order_relaxed);
                lease = s.lease.load(std::memory_order_relaxed);
            }
            c.q_out_i64   = owner;
            c.q_out_u64_a = last_used;
            c.q_out_u64_b = gen;
            c.q_out_i32_a = lease;
            break;
        }
        case QK_LEASE_STATS: {
            int32_t leased = 0; int64_t total = 0;
            for (const auto & s : g_mod.slots) {
                const int32_t l = s.lease.load(std::memory_order_relaxed);
                if (l > 0) { leased++; total += l; }
            }
            c.q_out_i32_a = leased;
            c.q_out_i64   = total;
            break;
        }
    }
    disp_complete(ci, 0);
}

// intent 결속 — 헤드 intent의 pb_sample 행(target 층)을 확보하고 intent 필드를 채운다(§4 ⓐ).
int32_t pb_bind_intent_row_locked(const ggml_moe_phaseb_intent & it) {
    const int32_t row = pb_row_locked(it.graph_seq, it.request_epoch, it.graph_epoch,
                                      it.target_mlpos, /*is_prefill=*/0);
    if (row < 0) return -1;
    ggml_moe_phaseb_sample & s = g_mod.pb_samples[(size_t)row];
    s.flags |= 0x01;                                   // intent_present
    s.elig_skip_reason = GGML_MOE_PB_SKIP_NONE;
    s.source_layer     = it.source_layer;
    s.k_effective      = (uint8_t)it.k;
    s.intent_ordinal   = it.intent_ordinal;
    for (int i = 0; i < GGML_MOE_PHASEB_TOPK; i++) s.intent_ids[i] = it.ids[i];
    s.capture_qpc      = it.capture_qpc;
    s.predict_end_qpc  = it.predict_end_qpc;
    return row;
}

// A0-5 정적 상한(등식·층별):
//   `N_cap(l) = min(N, floor(io_qd_prefetch_child_max / W_l),
//                      floor((qd_total − io_qd_demand_reserve) / W_l))`
// ★참고 실효(현 후보 상수 qwen: QD8·reserve4·child_max4·P_l=W_l=3) = **1**(사양 §2-0 A0-5 말미).
int32_t pbv_ncap_locked(int32_t mlpos, int32_t n_stamp) {
    const int32_t w = pbv_wave_width(mlpos);
    if (w <= 0 || n_stamp <= 0) return 0;
    const int32_t qd_total  = g_mod.v3_io_param_val[0];
    const int32_t reserve   = g_mod.v3_io_param_val[1];
    const int32_t child_max = g_mod.v3_io_param_val[2];
    int32_t cap = n_stamp;
    const int32_t c1 = child_max / w;
    const int32_t c2 = (qd_total > reserve) ? (qd_total - reserve) / w : 0;
    if (c1 < cap) cap = c1;
    if (c2 < cap) cap = c2;
    return cap < 0 ? 0 : cap;
}

// A0-4 admission credit 식 — **둘 다** 성립해야 admit.
//   ①`free_total = qd_total − child_inflight_demand − child_inflight_prefetch
//                          − child_reserved_prefetch ≥ W_l`            (실제 빈 credit)
//   ②`child_inflight_prefetch + child_reserved_prefetch + W_l
//        ≤ min(io_qd_prefetch_child_max, qd_total − io_qd_demand_reserve)`
//                                                                      (prefetch post-state)
// ★r5 차단 폐합분: ②가 reserve·child_max 를 **동시에** 지키므로 IO3 §5-6 cap 검산과 정합한다.
bool pbv_credit_ok_locked(int32_t w) {
    if (w <= 0) return false;
    const int32_t qd_total  = g_mod.v3_io_param_val[0];
    const int32_t reserve   = g_mod.v3_io_param_val[1];
    const int32_t child_max = g_mod.v3_io_param_val[2];
    const int32_t free_total = qd_total - g_mod.v3_inflight_demand - g_mod.v3_inflight_prefetch
                             - g_mod.pbv_child_reserved;
    if (free_total < w) return false;
    const int32_t post = g_mod.v3_inflight_prefetch + g_mod.pbv_child_reserved + w;
    int32_t bound = child_max;
    const int32_t b2 = qd_total - reserve;
    if (b2 < bound) bound = b2;
    return post <= bound;
}

// A0-4 **demand 상태 predicate(금지가 이긴다)**: `blocked ∨ queued ∨ deferred` 중 하나라도
// 참이면 신규 admission 금지. `active` 만 참인 상태에서만 위 credit 식으로 판단한다.
// ★joined demand waiter 존재 = **predicate-local blocked** ⇒ 금지. 단 기존 telemetry
//   `demand_blocked_by_prefetch_*` 는 여기서 자동 증가하지 **않는다**(그 카운터의 기존 정의 =
//   총 QD 만석 ∧ prefetch inflight>0 의 실대기 구간만 — 불변·r3 분리).
// ★`cmd_deferred` = 이 turn 의 demand command 계획 보류(`disp_can_plan_demand` 실패).
bool pbv_demand_predicate_blocks_locked(bool cmd_deferred) {
    if (cmd_deferred) return true;                                   // deferred
    { std::lock_guard<std::mutex> lk(g_mod.q_mutex);                 // queued(락 순서 g_mod→q)
      if (g_mod.q_count > 0) return true; }
    // blocked ① — 채널 부족으로 pump 가 멈춘 자리에 demand op 이 남아 있다.
    for (size_t k = g_mod.submit_head; k < g_mod.submit_q.size(); k++) {
        const int32_t oi = g_mod.submit_q[k];
        if (oi < 0) continue;                                        // ★A6 합침이 걷어간 자리
        if (g_mod.pending_ops[(size_t)oi].origin == PIO_ORIGIN_DEMAND) return true;
    }
    // blocked ② — joined demand waiter 가 붙은 prefetch record(predicate-local blocked).
    for (size_t i = 0; i < g_mod.pending_ops.size(); i++) {
        const PendingIo & op = g_mod.pending_ops[i];
        if (!op.in_use) continue;
        if (op.origin == PIO_ORIGIN_PREFETCH && !op.waiters.empty()) return true;
    }
    return false;
}

// A0-4 continuation wave 재검사 결과. ★★R-1 로 **실제 발동한다** — `disp_pump_locked` 가 이
//   turn 의 pump pass 에서 후속 wave 의 이중 credit 식을 원자 재검사하고, 실패한 record 를
//   제출·C3 계상 없이 건너뛰면서 이 플래그를 세운다. admission 은 그 turn 의 신규 admission 을
//   생략한다(순서 우선을 **용량 축적 우선**으로 승격 — r6 기아 차단 폐합분).
//   ★구 판은 이 함수가 상수 false 였다: `W_l==P_l` 을 형상 하한으로 강제해 wave 가 1개뿐이었기
//   때문이고, 그것이 교차 r1 F2(FROZEN A0-2 미구현)의 실체였다.
inline bool pbv_continuation_deferred_locked() { return g_mod.pbv_cont_deferred; }

inline uint64_t pbv_fp_mix(uint64_t h, uint64_t v) { h ^= v; return h * 1099511628211ull; }

// ①게이트 park 지문 — 진입 게이트가 읽는 **두 판정 결과만**. ★F5 의 직접 원인이 이 축이다:
//   command 를 소비하면 predicate 가 풀리는데 구 판은 그걸 해제 사건으로 보지 않았고, 그
//   국면엔 credit 변화가 영영 안 와서(in-flight 0) 래치가 영구화됐다.
uint64_t pbv_park_fp_gate_locked(bool blocks, bool cont) {
    uint64_t h = 1469598103934665603ull;                      // FNV-1a 64 offset basis
    h = pbv_fp_mix(h, blocks ? 1u : 0u);
    h = pbv_fp_mix(h, cont   ? 1u : 0u);
    return h;
}

// ②credit-defer park 지문 — **공급 축 + credit/record gauge 만**. predicate 는 일부러 뺀다:
//   무관한 command 왕복이 predicate 를 흔들어도 credit 부족이라는 사유는 그대로이므로 park 를
//   풀면 안 된다(풀면 `skipped_drain_turns` 가 turn 마다 올라 A0-4 ④ 위반 — V3P12 가 문다).
//   ★정적 노브(`v3_io_param_val`)는 seal 이후 불변이라 넣지 않는다.
uint64_t pbv_park_fp_credit_locked() {
    uint64_t h = 1099511628211ull;
    h = pbv_fp_mix(h, (uint64_t)ggml_moe_phaseb_intent_depth());   // 새 intent 도착 = 재시도 사유
    h = pbv_fp_mix(h, g_mod.pb_cur_active ? 1u : 0u);
    h = pbv_fp_mix(h, (uint64_t)(uint32_t)g_mod.pb_cur_rank);
    h = pbv_fp_mix(h, (uint64_t)(uint32_t)g_mod.v3_inflight_demand);
    h = pbv_fp_mix(h, (uint64_t)(uint32_t)g_mod.v3_inflight_prefetch);
    h = pbv_fp_mix(h, (uint64_t)(uint32_t)g_mod.pbv_child_reserved);
    h = pbv_fp_mix(h, (uint64_t)(uint32_t)g_mod.pbv_rec_active);
    return h;
}

// 현재 상태를 **park 를 세운 사유의 자로** 잰다.
uint64_t pbv_park_fp_now_locked(bool blocks, bool cont) {
    return (g_mod.pbv_park_reason == PBV_PARK_GATE)
         ? pbv_park_fp_gate_locked(blocks, cont)
         : pbv_park_fp_credit_locked();
}

int pb_admit_locked(bool cut_mode, bool cmd_deferred = false) {
    if (!g_mod.pb_prefetch_on) return PB_ADMIT_EMPTY;
    // ★C3 동결 게이트 — view 중 prefetch admission 금지(SPEC_SHADOW_DRAFT §3 C3).
    if (g_mod.shadow_view_open.load(std::memory_order_acquire) != 0) {
        g_mod.shadow_view_refused++;
        return PB_ADMIT_EMPTY;
    }
    // ★★PB virtual A2 — turn 진입 게이트(§2-0 A0-4). cut_mode 는 잔여 intent 전건 처분이
    //   목적이라 아래 두 게이트를 적용하지 않는다(CUT ④ 계약 불변).
    const bool pbv = v3_active();
    if (pbv && !cut_mode) {
        // ⓐ demand 상태 predicate(금지가 이긴다) — turn 단위로 1회만 평가한다.
        const bool blocks = pbv_demand_predicate_blocks_locked(cmd_deferred);
        // ⓑ drain-before-refill(A0-4 r6 폐합) — 그 turn 에 continuation 재검사가 1건이라도
        //   실패했으면 신규 admission 을 생략한다(순서 우선을 용량 축적 우선으로 승격).
        const bool cont   = pbv_continuation_deferred_locked();
        // ★★R-3(F5) — **사유 소멸 시 해제**. 지문이 달라졌으면 park 를 세운 근거가 이미 사라진
        //   것이므로 여기서 내린다. 구 판은 credit 변화만 해제 사건으로 봐서, command 소비로
        //   predicate 가 풀린 뒤에도(그리고 in-flight 가 0 이라 credit 변화가 영영 안 오는
        //   국면에서도) 래치가 남아 50ms polling 이 영구화됐다.
        if (g_mod.pbv_admit_parked &&
            pbv_park_fp_now_locked(blocks, cont) != g_mod.pbv_park_fp) {
            g_mod.pbv_admit_parked = false;
            g_mod.pbv_park_reason  = PBV_PARK_NONE;
        }
        if (blocks || cont) {
            // ★★U-3(교차 r4) — **결과 지점에서 센다**. 구 판은 이 분기 **밖에서** `if (cont)` 로
            //   셌는데, 그건 `cont` **입력**을 세는 것이라 "실제 차단만 되돌리는" 역변이
            //   (`if (blocks || cont)` → `if (blocks)`)에 **무감**이었다: admission 은 진행되는데
            //   카운터는 그대로 올랐다. 이제 이 줄은 **분기 안**에 있으므로, 차단이 사라지면
            //   여기 도달하지 않아 카운터도 멈춘다 = 역변이 감도가 선다.
            if (cont) {
                g_mod.pbv_admit_skipped_cont++;
                // ★★U-3 — 계수와 **같은 자리**에서 증인을 박는다(대기 고정). 여기서 재야
                //   ①아직 살아 있는 record 수 ②아직 큐에 남은 intent 수가 **생략 그 순간의
                //   값**이다. `intent_depth` 는 `g_pb.int_mu` 만 잡으며, 이 함수는 이미 아래에서
                //   같은 큐를 `intent_peek/pop` 으로 만지므로 잠금 순서가 새로 생기지 않는다.
                // ★**아직 '실 후보를 막은' 증인이 없을 때만** 갱신한다(sentinel 포함).
                //   생략이 공허할 수도 있기 때문이다 — 주입된 continuation 실패가 후보가 모두
                //   소진된 뒤 turn 에 착지하면 막을 것이 없다(실측: 3연속 중 1회 `rank_left=0`).
                //   ⇒ 증인은 "공허한 생략은 덮어쓰고, **실 후보를 막은 생략을 만나면 거기서
                //   동결**"한다. 관문은 그 동결 여부를 요구하므로 "한 번이라도 실제로 막았는가"가
                //   계약이 되고, 끝까지 못 잡으면 마지막 상태가 그대로 진단에 남는다.
                const bool have_witness = (g_mod.pbv_admit_skip_valid_left > 0 &&
                                           g_mod.pbv_admit_skip_ncap_room  > 0 &&
                                           g_mod.pbv_admit_skip_credit_ok  == 1);
                if (!have_witness) {
                    // ⓐ 유효 ID 잔여 — 진행 중 intent 의 **아직 안 쓴 rank** 중 NA 가 아니고
                    //    `< n_expert` 인 것만 센다(그것만이 admission 후보가 될 수 있다).
                    //    진행 중 intent 가 없으면 **큐 머리**를 비파괴로 들여다본다(peek).
                    ggml_moe_phaseb_intent probe{};
                    int32_t start = 0;
                    bool    have  = false;
                    if (g_mod.pb_cur_active) { probe = g_mod.pb_cur_intent; start = g_mod.pb_cur_rank; have = true; }
                    else if (ggml_moe_phaseb_intent_peek(&probe)) { start = 0; have = true; }
                    int32_t valid = 0, room = 0, cok = 0;
                    if (have) {
                        for (int32_t r = start; r < (int32_t)probe.k; r++) {
                            const uint16_t e = probe.ids[r];
                            if (e != GGML_MOE_PHASEB_NA_U16 && (int32_t)e < g_mod.n_expert) valid++;
                        }
                        // ⓑ N_cap 여유 — 소비 지점과 **같은 식**(post-state 검사)의 여유분이다.
                        room = pbv_ncap_locked(probe.target_mlpos, (int32_t)probe.n) - g_mod.pbv_rec_active;
                        // ⓒ credit — 소비 지점과 같은 폭 `w` 로 같은 술어를 묻는다.
                        cok  = pbv_credit_ok_locked(pbv_wave_width(probe.target_mlpos)) ? 1 : 0;
                    }
                    g_mod.pbv_admit_skip_rec_active   = g_mod.pbv_rec_active;
                    g_mod.pbv_admit_skip_intent_depth = ggml_moe_phaseb_intent_depth();
                    g_mod.pbv_admit_skip_valid_left   = valid;
                    g_mod.pbv_admit_skip_ncap_room    = room;
                    g_mod.pbv_admit_skip_credit_ok    = cok;
                    g_mod.pbv_admit_skip_admitted     = ggml_moe_phaseb_counter(GGML_MOE_PB_C_ADMITTED);
                }
            }
            // 신규 admission 없음·intent 미소비·C3 불변. 변화 없는 queued intent 만으로
            // 즉시 다음 turn 을 만들지 않도록 park(A0-4 wake 결과 계약 ②).
            // ★★이미 park 중이면 **사유를 덮어쓰지 않는다**. credit 부족으로 선 park 위로
            //   무관한 TEST_QUERY 가 잠깐 predicate 를 세웠다고 사유를 GATE 로 바꾸면, 그
            //   command 가 소비되는 순간 GATE 지문이 달라져 park 가 풀리고 → 같은 credit
            //   부족으로 다시 defer → `skipped_drain_turns` 가 command 왕복마다 오른다
            //   (A0-4 ④ 위반). 실제로 덮어쓰게 뒀다가 V3P12 가 drain 2→3 으로 잡아냈다.
            //   park 는 **먼저 선 사유가 소유**하고, 그 사유가 풀릴 때만 위 해제 검사가 내린다.
            if (!g_mod.pbv_admit_parked) {
                g_mod.pbv_admit_parked = true;
                g_mod.pbv_park_reason  = PBV_PARK_GATE;
                g_mod.pbv_park_fp      = pbv_park_fp_gate_locked(blocks, cont);
            }
            return PB_ADMIT_EMPTY;
        }
    }
    for (;;) {
        if (!g_mod.pb_cur_active) {
            if (!ggml_moe_phaseb_intent_peek(&g_mod.pb_cur_intent)) return PB_ADMIT_EMPTY;
            ggml_moe_phaseb_intent_pop();
            g_mod.pb_cur_active = true;
            g_mod.pb_cur_rank   = 0;
            g_mod.pb_cur_row    = pb_bind_intent_row_locked(g_mod.pb_cur_intent);
            // ★부록 14.6 — 수집 모드 rank 판정용 경량 예측열 스냅샷(off/observe 에서는 pb_sample
            //   행이 없어 blocked-rank 결합이 불가하다는 접점 조사 결과의 처분).
            pb_pred_snap_put_locked(g_mod.pb_cur_intent);
            // ★★PB virtual A2 — A0-5: `N_cap(l) = 0` 인 층 = **그 층 프리페치 기능 OFF**.
            //   후보가 설 수 없는 정적 형상이므로 disposition(C3 열)이 아니라 eligibility
            //   skip 이다 — intent 전체를 그 사유로 소비하고 C3 는 건드리지 않는다.
            //   ★`n == 0`(owner 스탬프 미결속)은 여기서 걸러내지 않는다 — 아래 ambient-read
            //   fail-close(fatal)가 그 계약의 소유자다.
            if (pbv && g_mod.pb_cur_intent.n > 0 &&
                pbv_ncap_locked(g_mod.pb_cur_intent.target_mlpos,
                                (int32_t)g_mod.pb_cur_intent.n) <= 0) {
                if (g_mod.pb_cur_row >= 0)
                    g_mod.pb_samples[(size_t)g_mod.pb_cur_row].elig_skip_reason =
                        GGML_MOE_PB_SKIP_NCAP_ZERO;
                ggml_moe_phaseb_count_skip(GGML_MOE_PB_SKIP_NCAP_ZERO,
                                           g_mod.pb_cur_intent.request_epoch);
                g_mod.pb_cur_active = false;
                return PB_ADMIT_DISPOSED;   // progress — 남은 intent 를 같은 turn 에 계속 소비
            }
        }
        const ggml_moe_phaseb_intent it = g_mod.pb_cur_intent;
        const int32_t row = g_mod.pb_cur_row;
        const uint32_t req = it.request_epoch;
        if (g_mod.pb_cur_rank >= (int32_t)it.k) { g_mod.pb_cur_active = false; continue; }
        const uint16_t rank = (uint16_t)g_mod.pb_cur_rank++;
        const uint16_t eid  = it.ids[rank];
        // ★5c-T(부록 14.3 ①) — admission rank 결정 1건 = ADMISSION_DECISION **정확히 1레코드**.
        //   ★r1 M3 두 지점 수리:
        //     ①snapshot 전 필드(E 포함)를 **mutation 전에** 계산해 조기 drop 에도 완전한 상태를
        //       남긴다(구 판은 QD/cap drop 이 E 계산 전이라 `n_free_slots=0` 의 불완전 snapshot).
        //     ②claim 경로의 레코드는 **제출 성공/실패가 확정된 뒤** 방출한다(구 판은 claim 시점에
        //       ISSUED 로 조기 방출해 실 제출 실패를 ledger 가 정정하지 못했다).
        //   `decision_seq` 는 candidate rank 순번(controller §2 결정 표본)이고, generation 순번은
        //   owner 스탬프의 `admission_seq` 로 별도 기입된다(r1 M1 분리).
        uint64_t dec_seq    = GGML_MOE_PHASEB_NA_U64;
        uint64_t owner_aseq = GGML_MOE_PHASEB_NA_U64;   // owner generation admission 순번
        int32_t  owner_qd   = 0, owner_rev = 0;         // owner 박제 QD·tuple_revision
        uint8_t  adm_cap    = 0;
        uint16_t adm_free   = 0;
        auto count_free_slots = [&]() -> int64_t {
            int64_t e = 0;
            for (int32_t i = 0; i < g_mod.n_slots; i++) {
                if (g_mod.slot_pending[(size_t)i]) continue;
                if (g_mod.slots[(size_t)i].lease.load(std::memory_order_relaxed) != 0) continue;
                e++;
            }
            return e;
        };
        // ★r2 M3 ① — 공통 snapshot(owner stamp · cap · E)은 **rank 판정 진입 전**에 확정한다.
        //   구 판은 candidate 계수 뒤에야 채워, 사전 처분(too_late/stale/resident/pending) 레코드가
        //   `admission_seq=NA · stamp_qd=0 · cap=0 · E=0` 인 결손 snapshot 으로 남았다(r2 가
        //   run3 보존 wire `case_318_pb5ct_xreq/pb_ledger.pb` 의 rank 1~7 에서 실물 재현).
        //   ★수집 OFF 경로에는 이 블록이 통째로 없다 — E 조기 계산이 새면 명령열이 달라진다.
        int64_t E = -1;
        if (g_mod.pb_collect) {
            int32_t ok_ = 0, on_ = 0;
            ggml_moe_phaseb_run_stamp(req, &ok_, &on_, &owner_qd, &owner_rev, &owner_aseq);
            adm_cap  = pb_clamp_u8(cut_mode ? 0 : g_mod.qd_effective);
            E        = count_free_slots();
            adm_free = (uint16_t)(E > 0xFFFF ? 0xFFFF : E);
        }
        auto build_decision = [&](uint8_t d) {
            ggml_moe_phaseb_ledger r = pb_led_new(GGML_MOE_PB_L_ADMISSION_DECISION);
            r.request_epoch  = req;
            r.graph_seq      = it.graph_seq;
            r.graph_epoch    = it.graph_epoch;
            r.intent_ordinal = it.intent_ordinal;
            r.rank           = rank;
            r.record_idx     = (int64_t)it.target_mlpos * g_mod.n_expert + (int64_t)eid;
            r.layer          = it.target_layer;
            r.origin         = 1;
            r.disposition    = d;
            r.decision_seq   = dec_seq;
            r.cap_total      = adm_cap;
            r.n_free_slots   = adm_free;
            // ★§1 owner-local read — ambient 전역이 아니라 owner intent/generation 의 스탬프.
            r.stamp_k        = pb_clamp_u8((int32_t)it.k);
            r.stamp_n        = pb_clamp_u8((int32_t)it.n);
            r.stamp_qd       = pb_clamp_u8(owner_qd);
            r.tuple_revision = pb_clamp_u8(owner_rev);
            r.admission_seq  = owner_aseq;
            return r;
        };
        auto led_decision = [&](uint8_t d) {
            if (!g_mod.pb_collect) return;
            ggml_moe_phaseb_ledger r = build_decision(d);
            const bool drop = (d == GGML_MOE_PB_D_DROP_NO_SLOT_QD ||
                               d == GGML_MOE_PB_D_DROP_PREFETCH_CAP ||
                               d == GGML_MOE_PB_D_DROP_RESERVE);
            if (!drop) { pb_led_emit_locked(r); return; }
            // DROP 계열엔 참여 집합 전량 동봉(QD/cap = in-flight prefetch owner · reserve 는
            // 점유 READY(PROBATION) owner 까지 — controller §6 "stamped prefetch 상태").
            std::vector<ggml_moe_phaseb_ledger_owner> parts;
            pb_collect_participants_locked(parts, /*include_ready=*/d == GGML_MOE_PB_D_DROP_RESERVE,
                                           /*inflight_only=*/false);
            pb_led_emit_locked(r, parts.data(), (uint32_t)parts.size());
        };
        auto dispose = [&](uint8_t d) {
            led_decision(d);
            pb_dispose_rank_locked(row, it.intent_ordinal, rank, d, req);
        };
        // n_expert < k 방어 — seal에서 프리페치를 끄므로 정상 운전에서는 도달하지 않는다.
        if (eid == GGML_MOE_PHASEB_NA_U16 || (int32_t)eid >= g_mod.n_expert) { dispose(GGML_MOE_PB_D_PRE_STALE); return PB_ADMIT_DISPOSED; }

        const int64_t record = (int64_t)it.target_mlpos * g_mod.n_expert + eid;

        // ── 사전 처분(§4.2 우선순서: too_late → stale → resident/pending) ──
        if (pb_is_served_locked(it.graph_seq, it.target_mlpos)) { dispose(GGML_MOE_PB_D_PRE_TOO_LATE); return PB_ADMIT_DISPOSED; }
        if (g_mod.pb_cur_graph_seq != GGML_MOE_PHASEB_NA_U32 && it.graph_seq != g_mod.pb_cur_graph_seq) {
            dispose(GGML_MOE_PB_D_PRE_STALE); return PB_ADMIT_DISPOSED;
        }
        {
            const RecordEntry & re = g_mod.record_table[(size_t)record];
            const int32_t si = re.slot_index.load(std::memory_order_relaxed);
            bool resident = false;
            if (si >= 0) {
                const Slot & s = g_mod.slots[(size_t)si];
                resident = s.generation.load(std::memory_order_relaxed) == re.slot_generation.load(std::memory_order_relaxed)
                        && s.owner_record == record;
            }
            if (resident) { dispose(GGML_MOE_PB_D_PRE_RESIDENT); return PB_ADMIT_DISPOSED; }
        }
        if (g_mod.pending_by_record[(size_t)record] >= 0) { dispose(GGML_MOE_PB_D_PRE_PENDING); return PB_ADMIT_DISPOSED; }

        // ── admission ①②③(순서 고정 — 먼저 걸린 사유 1개만) ──
        // ★5c-T §1 — N 은 **owner generation 의 스탬프**(`it.n`)만 읽는다. 전역 `g_mod.pb_n`
        //   직독은 런 중 N 이 바뀌면 구 generation 에 새 N 을 적용하는 계약 위반이었다
        //   (SPEC_PREFETCH_CONTROLLER §1 · 접점 조사: 판정 경로는 여기 1건뿐).
        //   ★r1 M2 — 전역 폴백은 **제거**했다. controller §1 은 도달 여부와 무관하게 ambient
        //   read 를 금지하므로, `it.n==0`(스탬프 미결속)은 조용히 전역으로 대체하지 않고
        //   fail-close 한다 — intent 는 seal 이후에만 발행되므로 정상 운전에서 도달하지 않는다.
        // ★★A2 — 이 검사는 **candidate 계상보다 앞**으로 옮겼다. virtual 의 credit defer 가
        //   `it.n` 을 읽어야 하고, defer 는 C3 를 건드리면 안 되기 때문이다(아래 참조).
        if (it.n == 0) fatal_abort("prefetch intent carries no owner N stamp (ambient read forbidden) "
                                   "intent_ordinal=%llu rank=%u",
                                   (unsigned long long)it.intent_ordinal, (unsigned)rank);
        if (pbv) {
            // ══ mode=virtual admission(§2-0 A0-4/A0-5) ══
            //   검사 순서 = eligibility(위 N_cap==0 skip) → **N_cap** → **credit** → slot reserve.
            //   ★두 판정을 **계상 전에** 함께 만든다: credit 부족은 `defer` 라 C3(candidate 포함)
            //     를 하나도 건드리면 안 되고, 동시 실패 시에는 순서상 N_cap 이 먼저다.
            const int32_t w    = pbv_wave_width(it.target_mlpos);
            const int32_t ncap = pbv_ncap_locked(it.target_mlpos, (int32_t)it.n);
            //   A0-5 소비 지점 — 후보 record 의 층 cap 을 **post-state** 로 검사(현재값=cap 이면 거부).
            //   gauge 는 전역 1개(`pbv_rec_active` = reserved+inflight record 수).
            const bool ncap_sat  = cut_mode || ((g_mod.pbv_rec_active + 1) > ncap);
            const bool credit_ok = !cut_mode && pbv_credit_ok_locked(w);
            if (!ncap_sat && !credit_ok) {
                // ★transient credit 부족 = **defer**(A0-4 r3 정정): 그 turn admission 없음 ·
                //   intent 잔존(rank 커서 되돌림) · **disposition 미발생**(candidate·기존 C3 열
                //   어디에도 계상하지 않는다 — `pb_pre_skipped_pending` 재사용 불가).
                //   관측은 C3 **밖** 신설 진단 카운터 2종뿐.
                g_mod.pb_cur_rank--;                       // 이 rank 는 소비되지 않았다
                g_mod.pb_admit_deferred_credit_attempts++; // retry-attempt(중복 계수 허용)
                if (!g_mod.pbv_admit_parked) {
                    // ★A0-4 ④ — 동일 상태 hot-loop 가 아니라 **실제 drain turn 당 1회**.
                    g_mod.pb_admit_skipped_drain_turns++;
                    g_mod.pbv_admit_parked = true;         // ② 변화 없는 intent 로는 재진입 금지
                    // ★★R-3 — 지문은 **여기 상태로** 뜬다. 위 게이트에서 뜬 값을 재사용하면 안
                    //   된다: 그건 intent pop(`pb_cur_active`/`pb_cur_rank` 변동) **이전** 값이라
                    //   다음 turn 이 "달라졌다"고 오판해 park 를 내리고 같은 상태로 재시도하며,
                    //   그러면 이 카운터가 turn 마다 올라 A0-4 ④ 를 정면으로 깬다. 위의 rank
                    //   되돌림(`pb_cur_rank--`)이 이미 끝났으므로 지금 상태가 곧 "다음 turn 이
                    //   볼 상태"다.
                    g_mod.pbv_park_reason = PBV_PARK_CREDIT;
                    g_mod.pbv_park_fp     = pbv_park_fp_credit_locked();
                }
                return PB_ADMIT_DEFERRED;
            }
            ggml_moe_phaseb_count(GGML_MOE_PB_C_CANDIDATE, req, 1);
            if (g_mod.pb_collect) dec_seq = ggml_moe_phaseb_next_decision_seq(req);
            //   ★N_cap 포화 = 기존 열 `pb_dropped_prefetch_cap` 으로 drop(intent 소비 ·
            //     C3 candidate 산입 — bin 의 cap drop 의미 승계·mode 간 동형 유지).
            //     cut_mode 도 같은 열로 닫는다(bin 의 `cap_total=0` 처분과 동형).
            if (ncap_sat) { dispose(GGML_MOE_PB_D_DROP_PREFETCH_CAP); return PB_ADMIT_DISPOSED; }
        } else {
            ggml_moe_phaseb_count(GGML_MOE_PB_C_CANDIDATE, req, 1);
            // ★decision_seq 는 candidate(= controller §2 결정 표본)에만 발급한다. 사전 처분은 NA 유지.
            if (g_mod.pb_collect) dec_seq = ggml_moe_phaseb_next_decision_seq(req);
            const int32_t total_inflight = g_mod.pb_inflight_demand + g_mod.pb_inflight_prefetch;
            const int32_t cap_total = cut_mode ? 0 : g_mod.qd_effective;
            if (total_inflight >= cap_total)                 { dispose(GGML_MOE_PB_D_DROP_NO_SLOT_QD);   return PB_ADMIT_DISPOSED; }
            if (g_mod.pb_inflight_prefetch >= (int32_t)it.n) { dispose(GGML_MOE_PB_D_DROP_PREFETCH_CAP); return PB_ADMIT_DISPOSED; }
        }
        // ③ mutation **전에** 투영: E − 1 ≥ n_expert_used (claim 전 reserve).
        //    ★슬롯 축은 mode 무관이다(A0-4 의 credit 식은 채널 축) — 양 경로 공통 관문으로 남는다.
        if (E < 0) E = count_free_slots();
        if (E - 1 < (int64_t)g_mod.n_expert_used)        { dispose(GGML_MOE_PB_D_DROP_RESERVE);      return PB_ADMIT_DISPOSED; }
        ggml_moe_phaseb_count(GGML_MOE_PB_C_ADMITTED, req, 1);

        // ★r2 M3 ② — claim 레코드의 `event_seq/qpc` 는 **어떤 mutation 보다 먼저** 발급한다.
        //   구 판은 victim 축출·slot mutation 뒤에 base record 를 만들어, 결과인 `READY_EVICT`
        //   가 원인인 admission 보다 앞선 event_seq 를 받는 인과 역전이 있었다(r2 적발).
        //   이후 제출 결과에서는 `disposition` 과 `io_event_seq` 만 patch 해 1회 방출한다.
        ggml_moe_phaseb_ledger claim_rec{};
        bool claim_rec_valid = false;
        if (g_mod.pb_collect) { claim_rec = build_decision(GGML_MOE_PB_D_ISSUED); claim_rec_valid = true; }

        // ── ④ 통과 시에만 claim ──
        int32_t victim = -1;
        for (int32_t i = 0; i < g_mod.n_slots; i++) {
            if (g_mod.slot_pending[(size_t)i]) continue;
            if (g_mod.slots[(size_t)i].lease.load(std::memory_order_relaxed) != 0) continue;
            if (victim_better_locked(i, victim)) victim = i;
        }
        const int32_t oi = victim >= 0 ? pio_alloc_locked() : -1;
        if (oi < 0) {   // 방어(E>=1 보장·pending 풀은 n_slots) — 계상은 이미 admitted라 issued/failed로 닫는다
            // pending 풀 고갈 = 제출 미도달 확정. 미리 만든 base record 를 그대로 닫는다.
            if (claim_rec_valid) {
                claim_rec.disposition = GGML_MOE_PB_D_SUBMIT_FAILED;
                pb_led_emit_locked(claim_rec);
            }
            pb_dispose_rank_locked(row, it.intent_ordinal, rank, GGML_MOE_PB_D_SUBMIT_FAILED, req);
            return PB_ADMIT_DISPOSED;
        }
        Slot & s = g_mod.slots[(size_t)victim];
        if (s.owner_record >= 0) {
            // ★prefetch 축출은 PA ledger(demand 스코프)에 먹이지 않는다(§2.3).
            pb_note_ready_evict_locked(s);   // ★5c-T 4분할 — 미소비 READY 축출 = resolved_wasted
            pb_transition_locked(GGML_MOE_PB_EV_EVICT, s.owner_record, victim,
                                 s.generation.load(std::memory_order_relaxed), s.pf_intent_ordinal);
            k2fac_record_slot_detach_locked(s.owner_record);
        }
        const LayerInfo & L = g_mod.layers[(size_t)it.target_mlpos];
        PendingIo & op = g_mod.pending_ops[(size_t)oi];
        op.m = BatchMiss();
        op.m.record_idx        = record;
        op.m.multiplicity      = 0;                                   // waiter 없음(speculative)
        op.m.generation        = k2fac_slot_generation_bump_locked(victim);
        op.m.slot              = victim;
        op.m.file_offset       = L.record_base + (uint64_t)eid * L.stride_bytes;
        op.m.logical_last_tick = g_mod.tick;   // ★demand LRU tick을 소비하지 않는다(§4)
        op.m.pa_batch_order    = 0;
        k2fac_slot_claim_victim_locked(victim, record, g_mod.tick);   // publish 전까지 READY_* 아님
        op.mlpos  = it.target_mlpos;
        op.phase  = Phase::DECODE;
        op.origin = PIO_ORIGIN_PREFETCH;
        op.state  = PIO_CLAIMED;
        op.request_identity = 0;
        op.intent_seq       = it.intent_ordinal;
        op.retry_of         = -1;
        op.pb_request_epoch = it.request_epoch;
        op.pb_graph_epoch   = it.graph_epoch;
        op.pb_graph_seq     = it.graph_seq;
        op.pb_rank          = rank;
        // ★5c-T §1 — owner 스탬프를 pending op 에 결속(병렬 side table · PendingIo 폭 불변).
        if (PbOpStamp * st = pb_op_stamp_at(oi)) {
            st->k   = pb_clamp_u8((int32_t)it.k);
            st->n   = pb_clamp_u8((int32_t)it.n);      // ★r1 M2 전역 폴백 없음
            st->qd  = pb_clamp_u8(owner_qd);
            st->rev = pb_clamp_u8(owner_rev);
            st->gen_admission_seq = owner_aseq;
            st->graph_epoch = it.graph_epoch;
            // ★r2 M3 ② — mutation 이전에 만든 base record 를 그대로 보류한다(재생성 금지 —
            //   재생성하면 event_seq/qpc 가 다시 축출 뒤 시점이 된다). 최종 disposition
            //   (ISSUED / SUBMIT_FAILED)만 제출 결과 확정 시점에 patch 해 1건 방출한다.
            if (claim_rec_valid) {
                st->admit_rec       = claim_rec;
                st->admit_rec_valid = 1;
                claim_rec_valid     = false;   // 소유권 이전(이중 방출 금지)
            }
        }
        // 방어 — side table 이 없는 조합에서 base record 를 잃지 않는다(유실 위장 금지).
        if (claim_rec_valid) { claim_rec.flags |= GGML_MOE_PB_L_F_UNSUBMITTED;
                               claim_rec.disposition = 0xFF; pb_led_emit_locked(claim_rec); }
        g_mod.pending_by_record[(size_t)record] = oi;
        g_mod.slot_pending[(size_t)victim] = 1;
        // ★★PB virtual A1/A2 — **admission commit**(§2-0 A0-3): 전역 record gauge +1 ·
        //   첫 wave 의 `W_l` child credit 원자 선점(②축 reserved 로 즉시 가시) · reservation 생성.
        //   claim 이 끝난 **뒤**에 세운다 — 앞서 세우면 claim 실패 경로가 예약을 흘린다.
        if (pbv) pbv_rec_admit_locked(oi, it.target_mlpos);
        g_mod.submit_q.push_back(oi);
        return PB_ADMIT_ISSUED;    // 턴당 1건
    }
}

// dispatcher 1 pass — command 수집/처리 → signaled 채널 회수 → demand 제출(B2 자리) → command 재검사.
// 반환: progressed(무언가 진행됨) / out_deferred(demand 계획을 미뤘음 = command_event 무시하고 대기).
bool disp_pass_locked(bool & out_deferred) {
    bool progressed = false;
    out_deferred = false;

    // ★★PB virtual A2 — turn 총순서(§2-0 A0-4):
    //     `signaled channel retire → demand 처리 → continuation wave → 신규 prefetch
    //      admission(턴당 1건) → command 재검사`
    //   구 순서는 command(demand) 가 retire 보다 앞이라 A0-4 와 어긋난다. 순서를 바꾸면
    //   demand 계획이 보는 캐시 상태(hit/miss)가 달라져 bin 의 ordered digest 까지 흔들리므로
    //   **virtual ∧ 프리페치 ON 에서만** retire 를 앞으로 올린다(Q4 OFF 무오염 유지).
    const bool pbv_turn = v3_active() && g_mod.pb_prefetch_on;
    if (pbv_turn && disp_retire_signaled_locked()) progressed = true;

    for (int pass = 0; pass < 2; pass++) { // ★"command 재검사"(B2 admission 자리 뒤 재확인)
        for (;;) {
            const int32_t ci = disp_peek();
            if (ci < 0) break;
            ControlCmd & c = g_mod.cmds[(size_t)ci];
            if (c.kind == CMD_DEMAND_ACQUIRE && !disp_can_plan_demand(c.n_ids)) { out_deferred = true; break; }
            disp_pop();
            progressed = true;
            // ★A0-4 wake 결과 계약 ③ "신규 demand/control 은 즉시 처리" 는 **여기서 이미
            //   성립한다**: park 는 command 처리 경로를 전혀 막지 않는다(command enqueue 는
            //   q_mutex 아래에서 event 를 세우고 `disp_peek` 은 q_count>0 이면 reset 하지 않는다).
            //   ★그래서 여기서 park 를 내리지 **않는다** — command 는 credit 을 바꾸지 않으므로,
            //   내리면 같은 상태로 재시도가 반복돼 `pb_admit_skipped_drain_turns` 가 A0-4 ④의
            //   "실제 drain turn 당 1회"를 깨고 hot-loop 계수가 된다. park 를 내리는 것은
            //   **credit 이 실제로 변하는 두 지점**뿐이다(child terminal · op 반납).
            switch (c.kind) {
                case CMD_DEMAND_ACQUIRE: disp_handle_demand_locked(ci); break;
                case CMD_RELEASE:        disp_handle_release_locked(ci); break;
                case CMD_TEST_QUERY:     disp_handle_query_locked(ci); break;
                case CMD_STOP:
                    g_mod.stat_stop_cmd++;
                    disp_complete(ci, 0);
                    break;
                default: disp_complete(ci, -1); break;
            }
        }
        if (pass == 0) {
            pb_drain_status_locked();   // ★R2 #5 — 비동기 사유의 행 사후 갱신
            if (disp_retire_signaled_locked()) progressed = true;
            if (disp_pump_locked())            progressed = true;
            // ── ★B2 prefetch admission 턴당 1건(§4) — demand pump 뒤이므로 demand 절대 우선. ──
            //   ★R1 #2 ② — DISPOSED(사전 처분·drop)도 progress로 인정한다. 그래야 남은 rank가
            //   같은 turn 흐름에서 계속 소비돼 미처분 잔존(complete=0)이 나지 않는다.
            // ★★PB virtual A2 — `out_deferred`(demand 계획 보류)를 predicate 에 전달한다
            //   (A0-4 `blocked ∨ queued ∨ deferred` 의 deferred 항).
            const int adm = pb_admit_locked(/*cut_mode=*/false, /*cmd_deferred=*/out_deferred);
            // ★DEFERRED 는 **progress 가 아니다**(A0-4 ① drain-defer): intent·C3 미소비라
            //   progress 로 세면 같은 상태로 즉시 재순환해 r7 hot-loop 가 그대로 산다.
            if (adm != PB_ADMIT_EMPTY && adm != PB_ADMIT_DEFERRED) {
                // ★admission 이 실제로 전진했다(ISSUED/DISPOSED) ⇒ 마스킹할 이유가 사라졌다.
                //   ★반대로 **재차 defer 한 turn 에서는 내리지 않는다** — 내리면 다음 turn 이
                //   같은 상태를 새 drain turn 으로 세어 A0-4 ④(“실제 drain turn 당 1회”)가 깨진다.
                g_mod.pbv_admit_parked = false;
                progressed = true;
                if (adm == PB_ADMIT_ISSUED) disp_pump_locked(); // claim한 speculative op를 같은 턴에 제출
            }
            // ★테스트 전용 abort 훅 — 제출된 op가 임계 이상이면 그 자리에서 취소 drain을 행사.
            if (g_mod.test_abort_after_submit > 0) {
                int32_t n_sub = 0;
                for (auto & ch : g_mod.channels) if (ch.busy) n_sub++;
                if (n_sub >= g_mod.test_abort_after_submit) {
                    g_mod.test_abort_after_submit = 0;
                    disp_abort_inflight_locked();
                    progressed = true;
                }
            }
        }
        if (out_deferred) break;
    }
    return progressed;
}

unsigned __stdcall dispatcher_main(void *) {
    // handle 0 = command_event, 1..QD = channel.ev → 1+QD <= 64(MAXIMUM_WAIT_OBJECTS).
    HANDLE handles[MOE_DIRECT_QD_MAX + 1];
    DWORD  n_handles = 0;
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        handles[n_handles++] = g_mod.command_event;
        for (auto & ch : g_mod.channels) handles[n_handles++] = ch.ev;
    }
    for (;;) {
        // ── 테스트 전용 pause(만석 큐 픽스처) — g_mod.mutex 무보유 상태로 정지한다. ──
        {
            std::unique_lock<std::mutex> lk(g_mod.q_mutex);
            if (g_mod.q_pause_req && !g_mod.q_stopping) {
                g_mod.q_paused = true;
                g_mod.q_pause_cv.notify_all();
                lk.unlock();
                WaitForSingleObject(g_mod.resume_event, INFINITE);
                ResetEvent(g_mod.resume_event);
                lk.lock();
                g_mod.q_paused = false;
                g_mod.q_pause_cv.notify_all();
                continue;
            }
        }
        bool deferred = false;
        bool progressed = false;
        bool exit_now = false;
        bool parked = false;   // ★★PB virtual A2 — drain-defer park(§2-0 A0-4 ②)
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            bool stopping, abort_mode;
            {
                std::lock_guard<std::mutex> lk(g_mod.q_mutex);
                stopping   = g_mod.q_stopping;
                abort_mode = g_mod.q_abort;
            }
            // ★보완 ② barrier — stopping 스냅샷 **직후**. 여기서 멈춘 동안 테스트가 q_stopping 을
            //   세우면, 이 pass 는 stale stopping=false 로 진행해 신호 소실 창을 정확히 재현한다.
            //   대기는 유한(5s)이라 픽스처가 실패해도 스레드가 영구 정지하지 않는다.
            if (g_mod.test_stop_race && !stopping) {
                g_mod.test_stop_race = 0;   // one-shot
                if (g_mod.stop_race_hit) SetEvent(g_mod.stop_race_hit);
                if (g_mod.stop_race_go)  WaitForSingleObject(g_mod.stop_race_go, 5000);
            }
            if (stopping && abort_mode) {
                // abort/test — 잔여 in-flight 취소 drain 후 대기 command 전부 error 종결.
                disp_abort_inflight_locked();
                for (;;) {
                    const int32_t ci = disp_peek();
                    if (ci < 0) break;
                    disp_pop();
                    disp_complete(ci, -1);
                }
                exit_now = true;
            } else {
                progressed = disp_pass_locked(deferred);
                parked = g_mod.pbv_admit_parked;   // ★락 안에서 스냅샷(대기 정책 선택용)
                if (stopping) {
                    // ★R1 #3 ② — CUT ② (predictor drain/stop join)가 끝나기 전에는 exit 금지.
                    //   조기 exit하면 worker가 뒤늦게 발행한 intent가 미처분으로 남는다.
                    const bool pred_done = ggml_moe_phaseb_predictor_drained();
                    // ★CUT ④ — pre-cut command drain(③) 뒤 잔여 intent 전건을 닫힌 enum으로 처분.
                    while (pb_admit_locked(/*cut_mode=*/true) != PB_ADMIT_EMPTY) { progressed = true; }
                    // graceful — 신규 command는 이미 거절(cq_alloc/cut 래치), 잔여를 drain.
                    int32_t q_left = 0;
                    { std::lock_guard<std::mutex> lk(g_mod.q_mutex); q_left = g_mod.q_count; }
                    // ★R2 #2 — demand 완료 판정에 **수령된 demand command 잔여**를 포함한다.
                    //   `pb_inflight_demand`(물리 demand op)만 보면, demand 가 join 해 기다리는
                    //   prefetch-origin op 가 남은 상태를 "demand 없음"으로 오판해 그 op 를
                    //   CancelIoEx 하고 정상 요청을 실패시킨다(R2 적발).
                    const bool cmd_done = (q_left == 0) && (g_mod.submit_head >= g_mod.submit_q.size())
                                       && (g_mod.pb_inflight_demand == 0)
                                       && (g_mod.demand_outstanding == 0);
                    // ★CUT ⑤ — demand drain 완료 후 잔여 prefetch in-flight를 실제 취소 회수.
                    // ★★PB virtual A4 — 판정원을 mode-neutral 로 넓힌다: `pb_inflight_prefetch`
                    //   는 ④PB physical(op) 축이라 virtual 에서 영영 0 이고, 그대로 두면 잔여
                    //   speculative child 가 취소되지 않은 채 graceful exit 조건을 기다린다.
                    const bool pf_left = (g_mod.pb_inflight_prefetch > 0) ||
                                         (v3_active() && g_mod.v3_inflight_prefetch > 0);
                    if (pred_done && cmd_done && pf_left) {
                        if (disp_cancel_prefetch_inflight_locked()) progressed = true;
                    }
                    size_t n_busy = 0;
                    for (auto & ch : g_mod.channels) if (ch.busy) n_busy++;
                    if (pred_done && cmd_done && n_busy == 0 &&
                        ggml_moe_phaseb_intent_depth() == 0 && !g_mod.pb_cur_active) {
                        disp_abort_inflight_locked(); // 잔여 0 확인용(no-op) — 방어적 drain
                        // ★CUT ⑥ — intent depth·rank cursor·command·pending·channel 전수 0 검사.
                        int32_t residual = 0;
                        if (ggml_moe_phaseb_intent_depth() != 0) residual |= 1;
                        if (g_mod.pb_cur_active)                 residual |= 2;
                        { std::lock_guard<std::mutex> lk(g_mod.q_mutex); if (g_mod.q_count != 0) residual |= 4; }
                        if ((int32_t)g_mod.pending_free.size() != g_mod.n_slots) residual |= 8;
                        for (auto & ch : g_mod.channels) if (ch.busy) residual |= 16;
                        if (g_mod.pb_inflight_demand != 0 || g_mod.pb_inflight_prefetch != 0) residual |= 32;
                        // ★★PB virtual A4 — 같은 bit 의 mode-neutral 확장(①record 운영축 +
                        //   ②child 물리축). bin 은 이 두 축을 쓰지 않으므로 무영향이다.
                        if (v3_active() && (g_mod.v3_inflight_demand != 0 ||
                                            g_mod.v3_inflight_prefetch != 0 ||
                                            g_mod.pbv_rec_active != 0 ||
                                            g_mod.pbv_child_reserved != 0)) residual |= 32;
                        // ★selftest 전용 잔여 주입(FROZEN_INCOMPLETE_REASON_MASK §9-1·§9-7) —
                        //   자연 발생 잔여는 결정론적으로 만들 수 없으므로 세부 비트를 주입해
                        //   echo·마스크 결속 등식을 행사한다. 프로덕션 경로는 항상 0.
                        residual |= g_mod.test_cut_residual_force;
                        g_mod.pb_cut_residual = residual;
                        if (residual != 0) ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_CUT_RESIDUAL);
                        exit_now = true;
                    }
                }
            }
        }
        if (exit_now) break;
        if (progressed) continue;
        if (deferred) {
            // command_event가 signaled인 채로 미룬 상태 — 채널 완료만 기다린다(busy-spin 방지).
            if (n_handles > 1) WaitForMultipleObjects(n_handles - 1, handles + 1, FALSE, 50);
            else               Sleep(1);
        } else {
            // ★★정지 중에는 무한 대기 금지(방어 심층) — graceful exit 조건에는 채널/커맨드
            //   이벤트를 동반하지 않고 변할 수 있는 항(demand_outstanding·intent_depth·
            //   pb_cur_active·pred_done)이 섞여 있다. 정지 국면에서만 유한 대기로 바꿔 조건을
            //   주기적으로 재평가한다(정상 국면은 기존대로 INFINITE — 공회전 0 유지).
            bool stopping_now = false;
            { std::lock_guard<std::mutex> lk(g_mod.q_mutex); stopping_now = g_mod.q_stopping; }
            // ★★R-3(교차 r1 F5) — **park 중 50ms polling 철거**(A0-4 ⑤ "event-driven 재시도").
            //   구 판은 park 마스킹이 만든 lost-wake 창을 20Hz 재평가로 방어했는데, 그 창은
            //   이제 **사유 지문**이 닫는다: park 를 세운 상태가 그대로면 재시도해도 결과가
            //   같으므로 깨어날 이유가 없고, 무엇이든 달라지면 그 변화는 반드시 이벤트를
            //   동반해 우리를 깨운다 —
            //     ·신규 intent      → PhaseB worker 가 `command_event` 를 SetEvent(등록: 아래
            //                          `set_wake_event(command_event)`)
            //     ·신규 command     → `cq_submit_and_wait` 이 같은 mutex 아래 SetEvent
            //     ·credit 변화      → 채널 완료 handle(1..QD) · copy worker 의 SetEvent
            //   그리고 `disp_peek` 의 reset 이 intent 신호를 삼키더라도 **같은 pass 의
            //   `pb_admit_locked` 가 그 intent 를 지문 변화로 보고 park 를 내려** 그 turn 안에
            //   소비하므로(admission 은 reset 보다 뒤에 돈다) 신호 소실로 정체되지 않는다.
            //   ⇒ 정상 국면은 공회전 0(INFINITE)이고 A0-4 ⑤ "terminal 전 무한 재진입 없음"은
            //     polling 없이 성립한다. 정지 국면의 유한 대기(20ms)만 종전대로 남는다.
            (void)parked;
            WaitForMultipleObjects(n_handles, handles, FALSE, stopping_now ? 20 : INFINITE);
        }
        g_mod.stat_wait_wakeups.fetch_add(1, std::memory_order_relaxed);
    }
    return 0;
}

// dispatcher 자원 해제(스레드는 이미 종료/미기동 상태여야 한다). 호출자는 g_mod.mutex 보유.
void dispatcher_cleanup_locked() {
    ggml_moe_phaseb_set_wake_event(nullptr);   // ★R1 #2 ① — 해제 전 등록 해지(dangling 금지)
    for (auto & ch : g_mod.channels) if (ch.ev) CloseHandle(ch.ev);
    g_mod.channels.clear();
    if (g_mod.command_event) { CloseHandle(g_mod.command_event); g_mod.command_event = nullptr; }
    if (g_mod.resume_event)  { CloseHandle(g_mod.resume_event);  g_mod.resume_event  = nullptr; }
    if (g_mod.stop_race_hit) { CloseHandle(g_mod.stop_race_hit); g_mod.stop_race_hit = nullptr; }
    if (g_mod.stop_race_go)  { CloseHandle(g_mod.stop_race_go);  g_mod.stop_race_go  = nullptr; }
    g_mod.pending_ops.clear();
    g_mod.pending_free.clear();
    g_mod.pending_by_record.clear();
    g_mod.slot_pending.clear();
    g_mod.v3_child.clear();
    g_mod.pbv_rec.clear();                 // ★★PB virtual A1 — 짝과 같은 시점에 접는다
    g_mod.pbv_rec_active = g_mod.pbv_rec_inflight = g_mod.pbv_child_reserved = 0;
    g_mod.pbv_admit_parked = false;
    g_mod.v3_chan_op.clear();
    g_mod.v3_chan_part.clear();
    g_mod.v3_chan_staging.clear();
    g_mod.v3_chan_len.clear();
    g_mod.v3_chan_coal.clear();
    g_mod.io3_op_row.clear();
    g_mod.io3_chan_row.clear();
    g_mod.io3_chan_copy_row.clear();
    g_mod.submit_q.clear();
    g_mod.submit_head = 0;
    g_mod.demand_outstanding = 0;
}

// seal 말미(g_mod.mutex 보유)에서 1회 기동. 채널·이벤트·pending 표는 여기서 확정되고 주소가 고정된다.
bool dispatcher_start_locked(std::string & err) {
    // ★R18 테스트 전용 one-shot — dispatcher 기동 실패를 **결정론적으로** 만든다. 실환경에서는
    //   CreateEvent/스레드 생성 실패로만 도달해 재현할 수 없는데, 그 경로의 되돌림(특히 이미
    //   열린 metrics 아티팩트 정리)이 실제로 서는지는 잠가야 한다. 프로덕션 0.
    if (g_mod.v3_test_fail_dispatcher_start) {
        g_mod.v3_test_fail_dispatcher_start = 0;
        err = "test-only injected dispatcher start failure";
        return false;
    }
    g_mod.command_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // manual reset
    g_mod.resume_event  = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    g_mod.stop_race_hit = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // ★보완 ② 테스트 barrier
    g_mod.stop_race_go  = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_mod.command_event || !g_mod.resume_event || !g_mod.stop_race_hit || !g_mod.stop_race_go) {
        err = "CreateEventW(dispatcher) failed"; dispatcher_cleanup_locked(); return false;
    }
    g_mod.channels = std::vector<IoChannel>((size_t)g_mod.qd_effective);
    for (auto & ch : g_mod.channels) {
        ch.ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ch.ev) { err = "CreateEventW(channel) failed"; dispatcher_cleanup_locked(); return false; }
    }
    g_mod.pending_ops = std::vector<PendingIo>((size_t)g_mod.n_slots);
    g_mod.pending_free.clear();
    g_mod.pending_free.reserve((size_t)g_mod.n_slots);
    for (int32_t i = g_mod.n_slots - 1; i >= 0; i--) g_mod.pending_free.push_back(i);
    g_mod.pending_by_record.assign(g_mod.record_table.size(), -1);
    g_mod.slot_pending.assign((size_t)g_mod.n_slots, 0);
    // ★§Z-⑥ A3 — child 병렬표는 자기 짝(pending_ops·channels)과 **같은 시점에 같은 크기로**
    //   확정된다. 짝의 인덱스로 접근하므로 크기가 어긋나면 그 순간 계약이 깨진다.
    g_mod.v3_child.assign((size_t)g_mod.n_slots, V3ChildIo{});
    // ★★PB virtual A1 — ①record 운영축 병렬표. 위 `v3_child` 와 같은 짝·같은 크기 규율.
    //   gauge 3종도 여기서 0 으로 확정한다(재seal 이 이전 런의 잔여를 물려받지 않게).
    g_mod.pbv_rec.assign((size_t)g_mod.n_slots, PbvRec{});
    g_mod.pbv_rec_active     = 0;
    g_mod.pbv_rec_inflight   = 0;
    g_mod.pbv_child_reserved = 0;
    g_mod.pbv_admit_parked   = false;
    g_mod.v3_chan_op.assign((size_t)g_mod.qd_effective, -1);
    g_mod.v3_chan_part.assign((size_t)g_mod.qd_effective, -1);
    g_mod.v3_chan_staging.assign((size_t)g_mod.qd_effective, -1);
    g_mod.v3_chan_len.assign((size_t)g_mod.qd_effective, 0u);        // ★R10
    g_mod.v3_chan_coal.assign((size_t)g_mod.qd_effective, -1);       // ★A6
    // ★R12 — IO3 병렬표는 **IO3 가 켜진 런에서만** 존재한다. "기본 off = 자료구조도 안 만든다"는
    //   §5-1 주장이 문자 그대로 참이어야 한다(구 판은 항상 할당했다 — 주장과 구현이 갈렸다).
    //   접근부는 전부 `size()` 경계를 보므로 빈 표에서 조용히 no-op 이 된다.
    if (g_mod.io3_on) {
        g_mod.io3_op_row.assign((size_t)g_mod.n_slots, -1);
        g_mod.io3_chan_row.assign((size_t)g_mod.qd_effective, -1);   // ★A4
        g_mod.io3_chan_copy_row.assign((size_t)g_mod.qd_effective, -1);  // ★A6
    } else {
        g_mod.io3_op_row.clear();
        g_mod.io3_chan_row.clear();
        g_mod.io3_chan_copy_row.clear();
    }
    g_mod.submit_q.clear();
    g_mod.submit_q.reserve((size_t)g_mod.n_slots);
    g_mod.submit_head = 0;
    g_mod.demand_outstanding = 0;
    {
        std::lock_guard<std::mutex> lk(g_mod.q_mutex);
        g_mod.q_free_top = 0;
        for (int32_t i = MOE_DIRECT_CONTROL_Q_CAP - 1; i >= 0; i--) {
            g_mod.cmds[(size_t)i].ids.reserve(cq_ids_capacity()); // capacity 확정 → 정상 운전 무할당
            g_mod.cmds[(size_t)i].req_join_op.reserve(cq_ids_capacity());
            g_mod.cmds[(size_t)i].in_use = false;
            g_mod.cmds[(size_t)i].done   = false;
            g_mod.q_free[g_mod.q_free_top++] = i;
        }
        g_mod.q_head = 0; g_mod.q_count = 0;
        g_mod.q_stopping = false; g_mod.q_abort = false;
        g_mod.q_pause_req = false; g_mod.q_paused = false;
    }
    unsigned tid = 0;
    // ★STACK_SIZE_PARAM_IS_A_RESERVATION — 없으면 stack_size가 초기 commit로 해석되고 실제
    //   reserve는 PE 헤더 기본값(1MiB)이 되어 mem echo(dispatcher_stack_reserve_bytes)가 거짓이 된다.
    g_mod.disp_thread = _beginthreadex(nullptr, MOE_DIRECT_DISP_STACK, dispatcher_main, nullptr,
                                       STACK_SIZE_PARAM_IS_A_RESERVATION, &tid);
    if (g_mod.disp_thread == 0) { err = "cannot create dispatcher thread"; dispatcher_cleanup_locked(); return false; }
    // ★R1 #2 ① — worker의 intent enqueue가 이 이벤트로 dispatcher를 깨운다.
    ggml_moe_phaseb_set_wake_event(g_mod.command_event);
    g_mod.disp_running.store(true, std::memory_order_release);
    return true;
}

// R-B5a facade (v): stopjoin telemetry (RB_DESIGN.md:1221 section 2-6 (v)). The three
// `g_xs_stopjoin_*` objects are storage-owned telemetry that the dispatcher's stop path writes,
// and RB_DESIGN section 1-5 fixes three lifetimes on them: the record is CLEARED before every
// early return, it is WRITTEN before the Wait, and the last call's values are read relaxed by
// the API entry point. The two members below take the first two; the third is a read in
// `ggml_moe_direct_test_stop_join_stage` and this atom does not touch it. Each member performs
// the same relaxed stores in the same order, and the calls sit exactly where the stores sat.
inline void xsfac_stopjoin_clear() {
    g_xs_stopjoin_n.store(0, std::memory_order_relaxed);
    g_xs_stopjoin_deadline.store(-1, std::memory_order_relaxed);
    g_xs_stopjoin_left.store(-1, std::memory_order_relaxed);
}

inline void xsfac_stopjoin_record(int64_t deadline_abs_ms, int32_t left) {
    g_xs_stopjoin_deadline.store(deadline_abs_ms, std::memory_order_relaxed);
    g_xs_stopjoin_left.store(left, std::memory_order_relaxed);
    g_xs_stopjoin_n.store(1, std::memory_order_relaxed);
}

// 종료 시퀀스(§2): STOPPING → 신규 command 거절 → active demand/release drain → 잔여 CancelIoEx →
// 전 채널 terminal 회수 → dispatcher exit/join → event 해제. ★호출자는 g_mod.mutex 미보유.
// ★★S-2(교차 r2 F-2) — **유계 종료**. 반환 1 = 정상 종료·정리 완료 / 0 = **timeout**
//   (스레드 미회수 — 호출자는 이걸 **실패로 판정**해야 한다).
//   `timeout_ms < 0` = INFINITE(프로덕션 경로·기존 거동 불변).
//   ★왜 필요한가: 구 판은 훅이 `WaitForSingleObject(..., INFINITE)` 라 **shutdown 회귀 시
//     관문이 FAIL 하는 게 아니라 selftest 자체가 교착**했다(1,300건 릴리스 관문에
//     교착형 게이트를 두는 것은 순손실). 유계화해야 그 회귀가 **붉은 수치**로 담릴다.
//   ★timeout 경로에서는 **cleanup 을 하지 않는다** — dispatcher 가 아직 그 상태를 쓰고
//     있을 수 있으므로 지우면 use-after-free 다. 관문이 붉어 런이 종료되는 게 정상 처분이다.
// E-2a (Codex r2 blocker 2) - the absolute-deadline form. Three stages inside this one call
// spend budget: the predictor's capture-idle wait, the predictor worker join, and the
// dispatcher join. The old form handed each of them the caller's timeout_ms afresh, so a
// remaining R could cost min(R,10s) + R + R - with the 30 s default, 10 s + 30 s + 30 s - and
// still return success. Here the deadline arrives as ONE instant and each stage spends only
// what is left of it at its own entry; nothing left means one check and no wait.
//   deadline_abs_ms < 0 = INFINITE. E-2b-0 (durable (g)): this used to add "(the production
//   path, behaviour unchanged)", which E-2a itself had already made false - the production
//   stop hands down a FINITE MOE_DRAIN_DEADLINE_MS (see dispatcher_stop_and_join below).
//   Only the module's internal unbounded wrappers still reach here with a negative value.
// When the drain is the caller this instant came straight off the drain_begin anchor, so the
// whole chain spends one budget measured from the DRAINING entry, with no re-derivation.
int dispatcher_stop_and_join_until(bool abort_mode, int64_t deadline_abs_ms) {
    // Clear the stage record for THIS stop before any early return, so "did the join stage run"
    // is answered about this call and not about some earlier one.
    xsfac_stopjoin_clear();
    if (!g_mod.disp_running.load(std::memory_order_acquire)) return 1;
    // ★★T-1 — 이미 timeout 으로 한 번 끝난 상태면 **재진입하지 않는다**
    //   (그 dispatcher 는 아직 돌고 있으므로 다시 기다리면 그자리서 교착한다).
    if (g_mod.disp_stop_timed_out) return 0;
    // ★R1 #3 ② — dispatcher 정지 전에 intent 생산자(predictor)를 먼저 닫고 drain 한다
    //   (§2 graceful 종료 시퀀스: "신규 capture/command 거절 → predictor stop/join → drain").
    //   이 래치가 서기 전에는 dispatcher graceful exit이 금지되므로 여기서 멱등으로 보장한다.
    ggml_moe_phaseb_cut_close_capture();
    // T-1 - the predictor drain is bounded too; when it was INFINITE a predictor-stop
    // regression deadlocked before the timeout could even be measured (r3 S-2 (1)).
    // E-2a - it is now handed the SAME deadline, not a fresh budget of the same size.
    if (!ggml_moe_phaseb_cut_drain_predictor_until(deadline_abs_ms)) {
        g_mod.disp_stop_timed_out = true;
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(g_mod.q_mutex);
        g_mod.q_stopping = true;
        g_mod.q_abort    = abort_mode;
        g_mod.q_pause_req = false;
        g_mod.q_free_cv.notify_all();   // 만석 대기 생산자 해제(신규 command는 거절된다)
        SetEvent(g_mod.command_event);
        SetEvent(g_mod.resume_event);
    }
    // Whatever the predictor already spent is gone from this number. This is the last stage of
    // the chain, and the place where the old form spent a third full copy of the budget.
    const int32_t left = xstore_budget_left_ms(deadline_abs_ms);
    xsfac_stopjoin_record(deadline_abs_ms, left);
    const DWORD wr = WaitForSingleObject((HANDLE)g_mod.disp_thread,
                                        left < 0 ? INFINITE : (DWORD)left);
    if (wr != WAIT_OBJECT_0) { g_mod.disp_stop_timed_out = true; return 0; }   // ★교착 대신 실패(cleanup 금지)
    CloseHandle((HANDLE)g_mod.disp_thread);
    g_mod.disp_thread = 0;
    g_mod.disp_running.store(false, std::memory_order_release);

    std::lock_guard<std::mutex> lock(g_mod.mutex);
    dispatcher_cleanup_locked();
    return 1;
}

// Relative-timeout wrapper: pin one deadline at entry and pass only that one downwards. The
// existing callers - the production stop wrapper (finite, MOE_DRAIN_DEADLINE_MS), the 8000 ms
// teardown, the test window - keep their text unchanged and inherit the single budget.
// (E-2b-0 durable (g): "the production INFINITE wrapper" was the old wording here, and E-2a's
//  own G-1 change had already made it false.)
int dispatcher_stop_and_join_bounded(bool abort_mode, int32_t timeout_ms) {
    return dispatcher_stop_and_join_until(abort_mode, xstore_budget_deadline_ms(timeout_ms));
}

// The production stop path. E-2b-0 (durable (g)): the comment that stood here claimed the
// signature AND the behaviour were unchanged, with INFINITE named as the behaviour. The
// SIGNATURE is unchanged; the POLICY is not - E-2a replaced the unbounded stop with the finite
// MOE_DRAIN_DEADLINE_MS (contract section 2 row 7), which is what the body below passes.
void dispatcher_stop_and_join(bool abort_mode) {
    // ★★T-1 — 유계 종료가 timeout 으로 끝난 뒤라면 **여기서도 들어가지 않는다**.
    //   `reset_for_test` 가 이 래퍼를 불러 정지에 재진입하는 것이 r3 S-2 ② 의
    //   "FAIL 기록 후 교착" 경로다. 래치는 테스트 유계 훅이 timeout 했을 때만 선다.
    //   E-2b-0 durable (g): the phrase corrected above claimed re-entry into an UNBOUNDED
    //   wait. Since E-2a that re-entry would be into a finite stop; the block still stands.
    if (g_mod.disp_stop_timed_out) {
        log_err("[moe-direct] dispatcher stop previously timed out - skipping the unbounded join "
                "(the gate already recorded a failure; re-entering would deadlock)");
        return;
    }
    // E-2a (contract section 2 row 7): the production stop policy is no longer
    // INFINITE. MOE_DRAIN_DEADLINE_MS is the frozen finite default, so a stop that
    // never completes lands as a recorded failure instead of a stall. The test
    // caller-supplied timeout path is unchanged - it calls the bounded form directly.
    (void)dispatcher_stop_and_join_bounded(abort_mode, /*timeout_ms=*/MOE_DRAIN_DEADLINE_MS);
}

// 영속 경로가 선택돼 있는가(selector는 seal 후 immutable — sealed는 release-store로 publish된다).
inline bool persistent_selected() {
    return g_mod.sealed.load(std::memory_order_acquire) && g_mod.io_path_persistent;
}

// 영속 경로 활성 여부(= command 수용 가능).
bool persistent_active() {
    return persistent_selected() && g_mod.disp_running.load(std::memory_order_acquire);
}

// ★영속 경로가 선택됐는데 dispatcher가 정지(STOPPING/종료)한 상태 = 신규 command 거절(§2 ⑤).
//   여기서 legacy 배치 pump로 조용히 우회하면 "종료 후에도 I/O가 계속되는" 계약 위반이 된다.
bool persistent_fail_closed() {
    return persistent_selected() && !g_mod.disp_running.load(std::memory_order_acquire);
}

// control_q 고정비 — command 구조체 + ids 사본 + (B2) 요청 인덱스별 join 대상 op 표.
// ★b1_mem·phaseb_mem 두 echo가 **같은 이 식**을 쓴다(SPEC_PHASEB_WIRE §9 게이트 16-d).
// ★R-B4b(RB4B_ATOM_SPEC.json :: seam_map.members `b4-control-q-bytes`): 익명 ns → named ns
//   승격 + `inline` 제거. 관측 TU 의 `metrics_write_header_locked` 가 이것을 부른다 —
//   정의는 여기 한 벌이고 관측 헤더가 프로토타입을 든다(본문 불변). R-B5a 가 그 선언을
//   dispatch 헤더로 옮긴다(RB_DESIGN 1170).
uint64_t control_q_bytes_locked() {
    return (uint64_t)MOE_DIRECT_CONTROL_Q_CAP *
           (uint64_t)(sizeof(ControlCmd) + 2 * cq_ids_capacity() * sizeof(int32_t));
}

}  // namespace ggml_moe_direct_impl

// R-B5b atom D-R1 (RB5B_ATOM_SPEC.json :: move_rows_extra ->
// atom2_reserved.denominators.b5b_move.q_lock_move_set).  The five definitions below are
// the core text at the span the move map names, moved verbatim.  They keep `extern "C"`
// linkage - a C entry point that acquired the named namespace would stop being the symbol
// the rest of the tree links against - so they sit outside `ggml_moe_direct_impl`, in the
// same `extern "C"` region shape they had in the core TU, and in their source order.
// The only declarator change is `remove-inline` on `dfac_q_request_stop`
// (RB5B_ATOM_SPEC.json :: inline_dispositions): `ggml_moe_direct_finalize` stays in the
// core TU and calls it, so the specifier goes and ggml-moe-direct-dispatch.h carries the
// prototype (`header_extra_prototypes`).  RB_DESIGN 2-6 r5-2 clause 1 is what asks for
// this move: every `g_mod.q_mutex` lock and token now sits inside this unit.
extern "C" {

// R-B5a facade (vi): API and TEST dispatcher state (RB_DESIGN.md:1221 section 2-6
// (vi)). `dfac_q_request_stop` takes the WHOLE `q_mutex` scope it replaced - both
// SetEvent calls included - so the lock covers exactly what it covered before; it is
// the only member here that takes a lock, and it is not `_locked` for that reason.
void dfac_q_request_stop() {
    std::lock_guard<std::mutex> lk(g_mod.q_mutex);
    g_mod.q_stopping = true;
    g_mod.q_abort    = false;
    g_mod.q_pause_req = false;
    g_mod.q_free_cv.notify_all();
    SetEvent(g_mod.command_event);
    SetEvent(g_mod.resume_event);
}

// R-B5a facade (vi): the third `q_mutex` site (RB_DESIGN.md:1221 section 2-6 (vi), lock gate 1
// of r5-2). The test entry point used to take the queue lock itself; the member takes the same
// lock over the same read and hands back the value.
inline bool dfac_q_stopping_get() {
    std::lock_guard<std::mutex> lk(g_mod.q_mutex);
    return g_mod.q_stopping;
}

// ★★T-4(교차 r3 S-5 ②) — 엔진이 **`q_stopping=true` 를 게시했는가**(테스트 전용).
//   구 V3P16 은 shutdown 함수 **호출 직전**에 플래그를 세워 barrier 라 불렀는데, 그건
//   엔진이 정지 국면에 **들어갔다는 증거가 아니다**(함수 안에서 predictor drain 이 먼저
//   돌고, 그 사이에 demand 가 선완주할 수 있다). 이 창구가 **진짜 barrier** 다.
int ggml_moe_direct_test_q_stopping(void) {
    return dfac_q_stopping_get() ? 1 : 0;
}

// R-B5a facade (vi): API and TEST dispatcher state (RB_DESIGN.md:1221 section 2-6
// (vi)). `dfac_q_request_stop` takes the WHOLE `q_mutex` scope it replaced - both
// SetEvent calls included - so the lock covers exactly what it covered before; it is
// the only member here that takes a lock, and it is not `_locked` for that reason.
inline void dfac_q_pause_set_locked(bool on) {
    g_mod.q_pause_req = on;
}

// 테스트 전용 — dispatcher를 pass 경계에서 정지/재개(control_q 만석 대기 픽스처용).
// on!=0이면 dispatcher가 실제로 정지할 때까지 블록한다(결정론 확보).
void ggml_moe_direct_test_dispatcher_pause(int32_t on) {
    if (!g_mod.disp_running.load(std::memory_order_acquire)) return;
    std::unique_lock<std::mutex> lk(g_mod.q_mutex);
    if (on) {
        dfac_q_pause_set_locked(true);
        SetEvent(g_mod.command_event); // 대기 중이면 깨워서 pause 지점으로 보낸다
        g_mod.q_pause_cv.wait(lk, [] { return g_mod.q_paused || g_mod.q_stopping; });
    } else {
        dfac_q_pause_set_locked(false);
        SetEvent(g_mod.resume_event);
        g_mod.q_pause_cv.wait(lk, [] { return !g_mod.q_paused || g_mod.q_stopping; });
    }
}

}  // extern "C"

#endif  // _WIN32
