#pragma once

// R-B2 (RB_DESIGN.md 2-3, RB2_ATOM_SPEC.json): the module-state seam.  Every definition
// below is the core text at the span the move map names, moved verbatim - body tokens
// unchanged, members neither added, removed, reordered nor retyped.  The three globals
// whose DEFINITION stays in ggml-moe-direct.cpp are declared here and defined there.
// Windows only, exactly like the region it came from: the core TU compiles its
// non-Windows half from the stub region at the bottom of that file, and this unit has
// no stub half of its own (the moved definitions have no stub counterparts).

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-moe-phaseb.h"
#include "ggml-moe-direct-types.h"
#include "ggml-moe-direct-manifest.h"

namespace ggml_moe_direct_impl {

struct RegEntry {
    int32_t mlpos;     // moe_layers 내 위치(0-based)
    int32_t part_idx;  // layers[mlpos].parts 내 위치
};

struct PendingReg {
    const ggml_tensor * tensor;
    std::string         name;    // source_tensor
    bool                is_host;
};

struct Slot {
    void *                base = nullptr;
    std::atomic<int32_t>  lease{0};
    std::atomic<uint64_t> generation{0};
    int64_t               owner_record = -1;
    uint64_t              last_used = 0;
    uint8_t               cache_class = GGML_MOE_PB_CLASS_NONE;
    uint64_t              pf_intent_ordinal = GGML_MOE_PHASEB_NA_U64;
    uint16_t              pf_rank = GGML_MOE_PHASEB_NA_U16;  // ★R1 #7 — probation 비교 키의 rank 성분
    int32_t               pf_io_index = -1;   // 이 슬롯을 publish한 pb_io 레코드(-1=없음)
    // ★5c-T §1 소유권 전파 폐합(SPEC_PREFETCH_5CT_TELEMETRY §1 · WIRE 부록 14.3) —
    //   READY slot 도 §6 참여 집합 귀속에 owner stamp 가 필요하다. publish 시 op 의 스탬프를
    //   그대로 전사하고(ambient 전역 재조회 금지) pf_* 리셋 지점에서 함께 NA 로 되돌린다.
    //   ★`sizeof(Slot)` 은 어떤 metrics/mem echo 에도 실리지 않는다(슬롯 데이터 버퍼는 별도
    //   풀) — 확인 결과 pending_table_bytes/intent_record_bytes 만 sizeof 를 echo 한다.
    uint32_t              pf_request_epoch = GGML_MOE_PHASEB_NA_U32;  // owner ref
    uint32_t              pf_graph_seq     = GGML_MOE_PHASEB_NA_U32;
    uint8_t               pf_stamp_k = 0;    // owner generation 스탬프 K(0 = NA)
    uint8_t               pf_stamp_n = 0;    // owner generation 스탬프 N(0 = NA)
    uint8_t               pf_stamp_qd = 0;   // 〃 QD
    uint8_t               pf_stamp_rev = 0;  // 〃 tuple_revision
    uint64_t              pf_gen_admission_seq = GGML_MOE_PHASEB_NA_U64;  // 〃 generation 순번
};

struct RecordEntry {
    std::atomic<int32_t>  slot_index{-1};
    std::atomic<uint64_t> slot_generation{0};
};

struct ProtInterval {
    uintptr_t start;
    uintptr_t end;
    uint32_t  page_base_index;
    int32_t   tensor_idx;
    // ★§Z-⑥ A2 — PAGE_NOACCESS 를 걸기 직전의 보호값. mode=virtual 의 경계 teardown 이 등록
    //   텐서를 seal 시도 전 상태로 되돌리는 데 쓴다(정렬 후에도 구간과 함께 움직여야 하므로
    //   병렬 배열이 아니라 이 구조체가 들고 있다). 0 = 미기록(v2 경로·구 항목).
    uint32_t  old_protect = 0;
};

struct RegisteredTensorInfo {
    int32_t   layer;
    int32_t   part;
    uintptr_t addr_start;
    size_t    nbytes;
};

struct TouchLogEntry {
    std::atomic<uint64_t> seq{0};
    uintptr_t addr = 0;
    uintptr_t rip = 0;
    uint32_t  tid = 0;
    int32_t   tensor_idx = -1;
};

enum class Phase { DECODE = 0, PREFILL = 1, COUNT = 2 };

struct PhaseCounters {
    std::atomic<uint64_t> read_count{0};
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<uint64_t> hit{0};
    std::atomic<uint64_t> miss{0};
    std::atomic<uint64_t> ensure_wait_ms_sum{0};
};

struct BatchMiss {
    int64_t  record_idx   = -1;
    int32_t  multiplicity = 0;   // 원래 요청 횟수 — lease를 이만큼 올린다(release underflow 방지)
    int32_t  slot         = -1;
    uint64_t generation   = 0;
    uint64_t file_offset  = 0;
    // ★논리 요청 순서로 예약된 LRU tick(claim에서 last_used에 그대로 대입 — claim 시점에 새 tick을
    // 발급하지 않는다). 배치 안에 같은 record가 여러 번 나오면 **마지막 출현**의 tick을 쓴다(구
    // 순차 구현에서 둘째 접근이 hit로 잡혀 LRU를 refresh하던 의미와 일치).
    uint64_t logical_last_tick = 0;
    uint64_t trace_op_id  = 0;        // O1 상관자(io_complete에 그대로 전달)
    uint64_t trace_wait_episode = 0;  // O3 상관자(O4에서 닫음)
    // ★B2 legacy 경로 pb_io 상관자(R1 #6 — selector 무관 demand 실값). -1 = 미기록.
    int32_t  pb_io_index  = -1;
    // ★Phase A op-level 사이드카 상관자(PA off면 채워지지 않고 읽히지도 않는다).
    uint64_t pa_submit_seq = 0;
    uint64_t pa_submit_qpc = 0;
    uint32_t pa_batch_order = 0;
    uint16_t trace_run_index = 0;
    uint32_t trace_graph_seq = 0;
    uint8_t  trace_layer  = 0;
    uint8_t  trace_expert = 0;
    uint8_t  trace_phase  = MOE_TRACE_PHASE_UNKNOWN;
};

struct IoChannel {
    OVERLAPPED ov{};
    HANDLE     ev = nullptr;   // manual-reset(ReadFile이 시작 시 non-signaled로 낮추고 완료 시 올림)
    size_t     miss_idx = 0;   // legacy 배치 pump 전용(misses[] 인덱스)
    int32_t    op_index = -1;  // ★B1 영속 경로 전용(pending_ops[] 인덱스 · -1=미점유)
    bool       busy = false;
};

constexpr int32_t MOE_DIRECT_CONTROL_Q_CAP  = 64;

enum CmdKind : uint8_t {
    CMD_DEMAND_ACQUIRE = 0,
    CMD_RELEASE        = 1,
    CMD_STOP           = 2,
    CMD_TEST_QUERY     = 3,
};

enum QueryKind : uint8_t {
    QK_ABSENT_COUNT = 0,
    QK_RECORD_SLOT  = 1,
    QK_SLOT_STATE   = 2,
    QK_LEASE_STATS  = 3,
};

enum PendingState : uint8_t { PIO_CLAIMED = 0, PIO_SUBMITTED = 1, PIO_TERMINAL = 2 };

enum PendingOrigin : uint8_t { PIO_ORIGIN_DEMAND = 0, PIO_ORIGIN_PREFETCH = 1 };

struct PendingIo {
    BatchMiss m;                       // record/slot/generation/offset/multiplicity/tick/상관자
    int32_t   mlpos    = -1;
    Phase     phase    = Phase::DECODE;
    uint8_t   origin   = PIO_ORIGIN_DEMAND;
    uint8_t   state    = PIO_CLAIMED;
    int32_t   channel  = -1;
    uint64_t  request_identity = 0;    // 최초 waiter command의 요청 seq
    uint64_t  intent_seq = 0;          // C1′ 결정론 키의 전사 = intent_ordinal(demand=NA)
    int32_t   retry_of = -1;           // 재시도 원 op의 pending 인덱스(-1=아님)
    bool      in_use   = false;
    std::vector<int32_t> waiters;      // 대기 중인 command slot 인덱스
    // ── ★B2 (SPEC_PHASEB_WIRE §5) ──
    uint32_t  pb_request_epoch = GGML_MOE_PHASEB_NA_U32;  // 보존 owner epoch(§0.1 귀속 규약)
    uint32_t  pb_graph_epoch   = GGML_MOE_PHASEB_NA_U32;
    uint32_t  pb_graph_seq     = GGML_MOE_PHASEB_NA_U32;
    uint16_t  pb_rank          = GGML_MOE_PHASEB_NA_U16;
    int32_t   pb_io_index      = -1;   // pb_io 레코드 인덱스(-1=미기록 모드)
    // ★R1 #9 (b) — joined retry의 owner epoch = **첫 waiter의 demand epoch**(원 prefetch 값 금지).
    uint32_t  pb_join_request_epoch = GGML_MOE_PHASEB_NA_U32;
    uint32_t  pb_join_graph_epoch   = GGML_MOE_PHASEB_NA_U32;
    uint32_t  pb_join_graph_seq     = GGML_MOE_PHASEB_NA_U32;
    int32_t   pb_join_cmd           = -1;   // 첫 waiter command(PA batch_order 별도 순번 발급용)
    uint64_t  pb_submit_event_seq   = GGML_MOE_PHASEB_NA_U64;
    uint64_t  pb_retry_of_event_seq = GGML_MOE_PHASEB_NA_U64;
    bool      pb_is_retry      = false;
    bool      pb_ever_joined   = false;  // ever_demand_consumed 래치(§6 used_join)
    uint64_t  pb_blocked_start = 0;      // demand submit 대기 시작 QPC(0=미대기)
};

struct Io3Header {
    char     magic[8];            // "MOEIO3" + 2 NUL
    uint32_t layout_version;      // 1
    uint32_t file_kind;           // 1=records 2=children 3=copies
    uint32_t record_bytes;        // 80 / 88 / 64
    uint32_t engine_mode;         // ★off 20 — 1=bin 2=virtual
    uint64_t record_count;
    uint64_t payload_digest;      // FNV-1a 64 over the exact payload bytes
    uint64_t qpc_frequency;
    uint64_t t0_qpc;
    uint32_t pid;
    uint32_t complete;            // 1 = arm 유효
    uint32_t n_expert;
    uint32_t n_moe_layers;
    uint32_t n_slots;
    uint32_t n_expert_used;
    uint64_t boot_id;             // = metrics header `io3_boot_id`(조인 키)
    uint64_t total_requests;
    uint64_t dropped_records;
    uint8_t  io_params[6];        // ★off 104..109 — resolved 6종(bin=전부 0)
    uint8_t  io_param_src[6];     // ★off 110..115 — source 코드 6종(bin=전부 0)
    uint8_t  reserved[12];        // off 116..127 = 0
};

struct Io3Record {                // kind 1 · 80B — record 채움 시도 1회(=한 generation)
    uint64_t record_seq;          // 프로세스 전역 단조(귀속 권위 키)
    uint64_t record_id;           // mlpos*n_expert + expert
    uint32_t graph_seq;
    uint32_t request_index;
    uint16_t layer;
    uint16_t expert;
    uint8_t  origin;              // 0=demand 1=prefetch
    uint8_t  flags;               // b0 phase(1=prefill) b1 is_retry b2 had_waiter_at_terminal
    uint8_t  terminal_status;     // 1=OK 2=READ_ERROR 3=COPY_ERROR 4=CANCELLED 5=SHUTDOWN
    uint8_t  attempt;             // 0=첫 시도 · 1=joined-prefetch retry(상한 1)
    uint32_t n_children;          // ★submit 사건에 **도달한** child 수(계획 수 아님)
    int32_t  slot;
    uint64_t legacy_bytes;        // = legacy_stride_bytes[layer](§7)
    uint64_t generation;
    uint64_t first_submit_qpc;
    uint64_t terminal_qpc;
    uint64_t publish_seq;         // 1-based · 0=미publish
};

struct Io3Child {                 // kind 2 · 88B — 물리 READ 제출 1건
    uint64_t record_seq;          // 단독=소속 records 행 · coalesced=NA
    uint64_t read_offset;
    uint32_t read_len;
    uint16_t layer;
    uint16_t expert;
    uint8_t  part;                // virtual=실제 part · bin=NA(0xFF)
    uint8_t  source_index;        // virtual=실 shard · bin=NA(0xFF)
    uint8_t  origin;
    uint8_t  flags;               // b0 pending b1 phase b2 is_retry b3 bounce b4 coalesced
    uint32_t coalesce_group;      // NA=0xFFFFFFFF
    uint64_t generation;
    uint64_t submit_event_seq;    // ★공용 event namespace(§5-6)
    uint64_t submit_qpc;
    uint64_t terminal_event_seq;
    uint64_t terminal_qpc;
    uint8_t  terminal_status;     // 1=OK 2=ERROR 3=CANCELLED
    uint8_t  qd_at_submit[3];     // demand/prefetch/total — 제출 직후(이 op 포함)
    uint8_t  qd_at_terminal[3];   // 〃 terminal 직후(이 op 제외)
    uint8_t  reserved;
    uint32_t graph_seq;
    uint32_t request_index;
};

struct Io3Copy {                  // kind 3 · 64B — ★copy **의무** 1건당 1행(실행 여부 무관)
    uint64_t record_seq;
    uint64_t src_child_submit_event_seq;   // 원 read child(유일 조인 키·NA 금지)
    uint32_t bytes;               // = slice_bytes(exact — 브라켓 미포함)
    uint16_t layer;
    uint16_t expert;
    uint8_t  part;
    uint8_t  worker;              // copy worker index · 미착수=0xFF
    uint8_t  status;              // 1=OK 2=ERROR 3=CANCELLED
    uint8_t  flags;               // b0 phase b1 src_kind(0=bounce 1=coalesce) b2 is_retry
    uint32_t coalesce_group;
    uint64_t enqueue_qpc;         // 미enqueue=0
    uint64_t start_qpc;           // 미착수=0
    uint64_t done_qpc;            // ★미enqueue 행만 0(§5-5 상태·시각 규약)
    uint64_t generation;
};

struct V3ChildIo {
    int32_t  n_children    = 0;   // 이 op 이 낼 물리 read 수(= 그 층의 part 수)
    int32_t  remaining     = 0;   // = remaining_reads(§3-4) — 아직 회수 안 된 물리 read 수
    // ★§Z-⑥ A4 — 바운스(비4K) child 는 read 회수 뒤에도 **copy 가 끝나야** 끝난 것이다.
    //   그래서 사양 §3-4 가 `remaining_reads` 와 `remaining_copies` 를 따로 둔다.
    //   op terminal 조건 = **둘 다 0**. aligned 전용 프로파일에서는 copies 가 항상 0 이라
    //   A3 의 거동과 동일하다.
    int32_t  remaining_copies = 0;
    uint64_t bytes_ok      = 0;   // 성공 회수한 child 바이트 합(진단·판정 보조)
    // ★§Z-⑥ A6 — 이 op 이 슬롯에 받아야 하는 총 바이트. 구 판은 층 상수
    //   (`child_bytes_expected`)를 그대로 썼지만 **coalesced part 는 브라켓이 아니라 slice 만**
    //   슬롯에 남기므로(scatter 가 exact slice 단위 — §3-6 ⑥) 층 상수와 갈린다. 제출 시점에
    //   part 별 처분(단독 aligned=region · 바운스·coalesced=slice)으로 확정한다.
    //   coalescing 이 없는 런에서는 층 상수와 값이 같다(거동 불변).
    uint64_t bytes_expected = 0;
    uint8_t  active        = 0;   // 이 op 이 virtual child 팬아웃을 쓰는가
    uint8_t  failure_latch = 0;   // ★첫 실패가 세운다. 이후 형제 성공은 publish 로 못 간다
    uint8_t  any_pending   = 0;   // child 중 하나라도 ERROR_IO_PENDING 이었나(was_pending 집계)
    uint8_t  cancel_issued = 0;   // 형제 CancelIoEx 1회성 래치
    uint8_t  status        = 0;   // 확정될 op terminal status(실패 래치가 정한다)
    // ★abort/shutdown drain 중인가. 서면 이 op 은 **어떤 경우에도 publish 로 가지 않는다** —
    //   전 child 가 성공해도 CANCEL_SUCCESS_UNPUB 이다(§6-4 ⑩ success-after-cancel publish 0).
    //   ERROR 로 닫으면 waiter 붙은 op 이 종료 중에 새 generation retry 를 발행해버린다.
    uint8_t  cancel_mode   = 0;
    // ★A5 — IO3 records 의 terminal_status 가 READ_ERROR / COPY_ERROR 를 구분해야 한다(§5-3).
    uint8_t  copy_error    = 0;
    // ★R12 — §5-3 `terminal_qpc` 는 **성공=publish 직전 · 실패=latch 시각**이다. drain 이 끝난
    //   시각(op terminal)을 실패 행에 찍으면 "언제 틀어졌나"가 사라진다(형제 취소 회수 시간이
    //   실패 시각으로 둔갑). 첫 실패/취소가 이 값을 세우고 op terminal 이 그대로 쓴다(0=미설정).
    uint64_t terminal_latch_qpc = 0;
    // ── ★★R-1 (FROZEN §2-0 A0-2 wave 상태기계) ──────────────────────────────
    //   구 판은 "record 의 전 child 를 한 번에 건다"(부분 제출 금지)였고, 그 금지를
    //   `io_part_parallelism >= max_parts` 형상 하한이 떠받쳤다. 사양은 그 반대를 동결한다:
    //     **동시 물리 폭 `W_l = min(P_l, io_part_parallelism)`** · record 는 **wave 순차 진행** ·
    //     **부분 제출 상태는 IO3 가 허용·검증하는 정상 상태**(gpt-oss 6-part = 명시된 다중 wave).
    //   `wave_next` = 다음 wave 가 제출할 **첫 part 인덱스**(= 지금까지 제출한 part 수).
    //     0            → 아직 한 wave 도 안 나갔다(admission 예약 보유 상태)
    //     0 < x < n_children → **부분 제출 중**(정상) · x == n_children → 제출 완주
    //   ★다음 wave 개시 조건에 `remaining + n_need <= io_part_parallelism` 이 붙는다 —
    //     IO3 §5-6 의 **part cap**("record 별 **동시** child membership ≤ io_part_parallelism")과
    //     A0-2 를 화해시키는 지점이다(wave k 가 아직 in-flight 인데 k+1 을 띄우면 membership 이
    //     2W 가 되어 그 cap 을 깬다). "wave 순차 진행"의 문자 그대로다.
    int32_t  wave_next     = 0;
};

struct PbvRec {
    uint8_t  active    = 0;   // admission commit ~ record terminal(①축 gauge 소속 구간)
    uint8_t  submitted = 0;   // 첫 child submit 이 났다(reservation→inflight 전환 완료)
    // ── ★A3(③ 논리 disposition 축) ──
    uint8_t  issued    = 0;   // A0-3: `issued ⇔ 성공 제출 child ≥ 1`(mode-neutral latch)
    uint8_t  disposed  = 0;   // 최종 rank disposition 1회 래치(issued / submit_failed)
    int32_t  reserved  = 0;   // 이 record 가 **현재 실제 보유한** child credit 예약분
    int32_t  wave      = 0;   // W_l = min(P_l, io_part_parallelism)
};

enum : uint8_t { V3_COAL_OFF = 0, V3_COAL_PREFILL = 1, V3_COAL_ALL = 2 };

enum : uint8_t {
    V3_COAL_R_NONE      = 0,   // 합침 성립
    V3_COAL_R_DISABLED  = 1,   // 기능 off(파라미터)
    V3_COAL_R_PHASE     = 2,   // §3-6 프리필 한정 — 디코드 요청
    V3_COAL_R_HOMOGEN   = 3,   // §5-4 8번째 MUST — origin/phase/graph_seq/request_index/is_retry
    V3_COAL_R_RUN       = 4,   // ② expert ID 연속 run 이 2 미만(gap=0B — 건너뛰기 금지)
    V3_COAL_R_SOURCE    = 5,   // ① 동일 source_index·동일 source tensor/part 위반
    V3_COAL_R_BOUNDARY  = 6,   // ③ 텐서·shard 경계 월경
    V3_COAL_R_ALIGN     = 7,   // ④ 브라켓 시작·길이·staging 정렬/용량
    V3_COAL_R_CLAIM     = 8,   // ⑤ 대상 record 사전 claim·pin 미충족
    V3_COAL_R_MIN       = 9,   // §5 io_coalesce_min — 합칠 값이 없다
    V3_COAL_R_RESOURCE  = 10,  // coalesce 버퍼·채널 부족(형상이 아니라 순간 자원)
};

struct V3CoalRead {
    uint8_t  active     = 0;
    int32_t  mlpos      = -1;
    int32_t  part       = -1;
    int32_t  expert0    = 0;          // run 시작 expert(= IO3 children.expert)
    uint32_t group_id   = 0xFFFFFFFFu;
    uint64_t offset     = 0;          // 물리 read 시작(A 배수)
    uint32_t len        = 0;          // 물리 read 길이(A 배수)
    uint64_t src_head   = 0;          // 대구간 안에서 expert0 슬라이스가 시작하는 바이트
    int32_t  pending_copies = 0;      // 이 대구간이 붙들고 있는 scatter copy 수(0 = 버퍼 반납)
    int32_t  io3_copy_row0  = -1;     // 참가 순서대로 연속 push 된 copies 행의 첫 인덱스
    std::vector<int32_t> ops;         // participant op 인덱스(expert 오름차순)
};

struct V3CoalPlan {
    int32_t  mlpos   = -1;
    int32_t  expert0 = 0;
    int32_t  n_parts = 0;
    std::vector<int32_t> ops;              // 참가 op(expert 오름차순 · 크기 = run 길이)
    std::vector<int32_t> cand;             // expert → op 인덱스(-1) 스크래치
    uint64_t off [64] = { 0 };             // part 별 물리 read 시작(A 배수)
    uint32_t len [64] = { 0 };             // 〃 길이(A 배수)
    uint64_t head[64] = { 0 };             // 〃 대구간 안에서 expert0 슬라이스가 시작하는 위치
};

struct PbOpStamp {
    uint8_t  k = 0;                                   // owner 스탬프 K(0 = NA)
    uint8_t  n = 0;                                   // owner 스탬프 N(0 = NA)
    uint8_t  qd = 0;                                  // owner 스탬프 QD(seal 고정)
    uint8_t  rev = 0;                                 // owner 스탬프 tuple_revision(5c-T=0)
    uint8_t  partitioned = 0;                         // 4분할 확정 래치(중복 전이 금지)
    uint8_t  partition = GGML_MOE_PB_P_NA;            // 확정된 분할(진단·oracle)
    uint32_t graph_epoch = GGML_MOE_PHASEB_NA_U32;    // owner ref 보조
    uint64_t gen_admission_seq = GGML_MOE_PHASEB_NA_U64;  // ★r1 M1 owner generation 순번
    // ★r1 M3 — admission 결정 레코드는 **제출 성공/실패가 확정된 뒤** 방출한다. 그때까지
    //   완전한 snapshot 을 이 자리에 보관한다(rank 당 정확히 1건 · 조기 ISSUED 방출 폐지).
    uint8_t  admit_rec_valid = 0;
    ggml_moe_phaseb_ledger admit_rec{};
    // ★r1 M4 — blocked interval 의 참여 집합은 **대기 시작 순간**에 고정한다(종료 시 재스캔은
    //   retire 선행으로 실제 blocker 가 사라져 빈 집합을 남기는 인과 누락이었다).
    std::vector<ggml_moe_phaseb_ledger_owner> blocked_parts;
};

struct ControlCmd {
    uint8_t  kind  = CMD_DEMAND_ACQUIRE;
    uint8_t  qkind = QK_ABSENT_COUNT;
    int32_t  mlpos = -1;
    int32_t  part_idx = -1;
    int32_t  n_ids = 0;
    Phase    phase = Phase::DECODE;
    std::vector<int32_t> ids;          // 입력 복사(façade ①에서 복사 — capacity 수렴 후 무할당)
    uint64_t pa_t_enter = 0;
    uint64_t seq = 0;                  // request identity
    // dependency — 아직 publish되지 않은 required record 수(0이 되는 순간에만 success signal).
    int32_t  deps = 0;
    // PA R_j(acquire_exit) 인자 — 마지막 terminal 직전에 발행.
    int32_t  pa_n_miss = 0;
    uint64_t pa_bytes  = 0;
    uint32_t pa_batch_next = 0;   // ★R1 #9 ④ — retry op의 PA batch_order 별도 순번
    bool     pa_open   = false;
    // TestQuery 입출력
    int64_t  q_arg_i64 = 0;
    int32_t  q_arg_i32 = 0;
    int64_t  q_out_i64 = 0;
    uint64_t q_out_u64_a = 0;
    uint64_t q_out_u64_b = 0;
    int32_t  q_out_i32_a = 0;
    int32_t  q_out_i32_b = 0;
    // ── ★B2 (SPEC_PHASEB_WIRE §4 pb_sample) ──
    uint32_t pb_request_epoch  = GGML_MOE_PHASEB_NA_U32;  // epoch 투영(§2.1) — façade에서 채취
    uint32_t pb_graph_epoch    = GGML_MOE_PHASEB_NA_U32;
    uint32_t pb_graph_seq      = GGML_MOE_PHASEB_NA_U32;
    uint32_t sp_ready = 0, sp_join = 0, sp_demand_read = 0, sp_join_attempt = 0; // 배타 3분할
    int32_t  pb_row = -1;               // 이 command가 채운 pb_sample 행(-1=행 미버퍼링)
    bool     pb_canonical = false;      // 이 (graph, layer)의 첫 acquire인가(PA ledger canonical과 1:1)
    std::vector<int32_t> req_join_op;   // 요청 인덱스별 join 대상 op(-1) — retry 시 join→demand_read 이동
    // 완료 핸드셰이크(q_mutex 아래에서만 접근)
    int32_t  result = 0;
    bool     done   = false;
    bool     in_use = false;
    std::condition_variable cv;
};

struct DemandPlan {
    std::vector<int32_t>   hit_slot;  // 요청 인덱스별 hit 슬롯(-1 = hit 아님)
    std::vector<int32_t>   req_miss;  // 요청 인덱스별 유니크 miss 인덱스(-1)
    std::vector<int32_t>   req_join;  // 요청 인덱스별 join 대상 pending op(-1) — 영속 전용
    std::vector<uint8_t>   slot_excl; // 슬롯별 배치 보호(hit/claimed/pending pin)
    std::vector<BatchMiss> miss;      // 유니크 miss(요청 논리 순서 = 첫 출현 순)
    std::vector<uint64_t>  pa_tick;   // ★PA 전용 — 요청 인덱스별 발급 tick
    std::vector<int32_t>   join_ops;  // 이 배치가 join한 유니크 pending op 목록
};

struct MetricsWriter {
    bool    enabled = false;
    HANDLE  file = INVALID_HANDLE_VALUE;
    // ★R18 — 열 때 쓴 경로. **실패한 seal 이 header 만 든 파일을 남기지 않도록** 되돌릴 때 쓴다
    //   (`CREATE_NEW` 배타 생성이라 잔류 파일은 같은 경로 재시도까지 막는다).
    std::wstring path_w;
    uint64_t seq = 0;
    std::string invocation, session, leg;
    uint32_t ops_since_periodic = 0;
    std::chrono::steady_clock::time_point last_periodic;
    bool finalized = false;

    void reset() {
        if (file != INVALID_HANDLE_VALUE && file != NULL) {
            CloseHandle(file);
        }
        file = INVALID_HANDLE_VALUE;
        enabled = false;
        seq = 0;
        invocation.clear();
        session.clear();
        leg.clear();
        ops_since_periodic = 0;
        finalized = false;
    }
};

struct ModuleState {
    std::mutex mutex;
    std::atomic<bool> sealed{false};

    // 등록 대기(seal 전 register()로 누적) + 확정 registry(R_host).
    std::vector<PendingReg> pending;
    std::unordered_map<const ggml_tensor *, RegEntry> registry;

    // manifest 파생 모델 메타
    std::string arch;
    int32_t  n_layer = 0;
    int32_t  n_expert = 0;
    int32_t  n_expert_used = 0;
    std::vector<int32_t> moe_layers;                 // 오름차순 실제 layer id
    std::unordered_map<int32_t, int32_t> layer_to_mlpos;

    // manifest 파생 레이아웃
    std::vector<LayerInfo> layers;   // 크기 = n_moe_layers, 인덱스 = mlpos
    std::vector<SourceInfo> sources;
    uint64_t align_bytes = 0;
    uint64_t slot_stride_max = 0;
    uint64_t bin_file_bytes = 0;
    std::string ref_profile_id;
    std::string ref_expect_sha256;
    // ★3단계 R4 — seal 게이트 ⑦이 이미 계산·대조한 **manifest.json 실 바이트 SHA-256**.
    //   `ref_expect_sha256` 는 profile 별 **승인 expect** 의 SHA 라서 같은 profile/expect 로 정상
    //   재팩한 **다른 manifest**(다른 재팩 산출물)를 구분하지 못한다 — 그 축을 metrics header 로
    //   내보내 계열 판독기의 cross-run identity 가 재팩물 단위까지 결속되게 한다.
    //   (experts.bin 재해시는 하지 않는다 — 신뢰 입력 모델·verify 게이트 계보 그대로.)
    std::string repack_manifest_sha256;
    // OPEN_ARCH B axis - the inventory digest the template branch RECOMPUTES INDEPENDENTLY from the
    // live tensors. Empty string on the model-pin path (that path never runs the template closure).
    std::string template_inventory_sha256;
    // ★R13 — manifest `schema_version` 원문("2.0"=bin · "3.0"=virtual). metrics header echo 전용.
    std::string manifest_schema_version;

    // 등록 attestation
    int32_t att_all = 0;      // ★부록A 이후 의미=managed(scope 적용 후 관리 대상). all scope면 raw와 동일.
    int32_t att_host = 0;
    int32_t att_nonhost = 0;
    int32_t att_n_slots = 0;
    int32_t att_n_slots_required = 0;
    // ★부록A dual-scope attestation 신규 3필드(기존 att_all/host/nonhost 의미·이름 불변).
    int32_t att_raw = 0;            // 등록 전수(pending 전체)
    int32_t att_managed = 0;        // scope 적용 후 관리 대상(= att_all)
    int32_t att_excluded_nextn = 0; // execution 제외 수(= raw - managed·all이면 0)
    std::string routed_scope = "all";

    // 파일/슬롯 풀
    // ★§Z-⑥ A2 — 주소 계약 분기의 단일 출처(SPEC_REPACK_V3 §3-3). BIN 이 기본이므로 v2 경로의
    //   의미는 불변이다.
    ManifestMode manifest_mode = ManifestMode::BIN;
    HANDLE   data_file = INVALID_HANDLE_VALUE;
    // ★§Z-⑥ A2 — mode=virtual 의 소스 핸들 집합(§3-1). bin 의 단일 `data_file` 에 대응하는
    //   축이며 인덱스 = sources[].index. bin 경로에서는 항상 비어 있다.
    std::vector<HANDLE> source_files;
    void *   pool_base = nullptr;
    size_t   pool_bytes = 0;
    int32_t  n_slots = 0;
    std::vector<Slot> slots;
    std::vector<RecordEntry> record_table; // n_moe_layers*n_expert
    uint64_t tick = 0;

    // =======================================================================
    // ★섀도 드래프트 관측 view (SPEC_SHADOW_DRAFT §3 C3) — 프로덕션 기본 완전 비활성.
    // view 가 열려 있는 동안 캐시 mutation(demand/prefetch/claim/victim/tick/lease/class/
    // pending)은 **거부**된다. 섀도는 토큰 확정 직후 동기 실행이라(SPEC D6) 정상 상태에서
    // main 의 mutation 과 겹치지 않는다 — 겹치면 그것이 결함이므로 조용히 통과시키지 않는다.
    // =======================================================================
    std::atomic<int32_t> shadow_view_open{0};   // 0=닫힘 1=열림(락 밖 생산자가 읽는다)
    std::vector<int32_t>  shadow_view_slot;     // record_idx -> slot(-1 = 비상주)
    std::vector<uint64_t> shadow_view_gen;      // record_idx -> 스냅샷 시점 slot generation
    uint64_t shadow_view_tick_begin      = 0;   // 진입 시 tick
    uint64_t shadow_view_digest_begin    = 0;   // 진입 시 cache/LRU digest
    uint64_t shadow_view_readfile_begin  = 0;   // 진입 시 읽기 제출 수(읽기 제출 0 증명)
    uint64_t shadow_view_refused         = 0;   // 동결 중 거부된 mutation 시도 수(0이어야 정상)

    // ★게이트 무관 읽기 제출 계수. 기존 readfile_ops 는 metrics 활성 시에만 증가하므로
    //   (digest_on 분기) C3 의 "읽기 제출 0" 판정에 쓰면 metrics OFF 에서 항상 참이 되는
    //   빈 검사가 된다. 이 계수는 io_submit_locked 진입마다 무조건 증가한다.
    uint64_t io_submit_ops = 0;

    // ★QD 레버 — seal마다 env에서 재해소한다(call_once 캐시 금지: QD를 바꿔 재seal하는 검증이
    // 가능해야 한다). metrics header에 정보성으로 실린다(판독기 소비 없음).
    int32_t     qd_effective = MOE_DIRECT_QD_DEFAULT;
    std::string qd_source    = "default";

    // =======================================================================
    // ★B1 영속 I/O dispatcher (PHASEB_DESIGN §2) — seal에서 1회 해소·기동, reset/shutdown에서 정지.
    // =======================================================================
    // selector 하드 구속(§2): ①qualification/test mode에서만 존재 ②seal 전 1회 해소·이후 immutable
    // ③미지정 production 기본값=persistent ④test mode 밖 legacy 요청=fail-close ⑤header echo.
    bool        io_path_persistent = true;
    std::string io_path_source     = "default";
    bool        io_path_qualified  = false;   // qualification/test mode 여부(echo)

    // dispatcher 스레드(★std::thread 미사용 — 전역 소멸자에서 joinable 스레드는 terminate를 부른다.
    // _beginthreadex는 스택 예약 크기를 명시할 수 있어 mem echo가 추정이 아닌 실제값이 된다).
    uintptr_t disp_thread   = 0;
    std::atomic<bool> disp_running{false};   // ★생산자가 락 없이 읽는다(persistent_active) — atomic 필수
    HANDLE    command_event = nullptr;   // handle 0 (manual reset)
    HANDLE    resume_event  = nullptr;   // 테스트 pause 해제용(manual reset)

    std::vector<IoChannel> channels;          // qd_effective — seal 시 1회 생성·주소 고정
    std::vector<PendingIo> pending_ops;       // 고정 풀(크기 n_slots — pending은 슬롯을 pin하므로 상한)
    std::vector<int32_t>   pending_free;      // 자유 인덱스 스택
    std::vector<int32_t>   pending_by_record; // n_moe_layers*n_expert 고정 인덱스(-1 = NA)
    std::vector<int32_t>   submit_q;          // 제출 대기 FIFO(pending op 인덱스)
    size_t                 submit_head = 0;
    std::vector<uint8_t>   slot_pending;      // 슬롯별 pending pin(terminal까지 축출/재사용 금지)
    // ★§Z-⑥ A3 — virtual child 상태 병렬 표(동결 구조에 필드를 못 넣는다 — V3ChildIo 주석).
    std::vector<V3ChildIo> v3_child;          // 크기 n_slots · 인덱스 = pending_ops 인덱스
    std::vector<int32_t>   v3_chan_op;        // 크기 qd_effective · 채널 → op 인덱스(-1=미점유)
    std::vector<int32_t>   v3_chan_part;      // 크기 qd_effective · 채널 → part 인덱스(-1=미점유)
    // ===================================================================
    // ★§Z-⑥ A4 — 바운스 경로(SPEC_REPACK_V3 §3-5 · 옵션 B · `aligned=false` part 한정)
    //   비4K part 는 브라켓 read 가 슬롯에 바로 못 앉는다(시작이 A 정렬이 아니다). 그래서
    //   **전용 staging** 으로 읽고, 채널은 **즉시 반환**한 뒤 bounded copy queue 가
    //   `staging + src_head(e)` → `slot_base + slot_offset[p]` 로 `slice_bytes` 만 옮긴다.
    //   ★dispatcher inline copy 금지 — memcpy 를 dispatcher 스레드에서 하면 그 시간만큼
    //   제출·회수가 멈춘다. 그래서 별도 copy worker 가 존재한다.
    // ===================================================================
    void *   v3_staging_base   = nullptr;  // VirtualAlloc(섹터 정렬 — NO_BUFFERING read 착지)
    size_t   v3_staging_stride = 0;        // 버퍼 1개 크기 = max(part.staging_bytes)
    int32_t  v3_staging_count  = 0;
    std::vector<uint8_t> v3_staging_busy;  // ★copy terminal 까지 재사용 금지(§3-5 refcount 소유권)
    std::vector<int32_t>   v3_chan_staging;  // 채널 → staging idx(-1 = 바운스 아님)
    // ★★R10(교차 r5 Y1) — 채널 → **그 child 가 실제로 제출한 read 길이**.
    //   완료 판정을 `pi.region` 으로 하면 비4K 에서 깨진다: 제출 길이는
    //   `align_up(src_head + slice, A)` 이고 region 은 `align_up(slice, A)` 라,
    //   `src_head > region - slice` 면 **정상 full read 가 region 보다 A 만큼 크다**.
    //   그러면 정상 완료가 실패 latch 로 가서 demand 가 fail-closed 된다.
    std::vector<uint32_t>  v3_chan_len;

    // bounded copy queue(§5 `io_bounce_copy_qd`) + worker.
    struct V3CopyJob {
        int32_t  oi = -1, part = -1, staging = -1;
        uint64_t bytes = 0;
        const char * src = nullptr;        // ★A6 — 원본 포인터를 **enqueue 시점에** 확정한다.
                                           //   풀이 둘(바운스 staging · coalesce 버퍼)이 되면서
                                           //   worker 가 풀 산술을 두 벌 갖는 것을 막는다.
        char *   dest = nullptr;
        uint8_t  ok = 1;                   // read 가 성공했나(실패면 copy 없이 실패 회수)
        int32_t  coal = -1;                // ★A6 — coalesce 대구간 인덱스(-1 = 바운스)
        int32_t  io3_copy_row = -1;        // ★A5/A6 — copies 행 인덱스(직접 결속).
                                           //   ★구 판은 `src_child_submit_event_seq` 로 역탐색
                                           //   했는데 coalesced 는 그 키가 **1:N**(참가 record
                                           //   마다 1행)이라 역탐색이 엉뚱한 행을 잡는다.
    };
    std::vector<V3CopyJob> v3_copy_q;      // 크기 = 용량(고정) · head/count 로 원형 운용
    int32_t  v3_copy_head = 0, v3_copy_count = 0, v3_copy_cap = 0;
    // ★★S1(교차 r8 Z2-2) — **공용 outstanding-copy credit**. 큰유는 바운스와 coalesce scatter 가
    //   함께 쓰는데, 구 판은 각 생산자가 자기 몫만 보고 제출했다 — 계획의 `copy_qd/parts` run
    //   상한은 **한 그룹의 필요조건**일 뿐 전역 안전조건이 아니다. 그래서 두 그룹이 공존하면
    //   회수 시점에 의무가 한꺼번에 몰려 큐가 넘치고 `fatal_abort` 로 갔다(유효한 opt-in 설정에서
    //   도달한다). ⇒ **제출 전에 원자 예약**하고 부족하면 아예 제출하지 않는다.
    //   ★계상 구간은 `예약~완료` 로 §5-6 의 `[enqueue, done)` 을 **포함하는 상위집합**이다
    //   (worker 가 꺼내 실행 중인 행도 완료 전까지 크레딧을 물고 있다 — cap 정의와 어긋나지 않는다).
    int32_t  v3_copy_outstanding = 0;
    uint64_t v3_copy_credit_stalls = 0;    // 크레딧 부족으로 미제출한 횟수(관문 근거)
    std::mutex v3_copy_mutex;              // ★g_mod.mutex 와 별개(worker 가 짧게만 잡는다)
    HANDLE   v3_copy_event = nullptr;      // 작업 도착 신호(auto-reset)
    uintptr_t v3_copy_thread = 0;
    std::atomic<bool> v3_copy_running{false};
    std::atomic<int32_t> v3_copy_pause{0}; // ★테스트 전용 — worker 를 멈춰 backpressure 를 만든다
    uint64_t v3_copies_done = 0, v3_copies_failed = 0;
    uint64_t v3_backpressure_stalls = 0;   // staging 부족으로 제출을 미룬 횟수(§6-4 ⑪ 관문 근거)
    // ★A5b — origin 별 **child** in-flight. IO3 children 의 QD 6필드 실값 근거다.
    //   PB 의 `pb_inflight_*` 는 **op** 단위 축이고 virtual 경로에선 증가하지 않는다 — 별개 축.
    // ★★PB virtual A1 — 이 둘이 곧 사양의 **②child 물리축**
    //   (`SPEC_IO_METRICS_V3` §3-5 의 `io_child_inflight_demand/prefetch` ·
    //    `SPEC_PB_VIRTUAL_RUNTIME_R1` §2-0 A0-1 ②). **always-on** 이다 — 구 판은 증감이
    //   IO3 방출 함수 안에 있어 `MOE_DIRECT_IO3=off` 런에서 0 으로 고정됐고, 그 상태로는
    //   admission credit 식이 상시 참이 된다(계측 스위치가 스케줄러를 바꾸는 결함).
    //   ⇒ 증감은 제출/회수 **경로**가 하고 IO3 는 그 값을 snapshot 만 한다.
    int32_t v3_inflight_demand = 0, v3_inflight_prefetch = 0;
    // ── ★★PB virtual A1 — ①record 운영축 + ②축 예약분(전부 mode-neutral·always-on) ──
    //   `pbv_rec_active` = A0-5 의 **전역 gauge** `rec_admitted_active_global`
    //     (= reserved + inflight prefetch record 수 · bin N 의 전역 의미 보존).
    //   `pbv_child_reserved` = admission 이 선점하고 아직 제출로 전환되지 않은 child credit 합
    //     (= A0-4 credit 식의 `child_reserved_prefetch`).
    int32_t  pbv_rec_active     = 0;
    int32_t  pbv_rec_inflight   = 0;
    int32_t  pbv_child_reserved = 0;
    std::vector<PbvRec> pbv_rec;            // 크기 n_slots · 인덱스 = pending_ops 인덱스
    // ★A2 진단 카운터 2종 — **C3 배타 closure 밖**(§2-0 A0-4 · PHASEB_DESIGN §1 C3 부기).
    //   `deferred_credit_attempts` = retry-attempt 계수(같은 intent 중복 계수 허용) ·
    //   `skipped_drain_turns`      = 실제 drain turn 당 1회(동일 상태 hot-loop 는 세지 않는다).
    uint64_t pb_admit_deferred_credit_attempts = 0;
    uint64_t pb_admit_skipped_drain_turns      = 0;
    // ★A2 drain-defer park 래치(A0-4 wake/turn 결과 계약 ②) — 서 있는 동안 "변화 없는 queued
    //   prefetch intent" 는 command_event 를 재무장하지 못한다. child terminal·credit 반환·
    //   신규 command 가 내린다(③ "child terminal/credit 변화에는 재시도").
    bool     pbv_admit_parked   = false;
    // ★★R-3(교차 r1 F5) — park 는 **무엇 때문에 멈췄는지**를 함께 들고 있는다. A0-4 ②의 문면은
    //   "**변화 없는** queued prefetch intent"이므로, park 의 정확한 유효 조건은 "credit 이
    //   안 변함"이 아니라 "**admission 이 읽는 상태가 그대로임**"이다. 구 판은 그 사유를 안
    //   들고 있어서 사유가 사라져도(=command 소비로 predicate 가 풀려도) 래치가 남았고,
    //   credit 변화만이 유일한 해제 사건이라 in-flight 가 아예 없는 국면에서는 해제원 자체가
    //   존재하지 않았다 — 그게 50ms 영구 polling 의 실체다(F5).
    //   ⇒ 지문이 달라지면 그건 새 사건이므로 park 를 내리고 재시도한다. 지문이 같으면 재진입은
    //     무의미하므로 계속 막는다(A0-4 ④ "실제 drain turn 당 1회" 보존).
    uint64_t pbv_park_fp        = 0;
    // ★★S-1(교차 r2 F-1) — **credit 예약/반납 누적 계수**. gauge(`pbv_child_reserved`)는
    //   dispatcher 정지 경로가 **무조건 0 으로 밀어** 그걸로는 "반환이 실제로 돌았는가"를
    //   쟴 수 없다(반환을 통째로 지운 역변이 M-H 가 전 관문을 통과한 이유). 그래서
    //   **사건 자체를 누적**한다 — teardown 은 이 두 값을 건드리지 않고
    //   `reset_for_test` 만 지운다. 완전 drain 후 **reserved_total == unreserved_total** 가
    //   A0-3 "현재 실제 보유한 reserved 만 반환·과반환 금지"의 관측형이다.
    uint64_t pbv_credit_reserved_total   = 0;   // admission 에서 선점한 W 누적
    uint64_t pbv_credit_unreserved_total = 0;   // 예약이 해소된 누적(제출 전환 + terminal 반환)
    // ★★R-3 — park 사유는 **두 종류**이고 지문도 사유별로 갈라야 한다(그러지 않으면 V3P12 가
    //   문다): ①진입 게이트(predicate·continuation) ②credit 부족 defer.
    //   ★왜 갈라야 하나: ②로 선 park 는 credit 이 풀릴 때까지 유지돼야 하는데, 지문에
    //     predicate 를 섞으면 **무관한 TEST_QUERY 왕복**이 predicate 를 true→false 로 흔들어
    //     park 가 풀렸다 다시 서고, 그때마다 `skipped_drain_turns` 가 올라 A0-4 ④("실제 drain
    //     turn 당 1회")가 깨진다. 실제로 그렇게 만들었다가 V3P12 가 drain 1→4 로 잡아냈다.
    //   ⇒ 각 park 는 **자기 사유가 읽는 축만** 지문에 담는다.
    uint8_t  pbv_park_reason    = 0;   // 0=미park · 1=게이트(predicate/cont) · 2=credit defer
    // ★★R-1 — 직전 pump pass 에서 continuation wave 재검사가 1건이라도 실패했는가.
    //   A0-4 drain-before-refill 의 입력이다(pump 가 쓰고 admission 이 읽는다 — 같은 turn).
    bool     pbv_cont_deferred  = false;
    // ★★S-4(교차 r2 F-4) — **continuation credit 재검사 실패 사건** 누적. A0-4
    //   drain-before-refill 이 **실발동**했는지를 재는 유일한 창구다(구 트리는 bool 만 있어
    //   V3P14b 가 `defer=0`으로 성공 경로만 만들고도 녹색이었다).
    uint64_t pbv_cont_defer_events = 0;
    // ★★T-3(교차 r3 S-4) — **drain-before-refill 이 실제로 신규 admission 을 생략한 횟수**.
    //   `pbv_cont_defer_events`(=재검사 실패 사건)와 **다른 축**이다: 그건 pump 가 세고,
    //   이건 admission 게이트가 센다. r3 지적의 핵심이 정확히 이 구분이었다 — `cont` 를
    //   무시하도록 되돌려도 실패 사건 계수는 그대로 오르고 record 는 결국 완주하므로,
    //   **결과 계약(생략했는가)** 을 재는 축이 따로 없으면 역변이 감도가 0 이다.
    uint64_t pbv_admit_skipped_cont = 0;
    // ★★U-3(교차 r4 · 대기 고정) — 생략이 일어난 **그 turn 의 증인**. 계수만으로는 "생략했다"
    //   까지고, r4 가 추가로 요구한 것은 "**막을 대상이 그 순간 실제로 대기 중이었다**"의 고정
    //   이다. 그걸 테스트 스레드에서 재려던 시도는 구조적으로 늦는다 — 하네스의 turn 유발
    //   창구(`test_lease_stats` 왕복)는 dispatcher 를 **깨울 뿐**이라, 테스트가 사건 계수를
    //   읽는 시점엔 이미 record 가 완주하고 대기 intent 도 소비된 뒤다(실측: 사건 시점
    //   스냅샷조차 `rec_active=0 intent_depth=0`). ⇒ 증인을 **엔진 안 생략 지점**에 둔다.
    //   ★기록 정책 = **공허한 생략은 덮어쓰고, 대상이 실재한 생략에서 동결**(갱신 조건은 소비
    //     지점 주석). 생략 자체는 계수가 세고, 이 증인은 "그 생략이 공허하지 않았다"를 맡는다.
    //   ★역변이 감도: 이 두 줄도 `if (blocks || cont)` **분기 안**에 있으므로, 실제 차단을
    //     되돌리면 도달하지 않아 증인이 sentinel 로 남는다(계수와 함께 죽는다).
    //   ★"막을 대상"은 **두 자리**에 설 수 있다 — 엔진은 intent 를 pop 한 뒤 rank 를 **turn 당
    //     1건**씩 소비하므로(`pb_cur_active`), 생략된 신규 admission 은 ①진행 중 intent 의 남은
    //     rank 이거나 ②큐에서 대기 중인 다음 intent 다. 둘 중 하나라도 서 있으면 그 생략은
    //     공허하지 않다. 두 축을 따로 남겨 어느 자리였는지까지 진단에 찍는다.
    //   ★★V(교차 r5) — **"대상이 있다"만으론 부족하다**. r5 가 실측으로 적발한 완화 2건:
    //     ①구 증인의 `rank_left` 는 **raw 잔여**라 `it.ids[r] == NA` 칸까지 셌다
    //      (`test_push_intent` 가 `k = pf_k` 로 두고 남는 칸을 NA 로 채운다) — NA rank 는 차단을
    //       없애도 `PRE_STALE` **처분**이지 admission 이 아니다.
    //     ②`N_cap` 이 포화(`rec_active + 1 > ncap`)면 유효 ID 라도 **cap-drop** 이지 admit 이 아니다.
    //     ⇒ 증인이 재는 것을 "잔여"에서 **"실 후보"** 로 올린다: `cont` 가 없었다면 그 자리에서
    //       ⓐ**유효 ID**(NA 아님 · `< n_expert`)가 있고 ⓑ**N_cap 여유**가 있고 ⓒ**credit 통과**
    //       였는가. 셋이 동시에 서야 "차단 제거 → 실제 admit 발생"이 성립한다.
    int32_t  pbv_admit_skip_rec_active   = -1;
    int32_t  pbv_admit_skip_intent_depth = -1;
    int32_t  pbv_admit_skip_valid_left   = -1;  // ⓐ 남은 rank 중 **유효 ID** 개수(NA 제외)
    int32_t  pbv_admit_skip_ncap_room    = -1;  // ⓑ `ncap - rec_active`(≥1 이어야 admit 여지)
    int32_t  pbv_admit_skip_credit_ok    = -1;  // ⓒ 그 시점 `pbv_credit_ok_locked(w)`
    //   ⓓ 생략 **그 순간**의 C3 `admitted` 값. 주입은 1회성이라 다음 turn 부터 `cont` 가 내려가므로,
    //     막힌 후보가 실 후보였다면 이 값보다 **뒤에 실제 admission 이 일어나야 한다**. 이 기준선을
    //     테스트가 나중에 읽으면 이미 재개가 끝난 뒤라 늦는다(실측 `resumed=0`) ⇒ 엔진이 박는다.
    uint64_t pbv_admit_skip_admitted     = 0;
    // ★★S-4 — continuation credit 재검사 **1회성 실패 주입**(테스트 전용 · 프로덕션 0).
    //   ★왜 필요한가(실측 근거): 노브만으로는 이 분기가 **도달 불가**다.
    //     ·part cap(`remaining + n_need > iop`)이 credit 재검사 **앞**에 서서, 앞 wave 가
    //      in-flight 인 동안엔 항상 그가 먼저 막는다(W=min(P,iop) 라 wave1 이 part cap 을
    //      정확히 포화시키기 때문 — "wave 순차 진행"의 직접 귀결).
    //     ·앞 wave 가 회수된 뒤라면 조건 ②는 N_cap 에 함의되어(§G-3 ①) 절대 안 깨진다.
    //     ·남는 유일한 실패원은 조건 ①(demand 가 free_total 을 먹는 경우)인데, 그건
    //      "wave1 회수 직후 · wave2 제출 직전"의 한 턴 창을 잡아야 해 하네스로 결정론적
    //      재현이 안 된다. ⇒ 주입으로 **결과 계약**(drain-before-refill)을 재는 쪽을 택한다.
    int32_t  test_cont_credit_fail_next = 0;
    // ★★T-1(교차 r3 S-2 ②) — 유계 종료가 **timeout 으로 끝난** 사실의 래치.
    //   이게 서 있으면 이후 **프로덕션 정지 래퍼도 재진입하지 않는다** — 안 그러면 관문이
    //   FAIL 을 기록한 직후 `reset_for_test` 가 다시 정지 대기로 들어가 **"FAIL 기록 후 교착"**
    //   이 된다(종단 유계가 깨지는 두 번째 길).
    //   ★프로덕션에선 이 래치가 서지 않는다(유계 훅은 테스트 전용이다).
    //   E-2b-0 durable (g): the two phrases corrected just above both asserted an UNBOUNDED
    //   production stop. It has been finite since E-2a (MOE_DRAIN_DEADLINE_MS, contract
    //   section 2 row 7), so what this latch blocks re-entry into is a bounded stop.
    bool     disp_stop_timed_out = false;
    // ★R12 — §2 재결속: virtual 의 `read_immediate_count`/`read_pending_count` 는 **demand-origin
    //   물리 child** 계수(children flags b0 기준)다. IO3 가 꺼진 런에서도 서야 하는 축이라
    //   사이드카가 아니라 엔진 카운터로 센다(IO3 는 같은 사실을 행에 담을 뿐이다).
    uint64_t v3_child_pending_demand = 0, v3_child_immediate_demand = 0;
    // ★R12 — §4 mode-local: virtual 의 `moe_readfile_ops/digest` 축(demand-origin child 제출).
    //   인코딩=18B/child `(u16 layer)(u16 expert)(u8 part)(u8 source_index)(u64 read_offset)
    //   (u32 read_len)` FNV-1a 64. 소비(metrics 방출)는 R13.
    uint64_t v3_readfile_ops    = 0;
    uint64_t v3_readfile_digest = 0xcbf29ce484222325ull;
    // ★R12 §3-1 — resolved IO 형상 파라미터 6종(값 + source enum 0=default 1=manifest
    //   2=dedicated_env 3=legacy_alias). virtual seal 이 채우고 IO3 헤더 off 104..115 와
    //   metrics header 가 **같은 배열**을 echo 한다(사본 축소 — 값 출처가 하나다).
    //   순서=`io_qd_total`·`io_qd_demand_reserve`·`io_qd_prefetch_child_max`·
    //   `io_part_parallelism`·`io_bounce_copy_workers`·`io_bounce_copy_qd`.
    int32_t v3_io_param_val[6] = { 0, 0, 0, 0, 0, 0 };
    uint8_t v3_io_param_src[6] = { 0, 0, 0, 0, 0, 0 };
    int32_t  v3_test_fail_copy = -1;       // ★테스트 전용 one-shot: 그 part 의 copy 를 실패로

    // ===================================================================
    // ★§Z-⑥ A6 — 프리필 coalescing(§3-6 7조건 · §5 파라미터 · §5-4/§5-5 wire 축)
    // ===================================================================
    uint8_t  v3_coal_phase     = V3_COAL_OFF;   // off | prefill | all
    uint64_t v3_coal_min_bytes = 0;             // §5 io_coalesce_min(합칠 값이 없으면 안 합친다)
    uint64_t v3_coal_max_bytes = 0;             // §5 io_coalesce_max(대구간 1건 상한)
    int32_t  v3_coal_buffers   = 0;             // §5 io_coalesce_buffers = 동시 대구간 **그룹** 수
    void *   v3_coal_base      = nullptr;       // VirtualAlloc(NO_BUFFERING 착지 — 섹터 정렬)
    size_t   v3_coal_stride    = 0;             // 버퍼 1칸 크기
    int32_t  v3_coal_count     = 0;             // = buffers × max_parts(그룹 1개가 part 수만큼 쓴다)
    std::vector<V3CoalRead> v3_coal;            // 버퍼 i 를 v3_coal[i] 가 소유(1:1)
    std::vector<int32_t>    v3_chan_coal;       // 채널 → 대구간 인덱스(-1 = coalesced 아님)
    uint32_t v3_coal_group_seq = 0;             // `coalesce_group` 발급(0xFFFFFFFF=NA 는 건너뛴다)
    uint64_t v3_coal_groups    = 0;             // 성립한 대구간 수(= coalesced child 수)
    uint64_t v3_coal_records   = 0;             // 대구간에 참가한 record 수(누계)
    uint64_t v3_coal_scatters  = 0;             // 완료된 scatter copy 수
    uint64_t v3_coal_reads_retired = 0;         // ★S3 — **회수 완료된** 대구간 read 수(⑦ 관문의 관측점)
    uint8_t  v3_coal_last_reject = V3_COAL_R_NONE;  // 직전 계획 판정(테스트 관측 — 위 enum)
    // ★한 acquire 는 pump 를 여러 번 돌리므로 "직전 값"만으로는 **어느 조건이 막았는지**를
    //   주장할 수 없다(마지막 패스는 후보가 1건뿐이라 늘 R_RUN 이 된다). 본 사유는 비트로 누적한다.
    uint32_t v3_coal_reject_mask = 0;
    V3CoalPlan v3_coal_plan;                    // 계획 스크래치(seal 에서 reserve — I/O 경로 무할당)
    // ★테스트 전용 **sticky 주입 훅**(프로덕션 0) — arm 하면 서고, 써도 내려가지 않으며,
    //   `arm_coalesce(0,0)`·`reset_for_test()` 만이 해제한다. 아래 둘은 **실 런타임에서 도달
    //   불가한 가드**를 결정론적으로 행사하는 유일한 창구다(V3f10 copy 실패 주입과 같은
    //   사유·같은 규율). ★한 번 쓰고 말 훅이었다면 한 acquire 안의 두 번째 계획이
    //   합쳐버려 "이 가드가 막았다"는 판정이 서지 않는다.
    int32_t  v3_test_coal_overrun = 0;     // armed: run 을 +1 expert 로 부풀려 ③ 경계 가드 행사
    int32_t  v3_test_coal_unpin   = 0;     // armed: head 를 미pin 으로 취급해 ⑤ 가드 행사

    // ===================================================================
    // ★§Z-⑥ A5 — IO3 사이드카 상태(SPEC_IO_METRICS_V3 §5). 기본 OFF 라 미설정 런은
    //   자료구조도 안 만든다(bin·virtual 양 모드 대칭 — A/B 오버헤드 동일).
    // ===================================================================
    bool     io3_on       = false;
    // ★R13 — `io3_on` 은 flush 에서 false 로 내려가므로(재진입 금지 래치) summary 시점에는
    //   "이 런이 IO3 를 요청했는가"를 알 수 없다. 그 사실을 따로 보존한다(`io3_sidecar` 보고 조건).
    bool     io3_requested = false;
    bool     io3_complete = true;      // 버퍼 초과·OUT 미설정 등에서 0 으로 내린다
    std::string io3_out;               // 경로 접두사(빈 문자열 = 미설정 → complete=0)
    uint64_t io3_buf_cap_bytes = 0;
    uint64_t io3_boot_id = 0, io3_t0_qpc = 0;
    uint64_t io3_event_seq  = 0;       // ★children submit/terminal 공용 전역 카운터(0-based)
    uint64_t io3_record_seq = 0;       // records 행 발급 카운터
    uint64_t io3_publish_seq = 0;      // publish 순번(1-based)
    uint64_t io3_dropped = 0;
    std::vector<Io3Record> io3_records;
    std::vector<Io3Child>  io3_children;
    std::vector<Io3Copy>   io3_copies;
    // ★R13 — flush 결과(§2 summary `io3_sidecar` 보고처). kind 1/2/3 순서.
    uint64_t io3_stat_records[3] = { 0, 0, 0 };
    int32_t  io3_stat_written[3] = { 0, 0, 0 };
    uint64_t io3_stat_digest[3]  = { 0, 0, 0 };
    uint64_t io3_stat_buffer_bytes = 0;
    // op 인덱스 → 그 op 이 소유한 records 행 인덱스(-1=없음). 병렬 표(동결 구조 불가침).
    std::vector<int32_t>   io3_op_row;
    // 채널 인덱스 → 그 child 의 children 행 인덱스(-1). 〃
    std::vector<int32_t>   io3_chan_row;
    // ★A6 — 채널 인덱스 → 그 바운스 child 의 copies 행 인덱스(-1). 구 판은 submit event seq 로
    //   역탐색했으나 그 키는 coalesced 에서 1:N 이라 단일 행을 지목하지 못한다(§5-5).
    std::vector<int32_t>   io3_chan_copy_row;
    // ★A3 테스트 전용 one-shot(프로덕션 0 · reset 에서 0 복귀 — 기존 test_* 훅과 같은 규율).
    //   동시성 경로는 정적 검토로 닫지 않는다(계획 §4-2): 아래 훅이 §6-4 ⑩ 이 요구하는 경로를
    //   **결정론적으로 발화**시키는 창구다. 실 저장장치에서는 완료 순서·부분 read 를 재현할 수
    //   없기 때문에 존재한다(bin 의 test_force_sync_next·test_pb_error_next 와 같은 사유).
    int32_t v3_test_reverse_children = 0;   // armed: 다음 op 의 part→채널 배정을 역순으로
    int32_t v3_test_fail_child       = -1;  // armed(>=0): 다음 op 의 그 part child 를 실패로 강등
    int32_t v3_test_fail_after_acquire = 0; // armed: seal 을 acquire 직후 실패(반납 총체성 회귀)
    int32_t v3_test_fail_dispatcher_start = 0; // ★R18 armed: dispatcher 기동을 실패시킨다
                                               //   (metrics 아티팩트 정리 경로 회귀)
    // ★재개봉 R3 — armed(>=0): 다음 op 의 그 part child 만 **동기 완료 경로**로 통과시킨다.
    //   NVMe+NO_BUFFERING 에서 ReadFile 즉시 TRUE 는 관측되지 않아 혼합 경로가 자연 발화하지
    //   않는다(bin 의 `test_force_sync_next` 와 같은 사유·같은 방식: 이벤트를 그 자리에서 기다려
    //   동기 분기로 밀어넣는다). 프로덕션 0.
    int32_t v3_test_sync_child       = -1;
    // ★A3 관측 계수(테스트 판정용 — 프로덕션 판단에 쓰지 않는다).
    uint64_t v3_children_submitted = 0;
    // ★★R-1(FROZEN §2-0 A0-2) — **wave 발행 수**. child 수만으로는 "한 번에 P 개"와 "W 개씩
    //   여러 번"이 구분되지 않는다(둘 다 child 총수는 같다). 이 계수가 wave 기계가 실제로
    //   돌았다는 **유일한 직접 증거**다: 같은 형상에서 `io_part_parallelism` 만 낮추면
    //   `waves = ceil(P/W) × records` 로 늘어야 한다.
    uint64_t v3_waves_submitted    = 0;
    uint64_t v3_children_retired   = 0;
    uint64_t v3_op_terminals       = 0;
    uint64_t v3_publishes          = 0;
    // ★재개봉 R3 — 혼합 완료 실증용. sync = 제출 자리에서 닫힌 child · async = 이벤트 회수 child.
    uint64_t v3_children_sync      = 0;
    uint64_t v3_children_async     = 0;
    // ★재개봉 R3 — **직전 op 의 child 회수 순서**(part 인덱스 열). out-of-order 를 "일어났다고
    //   믿는" 대신 실제 열을 기록해 단언한다(OS 완료 순서에 기대는 통과 금지 — 교차 X7 지적).
    std::vector<int32_t> v3_retire_order;
    DemandPlan             disp_plan;         // 8단계 플래너 스크래치(정적 → 멤버)
    int32_t                demand_outstanding = 0;

    // control_q — bounded MPSC 64(demand/release 절대 drop 금지·만석 시 생산자 대기[g_mod.mutex 무보유])
    std::mutex              q_mutex;
    std::condition_variable q_free_cv;
    std::condition_variable q_pause_cv;
    ControlCmd cmds[MOE_DIRECT_CONTROL_Q_CAP];
    int32_t    q_free[MOE_DIRECT_CONTROL_Q_CAP];
    int32_t    q_ring[MOE_DIRECT_CONTROL_Q_CAP];
    int32_t    q_free_top = 0;
    int32_t    q_head = 0, q_count = 0;
    uint64_t   q_seq_next = 1;
    bool       q_stopping = false;   // STOPPING — 신규 command 거절
    bool       q_abort    = false;   // abort/test 종료(전 waiter를 명시 error로 깨움)
    bool       q_pause_req = false, q_paused = false; // 테스트 전용(queue 만석 픽스처)

    // B1 계측(§7 착지 관문 — dispatcher_demand_cmd>0 활동 증명 + 큐 무손실 증명)
    uint64_t stat_demand_cmd = 0, stat_release_cmd = 0, stat_query_cmd = 0, stat_stop_cmd = 0;
    // ★q_mutex 아래에서 갱신되고 metrics/test 조회는 g_mod.mutex 아래라 교차 접근 — atomic 필수.
    std::atomic<uint64_t> stat_q_full_waits{0};
    std::atomic<uint64_t> stat_q_max_depth{0};
    // ★이벤트 대기에서 복귀한 횟수(락 밖에서 갱신) — idle 구간에 증가하면 hot-loop(공회전)다.
    std::atomic<uint64_t> stat_wait_wakeups{0};
    uint64_t stat_join = 0, stat_cancel = 0, stat_abort_terminal = 0;
    // 테스트 전용 결함 주입: 제출 op가 n건에 도달하면 dispatcher가 그 자리에서 abort 종료 시퀀스를
    // 실행한다(cancel→terminal drain 경로를 결정론적으로 행사하기 위한 훅 — 프로덕션 0).
    int32_t  test_abort_after_submit = 0;
    // 테스트 전용 one-shot: armed면 다음 pending 제출 1건의 완료를 그 자리에서 대기해 sync_done
    // 분기(즉시 완료 경로)를 그대로 통과시킨다. 행사 후 자동 disarm. 프로덕션 0.
    int32_t  test_force_sync_next = 0;
    // sync_done 분기(즉시 완료 · 강제 포함)를 통과한 terminal 수 — 회귀 픽스처의 행사 증명.
    uint64_t stat_sync_terminal = 0;
    // 테스트 전용 one-shot 결함 주입: armed면 다음 prefetch 성공 terminal 1건을 부분 read/오류
    // terminal(§5)로 강등해 fail-soft unpublish·joined retry 경로를 결정론적으로 행사한다.
    // ★프로덕션 0(reset에서 0 복귀) — 실 저장장치에서 부분 read를 재현할 수 없기 때문의 창구다.
    int32_t  test_pb_error_next = 0;
    // ★보완 ② — 테스트 전용 one-shot barrier. dispatcher 가 "stopping 스냅샷 직후" 지점에서
    //   1회 정지해, 테스트가 **그 창 안으로** stop 을 삽입할 수 있게 한다(수리 1의 결정적 회귀).
    //   프로덕션 0(reset 에서 0 복귀) — 기존 test_force_sync_next/test_abort_after_submit 선례와 동형.
    int32_t  test_stop_race = 0;
    HANDLE   stop_race_hit = nullptr;   // dispatcher 가 창에 도달했음(manual reset)
    HANDLE   stop_race_go  = nullptr;   // 테스트가 재개를 허가(manual reset)
    // 테스트 전용: armed면 signaled 채널 회수를 보류한다(op을 in-flight 상태로 고정) — join·
    // pending·취소 픽스처가 "완료 전 창"을 결정론적으로 잡기 위한 창구. 프로덕션 0.
    // ★★R-5 V3P17 — **취소 선별 관측 창구**(§3 ⑮ · A0-4 joined 보호). 구 트리에는 "무엇이
    //   취소됐는가"를 보는 창이 없어, joined 보호가 성립했는지 **간접 추론**밖에 못 했다
    //   (demand 가 성공했다 ⇒ 아마 취소 안 됐을 것). 선별의 두 결과를 직접 센다.
    uint64_t v3_cancel_selected      = 0;  // waiterless prefetch = 취소 대상으로 선별된 채널
    uint64_t v3_cancel_excl_joined   = 0;  // waiter(joined demand) 때문에 **제외**된 채널
    int32_t  test_hold_retire = 0;
    // ★★R-5(교차 r1 F3) — 테스트 전용 **제출 보류**. `test_hold_retire` 는 *회수*만 잡으므로
    //   admission~첫 submit 창(=record 가 `reserved>0` 인 유일한 구간)이 열리지 않는다. 그
    //   창이 없으면 A0-3 의 "미개시 예약분 반환" 갈래가 영영 미행사로 남는다(역변이 실증:
    //   그 반환을 지워도 V3P 32/32 가 통과했다). 프로덕션 0.
    int32_t  test_hold_submit = 0;
    // 테스트 전용: CUT 6 잔여 세부 비트(0x01..0x20) 주입 — echo·마스크 결속 등식 픽스처용.
    //   자연 발생 잔여는 결정론적 재현이 불가하다(FROZEN_INCOMPLETE_REASON_MASK §10 ⓕ). 프로덕션 0.
    int32_t  test_cut_residual_force = 0;

    // =======================================================================
    // ★B2 프리페치(PHASEB_DESIGN §4·§5 · SPEC_PHASEB_WIRE §4~§6) — dispatcher 결속분.
    //   자원(capture_q·intent_q·worker·snapshot)은 전부 ggml-moe-phaseb.cpp 소유.
    // =======================================================================
    bool     pb_prefetch_on = false;               // seal에서 해소된 K/N 유효성(런 중 불변)
    int32_t  pb_k = 0, pb_n = 0;
    int32_t  pb_inflight_demand = 0, pb_inflight_prefetch = 0;   // in-flight = submitted ∧ non-terminal
    uint64_t pb_io_event_seq = 0;                  // ★공용 io event namespace(§5.1 · dense)
    uint64_t pb_publish_seq  = 0;                  // 전역 단조 publish 순번(1-based)
    uint32_t pb_cur_graph_seq = GGML_MOE_PHASEB_NA_U32;   // dispatcher가 마지막으로 처리한 demand의 graph
    // intent 소비 커서 — 헤드 intent 1건의 rank를 턴당 1 후보씩 소비한다(§4 "턴당 1건").
    bool     pb_cur_active = false;
    int32_t  pb_cur_rank = 0, pb_cur_row = -1;
    ggml_moe_phaseb_intent pb_cur_intent{};
    // ★"이 (graph, layer)의 canonical demand가 치러졌는가" 표 — too_late 사전 처분(§4.2)과
    //   canonical 판정의 **모드 무관** 권위다(pb_sample 행은 PB off에서 버퍼링되지 않으므로
    //   프로덕션 기본에서 무한 성장 금지). ★R1 #8 — 8칸 링은 지연 intent를 too_late 대신 stale로
    //   왜곡했다. graph_seq가 프로세스 전역 단조이므로 **층별 "마지막 served graph_seq" 1개**로
    //   지연 무관·정확·고정 크기(n_moe_layers × u32)를 동시에 만족한다:
    //     too_late ⟺ served[mlpos] != NA ∧ intent.graph_seq <= served[mlpos]
    //   (같으면 그 graph의 target demand가 이미 치러졌고, 작으면 그 graph는 이미 지나갔다)
    std::vector<uint32_t> pb_served_graph;
    // pb_sample 행 표 — 키 (graph_seq, layer). ★PB 모드 != off 에서만 버퍼링된다.
    std::vector<ggml_moe_phaseb_sample>          pb_samples;
    std::unordered_map<uint64_t, int32_t>        pb_sample_index;
    // pb_io 레코드(성공 제출 물리 op 전건 — submit에서 생성·terminal에서 완성·CUT ⑧에서 flush)
    std::vector<ggml_moe_phaseb_io>              pb_ios;
    bool     pb_finalized = false;
    // ★R1 #3 ④ — CUT drain 후 전수 0 검사 결과(0=잔여 없음). 잔여면 complete=0 latch.
    //   ★R2 #1 — CUT ① 선형화 래치는 **PB 소유 원자 1개**(ggml_moe_phaseb_capture_closed)로
    //   통합됐다(여기 중복 래치를 두면 capture/command 가 서로 다른 순간에 닫힌다).
    int32_t  pb_cut_residual = 0;
    // =======================================================================
    // ★5c-T 수집 텔레메트리(SPEC_PHASEB_WIRE 부록 14) — **수집 ON 에서만** 동작한다.
    //   수집 OFF 에서는 아래 전부가 미할당/미갱신이라 종전 경로와 명령열이 같다.
    // =======================================================================
    bool     pb_collect = false;                   // seal 에서 해소된 수집 축(런 중 불변)
    std::vector<PbOpStamp> pb_op_stamp;            // pending_ops 병렬 side table(§1 전파)
    // canonical demand acquire 래치(command slot 인덱스 병렬 · 1 acquire 1계수 — 부록 14.1).
    uint8_t  pb_cmd_waited[MOE_DIRECT_CONTROL_Q_CAP]   = {0};
    uint8_t  pb_cmd_frontier[MOE_DIRECT_CONTROL_Q_CAP] = {0};
    // ★부록 14.6 — 수집 모드 rank 판정용 경량 예측열 스냅샷(유계 맵 · 사이드카 행 아님).
    //   키 = (graph_seq, target_mlpos) · 값 = 그 intent 의 ids[16] + k. blocked 계측점이 소비한다.
    struct PbPredSnap {
        uint64_t key = GGML_MOE_PHASEB_NA_U64;
        uint16_t ids[GGML_MOE_PHASEB_TOPK] = {0};
        uint8_t  k = 0;
    };
    static constexpr size_t PB_PRED_SNAP_MAX = 64; // 유계(FIFO 링 — 오래된 것부터 덮어쓴다)
    std::vector<PbPredSnap> pb_pred_snap;
    size_t   pb_pred_snap_next = 0;
    // ★부록 14.5 — censor cut(census) 1회 래치. 이 시점 이후 어떤 재계상도 금지된다.
    bool     pb_censused = false;
    // ★부록 14.3 — ledger flush 결과(summary 수집 전용 키 echo 용).
    uint64_t pb_ledger_records = 0, pb_ledger_digest = 0;
    int32_t  pb_ledger_written = 0;
    // ★R2 #5 — 아직 행이 없는 (graph, layer) 의 비동기 사유 보류함(행 생성 시 소비).
    //   ★누수 수리(R2 후속): 이 보류함은 "나중에 그 (graph, layer) demand 행이 생긴다"를 전제로
    //   소비된다. 그 demand 가 **영영 오지 않는 형상**(capture 만 도는 경로·중단된 graph)에서는
    //   소비처가 없어 무한 증가한다 — 상한을 두고 초과분은 버리되 관측 결손이므로 complete=0
    //   을 latch 한다(조용한 누락 금지).
    static const size_t PB_PENDING_STATUS_MAX = 4096;
    std::unordered_map<uint64_t, uint32_t> pb_pending_status;

    // 보호(PAGE_NOACCESS)+VEH
    std::vector<RegisteredTensorInfo> reg_tensor_info;
    std::vector<ProtInterval> prot_intervals;
    std::vector<std::atomic<uint8_t>> page_touched;
    std::vector<TouchLogEntry> touch_ring;
    std::atomic<uint64_t> touch_ring_next{0};
    void *   veh_handle = nullptr;
    uint32_t touch_abort_threshold = PROD_DEFAULT_TOUCH_ABORT;

    // 카운터
    std::atomic<uint64_t> touch_events{0};
    std::atomic<uint64_t> touch_pages{0};
    std::atomic<uint64_t> fallback_count{0};
    std::atomic<uint64_t> read_immediate_count{0};
    std::atomic<uint64_t> read_pending_count{0};

    // ★V1 무행동 증명(PREFETCH_DESIGN §3.5 · SPEC_PHASEA_WIRE §10): metrics가 켜져 있을 때만
    // 갱신되는 순서 digest. PA 모드와 무관 — pa_off arm(①)과 pa_timing arm(②③)의 paired 비교가
    // 성립하려면 양쪽 모두 산출해야 한다. miss 1건당 16B FNV(수 ns) 뿐이라 I/O(~100µs) 대비 무시
    // 가능하며 캐시/LRU/ReadFile 시퀀스에는 아무 영향이 없다.
    uint64_t readfile_digest = 0xcbf29ce484222325ull;
    uint64_t readfile_ops    = 0;
    // ★graph 형상 지문(Codex R1 필수 1) — PA와 무관하게 metrics 활성 시 매 graph 갱신.
    //   pa_off arm(①)에 pa_timing arm(②③)의 counts.graph_shape와 대조할 off측 비교값이 없으면
    //   PREFETCH_DESIGN §3.3 "PA=off와 동일함을 기록으로 증명"이 평가 불능이 된다.
    uint64_t graph_nodes_digest  = 0xcbf29ce484222325ull;
    uint64_t graph_splits_digest = 0xcbf29ce484222325ull;
    uint64_t graph_count         = 0;
    // ★PA 훅 게이트 — seal마다 env에서 재해소(call_once 캐시 금지: 테스트가 모드를 바꿔 재seal).
    bool     phasea_on = false;
    // ★Phase A op-level wire의 프로세스 전역 단조 순번(제출/회수는 별개 카운터 — QD>1의 본질).
    uint64_t pa_submit_seq_next = 0;
    uint64_t pa_retire_seq_next = 0;

    // 페이즈별 계측
    PhaseCounters phase[2];
    PhaseCounters phase_exact[2];
    PhaseCounters phase_exact_window[2];
    std::atomic<uint64_t> phase_ctx_unknown_count{0};

    MetricsWriter metrics;

    std::once_flag enabled_once;
    bool enabled_cached = false;
    std::once_flag no_prefetch_once;
    bool no_prefetch_cached = false;
};

extern ModuleState g_mod;

inline void resolve_part_addr(const PartInfo & p, uint64_t & out_offset, uint64_t & out_bytes) {
    if (g_mod.manifest_mode == ManifestMode::VIRTUAL) {
        out_offset = p.data_offset;
        out_bytes  = p.slice_bytes;
    } else {
        out_offset = p.part_offset;
        out_bytes  = p.part_bytes;
    }
}

inline bool v3_active() { return g_mod.manifest_mode == ManifestMode::VIRTUAL; }

inline int32_t v3_parts_of(int32_t mlpos) {
    return (int32_t)g_mod.layers[(size_t)mlpos].parts.size();
}

enum V3Stage : int32_t {
    V3_STAGE_NONE          = 0,   // v3 경로 미진입
    V3_STAGE_MODE_DETECTED = 1,   // schema 3.0 + mode virtual 확정
    V3_STAGE_INIT_COMBOS   = 2,   // IO_METRICS §3 거부 4조합 통과
    V3_STAGE_MANIFEST      = 3,   // §2-2·§2-4 manifest 검증 통과
    V3_STAGE_REFLOCK       = 4,   // reference_lock F1 결속 + bin 부재 통과
    V3_STAGE_PLAN_REPORT   = 5,   // §4-3 plan_report 소비 결속 통과
    V3_STAGE_ALIGN         = 6,   // A_runtime 사다리(전 source 볼륨) 통과
    V3_STAGE_REGISTRY      = 7,   // 등록 3집합(M_all == L_scope) 통과
    V3_STAGE_LIVE          = 8,   // §2-5 live 재파싱 + DF-1 digest 통과 = **A1 완주**
    // ---- ★A2 가 경계를 앞으로 민 구간(자원 확보). 여기부터는 "검증"이 아니라 "확보"라
    //      각 래치가 되돌릴 자원을 하나씩 늘린다 — A2 경계 teardown 이 그 역순으로 푼다. ----
    V3_STAGE_HANDLES       = 9,   // §3-1 N-shard 소스 핸들 집합 확보 + manifest 대조
    V3_STAGE_SLOTPOOL      = 10,  // 슬롯 풀 확보 + 레이아웃 확정(§4-7 n_slots >= n_expert)
    V3_STAGE_BIND          = 11,  // registry·attestation·보호 확정 = **A2 완주**
    // ---- ★A3 가 경계를 앞으로 민 구간. 여기서부터 seal 이 실제로 성립한다. ----
    V3_STAGE_SEALED        = 12,  // QD 불변식·VEH·dispatcher 기동 = **A3 완주(seal 성공)**
};

extern std::atomic<int32_t> g_v3_stage;

struct V3ResourceProbe {
    bool     valid = false;
    int32_t  n_handles = 0;
    int32_t  n_slots = 0;
    int32_t  att_managed = 0;
    int32_t  att_host = 0;
    int32_t  n_prot_intervals = 0;
    uint64_t pool_bytes = 0;
    uint64_t slot_stride_max = 0;
    uint64_t handle_bytes_total = 0;   // Σ GetFileSizeEx(확보한 핸들)
    // (mlpos ↑, parts 배열순) 으로 flatten 한 주소 계약. **resolve_part_addr 를 그대로 통과한
    // 값**이라 `part_info()` 가 virtual 에서 무엇을 반환하는지를 이 스냅샷이 대변한다.
    std::vector<std::pair<uint64_t, uint64_t>> part_addr;
};

extern V3ResourceProbe g_v3_probe;
} // namespace ggml_moe_direct_impl

#endif // _WIN32
