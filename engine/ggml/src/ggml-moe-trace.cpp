// G3-2c 겹침 — 계측 인프라(Phase 1) 구현. 사양: bench/moe-direct/SPEC_G32C.md v4.
// ggml-moe-direct.cpp(모듈 로직)와 의도적으로 분리된 별도 파일이다 — 사양 §7의 sensor_id를
// "이 파일들의 SHA"로 정의하기 위함(서문·헤더 주석 참조). 다른 TU의 익명 네임스페이스
// 심볼에 의존하지 않도록 필요한 소소한 유틸리티(env 파싱·UTF-8 변환)는 이 파일 안에서
// 독립적으로 재구현했다(의도적 중복 — 결합도를 낮추기 위함).

#include "ggml-moe-trace.h"
#include "ggml.h" // GGML_ABORT

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// 배선 4d(사양 §6-5 read_bytes delta_pre) — ggml_moe_direct_phase_read_bytes_hit_miss_exact()
// 조회용(R3 배선 — 외부 심의 OV5 차단 3 전까지는 부정확 카운터를 썼다). ★sensor_id 결합도(§7): 이 include 자체가 trace.cpp 파일 텍스트를 늘려 sensor_id에
// 영향을 주는 것은 불가피하지만(run_begin/run_end에 새 로직이 들어가므로 어차피 파일이
// 바뀐다), 실제 PhaseCounters 구현은 ggml-moe-direct.cpp/.h에 남겨 그쪽이 바뀌어도 이
// 파일(=sensor_id 재료)은 흔들리지 않게 했다(헤더 서문 "sensor_id를 이 두 파일의 SHA로
// 정의" 취지 유지 — 최종 보고서 "read_bytes 델타의 조회 경로" 항 참조).
#include "ggml-moe-direct.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// ===========================================================================
// §1. 상수
// ===========================================================================

constexpr uint64_t RING_BYTES             = 256ULL * 1024ULL * 1024ULL; // 배선 R1(사양 §1) — 64→256 MiB
constexpr uint32_t RECORD_SIZE            = 64; // 전 레코드 종류 공통 슬롯 크기
constexpr uint64_t RING_CAPACITY_RECORDS  = RING_BYTES / RECORD_SIZE;  // 4,194,304 = 2^22
constexpr uint64_t RING_MASK              = RING_CAPACITY_RECORDS - 1; // 2의 거듭제곱이므로 & 마스크 가능
static_assert((RING_CAPACITY_RECORDS & (RING_CAPACITY_RECORDS - 1)) == 0,
              "RING_CAPACITY_RECORDS must be a power of 2 for mask-based indexing");

// 배선 R1(사양 §2) — prefill 창 이중 버퍼(링과 별도, decode 표본 창처럼 index 산정이
// 불필요한 "마지막 prefill graph만" 보존하는 용도). 버퍼당 32 MiB, 2개(현재 기록 중 1개 +
// 직전 완료분 1개 — §2 "이중 버퍼인 이유" 참조).
constexpr uint64_t PREFILL_WIN_BYTES             = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t PREFILL_WIN_CAPACITY_RECORDS  = PREFILL_WIN_BYTES / RECORD_SIZE; // 524,288

// 배선 R1(사양 §3) — decode 표본 창(0-based decode graph 인덱스, run_begin에서 리셋).
constexpr uint32_t DECODE_WIN_BEGIN = 128;
constexpr uint32_t DECODE_WIN_END   = 192;

// ===========================================================================
// §2. 레코드 wire format — 실제 정의는 ggml-moe-trace.h(단일 소스 — 판독기·selftest도
// 그 정의를 그대로 쓴다). 여기서는 짧은 별칭 + 64B 정합성 static_assert만 둔다.
// ===========================================================================
using FixedRecord        = ggml_moe_trace_fixed_record;
using DemandEntry        = ggml_moe_trace_demand_entry;
using DemandHdrRecord    = ggml_moe_trace_demand_hdr_record;
using DemandContRecord   = ggml_moe_trace_demand_cont_record;
using SnapshotHdrRecord  = ggml_moe_trace_snapshot_hdr_record;
using SnapshotContRecord = ggml_moe_trace_snapshot_cont_record;
using WallStateAggRecord = ggml_moe_trace_wallstate_agg_record;
using IostateAggRecord   = ggml_moe_trace_iostate_agg_record; // 배선 4d
using TraceFileHeader    = ggml_moe_trace_file_header;

static_assert(sizeof(FixedRecord) == 64, "FixedRecord must be exactly 64 bytes (SPEC section 5-1)");
static_assert(offsetof(FixedRecord, qpc_tick)        == 8,  "qpc_tick offset drift");
static_assert(offsetof(FixedRecord, event_seq)       == 16, "event_seq offset drift");
static_assert(offsetof(FixedRecord, correlation_id)  == 24, "correlation_id offset drift");
// 배선 S6(사양 §5 확인5) — _pad[6]에 anchor delta를 싣기 시작하지만 필드 자체를 늘리지
// 않았으므로(위 struct 주석 참조), 오프셋·전체 크기 불변이 이 두 static_assert로 고정된다.
static_assert(offsetof(FixedRecord, _pad) == 58, "_pad offset drift - S6 anchor delta encoding assumes this");
static_assert(sizeof(FixedRecord) == 64, "FixedRecord must remain 64 bytes after S6 _pad reuse");
static_assert(sizeof(DemandHdrRecord) == 64,    "DemandHdrRecord must be exactly 64 bytes");
static_assert(sizeof(DemandContRecord) == 64,   "DemandContRecord must be exactly 64 bytes");
static_assert(MOE_TRACE_DEMAND_K == 17,         "DEMAND_K drifted - update capacity/report if intentional");
static_assert(sizeof(SnapshotHdrRecord) == 64,  "SnapshotHdrRecord must be exactly 64 bytes");
static_assert(sizeof(SnapshotContRecord) == 64, "SnapshotContRecord must be exactly 64 bytes");
static_assert(MOE_TRACE_SNAPSHOT_K == 6,        "SNAPSHOT_K drifted - update capacity/report if intentional");
static_assert(sizeof(WallStateAggRecord) == 64, "WallStateAggRecord must be exactly 64 bytes (wiring 4a)");
static_assert(sizeof(IostateAggRecord) == 64,   "IostateAggRecord must be exactly 64 bytes (wiring 4d)");
static_assert(sizeof(TraceFileHeader) == 64,    "TraceFileHeader must be exactly 64 bytes");

constexpr uint32_t DEMAND_K   = MOE_TRACE_DEMAND_K;
constexpr uint32_t SNAPSHOT_K = MOE_TRACE_SNAPSHOT_K;

// ===========================================================================
// §3. 전역 상태
// ===========================================================================

std::once_flag       g_enabled_once;
std::atomic<bool>    g_enabled_cached{false};

std::once_flag       g_init_once;
std::atomic<bool>    g_ready{false};
int64_t              g_qpc_freq = 0;
uint8_t *            g_ring_base = nullptr;

std::atomic<uint64_t> g_write_pos{0};      // 누적 논리 write 인덱스(래핑 포함)
std::atomic<uint64_t> g_drop_count{0};     // 준비 안 된 상태에서의 기록 시도 총수

// 배선 R1(사양 §2/§3/§4) — prefill 이중 버퍼 + 창 게이팅 + 덮어쓰기/오버플로 계수.
uint8_t *              g_prefill_base[2] = {nullptr, nullptr};
std::atomic<uint64_t>  g_prefill_write_pos[2]{0, 0}; // 각 버퍼의 write 카운터(fetch_add 예약)
std::atomic<int32_t>   g_prefill_cur{0};             // 현재 기록 중인 버퍼 인덱스(0/1)
std::atomic<int32_t>   g_prefill_last_buf{-1};       // 가장 최근 "완료된" prefill graph의 버퍼(초기 -1)
std::atomic<uint64_t>  g_prefill_last_count{0};      // 그 버퍼의 레코드 수
// g_window_mode: 0=발행 안 함, 1=decode 창(링), 2=prefill capture(현재 prefill 버퍼). 워커
// 스레드가 고빈도 write 경로에서 relaxed load로 읽는 atomic — 갱신은 반드시 §5 seq_lock
// 임계구역 안에서만(아래 issue_seq_tick_wall_graph_begin/end 참조, 사양 "코드로 확정").
std::atomic<int32_t>   g_window_mode{0};
uint32_t                g_run_decode_graph_idx = 0;  // 이 run 안에서의 0-based decode graph 인덱스
                                                      // (단일 호출 스레드 전용 — g_graph_context와
                                                      // 동일 근거로 락 불요, run_begin에서 리셋)
std::atomic<uint64_t>  g_overwrite_count{0};          // 링을 한 바퀴 넘게 덮어쓴 횟수
std::atomic<uint64_t>  g_prefill_overflow_count{0};   // prefill 버퍼가 가득 차 못 쓴 횟수
std::atomic<uint64_t>  g_flush_base_pos{0};           // 마지막 flush 시점의 write_pos(초기 0)
std::atomic<uint64_t>  g_anchor_negative_count{0};    // ★S7a — anchor delta raw 음수가 발생한
                                                       // COMPUTE_EXIT 수(세션 누적, O6당 1회).
                                                       // ★창 밖에서 발생한 것도 센다 — 레코드는
                                                       // write_gated_record_bytes()에서 생략되지만
                                                       // 이 계산·계수는 그 전에 일어나므로, 창
                                                       // 밖 역전은 오직 이 카운터로만 드러난다.

std::atomic_flag     g_seq_lock = ATOMIC_FLAG_INIT; // §5 선형화 스핀락
// ★1부터 시작(0이 아님) — 공개 API 전반이 "여는" 관측점의 반환값을 상관 ID로 쓰면서
// 동시에 반환값 0을 "off/미기록"의 얕은 게이트 신호로도 쓴다(모든 관측점 함수의
// `if (!trace_active_and_ready()) return 0;`). event_seq가 0부터 시작하면 프로세스의
// 정말 첫 이벤트가 정확히 그 신호값과 충돌해 "off였다"로 오독된다(실제로 selftest에서
// 발견 — 최초 compute_enter() 호출의 반환값이 0이 되어 버그가 표면화됨). 1부터 시작시켜
// 0을 전 API에 걸쳐 완전히 명확한 sentinel로 만든다(진짜 event_seq는 항상 >=1).
uint64_t              g_event_seq_next = 1;          // 스핀락 내부에서만 접근(비원자로 충분)

// 배선 S6(사양 §3 "델타 계산은 ggml-moe-trace.cpp 안에서") — compute_exit이 first_work_delta를
// 계산하려면 자신을 연 compute_enter(O5) 자신의 qpc_tick이 필요하다. compute_enter/compute_exit는
// 항상 같은 스레드가 짝으로(중첩 없이) 호출하므로(mul_mat_id의 cur_a 루프·add_id의 행 루프 —
// 호출측 참조) thread_local 슬롯 하나로 충분하다 — 다음 compute_enter가 항상 그 다음
// compute_exit보다 먼저 이 값을 덮어쓴다.
thread_local int64_t tls_last_compute_enter_tick = 0;

std::atomic<int32_t> g_compute_active{0};  // useful_compute_active
std::atomic<int32_t> g_app_qd{0};          // app_qd
std::atomic<int32_t> g_wait_active{0};     // 배선 4a 신설(사양 §2-1) — lock_wait_begin/data_wait_begin
                                            // 에서 +1, lock_acquired/data_wait_end에서 -1(app_qd·
                                            // useful_compute_active와 별개 — 위 헤더 API 주석 동일 문구).

// ===========================================================================
// §3-1. 온라인 wall-state 집계(배선 4a 신설 — 사양 §2-2 3상태 분류·§3-2 Lcrit 두 성분·§6-5
// A_pre_ref 집계 window). 세션 누적(cumulative) — phase 인덱스는 MOE_TRACE_PHASE_DECODE(0)/
// PREFILL(1)을 그대로 배열 첨자로 쓴다(헤더의 enum 값과 일치하도록 맞춰 별도 매핑표 불요).
// atomic으로 두는 이유: 쓰기는 항상 §5 seq_lock 임계구역 안(단일 writer, RMW 불요 — load 후
// store로 충분)이지만, 읽기(run_end의 baseline 차분·metrics JSONL 조회·selftest)는 그 락 밖의
// 임의 스레드에서 일어나므로 개별 필드 단위로는 원자성이 필요하다(필드 간 스냅샷 일관성까지는
// 보장하지 않음 — 기존 hit/miss 카운터 등 이 모듈의 다른 누적값들과 동일한 관례).
// ===========================================================================
struct WallStateAccum {
    std::atomic<int64_t>  c_ticks{0};       // |C| 누적(사양 §2-2)
    std::atomic<int64_t>  lcrit_ticks{0};   // |Lcrit| 누적
    std::atomic<int64_t>  other_ticks{0};   // |other| 누적
    std::atomic<int64_t>  l_lock_ticks{0};  // L_lock 성분 누적(사양 §3-2)
    std::atomic<int64_t>  l_data_ticks{0};  // L_data 성분 누적
    std::atomic<uint64_t> mismatch_count{0}; // §6-2 P4 항등식 불일치로 잡힌 graph 수
};
WallStateAccum g_wall_accum[2]; // [MOE_TRACE_PHASE_DECODE], [MOE_TRACE_PHASE_PREFILL]

// 배선 R9 신설(§6-2 P6 재정의 — 표본↔집계 tick 정확 일치 판정 복구) — 창 구간(g_window_mode
// != 0)에만 귀속되는 별도 누산기. 필드 구성은 g_wall_accum과 동일(WallStateAccum 재사용) —
// 3상태 분류·L_lock/L_data 누산 규칙을 두 번 하지 않고, wall_state_accumulate_locked()가 같은
// 분류 결과를 두 누산기에 나눠 담는다(아래 함수 참조).
WallStateAccum g_wall_accum_window[2]; // [MOE_TRACE_PHASE_DECODE], [MOE_TRACE_PHASE_PREFILL]

// ---- 아래는 §5 seq_lock 임계구역 "안에서만" 접근하는 순수 작업 상태(비원자 — g_event_seq_next
// 와 동일 관례). graph_begin~graph_end 구간에만 유효하고 그 밖에서는 g_wall_active_phase가
// UNKNOWN(=2)이라 wall_state_accumulate_locked()가 즉시 반환한다(그래프 밖 시간은 W에 넣지
// 않는다 — 사양 §2-2 "W의 경계 = graph 구간의 합"). ----
uint8_t g_wall_active_phase = MOE_TRACE_PHASE_UNKNOWN; // 활성 창의 phase(그 외엔 UNKNOWN)
int64_t g_wall_last_tick    = 0;                       // 마지막 전이(또는 graph_begin) tick
int32_t g_wall_lock_wait_count = 0;                    // 현재 lock_wait 중인 스레드 수(L_lock 존재 판정)
int32_t g_wall_data_wait_count = 0;                    // 현재 data_wait 중인 스레드 수(L_data 존재 판정)

// graph 항등식 검산(사양 §2-2 6항·§6-2 P4)용 — graph_begin 시점의 그 phase 세션 누적치 스냅샷.
int64_t g_wall_graph_base_c = 0, g_wall_graph_base_lcrit = 0, g_wall_graph_base_other = 0;
int64_t g_wall_graph_begin_tick = 0;

// run별 AGG 레코드(§6-5) 산출용 — run_begin 시점의 phase별 세션 누적치 스냅샷(단일-slot 벤치라
// run 간 겹침이 없으므로 락 없이 메인 스레드에서만 읽고 쓴다 — run_begin/run_end 호출부 참조).
int64_t  g_wall_run_base_c[2] = {0, 0}, g_wall_run_base_lcrit[2] = {0, 0}, g_wall_run_base_other[2] = {0, 0};
int64_t  g_wall_run_base_llock[2] = {0, 0}, g_wall_run_base_ldata[2] = {0, 0};
uint64_t g_wall_run_base_mismatch[2] = {0, 0};

// 배선 R9 신설 — g_wall_accum_window용 run_begin baseline 스냅샷(위 g_wall_run_base_*와 동일
// 패턴). run_end에서 이 baseline과의 차분을 MOE_TRACE_RT_WALLSTATE_WIN_AGG로 발행한다.
int64_t  g_wall_win_run_base_c[2] = {0, 0}, g_wall_win_run_base_lcrit[2] = {0, 0}, g_wall_win_run_base_other[2] = {0, 0};
int64_t  g_wall_win_run_base_llock[2] = {0, 0}, g_wall_win_run_base_ldata[2] = {0, 0};
uint64_t g_wall_win_run_base_mismatch[2] = {0, 0};

// 배선 4d(사양 §6-5 read_bytes delta_pre) — run_begin 시점의 phase별 read_bytes/hit/miss
// 세션 누적치 스냅샷(ggml_moe_direct_phase_read_bytes_hit_miss_exact() 조회값 — R3 배선).
// g_wall_run_base_*
// 와 동일 관례(단일-slot 벤치 — run 간 겹침 없으므로 락 없이 메인 스레드에서만 읽고 쓴다).
uint64_t g_io_run_base_read_bytes[2] = {0, 0}, g_io_run_base_hit[2] = {0, 0}, g_io_run_base_miss[2] = {0, 0};

// 배선 S3a(외부 심의 OV6 E절 차단 1 — IOSTATE_WIN_AGG) — 위 g_io_run_base_*와 동일 관례,
// ggml_moe_direct_phase_read_bytes_hit_miss_window() 조회값(창-scope)의 run_begin baseline.
uint64_t g_io_win_run_base_read_bytes[2] = {0, 0}, g_io_win_run_base_hit[2] = {0, 0}, g_io_win_run_base_miss[2] = {0, 0};

std::atomic<uint64_t> g_acquire_call_count{0};
std::atomic<uint64_t> g_n_ids_hist[8]{};   // idx0..6=n_ids 1..7, idx7=n_ids>=8

// 배선 3단계(사양 P2 대조용): mul_mat_id chunk while 루프의 실제 one_chunk 호출 수.
std::atomic<uint64_t> g_chunk_call_count{0};

// 배선 3단계(O8 "처음" 판정 — 사양 §3-1 O8/§8 U5): (layer, tensor_part, expert)마다 "마지막
// 으로 claim된 graph_seq"를 저장한다. ★크기 근거: wire format의 layer/expert 필드는 uint8_t
// (0..255)이고 tensor_part는 계약상 0..5 고정 6종이다 — moe-direct 모듈의 프로덕션 상수
// (36층·128 expert)에 결합하지 않고 필드 폭이 표현 가능한 전체 범위를 커버해 이 파일이
// moe-direct.cpp의 사적 상수를 몰라도 항상 안전하게 만들었다(헤더 서문의 "sensor_id를 이
// 파일들의 SHA로 정의" 취지 — 결합도 최소화). 크기 = 256×6×256×4B = 1.5MiB, 정적 배열
// (256MiB 링버퍼 예산 대비 무시 가능 — 배선 R1 §1). ★sentinel = UINT32_MAX(0이 아님) — graph_seq는 0부터
// 시작하는 정당한 값이므로(llama-context.cpp s_moe_trace_graph_seq_next 최초값 0, 26-07-20
// 확인) 0을 sentinel로 쓰면 그래프 0의 첫 claim이 잘못 판정된다. UINT32_MAX는 실제 그래프가
// 42억 개를 넘지 않는 한(비현실적) 항상 안전한 "미접촉" 표식이다.
constexpr uint32_t O8_LAYER_DIM  = 256;
constexpr uint32_t O8_PART_DIM   = 6;
constexpr uint32_t O8_EXPERT_DIM = 256;
constexpr size_t   O8_TABLE_SIZE = (size_t)O8_LAYER_DIM * (size_t)O8_PART_DIM * (size_t)O8_EXPERT_DIM;
constexpr uint32_t O8_UNTOUCHED  = 0xFFFFFFFFu;
std::atomic<uint32_t> g_o8_last_graph_seq[O8_TABLE_SIZE];

// 배선 2단계(사양 §4-1/§4-2 "코드로 확정" — 최종 보고서의 코드 확정 표 참조): run_index/
// graph_seq 카운터 + graph 문맥 스냅샷. run_index는 server-context.cpp(run_begin)가 발급하고
// llama-context.cpp(graph_begin)가 읽는다 — 두 TU가 서로 다른 링크 단위(서버 exe ↔
// libllama)라 상태를 ggml(공통 하위 의존)에 둬야 공유가 가능하다.
std::atomic<uint32_t> g_run_index_next{0};     // 다음에 발급할 run_index
std::atomic<uint32_t> g_current_run_index{0};  // 마지막으로 발급된 run_index(증가 없는 조회용)

// graph 문맥 스냅샷 — {run_index:16, graph_seq:32, phase:8}을 단일 원자 워드로 패킹한다.
// ★선택 근거(단일 atomic<uint64_t> release/acquire, seqlock 아님): 쓰기는 항상 graph_begin/
// end를 호출하는 그 서버 스레드 하나뿐이고(동시 writer 없음 — llama_context::decode()의
// ubatch 루프는 단일 스레드 순차 실행), 그 쓰기는 graph_compute() 호출(워커 스레드를
// 깨우는 지점) 이전에 완료된다 — 즉 "쓰기 happens-before 모든 읽기"가 호출 경계 자체로
// 보장되어 seqlock의 "쓰기 도중 읽기" 방어가 애초에 불필요하다. 3필드가 56비트에 들어가
// 하나의 원자 워드로 찢어짐 없이(torn-read 없이) 담기므로, release-store 1회 + acquire-load
// 1회로 충분하다(스핀/재시도 루프 불요 — O5/O6이 배선될 다음 배치에서 read 빈도가 더
// 늘어나므로 여기서 비용을 최소화해 둔다).
std::atomic<uint64_t> g_graph_context{(uint64_t)MOE_TRACE_PHASE_UNKNOWN};

inline uint64_t pack_graph_context(uint16_t run_index, uint32_t graph_seq, uint8_t phase) {
    return ((uint64_t)run_index << 40) | ((uint64_t)graph_seq << 8) | (uint64_t)phase;
}
inline void unpack_graph_context(uint64_t v, uint16_t & run_index, uint32_t & graph_seq, uint8_t & phase) {
    phase     = (uint8_t)(v & 0xFFu);
    graph_seq = (uint32_t)((v >> 8) & 0xFFFFFFFFu);
    run_index = (uint16_t)((v >> 40) & 0xFFFFu);
}

// ===========================================================================
// §4. 유틸리티
// ===========================================================================

bool local_env_is_1(const char * name) {
    const char * v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}

std::wstring local_utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

[[noreturn]] void fatal_trace_abort(const char * what) {
    GGML_ABORT("[moe-trace] FATAL gauge/metrics invariant violated: %s", what);
}

int16_t saturate_i32_to_i16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

// 배선 R1(사양 §6) — file header의 overwrite_count/prefill_overflow_count(uint32_t) 포화 변환.
uint32_t saturate_u64_to_u32(uint64_t v) {
    return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)v;
}

// 배선 S7a — 이전 saturate_anchor_delta()는 음수를 조용히 0으로 만들어 "정상 0-delta"와
// 구분 불가였다(가장 확실한 고장 신호가 정상값으로 세탁됨). 이제 음수를 out 파라미터로
// 알린다. wire 값은 계속 0을 싣는다(3바이트 필드에 음수를 표현할 수 없고 레코드 크기를
// 늘릴 수 없으므로 — 위치 정보는 flags, 총계는 header 카운터가 담당).
// ★S6 원주석에서 보존(메인 검수 26-07-21 — 시공이 교체하며 유실한 구체 사례): 음수는 §5
// 스핀락 설계상 도달 불가하지만, 이 함수가 받는 것은 호출측이 넘긴 원시 tick 두 개의
// 뺄셈이다(예: query_tick()이 off로 0을 반환한 값이 게이트 누락으로 그대로 들어온 경우).
// 그 사례가 곧 anchor_negative_count>0의 해석 단서다 — 카운터가 섰을 때 "역전이 났다"만이
// 아니라 "호출측 게이트가 뚫렸을 수 있다"까지 의심해야 한다.
constexpr int64_t ANCHOR_DELTA_CAP = 0xFFFFFF; // 16,777,215 — FixedRecord._pad 주석 참조
uint32_t encode_anchor_delta(int64_t raw, bool & negative_out) {
    if (raw < 0) {
        negative_out = true;
        return 0;
    }
    negative_out = false;
    return raw > ANCHOR_DELTA_CAP ? (uint32_t)ANCHOR_DELTA_CAP : (uint32_t)raw;
}

// ===========================================================================
// §5. 선형화 primitive (사양 §3-5-2 "코드로 확정" 지점)
//
// ★선택: 초경량 스핀락(test-and-set 기반) — "QPC 채취 + event_seq 발급"을 하나의 배타적
// 임계구역으로 묶는다. 후보였던 "단일 InterlockedIncrement 후 QPC 채취 + ±1 tick 역전
// 허용"은 MS 공식 문서(learn.microsoft.com/windows/win32/sysinfo/acquiring-high-resolution-
// time-stamps, 26-07-20 WebFetch 확인)가 명시하는 "스레드 간 비교 시 ±1 tick 모호성"과는
// 성격이 다른 위험을 동반한다 — 그 ±1 tick 모호성은 하드웨어(QPC 자체)의 코어 간 판독
// 오차이고, seq 증가와 QPC 판독을 두 개의 분리된 non-atomic 문장으로 실행하면 그 사이에
// 스레드가 선점(preemption)될 수 있어 seq 순서와 tick 순서가 "몇 tick"이 아니라 "스케줄러
// 지연만큼"(수 마이크로초~밀리초, 즉 수천~수백만 tick) 어긋날 수 있다. 스핀락은 이 위험을
// 완전히 제거한다(seq와 tick이 항상 같은 배타 구간에서 함께 결정되므로 역전이 원천적으로
// 0건 — spec의 ±1 tick 허용선보다 엄격).
//
// 비용: 이벤트가 chunk 단위가 아니라 op×worker 단위로 축소되어 8스레드가 이 임계구역을 두고
// 경합한다.
// ★★수치 정정(26-07-21 — 메인 검수): 구 주석의 "decode graph당 2,784건(29층×6op×8워커×2)"은
// **expert 루프를 누락한 값**이었다(외부 심의 OV5 차단 1이 지적한 바로 그 오산 — 사양은 v6에서
// 정정됐는데 이 주석만 남아 있었고, 실제로 배치 S6가 이 폐기값을 1차 소스로 삼아 비용을 절반으로
// 과소 산정하는 사고가 났다). `mul_mat_id`의 O5/O6은 `for (cur_a...)` **expert 루프 안**에
// 있으므로(ggml-cpu.c) 활성 expert(top-4)마다 찍힌다:
//     mul  29층 × 3 part × 4 expert × 8 워커 × 2(enter/exit) = 5,568
//     add  29층 × 3 part × 4 워커(nr=4<nth=8이라 빈 워커 발생) × 2 =   696
//     계  decode graph당 약 6,264건(= op×worker 쌍 약 3,132)
// 완전 직렬화 가정 시 graph당 6,264 × (~100 cycles) ≈ 626,400 cycles ≈ 3.7GHz 기준 약 169 μs
// — decode 1토큰의 벽시계 시간(수십 ms 규모, sustained 5~11 tok/s 실측 기준)에 비하면 여전히
// 작지만(약 0.2~1.7%) 구 주석이 말한 값의 2.25배이므로 그 전제로 계산하지 말 것.
// ★배치 S6의 anchor 채취(ggml_moe_trace_query_tick)는 이 임계구역 **밖**이라 경합 대상이
// 아니다 — op×worker 쌍당 2회(graph당 약 6,264회)가 추가되지만 직렬화되지 않는다. YieldProcessor()(winnt.h, x86/x64에서 _mm_pause
// 매핑 — learn.microsoft.com/windows/win32/api/winnt/nf-winnt-yieldprocessor 확인)로
// 스핀 중 하이퍼스레딩 형제 코어에 자원을 양보해 과도한 버스 점유를 줄인다.
//
// 대안 트레이드오프: "단일 증가+독립 QPC" 방식은 임계구역이 없어 이론상 더 저렴하지만,
// 위에서 설명한 무한(스케줄러 의존) 역전 위험을 안고 그 위험을 판독기가 "역전 감지+보정"
// 로직으로 흡수해야 한다 — 코드 복잡도가 계측 모듈에서 판독기로 이동할 뿐 사라지지
// 않는다. 이번 배치는 계측기 자체의 정직성이 목적(SPEC 상위 계약)이므로, 약간의 CPU
// 비용을 치르더라도 순서 보장을 하드웨어가 아니라 설계로 확정하는 쪽을 택했다.
// ===========================================================================

inline void seq_lock_acquire() {
    while (g_seq_lock.test_and_set(std::memory_order_acquire)) {
        YieldProcessor();
    }
}
inline void seq_lock_release() {
    g_seq_lock.clear(std::memory_order_release);
}

// ★배선 R2(사양 §1 불변식): 어떤 레코드든 (event_seq, qpc_tick, qd_after, compute_active_after)가
// 같은 임계구역에서 확정된 한 벌이어야 한다 — 아래 4개 seq 발급 경로(issue_seq_and_tick·
// issue_seq_tick_with_wall_transition·issue_seq_tick_wall_graph_begin·
// issue_seq_tick_wall_graph_end)는 전부 이 임계구역 안에서 두 게이지를 채취해 호출측에
// 반환한다(호출측이 나중에 게이지를 별도로 재조회하지 않도록).
uint64_t issue_seq_and_tick(int64_t & out_tick, int32_t & out_qd_after, int32_t & out_compute_active_after) {
    seq_lock_acquire();
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc); // 1차 소스 확인: XP+에서 유효 인자로는 항상 성공(반환값 무시 가능)
    out_tick = qpc.QuadPart;
    out_qd_after = g_app_qd.load(std::memory_order_relaxed);
    out_compute_active_after = g_compute_active.load(std::memory_order_relaxed);
    uint64_t seq = g_event_seq_next++;
    seq_lock_release();
    return seq;
}

// ===========================================================================
// §5-1. 온라인 wall-state 전이 누산(배선 4a — 사양 §2-2 "코드로 확정" 요구: "게이지 갱신·
// QPC 채취·상태 전이 누산을 모두 기존 선형화 스핀락 구간 안에서 수행"). 아래 세 함수는
// 위 issue_seq_and_tick()과 동일한 g_seq_lock 임계구역을 쓰되, compute_active/wait_active/
// lock_wait/data_wait 4종 게이지 중 하나라도 바뀌는 6개 관측점(O3a/O3b·O3/O4·O5/O6)과
// graph_begin/graph_end 전용으로 그 구간을 확장한다. io_submit/io_complete(app_qd)·
// ready_publish·first_arith_consume·demand/snapshot·run_begin/run_end는 이 4종 게이지를
// 건드리지 않으므로 위 issue_seq_and_tick() 그대로 쓴다(범위 최소화 — "겹침 구현·락 분해
// 금지, 기존 관측점·게이지 의미 불변" 지시 그대로).
//
// ★순서가 핵심이다: "이 이벤트로 게이지가 바뀌기 직전 상태"에 경과시간을 귀속시킨 "다음"에
// 게이지를 갱신해야 한다(뒤바뀌면 그 이벤트 자신이 만든 새 상태로 그 이전 구간까지 잘못
// 귀속된다). 아래 wall_state_accumulate_locked()가 그 귀속을, 각 issue_seq_tick_wall_*()가
// 그 다음의 게이지 갱신을 맡는다 — 반드시 이 순서로 호출할 것(이 파일 안에서만 쓰이는
// 내부 계약이라 별도 방어 코드는 두지 않는다).
// ===========================================================================

// 활성 창(graph_begin~graph_end)이 있을 때만: tick 이전까지의 경과를 "그 직전" 3-state
// (C/Lcrit/other)와 L_lock/L_data 존재 여부에 귀속하고 g_wall_last_tick을 tick으로 갱신한다.
// 호출측이 g_seq_lock을 이미 쥔 상태에서만 호출할 것(비원자 작업 상태를 읽고 쓴다).
inline void wall_state_accumulate_locked(int64_t tick) {
    if (g_wall_active_phase >= 2) return; // 활성 graph 없음(UNKNOWN) — 귀속 대상 없음(사양 §2-2 "W의 경계 = graph 구간의 합")
    int64_t elapsed = tick - g_wall_last_tick;
    if (elapsed < 0) elapsed = 0; // 방어적(이 임계구역 안에서는 QPC가 항상 비내림차순 — §5 설계 그대로 발생 불가)

    const int32_t compute_active = g_compute_active.load(std::memory_order_relaxed);
    const int32_t wait_active    = g_wait_active.load(std::memory_order_relaxed);
    WallStateAccum & acc = g_wall_accum[g_wall_active_phase];
    // 배선 R9(사양 §1) — g_window_mode는 이 함수의 모든 호출부가 이미 g_seq_lock 임계구역
    // 안에 있고(issue_seq_and_tick 계열 함수들 참조), g_window_mode 자체도 그 같은 임계구역
    // 안에서만 갱신되므로(issue_seq_tick_wall_graph_begin/end — 배선 R1) 이 시점의 판독은
    // "이 elapsed 구간 내내 유효했던 값"과 항상 일치한다(같은 락 보유자만 갱신 가능하므로
    // 구간 도중 값이 바뀐 채 이 함수가 호출될 수 없음).
    WallStateAccum * win_acc = g_window_mode.load(std::memory_order_relaxed) != 0
        ? &g_wall_accum_window[g_wall_active_phase] : nullptr;
    // 3상태 분류(사양 §2-2, 상호배타·완전분할 — C를 먼저 검사): C >= Lcrit >= other 우선순위.
    // 창 누산기는 분류를 다시 하지 않고 같은 분류 결과를 그대로 나눠 담는다.
    if (compute_active >= 1) {
        acc.c_ticks.fetch_add(elapsed, std::memory_order_relaxed);
        if (win_acc) win_acc->c_ticks.fetch_add(elapsed, std::memory_order_relaxed);
    } else if (wait_active >= 1) {
        acc.lcrit_ticks.fetch_add(elapsed, std::memory_order_relaxed);
        if (win_acc) win_acc->lcrit_ticks.fetch_add(elapsed, std::memory_order_relaxed);
    } else {
        acc.other_ticks.fetch_add(elapsed, std::memory_order_relaxed);
        if (win_acc) win_acc->other_ticks.fetch_add(elapsed, std::memory_order_relaxed);
    }
    // L_lock/L_data는 Lcrit과 "정의가 다르다"(사양 §3-2 코드 확정 사항) — compute_active와
    // 무관하게 "그 대기가 존재한 시간"을 각각 별도로 누산한다(산술 중에도 데이터 대기가
    // 걸려 있으면 L_data는 늘지만 위 Lcrit 분기는 C로 잡혀 늘지 않는다 — "가려진 대기").
    if (g_wall_lock_wait_count >= 1) {
        acc.l_lock_ticks.fetch_add(elapsed, std::memory_order_relaxed);
        if (win_acc) win_acc->l_lock_ticks.fetch_add(elapsed, std::memory_order_relaxed);
    }
    if (g_wall_data_wait_count >= 1) {
        acc.l_data_ticks.fetch_add(elapsed, std::memory_order_relaxed);
        if (win_acc) win_acc->l_data_ticks.fetch_add(elapsed, std::memory_order_relaxed);
    }

    g_wall_last_tick = tick;
}

// compute_active/wait_active(+lock/data wait count)/app_qd 중 하나를 갱신하는 관측점 전용.
// ★배선 R2(사양 §1-a): QdInc/QdDec 신설 — app_qd 증감을 이 임계구역 안으로 옮긴다. app_qd는
// wall-state 3분할 판정(wall_state_accumulate_locked)에 참여하지 않으므로 그 분류 로직은
// 그대로 두고, 같은 임계구역을 공유할 뿐이다(io_submit/io_complete 호출부 참조).
enum class WallGaugeOp {
    ComputeInc, ComputeDec,
    LockWaitInc, LockWaitDec,
    DataWaitInc, DataWaitDec,
    QdInc, QdDec,
};

// out_gauge_after = 이 op가 직접 갱신한 게이지의 갱신 후 값(app_qd 음수 등 불변식 검사용).
// out_qd_after/out_compute_active_after = 위 §1 불변식 요구대로 항상 두 게이지 모두 채취해
// 반환(이 op이 그 게이지를 갱신하지 않았어도 현재값을 그대로 싣는다).
uint64_t issue_seq_tick_with_wall_transition(WallGaugeOp op, int64_t & out_tick, int32_t & out_gauge_after,
                                              int32_t & out_qd_after, int32_t & out_compute_active_after) {
    seq_lock_acquire();
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    const int64_t tick = qpc.QuadPart;

    wall_state_accumulate_locked(tick); // 1) 바뀌기 "직전" 상태에 경과 귀속

    int32_t after = 0; // 2) 게이지 갱신(귀속 다음에만 — 순서 주석 참조)
    switch (op) {
        case WallGaugeOp::ComputeInc:
            after = g_compute_active.load(std::memory_order_relaxed) + 1;
            g_compute_active.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::ComputeDec:
            after = g_compute_active.load(std::memory_order_relaxed) - 1;
            g_compute_active.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::LockWaitInc:
            g_wall_lock_wait_count++;
            after = g_wait_active.load(std::memory_order_relaxed) + 1;
            g_wait_active.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::LockWaitDec:
            g_wall_lock_wait_count--;
            after = g_wait_active.load(std::memory_order_relaxed) - 1;
            g_wait_active.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::DataWaitInc:
            g_wall_data_wait_count++;
            after = g_wait_active.load(std::memory_order_relaxed) + 1;
            g_wait_active.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::DataWaitDec:
            g_wall_data_wait_count--;
            after = g_wait_active.load(std::memory_order_relaxed) - 1;
            g_wait_active.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::QdInc:
            after = g_app_qd.load(std::memory_order_relaxed) + 1;
            g_app_qd.store(after, std::memory_order_relaxed);
            break;
        case WallGaugeOp::QdDec:
            after = g_app_qd.load(std::memory_order_relaxed) - 1;
            g_app_qd.store(after, std::memory_order_relaxed);
            break;
    }

    out_qd_after = g_app_qd.load(std::memory_order_relaxed);
    out_compute_active_after = g_compute_active.load(std::memory_order_relaxed);

    uint64_t seq = g_event_seq_next++;
    seq_lock_release();

    out_tick = tick;
    out_gauge_after = after;
    return seq;
}

// graph_begin 전용 — 새 wall-state 창을 연다. 방어적으로 먼저 이전 창(있다면 — 정상 경로에서는
// 이미 graph_end가 닫아 g_wall_active_phase==UNKNOWN이라 사실상 no-op)의 잔여를 귀속한 뒤,
// 이 phase의 세션 누적치를 항등식 검산용 baseline으로 스냅샷하고 창을 연다.
// ★배선 R1(사양 §3): window_mode(0/1/2)·prefill_next_buf(전환할 버퍼, 전환 없으면 -1)는
// 호출측(ggml_moe_trace_graph_begin)이 단일 스레드 전용 상태(g_run_decode_graph_idx 등)로
// 미리 "결정"한 값을 받아 여기서 "발행"만 한다 — g_window_mode/g_prefill_cur는 워커가
// relaxed load로 읽는 atomic이므로 이 seq_lock 임계구역 안에서만 갱신해야 하기 때문이다.
uint64_t issue_seq_tick_wall_graph_begin(uint8_t phase, int32_t window_mode, int32_t prefill_next_buf,
                                          int64_t & out_tick, int32_t & out_qd_after,
                                          int32_t & out_compute_active_after) {
    seq_lock_acquire();
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    const int64_t tick = qpc.QuadPart;

    wall_state_accumulate_locked(tick); // 방어적(정상 경로에서는 활성 창 없음 — no-op)

    if (prefill_next_buf >= 0) {
        g_prefill_write_pos[prefill_next_buf].store(0, std::memory_order_relaxed);
        g_prefill_cur.store(prefill_next_buf, std::memory_order_relaxed);
    }
    g_window_mode.store(window_mode, std::memory_order_relaxed);

    g_wall_active_phase = phase;
    g_wall_last_tick    = tick;
    g_wall_graph_begin_tick = tick;
    if (phase < 2) {
        WallStateAccum & acc = g_wall_accum[phase];
        g_wall_graph_base_c     = acc.c_ticks.load(std::memory_order_relaxed);
        g_wall_graph_base_lcrit = acc.lcrit_ticks.load(std::memory_order_relaxed);
        g_wall_graph_base_other = acc.other_ticks.load(std::memory_order_relaxed);
    }

    // 배선 R2(사양 §1-b) — 이 임계구역 안에서 두 게이지 모두 채취해 반환.
    out_qd_after = g_app_qd.load(std::memory_order_relaxed);
    out_compute_active_after = g_compute_active.load(std::memory_order_relaxed);

    uint64_t seq = g_event_seq_next++;
    seq_lock_release();
    out_tick = tick;
    return seq;
}

// graph_end 전용 — 창을 닫는다: 이 tick까지 경과를 귀속하고, |C|+|Lcrit|+|other| == graph 구간
// 길이(사양 §6-2 P4)를 tick 단위로 정확히 검증한다. 불일치 시 그 phase의 mismatch_count를
// 증가시킨다(항상 이 함수 자체가 trace_active_and_ready() 게이트를 통과한 뒤에만 호출되므로
// 별도 게이트 불요).
uint64_t issue_seq_tick_wall_graph_end(uint8_t phase, int64_t & out_tick, int32_t & out_qd_after,
                                        int32_t & out_compute_active_after) {
    seq_lock_acquire();
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    const int64_t tick = qpc.QuadPart;

    wall_state_accumulate_locked(tick); // 이 graph의 마지막 구간을 귀속

    if (phase < 2) {
        WallStateAccum & acc = g_wall_accum[phase];
        const int64_t c_delta     = acc.c_ticks.load(std::memory_order_relaxed)     - g_wall_graph_base_c;
        const int64_t lcrit_delta = acc.lcrit_ticks.load(std::memory_order_relaxed) - g_wall_graph_base_lcrit;
        const int64_t other_delta = acc.other_ticks.load(std::memory_order_relaxed) - g_wall_graph_base_other;
        const int64_t expected    = tick - g_wall_graph_begin_tick;
        if (c_delta + lcrit_delta + other_delta != expected) {
            acc.mismatch_count.fetch_add(1, std::memory_order_relaxed);
            // 배선 R9(사양 §1 "mismatch_count도 동일하게 창 버전을 유지") — g_window_mode는
            // 이 graph 내내 상수다(issue_seq_tick_wall_graph_begin에서 설정된 뒤 아래에서만
            // 리셋 — 위 wall_state_accumulate_locked 주석과 동일 근거). 따라서 창이 활성이던
            // graph는 창 누산기의 증분도 세션과 정확히 같으므로(같은 elapsed가 매번 두
            // 누산기에 함께 더해짐) 별도 base 스냅샷 없이 이 c_delta/lcrit_delta/other_delta/
            // expected를 그대로 재사용해 판정을 공유한다.
            if (g_window_mode.load(std::memory_order_relaxed) != 0) {
                g_wall_accum_window[phase].mismatch_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // 배선 R1(사양 §3) — 창 닫기: 직전이 prefill capture였다면(mode==2) 그 버퍼를 "마지막
    // 완료된 prefill graph"로 확정한다(§2 이중 버퍼 계약 — flush()가 이 값을 읽어간다).
    // write 카운터는 버퍼 용량으로 클램프(§4 — 용량 초과분은 애초에 write_prefill_record_bytes
    // 가 물리적으로 쓰지 않았으므로, 클램프하지 않으면 이전 graph의 잔존 레코드까지 유효
    // 구간으로 잘못 확정된다).
    if (g_window_mode.load(std::memory_order_relaxed) == 2) {
        int32_t buf = g_prefill_cur.load(std::memory_order_relaxed);
        uint64_t written = g_prefill_write_pos[buf].load(std::memory_order_relaxed);
        if (written > PREFILL_WIN_CAPACITY_RECORDS) written = PREFILL_WIN_CAPACITY_RECORDS;
        g_prefill_last_buf.store(buf, std::memory_order_relaxed);
        g_prefill_last_count.store(written, std::memory_order_relaxed);
    }
    g_window_mode.store(0, std::memory_order_relaxed); // 창 밖(gating 사양 §3 "graph_end에서 0")

    g_wall_active_phase = MOE_TRACE_PHASE_UNKNOWN; // 창 닫음(밖에서 fire해도 안전하게 무해화)

    // 배선 R2(사양 §1-b) — 이 임계구역 안에서 두 게이지 모두 채취해 반환.
    out_qd_after = g_app_qd.load(std::memory_order_relaxed);
    out_compute_active_after = g_compute_active.load(std::memory_order_relaxed);

    uint64_t seq = g_event_seq_next++;
    seq_lock_release();
    out_tick = tick;
    return seq;
}

// ===========================================================================
// §6. 링버퍼 쓰기 — 슬롯 예약은 seq 스핀락과 독립된 lock-free atomic(§5 임계구역을
// 최대한 짧게 유지하기 위해 분리). 물리적 슬롯 순서가 event_seq 순서와 다를 수 있으나
// 무방하다 — 각 레코드가 자신의 event_seq/qpc_tick을 이미 담고 있으므로 canonical 순서는
// 그 필드로 복원한다(링 내 물리 위치는 순수 저장소일 뿐).
// ===========================================================================

void write_record_bytes(const void * rec, size_t n) {
    (void)n; // 항상 RECORD_SIZE(호출부가 자신의 구조체 sizeof를 넘기며, 전부 static_assert로 64 보장)
    uint64_t idx = g_write_pos.fetch_add(1, std::memory_order_relaxed);
    // 배선 R1(사양 §4) — 마지막 flush 이후 링을 한 바퀴 넘게 덮어쓰면 표본이 조용히 유실된다
    // (외부 심의 OV5 차단 1). 레코드는 그대로 쓰되(현행 동작 유지) 덮어쓴 횟수만 별도 계수해
    // timeline_drop_count 합성값으로 판독기가 무효화할 수 있게 한다.
    if (idx - g_flush_base_pos.load(std::memory_order_relaxed) >= RING_CAPACITY_RECORDS) {
        g_overwrite_count.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t slot = idx & RING_MASK;
    std::memcpy(g_ring_base + slot * RECORD_SIZE, rec, RECORD_SIZE);
}

// 배선 R1(사양 §2/§4) — 현재 prefill 버퍼에 쓴다. 용량 도달 시 더 쓰지 않고 오버플로만 계수
// (사양 §4 "카운터가 도달하면 더 쓰지 않고 fetch_add").
void write_prefill_record_bytes(const void * rec, size_t n) {
    (void)n;
    int32_t buf = g_prefill_cur.load(std::memory_order_relaxed);
    uint64_t idx = g_prefill_write_pos[buf].fetch_add(1, std::memory_order_relaxed);
    if (idx >= PREFILL_WIN_CAPACITY_RECORDS) {
        g_prefill_overflow_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::memcpy(g_prefill_base[buf] + idx * RECORD_SIZE, rec, RECORD_SIZE);
}

// 배선 R1(사양 §3) — 고빈도 이벤트 전용 창 게이팅 write. write_record_bytes 호출 여부만
// 좌우한다: 그 외 모든 상태 갱신(event_seq 발급·QPC 채취·게이지·wall-state 누산 등)은
// 호출측에서 게이팅과 무관하게 항상 수행된다(사양 "★절대 준수" — 이 함수는 그 원칙이 지켜진
// *이후*의 마지막 단계일 뿐이다). mode==0(창 밖)이면 아무것도 쓰지 않는다.
void write_gated_record_bytes(const void * rec, size_t n) {
    switch (g_window_mode.load(std::memory_order_relaxed)) {
        case 1: write_record_bytes(rec, n); break;
        case 2: write_prefill_record_bytes(rec, n); break;
        default: break; // 0 = 창 밖 — 발행 안 함
    }
}

// ===========================================================================
// §7. 마스터 스위치 + 지연 초기화
// ===========================================================================

bool trace_enabled_cached_check() {
    std::call_once(g_enabled_once, []() {
        // 배선 2단계: MOE_TRACE → MOE_P1_TRACE 개명(같은 트리의 examples/moe-trace/가
        // MOE_TRACE_OUT·MOE_TRACE_OFF를 쓰고 있어 이름 근접 혼동을 피한다 — 헤더 주석 참조).
        g_enabled_cached.store(local_env_is_1("MOE_P1_TRACE"), std::memory_order_relaxed);
    });
    return g_enabled_cached.load(std::memory_order_relaxed);
}

void do_init_once() {
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0) {
        return; // g_ready는 false로 남음 — 이후 모든 호출이 drop 계수
    }
    void * base = VirtualAlloc(nullptr, (SIZE_T)RING_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!base) {
        return;
    }
    // 배선 R1(사양 §2) — prefill 이중 버퍼 선할당. 실패 시 링과 동일하게 g_ready는 false로
    // 남긴다(이미 확보한 링 버퍼는 해제하지 않는다 — 기존 관례와 동일하게 프로세스 종료 시
    // OS가 회수).
    void * prefill_base0 = VirtualAlloc(nullptr, (SIZE_T)PREFILL_WIN_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void * prefill_base1 = VirtualAlloc(nullptr, (SIZE_T)PREFILL_WIN_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!prefill_base0 || !prefill_base1) {
        return;
    }
    g_ring_base = (uint8_t *)base;
    g_prefill_base[0] = (uint8_t *)prefill_base0;
    g_prefill_base[1] = (uint8_t *)prefill_base1;
    g_qpc_freq = freq.QuadPart;
    // O8 dedup 표를 "미접촉" sentinel(UINT32_MAX)로 채운다 — 트레이스가 실제로 켜질 때
    // (이 함수는 trace_active_and_ready() 경유로만 호출) 딱 한 번만 드는 비용이라 off
    // 세션에는 전혀 영향이 없다(g_o8_last_graph_seq[] 배열 자체는 BSS라 이 루프가 돌기
    // 전까지 커밋되지 않은 0-페이지 상태로 남는다).
    for (size_t i = 0; i < O8_TABLE_SIZE; i++) {
        g_o8_last_graph_seq[i].store(O8_UNTOUCHED, std::memory_order_relaxed);
    }
    g_ready.store(true, std::memory_order_release);
}

// off면 false(카운트 없음 — 설계상 정상 경로). on인데 초기화 실패/미완이면 drop 계수 후 false.
bool trace_active_and_ready() {
    if (!trace_enabled_cached_check()) return false;
    std::call_once(g_init_once, do_init_once);
    if (!g_ready.load(std::memory_order_acquire)) {
        g_drop_count.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// ===========================================================================
// §8. 레코드 빌더 헬퍼
// ===========================================================================

// "여는" 관측점 공통 — correlation_id = 자기 자신의 event_seq(§2 주석). ★_ex는 seq/tick을
// 이미 발급받은 경우(배선 4a — compute_enter·lock_wait_begin·data_wait_begin이 wall-state
// 전이 누산과 게이지 갱신을 겸하는 issue_seq_tick_with_wall_transition()을 먼저 호출해야
// 하므로) 그 값을 그대로 채워 넣는다. 필드 채우기 로직 중복을 피하기 위해 얇은 래퍼
// make_base_record()가 기존 호출부(seq/tick을 몰라도 되는 이벤트) 하위호환을 유지한다.
// ★배선 R2(사양 §1-c/§1-d): 게이지를 여기서 직접 읽지 않고 호출측이 같은 임계구역(§5 seq
// lock)에서 이미 채취한 값을 인자로 받는다 — 어떤 레코드든 (event_seq, qpc_tick, qd_after,
// compute_active_after)가 같은 임계구역에서 확정된 한 벌이어야 한다는 불변식(위 issue_seq_
// and_tick 주석 참조)을 이 함수가 스스로 깨지 않기 위함이다.
FixedRecord make_base_record_ex(uint8_t event_kind, uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                 uint8_t layer, uint8_t tensor_part, uint8_t expert, uint32_t generation_low32,
                                 uint64_t seq, int64_t tick, int32_t qd_after, int32_t compute_active_after) {
    FixedRecord r{};
    r.type = MOE_TRACE_RT_FIXED;
    r.event_kind = event_kind;
    r.phase = phase;
    r.tensor_part = tensor_part;
    r.layer = layer;
    r.expert = expert;
    r.flags = 0;
    r.status = MOE_TRACE_IO_STATUS_OK;
    r.qpc_tick = tick;
    r.event_seq = seq;
    r.correlation_id = seq;
    r.thread_id = (uint32_t)GetCurrentThreadId();
    r.generation_low32 = generation_low32;
    r.bytes_requested = 0;
    r.bytes_transferred = 0;
    r.graph_seq = graph_seq;
    r.run_index = run_index;
    r.qd_after = saturate_i32_to_i16(qd_after);
    r.compute_active_after = saturate_i32_to_i16(compute_active_after);
    return r;
}

FixedRecord make_base_record(uint8_t event_kind, uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                              uint8_t layer, uint8_t tensor_part, uint8_t expert, uint32_t generation_low32,
                              uint64_t & out_seq) {
    int64_t tick; int32_t qd_after, compute_active_after;
    uint64_t seq = issue_seq_and_tick(tick, qd_after, compute_active_after);
    out_seq = seq;
    return make_base_record_ex(event_kind, run_index, graph_seq, phase, layer, tensor_part, expert,
                                generation_low32, seq, tick, qd_after, compute_active_after);
}

// "닫는" 관측점 공통 — correlation_id = 여는 쪽이 반환했던 값(그대로 echo). 문맥
// (layer/tensor_part/expert/generation)은 여는 쪽 레코드에 이미 있으므로 correlation_id로
// 조인하면 되고, 여기서는 중복 전달을 요구하지 않는다(호출부 부담 최소화 — 태스크
// 지시 "전송 바이트·status 병기"처럼 닫는 쪽은 결과만 싣는다는 원칙). _ex/얇은 래퍼 관계는
// 위 make_base_record_ex와 동일 사유(배선 4a).
// ★배선 R2(사양 §2): layer/tensor_part/expert는 계속 NA로 둔다 — 그 문맥은 correlation_id로
// 여는 레코드에 조인해 얻는 것이 설계이며, 여기서 중복 기록하면 두 소스가 어긋날 수 있다.
// run_index/graph_seq/phase는 더 이상 0/UNKNOWN으로 지우지 않고 ggml_moe_trace_current_
// graph_context()가 읽는 것과 동일한 전역 스냅샷(g_graph_context)에서 읽어 채운다 — 닫는
// 이벤트가 어느 run·graph에서 발생했는지 판독기가 귀속할 수 있게 한다(외부 심의 OV5 차단
// 5/A절). qd_after/compute_active_after는 make_base_record_ex와 동일 사유(위 주석 참조)로
// 호출측이 같은 임계구역에서 채취한 값을 인자로 받는다.
FixedRecord make_close_record_ex(uint8_t event_kind, uint64_t opening_id, uint64_t seq, int64_t tick,
                                  int32_t qd_after, int32_t compute_active_after) {
    FixedRecord r{};
    r.type = MOE_TRACE_RT_FIXED;
    r.event_kind = event_kind;
    r.tensor_part = MOE_TRACE_NA_U8;
    r.layer = MOE_TRACE_NA_U8;
    r.expert = MOE_TRACE_NA_U8;
    r.flags = 0;
    r.status = MOE_TRACE_IO_STATUS_OK;
    r.qpc_tick = tick;
    r.event_seq = seq;
    r.correlation_id = opening_id;
    r.thread_id = (uint32_t)GetCurrentThreadId();
    r.generation_low32 = 0;
    r.bytes_requested = 0;
    r.bytes_transferred = 0;
    uint16_t ctx_run_index; uint32_t ctx_graph_seq; uint8_t ctx_phase;
    unpack_graph_context(g_graph_context.load(std::memory_order_acquire), ctx_run_index, ctx_graph_seq, ctx_phase);
    r.run_index = ctx_run_index;
    r.graph_seq = ctx_graph_seq;
    r.phase = ctx_phase;
    r.qd_after = saturate_i32_to_i16(qd_after);
    r.compute_active_after = saturate_i32_to_i16(compute_active_after);
    return r;
}

FixedRecord make_close_record(uint8_t event_kind, uint64_t opening_id) {
    int64_t tick; int32_t qd_after, compute_active_after;
    uint64_t seq = issue_seq_and_tick(tick, qd_after, compute_active_after);
    return make_close_record_ex(event_kind, opening_id, seq, tick, qd_after, compute_active_after);
}

} // anonymous namespace

// ===========================================================================
// §9. 공개 API 구현
// ===========================================================================

extern "C" {

bool ggml_moe_trace_enabled(void) { return trace_enabled_cached_check(); }

int32_t ggml_moe_trace_gauge_compute_active(void) { return g_compute_active.load(std::memory_order_relaxed); }
int32_t ggml_moe_trace_gauge_app_qd(void)         { return g_app_qd.load(std::memory_order_relaxed); }
int32_t ggml_moe_trace_gauge_wait_active(void)    { return g_wait_active.load(std::memory_order_relaxed); }
bool    ggml_moe_trace_window_active(void)        { return g_window_mode.load(std::memory_order_relaxed) != 0; }

int ggml_moe_trace_init(void) {
    if (!trace_enabled_cached_check()) return 0; // off — 무동작 성공
    std::call_once(g_init_once, do_init_once);
    return g_ready.load(std::memory_order_acquire) ? 0 : -1;
}

void ggml_moe_trace_flush(const char * path) {
    if (!trace_enabled_cached_check()) return;
    if (!g_ready.load(std::memory_order_acquire) || !path) return;

    uint64_t wp = g_write_pos.load(std::memory_order_acquire);
    // ★배선 R1 정정(메인 검수 26-07-20 — R1 사양 §6의 누락분): flush 범위의 하한을
    // g_flush_base_pos로 올린다. 링을 리셋하지 않으므로(현행 유지) 하한이 없으면 run N의
    // flush 파일에 run 1..N-1의 레코드가 전부 섞여 들어간다 — 파일이 run별로 분리되지 않으면
    // 판독기의 표본 창 occurrence가 run 경계를 넘어 오염되고(외부 심의 OV5 차단 2가 지적한
    // "occurrence가 run별이 아니라 세션 전역"의 뿌리 중 하나), 세션당 flush 바이트도 run마다
    // 누적 증가한다. 하한이 이미 링 밖으로 밀려난 경우에는 물리적으로 남아 있는 가장 오래된
    // 레코드부터 쓰되, 그 상황은 정의상 덮어쓰기이므로 overwrite_count > 0으로 판독기가
    // 무효화한다(사양 §4).
    uint64_t base_logical    = g_flush_base_pos.load(std::memory_order_relaxed);
    uint64_t oldest_in_ring  = wp > RING_CAPACITY_RECORDS ? (wp - RING_CAPACITY_RECORDS) : 0;
    uint64_t start_logical   = base_logical > oldest_in_ring ? base_logical : oldest_in_ring;
    uint64_t n_ring_records  = wp > start_logical ? (wp - start_logical) : 0;

    // 배선 R1(사양 §6) — prefill 세그먼트: "마지막으로 완료된" prefill graph 버퍼(있으면)의
    // 앞 n개 레코드.
    int32_t  prefill_buf = g_prefill_last_buf.load(std::memory_order_relaxed);
    uint64_t n_prefill_records = (prefill_buf >= 0)
        ? g_prefill_last_count.load(std::memory_order_relaxed) : 0;

    std::wstring wpath = local_utf8_to_wide(path);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        GGML_ABORT("[moe-trace] FATAL: flush CreateFileW failed err=%lu path=%s", (unsigned long)GetLastError(), path);
    }

    TraceFileHeader hdr{};
    std::memcpy(hdr.magic, "MOETRC1", 8);
    hdr.layout_version = 4; // 배선 S12(v4 slot identity — SPEC_G32C §8 U6'): O1/O2 _pad에 slot_index 탑재
    hdr.record_size = RECORD_SIZE;
    hdr.qpc_frequency = (uint64_t)g_qpc_freq;
    hdr.endianness_marker = 0x01020304u;
    hdr.record_count = n_ring_records + n_prefill_records;
    hdr.total_write_pos = wp;
    hdr.prefill_seg_record_count = (uint32_t)n_prefill_records; // PREFILL_WIN_CAPACITY_RECORDS로
                                                                  // 클램프돼 있어 uint32 범위 안전
    hdr.overwrite_count = saturate_u64_to_u32(g_overwrite_count.load(std::memory_order_relaxed));
    hdr.prefill_overflow_count = saturate_u64_to_u32(g_prefill_overflow_count.load(std::memory_order_relaxed));
    hdr.anchor_negative_count = saturate_u64_to_u32(g_anchor_negative_count.load(std::memory_order_relaxed));
    hdr.decode_window_begin = (uint16_t)DECODE_WIN_BEGIN;
    hdr.decode_window_end   = (uint16_t)DECODE_WIN_END;

    DWORD written = 0;
    if (!WriteFile(h, &hdr, (DWORD)sizeof(hdr), &written, nullptr) || written != sizeof(hdr)) {
        CloseHandle(h);
        GGML_ABORT("[moe-trace] FATAL: flush header WriteFile failed err=%lu", (unsigned long)GetLastError());
    }

    if (n_ring_records > 0) {
        uint64_t first_slot = start_logical & RING_MASK;
        uint64_t contig1 = RING_CAPACITY_RECORDS - first_slot;
        if (contig1 > n_ring_records) contig1 = n_ring_records;
        DWORD bytes1 = (DWORD)(contig1 * RECORD_SIZE);
        if (!WriteFile(h, g_ring_base + first_slot * RECORD_SIZE, bytes1, &written, nullptr) || written != bytes1) {
            CloseHandle(h);
            GGML_ABORT("[moe-trace] FATAL: flush segment 1 WriteFile failed err=%lu", (unsigned long)GetLastError());
        }
        uint64_t remain = n_ring_records - contig1;
        if (remain > 0) {
            DWORD bytes2 = (DWORD)(remain * RECORD_SIZE);
            if (!WriteFile(h, g_ring_base, bytes2, &written, nullptr) || written != bytes2) {
                CloseHandle(h);
                GGML_ABORT("[moe-trace] FATAL: flush segment 2 WriteFile failed err=%lu", (unsigned long)GetLastError());
            }
        }
    }

    if (n_prefill_records > 0) {
        DWORD bytesp = (DWORD)(n_prefill_records * RECORD_SIZE);
        if (!WriteFile(h, g_prefill_base[prefill_buf], bytesp, &written, nullptr) || written != bytesp) {
            CloseHandle(h);
            GGML_ABORT("[moe-trace] FATAL: flush prefill segment WriteFile failed err=%lu", (unsigned long)GetLastError());
        }
    }

    CloseHandle(h);

    // 배선 R1(사양 §4/§6) — flush 후 base_pos 갱신(다음 덮어쓰기 계수 기준점) + prefill 버퍼
    // 상태 리셋(같은 완료분을 다음 flush에서 중복 재기록하지 않도록). 링 자체는 리셋하지
    // 않는다(현행 동작 유지).
    g_flush_base_pos.store(wp, std::memory_order_relaxed);
    g_prefill_last_buf.store(-1, std::memory_order_relaxed);
    g_prefill_last_count.store(0, std::memory_order_relaxed);
}

void ggml_moe_trace_finalize(void) {
    // 멱등·무상태(seal()/모듈 metrics와 달리 "1회만 요약 기록" 개념이 없다 — flush가 이미
    // 명시 호출 계약이므로 여기서는 별도로 할 것이 없다. 향후 run_end 훅이 붙을 자리만
    // 예약해 둔다).
}

uint64_t ggml_moe_trace_graph_begin(uint16_t run_index, uint32_t graph_seq, uint8_t phase, uint32_t n_tokens) {
    if (!trace_active_and_ready()) return 0;
    // worker 문맥 전파(사양 §4-1) — graph_compute() 호출(워커 기상) 이전에 반드시 먼저
    // publish한다. 아래 write_record_bytes보다 앞선 순서 자체는 필수 아니지만(둘 다 이
    // 호출이 반환하기 전, 즉 graph_compute 이전에 끝남), 스냅샷을 먼저 publish해 두면 이
    // 함수 도중 어떤 이유로도 관측점이 문맥 없이 fire하는 창을 만들지 않는다.
    g_graph_context.store(pack_graph_context(run_index, graph_seq, phase), std::memory_order_release);

    // 배선 R1(사양 §3) — 창 게이팅 모드 "결정". g_run_decode_graph_idx/g_prefill_cur의 현재값
    // 읽기는 이 함수를 호출하는 단일 스레드 전용 상태라 락 불요(g_graph_context와 동일 근거 —
    // 위 주석 참조). 실제 g_window_mode/g_prefill_cur "발행"(워커가 읽는 atomic)은 아래
    // issue_seq_tick_wall_graph_begin() 안, 기존 seq_lock 임계구역에서 한다.
    int32_t window_mode = 0;
    int32_t prefill_next_buf = -1;
    if (phase == MOE_TRACE_PHASE_DECODE) {
        if (g_run_decode_graph_idx >= DECODE_WIN_BEGIN && g_run_decode_graph_idx < DECODE_WIN_END) {
            window_mode = 1;
        }
        g_run_decode_graph_idx++;
    } else if (phase == MOE_TRACE_PHASE_PREFILL) {
        window_mode = 2;
        prefill_next_buf = 1 - g_prefill_cur.load(std::memory_order_relaxed);
    }

    // 배선 4a: wall-state 창 열기(§5-1) — 이 seq/tick으로 GRAPH_BEGIN 레코드 자신도 채운다.
    int64_t tick; int32_t qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_wall_graph_begin(phase, window_mode, prefill_next_buf, tick,
                                                     qd_after, compute_active_after);
    FixedRecord r = make_base_record_ex(MOE_TRACE_EV_GRAPH_BEGIN, run_index, graph_seq, phase,
                                         MOE_TRACE_NA_U8, MOE_TRACE_NA_U8, MOE_TRACE_NA_U8, 0, seq, tick,
                                         qd_after, compute_active_after);
    // 배선 R1(사양 §5) — GRAPH_BEGIN은 bytes_requested 예외: ubatch.n_tokens를 싣는다(헤더
    // wire format 섹션 필드 주석 참조).
    r.bytes_requested = n_tokens;
    // 배선 R7a(사양 §6-2 P2 등식화) — GRAPH_BEGIN은 bytes_transferred 예외: chunk_call_count
    // 세션 누적 스냅샷(하위 32비트, 헤더 필드 주석 참조)을 싣는다. 이 시점엔 이 graph의
    // graph_compute()가 아직 호출되지 않았으므로(호출측 llama-context.cpp가 이 함수 반환 뒤에만
    // graph_compute를 부른다 — 워커는 그 안에서만 돎) 이 graph의 chunk 호출은 아직 0건이다.
    r.bytes_transferred = (uint32_t)g_chunk_call_count.load(std::memory_order_relaxed);
    write_record_bytes(&r, sizeof(r)); // GRAPH_BEGIN은 저빈도군 — 항상 링에 씀(게이팅 없음)
    return seq;
}

void ggml_moe_trace_graph_end(uint64_t graph_op_id, uint16_t run_index, uint32_t graph_seq, uint8_t phase) {
    if (!trace_active_and_ready()) return;
    // 배선 4a: wall-state 창 닫기 + 항등식 검산(§2-2 6항·§6-2 P4) — 이 seq/tick으로 GRAPH_END
    // 레코드 자신도 채운다.
    int64_t tick; int32_t qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_wall_graph_end(phase, tick, qd_after, compute_active_after);
    FixedRecord r = make_close_record_ex(MOE_TRACE_EV_GRAPH_END, graph_op_id, seq, tick,
                                          qd_after, compute_active_after);
    r.run_index = run_index;
    r.graph_seq = graph_seq;
    r.phase = phase;
    // 배선 R7a(사양 §6-2 P2 등식화) — GRAPH_END는 bytes_requested 예외: chunk_call_count 세션
    // 누적 스냅샷(하위 32비트, 헤더 필드 주석 참조)을 싣는다. graph_end는 항상 호출측
    // (llama-context.cpp)이 graph_compute() 반환을 받은 뒤에만 부르므로(그 안에서만 워커가 돎)
    // 이 시점엔 이 graph의 chunk 호출이 전부 끝나 있다.
    r.bytes_requested = (uint32_t)g_chunk_call_count.load(std::memory_order_relaxed);
    write_record_bytes(&r, sizeof(r));
    // 활성 graph 없음 표시(방어적 — 관측점은 구조상 graph 구간 밖에서 fire할 수 없지만,
    // 혹시 fire한다면 phase=UNKNOWN으로 그 이상이 트레이스에 드러나게 한다). run_index/
    // graph_seq는 직전 graph의 값을 남겨 둔다(디버깅 시 "마지막으로 어느 graph였는지"
    // 힌트가 되도록 — phase만으로 이미 "활성 아님"이 판별된다).
    g_graph_context.store(pack_graph_context(run_index, graph_seq, MOE_TRACE_PHASE_UNKNOWN), std::memory_order_release);
}

uint64_t ggml_moe_trace_run_begin(uint16_t run_index) {
    if (!trace_active_and_ready()) return 0;
    // 배선 R1(사양 §3) — decode 표본 창 인덱스를 이번 run의 시작으로 리셋(단일 호출 스레드
    // 전용 상태 — 위 g_run_decode_graph_idx 선언부 주석 참조).
    g_run_decode_graph_idx = 0;
    // 배선 4a: 이번 run의 AGG delta 산출용 baseline 스냅샷(§6-5) — 단일-slot 벤치라 run 간
    // 겹침이 없으므로(직전 run의 release()가 완전히 끝난 뒤에만 다음 run_begin이 온다 —
    // server-context.cpp 상태기계 순서) g_seq_lock 없이 읽어도 안전하다(다른 스레드의
    // wall-state 쓰기는 이미 직전 run의 graph_end에서 모두 종료됨).
    for (int ph = 0; ph < 2; ph++) {
        WallStateAccum & acc = g_wall_accum[ph];
        g_wall_run_base_c[ph]        = acc.c_ticks.load(std::memory_order_relaxed);
        g_wall_run_base_lcrit[ph]    = acc.lcrit_ticks.load(std::memory_order_relaxed);
        g_wall_run_base_other[ph]    = acc.other_ticks.load(std::memory_order_relaxed);
        g_wall_run_base_llock[ph]    = acc.l_lock_ticks.load(std::memory_order_relaxed);
        g_wall_run_base_ldata[ph]    = acc.l_data_ticks.load(std::memory_order_relaxed);
        g_wall_run_base_mismatch[ph] = acc.mismatch_count.load(std::memory_order_relaxed);
        // 배선 R9 — 창 누산기 baseline도 같은 시점에 스냅샷(사양 §2 "run_begin에서 창
        // 누산기의 baseline 스냅샷도 함께").
        WallStateAccum & win_acc = g_wall_accum_window[ph];
        g_wall_win_run_base_c[ph]        = win_acc.c_ticks.load(std::memory_order_relaxed);
        g_wall_win_run_base_lcrit[ph]    = win_acc.lcrit_ticks.load(std::memory_order_relaxed);
        g_wall_win_run_base_other[ph]    = win_acc.other_ticks.load(std::memory_order_relaxed);
        g_wall_win_run_base_llock[ph]    = win_acc.l_lock_ticks.load(std::memory_order_relaxed);
        g_wall_win_run_base_ldata[ph]    = win_acc.l_data_ticks.load(std::memory_order_relaxed);
        g_wall_win_run_base_mismatch[ph] = win_acc.mismatch_count.load(std::memory_order_relaxed);
    }
    // 배선 4d: 이번 run의 read_bytes/hit/miss IOSTATE_AGG delta 산출용 baseline(§6-5) — 위
    // wall-state 스냅샷과 동일 사유로 락 불요.
    for (int ph = 0; ph < 2; ph++) {
        uint64_t rb = 0, h = 0, m = 0;
        // R3 배선(외부 심의 OV5 차단 3 — 사양 §3): IOSTATE_AGG는 이제 graph 문맥 기반 정확
        // 카운터를 조회한다(값의 출처만 바뀜 — wire format은 그대로, ggml-moe-trace.h 레코드
        // 정의부 주석 참조).
        ggml_moe_direct_phase_read_bytes_hit_miss_exact(ph, &rb, &h, &m);
        g_io_run_base_read_bytes[ph] = rb;
        g_io_run_base_hit[ph]        = h;
        g_io_run_base_miss[ph]       = m;
        // 배선 S3a — 창-scope baseline도 같은 시점에 스냅샷(사양 "run_begin에서 창 카운터의
        // baseline 스냅샷도 함께").
        uint64_t win_rb = 0, win_h = 0, win_m = 0;
        ggml_moe_direct_phase_read_bytes_hit_miss_window(ph, &win_rb, &win_h, &win_m);
        g_io_win_run_base_read_bytes[ph] = win_rb;
        g_io_win_run_base_hit[ph]        = win_h;
        g_io_win_run_base_miss[ph]       = win_m;
    }
    uint64_t seq;
    FixedRecord r = make_base_record(MOE_TRACE_EV_RUN_BEGIN, run_index, 0, MOE_TRACE_PHASE_UNKNOWN,
                                      MOE_TRACE_NA_U8, MOE_TRACE_NA_U8, MOE_TRACE_NA_U8, 0, seq);
    write_record_bytes(&r, sizeof(r));
    return seq;
}

void ggml_moe_trace_run_end(uint64_t run_op_id, uint16_t run_index, uint32_t predicted_n, uint32_t prompt_n) {
    if (!trace_active_and_ready()) return;
    FixedRecord r = make_close_record(MOE_TRACE_EV_RUN_END, run_op_id);
    r.run_index = run_index;
    // 배선 2단계: 서버 권위값(사양 §4-3 대조용) — bytes_requested/bytes_transferred 재사용
    // (헤더의 FixedRecord 필드 주석 참조. 이 이벤트에서만 의미를 갖는다).
    r.bytes_requested   = prompt_n;
    r.bytes_transferred = predicted_n;
    write_record_bytes(&r, sizeof(r));

    // 배선 4a: phase당 1개씩 WALLSTATE_AGG 레코드 발행(사양 §2-2/§6-5 "run별 누산 결과를
    // RUN_END 레코드 또는 전용 AGG 레코드로") — run_begin이 떠 둔 baseline과의 차분이 이
    // run의 phase별 기여분이다.
    for (int ph = 0; ph < 2; ph++) {
        WallStateAccum & acc = g_wall_accum[ph];
        WallStateAggRecord agg{};
        agg.type      = MOE_TRACE_RT_WALLSTATE_AGG;
        agg.phase     = (uint8_t)ph;
        agg.run_index = run_index;
        int64_t tick; int32_t unused_qd, unused_ca;
        uint64_t aseq = issue_seq_and_tick(tick, unused_qd, unused_ca); // WallStateAggRecord에는
                                                   // qd_after/compute_active_after 필드가 없다
                                                   // (app_qd/compute_active 외 게이지를 건드리지
                                                   // 않으므로 일반 issue_seq_and_tick으로 충분).
        agg.qpc_tick  = (uint64_t)tick;
        agg.event_seq = aseq;
        agg.c_ticks     = acc.c_ticks.load(std::memory_order_relaxed)     - g_wall_run_base_c[ph];
        agg.lcrit_ticks = acc.lcrit_ticks.load(std::memory_order_relaxed) - g_wall_run_base_lcrit[ph];
        agg.other_ticks = acc.other_ticks.load(std::memory_order_relaxed) - g_wall_run_base_other[ph];
        agg.l_lock_ticks = acc.l_lock_ticks.load(std::memory_order_relaxed) - g_wall_run_base_llock[ph];
        agg.l_data_ticks = acc.l_data_ticks.load(std::memory_order_relaxed) - g_wall_run_base_ldata[ph];
        agg.wall_state_mismatch_count =
            (uint32_t)(acc.mismatch_count.load(std::memory_order_relaxed) - g_wall_run_base_mismatch[ph]);
        write_record_bytes(&agg, sizeof(agg));

        // 배선 R9(사양 §2 "run_end에서 기존 WALLSTATE_AGG를 발행하는 바로 그 루프에서 phase당
        // 1개씩 창-scope 버전도 발행") — 레이아웃은 WallStateAggRecord를 그대로 재사용(type
        // 필드만 다름 — 판독기가 같은 파서를 쓸 수 있게, 헤더 struct 주석 참조). 값은
        // run_begin 시점 창 누산기 baseline과의 차분(기존 AGG와 동일한 baseline 차분 방식).
        WallStateAccum & win_acc = g_wall_accum_window[ph];
        WallStateAggRecord win_agg{};
        win_agg.type      = MOE_TRACE_RT_WALLSTATE_WIN_AGG;
        win_agg.phase     = (uint8_t)ph;
        win_agg.run_index = run_index;
        int64_t win_tick; int32_t win_unused_qd, win_unused_ca;
        uint64_t win_aseq = issue_seq_and_tick(win_tick, win_unused_qd, win_unused_ca); // WallStateAggRecord
                                                   // 에는 qd_after/compute_active_after 필드가
                                                   // 없다(위 WALLSTATE_AGG와 동일 사유).
        win_agg.qpc_tick  = (uint64_t)win_tick;
        win_agg.event_seq = win_aseq;
        win_agg.c_ticks     = win_acc.c_ticks.load(std::memory_order_relaxed)     - g_wall_win_run_base_c[ph];
        win_agg.lcrit_ticks = win_acc.lcrit_ticks.load(std::memory_order_relaxed) - g_wall_win_run_base_lcrit[ph];
        win_agg.other_ticks = win_acc.other_ticks.load(std::memory_order_relaxed) - g_wall_win_run_base_other[ph];
        win_agg.l_lock_ticks = win_acc.l_lock_ticks.load(std::memory_order_relaxed) - g_wall_win_run_base_llock[ph];
        win_agg.l_data_ticks = win_acc.l_data_ticks.load(std::memory_order_relaxed) - g_wall_win_run_base_ldata[ph];
        win_agg.wall_state_mismatch_count =
            (uint32_t)(win_acc.mismatch_count.load(std::memory_order_relaxed) - g_wall_win_run_base_mismatch[ph]);
        write_record_bytes(&win_agg, sizeof(win_agg));

        // 배선 4d: read_bytes/hit/miss run별 델타(§6-5) — WALLSTATE_AGG와 동일 loop에서
        // phase당 1개씩 IOSTATE_AGG도 함께 발행한다(원본 카운터는 ggml-moe-direct.cpp의
        // phase_exact[] — ggml_moe_direct_phase_read_bytes_hit_miss_exact()로 조회, R3 배선).
        uint64_t rb_now = 0, hit_now = 0, miss_now = 0;
        // R3 배선(사양 §3) — 위 run_begin의 baseline 조회와 동일하게 정확 카운터로 전환.
        ggml_moe_direct_phase_read_bytes_hit_miss_exact(ph, &rb_now, &hit_now, &miss_now);
        IostateAggRecord io_agg{};
        io_agg.type      = MOE_TRACE_RT_IOSTATE_AGG;
        io_agg.phase     = (uint8_t)ph;
        io_agg.run_index = run_index;
        int64_t io_tick; int32_t io_unused_qd, io_unused_ca;
        uint64_t io_seq = issue_seq_and_tick(io_tick, io_unused_qd, io_unused_ca); // IostateAggRecord
                                                        // 에는 qd_after/compute_active_after 필드가
                                                        // 없다 — 일반 경로로 충분.
        io_agg.qpc_tick        = (uint64_t)io_tick;
        io_agg.event_seq       = io_seq;
        io_agg.read_bytes_delta = (int64_t)rb_now   - (int64_t)g_io_run_base_read_bytes[ph];
        io_agg.hit_delta        = (int64_t)hit_now  - (int64_t)g_io_run_base_hit[ph];
        io_agg.miss_delta       = (int64_t)miss_now - (int64_t)g_io_run_base_miss[ph];
        write_record_bytes(&io_agg, sizeof(io_agg));

        // 배선 S3a(외부 심의 OV6 E절 차단 1) — 위와 동일 loop에서 phase당 1개씩 창-scope
        // 버전도 발행한다. 레이아웃은 IostateAggRecord를 그대로 재사용(type 필드만 다름 —
        // 헤더 struct 주석 참조). 값은 run_begin 시점 창-scope baseline과의 차분(위 IOSTATE_AGG와
        // 동일한 baseline 차분 방식).
        uint64_t win_rb_now = 0, win_hit_now = 0, win_miss_now = 0;
        ggml_moe_direct_phase_read_bytes_hit_miss_window(ph, &win_rb_now, &win_hit_now, &win_miss_now);
        IostateAggRecord io_win_agg{};
        io_win_agg.type      = MOE_TRACE_RT_IOSTATE_WIN_AGG;
        io_win_agg.phase     = (uint8_t)ph;
        io_win_agg.run_index = run_index;
        int64_t io_win_tick; int32_t io_win_unused_qd, io_win_unused_ca;
        uint64_t io_win_seq = issue_seq_and_tick(io_win_tick, io_win_unused_qd, io_win_unused_ca); // IostateAggRecord
                                                        // 에는 qd_after/compute_active_after 필드가
                                                        // 없다(위 IOSTATE_AGG와 동일 사유).
        io_win_agg.qpc_tick        = (uint64_t)io_win_tick;
        io_win_agg.event_seq       = io_win_seq;
        io_win_agg.read_bytes_delta = (int64_t)win_rb_now   - (int64_t)g_io_win_run_base_read_bytes[ph];
        io_win_agg.hit_delta        = (int64_t)win_hit_now  - (int64_t)g_io_win_run_base_hit[ph];
        io_win_agg.miss_delta       = (int64_t)win_miss_now - (int64_t)g_io_win_run_base_miss[ph];
        write_record_bytes(&io_win_agg, sizeof(io_win_agg));
    }
}

uint16_t ggml_moe_trace_next_run_index(void) {
    if (!trace_active_and_ready()) return 0;
    uint32_t idx = g_run_index_next.fetch_add(1, std::memory_order_relaxed);
    g_current_run_index.store(idx, std::memory_order_release);
    return (uint16_t)idx;
}

uint16_t ggml_moe_trace_current_run_index(void) {
    return (uint16_t)g_current_run_index.load(std::memory_order_acquire);
}

void ggml_moe_trace_current_graph_context(uint16_t * out_run_index, uint32_t * out_graph_seq, uint8_t * out_phase) {
    uint64_t v = g_graph_context.load(std::memory_order_acquire);
    uint16_t ri; uint32_t gs; uint8_t ph;
    unpack_graph_context(v, ri, gs, ph);
    if (out_run_index) *out_run_index = ri;
    if (out_graph_seq) *out_graph_seq = gs;
    if (out_phase)     *out_phase     = ph;
}

// 배선 S12(v4 slot identity — SPEC_G32C §8 U6'): slot_index를 _pad[0..3]에 리틀엔디언 uint32로
// 인코딩. _pad[4..5]는 FixedRecord r{} zero-init로 이미 0(reserved) — 손대지 않는다. COMPUTE_EXIT
// anchor delta와 event_kind가 배타적이라(O1/O2는 절대 COMPUTE_EXIT이 아님) _pad 겹침 충돌 없음.
static inline void moe_trace_encode_slot_index(FixedRecord & r, uint32_t slot_index) {
    r._pad[0] = (uint8_t)(slot_index & 0xFFu);
    r._pad[1] = (uint8_t)((slot_index >> 8) & 0xFFu);
    r._pad[2] = (uint8_t)((slot_index >> 16) & 0xFFu);
    r._pad[3] = (uint8_t)((slot_index >> 24) & 0xFFu);
}

uint64_t ggml_moe_trace_io_submit(uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                   uint8_t layer, uint8_t tensor_part, uint8_t expert,
                                   uint32_t generation_low32, uint32_t bytes_requested,
                                   uint32_t slot_index) {
    if (!trace_active_and_ready()) return 0;
    // 배선 R2(사양 §1-a): app_qd 증감을 seq lock 임계구역 안으로 이동(WallGaugeOp::QdInc) —
    // 이 이벤트의 event_seq/qpc_tick·qd_after·compute_active_after가 모두 같은 임계구역에서
    // 확정된 한 벌이 되도록 한다(이전에는 fetch_add가 이 구역 밖에서 먼저 일어났다).
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::QdInc, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("app_qd went negative (io_submit)");
    FixedRecord r = make_base_record_ex(MOE_TRACE_EV_IO_SUBMIT, run_index, graph_seq, phase,
                                         layer, tensor_part, expert, generation_low32, seq, tick,
                                         qd_after, compute_active_after);
    r.bytes_requested = bytes_requested;
    moe_trace_encode_slot_index(r, slot_index); // 배선 S12(v4)
    write_record_bytes(&r, sizeof(r));
    return seq; // op_id
}

void ggml_moe_trace_io_complete(uint64_t op_id, uint8_t status, uint32_t bytes_transferred,
                                bool was_pending, uint32_t slot_index) {
    if (!trace_active_and_ready()) return;
    // 배선 R2(사양 §1-a): 위 io_submit과 동일 사유(WallGaugeOp::QdDec).
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::QdDec, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("app_qd went negative (io_complete)");
    FixedRecord r = make_close_record_ex(MOE_TRACE_EV_IO_COMPLETE, op_id, seq, tick,
                                          qd_after, compute_active_after);
    r.status = status;
    r.bytes_transferred = bytes_transferred;
    r.flags = was_pending ? 0x1 : 0x0;
    moe_trace_encode_slot_index(r, slot_index); // 배선 S12(v4) — O2도 O1과 동일 slot_index(일치 검사용)
    write_record_bytes(&r, sizeof(r));
}

uint64_t ggml_moe_trace_lock_wait_begin(uint16_t run_index, uint32_t graph_seq, uint8_t phase) {
    if (!trace_active_and_ready()) return 0;
    // 배선 4a: wait_active(+lock_wait_count) 게이지 갱신 + wall-state 전이 귀속을 이 레코드
    // 자신의 QPC/seq 발급과 같은 임계구역에서 수행(사양 §2-2 "코드로 확정" — §5-1 주석 참조).
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::LockWaitInc, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("wait_active went negative (lock_wait_begin)");
    FixedRecord r = make_base_record_ex(MOE_TRACE_EV_LOCK_WAIT_BEGIN, run_index, graph_seq, phase,
                                         MOE_TRACE_NA_U8, MOE_TRACE_NA_U8, MOE_TRACE_NA_U8, 0, seq, tick,
                                         qd_after, compute_active_after);
    write_record_bytes(&r, sizeof(r));
    return seq; // lock_episode_id
}

void ggml_moe_trace_lock_acquired(uint64_t lock_episode_id) {
    if (!trace_active_and_ready()) return;
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::LockWaitDec, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("wait_active went negative (lock_acquired)");
    FixedRecord r = make_close_record_ex(MOE_TRACE_EV_LOCK_ACQUIRED, lock_episode_id, seq, tick,
                                          qd_after, compute_active_after);
    write_record_bytes(&r, sizeof(r));
}

uint64_t ggml_moe_trace_data_wait_begin(uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                         uint8_t layer, uint8_t tensor_part, uint8_t expert,
                                         uint32_t generation_low32) {
    if (!trace_active_and_ready()) return 0;
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::DataWaitInc, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("wait_active went negative (data_wait_begin)");
    FixedRecord r = make_base_record_ex(MOE_TRACE_EV_DATA_WAIT_BEGIN, run_index, graph_seq, phase,
                                         layer, tensor_part, expert, generation_low32, seq, tick,
                                         qd_after, compute_active_after);
    write_record_bytes(&r, sizeof(r));
    return seq; // wait_episode_id
}

void ggml_moe_trace_data_wait_end(uint64_t wait_episode_id) {
    if (!trace_active_and_ready()) return;
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::DataWaitDec, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("wait_active went negative (data_wait_end)");
    FixedRecord r = make_close_record_ex(MOE_TRACE_EV_DATA_WAIT_END, wait_episode_id, seq, tick,
                                          qd_after, compute_active_after);
    write_record_bytes(&r, sizeof(r));
}

void ggml_moe_trace_ready_publish(uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                   uint8_t layer, uint8_t tensor_part, uint8_t expert,
                                   uint32_t generation_low32) {
    if (!trace_active_and_ready()) return;
    uint64_t seq;
    FixedRecord r = make_base_record(MOE_TRACE_EV_READY_PUBLISH, run_index, graph_seq, phase,
                                      layer, tensor_part, expert, generation_low32, seq);
    write_record_bytes(&r, sizeof(r));
}

uint64_t ggml_moe_trace_compute_enter(uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                       uint8_t layer, uint8_t tensor_part, int32_t nth) {
    if (!trace_active_and_ready()) return 0;
    // 배선 4a: useful_compute_active 갱신을 이 레코드의 QPC/seq 발급과 같은 임계구역에서
    // 수행(사양 §2-2 "코드로 확정" — 기존에는 fetch_add가 이 구역 밖에서 먼저 일어났다).
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::ComputeInc, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("useful_compute_active went negative (compute_enter)");
    if (nth > 0 && after > nth) fatal_trace_abort("useful_compute_active > nth (compute_enter upper bound violated)");
    tls_last_compute_enter_tick = tick; // 배선 S6 — 짝이 될 compute_exit의 first_work_delta 계산용
    FixedRecord r = make_base_record_ex(MOE_TRACE_EV_COMPUTE_ENTER, run_index, graph_seq, phase,
                                         layer, tensor_part, MOE_TRACE_NA_U8, 0, seq, tick,
                                         qd_after, compute_active_after);
    write_gated_record_bytes(&r, sizeof(r)); // 고빈도군(사양 §3) — 창 안에서만 발행
    return seq; // compute_op_seq
}

int64_t ggml_moe_trace_query_tick(void) {
    if (!trace_active_and_ready()) return 0;
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    return qpc.QuadPart;
}

void ggml_moe_trace_compute_exit(uint64_t compute_op_seq, uint32_t work_unit_count,
                                  int64_t first_work_tick, int64_t last_work_tick) {
    if (!trace_active_and_ready()) return;
    int64_t tick; int32_t after, qd_after, compute_active_after;
    uint64_t seq = issue_seq_tick_with_wall_transition(WallGaugeOp::ComputeDec, tick, after,
                                                         qd_after, compute_active_after);
    if (after < 0) fatal_trace_abort("useful_compute_active went negative (compute_exit)");
    FixedRecord r = make_close_record_ex(MOE_TRACE_EV_COMPUTE_EXIT, compute_op_seq, seq, tick,
                                          qd_after, compute_active_after);
    // 배선 R2(사양 §3) — 이 워커가 이 op에서 실제로 수행한 작업 단위 수(위 헤더 wire format
    // 섹션 bytes_transferred 필드 주석 참조).
    r.bytes_transferred = work_unit_count;
    // 배선 S6(사양 §1/§3) — anchor delta를 _pad[0..2]/_pad[3..5]에 3바이트 LE로 인코딩(위 헤더
    // FixedRecord._pad 주석 참조). work_unit_count==0(빈 작업 — 정상 호출점은 도달 안 함)이면
    // 두 값 모두 0(별도 sentinel 없음 — 헤더 주석에 명기된 그대로).
    uint32_t first_delta = 0, last_delta = 0;
    if (work_unit_count > 0) {
        bool first_neg = false, last_neg = false;
        first_delta = encode_anchor_delta(first_work_tick - tls_last_compute_enter_tick, first_neg);
        last_delta  = encode_anchor_delta(tick - last_work_tick, last_neg);
        // 배선 S7a — 역전 side를 레코드에 표시하고(관측된 표본의 현지화), 총계를 전역
        // 카운터에 남긴다(창 밖 표본까지 포괄). O6 하나당 1회만 센다 — 두 side가 모두
        // 역전이어도 "역전이 발생한 O6"는 하나다.
        if (first_neg) r.flags |= MOE_TRACE_FLAG_ANCHOR_FIRST_NEG;
        if (last_neg)  r.flags |= MOE_TRACE_FLAG_ANCHOR_LAST_NEG;
        if (first_neg || last_neg) {
            g_anchor_negative_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    r._pad[0] = (uint8_t)(first_delta & 0xFFu);
    r._pad[1] = (uint8_t)((first_delta >> 8) & 0xFFu);
    r._pad[2] = (uint8_t)((first_delta >> 16) & 0xFFu);
    r._pad[3] = (uint8_t)(last_delta & 0xFFu);
    r._pad[4] = (uint8_t)((last_delta >> 8) & 0xFFu);
    r._pad[5] = (uint8_t)((last_delta >> 16) & 0xFFu);
    write_gated_record_bytes(&r, sizeof(r)); // 고빈도군(사양 §3) — 창 안에서만 발행
}

void ggml_moe_trace_first_arith_consume(uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                         uint8_t layer, uint8_t tensor_part, uint8_t expert,
                                         uint32_t generation_low32) {
    if (!trace_active_and_ready()) return;
    uint64_t seq;
    FixedRecord r = make_base_record(MOE_TRACE_EV_FIRST_ARITH_CONSUME, run_index, graph_seq, phase,
                                      layer, tensor_part, expert, generation_low32, seq);
    write_gated_record_bytes(&r, sizeof(r)); // 고빈도군(사양 §3) — 창 안에서만 발행
}

bool ggml_moe_trace_first_arith_claim(uint32_t graph_seq, uint8_t layer, uint8_t tensor_part, uint8_t expert) {
    if (!trace_active_and_ready()) return false;
    if (tensor_part >= O8_PART_DIM) return false; // 방어적(계약 위반 — 정상 호출측은 항상 0..5)
    size_t idx = ((size_t)layer * O8_PART_DIM + (size_t)tensor_part) * O8_EXPERT_DIM + (size_t)expert;
    // 단일 exchange로 "이 graph_seq에서 최초"를 판정한다: 경합하는 여러 워커가 동시에
    // exchange를 실행해도 atomic RMW는 이 메모리 위치에 대해 전역 전순서(total order)를
    // 가지므로, 그 순서상 가장 먼저 실행된 하나만 "이전 값 != graph_seq"를 관측한다(그
    // 시점의 이전 값은 다른 그래프의 값이거나 O8_UNTOUCHED) — 그 뒤에 실행되는 나머지는
    // 이미 graph_seq로 갱신된 값을 읽어 false를 받는다(§5의 QPC 선형화 스핀락과 달리 여기는
    // CAS 루프조차 불필요 — exchange 자체가 이미 직렬화점).
    uint32_t prev = g_o8_last_graph_seq[idx].exchange(graph_seq, std::memory_order_relaxed);
    return prev != graph_seq;
}

void ggml_moe_trace_demand_digest(uint16_t run_index, uint32_t graph_seq, uint8_t phase,
                                   const struct ggml_moe_trace_demand_entry * entries, uint32_t n_entries) {
    if (!trace_active_and_ready()) return;
    if (n_entries > 0 && !entries) return;

    DemandHdrRecord hdr{};
    hdr.type = MOE_TRACE_RT_DEMAND_HDR;
    hdr.phase = phase;
    hdr.run_index = run_index;
    hdr.graph_seq = graph_seq;
    int64_t tick; int32_t unused_qd, unused_ca;
    uint64_t hseq = issue_seq_and_tick(tick, unused_qd, unused_ca); // DemandHdrRecord에는
                                                                     // qd_after/compute_active_after
                                                                     // 필드가 없다.
    hdr.qpc_tick = tick;
    hdr.event_seq = hseq;
    hdr.n_entries = n_entries;
    uint32_t n_cont = (n_entries + DEMAND_K - 1) / DEMAND_K;
    hdr.n_cont = n_cont;
    write_gated_record_bytes(&hdr, sizeof(hdr)); // 고빈도군(사양 §3) — 창 안에서만 발행

    uint32_t remaining = n_entries, off = 0;
    for (uint32_t c = 0; c < n_cont; c++) {
        DemandContRecord cont{};
        cont.type = MOE_TRACE_RT_DEMAND_CONT;
        cont.cont_index = (uint16_t)c;
        uint32_t take = remaining < DEMAND_K ? remaining : DEMAND_K;
        cont.entry_count = (uint8_t)take;
        cont.header_event_seq = hseq;
        for (uint32_t k = 0; k < take; k++) {
            cont.entries[k].layer = entries[off + k].layer;
            cont.entries[k].tensor_part = entries[off + k].tensor_part;
            cont.entries[k].expert = entries[off + k].expert;
        }
        write_gated_record_bytes(&cont, sizeof(cont)); // 고빈도군(사양 §3)
        off += take;
        remaining -= take;
    }
}

void ggml_moe_trace_snapshot(uint16_t run_index, uint32_t graph_seq,
                              const uint64_t * outstanding_op_ids, uint32_t n_ops,
                              const uint64_t * active_wait_episode_ids, uint32_t n_waits) {
    if (!trace_active_and_ready()) return;
    if ((n_ops > 0 && !outstanding_op_ids) || (n_waits > 0 && !active_wait_episode_ids)) return;

    SnapshotHdrRecord hdr{};
    hdr.type = MOE_TRACE_RT_SNAPSHOT_HDR;
    hdr.run_index = run_index;
    hdr.graph_seq = graph_seq;
    int64_t tick; int32_t unused_qd, unused_ca;
    uint64_t hseq = issue_seq_and_tick(tick, unused_qd, unused_ca); // SnapshotHdrRecord에는
                                                                     // qd_after/compute_active_after
                                                                     // 필드가 없다.
    hdr.qpc_tick = tick;
    hdr.event_seq = hseq;
    hdr.n_ops = n_ops;
    hdr.n_waits = n_waits;
    uint32_t n_cont_ops = (n_ops + SNAPSHOT_K - 1) / SNAPSHOT_K;
    uint32_t n_cont_waits = (n_waits + SNAPSHOT_K - 1) / SNAPSHOT_K;
    hdr.n_cont = n_cont_ops + n_cont_waits;
    write_record_bytes(&hdr, sizeof(hdr));

    auto emit_list = [&](const uint64_t * vals, uint32_t n, uint8_t list_kind) {
        uint32_t remaining = n, off = 0;
        uint32_t n_cont = (n + SNAPSHOT_K - 1) / SNAPSHOT_K;
        for (uint32_t c = 0; c < n_cont; c++) {
            SnapshotContRecord cont{};
            cont.type = MOE_TRACE_RT_SNAPSHOT_CONT;
            cont.list_kind = list_kind;
            cont.cont_index = (uint16_t)c;
            uint32_t take = remaining < SNAPSHOT_K ? remaining : SNAPSHOT_K;
            cont.entry_count = (uint8_t)take;
            cont.header_event_seq = hseq;
            for (uint32_t k = 0; k < take; k++) cont.values[k] = vals[off + k];
            write_record_bytes(&cont, sizeof(cont));
            off += take;
            remaining -= take;
        }
    };
    emit_list(outstanding_op_ids, n_ops, 0);
    emit_list(active_wait_episode_ids, n_waits, 1);
}

// 배선 4a — 세션 누적 wall-state 조회(ggml-moe-direct.cpp의 metrics JSONL·selftest용). phase가
// DECODE/PREFILL(0/1)이 아니면 전부 0으로 채운다(부작용 없음 — off 상태에서도 안전하게
// 호출 가능: 그 경우 g_wall_accum은 애초에 갱신된 적이 없으므로 자연히 전부 0).
void ggml_moe_trace_wall_state_totals(uint8_t phase,
                                       int64_t * out_c_ticks, int64_t * out_lcrit_ticks, int64_t * out_other_ticks,
                                       int64_t * out_l_lock_ticks, int64_t * out_l_data_ticks, uint64_t * out_mismatch_count) {
    int64_t c = 0, lcrit = 0, other = 0, l_lock = 0, l_data = 0;
    uint64_t mismatch = 0;
    if (phase < 2) {
        WallStateAccum & acc = g_wall_accum[phase];
        c      = acc.c_ticks.load(std::memory_order_relaxed);
        lcrit  = acc.lcrit_ticks.load(std::memory_order_relaxed);
        other  = acc.other_ticks.load(std::memory_order_relaxed);
        l_lock = acc.l_lock_ticks.load(std::memory_order_relaxed);
        l_data = acc.l_data_ticks.load(std::memory_order_relaxed);
        mismatch = acc.mismatch_count.load(std::memory_order_relaxed);
    }
    if (out_c_ticks) *out_c_ticks = c;
    if (out_lcrit_ticks) *out_lcrit_ticks = lcrit;
    if (out_other_ticks) *out_other_ticks = other;
    if (out_l_lock_ticks) *out_l_lock_ticks = l_lock;
    if (out_l_data_ticks) *out_l_data_ticks = l_data;
    if (out_mismatch_count) *out_mismatch_count = mismatch;
}

void ggml_moe_trace_note_acquire_call(int32_t n_ids) {
    // 배선 R2(사양 §4-d): 호출측(ggml_moe_direct_acquire) 게이트와 이중화 — 한쪽이 빠져도
    // MOE_P1_TRACE OFF에서의 바이트/실행 경로 불변(§3-5-7)이 깨지지 않도록.
    if (!trace_enabled_cached_check()) return;
    g_acquire_call_count.fetch_add(1, std::memory_order_relaxed);
    if (n_ids < 1) return;
    int idx = (n_ids >= 8) ? 7 : (n_ids - 1);
    g_n_ids_hist[idx].fetch_add(1, std::memory_order_relaxed);
}

uint64_t ggml_moe_trace_acquire_call_count(void) {
    return g_acquire_call_count.load(std::memory_order_relaxed);
}

void ggml_moe_trace_n_ids_histogram(uint64_t out8[8]) {
    if (!out8) return;
    for (int i = 0; i < 8; i++) out8[i] = g_n_ids_hist[i].load(std::memory_order_relaxed);
}

void ggml_moe_trace_note_chunk_call(void) {
    // ★acquire_call_count와 달리 트레이스 게이트 뒤에 둔다(호출측인 ggml-cpu.c가
    // moe_direct_active && ggml_moe_trace_enabled()로 이미 게이트하므로 여기서 재확인은
    // 방어적 성격 — off 세션에서 호출측이 아예 부르지 않으므로 정상 경로에서는 항상 true).
    if (!trace_active_and_ready()) return;
    g_chunk_call_count.fetch_add(1, std::memory_order_relaxed);
}

uint64_t ggml_moe_trace_chunk_call_count(void) { return g_chunk_call_count.load(std::memory_order_relaxed); }

uint64_t ggml_moe_trace_event_count(void)         { return g_write_pos.load(std::memory_order_relaxed); }
// 배선 R1(사양 §4) — 준비 안 된 상태 drop + 링 덮어쓰기 + prefill 오버플로의 합성값.
uint64_t ggml_moe_trace_timeline_drop_count(void) {
    return g_drop_count.load(std::memory_order_relaxed)
         + g_overwrite_count.load(std::memory_order_relaxed)
         + g_prefill_overflow_count.load(std::memory_order_relaxed);
}
uint64_t ggml_moe_trace_qpc_frequency(void)       { return (uint64_t)g_qpc_freq; }
uint64_t ggml_moe_trace_ring_capacity_records(void) { return RING_CAPACITY_RECORDS; }

uint64_t ggml_moe_trace_overwrite_count(void)        { return g_overwrite_count.load(std::memory_order_relaxed); }
uint64_t ggml_moe_trace_prefill_overflow_count(void) { return g_prefill_overflow_count.load(std::memory_order_relaxed); }
uint64_t ggml_moe_trace_anchor_negative_count(void) { return g_anchor_negative_count.load(std::memory_order_relaxed); }

void ggml_moe_trace_reset_for_test(void) {
    seq_lock_acquire();
    g_event_seq_next = 1; // 0은 sentinel(위 전역 선언부 주석 참조) — 리셋 후에도 1부터.
    // 배선 4a 신설 — §5-1 임계구역 전용 작업 상태도 이 lock 안에서 초기화한다(다른 스레드가
    // 동시에 이 구역을 쓰고 있을 수 없음을 이 lock이 보장 — g_event_seq_next와 동일 취급).
    g_wall_active_phase = MOE_TRACE_PHASE_UNKNOWN;
    g_wall_last_tick = 0;
    g_wall_lock_wait_count = 0;
    g_wall_data_wait_count = 0;
    g_wall_graph_base_c = g_wall_graph_base_lcrit = g_wall_graph_base_other = 0;
    g_wall_graph_begin_tick = 0;
    seq_lock_release();
    g_write_pos.store(0, std::memory_order_relaxed);
    g_drop_count.store(0, std::memory_order_relaxed);
    g_compute_active.store(0, std::memory_order_relaxed);
    g_app_qd.store(0, std::memory_order_relaxed);
    g_wait_active.store(0, std::memory_order_relaxed);
    for (int ph = 0; ph < 2; ph++) {
        WallStateAccum & acc = g_wall_accum[ph];
        acc.c_ticks.store(0, std::memory_order_relaxed);
        acc.lcrit_ticks.store(0, std::memory_order_relaxed);
        acc.other_ticks.store(0, std::memory_order_relaxed);
        acc.l_lock_ticks.store(0, std::memory_order_relaxed);
        acc.l_data_ticks.store(0, std::memory_order_relaxed);
        acc.mismatch_count.store(0, std::memory_order_relaxed);
        g_wall_run_base_c[ph] = g_wall_run_base_lcrit[ph] = g_wall_run_base_other[ph] = 0;
        g_wall_run_base_llock[ph] = g_wall_run_base_ldata[ph] = 0;
        g_wall_run_base_mismatch[ph] = 0;
        // 배선 R9 신설 상태도 함께 리셋 — g_wall_accum과 동일 취급.
        WallStateAccum & win_acc = g_wall_accum_window[ph];
        win_acc.c_ticks.store(0, std::memory_order_relaxed);
        win_acc.lcrit_ticks.store(0, std::memory_order_relaxed);
        win_acc.other_ticks.store(0, std::memory_order_relaxed);
        win_acc.l_lock_ticks.store(0, std::memory_order_relaxed);
        win_acc.l_data_ticks.store(0, std::memory_order_relaxed);
        win_acc.mismatch_count.store(0, std::memory_order_relaxed);
        g_wall_win_run_base_c[ph] = g_wall_win_run_base_lcrit[ph] = g_wall_win_run_base_other[ph] = 0;
        g_wall_win_run_base_llock[ph] = g_wall_win_run_base_ldata[ph] = 0;
        g_wall_win_run_base_mismatch[ph] = 0;
        // 배선 4d 신설 상태도 함께 리셋 — g_wall_run_base_*와 동일 취급.
        g_io_run_base_read_bytes[ph] = g_io_run_base_hit[ph] = g_io_run_base_miss[ph] = 0;
        // 배선 S3a 신설 상태도 함께 리셋 — g_io_run_base_*와 동일 취급.
        g_io_win_run_base_read_bytes[ph] = g_io_win_run_base_hit[ph] = g_io_win_run_base_miss[ph] = 0;
    }
    g_acquire_call_count.store(0, std::memory_order_relaxed);
    for (auto & c : g_n_ids_hist) c.store(0, std::memory_order_relaxed);
    // 배선 2단계 신설 상태도 함께 리셋(반복 테스트 간 run_index/graph_seq가 이어지지
    // 않도록 — g_compute_active/g_app_qd와 동일 취급).
    g_run_index_next.store(0, std::memory_order_relaxed);
    g_current_run_index.store(0, std::memory_order_relaxed);
    g_graph_context.store((uint64_t)MOE_TRACE_PHASE_UNKNOWN, std::memory_order_relaxed);
    // 배선 3단계 신설 상태도 함께 리셋 — chunk_call_count는 다른 hard counter들과 동일
    // 취급. O8 표는 selftest가 graph_seq=0 등 낮은 값을 여러 블록에서 재사용하므로(실제
    // llama-context.cpp의 프로세스 전역 단조 카운터를 거치지 않는 합성 호출) 여기서 다시
    // O8_UNTOUCHED로 채우지 않으면 이전 블록의 claim이 새 블록에서 "이미 봄"으로 잘못
    // 남는다 — reset_for_test는 selftest 전용 훅이라 393,216칸 재기입 비용은 무관하다.
    g_chunk_call_count.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < O8_TABLE_SIZE; i++) {
        g_o8_last_graph_seq[i].store(O8_UNTOUCHED, std::memory_order_relaxed);
    }
    // 배선 R1 신설 상태도 함께 리셋 — g_run_index_next 등 배선 2단계 상태와 동일 취급
    // (반복 테스트 간 창 게이팅·prefill 버퍼 상태가 이어지지 않도록).
    g_prefill_cur.store(0, std::memory_order_relaxed);
    g_prefill_write_pos[0].store(0, std::memory_order_relaxed);
    g_prefill_write_pos[1].store(0, std::memory_order_relaxed);
    g_prefill_last_buf.store(-1, std::memory_order_relaxed);
    g_prefill_last_count.store(0, std::memory_order_relaxed);
    g_window_mode.store(0, std::memory_order_relaxed);
    g_run_decode_graph_idx = 0;
    g_overwrite_count.store(0, std::memory_order_relaxed);
    g_prefill_overflow_count.store(0, std::memory_order_relaxed);
    g_flush_base_pos.store(0, std::memory_order_relaxed);
    g_anchor_negative_count.store(0, std::memory_order_relaxed);
    // ★env 캐시(g_enabled_cached)와 g_ready/g_ring_base/g_qpc_freq(1회 할당)는 리셋하지
    // 않는다 — ggml_moe_direct_reset_for_test()와 동일 관례(모듈 §13). MOE_P1_TRACE=1로
    // 고정된 프로세스 안에서 반복 테스트할 때 링버퍼를 재할당하지 않고 논리적으로만
    // 비운다(write_pos=0이면 다음 기록이 슬롯 0부터 다시 채운다).
}

bool ggml_moe_trace_debug_copy_record(uint64_t logical_index, void * out64) {
    if (!g_ready.load(std::memory_order_acquire) || !out64) return false;
    uint64_t wp = g_write_pos.load(std::memory_order_acquire);
    if (logical_index >= wp) return false;
    if (wp - logical_index > RING_CAPACITY_RECORDS) return false;
    uint64_t slot = logical_index & RING_MASK;
    std::memcpy(out64, g_ring_base + slot * RECORD_SIZE, RECORD_SIZE);
    return true;
}

bool ggml_moe_trace_ready_for_test(void) { return g_ready.load(std::memory_order_acquire); }

} // extern "C"

#else // !_WIN32 — 비Windows 스텁(SPEC: "비Windows는 전부 no-op·실패 반환 스텁")

extern "C" {

bool ggml_moe_trace_enabled(void) { return false; }
int32_t ggml_moe_trace_gauge_compute_active(void) { return 0; }
int32_t ggml_moe_trace_gauge_app_qd(void) { return 0; }
int32_t ggml_moe_trace_gauge_wait_active(void) { return 0; }
bool    ggml_moe_trace_window_active(void) { return false; }
int ggml_moe_trace_init(void) { return 0; }
void ggml_moe_trace_flush(const char *) {}
void ggml_moe_trace_finalize(void) {}
uint64_t ggml_moe_trace_graph_begin(uint16_t, uint32_t, uint8_t, uint32_t) { return 0; }
void ggml_moe_trace_graph_end(uint64_t, uint16_t, uint32_t, uint8_t) {}
uint64_t ggml_moe_trace_run_begin(uint16_t) { return 0; }
void ggml_moe_trace_run_end(uint64_t, uint16_t, uint32_t, uint32_t) {}
uint16_t ggml_moe_trace_next_run_index(void) { return 0; }
uint16_t ggml_moe_trace_current_run_index(void) { return 0; }
void ggml_moe_trace_current_graph_context(uint16_t * out_run_index, uint32_t * out_graph_seq, uint8_t * out_phase) {
    if (out_run_index) *out_run_index = 0;
    if (out_graph_seq) *out_graph_seq = 0;
    if (out_phase)     *out_phase     = MOE_TRACE_PHASE_UNKNOWN;
}
uint64_t ggml_moe_trace_io_submit(uint16_t, uint32_t, uint8_t, uint8_t, uint8_t, uint8_t, uint32_t, uint32_t, uint32_t) { return 0; }
void ggml_moe_trace_io_complete(uint64_t, uint8_t, uint32_t, bool, uint32_t) {}
uint64_t ggml_moe_trace_lock_wait_begin(uint16_t, uint32_t, uint8_t) { return 0; }
void ggml_moe_trace_lock_acquired(uint64_t) {}
uint64_t ggml_moe_trace_data_wait_begin(uint16_t, uint32_t, uint8_t, uint8_t, uint8_t, uint8_t, uint32_t) { return 0; }
void ggml_moe_trace_data_wait_end(uint64_t) {}
void ggml_moe_trace_ready_publish(uint16_t, uint32_t, uint8_t, uint8_t, uint8_t, uint8_t, uint32_t) {}
uint64_t ggml_moe_trace_compute_enter(uint16_t, uint32_t, uint8_t, uint8_t, uint8_t, int32_t) { return 0; }
int64_t ggml_moe_trace_query_tick(void) { return 0; }
void ggml_moe_trace_compute_exit(uint64_t, uint32_t, int64_t, int64_t) {}
void ggml_moe_trace_first_arith_consume(uint16_t, uint32_t, uint8_t, uint8_t, uint8_t, uint8_t, uint32_t) {}
bool ggml_moe_trace_first_arith_claim(uint32_t, uint8_t, uint8_t, uint8_t) { return false; }
void ggml_moe_trace_demand_digest(uint16_t, uint32_t, uint8_t, const struct ggml_moe_trace_demand_entry *, uint32_t) {}
void ggml_moe_trace_snapshot(uint16_t, uint32_t, const uint64_t *, uint32_t, const uint64_t *, uint32_t) {}
void ggml_moe_trace_wall_state_totals(uint8_t, int64_t * out_c, int64_t * out_lcrit, int64_t * out_other,
                                       int64_t * out_llock, int64_t * out_ldata, uint64_t * out_mismatch) {
    if (out_c) *out_c = 0;
    if (out_lcrit) *out_lcrit = 0;
    if (out_other) *out_other = 0;
    if (out_llock) *out_llock = 0;
    if (out_ldata) *out_ldata = 0;
    if (out_mismatch) *out_mismatch = 0;
}
void ggml_moe_trace_note_acquire_call(int32_t) {}
uint64_t ggml_moe_trace_acquire_call_count(void) { return 0; }
void ggml_moe_trace_n_ids_histogram(uint64_t out8[8]) { if (out8) for (int i = 0; i < 8; i++) out8[i] = 0; }
void ggml_moe_trace_note_chunk_call(void) {}
uint64_t ggml_moe_trace_chunk_call_count(void) { return 0; }
uint64_t ggml_moe_trace_event_count(void) { return 0; }
uint64_t ggml_moe_trace_timeline_drop_count(void) { return 0; }
uint64_t ggml_moe_trace_qpc_frequency(void) { return 0; }
uint64_t ggml_moe_trace_ring_capacity_records(void) { return 0; }
uint64_t ggml_moe_trace_overwrite_count(void) { return 0; }
uint64_t ggml_moe_trace_prefill_overflow_count(void) { return 0; }
uint64_t ggml_moe_trace_anchor_negative_count(void) { return 0; }
void ggml_moe_trace_reset_for_test(void) {}
bool ggml_moe_trace_debug_copy_record(uint64_t, void *) { return false; }
bool ggml_moe_trace_ready_for_test(void) { return false; }

} // extern "C"

#endif // _WIN32
