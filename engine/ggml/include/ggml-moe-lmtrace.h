// MoE-Direct 레버 A 단계 1 — layer-major 관측 모듈(LMTRACE). **관측 전용·기본 OFF.**
//
// 사양: SUPERBATCH_DESIGN.md v0.3 §3(a) 계측 · §5 G-L0 구조 게이트 ·
//       reviews/codex_techdev_speed_r11.md §⑤ 발주문 S0 절.
//
// 이 모듈은 아무 것도 바꾸지 않는다 — 그래프 형상·backend 배치·split 경계·커널 결정·출력 토큰
// 중 어느 것도 건드리지 않고 "이미 확정된 스케줄"을 읽어 지문만 남긴다. 그래서 ggml_backend_sched
// 의 eval callback(=split 을 graph view 로 재분할해 실행 형상을 바꾼다)은 **쓰지 않는다**.
//
// 기능 두 가지:
//   (A) graph/sched 시그니처 — 실행 직전의 ordered node(op/type/shape/stride/버퍼 사용)·split
//       경계·split별 backend 를 결정론 직렬화(길이 프레이밍)해 sidecar 에 적고 SHA-256 을 남긴다.
//       ★관측 범위는 **스케줄러 수준까지**다. CUDA 내부 fusion/커널 선택은 stock DLL 에서
//         관측 불가(근거는 아래 "CUDA 관측 한계" 절) — G-L0 문구와의 갭은 보고서에 표면화한다.
//   (B) request-level 카운터 — 프리필 요청 단위 graph_build/sched_alloc/submit_sync/boundary_copy
//       분해 시간 + 세그먼트 수 + experts 읽기 바이트(기존 direct 카운터의 스냅샷 델타).
//
// ★CUDA 관측 한계(1차 소스 조사 결과 — 시공 보고서와 세트):
//   ggml-cuda.cu 의 fusion 보고 경로(nodes_fused 로그)는 전부 `#ifdef GGML_CUDA_DEBUG` 이고 그
//   정의는 ggml/src/ggml-cuda/CMakeLists.txt 의 GGML_CUDA_DEBUG 옵션에서만 켜진다 — 배포 DLL 엔
//   없다. 노출 export 도 11개뿐이라 in-process 조회 API 가 없다. 따라서 이 모듈이 남기는 것은
//   **sched 수준 시그니처**이며, fusion 동일성은 별도 논거(동일 DLL·동일 그래프 형상)로만 말할 수
//   있다. 그 논거의 한계(fusion 이 버퍼 배치·포인터 겹침·CUDA graph capture 상태에도 의존)는
//   보고서 §설계 결정에 기록했다.
//
// 환경 변수(전부 기본 OFF):
//   MOE_DIRECT_LMTRACE=1        모듈 활성(최초 사용 시 1회 캐시)
//   MOE_DIRECT_LMTRACE_OUT=<접두사>   sidecar 경로 접두사. **미설정이면 파일을 만들지 않는다**
//                               (카운터·digest 만 메모리 유지). 산출물 3종:
//                               <접두사>.lmt    — 바이너리 레코드(길이 프레이밍)
//                               <접두사>.lmtidx — 레코드별 한 줄:
//                                 `seq= uid= nodes= leafs= splits= externs= bytes= sha256=`
//                                 (★externs 는 G-L0 fail-close 조건 — R13 ③8·④7)
//                               <접두사>.lmtreq — 요청 단위 카운터 JSON lines. 키:
//                                 req_seq·status·segments·prefill_tokens·graph_build_ms·
//                                 sched_alloc_ms·submit_host_ms·sync_ms·boundary_copy_ms·
//                                 experts_read_bytes·experts_hit·experts_miss·graph_records·
//                                 graph_chain_sha256
//   MOE_DIRECT_LMTRACE_MAX_MB=<n>   sidecar 바이너리 상한(기본 512). 초과분은 드롭 카운트만 증가.
//
// ★기존 metrics jsonl(MOE_DIRECT_METRICS)은 건드리지 않는다 — 소비자 중 하나가 header/summary
//   사이의 interior 레코드 type 을 "periodic 만"으로 강제 검사하므로(bench/g3_2bm_killscreen.py)
//   신규 type 주입은 기존 게이트를 깨뜨린다. 그래서 전용 sidecar 로 분리했다.
//
// Windows 전용: 비Windows 빌드에서는 ggml-moe-lmtrace.cpp 가 전부 no-op 스텁으로 컴파일된다.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 상수
// ---------------------------------------------------------------------------

// sidecar 바이너리 레코드 헤더 magic / layout version.
#define MOE_LMTRACE_MAGIC          "MOELMT1"
#define MOE_LMTRACE_LAYOUT_VERSION 1u

// 레코드 종류.
enum {
    MOE_LMTRACE_KIND_GRAPH_SIGNATURE = 1,
};

// ---------------------------------------------------------------------------
// ★S2 range 태그 (하위 호환 layout — SPEC_S2_RANGEGRAPH §5 ⓐ)
// ---------------------------------------------------------------------------
// S2 층-우선 프리필은 한 llama_decode 안에서 `for R(층 구간): for ub` 순회로 **(range, ub)마다
// 별도 그래프**를 실행한다. 판독기가 그 다중 레코드를 "ub당 논리 그래프 1개"로 재조립하려면
// 레코드에 (decode_call/range/ub/il 구간/경계 역할) 이 실려야 한다.
//
// ★하위 호환 규약(필수 — S0 baseline v1 파일이 계속 판독돼야 G-L0 비교가 성립한다):
//   `layout_version` 은 **1 그대로 둔다**. 대신 기존 헤더의 `reserved` u32 를 `tag_flags` 로
//   의미 부여한다 — 0 이면 태그 블록 없음(= 구 레이아웃과 **바이트 동일**), 1 이면 헤더 직후
//   (n_backends 다음) 에 아래 고정 48바이트 블록이 온다. 그래서
//     · S0 baseline 파일           → tag_flags=0 → 구 파서 경로 그대로
//     · S2 legacy(비 admission) 레코드 → tag_flags=0 → baseline 과 1:1 바이트 비교 가능
//     · S2 range 레코드             → tag_flags=1 → 재조립 대상
//   `tag_bytes` 는 블록 총 바이트(이 필드 포함)이며, 판독기는 48 초과분을 skip 해 전방 호환한다.
//
//   레이아웃(little-endian):
//     u32 tag_bytes      = MOE_LMTRACE_TAG_BYTES (48)
//     u32 tag_version    = MOE_LMTRACE_TAG_VERSION (1)
//     u64 decode_call    프로세스 전역 단조 증가 — layer-major admission 을 통과한 decode 호출 번호
//     u32 range_idx      0..n_ranges-1
//     u32 n_ranges
//     u32 ub_idx         0..n_ubatch-1
//     u32 n_ubatch       = K
//     i32 il_start
//     i32 il_end         (exclusive)
//     u32 boundary_role  bit0=bank 입력 leaf 소비 · bit1=경계 CPY 출력 생성
//     u32 mode           = 1 (layer-major range)
#define MOE_LMTRACE_TAG_FLAG_NONE  0u
#define MOE_LMTRACE_TAG_FLAG_RANGE 1u
#define MOE_LMTRACE_TAG_VERSION    1u
#define MOE_LMTRACE_TAG_BYTES      48u

enum {
    GGML_MOE_LMTRACE_BOUNDARY_ROLE_NONE = 0,
    GGML_MOE_LMTRACE_BOUNDARY_ROLE_IN   = 1,   // bank 슬라이스를 입력 leaf 로 소비
    GGML_MOE_LMTRACE_BOUNDARY_ROLE_OUT  = 2,   // 경계 CPY 로 bank 슬라이스에 기록
};

enum {
    GGML_MOE_LMTRACE_MODE_LEGACY = 0,
    GGML_MOE_LMTRACE_MODE_RANGE  = 1,
};

// (B) 요청 단위 고정비 분해 스테이지 — R11 §M-1 의 build/alloc/submit/sync **4분리**.
//   ★submit 과 sync 는 반드시 분리한다(R13 ③4 지적): `llama_context::graph_compute()` 는
//     `ggml_backend_sched_graph_compute_async()` 만 호출하므로 그 둘레는 **host 측 제출 시간**일
//     뿐이고, 실제 device 대기는 `llama_context::synchronize()` 안의
//     `ggml_backend_sched_synchronize()` 에서 발생한다. 합쳐 두면 이름과 달리 sync 가 빠진 값이다.
//   boundary_copy 는 S0 에서 **배선만**(현재 경로엔 호출자가 없어 항상 0) — 층 경계 F32 CPY 가
//   생기는 S2 에서 채워진다.
//   ★S2 에서의 정확한 의미(SPEC §5 ⓑ · r28 수정문안 7): 경계 CPY 는 range 그래프에 **융합된
//     노드**라 host 제출 단위로 분리되지 않는다. 따라서 `boundary_copy_ms` 는 **경계 CPY 를
//     포함한 range 그래프의 host-submit 시간 상계**이며 CPY 단독 시간도, device 시간도 아니다
//     (device 분리는 G-P2 계측 소관). S2 가 정확히 책임지는 값은 아래 bytes/count 다.
enum {
    GGML_MOE_LMTRACE_STAGE_GRAPH_BUILD   = 0,
    GGML_MOE_LMTRACE_STAGE_SCHED_ALLOC   = 1,
    GGML_MOE_LMTRACE_STAGE_SUBMIT_HOST   = 2,  // graph_compute() 둘레 = async 제출까지(호스트)
    GGML_MOE_LMTRACE_STAGE_SYNC          = 3,  // ggml_backend_sched_synchronize() 둘레 = device 대기
    GGML_MOE_LMTRACE_STAGE_BOUNDARY_COPY = 4,
    GGML_MOE_LMTRACE_STAGE_COUNT         = 5,
};

// 요청 폐합 사유 — `.lmtreq` 의 "status" 값.
#define MOE_LMTRACE_STATUS_DECODE  "decode"   // 첫 decode graph 진입으로 자연 폐합
#define MOE_LMTRACE_STATUS_REOPEN  "reopen"   // 다음 프리필 요청이 열리며 폐합(순수 프리필 요청)
#define MOE_LMTRACE_STATUS_FINISH  "finish"   // 명시 폐합(컨텍스트 소멸 등)
#define MOE_LMTRACE_STATUS_ABORT   "abort"    // 실패 경로 폐합(다음 요청과 합쳐지지 않게 상태 리셋)

// ---------------------------------------------------------------------------
// 게이트
// ---------------------------------------------------------------------------

// env MOE_DIRECT_LMTRACE=1 이면 true(최초 호출 시 1회 캐시). false 면 이 모듈의 모든 경로가
// 비활성이며, 침습 지점은 이 함수 하나만 평가하고 아무 일도 하지 않는다.
GGML_API bool ggml_moe_lmtrace_enabled(void);

// ---------------------------------------------------------------------------
// (A) graph/sched 시그니처
// ---------------------------------------------------------------------------

// split 경계 기술자 — ggml_backend_sched_split 은 ggml-backend.cpp 파일 전용 타입이라 훅 지점에서
// 이 POD 로 옮겨 넘긴다(스케줄러 내부 구조를 헤더로 노출하지 않기 위함).
struct ggml_moe_lmtrace_split_desc {
    int32_t backend_id;
    int32_t i_start;
    int32_t i_end;
    int32_t n_inputs;
};

// 스케줄 확정 이후·커널 발행 이전에 1회 호출한다(호출자가 게이트를 먼저 확인해야 한다).
//   graph            : 스케줄러가 조립한 실행 그래프(주입된 input_dep/input_cpy 포함)
//   node_backend_ids : [graph->n_nodes] — graph->nodes 와 인덱스 정렬
//   leaf_backend_ids : [graph->n_leafs] — graph->leafs 와 인덱스 정렬
//   backends         : [n_backends] — NULL 원소 허용(이름은 빈 문자열로 기록)
//   splits           : [n_splits]
// 포인터 값(data/extra)·할당 주소는 기록하지 않는다(할당자 배치는 정체성 계약의 대상이 아니다).
GGML_API void ggml_moe_lmtrace_graph_capture(
        const struct ggml_cgraph                   * graph,
        const int32_t                              * node_backend_ids,
        const int32_t                              * leaf_backend_ids,
        int32_t                                      n_backends,
        ggml_backend_t                             * backends,
        int32_t                                      n_splits,
        const struct ggml_moe_lmtrace_split_desc   * splits);

// ---------------------------------------------------------------------------
// (B) request-level 카운터
// ---------------------------------------------------------------------------

// graph 실행 1건의 경계 통지(계산 시작 전). 요청 상태기계는 여기서만 전이한다:
//   prefill graph 가 prefill 상태 밖에서 오면 = 새 요청 개시(이전 요청은 폐합·기록)
//   decode  graph 가 prefill 상태에서 오면  = 현 요청 폐합·기록
// 세그먼트 수 = 한 요청에 속한 prefill graph 실행 횟수.
GGML_API void ggml_moe_lmtrace_graph_boundary(int32_t is_prefill, int32_t n_tokens);

// 고정비 스테이지 구간 계측(중첩 없음 — 같은 stage 를 겹쳐 열지 않는다).
GGML_API void ggml_moe_lmtrace_stage_enter(int32_t stage);
GGML_API void ggml_moe_lmtrace_stage_leave(int32_t stage);

// ★S2 — 다음 graph_capture 에 실릴 range 태그를 설정/해제한다(설정 후 capture 1건에만 적용되는
// 것이 아니라, clear 될 때까지 유지된다 — 드라이버가 (range,ub) 마다 set 하고 구간을 벗어날 때
// clear 한다). 게이트 OFF 면 무동작. 스레드 간 공유 상태이므로 v1 단일 활성 요청 전제(SPEC §1)
// 하에서만 의미가 있다.
GGML_API void ggml_moe_lmtrace_range_tag_set(uint64_t decode_call,
                                             int32_t  range_idx, int32_t n_ranges,
                                             int32_t  ub_idx,    int32_t n_ubatch,
                                             int32_t  il_start,  int32_t il_end,
                                             int32_t  boundary_role);
GGML_API void ggml_moe_lmtrace_range_tag_clear(void);

// ★S2 — 층-우선 프리필 진입 통지. `graph_boundary` 와 달리 **한 llama_decode 진입당 1회만**
// 부르며 tokens/phase 를 그때 적산한다(SPEC §5 ⓑ: range compute 는 segment 만 늘린다).
// 이후 각 (range,ub) compute 는 `ggml_moe_lmtrace_range_segment()` 로 세그먼트만 증가시킨다.
GGML_API void ggml_moe_lmtrace_request_prefill_enter(int32_t n_tokens);
GGML_API void ggml_moe_lmtrace_range_segment(void);

// ★S2 — 경계 CPY 회계(정확값). count = 이번 range 그래프가 만든 경계 CPY 개수(0 또는 1),
// bytes = 그 CPY 가 옮기는 바이트. 요청 폐합 시 `.lmtreq` 의 boundary_copy_bytes/
// boundary_copy_count 로 나간다.
GGML_API void ggml_moe_lmtrace_boundary_copy_note(uint64_t bytes, uint64_t count);

// 경계 CPY 회계 조회(판독·selftest 전용 — 현재 열린 요청의 누적값).
GGML_API void ggml_moe_lmtrace_boundary_copy_counters(uint64_t * bytes, uint64_t * count);

// 열려 있는 요청을 즉시 정상 폐합·기록한다(status="finish"). 열린 요청이 없으면 무동작.
// 생산 호출자 = llama_context 소멸자 — decode 없이 끝난 순수 프리필 요청의 레코드가 유실되지
// 않게 한다.
GGML_API void ggml_moe_lmtrace_request_finish(void);

// ★실패 경로 폐합(status="abort"). 열린 요청을 기록한 뒤 **상태기계를 idle 로 되돌린다** —
// 이것이 없으면 실패 이후의 다음 프리필이 `phase_state==prefill` 인 채로 도착해 **이전 요청과
// 같은 요청으로 합쳐진다**(R13 ③4 가 지적한 REACHABLE 결함). 생산 호출자 =
// llama_context::process_ubatch() 의 실패 조기 반환 4지점.
GGML_API void ggml_moe_lmtrace_request_abort(void);

// ---------------------------------------------------------------------------
// 조회(판독·selftest 전용)
// ---------------------------------------------------------------------------

// ★도달성 앵커. NULL 인자는 건너뛴다.
//   probe_calls    : ggml_moe_lmtrace_graph_capture 진입 횟수(게이트 **이전**에 증가).
//                    프로덕션 OFF 에서는 호출자가 게이트로 막으므로 0 이어야 한다.
//   records        : 게이트를 통과해 실제로 직렬화된 레코드 수. OFF 면 0(신규 경로 미진입 증명).
//   dropped        : 상한 초과로 버린 레코드 수.
//   bytes          : 직렬화한 누적 바이트.
//   requests       : 폐합·기록된 요청 수.
GGML_API void ggml_moe_lmtrace_counters(uint64_t * probe_calls,
                                        uint64_t * records,
                                        uint64_t * dropped,
                                        uint64_t * bytes,
                                        uint64_t * requests);

// 마지막 레코드의 SHA-256(소문자 hex, NUL 종료). 65바이트 이상 필요. 성공=0, 버퍼 부족=-1.
GGML_API int ggml_moe_lmtrace_last_digest(char * out, int32_t cap);

// 지금까지의 레코드 전체를 순서대로 묶은 연쇄 SHA-256(chain = SHA256(prev_hex || rec_hex)).
// 한 런 전체의 graph 시그니처를 값 하나로 비교할 때 쓴다. 성공=0, 버퍼 부족=-1.
GGML_API int ggml_moe_lmtrace_chain_digest(char * out, int32_t cap);

// selftest 전용 — env 재해석 + 카운터/digest 초기화 + sidecar 핸들 닫기. 프로덕션 호출 금지.
GGML_API void ggml_moe_lmtrace_test_reset(void);

#ifdef __cplusplus
}
#endif
