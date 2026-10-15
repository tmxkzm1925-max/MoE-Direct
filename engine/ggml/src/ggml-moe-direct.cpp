// MoE 직독(direct-read) 모듈 본체 — v2(manifest-driven).
// 사양: REPACK_V2_DESIGN.md §4(직독 소비자 v2)·§3(manifest v2 스키마·불변식 8항).
// v1 "GPT-OSS 상수 검증기"를 해체하고 manifest v2를 유일 권위로 소비한다(PROD_* 전폐·층별
// stride·멀티-shard·384E uint16·등록 3집합·슬롯 용량 불변식).
// Windows 전용 — 비Windows는 파일 하단의 스텁 구현으로 컴파일된다.

#include "ggml-moe-direct.h"
#include "ggml-moe-direct-internal.h"
#include "ggml-moe-trace.h"
#include "ggml-moe-phasea.h"
#include "ggml-moe-phaseb.h"
#include "ggml-moe-shadow.h"   // ★C3 캐시 view 의 선언(구현은 이 파일 — 캐시 상태 소유자)
#include "ggml-moe-leadtime.h" // ★PROBE_LEADTIME_SPEC_R1 §2 — stamp B(T1) 결선 + arm/flush(기본 OFF)
#include "ggml-moe-gpuearly.h" // ★PROBE_GPU_EARLY_SPEC_R1 §4 — C 축 arm/flush 결선(기본 OFF)
#include "ggml.h"
#include "gguf.h"   // ★§Z-⑥ A1 — mode=virtual 부팅 대조가 원본 GGUF 헤더를 재파싱한다
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

#include <process.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>   // E-2b-0: std::unique_ptr owns the fixed-size lease-handle table
#include <mutex>
#include <new>      // E-2b-0 r4: placement new and std::launder for the in-place payload
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

namespace {

// ===========================================================================
// §1. 동결 type-trait 표 (bench/repack/repack_experts.py QUANT_TRAITS 와 정확 일치 —
//     b10057 ggml-common.h 전항 대조 완료분. 산술 권위는 항상 이 표에서 재계산).
// ===========================================================================







// OPEN_ARCH B axis - ggml_type enum -> frozen upper-case wire type spelling
// ("F32"/"MXFP4"/"Q3_K"...).
// NOTE: ggml_type_name() must NOT be used for wire work: it returns the lower-case ggml spellings
// ("f32", "mxfp4", "q3_K"), so feeding it into the frozen MOE-INVENTORY-V1 rows would silently
// change the digest input. The table above is the single source of truth (OPENARCH_B_SPEC 1).
// Returns nullptr when the live type is outside the frozen table.





// ★QD 레버(비동기 다중 outstanding — reviews/codex_qd_u6_design_xcheck.md 심의 3):
// 한 배치 acquire가 동시에 띄울 수 있는 최대 outstanding read 수(env MOE_DIRECT_QD).
// 기본 1 = 검증 기간 값(신 플래너의 concurrency-off 대조군 — 승격 후 별도 커밋에서 상향한다.
// 기본값을 먼저 8로 두면 회귀가 "플래너 재구성" 탓인지 "동시성" 탓인지 귀속되지 않는다).
// 상한 63 = 회수가 WaitForMultipleObjects(MAXIMUM_WAIT_OBJECTS=64) 한도 안에 있어야 한다.



// ★F1(reference_lock 소비자 결속) — bench/repack/repack_experts.py EXPECT_CATALOG 사본
// (2026-07-24 실측 sha256 대조 후 기입: gpt-oss·kimi·mistral expect 파일 재해시 일치). 소비자
// 기동 검증이 프로듀서와 무관하게 독립 재확인한다(설계 §3 불변식 8 "각각").
// ★부록A dual-scope: 항목에 scope 병기(부재=all). 카탈로그 scope == manifest routed_scope 대조로
// 4자 일치(CLI==expect==catalog==manifest)의 소비자 몫을 강제(불일치=fail-closed). 기존 4항목
// digest 불변·scope="all"(하위호환). 신규 qwen35-122b-nonextn만 새 digest+execution.



// ===========================================================================
// OPEN_ARCH B axis (OPENARCH_B_SPEC_DRAFT v0.2 sections 2-1 / 2-5) - approved arch template
// catalog. This is a SEPARATE axis from the model pin (EXPECT_CATALOG above): the pin approves
// "this model", the template approves "this arch formula". A SHA is a binding, not an approval,
// so the template branch admits only the template_id@version pairs frozen in this table, and the
// expected tensor-name set is generated INDEPENDENTLY by the formula below (the repacker's
// regexes are deliberately not reused - the two derivations agreeing on the digest is what makes
// the proof non-circular).
// Approval axis = {template_id, template_version}. The remaining fields are the closure formula
// of section 2-3b (OPEN_ARCH_DESIGN section 1 table) itself, so a rule change means a version
// bump, which means consumers must re-approve.
// ===========================================================================

// R-B3b: `OPEN_ARCH_TEMPLATE_ABI_STR` moved to ggml-moe-direct-test-entry.cpp with its only
// user, `ggml_moe_direct_openarch_probe_json` (RB3B_ATOM_SPEC.json :: destination.constants).

// template-mode reference_lock encoding: profile_id = "arch-template:<template_id>@<version>".








// All three frozen templates are separate gate/up/down (fused templates are not registered).




// SHA-256(BCrypt one-shot pseudo-handle) → 소문자 hex. 실패=false.




// 프로덕션=frozen EXPECT_CATALOG 조회. override_profile_id!=null 이면 테스트 전용 단일 엔트리
// 카탈로그(프로덕션 카탈로그 우회 아님 — seal_for_test에서만 주입). 반환 false=미등재.
// ★부록A: approved_scope 도 반환(카탈로그 등재 scope, override는 test_scope·null/빈="all").


// "blk.<digits>." 접두에서 layer id 추출(routed 텐서명 — §0 정규식과 정합). 실패=false.
// execution scope에서 pending 텐서를 제외 경계(layer >= n_layer-n_layer_nextn)로 분류하는 데 쓴다.


// OPEN_ARCH B axis - canonical routed tensor name parser.
// The grammar is exactly the loader's frozen regex (llama-model.cpp:1752
// `blk\.([0-9]+)\.ffn_(gate|up|down|gate_up)_exps\.(weight|bias)`). The loader registers only that
// shape with the module, so a non-canonical name reaching the template branch is a fixture/wiring
// defect (fail-closed). On success layer/kind/suffix are filled in.


// OPEN_ARCH B axis - the frozen inventory digest (section 1 wire fact):
//   sha256( b"MOE-INVENTORY-V1\n" + "".join(sorted(rows)) )
//   row = "<tensor-name>\t<type>\t<d0>,<d1>,..\t<layer>\t<scope>\n"  (utf-8)
// The producer (repack_experts.py:836) sorts Python str values and then encodes UTF-8. Every name,
// type and scope token the formula generates is ASCII, so the byte-wise sort below is equivalent to
// that (breaking the equivalence would require a non-ASCII tensor name, and at that point the sort
// convention itself is back up for review).


// verify_report.json(JSONL)의 마지막 비어있지 않은 레코드를 **release 게이트**로 소비한다.
// ★구 구현은 reference_lock만 읽어 verify FAIL 산출물이 "verified"로 통과할 수 있었다
// (reviews/codex_release_arch_xcheck.md §1-1 실결함). 통과 조건(하나라도 어긋나면 fail-close):
//   ① pass 가 boolean true(부재·비불리언·false 전부 거부)
//   ② pairs_pass == pairs_total == expected_pairs (세 필드 전부 존재·비음 정수)
//   ③ failures·padding_failures·problems 세 배열이 전부 존재하고 전부 빈 배열
//   ④ reference_lock.{profile_id,expect_sha256} 추출(manifest 대조는 호출자 몫)
//   ⑦ ★manifest_sha256 필수 — 그 verify 가 검증한 manifest.json **파일 바이트**의 SHA-256
//      (64자리 hex). 부재·비문자열·비16진 = 거부. 구 스키마 report(필드 없음)는 여기서 걸리며
//      `repack_experts.py --verify-only` 1회 재검증으로 승급한다. 실제 manifest 바이트와의 대조는
//      호출자(seal) 몫 — ④와 같은 분업(파일 경로를 아는 쪽이 대조).
// 스키마 권위 = bench/repack/repack_experts.py verify_repack() 반환 dict(:1534-1538).
// ★재팩 미완료 표식(experts.bin.partial) 검사는 호출자(seal)가 한다 — 그 표식은 out_dir 소유이고
//   프로듀서가 verify pass 시에만 제거하기 때문(repack_experts.py:1002·1027·1099).


// ===========================================================================
// §2. 내부 자료구조 (manifest v2 파생)
// ===========================================================================




// ★V1 무행동 증명용 FNV-1a 64(SPEC_PHASEA_WIRE §8) — metrics 활성 시에만 호출된다.








// 등록 엔트리(R_host 전용) — 텐서명→(mlpos, part_idx).


// 등록 대기(2단계 로더가 register()로 쌓음) — L_all/R_host 재구성 원천.


// 슬롯 하나 = experts.bin 레코드 1개(<= slot_stride_max)를 담는 고정 버퍼.
// ★B2(PHASEB_DESIGN §4 · C1′): READY_PROBATION 슬롯은 demand last_used와 **별개로**
//   cache_class와 prefetch_intent_ordinal을 보유한다(probation comparator의 실행 근거).
//   intent_ordinal은 intent_seq 5-튜플의 단조 전사(SPEC_PHASEB_WIRE §2.1)이므로
//   "오래된 intent 우선" 정렬에 ordinal 대소를 쓰는 것은 5-튜플 정렬과 동치다.


}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// ★5c-T §1 — READY slot 의 owner 스탬프 결속 해제. 기존 `pf_*` 리셋 6지점과 **항상 세트**로
//   호출한다(하나라도 빠지면 앞 owner 의 스탬프가 다음 슬롯 사용자로 새어 참여 집합·census
//   owner ref 가 오염된다 — PendingIo 재사용 누수(R1 #1)와 같은 계열의 결함).
void pb_slot_stamp_clear(Slot & s) {
    s.pf_request_epoch = GGML_MOE_PHASEB_NA_U32;
    s.pf_graph_seq     = GGML_MOE_PHASEB_NA_U32;
    s.pf_stamp_k       = 0;
    s.pf_stamp_n       = 0;
    s.pf_stamp_qd      = 0;
    s.pf_stamp_rev     = 0;
    s.pf_gen_admission_seq = GGML_MOE_PHASEB_NA_U64;
}













}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// ★QD 레버 — 배치 1건의 유니크 miss 하나 = 로드 에피소드 하나. slot/generation/offset은 claim
// 단계에서 확정되고(제출 전 전량 확정 → slot 배정이 QD 값과 무관하게 결정적), 트레이스 상관자만
// 제출·회수 단계에서 채워진다.
// ★B1(PHASEB_DESIGN §2): 이 구조체는 legacy 배치 pump와 영속 dispatcher가 **공유**한다 —
//   제출/회수/publish 본문(io_submit_locked/io_finish_locked)이 하나뿐이어야 두 경로의
//   ordered ReadFile digest·slot 배정·LRU가 동형(§7 ABBA 완전일치)임이 구조로 보장된다.


// ★QD 레버 — outstanding 채널 1개. OVERLAPPED와 event는 제출 시점부터 완료까지 주소가 고정돼야
// 한다(커널이 그 주소에 완료를 기록) → 채널 배열은 제출 전에 크기를 확정하고 이후 재할당하지
// 않는다(제출 후 vector resize = OVERLAPPED 주소 무효화).
// ★B1: 영속 경로에서는 seal 시 1회 생성되고 주소가 고정된다(§2 "IoChannel[qd_effective]").


// ===========================================================================
// §2.5 ★B1 영속 I/O dispatcher 자료구조 (PHASEB_DESIGN §2 — FROZEN 사양)
//      control_q(bounded MPSC 64) + PendingIo(demand pending table) + persistent IoChannel.
//      **B1 범위 밖(생성 금지)**: capture pool · router/norm snapshot · predictor worker ·
//      intent_q · prefetch admission — allocation·thread·실행 전부 0(§6 mem echo로 증명).
// ===========================================================================

      // bounded MPSC ring(무손실 — 만석=생산자 대기)




// TestQuery 하위 종류 — "release·absent query·test snapshot은 전부 command"(§2).


// PendingIo 상태(§2) — claimed(슬롯 점유·미제출) → submitted → terminal.

 // ★B2에서 prefetch 추가

// in-flight 읽기 1건. slot은 terminal까지 pin된다(축출/재사용 금지).


// ★5c-T OFF 바이트 동일성 잠금 — metrics `pending_table_bytes`(header·phaseb_mem 두 곳)가
//   `n_slots * sizeof(PendingIo)` 를 그대로 echo 한다. 그래서 5c-T 는 이 구조체에 필드를
//   **추가하지 않고** 병렬 side table(`ModuleState::pb_op_stamp`)로 owner 스탬프를 결속한다.
static_assert(sizeof(PendingIo) == 232,
              "sizeof(PendingIo) is echoed as metrics pending_table_bytes - changing it breaks "
              "OFF-run wire byte identity (SPEC_PHASEB_WIRE appendix 14)");

// ===========================================================================
// ★§Z-⑥ 원자 A5 — IO3 사이드카 wire (`MOEIO3` · SPEC_IO_METRICS_V3 §5 · 재팩 소유)
//
// virtual 은 "record 1건 = 물리 read 1건"이 깨진다. 그래서 PA/PB 채널(그 전제 위에 선
// bin 좌표 축)로는 물리 행을 낼 수 없고, **부모 논리 행(records)과 자식 물리 행(children)을
// 분리한 새 사이드카**가 필요하다(§3-8). copy 는 obligation 채널(copies)이 권위다.
//
// 인코딩(§5-0): 리틀엔디언·packed·NA 센티널(u16 0xFFFF · u32 0xFFFFFFFF · i32 -1 ·
//   u64 식별자류 전비트 1 · **시각(qpc)만 NA=0**).
// 헤더(§5-2): 128B — 오프셋·역할은 PA §2와 동일하고 magic·off20(engine_mode)·
//   off104..115(resolved IO 파라미터 6 + source 코드 6)만 IO3 고유다.
// ===========================================================================
#pragma pack(push, 1)




#pragma pack(pop)
static_assert(sizeof(Io3Header) == 128, "IO3 header must be 128B (SPEC_IO_METRICS_V3 section 5-2)");
static_assert(sizeof(Io3Record) ==  80, "IO3 records row must be 80B (section 5-3)");
static_assert(sizeof(Io3Child)  ==  88, "IO3 children row must be 88B (section 5-4)");
static_assert(sizeof(Io3Copy)   ==  64, "IO3 copies row must be 64B (section 5-5)");

// ===========================================================================
// ★§Z-⑥ 원자 A3 — mode=virtual child 상태(SPEC_REPACK_V3 §3-2·§3-4)
//
// bin 은 "record 1건 = 물리 read 1건"이라 PendingIo 하나가 곧 read 하나였다. virtual 은 한
// record 가 **part 수만큼의 물리 read**(각기 다른 shard·다른 offset)로 흩어지므로, op 과 물리
// read 사이에 child 층이 생긴다. A3 는 그 층을 넣는 작업이고 상태기계를 새로 만들지 않는다 —
// 슬롯 pin·generation·waiters·retry·cancel 은 전부 기존 PendingIo 계약 그대로다.
//
// ★**왜 병렬 side table 인가**: `PendingIo`·`IoChannel` 은 `sizeof` 가 metrics 로 echo 되는
//   동결 구조다(`pending_table_bytes` :3339·:3363·:7227 / channel 표 :3343·:3367·:7229 +
//   `static_assert(sizeof(PendingIo)==232)`). 필드를 늘리면 OFF 런 wire 바이트 동일성이
//   깨진다(SPEC_PHASEB_WIRE 부록 14 — 프리페치 소유 계약). 5c-T 가 같은 제약에서
//   `pb_op_stamp` 병렬 표로 푼 선례를 그대로 따른다.
// ===========================================================================


// ===========================================================================
// ★★PB virtual 런타임 배선 — 원자 A1(①record 운영축 · ②child 물리축)
//    사양 권위 = `SPEC_PB_VIRTUAL_RUNTIME_R1.md` §2-0 A0-1/A0-3 (FROZEN v1.0).
//
// bin 은 record 1건 = 물리 read 1건이라 `pb_inflight_*`(op 축) 하나가 admission cap 과
// 계측을 겸했다. virtual 은 그 등식이 깨져서 축이 둘로 갈린다:
//   ① **record 운영축** = admission commit · reservation · record inflight · 단일 terminal
//   ② **child 물리축**  = credit reserved / submitted / terminal (`v3_inflight_*`)
// 두 축 모두 **mode-neutral(always-on)** 이다 — 관측 기능(IO3·PB 사이드카)이 꺼져 있어도
// 증감한다(A0-1 "①②는 관측 기능과 무관하게 증감"). 그래서 아래 상태는 IO3 방출 함수가
// 아니라 제출/회수 **경로 자체**가 갱신한다.
//
// ★왜 병렬 side table 인가: `PendingIo` 는 `sizeof` 가 metrics 로 echo 되는 동결 구조다
//   (`static_assert(sizeof(PendingIo)==232)`). `V3ChildIo`·`PbOpStamp` 가 같은 제약에서 쓴
//   방식을 그대로 따른다 — `pending_ops[oi]` 와 **같은 인덱스**로 수명을 같이한다.
// ===========================================================================


// ===========================================================================
// ★§Z-⑥ 원자 A6 — 프리필 coalescing(SPEC_REPACK_V3 §3-6 · §5 파라미터)
//
// 같은 (source, tensor part)의 **인접 expert 미스 run** 을 단일 대구간 read 로 합치고,
// 착지는 참가 record 각자의 슬롯으로 **exact slice scatter** 한다. 대구간은 슬롯에 직착지할
// 수 없다(참가 record 의 슬롯이 서로 떨어져 있다) — 그래서 coalesced child 는 aligned part
// 여도 항상 전용 staging 을 거친다. 그것이 바운스와 같은 기제를 쓰면서도 **의무 단위가
// 다른** 이유다: 바운스=child 1건당 copy 1행 / coalesce=**participant record 1건당 1행**
// (SPEC_IO_METRICS_V3 §5-5).
//
// ★기본 OFF(`v3_coal_phase`) — §5 의 `io_coalesce_phase prefill` 은 **성능 후보값**이고
//   "마이크로벤치 전 동결 금지"가 걸려 있다. R12 가 `io_qd_total` 에서 같은 판단을 했다
//   (§5 의 8 을 엔진 기본값으로 굳히지 않고 기존 레버 기본값을 승계). 엔진의 기존 거동은
//   "합치지 않는다"이므로 그것이 기본값이고, `prefill` 은 명시 opt-in 이다.
// ===========================================================================


// 계획 거부 사유 — §3-6 의 7조건(+§5-4 8번째 MUST)에 1:1 로 대응한다. 네거티브 관문이
// "합쳐지지 않았다"만이 아니라 **어느 조건이 막았는지**를 주장할 수 있어야 공허하지 않다
// (A1 의 stage 래치와 같은 규율).


// 대구간 1건(= coalesced child 1건 = IO3 `coalesce_group` 1개).


// 계획 1건(제출 직전까지의 불변 스냅샷 — §5-5 "source read 제출 전에 participant 집합·copy
// obligation 계획을 불변으로 동결"). 매 pump 패스마다 재사용하는 스크래치라 힙 할당이 없다.


// ★5c-T §1 — pending op 1건에 결속되는 owner generation 스탬프(병렬 side table 원소).
//   `pending_ops[oi]` 와 **같은 인덱스**로 수명을 같이한다(pio_alloc/pio_release 에서 리셋).


// control_q 커맨드 1건. 슬롯은 고정 64개이며 생산자가 점유→dispatcher가 terminal→생산자가 반납.


// 8단계 플래너 스크래치(§2 "정적 scratch도 멤버로 이동") — legacy/영속 양쪽이 같은 타입을 쓴다.




// ★§Z-⑥ manifest 모드(SPEC_REPACK_V3 §2-1). seal 머리의 모드 게이트가 1회 확정하고, 이후
//   주소 계약 분기(`ptr()`·`part_info()`)가 이 값을 읽는다. bin 이 기본이라 v2 경로는 이 값을
//   설정하는 한 줄 외에 어떤 의미도 달라지지 않는다.


// ===========================================================================
// E-2a. New-surface store state (E2_DESIGN_CONTRACT v0.4 FROZEN - sections 0, 1, 2 row 7)
//
// Deliberately NOT a member of ModuleState. Keeping it in its own object means the
// existing struct's layout, and anything that might ever echo a size from it, does not
// move because of this atom. g_mod.mutex still guards every field that is not atomic.
// ===========================================================================

// One unique record captured by a batch token (contract section 1, token internals: the
// capture is per UNIQUE RECORD, not per requested id - a batch holds several records).
struct XStoreRecordRef {
    int64_t  record_idx      = -1;
    int32_t  slot_index      = -1;
    uint64_t slot_generation = 0;
    int32_t  multiplicity    = 0;   // how many times this record was asked for in the batch
};

// The batch token. Neither copyable nor movable: the handle table owns the object in place
// and callers never see it, which is the C-boundary form of contract L-4 (d). E-2b-0 dropped
// the move operations along with the vector that used to need them - entries are written
// where they sit now, so nothing ever relocates one.
//
// E-2b-0 lifetime split (contract section 1 v0.5):
//   gen_live  0 = this entry is FREE. Non-zero = LIVE, and the value is the handle generation
//             of the current occupant. One atomic carries both facts so a lock-free reader
//             settles liveness and identity in a single acquire-load. It doubles as the
//             "does a payload object exist in the storage below" flag; both transitions
//             happen under g_mod.mutex, and the two orderings below are chosen so that an
//             interruption between the flag and the object can only leak, never destroy
//             twice.
//   payload   mlpos / part / n_ids / store_epoch / records, held as ONE object that is
//             constructed in place before the release-store that publishes gen_live and
//             destroyed after the release-store that clears it. A LIVE entry is therefore
//             immutable, and that is what makes the read path safe without a lock.
//   next_free intrusive free list. There is no side vector to push to, so the "no push_back,
//             no resize" clause holds by construction rather than by discipline. It is NOT
//             payload: it means something only while the entry is free.
//
// E-2b-0 round 4 (Codex e2b0 r3). Round 3 made the payload `private` with four `friend`
// writers. What that proves is LEXICAL WRITER AUTHORITY - "only these four functions may
// spell these field names" - and r3 showed that this is not the invariant the contract
// asks for. A friend can take `int32_t * p = &tk.part_` before the publishing store and
// write through `p` after it, or hand `p` to a helper that never spells a private name at
// all, and both compile. Access control says WHO may write; it says nothing about WHEN.
//
// So the time invariant is expressed as a TYPE instead. The payload is a `const` object
// whose members are themselves `const`, created once by placement-new and ended by an
// explicit destructor call. Once it exists there is no non-const path to any of it anywhere
// in the program - not inside a friend, not through an escaped pointer, not through a
// template helper - because every pointer that can be formed to it is a
// `const XStorePayload *` and every member is const on top of that. Friendship is no longer
// load-bearing and is gone: this file declares no friends at all, which also means nothing
// outside this class can even name the raw storage or the free-list link.
//
// E-2b-0 round 5 (Codex e2b0 r4) narrows that claim where it was too wide. `const` proves the
// time invariant for the SCALAR members outright. For the record vector it proves it only
// together with the copy below: constness of a container does not reach elements that some
// pointer already addressed before the container acquired them. Round 4 moved the builder's
// vector into the payload, which handed the payload exactly those elements; round 5 copies
// instead, so the payload's buffer is born inside the copy constructor with no prior name.
//
// What is left for the static gate (g0_tools/e2b0_static_gate.py) is only what a type cannot
// say: that no cast or bulk-memory call launders the const away, that the builder's own
// buffer never escapes the function that fills it, that the few mutating members are called
// only where the contract allows and in the order it fixes, that every payload read is
// dominated by a liveness check, and that the read path takes no lock.

// The payload of a live token. EVERY member is const, so the object cannot be written even
// through a non-const XStorePayload * - and payload() only ever hands out a const one. An
// aggregate on purpose: with no constructor of its own, the placement-new below is a plain
// member-by-member initialisation with nothing hidden inside it.
//
// `records` is where round 4 was wrong and round 5 is careful. A const vector makes the
// ELEMENTS unreachable for writing only if nobody outside already holds a pointer INTO the
// buffer the vector owns. Moving a vector in hands the payload the caller's buffer, and a
// `data()` pointer cached before the move keeps addressing those very elements afterwards -
// no cast, no UB, ordinary C++. So this member is only ever COPY-initialised (see
// construct_payload): the buffer it owns is allocated inside std::vector's copy constructor
// and therefore did not exist, under any name, before the payload did.
//
// What that argument does NOT establish is that no writer of a live record exists at all.
// The elements of a const vector are not themselves const objects, so a pointer taken from
// records() and carried into a function this file does not constrain could still reach their
// bytes. Nothing in the type system closes that; what closes it is that every region allowed
// to name an accessor is fingerprint-pinned by the static gate, so no such carrier can be
// added without the gate going red (E2B0_ATOM_SPEC.md G-21).
// E-2b-1 perf-repair. The direct capture index, and the two thresholds that decide
// whether a token gets one.
//
// WHAT IT IS. `expert_to_ref[expert]` is the ORDINAL of that expert's record inside
// `records` - a value, not a pointer, and not slot or record STATE. NO_REF marks an
// expert this token did not capture. It answers exactly the question view step (3) asks
// and nothing else: every generation, epoch, membership, part and slot check that
// followed step (3) still runs, per call, in the same order.
//
// WHY IT IS OPTIONAL. An empty vector means "no index - scan as before". Two thresholds
// pick the path, and both exist to keep this from trading one measured cost for another:
//
//   DENSE_MAX_EXPERT  the table is O(n_expert) per LIVE token, so an unbounded table is
//                     space paid for time. At 1024 the table is 2 KiB per token and the
//                     theoretical ceiling - XSTORE_MAX_TOKENS live at once - is 128 MiB.
//                     The largest expert count in the repacker catalogue is 512, so real
//                     models sit inside this with room; anything wider keeps the scan,
//                     which is correct, merely slower.
//   DENSE_MIN_RECORDS below this the scan is at most a handful of comparisons and the
//                     table's own build cost would land on the ISSUE timer, which is a
//                     separate Level 1 gate with far less headroom than the view one.
//                     Small captures therefore keep the scan on purpose.
//
// The ordinal type also bounds the capture: a token holding NO_REF or more unique records
// cannot be indexed and falls back, so the cast below can never truncate.
static const uint16_t XSTORE_VIEW_NO_REF            = 0xFFFF;
static const int32_t  XSTORE_VIEW_DENSE_MAX_EXPERT  = 1024;
static const size_t   XSTORE_VIEW_DENSE_MIN_RECORDS = 8;

// E-2b-2R (R3). The small-return bound for ggml_moe_store_release: at or below this many lease
// units the RETURNED-ID RECONSTRUCTION uses no heap allocation - the id list is built in a stack
// buffer. ★That is the whole of the claim (Codex e2b2r_engine r2 NOTE 1): the release PATH can
// still allocate further down, inside xstore_release_common, through the persistent command's
// ids.assign and the PhaseA release hooks. Those are covered by the same fail-stop try.
//   (a) This is NOT a correctness constant. XSTORE_VIEW_DENSE_MAX_EXPERT above bounds what the
//       view can index and a wrong value there changes an ANSWER; a wrong value here changes
//       only where the bytes live.
//   (b) Crossing it is not an error and not a refusal - the token takes the SAME heap path this
//       function has always had, with the same reserve/push_back and the same fail-stop, so a
//       wider token is slower here and different in nothing else.
//   (c) 128 comes from the shapes that arrive. The section D-K Level 1 fixtures top out at
//       n_used 64, so the measured path has twice the headroom, and a real model's top-k sits
//       near 8. 128 int32_t is 512 B, a safe frame for this function. Raising it spends frame,
//       never correctness.
static const int32_t  XSTORE_RELEASE_SMALL          = 128;

struct XStorePayload {
    const uint64_t store_epoch;   // the store instance epoch this token belongs to
    const int32_t  mlpos;
    const int32_t  part;          // E-2b-0: the batch part captured at acquire
    const int32_t  n_ids;         // lease units held = sum of multiplicity
    const std::vector<XStoreRecordRef> records;  // unique records, first-appearance order
    // E-2b-1 perf-repair. Empty = this token has no direct index and the view scans
    // `records` exactly as it did before. Non-empty = size() is n_expert and each entry
    // is an ordinal into `records`, or NO_REF. Const like every other member, copied in
    // by construct_payload for the same reason `records` is, and published by the same
    // single release-store - so a reader either sees the whole payload or no payload.
    const std::vector<uint16_t> expert_to_ref;
};

// Declared here because the epoch publish path calls it thousands of lines before its
// definition and C++ has no implicit function declaration. The other three writers are
// defined ahead of their callers and need no forward declaration now that the friend
// declarations that used to name them are gone.
void xstore_table_construct_locked();

class XStoreToken {
public:
    // PUBLIC on purpose: the publish/retire flag is not payload, and every reader needs it.
    std::atomic<uint64_t> gen_live{0};

    XStoreToken()                                = default;
    XStoreToken(const XStoreToken &)             = delete;
    XStoreToken & operator=(const XStoreToken &) = delete;
    XStoreToken(XStoreToken &&)                  = delete;
    XStoreToken & operator=(XStoreToken &&)      = delete;
    // The table is deliberately never freed, but an entry that is still live when the
    // process tears its statics down would otherwise leak its record vector. gen_live is the
    // payload-exists flag, and a zero can only mean "already reclaimed", so this cannot
    // destroy the same object twice.
    ~XStoreToken() {
        if (gen_live.load(std::memory_order_relaxed) != 0) destroy_payload();
    }

    // ---- payload lifetime: these two are the only operations that exist ----------------
    // Called with the entry NOT live. The records arrive BY CONST REFERENCE and are COPIED:
    // the element buffer the payload ends up owning is allocated inside std::vector's copy
    // constructor, so it is NOT the buffer the caller filled and no pointer that existed
    // before this call can name an element of it. The object is complete from the instant it
    // exists, and nothing writable survives the call.
    //
    // There is deliberately no `vector&&` overload. Adding one would let a caller hand over
    // its own buffer again, and a `data()` pointer taken before that hand-over would still
    // address the payload's elements afterwards - which is exactly the hole this signature
    // exists to close, so the signature is part of the proof rather than a style choice.
    // E-2b-1 perf-repair adds `idx` on the same terms as `recs`: BY CONST REFERENCE and
    // COPIED. There is no rvalue overload for this one either, and for the same reason.
    void construct_payload(uint64_t epoch, int32_t ml, int32_t pt, int32_t units,
                           const std::vector<XStoreRecordRef> & recs,
                           const std::vector<uint16_t> & idx) {
        ::new (static_cast<void *>(storage_)) const XStorePayload{
            epoch, ml, pt, units, recs, idx};
    }
    // Called after the entry has been retired, so no reader can still be inside it.
    void destroy_payload() { payload()->~XStorePayload(); }

    // ---- reads. Defined only while the entry is LIVE (contract section 1) --------------
    const XStorePayload * payload() const {
        return std::launder(static_cast<const XStorePayload *>(
                                static_cast<const void *>(storage_)));
    }
    int32_t  mlpos()       const { return payload()->mlpos; }
    int32_t  part()        const { return payload()->part; }
    int32_t  n_ids()       const { return payload()->n_ids; }
    uint64_t store_epoch() const { return payload()->store_epoch; }
    const std::vector<XStoreRecordRef> & records() const { return payload()->records; }
    // E-2b-1 perf-repair. Empty means "no index"; see XStorePayload.
    const std::vector<uint16_t> & expert_to_ref() const { return payload()->expert_to_ref; }

    // ---- free-list link. Not payload: it is read and written only while the entry is free,
    //      always under g_mod.mutex, and never by the lock-free read path.
    int32_t next_free() const { return next_free_; }
    void    set_next_free(int32_t v) { next_free_ = v; }

private:
    // Storage for the payload. An array of uint64_t rather than an alignas'd byte array, so
    // the alignment comes from the element type: there is no alignment specifier that could
    // make the compiler pad the class, and the static_assert below turns the alignment claim
    // into something the compiler checks instead of something this comment asserts.
    uint64_t storage_[(sizeof(XStorePayload) + sizeof(uint64_t) - 1) / sizeof(uint64_t)];
    int32_t  next_free_ = -1;   // free-list link, -1 = end
};

static_assert(alignof(XStorePayload) <= alignof(uint64_t),
              "XStorePayload wants stricter alignment than XStoreToken's storage provides");

// Adapter map entry. Published at seal success, invalidated wholesale at teardown
// (contract section 1, adapter map lifecycle stages 2 and 4).
struct XStoreAdapterEntry {
    int32_t  mlpos    = -1;
    int32_t  part_idx = -1;
    uint64_t epoch    = 0;
};

// The logical key a resolution yields. Both surfaces reach the lease machinery through it.
struct XStoreAcquireKey {
    int32_t mlpos    = -1;
    int32_t part_idx = -1;
};

// Handle-table capacity. The index occupies 16 bits of the handle, one of which is spent on
// the "+1" that keeps 0 out of the live space.
const int32_t XSTORE_MAX_TOKENS = 65534;

struct XStoreState {
    // Handle table. handle = (generation << 16) | (index + 1), so 0 is never a live handle
    // and the index can never be mistaken for one. 48 generation bits are used because a
    // 32-bit generation is reachable by a long-lived server (a few thousand acquires per
    // second wraps in days), and a wrapped generation on a reused index is a silent alias.
    //
    // E-2b-0 (contract section 1 v0.5): the table is ONE array of XSTORE_MAX_TOKENS entries,
    // built once before the first epoch is published and never grown, shrunk or moved. It was
    // a std::vector that push_back'ed a new entry per unseen index; that meant the storage
    // could relocate under a reader, which is the whole reason the read path had to hold the
    // module lock. `table_own` owns it; `table` is the pointer a lock-free reader loads, and
    // it is published with a release-store so the reader either sees no table at all or sees
    // a fully constructed one. reset_for_test frees the ENTRIES and keeps the STORAGE - the
    // address stability is the point of the clause, not an implementation convenience.
    std::unique_ptr<XStoreToken[]> table_own;
    std::atomic<XStoreToken *>     table{nullptr};
    std::atomic<int32_t>           table_n{0};      // 0 until built, XSTORE_MAX_TOKENS after
    // Intrusive free list through XStoreToken::next_free. Nothing to push to, nothing to
    // resize; both are held under g_mod.mutex like every other allocation-side field.
    int32_t                        free_head  = -1;
    int32_t                        free_count = 0;
    int32_t                        live_count = 0;
    // Builder scratch for xstore_token_issue_locked, and NOT payload. A token's records are
    // assembled here and then COPIED into the payload, so this vector keeps its own buffer
    // and its own capacity: cleared rather than freed on every issue, it stops allocating
    // once its capacity has reached the largest n_ids seen so far. Touched under g_mod.mutex,
    // and in the current source only inside that one function - which the static gate holds
    // in place by pinning that function's text rather than by reasoning about the name.
    std::vector<XStoreRecordRef>   build_scratch;
    // E-2b-1 perf-repair. The second half of the same builder: the direct capture index
    // is assembled here and COPIED into the payload by the same construct call, so the
    // buffer the payload owns is again one that no pointer named before the payload
    // existed. Same lock, same lifetime, same reasoning as build_scratch above - it is
    // deliberately a sibling rather than a member of it, because the records vector is
    // handed to release verbatim and must keep its exact element type.
    std::vector<uint16_t>          build_index_scratch;
    // Generations are process-monotone ON PURPOSE: reset_for_test does NOT reset them, so a
    // handle minted before a reset can never alias one minted after it.
    uint64_t                 gen_next = 1;

    std::unordered_map<const ggml_tensor *, XStoreAdapterEntry> adapter;

    uint64_t epoch_seq = 0;   // monotone across seal / teardown / reset_for_test
};

XStoreState g_xs;

// (a) live batch tokens. A NEW-surface diagnostic only: it does not stand in for old-surface
// leases, because the old surface mints no token at all (contract section 2 row 7).
std::atomic<int64_t> g_xs_handles{0};
// (b) the lease-unit ledger, spanning BOTH surfaces. It is maintained at the four sites
// where Slot::lease itself moves, so "granted - returned" equals Sigma Slot::lease BY
// CONSTRUCTION and the 3-way comparison stays a real cross-check instead of a restatement
// of one number in two places.
std::atomic<int64_t> g_xs_units_granted{0};
std::atomic<int64_t> g_xs_units_returned{0};
// The live store instance epoch (0 = none).
std::atomic<uint64_t> g_xs_epoch{0};
// Lifecycle state, one of MOE_STORE_*. RUNNING is the only value that existed before this
// atom, so the RUNNING path pays exactly one acquire-load per acquire call.
std::atomic<int32_t> g_xs_state{MOE_STORE_RUNNING};

// (c) admissions in flight: acquires that have passed the RUNNING gate and have not yet
// finished. The lease-unit ledger alone cannot close the drain, because an acquire that has
// read RUNNING but has not yet reached the lease still owes one - it holds NOTHING the
// ledger can see, and a ledger-only zero-confirm would stop the store underneath it. The
// two sides use the standard store-then-load / increment-then-load pair, which needs
// sequential consistency to work at all: whichever way the two interleave, at least one of
// them observes the other.
std::atomic<int64_t> g_xs_admit{0};

// One acquire's admission. Increment FIRST, then read the state: an admission taken before
// the drain flipped the state is visible to the drain, and one attempted after the flip
// reads DRAINING and backs out. Both surfaces go through this because both surfaces go
// through xstore_acquire_common().
struct XStoreAdmission {
    bool admitted = false;
    XStoreAdmission() {
        g_xs_admit.fetch_add(1, std::memory_order_seq_cst);
        if (g_xs_state.load(std::memory_order_seq_cst) == MOE_STORE_RUNNING) {
            admitted = true;
        } else {
            g_xs_admit.fetch_sub(1, std::memory_order_seq_cst);
        }
    }
    ~XStoreAdmission() {
        if (admitted) g_xs_admit.fetch_sub(1, std::memory_order_seq_cst);
    }
    XStoreAdmission(const XStoreAdmission &)             = delete;
    XStoreAdmission & operator=(const XStoreAdmission &) = delete;
};

// The drain's ONE deadline is anchored here, at drain_begin, and NOT at drain_finish
// (contract section 2 row 7: a single deadline measured from the DRAINING entry). Every
// stage below spends what is LEFT of that one budget - a fresh full budget per stage is how
// a "30 s drain" quietly becomes 30 s of waiting per stage. -1 = no drain in progress.
std::atomic<int64_t> g_xs_drain_t0_ms{-1};

inline int64_t xstore_steady_ms() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Milliseconds left of the single deadline, for a caller-supplied deadline_ms.
//   deadline_ms >  0 : the budget IS deadline_ms, measured from the drain_begin instant
//   deadline_ms <= 0 : no budget at all - check once and fail. NEVER the 30000 default:
//                      substituting it there is precisely how "check once" turned into a
//                      30 s dispatcher wait in the first cut.
// A drain that was never begun has no anchor and therefore no budget.
inline int32_t xstore_drain_remaining_ms(int32_t deadline_ms) {
    if (deadline_ms <= 0) return 0;
    const int64_t t0 = g_xs_drain_t0_ms.load(std::memory_order_acquire);
    if (t0 < 0) return 0;
    const int64_t left = (int64_t)deadline_ms - (xstore_steady_ms() - t0);
    if (left <= 0) return 0;
    return (int32_t)left;
}

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// E-2a (Codex r2 blocker 2) - carrying the deadline as ONE absolute instant. r1's anchor got as
// far as "drain_finish computes what is left and hands it over", and left the reuse INSIDE the
// helper it handed it to: the predictor's idle wait, the predictor worker join and the
// dispatcher join each took that number as a fresh full budget, so a remaining R could cost
// about 3R. These two close that: the budget travels as an instant, and every stage asks only
// what is left of it at its own entry.
//   deadline_abs_ms <  0 : no budget bound at all = INFINITE - reported as -1. E-2b-0
//                          (durable (g)): this used to say "the production stop policy". It
//                          is not - E-2a made the production dispatcher stop FINITE
//                          (MOE_DRAIN_DEADLINE_MS, contract section 2 row 7). The unbounded
//                          value survives for the module's own internal wrappers only.
//   deadline_abs_ms >= 0 : steady_clock milliseconds. Already past = 0 left = check once, do
//                          not wait.
int32_t xstore_budget_left_ms(int64_t deadline_abs_ms) {
    if (deadline_abs_ms < 0) return -1;
    const int64_t left = deadline_abs_ms - xstore_steady_ms();
    if (left <= 0)         return 0;
    if (left > 0x7fffffff) return 0x7fffffff;
    return (int32_t)left;
}
// Pin a caller's relative timeout to an absolute instant at entry (negative stays INFINITE).
int64_t xstore_budget_deadline_ms(int32_t timeout_ms) {
    if (timeout_ms < 0) return -1;
    return xstore_steady_ms() + (int64_t)timeout_ms;
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// The drain's single deadline as an instant. It comes STRAIGHT off the anchor, so nothing
// anywhere down the chain re-derives it - the same anchor xstore_drain_remaining_ms reads,
// expressed as an end rather than a remainder.
//   deadline_ms <= 0, or no anchor = budget 0. Returning 0 (an instant long past on the steady
//   clock) makes every stage read "nothing left"; it can never be mistaken for the negative
//   value that means INFINITE.
inline int64_t xstore_drain_deadline_ms(int32_t deadline_ms) {
    if (deadline_ms <= 0) return 0;
    const int64_t t0 = g_xs_drain_t0_ms.load(std::memory_order_acquire);
    if (t0 < 0) return 0;
    return t0 + (int64_t)deadline_ms;
}

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// E-2a (Codex r2 blocker 2) - what the dispatcher-join stage measured against, and what it had
// left at its own entry (test observation only; nothing branches on it). Together with the two
// stages the predictor reports through ggml_moe_phaseb_test_drain_stages, a gate can read
// whether all three stages named ONE instant, and whether each of them was entered at all.
// Scoped to the LAST stop, not the process: every stop clears these on entry, exactly as the
// predictor drain clears its own stage record. A process-wide tally would be useless to a gate
// here - a full suite run performs hundreds of stops through reset_for_test before any of
// these checks run, so "has this stage ever been entered" is always yes and answers nothing.
std::atomic<int64_t> g_xs_stopjoin_deadline{-1};
std::atomic<int32_t> g_xs_stopjoin_left{-1};

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// E-2b-0 - state behind the test-only module-mutex hold. A gate cannot prove "the view read
// path does not take g_mod.mutex" by reading code from inside the process; it can prove it by
// holding that mutex on one thread and watching a view finish on another. These three fields
// live under their OWN mutex, never g_mod.mutex, so the release signal can always get through
// to a holder that is sitting on the module lock - the lock order is one-way by construction.
std::mutex              g_xs_hold_mu;
std::condition_variable g_xs_hold_cv;
bool                    g_xs_hold_held   = false;   // the module lock is actually held
bool                    g_xs_hold_go     = false;   // release was requested
int32_t                 g_xs_hold_safety = 0;       // 1 = the cap ended it, not a request
}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
std::atomic<int32_t> g_xs_stopjoin_n{0};      // 0 = the last stop did not reach the join

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
inline void xstore_note_lease_granted(int32_t units) {
    if (units > 0) g_xs_units_granted.fetch_add((int64_t)units, std::memory_order_relaxed);
}
}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
void xstore_note_lease_returned(int32_t units) {
    if (units > 0) g_xs_units_returned.fetch_add((int64_t)units, std::memory_order_relaxed);
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
inline int64_t xstore_outstanding_units() {
    return g_xs_units_granted.load(std::memory_order_relaxed)
         - g_xs_units_returned.load(std::memory_order_relaxed);
}

// The drain's zero-confirm. The ADMISSIONS ARE READ FIRST and the order is load-bearing: an
// admitted acquire adds its lease unit to the ledger before it drops its admission, so
// reading the ledger first and the admissions second can see 0 and 0 with a lease still on
// its way in. Reading the admissions first gives the opposite, safe implication - once the
// count is 0 no admitted acquire is in flight, and no new one can be admitted because the
// state is no longer RUNNING, so the ledger read that follows is final. The seq_cst
// read-modify-writes on the admission counter are what make the ledger write that precedes
// the decrement visible to the ledger read that follows this load.
inline bool xstore_drain_zero() {
    if (g_xs_admit.load(std::memory_order_seq_cst) != 0) return false;
    return xstore_outstanding_units() == 0;
}


// ===========================================================================
// §3. 모듈 전역 상태 (v2)
// ===========================================================================





// ---------------------------------------------------------------------------
// E-2a. Store instance epoch and adapter map lifecycle
//   contract section 0 item 4 (StoreInstanceEpoch) and section 1 (adapter map, 4 stages)
// ---------------------------------------------------------------------------

// Stage 2, publish. Called with g_mod.mutex held, immediately before sealed=true, on BOTH
// seal success boundaries (bin and virtual). The map is the adapter's own copy of the
// resolution the core already performed; the core keeps `registry` until E-2c retires the
// tensor* surface, so in THIS atom the two coexist by design and die together.
void xstore_publish_epoch_locked() {
    // E-2b-0 (contract section 1 v0.5): the handle table is built to its full fixed size
    // BEFORE the first epoch is published, which is what "pre-constructed" means - after this
    // line a handle can exist, and from here on the table never moves. Idempotent, so the
    // second and every later seal reuse the same storage.
    xstore_table_construct_locked();
    g_xs.epoch_seq++;                       // monotone - never restarts, not even on reset
    g_xs.adapter.clear();
    g_xs.adapter.reserve(g_mod.registry.size());
    for (const auto & kv : g_mod.registry) {
        XStoreAdapterEntry e;
        e.mlpos    = kv.second.mlpos;
        e.part_idx = kv.second.part_idx;
        e.epoch    = g_xs.epoch_seq;
        g_xs.adapter[kv.first] = e;
    }
    // Stage 3, the reachable part of it in this atom: the registration staging vector stops
    // holding caller tensor pointers past seal (K-2 counterexample (a)). Every reader of
    // `pending` is inside seal itself and has already run by the time this line executes -
    // the readers are register(), the two seal bodies, and reset_for_test, and nothing else
    // in the translation unit touches it - so no input is lost by clearing it here.
    // `registry` still keys on tensor*: the OLD surface needs it, and retiring that is E-2c.
    g_mod.pending.clear();
    // A newly sealed instance is RUNNING and has no drain in progress, so the previous
    // instance's deadline anchor must not survive into it.
    g_xs_drain_t0_ms.store(-1, std::memory_order_release);
    g_xs_state.store(MOE_STORE_RUNNING, std::memory_order_release);
    g_xs_epoch.store(g_xs.epoch_seq, std::memory_order_release);
}

// Stage 4, invalidate. The adapter map dies wholesale and the live epoch becomes 0, so every
// token and every part ordinal minted under it is out of epoch from here on. Tokens are NOT
// dropped from the handle table: a release afterwards must be able to say "outside its
// epoch" rather than degrade into the weaker "unknown handle".
void xstore_invalidate_epoch_locked() {
    g_xs.adapter.clear();
    g_xs_epoch.store(0, std::memory_order_release);
}

}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// Full reset - reset_for_test only. Every handle table ENTRY is freed. Handle generations are
// NOT reset, so a handle minted before the reset can never alias one minted after.
//
// E-2b-0: the STORAGE survives. "The table is built once and never moves" would be a much
// weaker claim if it only held between resets, and a full suite run resets hundreds of times;
// keeping the array is what lets a gate assert one address across the whole run.
void xstore_reset_locked() {
    xstore_invalidate_epoch_locked();
    XStoreToken * t = g_xs.table.load(std::memory_order_relaxed);
    if (t != nullptr) {
        const int32_t n = g_xs.table_n.load(std::memory_order_relaxed);
        g_xs.free_head = -1;
        for (int32_t i = n - 1; i >= 0; i--) {
            // Retire first, then reclaim - the same order the single-entry free uses, so the
            // lifetime reads the same way here. Only a LIVE entry owns a payload object, and
            // gen_live is the flag that says so, so the destructor call is conditional while
            // the retirement is not.
            const uint64_t gl = t[(size_t)i].gen_live.load(std::memory_order_relaxed);
            t[(size_t)i].gen_live.store(0, std::memory_order_release);
            if (gl != 0) t[(size_t)i].destroy_payload();
            t[(size_t)i].set_next_free(g_xs.free_head);
            g_xs.free_head = i;
        }
        g_xs.free_count = n;
        g_xs.live_count = 0;
    }
    g_xs_handles.store(0, std::memory_order_relaxed);
    g_xs_units_granted.store(0, std::memory_order_relaxed);
    g_xs_units_returned.store(0, std::memory_order_relaxed);
    g_xs_drain_t0_ms.store(-1, std::memory_order_release);
    g_xs_state.store(MOE_STORE_RUNNING, std::memory_order_release);
    // g_xs_admit is deliberately NOT reset. It is owned by live XStoreAdmission objects on
    // other stacks; zeroing it here would make their destructors drive it negative, and a
    // permanently non-zero admission count would then wedge every later drain.
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes


// ★§Z-⑥ A2 주소 계약(SPEC_REPACK_V3 §3-3) — 슬롯 안에서 이 part 의 실데이터가 시작하는
//   offset 과 per-expert 크기. bin=`part_offset`/`part_bytes`(v2 승계·의미 불변) ·
//   virtual=`data_offset`/`slice_bytes`. **분기는 이 함수 하나뿐**이고 `ptr()`·`part_info()`·
//   A2 자원 프로브가 전부 여기를 통과한다(같은 계약을 세 벌로 구현하지 않는다 — §2-8).


// ★§Z-⑥ A2/A3 — mode 분기와 층별 part 수. **정의는 여기 한 벌**이다(구 판은 §11 child
//   팬아웃 절에 있었는데, PB virtual A1 의 ①②축 상태기계가 그보다 앞 절[§10.5 pending 풀]
//   에서 이 둘을 쓰므로 공통 유틸리티로 끌어올렸다 — 거동 불변·선언 이동뿐).


// 이 층의 part 수 = 한 record 가 내는 child(물리 read) 수 = 사양의 membership `P_l`.


// ===========================================================================
// §4. 유틸리티
// ===========================================================================













// ★F2(전체 경로 비교 복원 — v1 KS1 필수2 계약): manifest 경로가 절대경로인지.
// 드라이브 형식은 "C:\..."까지 요구 — "C:relative"는 drive-relative(현재 디렉터리 기준)라 거부.


// ===========================================================================
// §5. 런타임 섹터 재질의 (IOCTL_STORAGE_QUERY_PROPERTY — v1 승계).
// ===========================================================================



// 전체 경로 정규화 비교(절대경로 변환 + Windows 대소문자 비민감).












// ===========================================================================
// §6. Manifest v2 파싱·검증 (REPACK_V2_DESIGN §3 불변식 8항 + §4-1 전항).
//     크래시 없이 err 수집. 성공 시 out 채움.
// ===========================================================================



// per-expert slice = (ne0/bv)*bb*prod(dims[1..last-1]). 표 밖·ne0%bv!=0 = false.







// ★§2-3 승계 필드 — `model` 블록의 계약은 v2·v3 **공통**이다(모드와 무관: arch·n_layer·
//   n_expert·n_expert_used·moe_layers·routed_scope + live 전항 대조 + scope 정규화).
//   두 검증기가 같은 계약을 두 벌로 구현하면 개정 때 한쪽만 낡는다(§2-8 문면 사본 드리프트가
//   실증한 병) — 그래서 한 벌로 두고 v2·v3 가 함께 부른다. err 는 접두 없이 사유만 담고,
//   호출자가 자기 접두("manifest vN validation failed: ")를 붙인다.


// v2·v3 공통 — `sources[i]` 실물 identity 대조(§2-5 ① · v2 §4-1 승계). digest(DF-1)는 virtual
// 전용이라 여기 들어오지 않는다(호출자가 따로 본다).


// v2·v3 공통 — quant_traits = 감사용 복제(불변식 7). 동결 표와 정확 일치만 수용.


// v2·v3 공통 — reference_lock 필드 무결(불변식 8). 소비자는 필드 무결만 보고, 카탈로그 대조는
// seal 의 F1 결속이 한다.






// ===========================================================================
// ★§Z-⑥ 원자 A1 — mode=virtual(v3) manifest 로더 · 부팅 fail-closed · init 거부 4조합
//
// 사양 권위: `SPEC_REPACK_V3.md` §2-1(모드·스키마)·§2-2(virtual 좌표계·bin 필드 금지)·
//   §2-4(슬롯 산술·checked uint64·bracket EOF)·§2-5(부팅 대조 5항)·§3-7(등록·A_runtime 승계)·
//   §4-3(plan_report 소비 결속) / `SPEC_IO_METRICS_V3.md` §3(virtual init 거부 4조합).
//
// ★bin 경로는 이 절의 어떤 함수도 호출하지 않는다 — 유일한 접점은 seal 머리의 모드 분기
//   (detect_manifest_mode) 한 줄이고, v2 manifest 는 그 분기에서 종전 경로로 그대로 빠진다.
// ★A1 의 천장: 이 절의 함수들은 **검증까지**다. 자원 확보(소스 핸들 집합·슬롯 풀·registry·
//   보호)는 A2 이고 child 상태기계는 A3 이므로, 전 관문을 통과한 v3 manifest 도 마지막에
//   명시적으로 기동 거부된다(`seal_virtual_locked` 말미의 현행 경계). 조용한 성공으로 빠지는
//   경로는 없다.
// ===========================================================================

// A1 진행 단계 래치(테스트 전용 관측 창구 — 프로덕션 판단에 쓰지 않는다). 네거티브 케이스가
// "거부됐다"가 아니라 "**의도한 관문에서** 거부됐다"를 주장할 수 있어야 각 거부 사유가 실제로
// 행사된 것이 된다(rc<0 만으로는 앞 관문이 우연히 잡은 것과 구분되지 않는다).


// R-B3b: `g_last_seal_err` moved to ggml-moe-direct-test-entry.cpp (RB3B_ATOM_SPEC.json ::
// design_type_list) - its only readers and writers are two test entry points there.

// §2-4 "전 산술 = checked uint64" — 오버플로는 조용히 wrap 하지 않고 기동 중단이 된다.






// §2-1 모드·스키마 게이트. bin = schema "2.0" ∧ mode 필드 **부재**(v2 바이트 규약 그대로) /
// virtual = schema "3.0" ∧ mode "virtual". `"3.0"+mode:"bin"` 은 생산 금지 조합이라 소비도
// 거부한다. mode 누락(3.0에서)·미지 값·모드↔스키마 불일치 전부 기동 중단.


// SPEC_IO_METRICS_V3 §3 — virtual 모드 거부 조합(지위: ★한시. 해제 경로=§3-2 승격 심의).
// 어느 하나라도 성립하면 기동 중단이며 stderr 에 사유 1줄을 남긴다(silent 무시 금지).
//   ① PHASEA != off  — PA `ops.file_offset` 이 experts.bin 좌표를 전제한다
//   ② PHASEB != off  — `pb_io` 가 "물리 op 1건 = record 1건 · stride_bytes" 를 전제한다
//   ③ IO_PATH=legacy — legacy pump 는 qualification 전용 대조군
//   ④ MOE_P1_TRACE=1 — O1/O2 가 "물리 op = 단일 layer/expert/slot/bytes" 를 전제하고 판독기가
//      `Σ O2 bytes == phase read_bytes` 를 하드 검사한다
// ★프리페치 **런타임 기능**은 거부 대상이 아니다(§3 말미) — 거부는 계측 wire·qualification
//   대조 경로에만 걸린다.


// ★§Z-⑥ 원자 경계 게이트(구 `v3_atom_env_boundary`) — **철거됨**(PB virtual 원자 A8).
//   이 게이트는 "이 원자가 아직 안 만든 것"을 조용히 통과시키지 않으려고 있었고, 해당 원자가
//   서면 사라지는 성격이었다(SPEC_IO_METRICS_V3 §3 이 동결한 **계약상 거부 4조합**과는 별개 —
//   그쪽은 `virtual_init_reject_check` 가 계속 소유한다).
//     ⓐ `MOE_DIRECT_METRICS` = R13 에서 해제(IO3 사이드카가 물리 행 채널을 가졌다).
//     ⓑ ★`MOE_DIRECT_PREFETCH_K/N` = **여기서 해제**. 재-fail-close 의 근거는 "virtual seal 이
//        `resolve_prefetch`/`note_seal` 을 부르지 않아 prefetch-origin op 자체가 생기지 않는다 ⇒
//        env 수용은 무경고 no-op" 이었다. 원자 A5 가 그 배선을 실제로 넣었고(seal ⓓ‴ 블록),
//        A1~A4 가 admission·회계·처분·취소를 virtual 축으로 옮겼으므로 그 전제가 소멸했다.
//        권위 = `SPEC_PB_VIRTUAL_RUNTIME_R1.md` §2-0(FROZEN v1.0) · §6 원자 A8.
//   ⇒ 남은 조항이 없어 함수 자체를 지운다(빈 게이트를 남기면 "무언가 막고 있다"는 오독을 만든다).

// §2·§2-4 mode=virtual manifest 검증. bin 좌표 필드 금지 + 슬롯 산술 자기일관 재계산 +
// records[]/source_tensors[] 비권위 witness 전항 재생성 대조.


// §4-3 소비 결속 — `plan_report.json`(단일 JSON 객체. v2 의 verify_report 는 JSONL 이었다).
// 통과 조건(하나라도 어긋나면 fail-close):
//   ① mode == "virtual"(타 모드 리포트 재사용 차단) ② pass 가 boolean true
//   ③ problems 가 존재하는 **빈** 배열(생산 측 pass ⟺ problems 공집합의 소비자 측 재확인)
//   ④ manifest_sha256 이 64자리 hex(실 manifest 바이트와의 대조는 호출자 몫 — 경로를 아는 쪽)
//   ⑤ reference_lock.{profile_id,expect_sha256} 추출(manifest 대조는 호출자 몫)
//   ⑥ cardinality.records 정수(totals.n_records 대조는 호출자 몫)
// 스키마 권위 = bench/repack/repack_experts.py verify_virtual_manifest() 반환 dict.


// §2-5 ③ + DF-1 — shard 헤더를 **한 번만** 읽어 두 의무를 함께 갚는다:
//   ⓐ [0, data_start) 재해시 → manifest sources[].digest 대조(정상 교체·bitrot 검출)
//   ⓑ 같은 버퍼에서 GGUF 헤더를 재파싱 → live 텐서 좌표(abs_offset·ne·type·bytes) 재도출 후
//      manifest 전항 대조
// ★이 대조의 전부는 "live 값의 출처가 manifest 가 아니라 **파일 자신**"이라는 데 있다 — 모델
//   재다운로드·shard 교체로 주소가 어긋나면 조용한 오독 대신 기동 실패가 된다(§2-5 취지).
// ★텐서 전역 유일성도 함께 본다: 어떤 part 의 텐서명이 선언된 shard **밖**에서도 발견되면
//   source_index 교환이 주소 우연 일치로 통과할 수 있으므로 거부한다.
// out_live_type = 텐서명 → live ggml_type(등록 텐서 대조에 쓴다).


// ===========================================================================
// §7. 파일 읽기 계약 (ReadFile/OVERLAPPED/GetOverlappedResult — v1 승계) +
//     ★QD 레버(비동기 다중 outstanding — reviews/codex_qd_u6_design_xcheck.md 심의 3).
//     want_bytes = 해당 층 stride(층별). 트레이스 layer = moe_layers[mlpos].
//     구 read_record_sync(record 1개를 ReadFile → GetOverlappedResult(TRUE)로 동기 완주)는
//     아래 배치 적재기 run_batch_ios_locked로 대체됐다 — 한 acquire의 유니크 miss 전량을 QD
//     상한까지 동시에 띄우고 wait-any로 회수한다. 호출자는 전 구간 g_mod.mutex를 보유한다
//     (락 재설계 없음 — OVERLAPPED 완료는 이 mutex를 필요로 하지 않아 자체 deadlock이 없다).
// ===========================================================================

// hit 계수(구 ensure_locked hit 분기와 동일 의미 — trace ON일 때만 exact/window 갱신).
// ★QD 레버: 슬롯 회계 자체는 그대로 두고 호출 위치만 배치 플래너 ④단계로 옮겼다.
void count_hit_locked(Phase phase) {
    g_mod.phase[(int)phase].hit.fetch_add(1, std::memory_order_relaxed);
    if (!ggml_moe_trace_enabled()) return;
    uint16_t ex_run_index; uint32_t ex_graph_seq; uint8_t ex_phase;
    ggml_moe_trace_current_graph_context(&ex_run_index, &ex_graph_seq, &ex_phase);
    const bool window_active = ggml_moe_trace_window_active();
    if (ex_phase == MOE_TRACE_PHASE_DECODE) {
        g_mod.phase_exact[(int)Phase::DECODE].hit.fetch_add(1, std::memory_order_relaxed);
        if (window_active) g_mod.phase_exact_window[(int)Phase::DECODE].hit.fetch_add(1, std::memory_order_relaxed);
    } else if (ex_phase == MOE_TRACE_PHASE_PREFILL) {
        g_mod.phase_exact[(int)Phase::PREFILL].hit.fetch_add(1, std::memory_order_relaxed);
        if (window_active) g_mod.phase_exact_window[(int)Phase::PREFILL].hit.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_mod.phase_ctx_unknown_count.fetch_add(1, std::memory_order_relaxed);
    }
}

// miss 계수(구 ensure_locked 미스 분기 말미와 동일 의미). trace_phase = claim 시점(구 구현의
// data_wait_begin 시점과 같은 위치)에 채취한 그래프 문맥 phase.
void count_miss_locked(Phase phase, uint64_t stride, uint8_t trace_phase) {
    g_mod.phase[(int)phase].miss.fetch_add(1, std::memory_order_relaxed);
    g_mod.phase[(int)phase].read_count.fetch_add(1, std::memory_order_relaxed);
    g_mod.phase[(int)phase].read_bytes.fetch_add(stride, std::memory_order_relaxed);
    if (!ggml_moe_trace_enabled()) return;
    if (trace_phase == MOE_TRACE_PHASE_DECODE || trace_phase == MOE_TRACE_PHASE_PREFILL) {
        const int pidx = trace_phase == MOE_TRACE_PHASE_DECODE ? (int)Phase::DECODE : (int)Phase::PREFILL;
        PhaseCounters & pc = g_mod.phase_exact[pidx];
        pc.miss.fetch_add(1, std::memory_order_relaxed);
        pc.read_count.fetch_add(1, std::memory_order_relaxed);
        pc.read_bytes.fetch_add(stride, std::memory_order_relaxed);
        if (ggml_moe_trace_window_active()) {
            PhaseCounters & pcw = g_mod.phase_exact_window[pidx];
            pcw.miss.fetch_add(1, std::memory_order_relaxed);
            pcw.read_count.fetch_add(1, std::memory_order_relaxed);
            pcw.read_bytes.fetch_add(stride, std::memory_order_relaxed);
        }
    } else {
        g_mod.phase_ctx_unknown_count.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// ★B1 — 회수 1건의 publish 본문(legacy 배치 pump와 영속 dispatcher 공용).
// byte 검증 → O2 → O4(★publish 이전 — buffer 사용 가능 시점) → lease/slot_generation 확정 →
// slot_index release-store(publish) → O7(publish 후) → 계수.
// ★C2: lease 반영은 **성공 terminal에서만**(join 시점 아님) — aggregate multiplicity를 여기서 1회.
// ★B2 §2.3(Phase A 스코프 보존): origin=prefetch 제출/회수는 **demand 스코프 채널을 일절
//   건드리지 않는다** — moe_readfile_digest/ops·PA ops 사이드카·PA io_qd 피드·phase hit/miss·
//   read_bytes·moe-trace 이벤트는 전부 demand-origin 가드 아래 있다. all-origin 관측은 PB 채널
//   (moe_pb_readfile_digest/ops·pb_io)이 담당한다.
void io_finish_locked(BatchMiss & m, DWORD bytes, bool was_pending, int32_t mlpos, Phase phase,
                      uint8_t origin = PIO_ORIGIN_DEMAND, uint32_t want_override = 0) {
    const LayerInfo & L = g_mod.layers[(size_t)mlpos];
    // ★§Z-⑥ A3 — bin 은 물리 read 1건 = stride_bytes. virtual 은 child N건의 합이 그 층의
    //   슬롯 점유(= Σ region = layer_slot_bytes)다. 호출자(v3_child_retire_locked)가 이미
    //   child 별 길이를 확인하고 합을 넘기므로 여기서는 그 합의 계약만 다시 선다.
    const bool virt = (g_mod.manifest_mode == ManifestMode::VIRTUAL);
    // ★A4 — virtual 의 기대 바이트는 **child 가 슬롯에 남긴 합**이다(`child_bytes_expected`):
    //   aligned=region(브라켓 전량 착지) · 비4K=slice_bytes(브라켓은 staging 으로 가고 copy 가
    //   슬라이스만 옮긴다). A3 까지는 전 part 가 aligned 라 `layer_slot_bytes` 와 같았고,
    //   혼합 층에서 처음 갈린다(그 차이가 A4 의 첫 실측 실패였다).
    // ★A6 — coalesced part 는 브라켓이 아니라 slice 만 슬롯에 남기므로 층 상수와 갈린다.
    //   그래서 기대치는 **그 op 이 제출 시점에 확정한 값**이 권위다(0=미지정=층 상수).
    const uint32_t want_bytes = want_override != 0 ? want_override
                              : (uint32_t)(virt ? L.child_bytes_expected : L.stride_bytes);
    const bool demand   = (origin == PIO_ORIGIN_DEMAND);
    const bool trace_on = demand && ggml_moe_trace_enabled();
    const bool pa_on    = demand && g_mod.phasea_on;
    if (bytes != want_bytes)
        fatal_abort("short read: got=%lu want=%u offset=%llu",
                    (unsigned long)bytes, want_bytes, (unsigned long long)m.file_offset);
    if (trace_on) {
        // 배선 S12(v4) — O2에 O1과 동일 slot_index 탑재(O1↔O2 slot 일치 검사용, SPEC_G32C §8 U6').
        ggml_moe_trace_io_complete(m.trace_op_id, MOE_TRACE_IO_STATUS_OK, (uint32_t)bytes, was_pending, (uint32_t)m.slot);
        ggml_moe_trace_data_wait_end(m.trace_wait_episode);
    }
    Slot & s = g_mod.slots[(size_t)m.slot];
    RecordEntry & re = g_mod.record_table[(size_t)m.record_idx];
    s.lease.fetch_add(m.multiplicity, std::memory_order_relaxed);
    xstore_note_lease_granted(m.multiplicity);   // E-2a lease-unit ledger (b)
    re.slot_generation.store(m.generation, std::memory_order_relaxed);
    re.slot_index.store(m.slot, std::memory_order_release);
    // ★B2 상태기계 기본값 = READY_DEMAND. waiter 없는 speculative publish만 호출자
    //   (disp_op_terminal_locked)가 READY_PROBATION으로 덮어쓴다(§4).
    s.cache_class       = GGML_MOE_PB_CLASS_DEMAND;
    s.pf_intent_ordinal = GGML_MOE_PHASEB_NA_U64;
    s.pf_rank           = GGML_MOE_PHASEB_NA_U16;
    s.pf_io_index       = -1;
    pb_slot_stamp_clear(s);   // ★5c-T §1
    if (trace_on) {
        ggml_moe_trace_ready_publish(m.trace_run_index, m.trace_graph_seq, m.trace_phase,
                                      m.trace_layer, MOE_TRACE_NA_U8, m.trace_expert,
                                      (uint32_t)(m.generation & 0xFFFFFFFFull));
    }
    if (demand) {
        if (was_pending) g_mod.read_pending_count.fetch_add(1, std::memory_order_relaxed);
        else             g_mod.read_immediate_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (pa_on) {
        // ★publish 직후 = buffer 사용 가능 시점(op record의 retire).
        ggml_moe_phasea_io_op(m.record_idx, m.file_offset, want_bytes, m.slot, m.pa_batch_order,
                              m.pa_submit_seq, m.pa_submit_qpc,
                              g_mod.pa_retire_seq_next++, ggml_moe_phasea_now_qpc(),
                              was_pending ? 1 : 0);
    }
    // ★A3 — 회계 축은 물리 바이트가 아니라 legacy-equivalent 다(§7·§3-8). bin 은 그 둘이
    //   같아서(stride_bytes) 구분이 없었고, virtual 에서 처음 갈린다.
    if (demand) count_miss_locked(phase, virt ? L.legacy_record_bytes : L.stride_bytes, m.trace_phase);
}

// ★B1 — 제출 1건의 본문(legacy 배치 pump와 영속 dispatcher 공용). batch_order = PA 배치 내 순번.
// 반환 true = ReadFile이 동기 완료(TRUE 반환) → 호출자가 그 자리에서 GetOverlappedResult+회수.
// ★ordered ReadFile digest(Phase A v1 wire 불변)는 여기 한 곳에서만 갱신된다 — 제출 순서가 곧
//   digest 순서이므로 두 경로가 같은 순서로 부르면 digest가 동일하다(§7 ABBA).
// ★B2: 반환 = 1(동기 완료) / 0(ERROR_IO_PENDING) / -1(제출 실패 — fail_soft에서만 반환되고,
//   demand는 기존 fail-stop 그대로 fatal_abort 한다). fail_soft 경로는 digest·pb_io를 남기지
//   않는다(SUBMIT_FAILED = "성공 제출"이 아니므로 §5 pb_io 미기록·§6 카운터가 권위).
int io_submit_locked(BatchMiss & m, IoChannel & ch, int32_t mlpos, uint32_t batch_order,
                     uint8_t origin = PIO_ORIGIN_DEMAND, bool fail_soft = false) {
    const LayerInfo & L = g_mod.layers[(size_t)mlpos];
    const uint32_t want_bytes = (uint32_t)L.stride_bytes;
    const bool demand    = (origin == PIO_ORIGIN_DEMAND);
    const bool trace_on  = demand && ggml_moe_trace_enabled(); // trace는 이벤트 발행만 가른다
    // ★게이트 무관 제출 계수(C3 "읽기 제출 0" 판정의 근거) — metrics/trace 상태와 독립.
    g_mod.io_submit_ops++;
    const bool pa_on     = demand && g_mod.phasea_on;
    const bool digest_on = demand && g_mod.metrics.enabled;

    ZeroMemory(&ch.ov, sizeof(ch.ov));
    ch.ov.hEvent     = ch.ev;
    ch.ov.Offset     = (DWORD)(m.file_offset & 0xFFFFFFFFu);
    ch.ov.OffsetHigh = (DWORD)(m.file_offset >> 32);
    ResetEvent(ch.ev); // 채널 재사용 — 이전 완료의 signaled 상태를 명시적으로 내린다
    if (trace_on) {
        m.trace_op_id = ggml_moe_trace_io_submit(
            m.trace_run_index, m.trace_graph_seq, m.trace_phase,
            m.trace_layer, MOE_TRACE_NA_U8, m.trace_expert,
            (uint32_t)(m.generation & 0xFFFFFFFFull), want_bytes, (uint32_t)m.slot); // 배선 S12(v4)
    }
    if (digest_on) {
        // ★V1 ordered ReadFile digest — (layer, expert, offset, size) 16B를 제출 순서대로.
        const uint16_t dl = (uint16_t)g_mod.moe_layers[(size_t)mlpos];
        const uint16_t de = (uint16_t)(m.record_idx % g_mod.n_expert);
        uint64_t h = g_mod.readfile_digest;
        h = fnv1a64_bytes(h, &dl, sizeof(dl));
        h = fnv1a64_bytes(h, &de, sizeof(de));
        h = fnv1a64_bytes(h, &m.file_offset, sizeof(m.file_offset));
        h = fnv1a64_bytes(h, &want_bytes, sizeof(want_bytes));
        g_mod.readfile_digest = h;
        g_mod.readfile_ops++;
    }
    if (pa_on) {
        m.pa_batch_order = batch_order;
        m.pa_submit_seq  = g_mod.pa_submit_seq_next++;
        m.pa_submit_qpc  = ggml_moe_phasea_now_qpc();
    }
    // ★테스트 전용 one-shot 제출 실패 주입(oracle §6-2 ⓑ) — speculative(fail-soft) 제출에만
    //   적용된다. 프로덕션 비용 = relaxed 로드 1회(플래그 0이면 완전 no-op).
    if (fail_soft && ggml_moe_phaseb_test_submit_fail_take()) return -1;
    BOOL ok = ReadFile(g_mod.data_file, g_mod.slots[(size_t)m.slot].base, want_bytes, nullptr, &ch.ov);
    if (ok) return 1;
    const DWORD e = GetLastError();
    if (e != ERROR_IO_PENDING) {
        if (fail_soft) return -1;   // ★prefetch fail-soft(§5) — 호출자가 slot 원복·unclaim·계수
        fatal_abort("ReadFile failed err=%lu offset=%llu", (unsigned long)e, (unsigned long long)m.file_offset);
    }
    ch.busy = true;
    return 0;
}

// 전이 digest(§8) — demand-origin 사건은 intent_ordinal=NA로 동일하게 먹인다.
void pb_transition_locked(uint8_t ev, int64_t record_id, int32_t slot, uint64_t generation,
                                 uint64_t intent_ordinal) {
    ggml_moe_phaseb_digest_transition(ev, (uint64_t)record_id, slot, generation, intent_ordinal);
}

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// 공용 io event namespace(§5.1) — submit/terminal이 같은 카운터에서 순번을 받는다(dense).
inline uint64_t pb_next_event_seq() { return g_mod.pb_io_event_seq++; }

// ===========================================================================
// ★5c-T 수집 텔레메트리 헬퍼(SPEC_PHASEB_WIRE 부록 14 · additive only)
//   전 함수가 `g_mod.pb_collect` 로 게이트된다 — 수집 OFF 런에서는 분기 1회가 전부이며
//   어떤 카운터·레코드·산출 바이트도 만들지 않는다(부록 14 서두 1급 계약).
// ===========================================================================

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// ★R1 #6 ① — 공용 PB 계측 core 전방 선언(정의는 §8.0). legacy 배치 pump가 이 본문을 공유한다.
int32_t pb_note_submit_locked(int64_t record_idx, int32_t mlpos, uint64_t file_offset, int32_t slot,
                              uint64_t generation, uint8_t origin, uint16_t rank,
                              uint64_t intent_ordinal, uint32_t req_epoch, uint32_t graph_seq,
                              bool is_prefill, bool is_retry, uint64_t retry_of,
                              uint64_t * out_submit_seq,
                              const struct PbOpStamp * owner_stamp = nullptr);
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// ★5c-T(부록 14.3 ②) — owner 상관자 5종을 **additive** 로 받는다(기본값 = 종전 호출과 동일한
//   NA). in-flight 감소 경계 레코드에 owner ref 를 실으려면 terminal 시그니처에 상관자가
//   있어야 한다는 접점 조사 결과(COE ⓕ "terminal 시그니처에 owner 상관자 미전달")의 처분이다.
void pb_note_terminal_locked(int32_t io_index, uint8_t origin, uint8_t status, bool had_waiter,
                             bool ever_joined, uint64_t publish_seq,
                             uint32_t req_epoch = GGML_MOE_PHASEB_NA_U32,
                             uint32_t graph_seq = GGML_MOE_PHASEB_NA_U32,
                             uint64_t intent_ordinal = GGML_MOE_PHASEB_NA_U64,
                             uint16_t rank = GGML_MOE_PHASEB_NA_U16,
                             int64_t record_idx = -1,
                             const struct PbOpStamp * owner_stamp = nullptr);

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// ★QD 레버 — 배치의 유니크 miss 전량을 sliding pump로 적재한다(legacy 경로 전용). 호출자는
// g_mod.mutex 보유이며 misses[]는 claim이 끝난 상태(slot/generation/file_offset 확정)여야 한다.
// 실패 의미론 = fail-stop(ReadFile/GetOverlappedResult 실패·byte 불일치 = fatal_abort).
void run_batch_ios_locked(std::vector<BatchMiss> & misses, int32_t mlpos, Phase phase) {
    if (misses.empty()) return;

    const bool pa_on = g_mod.phasea_on;

    const size_t n_miss = misses.size();
    size_t n_chan = (size_t)g_mod.qd_effective;
    if (n_chan > n_miss) n_chan = n_miss;
    if (n_chan < 1)      n_chan = 1;

    std::vector<IoChannel> chans(n_chan); // ★크기 확정 후 재할당 금지(주소 수명 고정)
    for (size_t i = 0; i < n_chan; i++) {
        chans[i].ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!chans[i].ev) fatal_abort("CreateEventW failed err=%lu", (unsigned long)GetLastError());
    }

    // ★R1 #6 ① — legacy 경로도 PB demand wire(pb_io·moe_pb_readfile digest·transition·QD)를
    //   기록한다. selector는 관측을 가르지 않는다(§7.3) — origin 스코프만 §2.3이 가른다.
    uint32_t pb_req = GGML_MOE_PHASEB_NA_U32, pb_gseq = GGML_MOE_PHASEB_NA_U32;
    ggml_moe_phaseb_current_epoch(&pb_req, nullptr, &pb_gseq, nullptr);

    auto finish = [&](BatchMiss & m, DWORD bytes, bool was_pending) {
        io_finish_locked(m, bytes, was_pending, mlpos, phase);
        const uint64_t pseq = ++g_mod.pb_publish_seq;
        pb_note_terminal_locked(m.pb_io_index, PIO_ORIGIN_DEMAND,
                                GGML_MOE_PB_T_SUCCESS_PUBLISHED, /*had_waiter=*/true,
                                /*ever_joined=*/false, pseq,
                                pb_req, pb_gseq, GGML_MOE_PHASEB_NA_U64,
                                GGML_MOE_PHASEB_NA_U16, m.record_idx);
        pb_transition_locked(GGML_MOE_PB_EV_PUBLISH_DEMAND, m.record_idx, m.slot, m.generation,
                             GGML_MOE_PHASEB_NA_U64);
    };

    // 제출 1건. 반환 true = ReadFile이 동기 완료(TRUE 반환)해 그 자리에서 회수까지 끝남(채널 미점유).
    auto submit = [&](IoChannel & ch, size_t mi) -> bool {
        BatchMiss & m = misses[mi];
        ch.miss_idx = mi;
        const int rc_sub = io_submit_locked(m, ch, mlpos, (uint32_t)mi);
        uint64_t sseq = 0;
        m.pb_io_index = pb_note_submit_locked(m.record_idx, mlpos, m.file_offset, m.slot,
                                              m.generation, PIO_ORIGIN_DEMAND,
                                              GGML_MOE_PHASEB_NA_U16, GGML_MOE_PHASEB_NA_U64,
                                              pb_req, pb_gseq, phase == Phase::PREFILL,
                                              false, GGML_MOE_PHASEB_NA_U64, &sseq);
        if (rc_sub == 0 && m.pb_io_index >= 0)
            g_mod.pb_ios[(size_t)m.pb_io_index].flags |= 0x01;   // pending
        if (rc_sub == 1) {
            // ★즉시 완료는 그 자리에서 회수한다 — wait pass로 미루면 outstanding이 실제보다 오래
            // 잡혀 app_qd가 부풀고 계측이 왜곡된다(심의 3 D-3).
            DWORD n = 0;
            if (!GetOverlappedResult(g_mod.data_file, &ch.ov, &n, TRUE))
                fatal_abort("GetOverlappedResult failed err=%lu offset=%llu",
                            (unsigned long)GetLastError(), (unsigned long long)m.file_offset);
            finish(m, n, /*was_pending=*/false);
            return true;
        }
        return false;
    };

    // 회수 1건(wait-any). 완료 순서로 회수하지만 slot 배정·LRU tick은 claim에서 이미 확정돼 있어
    // 최종 상태는 완료 순서에 의존하지 않는다(NVMe 완료 순서가 런마다 달라도 결정론 보존).
    auto retire_one = [&]() {
        HANDLE hs[MOE_DIRECT_QD_MAX + 1];
        size_t map[MOE_DIRECT_QD_MAX + 1];
        DWORD cnt = 0;
        for (size_t i = 0; i < n_chan; i++) if (chans[i].busy) { hs[cnt] = chans[i].ev; map[cnt] = i; cnt++; }
        if (cnt == 0) return;
        const DWORD w = WaitForMultipleObjects(cnt, hs, FALSE, INFINITE); // 상한 63 < MAXIMUM_WAIT_OBJECTS
        if (w >= cnt) fatal_abort("WaitForMultipleObjects failed ret=%lu err=%lu",
                                  (unsigned long)w, (unsigned long)GetLastError());
        IoChannel & ch = chans[map[w]];
        BatchMiss & m = misses[ch.miss_idx];
        DWORD n = 0;
        if (!GetOverlappedResult(g_mod.data_file, &ch.ov, &n, TRUE))
            fatal_abort("GetOverlappedResult failed err=%lu offset=%llu",
                        (unsigned long)GetLastError(), (unsigned long long)m.file_offset);
        ch.busy = false;
        finish(m, n, /*was_pending=*/true);
    };

    // sliding pump: QD만큼 제출 → 최소 1건 회수 → 다음 제출. 전량 제출 후 전량 대기 금지
    // (유니크 miss가 QD를 넘는 prefill에서 상한이 무의미해진다).
    size_t next = 0, n_busy = 0;
    while (next < n_miss || n_busy > 0) {
        while (next < n_miss && n_busy < n_chan) {
            size_t ci = 0;
            while (ci < n_chan && chans[ci].busy) ci++;
            if (ci >= n_chan) break; // n_busy < n_chan 이므로 도달 불가(방어)
            const bool immediate = submit(chans[ci], next);
            next++;
            if (!immediate) n_busy++;
        }
        if (pa_on) ggml_moe_phasea_io_qd((int32_t)n_busy, ggml_moe_phasea_now_qpc());
        if (n_busy == 0) continue;
        retire_one();
        n_busy--;
        if (pa_on) ggml_moe_phasea_io_qd((int32_t)n_busy, ggml_moe_phasea_now_qpc());
    }

    for (size_t i = 0; i < n_chan; i++) if (chans[i].ev) CloseHandle(chans[i].ev);
}

// ===========================================================================
// §8. 슬롯 풀 / LRU+lease. record_idx = mlpos*n_expert + expert. stride/record_base = 층별.
//     ★QD 레버: 구 ensure_locked(record 1개를 판별→축출→동기 read→publish까지 완주)는
//     ggml_moe_direct_acquire의 배치 플래너(§12 — resolve/preflight/claim/submit/retire 8단계)로
//     대체됐다. 슬롯 회계(hit/miss 계수)는 위 count_hit_locked/count_miss_locked가 그대로 승계한다.
//     ★B1: 그 배치 플래너 ①~⑤단계가 아래 plan_demand_locked로 추출돼 legacy 경로와 영속
//     dispatcher가 **같은 코드**를 쓴다(§7 "legacy↔persistent 8단계 동형"의 구조적 근거).
// ===========================================================================

// ===========================================================================
// §8.0 ★B2 공용 헬퍼 (PHASEB_DESIGN §4 상태기계 · SPEC_PHASEB_WIRE §4~§6·§8)
//      전부 g_mod.mutex 보유 상태에서만 호출된다.
// ===========================================================================

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// victim 비교(PHASEB_DESIGN §4): probation 우선(오래된 intent 먼저) → 그 외(last_used).
//   동률은 낮은 slot index(B1의 "ascending scan + strict <"와 동일 타이브레이크).
// ★사양 보완 1건(보고 대상): 설계 §4는 **probation/demand 두 class만** 열거한다. owner 없는
//   빈 슬롯은 어느 class에도 속하지 않는데, 이를 probation 뒤로 두면 캐시가 덜 찬 구간에서
//   프리페치가 **자기 결과를 먼저 축출**한다(빈 슬롯을 두고 probation을 깨는 지배당한 선택).
//   그래서 rank 0 = 빈 슬롯(owner_record<0 · 축출 비용 0) → rank 1 = probation → rank 2 = demand.
//   정상 운전(캐시 포화 = Qwen122 실형상)에는 빈 슬롯이 없어 설계 문구와 **동일**하고,
//   프리페치 OFF에서는 probation이 존재하지 않으며 빈 슬롯의 last_used=0이 최소라 B1의
//   min(last_used)와도 **동형**이다(무행동 회귀의 구조적 근거).
bool victim_better_locked(int32_t cand, int32_t best) {
    if (best < 0) return true;
    const Slot & a = g_mod.slots[(size_t)cand];
    const Slot & b = g_mod.slots[(size_t)best];
    auto rank_of = [](const Slot & s) -> int {
        if (s.owner_record < 0)                             return 0;   // 빈 슬롯
        if (s.cache_class == GGML_MOE_PB_CLASS_PROBATION)   return 1;   // READY_PROBATION
        return 2;                                                       // READY_DEMAND/기타
    };
    const int ra = rank_of(a), rb = rank_of(b);
    if (ra != rb) return ra < rb;
    const uint64_t ka = (ra == 1) ? a.pf_intent_ordinal : a.last_used;
    const uint64_t kb = (rb == 1) ? b.pf_intent_ordinal : b.last_used;
    if (ka != kb) return ka < kb;
    // ★R1 #7 — 같은 intent의 여러 rank가 동시에 probation이면 ordinal이 동일하므로 rank가
    //   비교 키의 다음 성분이어야 한다(§4 "(class=probation, prefetch_intent_seq, slot)"의
    //   intent_seq 5-튜플 마지막 성분 = rank).
    if (ra == 1 && a.pf_rank != b.pf_rank) return a.pf_rank < b.pf_rank;
    return cand < best;
}

uint64_t pb_row_key(uint32_t graph_seq, uint16_t layer) {
    return ((uint64_t)graph_seq << 16) | (uint64_t)layer;
}

// (graph, layer) canonical demand 표식 — 신규 표식이면 true(= 이 acquire가 canonical).
bool pb_mark_served_locked(uint32_t graph_seq, int32_t mlpos) {
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.pb_served_graph.size() ||
        graph_seq == GGML_MOE_PHASEB_NA_U32) return true;
    if (g_mod.pb_served_graph[(size_t)mlpos] == graph_seq) return false;  // 같은 graph 재호출 = 비canonical
    g_mod.pb_served_graph[(size_t)mlpos] = graph_seq;
    return true;
}
// too_late — 이 intent의 target 층 demand가 이미 치러졌는가(지연 graph 수와 무관하게 정확).
bool pb_is_served_locked(uint32_t graph_seq, int32_t mlpos) {
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.pb_served_graph.size() ||
        graph_seq == GGML_MOE_PHASEB_NA_U32) return false;
    const uint32_t last = g_mod.pb_served_graph[(size_t)mlpos];
    return last != GGML_MOE_PHASEB_NA_U32 && graph_seq <= last;
}

// 이미 만들어진 pb_sample 행 조회(없으면 -1) — 제출/처분 시점의 행 결속.
int32_t pb_find_row_locked(uint32_t graph_seq, int32_t mlpos) {
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.moe_layers.size()) return -1;
    auto it = g_mod.pb_sample_index.find(pb_row_key(graph_seq, (uint16_t)g_mod.moe_layers[(size_t)mlpos]));
    return it == g_mod.pb_sample_index.end() ? -1 : it->second;
}

// pb_sample 행 확보(§4 — per (graph, layer) 1개). intent 결속과 demand acquire 중 먼저 오는 쪽이 만든다.
int32_t pb_row_locked(uint32_t graph_seq, uint32_t request_index, uint32_t decode_ordinal,
                      int32_t mlpos, uint8_t is_prefill) {
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.moe_layers.size()) return -1;
    const uint16_t layer = (uint16_t)g_mod.moe_layers[(size_t)mlpos];
    const uint64_t key = pb_row_key(graph_seq, layer);
    auto it = g_mod.pb_sample_index.find(key);
    if (it != g_mod.pb_sample_index.end()) return it->second;
    // ★off = 사이드카 미버퍼링 · ★observe(8호)도 미버퍼링 — 모드↔파일 조합(§1.1)이 observe 에
    //   `sample`·`io` 부재를 요구하므로 **버퍼링 자체를 하지 않는다**(빈 파일 생성 경로 차단).
    if (ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_TRACE &&
        ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_EQUIV) return -1;
    // ★누수 수리(R2 후속) — 행 1건의 실 메모리는 레코드(112B) + 인덱스 맵 노드다. 레코드만
    //   청구하면 BUF_MB 상한이 실제 사용량보다 훨씬 늦게 걸려 프로세스가 먼저 포화한다.
    //   인덱스 노드 보수 추정치(64B)를 함께 청구해 상한이 실 메모리를 덮게 한다.
    static const uint64_t kRowIndexBytes = 64;
    if (!ggml_moe_phaseb_buffer_reserve(sizeof(ggml_moe_phaseb_sample) + kRowIndexBytes)) {
        ggml_moe_phaseb_note_dropped(1);
        return -1;
    }
    ggml_moe_phaseb_sample s{};
    s.graph_seq      = graph_seq;
    s.request_index  = request_index;
    s.decode_ordinal = is_prefill ? GGML_MOE_PHASEB_NA_U32 : decode_ordinal;
    s.layer          = layer;
    s.source_layer   = GGML_MOE_PHASEB_NA_U16;
    s.flags          = (uint8_t)(is_prefill ? 0x08 : 0x00);
    s.k_effective    = (uint8_t)(g_mod.pb_prefetch_on ? g_mod.pb_k : 0);
    s.intent_ordinal = GGML_MOE_PHASEB_NA_U64;
    for (int i = 0; i < GGML_MOE_PHASEB_TOPK; i++) {
        s.intent_ids[i]       = GGML_MOE_PHASEB_NA_U16;
        s.rank_disposition[i] = GGML_MOE_PB_D_RANK_NA;
    }
    // intent_present=0의 사유(닫힌 enum §4.1). 실제 intent가 결속되면 NONE으로 덮어쓴다.
    // ★R1 #10 ② — capture/worker가 이 (graph, target 층)에 기록한 **실제 사유 전 종**을 회수한다
    //   (MULTI_TOKEN·NONCONTIG·NONFINITE·CAPTURE_QUEUE_FULL·REQUEST_DISABLED 포함).
    uint8_t tgt_reason = 0, tgt_overflow = 0;
    ggml_moe_phaseb_target_status(graph_seq, mlpos, &tgt_reason, &tgt_overflow);
    // ★R2 #5 — graph 전환 뒤에 도착한(보류된) 비동기 사유도 여기서 소비한다.
    {
        auto pit = g_mod.pb_pending_status.find(key);
        if (pit != g_mod.pb_pending_status.end()) {
            const uint8_t r2 = (uint8_t)(pit->second >> 16);
            if (r2) tgt_reason = r2;
            if (pit->second & 1u) tgt_overflow = 1;
            g_mod.pb_pending_status.erase(pit);
        }
    }
    if (is_prefill)                        s.elig_skip_reason = GGML_MOE_PB_SKIP_PREFILL;
    else if (mlpos == 0)                   s.elig_skip_reason = GGML_MOE_PB_SKIP_NO_SOURCE;
    else if (tgt_reason != 0)              s.elig_skip_reason = tgt_reason;
    else if (!g_mod.pb_prefetch_on)        s.elig_skip_reason = GGML_MOE_PB_SKIP_PREFETCH_OFF;
    else                                   s.elig_skip_reason = GGML_MOE_PB_SKIP_SOURCE_RANGE;
    if (tgt_overflow) s.flags |= 0x04;     // ★R1 #10 ③ intent_dropped_overflow
    if (g_mod.pb_prefetch_on) s.flags |= 0x02;   // pf_enabled
    g_mod.pb_samples.push_back(s);
    const int32_t idx = (int32_t)g_mod.pb_samples.size() - 1;
    g_mod.pb_sample_index[key] = idx;
    return idx;
}

// ★R2 #5 — 비동기 사유(worker NONFINITE·intent overflow) 상태 큐를 소비해 **이미 생성된**
//   pb_sample 행까지 갱신한다. 행이 아직 없으면 보류함에 담아 행 생성 시 소비한다.
//   호출자는 g_mod.mutex 보유(dispatcher pass · finalize 직전 safety pass).
void pb_drain_status_locked() {
    uint32_t gs = 0; int32_t ml = -1; uint8_t reason = 0, ovf = 0;
    while (ggml_moe_phaseb_status_pop(&gs, &ml, &reason, &ovf)) {
        if (ml < 0 || ml >= (int32_t)g_mod.moe_layers.size()) continue;
        const uint64_t key = pb_row_key(gs, (uint16_t)g_mod.moe_layers[(size_t)ml]);
        auto it = g_mod.pb_sample_index.find(key);
        if (it != g_mod.pb_sample_index.end()) {
            ggml_moe_phaseb_sample & s = g_mod.pb_samples[(size_t)it->second];
            // intent 가 결속된 행(intent_present=1)의 사유는 NONE 이 권위 — 덮어쓰지 않는다.
            if (reason && !(s.flags & 0x01)) s.elig_skip_reason = reason;
            if (ovf) s.flags |= 0x04;
            continue;
        }
        // 행 미생성 — 보류함(상위 16bit=reason, 하위 1bit=overflow). ★상한 초과 = 결손 latch.
        auto pit = g_mod.pb_pending_status.find(key);
        if (pit == g_mod.pb_pending_status.end()) {
            if (g_mod.pb_pending_status.size() >= ModuleState::PB_PENDING_STATUS_MAX) {
                ggml_moe_phaseb_mark_incomplete(GGML_MOE_PB_INC_PENDING_CAP);
                continue;
            }
            pit = g_mod.pb_pending_status.emplace(key, 0u).first;
        }
        if (reason) pit->second = (pit->second & 0xFFFFu) | ((uint32_t)reason << 16);
        if (ovf)    pit->second |= 1u;
    }
}


// ★R1 #6 ① — 공용 submit 계측 core(§5·§5.1·§6·§8). PendingIo 유무와 무관하게 legacy 배치
//   pump와 영속 dispatcher가 **같은 본문**을 쓴다 — PB 채널의 demand 실값은 selector 무관이다
//   (§7.3 · §2.3은 origin 스코프만 가른다). 반환 = pb_io 레코드 인덱스(-1=미기록 모드).
int32_t pb_note_submit_locked(int64_t record_idx, int32_t mlpos, uint64_t file_offset, int32_t slot,
                              uint64_t generation, uint8_t origin, uint16_t rank,
                              uint64_t intent_ordinal, uint32_t req_epoch, uint32_t graph_seq,
                              bool is_prefill, bool is_retry, uint64_t retry_of,
                              uint64_t * out_submit_seq, const struct PbOpStamp * owner_stamp) {
    const bool prefetch = (origin == PIO_ORIGIN_PREFETCH);
    const uint64_t sseq = pb_next_event_seq();
    const uint64_t sqpc = ggml_moe_phaseb_now_qpc();
    if (out_submit_seq) *out_submit_seq = sseq;
    if (prefetch) g_mod.pb_inflight_prefetch++; else g_mod.pb_inflight_demand++;
    const int32_t qd = g_mod.pb_inflight_demand, qp = g_mod.pb_inflight_prefetch;
    ggml_moe_phaseb_note_qd((uint32_t)qd, (uint32_t)qp, (uint32_t)(qd + qp));
    // ★5c-T(부록 14.3 ②) — in-flight **증가 경계** 1지점. 시간가중 동시성의 원료다
    //   (엔진은 원시 사건만 적고 판정하지 않는다 — controller §6).
    if (g_mod.pb_collect) {
        ggml_moe_phaseb_ledger r = pb_led_new(GGML_MOE_PB_L_INFLIGHT_DELTA);
        r.io_event_seq   = sseq;
        r.qpc            = sqpc;
        r.delta          = +1;
        r.origin         = prefetch ? 1 : 0;
        r.request_epoch  = req_epoch;
        r.graph_seq      = graph_seq;
        r.record_idx     = record_idx;
        r.rank           = prefetch ? rank : GGML_MOE_PHASEB_NA_U16;
        r.intent_ordinal = prefetch ? intent_ordinal : GGML_MOE_PHASEB_NA_U64;
        // ★r3 수리 1 — owner stamp(K/N/QD·tuple_revision·admission_seq)는 **prefetch-origin
        //   에만** 싣는다. demand 는 부록 §14.3.3 규약상 stamp NA 다. 구 판은 owner_stamp 가
        //   없으면 `pb_led_stamp_from_request` 로 폴백해, qualification legacy demand 경로
        //   (`collect=on + MOE_DIRECT_QUALIFY=1 + MOE_DIRECT_IO_PATH=legacy + prefetch=off`)에서
        //   demand 레코드에 QD·admission_seq 가 새 규약과 실물이 어긋났다.
        if (prefetch && owner_stamp) pb_led_stamp_from_op(r, owner_stamp);
        pb_led_emit_locked(r);
    }

    const LayerInfo & L = g_mod.layers[(size_t)mlpos];
    const uint16_t layer  = (uint16_t)g_mod.moe_layers[(size_t)mlpos];
    const uint16_t expert = (uint16_t)(record_idx % g_mod.n_expert);
    // ★all-origin ordered digest(§8) — 기존 moe_readfile_digest(demand 전용)와 별도 열.
    ggml_moe_phaseb_digest_readfile(layer, expert, file_offset, (uint32_t)L.stride_bytes,
                                    prefetch ? 1 : 0);
    // ★observe(8호) = `pb_io` 부재가 speculative I/O 0 의 wire 증명(§13.3 ④) — 버퍼링 금지.
    if (ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_TRACE &&
        ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_EQUIV) return -1;
    if (!ggml_moe_phaseb_buffer_reserve(sizeof(ggml_moe_phaseb_io))) {
        ggml_moe_phaseb_note_dropped(1);
        return -1;
    }
    ggml_moe_phaseb_io r{};
    r.record_id      = (uint64_t)record_idx;
    r.file_offset    = file_offset;
    r.bytes          = (uint32_t)L.stride_bytes;
    r.expert         = expert;
    r.layer          = layer;
    r.graph_seq      = graph_seq;        // ★보존 owner epoch(ambient 금지)
    r.request_index  = req_epoch;
    r.intent_ordinal = prefetch ? intent_ordinal : GGML_MOE_PHASEB_NA_U64;
    r.rank           = prefetch ? rank : GGML_MOE_PHASEB_NA_U16;
    r.origin         = prefetch ? 1 : 0;
    r.flags          = (uint8_t)((is_prefill ? 0x02 : 0x00) | (is_retry ? 0x04 : 0x00));
    r.slot           = slot;
    r.generation     = generation;
    r.submit_event_seq   = sseq;
    r.submit_qpc         = sqpc;
    r.terminal_event_seq = GGML_MOE_PHASEB_NA_U64;
    r.terminal_qpc       = 0;
    r.publish_seq        = 0;            // 1-based · 0=미publish
    r.retry_of_submit_event_seq = is_retry ? retry_of : GGML_MOE_PHASEB_NA_U64;
    r.qd_demand_at_submit   = pb_clamp_u8(qd);
    r.qd_prefetch_at_submit = pb_clamp_u8(qp);
    r.qd_total_at_submit    = pb_clamp_u8(qd + qp);
    g_mod.pb_ios.push_back(r);
    ggml_moe_phaseb_note_sidecar_record(3, req_epoch);
    return (int32_t)g_mod.pb_ios.size() - 1;
}

// ★R1 #6 ① — 공용 terminal 계측 core. publish_seq 발급은 성공 terminal에서만.
void pb_note_terminal_locked(int32_t io_index, uint8_t origin, uint8_t status, bool had_waiter,
                             bool ever_joined, uint64_t publish_seq,
                             uint32_t req_epoch, uint32_t graph_seq, uint64_t intent_ordinal,
                             uint16_t rank, int64_t record_idx, const struct PbOpStamp * owner_stamp) {
    const bool prefetch = (origin == PIO_ORIGIN_PREFETCH);
    if (prefetch) g_mod.pb_inflight_prefetch--; else g_mod.pb_inflight_demand--;
    const uint64_t tseq = pb_next_event_seq();
    const uint64_t tqpc = ggml_moe_phaseb_now_qpc();
    const int32_t qd = g_mod.pb_inflight_demand, qp = g_mod.pb_inflight_prefetch;
    ggml_moe_phaseb_note_qd((uint32_t)qd, (uint32_t)qp, (uint32_t)(qd + qp));
    // ★5c-T(부록 14.3 ②) — in-flight **감소 경계** 1지점(after 스냅샷은 감소 후 값).
    if (g_mod.pb_collect) {
        ggml_moe_phaseb_ledger lr = pb_led_new(GGML_MOE_PB_L_INFLIGHT_DELTA);
        lr.io_event_seq   = tseq;
        lr.qpc            = tqpc;
        lr.delta          = -1;
        lr.origin         = prefetch ? 1 : 0;
        lr.request_epoch  = req_epoch;
        lr.graph_seq      = graph_seq;
        lr.intent_ordinal = intent_ordinal;
        lr.rank           = rank;
        lr.record_idx     = record_idx;
        lr.terminal_status = status;
        if (ever_joined) lr.flags |= GGML_MOE_PB_L_F_EVER_JOINED;
        if (had_waiter)  lr.flags |= GGML_MOE_PB_L_F_HAD_WAITER;
        // ★r3 수리 1 — submit 쪽과 같은 가드(demand = stamp NA).
        if (prefetch && owner_stamp) pb_led_stamp_from_op(lr, owner_stamp);
        pb_led_emit_locked(lr);
    }
    if (io_index < 0 || io_index >= (int32_t)g_mod.pb_ios.size()) return;
    ggml_moe_phaseb_io & r = g_mod.pb_ios[(size_t)io_index];
    r.terminal_event_seq = tseq;
    r.terminal_qpc       = tqpc;
    r.terminal_status    = status;
    if (had_waiter)  r.flags |= 0x20;   // had_waiter_at_terminal
    if (ever_joined) r.flags |= 0x08;   // ever_joined
    r.qd_demand_at_terminal   = pb_clamp_u8(qd);
    r.qd_prefetch_at_terminal = pb_clamp_u8(qp);
    r.qd_total_at_terminal    = pb_clamp_u8(qd + qp);
    r.publish_seq = publish_seq;
}

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// ★R2 #4 — legacy(배치 pump) acquire 의 pb_sample 행/3분할 폐합. persistent 경로의
//   canonical mark → 행 생성 → 3분할 → sidecar count 와 **동일 의미**를 selector 무관으로 낸다.
//   legacy 는 in-flight join 이 없으므로(persistent=false) join/join_attempt 는 0 이고 retry 도 없어
//   여기서 즉시 최종화할 수 있다.
void pb_legacy_demand_row_locked(int32_t mlpos, Phase phase, const DemandPlan & p, int32_t n_ids) {
    uint32_t req = GGML_MOE_PHASEB_NA_U32, gepoch = GGML_MOE_PHASEB_NA_U32, gseq = GGML_MOE_PHASEB_NA_U32;
    ggml_moe_phaseb_current_epoch(&req, &gepoch, &gseq, nullptr);
    if (!pb_mark_served_locked(gseq, mlpos)) return;      // 비canonical(같은 graph·층 재호출)
    const bool prefill = (phase == Phase::PREFILL);
    if (!prefill && mlpos == 0) ggml_moe_phaseb_count_skip(GGML_MOE_PB_SKIP_NO_SOURCE, req);
    uint32_t ready = 0, dread = 0;
    for (int32_t i = 0; i < n_ids; i++) {
        if (p.hit_slot[(size_t)i] >= 0) ready++; else dread++;
    }
    const int32_t row = pb_row_locked(gseq, req, gepoch, mlpos, prefill ? 1 : 0);
    if (row >= 0) {
        ggml_moe_phaseb_sample & s = g_mod.pb_samples[(size_t)row];
        s.n_required     = (uint16_t)n_ids;
        s.n_ready        = (uint16_t)ready;
        s.n_join         = 0;
        s.n_demand_read  = (uint16_t)dread;
        s.n_join_attempt = 0;
        ggml_moe_phaseb_note_sidecar_record(2, req);
    }
    ggml_moe_phaseb_count_split(req, (uint32_t)n_ids, ready, 0, dread, 0);
}

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// demand command 종결 시 3분할 최종값 확정(§4 행 closure · §7.4 counts.split).
// ★canonical acquire(행을 만든 command)만 계상한다 — pb_sample 행 합계와 정확히 일치해야 한다.
void pb_finish_demand_command_locked(ControlCmd & c) {
    if (c.kind != CMD_DEMAND_ACQUIRE || !c.pb_canonical) return;
    c.pb_canonical = false;
    if (c.pb_row >= 0 && c.pb_row < (int32_t)g_mod.pb_samples.size()) {
        ggml_moe_phaseb_sample & s = g_mod.pb_samples[(size_t)c.pb_row];
        s.n_join        = (uint16_t)c.sp_join;
        s.n_demand_read = (uint16_t)c.sp_demand_read;
    }
    ggml_moe_phaseb_count_split(c.pb_request_epoch, (uint32_t)c.n_ids, c.sp_ready, c.sp_join,
                                c.sp_demand_read, c.sp_join_attempt);
    // ★5c-T(부록 14.1) — wait/rank 술어의 분모 3종. **canonical demand acquire 1건당 1계수**
    //   (controller §3 C3 경계: 1 acquire 1계수·관측 전용·counterfactual 금지).
    //   `eligible_demand_acquire ≥ waited_acquire ≥ observed_frontier_match` 가 구조로 성립한다
    //   (frontier 래치는 waited 래치가 선 경우에만 세워진다 — blocked 계측점 참조).
    if (g_mod.pb_collect) {
        const int32_t ci = (int32_t)(&c - g_mod.cmds);
        ggml_moe_phaseb_count(GGML_MOE_PB_C_ELIGIBLE_DEMAND_ACQUIRE, c.pb_request_epoch, 1);
        if (ci >= 0 && ci < MOE_DIRECT_CONTROL_Q_CAP) {
            if (g_mod.pb_cmd_waited[(size_t)ci]) {
                ggml_moe_phaseb_count(GGML_MOE_PB_C_WAITED_ACQUIRE, c.pb_request_epoch, 1);
                if (g_mod.pb_cmd_frontier[(size_t)ci])
                    ggml_moe_phaseb_count(GGML_MOE_PB_C_OBSERVED_FRONTIER_MATCH, c.pb_request_epoch, 1);
            }
            g_mod.pb_cmd_waited[(size_t)ci]   = 0;
            g_mod.pb_cmd_frontier[(size_t)ci] = 0;
        }
    }
    if (c.pb_row >= 0) ggml_moe_phaseb_note_sidecar_record(2, c.pb_request_epoch);
    c.pb_row = -1;
}

uint64_t pb_qpc_to_us(uint64_t ticks) {
    static uint64_t freq = 0;
    if (freq == 0) { LARGE_INTEGER f{}; QueryPerformanceFrequency(&f); freq = (uint64_t)f.QuadPart; }
    return freq ? (ticks * 1000000ull / freq) : 0;
}

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// R-B5a facade (iii): the pending-join ordered sink (RB_DESIGN.md:1221 section 2-6
// (iii)). The seven PendingIo fields of the join transaction are written here, in
// their original order, with the caller's `g_mod.mutex` already held. The return
// value is the FIRST-JOIN latch: the caller keeps its `USED_JOIN` count and its
// `CONSUME_JOIN` transition at the positions they occupy today, which is what the
// design's "keeps its position and order" is about.
inline bool dfac_pending_join_note_locked(int32_t oi, uint64_t tick_i,
                                          uint32_t pb_epoch, uint32_t pb_gepoch,
                                          uint32_t pb_gseq, int32_t pb_cmd) {
    PendingIo & op = g_mod.pending_ops[(size_t)oi];
    op.m.multiplicity++;
    op.m.logical_last_tick = tick_i;
    if (op.pb_ever_joined) return false;
    op.pb_ever_joined = true;
    // ★R1 #9 ② — 첫 waiter(= 이 demand)의 epoch 를 보존한다(retry owner epoch).
    op.pb_join_request_epoch = pb_epoch;
    op.pb_join_graph_epoch   = pb_gepoch;
    op.pb_join_graph_seq     = pb_gseq;
    op.pb_join_cmd           = pb_cmd;
    return true;
}

}  // namespace (anonymous) - R-B5b (RB5B_ATOM_SPEC.json :: seam_map.members / declare_in_place)
// The definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the
// dispatch TU calls it, so it leaves the anonymous namespace and ggml-moe-direct-dispatch.h
// carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// 8단계 플래너 ①선검증 ②resolve ③preflight ④tick+hit lease ⑤claim. 호출자는 g_mod.mutex 보유.
// 반환 0=성공(⑥⑦ 제출 준비 완료), -1=거절(★어떤 mutation도 남기지 않는다).
// persistent=true면 pending 표를 참조해 in-flight join을 만든다(C2 — join은 waiter/multiplicity/
// 마지막 demand_tick만 누적하고 slot lease는 건드리지 않는다).
int plan_demand_locked(DemandPlan & p, int32_t mlpos, int32_t part_idx,
                       const int32_t * ids, int32_t n_ids, Phase phase,
                       bool pa_on, uint64_t pa_t_enter, bool persistent,
                       uint32_t pb_epoch = GGML_MOE_PHASEB_NA_U32,
                       uint32_t pb_gepoch = GGML_MOE_PHASEB_NA_U32,
                       uint32_t pb_gseq = GGML_MOE_PHASEB_NA_U32,
                       int32_t pb_cmd = -1) {
    // ── ⓪ ★C3 동결 게이트(SPEC_SHADOW_DRAFT §3) — 섀도 view 가 열려 있는 동안 어떤 캐시
    //      mutation 도 허용하지 않는다. 정상 실행에서는 도달하지 않는 경로이며(섀도는 동기
    //      실행이라 main demand 와 겹치지 않는다), 도달했다는 것 자체가 동결 위반이므로
    //      조용히 통과시키지 않고 계수 후 실패로 되돌린다. view 가 닫혀 있으면 atomic 로드
    //      1회가 전부다(프로덕션 비용). ──
    if (g_mod.shadow_view_open.load(std::memory_order_acquire) != 0) {
        g_mod.shadow_view_refused++;
        // ── ★TEST-ONLY 소비자 1개(M4 · SPEC_SHADOW_DRAFT §5 음성 대조 "tick 증가 → e FAIL") ──
        //   ★프로덕션 명령열 무증가: 이 블록 자체가 "동결 창 안에서 demand 가 왔다"는 **위반
        //     경로**라 정상 런에서는 도달 자체가 없다(섀도는 main graph 완주·synchronize 뒤
        //     동기 실행이라 main demand 와 겹치지 않는다). 즉 창이 닫혀 있는 프로덕션 비용은
        //     위 atomic 로드 1회 그대로이며 아래 조건 검사는 실행조차 되지 않는다.
        //   ★위협모델: GGML_MOE_SHADOW_FAULT_TICK_BUMP 를 세우는 창구는
        //     ggml_moe_shadow_test_fault() 하나뿐이고 프로덕션에는 그 호출자가 없다.
        //   ★목적: ④의 LRU tick 예약(++g_mod.tick)은 이 함수 안에만 있는데 그 함수가 동결 중
        //     첫 줄에서 거부하므로, "동결 창 안에서 tick 이 오른다"는 SPEC 문면의 음성 대조가
        //     트리 전체에서 비도달이었다(fault bit 소비자 0개). 여기서 그 한 상황만 만들어
        //     C3 진입/탈출 등식(tick·cache digest)이 실제로 PASS→FAIL 로 뒤집히는지 실증한다.
        //     캐시 mutation 은 여전히 하지 않는다 — 바뀌는 것은 tick 단 하나다.
        if ((ggml_moe_shadow_test_fault_bits() & GGML_MOE_SHADOW_FAULT_TICK_BUMP) != 0) {
            ++g_mod.tick;
        }
        return -1;
    }

    // ── ① 선검증(어떤 mutation보다 앞) ──
    for (int32_t i = 0; i < n_ids; i++) {
        const int32_t eid = ids[i];
        if (eid < 0 || eid >= g_mod.n_expert) return -1;
    }

    // ── ② resolve — hit/miss 판별만(mutation 없음). hit 슬롯은 배치 보호 집합에 넣고,
    //      miss는 record 단위로 dedupe해 multiplicity(원래 요청 횟수)를 센다. dedupe를 빠뜨리면
    //      같은 record를 두 번 읽고, multiplicity를 빠뜨리면 release에서 lease underflow /
    //      조기 victim 재사용이 난다. ──
    p.hit_slot.assign((size_t)n_ids, -1);
    p.req_miss.assign((size_t)n_ids, -1);
    p.req_join.assign((size_t)n_ids, -1);
    p.slot_excl.assign((size_t)g_mod.n_slots, 0);
    p.miss.clear();
    p.join_ops.clear();

    // ★영속 경로: in-flight pending이 점유한 슬롯은 terminal까지 pin(축출·재사용 금지 — §4 상태기계).
    //   legacy 경로에서는 pending이 존재할 수 없어 이 pass가 no-op이다(동형 보존).
    if (persistent) {
        for (size_t i = 0; i < g_mod.slot_pending.size(); i++) {
            if (g_mod.slot_pending[i]) p.slot_excl[i] = 1;
        }
    }

    for (int32_t i = 0; i < n_ids; i++) {
        const int64_t record_idx = (int64_t)mlpos * g_mod.n_expert + ids[i];
        RecordEntry & re = g_mod.record_table[(size_t)record_idx];
        const int32_t si = re.slot_index.load(std::memory_order_relaxed);
        if (si >= 0) {
            Slot & s = g_mod.slots[(size_t)si];
            if (s.generation.load(std::memory_order_relaxed) == re.slot_generation.load(std::memory_order_relaxed)
                && s.owner_record == record_idx) {
                p.hit_slot[(size_t)i] = si;
                p.slot_excl[(size_t)si] = 1; // 배치 보호 — 앞 miss의 victim이 되지 않게
                continue;
            }
        }
        if (persistent) {
            const int32_t po = g_mod.pending_by_record[(size_t)record_idx];
            if (po >= 0) {
                // ★C2 join — 새 물리 제출을 만들지 않는다(triple당 물리 submit<=1).
                p.req_join[(size_t)i] = po;
                bool seen = false;
                for (int32_t o : p.join_ops) if (o == po) { seen = true; break; }
                if (!seen) p.join_ops.push_back(po);
                continue;
            }
        }
        size_t k = 0;
        for (; k < p.miss.size(); k++) if (p.miss[k].record_idx == record_idx) break;
        if (k == p.miss.size()) { BatchMiss bm; bm.record_idx = record_idx; p.miss.push_back(bm); }
        p.miss[k].multiplicity++;
        p.req_miss[(size_t)i] = (int32_t)k;
    }

    // ── ③ capacity preflight — 유니크 miss 전부를 수용할 victim(lease==0·보호 집합 제외)이
    //      없으면 unpublish·lease 변경 전에 -1. 부분 claim 후 실패하면 되돌릴 수 없다. ──
    if (!p.miss.empty()) {
        int64_t avail = 0;
        for (int32_t i = 0; i < g_mod.n_slots; i++) {
            if (p.slot_excl[(size_t)i]) continue;
            if (g_mod.slots[(size_t)i].lease.load(std::memory_order_relaxed) != 0) continue;
            avail++;
        }
        if (avail < (int64_t)p.miss.size()) return -1;
    }

    // ── ★Phase A 진입 통지(관측 전용) — ③까지의 조기 -1 반환 경로를 전부 지난 뒤라
    //    "열린 채 버려지는 acquire 문맥"이 없다(④~⑦은 조기 반환이 없고 ⑧은 fail-stop). ──
    int32_t pa_canonical = 0;
    if (pa_on) {
        ggml_moe_phasea_acquire_enter(mlpos, part_idx, n_ids, ids,
                                      phase == Phase::PREFILL ? 1 : 0, pa_t_enter, &pa_canonical);
        p.pa_tick.assign((size_t)n_ids, 0);
    }

    // ── ④ LRU tick 예약(★논리 요청 순서 = 원 요청 인덱스 i=0..n_ids-1 순) + hit lease 취득 ──
    //      tick을 hit 전량 → miss 전량으로 나눠 배정하면 혼재 배치 {miss, hit}에서 입력상 뒤인
    //      hit가 더 오래된 tick을 받아 캐시 압력 하 victim 순서가 뒤집힌다(실 demand는 혼재가 정상).
    //      그래서 한 루프에서 요청 순서대로 tick을 발급하고, hit는 즉시 적용, miss는 해당 유니크
    //      엔트리에 예약만 해둔다(중복 출현이면 마지막 출현 tick으로 덮어써 refresh 의미 보존).
    //      completion 순서 배정 금지 — NVMe 완료 순서가 런마다 달라 재현성이 깨진다.
    for (int32_t i = 0; i < n_ids; i++) {
        const uint64_t tick_i = ++g_mod.tick;
        if (pa_on) p.pa_tick[(size_t)i] = tick_i;
        const int32_t si = p.hit_slot[(size_t)i];
        if (si >= 0) {
            Slot & s = g_mod.slots[(size_t)si];
            s.lease.fetch_add(1, std::memory_order_relaxed);
            xstore_note_lease_granted(1);   // E-2a lease-unit ledger (b)
            s.last_used = tick_i;
            // ★B2 promotion — READY_PROBATION 슬롯의 demand hit(§4 상태기계). 원 op당 최대 1회
            //   (승격이 class를 바꾸므로 재계상 불가). ever_promoted는 pb_io 레코드에 래치한다.
            if (s.cache_class == GGML_MOE_PB_CLASS_PROBATION) {
                const uint64_t ord = s.pf_intent_ordinal;
                // ★5c-T — 유예됐던 4분할이 **여기서** used_promoted 로 확정된다(부록 14.3 ④
                //   deferred). owner 스탬프를 지우기 **전에** 레코드를 만든다.
                // ★r1 M2 — 승격 계수의 귀속은 **READY 를 만든 owner generation**이다. 구 판은
                //   ledger 만 owner(`s.pf_request_epoch`)를 쓰고 카운터는 승격을 유발한 현재
                //   demand 의 `pb_epoch` 에 더해, 요청 A 의 READY 가 요청 B 에서 승격되면 A 에는
                //   issued 만 남고 B 에 used 가 들어가 per-generation 4분할이 깨졌다
                //   (취소 후 READY 생존은 정상 경로라 `-np 1` 에서도 도달).
                const uint32_t owner_req = s.pf_request_epoch;
                if (g_mod.pb_collect && !g_mod.pb_censused) {
                    ggml_moe_phaseb_ledger lr = pb_led_new(GGML_MOE_PB_L_OP_TERMINAL);
                    lr.request_epoch   = owner_req;
                    lr.graph_seq       = s.pf_graph_seq;
                    lr.intent_ordinal  = ord;
                    lr.rank            = s.pf_rank;
                    lr.record_idx      = s.owner_record;
                    lr.origin          = 1;
                    lr.terminal_status = GGML_MOE_PB_T_SUCCESS_PUBLISHED;
                    lr.partition       = GGML_MOE_PB_P_USED_PROMOTED;
                    pb_led_stamp_from_slot(lr, s);
                    lr.flags          |= GGML_MOE_PB_L_F_DEFERRED;
                    pb_led_emit_locked(lr);
                }
                s.cache_class = GGML_MOE_PB_CLASS_DEMAND;
                s.pf_intent_ordinal = GGML_MOE_PHASEB_NA_U64;
                s.pf_rank = GGML_MOE_PHASEB_NA_U16;
                if (s.pf_io_index >= 0 && s.pf_io_index < (int32_t)g_mod.pb_ios.size())
                    g_mod.pb_ios[(size_t)s.pf_io_index].flags |= 0x10;   // ever_promoted
                s.pf_io_index = -1;
                pb_slot_stamp_clear(s);   // ★5c-T §1
                ggml_moe_phaseb_count(GGML_MOE_PB_C_USED_PROMOTED,
                                      g_mod.pb_collect ? owner_req : pb_epoch, 1);
                pb_transition_locked(GGML_MOE_PB_EV_CONSUME_PROMOTION, s.owner_record, si,
                                     s.generation.load(std::memory_order_relaxed), ord);
            }
            count_hit_locked(phase);
        } else if (p.req_join[(size_t)i] >= 0) {
            // ★C2 — join 시점에는 waiter·multiplicity·마지막 demand_tick만 누적(lease 무변경).
            PendingIo & op = g_mod.pending_ops[(size_t)p.req_join[(size_t)i]];
            // ★used_join 래치 = **첫 join mutation 시점**(§6 · R2 #2 — ③ preflight 통과 뒤).
            //   publish 시점이 아니며 이후 오류 terminal이어도 used 유지.
            if (dfac_pending_join_note_locked(p.req_join[(size_t)i], tick_i,
                                             pb_epoch, pb_gepoch, pb_gseq, pb_cmd)) {
                if (op.origin == PIO_ORIGIN_PREFETCH)
                    ggml_moe_phaseb_count(GGML_MOE_PB_C_USED_JOIN, op.pb_request_epoch, 1);
                pb_transition_locked(GGML_MOE_PB_EV_CONSUME_JOIN, op.m.record_idx, op.m.slot,
                                     op.m.generation, op.intent_seq);
            }
        } else {
            p.miss[(size_t)p.req_miss[(size_t)i]].logical_last_tick = tick_i;
        }
    }

    // ── ⑤ claim — 유니크 miss마다(첫 출현 순) 서로 다른 victim을 선정(lease==0 · 보호/claimed
    //      제외)하고 즉시 claimed 마킹. 기존 owner unpublish → generation fetch_add → owner_record
    //      갱신 → ④에서 예약된 논리 tick 대입(여기서 새 tick을 발급하지 않는다).
    //      O3(data_wait_begin)를 record 단위로 발행하고 episode id를 보관한다.
    //      ★claim 전량을 submit보다 먼저 끝내므로 slot 배정·LRU가 QD 값과 무관하게 결정적이다. ──
    const LayerInfo & L = g_mod.layers[(size_t)mlpos];
    const bool trace_on = ggml_moe_trace_enabled();
    for (auto & m : p.miss) {
        int32_t victim = -1;
        for (int32_t i = 0; i < g_mod.n_slots; i++) {
            if (p.slot_excl[(size_t)i]) continue;
            if (g_mod.slots[(size_t)i].lease.load(std::memory_order_relaxed) != 0) continue;
            if (victim_better_locked(i, victim)) victim = i;
        }
        // preflight가 유니크 miss 수만큼의 victim을 보장하고 claim은 1건당 정확히 1개만 소모한다.
        if (victim < 0) fatal_abort("not enough claim victims (preflight invariant violated) n_unique_miss=%zu", p.miss.size());
        p.slot_excl[(size_t)victim] = 1;
        Slot & s = g_mod.slots[(size_t)victim];
        if (s.owner_record >= 0) {
            // ★Phase A cache mutation ledger — 축출 1건(축출된 record_idx 보존).
            if (pa_on) ggml_moe_phasea_acquire_evict(victim, s.owner_record,
                                                     (int32_t)(m.record_idx % g_mod.n_expert));
            pb_note_ready_evict_locked(s);   // ★5c-T 4분할 — 미소비 READY 축출 = resolved_wasted
            pb_transition_locked(GGML_MOE_PB_EV_EVICT, s.owner_record, victim,
                                 s.generation.load(std::memory_order_relaxed), s.pf_intent_ordinal);
            g_mod.record_table[(size_t)s.owner_record].slot_index.store(-1, std::memory_order_release);
        }
        m.generation   = s.generation.fetch_add(1, std::memory_order_relaxed) + 1;
        s.owner_record = m.record_idx;
        s.last_used    = m.logical_last_tick; // ★④에서 논리 요청 순서로 예약된 tick
        s.cache_class  = GGML_MOE_PB_CLASS_NONE;   // 아직 미publish(READY_*는 성공 terminal에서)
        s.pf_intent_ordinal = GGML_MOE_PHASEB_NA_U64;
        s.pf_rank      = GGML_MOE_PHASEB_NA_U16;
        s.pf_io_index  = -1;
        pb_slot_stamp_clear(s);   // ★5c-T §1
        m.slot         = victim;
        const int64_t expert = m.record_idx % g_mod.n_expert;
        m.file_offset  = L.record_base + (uint64_t)expert * L.stride_bytes;
        if (trace_on) {
            m.trace_layer  = (uint8_t)g_mod.moe_layers[(size_t)mlpos];
            m.trace_expert = (uint8_t)expert;
            ggml_moe_trace_current_graph_context(&m.trace_run_index, &m.trace_graph_seq, &m.trace_phase);
            m.trace_wait_episode = ggml_moe_trace_data_wait_begin(
                m.trace_run_index, m.trace_graph_seq, m.trace_phase,
                m.trace_layer, MOE_TRACE_NA_U8, m.trace_expert, 0);
        }
    }

    // ── ★Phase A ledger — 요청 인덱스 순서대로 hit/miss·slot·논리 tick(= ordered u16 expert IDs).
    //    claim 이후라 miss 의 slot 이 확정돼 있다. ──
    if (pa_on) {
        for (int32_t i = 0; i < n_ids; i++) {
            const int32_t si = p.hit_slot[(size_t)i];
            int32_t slot;
            if (si >= 0)                        slot = si;
            else if (p.req_join[(size_t)i] >= 0) slot = g_mod.pending_ops[(size_t)p.req_join[(size_t)i]].m.slot;
            else                                 slot = p.miss[(size_t)p.req_miss[(size_t)i]].slot;
            ggml_moe_phasea_acquire_expert(ids[i], si >= 0 ? 1 : 0, slot, p.pa_tick[(size_t)i]);
        }
    }
    return 0;
}

}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
// ===========================================================================
// §9. VEH — 보호 페이지 터치 감지 (v1 승계).
// ===========================================================================

thread_local bool t_in_veh_handler = false;

int32_t find_interval_locked_free(uintptr_t addr) {
    const auto & v = g_mod.prot_intervals;
    size_t lo = 0, hi = v.size();
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (v[mid].start <= addr) lo = mid + 1; else hi = mid;
    }
    if (lo == 0) return -1;
    const ProtInterval & iv = v[lo - 1];
    if (addr >= iv.start && addr < iv.end) return (int32_t)(lo - 1);
    return -1;
}

LONG WINAPI moe_direct_veh_handler(PEXCEPTION_POINTERS info) {
    if (t_in_veh_handler) return EXCEPTION_CONTINUE_SEARCH;
    PEXCEPTION_RECORD er = info->ExceptionRecord;
    if (er->ExceptionCode != (DWORD)EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    if (er->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    uintptr_t access_kind = (uintptr_t)er->ExceptionInformation[0];
    uintptr_t fault_addr = (uintptr_t)er->ExceptionInformation[1];
    int32_t idx = find_interval_locked_free(fault_addr);
    if (idx < 0) return EXCEPTION_CONTINUE_SEARCH;

    t_in_veh_handler = true;

    const ProtInterval & iv = g_mod.prot_intervals[(size_t)idx];
    uintptr_t page = fault_addr & ~(uintptr_t)0xFFFu;
    uint32_t page_idx = iv.page_base_index + (uint32_t)((page - iv.start) / 4096);

    g_mod.touch_events.fetch_add(1, std::memory_order_relaxed);

    uint8_t prev = g_mod.page_touched[page_idx].exchange(1, std::memory_order_acq_rel);
    if (prev == 0) {
        uint64_t upcount = g_mod.touch_pages.fetch_add(1, std::memory_order_relaxed) + 1;
        uint64_t seq = g_mod.touch_ring_next.fetch_add(1, std::memory_order_relaxed) + 1;
        size_t slot = (size_t)((seq - 1) % g_mod.touch_ring.size());
        TouchLogEntry & e = g_mod.touch_ring[slot];
        e.addr = fault_addr;
#if defined(_M_X64)
        e.rip = (uintptr_t)info->ContextRecord->Rip;
#elif defined(_M_IX86)
        e.rip = (uintptr_t)info->ContextRecord->Eip;
#elif defined(_M_ARM64)
        e.rip = (uintptr_t)info->ContextRecord->Pc;
#else
        e.rip = 0;
#endif
        e.tid = GetCurrentThreadId();
        e.tensor_idx = iv.tensor_idx;
        e.seq.store(seq, std::memory_order_release);

        if (upcount >= g_mod.touch_abort_threshold) {
            TerminateProcess(GetCurrentProcess(), MOE_DIRECT_VEH_KILL_EXIT_CODE);
        }
    }

    if (access_kind != 0) {
        t_in_veh_handler = false;
        return EXCEPTION_CONTINUE_SEARCH;
    }

    DWORD old_protect = 0;
    BOOL ok = VirtualProtect((LPVOID)page, 4096, PAGE_READONLY, &old_protect);
    t_in_veh_handler = false;
    if (!ok) return EXCEPTION_CONTINUE_SEARCH;
    return EXCEPTION_CONTINUE_EXECUTION;
}

// ===========================================================================
// §10. 계측 JSONL 쓰기 (v1 승계 — 헤더 필드는 v2 값).
// ===========================================================================

// ===========================================================================
// §10.5 ★B1 영속 I/O dispatcher (PHASEB_DESIGN §2 — FROZEN)
//
//   CPU acquire/release ── control_q ──> dispatcher ── IoChannel[QD]
//             │                              │
//             └──── per-command CV <──────────┘
//
//   ★단독 writer = dispatcher: slots.{owner,generation,last_used,lease,state}·record_table·
//     demand_tick·pending·channel·ReadFile submit/retire·digest. 예외 reader는
//     ggml_moe_direct_ptr()뿐(release-store publish된 (slot,generation)만).
//   ★락 순서: g_mod.mutex → q_mutex (역방향 금지). 생산자는 q_mutex 대기 구간에서
//     g_mod.mutex를 절대 보유하지 않는다(§2 핸드셰이크 ②).
//   ★lost-wake 방지: enqueue+SetEvent는 q_mutex 아래에서 한 묶음으로 하고, dispatcher는
//     큐 공백을 같은 q_mutex 아래에서 확인한 뒤에만 ResetEvent 한다.
// ===========================================================================


// ===========================================================================
// ★★PB virtual 원자 A1 — ①record 운영축 · ②child 물리축 상태기계
//    권위 = `SPEC_PB_VIRTUAL_RUNTIME_R1.md` §2-0 A0-1(축 분리) · A0-2(wave) · A0-3(증감 시점).
//    ★전 함수 mode-neutral(always-on) — IO3·PB 사이드카가 꺼져 있어도 돈다.
// ===========================================================================











// ===========================================================================
// ★§Z-⑥ 원자 A3 — mode=virtual child 팬아웃(제출·회수·실패 규칙)
//
// 계약(SPEC_REPACK_V3 §3-2·§3-4):
//   · 채널 = ChildIo 단위. record 1건이 part 수만큼의 물리 read 로 흩어진다.
//   · **성공 terminal 은 record 당 1회** — 전 child 성공이어야 op terminal 이 SUCCESS 로 선다.
//   · 첫 실패 = `failure_latch` → 형제 CancelIoEx → **전 child terminal drain 후에만** op
//     terminal(=slot unpin). success-after-cancel 은 publish 로 가지 않는다.
//   · child 완료는 lease·record table·deps 를 **직접 건드리지 않는다**(op terminal 만이 한다).
//     그래서 아래 함수들은 `disp_op_terminal_locked` 를 호출할 뿐 publish 를 스스로 하지 않는다.
// ===========================================================================

// ★`v3_active()` / `v3_parts_of()` 의 정의는 §4 유틸리티(`resolve_part_addr` 바로 아래)로
//   올라갔다 — PB virtual A1 의 ①②축 상태기계(§10.5 pending 풀 부근)가 이 둘을 쓰는데
//   그 자리가 여기보다 앞이기 때문이다. 정의는 **한 벌**이고 여기서는 이름만 계속 쓴다.




// child 완료 1건을 회계한다. 반환 true = 이 완료로 op 이 terminal 에 도달했다.
// ★out-of-order 안전: 어떤 순서로 와도 remaining 이 0 이 되는 **마지막** 완료만 op terminal 을
//   부른다. immediate(동기)·pending(비동기) 혼합도 같은 경로로 합류한다.
// ===========================================================================
// ★§Z-⑥ A5 — IO3 emission (호출자는 g_mod.mutex 보유). OFF 면 전부 즉시 반환한다.
// ===========================================================================
// ===========================================================================
// IO3 파일 3종 기록(finalize 1회 · 요청 중 파일 I/O 0 — §5-1).
// ===========================================================================
// ★A4 copy queue — bounded MPSC. 생산자=dispatcher(회수 직후) · 소비자=copy worker 1.
//   ★dispatcher 는 여기 넣기만 하고 memcpy 를 하지 않는다(§3-5 "dispatcher inline copy 금지").
// ★S1 — 공용 copy credit 예약/반납(호출자는 g_mod.mutex 보유). 예약은 **제출 전**에만 한다.



// ===========================================================================
// ★§Z-⑥ 원자 A6 — 프리필 coalescing(SPEC_REPACK_V3 §3-6 7조건 + §5-4 8번째 MUST)
//
// 계획은 **제출 전에 전부 확정**된다(§5-5): 참가 집합·대구간 기하·copy obligation 이 read
// 제출 앞에서 불변이 되고, 그 뒤에는 회수·scatter 만 남는다. 조건이 하나라도 안 서면 계획은
// 서지 않고 **기존 단독 경로가 그대로** 돈다(합치지 않는 것이 언제나 안전한 후퇴다).
// ===========================================================================



// ===========================================================================
// §10.6 ★B2 프리페치 admission (PHASEB_DESIGN §4 · SPEC_PHASEB_WIRE §4.2·§6)
// ===========================================================================

// ===========================================================================
// ★★PB virtual A2 — N_cap(A0-5) · admission credit 이중 조건(A0-4) · demand predicate
// ===========================================================================



// ===========================================================================
// ★§Z-⑥ 원자 A2 — mode=virtual 자원 확보(소스 핸들 집합·슬롯 풀·registry·보호)
//
// 사양 권위: `SPEC_REPACK_V3.md` §3-1(소스 핸들)·§3-3(주소 계약)·§3-7(등록·보호 승계)·
//   §2-4(슬롯 산술) / §4-7(n_slots >= n_expert).
//
// ★A2 도 아직 seal 을 성립시키지 않는다 — child 상태기계(A3)·바운스(A4)가 없으므로 경계는
//   여전히 기동 거부다. 다만 A1 과 달리 A2 는 **자원을 실제로 잡는다**. 그래서 경계에서
//   잡은 것을 전량 되돌린다(virtual_release_locked): 실패한 seal 이 핸들·커밋 메모리·
//   PAGE_NOACCESS 를 남기면 그건 "반쯤 세운 소비자"의 다른 이름이다. v2 경로도 자원 확보
//   이후의 모든 실패 분기에서 pool/handle 을 되돌리므로, 이건 같은 규율의 연장이다.
// ===========================================================================

// A2 자원 증거(테스트 전용 관측 — A1 의 stage 래치와 같은 계열). 경계 teardown 이 자원을
// 지우므로 "자원이 살아 있던 동안의 사실"은 여기 스냅샷으로만 남는다. 프로덕션은 안 읽는다.



}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
// A2 가 g_mod 에 심은 것을 **획득 역순으로** 전량 되돌린다. 부분 확보 상태에서도 안전해야
// 하므로(acquire 의 내부 실패 분기가 그대로 이걸 부른다) 각 항목은 비어 있어도 무해하다.
// ① 보호 원복 — 등록 텐서를 seal 시도 전 보호값으로 되돌린다. 여기서 안 풀면 실패한
//    seal 이 모델 텐서를 PAGE_NOACCESS 로 남기고, VEH 도 없는 상태라 다음 접근이 곧
//    하드 폴트다(A2 경계는 "거부"이지 "중독"이 아니어야 한다).
// ★★V3HEAP — 이 원자를 따로 떼어둔 이유: 구 판은 원복이 **실패 경로에만** 있었고 성공
//   teardown(`ggml_moe_direct_reset_for_test`)은 `prot_intervals` 를 기록만 버렸다. 그래서
//   호출자 소유 메모리가 PAGE_NOACCESS 인 채 해방됐다(V3PROD 힙 손상의 직접 원인). 두
//   경로가 **같은 코드**를 부르게 한다 — 같은 계약의 문면 사본을 두지 않는다(허브 §2-8).
void restore_prot_intervals_locked() {
    for (const auto & iv : g_mod.prot_intervals) {
        if (iv.old_protect == 0) continue;   // 미기록(v2 경로가 만든 항목) — 손대지 않는다
        DWORD prev = 0;
        VirtualProtect((LPVOID)iv.start, (SIZE_T)(iv.end - iv.start), (DWORD)iv.old_protect, &prev);
    }
    g_mod.prot_intervals.clear();
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes

// R-B5a facade (vii): SEAL -> dispatcher configure / start / unwind / stop.
// The seal path no longer writes the dispatcher-owned members of capture section E-3-d
// directly; it states its intent through the members below. Each one performs exactly the
// statements it replaced, in the same order, with the caller's `g_mod.mutex` already held -
// the `_locked` convention is unchanged, no lock is taken or released here, and no member
// adds, drops or reorders a statement. RB_DESIGN.md:1221 section 2-6, facade row (vii).
// These comments are deliberately plain: a ruled banner would be collected as a new banner
// section by rb_map.py and would shift every later section ordinal, which mis-binds the whole
// SEC2CL cluster attribution (measured on the first attempt at this atom).

// K4: the I/O path selector and the queue-depth echo.
inline void dfac_io_path_set_locked(bool persistent, const char * src, bool qualified) {
    g_mod.io_path_persistent = persistent;
    g_mod.io_path_source     = src;
    g_mod.io_path_qualified  = qualified;
}
inline void dfac_io_path_persistent_set_locked(bool v) { g_mod.io_path_persistent = v; }
inline void dfac_io_path_source_set_locked(const std::string & v) { g_mod.io_path_source = v; }
inline void dfac_io_path_qualified_set_locked(bool v) { g_mod.io_path_qualified = v; }
inline void dfac_qd_effective_set_locked(int32_t v) { g_mod.qd_effective = v; }
inline void dfac_qd_source_set_locked(const std::string & v) { g_mod.qd_source = v; }

}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the 6 definitions below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
// `inline` is removed: an inline function odr-used from a unit that carries no
// definition of it does not link (RB3B_ATOM_SPEC.json :: declarator_changes).
namespace ggml_moe_direct_impl {
// K9: the resolved I/O parameter vector and the coalescing configuration.
void dfac_io_param_val_set_locked(size_t i, int32_t v) { g_mod.v3_io_param_val[i] = v; }
void dfac_io_param_src_set_locked(size_t i, uint8_t v) { g_mod.v3_io_param_src[i] = v; }
void dfac_coal_phase_set_locked(uint8_t v) { g_mod.v3_coal_phase = v; }
void dfac_coal_min_bytes_set_locked(uint64_t v) { g_mod.v3_coal_min_bytes = v; }
void dfac_coal_max_bytes_set_locked(uint64_t v) { g_mod.v3_coal_max_bytes = v; }
void dfac_coal_buffers_set_locked(int32_t v) { g_mod.v3_coal_buffers = v; }
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
inline void dfac_coal_base_set_locked(void * v) { g_mod.v3_coal_base = v; }
inline void dfac_coal_stride_set_locked(size_t v) { g_mod.v3_coal_stride = v; }
inline void dfac_coal_count_set_locked(int32_t v) { g_mod.v3_coal_count = v; }
inline void dfac_coal_assign_locked(size_t n) { g_mod.v3_coal.assign(n, V3CoalRead{}); }
inline void dfac_coal_plan_ops_reserve_locked(size_t n) { g_mod.v3_coal_plan.ops.reserve(n); }
inline void dfac_coal_plan_cand_assign_locked(size_t n, int32_t v) {
    g_mod.v3_coal_plan.cand.assign(n, v);
}
// The three `io_coalesce_*` environment keys used to reach their members by ADDRESS, through a
// local table of pointers, so the write site was the address-taking row and no call could stand
// in for it. The table carries a member selector now and the loop stores through this member.
// Same two stores, same casts, same order; the parse, the range test, the error text and the
// early return above are untouched.
// Selector 0 = min bytes, 1 = max bytes, 2 = buffers. Plain integers rather than an enum:
// an enum would be a new TYPE in the after tree and layer 1 refuses an unmapped appearance.
inline void dfac_coal_env_set_locked(uint8_t sel, unsigned long long v) {
    if (sel == 0)      g_mod.v3_coal_min_bytes = (uint64_t)v;
    else if (sel == 1) g_mod.v3_coal_max_bytes = (uint64_t)v;
    else               g_mod.v3_coal_buffers   = (int32_t)v;
}

// K6: the copy ring, the copy worker event and the bounce staging pool.
inline void dfac_staging_base_set_locked(void * v) { g_mod.v3_staging_base = v; }
inline void dfac_staging_stride_set_locked(size_t v) { g_mod.v3_staging_stride = v; }
inline void dfac_staging_count_set_locked(int32_t v) { g_mod.v3_staging_count = v; }
inline void dfac_staging_busy_assign_locked(size_t n, uint8_t v) {
    g_mod.v3_staging_busy.assign(n, v);
}
inline void dfac_copy_event_set_locked(HANDLE v) { g_mod.v3_copy_event = v; }
// Each release member delegates to one-liners for the same reason the prefetch unwind does: a
// body that reproduced the replaced run verbatim would leave that run in the unit, and the move
// proof requires a declared before text to occur ZERO times after the atom.
inline void dfac_copy_q_clear_locked()   { g_mod.v3_copy_q.clear(); }
inline void dfac_copy_q_shrink_locked()  { g_mod.v3_copy_q.shrink_to_fit(); }
inline void dfac_copy_ring_counts_reset_locked() {
    g_mod.v3_copy_head  = 0;
    g_mod.v3_copy_count = 0;
    g_mod.v3_copy_cap   = 0;
}
}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
// `inline` is removed: an inline function odr-used from a unit that carries no
// definition of it does not link (RB3B_ATOM_SPEC.json :: declarator_changes).
namespace ggml_moe_direct_impl {
void dfac_copy_outstanding_set_locked(int32_t v) { g_mod.v3_copy_outstanding = v; }
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
inline void dfac_copy_ring_release_locked() {
    dfac_copy_q_clear_locked();
    dfac_copy_q_shrink_locked();
    dfac_copy_ring_counts_reset_locked();
    dfac_copy_outstanding_set_locked(0);
}
inline void dfac_staging_base_free_locked() {
    if (g_mod.v3_staging_base) {
        VirtualFree(g_mod.v3_staging_base, 0, MEM_RELEASE);
        g_mod.v3_staging_base = nullptr;
    }
}
inline void dfac_staging_busy_clear_locked() { g_mod.v3_staging_busy.clear(); }
}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
// `inline` is removed: an inline function odr-used from a unit that carries no
// definition of it does not link (RB3B_ATOM_SPEC.json :: declarator_changes).
namespace ggml_moe_direct_impl {
void dfac_staging_pool_release_locked() {
    dfac_staging_base_free_locked();
    dfac_staging_stride_set_locked(0);
    dfac_staging_count_set_locked(0);
    dfac_staging_busy_clear_locked();
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes
inline void dfac_coal_base_free_locked() {
    if (g_mod.v3_coal_base) {
        VirtualFree(g_mod.v3_coal_base, 0, MEM_RELEASE);
        g_mod.v3_coal_base = nullptr;
    }
}
inline void dfac_coal_clear_locked()           { g_mod.v3_coal.clear(); }
inline void dfac_coal_plan_ops_clear_locked()  { g_mod.v3_coal_plan.ops.clear(); }
inline void dfac_coal_plan_cand_clear_locked() { g_mod.v3_coal_plan.cand.clear(); }
}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
// `inline` is removed: an inline function odr-used from a unit that carries no
// definition of it does not link (RB3B_ATOM_SPEC.json :: declarator_changes).
namespace ggml_moe_direct_impl {
void dfac_coal_pool_release_locked() {
    dfac_coal_base_free_locked();
    dfac_coal_stride_set_locked(0);
    dfac_coal_count_set_locked(0);
    dfac_coal_clear_locked();
    dfac_coal_plan_ops_clear_locked();
    dfac_coal_plan_cand_clear_locked();
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes

// (vii) start / stop hand-off. The inherited r4S-1 row declares `v3_copy_stop_and_join` as a
// WRAPPED rollback token, so the seal stages reach it through this member. The callee itself is
// unchanged and stays dispatcher-owned, so capture section E-3-e still names it as the SEAL to
// dispatcher callee; what moves is which SEAL body makes the call.
inline void dfac_dctl_copy_stop_and_join() { v3_copy_stop_and_join(); }

}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
// `inline` is removed: an inline function odr-used from a unit that carries no
// definition of it does not link (RB3B_ATOM_SPEC.json :: declarator_changes).
namespace ggml_moe_direct_impl {
// K11: the test-only late-seal failure arm.
void dfac_test_fail_after_acquire_set_locked(int32_t v) {
    g_mod.v3_test_fail_after_acquire = v;
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes

// K13: the PhaseB prefetch configuration, noted at seal and unwound on failure.
// The three clear members are separate one-liners on purpose. A facade body that reproduced
// the replaced run verbatim would leave that run in the unit, and the move proof requires a
// declared before text to occur ZERO times after the atom (rb_move_proof.py:4128).
inline void dfac_op_stamp_clear_locked()     { g_mod.pb_op_stamp.clear(); }
inline void dfac_pred_snap_clear_locked()    { g_mod.pb_pred_snap.clear(); }
inline void dfac_served_graph_clear_locked() { g_mod.pb_served_graph.clear(); }
inline void dfac_prefetch_unwind_locked() {
    g_mod.pb_prefetch_on = false;
    g_mod.pb_k           = 0;
    g_mod.pb_n           = 0;
    g_mod.pb_collect     = false;
    dfac_op_stamp_clear_locked();
    dfac_pred_snap_clear_locked();
    dfac_served_graph_clear_locked();
}
inline void dfac_prefetch_note_flags_locked(size_t n_graph) {
    g_mod.pb_served_graph.assign(n_graph, GGML_MOE_PHASEB_NA_U32);
    g_mod.pb_prefetch_on = ggml_moe_phaseb_prefetch_on();
    g_mod.pb_k = ggml_moe_phaseb_prefetch_k();
    g_mod.pb_n = ggml_moe_phaseb_prefetch_n();
}
inline void dfac_prefetch_note_census_locked(int32_t n_slots) {
    g_mod.pb_collect  = ggml_moe_phaseb_collect_on();
    g_mod.pb_censused = false;
    if (g_mod.pb_collect) {
        g_mod.pb_op_stamp.assign((size_t)n_slots, PbOpStamp{});
        g_mod.pb_pred_snap.assign(ModuleState::PB_PRED_SNAP_MAX, ModuleState::PbPredSnap{});
        g_mod.pb_pred_snap_next = 0;
    }
}

void virtual_release_locked() {
    restore_prot_intervals_locked();
    g_mod.reg_tensor_info.clear();
    g_mod.page_touched.clear();
    g_mod.touch_ring.clear();
    g_mod.touch_ring_next.store(0, std::memory_order_relaxed);
    g_mod.touch_events.store(0, std::memory_order_relaxed);
    g_mod.touch_pages.store(0, std::memory_order_relaxed);
    g_mod.registry.clear();
    xstore_invalidate_epoch_locked();   // E-2a adapter map lifecycle stage 4
    g_mod.att_raw = g_mod.att_managed = g_mod.att_excluded_nextn = 0;
    g_mod.att_all = g_mod.att_host = g_mod.att_nonhost = 0;
    g_mod.att_n_slots = g_mod.att_n_slots_required = 0;

    // ①‴ ★★R-2(교차 r1 F1) — copy **큐**도 여기서 비운다. `v3_copy_stop_and_join()` 은
    //    `running==false` 면 즉시 반환하므로, worker 가 **뜨지 못한** 실패 경로에서는 큐·cap·
    //    미결 크레딧이 아무에게도 안 치워진 채 남는다(§2-2 "잔존 0" 위반). 정상 경로에서는
    //    stop_and_join 이 이미 비운 뒤라 여기 재실행은 무해한 멱등 연산이다.
    dfac_copy_ring_release_locked();
    if (g_mod.v3_copy_event && !g_mod.v3_copy_running.load(std::memory_order_acquire)) {
        CloseHandle(g_mod.v3_copy_event);
        dfac_copy_event_set_locked(nullptr);
    }

    // ①′ ★A4 바운스 staging 풀(copy worker 정지는 락 밖 — `v3_copy_stop_and_join()`)
    dfac_staging_pool_release_locked();

    // ①″ ★A6 coalesce 버퍼 풀(같은 규율 — 소비자[copy worker] 정지 뒤에만 반납된다)
    dfac_coal_pool_release_locked();

    // ② 슬롯 풀
    if (g_mod.pool_base) { VirtualFree(g_mod.pool_base, 0, MEM_RELEASE); g_mod.pool_base = nullptr; }
    g_mod.pool_bytes = 0;
    g_mod.n_slots = 0;
    g_mod.slots.clear();
    g_mod.record_table.clear();

    // ③ 소스 핸들 집합
    for (HANDLE h : g_mod.source_files) {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
    g_mod.source_files.clear();

    // ④ 레이아웃
    g_mod.arch.clear();
    g_mod.n_layer = g_mod.n_expert = g_mod.n_expert_used = 0;
    g_mod.moe_layers.clear();
    g_mod.layer_to_mlpos.clear();
    g_mod.layers.clear();
    g_mod.sources.clear();
    g_mod.align_bytes = g_mod.slot_stride_max = g_mod.bin_file_bytes = 0;
    g_mod.ref_profile_id.clear();
    g_mod.ref_expect_sha256.clear();
    g_mod.repack_manifest_sha256.clear();
    g_mod.routed_scope = "all";
    g_mod.manifest_mode = ManifestMode::BIN;
}

// A2 본체. 성공 시 g_mod 에 (핸들 집합·슬롯 풀·레이아웃·registry·보호)가 확정된다.
// 실패 시엔 자기가 심은 것을 전부 되돌린 뒤 false(err 에 사유).
// ★호출 시점 전제: A1 전 관문 통과 — pm 의 virtual 좌표는 전항 검증됐고, 등록 집합
//   (name_to_lp·pending_kept)과 각 텐서의 형상 대조도 §2-5 live 단계에서 이미 섰다.
//   그래서 여기서 그 검사들을 다시 하지 않는다(같은 계약의 두 번째 사본을 만들지 않는다 — §2-8).
bool virtual_acquire_locked(ParsedManifest & pm, uint64_t budget_bytes, uint32_t touch_abort,
                            const std::string & manifest_sha,
                            const std::unordered_map<std::string, std::pair<int32_t, int32_t>> & name_to_lp,
                            const std::vector<char> & pending_kept,
                            std::string & err) {
    // ---- ① §3-1 N-shard 소스 핸들 집합 ----
    // bin 의 단일 `experts.bin` 핸들에 대응하는 축이다. 플래그는 bin 경로와 같은
    // NO_BUFFERING|OVERLAPPED 이고, 공유 모드는 §3-1 대로 FILE_SHARE_READ.
    // ★dense mmap 과의 병행: llama.cpp 로더는 같은 shard 를 `ggml_fopen(...,"rb")` 로 열고
    //   그 핸들에 `CreateFileMappingA(PAGE_READONLY)` 를 건다(llama-mmap.cpp:88·542-544).
    //   양쪽 다 읽기 접근이고 CRT 의 fopen 은 읽기 공유를 허용하므로 이 조합이 성립한다는
    //   것이 코드 대조상의 판단이다 — **실제 병행 기동 실측은 §8 스모크 항목**이고 이 원자의
    //   selftest 는 (합성 shard 를 매핑하지 않으므로) 그 경로를 행사하지 않는다.
    // ★실패가 도달 가능한 조건: NTFS 압축·암호화 파일이나 일부 원격 볼륨은 NO_BUFFERING 을
    //   거부한다. 그 경우 조용한 우회 없이 기동 중단이다.
    for (size_t i = 0; i < pm.sources.size(); i++) {
        const SourceInfo & si = pm.sources[i];
        const std::wstring wpath = utf8_to_wide(si.path);
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            err = "cannot open source shard " + std::to_string(i) + " for unbuffered overlapped reads: " +
                  si.path + " (GetLastError=" + std::to_string((unsigned long)GetLastError()) +
                  "; NO_BUFFERING is refused on compressed/encrypted files and on some remote volumes)";
            virtual_release_locked();
            return false;
        }
        g_mod.source_files.push_back(h);
        // ★핸들 대조 — "이 핸들이 manifest 의 그 소스인가"를 **핸들 자신**에게 묻는다.
        //   §2-4 bracket EOF 산술이 `sources[].bytes` 를 전제로 섰는데 실제 읽기는 이 핸들로
        //   나가므로, 그 전제를 읽을 대상에 결속하는 것이 이 검사의 몫이다.
        LARGE_INTEGER fsz{};
        if (!GetFileSizeEx(h, &fsz)) {
            err = "GetFileSizeEx failed on the acquired handle for " + si.path +
                  " (GetLastError=" + std::to_string((unsigned long)GetLastError()) + ")";
            virtual_release_locked();
            return false;
        }
        if ((uint64_t)fsz.QuadPart != si.bytes) {
            err = "source shard size on the acquired handle(" + std::to_string((uint64_t)fsz.QuadPart) +
                  ") != manifest sources[].bytes(" + std::to_string(si.bytes) + ") for " + si.path;
            virtual_release_locked();
            return false;
        }
    }
    g_v3_stage.store(V3_STAGE_HANDLES, std::memory_order_relaxed);

    // ---- ② 슬롯 풀(§4-7 — bin 과 같은 식. virtual 은 stride 축이 slot_stride_max 이고
    //      그것이 0 이 아님은 validate_manifest_v3 의 invariant v3-5 가 이미 세웠다) ----
    const int32_t n_slots = (int32_t)(budget_bytes / pm.slot_stride_max);
    if (n_slots < pm.n_expert) {
        err = "slot capacity invariant violated: n_slots(" + std::to_string(n_slots) + ") < n_expert(" +
              std::to_string(pm.n_expert) + ") - budget_bytes(" + std::to_string(budget_bytes) +
              ") < n_expert*slot_stride_max(" + std::to_string((uint64_t)pm.n_expert * pm.slot_stride_max) + ")";
        virtual_release_locked();
        return false;
    }
    const size_t pool_bytes = (size_t)n_slots * (size_t)pm.slot_stride_max;
    void * pool_base = VirtualAlloc(nullptr, pool_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!pool_base) {
        err = "VirtualAlloc(slot pool) failed err=" + std::to_string((unsigned long)GetLastError());
        virtual_release_locked();
        return false;
    }
    g_mod.pool_base  = pool_base;
    g_mod.pool_bytes = pool_bytes;
    g_mod.n_slots    = n_slots;

    // ---- ③ 레이아웃 확정 ----
    g_mod.arch          = pm.arch;
    g_mod.n_layer       = pm.n_layer;
    g_mod.n_expert      = pm.n_expert;
    g_mod.n_expert_used = pm.n_expert_used;
    g_mod.moe_layers    = pm.moe_layers;
    g_mod.layer_to_mlpos.clear();
    for (size_t i = 0; i < pm.moe_layers.size(); i++) g_mod.layer_to_mlpos[pm.moe_layers[i]] = (int32_t)i;
    g_mod.layers          = std::move(pm.layers);
    g_mod.sources         = std::move(pm.sources);
    g_mod.align_bytes     = pm.align_bytes;
    g_mod.slot_stride_max = pm.slot_stride_max;
    g_mod.bin_file_bytes  = 0;   // ★virtual 엔 experts.bin 이 없다(§0) — bin 축은 0 으로 남는다
    g_mod.ref_profile_id  = pm.ref_profile_id;
    g_mod.ref_expect_sha256 = pm.ref_expect_sha256;
    g_mod.repack_manifest_sha256 = manifest_sha;   // 재팩물 단위 identity(v2 R4 축과 같은 의미)
    g_mod.routed_scope    = pm.routed_scope;
    g_mod.data_file       = INVALID_HANDLE_VALUE;  // virtual 은 단일 bin 핸들을 쓰지 않는다
    g_mod.touch_abort_threshold = touch_abort;

    // ★A3 — 층별 legacy-equivalent record bytes(§7 D-A2 분자축). 물리 Σ region 과 구분되는
    //   회계 축이라 여기서 1회 확정한다(io_finish 가 매번 재계산하지 않게).
    for (auto & L : g_mod.layers) {
        uint64_t payload = 0, expect = 0;
        for (const auto & p : L.parts) {
            payload += p.slice_bytes;                              // 전 항 검증된 값의 합
            expect  += p.aligned ? p.region : p.slice_bytes;        // ★A4 — 슬롯에 남는 바이트
        }
        L.legacy_record_bytes  = align_up(payload, pm.legacy_align_bytes);
        L.child_bytes_expected = expect;
    }

    g_mod.slots = std::vector<Slot>((size_t)n_slots);
    for (int32_t i = 0; i < n_slots; i++) {
        g_mod.slots[(size_t)i].base = (char *)pool_base + (size_t)i * (size_t)g_mod.slot_stride_max;
    }
    g_mod.record_table = std::vector<RecordEntry>((size_t)g_mod.moe_layers.size() * (size_t)g_mod.n_expert);
    g_v3_stage.store(V3_STAGE_SLOTPOOL, std::memory_order_relaxed);

    // ---- ④ registry(R_host) + attestation(§3-7 v2 승계 · 부록A raw/managed/excluded) ----
    // execution 제외분(pending_kept[i]==0)은 registry/보호/슬롯 대상이 아니다(일반 mmap 잔류).
    int32_t att_raw = 0, att_managed = 0, att_host = 0, att_nonhost = 0;
    g_mod.registry.clear();
    for (size_t i = 0; i < g_mod.pending.size(); i++) {
        const PendingReg & pr = g_mod.pending[i];
        att_raw++;
        if (i < pending_kept.size() && !pending_kept[i]) continue;
        att_managed++;
        const auto lp_it = name_to_lp.find(pr.name);
        if (lp_it == name_to_lp.end()) {
            // 도달 불가(A1 등록 3집합이 양방향 폐합을 이미 세웠다) — 그래도 조용히 스킵하면
            // registry 가 소리 없이 비게 되므로 명시 중단으로 남긴다.
            err = "internal: registered tensor is absent from the manifest index after the section 2-5 checks: " + pr.name;
            virtual_release_locked();
            return false;
        }
        if (pr.is_host) {
            att_host++;
            g_mod.registry[pr.tensor] = RegEntry{ lp_it->second.first, lp_it->second.second };
        } else {
            att_nonhost++;
        }
    }
    g_mod.att_raw            = att_raw;
    g_mod.att_managed        = att_managed;
    g_mod.att_excluded_nextn = att_raw - att_managed;
    g_mod.att_all            = att_managed;
    g_mod.att_host           = att_host;
    g_mod.att_nonhost        = att_nonhost;
    g_mod.att_n_slots          = n_slots;
    g_mod.att_n_slots_required = g_mod.n_expert;

    // ---- ⑤ 등록 텐서 보호(PAGE_NOACCESS) + interval 표(R_host 만 · v2 §4 승계) ----
    g_mod.reg_tensor_info.clear();
    g_mod.reg_tensor_info.reserve(g_mod.registry.size());
    std::vector<ProtInterval> raw_intervals;
    raw_intervals.reserve(g_mod.registry.size());
    for (auto & kv : g_mod.registry) {
        const ggml_tensor * t = kv.first;
        const RegEntry & re = kv.second;
        const int32_t tidx = (int32_t)g_mod.reg_tensor_info.size();
        RegisteredTensorInfo info{};
        info.layer      = g_mod.moe_layers[(size_t)re.mlpos];
        info.part       = re.part_idx;
        info.addr_start = (uintptr_t)t->data;
        info.nbytes     = ggml_nbytes(t);
        g_mod.reg_tensor_info.push_back(info);

        const uintptr_t astart = (uintptr_t)align_up(info.addr_start, 4096);
        const uintptr_t aend   = (uintptr_t)align_down(info.addr_start + info.nbytes, 4096);
        if (aend <= astart) continue;
        DWORD old_protect = 0;
        if (!VirtualProtect((LPVOID)astart, (SIZE_T)(aend - astart), PAGE_NOACCESS, &old_protect)) {
            err = "VirtualProtect(PAGE_NOACCESS) failed err=" + std::to_string((unsigned long)GetLastError());
            // 여기까지 성공적으로 건 구간은 아직 로컬 raw_intervals 에만 있다. release 는
            // g_mod.prot_intervals 를 보고 원복하므로, 넘겨준 뒤에 불러야 원복이 실제로 돈다.
            g_mod.prot_intervals = std::move(raw_intervals);
            virtual_release_locked();
            return false;
        }
        ProtInterval iv{};
        iv.start = astart; iv.end = aend; iv.tensor_idx = tidx; iv.page_base_index = 0;
        iv.old_protect = (uint32_t)old_protect;
        raw_intervals.push_back(iv);
    }
    std::sort(raw_intervals.begin(), raw_intervals.end(),
              [](const ProtInterval & a, const ProtInterval & b) { return a.start < b.start; });
    uint32_t total_pages = 0;
    for (auto & iv : raw_intervals) { iv.page_base_index = total_pages; total_pages += (uint32_t)((iv.end - iv.start) / 4096); }
    g_mod.prot_intervals = std::move(raw_intervals);

    g_mod.page_touched = std::vector<std::atomic<uint8_t>>(total_pages);
    for (auto & a : g_mod.page_touched) a.store(0, std::memory_order_relaxed);
    g_mod.touch_ring = std::vector<TouchLogEntry>(256);
    g_mod.touch_ring_next.store(0, std::memory_order_relaxed);
    g_mod.touch_events.store(0, std::memory_order_relaxed);
    g_mod.touch_pages.store(0, std::memory_order_relaxed);

    g_v3_stage.store(V3_STAGE_BIND, std::memory_order_relaxed);   // ★A2 완주
    return true;
}

// A2 경계에서 자원이 살아 있는 동안의 사실을 스냅샷으로 남긴다(teardown 직전 1회).
void virtual_fill_probe_locked() {
    g_v3_probe = V3ResourceProbe{};
    g_v3_probe.valid              = true;
    g_v3_probe.n_handles          = (int32_t)g_mod.source_files.size();
    g_v3_probe.n_slots            = g_mod.n_slots;
    g_v3_probe.att_managed        = g_mod.att_managed;
    g_v3_probe.att_host           = g_mod.att_host;
    g_v3_probe.n_prot_intervals   = (int32_t)g_mod.prot_intervals.size();
    g_v3_probe.pool_bytes         = (uint64_t)g_mod.pool_bytes;
    g_v3_probe.slot_stride_max    = g_mod.slot_stride_max;
    for (HANDLE h : g_mod.source_files) {
        LARGE_INTEGER fsz{};
        if (h != INVALID_HANDLE_VALUE && GetFileSizeEx(h, &fsz)) {
            g_v3_probe.handle_bytes_total += (uint64_t)fsz.QuadPart;
        }
    }
    for (const auto & L : g_mod.layers) {
        for (const auto & p : L.parts) {
            uint64_t off = 0, nb = 0;
            resolve_part_addr(p, off, nb);   // ★`part_info()` 와 **같은** 분기를 통과시킨다
            g_v3_probe.part_addr.push_back({ off, nb });
        }
    }
}

// ===========================================================================
// §11. seal 공통 로직 (v2 — 프로덕션/테스트 공유). 호출자는 g_mod.mutex 보유.
//      반환 0=성공, <0=실패(err에 사유).
// ===========================================================================

// ===========================================================================
// ★R12 — SPEC_IO_METRICS_V3 §3-1: IO 형상 파라미터 6종의 **전용 env 해소**.
//
//   지위=**virtual 전용**이다. bin 은 record-QD 체제고 §5-2 가 bin 헤더의 이 영역을 0(N/A)으로
//   고정하므로 이 함수는 bin 경로를 건드리지 않는다.
//
//   ★"해소값 = 실제로 서는 값" 규율: 헤더가 echo 하는 값은 런타임이 실제로 지키는 값이어야
//   한다(§5-6 cap 검산이 그 값과 관측을 대조하기 때문이다). 그래서 형상이 강제하는 하한보다
//   낮은 요청은 두 갈래로 처분한다 —
//     · **기본값(source=default)이 하한보다 낮다** = 운영자 입력이 없는 상태이므로 **끌어올린다**
//       (A3 의 QD 팬아웃 하한 선례: `MOE_DIRECT_QD_DEFAULT=1` 만으로 다중 part 모델이 전부
//       막히는 것을 실측으로 확인해 채택된 규율. source 는 `default` 로 남는다 — 엔진이 고른
//       값이라는 뜻이 그대로 참이다).
//     · **운영자가 명시한 값이 하한보다 낮다** = **fail-close**. 조용히 다른 값으로 돌면서
//       그 값을 echo 하면 헤더가 거짓말을 한다.
//   ★미구현 노브의 처분도 같다: `io_bounce_copy_workers` 는 copy worker 1개 설계(§3-5 옵션 B)
//   라 1 이외의 명시값은 거부한다(§5 sweep {1,2} 는 후속 홉).
//   ★`io_qd_demand_reserve`·`io_qd_prefetch_child_max` 는 **현 원자에서 소비자가 없다**
//   (virtual 에 prefetch-origin op 이 생성되지 않는다 — 봉인 §J-7 실측). 해소·echo 는 하되
//   "집행된다"고 주장하지 않는다. §5-6 의 두 cap 은 관측(항상 0)이 값을 넘지 않음을 확인한다.
// ===========================================================================
bool v3_resolve_io_params_locked(int32_t max_parts, int32_t staging_count, std::string & err) {
    static const char * kName[6] = {
        "io_qd_total", "io_qd_demand_reserve", "io_qd_prefetch_child_max",
        "io_part_parallelism", "io_bounce_copy_workers", "io_bounce_copy_qd",
    };
    static const char * kKey[6] = {
        "MOE_DIRECT_IO_QD_TOTAL", "MOE_DIRECT_IO_QD_DEMAND_RESERVE",
        "MOE_DIRECT_IO_QD_PREFETCH_CHILD_MAX", "MOE_DIRECT_IO_PART_PARALLELISM",
        "MOE_DIRECT_IO_BOUNCE_COPY_WORKERS", "MOE_DIRECT_IO_BOUNCE_COPY_QD",
    };
    // §5 [후보] 기본값. ★[0] 만은 기존 레버(`MOE_DIRECT_QD`)의 기본값을 그대로 승계한다 —
    //   §5 의 성능 후보값(8)을 엔진 기본값으로 굳히는 것은 "마이크로벤치 전 동결 금지"에
    //   걸리고, 이 원자는 wire 정합이지 성능 기본값 결정이 아니다.
    const int32_t kDefault[6] = { MOE_DIRECT_QD_DEFAULT, 4, 4, 3, 1, 2 };
    // 형상이 강제하는 하한(0=하한 없음). [5]=copy 큐가 최소한 staging 칸 수만큼은 담아야 한다
    //   (그보다 작으면 큐 만석 fatal).
    // ★★R-1(FROZEN §2-0 A0-2) — 구 판의 `[0]=[3]=max_parts` 를 **철거**했다.
    //   그 두 하한이 "record 의 전 child 를 한 번에"를 강제해 **wave 기계를 원천 금지**하고
    //   있었다(교차 r1 F2 의 직접 원인 — `io_part_parallelism` 은 그 자체가 **wave 폭 노브**라
    //   하한을 두면 `W_l = min(P_l, iop)` 가 영원히 `P_l` 이 된다).
    //     · `[3] io_part_parallelism` = 하한 전면 철거(기본값 3 이 그대로 산다).
    //     · `[0] io_qd_total` = **인덱스별 하한으로 둘 수 없다** — 필요 하한이 `max_l W_l` 인데
    //       그 값이 뒤에 해소되는 `[3]` 에 의존해 순환이다. ⇒ 6종 해소 **뒤 후검사**로 옮겼다
    //       (아래 ★wave 형상 후검사).
    const int32_t kFloor[6]   = { 0, 0, 0, 0, 0, staging_count };

    auto parse_pos = [](const std::string & s, int32_t & out) -> bool {
        if (s.empty()) return false;
        char * endp = nullptr;
        const long v = strtol(s.c_str(), &endp, 10);
        if (!endp || *endp != '\0' || v <= 0) return false;
        out = (int32_t)v;
        return true;
    };
    // ★R14 후속(리드 조건 1) — 거부 진단문은 **출구를 함께 준다**: 요구 하한(있으면)과
    //   처방 2갈래(값 상향 / 키 unset → 기본값 경로)를 한 줄에 담는다. 거부만 하고 출구를
    //   안 알려주면 운영자가 행동을 못 한다.
    auto how_to_fix = [&](int i) {
        std::string s = " - fix: ";
        if (kFloor[i] > 0) s += "set " + std::string(kKey[i]) + " >= " + std::to_string(kFloor[i]) +
                                " (shape floor), or ";
        else               s += "set " + std::string(kKey[i]) + " to a positive integer <= 255, or ";
        s += "unset it to take the engine default " + std::to_string(kDefault[i]);
        if (kFloor[i] > kDefault[i]) s += " (which is then raised to the floor " + std::to_string(kFloor[i]) + ")";
        return s;
    };

    for (int i = 0; i < 6; i++) {
        int32_t val = kDefault[i];
        uint8_t src = 0;                                   // 0=default
        int32_t dedicated = 0;
        const bool has_dedicated = parse_pos(getenv_str(kKey[i]), dedicated);
        if (!has_dedicated && !getenv_str(kKey[i]).empty()) {
            err = std::string(kKey[i]) + "=\"" + getenv_str(kKey[i]) +
                  "\" is invalid (positive integers only) - fail-closed (a resolved IO parameter must "
                  "never fall back silently: the header echoes it and section 5-6 checks against it)" +
                  how_to_fix(i);
            return false;
        }
        if (has_dedicated) { val = dedicated; src = 2; }   // 2=dedicated_env
        if (i == 0) {
            // ★precedence(§3-1) — `MOE_DIRECT_QD` 는 io_qd_total 의 하위 호환 별칭이다.
            //   둘 다 설정: 각각 해소한 뒤 **불일치면 기동 거부**·일치면 source=dedicated_env.
            const std::string qs = getenv_str("MOE_DIRECT_QD");
            int32_t legacy = 0;
            const bool has_legacy = parse_pos(qs, legacy);
            // ★★R16(교차 r6 G6) — **별칭도 명시값이다**. 구 판은 여기서 규율을 두 갈래로 어겼다:
            //   ①invalid 별칭을 경고 후 기본값으로 **성공**시켰고 ②별칭·전용을 63 으로 **먼저
            //   clamp** 해서 명시 64 가 조용히 63 으로 축약됐으며, 별칭 64 ↔ 전용 100 도 둘 다
            //   63 이 돼 "일치"로 통과했다. 둘 다 "명시값=운영자 계약·조용한 보정 금지"의 실도달
            //   반례다. ⇒ **명시값 축에는 관용도 clamp 도 없다**. 기본값 축(엔진이 고른 값)의
            //   하한 끌어올림·상한은 그대로 유효하다 — 그건 운영자 입력이 아니라 엔진의 선택이다.
            if (!qs.empty() && !has_legacy) {
                err = "MOE_DIRECT_QD=\"" + qs + "\" is invalid (positive integers only) - fail-closed "
                      "(the alias is an explicit operator value too: falling back to the default would "
                      "run at a value the operator never asked for) - fix: set MOE_DIRECT_QD to a positive "
                      "integer <= " + std::to_string((int)MOE_DIRECT_QD_MAX) + ", or unset MOE_DIRECT_QD "
                      "to take the engine default " + std::to_string((int)MOE_DIRECT_QD_DEFAULT);
                return false;
            }
            if (has_legacy || has_dedicated) {
                // ★불일치 판정은 **clamp 전 원값**으로(clamp 후 비교는 64↔100 을 같은 값으로 만든다).
                if (has_legacy && has_dedicated && legacy != dedicated) {
                    err = "MOE_DIRECT_QD(" + std::to_string(legacy) + ") and MOE_DIRECT_IO_QD_TOTAL(" +
                          std::to_string(dedicated) + ") are different values - fail-closed "
                          "(section 3-1 precedence; compared on the raw values, before any cap) - fix: "
                          "unset one of the two keys, or set both to the same value (MOE_DIRECT_QD is the "
                          "backward-compatible alias of io_qd_total; the shape floor here is " +
                          std::to_string(max_parts) + ")";
                    return false;
                }
                const int32_t raw = has_dedicated ? dedicated : legacy;
                if (raw > MOE_DIRECT_QD_MAX) {
                    err = std::string(has_dedicated ? "MOE_DIRECT_IO_QD_TOTAL" : "MOE_DIRECT_QD") + "=" +
                          std::to_string(raw) + " exceeds the channel hard cap " +
                          std::to_string((int)MOE_DIRECT_QD_MAX) + " - fail-closed (an explicit value is "
                          "not quietly clamped: the engine would run at a depth the operator never asked "
                          "for and echo that other value) - fix: set it to <= " +
                          std::to_string((int)MOE_DIRECT_QD_MAX) + ", or unset " +
                          (has_dedicated ? "MOE_DIRECT_IO_QD_TOTAL" : "MOE_DIRECT_QD") +
                          " to take the engine default " + std::to_string((int)MOE_DIRECT_QD_DEFAULT);
                    return false;
                }
                val = raw;
                src = has_dedicated ? 2 : 3;               // 2=dedicated_env · 3=legacy_alias
            }
        }
        if (i == 4 && val != 1) {
            err = "io_bounce_copy_workers=" + std::to_string(val) +
                  " is not implemented - the bounce copy path runs exactly one worker (section 3-5 option B); "
                  "the {1,2} sweep of section 5 needs a later hop - fix: set " + std::string(kKey[4]) +
                  "=1, or unset it to take the engine default 1";
            return false;
        }
        if (val > 255) {
            err = std::string(kName[i]) + "=" + std::to_string(val) +
                  " exceeds 255 - the IO3 header (section 5-2) echoes these as u8" + how_to_fix(i);
            return false;
        }
        if (kFloor[i] > 0 && val < kFloor[i]) {
            if (src == 0) {
                log_err("[moe-direct] mode=virtual raises %s %d -> %d (record fan-out shape floor)",
                        kName[i], (int)val, (int)kFloor[i]);
                val = kFloor[i];
            } else {
                err = std::string(kName[i]) + "=" + std::to_string(val) + " is below the shape floor " +
                      std::to_string(kFloor[i]) +
                      " (the copy queue must hold every in-flight bounce staging slot)" +
                      " - fail-closed: an explicit operator value is a contract, so the engine will not "
                      "quietly run at a different value than the one it echoes" + how_to_fix(i);
                return false;
            }
        }
        dfac_io_param_val_set_locked((size_t)i, val);
        dfac_io_param_src_set_locked((size_t)i, src);
    }
    // ★★R-1 wave 형상 후검사(FROZEN §2-0 A0-2) — `W_l = min(P_l, io_part_parallelism)` 는
    //   `[3]` 이 해소된 **뒤에야** 계산할 수 있으므로 인덱스별 하한이 아니라 여기서 본다.
    //   필요 조건은 "한 **wave** 를 걸 수 있는가"뿐이다(구 판의 "전 child 를 한 번에"가 아니다):
    //     `io_qd_total >= max_l W_l` · `max_l W_l <= MOE_DIRECT_QD_MAX`(채널 하드 캡).
    {
        const int32_t iop = g_mod.v3_io_param_val[3] > 0 ? g_mod.v3_io_param_val[3] : max_parts;
        const int32_t w_max = (max_parts < iop) ? max_parts : iop;
        if (w_max > MOE_DIRECT_QD_MAX) {
            err = "wave width(" + std::to_string(w_max) + " = min(record fan-out " +
                  std::to_string(max_parts) + ", io_part_parallelism " + std::to_string(iop) +
                  ")) exceeds the channel hard cap(" + std::to_string((int)MOE_DIRECT_QD_MAX) +
                  ") - fix: lower MOE_DIRECT_IO_PART_PARALLELISM to <= " +
                  std::to_string((int)MOE_DIRECT_QD_MAX) + " (a record is submitted wave by wave, so "
                  "the fan-out itself no longer has to fit the channel array)";
            return false;
        }
        if (g_mod.v3_io_param_val[0] < w_max) {
            if (g_mod.v3_io_param_src[0] == 0) {
                log_err("[moe-direct] mode=virtual raises io_qd_total %d -> %d (one wave must fit: "
                        "W = min(parts %d, io_part_parallelism %d))",
                        (int)g_mod.v3_io_param_val[0], (int)w_max, (int)max_parts, (int)iop);
                dfac_io_param_val_set_locked(0, w_max);
            } else {
                err = "io_qd_total=" + std::to_string(g_mod.v3_io_param_val[0]) +
                      " is below the wave width " + std::to_string(w_max) + " = min(record fan-out " +
                      std::to_string(max_parts) + ", io_part_parallelism " + std::to_string(iop) +
                      ") - a mode=virtual record is submitted **wave by wave** and one wave is acquired "
                      "atomically, so the channel array must hold at least one wave - fail-closed: an "
                      "explicit operator value is a contract - fix: set MOE_DIRECT_IO_QD_TOTAL >= " +
                      std::to_string(w_max) + ", or lower MOE_DIRECT_IO_PART_PARALLELISM";
                return false;
            }
        }
    }
    // 채널 수는 io_qd_total 그 자체다(별도 축 아님 — 사본 축소).
    dfac_qd_effective_set_locked(g_mod.v3_io_param_val[0]);
    // ★R-1 — 구 `v3_fanout_floor` 표기는 "팬아웃 전량이 QD 하한"이라는 폐기된 규칙의 이름이다.
    //   지금 기본값을 끌어올리는 사유는 **한 wave 가 들어가야 한다**이므로 이름도 그것으로 바꾼다.
    dfac_qd_source_set_locked((g_mod.v3_io_param_src[0] == 2) ? "dedicated_env"
                       : (g_mod.v3_io_param_src[0] == 3) ? "legacy_alias"
                       : (g_mod.v3_io_param_val[0] != MOE_DIRECT_QD_DEFAULT)
                             ? "v3_wave_floor" : "default");
    return true;
}

// ★§Z-⑥ A6 — coalescing 파라미터 해소(§5 `io_coalesce_*`). §3-1 6종과 **같은 규율**:
//   명시값은 운영자 계약이라 무효값·모순값을 조용히 기본값으로 되돌리지 않는다(fail-close).
//   ★기본 OFF: §5 의 `phase prefill` 은 성능 후보값이고 "마이크로벤치 전 동결 금지"가 걸려 있다.
//   엔진의 기존 거동(합치지 않음)이 기본값이고, `prefill` 은 명시 opt-in 이다(R12 의 [0] 판단 계열).
//   ★이 6종은 IO3 헤더 off 104..115 의 **6칸 밖**이다(그 배열은 §5-2 동결) — 관측 창구는
//   `ggml_moe_direct_test_v3_coalesce_params()` 와 seal 로그다.
bool v3_resolve_coalesce_params_locked(std::string & err) {
    dfac_coal_phase_set_locked(V3_COAL_OFF);              // §5 io_coalesce_phase default
    dfac_coal_min_bytes_set_locked(8ull  * 1024 * 1024);    // §5 io_coalesce_min 8MiB
    dfac_coal_max_bytes_set_locked(64ull * 1024 * 1024);    // §5 io_coalesce_max 64MiB
    dfac_coal_buffers_set_locked(2);                      // §5 io_coalesce_buffers
    const std::string ph = getenv_str("MOE_DIRECT_IO_COALESCE_PHASE");
    if (!ph.empty()) {
        if      (ph == "off")     dfac_coal_phase_set_locked(V3_COAL_OFF);
        else if (ph == "prefill") dfac_coal_phase_set_locked(V3_COAL_PREFILL);
        else if (ph == "all")     dfac_coal_phase_set_locked(V3_COAL_ALL);
        else {
            err = "MOE_DIRECT_IO_COALESCE_PHASE=\"" + ph + "\" is not a known value "
                  "(off|prefill|all) - fail-closed (an explicit operator value is a contract; "
                  "falling back would run a different read pattern than the one asked for) - fix: "
                  "set it to off|prefill|all, or unset it to take the engine default off "
                  "(section 3-6 limits coalescing to prefill; the section 5 candidate defaults are "
                  "not frozen before the microbenchmark)";
            return false;
        }
    }
    // ★단위는 **바이트**다(§5 문면은 MiB 로 적혀 있지만 그건 후보값의 표기일 뿐이다).
    //   MiB 단위 키로 두면 대구간 상한을 1MiB 미만으로 표현할 수 없어 형상 실험·관문이 막힌다.
    struct { const char * key; uint8_t sel; uint64_t lo, hi; } kn[3] = {
        { "MOE_DIRECT_IO_COALESCE_MIN_BYTES", 0, 0, 0xFFFFF000ull },
        { "MOE_DIRECT_IO_COALESCE_MAX_BYTES", 1, 1, 0xFFFFF000ull },
        { "MOE_DIRECT_IO_COALESCE_BUFFERS",   2,   1, 64 },
    };
    for (int i = 0; i < 3; i++) {
        const std::string s = getenv_str(kn[i].key);
        if (s.empty()) continue;
        char * endp = nullptr;
        errno = 0;
        const unsigned long long v = strtoull(s.c_str(), &endp, 10);
        if (!endp || *endp != '\0' || s[0] == '-' || errno != 0 ||
            (uint64_t)v < kn[i].lo || (uint64_t)v > kn[i].hi) {
            err = std::string(kn[i].key) + "=\"" + s + "\" is invalid (integers in [" +
                  std::to_string(kn[i].lo) + ", " + std::to_string(kn[i].hi) + "] only) - fail-closed "
                  "(an explicit operator value is a contract: the engine will not quietly run a "
                  "different read pattern) - fix: set " + kn[i].key + " to a value in that range, or "
                  "unset it to take the engine default";
            return false;
        }
        dfac_coal_env_set_locked(kn[i].sel, v);
    }
    if (g_mod.v3_coal_max_bytes < g_mod.v3_coal_min_bytes) {
        err = "io_coalesce_max(" + std::to_string(g_mod.v3_coal_max_bytes) +
              "B) is below io_coalesce_min(" + std::to_string(g_mod.v3_coal_min_bytes) +
              "B) - no run could ever satisfy both, so coalescing would be accepted and then never "
              "happen (a false success) - fix: raise MOE_DIRECT_IO_COALESCE_MAX_BYTES or lower "
              "MOE_DIRECT_IO_COALESCE_MIN_BYTES";
        return false;
    }
    return true;
}

// ===========================================================================
// ★§Z-⑥ 원자 A1+A2 — mode=virtual seal(검증 구간 + 자원 확보). 호출자는 g_mod.mutex 를 보유한다.
//
// ★**이 함수는 항상 실패를 반환한다**(현 경계 = A3 앞). A1 은 "v3 manifest 를 읽고 부팅 대조를
//   전항 세운다"까지, A2 는 "소스 핸들 집합·슬롯 풀·registry·보호를 확보한다"까지다. child
//   상태기계(A3)·바운스(A4)·MOEIO3(A5)는 아직 없으므로, 전 관문을 통과한 정상 manifest 도
//   마지막에 명시적 사유로 기동을 거부한다 — 반쯤 세워진 소비자로 seal 이 성립하면 acquire 가
//   빈 슬롯을 읽는다. ★A2 가 잡은 자원은 그 거부 직전에 전량 반납된다(virtual_release_locked).
//   통과 깊이는 g_v3_stage 가, 자원 사실은 g_v3_probe 가 남긴다(둘 다 테스트 전용 관측).
// ===========================================================================
// ★이름에서 원자 번호를 뺐다(구 `seal_virtual_locked_a1`) — 경계는 원자마다 앞으로 밀리는데
//   이름이 원자를 담으면 그 이름 자체가 매번 낡는 사본이 된다. 현 경계는 함수 말미가 권위다.
// ★★PB virtual A5 — seal 실패 rollback 의 프리페치 몫(§2-2 "실패 후 자원 잔존 0").
//   `ggml_moe_phaseb_note_seal` 이 capture 풀·intent_q·predictor worker 를 기동하므로, 그
//   뒤의 어떤 실패도 **worker 를 stop/join 한 뒤에** 반환해야 한다. CUT 의 ①②(신규 capture
//   거절 → capture drain + worker stop/join)를 그대로 쓴다 — 종료 순서를 두 벌로 만들지 않는다.
//   ★락 안전: predictor worker 는 `g_mod.mutex` 를 잡지 않는다(phaseb 소유 상태만 만진다).
//   게다가 이 시점엔 dispatcher 가 아직 안 돌아 capture 가 1건도 없다.
void pbv_seal_unwind_prefetch_locked() {
    // ★★R-2(교차 r1 F1) — 구 판은 predictor 만 접고 direct 쪽 `pb_*` 사본만 비웠다. PhaseB
    //   내부의 `sealed`·스냅샷 형상·`cap_slots`·`int_slots`·CUT 래치는 **그대로 남아** §2-2 의
    //   "실패 후 자원 잔존 0"을 위반했다(전건 초기화 수단이 test-only reset 뿐이었다).
    //   ⇒ PhaseB 소유분의 되돌림은 **PhaseB 가 소유한 unseal API** 가 한다(사본 축소 —
    //   direct 가 남의 내부 상태를 손으로 되돌리지 않는다). 그 안에서 capture 닫기·worker
    //   stop/join 까지 수행하므로 여기서 CUT 2단계를 따로 부르지 않는다.
    ggml_moe_phaseb_unseal_failed();
    ggml_moe_phaseb_set_wake_event(nullptr);
    dfac_prefetch_unwind_locked();
}

int seal_virtual_validate_locked(const std::wstring & dir_w, const json & j, const ggml_moe_direct_model & model,
                                 const ggml_moe_direct_source * srcs, int32_t n_srcs,
                                 bool allow_sector_fallback, const std::wstring & expects_dir_w,
                                 const char * override_profile_id, const char * override_digest,
                                 const char * override_scope, std::string & err,
                                 ParsedManifest & out_pm, std::string & out_manifest_sha,
                                 std::unordered_map<std::string, std::pair<int32_t, int32_t>> & out_name_to_lp,
                                 std::vector<char> & out_pending_kept) {
    g_v3_stage.store(V3_STAGE_MODE_DETECTED, std::memory_order_relaxed);

    // ---- ① IO_METRICS §3 거부 4조합 — 검증·자원 확보보다 **앞**(init 거부의 의미) ----
    if (!virtual_init_reject_check(err)) return -1;
    // ★구 `v3_atom_env_boundary` 호출 자리 — 원자 A8 에서 철거됐다(위 함수 자리 주석 참조).
    //   사양 §3 의 거부 4조합은 바로 위 `virtual_init_reject_check` 가 계속 소유한다.
    g_v3_stage.store(V3_STAGE_INIT_COMBOS, std::memory_order_relaxed);

    // ---- ② manifest v3 검증(§2-2 좌표계 · §2-4 산술 · records/source_tensors 전항 재생성) ----
    ParsedManifest pm;
    if (!validate_manifest_v3(j, model, srcs, n_srcs, pm, err)) return -1;
    // ★A3 이 세웠던 비4K 거부 게이트는 **A4 에서 철거**됐다 — 바운스 경로(§3-5)가 섰다.
    g_v3_stage.store(V3_STAGE_MANIFEST, std::memory_order_relaxed);

    // ---- ③ reference_lock 소비자 결속(v2 §4-1 F1 승계) + virtual 산출물 형상 ----
    {
        if (expects_dir_w.empty()) { err = "MOE_DIRECT_EXPECTS_DIR is not set (reference_lock cannot be bound; fail-closed)"; return -1; }
        std::string approved, approved_scope;
        if (!catalog_lookup(pm.ref_profile_id, override_profile_id, override_digest, override_scope, approved, approved_scope)) {
            err = "reference_lock.profile_id is not registered (not in the catalog): " + pm.ref_profile_id; return -1;
        }
        if (to_lower_hex(pm.ref_expect_sha256) != to_lower_hex(approved)) {
            err = "manifest.reference_lock.expect_sha256 != approved catalog digest"; return -1;
        }
        if (approved_scope != pm.routed_scope) {
            err = "catalog scope(" + approved_scope + ") != manifest routed_scope(" + pm.routed_scope + ")"; return -1;
        }
        std::wstring expect_path = expects_dir_w + L"\\" + utf8_to_wide(pm.ref_profile_id) + L".expect.json";
        std::string file_sha, sherr;
        if (!sha256_file(expect_path, file_sha, sherr)) { err = "cannot re-hash the expect file: " + sherr; return -1; }
        if (to_lower_hex(file_sha) != to_lower_hex(approved)) {
            err = "re-hashed expect file != approved digest (tampered or misplaced)"; return -1;
        }
        // ★virtual 은 experts.bin 을 **만들지 않는다**(§0). v3 manifest 옆의 bin 은 생산 측이
        //   명시적으로 거부하는 혼합 산출물이고, 조용한 오독이 나오는 바로 그 형상이다.
        //   ★INVALID_FILE_ATTRIBUTES 는 부재만이 아니라 모든 조회 실패에서 반환된다 —
        //   ERROR_FILE_NOT_FOUND 만 부재로 인정한다(v2 partial 마커와 같은 규율).
        for (const wchar_t * leaf : { L"\\experts.bin", L"\\experts.bin.partial" }) {
            const std::wstring pth = dir_w + leaf;
            if (GetFileAttributesW(pth.c_str()) != INVALID_FILE_ATTRIBUTES) {
                err = "a mode=virtual artifact directory must not contain experts.bin(.partial) - this is the mixed "
                      "artifact shape that silent misreads come from (fail-closed)";
                return -1;
            }
            const DWORD gle = GetLastError();
            if (gle != ERROR_FILE_NOT_FOUND) {
                err = "cannot query the legacy bin artifact (GetLastError=" + std::to_string((unsigned long)gle) +
                      ") - cannot confirm its absence (fail-closed)";
                return -1;
            }
        }
    }
    g_v3_stage.store(V3_STAGE_REFLOCK, std::memory_order_relaxed);

    // ---- ④ §4-3 소비 결속 — plan_report(v2 verify_report 결속의 v3 대응물) ----
    // ★게이트가 계산한 manifest.json 실 바이트 SHA 는 seal 밖으로 보존한다(v2 R4 와 같은 축 —
    //   같은 profile/expect 로 정상 재팩한 **다른 산출물**을 구분하는 identity).
    std::string manifest_sha;
    {
        std::string rp_pid, rp_sha, rp_manifest_sha, rperr;
        int64_t rp_records = 0;
        const std::wstring rp_path = dir_w + L"\\plan_report.json";
        if (!read_plan_report_gate(rp_path, rp_pid, rp_sha, rp_manifest_sha, rp_records, rperr)) {
            err = "plan_report gate failed: " + rperr; return -1;
        }
        if (rp_pid != pm.ref_profile_id || to_lower_hex(rp_sha) != to_lower_hex(pm.ref_expect_sha256)) {
            err = "plan_report.reference_lock != manifest.reference_lock"; return -1;
        }
        // cache key 결속: 리포트가 검증했다고 주장하는 manifest 바이트 == 지금 로드한 manifest.json.
        std::string mf_sha, mferr;
        if (!sha256_file(dir_w + L"\\manifest.json", mf_sha, mferr)) { err = "cannot re-hash manifest.json: " + mferr; return -1; }
        if (to_lower_hex(mf_sha) != to_lower_hex(rp_manifest_sha)) {
            err = "plan_report.manifest_sha256 != actual manifest.json hash - this report did not verify this "
                  "manifest (report reused from another artifact, or manifest changed after the plan; fail-closed)";
            return -1;
        }
        if (rp_records != pm.n_records) {
            err = "plan_report.cardinality.records(" + std::to_string(rp_records) + ") != manifest totals.n_records(" +
                  std::to_string(pm.n_records) + ")"; return -1;
        }
        manifest_sha = to_lower_hex(mf_sha);
    }
    g_v3_stage.store(V3_STAGE_PLAN_REPORT, std::memory_order_relaxed);

    // ---- ⑤ A_runtime 사다리(§3-7 v2 승계). virtual 엔 출력 볼륨이 없으므로 질의 대상은
    //      **각 source shard 가 놓인 볼륨**이다(§2-4 A 출처와 같은 대상).
    //      ★align_query[] provenance 의 재질의 대조는 여기 없다 [[C:repack.align-query]] —
    //      사양이 "소비 기동 시점 방어는 아직 미시공"으로 남긴 §Z 후속 계약이다. ----
    for (const auto & si : pm.sources) {
        uint64_t a_runtime = 0;
        std::string qerr;
        if (!query_a_runtime(utf8_to_wide(si.path), allow_sector_fallback, a_runtime, qerr)) {
            err = "sector re-query failed for source shard " + si.path + ": " + qerr; return -1;
        }
        if (a_runtime == 0 || pm.align_bytes % a_runtime != 0) {
            err = "align_bytes(" + std::to_string(pm.align_bytes) + ") % A_runtime(" + std::to_string(a_runtime) +
                  ") != 0 for source shard " + si.path; return -1;
        }
        if (pm.slot_stride_max % a_runtime != 0) {
            err = "slot_stride_max % A_runtime != 0 (A_runtime=" + std::to_string(a_runtime) + ")"; return -1;
        }
    }
    g_v3_stage.store(V3_STAGE_ALIGN, std::memory_order_relaxed);

    // ---- ⑥ 등록 3집합 대조(§2-5 ② · v2 §4-2① 승계 + 부록A scope) ----
    std::unordered_map<std::string, std::pair<int32_t, int32_t>> name_to_lp;
    std::vector<char> pending_kept(g_mod.pending.size(), 1);
    {
        for (size_t mlpos = 0; mlpos < pm.layers.size(); mlpos++) {
            for (size_t pidx = 0; pidx < pm.layers[mlpos].parts.size(); pidx++) {
                const std::string & nm = pm.layers[mlpos].parts[pidx].source_tensor;
                if (name_to_lp.count(nm)) { err = "M_all duplicate source_tensor: " + nm; return -1; }
                name_to_lp[nm] = { (int32_t)mlpos, (int32_t)pidx };
            }
        }
        const bool is_execution = (pm.routed_scope == "execution");
        const int32_t exec_threshold = model.n_layer - model.n_layer_nextn;
        if (is_execution) {
            for (size_t i = 0; i < g_mod.pending.size(); i++) {
                int32_t player = -1;
                if (!parse_blk_layer(g_mod.pending[i].name, player)) {
                    err = "execution scope: cannot parse the layer from a pending tensor name: " + g_mod.pending[i].name; return -1;
                }
                pending_kept[i] = (player < exec_threshold) ? 1 : 0;
            }
        }
        std::unordered_map<std::string, int> l_seen;
        int32_t l_kept = 0;
        for (size_t i = 0; i < g_mod.pending.size(); i++) {
            if (!pending_kept[i]) continue;
            const std::string & nm = g_mod.pending[i].name;
            if (l_seen.count(nm)) { err = "L_scope duplicate registration: " + nm; return -1; }
            l_seen[nm] = 1;
            l_kept++;
            if (!name_to_lp.count(nm)) { err = "L_scope overflow (tensor absent from the manifest): " + nm; return -1; }
        }
        if (l_kept != (int32_t)name_to_lp.size()) {
            err = "L_scope(" + std::to_string(l_kept) + ") != M_all(" + std::to_string(name_to_lp.size()) + ")"; return -1;
        }
        for (const auto & kv : name_to_lp) {
            if (!l_seen.count(kv.first)) { err = "M_all missing (tensor absent from the scope set): " + kv.first; return -1; }
        }
    }
    g_v3_stage.store(V3_STAGE_REGISTRY, std::memory_order_relaxed);

    // ---- ⑦ §2-5 ③ live GGUF 재파싱 + DF-1 digest + 등록 텐서 대조 ----
    {
        std::unordered_map<std::string, int32_t> live_type;
        if (!virtual_probe_sources(pm, live_type, err)) return -1;
        for (size_t i = 0; i < g_mod.pending.size(); i++) {
            if (!pending_kept[i]) continue;
            const PendingReg & pr = g_mod.pending[i];
            const ggml_tensor * t = pr.tensor;
            if (!t) continue;
            const auto lp = name_to_lp[pr.name];
            const PartInfo & part = pm.layers[(size_t)lp.first].parts[(size_t)lp.second];
            const auto lt = live_type.find(pr.name);
            if (lt != live_type.end() && (int32_t)t->type != lt->second) {
                err = "registered tensor type != the type recorded in the source GGUF header: " + pr.name; return -1;
            }
            if ((int32_t)t->ne[part.expert_axis] != pm.n_expert) {
                err = "tensor " + pr.name + " ne[expert_axis]=" + std::to_string(t->ne[part.expert_axis]) +
                      " != n_expert=" + std::to_string(pm.n_expert); return -1;
            }
            uint64_t want_bytes = 0;
            if (!ck_mul(part.slice_bytes, (uint64_t)pm.n_expert, want_bytes)) {
                err = "checked uint64 overflow while checking the registered tensor size of " + pr.name; return -1;
            }
            if ((uint64_t)ggml_nbytes(t) != want_bytes) {
                err = "tensor " + pr.name + " ggml_nbytes(" + std::to_string(ggml_nbytes(t)) +
                      ") != slice_bytes*n_expert(" + std::to_string(want_bytes) + ")"; return -1;
            }
        }
    }
    g_v3_stage.store(V3_STAGE_LIVE, std::memory_order_relaxed);   // ★A1 완주
    // R-B4S stage epilogue: what this stage produces for the stages after it
    out_pm = pm;
    out_manifest_sha = manifest_sha;
    out_name_to_lp = name_to_lp;
    out_pending_kept = pending_kept;
    return 0;
}

int seal_virtual_test_fail_hook_locked(std::string & err) {

    // ★테스트 전용 one-shot — **자원 확보 이후의 늦은 seal 실패**를 결정론적으로 만든다.
    //   A2 가 세운 "실패한 seal 은 잡은 것을 전량 반납한다(핸들·풀·보호 원복)"를 A3 가 경계를
    //   앞으로 밀고도 계속 지키는지 잠그는 창구다 — 그 경로는 실환경에서 CreateEvent/VEH 등록
    //   실패로만 도달해 재현할 수 없다. 프로덕션 0(reset 에서 0 복귀).
    if (g_mod.v3_test_fail_after_acquire) {
        dfac_test_fail_after_acquire_set_locked(0);
        virtual_release_locked();
        err = "test-only injected seal failure after resource acquisition";
        return -1;
    }
    return 0;
}

int seal_virtual_resolve_params_locked(std::string & err, int32_t & out_v3_max_bounce_per_record, uint64_t & out_v3_max_staging_bytes,
                                       int32_t & out_v3_staging_plan,
                                       int32_t & out_v3_coal_max_parts) {

    // ⓐ ★R12 — 형상 상수 산출 → §3-1 IO 파라미터 6종 해소(전용 env·precedence·하한).
    //    ★A3 불변식: 한 record 의 child 를 **한 번에** 걸 수 있어야 한다(부분 제출 금지 = 채널
    //    교착 부재의 근거). 그래서 io_qd_total·io_part_parallelism 에 형상 하한이 생긴다.
    //    ★staging 형상(아래 ⓓ′ 가 실제로 잡는 값)을 **여기서 먼저 계산**한다 — copy 큐 깊이
    //    (`io_bounce_copy_qd`)의 하한이 그 값이라 해소가 할당보다 앞서야 한다.
    int32_t v3_max_bounce_per_record = 0;
    uint64_t v3_max_staging_bytes = 0;
    int32_t v3_staging_plan = 0;
    int32_t v3_coal_max_parts = 0;
    {
        int32_t max_parts = 0;
        for (const auto & L : g_mod.layers) {
            if ((int32_t)L.parts.size() > max_parts) max_parts = (int32_t)L.parts.size();
            int32_t nb = 0;
            for (const auto & p : L.parts) {
                if (p.aligned) continue;
                nb++;
                if (p.staging_bytes > v3_max_staging_bytes) v3_max_staging_bytes = p.staging_bytes;
            }
            if (nb > v3_max_bounce_per_record) v3_max_bounce_per_record = nb;
        }
        // 풀 크기 = **동시 in-flight record 수 × record 당 바운스 part 수** 상한(구 ⓓ′ 산식 그대로).
        if (v3_max_bounce_per_record > 0) v3_staging_plan = v3_max_bounce_per_record * 2 + 2;
        if (!v3_resolve_io_params_locked(max_parts, v3_staging_plan, err)) {
            virtual_release_locked();
            return -1;
        }
        // ★A6 — coalescing 파라미터(§5). 형상 하한이 없는 대신 **명시값 무효는 fail-close**.
        if (!v3_resolve_coalesce_params_locked(err)) {
            virtual_release_locked();
            return -1;
        }
        v3_coal_max_parts = max_parts;
    }
    // R-B4S stage epilogue: what this stage produces for the stages after it
    out_v3_max_bounce_per_record = v3_max_bounce_per_record;
    out_v3_max_staging_bytes = v3_max_staging_bytes;
    out_v3_staging_plan = v3_staging_plan;
    out_v3_coal_max_parts = v3_coal_max_parts;
    return 0;
}

int seal_virtual_resolve_io3_locked() {

    // ⓑ io_path — legacy 는 §3 거부 조합 ③ 이 이미 막았다. 영속 dispatcher 고정.
    // IO3 sidecar resolve (SPEC_IO_METRICS_V3 section 5-1). Default off: an unset run
    // allocates nothing, so bin/virtual overhead stays symmetric.
    {
        const std::string v = getenv_str("MOE_DIRECT_IO3");
        bool on = false;
        if (v == "on")       on = true;
        else if (!v.empty() && v != "off")
            log_err("[moe-direct][io3] MOE_DIRECT_IO3=\"%s\" is not a known value (on|off) - IO3 stays off",
                    v.c_str());
        g_mod.io3_on       = on;
        g_mod.io3_requested= on;
        g_mod.io3_complete = true;
        g_mod.io3_out      = getenv_str("MOE_DIRECT_IO3_OUT");
        uint64_t buf_mb = 512;
        const std::string bs = getenv_str("MOE_DIRECT_IO3_BUF_MB");
        if (!bs.empty()) {
            char * endp = nullptr;
            const long bv = strtol(bs.c_str(), &endp, 10);
            if (endp && *endp == '\0' && bv > 0) buf_mb = (uint64_t)bv;
            else log_err("[moe-direct][io3] MOE_DIRECT_IO3_BUF_MB=\"%s\" is invalid (positive integers"
                         " only) - using the default 512", bs.c_str());
        }
        g_mod.io3_buf_cap_bytes = buf_mb * 1024ull * 1024ull;
        if (on) {
            LARGE_INTEGER c; QueryPerformanceCounter(&c);
            g_mod.io3_t0_qpc = (uint64_t)c.QuadPart;
            uint64_t h = 0xcbf29ce484222325ull;
            const uint32_t pid = (uint32_t)GetCurrentProcessId();
            h = fnv1a64_bytes(h, &pid, sizeof(pid));
            h = fnv1a64_bytes(h, &g_mod.io3_t0_qpc, sizeof(g_mod.io3_t0_qpc));
            g_mod.io3_boot_id = h;
            // OUT unset => nothing is written and the arm is invalid (section 5-1).
            if (g_mod.io3_out.empty()) g_mod.io3_complete = false;
        }
    }
    return 0;
}

int seal_virtual_note_external_locked() {

    // ★lead-time arm(PROBE_LEADTIME_SPEC_R1 §2 "seal 시 전량 선할당") — 위 IO3 boot_id 해소
    //   **뒤**여야 조인 키를 그대로 채택할 수 있다(§4b "boot_id/QPC 도메인 일치"). 게이트 OFF 면
    //   내부에서 즉시 반환하며 어떤 버퍼도 할당하지 않는다.
    //   ★이 호출이 env 해소·선할당의 **유일한 프로덕션 지점**이다 — 그래서 hot hook 은 getenv 도
    //   VirtualAlloc 도 유발할 수 없다(모듈 헤더 ★ 절). 위치가 계약의 일부다.
    //   n_expert/moe_layers 는 virtual_acquire_locked(:7440-7442)가 이미 채웠으므로 유효하다.
    ggml_moe_leadtime_note_seal(g_mod.io3_boot_id, /*engine_mode=*/2u,
                                (uint32_t)g_mod.n_expert, (uint32_t)g_mod.moe_layers.size());

    // ★GPU-early(C 축) arm(PROBE_GPU_EARLY_SPEC_R1 §4) — lead-time 과 **같은 이유로** IO3 boot_id
    //   해소 뒤다: C wire 의 boot_id 는 "기존 A/B wire 동일 계약"이라 세 축이 한 값을 공유해야
    //   판독기 조인이 성립한다. 게이트 OFF 면 내부에서 즉시 반환하며 어떤 버퍼도 할당하지 않는다.
    //   ★이 호출이 C 축 env 해소·선할당의 **유일한 프로덕션 지점**이다(hot hook 은 getenv 도
    //   VirtualAlloc 도 유발할 수 없다 — 모듈 헤더 hot-path 계약 ②).
    ggml_moe_gpuearly_note_seal(g_mod.io3_boot_id);
    return 0;
}

int seal_virtual_fix_io_path_locked() {

    dfac_io_path_set_locked(true, "default", getenv_flag_is_1("MOE_DIRECT_QUALIFY"));
    return 0;
}

int seal_virtual_install_veh_locked(std::string & err) {

    // ⓒ 보호 tripwire 의 짝 — VEH. dispatcher 기동보다 앞이어야 등록 텐서 접근이 잡힌다.
    g_mod.veh_handle = AddVectoredExceptionHandler(1, moe_direct_veh_handler);
    if (!g_mod.veh_handle) {
        virtual_release_locked();
        err = "AddVectoredExceptionHandler failed";
        return -1;
    }
    return 0;
}

int seal_virtual_alloc_coalesce_pool_locked(std::string & err, int32_t v3_coal_max_parts) {

    // ⓓ′ ★A4 — 바운스 staging 풀 + copy worker. 비4K part 가 하나도 없으면 아무것도 만들지
    //     않는다(전 aligned 프로파일의 형상·비용이 A3 와 동일하게 유지된다).
    // ⓓ″ ★A6 — coalesce 버퍼 풀. **켰을 때만** 만든다 ⇒ 기본 off 런은 **물리 I/O 형상이
    //     불변이고 coalesce staging pool 이 0** 이다. ★교차 r8(Z2-1) 정정 — 구 문면은 이걸
    //     **앞 원자와 동등한 비용**으로 적었으나 과장이었다: OFF 에서도 `v3_chan_coal` 은
    //     QD 크기로 할당되고 매 op 마다 planner 를 불러 disabled 마스크를 갱신한다.
    //     칸 수 = `io_coalesce_buffers × max_parts` — 대구간 그룹 1개가 part 수만큼의 칸을
    //     동시에 쓰기 때문이다(참가 집합이 part 마다 같은 all-or-nothing 설계).
    //     칸 크기 = 전 (층, part) 중 가장 큰 대구간 = `min(io_coalesce_max, align_up(A + n_expert×slice, A))`.
    if (g_mod.v3_coal_phase != V3_COAL_OFF && v3_coal_max_parts > 0) {
        const uint64_t A = g_mod.align_bytes;
        uint64_t want = 0;
        for (const auto & L : g_mod.layers) {
            for (const auto & p : L.parts) {
                uint64_t span = 0, full = 0;
                if (!ck_mul(p.slice_bytes, (uint64_t)g_mod.n_expert, span) ||
                    !ck_add(span, A, full)) { full = g_mod.v3_coal_max_bytes; }
                full = align_up(full, A);
                const uint64_t cap = align_up(g_mod.v3_coal_max_bytes, A);
                const uint64_t use = (full < cap) ? full : cap;
                if (use > want) want = use;
            }
        }
        if (want < A) want = A;
        const int32_t n_buf = g_mod.v3_coal_buffers * v3_coal_max_parts;
        void * base = VirtualAlloc(nullptr, (size_t)want * (size_t)n_buf,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!base) {
            RemoveVectoredExceptionHandler(g_mod.veh_handle); g_mod.veh_handle = nullptr;
            virtual_release_locked();
            err = "VirtualAlloc(coalesce buffer pool, " + std::to_string((uint64_t)want * (uint64_t)n_buf) +
                  " bytes) failed err=" + std::to_string((unsigned long)GetLastError()) +
                  " - fix: lower MOE_DIRECT_IO_COALESCE_MAX_BYTES or MOE_DIRECT_IO_COALESCE_BUFFERS, "
                  "or unset MOE_DIRECT_IO_COALESCE_PHASE to run without coalescing";
            return -1;
        }
        dfac_coal_base_set_locked(base);
        dfac_coal_stride_set_locked((size_t)want);
        dfac_coal_count_set_locked(n_buf);
        dfac_coal_assign_locked((size_t)n_buf);
        for (auto & cr : g_mod.v3_coal) cr.ops.reserve((size_t)g_mod.n_expert);
        dfac_coal_plan_ops_reserve_locked((size_t)g_mod.n_expert);
        dfac_coal_plan_cand_assign_locked((size_t)g_mod.n_expert, -1);
        log_err("[moe-direct] mode=virtual coalescing on (phase=%s min=%lluB max=%lluB buffers=%d "
                "-> %d slots x %llu bytes = %llu bytes; max run = io_bounce_copy_qd(%d) / parts(%d) = %d)",
                g_mod.v3_coal_phase == V3_COAL_ALL ? "all" : "prefill",
                (unsigned long long)g_mod.v3_coal_min_bytes,
                (unsigned long long)g_mod.v3_coal_max_bytes,
                (int)g_mod.v3_coal_buffers, (int)n_buf, (unsigned long long)want,
                (unsigned long long)((uint64_t)want * (uint64_t)n_buf),
                (int)g_mod.v3_io_param_val[5], (int)v3_coal_max_parts,
                (int)(g_mod.v3_io_param_val[5] / (v3_coal_max_parts > 0 ? v3_coal_max_parts : 1)));
    }
    return 0;
}

int seal_virtual_alloc_staging_pool_locked(std::string & err, int32_t v3_max_bounce_per_record, uint64_t v3_max_staging_bytes,
                                           int32_t v3_staging_plan) {
        if (v3_max_bounce_per_record > 0) {
            // 풀 크기 = **동시 in-flight record 수 × record 당 바운스 part 수** 상한(위 ⓐ 산출).
            //   채널이 QD 개이므로 동시 record 는 최대 QD/parts 개다. 여유 없이 딱 맞추면
            //   copy 가 끝나기 전에 다음 record 가 못 들어와 backpressure 가 상시화된다.
            const int32_t n_stg = v3_staging_plan;
            const size_t stride = (size_t)align_up(v3_max_staging_bytes, g_mod.align_bytes);
            void * base = VirtualAlloc(nullptr, stride * (size_t)n_stg, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!base) {
                RemoveVectoredExceptionHandler(g_mod.veh_handle); g_mod.veh_handle = nullptr;
                virtual_release_locked();
                err = "VirtualAlloc(bounce staging pool) failed err=" + std::to_string((unsigned long)GetLastError());
                return -1;
            }
            dfac_staging_base_set_locked(base);
            dfac_staging_stride_set_locked(stride);
            dfac_staging_count_set_locked(n_stg);
            dfac_staging_busy_assign_locked((size_t)n_stg, 0);
        }
    return 0;
}

int seal_virtual_start_copy_worker_locked(std::string & err, int32_t v3_max_bounce_per_record) {
        // ★A6 — copy worker 는 바운스뿐 아니라 **coalesce scatter** 의 소비자이기도 하다.
        //   전 aligned 프로파일이라도 합침을 켰으면 worker 가 있어야 한다(없으면 scatter 를
        //   태울 곳이 없다). 둘 다 아니면 종전대로 아무것도 만들지 않는다.
        if (v3_max_bounce_per_record > 0 || g_mod.v3_coal_phase != V3_COAL_OFF) {
            std::string cerr;
            // ★R12 — copy 큐 용량 = 해소된 `io_bounce_copy_qd`(§3-1). 하한이 staging 칸 수라
            //   만석 fatal 은 구조적으로 도달 불가이고, §5-6 copy queue cap 검산이 이 값과
            //   관측 겹침을 대조한다(구 판의 `n_stg + 8` 은 어떤 노브에도 결속되지 않은 상수였다).
            if (!v3_copy_start_locked(/*capacity=*/g_mod.v3_io_param_val[5], cerr)) {
                RemoveVectoredExceptionHandler(g_mod.veh_handle); g_mod.veh_handle = nullptr;
                virtual_release_locked();
                err = cerr;
                return -1;
            }
        }
    return 0;
}

int seal_virtual_note_prefetch_locked(const ggml_moe_direct_model & model, std::string & err) {

    // ⓓ‴ ★★PB virtual A5 — B2 프리페치 K/N 해소 + note_seal 배선
    //     (`SPEC_PB_VIRTUAL_RUNTIME_R1` §2-2 · v2 seal 의 같은 블록과 **같은 함수·같은 순서**).
    //     ★위치 구속 3가지:
    //       ①**metrics header 기록보다 앞** — header 가 `prefetch_on/k/n/off_reason` 을 echo 한다.
    //       ②**copy worker 확보 뒤** — 실패 rollback 이 아래 dispatcher 실패 경로와 같은 순서
    //         (predictor 접기 → copy worker 접기 → VEH 해제 → 자원 반납)로 닫힌다.
    //       ③**dispatcher 기동 앞** — dispatcher 가 g_mod.mutex 를 잡기 시작하면 락을 쥔 채로는
    //         어떤 worker 도 join 할 수 없다.
    //     ★rollback 계약(§2-2 "실패 후 자원 잔존 0"): note_seal 이 capture 풀·intent_q·
    //       predictor worker 를 기동하므로, 이 뒤의 어떤 실패도 그것을 되돌린 뒤에 반환한다.
        ggml_moe_phaseb_resolve_prefetch(g_mod.qd_effective);
        std::vector<uint64_t> pb_strides;
        pb_strides.reserve(g_mod.layers.size());
        for (const auto & L : g_mod.layers) pb_strides.push_back(L.stride_bytes);
        const int pb_rc = ggml_moe_phaseb_note_seal(
            g_mod.n_expert, g_mod.n_expert_used, (int32_t)g_mod.moe_layers.size(), g_mod.n_slots,
            g_mod.qd_effective, g_mod.moe_layers.data(), pb_strides.data(),
            model.n_layer - model.n_layer_nextn, model.n_embd,
            model.f_norm_rms_eps > 0.0f ? model.f_norm_rms_eps : 1e-6f);
        if (pb_rc != 0) {
            // ★worker 생성 실패 = **명시 실패 전파**(§2-2). 조용한 no-op(프리페치 ON 인데
            //   intent 생산자 없음)으로 빠지지 않는다.
            pbv_seal_unwind_prefetch_locked();
            dfac_dctl_copy_stop_and_join();
            RemoveVectoredExceptionHandler(g_mod.veh_handle);
            g_mod.veh_handle = nullptr;
            virtual_release_locked();
            err = "mode=virtual prefetch seal failed: the PhaseB predictor worker could not be "
                  "started (fail-closed - a prefetch-ON run with no intent producer would be a "
                  "silent no-op) - fix: unset MOE_DIRECT_PREFETCH_K/MOE_DIRECT_PREFETCH_N to run "
                  "without prefetch";
            return -1;
        }
        dfac_prefetch_note_flags_locked(g_mod.moe_layers.size());
        dfac_prefetch_note_census_locked(g_mod.n_slots);
    return 0;
}

int seal_virtual_apply_ncap_locked() {
        // ★§1 유지 거부 4종은 이미 `virtual_init_reject_check` 가 앞에서 막았다(PHASEA/PHASEB
        //   != off · legacy · trace). 그래서 v2 블록의 equiv/observe/legacy precheck 3종은
        //   virtual 에서 도달할 수 없다 — 여기 복제하지 않는다(같은 계약을 두 벌로 두지 않음).
        //   `io_path` 는 위 ⓑ 에서 persistent 로 고정돼 있다.
        if (g_mod.pb_prefetch_on) {
            // ★A0-5 — 전 층 `N_cap(l)=0` 이면 경고 1줄(기동은 유효). 층별 상한이 전부 0 이라는
            //   것은 형상·노브 조합이 프리페치를 실효 OFF 로 만들었다는 뜻이므로, 운영자가
            //   "켰는데 왜 안 도나"를 로그 없이 겪지 않게 한다.
            int32_t ncap_max = 0;
            for (size_t ml = 0; ml < g_mod.layers.size(); ml++) {
                const int32_t c = pbv_ncap_locked((int32_t)ml, g_mod.pb_n);
                if (c > ncap_max) ncap_max = c;
            }
            if (ncap_max <= 0) {
                log_err("[moe-direct] mode=virtual prefetch is on but N_cap(l)=0 for every layer "
                        "(N=%d io_qd_total=%d io_qd_demand_reserve=%d io_qd_prefetch_child_max=%d) "
                        "- no prefetch record can be admitted; raise the QD knobs or lower the "
                        "record fan-out", (int)g_mod.pb_n, (int)g_mod.v3_io_param_val[0],
                        (int)g_mod.v3_io_param_val[1], (int)g_mod.v3_io_param_val[2]);
            }
        }
    return 0;
}

int seal_virtual_open_metrics_locked(std::string & err) {

    // ⓓ″ ★R13 — 계측 JSONL(v2 와 같은 함수·같은 배타 생성 규약). 구 판은 이 호출이 **v2 seal
    //    에만** 있어서 `MOE_DIRECT_METRICS` 가 virtual 에서 조용히 무시됐다(교차 r5 Y4 — 파일이
    //    아예 안 생겼는데 관문은 seal 성공만 봤다).
    //    ★위치 판단: header 가 echo 하는 값(qd·io_params·io3 identity·layer 배열)은 여기서 전부
    //    확정돼 있고, **dispatcher 기동 앞**이라 실패 시 되돌림이 바로 아래 dispatcher 실패
    //    경로와 같은 3수(copy worker 접기 → VEH 해제 → 자원 반납)로 닫힌다. dispatcher 가 돈
    //    뒤였다면 그 스레드가 g_mod.mutex 를 잡으므로 락을 쥔 채로는 접을 수 없다.
    {
        std::string metrics_err;
        if (!metrics_open_if_requested_locked(metrics_err)) {
            pbv_seal_unwind_prefetch_locked();   // ★★A5 — predictor 를 먼저 접는다(§2-2 rollback)
            dfac_dctl_copy_stop_and_join();      // 이 시점 copy job 0(요청이 아직 없다 — 아래 주석과 같은 근거)
            RemoveVectoredExceptionHandler(g_mod.veh_handle);
            g_mod.veh_handle = nullptr;
            virtual_release_locked();
            err = metrics_err;
            return -1;
        }
    }
    return 0;
}

int seal_virtual_start_dispatcher_locked(std::string & err) {

    // ⓓ dispatcher 기동(채널·pending 풀·child 병렬표 포함 — dispatcher_start_locked 안에서 할당).
    {
        std::string disp_err;
        if (!dispatcher_start_locked(disp_err)) {
            // ★A4 — copy worker 를 먼저 접는다. 이 지점에서 **copy job 은 하나도 존재할 수 없다**
            //   (job 은 dispatcher 의 child 회수에서만 생기는데 dispatcher 가 아직 안 돌았다).
            //   그래서 worker 는 대기 루프에 있고 g_mod.mutex 를 잡지 않으므로, 락을 쥔 채
            //   join 해도 교착이 없다 — 이 순서가 성립하는 이유를 여기 적어 둔다.
            pbv_seal_unwind_prefetch_locked();   // ★★A5 — predictor 도 같은 사유로 여기서 접는다
            dfac_dctl_copy_stop_and_join();
            RemoveVectoredExceptionHandler(g_mod.veh_handle);
            g_mod.veh_handle = nullptr;
            // ★R18(교차 r6 G7) — metrics 는 바로 위에서 **이미 열렸다**. 여기서 안 닫으면
            //   handle 이 살아남고 header 1줄짜리 파일이 디스크에 남아 `CREATE_NEW` 재시도를
            //   막는다. 실패한 seal 은 계측 아티팩트를 남기지 않는다.
            metrics_discard_locked();
            virtual_release_locked();
            err = disp_err;
            return -1;
        }
    }
    return 0;
}

int seal_virtual_commit_locked() {

    xstore_publish_epoch_locked();   // E-2a adapter map lifecycle stages 2 and 3
    g_mod.sealed.store(true, std::memory_order_release);
    g_v3_stage.store(V3_STAGE_SEALED, std::memory_order_relaxed);   // ★A3 완주
    return 0;
}

int seal_virtual_locked(const std::wstring & dir_w, const json & j,
                        const ggml_moe_direct_model & model,
                        const ggml_moe_direct_source * srcs, int32_t n_srcs,
                        bool allow_sector_fallback,
                        uint64_t budget_bytes, uint32_t touch_abort,
                        const std::wstring & expects_dir_w,
                        const char * override_profile_id, const char * override_digest,
                        const char * override_scope, std::string & err) {
    int stage_rc = 0;
    ParsedManifest pm;
    std::string manifest_sha;
    std::unordered_map<std::string, std::pair<int32_t, int32_t>> name_to_lp;
    std::vector<char> pending_kept;
    int32_t v3_max_bounce_per_record = 0;
    uint64_t v3_max_staging_bytes = 0;
    int32_t v3_staging_plan = 0;
    int32_t v3_coal_max_parts = 0;


    stage_rc = seal_virtual_validate_locked(dir_w, j, model, srcs, n_srcs, allow_sector_fallback, expects_dir_w, override_profile_id,
                                            override_digest, override_scope, err, pm,
                                            manifest_sha, name_to_lp, pending_kept);
    if (stage_rc != 0) { return stage_rc; }

    // ---- ★A2 — 자원 확보(소스 핸들 집합 → 슬롯 풀 → registry → 보호) ----
    if (!virtual_acquire_locked(pm, budget_bytes, touch_abort, manifest_sha, name_to_lp, pending_kept, err)) {
        stage_rc = -1;
        return stage_rc;   // acquire returned false; it has already given back what it took
    }

    // ---- ★A3 — 소비자 기동(child scheduler 배선 + dispatcher). 여기서부터 seal 이 성립한다 ----
    virtual_fill_probe_locked();   // A2 자원 증거는 계속 남긴다(관측 창구 유지)

    stage_rc = seal_virtual_test_fail_hook_locked(err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_resolve_params_locked(err, v3_max_bounce_per_record, v3_max_staging_bytes, v3_staging_plan, v3_coal_max_parts);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_resolve_io3_locked();
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_note_external_locked();
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_fix_io_path_locked();
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_install_veh_locked(err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_alloc_coalesce_pool_locked(err, v3_coal_max_parts);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_alloc_staging_pool_locked(err, v3_max_bounce_per_record, v3_max_staging_bytes, v3_staging_plan);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_start_copy_worker_locked(err, v3_max_bounce_per_record);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_note_prefetch_locked(model, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_apply_ncap_locked();
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_open_metrics_locked(err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_start_dispatcher_locked(err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_virtual_commit_locked();
    if (stage_rc != 0) { return stage_rc; }

    return 0;
}

// ===========================================================================
// 11-0. OPEN_ARCH B axis - the template seal approval branch
//       (OPENARCH_B_SPEC_DRAFT v0.2 sections 2-2 / 2-3a / 2-3b).
//       It REPLACES the expect-approval part of the model-pin path. The common F1 tail (partial
//       marker, verify_report PASS gate, manifest cache-key binding) and everything after it in the
//       existing seal are NOT branched - they rejoin unchanged (sections 2-3c/d: no bypass).
//       The caller holds g_mod.mutex. Returns 0 on success, <0 on failure (reason in err).
// ===========================================================================

int arch_template_gate_locked(const std::wstring & dir_w,
                              const ggml_moe_direct_model & model,
                              const ParsedManifest & pm,
                              std::string & out_inventory_sha,
                              std::string & err) {
    // ---- 2-2: parse the lock prefix + catalog approval (an unregistered id and a version
    //      mismatch are reported as distinct reasons) ----
    const std::string suffix = pm.ref_profile_id.substr(std::strlen(ARCH_TEMPLATE_LOCK_PREFIX));
    const size_t at = suffix.find('@');
    if (at == std::string::npos || at == 0 || at + 1 >= suffix.size() || suffix.rfind('@') != at) {
        err = "arch-template reference_lock.profile_id is malformed (expected "
              "arch-template:<template_id>@<version>): " + pm.ref_profile_id;
        return -1;
    }
    const std::string tpl_id  = suffix.substr(0, at);
    const std::string tpl_ver = suffix.substr(at + 1);

    const ArchTemplateEntry * tpl = nullptr;
    bool id_registered = false;
    for (const auto & e : ARCH_TEMPLATE_CATALOG) {
        if (tpl_id == e.template_id) {
            id_registered = true;
            if (tpl_ver == e.template_version) tpl = &e;
            break;
        }
    }
    if (!id_registered) {
        err = "arch template is not registered in the frozen ARCH_TEMPLATE_CATALOG: " + tpl_id; return -1;
    }
    if (!tpl) {
        err = "arch template version is not approved: " + tpl_id + "@" + tpl_ver; return -1;
    }

    // ---- 2-3a: derived expect approval. The path is FIXED to the repack output directory
    //      (repack_experts.py:1074 - the bundle expects_dir is never used by the template branch). ----
    const std::wstring expect_path = dir_w + L"\\derived.expect.json";
    {
        std::string file_sha, sherr;
        if (!sha256_file(expect_path, file_sha, sherr)) {
            err = "cannot read derived.expect.json in the repack output directory: " + sherr; return -1;
        }
        if (to_lower_hex(file_sha) != to_lower_hex(pm.ref_expect_sha256)) {
            err = "re-hashed derived.expect.json != manifest.reference_lock.expect_sha256 "
                  "(tampered or misplaced derived expect)"; return -1;
        }
    }
    json ex;
    {
        std::ifstream ifs(expect_path, std::ios::binary);
        if (!ifs.is_open()) { err = "cannot open derived.expect.json"; return -1; }
        try {
            ifs >> ex;
        } catch (const std::exception & e) {
            err = std::string("derived.expect.json JSON parse failed: ") + e.what(); return -1;
        }
    }
    if (!ex.is_object()) { err = "the top level of derived.expect.json is not a JSON object"; return -1; }

    // Exactly 13 keys (section 1 schema - unlike a registered expect, only the template parser
    // enforces this: D3).
    static const char * const EXPECT_KEYS[] = {
        "expect_schema_version", "derived_from", "template_id", "template_version",
        "inventory_sha256", "routed_scope", "arch", "n_layer", "n_expert",
        "n_expert_used", "routed_tensors", "expert_bytes_total", "sources",
    };
    if (ex.size() != 13) {
        err = "derived.expect.json key count(" + std::to_string(ex.size()) + ") != 13 (frozen schema)"; return -1;
    }
    for (const char * k : EXPECT_KEYS) {
        if (!ex.contains(k)) { err = std::string("derived.expect.json is missing the key: ") + k; return -1; }
    }
    for (const char * k : { "expect_schema_version", "derived_from", "template_id", "template_version",
                            "inventory_sha256", "routed_scope", "arch" }) {
        if (!ex[k].is_string()) { err = std::string("derived.expect.json.") + k + " is not a string"; return -1; }
    }
    for (const char * k : { "n_layer", "n_expert", "n_expert_used", "routed_tensors" }) {
        if (!json_is_int(ex[k])) { err = std::string("derived.expect.json.") + k + " is not an integer"; return -1; }
    }
    if (!json_is_uint(ex["expert_bytes_total"])) {
        err = "derived.expect.json.expert_bytes_total is not an unsigned integer"; return -1;
    }
    if (!ex["sources"].is_array()) { err = "derived.expect.json.sources is not an array"; return -1; }

    // D3: only the template parser enforces "1.0" exactly (the registered path still looks at the
    // file hash alone - that behaviour is unchanged).
    if (ex["expect_schema_version"].get<std::string>() != "1.0") {
        err = "derived.expect.json.expect_schema_version != \"1.0\" (got: " +
              ex["expect_schema_version"].get<std::string>() + ")"; return -1;
    }
    // derived_from == the lock's <template_id>@<version>, three-way against the template_id /
    // template_version fields as well.
    const std::string want_from = tpl_id + "@" + tpl_ver;
    if (ex["derived_from"].get<std::string>() != want_from) {
        err = "derived.expect.json.derived_from(" + ex["derived_from"].get<std::string>() +
              ") != reference_lock(" + want_from + ")"; return -1;
    }
    if (ex["template_id"].get<std::string>() != tpl_id) {
        err = "derived.expect.json.template_id(" + ex["template_id"].get<std::string>() +
              ") != reference_lock template_id(" + tpl_id + ")"; return -1;
    }
    if (ex["template_version"].get<std::string>() != tpl_ver) {
        err = "derived.expect.json.template_version(" + ex["template_version"].get<std::string>() +
              ") != reference_lock template_version(" + tpl_ver + ")"; return -1;
    }
    // manifest routed_scope == expect routed_scope.
    const std::string ex_scope = ex["routed_scope"].get<std::string>();
    if (ex_scope != pm.routed_scope) {
        err = "derived.expect.json.routed_scope(" + ex_scope + ") != manifest routed_scope(" +
              pm.routed_scope + ")"; return -1;
    }
    const std::string ex_inv = to_lower_hex(ex["inventory_sha256"].get<std::string>());
    if (ex_inv.size() != 64) { err = "derived.expect.json.inventory_sha256 length != 64"; return -1; }
    for (char c : ex_inv) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            err = "derived.expect.json.inventory_sha256 is not hexadecimal"; return -1;
        }
    }

    // ---- 2-3b: independent inventory closure.
    //      The expected name set is GENERATED from the arch formula (reusing the repacker's
    //      "regex discovery" is forbidden). Type/dims/layer/scope are taken from the LIVE values of
    //      the registered canonical routed tensors. ----
    int32_t first_layer = 0;
    if (tpl->layer_rule == TplLayerRule::LEADING_DENSE) {
        if (model.n_layer_dense_lead < 0 || model.n_layer_dense_lead >= model.n_layer) {
            err = "n_layer_dense_lead(" + std::to_string(model.n_layer_dense_lead) +
                  ") is out of range (0 <= x < n_layer(" + std::to_string(model.n_layer) + "))"; return -1;
        }
        first_layer = model.n_layer_dense_lead;
    }
    const bool is_exec = (pm.routed_scope == "execution");
    if (is_exec && tpl->nextn != TplNextn::EXECUTION_DEFAULT) {
        err = "routed_scope=execution but the " + tpl_id + " arch template has no NextN clause"; return -1;
    }
    // The domain of n_layer_nextn (0 <= x < n_layer) and "execution but nextn==0" were already
    // rejected by validate_manifest_v2 - only its result is used here.
    const int32_t last_excl = model.n_layer - (is_exec ? model.n_layer_nextn : 0);
    if (first_layer >= last_excl) {
        err = "the arch template layer set is empty (first=" + std::to_string(first_layer) +
              ", end_exclusive=" + std::to_string(last_excl) + ", n_layer=" + std::to_string(model.n_layer) +
              ", scope=" + pm.routed_scope + ")"; return -1;
    }

    std::vector<std::string> want_suffixes;
    for (const char * k : ARCH_TEMPLATE_WEIGHT_KINDS) want_suffixes.push_back(std::string("ffn_") + k + "_exps.weight");
    if (tpl->bias == TplBias::REQUIRED) {
        for (const char * k : ARCH_TEMPLATE_WEIGHT_KINDS) want_suffixes.push_back(std::string("ffn_") + k + "_exps.bias");
    }
    std::vector<std::string> want_names;
    want_names.reserve((size_t)(last_excl - first_layer) * want_suffixes.size());
    for (int32_t l = first_layer; l < last_excl; l++) {
        for (const auto & s : want_suffixes) want_names.push_back("blk." + std::to_string(l) + "." + s);
    }
    std::unordered_map<std::string, int> want_index;
    for (size_t i = 0; i < want_names.size(); i++) want_index[want_names[i]] = 1;

    // The live registrations (scope applied). execution drops the NextN tail layers from the managed
    // set - the same boundary (layer < n_layer - n_layer_nextn) that pending_kept uses in the
    // section 11 body below.
    std::unordered_map<std::string, const PendingReg *> live;
    for (const auto & pr : g_mod.pending) {
        int32_t player = -1; std::string kind, suf;
        if (!parse_routed_tensor_name(pr.name, player, kind, suf)) {
            err = "arch template: a registered tensor name is not a canonical routed name: " + pr.name; return -1;
        }
        if (is_exec && player >= last_excl) continue;   // NextN-excluded (still registered, not managed)
        if (!live.emplace(pr.name, &pr).second) {
            err = "arch template: duplicate routed registration: " + pr.name; return -1;
        }
    }
    // A formula name missing from the live registrations is a failure.
    for (const auto & n : want_names) {
        if (!live.count(n)) {
            err = "the " + tpl_id + " arch template formula expects routed tensor " + n +
                  " but it is not registered (live inventory diverged from the formula)"; return -1;
        }
    }
    // A registered canonical routed tensor outside the formula is a failure (r1 F5 narrowed scope -
    // non-canonical expert-like tensors are never registered by the loader, so they are not in play).
    for (const auto & kv : live) {
        if (!want_index.count(kv.first)) {
            err = "registered routed tensor " + kv.first + " is outside the " + tpl_id +
                  " arch template formula (layers " + std::to_string(first_layer) + ".." +
                  std::to_string(last_excl - 1) + ", scope=" + pm.routed_scope + ")"; return -1;
        }
    }

    // Row assembly - the section 1 wire fact verbatim. D4: the type is the enum -> frozen wire-name
    // mapping and dims carries EVERY axis (that binding is the body of the type/shape check).
    std::vector<std::string> rows;
    rows.reserve(want_names.size());
    for (const auto & n : want_names) {
        const PendingReg * pr = live[n];
        const ggml_tensor * t = pr->tensor;
        if (!t) { err = "arch template: registered routed tensor has no live handle: " + n; return -1; }
        const char * wt = wire_type_name(t->type);
        if (!wt) {
            err = "arch template: the live type of " + n + " is not in the frozen type-trait table"; return -1;
        }
        const int nd = ggml_n_dims(t);
        if (nd < 2) {
            err = "arch template: live tensor " + n + " has too few axes (an expert axis is required)"; return -1;
        }
        int32_t layer = -1; std::string kind, suf;
        parse_routed_tensor_name(n, layer, kind, suf);   // want_names are formula output, so always succeeds
        std::string row = n;
        row += '\t'; row += wt;
        row += '\t';
        for (int d = 0; d < nd; d++) {
            if (d) row += ',';
            row += std::to_string((long long)t->ne[d]);
        }
        row += '\t'; row += std::to_string(layer);
        row += '\t'; row += pm.routed_scope;
        row += '\n';
        rows.push_back(std::move(row));
    }
    std::string got;
    if (!inventory_digest_hex(rows, got)) { err = "arch template: inventory digest computation failed"; return -1; }
    if (got != ex_inv) {
        err = "independently recomputed inventory_sha256(" + got + ") != derived.expect.json inventory_sha256(" +
              ex_inv + ") - the engine's arch-formula closure disagrees with the producer's derivation"; return -1;
    }
    out_inventory_sha = got;
    return 0;
}

int seal_bin_validate_locked(const std::wstring & dir_w, const ggml_moe_direct_model & model,
                             const ggml_moe_direct_source * srcs, int32_t n_srcs,
                             bool allow_sector_fallback, uint64_t budget_bytes,
                             uint32_t touch_abort, const std::wstring & expects_dir_w,
                             const char * override_profile_id, const char * override_digest,
                             const char * override_scope, std::string & err,
                             const std::wstring & manifest_path, json & j,
                             ParsedManifest & out_pm, std::string & out_repack_manifest_sha,
                             std::string & out_template_inventory_sha,
                             std::unordered_map<std::string, std::pair<int32_t, int32_t>> & out_name_to_lp,
                             std::vector<char> & out_pending_kept) {
    g_v3_stage.store(V3_STAGE_NONE, std::memory_order_relaxed);
    ManifestMode mmode = ManifestMode::BIN;
    if (!detect_manifest_mode(j, mmode, err)) return -1;
    // ★A2 — 주소 계약 분기가 읽는 단일 출처. 게이트가 확정한 값을 그대로 싣는다(bin 이면
    //   BIN 이라 v2 경로의 반환 의미는 달라지지 않는다).
    g_mod.manifest_mode = mmode;
    // ★R13 — metrics header `schema_version` echo(§2 식별 정직성). 게이트가 이미 검증한 값을
    //   그대로 싣는다(재파싱·재판정 없음 — 사본이 아니라 echo 다).
    if (j.contains("schema_version") && j["schema_version"].is_string())
        g_mod.manifest_schema_version = j["schema_version"].get<std::string>();
    if (mmode == ManifestMode::VIRTUAL) {
        return seal_virtual_locked(dir_w, j, model, srcs, n_srcs, allow_sector_fallback,
                                   budget_bytes, touch_abort, expects_dir_w,
                                   override_profile_id, override_digest, override_scope, err);
    }

    ParsedManifest pm;
    if (!validate_manifest_v2(j, model, srcs, n_srcs, pm, err)) return -1;

    // ---- ★F1 reference_lock 소비자 결속(설계 §3 불변식 8 — 소비자 기동 검증이 독립 재확인) ----
    // ① profile_id 카탈로그 등재 ② manifest.expect_sha256 == 승인 digest ③ expect 파일 재해시
    // == 승인 digest ④ 재팩 미완료 표식(experts.bin.partial) 부재 ⑤ verify_report.json 마지막
    // 레코드가 **verify 게이트 통과**(pass==true·pairs 등식·결함 배열 3종 공집합) + reference_lock
    // == manifest ⑥ ★report.manifest_sha256 == manifest.json 실제 바이트 해시(cache key 결속 —
    // 타 산출물 report 재사용/verify 이후 manifest 변경 차단). 하나라도 실패/미설정 = seal 중단(fail-closed).
    // ★R4 — 게이트 ⑦이 계산한 manifest 실SHA 를 seal 밖으로 보존한다(구: 대조 후 폐기).
    std::string repack_manifest_sha;
    std::string template_inventory_sha;
    {
        // OPEN_ARCH B axis (section 2-2): an "arch-template:" prefix on reference_lock.profile_id
        // selects the template branch. Without the prefix the model-pin path below is unchanged
        // (the branch is additive and confined to the expect-approval part).
        const size_t tpl_pre_len = std::strlen(ARCH_TEMPLATE_LOCK_PREFIX);
        const bool is_arch_template =
            pm.ref_profile_id.size() > tpl_pre_len &&
            pm.ref_profile_id.compare(0, tpl_pre_len, ARCH_TEMPLATE_LOCK_PREFIX) == 0;
        if (is_arch_template) {
            if (arch_template_gate_locked(dir_w, model, pm, template_inventory_sha, err) != 0) return -1;
        } else {
        if (expects_dir_w.empty()) { err = "MOE_DIRECT_EXPECTS_DIR is not set (reference_lock cannot be bound; fail-closed)"; return -1; }
        std::string approved, approved_scope;
        if (!catalog_lookup(pm.ref_profile_id, override_profile_id, override_digest, override_scope, approved, approved_scope)) {
            err = "reference_lock.profile_id is not registered (not in the catalog): " + pm.ref_profile_id; return -1;
        }
        if (to_lower_hex(pm.ref_expect_sha256) != to_lower_hex(approved)) {
            err = "manifest.reference_lock.expect_sha256 != approved catalog digest"; return -1;
        }
        // ★부록A: 카탈로그 scope == manifest routed_scope(정규화 후) 4자 일치의 소비자 몫 — 불일치=fail-closed.
        if (approved_scope != pm.routed_scope) {
            err = "catalog scope(" + approved_scope + ") != manifest routed_scope(" + pm.routed_scope + ")"; return -1;
        }
        std::wstring expect_path = expects_dir_w + L"\\" + utf8_to_wide(pm.ref_profile_id) + L".expect.json";
        std::string file_sha, sherr;
        if (!sha256_file(expect_path, file_sha, sherr)) { err = "cannot re-hash the expect file: " + sherr; return -1; }
        if (to_lower_hex(file_sha) != to_lower_hex(approved)) {
            err = "re-hashed expect file != approved digest (tampered or misplaced)"; return -1;
        }
        }   // section 2-3c: the tail below (partial marker, verify_report PASS gate, manifest
            // cache-key binding) is traversed by the template branch with THE SAME CODE - the
            // branch ends at the expect-approval part.
        // ★재팩 미완료 표식 — 프로듀서는 experts.bin 쓰기 직전에 <dir>\experts.bin.partial 을 만들고
        // verify pass 일 때만 제거한다(repack_experts.py:1002·1027·1099·1100). 잔존 = 중단된 재팩
        // 또는 verify 미통과 산출물 → seal 거부. ★INVALID_FILE_ATTRIBUTES는 "부재"만이 아니라
        // 모든 조회 실패(ACL·경로·파일시스템 오류)에서 반환된다 — ERROR_FILE_NOT_FOUND만 부재로
        // 인정하고 그 외 오류는 부재 확증 불가로 seal 거부(fail-closed·Codex 교차 정정 반영).
        std::wstring partial_path = dir_w + L"\\experts.bin.partial";
        if (GetFileAttributesW(partial_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
            err = "incomplete-repack marker present (experts.bin.partial) - aborted or unverified artifact (fail-closed)"; return -1;
        }
        {
            const DWORD gle = GetLastError();
            if (gle != ERROR_FILE_NOT_FOUND) {
                err = "cannot query the incomplete-repack marker (experts.bin.partial) (GetLastError=" + std::to_string((unsigned long)gle) +
                      ") - cannot confirm its absence (fail-closed)"; return -1;
            }
        }
        std::string vr_pid, vr_sha, vr_manifest_sha, vrerr;
        std::wstring vr_path = dir_w + L"\\verify_report.json";
        if (!read_verify_report_gate(vr_path, vr_pid, vr_sha, vr_manifest_sha, vrerr)) { err = "verify_report gate failed: " + vrerr; return -1; }
        if (vr_pid != pm.ref_profile_id || to_lower_hex(vr_sha) != to_lower_hex(pm.ref_expect_sha256)) {
            err = "verify_report.reference_lock != manifest.reference_lock"; return -1;
        }
        // ★⑦ cache key 결속: report 가 검증했다고 주장하는 manifest 바이트 == 지금 로드한 manifest.json.
        // 불일치 = 타 재팩물의 report 를 옮겨왔거나 verify 이후 manifest 가 바뀐 것 → seal 거부.
        // (한계: verify 이후의 experts.bin bit-rot 은 이 키로 검출되지 않는다 — 신뢰 입력 모델·
        //  reviews/codex_verify_gate_xcheck.md §3 명시. bin 전량 재읽기는 기각안.)
        {
            std::string mf_sha, mferr;
            if (!sha256_file(manifest_path, mf_sha, mferr)) { err = "cannot re-hash manifest.json: " + mferr; return -1; }
            repack_manifest_sha = to_lower_hex(mf_sha);   // ★R4 — identity echo 원천(추가 I/O 0)
            if (to_lower_hex(mf_sha) != to_lower_hex(vr_manifest_sha)) {
                err = "verify_report.manifest_sha256 != actual manifest.json hash - this report did not "
                      "verify this manifest (report reused from another artifact, or manifest changed after verify; fail-closed)"; return -1;
            }
        }
    }

    // ---- 이름→(mlpos,part_idx) 해소 맵 + M_all ----
    std::unordered_map<std::string, std::pair<int32_t,int32_t>> name_to_lp;
    for (size_t mlpos = 0; mlpos < pm.layers.size(); mlpos++) {
        for (size_t pidx = 0; pidx < pm.layers[mlpos].parts.size(); pidx++) {
            const std::string & nm = pm.layers[mlpos].parts[pidx].source_tensor;
            if (name_to_lp.count(nm)) { err = "M_all duplicate source_tensor: " + nm; return -1; }
            name_to_lp[nm] = { (int32_t)mlpos, (int32_t)pidx };
        }
    }

    // ---- ★부록A dual-scope: 집합 선택(등록은 전수 유지·seal 내에서 scope 집합만 관리 대상).
    //      L_raw = pending 전수 / L_exec = layer < (n_layer - n_layer_nextn) 인 pending.
    //      all → M == L_raw / execution → M == L_exec. 제외분(execution)은 등록 해제(일반 mmap 잔류). ----
    const bool is_execution = (pm.routed_scope == "execution");
    const int32_t exec_threshold = model.n_layer - model.n_layer_nextn;
    std::vector<char> pending_kept(g_mod.pending.size(), 1);
    if (is_execution) {
        for (size_t i = 0; i < g_mod.pending.size(); i++) {
            int32_t player = -1;
            if (!parse_blk_layer(g_mod.pending[i].name, player)) {
                err = "execution scope: cannot parse the layer from a pending tensor name: " + g_mod.pending[i].name; return -1;
            }
            pending_kept[i] = (player < exec_threshold) ? 1 : 0;
        }
    }

    // ---- 등록 3집합 대조: M_all == L_scope 양방향(§4-2① · 부록A scope 적용) ----
    // L_scope = pending 중 관리 대상(all=전수·execution=layer<threshold). M_all = manifest parts flatten.
    {
        std::unordered_map<std::string, int> l_seen; // scope 집합(kept)만
        int32_t l_kept = 0;
        for (size_t i = 0; i < g_mod.pending.size(); i++) {
            if (!pending_kept[i]) continue; // execution 제외분 — M 대조에서 빠짐(등록 해제 대상)
            const std::string & nm = g_mod.pending[i].name;
            if (l_seen.count(nm)) { err = "L_scope duplicate registration: " + nm; return -1; }
            l_seen[nm] = 1;
            l_kept++;
            if (!name_to_lp.count(nm)) { err = "L_scope overflow (tensor absent from the manifest): " + nm; return -1; }
        }
        if (l_kept != (int32_t)name_to_lp.size()) {
            err = "L_scope(" + std::to_string(l_kept) + ") != M_all(" + std::to_string(name_to_lp.size()) + ")";
            return -1;
        }
        for (const auto & kv : name_to_lp) {
            if (!l_seen.count(kv.first)) { err = "M_all missing (tensor absent from the scope set): " + kv.first; return -1; }
        }
    }

    // R-B4S stage epilogue: what this stage produces for the stages after it
    out_pm = pm;
    out_repack_manifest_sha = repack_manifest_sha;
    out_template_inventory_sha = template_inventory_sha;
    out_name_to_lp = name_to_lp;
    out_pending_kept = pending_kept;
    return 0;
}

int seal_bin_size_probe_locked(const std::wstring & bin_path, bool allow_sector_fallback, const ParsedManifest & pm,
                               std::string & err) {
    // ---- experts.bin 실물 크기 대조 ----
    HANDLE probe = CreateFileW(bin_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (probe == INVALID_HANDLE_VALUE) { err = "cannot open experts.bin (size query) err=" + std::to_string(GetLastError()); return -1; }
    LARGE_INTEGER file_size{};
    BOOL got_size = GetFileSizeEx(probe, &file_size);
    CloseHandle(probe);
    if (!got_size) { err = "experts.bin GetFileSizeEx failed"; return -1; }
    if ((uint64_t)file_size.QuadPart != pm.bin_file_bytes) {
        err = "experts.bin actual size mismatch: " + std::to_string(file_size.QuadPart) + " != " + std::to_string(pm.bin_file_bytes);
        return -1;
    }

    // ---- 섹터 재질의 ----
    uint64_t a_runtime = 0;
    if (!query_a_runtime(bin_path, allow_sector_fallback, a_runtime, err)) return -1;
    if (pm.align_bytes % a_runtime != 0) { err = "align_bytes % A_runtime != 0 (A_runtime=" + std::to_string(a_runtime) + ")"; return -1; }
    // slot_stride_max 도 A_runtime 배수(모든 stride가 align_bytes 배수·align_bytes%A_runtime==0)
    if (pm.slot_stride_max % a_runtime != 0) { err = "slot_stride_max % A_runtime != 0"; return -1; }

    return 0;
}

int seal_bin_open_data_file_locked(const std::wstring & bin_path, uint64_t budget_bytes, const ParsedManifest & pm,
                                   std::string & err, HANDLE & data_file, int32_t & out_n_slots,
                                   size_t & out_pool_bytes) {
    // ---- 데이터 파일 오픈(NO_BUFFERING|OVERLAPPED) ----
    data_file = CreateFileW(bin_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                    OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (data_file == INVALID_HANDLE_VALUE) { err = "cannot open experts.bin err=" + std::to_string(GetLastError()); return -1; }

    // ---- 슬롯 용량 불변식(§4-7): n_slots >= n_expert ----
    if (pm.slot_stride_max == 0) { CloseHandle(data_file); err = "slot_stride_max==0"; return -1; }
    int32_t n_slots = (int32_t)(budget_bytes / pm.slot_stride_max);
    if (n_slots < pm.n_expert) {
        CloseHandle(data_file);
        err = "slot capacity invariant violated: n_slots(" + std::to_string(n_slots) + ") < n_expert(" +
              std::to_string(pm.n_expert) + ") - budget_bytes(" + std::to_string(budget_bytes) +
              ") < n_expert*slot_stride_max(" + std::to_string((uint64_t)pm.n_expert * pm.slot_stride_max) + ")";
        return -1;
    }
    size_t pool_bytes = (size_t)n_slots * (size_t)pm.slot_stride_max;
    // R-B4S stage epilogue: what this stage produces for the stages after it
    out_n_slots = n_slots;
    out_pool_bytes = pool_bytes;
    return 0;
}

int seal_bin_alloc_slot_pool_locked(HANDLE data_file, size_t pool_bytes, std::string & err, void *& pool_base) {
    pool_base = VirtualAlloc(nullptr, pool_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!pool_base) { CloseHandle(data_file); err = "VirtualAlloc(slot pool) failed err=" + std::to_string(GetLastError()); return -1; }
    return 0;
}

int seal_bin_publish_layout_locked(uint32_t touch_abort, ParsedManifest & pm, const std::string & repack_manifest_sha,
                                   HANDLE data_file, int32_t n_slots, size_t pool_bytes,
                                   void * pool_base) {

    // ---- 상태 확정(레이아웃) ----
    g_mod.arch = pm.arch;
    g_mod.n_layer = pm.n_layer;
    g_mod.n_expert = pm.n_expert;
    g_mod.n_expert_used = pm.n_expert_used;
    g_mod.moe_layers = pm.moe_layers;
    g_mod.layer_to_mlpos.clear();
    for (size_t i = 0; i < pm.moe_layers.size(); i++) g_mod.layer_to_mlpos[pm.moe_layers[i]] = (int32_t)i;
    g_mod.layers = std::move(pm.layers);
    g_mod.sources = std::move(pm.sources);
    g_mod.align_bytes = pm.align_bytes;
    g_mod.slot_stride_max = pm.slot_stride_max;
    g_mod.bin_file_bytes = pm.bin_file_bytes;
    g_mod.ref_profile_id = pm.ref_profile_id;
    g_mod.ref_expect_sha256 = pm.ref_expect_sha256;
    g_mod.repack_manifest_sha256 = repack_manifest_sha;   // ★R4 — 재팩물 단위 identity 축
    g_mod.routed_scope = pm.routed_scope;
    g_mod.data_file = data_file;
    g_mod.pool_base = pool_base;
    g_mod.pool_bytes = pool_bytes;
    g_mod.n_slots = n_slots;
    g_mod.touch_abort_threshold = touch_abort;

    g_mod.slots = std::vector<Slot>((size_t)n_slots);
    for (int32_t i = 0; i < n_slots; i++) g_mod.slots[(size_t)i].base = (char *)pool_base + (size_t)i * (size_t)pm.slot_stride_max;
    g_mod.record_table = std::vector<RecordEntry>((size_t)g_mod.moe_layers.size() * (size_t)pm.n_expert);

    return 0;
}

int seal_bin_build_registry_locked(const ParsedManifest & pm,
                                   std::unordered_map<std::string, std::pair<int32_t, int32_t>> & name_to_lp,
                                   const std::vector<char> & pending_kept, HANDLE data_file,
                                   int32_t n_slots, void * pool_base, std::string & err) {
    // ---- registry 확정(R_host만) + 텐서별 대조 + attestation(★부록A: raw/managed/excluded) ----
    // execution 제외분(pending_kept[i]==0)은 registry/보호/슬롯 대상에서 제외 — 등록 해제(일반 mmap 잔류).
    int32_t att_raw = 0, att_managed = 0, att_host = 0, att_nonhost = 0;
    g_mod.registry.clear();
    for (size_t i = 0; i < g_mod.pending.size(); i++) {
        const PendingReg & pr = g_mod.pending[i];
        att_raw++;
        if (!pending_kept[i]) continue; // 제외분 — direct 등록·보호·슬롯 대상 아님(일반 mmap 잔류)
        att_managed++;
        auto lp = name_to_lp[pr.name];
        int32_t mlpos = lp.first, pidx = lp.second;
        const PartInfo & part = g_mod.layers[(size_t)mlpos].parts[(size_t)pidx];
        // 텐서별 대조(§4-2): live ne[expert_axis]==n_expert · ggml_nbytes==part_bytes*n_expert
        // (per-expert slice==part_bytes 는 manifest 검증에서 이미 표 재계산으로 확인). part.expert_axis
        // 는 dims.size()-1 로 검증됨.
        const ggml_tensor * t = pr.tensor;
        if (t) {
            if ((int32_t)t->ne[part.expert_axis] != pm.n_expert) {
                err = "tensor " + pr.name + " ne[expert_axis]=" + std::to_string(t->ne[part.expert_axis]) +
                      " != n_expert=" + std::to_string(pm.n_expert);
                VirtualFree(pool_base, 0, MEM_RELEASE); CloseHandle(data_file); return -1;
            }
            uint64_t tbytes = (uint64_t)ggml_nbytes(t);
            if (tbytes != part.part_bytes * (uint64_t)pm.n_expert) {
                err = "tensor " + pr.name + " ggml_nbytes(" + std::to_string(tbytes) + ") != part_bytes*n_expert(" +
                      std::to_string(part.part_bytes * (uint64_t)pm.n_expert) + ")";
                VirtualFree(pool_base, 0, MEM_RELEASE); CloseHandle(data_file); return -1;
            }
        }
        if (pr.is_host) {
            att_host++;
            g_mod.registry[pr.tensor] = RegEntry{ mlpos, pidx };
        } else {
            att_nonhost++;
        }
    }
    g_mod.att_raw = att_raw;
    g_mod.att_managed = att_managed;
    g_mod.att_excluded_nextn = att_raw - att_managed;
    g_mod.att_all = att_managed; // ★부록A: 기존 att_all 의미=managed로 자연 수렴(all이면 raw==managed·excluded 0)
    g_mod.att_host = att_host;
    g_mod.att_nonhost = att_nonhost;
    g_mod.att_n_slots = n_slots;
    g_mod.att_n_slots_required = pm.n_expert;
    return 0;
}

int seal_bin_protect_tensors_locked(HANDLE data_file, void * pool_base, std::string & err) {

    // ---- 등록 텐서 보호(PAGE_NOACCESS) + interval 표(R_host만) ----
    g_mod.reg_tensor_info.clear();
    g_mod.reg_tensor_info.reserve(g_mod.registry.size());
    std::vector<ProtInterval> raw_intervals;
    raw_intervals.reserve(g_mod.registry.size());

    for (auto & kv : g_mod.registry) {
        const ggml_tensor * t = kv.first;
        const RegEntry & re = kv.second;
        int32_t tidx = (int32_t)g_mod.reg_tensor_info.size();
        RegisteredTensorInfo info{};
        info.layer = g_mod.moe_layers[(size_t)re.mlpos];
        info.part = re.part_idx;
        info.addr_start = (uintptr_t)t->data;
        info.nbytes = ggml_nbytes(t);
        g_mod.reg_tensor_info.push_back(info);

        uintptr_t byte_start = info.addr_start;
        uintptr_t byte_end = byte_start + info.nbytes;
        uintptr_t astart = (uintptr_t)align_up(byte_start, 4096);
        uintptr_t aend   = (uintptr_t)align_down(byte_end, 4096);
        if (aend <= astart) continue;
        DWORD old_protect = 0;
        if (!VirtualProtect((LPVOID)astart, aend - astart, PAGE_NOACCESS, &old_protect)) {
            VirtualFree(pool_base, 0, MEM_RELEASE);
            CloseHandle(data_file);
            err = "VirtualProtect(PAGE_NOACCESS) failed err=" + std::to_string(GetLastError());
            return -1;
        }
        ProtInterval iv{};
        iv.start = astart; iv.end = aend; iv.tensor_idx = tidx; iv.page_base_index = 0;
        iv.old_protect = (uint32_t)old_protect;
        raw_intervals.push_back(iv);
    }

    std::sort(raw_intervals.begin(), raw_intervals.end(),
              [](const ProtInterval & a, const ProtInterval & b) { return a.start < b.start; });
    uint32_t total_pages = 0;
    for (auto & iv : raw_intervals) { iv.page_base_index = total_pages; total_pages += (uint32_t)((iv.end - iv.start) / 4096); }
    g_mod.prot_intervals = std::move(raw_intervals);

    g_mod.page_touched = std::vector<std::atomic<uint8_t>>(total_pages);
    for (auto & a : g_mod.page_touched) a.store(0, std::memory_order_relaxed);
    g_mod.touch_ring = std::vector<TouchLogEntry>(256);
    g_mod.touch_ring_next.store(0, std::memory_order_relaxed);
    g_mod.touch_events.store(0, std::memory_order_relaxed);
    g_mod.touch_pages.store(0, std::memory_order_relaxed);
    return 0;
}

int seal_bin_resolve_io_selector_locked(HANDLE data_file, void * pool_base, std::string & err) {

    // ---- ★QD 레버 상한 해소(reviews/codex_qd_u6_design_xcheck.md 심의 3 D-2) ----
    // MOE_DIRECT_QD = 한 배치 acquire의 최대 outstanding read 수. 양의 정수만 수용하고 그 외
    // (비정수·0·음수)는 경고 로그 후 기본값. 상한 63으로 clamp(회수가 WaitForMultipleObjects
    // 한도 안이어야 함). ★metrics header 기록(바로 아래)보다 앞서야 하므로 이 위치.
    {
        int32_t qd = MOE_DIRECT_QD_DEFAULT;
        const char * qd_src = "default";
        const std::string qs = getenv_str("MOE_DIRECT_QD");
        if (!qs.empty()) {
            char * endp = nullptr;
            long v = strtol(qs.c_str(), &endp, 10);
            if (endp && *endp == '\0' && v > 0) {
                qd = (v > (long)MOE_DIRECT_QD_MAX) ? MOE_DIRECT_QD_MAX : (int32_t)v;
                qd_src = "env";
            } else {
                log_err("MOE_DIRECT_QD=\"%s\" is invalid (positive integers only) - using the default %d",
                        qs.c_str(), (int)MOE_DIRECT_QD_DEFAULT);
            }
        }
        dfac_qd_effective_set_locked(qd);
        dfac_qd_source_set_locked(qd_src);
    }

    // ---- ★B1 selector 해소(PHASEB_DESIGN §2 하드 구속 — seal 전 1회·이후 immutable) ----
    // ①qualification/test mode에서만 선택자가 존재 ②seal 전 1회 해소 ③미지정 production 기본값
    // = persistent ④test mode 밖 legacy 요청 = fail-close ⑤선택값 header echo
    // ⑥같은 commit/binary/config에서 selector만 바꿔도 Phase A v1 wire·ordered digest 불변.
    // ★metrics header 기록보다 앞서야 하므로 이 위치(QD 해소 바로 뒤).
    {
        // ★"명시적" qualification mode = env MOE_DIRECT_QUALIFY=1 뿐이다(seal_for_test 라는 사실
        //   만으로 암묵 허용하지 않는다 — 그래야 ④ fail-close가 실제로 행사 가능한 규약이 된다).
        const bool qualify = getenv_flag_is_1("MOE_DIRECT_QUALIFY");
        bool        persistent = true;
        const char * src = "default";
        const std::string want = getenv_str("MOE_DIRECT_IO_PATH");
        if (!want.empty()) {
            if (want == "persistent") {
                persistent = true;
                src = qualify ? "qualify" : "env";
            } else if (want == "legacy") {
                if (!qualify) {
                    CloseHandle(data_file);
                    VirtualFree(pool_base, 0, MEM_RELEASE);
                    err = "MOE_DIRECT_IO_PATH=legacy is allowed only in qualification/test mode "
                          "(MOE_DIRECT_QUALIFY=1) - fail-close";
                    return -1;
                }
                persistent = false;
                src = "qualify";
            } else {
                CloseHandle(data_file);
                VirtualFree(pool_base, 0, MEM_RELEASE);
                err = "MOE_DIRECT_IO_PATH unknown value (persistent|legacy only): " + want;
                return -1;
            }
        }
        dfac_io_path_persistent_set_locked(persistent);
        dfac_io_path_source_set_locked(src);
        dfac_io_path_qualified_set_locked(qualify);
    }

    // ---- ★정정 6호 — 오힌트 test 표면 게이팅(SPEC_PHASEB_TEST_HINT §1) ----
    //   MOE_DIRECT_PHASEB_TEST_HINT 는 qualification 전용 표면이다. ①qualifier 부재 ②무효값
    //   ③MOE_DIRECT_PHASEB=equiv 동시 설정 = init fail-close(명시 stderr + seal 거부).
    //   ★위치 구속: 바로 아래 B2 블록의 note_seal 이 capture 풀·intent_q·predictor worker 를
    //   기동하므로, 이 검사는 **그보다 앞**이어야 "resource/worker 기동 전 거부"가 성립한다.
    //   qualifier 판정은 위에서 해소된 기존 latch(g_mod.io_path_qualified)를 그대로 쓴다.
    {
        const char * th_reason = nullptr;
        if (ggml_moe_phaseb_test_hint_precheck(g_mod.io_path_qualified, &th_reason) != 0) {
            CloseHandle(data_file);
            VirtualFree(pool_base, 0, MEM_RELEASE);
            err = std::string("MOE_DIRECT_PHASEB_TEST_HINT fail-close: ") + (th_reason ? th_reason : "");
            return -1;
        }
    }

    // ---- ★B2 프리페치 k/N 해소(SPEC_PHASEB_WIRE §1 — 기동 1회·이후 immutable·echo) ----
    //      K·N 둘 다 설정·둘 다 유효일 때만 ON. 하나라도 미설정/무효 = OFF + off_reason echo.
    //      ★metrics header 기록보다 앞서야 하므로 이 위치(selector 해소 바로 뒤).
    return 0;
}

int seal_bin_note_prefetch_locked(const ggml_moe_direct_model & model, int32_t n_slots, HANDLE data_file, void * pool_base,
                                  std::string & err) {
    {
        ggml_moe_phaseb_resolve_prefetch(g_mod.qd_effective);
        std::vector<uint64_t> pb_strides;
        pb_strides.reserve(g_mod.layers.size());
        for (const auto & L : g_mod.layers) pb_strides.push_back(L.stride_bytes);
        // note_seal에서 모델 상수 확정 + (프리페치 ON이면) capture 풀/intent_q/worker 기동.
        const int pb_rc = ggml_moe_phaseb_note_seal(
                                  g_mod.n_expert, g_mod.n_expert_used,
                                  (int32_t)g_mod.moe_layers.size(), n_slots, g_mod.qd_effective,
                                  g_mod.moe_layers.data(), pb_strides.data(),
                                  // ★R1 #5 ② — 스냅샷 형상 확정(header phaseb_mem 실값)
                                  model.n_layer - model.n_layer_nextn, model.n_embd,
                                  model.f_norm_rms_eps > 0.0f ? model.f_norm_rms_eps : 1e-6f);
        if (pb_rc != 0) {
            // ★v0.3.1 이월 원자 ⓐ — worker 생성 실패 = **명시 실패 전파**(§2-2). 구 판은 반환값을
            //   버려서 bin 만 조용한 no-op(프리페치 ON echo + intent 생산자 0)로 빠질 수 있었다.
            //   v3 경로(:9949)와 **같은 판정·같은 처방 문면**을 쓴다 — mode 어휘만 다르다.
            //   ★rollback 범위 = 이 시점 **보유분 전량**이다. 순서도 v3 와 같다: PhaseB 게시분
            //     접기 → 보호 원복 → registry/슬롯/레이아웃 제거 → 슬롯 풀 반납 → bin 전용 축인
            //     데이터 파일 핸들 반납. 중간 셋은 `virtual_release_locked()` 안에 이미 한 벌
            //     있으므로 여기서 손으로 다시 쓰지 않는다(같은 계약의 문면 사본을 만들지 않는다).
            //   ★단일 반납: `pool_base` 는 위 레이아웃 확정에서 `g_mod.pool_base` 로 **게시된 뒤**라
            //     `virtual_release_locked()` 가 그 전역 하나만 반납한다 — 지역 사본으로 다시
            //     `VirtualFree` 하면 이중 해제다.
            //   ★미보유라 넣지 않는 것: metrics(아래 계측 블록) · VEH · dispatcher.
            pbv_seal_unwind_prefetch_locked();
            virtual_release_locked();
            CloseHandle(data_file);
            g_mod.data_file = INVALID_HANDLE_VALUE;   // reset_for_test 이중 해제 방지(dispatcher 실패 분기와 동형)
            err = "mode=bin prefetch seal failed: the PhaseB predictor worker could not be "
                  "started (fail-closed - a prefetch-ON run with no intent producer would be a "
                  "silent no-op) - fix: unset MOE_DIRECT_PREFETCH_K/MOE_DIRECT_PREFETCH_N to run "
                  "without prefetch";
            return -1;
        }
        // ★정정 6호 — 오힌트 표면 기동 로그 1줄(SPEC_PHASEB_TEST_HINT §3). offset 은 seal 에서
        //   확정되므로 note_seal 뒤에 찍는다(단일 출처 — direct 가 n_expert 를 재계산하지 않음).
        if (ggml_moe_phaseb_test_hint_rule() != GGML_MOE_PHASEB_TEST_HINT_NONE) {
            fprintf(stderr, "[moe-direct][phaseb-test-hint] rule=%s offset=%d n_expert=%d\n",
                    ggml_moe_phaseb_test_hint_rule_name(),
                    (int)ggml_moe_phaseb_test_hint_offset(), (int)g_mod.n_expert);
            fflush(stderr);
        }
        dfac_prefetch_note_flags_locked(g_mod.moe_layers.size());
        // ★５c-T(부록 14.4) — 수집 축 해소(note_seal 에서 1회 확정된 값을 그대로 받는다).
        //   OFF 면 아래 자료구조가 미할당이라 종전 형상·명령열과 동일하다.
        dfac_prefetch_note_census_locked(g_mod.n_slots);
        // ★R1 #6 ② — 프리페치는 영속 dispatcher만 소비한다. legacy selector와의 조합은
        //   소비자 없는 intent_q(무한 적재·처분 불가)를 만들므로 명시 fail-close.
        if (g_mod.pb_prefetch_on && !g_mod.io_path_persistent) {
            CloseHandle(data_file);
            VirtualFree(pool_base, 0, MEM_RELEASE);
            err = "valid MOE_DIRECT_PREFETCH_K/N combined with MOE_DIRECT_IO_PATH=legacy is not allowed - "
                  "prefetch admission requires the persistent dispatcher (fail-close)";
            return -1;
        }
        // equiv 모드 fail-close(§1 ★R1 #5): PA accuracy + 유효 K/N **둘 다** 필수.
        const char * eq_reason = nullptr;
        if (ggml_moe_phaseb_equiv_precheck(ggml_moe_phasea_mode() == GGML_MOE_PHASEA_ACCURACY,
                                           &eq_reason) != 0) {
            CloseHandle(data_file);
            VirtualFree(pool_base, 0, MEM_RELEASE);
            err = std::string("MOE_DIRECT_PHASEB=equiv fail-close: ") + (eq_reason ? eq_reason : "");
            return -1;
        }
        // ★observe 모드 seal 전제 fail-close(SPEC_FAMILY_SIGNAL §2.1 · SPEC_PHASEB_WIRE §13):
        //   supported adapter · PHASEA=timing · TEST_HINT 부재 · PREFETCH_K/N raw key 둘 다 부재.
        //   판정은 note_seal 이 **자원 할당 전에** 끝냈고 여기서는 그 결과만 소비한다.
        const char * ob_reason = nullptr;
        if (ggml_moe_phaseb_observe_precheck(&ob_reason) != 0) {
            CloseHandle(data_file);
            VirtualFree(pool_base, 0, MEM_RELEASE);
            err = std::string("MOE_DIRECT_PHASEB=observe fail-close: ") + (ob_reason ? ob_reason : "");
            return -1;
        }
    }

    return 0;
}

int seal_bin_open_metrics_locked(HANDLE data_file, void * pool_base, std::string & err) {
    // ---- 계측 JSONL ----
    std::string metrics_err;
    if (!metrics_open_if_requested_locked(metrics_err)) {
        VirtualFree(pool_base, 0, MEM_RELEASE);
        CloseHandle(data_file);
        err = metrics_err;
        return -1;
    }

    return 0;
}

int seal_bin_install_veh_locked(HANDLE data_file, void * pool_base, std::string & err) {
    // ---- VEH 등록 ----
    g_mod.veh_handle = AddVectoredExceptionHandler(1, moe_direct_veh_handler);
    if (!g_mod.veh_handle) {
        VirtualFree(pool_base, 0, MEM_RELEASE);
        CloseHandle(data_file);
        err = "AddVectoredExceptionHandler failed";
        return -1;
    }

    return 0;
}

int seal_bin_note_phasea_locked() {
    // ---- ★프리페치 Phase A 통지(관측 전용 — off면 note_seal 자체가 즉시 반환) ----
    //      모델 상수 + arm 시작 슬롯 상태(ledger SLOT_INIT)를 여기서 확정한다.
    g_mod.phasea_on = ggml_moe_phasea_enabled();
    if (g_mod.phasea_on) {
        std::vector<uint64_t> strides;
        strides.reserve(g_mod.layers.size());
        for (const auto & L : g_mod.layers) strides.push_back(L.stride_bytes);
        ggml_moe_phasea_note_seal(g_mod.n_expert, g_mod.n_expert_used,
                                  (int32_t)g_mod.moe_layers.size(), g_mod.n_slots,
                                  g_mod.moe_layers.data(), strides.data());
    }

    return 0;
}

int seal_bin_start_dispatcher_locked(HANDLE data_file, void * pool_base, std::string & err) {
    // ---- ★B1 영속 dispatcher 기동(persistent 경로에서만 — legacy는 배치 pump 그대로) ----
    if (g_mod.io_path_persistent) {
        std::string disp_err;
        if (!dispatcher_start_locked(disp_err)) {
            RemoveVectoredExceptionHandler(g_mod.veh_handle);
            g_mod.veh_handle = nullptr;
            VirtualFree(pool_base, 0, MEM_RELEASE);
            CloseHandle(data_file);
            g_mod.pool_base = nullptr; g_mod.pool_bytes = 0;
            g_mod.data_file = INVALID_HANDLE_VALUE; // reset_for_test 이중 해제 방지
            err = disp_err;
            return -1;
        }
    }

    return 0;
}

int seal_bin_commit_locked(const std::string & template_inventory_sha) {
    // OPEN_ARCH B axis - record the independently recomputed template inventory digest at the REAL
    // success boundary, immediately before sealed=true. Recording it earlier (e.g. beside
    // repack_manifest_sha256) would leave a non-empty accessor behind on every path that still
    // returns -1 after that point (I/O selector, PhaseB precheck, VEH, dispatcher start).
    // Empty on the model-pin path: that path never fills template_inventory_sha.
    g_mod.template_inventory_sha256 = template_inventory_sha;
    xstore_publish_epoch_locked();   // E-2a adapter map lifecycle stages 2 and 3
    g_mod.sealed.store(true, std::memory_order_release);
    return 0;
}

}  // namespace (anonymous) - R-B3b (RB3B_ATOM_SPEC.json :: seam_map.members)
// R-B3b: the definition below STAYS IN THIS UNIT at the line it has always been on - its banner
// section, and so its cluster attribution, is unchanged.  Only its LINKAGE moves: the test
// entry points in ggml-moe-direct-test-entry.cpp call it, so it leaves the anonymous
// namespace and ggml-moe-direct-internal.h carries the prototype.  Body tokens unchanged.
namespace ggml_moe_direct_impl {
int seal_common_locked_v2(const std::wstring & dir_w,
                          const ggml_moe_direct_model & model,
                          const ggml_moe_direct_source * srcs, int32_t n_srcs,
                          bool allow_sector_fallback,
                          uint64_t budget_bytes, uint32_t touch_abort,
                          const std::wstring & expects_dir_w,
                          const char * override_profile_id, const char * override_digest,
                          const char * override_scope,
                          std::string & err) {
    if (g_mod.sealed) { err = "already sealed"; return -1; }

    // ---- manifest 로드+검증(§4-1 전항 + 불변식 8) ----
    std::wstring manifest_path = dir_w + L"\\manifest.json";
    std::wstring bin_path = dir_w + L"\\experts.bin";

    // ★trace fail-closed 게이트(§4-5·배선 S12 v4) — 모든 uint8 cast(kernel trace 경로)보다 선행.
    // wire expert 필드는 v4에서도 uint8(0xFF=NA)로 불변이므로 n_expert>255 && trace 활성이면 기동
    // 중단(ID 255=NA 센티널과 충돌 → 불완전 트레이스 방지). model.n_expert를 쓰며(manifest 검증
    // 전이라도 안전 — 값 불일치는 아래 validate가 별도로 잡는다), 이 게이트가 seal 전체에서 가장 이르다.
    if (model.n_expert > 255 && ggml_moe_trace_enabled()) {
        err = "trace enabled and n_expert(" + std::to_string(model.n_expert) +
              ") > 255 - exceeds the wire v4 uint8 expert limit (fail-closed, startup aborted)";
        return -1;
    }

    json j;
    if (!load_manifest_json(manifest_path, j, err)) return -1;

    // ---- ★§Z-⑥ 모드 분기(SPEC_REPACK_V3 §2-1) — bin 경로와 virtual 경로의 **유일한 접점**.
    //      v2 manifest("2.0" ∧ mode 필드 부재)는 여기서 종전 경로로 그대로 빠지고, 바이트
    //      규약·검증 순서·오류 문면 어느 것도 달라지지 않는다. ----

    int stage_rc = 0;
    ParsedManifest pm;
    std::string repack_manifest_sha;
    std::string template_inventory_sha;
    std::unordered_map<std::string, std::pair<int32_t, int32_t>> name_to_lp;
    std::vector<char> pending_kept;
    HANDLE seal_bin_data_file = INVALID_HANDLE_VALUE;
    int32_t n_slots = 0;
    size_t pool_bytes = 0;
    void * seal_bin_pool_base = nullptr;

    stage_rc = seal_bin_validate_locked(dir_w, model, srcs, n_srcs, allow_sector_fallback, budget_bytes, touch_abort, expects_dir_w,
                                        override_profile_id, override_digest, override_scope,
                                        err, manifest_path, j, pm, repack_manifest_sha,
                                        template_inventory_sha, name_to_lp, pending_kept);
    if (stage_rc != 0) { return stage_rc; }
    // R-B4S: stage (1) carries the mode branch, so the parent reads the latch that branch
    //        published and returns the virtual seal's own success here.
    if (g_mod.manifest_mode == ManifestMode::VIRTUAL) { return 0; }

    stage_rc = seal_bin_size_probe_locked(bin_path, allow_sector_fallback, pm, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_open_data_file_locked(bin_path, budget_bytes, pm, err, seal_bin_data_file, n_slots, pool_bytes);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_alloc_slot_pool_locked(seal_bin_data_file, pool_bytes, err, seal_bin_pool_base);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_publish_layout_locked(touch_abort, pm, repack_manifest_sha, seal_bin_data_file, n_slots, pool_bytes,
                                              seal_bin_pool_base);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_build_registry_locked(pm, name_to_lp, pending_kept, seal_bin_data_file, n_slots, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_protect_tensors_locked(seal_bin_data_file, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_resolve_io_selector_locked(seal_bin_data_file, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_note_prefetch_locked(model, n_slots, seal_bin_data_file, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_open_metrics_locked(seal_bin_data_file, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_install_veh_locked(seal_bin_data_file, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_note_phasea_locked();
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_start_dispatcher_locked(seal_bin_data_file, seal_bin_pool_base, err);
    if (stage_rc != 0) { return stage_rc; }

    stage_rc = seal_bin_commit_locked(template_inventory_sha);
    if (stage_rc != 0) { return stage_rc; }

    return 0;
}
}  // namespace ggml_moe_direct_impl
namespace {  // the anonymous namespace resumes

// ===========================================================================
// E-2a. Lease-token handle table, view resolution, DRAINING state machine
//   contract section 1 (token internals, view 6 steps), section 2 row 7, section 6 E2-U1
// ===========================================================================

inline uint64_t xstore_handle_make(uint64_t gen, int32_t idx) {
    return (gen << 16) | (uint64_t)((uint32_t)idx + 1u);
}

// E-2b-0 (contract section 1 v0.5) - build the handle table to its FULL size, once. Called
// with g_mod.mutex held, from the epoch publish path, so the table exists before any handle
// can, and idempotent because a second seal must not hand out a second array underneath the
// first one's readers.
//
// The array is deliberately never freed: reset_for_test frees the ENTRIES, and the storage
// stays where it is for the life of the process. That is the clause, not a shortcut - "the
// table never moves" is exactly what lets the read path drop the module lock, and a table
// that came and went across resets would only be stable between them.
void xstore_table_construct_locked() {
    if (g_xs.table.load(std::memory_order_relaxed) != nullptr) return;
    g_xs.table_own.reset(new XStoreToken[(size_t)XSTORE_MAX_TOKENS]);
    XStoreToken * t = g_xs.table_own.get();
    // Link every entry into the free list, last index first so handing them out walks
    // upwards - the ordering has no contractual meaning, it just keeps a debug dump readable.
    g_xs.free_head = -1;
    for (int32_t i = XSTORE_MAX_TOKENS - 1; i >= 0; i--) {
        t[(size_t)i].set_next_free(g_xs.free_head);
        g_xs.free_head = i;
    }
    g_xs.free_count = XSTORE_MAX_TOKENS;
    g_xs.live_count = 0;
    g_xs.table_n.store(XSTORE_MAX_TOKENS, std::memory_order_relaxed);
    // Published LAST, with a release-store: a lock-free reader loading this pointer with an
    // acquire-load either sees nothing or sees a table whose entries are all constructed and
    // all marked free.
    g_xs.table.store(t, std::memory_order_release);
}

// E-2b-0 - NO lock required. Returns the table index for a live handle, or -1. -1 means
// "this handle names nothing live": zero, fabricated, out of range, already released, or a
// stale generation on a reused slot. All of those are the same class of caller error.
//
// One acquire-load of gen_live settles both liveness and identity: 0 is free, and any other
// value IS the generation of the current occupant, so a stale handle on a reused index fails
// the same comparison a released one does. The read is safe without the module lock because
// the table pointer never changes after publication and a live entry's payload never changes
// at all - it is a const object, and g0_tools/e2b0_static_gate.py proves nothing launders
// that away.
// E-2b-1 perf-repair adds `out_entry`. This function already holds the acquire-loaded
// table pointer it needs, and every caller followed it with xstore_entry(), which loads
// that same atomic a second time. Handing the entry back costs nothing here and removes
// one acquire-load from the view path. It is an optional out-parameter rather than a
// changed return type so the cold callers (release and the two test windows) keep their
// text: the duplicate load there sits under the module lock and is not on any timer.
// The value is written ONLY on success, so a caller that ignores the -1 cannot read a
// pointer this function never validated.
int32_t xstore_handle_lookup(uint64_t h, XStoreToken ** out_entry = nullptr) {
    if (h == 0) return -1;
    const uint64_t low = h & 0xFFFFull;
    if (low == 0) return -1;
    const uint64_t idx = low - 1;
    XStoreToken * t = g_xs.table.load(std::memory_order_acquire);
    if (t == nullptr) return -1;
    if (idx >= (uint64_t)(int64_t)g_xs.table_n.load(std::memory_order_relaxed)) return -1;
    const uint64_t gl = t[(size_t)idx].gen_live.load(std::memory_order_acquire);
    if (gl == 0) return -1;
    if (gl != (h >> 16)) return -1;
    if (out_entry) *out_entry = t + (size_t)idx;
    return (int32_t)idx;
}

// Convenience for the paths that already hold g_mod.mutex and want the entry itself. The
// table pointer is stable once published, so this is the same load either way.
inline XStoreToken * xstore_entry(int32_t idx) {
    return g_xs.table.load(std::memory_order_acquire) + (size_t)idx;
}

// Contract section 6 (E2-U1): a stale handle, a double release, and a use outside the
// token's epoch are boundary programming errors, and the contract answers them with a fatal
// rather than a return code - a quietly ignored bad handle is exactly the silent progress
// the lease rules forbid. This is the same class as the module's existing
// "invariant violated" aborts (pending pool exhausted, not enough claim victims).
[[noreturn]] void xstore_fatal_handle(const char * where, uint64_t h, const char * why) {
    fatal_abort("xstore: %s received an unusable lease handle 0x%016llx (%s)",
                where, (unsigned long long)h, why);
}

// Caller holds g_mod.mutex, and the acquire this token describes has ALREADY succeeded, so
// every requested record is published and leased. lease > 0 pins the slot (E-1 L-2), so the
// slot index and generation captured here cannot move underneath the token.
uint64_t xstore_token_issue_locked(int32_t mlpos, int32_t part, const int32_t * ids,
                                   int32_t n_ids) {
    if (g_xs.table.load(std::memory_order_relaxed) == nullptr) {
        // Unreachable through the boundary: an acquire needs a live epoch, and the publish
        // that mints one builds the table first. Saying so loudly beats a null dereference.
        fatal_abort("xstore: lease-token table was never built - a token is being issued "
                    "before any epoch was published");
    }
    if (g_xs.free_head < 0) {
        // Reachable only through a leak: this many outstanding batch tokens means releases
        // stopped happening. Refusing quietly would hide the leak behind an acquire failure.
        fatal_abort("xstore: lease-token table exhausted (%d live handles) - batch tokens "
                    "are not being released", (int)g_xs.live_count);
    }
    // Contract: handle = (gen << 16) | (index + 1), so the mint keeps only the low 48 bits of
    // the generation while the entry stores it whole. Past 2^48 every handle this function
    // returns would fail its own lookup at whichever consumer touches it first - loudly, but
    // away from the mint and with a message that names a stale handle instead of an exhausted
    // counter. Refuse at the mint instead, where the cause is. This is the same class as the
    // table exhaustion above: a resource the process cannot renew, and the module's answer to
    // that is fail-stop. The `== 0` half cannot be reached by this counter today - it starts
    // at 1 and only ever increments - and is here so the predicate states the whole encoding
    // contract rather than half of it.
    if (g_xs.gen_next > (UINT64_MAX >> 16) || g_xs.gen_next == 0) {
        fatal_abort("xstore: lease-token generation space exhausted (gen_next=%llu, "
                    "the handle encoding carries %d bits) - handles minted past this point "
                    "cannot be looked up",
                    (unsigned long long)g_xs.gen_next, 48);
    }
    const int32_t idx = g_xs.free_head;
    XStoreToken & tk  = *xstore_entry(idx);
    g_xs.free_head    = tk.next_free();
    g_xs.free_count--;

    // ---- payload, assembled in the module builder and COPIED in ----------------------
    // Everything below this point and above the publishing store is the "before publish"
    // half of the entry's lifetime (contract section 5, E-2b-0 (c)). The records are
    // assembled in g_xs.build_scratch and COPIED into the payload, so what lands in the
    // entry is a finished const object whose element buffer no one else has ever held a
    // pointer to; the static gate asserts that the gen_live store comes after every one of
    // these lines, and that the builder is named here and nowhere else.
    //
    // The builder is spelled out in full at every use on purpose. An `auto &` or a reference
    // alias would be easier to read and would also be one more name the gate would have to
    // chase before it could say where the buffer went.
    tk.set_next_free(-1);
    // The LIVE epoch, not the sequence counter: if the epoch was invalidated between the
    // acquire and this capture, the token must be born already outside its epoch rather
    // than inheriting the last published value.
    const uint64_t epoch = g_xs_epoch.load(std::memory_order_acquire);
    // Everything that can allocate sits inside this try. By the time it runs, the free-list
    // entry has been popped and the caller's lease is already granted, so an escaping
    // bad_alloc would leave a table slot and a lease unit owned by a token that was never
    // issued - a logical leak with no handle to release it, which is exactly what the
    // contract's failure atomicity forbids. The answer is fail-stop, not rollback: see
    // E2B0_ATOM_SPEC.md G-20 for why unwinding here would be a worse trade.
    try {
        g_xs.build_scratch.clear();
        g_xs.build_scratch.reserve((size_t)n_ids);
        // E-2b-3 perf-repair (R1). The de-duplication below decides, for every id, which
        // record it belongs to. It used to answer that with a linear scan of everything
        // assembled so far, which is n(n-1)/2 comparisons per acquire - 2,016 of them at
        // n_ids=64, on the ISSUE timer, and the growth term Level 1 measured.
        //
        // WHAT REPLACES IT, AND WHAT DOES NOT CHANGE. The scan is still here and it is still
        // the answer; what is added in front of it is a two-case fast path that is only ever
        // taken when it is EXACT. `ascending` means "every record in the list is strictly
        // greater than the one before it", which this loop maintains rather than assumes. While
        // that holds, the last record settles the question on its own: equal to it means this
        // id repeats the previous one, greater than it means no earlier record can match, and
        // anything else drops to the scan and clears the flag for the rest of the call. So the
        // OUTPUT is identical for every input - same records, same first-appearance order, same
        // multiplicities - and only the number of comparisons differs. The callers that make it
        // pay are the ones that already hand over ascending ids (E2B2_ATOM_SPEC.md section A1-R
        // reads the backend's moe_direct_ids as ascending; the Level 1 fixture is 0..n_used-1),
        // and a caller that does not is left exactly where it was: correct, and no slower than
        // the scan it would have run anyway.
        //
        // WHY NOT A TABLE, AND WHY THE COMPARISON IS AGAINST A LOCAL. An expert-ordinal keyed
        // table would answer in O(1) for every input rather than for ascending ones, and the
        // index this function already builds is that table - but it would have to be READ per
        // id, and a read of a builder element is a form the E-2b-0 gate's builder census does
        // not admit (g0_tools/e2b0_static_gate.py builder_subscript_use). Widening that policy
        // is a larger change to the proof than this repair is worth. `last_idx` needs none of
        // it: it is a plain local that mirrors the last APPENDED record's ordinal, which while
        // `ascending` holds is exactly `build_scratch[size-1].record_idx`, so the fast path
        // adds no builder occurrence at all - the two subscripts below are the ones that were
        // already there.
        bool    ascending = true;
        int64_t last_idx  = -1;   // record_idx of the most recently appended record
        for (int32_t i = 0; i < n_ids; i++) {
            const int64_t record_idx = (int64_t)mlpos * g_mod.n_expert + ids[i];
            size_t k = g_xs.build_scratch.size();
            bool fresh = (k == 0);
            if (!fresh && ascending) {
                if (record_idx == last_idx) {
                    k = k - 1;
                } else if (record_idx > last_idx) {
                    fresh = true;
                } else {
                    ascending = false;
                }
            }
            if (!fresh && !ascending) {
                for (k = 0; k < g_xs.build_scratch.size(); k++)
                    if (g_xs.build_scratch[k].record_idx == record_idx) break;
                fresh = (k == g_xs.build_scratch.size());
            }
            if (fresh) {
                last_idx = record_idx;
                XStoreRecordRef r;
                r.record_idx = record_idx;
                const RecordEntry & re = g_mod.record_table[(size_t)record_idx];
                r.slot_index      = re.slot_index.load(std::memory_order_acquire);
                r.slot_generation = re.slot_generation.load(std::memory_order_relaxed);
                g_xs.build_scratch.push_back(r);
            }
            g_xs.build_scratch[k].multiplicity++;
        }
        // E-2b-1 perf-repair: the direct capture index, built from the records just
        // assembled and from nothing else. `record_idx - base` is the expert ordinal the
        // view will recompute from its own argument, so the table is keyed by exactly
        // the value the view looks up. Empty = the thresholds said scan; the view then
        // behaves as it did before this change.
        g_xs.build_index_scratch.clear();
        const size_t n_rec = g_xs.build_scratch.size();
        if (g_mod.n_expert > 0 && g_mod.n_expert <= XSTORE_VIEW_DENSE_MAX_EXPERT &&
            n_rec >= XSTORE_VIEW_DENSE_MIN_RECORDS &&
            n_rec < (size_t)XSTORE_VIEW_NO_REF) {
            g_xs.build_index_scratch.assign((size_t)g_mod.n_expert, XSTORE_VIEW_NO_REF);
            const int64_t base = (int64_t)mlpos * g_mod.n_expert;
            for (size_t k = 0; k < n_rec; k++) {
                const int64_t e = g_xs.build_scratch[k].record_idx - base;
                // Defensive, not expected: the ids the acquire accepted are already
                // range-checked, so an out-of-range ordinal here would mean the record
                // list disagrees with the layer it was built for. Dropping the index
                // entirely is the safe answer - the view falls back to the scan, which
                // is the pre-change behaviour and validates membership itself.
                if (e < 0 || e >= (int64_t)g_xs.build_index_scratch.size()) {
                    g_xs.build_index_scratch.clear();
                    break;
                }
                g_xs.build_index_scratch[(size_t)e] = (uint16_t)k;
            }
        }
        tk.construct_payload(epoch, mlpos, part, n_ids, g_xs.build_scratch,
                             g_xs.build_index_scratch);
    } catch (const std::bad_alloc &) {
        fatal_abort("xstore: allocation failure while issuing a lease token - fail-stop; "
                    "no partial state survives because the process does not");
    }
    // ---- publish --------------------------------------------------------------------
    // Release-store: a reader that acquire-loads a non-zero generation here is guaranteed to
    // see every payload write above it. From this instant until the matching free the entry
    // is immutable, and that is the whole basis of the lock-free read path.
    const uint64_t gen = g_xs.gen_next++;
    tk.gen_live.store(gen, std::memory_order_release);
    g_xs.live_count++;
    g_xs_handles.fetch_add(1, std::memory_order_relaxed);
    return xstore_handle_make(gen, idx);
}

// Test-only writer, and the fifth member of the xstore writer roster the static gate
// freezes. It writes ONE value - the last generation the handle encoding can carry - and the
// caller cannot choose it. An arbitrary `set(uint64_t)` would widen the roster by a whole
// counter instead of by a single boundary point, which is the difference between "a test can
// reach the boundary" and "a test can put the counter anywhere".
//
// Three premises, split by what can be proved where. The module lock is STATIC: the gate pins
// this writer's one call site and requires a g_mod.mutex lock_guard to dominate the call, so
// the wrapper below is the only way in and it takes the lock first. The other two depend on
// run-time state and are checked here, fatally, in the module's own idiom.
void xstore_arm_last_generation_locked() {
    if (g_xs.table.load(std::memory_order_relaxed) == nullptr) {
        fatal_abort("xstore: the last generation was armed before any epoch was published - "
                    "there is no handle table for the arm to be about");
    }
    if (g_xs.live_count != 0) {
        fatal_abort("xstore: the last generation was armed with %d live handle(s) - the jump "
                    "is only defined on a quiescent table, because a handle minted before it "
                    "carries a generation this arm does not move",
                    (int)g_xs.live_count);
    }
    g_xs.gen_next = (UINT64_MAX >> 16);
}

// Caller holds g_mod.mutex, and the contract's quiescence clause says every view of this
// token has already returned. The generation is not rolled back: the handle that named this
// entry is dead because gen_live is 0, and the next handle for this index carries a strictly
// larger generation, so the two can never alias.
void xstore_token_free_locked(int32_t idx) {
    XStoreToken & tk = *xstore_entry(idx);
    // Retired FIRST, with a release-store. Everything after this line is the "after
    // quiescence" half of the entry's lifetime: by the time it runs the entry is no longer
    // live, so no reader can be inside the payload this ends.
    tk.gen_live.store(0, std::memory_order_release);
    tk.destroy_payload();
    tk.set_next_free(g_xs.free_head);
    g_xs.free_head = idx;
    g_xs.free_count++;
    g_xs.live_count--;
    g_xs_handles.fetch_sub(1, std::memory_order_relaxed);
}

// Caller holds g_mod.mutex. The FULL capture comparison the contract puts on release
// (section 1, token internals: "release compares the captured values per record, a mismatch
// is fatal"). Rebuilding the id list from the capture and handing it to the lease machinery
// without comparing anything is NOT the same thing: after a mixed-surface release the slot
// this record sat in can be evicted and re-owned, and the rebuilt list would then either
// decrement a lease that now belongs to whatever moved in, or find slot_index < 0 and skip
// the record entirely - a quiet no-op return. Both are the silent mis-pairing the token was
// introduced to remove, so both are fatal here.
//
// Reachability is not hypothetical: the store_view step-5 arm reaches exactly this state
// through the ordinary two-surface window E-2b opens while consumers migrate one at a time.
// Every record is compared and the handle is not freed until all of them pass, so there is
// no partial release.
void xstore_verify_capture_or_fatal(const char * where, uint64_t h, int32_t mlpos,
                                    const XStoreRecordRef & r) {
    const int64_t base = (int64_t)mlpos * g_mod.n_expert;
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.layers.size() ||
        r.record_idx < base || r.record_idx >= base + g_mod.n_expert ||
        r.record_idx < 0 || r.record_idx >= (int64_t)g_mod.record_table.size()) {
        fatal_abort("xstore: %s handle 0x%016llx names record %lld, which is not a record of "
                    "layer position %d - the token capture is not self-consistent",
                    where, (unsigned long long)h, (long long)r.record_idx, (int)mlpos);
    }
    const RecordEntry & re = g_mod.record_table[(size_t)r.record_idx];
    const int32_t  si = re.slot_index.load(std::memory_order_acquire);
    const uint64_t rg = re.slot_generation.load(std::memory_order_relaxed);
    bool     ok       = (si >= 0 && si < g_mod.n_slots && si == r.slot_index &&
                         rg == r.slot_generation);
    int64_t  now_own  = -1;
    uint64_t now_gen  = 0;
    if (si >= 0 && si < g_mod.n_slots) {
        const Slot & s = g_mod.slots[(size_t)si];
        now_own = s.owner_record;
        now_gen = s.generation.load(std::memory_order_acquire);
        if (ok) ok = (now_own == r.record_idx) && (now_gen == r.slot_generation);
    }
    if (!ok) {
        fatal_abort("xstore: %s handle 0x%016llx no longer matches what it captured for "
                    "record %lld - captured slot=%d generation=%llu, the record now reads "
                    "slot=%d generation=%llu and that slot holds owner=%lld generation=%llu. "
                    "The lease this token names is gone, so returning it would decrement a "
                    "lease that belongs to someone else",
                    where, (unsigned long long)h, (long long)r.record_idx,
                    (int)r.slot_index, (unsigned long long)r.slot_generation,
                    (int)si, (unsigned long long)rg,
                    (long long)now_own, (unsigned long long)now_gen);
    }
}

// Caller holds g_mod.mutex. Sum of Slot::lease - the third value of the 3-way check.
int64_t xstore_slot_lease_sum_locked() {
    int64_t sum = 0;
    for (int32_t i = 0; i < g_mod.n_slots; i++) {
        sum += (int64_t)g_mod.slots[(size_t)i].lease.load(std::memory_order_relaxed);
    }
    return sum;
}

// Contract section 2 row 7, as corrected by r3: what the timeout prints is a LEASE
// inventory, not a token list. The old surface mints no token, so a legacy-only residue
// would print an empty token list and say nothing at all about what is still held. Both
// sides are enumerated here - old-surface rows come from the slots, new-surface rows add
// the handle and the captured multiplicity on top.
// Caller holds g_mod.mutex.
void xstore_log_residual_inventory_locked(int32_t deadline_ms) {
    int32_t leased_slots = 0;
    for (int32_t i = 0; i < g_mod.n_slots; i++) {
        if (g_mod.slots[(size_t)i].lease.load(std::memory_order_relaxed) != 0) leased_slots++;
    }
    log_err("[moe-direct][xstore] drain deadline %d ms expired with leases outstanding - "
            "residual lease inventory follows; no cleanup is attempted",
            (int)deadline_ms);
    log_err("[moe-direct][xstore] residual counts: handles=%lld lease_units=%lld "
            "sum_slot_lease=%lld leased_slots=%d",
            (long long)g_xs_handles.load(std::memory_order_relaxed),
            (long long)xstore_outstanding_units(),
            (long long)xstore_slot_lease_sum_locked(), (int)leased_slots);
    for (int32_t i = 0; i < g_mod.n_slots; i++) {
        const Slot & s = g_mod.slots[(size_t)i];
        const int32_t l = s.lease.load(std::memory_order_relaxed);
        if (l == 0) continue;
        log_err("[moe-direct][xstore] residual slot: slot=%d owner_record=%lld "
                "generation=%llu lease=%d",
                (int)i, (long long)s.owner_record,
                (unsigned long long)s.generation.load(std::memory_order_relaxed), (int)l);
    }
    int32_t live_tokens = 0;
    const XStoreToken * tbl = g_xs.table.load(std::memory_order_acquire);
    const int32_t       tn  = g_xs.table_n.load(std::memory_order_relaxed);
    for (int32_t i = 0; tbl != nullptr && i < tn; i++) {
        const XStoreToken & tk = tbl[(size_t)i];
        const uint64_t gl = tk.gen_live.load(std::memory_order_acquire);
        if (gl == 0) continue;
        live_tokens++;
        for (size_t r = 0; r < tk.records().size(); r++) {
            log_err("[moe-direct][xstore] residual token: token=0x%016llx mlpos=%d part=%d "
                    "record=%lld slot=%d generation=%llu multiplicity=%d",
                    (unsigned long long)xstore_handle_make(gl, i),
                    (int)tk.mlpos(), (int)tk.part(),
                    (long long)tk.records()[r].record_idx,
                    (int)tk.records()[r].slot_index,
                    (unsigned long long)tk.records()[r].slot_generation,
                    (int)tk.records()[r].multiplicity);
        }
    }
    log_err("[moe-direct][xstore] residual tokens=%d (0 together with a non-empty slot list "
            "above means the residue is legacy-only - the old surface mints no token)",
            (int)live_tokens);
}

// Caller holds g_mod.mutex. 0 = resolved, <0 = refuse. In adapter mode this IS the registry
// lookup the old body did, at the same point inside the same lock.
int xstore_resolve_key_locked(const ggml_tensor * t, bool by_tensor,
                              int32_t mlpos_in, int32_t part_in, XStoreAcquireKey & out) {
    if (by_tensor) {
        auto it = g_mod.registry.find(t);
        if (it == g_mod.registry.end()) return -1;
        out.mlpos    = it->second.mlpos;
        out.part_idx = it->second.part_idx;
        return 0;
    }
    if (mlpos_in < 0 || mlpos_in >= (int32_t)g_mod.layers.size()) return -1;
    if (part_in  < 0 || part_in  >= (int32_t)g_mod.layers[(size_t)mlpos_in].parts.size()) return -1;
    out.mlpos    = mlpos_in;
    out.part_idx = part_in;
    return 0;
}

// ===========================================================================
// E-2a. The common internal lease primitive - acquire.
//
// BOTH boundary surfaces call this DIRECTLY (contract section 1, coexistence rule). The old
// tensor* API does not route through the token API and never sees a token, so the
// "token with nowhere to return it" problem the r2 review named cannot arise here.
//
// The body below is the former ggml_moe_direct_acquire body, moved rather than rewritten.
// The single seam is the resolution step: where it called registry.find(t) it now calls
// xstore_resolve_key_locked(), which in adapter mode (by_tensor) IS that same lookup at the
// same point inside the same lock. So the sequence of operations a legacy consumer observes
// on the RUNNING path is what it was before this atom.
//   by_tensor = true   adapter mode: t resolves to (mlpos, part_idx)
//   by_tensor = false  logical-key mode: (mlpos_in, part_in) already ARE the key
//   out_key            filled on success only; the old surface passes nullptr
// ===========================================================================

// R-B5a facade (iv): command request / result lifecycle (RB_DESIGN.md:1221 section
// 2-6 (iv)). The caller states the command it wants captured; `g_mod.cmds[ci]` is
// filled here and nowhere outside the dispatch cluster. The statements below are the
// ones this member replaced, in the same order, with the caller's lock unchanged.
inline void dfac_cmd_demand_fill_locked(int32_t ci, int32_t mlpos, int32_t part_idx,
                                        int32_t n_ids, Phase phase,
                                        const int32_t * expert_ids, uint64_t pa_t_enter) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.kind     = CMD_DEMAND_ACQUIRE;
    c.mlpos    = mlpos;
    c.part_idx = part_idx;
    c.n_ids    = n_ids;
    c.phase    = phase;
    c.ids.assign(expert_ids, expert_ids + n_ids); // 입력 복사(호출자 버퍼 수명 비의존)
    c.pa_t_enter = pa_t_enter;
    // ★B2 epoch 투영(§2.1) — 사건의 **보존 epoch**을 여기서 채취해 op/행에 그대로 전파한다
    //   (요청 경계 뒤에 도착하는 terminal/처분이 다음 요청으로 새는 오귀속 차단 — §0.1).
    ggml_moe_phaseb_current_epoch(&c.pb_request_epoch, &c.pb_graph_epoch, &c.pb_graph_seq, nullptr);
}

int xstore_acquire_common(const ggml_tensor * t, bool by_tensor,
                          int32_t mlpos_in, int32_t part_in,
                          const int32_t * expert_ids, int32_t n_ids, int32_t n_tokens_hint,
                          XStoreAcquireKey * out_key) {
    if (!expert_ids || n_ids <= 0) return -1;
    if (by_tensor && !t) return -1;
    // DRAINING and STOPPED refuse every new acquire, on BOTH surfaces (contract section 2
    // row 7). The gate is an ADMISSION, not a bare state read: a bare read leaves the window
    // where this call sees RUNNING, a concurrent drain then confirms "0 lease units
    // outstanding" - true at that instant, because this acquire has not reached the lease
    // yet - stops the dispatcher and invalidates the epoch, and the lease lands afterwards
    // in a store that is already STOPPED, holding a token that can never be returned. The
    // admission closes it in both directions: taken before the state flipped, the drain sees
    // it and keeps waiting; attempted after, it reads DRAINING here and backs out.
    // Cost on the RUNNING path is two interlocked ops on a counter, inside a call that
    // already takes g_mod.mutex.
    XStoreAdmission admit;
    if (!admit.admitted) return -1;

    // ★Phase A D_j — "첫 canonical acquire 진입"(락 대기 이전 시각. 상태 변경 없음).
    const bool     pa_on      = g_mod.phasea_on;
    const uint64_t pa_t_enter = pa_on ? ggml_moe_phasea_now_qpc() : 0;

    // ★lead-time T1(PROBE_LEADTIME_SPEC_R1 §1) — acquire 진입 **pre-lock** 시각.
    //   PA D_j 와 같은 자리이지만 게이트가 완전히 독립이다: PA 는 timing 모드에서도 `phasea_on`
    //   을 세우고 `disp_can_plan_demand`(:6489-6492)가 그 플래그로 계획을 지연시킨다 — 관측이
    //   거동을 바꾼다. 이 프로브는 그 부작용을 피해야 하므로 PA 를 쓰지 않는다(사양 §2 CONFIRMED).
    //   락 대기·registry 조회 **이전** 값이어야 T1 이 "요구가 도착한 시각"이 된다.
    const bool     lt_on       = ggml_moe_leadtime_enabled();
    const uint64_t lt_t1_qpc   = lt_on ? ggml_moe_leadtime_now_qpc() : 0;

    uint64_t trace_lock_episode_id = 0;
    if (ggml_moe_trace_enabled()) {
        uint16_t ctx_run_index; uint32_t ctx_graph_seq; uint8_t ctx_phase;
        ggml_moe_trace_current_graph_context(&ctx_run_index, &ctx_graph_seq, &ctx_phase);
        trace_lock_episode_id = ggml_moe_trace_lock_wait_begin(ctx_run_index, ctx_graph_seq, ctx_phase);
    }

    // ══ ★B1 영속 경로(PHASEB_DESIGN §2 핸드셰이크) ══════════════════════════════
    // ① façade: null/범위/registry/lifecycle 선검증 + 입력 복사(8단계 ① 잔류)
    // ② DemandAcquire enqueue 후 per-command CV 대기(★구간 중 g_mod.mutex 무보유)
    // ③ success signal 조건 = 전 required record publish + 성공 terminal 경로의 lease 반영 완료
    //    (C2 — 반환 직후 ptr() 전건 유효). ⑤ STOPPING 후 신규 command 거절.
    if (persistent_active()) {
        int32_t mlpos = -1, part_idx = -1;
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            if (ggml_moe_trace_enabled()) ggml_moe_trace_lock_acquired(trace_lock_episode_id);
            if (ggml_moe_trace_enabled()) ggml_moe_trace_note_acquire_call(n_ids);
            if (!g_mod.sealed) return -1;
            XStoreAcquireKey k;   // E-2a seam - see the header comment
            if (xstore_resolve_key_locked(t, by_tensor, mlpos_in, part_in, k) != 0) return -1;
            mlpos    = k.mlpos;
            part_idx = k.part_idx;
            for (int32_t i = 0; i < n_ids; i++) {
                if (expert_ids[i] < 0 || expert_ids[i] >= g_mod.n_expert) return -1;
            }
        }
        const auto t0p = std::chrono::steady_clock::now();
        const int32_t ci = cq_alloc();
        if (ci < 0) return -1; // STOPPING — 신규 command 거절(fail-close)
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        dfac_cmd_demand_fill_locked(ci, mlpos, part_idx, n_ids,
                                    (n_tokens_hint > 1) ? Phase::PREFILL : Phase::DECODE,
                                    expert_ids, pa_t_enter);
        const int rc = cq_submit_and_wait(ci);
        const Phase ph = c.phase;
        cq_free(ci);
        // ★lead-time stamp B(사양 §2) — **`cq_submit_and_wait` 성공 뒤**(교차 r2 R3-3 폐합).
        //   좌표 이력: r1 판=`cq_alloc` 앞 → r1 SHOULD-1 로 `cq_alloc` 뒤 → **r2 R3-3 로 여기**.
        //   남아 있던 창: `cq_alloc` 은 성공했는데 그 직후 CUT 이 닫히면 `cq_submit_and_wait`(:4352)
        //   가 `q_stopping` 을 보고 command 를 **실행 없이 -1 로 거절**한다(:4355). 그 사이에 행을
        //   남기면 실행되지 않은 층의 유령 프레임이 되어 판독기 cardinality(프레임 수 = 층 실행 수)를
        //   거짓으로 깨뜨린다. 이제 **수용·실행이 확정된 뒤에만**(rc >= 0) 남긴다.
        //   ★T1 의 **값**은 여전히 진입부 pre-lock QPC(`lt_t1_qpc`)다 — 이동한 것은 append 위치뿐
        //   이므로 측정 의미는 불변이고(대기 시간이 T1 에 섞이지 않는다), 순차 acquire 의 pre-lock
        //   QPC 는 단조라 스트림 순서도 그대로다.
        //   ★rc < 0 이 유일하게 나오는 프로덕션 경로는 이 CUT 거절이고, 호출부는 그 값을 받으면
        //   `GGML_ABORT` 한다(ggml-cpu.c:1695 · ggml-backend.cpp 등록 텐서 경로) — 즉 rc<0 뒤에는
        //   더 이상 프레임이 생기지 않으므로 이 게이팅이 층을 누락시키지 않는다.
        //   ★B 스트림 계약(§2 join ① + 정오 1): persistent 경로에서 acquire 는 hit 에도 발화하므로
        //   이 행은 모든 층 실행에 존재하고, 같은 mlpos 의 연속 3행(up/gate/down)이 판독기의
        //   "완전집합 프레임"이 된다.
        if (lt_on && rc >= 0) {
            ggml_moe_leadtime_stamp_b(lt_t1_qpc, mlpos, part_idx, n_ids,
                                      (n_tokens_hint > 1) ? 1 : 0);
        }
        const auto t1p = std::chrono::steady_clock::now();
        g_mod.phase[(int)ph].ensure_wait_ms_sum.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(t1p - t0p).count(),
            std::memory_order_relaxed);
        if (rc == 0 && out_key) { out_key->mlpos = mlpos; out_key->part_idx = part_idx; }
        return rc;
    }
    // ★종료(STOPPING/정지) 후 신규 command 거절 — legacy pump 우회 금지(fail-close).
    if (persistent_fail_closed()) return -1;

    // ══ legacy 경로(qualification/test mode 전용 대조군 — 배치 pump 인라인) ═════════
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (ggml_moe_trace_enabled()) ggml_moe_trace_lock_acquired(trace_lock_episode_id);
    if (ggml_moe_trace_enabled()) ggml_moe_trace_note_acquire_call(n_ids);

    if (!g_mod.sealed) return -1;
    XStoreAcquireKey k;   // E-2a seam - see the header comment
    if (xstore_resolve_key_locked(t, by_tensor, mlpos_in, part_in, k) != 0) return -1;
    const int32_t mlpos = k.mlpos;
    const Phase phase = (n_tokens_hint > 1) ? Phase::PREFILL : Phase::DECODE;

    // ★QD 레버(비동기 다중 outstanding — reviews/codex_qd_u6_design_xcheck.md 심의 3):
    // 아래 ①~⑧ 순서 자체가 계약이다. 구 구현은 id별로 ensure_locked를 순차 호출해 miss마다 동기
    // 완주했지만, 비동기에서는 (a) publish 전이라 같은 record의 둘째 요청이 hit로 흡수되지 않고
    // (b) 앞 miss의 victim 선정이 같은 배치 뒤 id가 hit할 슬롯을 축출할 수 있으므로, 판별(resolve)
    // ·수용검사(preflight)·점유(claim)·제출(submit)·회수(retire)를 분리해야 한다.
    // ★QD=1은 이 신 플래너의 concurrency-off 대조군이지 구 구현과 캐시 완전 동등이 아니다
    // (resolve-all이 뒤쪽 hit 슬롯을 선보호하므로 구 구현에 있던 추가 축출·재읽기가 사라질 수 있다
    //  — 성능 A/B는 frozen anchor 외에 같은 바이너리의 QD1↔QD8 비교로 본다).
    // 전 구간 g_mod.mutex 보유(D-1) — OVERLAPPED 완료는 이 mutex를 필요로 하지 않는다.

    // ★8단계 플래너 ①~⑤는 plan_demand_locked(§8)로 추출돼 영속 dispatcher와 **같은 코드**를
    //   쓴다(§7 "legacy↔persistent 8단계 동형"의 구조적 근거). 스크래치는 모듈 멤버(disp_plan
    //   과는 별개 인스턴스 — legacy는 g_mod.mutex 전 구간 보유라 정적 재사용이 안전하다).
    static DemandPlan s_plan;

    auto t0 = std::chrono::steady_clock::now();

    // ── ①선검증 ②resolve ③preflight ④tick+hit lease ⑤claim ──
    if (plan_demand_locked(s_plan, mlpos, k.part_idx, expert_ids, n_ids, phase,
                           pa_on, pa_t_enter, /*persistent=*/false) != 0) {
        return -1;
    }
    const LayerInfo & L = g_mod.layers[(size_t)mlpos];

    // ★R2 #4 — legacy 팔도 pb_sample 행/3분할을 낸다(게이트 9-b: PA ledger canonical ACQ_CALL 과
    //   1:1). 제출 전에 행을 세워야 pb_io 의 ISSUED 결속과 순서가 persistent 와 동형이다.
    pb_legacy_demand_row_locked(mlpos, phase, s_plan, n_ids);

    // ── ⑥⑦ submit(QD 상한까지) / wait·retire + publish. ⑧ 실패 의미론 = fail-stop(내부 abort). ──
    run_batch_ios_locked(s_plan.miss, mlpos, phase);

    // ── ★Phase A R_j — demand record 전부 host 가용(배치 pump 종료 직후). ──
    if (pa_on) {
        ggml_moe_phasea_acquire_exit((int32_t)s_plan.miss.size(),
                                     (uint64_t)s_plan.miss.size() * L.stride_bytes,
                                     ggml_moe_phasea_now_qpc());
    }

    auto t1 = std::chrono::steady_clock::now();
    uint64_t wait_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    g_mod.phase[(int)phase].ensure_wait_ms_sum.fetch_add(wait_ms, std::memory_order_relaxed);

    metrics_maybe_periodic_locked();
    if (out_key) { out_key->mlpos = mlpos; out_key->part_idx = k.part_idx; }
    return 0;
}

// ===========================================================================
// E-2a. The common internal lease primitive - release. Same seam, same rule: the old
// surface calls this directly with a tensor, the token surface calls it directly with a
// logical key. DRAINING does NOT block a release - draining is precisely the state in which
// releases must still land (contract section 2 row 7).
// ===========================================================================

// R-B5a facade (iv): command request / result lifecycle (RB_DESIGN.md:1221 section
// 2-6 (iv)). The caller states the command it wants captured; `g_mod.cmds[ci]` is
// filled here and nowhere outside the dispatch cluster. The statements below are the
// ones this member replaced, in the same order, with the caller's lock unchanged.
inline void dfac_cmd_release_fill_locked(int32_t ci, int32_t mlpos, int32_t part_idx,
                                         int32_t n_ids, const int32_t * expert_ids) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.kind     = CMD_RELEASE;
    c.mlpos    = mlpos;
    c.part_idx = part_idx;
    c.n_ids    = n_ids;
    c.ids.assign(expert_ids, expert_ids + n_ids);
}

void xstore_release_common(const ggml_tensor * t, bool by_tensor,
                           int32_t mlpos_in, int32_t part_in,
                           const int32_t * expert_ids, int32_t n_ids) {
    if (!expert_ids || n_ids <= 0) return;
    if (by_tensor && !t) return;

    // ★B1 — release도 동기 command + ack(반환 전 lease 감소 완결 — §2 핸드셰이크 ④).
    if (persistent_active()) {
        int32_t mlpos = -1, part_idx = -1;
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            if (!g_mod.sealed) return;
            XStoreAcquireKey k;   // E-2a seam
            if (xstore_resolve_key_locked(t, by_tensor, mlpos_in, part_in, k) != 0) return;
            mlpos    = k.mlpos;
            part_idx = k.part_idx;
        }
        const int32_t ci = cq_alloc();
        if (ci < 0) return; // STOPPING — 신규 command 거절
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        dfac_cmd_release_fill_locked(ci, mlpos, part_idx, n_ids, expert_ids);
        cq_submit_and_wait(ci);
        cq_free(ci);
        return;
    }
    if (persistent_fail_closed()) return; // ★종료 후 신규 command 거절(fail-close)

    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.sealed) return;
    XStoreAcquireKey k;   // E-2a seam
    if (xstore_resolve_key_locked(t, by_tensor, mlpos_in, part_in, k) != 0) return;
    int32_t mlpos = k.mlpos;
    const bool pa_on = g_mod.phasea_on;
    if (pa_on) {
        ggml_moe_phasea_release_enter(mlpos, k.part_idx, n_ids, ggml_moe_phasea_now_qpc());
    }
    for (int32_t i = 0; i < n_ids; i++) {
        int32_t eid = expert_ids[i];
        if (eid < 0 || eid >= g_mod.n_expert) continue;
        int64_t record_idx = (int64_t)mlpos * g_mod.n_expert + eid;
        RecordEntry & re = g_mod.record_table[(size_t)record_idx];
        int32_t si = re.slot_index.load(std::memory_order_relaxed);
        if (si < 0) continue;
        int32_t prev = g_mod.slots[(size_t)si].lease.fetch_sub(1, std::memory_order_relaxed);
        if (prev <= 0) g_mod.slots[(size_t)si].lease.fetch_add(1, std::memory_order_relaxed);
        else           xstore_note_lease_returned(1);   // E-2a lease-unit ledger (b)
        // ★Phase A ledger — lease 전후값(underflow 복구 시 after == before).
        if (pa_on) ggml_moe_phasea_release_expert(eid, si, prev, prev > 0 ? prev - 1 : prev);
    }
}



}
// R-B2 (RB2_ATOM_SPEC.json declare_in_place): the module-state globals whose
// DEFINITION stays in this unit.  They leave the anonymous namespace so that
// ggml-moe-direct-state.h can declare them `extern`; their definition text is
// unchanged, which is what the forbidden line and e2b0's GMOD_DEF pin require.
namespace ggml_moe_direct_impl {
ModuleState g_mod;
std::atomic<int32_t> g_v3_stage{ V3_STAGE_NONE };
V3ResourceProbe g_v3_probe;

// R-B4b (RB4B_ATOM_SPEC.json :: seam_map.members `b3-io-param-snapshot`): 관측 TU 가
//   `g_mod.v3_io_param_val`·`g_mod.v3_io_param_src`(스케줄링 K9)를 직접 읽지 않도록,
//   core 가 두 배열을 **값으로** 복사해 넘긴다 — g_mod 로의 포인터가 TU 경계를 넘지
//   않는다. 호출자는 `g_mod.mutex` 를 이미 쥐고 있다(`_locked` 규약).
void v3_io_param_snapshot_locked(int32_t (&val)[6], uint8_t (&src)[6]) {
    for (int i = 0; i < 6; i++) {
        val[i] = g_mod.v3_io_param_val[i];
        src[i] = g_mod.v3_io_param_src[i];
    }
}
} // namespace ggml_moe_direct_impl
 // anonymous namespace

// ===========================================================================
// §12. 공개 API 구현
// ===========================================================================

extern "C" {

bool ggml_moe_direct_enabled(void) {
    std::call_once(g_mod.enabled_once, []() { g_mod.enabled_cached = getenv_flag_is_1("MOE_DIRECT"); });
    return g_mod.enabled_cached;
}

bool ggml_moe_no_prefetch(void) {
    std::call_once(g_mod.no_prefetch_once, []() { g_mod.no_prefetch_cached = getenv_flag_is_1("MOE_NO_PREFETCH"); });
    return g_mod.no_prefetch_cached;
}

int ggml_moe_direct_register(const struct ggml_tensor * t, const char * source_tensor_name, int32_t is_host) {
    if (!t || !source_tensor_name) return -1;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (g_mod.sealed) return -1;
    std::string nm = source_tensor_name;
    // 동일 텐서 재등록 멱등(이름·host 일치해야).
    for (const auto & pr : g_mod.pending) {
        if (pr.tensor == t) {
            if (pr.name != nm || pr.is_host != (is_host != 0)) return -1;
            return 0;
        }
    }
    g_mod.pending.push_back(PendingReg{ t, nm, is_host != 0 });
    return 0;
}

int ggml_moe_direct_seal(const struct ggml_moe_direct_model * model,
                         const struct ggml_moe_direct_source * sources, int32_t n_sources) {
    if (!model || !sources || n_sources <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mod.mutex);

    std::string dir = getenv_str("MOE_DIRECT_DIR");
    if (dir.empty()) dir = "D:\\moe-models\\gpt-oss-120b\\repack-v2";
    std::wstring dir_w = utf8_to_wide(dir);

    uint32_t budget_mb = PROD_DEFAULT_BUDGET_MB;
    {
        std::string s = getenv_str("MOE_DIRECT_BUDGET_MB");
        if (!s.empty()) {
            char * endp = nullptr;
            long v = strtol(s.c_str(), &endp, 10);
            if (endp && *endp == '\0' && v > 0) budget_mb = (uint32_t)v;
        }
    }
    uint32_t touch_abort = PROD_DEFAULT_TOUCH_ABORT;
    {
        std::string s = getenv_str("MOE_DIRECT_TOUCH_ABORT");
        if (!s.empty()) {
            char * endp = nullptr;
            long v = strtol(s.c_str(), &endp, 10);
            if (endp && *endp == '\0' && v > 0) touch_abort = (uint32_t)v;
        }
    }
    uint64_t budget_bytes = (uint64_t)budget_mb * 1024ULL * 1024ULL;

    // ★F1: MOE_DIRECT_EXPECTS_DIR 필수(디폴트 없음). 프로덕션은 frozen EXPECT_CATALOG 사용.
    std::string expects_dir = getenv_str("MOE_DIRECT_EXPECTS_DIR");
    std::wstring expects_dir_w = utf8_to_wide(expects_dir);

    std::string err;
    int rc = seal_common_locked_v2(dir_w, *model, sources, n_sources,
                                   /*allow_sector_fallback=*/false, budget_bytes, touch_abort,
                                   expects_dir_w, /*override_profile_id=*/nullptr, /*override_digest=*/nullptr,
                                   /*override_scope=*/nullptr, err);
    if (rc != 0) log_err("ggml_moe_direct_seal failed: %s", err.c_str());
    return rc;
}

void ggml_moe_direct_attestation(int32_t * out_all, int32_t * out_host, int32_t * out_nonhost,
                                 int32_t * out_n_slots, int32_t * out_n_slots_required) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_all) *out_all = g_mod.att_all;
    if (out_host) *out_host = g_mod.att_host;
    if (out_nonhost) *out_nonhost = g_mod.att_nonhost;
    if (out_n_slots) *out_n_slots = g_mod.att_n_slots;
    if (out_n_slots_required) *out_n_slots_required = g_mod.att_n_slots_required;
}

void ggml_moe_direct_scope_attestation(int32_t * out_raw, int32_t * out_managed, int32_t * out_excluded_nextn) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_raw) *out_raw = g_mod.att_raw;
    if (out_managed) *out_managed = g_mod.att_managed;
    if (out_excluded_nextn) *out_excluded_nextn = g_mod.att_excluded_nextn;
}

bool ggml_moe_direct_is_registered(const struct ggml_tensor * t) {
    if (!t) return false;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    return g_mod.registry.find(t) != g_mod.registry.end();
}

int ggml_moe_direct_acquire(const struct ggml_tensor * t, const int32_t * expert_ids, int32_t n_ids, int32_t n_tokens_hint) {
    // E-2a: the body moved to xstore_acquire_common() so the logical-key surface can call
    // the SAME primitive directly rather than through this one. Adapter mode below.
    if (!t) return -1;
    return xstore_acquire_common(t, /*by_tensor=*/true, /*mlpos_in=*/-1, /*part_in=*/-1,
                                 expert_ids, n_ids, n_tokens_hint, /*out_key=*/nullptr);
}

void ggml_moe_direct_release(const struct ggml_tensor * t, const int32_t * expert_ids, int32_t n_ids) {
    if (!t) return;
    xstore_release_common(t, /*by_tensor=*/true, /*mlpos_in=*/-1, /*part_in=*/-1,
                          expert_ids, n_ids);
}

const void * ggml_moe_direct_ptr(const struct ggml_tensor * t, int32_t expert_id) {
    if (!t) return nullptr;
    if (!g_mod.sealed.load(std::memory_order_acquire)) return nullptr;
    auto it = g_mod.registry.find(t);
    if (it == g_mod.registry.end()) return nullptr;
    int32_t mlpos = it->second.mlpos;
    int32_t part_idx = it->second.part_idx;
    if (expert_id < 0 || expert_id >= g_mod.n_expert) return nullptr;

    int64_t record_idx = (int64_t)mlpos * g_mod.n_expert + expert_id;
    const RecordEntry & re = g_mod.record_table[(size_t)record_idx];
    int32_t si = re.slot_index.load(std::memory_order_acquire);
    if (si < 0) return nullptr;
    uint64_t expect_gen = re.slot_generation.load(std::memory_order_relaxed);
    const Slot & s = g_mod.slots[(size_t)si];
    uint64_t actual_gen = s.generation.load(std::memory_order_acquire);
    if (actual_gen != expect_gen) return nullptr;

    // ★§Z-⑥ A2 — 슬롯 내 실데이터 시작(§3-3). bin=part_offset · virtual=data_offset.
    uint64_t off = 0, nb = 0;
    resolve_part_addr(g_mod.layers[(size_t)mlpos].parts[(size_t)part_idx], off, nb);
    return (const char *)s.base + off;
}

int ggml_moe_direct_part_info(const struct ggml_tensor * t, uint64_t * offset, uint64_t * bytes) {
    if (!t) return -1;
    if (!g_mod.sealed.load(std::memory_order_acquire)) return -1;
    auto it = g_mod.registry.find(t);
    if (it == g_mod.registry.end()) return -1;
    const PartInfo & p = g_mod.layers[(size_t)it->second.mlpos].parts[(size_t)it->second.part_idx];
    // ★§Z-⑥ A2 mode 분기(§3-3 — r3 권고). bin 반환 의미는 불변이고, virtual 에서만
    //   `data_offset`/`slice_bytes` 기준으로 선다(소비자 인터페이스 자체는 불변).
    uint64_t off = 0, nb = 0;
    resolve_part_addr(p, off, nb);
    if (offset) *offset = off;
    if (bytes) *bytes = nb;
    return 0;
}

int ggml_moe_direct_registered_layer_part(const struct ggml_tensor * t, int32_t * out_layer, int32_t * out_part) {
    if (!t) return -1;
    if (!g_mod.sealed.load(std::memory_order_acquire)) return -1;
    auto it = g_mod.registry.find(t);
    if (it == g_mod.registry.end()) return -1;
    if (out_layer) *out_layer = g_mod.moe_layers[(size_t)it->second.mlpos];
    if (out_part)  *out_part  = it->second.part_idx;
    return 0;
}

// ===========================================================================
// E-2a. New boundary surface - public entry points
//   E2_DESIGN_CONTRACT.md v0.4 FROZEN section 1
// ===========================================================================

int ggml_moe_store_acquire(int32_t mlpos, int32_t part, const int32_t * experts, int32_t n,
                           int32_t n_tokens_hint, uint64_t * out_token) {
    if (!experts || n <= 0 || !out_token) return -1;
    // A SECOND admission, held across the token issue as well. The primitive drops its own
    // admission when it returns, and at that moment the lease is already on the ledger but
    // the handle does not exist yet; a drain that reached zero in that gap would tear the
    // epoch down between the lease and the token that owns it. Nesting is free - the counter
    // is just a counter - and it makes "an acquire is in flight" mean the whole call.
    XStoreAdmission admit;
    if (!admit.admitted) return -1;
    XStoreAcquireKey key;
    // E-2b-0 (contract section 1 v0.5): the caller's own part goes down, where E-2a put a
    // literal 0. The demand path uses part_idx for Phase A / lead-time row correlation only -
    // it never addresses with it, a record is loaded whole - but "only correlation" is not
    // "harmless": the trace stream carries the consumer's real part on the other side of that
    // join, so a fixed 0 collapsed the (layer, part) key on one side and nothing lined up.
    // The resolution below also RANGE-CHECKS it against this layer's parts vector, which is
    // where an out-of-range part is refused with no token and no lease.
    const int rc = xstore_acquire_common(/*t=*/nullptr, /*by_tensor=*/false, mlpos, part,
                                         experts, n, n_tokens_hint, &key);
    if (rc != 0) return rc;   // failure path: no token issued, no partial lease (section 2 row 6 e)
    uint64_t h = 0;
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        // key.part_idx, not the raw argument: what the token captures is the value the
        // resolution accepted and handed to the ledgers, so a capture can never disagree with
        // the rows the same acquire produced.
        h = xstore_token_issue_locked(key.mlpos, key.part_idx, experts, n);
    }
    *out_token = h;
    return 0;
}

// E-2b-0 (contract section 1 v0.5): this body takes NO module lock. What it reads is either
// atomic, published once and then immutable (the table pointer), immutable while the entry is
// live (the token payload is a const object - g0_tools/e2b0_static_gate.py bans the casts
// that could launder that away), or the
// very same post-seal state the old lock-free ptr() has always read without one
// (n_expert / layers / record_table / slots). It introduces no new lock-free assumption; it
// stops paying a global mutex on a path whose replacement in E-2b-1 is called n_as * nth
// times per mul_mat_id and nr times per add_id.
int ggml_moe_store_view(uint64_t token, int32_t expert, int32_t part,
                        struct ggml_moe_buffer_view * out) {
    if (!out) return -1;
    // (1) handle generation, then epoch. Both are fatal classes, not soft failures.
    //     E-2b-1 perf-repair: the entry comes back from the lookup that already validated
    //     it, so this path loads the table pointer once instead of twice. The checks
    //     themselves are unchanged and still run on every call.
    XStoreToken * ent = nullptr;
    const int32_t idx = xstore_handle_lookup(token, &ent);
    if (idx < 0) xstore_fatal_handle("store_view", token, "stale, released or fabricated handle");
    const XStoreToken & tk = *ent;
    const uint64_t live_epoch = g_xs_epoch.load(std::memory_order_acquire);
    if (live_epoch == 0 || tk.store_epoch() != live_epoch)
        xstore_fatal_handle("store_view", token, "token used outside its store instance epoch");
    // (2) the token carries the batch mlpos; expert plus that mlpos give the record ordinal.
    if (expert < 0 || expert >= g_mod.n_expert) return -1;
    const int64_t record_idx = (int64_t)tk.mlpos() * g_mod.n_expert + expert;
    // (3) that record has to be one this token actually captured. E-2b-1 perf-repair
    //     replaces the linear scan with a direct lookup when the token carries an index;
    //     the QUESTION and its answer are identical, only the search is. The index is an
    //     immutable part of the same const payload, so consulting it introduces no state
    //     that could disagree with `records` - the ordinal is bounds-checked against the
    //     very vector it indexes, and a miss falls to the same `return -1` as before.
    const std::vector<uint16_t> & eix = tk.expert_to_ref();
    const XStoreRecordRef * ref = nullptr;
    if (!eix.empty()) {
        if ((size_t)expert >= eix.size()) return -1;
        const uint16_t ord = eix[(size_t)expert];
        if (ord == XSTORE_VIEW_NO_REF) return -1;
        if ((size_t)ord >= tk.records().size()) return -1;
        // The membership answer is still the record ordinal comparison, not the table
        // slot: an index that ever disagreed with the records it points into refuses
        // here rather than handing back a record this token did not capture.
        if (tk.records()[(size_t)ord].record_idx != record_idx) return -1;
        ref = &tk.records()[(size_t)ord];
    } else {
        for (size_t i = 0; i < tk.records().size(); i++) {
            if (tk.records()[i].record_idx == record_idx) { ref = &tk.records()[i]; break; }
        }
    }
    if (!ref) return -1;
    // (4) part is validated against THIS epoch's parts vector - a part ordinal is seal-scoped.
    //     Note this `part` is the caller's projection key and is deliberately NOT required to
    //     equal the batch part the token captured (contract section 1).
    if (tk.mlpos() < 0 || tk.mlpos() >= (int32_t)g_mod.layers.size()) return -1;
    const LayerInfo & L = g_mod.layers[(size_t)tk.mlpos()];
    if (part < 0 || part >= (int32_t)L.parts.size()) return -1;
    // (5) re-check the captured slot on the record table and on the slot itself. Reachable
    //     when the two surfaces are mixed - an old-surface release can drop the lease while
    //     the token still exists, and the slot is then evictable.
    //     E-2b-0 dropped the `s.owner_record != record_idx` arm that used to sit here. That
    //     field is a plain int64_t written under the module lock, so reading it from a
    //     lock-free path would be a data race - and it was the ONLY non-atomic mutable field
    //     this body touched. Nothing is lost: eviction bumps s.generation BEFORE it rewrites
    //     owner_record, so a slot whose generation still equals the captured one has not been
    //     re-claimed since the capture, and therefore still owns that record. The generation
    //     pair below implies the owner check rather than merely accompanying it.
    const RecordEntry & re = g_mod.record_table[(size_t)record_idx];
    const int32_t si = re.slot_index.load(std::memory_order_acquire);
    if (si < 0 || si != ref->slot_index) return -1;
    if (re.slot_generation.load(std::memory_order_relaxed) != ref->slot_generation) return -1;
    const Slot & s = g_mod.slots[(size_t)si];
    if (s.generation.load(std::memory_order_acquire) != ref->slot_generation) return -1;
    // (6) compute, then re-validate, then write. `s.base` is a plain pointer - the same one
    //     ptr() has always read unlocked - so the generation is loaded once more after the
    //     read and before anything reaches `out`. A slot re-claimed in between refuses
    //     softly here instead of publishing an address that moved underneath it, and `out`
    //     is still untouched on that path.
    uint64_t off = 0, nb = 0;
    resolve_part_addr(L.parts[(size_t)part], off, nb);
    const char * addr = (const char *)s.base + off;
    if (s.generation.load(std::memory_order_acquire) != ref->slot_generation) return -1;
    out->ptr   = addr;
    out->bytes = nb;
    return 0;
}

void ggml_moe_store_release(uint64_t token) {
    // E-2b-2 (D12 catch 1): the WHOLE body runs inside the fail-stop. Everything that can
    // allocate while a live token is being returned is in here - the local ids vector and,
    // through xstore_release_common, the persistent command's ids.assign and the PhaseA
    // release hooks. Contract v0.7 section 2 row 5: an allocation failure on this path does
    // not come back to the caller. The lock_guard unwinds before the catch runs, and abort
    // does not need the lock.
    try {
    int32_t mlpos = -1;
    int32_t part  = -1;
    // E-2b-2R (R3): the id list for a SMALL return is built in a stack buffer, so the
    // RETURNED-ID RECONSTRUCTION pays no malloc/free pair. The vector is still here and still
    // does the work whenever the token is wider than XSTORE_RELEASE_SMALL, which is why the
    // fail-stop above is unchanged: everything on this path that can allocate is still inside
    // this try.
    // ★Scope, precisely (Codex e2b2r_engine r2 NOTE 1): what becomes allocation-free below the
    // threshold is THIS reconstruction and nothing else. xstore_release_common further down can
    // still allocate - the persistent command's ids.assign, and the PhaseA release hooks - and
    // that is exactly why the try covers the whole body rather than just this block.
    int32_t ids_small[XSTORE_RELEASE_SMALL];
    std::vector<int32_t> ids;
    int32_t * idp = ids_small;
    int32_t   idn = 0;
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        const int32_t idx = xstore_handle_lookup(token);
        if (idx < 0)
            xstore_fatal_handle("store_release", token,
                                "stale handle or double release");
        // const: the payload is a const object, so it is read-only in every scope there
        // is. The retirement is xstore_token_free_locked's.
        const XStoreToken & tk = *xstore_entry(idx);
        const uint64_t live_epoch = g_xs_epoch.load(std::memory_order_acquire);
        if (live_epoch == 0 || tk.store_epoch() != live_epoch)
            xstore_fatal_handle("store_release", token,
                                "token released outside its store instance epoch");
        mlpos = tk.mlpos();
        // (2) E-2b-0 - the release order the contract fixes: compare the whole capture, copy
        // out the ids AND the captured part, invalidate the handle, then release. The part
        // travels with the token, which is why this entry point still takes nothing but a
        // handle (contract section 2, row 5).
        part  = tk.part();
        // (3) EVERY captured record is compared against the live state before anything is
        // given back - slot index, the record's generation and the slot's own owner and
        // generation, per record, one mismatch is fatal. This runs BEFORE the handle is
        // freed so a token whose capture went stale is reported as exactly that rather than
        // being quietly unwound halfway.
        for (size_t i = 0; i < tk.records().size(); i++) {
            xstore_verify_capture_or_fatal("store_release", token, mlpos, tk.records()[i]);
        }
        // E-2b-2R (R3). Both arms build the SAME sequence - unique records in first-appearance
        // order, each repeated by its multiplicity - and differ only in where it is written.
        // n_ids IS that sum: it is the `units` the payload was constructed with, from the same
        // record list, and the payload is const from then on. The small arm therefore writes
        // exactly `want` elements into a buffer admitted for at least that many.
        const int32_t want = tk.n_ids();
        if (want > XSTORE_RELEASE_SMALL) {
            ids.reserve((size_t)want);
            for (size_t i = 0; i < tk.records().size(); i++) {
                const int32_t eid = (int32_t)(tk.records()[i].record_idx
                                              - (int64_t)mlpos * g_mod.n_expert);
                for (int32_t m = 0; m < tk.records()[i].multiplicity; m++) ids.push_back(eid);
            }
            idp = ids.data();
            idn = (int32_t)ids.size();
        } else {
            for (size_t i = 0; i < tk.records().size(); i++) {
                const int32_t eid = (int32_t)(tk.records()[i].record_idx
                                              - (int64_t)mlpos * g_mod.n_expert);
                for (int32_t m = 0; m < tk.records()[i].multiplicity; m++) ids_small[idn++] = eid;
            }
        }
        // Freed only after the whole comparison passed, and before the leases go back, so a
        // double release is caught by the lookup above rather than by whatever the second
        // pass happens to do.
        xstore_token_free_locked(idx);
    }
    if (idn == 0) return;
    xstore_release_common(/*t=*/nullptr, /*by_tensor=*/false, mlpos, part,
                          idp, idn);
    } catch (const std::bad_alloc &) {
        fatal_abort("xstore: allocation failed while returning token %llu - fail-stop (contract section 2 row 5)",
                    (unsigned long long) token);
    }
}

int ggml_moe_adapter_resolve(const struct ggml_tensor * t, int32_t * out_mlpos, int32_t * out_part) {
    if (!t) return -1;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const uint64_t live_epoch = g_xs_epoch.load(std::memory_order_acquire);
    if (live_epoch == 0) return -1;
    auto it = g_xs.adapter.find(t);
    if (it == g_xs.adapter.end()) return -1;
    if (it->second.epoch != live_epoch) return -1;
    if (out_mlpos) *out_mlpos = it->second.mlpos;
    if (out_part)  *out_part  = it->second.part_idx;
    return 0;
}

uint64_t ggml_moe_store_epoch(void) {
    return g_xs_epoch.load(std::memory_order_acquire);
}

int32_t ggml_moe_store_state(void) {
    return g_xs_state.load(std::memory_order_acquire);
}

void ggml_moe_store_outstanding(int64_t * out_handles, int64_t * out_lease_units,
                                int64_t * out_slot_leases) {
    if (out_handles)     *out_handles     = g_xs_handles.load(std::memory_order_relaxed);
    if (out_lease_units) *out_lease_units = xstore_outstanding_units();
    if (out_slot_leases) {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        *out_slot_leases = xstore_slot_lease_sum_locked();
    }
}

int ggml_moe_store_drain_begin(void) {
    // Read the clock before the transition, publish it after: the anchor may only be written
    // by the caller that actually wins the transition. A losing caller that wrote the anchor
    // first would re-arm the deadline of the drain already running, which is exactly the
    // "single deadline" the contract forbids restarting.
    const int64_t t0 = xstore_steady_ms();
    int32_t want = MOE_STORE_RUNNING;
    if (!g_xs_state.compare_exchange_strong(want, MOE_STORE_DRAINING,
                                            std::memory_order_seq_cst,
                                            std::memory_order_seq_cst)) {
        return -1;
    }
    g_xs_drain_t0_ms.store(t0, std::memory_order_release);
    return 0;
}

int ggml_moe_store_drain_finish(int32_t deadline_ms) {
    if (g_xs_state.load(std::memory_order_seq_cst) != MOE_STORE_DRAINING) return -1;
    // ONE deadline, anchored at drain_begin (xstore_drain_remaining_ms). The lease wait
    // spends it first and the dispatcher join gets whatever is left; neither starts a clock
    // of its own and neither gets a fresh full budget. deadline_ms <= 0 means no budget at
    // all: one check, then fail - never the 30000 default, which is a stop policy and not a
    // stand-in for "do not wait".
    bool zero = false;
    for (;;) {
        zero = xstore_drain_zero();
        if (zero) break;
        if (xstore_drain_remaining_ms(deadline_ms) <= 0) break;   // includes the <= 0 case
        Sleep(1);
    }
    if (!zero) {
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            xstore_log_residual_inventory_locked(deadline_ms);
        }
        fatal_abort("xstore: graceful drain did not reach 0 outstanding lease units within "
                    "%d ms measured from drain_begin (lease_units=%lld admissions=%lld) - "
                    "see the residual lease inventory above",
                    (int)deadline_ms, (long long)xstore_outstanding_units(),
                    (long long)g_xs_admit.load(std::memory_order_seq_cst));
    }
    // Re-checked on the near side of the irreversible steps. Nothing may have changed since
    // the confirmation above - no admission is possible in DRAINING and every admitted
    // acquire had already finished - so a difference here is not a race to wait out, it is a
    // lease that appeared with no admission behind it, and stopping the store on top of it
    // is the outcome the whole gate exists to prevent.
    if (!xstore_drain_zero()) {
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            xstore_log_residual_inventory_locked(deadline_ms);
        }
        fatal_abort("xstore: a lease unit appeared after the drain confirmed zero "
                    "(lease_units=%lld admissions=%lld) - the admission gate and the "
                    "zero-confirm are no longer on the same serialization boundary",
                    (long long)xstore_outstanding_units(),
                    (long long)g_xs_admit.load(std::memory_order_seq_cst));
    }
    // Order is the contract: zero confirmed -> dispatcher stopped -> epoch invalidated.
    // Reversing the last two leaves a token with no path back to the lease it holds.
    // The stop is handed the deadline ITSELF, not a remaining slice of it, because the stop has
    // three internal waits of its own (the predictor's capture-idle wait, the predictor worker
    // join, the dispatcher join). Handing it a relative budget is what let each of those spend
    // a fresh full copy - r1 fixed the handoff and left the reuse INSIDE the helper (Codex r2
    // blocker 2). One absolute instant, derived straight from the drain_begin anchor, is
    // measured against by every stage; "check once" (deadline_ms <= 0) arrives as an instant
    // already past, so no stage waits and none of them substitutes the 30 s default.
    if (!dispatcher_stop_and_join_until(/*abort_mode=*/false,
                                        xstore_drain_deadline_ms(deadline_ms))) {
        return -1;   // the stop-timeout latch is already set; the harness turns that red
    }
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        xstore_invalidate_epoch_locked();
    }
    g_xs_state.store(MOE_STORE_STOPPED, std::memory_order_release);
    return 0;
}

int ggml_moe_direct_test_xstore_token(uint64_t token, int32_t * out_mlpos,
                                      int32_t * out_n_records, int32_t * out_lease_units,
                                      uint64_t * out_epoch) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const int32_t idx = xstore_handle_lookup(token);
    if (idx < 0) return -1;
    const XStoreToken & tk = *xstore_entry(idx);
    if (out_mlpos)       *out_mlpos       = tk.mlpos();
    if (out_n_records)   *out_n_records   = (int32_t)tk.records().size();
    if (out_lease_units) *out_lease_units = tk.n_ids();
    if (out_epoch)       *out_epoch       = tk.store_epoch();
    return 0;
}

// E-2b-0 - the captured batch part, in its own window. Widening the one above would have
// edited five sealed E-2a checks for no gain; this leaves them byte-identical.
int ggml_moe_direct_test_xstore_token_part(uint64_t token, int32_t * out_part) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    const int32_t idx = xstore_handle_lookup(token);
    if (idx < 0) return -1;
    if (out_part) *out_part = xstore_entry(idx)->part();
    return 0;
}

// E-2b-0 - the handle table as a gate sees it. The address is the load-bearing value: a gate
// that only compared the capacity against XSTORE_MAX_TOKENS would be echoing the constant it
// was handed, whereas "this address is the same one it was 256 acquires ago" is a fact about
// the storage that no constant can supply.
int ggml_moe_direct_test_xstore_table(uint64_t * out_addr, int32_t * out_capacity,
                                      int32_t * out_live, int32_t * out_free) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    XStoreToken * t = g_xs.table.load(std::memory_order_acquire);
    if (t == nullptr) return -1;
    if (out_addr)     *out_addr     = (uint64_t)(uintptr_t)t;
    if (out_capacity) *out_capacity = g_xs.table_n.load(std::memory_order_relaxed);
    if (out_live)     *out_live     = g_xs.live_count;
    if (out_free)     *out_free     = g_xs.free_count;
    return 0;
}

// E-2b-0 - hold the module mutex, for real, on the calling thread. This is not a hook that
// imitates contention: it takes the same std::mutex every production path takes, with a plain
// lock_guard, so a call that wants that lock genuinely waits. What the fixture controls is the
// DURATION, not the predicate. The cap exists because a defect in the fixture must not be able
// to wedge a 1400-check release gate; when it fires, safety_fired() says so and the gate that
// measured through that window knows its window was not the intended one. Blocks until
// released or capped, so the caller runs it on a thread of its own.
void ggml_moe_direct_test_mutex_hold_arm(int32_t safety_ms) {
    const int32_t cap = (safety_ms > 0) ? safety_ms : 5000;
    std::lock_guard<std::mutex> hold(g_mod.mutex);
    std::unique_lock<std::mutex> lk(g_xs_hold_mu);
    g_xs_hold_go     = false;
    g_xs_hold_safety = 0;
    g_xs_hold_held   = true;
    g_xs_hold_cv.notify_all();
    const bool released = g_xs_hold_cv.wait_for(lk, std::chrono::milliseconds(cap),
                                                []() { return g_xs_hold_go; });
    if (!released) g_xs_hold_safety = 1;
    g_xs_hold_held = false;
    g_xs_hold_cv.notify_all();
}

int32_t ggml_moe_direct_test_mutex_hold_wait(int32_t timeout_ms) {
    const int32_t cap = (timeout_ms > 0) ? timeout_ms : 2000;
    std::unique_lock<std::mutex> lk(g_xs_hold_mu);
    if (g_xs_hold_held) return 1;
    const bool ok = g_xs_hold_cv.wait_for(lk, std::chrono::milliseconds(cap),
                                          []() { return g_xs_hold_held; });
    return ok ? 1 : 0;
}

// Asks the holder to let go. It returns as soon as the request is posted; the caller learns
// that the lock is actually back by joining the holder thread, which is what the fixture does.
void ggml_moe_direct_test_mutex_hold_release(void) {
    std::lock_guard<std::mutex> lk(g_xs_hold_mu);
    g_xs_hold_go = true;
    g_xs_hold_cv.notify_all();
}

int32_t ggml_moe_direct_test_mutex_hold_safety_fired(void) {
    std::lock_guard<std::mutex> lk(g_xs_hold_mu);
    return g_xs_hold_safety;
}

int32_t ggml_moe_direct_test_xstore_drain_remaining(int32_t deadline_ms) {
    // Reports the SAME number drain_finish spends, through the same helper, so a gate built
    // on it cannot drift away from the production computation. <0 = no drain in progress,
    // which is a different answer from "the budget is exhausted" (0).
    if (g_xs_drain_t0_ms.load(std::memory_order_acquire) < 0) return -1;
    return xstore_drain_remaining_ms(deadline_ms);
}

int32_t ggml_moe_direct_test_xstore_drain_zero_ready(int64_t * out_admissions,
                                                     int64_t * out_lease_units) {
    const int32_t ready = xstore_drain_zero() ? 1 : 0;
    if (out_admissions)  *out_admissions  = g_xs_admit.load(std::memory_order_seq_cst);
    if (out_lease_units) *out_lease_units = xstore_outstanding_units();
    return ready;
}

void ggml_moe_direct_test_xstore_arm_last_generation(void) {
    // Shaped exactly like ggml_moe_direct_reset_for_test: the lock is taken HERE and the
    // locked writer is called exactly once, so the gate can read the module lock being held
    // at the call rather than merely existing somewhere in this function.
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    xstore_arm_last_generation_locked();
}

int32_t ggml_moe_direct_test_stop_join_stage(int64_t * out_deadline_abs_ms,
                                             int32_t * out_left_ms) {
    // The LAST stage of the stop chain. Reporting the deadline it measured against - rather
    // than the time it took - is what lets a gate compare it with the two stages the predictor
    // reports and with the caller's own clock reading, and so decide whether the three stages
    // shared ONE budget or each started a clock of its own.
    if (out_deadline_abs_ms) *out_deadline_abs_ms = g_xs_stopjoin_deadline.load(std::memory_order_relaxed);
    if (out_left_ms)         *out_left_ms         = g_xs_stopjoin_left.load(std::memory_order_relaxed);
    return g_xs_stopjoin_n.load(std::memory_order_relaxed);
}

int64_t ggml_moe_direct_test_steady_ms(void) {
    // The same clock, and the same epoch, every deadline above is expressed in. A gate that
    // reads it before drain_begin can derive `anchor + budget` for itself and compare that with
    // what the stages report - two independent derivations of one instant, not an echo.
    return xstore_steady_ms();
}


/* XBE-PIN-BEGIN emit-slab-helper */
// E-2b-2 (A1-R, spec section B-5): the slab / zero-fill loop BOTH emit surfaces share.
// Lifted out of the old tensor* body without changing a step - the one difference is that
// where the source pointer comes from is now a caller-supplied resolver. Static and
// template-free on purpose (e2b0 c5 region_preamble: a template header before a pinned head
// is a failure), and the resolver is a plain function pointer plus a context, not a lambda.
typedef const void * (*xbe_resolve_fn)(void * rctx, int32_t id);

static int xbe_emit_slabs(uint64_t expert_size, uint64_t total_bytes, int32_t n_expert,
                          const int32_t * used_ids, int32_t n_used,
                          void (*emit)(void * ud, const void * src, uint64_t dst_offset, uint64_t len),
                          void * ud, xbe_resolve_fn resolve, void * rctx) {
    if (!emit || n_used < 0) return -1;
    if (n_used == 0) return 0;
    if (!used_ids) return -1;

    for (int32_t k = 0; k < n_used; k++) {
        const int32_t id = used_ids[k];
        if (id < 0 || id >= n_expert) return -1;

        const void * src_ptr = resolve(rctx, id);
        if (!src_ptr) {
            ggml_moe_direct_fallback_hit("emit-expert-copies-ptr-null");
            GGML_ABORT("moe-direct: ptr null for expert=%d", id);
        }

        const uint64_t dst_offset = (uint64_t) id * expert_size;
        emit(ud, src_ptr, dst_offset, expert_size);

        const bool slab_boundary = (k == n_used - 1) || (used_ids[k + 1] != id + 1);
        if (slab_boundary) {
            const uint64_t slab_end = dst_offset + expert_size;
            const uint64_t remaining = (slab_end < total_bytes) ? (total_bytes - slab_end) : 0;
            const uint64_t zero_len = remaining < 512ULL ? remaining : 512ULL;
            if (zero_len > 0) emit(ud, nullptr, slab_end, zero_len);
        }
    }
    return 0;
}

/* XBE-PIN-END emit-slab-helper */

/* XBE-PIN-BEGIN store-emit-impl */
// The token-key resolver context. Records reference plus scalars, nothing else: there is
// deliberately no `const XStoreToken *` member, because a handle back to the token would be
// a handle back to the accessors, and "resolved once at entry" (contract section 1) is meant
// to be provable from the shape rather than promised in a comment.
struct XbeTokenEmitCtx {
    const std::vector<XStoreRecordRef> & records;   // bound once at entry from tk.records()
    const int64_t record_base;                      // mlpos * g_mod.n_expert - once at entry
    const uint64_t off;                             // resolve_part_addr(L.parts[part]) - once
    size_t cursor;                                  // merge-walk cursor: the only mutable member
};

// Protected region: static_cast only, no token accessor, no lock, no allocation. The record
// re-validation mirrors view steps 5 and 6 exactly - table slot, record generation, slot
// generation, address, then the generation once more after the read.
static const void * xbe_resolve_token(void * rctx, int32_t id) {
    XbeTokenEmitCtx * c = static_cast<XbeTokenEmitCtx *>(rctx);
    const int64_t record_idx = c->record_base + id;   // id was range-checked by the helper
    const XStoreRecordRef * ref = nullptr;
    const size_t n_rec = c->records.size();
    for (size_t n = 0; n < n_rec; n++) {
        size_t j = c->cursor + n;
        if (j >= n_rec) j -= n_rec;
        if (c->records[j].record_idx == record_idx) { ref = &c->records[j]; c->cursor = j + 1; break; }
    }
    if (!ref) return nullptr;
    const RecordEntry & re = g_mod.record_table[(size_t) record_idx];
    const int32_t si = re.slot_index.load(std::memory_order_acquire);
    if (si < 0 || si != ref->slot_index) return nullptr;
    if (re.slot_generation.load(std::memory_order_relaxed) != ref->slot_generation) return nullptr;
    const Slot & s = g_mod.slots[(size_t) si];
    if (s.generation.load(std::memory_order_acquire) != ref->slot_generation) return nullptr;
    const char * addr = static_cast<const char *>(s.base) + c->off;
    if (s.generation.load(std::memory_order_acquire) != ref->slot_generation) return nullptr;
    // The cast is explicit so the statement is not `return <identifier>;`: the gate's
    // declaration regex reads that shape as a local named addr of type `return`, which
    // then disagrees with the pinned local type. Same value, same order, parseable.
    return static_cast<const void *>(addr);
}

int ggml_moe_store_emit_expert_copies(
        uint64_t token, int32_t part,
        uint64_t expert_size, uint64_t total_bytes, int32_t n_expert,
        const int32_t * used_ids, int32_t n_used,
        void (*emit)(void * ud, const void * src, uint64_t dst_offset, uint64_t len),
        void * ud) {
    if (!emit || n_used < 0) return -1;
    if (n_used == 0) return 0;
    if (!used_ids) return -1;
    // (1) liveness: handle generation, then epoch - the same two fatal classes as view step 1.
    //     The two-argument lookup is view's E-2b-1 perf-repair form: the entry comes back from
    //     the call that already validated it, so the table pointer is loaded once.
    XStoreToken * ent = nullptr;
    const int32_t idx = xstore_handle_lookup(token, &ent);
    if (idx < 0) xstore_fatal_handle("store_emit", token, "stale, released or fabricated handle");
    const XStoreToken & tk = *ent;
    const uint64_t live_epoch = g_xs_epoch.load(std::memory_order_acquire);
    if (live_epoch == 0 || tk.store_epoch() != live_epoch)
        xstore_fatal_handle("store_emit", token, "token used outside its store instance epoch");
    // (2) bound ONCE: mlpos, the records reference, the record base and the part offset.
    //     Below this line there is not one token accessor call.
    const int32_t mlpos = tk.mlpos();
    if (mlpos < 0 || mlpos >= (int32_t) g_mod.layers.size()) return -1;
    const LayerInfo & L = g_mod.layers[(size_t) mlpos];
    if (part < 0 || part >= (int32_t) L.parts.size()) return -1;
    uint64_t off = 0, nb = 0;
    resolve_part_addr(L.parts[(size_t) part], off, nb);
    if (nb != expert_size)
        GGML_ABORT("moe-direct: view bytes %llu != expert_size %llu (token=%llu part=%d)",
                   (unsigned long long) nb, (unsigned long long) expert_size,
                   (unsigned long long) token, part);
    XbeTokenEmitCtx ctx{ tk.records(), (int64_t) mlpos * g_mod.n_expert, off, 0 };
    // (3) one walk over the captured records, through the shared slab loop.
    return xbe_emit_slabs(expert_size, total_bytes, n_expert, used_ids, n_used, emit, ud,
                          &xbe_resolve_token, &ctx);
}

/* XBE-PIN-END store-emit-impl */

// The tensor-key resolver context (spec amendment 26-09-01, rows g-1 and g-2). It carries a
// `const struct ggml_tensor *`, and a ctx is the only way to hand that to a `void *` resolver
// while keeping the const: the cast that would strip it is on this file's banned-token list,
// so this struct is structural rather than stylistic (spec H-2 row g-1 carries the full
// reasoning). Placed after the public emit and immediately before its only consumer, so no
// forward declaration is needed.
struct XbeTensorEmitCtx {
    const struct ggml_tensor * input;
};

static const void * xbe_resolve_tensor(void * rctx, int32_t id) {
    XbeTensorEmitCtx * c = static_cast<XbeTensorEmitCtx *>(rctx);
    // The tensor is bound to a local before the call so the argument list carries no
    // `->`: the gate's argument splitter counts `<` and `>` as bracket depth, so an
    // arrow inside an argument hides the comma and the call reads as one argument.
    const ggml_tensor * in = c->input;
    return ggml_moe_direct_ptr(in, id);
}

// Legacy tensor-key emit, now a wrapper over the shared loop (E-2c removes it). Keeping it on
// the same helper is what makes the group E checks evidence that the extraction did not
// regress: they exercise the helper through this surface.
int ggml_moe_direct_emit_expert_copies(
        const struct ggml_tensor * input,
        uint64_t expert_size,
        uint64_t total_bytes,
        int32_t n_expert,
        const int32_t * used_ids,
        int32_t n_used,
        void (*emit)(void * ud, const void * src, uint64_t dst_offset, uint64_t len),
        void * ud) {
    if (!input) return -1;
    XbeTensorEmitCtx ctx{ input };
    return xbe_emit_slabs(expert_size, total_bytes, n_expert, used_ids, n_used, emit, ud,
                          &xbe_resolve_tensor, &ctx);
}

void ggml_moe_direct_fallback_hit(const char * site) {
    g_mod.fallback_count.fetch_add(1, std::memory_order_relaxed);
    log_err("structural fallback (first line of defence) reached: site=%s", site ? site : "(null)");
}

void ggml_moe_prefetch_marker(uint64_t requested_bytes, uint32_t call_count, uint32_t suppressed_count) {
    fprintf(stderr,
        "[moe-direct][prefetch-marker] requested_bytes=%llu call_count=%u suppressed_count=%u moe_no_prefetch=%d\n",
        (unsigned long long)requested_bytes, call_count, suppressed_count, ggml_moe_no_prefetch() ? 1 : 0);
    fflush(stderr);
}

bool ggml_moe_direct_metrics_write_record(const char * type, const char * payload_json) {
    if (!type) return false;
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.metrics.enabled) return false;
    json j = common_fields(type);
    if (payload_json && *payload_json) {
        json p = json::parse(payload_json, nullptr, /*allow_exceptions=*/false);
        if (p.is_object()) {
            for (auto it = p.begin(); it != p.end(); ++it) j[it.key()] = it.value();
        } else {
            log_err("metrics_write_record: payload JSON parse failed(type=%s) - writing common fields only", type);
        }
    }
    metrics_write_line(j.dump() + "\n", true);
    return true;
}

// R-B5a facade (iv): command request / result lifecycle (RB_DESIGN.md:1221 section
// 2-6 (iv)). The caller states the command it wants captured; `g_mod.cmds[ci]` is
// filled here and nowhere outside the dispatch cluster. The statements below are the
// ones this member replaced, in the same order, with the caller's lock unchanged.
inline void dfac_cmd_query_absent_fill_locked(int32_t ci, int32_t mlpos, int32_t n,
                                              const int32_t * experts) {
    ControlCmd & c = g_mod.cmds[(size_t)ci];
    c.kind  = CMD_TEST_QUERY;
    c.qkind = QK_ABSENT_COUNT;
    c.mlpos = mlpos;
    c.n_ids = n;
    c.ids.assign(experts, experts + n);
}

int32_t ggml_moe_direct_absent_count(int32_t mlpos, const int32_t * experts, int32_t n) {
    if (!experts || n <= 0) return 0;
    // ★B1 — absent query도 command(§2 "release·absent query·test snapshot은 전부 command").
    if (persistent_active()) {
        const int32_t ci = cq_alloc();
        if (ci < 0) return 0;
        ControlCmd & c = g_mod.cmds[(size_t)ci];
        dfac_cmd_query_absent_fill_locked(ci, mlpos, n, experts);
        cq_submit_and_wait(ci);
        const int32_t absent = c.q_out_i32_a;
        cq_free(ci);
        return absent;
    }
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.sealed) return 0;
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.moe_layers.size()) return 0;
    int32_t absent = 0;
    for (int32_t i = 0; i < n; i++) {
        const int32_t eid = experts[i];
        if (eid < 0 || eid >= g_mod.n_expert) continue;
        const int64_t record_idx = (int64_t)mlpos * g_mod.n_expert + eid;
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
    return absent;
}

void ggml_moe_direct_graph_shape_note(int32_t n_nodes, int32_t n_splits) {
    // ★PA 무관 graph 형상 지문(Codex R1 필수 1) — metrics 활성 시에만. 호출자는 llama-context의
    //   graph compute 직후(매 graph 1회·decode 스레드). 비용 = FNV 8B + graph당 락 1회.
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.metrics.enabled) return;
    const uint32_t nn = (uint32_t)n_nodes;
    const uint32_t ns = (uint32_t)n_splits;
    g_mod.graph_nodes_digest  = fnv1a64_bytes(g_mod.graph_nodes_digest,  &nn, sizeof(nn));
    g_mod.graph_splits_digest = fnv1a64_bytes(g_mod.graph_splits_digest, &ns, sizeof(ns));
    g_mod.graph_count++;
}

// ★5c-T censor cut(SPEC_PHASEB_WIRE 부록 14.5 · controller §2) — **관측이 종료되어 사용 여부가
//   미결인 채 더는 판정할 수 없는 시점**에 잔여 OPEN 전건을 일괄 `censored` 로 닫는다.
//   대상 = ①pending/in-flight prefetch op 중 issued(=제출됨) ∧ 미분할 ②미소비 READY(PROBATION)
//   슬롯(접점 실증: 현행 shutdown 시퀀스는 slot cache_class 를 검사하지 않아 무처분으로 남았다).
//   ★취소 통지(run_end)는 cut 이 아니다 — 그 뒤에도 op·READY 는 정상 생존·사용되며 그 사용은
//   정상 used_* 로 계상된다. 이 함수 호출 이후 어떤 분할로의 재계상도 금지된다(pb_censused).
void pb_censor_cut() {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.pb_collect || g_mod.pb_censused) return;
    std::vector<ggml_moe_phaseb_ledger_owner> parts;
    for (size_t i = 0; i < g_mod.pending_ops.size(); i++) {
        PendingIo & op = g_mod.pending_ops[i];
        if (!op.in_use || op.origin != PIO_ORIGIN_PREFETCH) continue;
        if (op.pb_submit_event_seq == GGML_MOE_PHASEB_NA_U64) continue;   // 미제출 = issued 아님
        if (!pb_partition_op_locked((int32_t)i, GGML_MOE_PB_P_CENSORED)) continue;
        ggml_moe_phaseb_count(GGML_MOE_PB_C_CENSORED, op.pb_request_epoch, 1);   // owner 귀속
        ggml_moe_phaseb_ledger_owner o{};
        o.request_epoch  = op.pb_request_epoch;
        o.graph_seq      = op.pb_graph_seq;
        o.intent_ordinal = op.intent_seq;
        parts.push_back(o);
    }
    for (size_t i = 0; i < g_mod.slots.size(); i++) {
        Slot & s = g_mod.slots[i];
        if (s.cache_class != GGML_MOE_PB_CLASS_PROBATION) continue;
        ggml_moe_phaseb_count(GGML_MOE_PB_C_CENSORED, s.pf_request_epoch, 1);
        ggml_moe_phaseb_ledger_owner o{};
        o.request_epoch  = s.pf_request_epoch;
        o.graph_seq      = s.pf_graph_seq;
        o.intent_ordinal = s.pf_intent_ordinal;
        parts.push_back(o);
    }
    ggml_moe_phaseb_ledger r = pb_led_new(GGML_MOE_PB_L_CENSOR_CENSUS);
    r.partition = GGML_MOE_PB_P_CENSORED;
    // ★r1 M4 — census owner 목록은 **전량**이다(구 판의 48개 절단은 카운터와 ledger 를
    //   불일치시켰다). 버퍼 부족이면 그룹 전체가 드롭되고 complete=0 으로 fail-close 된다.
    pb_led_emit_locked(r, parts.data(), (uint32_t)parts.size());
    g_mod.pb_censused = true;   // ★이 시점 이후 재계상 금지(controller §2)
}

void ggml_moe_direct_finalize(void) {
    // ★PB_FINALIZE_CUT(SPEC_PHASEB_WIRE §0.1) — run-end 판정의 선형화 경계. 이 순서 그대로.
    // E-2b-2R (Codex e2b2r_engine r1 REVISE 3). The once latch covers the WHOLE function now,
    // not just the CUT block below it. It used to close before the IO3 flush and the Phase B
    // encode/write block, and those writers have no latch of their own - the Phase B sidecar
    // writer opens CREATE_NEW and nothing else. With alpha-1 the second call became certain:
    // the server calls ggml_moe_direct_finalize() and then llama_backend_free(), which now
    // finalizes too, so an ACTIVE run finalized twice. Codex reproduced it - the first call
    // wrote _sample.pb and _io.pb, deleting them and calling again RE-CREATED both, and not
    // deleting them made both fail with ERROR_FILE_EXISTS(80). Neither is what "finalize is
    // idempotent" is supposed to mean.
    //
    // The shape is a plain early return rather than a wider brace so the CUT block below keeps
    // its exact text and order; g_mod.pb_finalized is read and written under the same rules it
    // always was, is reset per seal at the reset path, and is named nowhere else in the tree.
    // The metrics block at the tail keeps its own metrics.finalized latch - now redundant, and
    // left alone because removing a latch is not what this repair is for.
    if (g_mod.pb_finalized) return;
    g_mod.pb_finalized = true;
    {
        // ① 선형화점 — 신규 capture와 신규 외부 control command를 **하나의 래치로 동시에** 거절
        //    한다(★R2 #1: 래치가 둘이면 그 사이에 시작된 capture 가 command CUT 이후에도 수용된다.
        //    이제 단일 원자 래치 1개를 capture 훅·cq_alloc 이 함께 확인한다).
        ggml_moe_phaseb_cut_close_capture();
        if (g_mod.disp_running.load(std::memory_order_acquire)) {
            dfac_q_request_stop();
        }
        // ② capture_q/predictor drain — worker가 잔여 capture 전건 소진·intent 발행 완결 후 stop/join.
        ggml_moe_phaseb_cut_drain_predictor();
        // ③ pre-cut 수령분 command drain + ④ 잔여 intent 전건 처분 + ⑤ 잔여 speculative
        //    CancelIoEx → 전 채널 terminal 회수 + ⑥ 잔여 0 확인 — dispatcher graceful 루프가 수행.
        dispatcher_stop_and_join(/*abort_mode=*/false);
        // ★5c-T censor cut(부록 14.5) — 기존 CUT ①~⑧ 불변, **⑥과 ⑦ 사이**의 additive 단계다.
        //   수집 OFF 면 내부에서 즉시 반환한다(어떤 상태도 만들지 않음).
        pb_censor_cut();
    }
    // ★R12 — IO3 사이드카 flush(§5-1: 1회 · 요청 중 파일 I/O 0). ★위치가 계약의 일부다:
    //   구 판은 finalize **첫 줄**에서 flush 하고 `io3_on=false` 로 내렸다 — 그 뒤의 CUT ⑤
    //   (잔여 CancelIoEx → 전 채널 terminal 회수)가 만드는 terminal 사건이 wire 에 하나도
    //   안 들어가는데 헤더는 `complete=1` 이었다(교차 r5 Y4-④). dispatcher drain **뒤**로 옮긴다.
    //   ★copies 는 dispatcher 가 아니라 copy worker 가 닫으므로 그 quiescence 를 짧게 기다린다
    //   (worker 는 g_mod.mutex 를 잡으므로 락을 쥔 채 기다리면 교착이다 — 잠금 밖 폴링).
    //   기다려도 미종결 행이 남으면 `io3_finalize_locked()` 의 자기검사가 complete=0 으로 내린다.
    {
        for (int i = 0; i < 2000; i++) {
            {
                std::lock_guard<std::mutex> lock(g_mod.mutex);
                if (!g_mod.io3_on || io3_all_terminal_locked()) break;
            }
            Sleep(1);
        }
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        io3_finalize_locked();
    }
    // ⑦ used/wasted 확정 + ⑧ 전 채널 encode·digest 확정·flush → summary.
    //    ★기존 Phase A flush를 이 시점(dispatcher quiescence 뒤)으로 재배열(§0.1 ★R2 #1).
    //    PB 사이드카·phase_b_run은 락 밖에서(내부 metrics_write_record가 mutex를 다시 잡는다).
    {
        uint64_t sample_records = 0, io_records = 0, a0_records = 0, obs_records = 0;
        uint64_t sample_digest = 0, io_digest = 0, a0_digest = 0, obs_digest = 0;
        int sample_written = 0, io_written = 0, a0_written = 0, obs_written = 0;
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            pb_drain_status_locked();   // ★R2 #5 — encode 직전 safety patch pass
            sample_records = (uint64_t)g_mod.pb_samples.size();
            io_records     = (uint64_t)g_mod.pb_ios.size();
        }
        a0_written     = ggml_moe_phaseb_write_a0equiv(&a0_records, &a0_digest);
        // ★8호 — observe 사이드카(모듈 소유 버퍼). 내부에서 tally 를 1회 확정한 뒤 기록한다.
        obs_written    = ggml_moe_phaseb_write_observe(&obs_records, &obs_digest);
        // ★5c-T — pb_ledger 사이드카(모듈 소유 버퍼 · 수집 OFF 면 미기록·0건). census 뒤에
        //   써야 CENSOR_CENSUS 레코드가 포함된다. 결과는 summary 에서 수집 전용 키로 echo.
        {
            uint64_t led_records = 0, led_digest = 0;
            const int led_written = ggml_moe_phaseb_write_ledger(&led_records, &led_digest);
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            g_mod.pb_ledger_records = led_records;
            g_mod.pb_ledger_digest  = led_digest;
            g_mod.pb_ledger_written = led_written;
        }
        {
            std::lock_guard<std::mutex> lock(g_mod.mutex);
            sample_written = ggml_moe_phaseb_write_sidecar(2, g_mod.pb_samples.data(), sample_records,
                                                           (uint32_t)sizeof(ggml_moe_phaseb_sample), &sample_digest);
            io_written     = ggml_moe_phaseb_write_sidecar(3, g_mod.pb_ios.data(), io_records,
                                                           (uint32_t)sizeof(ggml_moe_phaseb_io), &io_digest);
        }
        auto blk = [](uint64_t recs, int written, uint64_t dg) {
            char b[192];
            snprintf(b, sizeof(b), "{\"records\":%llu,\"written\":%d,\"digest\":\"0x%016llx\"}",
                     (unsigned long long)recs, written, (unsigned long long)dg);
            return std::string(b);
        };
        if (g_mod.metrics.enabled) {
            ggml_moe_phaseb_emit_runs(blk(a0_records, a0_written, a0_digest).c_str(),
                                      blk(sample_records, sample_written, sample_digest).c_str(),
                                      blk(io_records, io_written, io_digest).c_str(),
                                      blk(obs_records, obs_written, obs_digest).c_str(),
                                      ggml_moe_phaseb_complete() ? 1 : 0,
                                      ggml_moe_phaseb_dropped_records(),
                                      ggml_moe_phaseb_buffer_used());
        }
    }
    // ★Phase A flush — 락 밖에서(내부에서 metrics_write_record 가 이 mutex를 다시 잡는다.
    //   락 순서 역전 방지: phasea → direct 방향만 허용).
    ggml_moe_phasea_flush();
    // ★섀도 사이드카 flush — Phase A 뒤(같은 boot_id 를 쓰므로 PA 가 먼저 해소돼 있어야 한다).
    //   OFF 면 내부에서 즉시 반환한다(어떤 상태도 만들지 않음).
    ggml_moe_shadow_flush();
    // ★lead-time 사이드카 flush(사양 §2 "flush: dispatcher drain 후·once 래치") — 위 CUT ①~⑧ ·
    //   IO3 flush · PA/섀도 flush 가 모두 끝난 뒤다. IO3 와 같은 boot_id 를 쓰므로 IO3 쪽이 먼저
    //   닫혀 있어야 조인 대상이 확정되고, 이 시점엔 dispatcher/worker 가 정지해 in-flight 스탬프가
    //   없다. 게이트 OFF 면 내부에서 즉시 반환한다(g_mod.mutex 밖 — 파일 I/O 를 락 안에서 하지 않는다).
    ggml_moe_leadtime_flush();
    // ★GPU-early(C 축) 사이드카 flush(PROBE_GPU_EARLY_SPEC_R1 §4) — lead-time flush **뒤**다.
    //   이 시점엔 dispatcher/worker 와 graph 스레드가 모두 정지해 있으므로 in-flight 커밋이 없고,
    //   남아 있는 pending 이 있으면 그것이 곧 "소비되지 못한 launch" 이므로 헤더의
    //   `pending_unconsumed_count` 로 정직하게 내보낸다(§3 유효 조건 = 0).
    //   게이트 OFF 면 내부에서 즉시 반환한다(g_mod.mutex 밖 — 파일 I/O 를 락 안에서 하지 않는다).
    ggml_moe_gpuearly_flush();
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.metrics.enabled || g_mod.metrics.finalized) return;
    metrics_write_snapshot_locked("summary");
    g_mod.metrics.finalized = true;
}

// ===========================================================================
// §13. 테스트 전용 진입점
// ===========================================================================

// R-B3b (RB3B_ATOM_SPEC.json): the 58 test entry points of this section moved to
// ggml-moe-direct-test-entry.cpp, verbatim and in source order.  The banner stays here as an
// empty shell so that every later banner ordinal keeps its cluster binding.

// ===========================================================================
// §13.1 ★B1 영속 dispatcher 테스트/조회 진입점 (PHASEB_DESIGN §7 착지 관문 계측 창구)
// ===========================================================================

// R-B3b (RB3B_ATOM_SPEC.json): the 30 test entry points of this section moved to
// ggml-moe-direct-test-entry.cpp, verbatim and in source order.  The banner stays here as an
// empty shell so that every later banner ordinal keeps its cluster binding.

// ===========================================================================
// ★§14. 섀도 드래프트 캐시 view (SPEC_SHADOW_DRAFT §3 C3)
//
//   선언은 ggml-moe-shadow.h 에 있고 구현이 여기 있는 이유: view 는 record_table/slots/tick/
//   dispatcher 를 직접 읽어야 하며 그 상태의 소유자가 이 모듈이기 때문이다. 섀도 모듈이
//   내부 구조를 헤더로 넘겨받는 대신, 소유자가 읽기 전용 창구를 제공하는 형태를 택했다.
//
//   진입/탈출 등식(C3): slot owner/generation/last_used/lease/class · record table · tick ·
//   pending I/O · control queue · cache/read-order digest 가 전부 동일해야 한다. 이 등식은
//   view_end 가 직접 재검증하고, 위반 시 음수를 돌려 fail-close 한다.
// ===========================================================================

// C3 등식 판정용 접기 — test_digests 의 LRU/tick digest 와 **같은 순서·같은 필드**를 쓰되
// cache_class 까지 포함한다(C3 는 class 불변도 요구하므로 등식 대상에 넣어야 한다).
// (extern "C" 블록 안이라 namespace 대신 static 내부 링키지로 둔다.)
static uint64_t shadow_cache_digest_locked() {
    uint64_t d = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < g_mod.record_table.size(); i++) {
        const uint32_t idx = (uint32_t)i;
        const int32_t  sl  = g_mod.record_table[i].slot_index.load(std::memory_order_relaxed);
        const uint64_t gen = g_mod.record_table[i].slot_generation.load(std::memory_order_relaxed);
        d = fnv1a64_bytes(d, &idx, sizeof(idx));
        d = fnv1a64_bytes(d, &sl,  sizeof(sl));
        d = fnv1a64_bytes(d, &gen, sizeof(gen));
    }
    for (size_t i = 0; i < g_mod.slots.size(); i++) {
        const uint32_t idx = (uint32_t)i;
        const int64_t  own = g_mod.slots[i].owner_record;
        const uint64_t lu  = g_mod.slots[i].last_used;
        const int32_t  ls  = g_mod.slots[i].lease.load(std::memory_order_relaxed);
        const uint8_t  cc  = g_mod.slots[i].cache_class;
        d = fnv1a64_bytes(d, &idx, sizeof(idx));
        d = fnv1a64_bytes(d, &own, sizeof(own));
        d = fnv1a64_bytes(d, &lu,  sizeof(lu));
        d = fnv1a64_bytes(d, &ls,  sizeof(ls));
        d = fnv1a64_bytes(d, &cc,  sizeof(cc));
    }
    d = fnv1a64_bytes(d, &g_mod.tick, sizeof(g_mod.tick));
    return d;
}

int ggml_moe_shadow_view_begin(void) {
    if (!g_mod.sealed.load(std::memory_order_acquire)) return -1;
    if (g_mod.shadow_view_open.load(std::memory_order_acquire) != 0) return -2;   // 중첩 금지

    // ── ① dispatcher drain — 상태 큐를 보류함으로 내리고 in-flight 가 0 인지 확인한다.
    //     섀도는 main graph 가 완주·synchronize 된 뒤 동기 실행되므로 정상 상태에서 0 이다.
    //     0 이 아니면 동결이 성립하지 않으므로 view 를 열지 않는다(조용한 통과 금지).
    if (persistent_active()) {
        // prefetch in-flight 소진 + intent 큐 소진까지 유한 대기(결정론 확보).
        if (ggml_moe_direct_test_pb_quiesce(2000) != 1) return -3;
    }
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);
        pb_drain_status_locked();
        if (g_mod.demand_outstanding != 0) return -3;
        if (g_mod.submit_head < g_mod.submit_q.size()) return -3;

        // ── ② (record, slot, generation, ptr) view 구성 ──
        const size_t n_rec = g_mod.record_table.size();
        g_mod.shadow_view_slot.assign(n_rec, -1);
        g_mod.shadow_view_gen .assign(n_rec, 0);
        for (size_t i = 0; i < n_rec; i++) {
            const int32_t si = g_mod.record_table[i].slot_index.load(std::memory_order_relaxed);
            if (si < 0) continue;
            const Slot & s = g_mod.slots[(size_t)si];
            const uint64_t want = g_mod.record_table[i].slot_generation.load(std::memory_order_relaxed);
            if (s.generation.load(std::memory_order_relaxed) != want) continue;
            if (s.owner_record != (int64_t)i) continue;
            g_mod.shadow_view_slot[i] = si;
            g_mod.shadow_view_gen [i] = want;
        }

        // ── ③ 진입 스냅샷(탈출 등식의 좌변) ──
        g_mod.shadow_view_tick_begin     = g_mod.tick;
        g_mod.shadow_view_digest_begin   = shadow_cache_digest_locked();
        g_mod.shadow_view_readfile_begin = g_mod.io_submit_ops;
        g_mod.shadow_view_refused        = 0;

        // ── ④ mutation epoch 동결 — 이 store 이후 plan_demand/pb_admit 은 거부된다 ──
        g_mod.shadow_view_open.store(1, std::memory_order_release);
    }
    return 0;
}

int ggml_moe_shadow_view_end(void) {
    if (g_mod.shadow_view_open.load(std::memory_order_acquire) == 0) return -1;

    int rc = 0;
    {
        std::lock_guard<std::mutex> lock(g_mod.mutex);

        // ── ① generation 재검증 — 스냅샷의 전 항목이 그대로여야 한다 ──
        for (size_t i = 0; i < g_mod.shadow_view_slot.size(); i++) {
            const int32_t si = g_mod.shadow_view_slot[i];
            if (si < 0) continue;
            const Slot & s = g_mod.slots[(size_t)si];
            if (s.generation.load(std::memory_order_relaxed) != g_mod.shadow_view_gen[i]) { rc = -2; break; }
            if (s.owner_record != (int64_t)i)                                             { rc = -2; break; }
        }

        // ── ② C3 진입/탈출 등식 ──
        if (rc == 0 && g_mod.tick       != g_mod.shadow_view_tick_begin)     rc = -3;
        if (rc == 0 && shadow_cache_digest_locked() != g_mod.shadow_view_digest_begin) rc = -4;
        if (rc == 0 && g_mod.io_submit_ops != g_mod.shadow_view_readfile_begin) rc = -5;  // 읽기 제출 0
        if (rc == 0 && g_mod.shadow_view_refused != 0)                       rc = -6;

        // ── ③ 동결 해제 + 스냅샷 폐기(실패해도 반드시 닫는다 — 영구 동결 금지) ──
        g_mod.shadow_view_open.store(0, std::memory_order_release);
        g_mod.shadow_view_slot.clear();
        g_mod.shadow_view_gen .clear();
        // ★r2 M1 — shrink_to_fit() 제거. 이 함수는 ShadowScopeGuard 의 **noexcept 소멸**
        //   정리 중에도 불리는데, shrink_to_fit 은 재할당을 시도하므로 bad_alloc 를
        //   던질 수 있다 — 그 자리에서 terminate 로 간다. clear() 로 남기는 용량은
        //   다음 view 가 그대로 재사용하므로 실질 누수도 아니다(record 수는 seal 고정).
        //   (capacity 반납이 필요하면 reset_for_test 에서 한다 — 거긴 예외 문맥이 아니다.)
    }
    return rc;
}

bool ggml_moe_shadow_view_active(void) {
    return g_mod.shadow_view_open.load(std::memory_order_acquire) != 0;
}

int32_t ggml_moe_shadow_view_present_mask(int32_t mlpos, const int32_t * experts, int32_t n,
                                          uint16_t * out_mask) {
    if (out_mask) *out_mask = 0;
    if (!experts || n <= 0 || n > 16) return -1;
    if (g_mod.shadow_view_open.load(std::memory_order_acquire) == 0) return -2;
    if (!g_mod.sealed.load(std::memory_order_acquire)) return -3;
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.moe_layers.size()) return -4;

    // ★동결 중이므로 스냅샷 벡터에 쓰는 주체가 없다 — 락 없이 읽는다(tick-중립·읽기 제출 0).
    uint16_t mask = 0;
    int32_t  present = 0;
    for (int32_t i = 0; i < n; i++) {
        const int32_t eid = experts[i];
        if (eid < 0 || eid >= g_mod.n_expert) continue;
        const size_t record_idx = (size_t)mlpos * (size_t)g_mod.n_expert + (size_t)eid;
        if (record_idx >= g_mod.shadow_view_slot.size()) continue;
        if (g_mod.shadow_view_slot[record_idx] >= 0) {
            mask |= (uint16_t)(1u << i);
            present++;
        }
    }
    if (out_mask) *out_mask = mask;
    return present;
}

const void * ggml_moe_shadow_view_ptr(const struct ggml_tensor * t, int32_t expert_id) {
    if (!t) return nullptr;
    if (g_mod.shadow_view_open.load(std::memory_order_acquire) == 0) return nullptr;
    if (!g_mod.sealed.load(std::memory_order_acquire)) return nullptr;
    auto it = g_mod.registry.find(t);
    if (it == g_mod.registry.end()) return nullptr;
    const int32_t mlpos    = it->second.mlpos;
    const int32_t part_idx = it->second.part_idx;
    if (expert_id < 0 || expert_id >= g_mod.n_expert) return nullptr;

    const size_t record_idx = (size_t)mlpos * (size_t)g_mod.n_expert + (size_t)expert_id;
    if (record_idx >= g_mod.shadow_view_slot.size()) return nullptr;
    const int32_t si = g_mod.shadow_view_slot[record_idx];
    if (si < 0) return nullptr;
    const Slot & s = g_mod.slots[(size_t)si];
    // 스냅샷 generation 과 현재 generation 이 다르면 동결이 깨진 것이다 — 포인터를 주지 않는다.
    if (s.generation.load(std::memory_order_acquire) != g_mod.shadow_view_gen[record_idx]) return nullptr;
    return (const char *)s.base + g_mod.layers[(size_t)mlpos].parts[(size_t)part_idx].part_offset;
}

void ggml_moe_shadow_state_digest(uint64_t * out_cache_digest, uint64_t * out_tick,
                                  uint64_t * out_readfile_ops, uint64_t * out_pending) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (out_cache_digest) *out_cache_digest = g_mod.sealed.load(std::memory_order_relaxed)
                                            ? shadow_cache_digest_locked() : 0;
    if (out_tick)         *out_tick         = g_mod.tick;
    if (out_readfile_ops) *out_readfile_ops = g_mod.io_submit_ops;
    if (out_pending)      *out_pending      = (uint64_t)g_mod.demand_outstanding;
}

uint64_t ggml_moe_shadow_layer_stride_bytes(int32_t mlpos) {
    std::lock_guard<std::mutex> lock(g_mod.mutex);
    if (!g_mod.sealed.load(std::memory_order_relaxed)) return 0;
    if (mlpos < 0 || mlpos >= (int32_t)g_mod.layers.size()) return 0;
    return g_mod.layers[(size_t)mlpos].stride_bytes;
}

int32_t ggml_moe_shadow_mlpos_of(const struct ggml_tensor * t) {
    if (!t) return -1;
    if (!g_mod.sealed.load(std::memory_order_acquire)) return -1;
    auto it = g_mod.registry.find(t);
    if (it == g_mod.registry.end()) return -1;
    return it->second.mlpos;
}

} // extern "C"

#else // !_WIN32 — 비Windows 스텁

extern "C" {

bool ggml_moe_direct_enabled(void) { return false; }
bool ggml_moe_no_prefetch(void) { return false; }
int  ggml_moe_direct_register(const struct ggml_tensor *, const char *, int32_t) { return -1; }
int  ggml_moe_direct_seal(const struct ggml_moe_direct_model *, const struct ggml_moe_direct_source *, int32_t) { return -1; }
void ggml_moe_direct_attestation(int32_t * a, int32_t * b, int32_t * c, int32_t * d, int32_t * e) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0;
}
void ggml_moe_direct_scope_attestation(int32_t * a, int32_t * b, int32_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
bool ggml_moe_direct_is_registered(const struct ggml_tensor *) { return false; }
int  ggml_moe_direct_acquire(const struct ggml_tensor *, const int32_t *, int32_t, int32_t) { return -1; }
void ggml_moe_direct_release(const struct ggml_tensor *, const int32_t *, int32_t) {}
const void * ggml_moe_direct_ptr(const struct ggml_tensor *, int32_t) { return NULL; }
int  ggml_moe_direct_part_info(const struct ggml_tensor *, uint64_t *, uint64_t *) { return -1; }
int  ggml_moe_direct_registered_layer_part(const struct ggml_tensor *, int32_t *, int32_t *) { return -1; }
int  ggml_moe_direct_emit_expert_copies(const struct ggml_tensor *, uint64_t, uint64_t, int32_t,
                                         const int32_t *, int32_t,
                                         void (*)(void *, const void *, uint64_t, uint64_t), void *) { return -1; }
uint64_t ggml_moe_direct_touch_events(void) { return 0; }
uint64_t ggml_moe_direct_touch_pages(void) { return 0; }
uint64_t ggml_moe_direct_fallback_count(void) { return 0; }
void ggml_moe_direct_phase_read_bytes_hit_miss(int32_t, uint64_t * a, uint64_t * b, uint64_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
void ggml_moe_direct_phase_read_bytes_hit_miss_exact(int32_t, uint64_t * a, uint64_t * b, uint64_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
void ggml_moe_direct_phase_read_bytes_hit_miss_window(int32_t, uint64_t * a, uint64_t * b, uint64_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
void ggml_moe_direct_fallback_hit(const char *) {}
bool ggml_moe_direct_metrics_write_record(const char *, const char *) { return false; }
int32_t ggml_moe_direct_absent_count(int32_t, const int32_t *, int32_t) { return 0; }
void ggml_moe_direct_graph_shape_note(int32_t, int32_t) {}
void ggml_moe_prefetch_marker(uint64_t, uint32_t, uint32_t) {}
void ggml_moe_direct_finalize(void) {}
int  ggml_moe_direct_seal_for_test(const char *, const struct ggml_moe_direct_model *,
                                   const struct ggml_moe_direct_source *, int32_t, uint64_t, uint32_t,
                                   const char *, const char *, const char *, const char *) { return -1; }
void ggml_moe_direct_reset_for_test(void) {}
const char * ggml_moe_direct_openarch_probe_json(void) { return "{\"abi\":null,\"templates\":[]}"; }
int  ggml_moe_direct_test_template_inventory_sha256(char * out, int32_t cap) {
    if (!out || cap <= 0) return -1;
    out[0] = '\0';
    return 0;
}
int32_t ggml_moe_direct_test_v3_stage(void) { return 0; }
int  ggml_moe_direct_test_v3_probe(int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                                   uint64_t *, uint64_t *, uint64_t *) { return -1; }
int  ggml_moe_direct_test_v3_probe_part(int32_t, uint64_t *, uint64_t *) { return -1; }
void ggml_moe_direct_test_seal_residue(int32_t * a, int32_t * b, uint64_t * c, int32_t * d, int32_t * e,
                                       int32_t * f, int32_t * g, int32_t * h, int32_t * i,
                                       int32_t * j, int32_t * k) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0;
    if (g) *g = 0; if (h) *h = 0; if (i) *i = 0; if (j) *j = 0; if (k) *k = 0;
}
void ggml_moe_direct_test_v3_wave_stats(uint64_t * a, int32_t * b) { if (a) *a = 0; if (b) *b = 0; }
void ggml_moe_direct_test_v3_child_stats(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
void ggml_moe_direct_test_v3_arm(int32_t, int32_t, int32_t) {}
void ggml_moe_direct_test_v3_arm_sync_child(int32_t) {}
void ggml_moe_direct_test_v3_bounce_stats(uint64_t * a, uint64_t * b, uint64_t * c, int32_t * d, int32_t * e) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0;
}
void ggml_moe_direct_test_v3_arm_copy(int32_t, int32_t) {}
void ggml_moe_direct_test_v3_coalesce_stats(int32_t * a, uint64_t * b, uint64_t * c, int32_t * d,
                                            int32_t * e, uint64_t * f, uint64_t * g, uint64_t * h,
                                            int32_t * i, uint32_t * j, uint64_t * k, int32_t * l,
                                            uint64_t * m) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0;
    if (f) *f = 0; if (g) *g = 0; if (h) *h = 0; if (i) *i = 0; if (j) *j = 0;
    if (k) *k = 0; if (l) *l = 0; if (m) *m = 0;
}
void ggml_moe_direct_test_v3_arm_coalesce(int32_t, int32_t) {}
void ggml_moe_direct_test_v3_set_io3_cap(uint64_t) {}
void ggml_moe_direct_test_v3_io_stats(int32_t * a, uint8_t * b, uint64_t * c, uint64_t * d,
                                      uint64_t * e, uint64_t * f, int32_t * g) {
    if (a) for (int i = 0; i < 6; i++) a[i] = 0;
    if (b) for (int i = 0; i < 6; i++) b[i] = 0;
    if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0; if (g) *g = 0;
}
void ggml_moe_direct_test_pbv_stats(int32_t * a, int32_t * b, int32_t * c, int32_t * d, int32_t * e,
                                    uint64_t * f, uint64_t * g, int32_t * h) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
    if (e) *e = 0; if (f) *f = 0; if (g) *g = 0; if (h) *h = 0;
}
int32_t ggml_moe_direct_test_pbv_ncap(int32_t, int32_t, int32_t * out_wave) {
    if (out_wave) *out_wave = 0;
    return -1;
}
void ggml_moe_direct_test_pbv_wake_state(int32_t * a, int32_t * b, int32_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
int32_t ggml_moe_direct_test_io3_children(int32_t, uint64_t *, uint64_t *, uint8_t *) { return 0; }
int32_t ggml_moe_direct_test_io3_copies(int32_t, uint64_t *, uint64_t *, uint64_t *,
                                        uint8_t *, uint8_t *) { return 0; }
void ggml_moe_direct_test_last_seal_error(char * buf, int32_t cap) { if (buf && cap > 0) buf[0] = '\0'; }
void ggml_moe_direct_test_v3_arm_dispatcher_fail(int32_t) {}
int32_t ggml_moe_direct_test_v3_retire_order(int32_t *, int32_t) { return 0; }
void ggml_moe_direct_test_v3_sync_stats(uint64_t * a, uint64_t * b) { if (a) *a = 0; if (b) *b = 0; }
void ggml_moe_direct_test_read_stats(uint64_t * a, uint64_t * b) { if (a) *a = 0; if (b) *b = 0; }
void ggml_moe_direct_test_lease_stats(int32_t * a, int64_t * b) { if (a) *a = 0; if (b) *b = 0; }
void ggml_moe_direct_test_record_slot(int64_t, int32_t * a, uint64_t * b) { if (a) *a = -1; if (b) *b = 0; }
void ggml_moe_direct_test_slot_state(int32_t, int64_t * a, uint64_t * b, uint64_t * c, int32_t * d) {
    if (a) *a = -1; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
int  ggml_moe_direct_test_mapped_view_roundtrip(void *, size_t) { return -1; }
void ggml_moe_direct_test_dispatcher_stats(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d,
                                            uint64_t * e, uint64_t * f, uint64_t * g, uint64_t * h,
                                            int32_t * i) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
    if (e) *e = 0; if (f) *f = 0; if (g) *g = 0; if (h) *h = 0; if (i) *i = 0;
}
void ggml_moe_direct_test_mem_echo(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d,
                                    uint64_t * e, uint64_t * f, uint64_t * g, uint64_t * h) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
    if (e) *e = 0; if (f) *f = 0; if (g) *g = 0; if (h) *h = 0;
}
void ggml_moe_direct_test_digests(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
void ggml_moe_direct_test_channel_event_state(int32_t * a, int32_t * b, int32_t * c, uint64_t * d, uint64_t * e) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0;
}
void ggml_moe_direct_test_force_sync_next(int32_t) {}
void ggml_moe_direct_test_dispatcher_pause(int32_t) {}
void ggml_moe_direct_test_abort_after_submit(int32_t) {}
void ggml_moe_direct_test_shutdown(int32_t) {}
void ggml_moe_direct_test_pb_counts(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d,
                                     int32_t * e, int32_t * f, int32_t * g) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0; if (g) *g = 0;
}
int  ggml_moe_direct_test_pb_sample(uint64_t, void *, uint64_t) { return -1; }
int  ggml_moe_direct_test_pb_io(uint64_t, void *, uint64_t) { return -1; }
void ggml_moe_direct_test_slot_class(int32_t, int32_t * a, uint64_t * b, int32_t * c) { if (a) *a = 0; if (b) *b = 0; if (c) *c = -1; }
void ggml_moe_direct_test_pb_fault_next(int32_t) {}
int  ggml_moe_direct_test_pb_quiesce(int32_t) { return 1; }
void ggml_moe_direct_test_hold_retire(int32_t) {}
void ggml_moe_direct_test_hold_submit(int32_t) {}
int  ggml_moe_direct_test_shutdown_bounded(int32_t, int32_t) { return 1; }
void ggml_moe_direct_test_pbv_credit_totals(uint64_t * a, uint64_t * b) { if (a) *a = 0; if (b) *b = 0; }
void ggml_moe_direct_test_pbv_cont_defer_events(uint64_t * a) { if (a) *a = 0; }
void ggml_moe_direct_test_pbv_admit_skipped_cont(uint64_t * a) { if (a) *a = 0; }
void ggml_moe_direct_test_pbv_admit_skip_witness(int32_t * a, int32_t * b, int32_t * c, int32_t * d, int32_t * e) { if (a) *a = -1; if (b) *b = -1; if (c) *c = -1; if (d) *d = -1; if (e) *e = -1; }
void ggml_moe_direct_test_pbv_admit_skip_admitted(uint64_t * a) { if (a) *a = 0; }
void ggml_moe_direct_test_cont_credit_fail_next(int32_t) {}
int  ggml_moe_direct_test_q_stopping(void) { return 0; }
int  ggml_moe_direct_test_stop_timed_out(void) { return 0; }
void ggml_moe_direct_test_v3_cancel_stats(uint64_t * a, uint64_t * b) { if (a) *a = 0; if (b) *b = 0; }
int  ggml_moe_direct_test_stop_race_arm(int32_t) { return -1; }
void ggml_moe_direct_test_stop_race_release(void) {}
int32_t ggml_moe_direct_test_cut_residual(void) { return 0; }
void    ggml_moe_direct_test_force_cut_residual(int32_t) {}
int32_t ggml_moe_direct_test_drain_status(void) { return 0; }
int32_t ggml_moe_direct_test_pending_reuse_probe(void) { return -1; }

// E-2a new boundary surface - non-Windows stubs. (E-2b-0 added the acquire `part`.)
int ggml_moe_store_acquire(int32_t, int32_t, const int32_t *, int32_t, int32_t, uint64_t *) { return -1; }
int ggml_moe_store_view(uint64_t, int32_t, int32_t, struct ggml_moe_buffer_view *) { return -1; }
void ggml_moe_store_release(uint64_t) {}
int ggml_moe_store_emit_expert_copies(uint64_t, int32_t, uint64_t, uint64_t, int32_t,
                                      const int32_t *, int32_t,
                                      void (*)(void *, const void *, uint64_t, uint64_t),
                                      void *) { return -1; }
int ggml_moe_adapter_resolve(const struct ggml_tensor *, int32_t *, int32_t *) { return -1; }
uint64_t ggml_moe_store_epoch(void) { return 0; }
int32_t ggml_moe_store_state(void) { return MOE_STORE_STOPPED; }
void ggml_moe_store_outstanding(int64_t * a, int64_t * b, int64_t * c) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0;
}
int ggml_moe_store_drain_begin(void) { return -1; }
int ggml_moe_store_drain_finish(int32_t) { return -1; }
int ggml_moe_direct_test_xstore_token(uint64_t, int32_t *, int32_t *, int32_t *, uint64_t *) { return -1; }
// E-2b-0 non-Windows stubs.
int ggml_moe_direct_test_xstore_token_part(uint64_t, int32_t *) { return -1; }
int ggml_moe_direct_test_xstore_table(uint64_t *, int32_t *, int32_t *, int32_t *) { return -1; }
void ggml_moe_direct_test_mutex_hold_arm(int32_t) {}
int32_t ggml_moe_direct_test_mutex_hold_wait(int32_t) { return 0; }
void ggml_moe_direct_test_mutex_hold_release(void) {}
int32_t ggml_moe_direct_test_mutex_hold_safety_fired(void) { return 0; }
int32_t ggml_moe_direct_test_xstore_drain_remaining(int32_t) { return -1; }
int32_t ggml_moe_direct_test_xstore_drain_zero_ready(int64_t * a, int64_t * b) {
    if (a) *a = 0;
    if (b) *b = 0;
    return 1;
}
void ggml_moe_direct_test_xstore_arm_last_generation(void) {}
int32_t ggml_moe_direct_test_stop_join_stage(int64_t * a, int32_t * b) {
    if (a) *a = -1;
    if (b) *b = -1;
    return 0;
}
int64_t ggml_moe_direct_test_steady_ms(void) { return 0; }

// ★섀도 캐시 view (SPEC_SHADOW_DRAFT §3 C3) — 비Windows 스텁.
int  ggml_moe_shadow_view_begin(void) { return -1; }
int  ggml_moe_shadow_view_end(void)   { return -1; }
bool ggml_moe_shadow_view_active(void) { return false; }
int32_t ggml_moe_shadow_view_present_mask(int32_t, const int32_t *, int32_t, uint16_t * m) {
    if (m) *m = 0;
    return -1;
}
const void * ggml_moe_shadow_view_ptr(const struct ggml_tensor *, int32_t) { return nullptr; }
void ggml_moe_shadow_state_digest(uint64_t * a, uint64_t * b, uint64_t * c, uint64_t * d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
uint64_t ggml_moe_shadow_layer_stride_bytes(int32_t) { return 0; }
int32_t  ggml_moe_shadow_mlpos_of(const struct ggml_tensor *) { return -1; }

} // extern "C"

#endif // _WIN32
