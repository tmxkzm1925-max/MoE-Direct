#include "selftest-harness.h"
#include "selftest-suites.h"

#define LLAMA_MEMPLAN_C_ONLY
#include "llama-memplan.h"

// ===========================================================================
// P41. ★레버 A 단계 1 S1 — immutable ubatch memory plan (memplan)
//   SPEC_S1_MEMPLAN.md v1.0 §4 픽스처 ⓐ~ⓛ × top-level 3계열(standard KV·hybrid·DSV4).
//   실 memory 객체(llama_kv_cache / llama_memory_hybrid / llama_kv_cache_dsv4)를 세워야 하는데
//   그 심볼들이 DLL 밖으로 나오지 않으므로, 픽스처 본체는 llama-memplan.cpp 안에서 돌고
//   이 하네스는 PASS/FAIL 회계·섹션 명명·총계를 그대로 소유한다(라이브 서버·GPU 불요).
// ===========================================================================

static void memplan_s1_cb(void * ud, int ok, const char * name) {
    (void) ud;
    check(ok != 0, name ? name : "(unnamed memplan check)");
}

void test_memplan_s1() {
    section("P41. ★레버 A 단계 1 S1 — immutable ubatch memory plan(memplan): "
            "3계열(standard KV·hybrid·DSV4) × SPEC §4 ⓐ~ⓛ");

    const int before_total  = g_total;
    const int before_failed = g_failed;

    const int engine_failed = llama_memplan_selfcheck(memplan_s1_cb, nullptr);

    const int harness_total  = g_total  - before_total;
    const int harness_failed = g_failed - before_failed;

    check(harness_total > 0 && engine_failed == harness_failed,
          "P41z. ★회계 일치 — llama_memplan_selfcheck 가 돌려준 실패 수와 하네스가 센 FAIL 수가 같다"
          "(콜백 누락·이중 계수 0)");
}
