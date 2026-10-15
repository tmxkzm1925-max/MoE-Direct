#pragma once

// R-B1 (RB_DESIGN.md 2-2, RB1_ATOM_SPEC.json): definitions moved out of
// ggml-moe-direct.cpp by the pure-TU extraction atom.  Every definition below is the
// core text at the span the move map names, moved verbatim - body tokens unchanged.
// Windows only, exactly like the region it came from: the core TU compiles its
// non-Windows half from the stub region at the bottom of that file, and this unit has
// no stub half of its own (the moved definitions have no stub counterparts).

#ifdef _WIN32

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

#include "ggml.h"
#include "ggml-moe-direct.h"
#include "ggml-moe-direct-types.h"

namespace ggml_moe_direct_impl {

// The core TU spells this type `json` and the moved bodies keep that spelling (body
// tokens are unchanged), so the alias travels with them.  It names the same type as the
// core's own alias, which is why a caller that sees both through the using-directive is
// not ambiguous (measured: same type -> rc 0, different types -> C2872).
using json = nlohmann::json;

inline constexpr QuantTrait QUANT_TRAITS[] = {
    { "F32",   1,   4,   GGML_TYPE_F32   },
    { "F16",   1,   2,   GGML_TYPE_F16   },
    { "MXFP4", 32,  17,  GGML_TYPE_MXFP4 },
    { "Q3_K",  256, 110, GGML_TYPE_Q3_K  },
    { "Q4_K",  256, 144, GGML_TYPE_Q4_K  },
    { "Q5_K",  256, 176, GGML_TYPE_Q5_K  },
    { "Q6_K",  256, 210, GGML_TYPE_Q6_K  },
    { "Q8_0",  32,  34,  GGML_TYPE_Q8_0  },
};

inline constexpr uint32_t PROD_DEFAULT_BUDGET_MB  = 2048;

inline constexpr uint32_t PROD_DEFAULT_TOUCH_ABORT = 64;

inline constexpr int32_t MOE_DIRECT_QD_DEFAULT = 1;

inline constexpr int32_t MOE_DIRECT_QD_MAX     = 63;

inline constexpr ExpectCatalogEntry EXPECT_CATALOG[] = {
    { "gpt-oss-120b",        "be127b3e27454eb369eeca253decf7eab1ac4964849a731d352ffe72d5be828e", "all" },
    { "kimi-k2.6-ram-447gb", "fee9902ca2ed77b4b0c06be49ceb4463fe0ff1d5d42750ed43cae82f162681f6", "all" },
    { "mistral-small-4",     "00a9f67ae8a5844b1ebbf2d82f8f2d869478d953df010f02d8715ad6c7865d7c", "all" }, // 사다리 ①(2026-07-24 실측 재해시 대조)
    { "qwen35-122b",         "982639f39e15f04716415a65b909f5d761884880d1ddab87c04d9734e55bc15f", "all" }, // 사다리 ②(2026-07-24 저장소 실해시·repacker 카탈로그·manifest 3자 대조)
    { "qwen35-122b-nonextn", "882fdac06cf7d75c5d9944f6eb373a204667186cb96929dff605e5f7f9f9a800", "execution" }, // 부록A(2026-07-25 등재): NextN 마지막 1층 제외(routed 147→144)
    { "qwen35-35b",          "616eb096c272ecdc299166afea9c47ebf84f93ae10363c462c84c6c358262d83", "all" }, // 노트북 티어 ⓪(2026-07-28 등재): repacker 카탈로그·expect·실기동 verify_report 3자 대조
    { "qwen35-397b",         "21b698ca0267724ceebbeb4255585dcb24eef9ad7d9d3bad6480833ef3d77f14", "all" }, // 397B(2026-07-29 등재·당일 정정): repacker 카탈로그 등재값 대조 기입. 60층 전층 MoE·routed 180·n_expert=512·MTP/NextN 부재 → scope="all". 정정=메타 전용 shard1 의 data_start 를 upstream 정합(gguf.cpp:756 — n_tensors==0 이면 정렬 seek 없음)으로 수리 → expect 재해시 00ad762c…→21b698ca…
    { "deepseek-v4-flash",   "b2be25b3d3739cdbf97aca583fc993070e8a98a75f1d4b1d7714eb6b6af1ab9b", "all" }, // DeepSeek-V4-Flash(2026-08-01 등재·Codex sealcat r1 [MODIFY 1]에서 이 행 확정 수용): repacker 카탈로그(당일 Codex r1 [ACCEPT] cff054d)·expect 재해시·재팩 manifest reference_lock 3자 대조 일치. 43층 전층 MoE·routed 129 전량 MXFP4·256E·top-6·MTP/NextN 부재(변환기 제외 확인) → scope="all". 재팩 verify 33,024/33,024 PASS. ★minimax-m27 동승 등재는 r1에서 보류 처분(재팩 실물이 26-07-29 재고 정리로 삭제 상태 — manifest 대조 불가. 재팩 재생성 후 재등재)
};

inline constexpr const char * ARCH_TEMPLATE_LOCK_PREFIX = "arch-template:";

inline constexpr ArchTemplateEntry ARCH_TEMPLATE_CATALOG[] = {
    { "gpt-oss",   "1", TplBias::REQUIRED,  TplLayerRule::ALL,           TplNextn::FORBIDDEN         },
    { "qwen35moe", "1", TplBias::FORBIDDEN, TplLayerRule::ALL,           TplNextn::EXECUTION_DEFAULT },
    { "deepseek2", "1", TplBias::FORBIDDEN, TplLayerRule::LEADING_DENSE, TplNextn::FORBIDDEN         },
};

inline constexpr const char * ARCH_TEMPLATE_WEIGHT_KINDS[] = { "gate", "up", "down" };

inline uint64_t align_up(uint64_t v, uint64_t a)   { return (v + a - 1) / a * a; }

inline uint64_t align_down(uint64_t v, uint64_t a) { return v / a * a; }

inline uint64_t fnv1a64_bytes(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

inline bool ck_add(uint64_t a, uint64_t b, uint64_t & out) {
    if (a > (~(uint64_t)0) - b) return false;
    out = a + b; return true;
}

inline bool ck_mul(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > (~(uint64_t)0) / a) return false;
    out = a * b; return true;
}

const char * wire_type_name(ggml_type gt);
std::string to_lower_hex(const std::string & s);
bool sha256_file(const std::wstring & path, std::string & hex_out, std::string & err);
bool catalog_lookup(const std::string & profile_id, const char * override_profile_id,
                    const char * override_digest, const char * override_scope,
                    std::string & approved_digest, std::string & approved_scope);
bool parse_blk_layer(const std::string & name, int32_t & out_layer);
bool parse_routed_tensor_name(const std::string & name, int32_t & out_layer,
                              std::string & out_kind, std::string & out_suffix);
bool inventory_digest_hex(std::vector<std::string> & rows, std::string & out_hex);
bool read_verify_report_gate(const std::wstring & path, std::string & out_pid, std::string & out_sha,
                             std::string & out_manifest_sha, std::string & err);
void log_err(const char * fmt, ...);
[[noreturn]] void fatal_abort(const char * fmt, ...);
std::string getenv_str(const char * name);
bool getenv_flag_is_1(const char * name);
std::wstring utf8_to_wide(const std::string & s);
bool query_a_runtime(const std::wstring & file_path, bool allow_fallback, uint64_t & a_runtime, std::string & err);
bool json_is_uint(const json & j);
bool json_is_int(const json & j);
bool validate_manifest_v2(const json & j,
                          const ggml_moe_direct_model & model,
                          const ggml_moe_direct_source * srcs, int32_t n_srcs,
                          ParsedManifest & out, std::string & err);
bool load_manifest_json(const std::wstring & manifest_path, json & out, std::string & err);
bool detect_manifest_mode(const json & j, ManifestMode & out, std::string & err);
bool virtual_init_reject_check(std::string & err);
bool validate_manifest_v3(const json & j,
                          const ggml_moe_direct_model & model,
                          const ggml_moe_direct_source * srcs, int32_t n_srcs,
                          ParsedManifest & out, std::string & err);
bool read_plan_report_gate(const std::wstring & path, std::string & out_pid, std::string & out_sha,
                           std::string & out_manifest_sha, int64_t & out_records, std::string & err);
bool virtual_probe_sources(const ParsedManifest & pm,
                           std::unordered_map<std::string, int32_t> & out_live_type,
                           std::string & err);

} // namespace ggml_moe_direct_impl

#endif // _WIN32
