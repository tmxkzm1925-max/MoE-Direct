#pragma once

// R-B1 (RB_DESIGN.md 2-2, RB1_ATOM_SPEC.json): definitions moved out of
// ggml-moe-direct.cpp by the pure-TU extraction atom.  Every definition below is the
// core text at the span the move map names, moved verbatim - body tokens unchanged.
// Windows only, exactly like the region it came from: the core TU compiles its
// non-Windows half from the stub region at the bottom of that file, and this unit has
// no stub half of its own (the moved definitions have no stub counterparts).

#ifdef _WIN32

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace ggml_moe_direct_impl {

struct QuantTrait { const char * type; uint32_t block_values; uint32_t block_bytes; ggml_type gt; };

struct ExpectCatalogEntry { const char * profile_id; const char * approved_sha256; const char * scope; };

enum class TplBias      { FORBIDDEN, REQUIRED };

enum class TplLayerRule { ALL, LEADING_DENSE };

enum class TplNextn     { FORBIDDEN, EXECUTION_DEFAULT };

struct ArchTemplateEntry {
    const char * template_id;       // = the exact GGUF arch value (distinct from internal adapter names)
    const char * template_version;
    TplBias      bias;
    TplLayerRule layer_rule;
    TplNextn     nextn;
};

struct PartInfo {
    std::string name;
    std::string source_tensor;
    int32_t     source_index = 0;
    std::string type;
    std::vector<int64_t> dims;
    int32_t     expert_axis = 0;
    uint64_t    part_offset = 0;
    uint64_t    part_bytes = 0;
    // ---- ★§Z-⑥ mode=virtual(v3) 전용 좌표(SPEC_REPACK_V3 §2-2·§2-4). bin 경로에서는 전부
    //      기본값으로 남는다 — v2 manifest 에는 이 필드들이 아예 없다. ----
    uint64_t    abs_offset = 0;     // shard 내 텐서 데이터 시작 절대 오프셋
    uint64_t    slice_bytes = 0;    // per-expert per-part
    uint64_t    region = 0;         // align_up(bracket_head + slice, A) — 파생값(재계산)
    uint64_t    slot_offset = 0;
    uint64_t    data_offset = 0;    // ptr()/part_info() 반환 기준
    uint64_t    bracket_head = 0;   // aligned part 전용(비4K 는 미정의)
    uint64_t    staging_bytes = 0;  // 비4K part 전용
    bool        aligned = false;
};

struct LayerInfo {
    int32_t  layer = 0;         // 실제 layer id
    uint64_t payload_bytes = 0;
    uint64_t stride_bytes = 0;  // ★층별
    uint64_t record_base = 0;
    uint64_t layer_slot_bytes = 0;  // ★virtual 전용(= Σ region) — bin 의 stride_bytes 대응축
    // ★§Z-⑥ A3 — virtual 전용 **legacy-equivalent record bytes**(§7 D-A2 분자·§Z-③
    //   `legacy_align_bytes` 축). 물리 바이트(Σ region)가 아니라 "같은 record 를 bin 으로
    //   재팩했다면 읽었을 바이트"다 — 회계는 이 축으로만 선다(§3-8 parent logical row).
    uint64_t legacy_record_bytes = 0;
    // ★§Z-⑥ A4 — 이 층의 child 가 **슬롯에 남길** 바이트 합. aligned=region(브라켓 전량이
    //   슬롯에 앉는다) · 비4K=slice_bytes(브라켓은 staging 으로 가고 copy 가 슬라이스만 옮긴다).
    //   A3 까지는 전부 aligned 라 layer_slot_bytes 와 같았고, 혼합 층에서 처음 갈린다.
    uint64_t child_bytes_expected = 0;
    std::vector<PartInfo> parts;
};

struct SourceInfo {
    int32_t  index = 0;
    std::string path;
    uint64_t bytes = 0;
    int64_t  mtime_epoch = 0;
    uint32_t gguf_version = 0;
    uint32_t alignment = 0;
    uint64_t data_start = 0;
    // ★DF-1(SPEED_LEVER_LEDGER · SPEC_REPACK_V3 부속 정오 1 기술 귀결) — virtual 에서는 원본
    //   GGUF 가 런타임 데이터 원천이라 그 무결성이 무손실 사슬에 편입된다. 주소 권위 영역
    //   [0, data_start) 의 digest 이며, 값 대조는 부팅 프로브가 재해시로 행사한다.
    std::string digest_sha256;      // virtual 전용(bin=빈 문자열)
    uint64_t    digest_header_bytes = 0;
};

enum class ManifestMode { BIN, VIRTUAL };

struct ParsedManifest {
    std::string arch;
    int32_t n_layer = 0, n_expert = 0, n_expert_used = 0;
    std::vector<int32_t> moe_layers;
    std::vector<LayerInfo> layers;
    std::vector<SourceInfo> sources;
    uint64_t align_bytes = 0, slot_stride_max = 0, bin_file_bytes = 0;
    std::string ref_profile_id, ref_expect_sha256;
    std::string routed_scope = "all"; // ★부록A: manifest.model.routed_scope(부재=all·"all"|"execution"만)
    // ---- ★§Z-⑥ mode=virtual(v3) 전용 ----
    bool     virtual_mode = false;
    uint64_t virtual_payload_bytes = 0;
    uint64_t legacy_align_bytes = 0;      // §Z-③ D-A2 분자축(source 볼륨 A 와 독립) [[C:repack.legacy-align]]
    std::string legacy_align_source;      // "paired_v2" | "canonical_4096"
    int64_t  n_records = 0;
};

} // namespace ggml_moe_direct_impl

#endif // _WIN32
