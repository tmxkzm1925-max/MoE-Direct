// R-B1 (RB_DESIGN.md 2-2, RB1_ATOM_SPEC.json): definitions moved out of
// ggml-moe-direct.cpp by the pure-TU extraction atom.  Every definition below is the
// core text at the span the move map names, moved verbatim - body tokens unchanged.
// Windows only, exactly like the region it came from: the core TU compiles its
// non-Windows half from the stub region at the bottom of that file, and this unit has
// no stub half of its own (the moved definitions have no stub counterparts).

// The head include block of ggml-moe-direct.cpp, reproduced so the moved definitions compile
// in the environment they were written in (they call into the trace/phasea/phaseb surfaces).
#include "ggml-moe-direct.h"
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
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-moe-direct-types.h"
#include "ggml-moe-direct-manifest.h"

namespace ggml_moe_direct_impl {

bool lookup_quant_trait(const std::string & type, uint32_t & bv, uint32_t & bb) {
    for (const auto & t : QUANT_TRAITS) {
        if (type == t.type) { bv = t.block_values; bb = t.block_bytes; return true; }
    }
    return false;
}

const char * wire_type_name(ggml_type gt) {
    for (const auto & t : QUANT_TRAITS) {
        if (t.gt == gt) return t.type;
    }
    return nullptr;
}

std::string to_lower_hex(const std::string & s) {
    std::string o = s;
    for (auto & c : o) c = (char)tolower((unsigned char)c);
    return o;
}

bool sha256_bytes(const uint8_t * data, size_t len, std::string & hex_out) {
    unsigned char digest[32];
    NTSTATUS st = BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                             (PUCHAR)data, (ULONG)len, digest, sizeof(digest));
    if (st != 0) return false;
    static const char * hx = "0123456789abcdef";
    hex_out.clear();
    hex_out.reserve(64);
    for (int i = 0; i < 32; i++) { hex_out.push_back(hx[digest[i] >> 4]); hex_out.push_back(hx[digest[i] & 0xF]); }
    return true;
}

bool sha256_file(const std::wstring & path, std::string & hex_out, std::string & err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) { err = "cannot open file"; return false; }   // 호출자가 어떤 파일인지 문맥을 붙인다
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (!sha256_bytes(buf.data(), buf.size(), hex_out)) { err = "BCryptHash failed"; return false; }
    return true;
}

bool catalog_lookup(const std::string & profile_id, const char * override_profile_id,
                    const char * override_digest, const char * override_scope,
                    std::string & approved_digest, std::string & approved_scope) {
    if (override_profile_id) {
        if (profile_id == override_profile_id) {
            approved_digest = to_lower_hex(override_digest ? override_digest : "");
            approved_scope  = (override_scope && *override_scope) ? std::string(override_scope) : std::string("all");
            return true;
        }
        return false;
    }
    for (const auto & e : EXPECT_CATALOG) {
        if (profile_id == e.profile_id) { approved_digest = e.approved_sha256; approved_scope = e.scope; return true; }
    }
    return false;
}

bool parse_blk_layer(const std::string & name, int32_t & out_layer) {
    const char * p = name.c_str();
    if (std::strncmp(p, "blk.", 4) != 0) return false;
    p += 4;
    if (!std::isdigit((unsigned char)*p)) return false;
    int64_t v = 0;
    while (std::isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        if (v > 0x7fffffff) return false;
        p++;
    }
    if (*p != '.') return false;
    out_layer = (int32_t)v;
    return true;
}

bool parse_routed_tensor_name(const std::string & name, int32_t & out_layer,
                              std::string & out_kind, std::string & out_suffix) {
    if (!parse_blk_layer(name, out_layer)) return false;
    const size_t dot = name.find('.', 4);          // the '.' after the digits following "blk."
    if (dot == std::string::npos) return false;
    const std::string rest = name.substr(dot + 1); // ffn_<kind>_exps.<suffix>
    static const char * const KINDS[] = { "gate_up", "gate", "up", "down" };
    for (const char * k : KINDS) {
        const std::string pre = std::string("ffn_") + k + "_exps.";
        if (rest.size() > pre.size() && rest.compare(0, pre.size(), pre) == 0) {
            const std::string suf = rest.substr(pre.size());
            if (suf != "weight" && suf != "bias") return false;
            out_kind = k; out_suffix = suf;
            return true;
        }
    }
    return false;
}

bool inventory_digest_hex(std::vector<std::string> & rows, std::string & out_hex) {
    std::sort(rows.begin(), rows.end());
    std::string payload = "MOE-INVENTORY-V1\n";
    for (const auto & r : rows) payload += r;
    return sha256_bytes((const uint8_t *)payload.data(), payload.size(), out_hex);
}

bool read_verify_report_gate(const std::wstring & path, std::string & out_pid, std::string & out_sha,
                             std::string & out_manifest_sha, std::string & err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) { err = "cannot open verify_report.json"; return false; }
    std::string line, last;
    while (std::getline(ifs, line)) {
        std::string t = line;
        while (!t.empty() && (t.back() == '\r' || t.back() == '\n' || t.back() == ' ' || t.back() == '\t')) t.pop_back();
        if (!t.empty()) last = t;
    }
    if (last.empty()) { err = "verify_report.json has no valid record"; return false; }
    try {
        json j = json::parse(last);
        // ① pass==true
        if (!j.contains("pass") || !j["pass"].is_boolean()) { err = "verify_report.pass missing or not a boolean"; return false; }
        if (!j["pass"].get<bool>()) { err = "verify_report.pass=false - artifact failed verification"; return false; }
        // ② pairs 등식(전부 존재·비음 정수·3자 동일)
        static const char * const PAIR_KEYS[3] = { "pairs_total", "pairs_pass", "expected_pairs" };
        uint64_t pv[3] = { 0, 0, 0 };
        for (int i = 0; i < 3; i++) {
            if (!j.contains(PAIR_KEYS[i]) || !j[PAIR_KEYS[i]].is_number_integer()) {
                err = std::string("verify_report.") + PAIR_KEYS[i] + " missing or not an integer"; return false;
            }
            const json & v = j[PAIR_KEYS[i]];
            if (v.is_number_unsigned()) {
                pv[i] = v.get<uint64_t>();
            } else {
                int64_t s = v.get<int64_t>();
                if (s < 0) { err = std::string("verify_report.") + PAIR_KEYS[i] + " is negative"; return false; }
                pv[i] = (uint64_t)s;
            }
        }
        if (pv[0] != pv[1] || pv[1] != pv[2]) {
            err = "verify_report pairs identity mismatch: pairs_total=" + std::to_string(pv[0]) +
                  " pairs_pass=" + std::to_string(pv[1]) + " expected_pairs=" + std::to_string(pv[2]);
            return false;
        }
        // ③ 결함 배열 3종 전부 존재 + 전부 빈 배열
        static const char * const EMPTY_KEYS[3] = { "failures", "padding_failures", "problems" };
        for (int i = 0; i < 3; i++) {
            if (!j.contains(EMPTY_KEYS[i]) || !j[EMPTY_KEYS[i]].is_array()) {
                err = std::string("verify_report.") + EMPTY_KEYS[i] + " missing or not an array"; return false;
            }
            if (!j[EMPTY_KEYS[i]].empty()) {
                err = std::string("verify_report.") + EMPTY_KEYS[i] + " is not empty(" +
                      std::to_string(j[EMPTY_KEYS[i]].size()) + " entries)"; return false;
            }
        }
        // ④ reference_lock
        if (!j.contains("reference_lock") || !j["reference_lock"].is_object()) { err = "verify_report reference_lock missing"; return false; }
        const json & rl = j["reference_lock"];
        if (!rl.contains("profile_id") || !rl["profile_id"].is_string() ||
            !rl.contains("expect_sha256") || !rl["expect_sha256"].is_string()) { err = "verify_report reference_lock fields missing"; return false; }
        out_pid = rl["profile_id"].get<std::string>();
        out_sha = rl["expect_sha256"].get<std::string>();
        // ⑦ ★manifest_sha256(cache key 결속) — 필수·64자리 hex. 값 대조는 호출자(seal).
        if (!j.contains("manifest_sha256") || !j["manifest_sha256"].is_string()) {
            err = "verify_report.manifest_sha256 missing or not a string - report is not cache-key bound"
                  " (re-verify with repack_experts.py --verify-only)"; return false;
        }
        out_manifest_sha = j["manifest_sha256"].get<std::string>();
        if (out_manifest_sha.size() != 64) { err = "verify_report.manifest_sha256 length != 64"; return false; }
        for (char c : out_manifest_sha) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
                err = "verify_report.manifest_sha256 is not hexadecimal"; return false;
            }
        }
    } catch (const std::exception & e) {
        err = std::string("verify_report JSON parse failed: ") + e.what();
        return false;
    }
    return true;
}

void log_err(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[moe-direct] ERROR: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

[[noreturn]] void fatal_abort(const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    GGML_ABORT("[moe-direct] FATAL: %s", buf);
}

std::string getenv_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

bool getenv_flag_is_1(const char * name) {
    return getenv_str(name) == "1";
}

std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

std::string wide_to_utf8(const std::wstring & w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

bool path_is_absolute(const std::string & p) {
    if (p.size() >= 3 && ((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) && p[1] == ':'
        && (p[2] == '\\' || p[2] == '/')) return true;                                                // 드라이브 절대
    if (p.size() >= 2 && (p[0] == '\\' || p[0] == '/') && (p[1] == '\\' || p[1] == '/')) return true; // UNC
    return false;
}

std::wstring normalize_path_w(const std::wstring & in) {
    // \\?\ 확장 접두 제거(GetFinalPathNameByHandleW 유래) 후 GetFullPathNameW + 소문자.
    std::wstring s = in;
    const std::wstring ext = L"\\\\?\\";
    if (s.compare(0, ext.size(), ext) == 0) s.erase(0, ext.size());
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetFullPathNameW(s.c_str(), (DWORD)(sizeof(buf) / sizeof(buf[0])), buf, nullptr);
    std::wstring out = (n > 0 && n < sizeof(buf) / sizeof(buf[0])) ? std::wstring(buf, n) : s;
    for (auto & c : out) c = (wchar_t)towlower(c);
    return out;
}

bool paths_equal_full(const std::string & a_utf8, const std::string & b_utf8) {
    return normalize_path_w(utf8_to_wide(a_utf8)) == normalize_path_w(utf8_to_wide(b_utf8));
}

std::wstring drive_root_of(const std::wstring & path) {
    if (path.size() >= 2 && path[1] == L':') return path.substr(0, 2);
    return path;
}

bool validate_alignment_descriptor(DWORD bytes_returned, DWORD version, DWORD size,
                                    DWORD logical, DWORD physical, std::string & err) {
    const DWORD struct_size = sizeof(STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR);
    if (bytes_returned < struct_size) { err = "IOCTL bytes_returned too small"; return false; }
    if (version != struct_size)       { err = "descriptor Version != struct size"; return false; }
    if (size < struct_size)           { err = "descriptor Size < struct size"; return false; }
    if (logical == 0 || physical == 0) { err = "logical/physical 0"; return false; }
    if ((logical & (logical - 1)) != 0 || (physical & (physical - 1)) != 0) {
        err = "logical/physical is not a power of two"; return false;
    }
    if (physical % logical != 0) { err = "physical is not a multiple of logical"; return false; }
    return true;
}

bool query_sector_via_ioctl(const std::wstring & drive_root, DWORD & logical, DWORD & physical, std::string & err) {
    std::wstring volume_path = L"\\\\.\\" + drive_root;
    HANDLE h = CreateFileW(volume_path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = "CreateFileW(volume) failed err=" + std::to_string(GetLastError());
        return false;
    }
    STORAGE_PROPERTY_QUERY query{};
    query.PropertyId = StorageAccessAlignmentProperty;
    query.QueryType = PropertyStandardQuery;
    STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR desc{};
    DWORD bytes_returned = 0;
    BOOL ok = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
                               &desc, sizeof(desc), &bytes_returned, nullptr);
    DWORD last_err = ok ? 0 : GetLastError();
    CloseHandle(h);
    if (!ok) {
        err = "DeviceIoControl(IOCTL_STORAGE_QUERY_PROPERTY) failed err=" + std::to_string(last_err);
        return false;
    }
    if (!validate_alignment_descriptor(bytes_returned, desc.Version, desc.Size,
                                        desc.BytesPerLogicalSector, desc.BytesPerPhysicalSector, err)) {
        return false;
    }
    logical = desc.BytesPerLogicalSector;
    physical = desc.BytesPerPhysicalSector;
    return true;
}

bool query_sector_via_getdiskfreespace(const std::wstring & drive_root, DWORD & logical, std::string & err) {
    std::wstring root = drive_root + L"\\";
    DWORD spc, bps, nfc, tnc;
    if (!GetDiskFreeSpaceW(root.c_str(), &spc, &bps, &nfc, &tnc)) {
        err = "GetDiskFreeSpaceW failed err=" + std::to_string(GetLastError());
        return false;
    }
    logical = bps;
    return true;
}

bool query_a_runtime(const std::wstring & file_path, bool allow_fallback, uint64_t & a_runtime, std::string & err) {
    std::wstring root = drive_root_of(normalize_path_w(file_path));
    DWORD logical = 0, physical = 0;
    std::string ioctl_err;
    if (query_sector_via_ioctl(root, logical, physical, ioctl_err)) {
        a_runtime = std::max<uint64_t>(4096ULL, std::max<uint64_t>(logical, physical));
        return true;
    }
    if (!allow_fallback) {
        err = "IOCTL sector query failed (fallback not allowed): " + ioctl_err;
        return false;
    }
    DWORD fb_logical = 0;
    std::string fb_err;
    if (query_sector_via_getdiskfreespace(root, fb_logical, fb_err)) {
        a_runtime = std::max<uint64_t>(4096ULL, std::max<uint64_t>(fb_logical, 4096ULL));
        return true;
    }
    err = "IOCTL failed(" + ioctl_err + ") and GetDiskFreeSpaceW fallback also failed(" + fb_err + ")";
    return false;
}

bool per_expert_slice_bytes(const std::string & type, const std::vector<int64_t> & dims,
                            uint64_t & out, std::string & err) {
    uint32_t bv, bb;
    if (!lookup_quant_trait(type, bv, bb)) { err = "type not in the type-trait table: " + type; return false; }
    if (dims.size() < 2) { err = "dims has too few axes (expert axis required): " + type; return false; }
    int64_t ne0 = dims[0];
    if (ne0 <= 0 || (ne0 % (int64_t)bv) != 0) { err = "ne0 % block_values != 0"; return false; }
    uint64_t middle = 1;
    for (size_t i = 1; i + 1 < dims.size(); i++) {
        if (dims[i] <= 0) { err = "dims has a negative or zero element"; return false; }
        middle *= (uint64_t)dims[i];
    }
    out = (uint64_t)(ne0 / (int64_t)bv) * (uint64_t)bb * middle;
    return true;
}

bool json_is_uint(const json & j) { return j.is_number_unsigned(); }

bool json_is_int(const json & j)  { return j.is_number_integer(); }

bool parse_dims(const json & j, std::vector<int64_t> & out, std::string & err) {
    if (!j.is_array() || j.empty()) { err = "dims is not an array or is empty"; return false; }
    out.clear();
    for (const auto & e : j) {
        if (!json_is_int(e)) { err = "dims element is not an integer"; return false; }
        out.push_back(e.get<int64_t>());
    }
    return true;
}

bool validate_model_block(const json & j,
                          const ggml_moe_direct_model & model,
                          ParsedManifest & out, std::string & err) {
    auto fail = [&](const std::string & what) -> bool { err = what; return false; };
    if (!j.contains("model") || !j["model"].is_object()) return fail("model missing");
    const json & jm = j["model"];
    if (!jm.contains("arch") || !jm["arch"].is_string()) return fail("model.arch missing or wrong type");
    out.arch = jm["arch"].get<std::string>();
    for (const char * k : {"n_layer", "n_expert", "n_expert_used"}) {
        if (!jm.contains(k) || !json_is_int(jm[k])) return fail(std::string("model.") + k + " missing or wrong type");
    }
    out.n_layer = jm["n_layer"].get<int32_t>();
    out.n_expert = jm["n_expert"].get<int32_t>();
    out.n_expert_used = jm["n_expert_used"].get<int32_t>();
    if (out.n_expert <= 0) return fail("n_expert <= 0");
    // ★uint16 상한(설계 §4-4): n_expert < 0xFFFF (NA sentinel 0xFFFF 여유).
    if (out.n_expert >= 0xFFFF) return fail("n_expert >= 0xFFFF (violates the uint16 NA headroom)");
    if (!jm.contains("moe_layers") || !jm["moe_layers"].is_array()) return fail("model.moe_layers missing or wrong type");
    for (const auto & e : jm["moe_layers"]) {
        if (!json_is_int(e)) return fail("moe_layers element is not an integer");
        out.moe_layers.push_back(e.get<int32_t>());
    }
    if (out.moe_layers.empty()) return fail("moe_layers is empty");
    // 오름차순·유일
    for (size_t i = 1; i < out.moe_layers.size(); i++) {
        if (out.moe_layers[i] <= out.moe_layers[i - 1]) return fail("moe_layers is not strictly ascending");
    }
    if (out.moe_layers.back() >= out.n_layer) return fail("max(moe_layers) >= n_layer");

    // ---- ★부록A 검증순서①: routed_scope 파싱(선택 필드·부재=all·"all"|"execution"만·그 외 fail-closed) ----
    {
        std::string scope = "all";
        if (jm.contains("routed_scope")) {
            if (!jm["routed_scope"].is_string()) return fail("model.routed_scope is not a string");
            scope = jm["routed_scope"].get<std::string>();
            if (scope != "all" && scope != "execution") return fail("model.routed_scope unknown value (all|execution only): " + scope);
        }
        out.routed_scope = scope;
    }

    // ---- ★부록A 검증순서②: live n_layer_nextn 범위(0 <= nextn < n_layer) + execution인데 nextn==0=실패.
    //      (moe_layers 대조보다 먼저 — 2차 조건②. n_layer 는 raw=block_count 기준.) ----
    if (model.n_layer_nextn < 0 || model.n_layer_nextn >= model.n_layer)
        return fail("n_layer_nextn out of range (0 <= n_layer_nextn < n_layer): " + std::to_string(model.n_layer_nextn));
    if (out.routed_scope == "execution" && model.n_layer_nextn == 0)
        return fail("execution scope but n_layer_nextn==0 (nothing to exclude)");

    // ---- live model 전항 대조(§4-2④) — 검증순서③: arch/n_layer/n_expert/n_expert_used = raw 그대로 ----
    if (out.arch != std::string(model.arch ? model.arch : "")) return fail("model.arch != live");
    if (out.n_layer != model.n_layer) return fail("model.n_layer != live");
    if (out.n_expert != model.n_expert) return fail("model.n_expert != live");
    if (out.n_expert_used != model.n_expert_used) return fail("model.n_expert_used != live");

    // ---- 검증순서④: moe_layers 대조 — scope-normalized live 집합으로.
    //      all       → live moe_layers 전체
    //      execution → live에서 layer < (n_layer - n_layer_nextn) 인 층만 골라 수·값 대조. ----
    {
        std::vector<int32_t> live_norm;
        const int32_t exec_threshold = model.n_layer - model.n_layer_nextn;
        for (int32_t i = 0; i < model.n_moe_layers; i++) {
            const int32_t lyr = model.moe_layers[i];
            if (out.routed_scope == "execution" && lyr >= exec_threshold) continue;
            live_norm.push_back(lyr);
        }
        if ((int32_t)out.moe_layers.size() != (int32_t)live_norm.size())
            return fail("moe_layers count != scope-normalized live count(" + std::to_string(out.moe_layers.size()) +
                        " vs " + std::to_string(live_norm.size()) + ")");
        for (size_t i = 0; i < out.moe_layers.size(); i++) {
            if (out.moe_layers[i] != live_norm[i]) return fail("moe_layers != scope-normalized live");
        }
    }
    return true;
}

bool compare_source_identity(const SourceInfo & si, int32_t i,
                             const ggml_moe_direct_source & live, std::string & err) {
    if (si.index != i) { err = "sources[i].index != i"; return false; }
    // ★F2: 전체 경로 비교(절대경로 정규화·Windows 대소문자 비민감). 상대경로=중단.
    if (!path_is_absolute(si.path)) { err = "sources[i].path is relative (absolute path required)"; return false; }
    if (!paths_equal_full(si.path, live.path ? live.path : "")) { err = "sources[i].path (full path) != live"; return false; }
    if (si.bytes != live.bytes) { err = "sources[i].bytes != live"; return false; }
    if (si.mtime_epoch != live.mtime_epoch) { err = "sources[i].mtime (seconds) != live"; return false; }
    if (si.data_start != live.data_start) { err = "sources[i].data_start != live"; return false; }
    if (si.gguf_version != live.gguf_version) { err = "sources[i].gguf_version != live"; return false; }
    if (si.alignment != live.alignment) { err = "sources[i].alignment != live"; return false; }
    return true;
}

bool validate_quant_traits(const json & j, std::string & err) {
    if (!j.contains("quant_traits") || !j["quant_traits"].is_object()) { err = "quant_traits missing"; return false; }
    for (auto it = j["quant_traits"].begin(); it != j["quant_traits"].end(); ++it) {
        uint32_t bv, bb;
        if (!lookup_quant_trait(it.key(), bv, bb)) { err = "invariant 7: quant_traits type not in the table " + it.key(); return false; }
        const json & tr = it.value();
        if (!tr.is_object() || !tr.contains("block_values") || !tr.contains("block_bytes") ||
            !json_is_int(tr["block_values"]) || !json_is_int(tr["block_bytes"]) ||
            tr["block_values"].get<uint32_t>() != bv || tr["block_bytes"].get<uint32_t>() != bb) {
            err = "invariant 7: quant_traits[" + it.key() + "] != frozen table"; return false;
        }
    }
    return true;
}

bool validate_reference_lock_fields(const json & j, ParsedManifest & out, std::string & err) {
    if (!j.contains("reference_lock") || !j["reference_lock"].is_object()) { err = "reference_lock missing"; return false; }
    const json & rl = j["reference_lock"];
    if (!rl.contains("profile_id") || !rl["profile_id"].is_string()) { err = "reference_lock.profile_id missing or wrong type"; return false; }
    if (!rl.contains("expect_sha256") || !rl["expect_sha256"].is_string()) { err = "reference_lock.expect_sha256 missing or wrong type"; return false; }
    out.ref_profile_id = rl["profile_id"].get<std::string>();
    out.ref_expect_sha256 = rl["expect_sha256"].get<std::string>();
    if (out.ref_expect_sha256.size() != 64) { err = "reference_lock.expect_sha256 length != 64"; return false; }
    for (char c : out.ref_expect_sha256) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            err = "reference_lock.expect_sha256 is not hexadecimal"; return false;
        }
    }
    return true;
}

bool validate_manifest_v2(const json & j,
                          const ggml_moe_direct_model & model,
                          const ggml_moe_direct_source * srcs, int32_t n_srcs,
                          ParsedManifest & out, std::string & err) {
    auto fail = [&](const std::string & what) -> bool { err = "manifest v2 validation failed: " + what; return false; };

    // ---- schema_version ----
    if (!j.contains("schema_version") || !j["schema_version"].is_string() ||
        j["schema_version"].get<std::string>() != "2.0") return fail("schema_version != \"2.0\"");

    // ---- model / routed_scope / live 전항 대조 / moe_layers(§2-3 승계 — v3 와 같은 구현) ----
    if (!validate_model_block(j, model, out, err)) return fail(err);

    // ---- sources[] (전 shard 실물 대조) ----
    if (!j.contains("sources") || !j["sources"].is_array()) return fail("sources missing");
    const json & jsrc = j["sources"];
    if ((int32_t)jsrc.size() != n_srcs) return fail("sources count != live shard count");
    for (int32_t i = 0; i < n_srcs; i++) {
        const json & s = jsrc[(size_t)i];
        if (!s.is_object()) return fail("sources[i] is not an object");
        for (const char * k : {"index", "gguf_version", "alignment"}) {
            if (!s.contains(k) || !json_is_int(s[k])) return fail(std::string("sources[i].") + k + " missing or wrong type");
        }
        if (!s.contains("path") || !s["path"].is_string()) return fail("sources[i].path missing or wrong type");
        if (!s.contains("bytes") || !json_is_uint(s["bytes"])) return fail("sources[i].bytes missing or wrong type");
        if (!s.contains("data_start") || !json_is_uint(s["data_start"])) return fail("sources[i].data_start missing or wrong type");
        if (!s.contains("mtime") || !s["mtime"].is_number()) return fail("sources[i].mtime missing or wrong type");
        SourceInfo si;
        si.index = s["index"].get<int32_t>();
        si.path = s["path"].get<std::string>();
        si.bytes = s["bytes"].get<uint64_t>();
        si.gguf_version = s["gguf_version"].get<uint32_t>();
        si.alignment = s["alignment"].get<uint32_t>();
        si.data_start = s["data_start"].get<uint64_t>();
        si.mtime_epoch = (int64_t)std::floor(s["mtime"].get<double>());
        // live shard 대조 — index 순서 일치 전제.
        if (!compare_source_identity(si, i, srcs[i], err)) return fail(err);
        out.sources.push_back(si);
    }

    // ---- layout ----
    if (!j.contains("layout") || !j["layout"].is_object()) return fail("layout missing");
    const json & jl = j["layout"];
    if (!jl.contains("align_bytes") || !json_is_uint(jl["align_bytes"])) return fail("align_bytes missing or wrong type");
    out.align_bytes = jl["align_bytes"].get<uint64_t>();
    if (out.align_bytes == 0 || (out.align_bytes & (out.align_bytes - 1)) != 0) return fail("align_bytes is zero or not a power of two");
    if (!jl.contains("align_query") || !jl["align_query"].is_object()) return fail("align_query missing");
    if (!jl.contains("slot_stride_max") || !json_is_uint(jl["slot_stride_max"])) return fail("slot_stride_max missing or wrong type");
    out.slot_stride_max = jl["slot_stride_max"].get<uint64_t>();
    if (!jl.contains("layers") || !jl["layers"].is_array()) return fail("layout.layers missing");
    const json & jlayers = jl["layers"];
    if ((int32_t)jlayers.size() != (int32_t)out.moe_layers.size()) return fail("layers count != moe_layers count");

    // ---- layers[] 파싱 + 불변식 2·3·4 + 파트별 type/dims/expert_axis/slice ----
    uint64_t base = 0;
    uint64_t max_stride = 0;
    for (size_t li = 0; li < jlayers.size(); li++) {
        const json & L = jlayers[li];
        if (!L.is_object()) return fail("layers[i] is not an object");
        for (const char * k : {"payload_bytes", "stride_bytes", "record_base"}) {
            if (!L.contains(k) || !json_is_uint(L[k])) return fail(std::string("layers[i].") + k + " missing or wrong type");
        }
        if (!L.contains("layer") || !json_is_int(L["layer"])) return fail("layers[i].layer missing or wrong type");
        LayerInfo li_info;
        li_info.layer = L["layer"].get<int32_t>();
        li_info.payload_bytes = L["payload_bytes"].get<uint64_t>();
        li_info.stride_bytes = L["stride_bytes"].get<uint64_t>();
        li_info.record_base = L["record_base"].get<uint64_t>();
        // 불변식 2: layers[].layer == moe_layers(순서·값)
        if (li_info.layer != out.moe_layers[li]) return fail("invariant 2: layers[].layer != moe_layers");
        // 불변식 4: record_base 연쇄·stride%A==0·stride>=payload
        if (li_info.record_base != base) return fail("invariant 4: record_base chain violated");
        if (li_info.stride_bytes % out.align_bytes != 0) return fail("invariant 4: stride % align != 0");
        if (li_info.stride_bytes < li_info.payload_bytes) return fail("invariant 4: stride < payload");
        // stride == ceil(payload/A)*A (레이아웃 재계산 일치)
        if (li_info.stride_bytes != align_up(li_info.payload_bytes, out.align_bytes))
            return fail("invariant 4: stride != ceil(payload/A)*A");
        base += (uint64_t)out.n_expert * li_info.stride_bytes;
        if (li_info.stride_bytes > max_stride) max_stride = li_info.stride_bytes;

        // parts
        if (!L.contains("parts") || !L["parts"].is_array() || L["parts"].empty()) return fail("layers[i].parts missing or empty");
        const json & jparts = L["parts"];
        uint64_t part_off = 0;
        std::vector<std::string> seen_names, seen_stensors;
        for (const auto & P : jparts) {
            if (!P.is_object()) return fail("part is not an object");
            for (const char * k : {"name", "source_tensor", "type"}) {
                if (!P.contains(k) || !P[k].is_string()) return fail(std::string("part.") + k + " missing or wrong type");
            }
            for (const char * k : {"source_index", "expert_axis"}) {
                if (!P.contains(k) || !json_is_int(P[k])) return fail(std::string("part.") + k + " missing or wrong type");
            }
            for (const char * k : {"part_offset", "part_bytes"}) {
                if (!P.contains(k) || !json_is_uint(P[k])) return fail(std::string("part.") + k + " missing or wrong type");
            }
            if (!P.contains("dims")) return fail("part.dims missing");
            PartInfo pi;
            pi.name = P["name"].get<std::string>();
            pi.source_tensor = P["source_tensor"].get<std::string>();
            pi.source_index = P["source_index"].get<int32_t>();
            pi.type = P["type"].get<std::string>();
            pi.expert_axis = P["expert_axis"].get<int32_t>();
            pi.part_offset = P["part_offset"].get<uint64_t>();
            pi.part_bytes = P["part_bytes"].get<uint64_t>();
            if (!parse_dims(P["dims"], pi.dims, err)) return fail("part.dims: " + err);
            // 불변식 3: 유일 name/source_tensor
            for (const auto & n : seen_names) if (n == pi.name) return fail("invariant 3: duplicate part name");
            for (const auto & n : seen_stensors) if (n == pi.source_tensor) return fail("invariant 3: duplicate source_tensor");
            seen_names.push_back(pi.name);
            seen_stensors.push_back(pi.source_tensor);
            // 불변식 3: part_offset 연쇄
            if (pi.part_offset != part_off) return fail("invariant 3: part_offset chain violated");
            // source_index 유효
            if (pi.source_index < 0 || pi.source_index >= n_srcs) return fail("part.source_index out of range");
            // expert_axis == 마지막 축 && ne[last]==n_expert
            if (pi.expert_axis != (int32_t)pi.dims.size() - 1) return fail("part.expert_axis != last axis");
            if (pi.dims.back() != out.n_expert) return fail("part.dims[last] != n_expert");
            // per-expert slice 재계산 == part_bytes (표에서 재산출 — 산술 권위)
            uint64_t slice = 0;
            if (!per_expert_slice_bytes(pi.type, pi.dims, slice, err)) return fail("part slice: " + err);
            if (slice != pi.part_bytes) return fail("invariant: recomputed per-expert slice != part_bytes");
            part_off += pi.part_bytes;
            li_info.parts.push_back(std::move(pi));
        }
        // 불변식 3: parts 합 == payload_bytes
        if (part_off != li_info.payload_bytes) return fail("invariant 3: sum(parts) != payload_bytes");
        out.layers.push_back(std::move(li_info));
    }
    // 불변식 4: slot_stride_max == max(stride)
    if (out.slot_stride_max != max_stride) return fail("invariant 4: slot_stride_max != max(stride)");

    // ---- totals ----
    if (!j.contains("totals") || !j["totals"].is_object()) return fail("totals missing");
    const json & jt = j["totals"];
    if (!jt.contains("bin_file_bytes") || !json_is_uint(jt["bin_file_bytes"])) return fail("totals.bin_file_bytes missing or wrong type");
    out.bin_file_bytes = jt["bin_file_bytes"].get<uint64_t>();
    if (out.bin_file_bytes != base) return fail("totals.bin_file_bytes != final base");
    if (jt.contains("n_records") && json_is_int(jt["n_records"])) {
        int64_t nrec = jt["n_records"].get<int64_t>();
        if (nrec != (int64_t)out.moe_layers.size() * out.n_expert) return fail("totals.n_records mismatch");
    }
    if (jt.contains("expert_payload_total") && json_is_uint(jt["expert_payload_total"])) {
        uint64_t ept = 0;
        for (const auto & L : out.layers) ept += (uint64_t)out.n_expert * L.payload_bytes;
        if (jt["expert_payload_total"].get<uint64_t>() != ept) return fail("totals.expert_payload_total mismatch");
    }

    // ---- records[] = 비권위 witness(불변식 5) ----
    if (!j.contains("records") || !j["records"].is_array()) return fail("records missing");
    const json & jrec = j["records"];
    int64_t expect_n = (int64_t)out.moe_layers.size() * out.n_expert;
    if ((int64_t)jrec.size() != expect_n) return fail("records length mismatch");
    int64_t idx = 0;
    for (size_t mlpos = 0; mlpos < out.layers.size(); mlpos++) {
        const LayerInfo & L = out.layers[mlpos];
        for (int32_t e = 0; e < out.n_expert; e++) {
            const json & r = jrec[(size_t)idx];
            if (!r.is_object() || !r.contains("layer") || !r.contains("expert") ||
                !r.contains("offset") || !r.contains("payload_bytes")) return fail("records[i] field missing");
            if (!json_is_int(r["layer"]) || r["layer"].get<int32_t>() != L.layer) return fail("records[i].layer");
            if (!json_is_int(r["expert"]) || r["expert"].get<int32_t>() != e) return fail("records[i].expert");
            uint64_t want_off = L.record_base + (uint64_t)e * L.stride_bytes;
            if (!json_is_uint(r["offset"]) || r["offset"].get<uint64_t>() != want_off) return fail("records[i].offset (non-authoritative witness)");
            if (!json_is_uint(r["payload_bytes"]) || r["payload_bytes"].get<uint64_t>() != L.payload_bytes) return fail("records[i].payload_bytes");
            idx++;
        }
    }

    // ---- source_tensors[] <-> flattened parts 1:1 (불변식 6) ----
    if (!j.contains("source_tensors") || !j["source_tensors"].is_array()) return fail("source_tensors missing");
    const json & jst = j["source_tensors"];
    std::vector<const PartInfo *> flat;
    for (const auto & L : out.layers) for (const auto & p : L.parts) flat.push_back(&p);
    if ((int64_t)jst.size() != (int64_t)flat.size()) return fail("invariant 6: source_tensors count != flattened parts");
    for (size_t i = 0; i < flat.size(); i++) {
        const json & st = jst[i];
        const PartInfo & p = *flat[i];
        if (!st.is_object()) return fail("invariant 6: source_tensors[i] is not an object");
        if (!st.contains("name") || !st["name"].is_string() || st["name"].get<std::string>() != p.source_tensor)
            return fail("invariant 6: source_tensors[i].name != part.source_tensor");
        if (!st.contains("source_index") || !json_is_int(st["source_index"]) || st["source_index"].get<int32_t>() != p.source_index)
            return fail("invariant 6: source_tensors[i].source_index");
        if (!st.contains("type") || !st["type"].is_string() || st["type"].get<std::string>() != p.type)
            return fail("invariant 6: source_tensors[i].type");
        std::vector<int64_t> sdims;
        if (!st.contains("dims") || !parse_dims(st["dims"], sdims, err) || sdims != p.dims)
            return fail("invariant 6: source_tensors[i].dims");
        if (!st.contains("bytes") || !json_is_uint(st["bytes"]) ||
            st["bytes"].get<uint64_t>() != p.part_bytes * (uint64_t)out.n_expert)
            return fail("invariant 6: source_tensors[i].bytes != part_bytes*n_expert");
    }

    // ---- quant_traits = 감사용 복제(불변식 7) — 동결 표와 정확 일치 ----
    if (!validate_quant_traits(j, err)) return fail(err);

    // ---- reference_lock 필드 무결(불변식 8 — 소비자는 필드 무결만, 카탈로그 대조는 repacker 몫) ----
    if (!validate_reference_lock_fields(j, out, err)) return fail(err);

    return true;
}

bool load_manifest_json(const std::wstring & manifest_path, json & out, std::string & err) {
    std::ifstream ifs(manifest_path, std::ios::binary);
    if (!ifs.is_open()) { err = "cannot open manifest.json"; return false; }
    try {
        ifs >> out;
    } catch (const std::exception & e) {
        err = std::string("manifest.json JSON parse failed: ") + e.what();
        return false;
    }
    return true;
}

inline bool ck_align_up(uint64_t v, uint64_t a, uint64_t & out) {
    if (a == 0) return false;
    uint64_t t = 0;
    if (!ck_add(v, a - 1, t)) return false;
    out = t / a * a; return true;
}

bool is_hex64(const std::string & s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    }
    return true;
}

bool detect_manifest_mode(const json & j, ManifestMode & out, std::string & err) {
    if (!j.is_object()) { err = "the top level of manifest.json is not a JSON object"; return false; }
    if (!j.contains("schema_version") || !j["schema_version"].is_string()) {
        err = "manifest.schema_version is missing or not a string"; return false;
    }
    const std::string sv = j["schema_version"].get<std::string>();
    const bool has_mode = j.contains("mode");
    if (sv == "2.0") {
        if (has_mode) {
            err = "mode/schema mismatch: schema_version \"2.0\" carries a mode field, but bin manifests are "
                  "produced with no mode field (fail-closed)";
            return false;
        }
        out = ManifestMode::BIN;   // implicit bin
        return true;
    }
    if (sv == "3.0") {
        if (!has_mode) { err = "schema_version \"3.0\" is missing the mandatory mode field"; return false; }
        if (!j["mode"].is_string()) { err = "manifest.mode is not a string"; return false; }
        const std::string mode = j["mode"].get<std::string>();
        if (mode == "virtual") { out = ManifestMode::VIRTUAL; return true; }
        if (mode == "bin") {
            err = "forbidden combination: schema_version \"3.0\" with mode \"bin\" - bin is always \"2.0\" and "
                  "carries no mode field (this artifact is not producible; fail-closed)";
            return false;
        }
        err = "unknown manifest mode value at schema_version \"3.0\" (\"virtual\" only): " + mode;
        return false;
    }
    err = "unsupported manifest schema_version (\"2.0\"=bin | \"3.0\"=virtual): " + sv;
    return false;
}

bool virtual_init_reject_check(std::string & err) {
    const char * reason = nullptr;
    if (ggml_moe_phasea_mode() != GGML_MOE_PHASEA_OFF) {
        reason = "MOE_DIRECT_PHASEA != off (Phase A records experts.bin file offsets, which do not exist in "
                 "mode=virtual)";
    } else if (ggml_moe_phaseb_mode() != GGML_MOE_PHASEB_OFF) {
        reason = "MOE_DIRECT_PHASEB != off (pb_io assumes one physical op per record with a bin stride)";
    } else if (getenv_str("MOE_DIRECT_IO_PATH") == "legacy") {
        reason = "MOE_DIRECT_IO_PATH=legacy (the legacy pump is a qualification-only control arm)";
    } else if (ggml_moe_trace_enabled()) {
        reason = "MOE_P1_TRACE=1 (O1/O2 assume one physical op per layer/expert/slot and the readers hard-check "
                 "sum(O2 bytes) == phase read_bytes)";
    }
    if (!reason) return true;
    err = std::string("mode=virtual init combination is rejected: ") + reason +
          " - see SPEC_IO_METRICS_V3 section 3 (status: temporary; the lift path is the section 3-2 promotion review)";
    log_err("[moe-direct] %s", err.c_str());
    return false;
}

bool validate_manifest_v3(const json & j,
                          const ggml_moe_direct_model & model,
                          const ggml_moe_direct_source * srcs, int32_t n_srcs,
                          ParsedManifest & out, std::string & err) {
    auto fail = [&](const std::string & what) -> bool { err = "manifest v3 validation failed: " + what; return false; };
    out.virtual_mode = true;

    // ---- §2-3 승계: model 블록(v2 와 같은 계약·같은 구현) ----
    if (!validate_model_block(j, model, out, err)) return fail(err);

    // ---- sources[] 실물 대조(§2-5 ①) + DF-1 digest 필드 무결 ----
    if (!j.contains("sources") || !j["sources"].is_array()) return fail("sources missing");
    const json & jsrc = j["sources"];
    if ((int32_t)jsrc.size() != n_srcs) return fail("sources count != live shard count");
    for (int32_t i = 0; i < n_srcs; i++) {
        const json & s = jsrc[(size_t)i];
        if (!s.is_object()) return fail("sources[i] is not an object");
        for (const char * k : { "index", "gguf_version", "alignment" }) {
            if (!s.contains(k) || !json_is_int(s[k])) return fail(std::string("sources[i].") + k + " missing or wrong type");
        }
        if (!s.contains("path") || !s["path"].is_string()) return fail("sources[i].path missing or wrong type");
        if (!s.contains("bytes") || !json_is_uint(s["bytes"])) return fail("sources[i].bytes missing or wrong type");
        if (!s.contains("data_start") || !json_is_uint(s["data_start"])) return fail("sources[i].data_start missing or wrong type");
        if (!s.contains("mtime") || !s["mtime"].is_number()) return fail("sources[i].mtime missing or wrong type");
        SourceInfo si;
        si.index        = s["index"].get<int32_t>();
        si.path         = s["path"].get<std::string>();
        si.bytes        = s["bytes"].get<uint64_t>();
        si.gguf_version = s["gguf_version"].get<uint32_t>();
        si.alignment    = s["alignment"].get<uint32_t>();
        si.data_start   = s["data_start"].get<uint64_t>();
        si.mtime_epoch  = (int64_t)std::floor(s["mtime"].get<double>());
        if (!compare_source_identity(si, i, srcs[i], err)) return fail(err);
        // ★DF-1 — 주소 권위 영역 [0, data_start) digest. 값 재해시는 부팅 프로브(§2-5)가 한다.
        if (!s.contains("digest") || !s["digest"].is_object())
            return fail("sources[i].digest missing (DF-1: mode=virtual promotes the source header digest to a "
                        "mandatory field because the original GGUF is the runtime data source)");
        const json & dg = s["digest"];
        if (!dg.contains("algo") || !dg["algo"].is_string() || dg["algo"].get<std::string>() != "sha256")
            return fail("sources[i].digest.algo != \"sha256\"");
        if (!dg.contains("scope") || !dg["scope"].is_string() || dg["scope"].get<std::string>() != "header")
            return fail("sources[i].digest.scope != \"header\"");
        if (!dg.contains("header_bytes") || !json_is_uint(dg["header_bytes"]))
            return fail("sources[i].digest.header_bytes missing or wrong type");
        if (!dg.contains("sha256") || !dg["sha256"].is_string())
            return fail("sources[i].digest.sha256 missing or wrong type");
        si.digest_header_bytes = dg["header_bytes"].get<uint64_t>();
        si.digest_sha256       = dg["sha256"].get<std::string>();
        if (!is_hex64(si.digest_sha256)) return fail("sources[i].digest.sha256 is not 64 hexadecimal characters");
        if (si.digest_header_bytes != si.data_start)
            return fail("sources[i].digest.header_bytes != data_start (the address-authority region is exactly "
                        "[0, data_start))");
        if (si.data_start == 0) return fail("sources[i].data_start == 0 (a GGUF shard always has a header)");
        out.sources.push_back(si);
    }

    // ---- layout ----
    if (!j.contains("layout") || !j["layout"].is_object()) return fail("layout missing");
    const json & jl = j["layout"];
    if (!jl.contains("align_bytes") || !json_is_uint(jl["align_bytes"])) return fail("layout.align_bytes missing or wrong type");
    out.align_bytes = jl["align_bytes"].get<uint64_t>();
    const uint64_t A = out.align_bytes;
    if (A == 0 || (A & (A - 1)) != 0) return fail("layout.align_bytes is zero or not a power of two");
    if (A < 4096) return fail("layout.align_bytes < 4096 (section 2-4: A = max(4096, sector query))");

    // align_query[] — §2-4 provenance(소스 볼륨 질의 결과). ★**소비 기동 시점의 재질의 대조는
    // 아직 미시공**이다 [[C:repack.align-query]] — 사양이 명문으로 남긴 §Z 후속 계약이라 여기서
    // 임의로 열지 않고, 형상과 sources[] 결속만 확인한다(그 계약이 열릴 때 값 재질의가 붙는다).
    if (!jl.contains("align_query") || !jl["align_query"].is_array())
        return fail("layout.align_query missing or not an array (v3 records one entry per source volume)");
    const json & jaq = jl["align_query"];
    if ((int32_t)jaq.size() != n_srcs) return fail("layout.align_query count != sources count");
    for (int32_t i = 0; i < n_srcs; i++) {
        const json & q = jaq[(size_t)i];
        if (!q.is_object()) return fail("layout.align_query[i] is not an object");
        if (!q.contains("source_index") || !json_is_int(q["source_index"]) || q["source_index"].get<int32_t>() != i)
            return fail("layout.align_query[i].source_index != i (the provenance list must be bound to sources[])");
        if (!q.contains("method") || !q["method"].is_string()) return fail("layout.align_query[i].method missing or wrong type");
        for (const char * k : { "logical", "physical" }) {
            if (!q.contains(k) || !json_is_int(q[k])) return fail(std::string("layout.align_query[i].") + k + " missing or wrong type");
        }
    }

    // legacy alignment(§Z-③ [[C:repack.legacy-align]]) — D-A2 분자축. source 볼륨 A 와 **독립**
    // 이다(과거 v2 output 볼륨의 stride 와 같다는 보장이 없어 별도 필드로 분리됐다).
    if (!jl.contains("legacy_align_bytes") || !json_is_uint(jl["legacy_align_bytes"]))
        return fail("layout.legacy_align_bytes missing or wrong type");
    out.legacy_align_bytes = jl["legacy_align_bytes"].get<uint64_t>();
    if (out.legacy_align_bytes == 0 || (out.legacy_align_bytes & (out.legacy_align_bytes - 1)) != 0)
        return fail("layout.legacy_align_bytes is zero or not a power of two");
    if (!jl.contains("legacy_align_source") || !jl["legacy_align_source"].is_string())
        return fail("layout.legacy_align_source missing or wrong type");
    out.legacy_align_source = jl["legacy_align_source"].get<std::string>();
    if (out.legacy_align_source != "paired_v2" && out.legacy_align_source != "canonical_4096")
        return fail("layout.legacy_align_source unknown value (paired_v2|canonical_4096): " + out.legacy_align_source);
    if (out.legacy_align_source == "canonical_4096" && out.legacy_align_bytes != 4096)
        return fail("legacy_align_source=canonical_4096 but legacy_align_bytes != 4096");

    if (!jl.contains("slot_stride_max") || !json_is_uint(jl["slot_stride_max"])) return fail("layout.slot_stride_max missing or wrong type");
    out.slot_stride_max = jl["slot_stride_max"].get<uint64_t>();

    // ---- §2-2 bin 전용 좌표 필드 금지(존재 자체가 기동 중단) ----
    if (j.contains("totals") && j["totals"].is_object() && j["totals"].contains("bin_file_bytes"))
        return fail("bin-only field totals.bin_file_bytes is present in a mode=virtual manifest");

    // ---- layers[] + parts[] : §2-4 슬롯 산술 자기일관 재계산 ----
    if (!jl.contains("layers") || !jl["layers"].is_array()) return fail("layout.layers missing");
    const json & jlayers = jl["layers"];
    if ((int32_t)jlayers.size() != (int32_t)out.moe_layers.size()) return fail("layers count != moe_layers count");

    uint64_t max_lsb = 0, payload_total = 0;
    int64_t  rec_total = 0;
    for (size_t li = 0; li < jlayers.size(); li++) {
        const json & L = jlayers[li];
        if (!L.is_object()) return fail("layers[i] is not an object");
        if (L.contains("record_base"))  return fail("bin-only field layers[].record_base is present in a mode=virtual manifest");
        if (L.contains("stride_bytes")) return fail("bin-only field layers[].stride_bytes is present in a mode=virtual manifest");
        if (!L.contains("layer") || !json_is_int(L["layer"])) return fail("layers[i].layer missing or wrong type");
        if (!L.contains("layer_slot_bytes") || !json_is_uint(L["layer_slot_bytes"]))
            return fail("layers[i].layer_slot_bytes missing or wrong type");
        LayerInfo li_info;
        li_info.layer            = L["layer"].get<int32_t>();
        li_info.layer_slot_bytes = L["layer_slot_bytes"].get<uint64_t>();
        if (li_info.layer != out.moe_layers[li]) return fail("invariant v3-1: layers[].layer != moe_layers");

        if (!L.contains("parts") || !L["parts"].is_array() || L["parts"].empty()) return fail("layers[i].parts missing or empty");
        uint64_t slot_off = 0;
        std::vector<std::string> seen_names, seen_stensors;
        for (const auto & P : L["parts"]) {
            if (!P.is_object()) return fail("part is not an object");
            if (P.contains("part_offset")) return fail("bin-only field parts[].part_offset is present in a mode=virtual manifest");
            for (const char * k : { "name", "source_tensor", "type" }) {
                if (!P.contains(k) || !P[k].is_string()) return fail(std::string("part.") + k + " missing or wrong type");
            }
            for (const char * k : { "source_index", "expert_axis" }) {
                if (!P.contains(k) || !json_is_int(P[k])) return fail(std::string("part.") + k + " missing or wrong type");
            }
            for (const char * k : { "abs_offset", "slice_bytes", "slot_offset", "data_offset" }) {
                if (!P.contains(k) || !json_is_uint(P[k])) return fail(std::string("part.") + k + " missing or wrong type");
            }
            if (!P.contains("aligned") || !P["aligned"].is_boolean()) return fail("part.aligned missing or not a boolean");
            if (!P.contains("dims")) return fail("part.dims missing");
            PartInfo pi;
            pi.name          = P["name"].get<std::string>();
            pi.source_tensor = P["source_tensor"].get<std::string>();
            pi.source_index  = P["source_index"].get<int32_t>();
            pi.type          = P["type"].get<std::string>();
            pi.expert_axis   = P["expert_axis"].get<int32_t>();
            pi.abs_offset    = P["abs_offset"].get<uint64_t>();
            pi.slice_bytes   = P["slice_bytes"].get<uint64_t>();
            pi.slot_offset   = P["slot_offset"].get<uint64_t>();
            pi.data_offset   = P["data_offset"].get<uint64_t>();
            pi.aligned       = P["aligned"].get<bool>();
            if (!parse_dims(P["dims"], pi.dims, err)) return fail("part.dims: " + err);
            for (const auto & n : seen_names)    if (n == pi.name)          return fail("invariant v3-2: duplicate part name");
            for (const auto & n : seen_stensors) if (n == pi.source_tensor) return fail("invariant v3-2: duplicate source_tensor");
            seen_names.push_back(pi.name);
            seen_stensors.push_back(pi.source_tensor);
            if (pi.source_index < 0 || pi.source_index >= n_srcs) return fail("part.source_index out of range");
            if (pi.expert_axis != (int32_t)pi.dims.size() - 1) return fail("part.expert_axis != last axis");
            if (pi.dims.back() != out.n_expert) return fail("part.dims[last] != n_expert");
            // per-expert slice 재계산 == slice_bytes (동결 표에서 재산출 — 산술 권위)
            uint64_t slice = 0;
            if (!per_expert_slice_bytes(pi.type, pi.dims, slice, err)) return fail("part slice: " + err);
            if (slice != pi.slice_bytes) return fail("invariant v3-3: recomputed per-expert slice != slice_bytes");
            if (pi.slice_bytes == 0) return fail("part.slice_bytes == 0");

            // §2-4: aligned 판정은 산술이지 신고가 아니다.
            if (pi.aligned != (pi.slice_bytes % A == 0))
                return fail("invariant v3-4: part.aligned != (slice_bytes % A == 0)");
            if (pi.aligned) {
                if (!P.contains("bracket_head") || !json_is_uint(P["bracket_head"]))
                    return fail("aligned part is missing bracket_head");
                pi.bracket_head = P["bracket_head"].get<uint64_t>();
                if (pi.bracket_head != pi.abs_offset % A) return fail("invariant v3-4: bracket_head != abs_offset % A");
                uint64_t span = 0, want_data = 0;
                if (!ck_add(pi.bracket_head, pi.slice_bytes, span) || !ck_align_up(span, A, pi.region))
                    return fail("checked uint64 overflow while recomputing region");
                if (!ck_add(pi.slot_offset, pi.bracket_head, want_data))
                    return fail("checked uint64 overflow while recomputing data_offset");
                if (pi.data_offset != want_data) return fail("invariant v3-4: data_offset != slot_offset + bracket_head");
            } else {
                // 비4K: h·bracket_head 는 미정의라 **기재 금지**(§2-4). 실데이터는 복사가 정렬을
                // 흡수하므로 슬롯 시작에서 시작한다.
                if (P.contains("bracket_head"))
                    return fail("a non-4K (aligned=false) part must not carry bracket_head (undefined by section 2-4)");
                if (!ck_align_up(pi.slice_bytes, A, pi.region)) return fail("checked uint64 overflow while recomputing region");
                if (pi.data_offset != pi.slot_offset) return fail("invariant v3-4: non-4K part data_offset != slot_offset");
                if (!P.contains("staging_bytes") || !json_is_uint(P["staging_bytes"]))
                    return fail("a non-4K part is missing staging_bytes");
                pi.staging_bytes = P["staging_bytes"].get<uint64_t>();
                uint64_t want_staging = 0;
                if (!ck_add(pi.region, A, want_staging)) return fail("checked uint64 overflow while recomputing staging_bytes");
                if (pi.staging_bytes < want_staging)
                    return fail("part.staging_bytes < align_up(slice,A)+A (worst-case src_head would not fit the "
                                "dedicated staging buffer)");
            }
            // 비중첩 점화식 — slot_offset[p+1] = slot_offset[p] + region[p].
            if (pi.slot_offset != slot_off)
                return fail("invariant v3-5: slot_offset chain violated (overlapping, reordered or gapped regions)");
            if (!ck_add(slot_off, pi.region, slot_off)) return fail("checked uint64 overflow in the slot_offset chain");

            // ★경계 강제(§2-4 bracket EOF) — 생산 단계의 자동 bin 회귀 판정을 소비자가 재확인한다.
            uint64_t tensor_bytes = 0, payload_end = 0, bracket_end = 0;
            if (!ck_mul(pi.slice_bytes, (uint64_t)out.n_expert, tensor_bytes) ||
                !ck_add(pi.abs_offset, tensor_bytes, payload_end) ||
                !ck_align_up(payload_end, A, bracket_end))
                return fail("checked uint64 overflow while checking the bracket EOF boundary");
            const uint64_t src_bytes = out.sources[(size_t)pi.source_index].bytes;
            if (payload_end > src_bytes)
                return fail("routed tensor payload exceeds the source EOF: " + pi.source_tensor);
            if (bracket_end > src_bytes)
                return fail("bracket EOF violation (section 2-4): the last aligned bracket read would run past the "
                            "end of the shard for " + pi.source_tensor + " - this profile is not virtual-eligible");
            if (!ck_add(payload_total, tensor_bytes, payload_total))
                return fail("checked uint64 overflow while accumulating totals.virtual_payload_bytes");
            li_info.parts.push_back(std::move(pi));
        }
        if (li_info.layer_slot_bytes != slot_off) return fail("invariant v3-5: layer_slot_bytes != sum(region)");
        if (li_info.layer_slot_bytes % A != 0)    return fail("invariant v3-5: layer_slot_bytes % A != 0");
        if (li_info.layer_slot_bytes > max_lsb) max_lsb = li_info.layer_slot_bytes;
        rec_total += (int64_t)out.n_expert * (int64_t)li_info.parts.size();
        out.layers.push_back(std::move(li_info));
    }
    if (out.slot_stride_max != max_lsb) return fail("invariant v3-5: slot_stride_max != max(layer_slot_bytes)");
    if (out.slot_stride_max == 0)       return fail("slot_stride_max == 0");

    // ---- totals ----
    if (!j.contains("totals") || !j["totals"].is_object()) return fail("totals missing");
    const json & jt = j["totals"];
    if (!jt.contains("virtual_payload_bytes") || !json_is_uint(jt["virtual_payload_bytes"]))
        return fail("totals.virtual_payload_bytes missing or wrong type");
    out.virtual_payload_bytes = jt["virtual_payload_bytes"].get<uint64_t>();
    if (out.virtual_payload_bytes != payload_total)
        return fail("totals.virtual_payload_bytes != sum(slice_bytes * n_expert)");
    if (!jt.contains("n_records") || !json_is_int(jt["n_records"])) return fail("totals.n_records missing or wrong type");
    out.n_records = jt["n_records"].get<int64_t>();
    if (out.n_records != rec_total) return fail("totals.n_records != sum_layers(n_expert * n_parts)");

    // ---- records[] = 비권위 witness — 전항 재생성 대조(누락·중복·순서 변조까지 판정) ----
    if (!j.contains("records") || !j["records"].is_array()) return fail("records missing");
    const json & jrec = j["records"];
    if ((int64_t)jrec.size() != rec_total) return fail("records length mismatch");
    int64_t idx = 0;
    for (size_t mlpos = 0; mlpos < out.layers.size(); mlpos++) {
        const LayerInfo & L = out.layers[mlpos];
        for (int32_t e = 0; e < out.n_expert; e++) {
            for (size_t pidx = 0; pidx < L.parts.size(); pidx++) {
                const PartInfo & p = L.parts[pidx];
                const json & r = jrec[(size_t)idx];
                if (!r.is_object()) return fail("records[i] is not an object");
                for (const char * k : { "layer", "expert", "part", "source_index", "src_offset", "slice_bytes", "data_offset" }) {
                    if (!r.contains(k)) return fail(std::string("records[i].") + k + " missing");
                }
                if (!json_is_int(r["layer"]) || r["layer"].get<int32_t>() != L.layer) return fail("records[i].layer");
                if (!json_is_int(r["expert"]) || r["expert"].get<int32_t>() != e) return fail("records[i].expert");
                if (!r["part"].is_string() || r["part"].get<std::string>() != p.name) return fail("records[i].part");
                if (!json_is_int(r["source_index"]) || r["source_index"].get<int32_t>() != p.source_index)
                    return fail("records[i].source_index");
                uint64_t want_src = 0;
                if (!ck_mul((uint64_t)e, p.slice_bytes, want_src) || !ck_add(p.abs_offset, want_src, want_src))
                    return fail("checked uint64 overflow while regenerating records[i].src_offset");
                if (!json_is_uint(r["src_offset"]) || r["src_offset"].get<uint64_t>() != want_src)
                    return fail("records[i].src_offset (non-authoritative witness disagrees with the regenerated address)");
                if (!json_is_uint(r["slice_bytes"]) || r["slice_bytes"].get<uint64_t>() != p.slice_bytes)
                    return fail("records[i].slice_bytes");
                if (!json_is_uint(r["data_offset"]) || r["data_offset"].get<uint64_t>() != p.data_offset)
                    return fail("records[i].data_offset");
                idx++;
            }
        }
    }

    // ---- source_tensors[] <-> flattened parts 1:1(비권위 witness — abs_offset 까지 전항) ----
    if (!j.contains("source_tensors") || !j["source_tensors"].is_array()) return fail("source_tensors missing");
    const json & jst = j["source_tensors"];
    std::vector<const PartInfo *> flat;
    for (const auto & L : out.layers) for (const auto & p : L.parts) flat.push_back(&p);
    if ((int64_t)jst.size() != (int64_t)flat.size()) return fail("invariant v3-6: source_tensors count != flattened parts");
    for (size_t i = 0; i < flat.size(); i++) {
        const json & st = jst[i];
        const PartInfo & p = *flat[i];
        if (!st.is_object()) return fail("invariant v3-6: source_tensors[i] is not an object");
        if (!st.contains("name") || !st["name"].is_string() || st["name"].get<std::string>() != p.source_tensor)
            return fail("invariant v3-6: source_tensors[i].name != part.source_tensor");
        if (!st.contains("source_index") || !json_is_int(st["source_index"]) ||
            st["source_index"].get<int32_t>() != p.source_index)
            return fail("invariant v3-6: source_tensors[i].source_index");
        if (!st.contains("abs_offset") || !json_is_uint(st["abs_offset"]) ||
            st["abs_offset"].get<uint64_t>() != p.abs_offset)
            return fail("invariant v3-6: source_tensors[i].abs_offset");
        if (!st.contains("type") || !st["type"].is_string() || st["type"].get<std::string>() != p.type)
            return fail("invariant v3-6: source_tensors[i].type");
        std::vector<int64_t> sdims;
        if (!st.contains("dims") || !parse_dims(st["dims"], sdims, err) || sdims != p.dims)
            return fail("invariant v3-6: source_tensors[i].dims");
        uint64_t want_bytes = 0;
        if (!ck_mul(p.slice_bytes, (uint64_t)out.n_expert, want_bytes)) return fail("checked uint64 overflow (source_tensors bytes)");
        if (!st.contains("bytes") || !json_is_uint(st["bytes"]) || st["bytes"].get<uint64_t>() != want_bytes)
            return fail("invariant v3-6: source_tensors[i].bytes != slice_bytes * n_expert");
    }

    // ---- quant_traits / reference_lock(v2 와 같은 계약·같은 구현) ----
    if (!validate_quant_traits(j, err)) return fail(err);
    if (!validate_reference_lock_fields(j, out, err)) return fail(err);
    // §Z-③ paired 결속: identity 권위는 SHA **하나뿐**이고 경로는 비권위 locator다.
    if (out.legacy_align_source == "paired_v2") {
        const json & rl = j["reference_lock"];
        if (!rl.contains("legacy_v2_manifest_sha256") || !rl["legacy_v2_manifest_sha256"].is_string() ||
            !is_hex64(rl["legacy_v2_manifest_sha256"].get<std::string>()))
            return fail("legacy_align_source=paired_v2 requires reference_lock.legacy_v2_manifest_sha256 "
                        "(64 hexadecimal characters)");
    }
    return true;
}

bool read_plan_report_gate(const std::wstring & path, std::string & out_pid, std::string & out_sha,
                           std::string & out_manifest_sha, int64_t & out_records, std::string & err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) { err = "cannot open plan_report.json"; return false; }
    json j;
    try {
        ifs >> j;
    } catch (const std::exception & e) {
        err = std::string("plan_report.json JSON parse failed: ") + e.what(); return false;
    }
    if (!j.is_object()) { err = "the top level of plan_report.json is not a JSON object"; return false; }
    if (!j.contains("mode") || !j["mode"].is_string() || j["mode"].get<std::string>() != "virtual") {
        err = "plan_report.mode != \"virtual\" (this report did not verify a virtual artifact)"; return false;
    }
    if (!j.contains("pass") || !j["pass"].is_boolean()) { err = "plan_report.pass missing or not a boolean"; return false; }
    if (!j["pass"].get<bool>()) { err = "plan_report.pass=false - the manifest failed independent verification"; return false; }
    if (!j.contains("problems") || !j["problems"].is_array()) { err = "plan_report.problems missing or not an array"; return false; }
    if (!j["problems"].empty()) {
        err = "plan_report.problems is not empty(" + std::to_string(j["problems"].size()) +
              ") while pass=true - the report contradicts itself"; return false;
    }
    if (!j.contains("manifest_sha256") || !j["manifest_sha256"].is_string()) {
        err = "plan_report.manifest_sha256 missing or not a string - the report is not cache-key bound"; return false;
    }
    out_manifest_sha = j["manifest_sha256"].get<std::string>();
    if (!is_hex64(out_manifest_sha)) { err = "plan_report.manifest_sha256 is not 64 hexadecimal characters"; return false; }
    if (!j.contains("reference_lock") || !j["reference_lock"].is_object()) { err = "plan_report.reference_lock missing"; return false; }
    const json & rl = j["reference_lock"];
    if (!rl.contains("profile_id") || !rl["profile_id"].is_string() ||
        !rl.contains("expect_sha256") || !rl["expect_sha256"].is_string()) {
        err = "plan_report.reference_lock fields missing"; return false;
    }
    out_pid = rl["profile_id"].get<std::string>();
    out_sha = rl["expect_sha256"].get<std::string>();
    if (!j.contains("cardinality") || !j["cardinality"].is_object() ||
        !j["cardinality"].contains("records") || !json_is_int(j["cardinality"]["records"])) {
        err = "plan_report.cardinality.records missing or not an integer"; return false;
    }
    out_records = j["cardinality"]["records"].get<int64_t>();
    return true;
}

bool virtual_probe_sources(const ParsedManifest & pm,
                           std::unordered_map<std::string, int32_t> & out_live_type,
                           std::string & err) {
    size_t resolved = 0, total_parts = 0;
    for (const auto & L : pm.layers) total_parts += L.parts.size();
    for (size_t si_i = 0; si_i < pm.sources.size(); si_i++) {
        const SourceInfo & si = pm.sources[si_i];
        std::ifstream ifs(utf8_to_wide(si.path), std::ios::binary);
        if (!ifs.is_open()) { err = "cannot open the source shard for the boot re-parse: " + si.path; return false; }
        std::vector<uint8_t> head((size_t)si.data_start);
        ifs.read((char *)head.data(), (std::streamsize)head.size());
        if ((uint64_t)ifs.gcount() != si.data_start) {
            err = "short read of the header region [0, data_start) of " + si.path; return false;
        }
        std::string hex;
        if (!sha256_bytes(head.data(), head.size(), hex)) { err = "BCryptHash failed on the header region"; return false; }
        if (to_lower_hex(hex) != to_lower_hex(si.digest_sha256)) {
            err = "DF-1 source header digest mismatch for " + si.path +
                  " - the shard was replaced or its header changed since the plan (fail-closed)";
            return false;
        }
        gguf_init_params gp{};
        gp.no_alloc = true;
        gp.ctx      = nullptr;   // 헤더만 판다 — 데이터 블롭 미적재
        gguf_context * gc = gguf_init_from_buffer(head.data(), head.size(), gp);
        if (!gc) { err = "GGUF header re-parse failed for " + si.path; return false; }
        struct GgufGuard { gguf_context * c; ~GgufGuard() { if (c) gguf_free(c); } } guard{ gc };

        if ((uint64_t)gguf_get_data_offset(gc) != si.data_start) {
            err = "live GGUF data offset != sources[].data_start for " + si.path; return false;
        }
        if ((uint32_t)gguf_get_version(gc) != si.gguf_version) {
            err = "live GGUF version != sources[].gguf_version for " + si.path; return false;
        }
        if ((uint32_t)gguf_get_alignment(gc) != si.alignment) {
            err = "live GGUF alignment != sources[].alignment for " + si.path; return false;
        }
        for (const auto & L : pm.layers) {
            for (const auto & p : L.parts) {
                const int64_t id = gguf_find_tensor(gc, p.source_tensor.c_str());
                if (p.source_index != (int32_t)si_i) {
                    if (id >= 0) {
                        err = "tensor name is present in more than one shard: " + p.source_tensor +
                              " (declared source_index=" + std::to_string(p.source_index) + ", also found in " +
                              std::to_string(si_i) + ")";
                        return false;
                    }
                    continue;
                }
                if (id < 0) {
                    err = "manifest part tensor is absent from its declared shard: " + p.source_tensor +
                          " (source_index=" + std::to_string(p.source_index) + ")";
                    return false;
                }
                uint64_t live_abs = 0;
                if (!ck_add(si.data_start, (uint64_t)gguf_get_tensor_offset(gc, id), live_abs)) {
                    err = "checked uint64 overflow while re-deriving the live abs_offset of " + p.source_tensor; return false;
                }
                if (p.abs_offset != live_abs) {
                    err = "live abs_offset(" + std::to_string(live_abs) + ") != manifest abs_offset(" +
                          std::to_string(p.abs_offset) + ") for " + p.source_tensor; return false;
                }
                const int64_t * ne = gguf_get_tensor_ne(gc, id);
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    const int64_t want = (d < (int)p.dims.size()) ? p.dims[(size_t)d] : 1;
                    if (ne[d] != want) { err = "live tensor ne != manifest dims for " + p.source_tensor; return false; }
                }
                // type 대조는 **동결 표의 traits**로 한다 — 엔진에 type 이름 표를 또 복사하면
                // 그 사본이 낡는다(§2-8). manifest 의 type 문자열은 위에서 표로 검증됐고, live
                // ggml_type 의 (block_values, block_bytes)가 그 표의 값과 같아야 한다.
                uint32_t bv = 0, bb = 0;
                if (!lookup_quant_trait(p.type, bv, bb)) { err = "type not in the type-trait table: " + p.type; return false; }
                const ggml_type gt = gguf_get_tensor_type(gc, id);
                if ((uint32_t)ggml_blck_size(gt) != bv || (uint32_t)ggml_type_size(gt) != bb) {
                    err = "live tensor type traits != manifest type(" + p.type + ") traits for " + p.source_tensor;
                    return false;
                }
                uint64_t want_bytes = 0;
                if (!ck_mul(p.slice_bytes, (uint64_t)pm.n_expert, want_bytes)) {
                    err = "checked uint64 overflow while computing the live tensor size of " + p.source_tensor; return false;
                }
                if ((uint64_t)gguf_get_tensor_size(gc, id) != want_bytes) {
                    err = "live tensor size != slice_bytes * n_expert for " + p.source_tensor; return false;
                }
                out_live_type[p.source_tensor] = (int32_t)gt;
                resolved++;
            }
        }
    }
    if (resolved != total_parts) {
        err = "boot re-parse resolved " + std::to_string(resolved) + " of " + std::to_string(total_parts) +
              " manifest parts"; return false;
    }
    return true;
}

} // namespace ggml_moe_direct_impl

#endif // _WIN32
