// moe-trace: records the top-k expert selection (ffn_moe_topk) of a MoE model
// for every token position in a prefill + greedy generation run.
//
// Custom configuration is read from environment variables (the common arg
// table is intentionally left untouched):
//   MOE_TRACE_OUT  output path prefix (default: "moe_trace"). Produces
//                  "<prefix>.bin" (binary trace) and "<prefix>.meta.json"
//                  (sidecar metadata + self-verification).
//   MOE_TRACE_OFF  if set to "1", the cb_eval callback is not installed at
//                  all (overhead baseline run, no trace files written).
//   MOE_TRACE_STOP_EOS  if set to "1", stop generation naturally when an
//                  end-of-generation token is sampled; the EOS token itself
//                  is neither recorded nor decoded. Default: generate
//                  exactly -n tokens.
//   MOE_TRACE_RANDOM_N  if set to a positive integer, replace the prompt with
//                  that many random vocabulary tokens (control/EOG ids
//                  excluded, BOS not prepended) instead of tokenizing -p/-f.
//   MOE_TRACE_RANDOM_SEED  seed for the MOE_TRACE_RANDOM_N token generator
//                  (default: 12345).
//
// Prefill is submitted in n_ubatch-sized chunks (one llama_decode call per
// ubatch) so the cb_eval position labeling stays correct for prompts longer
// than n_ubatch.
//
// Binary trace format ("<prefix>.bin", little-endian):
//   header: char[4] magic = "MOET"; u32 version = 1; u32 n_layer;
//           u32 n_expert; u32 top_k
//   record (repeated, one per traced column/token): u32 pos; u16 layer;
//           u16 n_k; u32 ids[n_k]
//
// See examples/debug/debug.cpp and common/debug.cpp for the cb_eval
// ask/collect pattern this file reuses, and examples/simple/simple.cpp for
// the greedy generation loop pattern.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char * const MOE_TOPK_PREFIX     = "ffn_moe_topk";
const size_t        MOE_TOPK_PREFIX_LEN = 12; // strlen("ffn_moe_topk")

struct moe_trace_ctx {
    std::ofstream out;

    uint32_t n_layer  = 0;
    uint32_t n_expert = 0;
    uint32_t top_k    = 0;
    bool     header_written = false;

    // Set by main() right before each llama_decode() call so the callback
    // can label columns with the correct absolute token position.
    uint64_t call_base_pos = 0; // n_past before this decode call
    uint64_t call_n_tokens  = 0; // number of tokens submitted in this decode call

    uint64_t n_records_observed = 0;
    uint64_t n_id_violations    = 0; // ids >= n_expert

    std::vector<uint8_t> scratch; // reused buffer for ggml_backend_tensor_get
};

// GGML operations callback during graph execution.
// ask == true : the scheduler wants to know if we need data from this tensor.
// ask == false: the tensor has been computed, this is the collect step.
// Returning false during collect would cancel the rest of the graph compute,
// so this callback always returns true once past the ask step.
bool moe_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * mt = static_cast<moe_trace_ctx *>(user_data);

    if (ask) {
        return strncmp(t->name, MOE_TOPK_PREFIX, MOE_TOPK_PREFIX_LEN) == 0;
    }

    // parse layer index from tensor name "ffn_moe_topk-<il>"
    int il = -1;
    const char * dash = strrchr(t->name, '-');
    if (dash != nullptr) {
        il = atoi(dash + 1);
    } else {
        LOG_WRN("%s: could not parse layer index from tensor name '%s'\n", __func__, t->name);
    }

    const int64_t n_k   = t->ne[0]; // top_k (n_expert_used)
    const int64_t n_col = t->ne[1]; // number of token columns in this invocation

    if (!mt->header_written) {
        mt->top_k = (uint32_t) n_k;

        const char magic[4] = { 'M', 'O', 'E', 'T' };
        const uint32_t version = 1;
        mt->out.write(magic, sizeof(magic));
        mt->out.write(reinterpret_cast<const char *>(&version),   sizeof(version));
        mt->out.write(reinterpret_cast<const char *>(&mt->n_layer),  sizeof(mt->n_layer));
        mt->out.write(reinterpret_cast<const char *>(&mt->n_expert), sizeof(mt->n_expert));
        mt->out.write(reinterpret_cast<const char *>(&mt->top_k),    sizeof(mt->top_k));

        mt->header_written = true;
    }

    const size_t n_bytes = ggml_nbytes(t);
    mt->scratch.resize(n_bytes);
    ggml_backend_tensor_get(t, mt->scratch.data(), 0, n_bytes);
    const uint8_t * data = mt->scratch.data();

    // The last layer only computes output rows during prefill (n_col < n_tokens
    // submitted to this decode call), so the traced column(s) are always the
    // *last* n_col positions of the current decode call.
    const uint64_t pos_base = mt->call_base_pos + (mt->call_n_tokens - (uint64_t) n_col);

    for (int64_t i1 = 0; i1 < n_col; ++i1) {
        const uint32_t pos    = (uint32_t) (pos_base + (uint64_t) i1);
        const uint16_t layer  = (uint16_t) il;
        const uint16_t n_k_16 = (uint16_t) n_k;

        mt->out.write(reinterpret_cast<const char *>(&pos),    sizeof(pos));
        mt->out.write(reinterpret_cast<const char *>(&layer),  sizeof(layer));
        mt->out.write(reinterpret_cast<const char *>(&n_k_16), sizeof(n_k_16));

        for (int64_t i0 = 0; i0 < n_k; ++i0) {
            const size_t byte_off = (size_t) i1 * t->nb[1] + (size_t) i0 * t->nb[0];
            int32_t id;
            memcpy(&id, data + byte_off, sizeof(id));
            const uint32_t uid = (uint32_t) id;
            mt->out.write(reinterpret_cast<const char *>(&uid), sizeof(uid));
            if (uid >= mt->n_expert) {
                mt->n_id_violations++;
            }
        }

        mt->n_records_observed++;
    }

    return true;
}

// n_expert has no dedicated public getter in llama.h; query it from the
// loaded model's GGUF metadata instead ("<arch>.expert_count"), same key
// llama-model.cpp itself reads into hparams.n_expert.
uint32_t query_model_n_expert(const llama_model * model) {
    char arch_buf[128] = { 0 };
    const int32_t arch_len = llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));
    if (arch_len < 0) {
        LOG_ERR("%s: failed to read general.architecture from model metadata\n", __func__);
        return 0;
    }

    const std::string key = std::string(arch_buf) + ".expert_count";
    char val_buf[64] = { 0 };
    const int32_t val_len = llama_model_meta_val_str(model, key.c_str(), val_buf, sizeof(val_buf));
    if (val_len < 0) {
        LOG_ERR("%s: failed to read %s from model metadata\n", __func__, key.c_str());
        return 0;
    }

    try {
        return (uint32_t) std::stoul(val_buf);
    } catch (...) {
        LOG_ERR("%s: failed to parse expert count '%s'\n", __func__, val_buf);
        return 0;
    }
}

std::string json_escape(const std::string & s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:   r += c;      break;
        }
    }
    return r;
}

std::string token_array_to_json(const std::vector<llama_token> & toks) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < toks.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << toks[i];
    }
    oss << "]";
    return oss.str();
}

std::string current_timestamp_iso() {
    const std::time_t t = std::time(nullptr);
    std::tm tm_utc;
    gmtime_s(&tm_utc, &t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Force-disable the internal warmup pass: an untracked warmup graph
    // execution would otherwise leak extra ffn_moe_topk records into the
    // trace (confirmed empirically to contaminate the record count).
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    const char * env_out = std::getenv("MOE_TRACE_OUT");
    const std::string out_prefix = (env_out != nullptr && env_out[0] != '\0') ? env_out : "moe_trace";

    const char * env_off = std::getenv("MOE_TRACE_OFF");
    const bool trace_off = (env_off != nullptr && std::strcmp(env_off, "1") == 0);

    const char * env_stop_eos = std::getenv("MOE_TRACE_STOP_EOS");
    const bool stop_eos = (env_stop_eos != nullptr && std::strcmp(env_stop_eos, "1") == 0);

    const char * env_rand_n = std::getenv("MOE_TRACE_RANDOM_N");
    const int32_t rand_n = (env_rand_n != nullptr) ? atoi(env_rand_n) : 0;
    const char * env_rand_seed = std::getenv("MOE_TRACE_RANDOM_SEED");
    const uint32_t rand_seed = (env_rand_seed != nullptr) ? (uint32_t) std::strtoul(env_rand_seed, nullptr, 10) : 12345u;

    moe_trace_ctx trace;

    if (!trace_off) {
        params.cb_eval           = moe_trace_cb_eval;
        params.cb_eval_user_data = &trace;
    }

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string bin_path;
    if (!trace_off) {
        trace.n_layer  = (uint32_t) llama_model_n_layer(model);
        trace.n_expert = query_model_n_expert(model);

        bin_path = out_prefix + ".bin";
        trace.out.open(bin_path, std::ios::binary);
        if (!trace.out) {
            LOG_ERR("%s: failed to open trace output file: %s\n", __func__, bin_path.c_str());
            return 1;
        }
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> prompt_tokens;
    if (rand_n > 0) {
        // control/EOG ids are excluded so the random arm never injects
        // special tokens; the seed makes the prompt reproducible.
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::mt19937 rng(rand_seed);
        std::uniform_int_distribution<int32_t> dist(0, n_vocab - 1);
        prompt_tokens.reserve(rand_n);
        while ((int32_t) prompt_tokens.size() < rand_n) {
            const llama_token id = dist(rng);
            if (llama_vocab_is_control(vocab, id) || llama_vocab_is_eog(vocab, id)) {
                continue;
            }
            prompt_tokens.push_back(id);
        }
    } else {
        prompt_tokens = common_tokenize(ctx, params.prompt, add_bos);
    }

    if (prompt_tokens.empty()) {
        LOG_ERR("%s: no input tokens - provide a prompt with -p or -f\n", __func__);
        return 1;
    }

    const uint64_t n_prompt = prompt_tokens.size();

    // cb_eval fires once per ubatch graph, and the callback labels positions
    // from the per-call {call_base_pos, call_n_tokens} pair. Submitting more
    // than n_ubatch tokens in one llama_decode call would split into multiple
    // graphs and mislabel every column, so each call submits exactly one
    // ubatch worth of tokens.
    uint64_t n_prefill_calls = 0;
    {
        const uint64_t n_ub = params.n_ubatch > 0 ? (uint64_t) params.n_ubatch : 512;
        uint64_t done = 0;
        while (done < n_prompt) {
            const uint64_t n_chunk = std::min(n_ub, n_prompt - done);
            if (!trace_off) {
                trace.call_base_pos = done;
                trace.call_n_tokens  = n_chunk;
            }
            if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt_tokens.data()) + done, (int32_t) n_chunk))) {
                LOG_ERR("%s: prefill decode failed at token offset %llu\n", __func__, (unsigned long long) done);
                return 1;
            }
            done += n_chunk;
            n_prefill_calls++;
        }
    }

    uint64_t n_past = n_prompt;

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const int n_gen = params.n_predict > 0 ? params.n_predict : 32;

    std::vector<llama_token> gen_tokens;
    gen_tokens.reserve(n_gen);
    bool stopped_at_eos = false;

    LOG("\n");

    const int64_t t_gen_start = ggml_time_us();

    for (int i = 0; i < n_gen; ++i) {
        // By default llama_vocab_is_eog is intentionally ignored: we always
        // generate exactly n_gen tokens to keep the expected record count
        // exact. If MOE_TRACE_STOP_EOS=1, stop naturally on EOS instead; the
        // EOS token itself is neither pushed, printed, nor decoded, so
        // expected_decode = n_layer * gen_tokens.size() stays exact.
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
        if (stop_eos && llama_vocab_is_eog(vocab, new_token)) {
            stopped_at_eos = true;
            break;
        }
        gen_tokens.push_back(new_token);

        char piece_buf[128];
        const int n_piece = llama_token_to_piece(vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (n_piece >= 0) {
            std::string piece(piece_buf, n_piece);
            printf("%s", piece.c_str());
            fflush(stdout);
        }

        if (!trace_off) {
            trace.call_base_pos = n_past;
            trace.call_n_tokens  = 1;
        }

        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            LOG_ERR("\n%s: decode failed at generation step %d\n", __func__, i);
            break;
        }

        n_past += 1;
    }

    const int64_t t_gen_end = ggml_time_us();
    const double decode_ms = (t_gen_end - t_gen_start) / 1000.0;

    printf("\n");

    const uint64_t expected_prefill = trace.n_layer > 0 ? (uint64_t) (trace.n_layer - 1) * n_prompt + n_prefill_calls : 0;
    const uint64_t expected_decode  = (uint64_t) trace.n_layer * (uint64_t) gen_tokens.size();
    const uint64_t expected_total   = expected_prefill + expected_decode;
    const bool     records_match    = !trace_off && (trace.n_records_observed == expected_total);

    if (!trace_off) {
        LOG("moe-trace: expected records = %llu, observed = %llu, match = %s\n",
            (unsigned long long) expected_total, (unsigned long long) trace.n_records_observed,
            records_match ? "true" : "false");
        LOG("moe-trace: id violations (id >= n_expert) = %llu\n", (unsigned long long) trace.n_id_violations);

        if (!records_match) {
            LOG_WRN("%s: record count mismatch (expected %llu, observed %llu)\n",
                __func__, (unsigned long long) expected_total, (unsigned long long) trace.n_records_observed);
        }
        if (trace.n_id_violations > 0) {
            LOG_WRN("%s: %llu expert id(s) >= n_expert observed\n",
                __func__, (unsigned long long) trace.n_id_violations);
        }

        trace.out.flush();
        trace.out.close();

        std::ostringstream js;
        js << "{\n";
        js << "  \"model_path\": \"" << json_escape(params.model.path) << "\",\n";
        js << "  \"n_layer\": " << trace.n_layer << ",\n";
        js << "  \"n_expert\": " << trace.n_expert << ",\n";
        js << "  \"top_k\": " << trace.top_k << ",\n";
        js << "  \"n_prompt\": " << n_prompt << ",\n";
        js << "  \"n_prefill_calls\": " << n_prefill_calls << ",\n";
        js << "  \"random_mode\": " << (rand_n > 0 ? "true" : "false") << ",\n";
        js << "  \"random_seed\": " << rand_seed << ",\n";
        js << "  \"n_gen\": " << gen_tokens.size() << ",\n";
        js << "  \"stop_eos_enabled\": " << (stop_eos ? "true" : "false") << ",\n";
        js << "  \"stopped_at_eos\": " << (stopped_at_eos ? "true" : "false") << ",\n";
        js << "  \"prompt_tokens\": " << token_array_to_json(prompt_tokens) << ",\n";
        js << "  \"generated_tokens\": " << token_array_to_json(gen_tokens) << ",\n";
        js << "  \"validation\": {\n";
        js << "    \"expected_records\": " << expected_total << ",\n";
        js << "    \"observed_records\": " << trace.n_records_observed << ",\n";
        js << "    \"records_match\": " << (records_match ? "true" : "false") << ",\n";
        js << "    \"id_violations\": " << trace.n_id_violations << "\n";
        js << "  },\n";
        js << "  \"decode_ms\": " << decode_ms << ",\n";
        js << "  \"timestamp\": \"" << current_timestamp_iso() << "\"\n";
        js << "}\n";

        const std::string meta_path = out_prefix + ".meta.json";
        std::ofstream meta_out(meta_path);
        if (!meta_out) {
            LOG_ERR("%s: failed to open meta output file: %s\n", __func__, meta_path.c_str());
        } else {
            meta_out << js.str();
        }

        LOG("moe-trace: bin written to %s\n", bin_path.c_str());
        LOG("moe-trace: meta written to %s\n", meta_path.c_str());
    }

    LOG("\nmoe-trace: decode loop time = %.2f ms for %zu tokens\n", decode_ms, gen_tokens.size());

    llama_sampler_free(smpl);
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
