#pragma once

#include <cstdint>
#include <string>

// counters / accounting  (single definition lives in selftest-harness.cpp)
extern int g_total;
extern int g_failed;
extern int g_groups_run;
extern int g_groups_skipped;

// filter
extern std::string g_filter;

// core harness
bool check(bool cond, const std::string & name);
void run_group(const char * tag, void (*fn)());
void section(const std::string & title);

// text / env helpers
std::wstring utf8_to_wide(const std::string & s);
std::string  wide_to_utf8(const std::wstring & w);
void set_env(const char * name, const std::string & value);
void clear_env(const char * name);

// scratch subsystem  (HANDOFF_DEV section 2-10)
extern std::wstring g_scratch_root;
bool remove_tree_w(const std::wstring & dir, uint64_t * out_bytes);
bool pid_is_alive(unsigned long pid);
int  sweep_stale_scratch(uint64_t * out_bytes, bool quiet);
bool teardown_scratch_root(bool success);
// `announce` exists for ONE caller: the --xbe-perf-ab child, whose stdout is a parsed
// protocol and not a log. The default is true, so every other selftest path keeps the
// line it has always printed - the quiet is scoped to the caller that asks for it, and
// the output contract of the other consumers is not touched (Codex e2b2_perf_arm r3 F1).
void init_scratch_root(bool announce = true);
std::wstring make_case_dir(const std::string & tag);
