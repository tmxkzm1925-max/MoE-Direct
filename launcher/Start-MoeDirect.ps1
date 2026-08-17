#Requires -Version 5.1
<#
    Start-MoeDirect.ps1 - MoE-Direct release launcher.

    Authority (frozen, do not re-interpret):
      LAUNCHER_SPEC.md v0.4 (FROZEN 26-07-30, commit 7e2cb41)  -> "LS <section>"
      RELEASE_SPEC.md  v0.1 (FROZEN)                           -> "RS <section>"
      bench/techdev/SPEC_PREFETCH_INIT.md      v1.0 (FROZEN)   -> "PI <section>"
      bench/techdev/SPEC_PREFETCH_P4_LAUNCHER.md v1.0 (FROZEN) -> "P4 <section>"
      PI/P4 supersede LS on the prefetch surface ONLY, and only where LAUNCHER_SPEC.md's own
      "later-authority" clause lists it (LS 1-2 prefetch state/wire, LS 5 catalog semantic
      failure disposition, LS 7 engine prefetch_state description, LS 12-3 selection signal,
      LS 15 derived-profile prefetch fields). Every other v0.4 contract is unchanged.

    Wire contract (consumer-visible, implementation has no discretion):
      - final stderr line: "[moe-launcher] status=<enum>"  exactly one line   (LS 5)
      - status enum (17) and exit code mapping                                (LS 5)
      - effective_prefetch echo strings                                       (LS 1-2)
      - engine policy anchor "[moe-direct] startup_reject=engine_policy_gate" (LS 5 / LS 7)
      - user preset required fields + schema_version exact-match              (LS 1-7)

    Precedent reused (no new invention, per build order):
      - kill-on-close Job Object          bench/moe-direct/moe_serve.ps1 :582-607
      - PID-bound health                  bench/moe-direct/moe_serve.ps1 :695-711
      - single-instance named mutex       bench/moe-direct/moe_serve.ps1 :802-807
      - CREATE_NEW_PROCESS_GROUP + GenerateConsoleCtrlEvent(CTRL_BREAK)
                                          bench/moe-direct/SPEC.md KS3 (:194)
      - multi-shard discovery rules       bench/repack/repack_experts.py :252-322

    All output is English ASCII (LS 8). File must stay UTF-8 with BOM (PS 5.1 CP949 hazard).
#>

[CmdletBinding()]
param(
    # Model GGUF path (any shard of a split set is accepted; siblings are discovered).
    [string] $Model,
    # Bundle root (defaults to this script's directory).
    [string] $BundleRoot,
    # Repack output directory (defaults to "<model dir>\repack").
    [string] $OutDir,
    # LS 11 (UI-1 3-c): extra root scanned for *.gguf candidates in the first-run selection menu.
    # CLI only - no setting file, no preset field, and it never reaches the child argv/env.
    [string] $ModelsRoot,
    # RS 8 first-run smoke checklist (1..7), then teardown.
    [switch] $Smoke,
    # LS 13-2 (WS-1 / WARMSTART_SPEC A-6): a reproducibility or benchmark run. It decides exactly
    # two things: warmstart hard-OFF, and - since BUDGET_AUTOTUNE_SPEC v0.2 section 2 - that the RAM
    # budget autotune is forbidden, so the budget falls back to the profile's own min_budget_mb and
    # two machines size the same run identically. Nothing else: the QD sweep, the probe binding and
    # the preset round trip are still not touched by it. -Smoke and -Repro may be given together
    # (both converge on hard-OFF); -Smoke alone does NOT disable the autotune.
    [switch] $Repro,
    # Always show the repack plan / expectation block, even on later runs.
    [switch] $Plan,
    # Never prompt. Menu answer comes from -Action, confirmations from -AssumeYes/-AssumeNo.
    [switch] $NonInteractive,
    [switch] $AssumeYes,
    [switch] $AssumeNo,
    # R1-9: taken as raw strings on purpose. A [ValidateSet]/[int] binder failure terminates the
    # script before any status line can be written, which would produce a zero-status-line exit.
    [string] $Action = 'start',
    # Unattended serve duration in seconds (-NonInteractive only). 0 = stop right after ready.
    [string] $RunSeconds = '0',
    # Discard the stored user preset before loading (LS 1-7 "reset").
    [switch] $ResetPreset,
    # Allowlist overrides (LS 1-2). Declared as [string] on purpose: parameter-binder type
    # failures would terminate before a status line could be emitted, so this script parses and
    # bounds-checks them itself and reports fail_custom_args.
    [string] $Port,
    [string] $Ctx,
    [string] $Threads,
    [string] $BudgetMB,
    [string] $QD,
    [string] $Warmup,
    # LS 13-2 (WS-1): the soft-OFF override channel. Same three layers (CLI / preset / interactive
    # custom) and the same parsing discipline as the six keys above; on|off, absent = on. It can
    # never raise a hard-OFF mode back to ON (the mode decision is above this layer).
    [string] $Warmstart,
    # LS 13-8 (AUTOSAVE_SPEC 2-D): the periodic crash-recovery save. on | off | <minutes>, absent =
    # on at the default period. Same three layers and the same raw-string parsing discipline as the
    # keys above. It is subordinate to warmstart: hard-OFF and soft-OFF turn autosave off as well,
    # and this key can never raise either of them back on.
    [string] $Autosave,
    # P4 4: the prefetch opt-in surface. catalog | init | adapt, absent = catalog (v0.4 behaviour).
    # Exactly two layers carry it - this CLI parameter and the stored preset - because PI 3
    # invariant 3 limits the opt-in to "the launcher's explicit CLI/preset only". It is deliberately
    # NOT offered by Invoke-CustomEditor and there is no path that changes it after the status
    # screen. Raw string for the same reason as the six keys above: a binder failure would kill the
    # run before a status line exists, so this script parses it and reports fail_custom_args.
    [string] $Prefetch,
    # RV 3: the repack mode opt-in. packed (and absence) = the v0.4 bin repack; virtual = the
    # schema 3.0 in-place plan. Raw string for the same reason as the keys above - a [ValidateSet]
    # binder failure would terminate before a status line exists, so this script parses it itself
    # and reports fail_custom_args.
    # It is NOT an allowlist override key and it deliberately takes no part in Get-CliOverrides or
    # $overrides (RV 3 prohibition): those feed Test-CustomProvenance, and a mode selection is not
    # a custom performance value.
    [string] $RepackMode,
    # UX 1-1-2: the canonical arch-template control. Taken as a raw [string] for the same reason as
    # the six allowlist keys above - a [ValidateSet] binder failure would terminate before a status
    # line could be written. It is NOT an allowlist override key: it is resolved BEFORE model
    # identification (a preset is profile-bound and cannot exist that early), so it travels the
    # global preference file instead of the preset. on | off, absent = the resolution order below.
    # Naming note: Invoke-Repacker has its own [bool] $ArchTemplate parameter with an explicit
    # default, so it shadows this one inside that function and never reads it.
    [string] $ArchTemplate,
    # OPEN_ARCH C axis (LS OA-1): the ORIGINAL private entry point to the arch-template path.
    # DEPRECATED by UX 1-1-2 - kept working for anyone who scripted it, but it is now only the
    # second rung of the resolution order: it maps to 'on' when the canonical -ArchTemplate is
    # absent, and "-ExperimentalArchTemplate -ArchTemplate off" resolves to off.
    [switch] $ExperimentalArchTemplate,
    # Dot-source hook for launcher_selftest.ps1: define everything, run nothing.
    [switch] $LibraryMode
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Off

# ============================================================================
# region 1. WIRE CONSTANTS (LS 5 / LS 1-2 / LS 7) - frozen, not tunable
# ============================================================================

# status enum -> exit code. 17 entries. No status outside this table may be emitted.
$script:STATUS_EXIT = [ordered]@{
    'ok'                     = 0
    'ok_smoke'               = 0
    'cancelled_user'         = 2
    'fail_model_path'        = 3
    'fail_resource'          = 3
    'fail_instance_lock'     = 3
    'fail_partial_cleanup'   = 3
    'fail_repack'            = 3
    'fail_custom_args'       = 3
    'fail_gate_bundle'       = 4
    'fail_gate_catalog'      = 4
    'fail_gate_verify'       = 4
    'fail_gate_engine_seal'  = 4
    'fail_server_start'      = 5
    'fail_runtime_exit'      = 5
    'fail_teardown'          = 5
    'fail_smoke'             = 6
}

$script:STATUS_LINE_PREFIX = '[moe-launcher] status='

# LS 5 / LS 7 : complete ASCII line, exact match only, substring detection forbidden.
$script:ENGINE_POLICY_ANCHOR = '[moe-direct] startup_reject=engine_policy_gate'

# R2-6 engine seal SUCCESS line. 1st source (verified):
#   bench/moe-direct/repro/moedirect-v2-b10057.patch:14681
#   LLAMA_LOG_INFO("%s: moe-direct: sealed all=%d host=%d nonhost=%d slots=%d/%d moe_layers=%zu"
#                  " (matched live=%d host=%d nonhost=%d)\n", ...)
# emitted once, immediately after ggml_moe_direct_seal() returns >= 0.
# NOTE Why this is matched differently from the policy anchor: the policy anchor is a frozen fixed
# wire with no variable fields, so it is required to be an EXACT complete line and substring
# detection is forbidden. This success line intentionally carries variable numeric fields AND is
# emitted through the engine's log framework, which prefixes it (real capture:
# "0.09.636.808 I load_tensors: moe-direct: sealed all=..."). It is therefore matched as a marker
# CONTAINED IN a complete (newline-terminated) line - a line-start anchor would never fire on a
# real run. That is a documented difference in kind, not a relaxation of the anchor rule.
$script:ENGINE_SEAL_MARKER      = 'moe-direct: sealed all='
# Parsed for the diagnostic log only. slots=X/Y is NOT an equality invariant: a real passing run
# emitted slots=648/128 (attested slots vs required slots are different quantities).
# Counter-evidence: bench_results/g4_1a/20260724T223305Z_32dc7208/
#   srv_err_on1_attempt0_20260724T223532Z.log:2249
#   "... moe-direct: sealed all=216 host=174 nonhost=42 slots=648/128 moe_layers=36 (matched ...)"
# The seal's own fail-close already happened upstream (seal_rc < 0 aborts startup), so the
# presence of this line exactly once IS the attestation.
$script:ENGINE_SEAL_SLOTS_REGEX = 'slots=(\d+)/(\d+)'
$script:ENGINE_SEAL_COUNTS_REGEX = 'sealed all=(\d+) host=(\d+) nonhost=(\d+)'

# R2-4 / R3-1 cancel evidence, bound to a TASK ID. 1st sources (verified):
#   tools/server/server-queue.cpp:441   server_response_reader::stop()
#       SRV_WRN("cancel task, id_task = %d\n", id_task)
#       SRV_WRN prefix (server-common.h:28) = "srv  %12.*s: "
#       -> "srv          stop: cancel task, id_task = 12"
#   tools/server/server-context.cpp:492 server_slot::release()
#       SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", ...)
#       SLT_INF prefix (server-common.h:20) = "slot %12.*s: id %2d | task %d | "
#       -> "slot      release: id  0 | task 12 | stop processing: n_tokens = 24, truncated = 0"
# NOTE Counting release lines is NOT sufficient (R3 finding). The INFO logger flushes from its own
# queue/worker thread (common/log.cpp:200,267), so the PREVIOUS request's release line can appear
# in stderr after the launcher has taken its baseline. A total-count comparison then mistakes that
# late line for the cancelled request's release - a false positive that even a server ignoring the
# disconnect could satisfy. So the evidence must be bound: find the cancel warning for a task, then
# require that same task's release line to appear AFTER it.
$script:SLOT_RELEASE_MARKER  = 'stop processing:'
$script:CANCEL_TASK_REGEX    = 'cancel task, id_task = (\d+)'
$script:TASK_RELEASE_REGEX   = '\|\s*task\s+(\d+)\s*\|\s*stop processing:'

# UI-9 prefill progress line (DISPLAY ONLY - never a gate, never part of the wire). Captured shape,
# 1st source bench_results/g2/g2_3/srv_err_20260719-063645.log:21-24,46-47:
#   "4.53.529.639 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =   2048,
#    progress = 0.33, t = 277.07 s / 7.39 tokens per second"
# Groups: 1=n_tokens 2=progress 3=tokens per second (the t= field is matched but not captured).
# Matched as a fragment inside a complete line, like ENGINE_SEAL_MARKER and for the same reason: the
# engine's log framework prefixes the line. An ANSI colour escape, when present, sits at the start of
# the line only, so it can never fall inside this pattern - no stripping is needed.
$script:PREFILL_PROGRESS_REGEX = 'prompt processing, n_tokens =\s*(\d+), progress =\s*([0-9.]+), t =\s*[0-9.]+ s / ([0-9.]+) tokens per second'
# UI-9b request-boundary tag: captures task id from the same line, used only to tell whether two
# progress lines belong to the same request. Display only, same as PREFILL_PROGRESS_REGEX.
$script:PREFILL_TASK_REGEX = '\|\s*task\s+(\d+)\s*\|\s*prompt processing,'

# LS 1-2 : effective_prefetch echo strings (wire). R6 revision: 4 reasons, not 3.
$script:PREFETCH_ECHO_ON                = 'on'
# R6: the engine turns prefetch OFF by itself when N >= qd_effective (invalid_range, K/N=0), so a
# launcher that still echoed "on[K8 N4]" was reporting a state the engine never entered. This is
# the control-plane half of that fix: below the depth the launcher declares OFF and injects no K/N.
$script:PREFETCH_ECHO_QD_BELOW_DEPTH    = 'off(reason=qd_below_prefetch_depth)'

# ---------------------------------------------------------------------------
# P4 2 : the CLOSED off_reason enum (wire). 13 literals, exhaustively listed - a reason string
# outside this table may not be emitted, and every echo is exactly "off(reason=<literal>)".
# The two v0.4 one-axis literals 'reference_only_live_forbidden' and 'catalog_disabled' are
# RETIRED with the one-axis prefetch_state field they described; nothing in this generation emits
# them (selftest asserts their absence).
# ---------------------------------------------------------------------------
$script:PREFETCH_OFF_REASONS = @(
    # carried over from v0.4 (unchanged meaning)
    'probe_failed',
    'qd_below_prefetch_depth',
    # P4 1-b / 2 : the catalog row parsed but says something impossible
    'catalog_semantic_invalid',
    # P4 2.5 : catalog-fixed demands exact catalog identity, not a header fingerprint
    'identity_not_exact',
    # P4 3 : a derived profile whose plan text and GGUF header disagree about n_expert_used
    'derived_t_mismatch',
    # P4 2 step 4 : init opt-in refusals
    'init_t_out_of_range',
    'engine_env_k_floor_8_pre_p4a',
    'phase4_hold_unresolved',
    # P4 5 : adapt is refused on every path in Phase 4
    'adapt_forbidden_in_repro_bench',
    'adapt_controller_not_shipped_phase5',
    # P4 2 step 3 : not opted in - the row's evidence value substituted into one token
    'not_opted_in_evidence_observe',
    'not_opted_in_evidence_unverified',
    'not_opted_in_evidence_causal_replay')

$script:PREFETCH_ECHO_PROBE_FAILED      = 'off(reason=probe_failed)'

# PI 3 : the two catalog-stored axes. capability is NOT one of them - it is a seal-time runtime
# result and storing it would make a forgeable field out of an unforgeable one.
$script:PREFETCH_EVIDENCE_VALUES   = @('unverified', 'observe', 'causal-replay', 'paired-live')
# P4 1 closure rule: the catalog may only ever store these two. 'opt-in-fixed' / 'opt-in-adaptive'
# are RUNTIME activations computed from an opt-in, so a catalog that stores one is a semantic
# defect, not a shortcut into the opt-in path.
$script:PREFETCH_ACTIVATION_STORED  = @('off', 'catalog-fixed')
$script:PREFETCH_ACTIVATION_RUNTIME = @('opt-in-fixed', 'opt-in-adaptive')

# P4 4 : the public opt-in surface. 'catalog' (and absence) normalise to the internal arm 'none';
# that mapping is written exactly once, in ConvertTo-PrefetchOptIn.
$script:PREFETCH_REQUEST_VALUES = @('catalog', 'init', 'adapt')
$script:PREFETCH_REQUEST_DEFAULT = 'catalog'
# Internal arms. 'none' = no opt-in; the other two are the two ways a K/N pair can be produced.
$script:PREFETCH_ARM_NONE    = 'none'
$script:PREFETCH_ARM_CATALOG = 'catalog-fixed'
$script:PREFETCH_ARM_INIT    = 'init'

# PI 3 invariant 7 : provenance labels. 'env-override' is reserved and has NO issuing path in this
# atomic step (the offline explicit override is opened together with the engine env K range).
$script:PREFETCH_PROVENANCE_CATALOG = 'catalog-validated'
$script:PREFETCH_PROVENANCE_INIT    = 'init_v1-unvalidated'
$script:PREFETCH_PROVENANCE_ENV     = 'env-override'
# PI 2 : the versioned prior. The version travels with every init-produced pair so a number can
# never be read as "the validated value".
$script:PREFETCH_INIT_VERSION = 'prefetch_init_v1'
$script:PREFETCH_INIT_N_CAP   = 4
$script:PREFETCH_INIT_T_MIN   = 1
# PI 4-1 : pred wire is a top-16 ABI (SPEC_PHASEB_WIRE 13.2 'pred u16[16]').
$script:PREFETCH_INIT_T_MAX   = 16
# ---------------------------------------------------------------------------
# P4 2 : TEMPORARY launcher-side copy of the engine's env K range floor.
# 1st source (verified): bench/moe-direct/repro/moedirect-v2-b10057.patch:11054 - the engine
# rejects MOE_DIRECT_PREFETCH_K outside "kv < 8 || kv > 16". This constant does NOT decide
# capability (the seal does); it exists so the launcher cannot print a candidate ON for a family
# whose t is below the floor the engine would reject. The follow-on atomic step that widens the
# engine range to 1..16 changes the engine and THIS constant in one commit and one test run.
# ---------------------------------------------------------------------------
$script:ENGINE_ENV_K_FLOOR = 8
# P4 2.5 : catalog-fixed turns ON only for an exactly identified model. Resolve-ProfileSelection
# already answers this question; 'pinned' is the only verdict that means "the file bytes are the
# ones the catalog measured".
$script:PREFETCH_IDENTITY_EXACT = 'pinned'

# P4 2 : one formatter for the whole enum, so an off reason cannot be spelled two ways.
function Get-PrefetchOffEcho {
    param([string] $Reason)
    if ($script:PREFETCH_OFF_REASONS -cnotcontains $Reason) {
        # Unreachable by construction (every caller passes a literal from the table above); it is a
        # loud internal error rather than a silent new wire string.
        Stop-Launcher 'fail_gate_catalog' ('internal: off reason outside the closed enum: ' + $Reason)
    }
    return ('off(reason=' + $Reason + ')')
}

# LS 1-7 : user preset required fields + exact schema version.
# LS 13-2: PRESET_SCHEMA_VERSION stays 1. The unknown-key drop rule already gives both directions
# of compatibility for the added 'warmstart' key (absent = on), so bumping it - which would discard
# every stored user preset - buys nothing.
$script:PRESET_SCHEMA_VERSION = 1
$script:PRESET_REQUIRED_FIELDS = @('schema_version', 'source_tag', 'profile_id', 'expect_digest', 'overrides')
# P4 4: 'prefetch' joins the allowlist as the second (and last) opt-in surface. The unknown-key
# drop rule already gives both compatibility directions, so PRESET_SCHEMA_VERSION still stays 1.
$script:PRESET_ALLOWLIST_KEYS  = @('port', 'ctx', 'threads', 'warmup', 'budget_mb', 'qd', 'warmstart', 'autosave', 'prefetch')

# LS 13-2: override keys that take no part in argv/env and cannot change a performance condition.
# They are excluded from the "is this a custom configuration" decision, so following the README's
# advice to switch warmstart off does not demote the user's performance gate to [unmeasured].
# Transparency for these keys is carried by their own status line instead (the 'kv :' line).
$script:PERF_NEUTRAL_OVERRIDE_KEYS = @('warmstart')

# ---------------------------------------------------------------------------
# WARMFILE_DESIGN v0.2 section 1 - the third value the single 'warmup' key accepts.
# No new setting key: 'off' / 'on' / 'file:<path>' are three modes of the one key that already
# travels the CLI / stored preset / interactive custom layers, with the priority rule unchanged.
# Only the PREFIX is case-insensitive; the path after it is kept exactly as the user wrote it.
# ---------------------------------------------------------------------------
$script:WARMUP_FILE_PREFIX = 'file:'
# The warmfile request is a full cold prefill of the whole file, so it is bounded far above the
# generic one-token warmup's 300 s. Measured anchor, 1st source
# bench_results/g2/g2_3/srv_err_20260719-063645.log:21 - a real run reported
# "prompt processing, n_tokens = 2048, progress = 0.33, t = 277.07 s / 7.39 tokens per second",
# i.e. an ordinary few-thousand-token file alone reaches 300 s. Still bounded: a server that never
# answers cannot hold the launcher for ever, and a timeout is a degraded, non-terminal branch like
# every other warmup failure (RS 5).
$script:WARMFILE_TIMEOUT_SEC = 1800

# LS 2 : the four artifacts deleted when a .partial marker is found. BIN ONLY - a virtual output
# directory holds none of these, and RV 1-1 gives it its own set below.
$script:PARTIAL_DELETE_SET = @('experts.bin.partial', 'experts.bin', 'manifest.json', 'verify_report.json')

# ---------------------------------------------------------------------------
# RV (SPEC_LAUNCHER_VIRTUAL_R1, FROZEN v1.0) - the virtual repack opt-in surface.
# ---------------------------------------------------------------------------
# RV 3 : the two request values. Manual validation, no [ValidateSet] (see the parameter comment).
$script:REPACK_MODE_PACKED  = 'packed'
$script:REPACK_MODE_VIRTUAL = 'virtual'
$script:REPACK_MODE_VALUES  = @('packed', 'virtual')
$script:REPACK_MODE_DEFAULT = 'packed'
# RV 1 : the three outcomes of the manifest mode detector - the engine's own truth table
# (detect_manifest_mode, ggml-moe-direct.cpp:2110-2137). 'unrecognized' is a fail-close, not a
# fourth mode: nothing may be served on it.
$script:MANIFEST_MODE_BIN          = 'bin'
$script:MANIFEST_MODE_VIRTUAL      = 'virtual'
$script:MANIFEST_MODE_UNRECOGNIZED = 'unrecognized'
$script:MANIFEST_MODE_UNRECOGNIZED_REASON = 'mode: unrecognized manifest schema/mode'
# RV 1-1 [2] : the default output directory leaf per mode. The two modes never share a default
# directory, so an opt-in run cannot silently consume the other mode's artifacts.
$script:REPACK_DIR_PACKED  = 'repack'
$script:REPACK_DIR_VIRTUAL = 'repack-virtual'
# RV 1-1 [3] : the incomplete-production set of a VIRTUAL output directory. The repacker promotes
# plan_report.json first and manifest.json.partial -> manifest.json second, so an interruption
# between the two atomic replacements leaves exactly these (repack_experts.py:3997).
# PARTIAL_DELETE_SET is the bin set and must NOT be reused here.
$script:VIRTUAL_PARTIAL_DELETE_SET = @('plan_report.json', 'manifest.json.partial')
# RV 2-4 : the shape the preview is pinned to - the f3c GO run measured io_qd_total = 8. Every QD
# request path is refused in virtual rather than folded in, so this is the effective QD by
# construction and not a default that something later can outrank.
$script:VIRTUAL_PINNED_QD = 8
$script:VIRTUAL_PIN_REASON = 'virtual preview pins the measured shape'
# RV 2-5 : the disk reservation for a virtual repack. A virtual run moves 0 bytes of expert data;
# its whole output is manifest.json + plan_report.json (6.11 MiB on the preview's own model), so
# the bin path's expert_bytes_total (~65 GiB) would be a false resource refusal. 128 MiB is a
# conservative reservation for the v0.3 preview and the CURRENT catalog - it is not a mathematical
# upper bound for an arbitrary schema 3.0 output.
$script:VIRTUAL_REPACK_RESERVE_MB = 128

# Catalog schema (see report: schema is launcher-side until RELEASE_SPEC required item 2 lands).
# P4 1: bumped 1 -> 2 with the one-axis prefetch_state field's replacement by the two stored axes.
# The launcher and the catalog ship in the same bundle, so the version is an exact match and an old
# launcher meeting a new catalog stops at fail_gate_catalog instead of guessing.
$script:CATALOG_SCHEMA_VERSION = 2
$script:CATALOG_FILE_NAME      = 'models.json'
$script:BUNDLE_MANIFEST_NAME   = 'bundle_manifest.json'
$script:BUNDLE_MANIFEST_VERSION = 1

# ---------------------------------------------------------------------------
# OPEN_ARCH C axis (OPEN_ARCH_DESIGN.md v0.2 sections 0/3, LAUNCHER_SPEC OA-1).
# ---------------------------------------------------------------------------
# M5 atomic activation token. The repacker, the engine and this launcher must all carry the SAME
# string or the bundle assembly fails - a standing assembler gate, not tied to one release. The
# other two adopt it in their own rounds; the value is frozen here and quoted by LS OA-1.
$script:OPEN_ARCH_TEMPLATE_ABI = 'open-arch-template/1'
# The derived profile is a schema of its own, NOT a catalog profile with invented fields. Its
# validator rejects hf_repo / hf_revision outright: there is no upstream repository for a model the
# user derived on the spot, and writing a plausible-looking one would be a fabricated provenance.
$script:DERIVED_PROFILE_SCHEMA_VERSION = 1
# Written by the repacker into the repack OUTPUT directory, never into the bundle expects dir
# (repack_experts.py:162 / :1072-1081).
$script:DERIVED_EXPECT_FILE_NAME = 'derived.expect.json'
# The profile id doubles as a kv directory name (Get-KvProfileDir) and as a lock token, and the arch
# it embeds comes from a GGUF the user downloaded - untrusted input. Both are therefore constrained
# to a path-safe token, and the id is checked against the full shape before anything uses it.
$script:ARCH_TOKEN_REGEX         = '^[a-z0-9._-]{1,32}$'
$script:DERIVED_PROFILE_ID_REGEX = '^derived-[a-z0-9._-]{1,32}-[0-9a-f]{16}$'
# reference_lock.profile_id the repacker writes in template mode (repack_experts.py:1067-1069).
$script:DERIVED_LOCK_ID_PREFIX = 'arch-template:'
# Length of the derivation digest carried in the profile id. The digest itself is the full
# inventory_sha256; 16 hex is what goes into the id so the directory name stays short while the
# collision domain stays far beyond the number of models one machine will ever derive.
$script:DERIVED_DIGEST_CHARS = 16

# ---------------------------------------------------------------------------
# UX 1-3: the arch families the REPACKER carries a template for - the key set of ARCH_TEMPLATES
# (repack_experts.py:162). Exactly TWO consumers are allowed to read it, and the launcher selftest
# asserts exact-set equality against the repacker table so a drift cannot ship:
#   1. the model-menu family label (UX 1-3)
#   2. the early admissibility gate at the selection call (UX 1-1-3)
# The template DERIVATION logic is deliberately NOT duplicated here - that would be the "second
# GGUF/template parser" the derive-plan note below (region 7b) forbids. A membership set is not a
# parser: it decides admissibility, never what the inventory contains.
$script:ARCH_TEMPLATE_FAMILIES = @('gpt-oss', 'qwen35moe', 'deepseek2')

# ---------------------------------------------------------------------------
# UX 1-1-1: the arch-template preference is a GLOBAL launcher preference, not a preset override.
# Reason (frozen): its only consumer is the model-identification step, and identification runs
# BEFORE the CLI override parse, the preset load, the effective config and the interactive custom
# editor. A preset is bound to a source_tag/profile_id/expect_digest triple that does not exist yet
# at that point, so a stored preset value could never reach the gate that needs it - it would show
# "off" on the status screen while an existing template repack was already being served.
$script:ARCH_TEMPLATE_PREF_FILE_NAME = 'arch_template_pref.json'
$script:ARCH_TEMPLATE_PREF_SCHEMA_VERSION = 1
# Two fields, both required, nothing else accepted. Deliberately NOT the preset's 5-field schema:
# this file is machine-owned, tiny, and its whole job is to survive a version skew as either a
# clean value or a clean discard.
$script:ARCH_TEMPLATE_PREF_REQUIRED_FIELDS = @('schema_version', 'arch_template')
# The product default once the file is absent (UX 1-1-2 rung 4).
$script:ARCH_TEMPLATE_DEFAULT = 'on'
# UX 1-1-1 discard policy: file ABSENT = the product default; file PRESENT but rejected by the
# strict load = 'off' for this run (fail-close). The asymmetry is the point. needRepack=false skips
# the repack confirmation entirely, so a damaged preference that failed OPEN would re-enable the
# serving of an existing template artifact with no gate in front of it. Recovery is the explicit
# CLI '-ArchTemplate on' only.
$script:ARCH_TEMPLATE_PREF_DISCARD_VALUE = 'off'
# The one line every interactive writer prints instead of acting, once the discard has latched.
# Saying it out loud matters: silently hiding the toggle would look like a missing feature rather
# than a deliberate lock, and the user needs to be told which command reopens it.
$script:ARCH_TEMPLATE_DISCARD_LOCK_NOTE =
    'arch template stays off for this run (the stored preference failed its strict load); it can only be re-enabled with -ArchTemplate on'

# UX 1-3 label vocabulary. Provisional by construction: the final catalog verdict also weighs shard
# count, per-shard bytes and the source pin (Get-StructuralProfileCandidates), none of which a
# one-file header read can see.
$script:LABEL_CATALOG           = '[catalog]'
$script:LABEL_TEMPLATE_PREFIX   = '[template: '
$script:LABEL_UNSUPPORTED       = '[unsupported]'
$script:LABEL_IDENTIFY_PENDING  = '[identify pending]'
$script:LABEL_PROVISIONAL_NOTE  = 'labels are provisional; final identification happens at start'

# ---------------------------------------------------------------------------
# UX 1-4 / 1-5: the launcher-side warmup default is ON, and the two bench modes force it back off.
# The forced value carries a REASON so the single ready-side record can name it; the reason enum is
# what replaced the old "RELEASE_SPEC 8 default" text, which becomes false once the default is on.
$script:WARMUP_PRODUCT_DEFAULT = 'on'
$script:WARMUP_FORCED_REASON_BENCH =
    'repro/smoke forces warmup off (bench cache-state preservation)'
$script:WARMUP_SKIP_FORCED_BENCH = 'forced_bench'
$script:WARMUP_SKIP_USER_OFF     = 'user_off'

# LS OA-1 surface axes (wire). Three separate questions, never collapsed into one badge:
#   copy integrity     - did every selected routed slice verify byte-for-byte?
#   inventory authority- who decided WHICH tensors the routed inventory contains?
#   serving validation - has this configuration been validated for serving?
$script:AXIS_COPY_PASS            = 'PASS (every selected routed slice verified)'
$script:AXIS_COPY_PENDING         = 'pending (repack verify has not run yet)'
$script:AXIS_INVENTORY_PIN        = 'model-pin'
$script:AXIS_INVENTORY_UNPINNED   = 'model-pin(unpinned)'
# P4 2.5: the fourth inventory answer. 'unpinned' means "the bytes were never checked"; this one
# means "they were checked and they are NOT the catalog's bytes" - a strictly stronger statement,
# so it may not share the unpinned wording.
$script:AXIS_INVENTORY_MISMATCH   = 'model-pin(mismatch)'
$script:AXIS_INVENTORY_TEMPLATE   = 'arch-template'
$script:AXIS_SERVING_VALIDATED    = 'validated'
$script:AXIS_SERVING_UNVALIDATED  = 'unvalidated'
# The claim the template path is allowed to make, verbatim. It says what was copied and from where -
# it does NOT say "your file is byte-verified", which would claim an authority nobody established.
$script:TEMPLATE_COPY_SENTENCE =
    'the template-selected routed-expert inventory was copied byte-for-byte from your file'
$script:UNPINNED_NOTE =
    'no source pin recorded for this catalog profile: the header fingerprint matched but the file bytes were not checked'
$script:MISMATCH_NOTE =
    'this file has the catalog profile shape but NOT its bytes: every published number for this profile describes a different file, prefetch is off and no reference claim is made'

# Derived defaults.argv skeleton. NOT a guess: all five shipped catalog profiles carry a
# byte-identical argument list apart from '--n-cpu-moe <n_layer>' and '-c' (12288 on four,
# 4096 on qwen35-35b) - measured 26-08-02 over launcher\models.json. The model-dependent slot is
# therefore exactly one, and it is filled from the derivation. '-c' takes the CONSERVATIVE observed
# value because an unvalidated model has no measured context budget; the ctx allowlist key raises it.
$script:DERIVED_ARGV_NCPUMOE_SLOT = '<n_layer>'
$script:DERIVED_ARGV_SKELETON = @(
    '-ngl', '99', '--n-cpu-moe', $script:DERIVED_ARGV_NCPUMOE_SLOT, '-c', '4096', '-t', '8',
    '-b', '2048', '-ub', '512', '-fa', 'on', '-np', '1',
    '--host', '127.0.0.1', '--port', '8093', '--no-webui', '--no-warmup')
# Same source, same rule: the four bounds every shipped profile agrees on, and ctx capped at the
# SMALLER of the two observed maxima. budget_mb.min is not here - it is the derived structural
# minimum, which is only known after the plan.
$script:DERIVED_BOUNDS_PORT    = @{ min = 1024; max = 65535 }
$script:DERIVED_BOUNDS_CTX     = @{ min = 2048; max = 131072 }
$script:DERIVED_BOUNDS_THREADS = @{ min = 1;    max = 64 }
$script:DERIVED_BOUNDS_QD      = @{ min = 1;    max = 63 }
$script:DERIVED_BOUNDS_BUDGET_MAX = 65536

# Engine env var names (1st source: bench/moe-direct/repro/moedirect-v2-b10057.patch,
# bench/moe-direct/moe_serve.ps1:497-499).
$script:ENV_DIRECT       = 'MOE_DIRECT'
$script:ENV_DIRECT_DIR   = 'MOE_DIRECT_DIR'
$script:ENV_BUDGET_MB    = 'MOE_DIRECT_BUDGET_MB'
$script:ENV_EXPECTS_DIR  = 'MOE_DIRECT_EXPECTS_DIR'
$script:ENV_QD           = 'MOE_DIRECT_QD'
$script:ENV_PREFETCH_K   = 'MOE_DIRECT_PREFETCH_K'
$script:ENV_PREFETCH_N   = 'MOE_DIRECT_PREFETCH_N'
$script:ENV_NO_PREFETCH  = 'MOE_NO_PREFETCH'
$script:ENV_METRICS      = 'MOE_DIRECT_METRICS'

# Engine hard caps (1st source: moedirect-v2-b10057.patch:3655-3656).
$script:ENGINE_QD_MIN = 1
$script:ENGINE_QD_MAX = 63

# R1-11 deny-by-default child environment sanitation. Everything here is stripped from the child
# block before the launcher injects its own values, so an ambient value from a previous run or a
# user shell cannot change engine behaviour or bypass the bundle backend closure.
# Precedent: moe_serve.ps1:560-566 removes every MOE_* plus GGML_BACKEND_PATH.
$script:ENV_DENY_PREFIXES = @('MOE_', 'GGML_', 'LLAMA_')
$script:ENV_DENY_NAMES    = @('GGML_BACKEND_PATH', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY',
                              'GGML_CUDA_FORCE_MMQ', 'GGML_CUDA_FORCE_CUBLAS',
                              'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                              'LLAMA_ARG_MODEL', 'LLAMA_ARG_HOST', 'LLAMA_ARG_PORT',
                              'LLAMA_ARG_N_PARALLEL', 'LLAMA_ARG_CTX_SIZE', 'LLAMA_ARG_THREADS',
                              'LLAMA_ARG_N_GPU_LAYERS', 'LLAMA_ARG_NO_WARMUP', 'LLAMA_CACHE')

# ---------------------------------------------------------------------------
# LS 13-5 (WS-1) explicit child environment block - SERVER ROLE ONLY.
# Frozen OS bootstrap allowlist, 26 keys, WARMSTART_SPEC A-4c is the authority for the list. These
# keys are OS bootstrap / path / identity only and therefore take no part in the state-semantics
# projection: the engine's own bytes are bound by engine_bundle_sha256 and the bundle's DLLs are
# resolved from the application directory before PATH.
# Why an explicit block at all: with an inherited (ambient) environment a computation-changing
# variable such as NVIDIA_TF32_OVERRIDE survives into the child, which breaks the premise that
# $config.env is the complete semantic environment surface. The repacker role deliberately keeps
# the old ambient + deny-list contract (LS 13-5 - out of scope for this change).
# ---------------------------------------------------------------------------
$script:ENV_OS_BOOTSTRAP_ALLOWLIST = @(
    'SystemRoot', 'windir', 'SystemDrive', 'ComSpec', 'PATHEXT', 'PATH', 'TEMP', 'TMP',
    'USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'ProgramData', 'PUBLIC',
    'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'PROCESSOR_IDENTIFIER',
    'PROCESSOR_LEVEL', 'PROCESSOR_REVISION', 'OS', 'COMPUTERNAME', 'USERNAME',
    'USERDOMAIN', 'SESSIONNAME', 'LOGONSERVER', 'HOMEDRIVE', 'HOMEPATH')

# ---------------------------------------------------------------------------
# LS 13 (WS-1) warmstart / slot-save constants. WARMSTART_SPEC v0.5 is the value authority.
# ---------------------------------------------------------------------------
# A-1: the KV directory always derives from Get-LauncherStateDir, so the argument the server gets
# and the directory the GC walks can never drift apart.
$script:KV_DIR_NAME             = 'kv'
# A-4b name contract. canonical = the one generation that may ever be restored.
$script:KV_CANONICAL_DATA       = 'slot0.kv'
$script:KV_CANONICAL_META       = 'slot0.kv.meta.json'
# A-4b GC: the four generation patterns, retention 0 for both tmp and stale.
$script:KV_GC_PATTERNS          = @('slot0.kv.tmp.*', 'slot0.kv.stale.*',
                                    'slot0.kv.meta.json.tmp.*', 'slot0.kv.meta.json.stale.*')
# A-4b: more matches than this is a crash-loop signal - WARN diagnostic, same delete-everything action.
$script:KV_GC_WARN_MATCHES      = 16
# A-4b canonical retention: at most this many profile directories under <StateDir>\kv\.
$script:KV_PROFILE_RETENTION    = 4
$script:KV_META_SCHEMA_VERSION  = 1
$script:KV_SLOT_ID              = 0
$script:ARG_SLOT_SAVE_PATH      = '--slot-save-path'
# Cache of per-shard model hashes (A-4 "safe cache after the first computation").
# Version 2 = the entry key is the Windows file identity (64 bit volume serial + 128 bit FILE_ID_INFO
# id + size + mtime). Version 1 keyed on the path and is discarded wholesale: it was not an identity.
# LS OA-1: the cache now has TWO readers - warmstart eligibility and the M1 source pin - so it moved
# one level up, out of the kv tree, and lives directly under the state directory. It had to: the
# LS 13-2 truth table gives -Smoke / -Repro "no contact with the kv tree at all", and a pin hash
# writing its cache under kv\ would have broken that contract on every reproducibility run. A file
# left at the old kv\ location by an earlier build is simply not read (one re-hash, then the new
# location serves it).
$script:KV_SHARD_CACHE_FILE     = 'model_shards.cache.json'
$script:KV_SHARD_CACHE_VERSION  = 2
# Structural deadlines (protocol shape, NOT measured thresholds - the same kind of value as
# PLAN_TIMEOUT_S / READY_TIMEOUT_S). A-2 (3) only requires the wait to be finite: exceeding it
# abandons the save and lets the normal teardown continue.
$script:KV_SAVE_TIMEOUT_S       = 900
$script:KV_RESTORE_TIMEOUT_S    = 900
$script:KV_ERASE_TIMEOUT_S      = 120

# LS 13-6 reason enum (frozen). Field disagreements render as "<exact sidecar key>_mismatch";
# these nine are the special values that are not a field disagreement.
$script:KV_REASON_SIDECAR_MISSING   = 'sidecar_missing'
$script:KV_REASON_KV_FILE_MISSING   = 'kv_file_missing'
$script:KV_REASON_META_PARSE_FAILED = 'meta_parse_failed'
$script:KV_REASON_FILE_INTEGRITY    = 'file_integrity_broken'
$script:KV_REASON_OFF_USER          = 'off_user'
$script:KV_REASON_OFF_MODE          = 'off_mode'
$script:KV_REASON_RECOVERY_COLD     = 'recovery_cold'
$script:KV_REASON_UNAVAILABLE       = 'eligibility_unavailable'
$script:KV_REASON_RESTORE_FAILED    = 'restore_failed'
$script:KV_SPECIAL_REASONS = @(
    $script:KV_REASON_SIDECAR_MISSING, $script:KV_REASON_KV_FILE_MISSING,
    $script:KV_REASON_META_PARSE_FAILED, $script:KV_REASON_FILE_INTEGRITY,
    $script:KV_REASON_OFF_USER, $script:KV_REASON_OFF_MODE,
    $script:KV_REASON_RECOVERY_COLD, $script:KV_REASON_UNAVAILABLE,
    $script:KV_REASON_RESTORE_FAILED)

# ---------------------------------------------------------------------------
# LS 13-8 / AUTOSAVE_SPEC v0.1 - periodic crash-recovery save while serving.
# ---------------------------------------------------------------------------
# B: two alternating generations. The generation being written is the only one that can be damaged
# by a crash, which is what leaves the previous one intact. The sidecar name of any generation is
# its data name + '.meta.json' (the canonical pair follows the same rule).
$script:KV_AUTOSAVE_GEN_A       = 'slot0.auto.a.kv'
$script:KV_AUTOSAVE_GEN_B       = 'slot0.auto.b.kv'
$script:KV_AUTOSAVE_GENERATIONS = @($script:KV_AUTOSAVE_GEN_A, $script:KV_AUTOSAVE_GEN_B)
# A-4c kin: the sidecar of an autosave generation carries the same wire fields plus this origin.
# An absent origin means the normal stop save (every sidecar written before this feature existed).
$script:KV_ORIGIN_AUTOSAVE      = 'autosave'
$script:KV_ORIGIN_STOP          = 'stop'
# P1 default period, and the structural bounds of the <minutes> form of the surface.
$script:KV_AUTOSAVE_DEFAULT_MIN = 5
$script:KV_AUTOSAVE_MIN_MINUTES = 1
$script:KV_AUTOSAVE_MAX_MINUTES = 1440
# Idle probe deadline. Protocol shape, not a measured threshold: GET /slots is answered from the
# server's task queue as a high-priority task, so a slow answer means the server is not in a state
# worth autosaving into anyway - the tick is skipped.
$script:KV_SLOTS_TIMEOUT_S      = 30

# LS 13-3: three degraded diagnostic kinds. None of them creates a status enum or an exit code.
$script:KV_DIAG_SAVE_FAILED    = 'warmstart_save_failed'
$script:KV_DIAG_RESTORE_FAILED = 'warmstart_restore_failed'
$script:KV_DIAG_GC_FAILED      = 'warmstart_gc_failed'

# A-4c exclusion table (frozen). A flag is dropped together with its value, at EVERY occurrence,
# and the comparison is case sensitive: anything not listed here is a semantic key and is hashed,
# so an unknown or newly added argument can only ever cause an unnecessary cold start (safe), never
# a restore onto a changed configuration.
$script:KV_SEMANTICS_ARGV_DROP = @(
    @{ flag = '--port';            arity = 1 },   # transport
    @{ flag = '--host';            arity = 1 },   # transport (loopback already locked)
    @{ flag = '-m';                arity = 1 },   # path string; content is bound by model_shards_sha256[]
    @{ flag = '-t';                arity = 1 },   # scheduling; takes no part in KV state meaning
    @{ flag = $script:ARG_SLOT_SAVE_PATH; arity = 1 },   # this feature's own storage location
    @{ flag = '--no-webui';        arity = 0 })   # UI surface
# Same rule for env. The comparison is case insensitive here because Windows environment variable
# names are: two spellings would be ONE variable in the child, so treating them as two would let a
# per-run path (the metrics file) leak into the hash and make every run cold.
$script:KV_SEMANTICS_ENV_DROP = @(
    'MOE_DIRECT_METRICS',       # per-run timestamped path - would make every run cold
    'MOE_DIRECT_DIR',           # path; content is bound by repack_manifest_sha256
    'MOE_DIRECT_EXPECTS_DIR',   # path; content is bound by the engine seal gate
    'MOE_DIRECT_PREFETCH_K', 'MOE_DIRECT_PREFETCH_N', 'MOE_NO_PREFETCH',
    # WARMSTART_SPEC A-4c / A-7 stage 2 (measured 26-08-02). These two were hashed while the
    # byte-identity claim was unmeasured. The stage 2 run saved four arms from one token prefix -
    # baseline, budget-only (8192 -> 10240), QD-only (8 -> 7), K/N-only - and all four produced the
    # same n_tokens (1397), the same n_bytes (190670760) and the same kv_file_sha256
    # (37c88ccfcacab1507ef8e903eab855407e969125b7d22dc2e6977cf77dda8067), so neither the resident
    # budget nor the read queue depth changes the stored KV bytes. Promotion condition met -> both
    # move here. The K/N result stays an observation only: that pair already had its own exclusion.
    'MOE_DIRECT_BUDGET_MB', 'MOE_DIRECT_QD')

# Probe binding records (launcher-internal state, not wire).
# LS 12-4 gives probe.state.json to the SWEEP binding (state_version 2, a target-key map), so the
# LS 1-4 scratch record - which is provisional and no longer decides QD - keeps its own file. A
# legacy probe.state.json written by an older build parses as state_version 1 and is therefore a
# cache miss for the sweep, which is exactly the v1 migration rule LS 12-4 asks for.
$script:PROBE_STATE_FILE = 'probe.scratch.json'
$script:PROBE_STATE_VERSION = 1
$script:SWEEP_STATE_FILE = 'probe.state.json'
$script:SWEEP_STATE_VERSION = 2

# Structural (non-measured) bounds.
$script:PORT_MIN = 1024
$script:PORT_MAX = 65535

# ---------------------------------------------------------------------------
# [UNMEASURED-TODO] LS 9 item 1 / RS 10 item 3.
# Numeric thresholds below are deliberately unset. The branch structure is frozen
# (LS 1-2, LS 1-4); only the numbers are open, and they must come from measurement,
# never from a guess. While they are $null the launcher degrades honestly instead of
# fabricating a value (status screen shows [unmeasured], diagnostic log records it).
# ---------------------------------------------------------------------------
# SSD probe (MiB/s) -> default queue depth mapping table. Shape when measured:
#   @( @{ min_mibps = <num>; qd = <int> }, ... ) ordered high -> low.
$script:PROBE_QD_MAP = $null                 # TODO[unmeasured]
# Conservative default used until PROBE_QD_MAP is measured, and the RS 5 degraded value.
$script:QD_DEGRADED = 1                      # RS 5 (spec-given, not a measurement)
# Available-RAM (MiB) -> default budget mapping table. Same shape as PROBE_QD_MAP with 'budget_mb'.
$script:PROBE_BUDGET_MAP = $null             # TODO[unmeasured]
# LS 1-9 RAM verdict formula terms: budget + non-cache fixed term (dense resident + KV + server
# overhead) + safety headroom. KV scales with ctx, so it is a separate per-token term.
$script:RAM_DENSE_RESIDENT_MB = $null        # TODO[unmeasured]
$script:RAM_KV_MB_PER_1K_CTX  = $null        # TODO[unmeasured]
$script:RAM_SERVER_OVERHEAD_MB = $null       # TODO[unmeasured]
$script:RAM_HEADROOM_MB   = $null            # TODO[unmeasured]
# Disk: post-repack residual reserve shown separately (RS 3 preflight). Threshold unmeasured;
# the free-space hard stop below does not depend on it.
$script:DISK_POST_RESERVE_MB = $null         # TODO[unmeasured]
# SSD probe shape. Block size falls back to this when no manifest slot stride is available.
$script:PROBE_BLOCK_BYTES_FALLBACK = 1048576 # structural (1 MiB, sector-aligned), not a threshold
$script:PROBE_SAMPLES              = 64
$script:PROBE_SECTOR_ALIGN         = 4096
# Scratch file written on the OUTPUT volume when no repack artifact exists yet (R1-2: the probe
# must measure the volume the expert cache will be read from, not the source model's volume).
$script:PROBE_SCRATCH_BYTES        = 67108864   # 64 MiB
$script:PROBE_SCRATCH_NAME         = '.moe-probe.tmp'

# ---------------------------------------------------------------------------
# LS 12 (QD-1) startup queue-depth sweep. Every value here is STRUCTURAL (protocol shape), not a
# measured threshold: the sweep produces the numbers, it is not configured by them.
# ---------------------------------------------------------------------------
$script:SWEEP_PROBE_ALGORITHM    = 'qd-sweep-v1'
$script:SWEEP_MEASUREMENT_METHOD = 'engine-overlapped-v1'
$script:SWEEP_QD_POINTS          = @(1, 2, 4, 8)
$script:SWEEP_ORDER_FORWARD      = @(1, 2, 4, 8)
$script:SWEEP_ORDER_REVERSE      = @(8, 4, 2, 1)
# 2.5 s per point, split into two crossed 1.25 s windows (LS 12-2).
$script:SWEEP_WINDOW_MS          = 1250
$script:SWEEP_BLOCK_MIN_BYTES    = 1048576     # 1 MiB
$script:SWEEP_BLOCK_MAX_BYTES    = 16777216    # 16 MiB
# The target must hold at least this many distinct full blocks, otherwise the deterministic offset
# sequence would keep re-reading the same few blocks and the point would measure the device cache
# rather than the read path. Same kind of precondition as the scratch probe's "4 blocks" floor.
$script:SWEEP_MIN_BLOCK_SPAN     = 32
$script:SWEEP_OFFSET_COUNT       = 8192
# Buffer/offset/length alignment floor for FILE_FLAG_NO_BUFFERING. The real physical sector size is
# queried per target and used when it is larger.
$script:SWEEP_FALLBACK_ALIGN     = 4096

# Timeouts (seconds). Structural, not performance thresholds.
# R2-1: repack --plan only parses GGUF headers and prints a summary (repack_experts.py cmd_plan -
# no payload copy, no hashing), so it is inherently short. An unbounded wait here can therefore
# only mean a stuck child, which previously froze the launcher for ever. The full repack run
# deliberately has NO deadline (it legitimately takes minutes to hours) and relies on the console
# stop request instead.
$script:PLAN_TIMEOUT_S     = 120
# R3-2: the launcher itself sent the abort, so the server owes it a "cancel task, id_task = N"
# warning. The INFO/WRN logger flushes from its own worker thread, so under load that warning can
# land after the cancelled stream's natural end. This is a bounded, DIAGNOSTIC-ONLY extra wait used
# to classify why item 4 failed; it can never turn a failure into a pass, because the pass decision
# is made against the prompt budget measured from the abort instant.
$script:CANCEL_WARN_DIAG_MS = 3000
$script:READY_TIMEOUT_S    = 900
$script:GRACEFUL_STOP_S    = 60
$script:LISTENER_GONE_S    = 30
$script:HEALTH_POLL_MS     = 500

# endregion

# ============================================================================
# region 2. NATIVE INTEROP (P/Invoke)
# ============================================================================

if (-not ('MoeLauncher.LauncherExit' -as [type])) {
    Add-Type -TypeDefinition @'
namespace MoeLauncher {
    public class LauncherExit : System.Exception {
        public string Status;
        public LauncherExit(string status, string message) : base(message) { this.Status = status; }
    }
}
'@
}

if (-not ('MoeLauncher.Native' -as [type])) {
    Add-Type -Namespace 'MoeLauncher' -Name 'Native' -MemberDefinition @'
    [StructLayout(LayoutKind.Sequential)]
    public struct MEMORYSTATUSEX {
        public uint dwLength; public uint dwMemoryLoad;
        public ulong ullTotalPhys; public ulong ullAvailPhys;
        public ulong ullTotalPageFile; public ulong ullAvailPageFile;
        public ulong ullTotalVirtual; public ulong ullAvailVirtual; public ulong ullAvailExtendedVirtual;
    }
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX m);

    // INSTALLED physical memory from the SMBIOS tables, in KB. Deliberately separate from the
    // MEMORYSTATUSEX total above: that one is what the OS can address after firmware reservations
    // (31,900 MiB on a 32 GiB box), this one is what is in the slots. The budget autotune is
    // calibrated on the installed number (region 8b).
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetPhysicallyInstalledSystemMemory(out ulong TotalMemoryInKilobytes);

    // ---- kill-on-close job object (precedent: moe_serve.ps1:582-607) ----
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit; public UIntPtr Affinity; public uint PriorityClass; public uint SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS { public ulong r, w, o, rt, wt, ot; }
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed;
    }
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern IntPtr CreateJobObjectW(IntPtr sa, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, int len);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    // ---- process creation with explicit creation flags (CREATE_NEW_PROCESS_GROUP) ----
    [StructLayout(LayoutKind.Sequential)]
    public struct SECURITY_ATTRIBUTES { public int nLength; public IntPtr lpSecurityDescriptor; public int bInheritHandle; }
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct STARTUPINFO {
        public int cb; public string lpReserved; public string lpDesktop; public string lpTitle;
        public int dwX; public int dwY; public int dwXSize; public int dwYSize;
        public int dwXCountChars; public int dwYCountChars; public int dwFillAttribute; public int dwFlags;
        public short wShowWindow; public short cbReserved2; public IntPtr lpReserved2;
        public IntPtr hStdInput; public IntPtr hStdOutput; public IntPtr hStdError;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION { public IntPtr hProcess; public IntPtr hThread; public int dwProcessId; public int dwThreadId; }

    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool CreateProcessW(string lpApplicationName, System.Text.StringBuilder lpCommandLine,
        IntPtr lpProcessAttributes, IntPtr lpThreadAttributes, bool bInheritHandles, uint dwCreationFlags,
        IntPtr lpEnvironment, string lpCurrentDirectory, ref STARTUPINFO si, out PROCESS_INFORMATION pi);

    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern IntPtr CreateFileW(string name, uint access, uint share, ref SECURITY_ATTRIBUTES sa,
        uint create, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode, EntryPoint="CreateFileW")]
    public static extern IntPtr CreateFileNoSaW(string name, uint access, uint share, IntPtr sa,
        uint create, uint flags, IntPtr template);
    // WARMSTART A-4 cache key, size + mtime half. Every FILETIME is expanded into its two DWORDs so
    // the struct needs no imported time type and stays a plain sequential blob.
    // NOTE: nFileIndexHigh/Low are part of the OS layout and must stay for it to line up, but they
    // are deliberately NOT read - see QueryFileId128.
    [StructLayout(LayoutKind.Sequential)]
    public struct BY_HANDLE_FILE_INFORMATION {
        public uint dwFileAttributes;
        public uint ftCreationTimeLow;   public uint ftCreationTimeHigh;
        public uint ftLastAccessTimeLow; public uint ftLastAccessTimeHigh;
        public uint ftLastWriteTimeLow;  public uint ftLastWriteTimeHigh;
        public uint dwVolumeSerialNumber;
        public uint nFileSizeHigh;       public uint nFileSizeLow;
        public uint nNumberOfLinks;
        public uint nFileIndexHigh;      public uint nFileIndexLow;
    }
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetFileInformationByHandle(IntPtr h, out BY_HANDLE_FILE_INFORMATION info);

    // WARMSTART A-4 cache key, IDENTITY half (LS 13-7 (9)). The 64 bit index in
    // BY_HANDLE_FILE_INFORMATION is documented as NOT guaranteed unique on ReFS - and a Windows 11
    // Dev Drive IS ReFS - so it cannot key a cache that decides whether a stored model state may be
    // restored. FILE_ID_INFO (FILE_INFO_BY_HANDLE_CLASS.FileIdInfo = 18) is the 64 bit volume serial
    // plus the 128 bit file id, which is unique on every supported filesystem.
    // Read through a raw buffer rather than a marshalled struct: a 16 byte inline array inside an
    // out-parameter struct is exactly the shape whose marshalling differs between runtimes, and the
    // answer here decides a fail-close, so it is assembled byte by byte instead.
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetFileInformationByHandleEx(IntPtr h, int infoClass, IntPtr info, uint size);
    public const int FILE_ID_INFO_CLASS = 18;
    // "<16 hex volume serial>:<32 hex file id>", or null when the filesystem or the OS cannot
    // answer (pre-Windows 8 returns ERROR_INVALID_PARAMETER for this class). Null means "no
    // identity", which the caller turns into "never cache this file".
    public static string QueryFileId128(IntPtr h) {
        IntPtr buf = Marshal.AllocHGlobal(24);
        try {
            for (int i = 0; i < 24; i++) { Marshal.WriteByte(buf, i, 0); }
            if (!GetFileInformationByHandleEx(h, FILE_ID_INFO_CLASS, buf, 24)) { return null; }
            ulong vol = (ulong)Marshal.ReadInt64(buf, 0);
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            sb.Append(vol.ToString("x16"));
            sb.Append(":");
            bool anyId = false;
            for (int i = 0; i < 16; i++) {
                byte b = Marshal.ReadByte(buf, 8 + i);
                if (b != 0) { anyId = true; }
                sb.Append(b.ToString("x2"));
            }
            // An all-zero 128 bit id is not an identity; refuse it exactly like a failed call.
            if (!anyId) { return null; }
            return sb.ToString();
        } finally { Marshal.FreeHGlobal(buf); }
    }
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool ReadFile(IntPtr h, IntPtr buf, uint toRead, out uint read, IntPtr ov);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetFilePointerEx(IntPtr h, long dist, out long newPos, uint method);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint WaitForSingleObject(IntPtr h, uint ms);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetExitCodeProcess(IntPtr h, out uint code);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool TerminateProcess(IntPtr h, uint code);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GenerateConsoleCtrlEvent(uint ctrlEvent, uint processGroupId);
    // R1-3: the .partial absence decision must use GetFileAttributesW + GetLastError, exactly like
    // the C++ seal. FileInfo.Exists returns false for permission/IO errors too, which would turn a
    // "cannot prove absence" into a silent "absent".
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern uint GetFileAttributesW(string path);
    // Atomic replace. System.IO.File.Replace cannot be used from PowerShell with a null backup
    // path (PS marshals $null to "" and the API rejects it as an illegal path), and it also fails
    // when the destination does not exist yet. MoveFileEx handles both cases.
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool MoveFileExW(string existing, string newName, uint flags);
    // R2-3: real volume identity. A path root ("C:\") is not an identity - C:\mnt\ssdA and
    // C:\mnt\ssdB can be different mounted volumes, and a drive letter can be reassigned to a
    // different disk. GetVolumePathNameW gives the actual mount point of the path and
    // GetVolumeNameForVolumeMountPointW turns that into the volume GUID name.
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool GetVolumePathNameW(string fileName, System.Text.StringBuilder volumePathName, uint bufferLength);
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool GetVolumeNameForVolumeMountPointW(string volumeMountPoint, System.Text.StringBuilder volumeName, uint bufferLength);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr GetStdHandle(int nStdHandle);
    // LS 11-7 a: discard console keys that arrived while nothing was reading input.
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool FlushConsoleInputBuffer(IntPtr hConsoleInput);
'@
}

# R1-4: parent-side console control handler. The handler only records the request; the main loop
# decides what it means for the current state (pre-ready = cancelled_user, after ready = graceful
# stop). Returning true stops the default terminate-the-process behaviour.
if (-not ('MoeLauncher.CtrlHandler' -as [type])) {
    Add-Type -TypeDefinition @'
namespace MoeLauncher {
    public class CtrlHandler {
        public delegate bool Routine(uint ctrlType);
        [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool SetConsoleCtrlHandler(Routine handler, bool add);
        public static bool Requested = false;
        public static uint LastEvent = 999;
        public static bool Installed = false;
        private static Routine _kept;
        // public so the selftest can prove the CTRL_C_EVENT(0) contract directly: Windows cannot
        // deliver CTRL_C to a specific foreign process from a test harness (measured - see the
        // selftest's console-signal section), so the identical treatment of event 0 and event 1 is
        // asserted by invoking the handler, while OS delivery is proven with CTRL_BREAK_EVENT(1).
        public static bool OnCtrl(uint t) {
            // CTRL_C_EVENT(0) and CTRL_BREAK_EVENT(1) are both "the user asked to stop".
            if (t == 0 || t == 1) { LastEvent = t; Requested = true; return true; }
            return false;
        }
        public static bool Install() {
            if (Installed) { return true; }
            _kept = new Routine(OnCtrl);
            Installed = SetConsoleCtrlHandler(_kept, true);
            return Installed;
        }
    }
}
'@
}

# LS 12-2 sweep executor: measurement_method 'engine-overlapped-v1'. ONE file handle opened with
# FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, one OVERLAPPED + one event per outstanding read,
# and a completed read is immediately re-issued so the queue stays at the target depth (the engine's
# own queue-depth semantics). A thread-per-read or handle-per-read arrangement would measure a
# different thing, which is why the method name is recorded in the binding key.
if (-not ('MoeLauncher.Sweep' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace MoeLauncher {

    // One measured window for one queue depth. ElapsedTicks is in 100 ns units.
    public class SweepPoint {
        public int Qd;
        public bool Ok;
        public long Bytes;
        public long ElapsedTicks;
        public int Reads;
        public string Error;
    }

    public class Sweep {
        const uint GENERIC_READ = 0x80000000;
        const uint FILE_SHARE_READ = 0x00000001;
        const uint FILE_SHARE_WRITE = 0x00000002;
        const uint OPEN_EXISTING = 3;
        const uint FILE_FLAG_NO_BUFFERING = 0x20000000;
        const uint FILE_FLAG_OVERLAPPED = 0x40000000;
        const uint MEM_COMMIT = 0x1000;
        const uint MEM_RESERVE = 0x2000;
        const uint MEM_RELEASE = 0x8000;
        const uint PAGE_READWRITE = 0x04;
        const uint WAIT_TIMEOUT_CODE = 258;
        const int ERROR_IO_PENDING = 997;
        const int READ_WAIT_MS = 30000;
        const int DRAIN_WAIT_MS = 60000;

        [StructLayout(LayoutKind.Sequential)]
        private struct OVERLAPPED_X {
            public IntPtr Internal;
            public IntPtr InternalHigh;
            public uint Offset;
            public uint OffsetHigh;
            public IntPtr hEvent;
        }

        [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
        private static extern IntPtr CreateFileW(string name, uint access, uint share, IntPtr sa,
            uint create, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool ReadFile(IntPtr h, IntPtr buf, uint toRead, IntPtr bytesRead, IntPtr overlapped);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool GetOverlappedResult(IntPtr h, IntPtr overlapped, out uint transferred, bool wait);
        [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
        private static extern IntPtr CreateEventW(IntPtr sa, bool manualReset, bool initialState, string name);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool ResetEvent(IntPtr h);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern uint WaitForMultipleObjects(uint count, IntPtr[] handles, bool waitAll, uint ms);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool CloseHandle(IntPtr h);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern IntPtr VirtualAlloc(IntPtr addr, UIntPtr size, uint allocType, uint protect);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool VirtualFree(IntPtr addr, UIntPtr size, uint freeType);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool QueryPerformanceCounter(out long value);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool QueryPerformanceFrequency(out long value);
        [DllImport("kernel32.dll", SetLastError=true)]
        private static extern bool GetFileInformationByHandleEx(IntPtr h, int infoClass, IntPtr info, uint size);

        private static readonly IntPtr INVALID = new IntPtr(-1);

        // FILE_STORAGE_INFO (FILE_INFO_BY_HANDLE_CLASS.FileStorageInfo = 16): first two ULONGs are
        // LogicalBytesPerSector and PhysicalBytesPerSector. The answer is sanity-checked here, so a
        // wrong class id or an unsupported filesystem can only yield 0 ("unknown"), never a value
        // that would relax the caller's alignment.
        public static long QueryPhysicalSectorBytes(string path) {
            IntPtr h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, IntPtr.Zero,
                                   OPEN_EXISTING, 0, IntPtr.Zero);
            if (h == INVALID) { return 0; }
            IntPtr buf = Marshal.AllocHGlobal(64);
            try {
                for (int i = 0; i < 64; i++) { Marshal.WriteByte(buf, i, 0); }
                if (!GetFileInformationByHandleEx(h, 16, buf, 64)) { return 0; }
                long logical = (long)((uint)Marshal.ReadInt32(buf, 0));
                long physical = (long)((uint)Marshal.ReadInt32(buf, 4));
                if (!IsSaneSectorSize(logical) || !IsSaneSectorSize(physical)) { return 0; }
                if (physical < logical) { return 0; }
                return physical;
            } catch (Exception) {
                return 0;
            } finally {
                Marshal.FreeHGlobal(buf);
                CloseHandle(h);
            }
        }

        private static bool IsSaneSectorSize(long v) {
            if (v < 512 || v > 1048576) { return false; }
            return ((v & (v - 1)) == 0);
        }

        // One measured window at one queue depth. The offset sequence is supplied by the caller and
        // is identical for every point (LS 12-2 determinism).
        public static SweepPoint RunPoint(string path, int qd, long blockBytes, long[] offsets,
                                          int windowMs, long alignBytes) {
            SweepPoint r = new SweepPoint();
            r.Qd = qd; r.Ok = false; r.Bytes = 0; r.ElapsedTicks = 0; r.Reads = 0; r.Error = null;
            if (qd < 1 || qd > 64) { r.Error = "queue depth out of range"; return r; }
            if (offsets == null || offsets.Length == 0) { r.Error = "empty offset sequence"; return r; }
            if (blockBytes <= 0 || blockBytes > 268435456L) { r.Error = "block size out of range"; return r; }
            if (windowMs <= 0) { r.Error = "window duration out of range"; return r; }

            IntPtr h = INVALID;
            IntPtr bufBase = IntPtr.Zero;
            IntPtr ovBase = IntPtr.Zero;
            IntPtr[] events = new IntPtr[qd];
            bool[] pending = new bool[qd];
            int ovSize = Marshal.SizeOf(typeof(OVERLAPPED_X));
            long bytes = 0;
            long ticks = 0;
            int reads = 0;
            try {
                long freq = 0;
                if (!QueryPerformanceFrequency(out freq) || freq <= 0) {
                    r.Error = "QueryPerformanceFrequency failed"; return r;
                }
                h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, IntPtr.Zero,
                                OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, IntPtr.Zero);
                if (h == INVALID) {
                    r.Error = "CreateFile(NO_BUFFERING|OVERLAPPED) failed gle=" + Marshal.GetLastWin32Error();
                    return r;
                }
                bufBase = VirtualAlloc(IntPtr.Zero, new UIntPtr((ulong)blockBytes * (ulong)qd),
                                       MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (bufBase == IntPtr.Zero) {
                    r.Error = "VirtualAlloc failed gle=" + Marshal.GetLastWin32Error(); return r;
                }
                if (alignBytes > 0 && (bufBase.ToInt64() % alignBytes) != 0) {
                    r.Error = "read buffer is not aligned to the physical sector size"; return r;
                }
                ovBase = Marshal.AllocHGlobal(ovSize * qd);
                for (int i = 0; i < ovSize * qd; i++) { Marshal.WriteByte(ovBase, i, 0); }
                for (int i = 0; i < qd; i++) {
                    events[i] = CreateEventW(IntPtr.Zero, true, false, null);
                    if (events[i] == IntPtr.Zero) {
                        r.Error = "CreateEvent failed gle=" + Marshal.GetLastWin32Error(); return r;
                    }
                }

                string err = null;
                int idx = 0;
                // Fill the queue first. This ramp is NOT part of the measured window (LS 12-2).
                for (int i = 0; i < qd; i++) {
                    err = Issue(h, ovBase, ovSize, bufBase, blockBytes, events, pending, i,
                                offsets[idx % offsets.Length]);
                    idx++;
                    if (err != null) { break; }
                }

                if (err == null) {
                    long windowTicks = (long)((double)windowMs / 1000.0 * (double)freq);
                    if (windowTicks < 1) { windowTicks = 1; }
                    long t0 = 0;
                    long tLast = 0;
                    bool started = false;
                    while (true) {
                        int slot = -1;
                        uint got = 0;
                        err = WaitOne(h, ovBase, ovSize, events, pending, qd, out slot, out got);
                        if (err != null) { break; }
                        long now = 0;
                        QueryPerformanceCounter(out now);
                        if (!started) {
                            // The queue is full from here on: start the clock, drop this read's bytes.
                            started = true; t0 = now; tLast = now;
                        } else {
                            bytes += (long)got; reads++; tLast = now;
                            if ((now - t0) >= windowTicks) { break; }
                        }
                        err = Issue(h, ovBase, ovSize, bufBase, blockBytes, events, pending, slot,
                                    offsets[idx % offsets.Length]);
                        idx++;
                        if (err != null) { break; }
                    }
                    if (started) {
                        ticks = (long)((double)(tLast - t0) / (double)freq * 10000000.0);
                    }
                }
                r.Error = err;
                return r;
            } catch (Exception ex) {
                r.Error = "sweep point threw: " + ex.Message;
                return r;
            } finally {
                // Nothing may be freed while a read can still write into it. If the drain does not
                // complete the buffers are deliberately leaked and the point is reported failed.
                bool drained = DrainAll(events, pending, qd);
                if (h != INVALID && h != IntPtr.Zero) { CloseHandle(h); }
                if (drained) {
                    if (bufBase != IntPtr.Zero) { VirtualFree(bufBase, UIntPtr.Zero, MEM_RELEASE); }
                    if (ovBase != IntPtr.Zero) { Marshal.FreeHGlobal(ovBase); }
                } else if (r.Error == null) {
                    r.Error = "outstanding reads did not complete";
                }
                for (int i = 0; i < qd; i++) {
                    if (events[i] != IntPtr.Zero) { CloseHandle(events[i]); }
                }
                r.Bytes = bytes;
                r.Reads = reads;
                r.ElapsedTicks = ticks;
                r.Ok = (r.Error == null && drained && bytes > 0 && ticks > 0);
            }
        }

        private static string Issue(IntPtr h, IntPtr ovBase, int ovSize, IntPtr bufBase, long blockBytes,
                                    IntPtr[] events, bool[] pending, int slot, long offset) {
            IntPtr ov = new IntPtr(ovBase.ToInt64() + (long)slot * (long)ovSize);
            if (!ResetEvent(events[slot])) { return "ResetEvent failed gle=" + Marshal.GetLastWin32Error(); }
            OVERLAPPED_X o = new OVERLAPPED_X();
            o.Internal = IntPtr.Zero;
            o.InternalHigh = IntPtr.Zero;
            o.Offset = (uint)(offset & 0xFFFFFFFFL);
            o.OffsetHigh = (uint)((offset >> 32) & 0xFFFFFFFFL);
            o.hEvent = events[slot];
            Marshal.StructureToPtr(o, ov, false);
            IntPtr buf = new IntPtr(bufBase.ToInt64() + (long)slot * blockBytes);
            bool ok = ReadFile(h, buf, (uint)blockBytes, IntPtr.Zero, ov);
            if (!ok) {
                int gle = Marshal.GetLastWin32Error();
                if (gle != ERROR_IO_PENDING) { return "ReadFile failed gle=" + gle; }
            }
            pending[slot] = true;
            return null;
        }

        private static string WaitOne(IntPtr h, IntPtr ovBase, int ovSize, IntPtr[] events, bool[] pending,
                                      int qd, out int slot, out uint transferred) {
            slot = -1;
            transferred = 0;
            int n = 0;
            int[] map = new int[qd];
            for (int i = 0; i < qd; i++) { if (pending[i]) { map[n] = i; n++; } }
            if (n == 0) { return "no outstanding read to wait for"; }
            IntPtr[] use = new IntPtr[n];
            for (int i = 0; i < n; i++) { use[i] = events[map[i]]; }
            uint w = WaitForMultipleObjects((uint)n, use, false, (uint)READ_WAIT_MS);
            if (w == WAIT_TIMEOUT_CODE) { return "read did not complete within the wait limit"; }
            if (w >= (uint)n) { return "WaitForMultipleObjects failed gle=" + Marshal.GetLastWin32Error(); }
            slot = map[(int)w];
            pending[slot] = false;
            IntPtr ov = new IntPtr(ovBase.ToInt64() + (long)slot * (long)ovSize);
            uint got = 0;
            if (!GetOverlappedResult(h, ov, out got, false)) {
                return "GetOverlappedResult failed gle=" + Marshal.GetLastWin32Error();
            }
            if (got == 0) { return "read returned zero bytes"; }
            transferred = got;
            return null;
        }

        private static bool DrainAll(IntPtr[] events, bool[] pending, int qd) {
            int n = 0;
            for (int i = 0; i < qd; i++) { if (pending[i] && events[i] != IntPtr.Zero) { n++; } }
            if (n == 0) { return true; }
            IntPtr[] use = new IntPtr[n];
            int k = 0;
            for (int i = 0; i < qd; i++) {
                if (pending[i] && events[i] != IntPtr.Zero) { use[k] = events[i]; k++; }
            }
            uint w = WaitForMultipleObjects((uint)n, use, true, (uint)DRAIN_WAIT_MS);
            if (w != 0) { return false; }
            for (int i = 0; i < qd; i++) { pending[i] = false; }
            return true;
        }
    }
}
'@
}

$script:JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
$script:JOBOBJECTCLASS_EXTENDED     = 9
$script:CREATE_NEW_PROCESS_GROUP    = 0x00000200
# LS 13-5: mandatory whenever lpEnvironment is a real pointer here - the block is UTF-16.
$script:CREATE_UNICODE_ENVIRONMENT  = 0x00000400
$script:STARTF_USESTDHANDLES        = 0x00000100
$script:GENERIC_READ                = [uint32]2147483648   # 0x80000000
$script:GENERIC_WRITE               = 0x40000000
$script:FILE_SHARE_READ             = 0x00000001
$script:FILE_SHARE_WRITE            = 0x00000002
$script:FILE_SHARE_DELETE           = 0x00000004
# Metadata-only open (WARMSTART A-4 identity probe): no data access is requested at all.
$script:FILE_READ_ATTRIBUTES        = 0x00000080
$script:OPEN_EXISTING               = 3
$script:CREATE_ALWAYS               = 2
$script:FILE_FLAG_NO_BUFFERING      = 0x20000000
$script:FILE_FLAG_RANDOM_ACCESS     = 0x10000000
$script:INVALID_HANDLE              = [IntPtr]::new(-1)
$script:CTRL_BREAK_EVENT            = 1
$script:WAIT_OBJECT_0               = 0
$script:WAIT_TIMEOUT                = 258
$script:STILL_ACTIVE                = [uint32]259
$script:FILE_BEGIN                  = 0
$script:INVALID_FILE_ATTRIBUTES     = [uint32]4294967295   # 0xFFFFFFFF
$script:ERROR_FILE_NOT_FOUND        = 2
$script:ERROR_PATH_NOT_FOUND        = 3
$script:STD_INPUT_HANDLE            = -10

# endregion

# ============================================================================
# region 3. OUTPUT / DIAGNOSTIC LOG / TERMINATION (LS 5, LS 8)
# ============================================================================

$script:StatusEmitted = $false
$script:DiagPath      = $null
$script:FailureStage  = 'fail_gate_bundle'   # classification for an unexpected internal error

function Write-Line {
    param([string] $Text = '')
    try { [Console]::Out.WriteLine($Text) } catch { Write-Output $Text }
}

function Write-Diag {
    param([string] $Kind, $Data)
    $rec = [ordered]@{ ts = (Get-Date).ToUniversalTime().ToString('o'); kind = $Kind }
    if ($null -ne $Data) { $rec['data'] = $Data }
    if (-not $script:DiagPath) { return }
    try {
        $line = ($rec | ConvertTo-Json -Compress -Depth 12)
        [System.IO.File]::AppendAllText($script:DiagPath, $line + [Environment]::NewLine,
            (New-Object System.Text.UTF8Encoding($false)))
    } catch { }
}

function Get-LauncherStateDir {
    $base = $env:LOCALAPPDATA
    if (-not $base) { $base = [System.IO.Path]::GetTempPath() }
    return (Join-Path $base 'MoE-Direct')
}

function Initialize-DiagLog {
    $dir = Join-Path (Get-LauncherStateDir) 'logs'
    try {
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
        $script:DiagPath = Join-Path $dir ("launcher_{0}_{1}.jsonl" -f $stamp, $PID)
        Write-Diag -Kind 'LAUNCH' -Data @{ pid = $PID; argv = @($MyInvocation.Line); ps = $PSVersionTable.PSVersion.ToString() }
    } catch { $script:DiagPath = $null }
}

# The one machine-readable wire line. Written to stderr, exactly once, as the final line.
function Write-StatusLine {
    param([string] $Status)
    if ($script:StatusEmitted) { return }
    $script:StatusEmitted = $true
    try { [Console]::Error.WriteLine($script:STATUS_LINE_PREFIX + $Status) }
    catch { [Console]::Out.WriteLine($script:STATUS_LINE_PREFIX + $Status) }
}

# LS 11-8: the enum is for machines; this is the same verdict for the person watching the console.
# One entry per fail_* status, each restating the meaning already fixed in LAUNCHER_SPEC 5 - no new
# failure meaning is created here. ok / ok_smoke / cancelled_user are omitted on purpose (they are
# self-evident and a hint there would only be noise).
$script:STATUS_HINT = [ordered]@{
    'fail_model_path'       = 'the model path could not be used: file missing, unsupported GGUF, or an incomplete/ambiguous shard set'
    'fail_resource'         = 'preflight stopped the run: not enough RAM or disk space for this configuration'
    'fail_instance_lock'    = 'another launcher instance holds the single-instance mutex or a profile/output/port lock'
    'fail_partial_cleanup'  = 'the leftover repack outputs could not be deleted or confirmed absent'
    'fail_repack'           = 'the repacker exited abnormally or produced no verify report'
    'fail_custom_args'      = 'a custom value failed the type/bounds check in non-interactive mode'
    'fail_gate_bundle'      = 'bundle integrity failed: its manifest, schema, or file set did not match the sealed bundle'
    'fail_gate_catalog'     = 'models.json failed the catalog schema, the prefetch axis structure or the expect digest check'
    'fail_gate_verify'      = 'the 7-item repack gate rejected the verify report or its manifest binding'
    'fail_gate_engine_seal' = 'the engine refused to start and printed its policy gate reject line'
    'fail_server_start'     = 'the server never reached ready: spawn, port, listener PID, health, or an early exit'
    'fail_runtime_exit'     = 'the server exited unexpectedly after it had reached ready'
    'fail_teardown'         = 'shutdown did not complete cleanly: signal, grace period, or a surviving child/listener'
    'fail_smoke'            = 'a smoke assertion failed while the shutdown itself completed'
}

# LS 11-8: printed on STDOUT immediately before the wire line. stderr gains nothing - the status
# line stays the single, final stderr line every existing consumer parses.
function Write-StatusHint {
    param([string] $Status)
    if (-not $script:STATUS_HINT.Contains($Status)) { return }
    Write-Line ('what happened : ' + [string]$script:STATUS_HINT[$Status])
    Write-Line ('see           : README.md > Troubleshooting > ' + $Status)
}

function Get-StatusExitCode {
    param([string] $Status)
    if ($script:STATUS_EXIT.Contains($Status)) { return [int]$script:STATUS_EXIT[$Status] }
    # Unreachable by construction; classify as the most conservative failure rather than
    # emitting an out-of-enum status.
    return 5
}

function Set-FailureStage {
    param([string] $Status)
    $script:FailureStage = $Status
}

# ---- R1-4 Ctrl+C / Ctrl+Break -------------------------------------------------------------
$script:ConsoleHandlerInstalled = $false

function Install-CtrlHandler {
    try { $script:ConsoleHandlerInstalled = [MoeLauncher.CtrlHandler]::Install() }
    catch { $script:ConsoleHandlerInstalled = $false }
    Write-Diag -Kind 'CTRL_HANDLER' -Data @{ installed = $script:ConsoleHandlerInstalled }
    return $script:ConsoleHandlerInstalled
}

function Test-CancelRequested {
    try { return [bool][MoeLauncher.CtrlHandler]::Requested } catch { return $false }
}

function Clear-CancelRequest {
    try { [MoeLauncher.CtrlHandler]::Requested = $false } catch { }
}

# Called from every pre-ready wait/prompt point: before ready, a console stop request is a plain
# user cancellation (LS 1-8 a).
function Assert-NotCancelledPreReady {
    if (Test-CancelRequested) {
        Stop-Launcher 'cancelled_user' 'console stop request received before ready'
    }
}

# The launcher can only signal an owned process group when it actually owns a console.
function Test-ConsoleAvailable {
    try { return ([MoeLauncher.Native]::GetStdHandle($script:STD_INPUT_HANDLE) -ne [IntPtr]::Zero) }
    catch { return $false }
}

# LS 11-7 a: keys pressed while the launcher was NOT reading input (identify, SSD probe, repack,
# verify - minutes of silence) stay in the console input queue and drive the NEXT read. Measured
# consequences: a Ctrl+C pressed during the repack is consumed by the first menu ReadKey and stops
# the pipeline, and a stray Enter answers the three-choice menu with "start" nobody confirmed. Both
# are removed by flushing the queue at the moment a prompt is armed.
# Interactive consoles only: with stdin redirected (pipe, CI, the selftest harness) this is a no-op
# and returns $false, so every non-interactive byte and every existing regression stays as it was.
function Clear-ConsoleInputQueue {
    try { if ([Console]::IsInputRedirected) { return $false } } catch { return $false }
    try {
        $h = [MoeLauncher.Native]::GetStdHandle($script:STD_INPUT_HANDLE)
        if ($h -eq [IntPtr]::Zero -or $h -eq $script:INVALID_HANDLE) { return $false }
        return [bool][MoeLauncher.Native]::FlushConsoleInputBuffer($h)
    } catch { return $false }
}

# ---- V-2: the interactive serving loop's command channel ------------------------------------
# Measured 26-08-02 (V-2, real console): [Console]::In.Peek() BLOCKS on a real console stdin until
# the user presses Enter. The interactive serving loop polled its command channel with that Peek,
# so between keystrokes the whole loop stood still - the autosave tick never fired (11 minutes with
# zero autosave diagnostics), the UI-9 prefill echo printed nothing, and, worst of all, the
# Test-ChildExited death check never ran, so a server that died was not noticed. "stop" + Enter was
# answered instantly, which is the tell: that keystroke is what released the blocked Peek.
# Why no test caught it: every harness child runs with stdin REDIRECTED, and on a redirected stream
# Peek returns immediately (a byte, or -1 at EOF). The blocking form only exists on a real console.
# The gate below therefore splits the two worlds:
#   redirected stdin (pipe, file, CI, the selftest harness) -> 'peek', the pre-existing path, kept
#                                                              byte for byte so no existing case moves
#   real console -> KeyAvailable is asked FIRST, and only a key already waiting authorises a read
#                   ('read'); with nothing waiting the answer is 'skip' and the loop keeps ticking
# Accepted cost of 'read': the ReadLine that follows still blocks until Enter, so monitoring pauses
# while a command is being typed. That is the deliberate trade against putting a per-key state
# machine on the serving path; the pre-fix behaviour was that same pause with no keystroke needed.
# Pure, so the branch table is unit-testable. $KeyAvailable is $true/$false, or $null when the
# console could not be probed at all - which falls back to the pre-existing 'peek' path.
function Get-ServeInputGate {
    param([bool] $Redirected, $KeyAvailable)
    if ($Redirected) { return 'peek' }
    if ($KeyAvailable -isnot [bool]) { return 'peek' }
    if ($KeyAvailable) { return 'read' }
    return 'skip'
}

# The live probe for the two facts above. Any [Console] failure (no console handle, a host without a
# real input buffer, ISE) is answered with the pre-existing path rather than a new throw.
function Get-ServeInputGateLive {
    $redirected = $true
    try { $redirected = [bool][Console]::IsInputRedirected } catch { return 'peek' }
    if ($redirected) { return 'peek' }
    try { return (Get-ServeInputGate -Redirected $false -KeyAvailable ([bool][Console]::KeyAvailable)) }
    catch { return 'peek' }
}

# One poll of the serving-loop command channel, non-blocking unless a key is already waiting.
# Returns the trimmed lower-case command, $null when there was nothing to read, and fault=$true when
# the read itself failed - the caller then backs off exactly as it did before this fix.
# Known residue, deliberately not handled: a multi-line paste can leave a second line inside the
# [Console]::In reader where KeyAvailable cannot see it. Harmless here, because the only commands
# are stop/q/quit and the first of them already leaves the loop.
function Read-ServeCommandLine {
    $gate = Get-ServeInputGateLive
    if ($gate -eq 'skip') { return @{ line = $null; fault = $false } }
    try {
        $raw = $null
        if ($gate -eq 'read') { $raw = [Console]::In.ReadLine() }
        elseif ([Console]::In.Peek() -ge 0) { $raw = [Console]::In.ReadLine() }
        if ($null -eq $raw) { return @{ line = $null; fault = $false } }
        return @{ line = ([string]$raw).Trim().ToLowerInvariant(); fault = $false }
    } catch { return @{ line = $null; fault = $true } }
}

function Stop-Launcher {
    param([string] $Status, [string] $Reason)
    Write-Diag -Kind 'STOP' -Data @{ status = $Status; reason = $Reason }
    throw (New-Object -TypeName 'MoeLauncher.LauncherExit' -ArgumentList $Status, $Reason)
}

# endregion

# ============================================================================
# region 4. STRICT FILE / JSON READERS (LS 3 strict parse)
# ============================================================================

function Read-FileBytesStrict {
    param([string] $Path)
    try {
        $bytes = [System.IO.File]::ReadAllBytes($Path)
        return @{ ok = $true; bytes = $bytes }
    } catch {
        return @{ ok = $false; reason = ("read failed: " + $_.Exception.Message) }
    }
}

function ConvertFrom-Utf8Strict {
    param([byte[]] $Bytes)
    try {
        $enc = New-Object System.Text.UTF8Encoding($false, $true)
        $off = 0
        if ($Bytes.Length -ge 3 -and $Bytes[0] -eq 0xEF -and $Bytes[1] -eq 0xBB -and $Bytes[2] -eq 0xBF) { $off = 3 }
        return @{ ok = $true; text = $enc.GetString($Bytes, $off, $Bytes.Length - $off) }
    } catch {
        return @{ ok = $false; reason = ("utf-8 decode failed: " + $_.Exception.Message) }
    }
}

# Never use "Get-Content -Raw | ConvertFrom-Json": encoding detection and ETS wrapping both
# weaken the strict-parse requirement. Bytes -> explicit UTF-8 -> ConvertFrom-Json.
function Read-JsonFileStrict {
    param([string] $Path)
    $b = Read-FileBytesStrict -Path $Path
    if (-not $b.ok) { return @{ ok = $false; reason = $b.reason } }
    $t = ConvertFrom-Utf8Strict -Bytes $b.bytes
    if (-not $t.ok) { return @{ ok = $false; reason = $t.reason } }
    return (ConvertFrom-JsonStrict -Text $t.text)
}

function ConvertFrom-JsonStrict {
    param([string] $Text)
    if ($null -eq $Text -or $Text.Trim().Length -eq 0) { return @{ ok = $false; reason = 'empty json document' } }
    try {
        $v = ConvertFrom-Json -InputObject $Text -ErrorAction Stop
        if ($null -eq $v) { return @{ ok = $false; reason = 'json parsed to null' } }
        return @{ ok = $true; value = $v }
    } catch {
        return @{ ok = $false; reason = ("json parse failed: " + $_.Exception.Message) }
    }
}

function Test-JsonHas {
    param($Obj, [string] $Name)
    if ($null -eq $Obj) { return $false }
    $props = $Obj.PSObject.Properties
    if ($null -eq $props) { return $false }
    return ($null -ne $props[$Name])
}

# Raw property read. The leading comma is required: a bare "return @()" is unrolled by PowerShell
# into "no output" ($null), which would make an empty JSON array indistinguishable from a missing
# key - exactly the distinction gate 5 has to make.
function Get-JsonValue {
    param($Obj, [string] $Name)
    if (-not (Test-JsonHas -Obj $Obj -Name $Name)) { return $null }
    return , ($Obj.PSObject.Properties[$Name].Value)
}

# Iteration/count accessor. Get-JsonValue deliberately returns the array as ONE pipeline object so
# that "[] present" stays distinguishable from "key missing"; that also means @(Get-JsonValue ...)
# would wrap it and report Count 1. Use this whenever the value is meant to be walked or counted.
function Get-JsonArray {
    param($Obj, [string] $Name)
    $v = Get-JsonValue -Obj $Obj -Name $Name
    if ($null -eq $v) { return @() }
    if ($v -is [System.Array]) { return $v }
    return @($v)
}

function Get-JsonKeys {
    param($Obj)
    if ($null -eq $Obj) { return , @() }
    return , @($Obj.PSObject.Properties | ForEach-Object { $_.Name })
}

function Test-JsonBooleanTrue {
    param($Value)
    if ($Value -is [bool]) { return [bool]$Value }
    return $false
}

function Test-JsonBoolean {
    param($Value)
    return ($Value -is [bool])
}

function Test-JsonNonNegativeInteger {
    param($Value)
    if ($Value -is [bool]) { return $false }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [int16] -or $Value -is [byte] -or $Value -is [uint32] -or $Value -is [uint64]) {
        return ([long]$Value -ge 0)
    }
    return $false
}

function Test-JsonEmptyArray {
    param($Value)
    if ($Value -is [System.Array]) { return (@($Value).Count -eq 0) }
    return $false
}

function Test-JsonArray {
    param($Value)
    return ($Value -is [System.Array])
}

function Test-JsonNonEmptyString {
    param($Value)
    if ($Value -is [string]) { return ($Value.Length -gt 0) }
    return $false
}

function Test-Sha256Hex {
    param($Value)
    if (-not ($Value -is [string])) { return $false }
    if ($Value.Length -ne 64) { return $false }
    return ($Value -match '^[0-9a-fA-F]{64}$')
}

# Atomic publish of a temp file over its final name (creates or replaces).
function Move-FileAtomic {
    param([string] $TempPath, [string] $FinalPath)
    $flags = [uint32](0x1 -bor 0x8)   # MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    if (-not [MoeLauncher.Native]::MoveFileExW($TempPath, $FinalPath, $flags)) {
        throw ('atomic replace failed (GetLastError=' + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() + ')')
    }
}

function Get-FileSha256Lower {
    param([string] $Path)
    try {
        $h = Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop
        return @{ ok = $true; sha = $h.Hash.ToLowerInvariant() }
    } catch {
        return @{ ok = $false; reason = ("hash failed: " + $_.Exception.Message) }
    }
}

# WARMSTART A-4 cache identity. A (path, length, mtime) triple is NOT an identity: swapping a
# different GGUF into the same path while preserving its length and timestamp would reuse the stored
# hash and let a foreign model's stored state pass the binding check.
# LS 13-7 (9): the identity is the 64 bit volume serial + the 128 bit FILE_ID_INFO id, taken on ONE
# handle together with the size and mtime. The 64 bit index of BY_HANDLE_FILE_INFORMATION is not
# used at all: Microsoft documents it as not guaranteed unique on ReFS, which is a real shipping
# filesystem here (a Windows 11 Dev Drive is ReFS), so it cannot decide a restore. A filesystem or
# an OS that cannot answer gets NO cache participation - neither a hit nor a stored entry - and the
# file is simply re-hashed on every start.
function Get-FileIdentity {
    param([string] $Path)
    $h = $script:INVALID_HANDLE
    try {
        # FILE_READ_ATTRIBUTES only, sharing everything including delete: this probe must never
        # disturb a file another process is reading or replacing.
        $h = [MoeLauncher.Native]::CreateFileNoSaW($Path, [uint32]$script:FILE_READ_ATTRIBUTES,
                 [uint32]($script:FILE_SHARE_READ -bor $script:FILE_SHARE_WRITE -bor $script:FILE_SHARE_DELETE),
                 [IntPtr]::Zero, [uint32]$script:OPEN_EXISTING, [uint32]0, [IntPtr]::Zero)
        if ($h -eq $script:INVALID_HANDLE -or $h -eq [IntPtr]::Zero) {
            return @{ ok = $false; reason = ('open for identity failed (GetLastError=' +
                          [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() + ')') }
        }
        $ident = [MoeLauncher.Native]::QueryFileId128($h)
        if ([string]::IsNullOrEmpty($ident)) {
            return @{ ok = $false; reason = ('FILE_ID_INFO unavailable (GetLastError=' +
                          [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() +
                          '); this filesystem supplies no 128 bit file id') }
        }
        $info = New-Object 'MoeLauncher.Native+BY_HANDLE_FILE_INFORMATION'
        if (-not [MoeLauncher.Native]::GetFileInformationByHandle($h, [ref]$info)) {
            return @{ ok = $false; reason = ('GetFileInformationByHandle failed (GetLastError=' +
                          [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() + ')') }
        }
        $size  = ([long]([uint32]$info.nFileSizeHigh)  -shl 32) -bor [long]([uint32]$info.nFileSizeLow)
        $mtime = ([long]([uint32]$info.ftLastWriteTimeHigh) -shl 32) -bor [long]([uint32]$info.ftLastWriteTimeLow)
        $parts = ([string]$ident).Split(':')
        return @{ ok = $true; identity = [string]$ident; volume = [string]$parts[0]; file_id = [string]$parts[1]
                  size = $size; mtime = $mtime }
    } catch {
        return @{ ok = $false; reason = ('file identity query failed: ' + $_.Exception.Message) }
    } finally {
        if ($h -ne $script:INVALID_HANDLE -and $h -ne [IntPtr]::Zero) { [void][MoeLauncher.Native]::CloseHandle($h) }
    }
}

# endregion

# ============================================================================
# region 5. BUNDLE INTEGRITY (LS 2 first action, RS 6-2 item 2)
# ============================================================================

function Resolve-BundleRoot {
    if ($BundleRoot) {
        if (-not (Test-Path -LiteralPath $BundleRoot -PathType Container)) {
            Stop-Launcher 'fail_gate_bundle' ("bundle root not found: " + $BundleRoot)
        }
        return (Resolve-Path -LiteralPath $BundleRoot).ProviderPath
    }
    return (Split-Path -Parent $PSCommandPath)
}

function Assert-BundleIntegrity {
    param([string] $Root)
    $manifestPath = Join-Path $Root $script:BUNDLE_MANIFEST_NAME
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        Stop-Launcher 'fail_gate_bundle' ("bundle SHA manifest missing: " + $manifestPath)
    }
    $r = Read-JsonFileStrict -Path $manifestPath
    if (-not $r.ok) { Stop-Launcher 'fail_gate_bundle' ("bundle manifest unreadable - " + $r.reason) }
    $m = $r.value

    $ver = Get-JsonValue -Obj $m -Name 'bundle_manifest_version'
    if (-not (Test-JsonNonNegativeInteger $ver) -or ([long]$ver -ne $script:BUNDLE_MANIFEST_VERSION)) {
        Stop-Launcher 'fail_gate_bundle' 'bundle_manifest_version missing or not an exact match'
    }
    $files = Get-JsonValue -Obj $m -Name 'files'
    if (-not (Test-JsonArray $files) -or (@($files).Count -eq 0)) {
        Stop-Launcher 'fail_gate_bundle' 'bundle manifest files[] missing or empty'
    }

    $listed = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($e in @($files)) {
        $rel = Get-JsonValue -Obj $e -Name 'path'
        $sha = Get-JsonValue -Obj $e -Name 'sha256'
        if (-not (Test-JsonNonEmptyString $rel)) { Stop-Launcher 'fail_gate_bundle' 'bundle manifest entry without path' }
        if (-not (Test-Sha256Hex $sha))          { Stop-Launcher 'fail_gate_bundle' ("bundle manifest entry sha256 invalid: " + $rel) }
        if ($rel.Contains('..'))                 { Stop-Launcher 'fail_gate_bundle' ("bundle manifest path escapes root: " + $rel) }
        $abs = Join-Path $Root $rel
        if (-not (Test-Path -LiteralPath $abs -PathType Leaf)) {
            Stop-Launcher 'fail_gate_bundle' ("bundle file listed but missing: " + $rel)
        }
        $h = Get-FileSha256Lower -Path $abs
        if (-not $h.ok) { Stop-Launcher 'fail_gate_bundle' ("bundle file hash failed: " + $rel + " - " + $h.reason) }
        if ($h.sha -ne $sha.ToLowerInvariant()) {
            Stop-Launcher 'fail_gate_bundle' ("bundle hash mismatch: " + $rel)
        }
        [void]$listed.Add(($rel -replace '/', '\'))
    }

    # Both directions: an unlisted payload file inside the bundle is also a manifest mismatch.
    $ignore = @($script:BUNDLE_MANIFEST_NAME)
    $extra = @()
    foreach ($f in @(Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue)) {
        $rel = $f.FullName.Substring($Root.Length).TrimStart('\')
        if ($ignore -contains $rel) { continue }
        if (-not $listed.Contains($rel)) { $extra += $rel }
    }
    if ($extra.Count -gt 0) {
        Stop-Launcher 'fail_gate_bundle' ("unlisted files inside bundle: " + ($extra -join ', '))
    }

    Write-Diag -Kind 'BUNDLE_OK' -Data @{ root = $Root; files = @($files).Count }
    Write-Line ("[bundle] SHA manifest verified: {0} file(s)" -f @($files).Count)
}

# endregion

# ============================================================================
# region 6. CATALOG (models.json) - strict parse, deny-by-default (LS 1-2, RS 4)
# ============================================================================

$script:CATALOG_TOP_KEYS = @('catalog_schema_version', 'source_tag', 'runtime', 'profiles')
$script:CATALOG_RUNTIME_KEYS = @('server_exe', 'repacker_exe', 'repacker_argv', 'expects_dir', 'webui')
# P4 1: the one-axis 'prefetch_state' is gone and the two STORED axes of PI 3 take its place. Both
# are mandatory, so a catalog that still carries the old field (or carries both) fails the
# deny-by-default key check - which is exactly the "old launcher + new catalog / new launcher + old
# catalog" stop the schema bump exists to produce.
$script:CATALOG_PROFILE_KEYS = @(
    'profile_id', 'display_name', 'hf_repo', 'hf_revision', 'routed_scope',
    'expect_file', 'expect_sha256', 'identify', 'min_budget_mb',
    'prefetch_evidence', 'prefetch_activation',
    'prefetch', 'gates', 'reference_measurements', 'allowlist_bounds', 'defaults')
# LS OA-1 (M1). Part of the profile schema - deny-by-default still rejects any key outside the two
# lists - but ABSENT is a legal state that means exactly the same thing as an empty array: no source
# pin was recorded for this profile. It is optional rather than mandatory because the digests can
# only be collected by hashing the reference model, so a profile whose model nobody on this machine
# holds could never satisfy a mandatory key. As of 26-08-07 the shipped catalog is PARTIALLY pinned:
# qwen35-122b-nonextn carries its two shard digests (hashed from the local files that produced the
# official G run), and the other five profiles remain unpinned. Unpinned is never silently upgraded -
# it is surfaced as model-pin(unpinned) / unvalidated and it self-heals the moment the digests land.
# P4 1 closure rule: prefetch_promotion_hold is optional and ABSENT means false. It is the 397B
# row's Phase 4 lock and nothing else reads it, so a catalog that omits it everywhere is complete.
$script:CATALOG_PROFILE_OPTIONAL_KEYS = @('source_shards_sha256', 'prefetch_promotion_hold')
$script:CATALOG_IDENTIFY_KEYS = @('arch', 'n_layer', 'n_expert', 'n_expert_used')
$script:CATALOG_BOUND_KEYS = @('port', 'ctx', 'threads', 'budget_mb', 'qd')
# LS 1-9: every condition column must be present, otherwise the number is hidden as [unmeasured].
$script:MEASUREMENT_COLUMNS = @('model', 'tier', 'machine_storage', 'budget_qd_prefetch', 'workload_window', 'observed_tok_s')

function Deny-UnknownKeys {
    param($Obj, [string[]] $Allowed, [string] $Where, [string[]] $Optional = @())
    foreach ($k in (Get-JsonKeys -Obj $Obj)) {
        if ($Allowed -notcontains $k -and $Optional -notcontains $k) {
            Stop-Launcher 'fail_gate_catalog' ("unknown key '" + $k + "' in " + $Where + " (deny-by-default)")
        }
    }
    foreach ($k in $Allowed) {
        if (-not (Test-JsonHas -Obj $Obj -Name $k)) {
            Stop-Launcher 'fail_gate_catalog' ("required key '" + $k + "' missing in " + $Where)
        }
    }
}

function Read-Catalog {
    param([string] $Root)
    $path = Join-Path $Root $script:CATALOG_FILE_NAME
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Stop-Launcher 'fail_gate_catalog' ("catalog missing: " + $path)
    }
    $r = Read-JsonFileStrict -Path $path
    if (-not $r.ok) { Stop-Launcher 'fail_gate_catalog' ("catalog unreadable - " + $r.reason) }
    $c = $r.value

    Deny-UnknownKeys -Obj $c -Allowed $script:CATALOG_TOP_KEYS -Where 'models.json'
    $sv = Get-JsonValue -Obj $c -Name 'catalog_schema_version'
    if (-not (Test-JsonNonNegativeInteger $sv) -or ([long]$sv -ne $script:CATALOG_SCHEMA_VERSION)) {
        Stop-Launcher 'fail_gate_catalog' 'catalog_schema_version is not an exact match'
    }
    if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $c -Name 'source_tag'))) {
        Stop-Launcher 'fail_gate_catalog' 'source_tag missing'
    }

    $rt = Get-JsonValue -Obj $c -Name 'runtime'
    Deny-UnknownKeys -Obj $rt -Allowed $script:CATALOG_RUNTIME_KEYS -Where 'runtime'
    foreach ($k in @('server_exe', 'repacker_exe', 'expects_dir')) {
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $rt -Name $k))) {
            Stop-Launcher 'fail_gate_catalog' ("runtime." + $k + " must be a non-empty string")
        }
    }
    if (-not (Test-JsonArray (Get-JsonValue -Obj $rt -Name 'repacker_argv'))) {
        Stop-Launcher 'fail_gate_catalog' 'runtime.repacker_argv must be an array'
    }
    if (-not (Test-JsonBoolean (Get-JsonValue -Obj $rt -Name 'webui'))) {
        Stop-Launcher 'fail_gate_catalog' 'runtime.webui must be a JSON boolean'
    }

    $profiles = Get-JsonValue -Obj $c -Name 'profiles'
    if (-not (Test-JsonArray $profiles) -or (@($profiles).Count -eq 0)) {
        Stop-Launcher 'fail_gate_catalog' 'profiles[] missing or empty'
    }
    $seen = @{}
    # LS OA-1 (M1) - Codex r1 F1: the ordered digest vector is a whole-catalog identity, so its
    # uniqueness has to be decided over ALL profiles, once, here. The check inside
    # Resolve-ProfileSelection only ever sees the profiles that survived the structural prefilter,
    # so two profiles pinned to the same bytes but differing in a header field would slip past it and
    # only collide on some future model. Two profiles claiming the same source bytes is a catalog
    # defect either way: exactly one of them can be right, and nothing here can say which.
    $pinSeen = @{}
    foreach ($p in @($profiles)) {
        Test-CatalogProfile -Profile $p -Root $Root -Runtime $rt
        $pid0 = [string](Get-JsonValue -Obj $p -Name 'profile_id')
        if ($seen.ContainsKey($pid0)) { Stop-Launcher 'fail_gate_catalog' ("duplicate profile_id: " + $pid0) }
        $seen[$pid0] = $true
        $pin = Get-ProfilePinShas -Profile $p
        if (@($pin).Count -gt 0) {
            $vec = (@($pin) -join '|')
            if ($pinSeen.ContainsKey($vec)) {
                Stop-Launcher 'fail_gate_catalog' ("profiles '" + $pinSeen[$vec] + "' and '" + $pid0 +
                    "' are pinned to the same source digests")
            }
            $pinSeen[$vec] = $pid0
        }
    }

    Write-Diag -Kind 'CATALOG_OK' -Data @{ source_tag = [string](Get-JsonValue -Obj $c -Name 'source_tag'); profiles = @($profiles).Count }
    return $c
}

function Test-CatalogProfile {
    param($Profile, [string] $Root, $Runtime)
    Deny-UnknownKeys -Obj $Profile -Allowed $script:CATALOG_PROFILE_KEYS -Where 'profiles[]' `
        -Optional $script:CATALOG_PROFILE_OPTIONAL_KEYS
    $pid0 = Get-JsonValue -Obj $Profile -Name 'profile_id'
    if (-not (Test-JsonNonEmptyString $pid0)) { Stop-Launcher 'fail_gate_catalog' 'profile_id must be a non-empty string' }
    foreach ($k in @('display_name', 'hf_repo', 'hf_revision', 'expect_file')) {
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $Profile -Name $k))) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": " + $k + " must be a non-empty string")
        }
    }
    $scope = Get-JsonValue -Obj $Profile -Name 'routed_scope'
    if (@('all', 'execution') -notcontains $scope) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": routed_scope must be 'all' or 'execution'")
    }
    if (-not (Test-Sha256Hex (Get-JsonValue -Obj $Profile -Name 'expect_sha256'))) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": expect_sha256 missing or not 64 hex (expect digest absent)")
    }

    $id = Get-JsonValue -Obj $Profile -Name 'identify'
    Deny-UnknownKeys -Obj $id -Allowed $script:CATALOG_IDENTIFY_KEYS -Where ($pid0 + '.identify')
    if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $id -Name 'arch'))) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": identify.arch must be a non-empty string")
    }
    foreach ($k in @('n_layer', 'n_expert', 'n_expert_used')) {
        if (-not (Test-JsonNonNegativeInteger (Get-JsonValue -Obj $id -Name $k))) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": identify." + $k + " must be a non-negative integer")
        }
    }
    if (-not (Test-JsonNonNegativeInteger (Get-JsonValue -Obj $Profile -Name 'min_budget_mb'))) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": min_budget_mb must be a non-negative integer")
    }

    # LS OA-1 (M1). When the key is present it must be a well formed array of lowercase-comparable
    # 64 hex digests - one per source shard, in shard order. A malformed pin is a catalog defect, not
    # an "unpinned" profile: silently reading a broken pin as "no pin" would turn a typo into a
    # silent downgrade of exactly the check M1 exists to add. The ARITY check lives further down,
    # where the expect file has been hashed and can be parsed for its source count.
    if (Test-JsonHas -Obj $Profile -Name 'source_shards_sha256') {
        $ss = Get-JsonValue -Obj $Profile -Name 'source_shards_sha256'
        if (-not (Test-JsonArray $ss)) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": source_shards_sha256 must be a JSON array (absent or [] = unpinned)")
        }
        foreach ($h in @($ss)) {
            if (-not (Test-Sha256Hex $h)) {
                Stop-Launcher 'fail_gate_catalog' ($pid0 + ": source_shards_sha256 entries must all be 64 hex digests")
            }
        }
    }

    # -----------------------------------------------------------------------------------------
    # P4 1-b, the STRUCTURAL half of the two-layer disposition. Only the errors that make the row
    # unparseable stop the launcher here:
    #   - a missing axis key                (Deny-UnknownKeys above, required-key branch)
    #   - the retired one-axis field, alone or beside the new ones (unknown key, same branch)
    #   - an axis whose value is not a string / a hold that is not a boolean  (this block)
    # Everything the parser CAN read but that says something impossible - an unknown enum member,
    # a stored runtime activation, a missing or stray K/N tuple, an evidence value that does not
    # support the activation - is a SEMANTIC error. Those are decided in Get-PrefetchCatalogAxes
    # and disposed of per row (boot continues, that row's prefetch is OFF with a reason), because a
    # single mis-authored row must not deny the user the five rows that are fine.
    # -----------------------------------------------------------------------------------------
    foreach ($k in @('prefetch_evidence', 'prefetch_activation')) {
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $Profile -Name $k))) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": " + $k + " must be a non-empty string")
        }
    }
    if (Test-JsonHas -Obj $Profile -Name 'prefetch_promotion_hold') {
        if (-not (Test-JsonBoolean (Get-JsonValue -Obj $Profile -Name 'prefetch_promotion_hold'))) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": prefetch_promotion_hold must be a JSON boolean (absent = false)")
        }
    }
    $pf = Get-JsonValue -Obj $Profile -Name 'prefetch'
    if ($null -ne $pf) {
        # A tuple that exists at all must be shaped like a tuple. WHETHER it may exist for this
        # row's activation is the semantic question, decided per row further down.
        Deny-UnknownKeys -Obj $pf -Allowed @('k', 'n') -Where ($pid0 + '.prefetch')
    }

    $g = Get-JsonValue -Obj $Profile -Name 'gates'
    Deny-UnknownKeys -Obj $g -Allowed @('format_validated', 'performance_validated') -Where ($pid0 + '.gates')
    foreach ($k in @('format_validated', 'performance_validated')) {
        if (-not (Test-JsonBoolean (Get-JsonValue -Obj $g -Name $k))) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": gates." + $k + " must be a JSON boolean")
        }
    }
    if (-not (Test-JsonArray (Get-JsonValue -Obj $Profile -Name 'reference_measurements'))) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": reference_measurements must be an array")
    }

    $ab = Get-JsonValue -Obj $Profile -Name 'allowlist_bounds'
    Deny-UnknownKeys -Obj $ab -Allowed $script:CATALOG_BOUND_KEYS -Where ($pid0 + '.allowlist_bounds')
    foreach ($k in $script:CATALOG_BOUND_KEYS) {
        $b = Get-JsonValue -Obj $ab -Name $k
        Deny-UnknownKeys -Obj $b -Allowed @('min', 'max') -Where ($pid0 + '.allowlist_bounds.' + $k)
        foreach ($mk in @('min', 'max')) {
            if (-not (Test-JsonNonNegativeInteger (Get-JsonValue -Obj $b -Name $mk))) {
                Stop-Launcher 'fail_gate_catalog' ($pid0 + ": allowlist_bounds." + $k + "." + $mk + " must be a non-negative integer")
            }
        }
        if ([long](Get-JsonValue -Obj $b -Name 'min') -gt [long](Get-JsonValue -Obj $b -Name 'max')) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": allowlist_bounds." + $k + " min > max")
        }
    }
    $qmin = [long](Get-JsonValue -Obj (Get-JsonValue -Obj $ab -Name 'qd') -Name 'min')
    $qmax = [long](Get-JsonValue -Obj (Get-JsonValue -Obj $ab -Name 'qd') -Name 'max')
    if ($qmin -lt $script:ENGINE_QD_MIN -or $qmax -gt $script:ENGINE_QD_MAX) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": allowlist_bounds.qd outside engine range 1..63")
    }
    $bmin = [long](Get-JsonValue -Obj (Get-JsonValue -Obj $ab -Name 'budget_mb') -Name 'min')
    if ($bmin -lt [long](Get-JsonValue -Obj $Profile -Name 'min_budget_mb')) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": allowlist_bounds.budget_mb.min below min_budget_mb")
    }

    $d = Get-JsonValue -Obj $Profile -Name 'defaults'
    Deny-UnknownKeys -Obj $d -Allowed @('argv', 'env') -Where ($pid0 + '.defaults')
    $argv = Get-JsonValue -Obj $d -Name 'argv'
    if (-not (Test-JsonArray $argv)) { Stop-Launcher 'fail_gate_catalog' ($pid0 + ": defaults.argv must be an array") }
    foreach ($a in @($argv)) {
        if (-not ($a -is [string])) { Stop-Launcher 'fail_gate_catalog' ($pid0 + ": defaults.argv entries must be strings") }
    }

    # Expect digest is re-hashed here so identification cannot run on an unverified expect file.
    $expectPath = Join-Path (Join-Path $Root ([string](Get-JsonValue -Obj $Runtime -Name 'expects_dir'))) ([string](Get-JsonValue -Obj $Profile -Name 'expect_file'))
    if (-not (Test-Path -LiteralPath $expectPath -PathType Leaf)) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": expect file missing: " + $expectPath)
    }
    $h = Get-FileSha256Lower -Path $expectPath
    if (-not $h.ok) { Stop-Launcher 'fail_gate_catalog' ($pid0 + ": expect hash failed - " + $h.reason) }
    if ($h.sha -ne ([string](Get-JsonValue -Obj $Profile -Name 'expect_sha256')).ToLowerInvariant()) {
        Stop-Launcher 'fail_gate_catalog' ($pid0 + ": expect file re-hash != catalog expect_sha256")
    }

    # LS OA-1 (M1) - Codex r1 F1: a pin's ARITY is part of its meaning, and the catalog already knows
    # the answer. expect.sources[] is the shard count this profile describes, so a pin with a
    # different length can never match ANY file set. Left unchecked it looks like a formatting
    # success and then fails at comparison time, which sends a model the catalog does actually
    # describe down the arch-template row - laundering a catalog typo into an "experimental"
    # downgrade, the exact substitution M1's hard-fail rule exists to forbid. Checked here, where the
    # expect has just been proven to be the approved bytes. The expect is parsed only when a pin is
    # present, so an unpinned catalog reaches no new failure mode.
    # Assign, then wrap: Get-JsonValue returns its value through the unary-comma idiom, so
    # "@(Get-JsonValue ...)" would count the wrapper and report 1 for every array.
    $pinArr = @()
    if (Test-JsonHas -Obj $Profile -Name 'source_shards_sha256') {
        $pinRaw = Get-JsonValue -Obj $Profile -Name 'source_shards_sha256'
        $pinArr = @($pinRaw)
    }
    if ($pinArr.Count -gt 0) {
        $er = Read-JsonFileStrict -Path $expectPath
        if (-not $er.ok) { Stop-Launcher 'fail_gate_catalog' ($pid0 + ": expect unreadable while checking the source pin - " + $er.reason) }
        $srcs = Get-JsonValue -Obj $er.value -Name 'sources'
        if (-not (Test-JsonArray $srcs)) { Stop-Launcher 'fail_gate_catalog' ($pid0 + ": expect sources[] missing while checking the source pin") }
        if (@($srcs).Count -ne $pinArr.Count) {
            Stop-Launcher 'fail_gate_catalog' ($pid0 + ": source_shards_sha256 has " + $pinArr.Count +
                " digest(s) but the expect describes " + @($srcs).Count + " source shard(s)")
        }
    }
}

function Get-CatalogProfileById {
    param($Catalog, [string] $ProfileId)
    foreach ($p in (Get-JsonArray -Obj $Catalog -Name 'profiles')) {
        if ([string](Get-JsonValue -Obj $p -Name 'profile_id') -ceq $ProfileId) { return $p }
    }
    return $null
}

function Get-ExpectPath {
    param([string] $Root, $Catalog, $Profile)
    $rt = Get-JsonValue -Obj $Catalog -Name 'runtime'
    return (Join-Path (Join-Path $Root ([string](Get-JsonValue -Obj $rt -Name 'expects_dir'))) ([string](Get-JsonValue -Obj $Profile -Name 'expect_file')))
}

# LS 1-9: catalog is the single source of truth for published numbers; any measurement whose
# condition columns are incomplete is hidden and rendered as [unmeasured].
function Format-ReferenceMeasurements {
    param($Profile)
    $rows = @()
    foreach ($m in (Get-JsonArray -Obj $Profile -Name 'reference_measurements')) {
        $complete = $true
        foreach ($c in $script:MEASUREMENT_COLUMNS) {
            if (-not (Test-JsonHas -Obj $m -Name $c)) { $complete = $false; break }
            if ($null -eq (Get-JsonValue -Obj $m -Name $c)) { $complete = $false; break }
        }
        if (-not $complete) { $rows += '  [unmeasured] (condition columns incomplete - hidden by LS 1-9)'; continue }
        $rows += ('  {0} | {1} | {2} | {3} | {4} | {5} tok/s' -f
            [string](Get-JsonValue -Obj $m -Name 'model'),
            [string](Get-JsonValue -Obj $m -Name 'tier'),
            [string](Get-JsonValue -Obj $m -Name 'machine_storage'),
            [string](Get-JsonValue -Obj $m -Name 'budget_qd_prefetch'),
            [string](Get-JsonValue -Obj $m -Name 'workload_window'),
            [string](Get-JsonValue -Obj $m -Name 'observed_tok_s'))
    }
    if ($rows.Count -eq 0) { $rows = @('  (no reference measurement rows in catalog)') }
    return , $rows
}

# endregion

# ============================================================================
# region 7. GGUF HEADER + MULTI-SHARD IDENTIFICATION (LS 1-5)
#   Rules ported 1:1 from bench/repack/repack_experts.py:252-322 (discover_shard_paths /
#   load_model_shards). No new heuristic is introduced here.
# ============================================================================

$script:SPLIT_REGEX = '^(?<base>.+)-(?<idx>\d{5})-of-(?<cnt>\d{5})\.gguf$'

function Get-ShardPaths {
    param([string] $ModelPath)
    $base = [System.IO.Path]::GetFileName($ModelPath)
    $m = [regex]::Match($base, $script:SPLIT_REGEX)
    if (-not $m.Success) { return @{ ok = $true; paths = @($ModelPath); split = $false; count = 1 } }
    $dir = [System.IO.Path]::GetDirectoryName($ModelPath)
    $prefix = $m.Groups['base'].Value
    $cnt = [int]$m.Groups['cnt'].Value
    if ($cnt -lt 1) { return @{ ok = $false; reason = ("split count 0: " + $base) } }
    $paths = @()
    for ($i = 1; $i -le $cnt; $i++) {
        $sib = Join-Path $dir ('{0}-{1:d5}-of-{2:d5}.gguf' -f $prefix, $i, $cnt)
        if (-not (Test-Path -LiteralPath $sib -PathType Leaf)) {
            return @{ ok = $false; reason = ("incomplete shard set: missing sibling " + $sib + " (expected " + $cnt + " shards)") }
        }
        $paths += $sib
    }
    if ($paths.Count -ne $cnt) {
        return @{ ok = $false; reason = ("incomplete shard set: found " + $paths.Count + " != split count " + $cnt) }
    }
    return @{ ok = $true; paths = $paths; split = $true; count = $cnt }
}

# GGUF value type ids (gguf.h). Arrays recurse; strings are length-prefixed.
function Read-GgufString {
    param($Reader)
    $len = $Reader.ReadUInt64()
    if ($len -gt 16777216) { throw "gguf string length out of range" }
    $bytes = $Reader.ReadBytes([int]$len)
    return [System.Text.Encoding]::UTF8.GetString($bytes)
}

function Skip-GgufString {
    param($Reader)
    $len = $Reader.ReadUInt64()
    [void]$Reader.BaseStream.Seek([long]$len, [System.IO.SeekOrigin]::Current)
}

function Read-GgufValue {
    param($Reader, [uint32] $Type, [bool] $Want)
    switch ($Type) {
        0  { $v = $Reader.ReadByte();    if ($Want) { return [long]$v } ; return $null }
        1  { $v = $Reader.ReadSByte();   if ($Want) { return [long]$v } ; return $null }
        2  { $v = $Reader.ReadUInt16();  if ($Want) { return [long]$v } ; return $null }
        3  { $v = $Reader.ReadInt16();   if ($Want) { return [long]$v } ; return $null }
        4  { $v = $Reader.ReadUInt32();  if ($Want) { return [long]$v } ; return $null }
        5  { $v = $Reader.ReadInt32();   if ($Want) { return [long]$v } ; return $null }
        6  { $v = $Reader.ReadSingle();  if ($Want) { return [double]$v } ; return $null }
        7  { $v = $Reader.ReadByte();    if ($Want) { return ($v -ne 0) } ; return $null }
        8  { if ($Want) { return (Read-GgufString -Reader $Reader) } ; Skip-GgufString -Reader $Reader; return $null }
        9  {
            $et = $Reader.ReadUInt32()
            $n  = $Reader.ReadUInt64()
            for ($i = [uint64]0; $i -lt $n; $i++) { [void](Read-GgufValue -Reader $Reader -Type $et -Want $false) }
            return $null   # array values are not used for identification
        }
        10 { $v = $Reader.ReadUInt64();  if ($Want) { return [long]$v } ; return $null }
        11 { $v = $Reader.ReadInt64();   if ($Want) { return [long]$v } ; return $null }
        12 { $v = $Reader.ReadDouble();  if ($Want) { return [double]$v } ; return $null }
        default { throw ("unknown gguf value type " + $Type) }
    }
}

# UX 1-3 (-LabelMode): the model-menu label needs the four IDENTIFICATION fields and nothing else.
# The ordinary counter below also counts the three split.* keys, so a split model that writes them
# before its structural keys can satisfy the count with zero structural fields read - which would
# turn a perfectly readable model into "[identify pending]". Label mode therefore wants
# general.architecture plus the three <arch>.* suffix keys only, and leaves split.* uncounted.
# It changes nothing for identify, which needs the split keys and keeps the default.
function Read-GgufHeader {
    param([string] $Path, [bool] $ExpectSplitKeys = $false, [bool] $LabelMode = $false)
    $fs = $null; $br = $null
    try {
        $fs = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
                  [System.IO.FileShare]::ReadWrite, 1048576, [System.IO.FileOptions]::SequentialScan)
        $br = New-Object System.IO.BinaryReader($fs)
        $magic = $br.ReadBytes(4)
        if ($magic.Length -ne 4 -or $magic[0] -ne 0x47 -or $magic[1] -ne 0x47 -or $magic[2] -ne 0x55 -or $magic[3] -ne 0x46) {
            return @{ ok = $false; reason = ("GGUF magic mismatch: " + $Path) }
        }
        $ver = $br.ReadUInt32()
        $nTensors = $br.ReadUInt64()
        $nKv = $br.ReadUInt64()
        $meta = @{}
        # Keys that identification needs. Parsing stops as soon as all of them are seen so a
        # 150k-entry tokenizer array is normally never walked.
        $needSuffix = @('.block_count', '.expert_count', '.expert_used_count')
        $needExact  = @('general.architecture', 'split.count', 'split.no', 'split.tensors.count')
        $wantCount = 4
        if ($ExpectSplitKeys) { $wantCount = 7 }
        if ($LabelMode) { $needExact = @('general.architecture'); $wantCount = 4 }
        $got = 0
        for ($i = [uint64]0; $i -lt $nKv; $i++) {
            $key = Read-GgufString -Reader $br
            $t = $br.ReadUInt32()
            $want = $false
            if ($needExact -contains $key) { $want = $true }
            else { foreach ($s in $needSuffix) { if ($key.EndsWith($s)) { $want = $true; break } } }
            $val = Read-GgufValue -Reader $br -Type $t -Want $want
            if ($want) {
                if (-not $meta.ContainsKey($key)) { $got++ }
                $meta[$key] = $val
            }
            if ($got -ge $wantCount) { break }
        }
        return @{ ok = $true; path = $Path; gguf_version = [long]$ver; n_tensors = [long]$nTensors;
                  meta = $meta; file_bytes = (New-Object System.IO.FileInfo($Path)).Length }
    } catch {
        return @{ ok = $false; reason = ("GGUF header parse failed (" + $Path + "): " + $_.Exception.Message) }
    } finally {
        if ($br) { $br.Dispose() }
        if ($fs) { $fs.Dispose() }
    }
}

function Get-ModelShardSet {
    param([string] $ModelPath)
    $disc = Get-ShardPaths -ModelPath $ModelPath
    if (-not $disc.ok) { Stop-Launcher 'fail_model_path' $disc.reason }

    # LS 11-6-b (UI-4): measured on the real sets this stage is the long silent one -
    # 26.0 s for the 397B 6-shard set and 19.7 s for the 4-shard MiniMax set on this machine.
    # Announce the stage, and for a split set report each shard as it is parsed, so a half-minute
    # of header reading cannot read as a frozen launcher.
    $shardCount = @($disc.paths).Count
    Write-Line ('[identify] reading GGUF headers ({0} shard(s)); large split sets take a moment...' -f $shardCount)

    $shards = @()
    $idx = 0
    foreach ($p in @($disc.paths)) {
        if ($shardCount -gt 1) {
            Write-Line ('           shard {0}/{1}: {2}' -f ($idx + 1), $shardCount, [System.IO.Path]::GetFileName($p))
        }
        $h = Read-GgufHeader -Path $p -ExpectSplitKeys ([bool]$disc.split)
        if (-not $h.ok) { Stop-Launcher 'fail_model_path' ("unsupported GGUF: " + $h.reason) }
        $h['source_index'] = $idx
        $shards += $h
        $idx++
    }

    # arch consistency across shards that carry the key (repack_experts.py:283-292)
    $arch = $null
    foreach ($h in $shards) {
        if ($h.meta.ContainsKey('general.architecture')) {
            $a = [string]$h.meta['general.architecture']
            if ($null -eq $arch) { $arch = $a }
            elseif ($a -cne $arch) {
                Stop-Launcher 'fail_model_path' ("architecture metadata conflicts between shards: " + $arch + " vs " + $a)
            }
        }
    }
    if ($null -eq $arch) { Stop-Launcher 'fail_model_path' 'unsupported GGUF: general.architecture absent in every shard' }

    # split KV cross-check, present-only (repack_experts.py:294-317)
    $totalTensors = 0
    foreach ($h in $shards) { $totalTensors += [long]$h.n_tensors }
    foreach ($h in $shards) {
        if ($h.meta.ContainsKey('split.count')) {
            $sc = [long]$h.meta['split.count']
            if ($sc -ne $shards.Count) {
                Stop-Launcher 'fail_model_path' ("incomplete shard set: split.count(" + $sc + ") != discovered shards(" + $shards.Count + ")")
            }
        }
        if ($h.meta.ContainsKey('split.no')) {
            $sn = [long]$h.meta['split.no']
            if ($sn -ne [long]$h.source_index) {
                Stop-Launcher 'fail_model_path' ("incomplete shard set: split.no(" + $sn + ") != source_index(" + $h.source_index + ")")
            }
        }
        if ($h.meta.ContainsKey('split.tensors.count')) {
            $stc = [long]$h.meta['split.tensors.count']
            if ($stc -ne $totalTensors) {
                Stop-Launcher 'fail_model_path' ("incomplete shard set: split.tensors.count(" + $stc + ") != summed tensors(" + $totalTensors + ")")
            }
        }
    }

    $merged = @{}
    foreach ($h in $shards) {
        foreach ($k in $h.meta.Keys) { if (-not $merged.ContainsKey($k)) { $merged[$k] = $h.meta[$k] } }
    }
    $totalBytes = 0
    foreach ($h in $shards) { $totalBytes += [long]$h.file_bytes }

    Write-Diag -Kind 'SHARDS' -Data @{ count = $shards.Count; split = $disc.split; arch = $arch;
                                       total_bytes = $totalBytes; tensors = $totalTensors }
    return @{ shards = $shards; arch = $arch; meta = $merged; total_bytes = $totalBytes;
              total_tensors = $totalTensors; is_split = $disc.split }
}

function Get-ArchMetaLong {
    param($ModelSet, [string] $Suffix)
    $key = $ModelSet.arch + $Suffix
    if ($ModelSet.meta.ContainsKey($key)) { return [long]$ModelSet.meta[$key] }
    return $null
}

# ---------------------------------------------------------------------------------------------
# Source shard hashing - ONE implementation, two readers (LS OA-1 M1 + WARMSTART A-4).
# The digests are computed once per FILE IDENTITY (volume serial + 128 bit file id + size + mtime)
# and cached under the launcher state directory, because LS 1-5 forbids re-hashing a 400 GB model at
# every start. The path is deliberately not part of the key and never sufficient on its own: a
# replacement file at the same path with the same length and timestamp has a different file id and
# is re-hashed. A file whose identity cannot be obtained takes no part in the cache at all - neither
# a hit nor a stored entry - and is hashed on every pass instead, which costs time and can never
# mis-bind. The caller owns its own process-level cache and its own diagnostic kinds, so the two
# readers cannot borrow each other's latches or pollute each other's diagnostic counts.
# ---------------------------------------------------------------------------------------------
function Get-ShardShaCachePath { return (Join-Path (Get-LauncherStateDir) $script:KV_SHARD_CACHE_FILE) }

function Get-ModelShardSha256Set {
    param($ModelSet, [string] $NoticeTag = 'kv',
          [string] $IdentityDiagKind = 'WARMSTART_SHARD_IDENTITY_UNAVAILABLE',
          [string] $HashFailDiagKind = 'WARMSTART_SHARD_HASH_FAILED',
          [string] $CacheFailDiagKind = 'WARMSTART_SHARD_CACHE_FAILED')
    if ($null -eq $ModelSet) { return @{ ok = $false; reason = 'no identified shard set' } }
    $cache = @{}
    $cachePath = Get-ShardShaCachePath
    $r = $null
    if (Test-Path -LiteralPath $cachePath -PathType Leaf) { $r = Read-JsonFileStrict -Path $cachePath }
    if ($null -ne $r -and $r.ok) {
        $ver = Get-JsonValue -Obj $r.value -Name 'cache_version'
        if ((Test-JsonNonNegativeInteger $ver) -and ([long]$ver -eq [long]$script:KV_SHARD_CACHE_VERSION)) {
            foreach ($e in (Get-JsonArray -Obj $r.value -Name 'entries')) {
                $k = [string](Get-JsonValue -Obj $e -Name 'key')
                $v = [string](Get-JsonValue -Obj $e -Name 'sha256')
                if ($k -and (Test-Sha256Hex $v)) { $cache[$k] = $v.ToLowerInvariant() }
            }
        }
    }
    $out = @()
    $dirty = $false
    foreach ($s in @($ModelSet.shards)) {
        $path = [string]$s.path
        $id = Get-FileIdentity -Path $path
        $key = $null
        $len = [long]0
        if ($id.ok) {
            # <volume serial>:<128 bit file id> | <size> | <mtime> - LS 13-7 (9).
            $key = ('v2|' + $id.identity + '|' + $id.size + '|' + $id.mtime)
            $len = [long]$id.size
        } else {
            Write-Diag -Kind $IdentityDiagKind -Data @{ path = $path; reason = $id.reason }
            try { $len = [long](New-Object System.IO.FileInfo($path)).Length }
            catch {
                Write-Diag -Kind $HashFailDiagKind -Data @{ path = $path; reason = $_.Exception.Message }
                return @{ ok = $false; reason = [string]$_.Exception.Message }
            }
        }
        if ($null -ne $key -and $cache.ContainsKey($key)) { $out += $cache[$key]; continue }
        Write-KvHashNotice -What ('hashing model shard ' + [System.IO.Path]::GetFileName($path)) -Bytes $len -Tag $NoticeTag
        $h = Get-FileSha256Lower -Path $path
        if (-not $h.ok) {
            Write-Diag -Kind $HashFailDiagKind -Data @{ path = $path; reason = $h.reason }
            return @{ ok = $false; reason = [string]$h.reason }
        }
        if ($null -ne $key) { $cache[$key] = $h.sha; $dirty = $true }
        $out += $h.sha
    }
    if ($dirty) {
        # Best effort: a cache that cannot be stored only costs time on the next run.
        try {
            $entries = @()
            foreach ($k in $cache.Keys) { $entries += [ordered]@{ key = [string]$k; sha256 = [string]$cache[$k] } }
            $obj = [ordered]@{ cache_version = [int]$script:KV_SHARD_CACHE_VERSION; entries = $entries }
            $dir = Split-Path -Parent $cachePath
            if (-not (Test-Path -LiteralPath $dir -PathType Container)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
            $tmp = $cachePath + '.tmp'
            [System.IO.File]::WriteAllText($tmp, ($obj | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
            Move-FileAtomic -TempPath $tmp -FinalPath $cachePath
        } catch {
            Write-Diag -Kind $CacheFailDiagKind -Data @{ path = $cachePath; reason = $_.Exception.Message }
        }
    }
    return @{ ok = $true; shas = @($out) }
}

# LS OA-1: the profile id becomes a directory name and a lock token, and the arch it embeds is read
# out of a GGUF the user downloaded. That is untrusted input on a real, reachable path, so the token
# shape is enforced before anything joins it onto a path.
function Test-PathSafeToken {
    param([string] $Value, [string] $Pattern)
    if ($null -eq $Value) { return $false }
    if ($Value -cne $Value.ToLowerInvariant()) { return $false }
    return ($Value -cmatch $Pattern)
}

# LS OA-1 (M1): the structural fingerprint is now a PREFILTER that narrows the SHA candidates, not
# an authority of its own. This is the loop that used to be the whole of Select-Profile; it is
# extracted verbatim so both the pinned path and the legacy unpinned path run the exact same
# comparison (arch + layer/expert counts + per-shard file_bytes from the profile's expect + total).
function Get-StructuralProfileCandidates {
    param($Catalog, $ModelSet, [string] $Root)
    $nLayer = Get-ArchMetaLong -ModelSet $ModelSet -Suffix '.block_count'
    $nExp   = Get-ArchMetaLong -ModelSet $ModelSet -Suffix '.expert_count'
    $nExpU  = Get-ArchMetaLong -ModelSet $ModelSet -Suffix '.expert_used_count'

    $matches = @()
    foreach ($p in (Get-JsonArray -Obj $Catalog -Name 'profiles')) {
        $id = Get-JsonValue -Obj $p -Name 'identify'
        if ([string](Get-JsonValue -Obj $id -Name 'arch') -cne $ModelSet.arch) { continue }
        if ($null -eq $nLayer -or [long](Get-JsonValue -Obj $id -Name 'n_layer') -ne $nLayer) { continue }
        if ($null -eq $nExp   -or [long](Get-JsonValue -Obj $id -Name 'n_expert') -ne $nExp) { continue }
        if ($null -eq $nExpU  -or [long](Get-JsonValue -Obj $id -Name 'n_expert_used') -ne $nExpU) { continue }

        $expectPath = Get-ExpectPath -Root $Root -Catalog $Catalog -Profile $p
        $er = Read-JsonFileStrict -Path $expectPath
        if (-not $er.ok) { Stop-Launcher 'fail_gate_catalog' ("expect unreadable: " + $expectPath + " - " + $er.reason) }
        $sources = Get-JsonValue -Obj $er.value -Name 'sources'
        if (-not (Test-JsonArray $sources)) { Stop-Launcher 'fail_gate_catalog' ("expect sources[] missing: " + $expectPath) }
        if (@($sources).Count -ne $ModelSet.shards.Count) { continue }
        $ok = $true
        $sum = 0
        for ($i = 0; $i -lt @($sources).Count; $i++) {
            $fb = Get-JsonValue -Obj (@($sources)[$i]) -Name 'file_bytes'
            if (-not (Test-JsonNonNegativeInteger $fb)) { $ok = $false; break }
            if ([long]$fb -ne [long]$ModelSet.shards[$i].file_bytes) { $ok = $false; break }
            $sum += [long]$fb
        }
        if (-not $ok) { continue }
        if ($sum -ne [long]$ModelSet.total_bytes) { continue }
        $matches += $p
    }
    return , @($matches)
}

# Absent key and empty array are the SAME state: no pin recorded (LS OA-1, catalog optional key).
function Get-ProfilePinShas {
    param($Profile)
    if (-not (Test-JsonHas -Obj $Profile -Name 'source_shards_sha256')) { return , @() }
    $v = Get-JsonValue -Obj $Profile -Name 'source_shards_sha256'
    if (-not (Test-JsonArray $v)) { return , @() }
    $out = @()
    foreach ($h in @($v)) { $out += ([string]$h).ToLowerInvariant() }
    return , @($out)
}

# Ordered, exact, whole-set equality. Order matters because the pin is written per shard index and
# the shard set is discovered in split order, so a permutation is a different file set.
function Test-ProfilePinMatch {
    param([string[]] $Pin, [string[]] $Actual)
    if (@($Pin).Count -eq 0) { return $false }
    if (@($Pin).Count -ne @($Actual).Count) { return $false }
    for ($i = 0; $i -lt @($Pin).Count; $i++) {
        if (([string]@($Pin)[$i]).ToLowerInvariant() -cne ([string]@($Actual)[$i]).ToLowerInvariant()) { return $false }
    }
    return $true
}

# LS 1-5: size comparison is over the summed shard set; the single selected file is never used.
# 397B shard 1 is metadata-only, so per-shard size comes from expect.sources[] and never from a
# size heuristic.
function Select-Profile {
    param($Catalog, $ModelSet, [string] $Root, $Candidates = $null)
    $nLayer = Get-ArchMetaLong -ModelSet $ModelSet -Suffix '.block_count'
    $nExp   = Get-ArchMetaLong -ModelSet $ModelSet -Suffix '.expert_count'
    $nExpU  = Get-ArchMetaLong -ModelSet $ModelSet -Suffix '.expert_used_count'

    # The prefilter moved into its own function (LS OA-1) so the pinned path can reuse it; passing a
    # pre-computed candidate list in avoids running it twice. Behaviour with no list supplied is the
    # v0.4 behaviour, unchanged, including the "unsupported GGUF" stop and the ambiguity prompt.
    # The candidate list is ASSIGNED before it is wrapped. Get-StructuralProfileCandidates returns
    # its array through the unary-comma idiom so an empty result survives as an empty array, and
    # "@(f)" around such a call collects the wrapper itself - one element, always. Assigning first
    # unwraps it exactly once, which is what every other reader of this idiom in the file does.
    $matches = @()
    if ($null -eq $Candidates) {
        $found = Get-StructuralProfileCandidates -Catalog $Catalog -ModelSet $ModelSet -Root $Root
        $matches = @($found)
    } else { $matches = @($Candidates) }

    if ($matches.Count -eq 0) {
        Write-Line ''
        Write-Line '[identify] Unsupported GGUF - stopping before any write.'
        Write-Line ('           arch={0} n_layer={1} n_expert={2} n_expert_used={3} shards={4} bytes={5}' -f
            $ModelSet.arch, $nLayer, $nExp, $nExpU, $ModelSet.shards.Count, $ModelSet.total_bytes)
        Write-Line '           Please report this model at the project issue tracker (see README).'
        Stop-Launcher 'fail_model_path' 'unsupported GGUF: no catalog profile matches the header fingerprint'
    }
    if ($matches.Count -gt 1) {
        if ($NonInteractive) {
            Stop-Launcher 'fail_model_path' 'ambiguous shard set: multiple catalog profiles match (non-interactive)'
        }
        Write-Line ''
        Write-Line '[identify] Multiple catalog profiles match this model. Select one:'
        for ($i = 0; $i -lt $matches.Count; $i++) {
            Write-Line ('  {0}) {1}' -f ($i + 1), [string](Get-JsonValue -Obj $matches[$i] -Name 'profile_id'))
        }
        $ans = Read-UserLine -Prompt 'select> '
        $n = 0
        if (-not [int]::TryParse(([string]$ans).Trim(), [ref]$n) -or $n -lt 1 -or $n -gt $matches.Count) {
            Stop-Launcher 'fail_model_path' 'ambiguous shard set: no valid profile selected'
        }
        return $matches[$n - 1]
    }
    return $matches[0]
}

# ---------------------------------------------------------------------------------------------
# LS OA-1 (M1) - the identification verdict.
#
# Before M1 a header fingerprint alone granted a catalog profile, which meant an identically shaped
# fine-tune (same arch, same counts, same per-shard byte sizes - a normal outcome of re-quantising
# the same architecture) was served as the reference-validated model. The fingerprint is now a
# prefilter over SHA candidates and the answer is one of three:
#   pinned    the profile records source digests AND they match this file set  -> catalog path
#   unpinned  the profile records no digests at all                            -> catalog path,
#             surfaced as model-pin(unpinned) / unvalidated (never silently "validated")
#   mismatch  every structural match records a pin and every one DISAGREED, and the private
#             arch-template switch was NOT given                               -> catalog path,
#             surfaced as model-pin(mismatch) / unvalidated, prefetch off(reason=identity_not_exact)
#   template  nothing structural matched, or every structural match had a pin that DISAGREED AND
#             the private switch WAS given
#             -> the arch-template path, which is the normal home of a file the catalog never saw
# A pin that MATCHED is a latch: from that point the template path is closed for this run, so a
# later catalog / expect / seal failure is a hard stop and can never be laundered into an
# "experimental" downgrade (OPEN_ARCH_DESIGN section 0).
#
# ---- why 'mismatch' exists, and exactly where its boundary with 'template' runs ----------------
# P4 2.5 / PI 3 invariant 5 require a disagreeing pin to be a PREFETCH disposition, not a refusal
# to run: "off(reason=identity_not_exact)" with the catalog row retained, because the direct-read
# body is unaffected by a prefetch decision. Before this branch existed that reason string was
# unreachable on the real path - with no unpinned sibling the run fell through to the
# "unsupported GGUF" fail_model_path stop below, so a re-quantised GGUF (the exact shape the spec
# names as its threat) could not run at all.
#
# The boundary is drawn at the private switch, deliberately:
#   - WITHOUT -ExperimentalArchTemplate this branch keeps the catalog row and reports 'mismatch'.
#     That is the branch P4 contradicts, and the only one it may change: today's behaviour there
#     is a hard stop, and PI/P4 require a non-terminal prefetch-only disposition instead.
#   - WITH -ExperimentalArchTemplate the v0.4 OA-1 semantics stand unchanged (template row, derived
#     profile, no catalog claims). LAUNCHER_SPEC's later-authority clause hands P4 exactly five
#     surfaces, and of LS 15 it hands over only the "derived-profile prefetch fields" - NOT OA-1's
#     selection branches. A user who explicitly opened the template path asked for that path, and
#     rewriting it here would edit a frozen contract this atomic step has no authority over.
# The safety property OA-1 M1 was built for is preserved in BOTH branches: a mismatched file never
# inherits the reference model's claims. It is served from the catalog row's geometry only, while
# Get-SurfaceAxes reports model-pin(mismatch) / unvalidated and the performance gate is demoted -
# the row's numbers describe a file this one demonstrably is not.
# ---------------------------------------------------------------------------------------------
$script:PinMatchedLatch = $false
# P4 2.5 companion latch. It is set by the same function that decides the verdict, and it is what
# stops a catalog row's published numbers from being printed next to a file that has just been
# PROVEN not to be the bytes those numbers were measured on. 'unpinned' does not set it: unchecked
# is not the same statement as checked-and-different, and OA-1 already answers unchecked in the
# surface-axes block.
$script:PinMismatchLatch = $false

function Resolve-ProfileSelection {
    param($Catalog, $ModelSet, [string] $Root, [bool] $TemplateAllowed)
    # Assign, then wrap - see the note in Select-Profile. The same applies to every Get-ProfilePinShas
    # result below.
    $candsRaw = Get-StructuralProfileCandidates -Catalog $Catalog -ModelSet $ModelSet -Root $Root
    $cands = @($candsRaw)
    $pinned = @()
    $unpinned = @()
    foreach ($p in $cands) {
        $pinOf = Get-ProfilePinShas -Profile $p
        if (@($pinOf).Count -gt 0) { $pinned += $p } else { $unpinned += $p }
    }

    # Hashing a multi-hundred-GB model is only worth doing when the answer can change the outcome:
    # a pin has to be checked, and the template path records the source attestation. Since 26-08-07
    # the catalog is partially pinned (qwen35-122b-nonextn), so a run that structurally matches THAT
    # profile does pay one hash of its two shards - the cost the exact-identity gate is made of, and
    # the result is cached per file. A run matching any of the five unpinned profiles still pays
    # nothing.
    $needShas = ($pinned.Count -gt 0) -or ($TemplateAllowed -and $unpinned.Count -eq 0)
    $shas = @()
    if ($needShas) {
        Write-Line '[identify] hashing the source shards (once per file; the result is cached)...'
        $r = Get-ModelShardSha256Set -ModelSet $ModelSet -NoticeTag 'identify' `
                 -IdentityDiagKind 'IDENTIFY_SHARD_IDENTITY_UNAVAILABLE' `
                 -HashFailDiagKind 'IDENTIFY_SHARD_HASH_FAILED' `
                 -CacheFailDiagKind 'IDENTIFY_SHARD_CACHE_FAILED'
        if (-not $r.ok) {
            # Attempted only where the answer decides something, so an unanswerable hash is a
            # refusal, not a downgrade: the alternative would be to pick a profile whose byte claim
            # could not be checked.
            Stop-Launcher 'fail_model_path' ('source shard hashing failed, the profile pin cannot be decided: ' + [string]$r.reason)
        }
        $shas = @($r.shas)
    }

    $hits = @()
    foreach ($p in $pinned) {
        $pinOf = Get-ProfilePinShas -Profile $p
        if (Test-ProfilePinMatch -Pin $pinOf -Actual $shas) { $hits += $p }
    }
    if ($hits.Count -gt 1) {
        Stop-Launcher 'fail_gate_catalog' ('two catalog profiles are pinned to the same source bytes: ' +
            (($hits | ForEach-Object { [string](Get-JsonValue -Obj $_ -Name 'profile_id') }) -join ', '))
    }
    if ($hits.Count -eq 1) {
        $script:PinMatchedLatch = $true
        Write-Diag -Kind 'PROFILE_PIN' -Data @{ verdict = 'pinned'; shards = @($shas).Count
                                                 profile = [string](Get-JsonValue -Obj $hits[0] -Name 'profile_id') }
        return @{ kind = 'pinned'; profile = $hits[0]; shas = @($shas); candidates = @($cands) }
    }

    if ($unpinned.Count -gt 0) {
        $prof = Select-Profile -Catalog $Catalog -ModelSet $ModelSet -Root $Root -Candidates $unpinned
        Write-Diag -Kind 'PROFILE_PIN' -Data @{ verdict = 'unpinned'; shards = @($shas).Count
                                                 pinned_candidates_rejected = $pinned.Count
                                                 profile = [string](Get-JsonValue -Obj $prof -Name 'profile_id') }
        return @{ kind = 'unpinned'; profile = $prof; shas = @($shas); candidates = @($cands) }
    }

    # P4 2.5 - the disagreeing pin. See the boundary note above the function: this branch is the
    # DEFAULT path only. Ambiguity is resolved by the same Select-Profile rules the unpinned branch
    # uses, so a multi-candidate catalog keeps its existing prompt / non-interactive refusal.
    if ($pinned.Count -gt 0 -and -not $TemplateAllowed) {
        $script:PinMismatchLatch = $true
        $prof = Select-Profile -Catalog $Catalog -ModelSet $ModelSet -Root $Root -Candidates $pinned
        Write-Line '[identify] the catalog pin for this profile does NOT match this file - continuing with'
        Write-Line '           the catalog geometry, prefetch off, and every reference claim withheld.'
        Write-Diag -Kind 'PROFILE_PIN' -Data @{ verdict = 'mismatch'; shards = @($shas).Count
                                                 pinned_candidates_rejected = $pinned.Count
                                                 profile = [string](Get-JsonValue -Obj $prof -Name 'profile_id') }
        return @{ kind = 'mismatch'; profile = $prof; shas = @($shas); candidates = @($cands) }
    }

    if (-not $TemplateAllowed) {
        # No template entry: reproduce the v0.4 refusal exactly, including its message block. An
        # empty candidate list is what Select-Profile turns into the "unsupported GGUF" stop.
        [void](Select-Profile -Catalog $Catalog -ModelSet $ModelSet -Root $Root -Candidates @())
    }
    Write-Diag -Kind 'PROFILE_PIN' -Data @{ verdict = 'template'; shards = @($shas).Count
                                             structural_candidates = $cands.Count
                                             pinned_candidates_rejected = $pinned.Count }
    return @{ kind = 'template'; profile = $null; shas = @($shas); candidates = @($cands) }
}

# endregion

# ============================================================================
# region 7b. DERIVED PROFILE / derive-plan (LS OA-1, OPEN_ARCH_DESIGN section 3)
#
# A model the catalog never saw has no profile, and the launcher needs one before it can size
# anything: the budget floor is n_expert * slot_stride_max, and neither number exists until an
# alignment query has been made against the OUTPUT volume. So the unregistered path runs a
# WRITE-NOTHING derive-plan first, in six steps:
#   1 parse every shard header and decide the source pin        (region 7, already done by caller)
#   2 close the routed inventory from the frozen arch template
#   3 query the output volume alignment -> slot_stride_max
#   4 min_budget = ceil(n_expert * slot_stride_max / MiB)
#   5 complete the derived profile in memory and validate it
#   6 resource gate + user confirmation, THEN the real repack writes the derived expect
# Steps 2 and 3 are one repacker "--plan --experimental-arch-template" call: that mode already
# closes the inventory (repack_experts.py:840 derive_arch_template), already resolves the output
# alignment (:1449 resolve_alignment) and already prints both plus the derived expect body, while
# writing zero bytes (cmd_plan calls neither _append_repack_log nor write_derived_expect). Writing
# a second GGUF/template parser here would be a second source of truth for the same question - the
# independent cross-check belongs to the ENGINE (B axis), which regenerates the expected tensor set
# from the live arch instead of reusing the repacker's regexes.
# ============================================================================

function Get-DerivedExpectPath {
    param([string] $OutputDir)
    return (Join-Path $OutputDir $script:DERIVED_EXPECT_FILE_NAME)
}

function Get-DerivedLockId {
    param([string] $DerivedFrom)
    return ($script:DERIVED_LOCK_ID_PREFIX + $DerivedFrom)
}

# ---------------------------------------------------------------------------------------------
# The plan is consumed as text, so this parser is the whole contract surface between the two
# programs and it is deliberately strict: every keyed line must appear EXACTLY once (a duplicate
# means the capture is not a single clean plan), every number must be a number, and the derived
# expect body is re-parsed and cross-checked field by field against the summary lines. That last
# step is what makes a partially garbled capture fail instead of half-parsing.
# Line formats are 1st source repack_experts.py:1376-1406 (_print_plan_summary) and :2122-2126
# (cmd_plan). The only real drift risk is that the producer's wording changes under us; the selftest
# case that runs the REAL repacker (E5-j) is the only thing that can detect that, which is why it
# exists and why it may not be replaced by the mock alone.
# ---------------------------------------------------------------------------------------------
function ConvertFrom-TemplatePlanText {
    param([string] $Text)
    if ([string]::IsNullOrEmpty($Text)) { return @{ ok = $false; reason = 'the plan produced no output' } }

    $reDerive = '^\[EXPERIMENTAL arch-template\] derived_from=(\S+) routed_scope=(\S+) \(template default=(\S+)\) inventory_sha256=([0-9a-f]{64})$'
    $reTpl    = '^\[EXPERIMENTAL arch-template\] template layers=(\d+)\.\.(\d+) \((\d+)\) routed_tensors=(\d+) '
    $reArch   = '^arch=(\S+) n_layer=(\d+) n_expert=(\d+) n_expert_used=(\d+) schema=(\S+) bias=(\S+)$'
    $reMoe    = '^moe_layers: (\d+) entries \[(\d+)\.\.(\d+)\]'
    $reStride = '^output alignment A=(\d+), stride\[l\] .* \(min=(\d+) max=(\d+)\), slot_stride_max=(\d+)$'
    $reBytes  = '^expert_payload_total\(=expert_bytes\)=(\d+)$'
    $expectHead = '--- derived expect (' + $script:DERIVED_EXPECT_FILE_NAME + ', not written in --plan) ---'
    $planDone   = '--plan done (0 bytes written, no GPU used)'

    $found = @{}
    $lines = $Text -split "`n"
    $expectLines = @()
    $inExpect = $false
    $sawDone = $false
    foreach ($raw in $lines) {
        $ln = ([string]$raw).TrimEnd("`r")
        if ($ln -ceq $planDone) { $sawDone = $true; $inExpect = $false; continue }
        if ($ln -ceq $expectHead) {
            if ($found.ContainsKey('expect_head')) { return @{ ok = $false; reason = 'the derived expect header appears more than once' } }
            $found['expect_head'] = $true
            $inExpect = $true
            continue
        }
        if ($inExpect) { $expectLines += $ln; continue }
        foreach ($pair in @(@('derive', $reDerive), @('tpl', $reTpl), @('arch', $reArch),
                            @('moe', $reMoe), @('stride', $reStride), @('bytes', $reBytes))) {
            $m = [regex]::Match($ln, $pair[1])
            if ($m.Success) {
                if ($found.ContainsKey($pair[0])) {
                    return @{ ok = $false; reason = ('the plan carries more than one "' + $pair[0] + '" line') }
                }
                $found[$pair[0]] = $m
            }
        }
    }
    foreach ($k in @('derive', 'tpl', 'arch', 'moe', 'stride', 'bytes')) {
        if (-not $found.ContainsKey($k)) { return @{ ok = $false; reason = ('the plan has no "' + $k + '" line') } }
    }
    if (-not $found.ContainsKey('expect_head')) { return @{ ok = $false; reason = 'the plan printed no derived expect body' } }
    if (-not $sawDone) { return @{ ok = $false; reason = 'the plan output is not terminated by its completion line (truncated capture)' } }

    $derivedFrom = [string]$found['derive'].Groups[1].Value
    $parts = $derivedFrom -split '@'
    if ($parts.Count -ne 2) { return @{ ok = $false; reason = ('derived_from is not <template_id>@<version>: ' + $derivedFrom) } }
    $templateId = [string]$parts[0]
    $templateVersion = [string]$parts[1]
    if (-not (Test-PathSafeToken -Value $templateId -Pattern $script:ARCH_TOKEN_REGEX)) {
        return @{ ok = $false; reason = ('template id is not a safe token: ' + $templateId) }
    }
    if ($templateVersion -cnotmatch '^\d{1,8}$') {
        return @{ ok = $false; reason = ('template version is not a number: ' + $templateVersion) }
    }
    $scope = [string]$found['derive'].Groups[2].Value
    if (@('all', 'execution') -notcontains $scope) { return @{ ok = $false; reason = ('unknown routed scope: ' + $scope) } }
    $arch = [string]$found['arch'].Groups[1].Value
    if (-not (Test-PathSafeToken -Value $arch -Pattern $script:ARCH_TOKEN_REGEX)) {
        return @{ ok = $false; reason = ('arch is not a safe token: ' + $arch) }
    }
    if ($arch -cne $templateId) {
        return @{ ok = $false; reason = ('arch (' + $arch + ') and template id (' + $templateId + ') disagree') }
    }

    $out = @{
        ok                 = $true
        derived_from       = $derivedFrom
        template_id        = $templateId
        template_version   = $templateVersion
        routed_scope       = $scope
        inventory_sha256   = ([string]$found['derive'].Groups[4].Value).ToLowerInvariant()
        arch               = $arch
        n_layer            = [long]$found['arch'].Groups[2].Value
        n_expert           = [long]$found['arch'].Groups[3].Value
        n_expert_used      = [long]$found['arch'].Groups[4].Value
        moe_layers         = [long]$found['moe'].Groups[1].Value
        template_layers    = [long]$found['tpl'].Groups[3].Value
        routed_tensors     = [long]$found['tpl'].Groups[4].Value
        slot_stride_max    = [long]$found['stride'].Groups[4].Value
        expert_bytes_total = [long]$found['bytes'].Groups[1].Value
        expect_text        = ($expectLines -join "`n")
    }
    if ($out.n_expert -le 0)        { return @{ ok = $false; reason = 'n_expert is not positive' } }
    if ($out.slot_stride_max -le 0) { return @{ ok = $false; reason = 'slot_stride_max is not positive' } }
    if ($out.moe_layers -le 0)      { return @{ ok = $false; reason = 'the plan reports no MoE layers' } }
    if ($out.moe_layers -ne $out.template_layers) {
        return @{ ok = $false; reason = ('the template layer count (' + $out.template_layers +
                                         ') and the layout layer count (' + $out.moe_layers + ') disagree') }
    }

    # Second read of the same facts, from the expect body the repacker is about to write. The two
    # have to agree or the plan is not describing one consistent derivation.
    $er = ConvertFrom-JsonStrict -Text $out.expect_text
    if (-not $er.ok) { return @{ ok = $false; reason = ('the derived expect body is not strict JSON - ' + $er.reason) } }
    $ex = $er.value
    foreach ($chk in @(@('derived_from', $out.derived_from), @('template_id', $out.template_id),
                       @('template_version', $out.template_version), @('routed_scope', $out.routed_scope),
                       @('arch', $out.arch), @('inventory_sha256', $out.inventory_sha256))) {
        $v = Get-JsonValue -Obj $ex -Name $chk[0]
        if (([string]$v) -cne ([string]$chk[1])) {
            return @{ ok = $false; reason = ('derived expect ' + $chk[0] + '=' + [string]$v +
                                             ' disagrees with the plan summary (' + [string]$chk[1] + ')') }
        }
    }
    foreach ($chk in @(@('n_layer', $out.n_layer), @('n_expert', $out.n_expert),
                       @('n_expert_used', $out.n_expert_used), @('routed_tensors', $out.routed_tensors),
                       @('expert_bytes_total', $out.expert_bytes_total))) {
        $v = Get-JsonValue -Obj $ex -Name $chk[0]
        if (-not (Test-JsonNonNegativeInteger $v) -or [long]$v -ne [long]$chk[1]) {
            return @{ ok = $false; reason = ('derived expect ' + $chk[0] + '=' + [string]$v +
                                             ' disagrees with the plan summary (' + [string]$chk[1] + ')') }
        }
    }
    return $out
}

# ---------------------------------------------------------------------------------------------
# The derived profile is its own schema. It does NOT borrow the catalog's, because the catalog
# schema requires an upstream repository and revision and there is no honest value for either: a
# derived profile describes a file on this machine that no published measurement covers.
# ---------------------------------------------------------------------------------------------
function New-DerivedProfile {
    param($Parsed, [string] $OutputDir, [string[]] $Shas)
    $digest = ([string]$Parsed.inventory_sha256).Substring(0, $script:DERIVED_DIGEST_CHARS)
    $profileId = 'derived-' + [string]$Parsed.arch + '-' + $digest
    $minBudget = Get-CeilMib -Bytes ([long]$Parsed.n_expert * [long]$Parsed.slot_stride_max)
    $argv = @()
    foreach ($a in $script:DERIVED_ARGV_SKELETON) {
        if ([string]$a -ceq $script:DERIVED_ARGV_NCPUMOE_SLOT) { $argv += [string]$Parsed.n_layer }
        else { $argv += [string]$a }
    }
    $obj = [ordered]@{
        derived_profile_schema_version = [int]$script:DERIVED_PROFILE_SCHEMA_VERSION
        profile_id   = $profileId
        display_name = ([string]$Parsed.arch + ' via arch-template ' + [string]$Parsed.derived_from + ' (experimental)')
        routed_scope = [string]$Parsed.routed_scope
        identify     = [ordered]@{ arch = [string]$Parsed.arch; n_layer = [long]$Parsed.n_layer
                                   n_expert = [long]$Parsed.n_expert; n_expert_used = [long]$Parsed.n_expert_used }
        # step 4. Identical arithmetic to BUDGET_AUTOTUNE_SPEC v0.2 structural_min - the same
        # ceil(n_expert * slot_stride_max / MiB) the autotune computes in Get-BudgetAutoCandidate -
        # because it answers the same question: below it the engine cannot start (n_slots >= n_expert).
        min_budget_mb  = [long]$minBudget
        # P4 1: the derived row states the same thing the old one-axis 'disabled' stated, on the two
        # axes that replaced it. Nothing has been measured for a model derived on the spot, so the
        # evidence axis is 'unverified' and the activation axis is off by contract.
        prefetch_evidence   = 'unverified'
        prefetch_activation = 'off'
        prefetch       = $null
        # format_validated is true and it is not a courtesy: the derived path runs the SAME repack
        # verify and the same seven-item gate. performance_validated is false and stays false - no
        # published measurement covers a model derived on the spot.
        gates          = [ordered]@{ format_validated = $true; performance_validated = $false }
        reference_measurements = @()
        allowlist_bounds = [ordered]@{
            port      = [ordered]@{ min = [int]$script:DERIVED_BOUNDS_PORT.min;    max = [int]$script:DERIVED_BOUNDS_PORT.max }
            ctx       = [ordered]@{ min = [int]$script:DERIVED_BOUNDS_CTX.min;     max = [int]$script:DERIVED_BOUNDS_CTX.max }
            threads   = [ordered]@{ min = [int]$script:DERIVED_BOUNDS_THREADS.min; max = [int]$script:DERIVED_BOUNDS_THREADS.max }
            budget_mb = [ordered]@{ min = [long]$minBudget;                        max = [long]$script:DERIVED_BOUNDS_BUDGET_MAX }
            qd        = [ordered]@{ min = [int]$script:DERIVED_BOUNDS_QD.min;      max = [int]$script:DERIVED_BOUNDS_QD.max }
        }
        defaults = [ordered]@{ argv = @($argv); env = [ordered]@{} }
        derivation = [ordered]@{
            abi                = [string]$script:OPEN_ARCH_TEMPLATE_ABI
            template_id        = [string]$Parsed.template_id
            template_version   = [string]$Parsed.template_version
            derived_from       = [string]$Parsed.derived_from
            inventory_sha256   = [string]$Parsed.inventory_sha256
            derivation_digest  = [string]$digest
            slot_stride_max    = [long]$Parsed.slot_stride_max
            moe_layers         = [long]$Parsed.moe_layers
            routed_tensors     = [long]$Parsed.routed_tensors
            expert_bytes_total = [long]$Parsed.expert_bytes_total
            source_shards_sha256 = @($Shas)
            expect_path        = (Get-DerivedExpectPath -OutputDir $OutputDir)
            lock_id            = (Get-DerivedLockId -DerivedFrom ([string]$Parsed.derived_from))
        }
    }
    # Round-tripped through JSON so the object the rest of the launcher sees is shaped exactly like a
    # catalog profile at the ACCESSOR level (PSObject properties, JSON number and boolean types).
    # A hashtable would not be: Get-JsonValue reads PSObject.Properties, which on a Hashtable are the
    # hashtable's own members, not its entries.
    $json = $obj | ConvertTo-Json -Depth 12
    $r = ConvertFrom-JsonStrict -Text $json
    if (-not $r.ok) { Stop-Launcher 'fail_model_path' ('the derived profile could not be materialised - ' + $r.reason) }
    return $r.value
}

$script:DERIVED_PROFILE_KEYS = @(
    'derived_profile_schema_version', 'profile_id', 'display_name', 'routed_scope', 'identify',
    'min_budget_mb', 'prefetch_evidence', 'prefetch_activation', 'prefetch', 'gates',
    'reference_measurements', 'allowlist_bounds', 'defaults', 'derivation')
# Catalog-only keys. Their presence is not a harmless extra: hf_repo / hf_revision would assert an
# upstream identity nobody established, and expect_file / expect_sha256 would point the seven-item
# gate at the bundle expects directory instead of the derived expect in the output directory.
$script:DERIVED_PROFILE_FORBIDDEN_KEYS = @('hf_repo', 'hf_revision', 'expect_file', 'expect_sha256',
                                           'source_shards_sha256')

function Test-DerivedProfile {
    param($Profile)
    if ($null -eq $Profile) { return @{ ok = $false; reason = 'no derived profile' } }
    foreach ($k in $script:DERIVED_PROFILE_FORBIDDEN_KEYS) {
        if (Test-JsonHas -Obj $Profile -Name $k) {
            return @{ ok = $false; reason = ("a derived profile must not carry the catalog key '" + $k + "'") }
        }
    }
    foreach ($k in (Get-JsonKeys -Obj $Profile)) {
        if ($script:DERIVED_PROFILE_KEYS -notcontains $k) {
            return @{ ok = $false; reason = ("unknown key '" + $k + "' in the derived profile (deny-by-default)") }
        }
    }
    foreach ($k in $script:DERIVED_PROFILE_KEYS) {
        if (-not (Test-JsonHas -Obj $Profile -Name $k)) {
            return @{ ok = $false; reason = ("required key '" + $k + "' missing in the derived profile") }
        }
    }
    $sv = Get-JsonValue -Obj $Profile -Name 'derived_profile_schema_version'
    if (-not (Test-JsonNonNegativeInteger $sv) -or [long]$sv -ne [long]$script:DERIVED_PROFILE_SCHEMA_VERSION) {
        return @{ ok = $false; reason = 'derived_profile_schema_version is not an exact match' }
    }
    $d = Get-JsonValue -Obj $Profile -Name 'derivation'
    $arch = [string](Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'identify') -Name 'arch')
    if (-not (Test-PathSafeToken -Value $arch -Pattern $script:ARCH_TOKEN_REGEX)) {
        return @{ ok = $false; reason = ('identify.arch is not a path-safe token: ' + $arch) }
    }
    $inv = [string](Get-JsonValue -Obj $d -Name 'inventory_sha256')
    if (-not (Test-Sha256Hex $inv) -or $inv -cne $inv.ToLowerInvariant()) {
        return @{ ok = $false; reason = 'derivation.inventory_sha256 is not a lowercase 64 hex digest' }
    }
    $pid0 = [string](Get-JsonValue -Obj $Profile -Name 'profile_id')
    if (-not (Test-PathSafeToken -Value $pid0 -Pattern $script:DERIVED_PROFILE_ID_REGEX)) {
        return @{ ok = $false; reason = ('profile_id is not a path-safe derived id: ' + $pid0) }
    }
    $expectId = 'derived-' + $arch + '-' + $inv.Substring(0, $script:DERIVED_DIGEST_CHARS)
    if ($pid0 -cne $expectId) {
        return @{ ok = $false; reason = ('profile_id is not the deterministic id for this derivation (expected ' + $expectId + ')') }
    }
    if (([string](Get-JsonValue -Obj $d -Name 'abi')) -cne [string]$script:OPEN_ARCH_TEMPLATE_ABI) {
        return @{ ok = $false; reason = 'derivation.abi does not match this launcher OPEN_ARCH_TEMPLATE_ABI' }
    }
    if (([string](Get-JsonValue -Obj $d -Name 'derived_from')) -cne
        ([string](Get-JsonValue -Obj $d -Name 'template_id') + '@' + [string](Get-JsonValue -Obj $d -Name 'template_version'))) {
        return @{ ok = $false; reason = 'derivation.derived_from is not template_id@template_version' }
    }
    if (([string](Get-JsonValue -Obj $Profile -Name 'prefetch_evidence')) -cne 'unverified' -or
        ([string](Get-JsonValue -Obj $Profile -Name 'prefetch_activation')) -cne 'off') {
        return @{ ok = $false; reason = 'a derived profile is prefetch_evidence=unverified / prefetch_activation=off by contract' }
    }
    if ($null -ne (Get-JsonValue -Obj $Profile -Name 'prefetch')) {
        return @{ ok = $false; reason = 'a derived profile carries no prefetch tuple' }
    }
    if (Test-JsonBooleanTrue (Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'gates') -Name 'performance_validated')) {
        return @{ ok = $false; reason = 'a derived profile can never claim performance_validated' }
    }
    $nExp   = [long](Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'identify') -Name 'n_expert')
    $nLayer = [long](Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'identify') -Name 'n_layer')
    $stride = [long](Get-JsonValue -Obj $d -Name 'slot_stride_max')
    $layers = [long](Get-JsonValue -Obj $d -Name 'moe_layers')
    $minB   = [long](Get-JsonValue -Obj $Profile -Name 'min_budget_mb')
    if ($nExp -le 0 -or $stride -le 0 -or $layers -le 0) {
        return @{ ok = $false; reason = 'the slot geometry is not positive' }
    }
    if ($minB -ne (Get-CeilMib -Bytes ($nExp * $stride))) {
        return @{ ok = $false; reason = 'min_budget_mb is not ceil(n_expert * slot_stride_max / MiB)' }
    }
    # BUDGET_AUTOTUNE_SPEC v0.2 section 4-1 item 6: a minimum above full slot residency would make
    # the autotune structurally impossible. For a derived profile this holds by construction
    # (model_cap = structural_min * layers, layers >= 1) - asserted rather than assumed.
    if ($minB -gt (Get-CeilMib -Bytes ($nExp * $layers * $stride))) {
        return @{ ok = $false; reason = 'min_budget_mb exceeds the full slot residency (model_cap)' }
    }
    $bb = Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'allowlist_bounds') -Name 'budget_mb'
    if ([long](Get-JsonValue -Obj $bb -Name 'min') -ne $minB) {
        return @{ ok = $false; reason = 'allowlist_bounds.budget_mb.min is not the derived minimum' }
    }
    $argv = @(Get-JsonArray -Obj (Get-JsonValue -Obj $Profile -Name 'defaults') -Name 'argv')
    $ncm = Get-ArgvValue -Argv $argv -Flag '--n-cpu-moe'
    if ([string]$ncm -cne [string]$nLayer) {
        return @{ ok = $false; reason = 'defaults.argv --n-cpu-moe is not the derived layer count' }
    }
    if (-not (Test-LoopbackAddress -Address ([string](Get-ArgvValue -Argv $argv -Flag '--host')))) {
        return @{ ok = $false; reason = 'defaults.argv binds a non-loopback host' }
    }
    return @{ ok = $true }
}

# ---------------------------------------------------------------------------------------------
# Steps 2..5. Step 6 (the resource gate and the confirmation) stays in the main flow, where the
# ordinary catalog path performs it too - one confirmation point, not two.
# ---------------------------------------------------------------------------------------------
function Invoke-DerivePlan {
    param($Catalog, [string] $Root, [string] $ModelPath, [string] $OutputDir, $ModelSet, [string[]] $Shas)
    if ($script:PinMatchedLatch) {
        # A matched pin closed this door (OPEN_ARCH_DESIGN section 0: a pinned model that then fails
        # a catalog / expect / seal check is a hard failure, never an experimental downgrade).
        Stop-Launcher 'fail_model_path' 'internal: the arch-template path is closed after a source pin matched'
    }
    Write-Line ''
    Write-Line '=== derive-plan (EXPERIMENTAL arch-template) ==='
    Write-Line 'This model is not in the catalog. Deriving its routed inventory from the frozen'
    Write-Line 'architecture template. Nothing is written until you confirm the repack.'
    Write-Line '[derive] steps 2-3/6: closing the inventory and querying the output volume alignment...'
    $plan = Invoke-Repacker -Catalog $Catalog -Root $Root -Profile $null -ModelPath $ModelPath `
                -OutputDir $OutputDir -PlanOnly $true -ArchTemplate $true -FailStatus 'fail_model_path'
    $parsed = ConvertFrom-TemplatePlanText -Text $plan.text
    if (-not $parsed.ok) {
        Stop-Launcher 'fail_model_path' ('the arch-template plan could not be read - ' + [string]$parsed.reason)
    }
    Write-Line ('[derive] step 4/6: min budget = ceil({0} experts x {1} B / MiB)' -f $parsed.n_expert, $parsed.slot_stride_max)
    Write-Line '[derive] step 5/6: completing and validating the derived profile...'
    $profile = New-DerivedProfile -Parsed $parsed -OutputDir $OutputDir -Shas $Shas
    $v = Test-DerivedProfile -Profile $profile
    if (-not $v.ok) { Stop-Launcher 'fail_model_path' ('the derived profile failed its own validator - ' + [string]$v.reason) }

    $profileId = [string](Get-JsonValue -Obj $profile -Name 'profile_id')
    Write-Diag -Kind 'DERIVE_PLAN' -Data @{
        profile_id = $profileId; derived_from = $parsed.derived_from
        inventory_sha256 = $parsed.inventory_sha256; routed_scope = $parsed.routed_scope
        arch = $parsed.arch; n_layer = $parsed.n_layer; n_expert = $parsed.n_expert
        moe_layers = $parsed.moe_layers; routed_tensors = $parsed.routed_tensors
        slot_stride_max = $parsed.slot_stride_max; expert_bytes_total = $parsed.expert_bytes_total
        min_budget_mb = [long](Get-JsonValue -Obj $profile -Name 'min_budget_mb')
        source_shards = @($Shas).Count; abi = $script:OPEN_ARCH_TEMPLATE_ABI }
    Write-Line ('[derive] profile {0} ({1}, routed scope {2}, {3} routed tensors)' -f
        $profileId, $parsed.derived_from, $parsed.routed_scope, $parsed.routed_tensors)
    return @{ profile = $profile; parsed = $parsed; plan_text = [string]$plan.text
              min_budget_mb = [long](Get-JsonValue -Obj $profile -Name 'min_budget_mb')
              expected_bytes = [long]$parsed.expert_bytes_total
              lock_id = (Get-DerivedLockId -DerivedFrom ([string]$parsed.derived_from)) }
}

# ---------------------------------------------------------------------------------------------
# LS OA-1 surface axes. Three questions that a single badge would blur:
#   copy integrity      is about BYTES and is answered by the repack verify + the seven-item gate;
#   inventory authority is about WHICH tensors were selected, and by whom;
#   serving validation  is a property of the PROFILE (is this configuration a published, measured
#                       one?), not of this run's config - the performance gate line above it already
#                       carries the run-level answer, and the two are deliberately separate.
# ---------------------------------------------------------------------------------------------
function Get-SurfaceAxes {
    param([string] $Kind, $Profile, [bool] $CopyVerified)
    $copy = $script:AXIS_COPY_PENDING
    if ($CopyVerified) { $copy = $script:AXIS_COPY_PASS }
    $note = $null
    if ($Kind -ceq 'template') {
        $inventory = $script:AXIS_INVENTORY_TEMPLATE
        $serving   = $script:AXIS_SERVING_UNVALIDATED
        $note      = $script:TEMPLATE_COPY_SENTENCE
    } elseif ($Kind -ceq 'unpinned') {
        $inventory = $script:AXIS_INVENTORY_UNPINNED
        $serving   = $script:AXIS_SERVING_UNVALIDATED
        $note      = $script:UNPINNED_NOTE
    } elseif ($Kind -ceq 'mismatch') {
        # P4 2.5: serving validation is UNCONDITIONALLY unvalidated here - it does not consult
        # gates.performance_validated at all, because that gate describes the bytes the catalog
        # measured and this file has just been proven not to be them.
        $inventory = $script:AXIS_INVENTORY_MISMATCH
        $serving   = $script:AXIS_SERVING_UNVALIDATED
        $note      = $script:MISMATCH_NOTE
    } else {
        $inventory = $script:AXIS_INVENTORY_PIN
        $serving   = $script:AXIS_SERVING_UNVALIDATED
        if (Test-JsonBooleanTrue (Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'gates') -Name 'performance_validated')) {
            $serving = $script:AXIS_SERVING_VALIDATED
        }
    }
    return @{ kind = $Kind; copy_integrity = $copy; inventory_authority = $inventory
              serving_validation = $serving; note = $note }
}

# endregion

# ============================================================================
# region 8. PREFLIGHT (LS 4, RS 3/5)
# ============================================================================

function Get-MemStatus {
    $m = New-Object 'MoeLauncher.Native+MEMORYSTATUSEX'
    $m.dwLength = [System.Runtime.InteropServices.Marshal]::SizeOf($m)
    if (-not [MoeLauncher.Native]::GlobalMemoryStatusEx([ref]$m)) { return @{ ok = $false } }
    return @{ ok = $true
              total_phys_mb = [long]($m.ullTotalPhys / 1MB)
              avail_phys_mb = [long]($m.ullAvailPhys / 1MB) }
}

function Get-VolumeFreeMb {
    param([string] $Path)
    try {
        $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
        $di = New-Object System.IO.DriveInfo($root)
        return @{ ok = $true; free_mb = [long]($di.AvailableFreeSpace / 1MB); total_mb = [long]($di.TotalSize / 1MB); root = $root }
    } catch {
        return @{ ok = $false; reason = $_.Exception.Message }
    }
}

function Get-VramInfo {
    # LS 4: best-effort display + diagnostic log only. Never a hard-stop gate.
    try {
        $smi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
        if ($smi) {
            $out = & $smi.Source '--query-gpu=memory.total,memory.free' '--format=csv,noheader,nounits' 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $parts = ([string](@($out)[0])).Split(',')
                if ($parts.Count -ge 2) {
                    return @{ ok = $true; source = 'nvidia-smi'; total_mb = [long]$parts[0].Trim(); free_mb = [long]$parts[1].Trim() }
                }
            }
        }
    } catch { }
    try {
        $c = @(Get-CimInstance Win32_VideoController -ErrorAction Stop)[0]
        if ($c -and $c.AdapterRAM) {
            return @{ ok = $true; source = 'Win32_VideoController'; total_mb = [long]([uint32]$c.AdapterRAM / 1MB); free_mb = $null }
        }
    } catch { }
    return @{ ok = $false }
}

function Get-ExpectedRepackBytes {
    param([string] $ExpectPath)
    $r = Read-JsonFileStrict -Path $ExpectPath
    if (-not $r.ok) { return $null }
    $v = Get-JsonValue -Obj $r.value -Name 'expert_bytes_total'
    if (-not (Test-JsonNonNegativeInteger $v)) { return $null }
    return [long]$v
}

# LS 1-9 structure: required = budget + non-cache fixed term + safety headroom.
# The fixed terms are [UNMEASURED-TODO], and inventing a number for them is forbidden - but the
# contract still has one half that does not depend on them at all (see the budget-only branch
# below), so an unset term degrades the verdict to 'unmeasured' only where it actually decides
# anything. 'unmeasured' is non-terminal and shown honestly; it is never an approval.
function Test-RamVerdict {
    param([long] $BudgetMb, [long] $CtxTokens = 0)
    $mem = Get-MemStatus
    if (-not $mem.ok) { return @{ verdict = 'unmeasured'; reason = 'GlobalMemoryStatusEx failed' } }
    if ($null -eq $script:RAM_DENSE_RESIDENT_MB -or $null -eq $script:RAM_KV_MB_PER_1K_CTX -or
        $null -eq $script:RAM_SERVER_OVERHEAD_MB -or $null -eq $script:RAM_HEADROOM_MB) {
        # Codex r1 F1. Every missing term is a NON-NEGATIVE addend, so required >= budget holds
        # whatever they turn out to be: a budget that already exceeds available RAM is a CONFIRMED
        # shortfall, not an unknown one, and admission decides it here instead of waving the run
        # through with an [unmeasured] line (measured: auto 12,288 MB proceeded at 316 MB available).
        # This is NOT "raw budget alone decides": the opposite direction is never approved from here
        # - a budget under available still falls through to 'unmeasured'.
        if ($BudgetMb -gt $mem.avail_phys_mb) {
            return @{ verdict = 'insufficient'; basis = 'budget_only'
                      reason = 'selected budget alone exceeds available RAM (the unmeasured fixed terms can only add to it)'
                      required_mb = $BudgetMb; kv_mb = $null; avail_mb = $mem.avail_phys_mb
                      total_mb = $mem.total_phys_mb; budget_mb = $BudgetMb; ctx = $CtxTokens }
        }
        return @{ verdict = 'unmeasured'; reason = 'dense/KV/server/headroom terms not yet measured (LS 9 item 1)';
                  avail_mb = $mem.avail_phys_mb; total_mb = $mem.total_phys_mb; budget_mb = $BudgetMb; ctx = $CtxTokens }
    }
    # LS 1-9 structure: budget + non-cache fixed term (dense resident + KV(ctx) + server) + headroom.
    $kv = [long]([math]::Ceiling($CtxTokens / 1024.0) * [double]$script:RAM_KV_MB_PER_1K_CTX)
    $required = $BudgetMb + [long]$script:RAM_DENSE_RESIDENT_MB + $kv +
                [long]$script:RAM_SERVER_OVERHEAD_MB + [long]$script:RAM_HEADROOM_MB
    if ($required -gt $mem.avail_phys_mb) {
        return @{ verdict = 'insufficient'; basis = 'full_formula'
                  reason = 'budget + dense + KV(ctx) + server + headroom exceeds available RAM'
                  required_mb = $required; kv_mb = $kv; avail_mb = $mem.avail_phys_mb }
    }
    return @{ verdict = 'ok'; basis = 'full_formula'; required_mb = $required; kv_mb = $kv; avail_mb = $mem.avail_phys_mb }
}

# P4 2.5 (b) / r3 C-1: the bytes a stale-artifact replacement will hand back to the volume.
# Only files that exist RIGHT NOW are counted, and an unreadable entry contributes nothing, so the
# number can only ever understate what the deletion frees. It is a claim about the current state of
# the disk, not a promise - the deletion still happens after the confirmation and nowhere else.
function Get-StaleArtifactBytes {
    param([string] $OutputDir)
    $sum = [long]0
    foreach ($n in $script:PARTIAL_DELETE_SET) {
        $p = Join-Path $OutputDir $n
        try {
            $fi = New-Object System.IO.FileInfo($p); $fi.Refresh()
            if ($fi.Exists) { $sum = $sum + [long]$fi.Length }
        } catch { }
    }
    return $sum
}

# r4: the MB figure the disk gate consumes. A bare [long](bytes / 1MB) ROUNDS in PowerShell 5.1
# (524289 bytes -> 1), which would OVERSTATE the reclaim on a fractional-MB tail - the one
# direction this figure must never err. An explicit Floor keeps the understatement claim of
# Get-StaleArtifactBytes true in MB as well as in bytes, and matches the surrounding free_mb /
# needMb arithmetic.
function Get-StaleArtifactReclaimMb {
    param([string] $OutputDir)
    return [long][math]::Floor((Get-StaleArtifactBytes -OutputDir $OutputDir) / 1MB)
}

# -ReclaimableMb (r3 C-1) is the conditional headroom above. It is passed ONLY on the mismatch
# stale path; every other caller leaves it at 0, where the arithmetic and the printed lines are
# byte-identical to what they were before it existed.
function Invoke-Preflight {
    param([string] $OutputDir, [string] $ExpectPath, [long] $BudgetMb, [bool] $NeedsRepack,
          [long] $CtxTokens = 0, [long] $ExpectedBytes = -1, [long] $ReclaimableMb = 0)

    $vol = Get-VolumeFreeMb -Path $OutputDir
    if (-not $vol.ok) { Stop-Launcher 'fail_resource' ("output volume query failed: " + $vol.reason) }
    # LS OA-1: on the derived path the expect file does not exist yet on a first run - the repacker
    # writes it only after the confirmation this preflight gates. The size therefore comes from the
    # plan that has already been read, and it is the SAME number (expert_bytes_total) the expect
    # carries; the file is still read on every later run, where it does exist.
    $needBytes = $null
    if ($ExpectedBytes -ge 0) { $needBytes = [long]$ExpectedBytes }
    else { $needBytes = Get-ExpectedRepackBytes -ExpectPath $ExpectPath }
    $needMb = 0
    if ($null -ne $needBytes) { $needMb = [long]($needBytes / 1MB) }

    Write-Line ''
    Write-Line '=== preflight ==='
    Write-Line ('  disk   : volume {0} free {1} MB' -f $vol.root, $vol.free_mb)
    if ($NeedsRepack) {
        Write-Line ('           repack artifact ~{0} MB' -f $needMb)
        # r3 C-1: when this repack REPLACES artifacts that are about to be deleted, the volume the
        # new one lands on is the current free space plus what the deletion returns. Gating on the
        # pre-deletion figure alone would refuse a replacement that fits perfectly well - a false
        # resource shortage, and on a large model the ordinary case rather than an edge one.
        $freeForRepack = $vol.free_mb
        if ($ReclaimableMb -gt 0) {
            $freeForRepack = $vol.free_mb + $ReclaimableMb
            Write-Line ('           reclaimed by replacing the stale artifacts ~{0} MB (deleted only if you approve)' -f $ReclaimableMb)
            Write-Line ('           free after that reclaim ~{0} MB' -f $freeForRepack)
        }
        $residual = $freeForRepack - $needMb
        Write-Line ('           residual after repack ~{0} MB' -f $residual)
        if ($null -eq $script:DISK_POST_RESERVE_MB) {
            Write-Line '           reserve policy: [unmeasured] (LS 9 item 1) - displayed, not gated'
        }
        if ($needMb -gt $freeForRepack) {
            $why = ("disk preflight hard stop: need " + $needMb + " MB, free " + $vol.free_mb + " MB")
            if ($ReclaimableMb -gt 0) { $why = $why + " (+" + $ReclaimableMb + " MB reclaimable by replacing the stale artifacts)" }
            Stop-Launcher 'fail_resource' $why
        }
    }

    $ram = Test-RamVerdict -BudgetMb $BudgetMb -CtxTokens $CtxTokens
    if ($ram.verdict -eq 'insufficient') {
        Write-Line ('  ram    : required {0} MB > available {1} MB (ctx {2}, basis {3})' -f $ram.required_mb, $ram.avail_mb, $CtxTokens, $ram.basis)
        Stop-Launcher 'fail_resource' ('RAM preflight hard stop: ' + $ram.reason)
    } elseif ($ram.verdict -eq 'unmeasured') {
        $mem = Get-MemStatus
        $availTxt = 'n/a'
        if ($mem.ok) { $availTxt = [string]$mem.avail_phys_mb }
        Write-Line ('  ram    : budget {0} MB, ctx {1} | verdict [unmeasured] ({2}) | available {3} MB' -f $BudgetMb, $CtxTokens, $ram.reason, $availTxt)
    } else {
        Write-Line ('  ram    : required {0} MB (incl. KV {1} MB @ ctx {2}) <= available {3} MB' -f $ram.required_mb, $ram.kv_mb, $CtxTokens, $ram.avail_mb)
    }

    $vram = Get-VramInfo
    if ($vram.ok) {
        $freeTxt = 'n/a'
        if ($null -ne $vram.free_mb) { $freeTxt = [string]$vram.free_mb }
        Write-Line ('  vram   : total {0} MB free {1} MB (source {2}) - display only, not a gate (LS 4)' -f $vram.total_mb, $freeTxt, $vram.source)
    } else {
        Write-Line '  vram   : not detected - display only, not a gate (LS 4)'
    }

    Write-Diag -Kind 'PREFLIGHT' -Data @{ disk = $vol; need_mb = $needMb; ram = $ram; vram = $vram; budget_mb = $BudgetMb }
    return @{ ram = $ram; disk = $vol; vram = $vram }
}

# endregion

# ============================================================================
# region 8b. RAM BUDGET AUTOTUNE (BUDGET_AUTOTUNE_SPEC.md v0.2)
#   With no explicit value the expert-cache budget is no longer one catalog number for every
#   machine (a 16 GB laptop and a 64 GB desktop both got 8192): it is sized per boot from the
#   INSTALLED RAM and the repack slot geometry. Three properties hold the rest together:
#     - explicit wins. CLI, stored preset and interactive custom all arrive in the same override
#       map, and any of them outranks the autotune (v0.2 section 2).
#     - this is a DEFAULT SELECTOR, not a safety gate. Admission stays with the LS 1-9 RAM contract
#       in Test-RamVerdict, which receives the selected value and hard-stops on its own - today for
#       the half of the contract that is decided (budget > available), with the unmeasured fixed
#       terms still only displayed (v0.2 section 1, "enforced scope of this build"). Available RAM
#       is therefore never an input to the arithmetic below, and nothing here ever lowers a value
#       dynamically to make it fit (v0.2 section 1).
#     - nothing is silent. Every rebuild writes a BUDGET_AUTOTUNE record before the caller's
#       EFFECTIVE record, and the decision is echoed on one banner line (v0.2 section 3).
# ============================================================================

# v0.2 section 1. RESERVE is a static selector constant calibrated on INSTALLED RAM
# (32768 - 20480 = 12288, the (a)-axis verdict in reviews/codex_budget_final_r2.md). It is not an
# availability guarantee and must not be re-read as one. FLOOR and LADDER_MAX bound the ladder to
# rungs that have actually been measured: a larger machine may override explicitly, but auto never
# walks past the last measured rung on its own.
$script:BUDGET_RESERVE_MIB    = 20480
$script:BUDGET_FLOOR_MIB      = 4096
$script:BUDGET_LADDER_MAX_MIB = 12288

# Banner de-duplication. Build-EffectiveConfig runs on every rebuild (preset bind, custom edit,
# pre-spawn re-check) and the answer is normally identical, so the line is printed when the decision
# text CHANGES rather than once per rebuild - a custom edit that changes the answer re-announces it.
$script:BudgetBannerLast = $null

# INSTALLED RAM, deliberately not the OS-visible total. GlobalMemoryStatusEx reports what the OS can
# address (31,900 MiB on the 32 GiB reference box, firmware reservations already removed), which
# would put the reserve arithmetic 868 MiB under the calibration point and make that same box pick
# 11,420 instead of 12,288. GetPhysicallyInstalledSystemMemory reads the SMBIOS total instead. It is
# allowed to fail (malformed SMBIOS tables); that is a probe failure with a reason, never a licence
# to substitute the visible number.
function Get-InstalledMemoryMib {
    $kb = [uint64]0
    try {
        if (-not [MoeLauncher.Native]::GetPhysicallyInstalledSystemMemory([ref]$kb)) {
            return @{ ok = $false; method = 'GetPhysicallyInstalledSystemMemory'
                      reason = ('GetPhysicallyInstalledSystemMemory failed (GetLastError=' +
                                [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() + ')') }
        }
    } catch {
        return @{ ok = $false; method = 'GetPhysicallyInstalledSystemMemory'
                  reason = ('GetPhysicallyInstalledSystemMemory threw: ' + $_.Exception.Message) }
    }
    if ([uint64]$kb -eq [uint64]0) {
        return @{ ok = $false; method = 'GetPhysicallyInstalledSystemMemory'
                  reason = 'GetPhysicallyInstalledSystemMemory reported 0 KB' }
    }
    return @{ ok = $true; method = 'GetPhysicallyInstalledSystemMemory'
              installed_mib = [long][math]::Floor([double][uint64]$kb / 1024.0) }
}

# v0.2 section 1: the model axis is SLOT geometry, not the size of experts.bin. Slots are a uniform
# budget / slot_stride_max division, so full residency costs n_expert * (MoE layers) * stride, which
# for qwen122 is LARGER than experts.bin itself (70.03 GiB of slots vs 65.39 GiB of payload). All
# three numbers come from the repack manifest the verify gate has already bound to the cache key,
# read through the strict reader region 9 already uses for the same file. Anything missing or
# malformed leaves the geometry unavailable and the caller falls back - it never guesses a shape.
function Get-BudgetSlotGeometry {
    param([string] $OutputDir)
    $path = Join-Path $OutputDir 'manifest.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return @{ ok = $false; reason = 'manifest.json not present' }
    }
    $r = Read-JsonFileStrict -Path $path
    if (-not $r.ok) { return @{ ok = $false; reason = ('manifest.json unreadable - ' + $r.reason) } }
    $layout = Get-JsonValue -Obj $r.value -Name 'layout'
    if ($null -eq $layout) { return @{ ok = $false; reason = 'manifest.json has no layout object' } }
    $ne = Get-JsonValue -Obj (Get-JsonValue -Obj $r.value -Name 'model') -Name 'n_expert'
    if ((-not (Test-JsonNonNegativeInteger $ne)) -or [long]$ne -le 0) {
        return @{ ok = $false; reason = 'manifest model.n_expert is missing or not a positive integer' }
    }
    $stride = Get-JsonValue -Obj $layout -Name 'slot_stride_max'
    if ((-not (Test-JsonNonNegativeInteger $stride)) -or [long]$stride -le 0) {
        return @{ ok = $false; reason = 'manifest layout.slot_stride_max is missing or not a positive integer' }
    }
    # layout.layers holds one entry per MoE layer (the repacker writes exactly the routed layers),
    # so its length is the layer factor of full slot residency.
    $layers = [long](@(Get-JsonArray -Obj $layout -Name 'layers').Count)
    if ($layers -le 0) { return @{ ok = $false; reason = 'manifest layout.layers is absent or empty' } }
    return @{ ok = $true; n_expert = [long]$ne; slot_stride_max = [long]$stride; layers = $layers }
}

# Integer ceiling in MiB. PowerShell's "/" on two longs yields a double and a [long] cast rounds it,
# so the conversion is spelled out rather than left to the cast.
function Get-CeilMib {
    param([long] $Bytes)
    return [long][math]::Ceiling([double]$Bytes / 1048576.0)
}

# v0.2 section 1, pure arithmetic - no I/O, no clock, nothing global but the three constants - so
# the selftest drives every boundary with synthetic machines and model geometries:
#   machine_cap    = clamp(installed - RESERVE, FLOOR, LADDER_MAX)
#   structural_min = ceil(n_expert * stride / MiB)            engine start condition n_slots >= n_expert
#   supported_min  = max(structural_min, profile.min_budget_mb)
#   model_cap      = ceil(n_expert * layers * stride / MiB)   full slot residency
#   candidate      = min(machine_cap, model_cap)              below supported_min: auto is impossible
# A tie gives the limiting axis to machine_cap: at equal values the machine is the constraint that
# survives a change of model.
function Get-BudgetAutoCandidate {
    param([long] $InstalledMib, [long] $NExpert, [long] $Layers, [long] $SlotStrideMax,
          [long] $ProfileMinBudgetMb)
    $machineCap = $InstalledMib - [long]$script:BUDGET_RESERVE_MIB
    if ($machineCap -lt [long]$script:BUDGET_FLOOR_MIB)      { $machineCap = [long]$script:BUDGET_FLOOR_MIB }
    if ($machineCap -gt [long]$script:BUDGET_LADDER_MAX_MIB) { $machineCap = [long]$script:BUDGET_LADDER_MAX_MIB }
    $structuralMin = Get-CeilMib -Bytes ($NExpert * $SlotStrideMax)
    $supportedMin = $structuralMin
    if ($ProfileMinBudgetMb -gt $supportedMin) { $supportedMin = $ProfileMinBudgetMb }
    $modelCap = Get-CeilMib -Bytes ($NExpert * $Layers * $SlotStrideMax)
    $candidate = $machineCap
    $axis = 'machine_cap'
    if ($modelCap -lt $machineCap) { $candidate = $modelCap; $axis = 'model_cap' }
    return @{ machine_cap = $machineCap; model_cap = $modelCap
              structural_min = $structuralMin; supported_min = $supportedMin
              candidate = $candidate; limiting_axis = $axis; ok = ($candidate -ge $supportedMin) }
}

# v0.2 sections 2-3. One decision point for the whole launcher: the priority ladder, the three
# fallbacks, the impossible-auto branch, the diagnostic record and the banner all live here. The
# caller passes the probe results in rather than having them read behind its back - the same shape
# as Resolve-EffectivePrefetch taking -ProbeOk - which is what lets the selftest drive every row.
function Resolve-BudgetAutotune {
    param($Profile, $Overrides, $Installed, $Geometry, $Mem, [bool] $ReproMode, [bool] $PerfCustom,
          [bool] $WarmPath = $false)

    $profileMin = [long](Get-JsonValue -Obj $Profile -Name 'min_budget_mb')
    $calc = $null
    if ($Installed.ok -and $Geometry.ok) {
        $calc = Get-BudgetAutoCandidate -InstalledMib ([long]$Installed.installed_mib) `
                    -NExpert ([long]$Geometry.n_expert) -Layers ([long]$Geometry.layers) `
                    -SlotStrideMax ([long]$Geometry.slot_stride_max) -ProfileMinBudgetMb $profileMin
    }

    # The priority ladder (v0.2 section 2). MOE_DIRECT_BUDGET_MB is NOT an input channel - it is the
    # launcher's own output wire to the child (:3539) - so an ambient value cannot reach this
    # decision. Every fallback lands on the PROFILE minimum, never on a global 8192.
    $budget = $profileMin
    if ($null -ne $Overrides -and $Overrides.ContainsKey('budget_mb')) {
        $source = 'explicit'
        $budget = [long]$Overrides['budget_mb']
        $reason = 'explicit budget (CLI / stored preset / interactive custom) outranks the autotune'
    } elseif ($ReproMode) {
        $source = 'repro_fallback'
        $reason = '-Repro forbids the autotune; profile min_budget_mb is the reproducible fallback'
    } elseif (-not $Installed.ok) {
        $source = 'probe_failed_fallback'
        $reason = ('autotune off, installed-RAM probe failed: ' + [string]$Installed.reason)
    } elseif (-not $Geometry.ok) {
        $source = 'geometry_unavailable_fallback'
        $reason = ('autotune off, slot geometry unavailable: ' + [string]$Geometry.reason)
    } elseif (-not $calc.ok) {
        # Not a degraded path: below the supported minimum the engine cannot start at all
        # (n_slots >= n_expert), so silently serving a smaller number would be a lie.
        $source = 'fail_resource'
        $budget = 0
        $reason = ('candidate ' + $calc.candidate + ' MB is below the supported minimum ' +
                   $calc.supported_min + ' MB (structural_min ' + $calc.structural_min +
                   ' MB, profile min_budget_mb ' + $profileMin + ' MB; machine_cap ' + $calc.machine_cap +
                   ' MB from installed ' + $Installed.installed_mib + ' MiB, model_cap ' + $calc.model_cap + ' MB)')
    } else {
        $source = 'auto'
        $budget = [long]$calc.candidate
        $reason = ('installed ' + $Installed.installed_mib + ' MiB - reserve ' + $script:BUDGET_RESERVE_MIB +
                   ' -> machine_cap ' + $calc.machine_cap + ' MB, model_cap ' + $calc.model_cap +
                   ' MB, supported_min ' + $calc.supported_min + ' MB; limiting axis ' + $calc.limiting_axis)
    }

    # v0.2 section 2 provenance: an auto value that is not the catalog's measured budget is not the
    # measured configuration any more, so it demotes the performance axis exactly like a custom edit
    # does. An explicit value already demotes through the existing custom-provenance rule.
    $unmeasured = (($source -ceq 'auto') -and ($budget -ne $profileMin))
    # Codex r1 F2 + r2 F2-b. performance_identity is this record's claim that the run is STILL on
    # the catalog's measured operating point, so it has to answer the same question the EFFECTIVE
    # record answers with performance_gate. That gate demotes on the WHOLE performance provenance -
    # Test-CustomProvenance over the same overrides map, which counts far more than the budget key.
    # That function decides in three tiers: the PERF_NEUTRAL list ('warmstart') is exempt outright,
    # a key whose final VALUE proves the run never left the measured condition is exempt for that
    # value only, and everything else - port, ctx, threads, qd, autosave - counts on presence alone.
    # The value-based tier is deliberately not re-listed here: that function carries the current set
    # and the proof each member owes. A source-only formula answered 'true' for a valid '-Repro -Port <n>'
    # run while EFFECTIVE said custom/[unmeasured] (r2 F2-b). So the caller passes that one
    # provenance answer in as -PerfCustom (Build-EffectiveConfig computes it from the same
    # overrides map the EFFECTIVE writer uses), a fail_resource row still never claims the
    # identity (it serves no budget at all), and 'explicit' no longer needs a clause of its own -
    # a budget_mb override is never performance-neutral, so it already arrives as PerfCustom.
    # UX 1-4 (Codex build r1 M4) adds the WARMUP dimension to that same question. Since v0.2.3 the
    # launcher warms up by default while every published number was measured cold, so a warm run is
    # not the measured operating point no matter how the budget landed - and the status screen
    # already says so. Without this clause the record would answer 'identity true' next to a screen
    # reading [unmeasured], which is the exact disagreement the r1 F2 repair exists to prevent.
    # It arrives as a parameter rather than being read here because the value is a decision of
    # Build-EffectiveConfig (the bench force can still turn the warmup off after the override layers).
    $identity = ((-not $PerfCustom) -and (-not $WarmPath) -and ($source -cne 'fail_resource') -and ($budget -eq $profileMin))

    $rec = [ordered]@{
        source                = $source
        reason                = $reason
        budget_mb             = $(if ($source -ceq 'fail_resource') { $null } else { [long]$budget })
        profile_min_budget_mb = $profileMin
        repro                 = [bool]$ReproMode
        # UX 1-4 (Codex build r1 M4): the warmup dimension that fed performance_identity above, so
        # the record explains its own verdict instead of leaving a reader to guess why.
        warm_path             = [bool]$WarmPath
        explicit_override     = ($null -ne $Overrides -and $Overrides.ContainsKey('budget_mb'))
        probe_method          = [string]$Installed.method
        probe_ok              = [bool]$Installed.ok
        probe_error           = $(if ($Installed.ok) { $null } else { [string]$Installed.reason })
        installed_mib         = $(if ($Installed.ok) { [long]$Installed.installed_mib } else { $null })
        visible_mib           = $(if ($Mem -and $Mem.ok) { [long]$Mem.total_phys_mb } else { $null })
        available_mib         = $(if ($Mem -and $Mem.ok) { [long]$Mem.avail_phys_mb } else { $null })
        reserve_mib           = [long]$script:BUDGET_RESERVE_MIB
        floor_mib             = [long]$script:BUDGET_FLOOR_MIB
        ladder_max_mib        = [long]$script:BUDGET_LADDER_MAX_MIB
        geometry_ok           = [bool]$Geometry.ok
        geometry_error        = $(if ($Geometry.ok) { $null } else { [string]$Geometry.reason })
        n_expert              = $(if ($Geometry.ok) { [long]$Geometry.n_expert } else { $null })
        layers                = $(if ($Geometry.ok) { [long]$Geometry.layers } else { $null })
        slot_stride_max       = $(if ($Geometry.ok) { [long]$Geometry.slot_stride_max } else { $null })
        machine_cap_mb        = $(if ($null -ne $calc) { [long]$calc.machine_cap } else { $null })
        model_cap_mb          = $(if ($null -ne $calc) { [long]$calc.model_cap } else { $null })
        structural_min_mb     = $(if ($null -ne $calc) { [long]$calc.structural_min } else { $null })
        supported_min_mb      = $(if ($null -ne $calc) { [long]$calc.supported_min } else { $null })
        candidate_mb          = $(if ($null -ne $calc) { [long]$calc.candidate } else { $null })
        limiting_axis         = $(if ($null -ne $calc) { [string]$calc.limiting_axis } else { $null })
        performance_identity  = $identity
        admission             = 'LS 1-9 preflight RAM contract gates the selected value: budget > available is a hard stop; the dense/KV/server/headroom terms stay [unmeasured] until they are measured. No dynamic lowering either way.'
    }
    # Written BEFORE the banner and before any termination, so the record exists even for the run
    # that stops - and always ahead of the caller's EFFECTIVE record (v0.2 section 3).
    Write-Diag -Kind 'BUDGET_AUTOTUNE' -Data $rec

    if ($source -ceq 'fail_resource') {
        $banner = ('[budget] autotune cannot serve this model on this machine - ' + $reason)
    } else {
        $banner = ('[budget] ' + $budget + ' MB [' + $source + '] - ' + $reason)
    }
    if ($banner -cne [string]$script:BudgetBannerLast) {
        Write-Line $banner
        $script:BudgetBannerLast = $banner
    }

    if ($source -ceq 'fail_resource') {
        Stop-Launcher 'fail_resource' ('budget autotune: ' + $reason)
    }
    return @{ budget_mb = [long]$budget; source = $source; unmeasured = $unmeasured; reason = $reason }
}

# endregion

# ============================================================================
# region 9. SSD PROBE (LS 1-4) - in-script unbuffered large-block random read
# ============================================================================

function Measure-SsdRandomRead {
    param([string] $Path, [long] $BlockBytes, [int] $Samples)
    # FILE_FLAG_NO_BUFFERING requires sector-aligned offsets, sizes and buffer addresses, so the
    # buffer is hand-aligned from unmanaged memory rather than a managed byte[].
    $align = [long]$script:PROBE_SECTOR_ALIGN
    $block = [long]([math]::Floor($BlockBytes / $align) * $align)
    if ($block -lt $align) { $block = $align }

    $h = [IntPtr]::Zero
    $raw = [IntPtr]::Zero
    try {
        $fi = New-Object System.IO.FileInfo($Path)
        if (-not $fi.Exists) { return @{ ok = $false; reason = 'probe target missing' } }
        $size = [long]$fi.Length
        if ($size -lt ($block * 4)) { return @{ ok = $false; reason = 'probe target smaller than 4 blocks' } }

        $flags = $script:FILE_FLAG_NO_BUFFERING -bor $script:FILE_FLAG_RANDOM_ACCESS
        $h = [MoeLauncher.Native]::CreateFileNoSaW($Path, $script:GENERIC_READ,
                ($script:FILE_SHARE_READ -bor $script:FILE_SHARE_WRITE), [IntPtr]::Zero,
                $script:OPEN_EXISTING, [uint32]$flags, [IntPtr]::Zero)
        if ($h -eq $script:INVALID_HANDLE) {
            return @{ ok = $false; reason = ("CreateFile(NO_BUFFERING) failed gle=" + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
        }
        $raw = [System.Runtime.InteropServices.Marshal]::AllocHGlobal([int]($block + $align))
        $addr = [long]$raw
        $alignedAddr = [long](([math]::Floor(($addr + $align - 1) / $align)) * $align)
        $buf = [IntPtr]::new($alignedAddr)

        $maxBlocks = [long][math]::Floor($size / $block) - 1
        if ($maxBlocks -lt 1) { return @{ ok = $false; reason = 'probe target has no full block' } }
        $rng = New-Object System.Random(20260730)
        $sw = New-Object System.Diagnostics.Stopwatch
        $totalBytes = [long]0
        $sw.Start()
        for ($i = 0; $i -lt $Samples; $i++) {
            $blk = [long]$rng.Next(0, [int][math]::Min($maxBlocks, [long][int]::MaxValue))
            $off = $blk * $block
            $newPos = [long]0
            if (-not [MoeLauncher.Native]::SetFilePointerEx($h, $off, [ref]$newPos, $script:FILE_BEGIN)) {
                return @{ ok = $false; reason = 'SetFilePointerEx failed' }
            }
            $read = [uint32]0
            if (-not [MoeLauncher.Native]::ReadFile($h, $buf, [uint32]$block, [ref]$read, [IntPtr]::Zero)) {
                return @{ ok = $false; reason = ("ReadFile failed gle=" + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
            }
            if ($read -eq 0) { return @{ ok = $false; reason = 'ReadFile returned 0 bytes' } }
            $totalBytes += [long]$read
        }
        $sw.Stop()
        $secs = $sw.Elapsed.TotalSeconds
        if ($secs -le 0) { return @{ ok = $false; reason = 'probe elapsed time not measurable' } }
        $mibps = ($totalBytes / 1MB) / $secs
        return @{ ok = $true; mibps = [math]::Round($mibps, 2); block_bytes = $block; samples = $Samples; bytes = $totalBytes }
    } catch {
        return @{ ok = $false; reason = ("probe exception: " + $_.Exception.Message) }
    } finally {
        if ($raw -ne [IntPtr]::Zero) { [System.Runtime.InteropServices.Marshal]::FreeHGlobal($raw) }
        if ($h -ne [IntPtr]::Zero -and $h -ne $script:INVALID_HANDLE) { [void][MoeLauncher.Native]::CloseHandle($h) }
    }
}

function Get-ProbeBlockBytes {
    param([string] $OutputDir)
    $mf = Join-Path $OutputDir 'manifest.json'
    if (Test-Path -LiteralPath $mf -PathType Leaf) {
        $r = Read-JsonFileStrict -Path $mf
        if ($r.ok) {
            $layout = Get-JsonValue -Obj $r.value -Name 'layout'
            $stride = Get-JsonValue -Obj $layout -Name 'slot_stride_max'
            if (Test-JsonNonNegativeInteger $stride -and [long]$stride -gt 0) { return [long]$stride }
        }
    }
    return [long]$script:PROBE_BLOCK_BYTES_FALLBACK
}

# R1-2: the probe must measure the volume the expert cache will be READ FROM (the repack output),
# not the source model's volume - they are frequently different SSDs. Prefer the real artifact;
# before the first repack, use a scratch file on the same volume and delete it afterwards.
function Get-ProbeTarget {
    param([string] $OutputDir)
    $bin = Join-Path $OutputDir 'experts.bin'
    $st = Get-FileAbsenceState -Path $bin
    if ($st.state -eq 'present') {
        try {
            if ((New-Object System.IO.FileInfo($bin)).Length -ge ($script:PROBE_BLOCK_BYTES_FALLBACK * 4)) {
                return @{ ok = $true; path = $bin; scratch = $false }
            }
        } catch { }
    }
    $scratch = Join-Path $OutputDir $script:PROBE_SCRATCH_NAME
    try {
        $fs = [System.IO.File]::Open($scratch, 'Create', 'Write', 'None')
        try {
            $blk = New-Object byte[] 1048576
            $rng = New-Object System.Random(20260730)
            $rng.NextBytes($blk)
            $written = [long]0
            while ($written -lt $script:PROBE_SCRATCH_BYTES) { $fs.Write($blk, 0, $blk.Length); $written += $blk.Length }
            $fs.Flush($true)
        } finally { $fs.Dispose() }
        return @{ ok = $true; path = $scratch; scratch = $true }
    } catch {
        try { if (Test-Path -LiteralPath $scratch) { Remove-Item -LiteralPath $scratch -Force -ErrorAction SilentlyContinue } } catch { }
        return @{ ok = $false; reason = ('probe scratch file could not be written on the output volume: ' + $_.Exception.Message) }
    }
}

# R1-2: a probe result is only reusable when it is bound to the same source_tag / profile /
# expect digest / OUTPUT VOLUME. Without a bound success record the launcher stays conservative
# (probe_failed), so a first-run probe failure can never be laundered into an enabled prefetch on
# the next run.
function Get-ProbeStatePath { return (Join-Path (Get-LauncherStateDir) $script:PROBE_STATE_FILE) }

# R2-3: resolves a path to the GUID name of the volume that actually backs it. A failure to
# resolve is reported as such; callers must then refuse to reuse a stored probe record
# (fail-closed) rather than fall back to a weaker key.
function Get-VolumeIdentity {
    param([string] $Path)
    try {
        $full = [System.IO.Path]::GetFullPath($Path)
        $mount = New-Object System.Text.StringBuilder 260
        if (-not [MoeLauncher.Native]::GetVolumePathNameW($full, $mount, 260)) {
            return @{ ok = $false; reason = ('GetVolumePathNameW failed (GetLastError=' + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() + ')') }
        }
        $guid = New-Object System.Text.StringBuilder 260
        if (-not [MoeLauncher.Native]::GetVolumeNameForVolumeMountPointW($mount.ToString(), $guid, 260)) {
            return @{ ok = $false; reason = ('GetVolumeNameForVolumeMountPointW failed (GetLastError=' + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error() + ')') }
        }
        return @{ ok = $true; id = $guid.ToString().ToLowerInvariant(); mount = $mount.ToString() }
    } catch {
        return @{ ok = $false; reason = ('volume identity query threw: ' + $_.Exception.Message) }
    }
}

# Display helper only - never used as an identity key.
function Get-VolumeKey {
    param([string] $Path)
    try { return ([System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))).ToLowerInvariant() }
    catch { return '?' }
}

function Read-ProbeBinding {
    param([string] $SourceTag, [string] $ProfileId, [string] $ExpectDigest, [string] $OutputDir)
    $path = Get-ProbeStatePath
    $st = Get-FileAbsenceState -Path $path
    if ($st.state -ne 'present') { return @{ ok = $false; reason = 'no stored probe record' } }
    $r = Read-JsonFileStrict -Path $path
    if (-not $r.ok) { return @{ ok = $false; reason = ('probe record unreadable - ' + $r.reason) } }
    $o = $r.value
    if ([long](Get-JsonValue -Obj $o -Name 'state_version') -ne [long]$script:PROBE_STATE_VERSION) {
        return @{ ok = $false; reason = 'probe record schema version mismatch' }
    }
    if ([string](Get-JsonValue -Obj $o -Name 'source_tag')  -cne $SourceTag) { return @{ ok = $false; reason = 'probe record source_tag mismatch' } }
    if ([string](Get-JsonValue -Obj $o -Name 'profile_id')  -cne $ProfileId) { return @{ ok = $false; reason = 'probe record profile_id mismatch' } }
    if (([string](Get-JsonValue -Obj $o -Name 'expect_digest')).ToLowerInvariant() -ne $ExpectDigest.ToLowerInvariant()) {
        return @{ ok = $false; reason = 'probe record expect_digest mismatch' }
    }
    # R2-3: the binding is to the volume GUID. If the identity cannot be resolved now, or the
    # stored record has no identity, the record is NOT reusable (fail-closed).
    $vol = Get-VolumeIdentity -Path $OutputDir
    if (-not $vol.ok) { return @{ ok = $false; reason = ('output volume identity unavailable - ' + $vol.reason) } }
    $storedId = Get-JsonValue -Obj $o -Name 'output_volume_id'
    if (-not (Test-JsonNonEmptyString $storedId)) { return @{ ok = $false; reason = 'probe record carries no volume identity' } }
    if ([string]$storedId -ne $vol.id) {
        return @{ ok = $false; reason = 'probe record was taken on a different output volume' }
    }
    if (-not (Test-JsonBooleanTrue (Get-JsonValue -Obj $o -Name 'probe_ok'))) {
        return @{ ok = $false; reason = 'stored probe record is a failure record' }
    }
    return @{ ok = $true; qd = [int](Get-JsonValue -Obj $o -Name 'qd'); mibps = (Get-JsonValue -Obj $o -Name 'mibps')
              qd_source = 'stored-binding' }
}

function Write-ProbeBinding {
    param([string] $SourceTag, [string] $ProfileId, [string] $ExpectDigest, [string] $OutputDir, $Result)
    try {
        $dir = Get-LauncherStateDir
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        # R2-3: without a resolvable volume identity a SUCCESS record must not be written at all -
        # an unbound success would be reusable on the wrong volume next time.
        $vol = Get-VolumeIdentity -Path $OutputDir
        if ((-not $vol.ok) -and [bool]$Result.ok) {
            Write-Diag -Kind 'probe_binding_not_stored' -Data @{ reason = ('volume identity unavailable - ' + $vol.reason) }
            Write-Line ('[probe] NOTE: probe result not stored (output volume identity unavailable: ' + $vol.reason + '); the next run will re-probe.')
            return
        }
        $volId = ''
        if ($vol.ok) { $volId = $vol.id }
        $o = [ordered]@{
            state_version    = [int]$script:PROBE_STATE_VERSION
            source_tag       = $SourceTag
            profile_id       = $ProfileId
            expect_digest    = $ExpectDigest
            output_volume_id = $volId
            output_volume_mount = [string]$vol.mount
            probe_ok         = [bool]$Result.ok
            qd               = [int]$Result.qd
            mibps            = $Result.mibps
            ts               = (Get-Date).ToUniversalTime().ToString('o')
        }
        $path = Get-ProbeStatePath
        $tmp = $path + '.tmp'
        [System.IO.File]::WriteAllText($tmp, ($o | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
        Move-FileAtomic -TempPath $tmp -FinalPath $path
        Write-Diag -Kind 'PROBE_BINDING_SAVED' -Data $o
    } catch {
        Write-Diag -Kind 'probe_binding_save_failed' -Data @{ reason = $_.Exception.Message }
    }
}

# LS 1-4 / RS 5: probe failure is a degraded branch, never a terminal error.
function Invoke-StartupProbe {
    param([string] $OutputDir)
    $block = Get-ProbeBlockBytes -OutputDir $OutputDir
    $target = Get-ProbeTarget -OutputDir $OutputDir
    if (-not $target.ok) {
        # LS 12-1: this tier is a PROVISIONAL pre-repack I/O sanity check. It does not set QD and it
        # does not decide prefetch - the LS 12 sweep after the verify gate is the measurement
        # authority, and a failed scratch tier plus a successful sweep recovers.
        Write-Line ('[probe] scratch sanity measurement could not start ({0}); provisional only - the QD sweep after the verify gate decides QD (RS 5)' -f $target.reason)
        Write-Diag -Kind 'PROBE_FAILED' -Data @{ reason = $target.reason }
        return @{ ok = $false; qd = [int]$script:QD_DEGRADED; reason = $target.reason }
    }
    $ProbeTarget = $target.path
    try {
    # LS 11-6-b (UI-4): stage start line. Measured at 0.06 s on this machine, but it sits inside the
    # same silent window and a cold or slow volume is exactly when it would be worth announcing.
    Write-Line ('[probe] measuring SSD random read ({0} samples x {1} B unbuffered)...' -f $script:PROBE_SAMPLES, $block)
    $res = Measure-SsdRandomRead -Path $ProbeTarget -BlockBytes $block -Samples $script:PROBE_SAMPLES
    if (-not $res.ok) {
        Write-Line ('[probe] scratch sanity read failed ({0}); provisional only - the QD sweep after the verify gate decides QD (RS 5)' -f $res.reason)
        Write-Diag -Kind 'PROBE_FAILED' -Data $res
        return @{ ok = $false; qd = [int]$script:QD_DEGRADED; reason = $res.reason }
    }
    $qd = [int]$script:QD_DEGRADED
    $qdSource = 'conservative-default'
    if ($null -ne $script:PROBE_QD_MAP) {
        foreach ($row in @($script:PROBE_QD_MAP)) {
            if ($res.mibps -ge [double]$row.min_mibps) { $qd = [int]$row.qd; $qdSource = 'probe-map'; break }
        }
    }
    # Same LS 12-1 rule on the success path: this number is a provisional sanity reading, so it is
    # reported WITHOUT a QD verdict (the sweep below owns that).
    Write-Line ('[probe] scratch sanity read {0} MiB/s @ {1} B block x{2} on {3} (provisional)' -f $res.mibps, $res.block_bytes, $res.samples, (Get-VolumeKey -Path $ProbeTarget))
    if ($null -eq $script:PROBE_QD_MAP) {
        Write-Line '        the scratch probe->QD threshold table is [unmeasured] (LS 9 item 1); QD comes from the LS 12 sweep.'
    }
    Write-Diag -Kind 'PROBE_OK' -Data @{ mibps = $res.mibps; block_bytes = $res.block_bytes; qd = $qd
                                         qd_source = $qdSource; target = $ProbeTarget; scratch = $target.scratch }
    return @{ ok = $true; qd = $qd; mibps = $res.mibps; qd_source = $qdSource }
    } finally {
        if ($target.scratch) {
            try { if (Test-Path -LiteralPath $ProbeTarget -PathType Leaf) { Remove-Item -LiteralPath $ProbeTarget -Force -ErrorAction SilentlyContinue } } catch { }
        }
    }
}

# endregion

# ============================================================================
# region 9-B. STARTUP QD SWEEP (LS 12 / QD-1)
#   The scratch probe above is a pre-repack I/O sanity check and stays PROVISIONAL. This sweep -
#   or a valid v2 binding produced by an earlier one - is the single measurement authority for the
#   AUTOMATIC QD default (LS 12-1, R10-2 wording); the QD priority chain
#   (session/CLI override > stored preset > measured-sweep > conservative-default) then decides the
#   final effective QD.
# ============================================================================

# LS 12-2 block population: manifest.layout.layers[*].stride_bytes.
function Get-ManifestStrides {
    param([string] $ManifestPath)
    $r = Read-JsonFileStrict -Path $ManifestPath
    if (-not $r.ok) { return @{ ok = $false; reason = ('manifest.json unreadable - ' + $r.reason) } }
    $layout = Get-JsonValue -Obj $r.value -Name 'layout'
    if ($null -eq $layout) { return @{ ok = $false; reason = 'manifest.json has no layout object' } }
    $vals = @()
    foreach ($L in (Get-JsonArray -Obj $layout -Name 'layers')) {
        $s = Get-JsonValue -Obj $L -Name 'stride_bytes'
        if (-not (Test-JsonNonNegativeInteger $s) -or [long]$s -le 0) {
            return @{ ok = $false; reason = 'layout.layers[*].stride_bytes is missing or not a positive integer' }
        }
        $vals += [long]$s
    }
    if ($vals.Count -eq 0) { return @{ ok = $false; reason = 'manifest.json carries no layout.layers[*].stride_bytes population' } }
    # LS 12-2 "length and offset follow the logical sector AND the manifest alignment": the repack
    # output's own alignment is part of that constraint, so it is read here and folded into the
    # applied alignment by Get-SweepAlignment. Absent/invalid = no additional constraint.
    $ab = Get-JsonValue -Obj $layout -Name 'align_bytes'
    $align = [long]0
    if ((Test-JsonNonNegativeInteger $ab) -and [long]$ab -gt 0) { $align = [long]$ab }
    return @{ ok = $true; strides = $vals; align_bytes = $align }
}

# LS 12-2 representative block (pure function): lower median of the stride population, then the
# nearest size inside [1 MiB, 16 MiB] that satisfies the runtime alignment; ties take the smaller
# size; no candidate inside the range = the sweep fails.
function Get-SweepBlockBytes {
    param($Strides, [long] $AlignBytes)
    $vals = @()
    foreach ($s in @($Strides)) { $vals += [long]$s }
    if ($vals.Count -eq 0) { return @{ ok = $false; reason = 'empty stride population' } }
    if ($AlignBytes -le 0) { return @{ ok = $false; reason = 'alignment is not a positive number' } }
    $sorted = @($vals | Sort-Object)
    $median = [long]$sorted[[int][math]::Floor(($sorted.Count - 1) / 2)]
    $lo = [long]$script:SWEEP_BLOCK_MIN_BYTES
    $hi = [long]$script:SWEEP_BLOCK_MAX_BYTES
    $target = $median
    if ($target -lt $lo) { $target = $lo }
    if ($target -gt $hi) { $target = $hi }
    $cands = @()
    foreach ($c in @([long]([math]::Floor($target / $AlignBytes) * $AlignBytes),
                     [long]([math]::Ceiling($target / $AlignBytes) * $AlignBytes))) {
        if ($c -ge $lo -and $c -le $hi) { $cands += [long]$c }
    }
    if ($cands.Count -eq 0) {
        return @{ ok = $false; median = $median
                  reason = ('no block size in [1 MiB, 16 MiB] satisfies the ' + $AlignBytes + ' B alignment') }
    }
    $best = $null
    $bestDist = $null
    foreach ($c in @($cands | Sort-Object)) {
        $d = [math]::Abs($c - $target)
        if ($null -eq $bestDist -or $d -lt $bestDist) { $best = [long]$c; $bestDist = $d }
    }
    return @{ ok = $true; block_bytes = [long]$best; median = $median; align_bytes = [long]$AlignBytes }
}

# LS 12-2 alignment, all three constraints at once: length and offset follow the logical sector AND
# the manifest alignment, the buffer address follows the PHYSICAL sector size. One value satisfying
# the maximum of the three satisfies all three. The page-size floor keeps this valid when the
# physical query fails (it returns 0), and taking the manifest value into the maximum means a failed
# device query can never land BELOW the alignment the repack output was written with.
function Get-SweepAlignment {
    param([string] $Path, [long] $ManifestAlignBytes = 0)
    $phys = 0
    try { $phys = [long][MoeLauncher.Sweep]::QueryPhysicalSectorBytes($Path) } catch { $phys = 0 }
    $align = [long]$script:SWEEP_FALLBACK_ALIGN
    if ($phys -gt $align) { $align = [long]$phys }
    if ($ManifestAlignBytes -gt $align) { $align = [long]$ManifestAlignBytes }
    return @{ align_bytes = [long]$align; physical_bytes = [long]$phys; manifest_bytes = [long]$ManifestAlignBytes }
}

# LS 12-2 determinism: every QD point walks the SAME offset sequence, derived from manifest_sha256
# (no wall clock, no per-point RNG drift).
function New-SweepOffsetSequence {
    param([string] $ManifestSha256, [long] $BlockCount, [long] $BlockBytes, [int] $Count)
    if ($BlockCount -lt 1) { return @{ ok = $false; reason = 'target holds no full block' } }
    if (-not (Test-Sha256Hex $ManifestSha256)) { return @{ ok = $false; reason = 'manifest_sha256 is not a 64 hex string' } }
    $seed = New-Object byte[] 32
    for ($i = 0; $i -lt 32; $i++) {
        $seed[$i] = [byte][convert]::ToInt32($ManifestSha256.Substring($i * 2, 2), 16)
    }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $out = New-Object 'System.Collections.Generic.List[long]'
        $buf = New-Object byte[] 36
        [System.Array]::Copy($seed, 0, $buf, 0, 32)
        for ($i = 0; $i -lt $Count; $i++) {
            $ib = [System.BitConverter]::GetBytes([int]$i)
            [System.Array]::Copy($ib, 0, $buf, 32, 4)
            $d = $sha.ComputeHash($buf)
            $v = [System.BitConverter]::ToUInt64($d, 0)
            $out.Add([long](([long]($v % [uint64]$BlockCount)) * $BlockBytes))
        }
        return @{ ok = $true; offsets = $out.ToArray() }
    } catch {
        return @{ ok = $false; reason = ('offset sequence derivation failed: ' + $_.Exception.Message) }
    } finally { $sha.Dispose() }
}

# LS 12-2 point record. mibps is a DISPLAY value; the comparison value used by the selection
# function is the raw bytes/elapsed pair, before any rounding.
function ConvertTo-SweepPointRecord {
    param($Raw)
    $rec = [ordered]@{ qd = [int]$Raw.Qd; status = 'failed'; bytes = [long]$Raw.Bytes
                       elapsed_ticks = [long]$Raw.ElapsedTicks; reads = [int]$Raw.Reads
                       mibps = $null; error = $null }
    if ($Raw.Error) { $rec['error'] = [string]$Raw.Error }
    if ([bool]$Raw.Ok -and [long]$Raw.Bytes -gt 0 -and [long]$Raw.ElapsedTicks -gt 0) {
        $m = ([double]$Raw.Bytes / 1MB) / ([double]$Raw.ElapsedTicks / 10000000.0)
        if ([double]::IsNaN($m) -or [double]::IsInfinity($m) -or $m -le 0) {
            if (-not $rec['error']) { $rec['error'] = 'throughput is not a finite positive number' }
        } else {
            $rec['status'] = 'ok'
            $rec['mibps'] = [math]::Round($m, 2)
        }
    } elseif (-not $rec['error']) {
        $rec['error'] = 'no measurable window'
    }
    return $rec
}

function Invoke-SweepRound {
    param([string] $TargetPath, [long] $BlockBytes, $Offsets, [int] $WindowMs, $QdOrder, [long] $AlignBytes)
    $pts = @()
    foreach ($qd in @($QdOrder)) {
        $raw = [MoeLauncher.Sweep]::RunPoint($TargetPath, [int]$qd, [long]$BlockBytes, $Offsets,
                                            [int]$WindowMs, [long]$AlignBytes)
        $rec = ConvertTo-SweepPointRecord -Raw $raw
        Write-Diag -Kind 'SWEEP_POINT' -Data $rec
        $pts += $rec
    }
    return , $pts
}

# The two crossed windows (and the confirmation round, when it runs) are one measurement per QD:
# the point's bytes and elapsed time are summed. A window that failed marks the whole point failed -
# a point mixing a failure with a success is not a clean measurement.
function Merge-SweepRounds {
    param($Rounds)
    $acc = @{}
    foreach ($qd in $script:SWEEP_QD_POINTS) {
        $acc[[string]$qd] = @{ qd = [int]$qd; bytes = [long]0; elapsed_ticks = [long]0; reads = 0
                               windows = 0; failed = $false; error = $null }
    }
    foreach ($round in @($Rounds)) {
        foreach ($p in @($round)) {
            $k = [string]$p.qd
            if (-not $acc.ContainsKey($k)) { continue }
            $slot = $acc[$k]
            $slot.windows = [int]$slot.windows + 1
            if ([string]$p.status -cne 'ok') {
                $slot.failed = $true
                if (-not $slot.error) { $slot.error = [string]$p.error }
                continue
            }
            $slot.bytes = [long]$slot.bytes + [long]$p.bytes
            $slot.elapsed_ticks = [long]$slot.elapsed_ticks + [long]$p.elapsed_ticks
            $slot.reads = [int]$slot.reads + [int]$p.reads
        }
    }
    $out = @()
    foreach ($qd in $script:SWEEP_QD_POINTS) {
        $slot = $acc[[string]$qd]
        $rec = [ordered]@{ qd = [int]$qd; status = 'failed'; bytes = [long]$slot.bytes
                           elapsed_ticks = [long]$slot.elapsed_ticks; reads = [int]$slot.reads
                           windows = [int]$slot.windows; mibps = $null; error = $slot.error }
        if ((-not $slot.failed) -and $slot.windows -gt 0 -and $slot.bytes -gt 0 -and $slot.elapsed_ticks -gt 0) {
            $m = ([double]$slot.bytes / 1MB) / ([double]$slot.elapsed_ticks / 10000000.0)
            if ([double]::IsNaN($m) -or [double]::IsInfinity($m) -or $m -le 0) {
                if (-not $rec['error']) { $rec['error'] = 'throughput is not a finite positive number' }
            } else {
                $rec['status'] = 'ok'
                $rec['mibps'] = [math]::Round($m, 2)
            }
        } elseif (-not $rec['error']) {
            $rec['error'] = 'no measured window for this queue depth'
        }
        $out += $rec
    }
    return , $out
}

# LS 12-3 selection (MUST - single-solution formula). PURE: it reads only the point records it is
# handed, which is what makes it directly regression-testable.
#   V = {(q,bq) | status(q)=ok, bq finite positive}, bq = bytes/elapsed BEFORE display rounding
#   M = max(bq) ; S90 = {q | 10*bq >= 9*M} ; q_base = min(S90)
#   arm 'catalog-fixed' only: E      = {q in S90 | q >= N+1} ; min(E) when E is non-empty
#   arm 'init' only:          E_init = {q in S90 | q >= 2}   ; min(E_init) when it is non-empty
# 'prefetch-preferred' is a bounded preference applied from a validated prior - it is NOT new
# evidence that this device gains end to end (LS 12-3 wording rule). 'init-prefetch-preferred' is
# the same kind of preference applied from an UNVALIDATED prior, which is why it is a different
# word: PI 2 forbids an init-produced number from wearing a validated label.
# P4 2 / 3: the arm arrives already decided (Resolve-PrefetchArm), because a refused opt-in must
# not move the QD of the run it was refused for. This function is handed 'none' in that case, and
# the v0.4 formula and the 'prefetch-preferred' literal below are unchanged character for
# character - only the signal that selects the branch changed from the retired one-axis state.
function Select-SweepQd {
    param($Points, [string] $PrefetchArm, $CatalogN)
    $valid = @()
    foreach ($p in @($Points)) {
        if ([string]$p.status -cne 'ok') { continue }
        $b = [double]$p.bytes
        $t = [double]$p.elapsed_ticks
        if ($b -le 0 -or $t -le 0) { continue }
        $bq = $b / $t
        if ([double]::IsNaN($bq) -or [double]::IsInfinity($bq) -or $bq -le 0) { continue }
        $valid += @{ qd = [int]$p.qd; bq = [double]$bq }
    }
    if ($valid.Count -eq 0) {
        return @{ ok = $false; qd = [int]$script:QD_DEGRADED; reason = $null; degraded = $true
                  qd_source = 'conservative-default'; s90 = @(); q_base = $null }
    }
    $m = [double]0
    foreach ($v in $valid) { if ($v.bq -gt $m) { $m = [double]$v.bq } }
    $s90 = @()
    foreach ($v in $valid) { if ((10.0 * $v.bq) -ge (9.0 * $m)) { $s90 += [int]$v.qd } }
    $s90 = @($s90 | Sort-Object)
    $qBase = [int]$s90[0]
    $qd = $qBase
    $reason = 'io-knee'
    if ($PrefetchArm -ceq $script:PREFETCH_ARM_CATALOG -and $null -ne $CatalogN) {
        $need = [long]$CatalogN + 1
        $e = @()
        foreach ($q in $s90) { if ([long]$q -ge $need) { $e += [int]$q } }
        if ($e.Count -gt 0) {
            $pick = [int](@($e | Sort-Object)[0])
            if ($pick -ne $qBase) { $qd = $pick; $reason = 'prefetch-preferred' }
        }
    }
    # PI 4-2 step 2: the init arm has no catalog N to clear, so its bound is the depth below which
    # the formula produces nothing at all (QD1 -> OFF). Same shape, different prior, different word.
    if ($PrefetchArm -ceq $script:PREFETCH_ARM_INIT) {
        $eInit = @()
        foreach ($q in $s90) { if ([long]$q -ge 2) { $eInit += [int]$q } }
        if ($eInit.Count -gt 0) {
            $pick = [int](@($eInit | Sort-Object)[0])
            if ($pick -ne $qBase) { $qd = $pick; $reason = 'init-prefetch-preferred' }
        }
    }
    return @{ ok = $true; qd = [int]$qd; reason = $reason; degraded = $false
              qd_source = 'measured-sweep'; s90 = $s90; q_base = $qBase; max_bq = $m }
}

# LS 12-5: the expectation calculator may consume ONLY the sweep point that matches the final QD.
# A different point (or the QD1 number) must never stand in for it.
function Get-SweepPointForQd {
    param($Points, [int] $Qd)
    foreach ($p in @($Points)) {
        if ([int]$p.qd -eq $Qd -and [string]$p.status -ceq 'ok' -and $null -ne $p.mibps) { return $p }
    }
    return $null
}

# ---- LS 12-4 binding v2 (target-key map) ----------------------------------------------------
function Get-SweepStatePath { return (Join-Path (Get-LauncherStateDir) $script:SWEEP_STATE_FILE) }

# One cache lookup, one sweep and one save attempt per target key per launcher process (LS 12-4).
$script:SweepLookups = @{}
$script:SweepRuns    = @{}
$script:SweepSaves   = @{}

function Get-SweepTargetKey {
    param([string] $SourceTag, [string] $ProfileId, [string] $ExpectDigest,
          [string] $ManifestSha256, [string] $VolumeGuid)
    $canon = ($script:SWEEP_PROBE_ALGORITHM + '|' + $script:SWEEP_MEASUREMENT_METHOD + '|' +
              [string]$SourceTag + '|' + [string]$ProfileId + '|' +
              ([string]$ExpectDigest).ToLowerInvariant() + '|' +
              ([string]$ManifestSha256).ToLowerInvariant() + '|' +
              ([string]$VolumeGuid).ToLowerInvariant())
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $h = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($canon))
        return @{ canonical = $canon
                  key = ([System.BitConverter]::ToString($h).Replace('-', '').ToLowerInvariant()) }
    } finally { $sha.Dispose() }
}

# LS 12-4 freshness needs real instants, not just non-empty strings. Both the launcher ('o' round
# trip) and repack_experts.py (datetime.isoformat with a +00:00 offset) are accepted; a value with
# no offset is read as UTC.
function ConvertTo-UtcInstant {
    param($Value)
    if (-not (Test-JsonNonEmptyString $Value)) { return @{ ok = $false; reason = 'missing or not a string' } }
    $dt = [datetime]::MinValue
    $styles = ([System.Globalization.DateTimeStyles]::AdjustToUniversal -bor
               [System.Globalization.DateTimeStyles]::AssumeUniversal -bor
               [System.Globalization.DateTimeStyles]::AllowWhiteSpaces)
    if ([datetime]::TryParse([string]$Value, [System.Globalization.CultureInfo]::InvariantCulture, $styles, [ref]$dt)) {
        return @{ ok = $true; utc = $dt }
    }
    return @{ ok = $false; reason = 'not a parsable UTC timestamp' }
}

# R11-1 / LS 12-4: freshness gate for a cache HIT. The repack completion time is the verify report's
# checked_at, and the target key cannot see it: `repack_experts.py --verify-only` (a supported path,
# repack_experts.py:1574-1578) appends a NEW verify record while experts.bin and manifest.json stay
# byte-identical, so profile / expect digest / manifest_sha256 / volume GUID all stay the same. A
# stored sweep is therefore only reusable while it is bound to the CURRENT checked_at, and it must
# not predate it. Unusable/absent timestamps fail closed to a cache miss (one extra sweep), never to
# a hit.
function Test-SweepFreshness {
    param($Entry, [string] $CheckedAt)
    $cur = ConvertTo-UtcInstant -Value $CheckedAt
    if (-not $cur.ok) {
        return @{ ok = $false; reason = ('current verify_report.checked_at unusable - ' + $cur.reason) }
    }
    if ($null -eq $Entry.repack_checked_at_utc -or $null -eq $Entry.swept_at_utc) {
        return @{ ok = $false; reason = 'stored entry carries no usable timestamps' }
    }
    if ([datetime]$Entry.repack_checked_at_utc -ne [datetime]$cur.utc) {
        return @{ ok = $false; reason = 'stored sweep is bound to a different verify_report.checked_at' }
    }
    if ([datetime]$Entry.swept_at_utc -lt [datetime]$cur.utc) {
        return @{ ok = $false; reason = 'stored sweep predates the current verify_report.checked_at' }
    }
    return @{ ok = $true }
}

function Test-SweepEntry {
    param($Entry)
    if ($null -eq $Entry) { return @{ ok = $false; reason = 'entry is null' } }
    if ([string](Get-JsonValue -Obj $Entry -Name 'probe_algorithm') -cne $script:SWEEP_PROBE_ALGORITHM) {
        return @{ ok = $false; reason = 'probe_algorithm mismatch' }
    }
    if ([string](Get-JsonValue -Obj $Entry -Name 'measurement_method') -cne $script:SWEEP_MEASUREMENT_METHOD) {
        return @{ ok = $false; reason = 'measurement_method mismatch' }
    }
    foreach ($k in @('profile_id', 'expectation_digest', 'manifest_sha256', 'output_volume_guid',
                     'selected_reason', 'swept_at', 'repack_checked_at')) {
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $Entry -Name $k))) {
            return @{ ok = $false; reason = ($k + ' missing or not a non-empty string') }
        }
    }
    # R11-1: both times are part of the schema, so an unparsable one is schema-invalid (= miss).
    $sweptAt = ConvertTo-UtcInstant -Value (Get-JsonValue -Obj $Entry -Name 'swept_at')
    if (-not $sweptAt.ok) { return @{ ok = $false; reason = ('swept_at is ' + $sweptAt.reason) } }
    $repackAt = ConvertTo-UtcInstant -Value (Get-JsonValue -Obj $Entry -Name 'repack_checked_at')
    if (-not $repackAt.ok) { return @{ ok = $false; reason = ('repack_checked_at is ' + $repackAt.reason) } }
    foreach ($k in @('selected_qd', 'block_bytes', 'window_duration_ms')) {
        $v = Get-JsonValue -Obj $Entry -Name $k
        if (-not (Test-JsonNonNegativeInteger $v) -or [long]$v -le 0) {
            return @{ ok = $false; reason = ($k + ' missing or not a positive integer') }
        }
    }
    $pts = @(Get-JsonArray -Obj $Entry -Name 'points')
    if ($pts.Count -ne @($script:SWEEP_QD_POINTS).Count) {
        return @{ ok = $false; reason = 'points array does not carry every swept queue depth' }
    }
    $recs = @()
    foreach ($p in $pts) {
        $q = Get-JsonValue -Obj $p -Name 'qd'
        if (-not (Test-JsonNonNegativeInteger $q) -or [long]$q -le 0) {
            return @{ ok = $false; reason = 'point qd is not a positive integer' }
        }
        $st = [string](Get-JsonValue -Obj $p -Name 'status')
        if ($st -cne 'ok' -and $st -cne 'failed') { return @{ ok = $false; reason = 'point status is not ok/failed' } }
        $rec = [ordered]@{ qd = [int]$q; status = $st; bytes = [long]0; elapsed_ticks = [long]0
                           mibps = $null; error = [string](Get-JsonValue -Obj $p -Name 'error') }
        if ($st -ceq 'ok') {
            $b = Get-JsonValue -Obj $p -Name 'bytes'
            $t = Get-JsonValue -Obj $p -Name 'elapsed_ticks'
            if (-not (Test-JsonNonNegativeInteger $b) -or [long]$b -le 0 -or
                -not (Test-JsonNonNegativeInteger $t) -or [long]$t -le 0) {
                return @{ ok = $false; reason = 'ok point without a positive bytes/elapsed pair' }
            }
            $rec['bytes'] = [long]$b
            $rec['elapsed_ticks'] = [long]$t
            $mv = Get-JsonValue -Obj $p -Name 'mibps'
            if ($null -ne $mv) { $rec['mibps'] = [double]$mv }
        }
        $recs += $rec
    }
    $selQd = [int](Get-JsonValue -Obj $Entry -Name 'selected_qd')
    return @{ ok = $true; qd = $selQd; selected_reason = [string](Get-JsonValue -Obj $Entry -Name 'selected_reason')
              points = $recs; block_bytes = [long](Get-JsonValue -Obj $Entry -Name 'block_bytes')
              window_duration_ms = [long](Get-JsonValue -Obj $Entry -Name 'window_duration_ms')
              swept_at = [string](Get-JsonValue -Obj $Entry -Name 'swept_at')
              repack_checked_at = [string](Get-JsonValue -Obj $Entry -Name 'repack_checked_at')
              swept_at_utc = $sweptAt.utc; repack_checked_at_utc = $repackAt.utc }
}

# LS 12-4 miss conditions - v1 / schema-invalid / parse failure / key mismatch - plus the R11-1
# freshness condition, which is the one thing the target key cannot express (see Test-SweepFreshness).
# The lookup happens at most once per key per process.
function Read-SweepBinding {
    param([string] $Key, [string] $CheckedAt)
    if ($script:SweepLookups.ContainsKey($Key)) { return $script:SweepLookups[$Key] }
    $res = @{ ok = $false; reason = 'no stored sweep binding' }
    $path = Get-SweepStatePath
    $st = Get-FileAbsenceState -Path $path
    if ($st.state -eq 'unknown') {
        $res = @{ ok = $false; reason = ('sweep binding state not observable - ' + $st.reason) }
    } elseif ($st.state -eq 'present') {
        $r = Read-JsonFileStrict -Path $path
        if (-not $r.ok) {
            $res = @{ ok = $false; reason = ('sweep binding parse failure - ' + $r.reason) }
        } else {
            $ver = Get-JsonValue -Obj $r.value -Name 'state_version'
            if (-not (Test-JsonNonNegativeInteger $ver) -or [long]$ver -ne [long]$script:SWEEP_STATE_VERSION) {
                $res = @{ ok = $false; reason = 'sweep binding is not state_version 2 (v1 or unknown schema)' }
            } else {
                $entries = Get-JsonValue -Obj $r.value -Name 'entries'
                $e = $null
                if ($null -ne $entries) { $e = Get-JsonValue -Obj $entries -Name $Key }
                if ($null -eq $e) {
                    $res = @{ ok = $false; reason = 'no entry for this target key' }
                } else {
                    $chk = Test-SweepEntry -Entry $e
                    if (-not $chk.ok) { $res = @{ ok = $false; reason = ('stored entry is schema-invalid - ' + $chk.reason) } }
                    else {
                        $fresh = Test-SweepFreshness -Entry $chk -CheckedAt $CheckedAt
                        if (-not $fresh.ok) {
                            $res = @{ ok = $false; reason = ('stale sweep binding - ' + $fresh.reason) }
                        } else {
                            $res = @{ ok = $true; qd = [int]$chk.qd; reason = [string]$chk.selected_reason
                                      points = $chk.points; block_bytes = $chk.block_bytes
                                      window_duration_ms = $chk.window_duration_ms; swept_at = $chk.swept_at
                                      repack_checked_at = $chk.repack_checked_at }
                        }
                    }
                }
            }
        }
    }
    $script:SweepLookups[$Key] = $res
    Write-Diag -Kind 'SWEEP_BINDING_READ' -Data @{ key = $Key; hit = [bool]$res.ok; reason = [string]$res.reason }
    return $res
}

# LS 12-4 publish: same atomic MoveFileExW path the scratch record uses. A save failure is confined
# to the cache layer - the run keeps its in-memory measurement, surfaces binding_persist=failed and
# does NOT retry in the same run.
function Write-SweepBinding {
    param([string] $Key, $Entry)
    if ($script:SweepSaves.ContainsKey($Key)) { return $script:SweepSaves[$Key] }
    $res = @{ ok = $false; reason = 'not attempted' }
    try {
        $dir = Get-LauncherStateDir
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $path = Get-SweepStatePath
        $entries = [ordered]@{}
        # Other target keys survive: the map exists so that switching profiles back and forth does
        # not re-sweep. A v1 / unparsable document is simply replaced (the LS 12-4 migration).
        $cur = Read-JsonFileStrict -Path $path
        if ($cur.ok) {
            $ver = Get-JsonValue -Obj $cur.value -Name 'state_version'
            if ((Test-JsonNonNegativeInteger $ver) -and [long]$ver -eq [long]$script:SWEEP_STATE_VERSION) {
                $old = Get-JsonValue -Obj $cur.value -Name 'entries'
                foreach ($k in (Get-JsonKeys -Obj $old)) {
                    if ($k -cne $Key) { $entries[$k] = (Get-JsonValue -Obj $old -Name $k) }
                }
            }
        }
        $entries[$Key] = $Entry
        $doc = [ordered]@{ state_version = [int]$script:SWEEP_STATE_VERSION; entries = $entries }
        $tmp = $path + '.tmp'
        [System.IO.File]::WriteAllText($tmp, ($doc | ConvertTo-Json -Depth 12), (New-Object System.Text.UTF8Encoding($false)))
        Move-FileAtomic -TempPath $tmp -FinalPath $path
        $res = @{ ok = $true }
        Write-Diag -Kind 'SWEEP_BINDING_SAVED' -Data @{ key = $Key; selected_qd = $Entry['selected_qd'] }
    } catch {
        $res = @{ ok = $false; reason = $_.Exception.Message }
        Write-Diag -Kind 'sweep_binding_save_failed' -Data @{ key = $Key; reason = $_.Exception.Message }
    }
    $script:SweepSaves[$Key] = $res
    return $res
}

# ---- LS 12-2 sweep run ----------------------------------------------------------------------
function Invoke-QdSweep {
    param([string] $OutputDir, [string] $ManifestSha256, [string] $PrefetchArm, $CatalogN)
    $bin = Join-Path $OutputDir 'experts.bin'
    $st = Get-FileAbsenceState -Path $bin
    if ($st.state -ne 'present') { return @{ ok = $false; reason = 'experts.bin is not available for the sweep' } }
    $size = [long]0
    try { $size = [long](New-Object System.IO.FileInfo($bin)).Length }
    catch { return @{ ok = $false; reason = ('experts.bin size query failed: ' + $_.Exception.Message) } }

    $strides = Get-ManifestStrides -ManifestPath (Join-Path $OutputDir 'manifest.json')
    if (-not $strides.ok) { return @{ ok = $false; reason = $strides.reason } }
    $al = Get-SweepAlignment -Path $bin -ManifestAlignBytes $strides.align_bytes
    $blk = Get-SweepBlockBytes -Strides $strides.strides -AlignBytes $al.align_bytes
    if (-not $blk.ok) { return @{ ok = $false; reason = $blk.reason } }
    $blockCount = [long][math]::Floor($size / $blk.block_bytes)
    if ($blockCount -lt [long]$script:SWEEP_MIN_BLOCK_SPAN) {
        return @{ ok = $false; reason = ('experts.bin holds ' + $blockCount + ' full ' + $blk.block_bytes +
                  ' B blocks, below the ' + $script:SWEEP_MIN_BLOCK_SPAN + ' block span the sweep needs') }
    }
    $seq = New-SweepOffsetSequence -ManifestSha256 $ManifestSha256 -BlockCount $blockCount `
               -BlockBytes $blk.block_bytes -Count $script:SWEEP_OFFSET_COUNT
    if (-not $seq.ok) { return @{ ok = $false; reason = $seq.reason } }

    $win = [int]$script:SWEEP_WINDOW_MS
    $totalMs = $win * 2 * @($script:SWEEP_QD_POINTS).Count
    # LS 11-6-b (UI-4): this is a multi-second silent window, so it announces itself first.
    Write-Line ('[sweep] measuring queue depth {0} on experts.bin ({1} B block, 2 x {2} ms per point, about {3} s)...' -f `
                (($script:SWEEP_QD_POINTS) -join '/'), $blk.block_bytes, $win, [int]($totalMs / 1000))
    $rounds = @()
    $rounds += , (Invoke-SweepRound -TargetPath $bin -BlockBytes $blk.block_bytes -Offsets $seq.offsets `
                      -WindowMs $win -QdOrder $script:SWEEP_ORDER_FORWARD -AlignBytes $al.align_bytes)
    $rounds += , (Invoke-SweepRound -TargetPath $bin -BlockBytes $blk.block_bytes -Offsets $seq.offsets `
                      -WindowMs $win -QdOrder $script:SWEEP_ORDER_REVERSE -AlignBytes $al.align_bytes)
    $selA = Select-SweepQd -Points $rounds[0] -PrefetchArm $PrefetchArm -CatalogN $CatalogN
    $selB = Select-SweepQd -Points $rounds[1] -PrefetchArm $PrefetchArm -CatalogN $CatalogN
    $confirm = $false
    if ([int]$selA.qd -ne [int]$selB.qd) {
        # LS 12-2: the two crossed rounds disagreed once - one confirmation round, never a loop.
        $confirm = $true
        Write-Line ('[sweep] the two crossed rounds selected QD{0} and QD{1}; running one confirmation round...' -f $selA.qd, $selB.qd)
        $rounds += , (Invoke-SweepRound -TargetPath $bin -BlockBytes $blk.block_bytes -Offsets $seq.offsets `
                          -WindowMs $win -QdOrder $script:SWEEP_ORDER_FORWARD -AlignBytes $al.align_bytes)
    }
    $points = Merge-SweepRounds -Rounds $rounds
    $sel = Select-SweepQd -Points $points -PrefetchArm $PrefetchArm -CatalogN $CatalogN
    return @{ ok = $true; points = $points; selection = $sel; block_bytes = [long]$blk.block_bytes
              window_duration_ms = $win; rounds = @($rounds).Count; confirm_round = $confirm
              align_bytes = [long]$al.align_bytes; physical_sector = [long]$al.physical_bytes
              manifest_align = [long]$al.manifest_bytes
              block_count = $blockCount; stride_median = [long]$blk.median }
}

# LS 12-1 / 12-4 orchestration: look the target key up once, sweep at most once, publish once.
# Every failure here is NON-TERMINAL - it degrades to QD1 / conservative-default exactly like the
# scratch probe's failure branch (RS 5).
# P4 2: -PrefetchArm arrives from Resolve-PrefetchArm, i.e. AFTER the preset/CLI merge and BEFORE
# any QD is picked. The sweep no longer reads a policy field off the profile itself: a row that is
# catalog-fixed on paper but refused at runtime (identity, adapt, semantic) must sweep exactly like
# an unpreferred row, which is only true if the caller decides the arm.
function Resolve-QdSweep {
    param([string] $OutputDir, [string] $SourceTag, [string] $ProfileId, [string] $ExpectDigest,
          [string] $ManifestSha256, [string] $CheckedAt, $Profile, [string] $PrefetchArm = 'none')
    $out = @{ ok = $false; qd = [int]$script:QD_DEGRADED; qd_source = 'conservative-default'
              reason = $null; points = @(); from_binding = $false; persist_failed = $false
              detail = 'sweep not run'; block_bytes = $null; window_duration_ms = $null }
    try {
        $pfN = $null
        $pf = Get-JsonValue -Obj $Profile -Name 'prefetch'
        if ($null -ne $pf) {
            $n = Get-JsonValue -Obj $pf -Name 'n'
            if (Test-JsonNonNegativeInteger $n) { $pfN = [long]$n }
        }

        $vol = Get-VolumeIdentity -Path $OutputDir
        $key = $null
        if ($vol.ok) {
            $tk = Get-SweepTargetKey -SourceTag $SourceTag -ProfileId $ProfileId -ExpectDigest $ExpectDigest `
                      -ManifestSha256 $ManifestSha256 -VolumeGuid $vol.id
            $key = $tk.key
            # R11-1: the current verify_report.checked_at is part of the hit condition, not just of
            # the stored record.
            $hit = Read-SweepBinding -Key $key -CheckedAt $CheckedAt
            if ($hit.ok) {
                $sel = Select-SweepQd -Points $hit.points -PrefetchArm $PrefetchArm -CatalogN $pfN
                $out['points'] = $hit.points
                $out['from_binding'] = $true
                $out['block_bytes'] = $hit.block_bytes
                $out['window_duration_ms'] = $hit.window_duration_ms
                if ($sel.ok) {
                    $out['ok'] = $true
                    $out['qd'] = [int]$sel.qd
                    $out['reason'] = [string]$sel.reason
                    $out['qd_source'] = 'measured-sweep'
                    $out['detail'] = ('stored binding from ' + $hit.swept_at)
                } else {
                    $out['detail'] = 'stored binding carries no valid point'
                }
                Write-Line ('[sweep] reusing the bound sweep record (QD{0}, {1})' -f $out['qd'], $out['detail'])
                Write-Diag -Kind 'SWEEP_BINDING_HIT' -Data @{ key = $key; qd = $out['qd']; reason = $out['reason'] }
                return $out
            }
            $out['detail'] = [string]$hit.reason
        } else {
            $out['detail'] = ('output volume identity unavailable - ' + $vol.reason)
        }

        if ($null -ne $key) {
            if ($script:SweepRuns.ContainsKey($key)) { return $script:SweepRuns[$key] }
        }
        $run = Invoke-QdSweep -OutputDir $OutputDir -ManifestSha256 $ManifestSha256 `
                   -PrefetchArm $PrefetchArm -CatalogN $pfN
        if (-not $run.ok) {
            $out['detail'] = [string]$run.reason
            Write-Line ('[sweep] QD sweep could not run ({0}) -> degraded: QD{1}, conservative default (RS 5)' -f $run.reason, $script:QD_DEGRADED)
            Write-Diag -Kind 'SWEEP_FAILED' -Data @{ reason = $run.reason }
            if ($null -ne $key) { $script:SweepRuns[$key] = $out }
            return $out
        }
        $sel = $run.selection
        $out['points'] = $run.points
        $out['block_bytes'] = $run.block_bytes
        $out['window_duration_ms'] = $run.window_duration_ms
        if (-not $sel.ok) {
            $out['detail'] = 'every swept point failed'
            Write-Line ('[sweep] every swept point failed -> degraded: QD{0}, prefetch OFF (RS 5)' -f $script:QD_DEGRADED)
            Write-Diag -Kind 'SWEEP_ALL_POINTS_FAILED' -Data @{ points = $run.points }
            if ($null -ne $key) { $script:SweepRuns[$key] = $out }
            return $out
        }
        $out['ok'] = $true
        $out['qd'] = [int]$sel.qd
        $out['reason'] = [string]$sel.reason
        $out['qd_source'] = 'measured-sweep'
        $out['detail'] = ('measured in this run ({0} round(s))' -f $run.rounds)
        Write-Line ('[sweep] {0} -> QD{1} ({2})' -f (Format-SweepPoints -Points $run.points), $out['qd'], $out['reason'])
        Write-Diag -Kind 'SWEEP_SELECTED' -Data @{ qd = $out['qd']; reason = $out['reason']; s90 = $sel.s90
                                                    probe_algorithm = $script:SWEEP_PROBE_ALGORITHM
                                                    measurement_method = $script:SWEEP_MEASUREMENT_METHOD
                                                    q_base = $sel.q_base; points = $run.points
                                                    block_bytes = $run.block_bytes; rounds = $run.rounds
                                                    confirm_round = $run.confirm_round
                                                    align_bytes = $run.align_bytes
                                                    physical_sector = $run.physical_sector
                                                    manifest_align = $run.manifest_align }
        # A success is only stored when it is bound to a resolvable volume identity; an unbound
        # record would be reusable on the wrong volume next time (same rule as the scratch record).
        # R11-1 adds the same rule for time: without a usable verify_report.checked_at the entry
        # could never satisfy the freshness condition, so it is not written at all.
        $curChecked = ConvertTo-UtcInstant -Value $CheckedAt
        if ($null -ne $key -and (-not $curChecked.ok)) {
            $out['persist_failed'] = $true
            Write-Line ('[sweep] binding_persist=failed (verify_report.checked_at is {0}); this run uses the measurement in memory.' -f $curChecked.reason)
            $script:SweepRuns[$key] = $out
            return $out
        }
        if ($null -ne $key) {
            $entry = [ordered]@{
                probe_algorithm    = $script:SWEEP_PROBE_ALGORITHM
                measurement_method = $script:SWEEP_MEASUREMENT_METHOD
                source_tag         = $SourceTag
                profile_id         = $ProfileId
                expectation_digest = ([string]$ExpectDigest).ToLowerInvariant()
                manifest_sha256    = ([string]$ManifestSha256).ToLowerInvariant()
                output_volume_guid = [string]$vol.id
                selected_qd        = [int]$sel.qd
                selected_reason    = [string]$sel.reason
                block_bytes        = [long]$run.block_bytes
                window_duration_ms = [int]$run.window_duration_ms
                windows_per_point  = [int]$run.rounds
                points             = @($run.points)
                swept_at           = (Get-Date).ToUniversalTime().ToString('o')
                repack_checked_at  = [string]$CheckedAt
            }
            $save = Write-SweepBinding -Key $key -Entry $entry
            if (-not $save.ok) {
                $out['persist_failed'] = $true
                Write-Line ('[sweep] binding_persist=failed ({0}); this run uses the measurement in memory and the next start re-probes.' -f $save.reason)
            }
        } else {
            $out['persist_failed'] = $true
            Write-Line '[sweep] binding_persist=failed (no output volume identity); this run uses the measurement in memory.'
        }
        if ($null -ne $key) { $script:SweepRuns[$key] = $out }
        return $out
    } catch {
        $out['ok'] = $false
        $out['qd'] = [int]$script:QD_DEGRADED
        $out['qd_source'] = 'conservative-default'
        $out['detail'] = ('sweep fault: ' + $_.Exception.Message)
        Write-Line ('[sweep] QD sweep faulted ({0}) -> degraded: QD{1} (RS 5)' -f $_.Exception.Message, $script:QD_DEGRADED)
        Write-Diag -Kind 'SWEEP_FAULT' -Data @{ reason = $_.Exception.Message }
        return $out
    }
}

function Format-SweepPoints {
    param($Points)
    $parts = @()
    foreach ($qd in $script:SWEEP_QD_POINTS) {
        $txt = 'n/a'
        foreach ($p in @($Points)) {
            if ([int]$p.qd -ne [int]$qd) { continue }
            if ([string]$p.status -ceq 'ok') { $txt = ('{0} MiB/s' -f $p.mibps) } else { $txt = 'failed' }
        }
        $parts += ('QD{0} {1}' -f $qd, $txt)
    }
    return ($parts -join ' | ')
}

# LS 12-1 priority: session/CLI override > stored preset > measured-sweep > conservative-default.
# Preset and CLI values arrive in the same override table, and both are user intent, so both are
# reported as user-override.
function Get-QdSource {
    param($Overrides, $Sweep)
    if ($null -ne $Overrides -and $Overrides.ContainsKey('qd')) { return 'user-override' }
    if ($null -ne $Sweep) { return [string]$Sweep.qd_source }
    return 'conservative-default'
}

# RV 2-4. The QD sweep is a PHYSICAL probe of experts.bin (:4154 opens it), so in virtual it can
# only ever fail - and its failure is not neutral: ProbeOk=$false is exactly what turns a
# catalog-fixed row OFF (:5424), which would silently disable the one thing this preview exists to
# serve. Virtual therefore does not run the sweep and does not consume its verdict; it serves the
# shape the f3c GO run actually measured (io_qd_total = 8) with the profile's own catalog prefetch
# policy untouched - catalog-fixed rows stay ON with their K/N, off/hold rows stay OFF.
#
# qd_source is 'virtual-pinned', never 'measured-sweep': no sweep ran in this process and the
# status screen may not claim one. points stays empty for the same reason, so the "measured io"
# row honestly reports that it has no point at this QD.
function New-VirtualPinnedQd {
    Write-Line ('[sweep] not run in mode=virtual (no experts.bin to probe); QD pinned to {0}, the measured shape.' -f `
                $script:VIRTUAL_PINNED_QD)
    Write-Diag -Kind 'SWEEP_SKIPPED_VIRTUAL' -Data @{ qd = [int]$script:VIRTUAL_PINNED_QD
                                                       reason = 'mode=virtual has no experts.bin to probe' }
    return @{ ok = $true; qd = [int]$script:VIRTUAL_PINNED_QD; qd_source = 'virtual-pinned'
              reason = 'mode=virtual pinned shape'; points = @(); from_binding = $false
              persist_failed = $false; detail = 'sweep not run (mode=virtual)'
              block_bytes = $null; window_duration_ms = $null }
}

# endregion

# ============================================================================
# region 10. INSTANCE / OUTPUT / PORT LOCKS (LS 1-8, LS 2)
# ============================================================================

$script:HeldMutexes = @()
$script:HeldFileLocks = @()

function Acquire-NamedMutex {
    param([string] $Name)
    # Global\ first (moe_serve.ps1:804 precedent). A filtered token has no SeCreateGlobalPrivilege,
    # so fall back to Local\ and record the downgrade rather than failing the launch.
    foreach ($prefix in @('Global\', 'Local\')) {
        try {
            $created = $false
            $mx = New-Object System.Threading.Mutex($true, ($prefix + $Name), [ref] $created)
            if (-not $created) {
                try { $mx.Dispose() } catch { }
                return @{ ok = $false; reason = ('lock already held: ' + $prefix + $Name) }
            }
            if ($prefix -eq 'Local\') { Write-Diag -Kind 'MUTEX_SCOPE_DOWNGRADE' -Data @{ name = $Name } }
            $script:HeldMutexes += $mx
            return @{ ok = $true; mutex = $mx; name = ($prefix + $Name) }
        } catch [System.UnauthorizedAccessException] {
            continue
        } catch {
            return @{ ok = $false; reason = ('mutex error: ' + $_.Exception.Message) }
        }
    }
    return @{ ok = $false; reason = 'mutex could not be created in either scope' }
}

function Acquire-FileLock {
    param([string] $Path)
    try {
        $fs = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::OpenOrCreate,
                  [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        $script:HeldFileLocks += $fs
        return @{ ok = $true; stream = $fs }
    } catch {
        return @{ ok = $false; reason = ('exclusive output lock unavailable: ' + $_.Exception.Message) }
    }
}

function Release-AllLocks {
    foreach ($fs in $script:HeldFileLocks) { try { $fs.Dispose() } catch { } }
    $script:HeldFileLocks = @()
    foreach ($mx in $script:HeldMutexes) { try { $mx.ReleaseMutex() } catch { } ; try { $mx.Dispose() } catch { } }
    $script:HeldMutexes = @()
    $script:PortMutex = $null
}

function Get-SafeToken {
    param([string] $Text)
    $sha = [System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Text))
    return (([System.BitConverter]::ToString($sha) -replace '-', '').ToLowerInvariant().Substring(0, 16))
}

# R1-1: two separate lock stages.
#   (1) instance + profile + output lock - taken before anything destructive (.partial cleanup,
#       repack) can touch the output directory.
#   (2) effective-port lock - taken only once the port is final, i.e. after preset + CLI have been
#       bound and the effective config has been built.
$script:PortMutex = $null

# LS OA-1: the profile lock is taken separately because the arch-template path does not know its
# profile id until the derive-plan has run, and that plan must already be under the instance and
# output locks (it is read-only, but it decides what the following repack will write there). The
# catalog path is unchanged: it passes -ProfileId and both locks are taken in the same call, in the
# same order, as before.
function Add-ProfileLock {
    param([string] $ProfileId)
    $r = Acquire-NamedMutex -Name ('moe_direct_launcher_profile_' + (Get-SafeToken -Text $ProfileId))
    if (-not $r.ok) { Stop-Launcher 'fail_instance_lock' ('profile lock: ' + $r.reason) }
}

function Acquire-LauncherLocks {
    param([string] $ProfileId, [string] $OutputDir)
    $r = Acquire-NamedMutex -Name 'moe_direct_launcher'
    if (-not $r.ok) { Stop-Launcher 'fail_instance_lock' ('single-instance: ' + $r.reason) }
    if ($ProfileId) { Add-ProfileLock -ProfileId $ProfileId }
    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        try { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }
        catch { Stop-Launcher 'fail_resource' ('cannot create output directory: ' + $_.Exception.Message) }
    }
    $r = Acquire-FileLock -Path (Join-Path $OutputDir '.moe-launcher.lock')
    if (-not $r.ok) { Stop-Launcher 'fail_instance_lock' ('output lock: ' + $r.reason) }
    Write-Diag -Kind 'LOCKS_ACQUIRED' -Data @{ profile = $ProfileId; out = $OutputDir }
}

function Set-EffectivePortLock {
    param([int] $PortNumber)
    if ($null -ne $script:PortMutex -and [int]$script:PortMutex.port -eq $PortNumber) { return }
    if ($null -ne $script:PortMutex) {
        # the user changed the port in the custom loop: release the old reservation first
        try { $script:PortMutex.mutex.ReleaseMutex() } catch { }
        try { $script:PortMutex.mutex.Dispose() } catch { }
        $script:HeldMutexes = @($script:HeldMutexes | Where-Object { $_ -ne $script:PortMutex.mutex })
        $script:PortMutex = $null
    }
    $r = Acquire-NamedMutex -Name ('moe_direct_launcher_port_' + $PortNumber)
    if (-not $r.ok) { Stop-Launcher 'fail_instance_lock' ('effective port lock: ' + $r.reason) }
    $script:PortMutex = @{ mutex = $r.mutex; port = $PortNumber }
    Write-Diag -Kind 'PORT_LOCK_ACQUIRED' -Data @{ port = $PortNumber }
}

# endregion

# ============================================================================
# region 11. .partial DETECTION AND CLEANUP (LS 2, LS 3 item 1)
# ============================================================================

# LS 3 item 1 / R1-3: only FileNotFound proves absence. Any other query error is a hard stop,
# never "absent" - identical to the C++ GetLastError()==ERROR_FILE_NOT_FOUND contract
# (moedirect-v2-b10057.patch:7327-7338). FileInfo.Exists cannot be used: it reports false for
# permission and IO errors too, which silently converts "cannot prove absence" into "absent".
function Get-FileAbsenceState {
    param([string] $Path)
    $attrs = [MoeLauncher.Native]::GetFileAttributesW($Path)
    if ($attrs -ne $script:INVALID_FILE_ATTRIBUTES) { return @{ state = 'present'; path = $Path } }
    $gle = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    if ($gle -eq $script:ERROR_FILE_NOT_FOUND) { return @{ state = 'absent'; path = $Path; gle = $gle } }
    if ($gle -eq $script:ERROR_PATH_NOT_FOUND) {
        # A component of the path is missing. That proves the file cannot exist, but only if the
        # directory's own absence is itself provable by the same rule; anything else stays unknown.
        $dir = [System.IO.Path]::GetDirectoryName($Path)
        $dattrs = [MoeLauncher.Native]::GetFileAttributesW($dir)
        if ($dattrs -eq $script:INVALID_FILE_ATTRIBUTES) {
            $dgle = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
            if ($dgle -eq $script:ERROR_FILE_NOT_FOUND -or $dgle -eq $script:ERROR_PATH_NOT_FOUND) {
                return @{ state = 'absent'; path = $Path; gle = $gle; dir_gle = $dgle }
            }
        }
        return @{ state = 'unknown'; path = $Path; gle = $gle; reason = ("directory query inconclusive (GetLastError=" + $gle + ")") }
    }
    return @{ state = 'unknown'; path = $Path; gle = $gle
              reason = ("GetFileAttributesW failed (GetLastError=" + $gle + ") - absence not provable") }
}

function Get-PartialMarkerState {
    param([string] $OutputDir)
    return (Get-FileAbsenceState -Path (Join-Path $OutputDir 'experts.bin.partial'))
}

# LS 2: the four artifacts are removed together. Deleting only the marker would not restart the
# repacker (repack_experts.py:1030 aborts while experts.bin / manifest.json survive without --force).
function Invoke-PartialCleanup {
    param([string] $OutputDir)
    Write-Line ''
    Write-Line '[partial] An interrupted repack was detected in this output directory.'
    Write-Line '          v1 has no resume: the repack restarts from the beginning.'
    Write-Line '          The following artifacts will be DELETED before restarting:'
    foreach ($n in $script:PARTIAL_DELETE_SET) { Write-Line ('            - ' + (Join-Path $OutputDir $n)) }
    if (-not (Confirm-User -Question 'Delete these artifacts and restart the repack? [y/N] ')) {
        Stop-Launcher 'cancelled_user' 'user declined .partial cleanup'
    }
    $failed = @()
    foreach ($n in $script:PARTIAL_DELETE_SET) {
        $p = Join-Path $OutputDir $n
        try {
            if (Test-Path -LiteralPath $p -PathType Leaf) { Remove-Item -LiteralPath $p -Force -ErrorAction Stop }
        } catch {
            $failed += ($n + ': ' + $_.Exception.Message)
            continue
        }
        try {
            $fi = New-Object System.IO.FileInfo($p); $fi.Refresh()
            if ($fi.Exists) { $failed += ($n + ': still present after delete') }
        } catch {
            $failed += ($n + ': absence check failed - ' + $_.Exception.Message)
        }
    }
    if ($failed.Count -gt 0) {
        Write-Diag -Kind 'PARTIAL_CLEANUP_FAILED' -Data @{ failed = $failed }
        Stop-Launcher 'fail_partial_cleanup' ('.partial cleanup failed: ' + ($failed -join '; '))
    }
    Write-Diag -Kind 'PARTIAL_CLEANUP_OK' -Data @{ out = $OutputDir; deleted = $script:PARTIAL_DELETE_SET }
    Write-Line '[partial] Cleanup complete; repack will start from the beginning.'
}

# ---------------------------------------------------------------------------------------------
# RV 1-1 [3] - the same recovery for a VIRTUAL output directory, and deliberately not the same
# function: the file SET is different (VIRTUAL_PARTIAL_DELETE_SET, not PARTIAL_DELETE_SET) and the
# reachable state is different. The virtual repacker promotes plan_report.json first and
# manifest.json.partial -> manifest.json second (repack_experts.py:3997); an interruption between
# those two atomic replacements leaves a report without a manifest, which the repacker then refuses
# to overwrite without --force (:3915). Without this transition a TRUSTED producer's ordinary
# interruption would have no successful continuation at all.
#
# The confirmation UX is inherited unchanged from Confirm-User: -AssumeYes proceeds, -NonInteractive
# alone and -AssumeNo answer no and stop at cancelled_user with nothing deleted.
# --force is NOT used: the artifacts are removed here, and the repacker then runs on a clean
# directory exactly as it does on a first run.
# ---------------------------------------------------------------------------------------------
function Invoke-VirtualPartialCleanup {
    param([string] $OutputDir)
    Write-Line ''
    Write-Line '[partial] An interrupted VIRTUAL repack was detected in this output directory.'
    Write-Line '          The plan is rebuilt from the beginning; 0 bytes of expert data are moved.'
    Write-Line '          The following incomplete artifacts will be DELETED before restarting:'
    foreach ($n in $script:VIRTUAL_PARTIAL_DELETE_SET) { Write-Line ('            - ' + (Join-Path $OutputDir $n)) }
    if (-not (Confirm-User -Question 'Delete these artifacts and restart the plan? [y/N] ')) {
        Stop-Launcher 'cancelled_user' 'user declined the incomplete virtual artifact cleanup'
    }
    $failed = @()
    foreach ($n in $script:VIRTUAL_PARTIAL_DELETE_SET) {
        $p = Join-Path $OutputDir $n
        try {
            if (Test-Path -LiteralPath $p -PathType Leaf) { Remove-Item -LiteralPath $p -Force -ErrorAction Stop }
        } catch {
            $failed += ($n + ': ' + $_.Exception.Message)
            continue
        }
        try {
            $fi = New-Object System.IO.FileInfo($p); $fi.Refresh()
            if ($fi.Exists) { $failed += ($n + ': still present after delete') }
        } catch {
            $failed += ($n + ': absence check failed - ' + $_.Exception.Message)
        }
    }
    if ($failed.Count -gt 0) {
        Write-Diag -Kind 'VIRTUAL_PARTIAL_CLEANUP_FAILED' -Data @{ failed = $failed }
        Stop-Launcher 'fail_partial_cleanup' ('virtual .partial cleanup failed: ' + ($failed -join '; '))
    }
    Write-Diag -Kind 'VIRTUAL_PARTIAL_CLEANUP_OK' -Data @{ out = $OutputDir; deleted = $script:VIRTUAL_PARTIAL_DELETE_SET }
    Write-Line '[partial] Cleanup complete; the virtual plan will start from the beginning.'
}

# ---------------------------------------------------------------------------------------------
# P4 2.5 (b) - stale repack artifacts on the identity-mismatch path.
#
# The default output directory is "<model folder>\repack", so the original model and a same-shaped
# file sitting beside it SHARE one output directory. A mismatch run keeps the catalog row, which
# means it also keeps that row's profile_id / expect digest / lock id - and the artifact-reuse test
# is "do the three files exist", while the 7-item gate checks the reference lock and the manifest's
# own digest. None of those can answer "was this experts.bin built from THESE bytes", so without
# this step a mismatch run could serve the ORIGINAL model's experts.bin as if it had been
# copy-verified from the file the user actually named.
#
# INVESTIGATED, not assumed: the repacker's manifest carries no source digest at all -
# build_manifest writes sources[] as {index, path, bytes, mtime, gguf_version, alignment,
# data_start} (bench/repack/repack_experts.py:1326-1331), and every sha256 in that file is an
# EXPECT digest, an inventory digest or the manifest's own digest - never a source shard hash.
# So there is nothing to compare the current shard SHAs against, and the disposition is the
# second of the two the spec allows: on a mismatch run the artifacts are treated as stale
# regardless of their presence, and the repack is re-run from the current bytes.
#
# Deliberately scoped to the mismatch verdict: the pinned and unpinned paths keep their reuse
# decision, their cost and their failure modes exactly as they were, and gain no new hashing -
# the identity answer this depends on was already computed by Resolve-ProfileSelection.
#
# The deletion is NOT confirmed separately. It happens after the ordinary "Proceed with the repack
# now?" confirmation, which the plan block warns beforehand (one confirmation point, not two -
# the same rule the derived path follows). Failure reuses fail_partial_cleanup; no new status.
# ---------------------------------------------------------------------------------------------
function Remove-StaleRepackArtifacts {
    param([string] $OutputDir)
    $failed = @()
    foreach ($n in $script:PARTIAL_DELETE_SET) {
        $p = Join-Path $OutputDir $n
        try {
            if (Test-Path -LiteralPath $p -PathType Leaf) { Remove-Item -LiteralPath $p -Force -ErrorAction Stop }
        } catch {
            $failed += ($n + ': ' + $_.Exception.Message)
            continue
        }
        try {
            $fi = New-Object System.IO.FileInfo($p); $fi.Refresh()
            if ($fi.Exists) { $failed += ($n + ': still present after delete') }
        } catch {
            $failed += ($n + ': absence check failed - ' + $_.Exception.Message)
        }
    }
    if ($failed.Count -gt 0) {
        Write-Diag -Kind 'STALE_ARTIFACT_CLEANUP_FAILED' -Data @{ failed = $failed }
        Stop-Launcher 'fail_partial_cleanup' ('stale repack artifact cleanup failed: ' + ($failed -join '; '))
    }
    Write-Diag -Kind 'STALE_ARTIFACT_CLEANUP_OK' -Data @{ out = $OutputDir; deleted = $script:PARTIAL_DELETE_SET
                                                           reason = 'identity_mismatch' }
    Write-Line '[stale] Previous repack artifacts deleted; the repack restarts from the current file.'
}

# endregion

# ============================================================================
# region 12. 7-ITEM VERIFY GATE (LS 3) - isomorphic with the C++ consumer
#   1st source: bench/moe-direct/repro/moedirect-v2-b10057.patch:3738-3830 (read_verify_report_gate)
#               and :7299-7360 (seal reference_lock / partial / manifest_sha256 binding)
# ============================================================================

function Get-LastNonEmptyRecordText {
    param([string] $Path)
    $b = Read-FileBytesStrict -Path $Path
    if (-not $b.ok) { return @{ ok = $false; reason = $b.reason } }
    $t = ConvertFrom-Utf8Strict -Bytes $b.bytes
    if (-not $t.ok) { return @{ ok = $false; reason = $t.reason } }
    $lines = $t.text -split "`n"
    $last = $null
    foreach ($ln in $lines) {
        $trim = $ln.TrimEnd("`r", "`n", ' ', "`t")
        if ($trim.Length -gt 0) { $last = $trim }
    }
    # A normal trailing newline leaves an empty final element and is skipped above; a truncated
    # last record stays non-empty and must be rejected by strict parse (no earlier-line fallback).
    if ($null -eq $last) { return @{ ok = $false; reason = 'verify_report.json has no non-empty record' } }
    return @{ ok = $true; text = $last }
}

function Assert-VerifyGate {
    param([string] $OutputDir, [string] $ProfileId, [string] $ExpectSha)

    # (1) .partial absence - FileNotFound only
    $pm = Get-PartialMarkerState -OutputDir $OutputDir
    if ($pm.state -eq 'present') { Stop-Launcher 'fail_gate_verify' 'gate 1: experts.bin.partial present (interrupted or unverified artifact)' }
    if ($pm.state -eq 'unknown') { Stop-Launcher 'fail_gate_verify' ('gate 1: experts.bin.partial absence not provable - ' + $pm.reason) }

    $reportPath  = Join-Path $OutputDir 'verify_report.json'
    $manifestPath = Join-Path $OutputDir 'manifest.json'
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf))   { Stop-Launcher 'fail_gate_verify' 'gate 2: verify_report.json missing' }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { Stop-Launcher 'fail_gate_verify' 'gate 6: manifest.json missing' }

    # (2) last non-empty physical record, strict parse
    $lr = Get-LastNonEmptyRecordText -Path $reportPath
    if (-not $lr.ok) { Stop-Launcher 'fail_gate_verify' ('gate 2: ' + $lr.reason) }
    $pr = ConvertFrom-JsonStrict -Text $lr.text
    if (-not $pr.ok) { Stop-Launcher 'fail_gate_verify' ('gate 2: last record strict parse failed - ' + $pr.reason) }
    $rec = $pr.value

    # (3) pass must be JSON boolean true
    if (-not (Test-JsonHas -Obj $rec -Name 'pass')) { Stop-Launcher 'fail_gate_verify' 'gate 3: pass key missing' }
    $passVal = Get-JsonValue -Obj $rec -Name 'pass'
    if (-not (Test-JsonBoolean $passVal))    { Stop-Launcher 'fail_gate_verify' 'gate 3: pass is not a JSON boolean' }
    if (-not (Test-JsonBooleanTrue $passVal)) { Stop-Launcher 'fail_gate_verify' 'gate 3: pass is false' }

    # (4) three counts present, non-negative integers, all equal
    $counts = @{}
    foreach ($k in @('pairs_pass', 'pairs_total', 'expected_pairs')) {
        if (-not (Test-JsonHas -Obj $rec -Name $k)) { Stop-Launcher 'fail_gate_verify' ('gate 4: ' + $k + ' missing') }
        $v = Get-JsonValue -Obj $rec -Name $k
        if (-not (Test-JsonNonNegativeInteger $v)) { Stop-Launcher 'fail_gate_verify' ('gate 4: ' + $k + ' is not a non-negative integer') }
        $counts[$k] = [long]$v
    }
    if ($counts['pairs_pass'] -ne $counts['pairs_total'] -or $counts['pairs_total'] -ne $counts['expected_pairs']) {
        Stop-Launcher 'fail_gate_verify' ('gate 4: count equality broken pairs_pass=' + $counts['pairs_pass'] +
            ' pairs_total=' + $counts['pairs_total'] + ' expected_pairs=' + $counts['expected_pairs'])
    }

    # (5) three defect keys present, JSON arrays, length 0 (missing key != empty array)
    foreach ($k in @('problems', 'failures', 'padding_failures')) {
        if (-not (Test-JsonHas -Obj $rec -Name $k)) { Stop-Launcher 'fail_gate_verify' ('gate 5: ' + $k + ' missing (missing key is not an empty array)') }
        $v = Get-JsonValue -Obj $rec -Name $k
        if (-not (Test-JsonArray $v))      { Stop-Launcher 'fail_gate_verify' ('gate 5: ' + $k + ' is not a JSON array') }
        if (-not (Test-JsonEmptyArray $v)) { Stop-Launcher 'fail_gate_verify' ('gate 5: ' + $k + ' is not empty') }
    }

    # (6) reference_lock three-way equality
    $mr = Read-JsonFileStrict -Path $manifestPath
    if (-not $mr.ok) { Stop-Launcher 'fail_gate_verify' ('gate 6: manifest.json unreadable - ' + $mr.reason) }
    $mLock = Get-JsonValue -Obj $mr.value -Name 'reference_lock'
    $rLock = Get-JsonValue -Obj $rec -Name 'reference_lock'
    foreach ($pair in @(@('manifest', $mLock), @('verify_report', $rLock))) {
        $lock = $pair[1]
        if ($null -eq $lock) { Stop-Launcher 'fail_gate_verify' ('gate 6: ' + $pair[0] + '.reference_lock missing') }
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $lock -Name 'profile_id'))) {
            Stop-Launcher 'fail_gate_verify' ('gate 6: ' + $pair[0] + '.reference_lock.profile_id missing or not a string')
        }
        if (-not (Test-Sha256Hex (Get-JsonValue -Obj $lock -Name 'expect_sha256'))) {
            Stop-Launcher 'fail_gate_verify' ('gate 6: ' + $pair[0] + '.reference_lock.expect_sha256 missing or not 64 hex')
        }
    }
    $mPid = [string](Get-JsonValue -Obj $mLock -Name 'profile_id')
    $rPid = [string](Get-JsonValue -Obj $rLock -Name 'profile_id')
    $mSha = ([string](Get-JsonValue -Obj $mLock -Name 'expect_sha256')).ToLowerInvariant()
    $rSha = ([string](Get-JsonValue -Obj $rLock -Name 'expect_sha256')).ToLowerInvariant()
    $cSha = $ExpectSha.ToLowerInvariant()
    if (-not ($mPid -ceq $ProfileId -and $rPid -ceq $ProfileId)) {
        Stop-Launcher 'fail_gate_verify' ("gate 6: reference_lock.profile_id three-way mismatch (manifest=" + $mPid + " report=" + $rPid + " selected=" + $ProfileId + ")")
    }
    if (-not ($mSha -eq $cSha -and $rSha -eq $cSha)) {
        Stop-Launcher 'fail_gate_verify' 'gate 6: reference_lock.expect_sha256 three-way mismatch against the catalog expect hash'
    }

    # (7) cache key binding: manifest_sha256 is 64 hex AND equals the real manifest.json bytes
    if (-not (Test-JsonHas -Obj $rec -Name 'manifest_sha256')) {
        Stop-Launcher 'fail_gate_verify' 'gate 7: manifest_sha256 missing (report not bound to a cache key)'
    }
    $repSha = Get-JsonValue -Obj $rec -Name 'manifest_sha256'
    if (-not (Test-Sha256Hex $repSha)) { Stop-Launcher 'fail_gate_verify' 'gate 7: manifest_sha256 is not a 64 hex string' }
    $realSha = Get-FileSha256Lower -Path $manifestPath
    if (-not $realSha.ok) { Stop-Launcher 'fail_gate_verify' ('gate 7: manifest.json re-hash failed - ' + $realSha.reason) }
    if ($realSha.sha -ne ([string]$repSha).ToLowerInvariant()) {
        Stop-Launcher 'fail_gate_verify' 'gate 7: manifest_sha256 != real manifest.json bytes (report belongs to a different artifact)'
    }

    Write-Diag -Kind 'VERIFY_GATE_OK' -Data @{ out = $OutputDir; profile = $ProfileId;
        pairs = $counts['pairs_total']; manifest_sha256 = $realSha.sha }
    Write-Line ('[gate] verify gate PASS (7/7) - pairs {0}, manifest {1}' -f $counts['pairs_total'], $realSha.sha.Substring(0, 12))
    # checked_at is passed through for LS 12-4 freshness bookkeeping only (the repack completion
    # time is the REPORT's checked_at, not the manifest's creation time). It is not a gate input:
    # no gate branch above reads it, and its absence cannot change this function's verdict.
    return @{ pairs = $counts['pairs_total']; manifest_sha256 = $realSha.sha
              checked_at = [string](Get-JsonValue -Obj $rec -Name 'checked_at') }
}

# endregion

# ============================================================================
# region 12b. VIRTUAL MODE DETECTION + PLAN GATE (RV 1 / RV 1-1 / RV 2-3)
#   1st source: the engine's own consumers -
#     detect_manifest_mode()    ggml-moe-direct.cpp:2110-2137
#     read_plan_report_gate()   ggml-moe-direct.cpp:2510-2549 (+ the caller equality at :8955)
#   Nothing here relaxes either of them; where the two differ this side is the stricter one.
# ============================================================================

# RV 1. The mode truth table, mirrored, not paraphrased:
#     "2.0" + NO mode key                -> BIN
#     "3.0" + mode == "virtual" (string) -> VIRTUAL
#     everything else                    -> fail-close
# Key PRESENCE is the test, never the value: "2.0" with a mode key of ANY value (null, "bin",
# "virtual") is a fail-close, and so is "3.0" with the key absent. Test-JsonHas is what makes that
# distinction - a null-value check would collapse "absent" and "present but null" into one answer.
function Get-ManifestMode {
    param([string] $ManifestPath)
    $r = Read-JsonFileStrict -Path $ManifestPath
    if (-not $r.ok) { return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = $r.reason } }
    $obj = $r.value
    if (-not ($obj -is [System.Management.Automation.PSCustomObject])) {
        return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = 'manifest root is not a JSON object' }
    }
    if (-not (Test-JsonHas -Obj $obj -Name 'schema_version')) {
        return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = 'schema_version key missing' }
    }
    $sv = Get-JsonValue -Obj $obj -Name 'schema_version'
    if (-not (Test-JsonNonEmptyString $sv)) {
        return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = 'schema_version is not a JSON string' }
    }
    $svText = [string]$sv
    $hasMode = Test-JsonHas -Obj $obj -Name 'mode'
    if ($svText -ceq '2.0') {
        if ($hasMode) {
            return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = 'schema_version 2.0 carries a mode key' }
        }
        return @{ mode = $script:MANIFEST_MODE_BIN; reason = $null }
    }
    if ($svText -ceq '3.0') {
        if (-not $hasMode) {
            return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = 'schema_version 3.0 has no mode key' }
        }
        $mv = Get-JsonValue -Obj $obj -Name 'mode'
        if (-not (Test-JsonNonEmptyString $mv)) {
            return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = 'mode is not a JSON string' }
        }
        if ([string]$mv -ceq $script:MANIFEST_MODE_VIRTUAL) {
            return @{ mode = $script:MANIFEST_MODE_VIRTUAL; reason = $null }
        }
        return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = ("mode is '" + [string]$mv + "'") }
    }
    return @{ mode = $script:MANIFEST_MODE_UNRECOGNIZED; reason = ("schema_version is '" + $svText + "'") }
}

# RV 1 fail-close status. One reason string for the whole unrecognized branch, kept separate from
# the mode-MISMATCH reason below: "this manifest cannot be read at all" and "this manifest is the
# other mode" are different verdicts and may not be spelled the same way.
function Stop-UnrecognizedManifest {
    param($ModeResult)
    $why = $script:MANIFEST_MODE_UNRECOGNIZED_REASON
    if ($ModeResult -and $ModeResult.reason) { $why = $why + ' (' + [string]$ModeResult.reason + ')' }
    Stop-Launcher 'fail_gate_verify' $why
}

function Stop-ModeMismatch {
    param([string] $Existing, [string] $Requested)
    Stop-Launcher 'fail_gate_verify' ('mode: existing artifacts are ' + $Existing + ', requested ' + $Requested)
}

# RV 2-3. The virtual counterpart of the 7-item gate. Same discipline, different evidence: there is
# no experts.bin to bind, so the binding is plan_report.json <-> manifest.json <-> the catalog.
# Every item fails closed and the order is the contract (the first failure is the reason reported).
#
# What this gate does NOT do: re-run validate_manifest_v3. The engine checks every one of those
# invariants before it acquires a single resource (:2284, :2404, :8888), so the terminal fail-close
# already exists. The launcher owns exactly two things - the type/positivity of the fields IT
# consumes before the engine runs, and the plan-report binding below.
function Assert-VirtualPlanGate {
    param([string] $OutputDir, [string] $ProfileId, [string] $ExpectSha)

    $manifestPath = Join-Path $OutputDir 'manifest.json'
    $reportPath   = Join-Path $OutputDir 'plan_report.json'

    # (v1) no bin artifact may sit in a virtual output directory. FileNotFound is the ONLY proof of
    # absence - a permission or IO error is "cannot prove absent", never "absent"
    # (repack_experts.py:3880, engine :8913, and the LS 3 item 1 rule Get-FileAbsenceState encodes).
    foreach ($n in @('experts.bin', 'experts.bin.partial')) {
        $st = Get-FileAbsenceState -Path (Join-Path $OutputDir $n)
        if ($st.state -eq 'present') {
            Stop-Launcher 'fail_gate_verify' ('vgate 1: ' + $n + ' present in a virtual output directory')
        }
        if ($st.state -ne 'absent') {
            Stop-Launcher 'fail_gate_verify' ('vgate 1: ' + $n + ' absence not provable - ' + $st.reason)
        }
    }

    # (v2) plan_report.json is a SINGLE JSON object, not the JSONL verify_report is - so it is read
    # whole and parsed strictly, with no last-record rule.
    $pr = Read-JsonFileStrict -Path $reportPath
    if (-not $pr.ok) { Stop-Launcher 'fail_gate_verify' ('vgate 2: plan_report.json - ' + $pr.reason) }
    $rec = $pr.value
    if (-not ($rec -is [System.Management.Automation.PSCustomObject])) {
        Stop-Launcher 'fail_gate_verify' 'vgate 2: plan_report.json root is not a JSON object'
    }

    # (v3) the report says which mode produced it
    if (-not (Test-JsonHas -Obj $rec -Name 'mode')) { Stop-Launcher 'fail_gate_verify' 'vgate 3: mode key missing' }
    $mv = Get-JsonValue -Obj $rec -Name 'mode'
    if (-not (Test-JsonNonEmptyString $mv))         { Stop-Launcher 'fail_gate_verify' 'vgate 3: mode is not a JSON string' }
    if ([string]$mv -cne $script:MANIFEST_MODE_VIRTUAL) {
        Stop-Launcher 'fail_gate_verify' ("vgate 3: plan_report.mode is '" + [string]$mv + "', not 'virtual'")
    }

    # (v4) pass must be JSON boolean true
    if (-not (Test-JsonHas -Obj $rec -Name 'pass')) { Stop-Launcher 'fail_gate_verify' 'vgate 4: pass key missing' }
    $passVal = Get-JsonValue -Obj $rec -Name 'pass'
    if (-not (Test-JsonBoolean $passVal))     { Stop-Launcher 'fail_gate_verify' 'vgate 4: pass is not a JSON boolean' }
    if (-not (Test-JsonBooleanTrue $passVal)) { Stop-Launcher 'fail_gate_verify' 'vgate 4: pass is false' }

    # (v5) problems present, a JSON array, and empty (a missing key is NOT an empty array)
    if (-not (Test-JsonHas -Obj $rec -Name 'problems')) {
        Stop-Launcher 'fail_gate_verify' 'vgate 5: problems missing (missing key is not an empty array)'
    }
    $probs = Get-JsonValue -Obj $rec -Name 'problems'
    if (-not (Test-JsonArray $probs))      { Stop-Launcher 'fail_gate_verify' 'vgate 5: problems is not a JSON array' }
    if (-not (Test-JsonEmptyArray $probs)) { Stop-Launcher 'fail_gate_verify' 'vgate 5: problems is not empty' }

    # (v6) cache-key binding: the report names the manifest it belongs to, and that name is the
    # real bytes on disk - not a value copied out of the manifest itself.
    if (-not (Test-JsonHas -Obj $rec -Name 'manifest_sha256')) {
        Stop-Launcher 'fail_gate_verify' 'vgate 6: manifest_sha256 missing (report not bound to a manifest)'
    }
    $repSha = Get-JsonValue -Obj $rec -Name 'manifest_sha256'
    if (-not (Test-Sha256Hex $repSha)) { Stop-Launcher 'fail_gate_verify' 'vgate 6: manifest_sha256 is not a 64 hex string' }
    $realSha = Get-FileSha256Lower -Path $manifestPath
    if (-not $realSha.ok) { Stop-Launcher 'fail_gate_verify' ('vgate 6: manifest.json re-hash failed - ' + $realSha.reason) }
    if ($realSha.sha -ne ([string]$repSha).ToLowerInvariant()) {
        Stop-Launcher 'fail_gate_verify' 'vgate 6: manifest_sha256 != real manifest.json bytes (report belongs to a different artifact)'
    }

    # (v7) reference_lock three-way: manifest, plan report and the catalog say the same thing
    $mr = Read-JsonFileStrict -Path $manifestPath
    if (-not $mr.ok) { Stop-Launcher 'fail_gate_verify' ('vgate 7: manifest.json unreadable - ' + $mr.reason) }
    $mLock = Get-JsonValue -Obj $mr.value -Name 'reference_lock'
    $rLock = Get-JsonValue -Obj $rec -Name 'reference_lock'
    foreach ($pair in @(@('manifest', $mLock), @('plan_report', $rLock))) {
        $lock = $pair[1]
        if ($null -eq $lock) { Stop-Launcher 'fail_gate_verify' ('vgate 7: ' + $pair[0] + '.reference_lock missing') }
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $lock -Name 'profile_id'))) {
            Stop-Launcher 'fail_gate_verify' ('vgate 7: ' + $pair[0] + '.reference_lock.profile_id missing or not a string')
        }
        if (-not (Test-Sha256Hex (Get-JsonValue -Obj $lock -Name 'expect_sha256'))) {
            Stop-Launcher 'fail_gate_verify' ('vgate 7: ' + $pair[0] + '.reference_lock.expect_sha256 missing or not 64 hex')
        }
    }
    $mPid = [string](Get-JsonValue -Obj $mLock -Name 'profile_id')
    $rPid = [string](Get-JsonValue -Obj $rLock -Name 'profile_id')
    $mSha = ([string](Get-JsonValue -Obj $mLock -Name 'expect_sha256')).ToLowerInvariant()
    $rSha = ([string](Get-JsonValue -Obj $rLock -Name 'expect_sha256')).ToLowerInvariant()
    $cSha = $ExpectSha.ToLowerInvariant()
    if (-not ($mPid -ceq $ProfileId -and $rPid -ceq $ProfileId)) {
        Stop-Launcher 'fail_gate_verify' ("vgate 7: reference_lock.profile_id three-way mismatch (manifest=" + $mPid +
            " plan_report=" + $rPid + " selected=" + $ProfileId + ")")
    }
    if (-not ($mSha -eq $cSha -and $rSha -eq $cSha)) {
        Stop-Launcher 'fail_gate_verify' 'vgate 7: reference_lock.expect_sha256 three-way mismatch against the catalog expect hash'
    }

    # (v8) cardinality: a non-negative integer AND the same count the manifest's own totals state.
    # The equality is the load-bearing half - the engine's caller performs it (:8955), and a report
    # whose record count disagrees with its manifest describes a different plan.
    $card = Get-JsonValue -Obj $rec -Name 'cardinality'
    if ($null -eq $card) { Stop-Launcher 'fail_gate_verify' 'vgate 8: cardinality missing' }
    if (-not (Test-JsonHas -Obj $card -Name 'records')) { Stop-Launcher 'fail_gate_verify' 'vgate 8: cardinality.records missing' }
    $recs = Get-JsonValue -Obj $card -Name 'records'
    if (-not (Test-JsonNonNegativeInteger $recs)) {
        Stop-Launcher 'fail_gate_verify' 'vgate 8: cardinality.records is not a non-negative integer'
    }
    $totals = Get-JsonValue -Obj $mr.value -Name 'totals'
    if ($null -eq $totals) { Stop-Launcher 'fail_gate_verify' 'vgate 8: manifest totals missing' }
    if (-not (Test-JsonHas -Obj $totals -Name 'n_records')) { Stop-Launcher 'fail_gate_verify' 'vgate 8: manifest totals.n_records missing' }
    $nrec = Get-JsonValue -Obj $totals -Name 'n_records'
    if (-not (Test-JsonNonNegativeInteger $nrec)) {
        Stop-Launcher 'fail_gate_verify' 'vgate 8: manifest totals.n_records is not a non-negative integer'
    }
    if ([long]$recs -ne [long]$nrec) {
        Stop-Launcher 'fail_gate_verify' ('vgate 8: cardinality.records(' + [long]$recs +
            ') != manifest totals.n_records(' + [long]$nrec + ')')
    }

    Write-Diag -Kind 'VIRTUAL_PLAN_GATE_OK' -Data @{ out = $OutputDir; profile = $ProfileId; mode = $script:MANIFEST_MODE_VIRTUAL
        records = [long]$recs; manifest_sha256 = $realSha.sha }
    Write-Line ('[gate] virtual plan gate PASS (8/8) - mode=virtual, records {0}, manifest {1}' -f `
                ([long]$recs), $realSha.sha.Substring(0, 12))
    # Same ABI as Assert-VerifyGate's return, minus the pairs count that has no virtual meaning:
    # the callers consume manifest_sha256 (warmstart binding, smoke item 5) and checked_at.
    return @{ records = [long]$recs; manifest_sha256 = $realSha.sha
              checked_at = [string](Get-JsonValue -Obj $rec -Name 'checked_at') }
}

# RV 1-1 [3] - the virtual disposition table. Returns $true when the caller must repack; every row
# whose disposition is "stop" terminates inside this function, so the caller never has to decide.
#
#   manifest VIRTUAL + gate PASS   -> reuse            (the repacker is not run at all)
#   manifest VIRTUAL + gate FAIL   -> stop             (a verification failure is not overwritten)
#   manifest BIN                   -> stop             (mode mismatch, both directions)
#   manifest unreadable/unknown    -> stop             (fail-close, its own reason)
#   bin residue, no virtual manifest -> stop           (no virtual output on top of bin leftovers)
#   plan_report or manifest.json.partial alone -> cleanup, then repack
#   nothing                        -> repack
#
# The bin .partial confirm/cleanup flow (Invoke-PartialCleanup) is NOT run here: it deletes
# manifest.json among four bin artifacts, which is the opposite disposition to v1's "a bin artifact
# in this directory is a hard stop". One file, one verdict.
function Resolve-VirtualArtifactState {
    param([string] $OutputDir, [string] $ProfileId, [string] $ExpectSha)
    $manifestPath = Join-Path $OutputDir 'manifest.json'

    # Existence is decided by the FileNotFound-only rule for every file below, manifest and report
    # included: an ACL or IO error must not be read as "absent" and silently trigger a fresh repack.
    $mSt = Get-FileAbsenceState -Path $manifestPath
    if ($mSt.state -eq 'unknown') { Stop-Launcher 'fail_gate_verify' ('manifest.json absence not provable - ' + $mSt.reason) }

    if ($mSt.state -eq 'present') {
        $mode = Get-ManifestMode -ManifestPath $manifestPath
        if ([string]$mode.mode -ceq $script:MANIFEST_MODE_UNRECOGNIZED) { Stop-UnrecognizedManifest -ModeResult $mode }
        if ([string]$mode.mode -cne $script:MANIFEST_MODE_VIRTUAL) {
            Stop-ModeMismatch -Existing ([string]$mode.mode) -Requested $script:REPACK_MODE_VIRTUAL
        }
        # A virtual manifest is here. The gate decides whether it may be served; it never decides
        # whether to rebuild - a FAIL stops, because regenerating over a failed verification would
        # destroy the evidence of what failed.
        [void](Assert-VirtualPlanGate -OutputDir $OutputDir -ProfileId $ProfileId -ExpectSha $ExpectSha)
        Write-Line '[repack] reusing the existing virtual plan in this output directory.'
        Write-Diag -Kind 'VIRTUAL_ARTIFACTS_REUSED' -Data @{ out = $OutputDir }
        return $false
    }

    # No manifest. Bin leftovers are checked BEFORE the incomplete-production rows: a directory that
    # still holds experts.bin is a bin directory, and this run may neither serve nor tidy it.
    foreach ($n in @('experts.bin', 'experts.bin.partial')) {
        $b = Get-FileAbsenceState -Path (Join-Path $OutputDir $n)
        if ($b.state -eq 'present') {
            Stop-Launcher 'fail_gate_verify' ('vgate 1: ' + $n + ' present in a virtual output directory')
        }
        if ($b.state -ne 'absent') {
            Stop-Launcher 'fail_gate_verify' ('vgate 1: ' + $n + ' absence not provable - ' + $b.reason)
        }
    }

    $incomplete = @()
    foreach ($n in $script:VIRTUAL_PARTIAL_DELETE_SET) {
        $st = Get-FileAbsenceState -Path (Join-Path $OutputDir $n)
        if ($st.state -eq 'unknown') { Stop-Launcher 'fail_gate_verify' ($n + ' absence not provable - ' + $st.reason) }
        if ($st.state -eq 'present') { $incomplete += $n }
    }
    if ($incomplete.Count -gt 0) {
        Write-Diag -Kind 'VIRTUAL_INCOMPLETE_DETECTED' -Data @{ out = $OutputDir; present = $incomplete }
        Invoke-VirtualPartialCleanup -OutputDir $OutputDir
    }
    return $true
}

# endregion

# ============================================================================
# region 13. USER PRESET (LS 1-7) - atomic round trip, zero partial application
# ============================================================================

function Get-PresetPath {
    return (Join-Path (Get-LauncherStateDir) 'presets.user.json')
}

# Load order is fixed by LS 1-7: (1) atomic read (2) strict schema + exact schema_version
# (3) source_tag/profile_id/expect_digest binding (4) allowlist projection (5) type/bounds
# (6) regeneration from the current catalog (7) re-sizing (8) status. Any failure at any step
# discards the whole preset - never a partial application.
function Read-UserPreset {
    param([string] $SourceTag, [string] $ProfileId, [string] $ExpectDigest, $Bounds)
    $path = Get-PresetPath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return @{ applied = $false; reason = 'no stored preset'; overrides = @{} }
    }
    # (1) atomic read
    $b = Read-FileBytesStrict -Path $path
    if (-not $b.ok) { return (Discard-Preset -Reason ('unreadable - ' + $b.reason)) }
    $t = ConvertFrom-Utf8Strict -Bytes $b.bytes
    if (-not $t.ok) { return (Discard-Preset -Reason ('not valid utf-8 - ' + $t.reason)) }
    $pr = ConvertFrom-JsonStrict -Text $t.text
    if (-not $pr.ok) { return (Discard-Preset -Reason ('corrupt or truncated json - ' + $pr.reason)) }
    $obj = $pr.value

    # (2) strict schema + exact schema_version
    foreach ($f in $script:PRESET_REQUIRED_FIELDS) {
        if (-not (Test-JsonHas -Obj $obj -Name $f)) { return (Discard-Preset -Reason ('required field missing: ' + $f)) }
    }
    foreach ($k in (Get-JsonKeys -Obj $obj)) {
        if ($script:PRESET_REQUIRED_FIELDS -notcontains $k) { return (Discard-Preset -Reason ('unknown top-level field: ' + $k)) }
    }
    $sv = Get-JsonValue -Obj $obj -Name 'schema_version'
    if (-not (Test-JsonNonNegativeInteger $sv)) { return (Discard-Preset -Reason 'schema_version is not an integer') }
    if ([long]$sv -ne [long]$script:PRESET_SCHEMA_VERSION) { return (Discard-Preset -Reason ('stale schema_version ' + $sv)) }

    # (3) binding
    if ([string](Get-JsonValue -Obj $obj -Name 'source_tag')    -cne $SourceTag)    { return (Discard-Preset -Reason 'source_tag binding mismatch') }
    if ([string](Get-JsonValue -Obj $obj -Name 'profile_id')    -cne $ProfileId)    { return (Discard-Preset -Reason 'profile_id binding mismatch') }
    if (([string](Get-JsonValue -Obj $obj -Name 'expect_digest')).ToLowerInvariant() -ne $ExpectDigest.ToLowerInvariant()) {
        return (Discard-Preset -Reason 'expect_digest binding mismatch')
    }

    # (4) allowlist projection - keys outside the allowlist are dropped, never applied
    $ovIn = Get-JsonValue -Obj $obj -Name 'overrides'
    if ($null -eq $ovIn) { return (Discard-Preset -Reason 'overrides missing') }
    $dropped = @()
    $proj = @{}
    foreach ($k in (Get-JsonKeys -Obj $ovIn)) {
        if ($script:PRESET_ALLOWLIST_KEYS -contains $k) { $proj[$k] = (Get-JsonValue -Obj $ovIn -Name $k) }
        else { $dropped += $k }
    }

    # (5) type / bounds
    $checked = @{}
    foreach ($k in $proj.Keys) {
        $v = Test-OverrideValue -Key $k -Value ([string]$proj[$k]) -Bounds $Bounds
        if (-not $v.ok) { return (Discard-Preset -Reason ('override ' + $k + ' rejected: ' + $v.reason)) }
        $checked[$k] = $v.value
    }

    Write-Diag -Kind 'PRESET_LOADED' -Data @{ path = $path; overrides = $checked; dropped_keys = $dropped }
    return @{ applied = $true; overrides = $checked; dropped = $dropped; path = $path }
}

function Discard-Preset {
    param([string] $Reason)
    # LS 1-7 / LS 5: preset discard is a degraded, non-terminal path.
    Write-Line ('[preset] stored preset discarded (' + $Reason + '); falling back to catalog defaults.')
    Write-Diag -Kind 'PRESET_DISCARDED' -Data @{ reason = $Reason }
    return @{ applied = $false; reason = $Reason; overrides = @{} }
}

function Save-UserPreset {
    param([string] $SourceTag, [string] $ProfileId, [string] $ExpectDigest, $Overrides)
    $path = Get-PresetPath
    $dir = Split-Path -Parent $path
    try {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $obj = [ordered]@{
            schema_version = [int]$script:PRESET_SCHEMA_VERSION
            source_tag     = $SourceTag
            profile_id     = $ProfileId
            expect_digest  = $ExpectDigest
            overrides      = [ordered]@{}
        }
        foreach ($k in $script:PRESET_ALLOWLIST_KEYS) {
            if ($Overrides.ContainsKey($k)) { $obj['overrides'][$k] = [string]$Overrides[$k] }
        }
        $json = ($obj | ConvertTo-Json -Depth 6)
        $tmp = $path + '.tmp'
        [System.IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding($false)))
        # read-back re-parse before the atomic replace
        $rb = Read-JsonFileStrict -Path $tmp
        if (-not $rb.ok) { throw ('read-back parse failed: ' + $rb.reason) }
        if ([long](Get-JsonValue -Obj $rb.value -Name 'schema_version') -ne [long]$script:PRESET_SCHEMA_VERSION) {
            throw 'read-back schema_version mismatch'
        }
        Move-FileAtomic -TempPath $tmp -FinalPath $path
        Write-Diag -Kind 'PRESET_SAVED' -Data @{ path = $path; overrides = $Overrides }
        Write-Line ('[preset] saved: ' + $path)
        return $true
    } catch {
        # LS 1-7 save failure is a non-terminal warning; the session keeps the effective values.
        Write-Line ('[preset] WARNING: could not save preset (' + $_.Exception.Message + '). This session continues with the effective values; they will not persist.')
        Write-Diag -Kind 'preset_save_failed' -Data @{ path = $path; reason = $_.Exception.Message }
        try { if (Test-Path -LiteralPath ($path + '.tmp') -PathType Leaf) { Remove-Item -LiteralPath ($path + '.tmp') -Force -ErrorAction SilentlyContinue } } catch { }
        return $false
    }
}

function Remove-UserPreset {
    $path = Get-PresetPath
    try {
        if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force -ErrorAction Stop }
        Write-Line '[preset] stored preset reset.'
        Write-Diag -Kind 'PRESET_RESET' -Data @{ path = $path }
    } catch {
        Write-Line ('[preset] WARNING: reset failed (' + $_.Exception.Message + ')')
        Write-Diag -Kind 'preset_save_failed' -Data @{ path = $path; reason = ('reset: ' + $_.Exception.Message) }
    }
}

# endregion

# ============================================================================
# region 13b. ARCH-TEMPLATE PREFERENCE (UX 1-1) - global, resolved before identification
#
# Everything in this region runs BEFORE Resolve-ProfileSelection, so nothing here may depend on a
# profile, a catalog entry or an expect digest - that is exactly why the value cannot live in the
# preset (which is bound to all three) and needs a file of its own. See the constants block in
# region 1 for the full reasoning and for the discard policy.
# ============================================================================

function Get-ArchTemplatePrefPath {
    # Same state directory as the user preset (LS 1-7): one directory, two independent files. The
    # preset's 8-step load contract is untouched by this - a separate file cannot partially apply.
    return (Join-Path (Get-LauncherStateDir) $script:ARCH_TEMPLATE_PREF_FILE_NAME)
}

# Strict load with exactly three outcomes and no fourth:
#   absent  - no file at all                    -> the caller applies the product default
#   valid   - every check passed                -> the stored value
#   discard - the file EXISTS and failed a check -> the caller fails CLOSED (UX 1-1-1)
# "Exists but unreadable" is deliberately a discard rather than an absence: an I/O error on a file
# the user did create is not the same statement as never having chosen.
function Read-ArchTemplatePref {
    $path = Get-ArchTemplatePrefPath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return @{ state = 'absent'; path = $path }
    }
    $b = Read-FileBytesStrict -Path $path
    if (-not $b.ok) { return @{ state = 'discard'; path = $path; reason = ('unreadable - ' + $b.reason) } }
    $t = ConvertFrom-Utf8Strict -Bytes $b.bytes
    if (-not $t.ok) { return @{ state = 'discard'; path = $path; reason = ('not valid utf-8 - ' + $t.reason) } }
    $pr = ConvertFrom-JsonStrict -Text $t.text
    if (-not $pr.ok) { return @{ state = 'discard'; path = $path; reason = ('corrupt or truncated json - ' + $pr.reason) } }
    $obj = $pr.value
    foreach ($f in $script:ARCH_TEMPLATE_PREF_REQUIRED_FIELDS) {
        if (-not (Test-JsonHas -Obj $obj -Name $f)) {
            return @{ state = 'discard'; path = $path; reason = ('required field missing: ' + $f) }
        }
    }
    # Unknown top-level key discards the WHOLE file - the preset's rule (LS 1-7 step 2) for the same
    # reason: a file written by a newer launcher carries a meaning this one cannot claim to know.
    foreach ($k in (Get-JsonKeys -Obj $obj)) {
        if ($script:ARCH_TEMPLATE_PREF_REQUIRED_FIELDS -notcontains $k) {
            return @{ state = 'discard'; path = $path; reason = ('unknown top-level field: ' + $k) }
        }
    }
    $sv = Get-JsonValue -Obj $obj -Name 'schema_version'
    if (-not (Test-JsonNonNegativeInteger $sv)) {
        return @{ state = 'discard'; path = $path; reason = 'schema_version is not an integer' }
    }
    if ([long]$sv -ne [long]$script:ARCH_TEMPLATE_PREF_SCHEMA_VERSION) {
        return @{ state = 'discard'; path = $path; reason = ('stale schema_version ' + $sv) }
    }
    $v = Get-JsonValue -Obj $obj -Name 'arch_template'
    if (-not (Test-JsonNonEmptyString $v)) {
        return @{ state = 'discard'; path = $path; reason = 'arch_template is not a string' }
    }
    # Exact and lower-case. The 'true'/'1' spellings the CLI accepts are a typing convenience on the
    # command line; this file is written by the launcher itself and is held to the canonical form.
    if ([string]$v -cne 'on' -and [string]$v -cne 'off') {
        return @{ state = 'discard'; path = $path; reason = "arch_template must be 'on' or 'off'" }
    }
    return @{ state = 'valid'; path = $path; value = [string]$v }
}

# tmp + read-back + atomic replace - the preset saver's pattern (LS 1-7), adopted for the same
# reason: a half-written preference is precisely the corrupt file the discard rule then has to fail
# closed on. A write failure is non-terminal, because this run already holds its resolved value.
function Save-ArchTemplatePref {
    param([string] $Value)
    $path = Get-ArchTemplatePrefPath
    $dir = Split-Path -Parent $path
    try {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $obj = [ordered]@{
            schema_version = [int]$script:ARCH_TEMPLATE_PREF_SCHEMA_VERSION
            arch_template  = [string]$Value
        }
        $tmp = $path + '.tmp'
        [System.IO.File]::WriteAllText($tmp, ($obj | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))
        $rb = Read-JsonFileStrict -Path $tmp
        if (-not $rb.ok) { throw ('read-back parse failed: ' + $rb.reason) }
        if ([string](Get-JsonValue -Obj $rb.value -Name 'arch_template') -cne [string]$Value) { throw 'read-back value mismatch' }
        Move-FileAtomic -TempPath $tmp -FinalPath $path
        Write-Diag -Kind 'ARCH_TEMPLATE_PREF_SAVED' -Data @{ path = $path; value = [string]$Value }
        return $true
    } catch {
        Write-Line ('[arch template] WARNING: could not save the preference (' + $_.Exception.Message +
                    '). This run uses ' + $Value + '; the choice will not persist.')
        Write-Diag -Kind 'arch_template_pref_save_failed' -Data @{ path = $path; reason = $_.Exception.Message }
        try { if (Test-Path -LiteralPath ($path + '.tmp') -PathType Leaf) { Remove-Item -LiteralPath ($path + '.tmp') -Force -ErrorAction SilentlyContinue } } catch { }
        return $false
    }
}

# Raw-string validation for the canonical CLI value, the same discipline as the six allowlist keys.
# An unusable value terminates as fail_custom_args and is never silently promoted to the default -
# that would be a fail-OPEN on the one control the whole admissibility gate hangs on.
function Test-ArchTemplateValue {
    param([string] $Value)
    $v = ([string]$Value).Trim().ToLowerInvariant()
    if ($v -eq 'on'  -or $v -eq 'true'  -or $v -eq '1') { return @{ ok = $true; value = 'on' } }
    if ($v -eq 'off' -or $v -eq 'false' -or $v -eq '0') { return @{ ok = $true; value = 'off' } }
    return @{ ok = $false; reason = "arch template must be 'on' or 'off'" }
}

$script:ArchTemplateResolved     = $null
$script:ArchTemplateSource       = $null
$script:ArchTemplateCanonicalCli = $false
# UX 1-1-5: set once the model-selection menu has actually rendered its toggle item. It is what
# tells the -Model fallback question that the user has already been offered the same control, so
# the two entry points can never both fire in one run.
$script:ArchTemplateToggleOffered = $false

function Set-ArchTemplateResolved {
    param([string] $Value, [string] $Source)
    $script:ArchTemplateResolved = [string]$Value
    $script:ArchTemplateSource   = [string]$Source
    Write-Diag -Kind 'ARCH_TEMPLATE_RESOLVED' -Data @{ value = [string]$Value; source = [string]$Source }
}

# UX 1-1-2 - the whole resolution order in one place, run ONCE and before identification:
#   1 canonical -ArchTemplate   given = final, and the only recovery from a discarded preference
#   2 a preference file that FAILED the strict load -> 'off' (fail-close). It sits above the
#     deprecated switch on purpose: a damaged preference must not be revivable by the old switch,
#     because needRepack=false would then serve an existing template artifact past a skipped gate.
#   3 -ExperimentalArchTemplate  deprecated, maps to 'on' only when 1 is absent
#   4 a valid preference file
#   5 the product default
# The answer is latched into $script:ArchTemplateResolved and every later consumer - the selection
# gate, the status line, the interactive toggle - reads that one variable and nothing else.
function Resolve-ArchTemplate {
    if ($null -ne $ArchTemplate -and ([string]$ArchTemplate).Trim().Length -gt 0) {
        $r = Test-ArchTemplateValue -Value ([string]$ArchTemplate)
        if (-not $r.ok) { Stop-Launcher 'fail_custom_args' ("invalid -ArchTemplate '" + $ArchTemplate + "': " + $r.reason) }
        $script:ArchTemplateCanonicalCli = $true
        Set-ArchTemplateResolved -Value $r.value -Source 'cli'
        return
    }
    $pref = Read-ArchTemplatePref
    if ($pref.state -eq 'discard') {
        Write-Line ('[arch template] stored preference discarded (' + $pref.reason +
                    '); this run continues with arch template OFF. Recover with -ArchTemplate on.')
        Write-Diag -Kind 'ARCH_TEMPLATE_PREF_DISCARDED' -Data @{ path = $pref.path; reason = $pref.reason }
        Set-ArchTemplateResolved -Value $script:ARCH_TEMPLATE_PREF_DISCARD_VALUE -Source 'pref_discarded'
        return
    }
    if ($ExperimentalArchTemplate) {
        Set-ArchTemplateResolved -Value 'on' -Source 'deprecated_switch'
        return
    }
    if ($pref.state -eq 'valid') {
        Set-ArchTemplateResolved -Value ([string]$pref.value) -Source 'preference'
        return
    }
    Set-ArchTemplateResolved -Value $script:ARCH_TEMPLATE_DEFAULT -Source 'default'
}

# UX 1-1-3: template admissibility, decided at the selection call and nowhere else. Closing it
# closes the template FALLBACK only - a catalog candidate, pinned or unpinned, is returned before
# this can matter, so an out-of-family arch the catalog DOES describe keeps its catalog pin verdict.
function Test-TemplateAdmissible {
    param([string] $Arch)
    if ($script:ArchTemplateResolved -cne 'on') { return $false }
    return ($script:ARCH_TEMPLATE_FAMILIES -ccontains [string]$Arch)
}

# UX 1-1-1 fail-close, interactive half (Codex build r1 M1). The frozen recovery rule is "explicit
# -ArchTemplate on ONLY". Resolve-ArchTemplate honours it, but an interactive control that WRITES a
# fresh preference would launder the discard away inside the same run: the damaged file is replaced
# by a clean one and the value flips to on with no CLI anywhere in the story. The reachable path is
# an ordinary one - a partially written file or a version skew from a newer launcher, then the model
# menu. So every interactive writer asks here first, and a locked run says so rather than going
# quiet. Reading is unaffected; only writing and re-latching are closed.
function Test-ArchTemplateInteractiveAllowed {
    return ($script:ArchTemplateSource -cne 'pref_discarded')
}

# UX 1-1-5: an interactive choice is a decision taken LATER than any command line, so it both
# persists and re-latches the value for this run. That is the entire point of the toggle - the
# interactive custom editor is only reached after selection, derive-plan and the repack
# confirmation, which is too late to be the first place a user can say "no".
function Set-ArchTemplateInteractive {
    param([string] $Value)
    [void](Save-ArchTemplatePref -Value $Value)
    Set-ArchTemplateResolved -Value ([string]$Value) -Source 'interactive'
}

# UX 1-1-5, second half: -Model skips the selection menu, so the same pre-identification decision
# needs an entry point on that path too. Scope is frozen and narrow - only when this run would
# actually take the template path, only when no explicit choice has ever been stored, and never
# under -NonInteractive or a canonical CLI value. The answer is stored either way, so a later run
# never asks again. Deliberately NOT Confirm-User: that helper's default answer is "no" and obeys
# -AssumeYes/-AssumeNo, which belong to the repack confirmation; here the default answer has to be
# the product default (keep it on) and a bare Enter must mean exactly that.
function Confirm-ArchTemplateBeforeIdentify {
    param($Catalog, $ModelSet, [string] $Root)
    if ($NonInteractive) { return }
    if ($script:ArchTemplateCanonicalCli) { return }
    if ($script:ArchTemplateToggleOffered) { return }
    # Codex build r1 M1: a discarded preference is locked off; an interactive answer here would write
    # a clean file and undo that, so the question is not asked at all (the value is already off, so
    # Test-TemplateAdmissible below would refuse too - this states the reason explicitly).
    if (-not (Test-ArchTemplateInteractiveAllowed)) { return }
    $Arch = [string]$ModelSet.arch
    if (-not (Test-TemplateAdmissible -Arch $Arch)) { return }
    if ((Read-ArchTemplatePref).state -ne 'absent') { return }
    # Codex build r1 M2: family membership alone is NOT "this run will take the template path".
    # Resolve-ProfileSelection returns a catalog candidate - pinned or unpinned - before the template
    # value can matter, so a catalogued gpt-oss/qwen35moe/deepseek2 model would be asked a question
    # that changes nothing about its own run while still rewriting a GLOBAL preference. This is the
    # exact predicate that call uses (Get-StructuralProfileCandidates), not the menu label's weaker
    # 4-field comparison: two quantisations of one model share all four identify fields but differ in
    # shard bytes, so the label-level check would skip the question on a run that really does go
    # template. The candidates are recomputed a few lines later by the selection itself; that costs
    # one expect read for the profiles that pass the header prefilter, and no GGUF is reopened.
    # Assign, then wrap - Get-StructuralProfileCandidates returns its array through the ", @(...)"
    # idiom, so "@(Get-StructuralProfileCandidates ...)" re-wraps it and reports Count 1 even when it
    # is empty (the same trap Get-JsonValue documents). Resolve-ProfileSelection assigns first for
    # exactly this reason, and the question below has to see the same count that call will see.
    $catalogCands = Get-StructuralProfileCandidates -Catalog $Catalog -ModelSet $ModelSet -Root $Root
    if (@($catalogCands).Count -gt 0) { return }
    Write-Line ''
    Write-Line ('[arch template] on - this architecture (' + $Arch + ') has an EXPERIMENTAL arch template, so an')
    Write-Line '                unlisted GGUF can be prepared without a catalog entry. No published'
    Write-Line '                measurement covers such a model. Nothing is written before the plan'
    Write-Line '                and its confirmation; turning it off reproduces the "unsupported GGUF"'
    Write-Line '                refusal instead. This is asked once and the answer is remembered.'
    $ans = Read-UserLine -Prompt '                Keep arch template enabled? [Y/n] '
    $value = 'on'
    if ($null -ne $ans) {
        $a = ([string]$ans).Trim().ToLowerInvariant()
        if ($a -eq 'n' -or $a -eq 'no') { $value = 'off' }
    }
    Set-ArchTemplateInteractive -Value $value
    Write-Line ('[arch template] ' + $value + ' (stored; use -ArchTemplate to override a single run)')
}

# endregion

# ============================================================================
# region 14. ALLOWLIST OVERRIDES / EFFECTIVE CONFIG (LS 1-2, LS 1-3)
# ============================================================================

function Get-BoundPair {
    param($Bounds, [string] $Key)
    $b = Get-JsonValue -Obj $Bounds -Name $Key
    return @{ min = [long](Get-JsonValue -Obj $b -Name 'min'); max = [long](Get-JsonValue -Obj $b -Name 'max') }
}

# WARMFILE_DESIGN v0.2 section 1. Returns the path when a warmup value selects the warmfile mode,
# $null otherwise. ONE rule serves both users of it - the allowlist validator below and the
# post-ready dispatcher - so a value that validated as a warmfile can never dispatch as a generic
# warmup. The prefix is matched case-insensitively; nothing else about the value is touched, because
# the remainder is a path: lower-casing it would corrupt a case-sensitive share path and would stop
# the stored preset from round-tripping what the user typed.
function Get-WarmupFilePath {
    param([string] $Value)
    $v = [string]$Value
    $p = $script:WARMUP_FILE_PREFIX
    if ($v.Length -le $p.Length) { return $null }
    if ($v.Substring(0, $p.Length).ToLowerInvariant() -cne $p) { return $null }
    return $v.Substring($p.Length)
}

function Test-OverrideValue {
    param([string] $Key, [string] $Value, $Bounds)
    # WARMFILE_DESIGN v0.2 section 1: the warmfile mode is decided BEFORE the on/off branch, because
    # that branch lower-cases and trims its input, and neither may be done to a path. A bare 'file:'
    # with nothing after it is not a mode - it falls through and is rejected as a bad value exactly
    # like 'maybe' (a value-shape violation on the settings surface, not a runtime warmup failure).
    # r1 F1: leading whitespace is skipped ONLY to find the prefix. Everything after 'file:' is taken
    # byte for byte - a full Trim() would also eat TRAILING whitespace, and whitespace (including
    # U+00A0, which Char.IsWhiteSpace accepts) is legal in a Windows file name and is in neither
    # GetInvalidFileNameChars() nor GetInvalidPathChars(). Trimming it would silently point the run
    # at a different file, or degrade it to warmup_failed, on all three layers at once.
    if ($Key -eq 'warmup') {
        $raw = [string]$Value
        $lead = 0
        while ($lead -lt $raw.Length -and [char]::IsWhiteSpace($raw[$lead])) { $lead = $lead + 1 }
        $wf = Get-WarmupFilePath -Value $raw.Substring($lead)
        if ($null -ne $wf) { return @{ ok = $true; value = ($script:WARMUP_FILE_PREFIX + $wf) } }
    }
    # LS 13-8: 'autosave' is on | off | <minutes>, so it deliberately does NOT reuse the warmup
    # on/true/1 spellings: '1' has to mean one minute, not "on". Everything else about the discipline
    # is identical (raw string in, normalised value out, rejection = fail_custom_args or an
    # interactive re-loop). The minute bounds are structural, like PORT_MIN/PORT_MAX: the catalog
    # carries no allowlist_bounds entry for this key and none is invented from it.
    if ($Key -eq 'autosave') {
        $v = ([string]$Value).Trim().ToLowerInvariant()
        if ($v -eq 'on')  { return @{ ok = $true; value = 'on' } }
        if ($v -eq 'off') { return @{ ok = $true; value = 'off' } }
        $m = [long]0
        if (-not [long]::TryParse($v, [ref]$m)) {
            return @{ ok = $false; reason = "autosave must be 'on', 'off' or a whole number of minutes" }
        }
        if ($m -lt $script:KV_AUTOSAVE_MIN_MINUTES -or $m -gt $script:KV_AUTOSAVE_MAX_MINUTES) {
            return @{ ok = $false; reason = ('autosave minutes out of bounds [' + $script:KV_AUTOSAVE_MIN_MINUTES +
                                             '..' + $script:KV_AUTOSAVE_MAX_MINUTES + ']') }
        }
        return @{ ok = $true; value = [string]$m }
    }
    # P4 4: 'prefetch' is a closed three-value enum, so it takes neither the on/off spellings nor the
    # numeric branch. Same discipline as every other key: raw string in, normalised value out, a
    # rejection is fail_custom_args when non-interactive and a re-loop when interactive.
    if ($Key -eq 'prefetch') {
        $v = ([string]$Value).Trim().ToLowerInvariant()
        if ($script:PREFETCH_REQUEST_VALUES -ccontains $v) { return @{ ok = $true; value = $v } }
        return @{ ok = $false; reason = ("prefetch must be one of " + ($script:PREFETCH_REQUEST_VALUES -join '|')) }
    }
    # LS 13-2: 'warmstart' reuses the existing on/off branch byte for byte - same accepted spellings,
    # same rejection, and therefore the same fail_custom_args / interactive re-loop behaviour.
    if ($Key -eq 'warmup' -or $Key -eq 'warmstart') {
        $v = ([string]$Value).Trim().ToLowerInvariant()
        if ($v -eq 'on' -or $v -eq 'true' -or $v -eq '1')  { return @{ ok = $true; value = 'on' } }
        if ($v -eq 'off' -or $v -eq 'false' -or $v -eq '0') { return @{ ok = $true; value = 'off' } }
        if ($Key -eq 'warmup') { return @{ ok = $false; reason = "warmup must be 'on', 'off' or 'file:<path>'" } }
        return @{ ok = $false; reason = ($Key + " must be 'on' or 'off'") }
    }
    $n = [long]0
    if (-not [long]::TryParse(([string]$Value).Trim(), [ref]$n)) {
        return @{ ok = $false; reason = ($Key + ' must be an integer') }
    }
    $bp = Get-BoundPair -Bounds $Bounds -Key $Key
    $lo = $bp.min; $hi = $bp.max
    if ($Key -eq 'port') {
        if ($lo -lt $script:PORT_MIN) { $lo = $script:PORT_MIN }
        if ($hi -gt $script:PORT_MAX) { $hi = $script:PORT_MAX }
    }
    if ($Key -eq 'qd') {
        if ($lo -lt $script:ENGINE_QD_MIN) { $lo = $script:ENGINE_QD_MIN }
        if ($hi -gt $script:ENGINE_QD_MAX) { $hi = $script:ENGINE_QD_MAX }
    }
    if ($n -lt $lo -or $n -gt $hi) {
        return @{ ok = $false; reason = ($Key + ' out of bounds [' + $lo + '..' + $hi + ']') }
    }
    return @{ ok = $true; value = [string]$n }
}

function Get-ArgvValue {
    param([string[]] $Argv, [string] $Flag)
    for ($i = 0; $i -lt $Argv.Count - 1; $i++) {
        if ($Argv[$i] -ceq $Flag) { return $Argv[$i + 1] }
    }
    return $null
}

function Set-ArgvValue {
    param([string[]] $Argv, [string] $Flag, [string] $Value)
    $out = @()
    $replaced = $false
    for ($i = 0; $i -lt $Argv.Count; $i++) {
        if ($Argv[$i] -ceq $Flag -and $i -lt $Argv.Count - 1) {
            $out += $Flag; $out += $Value; $i++; $replaced = $true
        } else { $out += $Argv[$i] }
    }
    if (-not $replaced) { $out += $Flag; $out += $Value }
    return , $out
}

function Remove-ArgvFlag {
    param([string[]] $Argv, [string] $Flag)
    $out = @()
    foreach ($a in $Argv) { if ($a -cne $Flag) { $out += $a } }
    return , $out
}

# Removes a flag together with its value, at every occurrence (arity 1). Used so a catalog-supplied
# --slot-save-path can never survive into the effective argv: this launcher owns that argument.
function Remove-ArgvPair {
    param([string[]] $Argv, [string] $Flag)
    $out = @()
    $i = 0
    while ($i -lt $Argv.Count) {
        if ($Argv[$i] -ceq $Flag) { $i = $i + 2; continue }
        $out += [string]$Argv[$i]
        $i = $i + 1
    }
    return , $out
}

# =============================================================================================
# P4 1 / 2 - the three-axis prefetch state machine.
#
# Axis ownership (PI 3): capability is a SEAL-TIME runtime result and is never stored or decided
# here; evidence and activation are catalog-stored; the opt-in arm is a runtime input. This
# launcher therefore decides a CANDIDATE and nothing more - the final activation authority is the
# engine seal, exactly as it was in v0.4.
# =============================================================================================

# The semantic reader. It never terminates: a row it cannot make sense of comes back marked
# semantic_invalid, and the resolver turns that row's prefetch off with a reason (P4 1-b).
function Get-PrefetchCatalogAxes {
    param($Profile)
    $out = @{ evidence = ''; activation = ''; hold = $false; k = $null; n = $null
              semantic_invalid = $false; detail = $null }
    $ev  = Get-JsonValue -Obj $Profile -Name 'prefetch_evidence'
    $act = Get-JsonValue -Obj $Profile -Name 'prefetch_activation'
    $out.evidence   = [string]$ev
    $out.activation = [string]$act
    if (Test-JsonHas -Obj $Profile -Name 'prefetch_promotion_hold') {
        $out.hold = [bool](Test-JsonBooleanTrue (Get-JsonValue -Obj $Profile -Name 'prefetch_promotion_hold'))
    }
    $pf = Get-JsonValue -Obj $Profile -Name 'prefetch'
    $tupleOk = $false
    if ($null -ne $pf) {
        $kv = Get-JsonValue -Obj $pf -Name 'k'
        $nv = Get-JsonValue -Obj $pf -Name 'n'
        if ((Test-JsonNonNegativeInteger $kv) -and (Test-JsonNonNegativeInteger $nv)) {
            $out.k = [long]$kv; $out.n = [long]$nv; $tupleOk = $true
        }
    }

    function Set-Invalid { param($Bag, [string] $Why) ; $Bag.semantic_invalid = $true ; $Bag.detail = $Why }

    if ($script:PREFETCH_EVIDENCE_VALUES -cnotcontains $out.evidence) {
        Set-Invalid -Bag $out -Why ('unknown prefetch_evidence: ' + $out.evidence) ; return $out
    }
    if ($script:PREFETCH_ACTIVATION_RUNTIME -ccontains $out.activation) {
        # PI 3 invariant 3: an opt-in activation is produced by an opt-in, never read out of a file.
        Set-Invalid -Bag $out -Why ('prefetch_activation ' + $out.activation + ' is a runtime value and may not be stored') ; return $out
    }
    if ($script:PREFETCH_ACTIVATION_STORED -cnotcontains $out.activation) {
        Set-Invalid -Bag $out -Why ('unknown prefetch_activation: ' + $out.activation) ; return $out
    }
    # P4 1 closure rule, both directions of "catalog-fixed <=> paired-live AND a valid tuple".
    if ($out.activation -ceq 'catalog-fixed') {
        if ($out.evidence -cne 'paired-live') {
            # Evidence regression: the row claims the validated activation on weaker evidence.
            Set-Invalid -Bag $out -Why ('catalog-fixed requires prefetch_evidence=paired-live, not ' + $out.evidence) ; return $out
        }
        if (-not $tupleOk) {
            Set-Invalid -Bag $out -Why 'catalog-fixed requires a valid prefetch {k,n} tuple' ; return $out
        }
    } else {
        if ($null -ne $pf) {
            Set-Invalid -Bag $out -Why 'prefetch_activation=off requires prefetch=null (stray tuple)' ; return $out
        }
        if ($out.evidence -ceq 'paired-live') {
            # The other direction of the same closure rule. paired-live is the evidence that
            # catalog-fixed is made of, so a paired-live row parked on 'off' is an authoring state
            # the total function of P4 1 does not contain - and it is the one evidence value the
            # not_opted_in_* token set deliberately has no member for.
            Set-Invalid -Bag $out -Why 'prefetch_evidence=paired-live with prefetch_activation=off is not a closed state' ; return $out
        }
    }
    return $out
}

# P4 4: the public enum -> internal arm mapping, written exactly once. 'catalog' and absence are
# the SAME request (the v0.4 behaviour), and both mean "no opt-in".
function ConvertTo-PrefetchOptIn {
    param([string] $Request)
    $r = [string]$Request
    if ($r.Length -eq 0) { $r = $script:PREFETCH_REQUEST_DEFAULT }
    if ($r -ceq 'catalog') { return $script:PREFETCH_ARM_NONE }
    if ($r -ceq 'init')    { return $script:PREFETCH_ARM_INIT }
    if ($r -ceq 'adapt')   { return 'adapt' }
    # Unreachable: Test-OverrideValue is the only producer and it rejects everything else.
    return $script:PREFETCH_ARM_NONE
}

# P4 5: adapt is refused on every path in Phase 4, and the two refusals say different true things.
function Get-PrefetchAdaptRefusal {
    param([bool] $ReproOrBench)
    if ($ReproOrBench) { return 'adapt_forbidden_in_repro_bench' }
    return 'adapt_controller_not_shipped_phase5'
}

# PI 2: N0 = min(4, QD-1, max(1, floor(QD/2))). QD1 has no legal N (N < QD is an admission
# invariant), so QD1 is OFF rather than N0. The cap of 4 is the outer edge of the evidence, not a
# claim that 4 is optimal.
function Get-PrefetchInitN {
    param([int] $EffectiveQd)
    if ([int]$EffectiveQd -lt 2) { return $null }
    $half = [int][math]::Floor([double]$EffectiveQd / 2.0)
    if ($half -lt 1) { $half = 1 }
    $n = $script:PREFETCH_INIT_N_CAP
    if (($EffectiveQd - 1) -lt $n) { $n = $EffectiveQd - 1 }
    if ($half -lt $n) { $n = $half }
    return [long]$n
}

# ---------------------------------------------------------------------------------------------
# P4 2 - the resolution order, as a total function over (row axes x opt-in x identity x probe x QD).
# Every branch is fail-close and every OFF carries exactly one reason from the closed enum, the
# first one that matched.
#
#   0. the row cannot be read                -> catalog_semantic_invalid
#   0b. a derived row whose plan text and GGUF header disagree about t -> derived_t_mismatch
#   1. activation=catalog-fixed              -> adapt refused / identity gate / probe / ON
#   2. activation=off, no opt-in             -> not_opted_in_evidence_<evidence>
#   3. activation=off, init opt-in           -> hold / t range / engine floor / probe / init_v1
#   4. adapt opt-in                          -> refused (P4 5)
#
# -EffectiveQd is only read by the init arm (the catalog arm's K/N come from the tuple and the
# QD relation is re-checked downstream by Resolve-PrefetchForQd, on every config rebuild).
# -DerivedHeaderExpertUsed is supplied only on the derived path, where the profile's t comes from
# the repacker's plan text instead of the catalog. On the catalog path the structural prefilter has
# already compared identify.n_expert_used with the GGUF header, so there is nothing left to check
# and no second GGUF parser is introduced here.
# ---------------------------------------------------------------------------------------------
function Resolve-EffectivePrefetch {
    param($Profile, [bool] $ProbeOk, [string] $OptIn = 'none', [int] $EffectiveQd = 0,
          [string] $IdentityVerdict = 'unpinned', $DerivedHeaderExpertUsed = $null,
          [bool] $ReproOrBench = $false, [string] $Request = '')

    $axes = Get-PrefetchCatalogAxes -Profile $Profile
    $req = [string]$Request
    if ($req.Length -eq 0) { $req = $script:PREFETCH_REQUEST_DEFAULT }

    # The echo fields every branch carries, so a consumer never has to ask a second function what
    # the row said (P4 3 mandatory echo).
    function New-PfBase {
        param($Axes, [string] $Req, [string] $Identity)
        return @{ request = $Req; evidence = $Axes.evidence; activation = $Axes.activation
                  promotion_hold = [bool]$Axes.hold; identity = $Identity
                  init_version = $script:PREFETCH_INIT_VERSION
                  provenance = $null; warning = $null; off_reason = $null
                  candidate_activation = 'off' }
    }
    function New-PfOff {
        param($Base, [string] $Reason)
        $o = @{}
        foreach ($k in $Base.Keys) { $o[$k] = $Base[$k] }
        $o['on'] = $false
        $o['off_reason'] = $Reason
        $o['echo'] = (Get-PrefetchOffEcho -Reason $Reason)
        $o['candidate_activation'] = 'off'
        return $o
    }

    $base = New-PfBase -Axes $axes -Req $req -Identity $IdentityVerdict

    # 0. semantic failure - before every arm, so a broken row can never reach an opt-in path.
    if ($axes.semantic_invalid) {
        Write-Diag -Kind 'PREFETCH_CATALOG_SEMANTIC_INVALID' -Data @{
            profile = [string](Get-JsonValue -Obj $Profile -Name 'profile_id'); detail = $axes.detail }
        return (New-PfOff -Base $base -Reason 'catalog_semantic_invalid')
    }

    # 0b. derived t cross-check (P4 3). The plan summary and the header are two independent reads
    # of the same fact; when they disagree there is no t this launcher is entitled to use.
    if ($null -ne $DerivedHeaderExpertUsed) {
        $tProfile = Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'identify') -Name 'n_expert_used'
        if ((-not (Test-JsonNonNegativeInteger $tProfile)) -or
            ([long]$tProfile -ne [long]$DerivedHeaderExpertUsed)) {
            return (New-PfOff -Base $base -Reason 'derived_t_mismatch')
        }
    }

    # 1. the catalog-fixed row.
    if ($axes.activation -ceq 'catalog-fixed') {
        # ORDER IS THE CONTRACT, not a tie-break. P4 2 step 2 reads "identity 3-gate -> on pass,
        # the v0.4 validated path; init ignored, adapt refused", and P4 2's closing rule fixes the
        # observable reason priority to that same sequence ("only the first matched reason is
        # echoed"). So the identity gate answers first: a row whose model is not the catalog's
        # model has nothing to grant or refuse an opt-in about. Reachable counter-example that
        # decides it: K2.6 is unpinned today, so `-Prefetch adapt` on it must echo
        # identity_not_exact, not an adapt reason.
        # P4 2.5: exact catalog identity, not a structural fingerprint. Absent pin and empty pin
        # array are the same state (unpinned) and both land here, as does a pin that disagreed.
        if ($IdentityVerdict -cne $script:PREFETCH_IDENTITY_EXACT) {
            return (New-PfOff -Base $base -Reason 'identity_not_exact')
        }
        if ($OptIn -ceq 'adapt') {
            return (New-PfOff -Base $base -Reason (Get-PrefetchAdaptRefusal -ReproOrBench $ReproOrBench))
        }
        if (-not $ProbeOk) {
            $o = New-PfOff -Base $base -Reason 'probe_failed'
            $o['degraded'] = $true
            return $o
        }
        $o = @{}
        foreach ($k in $base.Keys) { $o[$k] = $base[$k] }
        $o['on'] = $true
        $o['echo'] = $script:PREFETCH_ECHO_ON
        $o['k'] = [long]$axes.k
        $o['n'] = [long]$axes.n
        $o['provenance'] = $script:PREFETCH_PROVENANCE_CATALOG
        $o['candidate_activation'] = 'catalog-fixed'
        if ($OptIn -ceq $script:PREFETCH_ARM_INIT) {
            # P4 2 step 2: the request is IGNORED, loudly. The measured pair keeps its provenance -
            # a request that changed nothing may not demote what was validated.
            $o['warning'] = 'prefetch=init ignored: this profile is catalog-fixed (validated K/N); the measured pair is kept'
        }
        return $o
    }

    # 2. activation=off and no opt-in - the ordinary shipped state for five of the six rows.
    if ($OptIn -ceq $script:PREFETCH_ARM_NONE) {
        $token = 'not_opted_in_evidence_' + ($axes.evidence -replace '-', '_')
        return (New-PfOff -Base $base -Reason $token)
    }

    # 3. activation=off with the init opt-in.
    if ($OptIn -ceq $script:PREFETCH_ARM_INIT) {
        if ($axes.hold) {
            # A Phase 4 lock only. It does not answer whether a sealed opt-in may ever be allowed
            # for this row, nor whether the row can be promoted - both are a later round's question.
            return (New-PfOff -Base $base -Reason 'phase4_hold_unresolved')
        }
        $tRaw = Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'identify') -Name 'n_expert_used'
        if (-not (Test-JsonNonNegativeInteger $tRaw)) {
            return (New-PfOff -Base $base -Reason 'init_t_out_of_range')
        }
        $t = [long]$tRaw
        if ($t -lt $script:PREFETCH_INIT_T_MIN -or $t -gt $script:PREFETCH_INIT_T_MAX) {
            return (New-PfOff -Base $base -Reason 'init_t_out_of_range')
        }
        if ($t -lt [long]$script:ENGINE_ENV_K_FLOOR) {
            # Not a capability verdict - a refusal to print a candidate ON that the engine's current
            # env parser would reject outright.
            return (New-PfOff -Base $base -Reason 'engine_env_k_floor_8_pre_p4a')
        }
        # PI 4-2 step 5: a failed probe is the degraded path and an override may not revive it.
        if (-not $ProbeOk) {
            $o = New-PfOff -Base $base -Reason 'probe_failed'
            $o['degraded'] = $true
            return $o
        }
        $n0 = Get-PrefetchInitN -EffectiveQd $EffectiveQd
        if ($null -eq $n0) {
            return (New-PfOff -Base $base -Reason 'qd_below_prefetch_depth')
        }
        $o = @{}
        foreach ($k in $base.Keys) { $o[$k] = $base[$k] }
        $o['on'] = $true
        $o['echo'] = $script:PREFETCH_ECHO_ON
        $o['k'] = [long]$t
        $o['n'] = [long]$n0
        $o['provenance'] = $script:PREFETCH_PROVENANCE_INIT
        $o['candidate_activation'] = 'opt-in-fixed'
        $o['warning'] = ('prefetch_init_v1 is an unvalidated starting point (K=n_expert_used, ' +
                         'N=min(4,QD-1,floor(QD/2))): no performance claim is made for this pair')
        return $o
    }

    # 4. the adapt opt-in on a non-catalog-fixed row.
    if ($OptIn -ceq 'adapt') {
        return (New-PfOff -Base $base -Reason (Get-PrefetchAdaptRefusal -ReproOrBench $ReproOrBench))
    }

    # 5. total-function tail: an arm nothing above claimed is treated like an unreadable row.
    return (New-PfOff -Base $base -Reason 'catalog_semantic_invalid')
}

# ---------------------------------------------------------------------------------------------
# P4 2 "arm selection happens BEFORE the QD is chosen". This answers one question only - which arm
# the QD sweep should prefer - and it must agree with Resolve-EffectivePrefetch on every refusal,
# because a refused opt-in that still moved the QD would change the run it was refused for.
# It reads no QD and produces no K/N.
# ---------------------------------------------------------------------------------------------
function Resolve-PrefetchArm {
    param($Profile, [string] $OptIn = 'none', [string] $IdentityVerdict = 'unpinned',
          $DerivedHeaderExpertUsed = $null, [bool] $ReproOrBench = $false)
    # Two placeholders, both deliberate. Probe success is assumed because a probe FAILURE only ever
    # turns the arm off and the sweep's own degraded path already forces QD1 - so it cannot change
    # the answer this function exists to give, and the probe result does not exist yet anyway.
    # EffectiveQd=2 is the smallest depth at which the init formula yields a pair; the arm question
    # itself is QD-independent, and using the real QD here would be the circular dependency PI 4-2
    # step 5 exists to avoid.
    $d = Resolve-EffectivePrefetch -Profile $Profile -ProbeOk $true -OptIn $OptIn `
             -EffectiveQd 2 -IdentityVerdict $IdentityVerdict `
             -DerivedHeaderExpertUsed $DerivedHeaderExpertUsed -ReproOrBench $ReproOrBench
    if (-not $d.on) { return $script:PREFETCH_ARM_NONE }
    if ([string]$d.candidate_activation -ceq 'catalog-fixed') { return $script:PREFETCH_ARM_CATALOG }
    return $script:PREFETCH_ARM_INIT
}

# LS 1-2 (R6 revision) - the QD-dependent row of the same table, kept separate because effective_qd
# is only known after preset / CLI / interactive-custom overrides have been folded in. It is applied
# inside Build-EffectiveConfig, which is what makes it re-decided on EVERY config rebuild.
# The launcher never raises QD to N+1 and never clamps the catalog N to QD-1: the catalog K/N pair
# is a locked, verified combination, so an unverified one is not synthesised here.
function Resolve-PrefetchForQd {
    param($Decision, [int] $EffectiveQd)
    if ($null -eq $Decision -or -not $Decision.on) { return $Decision }
    if ([long]$EffectiveQd -ge ([long]$Decision.n + 1)) { return $Decision }
    # P4 3 mandatory echo: a demotion changes the verdict, not the row's story. The catalog axes,
    # the request and the identity verdict travel with the demoted decision so the status screen and
    # the EFFECTIVE record still say WHICH row was demoted and what had been asked for.
    $o = @{}
    foreach ($k in $Decision.Keys) { $o[$k] = $Decision[$k] }
    $o['on'] = $false
    $o['echo'] = $script:PREFETCH_ECHO_QD_BELOW_DEPTH
    $o['off_reason'] = 'qd_below_prefetch_depth'
    $o['candidate_activation'] = 'off'
    $o['provenance'] = $null
    $o['qd_demoted'] = $true
    $o['catalog_k'] = $Decision.k
    $o['catalog_n'] = $Decision.n
    $o['effective_qd'] = [int]$EffectiveQd
    $o.Remove('k') | Out-Null
    $o.Remove('n') | Out-Null
    return $o
}

# The one place that folds a QD override into the swept default. P4 2 puts the init N0 derivation
# on the FINAL effective QD, so the resolver and this builder have to agree on that number exactly -
# which they only do if they compute it the same way, here.
function Get-EffectiveQd {
    param($Overrides, [int] $Qd)
    if ($null -ne $Overrides -and $Overrides.ContainsKey('qd')) { return [int]$Overrides['qd'] }
    return [int]$Qd
}

function Build-EffectiveConfig {
    param($Catalog, $Profile, [string] $Root, [string] $OutputDir, [string] $ModelPath,
          $Overrides, $PrefetchDecision, [int] $Qd)
    $d = Get-JsonValue -Obj $Profile -Name 'defaults'
    $argv = @()
    foreach ($a in (Get-JsonArray -Obj $d -Name 'argv')) { $argv += [string]$a }

    # model path is launcher-controlled (locked): the catalog cannot know where the user put it.
    $argv = Set-ArgvValue -Argv $argv -Flag '-m' -Value $ModelPath

    # locked layer: everything not listed below stays exactly as the catalog declared it.
    $port = $null
    if ($Overrides.ContainsKey('port')) { $port = [string]$Overrides['port'] } else { $port = Get-ArgvValue -Argv $argv -Flag '--port' }
    if ($null -eq $port) { Stop-Launcher 'fail_gate_catalog' 'catalog defaults.argv has no --port and no override supplied' }
    $argv = Set-ArgvValue -Argv $argv -Flag '--port' -Value $port
    if ($Overrides.ContainsKey('ctx'))     { $argv = Set-ArgvValue -Argv $argv -Flag '-c' -Value ([string]$Overrides['ctx']) }
    if ($Overrides.ContainsKey('threads')) { $argv = Set-ArgvValue -Argv $argv -Flag '-t' -Value ([string]$Overrides['threads']) }

    # R2-2 warmup single owner: the ENGINE warmup is always off - '--no-warmup' is forced into the
    # effective argv in every configuration. warmup=on only enables the launcher's own post-ready
    # warmup request (Invoke-LauncherWarmup). Two reasons this direction and not the other:
    #   - an engine-internal warmup failure happens BEFORE ready, so it can only surface as
    #     fail_server_start/5 and can never reach the non-terminal degraded branch RS 5 requires;
    #   - one observable warmup owner keeps startup deterministic and the failure reportable.
    # UX 1-4: the launcher-side default is ON since v0.2.3 (a generic one-token request after ready,
    # a few seconds). The engine side is untouched by that reversal - '--no-warmup' below is still
    # forced in every configuration, so there is still exactly one warmup owner.
    $warm = $script:WARMUP_PRODUCT_DEFAULT
    if ($Overrides.ContainsKey('warmup')) { $warm = [string]$Overrides['warmup'] }
    if (-not ($argv -ccontains '--no-warmup')) { $argv += '--no-warmup' }
    # UX 1-5 (Codex build r1 M4): the bench force is DECIDED here, beside the value it overrides, and
    # APPLIED at the tail of this function - the two halves are split on purpose. The decision has to
    # exist this early because the BUDGET_AUTOTUNE record further down claims (or declines) the
    # measured operating point, and that claim now includes the warmup dimension; it must therefore
    # see the value this run will really serve, not the pre-force one. The application stays at the
    # tail so it remains the LAST write to $warm and no override layer can re-raise it.
    # This is the only place the bench flags DECIDE the warmup value. Test-WarmupOverrideNeutral
    # reads the same two flags for a different question (is a stored override still performance
    # custom - UX 1-4), so a grep finds two sites and only this one owns the value.
    $warmForcedReason = $null
    if ($Smoke -or $Repro) { $warmForcedReason = $script:WARMUP_FORCED_REASON_BENCH }
    $warmFinal = $warm
    if ($null -ne $warmForcedReason) { $warmFinal = 'off' }

    # locked layer, restated at build time so a catalog typo cannot relax it (RS 7-3).
    if (-not ($argv -ccontains '-np')) { $argv += @('-np', '1') }
    else { $argv = Set-ArgvValue -Argv $argv -Flag '-np' -Value '1' }
    $host0 = Get-ArgvValue -Argv $argv -Flag '--host'
    if ($null -eq $host0) { $argv = Set-ArgvValue -Argv $argv -Flag '--host' -Value '127.0.0.1' ; $host0 = '127.0.0.1' }
    if (-not (Test-LoopbackAddress -Address $host0)) {
        Stop-Launcher 'fail_gate_catalog' ('catalog defaults.argv binds a non-loopback host (' + $host0 + '); loopback-only is locked (RS 7-3)')
    }

    # LS 13-2 (WS-1): the warmstart override is an allowlist key, so it is bound here - on every
    # rebuild, before the argument surface is fixed. LS 13-1 then requires the directory to exist
    # BEFORE --slot-save-path is injected (the server refuses to start when it does not), and the
    # A-1 latch makes a failed guarantee turn the whole feature off for this run rather than
    # terminating. Doing both here is what keeps argv, the status line and the diagnostic log
    # describing the same decision.
    $wsOverride = 'on'
    if ($Overrides.ContainsKey('warmstart')) { $wsOverride = [string]$Overrides['warmstart'] }
    $script:WarmstartCtx.override = $wsOverride
    # LS 13-8: autosave is bound on the same rebuild. It takes no part in argv or env - it only
    # decides whether the serving loop ticks - so it is resolved here purely to keep one binding
    # point per rebuild.
    $asOverride = 'on'
    if ($Overrides.ContainsKey('autosave')) { $asOverride = [string]$Overrides['autosave'] }
    Set-AutosaveSetting -Value $asOverride
    $argv = Remove-ArgvPair -Argv $argv -Flag $script:ARG_SLOT_SAVE_PATH
    $dirOk = Confirm-WarmstartDirectory -Stage 'config'
    # LS 13-1 (2): the profile cap runs once, right here - after the directory guarantee (so the ON
    # row counts and pins a current directory that now exists) and BEFORE the eligibility decision
    # at the tail of this function.
    Invoke-KvProfileCapOnce
    if ($dirOk) {
        $argv = Set-ArgvValue -Argv $argv -Flag $script:ARG_SLOT_SAVE_PATH -Value ($script:WarmstartCtx.dir.TrimEnd('\') + '\')
    }

    $env0 = @{}
    $de = Get-JsonValue -Obj $d -Name 'env'
    foreach ($k in (Get-JsonKeys -Obj $de)) { $env0[$k] = [string](Get-JsonValue -Obj $de -Name $k) }
    $env0[$script:ENV_DIRECT]      = '1'
    $env0[$script:ENV_DIRECT_DIR]  = $OutputDir
    $env0[$script:ENV_EXPECTS_DIR] = (Join-Path $Root ([string](Get-JsonValue -Obj (Get-JsonValue -Obj $Catalog -Name 'runtime') -Name 'expects_dir')))
    # BUDGET_AUTOTUNE_SPEC v0.2: the default budget is the autotune's answer and an explicit value
    # (CLI / stored preset / interactive custom - they all arrive in $Overrides) outranks it. Both
    # decisions live in Resolve-BudgetAutotune, so a rebuild re-decides the budget exactly like it
    # re-decides the prefetch QD row above, and this build's BUDGET_AUTOTUNE record always lands
    # before the caller's EFFECTIVE record. -PerfCustom is Test-CustomProvenance over THIS SAME
    # overrides map - the map the main flow hands to that same function for the EFFECTIVE record's
    # provenance/performance_gate - so the two records cannot answer the identity question from
    # two different judgments (r2 F2-b).
    $bt = Resolve-BudgetAutotune -Profile $Profile -Overrides $Overrides `
              -Installed (Get-InstalledMemoryMib) `
              -Geometry (Get-BudgetSlotGeometry -OutputDir $OutputDir) `
              -Mem (Get-MemStatus) -ReproMode ([bool]$Repro) `
              -PerfCustom (Test-CustomProvenance -Overrides $Overrides) `
              -WarmPath (Test-WarmPathBaseline -Config @{ warmup = $warmFinal })
    $budget = [long]$bt.budget_mb
    $env0[$script:ENV_BUDGET_MB] = [string]$budget
    $qdEff = Get-EffectiveQd -Overrides $Overrides -Qd $Qd
    $env0[$script:ENV_QD] = [string]$qdEff

    # Default metrics accounting log (LAUNCHER_SPEC 8b: stop -> "graceful cleanup(metrics summary
    # flush)" is only real if MOE_DIRECT_METRICS points somewhere). A value already supplied -
    # via the profile/preset/custom env section above, or set ambient by the caller's own shell -
    # is respected as-is; the launcher only fills the gap.
    $metricsInjected = $false
    if (-not $env0.ContainsKey($script:ENV_METRICS)) {
        $extMetrics = [System.Environment]::GetEnvironmentVariable($script:ENV_METRICS)
        if ($extMetrics) {
            $env0[$script:ENV_METRICS] = $extMetrics
        } else {
            $logDir = Join-Path (Get-LauncherStateDir) 'logs'
            if (-not (Test-Path -LiteralPath $logDir -PathType Container)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
            $env0[$script:ENV_METRICS] = Join-Path $logDir ('metrics_{0}_{1}.jsonl' -f (Get-Date -Format 'yyyyMMddTHHmmss'), $PID)
            $metricsInjected = $true
        }
    }
    # Codex xcheck (reviews/codex_warmstart_a_xcheck.md, 편승 563ccc8 #2): validated here regardless
    # of provenance (profile env / ambient / generated default above) - the child's working
    # directory is the bundle root, so an unvalidated relative or in-bundle value becomes an
    # unlisted bundle file that the NEXT launch self-rejects as fail_gate_bundle.
    $metricsRaw = $env0[$script:ENV_METRICS]
    if (-not [System.IO.Path]::IsPathRooted($metricsRaw)) {
        Stop-Launcher 'fail_gate_catalog' ('MOE_DIRECT_METRICS must be an absolute path (relative): ' + $metricsRaw)
    }
    $metricsCanonical = [System.IO.Path]::GetFullPath($metricsRaw)
    $rootWithSep = $Root.TrimEnd('\') + '\'
    if ($metricsCanonical.Equals($Root, [System.StringComparison]::OrdinalIgnoreCase) -or
        $metricsCanonical.StartsWith($rootWithSep, [System.StringComparison]::OrdinalIgnoreCase)) {
        Stop-Launcher 'fail_gate_catalog' ('MOE_DIRECT_METRICS must be outside the bundle root (inside-bundle): ' + $metricsRaw)
    }
    $env0[$script:ENV_METRICS] = $metricsCanonical
    Write-Diag -Kind 'METRICS_ENV' -Data @{ injected = $metricsInjected; path = $env0[$script:ENV_METRICS] }

    # LS 1-2 (R6): effective_qd is final only here, so the QD row of the table is decided here -
    # every rebuild (a custom edit, a stored preset, a CLI override) re-decides it.
    $pf = Resolve-PrefetchForQd -Decision $PrefetchDecision -EffectiveQd $qdEff

    # LS 1-2: outside the single 'on' row the raw K/N keys must be ABSENT, not set to zero.
    $env0.Remove($script:ENV_PREFETCH_K) | Out-Null
    $env0.Remove($script:ENV_PREFETCH_N) | Out-Null
    $env0.Remove($script:ENV_NO_PREFETCH) | Out-Null
    if ($pf.on) {
        $env0[$script:ENV_PREFETCH_K] = [string]$pf.k
        $env0[$script:ENV_PREFETCH_N] = [string]$pf.n
    } else {
        $env0[$script:ENV_NO_PREFETCH] = '1'
    }
    if ($pf.qd_demoted) {
        Write-Diag -Kind 'PREFETCH_QD_DEMOTED' -Data @{ effective_qd = $qdEff; catalog_n = $pf.catalog_n
                                                        catalog_k = $pf.catalog_k; echo = $pf.echo }
    }

    # LS 13-1: config-dependent eligibility is re-decided on EVERY returned effective config, and it
    # is done HERE rather than at each call site so a future caller cannot forget it. The status
    # screen renders $script:WarmstartCtx.status_text, so the 'kv :' line can never lag behind a
    # custom edit.
    Update-WarmstartEligibility -Argv $argv -EnvVars $env0

    # UX 1-5: bench protection APPLIED, and this is the LAST thing that touches $warm on purpose.
    # Every runtime config in this launcher is produced by this one function, and the ready-side
    # warmup consumes $Config.warmup alone, so overwriting the value here - after the CLI/preset/
    # custom layers have all been folded in - is the single point no later settings layer can
    # re-raise. The decision itself was taken next to the binding above (see the note there); this
    # is only where it takes effect. The reason travels IN the config rather than being recorded
    # here, because this function re-runs on every custom edit and the record must happen exactly
    # once, at ready.
    $warm = $warmFinal

    return @{ argv = $argv; env = $env0; port = [int]$port; budget_mb = $budget; qd = $qdEff;
              warmup = $warm; prefetch = $pf; host = $host0; warmstart = $wsOverride
              autosave = $asOverride; warmup_forced_reason = $warmForcedReason
              budget_source = $bt.source; budget_unmeasured = [bool]$bt.unmeasured }
}

# LS 13-2: performance-neutral override keys do not make a configuration "custom". They change no
# argv, no env and no measurement condition, so demoting the performance gate for them would punish
# a user for following the README.
# Separate workstreams then added VALUE-dependent members to that idea, and the rule that admits one
# is narrow: KEY PRESENCE stays the general test, and a key is exempted only where its final VALUE
# proves the run still sits on the measured condition. Each exemption carries that proof at its own
# key below; this is not a closed list, and nothing joins it without one.
#   'prefetch' (P4 4) normally IS a performance condition (an init opt-in injects a K/N pair the
#   catalog never measured), but two of its outcomes change nothing at all: an explicit 'catalog'
#   request is the default behaviour spelled out, and a request the resolver IGNORED (init on a
#   catalog-fixed row) leaves the measured pair exactly where it was. Demoting the published numbers
#   for a request that changed no argv and no env would be a false statement in the other direction.
#   'warmup' (UX 1-4) is decided by the final value because the v0.2.2 advice was "switch warmup on"
#   and that stored key is now the product default - a user who followed the README must not be
#   reclassified as custom by the default reversal itself.
$script:PrefetchRequestIgnored = $false
function Test-CustomProvenance {
    param($Overrides)
    foreach ($k in $Overrides.Keys) {
        if ($script:PERF_NEUTRAL_OVERRIDE_KEYS -contains [string]$k) { continue }
        if ([string]$k -ceq 'prefetch') {
            if ([string]$Overrides[$k] -ceq 'catalog') { continue }
            if ($script:PrefetchRequestIgnored) { continue }
        }
        if ([string]$k -ceq 'warmup' -and (Test-WarmupOverrideNeutral -Value ([string]$Overrides[$k]))) { continue }
        return $true
    }
    return $false
}

# UX 1-4: a stored warmup override is performance-neutral when it cannot change the run's warmup
# behaviour - either because it equals the product default, or because UX 1-5 has already forced the
# value off and the stored key no longer decides anything. The second half is the frozen wording
# "a stored warmup override invalidated by the -Repro/-Smoke force is not counted as performance
# custom": counting it would demote a bench run for a setting that run does not use.
function Test-WarmupOverrideNeutral {
    param([string] $Value)
    if ($Smoke -or $Repro) { return $true }
    return ([string]$Value -ceq $script:WARMUP_PRODUCT_DEFAULT)
}

# UX 1-4: the warmup dimension of the performance gate, as one predicate so the screen and the
# EFFECTIVE record cannot answer it differently. Only 'off' matches the official cold-cache
# condition; 'on' (the product default) and 'file:<path>' both leave the machine warm before the
# first measured token. It says nothing about the OTHER gate requirements - catalog performance
# validation, measured-budget identity and the absence of performance-changing overrides are all
# still required on their own (UX 1-4, "no over-claiming").
function Test-WarmPathBaseline {
    param($Config)
    return ([string]$Config.warmup -cne 'off')
}

function Test-LoopbackAddress {
    param([string] $Address)
    if ($Address -eq 'localhost') { return $true }
    $ip = $null
    if ([System.Net.IPAddress]::TryParse($Address, [ref]$ip)) { return [System.Net.IPAddress]::IsLoopback($ip) }
    return $false
}

# endregion

# ============================================================================
# region 14b. WARMSTART / slot-save wiring (LS 13 = WS-1)
#   Value authority : WARMSTART_SPEC.md v0.5 FROZEN (A-1 .. A-9)
#   Wiring authority: LAUNCHER_SPEC 13 (insertion points, argument surface, truth table,
#                     recovery state machine, env block, diagnostic enum, selftest duties)
#
#   Two invariants hold for everything below and are worth stating once:
#     - No new status enum and no new exit code. Every failure here is NON-TERMINAL and degraded
#       (LS 13-3). The single exception is the restore ladder's last rung, which reuses the
#       existing fail_server_start, and old-child cleanup failure, which reuses fail_teardown.
#     - The mode decision comes from the CLI switches ONLY and is taken before any preset, catalog
#       or custom layer is read, so no later layer can turn a hard-OFF run back on.
# ============================================================================

# A-6 mode source. -Smoke and -Repro both mean hard-OFF; giving both is not a contradiction because
# they converge on the same result, so no combination check is needed.
function Get-WarmstartMode {
    param([bool] $SmokeMode, [bool] $ReproMode)
    if ($SmokeMode -or $ReproMode) { return 'hard_off' }
    return 'product'
}

function New-WarmstartState {
    return @{
        initialized      = $false        # until Initialize-Warmstart runs, everything behaves hard-OFF
        mode             = 'hard_off'    # product | hard_off  (from the CLI switches, A-6)
        override         = 'on'          # on | off            (soft-OFF allowlist key)
        latched_off      = $false        # A-1 directory latch: whole feature off for this run
        latch_reason     = $null
        profile_id       = $null
        dir              = $null
        root             = $null
        bundle_sha       = $null
        manifest_sha     = $null
        model_set        = $null
        model_shas       = $null         # process-once cache (A-4 file-dependent input)
        model_shas_attempted = $false    # process-once FAILURE latch: no repeated multi-GB retries
        model_shas_error = $null
        file_facts       = $null         # process-once cache (A-4 file-dependent input)
        eligible         = $false
        reason           = 'off_mode'
        status_text      = 'off(mode)'
        detail           = $null
        restore_latched  = $false        # 13-4b (1): restore is attempted at most once per process
        restore_done     = $false
        recovery_count   = 0
        gc_done          = $false        # LS 13-1 (1) generation sweep, once at lock time
        # LS 13-1 (2) profile cap latch, STATE TRANSITION rather than a plain once-flag:
        # none -> soft_off -> on is a real second run (a custom off->on materialises the current
        # directory the soft-OFF pass could not count), while a rebuild in the same state is not.
        cap_state        = 'none'        # none | soft_off | on
        unavailable_warned = $false      # one operational-failure warning per run (A-2 (6))
        dir_ready        = $false
        hash_notice      = $false
        tmp_after_join   = @()           # A-2 (8) recovery timing (2): after the child is joined
        stale_after_stop = @()           # A-4b (5) recovery timing (3): at the end of teardown
        # --- LS 13-8 / AUTOSAVE_SPEC ---
        auto_facts       = @{}           # per-generation file-facts cache (A-4 evaluation split)
        selected_name    = $script:KV_CANONICAL_DATA   # C: the candidate the restore will ask for
        selected_origin  = $script:KV_ORIGIN_STOP
        selected_saved_at = $null
        autosave_setting = 'on'          # on | off | <minutes>, as parsed from the allowlist key
        autosave_minutes = $script:KV_AUTOSAVE_DEFAULT_MIN
        autosave_enabled = $true         # the key's own verdict; the warmstart state gates it again
        autosave_next    = $null         # the generation the next write targets (decided lazily)
        autosave_tokens  = $null         # A: last saved/restored sequence length ($null = nothing yet)
        autosave_clock   = $null         # A: the running tick deadline
        autosave_count   = 0
        autosave_stopped = $false        # a save whose answer never came latches the feature off
        autosave_warned  = $false        # one operational warning per run (A-2 (6) shape)
    }
}

$script:WarmstartCtx = New-WarmstartState
# Decided from the bound parameters, at load time - ahead of the catalog, the preset, the override
# merge and Build-EffectiveConfig, exactly as A-6 requires.
$script:WarmstartModeSwitch = Get-WarmstartMode -SmokeMode ([bool]$Smoke) -ReproMode ([bool]$Repro)

function Get-KvRootDir { return (Join-Path (Get-LauncherStateDir) $script:KV_DIR_NAME) }

function Get-KvProfileDir {
    param([string] $ProfileId)
    return (Join-Path (Get-KvRootDir) $ProfileId)
}

# The four rows of the LS 13-2 truth table collapse to this one answer.
#   hard_off     -Smoke / -Repro          : no argument, NO contact with the kv tree at all
#   soft_off     warmstart=off            : no argument, no POST, but the GC still runs
#   latched_off  A-1 directory failure    : no argument, no POST, non-terminal degraded
#   on           product default          : argument injected, A-2 / A-3 apply
function Get-WarmstartState {
    if (-not $script:WarmstartCtx.initialized) { return 'hard_off' }
    if ($script:WarmstartCtx.mode -cne 'product') { return 'hard_off' }
    if ($script:WarmstartCtx.override -ceq 'off') { return 'soft_off' }
    if ($script:WarmstartCtx.latched_off) { return 'latched_off' }
    return 'on'
}

function Test-WarmstartActive { return ((Get-WarmstartState) -ceq 'on') }

# LS 13-6: exactly one reason value per reachable cold path, and the status text is a pure
# rendering of it.
function Get-KvStatusText {
    param([bool] $Eligible, [string] $Reason)
    if ($Eligible) { return 'eligible' }
    if ($Reason -ceq $script:KV_REASON_OFF_USER) { return 'off(user)' }
    if ($Reason -ceq $script:KV_REASON_OFF_MODE) { return 'off(mode)' }
    return ('cold(' + $Reason + ')')
}

function Set-KvVerdict {
    param([bool] $Eligible, [string] $Reason, $Detail)
    $script:WarmstartCtx.eligible = $Eligible
    $script:WarmstartCtx.reason = $Reason
    $script:WarmstartCtx.detail = $Detail
    $script:WarmstartCtx.status_text = (Get-KvStatusText -Eligible $Eligible -Reason $Reason)
}

# A-1 / LS 13-1: either confirmation failing switches the whole feature off for this run. It is a
# warning plus a diagnostic line, never a termination - the server still starts, just cold.
function Set-WarmstartOffLatch {
    param([string] $Stage, [string] $Reason)
    if ($script:WarmstartCtx.latched_off) { return }
    $script:WarmstartCtx.latched_off = $true
    $script:WarmstartCtx.latch_reason = $Reason
    Write-Line ('[kv] WARNING: warmstart is off for this run (' + $Reason + '); the server starts normally without it.')
    Write-Diag -Kind 'WARMSTART_OFF_LATCH' -Data @{ stage = $Stage; reason = $Reason
                                                    effect = 'no --slot-save-path, no restore, no save' }
}

# A-1 directory guarantee. Called on every effective-config build (which covers the "at GC time"
# confirmation, because the first build is the first moment the soft-OFF override is known) and
# once more immediately before the child is spawned, ahead of the final EFFECTIVE diagnostic.
function Confirm-WarmstartDirectory {
    param([string] $Stage)
    if ((Get-WarmstartState) -cne 'on') { return $false }
    $dir = $script:WarmstartCtx.dir
    if (-not $dir) {
        Set-WarmstartOffLatch -Stage $Stage -Reason 'kv directory path was never resolved'
        return $false
    }
    try {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) {
            New-Item -ItemType Directory -Path $dir -Force -ErrorAction Stop | Out-Null
        }
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { throw 'directory still absent after create' }
    } catch {
        Set-WarmstartOffLatch -Stage $Stage -Reason ('kv directory could not be created: ' + $_.Exception.Message)
        return $false
    }
    if (-not $script:WarmstartCtx.dir_ready) {
        $script:WarmstartCtx.dir_ready = $true
        Write-Diag -Kind 'WARMSTART_DIR' -Data @{ stage = $Stage; dir = $dir }
    }
    return $true
}

# LS 13-1 startup insertion point: called straight after the exclusive locks and before any
# eligibility decision. hard-OFF returns without touching the kv tree in any way.
# LS OA-1 / OPEN_ARCH_DESIGN section 3: warmstart and autosave MAY be switched on for a derived
# profile - nothing about the sidecar machinery is catalog specific - but only once four conditions
# hold together. Three of them are wiring questions, the fourth is a measurement, and the verdict is
# the M5 end-to-end round's to make, not this one's. Until then the predicate answers false, which
# is what makes v0.2.1 ship the derived path with both features OFF. The conditions are stated here
# rather than in a comment so the eventual flip is a data change with a record behind it.
function Test-DerivedWarmstartGate {
    $c = [ordered]@{
        # a stable id is required or every boot would look like a different profile
        deterministic_profile_id             = $true
        # the sidecar binding must carry template id/version + the derived expect digest, so a
        # re-derivation under a changed template can never restore onto the old state
        manifest_binding_carries_template    = $false
        # restore only after the engine seal and after ready - already true of the existing ladder
        restore_after_seal_and_ready         = $true
        # measured, not argued: a cold save -> restore and an autosave generation restore, both
        # passing in the M5 end-to-end round
        e2e_cold_and_autosave_restore_passed = $false
    }
    $unmet = @()
    foreach ($k in $c.Keys) { if (-not $c[$k]) { $unmet += [string]$k } }
    return @{ eligible = ($unmet.Count -eq 0); conditions = $c; unmet = @($unmet) }
}

function Initialize-Warmstart {
    param([string] $ProfileId, [bool] $DerivedProfile = $false)
    $script:WarmstartCtx.initialized = $true
    $script:WarmstartCtx.mode = $script:WarmstartModeSwitch
    $derivedGate = $null
    if ($DerivedProfile -and $script:WarmstartCtx.mode -ceq 'product') {
        $derivedGate = Test-DerivedWarmstartGate
        if (-not $derivedGate.eligible) {
            # Reuses the existing hard-OFF row of the LS 13-2 truth table rather than inventing a
            # fifth: no --slot-save-path, no contact with the kv tree, "kv : off(mode)", autosave
            # off with it. LS 13-6's reason enum is frozen, so the real cause is carried by the
            # diagnostic record below instead of by a new reason string.
            $script:WarmstartCtx.mode = 'arch_template'
        }
        Write-Diag -Kind 'WARMSTART_DERIVED_GATE' -Data @{ eligible = [bool]$derivedGate.eligible
                                                            unmet = @($derivedGate.unmet)
                                                            conditions = $derivedGate.conditions }
    }
    $script:WarmstartCtx.profile_id = $ProfileId
    $script:WarmstartCtx.dir = Get-KvProfileDir -ProfileId $ProfileId
    Write-Diag -Kind 'WARMSTART_MODE' -Data @{ mode = $script:WarmstartCtx.mode; smoke = [bool]$Smoke
                                               repro = [bool]$Repro; profile = $ProfileId
                                               derived = [bool]$DerivedProfile
                                               dir = $script:WarmstartCtx.dir }
    if ($script:WarmstartCtx.mode -cne 'product') {
        Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_OFF_MODE -Detail $null
        return
    }
    # LS 13-1 (1): only the generation sweep runs here. The profile cap needs the FINAL warmstart
    # override - soft-OFF never creates a current directory, so counting one in would recover a
    # profile that was never over the cap - and the override is not known until the first
    # Build-EffectiveConfig, which is where phase two runs.
    Invoke-KvGenerationSweep
    $script:WarmstartCtx.gc_done = $true
}

function Set-WarmstartBindings {
    param([string] $Root, [string] $ManifestSha, $ModelSet)
    $script:WarmstartCtx.root = $Root
    $script:WarmstartCtx.manifest_sha = ([string]$ManifestSha).ToLowerInvariant()
    $script:WarmstartCtx.model_set = $ModelSet
}

# ---------------------------------------------------------------------------------------------
# A-4b GC. Runs once, at startup, under the instance lock, before any eligibility decision.
# Retention is 0 for both .tmp and .stale (the failure ladder is mismatch -> cold, never a rollback
# onto a stale generation), and at most KV_PROFILE_RETENTION profile directories survive.
# ---------------------------------------------------------------------------------------------
function Get-KvProfileCanonicalTicks {
    param([string] $Dir)
    # A-4b: a profile without a COMPLETE data+meta pair is recovered first. Int64.MinValue is that
    # "oldest possible" marker, and it is reached normally - a profile whose very first save failed,
    # or whose startup died before one, never has a pair.
    # LS 13-8: recency is the newest of EVERY complete pair the profile holds, not the canonical's
    # alone. An autosave generation is a restorable recovery point by exactly the same machine, so a
    # profile that has never managed a normal stop - which is precisely the crash-prone machine this
    # feature exists for - must not be ranked "oldest possible" and deleted ahead of a profile whose
    # canonical is a year old. The cap itself is untouched (it counts directories, not files).
    try {
        $best = [long][Int64]::MinValue
        foreach ($n in (@([string]$script:KV_CANONICAL_DATA) + @($script:KV_AUTOSAVE_GENERATIONS))) {
            $d = Join-Path $Dir $n
            $m = Join-Path $Dir (Get-KvMetaName -Name $n)
            if (-not (Test-Path -LiteralPath $d -PathType Leaf)) { continue }
            if (-not (Test-Path -LiteralPath $m -PathType Leaf)) { continue }
            $t1 = [long](Get-Item -LiteralPath $d).LastWriteTimeUtc.Ticks
            $t2 = [long](Get-Item -LiteralPath $m).LastWriteTimeUtc.Ticks
            $t = $t1
            if ($t2 -gt $t1) { $t = [long]$t2 }
            if ($t -gt $best) { $best = [long]$t }
        }
        return @{ ok = $true; ticks = [long]$best }
    } catch {
        # NOT the "oldest possible" marker: a profile whose timestamp cannot be READ is a profile
        # whose age is unknown, and ranking an unknown age first would make an unreadable directory
        # the first thing deleted. The caller drops it from the candidate list and records it.
        return @{ ok = $false; ticks = [long][Int64]::MinValue; reason = [string]$_.Exception.Message }
    }
}

# Ordinal ordering key for step (3). Real tick values are always positive, so the sentinel sorts
# first without any arithmetic that could overflow Int64.
function Get-KvProfileSortKey {
    param([long] $Ticks, [string] $Name)
    $rank = '0' + ('0' * 19)
    if ($Ticks -gt 0) { $rank = '1' + ([string]$Ticks).PadLeft(19, '0') }
    return ($rank + '|' + $Name)
}

function Remove-KvGenerationFiles {
    param([string] $Dir, $Failures)
    $names = @()
    try {
        foreach ($f in @(Get-ChildItem -LiteralPath $Dir -File -ErrorAction Stop)) {
            foreach ($pat in $script:KV_GC_PATTERNS) {
                # -like, not -Filter: the Win32 wildcard matcher used by -Filter also honours short
                # names and treats '.' specially, which would make the four patterns imprecise.
                if ($f.Name -like $pat) { $names += [string]$f.Name; break }
            }
        }
    } catch {
        $Failures.Add(('enumerate ' + $Dir + ': ' + $_.Exception.Message)) | Out-Null
        return 0
    }
    # Deterministic delete order (A-4b): ordinal file-name ascending.
    $arr = [string[]]$names
    if ($arr.Count -gt 1) { [Array]::Sort($arr, [StringComparer]::Ordinal) }
    $removed = 0
    foreach ($n in $arr) {
        try {
            Remove-Item -LiteralPath (Join-Path $Dir $n) -Force -ErrorAction Stop
            $removed++
        } catch {
            # Per-file failure: record and keep going. The GC never blocks startup.
            $Failures.Add(($n + ': ' + $_.Exception.Message)) | Out-Null
        }
    }
    return @{ matched = $arr.Count; removed = $removed }
}

# A-2 (6) fixes the shape of every housekeeping failure: one warning line plus one diagnostic
# record, never a status or an exit code. Each GC phase emits at most one of these.
function Write-KvGcDegraded {
    param([string] $Reason, $Data)
    $d = @{ reason = $Reason }
    if ($null -ne $Data) { foreach ($k in $Data.Keys) { $d[$k] = $Data[$k] } }
    Write-Diag -Kind $script:KV_DIAG_GC_FAILED -Data $d
    Write-Line ('[kv] WARNING: kv housekeeping did not fully complete (' + $Reason + '); the server starts normally.')
}

# LS 13-1 (1) PHASE ONE, at the exclusive lock: every existing profile's tmp/stale generations,
# retention 0. This phase never deletes a profile directory, so it is safe before the final
# warmstart override is known. hard-OFF does not reach it at all.
function Invoke-KvGenerationSweep {
    $root = Get-KvRootDir
    $failures = New-Object System.Collections.ArrayList
    $matched = 0
    $removedTmp = 0
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        # A-4b: an absent directory is not an error - there are simply zero targets.
        Write-Diag -Kind 'WARMSTART_GC' -Data @{ root = $root; phase = 'generations'; state = 'absent'
                                                 matched = 0; removed_generations = 0 }
        return
    }
    $dirs = @()
    try { $dirs = @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction Stop) }
    catch {
        Write-KvGcDegraded -Reason ('profile enumeration failed: ' + $_.Exception.Message) `
            -Data @{ root = $root; phase = 'generations' }
        return
    }
    foreach ($d in $dirs) {
        $r = Remove-KvGenerationFiles -Dir $d.FullName -Failures $failures
        if ($r -is [hashtable]) { $matched += [int]$r.matched; $removedTmp += [int]$r.removed }
    }
    if ($matched -gt $script:KV_GC_WARN_MATCHES) {
        # Crash-loop signal. The action is unchanged (delete them all); only the record differs.
        Write-Diag -Kind 'WARMSTART_GC_WARN' -Data @{ root = $root; matched = $matched
                                                      threshold = $script:KV_GC_WARN_MATCHES
                                                      note = 'unusually many leftover generations (crash loop signal)' }
    }
    Write-Diag -Kind 'WARMSTART_GC' -Data @{ root = $root; phase = 'generations'; matched = $matched
                                             removed_generations = $removedTmp; failures = @($failures) }
    if ($failures.Count -gt 0) {
        Write-KvGcDegraded -Reason 'some leftover generations could not be removed' `
            -Data @{ root = $root; phase = 'generations'; failures = @($failures) }
    }
}

# (5) the cap is re-checked against reality. Split out so the count is one testable answer rather
# than a swallowed exception: a failed re-check is a recorded failure, not a silent -1.
function Measure-KvProfileDirs {
    param([string] $Root)
    try { return @{ ok = $true; count = [int]@(Get-ChildItem -LiteralPath $Root -Directory -ErrorAction Stop).Count } }
    catch { return @{ ok = $false; count = -1; reason = [string]$_.Exception.Message } }
}

# LS 13-1 (2) PHASE TWO, at the FIRST effective config - the first moment the final warmstart
# override is known - and ahead of any eligibility decision. The cap is applied to the directories
# that REALLY exist: the ON path has just created the current one (A-1 first confirmation) so it is
# counted and pinned, while soft-OFF creates nothing and must not reserve a slot for a directory
# that will never exist (reserving one recovers a fourth profile that was never over the cap).
function Invoke-KvProfileCapAdjust {
    param([string] $CurrentProfileId, [bool] $Corrective = $false)
    $root = Get-KvRootDir
    $failures = New-Object System.Collections.ArrayList
    $removedProfiles = @()
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        Write-Diag -Kind 'WARMSTART_GC' -Data @{ root = $root; phase = 'profile_cap'; state = 'absent'
                                                 corrective = $Corrective
                                                 removed_profiles = @(); profiles_after = 0 }
        return
    }
    $dirs = @()
    try { $dirs = @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction Stop) }
    catch {
        Write-KvGcDegraded -Reason ('profile enumeration failed: ' + $_.Exception.Message) `
            -Data @{ root = $root; phase = 'profile_cap'; corrective = $Corrective }
        return
    }

    # (1) the current profile is pinned, (3) the rest are recovered oldest first.
    $entries = @()
    foreach ($d in $dirs) {
        if ($d.Name -ceq $CurrentProfileId) { continue }
        $t = Get-KvProfileCanonicalTicks -Dir $d.FullName
        if (-not $t.ok) {
            # (4) an unreadable timestamp removes the profile from the CANDIDATES, not from the
            # disk. It still counts against the cap, so the overflow simply stays recorded.
            $failures.Add(('profile ' + $d.Name + ': timestamp unreadable - ' + [string]$t.reason)) | Out-Null
            continue
        }
        $entries += @{ name = [string]$d.Name; path = [string]$d.FullName
                       key = (Get-KvProfileSortKey -Ticks ([long]$t.ticks) -Name ([string]$d.Name)) }
    }
    $excess = @($dirs).Count - $script:KV_PROFILE_RETENTION
    if ($excess -gt 0 -and $entries.Count -gt 0) {
        $keys = [string[]]@($entries | ForEach-Object { $_.key })
        if ($keys.Count -gt 1) { [Array]::Sort($keys, [StringComparer]::Ordinal) }
        foreach ($k in $keys) {
            if ($excess -le 0) { break }
            $hit = $null
            foreach ($e in $entries) { if ($e.key -ceq $k) { $hit = $e; break } }
            if ($null -eq $hit) { continue }
            try {
                Remove-Item -LiteralPath $hit.path -Recurse -Force -ErrorAction Stop
                $removedProfiles += $hit.name
                $excess = $excess - 1
            } catch {
                # (4) an individual failure moves on to the next candidate.
                $failures.Add(('profile ' + $hit.name + ': ' + $_.Exception.Message)) | Out-Null
            }
        }
    }

    # (5) the cap is re-checked against reality, and only a real overflow is degraded.
    $m = Measure-KvProfileDirs -Root $root
    $final = [int]$m.count
    if (-not $m.ok) { $failures.Add(('final profile count failed: ' + [string]$m.reason)) | Out-Null }
    Write-Diag -Kind 'WARMSTART_GC' -Data @{ root = $root; phase = 'profile_cap'; corrective = $Corrective
                                             removed_profiles = $removedProfiles; profiles_after = $final
                                             retention = $script:KV_PROFILE_RETENTION
                                             failures = @($failures) }
    if ($final -gt $script:KV_PROFILE_RETENTION -or $failures.Count -gt 0) {
        Write-KvGcDegraded -Reason ('the profile cap could not be fully enforced (profiles_after=' + $final + ')') `
            -Data @{ root = $root; phase = 'profile_cap'; profiles_after = $final
                     retention = $script:KV_PROFILE_RETENTION; failures = @($failures) }
    }
}

# LS 13-1 (2) latch = a state transition, not a plain once-flag. The three-choice loop rebuilds the
# effective config on every custom edit and the cap must not walk the tree on each of them - but
# `warmstart` IS one of those custom keys, and an off -> on edit CREATES the current directory that
# the soft-OFF pass deliberately did not count. Without a corrective run that path reaches five
# profile directories through the normal UI alone (soft-OFF's documented "on" return contract and
# the cap of four have to hold at the same time).
#   none      -> first adjustment, in whichever state the first build resolved to
#   soft_off  -> on : one corrective adjustment, then latched
#   on        -> terminal: the current directory already exists and is already counted, so no later
#                rebuild (in either direction) can add a directory the cap has not seen
function Invoke-KvProfileCapOnce {
    if (-not $script:WarmstartCtx.initialized) { return }
    # hard-OFF performs neither phase: the kv tree is untouched (LS 13-2 truth table row 1/2).
    if ($script:WarmstartCtx.mode -cne 'product') { return }
    if ($script:WarmstartCtx.cap_state -ceq 'on') { return }
    # latched_off counts as soft_off here: its current directory could not be created either, so
    # there is nothing extra to count.
    $target = 'soft_off'
    if ((Get-WarmstartState) -ceq 'on') { $target = 'on' }
    if ($script:WarmstartCtx.cap_state -ceq $target) { return }
    $corrective = ($script:WarmstartCtx.cap_state -ceq 'soft_off')
    $script:WarmstartCtx.cap_state = $target
    Invoke-KvProfileCapAdjust -CurrentProfileId ([string]$script:WarmstartCtx.profile_id) -Corrective $corrective
}

# ---------------------------------------------------------------------------------------------
# A-4c canonical projection and its hash.
# The serialiser is written out by hand on purpose: ConvertTo-Json's escaping, key order and
# separator choices drift between PowerShell versions, and this value is a stored binding that has
# to reproduce byte for byte on a later build.
# ---------------------------------------------------------------------------------------------
function ConvertTo-KvJsonString {
    param([string] $Value)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    foreach ($ch in ([string]$Value).ToCharArray()) {
        $c = [int][char]$ch
        if ($ch -eq '"')  { [void]$sb.Append('\"'); continue }
        if ($ch -eq '\')  { [void]$sb.Append('\\'); continue }
        # Control characters and every non-ASCII UTF-16 unit become \uXXXX with lowercase hex. The
        # short forms (\n, \t, ...) are deliberately NOT used: one escape rule means one output.
        if ($c -lt 32 -or $c -gt 126) { [void]$sb.Append('\u'); [void]$sb.Append($c.ToString('x4')); continue }
        [void]$sb.Append($ch)
    }
    [void]$sb.Append('"')
    return $sb.ToString()
}

function ConvertTo-KvCanonicalJson {
    param([string[]] $Argv, [hashtable] $EnvVars)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('{"schema":1,"argv":[')
    $first = $true
    foreach ($a in @($Argv)) {
        if (-not $first) { [void]$sb.Append(',') }
        [void]$sb.Append((ConvertTo-KvJsonString -Value ([string]$a)))
        $first = $false
    }
    [void]$sb.Append('],"env":{')
    $keys = @()
    foreach ($k in $EnvVars.Keys) { $keys += [string]$k }
    $arr = [string[]]$keys
    if ($arr.Count -gt 1) { [Array]::Sort($arr, [StringComparer]::Ordinal) }
    $first = $true
    foreach ($k in $arr) {
        if (-not $first) { [void]$sb.Append(',') }
        [void]$sb.Append((ConvertTo-KvJsonString -Value $k))
        [void]$sb.Append(':')
        [void]$sb.Append((ConvertTo-KvJsonString -Value ([string]$EnvVars[$k])))
        $first = $false
    }
    [void]$sb.Append('}}')
    return $sb.ToString()
}

function Get-KvSemanticsProjection {
    param([string[]] $Argv, [hashtable] $EnvVars)
    $outArgv = @()
    $i = 0
    $src = @($Argv)
    while ($i -lt $src.Count) {
        $a = [string]$src[$i]
        $drop = $null
        foreach ($e in $script:KV_SEMANTICS_ARGV_DROP) { if ($a -ceq [string]$e.flag) { $drop = $e; break } }
        if ($null -ne $drop) { $i = $i + 1 + [int]$drop.arity; continue }
        $outArgv += $a
        $i = $i + 1
    }
    $outEnv = @{}
    foreach ($k in $EnvVars.Keys) {
        $skip = $false
        foreach ($d in $script:KV_SEMANTICS_ENV_DROP) {
            if ([string]::Equals([string]$k, $d, [System.StringComparison]::OrdinalIgnoreCase)) { $skip = $true; break }
        }
        # A-4c: the 26 OS bootstrap allowlist keys take no part in the projection either. They are
        # in the child's block by contract (LS 13-5) and $config.env is explicitly allowed to
        # override one of them, so hashing them would make an operator's PATH edit a cold start
        # while binding nothing: engine bytes are bound by engine_bundle_sha256 and the bundle's
        # DLLs resolve from the application directory ahead of PATH.
        if (-not $skip) {
            foreach ($d in $script:ENV_OS_BOOTSTRAP_ALLOWLIST) {
                if ([string]::Equals([string]$k, $d, [System.StringComparison]::OrdinalIgnoreCase)) { $skip = $true; break }
            }
        }
        if (-not $skip) { $outEnv[[string]$k] = [string]$EnvVars[$k] }
    }
    # No unary comma on the member: a hashtable value is stored as-is, and wrapping the array here
    # would make the caller's [string[]] binder flatten the whole argv into one joined string.
    return @{ argv = @($outArgv); env = $outEnv }
}

function Get-KvSemanticsSha256 {
    param([string[]] $Argv, [hashtable] $EnvVars)
    $p = Get-KvSemanticsProjection -Argv $Argv -EnvVars $EnvVars
    $doc = ConvertTo-KvCanonicalJson -Argv $p.argv -EnvVars $p.env
    $bytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes($doc)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $h = $sha.ComputeHash($bytes) } finally { $sha.Dispose() }
    return (([System.BitConverter]::ToString($h) -replace '-', '').ToLowerInvariant())
}

# ---------------------------------------------------------------------------------------------
# Binding inputs.
# ---------------------------------------------------------------------------------------------
# A-5 (UI-4 kin): hashing a multi-GB file is a long silent stretch, and this launcher's rule is
# that no such stretch exists without a line announcing it.
function Write-KvHashNotice {
    param([string] $What, [long] $Bytes, [string] $Tag = 'kv')
    Write-Line ('[{0}] {1} ({2} MB)...' -f $Tag, $What, [long]([Math]::Round($Bytes / 1MB)))
}

function Get-KvBundleSha {
    if ($script:WarmstartCtx.bundle_sha) { return $script:WarmstartCtx.bundle_sha }
    if (-not $script:WarmstartCtx.root) { return $null }
    $p = Join-Path $script:WarmstartCtx.root $script:BUNDLE_MANIFEST_NAME
    $h = Get-FileSha256Lower -Path $p
    if (-not $h.ok) { return $null }
    $script:WarmstartCtx.bundle_sha = $h.sha
    return $h.sha
}

# A-4: the base GGUF shard hashes. The computation, the identity key and the persistent cache all
# live in Get-ModelShardSha256Set (region 7) since LS OA-1 gave them a second reader - the M1 source
# pin. What stays here is the part that is specific to warmstart: the process-level cache and the
# FAILURE latch. Eligibility is re-evaluated on every effective config rebuild, and without that
# latch a custom edit would restart the multi-GB hashing pass that just failed, over and over.
function Get-KvModelShardShas {
    if ($null -ne $script:WarmstartCtx.model_shas) { return , $script:WarmstartCtx.model_shas }
    if ($script:WarmstartCtx.model_shas_attempted) { return $null }
    $set = $script:WarmstartCtx.model_set
    if ($null -eq $set) { return $null }
    $script:WarmstartCtx.model_shas_attempted = $true
    $r = Get-ModelShardSha256Set -ModelSet $set -NoticeTag 'kv'
    if (-not $r.ok) {
        $script:WarmstartCtx.model_shas_error = [string]$r.reason
        return $null
    }
    $script:WarmstartCtx.model_shas = @($r.shas)
    return , $script:WarmstartCtx.model_shas
}

# The first eight bytes of the state file. They are recorded at save time and re-checked at
# eligibility time, so the check is self-consistent whatever the engine's exact header layout is;
# WARMSTART_SPEC 1 documents those bytes as the GGSQ magic plus the sequence-state version.
function Get-KvFileHeaderFields {
    param([string] $Path)
    $fs = $null
    try {
        $fs = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
                  [System.IO.FileShare]::ReadWrite)
        $buf = New-Object byte[] 8
        $n = $fs.Read($buf, 0, 8)
        if ($n -ne 8) { return @{ ok = $false; reason = 'state file is shorter than its 8 byte header' } }
        $magic = ''
        for ($i = 0; $i -lt 4; $i++) { $magic = $magic + $buf[$i].ToString('x2') }
        $ver = [long]([uint32]$buf[4] -bor ([uint32]$buf[5] -shl 8) -bor ([uint32]$buf[6] -shl 16) -bor ([uint32]$buf[7] -shl 24))
        return @{ ok = $true; magic = $magic; version = $ver }
    } catch {
        return @{ ok = $false; reason = ('state file header read failed: ' + $_.Exception.Message) }
    } finally {
        if ($fs) { $fs.Dispose() }
    }
}

function Get-KvArgvLong {
    param([string[]] $Argv, [string] $Flag)
    # 0 means "the catalog default applies, no explicit value in argv". A real ctx or -np is never
    # 0, so the sentinel cannot collide with a genuine value.
    $v = Get-ArgvValue -Argv $Argv -Flag $Flag
    if ($null -eq $v) { return [long]0 }
    $n = [long]0
    if (-not [long]::TryParse(([string]$v).Trim(), [ref]$n)) { return [long]0 }
    if ($n -lt 0) { return [long]0 }
    return $n
}

# ---------------------------------------------------------------------------------------------
# A-4 sidecar.
# ---------------------------------------------------------------------------------------------
# LS 13-8: one naming rule for every stored generation - the sidecar is the data name plus
# '.meta.json'. The canonical pair obeys it (slot0.kv -> slot0.kv.meta.json), which is why the
# autosave generations need no second convention.
function Get-KvMetaName {
    param([string] $Name)
    return ([string]$Name + '.meta.json')
}
function Get-KvDataPath {
    param([string] $Name = $script:KV_CANONICAL_DATA)
    return (Join-Path $script:WarmstartCtx.dir $Name)
}
function Get-KvMetaPath {
    param([string] $Name = $script:KV_CANONICAL_DATA)
    return (Join-Path $script:WarmstartCtx.dir (Get-KvMetaName -Name $Name))
}
function Get-KvCanonicalDataPath { return (Join-Path $script:WarmstartCtx.dir $script:KV_CANONICAL_DATA) }
function Get-KvCanonicalMetaPath { return (Join-Path $script:WarmstartCtx.dir $script:KV_CANONICAL_META) }

# C: the restore candidates, in a fixed order. The canonical stop save comes first only so that its
# verdict is the one the status line falls back to; the actual choice between eligible candidates is
# made on saved_at alone (origin-agnostic).
function Get-KvCandidateNames {
    $out = @([string]$script:KV_CANONICAL_DATA)
    foreach ($n in $script:KV_AUTOSAVE_GENERATIONS) { $out += [string]$n }
    return , $out
}

# The verification order for C: newest stored saved_at first. Reading a small sidecar is cheap and
# an unusable or absent one sorts last, so the expensive half (the whole-file hash) is only paid for
# the generations that are actually in the running - and, in the overwhelming case of a single
# stored generation, exactly once. Ties keep the fixed candidate order (canonical, a, b).
function Get-KvCandidateOrder {
    $items = @()
    $idx = 0
    foreach ($n in (Get-KvCandidateNames)) {
        $ticks = [long][Int64]::MinValue
        $m = Read-KvSidecar -Path (Get-KvMetaPath -Name $n)
        if ($m.ok) { $ticks = Get-KvSavedAtTicks -Meta $m.value }
        $items += @{ name = [string]$n; ticks = [long]$ticks; index = [int]$idx }
        $idx = $idx + 1
    }
    $out = @()
    $taken = @{}
    while ($out.Count -lt $items.Count) {
        $pick = $null
        foreach ($it in $items) {
            if ($taken.ContainsKey([string]$it.name)) { continue }
            if ($null -eq $pick) { $pick = $it; continue }
            if ([long]$it.ticks -gt [long]$pick.ticks) { $pick = $it; continue }
            if ([long]$it.ticks -eq [long]$pick.ticks -and [int]$it.index -lt [int]$pick.index) { $pick = $it }
        }
        $taken[[string]$pick.name] = $true
        $out += [string]$pick.name
    }
    return , $out
}

# Strict sidecar read: any structural problem is "the file cannot be used with ANY configuration",
# which LS 13-6 maps to the single reason meta_parse_failed.
function Read-KvSidecar {
    param([string] $Path)
    $r = Read-JsonFileStrict -Path $Path
    if (-not $r.ok) { return @{ ok = $false; reason = $r.reason } }
    $o = $r.value
    foreach ($k in @('meta_schema_version', 'n_ctx', 'n_parallel', 'llama_state_seq_version', 'n_tokens', 'n_bytes')) {
        if (-not (Test-JsonNonNegativeInteger (Get-JsonValue -Obj $o -Name $k))) {
            return @{ ok = $false; reason = ($k + ' missing or not a non-negative integer') }
        }
    }
    foreach ($k in @('profile_id', 'llama_state_seq_magic', 'generation_id', 'saved_at')) {
        if (-not (Test-JsonNonEmptyString (Get-JsonValue -Obj $o -Name $k))) {
            return @{ ok = $false; reason = ($k + ' missing or not a non-empty string') }
        }
    }
    foreach ($k in @('repack_manifest_sha256', 'engine_bundle_sha256', 'effective_state_semantics_sha256', 'kv_file_sha256')) {
        if (-not (Test-Sha256Hex (Get-JsonValue -Obj $o -Name $k))) {
            return @{ ok = $false; reason = ($k + ' missing or not a 64 hex digest') }
        }
    }
    $shards = Get-JsonValue -Obj $o -Name 'model_shards_sha256'
    if (-not (Test-JsonArray $shards) -or @($shards).Count -eq 0) {
        return @{ ok = $false; reason = 'model_shards_sha256 missing or not a non-empty array' }
    }
    foreach ($s in @($shards)) {
        if (-not (Test-Sha256Hex $s)) { return @{ ok = $false; reason = 'model_shards_sha256 entry is not a 64 hex digest' } }
    }
    return @{ ok = $true; value = $o }
}

function Get-KvMetaShards {
    param($Meta)
    $out = @()
    foreach ($s in (Get-JsonArray -Obj $Meta -Name 'model_shards_sha256')) { $out += ([string]$s).ToLowerInvariant() }
    return , $out
}

# A-4b: class (a) - the file cannot be restored under ANY configuration, so both halves go and the
# next start converges on a clean cold. Class (b) - the file is internally sound and only disagrees
# with the current configuration, so it is KEPT: putting the configuration back must bring it back.
function Remove-KvCanonicalPair {
    param([string] $Why, [string] $Name = $script:KV_CANONICAL_DATA)
    $removed = @()
    $failed = @()
    foreach ($p in @((Get-KvDataPath -Name $Name), (Get-KvMetaPath -Name $Name))) {
        try {
            if (Test-Path -LiteralPath $p -PathType Leaf) { Remove-Item -LiteralPath $p -Force -ErrorAction Stop; $removed += $p }
        } catch { $failed += ($p + ': ' + $_.Exception.Message) }
    }
    Write-Diag -Kind 'WARMSTART_CANONICAL_DISCARDED' -Data @{ why = $Why; generation = $Name
                                                              removed = $removed; failed = $failed }
}

# A-4 file-dependent half: computed once per process AND PER GENERATION. It also performs the class
# (a) recovery, which is why it must not run more than once for the same name.
function Measure-KvFileFacts {
    param([string] $Name = $script:KV_CANONICAL_DATA)
    $dataPath = Get-KvDataPath -Name $Name
    $metaPath = Get-KvMetaPath -Name $Name
    if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
        return @{ ok = $false; reason = $script:KV_REASON_SIDECAR_MISSING }
    }
    if (-not (Test-Path -LiteralPath $dataPath -PathType Leaf)) {
        return @{ ok = $false; reason = $script:KV_REASON_KV_FILE_MISSING }
    }
    $meta = Read-KvSidecar -Path $metaPath
    if (-not $meta.ok) {
        Remove-KvCanonicalPair -Why ('meta_parse_failed: ' + $meta.reason) -Name $Name
        return @{ ok = $false; reason = $script:KV_REASON_META_PARSE_FAILED; detail = $meta.reason }
    }
    $len = [long]0
    try { $len = [long](New-Object System.IO.FileInfo($dataPath)).Length }
    catch {
        return @{ ok = $false; reason = $script:KV_REASON_UNAVAILABLE; detail = ('length query failed: ' + $_.Exception.Message) }
    }
    if ($len -ne [long](Get-JsonValue -Obj $meta.value -Name 'n_bytes')) {
        Remove-KvCanonicalPair -Why 'file_integrity_broken: n_bytes != real length' -Name $Name
        return @{ ok = $false; reason = $script:KV_REASON_FILE_INTEGRITY
                  detail = ('n_bytes=' + [string](Get-JsonValue -Obj $meta.value -Name 'n_bytes') + ' real=' + $len) }
    }
    Write-KvHashNotice -What 'verifying stored slot state' -Bytes $len
    $h = Get-FileSha256Lower -Path $dataPath
    if (-not $h.ok) {
        return @{ ok = $false; reason = $script:KV_REASON_UNAVAILABLE; detail = $h.reason }
    }
    if ($h.sha -cne ([string](Get-JsonValue -Obj $meta.value -Name 'kv_file_sha256')).ToLowerInvariant()) {
        Remove-KvCanonicalPair -Why 'file_integrity_broken: kv_file_sha256 != real file hash' -Name $Name
        return @{ ok = $false; reason = $script:KV_REASON_FILE_INTEGRITY; detail = 'kv_file_sha256 mismatch' }
    }
    $hdr = Get-KvFileHeaderFields -Path $dataPath
    if (-not $hdr.ok) {
        Remove-KvCanonicalPair -Why ('file_integrity_broken: ' + $hdr.reason) -Name $Name
        return @{ ok = $false; reason = $script:KV_REASON_FILE_INTEGRITY; detail = $hdr.reason }
    }
    # Unreachable once the whole-file hash agreed (the header bytes are inside that hash); kept as
    # the same-class double fail-close A-4 asks for, and it emits no new reason value.
    if (($hdr.magic -cne ([string](Get-JsonValue -Obj $meta.value -Name 'llama_state_seq_magic')).ToLowerInvariant()) -or
        ([long]$hdr.version -ne [long](Get-JsonValue -Obj $meta.value -Name 'llama_state_seq_version'))) {
        Remove-KvCanonicalPair -Why 'file_integrity_broken: state header disagrees with the sidecar' -Name $Name
        return @{ ok = $false; reason = $script:KV_REASON_FILE_INTEGRITY; detail = 'state header mismatch' }
    }
    return @{ ok = $true; meta = $meta.value; length = $len; sha = $h.sha }
}

# The canonical generation keeps its own field rather than an entry in the map: the harness clears
# `file_facts` to force a re-evaluation, and that knob has to keep meaning exactly what it meant.
function Get-KvFileFacts {
    param([string] $Name = $script:KV_CANONICAL_DATA)
    if ($Name -ceq $script:KV_CANONICAL_DATA) {
        if ($null -ne $script:WarmstartCtx.file_facts) { return $script:WarmstartCtx.file_facts }
        $r = Measure-KvFileFacts -Name $Name
        $script:WarmstartCtx.file_facts = $r
        return $r
    }
    if ($null -eq $script:WarmstartCtx.auto_facts) { $script:WarmstartCtx.auto_facts = @{} }
    if ($script:WarmstartCtx.auto_facts.ContainsKey($Name)) { return $script:WarmstartCtx.auto_facts[$Name] }
    $r = Measure-KvFileFacts -Name $Name
    $script:WarmstartCtx.auto_facts[$Name] = $r
    return $r
}

# C: saved_at is the ONLY ordering key between eligible candidates, and it is informational in the
# sidecar (never part of the match predicate), so an unusable value must not disqualify a file - it
# just sorts last. Int64.MinValue is that "oldest possible" marker, as in the profile GC.
function Get-KvSavedAtTicks {
    param($Meta)
    $inst = ConvertTo-UtcInstant -Value (Get-JsonValue -Obj $Meta -Name 'saved_at')
    if (-not $inst.ok) { return [long][Int64]::MinValue }
    return [long]$inst.utc.Ticks
}

function Get-KvOrigin {
    param($Meta)
    $o = Get-JsonValue -Obj $Meta -Name 'origin'
    if (Test-JsonNonEmptyString $o) { return [string]$o }
    return [string]$script:KV_ORIGIN_STOP
}

# A-4 configuration-dependent half: re-decided on EVERY Build-EffectiveConfig return, because a
# custom edit can flip the verdict. Field disagreements render as "<exact sidecar key>_mismatch".
function Compare-KvConfigBinding {
    param($Meta, [string[]] $Argv, [hashtable] $EnvVars)
    if ([long](Get-JsonValue -Obj $Meta -Name 'meta_schema_version') -ne [long]$script:KV_META_SCHEMA_VERSION) {
        return @{ ok = $false; reason = 'meta_schema_version_mismatch' }
    }
    if ([string](Get-JsonValue -Obj $Meta -Name 'profile_id') -cne [string]$script:WarmstartCtx.profile_id) {
        return @{ ok = $false; reason = 'profile_id_mismatch' }
    }
    if (([string](Get-JsonValue -Obj $Meta -Name 'repack_manifest_sha256')).ToLowerInvariant() -cne [string]$script:WarmstartCtx.manifest_sha) {
        return @{ ok = $false; reason = 'repack_manifest_sha256_mismatch' }
    }
    $bundle = Get-KvBundleSha
    if ($null -eq $bundle) { return @{ ok = $false; reason = $script:KV_REASON_UNAVAILABLE; detail = 'bundle manifest hash unavailable' } }
    if (([string](Get-JsonValue -Obj $Meta -Name 'engine_bundle_sha256')).ToLowerInvariant() -cne $bundle) {
        return @{ ok = $false; reason = 'engine_bundle_sha256_mismatch' }
    }
    if ([long](Get-JsonValue -Obj $Meta -Name 'n_ctx') -ne (Get-KvArgvLong -Argv $Argv -Flag '-c')) {
        return @{ ok = $false; reason = 'n_ctx_mismatch' }
    }
    if ([long](Get-JsonValue -Obj $Meta -Name 'n_parallel') -ne (Get-KvArgvLong -Argv $Argv -Flag '-np')) {
        return @{ ok = $false; reason = 'n_parallel_mismatch' }
    }
    $sem = Get-KvSemanticsSha256 -Argv $Argv -EnvVars $EnvVars
    if (([string](Get-JsonValue -Obj $Meta -Name 'effective_state_semantics_sha256')).ToLowerInvariant() -cne $sem) {
        return @{ ok = $false; reason = 'effective_state_semantics_sha256_mismatch'; detail = ('now=' + $sem) }
    }
    # Last, because it is the only expensive input left and every cheaper disagreement has already
    # short-circuited it.
    $shas = Get-KvModelShardShas
    if ($null -eq $shas) { return @{ ok = $false; reason = $script:KV_REASON_UNAVAILABLE; detail = 'model shard hashes unavailable' } }
    $stored = Get-KvMetaShards -Meta $Meta
    if (@($stored).Count -ne @($shas).Count) { return @{ ok = $false; reason = 'model_shards_sha256_mismatch' } }
    for ($i = 0; $i -lt @($shas).Count; $i++) {
        if ([string]@($stored)[$i] -cne [string]@($shas)[$i]) { return @{ ok = $false; reason = 'model_shards_sha256_mismatch' } }
    }
    return @{ ok = $true; semantics = $sem }
}

# A-2 (6): an OPERATIONAL failure (hashing, a length query, a binding input that cannot be read) is
# eligibility_unavailable, and the frozen shape for it is one warning line plus the diagnostic that
# Update-WarmstartEligibility already writes. Latched for the run: the verdict is recomputed on
# every effective config rebuild and the same failure must not repaint the console each time.
function Write-KvUnavailableWarning {
    param([string] $Detail)
    if ($script:WarmstartCtx.unavailable_warned) { return }
    $script:WarmstartCtx.unavailable_warned = $true
    $why = 'reason unavailable'
    if ($Detail) { $why = $Detail }
    Write-Line ('[kv] WARNING: the stored slot state could not be evaluated (' + $why + '); this run starts cold.')
}

# Called from the tail of Build-EffectiveConfig, so "re-evaluated on every returned effective
# config" is structural rather than a rule someone has to remember at each call site.
function Update-WarmstartEligibility {
    param([string[]] $Argv, [hashtable] $EnvVars)
    $state = Get-WarmstartState
    if ($state -ceq 'hard_off')    { Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_OFF_MODE -Detail $null; return }
    if ($state -ceq 'soft_off')    { Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_OFF_USER -Detail $null; return }
    if ($state -ceq 'latched_off') { Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_UNAVAILABLE -Detail $script:WarmstartCtx.latch_reason; return }
    # After ready the verdict belongs to the actual restore result, which must not be overwritten
    # by a later configuration rebuild.
    if ($script:WarmstartCtx.restore_done) { return }
    if (-not $script:WarmstartCtx.root) {
        Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_UNAVAILABLE -Detail 'bindings not established yet'
        return
    }
    # C: every stored generation is judged by the SAME machine (file facts, then the configuration
    # binding); the winner is simply the eligible one with the newest saved_at, whatever wrote it.
    # The candidates are VERIFIED in newest-first order and the first one to survive both halves is
    # the answer - but the loop does NOT stop there. Get-KvFileFacts is where the A-4b class (a)
    # recovery happens, so skipping the lower-ranked candidates would silently retire that contract
    # for every generation below the winner: a damaged canonical sitting under a newer sound
    # autosave would never be discarded and would keep failing the same way at every start. The
    # election is unchanged (newest eligible wins); only the damage check is exhaustive.
    # The canonical verdict is kept separately because it is the one the status line falls back to
    # when nothing is eligible - that keeps the reason enum and every existing cold path unchanged.
    $summary = @()
    $canon = $null
    $best = $null
    foreach ($name in (Get-KvCandidateOrder)) {
        $isCanon = ($name -ceq $script:KV_CANONICAL_DATA)
        $facts = Get-KvFileFacts -Name $name
        if (-not $facts.ok) {
            $v = @{ generation = $name; eligible = $false; reason = [string]$facts.reason; detail = $facts.detail; stage = 'file' }
            if ($isCanon) { $canon = $v }
            # Absence is the normal state of a generation that was never written: it is not worth a
            # record of its own, and it must not drown the real verdict in the diagnostic log.
            if (-not $isCanon -and @($script:KV_REASON_SIDECAR_MISSING, $script:KV_REASON_KV_FILE_MISSING) -cnotcontains [string]$facts.reason) {
                $summary += $v
            }
            if ([string]$facts.reason -ceq $script:KV_REASON_UNAVAILABLE) { Write-KvUnavailableWarning -Detail ([string]$facts.detail) }
            continue
        }
        $cmp = Compare-KvConfigBinding -Meta $facts.meta -Argv $Argv -EnvVars $EnvVars
        if (-not $cmp.ok) {
            $v = @{ generation = $name; eligible = $false; reason = [string]$cmp.reason; detail = $cmp.detail; stage = 'config' }
            if ($isCanon) { $canon = $v }
            if (-not $isCanon) { $summary += $v }
            if ([string]$cmp.reason -ceq $script:KV_REASON_UNAVAILABLE) { Write-KvUnavailableWarning -Detail ([string]$cmp.detail) }
            continue
        }
        $cand = @{ generation = $name; eligible = $true; facts = $facts; cmp = $cmp
                   origin = (Get-KvOrigin -Meta $facts.meta)
                   saved_at = [string](Get-JsonValue -Obj $facts.meta -Name 'saved_at') }
        if ($isCanon) { $canon = @{ generation = $name; eligible = $true; reason = 'eligible' } }
        if (-not $isCanon) { $summary += @{ generation = $name; eligible = $true; saved_at = $cand.saved_at } }
        # Newest first, so the first candidate that survives both halves IS the answer; the rest are
        # still evaluated (that is the class (a) recovery above) but can no longer be elected.
        if ($null -eq $best) { $best = $cand }
    }

    if ($null -eq $best) {
        # No generation is usable. The verdict, the reason and the recovery all stay exactly what
        # they were before autosave existed: the canonical's own answer.
        if ($null -eq $canon) { $canon = @{ reason = $script:KV_REASON_SIDECAR_MISSING; detail = $null; stage = 'file' } }
        Set-KvVerdict -Eligible $false -Reason $canon.reason -Detail $canon.detail
        Write-Diag -Kind 'WARMSTART_ELIGIBILITY' -Data @{ eligible = $false; reason = $canon.reason
                                                          detail = $canon.detail; stage = $canon.stage
                                                          candidates = $summary }
        return
    }
    Set-KvVerdict -Eligible $true -Reason 'eligible' -Detail $null
    $script:WarmstartCtx.meta = $best.facts.meta
    $script:WarmstartCtx.selected_name = [string]$best.generation
    $script:WarmstartCtx.selected_origin = [string]$best.origin
    $script:WarmstartCtx.selected_saved_at = [string]$best.saved_at
    Write-Diag -Kind 'WARMSTART_ELIGIBILITY' -Data @{ eligible = $true; reason = 'eligible'
                                                      n_bytes = $best.facts.length; kv_file_sha256 = $best.facts.sha
                                                      effective_state_semantics_sha256 = $best.cmp.semantics
                                                      generation_id = [string](Get-JsonValue -Obj $best.facts.meta -Name 'generation_id')
                                                      selected = [string]$best.generation
                                                      origin = [string]$best.origin
                                                      saved_at = [string]$best.saved_at
                                                      candidates = $summary }
}

# ---------------------------------------------------------------------------------------------
# A-2 / A-3 wire. Routes and field names are the measured contract recorded in WARMSTART_SPEC 1.
# ---------------------------------------------------------------------------------------------
function Invoke-KvSlotAction {
    param($Config, [string] $Action, [string] $FileName, [int] $TimeoutSec)
    $uri = ('http://{0}:{1}/slots/{2}?action={3}' -f $Config.host, $Config.port, $script:KV_SLOT_ID, $Action)
    $body = $null
    if ($FileName) { $body = '{"filename":' + (ConvertTo-KvJsonString -Value $FileName) + '}' }
    return (Invoke-HttpJson -Uri $uri -Method 'POST' -Body $body -TimeoutSec $TimeoutSec)
}

# A-2 (4): HTTP 200 alone does NOT prove a save happened - llama_state_seq_save_file() returning 0
# is still answered with a 200. Every one of these has to hold.
function Test-KvSlotResponse {
    param($Response, [string] $FileName, [string[]] $PositiveFields)
    if ($null -eq $Response -or -not $Response.ok) {
        $why = 'no response'
        if ($null -ne $Response -and $Response.reason) { $why = [string]$Response.reason }
        # A thrown non-2xx IS a delivered answer (PS 5.1 turns it into a terminating error rather
        # than a response object); only a transport failure is truly undelivered.
        $got = $false
        if ($null -ne $Response) { $got = [bool]$Response.response_received }
        return @{ ok = $false; delivered = $got; reason = $why }
    }
    if ([int]$Response.status -ne 200) {
        return @{ ok = $false; delivered = $true; reason = ('http status ' + $Response.status) }
    }
    $p = ConvertFrom-JsonStrict -Text $Response.body
    if (-not $p.ok) { return @{ ok = $false; delivered = $true; reason = ('malformed response body - ' + $p.reason) } }
    $j = $p.value
    $slot = Get-JsonValue -Obj $j -Name 'id_slot'
    if (-not (Test-JsonNonNegativeInteger $slot) -or [long]$slot -ne [long]$script:KV_SLOT_ID) {
        return @{ ok = $false; delivered = $true; reason = 'id_slot missing or not slot 0' }
    }
    $echo = Get-JsonValue -Obj $j -Name 'filename'
    if (-not (Test-JsonNonEmptyString $echo) -or ([string]$echo -cne $FileName)) {
        return @{ ok = $false; delivered = $true; reason = 'filename echo does not match the requested name' }
    }
    $vals = @{}
    foreach ($f in $PositiveFields) {
        $v = Get-JsonValue -Obj $j -Name $f
        if (-not (Test-JsonNonNegativeInteger $v)) { return @{ ok = $false; delivered = $true; reason = ($f + ' missing or not an integer') } }
        if ([long]$v -le 0) { return @{ ok = $false; delivered = $true; reason = ($f + ' is 0') } }
        $vals[$f] = [long]$v
    }
    return @{ ok = $true; delivered = $true; values = $vals }
}

# A-3 / A-4b. Runs after ready, before the launcher warmup. The restore attempt is latched to once
# per process so the ladder can never re-enter itself.
function Invoke-WarmstartRestore {
    param($Config)
    $out = @{ restored = $false; recovery = $false; n_restored = 0 }
    if (-not (Test-WarmstartActive)) { return $out }
    if ($script:WarmstartCtx.restore_latched) { return $out }
    if (-not $script:WarmstartCtx.eligible) { return $out }
    $script:WarmstartCtx.restore_latched = $true

    # C: the generation eligibility selected - the newest stored one, whatever wrote it. With no
    # autosave present that is the canonical name, byte for byte as before.
    $name = [string]$script:WarmstartCtx.selected_name
    if (-not $name) { $name = [string]$script:KV_CANONICAL_DATA }
    $r = Invoke-KvSlotAction -Config $Config -Action 'restore' -FileName $name `
             -TimeoutSec $script:KV_RESTORE_TIMEOUT_S
    $v = Test-KvSlotResponse -Response $r -FileName $name -PositiveFields @('n_restored', 'n_read')
    if ($v.ok) {
        $out.restored = $true
        $out.n_restored = [long]$v.values['n_restored']
        $script:WarmstartCtx.restore_done = $true
        # A (change gate): the restored length is what "no change since the last save" means for the
        # first autosave tick of this run.
        $script:WarmstartCtx.autosave_tokens = [long]$out.n_restored
        # B (survivor pin): a restored autosave generation is the only complete recovery point this
        # run is known to have, so the next autosave aims at the OTHER one whatever the sidecar
        # clocks say. Without the pin the selector's "oldest saved_at first" rule can hand the next
        # write the generation that was just restored - reachable whenever the newer generation lost
        # eligibility on the configuration, because a class (b) disagreement KEEPS the file and it
        # stays the newest by saved_at - and a crash during that write would leave nothing
        # restorable at all. The pin lifts itself on the first successful write (which flips the
        # target back); a failed write keeps aiming at the generation it already lost, so the
        # restored one stays protected until a complete replacement exists.
        if ($script:KV_AUTOSAVE_GENERATIONS -ccontains $name) {
            $script:WarmstartCtx.autosave_next = (Get-KvOtherGeneration -Name $name)
        }
        Set-KvVerdict -Eligible $true -Reason 'restored' -Detail $null
        $script:WarmstartCtx.status_text = ('restored(' + $out.n_restored + ' tokens)')
        Write-Line ('[kv] restored(' + $out.n_restored + ' tokens) from the stored slot state.')
        Write-Diag -Kind 'WARMSTART_RESTORE_OK' -Data @{ n_restored = $out.n_restored
                                                         n_read = [long]$v.values['n_read']
                                                         generation = $name
                                                         origin = [string]$script:WarmstartCtx.selected_origin
                                                         saved_at = [string]$script:WarmstartCtx.selected_saved_at }
        return $out
    }

    # Ladder rung 2: anything other than a clean restore erases the slot before going cold.
    Write-Diag -Kind $script:KV_DIAG_RESTORE_FAILED -Data @{ reason = $v.reason; delivered = $v.delivered }
    Write-Line ('[kv] WARNING: slot restore failed (' + $v.reason + '); erasing the slot and starting cold.')
    $e = Invoke-KvSlotAction -Config $Config -Action 'erase' -FileName $null -TimeoutSec $script:KV_ERASE_TIMEOUT_S
    $eraseOk = ($null -ne $e -and $e.ok -and [int]$e.status -eq 200)
    if ($eraseOk) {
        # Rung 3: cold only AFTER the erase is confirmed.
        $script:WarmstartCtx.restore_done = $true
        Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_RESTORE_FAILED -Detail $v.reason
        Write-Line '[kv] cold(restore_failed) - slot erased, this run starts without a warm cache.'
        Write-Diag -Kind 'WARMSTART_RESTORE_COLD' -Data @{ reason = $script:KV_REASON_RESTORE_FAILED; erase = 'ok' }
        return $out
    }
    # Rung 4: the erase failed too, so the slot may hold a partially restored state. The only
    # remaining way to a known-clean slot is a fresh server.
    $why = 'erase request failed'
    if ($null -ne $e -and $e.reason) { $why = [string]$e.reason }
    elseif ($null -ne $e -and $e.ok) { $why = ('erase http status ' + $e.status) }
    Write-Diag -Kind $script:KV_DIAG_RESTORE_FAILED -Data @{ reason = $why; stage = 'erase'; next = 'recovery restart' }
    $out.recovery = $true
    return $out
}

# ---------------------------------------------------------------------------------------------
# A-2 save. Inserted immediately before the graceful stop signal, because the save response has to
# arrive before CTRL_BREAK. Fully self-contained: nothing here throws, so a save problem can never
# be promoted into fail_teardown by the caller's own catch.
# ---------------------------------------------------------------------------------------------
function Test-WarmstartSaveAllowed {
    param($Child, [string] $PendingStatus)
    if (-not (Test-WarmstartActive)) { return $false }
    if (-not $script:ChildWasReady) { return $false }
    if ($null -eq $Child) { return $false }
    # Only a requested, normal stop saves. Every fail_* path leaves the stored state alone.
    if (@('ok', 'ok_smoke') -notcontains [string]$PendingStatus) { return $false }
    try { if ((Test-ChildExited -Child $Child).exited) { return $false } } catch { return $false }
    return $true
}

function Get-KvGenerationId {
    return ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfff') + 'Z_' + $PID)
}

function Write-KvSaveDegraded {
    param([string] $Reason, $Data)
    $d = @{ reason = $Reason }
    if ($null -ne $Data) { foreach ($k in $Data.Keys) { $d[$k] = $Data[$k] } }
    Write-Diag -Kind $script:KV_DIAG_SAVE_FAILED -Data $d
    Write-Line ('[kv] WARNING: slot save skipped or failed (' + $Reason + '); shutdown continues normally.')
}

function Invoke-WarmstartSave {
    param($Child, [string] $PendingStatus, $Config)
    # Declared ahead of the try so the outer catch can recover them: a fault anywhere after the
    # server has created the tmp file must not leave that generation on the volume (A-2 (8)).
    $tmpPath = $null
    $metaTmp = $null
    $responseReceived = $false
    try {
        if (-not (Test-WarmstartSaveAllowed -Child $Child -PendingStatus $PendingStatus)) { return }
        if ($null -eq $Config) { return }
        $script:WarmstartCtx.save_attempted = $true
        $dir = $script:WarmstartCtx.dir
        $dataPath = Get-KvCanonicalDataPath
        $metaPath = Get-KvCanonicalMetaPath
        $gen = Get-KvGenerationId
        $tmpName  = $script:KV_CANONICAL_DATA + '.tmp.' + $gen
        $tmpPath  = Join-Path $dir $tmpName
        $metaTmp  = Join-Path $dir ($script:KV_CANONICAL_META + '.tmp.' + $gen)
        $dataStale = Join-Path $dir ($script:KV_CANONICAL_DATA + '.stale.' + $gen)
        $metaStale = Join-Path $dir ($script:KV_CANONICAL_META + '.stale.' + $gen)

        # A-2 (7): with a previous generation on disk its size is a real estimate of this one's, so
        # a save that would obviously not fit is skipped instead of filling the volume. A failure of
        # the query itself is treated the same way - no freedom is left here.
        $prev = Read-KvSidecar -Path $metaPath
        if ($prev.ok) {
            $need = [long](Get-JsonValue -Obj $prev.value -Name 'n_bytes')
            $free = Get-VolumeFreeMb -Path $dir
            if (-not $free.ok) {
                Write-KvSaveDegraded -Reason ('kv volume free space query failed: ' + $free.reason) -Data $null
                return
            }
            $needMb = [long]([Math]::Ceiling(([double]$need * 1.1) / 1MB))
            if ([long]$free.free_mb -lt $needMb) {
                Write-KvSaveDegraded -Reason ('kv volume free space ' + $free.free_mb + ' MB is below the ' + $needMb + ' MB this save needs') -Data $null
                return
            }
        }

        Write-Line '[kv] saving the slot state before shutdown...'
        $r = Invoke-KvSlotAction -Config $Config -Action 'save' -FileName $tmpName -TimeoutSec $script:KV_SAVE_TIMEOUT_S
        $v = Test-KvSlotResponse -Response $r -FileName $tmpName -PositiveFields @('n_saved', 'n_written')
        $responseReceived = [bool]$v.delivered
        if (-not $v.ok) {
            if ($v.delivered) {
                # Recovery timing (1): a verdict was reached with the server still holding nothing
                # open, so the generation's tmp goes now.
                Remove-KvPathBestEffort -Path $tmpPath -Why 'save failed (response received)'
            } else {
                # Recovery timing (2): no response came back, so the server may still be writing.
                # Deletion waits until the child has been joined.
                $script:WarmstartCtx.tmp_after_join += $tmpPath
            }
            Write-KvSaveDegraded -Reason $v.reason -Data @{ filename = $tmpName; delivered = $v.delivered }
            return
        }

        $len = [long]0
        try { $len = [long](New-Object System.IO.FileInfo($tmpPath)).Length }
        catch {
            Remove-KvPathBestEffort -Path $tmpPath -Why 'save length query failed'
            Write-KvSaveDegraded -Reason ('saved file length query failed: ' + $_.Exception.Message) -Data $null
            return
        }
        if ($len -le 0) {
            Remove-KvPathBestEffort -Path $tmpPath -Why 'saved file is empty'
            Write-KvSaveDegraded -Reason 'the saved file is empty on disk' -Data $null
            return
        }
        Write-KvHashNotice -What 'checksumming the saved slot state' -Bytes $len
        $h = Get-FileSha256Lower -Path $tmpPath
        if (-not $h.ok) {
            Remove-KvPathBestEffort -Path $tmpPath -Why 'save hashing failed'
            Write-KvSaveDegraded -Reason $h.reason -Data $null
            return
        }
        $hdr = Get-KvFileHeaderFields -Path $tmpPath
        if (-not $hdr.ok) {
            Remove-KvPathBestEffort -Path $tmpPath -Why 'save header read failed'
            Write-KvSaveDegraded -Reason $hdr.reason -Data $null
            return
        }
        $shards = Get-KvModelShardShas
        $bundle = Get-KvBundleSha
        if ($null -eq $shards -or $null -eq $bundle) {
            Remove-KvPathBestEffort -Path $tmpPath -Why 'binding inputs unavailable'
            Write-KvSaveDegraded -Reason 'model shard or bundle binding inputs are unavailable' -Data $null
            return
        }

        $meta = [ordered]@{
            meta_schema_version              = [int]$script:KV_META_SCHEMA_VERSION
            profile_id                       = [string]$script:WarmstartCtx.profile_id
            model_shards_sha256              = @($shards)
            repack_manifest_sha256           = [string]$script:WarmstartCtx.manifest_sha
            engine_bundle_sha256             = [string]$bundle
            effective_state_semantics_sha256 = (Get-KvSemanticsSha256 -Argv $Config.argv -EnvVars $Config.env)
            n_ctx                            = [long](Get-KvArgvLong -Argv $Config.argv -Flag '-c')
            n_parallel                       = [long](Get-KvArgvLong -Argv $Config.argv -Flag '-np')
            llama_state_seq_magic            = [string]$hdr.magic
            llama_state_seq_version          = [long]$hdr.version
            n_tokens                         = [long]$v.values['n_saved']
            n_bytes                          = [long]$len
            kv_file_sha256                   = [string]$h.sha
            generation_id                    = [string]$gen
            saved_at                         = (Get-Date).ToUniversalTime().ToString('o')
        }

        # A-4b write transaction, order fixed: data first, then the meta that carries the data's
        # hash. A power loss at any gap therefore leaves either a cold start or a consistent older
        # generation - never a meta that describes bytes which are not there.
        try {
            if (Test-Path -LiteralPath $dataPath -PathType Leaf) {
                [System.IO.File]::Replace($tmpPath, $dataPath, $dataStale)
                $script:WarmstartCtx.stale_after_stop += $dataStale
            } else {
                Move-FileAtomic -TempPath $tmpPath -FinalPath $dataPath
            }
        } catch {
            Remove-KvPathBestEffort -Path $tmpPath -Why 'data publish failed'
            Write-KvSaveDegraded -Reason ('publishing the state file failed: ' + $_.Exception.Message) -Data $null
            return
        }
        try {
            [System.IO.File]::WriteAllText($metaTmp, ($meta | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
            $rb = Read-KvSidecar -Path $metaTmp
            if (-not $rb.ok) { throw ('read-back parse failed: ' + $rb.reason) }
            if (Test-Path -LiteralPath $metaPath -PathType Leaf) {
                [System.IO.File]::Replace($metaTmp, $metaPath, $metaStale)
                $script:WarmstartCtx.stale_after_stop += $metaStale
            } else {
                Move-FileAtomic -TempPath $metaTmp -FinalPath $metaPath
            }
        } catch {
            Remove-KvPathBestEffort -Path $metaTmp -Why 'meta publish failed'
            Write-KvSaveDegraded -Reason ('publishing the sidecar failed: ' + $_.Exception.Message) -Data $null
            return
        }
        Write-Line ('[kv] saved ' + $v.values['n_saved'] + ' tokens (' + [long]([Math]::Round($len / 1MB)) + ' MB) for the next start.')
        Write-Diag -Kind 'WARMSTART_SAVE_OK' -Data @{ generation_id = $gen; n_tokens = [long]$v.values['n_saved']
                                                      n_bytes = $len; kv_file_sha256 = $h.sha }
    } catch {
        # Total containment (LS 13-1). A save fault is degraded, never a teardown status.
        try { Write-Diag -Kind $script:KV_DIAG_SAVE_FAILED -Data @{ reason = ('save stage fault: ' + $_.Exception.Message)
                                                                    response_received = $responseReceived } } catch { }
        try { Write-Line ('[kv] WARNING: slot save failed (' + $_.Exception.Message + '); shutdown continues normally.') } catch { }
        # A-2 (8) with the same two timings as the normal failure path: an answered request means
        # the server is done writing, so the generation goes now; anything else waits for the join.
        try {
            foreach ($p in @($tmpPath, $metaTmp)) {
                if (-not $p) { continue }
                if ($responseReceived) { [void](Remove-KvPathBestEffort -Path $p -Why 'save stage fault (response received)') }
                else { $script:WarmstartCtx.tmp_after_join += $p }
            }
        } catch { }
    }
}

function Remove-KvPathBestEffort {
    param([string] $Path, [string] $Why)
    try {
        if (Test-Path -LiteralPath $Path -PathType Leaf) { Remove-Item -LiteralPath $Path -Force -ErrorAction Stop }
        return $true
    } catch {
        Write-Diag -Kind $script:KV_DIAG_SAVE_FAILED -Data @{ reason = ('leftover cleanup failed: ' + $_.Exception.Message)
                                                              path = $Path; why = $Why }
        return $false
    }
}

# Recovery timings (2) and (3): the deferred tmp of a save whose response never came, once the
# child is gone, and this transaction's stale pair at the very end of teardown.
function Complete-WarmstartTeardownCleanup {
    try {
        foreach ($p in @($script:WarmstartCtx.tmp_after_join)) { [void](Remove-KvPathBestEffort -Path $p -Why 'save timed out') }
        $script:WarmstartCtx.tmp_after_join = @()
        foreach ($p in @($script:WarmstartCtx.stale_after_stop)) { [void](Remove-KvPathBestEffort -Path $p -Why 'superseded generation') }
        $script:WarmstartCtx.stale_after_stop = @()
    } catch { }
}

# ---------------------------------------------------------------------------------------------
# LS 13-8 / AUTOSAVE_SPEC v0.1 - the periodic crash-recovery save.
#
# Three properties carry the whole design:
#   A  a tick only fires when the server is IDLE and the sequence has CHANGED since the last save,
#      so the feature costs a serving run nothing it can feel;
#   B  the two generations are written alternately, so the one being written is the only one a
#      crash can damage and the previous one always survives;
#   D  it is subordinate to warmstart - hard-OFF, soft-OFF and the A-1 directory latch all switch
#      it off, and its own key can never switch any of them back on.
# Every failure here is non-terminal and degraded, and reuses the existing warmstart_save_failed
# diagnostic kind: LS 13-3 freezes the degraded list at three kinds and no new one is invented.
# ---------------------------------------------------------------------------------------------
function Set-AutosaveSetting {
    param([string] $Value)
    $v = ([string]$Value).Trim().ToLowerInvariant()
    $script:WarmstartCtx.autosave_setting = $v
    if ($v -ceq 'off') {
        $script:WarmstartCtx.autosave_enabled = $false
        $script:WarmstartCtx.autosave_minutes = [int]$script:KV_AUTOSAVE_DEFAULT_MIN
        return
    }
    $script:WarmstartCtx.autosave_enabled = $true
    $m = [long]0
    if ([long]::TryParse($v, [ref]$m) -and
        $m -ge [long]$script:KV_AUTOSAVE_MIN_MINUTES -and $m -le [long]$script:KV_AUTOSAVE_MAX_MINUTES) {
        $script:WarmstartCtx.autosave_minutes = [int]$m
    } else {
        # 'on' and anything the parser already normalised away both land on the default period.
        $script:WarmstartCtx.autosave_minutes = [int]$script:KV_AUTOSAVE_DEFAULT_MIN
    }
}

function Test-AutosaveActive {
    if (-not (Test-WarmstartActive)) { return $false }
    if (-not $script:WarmstartCtx.autosave_enabled) { return $false }
    if ($script:WarmstartCtx.autosave_stopped) { return $false }
    return $true
}

# A-2 (6) shape, adapted to something that repeats: the diagnostic is written every time, the
# console warning only once per run - a five minute cadence must not repaint the same line forever.
function Write-KvAutosaveDegraded {
    param([string] $Reason, $Data)
    $d = @{ reason = $Reason; stage = 'autosave' }
    if ($null -ne $Data) { foreach ($k in $Data.Keys) { $d[$k] = $Data[$k] } }
    Write-Diag -Kind $script:KV_DIAG_SAVE_FAILED -Data $d
    if ($script:WarmstartCtx.autosave_warned) { return }
    $script:WarmstartCtx.autosave_warned = $true
    Write-Line ('[kv] WARNING: autosave skipped (' + $Reason + '); serving continues normally.')
}

# A (idle gate): GET /slots is the server's own answer about its own slots - it is served from the
# task queue as a high-priority task, so "is_processing" is the state the save task would meet.
# `n_prompt_tokens` is the slot's cached sequence length (prompt + generated), which is exactly the
# quantity a save stores; it is only present once the slot has held a task, and its absence is
# therefore read as "no evidence of a change", never as zero.
function Get-KvSlotsSnapshot {
    param($Config, [int] $TimeoutSec = 0)
    if ($TimeoutSec -le 0) { $TimeoutSec = [int]$script:KV_SLOTS_TIMEOUT_S }
    $uri = ('http://{0}:{1}/slots' -f $Config.host, $Config.port)
    $r = Invoke-HttpJson -Uri $uri -Method 'GET' -TimeoutSec $TimeoutSec
    if ($null -eq $r -or -not $r.ok) {
        $why = 'no response'
        if ($null -ne $r -and $r.reason) { $why = [string]$r.reason }
        return @{ ok = $false; reason = $why }
    }
    if ([int]$r.status -ne 200) { return @{ ok = $false; reason = ('http status ' + $r.status) } }
    $p = ConvertFrom-JsonStrict -Text $r.body
    if (-not $p.ok) { return @{ ok = $false; reason = ('malformed /slots body - ' + $p.reason) } }
    # A one element array is unrolled into a single object by ConvertFrom-Json, and the launcher
    # locks -np 1, so that IS the normal shape here.
    $items = @($p.value)
    $busy = $false
    $slot = $null
    $seen = 0
    foreach ($s in $items) {
        $proc = Get-JsonValue -Obj $s -Name 'is_processing'
        if (-not (Test-JsonBoolean $proc)) { continue }
        $seen = $seen + 1
        if ([bool]$proc) { $busy = $true }
        $id = Get-JsonValue -Obj $s -Name 'id'
        if ((Test-JsonNonNegativeInteger $id) -and [long]$id -eq [long]$script:KV_SLOT_ID) { $slot = $s }
    }
    if ($seen -eq 0) { return @{ ok = $false; reason = 'the /slots answer carries no slot state' } }
    if ($null -eq $slot) { return @{ ok = $false; reason = ('the /slots answer carries no slot ' + $script:KV_SLOT_ID) } }
    $tok = Get-JsonValue -Obj $slot -Name 'n_prompt_tokens'
    $has = (Test-JsonNonNegativeInteger $tok)
    $n = [long]0
    if ($has) { $n = [long]$tok }
    return @{ ok = $true; idle = (-not $busy); has_tokens = $has; tokens = $n }
}

# B: the next write targets the OLDER generation, so the newer one is the one that survives a crash
# during it. A generation whose sidecar is absent or unreadable is the oldest possible candidate,
# which is also how a half-written generation gets reclaimed by the next write.
function Select-AutosaveGeneration {
    if ($script:WarmstartCtx.autosave_next) { return [string]$script:WarmstartCtx.autosave_next }
    $pick = $null
    foreach ($n in $script:KV_AUTOSAVE_GENERATIONS) {
        $ticks = [long][Int64]::MinValue
        $m = Read-KvSidecar -Path (Get-KvMetaPath -Name $n)
        if ($m.ok) { $ticks = Get-KvSavedAtTicks -Meta $m.value }
        if ($null -eq $pick -or [long]$ticks -lt [long]$pick.ticks) { $pick = @{ name = [string]$n; ticks = [long]$ticks } }
    }
    $script:WarmstartCtx.autosave_next = [string]$pick.name
    return [string]$pick.name
}

function Get-KvOtherGeneration {
    param([string] $Name)
    foreach ($n in $script:KV_AUTOSAVE_GENERATIONS) { if ($n -cne $Name) { return [string]$n } }
    return [string]$Name
}

# A-2 (7) input: the best available measurement of what this save is about to write. The target's
# own previous size first, then the other generation, then the stop save - all three describe the
# same slot on the same configuration. Nothing available (the very first save on this machine) means
# no estimate exists and the check is skipped, exactly as A-2 (7) says.
function Get-KvAutosaveSizeHint {
    param([string] $Name)
    $names = @([string]$Name, (Get-KvOtherGeneration -Name $Name), [string]$script:KV_CANONICAL_DATA)
    foreach ($n in $names) {
        $m = Read-KvSidecar -Path (Get-KvMetaPath -Name $n)
        if ($m.ok) {
            $b = Get-JsonValue -Obj $m.value -Name 'n_bytes'
            if ((Test-JsonNonNegativeInteger $b) -and [long]$b -gt 0) { return [long]$b }
        }
    }
    return [long]0
}

# One autosave transaction. B fixes the order: save API -> validated success -> checksum -> sidecar.
# The generation is written under its final name on purpose - the alternation IS the atomicity, and
# a half-written generation is caught by the same fail-close that catches any other damaged pair
# (its sidecar still describes the previous bytes, so length/hash disagree and both halves go).
function Invoke-AutosaveWrite {
    param($Config, [long] $Tokens)
    $name = Select-AutosaveGeneration
    $dataPath = Get-KvDataPath -Name $name
    $metaPath = Get-KvMetaPath -Name $name

    $need = Get-KvAutosaveSizeHint -Name $name
    if ($need -gt 0) {
        $free = Get-VolumeFreeMb -Path $script:WarmstartCtx.dir
        if (-not $free.ok) {
            Write-KvAutosaveDegraded -Reason ('kv volume free space query failed: ' + $free.reason) -Data $null
            return $false
        }
        $needMb = [long]([Math]::Ceiling(([double]$need * 1.1) / 1MB))
        if ([long]$free.free_mb -lt $needMb) {
            Write-KvAutosaveDegraded -Reason ('kv volume free space ' + $free.free_mb + ' MB is below the ' +
                                              $needMb + ' MB this autosave needs') -Data $null
            return $false
        }
    }

    Write-Line ('[kv] autosave: the server is idle, storing the current ' + $Tokens + ' token state...')
    # V-4 input. The save duration has never been measured (AUTOSAVE_SPEC 1-2 records it as
    # unmeasured), and the V-2 run is the first occasion that can produce it - so the two windows
    # this transaction has are timed here rather than inferred later from record timestamps:
    # elapsed_ms is the save itself (request out -> answer in) and total_ms adds the launcher's own
    # verification pass over the same bytes (checksum + header + sidecar publish).
    $tStart = Get-Date
    $r = Invoke-KvSlotAction -Config $Config -Action 'save' -FileName $name -TimeoutSec $script:KV_SAVE_TIMEOUT_S
    $saveMs = [long][Math]::Round(((Get-Date) - $tStart).TotalMilliseconds)
    $v = Test-KvSlotResponse -Response $r -FileName $name -PositiveFields @('n_saved', 'n_written')
    if (-not $v.ok) {
        if ($v.delivered) {
            # A verdict with the server no longer holding the file: this generation is dead, and its
            # old sidecar now describes bytes the truncating save destroyed, so both halves go.
            [void](Remove-KvPathBestEffort -Path $dataPath -Why 'autosave failed (response received)')
            [void](Remove-KvPathBestEffort -Path $metaPath -Why 'autosave failed (response received)')
        } else {
            # No answer at all: the server may still be writing that file, so the deletion waits for
            # the join (A-2 (8) timing (2)) - and autosave stops for this run, because reusing a
            # generation the server might still be writing is the one thing the alternation cannot
            # protect against.
            $script:WarmstartCtx.tmp_after_join += $dataPath
            $script:WarmstartCtx.tmp_after_join += $metaPath
            $script:WarmstartCtx.autosave_stopped = $true
        }
        Write-KvAutosaveDegraded -Reason $v.reason -Data @{ generation = $name; delivered = $v.delivered }
        return $false
    }

    $len = [long]0
    try { $len = [long](New-Object System.IO.FileInfo($dataPath)).Length }
    catch {
        [void](Remove-KvPathBestEffort -Path $dataPath -Why 'autosave length query failed')
        [void](Remove-KvPathBestEffort -Path $metaPath -Why 'autosave length query failed')
        Write-KvAutosaveDegraded -Reason ('saved file length query failed: ' + $_.Exception.Message) -Data $null
        return $false
    }
    if ($len -le 0) {
        [void](Remove-KvPathBestEffort -Path $dataPath -Why 'autosave produced an empty file')
        [void](Remove-KvPathBestEffort -Path $metaPath -Why 'autosave produced an empty file')
        Write-KvAutosaveDegraded -Reason 'the saved file is empty on disk' -Data $null
        return $false
    }
    Write-KvHashNotice -What 'checksumming the autosaved slot state' -Bytes $len
    $h = Get-FileSha256Lower -Path $dataPath
    $hdr = $null
    if ($h.ok) { $hdr = Get-KvFileHeaderFields -Path $dataPath }
    $shards = Get-KvModelShardShas
    $bundle = Get-KvBundleSha
    $why = $null
    if (-not $h.ok) { $why = ('autosave hashing failed: ' + [string]$h.reason) }
    elseif ($null -eq $hdr) { $why = 'the state header was never read' }
    elseif (-not $hdr.ok) { $why = [string]$hdr.reason }
    elseif ($null -eq $shards -or $null -eq $bundle) { $why = 'model shard or bundle binding inputs are unavailable' }
    if ($null -ne $why -and $why.Length -eq 0) { $why = 'the saved generation could not be verified' }
    if ($why) {
        [void](Remove-KvPathBestEffort -Path $dataPath -Why 'autosave verification failed')
        [void](Remove-KvPathBestEffort -Path $metaPath -Why 'autosave verification failed')
        Write-KvAutosaveDegraded -Reason $why -Data @{ generation = $name }
        return $false
    }

    $gen = Get-KvGenerationId
    $meta = [ordered]@{
        meta_schema_version              = [int]$script:KV_META_SCHEMA_VERSION
        profile_id                       = [string]$script:WarmstartCtx.profile_id
        model_shards_sha256              = @($shards)
        repack_manifest_sha256           = [string]$script:WarmstartCtx.manifest_sha
        engine_bundle_sha256             = [string]$bundle
        effective_state_semantics_sha256 = (Get-KvSemanticsSha256 -Argv $Config.argv -EnvVars $Config.env)
        n_ctx                            = [long](Get-KvArgvLong -Argv $Config.argv -Flag '-c')
        n_parallel                       = [long](Get-KvArgvLong -Argv $Config.argv -Flag '-np')
        llama_state_seq_magic            = [string]$hdr.magic
        llama_state_seq_version          = [long]$hdr.version
        n_tokens                         = [long]$v.values['n_saved']
        n_bytes                          = [long]$len
        kv_file_sha256                   = [string]$h.sha
        generation_id                    = [string]$gen
        saved_at                         = (Get-Date).ToUniversalTime().ToString('o')
        # A-4c kin: the same wire fields, plus the provenance of this generation. Restore ignores it
        # (C is origin-agnostic); it exists so a stored tree can be read back honestly.
        origin                           = [string]$script:KV_ORIGIN_AUTOSAVE
    }
    try {
        [System.IO.File]::WriteAllText($metaPath, ($meta | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
        $rb = Read-KvSidecar -Path $metaPath
        if (-not $rb.ok) { throw ('read-back parse failed: ' + $rb.reason) }
    } catch {
        [void](Remove-KvPathBestEffort -Path $dataPath -Why 'autosave sidecar write failed')
        [void](Remove-KvPathBestEffort -Path $metaPath -Why 'autosave sidecar write failed')
        Write-KvAutosaveDegraded -Reason ('writing the autosave sidecar failed: ' + $_.Exception.Message) -Data @{ generation = $name }
        return $false
    }

    # The write is complete: only now does the change gate move on, and only now does the other
    # generation become the target (a failed write keeps aiming at the generation it already lost).
    $script:WarmstartCtx.autosave_tokens = [long]$v.values['n_saved']
    $script:WarmstartCtx.autosave_next = (Get-KvOtherGeneration -Name $name)
    $script:WarmstartCtx.autosave_count = [int]$script:WarmstartCtx.autosave_count + 1
    Write-Line ('[kv] autosave: stored ' + $v.values['n_saved'] + ' tokens (' +
                [long]([Math]::Round($len / 1MB)) + ' MB) as a crash recovery point.')
    Write-Diag -Kind 'WARMSTART_AUTOSAVE_OK' -Data @{ generation = $name; generation_id = $gen
                                                      n_tokens = [long]$v.values['n_saved']; n_bytes = $len
                                                      kv_file_sha256 = $h.sha
                                                      origin = [string]$script:KV_ORIGIN_AUTOSAVE
                                                      autosave_count = [int]$script:WarmstartCtx.autosave_count
                                                      started_at = $tStart.ToUniversalTime().ToString('o')
                                                      elapsed_ms = [long]$saveMs
                                                      total_ms = [long][Math]::Round(((Get-Date) - $tStart).TotalMilliseconds) }
    return $true
}

# The serving loop calls this on every iteration; everything below the deadline is a cheap return.
# Fully self-contained: like the UI-9 echo it may never throw, because an autosave defect must not
# become a serving verdict.
function Invoke-AutosaveTick {
    param($Config)
    try {
        if (-not (Test-AutosaveActive)) { return }
        if ($null -eq $Config) { return }
        $now = (Get-Date)
        if ($null -eq $script:WarmstartCtx.autosave_clock) {
            # The clock starts when serving starts, so the first tick is one full period after ready.
            $script:WarmstartCtx.autosave_clock = $now
            return
        }
        if ((($now - [datetime]$script:WarmstartCtx.autosave_clock).TotalMinutes) -lt [double]$script:WarmstartCtx.autosave_minutes) { return }
        # The tick is consumed whatever the gates decide: an unmet condition means no action, and
        # the next tick is a full period away (A).
        $script:WarmstartCtx.autosave_clock = $now
        $snap = Get-KvSlotsSnapshot -Config $Config
        if (-not $snap.ok) {
            Write-KvAutosaveDegraded -Reason ('the idle probe failed: ' + $snap.reason) -Data $null
            return
        }
        if (-not $snap.idle) {
            Write-Diag -Kind 'WARMSTART_AUTOSAVE_SKIP' -Data @{ reason = 'a request is in flight' }
            return
        }
        if (-not $snap.has_tokens) {
            Write-Diag -Kind 'WARMSTART_AUTOSAVE_SKIP' -Data @{ reason = 'the slot holds no sequence yet' }
            return
        }
        if ($null -ne $script:WarmstartCtx.autosave_tokens -and
            [long]$snap.tokens -eq [long]$script:WarmstartCtx.autosave_tokens) {
            Write-Diag -Kind 'WARMSTART_AUTOSAVE_SKIP' -Data @{ reason = 'unchanged since the last save'
                                                                n_tokens = [long]$snap.tokens }
            return
        }
        [void](Invoke-AutosaveWrite -Config $Config -Tokens ([long]$snap.tokens))
    } catch {
        # Same shape as the UI-9 echo: one record, then silence for the rest of the run.
        try { $script:WarmstartCtx.autosave_stopped = $true } catch { }
        try { Write-Diag -Kind $script:KV_DIAG_SAVE_FAILED -Data @{ stage = 'autosave'
                                                                    reason = ('autosave tick fault: ' + $_.Exception.Message)
                                                                    effect = 'autosave is off for the rest of this run' } } catch { }
    }
}

# ---------------------------------------------------------------------------------------------
# LS 13-4b recovery restart - the restore ladder's last rung.
# ---------------------------------------------------------------------------------------------
# A "fresh" output path (13-4b (4)): different from the previous incarnation's AND provably absent
# at spawn time. A second-resolution timestamp alone is not enough - two recovery restarts inside
# the same second would collide on the same name, and the launcher PID is identical too - so the
# incarnation number is part of the name and the absence is then confirmed.
function New-KvRecoveryPath {
    param([string] $Path, [int] $Incarnation)
    $dir = [System.IO.Path]::GetDirectoryName($Path)
    $base = [System.IO.Path]::GetFileNameWithoutExtension($Path)
    $ext = [System.IO.Path]::GetExtension($Path)
    $n = $Incarnation
    $tries = 0
    while ($tries -lt 64) {
        $cand = Join-Path $dir ($base + '.recovery' + $n + $ext)
        if (-not (Test-Path -LiteralPath $cand)) { return $cand }
        $n = $n + 1
        $tries = $tries + 1
    }
    throw ('no free recovery path derived from ' + $Path)
}

# (2) The old child is stopped WITHOUT a save attempt and WITHOUT releasing any lock.
# Complete-Teardown is deliberately not reused: it also releases the launcher/profile/output/port
# locks, and this process is still holding them for the replacement child.
function Stop-OwnedChildForRecovery {
    param($Child, [int] $PortNumber)
    $res = Stop-OwnedChildGraceful -Child $Child -PortNumber $PortNumber
    Close-OwnedChildHandles
    $why = ''
    if ($res.taskkill_used)          { $why = 'taskkill fallback was used' }
    elseif ($res.ctrl_attempted -and -not $res.ctrl_sent) { $why = 'CTRL_BREAK could not be delivered' }
    elseif ($res.grace_exceeded)     { $why = 'graceful grace period exceeded' }
    elseif ($res.stop_nonzero)       { $why = 'child exited non-zero during stop' }
    elseif (-not $res.child_gone)    { $why = 'child process still alive' }
    elseif (-not $res.listener_gone) { $why = 'port listener still present' }
    Write-Diag -Kind 'WARMSTART_RECOVERY' -Data @{ stage = 'stop_old_child'; failed = $why }
    # (3) old-child cleanup failure keeps the existing top-priority teardown verdict.
    if ($why) { Stop-Launcher 'fail_teardown' ('warmstart recovery could not stop the previous server: ' + $why) }
}

function Invoke-WarmstartRecoveryRestart {
    param($Config, [string] $ServerExe, [string] $Root, [string] $StdOutPath, [string] $StdErrPath)
    $script:WarmstartCtx.recovery_count = $script:WarmstartCtx.recovery_count + 1
    $n = [int]$script:WarmstartCtx.recovery_count
    Write-Line '[kv] the slot erase failed as well; restarting the server so the slot is known clean.'
    Stop-OwnedChildForRecovery -Child $script:OwnedChild -PortNumber ([int]$script:LastServerPort)
    $script:ChildWasReady = $false

    # (4) a new process group and an entirely fresh file surface: metrics, stdout and stderr.
    $newOut = New-KvRecoveryPath -Path $StdOutPath -Incarnation $n
    $newErr = New-KvRecoveryPath -Path $StdErrPath -Incarnation $n
    $env2 = @{}
    foreach ($k in $Config.env.Keys) { $env2[[string]$k] = [string]$Config.env[$k] }
    if ($env2.ContainsKey($script:ENV_METRICS)) {
        # Even an externally fixed metrics path is derived into a new sibling: the engine opens that
        # file with CREATE_NEW, so re-using the name would fail the new incarnation outright.
        $env2[$script:ENV_METRICS] = New-KvRecoveryPath -Path ([string]$env2[$script:ENV_METRICS]) -Incarnation $n
        Write-Diag -Kind 'METRICS_ENV' -Data @{ injected = $true; path = $env2[$script:ENV_METRICS]
                                                recovery_incarnation = $n }
    }
    $cfg2 = @{}
    foreach ($k in $Config.Keys) { $cfg2[[string]$k] = $Config[$k] }
    $cfg2['env'] = $env2

    $sr = Start-OwnedChild -Exe $ServerExe -Args0 $Config.argv -EnvVars $env2 -WorkDir $Root `
              -StdOutPath $newOut -StdErrPath $newErr -NewProcessGroup $true -Role 'server'
    # (6) only a spawn or health failure of the replacement is terminal, and it reuses the existing
    # fail_server_start rather than inventing a status.
    if (-not $sr.ok) { Stop-Launcher 'fail_server_start' ('warmstart recovery server start failed: ' + $sr.reason) }
    $child = $sr.child
    Write-Diag -Kind 'WARMSTART_RECOVERY' -Data @{ stage = 'respawn'; incarnation = $n; pid = $child.pid
                                                   out_log = $newOut; err_log = $newErr }
    Write-Line ('[start] recovery server pid {0}; waiting for health on http://{1}:{2}/health' -f $child.pid, $cfg2.host, $cfg2.port)
    Wait-ForServerReady -Child $child -Config $cfg2 -ErrLog $newErr
    $script:ChildWasReady = $true

    # (5) the restore latch is spent, so no second attempt; the save right is kept, because storing
    # the cold-grown state on a normal stop is harmless now and useful at the next boot.
    $script:WarmstartCtx.restore_done = $true
    Set-KvVerdict -Eligible $false -Reason $script:KV_REASON_RECOVERY_COLD -Detail $null
    # (7) the launcher keeps warmup ownership: this is not a successful restore.
    Write-Line '[kv] cold(recovery_cold) - the server was restarted, so this run continues without a warm cache.'
    Write-Diag -Kind 'WARMSTART_RECOVERY' -Data @{ stage = 'ready'; incarnation = $n
                                                   reason = $script:KV_REASON_RECOVERY_COLD }
    return @{ child = $child; config = $cfg2; err_log = $newErr }
}

# endregion

# ============================================================================
# region 15. STATUS SCREEN + 3-CHOICE MENU (LS 1-1, LS 1-3, LS 6)
# ============================================================================

function Read-UserLine {
    param([string] $Prompt)
    if ($NonInteractive) { return $null }
    try { [Console]::Out.Write($Prompt) } catch { }
    try { return [Console]::In.ReadLine() } catch { return $null }
}

function Confirm-User {
    param([string] $Question)
    if ($AssumeYes) { return $true }
    if ($AssumeNo)  { return $false }
    if ($NonInteractive) { return $false }
    $a = Read-UserLine -Prompt $Question
    if ($null -eq $a) { return $false }
    $a = $a.Trim().ToLowerInvariant()
    return ($a -eq 'y' -or $a -eq 'yes')
}

function Show-Status {
    param($Profile, $Config, $ProbeResult, $Custom, $RamVerdict, $Sweep, [string] $QdSource, $SurfaceAxes = $null)
    $gates = Get-JsonValue -Obj $Profile -Name 'gates'
    $fmt = Test-JsonBooleanTrue (Get-JsonValue -Obj $gates -Name 'format_validated')
    $perf = Test-JsonBooleanTrue (Get-JsonValue -Obj $gates -Name 'performance_validated')

    Write-Line ''
    Write-Line '================ MoE-Direct - launch configuration ================'
    Write-Line ('  profile          : {0} ({1})' -f [string](Get-JsonValue -Obj $Profile -Name 'profile_id'),
                                                    [string](Get-JsonValue -Obj $Profile -Name 'display_name'))
    # LS 1-3: three separate axes. Custom only downgrades the performance axis.
    $fmtTxt = 'FAIL'
    if ($fmt) { $fmtTxt = 'PASS (repack verify)' }
    Write-Line ('  format gate      : {0}' -f $fmtTxt)
    # BUDGET_AUTOTUNE_SPEC v0.2 section 2: an autotuned budget that is not the catalog's measured
    # one puts the run off the measured operating point, so it demotes this axis for the same reason
    # a custom edit does. The rows stay ordered custom -> auto -> catalog: a custom edit is the
    # stronger statement about provenance and keeps its own wording.
    # P4 2.5 goes FIRST in this ladder: the other three rows all describe a configuration applied to
    # the catalog's model, while this one says the model itself is not the measured file. Nothing
    # below it could be a true statement about this run.
    if ($script:PinMismatchLatch) {
        Write-Line '  performance gate : [unmeasured] (the catalog source pin does not match this file)'
    } elseif ($Custom) {
        Write-Line '  performance gate : [unmeasured] (custom configuration)'
    } elseif ($Config.budget_unmeasured) {
        Write-Line '  performance gate : [unmeasured] (auto budget differs from the measured configuration)'
    } elseif (Test-WarmPathBaseline -Config $Config) {
        # UX 1-4: the product default is now warmup ON, and every published number was measured on a
        # COLD cache. Such a run is neither custom nor auto-budgeted - it is simply not the condition
        # the catalog measured - so it gets its own honest row instead of borrowing PASS. The
        # -Repro/-Smoke forced off is what puts the warmup dimension back on the official condition,
        # and it reaches this row through the same value, not through a special case.
        Write-Line '  performance gate : [unmeasured] (product warm-path baseline; official measurements are cold-cache)'
    } elseif ($perf) {
        Write-Line '  performance gate : PASS (reference machine)'
    } else {
        Write-Line '  performance gate : [unmeasured] (not performance-validated in the catalog)'
    }
    $prov = 'catalog defaults'
    if ($Custom) { $prov = 'custom' } elseif ($Config.budget_unmeasured) { $prov = 'auto' }
    Write-Line ('  config provenance: {0}' -f $prov)
    # LS OA-1 surface axes. Deliberately a block of their own rather than three more values folded
    # into the gate lines above: they answer different questions (bytes / selection authority /
    # serving validation) and the whole point of stating them separately is that a reader cannot
    # take one of the three as an answer to another.
    if ($null -ne $SurfaceAxes) {
        Write-Line ''
        Write-Line ('  copy integrity     : {0}' -f $SurfaceAxes.copy_integrity)
        Write-Line ('  inventory authority: {0}' -f $SurfaceAxes.inventory_authority)
        Write-Line ('  serving validation : {0}' -f $SurfaceAxes.serving_validation)
        if ($SurfaceAxes.note) { Write-Line ('                       {0}' -f $SurfaceAxes.note) }
    }
    Write-Line ''
    Write-Line ('  port             : {0} (loopback {1})' -f $Config.port, $Config.host)
    $ctxTxt = Get-ArgvValue -Argv $Config.argv -Flag '-c'
    if ($null -eq $ctxTxt) { $ctxTxt = '(catalog default)' }
    $thrTxt = Get-ArgvValue -Argv $Config.argv -Flag '-t'
    if ($null -eq $thrTxt) { $thrTxt = '(catalog default)' }
    Write-Line ('  ctx / threads    : {0} / {1}' -f $ctxTxt, $thrTxt)
    # BUDGET_AUTOTUNE_SPEC v0.2 section 3: no silent automatic value - the source travels with the
    # number on the launch screen, and the banner line above carries the arithmetic behind it.
    Write-Line ('  budget / QD      : {0} MB [{1}] / {2}' -f $Config.budget_mb, $Config.budget_source, $Config.qd)
    # UX 1-2 / 1-4: the default is ON since v0.2.3. When a bench mode forced it back off the line
    # states the FORCING instead of the default, because "off (default ON)" would leave the reader
    # to guess who decided (UX 1-4 table, row 3).
    $warmTxt = ('{0} (default ON)' -f $Config.warmup)
    if ($Config.warmup_forced_reason) { $warmTxt = ('{0} (forced: {1})' -f $Config.warmup, $Config.warmup_forced_reason) }
    Write-Line ('  warmup           : {0}' -f $warmTxt)
    # UX 1-2: the arch-template state belongs on the same screen as the rest of the configuration.
    # Its source is the value latched before identification - NOT the effective config, which never
    # carries it, because this is a global preference and not an allowlist override (UX 1-1).
    # The empty guard covers the dot-sourced -LibraryMode case, where nothing resolved it.
    $atTxt = [string]$script:ArchTemplateResolved
    if ($atTxt.Length -eq 0) { $atTxt = '(unresolved)' }
    Write-Line ('  arch template    : {0} (experimental - unlisted GGUFs of known architectures)' -f $atTxt)
    # LS 13-1: pre-start eligibility, NOT the result. The actual restore outcome is echoed after
    # ready, on its own line.
    Write-Line ('  kv               : {0}' -f $script:WarmstartCtx.status_text)
    # LS 13-8 (D5): the periodic save writes to the user's own machine on a cadence, so it is stated
    # on the launch screen next to the rest of the kv family rather than only in the diagnostics.
    # It reports the EFFECTIVE answer: the key can only ever turn autosave off, so a run whose
    # warmstart is off reads "off" here no matter what the key says, and says which one decided.
    $asTxt = 'off'
    if (Test-AutosaveActive) { $asTxt = ('on (every {0} min)' -f [int]$script:WarmstartCtx.autosave_minutes) }
    elseif ($script:WarmstartCtx.autosave_enabled) { $asTxt = 'off (warmstart is off)' }
    Write-Line ('  autosave         : {0}' -f $asTxt)
    # P4 3 mandatory pre-seal echo. Every field the spec names is printed on EVERY run under the
    # spec's own field name, including the ones that are null - a field that disappears when it is
    # empty cannot be consumed as a contract, because its absence and a parse failure look the same
    # to the reader. The activation field says "candidate" on purpose: the launcher proposes K/N,
    # and the engine seal is what makes an activation real.
    $pfd = $Config.prefetch
    function Format-PfField { param($V) ; if ($null -eq $V -or [string]$V -eq '') { return '(null)' } ; return [string]$V }
    Write-Line ('  effective_prefetch: {0}   [catalog evidence={1} activation={2}]' -f
                $pfd.echo, [string]$pfd.evidence, [string]$pfd.activation)
    Write-Line ('                     prefetch_request={0} catalog_evidence={1} catalog_activation={2}' -f
                (Format-PfField $pfd.request), (Format-PfField $pfd.evidence), (Format-PfField $pfd.activation))
    Write-Line ('                     launcher_candidate_activation={0} prefetch_identity={1}' -f
                (Format-PfField $pfd.candidate_activation), (Format-PfField $pfd.identity))
    Write-Line ('                     requested_k={0} requested_n={1} requested_qd={2}' -f
                (Format-PfField $pfd.k), (Format-PfField $pfd.n), (Format-PfField $Config.qd))
    Write-Line ('                     prefetch_provenance={0} prefetch_init_version={1}' -f
                (Format-PfField $pfd.provenance), (Format-PfField $pfd.init_version))
    Write-Line ('                     off_reason={0}' -f (Format-PfField $pfd.off_reason))
    Write-Line ('                     warning={0}' -f (Format-PfField $pfd.warning))
    if ($pfd.on) {
        Write-Line '                     candidate only - the engine seal decides the final activation.'
    }
    if ($RamVerdict -and $RamVerdict.verdict -eq 'unmeasured') {
        Write-Line ('  ram verdict      : [unmeasured] ({0})' -f $RamVerdict.reason)
    }
    # LS 1-4 scratch tier: pre-repack I/O sanity, provisional. Since LS 12 it no longer decides QD.
    if ($ProbeResult -and (-not $ProbeResult.ok)) {
        Write-Line ('  ssd probe        : failed ({0}) - scratch sanity only, provisional' -f $ProbeResult.reason)
    } elseif ($ProbeResult -and $ProbeResult.ok) {
        Write-Line ('  ssd probe        : {0} MiB/s (scratch sanity, provisional)' -f $ProbeResult.mibps)
    }
    # LS 12-5: the four sweep points, the adopted QD and the reason share the ssd probe block.
    if ($Sweep) {
        if ($Sweep.ok) {
            $line = ('  qd sweep         : {0} -> QD{1} ({2})' -f (Format-SweepPoints -Points $Sweep.points), $Sweep.qd, $Sweep.reason)
        } elseif (@($Sweep.points).Count -gt 0) {
            $line = ('  qd sweep         : {0} -> QD{1} (degraded, conservative default)' -f (Format-SweepPoints -Points $Sweep.points), $script:QD_DEGRADED)
        } else {
            $line = ('  qd sweep         : unavailable ({0}) -> QD{1} (conservative default)' -f $Sweep.detail, $script:QD_DEGRADED)
        }
        if ($Sweep.from_binding) { $line = $line + ' [stored binding]' }
        if ($Sweep.persist_failed) { $line = $line + ' [binding_persist=failed]' }
        Write-Line $line
        Write-Line ('  effective QD     : {0} (qd_source={1})' -f $Config.qd, $QdSource)
        # LS 12-5: only the sweep point that matches the FINAL QD may be consumed as the measured
        # read rate. No stand-in - not the selected point, not the QD1 number.
        $pt = Get-SweepPointForQd -Points $Sweep.points -Qd ([int]$Config.qd)
        if ($null -ne $pt) {
            Write-Line ('  measured io      : {0} MiB/s at QD{1} (sweep point)' -f $pt.mibps, $Config.qd)
        } else {
            Write-Line ('  measured io      : unavailable (no valid sweep point at QD{0})' -f $Config.qd)
        }
    }
    Write-Line ''
    Write-Line '  reference measurements (catalog-rendered; conditions required):'
    foreach ($row in (Format-ReferenceMeasurements -Profile $Profile)) { Write-Line $row }
    Write-Line ''
    Write-Line '  I/O ceiling is not an expected tok/s. Fixed per-token cost is machine specific and'
    Write-Line '  is shown as [unmeasured] for machines other than the reference machine.'
    Write-Line '==================================================================='
}

function Show-CustomWarning {
    Write-Line ''
    Write-Line '--- custom configuration warning ---------------------------------'
    Write-Line 'You are leaving the launcher-recommended configuration for this model.'
    Write-Line 'Custom settings are not covered by any published performance numbers'
    Write-Line '(performance shown as [unmeasured]; format integrity checks remain enforced).'
    Write-Line 'Custom values may hit RAM/VRAM limits. Locked items (prefetch state, integrity'
    Write-Line 'gates, single-slot, loopback bind) cannot be modified here.'
    Write-Line '------------------------------------------------------------------'
}

function Invoke-CustomEditor {
    param($Overrides, $Bounds)
    Show-CustomWarning
    $keys = @('port', 'ctx', 'threads', 'budget_mb', 'qd', 'warmup', 'warmstart', 'autosave')
    Write-Line 'Editable items (blank keeps the current value):'
    foreach ($k in $keys) {
        $cur = '(catalog default)'
        if ($Overrides.ContainsKey($k)) { $cur = [string]$Overrides[$k] }
        # UX 1-4: the third warmup mode was only discoverable from the README, and the default has
        # just reversed - so the one screen where the value is edited states the whole grammar.
        if ($k -eq 'warmup') { Write-Line '  warmup accepts on | off | file:<path>   (default on)' }
        $ans = Read-UserLine -Prompt ('  ' + $k + ' [' + $cur + ']: ')
        if ($null -eq $ans) { continue }
        # r1 F1: "was anything typed at all" is decided on a TRIMMED COPY, but the value handed to
        # the validator stays the raw line. Trailing whitespace is legal in a Windows path, so
        # trimming here would corrupt a warmup 'file:<path>' value before it is ever parsed. Nothing
        # else changes: every branch of Test-OverrideValue already trims its own input.
        if (([string]$ans).Trim().Length -eq 0) { continue }
        $v = Test-OverrideValue -Key $k -Value ([string]$ans) -Bounds $Bounds
        if (-not $v.ok) {
            # LS 5: interactive violations re-loop, they never terminate.
            Write-Line ('    rejected: ' + $v.reason)
            continue
        }
        $Overrides[$k] = $v.value
    }
    # UX 1-1-4: arch template is editable here too, but it is NOT an override key. It is written to
    # the GLOBAL preference file and PRESET_ALLOWLIST_KEYS stays untouched (the preset schema and
    # its schema_version are unchanged by this whole round). It also cannot take effect now -
    # identification finished several stages ago - so the echo says "from the next start" instead of
    # pretending a value changed something. The pre-identification controls are the model-menu
    # toggle and, on the -Model path, the one-shot question (UX 1-1-5).
    $curAt = [string]$script:ArchTemplateResolved
    if (-not (Test-ArchTemplateInteractiveAllowed)) {
        # Codex build r1 M1: writing here would replace the damaged preference with a clean one and
        # thereby undo the fail-close for the NEXT run - the same laundering the menu toggle is
        # blocked from doing. State it instead of prompting for a value that would be refused.
        Write-Line ('  arch template    : ' + $script:ARCH_TEMPLATE_DISCARD_LOCK_NOTE)
        return $Overrides
    }
    $ansAt = Read-UserLine -Prompt ('  arch template [' + $curAt + ']: ')
    if ($null -ne $ansAt -and ([string]$ansAt).Trim().Length -gt 0) {
        $vAt = Test-ArchTemplateValue -Value ([string]$ansAt)
        if (-not $vAt.ok) {
            # LS 5: interactive violations re-loop / are ignored, they never terminate.
            Write-Line ('    rejected: ' + $vAt.reason)
        } elseif (Save-ArchTemplatePref -Value $vAt.value) {
            Write-Line ('    arch template ' + $vAt.value + ' stored; applies from the next start (this run stays ' + $curAt + ').')
        }
    }
    return $Overrides
}

# R1-9: only an empty line means "take the default". Any other unrecognised input re-asks; it must
# never be silently promoted to start.
# LS 11 (UI-1 1/2): when - and only when - a real console is attached, the same three answers are
# offered as an arrow-key menu first. Every other path (pipe, redirect, CI, selftest, -NonInteractive,
# menu render failure) falls through to the text prompt below, unchanged byte for byte.
function Read-MenuChoice {
    if ($NonInteractive) { return $script:ActionResolved }
    if (Test-MenuModeAvailable) {
        $picked = Read-MenuChoiceInteractive
        if ($null -ne $picked) { return $picked }
    }
    # LS 11-7 a: an interactive text prompt is armed here as well (menu mode unavailable, or the
    # menu fell back), so the same queue flush applies. No-op with stdin redirected, which is why
    # the non-interactive prompt below keeps its exact v0.4 bytes and its existing regressions.
    $null = Clear-ConsoleInputQueue
    while ($true) {
        Assert-NotCancelledPreReady
        Write-Line ''
        Write-Line 'Choose: [start] / custom / stop     (Enter = start)'
        $a = Read-UserLine -Prompt 'choice> '
        if ($null -eq $a) { return 'start' }   # stdin closed: take the default
        $a = $a.Trim().ToLowerInvariant()
        if ($a.Length -eq 0) { return 'start' }
        if ($a -eq 'start')  { return 'start' }
        if ($a -eq 'custom') { return 'custom' }
        if ($a -eq 'stop')   { return 'stop' }
        Write-Line ('  unrecognised choice "' + $a + '". Enter start, custom or stop (or press Enter for start).')
    }
}

# ---------------------------------------------------------------------------------------------
# LS 11 (UI-1) - interactive selection layer.
# Display and input collection only. Nothing below writes a wire string, a status enum, an exit
# code, a gate verdict or an argv/env entry: a menu selection is injected into exactly the same
# variable the text prompt fills, and identify / shard discovery / every gate run afterwards
# unchanged. The only new stored file is the recent list (UI-1 4).
# ---------------------------------------------------------------------------------------------

$script:RECENT_MODELS_SCHEMA = 1
$script:RECENT_MODELS_MAX    = 24   # entries kept in recent_models.json
$script:RECENT_MODELS_SHOW   = 8    # recent entries offered in the menu
$script:SCAN_MODELS_SHOW     = 12   # scanned entries offered in the menu
$script:SCAN_MAX_DEPTH       = 3    # path components below the root, file included
$script:MODELS_DIR_NAME      = 'moe-models'

function Get-RecentModelsPath {
    return (Join-Path (Get-LauncherStateDir) 'recent_models.json')
}

# UI-1 3-a: the recent list is a convenience cache, never a gate input. Anything unreadable,
# undecodable, unparsable or shaped wrong is ignored (diagnostic log only) and the menu simply
# loses that source - deliberately NOT the atomic preset round trip of LS 1-7.
function Read-RecentModels {
    $path = Get-RecentModelsPath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return @() }
    $b = Read-FileBytesStrict -Path $path
    if (-not $b.ok) { Write-Diag -Kind 'RECENT_IGNORED' -Data @{ reason = $b.reason }; return @() }
    $t = ConvertFrom-Utf8Strict -Bytes $b.bytes
    if (-not $t.ok) { Write-Diag -Kind 'RECENT_IGNORED' -Data @{ reason = $t.reason }; return @() }
    $pr = ConvertFrom-JsonStrict -Text $t.text
    if (-not $pr.ok) { Write-Diag -Kind 'RECENT_IGNORED' -Data @{ reason = $pr.reason }; return @() }
    if (-not (Test-JsonArray (Get-JsonValue -Obj $pr.value -Name 'paths'))) {
        Write-Diag -Kind 'RECENT_IGNORED' -Data @{ reason = 'paths[] missing or not an array' }
        return @()
    }
    $out = @()
    foreach ($p in (Get-JsonArray -Obj $pr.value -Name 'paths')) {
        if (-not (Test-JsonNonEmptyString $p)) { continue }
        $out += [string]$p
    }
    return $out
}

# UI-1 3-a: append on identify success. Most recent first, duplicates removed, capped. A write
# failure is silent to the user and non-terminal - losing the cache costs a menu entry, nothing more.
function Add-RecentModel {
    param([string] $Path)
    try {
        if (-not $Path) { return }
        $list = @([string]$Path)
        foreach ($p in (Read-RecentModels)) {
            if ($p -eq $Path) { continue }
            $list += $p
        }
        if ($list.Count -gt $script:RECENT_MODELS_MAX) { $list = @($list[0..($script:RECENT_MODELS_MAX - 1)]) }
        $path0 = Get-RecentModelsPath
        $dir = Split-Path -Parent $path0
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $json = ([ordered]@{ schema_version = [int]$script:RECENT_MODELS_SCHEMA; paths = @($list) } | ConvertTo-Json -Depth 4)
        [System.IO.File]::WriteAllText($path0, $json, (New-Object System.Text.UTF8Encoding($false)))
        Write-Diag -Kind 'RECENT_APPENDED' -Data @{ path = $Path; count = $list.Count }
    } catch {
        Write-Diag -Kind 'RECENT_IGNORED' -Data @{ reason = ('append failed: ' + $_.Exception.Message) }
    }
}

# UI-1 3-b: a split set is offered once, through its -00001-of- member. This hides siblings from
# the LIST only; LS 1-5 discovery still runs on whatever file is finally selected.
function Test-ShardRepresentative {
    param([string] $FileName)
    $m = [regex]::Match([string]$FileName, $script:SPLIT_REGEX)
    if (-not $m.Success) { return $true }
    return ([int]$m.Groups['idx'].Value -eq 1)
}

# UI-1 3-b: *.gguf under the given roots, at most $MaxDepth path components below each root
# (1 = directly in the root, 3 = two directories down). 3 is the frozen value because the typical
# Hugging Face download lands as <root>\<model>\<hf repo name>\model.gguf - a depth-2 cut would
# miss exactly that layout. Names and sizes only: no header is opened here, identification stays
# the sole judge of what a file is.
# R6 required fix: "C:\" is a VOLUME ROOT, "C:" is that drive's current directory - a different
# place entirely (PowerShell resolves the latter per-drive). An unconditional TrimEnd('\') turned a
# legitimate "-ModelsRoot C:\" into a scan of whatever directory happened to be current on C:.
# Trim the trailing separator only when the result is still the same location.
function Get-NormalizedScanRoot {
    param([string] $Root)
    $r = [string]$Root
    if ($r.Length -eq 0) { return $r }
    $trimmed = $r.TrimEnd('\')
    if ($trimmed.Length -eq 0) { return $r }                          # "\" / "\\" - leave as given
    if ($trimmed -match '^[A-Za-z]:$') { return ($trimmed + '\') }    # C:\ stays C:\ ; bare C: becomes the root
    return $trimmed
}

function Get-GgufScanCandidates {
    param([string[]] $Roots, [int] $MaxDepth = 3)
    $out = @()
    foreach ($root in @($Roots)) {
        if (-not $root) { continue }
        $r = Get-NormalizedScanRoot -Root ([string]$root)
        if (-not (Test-Path -LiteralPath $r -PathType Container)) { continue }
        $files = @()
        try { $files = @(Get-ChildItem -LiteralPath $r -Recurse -File -Filter '*.gguf' -ErrorAction SilentlyContinue) }
        catch { Write-Diag -Kind 'SCAN_IGNORED' -Data @{ root = $r; reason = $_.Exception.Message }; continue }
        foreach ($f in $files) {
            if (-not $f.FullName.StartsWith($r, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
            $rel = $f.FullName.Substring($r.Length).TrimStart('\')
            if (@($rel -split '\\').Count -gt $MaxDepth) { continue }
            if (-not (Test-ShardRepresentative -FileName $f.Name)) { continue }
            $out += @{ path = $f.FullName; name = $f.Name; bytes = [long]$f.Length; mtime = $f.LastWriteTimeUtc }
        }
    }
    return $out
}

# UI-3 (LS 11-3 b): the size shown for a split set is the sum of its name-pattern siblings, not the
# size of the one representative file. Shard 1 is frequently metadata-only (the 397B set shows
# 10.4 MB for a 244 GB model), and a quarter-sized chunk of a 148 GB model is just as misleading.
# Contract kept: no GGUF header is opened here. Only directory-level sizes are summed, only the
# siblings that actually exist are counted, and nothing here decides whether the set is complete -
# that stays with identify / Get-ModelShardSet (LS 1-5), which runs after the selection.
function Get-ShardDisplayAggregate {
    param([string] $Path)
    $res = @{ bytes = [long](-1); shards = 1; split = $false; declared = 1 }
    $name = [System.IO.Path]::GetFileName([string]$Path)
    $m = [regex]::Match($name, $script:SPLIT_REGEX)
    if (-not $m.Success) {
        try { $res.bytes = [long](Get-Item -LiteralPath $Path -ErrorAction Stop).Length } catch { }
        return $res
    }
    $res.split = $true
    $dir = [System.IO.Path]::GetDirectoryName([string]$Path)
    $prefix = $m.Groups['base'].Value
    $cnt = [int]$m.Groups['cnt'].Value
    $res.declared = $cnt
    $sum = [long]0
    $found = 0
    for ($i = 1; $i -le $cnt; $i++) {
        # same naming rule as Get-ShardPaths (LS 1-5), but display-only: a missing sibling is
        # simply not summed instead of being an error.
        $sib = Join-Path $dir ('{0}-{1:d5}-of-{2:d5}.gguf' -f $prefix, $i, $cnt)
        if (-not (Test-Path -LiteralPath $sib -PathType Leaf)) { continue }
        try { $sum = $sum + [long](Get-Item -LiteralPath $sib -ErrorAction Stop).Length; $found++ } catch { }
    }
    if ($found -eq 0) { return $res }
    $res.bytes = $sum
    # The count describes what was actually summed, so size and count can never disagree.
    $res.shards = $found
    return $res
}

# LS 11-6-e: "is this one ready to use right now?" - the three repack artifacts sitting in the
# launcher's DEFAULT output directory for that model (<model dir>\repack\). EXISTENCE ONLY: this is
# not a verify verdict and it deliberately opens nothing. The 7-item gate still runs afterwards and
# remains the only thing that can call a repack good. A custom -OutDir is not consulted here, so a
# model repacked elsewhere simply shows no label rather than a wrong one.
function Test-RepackArtifactsPresent {
    param([string] $ModelPath)
    try {
        $dir = [System.IO.Path]::GetDirectoryName([string]$ModelPath)
        if (-not $dir) { return $false }
        $repack = Join-Path $dir 'repack'
        foreach ($n in @('experts.bin', 'manifest.json', 'verify_report.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $repack $n) -PathType Leaf)) { return $false }
        }
        return $true
    } catch { return $false }
}

# UI-1 3-b/3-c: -ModelsRoot first (when it exists), then <X>:\moe-models\ on every ready fixed drive.
function Get-ModelScanRoots {
    $roots = @()
    if ($ModelsRoot) {
        $r = ([string]$ModelsRoot).Trim().Trim('"')
        if ($r.Length -gt 0 -and (Test-Path -LiteralPath $r -PathType Container)) {
            $roots += (Get-NormalizedScanRoot -Root (Resolve-Path -LiteralPath $r).ProviderPath)
        } else {
            Write-Diag -Kind 'SCAN_IGNORED' -Data @{ root = [string]$ModelsRoot; reason = '-ModelsRoot is not an existing directory' }
        }
    }
    try {
        foreach ($d in [System.IO.DriveInfo]::GetDrives()) {
            if ($d.DriveType -ne [System.IO.DriveType]::Fixed) { continue }
            if (-not $d.IsReady) { continue }
            $conv = Join-Path $d.Name $script:MODELS_DIR_NAME
            if (Test-Path -LiteralPath $conv -PathType Container) { $roots += $conv }
        }
    } catch { Write-Diag -Kind 'SCAN_IGNORED' -Data @{ root = '(fixed drives)'; reason = $_.Exception.Message } }
    return $roots
}

# UX 1-3: the -00001-of- member of a split set, when the given path belongs to one and that member
# exists. The recent list stores whatever path the user last identified with - LS 1-5 discovery
# accepts any shard - but only shard 1 carries the structural metadata a label needs. Label reading
# only: the candidate keeps the path it was offered under, and a missing shard 1 is left for
# identify to refuse rather than being turned into an error here.
function Get-ShardRepresentativePath {
    param([string] $Path)
    $name = [System.IO.Path]::GetFileName([string]$Path)
    $m = [regex]::Match($name, $script:SPLIT_REGEX)
    if (-not $m.Success) { return [string]$Path }
    $dir = [System.IO.Path]::GetDirectoryName([string]$Path)
    $first = Join-Path $dir ('{0}-{1:d5}-of-{2:d5}.gguf' -f $m.Groups['base'].Value, 1, [int]$m.Groups['cnt'].Value)
    if (Test-Path -LiteralPath $first -PathType Leaf) { return $first }
    return [string]$Path
}

# UX 1-3: the identify block of a catalog profile, and deliberately nothing else. The REAL catalog
# verdict also weighs the shard count, the per-shard byte sizes and the source pin
# (Get-StructuralProfileCandidates / Resolve-ProfileSelection) - which is exactly why this label is
# provisional and the menu says so on screen.
function Test-CatalogIdentifyMatch {
    param($Catalog, [string] $Arch, [long] $NLayer, [long] $NExpert, [long] $NExpertUsed)
    if ($null -eq $Catalog) { return $false }
    foreach ($p in (Get-JsonArray -Obj $Catalog -Name 'profiles')) {
        $id = Get-JsonValue -Obj $p -Name 'identify'
        if ([string](Get-JsonValue -Obj $id -Name 'arch') -cne $Arch) { continue }
        if ([long](Get-JsonValue -Obj $id -Name 'n_layer') -ne $NLayer) { continue }
        if ([long](Get-JsonValue -Obj $id -Name 'n_expert') -ne $NExpert) { continue }
        if ([long](Get-JsonValue -Obj $id -Name 'n_expert_used') -ne $NExpertUsed) { continue }
        return $true
    }
    return $false
}

# UX 1-3: one candidate's provisional family label. Four answers, and the fourth is the one that
# carries the contract: a header that does not yield all four identification fields is reported as
# pending and is never guessed into a template claim (r1 Q4).
# The four fields are re-checked BY ARCH after the read rather than trusted from the reader's
# counter, because the counter matches '.block_count' on any prefix - the arch-qualified lookup is
# what actually proves the model described itself completely.
# Cost boundary: one header read per candidate, once, while the menu is built, and for a split set
# only the representative shard. Nothing here opens the body or walks a tensor table.
function Get-ModelCandidateLabel {
    param([string] $Path, $Catalog)
    $h = Read-GgufHeader -Path (Get-ShardRepresentativePath -Path $Path) -LabelMode $true
    if (-not $h.ok) { return $script:LABEL_IDENTIFY_PENDING }
    $meta = $h.meta
    if ($null -eq $meta -or -not $meta.ContainsKey('general.architecture')) { return $script:LABEL_IDENTIFY_PENDING }
    $arch = [string]$meta['general.architecture']
    $vals = @{}
    foreach ($s in @('.block_count', '.expert_count', '.expert_used_count')) {
        if (-not $meta.ContainsKey($arch + $s)) { return $script:LABEL_IDENTIFY_PENDING }
        $vals[$s] = [long]$meta[$arch + $s]
    }
    if (Test-CatalogIdentifyMatch -Catalog $Catalog -Arch $arch -NLayer $vals['.block_count'] `
            -NExpert $vals['.expert_count'] -NExpertUsed $vals['.expert_used_count']) {
        return $script:LABEL_CATALOG
    }
    if ($script:ARCH_TEMPLATE_FAMILIES -ccontains $arch) { return ($script:LABEL_TEMPLATE_PREFIX + $arch + ']') }
    return $script:LABEL_UNSUPPORTED
}

# UI-1 3: recent (existing only, newest first, max 8) followed by the scan (newest first, max 12).
# A path already offered by the recent source is not offered a second time by the scan, so every
# menu index maps to exactly one file.
# UX 1-3: this is also where the family label is read - once per offered candidate, at menu build
# time. $Catalog is optional so the label degrades to the non-catalog answers rather than throwing
# when a caller has no catalog to hand.
function Build-ModelCandidates {
    param([string[]] $RecentPaths, [string[]] $ScanRoots, $Catalog = $null)
    $items = @()
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $recentAdded = 0
    foreach ($p in @($RecentPaths)) {
        if ($recentAdded -ge $script:RECENT_MODELS_SHOW) { break }
        if (-not $p) { continue }
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { continue }
        if (-not $seen.Add([string]$p)) { continue }
        $agg = Get-ShardDisplayAggregate -Path ([string]$p)
        $items += @{ path = [string]$p; name = [System.IO.Path]::GetFileName([string]$p)
                     bytes = [long]$agg.bytes; shards = [int]$agg.shards
                     repacked = (Test-RepackArtifactsPresent -ModelPath ([string]$p))
                     label = (Get-ModelCandidateLabel -Path ([string]$p) -Catalog $Catalog); source = 'recent' }
        $recentAdded++
    }
    $scan = @(Get-GgufScanCandidates -Roots $ScanRoots -MaxDepth $script:SCAN_MAX_DEPTH)
    $scan = @($scan | Sort-Object -Property @{ Expression = { $_.mtime } } -Descending)
    $scanAdded = 0
    $truncated = $false
    foreach ($c in $scan) {
        if (-not $seen.Add([string]$c.path)) { continue }
        if ($scanAdded -ge $script:SCAN_MODELS_SHOW) { $truncated = $true; break }
        $agg = Get-ShardDisplayAggregate -Path ([string]$c.path)
        $items += @{ path = [string]$c.path; name = [string]$c.name
                     bytes = [long]$agg.bytes; shards = [int]$agg.shards
                     repacked = (Test-RepackArtifactsPresent -ModelPath ([string]$c.path))
                     label = (Get-ModelCandidateLabel -Path ([string]$c.path) -Catalog $Catalog); source = 'scan' }
        $scanAdded++
    }
    return @{ items = @($items); truncated = $truncated; recent_count = $recentAdded; scan_count = $scanAdded }
}

function Format-CandidateSize {
    param([long] $Bytes)
    if ($Bytes -lt 0) { return '(size unavailable)' }
    if ($Bytes -ge 1073741824) { return ('{0:N1} GB' -f ($Bytes / 1073741824.0)) }
    if ($Bytes -ge 1048576)    { return ('{0:N1} MB' -f ($Bytes / 1048576.0)) }
    return ('{0} B' -f $Bytes)
}

# UI-1 3 / UI-3: file name and size in the first bracket - no path, and no verdict. For a split set
# the size is the summed one and the number of summed shards is stated next to it, so a
# metadata-only shard 1 cannot read as the whole model. ASCII separator on purpose: LS 8 freezes the
# output surface as English ASCII.
# UX 1-3 revises the v0.2.2 "the header is never read for a label" rule: the family label IS read
# from one header per candidate (Get-ModelCandidateLabel), because "can this launcher prepare this
# file at all" is the question the first-run menu could not answer before. What has NOT changed is
# who decides: the label is provisional, identify after the selection is still the only verdict.
function Format-ModelCandidate {
    param($Candidate)
    $size = Format-CandidateSize -Bytes ([long]$Candidate.bytes)
    $n = 1
    if ($null -ne $Candidate.shards) { $n = [int]$Candidate.shards }
    $parts = @($size)
    if ($n -gt 1) { $parts += ('{0} shards' -f $n) }
    # LS 11-6-e: presence of the three artifacts only - never a verify verdict.
    if ($Candidate.repacked) { $parts += 'repacked' }
    $line = ('{0}   [{1}]' -f [string]$Candidate.name, ($parts -join ', '))
    # UX 1-3: a SEPARATE bracket, not one more comma item in the size bracket. The two answer
    # different questions - how big is it / can this launcher prepare it - and folding them together
    # would let one read as an answer to the other (the same rule as the LS OA-1 three axes).
    if ($Candidate.label) { $line = $line + '   ' + [string]$Candidate.label }
    return $line
}

# UI-1 1: menu mode requires a real console. Redirected stdin (pipe, file, CI, the selftest
# harness), -NonInteractive, or a host without RawUI key input all keep the v0.4 text prompts.
function Test-MenuModeAvailable {
    if ($NonInteractive) { return $false }
    try { if ([Console]::IsInputRedirected) { return $false } } catch { return $false }
    try {
        $ru = $Host.UI.RawUI
        if ($null -eq $ru) { return $false }
        $null = $ru.KeyAvailable
        $null = $ru.CursorPosition
        return $true
    } catch { return $false }
}

# UI-1 1/2: the widget. Up/Down move, Enter confirms; with -AcceptTyping the caller's own words can
# still be typed instead (the v0.4 answers stay reachable). Any host/console failure is thrown to
# the caller, which logs one diagnostic line and returns to the text prompt.
# $FocusHints (UI-6): one hint per item. Enter always confirms the FOCUSED item, so a fixed
# "Enter = start" footer read as a contradiction once the focus moved (real screen: "> stop" under
# a footer promising start). Display only - the Enter behaviour itself is unchanged. The
# non-interactive text prompt keeps its own "(Enter = start)" wording, which is accurate there.
function Show-SelectionMenu {
    param([string] $Title, [string[]] $Items, [int] $InitialIndex = 0, [string] $Hint = '',
          [string[]] $FocusHints = @(), [switch] $AcceptTyping)
    $ru = $Host.UI.RawUI
    $items0 = @($Items)
    if ($items0.Count -eq 0) { throw 'selection menu called with no items' }
    $idx = $InitialIndex
    if ($idx -lt 0 -or $idx -ge $items0.Count) { $idx = 0 }
    $typed = ''

    Write-Line ''
    if ($Title) { Write-Line $Title }
    # Pre-render blank lines first so that any scrolling happens BEFORE the redraw anchor is taken;
    # an anchor captured above a scroll would repaint the wrong rows.
    $lineCount = $items0.Count + 1
    for ($i = 0; $i -lt $lineCount; $i++) { Write-Line '' }
    $pos = $ru.CursorPosition
    $anchor = New-Object System.Management.Automation.Host.Coordinates 0, ([Math]::Max(0, [int]$pos.Y - $lineCount))

    # LS 11-7 a: arm on an empty queue - see Clear-ConsoleInputQueue.
    $null = Clear-ConsoleInputQueue
    # LS 11-7 b: Ctrl+C must arrive here AS A KEY, never as a host break. Injection probe
    # (fixtures\ctrlc_inject_probe.ps1, measured 26-07-30):
    #   ReadKey 'NoEcho,IncludeKeyDown'                        -> pipeline stopped silently, no
    #                                                             catch reached, process exit 0
    #   TreatControlCAsInput alone, same ReadKey options       -> same silent stop
    #   AllowCtrlC (with or without TreatControlCAsInput)      -> key returned (char 3), alive
    # AllowCtrlC is what stops the silent death; TreatControlCAsInput is what stops a Ctrl+C pressed
    # WHILE this read blocks from being turned into a signal that no blocked ReadKey wakes up for
    # (measured: with it set, a real Ctrl+C keystroke arrives as char 3 and no signal is raised).
    # Scoped: the previous value is restored on every exit path, including the cancellation throw.
    $ctrlcPrev = $null
    try { $ctrlcPrev = [Console]::TreatControlCAsInput; [Console]::TreatControlCAsInput = $true }
    catch { $ctrlcPrev = $null }
    try {
        while ($true) {
            Assert-NotCancelledPreReady
            $width = 78
            try { $width = [Math]::Max(20, [int]$ru.BufferSize.Width - 1) } catch { }
            $ru.CursorPosition = $anchor
            for ($i = 0; $i -lt $items0.Count; $i++) {
                $mark = '    '
                if ($i -eq $idx) { $mark = '  > ' }
                $line = $mark + [string]$items0[$i]
                if ($line.Length -gt $width) { $line = $line.Substring(0, $width) }
                Write-Line $line.PadRight($width)
            }
            $foot = [string]$Hint
            if (@($FocusHints).Count -gt $idx -and $null -ne @($FocusHints)[$idx]) { $foot = [string](@($FocusHints)[$idx]) }
            if ($AcceptTyping -and $typed.Length -gt 0) { $foot = '  typed: ' + $typed }
            if ($foot.Length -gt $width) { $foot = $foot.Substring(0, $width) }
            Write-Line $foot.PadRight($width)

            $k = $ru.ReadKey('NoEcho,IncludeKeyDown,AllowCtrlC')
            # LS 11-7 b: Ctrl+C is a cancellation, and it takes the SAME pre-ready path the console
            # signal takes (cancelled_user, STOP diagnostic, exit 2). It may never end the process
            # without a status line. Ctrl+Break still travels the signal latch, unchanged.
            if ([int][char]$k.Character -eq 3) {
                Stop-Launcher 'cancelled_user' 'ctrl+c received at the selection menu'
            }
            $vk = [int]$k.VirtualKeyCode
            if ($vk -eq 38) { $idx--; if ($idx -lt 0) { $idx = $items0.Count - 1 }; continue }          # Up
            if ($vk -eq 40) { $idx++; if ($idx -ge $items0.Count) { $idx = 0 }; continue }              # Down
            if ($vk -eq 13) { return @{ index = $idx; typed = $typed } }                                # Enter
            if (-not $AcceptTyping) { continue }
            if ($vk -eq 8) { if ($typed.Length -gt 0) { $typed = $typed.Substring(0, $typed.Length - 1) }; continue }
            $ch = [char]$k.Character
            if ([int]$ch -ge 32 -and [int]$ch -lt 127 -and $typed.Length -lt 32) { $typed = $typed + $ch }
        }
    } finally {
        if ($null -ne $ctrlcPrev) { try { [Console]::TreatControlCAsInput = $ctrlcPrev } catch { } }
    }
}

# UI-1 2: the arrow-key form of the v0.4 three-choice prompt. Returns start/custom/stop, or $null
# when the menu could not be driven - and $null means "use the text prompt", never "assume start".
function Read-MenuChoiceInteractive {
    $words = @('start', 'custom', 'stop')
    try {
        $items = @('start   - launch the server with the configuration above',
                   'custom  - edit the allowlisted values first',
                   'stop    - quit without starting')
        while ($true) {
            $r = Show-SelectionMenu -Title 'Choose: [start] / custom / stop     (Up/Down + Enter, or type the word)' `
                     -Items $items -InitialIndex 0 -Hint '  Enter = start' `
                     -FocusHints @('  Enter = start', '  Enter = custom', '  Enter = stop') -AcceptTyping
            $t = ([string]$r.typed).Trim().ToLowerInvariant()
            if ($t.Length -eq 0) { return $words[[int]$r.index] }
            if ($words -contains $t) { return $t }
            # R1-9 also holds here: an unrecognised answer re-asks, it is never promoted to start.
            Write-Line ('  unrecognised choice "' + $t + '". Enter start, custom or stop (or press Enter for start).')
        }
    } catch {
        # LS 11-7 b: a cancellation is an intent, not a render failure. Demoting it to the text
        # prompt would swallow the user's Ctrl+C; only real host/console faults fall back.
        if ($null -ne $_.Exception -and $_.Exception.GetType().FullName -eq 'MoeLauncher.LauncherExit') { throw }
        Write-Diag -Kind 'MENU_FALLBACK' -Data @{ where = 'choice'; reason = $_.Exception.Message }
        return $null
    }
}

# UI-1 3: the first-run model selection. Returns a path, or $null for "fall through to the text
# prompt" - which covers menu mode unavailable, zero candidates, the explicit "enter path manually"
# item and any render failure. The returned path is NOT validated here; Resolve-ModelPath and
# identify treat it exactly like a typed one.
function Select-ModelPathInteractive {
    param($Catalog = $null)
    if (-not (Test-MenuModeAvailable)) { return $null }
    try {
        $cand = Build-ModelCandidates -RecentPaths (Read-RecentModels) -ScanRoots (Get-ModelScanRoots) -Catalog $Catalog
        $items0 = @($cand.items)
        Write-Diag -Kind 'MODEL_MENU' -Data @{ recent = $cand.recent_count; scanned = $cand.scan_count
                                               truncated = $cand.truncated }
        if ($items0.Count -eq 0) { return $null }
        # UX 1-1-5: the toggle is only offered while the canonical CLI has not already decided this
        # run - with -ArchTemplate given, offering to change it here would be offering a lie.
        # Codex build r1 M1: and never after a strict-load discard. The toggle WRITES a preference,
        # so offering it there would recover a damaged file without the canonical CLI - exactly what
        # the UX 1-1-1 fail-close forbids. Say why, rather than dropping the row unexplained.
        $toggle = (-not $script:ArchTemplateCanonicalCli) -and (Test-ArchTemplateInteractiveAllowed)
        if ((-not $script:ArchTemplateCanonicalCli) -and (-not (Test-ArchTemplateInteractiveAllowed))) {
            Write-Line ''
            Write-Line ('  ' + $script:ARCH_TEMPLATE_DISCARD_LOCK_NOTE)
        }
        # The candidate list (and its header reads) is built ONCE; only the toggle row is re-rendered.
        while ($true) {
            $labels = @()
            foreach ($c in $items0) { $labels += (Format-ModelCandidate -Candidate $c) }
            $labels += 'enter path manually'
            $toggleIndex = -1
            if ($toggle) {
                $toggleIndex = $labels.Count
                $labels += ('arch template: {0} (toggle)' -f $script:ArchTemplateResolved)
            }
            $title = 'Select the model GGUF   (Up/Down + Enter)'
            if ($cand.truncated) {
                $title = $title + ('   [scan list truncated to the {0} most recent files]' -f $script:SCAN_MODELS_SHOW)
            }
            # UX 1-3: each label came from a single header read and cannot see the shard count, the
            # per-shard bytes or the source pin, so the screen says so rather than letting a
            # provisional answer read as the final one.
            $title = $title + "`n  " + $script:LABEL_PROVISIONAL_NOTE
            $r = Show-SelectionMenu -Title $title -Items $labels -InitialIndex 0 `
                     -Hint ('  {0} recent, {1} found under <drive>:\{2}' -f $cand.recent_count, $cand.scan_count, $script:MODELS_DIR_NAME)
            # Codex build r1 M3: recorded only NOW, because only now has the toggle actually been on
            # screen. Setting it while building the item list meant a render/host fault - which falls
            # back to the text prompt without ever drawing anything - still counted as "the user was
            # offered this control", and that silently deleted the -Model fallback question.
            if ($toggle) { $script:ArchTemplateToggleOffered = $true }
            $i = [int]$r.index
            if ($toggle -and $i -eq $toggleIndex) {
                # UX 1-1-5: recorded immediately AND latched for this run, then the menu is redrawn
                # so the new state is visible before a model is picked. This is the only control that
                # reaches the user before the first template repack - the custom editor is three
                # stages too late (selection, derive-plan, repack confirmation).
                $next = 'off'
                if ($script:ArchTemplateResolved -cne 'on') { $next = 'on' }
                Set-ArchTemplateInteractive -Value $next
                continue
            }
            if ($i -lt 0 -or $i -ge $items0.Count) { return $null }   # "enter path manually"
            return [string]$items0[$i].path
        }
    } catch {
        # LS 11-7 b: same rule as the choice menu - a LauncherExit (cancellation) is re-thrown.
        if ($null -ne $_.Exception -and $_.Exception.GetType().FullName -eq 'MoeLauncher.LauncherExit') { throw }
        Write-Diag -Kind 'MENU_FALLBACK' -Data @{ where = 'model'; reason = $_.Exception.Message }
        return $null
    }
}

# endregion

# ============================================================================
# region 16. CHILD LIFECYCLE (LS 1-8) - spawn / job bind / health / stop
# ============================================================================

$script:OwnedChild = $null
$script:LastServerPort = 0   # port the teardown path must observe as released
$script:ChildWasReady = $false   # only a child that reached ready owned the port listener
# LS 13-1: the teardown save needs the effective config of the child that is actually running
# (host/port for the POST, argv/env for the sidecar binding). After a recovery restart this points
# at the new incarnation's config, not the original one.
$script:LastServerConfig = $null

function New-InheritableFile {
    param([string] $Path)
    $sa = New-Object 'MoeLauncher.Native+SECURITY_ATTRIBUTES'
    $sa.nLength = [System.Runtime.InteropServices.Marshal]::SizeOf($sa)
    $sa.lpSecurityDescriptor = [IntPtr]::Zero
    $sa.bInheritHandle = 1
    $h = [MoeLauncher.Native]::CreateFileW($Path, $script:GENERIC_WRITE,
            ($script:FILE_SHARE_READ -bor $script:FILE_SHARE_WRITE), [ref]$sa, $script:CREATE_ALWAYS, 0, [IntPtr]::Zero)
    if ($h -eq $script:INVALID_HANDLE) { return [IntPtr]::Zero }
    return $h
}

function New-InheritableNulRead {
    $sa = New-Object 'MoeLauncher.Native+SECURITY_ATTRIBUTES'
    $sa.nLength = [System.Runtime.InteropServices.Marshal]::SizeOf($sa)
    $sa.lpSecurityDescriptor = [IntPtr]::Zero
    $sa.bInheritHandle = 1
    $h = [MoeLauncher.Native]::CreateFileW('NUL', $script:GENERIC_READ,
            ($script:FILE_SHARE_READ -bor $script:FILE_SHARE_WRITE), [ref]$sa, $script:OPEN_EXISTING, 0, [IntPtr]::Zero)
    if ($h -eq $script:INVALID_HANDLE) { return [IntPtr]::Zero }
    return $h
}

# R1-8: Windows CRT argument quoting. A naive Replace('"','\"') mis-encodes any argument that ends
# in a backslash - "D:\Model Cache\" would escape the closing quote and swallow the next argument.
# Rule (learn.microsoft.com "Parsing C command-line arguments"): a run of N backslashes is doubled
# when it precedes a quote or the closing quote, and left as-is otherwise.
function ConvertTo-CrtArgument {
    param([string] $Arg)
    if ($Arg.Length -gt 0 -and $Arg -notmatch '[\s"]') { return $Arg }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    $slashes = 0
    for ($i = 0; $i -lt $Arg.Length; $i++) {
        $c = $Arg[$i]
        if ($c -eq '\') { $slashes++; continue }
        if ($c -eq '"') {
            [void]$sb.Append([char]0x5C, ($slashes * 2 + 1))
            [void]$sb.Append('"')
            $slashes = 0
            continue
        }
        if ($slashes -gt 0) { [void]$sb.Append([char]0x5C, $slashes); $slashes = 0 }
        [void]$sb.Append($c)
    }
    if ($slashes -gt 0) { [void]$sb.Append([char]0x5C, ($slashes * 2)) }
    [void]$sb.Append('"')
    return $sb.ToString()
}

function ConvertTo-CommandLine {
    param([string] $Exe, [string[]] $Args0)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append((ConvertTo-CrtArgument -Arg $Exe))
    foreach ($a in $Args0) {
        [void]$sb.Append(' ')
        [void]$sb.Append((ConvertTo-CrtArgument -Arg ([string]$a)))
    }
    return $sb
}

# LS 13-5: the server child's environment is BUILT, not inherited.
# Merge rule (frozen): the frozen 26-key OS bootstrap allowlist, read from this process, plus
# $config.env; on a case-insensitive key collision $config.env wins, because that value is the
# launcher's own computed decision. Keys are de-duplicated and ordered ordinal-ignore-case so the
# block is byte-reproducible. The old ENV_DENY approach (strip a deny list, inherit the rest) is
# deliberately NOT reused here: it leaves every unlisted ambient variable in place, and one of
# those - NVIDIA_TF32_OVERRIDE - changes how FP32 is computed, which would break the premise that
# $config.env is the complete semantic environment surface (WARMSTART A-4c).
function New-ExplicitEnvironmentPairs {
    param([hashtable] $EnvVars)
    $merged = New-Object 'System.Collections.Generic.Dictionary[string,string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($k in $script:ENV_OS_BOOTSTRAP_ALLOWLIST) {
        $v = [System.Environment]::GetEnvironmentVariable($k)
        if ($null -ne $v) { $merged[$k] = [string]$v }
    }
    if ($null -ne $EnvVars) {
        foreach ($k in $EnvVars.Keys) { $merged[[string]$k] = [string]$EnvVars[$k] }
    }
    $keys = [string[]]@($merged.Keys)
    if ($keys.Count -gt 1) { [Array]::Sort($keys, [StringComparer]::OrdinalIgnoreCase) }
    $pairs = @()
    foreach ($k in $keys) { $pairs += @{ name = [string]$k; value = [string]$merged[$k] } }
    return , $pairs
}

# UTF-16 "KEY=VALUE\0...\0\0" block. CREATE_UNICODE_ENVIRONMENT is mandatory with this shape, and
# non-ASCII values are real here (Korean user paths reach TEMP / LOCALAPPDATA / MOE_DIRECT_DIR).
function ConvertTo-EnvironmentBlockText {
    param($Pairs)
    $sb = New-Object System.Text.StringBuilder
    foreach ($p in @($Pairs)) {
        [void]$sb.Append([string]$p.name)
        [void]$sb.Append('=')
        [void]$sb.Append([string]$p.value)
        [void]$sb.Append([char]0)
    }
    [void]$sb.Append([char]0)
    return $sb.ToString()
}

# LS 1-8 spawn binding: ownership is taken at the instant CreateProcess returns; the job bind is
# the mandatory safety net (bind failure kills the child and refuses to launch).
function Start-OwnedChild {
    param([string] $Exe, [string[]] $Args0, [hashtable] $EnvVars, [string] $WorkDir,
          [string] $StdOutPath, [string] $StdErrPath, [bool] $NewProcessGroup, [string] $Role)

    if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
        return @{ ok = $false; reason = ('executable missing: ' + $Exe) }
    }
    $hOut = New-InheritableFile -Path $StdOutPath
    $hErr = New-InheritableFile -Path $StdErrPath
    # R1-8: STARTF_USESTDHANDLES requires all three handles to be valid. A NULL stdin makes the
    # child's CRT see an invalid handle; give it an inheritable NUL device instead.
    $hIn = New-InheritableNulRead
    if ($hOut -eq [IntPtr]::Zero -or $hErr -eq [IntPtr]::Zero -or $hIn -eq [IntPtr]::Zero) {
        foreach ($h in @($hOut, $hErr, $hIn)) { if ($h -ne [IntPtr]::Zero) { [void][MoeLauncher.Native]::CloseHandle($h) } }
        return @{ ok = $false; reason = 'could not create redirected stdin/stdout/stderr handles' }
    }

    $si = New-Object 'MoeLauncher.Native+STARTUPINFO'
    $si.cb = [System.Runtime.InteropServices.Marshal]::SizeOf($si)
    $si.dwFlags = $script:STARTF_USESTDHANDLES
    $si.hStdInput  = $hIn
    $si.hStdOutput = $hOut
    $si.hStdError  = $hErr
    $pi = New-Object 'MoeLauncher.Native+PROCESS_INFORMATION'

    $flags = [uint32]0
    if ($NewProcessGroup) { $flags = [uint32]$script:CREATE_NEW_PROCESS_GROUP }

    # LS 13-5: the server role gets an explicit environment block; every other role (the repacker)
    # keeps the v0.4 contract - ambient inheritance with the deny list stripped - because changing
    # the repacker's environment surface is out of this revision's scope.
    $explicit = ($Role -ceq 'server')
    $envPtr = [IntPtr]::Zero
    $touch = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $snap = @{}
    if ($explicit) {
        $pairs = New-ExplicitEnvironmentPairs -EnvVars $EnvVars
        $flags = [uint32]($flags -bor $script:CREATE_UNICODE_ENVIRONMENT)
        # StringToHGlobalUni appends its own terminator after the string, so the block already ends
        # in the required double NUL and the extra unit past it is never read.
        $envPtr = [System.Runtime.InteropServices.Marshal]::StringToHGlobalUni((ConvertTo-EnvironmentBlockText -Pairs $pairs))
        Write-Diag -Kind 'ENV_BLOCK' -Data @{ role = $Role; mode = 'explicit'
                                              keys = @($pairs | ForEach-Object { $_.name })
                                              config_keys = @($EnvVars.Keys | ForEach-Object { [string]$_ }) }
    } else {
        # env is applied transiently to this process so the child inherits exactly the intended block
        # (moe_serve.ps1:556-580 precedent), then restored in finally.
        # R1-11: deny-by-default. Every ambient variable that can change engine behaviour or move the
        # backend search path is stripped, not just MOE_*; otherwise GGML_BACKEND_PATH from a previous
        # run or a user shell would bypass the bundle backend closure.
        foreach ($e in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
            $k = [string]$e.Key
            foreach ($pfx in $script:ENV_DENY_PREFIXES) {
                if ($k.StartsWith($pfx, [System.StringComparison]::OrdinalIgnoreCase)) { [void]$touch.Add($k); break }
            }
        }
        foreach ($n in $script:ENV_DENY_NAMES) { [void]$touch.Add($n) }
        foreach ($k in $EnvVars.Keys) { [void]$touch.Add($k) }
        $stripped = @()
        foreach ($k in $touch) {
            if ($EnvVars.ContainsKey($k)) { continue }
            if ($null -ne [System.Environment]::GetEnvironmentVariable($k)) { $stripped += $k }
        }
        if ($stripped.Count -gt 0) { Write-Diag -Kind 'ENV_STRIPPED' -Data @{ role = $Role; names = $stripped } }
        foreach ($k in $touch) { $snap[$k] = [System.Environment]::GetEnvironmentVariable($k) }
    }

    $created = $false
    try {
        if (-not $explicit) {
            foreach ($k in $touch) { [System.Environment]::SetEnvironmentVariable($k, $null) }
            foreach ($k in $EnvVars.Keys) { [System.Environment]::SetEnvironmentVariable($k, [string]$EnvVars[$k]) }
        }
        $cmd = ConvertTo-CommandLine -Exe $Exe -Args0 $Args0
        $created = [MoeLauncher.Native]::CreateProcessW($Exe, $cmd, [IntPtr]::Zero, [IntPtr]::Zero, $true,
                        $flags, $envPtr, $WorkDir, [ref]$si, [ref]$pi)
        if ($created) {
            # provisional ownership, before anything else can throw
            $script:OwnedChild = @{ role = $Role; handle = $pi.hProcess; thread = $pi.hThread; pid = $pi.dwProcessId
                                    job = [IntPtr]::Zero; out_log = $StdOutPath; err_log = $StdErrPath
                                    new_group = $NewProcessGroup; taskkill_used = $false; exited = $false; exit_code = $null }
        }
    } finally {
        if (-not $explicit) {
            foreach ($k in $touch) { [System.Environment]::SetEnvironmentVariable($k, $snap[$k]) }
        }
        if ($envPtr -ne [IntPtr]::Zero) { [System.Runtime.InteropServices.Marshal]::FreeHGlobal($envPtr) }
        [void][MoeLauncher.Native]::CloseHandle($hOut)
        [void][MoeLauncher.Native]::CloseHandle($hErr)
        [void][MoeLauncher.Native]::CloseHandle($hIn)
    }
    if (-not $created) {
        return @{ ok = $false; reason = ('CreateProcess failed gle=' + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
    }

    # kill-on-close job object = mandatory safety net (moe_serve.ps1:582-607)
    $job = [MoeLauncher.Native]::CreateJobObjectW([IntPtr]::Zero, [NullString]::Value)
    $bound = $false
    if ($job -ne [IntPtr]::Zero) {
        $info = New-Object 'MoeLauncher.Native+JOBOBJECT_EXTENDED_LIMIT_INFORMATION'
        $basic = $info.BasicLimitInformation
        $basic.LimitFlags = $script:JOB_LIMIT_KILL_ON_JOB_CLOSE
        $info.BasicLimitInformation = $basic
        $len = [System.Runtime.InteropServices.Marshal]::SizeOf($info)
        if ([MoeLauncher.Native]::SetInformationJobObject($job, $script:JOBOBJECTCLASS_EXTENDED, [ref]$info, $len)) {
            $bound = [MoeLauncher.Native]::AssignProcessToJobObject($job, $script:OwnedChild.handle)
        }
    }
    if (-not $bound) {
        try { [void][MoeLauncher.Native]::TerminateProcess($script:OwnedChild.handle, 1) } catch { }
        if ($job -ne [IntPtr]::Zero) { [void][MoeLauncher.Native]::CloseHandle($job) }
        Close-OwnedChildHandles
        return @{ ok = $false; reason = 'job object bind failed (KILL_ON_JOB_CLOSE unset or assign failed); child killed to avoid an orphan' }
    }
    $script:OwnedChild.job = $job
    Write-Diag -Kind 'CHILD_START' -Data @{ role = $Role; pid = $script:OwnedChild.pid; exe = $Exe;
                                            new_process_group = $NewProcessGroup; err_log = $StdErrPath }
    return @{ ok = $true; child = $script:OwnedChild }
}

function Close-OwnedChildHandles {
    if ($null -eq $script:OwnedChild) { return }
    foreach ($k in @('thread', 'handle')) {
        $h = $script:OwnedChild[$k]
        if ($h -and $h -ne [IntPtr]::Zero) { [void][MoeLauncher.Native]::CloseHandle($h) }
        $script:OwnedChild[$k] = [IntPtr]::Zero
    }
    if ($script:OwnedChild.job -and $script:OwnedChild.job -ne [IntPtr]::Zero) {
        [void][MoeLauncher.Native]::CloseHandle($script:OwnedChild.job)
        $script:OwnedChild.job = [IntPtr]::Zero
    }
    $script:OwnedChild = $null
}

# R1-5: a Windows crash code such as 0xC0000005 does not fit [int]; converting it in PS 5.1 throws
# an overflow exception, which on the teardown path could swallow the final status line. The DWORD
# is kept unsigned end to end and only ever rendered as hex/uint text.
function Format-ExitCode {
    param($Code)
    if ($null -eq $Code) { return 'n/a' }
    return ('{0} (0x{1:X8})' -f [uint32]$Code, [uint32]$Code)
}

function Test-ChildExited {
    param($Child)
    $code = [uint32]0
    if (-not [MoeLauncher.Native]::GetExitCodeProcess($Child.handle, [ref]$code)) { return @{ exited = $false } }
    if ([uint32]$code -eq $script:STILL_ACTIVE) { return @{ exited = $false } }
    return @{ exited = $true; code = [uint32]$code }
}

function Wait-ChildExit {
    param($Child, [int] $TimeoutMs)
    $r = [MoeLauncher.Native]::WaitForSingleObject($Child.handle, [uint32]$TimeoutMs)
    if ($r -ne $script:WAIT_OBJECT_0) { return @{ exited = $false } }
    return (Test-ChildExited -Child $Child)
}

function Read-ChildStderrComplete {
    param($Child)
    # complete lines only: a partially written trailing line must never satisfy the exact-match anchor.
    try {
        $fs = [System.IO.File]::Open($Child.err_log, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            $sr = New-Object System.IO.StreamReader($fs, (New-Object System.Text.UTF8Encoding($false)), $false)
            $raw = $sr.ReadToEnd()
        } finally { $fs.Dispose() }
    } catch { return , @() }
    $idx = $raw.LastIndexOf("`n")
    if ($idx -lt 0) { return , @() }
    $complete = $raw.Substring(0, $idx + 1)
    # $complete always ends with a newline by construction, so splitting on it yields one trailing
    # empty element that is NOT a line. It must be dropped: callers use the array LENGTH as a
    # baseline line index, and a phantom element makes that baseline one too high, which silently
    # skips the first line written after the baseline was taken.
    $parts = $complete -split "`n"
    # Preallocated fill, never $out += $ln: append reallocates the whole array per line, which is
    # O(N^2) over the capture. UI-9 re-reads the full capture every 10 s for the whole serving
    # phase, so the quadratic cost would grow without bound on a long session (measured: 25 s for
    # a 40k-line capture, under 1 s for this form on a 222k-line real log).
    $out = New-Object string[] ($parts.Count - 1)
    for ($i = 0; $i -lt ($parts.Count - 1); $i++) {
        $ln = [string]$parts[$i]
        if ($ln.EndsWith("`r")) { $ln = $ln.Substring(0, $ln.Length - 1) }
        $out[$i] = $ln
    }
    return , $out
}

# LS 5 / LS 7: exact complete line, exactly once. Substring detection is forbidden.
function Test-EnginePolicyAnchor {
    param($Child)
    $lines = Read-ChildStderrComplete -Child $Child
    $n = 0
    foreach ($ln in $lines) { if ($ln -ceq $script:ENGINE_POLICY_ANCHOR) { $n++ } }
    return ($n -eq 1)
}

# R2-6: engine seal SUCCESS evidence. See the ENGINE_SEAL_MARKER comment for why this one is a
# marker-inside-a-complete-line match instead of an exact whole-line literal.
# Gate = the marker appears in a complete line EXACTLY ONCE and its slots=X/Y field is parsable.
# The two slot numbers are recorded, never compared: X == Y is not an invariant (real passing run
# emitted 648/128 - see the ENGINE_SEAL_SLOTS_REGEX comment for the captured counter-evidence).
function Get-EngineSealAttestation {
    param($Child)
    $lines = Read-ChildStderrComplete -Child $Child
    $hits = @()
    foreach ($ln in $lines) { if ($ln.Contains($script:ENGINE_SEAL_MARKER)) { $hits += $ln } }
    if ($hits.Count -eq 0) { return @{ ok = $false; count = 0; reason = 'no seal success line' } }
    if ($hits.Count -ne 1) { return @{ ok = $false; count = $hits.Count; reason = ('seal success line seen ' + $hits.Count + ' times (expected exactly 1)') } }
    $m = [regex]::Match($hits[0], $script:ENGINE_SEAL_SLOTS_REGEX)
    if (-not $m.Success) { return @{ ok = $false; count = 1; reason = 'seal success line carries no parsable slots=X/Y'; line = $hits[0] } }
    $res = @{ ok = $true; count = 1; slots_have = [long]$m.Groups[1].Value; slots_need = [long]$m.Groups[2].Value; line = $hits[0] }
    $c = [regex]::Match($hits[0], $script:ENGINE_SEAL_COUNTS_REGEX)
    if ($c.Success) {
        $res['all'] = [long]$c.Groups[1].Value
        $res['host'] = [long]$c.Groups[2].Value
        $res['nonhost'] = [long]$c.Groups[3].Value
    }
    return $res
}

# Diagnostic only (NOT the cancel gate - see Find-BoundCancelRelease). Total number of complete
# stderr lines carrying the release() marker, useful for the diagnostic log.
function Get-SlotReleaseCount {
    param($Child)
    $lines = Read-ChildStderrComplete -Child $Child
    $n = 0
    foreach ($ln in $lines) { if ($ln.Contains($script:SLOT_RELEASE_MARKER)) { $n++ } }
    return $n
}

# @($null).Count is 1 in PowerShell, so a null/empty list must be normalised before its length is
# used as a signal. Returns a real array with the empty and null entries removed.
function Get-NonEmptyList {
    param($Value)
    if ($null -eq $Value) { return , @() }
    return , @(@($Value) | Where-Object { $null -ne $_ -and ([string]$_) -ne '' })
}

function Get-StderrLineCount {
    param($Child)
    # Plain assignment, never @(...): Read-ChildStderrComplete emits the whole array as ONE pipeline
    # object, so @(...) would wrap it and report a count of 1.
    $lines = Read-ChildStderrComplete -Child $Child
    return @($lines).Count
}

# R3-1: task-ID-bound cancel evidence. Scans complete stderr lines from $FromLineIndex for a
# "cancel task, id_task = N" warning, then requires a "| task N | stop processing:" release line
# strictly AFTER that warning. A release belonging to any other task - including a previous
# request's line that the async logger flushed late - can never satisfy this.
function Find-BoundCancelRelease {
    param($Child, [int] $FromLineIndex)
    # Plain assignment, never @(...): the reader emits the whole array as ONE pipeline object.
    $lines = Read-ChildStderrComplete -Child $Child
    $lineCount = @($lines).Count
    # A plain hashtable, NOT [ordered]: OrderedDictionary's indexer also has an [int index]
    # overload, so $dict['7'] would be resolved as "element number 7" instead of "key 7".
    # Insertion order is tracked separately so the earliest cancel is examined first.
    $cancelAt = @{}
    $cancelOrder = @()
    for ($i = $FromLineIndex; $i -lt $lineCount; $i++) {
        $m = [regex]::Match([string]$lines[$i], $script:CANCEL_TASK_REGEX)
        if ($m.Success) {
            $id = [string]$m.Groups[1].Value
            if (-not $cancelAt.ContainsKey($id)) { $cancelAt[$id] = $i; $cancelOrder += $id }
        }
    }
    if ($cancelAt.Count -eq 0) {
        # cancel_task_ids is ALWAYS present, even empty: callers latch on its length, and
        # @($null).Count is 1 in PowerShell, which would fake a warning sighting.
        return @{ found = $false; reason = 'no "cancel task, id_task" warning yet'
                  lines = $lineCount; cancel_task_ids = @() }
    }
    foreach ($id in $cancelOrder) {
        for ($j = ([int]$cancelAt[$id] + 1); $j -lt $lineCount; $j++) {
            $m2 = [regex]::Match([string]$lines[$j], $script:TASK_RELEASE_REGEX)
            if ($m2.Success -and ([string]$m2.Groups[1].Value) -eq $id) {
                return @{ found = $true; task_id = $id; cancel_index = [int]$cancelAt[$id]
                          release_index = $j; lines = $lineCount; release_line = [string]$lines[$j]
                          cancel_task_ids = $cancelOrder }
            }
        }
    }
    return @{ found = $false; lines = $lineCount; cancel_task_ids = $cancelOrder
              reason = ('cancel warning seen for task(s) ' + ($cancelOrder -join ',') + ' but no matching release after it') }
}

# LS 1-8: ready = HTTP /health 200 AND the port listener PID == owned child PID.
function Test-HealthPidBound {
    param($Child, [string] $Address, [int] $PortNumber)
    try {
        $resp = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $Address, $PortNumber) -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -ne 200) { return @{ ok = $false; reason = ('health status ' + $resp.StatusCode) } }
    } catch { return @{ ok = $false; reason = ('health request failed: ' + $_.Exception.Message) } }
    $owners = Get-PortOwnerPids -PortNumber $PortNumber
    if ($null -eq $owners) { return @{ ok = $false; reason = 'listener owner not resolvable (fail-close)' } }
    if ($owners -notcontains [int]$Child.pid) {
        Write-Diag -Kind 'HEALTH_PID_MISMATCH' -Data @{ port = $PortNumber; owners = @($owners); child_pid = $Child.pid }
        return @{ ok = $false; reason = ('listener PID mismatch: owners=' + ($owners -join ',') + ' child=' + $Child.pid); fatal = $true }
    }
    return @{ ok = $true }
}

# Returns an int[] of owning PIDs, an empty array when nothing listens, or $null when the query
# itself failed (unknown -> callers must fail-close, never treat as "no listener").
function Get-PortOwnerPids {
    param([int] $PortNumber)
    try {
        $conns = @(Get-NetTCPConnection -State Listen -LocalPort $PortNumber -ErrorAction Stop)
        return , @($conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { [int]$_ })
    } catch {
        # Get-NetTCPConnection reports "no matching listener" as ObjectNotFound; that is a real
        # answer (empty set). Anything else is an unknown and must fail-close.
        if ([string]$_.CategoryInfo.Category -eq 'ObjectNotFound') { return , @() }
        return $null
    }
}

function Test-PortListenerGone {
    param([int] $PortNumber)
    $owners = Get-PortOwnerPids -PortNumber $PortNumber
    if ($null -eq $owners) { return $false }
    return (@($owners).Count -eq 0)
}

function Test-LoopbackOnlyBinding {
    param([int] $PortNumber)
    try {
        $conns = @(Get-NetTCPConnection -State Listen -LocalPort $PortNumber -ErrorAction Stop)
        if ($conns.Count -eq 0) { return @{ ok = $false; reason = 'no listener found' } }
        foreach ($c in $conns) {
            if (-not (Test-LoopbackAddress -Address ([string]$c.LocalAddress))) {
                return @{ ok = $false; reason = ('non-loopback listener: ' + $c.LocalAddress) }
            }
        }
        return @{ ok = $true; addresses = @($conns | ForEach-Object { [string]$_.LocalAddress }) }
    } catch {
        return @{ ok = $false; reason = ('listener enumeration failed: ' + $_.Exception.Message) }
    }
}

# LS 1-8 stop procedure (b)(c)(d): CTRL_BREAK -> graceful grace period -> taskkill fallback.
# This function only reports facts; Complete-Teardown maps them to a status, because the
# "exit 0 needs all four" rule (LS 1-8 d) applies to the requested-stop path, while a failure
# path only has to prove the child and the listener are gone (LS 1-8 e).
function Stop-OwnedChildGraceful {
    param($Child, [int] $PortNumber)
    $result = @{ ctrl_attempted = $false; ctrl_sent = $false; graceful = $false; exit_code = $null
                 taskkill_used = $false; child_gone = $false; listener_gone = $false; pre_exited = $false
                 grace_exceeded = $false; stop_nonzero = $false; reason = '' }

    $pre = Test-ChildExited -Child $Child
    if ($pre.exited) {
        $result.child_gone = $true
        $result.pre_exited = $true
        $result.exit_code = $pre.code
        $result.reason = 'child had already exited before stop was requested'
    } else {
        # LS 1-8 (b): CTRL_BREAK to the owned process group. Requires both a console on this side
        # and a child created with CREATE_NEW_PROCESS_GROUP (R1-4).
        if ($Child.new_group -and (Test-ConsoleAvailable)) {
            $result.ctrl_attempted = $true
            $result.ctrl_sent = [MoeLauncher.Native]::GenerateConsoleCtrlEvent([uint32]$script:CTRL_BREAK_EVENT, [uint32]$Child.pid)
            if (-not $result.ctrl_sent) {
                $result.reason = ('GenerateConsoleCtrlEvent failed gle=' + [System.Runtime.InteropServices.Marshal]::GetLastWin32Error())
            }
        } else {
            $result.ctrl_attempted = $true
            $result.ctrl_sent = $false
            $result.reason = 'no console or child was not created with CREATE_NEW_PROCESS_GROUP'
        }
        if ($result.ctrl_sent) {
            $w = Wait-ChildExit -Child $Child -TimeoutMs ($script:GRACEFUL_STOP_S * 1000)
            if ($w.exited) {
                $result.child_gone = $true
                $result.exit_code = $w.code
                if ([uint32]$w.code -eq [uint32]0) { $result.graceful = $true }
                else { $result.stop_nonzero = $true; $result.reason = ('child exited non-zero during stop: ' + (Format-ExitCode $w.code)) }
            } else {
                $result.grace_exceeded = $true
                $result.reason = 'graceful grace period exceeded'
            }
        }
    }

    if (-not $result.child_gone) {
        $result.taskkill_used = $true
        try { & taskkill.exe /PID $Child.pid /T /F | Out-Null } catch { }
        try { [void][MoeLauncher.Native]::TerminateProcess($Child.handle, 1) } catch { }
        $w = Wait-ChildExit -Child $Child -TimeoutMs 15000
        if ($w.exited) { $result.child_gone = $true; $result.exit_code = $w.code }
    }

    $deadline = (Get-Date).AddSeconds($script:LISTENER_GONE_S)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListenerGone -PortNumber $PortNumber) { $result.listener_gone = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $result.listener_gone) { $result.listener_gone = (Test-PortListenerGone -PortNumber $PortNumber) }

    if (-not $result.reason) {
        if ($result.taskkill_used)          { $result.reason = 'taskkill fallback was used' }
        elseif (-not $result.child_gone)    { $result.reason = 'child process still alive' }
        elseif (-not $result.listener_gone) { $result.reason = 'port listener still present' }
    }
    if ($null -ne $result.exit_code) { $result.exit_code_text = (Format-ExitCode $result.exit_code) }
    Write-Diag -Kind 'TEARDOWN' -Data $result
    return $result
}

# The ready wait, factored out so the WS-1 recovery restart (LS 13-4b) runs the identical
# admission - same policy anchor classification, same PID-bound health, same timeout - instead of a
# second, subtly different copy.
function Wait-ForServerReady {
    param($Child, $Config, [string] $ErrLog)
    $t0 = Get-Date
    while ($true) {
        # LS 1-8 (a): a console stop request before ready is a plain user cancellation. The child
        # is torn down by Complete-Teardown on the way out.
        Assert-NotCancelledPreReady
        $ex = Test-ChildExited -Child $Child
        if ($ex.exited) {
            # LS 5 / LS 7: exact complete anchor line, exactly once -> policy rejection (exit 4).
            # No anchor -> resource/startup failure (exit 5).
            if (Test-EnginePolicyAnchor -Child $Child) {
                Stop-Launcher 'fail_gate_engine_seal' ('engine refused startup by policy gate (see ' + $ErrLog + ')')
            }
            Stop-Launcher 'fail_server_start' ('server child exited before ready (code=' + (Format-ExitCode $ex.code) + '); see ' + $ErrLog)
        }
        if (((Get-Date) - $t0).TotalSeconds -gt $script:READY_TIMEOUT_S) {
            Stop-Launcher 'fail_server_start' 'ready timeout'
        }
        $h = Test-HealthPidBound -Child $Child -Address $Config.host -PortNumber $Config.port
        if ($h.ok) { return }
        if ($h.fatal) { Stop-Launcher 'fail_server_start' $h.reason }
        Start-Sleep -Milliseconds $script:HEALTH_POLL_MS
    }
}

# ---------------------------------------------------------------------------------------------
# UI-9 - prefill progress echo while serving.
# A long prompt is processed 2048 tokens at a time and each chunk takes minutes on a large model,
# so a serving console looks frozen with no output at all. The engine already reports every chunk
# on its stderr; these two helpers only READ that capture and echo a human line. Display only:
# no gate, no status, no exit code and no new failure path depend on anything below.
# ---------------------------------------------------------------------------------------------

# Pure function (selftest target): new stderr lines + carried state -> display strings.
# $State = @{ next_line; prev_n; prev_p; prev_task; total_est } and is updated in place.
function Convert-PrefillProgressLines {
    param($Lines, $State)
    # @($null).Count is 1 in PowerShell, so a null list must never reach the loop: it would advance
    # next_line past a line that was never examined. The caller hands over the reader's array with a
    # plain assignment, so normalising it here is enough.
    if ($null -eq $Lines) { return , @() }
    $arr = @($Lines)
    $out = @()
    for ($i = [int]$State.next_line; $i -lt $arr.Count; $i++) {
        $m = [regex]::Match([string]$arr[$i], $script:PREFILL_PROGRESS_REGEX)
        if (-not $m.Success) { continue }
        $n    = [long]   $m.Groups[1].Value
        $p    = [double] $m.Groups[2].Value
        $rate = [double] $m.Groups[3].Value
        # UI-9b: task id from the same line, used to tell requests apart. $null (no match) is
        # fail-open - the line still displays, just without the task tag.
        $tm = [regex]::Match([string]$arr[$i], $script:PREFILL_TASK_REGEX)
        $taskId = if ($tm.Success) { $tm.Groups[1].Value } else { $null }
        # The total prompt size is estimated from two CONSECUTIVE lines of the SAME task only
        # (n_tokens exactly +2048, AND the same task id). A new task - a retry, or the next request
        # reusing the cache - restarts n_tokens at 2048 while progress jumps to whatever the cache
        # already covered (real capture: 2048/0.33 -> 4096/0.66 -> new task 2048/0.33, see
        # PREFILL_PROGRESS_REGEX). Feeding that pair into the estimate would corrupt it; the +2048
        # condition and the task id match exclude it by construction. Reference implementation:
        # bench/moe-direct/watch_prefill.ps1.
        if ($null -ne $State.prev_n -and $null -ne $State.prev_p -and
            $n -eq ([long]$State.prev_n + 2048) -and $p -gt [double]$State.prev_p -and
            $null -ne $taskId -and $taskId -eq $State.prev_task) {
            $dp = $p - [double]$State.prev_p
            if ($dp -gt 0.0001) { $State.total_est = [math]::Round(2048 / $dp) }
        }
        $State.prev_n = $n
        $State.prev_p = $p
        $State.prev_task = $taskId
        # The rate is a double: "-f" would format it with the current culture (de-DE prints 6,35).
        # The display contract is fixed ASCII with a period, so pin the invariant culture here.
        $rateTxt = ([math]::Round($rate, 2)).ToString('0.##', [System.Globalization.CultureInfo]::InvariantCulture)
        $pctTxt = if ($null -ne $taskId) { 'task {0}: {1}%' -f $taskId, [int][math]::Round($p * 100) } else { '{0}%' -f [int][math]::Round($p * 100) }
        $txt = ('[prefill] {0} at {1} tok/s' -f $pctTxt, $rateTxt)
        if ($null -ne $State.total_est -and $rate -gt 0) {
            # Estimate, never a promise: the remaining tokens come from the estimated total and the
            # rate is the one the engine reported for the chunk that just finished.
            $etaMin = [int][math]::Round((1 - $p) * [double]$State.total_est / $rate / 60)
            $txt = $txt + (' - about {0} min left (estimate)' -f $etaMin)
        }
        $out += $txt
    }
    $State.next_line = $arr.Count
    return , $out
}

# Shell (one call per serving-loop iteration). Never throws: the serving loop owns the outcome.
function Show-PrefillProgressTick {
    param($Child, $State)
    if ($State.disabled) { return }
    try {
        # 10 s throttle. A progress line lands about every 2048 tokens, which is minutes apart on a
        # large model, so a faster poll would only re-read the same stderr capture.
        $now = Get-Date
        if (($now - $State.last_check).TotalSeconds -lt 10) { return }
        $State.last_check = $now
        # Plain assignment, never @(...): the reader emits the whole array as ONE pipeline object.
        $lines = Read-ChildStderrComplete -Child $Child
        foreach ($t in (Convert-PrefillProgressLines -Lines $lines -State $State)) { Write-Line $t }
    } catch {
        # Display only: an echo fault must never disturb serving. Record it once, switch the echo
        # off for the rest of the run, and return quietly - rethrowing here would be misread as a
        # runtime failure of the server.
        $State.disabled = $true
        Write-Diag -Kind 'PREFILL_ECHO_OFF' -Data @{ reason = $_.Exception.Message }
    }
}

# endregion

# ============================================================================
# region 17. REPACK (LS 2 step 4/6)
# ============================================================================

# ---------------------------------------------------------------------------------------------
# LS 11 (UI-2) - live progress tee.
# A child's stdout is captured to a log file through an inherited handle (Start-OwnedChild); that
# file stays the authoritative capture and every existing reader/parser of it is untouched. These
# two helpers additionally echo COMPLETE lines to the console as they land, so a multi-minute
# repack cannot look frozen. Nothing here is buffered until exit: each poll iteration emits what
# has arrived since the previous one.
# ---------------------------------------------------------------------------------------------
function New-OutputTailState {
    param([string] $Path)
    # The decoder is stateful on purpose: a UTF-8 sequence split across two reads (a Korean
    # progress line straddling the current end of file) must not be decoded as replacement chars.
    return @{ path = $Path; offset = [long]0; partial = ''
              decoder = ([System.Text.Encoding]::UTF8.GetDecoder()) }
}

function Write-OutputTail {
    param($State, [switch] $Final)
    if ($null -eq $State) { return }
    # UI-2: non-interactive keeps the v0.4 behaviour exactly (the log file, nothing on the console).
    if ($NonInteractive) { return }
    try {
        if (-not (Test-Path -LiteralPath $State.path -PathType Leaf)) { return }
        $read = 0
        $buf = $null
        # Opened share-ReadWrite: the child still holds the same file open for writing.
        $fs = [System.IO.File]::Open($State.path, 'Open', 'Read', 'ReadWrite')
        try {
            if ($fs.Length -gt $State.offset) {
                $n = [int][Math]::Min([long]1048576, ($fs.Length - $State.offset))
                [void]$fs.Seek($State.offset, 'Begin')
                $buf = New-Object byte[] $n
                $read = $fs.Read($buf, 0, $n)
                $State.offset = [long]$State.offset + [long]$read
            }
        } finally { $fs.Dispose() }
        if ($read -gt 0) {
            $chars = New-Object char[] ($State.decoder.GetCharCount($buf, 0, $read, $false))
            $cn = $State.decoder.GetChars($buf, 0, $read, $chars, 0, $false)
            $State.partial = $State.partial + (New-Object string ($chars, 0, $cn))
        }
        while ($true) {
            $nl = $State.partial.IndexOf("`n")
            if ($nl -lt 0) { break }
            Write-Line ($State.partial.Substring(0, $nl).TrimEnd("`r"))
            $State.partial = $State.partial.Substring($nl + 1)
        }
        # A last line without a trailing newline is only worth printing once the child is gone.
        if ($Final -and $State.partial.Length -gt 0) {
            Write-Line $State.partial.TrimEnd("`r")
            $State.partial = ''
        }
    } catch {
        # Display only: a tee failure must never change the outcome of the repack.
        Write-Diag -Kind 'TAIL_FAILED' -Data @{ path = $State.path; reason = $_.Exception.Message }
    }
}

function Invoke-Repacker {
    param($Catalog, [string] $Root, $Profile, [string] $ModelPath, [string] $OutputDir, [bool] $PlanOnly,
          [bool] $ArchTemplate = $false, [string] $FailStatus = 'fail_repack')
    $rt = Get-JsonValue -Obj $Catalog -Name 'runtime'
    $exe = Join-Path $Root ([string](Get-JsonValue -Obj $rt -Name 'repacker_exe'))
    $args0 = @()
    foreach ($a in (Get-JsonArray -Obj $rt -Name 'repacker_argv')) {
        $s = [string]$a
        if ($s.StartsWith('./') -or $s.StartsWith('.\')) { $s = Join-Path $Root $s.Substring(2) }
        $args0 += $s
    }
    if ($PlanOnly) { $args0 += '--plan' }
    # RV 1-1 [4]: the SINGLE owner of --mode. It is read from the resolved run mode instead of being
    # taken as a parameter precisely so that no call site can decide it - the catalog plan run and
    # the real repack run are given the same mode by construction, not by two callers agreeing.
    # packed inserts nothing at all, which is what keeps the no-flag argv byte-identical (RV 4).
    if (Test-VirtualRepack) { $args0 += @('--mode', $script:REPACK_MODE_VIRTUAL) }
    if ($ArchTemplate) {
        # Two omissions, both required by the repacker's own CLI contract:
        #   --profile  is REJECTED together with --experimental-arch-template (repack_experts.py:
        #              3545-3551) - the derived expect replaces the catalog reference lock, and
        #              accepting both would leave "which one is authoritative" ambiguous.
        #   --scope    is not ours to state: the template decides it (a qwen35moe with
        #              nextn_predict_layers > 0 defaults to execution, :883-894), and there is no
        #              catalog profile here to take a routed_scope from.
        $args0 += '--experimental-arch-template'
        $args0 += @('--model', $ModelPath, '--out', $OutputDir)
    } else {
        $args0 += @('--profile', [string](Get-JsonValue -Obj $Profile -Name 'profile_id'),
                    '--model', $ModelPath, '--out', $OutputDir,
                    '--scope', [string](Get-JsonValue -Obj $Profile -Name 'routed_scope'))
    }

    $logDir = Join-Path (Get-LauncherStateDir) 'logs'
    if (-not (Test-Path -LiteralPath $logDir -PathType Container)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    # RC-1: the repacker's append-only log defaults to the directory it lives in - which, inside a
    # release bundle, is the bundle itself. The next launch then fails its own SHA manifest gate
    # (fail_gate_bundle / exit 4, reproduced twice in rehearsal). Redirect it out of the bundle.
    # Passed on every invocation: --plan writes no log, so the flag is simply unused there.
    $args0 += @('--log-path', (Join-Path $logDir 'repack_log.jsonl'))
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $tag = 'repack'
    if ($PlanOnly) { $tag = 'repack_plan' }
    $outLog = Join-Path $logDir ("{0}_{1}_out.log" -f $tag, $stamp)
    $errLog = Join-Path $logDir ("{0}_{1}_err.log" -f $tag, $stamp)

    # LS 11 (UI-2): PYTHONUNBUFFERED is load-bearing, not a nicety. CPython block-buffers stdout
    # whenever it is not a TTY, and the repacker's stdout IS a file handle here - without this the
    # per-layer progress lines (repack_experts.py:1099) would sit in the child's buffer and only
    # appear at exit, which is exactly the "buffer everything until the end" shape UI-2 forbids.
    $env0 = @{ 'PYTHONIOENCODING' = 'utf-8'; 'PYTHONUNBUFFERED' = '1' }
    $r = Start-OwnedChild -Exe $exe -Args0 $args0 -EnvVars $env0 -WorkDir $Root `
             -StdOutPath $outLog -StdErrPath $errLog -NewProcessGroup $false -Role 'repacker'
    if (-not $r.ok) { Stop-Launcher $FailStatus ('repacker spawn failed: ' + $r.reason) }
    $child = $r.child
    # R1-4: poll instead of blocking so a console stop request during a long repack is honoured.
    # The repacker is a batch child, so cancellation kills it directly and reports cancelled_user;
    # the interrupted output is left as a .partial for the next run to clean up.
    # R2-1: --plan additionally gets an explicit deadline. A stuck plan child previously froze the
    # launcher for ever; the confirmation step must never be reached in that case.
    $code = $null
    $planDeadline = $null
    if ($PlanOnly) { $planDeadline = (Get-Date).AddSeconds($script:PLAN_TIMEOUT_S) }
    # LS 11 (UI-2): only the real repack is teed. The --plan text is already printed in one piece
    # by the caller after this returns, and echoing it live as well would print it twice.
    $tail = $null
    if (-not $PlanOnly) { $tail = New-OutputTailState -Path $outLog }
    while ($true) {
        $w = Wait-ChildExit -Child $child -TimeoutMs 500
        Write-OutputTail -State $tail
        if ($w.exited) { $code = $w.code; break }
        if (Test-CancelRequested) {
            try { [void][MoeLauncher.Native]::TerminateProcess($child.handle, 1) } catch { }
            [void](Wait-ChildExit -Child $child -TimeoutMs 10000)
            Close-OwnedChildHandles
            Write-Diag -Kind 'REPACK_CANCELLED' -Data @{ plan_only = $PlanOnly }
            Stop-Launcher 'cancelled_user' 'console stop request received during the repacker run'
        }
        if ($null -ne $planDeadline -and (Get-Date) -gt $planDeadline) {
            try { [void][MoeLauncher.Native]::TerminateProcess($child.handle, 1) } catch { }
            [void](Wait-ChildExit -Child $child -TimeoutMs 10000)
            Close-OwnedChildHandles
            Write-Diag -Kind 'REPACK_PLAN_TIMEOUT' -Data @{ timeout_s = $script:PLAN_TIMEOUT_S; err_log = $errLog }
            Stop-Launcher $FailStatus ('repacker --plan exceeded the ' + $script:PLAN_TIMEOUT_S + ' s deadline and was terminated; see ' + $errLog)
        }
    }
    Write-OutputTail -State $tail -Final
    Close-OwnedChildHandles
    Write-Diag -Kind 'REPACK_DONE' -Data @{ plan_only = $PlanOnly; exit_code = $code; out_log = $outLog; err_log = $errLog }

    # R2-5b precedence: a real Ctrl+C reaches every process on the console, so the repacker child
    # (which shares the launcher's process group by design - only the server child gets its own)
    # dies from the same event. The user's stop request outranks "the repacker exited abnormally":
    # classify by intent, not by which side was observed first.
    if (Test-CancelRequested) {
        Write-Diag -Kind 'REPACK_CANCELLED' -Data @{ plan_only = $PlanOnly; exit_code = $code; via = 'console stop request' }
        Stop-Launcher 'cancelled_user' 'console stop request received during the repacker run'
    }

    $text = ''
    try { $text = [System.IO.File]::ReadAllText($outLog) } catch { }
    # R1-12: a --plan run that timed out, failed to spawn or exited non-zero must never reach the
    # user confirmation step - an empty or partial plan would be confirmed as if it were complete.
    if ($null -eq $code -or [uint32]$code -ne [uint32]0) {
        $what = 'repacker'
        if ($PlanOnly) { $what = 'repacker --plan' }
        # LS OA-1: on the arch-template plan the caller maps this to fail_model_path instead. A
        # template that fails closed (unsupported arch, an inventory that will not close) is a
        # statement about the MODEL, not a repacker malfunction.
        Stop-Launcher $FailStatus ($what + ' exited abnormally (code=' + (Format-ExitCode $code) + '); see ' + $errLog)
    }
    if ($PlanOnly) { return @{ exit_code = $code; text = $text; out_log = $outLog; err_log = $errLog } }
    # RV 1-1 [5]: the postcondition is mode-specific because the two modes produce different
    # evidence. bin ends with verify_report.json (the copy verifier's JSONL); virtual moves no bytes
    # and therefore has no copy to verify - its evidence is plan_report.json, and requiring a verify
    # report there would fail every first virtual creation.
    $postFile = 'verify_report.json'
    if (Test-VirtualRepack) { $postFile = 'plan_report.json' }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDir $postFile) -PathType Leaf)) {
        Stop-Launcher 'fail_repack' ('repacker finished but ' + $postFile + ' was not produced')
    }
    return @{ exit_code = $code; text = $text; out_log = $outLog; err_log = $errLog }
}

# endregion

# ============================================================================
# region 18. SMOKE (RS 8 checklist 1..7)
# ============================================================================

# Windows PowerShell 5.1's Invoke-WebRequest THROWS on any non-2xx instead of returning the
# response, so "did the server answer at all" has to be recovered from the exception. A delivered
# non-2xx carries the response object (WebException.Response); a timeout, a refused connection or a
# dropped socket carries none. WARMSTART A-2 (8) branches on exactly this distinction: an answered
# request means the server is no longer writing, so the failed generation's tmp can be deleted now,
# while an unanswered one has to wait until the child has been joined.
function Test-HttpResponseDelivered {
    param($ErrorRecord)
    try {
        if ($null -eq $ErrorRecord) { return $false }
        $ex = $ErrorRecord.Exception
        $depth = 0
        while ($null -ne $ex -and $depth -lt 8) {
            $r = $null
            try { $r = $ex.Response } catch { $r = $null }
            if ($null -ne $r) { return $true }
            $ex = $ex.InnerException
            $depth = $depth + 1
        }
    } catch { }
    return $false
}

function Invoke-HttpJson {
    param([string] $Uri, [string] $Method = 'GET', [string] $Body = $null, [int] $TimeoutSec = 60)
    try {
        if ($Body) {
            $resp = Invoke-WebRequest -Uri $Uri -Method $Method -Body $Body -ContentType 'application/json' `
                        -TimeoutSec $TimeoutSec -UseBasicParsing -ErrorAction Stop
        } else {
            $resp = Invoke-WebRequest -Uri $Uri -Method $Method -TimeoutSec $TimeoutSec -UseBasicParsing -ErrorAction Stop
        }
        return @{ ok = $true; status = [int]$resp.StatusCode; body = [string]$resp.Content; response_received = $true }
    } catch {
        return @{ ok = $false; reason = $_.Exception.Message
                  response_received = (Test-HttpResponseDelivered -ErrorRecord $_) }
    }
}

function Invoke-SmokeChecklist {
    param($Config, $Catalog, $Child, $GateInfo)
    $base = ('http://{0}:{1}' -f $Config.host, $Config.port)
    $fails = @()
    Write-Line ''
    Write-Line '=== first-run smoke (RELEASE_SPEC 8) ==='

    # (1) /health 200
    $r = Invoke-HttpJson -Uri ($base + '/health')
    if ($r.ok -and $r.status -eq 200) { Write-Line '  [1] /health 200                      PASS' }
    else { $fails += '1:/health'; Write-Line '  [1] /health 200                      FAIL' }

    # (2) /v1/models
    $r = Invoke-HttpJson -Uri ($base + '/v1/models')
    $ok2 = $false
    if ($r.ok -and $r.status -eq 200) {
        $j = ConvertFrom-JsonStrict -Text $r.body
        if ($j.ok -and (Test-JsonArray (Get-JsonValue -Obj $j.value -Name 'data'))) { $ok2 = $true }
    }
    if ($ok2) { Write-Line '  [2] /v1/models                       PASS' }
    else { $fails += '2:/v1/models'; Write-Line '  [2] /v1/models                       FAIL' }

    # Every stream-dependent item runs inside a guard: see the catch block for why.
    $naturalMs = 0
    $script:SmokeItemLabel = '3'
    try {
    # (3) chat completion: non-stream + stream, both must produce tokens and terminate normally.
    # R1-7: the stream is parsed as SSE JSON and real generated content is counted; a stream that
    # yields no token or never reaches [DONE] is a failure. The full run is also timed, because
    # check (4) needs to know how long an uncancelled stream naturally takes.
    $body = '{"model":"local","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"stream":false}'
    $r = Invoke-HttpJson -Uri ($base + '/v1/chat/completions') -Method 'POST' -Body $body
    $ok3a = $false
    if ($r.ok -and $r.status -eq 200) {
        $j = ConvertFrom-JsonStrict -Text $r.body
        if ($j.ok) {
            $ch = Get-JsonValue -Obj $j.value -Name 'choices'
            if ((Test-JsonArray $ch) -and @($ch).Count -gt 0) {
                $msg = Get-JsonValue -Obj (@($ch)[0]) -Name 'message'
                $content = [string](Get-JsonValue -Obj $msg -Name 'content')
                if ($content.Length -gt 0) { $ok3a = $true }
            }
        }
    }
    $stream = Invoke-SmokeStream -Uri ($base + '/v1/chat/completions') -AbortAfterTokens 0
    $ok3b = ($stream.ok -and $stream.tokens -gt 0 -and $stream.done)
    if ($ok3a -and $ok3b) { Write-Line ('  [3] chat completion stream+non-stream PASS ({0} streamed tokens, [DONE] seen)' -f $stream.tokens) }
    else {
        $fails += '3:chat'
        Write-Line ('  [3] chat completion stream+non-stream FAIL (non-stream={0} stream_tokens={1} done={2})' -f $ok3a, $stream.tokens, $stream.done)
    }
    $naturalMs = $stream.elapsed_ms
    } catch {
        $fails += $script:SmokeItemLabel + ':internal-fault'
        Write-Line ('  [' + $script:SmokeItemLabel + '] item aborted                 FAIL (internal fault: ' + $_.Exception.Message + ')')
        Write-Diag -Kind 'smoke_item_fault' -Data @{ item = $script:SmokeItemLabel; reason = $_.Exception.Message }
    }

    $script:SmokeItemLabel = '4'
    try {
    # (4) cancel mid-stream and prove the single slot really came back.
    # R2-4: a total-elapsed-time comparison is NOT a proof - a warm cache or a shorter follow-up
    # request can beat the reference time while the cancelled request still owns the slot. The
    # proof used here is the server's own request-bound release() line (SLOT_RELEASE_MARKER):
    #   1. note where stderr currently ends (a line index, not a release count)
    #   2. cancel after at least one real token, and remember how much of the natural stream was
    #      still outstanding at that instant
    #   3. require the server's own "cancel task, id_task = N" warning and then THAT SAME task's
    #      release line after it (Find-BoundCancelRelease), strictly BEFORE the cancelled stream
    #      would naturally have ended - a server that ignores the disconnect and keeps the slot
    #      until natural completion cannot satisfy this, and a late-flushed release belonging to the
    #      PREVIOUS request cannot either, because its task id does not match
    #   4. only then send a follow-up request and require it to be served
    $lineBefore = Get-StderrLineCount -Child $Child
    $relBefore = Get-SlotReleaseCount -Child $Child
    $cancel = Invoke-SmokeStream -Uri ($base + '/v1/chat/completions') -AbortAfterTokens 1
    $cancelAt = Get-Date
    $outstandingMs = $naturalMs - $cancel.elapsed_ms
    if ($outstandingMs -lt 0) { $outstandingMs = 0 }
    $naturalEnd = $cancelAt.AddMilliseconds($outstandingMs)
    # R3-2: the observation is STICKY. Polling only keeps the last sample, so under load the final
    # sample could still read "no cancel warning yet" even though the warning had arrived - which
    # made the reported failure reason timing dependent. The warning sighting is latched, and if it
    # was never seen inside the natural window a bounded diagnostic-only wait is added: the launcher
    # sent the abort, so the warning is owed to it. This cannot change the verdict - the verdict uses
    # $releaseAtMs against the prompt budget, both measured from the abort instant.
    $releaseSeen = $false
    $releaseAtMs = -1
    $warnSeen = $false
    $warnTaskIds = @()
    $bound = @{ found = $false; reason = 'not polled' }
    while ((Get-Date) -lt $naturalEnd) {
        $bound = Find-BoundCancelRelease -Child $Child -FromLineIndex $lineBefore
        $ids = Get-NonEmptyList -Value $bound.cancel_task_ids
        if ($ids.Count -gt 0) { $warnSeen = $true; $warnTaskIds = $ids }
        if ($bound.found) {
            $warnSeen = $true
            $releaseSeen = $true
            $releaseAtMs = ((Get-Date) - $cancelAt).TotalMilliseconds
            break
        }
        Start-Sleep -Milliseconds 50
    }
    if (-not $warnSeen) {
        $warnDeadline = (Get-Date).AddMilliseconds($script:CANCEL_WARN_DIAG_MS)
        while ((Get-Date) -lt $warnDeadline) {
            $probe = Find-BoundCancelRelease -Child $Child -FromLineIndex $lineBefore
            $probeIds = Get-NonEmptyList -Value $probe.cancel_task_ids
            if ($probeIds.Count -gt 0 -or $probe.found) {
                $warnSeen = $true
                $warnTaskIds = $probeIds
                $bound = $probe
                # R4-1: this wait can find the warning AND its bound release together - an async
                # logger under load flushes both past the natural end, so the launcher first sees
                # the pair here. Latching only the warning made the taxonomy below report "no
                # matching release after it" while the very same record already carried the bound
                # task id and the release line index. The release is therefore latched too, timed
                # from the abort like every other sample. That instant is necessarily past the
                # natural end and thus past the prompt budget (which is half the outstanding time),
                # so $releasedPromptly stays false: the classification gets more accurate, the
                # verdict can never turn into a PASS.
                if ($probe.found) {
                    $releaseSeen = $true
                    $releaseAtMs = ((Get-Date) - $cancelAt).TotalMilliseconds
                }
                break
            }
            Start-Sleep -Milliseconds 50
        }
    }
    # Discriminator, not a tuned threshold: a server that honours the disconnect releases the slot
    # essentially at once (latency -> 0% of the time that was still outstanding), while a server
    # that ignores it releases only when generation finishes naturally (latency -> 100%). Any split
    # inside (0,1) separates the two populations; the midpoint is the most robust choice and leaves
    # the widest margin on both sides. "Observed before the natural end" alone is NOT sufficient,
    # because an ignoring server's release lands right at that boundary and can win the race.
    $promptBudgetMs = $outstandingMs / 2.0
    $releasedPromptly = ($releaseSeen -and $releaseAtMs -ge 0 -and $releaseAtMs -lt $promptBudgetMs)
    # R3-3: complete, deterministic reason taxonomy. A server that ignores the disconnect has TWO
    # legitimate failing outcomes and which one is observed depends only on where the poll boundary
    # falls: the release may not be seen inside the window at all, or it may be seen having arrived
    # at/after the natural end. Reporting both as one vague "bound release observed" made the reason
    # look non-deterministic even though the verdict was always correct. Every branch below names the
    # evidence that produced the verdict, and all of them are task-binding evidence.
    $bindReason = ''
    if ($releasedPromptly) {
        $bindReason = ('bound release for task ' + $bound.task_id + ' observed ' + [int]$releaseAtMs +
                       ' ms after the abort, inside the ' + [int]$promptBudgetMs + ' ms prompt budget')
    } elseif ($releaseSeen) {
        $bindReason = ('bound release for task ' + $bound.task_id + ' arrived at ' + [int]$releaseAtMs +
                       ' ms, at or past the ' + [int]$promptBudgetMs + ' ms prompt budget')
    } elseif ($warnSeen) {
        $bindReason = ('cancel warning seen for task(s) ' + ($warnTaskIds -join ',') + ' but no matching release after it')
    } else {
        $bindReason = ('no "cancel task, id_task" warning within ' + $script:CANCEL_WARN_DIAG_MS + ' ms of the abort')
    }
    $after = @{ ok = $false }
    if ($releasedPromptly) { $after = Invoke-HttpJson -Uri ($base + '/v1/chat/completions') -Method 'POST' -Body $body }
    $ok4 = ($cancel.aborted -and $cancel.tokens -ge 1 -and $outstandingMs -gt 0 -and
            $releasedPromptly -and $after.ok -and $after.status -eq 200)
    if ($ok4) {
        Write-Line ('  [4] cancel + slot reclaim            PASS (aborted after {0} token(s); task {1} cancel warning then its own release observed {2} ms later, well inside the {3} ms that were still outstanding; next request served)' -f
            $cancel.tokens, $bound.task_id, [int]$releaseAtMs, [int]$outstandingMs)
    } else {
        $fails += '4:cancel'
        Write-Line ('  [4] cancel + slot reclaim            FAIL (aborted={0} tokens={1} bound_release={2} cancel_warning_seen={3} release_at_ms={4} prompt_budget_ms={5} outstanding_ms={6} next_ok={7} reason={8})' -f
            $cancel.aborted, $cancel.tokens, $releaseSeen, $warnSeen, [int]$releaseAtMs, [int]$promptBudgetMs, [int]$outstandingMs, $after.ok, $bindReason)
    }
    Write-Diag -Kind 'SMOKE_CANCEL' -Data @{ tokens = $cancel.tokens; aborted = $cancel.aborted
        bound_release = $releaseSeen; bound_task_id = [string]$bound.task_id
        cancel_warning_seen = $warnSeen; cancel_warning_task_ids = $warnTaskIds
        cancel_line_index = $bound.cancel_index; release_line_index = $bound.release_index
        bind_reason = $bindReason
        release_at_ms = $releaseAtMs; prompt_budget_ms = $promptBudgetMs
        outstanding_ms = $outstandingMs; natural_ms = $naturalMs; next_ok = $after.ok
        release_lines_total_before = $relBefore; stderr_lines_before = $lineBefore }


    } catch {
        # R3-3: a fault inside one smoke item must become THAT ITEM'S stated failure, never a
        # silent truncation of the checklist. Before this guard a load-induced exception aborted
        # Invoke-SmokeChecklist, so every later item - including the cancel verdict - emitted
        # nothing at all and the operator lost the per-item result they came for.
        $fails += $script:SmokeItemLabel + ':internal-fault'
        Write-Line ('  [' + $script:SmokeItemLabel + '] item aborted                 FAIL (internal fault: ' + $_.Exception.Message + ')')
        Write-Diag -Kind 'smoke_item_fault' -Data @{ item = $script:SmokeItemLabel; reason = $_.Exception.Message }
    }
    # (5) verify PASS consumption evidence - gated, no longer deferred.
    # NOTE Correction of an earlier 1st-source error: an earlier revision of this file claimed "there
    # is no seal-success wire line". That was WRONG. moedirect-v2-b10057.patch:14681 emits
    # LLAMA_LOG_INFO("%s: moe-direct: sealed all=... slots=X/Y ...") immediately after
    # ggml_moe_direct_seal() succeeds, i.e. after the seal has consumed verify_report.json
    # (read_verify_report_gate :3738, seal binding :7299) and fails closed on anything else.
    # So the launcher gates item 5 on that line: present in a complete line exactly once with a
    # parsable slots=X/Y field, together with the launcher's own gate having produced a
    # manifest_sha256. The slot numbers are echoed, NOT compared - X == Y is not an invariant
    # (a real passing run emitted slots=648/128; see ENGINE_SEAL_SLOTS_REGEX for the capture).
    $seal = Get-EngineSealAttestation -Child $Child
    $ok5 = ($seal.ok -and $null -ne $GateInfo -and $null -ne $GateInfo.manifest_sha256)
    if ($ok5) {
        Write-Line ('  [5] verify PASS consumption          PASS (engine seal line x1, slots {0}/{1}; launcher gate manifest {2})' -f
            $seal.slots_have, $seal.slots_need, $GateInfo.manifest_sha256.Substring(0, 12))
    } else {
        $fails += '5:verify-consume'
        Write-Line ('  [5] verify PASS consumption          FAIL ({0})' -f $seal.reason)
    }
    Write-Diag -Kind 'SMOKE_ITEM5' -Data @{ ok = $ok5; seal = $seal; manifest_sha256 = $GateInfo.manifest_sha256
        note = 'gated on the engine post-seal INFO line (patch:14681); a stronger always-on identity echo carrying profile_id/expect_sha256/manifest_sha256 remains an engine-round item' }

    # (6) loopback-only binding
    $lb = Test-LoopbackOnlyBinding -PortNumber $Config.port
    if ($lb.ok) { Write-Line ('  [6] loopback-only binding            PASS (' + ($lb.addresses -join ',') + ')') }
    else { $fails += '6:loopback'; Write-Line ('  [6] loopback-only binding            FAIL (' + $lb.reason + ')') }

    # (7) built-in web UI: load AND send. R1-7: a GET alone does not prove the UI can talk to the
    # server, so the check also posts a completion through the same origin.
    $webui = Test-JsonBooleanTrue (Get-JsonValue -Obj (Get-JsonValue -Obj $Catalog -Name 'runtime') -Name 'webui')
    if ($webui) {
        $rGet = Invoke-HttpJson -Uri ($base + '/')
        $okLoad = ($rGet.ok -and $rGet.status -eq 200 -and $rGet.body.Length -gt 0)
        $rSend = Invoke-HttpJson -Uri ($base + '/v1/chat/completions') -Method 'POST' -Body $body
        $okSend = $false
        if ($rSend.ok -and $rSend.status -eq 200) {
            $j = ConvertFrom-JsonStrict -Text $rSend.body
            if ($j.ok) {
                $ch = Get-JsonValue -Obj $j.value -Name 'choices'
                if ((Test-JsonArray $ch) -and @($ch).Count -gt 0) { $okSend = $true }
            }
        }
        if ($okLoad -and $okSend) { Write-Line '  [7] built-in web UI load + send      PASS' }
        else { $fails += '7:webui'; Write-Line ('  [7] built-in web UI load + send      FAIL (load={0} send={1})' -f $okLoad, $okSend) }
    } else {
        Write-Line '  [7] built-in web UI                  SKIP (API-only bundle)'
    }

    Write-Diag -Kind 'SMOKE' -Data @{ failures = $fails }
    if ($fails.Count -gt 0) { return @{ ok = $false; failures = $fails } }
    return @{ ok = $true }
}

# R1-7: SSE reader that counts REAL generated tokens (non-empty delta content) and requires the
# terminating [DONE]. An abort is only reported as such when at least the requested number of
# tokens was actually received first - a connection exception before any token is a failure.
function Invoke-SmokeStream {
    param([string] $Uri, [int] $AbortAfterTokens)
    $body = '{"model":"local","messages":[{"role":"user","content":"ping"}],"max_tokens":64,"stream":true}'
    $req = $null
    $tokens = 0
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $req = [System.Net.HttpWebRequest]::Create($Uri)
        $req.Method = 'POST'
        $req.ContentType = 'application/json'
        $req.Timeout = 60000
        $req.ReadWriteTimeout = 60000
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
        $req.ContentLength = $bytes.Length
        $rs = $req.GetRequestStream()
        $rs.Write($bytes, 0, $bytes.Length)
        $rs.Close()
        $resp = $req.GetResponse()
        $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
        $done = $false
        while (-not $sr.EndOfStream) {
            $line = $sr.ReadLine()
            if ($null -eq $line) { break }
            if (-not $line.StartsWith('data:')) { continue }
            $payload = $line.Substring(5).Trim()
            if ($payload -eq '[DONE]') { $done = $true; break }
            # count only chunks that carry real generated content
            $pj = ConvertFrom-JsonStrict -Text $payload
            if (-not $pj.ok) { continue }
            $ch = Get-JsonValue -Obj $pj.value -Name 'choices'
            if (-not (Test-JsonArray $ch) -or @($ch).Count -eq 0) { continue }
            $delta = Get-JsonValue -Obj (@($ch)[0]) -Name 'delta'
            $content = ''
            if ($null -ne $delta) { $content = [string](Get-JsonValue -Obj $delta -Name 'content') }
            if ($content.Length -eq 0) { continue }
            $tokens++
            if ($AbortAfterTokens -gt 0 -and $tokens -ge $AbortAfterTokens) {
                $req.Abort()
                try { $sr.Dispose() } catch { }
                try { $resp.Close() } catch { }
                $sw.Stop()
                return @{ ok = $true; aborted = $true; tokens = $tokens; done = $false; elapsed_ms = $sw.Elapsed.TotalMilliseconds }
            }
        }
        try { $sr.Dispose() } catch { }
        try { $resp.Close() } catch { }
        $sw.Stop()
        return @{ ok = $true; aborted = $false; tokens = $tokens; done = $done; elapsed_ms = $sw.Elapsed.TotalMilliseconds }
    } catch {
        $sw.Stop()
        # An exception before any token was received is NOT a proven cancel.
        return @{ ok = $false; reason = $_.Exception.Message; aborted = ($AbortAfterTokens -gt 0 -and $tokens -ge $AbortAfterTokens)
                  tokens = $tokens; done = $false; elapsed_ms = $sw.Elapsed.TotalMilliseconds }
    }
}

# endregion

# ============================================================================
# region 19. MAIN
# ============================================================================

# R1-9: -Action / -RunSeconds are validated here, not by the parameter binder, so a bad value still
# produces a status line instead of a bare exit 1 with zero wire output.
$script:ActionResolved = 'start'
$script:RunSecondsResolved = 0
# RV 3: resolved once at the top of the run and read everywhere else. Initialised here so the
# dot-sourced -LibraryMode path (launcher_selftest.ps1) always has a defined mode.
$script:RepackModeResolved = 'packed'
# RV 2-4 (2): the preset read the virtual path performs EARLY, kept so the v0.4 merge point reuses
# it instead of reading - and logging - the same preset a second time.
$script:VirtualPresetEarly = $null

function Test-VirtualRepack {
    return ([string]$script:RepackModeResolved -ceq $script:REPACK_MODE_VIRTUAL)
}

# RV 3: same discipline as -Action - a raw string, trimmed and lower-cased, validated here so an
# invalid value still produces a status line instead of a binder death with no wire output.
function Resolve-RepackMode {
    # The early-preset slot belongs to ONE run; clearing it here means a dot-sourced host that
    # drives this function twice cannot inherit the previous run's preset.
    $script:VirtualPresetEarly = $null
    $raw = ([string]$RepackMode).Trim()
    if ($raw.Length -eq 0) { $script:RepackModeResolved = $script:REPACK_MODE_DEFAULT; return }
    $v = $raw.ToLowerInvariant()
    if ($script:REPACK_MODE_VALUES -cnotcontains $v) {
        Stop-Launcher 'fail_custom_args' ("invalid -RepackMode '" + $RepackMode + "': expected packed or virtual")
    }
    $script:RepackModeResolved = $v
}

# RV 2-4: the CLI half of the pinned-shape refusal. Timing is the contract, not a preference: it
# runs after the mode is fixed and while the RAW values still exist, because Get-CliOverrides
# silently drops an out-of-bounds value on an interactive run - after it, a refused -QD and an
# absent one are indistinguishable. A silent ignore is exactly what this must not do.
#
# -Prefetch is judged on the REQUEST enum (catalog | init | adapt): 'catalog-fixed' is a resolver
# RESULT and never an accepted request. Absent and 'catalog' pass; anything else stops.
function Assert-VirtualCliPins {
    if (-not (Test-VirtualRepack)) { return }
    if ($null -ne $QD -and ([string]$QD).Trim().Length -gt 0) {
        Stop-Launcher 'fail_custom_args' ("-QD '" + $QD + "' is not accepted with -RepackMode virtual: " + $script:VIRTUAL_PIN_REASON)
    }
    $pf = ([string]$Prefetch).Trim()
    if ($pf.Length -gt 0 -and $pf.ToLowerInvariant() -cne $script:PREFETCH_REQUEST_DEFAULT) {
        Stop-Launcher 'fail_custom_args' ("-Prefetch '" + $Prefetch + "' is not accepted with -RepackMode virtual: " + $script:VIRTUAL_PIN_REASON)
    }
}

# RV 2-4: the same refusal for the two override MAP layers - the stored preset and the interactive
# custom editor. Both call sites invoke this before anything downstream can act on the value: the
# preset before the partial cleanup, the preflight, the plan and the repack; the custom editor
# before the prefetch re-resolve, the config rebuild and the preset save.
function Assert-VirtualOverridePins {
    param($Overrides, [string] $Origin)
    if (-not (Test-VirtualRepack)) { return }
    if ($null -eq $Overrides) { return }
    if ($Overrides.ContainsKey('qd')) {
        Stop-Launcher 'fail_custom_args' ('qd from ' + $Origin + ' is not accepted with -RepackMode virtual: ' + $script:VIRTUAL_PIN_REASON)
    }
    if ($Overrides.ContainsKey('prefetch')) {
        $v = ([string]$Overrides['prefetch']).Trim().ToLowerInvariant()
        if ($v -cne $script:PREFETCH_REQUEST_DEFAULT) {
            Stop-Launcher 'fail_custom_args' ("prefetch '" + $v + "' from " + $Origin +
                ' is not accepted with -RepackMode virtual: ' + $script:VIRTUAL_PIN_REASON)
        }
    }
}

function Resolve-ExtraCliArgs {
    $a = ([string]$Action).Trim().ToLowerInvariant()
    if ($a.Length -eq 0) { $a = 'start' }
    if (@('start', 'stop') -notcontains $a) {
        Stop-Launcher 'fail_custom_args' ("invalid -Action '" + $Action + "': expected start or stop")
    }
    $script:ActionResolved = $a

    $rsRaw = ([string]$RunSeconds).Trim()
    if ($rsRaw.Length -eq 0) { $rsRaw = '0' }
    $rs = [long]0
    if (-not [long]::TryParse($rsRaw, [ref]$rs)) {
        Stop-Launcher 'fail_custom_args' ("invalid -RunSeconds '" + $RunSeconds + "': not an integer")
    }
    if ($rs -lt 0) { Stop-Launcher 'fail_custom_args' ("invalid -RunSeconds '" + $RunSeconds + "': must not be negative") }
    if ($rs -gt 86400) { Stop-Launcher 'fail_custom_args' ("invalid -RunSeconds '" + $RunSeconds + "': above the 86400 s ceiling") }
    $script:RunSecondsResolved = [int]$rs
}

function Get-CliOverrides {
    param($Bounds)
    $ov = @{}
    $pairs = @(@('port', $Port), @('ctx', $Ctx), @('threads', $Threads),
               @('budget_mb', $BudgetMB), @('qd', $QD), @('warmup', $Warmup),
               @('warmstart', $Warmstart), @('autosave', $Autosave),
               @('prefetch', $Prefetch))
    foreach ($p in $pairs) {
        $k = $p[0]; $v = $p[1]
        if ($null -eq $v -or ([string]$v).Trim().Length -eq 0) { continue }
        $r = Test-OverrideValue -Key $k -Value ([string]$v) -Bounds $Bounds
        if (-not $r.ok) {
            # LS 5: non-interactive (argument driven) type/bounds violations terminate.
            if ($NonInteractive) { Stop-Launcher 'fail_custom_args' ('invalid -' + $k + ': ' + $r.reason) }
            Write-Line ('[custom] ignoring invalid -' + $k + ': ' + $r.reason)
            continue
        }
        $ov[$k] = $r.value
    }
    return $ov
}

# R1-8: every path the child will see is absolutised once against the launcher's own working
# directory. The children run with the bundle root as workdir, so a relative -OutDir such as
# ".\cache" would otherwise be checked here and resolved somewhere else there.
# Pure path arithmetic - it creates nothing. That property is what lets the derived branch call it
# early (the derive-plan needs the output VOLUME) while the catalog branch keeps calling it at the
# v0.4 point, after the CLI overrides have been validated.
function Resolve-OutputDirectory {
    param([string] $ModelPath)
    $outputDir = $OutDir
    if (-not $outputDir) {
        # RV 1-1 [2]: the two modes get separate default directories, so an opt-in run never lands
        # on the other mode's artifacts by accident. An explicit -OutDir still wins, and the
        # mode-intent check on the existing artifacts is what covers that case instead.
        $leaf = $script:REPACK_DIR_PACKED
        if (Test-VirtualRepack) { $leaf = $script:REPACK_DIR_VIRTUAL }
        $outputDir = Join-Path ([System.IO.Path]::GetDirectoryName($ModelPath)) $leaf
    }
    try {
        if (-not [System.IO.Path]::IsPathRooted($outputDir)) {
            $outputDir = Join-Path (Get-Location).ProviderPath $outputDir
        }
        $outputDir = [System.IO.Path]::GetFullPath($outputDir)
        if ($outputDir.Length -gt 3) { $outputDir = $outputDir.TrimEnd('\') }
    } catch { Stop-Launcher 'fail_model_path' ('output directory path is not valid: ' + $outputDir) }
    return $outputDir
}

function Resolve-ModelPath {
    # UX 1-3: the catalog is threaded through purely so a candidate whose four identification fields
    # match a shipped profile can say [catalog] instead of claiming a template it will never take.
    param($Catalog = $null)
    if ($Model) { $p = $Model }
    else {
        # LS 11 (UI-1 3): the selection menu fills the SAME variable the text prompt fills, and
        # returns $null whenever it is not applicable - menu mode unavailable, no candidate, or
        # "enter path manually". Everything after this point is the v0.4 path, unchanged.
        $p = Select-ModelPathInteractive -Catalog $Catalog
        if ($null -eq $p) {
            $p = Read-UserLine -Prompt 'Model GGUF path> '
            if ($null -eq $p) { Stop-Launcher 'fail_model_path' 'no model path supplied' }
        }
    }
    $p = ([string]$p).Trim().Trim('"')
    if ($p.Length -eq 0) { Stop-Launcher 'fail_model_path' 'empty model path' }
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Stop-Launcher 'fail_model_path' ('model file not found: ' + $p) }
    return (Resolve-Path -LiteralPath $p).ProviderPath
}

# R1-1: LS 1-7 step 7 "1-shot re-sizing". Runs the RAM/disk verdict against the effective budget
# and ctx (not the catalog defaults), then reserves the effective port. Called once after the
# preset+CLI binding and again after every custom edit.
function Confirm-EffectiveSizing {
    param([string] $OutputDir, [string] $ExpectPath, $Config, $Profile)
    Set-FailureStage 'fail_resource'
    $ctx = 0
    $ctxTxt = Get-ArgvValue -Argv $Config.argv -Flag '-c'
    if ($null -ne $ctxTxt) { [void][int]::TryParse([string]$ctxTxt, [ref]$ctx) }
    Write-Line ''
    Write-Line '=== effective sizing (after preset + CLI binding) ==='
    $pre = Invoke-Preflight -OutputDir $OutputDir -ExpectPath $ExpectPath -BudgetMb ([long]$Config.budget_mb) `
               -NeedsRepack $false -CtxTokens ([long]$ctx)
    Set-FailureStage 'fail_instance_lock'
    Set-EffectivePortLock -PortNumber ([int]$Config.port)
    Write-Diag -Kind 'EFFECTIVE_SIZING' -Data @{ budget_mb = $Config.budget_mb; ctx = $ctx; port = $Config.port
                                                 ram_verdict = $pre.ram.verdict }
    return $pre
}

# LS 13-1 / LS 13-7 (8): A-1 confirmation #2 plus the EFFECTIVE record, as ONE step.
# The three-choice loop can wait indefinitely, so the directory that existed at the first
# confirmation may be gone by now; the re-check is performed by rebuilding the effective config
# (the confirmation lives inside that function), which is what keeps argv, the kv verdict and the
# EFFECTIVE diagnostic describing the same decision. A failure here latches the feature off instead
# of terminating, and the latch record is therefore written BEFORE the EFFECTIVE record.
# The rebuild and the record are one function precisely so that ordering cannot drift: the selftest
# drives this same function rather than a copy of its two halves.
function Complete-PreSpawnConfig {
    param($Catalog, $Profile, [string] $Root, [string] $OutputDir, [string] $ModelPath,
          $Overrides, $PrefetchDecision, [int] $Qd, $Sweep, [string] $QdSource, [bool] $Custom)
    $kvBefore = [string]$script:WarmstartCtx.status_text
    $config = Build-EffectiveConfig -Catalog $Catalog -Profile $Profile -Root $Root -OutputDir $OutputDir `
                  -ModelPath $ModelPath -Overrides $Overrides -PrefetchDecision $PrefetchDecision -Qd $Qd
    if ([string]$script:WarmstartCtx.status_text -cne $kvBefore) {
        Write-Line ('  kv               : {0} (re-checked before start)' -f $script:WarmstartCtx.status_text)
    }

    Write-Diag -Kind 'EFFECTIVE' -Data @{ argv = $config.argv; env = $config.env; port = $config.port
                                          budget_mb = $config.budget_mb; qd = $config.qd
                                          qd_source = $QdSource
                                          sweep_qd = $Sweep.qd; sweep_reason = $Sweep.reason
                                          sweep_from_binding = $Sweep.from_binding
                                          binding_persist = $(if ($Sweep.persist_failed) { 'failed' } else { 'ok' })
                                          effective_prefetch = $config.prefetch.echo
                                          # P4 3: the mandatory pre-seal echo, one field per
                                          # question. 'launcher_candidate_activation' is
                                          # deliberately not called an effective activation - the
                                          # engine seal owns that word.
                                          prefetch_request = [string]$config.prefetch.request
                                          catalog_evidence = [string]$config.prefetch.evidence
                                          catalog_activation = [string]$config.prefetch.activation
                                          launcher_candidate_activation = [string]$config.prefetch.candidate_activation
                                          prefetch_identity = [string]$config.prefetch.identity
                                          requested_k = $config.prefetch.k
                                          requested_n = $config.prefetch.n
                                          requested_qd = [int]$config.qd
                                          prefetch_provenance = $config.prefetch.provenance
                                          prefetch_init_version = [string]$config.prefetch.init_version
                                          # P4 3 names this field 'warning', not 'prefetch_warning':
                                          # the mandatory echo is a field-name contract, so the
                                          # record has to be readable by the name the spec gives.
                                          warning = $config.prefetch.warning
                                          off_reason = $config.prefetch.off_reason
                                          warmstart_mode = $script:WarmstartCtx.mode
                                          warmstart_override = $script:WarmstartCtx.override
                                          warmstart_state = (Get-WarmstartState)
                                          autosave = [string]$script:WarmstartCtx.autosave_setting
                                          autosave_active = (Test-AutosaveActive)
                                          autosave_minutes = [int]$script:WarmstartCtx.autosave_minutes
                                          # UX 1-2 / 1-4 / 1-5: the three new decisions this record
                                          # has to be able to explain - which arch-template value the
                                          # run resolved, whether the warmup dimension was on the
                                          # official cold condition, and who forced it if it was.
                                          arch_template = [string]$script:ArchTemplateResolved
                                          arch_template_source = [string]$script:ArchTemplateSource
                                          warmup = [string]$config.warmup
                                          warmup_forced_reason = [string]$config.warmup_forced_reason
                                          warm_path_baseline = (Test-WarmPathBaseline -Config $config)
                                          kv = $script:WarmstartCtx.status_text
                                          kv_reason = $script:WarmstartCtx.reason
                                          budget_source = $config.budget_source
                                          provenance = $(if ($Custom) { 'custom' } elseif ($config.budget_unmeasured) { 'auto' } else { 'catalog defaults' })
                                          format_gate = (Test-JsonBooleanTrue (Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'gates') -Name 'format_validated'))
                                          # UX 1-4 (Codex build r1 M4): the warm-path dimension counts
                                          # HERE too. This field is the record's conclusion and it
                                          # carries the same name as the screen's row, so it may not
                                          # answer 'true' while the screen prints [unmeasured]
                                          # (product warm-path baseline). One predicate, both writers.
                                          # The disjunction therefore carries EVERY demoting term the
                                          # screen ladder tests, in the ladder's own order (P4 2.5 pin
                                          # mismatch, custom, auto budget, warm path). Dropping a term
                                          # here republishes an unmeasured number as measured.
                                          performance_gate = $(if ($script:PinMismatchLatch -or $Custom -or $config.budget_unmeasured -or (Test-WarmPathBaseline -Config $config)) { 'unmeasured' } else { (Test-JsonBooleanTrue (Get-JsonValue -Obj (Get-JsonValue -Obj $Profile -Name 'gates') -Name 'performance_validated')) }) }
    return $config
}

function Invoke-LauncherMain {
    Initialize-DiagLog
    # [void] is load-bearing: this function's return value IS the status, so any stray pipeline
    # output would turn it into an array and lose the enum.
    [void](Install-CtrlHandler)

    # (0) launcher-owned CLI validation before anything else can fail without a status line
    Set-FailureStage 'fail_custom_args'
    Resolve-ExtraCliArgs
    # RV 3 / RV 2-4: the run mode is fixed before anything can read it, and the CLI half of the
    # pinned-shape refusal runs immediately after - while the raw -QD / -Prefetch strings still
    # exist. Both are launcher-owned CLI validation, so they belong in this step with -Action.
    Resolve-RepackMode
    Assert-VirtualCliPins
    # UX 1-1-2: the arch-template answer is latched HERE, at the top of the run. It has to be
    # decided before the model selection menu offers its toggle and long before the selection call
    # consumes it, and its only inputs (CLI, the global preference file) are all available now.
    Resolve-ArchTemplate

    # (1) bundle integrity comes first (LS 2 "launcher first action")
    Set-FailureStage 'fail_gate_bundle'
    $root = Resolve-BundleRoot
    Assert-BundleIntegrity -Root $root

    # (2) catalog
    Set-FailureStage 'fail_gate_catalog'
    $catalog = Read-Catalog -Root $root
    $sourceTag = [string](Get-JsonValue -Obj $catalog -Name 'source_tag')

    # (3) model + identification
    Set-FailureStage 'fail_model_path'
    $modelPath = Resolve-ModelPath -Catalog $catalog
    $modelSet = Get-ModelShardSet -ModelPath $modelPath

    # UX 1-1-5: a run that reached the model MENU was already offered the toggle. A run that did not
    # (-Model, or any fallback to the text prompt) gets the equivalent one-shot question here -
    # after the arch is known, which is what makes the "would this run take the template path"
    # scope check possible at all, and still before the selection call below closes the door.
    Confirm-ArchTemplateBeforeIdentify -Catalog $catalog -ModelSet $modelSet -Root $root

    # LS OA-1 (M1): the header fingerprint narrows the candidates, the source pin decides.
    # UX 1-1-3: admissibility is folded into that same argument. The arch is the shard-set consensus
    # Get-ModelShardSet already produced (no new header read), and a family miss closes the template
    # FALLBACK here - before the SHA cache, the output lock and the derive-plan can leave anything
    # on disk - while a catalog candidate still returns on its own merits further in.
    $selection = Resolve-ProfileSelection -Catalog $catalog -ModelSet $modelSet -Root $root `
                     -TemplateAllowed (Test-TemplateAdmissible -Arch ([string]$modelSet.arch))
    $derived = $null
    $profile = $selection.profile
    $outputDir = $null

    # LS OA-1 - Codex r1 F2: the reordering below is DERIVED-ONLY. That branch has to resolve the
    # output path and take the instance + output locks before its plan runs (the plan reads the
    # output volume, and it decides what the following repack will write there), and it cannot take
    # the profile lock any earlier because the derived profile id does not exist until the plan has
    # run. The catalog branch keeps the v0.4 order exactly - bounds, CLI overrides, PATHS, then
    # locks - so a run with an out-of-bounds override still terminates as fail_custom_args without
    # having created the output directory or the lock file.
    if ($selection.kind -ceq 'template') {
        # RV 0: the derived/template path parses the repacker's PLAN TEXT, and those regexes are
        # bin-only (ConvertFrom-TemplatePlanText :2586 / the derive plan :2889, against the
        # repacker's own template branch repack_experts.py:4145). A virtual plan prints a different
        # summary, so the combination cannot be consumed - support for it is outside this preview.
        # Refused HERE, before the derive-plan runs, so no output directory is created, no lock is
        # taken and no plan text is produced for a run that cannot finish.
        if (Test-VirtualRepack) {
            Set-FailureStage 'fail_custom_args'
            Stop-Launcher 'fail_custom_args' 'derived/template profiles do not support -RepackMode virtual in this preview'
        }
        $outputDir = Resolve-OutputDirectory -ModelPath $modelPath
        Set-FailureStage 'fail_instance_lock'
        Acquire-LauncherLocks -OutputDir $outputDir
        Set-FailureStage 'fail_model_path'
        # Steps 2..5 of the derive-plan. Writes nothing; the confirmation that gates the first write
        # is the same one the catalog path uses, further down.
        $derived = Invoke-DerivePlan -Catalog $catalog -Root $root -ModelPath $modelPath `
                       -OutputDir $outputDir -ModelSet $modelSet -Shas $selection.shas
        $profile = $derived.profile
        Add-ProfileLock -ProfileId ([string](Get-JsonValue -Obj $profile -Name 'profile_id'))
    }
    $profileId = [string](Get-JsonValue -Obj $profile -Name 'profile_id')

    # Two different digests, on purpose.
    #   $expectDigest binds STATE to this model+expectation (preset, probe record, sweep target key).
    #                 For a catalog profile that is the catalog's approved expect hash; for a derived
    #                 one it is the inventory digest, which is known at plan time and binds the
    #                 actual tensor set rather than a file's bytes.
    #   $expectPath   is where the seven-item gate's expect lives - the bundle for a catalog profile,
    #                 the repack output directory for a derived one (never the bundle expects dir).
    #   $lockId       is what the repacker writes into reference_lock: the profile id, or the
    #                 arch-template marker.
    if ($null -ne $derived) {
        $expectDigest = [string](Get-JsonValue -Obj (Get-JsonValue -Obj $profile -Name 'derivation') -Name 'inventory_sha256')
        $expectPath   = Get-DerivedExpectPath -OutputDir $outputDir
        $lockId       = [string]$derived.lock_id
    } else {
        $expectDigest = ([string](Get-JsonValue -Obj $profile -Name 'expect_sha256')).ToLowerInvariant()
        $expectPath   = Get-ExpectPath -Root $root -Catalog $catalog -Profile $profile
        $lockId       = $profileId
    }
    $expectSha = $expectDigest
    Write-Line ('[identify] profile {0} ({1} shard(s), {2} bytes)' -f $profileId, $modelSet.shards.Count, $modelSet.total_bytes)
    # LS 11 (UI-1 3-a): identify succeeded, so this path is worth offering next time. Cache only -
    # it is never read back as a gate input and a failure to write it is silent.
    Add-RecentModel -Path $modelPath

    $bounds = Get-JsonValue -Obj $profile -Name 'allowlist_bounds'
    Set-FailureStage 'fail_custom_args'
    $cliOverrides = Get-CliOverrides -Bounds $bounds

    # RV 2-4 (2): the preset half of the pinned-shape refusal. The v0.4 order reads the preset AFTER
    # the repack and the gate, which would let a stored qd be refused only once the run had already
    # cleaned up, planned and repacked - the side effects would land before the refusal. Virtual
    # therefore reads the preset HERE, after -ResetPreset has been honoured and before any of that,
    # and the read is kept so the merge point below reuses it rather than reading (and logging) the
    # same file twice. The packed path is untouched: nothing above runs for it.
    if (Test-VirtualRepack) {
        if ($ResetPreset) { Remove-UserPreset }
        $script:VirtualPresetEarly = Read-UserPreset -SourceTag $sourceTag -ProfileId $profileId `
                                         -ExpectDigest $expectSha -Bounds $bounds
        Assert-VirtualOverridePins -Overrides $script:VirtualPresetEarly.overrides -Origin 'the stored preset'
    }

    if ($null -eq $outputDir) { $outputDir = Resolve-OutputDirectory -ModelPath $modelPath }
    Write-Diag -Kind 'PATHS' -Data @{ model = $modelPath; out = $outputDir; bundle = $root; expect = $expectPath
                                      selection = $selection.kind; lock_id = $lockId
                                      repack_mode = [string]$script:RepackModeResolved }

    # (4) locks stage 1: instance + profile + output. The effective-port lock is deliberately NOT
    # taken here - the port is not final until preset and CLI overrides have been bound (R1-1).
    # The derived branch has already taken all three above (see the note there).
    if ($selection.kind -cne 'template') {
        Set-FailureStage 'fail_instance_lock'
        Acquire-LauncherLocks -ProfileId $profileId -OutputDir $outputDir
    }

    # (4b) LS 13-1 startup insertion point: warmstart mode + kv GC, straight after the exclusive
    # locks and before any eligibility decision. hard-OFF returns without touching the kv tree.
    Initialize-Warmstart -ProfileId $profileId -DerivedProfile ($null -ne $derived)

    if ($script:ActionResolved -eq 'stop' -and $NonInteractive) {
        Stop-Launcher 'cancelled_user' 'stop requested before start'
    }
    Assert-NotCancelledPreReady

    # (5) .partial handling (LS 2) / RV 1-1 [3] artifact lifecycle
    Set-FailureStage 'fail_gate_verify'
    $needRepack = $true
    $staleArtifacts = $false
    if (Test-VirtualRepack) {
        # RV 1-1 [3]: the whole virtual disposition table lives in one function, and the bin
        # .partial flow below is NOT run - it deletes manifest.json, which is the opposite verdict
        # to the virtual gate's "a bin artifact here is a hard stop". Stale-artifact replacement is
        # likewise not introduced: the engine's own source binding (:2186) is the terminal authority
        # on whether a plan still describes the model being served, and duplicating it here would
        # add a second, weaker opinion.
        # The ambient stage stays fail_gate_verify: every disposition that terminates in there names
        # its own status explicitly (fail_gate_verify, cancelled_user, fail_partial_cleanup), so the
        # ambient one only covers an UNEXPECTED fault - and a gate fault is not a user cancellation.
        $needRepack = Resolve-VirtualArtifactState -OutputDir $outputDir -ProfileId $lockId -ExpectSha $expectSha
    } else {
    # --- packed (v0.4) branch, deliberately left at its original indentation so the differential
    #     review sees an unchanged block rather than a re-indented one. Ends at "end packed branch".
    $pm = Get-PartialMarkerState -OutputDir $outputDir
    if ($pm.state -eq 'unknown') { Stop-Launcher 'fail_gate_verify' ('experts.bin.partial absence not provable - ' + $pm.reason) }
    if ($pm.state -eq 'present') {
        Set-FailureStage 'cancelled_user'
        Invoke-PartialCleanup -OutputDir $outputDir
        $needRepack = $true
    } else {
        $have = $true
        foreach ($n in @('experts.bin', 'manifest.json', 'verify_report.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $outputDir $n) -PathType Leaf)) { $have = $false }
        }
        $needRepack = (-not $have)
        # RV 1-1 [3], the OTHER direction of the mode-intent binding. A complete bin artifact set is
        # a v2 manifest by construction (one trusted producer wrote all three), so the detector is
        # only consulted where the set is INCOMPLETE and a manifest is nevertheless present - which
        # is what an explicit -OutDir pointing at a virtual output directory looks like. A healthy
        # packed directory never reaches it, so the no-flag path keeps its reuse verdict, its cost
        # and its failure surface unchanged (RV 4). Only the VIRTUAL verdict stops here: an
        # unreadable v2 manifest is hand-edit damage, outside the threat model, and is left to the
        # gate that already reports it.
        if (-not $have) {
            $mSt = Get-FileAbsenceState -Path (Join-Path $outputDir 'manifest.json')
            if ($mSt.state -eq 'present') {
                $existingMode = Get-ManifestMode -ManifestPath (Join-Path $outputDir 'manifest.json')
                if ([string]$existingMode.mode -ceq $script:MANIFEST_MODE_VIRTUAL) {
                    Stop-ModeMismatch -Existing $script:MANIFEST_MODE_VIRTUAL -Requested $script:REPACK_MODE_PACKED
                }
            }
        }
        # P4 2.5 (b): see Remove-StaleRepackArtifacts. A mismatch run may not reuse artifacts whose
        # binding to the CURRENT bytes nothing can establish - the manifest records no source digest.
        # Presence therefore stops being evidence on this path, and only on this path.
        if ($have -and ([string]$selection.kind -ceq 'mismatch')) {
            $staleArtifacts = $true
            $needRepack = $true
            Write-Line '[stale] This file is not the bytes the catalog profile pins, and the repack artifacts'
            Write-Line '        already in this directory cannot be shown to have been built from it.'
            Write-Diag -Kind 'STALE_ARTIFACTS_DETECTED' -Data @{ out = $outputDir; selection = [string]$selection.kind
                                                                 reason = 'identity_mismatch_no_source_binding' }
        }
    }
    }   # end packed branch

    $preliminaryBudget = [long](Get-JsonValue -Obj $profile -Name 'min_budget_mb')
    if ($cliOverrides.ContainsKey('budget_mb')) { $preliminaryBudget = [long]$cliOverrides['budget_mb'] }

    # (6) preflight pass 1 - the disk decision that gates the repack. The authoritative RAM/port
    # sizing runs again in step (10) once the effective config exists (R1-1).
    # LS OA-1 step 6: this is also the derived path's resource gate, and it runs against the derived
    # minimum that the plan just produced.
    Set-FailureStage 'fail_resource'
    $expectedBytes = [long](-1)
    if ($null -ne $derived) { $expectedBytes = [long]$derived.expected_bytes }
    # RV 2-5: a virtual repack moves 0 bytes of expert data. Sizing it from the expect's
    # expert_bytes_total (~65 GiB on the preview's model) would be a reachable FALSE refusal, so the
    # requirement is the reservation constant instead - the output is manifest.json + plan_report.json.
    elseif (Test-VirtualRepack) { $expectedBytes = [long]$script:VIRTUAL_REPACK_RESERVE_MB * 1MB }
    # r3 C-1: a stale replacement frees what it is about to overwrite. Measured here, BEFORE the
    # gate and long before the deletion, and only on the path that will actually delete something.
    $reclaimableMb = [long]0
    if ($staleArtifacts) {
        # r4: Floor, not the rounding a bare [long] cast performs (see Get-StaleArtifactReclaimMb).
        $reclaimableMb = Get-StaleArtifactReclaimMb -OutputDir $outputDir
        Write-Diag -Kind 'STALE_ARTIFACT_RECLAIM' -Data @{ out = $outputDir; reclaimable_mb = $reclaimableMb }
    }
    $pre = Invoke-Preflight -OutputDir $outputDir -ExpectPath $expectPath -BudgetMb $preliminaryBudget `
               -NeedsRepack $needRepack -ExpectedBytes $expectedBytes -ReclaimableMb $reclaimableMb

    # (7) plan + explicit confirmation, then probe + repack (first run only)
    $probe = $null
    if ($needRepack -or $Plan) {
        Set-FailureStage 'fail_repack'
        Write-Line ''
        Write-Line '=== repack plan (--plan) ==='
        if ($null -ne $derived) {
            # The derived path already ran this exact plan, in arch-template mode, to obtain the
            # slot geometry. Running it a second time would re-parse every shard header for an
            # answer already in hand (26 s on the six-shard 397B set), so the captured text is
            # printed instead - the user sees the same plan the derivation was built from.
            Write-Line $derived.plan_text
        } else {
            # LS 11-6-b (UI-4): stage start line. The child writes nothing until it has parsed the
            # headers, so without this the section header alone would sit there looking stalled.
            Write-Line '[plan] running the repacker in --plan mode (header analysis, writes 0 bytes)...'
            # R1-12: Invoke-Repacker fails closed on a non-zero / timed-out / unspawnable plan run,
            # so the confirmation below is only ever reached with a complete plan.
            $planRes = Invoke-Repacker -Catalog $catalog -Root $root -Profile $profile -ModelPath $modelPath -OutputDir $outputDir -PlanOnly $true
            if ($planRes.text) { Write-Line $planRes.text }
        }
        Write-Line ('  repack cache directory : {0}' -f $outputDir)
        Write-Line ('  free space on volume   : {0} MB' -f $pre.disk.free_mb)
        Write-Line '  v1 has no resume: an interrupted repack restarts from the beginning.'
        # P4 2.5 (b): the confirmation below is the ONLY consent point, so what it consents to is
        # stated here - the artifacts of whatever was repacked in this directory before will be
        # deleted. Declining leaves them untouched (cancelled_user).
        if ($staleArtifacts) {
            Write-Line '  ---------------------------------------------------------------'
            # r3 C-2: state the epistemic position, not a fact about provenance. After an accepted
            # replacement the artifacts here WERE built from this file - the launcher simply has no
            # way to know that (the manifest records no source digest), so it treats them as stale
            # again. "was not built from this file" would be a false claim on exactly that cycle.
            Write-Line '  STALE ARTIFACTS: this directory already holds a repack, and nothing here can bind it'
            Write-Line '  to this file (the catalog pin does not match these bytes, and the manifest records no'
            Write-Line '  source digest). It may or may not have come from this file - that cannot be proven,'
            Write-Line '  so it is treated as stale. Proceeding DELETES:'
            foreach ($n in $script:PARTIAL_DELETE_SET) { Write-Line ('    - ' + (Join-Path $outputDir $n)) }
            Write-Line '  Answer N and re-run with -OutDir <other path> to keep them.'
            Write-Line '  ---------------------------------------------------------------'
        }
        if ($null -ne $derived) {
            Write-Line ('  {0}' -f $script:TEMPLATE_COPY_SENTENCE)
            Write-Line '  This model is EXPERIMENTAL: no published measurement covers it.'
        }
    }
    if ($needRepack) {
        Set-FailureStage 'cancelled_user'
        Assert-NotCancelledPreReady
        # LS 11-7 a: this y/N sits right after the longest pre-repack silence (identify + preflight
        # + --plan, ~1 minute) - a stale 'y'+Enter typed into that silence must not approve a repack
        # nobody confirmed. Same no-op contract with stdin redirected.
        $null = Clear-ConsoleInputQueue
        if (-not (Confirm-User -Question 'Proceed with the repack now? [y/N] ')) {
            Stop-Launcher 'cancelled_user' 'user declined the repack plan'
        }
        Set-FailureStage 'fail_partial_cleanup'
        # P4 2.5 (b): after consent, before the repacker - repack_experts.py:1030 aborts while
        # experts.bin / manifest.json survive without --force, so the stale set is cleared here.
        if ($staleArtifacts) { Remove-StaleRepackArtifacts -OutputDir $outputDir }
        Set-FailureStage 'fail_repack'
        $probe = Invoke-StartupProbe -OutputDir $outputDir
        Write-ProbeBinding -SourceTag $sourceTag -ProfileId $profileId -ExpectDigest $expectSha -OutputDir $outputDir -Result $probe
        # LS OA-1 step 6: the derived expect is written HERE, atomically, by the repacker itself
        # (repack_experts.py:1509-1512) - after the resource gate above and after the confirmation.
        Invoke-Repacker -Catalog $catalog -Root $root -Profile $profile -ModelPath $modelPath `
            -OutputDir $outputDir -PlanOnly $false -ArchTemplate ($null -ne $derived) | Out-Null
    } else {
        # LS 12-1: a later run skips ONLY the repack. The stored scratch record is still read, but
        # from here it is a diagnostic: since LS 12 the QD authority is the sweep in step (8b), and
        # LS 12-1 states explicitly that a failed scratch tier plus a successful sweep recovers.
        $bound = Read-ProbeBinding -SourceTag $sourceTag -ProfileId $profileId -ExpectDigest $expectSha -OutputDir $outputDir
        if ($bound.ok) {
            $probe = @{ ok = $true; mibps = $bound.mibps; qd_source = $bound.qd_source; provisional = $true }
            Write-Line ('[probe] scratch sanity record from an earlier run ({0} MiB/s, provisional)' -f $bound.mibps)
        } else {
            $probe = $null
            Write-Line ('[probe] no scratch sanity record ({0}); the QD sweep below is the measurement authority.' -f $bound.reason)
        }
        Write-Diag -Kind 'PROBE_SCRATCH_RECORD' -Data $bound
    }

    # (8) 7-item verify gate
    Set-FailureStage 'fail_gate_verify'
    # Gate item 6 compares reference_lock three ways, so it needs the two values the REPACKER wrote:
    # the lock id and the hash of the expect it locked against. For a catalog profile both come from
    # the catalog. For a derived profile the lock id is the arch-template marker and the hash is the
    # real bytes of derived.expect.json in the output directory - re-hashed here rather than trusted
    # from the plan, so a file that changed after the repack cannot pass.
    $gateExpectSha = $expectSha
    if ($null -ne $derived) {
        $dh = Get-FileSha256Lower -Path $expectPath
        if (-not $dh.ok) { Stop-Launcher 'fail_gate_verify' ('the derived expect could not be hashed - ' + $dh.reason) }
        $gateExpectSha = $dh.sha
    }
    # RV 1-1 [6]: after the artifacts exist the mode is detected AGAIN, and the gate is chosen from
    # that answer rather than from the request - a repacker that produced the wrong shape must not
    # be met by the gate that would accept it. Only the virtual path re-detects: a bin run's v2
    # manifest is produced by the same trusted tool that wrote its verify report, and adding a
    # second detection there would put a new failure on the packed success path for no gain (RV 4).
    $gateInfo = $null
    if (Test-VirtualRepack) {
        $producedMode = Get-ManifestMode -ManifestPath (Join-Path $outputDir 'manifest.json')
        if ([string]$producedMode.mode -ceq $script:MANIFEST_MODE_UNRECOGNIZED) { Stop-UnrecognizedManifest -ModeResult $producedMode }
        if ([string]$producedMode.mode -cne $script:MANIFEST_MODE_VIRTUAL) {
            Stop-ModeMismatch -Existing ([string]$producedMode.mode) -Requested $script:REPACK_MODE_VIRTUAL
        }
        $gateInfo = Assert-VirtualPlanGate -OutputDir $outputDir -ProfileId $lockId -ExpectSha $gateExpectSha
    } else {
    $gateInfo = Assert-VerifyGate -OutputDir $outputDir -ProfileId $lockId -ExpectSha $gateExpectSha
    }
    # WARMSTART A-4: the sidecar binding axes. The repack axis comes from the gate that just ran
    # (manifest.json's real bytes), the engine axis from the bundle manifest the integrity gate
    # already verified, and the model axis from the identified shard set.
    Set-WarmstartBindings -Root $root -ManifestSha $gateInfo.manifest_sha256 -ModelSet $modelSet

    # -----------------------------------------------------------------------------------------
    # (8b) P4 2 execution order. The v0.4 order swept first and read the preset afterwards, which
    # cannot work once the opt-in lives in the preset: the arm the sweep prefers is decided BY the
    # opt-in. The order is therefore:
    #   preset reset/read -> preset < CLI merge -> opt-in normalisation -> arm selection ->
    #   sweep (S90 / q_base with that arm) -> QD override -> K/N at the FINAL QD ->
    #   Resolve-PrefetchForQd invariant re-check (inside Build-EffectiveConfig, on every rebuild)
    # Every refusal (semantic, derived t, identity, hold, engine floor, t range, adapt) lands on
    # arm 'none' BEFORE the sweep runs, so a refused request cannot move the QD of the run it was
    # refused for.
    # -----------------------------------------------------------------------------------------
    Set-FailureStage 'fail_custom_args'
    # RV 2-4 (2): virtual already read the preset - and already honoured -ResetPreset - before the
    # cleanup/preflight/plan/repack sequence, so that a stored qd could be refused before any of
    # those had happened. Reusing that read here keeps it at one read and one log line per run; the
    # packed path takes the v0.4 branch unchanged.
    $preset = $script:VirtualPresetEarly
    if ($null -eq $preset) {
    if ($ResetPreset) { Remove-UserPreset }
    $preset = Read-UserPreset -SourceTag $sourceTag -ProfileId $profileId -ExpectDigest $expectSha -Bounds $bounds
    }
    $overrides = @{}
    foreach ($k in $preset.overrides.Keys) { $overrides[$k] = $preset.overrides[$k] }
    foreach ($k in $cliOverrides.Keys) { $overrides[$k] = $cliOverrides[$k] }

    Set-FailureStage 'fail_gate_catalog'
    $prefetchRequest = $script:PREFETCH_REQUEST_DEFAULT
    if ($overrides.ContainsKey('prefetch')) { $prefetchRequest = [string]$overrides['prefetch'] }
    $prefetchOptIn = ConvertTo-PrefetchOptIn -Request $prefetchRequest
    # PI 3 / P4 4: -Repro and -Smoke are the two flags that make a run a benchmark or reproduction
    # run, and P4 5 refuses adapt on those with its own reason.
    $reproOrBench = ([bool]$Repro -or [bool]$Smoke)
    # P4 3: the derived path is the only one where the profile's t did not come from the GGUF header,
    # so it is the only one that needs the cross-check. The catalog path's structural prefilter has
    # already compared identify.n_expert_used against the header.
    $derivedHeaderT = $null
    if ($null -ne $derived) { $derivedHeaderT = Get-ArchMetaLong -ModelSet $modelSet -Suffix '.expert_used_count' }
    $prefetchArm = Resolve-PrefetchArm -Profile $profile -OptIn $prefetchOptIn `
                       -IdentityVerdict ([string]$selection.kind) `
                       -DerivedHeaderExpertUsed $derivedHeaderT -ReproOrBench $reproOrBench
    Write-Diag -Kind 'PREFETCH_ARM' -Data @{ request = $prefetchRequest; opt_in = $prefetchOptIn
                                              arm = $prefetchArm; identity = [string]$selection.kind
                                              repro_or_bench = $reproOrBench }

    # LS 12 QD sweep - the single measurement authority for the automatic QD default. It runs after
    # the verify gate, on the sealed experts.bin, read-only. Every failure inside is non-terminal
    # (degraded QD1 / conservative default, RS 5).
    Set-FailureStage 'fail_gate_verify'
    # RV 2-4: virtual has no experts.bin to sweep, and a swept-and-failed verdict is NOT the same
    # as "not swept" - the first turns the catalog prefetch row off. See New-VirtualPinnedQd.
    if (Test-VirtualRepack) {
        $sweep = New-VirtualPinnedQd
    } else {
    $sweep = Resolve-QdSweep -OutputDir $outputDir -SourceTag $sourceTag -ProfileId $profileId `
                 -ExpectDigest $expectSha -ManifestSha256 $gateInfo.manifest_sha256 `
                 -CheckedAt $gateInfo.checked_at -Profile $profile -PrefetchArm $prefetchArm
    }

    # (9) prefetch decision at the FINAL effective QD + effective config
    Set-FailureStage 'fail_gate_catalog'
    $qd = [int]$sweep.qd
    if (-not $sweep.ok) { $qd = [int]$script:QD_DEGRADED }
    $prefetchDecision = Resolve-EffectivePrefetch -Profile $profile -ProbeOk ([bool]$sweep.ok) `
                            -OptIn $prefetchOptIn -EffectiveQd (Get-EffectiveQd -Overrides $overrides -Qd $qd) `
                            -IdentityVerdict ([string]$selection.kind) `
                            -DerivedHeaderExpertUsed $derivedHeaderT -ReproOrBench $reproOrBench `
                            -Request $prefetchRequest
    # P4 4: a request that changed nothing may not demote the published numbers (see
    # Test-CustomProvenance). This is the only writer of that latch.
    $script:PrefetchRequestIgnored = ($prefetchOptIn -ceq $script:PREFETCH_ARM_INIT -and
                                      [string]$prefetchDecision.candidate_activation -ceq 'catalog-fixed')

    Set-FailureStage 'fail_custom_args'
    $config = Build-EffectiveConfig -Catalog $catalog -Profile $profile -Root $root -OutputDir $outputDir `
                  -ModelPath $modelPath -Overrides $overrides -PrefetchDecision $prefetchDecision -Qd $qd
    $custom = Test-CustomProvenance -Overrides $overrides
    $qdSource = Get-QdSource -Overrides $overrides -Sweep $sweep

    # (10) LS 1-7 step 7: re-run sizing against the EFFECTIVE values, then reserve the effective
    # port. A stored preset carrying a larger budget, a larger ctx or a different port must not be
    # able to bypass the sizing and the lock that ran on the catalog defaults (R1-1).
    $pre = Confirm-EffectiveSizing -OutputDir $outputDir -ExpectPath $expectPath -Config $config -Profile $profile

    # (11) status + 3-choice loop
    # LS OA-1: the three surface axes are fixed for the whole loop - the verify gate above has
    # already decided copy integrity, and neither the inventory authority nor the serving validation
    # can be changed by a custom edit.
    $axes = Get-SurfaceAxes -Kind ([string]$selection.kind) -Profile $profile -CopyVerified $true
    Write-Diag -Kind 'SURFACE_AXES' -Data $axes
    while ($true) {
        Show-Status -Profile $profile -Config $config -ProbeResult $probe -Custom $custom -RamVerdict $pre.ram `
                    -Sweep $sweep -QdSource $qdSource -SurfaceAxes $axes
        $choice = Read-MenuChoice
        if ($choice -eq 'stop') { Stop-Launcher 'cancelled_user' 'user selected stop before start' }
        if ($choice -eq 'custom') {
            $overrides = Invoke-CustomEditor -Overrides $overrides -Bounds $bounds
            Set-FailureStage 'fail_custom_args'
            # RV 2-4 (3): the third QD request path. Refused the moment the editor returns - before
            # the prefetch decision is re-resolved, before the config is rebuilt and before the
            # preset is saved - so a pinned-shape violation can neither reach an output nor be
            # written back for the next run to inherit.
            Assert-VirtualOverridePins -Overrides $overrides -Origin 'the custom editor'
            $custom = Test-CustomProvenance -Overrides $overrides
            # P4 2: the custom editor cannot change the prefetch REQUEST (it is not offered there and
            # the sweep is never re-run), but it can change the QD - and the init arm's N0 is derived
            # from the final QD. So the decision is re-resolved at the new QD on the same inputs; the
            # arm, the identity verdict and the request are all unchanged by construction.
            $prefetchDecision = Resolve-EffectivePrefetch -Profile $profile -ProbeOk ([bool]$sweep.ok) `
                                    -OptIn $prefetchOptIn -EffectiveQd (Get-EffectiveQd -Overrides $overrides -Qd $qd) `
                                    -IdentityVerdict ([string]$selection.kind) `
                                    -DerivedHeaderExpertUsed $derivedHeaderT -ReproOrBench $reproOrBench `
                                    -Request $prefetchRequest
            $config = Build-EffectiveConfig -Catalog $catalog -Profile $profile -Root $root -OutputDir $outputDir `
                          -ModelPath $modelPath -Overrides $overrides -PrefetchDecision $prefetchDecision -Qd $qd
            # LS 12-1: a custom edit can only move the QD priority between user-override and the
            # measured default - it never re-runs the sweep (one sweep per process, LS 12-4).
            $qdSource = Get-QdSource -Overrides $overrides -Sweep $sweep
            # every custom edit re-runs the same effective sizing + port reservation
            $pre = Confirm-EffectiveSizing -OutputDir $outputDir -ExpectPath $expectPath -Config $config -Profile $profile
            [void](Save-UserPreset -SourceTag $sourceTag -ProfileId $profileId -ExpectDigest $expectSha -Overrides $overrides)
            continue
        }
        break
    }
    Assert-NotCancelledPreReady

    $config = Complete-PreSpawnConfig -Catalog $catalog -Profile $profile -Root $root -OutputDir $outputDir `
                  -ModelPath $modelPath -Overrides $overrides -PrefetchDecision $prefetchDecision -Qd $qd `
                  -Sweep $sweep -QdSource $qdSource -Custom $custom

    # (11) start the server child
    Set-FailureStage 'fail_server_start'
    $rt = Get-JsonValue -Obj $catalog -Name 'runtime'
    $serverExe = Join-Path $root ([string](Get-JsonValue -Obj $rt -Name 'server_exe'))
    $logDir = Join-Path (Get-LauncherStateDir) 'logs'
    if (-not (Test-Path -LiteralPath $logDir -PathType Container)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $srvOut = Join-Path $logDir ("server_{0}_out.log" -f $stamp)
    $srvErr = Join-Path $logDir ("server_{0}_err.log" -f $stamp)

    $script:LastServerPort = [int]$config.port
    $script:LastServerConfig = $config
    $sr = Start-OwnedChild -Exe $serverExe -Args0 $config.argv -EnvVars $config.env -WorkDir $root `
              -StdOutPath $srvOut -StdErrPath $srvErr -NewProcessGroup $true -Role 'server'
    if (-not $sr.ok) { Stop-Launcher 'fail_server_start' ('server start failed: ' + $sr.reason) }
    $child = $sr.child
    Write-Line ''
    Write-Line ('[start] server pid {0}; waiting for health on http://{1}:{2}/health' -f $child.pid, $config.host, $config.port)

    Wait-ForServerReady -Child $child -Config $config -ErrLog $srvErr
    $script:ChildWasReady = $true
    Write-Line ('[ready] server is ready on http://{0}:{1}' -f $config.host, $config.port)
    Write-Diag -Kind 'READY' -Data @{ pid = $child.pid; port = $config.port }

    # (11b) LS 13-1 ready-side insertion point: restore, then the A-4b ladder. A successful restore
    # takes warmup ownership away from the launcher (A-3) - the generic warmup request would
    # otherwise overwrite the prefix that was just restored into slot 0.
    $wsRestore = Invoke-WarmstartRestore -Config $config
    if ($wsRestore.recovery) {
        $rec = Invoke-WarmstartRecoveryRestart -Config $config -ServerExe $serverExe -Root $root `
                   -StdOutPath $srvOut -StdErrPath $srvErr
        $child = $rec.child
        $config = $rec.config
        $srvErr = $rec.err_log
        $script:LastServerConfig = $config
        $wsRestore = @{ restored = $false; recovery = $true; n_restored = 0 }
    }

    # (12) RS 5 degraded branches: warmup and browser open are both best-effort and never terminal.
    Invoke-ReadyWarmup -Config $config -Restore $wsRestore
    Open-BrowserBestEffort -Config $config -Catalog $catalog

    # (13) smoke or interactive serve
    if ($Smoke) {
        Set-FailureStage 'fail_smoke'
        $sm = Invoke-SmokeChecklist -Config $config -Catalog $catalog -Child $child -GateInfo $gateInfo
        if (-not $sm.ok) { Stop-Launcher 'fail_smoke' ('smoke assertions failed: ' + ($sm.failures -join ', ')) }
        return 'ok_smoke'
    }

    Set-FailureStage 'fail_runtime_exit'
    Write-Line ''
    Write-Line 'Server is running. Press Ctrl+C or Ctrl+Break, or type "stop" + Enter, to stop it.'
    # UI-9: one echo state for the whole serving phase, shared by both loops below (only one of
    # them ever runs). disabled=$true after a fault is the permanent off switch for this run.
    $pfState = @{ next_line = 0; prev_n = $null; prev_p = $null; prev_task = $null; total_est = $null
                  disabled = $false; last_check = [datetime]::MinValue }
    if ($NonInteractive) {
        $until = (Get-Date).AddSeconds($script:RunSecondsResolved)
        while ((Get-Date) -lt $until) {
            # LS 1-8 (b): after ready a console stop request means "run the graceful stop", not
            # "cancel" - leaving the loop takes us into Complete-Teardown.
            if (Test-CancelRequested) { Write-Line '[stop] console stop request received.'; break }
            $ex = Test-ChildExited -Child $child
            if ($ex.exited) { Stop-Launcher 'fail_runtime_exit' ('server exited unexpectedly (code=' + (Format-ExitCode $ex.code) + ')') }
            Show-PrefillProgressTick -Child $child -State $pfState
            # LS 13-8: the autosave tick. Self-contained and below its deadline it is a cheap return.
            Invoke-AutosaveTick -Config $config
            Start-Sleep -Milliseconds 250
        }
        return 'ok'
    }
    while ($true) {
        if (Test-CancelRequested) { Write-Line '[stop] console stop request received.'; break }
        $ex = Test-ChildExited -Child $child
        if ($ex.exited) { Stop-Launcher 'fail_runtime_exit' ('server exited unexpectedly (code=' + (Format-ExitCode $ex.code) + ')') }
        Show-PrefillProgressTick -Child $child -State $pfState
        # LS 13-8: the same tick on the interactive path - one insertion per serving loop, no third
        # copy of the condition table.
        Invoke-AutosaveTick -Config $config
        # V-2: this poll must not block, or none of the three checks above it ever runs again on a
        # real console. See Get-ServeInputGate for the measurement and the branch table.
        $ri = Read-ServeCommandLine
        if ($ri.fault) { Start-Sleep -Milliseconds 500; continue }
        $line = $ri.line
        if ($null -ne $line) {
            if ($line -eq 'stop' -or $line -eq 'q' -or $line -eq 'quit') { break }
        }
        Start-Sleep -Milliseconds 250
    }
    return 'ok'
}

# ---- R1-10 degraded branches (RS 5: both are non-terminal, reason echoed) -------------------
# LS 13-1 / WARMSTART A-3: the ready-side warmup decision. It is a function of its own so that the
# rule "a successful restore owns the prefix, therefore the launcher does not warm up" is executed
# by the selftest instead of by a second copy of the condition (WARMFILE_DESIGN gate 1: restore
# success -> 0 POSTs, recovery-cold -> exactly 1 POST). Behaviour and the skip reason string are
# the v0.4 ones, unchanged.
function Invoke-ReadyWarmup {
    param($Config, $Restore)
    if ($Restore.restored) {
        Write-Diag -Kind 'WARMUP_SKIPPED' -Data @{ reason = 'slot state restored (WARMSTART A-3): a launcher warmup would overwrite the restored prefix' }
        return
    }
    Invoke-LauncherWarmup -Config $Config
}

# R2-2: engine-side warmup is off in EVERY configuration ('--no-warmup' is unconditionally forced
# into the effective argv by Build-EffectiveConfig). warmup=on therefore only turns on this
# post-ready launcher request, which is the only warmup that can fail non-terminally.
# WARMFILE_DESIGN v0.2 section 1: 'file:<path>' is a THIRD mode of the same key and takes a separate
# branch - it must not be re-wrapped by the chat template, so it never reaches the generic request
# built below.
function Invoke-LauncherWarmup {
    param($Config)
    $wf = Get-WarmupFilePath -Value ([string]$Config.warmup)
    if ($null -ne $wf) { Invoke-LauncherWarmfile -Config $Config -RawPath $wf; return }
    if ($Config.warmup -ne 'on') {
        # UX 1-5: the skip reason is an ENUM now. 'RELEASE_SPEC 8 default' died with the default
        # reversal - it was already false for a bench run, and since v0.2.3 there is no default-off
        # case left at all. Two reasons remain and they are not interchangeable: 'forced_bench' is
        # the launcher protecting a measurement, 'user_off' is a choice the user made. Recorded HERE
        # and only here: Build-EffectiveConfig re-runs on every custom edit, so putting the record
        # there would log the same forcing several times per run.
        $reasonEnum = $script:WARMUP_SKIP_USER_OFF
        $reasonText = 'warmup off (user or stored preset)'
        if ($Config.warmup_forced_reason) {
            $reasonEnum = $script:WARMUP_SKIP_FORCED_BENCH
            $reasonText = [string]$Config.warmup_forced_reason
            # RS 5 lineage: a degraded/skipped branch states its reason on the console. Only the
            # forced case echoes - a run the user switched off does not need to be told twice.
            Write-Line ('[warmup] skipped: ' + $reasonText + ' [reason=' + $reasonEnum + ']')
        }
        Write-Diag -Kind 'WARMUP_SKIPPED' -Data @{ reason = $reasonText; reason_enum = $reasonEnum }
        return
    }
    $uri = ('http://{0}:{1}/v1/chat/completions' -f $Config.host, $Config.port)
    $body = '{"model":"local","messages":[{"role":"user","content":"warmup"}],"max_tokens":1,"stream":false}'
    $r = Invoke-HttpJson -Uri $uri -Method 'POST' -Body $body -TimeoutSec 300
    if ($r.ok -and $r.status -eq 200) {
        Write-Line '[warmup] launcher warmup request completed.'
        Write-Diag -Kind 'WARMUP_OK' -Data @{ status = $r.status }
        return
    }
    $reason = $r.reason
    if (-not $reason) { $reason = ('status ' + $r.status) }
    Write-Line ('[warmup] WARNING: warmup request failed (' + $reason + '); continuing without warmup (degraded, non-terminal).')
    Write-Diag -Kind 'warmup_failed' -Data @{ reason = $reason }
}

# WARMFILE_DESIGN v0.2 section 1 - every warmfile failure is degraded and non-terminal (RS 5
# lineage): missing file, empty file, context overflow, timeout and HTTP error all land here and
# none of them may end the run. The path lives in the diagnostic record only; the console line
# carries the reason.
function Write-WarmfileFailed {
    param([string] $Reason, [string] $File = $null, $Bytes = $null)
    Write-Line ('[warmup] WARNING: warmup file precompute failed (' + $Reason + '); continuing without warmup (degraded, non-terminal).')
    $data = [ordered]@{ reason = $Reason }
    if ($File) { $data['file'] = $File }
    if ($null -ne $Bytes) { $data['bytes'] = [long]$Bytes }
    Write-Diag -Kind 'warmup_failed' -Data $data
}

# JSON string serialiser for the warmfile request body. Deliberately NOT ConvertTo-KvJsonString:
# that one is the input to a stored binding hash, so the two must stay free to change independently.
# The escape rule is the same and is chosen for a second reason here - every character above ASCII
# 126 becomes \uXXXX, so the request body is pure ASCII and cannot be altered by whatever encoding
# Invoke-WebRequest picks for a string body on PS 5.1. Newlines, tabs and CR are covered by the same
# rule, which is what keeps a multi-line prompt file byte-exact on the wire.
function ConvertTo-WarmfileJsonString {
    param([string] $Value)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    foreach ($ch in ([string]$Value).ToCharArray()) {
        $c = [int][char]$ch
        if ($ch -eq '"')  { [void]$sb.Append('\"'); continue }
        if ($ch -eq '\')  { [void]$sb.Append('\\'); continue }
        if ($c -lt 32 -or $c -gt 126) { [void]$sb.Append('\u'); [void]$sb.Append($c.ToString('x4')); continue }
        [void]$sb.Append($ch)
    }
    [void]$sb.Append('"')
    return $sb.ToString()
}

# N is the SERVER's own count and is never estimated (WARMFILE_DESIGN v0.2 section 1).
# 1st sources, llama.cpp b10057:
#   tools/server/server.cpp:233            POST /completion -> routes.post_completions
#   tools/server/server-context.cpp:4731-4740  post_completions passes TASK_RESPONSE_TYPE_NONE,
#                                          i.e. the non-OAI result shape below
#   tools/server/server-task.cpp:364,373   to_json_non_oaicompat emits {"tokens_evaluated", n_prompt_tokens}
#   tools/server/server-context.cpp:2150   res->n_prompt_tokens = slot.task->n_tokens()
#                                          = the token count this prompt rendered into (cached + new)
#   tools/server/server-task.cpp:240-244   timings carry {"cache_n", cache_n} and {"prompt_n", prompt_n}
#   tools/server/server-context.cpp:555,557 cache_n = n_prompt_tokens_cache (reused),
#                                          prompt_n = n_prompt_tokens_processed (newly evaluated)
#   tools/server/tests/unit/test_completion.py:659  the engine's own test asserts
#                                          timings["prompt_n"] + timings["cache_n"] == n_prompt
# so tokens_evaluated is the primary field and prompt_n + cache_n is the same number.
# NOTE 'tokens_cached' is NOT part of any sum here (r1 F2 corrects the earlier N+1 wording):
#   server-context.cpp:2152  res->n_tokens_cached = slot.prompt.n_tokens()
# is the slot's WHOLE prompt residency, not a disjoint half of the prompt - only prompt_n / cache_n
# partition it. For this request shape it is exactly N, not N+1: with n_predict=1 the first sampled
# token trips the budget stop (server-context.cpp:400 has_budget, :1918 STOP_TYPE_LIMIT), so
# process_token returns false and send_final_response runs immediately (:3853-3856). The sampled
# token is only appended to slot.prompt by handle_last_sampled_token (:456, push_back at :488),
# which belongs to the NEXT decode iteration and is never reached. Adding tokens_cached to
# tokens_evaluated would therefore count the same N tokens twice.
function Get-WarmfileTokenCount {
    param([string] $Body)
    $j = ConvertFrom-JsonStrict -Text $Body
    if (-not $j.ok) { return $null }
    $te = Get-JsonValue -Obj $j.value -Name 'tokens_evaluated'
    if (Test-JsonNonNegativeInteger $te) { return [long]$te }
    $tm = Get-JsonValue -Obj $j.value -Name 'timings'
    $pn = Get-JsonValue -Obj $tm -Name 'prompt_n'
    $cn = Get-JsonValue -Obj $tm -Name 'cache_n'
    if ((Test-JsonNonNegativeInteger $pn) -and (Test-JsonNonNegativeInteger $cn)) { return ([long]$pn + [long]$cn) }
    return $null
}

# WARMFILE_DESIGN v0.2 section 1: one synchronous request, sent to /completion so the server
# tokenises the file text as given. The chat endpoint would wrap it in the model's chat template and
# the precomputed tokens would then not be a prefix of what the client's first request renders.
function Invoke-LauncherWarmfile {
    param($Config, [string] $RawPath)
    # R1-8 rule: the child runs with the bundle root as its working directory, so a relative path
    # given to the launcher is resolved HERE, against the launcher's own working directory.
    $file = [string]$RawPath
    try {
        if (-not [System.IO.Path]::IsPathRooted($file)) { $file = Join-Path (Get-Location).ProviderPath $file }
        $file = [System.IO.Path]::GetFullPath($file)
    } catch {
        Write-WarmfileFailed -Reason ('warmup file path is not usable: ' + $_.Exception.Message)
        return
    }
    $b = Read-FileBytesStrict -Path $file
    if (-not $b.ok) { Write-WarmfileFailed -Reason ('warmup file could not be read: ' + $b.reason) -File $file; return }
    $t = ConvertFrom-Utf8Strict -Bytes $b.bytes
    if (-not $t.ok) { Write-WarmfileFailed -Reason ('warmup file is not valid UTF-8: ' + $t.reason) -File $file -Bytes $b.bytes.Length; return }
    $text = [string]$t.text
    if ($text.Length -eq 0) {
        Write-WarmfileFailed -Reason 'warmup file has no text to precompute' -File $file -Bytes $b.bytes.Length
        return
    }

    $body = '{"prompt":' + (ConvertTo-WarmfileJsonString -Value $text) +
            ',"cache_prompt":true,"n_predict":1,"stream":false}'
    $uri = ('http://{0}:{1}/completion' -f $Config.host, $Config.port)
    # LS 11 UI-4 lineage (no silent window): this is a full cold prefill and can run for minutes, so
    # the stage announces itself before it blocks. Display only - the path is not on this line.
    Write-Line ('[warmup] precomputing the warmup file prefix ({0} bytes); this runs once and can take minutes.' -f $b.bytes.Length)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-HttpJson -Uri $uri -Method 'POST' -Body $body -TimeoutSec $script:WARMFILE_TIMEOUT_SEC
    $sw.Stop()
    if (-not ($r.ok -and $r.status -eq 200)) {
        $reason = $r.reason
        if (-not $reason) { $reason = ('status ' + $r.status) }
        Write-WarmfileFailed -Reason ('warmup file request failed: ' + $reason) -File $file -Bytes $b.bytes.Length
        return
    }
    $n = Get-WarmfileTokenCount -Body $r.body
    if ($null -eq $n) {
        Write-WarmfileFailed -Reason 'warmup file response carried no server token count (no value is estimated)' `
            -File $file -Bytes $b.bytes.Length
        return
    }
    # WARMFILE_DESIGN v0.2 section 1, fixed wording. The launcher is not a proxy and never sees the
    # client's own request, so it states what it precomputed and hands the verification to the user.
    # The path is deliberately absent from this line; it lives in the WARMFILE_OK record.
    Write-Line ('[warmup] Precomputed {0} tokens. The launcher cannot observe client reuse; check the first response timings.cache_n (expected close to {0} (tokenizer boundaries and cache checkpoints may re-evaluate a small tail)).' -f $n)
    Write-Diag -Kind 'WARMFILE_OK' -Data @{ n_tokens = [long]$n; elapsed_ms = [long]$sw.Elapsed.TotalMilliseconds
                                            file = $file; bytes = [long]$b.bytes.Length }
}

function Open-BrowserBestEffort {
    param($Config, $Catalog)
    $webui = Test-JsonBooleanTrue (Get-JsonValue -Obj (Get-JsonValue -Obj $Catalog -Name 'runtime') -Name 'webui')
    if (-not $webui) {
        Write-Diag -Kind 'BROWSER_SKIPPED' -Data @{ reason = 'API-only bundle (runtime.webui=false)' }
        return
    }
    if ($Smoke -or $NonInteractive) {
        Write-Diag -Kind 'BROWSER_SKIPPED' -Data @{ reason = 'non-interactive or smoke run' }
        return
    }
    $url = ('http://{0}:{1}/' -f $Config.host, $Config.port)
    try {
        Start-Process -FilePath $url -ErrorAction Stop | Out-Null
        Write-Line ('[ui] opened ' + $url)
        Write-Diag -Kind 'BROWSER_OPENED' -Data @{ url = $url }
    } catch {
        Write-Line ('[ui] WARNING: could not open a browser (' + $_.Exception.Message + '); the API stays available at ' + $url)
        Write-Diag -Kind 'browser_open_failed' -Data @{ url = $url; reason = $_.Exception.Message }
    }
}

function Complete-Teardown {
    param([string] $PendingStatus)
    $status = $PendingStatus
    # R1-5: teardown itself must never be able to lose the final status line. Everything here is
    # inside a guard; an internal teardown fault fails closed to fail_teardown.
    try {
        if ($null -ne $script:OwnedChild) {
            $child = $script:OwnedChild
            # LS 13-1 teardown insertion point: the save POST has to be answered BEFORE the stop
            # signal goes out (A-2 (2)), so it sits immediately ahead of Stop-OwnedChildGraceful.
            # The helper is fully self-contained - it never throws and never touches $status - so a
            # save problem stays degraded and cannot be promoted into fail_teardown by the catch
            # below. The teardown verdict further down is decided on its own facts only.
            Invoke-WarmstartSave -Child $child -PendingStatus $PendingStatus -Config $script:LastServerConfig
            $res = Stop-OwnedChildGraceful -Child $child -PortNumber ([int]$script:LastServerPort)
            Close-OwnedChildHandles
            # A-2 (8) recovery timing (2): a save whose response never arrived may still have had
            # its .tmp written; now that the child is joined it can go.
            Complete-WarmstartTeardownCleanup

            # R1-6 / judgement (b): LS 5 makes teardown failure unconditional and top priority.
            # Using the taskkill fallback, a failed CTRL_BREAK send, an exceeded grace period or a
            # non-zero child exit during stop is a teardown failure no matter what the pending
            # status was - the fallback never restores 'ok' and never preserves fail_smoke either.
            $teardownFailed = $false
            $why = ''
            if ($res.taskkill_used)      { $teardownFailed = $true; $why = 'taskkill fallback was used' }
            elseif ($res.ctrl_attempted -and -not $res.ctrl_sent) { $teardownFailed = $true; $why = 'CTRL_BREAK could not be delivered' }
            elseif ($res.grace_exceeded) { $teardownFailed = $true; $why = 'graceful grace period exceeded' }
            elseif ($res.stop_nonzero)   { $teardownFailed = $true; $why = 'child exited non-zero during stop' }
            elseif (-not $res.child_gone) { $teardownFailed = $true; $why = 'child process still alive' }

            if (-not $teardownFailed) {
                # The listener condition only applies to a child that actually owned the port; a
                # foreign listener the child never owned must not be attributed to our teardown.
                $requestedStop = ($PendingStatus -eq 'ok' -or $PendingStatus -eq 'ok_smoke')
                if ($script:ChildWasReady -and -not $res.listener_gone) {
                    $teardownFailed = $true; $why = 'port listener still present'
                }
                # LS 1-8 (d): exit 0 additionally requires a graceful exit code 0.
                if ((-not $teardownFailed) -and $requestedStop -and -not $res.graceful) {
                    $teardownFailed = $true; $why = 'stop did not end in a graceful exit 0'
                }
            }

            if ($teardownFailed) {
                Write-Line ('[teardown] FAILED: ' + $why)
                Write-Diag -Kind 'TEARDOWN_PROMOTED' -Data @{ from = $PendingStatus; reason = $why }
                $status = 'fail_teardown'
            } elseif ($res.pre_exited) {
                Write-Line '[teardown] child had already exited; no cleanup action required.'
            } else {
                Write-Line '[teardown] child stopped gracefully; port released.'
            }
        }
    } catch {
        Write-Line ('[teardown] FAILED: internal teardown fault - ' + $_.Exception.Message)
        Write-Diag -Kind 'TEARDOWN_FAULT' -Data @{ reason = $_.Exception.Message }
        $status = 'fail_teardown'
    }
    # A-4b (5) recovery timing (3): this transaction's superseded generations, at the very end.
    # Also covers the case where the teardown above faulted before the first cleanup call.
    Complete-WarmstartTeardownCleanup
    try { Release-AllLocks } catch { }
    return $status
}

function Invoke-Launcher {
    $status = $null
    try {
        $status = Invoke-LauncherMain
    } catch {
        $ex = $_.Exception
        if ($null -ne $ex -and $ex.GetType().FullName -eq 'MoeLauncher.LauncherExit') {
            $status = [string]$ex.Status
            if ($ex.Message) { Write-Line ('[error] ' + $ex.Message) }
        } else {
            $msg = 'internal error'
            if ($null -ne $ex) { $msg = $ex.Message }
            Write-Line ('[error] ' + $msg)
            Write-Diag -Kind 'INTERNAL_ERROR' -Data @{ message = $msg; stage = $script:FailureStage
                                                       trace = [string]$_.ScriptStackTrace }
            $status = $script:FailureStage
        }
    }
    # R1-5: teardown runs inside its own guard. Complete-Teardown already fails closed internally;
    # this second guard makes sure that even a fault in that guard still yields an enum, so the
    # single "[moe-launcher] status=" line can never be lost.
    try {
        $status = Complete-Teardown -PendingStatus $status
    } catch {
        Write-Diag -Kind 'TEARDOWN_FAULT_OUTER' -Data @{ reason = $_.Exception.Message }
        $status = 'fail_teardown'
    }
    # Defence in depth: if anything ever leaks extra pipeline output into the status, take the last
    # emitted value rather than silently degrading a good run to fail_teardown.
    if ($status -is [System.Array]) {
        Write-Diag -Kind 'STATUS_MULTIVALUE' -Data @{ values = @($status | ForEach-Object { [string]$_ }) }
        $status = @($status)[@($status).Count - 1]
    }
    if ($null -eq $status -or -not $script:STATUS_EXIT.Contains($status)) { $status = 'fail_teardown' }
    return $status
}

# endregion

# ============================================================================
# region 20. ENTRY POINT
# ============================================================================

if (-not $LibraryMode) {
    $final = Invoke-Launcher
    Write-StatusHint -Status $final
    Write-StatusLine -Status $final
    exit (Get-StatusExitCode -Status $final)
}

# endregion
