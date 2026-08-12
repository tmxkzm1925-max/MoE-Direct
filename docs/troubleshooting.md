# Troubleshooting and status codes

[Back to the README](../README.md)

## Reporting problems

Every run ends with one machine-readable line on stderr - `[moe-launcher] status=<enum>` - and, for
failures, two human-readable lines just above it saying what happened and which section of this
document to read.

Diagnostic files, all local:

| File | Contents |
|---|---|
| `%LOCALAPPDATA%\MoE-Direct\logs\launcher_<timestamp>_<pid>.jsonl` | The launcher's own timeline: preflight, probe results, applied arguments, gate decisions, child start, teardown. **This is the file to attach.** |
| `%LOCALAPPDATA%\MoE-Direct\logs\server_<timestamp>_err.log` / `_out.log` | The server's own output. Attach when the failure is `fail_server_start`, `fail_runtime_exit` or `fail_gate_engine_seal`. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_log.jsonl` | The repacker's record. Attach for repack failures. |
| `%LOCALAPPDATA%\MoE-Direct\logs\metrics_<timestamp>.jsonl` | Cache accounting for each serving session: hit/miss counts, bytes read, wait times, slot and queue-depth configuration. **No prompts, no responses, no token content - you can open it in a text editor and check.** This is the file that makes a performance report (issue form 3) actionable. A few hundred KB per session; safe to delete like any other log. |

Four issue forms exist so that the first reply is not "please send more information":

1. **Blocked before it started** - no console window, no status line, SmartScreen/Smart App
   Control/antivirus intervened. There is no log to attach in this case, which is exactly why it
   has its own form.
2. **The launcher printed a status and stopped** - pick the status from the list, attach the
   launcher JSONL.
3. **Performance report** - your hardware, the launcher's probe/sweep output and its estimate, and
   what you actually got. These build the community measurement set.
4. **Anything else**, including a request to support a GGUF that is not in the catalog.

GitHub's upload box does not accept `.jsonl` directly - put the logs in a `.zip` first. Please
skim the JSONL before attaching: it records local file paths, which usually include your Windows
user name.

**Security issues do not go in public issues** - use the private route described in `SECURITY.md`.

## Troubleshooting

Each status below is a heading, so the launcher's `see : README.md > Troubleshooting > <status>`
line corresponds to `README.md#<status>`.

Exit codes, for scripting: `0` clean stop, `2` you cancelled, `3` path/resource/repack preparation
failed, `4` a policy gate refused, `5` server start, runtime or shutdown failure, `6` smoke check
failed with a clean shutdown.

### ok

Exit `0`. The server ran and was shut down cleanly: the child process exited on its own signal, no
force-kill was needed, and the listening port is gone. Nothing to do.

### ok_smoke

Exit `0`. The `-Smoke` self-check passed end to end and the server was shut down cleanly. Nothing
to do.

### cancelled_user

Exit `2`. You cancelled before the server became ready. Two cases, and they leave different things
behind:

- **Cancelled at a prompt** - you chose `stop`, pressed Ctrl+C, or declined the repack plan or the
  deletion of a leftover partial repack. **Nothing was started and nothing was deleted.**
- **Cancelled while the repacker was running** - the same status, but the repack was already under
  way, so the interrupted output stays on disk as a `.partial` file. There is no resume: the next
  run detects the leftover, tells you, and asks before deleting it and starting over.

Not an error either way; run it again when you are ready.

### fail_model_path

Exit `3`. *The model path could not be used: the file is missing, the GGUF is not one of the
supported profiles, or the shard set is incomplete/ambiguous.*

What to do:

- Check the path, including that the file finished downloading.
- For a multi-shard model, all shards must be in the same folder and be one complete set - a
  partial download or two different quantizations mixed in one folder both land here.
- If your model genuinely is not in [Supported models](models.md#supported-models), that is expected: the
  launcher refuses to write hundreds of GB for a layout it has never verified. Use issue form 4 to
  request the profile, and include the model's repo, revision and quantization.

### fail_resource

Exit `3`. *Preflight stopped the run: not enough RAM or disk space for this configuration.*

The message states which one and by how much. Free the space (the repack needs roughly the model's
size again, plus reserve) or lower the cache budget through the custom path - but note the profile's
minimum budget, below which the model cannot be served at all. Repacking onto the OS volume when it
is nearly full is refused on purpose.

### fail_instance_lock

Exit `3`. *Another launcher instance holds the single-instance mutex, or a profile/output/port lock
is already taken.*

This status is only about lock acquisition - the launcher-wide mutex, or the exclusive lock on this
profile / output folder / port combination. Close the other MoE-Direct window. If none is open, a
previous run may still be exiting - wait a few seconds and retry. (A server left listening on the
port from an earlier crash is a different failure: it lands in
[`fail_server_start`](#fail_server_start).)

### fail_partial_cleanup

Exit `3`. *The leftover repack outputs could not be deleted or confirmed absent.*

A previous repack was interrupted, you approved the cleanup, and the deletion failed. Usually
something else holds the files open (antivirus scan, an Explorer preview, another launcher). Close
those, or delete the `repack\` folder next to your GGUF by hand, then run again. The launcher never
deletes these without asking, and never resumes a partial repack.

### fail_repack

Exit `3`. *The repacker exited abnormally, or produced no verify report.*

Attach both `launcher_*.jsonl` and `repack_log.jsonl` (issue form 2). Common causes worth checking
first: the drive filled up mid-write, or the source GGUF is corrupt - re-verify the download's
checksum against the Hugging Face repo.

### fail_custom_args

Exit `3`. *A custom value failed the type or bounds check in non-interactive mode.*

Only reachable when the launcher is driven with arguments. The message names the offending value
and its allowed range. In interactive mode a bad value simply re-prompts instead of exiting.

### fail_gate_bundle

Exit `4`. *Bundle integrity check failed: the manifest, the schema, or the file set did not match
the sealed bundle.*

The extracted folder is not byte-identical to the released one. In order of likelihood:

- The zip was extracted over an existing folder, or files from two versions were mixed. Extract the
  release into a **new, empty** folder.
- Something was added, edited or removed inside the bundle folder - including files written there
  by another tool. Keep the bundle read-only in practice; the launcher writes its own state to
  `%LOCALAPPDATA%\MoE-Direct\`, never into the bundle.
- The download is damaged. Re-verify the zip's SHA-256 against `SHA256SUMS.txt`.

### fail_gate_catalog

Exit `4`. *`models.json` failed the catalog schema, the prefetch-state check, or the expect-digest
check.*

The catalog inside the bundle is not the released one. Re-extract from a fresh download. If you
edited `models.json` yourself, restore it - hand-edited catalogs are rejected by design, because
they are how a model would end up served under someone else's verified identity.

### fail_gate_verify

Exit `4`. *The 7-item repack gate rejected the verify report or its binding to the manifest.*

This is the gate that stands between you and serving unverified weights, so it fails closed on
anything it cannot positively confirm - a partial file present, a report that does not say `pass`,
counts that disagree, a non-empty problem list, an identity mismatch, or an unreadable file.

What to do: delete the `repack\` folder next to your GGUF and repack (the launcher will offer
this). If it fails a second time on the same model, that is worth an issue - attach both the
launcher JSONL and `verify_report.json`.

### fail_gate_engine_seal

Exit `4`. *The engine refused to start and printed its policy-gate reject line.*

The launcher's checks passed but the engine's independent checks did not - the two sides are
deliberately not allowed to trust each other. Typical cause: an argument or environment variable
that enables a lever on a profile it is not validated for (prefetch on a non-`validated` profile is
the usual one). If you set `MOE_DIRECT_*` variables yourself, clear them and retry with defaults.
Attach `server_*_err.log`.

### fail_server_start

Exit `5`. *The server never reached ready: process spawn, port, listener PID, health check, an
early exit, or CUDA out-of-memory.*

Check, in this order:

1. **`server_*_err.log`** - it usually says exactly what happened.
2. **Missing MSVC runtime.** A process that dies immediately with no useful output is the classic
   symptom; install the Microsoft Visual C++ Redistributable (x64) from Microsoft's
   [Latest supported Visual C++ Redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   page.
3. **Port already in use.** Change the port through the custom path.
4. **CUDA out of memory** - another program is holding VRAM (a game, another inference server, a
   browser doing GPU work). VRAM is checked and reported but not gated, because it can change
   between the check and the allocation.

### fail_runtime_exit

Exit `5`. *The server exited unexpectedly after it had reached ready.*

The interesting evidence is in `server_*_err.log` and the last requests you sent. Out-of-memory
under load, and a crash on an unusual request, both land here. Please report it with the log and,
if you can, the request that preceded it.

### fail_teardown

Exit `5`. *Shutdown did not complete cleanly: the signal, the grace period, or a surviving child
process or listener.*

This status takes priority over every other failure in the same run, on purpose: a run that left a
process behind is not a clean run, whatever else happened. **Check Task Manager for a surviving
`llama-server.exe` and end it** before starting again, or the next run will find the port still
taken and stop at [`fail_server_start`](#fail_server_start). Worth reporting - it means the ordinary
shutdown path did not work on your machine.

### fail_smoke

Exit `6`. *A smoke assertion failed, while the shutdown itself completed cleanly.*

Only reachable with `-Smoke`. The failing assertion is named in the output; attach the launcher
JSONL.
