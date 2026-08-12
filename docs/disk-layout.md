# Disk layout

[Back to the README](../README.md)

## What gets written to your disk

Transparency about files is cheap and surprises are expensive, so here is the complete list.
Everything below is local. Nothing is uploaded, nothing touches the registry, and every file is
either plain text (JSON/JSONL/log) you can open and read, or an output you explicitly approved.
Launcher state and logs are written from the moment the launcher starts; **no model or repack
output is created before you consent to the plan** (the one exception in the repack folder is a
tiny lock file, listed below, that marks the folder as claimed - it contains no model data).

| File | Written when | What it is for |
|---|---|---|
| the bundle folder | **never** | The bundle is read-only by design. The launcher verifies every file in it against the internal SHA manifest on each start and refuses to run on a mismatch. Its own state goes below - never into the bundle. |
| `%LOCALAPPDATA%\MoE-Direct\presets.user.json` | when you save a configuration | Saved launcher configurations, offered on the next run. |
| `%LOCALAPPDATA%\MoE-Direct\probe.state.json` | after the startup queue-depth sweep | Cached drive-sweep measurements per model/profile/volume, so the sweep does not rerun on every start. Re-measured when any of those change. |
| `%LOCALAPPDATA%\MoE-Direct\probe.scratch.json` | first run, right after you approve the repack | The provisional read-speed reading from the pre-repack sanity probe, bound to the same model/profile/volume. Diagnostic only since the startup sweep took over the queue-depth decision. |
| `.moe-probe.tmp` in the repack output folder | same probe, and only while no `experts.bin` exists there yet | A 64 MiB scratch file, written so the probe measures the volume the expert store will actually be read from, and **deleted as soon as the measurement ends**. |
| `.moe-launcher.lock` in the repack output folder | when the launcher takes its locks - before the repack plan is shown | The instance lock that stops two launchers from writing the same output folder. A few bytes, no model data; the file itself remains after exit (the lock is released) and is harmless. |
| `probe.scratch.json.tmp` / `probe.state.json.tmp` / `presets.user.json.tmp` in `%LOCALAPPDATA%\MoE-Direct\` | whenever the matching state file is updated | Scratch files for atomic replacement - written first, then renamed over the real file. Removed on success; one may survive a crash, and deleting it is always safe. |
| `%LOCALAPPDATA%\MoE-Direct\recent_models.json` | after model selection | The recent-models list behind the arrow-key picker. |
| `%LOCALAPPDATA%\MoE-Direct\logs\launcher_<timestamp>_<pid>.jsonl` | every run | The launcher's decision timeline: preflight, probe results, applied arguments, gate decisions, child start, teardown. Records local file paths, which usually include your Windows user name - skim before sharing. |
| `%LOCALAPPDATA%\MoE-Direct\logs\server_<timestamp>_err.log` / `_out.log` | every server start | The engine's own output. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_log.jsonl` | every repack | Append-only record of repack attempts and their outcomes. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_plan_<timestamp>_out.log` / `_err.log` | when the repack plan is computed - **before** you approve anything | The repacker's own output while it is only costing the job out. This is a log, not repack output: no model bytes are written at this step. |
| `%LOCALAPPDATA%\MoE-Direct\logs\repack_<timestamp>_out.log` / `_err.log` | every repack you approve | The repacker's own output during the repack itself, including the per-layer progress lines. |
| `%LOCALAPPDATA%\MoE-Direct\logs\metrics_<timestamp>.jsonl` | every serving session | Cache accounting: hit/miss counts, bytes read, wait times, slot and queue-depth configuration. **No prompts, no responses, no token content** - open it in a text editor and check. A few hundred KB per session. |
| `%LOCALAPPDATA%\MoE-Direct\kv\<profile id>\slot0.kv` and `slot0.kv.meta.json` | when you stop the server cleanly, with warm start on | The saved slot state, and the sidecar that binds it to your model, your repack, this bundle and the server configuration it was saved under. **The state file contains the session's tokens verbatim** - see [Warm start](warm-start.md#warm-start). Roughly 190 MB at a 12k context on the reference machine. |
| `%LOCALAPPDATA%\MoE-Direct\kv\<profile id>\slot0.auto.a.kv` / `slot0.auto.b.kv` and their `.meta.json` sidecars | on each autosave tick that fires | The two alternating crash-recovery generations. Same contents, same sensitivity, same folder. |
| `*.tmp.<generation>` / `*.stale.<generation>` in the same `kv\` folder | while a save is being written | A save is written under a temporary name and swapped in, so an interrupted save cannot damage the state you already had. Both are cleaned up at the next start; a cleanup that fails is recorded as a diagnostic and leaves the file in place. |
| `repack\` next to your GGUF (`experts.bin`, `manifest.json`, `verify_report.json`) | once per model, after you approve the plan | The direct-read expert store, its identity manifest, and the byte-level verification report that the serving gates consume - the repack output. Tens to hundreds of GB. An interrupted repack leaves `experts.bin.partial` here as well. **Never deleted automatically** - see [Update, reset, uninstall](#update-reset-uninstall). |

That is the whole list for the launcher flow. If you ever catch this project writing anywhere
else, that is a bug - please report it. (One advanced exception you create yourself: if you use
[start the server yourself](warm-start.md#starting-the-server-yourself), the slot files you request are
written to the directory you point `--slot-save-path` at.)

## Update, reset, uninstall

There is no installer, and nothing is written to the registry. Four things have separate
lifetimes, and you control all four:

| Thing | Where | What to do |
|---|---|---|
| The bundle | wherever you extracted the zip | **To update:** extract the new version into a *new* folder. Do not overwrite an existing bundle - the integrity manifest covers the folder as a set. Delete the old folder when you are done. |
| Launcher state and logs | `%LOCALAPPDATA%\MoE-Direct\` | **To reset settings:** delete `presets.user.json` (saved configurations), `probe.state.json` and `probe.scratch.json` (measured drive results), `recent_models.json` (the recent list). They are all rebuilt on the next run. Logs live in the `logs\` subfolder. |
| Saved session state | `%LOCALAPPDATA%\MoE-Direct\kv\` | **To delete it:** remove that folder, or the `<profile id>` subfolder for one model. Setting `warmstart` to `off` stops new saves but does not delete what is there. The launcher itself tries to hold at most four profiles, removing the least recently used beyond that, and records a diagnostic instead of failing the run when a removal does not succeed. |
| Repack output | a `repack\` folder next to each of your GGUFs | **Not deleted automatically, ever.** This is the big one - tens to hundreds of GB. Delete the `repack\` folder to reclaim the space; the next run for that model will repack from scratch. |

Uninstalling completely = delete all four. Your GGUFs are yours and are never touched.

## From the roadmap

| Where we are | Status |
|---|---|
| The repack that costs your disk the model's size a second time | **Being finished now, targeted at v0.3.** Instead of writing a second copy of the expert data, the launcher reads the experts out of your original file in place, so the space cost is the model and nothing more, and onboarding stops moving hundreds of gigabytes around. Nothing of it is in this build, and it ships only if it holds read performance - the rule this project works under is that space is never bought with speed. |

Next: [Troubleshooting and status codes](troubleshooting.md)
