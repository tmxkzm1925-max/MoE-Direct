# Warm start and prompt precompute

[Back to the README](../README.md)

## Warm start

Stopping the server used to throw the whole session away. The prefix cache went with the process, so
the next start had to prefill your prompt again from nothing, and on a long agent-style prompt that
is minutes of work you had already paid for. The launcher saves the slot state when you stop cleanly and
restores it the next time you start.

You do not have to do anything to get it. The saved state goes under
`%LOCALAPPDATA%\MoE-Direct\kv\<profile id>\`. One thing to expect on the first clean stop with a
given model: the integrity sidecar hashes the model file once, and on a very large model that is a
few minutes of reading with no output yet; the hash is cached, so later stops skip it. The status
screen tells you what it found before you press Enter:

```
  kv               : eligible
  autosave         : on (every 5 min)
```

`eligible` means a saved state passed every check and will be restored. `cold(<reason>)` means it
will not be, and names the reason. `off(user)` and `off(mode)` mean you turned it off, or the run is
a self-check or reproducibility run where it is off by design. Once the server is up a second line
reports what actually happened, `kv=restored(<n> tokens)` or `kv=cold(<reason>)`. Verifying a large
saved state takes a moment, and the launcher prints a line while it does that rather than going
quiet.

**Restoring the wrong state would be worse than starting cold, so it is not allowed to happen.** A
slot file written by llama.cpp carries no model hash, no vocabulary, no RoPE settings, no engine
identity and no checksum of its own; the engine validates its format, not whether it belongs to this
model. The launcher supplies what the engine cannot. Next to every saved state it writes a sidecar
recording the profile, the SHA-256 of every shard of your GGUF, the repack manifest hash, the bundle
hash, a canonical hash of the server configuration that affects state, the context size, the token
count, the byte count, and the SHA-256 of the saved file itself. All of it has to match before a
restore is attempted. Anything that does not match is a cold start with the mismatching field named,
not a gamble.

**Autosave, for the stops you did not plan.** Saving on a clean stop does nothing for a crash, a
power cut or a forced kill, so the launcher also saves while it is serving. The default is a five-minute
tick that fires only when two things hold together: nothing is in flight, and the token count has
actually changed since the last save. Writes alternate between two generations, so a crash during a
save damages only the generation being written and the previous complete one survives. On the next
start the most recent state that passes the checks is restored, whether it came from a clean stop or
from a tick.

**What it does not do.** A slot file holds tokens and cache state, not the server's own prefix
checkpoints. A request that extends the restored prompt exactly reuses it; a request that diverges
from it is reprocessed in full on hybrid-attention models such as Qwen3.5. Measured figures for both
cases are in [TECHNICAL.md](../TECHNICAL.md#warm-start). It also does nothing for the expert slot
cache, which still fills from the NVMe as you use the model; pre-filling that is separate work and
is not in this build.

**These files hold your conversation.** A saved state contains the session's tokens verbatim, so
treat it the way you would treat the conversation itself. It is local, it is never uploaded, and
nothing in this project collects it. At a 12k context it runs about 190 MB per state. Each profile
keeps one clean-stop state and two autosave generations, and the launcher tries to hold at most
four profiles' worth, removing the least recently used beyond that. A removal that fails is recorded
as a diagnostic rather than failing the run, so the folder can end up holding more than that.

To turn the feature off, set `warmstart` to `off` on the custom path or start the launcher with
`-Warmstart off`. To keep warm start but stop the periodic saves, set `autosave` to `off`, or to a
number of minutes between 1 and 1440 to change the interval. Turning it off stops new saves; it does
not delete what is already there. Delete `%LOCALAPPDATA%\MoE-Direct\kv\` yourself when you want it
gone.

### Precomputing a prompt file at start

Warm start only helps a start that has something to restore. The ones that do not - a first run with
a new model, a start after you deleted the saved state, a session that begins on a different prompt -
still pay the full cold prefill on the first turn, and on an agent-style system prompt that is
minutes of silence before the first token appears. Those tokens have to be evaluated once and no
setting makes that free. What you can change is *when* it happens, and whether you are the one
sitting in front of it. Point the launcher at the file your client sends as its system prompt and
that prefix is computed right after the server comes up, before you ask anything:

```powershell
.\Start-MoeDirect.ps1 -Warmup file:C:\path\to\systemprompt.txt
```

On the custom path, or in a saved preset, the same thing is the `warmup` key set to
`file:C:\path\to\systemprompt.txt`. That key takes three values: `on`, `off` and `file:<path>`.
Quote the whole value if your path contains spaces; the part after `file:` is taken exactly as you
wrote it.

**The default changed in v0.2.3: `warmup` is now `on`, where v0.2.2 and earlier had it `off`.** `on`
is the one-token warm-up request that has been in the launcher all along, sent once after the server
reports ready. It is one small request, and it means the first thing you type is not also the thing
that pays for the very first token of the run. If you preferred the old behaviour, `-Warmup off` or
the `warmup` key set to `off` restores it exactly.

That default has one honest consequence, and it is worth stating plainly because it changes what
the status screen tells you. **Every published number in [Measurements](measured-results.md) was
measured on a cold cache, and a warmed-up run is not that condition.** So a default start now reports its performance gate as
`[unmeasured] (product warm-path baseline; official measurements are cold-cache)` rather than
claiming a measured result it is not entitled to. This is a labelling change, not a slower run -
warming up does not make anything worse, it just puts the machine in a state the official figures
did not measure. `-Repro` and `-Smoke` force `warmup` back off for exactly this reason, so a
reproduction or benchmark run stands on the same cold-cache footing as the published pairs; the
status screen names the force on the warmup line when it happens.

It is one request, sent once, and you watch it happen: the launcher reads the file as UTF-8 and
sends the text to the server exactly as the file holds it, with prompt caching on and one token of
output asked for. It is deliberately not passed through the model's chat template on the way,
because tokens computed from a re-wrapped copy would not be a prefix of what your client's first
request renders, and being that prefix is the entire point. When it finishes, the launcher prints
what the server counted (one line, wrapped here to fit):

```
[warmup] Precomputed 282 tokens. The launcher cannot observe client reuse; check the first
response timings.cache_n (expected close to 282 (tokenizer boundaries and cache checkpoints may
re-evaluate a small tail)).
```

**The verification is handed to you because the launcher cannot do it itself.** It is not a proxy
and never sees your client's request, so it states what it precomputed and stops there. Look at
`timings.cache_n` in the first response you get back: close to N means the precompute was reused, and
a shortfall of a few tokens is normal rather than a failure - the tokenizer boundary where your file
meets whatever your client appends can move a token, and a model that keeps cache checkpoints
re-evaluates a short tail. A `cache_n` near zero is the answer that means it did not work. Reuse
needs your client's rendered token sequence to begin with all N precomputed tokens, so a system
prompt differing by one character, role marker or line ending is a different prefix even when the
file looks the same.

**Warm start wins when both could apply.** If a saved state is restored, the precompute is skipped
and says so: the restored prefix is already in the slot, and overwriting it would cost you more than
the precompute buys. If you want the file precomputed for a new conversation that will diverge from
the restored one, start with `-Warmstart off`.

**Nothing here can fail your run.** A missing file, an empty file, a file that is not valid UTF-8, an
HTTP failure, a timeout: each of those prints a warning, records the reason and carries on serving
without the precompute. The request is bounded at 30 minutes, because a genuine cold prefill of a
large file takes minutes and a server that never answers must not hold the launcher for ever.

One measurement, from this machine and claimed no wider: on Qwen3.5-122B a system prompt file
precomputed to 282 tokens in 12.0 s, and the first real request after it reported `cache_n` 278 of
its 299 prompt tokens - 93 % of the precomputed prefix reused, 21 tokens of genuinely new text
evaluated. The 4-token difference is the two effects named above, one token at the tokenizer seam
and three from the checkpoint tail this model family keeps. The mechanism and the full conditions
are in [TECHNICAL.md](../TECHNICAL.md#warm-up-file-precompute).

### Starting the server yourself

The save and restore calls are upstream llama.cpp, and this project changes nothing about them, so
they are available to anyone who starts `llama-server` directly instead of using the launcher. Add
`--slot-save-path <directory>` to the server arguments and **create that directory first**: if it
does not exist the server refuses to start while it is still parsing arguments. The slot id is `0`,
because this project serves one request at a time from a single slot.

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8093/slots/0?action=save" -ContentType "application/json" -Body '{"filename":"my_session.kv"}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8093/slots/0?action=restore" -ContentType "application/json" -Body '{"filename":"my_session.kv"}'
```

**HTTP 200 is not the same as success.** Read the response body: a save needs `n_saved > 0` and
`n_written > 0`, a restore needs `n_restored > 0` and `n_read > 0`. If a restore is not confirmed,
`POST /slots/0?action=erase` and continue cold only once the erase succeeds, otherwise restart the
server without restoring. Restored counts alone do not prove the prompt was reused either, so
compare `timings.cache_n` against `timings.prompt_n` on your next request and check its
time-to-first-token against a cold one.

On this path none of the identity binding above applies, because that is the launcher's work. The
engine will happily restore a file saved from a different model, and the result is undefined
behaviour rather than an error message. Restore only into exactly the same model file, the same
engine build and bundle, the same context size and slot layout, and the same state-relevant server
configuration you saved from.

## From the roadmap

| Where we are | Status |
|---|---|
| Warm start: saving and restoring slot state across restarts | **Shipped in v0.2.1.** Stopping the server in v0.2 cleared both the prefix cache and the expert slot cache, so the next start began cold. v0.2.1 saves the slot on a clean stop, saves again on a timer while serving, and restores on the next start, measured at **8.1x** faster time to first token on a strict same-prompt pair. The protocol and the caveats are in [TECHNICAL.md](../TECHNICAL.md#warm-start), and the user-facing description is under [Warm start](#warm-start). |
| Warm-up file precompute (`warmup` gains a `file:<path>` mode) | **Shipped in v0.2.2.** A fresh session's first turn still pays the full cold prefill; this lets you point the launcher at your actual system-prompt file so that prefix is precomputed right after start, once, while you watch. Usage and the one measurement behind it are under [Precomputing a prompt file at start](#precomputing-a-prompt-file-at-start). |
| Expert-cache warmer | **Designed, targeted at a later release.** Filling expert slots ahead of the first turn instead of letting them fill as you chat. The design is written and reviewed; nothing of it is in this build. |

Next: [Disk layout](disk-layout.md)
