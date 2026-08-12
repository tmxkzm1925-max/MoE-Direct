# Clients

[Back to the README](../README.md)

## Connecting a client

When the launcher says `ready` you have an endpoint speaking the documented OpenAI-compatible
subset on `http://127.0.0.1:<port>/v1`, bound to loopback only. No API key is required locally;
clients that insist on one will accept any string.

```bash
curl http://127.0.0.1:8093/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8093/v1", api_key="none")
r = client.chat.completions.create(model="local",
        messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

Three settings cover most chat apps: **base URL** (`http://127.0.0.1:8093/v1`), **API key** (any
non-empty string), **model name** (whatever `/v1/models` returns).

**Read this before you file a compatibility issue:**

- This serves the **implemented OpenAI-compatible subset** - the common chat-completions surface,
  streaming, cancel and `/v1/models`. It is not full OpenAI API compatibility.
- **One request at a time** (`-np 1`). A second concurrent request waits. Restarting the server
  loses the queue.
- Long system prompts are expensive here: the first turn pays full prefill. Agent-style clients
  that ship 15k+ token system prompts will feel that on turn one, and much less afterwards, because
  the prefix cache absorbs the repeat.

**Clients verified on the v0.2 staging stack:** Hermes Agent (Qwen3.5-122B profile, context raised
to 65536 through the launcher's custom path; also driven against the K2.6 1T-class profile). Notes
from those sessions: agent-style clients tend to
require a large minimum context; "thinking" mode works, but the thinking tokens are generated at
disk-tier decode speed before the answer starts, so it carries an extra time cost on top of every
reply - whether that trade is worth it is your call, per task; and if
the app caches model metadata you may need to re-register the endpoint after a restart. Other
clients may work if they use the documented subset - they are simply not on this list until someone
has run them.

**Raise your client's timeouts before a first turn on a very large model.** During prefill the
server streams nothing back, and on the largest profile that silence is long. `LIVE` observation,
reference machine, K2.6 profile (8 GiB budget, QD 8, prefetch on K8/N4, ctx 65536,
`--no-kv-offload`): a ~15.7k-token first prompt tripped the client's default stale-stream watchdog
(900 s) repeatedly. The first attempt was interrupted three times by the watchdog, then ended in
an app error that needed a restart; the retried attempt after the restart completed in **about 44 minutes end
to end**, absorbing two more watchdog aborts on the way. Every abort resumed from the server's
prompt cache, so no server work was lost - the cost was client-side friction, not recomputation.
If your client exposes timeout settings, raise anything that watches for "no bytes received"
**above your expected first-turn time - 3600 s is a reasonable floor for 1T-class, 7200 s gives
comfortable margin**. In Hermes Agent's config the relevant keys are
`agent.local_stream_stale_timeout` (default 900) and `agent.gateway_timeout` (default 1800) - the
session above ran on those defaults, which is exactly how it collected the aborts; raising both is
our recommendation, not something the app does for you. `agent.api_max_retries: 3` can stay, since
aborted attempts resume from the prompt cache. This is a first-turn cost only: in the same session
the second turn entered generation in under a minute, because the prefix cache absorbed the
repeated system prompt.

Next: [Warm start and prompt precompute](warm-start.md)
