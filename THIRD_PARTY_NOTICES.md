# Third-party notices

## llama.cpp

MoE-Direct is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp),
MIT License, Copyright (c) 2023-2026 The ggml authors.

- Base: tag `b10057`, commit `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- MoE-Direct will ship as a patch set / modified tree on that base; all upstream
  copyright headers and the upstream `LICENSE` will be preserved in source
  releases.
- MoE-Direct is an independent project and is not affiliated with or endorsed
  by the llama.cpp / ggml maintainers.

## Models

MoE-Direct does not include or redistribute any model weights. Users supply
their own GGUF files under the respective model licenses; the repacker runs
locally and records the SHA-256 of its input. Qwen, Kimi, Mistral, gpt-oss and
all other model names and trademarks belong to their respective owners; this
project is affiliated with and endorsed by none of them.
