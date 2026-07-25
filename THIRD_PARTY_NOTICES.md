# Third-party notices

## llama.cpp

NVMoE is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp),
MIT License, Copyright (c) 2023-2026 The ggml authors.

- Base: tag `b10057`, commit `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- NVMoE ships as a patch set / modified tree on that base; all upstream
  copyright headers and the upstream `LICENSE` are preserved in source releases.
- NVMoE is an independent project and is not affiliated with or endorsed by
  the llama.cpp / ggml maintainers.

## Models

NVMoE does not include or redistribute any model weights. Users supply their
own GGUF files under the respective model licenses; the repacker runs locally
and records the SHA-256 of its input.
