# Third-party notices

## llama.cpp

MoE-Direct is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp),
MIT License, Copyright (c) 2023-2026 The ggml authors.

- Base: tag `b10057`, commit `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- MoE-Direct ships as a patch set / modified tree on that base; the release
  bundle contains binaries compiled from that modified tree (`llama-server.exe`,
  `llama.dll`, `ggml*.dll` and the other tools). All upstream copyright headers
  are preserved in source releases, and the upstream license text ships in the
  bundle as `LICENSE.llama.cpp.txt`.
- MoE-Direct is an independent project and is not affiliated with or endorsed
  by the llama.cpp / ggml maintainers.

## NVIDIA CUDA runtime libraries

The release bundle redistributes the following NVIDIA CUDA runtime libraries so
that the dense-layer CUDA path works without a separate CUDA Toolkit install:
`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`.

- These are redistributable components of the NVIDIA CUDA Toolkit and are
  redistributed under the terms of the
  [NVIDIA CUDA Toolkit End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html).
- NVIDIA and CUDA are trademarks and/or registered trademarks of NVIDIA
  Corporation. This project is not affiliated with or endorsed by NVIDIA.
- (`ggml-cuda.dll` is not an NVIDIA library - it is compiled from the llama.cpp
  tree above and covered by the MIT license.)

## CPython (embeddable distribution)

The bundle ships the official CPython **3.11.9** Windows embeddable
distribution under `repacker\python\`, so the repacker runs without a separate
Python install.

- License: Python Software Foundation License Version 2; the full text ships
  inside the bundle at `repacker\python\LICENSE.txt`.
- Python is a trademark of the Python Software Foundation. This project is not
  affiliated with or endorsed by the PSF.

## Models

MoE-Direct does not include or redistribute any model weights. Users supply
their own GGUF files under the respective model licenses; the repacker runs
locally against those files. Qwen, Kimi, Mistral, gpt-oss and
all other model names and trademarks belong to their respective owners; this
project is affiliated with and endorsed by none of them.
