# Source and dependency notices

This directory contains a local release candidate assembled from the supplied GUI-PRA research workspace. No project-level LICENSE file was found in the selected UI-Tool tree or the checked OS-Oracle source repository. A license for the research code has therefore not been invented or applied during preparation; the project owners need to supply their intended license before a public release.

| Component | Source | Included here |
| --- | --- | --- |
| OS-Oracle evaluator | https://github.com/numbmelon/OS-Oracle | Adapted evaluation/formatting/inference/metric code and upstream README |
| OS-Critic Bench data | https://huggingface.co/datasets/OS-Copilot/OS-Critic-Bench | Dataset card only; it declares Apache-2.0 for the dataset |
| OmniParser utilities | https://github.com/microsoft/OmniParser | Utilities carried by the supplied UI-Tool source; model weights are external |
| Molmo | https://huggingface.co/allenai/Molmo-7B-D-0924 | Worker integration; weights/model-provided Python code are external |
| Florence-2 | https://huggingface.co/microsoft/Florence-2-base | Loader and bundle preparation helper; weights/model-provided Python code are external |
| Qwen2.5-VL / Qwen3-VL | Respective Qwen model repositories | Critic and HTTP adapters; weights are external |
| PyTorch, Transformers, PaddleOCR, EasyOCR, Ultralytics, Supervision, FastAPI, vLLM | Respective package repositories | Dependency references only; installed environments are not copied |

Code, dependencies, data and model weights retain their respective terms. The dataset card does not establish the license of the OS-Oracle code. The individual OmniParser code and detector/caption weights may also have different terms; retain their model cards and license files when downloading them. This preparation package does not relicense third-party components.

The source archive is built from an explicit directory allowlist. It omits local environment overrides, credentials, runtime logs, evaluation outputs, model weights and downloaded benchmark data. The synthetic `assets/smoke_ui.png` was created for this package and contains no benchmark example.

The copied source manifest and environment metadata are available in `provenance/`. The upstream README and dataset card are preserved in `third_party/`.
