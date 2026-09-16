<h1 align="center">GUI-PRA (EMNLP2026)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/HTTP-FastAPI-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Models-Molmo%20%7C%20OmniParser%20%7C%20Qwen3--VL-6f42c1.svg?style=for-the-badge" alt="Model stack">
</p>

GUI-PRA is a process-reward agent for GUI tasks. It combines experience-injected criterion synthesis with criterion-guided autoregressive perception, allowing a judge to identify what should be verified, gather targeted visual evidence and score each action step.

The release contains the method description and the reproducible visual tools used by GUI-PRA. The same HTTP interfaces can be connected to an agent, evaluator or another development environment.

## 📌 Planned open-source contents

- [x] Tool code release
- [ ] Benchmark testing release

The two items are kept separate so the visual tools can be integrated independently of any benchmark checkout or evaluation backend.

## 🧠 Method overview

The method has an offline phase that samples contrasting trajectories, abstracts verification principles with a meta-judge and consolidates them into a principle repository. During online inference, the agent retrieves relevant criteria, identifies an information gap, calls a visual tool when needed, updates its state and produces an evidence chain for the step-level judge.

<p align="center">
  <img src="assets/method/method_20260824.png" width="100%" alt="GUI-PRA method overview">
</p>

<p align="center"><a href="assets/method/method_20260824.pdf">Download the method figure as PDF</a></p>

## 🌟 Key Features

### 🧩 Controller + worker architecture

A lightweight controller keeps track of available workers and dispatches requests. Each worker runs as an independent process with its own port, GPU assignment, model cache and log file. Add or remove a worker by changing the YAML configuration; clients continue to use the controller API.

### 🎯 Point grounding with Molmo

`Point` accepts an image and a natural-language description such as “the Save button”. It returns the model coordinates and an annotated image with the predicted point. The worker keeps the model on its assigned GPU and limits concurrent requests to avoid accidental memory spikes.

### 🗺️ Global UI parsing with OmniParser

`omni_parser` combines OCR, icon detection and captioning to produce a structured list of UI elements. It is useful when the current screenshot does not expose a reliable element inventory to the caller.

### 🔍 Local inspection with ZoomInSubfigure

`ZoomInSubfigure` asks an OpenAI-compatible vision API for a region described in natural language, converts the model's 0–999 coordinates to pixels, adds context padding and returns the crop. The local worker is stateless; the region-localization model can run on a separate GPU or host.

### 🧪 Safe local lifecycle and smoke checks

The release launcher validates model paths and ports before startup, waits for worker health, writes per-run logs and stops only the process groups it created. A dry-run prints resolved commands without loading models. The included smoke checker can exercise all three tools on a synthetic screenshot.

### 🔌 Simple HTTP contract

The controller uses `/list_models` and `/get_worker_address`. Workers accept `POST /worker_generate` with base64 image data and return JSON containing `error_code`, `text` and an optional `edited_image`. The same contract works for GUI-PRA, a benchmark harness or a custom application.

## 🧩 System Architecture

```text
Client (GUI-PRA, evaluator, or application)
                    │
                    ▼
Controller :20001  ──► Point :20027          ──► Molmo
                    ├──► omni_parser :20030  ──► YOLO + OCR + Florence-2
                    └──► ZoomInSubfigure :20039 ─► Qwen3-VL API :8008
```

The controller performs discovery and shortest-queue dispatch. It does not load a vision model. Point and OmniParser are local GPU workers; OCR runs on CPU inside the OmniParser process. ZoomInSubfigure uses a separate HTTP model endpoint and only performs crop extraction locally.

## ⚙️ Installation

### 1️⃣ Get the source

This directory is the standalone release candidate prepared from the GUI-PRA workspace. If it is copied into a repository, run all commands from its root:

```bash
cd GUI-PRA-open-source
```

The source archive can be rebuilt with:

```bash
python scripts/build_tool_release.py
```

The tool-only archive excludes the benchmark evaluator, downloaded data, model weights, runtime logs and local configuration overrides. The full preparation archive remains available through `python scripts/build_release.py` when the optional benchmark integration is needed.

### 2️⃣ Create the UI-Tool environment

Use the separate environment because Molmo/Florence and the benchmark critic use different Transformers versions:

```bash
conda env create -f environments/ui-tool.yml
conda activate gui-pra-ui-tool
python -m pip install -r environments/requirements-ui-tool.txt
python -m pip check
```

The recipe is a reproducible starting point, not a historical lockfile. The installed CUDA wheel must be compatible with the host driver. Confirm the environment before loading a model:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

### 3️⃣ Prepare model assets

Set the model paths in `configs/local.env`. The tool server does not download weights during startup.

| Worker | Suggested asset | Environment variable |
| --- | --- | --- |
| Point | [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) | `MOLMO_MODEL_PATH` |
| OmniParser detector | [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0), `icon_detect/model.pt` | `OMNIPARSER_DETECT_PATH` |
| OmniParser captioner | A complete Florence-2 + OmniParser caption bundle | `OMNIPARSER_CAPTION_PATH` |
| ZoomInSubfigure | Any OpenAI-compatible Qwen3-VL endpoint | `ZOOM_API_URL`, `ZOOM_MODEL_NAME` |

A caption bundle needs its configuration, tokenizer, processor, model-provided Python code and weights. If you have a Florence base directory and OmniParser caption weights, create a new bundle without modifying either source:

```bash
python scripts/prepare_caption_bundle.py \
  --florence-base models/Florence-2-base \
  --icon-caption models/OmniParser-v2.0/icon_caption \
  --output models/OmniParser-caption-bundle
```

Follow each model card for access, license and download instructions. Model files are intentionally external to this source package.

### 4️⃣ Configure the service

```bash
cp configs/local.env.example configs/local.env
# Edit model paths, GPU IDs, API endpoints and ports.
source configs/local.env
bash scripts/start_ui_tool.sh --dry-run
```

The default release configuration uses Controller `20001`, Point `20027`, OmniParser `20030` and Zoom `20039`. Point and OmniParser default to GPUs `0` and `1`; change these values if those devices are already in use. Reserve separate GPU resources for the external VLM APIs.

## 🚀 Run GUI-PRA

### 1️⃣ Start the Zoom and router VLM APIs

The controller and workers expect an OpenAI-compatible endpoint. `scripts/serve_vlm.sh` is an adjustable vLLM example:

```bash
conda activate gui-pra-vllm
VLM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
VLM_MODEL_NAME=qwen3-vl-8b-instruct VLM_GPUS=3 VLM_PORT=6008 \
  bash scripts/serve_vlm.sh

VLM_MODEL_PATH=/path/to/Qwen3-VL-32B-Instruct \
VLM_MODEL_NAME=qwen3-vl-32b-instruct VLM_GPUS=4,5 VLM_TP=2 VLM_PORT=8008 \
  bash scripts/serve_vlm.sh
```

The first endpoint is used by GUI-PRA's principle selection, tool routing and observation filtering. The second endpoint is used by ZoomInSubfigure. Check `vllm serve --help` in the selected vLLM version before changing memory, context or parallelism settings.

### 2️⃣ Start Controller and workers

```bash
conda activate gui-pra-ui-tool
source configs/local.env
export PYTHONPATH="$PWD"
python UI_Tool/launch_scripts/start_server_local.py \
  --config UI_Tool/launch_scripts/config/all_service_example_local_1.yaml
```

This is the same launch shape as the original project command, with a portable configuration:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python UI_Tool/launch_scripts/start_server_local.py \
  --config UI_Tool/launch_scripts/config/all_service_example_local_1.yaml
```

Preview the resolved commands first:

```bash
python UI_Tool/launch_scripts/start_server_local.py \
  --config UI_Tool/launch_scripts/config/all_service_example_local_1.yaml --dry-run
```

The launcher creates an isolated directory under `runtime/ui-tool/<run-id>/`. Press `Ctrl+C` to stop the processes started by that terminal. It never searches for or terminates unrelated workers.

### 3️⃣ Check registration and run a smoke test

In a second terminal:

```bash
conda activate gui-pra-ui-tool
source configs/local.env
python scripts/check_services.py
python scripts/check_services.py --smoke
```

The first command checks the controller, the three required worker names and the configured VLM model IDs. The `--smoke` variant sends `assets/smoke_ui.png` to Point, OmniParser and ZoomInSubfigure and writes returned images/text under `runtime/tool-smoke/`. A successful HTTP response confirms the service path; inspect the returned image and text for model quality.

## 📡 HTTP API

### Discover workers

```bash
curl -sS -X POST http://127.0.0.1:20001/list_models
curl -sS -X POST http://127.0.0.1:20001/get_worker_address \
  -H 'Content-Type: application/json' \
  -d '{"model":"Point"}'
```

The expected registered names are `Point`, `omni_parser` and `ZoomInSubfigure`. `OmniParser` is the human-readable YAML/log label; the controller lookup name is lowercase `omni_parser`.

### Call a worker directly

The worker request contains raw base64 image data. `Point` and `ZoomInSubfigure` also require `param`; OmniParser only needs `image`:

```python
import base64
import requests

image = base64.b64encode(open("assets/smoke_ui.png", "rb").read()).decode()
controller = "http://127.0.0.1:20001"
worker = requests.post(
    controller + "/get_worker_address", json={"model": "Point"}, timeout=10
).json()["address"]
response = requests.post(
    worker + "/worker_generate",
    json={"image": image, "param": "The blue Save button"},
    timeout=120,
)
response.raise_for_status()
print(response.json()["text"])
```

A successful response has `error_code: 0`. `edited_image` is a base64 PNG/JPEG when the worker produces an annotated or cropped image. Coordinates from Point are represented in the source screenshot coordinate system; ZoomInSubfigure consumes and returns a crop after converting its model response from the 0–999 normalized convention.

## 🧰 Configuration reference

The main file is [`UI_Tool/launch_scripts/config/all_service_example_local_1.yaml`](UI_Tool/launch_scripts/config/all_service_example_local_1.yaml). Environment expansion supports `${NAME}` and `${NAME:-default}`. Important settings are:

```yaml
base_dir: "${PROJECT_ROOT}/UI_Tool"
runtime_dir: "${UI_TOOL_RUNTIME_DIR:-runtime/ui-tool}"
controller_config:
  cmd:
    port: "${UI_TOOL_CONTROLLER_PORT:-20001}"
tool_worker_config:
  - Point:
      cuda_visible_devices: "${POINT_GPU:-0}"
      cmd:
        model_path: "${MOLMO_MODEL_PATH:-/path/to/Molmo-7B-D-0924}"
```

- `POINT_GPU` and `OMNIPARSER_GPU` must identify free GPUs. An empty GPU value is used for the stateless Zoom worker.
- `MOLMO_MODEL_PATH`, `OMNIPARSER_DETECT_PATH` and `OMNIPARSER_CAPTION_PATH` must point to readable local assets.
- `ZOOM_API_URL` must be an OpenAI-compatible `/v1/chat/completions` endpoint. `ZOOM_MODEL_NAME` must match `--served-model-name`.
- `UI_TOOL_CONTROLLER_PORT`, `POINT_PORT`, `OMNIPARSER_PORT` and `ZOOM_PORT` must be unique and reachable.
- `UI_TOOL_RUNTIME_DIR` and `UI_TOOL_HF_HOME` keep runtime files and model caches outside the source tree.

For separate hosts, expose controller and worker ports, set `UI_TOOL_HOST=0.0.0.0`, and point `GUIPRA_TOOL_CONTROLLER_ADDR` at the controller host. The HTTP interface has no built-in application authentication; use a trusted network or an authenticated reverse proxy.

## 🧪 Development checks

The complete source release includes an offline test suite that starts the real controller with a test-only fake worker; it does not load a model or call an external API. The tool-only archive can use the launcher dry-run and `scripts/check_services.py --help`. In the complete source tree, run the evaluator-backed regression suite with:

```bash
conda activate gui-pra-benchmark
python -m pip install -r environments/requirements-test.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

For a code-only syntax check:

```bash
python -m compileall -q UI_Tool os-critic-bench scripts tests
```

## 🔌 Integration with other development environments

The visual workers expose a small HTTP contract and can be integrated into an existing agent or evaluation stack. Start the services, obtain a worker address from `/get_worker_address`, then send a base64-encoded screenshot to `/worker_generate`. Keep the controller and worker ports reachable from the host running your application, and map the returned `text` and `edited_image` fields into your own observation pipeline. The included configuration and scripts are templates; you can replace the model endpoints, process manager or client while keeping the request format.

## 📜 Citation

If you use GUI-PRA, please cite:

```bibtex
@misc{xiong2026guipraprocessrewardagent,
      title={GUI-PRA: Process Reward Agent for GUI Tasks},
      author={Tao Xiong and Xavier Hu and Yurun Chen and Yuhang Liu and Changqiao Wu and Pengzhi Gao and Wei Liu and Jian Luan and Shengyu Zhang},
      year={2026},
      eprint={2509.23263},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.23263},
}
```

Please also cite the upstream model and tool projects listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## 📨 Contact

Feel free to contact us: xiongtao@zju.edu.cn
