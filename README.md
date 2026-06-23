<div align="center">

# FFXIV Character Card Studio

**Turn Final Fantasy XIV character screenshots into editable profiles and character-card prompts — fully local.**

![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/milestone-VLM→prompt-orange)

</div>

> Of all my side projects this is the most personal one: it is built around a game I actually play (on the Japanese data centers). It grows one milestone at a time rather than all at once.

---

## Disclaimer

This is an unofficial, personal fan project. FINAL FANTASY XIV and all related names, assets, and trademarks belong to SQUARE ENIX. This repository does **not** redistribute game assets, screenshots, or model weights.

---

## Status

Current milestone: **screenshots → VLM features → character prompt.** This part is implemented and tested.

Everything after prompt building (image generation, retrieval, card layout) is work in progress and intentionally not the focus of this release. See [Roadmap](#roadmap).

## Pipeline

```text
FFXIV screenshots (private, never committed)
  │
  ▼  triage: score shots, drop dark / back-facing / too-distant frames   [scripts/triage_screenshots.py]
  ▼  subject crop + background removal (rembg)                            [src/preprocessing]
  ▼  local VLM feature extraction (Qwen3-VL-4B)                           [src/vlm/qwen_backend.py]
  ▼  editable features.json  (evidence-first, nothing invented)
  ▼  CLIP-fit character prompt                                            [scripts/run_baseline_experiment.py]
  ══════════════════════════════════ current milestone ends here ══════════════════════════════════
  ▼  (WIP)     SDXL / IP-Adapter image generation
  ▼  (planned) entity database + RAG, deterministic card layout
```

## Design rules

- Screenshot evidence comes first; the tool never invents jobs, weapons, pets, or props.
- Every detected feature stays editable and is flagged when uncertain.
- Job and weapon sections are optional and hidden unless visible.
- Japanese, Simplified Chinese, Traditional Chinese, and English names map to language-neutral entity IDs.
- Private screenshots, model weights, LoRA files, and full-size outputs stay out of git.

## Quick start

Base environment (CPU, no model weights):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU stack for the local VLM (install the matching NVIDIA build of `torch`/`torchvision` first — see [docs/gpu_and_models.md](docs/gpu_and_models.md)):

```powershell
python -m pip install -r requirements-ml.txt
```

The default VLM is `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0). Weights are downloaded locally and never committed.

## Running the milestone

Two PowerShell entry points keep all tunables at the top of one file, so a run is a single edit.

**1. Triage** — score a folder of screenshots and crop the usable ones:

```powershell
.\triage.ps1     # edit $InputDir / thresholds, then run; review crops under outputs/triage/
```

**2. Screenshots → prompt** — extract features and build the prompt (no image generation):

```powershell
.\run.ps1        # set $FeaturesOnly = $true to stop after features.json + prompt.txt
```

Both wrap CLIs you can also call directly:

```powershell
python scripts/run_baseline_experiment.py `
  --character-image "path\to\character.png" `
  --weapon-image "path\to\weapon.png" `
  --output-dir "outputs\experiments\my-run" `
  --features-only
```

Outputs (`features.json`, `prompt.txt`, cropped previews) land under `outputs/`, which git ignores.

## How it works

| Stage | Module | Responsibility |
| --- | --- | --- |
| Triage | [src/preprocessing/triage.py](src/preprocessing/triage.py) | Score each shot by subject size, brightness, and sharpness; drop unusable frames. |
| Crop + background | [src/preprocessing/background.py](src/preprocessing/background.py) | rembg subject mask → bbox crop → white-background composite. Works on any background, not just the blue glamour backdrop. |
| VLM | [src/vlm/qwen_backend.py](src/vlm/qwen_backend.py) | Qwen3-VL returns evidence-first JSON: identity, outfit, optional job/weapon. |
| Prompt | [scripts/run_baseline_experiment.py](scripts/run_baseline_experiment.py) | Deduplicate and compact features, then trim to the CLIP 77-token limit so no detail is silently dropped. |
| Knowledge | [src/catalog/entity_catalog.py](src/catalog/entity_catalog.py), [content_packs/ffxiv/](content_packs/ffxiv/) | Multilingual alias → canonical entity IDs and anatomy rules (advisory by default). |

The FastAPI + Gradio app ([app/](app/)) currently exposes the preprocessing steps (background removal, palette) for quick visual checks.

## Project structure

```text
app/                 FastAPI + Gradio entry points (preprocessing demo)
src/domain/          profile, evidence, and prompt schemas (pydantic)
src/preprocessing/   image IO, triage, subject crop, background removal, palette
src/vlm/             replaceable VLM adapter + feature extraction
src/catalog/         multilingual FFXIV entity resolution
src/prompting/       confirmed-profile prompt compiler
scripts/             triage and screenshot→prompt runners
content_packs/ffxiv/ locale, entity, and anatomy data (the source of truth, not the model)
configs/presets/     style / pose / expression / product presets
docs/                architecture, data policy, GPU/model setup, progress log
tests/               schema, preprocessing, and prompt checks
```

## What is tracked vs local-only

| Tracked in git | Local-only (git-ignored) |
| --- | --- |
| Source code, tests, content packs, configs, docs | `models/` — VLM, diffusion, ControlNet, IP-Adapter, LoRA weights |
| Example manifests (`models/model-manifest.example.yaml`) | `character/`, `private_inputs/` — your FFXIV screenshots |
| README, license, environment example | `outputs/` — features, prompts, generated images |
| | `.env`, `.venv/`, caches |

## Roadmap

- [x] Screenshot triage, subject crop, background removal
- [x] Local VLM feature extraction (evidence-first JSON)
- [x] Compact, CLIP-safe prompt building
- [ ] **Next** — knowledge-DB-driven race / anatomy recognition, so screenshots are understood without the user writing prompts (see [docs/knowledge-layer.md](docs/knowledge-layer.md))
- [ ] **WIP** — SDXL / IP-Adapter generation (identity-preserving redraw, face detailing, upscale)
- [ ] Deterministic card layout (text rendered by code, not the diffusion model)

Design and direction: [docs/knowledge-layer.md](docs/knowledge-layer.md) · [docs/architecture.md](docs/architecture.md). Progress is logged per session in [docs/progress-log.md](docs/progress-log.md).

## Tech stack

Python 3.12 · FastAPI · Gradio · pydantic · Pillow / OpenCV · rembg (ONNX Runtime) · Hugging Face Transformers (Qwen3-VL) · Diffusers (experimental generation).

## License

[MIT](LICENSE) for the code. Game assets, model weights, and LoRA files are **not** covered and are never distributed here.
