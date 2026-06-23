<div align="center">

# FFXIV Character Card Studio

**Turn Final Fantasy XIV character screenshots into editable profiles and character-card prompts — fully local.**

ファイナルファンタジーXIV のキャラクタースクリーンショットを、編集可能なプロフィールとキャラクターカード用プロンプトに変換するローカルツール。

將《Final Fantasy XIV》的角色截圖，轉換成可編輯的角色設定與角色卡 prompt 的本機工具。

![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/milestone-VLM→prompt-orange)

</div>

> Of all my side projects this is the most personal one: it is built around a game I actually play (on the Japanese data centers). Naming and scope are still settling, so expect the structure to grow one milestone at a time rather than all at once.

---

## About / 概要 / 簡介

**EN** — A local pipeline that reads your own FFXIV screenshots, extracts *only what is visible* (hair, horns, scales, tail, outfit, accessories, visible weapon), and turns the reviewed result into a structured generation prompt. Everything runs on your machine; no screenshots or model weights leave it.

**日本語** — 手元の FFXIV スクリーンショットから「実際に見えている要素」（髪・角・鱗・尻尾・衣装・アクセサリー・見えている武器）だけを抽出し、確認後に構造化されたプロンプトへ変換するローカルパイプラインです。すべて自分の PC 上で動作し、スクリーンショットやモデルの重みは外部に送信されません。

**繁體中文** — 一條本機流程：讀取你自己的 FFXIV 截圖，只擷取「畫面中實際可見」的特徵（髮色、角、鱗、尾、服裝、配件、可見武器），經你確認後轉成結構化的生成 prompt。全程在本機執行，截圖與模型權重都不會外傳。

## Disclaimer / 免責事項 / 免責聲明

**EN** — This is an unofficial, personal fan project. FINAL FANTASY XIV and all related names, assets, and trademarks belong to SQUARE ENIX. This repository does **not** redistribute game assets, screenshots, or model weights.

**日本語** — 本プロジェクトは非公式の個人ファンプロジェクトです。『ファイナルファンタジーXIV』および関連する名称・素材・商標はすべて株式会社スクウェア・エニックスに帰属します。本リポジトリはゲーム素材・スクリーンショット・モデルの重みを再配布しません。

**繁體中文** — 本專案為非官方的個人同人專案。《FINAL FANTASY XIV》及其相關名稱、素材與商標皆屬於 SQUARE ENIX 所有。本儲存庫**不會**散布遊戲素材、截圖或模型權重。

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
- [ ] **WIP** — SDXL / IP-Adapter generation (identity-preserving redraw, face detailing, upscale)
- [ ] Entity database + RAG for fuzzy equipment / NPC names
- [ ] Deterministic card layout (text rendered by code, not the diffusion model)

Progress is logged per session in [docs/progress-log.md](docs/progress-log.md).

## Tech stack

Python 3.12 · FastAPI · Gradio · pydantic · Pillow / OpenCV · rembg (ONNX Runtime) · Hugging Face Transformers (Qwen3-VL) · Diffusers (experimental generation).

## License

[MIT](LICENSE) for the code. Game assets, model weights, and LoRA files are **not** covered and are never distributed here.
