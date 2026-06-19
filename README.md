# FFXIV-Character-Card-Studio

Local tool for turning Final Fantasy XIV character screenshots into editable character profiles, avatars, expression sheets, and introduction cards.

This is an unofficial personal project. FINAL FANTASY XIV and related names and assets belong to their respective rights holders; the repository does not redistribute game assets, screenshots, or model weights.

The first milestone covers screenshot import, background removal, VLM feature candidates, user review, multilingual entity mapping, and structured prompt plans. Image generation and LoRA training are installed separately after the GPU environment is confirmed.

## Workflow

```text
Character screenshots
  -> background removal and crop review
  -> local VLM feature candidates
  -> multilingual FFXIV entity matching
  -> user confirmation
  -> character and outfit profiles
  -> prompt plan
  -> panel generation
  -> postcard / avatar composition
```

## Design Rules

- Screenshot evidence comes first; the app does not invent jobs, weapons, pets, or props.
- Every detected feature remains editable and requires confirmation when uncertain.
- Job and weapon sections are optional and hidden when not selected.
- Japanese, Chinese, and English names map to language-neutral entity IDs.
- Exact card text is rendered by the layout engine rather than the diffusion model.
- Private screenshots, model weights, LoRA files, and full-size outputs stay outside git.

## Base Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the API:

```powershell
uvicorn app.main:app --reload
```

Run the UI:

```powershell
python -m app.ui.gradio_app
```

## GPU Setup

Torch, the VLM stack, diffusion models, ControlNet, IP-Adapter, and LoRA training tools are intentionally not installed by `requirements.txt`. See `docs/gpu_and_models.md` after confirming the GPU model and VRAM.

The default VLM target is `Qwen/Qwen3-VL-4B-Instruct`; no model weights are downloaded during the base setup.

## Repository Layout

```text
app/                 FastAPI and Gradio entry points
src/domain/          profile, evidence, and generation-plan schemas
src/preprocessing/   image loading, background removal, crops, color palette
src/vlm/             replaceable VLM adapter and feature analysis
src/catalog/         multilingual FFXIV entity resolution
src/prompting/       structured prompt fragments and compiler
content_packs/ffxiv/ locale and entity data
configs/presets/     output, style, pose, and expression presets
private_inputs/      local screenshots ignored by git
outputs/             local generated files ignored by git
docs/                architecture, data policy, and GPU/model setup
tests/               schema and preprocessing checks
```
