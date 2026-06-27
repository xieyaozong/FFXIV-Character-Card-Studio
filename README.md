<div align="center">

# FFXIV Character Card Studio

**Local framework for turning Final Fantasy XIV screenshots into editable character profiles and card prompts.**

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-local_framework-orange)

</div>

This is an unofficial personal fan project. FINAL FANTASY XIV and all related names, assets, and trademarks belong to
SQUARE ENIX. This repository does not redistribute game assets, screenshots, model weights, LoRAs, or curated FFXIV
knowledge data.

## Scope

The public repository contains the framework only:

- screenshot triage and subject cropping
- background removal
- local vision-model feature extraction
- FFXIV knowledge-file templates
- prompt and generation-spec utilities
- local experiment runners and tests

The maintained FFXIV race data, reference images, model weights, LoRAs, screenshots, and generated outputs are private
local files. They are intentionally ignored by git.

## Pipeline

```text
private screenshot
  -> triage and crop
  -> background removal
  -> local vision model
  -> editable features.json
  -> FFXIV knowledge checks
  -> prompt / generation spec
  -> local generation experiment
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the matching CUDA build of `torch` and `torchvision` yourself, then install the local model stack:

```powershell
python -m pip install -r requirements-ml.txt
```

## Run

Triage screenshots:

```powershell
python scripts\triage_screenshots.py --input-dir "path\to\screenshots" --output-dir "outputs\triage"
```

Extract features only:

```powershell
python scripts\run_baseline_experiment.py `
  --character-image "path\to\character.png" `
  --weapon-image "path\to\weapon.png" `
  --output-dir "outputs\experiments\my-run" `
  --features-only
```

For local tuning from VS Code PowerShell, use the ignored `run.ps1` wrapper and edit only its top block. See
[docs/manual_experiments.md](docs/manual_experiments.md).

## Structure

```text
src/domain/          profile and evidence models
src/preprocessing/   image loading, crop, background, palette, control maps
src/vlm/             local vision-model adapters and prompts
src/knowledge/       FFXIV entity, race, gear, and asset matching
src/prompting/       prompt and generation-spec compilation
scripts/             command-line workflows
knowledge/ffxiv/     public templates plus private local data
configs/presets/     product, pose, expression, and style presets
tests/               lightweight checks; no model downloads required
```

## Private Data

These stay local:

- `models/`
- `outputs/`
- `private_inputs/`
- `datasets/`
- `knowledge/ffxiv/*.yaml` except `*.example.yaml`
- `knowledge/ffxiv/*.npz`
- `knowledge/ffxiv/reference/`
- `knowledge/ffxiv/lore/`

## License

[MIT](LICENSE) for the code. Game assets, model weights, LoRAs, screenshots, and curated FFXIV data are not included.
