<div align="center">

# FFXIV Character Card Studio

**Local framework for turning character screenshots into structured profiles, prompt plans, and card layouts.**

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-framework-orange)
![tests](https://img.shields.io/badge/tests-lightweight-brightgreen)

</div>

FFXIV Character Card Studio is a local experiment framework for a narrow workflow: inspect a character screenshot,
extract visible traits, compile an editable prompt plan, and render character-card style outputs. The project is built
around small Python modules so the vision, knowledge, prompt, and rendering steps can be tested separately.

## What It Does

- scores and crops screenshots before they reach heavier models
- removes simple backgrounds for cleaner character inputs
- extracts visible character traits with a local VLM backend
- maps traits through editable knowledge templates
- compiles prompt and generation specs from structured choices
- renders card-layout previews from generated or prepared assets

## Flow

```text
Screenshot
  -> triage / crop
  -> background cleanup
  -> VLM trait extraction
  -> knowledge checks
  -> prompt plan
  -> generation experiment
  -> card layout
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the matching CUDA build of `torch` and `torchvision` for your GPU, then install the optional model stack:

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

For local tuning from VS Code PowerShell, see [docs/manual_experiments.md](docs/manual_experiments.md).

## Structure

```text
src/domain/          profile and evidence models
src/preprocessing/   image loading, crop, background, palette, control maps
src/vlm/             local vision-model adapters and prompts
src/knowledge/       FFXIV entity, race, gear, and asset matching
src/prompting/       prompt and generation-spec compilation
scripts/             command-line workflows
knowledge/ffxiv/     editable knowledge templates
configs/presets/     product, pose, expression, and style presets
tests/               lightweight checks; no model downloads required
```

## Disclaimer

This is an unofficial fan-made tool. It is not affiliated with, endorsed by, or sponsored by Square Enix. FINAL FANTASY
XIV and related names, assets, and trademarks belong to their respective owners.

## License

[MIT](LICENSE)
