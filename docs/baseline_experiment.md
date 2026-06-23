# Baseline Experiment

Date: 2026-06-19

## Goal

Check whether two local screenshots are enough to extract visible character features and produce a repeatable
hand-drawn SDXL img2img baseline before adding IP-Adapter, ControlNet, or a style LoRA.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM
- Python: 3.12
- Torch: 2.12.1+cu130
- VLM: Qwen3-VL-4B-Instruct
- Generator: Stable Diffusion XL Base 1.0 FP16
- Resolution: 768 x 1024
- Seed: 20260511
- Steps: 28
- Guidance scale: 6.5

The experiment used one clean full-body screenshot and one in-game screenshot with a visible weapon. Both remain
local and are not included in the repository.

## Results

| Run | Input | Strength | Result |
| --- | --- | ---: | --- |
| 001 | Original blue background | 0.45 | Character colors remained recognizable, but the game background dominated. |
| 002 | Blue-screen removal, white background | 0.60 | Stronger linework and white paper look, with clear identity drift. |
| 003 | Blue-screen removal, white background | 0.45 | Best balance of silhouette and sketch treatment; the initial prompt exceeded CLIP's token limit. |
| 004 | White background, prioritized short prompt | 0.45 | No prompt truncation; outfit and palette remained stable, but horns and headphones were not preserved. |

Qwen identified the visible hair, skin, horns, tail, glasses, cap, outfit construction, colors, and accessories. It
detected a large edged weapon while leaving the job unset because the screenshots did not provide direct job evidence.

## Findings

- Two screenshots are enough for a useful editable feature draft.
- A short prioritized prompt is more reliable than passing every VLM phrase directly to CLIP.
- Simple blue-screen keying produces a useful white-background input but can leave small holes in dark clothing.
- SDXL Base alone cannot reliably preserve small identity markers such as horns and headphones.
- The next controlled experiment should add IP-Adapter for identity and Depth ControlNet for pose, then compare a
  rights-clear hand-drawn style LoRA against this baseline.

Generated files are stored under `outputs/experiments/` and remain ignored by git.

## Performance Follow-up

The first cold VLM run took 103.3 seconds. About 58 seconds were model loading and the remaining time was mostly image
and JSON generation. Limiting VLM images to a 1024-pixel long edge and using a compact response reduced a warm run to
19.6 seconds: 3.7 seconds loading and 15.4 seconds analysis.

Generation experiments can now reuse a reviewed `features.json`. That path takes about 0.25 seconds before SDXL loading
and avoids rerunning Qwen for every seed, strength, or style change.

## Illustrious Follow-up

Date: 2026-06-21

Illustrious XL v0.1 and the Au Ra Raen face-type 3 LoRA were tested with the saved VLM features. Both runs used 24
steps and LoRA weight `0.75`.

| Run | Strength | Generation | Observation |
| --- | ---: | ---: | --- |
| `illustrious-001-au-ra` | 0.50 | 21.28 s | Horns, neck scales, tail, black-and-silver jacket, and boots appeared; cap and headphones were lost. |
| `illustrious-002-priority` | 0.46 | 16.46 s | Cap, purple glasses, headphones, green trim, outfit silhouette, and tail were retained more clearly. |

The second run is the stronger identity baseline, but it is still clean digital concept art rather than the target
colored-pencil character page. The Keyframe Animation and Character Sheet files were rejected before loading because
they were Civitai login HTML pages, not SafeTensor weights. The style comparison resumes after valid files are present.

The dependency matrix now pins `transformers==4.57.6` and `huggingface-hub==0.36.2`. This keeps Qwen3-VL available and
restores Diffusers single-file SDXL loading. Gradio is pinned to `6.0.2` so the environment passes `pip check`.

## LoRA Follow-up

Date: 2026-06-21

| Run | LoRA weights | Strength | Generation | Observation |
| --- | --- | ---: | ---: | --- |
| `illustrious-003-keyframe` | Au Ra `0.75`, Keyframe `0.65` | 0.46 | 81.68 s cold | Identity stayed close; linework became slightly rougher. |
| `illustrious-004-character-sheet-compat` | Au Ra `0.75`, Keyframe `0.65`, Character Sheet `0.25` | 0.62 | 14.16 s | No multiple views; hair and outfit drift increased. |
| `illustrious-005-keyframe-strong` | Au Ra `0.75`, Keyframe `0.85` | 0.68 | 11.44 s | Stronger redraw still looked digital and changed hair color. |

Keyframe remains useful at moderate weight for individual panels. Character Sheet is Pony-based and is disabled for
the Illustrious pipeline. Multi-view cards will use separate panel generations and deterministic layout instead.
