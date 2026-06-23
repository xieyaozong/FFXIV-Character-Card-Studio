# GPU And Model Setup

The base environment deliberately excludes Torch and model weights.

The verified local model stack uses `diffusers==0.38.0`, `transformers==4.57.6`, and
`huggingface-hub==0.36.2`. Transformers 5.12 removed the CLIP structure expected by Diffusers single-file SDXL
loading, so it is not used in this environment.

## GPU Packages Not Installed

Use the official PyTorch wheel index that matches the NVIDIA driver:

```text
torch==2.12.1
torchvision==0.27.1
```

Official wheels were available for CUDA 12.6 and CUDA 13.0 when this environment was prepared. Do not install the default CPU wheel by accident.

After Torch:

```powershell
python -m pip install -r requirements-ml.txt
```

## VLM

Recommended starting point:

```text
Qwen/Qwen3-VL-4B-Instruct
```

Use it for structured identity, outfit, accessory, and visible-item candidates. It is an Apache-2.0 image-to-text model and does not require gated access. `Qwen/Qwen3-VL-8B-Instruct` can be tested later when VRAM allows. `Qwen/Qwen2.5-VL-3B-Instruct` remains a lower-memory fallback. The model must run locally and return evidence-first JSON; job and weapon candidates remain optional and require user confirmation.

## Background Removal

The base environment installs `rembg`, but its segmentation checkpoint is downloaded separately on first use. The UI also includes a lightweight blue-screen mode for the current character screenshots.

Optional higher-quality stage:

```text
SAM 2.1 checkpoint
```

SAM 2.1 requires Torch and a separate checkpoint, so it is not part of the base installation.

## Image Generation

Primary anime generator for the next experiment:

```text
OnomaAIResearch/Illustrious-xl-early-release-v0
```

Use `Illustrious-XL-v0.1.safetensors` with the matching Illustrious LoRA set. Keep
`stabilityai/stable-diffusion-xl-base-1.0` only as the original baseline; SDXL and Illustrious LoRAs must not be mixed
without an explicit compatibility test.

Additional conditioning:

```text
h94/IP-Adapter
SDXL IP-Adapter Plus image encoder and weights
xinsir/controlnet-openpose-sdxl-1.0
optional SDXL VAE fix
```

The listed IP-Adapter and OpenPose ControlNet use Apache-2.0 model-card licenses. Pin a model revision before running repeatable experiments.

The app should generate individual panels and compose cards afterward. Exact Chinese, Japanese, and English text must be rendered by Pillow or HTML, not by the diffusion model.

## LoRA Roles

Keep two concepts separate:

```text
Anatomy LoRA: FFXIV race traits such as horns, scales, ears, and tail shape
Reference conditioning: the current character's face, outfit, palette, and accessories
Style LoRA: colored pencil, rough line art, and paper texture
Composition LoRA: optional multi-view or reference-sheet arrangement
```

Do not train one LoRA to own all four roles. The first Illustrious download batch and exact Windows PowerShell commands
are listed in `docs/model_downloads.md`.

Character LoRA preparation:

- 20 to 50 curated screenshots
- multiple angles and crops
- duplicate frames removed
- outfit groups captioned separately
- private images and trained weights excluded from git

Approximate VRAM planning:

| VRAM | Practical starting point |
| --- | --- |
| 8 GB | 3B VLM with quantization; SD 1.5 or constrained SDXL inference |
| 12 GB | 3B VLM; SDXL inference; memory-optimized LoRA experiments |
| 16 GB | SDXL inference and character LoRA with optimization |
| 24 GB+ | more comfortable SDXL LoRA and larger VLM experiments |

Final CUDA index, quantization, attention backend, and training settings depend on the exact GPU model and driver.

See `models/model-manifest.example.yaml` for the model IDs and license fields that should be recorded locally.
