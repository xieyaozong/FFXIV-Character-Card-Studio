# Model Downloads

The next experiment uses one Illustrious checkpoint, two matching LoRAs, and one Pony compatibility candidate. Keep
every weight local; model files remain ignored by git.

## First Batch

| Role | Model | File | Size | Local path |
| --- | --- | --- | ---: | --- |
| Foundation checkpoint | [Illustrious XL v0.1](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0) | `Illustrious-XL-v0.1.safetensors` | 6.94 GB | `models/diffusion/illustrious-xl-v0.1/` |
| FFXIV anatomy LoRA | [Au Ra - Raen Facetype-3-ver-2](https://civitai.com/models/2541330) | SafeTensor LoRA | 446 MB | `models/loras/illustrious/ffxiv/` |
| Colored sketch style LoRA | [Keyframe Animation V1.1](https://civitai.com/models/1660846) | SafeTensor LoRA | 223 MB | `models/loras/illustrious/style/` |
| Reference-sheet composition LoRA | [Character Sheet](https://civitai.com/models/368139/character-sheet) | SafeTensor LoRA | 456 MB | `models/loras/pony/composition/` |

The Au Ra LoRA is the first anatomy candidate because the current character has Raen-colored horns, scales, and tail.
It targets face type 3 and must be treated as an experiment until the generated face is compared with the screenshots.

## Windows PowerShell

Run from the repository root with `.venv` activated:

```powershell
New-Item -ItemType Directory -Force models\diffusion\illustrious-xl-v0.1
New-Item -ItemType Directory -Force models\loras\illustrious\ffxiv
New-Item -ItemType Directory -Force models\loras\illustrious\style
New-Item -ItemType Directory -Force models\loras\pony\composition

.\.venv\Scripts\hf.exe download `
  OnomaAIResearch/Illustrious-xl-early-release-v0 `
  Illustrious-XL-v0.1.safetensors `
  --local-dir models\diffusion\illustrious-xl-v0.1

$token = $env:CIVITAI_API_TOKEN

Invoke-WebRequest `
  -Uri "https://civitai.com/api/download/models/2856048?token=$token" `
  -OutFile "models\loras\illustrious\ffxiv\au-ra-raen-facetype3-v2.safetensors"

Invoke-WebRequest `
  -Uri "https://civitai.com/api/download/models/1879847?token=$token" `
  -OutFile "models\loras\illustrious\style\keyframe-animation-v1.1.safetensors"

Invoke-WebRequest `
  -Uri "https://civitai.com/api/download/models/411375?token=$token" `
  -OutFile "models\loras\pony\composition\character-sheet-v1.safetensors"
```

Civitai currently requires an authenticated download. Set `CIVITAI_API_TOKEN` in the terminal session or download the
files while signed in through a browser. A file around 80 KB beginning with `<!DOCTYPE html>` is the login page, not a
LoRA. The expected Keyframe and Character Sheet files are about 228 MB and 456 MB respectively.

## Trigger Words

| LoRA | Initial test weight | Trigger words |
| --- | ---: | --- |
| Au Ra Raen | `0.80` | `REAN, REAN3-scales-ears, type3rean-white_horns, type3rean-white_tail` |
| Keyframe Animation | `0.65` | `keyframe, genga_style, colored_pencil_sketch, rough_lineart, off_white_background, paper_texture, color trace` |
| Character Sheet | `0.25` | `multiple views, reference sheet, simple background, white background` |

The Character Sheet LoRA was trained on Pony Diffusion and failed to produce multiple views with the current
Illustrious pipeline. It remains stored but disabled. The final card is assembled by code; diffusion does not render
Chinese, Japanese, or English text.

## Publication Status

- The Illustrious model card must be reviewed before redistribution.
- The Keyframe and Au Ra Civitai pages currently allow derivatives and broad commercial use.
- Character Sheet has narrower Civitai commercial-use flags and remains local-only unless reviewed again.

These notes do not block local experiments; they preserve enough provenance for a later GitHub decision.
