# Fidelity without asking users for more screenshots

Why the current outputs get the horns/hair/scales wrong, and how to fix it **from a single user
screenshot** — without making end users supply a training set.

## The root cause

Nothing in the current generator constrains *this character's actual geometry*. Identity is
expressed only as text and generic weights, so every component fills in a race-average guess:

| Component | What it constrains | Why the character is still wrong |
| --- | --- | --- |
| Prompt tokens (`REAN`, `horns`…) | A text concept | Model draws an *average* Au Ra horn, not this pair |
| Generic Au Ra LoRA | What the race looks like | Generic horns/face, not the screenshot's |
| Single-image IP-Adapter plus-face | Face embedding | Barely touches horns, hair, scales, sunglasses (structure) |
| img2img + hi-res | Polish | Sharpens / reinvents the wrong shapes |

The true geometry already exists **in the user's screenshot**. The fix is to read it from there,
not to learn it from many images.

## The product constraint

The project's premise is **one screenshot in, zero training by the user**. So fidelity must come
from training-free, single-image methods at runtime. Heavy training is allowed only for the
**maintainer**, and only for things that *repeat across users* (a race, a gear set) — never
per-end-user-character.

## Two tiers — keep them separate

| | Maintainer (builds the DB, once) | End user (the product, one screenshot) |
| --- | --- | --- |
| Effort | May train / curate references | **Zero training, one image** |
| Asset granularity | **Race / clan + gear / glamour** (repeat across users) | Trains nothing |
| Fidelity source | Canonical references stored in the DB | Reads geometry from *their* screenshot + DB assets |

The existing `au-ra-raen-facetype3` LoRA is already a *race-level* maintainer asset — the right
pattern. We do **not** train one LoRA per user's character.

## The runtime fix (training-free): ControlNet

Read the geometry from the single screenshot and pin it:

1. **ControlNet structural conditioning** — derive **lineart/softedge** (horn shape, hairstyle
   flow) and/or **depth** (body, pose) from the screenshot, so the redraw follows the real shapes
   instead of inventing them. No training; uses only the one image.
2. Combine with: the race LoRA (anatomy correctness), IP-Adapter / PuLID (face identity), and any
   canonical gear references the DB loads once the gear is recognized.

ControlNet is the primary lever for the "horns/hair drawn wrong" complaint and fits the zero-input
premise. It is the next implementation brick (below).

## Maintainer/showcase tier: per-character LoRA (optional)

`scripts/prepare_character_dataset.py` builds a per-character LoRA dataset.
This is **not** the product path — it is an optional tool for the *maintainer* to make a
high-fidelity showcase LoRA of their own hero character (where they already own a clean reference
set). The runtime path above must work without it.

## Knowledge-DB assets

Per the product vision (see [knowledge-layer.md](knowledge-layer.md)), the DB stores reusable,
maintainer-curated assets keyed by recognizable things, not per user:

```text
content_packs/ffxiv/
  races/<race>/        race LoRA + canonical anatomy references
  gear/<set>/          glamour/gear reference images for identity injection
```

"Zero-prompt" is realized when the recognizer maps a user screenshot to race + gear, loads those
assets, and ControlNet lifts the screenshot's own geometry to a finished card.

## Brick sequence

- [x] **Dataset prep (maintainer tier)** — `scripts/prepare_character_dataset.py`. Kept as an
  optional showcase tool, not the product path.
- [x] **ControlNet structure (runtime)** — `StableDiffusionXLControlNetImg2ImgPipeline` driven by
  a control map (canny/lineart for horn/hair edges; depth for body) extracted from the user's
  screenshot. See `src/preprocessing/control_images.py`.
- [x] **Recognition-driven asset loading** — the recognized race auto-loads its curated LoRA(s)
  and optional reference image from the content pack (`assets` block in `anatomy_rules.yaml`), so
  the user no longer hand-wires LoRAs. This is the zero-prompt loop made real.
- [ ] **Stronger single-image identity** — *not* InstantID/PuLID: those are insightface-based
  realistic-face methods that fight an anime base. IP-Adapter plus-face stays the anime-appropriate
  tool; revisit only if a face still drifts.
- [x] **Gear/glamour recognition** — `src/catalog/gear_recognizer.py` matches the outfit text to a
  curated equipment entity and injects its verified `visual_prompt`/`negative_prompt`, so the DB
  backstops the look (token-level). Image injection of `reference_image` is the follow-up below.
- [ ] **Gear reference-image injection** — needs a general (non-face) IP-Adapter download + dual
  IP-Adapter wiring (face from the screenshot, appearance from the DB reference image).
- [ ] **OCR facts** — PaddleOCR on the Adventurer Plate / gear view for deterministic name/gear
  facts (parallel recognition-layer increment).

## Brick 1 — dataset prep (done)

```powershell
python scripts/prepare_character_dataset.py `
    --input-dir private_inputs/<character>/refs `
    --output-dir datasets/<character> `
    --trigger <character_token>
```

Guidance:

- **10–20 high-quality shots** beat 40 mediocre ones. Front + 3/4 + side + a couple of expressions.
  Vary pose/background; don't repeat the same composition.
- Use a **unique trigger** (not a common word) so it doesn't collide with the base model's vocab.
- Captions hold only `trigger, class-tag` so all constant traits bind to the trigger. Add only
  *varying* tags (a pose or expression) per-image `.txt` if a shot is unusual.
- Crop is on by default (frames the character); triage is advisory (curated shots are kept). Pass
  `--triage-skip` to drop blurry/dark ones automatically.

Output: `datasets/<character>/img/<repeats>_<trigger>/*.png` + `.txt`, and `train_config.toml`.

## Brick 2 — training (GPU, maintainer-run)

kohya sd-scripts is the reliable standard for SDXL/Illustrious LoRA; install it in its own venv
(keeps the project's diffusers stack clean):

```powershell
git clone https://github.com/kohya-ss/sd-scripts
# in its own venv, install its requirements + the matching torch build, then:
accelerate launch sdxl_train_network.py --config_file datasets/<character>/train_config.toml
```

The generated `train_config.toml` is pre-tuned for a character LoRA on Illustrious: `network_dim
16 / alpha 8`, `AdamW8bit`, cosine-with-restarts, ~12 epochs, bucketed at 1024. Tune epochs to
dataset size; checkpoints land in `datasets/<character>/lora/`. Pick the epoch that reproduces the
character without over-baking the pose.

## Generate with a showcase LoRA (maintainer)

Pass the trained LoRA to the generator alongside the style LoRA, and lean on the trigger word
instead of a token dump:

```powershell
python scripts/run_baseline_experiment.py `
  --lora "models\loras\illustrious\style\keyframe-animation-v1.1.safetensors=0.6" `
  --lora "datasets\<character>\lora\<trigger>-illustrious-v1.safetensors=0.85" `
  --extra-prompt "<trigger>"  # appearance comes from the LoRA, not a token dump
```

Expect to *lower* the content-term prompt weight and lean on the trigger. The race guardrail
tokens stay as a backstop.

## Brick 4 — ControlNet structure (next implementation brick)

Already have `models/controlnet/depth-sdxl`. To wire it:

- Switch base generation to `StableDiffusionXLControlNetImg2ImgPipeline`.
- Produce a control image from the source screenshot: depth (needs a depth estimator such as
  Depth-Anything-V2-small) for silhouette; add a lineart/softedge ControlNet for horn/hair edges.
- Recommended download: the Illustrious-tuned ControlNet union or lineart model for anime edges.

Depth holds the body; lineart holds the fine outlines the prompt and IP-Adapter currently miss.
