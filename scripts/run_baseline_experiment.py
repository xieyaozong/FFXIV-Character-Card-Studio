"""Screenshot -> character-card pipeline runner.

Stable milestone: screenshots -> VLM features -> CLIP-fit prompt. Run with `--features-only`
to stop after writing `features.json` and `prompt.txt`.

Work in progress: the SDXL / IP-Adapter image generation that follows is still being tuned
and is not part of the published milestone.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Style is the USER's choice (their style LoRA + trigger word + --style-prompt), never forced by
# the architecture — a photorealistic real-person LoRA must be just as valid as an anime one. So
# the default carries only style-agnostic "completion" terms: full body framing + quality, with no
# medium/look words. Override per run with --style-prompt.
STYLE_PROMPT = "best quality, highly detailed, full body"
# Negatives are style-agnostic too: only anatomy/quality defects and artifacts, never a particular
# look. (No "photorealistic"/"3d render"/"game screenshot" here — those are valid target styles.)
NEGATIVE_PROMPT = (
    "lowres, worst quality, text, watermark, signature, "
    "extra limbs, duplicate person, "
    "bad hands, malformed hands, mutated hands, extra fingers, missing fingers, fused fingers, "
    "bad feet, missing toes, fused toes, extra toes, deformed limbs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first local FFXIV character-card baseline experiment.")
    parser.add_argument(
        "--character-image",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="Character screenshot. Repeat for multiple angles — features are merged (union) across "
        "them so an occluded part (glove under smoke, horns under a hat) is filled from another shot. "
        "The FIRST image drives pose / img2img init / ControlNet.",
    )
    parser.add_argument("--weapon-image", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, default=Path("models/vlm/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--vlm-4bit", action="store_true", help="Load the VLM in 4-bit (needed to fit the 8B in 16GB).")
    parser.add_argument(
        "--sdxl-model",
        type=Path,
        default=Path("models/diffusion/stable-diffusion-xl-base-1.0"),
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("models/diffusion/stable-diffusion-xl-base-1.0"),
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="PATH=WEIGHT",
        help="Load a LoRA adapter. Repeat this option to combine adapters.",
    )
    parser.add_argument("--extra-prompt", default="")
    parser.add_argument(
        "--style-prompt",
        default=STYLE_PROMPT,
        help="Style/look layer (the user's choice). Defaults to style-agnostic completion terms; "
        "set this to drive any look — anime, photoreal, painterly — alongside the chosen style LoRA.",
    )
    parser.add_argument("--prompt-override", default="")
    parser.add_argument("--extra-negative", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/baseline-001"))
    parser.add_argument("--features-file", type=Path)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument(
        "--background-backend",
        choices=("none", "blue_screen", "rembg"),
        default="rembg",
    )
    parser.add_argument(
        "--crop-subject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop each input to the detected character before VLM and SDXL.",
    )
    parser.add_argument("--crop-backend", choices=("rembg", "blue_screen"), default="rembg")
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument(
        "--ip-adapter-scale",
        type=float,
        default=0.0,
        help="IP-Adapter identity strength (0 disables). Lets strength go high while keeping the face.",
    )
    parser.add_argument("--ip-adapter-dir", type=Path, default=Path("models/ip-adapter"))
    parser.add_argument("--ip-adapter-subfolder", default="sdxl_models")
    parser.add_argument("--ip-adapter-weight", default="ip-adapter-plus-face_sdxl_vit-h.safetensors")
    parser.add_argument(
        "--ip-adapter-image-encoder",
        type=Path,
        default=Path("models/ip-adapter/models/image_encoder"),
        help="CLIP ViT-H image encoder folder the *_vit-h adapter expects (not the bigG one).",
    )
    parser.add_argument(
        "--ip-adapter-image",
        type=Path,
        help="Reference image for IP-Adapter identity (defaults to the character face crop).",
    )
    parser.add_argument(
        "--guardrails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recognize the race and inject its lore guardrails (required tokens, forbidden negatives).",
    )
    parser.add_argument("--race-signatures", type=Path, default=Path("knowledge/ffxiv/race_signatures.yaml"))
    parser.add_argument("--anatomy-rules", type=Path, default=Path("knowledge/ffxiv/anatomy_rules.yaml"))
    parser.add_argument("--entities", type=Path, default=Path("knowledge/ffxiv/entities.yaml"))
    parser.add_argument(
        "--head-zoom-traits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Second VLM pass on a zoomed head crop to catch small horns/ears/scales the full shot misses.",
    )
    parser.add_argument(
        "--ensemble-race",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Combine VLM traits with image-embedding kNN over the race index for race recognition.",
    )
    parser.add_argument("--race-index", type=Path, default=Path("knowledge/ffxiv/race_index.npz"))
    parser.add_argument(
        "--race",
        default=None,
        help="Force a known race_id (e.g. au_ra): loads its LoRA + tokens + guardrails, skips recognition.",
    )
    parser.add_argument(
        "--auto-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-load the recognized race's curated LoRA(s) and reference image from the knowledge data.",
    )
    parser.add_argument(
        "--face-detail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-render the head region at high resolution and blend it back (face quality + repair).",
    )
    parser.add_argument("--face-detail-strength", type=float, default=0.5)
    parser.add_argument(
        "--detail-hands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repair YOLO-detected hands (img2img locked to the screenshot's correct hand silhouette).",
    )
    parser.add_argument(
        "--detail-feet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repair the feet region. Content-aware: redraws the actual footwear (boots/shoes) the "
        "character wears, or bare feet with toes if barefoot. Off by default — only enable on a real defect.",
    )
    parser.add_argument("--hand-feet-strength", type=float, default=0.55)
    parser.add_argument(
        "--hand-detector",
        type=Path,
        help="ultralytics YOLO hand model (e.g. hand_yolov8s.pt) used by --detail-hands.",
    )
    parser.add_argument(
        "--expressions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate the card's expression sheet (6 head panels with different expressions).",
    )
    parser.add_argument("--expression-strength", type=float, default=0.55)
    parser.add_argument(
        "--expression-control-scale",
        type=float,
        default=0.2,
        help="ControlNet scale for expression panels — kept low so the expression can change.",
    )
    parser.add_argument(
        "--controlnet-model",
        type=Path,
        help="ControlNet folder. Set to pin geometry (horns/hair/pose) from the screenshot.",
    )
    parser.add_argument("--controlnet-scale", type=float, default=0.6)
    parser.add_argument(
        "--slot-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slot-aware generation: a FREE base (pose/body/face) at --base-control-scale, then a HIGH-"
        "control reproduce pass on the race head. Decouples 'free pose' from 'locked race head'.",
    )
    parser.add_argument(
        "--base-control-scale",
        type=float,
        default=0.5,
        help="ControlNet scale for the free base pass when --slot-aware (lower = freer pose/body).",
    )
    parser.add_argument(
        "--slot-control-scale",
        type=float,
        default=0.85,
        help="HIGH ControlNet scale for slot reproduce passes (race head) when --slot-aware.",
    )
    parser.add_argument(
        "--control-preprocessor",
        choices=("none", "canny", "depth"),
        default="canny",
        help="How to derive the control map from the screenshot. Must match the ControlNet type.",
    )
    parser.add_argument("--depth-model", type=Path, help="Depth estimator path (for --control-preprocessor depth).")
    parser.add_argument("--control-image", type=Path, help="Precomputed control map (overrides the preprocessor).")
    parser.add_argument(
        "--hires",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upscale and lightly re-render the whole image for resolution and polish.",
    )
    parser.add_argument("--hires-scale", type=float, default=2.0)
    parser.add_argument("--hires-strength", type=float, default=0.35)
    parser.add_argument(
        "--upscaler-model",
        type=Path,
        help="ESRGAN upscaler (e.g. RealESRGAN anime) used for the hi-res upscale step; LANCZOS if unset.",
    )
    return parser.parse_args()


# Values that mean "no such feature / not shown" — these must never become prompt terms.
NON_VISIBLE_VALUES = {"none", "absent", "occluded", "not visible", "n/a", "unknown", ""}

# identity keys that name a body feature: their value is just a descriptor, so append the noun
# (e.g. {"horns": "white"} -> "white horns"). Keys not listed here carry self-describing values
# (e.g. headwear "black cap", accessory "yellow headphones") and are added as-is.
IDENTITY_NOUNS = {
    "hair_color": "hair",
    "hair": "hair",
    "skin_tone": "skin",
    "skin": "skin",
    "eye_color": "eyes",
    "eyes": "eyes",
    "horns": "horns",
    "horn_color": "horns",
    "ears": "ears",
    "tail": "tail",
    "tail_color": "tail",
    "scales": "scales",
    "glasses": "sunglasses",
    "glasses_color": "sunglasses",
}

# Garment nouns: if an outfit value already contains one, it names its own garment, so we don't
# append the slot key (avoids "black t-shirt ... top").
GARMENT_WORDS = (
    "shirt", "t-shirt", "tee", "jacket", "coat", "hoodie", "sweater", "dress", "skirt", "shorts",
    "pants", "trousers", "jeans", "leggings", "stockings", "glove", "boot", "shoe", "sock", "top",
    "cape", "cloak", "robe", "armor", "armour", "scarf", "belt", "vest", "suit", "bodysuit",
)


def is_visible(value: str) -> bool:
    normalized = value.strip().strip(".").lower().replace("_", " ")
    return normalized not in NON_VISIBLE_VALUES and "not visible" not in normalized


def primary_clause(value: str) -> str:
    """Keep the leading descriptor and drop positional notes like "visible behind right thigh"."""
    return value.split(",")[0].strip()


def content_terms(features: dict[str, object]) -> list[str]:
    """Flatten the VLM's identity + outfit lists into prompt terms.

    The VLM now keys identity by feature (hair_color, horns, glasses, headwear, accessory, ...) and
    outfit by garment slot (jacket, top, shorts, boots, ...), with self-describing values. So we
    iterate every entry rather than looking up fixed keys: body-feature keys get their noun appended,
    everything else (which already names itself) is added verbatim. Identity comes first so CLIP
    trimming drops garments before core identity.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip().strip(",").strip()
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)

    for item in features.get("identity", []):
        value = str(item["value"])
        if not is_visible(value):
            continue
        value = primary_clause(value)
        if not is_visible(value):
            continue
        noun = IDENTITY_NOUNS.get(str(item["key"]).lower())
        add(value if not noun or noun in value.lower() else f"{value} {noun}")

    for item in features.get("outfit", []):
        value = str(item["value"])
        if not is_visible(value):
            continue
        value = primary_clause(value)
        # Terse values like shorts="gray" lose the garment noun; append the slot key when the value
        # names no garment itself (but not when it already says t-shirt/jacket/etc.).
        key = str(item["key"]).lower()
        if not any(word in value.lower() for word in GARMENT_WORDS):
            value = f"{value} {key}".strip()
        add(value)

    return terms


def build_prompt(features: dict[str, object], extra_prompt: str = "") -> str:
    body = ", ".join(content_terms(features))
    return ", ".join(part for part in (extra_prompt, STYLE_PROMPT, body) if part)


def fit_prompt_to_clip(prompt: str, tokenizer, max_tokens: int = 77) -> str:
    """Greedily keep leading comma fragments so CLIP never silently truncates the tail."""
    fragments = [fragment for fragment in prompt.split(", ") if fragment]
    kept: list[str] = []
    for fragment in fragments:
        candidate = ", ".join([*kept, fragment])
        # verbose=False: probing over-length candidates is expected, so silence the CLIP
        # "token indices longer than 77" warning — the returned prompt is always within budget.
        if len(tokenizer(candidate, truncation=False, verbose=False).input_ids) > max_tokens:
            break
        kept.append(fragment)
    return ", ".join(kept)


def parse_lora_specs(values: list[str]) -> list[tuple[Path, float]]:
    specs: list[tuple[Path, float]] = []
    for value in values:
        try:
            path_text, weight_text = value.rsplit("=", 1)
            path = Path(path_text).resolve()
            weight = float(weight_text)
        except ValueError as error:
            raise ValueError(f"Invalid LoRA specification: {value!r}. Use PATH=WEIGHT.") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append((path, weight))
    return specs


def load_upscaler(model_path: Path):
    """Load an ESRGAN-family upscaler via spandrel (clean loader, modern-torch compatible)."""
    try:
        from spandrel import ModelLoader
    except ImportError as exc:
        raise RuntimeError("pip install spandrel to use --upscaler-model.") from exc
    import torch

    return ModelLoader().load_from_file(str(model_path)).to("cuda" if torch.cuda.is_available() else "cpu").eval()


def esrgan_upscale(image, upscaler, target: tuple[int, int]):
    """Run the upscaler (e.g. RealESRGAN anime) then resize to the exact target — crisp anime lines."""
    import numpy as np
    import torch

    device = next(upscaler.model.parameters()).device
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = upscaler(tensor).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    upscaled = Image.fromarray((output * 255).round().astype("uint8"))
    return upscaled.resize(target, Image.Resampling.LANCZOS)


def hires_fix(
    pipeline, image, *, prompt, negative, ip_image, control_image, control_scale,
    scale, strength, steps, guidance, generator, upscaler=None,
):
    """Upscale the whole image and lightly re-render it for higher resolution and polish."""
    width, height = image.size
    target = (max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8))
    upscaled = esrgan_upscale(image, upscaler, target) if upscaler is not None else image.resize(
        target, Image.Resampling.LANCZOS
    )
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative,
        "image": upscaled,
        "strength": strength,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if ip_image is not None:
        kwargs["ip_adapter_image"] = ip_image
    if control_image is not None:
        kwargs["control_image"] = control_image.resize(target, Image.Resampling.LANCZOS)
        kwargs["controlnet_conditioning_scale"] = control_scale
    return pipeline(**kwargs).images[0]


def detail_region(
    pipeline, image, box, *, prompt, negative, ip_image, control_image, control_scale,
    strength, steps, guidance, generator, upscale_to=1024,
):
    """Crop a region, re-render it at high resolution, and feather it back (adetailer-style)."""
    box = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
    crop = image.crop(box)
    if min(crop.size) < 16:
        return image
    scale = upscale_to / max(crop.size)
    target = (max(8, round(crop.size[0] * scale / 8) * 8), max(8, round(crop.size[1] * scale / 8) * 8))

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative,
        "image": crop.resize(target, Image.Resampling.LANCZOS),
        "strength": strength,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if ip_image is not None:
        kwargs["ip_adapter_image"] = ip_image
    if control_image is not None:
        region_control = control_image.resize(image.size, Image.Resampling.LANCZOS).crop(box)
        kwargs["control_image"] = region_control.resize(target, Image.Resampling.LANCZOS)
        kwargs["controlnet_conditioning_scale"] = control_scale
    detailed = pipeline(**kwargs).images[0].resize(crop.size, Image.Resampling.LANCZOS)

    mask = Image.new("L", crop.size, 0)
    margin_x = max(1, crop.size[0] // 10)
    margin_y = max(1, crop.size[1] // 10)
    mask.paste(255, (margin_x, margin_y, crop.size[0] - margin_x, crop.size[1] - margin_y))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(margin_x, margin_y) / 1.5))

    result = image.copy()
    result.paste(detailed, box[:2], mask)
    return result


def detail_face(pipeline, image, **kwargs):
    """Re-render the head region (race anatomy + face quality). Thin wrapper over detail_region."""
    width, height = image.size
    box = (round(width * 0.16), 0, round(width * 0.84), round(height * 0.44))
    return detail_region(pipeline, image, box, **kwargs)


def detect_boxes(image, model_path, *, conf=0.3, pad=0.18):
    """Detect object boxes (e.g. hands) with an ultralytics YOLO model; padded, original coords."""
    from ultralytics import YOLO

    result = YOLO(str(model_path)).predict(image, conf=conf, verbose=False)[0]
    boxes = []
    for x0, y0, x1, y1 in result.boxes.xyxy.cpu().numpy():
        bw, bh = x1 - x0, y1 - y0
        boxes.append((int(x0 - bw * pad), int(y0 - bh * pad), int(x1 + bw * pad), int(y1 + bh * pad)))
    return boxes


def make_inpaint_pipeline(pipeline):
    """Build an SDXL inpaint pipeline from an existing pipeline's components (no extra VRAM).

    Reuses the already-loaded VAE/UNet/text-encoders, so the inpaint pass costs nothing extra to
    load. Deliberately drops the ControlNet: the feet area has no usable edges in the control map
    (bare feet on a pale floor), so conditioning on it just reproduces the "nothing there" stumps.
    Inpaint lets the model repaint that region freely from the surrounding leg context instead.
    """
    from diffusers import StableDiffusionXLInpaintPipeline

    return StableDiffusionXLInpaintPipeline(
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        text_encoder_2=pipeline.text_encoder_2,
        tokenizer=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
    )


def inpaint_region(
    inpaint_pipeline, image, box, mask_box, *, prompt, negative,
    strength, steps, guidance, generator, upscale_to=1024,
):
    """Crop a region, repaint only the masked sub-area from scratch, feather it back.

    Unlike detail_region (img2img, which preserves the init and so cannot add what is missing),
    this masks `mask_box` and inpaints it, so missing feet/toes are drawn anew. `box` carries
    surrounding context (the legs) that the model grows the anatomy from; `mask_box` (relative to
    `box`) is the part actually repainted.
    """
    box = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
    crop = image.crop(box)
    if min(crop.size) < 16:
        return image
    scale = upscale_to / max(crop.size)
    target = (max(8, round(crop.size[0] * scale / 8) * 8), max(8, round(crop.size[1] * scale / 8) * 8))
    crop_up = crop.resize(target, Image.Resampling.LANCZOS)

    # Mask (in crop coordinates): white = repaint, feathered so the seam with the legs is soft.
    mask = Image.new("L", crop.size, 0)
    mask.paste(255, (mask_box[0], mask_box[1], mask_box[2], mask_box[3]))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(crop.size) // 24 + 1))
    mask_up = mask.resize(target, Image.Resampling.LANCZOS)

    repainted = inpaint_pipeline(
        prompt=prompt,
        negative_prompt=negative,
        image=crop_up,
        mask_image=mask_up,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    ).images[0].resize(crop.size, Image.Resampling.LANCZOS)

    result = image.copy()
    result.paste(repainted, box[:2], mask)
    return result


def detail_hands_feet(
    pipeline, image, *, base_prompt, negative, ip_image, control_image, control_scale,
    strength, steps, guidance, generator, tokenizer, hand_detector=None,
    repair_hands=True, repair_feet=False, footwear=None, glove=None,
):
    """Content-aware anatomy repair: detected hands (img2img) and, optionally, the feet (inpaint).

    Each pass is independent and opt-in so a good region is never "repaired" into a worse one:
    - Hands stay on img2img locked to the screenshot's correct hand edges with a HIGH ControlNet
      scale, so a badly-drawn hand is pulled back to the real silhouette while keeping its grip on a
      weapon (a full repaint would drop the weapon). The prompt names the actual glove if worn.
    - Feet use a control-free inpaint (the feet area has no usable control edges). It is content-aware:
      it redraws the ACTUAL footwear the character wears (e.g. boots) rather than assuming bare feet,
      so it does not turn boots into toed stumps. Off by default; enable only on a real foot defect.
    """
    out = image
    width, height = image.size
    # Lock to the screenshot's correct hand silhouette (much higher than the base scale).
    lock_scale = min(1.0, max(control_scale, 0.85)) if control_image is not None else control_scale

    if repair_hands and hand_detector is not None and Path(hand_detector).is_file():
        hand_desc = f"detailed {glove}" if glove else "detailed bare hand"
        hand_prompt = fit_prompt_to_clip(
            f"{hand_desc}, five fingers, anatomically correct hand gripping, {base_prompt}", tokenizer
        )
        for box in detect_boxes(out, hand_detector):
            out = detail_region(
                pipeline, out, box, prompt=hand_prompt, negative=negative, ip_image=ip_image,
                control_image=control_image, control_scale=lock_scale, strength=strength,
                steps=steps, guidance=guidance, generator=generator, upscale_to=768,
            )

    if repair_feet:
        # Box carries the lower legs as context; only the lower portion is masked and repainted, so
        # the model grows the footwear/feet off the leg above. Describe the real footwear so a booted
        # character stays booted.
        feet_desc = footwear if footwear else "bare feet with five toes, anatomically correct feet"
        feet_box = (round(width * 0.14), round(height * 0.58), round(width * 0.86), height)
        fb_w, fb_h = feet_box[2] - feet_box[0], feet_box[3] - feet_box[1]
        feet_mask = (round(fb_w * 0.04), round(fb_h * 0.45), round(fb_w * 0.96), fb_h)
        feet_prompt = fit_prompt_to_clip(
            f"{feet_desc}, ankles, legs, feet on the ground, plain white background, {base_prompt}",
            tokenizer,
        )
        inpaint_pipeline = make_inpaint_pipeline(pipeline)
        out = inpaint_region(
            inpaint_pipeline, out, feet_box, feet_mask, prompt=feet_prompt, negative=negative,
            strength=max(strength, 0.9), steps=steps, guidance=guidance, generator=generator,
            upscale_to=1024,
        )

    return out


# Card expression sheet: label -> English expression phrase injected into the head prompt.
EXPRESSIONS = [
    ("開心", "happy, bright smile, open mouth"),
    ("生氣", "angry, furrowed brow, pouting"),
    ("愛睏", "sleepy, half-closed drowsy eyes"),
    ("害羞", "shy, blushing, looking away"),
    ("偷笑", "smug smirk, sly grin"),
    ("無言", "blank expressionless deadpan stare"),
]


def generate_expressions(
    pipeline, image, *, base_prompt, negative, ip_image, control_image, control_scale,
    strength, steps, guidance, generator, tokenizer,
):
    """Re-render the head with each expression prompt (IP-Adapter holds identity, ControlNet kept
    low so the expression can actually change). Returns [(label, head_image), ...]."""
    width, height = image.size
    # Tight face crop (not the wider face-detail box): the expression must read clearly.
    box = (round(width * 0.30), 0, round(width * 0.70), round(height * 0.30))
    crop = image.crop(box)
    scale = 1024 / max(crop.size)
    target = (max(8, round(crop.size[0] * scale / 8) * 8), max(8, round(crop.size[1] * scale / 8) * 8))
    head = crop.resize(target, Image.Resampling.LANCZOS)
    head_control = (
        control_image.resize(image.size, Image.Resampling.LANCZOS).crop(box).resize(target, Image.Resampling.LANCZOS)
        if control_image is not None
        else None
    )

    panels = []
    for label, phrase in EXPRESSIONS:
        prompt = fit_prompt_to_clip(f"portrait, detailed face, {phrase}, {base_prompt}", tokenizer)
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "image": head,
            "strength": strength,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": generator,
        }
        if ip_image is not None:
            kwargs["ip_adapter_image"] = ip_image
        if head_control is not None:
            kwargs["control_image"] = head_control
            kwargs["controlnet_conditioning_scale"] = control_scale
        panels.append((label, pipeline(**kwargs).images[0].resize(crop.size, Image.Resampling.LANCZOS)))
    return panels


def main() -> None:
    from src.preprocessing.background import remove_background
    from src.preprocessing.crops import build_review_crops
    from src.preprocessing.triage import crop_subject, triage_image
    from src.vlm.qwen_backend import QwenVLMBackend

    args = parse_args()
    started = perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    character_images = [Image.open(path).convert("RGB") for path in args.character_image]
    weapon_image = Image.open(args.weapon_image).convert("RGB")

    crop_info: dict[str, object] = {}
    if args.crop_subject:
        weapon_result = triage_image(weapon_image, mask_backend=args.crop_backend)
        weapon_image = crop_subject(weapon_image, weapon_result, pad_ratio=args.crop_pad)
        weapon_image.save(output_dir / "input_weapon_crop.png")
        crop_info["weapon"] = {
            "usable": weapon_result.usable, "score": weapon_result.score,
            "reasons": weapon_result.reasons, "bbox": weapon_result.bbox,
        }
        cropped_characters = []
        for index, image in enumerate(character_images):
            result = triage_image(image, mask_backend=args.crop_backend)
            cropped = crop_subject(image, result, pad_ratio=args.crop_pad)
            suffix = "" if index == 0 else f"_{index}"
            cropped.save(output_dir / f"input_character{suffix}_crop.png")
            crop_info[f"character{suffix}"] = {
                "usable": result.usable, "score": result.score,
                "reasons": result.reasons, "bbox": result.bbox,
            }
            cropped_characters.append(cropped)
        character_images = cropped_characters

    # The first image is the primary: it drives the img2img init + ControlNet (pose/structure).
    character_image = character_images[0]
    foreground = remove_background(character_image, args.background_backend)
    foreground.save(output_dir / "input_foreground.png")
    white_background = Image.new("RGBA", foreground.size, "white")
    composited_image = Image.alpha_composite(white_background, foreground).convert("RGB")

    feature_started = perf_counter()
    if args.features_file:
        features = json.loads(args.features_file.read_text(encoding="utf-8"))
        load_seconds = 0.0
        analysis_seconds = 0.0
    else:
        load_started = perf_counter()
        vlm = QwenVLMBackend(str(args.vlm_model.resolve()), max_new_tokens=1024, load_in_4bit=args.vlm_4bit)
        load_seconds = perf_counter() - load_started
        analysis_started = perf_counter()
        raw_per_image: list[str | None] = []
        try:
            from src.domain.models import RaceTraits
            from src.vlm.feature_merger import merge_feature_responses, merge_head_traits

            # One pass per angle (weapon stays as image_2 context each time), then union the readings
            # so a part hidden in one shot is recovered from another.
            responses = []
            for image in character_images:
                responses.append(vlm.analyze([image, weapon_image]))
                raw_per_image.append(vlm.last_raw_response)
            merged_response = merge_feature_responses(responses)
            features = merged_response.model_dump(mode="json")
            if len(character_images) > 1:
                print(f"merged features from {len(character_images)} character images")
            if args.head_zoom_traits:
                cw, ch = character_image.size
                # Keep near-full width so side-protruding horns/fin-ears are not cropped off.
                head_crop = character_image.crop((round(cw * 0.05), 0, round(cw * 0.95), round(ch * 0.45)))
                head_crop.save(output_dir / "head_zoom.png")
                head_traits = vlm.extract_traits(head_crop)
                full_traits = RaceTraits.model_validate(features.get("traits", {}))
                merged = merge_head_traits(full_traits, head_traits)
                features["traits"] = merged.model_dump(mode="json")
                print(
                    f"head-zoom traits: horns={head_traits.horns} ear={head_traits.ear_type} "
                    f"scales={head_traits.scales} face={head_traits.face_type} "
                    f"(full had horns={full_traits.horns})"
                )
        finally:
            # Always dump the raw VLM text (pre-parse), even if validation failed, so a bad
            # response can be diagnosed without re-running the model.
            raw_dump = {"feature_extraction_per_image": raw_per_image, "head_traits": vlm.last_raw_traits}
            (output_dir / "vlm_raw.json").write_text(
                json.dumps(raw_dump, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        analysis_seconds = perf_counter() - analysis_started
        del vlm
        gc.collect()
    feature_seconds = perf_counter() - feature_started
    (output_dir / "features.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    negative_prompt = ", ".join(part for part in (NEGATIVE_PROMPT, args.extra_negative) if part)
    race_id: str | None = None
    gear_id: str | None = None
    gear_reference: str | None = None
    auto_ip_adapter_image: Path | None = None

    # Gear recognition: match the outfit to a curated equipment set and inject its verified
    # appearance tokens, so the DB backstops the look instead of the user prompting it.
    terms = content_terms(features)
    if not args.prompt_override.strip() and args.entities.is_file():
        from src.knowledge.entities import EntityStore
        from src.knowledge.gear import recognize_gear

        gear_match = recognize_gear(terms, EntityStore.load(args.entities))
        if gear_match:
            gear = gear_match.record
            gear_id = gear.canonical_id
            gear_reference = gear.reference_image or None
            if gear.visual_prompt:
                terms = [gear.visual_prompt, *terms]
            if gear.negative_prompt:
                negative_prompt = ", ".join(part for part in (negative_prompt, *gear.negative_prompt) if part)
            print(f"recognized gear: {gear_id} (matched: {', '.join(gear_match.matched)})")

    # Anchor the held weapon in the prompt (a reproduction target). Done after gear matching so the
    # weapon description never perturbs gear recognition.
    weapon = features.get("weapon") or {}
    if weapon.get("include"):
        for candidate in weapon.get("candidates", []):
            value = primary_clause(str(candidate.get("value", "")))
            if is_visible(value) and value.casefold() not in {t.casefold() for t in terms}:
                terms.append(value)

    guardrails_on = (
        args.guardrails
        and not args.prompt_override.strip()
        and args.race_signatures.is_file()
        and args.anatomy_rules.is_file()
    )
    if args.prompt_override.strip():
        prompt = args.prompt_override.strip()
    elif guardrails_on:
        import yaml

        from src.domain.models import RaceTraits
        from src.knowledge.races import load_race_signatures
        from src.prompting.plan import compile_generation_plan

        anatomy_rules_data = yaml.safe_load(args.anatomy_rules.read_text(encoding="utf-8")) or {}
        signatures = load_race_signatures(args.race_signatures)
        traits_obj = RaceTraits.model_validate(features.get("traits", {}))

        # Race resolution. --race forces a known race (loads its LoRA + tokens + guardrails even when
        # recognition is uncertain), so a known character's special features are never left to a
        # silently-dropped LoRA. Otherwise fall back to the VLM+embedding ensemble.
        ensemble_race: str | None = None
        ensemble_used = False
        if args.race:
            ensemble_race = args.race
            ensemble_used = True
            print(f"forced race: {ensemble_race}")
        elif args.ensemble_race and args.race_index.is_file():
            from src.knowledge.race_index import ClipEmbedder, RaceIndex, head_region, on_white
            from src.knowledge.race_matcher import recognize_race_ensemble

            embedder = ClipEmbedder(args.ip_adapter_image_encoder)
            head = on_white(head_region(character_image))
            head.save(output_dir / "race_query.png")
            ensemble = recognize_race_ensemble(
                traits_obj, signatures, embedding=embedder.embed(head), index=RaceIndex.load(args.race_index)
            )
            ensemble_race = ensemble.race_id
            ensemble_used = True
            del embedder
            gc.collect()
            print(f"ensemble race: {ensemble_race} conf={ensemble.confidence} ({'; '.join(ensemble.reasons)})")

        plan = compile_generation_plan(
            content_terms=terms,
            style_prompt=args.style_prompt,
            base_negative=negative_prompt,
            traits=traits_obj,
            race_signatures=signatures,
            anatomy_rules=anatomy_rules_data,
            extra_prompt=args.extra_prompt,
            race_id=ensemble_race,
            recognized=ensemble_used,
            base_control_scale=args.base_control_scale,
            slot_control_scale=args.slot_control_scale,
        )
        prompt = plan.positive_prompt
        negative_prompt = plan.negative_prompt
        race_id = plan.race_id
        (output_dir / "constraints.json").write_text(
            json.dumps(
                {"race_id": race_id, "gear_id": gear_id, "gear_reference": gear_reference, **plan.constraints},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # Save the compiled slot plan for transparency (the diffusion-facing decision record).
        (output_dir / "plan.json").write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"recognized race: {race_id}")

        # Recognition drives generation: auto-load the race's curated LoRA(s) + reference image.
        if args.auto_assets and race_id:
            from src.knowledge.assets import resolve_race_assets

            assets = resolve_race_assets(race_id, anatomy_rules_data)
            existing = {Path(spec_str.rsplit("=", 1)[0]).resolve() for spec_str in args.lora}
            for lora_spec in assets["loras"]:
                if Path(lora_spec.rsplit("=", 1)[0]).resolve() not in existing:
                    args.lora.append(lora_spec)
                    print(f"  auto-loaded race LoRA: {lora_spec}")
            reference = assets["ip_adapter_image"]
            if reference and not args.ip_adapter_image and Path(reference).is_file():
                auto_ip_adapter_image = Path(reference)
                print(f"  auto-loaded reference image: {reference}")
    else:
        body = ", ".join(terms)
        prompt = ", ".join(part for part in (args.extra_prompt, args.style_prompt, body) if part)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if args.features_only:
        timings = {
            "load_seconds": load_seconds,
            "analysis_seconds": analysis_seconds,
            "feature_seconds": feature_seconds,
            "total_seconds": perf_counter() - started,
        }
        (output_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        print(json.dumps(timings))
        return

    import torch

    torch.cuda.empty_cache()
    # Pad (letterbox), not fit/crop: a tight subject crop is tall, and fit would cut off head and feet.
    input_image = ImageOps.pad(
        composited_image,
        (768, 1024),
        method=Image.Resampling.LANCZOS,
        color="white",
        centering=(0.5, 0.5),
    )
    input_image.save(output_dir / "input_character.png")

    # ControlNet: pin the screenshot's real geometry (horns, hair, pose) so the redraw follows it.
    control_image = None
    if args.controlnet_model:
        if args.control_image:
            control_image = Image.open(args.control_image).convert("RGB")
        elif args.control_preprocessor == "canny":
            from src.preprocessing.control_images import build_control_image

            # Canny from the WITH-BACKGROUND crop: background removal erases a held weapon / the hand
            # gripping it (glow + smoke confuse it), which then has no control edges and gets
            # reinvented. The dilated foreground silhouette keeps the subject + adjacent weapon edges
            # while dropping distant scene clutter.
            control_source = ImageOps.pad(
                character_image, (768, 1024), method=Image.Resampling.LANCZOS,
                color="white", centering=(0.5, 0.5),
            )
            keep_mask = ImageOps.pad(
                foreground.split()[-1], (768, 1024), method=Image.Resampling.LANCZOS,
                color=0, centering=(0.5, 0.5),
            )
            control_image = build_control_image(control_source, "canny", keep_mask=keep_mask)
        else:
            from src.preprocessing.control_images import build_control_image

            control_image = build_control_image(
                input_image,
                args.control_preprocessor,
                depth_model=args.depth_model,
                device="cuda",
            )
        control_image = control_image.resize(input_image.size, Image.Resampling.LANCZOS)
        control_image.save(output_dir / "control_image.png")

    generation_started = perf_counter()
    model_path = args.sdxl_model.resolve()
    if args.controlnet_model:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        try:
            controlnet = ControlNetModel.from_pretrained(
                str(args.controlnet_model.resolve()),
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=True,
            )
        except Exception:
            # Models without an fp16 variant (e.g. xinsir canny) load from the plain weights.
            controlnet = ControlNetModel.from_pretrained(
                str(args.controlnet_model.resolve()),
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )
        pipeline_cls = StableDiffusionXLControlNetImg2ImgPipeline
        extra_kwargs = {"controlnet": controlnet}
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipeline_cls = StableDiffusionXLImg2ImgPipeline
        extra_kwargs = {}

    if model_path.is_file():
        pipeline = pipeline_cls.from_single_file(
            str(model_path),
            config=str(args.pipeline_config.resolve()),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
            **extra_kwargs,
        )
    else:
        pipeline = pipeline_cls.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
            **extra_kwargs,
        )

    lora_specs = parse_lora_specs(args.lora)
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    for index, (lora_path, weight) in enumerate(lora_specs):
        adapter_name = f"adapter_{index}"
        pipeline.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name=adapter_name,
            local_files_only=True,
        )
        adapter_names.append(adapter_name)
        adapter_weights.append(weight)
    if adapter_names:
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

    # IP-Adapter injects face identity from a reference, so we can redraw at high strength
    # without losing the face.
    ip_adapter_image = None
    if args.ip_adapter_scale > 0:
        if args.ip_adapter_image:
            ip_adapter_image = Image.open(args.ip_adapter_image).convert("RGB")
        elif auto_ip_adapter_image is not None:
            ip_adapter_image = Image.open(auto_ip_adapter_image).convert("RGB")
        else:
            ip_adapter_image = build_review_crops(composited_image)["face"]
        ip_adapter_image.save(output_dir / "ip_adapter_image.png")
        # The *_vit-h adapter needs the ViT-H encoder (1280-dim), not the bigG one shipped
        # under sdxl_models. Load it explicitly and register it so to("cuda") moves it too.
        from transformers import CLIPVisionModelWithProjection

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(args.ip_adapter_image_encoder.resolve()),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipeline.register_modules(image_encoder=image_encoder)
        pipeline.load_ip_adapter(
            str(args.ip_adapter_dir.resolve()),
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight,
            image_encoder_folder=None,
        )
        pipeline.set_ip_adapter_scale(args.ip_adapter_scale)

    pipeline = pipeline.to("cuda")
    pipeline.vae.enable_slicing()
    prompt_used = fit_prompt_to_clip(prompt, pipeline.tokenizer)
    negative_used = fit_prompt_to_clip(negative_prompt, pipeline.tokenizer)
    if prompt_used != prompt:
        (output_dir / "prompt.txt").write_text(prompt_used, encoding="utf-8")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    generation_kwargs = {
        "prompt": prompt_used,
        "negative_prompt": negative_used,
        "image": input_image,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.steps,
        "generator": generator,
    }
    if ip_adapter_image is not None:
        generation_kwargs["ip_adapter_image"] = ip_adapter_image
    # Slot-aware control: a FREE base/hires (low control -> pose/body/face free) and a HIGH-control
    # race-head reproduce pass (locks horns/scales/head shape). Off => one global scale as before.
    base_control = args.base_control_scale if args.slot_aware else args.controlnet_scale
    head_control = args.slot_control_scale if args.slot_aware else args.controlnet_scale
    if args.slot_aware:
        print(f"slot-aware: base control={base_control}, race-head control={head_control}")
    if control_image is not None:
        generation_kwargs["control_image"] = control_image
        generation_kwargs["controlnet_conditioning_scale"] = base_control
    output_image = pipeline(**generation_kwargs).images[0]
    output_image.save(output_dir / "result_base.png")
    upscaler = load_upscaler(args.upscaler_model.resolve()) if args.upscaler_model else None
    if args.hires:
        output_image = hires_fix(
            pipeline,
            output_image,
            prompt=prompt_used,
            negative=negative_used,
            ip_image=ip_adapter_image,
            control_image=control_image,
            control_scale=base_control,
            scale=args.hires_scale,
            strength=args.hires_strength,
            steps=args.steps,
            guidance=args.guidance_scale,
            generator=generator,
            upscaler=upscaler,
        )
    if args.face_detail:
        face_prompt = fit_prompt_to_clip("portrait, detailed face, " + prompt_used, pipeline.tokenizer)
        output_image = detail_face(
            pipeline,
            output_image,
            prompt=face_prompt,
            negative=negative_used,
            ip_image=ip_adapter_image,
            control_image=control_image,
            control_scale=head_control,
            strength=args.face_detail_strength,
            steps=args.steps,
            guidance=args.guidance_scale,
            generator=generator,
        )
    if args.detail_hands or args.detail_feet:
        # Pull the actual footwear / glove descriptions from the recognized features so the repair
        # redraws what the character really wears (boots stay boots) instead of a hardcoded guess.
        garment_terms = content_terms(features)
        footwear = next(
            (t for t in garment_terms if any(w in t.lower() for w in ("boot", "shoe", "sandal", "heel", "sneaker"))),
            None,
        )
        glove = next((t for t in garment_terms if "glove" in t.lower()), None)
        output_image = detail_hands_feet(
            pipeline,
            output_image,
            base_prompt=prompt_used,
            negative=negative_used,
            ip_image=ip_adapter_image,
            control_image=control_image,
            control_scale=base_control,
            strength=args.hand_feet_strength,
            steps=args.steps,
            guidance=args.guidance_scale,
            generator=generator,
            tokenizer=pipeline.tokenizer,
            hand_detector=args.hand_detector,
            repair_hands=args.detail_hands,
            repair_feet=args.detail_feet,
            footwear=footwear,
            glove=glove,
        )
    output_image.save(output_dir / "result.png")

    if args.expressions:
        expression_dir = output_dir / "expressions"
        expression_dir.mkdir(parents=True, exist_ok=True)
        panels = generate_expressions(
            pipeline,
            output_image,
            base_prompt=prompt_used,
            negative=negative_used,
            ip_image=ip_adapter_image,
            control_image=control_image,
            control_scale=args.expression_control_scale,
            strength=args.expression_strength,
            steps=args.steps,
            guidance=args.guidance_scale,
            generator=generator,
            tokenizer=pipeline.tokenizer,
        )
        for label, panel in panels:
            panel.save(expression_dir / f"{label}.png")
        print(f"expressions: {', '.join(label for label, _ in panels)} -> {expression_dir}")

    generation_seconds = perf_counter() - generation_started

    metadata = {
        "character_image": [str(path.resolve()) for path in args.character_image],
        "weapon_image": str(args.weapon_image.resolve()),
        "vlm_model": str(args.vlm_model.resolve()),
        "sdxl_model": str(args.sdxl_model.resolve()),
        "prompt": prompt,
        "prompt_used": prompt_used,
        "negative_prompt": negative_prompt,
        "loras": [
            {"path": str(path), "weight": weight}
            for path, weight in lora_specs
        ],
        "seed": args.seed,
        "strength": args.strength,
        "background_backend": args.background_backend,
        "ip_adapter_scale": args.ip_adapter_scale,
        "ip_adapter_weight": args.ip_adapter_weight if args.ip_adapter_scale > 0 else None,
        "controlnet_model": str(args.controlnet_model.resolve()) if args.controlnet_model else None,
        "controlnet_scale": args.controlnet_scale if args.controlnet_model else None,
        "control_preprocessor": args.control_preprocessor if args.controlnet_model else None,
        "race_id": race_id,
        "gear_id": gear_id,
        "gear_reference": gear_reference,
        "upscaler_model": str(args.upscaler_model.resolve()) if args.upscaler_model else None,
        "crop_subject": args.crop_subject,
        "crop": crop_info,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "width": 768,
        "height": 1024,
        "load_seconds": load_seconds,
        "analysis_seconds": analysis_seconds,
        "feature_seconds": feature_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": perf_counter() - started,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
