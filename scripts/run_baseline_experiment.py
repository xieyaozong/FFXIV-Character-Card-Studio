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


STYLE_PROMPT = (
    "hand-drawn anime character sheet, black ink linework, colored pencil, white paper, full body"
)
NEGATIVE_PROMPT = (
    "photorealistic, 3d render, game screenshot, scenery, text, watermark, "
    "extra limbs, duplicate person, "
    "bad hands, malformed hands, mutated hands, extra fingers, missing fingers, fused fingers, "
    "bad feet, missing toes, fused toes, extra toes, deformed limbs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first local FFXIV character-card baseline experiment.")
    parser.add_argument("--character-image", type=Path, required=True)
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
        "--detail-hands-feet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-render feet (fixed region) and detected hands to fix SDXL anatomy errors.",
    )
    parser.add_argument("--hand-feet-strength", type=float, default=0.55)
    parser.add_argument(
        "--hand-detector",
        type=Path,
        help="ultralytics YOLO hand model (e.g. hand_yolov8s.pt); without it only feet are fixed.",
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


def is_visible(value: str) -> bool:
    normalized = value.lower().replace("_", " ")
    return "not visible" not in normalized


def primary_clause(value: str) -> str:
    """Keep the leading descriptor and drop positional notes like "visible behind right thigh"."""
    return value.split(",")[0].strip()


def content_terms(features: dict[str, object]) -> list[str]:
    identity = {item["key"]: str(item["value"]) for item in features["identity"]}
    outfit = {item["key"]: str(item["value"]) for item in features["outfit"]}

    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip().strip(",").strip()
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)

    # Core identity first so any later trimming drops the least important details.
    labels = {
        "hair_color": "hair",
        "horns": "horns",
        "horn_color": "horns",
        "tail": "tail",
        "tail_color": "tail",
        "glasses": "sunglasses",
        "glasses_color": "sunglasses",
        "skin": "skin",
        "skin_color": "skin",
    }
    for key, label in labels.items():
        value = identity.get(key)
        if value and is_visible(value):
            value = primary_clause(value)
            add(value if label in value.lower() else f"{value} {label}")

    # The VLM may file headwear/glasses under identity or outfit; check both.
    headwear = identity.get("headwear") or outfit.get("headwear")
    if headwear and is_visible(headwear):
        add(primary_clause(headwear))
    glasses = outfit.get("glasses")
    if glasses and is_visible(glasses):
        glasses = primary_clause(glasses)
        add(glasses if "glass" in glasses.lower() else f"{glasses} sunglasses")

    # Construction already names colors, so use it and skip the redundant color summary.
    construction = outfit.get("clothing_construction")
    colors = outfit.get("clothing_colors") or outfit.get("clothing_color")
    if construction and is_visible(construction):
        for piece in construction.split(","):
            add(piece)
    elif colors and is_visible(colors):
        add(f"{colors} outfit")

    accessories = outfit.get("accessories", "")
    for item in accessories.split(","):
        if any(name in item.lower() for name in ("headphone", "glove")):
            add(item)

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


def detail_hands_feet(
    pipeline, image, *, base_prompt, negative, ip_image, control_image, control_scale,
    strength, steps, guidance, generator, tokenizer, hand_detector=None,
):
    """Fix the worst SDXL anatomy: re-render feet (fixed bottom region) and detected hands."""
    out = image
    width, height = image.size
    feet_box = (round(width * 0.20), round(height * 0.82), round(width * 0.80), height)
    feet_prompt = fit_prompt_to_clip(f"detailed feet, anatomically correct toes, {base_prompt}", tokenizer)
    out = detail_region(
        pipeline, out, feet_box, prompt=feet_prompt, negative=negative, ip_image=ip_image,
        control_image=control_image, control_scale=control_scale, strength=strength,
        steps=steps, guidance=guidance, generator=generator, upscale_to=768,
    )
    if hand_detector is not None and Path(hand_detector).is_file():
        hand_prompt = fit_prompt_to_clip(
            f"detailed hand, five fingers, anatomically correct hand, {base_prompt}", tokenizer
        )
        for box in detect_boxes(out, hand_detector):
            out = detail_region(
                pipeline, out, box, prompt=hand_prompt, negative=negative, ip_image=ip_image,
                control_image=control_image, control_scale=control_scale, strength=strength,
                steps=steps, guidance=guidance, generator=generator, upscale_to=768,
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

    character_image = Image.open(args.character_image).convert("RGB")
    weapon_image = Image.open(args.weapon_image).convert("RGB")

    crop_info: dict[str, object] = {}
    if args.crop_subject:
        for label, image in (("character", character_image), ("weapon", weapon_image)):
            result = triage_image(image, mask_backend=args.crop_backend)
            cropped = crop_subject(image, result, pad_ratio=args.crop_pad)
            cropped.save(output_dir / f"input_{label}_crop.png")
            crop_info[label] = {
                "usable": result.usable,
                "score": result.score,
                "reasons": result.reasons,
                "bbox": result.bbox,
            }
            if label == "character":
                character_image = cropped
            else:
                weapon_image = cropped

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
        vlm = QwenVLMBackend(str(args.vlm_model.resolve()), max_new_tokens=500, load_in_4bit=args.vlm_4bit)
        load_seconds = perf_counter() - load_started
        analysis_started = perf_counter()
        features = vlm.analyze([character_image, weapon_image]).model_dump(mode="json")
        if args.head_zoom_traits:
            from src.domain.models import RaceTraits
            from src.vlm.feature_merger import merge_head_traits

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
        from src.prompting.spec import compile_generation_spec

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

        spec = compile_generation_spec(
            content_terms=terms,
            style_prompt=STYLE_PROMPT,
            base_negative=negative_prompt,
            traits=traits_obj,
            race_signatures=signatures,
            anatomy_rules=anatomy_rules_data,
            extra_prompt=args.extra_prompt,
            race_id=ensemble_race,
            recognized=ensemble_used,
        )
        prompt = spec.positive_prompt
        negative_prompt = spec.negative_prompt
        race_id = spec.race_id
        (output_dir / "constraints.json").write_text(
            json.dumps(
                {"race_id": race_id, "gear_id": gear_id, "gear_reference": gear_reference, **spec.constraints},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
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
        prompt = ", ".join(part for part in (args.extra_prompt, STYLE_PROMPT, body) if part)
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
    if control_image is not None:
        generation_kwargs["control_image"] = control_image
        generation_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
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
            control_scale=args.controlnet_scale,
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
            control_scale=args.controlnet_scale,
            strength=args.face_detail_strength,
            steps=args.steps,
            guidance=args.guidance_scale,
            generator=generator,
        )
    if args.detail_hands_feet:
        output_image = detail_hands_feet(
            pipeline,
            output_image,
            base_prompt=prompt_used,
            negative=negative_used,
            ip_image=ip_adapter_image,
            control_image=control_image,
            control_scale=args.controlnet_scale,
            strength=args.hand_feet_strength,
            steps=args.steps,
            guidance=args.guidance_scale,
            generator=generator,
            tokenizer=pipeline.tokenizer,
            hand_detector=args.hand_detector,
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
        "character_image": str(args.character_image.resolve()),
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
