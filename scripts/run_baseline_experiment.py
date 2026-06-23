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

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STYLE_PROMPT = (
    "hand-drawn anime character sheet, black ink linework, colored pencil, white paper, full body"
)
NEGATIVE_PROMPT = (
    "photorealistic, 3d render, game screenshot, scenery, text, watermark, "
    "extra limbs, extra fingers, missing hands, duplicate person"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first local FFXIV character-card baseline experiment.")
    parser.add_argument("--character-image", type=Path, required=True)
    parser.add_argument("--weapon-image", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, default=Path("models/vlm/Qwen3-VL-4B-Instruct"))
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
    return parser.parse_args()


def is_visible(value: str) -> bool:
    normalized = value.lower().replace("_", " ")
    return "not visible" not in normalized


def primary_clause(value: str) -> str:
    """Keep the leading descriptor and drop positional notes like "visible behind right thigh"."""
    return value.split(",")[0].strip()


def build_prompt(features: dict[str, object], extra_prompt: str = "") -> str:
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

    headwear = identity.get("headwear")
    if headwear and is_visible(headwear):
        add(primary_clause(headwear))

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

    body = ", ".join(terms)
    return ", ".join(part for part in (extra_prompt, STYLE_PROMPT, body) if part)


def fit_prompt_to_clip(prompt: str, tokenizer, max_tokens: int = 77) -> str:
    """Greedily keep leading comma fragments so CLIP never silently truncates the tail."""
    fragments = [fragment for fragment in prompt.split(", ") if fragment]
    kept: list[str] = []
    for fragment in fragments:
        candidate = ", ".join([*kept, fragment])
        if len(tokenizer(candidate, truncation=False).input_ids) > max_tokens:
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
        vlm = QwenVLMBackend(str(args.vlm_model.resolve()), max_new_tokens=500)
        load_seconds = perf_counter() - load_started
        analysis_started = perf_counter()
        features = vlm.analyze([character_image, weapon_image]).model_dump(mode="json")
        analysis_seconds = perf_counter() - analysis_started
        del vlm
        gc.collect()
    feature_seconds = perf_counter() - feature_started
    (output_dir / "features.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prompt = args.prompt_override.strip() or build_prompt(features, args.extra_prompt)
    negative_prompt = ", ".join(
        part for part in (NEGATIVE_PROMPT, args.extra_negative) if part
    )
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
    from diffusers import StableDiffusionXLImg2ImgPipeline

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

    generation_started = perf_counter()
    model_path = args.sdxl_model.resolve()
    if model_path.is_file():
        pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(model_path),
            config=str(args.pipeline_config.resolve()),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
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
    output_image = pipeline(**generation_kwargs).images[0]
    output_image.save(output_dir / "result.png")
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
