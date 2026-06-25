"""Turn a folder of multi-angle character screenshots into a per-character LoRA dataset.

A per-character LoRA is the highest-fidelity way to lock *this* character's exact horns,
hairstyle, scales, and accessories into the model — far stronger than prompt tokens or a
single-image identity adapter, which only describe a generic member of the race.

This step is CPU-only and safe to run repeatedly. It reuses the screenshot triage/crop
pipeline to frame each reference on the character, then writes a kohya-sd-scripts compatible
dataset layout with one caption file per image and a ready training config.

    python scripts/prepare_character_dataset.py \
        --input-dir private_inputs/<character>/refs \
        --output-dir datasets/<character> \
        --trigger <character_token>

Captions intentionally hold only the trigger word (plus an optional class tag): everything
constant about the character should bind to the trigger, so leave the locked traits out and
add only *varying* tags (pose, expression) per image if you want.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-character LoRA training dataset from reference shots.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of reference screenshots.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset root to create.")
    parser.add_argument("--trigger", required=True, help="Unique trigger token, e.g. the character's name.")
    parser.add_argument("--class-tag", default="1girl", help="Optional class tag appended to every caption.")
    parser.add_argument("--repeats", type=int, default=10, help="kohya repeats per image (folder name prefix).")
    parser.add_argument("--resolution", type=int, default=1024, help="Longest-edge resize target.")
    parser.add_argument("--crop-pad", type=float, default=0.12, help="Padding ratio around the detected subject.")
    parser.add_argument("--crop-backend", choices=("rembg", "blue_screen"), default="rembg")
    parser.add_argument(
        "--crop-subject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Frame each shot on the character before resizing.",
    )
    parser.add_argument(
        "--triage-skip",
        action="store_true",
        help="Drop shots the triage gates flag as unusable (off by default: you curated these).",
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def resize_longest(image: Image.Image, longest: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= longest:
        return image
    scale = longest / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def main() -> None:
    from src.preprocessing.triage import crop_subject, triage_image

    args = parse_args()
    images = iter_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found under {args.input_dir}")

    img_root = args.output_dir / "img" / f"{args.repeats}_{args.trigger}"
    img_root.mkdir(parents=True, exist_ok=True)
    caption = ", ".join(part for part in (args.trigger, args.class_tag) if part)

    kept = 0
    skipped: list[str] = []
    for index, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        result = triage_image(image, mask_backend=args.crop_backend)
        if args.triage_skip and not result.usable:
            skipped.append(f"{path.name}: {', '.join(result.reasons)}")
            continue
        if args.crop_subject:
            image = crop_subject(image, result, pad_ratio=args.crop_pad)
        image = resize_longest(image, args.resolution)

        stem = f"{index:03d}_{path.stem}"
        image.save(img_root / f"{stem}.png")
        (img_root / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
        kept += 1
        flag = "" if result.usable else f"  (triage: {', '.join(result.reasons)})"
        print(f"  + {stem}.png  score={result.score}{flag}")

    write_training_config(args, kept)
    print(f"\nprepared {kept} image(s) -> {img_root}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for line in skipped:
            print(f"  - {line}")
    print(f"caption: {caption!r}")
    print(f"next: train with kohya sd-scripts using {args.output_dir / 'train_config.toml'}")
    print("see docs/character-lora.md for the training step")


def write_training_config(args: argparse.Namespace, image_count: int) -> None:
    """Emit a kohya sd-scripts config tuned for an Illustrious/SDXL per-character LoRA."""
    output_dir = args.output_dir.resolve()
    config = f"""# kohya sd-scripts SDXL LoRA config for character `{args.trigger}` ({image_count} images).
# Train in the sd-scripts venv:  accelerate launch sdxl_train_network.py --config_file train_config.toml
# Adjust pretrained_model_name_or_path to your local Illustrious checkpoint.

pretrained_model_name_or_path = "models/diffusion/illustrious-xl-v0.1/Illustrious-XL-v0.1.safetensors"
train_data_dir = "{(output_dir / 'img').as_posix()}"
output_dir = "{(output_dir / 'lora').as_posix()}"
output_name = "{args.trigger}-illustrious-v1"
resolution = "{args.resolution},{args.resolution}"
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = {max(args.resolution, 1536)}

network_module = "networks.lora"
network_dim = 16
network_alpha = 8
train_batch_size = 1
max_train_epochs = 12
learning_rate = 1e-4
unet_lr = 1e-4
text_encoder_lr = 5e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 0
optimizer_type = "AdamW8bit"
mixed_precision = "fp16"
save_precision = "fp16"
save_every_n_epochs = 2
save_model_as = "safetensors"
xformers = false
sdpa = true
gradient_checkpointing = true
cache_latents = true
seed = 42
"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_config.toml").write_text(config, encoding="utf-8")


if __name__ == "__main__":
    main()
