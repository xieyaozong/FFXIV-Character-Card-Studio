"""Turn the base_character reference DB into per-leaf kohya LoRA datasets (hybrid granularity).

Walks ``datasets/base_character`` and treats every folder that DIRECTLY contains images as one
"leaf" -> one LoRA dataset. The folder layout itself decides the granularity, so the hybrid plan
needs no manifest:

    Au Ra/Raen/Female/horntype3/*.png   -> one variant LoRA  (variant-rich races: split by variant)
    Hyur/Midlander/Female/*.png         -> one clan-gender LoRA (variant-light races: no split)

The trigger token and class tag (1girl/1boy) are derived from the path. Captions hold only the
trigger + class tag: everything constant about that race/clan/gender/variant binds to the trigger
(kohya convention, same as prepare_character_dataset.py). Capture focus is HEAD/FACE — body build is
deliberately not locked (see the reproduction-vs-generation rules).

    # See the plan (leaves found, image counts, derived triggers) without writing anything:
    python scripts/prepare_race_datasets.py --list

    # Build datasets for one subtree (pilot), or drop --filter for everything:
    python scripts/prepare_race_datasets.py --filter "Au Ra/Raen/Female" --output-dir datasets/race_lora

This step is CPU-only and safe to re-run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# Path parts that name the gender level -> SDXL/anime class tag.
GENDER_TAGS = {"female": "1girl", "male": "1boy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-leaf race/variant LoRA datasets from base_character.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/base_character"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/race_lora"))
    parser.add_argument(
        "--filter",
        default="",
        help="Only process leaves whose path contains this substring (e.g. 'Au Ra/Raen/Female').",
    )
    parser.add_argument("--list", action="store_true", help="List the leaves/triggers/counts and exit (no writing).")
    parser.add_argument("--resolution", type=int, default=1024, help="Longest-edge resize target.")
    parser.add_argument("--crop-pad", type=float, default=0.12, help="Padding ratio around the detected subject.")
    parser.add_argument("--crop-backend", choices=("rembg", "blue_screen"), default="rembg")
    parser.add_argument(
        "--crop-subject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Frame each shot on the character before resizing.",
    )
    parser.add_argument("--repeats", type=int, default=10, help="kohya repeats per image (folder name prefix).")
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Warn about leaves with fewer than this many images (thin training sets).",
    )
    return parser.parse_args()


def find_leaf_dirs(base_dir: Path) -> list[Path]:
    """Folders that directly contain at least one image (a folder with image subfolders is not a leaf)."""
    leaves = []
    for directory in sorted(p for p in base_dir.rglob("*") if p.is_dir()):
        if any(child.suffix.lower() in IMAGE_SUFFIXES for child in directory.iterdir() if child.is_file()):
            leaves.append(directory)
    return leaves


def leaf_images(leaf: Path) -> list[Path]:
    return sorted(p for p in leaf.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def trigger_from_parts(parts: tuple[str, ...]) -> str:
    """Sanitize the path under base_character into one kohya-safe trigger token (lowercase alnum)."""
    joined = "".join(re.sub(r"[^0-9a-zA-Z]+", "", part).lower() for part in parts)
    return joined or "char"


def class_tag_from_parts(parts: tuple[str, ...]) -> str:
    for part in parts:
        tag = GENDER_TAGS.get(part.strip().lower())
        if tag:
            return tag
    return ""


def resize_longest(image: Image.Image, longest: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= longest:
        return image
    scale = longest / max(width, height)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)


def write_training_config(output_dir: Path, trigger: str, resolution: int) -> None:
    config = f"""# kohya sd-scripts SDXL LoRA config for `{trigger}` (race/variant reference set).
# Train in the sd-scripts venv:  accelerate launch sdxl_train_network.py --config_file train_config.toml

pretrained_model_name_or_path = "models/diffusion/illustrious-xl-v0.1/Illustrious-XL-v0.1.safetensors"
train_data_dir = "{(output_dir / 'img').as_posix()}"
output_dir = "{(output_dir / 'lora').as_posix()}"
output_name = "{trigger}-illustrious-v1"
resolution = "{resolution},{resolution}"
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = {max(resolution, 1536)}
network_module = "networks.lora"
network_dim = 16
network_alpha = 8
train_batch_size = 2
max_train_epochs = 16
mixed_precision = "fp16"
save_precision = "fp16"
optimizer_type = "AdamW8bit"
learning_rate = 1e-4
lr_scheduler = "cosine_with_restarts"
"""
    (output_dir / "train_config.toml").write_text(config, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.base_dir.is_dir():
        raise SystemExit(f"base dir not found: {args.base_dir}")

    leaves = find_leaf_dirs(args.base_dir)
    if args.filter:
        needle = args.filter.replace("\\", "/").lower()
        leaves = [leaf for leaf in leaves if needle in leaf.as_posix().lower()]
    if not leaves:
        raise SystemExit("no leaf folders matched.")

    plan = []
    for leaf in leaves:
        parts = leaf.relative_to(args.base_dir).parts
        plan.append((leaf, trigger_from_parts(parts), class_tag_from_parts(parts), len(leaf_images(leaf))))

    print(f"{'TRIGGER':<34} {'CLASS':<7} {'IMGS':>5}  PATH")
    for leaf, trigger, class_tag, count in plan:
        flag = "  <-- thin" if count < args.min_images else ""
        print(f"{trigger:<34} {class_tag:<7} {count:>5}  {leaf.relative_to(args.base_dir).as_posix()}{flag}")
    if args.list:
        print(f"\n{len(plan)} leaf dataset(s). Re-run without --list to build.")
        return

    from src.preprocessing.triage import crop_subject, triage_image

    for leaf, trigger, class_tag, _ in plan:
        out = args.output_dir / trigger
        img_root = out / "img" / f"{args.repeats}_{trigger}"
        img_root.mkdir(parents=True, exist_ok=True)
        caption = ", ".join(part for part in (trigger, class_tag) if part)
        kept = 0
        for index, path in enumerate(leaf_images(leaf)):
            image = Image.open(path).convert("RGB")
            if args.crop_subject:
                result = triage_image(image, mask_backend=args.crop_backend)
                image = crop_subject(image, result, pad_ratio=args.crop_pad)
            image = resize_longest(image, args.resolution)
            stem = f"{index:03d}_{path.stem}"
            image.save(img_root / f"{stem}.png")
            (img_root / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
            kept += 1
        write_training_config(out, trigger, args.resolution)
        print(f"  built {trigger}: {kept} imgs -> {out}")

    print(f"\nprepared {len(plan)} dataset(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
