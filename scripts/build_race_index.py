"""Build the image-embedding race index from the labeled base_character reference DB.

Embeds each reference's head region with CLIP and saves a labeled index for runtime kNN race
recognition (see src/knowledge/race_index.py). Run once; re-run when the reference DB changes.

    python scripts/build_race_index.py --base-dir datasets/base_character --out knowledge/ffxiv/race_index.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CLIP race index from base_character.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/base_character"))
    parser.add_argument("--out", type=Path, default=Path("knowledge/ffxiv/race_index.npz"))
    parser.add_argument("--encoder", type=Path, default=Path("models/ip-adapter/models/image_encoder"))
    parser.add_argument("--per-leaf", type=int, default=0, help="Cap images per clan/gender folder (0 = all).")
    parser.add_argument(
        "--remove-bg",
        action="store_true",
        help="Background-remove each head crop so the index matches bg-removed runtime queries.",
    )
    return parser.parse_args()


def collect(base_dir: Path, per_leaf: int) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for race in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for clan in sorted(p for p in race.iterdir() if p.is_dir()):
            for gender in sorted(p for p in clan.iterdir() if p.is_dir()):
                imgs = sorted(p for p in gender.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                if per_leaf:
                    imgs = imgs[:per_leaf]
                items += [(race.name, p) for p in imgs]
    return items


def main() -> None:
    from src.knowledge.race_index import ClipEmbedder, RaceIndex, character_frame, head_region, on_white

    args = parse_args()
    items = collect(args.base_dir, args.per_leaf)
    if not items:
        raise SystemExit(f"No reference images under {args.base_dir}")

    embedder = ClipEmbedder(args.encoder)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for index, (race, path) in enumerate(items, 1):
        head = head_region(character_frame(Image.open(path)))
        if args.remove_bg:
            head = on_white(head)
        vectors.append(embedder.embed(head))
        labels.append(race)
        if index % 25 == 0 or index == len(items):
            print(f"  embedded {index}/{len(items)}", file=sys.stderr)

    RaceIndex(np.stack(vectors), labels).save(args.out)
    counts = {race: labels.count(race) for race in sorted(set(labels))}
    print(f"saved index: {args.out}  ({len(labels)} refs)")
    print("per race:", counts)


if __name__ == "__main__":
    main()
