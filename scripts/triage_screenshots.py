from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score character screenshots and keep the ones worth sending to the VLM."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of screenshots.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/triage"))
    parser.add_argument("--top", type=int, default=0, help="Keep only the best N usable images (0 = all usable).")
    parser.add_argument("--mask-backend", choices=("rembg", "blue_screen"), default="rembg")
    parser.add_argument("--pad", type=float, default=0.08, help="Crop padding ratio around the subject.")
    parser.add_argument("--min-brightness", type=float, default=40.0)
    parser.add_argument("--min-subject-height", type=int, default=360)
    parser.add_argument("--min-coverage", type=float, default=0.012)
    parser.add_argument("--min-sharpness", type=float, default=10.0)
    parser.add_argument("--no-crops", action="store_true", help="Only write the report, skip cropped previews.")
    return parser.parse_args()


def main() -> None:
    from src.preprocessing.triage import TriageThresholds, crop_subject, triage_image

    args = parse_args()
    thresholds = TriageThresholds(
        min_brightness=args.min_brightness,
        min_subject_height=args.min_subject_height,
        min_coverage=args.min_coverage,
        min_sharpness=args.min_sharpness,
    )

    paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scored: list[dict] = []
    for path in paths:
        image = Image.open(path)
        result = triage_image(image, thresholds=thresholds, mask_backend=args.mask_backend)
        metrics = result.metrics
        scored.append(
            {
                "file": path.name,
                "path": str(path),
                "usable": result.usable,
                "score": result.score,
                "reasons": result.reasons,
                "brightness": round(metrics.brightness, 1),
                "subject_height_px": metrics.subject_height_px,
                "coverage": round(metrics.coverage, 4),
                "sharpness": round(metrics.sharpness, 1),
                "bbox": metrics.bbox,
                "_image": image,
                "_result": result,
            }
        )

    scored.sort(key=lambda item: (item["usable"], item["score"]), reverse=True)
    usable = [item for item in scored if item["usable"]]
    selected = usable[: args.top] if args.top > 0 else usable

    if not args.no_crops:
        for rank, item in enumerate(selected, start=1):
            crop = crop_subject(item["_image"], item["_result"], pad_ratio=args.pad)
            crop.save(output_dir / f"{rank:02d}_score{item['score']:.2f}_{Path(item['file']).stem}.png")

    report = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in scored
    ]
    (output_dir / "triage_report.json").write_text(
        json.dumps(
            {
                "thresholds": thresholds.__dict__,
                "mask_backend": args.mask_backend,
                "total": len(scored),
                "usable": len(usable),
                "selected": [item["file"] for item in selected],
                "images": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{'use':>4} {'score':>6} {'bright':>7} {'height':>7} {'sharp':>7}  file  (reasons)")
    for item in scored:
        flag = "OK" if item["usable"] else "--"
        reasons = "" if item["usable"] else "  " + ",".join(item["reasons"])
        print(
            f"{flag:>4} {item['score']:>6.2f} {item['brightness']:>7.1f} "
            f"{item['subject_height_px']:>7d} {item['sharpness']:>7.1f}  {item['file']}{reasons}"
        )
    print(f"\nusable {len(usable)}/{len(scored)}, kept {len(selected)} crops -> {output_dir}")


if __name__ == "__main__":
    main()
