#!/usr/bin/env python3
"""Compute CLIP-IQA+, TOPIQ, and NIQE for a GenEval image tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("clipiqa+", "topiq_nr", "niqe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-images", type=int)
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    images = sorted(root.glob("*/samples/*.png"))
    if not images:
        images = sorted(root.rglob("*.png"))
    return images


def scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "numel") and value.numel() != 1:
        raise ValueError(f"Metric returned {value.numel()} values for one image")
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric returned non-finite value: {result}")
    return result


def main() -> int:
    args = parse_args()
    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit("PyIQA is not installed. Run: python -m pip install pyiqa") from exc

    root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    images = collect_images(root)
    if args.limit_images is not None:
        if args.limit_images <= 0:
            raise ValueError("--limit-images must be positive")
        images = images[: args.limit_images]
    if not images:
        raise FileNotFoundError(f"No PNG images found under {root}")

    available_metrics = set(pyiqa.list_models())
    missing_metrics = [name for name in METRICS if name not in available_metrics]
    if missing_metrics:
        raise RuntimeError(
            "This PyIQA installation does not provide the required metric identifiers: "
            f"{', '.join(missing_metrics)}. Available models can be printed with "
            "python -c \"import pyiqa; print('\\n'.join(pyiqa.list_models()))\""
        )

    metric_objects = {
        name: pyiqa.create_metric(name, device=args.device) for name in METRICS
    }
    metric_directions = {
        name: (
            "lower_is_better"
            if bool(getattr(metric, "lower_better", name == "niqe"))
            else "higher_is_better"
        )
        for name, metric in metric_objects.items()
    }
    rows: list[dict[str, object]] = []
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    failed_images: list[dict[str, str]] = []

    for index, image_path in enumerate(images, start=1):
        relative_path = str(image_path.relative_to(root))
        row: dict[str, object] = {"image": relative_path}
        try:
            image_scores = {
                name: scalar(metric(str(image_path)))
                for name, metric in metric_objects.items()
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            row.update({name: "ERROR" for name in METRICS})
            rows.append(row)
            failed_images.append({"image": relative_path, "error": error})
            print(f"[{index}/{len(images)}] ERROR {relative_path}: {error}")
            continue

        row.update(image_scores)
        for name, score in image_scores.items():
            values[name].append(score)
        rows.append(row)
        print(f"[{index}/{len(images)}] {row['image']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pyiqa_per_image.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "image_root": str(root),
        "image_count": len(images),
        "successful_image_count": len(images) - len(failed_images),
        "failed_image_count": len(failed_images),
        "failed_images": failed_images,
        "pyiqa_version": getattr(pyiqa, "__version__", "unknown"),
        "metrics": {
            name: {
                "mean": statistics.fmean(scores) if scores else None,
                "population_stddev": statistics.pstdev(scores) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "direction": metric_directions[name],
            }
            for name, scores in values.items()
        },
    }
    summary_path = output_dir / "pyiqa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
