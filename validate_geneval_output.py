#!/usr/bin/env python3
"""Validate a generated image tree against GenEval's required layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.image_root.resolve()
    errors: list[str] = []
    total_images = 0

    for index in range(args.expected_prompts):
        prompt_dir = root / f"{index:05d}"
        metadata_path = prompt_dir / "metadata.jsonl"
        sample_dir = prompt_dir / "samples"
        if not metadata_path.is_file():
            errors.append(f"Missing {metadata_path}")
        else:
            lines = [line for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1:
                errors.append(f"{metadata_path} must contain exactly one JSON line")
            else:
                try:
                    value = json.loads(lines[0])
                    if not isinstance(value.get("prompt"), str):
                        errors.append(f"{metadata_path} has no string prompt")
                except (json.JSONDecodeError, AttributeError) as exc:
                    errors.append(f"Invalid JSON in {metadata_path}: {exc}")

        for sample_index in range(args.samples_per_prompt):
            candidates = [
                sample_dir / f"{sample_index:04d}.png",
                sample_dir / f"{sample_index:05d}.png",
            ]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                errors.append(f"Missing sample {sample_index} in {sample_dir}")
                continue
            total_images += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    if args.expected_width is not None and image.width != args.expected_width:
                        errors.append(f"{image_path}: width {image.width}, expected {args.expected_width}")
                    if args.expected_height is not None and image.height != args.expected_height:
                        errors.append(f"{image_path}: height {image.height}, expected {args.expected_height}")
            except Exception as exc:  # Pillow exposes several format-specific exceptions.
                errors.append(f"Unreadable image {image_path}: {exc}")

    if errors:
        print(f"FAILED with {len(errors)} issue(s):")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} additional issue(s)")
        return 1

    print(f"PASS: {args.expected_prompts} prompt folders, {total_images} readable images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

