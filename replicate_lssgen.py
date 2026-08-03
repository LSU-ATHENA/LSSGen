#!/usr/bin/env python3
"""Safely invoke LSSGen's official GenEval batch generator.

Dry-run is the default. Pass --execute only after reviewing the printed command.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_VERSION = "1.1"


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    upsampler: str
    pipeline_type: str
    width: int
    height: int
    min_size: int
    steps: int
    guidance: float


PAPER_PROFILES = {
    "sdxl": ModelProfile(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        upsampler="weights/SDXL-VAE-scaler",
        pipeline_type="lsssdxl",
        width=2048,
        height=2048,
        min_size=1024,
        steps=50,
        guidance=5.0,
    ),
    "sd15": ModelProfile(
        model_id="runwayml/stable-diffusion-v1-5",
        upsampler="weights/SD-VAE-scaler",
        pipeline_type="lsssd",
        width=1024,
        height=1024,
        min_size=512,
        steps=50,
        guidance=7.5,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(PAPER_PROFILES), required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp16")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def read_metadata(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
            raise ValueError(f"Invalid prompt metadata on line {number}")
    return lines


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    metadata = args.metadata.resolve()
    outdir = args.outdir.resolve()
    generator = repo / "inference_dyn_scaling_t2i.py"

    if not generator.is_file():
        raise FileNotFoundError(f"Official LSSGen generator not found: {generator}")
    if not metadata.is_file():
        raise FileNotFoundError(f"GenEval metadata not found: {metadata}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    source_lines = read_metadata(metadata)
    default_limit = 1 if args.profile == "smoke" else len(source_lines)
    limit = args.limit_prompts if args.limit_prompts is not None else default_limit
    samples = args.samples if args.samples is not None else (1 if args.profile == "smoke" else 4)
    if not 1 <= limit <= len(source_lines):
        raise ValueError(f"--limit-prompts must be between 1 and {len(source_lines)}")
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if args.batch_size > samples:
        raise ValueError("--batch-size cannot exceed --samples")

    base = PAPER_PROFILES[args.model]
    profile = base
    if args.profile == "smoke":
        profile = ModelProfile(
            model_id=base.model_id,
            upsampler=base.upsampler,
            pipeline_type=base.pipeline_type,
            width=1024,
            height=1024,
            min_size=512,
            steps=8,
            guidance=base.guidance,
        )

    upsampler_path = (repo / profile.upsampler).resolve()
    if not upsampler_path.exists():
        raise FileNotFoundError(f"LSSGen upsampler not found: {upsampler_path}")

    subset_path = outdir / "input_metadata.jsonl"
    command = [
        sys.executable,
        str(generator),
        str(subset_path),
        "--model",
        profile.model_id,
        "--upscale_model",
        str(upsampler_path),
        "--pipeline_type",
        profile.pipeline_type,
        "--outdir",
        str(outdir),
        "--n_samples",
        str(samples),
        "--min_size",
        str(profile.min_size),
        "--steps",
        str(profile.steps),
        "--H",
        str(profile.height),
        "--W",
        str(profile.width),
        "--scale",
        str(profile.guidance),
        "--sigma",
        "0.75",
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
    ]

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Source prompts: {len(source_lines)}")
    print(f"Selected prompts: {limit}")
    print(f"Expected images: {limit * samples}")
    print("Command:")
    print(shlex.join(command))

    if not args.execute:
        print("Dry run only. Add --execute after reviewing the command.")
        return 0

    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    subset_path.write_text("\n".join(source_lines[:limit]) + "\n", encoding="utf-8")
    invocation = {
        "script_version": SCRIPT_VERSION,
        "wrapper": vars(args) | {"repo": str(repo), "metadata": str(metadata), "outdir": str(outdir)},
        "profile": asdict(profile),
        "source_prompt_count": len(source_lines),
        "selected_prompt_count": limit,
        "expected_image_count": limit * samples,
        "command": command,
    }
    (outdir / "reproduction_invocation.json").write_text(
        json.dumps(invocation, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(command, cwd=repo, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
