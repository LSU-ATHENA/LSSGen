#!/usr/bin/env python3
"""Generate GenEval-format SDXL images for a two-stage step experiment.

The step sweep keeps 87 total denoising calls and reallocates them between
the 1024x1024 and 2048x2048 stages in five-call increments. Dry-run is the
default; pass --execute after reviewing sweep_plan.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.3"
TOTAL_STAGE_STEPS = 87
LOW_STAGE_STEP_VALUES = tuple(range(0, 86, 5))


def step_allocation_pairs() -> list[tuple[int, int]]:
    """Generate the 87-step allocation sweep, including the 50/37 baseline."""
    return [
        (stage1_steps, TOTAL_STAGE_STEPS - stage1_steps)
        for stage1_steps in LOW_STAGE_STEP_VALUES
    ]


@dataclass(frozen=True)
class Allocation:
    index: int
    stage1_steps: int
    stage2_steps: int

    @property
    def total_steps(self) -> int:
        return self.stage1_steps + self.stage2_steps

    @property
    def allocation_id(self) -> str:
        return f"s1_{self.stage1_steps:03d}_s2_{self.stage2_steps:03d}"

    @property
    def cumulative_endpoints(self) -> list[int]:
        return [self.stage1_steps, self.total_steps]


def parse_positive_int_csv(text: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return values


def parse_allocation_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw in text.split(","):
        item = raw.strip().lower().replace(" ", "")
        if not item:
            continue
        separator = ":" if ":" in item else "+" if "+" in item else None
        if separator is None:
            raise ValueError("--allocations entries must use STAGE1:STAGE2, for example 10:40,15:35")
        left, right = item.split(separator, 1)
        try:
            stage1_steps, stage2_steps = int(left), int(right)
        except ValueError as exc:
            raise ValueError(f"Invalid allocation pair: {raw}") from exc
        if stage1_steps < 0 or stage2_steps <= 0:
            raise ValueError("Stage 1 must be non-negative and Stage 2 must be positive")
        pairs.append((stage1_steps, stage2_steps))
    if not pairs:
        raise ValueError("--allocations cannot be empty")
    if len(set(pairs)) != len(pairs):
        raise ValueError("--allocations contains duplicate pairs")
    return pairs


def parse_resolutions(text: str) -> list[tuple[int, int]]:
    resolutions: list[tuple[int, int]] = []
    for raw in text.split(","):
        item = raw.strip().lower().replace(" ", "")
        if not item:
            continue
        if "x" in item:
            width_text, height_text = item.split("x", 1)
            width, height = int(width_text), int(height_text)
        else:
            width = height = int(item)
        if width <= 0 or height <= 0 or width % 8 or height % 8:
            raise ValueError(f"Invalid resolution {width}x{height}; dimensions must be positive multiples of 8")
        resolutions.append((width, height))
    if len(resolutions) != 2:
        raise ValueError("This experiment requires exactly two stage resolutions")
    return resolutions


def read_metadata(path: Path) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
                raise ValueError(f"Line {line_number} does not contain a string 'prompt'")
            metadata.append(value)
    if not metadata:
        raise ValueError(f"No prompts found in {path}")
    return metadata


def build_allocations(stage1_values: list[int], stage2_values: list[int]) -> list[Allocation]:
    allocations: list[Allocation] = []
    for stage1_steps in stage1_values:
        for stage2_steps in stage2_values:
            allocations.append(Allocation(len(allocations), stage1_steps, stage2_steps))
    return allocations


def indexed_allocations(pairs: list[tuple[int, int]]) -> list[Allocation]:
    return [Allocation(index, pair[0], pair[1]) for index, pair in enumerate(pairs)]


def risk_label(total_steps: int) -> str:
    if total_steps < 20:
        return "cheap_quality_check"
    if total_steps <= 40:
        return "initial_exploration"
    if total_steps <= 60:
        return "expensive"
    return "very_expensive"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_plan(
    output_root: Path,
    allocations: list[Allocation],
    resolutions: list[tuple[int, int]],
    allocation_mode: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "sweep_plan.csv"
    temporary_plan = plan_path.with_name(f"{plan_path.name}.{os.getpid()}.tmp")
    with temporary_plan.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "allocation_index",
                "allocation_mode",
                "allocation_id",
                "stage1_resolution",
                "stage1_steps",
                "stage2_resolution",
                "stage2_steps",
                "total_calls",
                "cost_guidance",
            ),
        )
        writer.writeheader()
        for allocation in allocations:
            writer.writerow(
                {
                    "allocation_index": allocation.index,
                    "allocation_mode": allocation_mode,
                    "allocation_id": allocation.allocation_id,
                    "stage1_resolution": f"{resolutions[0][0]}x{resolutions[0][1]}",
                    "stage1_steps": allocation.stage1_steps,
                    "stage2_resolution": f"{resolutions[1][0]}x{resolutions[1][1]}",
                    "stage2_steps": allocation.stage2_steps,
                    "total_calls": allocation.total_steps,
                    "cost_guidance": risk_label(allocation.total_steps),
                }
            )
    os.replace(temporary_plan, plan_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_file", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--step-sweep",
        action="store_true",
        help="Use the 0/87 through 85/2 fixed-total step sweep",
    )
    parser.add_argument("--allocations", help="Explicit pairs, e.g. 10:40,15:35,20:30")
    parser.add_argument("--stage1-values", help="Comma-separated Stage 1 values")
    parser.add_argument("--stage2-values", help="Comma-separated Stage 2 values for an independent grid")
    parser.add_argument(
        "--fixed-total",
        type=int,
        help="Derive Stage 2 as fixed total minus each --stage1-values entry",
    )
    parser.add_argument("--stage-resolutions", default="1024x1024,2048x2048")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-start", type=int, default=0)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-limit", type=int, default=1)
    prompt_group.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--allocation-index", type=int)

    parser.add_argument("--model-path", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--upsampler-path", default="weights/SDXL-VAE-scaler")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--start-sigma", type=float, default=0.75)
    parser.add_argument("--base-resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--no-safetensors", action="store_true")
    parser.add_argument("--enable-attention-slicing", action="store_true")
    parser.add_argument("--no-vae-slicing", action="store_true")
    parser.add_argument("--no-vae-tiling", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def validated_experiment(args: argparse.Namespace):
    metadata_path = args.metadata_file.resolve()
    output_root = args.output_root.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8:
        raise ValueError("--width and --height must be positive multiples of 8")
    if args.samples_per_prompt <= 0:
        raise ValueError("--samples-per-prompt must be positive")
    if args.prompt_start < 0:
        raise ValueError("--prompt-start cannot be negative")
    if args.guidance <= 0:
        raise ValueError("--guidance must be positive")
    if not 0 < args.start_sigma <= 1:
        raise ValueError("--start-sigma must be in (0, 1]")

    if args.step_sweep:
        if args.allocations or args.stage1_values or args.stage2_values or args.fixed_total is not None:
            raise ValueError("--step-sweep cannot be combined with another allocation mode")
        allocations = indexed_allocations(step_allocation_pairs())
        allocation_mode = "step_sweep_87"
    elif args.allocations:
        if args.stage1_values or args.stage2_values or args.fixed_total is not None:
            raise ValueError("--allocations cannot be combined with stage value or fixed-total options")
        allocations = indexed_allocations(parse_allocation_pairs(args.allocations))
        allocation_mode = "explicit_pairs"
    elif args.fixed_total is not None:
        if args.fixed_total <= 1:
            raise ValueError("--fixed-total must be greater than 1")
        if not args.stage1_values or args.stage2_values:
            raise ValueError("Fixed-total mode requires --stage1-values and forbids --stage2-values")
        stage1_values = parse_positive_int_csv(args.stage1_values, "--stage1-values")
        pairs = [(stage1, args.fixed_total - stage1) for stage1 in stage1_values]
        if any(stage2 <= 0 for _, stage2 in pairs):
            raise ValueError("Every Stage 1 value must be below --fixed-total")
        allocations = indexed_allocations(pairs)
        allocation_mode = "fixed_total"
    else:
        if not args.stage1_values or not args.stage2_values:
            raise ValueError(
                "Choose one mode: --allocations; --fixed-total with --stage1-values; "
                "or both --stage1-values and --stage2-values for a grid"
            )
        stage1_values = parse_positive_int_csv(args.stage1_values, "--stage1-values")
        stage2_values = parse_positive_int_csv(args.stage2_values, "--stage2-values")
        allocations = build_allocations(stage1_values, stage2_values)
        allocation_mode = "independent_grid"

    if allocation_mode == "step_sweep_87":
        if len(allocations) != 18:
            raise RuntimeError("The step sweep must contain exactly 18 allocations")
        if any(allocation.total_steps != TOTAL_STAGE_STEPS for allocation in allocations):
            raise RuntimeError("Every step-sweep allocation must total 87 calls")
        if (50, 37) not in {
            (allocation.stage1_steps, allocation.stage2_steps) for allocation in allocations
        }:
            raise RuntimeError("The step sweep must include the 50/37 baseline")
    resolutions = parse_resolutions(args.stage_resolutions)
    if resolutions[-1] != (args.width, args.height):
        raise ValueError(
            f"Final stage {resolutions[-1][0]}x{resolutions[-1][1]} must match "
            f"--width/--height {args.width}x{args.height}"
        )

    metadata = read_metadata(metadata_path)
    if args.prompt_start >= len(metadata):
        raise ValueError(f"--prompt-start must be below the {len(metadata)} available prompts")
    if args.all_prompts:
        selected_metadata = list(enumerate(metadata))[args.prompt_start :]
    else:
        if args.prompt_limit is None or args.prompt_limit <= 0:
            raise ValueError("--prompt-limit must be positive")
        selected_metadata = list(enumerate(metadata))[
            args.prompt_start : args.prompt_start + args.prompt_limit
        ]
    if not selected_metadata:
        raise ValueError("The selected prompt range is empty")

    if args.allocation_index is not None:
        if not 0 <= args.allocation_index < len(allocations):
            raise ValueError(f"--allocation-index must be between 0 and {len(allocations) - 1}")
        selected_allocations = [allocations[args.allocation_index]]
    else:
        selected_allocations = allocations

    return (
        metadata_path,
        output_root,
        resolutions,
        selected_metadata,
        allocations,
        selected_allocations,
        allocation_mode,
    )


def load_pipeline(args: argparse.Namespace):
    import torch

    from network.models.upsampler import LatentUpSampler
    from network.pipelines.pipeline_lss_stable_diffusion_xl_cli_manual import (
        LSSStableDiffusionXLPipeline,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading latent upsampler once: {args.upsampler_path}")
    upsampler = LatentUpSampler.from_pretrained(args.upsampler_path, torch_dtype=dtype).to(args.device)
    pipe_kwargs: dict[str, Any] = {
        "latent_upsampler": upsampler,
        "torch_dtype": dtype,
        "use_safetensors": not args.no_safetensors,
    }
    if args.variant and not args.no_safetensors:
        pipe_kwargs["variant"] = args.variant

    print(f"Loading SDXL once: {args.model_path}")
    pipe = LSSStableDiffusionXLPipeline.from_pretrained(args.model_path, **pipe_kwargs).to(args.device)
    scheduler_class = type(pipe.scheduler).__name__
    print(f"Scheduler loaded from model configuration: {scheduler_class}")
    if not args.no_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if not args.no_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if args.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return torch, pipe, scheduler_class


def expected_config(
    args: argparse.Namespace,
    metadata_path: Path,
    resolutions: list[tuple[int, int]],
    allocation: Allocation,
    selected_metadata: list[tuple[int, dict[str, Any]]],
    scheduler_class: str,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "metadata_file": str(metadata_path),
        "model_path": args.model_path,
        "upsampler_path": str(Path(args.upsampler_path).resolve()),
        "allocation": asdict(allocation),
        "allocation_id": allocation.allocation_id,
        "allocation_mode": args.allocation_mode,
        "total_steps": allocation.total_steps,
        "manual_stage_resolutions": [list(item) for item in resolutions],
        "manual_stage_cumulative_steps": allocation.cumulative_endpoints,
        "manual_stage_start_sigmas": [1.0, args.start_sigma],
        "width": args.width,
        "height": args.height,
        "samples_per_prompt": args.samples_per_prompt,
        "selected_prompt_indices": [index for index, _ in selected_metadata],
        "guidance": args.guidance,
        "start_sigma": args.start_sigma,
        "base_resolution": args.base_resolution,
        "negative_prompt": args.negative_prompt,
        "seed_policy": "base_seed_plus_sample_index_reset_for_each_prompt",
        "base_seed": args.seed,
        "scheduler_class": scheduler_class,
        "device": args.device,
        "dtype": args.dtype,
        "variant": args.variant,
        "use_safetensors": not args.no_safetensors,
        "attention_slicing": args.enable_attention_slicing,
        "vae_slicing": not args.no_vae_slicing,
        "vae_tiling": not args.no_vae_tiling,
    }


def ensure_allocation_config(allocation_dir: Path, config: dict[str, Any], resume: bool) -> None:
    config_path = allocation_dir / "config.json"
    if allocation_dir.exists() and any(allocation_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Output exists and --no-resume was set: {allocation_dir}")
        if not config_path.is_file():
            raise FileExistsError(f"Non-empty output has no config.json: {allocation_dir}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"Existing configuration does not match this run: {allocation_dir}")
    else:
        allocation_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)


def verify_stage_info(stage_info: list[dict[str, Any]], allocation: Allocation, resolutions) -> None:
    expected_lengths = [allocation.stage1_steps, allocation.stage2_steps]
    actual_lengths = [int(item["actual_steps"]) for item in stage_info]
    actual_resolutions = [item["resolution"] for item in stage_info]
    expected_resolutions = [f"{width}x{height}" for width, height in resolutions]
    if actual_lengths != expected_lengths:
        raise RuntimeError(f"Actual stage calls {actual_lengths} do not match requested {expected_lengths}")
    if actual_resolutions != expected_resolutions:
        raise RuntimeError(
            f"Actual stage resolutions {actual_resolutions} do not match requested {expected_resolutions}"
        )
    if not all(bool(item.get("manual_stage")) for item in stage_info):
        raise RuntimeError("Pipeline did not report manual_stage=True for every stage")


def generate_allocation(
    torch,
    pipe,
    args: argparse.Namespace,
    metadata_path: Path,
    output_root: Path,
    resolutions: list[tuple[int, int]],
    selected_metadata: list[tuple[int, dict[str, Any]]],
    allocation: Allocation,
    scheduler_class: str,
) -> None:
    from PIL import Image

    allocation_dir = output_root / allocation.allocation_id
    config = expected_config(
        args,
        metadata_path,
        resolutions,
        allocation,
        selected_metadata,
        scheduler_class,
    )
    ensure_allocation_config(allocation_dir, config, resume=not args.no_resume)

    min_resolution = min(min(width, height) for width, height in resolutions)
    generated = 0
    skipped = 0
    total_generation_seconds = 0.0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for prompt_position, (prompt_index, metadata) in enumerate(selected_metadata, start=1):
        prompt_dir = allocation_dir / f"{prompt_index:05d}"
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        metadata_path_out = prompt_dir / "metadata.jsonl"
        expected_metadata_line = json.dumps(metadata, ensure_ascii=False) + "\n"
        if metadata_path_out.exists():
            if metadata_path_out.read_text(encoding="utf-8") != expected_metadata_line:
                raise ValueError(f"Existing prompt metadata differs: {metadata_path_out}")
        else:
            metadata_path_out.write_text(expected_metadata_line, encoding="utf-8")

        prompt_records: list[dict[str, Any]] = []
        for sample_index in range(args.samples_per_prompt):
            image_path = sample_dir / f"{sample_index:04d}.png"
            sample_seed = args.seed + sample_index
            if image_path.is_file():
                with Image.open(image_path) as image:
                    if image.size != (args.width, args.height):
                        raise ValueError(f"Existing image has wrong dimensions: {image_path} -> {image.size}")
                    image.verify()
                skipped += 1
                prompt_records.append(
                    {"sample_index": sample_index, "seed": sample_seed, "status": "skipped_existing"}
                )
                continue

            generator = torch.Generator(device=args.device).manual_seed(sample_seed)
            print(
                f"[{allocation.allocation_id}] prompt {prompt_position}/{len(selected_metadata)}, "
                f"sample {sample_index + 1}/{args.samples_per_prompt}, seed {sample_seed}"
            )
            started = time.perf_counter()
            with torch.inference_mode():
                image = pipe(
                    prompt=metadata["prompt"],
                    negative_prompt=args.negative_prompt.strip() or None,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=allocation.total_steps,
                    guidance_scale=args.guidance,
                    base_height=args.base_resolution,
                    base_width=args.base_resolution,
                    start_sigma=args.start_sigma,
                    min_resolution=min_resolution,
                    shorten_intermediate_steps=False,
                    manual_stage_resolutions=resolutions,
                    manual_stage_steps=allocation.cumulative_endpoints,
                    manual_stage_start_sigmas=[1.0, args.start_sigma],
                    scaling_space="resnet",
                    generator=generator,
                ).images[0]
            elapsed = time.perf_counter() - started
            stage_info = list(pipe.last_stage_info)
            verify_stage_info(stage_info, allocation, resolutions)
            if image.size != (args.width, args.height):
                raise RuntimeError(f"Generated image size {image.size}, expected {(args.width, args.height)}")

            temporary_image = image_path.with_name(image_path.stem + ".tmp.png")
            image.save(temporary_image)
            os.replace(temporary_image, image_path)
            generated += 1
            total_generation_seconds += elapsed
            prompt_records.append(
                {
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "status": "generated",
                    "seconds": elapsed,
                    "stage_info": stage_info,
                }
            )
            write_json(
                prompt_dir / "result.json",
                {
                    "prompt_index": prompt_index,
                    "prompt": metadata["prompt"],
                    "allocation_id": allocation.allocation_id,
                    "samples": prompt_records,
                },
            )

    peak_vram_gib = None
    if args.device.startswith("cuda"):
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
    write_json(
        allocation_dir / "allocation_summary.json",
        {
            "allocation_id": allocation.allocation_id,
            "stage1_steps": allocation.stage1_steps,
            "stage2_steps": allocation.stage2_steps,
            "total_calls_per_image": allocation.total_steps,
            "scheduler_class": scheduler_class,
            "selected_prompt_count": len(selected_metadata),
            "samples_per_prompt": args.samples_per_prompt,
            "expected_image_count": len(selected_metadata) * args.samples_per_prompt,
            "generated_this_invocation": generated,
            "skipped_existing_this_invocation": skipped,
            "generation_seconds_this_invocation": total_generation_seconds,
            "peak_vram_gib_this_invocation": peak_vram_gib,
            "complete": generated + skipped == len(selected_metadata) * args.samples_per_prompt,
        },
    )
    print(
        f"Completed {allocation.allocation_id}: generated={generated}, skipped={skipped}, "
        f"seconds={total_generation_seconds:.2f}, peak_vram_gib={peak_vram_gib}"
    )


def main() -> int:
    args = parse_args()
    (
        metadata_path,
        output_root,
        resolutions,
        selected_metadata,
        allocations,
        selected_allocations,
        allocation_mode,
    ) = validated_experiment(args)
    args.allocation_mode = allocation_mode
    write_plan(output_root, allocations, resolutions, allocation_mode)

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Available prompts: {len(read_metadata(metadata_path))}")
    print(f"Selected prompts: {len(selected_metadata)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Allocation mode: {allocation_mode}")
    print(f"Planned allocations: {len(allocations)}")
    print(f"Allocations in this process: {len(selected_allocations)}")
    for allocation in allocations:
        marker = "RUN" if allocation in selected_allocations else "---"
        print(
            f"[{allocation.index:02d}] {marker} {allocation.allocation_id}: "
            f"{allocation.stage1_steps} + {allocation.stage2_steps} = "
            f"{allocation.total_steps} calls ({risk_label(allocation.total_steps)})"
        )
    print(f"Plan: {output_root / 'sweep_plan.csv'}")

    if not args.execute:
        print("Dry run only. Add --execute after reviewing the plan.")
        return 0

    torch, pipe, scheduler_class = load_pipeline(args)
    for allocation in selected_allocations:
        generate_allocation(
            torch,
            pipe,
            args,
            metadata_path,
            output_root,
            resolutions,
            selected_metadata,
            allocation,
            scheduler_class,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
