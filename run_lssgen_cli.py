import argparse
import sys
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler

from network.models.upsampler import LatentUpSampler
from network.pipelines.pipeline_lss_stable_diffusion_xl_cli_manual import LSSStableDiffusionXLPipeline


RUNNER_VERSION = "manual-cli"


def die(message: str, code: int = 2):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_int_list(text: str, name: str):
    if text is None or str(text).strip() == "":
        return None

    values = []
    for raw in str(text).replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(int(raw))
        except ValueError:
            die(f"{name} must be comma-separated integers. Got: {text}")

    if not values:
        die(f"{name} cannot be empty.")
    return values


def round_to_multiple(value: float, multiple: int = 8) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def build_resolution_ladder(width: int, height: int, min_resolution: int):
    if width <= 0 or height <= 0:
        die("--width and --height must be positive.")
    if width % 8 != 0 or height % 8 != 0:
        die("--width and --height must be divisible by 8.")
    if min_resolution <= 0 or min_resolution % 8 != 0:
        die("--min-resolution must be positive and divisible by 8.")

    final_max = max(width, height)
    if min_resolution > final_max:
        die("--min-resolution cannot be larger than the final resolution.")

    stages = []
    current_max = min_resolution

    while current_max < final_max:
        scale = current_max / final_max
        stage_w = min(round_to_multiple(width * scale, 8), width)
        stage_h = min(round_to_multiple(height * scale, 8), height)
        if not stages or stages[-1] != (stage_w, stage_h):
            stages.append((stage_w, stage_h))
        current_max *= 2

    if not stages or stages[-1] != (width, height):
        stages.append((width, height))

    return stages


def parse_resolution_list(text: str):
    if text is None or str(text).strip() == "":
        return None

    resolutions = []
    for raw in str(text).replace(";", ",").split(","):
        raw = raw.strip().lower().replace(" ", "")
        if not raw:
            continue

        try:
            if "x" in raw:
                w_text, h_text = raw.split("x", 1)
                w, h = int(w_text), int(h_text)
            else:
                w = h = int(raw)
        except ValueError:
            die(f"Bad resolution entry: {raw}")

        if w <= 0 or h <= 0 or w % 8 != 0 or h % 8 != 0:
            die(f"Resolution must be positive and divisible by 8: {w}x{h}")
        resolutions.append((w, h))

    return resolutions or None


def format_resolutions(resolutions) -> str:
    return ",".join(f"{w}x{h}" for w, h in resolutions)


def cumulative_from_lengths(lengths):
    total = 0
    endpoints = []
    for length in lengths:
        total += int(length)
        endpoints.append(total)
    return endpoints


def build_manual_schedule(args):
    resolutions = parse_resolution_list(args.manual_stage_resolutions)
    if resolutions is None:
        resolutions = build_resolution_ladder(args.width, args.height, args.min_resolution)

    stage_count = len(resolutions)
    total_steps = int(args.steps)

    stage_lengths = parse_int_list(args.stage_lengths, "--stage-lengths")
    stage_steps = parse_int_list(args.stage_steps, "--stage-steps")
    manual_steps = parse_int_list(args.manual_stage_steps, "--manual-stage-steps")

    provided = sum(value is not None for value in (stage_lengths, stage_steps, manual_steps))
    if provided > 1:
        die("Use only one of --stage-lengths, --stage-steps, or --manual-stage-steps.")

    if provided == 0:
        return None, None, None, resolutions, None

    if stage_lengths is not None:
        if len(stage_lengths) != stage_count:
            die(f"--stage-lengths needs {stage_count} values for {stage_count} stages. Got {len(stage_lengths)}.")
        if any(value <= 0 for value in stage_lengths):
            die("--stage-lengths values must be positive.")
        if sum(stage_lengths) != total_steps:
            die(f"--stage-lengths must sum to --steps. Got {sum(stage_lengths)} but --steps is {total_steps}.")
        manual_steps = cumulative_from_lengths(stage_lengths)

    elif stage_steps is not None:
        if len(stage_steps) == stage_count and sum(stage_steps) == total_steps and stage_steps[-1] != total_steps:
            manual_steps = cumulative_from_lengths(stage_steps)
        else:
            manual_steps = list(stage_steps)
            if manual_steps[-1] != total_steps:
                manual_steps.append(total_steps)

    if len(manual_steps) != stage_count:
        die(f"Manual stage steps need {stage_count} cumulative values for {stage_count} stages. Got {len(manual_steps)}.")
    if manual_steps[-1] != total_steps:
        die(f"Final manual stage step must equal --steps ({total_steps}).")
    if manual_steps != sorted(manual_steps) or len(set(manual_steps)) != len(manual_steps):
        die("Manual stage steps must be strictly increasing.")
    if any(value <= 0 for value in manual_steps):
        die("Manual stage steps must be positive.")

    previous = 0
    lengths = []
    for endpoint in manual_steps:
        lengths.append(endpoint - previous)
        previous = endpoint

    return (
        format_resolutions(resolutions),
        ",".join(str(value) for value in manual_steps),
        args.manual_stage_start_sigmas.strip() or None,
        resolutions,
        lengths,
    )


def dtype_from_string(value: str):
    value = str(value).lower().strip()
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    if value in ("float32", "fp32"):
        return torch.float32
    die("--dtype must be float16, bfloat16, or float32.")


def seed_generator(seed: int, device: str):
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    return seed, torch.Generator(device=device).manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Minimal LSSGen SDXL CLI runner")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", "--negative-prompt", dest="negative", default="")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", "--cfg", dest="guidance", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="output.png")

    parser.add_argument("--start-sigma", type=float, default=0.75)
    parser.add_argument("--min-resolution", type=int, default=512)
    parser.add_argument("--stage-lengths", default=None, help="Per-stage lengths, e.g. 15,15,20")
    parser.add_argument("--stage-steps", default=None, help="Cumulative endpoints, e.g. 15,30,50")
    parser.add_argument("--manual-stage-resolutions", default="", help="Manual resolutions, e.g. 512x512,1024x1024,2048x2048")
    parser.add_argument("--manual-stage-steps", default="", help="Manual cumulative endpoints, e.g. 15,30,50")
    parser.add_argument("--manual-stage-start-sigmas", default="")
    parser.add_argument("--shorten-intermediate-steps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-resolution", type=int, default=1024)
    parser.add_argument("--scaling-space", choices=["resnet", "latent"], default="resnet")

    parser.add_argument("--model-path", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--upsampler-path", default="weights/SDXL-VAE-scaler")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--no-safetensors", action="store_true")

    parser.add_argument("--clear-cuda-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-attention-slicing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-vae-slicing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-vae-tiling", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if args.steps <= 0:
        die("--steps must be positive.")
    if args.guidance <= 0:
        die("--guidance must be positive.")
    if args.start_sigma <= 0:
        die("--start-sigma must be positive.")
    if args.base_resolution <= 0:
        die("--base-resolution must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        die("CUDA was requested but torch.cuda.is_available() is false. Use --device cpu or fix CUDA/PyTorch.")

    manual_resolutions, manual_steps, manual_sigmas, schedule_resolutions, schedule_lengths = build_manual_schedule(args)

    print(f"Runner version: {RUNNER_VERSION}")
    if manual_steps:
        print("Manual schedule:")
        for index, ((w, h), length) in enumerate(zip(schedule_resolutions, schedule_lengths), start=1):
            print(f"  Stage {index}: {w}x{h}, {length} steps")
        print(f"manual_stage_resolutions = {manual_resolutions}")
        print(f"manual_stage_steps       = {manual_steps}")
    else:
        print("Manual schedule: disabled; pipeline will use its default schedule.")

    if args.clear_cuda_cache and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    dtype = dtype_from_string(args.dtype)

    print(f"Loading upsampler: {args.upsampler_path}")
    upsampler = LatentUpSampler.from_pretrained(args.upsampler_path, torch_dtype=dtype).to(args.device)

    pipe_kwargs = {
        "latent_upsampler": upsampler,
        "torch_dtype": dtype,
        "use_safetensors": not args.no_safetensors,
    }
    if args.variant and not args.no_safetensors:
        pipe_kwargs["variant"] = args.variant

    print(f"Loading SDXL pipeline: {args.model_path}")
    pipe = LSSStableDiffusionXLPipeline.from_pretrained(args.model_path, **pipe_kwargs).to(args.device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    if args.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if args.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if args.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    seed, generator = seed_generator(args.seed, args.device)

    call_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative.strip() or None,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "base_height": args.base_resolution,
        "base_width": args.base_resolution,
        "start_sigma": args.start_sigma,
        "min_resolution": args.min_resolution,
        "shorten_intermediate_steps": args.shorten_intermediate_steps,
        "manual_stage_resolutions": manual_resolutions,
        "manual_stage_steps": manual_steps,
        "manual_stage_start_sigmas": manual_sigmas,
        "scaling_space": args.scaling_space,
        "generator": generator,
    }

    print(f"Generating with seed: {seed}")
    with torch.inference_mode():
        image = pipe(**call_kwargs).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    main()
