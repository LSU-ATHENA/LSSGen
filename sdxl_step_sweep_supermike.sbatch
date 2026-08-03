#!/bin/bash
# Review template. No allocation is embedded in this file.
#SBATCH -A hpc_aigc_optim
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=sdxl-step-sweep
#SBATCH --output=/work/hchoi12/logs/sdxl-step-%A_%a.out
#SBATCH --error=/work/hchoi12/logs/sdxl-step-%A_%a.err

set -euo pipefail

source /work/hchoi12/miniforge3/etc/profile.d/conda.sh
conda activate /work/hchoi12/conda_envs/lssgen

export HF_HOME=/work/hchoi12/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/work/hchoi12/cache/huggingface/hub

cd /work/hchoi12/LSSGen-replication

: "${APPROVED_ALLOCATIONS:?Submit with an approved pair, for example --export=ALL,APPROVED_ALLOCATIONS=10:40}"

python generate_geneval_sdxl_step_sweep.py \
  /work/hchoi12/geneval/prompts/evaluation_metadata.jsonl \
  --output-root /work/hchoi12/outputs/sdxl_step_sweep \
  --allocations "${APPROVED_ALLOCATIONS}" \
  --stage-resolutions 1024x1024,2048x2048 \
  --width 2048 \
  --height 2048 \
  --samples-per-prompt 4 \
  --all-prompts \
  --guidance 5.0 \
  --start-sigma 0.75 \
  --dtype float16 \
  --execute
