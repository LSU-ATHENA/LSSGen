#!/bin/bash
# One prompt and one image per row using the full paper settings.
# Array task 0: LSS-SDXL at 2048x2048
# Array task 1: LSS-SD1.5 at 1024x1024
#SBATCH -A hpc_aigc_optim
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --job-name=lss-repro-smoke
#SBATCH --output=/work/hchoi12/logs/lss-repro-smoke-%A_%a.out
#SBATCH --error=/work/hchoi12/logs/lss-repro-smoke-%A_%a.err

set -euo pipefail

source /work/hchoi12/miniforge3/etc/profile.d/conda.sh
conda activate /work/hchoi12/conda_envs/lssgen

export HF_HOME=/work/hchoi12/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/work/hchoi12/cache/huggingface/hub

REPO_DIR="/work/hchoi12/LSSGen-replication"
METADATA_FILE="/work/hchoi12/geneval/prompts/evaluation_metadata.jsonl"
RUN_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "${REPO_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL="sdxl"
    OUTPUT_DIR="/work/hchoi12/outputs/smoke/lss_sdxl_${RUN_ID}"
    EXPECTED_WIDTH=2048
    EXPECTED_HEIGHT=2048
    ;;
  1)
    MODEL="sd15"
    OUTPUT_DIR="/work/hchoi12/outputs/smoke/lss_sd15_${RUN_ID}"
    EXPECTED_WIDTH=1024
    EXPECTED_HEIGHT=1024
    ;;
  *)
    echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

python replicate_lssgen.py \
  --repo "${REPO_DIR}" \
  --metadata "${METADATA_FILE}" \
  --model "${MODEL}" \
  --profile paper \
  --limit-prompts 1 \
  --samples 1 \
  --outdir "${OUTPUT_DIR}" \
  --dtype fp16 \
  --execute

python validate_geneval_output.py \
  "${OUTPUT_DIR}" \
  --expected-prompts 1 \
  --samples-per-prompt 1 \
  --expected-width "${EXPECTED_WIDTH}" \
  --expected-height "${EXPECTED_HEIGHT}"
