#!/bin/bash
# Generates the two proposed Table 1 reproduction rows with the authors' unchanged batch generator.
# Array task 0: LSS-SDXL at 2048x2048
# Array task 1: LSS-SD1.5 at 1024x1024
#SBATCH -A hpc_aigc_optim
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --job-name=lss-reproduce
#SBATCH --output=/work/hchoi12/logs/lss-reproduce-%A_%a.out
#SBATCH --error=/work/hchoi12/logs/lss-reproduce-%A_%a.err

set -euo pipefail

source /work/hchoi12/miniforge3/etc/profile.d/conda.sh
conda activate /work/hchoi12/conda_envs/lssgen

export HF_HOME=/work/hchoi12/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/work/hchoi12/cache/huggingface/hub

cd /work/hchoi12/LSSGen-replication

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL="sdxl"
    OUTPUT_DIR="/work/hchoi12/outputs/replication/lss_sdxl_2048"
    EXPECTED_WIDTH=2048
    EXPECTED_HEIGHT=2048
    ;;
  1)
    MODEL="sd15"
    OUTPUT_DIR="/work/hchoi12/outputs/replication/lss_sd15_1024"
    EXPECTED_WIDTH=1024
    EXPECTED_HEIGHT=1024
    ;;
  *)
    echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

METADATA_FILE="/work/hchoi12/geneval/prompts/evaluation_metadata.jsonl"
PROMPT_COUNT="$(awk 'NF {count++} END {print count+0}' "${METADATA_FILE}")"

python replicate_lssgen.py \
  --repo /work/hchoi12/LSSGen-replication \
  --metadata "${METADATA_FILE}" \
  --model "${MODEL}" \
  --profile paper \
  --outdir "${OUTPUT_DIR}" \
  --dtype fp16 \
  --execute

python validate_geneval_output.py \
  "${OUTPUT_DIR}" \
  --expected-prompts "${PROMPT_COUNT}" \
  --samples-per-prompt 4 \
  --expected-width "${EXPECTED_WIDTH}" \
  --expected-height "${EXPECTED_HEIGHT}"
