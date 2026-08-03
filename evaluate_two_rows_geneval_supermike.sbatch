#!/bin/bash
# Run only after the separate GenEval environment and detector weights are validated.
#SBATCH -A hpc_aigc_optim
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --job-name=lss-geneval
#SBATCH --output=/work/hchoi12/logs/lss-geneval-%A_%a.out
#SBATCH --error=/work/hchoi12/logs/lss-geneval-%A_%a.err

set -euo pipefail

source /work/hchoi12/miniforge3/etc/profile.d/conda.sh
conda activate /work/hchoi12/conda_envs/geneval

cd /work/hchoi12/geneval

case "${SLURM_ARRAY_TASK_ID}" in
  0) IMAGE_DIR="/work/hchoi12/outputs/replication/lss_sdxl_2048" ;;
  1) IMAGE_DIR="/work/hchoi12/outputs/replication/lss_sd15_1024" ;;
  *) echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

RESULT_DIR="${IMAGE_DIR}/metrics"
mkdir -p "${RESULT_DIR}"

python evaluation/evaluate_images.py \
  "${IMAGE_DIR}" \
  --outfile "${RESULT_DIR}/geneval_results.jsonl" \
  --model-path evaluation/models

python evaluation/summary_scores.py \
  "${RESULT_DIR}/geneval_results.jsonl" \
  | tee "${RESULT_DIR}/geneval_summary.txt"
