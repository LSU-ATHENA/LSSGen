#!/bin/bash
# Run only after both reproduction image directories pass layout validation.
#SBATCH -A hpc_aigc_optim
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --job-name=lss-pyiqa
#SBATCH --output=/work/hchoi12/logs/lss-pyiqa-%A_%a.out
#SBATCH --error=/work/hchoi12/logs/lss-pyiqa-%A_%a.err

set -euo pipefail

source /work/hchoi12/miniforge3/etc/profile.d/conda.sh
conda activate /work/hchoi12/conda_envs/lssgen-metrics

cd /work/hchoi12/LSSGen-replication
python -c "import pyiqa, setuptools; assert pyiqa.__version__ == '0.1.16'; assert setuptools.__version__ == '80.10.2'; print('PyIQA', pyiqa.__version__, '| Setuptools', setuptools.__version__)"

case "${SLURM_ARRAY_TASK_ID}" in
  0) IMAGE_DIR="/work/hchoi12/outputs/replication/lss_sdxl_2048" ;;
  1) IMAGE_DIR="/work/hchoi12/outputs/replication/lss_sd15_1024" ;;
  *) echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

python score_pyiqa.py \
  "${IMAGE_DIR}" \
  --output-dir "${IMAGE_DIR}/metrics" \
  --device cuda
