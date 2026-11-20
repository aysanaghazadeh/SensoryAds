#!/bin/bash
# Usage: sbatch scripts/LLM_image_generation_with_sensation.slurm
#SBATCH --job-name=LLM_sensation_gen
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=LLM_sensation_gen_%j.out
#SBATCH --mail-user=aya34@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge

# The env's bin goes first on PATH because the called script runs plain `python`
# (conda's activate relies on a shell function that never gets initialized in a
# non-interactive batch shell, which silently leaves you on the system python).
export PATH=/ix/akovashka/aya34/envs/SensoryAds/bin:$PATH

: "${HF_TOKEN:?set 'export HF_TOKEN=...' in ~/.bashrc on the cluster; do not commit it}"
export HF_HOME=/ix/akovashka/aya34/HF_CACHE
export DATA_PATH=/ix/akovashka/aya34/projects/Data/PittAd
export EXPERIMENTS_PATH=/ix/akovashka/aya34/projects/experiments
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

cd $SLURM_SUBMIT_DIR

bash scripts/LLM_image_generation_with_sensation.sh
status=$?

crc-job-stats

# Exit on the run's status, not crc-job-stats', so Slurm reports failures.
exit $status
