#!/bin/bash
# Usage: sbatch scripts/AR_image_generation_FluxKontext_ablation.slurm [ablation_name ...]
# With no names every ablation runs in this one job; pass names to run a subset
# (e.g. one job per ablation: sbatch scripts/...slurm no_critic).
#SBATCH --job-name=flux_kontext_ablation
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=flux_kontext_ablation_%j.out
#SBATCH --mail-user=aya34@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge

# The env's bin goes first on PATH because the called script runs plain `python`
# (conda's activate relies on a shell function that never gets initialized in a
# non-interactive batch shell, which silently leaves you on the system python).
export PATH=/ix/akovashka/aya34/envs/SensoryAds/bin:$PATH

: "${HF_TOKEN:?set 'export HF_TOKEN=...' in ~/.bashrc on the cluster; do not commit it}"
# The planner/critic/refiner agents call gpt-4o.
: "${OPENAI_API_KEY:?set 'export OPENAI_API_KEY=...' in ~/.bashrc on the cluster; do not commit it}"
export HF_HOME=/ix/akovashka/aya34/HF_CACHE
export DATA_PATH=/ix/akovashka/aya34/projects/Data/PittAd
export EXPERIMENTS_PATH=/ix/akovashka/aya34/projects/experiments
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Deliberately NOT setting HF_HUB_OFFLINE=1 here (unlike the eval jobs): this job
# needs outbound access for the OpenAI API and W&B. If compute nodes have none,
# use WANDB_MODE=offline and expect the GPT-4o agent calls to fail.

cd $SLURM_SUBMIT_DIR

# Same RUN_ID for every ablation in this job so the output folders group together.
export RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

bash scripts/AR_image_generation_FluxKontext_ablation.sh "$@"
status=$?

crc-job-stats

# Exit on the run's status, not crc-job-stats', so Slurm reports failures.
exit $status
