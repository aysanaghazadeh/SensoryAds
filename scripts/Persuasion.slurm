#!/bin/bash
#SBATCH --job-name=persuasion_eval
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=persuasion_eval_%j.out
#SBATCH --mail-user=aya34@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge

: "${HF_TOKEN:?set 'export HF_TOKEN=...' in ~/.bashrc on the cluster; do not commit it}"
export HF_HOME=/ix/akovashka/aya34/HF_CACHE
export DATA_PATH=/ix/akovashka/aya34/projects/Data/PittAd
export EXPERIMENTS_PATH=/ix/akovashka/aya34/projects/experiments
export TOKENIZERS_PARALLELISM=false
# Python block-buffers stdout when it isn't a TTY, so prints would otherwise
# only reach the .out file in chunks (or at process exit).
export PYTHONUNBUFFERED=1
# Compute nodes have no outbound internet, so any Hub lookup hangs instead of
# failing fast. Requires the weights to already be in HF_HOME.
export HF_HUB_OFFLINE=1

cd $SLURM_SUBMIT_DIR

# Persuasion.sh itself backgrounds each evaluate.py call under its own
# CUDA_VISIBLE_DEVICES and waits on all three, so a plain bash (no srun,
# no accelerate) is all that's needed here to run them in parallel.
bash scripts/Persuasion.sh
job_status=$?

crc-job-stats

# Exit on the job's status, not crc-job-stats', so Slurm reports failures.
exit $job_status
