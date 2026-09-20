#!/bin/bash
#SBATCH --job-name=hcpo_llama3
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=0-05:00:00
#SBATCH --output=hcpo_llama3_%j.out
#SBATCH --mail-user=aya34@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge

# Calling the env's interpreter directly: conda's activate relies on a shell
# function that never gets initialized in a non-interactive batch shell, which
# silently leaves you on the system python instead.
export PYTHON=/ix/akovashka/aya34/envs/SensoryAds/bin/python

: "${HF_TOKEN:?set 'export HF_TOKEN=...' in ~/.bashrc on the cluster; do not commit it}"
export HF_HOME=/ix/akovashka/aya34/HF_CACHE
export TOKENIZERS_PARALLELISM=false
# Python block-buffers stdout when it isn't a TTY, so prints would otherwise
# only reach the .out file in chunks (or at process exit).
export PYTHONUNBUFFERED=1
# Compute nodes have no outbound internet, so any Hub lookup hangs instead of
# failing fast. Requires the weights to already be in HF_HOME.
export HF_HUB_OFFLINE=1

cd $SLURM_SUBMIT_DIR

# No srun here: accelerate launch spawns one process per GPU itself, so wrapping
# it in an srun with multiple tasks would start the whole training N times over.
bash scripts/HierarchicalCPOTraining.sh
training_status=$?

crc-job-stats

# Exit on training's status, not crc-job-stats', so Slurm reports failures.
exit $training_status
