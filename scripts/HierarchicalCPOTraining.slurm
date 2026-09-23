#!/bin/bash
#SBATCH --job-name=hcpo_llama3
#SBATCH --cluster=gpu
#SBATCH --partition=h200
# gpu-n89 has twice given a fresh `ls` a false "file not found" for data that
# was present moments earlier from the login node (jobs 3994716, 3994726) -
# excluding it to test whether gpu-n90 doesn't have the same issue.
#SBATCH --exclude=gpu-n89
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
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

# Compute nodes have shown stale NFS lookups for files that exist and were
# recently written from the login node (seen twice now, always as a spurious
# FileNotFoundError). A fresh ls/stat here forces the client to re-resolve the
# path before Python ever opens it, which clears a stale negative cache entry.
echo "-- verifying data files resolve on this node --"
ls -la "$DATA_PATH/train/sensation_annotations_parsed_1200_train.json" \
       "$DATA_PATH/train/sensation_annotations_parsed_1200_train.csv" \
       "$EXPERIMENTS_PATH/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_train_ALL_real_description_generation.csv"

# No srun here: accelerate launch spawns one process per GPU itself, so wrapping
# it in an srun with multiple tasks would start the whole training N times over.
bash scripts/HierarchicalCPOTraining.sh
training_status=$?

crc-job-stats

# Exit on training's status, not crc-job-stats', so Slurm reports failures.
exit $training_status
