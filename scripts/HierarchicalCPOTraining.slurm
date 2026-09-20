#!/bin/bash
#SBATCH --job-name=hcpo_llama3
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=0-01:00:00
#SBATCH --output=hcpo_llama3_%j.out
#SBATCH --mail-user=aya34@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge
module load python/ondemand-jupyter-python3.11

# `conda activate` needs a shell function this module never initializes;
# `source activate` uses the script on PATH and does work here.
source activate /ix/akovashka/aya34/envs/SensoryAds/

: "${HF_TOKEN:?set 'export HF_TOKEN=...' in ~/.bashrc on the cluster; do not commit it}"
export HF_HOME=/ix/akovashka/aya34/HF_CACHE
export TOKENIZERS_PARALLELISM=false

cd $SLURM_SUBMIT_DIR

# No srun here: accelerate launch spawns one process per GPU itself, so wrapping
# it in an srun with multiple tasks would start the whole training N times over.
bash scripts/HierarchicalCPOTraining.sh

crc-job-stats
