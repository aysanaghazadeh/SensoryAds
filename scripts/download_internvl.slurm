#!/bin/bash
#SBATCH --job-name=download_internvl
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=download_internvl_%j.out
#SBATCH --mail-user=aya34@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge

: "${HF_TOKEN:?set 'export HF_TOKEN=...' in ~/.bashrc on the cluster; do not commit it}"
export HF_HOME=/ix/akovashka/aya34/HF_CACHE
export PATH=/ix/akovashka/aya34/envs/SensoryAds/bin:$PATH

# Not every cluster/partition has outbound internet; fail fast instead of hanging.
if ! curl -sS -m 20 -o /dev/null -I https://huggingface.co; then
    echo "no outbound internet from $(hostname); download from a login node or your laptop instead"
    exit 1
fi

# Resumes the .incomplete blobs left by the login-node attempt.
hf download OpenGVLab/InternVL3_5-8B --max-workers 8
job_status=$?

crc-job-stats
exit $job_status
