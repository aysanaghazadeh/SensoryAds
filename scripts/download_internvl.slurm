#!/bin/bash
#SBATCH --job-name=download_internvl
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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

# A stalled connection makes hf download hang without erroring, so cap each attempt and retry.
# Every attempt resumes from the .incomplete blobs, so nothing is lost by restarting.
export HF_HUB_DOWNLOAD_TIMEOUT=60
job_status=1
for attempt in $(seq 1 30); do
    echo "attempt $attempt: $(du -sb $HF_HOME/hub/models--OpenGVLab--InternVL3_5-8B/blobs 2>/dev/null | cut -f1) bytes on disk"
    timeout 10m hf download OpenGVLab/InternVL3_5-8B --max-workers 8
    job_status=$?
    [ $job_status -eq 0 ] && break
    sleep 5
done

crc-job-stats
exit $job_status
