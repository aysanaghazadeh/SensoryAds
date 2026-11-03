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
# The Xet backend's parallel connections get reset on this network ("tls handshake eof"),
# so fall back to the plain HTTPS path, which resumes the same .incomplete blobs.
export HF_HUB_DISABLE_XET=1
BLOBS=$HF_HOME/hub/models--OpenGVLab--InternVL3_5-8B/blobs

# LFS blobs are named after their own SHA-256, so a finished download can be verified by rehashing it.
# Deletes any blob whose hash doesn't match its name (so the next attempt re-fetches it); returns 1 if any.
verify_blobs() {
    local bad=0 f
    for f in $(ls "$BLOBS" | grep -E '^[0-9a-f]{64}$'); do
        if [ "$(sha256sum "$BLOBS/$f" | cut -d' ' -f1)" = "$f" ]; then
            echo "OK  $f"
        else
            echo "BAD $f (deleted)"; rm -f "$BLOBS/$f"; bad=1
        fi
    done
    return $bad
}

job_status=1
for attempt in $(seq 1 30); do
    echo "attempt $attempt: $(du -sb $BLOBS 2>/dev/null | cut -f1) bytes on disk"
    timeout 10m hf download OpenGVLab/InternVL3_5-8B --max-workers 4
    job_status=$?
    if [ $job_status -eq 0 ]; then
        verify_blobs && break
        job_status=1
    fi
    sleep 5
done

crc-job-stats
exit $job_status
