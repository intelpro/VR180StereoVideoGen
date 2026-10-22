#!/bin/bash
#SBATCH --job-name=s4d-erp
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --constraint=zone-sof1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/group/video-tempo-control/VRStereoVideoGen/logs/%x_%A_%a.out

# CPU-only. Stereo4D raw clips -> rectified 1024x1024 half-ERP L/R videos.
# One shard per array task (sharded by video id hash). Resumable: resubmit to
# pick up clips that errored or videos downloaded since the last run.
mkdir -p /group/video-tempo-control/VRStereoVideoGen/logs
~/miniconda3/envs/tempocontrol/bin/python -u \
  /group/video-tempo-control/VRStereoVideoGen/scripts/stereo4d_to_erp.py \
  --root /group/video-tempo-control/VRStereoVideoGen/data/stereo4d \
  --size 1024 --workers 8 --threads 4 \
  --shard "$SLURM_ARRAY_TASK_ID" --num_shards "$SLURM_ARRAY_TASK_COUNT"
