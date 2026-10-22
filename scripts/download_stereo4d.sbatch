#!/bin/bash
#SBATCH --job-name=dl-stereo4d
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/group/video-tempo-control/VRStereoVideoGen/logs/%x_%j.out

# CPU-only; network/disk bound. Resumable: resubmit to continue.
mkdir -p /group/video-tempo-control/VRStereoVideoGen/logs
~/miniconda3/envs/tempocontrol/bin/python -u \
  /group/video-tempo-control/VRStereoVideoGen/scripts/download_stereo4d.py \
  --root /group/video-tempo-control/VRStereoVideoGen/data/stereo4d \
  --num_videos 2000 --max_duration 600 --workers 4 --sleep 5
