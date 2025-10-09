#! /bin/bash

# Usage:
#   ./run_unpaired.sh [CONFIG_PATH] [GPUS]
# Examples:
#   ./run_unpaired.sh configs/exp_01.yaml
#   ./run_unpaired.sh configs/exp_01.yaml 4

# Generate log file name with timestamp
current_time=$(date "+%Y-%m-%d_%H-%M")

# CHANGE!!!!
TASK='2_train'
log_file="/home/bojing/Image-Adaptive-3DLUT/logs/${TASK}_${current_time}.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $@"  | tee -a "$log_file"
}



set -e

#CONFIG="${1:-configs/exp_01.yaml}"
#GPUS="${2:-4}"
# go to project root (parent of bin)
cd "$(dirname "$0")/.."
CONFIG="/home/bojing/Image-Adaptive-3DLUT/configs/exp_02.yaml"
GPUS=4

SECONDS=0
python -u /home/bojing/Image-Adaptive-3DLUT/train_unpaired_new.py \
  --config "$CONFIG" \
  --gpus "$GPUS" | tee -a "$log_file"
echo "Time used: ${SECONDS}s"
