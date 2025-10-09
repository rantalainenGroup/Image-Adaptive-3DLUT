#! /bin/bash

# Usage:
#   ./run_unpaired.sh [CONFIG_PATH] [GPUS]
# Examples:
#   ./run_unpaired.sh configs/exp_01.yaml
#   ./run_unpaired.sh configs/exp_01.yaml 4

# Generate log file name with timestamp
current_time=$(date "+%Y-%m-%d_%H-%M")

# CHANGE!!!!
TASK='3_evaluation'
log_file="/home/bojing/Image-Adaptive-3DLUT/logs/${TASK}_${current_time}.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $@"  | tee -a "$log_file"
}



set -e

#CONFIG="${1:-configs/exp_01.yaml}"
#GPUS="${2:-4}"
# go to project root (parent of bin)
cd "$(dirname "$0")/.."
CONFIG="/home/bojing/Image-Adaptive-3DLUT/configs/exp_03.yaml"
GPUS=4


# --- timing (simple) ---
SECONDS=0

python /home/bojing/Image-Adaptive-3DLUT/evaluation_new.py \
  --config "$CONFIG" \
  --gpus "$GPUS" | tee -a "$log_file"    

echo "Time used: ${SECONDS}s"