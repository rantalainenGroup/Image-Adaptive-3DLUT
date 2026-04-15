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
log_file="/home/ferbue/Image-Adaptive-3DLUT/logs/${TASK}_${current_time}.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $@"  >> "$log_file"
}

# activate env
eval "$(conda shell.bash hook)"
conda activate chime_env_v2_lut
if [ $? -eq 0 ]; then
    echo "Environment 'chime_env_v2' activated successfully."
else
    echo "Failed to activate environment 'chime_env_v2'."
fi


set -e

#CONFIG="${1:-configs/exp_01.yaml}"
#GPUS="${2:-4}"
# go to project root (parent of bin)
cd "$(dirname "$0")/.."
CONFIGS=(
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_03.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_04.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_05.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_06.yaml"
  # "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_15.yaml"
  # "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_20.yaml"
  # "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_21.yaml"
  "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_22.yaml"

)
BATCH_SIZE=256
# --- timing (simple) ---
for CONFIG in "${CONFIGS[@]}"; do
  SECONDS=0

  python /home/ferbue/Image-Adaptive-3DLUT/evaluation_new.py \
    --config "$CONFIG" --batch_size "$BATCH_SIZE" >> "$log_file"  
  log "Finished $CONFIG. Time used: ${SECONDS}s"
done