#! /bin/bash
current_time=$(date "+%Y-%m-%d_%H-%M")
TASK='2_train'
log_file="/home/bojing/Image-Adaptive-3DLUT/logs/${TASK}_${current_time}.log"

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') - $@" >> "$log_file"; }

set -e
cd "$(dirname "$0")/.."

# >>> Define your YAMLs here <<<
CONFIGS=(
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_01.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_04.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_05.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_06.yaml"
  "/home/bojing/Image-Adaptive-3DLUT/configs/exp_07.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  SECONDS=0
  log "Starting training with $CONFIG"
  python -u /home/bojing/Image-Adaptive-3DLUT/train_unpaired_new.py \
  --config "$CONFIG" >> "$log_file"
  log "Finished $CONFIG. Time used: ${SECONDS}s"
done


