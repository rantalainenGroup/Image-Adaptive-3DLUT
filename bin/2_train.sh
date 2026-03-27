#! /bin/bash
current_time=$(date "+%Y-%m-%d_%H-%M")
TASK='2_train'
log_file="/home/ferbue/Image-Adaptive-3DLUT/logs/${TASK}_${current_time}.log"

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') - $@" >> "$log_file"; }

set -e
cd "$(dirname "$0")/.."

# >>> Define your YAMLs here <<<
CONFIGS=(
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_01.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_04.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_05.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_06.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_07.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_08.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_09.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_10.yaml"
  #"/home/bojing/Image-Adaptive-3DLUT/configs/exp_11.yaml"
  # "/home/bojing/Image-Adaptive-3DLUT/configs/exp_12.yaml"
  # "/home/bojing/Image-Adaptive-3DLUT/configs/exp_13.yaml"
  # "/home/bojing/Image-Adaptive-3DLUT/configs/exp_14.yaml"
  # "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_15.yaml"
  # "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_19.yaml"
  # "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_20.yaml"
  "/home/ferbue/Image-Adaptive-3DLUT/configs/exp_21.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  SECONDS=0
  log "Starting training with $CONFIG"
  python -u /home/ferbue/Image-Adaptive-3DLUT/train_unpaired_new.py \
  --config "$CONFIG" >> "$log_file"
  log "Finished $CONFIG. Time used: ${SECONDS}s"
done

