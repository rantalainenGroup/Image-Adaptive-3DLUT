#!/bin/bash

# Generate log file name with timestamp
current_time=$(date "+%Y-%m-%d_%H-%M")
# CHANGE!!!!
TASK='inference_uni'
log_file="/home/ferbue/Image-Adaptive-3DLUT/logs/4_${TASK}_${current_time}.log"

# Function to log messages with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$log_file"
}


# Start the timer
start_time=$(date +%s)

# activate env
eval "$(conda shell.bash hook)"
conda activate chime_env_v2_lut
if [ $? -eq 0 ]; then
    echo "Environment 'chime_env_v2' activated successfully."
else
    echo "Failed to activate environment 'chime_env_v2'."
fi

# Define the number of GPUs to use
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

dataset_name="scanb_malmo"
checkpoint="/home/ferbue/Model_weights/UNI/pytorch_model.bin"
data_out_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_20"

#df_path="/mnt/ssd/bojing/Image-Adaptive-3DLUT/dataframes/scanb_malmo_philips_xr_test_macenko.csv"  # change
#df_feature_name="scanb_malmo_philips_xr_test_macenko_uni" # changes

# df_path="/mnt/ssd/bojing/Image-Adaptive-3DLUT/dataframes/scanb_malmo_philips_xr_test.csv"  # change
# df_feature_name="scanb_malmo_philips_xr_test_uni" # changes


# df_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp20.csv"  # change
# df_feature_name="scanb_malmo_lut_exp20_uni" # changes

df_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp20_bothscanners.csv" # change
df_feature_name="scanb_malmo_lut_exp20_bothscans_uni" # changes


batch_size=256   # change
tile_size=224
out_dim=1024
model_name='uni'
tile_path='png_tile_path' # For LUT
# tile_path='crude_tile_path' # For original tiles
tile_name='tile_name'
#tile_path='macenko_tile_path'  # change

log "Executing script at version: $(git log -1 --format="%H" -- /home/ferbue/Image-Adaptive-3DLUT/run_inference_lut.py)"

python /home/ferbue/Image-Adaptive-3DLUT/run_inference_lut.py \
    --dataset_name $dataset_name \
    --data_out_path $data_out_path \
    --df_path $df_path \
    --tile_path $tile_path \
    --tile_name $tile_name \
    --tile_size $tile_size \
    --out_dim $out_dim \
    --model_name $model_name \
    --pretrained \
    --checkpoint  $checkpoint \
    --batch_size $batch_size \
    --df_feature_name "$df_feature_name" 2>&1 | tee -a "$log_file"

log "Script execution completed at $(date)"
# Calculate and log the elapsed time
end_time=$(date +%s)

# Calculate and log the elapsed time
elapsed_time=$((end_time - start_time))
log "Total execution time: $elapsed_time seconds"