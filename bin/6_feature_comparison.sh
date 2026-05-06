#!/bin/bash

# Generate log file name with timestamp


# Function to log messages with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$log_file"
}

log "Started job with PID: $$"


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

current_time=$(date "+%Y-%m-%d_%H-%M")
# CHANGE!!!!-----------------------------------------------------------------------------------------------------------------
TASK='Feature comparison'
log_file="/home/ferbue/Image-Adaptive-3DLUT/logs/6_${TASK}_${current_time}.log"

exp_id="20" # change
source_domain="PHILIPS" # change
target_domain="XR" # change
feat_csv="scanb_malmo_source_exp20_bothscans_uni.pkl" # change
feat_lut_csv="scanb_malmo_lut_exp20_bothscans_uni.pkl" # change
feat_macenko_csv="scanb_malmo_macenko_bothscans_uni.pkl" # change
test_df_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp20_bothscanners_v2.csv" # change


log "Executing script at version: $(git log -1 --format="%H" -- /home/ferbue/Image-Adaptive-3DLUT/run_inference_lut.py)"


python /home/ferbue/Image-Adaptive-3DLUT/Feature_comparison.py \
    --exp_id $exp_id \
    --source_domain $source_domain \
    --target_domain $target_domain \
    --feat_csv $feat_csv \
    --feat_lut_csv $feat_lut_csv \
    --feat_macenko_csv $feat_macenko_csv \
    --test_df_path $test_df_path 2>&1 | tee -a "$log_file"



current_time=$(date "+%Y-%m-%d_%H-%M")
# CHANGE!!!!!-----------------------------------------------------------------------------------------------------------------
TASK='Feature comparison'
log_file="/home/ferbue/Image-Adaptive-3DLUT/logs/6_${TASK}_${current_time}.log"

exp_id="21" # change
source_domain="S360" # change
target_domain="XR" # change
feat_csv="scanb_malmo_source_s360_xr_uni.pkl" # change
feat_lut_csv="scanb_malmo_lut_s360_xr_uni.pkl" # change
feat_macenko_csv="scanb_malmo_macenko_s360_xr_uni.pkl" # change
test_df_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp21_bothscanners_v2.csv" # change


log "Executing script at version: $(git log -1 --format="%H" -- /home/ferbue/Image-Adaptive-3DLUT/run_inference_lut.py)"


python /home/ferbue/Image-Adaptive-3DLUT/Feature_comparison.py \
    --exp_id $exp_id \
    --source_domain $source_domain \
    --target_domain $target_domain \
    --feat_csv $feat_csv \
    --feat_lut_csv $feat_lut_csv \
    --feat_macenko_csv $feat_macenko_csv \
    --test_df_path $test_df_path 2>&1 | tee -a "$log_file"



current_time=$(date "+%Y-%m-%d_%H-%M")
# # CHANGE!!!!-----------------------------------------------------------------------------------------------------------------
TASK='Feature comparison'
log_file="/home/ferbue/Image-Adaptive-3DLUT/logs/6_${TASK}_${current_time}.log"

exp_id="22" # change
source_domain="APERIO" # change
target_domain="XR" # change
feat_csv="scanb_malmo_source_aperio_xr_uni.pkl" # change
feat_lut_csv="scanb_malmo_lut_aperio_xr_uni.pkl" # change
feat_macenko_csv="scanb_malmo_macenko_aperio_xr_uni.pkl" # change
test_df_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp22_bothscanners_v2.csv" # change


log "Executing script at version: $(git log -1 --format="%H" -- /home/ferbue/Image-Adaptive-3DLUT/run_inference_lut.py)"


python /home/ferbue/Image-Adaptive-3DLUT/Feature_comparison.py \
    --exp_id $exp_id \
    --source_domain $source_domain \
    --target_domain $target_domain \
    --feat_csv $feat_csv \
    --feat_lut_csv $feat_lut_csv \
    --feat_macenko_csv $feat_macenko_csv \
    --test_df_path $test_df_path 2>&1 | tee -a "$log_file"


log "Script execution completed at $(date)"
# Calculate and log the elapsed time
end_time=$(date +%s)

# Calculate and log the elapsed time
elapsed_time=$((end_time - start_time))
log "Total execution time: $elapsed_time seconds"