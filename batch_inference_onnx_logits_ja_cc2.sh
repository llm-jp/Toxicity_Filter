#!/bin/bash

mkdir -p toxic_scores2

# Array mapping GPU IDs to their respective file lists
declare -A gpu_file_lists=(
    [0]=file_list2_01
    [1]=file_list2_02
    [2]=file_list2_03
    [3]=file_list2_04
)

# Function to run inference on a specific GPU and file list
run_inference() {
    local gpu_id=$1
    local file_list=$2

    echo "Starting inference on GPU $gpu_id with file list $file_list..."

    parallel -j1 'CUDA_VISIBLE_DEVICES='"$gpu_id"' \
        python3 batch_inference_onnx_logits.py \
        --model_path deberta.onnx \
        --tokenizer_model final_model/ \
        --input_data_path ja_cc2/{}.jsonl.gz \
        --output_path toxic_scores2/{}.txt \
        --batch_size 512 \
        --max_length 256 \
        2> toxic_scores2/{}.err \
        > toxic_scores2/{}.out' :::: "$file_list"

    echo "Completed inference on GPU $gpu_id."
}

# Iterate over each GPU and its corresponding file list
for gpu_id in "${!gpu_file_lists[@]}"; do
    file_list="${gpu_file_lists[$gpu_id]}"
    
    # Run each inference task in the background
    run_inference "$gpu_id" "$file_list" &
done

# Wait for all background processes to finish
wait

echo "All inference tasks have been completed successfully."

