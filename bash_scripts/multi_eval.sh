#!/bin/bash

# This script is used to evaluate multiple trained checkpoints on a given dataset.
# the folder, where the models lie and the selected model range has to be given as arguments.

# sample use ./multi_eval.sh /scratch2/merler/code/saved_models/pan 17 21
#/scratch2/merler/code/saved_models/pan/experiment_20/args.csv

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <folder_path> <start_number> <stop_number>"
    exit 1
fi

folder_path=$1
start_number=$2
stop_number=$3

# Loop through the range of numbers
for (( i=$start_number; i<=$stop_number; i++ ))
do
    experiment_path="$folder_path/experiment_$i"
    args_file="$experiment_path/args.csv"
    model_path="$experiment_path/last_model.pth"

    # Check if the args.csv file exists
    if [ -f "$args_file" ]; then
        scaling=$(awk -F',' '/scaling/ {print $2}' "$args_file")
        echo $scaling
        echo $experiment_path
        echo $model_path
        
        python /scratch2/merler/code/graph-super-resolution-pan/run_eval.py \
        --checkpoint $model_path \
        --dataset pan \
        --data-dir /scratch2/merler/code/data \
        --subset schweiz_random_200 \
        --scaling $scaling

    else
        echo "File $args_file not found"
    fi
done