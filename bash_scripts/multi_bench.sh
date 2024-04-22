#!/bin/bash

# Bash sciript to runn a benchmark on multiple scaling factors and upscalers in one go.
# The results are written to a .csv file named benchmark_results

# sample use ./multi_eval.sh /scratch2/merler/code/saved_models/pan
#/scratch2/merler/code/saved_models/pan/experiment_20/args.csv

# Check if the required arguments are provided

upscaling_factors=(1 2 4 8 16 32)
# "pansharpen_pixel_average" "scale_mean_values" 
upsamplers=("bicubic_upsample")
dataset="schweiz_random_200"

# Loop through the range of numbers
for upsampler in "${upsamplers[@]}"
do 
    for upscaling_factor in ${upscaling_factors[@]}
    do
        echo $upsampler
        echo $upscaling_factor
        python /scratch2/merler/code/graph-super-resolution-pan/benchmark.py \
        --dataset pan \
        --subset $dataset \
        --data-dir /scratch2/merler/code/data \
        --scaling $upscaling_factor \
        --upsampler $upsampler
    done
done

