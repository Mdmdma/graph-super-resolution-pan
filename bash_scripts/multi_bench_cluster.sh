#!/bin/bash

# Function to generate a Slurm job script for a single benchmark run
generate_job_script() {
    local upsampler=$1
    local upscaling_factor=$2
    local job_script=$(mktemp)

    cat << EOF > "$job_script"
#!/bin/bash
#SBATCH -A es_schin
#SBATCH --time=06:06:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10G

python /cluster/home/merler/graph-super-resolution-pan/benchmark.py \
    --dataset pan \
    --subset test_small \
    --data-dir /cluster/scratch/merler/data \
    --scaling $upscaling_factor \
    --upsampler $upsampler \
    --crop-size 256
EOF

    echo "$job_script"
}

# Check if the required arguments are provided

upscaling_factors=(1 2)
upsamplers=("pansharpen_pixel_average" "bicubic_upsample" "pansharpen_hsv")

# Loop through the range of numbers
for upsampler in "${upsamplers[@]}"
do
    for upscaling_factor in "${upscaling_factors[@]}"
    do
        job_script=$(generate_job_script "$upsampler" "$upscaling_factor")
        sbatch "$job_script"
        rm "$job_script"
    done
done