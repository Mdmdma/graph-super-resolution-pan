#!/bin/bash

# Function to generate a Slurm job script for a single evaluation run
generate_job_script() {
    local folder_path=$1
    local experiment_number=$2
    local job_script=$(mktemp)

    experiment_path="$folder_path/experiment_$experiment_number"
    args_file="$experiment_path/args.csv"
    model_path="$experiment_path/last_model.pth"

    # Check if the args.csv file exists
    if [ -f "$args_file" ]; then
        scaling=$(awk -F',' '/scaling/ {print $2}' "$args_file")
        training_mode=$(awk -F',' '/training_mode/ {print $2}' "$args_file")
        batch_size=$(awk -F',' '/batch_size/ {print $2}' "$args_file")
        crop_size=$(awk -F',' '/crop_size/ {print $2}' "$args_file")

        # Overwrite training_mode
        training_mode="graph-plus"

        cat << EOF > "$job_script"
#!/bin/bash
#SBATCH -A es_schin
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10G

python /cluster/home/merler/graph-super-resolution-pan/run_eval.py \
    --checkpoint $model_path \
    --dataset pan \
    --data-dir /cluster/scratch/merler/data \
    --subset test_small \
    --scaling $scaling \
    --training_mode $training_mode \
    --batch-size $batch_size \
    --crop-size $crop_size
EOF
    else
        echo "File $args_file not found"
        job_script=""
    fi

    echo "$job_script"
}

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <folder_path> <start_number> <stop_number>"
    exit 1
fi

folder_path=$1
start_number=$2
stop_number=$3

# Loop through the range of numbers
for ((i=$start_number; i<=$stop_number; i++))
do
    job_script=$(generate_job_script "$folder_path" "$i")
    if [ -n "$job_script" ]; then
        sbatch "$job_script"
        rm "$job_script"
    fi
done