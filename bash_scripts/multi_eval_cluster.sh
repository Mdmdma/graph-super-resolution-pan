#!/bin/bash

# This script is used to evaluate multiple trained checkpoints on a given dataset.
# The folder where the models lie and the selected model range has to be given as arguments.

# Sample usage: ./multi_eval.sh /scratch2/merler/code/saved_models/pan 17 21

# Function to generate a Slurm job script for a single evaluation run
generate_job_script() {
    local model_path=$1
    local scaling=$2
    local training_mode=$3
    local batch_size=$4
    local crop_size=$5
    local job_script=$(mktemp)
    #training_mode="graph"

    cat << EOF > "$job_script"
#!/bin/bash
#SBATCH -A es_schin
#SBATCH --time=06:06:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10G
source '/cluster/home/merler/miniconda3/bin/activate' '/cluster/home/merler/miniconda3/envs/test'

python /cluster/home/merler/graph-super-resolution-pan/run_eval.py \
    --checkpoint $model_path \
    --dataset pan \
    --data-dir /cluster/scratch/merler/data  \
    --subset schweiz_random_200 \
    --scaling $scaling \
    --crop-size $crop_size \
    --batch-size $batch_size \
    --training_mode $training_mode
EOF

    # Convert the job script to UNIX line endings
    dos2unix "$job_script" > /dev/null 2>&1

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
for (( i=$start_number; i<=$stop_number; i++ ))
do
    experiment_path="$folder_path/experiment_$i"
    args_file="$experiment_path/args.csv"
    model_path="$experiment_path/last_model.pth"
    
    # Check if the args.csv file exists
    if [ -f "$args_file" ]; then
        scaling=$(awk -F',' '/scaling/ {print $2}' "$args_file")
        training_mode=$(awk -F',' '/training_mode/ {print $2}' "$args_file")
        batch_size=$(awk -F',' '/batch_size/ {print $2}' "$args_file")
        crop_size=$(awk -F',' '/crop_size/ {print $2}' "$args_file")

        #this line is needed to strip a \r from the return. This is hard to debug, as the \r is not wisible in print statements
        training_mode=$(echo "$training_mode" | tr -d '\r')
        batch_size=$(echo "$batch_size" | tr -d '\r')
        if [ -z "$training_mode" ]; then
            echo "no training mode in the args file, use w/o-graph as the traing mode"
            training_mode="w/o-graph"
            echo "$training_mode"
        else
            echo "traingmode was detected in the arfs file"
            echo "$training_mode"
        fi

        echo "Scaling: $scaling"
        echo "Experiment Path: $experiment_path"
        echo "Model Path: $model_path"
        echo "Training Mode: $training_mode"
        echo "Batch_size: $batch_size"
        echo "Crop_size: $crop_size"
        
        job_script=$(generate_job_script "$model_path" "$scaling" "$training_mode" "$batch_size" "$crop_size")
        sbatch "$job_script"
        rm "$job_script"

    else
        echo "File $args_file not found in $experiment_path"
    fi
done