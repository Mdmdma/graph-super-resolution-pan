#!/bin/bash
module load eth_proxy 
sbatch -A es_schin --time=48:00:00 --ntasks=16 --mem-per-cpu=4G --gpus=1 --gres=gpumem:10G experiment.sh
#srun -A es_schin --time=02:00:00 --ntasks=16 --mem-per-cpu=4G --gpus=1 --gres=gpumem:10G experiment.sh