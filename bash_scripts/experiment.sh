#!/bin/bash
source '/cluster/home/merler/miniconda3/bin/activate' '/cluster/home/merler/miniconda3/envs/test'
wandb login

python -u /cluster/home/merler/graph-super-resolution-pan/run_train.py\
 --dataset pan\
 --data-dir /cluster/scratch/merler/data \
 --save-dir /cluster/scratch/merler/code/saved_models_cluster \
 --wandb \
 --subset schweiz_random_200 \
 --scaling 32\
 --batch-size 32

 python -u /cluster/home/merler/graph-super-resolution-pan/run_train.py --dataset pan --data-dir /cluster/scratch/merler/data  --save-dir /cluster/home/merler/saved_models  --wandb  --subset test --scaling 32
 