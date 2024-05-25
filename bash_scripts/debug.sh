#!/bin/bash
source '/cluster/home/merler/miniconda3/bin/activate' '/cluster/home/merler/miniconda3/envs/test'
wandb login

python -u /cluster/home/merler/graph-super-resolution-pan/run_train.py\
 --dataset pan\
 --data-dir /cluster/scratch/merler/data \
 --save-dir /cluster/home/merler/saved_models \
 --wandb \
 --subset test \
 --scaling 32