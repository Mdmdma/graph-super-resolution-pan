import run_eval
import benchmark
import pandas as pd
import numpy as np
import subprocess

if __name__ == '__main__':

    file_path = '/cluster/scratch/merler/data/pan10/evaluation_results/results_eval_without_graph_complete_new.csv'



    df = pd.read_csv(file_path)
    df = df[df['training_mode'] == 'w/o-graph']
    df = df.drop_duplicates()
    df = df.sort_values(by=['batch_size', 'scaling_factor'])
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'training_mode': 'model'}, inplace=True)
   
    df = df[df['batch_size'] == 8]

    df_n = df
    print(df)

    subset  = 'test_small'
    datadir = '/cluster/scratch/merler/data'

    for index, row in df.iterrows():
        scaling_factor = row['scaling_factor']
        batch_size = row['batch_size']
        crop_size = row['crop_size']

        checkpoint = row['checkpoint']
        model = row['model']

        
        command = f"python run_eval.py --checkpoint {checkpoint} --dataset {'pan'} --data-dir {datadir} --subset {subset} --batch-size {4} --crop-size {crop_size} --scaling {int(scaling_factor)} --training_mode {model} "
        print('run')
        subprocess.run(command, shell=True)