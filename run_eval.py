import os
import argparse
from collections import defaultdict
import time
import csv
import numpy as np

import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from arguments import eval_parser
from model import GraphSuperResolutionNet, GraphSuperResolutionNet_ex, GraphSuperResolutionNet_ex_plus
from data import PanDataset
from utils import to_cuda
from pan_additions.naive_upsampling import visualize_tensor, save_tensor_as_image

# Sample use python run_eval.py --checkpoint /scratch2/merler/pan/experiment_23/best_model.pth --dataset pan --data-dir /scratch2/merler/code/data --subset schweiz_random_200

# Cluster use python run_eval.py --checkpoint /cluster/scratch/merler/code/saved_models_cluster_graph/pan/experiment_0/best_model.pth --dataset pan --data-dir /cluster/scratch/merler/data --subset test --data_output_dir /cluster/scratch/merler/data/pan10/evaluation_results/test --crop-size 64 --training_mode graph    

class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)
        if args.training_mode == 'w/o-graph':
            self.model = GraphSuperResolutionNet(args.scaling, args.crop_size, args.feature_extractor, True)
        elif args.training_mode == 'graph':
            self.model = GraphSuperResolutionNet_ex(args.scaling, args.crop_size, args.feature_extractor)
        elif args.training_mode == 'graph-plus':
            self.model = GraphSuperResolutionNet_ex_plus(args.scaling, args.crop_size, args.feature_extractor)
        else:
            raise NotImplementedError(f'Training mode {args.training_mode}')
        
        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

        torch.set_grad_enabled(False)

    def evaluate(self):
        test_stats = defaultdict(float)
        problemcounter = 0
        counter = 0
        for sample in tqdm(self.dataloader, leave=False):
            sample = to_cuda(sample)

            output = self.model(sample)

            picture = output['y_pred']
            #picture = sample['y']
            # visualize_tensor(picture, title='model_upsampling')
            if counter < 8:
                title = f"{'w_o_graph'} {args.scaling} {counter}"
                #title = 'original'
                save_tensor_as_image(picture, title=title)
                
            counter +=1
            if counter == 9:  
                exit()
            

            


            _, loss_dict = self.model.get_loss(output, sample)
            for key in loss_dict:
                if key == 'sam':
                    if np.isnan(loss_dict[key]):
                        problemcounter += 1
                        print('problem')
                        print(problemcounter)
                        print(loss_dict[key])
                        loss_dict[key] = 0
                test_stats[key] += loss_dict[key]

        return {k: v / (len(self.dataloader)-problemcounter) for k, v in test_stats.items()}

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': 0,
            'do_horizontal_flip': False,
            'crop_valid': True,
            'crop_deterministic': True,
            'scaling': args.scaling
        }

        if args.dataset == 'pan':
            if args.crop_size == 64:
                data_args['crop_deterministic'] = True # this is needed to keep the evaluation time resonable. 
            dataset = PanDataset(os.path.join(args.data_dir, 'pan10/images_processed/', args.subset), **data_args, split='test')
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())
    print(args)

    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print('Evaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(stats)

    # Writes the resluts in a csv file 
    output_dir = args.data_output_dir
    my_dict = stats
    scaling_factor = args.scaling
    checkpoint = args.checkpoint
    subset = args.subset
    batch_size = args.batch_size
    training_mode = args.training_mode
    crop_size = args.crop_size

    # Create the 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Construct the path to the CSV file
    csv_file_path = os.path.join(output_dir, 'results_eval.csv')

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = list(my_dict.keys()) + ['scaling_factor', 'batch_size','crop_size', 'subset', 'checkpoint','training_mode']  # Define the field names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row if the file is empty
        if os.path.getsize(csv_file_path) == 0:
            writer.writeheader()

        # Write a new row with dictionary values and additional arguments
        row_dict = my_dict.copy()
        row_dict['scaling_factor'] = scaling_factor
        row_dict['batch_size'] = batch_size
        row_dict['crop_size'] = crop_size
        row_dict['subset'] = subset
        row_dict['checkpoint'] = checkpoint
        row_dict['training_mode'] = training_mode
        writer.writerow(row_dict)