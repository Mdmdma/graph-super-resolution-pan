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
from model import GraphSuperResolutionNet
from data import PanDataset
from utils import to_cuda
from pan_additions.naive_upsampling import visualize_tensor, save_tensor_as_image

# Sample use python run_eval.py --checkpoint /scratch2/merler/pan/experiment_23/best_model.pth --dataset pan --data-dir /scratch2/merler/code/data --subset schweiz_random_200

class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)

        self.model = GraphSuperResolutionNet(args.scaling, args.crop_size, args.feature_extractor)
        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

        torch.set_grad_enabled(False)

    def evaluate(self):
        test_stats = defaultdict(float)
        problemcounter = 0
        for sample in tqdm(self.dataloader, leave=False):
            sample = to_cuda(sample)

            output = self.model(sample)

            picture = output['y_pred']
            # visualize_tensor(picture, title='model_upsampling')
            # save_tensor_as_image(picture, 'model_upsampling')


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
            #'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        if args.dataset == 'pan':
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

    # Create the 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Construct the path to the CSV file
    csv_file_path = os.path.join('/scratch2/merler/code/data/pan10/evaluation_results', 'results_eval.csv')

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = list(my_dict.keys()) + ['scaling_factor', 'checkpoint', 'subset']  # Define the field names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row if the file is empty
        if os.path.getsize(csv_file_path) == 0:
            writer.writeheader()

        # Write a new row with dictionary values and additional arguments
        row_dict = my_dict.copy()
        row_dict['scaling_factor'] = scaling_factor
        row_dict['checkpoint'] = checkpoint
        row_dict['subset'] = subset
        writer.writerow(row_dict)