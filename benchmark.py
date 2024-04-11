import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from arguments import eval_parser
from model import GraphSuperResolutionNet
from data import PanDataset
from utils import to_cuda

from pan_additions.naive_upsampling import pansharpen_pixel_average, scale_mean_values, bicubic_upsample, visualize_tensor, save_tensor_as_image

# Sample use python benchmark.py --dataset pan --data-dir /scratch2/merler/code/data --subset test
class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)

        self.model = GraphSuperResolutionNet(args.scaling, args.crop_size, args.feature_extractor)
        self.model.cuda().eval()

        torch.set_grad_enabled(False)


    def evaluate(self):
        test_stats = defaultdict(float)

        for sample in tqdm(self.dataloader, leave=False):
            sample = to_cuda(sample)

            # choose pansharpening method to use
            
            # output = pansharpen_pixel_average(sample)
            # output = scale_mean_values(sample)
            output = bicubic_upsample(sample)
            # output = gram_schmidt(sample)

            # visualize the output
            # picture = output['y_pred']
            # visualize_tensor(picture, title='naive_upsampling')
            # save_tensor_as_image(picture, ' pixels_scaled_to_match_bw_upsampled')


            _, loss_dict = self.model.get_loss(output, sample)

            for key in loss_dict:
                test_stats[key] += loss_dict[key]

        return {k: v / len(self.dataloader) for k, v in test_stats.items()}

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
            dataset = PanDataset(os.path.join(args.data_dir, 'pan10/images_processed/', args.subset), **data_args, split='test')
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print('Evaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(stats)