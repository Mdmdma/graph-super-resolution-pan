import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=False, help='Checkpoint path to evaluate')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=2, help='Scaling factor')
parser.add_argument('--in-memory', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--feature-extractor', type=str, default='UResNet', help='Feature extractor for edge potentials')
parser.add_argument('--subset', type=str, required=True, help='Name of the subset') 
parser.add_argument('--upsampler', type=str, help='name of the upsampler for the comparison baselines') 
parser.add_argument('--data_output_dir', type=str, default='/cluster/scratch/merler/data/pan10/evaluation_results',required=False, help='defines_the_output_directory_for_the_results') 
parser.add_argument('--evaluation', type=str, default=True, help='enables extra metrics for testing')
parser.add_argument('--training_mode', type=str, default='w/o-graph', help='adds the graph layer: w/o-graph, graph, graph-plus')
