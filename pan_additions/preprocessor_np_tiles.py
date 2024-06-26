import sys
import os
from pathlib import Path
import cv2

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# python preprocesser_np_tiles.py /scratch2/merler/code/data/pan10/images_processed/schweiz_random_250 0.5 0.25 


def split_image_into_tiles(image, tile_size):
    tiles = []
    height, width, _ = image.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            left = x
            top = y
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)
            tile = image[top:bottom, left:right]
            tiles.append(tile)
    return tiles

def process_images(image_paths, output_file, tile_size=1000):
    num_images = len(image_paths)
    first_image = cv2.imread(str(image_paths[0]))
    height, width, channels = first_image.shape
    num_tiles_per_image = (height // tile_size) * (width // tile_size)
    total_tiles = num_images * num_tiles_per_image

    # Create a memory-mapped NumPy array to store the tiles
    memmap_tiles = np.memmap('temp', dtype=np.uint8, mode='w+', shape=(total_tiles, channels, tile_size, tile_size ))

    tile_index = 0
    for image_path in tqdm(image_paths, desc="Processing images", leave=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiles = split_image_into_tiles(image, tile_size)
        for tile in tiles:
            memmap_tiles[tile_index] = tile.transpose(2,0,1)
            tile_index += 1

    np.save(SAVE_PATH / output_file  , memmap_tiles)

    # Flush the memmap and delete the object
    memmap_tiles.flush()
    del memmap_tiles


# This script is used to preprocess the Pan dataset into numpy arrays for training
# The script takes in the path to the Pan dataset and the train, validation, and test ratios
# Sample usage: python preprocesser_np_tiles.py /scratch2/merler/code/data/pan10/images_processed/schweiz_random_250 0.5 0.25 


RANDOM_STATE=42


folder_path = Path(sys.argv[1]).resolve()
# Assuming the next two arguments are the train and validation ratios
train_ratio = float(sys.argv[2])
val_ratio = float(sys.argv[3])
# Ensure the test ratio is the remaining portion
test_ratio = 1 - (train_ratio + val_ratio)

# Validate the ratios sum to 1
assert train_ratio < 1 and val_ratio < 1 and test_ratio < 1

SAVE_PATH = folder_path / 'npy'
SAVE_PATH.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

# Using list comprehension with pathlib for more concise and readable code

data = {
    'train': [],
    'val': [],
    'test': []
}

for location_path in folder_path.iterdir():
    if location_path.is_dir() and not location_path.name == 'npy':
        image_files = [str(f) for f in location_path.iterdir() if f.is_file()]  # Fix the iteration over the location_path
        length = len(image_files)
        print(length)
        train_files, test_files = train_test_split(image_files, train_size=train_ratio, random_state=RANDOM_STATE)
        val_files, test_files = train_test_split(test_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=RANDOM_STATE)

        # Splitting the image files list into train, val, and test based on the ratios
        data['train'] = train_files
        data['val'] = val_files
        data['test'] = test_files

        print(data)


for split in tqdm(data, desc="Processing splits", leave=False):
    image_files = data[split]
    process_images(image_files, f'images_{split}.npy')
    print(f'{split}: {len(image_files)} images saved')

