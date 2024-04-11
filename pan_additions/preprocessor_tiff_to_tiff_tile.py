import sys
import os
from pathlib import Path
import cv2

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# python preprocessor_tiff_to_tiff_tile.py /scratch2/merler/code/data/pan10/images_processed/basel 0.8 0.1
# to enabel the saving as a tiff file, the tiles are in the shape 1000x1000x3, but the image tensor in the training has to
# be in the shape 3x1000x1000


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

def process_images(image_paths, output_folder, tile_size=1000):
    os.makedirs(SAVE_PATH / output_folder, exist_ok=True)
    for image_path in tqdm(image_paths, desc="Processing images", leave=False):
        image = cv2.imread((image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiles = split_image_into_tiles(image, tile_size)
        tile_index = 0
        for tile in tqdm(tiles, desc="Processing tiles", leave=False):
            # Save tile as tif file
            tile_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile{tile_index}.tif"
            tile_filepath = os.path.join(SAVE_PATH / output_folder, tile_filename)
            tile = tile.copy()
            cv2.imwrite(str(tile_filepath), tile)
            tile_index += 1

# This script is used to preprocess the Pan dataset into numpy arrays for training
# The script takes in the path to the Pan dataset and the train, validation, and test ratios



RANDOM_STATE=42


folder_path = Path(sys.argv[1]).resolve()
# Assuming the next two arguments are the train and validation ratios
train_ratio = float(sys.argv[2])
val_ratio = float(sys.argv[3])
# Ensure the test ratio is the remaining portion
test_ratio = 1 - (train_ratio + val_ratio)

# Validate the ratios sum to 1
assert train_ratio < 1 and val_ratio < 1 and test_ratio < 1

SAVE_PATH = folder_path / 'tiff'
SAVE_PATH.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

# Using list comprehension with pathlib for more concise and readable code

data = {
    'train': [],
    'val': [],
    'test': []
}

for location_path in folder_path.iterdir():
    if location_path.is_dir() and not (location_path.name == 'npy' or location_path.name == 'npz' or location_path.name == 'tiff'):
        image_files = [str(f) for f in location_path.iterdir() if f.is_file()]  # Fix the iteration over the location_path
        train_files, test_files = train_test_split(image_files, train_size=train_ratio, random_state=RANDOM_STATE)
        val_files, test_files = train_test_split(test_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=RANDOM_STATE)

        # Splitting the image files list into train, val, and test based on the ratios
        data['train'].extend(train_files)
        data['val'].extend(val_files)
        data['test'].extend(test_files)


for split in tqdm(data, desc="Processing splits"):
    image_files = data[split]
    process_images(image_files, f'images_{split}')
    print(f'{split}: {len(image_files)} images saved')

