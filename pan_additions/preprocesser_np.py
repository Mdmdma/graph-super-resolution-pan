import sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# This script is used to preprocess the Pan dataset into numpy arrays for training
# The script takes in the path to the Pan dataset and the train, validation, and test ratios
# Sample usage: python preprocesser_np.py /scratch2/merler/code/data/pan10/images 0.8 0.1 0.1

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
random_state = 42


# Using list comprehension with pathlib for more concise and readable code

data = {
    'train': [],
    'val': [],
    'test': []
}

for location_path in folder_path.iterdir():
    if location_path.is_dir() and not location_path.name == 'npy':
        image_files = [f for f in location_path.iterdir() if f.is_file()]  # Fix the iteration over the location_path
        
        train_files, test_files = train_test_split(image_files, train_size=train_ratio+val_ratio, random_state=random_state)
        val_files, test_files = train_test_split(test_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=random_state)

        # Splitting the image files list into train, val, and test based on the ratios
        data['train'] = train_files
        data['val'] = val_files
        data['test'] = test_files

image_shape = (3, 10000, 10000)  # Assuming all images have the same shape<
dtype = np.uint8 


for split in data:
    image_files = data[split]
    num_images = len(image_files)*100
    
    # Create a memory-mapped array with shape to hold all images for the current split
    memmap_path = SAVE_PATH / f'images_{split}.dat'  # Temporary file for memory-mapped array
    images_memmap = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=(num_images,) + image_shape)
    
    file_index = 0
    # Incrementally load images and store them in the memory-mapped array
    for image_file in data[split]:
        # Load the image file using tifffile
        image = cv2.imread(image_file)
        # Convert the image to a numpy array and store it
        #selct the upper left 1000x1000 pixels
        #image = image[:1000, :1000, :]

        images_memmap[file_index] = image
        file_index += 1

    # Convert the memory-mapped array back to a regular numpy array (optional, for demonstration)
    # Note: This step is not memory-efficient and is only shown for completeness.
    # For large datasets, work directly with the memory-mapped array or save it directly.
    np.save(SAVE_PATH / f'images_{split}.npy', images_memmap)
    
    print(f'{split}: {len(images_memmap)} images saved')
    print('split done')
    
    # Clean up: delete the memory-mapped array's underlying file
    del images_memmap  # Delete the memmap object
    memmap_path.unlink()  # Delete the file from disk



