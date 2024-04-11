import shutil
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import tifffile
# Sample usage: python preprocessor_raw.py scratch2/merler/code/data/pan10/images 0.8 0.1 0.1

def split_dataset_into_subfolders(folder_path, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Splits files in each subfolder of the given folder into train, test, and val subfolders randomly.
    The files from all subfolders are merged together in the respective train, val, and test folders.
    
    Parameters:
    - folder_path: Path to the folder containing subfolders with files to be split.
    - train_size: Proportion of the dataset to include in the train split.
    - val_size: Proportion of the dataset to include in the validation split.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed used by the random number generator for reproducibility.
    """
    
    folder_path = Path(folder_path)
    subfolders = [f for f in folder_path.iterdir() if f.is_dir()]

    # Create train, val, and test folders if they don't exist
    for split in ['train', 'val', 'test']:
        (folder_path / split).mkdir(exist_ok=True)
    
    for subfolder in subfolders:
        files = [file for file in subfolder.iterdir() if file.is_file()]
        filenames = [file.name for file in files]
        
        # Create train, val, and test splits for the current subfolder
        train_files, test_files = train_test_split(filenames, train_size=train_size+val_size, random_state=random_state)
        val_files, test_files = train_test_split(test_files, train_size=val_size/(val_size+test_size), random_state=random_state)
        
        # Function to move files to their respective merged subfolder
        def move_files(files, split_name):
            for filename in files:
                source_path = subfolder / filename
                target_path = folder_path / split_name / filename

                if not target_path.exists():
                    shutil.move(source_path, target_path)
                else:
                    print(f"File {target_path} already exists. Consider manually handling this case.")

                # Load the file as a numpy array
                if filename.endswith('.tif'):
                    data = tifffile.imread(target_path)
                elif filename.endswith('.npy'):
                    continue
                else:
                    print(f"Unsupported file type: {filename}")
                    continue

                # Save the numpy array to the target path
                np.save(target_path.with_suffix('.npy'), data)
                target_path.unlink()

                    
        # Move files to their respective splits at the root level
        move_files(train_files, 'train')
        move_files(val_files, 'val')
        move_files(test_files, 'test')
    
    print(f"Dataset from subfolders merged into train, val, and test subfolders at '{folder_path}'")

# Example usage
split_dataset_into_subfolders(Path(sys.argv[1]).resolve(), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

