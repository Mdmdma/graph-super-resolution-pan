import os
import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import argparse
import random

# This function downloads files from a CSV file containing links to the files
# To run the script, use the following command: python downloader.py --link_csv /scratch2/merler/code/data/pan10/links/schweiz.csv --output_dir /scratch2/merler/code/data/pan10/images/schweiz_random_250

PICK_RANDOM_SUBSET = 200

def downloader(link_cvs, output_dir):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file        
    df = pd.read_csv(link_cvs)

    if PICK_RANDOM_SUBSET > 0:
        if len(df) < PICK_RANDOM_SUBSET:
            raise ValueError("The list has fewer than x entries.")
        
        # Select x random entries from the list
        df = df.sample(PICK_RANDOM_SUBSET, random_state=42)


    # Download the files
    for i in tqdm(range(len(df))):
        url = df.iloc[i, 0]
        file_name = os.path.basename(urlparse(url).path)
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            r = requests.get(url, allow_redirects=True)
            open(file_path, 'wb').write(r.content)

def main():
    parser = argparse.ArgumentParser(description="Download files listed in a CSV file.")
    parser.add_argument('--link_csv', type=str, help='Path to the CSV file containing the links to download.')
    parser.add_argument('--output_dir', type=str, help='Directory where the downloaded files will be stored.')

    args = parser.parse_args()

    downloader(args.link_csv, args.output_dir)

if __name__ == '__main__':
    main()

