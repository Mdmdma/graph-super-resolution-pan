import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Sample use python visualize_dataset_location.py scratch2/merler/code/data/pan10/images/schweiz_random_150

RANDOM_STATE = 42
# Transformer to convert from the swiss coordinate grit to longitude and latidute
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326")

#select the dataset to be plotted
directory = sys.argv[1]

train_ratio = sys.argv[2]
val_ratio = sys.argv[3]
test_ratio = 1 - (train_ratio + val_ratio)
assert train_ratio < 1 and val_ratio < 1 and test_ratio < 1

# Get a list of all filenames in the directory
filenames = os.listdir(directory)


train_files, test_files = train_test_split(filenames, train_size=train_ratio+val_ratio, random_state=RANDOM_STATE)
val_files, test_files = train_test_split(test_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=RANDOM_STATE)
     # Splitting the image files list into train, val, and test based on the ratios

# Create a DataFrame from the list of filenames
df = pd.DataFrame(filenames, columns=['filename'])
df['color'] = ['black' if filename in train_files else 'red' if filename in val_files else 'green' for filename in df['filename']]

# Ectract the coordinates out of the filenames
df[['part1', 'part2', 'part3', 'part4', 'part5']] = df['filename'].str.split('_', expand=True)
df['part1'] = df['part3'].str.extract('(\d+)', expand=False)
df[['x', 'y']] = df['part3'].str.split('-', expand=True).astype(int)*1000
df[ 'lat'] = df.apply(lambda row: transformer.transform(row['x'], row['y'])[0], axis=1)
df[ 'long'] = df.apply(lambda row: transformer.transform(row['x'], row['y'])[1], axis=1)
# Drop the columns you don't need
df = df.drop(columns=['filename', 'part1', 'part2', 'part3', 'part4', 'part5'])


# Generate a plot with all the sites
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})
    
# Set the extent to focus on Switzerland (approximate bounds)
ax.set_extent([5.8, 10.6, 45.8, 47.9], crs=ccrs.PlateCarree())

# Add features for context
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=True)
ax.set_title('Location of the data tiles')

# Plot each coordinate with optional labels
df.apply(lambda row: ax.plot(row['long'], row['lat'] , marker='s', color=row['color'], transform=ccrs.Geodetic(),markersize=3), axis=1)

plt.show()