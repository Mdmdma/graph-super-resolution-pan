import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def pansharpen_pixel_average(sample):
    pan_hr = sample['guide']
    rgb_upsampled = sample['y_bicubic']
    rgb_bw = torch.mean(rgb_upsampled, 1, keepdim=True)
    pansharpen_factor = pan_hr / rgb_bw
    target = rgb_upsampled * pansharpen_factor
    return {'y_pred': target}

def scale_mean_values(sample):
    rgb_hr = sample['y']
    target = rgb_hr * 0.5
    return {'y_pred': target}

def bicubic_upsample(sample):
    y_bicubic = sample['y_bicubic']
    return {'y_pred': y_bicubic}




def visualize_tensor(tensor, title=None):
    img = tensor.cpu()
    img = img.squeeze().numpy() 
    img = img[0,:,:,:]  #chose first image of the batch
    img = img.transpose(1, 2, 0) 

    # Visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()
    
    input()


def save_tensor_as_image(tensor, title=None):
    img = tensor.cpu()
    img = img.squeeze().numpy()
    img = img[0, :, :, :]  # Choose the first image of the batch
    img = img.transpose(1, 2, 0)

    # Save the image
    img = (img * 255).astype(np.uint8)  # Scale to uint8 range
    image = Image.fromarray(img)
    image.save(f'/scratch2/merler/code/data/pan10/output_images/{title}.png')
    