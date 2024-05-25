import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2


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

def pansharpen_hsv(sample):
    pan_hr = sample['guide'].cpu().numpy()
    rgb_upsampled = sample['y_bicubic'].cpu().numpy()
    batch_size = rgb_upsampled.shape[0]
    transformed_batch = []

    for batch_idx in range(batch_size):
        rgb_img = rgb_upsampled[batch_idx]
        pan_img = pan_hr[batch_idx]
        rgb_img = rgb_img.transpose(1, 2, 0)
        pan_img = pan_img.transpose(1, 2, 0)
        hsi = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        hsi[:, :, 2] = pan_img[:,:,0]
        target = cv2.cvtColor(hsi, cv2.COLOR_HSV2RGB).transpose(2, 0, 1)

        transformed_batch.append(target)

    transformed_batch = np.stack(transformed_batch, axis=0)
    target = torch.from_numpy(transformed_batch).to(sample['guide'].device)
    return {'y_pred': target}




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
    