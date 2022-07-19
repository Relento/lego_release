"""This module contains simple helper functions """
from __future__ import print_function

import colorsys
import io
import os
import random

import numpy as np
import skimage.transform
import torch
from PIL import Image


def resize_masks(masks, image_size):
    """
    Resize masks size
    :param masks: numpy of shape (n, 1, h, w)
    :param image_size: H, W
    :return: numpy array of shape (n, H, W)
    """
    masks_n = masks.squeeze()
    masks_resize = np.zeros((masks_n.shape[0], image_size[0], image_size[1]))
    for i in range(masks_n.shape[0]):
        masks_resize[i] = skimage.transform.resize(masks_n[i], image_size, order=3)
        masks_resize[i] = (masks_resize[i] >= 0.75).astype('uint8')
    return masks_resize


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + \
                                  alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_image(image, masks):
    image_mask = image
    colors = random_colors(masks.shape[0])
    for i in range(masks.shape[0]):
        image_mask = apply_mask(image, masks[i], colors[i])
    return image_mask


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def transform_points_screen(
        transform, points, image_size
) -> torch.Tensor:
    """
    Transform input points from world to screen space.
    Args:
        transform: ndc transform
        points: torch tensor of shape (N, V, 3).
        image_size: torch tensor of shape (N, 2)
    Returns
        new_points: transformed points with the same shape as the input.
    """

    ndc_points = transform.transform_points(points)

    if not torch.is_tensor(image_size):
        image_size = torch.tensor(
            image_size, dtype=torch.int64, device=points.device
        )
    if (image_size < 1).any():
        raise ValueError("Provided image size is invalid.")

    image_width, image_height = image_size.unbind(1)
    image_width = image_width.view(-1, 1)  # (N, 1)
    image_height = image_height.view(-1, 1)  # (N, 1)

    ndc_z = ndc_points[..., 2]
    screen_x = (image_width - 1.0) / 2.0 * (1.0 - ndc_points[..., 0])
    screen_y = (image_height - 1.0) / 2.0 * (1.0 - ndc_points[..., 1])

    return torch.stack((screen_x, screen_y, ndc_z), dim=2)


def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)
