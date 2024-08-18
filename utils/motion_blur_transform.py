import torch
import numpy as np
from PIL import Image
import math
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import scipy.ndimage as ndimage

def motion_blur_transform_3d(img, pose1=None, pose2=None, kernel_size=15):
    if pose1 is None or pose2 is None:
        raise ValueError("Both pose1 and pose2 must be provided.")

    translation_vector = np.array(pose2[:3]) - np.array(pose1[:3])
    
    q1 = R.from_quat(qexp_1d(pose1[3:]))
    q2 = R.from_quat(qexp_1d(pose2[3:]))
    relative_rotation = q2.inv() * q1

    direction_vector = np.concatenate((translation_vector[:2], relative_rotation.apply([1, 0, 0])[:2]))
    direction_vector /= np.linalg.norm(direction_vector)

    motion_blur_kernel = np.zeros((kernel_size, kernel_size))
    motion_blur_kernel[np.arange(kernel_size), np.arange(kernel_size)] = 1.0
    motion_blur_kernel /= kernel_size

    image_array = np.array(img).astype(np.float32) / 255.0

    blur_direction = np.arctan2(direction_vector[1], direction_vector[0])
    blur_direction = blur_direction * (180.0 / math.pi)

    rotated_kernel = ndimage.rotate(motion_blur_kernel, blur_direction, reshape=False, mode='nearest')

    # Center the kernel within the image dimensions
    pad_size = (kernel_size - 1) // 2
    blurred_image_array = np.zeros_like(image_array)
    for channel in range(image_array.shape[2]):
        blurred_image_array[:, :, channel] = ndimage.convolve(image_array[:, :, channel], rotated_kernel, mode='constant', cval=0.0)

    blurred_image_array = (blurred_image_array * 255).astype(np.uint8)
    blurred_image = Image.fromarray(blurred_image_array)

    return blurred_image




def qexp_1d(q):
    """
    Applies the exponential map to q
    :param q: (N, 3)
    :return: (N, 4)
    """
    n = np.linalg.norm(q)
    q_expanded = np.array([np.cos(n), np.sinc(n / np.pi) * q[0], np.sinc(n / np.pi) * q[1], np.sinc(n / np.pi) * q[2]])
    return q_expanded