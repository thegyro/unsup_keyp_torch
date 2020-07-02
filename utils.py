import glob
import os

import torch
import numpy as np
import random

from skimage.transform import  resize
from skimage import img_as_ubyte

from copy import deepcopy

def img_torch_to_numpy(img):
    """img: [bs, channel, H, W,] or [bs, T, channel, H, W]"""
    if len(list(img.shape)) == 4:
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        return img
    elif len(list(img.shape)) == 5:
        img = img.permute(0, 1, 3, 4, 2).cpu().numpy()
        return img
    elif len(list(img.shape)) == 3:
        assert img.shape[0] == 3
        return img.permute(1,2,0).cpu().numpy()

def stack_time(x):
    return torch.stack(x, dim=1)

def unstack_time(x):
    return torch.unbind(x, dim=1)


def get_latest_checkpoint(dir_path, ext="*.ckpt"):
    list_of_files = glob.glob(os.path.join(dir_path, ext))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def unnormalize_image(img):
    img = np.clip(img, -0.5, 0.5)
    return ((img + 0.5) * 255).astype(np.uint8)

def project_keyp(keyp):
    x, y, mu = keyp[:, 0], keyp[:, 1], keyp[:, 2]
    #x, y = x[mu >= 0.5], y[mu >= 0.5]
    x, y = 8 * x, 8 * y
    x, y = x + 8, 8 - y
    x, y = (64 / 16) * x, (64 / 16) * y

    N = x.shape[0]

    #return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)), mu.reshape(N,1)))
    return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)))), mu

def unproject_keyp(keyp):
    x, y = keyp[:,0], keyp[:, 1]
    x, y = x * (16/64), y * (16/64)
    x, y = x - 8, y - 8
    x, y = x/8, y/8

    return x,y

def get_frame(env, crop=(50,350), size=(64,64)):
    frame = env.render(mode='rgb_array')
    if crop: frame = frame[crop[0]:crop[1], crop[0]:crop[1]]
    if size: frame = img_as_ubyte(resize(frame, size))
    return frame