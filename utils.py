import glob
import os

import torch
import numpy as np
import random

def img_torch_to_numpy(img):
    """img: [bs, channel, H, W,] or [bs, T, channel, H, W]"""
    if len(list(img.shape)) == 4:
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        return img
    else:
        img = img.permute(0, 1, 3, 4, 2).cpu().numpy()
        return img

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