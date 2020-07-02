"""
From https://github.com/denisyarats/pytorch_sac_ae
"""
import imageio
import os
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np

class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, crop=(80,350)):
        frame = env.render(mode='rgb_array')
        if crop: frame = frame[crop[0]:crop[1], crop[0]:crop[1]]
        frame = img_as_ubyte(resize(frame, (self.height, self.width)))
        self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)