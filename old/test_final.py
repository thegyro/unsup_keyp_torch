import torch
import torch.nn as nn
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from train_pl import KeypointModel
from models_tf import build_model_keyp

import hyperparameters, hyperparameters_tf

tf.enable_eager_execution()

import numpy as np

w_init = 0.00001

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant(m.weight.data, w_init)
        nn.init.constant(m.bias.data, 0.0)

from register_args import get_argparse
args = get_argparse(False).parse_args()

cfg_tor = hyperparameters.get_config(args)
cfg_tf  = hyperparameters_tf.get_config(args)

x = np.random.randn((2, 8, 64, 64, 3)).astype(np.float32)

xtor = torch.from_numpy(x).permute(0, 3, 1, 2)
xtf = tf.convert_to_tensor(x)

model_tor = KeypointModel(cfg_tor)
model_tor.apply(weights_init)

model_tf = build_model_keyp(cfg_tf, {'image': (None, 8, 64, 64, 3)})

