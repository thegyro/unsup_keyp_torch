import functools

import torch
import torch.nn as nn
import tensorflow.compat.v1 as tf

import ops_tf, ops

tf.enable_eager_execution()

import numpy as np

w_init = 0.001


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant(m.weight.data, w_init)
        nn.init.constant(m.bias.data, 0.0)


class ImageDecoder(nn.Module):
    def __init__(self, input_shape, output_width, layers_per_scale=1, debug=False,
                 **conv_layer_kwargs):
        """
        :param input_shape: (C, H, W)
        :param output_width:
        :param layers_per_scale:
        :param conv_layer_kwargs:
        """

        super(ImageDecoder, self).__init__()

        num_levels = int(np.log2(output_width / input_shape[1]))
        num_filters, H, W = input_shape

        if num_levels % 1:
            raise ValueError('The ratio of output_width and input width must be a perfect '
                             'square, but got {} and {} with ratio {}'.format(
                output_width, input_shape[0], output_width / input_shape[0]))

        # Expand until we have filters_out channels:
        layers = []
        for i in range(num_levels):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            c_in = num_filters
            for _ in range(layers_per_scale):
                layers.extend([nn.Conv2d(c_in, num_filters // 2, kernel_size=3, padding=1), nn.LeakyReLU(0.2)])
                c_in = c_in // 2

            num_filters = num_filters // 2

        self.decoder = nn.Sequential(*layers)
        self.debug = debug

    def forward(self, x):
        if self.debug: print("Decoder Input shape: ", x.shape)

        ys = [x]
        y = x
        for s in self.decoder:
            y = s(y)
            if isinstance(s, nn.Conv2d) or isinstance(s, nn.Upsample): print(y.shape)
            if isinstance(s, nn.LeakyReLU) or isinstance(s, nn.Upsample):
                ys.append(y)

        x = self.decoder(x)
        if self.debug: print("Decoded shape: ", x.shape)
        return x, ys


class KeypointsToHeatmaps(nn.Module):
    def __init__(self, sigma, heatmap_width):
        super(KeypointsToHeatmaps, self).__init__()

        self.sigma = sigma
        self.heatmap_width = heatmap_width

    def forward(self, keypoints):
        return ops.keypoints_to_maps(keypoints, self.sigma, self.heatmap_width)


def build_image_decoder(
    input_shape, output_width, layers_per_scale=1, **conv_layer_kwargs):

    feature_maps = tf.keras.Input(shape=input_shape, name='feature_maps')
    num_levels = np.log2(output_width / input_shape[0])

    if num_levels % 1:
        raise ValueError(
            'The ratio of output_width and input width must be a perfect '
            'square, but got {} and {} with ratio {}'.format(
                output_width, input_shape[0], output_width / input_shape[0]))

    # Expand until we have filters_out channels:
    x = feature_maps
    ops = [x]
    num_filters = input_shape[-1]

    def upsample(x):
        new_size = [x.get_shape()[1] * 2, x.get_shape()[2] * 2]
        return tf.image.resize_bilinear(x, new_size, align_corners=True)

    for _ in range(int(num_levels)):
        num_filters //= 2
        x = tf.keras.layers.Lambda(upsample)(x)
        ops.append(x)

        # Apply additional layers:
        for _ in range(layers_per_scale):
            x = tf.keras.layers.Conv2D(num_filters, **conv_layer_kwargs)(x)
            ops.append(x)

    return tf.keras.Model(inputs=feature_maps, outputs=[x,ops], name='image_decoder')


#x = np.load("torch_test_4.npy").astype(np.float32)
x = np.load("../gmaps.npy").astype(np.float32)
#x  = 0.5*np.ones((32,16,16,128)).astype(np.float32)

xtor = torch.from_numpy(x).permute(0,3,1,2)
#xtor = torch.from_numpy(x)
xtf = tf.convert_to_tensor(x)

#m1Conv = KeypointsToHeatmaps(1.5, 16)
#m1Conv.apply(weights_init)
m1 = ImageDecoder((128, 16, 16), 64, 2, **{'kernel_size': 3, 'padding': 1})
m1.apply(weights_init)

#m1conv.apply(weights_init)

# m2Conv = tf.keras.layers.Lambda(
#     functools.partial(
#         ops_tf.keypoints_to_maps,
#         sigma=1.5,
#         heatmap_width=16))
m2 = build_image_decoder((16, 16, 128), 64, 2, **{'kernel_size': 3,
                                                    'padding': [[0, 0], [1, 1], [1, 1], [0, 0]],
                                                    'activation': tf.nn.leaky_relu,
                                                    'kernel_initializer': tf.keras.initializers.Constant(w_init)
                                                    })

#xtor_1 = m1Conv(xtor)
a, axs = m1(xtor)
#axs.append(m1conv(axs[-1]))
tr = lambda x: x.permute(0, 2, 3, 1).detach().numpy()
axs = [tr(ax) for ax in axs]
a = tr(a)

#xtf_1 = m2Conv(xtf)
b, bxs = m2(xtf)
#bxs.append(m2conv(bxs[-1]))
bxs = [bx.numpy() for bx in bxs]
b = b.numpy()

for i, (ax, bx) in enumerate(zip(axs, bxs)):
    print('check', i, np.allclose(ax, bx, ), np.mean(np.abs(ax - bx)), ax.shape)


f1s = ['torch_test_dec_{}.npy'.format(i) for i in range(6)]
f2s = ['../unsup_keyp/tf_test_dec_{}.npy'.format(i) for i in range(6)]

for i, (f1,f2) in enumerate(zip(f1s, f2s)):
    t1 = np.load(f1)
    t2 = np.load(f2)

    print(i, np.allclose(t1, t2), t1.shape , np.mean(np.abs(t1-t2)))
