import torch
import torch.nn as nn
import tensorflow.compat.v1 as tf

import losses

tf.enable_eager_execution()

import numpy as np

w_init = 0.001

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant(m.weight.data, w_init)
        nn.init.constant(m.bias.data, 0.0)


class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leaky(self.conv1(x))

class ImageEncoder(nn.Module):
    def __init__(self, input_shape,
                 initial_num_filters=32,
                 output_map_width=16,
                 layers_per_scale=1, debug=False,
                 **conv_layer_kwargs):

        super(ImageEncoder, self).__init__()

        if np.log2(input_shape[1] / output_map_width) % 1:
            raise ValueError(
                'The ratio of input width and output_map_width must be a perfect '
                'square, but got {} and {} with ratio {}'.format(
                    input_shape[1], output_map_width, input_shape[1] / output_map_width))

        C, H, W = input_shape

        layers = []
        layers.extend([nn.Conv2d(C, initial_num_filters, kernel_size=3, padding=1), nn.LeakyReLU(0.2)])
        #layers.extend([nn.Conv2d(C, initial_num_filters, kernel_size=3, padding=0)])
        for _ in range(layers_per_scale):
            layers.extend([nn.Conv2d(initial_num_filters, initial_num_filters, kernel_size=3, padding=1),
                          nn.LeakyReLU(0.2)])
            #layers.extend([nn.Conv2d(initial_num_filters, initial_num_filters, padding=0, kernel_size=3)])

        width = W
        num_filters = initial_num_filters
        while width > output_map_width:
            # Reduce resolution:
            layers.extend([nn.Conv2d(num_filters, 2*num_filters, stride=2, padding=1, kernel_size=3),
                          nn.LeakyReLU(0.2)])
            #layers.extend([nn.Conv2d(num_filters, 2*num_filters, stride=2, padding=0, kernel_size=3)])

            num_filters *= 2
            width //= 2

            # Apply additional layers:
            for _ in range(layers_per_scale):
                layers.extend([nn.Conv2d(num_filters, num_filters, padding=1, kernel_size=3),
                               nn.LeakyReLU(0.2)])
                # layers.extend([nn.Conv2d(num_filters, num_filters, stride=1, padding=0, kernel_size=3)])

        self.encoder = nn.Sequential(*layers)

        self.c_out = num_filters
        self.debug  = debug

    def forward(self, x):
        if self.debug: print("Encoder Input shape: ", x.shape)

        ys = [x]
        y = x
        for i, s in enumerate(self.encoder):
            y = s(y)
            if isinstance(s, nn.Conv2d): print(i, s, y.shape)
            if isinstance(s, nn.LeakyReLU): ys.append(y)

        x = self.encoder(x)
        if self.debug: print("Encoded Image shape: ", x.shape)
        return x, ys

def build_image_encoder(
    input_shape, initial_num_filters=32, output_map_width=16,
    layers_per_scale=1, **conv_layer_kwargs):
    inputs = tf.keras.Input(shape=input_shape, name='encoder_input')

    if np.log2(input_shape[0] / output_map_width) % 1:
        raise ValueError(
            'The ratio of input width and output_map_width must be a perfect '
            'square, but got {} and {} with ratio {}'.format(
                input_shape[0], output_map_width, inputs[0] / output_map_width))
    ops = [inputs]
    # Expand image to initial_num_filters maps:
    x = tf.keras.layers.Conv2D(initial_num_filters, **conv_layer_kwargs)(inputs)
    ops.append(x)
    for _ in range(layers_per_scale):
        x = tf.keras.layers.Conv2D(initial_num_filters, **conv_layer_kwargs)(x)
        ops.append(x)

    # Apply downsampling blocks until feature map width is output_map_width:
    width = int(inputs.get_shape()[1])
    num_filters = initial_num_filters
    i = 0

    while width > output_map_width:
        num_filters *= 2
        width //= 2

        # Reduce resolution:
        x = tf.keras.layers.Conv2D(num_filters, strides=2, **conv_layer_kwargs)(x)

        ops.append(x)
        # Apply additional layers:
        for _ in range(layers_per_scale):
            x = tf.keras.layers.Conv2D(num_filters, strides=1, **conv_layer_kwargs)(x)
            i += 1
            ops.append(x)

    return tf.keras.Model(inputs=inputs, outputs=[x, ops], name='image_encoder')


x = np.load("../tf.npy").astype(np.float32)

xtor = torch.from_numpy(x).permute(0, 3, 1, 2)
xtf = tf.convert_to_tensor(x)


m1 = ImageEncoder((5,64,64), 32, 16, 2, **{'kernel_size': 3, 'padding':1})
m1.apply(weights_init)

m1conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            nn.Softplus())
m1conv.apply(weights_init)

m2 = build_image_encoder((64,64,5), 32, 16, 2, **{'kernel_size': 3,
                                                  'padding':[[0,0],[1,1],[1,1],[0,0]],
                                                  'activation':tf.nn.leaky_relu,
                                                  'kernel_initializer': tf.keras.initializers.Constant(w_init)
                                                  })

m2conv = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=1,
    padding='valid',
    activation=tf.nn.softplus,
    kernel_initializer=tf.keras.initializers.Constant(w_init))

a, axs = m1(xtor)
axs.append(m1conv(axs[-1]))
ax_pen=  losses.get_heatmap_penalty(axs[-1], 1)

tr = lambda x: x.permute(0,2,3,1).detach().numpy()
axs = [tr(ax) for ax in axs]
a = tr(a)

b, bxs = m2(xtf)
bxs.append(m2conv(bxs[-1]))
bx_pen=  losses.get_heatmap_penalty_tf(bxs[-1], 1)
bxs = [bx.numpy() for bx in bxs]
b = b.numpy()

for i,(ax,bx) in enumerate(zip(axs, bxs)):
    print('check', i, np.allclose(ax,bx, ), np.mean(np.abs(ax-bx)), ax.shape)

print(ax_pen, bx_pen)