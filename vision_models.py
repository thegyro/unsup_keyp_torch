from __future__ import print_function

import numpy as np
from torch import nn

import ops

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
        for _ in range(layers_per_scale):
            layers.extend([nn.Conv2d(initial_num_filters, initial_num_filters, kernel_size=3, padding=1),
                          nn.LeakyReLU(0.2)])

        width = W
        num_filters = initial_num_filters
        while width > output_map_width:
            # Reduce resolution:
            layers.extend([nn.Conv2d(num_filters, 2*num_filters, stride=2, kernel_size=3, padding=1),
                          nn.LeakyReLU(0.2)])

            num_filters *= 2
            width //= 2

            # Apply additional layers:
            for _ in range(layers_per_scale):
                layers.extend([nn.Conv2d(num_filters, num_filters, stride=1, kernel_size=3, padding=1),
                               nn.LeakyReLU(0.2)])

        self.encoder = nn.Sequential(*layers)

        self.c_out = num_filters
        self.debug  = debug

    def forward(self, x):
        if self.debug: print("Encoder Input shape: ", x.shape)

        x = self.encoder(x)
        if self.debug: print("Encoded Image shape: ", x.shape)

        return x

class ImageDecoder(nn.Module):
    def __init__(self, input_shape, output_width,
                 layers_per_scale=1, debug=False,
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
                    output_width, input_shape[0], output_width/input_shape[0]))

        # Expand until we have filters_out channels:
        layers = []
        for i in range(num_levels):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            c_in = num_filters
            for _ in range(layers_per_scale):
                layers.extend([nn.Conv2d(c_in, num_filters//2, kernel_size=3, padding=1), nn.LeakyReLU(0.2)])
                c_in = c_in//2

            num_filters = num_filters//2

        self.decoder = nn.Sequential(*layers)
        self.debug = debug

    def forward(self, x):
        if self.debug: print("Decoder Input shape: ", x.shape)
        x = self.decoder(x)
        if self.debug: print("Decoded shape: ", x.shape)
        return x

class KeypointsToHeatmaps(nn.Module):
    def __init__(self, sigma, heatmap_width):
        super(KeypointsToHeatmaps, self).__init__()

        self.sigma = sigma
        self.heatmap_width = heatmap_width

    def forward(self, keypoints):
        return ops.keypoints_to_maps(keypoints, self.sigma, self.heatmap_width)