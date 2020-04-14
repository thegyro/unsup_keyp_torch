from __future__ import print_function

import numpy as np
import torch
from torch import nn

import ops

from vision_models import ImageEncoder, ImageDecoder, KeypointsToHeatmaps


class ImagesToKeypEncoder(nn.Module):
    def __init__(self, cfg, image_shape, debug=False):
        """

        :param cfg:
        :param image_shape: (im_C, im_H, im_W)
        """

        super(ImagesToKeypEncoder, self).__init__()

        im_C, im_H, im_W = image_shape

        # Encoude image to features (add 2 to C for coord channels)
        self.image_encoder = ImageEncoder(
            input_shape=(im_C+2, im_H, im_W), initial_num_filters=cfg.num_encoder_filters,
            output_map_width=cfg.heatmap_width, layers_per_scale=cfg.layers_per_scale,
            debug=debug,
            **cfg.conv_layer_kwargs)

        c_in = self.image_encoder.c_out

        self.feats_heatmap = nn.Sequential(
            nn.Conv2d(c_in, cfg.num_keypoints, kernel_size=1, padding=0),
            nn.Softplus(threshold=1000))

        self.debug = debug

    def forward(self, img):
        # Image to keypoints:
        if self.debug: print('Image shape: ', img.shape)

        img  = ops.add_coord_channels(img)
        encoded = self.image_encoder(img)
        heatmaps = self.feats_heatmap(encoded)

        if self.debug: print("Heatmaps shape:", heatmaps.size())

        keypoints = ops.maps_to_keypoints(heatmaps)

        if self.debug: print("Keypoints shape:", keypoints.shape)
        if self.debug: print()

        return keypoints, heatmaps

class KeypToImagesDecoder(nn.Module):
    def __init__(self, cfg, image_shape, debug=False):
        super(KeypToImagesDecoder, self).__init__()

        im_C, im_H, im_W = image_shape

        self.keypoints_to_maps = KeypointsToHeatmaps(cfg.keypoint_width, cfg.heatmap_width)

        num_encoder_output_channels = (
            cfg.num_encoder_filters * im_W // cfg.heatmap_width)

        decoder_input_shape = (num_encoder_output_channels, cfg.heatmap_width,
                               cfg.heatmap_width)

        self.image_decoder = ImageDecoder(input_shape=decoder_input_shape,
                                          output_width=im_W, layers_per_scale=cfg.layers_per_scale,
                                          debug=debug,
                                          **cfg.conv_layer_kwargs)

        kwargs = dict(cfg.conv_layer_kwargs)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        self.adjust_channels_of_decoder_input = nn.Sequential(
            nn.Conv2d(cfg.num_keypoints+2,num_encoder_output_channels, **kwargs),
            nn.LeakyReLU(0.2))

        self.adjust_channels_of_output_image = nn.Conv2d(cfg.num_encoder_filters, im_C, **kwargs)

        self.debug = debug

    def forward(self, keypoints):
        """ keypoints: [batch_size, num_keypoints, 3] """

        if self.debug: print("Keypoints shape: ", keypoints.shape)

        gaussian_maps = self.keypoints_to_maps(keypoints)
        if self.debug: print("Gaussian Heatmap: ", gaussian_maps.shape)

        gaussian_maps = ops.add_coord_channels(gaussian_maps)
        if self.debug: print("Gaussian Heatmap with CoordConv: ", gaussian_maps.shape)

        gaussian_maps = self.adjust_channels_of_decoder_input(gaussian_maps)
        if self.debug: print("Gaussian Heatmap before decoder: ", gaussian_maps.shape)

        decoded_rep = self.image_decoder(gaussian_maps)
        if self.debug: print("Decoded Representation: ", decoded_rep.shape)

        reconstructed_img = self.adjust_channels_of_output_image(decoded_rep)
        if self.debug: print("Reconstructed Img: ", reconstructed_img.shape)
        if self.debug: print()

        return reconstructed_img

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant(m.weight.data, 0.001)
        nn.init.constant(m.bias.data, 0.0)

def run(args):
    import hyperparameters

    torch.set_printoptions(precision=10)

    cfg = hyperparameters.get_config(args)
    # cfg.layers_per_scale=1
    # cfg.num_keypoints=32
    # cfg.batch_size = 25

    imgs_to_keyp_model = ImagesToKeypEncoder(cfg, (3, 64, 64), debug=True)
    imgs_to_keyp_model.apply(weights_init)
    keyp_to_imgs_model = KeypToImagesDecoder(cfg, (3, 64, 64), debug=True)
    keyp_to_imgs_model.apply(weights_init)


    print(imgs_to_keyp_model)
    print(keyp_to_imgs_model)

    #summary(model, input_size=(2, 3, 64, 64))

    k,h = imgs_to_keyp_model(0.5*torch.ones((cfg.batch_size, 3,64,64)))


    print(k[0])
    k_tf = torch.from_numpy(np.load("../unsup_keyp/tf_test_dec_0.npy"))

    r, xs = keyp_to_imgs_model(k_tf)

    for i,y in enumerate(xs):
        if len(y.shape) == 4:
            np.save('torch_test_dec_{}.npy'.format(i), y.detach().permute(0,2,3,1).numpy())
        else:
            np.save('torch_test_dec_{}.npy'.format(i), y.detach().numpy())

    print(k.shape, h.shape, r.shape)

    b = sum([np.prod(list(params.size())) for params in imgs_to_keyp_model.parameters()])
    print("Encodeer params: ", b)
    c = sum([np.prod(list(params.size())) for params in keyp_to_imgs_model.parameters()])
    print("Decoder params: ", c)

    print("Model parameters: ", b+c)


if __name__ == "__main__":
    from register_args import get_argparse

    run(get_argparse(force_exp_name=False).parse_args())