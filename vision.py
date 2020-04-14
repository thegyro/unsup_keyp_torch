from __future__ import print_function

import numpy as np
import torch
from torch import nn

import ops

from vision_models import ImageEncoder, ImageDecoder, KeypointsToHeatmaps
from utils import unstack_time, stack_time


class ImagesToKeypEncoder(nn.Module):
    def __init__(self, cfg, image_shape, debug=False):
        """

        :param cfg:
        :param image_shape: (im_C, im_H, im_W)
        """

        super(ImagesToKeypEncoder, self).__init__()

        T, im_C, im_H, im_W = image_shape

        # Encoude image to features (add 2 to C for coord channels)
        self.image_encoder = ImageEncoder(
            input_shape=(im_C + 2, im_H, im_W), initial_num_filters=cfg.num_encoder_filters,
            output_map_width=cfg.heatmap_width, layers_per_scale=cfg.layers_per_scale,
            debug=debug,
            **cfg.conv_layer_kwargs)

        c_in = self.image_encoder.c_out

        self.feats_heatmap = nn.Sequential(
            nn.Conv2d(c_in, cfg.num_keypoints, kernel_size=1, padding=0),
            nn.Softplus())

        # TODO: Activation Regularizer Heatmap

        self.debug = debug

    def forward(self, img_seq):
        # Image to keypoints:
        if self.debug: print('Image shape: ', img_seq.shape)

        img_list = unstack_time(img_seq)

        keypoints_list = []
        heatmaps_list = []
        for img in img_list:
            img = ops.add_coord_channels(img)
            encoded = self.image_encoder(img)
            heatmaps = self.feats_heatmap(encoded)

            if self.debug: print("Heatmaps shape:", heatmaps.size())

            keypoints = ops.maps_to_keypoints(heatmaps)

            if self.debug: print("Keypoints shape:", keypoints.shape)
            if self.debug: print()

            heatmaps_list.append(heatmaps)
            keypoints_list.append(keypoints)

        keypoints_seq = stack_time(keypoints_list)
        heatmaps_seq = stack_time(heatmaps_list)

        return keypoints_seq, heatmaps_seq


class KeypToImagesDecoder(nn.Module):
    def __init__(self, cfg, image_shape, debug=False):
        super(KeypToImagesDecoder, self).__init__()

        T, im_C, im_H, im_W = image_shape

        self.keypoints_to_maps = KeypointsToHeatmaps(cfg.keypoint_width, cfg.heatmap_width)

        num_encoder_output_channels = (
            cfg.num_encoder_filters * im_W // cfg.heatmap_width)

        decoder_input_shape = (num_encoder_output_channels, cfg.heatmap_width,
                               cfg.heatmap_width)

        self.image_decoder = ImageDecoder(input_shape=decoder_input_shape,
                                          output_width=im_W, layers_per_scale=cfg.layers_per_scale,
                                          debug=debug,
                                          **cfg.conv_layer_kwargs)

        self.appearance_feature_extractor = ImageEncoder(
            input_shape=(im_C, im_H, im_W),
            initial_num_filters=cfg.num_encoder_filters,
            layers_per_scale=cfg.layers_per_scale,
            **cfg.conv_layer_kwargs)

        kwargs = dict(cfg.conv_layer_kwargs)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        self.adjust_channels_of_decoder_input = nn.Sequential(
            nn.Conv2d(2*cfg.num_keypoints +
                      num_encoder_output_channels + 2 , num_encoder_output_channels, **kwargs),
            nn.LeakyReLU(0.2))

        self.adjust_channels_of_output_image = nn.Conv2d(cfg.num_encoder_filters, im_C, **kwargs)

        self.debug = debug

    def forward(self, keypoints_seq, first_frame, first_frame_keypoints):
        """ keypoints: [batch_size, T, num_keypoints, 3]
            first_frame: [batch_size, 3, IM_H, IM_W]
            first_frame_keypoints: [batch_size, num_keypoints, 3]
        """

        if self.debug: print("Keypoints shape: ", keypoints_seq.shape)

        first_frame_features = self.appearance_feature_extractor(first_frame) # batch_size x 128 x 16 x 16
        first_frame_gaussian_maps = self.keypoints_to_maps(first_frame_keypoints) # batch_size x 64 x 16 x 16

        keypoints_list = unstack_time(keypoints_seq)

        reconstructed_img_list = []
        for keypoints in keypoints_list:
            gaussian_maps = self.keypoints_to_maps(keypoints)
            if self.debug: print("Gaussian Heatmap: ", gaussian_maps.shape)

            combined_maps = torch.cat([gaussian_maps,
                                      first_frame_features,
                                      first_frame_gaussian_maps], dim=1)

            combined_maps = ops.add_coord_channels(combined_maps)
            if self.debug: print("Gaussian Heatmap with CoordConv: ", combined_maps.shape)

            combined_maps = self.adjust_channels_of_decoder_input(combined_maps)
            if self.debug: print("Gaussian Heatmap before decoder: ", combined_maps.shape)

            decoded_rep = self.image_decoder(combined_maps)
            if self.debug: print("Decoded Representation: ", decoded_rep.shape)

            reconstructed_img = self.adjust_channels_of_output_image(decoded_rep)
            if self.debug: print("Reconstructed Img: ", reconstructed_img.shape)

            if self.debug: print()

            reconstructed_img_list.append(reconstructed_img)

        reconstructed_img_seq = stack_time(reconstructed_img_list)
        reconstructed_img_seq = reconstructed_img_seq + first_frame[:, None, :, :, :]
        return reconstructed_img_seq

def run(args):
    import hyperparameters

    cfg = hyperparameters.get_config(args)
    # cfg.layers_per_scale=1
    # cfg.num_keypoints=32
    # cfg.batch_size = 25

    imgs_to_keyp_model = ImagesToKeypEncoder(cfg, (8, 3, 64, 64), debug=True)
    keyp_to_imgs_model = KeypToImagesDecoder(cfg, (8, 3, 64, 64), debug=True)

    print(imgs_to_keyp_model)
    print(keyp_to_imgs_model)

    # summary(model, input_size=(2, 3, 64, 64))
    img = 0.5 * torch.ones((1, 8, 3, 64, 64))
    k, h = imgs_to_keyp_model(img)

    r = keyp_to_imgs_model(k, img[:, 0], k[:, 0])

    print(k.shape, h.shape, r.shape)

    b = sum([np.prod(list(params.size())) for params in imgs_to_keyp_model.parameters()])
    print("Encodeer params: ", b)
    c = sum([np.prod(list(params.size())) for params in keyp_to_imgs_model.parameters()])
    print("Decoder params: ", c)

    print("Model parameters: ", b + c)


if __name__ == "__main__":
    from register_args import get_argparse

    run(get_argparse(force_exp_name=False).parse_args())
