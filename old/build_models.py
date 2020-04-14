
from torch import nn
import vision_noseq


def build_model_noseq(cfg, data_shapes):
    print("Data Shape:", data_shapes)

    input_shape_no_batch = data_shapes['image'][1:]

    # define all the models
    images_to_keypoints_net = vision_noseq.ImagesToKeypEncoder(cfg, input_shape_no_batch)
    keypoints_to_images_net = vision_noseq.KeypToImagesDecoder(cfg, input_shape_no_batch)

    return nn.ModuleList([images_to_keypoints_net, keypoints_to_images_net])
