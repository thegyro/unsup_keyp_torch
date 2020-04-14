import glob
import os
from itertools import islice

import torch

import datasets
import hyperparameters
from build_models import build_model_noseq
from utils import img_torch_to_numpy, get_latest_checkpoint
from visualizer import viz_all




def main(args):
    cfg = hyperparameters.get_config(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    l_dir = cfg.train_dir if args.is_train else args.test_dir
    print("Data loader: ", l_dir)
    loader, data_shapes = datasets.get_dataset(
        data_dir=os.path.join(cfg.data_dir, l_dir),
        batch_size=cfg.batch_size)

    models = build_model_noseq(cfg, data_shapes).to(device)

    if args.pretrained_path:
        model_path = get_latest_checkpoint(args.pretrained_path, "*.pth")
        print("Loading model from: ", model_path)
        models.load_state_dict(torch.load(model_path))
        models.eval()

    images_to_keypoints_net, keypoints_to_images_net = models

    with torch.no_grad():
        for i, data in islice(enumerate(loader), 5):
            img = data['image'].to(device)

            keypoints, _ = images_to_keypoints_net(img)
            pred_img = keypoints_to_images_net(keypoints)

            print(img.shape, keypoints.shape, pred_img.shape)

            save_path = os.path.join(args.vids_dir, args.vids_path + "_" + l_dir + "_{}.mp4".format(i))
            print(i, "Video Save Path", save_path)

            imgs_np, pred_img_np = img_torch_to_numpy(img), img_torch_to_numpy(pred_img)
            keypoints_np = keypoints.cpu().numpy()

            viz_all(imgs_np, pred_img_np, keypoints_np, True, 300, save_path)

if __name__ == "__main__":
    from register_args import get_argparse

    args = get_argparse(False).parse_args()

    main(args)