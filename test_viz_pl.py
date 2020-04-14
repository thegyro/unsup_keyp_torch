import glob
import os
from itertools import islice

import torch

import datasets
import hyperparameters
import train_noseq_pl, train_pl, train
from utils import img_torch_to_numpy, get_latest_checkpoint
from visualizer import viz_all

import torch.nn.functional as F


def viz_noseq(args):
    cfg = hyperparameters.get_config(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    l_dir = cfg.train_dir if args.is_train else args.test_dir
    print("Data loader: ", l_dir)
    loader, data_shapes = datasets.get_dataset(
        data_dir=os.path.join(cfg.data_dir, l_dir),
        batch_size=cfg.batch_size, shuffle=False)

    cfg.data_shapes = data_shapes
    model = train_noseq_pl.KeypointModel(cfg).to(device)

    if args.pretrained_path:
        checkpoint_path = get_latest_checkpoint(args.pretrained_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        print("Loading model from: ", checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

    with torch.no_grad():
        for i, data in islice(enumerate(loader), 3):
            img = data['image'].to(device)

            keypoints, heatmaps, pred_img = model(img)

            print(F.mse_loss(img, pred_img, reduction='sum')/img.shape[0])

            print(img.shape, keypoints.shape, pred_img.shape)

            save_path = os.path.join(args.vids_dir, args.vids_path + "_" + l_dir + "_{}.mp4".format(i))
            print(i, "Video Save Path", save_path)

            imgs_np, pred_img_np = img_torch_to_numpy(img), img_torch_to_numpy(pred_img)
            keypoints_np = keypoints.cpu().numpy()

            viz_all(imgs_np, pred_img_np, keypoints_np, True, 300, save_path)

def viz_seq(args):
    cfg = hyperparameters.get_config(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    l_dir = cfg.train_dir if args.is_train else args.test_dir
    print("Data loader: ", l_dir)
    loader, data_shapes = datasets.get_sequence_dataset(
        data_dir=os.path.join(cfg.data_dir, l_dir),
        batch_size=cfg.batch_size,
        num_timesteps=16, shuffle=False)

    cfg.data_shapes = data_shapes
    model = train.KeypointModel(cfg).to(device)
    #model = train_pl.KeypointModel(cfg).to(device)

    if args.pretrained_path:
        checkpoint_path = get_latest_checkpoint(args.pretrained_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        print("Loading model from: ", checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

    with torch.no_grad():
        for data in islice(loader, 1):
            img_seq = data['image'].to(device)

            keypoints_seq, heatmaps_seq, pred_img_seq = model(img_seq)
            #keypoints_seq, heatmaps_seq, pred_img_seq,pred_keypoints_seq, _= model(img_seq)

            print("LOSS:", F.mse_loss(img_seq, pred_img_seq, reduction='sum')/((img_seq.shape[0]) * img_seq.shape[1]))

            print(img_seq.shape, keypoints_seq.shape, pred_img_seq.shape)

            imgs_seq_np, pred_img_seq_np = img_torch_to_numpy(img_seq), img_torch_to_numpy(pred_img_seq)
            keypoints_seq_np = keypoints_seq.cpu().numpy()

            num_seq = imgs_seq_np.shape[0]
            for i in islice(range(num_seq),3):
                save_path = os.path.join(args.vids_dir, args.vids_path + "_" + l_dir + "_{}.mp4".format(i))
                print(i, "Video Save Path", save_path)
                viz_all(imgs_seq_np[i], pred_img_seq_np[i], keypoints_seq_np[i], True, 100, save_path)

if __name__ == "__main__":
    from register_args import get_argparse

    args = get_argparse(False).parse_args()

    viz_seq(args)