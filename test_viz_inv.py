import glob
import os
from itertools import islice

import torch
from pytorch_lightning import Trainer

import datasets
import hyperparameters
import train, train_nofirst, train_keyp_pred
import train_keyp_inverse
import train_keyp_inverse_forward
import utils
from utils import img_torch_to_numpy, get_latest_checkpoint
from visualizer import viz_all, viz_track, viz_all_unroll, viz_keyp_action_pred, viz_keypoints, viz_keyp_hmap
import numpy as np

import torch.nn.functional as F

def viz_seq(args):
    utils.set_seed_everywhere(args.seed)
    cfg = hyperparameters.get_config(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    l_dir = cfg.train_dir if args.is_train else args.test_dir
    print("Data loader: ", l_dir)
    loader, data_shapes = datasets.get_sequence_dataset(
        data_dir=os.path.join(cfg.data_dir, l_dir),
        batch_size=10,
        num_timesteps=args.timesteps, shuffle=True)

    cfg.data_shapes = data_shapes

    model = train_keyp_inverse_forward.KeypointModel(cfg).to(device)

    if args.pretrained_path:
        if args.ckpt:
            checkpoint_path = os.path.join(args.pretrained_path, "_ckpt_epoch_" + args.ckpt + ".ckpt")
        else:
            print("Loading latest")
            checkpoint_path = get_latest_checkpoint(args.pretrained_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        print("Loading model from: ", checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("Load complete")

    with torch.no_grad():
        for data in islice(loader, 1):
            img_seq = data['image'].to(device)
            action_seq = data['action'].to(device)

            keypoints_seq, heatmaps_seq, pred_keyp_seq, pred_action_seq = model(img_seq, action_seq)
            print("Keypoint Pred LOSS:",
                  F.mse_loss(pred_keyp_seq[Ellipsis, :2], keypoints_seq[:, 1:, :, :2], reduction='sum')
                  / ((pred_keyp_seq.shape[0]) * pred_keyp_seq.shape[1]))
            if args.unroll:
                pred_keyp_seq = model.unroll(img_seq, action_seq)

            pred_keyp_seq_np = pred_keyp_seq.cpu().numpy()

            print(img_seq.shape, keypoints_seq.shape)

            img_seq_np = img_torch_to_numpy(img_seq)
            heatmaps_seq_np = heatmaps_seq.permute(0, 1, 3, 4, 2).cpu().numpy()
            keypoints_seq_np = keypoints_seq.cpu().numpy()

            d = {'img': img_seq_np,
                 'keyp':keypoints_seq_np,
                 'heatmap': heatmaps_seq.permute(0,1,3,4,2).cpu().numpy(),
                 'action': data['action'].cpu().numpy() if 'action' in data else None}

            tmp_save_path = 'tmp_data/{}_data_{}_seed_{}'.format(l_dir, args.vids_path, args.seed)
            print("Save intermediate data path: ", tmp_save_path)
            np.savez(tmp_save_path, **d)

            num_seq = img_seq_np.shape[0]
            for i in islice(range(num_seq),3):
                save_path = os.path.join(args.vids_dir, args.vids_path + "_" + l_dir + "_{}_seed_{}.mp4"
                                         .format(i, args.seed))
                print(i, "Video Save Path", save_path)
                viz_keypoints(img_seq_np[i], keypoints_seq_np[i], True, 100, save_path, args.annotate)
                #viz_keyp_hmap(img_seq_np[i], keypoints_seq_np[i], heatmaps_seq_np[i], 2, True, 100, save_path)

def run_final_test(args):
    utils.set_seed_everywhere(args.seed)
    cfg = hyperparameters.get_config(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    l_dir = cfg.train_dir if args.is_train else args.test_dir
    print("Data loader: ", l_dir)
    loader, data_shapes = datasets.get_sequence_dataset(
        data_dir=os.path.join(cfg.data_dir, l_dir),
        batch_size=cfg.batch_size,
        num_timesteps=2*args.timesteps, shuffle=True)

    cfg.log_training = args.log_training
    cfg.log_training_path = os.path.join(args.log_training_path)

    cfg.data_shapes = data_shapes
    if args.no_first:
        if args.keyp_pred:
            print("Loding keyp pred")
            model = train_keyp_pred.KeypointModel(cfg).to(device)
        elif args.keyp_inverse:
            print("Loding Inverse Model")
            model = train_keyp_inverse.KeypointModel(cfg).to(device)
        else:
            pass
    else:
        model = train.KeypointModel(cfg).to(device)

    if args.pretrained_path:
        if args.ckpt:
            checkpoint_path = os.path.join(args.pretrained_path, "_ckpt_epoch_" + args.ckpt + ".ckpt")
        else:
            print("Loading latest")
            checkpoint_path = get_latest_checkpoint(args.pretrained_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        print("Loading model from: ", checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("Load complete")

    trainer = Trainer(gpus=1,
                      progress_bar_refresh_rate=1,
                      show_progress_bar=True)

    trainer.test(model)

if __name__ == "__main__":
    from register_args import get_argparse

    args = get_argparse(False).parse_args()

    viz_seq(args)

    #run_final_test(args)