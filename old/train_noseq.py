from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
from datetime import datetime

from pytz import timezone
from tqdm import tqdm

import datasets
import hyperparameters
from build_models import build_model_noseq
from losses import get_heatmap_penalty

from register_args import save_config

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch import optim
import torch

import numpy as np

LOG_INTERVAL = None
SAVE_PATH = None

def train_epoch(epoch, model_dict, cfg):
    models = model_dict['models']
    images_to_keypoints_net, keypoints_to_images_net = models
    optimizer = model_dict['optimizer']
    device = model_dict['device']
    writer = model_dict['writer']
    train_loader = model_dict['train_loader']

    images_to_keypoints_net.train()
    keypoints_to_images_net.train()

    train_loss = 0.0
    steps = 0
    recon_loss, heatmap_loss = 0.0, 0.0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        img = data['image'].to(device)
        keypoints, heatmaps = images_to_keypoints_net(img)
        reconstructed_img = keypoints_to_images_net(keypoints)

        reconstruction_loss = F.mse_loss(img, reconstructed_img, reduction='sum')
        reconstruction_loss /= img.shape[0]

        heatmap_loss = get_heatmap_penalty(heatmaps, cfg.heatmap_regularization)

        #print(heatmaps[0][0])

        #loss = reconstruction_loss + heatmap_loss
        loss = reconstruction_loss

        loss.backward()

        train_loss += loss.item()
        recon_loss += reconstruction_loss.item()
        heatmap_loss += heatmap_loss.item()
        #orch.nn.utils.clip_grad_norm_(models.parameters(),cfg.clipnorm)

        optimizer.step()

        # if batch_idx % LOG_INTERVAL == 0:
        #     print('Train Epoch: {} [{}]\t Recon Loss: {:.6f}'.format(
        #         epoch, batch_idx, loss.item()))

        steps += 1
        break

    writer.add_scalar("train_loss", train_loss/steps, epoch)
    writer.add_scalar("train_recon_loss", recon_loss/steps, epoch)
    writer.add_scalar("train_hetmap_loss", heatmap_loss/steps, epoch)

    print('\n====> Epoch: {} Average loss: {:.4f} heatmap_loss: {}'.format(epoch, train_loss / steps, heatmap_loss/steps))

    path = SAVE_PATH + str(epoch) + ".pth"
    torch.save(models.state_dict(), path)

def test_epoch(epoch, model_dict, cfg):
    images_to_keypoints_net, keypoints_to_images_net = model_dict['models']
    device = model_dict['device']
    writer = model_dict['writer']
    test_loader = model_dict['test_loader']

    images_to_keypoints_net.eval(), keypoints_to_images_net.eval()
    test_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['image'].to(device)
            keypoints, heatmaps = images_to_keypoints_net(img)
            reconstructed_img = keypoints_to_images_net(keypoints)

            reconstruction_loss = F.mse_loss(img, reconstructed_img, reduction='sum')
            reconstruction_loss /= img.shape[0]

            test_loss += reconstruction_loss.item()

            steps += 1
            break

    print('====> Epoch: {} Test Average loss: {:.4f}'.format(epoch, test_loss / steps))
    writer.add_scalar("test_recon_loss", test_loss / steps, epoch)

def main(args):
    global LOG_INTERVAL, SAVE_PATH

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = hyperparameters.get_config(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Using device: ", device)

    time_str = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = os.path.join(cfg.base_dir, time_str)
    checkpoint_dir = os.path.join(exp_dir, cfg.checkpoint_dir)
    log_dir = os.path.join(exp_dir, cfg.log_dir)

    print("Log path: ", log_dir, "Checkpoint Dir: ", checkpoint_dir)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

    LOG_INTERVAL = args.log_interval
    SAVE_PATH = os.path.join(checkpoint_dir, "model-")

    save_config(cfg, exp_dir, "config.json")

    train_loader, data_shapes = datasets.get_dataset(
        data_dir=os.path.join(cfg.data_dir, cfg.train_dir),
        batch_size=cfg.batch_size, shuffle=False)

    test_loader, _ = datasets.get_dataset(
        data_dir=os.path.join(cfg.data_dir, cfg.test_dir),
        batch_size=cfg.batch_size, shuffle=False)

    models = build_model_noseq(cfg, data_shapes).to(device)
    optimizer = optim.Adam(models.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    model_dict = {'models': models, 'optimizer': optimizer,
                  'train_loader': train_loader, 'test_loader': test_loader,
                  'writer': SummaryWriter(log_dir), 'device': device}

    for i in tqdm(range(args.num_epochs)):
        train_epoch(i, model_dict, cfg)
        #test_epoch(i, model_dict, cfg)

if __name__ == '__main__':
    from register_args import get_argparse

    args = get_argparse(False).parse_args()

    main(args)
