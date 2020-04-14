import os
from datetime import datetime
from pytz import timezone


import torch.nn.functional as F
from torch import optim

import datasets
import hyperparameters
from losses import get_heatmap_penalty
import torch

import numpy as np

from utils import get_latest_checkpoint
from vision_noseq import ImagesToKeypEncoder, KeypToImagesDecoder

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class KeypointModel(pl.LightningModule):

    def __init__(self, hparams):
        super(KeypointModel, self).__init__()

        cfg = hparams
        input_shape_no_batch = cfg.data_shapes['image'][1:]
        # define all the models
        self.images_to_keypoints_net = ImagesToKeypEncoder(cfg, input_shape_no_batch)
        self.keypoints_to_images_net = KeypToImagesDecoder(cfg, input_shape_no_batch)

        torch.nn.init.xavier_uniform_(self.images_to_keypoints_net.feats_heatmap[0].weight)

        self.cfg = cfg
        self.hparams = cfg

    def forward(self, img):
        keypoints, heatmaps = self.images_to_keypoints_net(img)
        reconstructed_img =   self.keypoints_to_images_net(keypoints)

        return keypoints, heatmaps, reconstructed_img

    def training_step(self, batch, batch_idx):
        data = batch
        img = data['image']

        keypoints, heatmaps, reconstructed_img = self.forward(img)

        reconstruction_loss = F.mse_loss(img, reconstructed_img, reduction='sum')/2.0
        reconstruction_loss /= img.shape[0]

        heatmap_loss = get_heatmap_penalty(heatmaps, self.cfg.heatmap_regularization)

        loss = reconstruction_loss + heatmap_loss

        log_dict = {'recon_loss': reconstruction_loss, 'loss': loss, 'heatmap_loss': heatmap_loss}

        output= {
            'loss': loss,
            'recon_loss': reconstruction_loss,
            'heatmap_loss': heatmap_loss/self.cfg.heatmap_regularization,
            'log': log_dict
        }

        #print(reconstruction_loss.item(), heatmap_loss.item(), '\n')

        return output

    def validation_step(self, batch, batch_idx):
        data = batch
        img = data['image']

        keypoints, heatmaps, reconstructed_img = self.forward(img)

        reconstruction_loss = F.mse_loss(img, reconstructed_img, reduction='sum')/2.0
        reconstruction_loss /= img.shape[0]

        heatmap_loss = get_heatmap_penalty(heatmaps, self.cfg.heatmap_regularization)

        loss = reconstruction_loss + heatmap_loss
        #loss = reconstruction_loss

        tqdm_dict = {'val_loss': loss, 'val_recon_loss': reconstruction_loss, 'val_hmap_loss:': heatmap_loss}

        return {'val_loss': loss,
                'val_recon_loss': reconstruction_loss,
                'val_hmap_loss': heatmap_loss/self.cfg.heatmap_regularization,
                'progress_bar': tqdm_dict}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        avg_hmap_loss = torch.stack([x['val_hmap_loss'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_recon_loss': avg_recon_loss, 'test_hmap_loss': avg_hmap_loss,
                'step': self.current_epoch}
        print()
        return {'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.cfg.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_loader, _ = datasets.get_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.train_dir),
            batch_size=self.cfg.batch_size, shuffle=True)

        return train_loader

    def val_dataloader(self):
        val_loader, _ = datasets.get_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.test_dir),
            batch_size=self.cfg.batch_size)

        return val_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = hyperparameters.get_config(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    time_str = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = os.path.join(cfg.base_dir, time_str)
    checkpoint_dir = os.path.join(exp_dir, cfg.checkpoint_dir)
    log_dir = os.path.join(exp_dir, cfg.log_dir)

    save_config(cfg, exp_dir, "config.json")

    print("Log path: ", log_dir, "Checkpoint Dir: ", checkpoint_dir)

    data_shape = {'image': (None, 3, 64, 64)}
    cfg.data_shapes = data_shape

    model = KeypointModel(cfg)

    cp_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir,"model_"),
                                  period=2, save_top_k=-1)
    logger = TensorBoardLogger(log_dir, name="", version=None)

    gpus = 1 if args.cuda else None

    if args.pretrained_path:
        checkpoint_path = get_latest_checkpoint(args.pretrained_path)
        import json
        model = KeypointModel.load_from_checkpoint(checkpoint_path)
        print(json.dumps(model.cfg, indent=4))

    print("On GPU Device: ", gpus)
    trainer = Trainer(max_epochs=args.num_epochs,
                      logger=logger,
                      checkpoint_callback=cp_callback,
                      gpus=gpus,
                      progress_bar_refresh_rate=1,
                      gradient_clip_val=cfg.clipnorm,
                      fast_dev_run=False)
    trainer.fit(model)

if __name__ == "__main__":
    from register_args import get_argparse, save_config

    args = get_argparse(False).parse_args()

    main(args)