import os
from datetime import datetime
from pytz import timezone


import torch.nn.functional as F
import torch.nn as nn
from torch import optim

import datasets
import hyperparameters
from losses import get_heatmap_penalty
import torch

import numpy as np

from utils import get_latest_checkpoint

from vision_models import ImageEncoder, ImageDecoder

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class Autoencoder(pl.LightningModule):

    def __init__(self, hparams):
        super(Autoencoder, self).__init__()

        cfg = hparams
        # define all the models
        self.img_enc = ImageEncoder((3, 64, 64), **{'kernel_size': 3, 'padding':1})
        self.img_dec = ImageDecoder((128, 16, 16), output_width=64, layers_per_scale=2,
                                    **{'kernel_size': 3, 'padding':1})
        self.adjust = nn.Conv2d(cfg.num_encoder_filters, 3, kernel_size=1, padding=0)

        self.cfg = cfg
        self.hparams = cfg

    def forward(self, img):
        latent  = self.img_enc(img)
        recon_img =   self.adjust(self.img_dec(latent))

        return recon_img

    def training_step(self, batch, batch_idx):
        data = batch
        img = data['image']

        reconstructed_img = self.forward(img)

        reconstruction_loss = F.mse_loss(img, reconstructed_img, reduction='sum')
        reconstruction_loss /= img.shape[0]

        loss = reconstruction_loss

        log_dict = {'recon_loss': reconstruction_loss, 'loss': loss}

        output= {
            'loss': loss,
            'recon_loss': reconstruction_loss,
            'log': log_dict
        }

        #print(reconstruction_loss.item())

        return output

    def validation_step(self, batch, batch_idx):
        data = batch
        img = data['image']

        reconstructed_img = self.forward(img)

        reconstruction_loss = F.mse_loss(img, reconstructed_img, reduction='sum')
        reconstruction_loss /= img.shape[0]

        loss = reconstruction_loss

        tqdm_dict = {'val_loss': loss, 'val_recon_loss': reconstruction_loss}

        return {'val_loss': loss,
                'val_recon_loss': reconstruction_loss,
                'progress_bar': tqdm_dict}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_recon_loss': avg_recon_loss}
        return {'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.cfg.learning_rate, weight_decay=1e-4)
        return optimizer

    def train_dataloader(self):
        train_loader, _ = datasets.get_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.train_dir),
            batch_size=self.cfg.batch_size, shuffle=False)

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

    model = Autoencoder(cfg)

    cp_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir,"model_"),
                                  period=2, save_top_k=-1)
    logger = TensorBoardLogger(log_dir, name="", version=None)

    gpus = 1 if args.cuda else None

    print("On GPU Device: ", gpus)
    trainer = Trainer(max_epochs=args.num_epochs,
                      logger=logger,
                      checkpoint_callback=cp_callback,
                      gpus=gpus,
                      progress_bar_refresh_rate=1,
                      gradient_clip_val=0.0,
                      fast_dev_run=False)
    trainer.fit(model)

if __name__ == "__main__":
    from register_args import get_argparse, save_config

    args = get_argparse(False).parse_args()

    main(args)