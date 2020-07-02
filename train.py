import os
from datetime import datetime
from pytz import timezone

import torch.nn.functional as F
from torch import optim

import datasets
import hyperparameters
import utils
from losses import temporal_separation_loss, get_heatmap_seq_loss
import torch


from utils import get_latest_checkpoint
from vision import ImagesToKeypEncoder, KeypToImagesDecoder

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

        self.cfg = cfg
        self.hparams = cfg

    def forward(self, img_seq):
        keypoints_seq, heatmaps_seq = self.images_to_keypoints_net(img_seq)

        reconstructed_img_seq = self.keypoints_to_images_net(keypoints_seq,
                                                             img_seq[:, 0, :, :, :],
                                                             keypoints_seq[:, 0, :, :])

        return keypoints_seq, \
               heatmaps_seq, \
               reconstructed_img_seq

    def step(self, batch, batch_idx, is_train=True):
        data = batch
        img_seq = data['image']

        keypoints_seq, heatmaps_seq, reconstructed_img_seq = self.forward(img_seq)

        reconstruction_loss = F.mse_loss(img_seq, reconstructed_img_seq, reduction='sum')
        reconstruction_loss /= (img_seq.shape[0] * img_seq.shape[1])

        heatmap_loss = get_heatmap_seq_loss(heatmaps_seq)

        T = self.cfg.observed_steps
        temporal_loss = temporal_separation_loss(self.cfg,
                                                 keypoints_seq[:, :T])


        loss = reconstruction_loss + \
               (heatmap_loss * self.cfg.heatmap_regularization) + \
               (temporal_loss * self.cfg.separation_loss_scale)


        pfx = '' if is_train else 'test_'
        output = {
            pfx + 'loss': loss,
            pfx + 'recon_loss': reconstruction_loss,
            pfx + 'hmap_loss': heatmap_loss,
            pfx + 'temporal_loss': temporal_loss
        }

        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, False)

    def aggregate_metrics(self, outputs, is_train=True):
        pfx = '' if is_train else 'test_'
        avg_loss = torch.stack([x[pfx + 'loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x[pfx + 'recon_loss'] for x in outputs]).mean()
        avg_hmap_loss = torch.stack([x[pfx + 'hmap_loss'] for x in outputs]).mean()
        avg_temporal_loss = torch.stack([x[pfx + 'temporal_loss'] for x in outputs]).mean()

        pfx = "train/" if is_train else "test/"
        logs = {
            pfx+'loss': avg_loss,
            pfx+'recon_loss': avg_recon_loss,
            pfx+'hmap_loss': avg_hmap_loss,
            pfx+'temporal_loss': avg_temporal_loss
        }
        return logs

    def training_epoch_end(self, outputs):
        logs = self.aggregate_metrics(outputs, True)
        return {'log': logs, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        logs = self.aggregate_metrics(outputs, False)
        print()
        return {'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.75)
        #return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self):
        train_loader, _ = datasets.get_sequence_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.train_dir),
            batch_size=self.cfg.batch_size,
            num_timesteps=self.cfg.observed_steps + self.cfg.predicted_steps)

        return train_loader

    def val_dataloader(self):
        val_loader, _ = datasets.get_sequence_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.test_dir),
            batch_size=self.cfg.batch_size,
            num_timesteps=self.cfg.observed_steps + self.cfg.predicted_steps, shuffle=False)

        return val_loader


def main(args):
    utils.set_seed_everywhere(args.seed)

    cfg = hyperparameters.get_config(args)
    cfg.seed = args.seed

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    time_str = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = os.path.join(cfg.base_dir, time_str)
    checkpoint_dir = os.path.join(exp_dir, cfg.checkpoint_dir)
    log_dir = os.path.join(exp_dir, cfg.log_dir)

    save_config(cfg, exp_dir, "config.json")

    print("Log path: ", log_dir, "Checkpoint Dir: ", checkpoint_dir)

    num_timsteps = cfg.observed_steps + cfg.predicted_steps
    data_shape = {'image': (None, num_timsteps, 3, 64, 64)}
    cfg.data_shapes = data_shape

    model = KeypointModel(cfg)

    cp_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "model_"),
                                  period=25, save_top_k=-1)

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
                      #distributed_backend='dp',
                      progress_bar_refresh_rate=1,
                      #gradient_clip_val=cfg.clipnorm,
                      fast_dev_run=False,
                      #train_percent_check=0.1,val_percent_check=0.0,
                      #val_percent_check=0.3,
                      track_grad_norm=2,
                      show_progress_bar=True)
    trainer.fit(model)
    save_path = os.path.join(checkpoint_dir, "model_final_" + str(args.num_epochs) + ".ckpt")
    print("Saving model finally:")
    trainer.save_checkpoint(save_path)

if __name__ == "__main__":
    from register_args import get_argparse, save_config

    args = get_argparse(False).parse_args()

    main(args)