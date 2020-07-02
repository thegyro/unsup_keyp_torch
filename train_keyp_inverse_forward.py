import os
import time
from datetime import datetime
from itertools import islice

from pytz import timezone

import torch.nn.functional as F
from torch import optim

import datasets
import hyperparameters
import utils
from losses import temporal_separation_loss, get_heatmap_seq_loss
import torch


from utils import get_latest_checkpoint
from vision import ImagesToKeypEncoder, KeypToImagesDecoder, KeypToImagesDecoderNoFirst, KeypPredictor, KeypInverseModel

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np

from visualizer import save_img_keyp


class KeypointModel(pl.LightningModule):

    def __init__(self, hparams):
        super(KeypointModel, self).__init__()

        cfg = hparams
        input_shape_no_batch = cfg.data_shapes['image'][1:]

        # define all the models
        self.images_to_keypoints_net = ImagesToKeypEncoder(cfg, input_shape_no_batch)
        self.keyp_inverse_net = KeypInverseModel(cfg)
        self.keyp_pred_net = KeypPredictor(cfg)

        self.cfg = cfg
        self.hparams = cfg

        self.log_steps = 0

    def forward(self, img_seq, action_seq):
        keypoints_seq, heatmaps_seq = self.images_to_keypoints_net(img_seq)
        pred_action_seq = self.keyp_inverse_net(keypoints_seq[Ellipsis, :2])

        pred_keyp_seq = self.keyp_pred_net(keypoints_seq[Ellipsis, :2], action_seq)
        pred_keyp_seq = torch.cat((pred_keyp_seq, keypoints_seq[:, 1:, :, 2].unsqueeze(3)), dim=3)

        return keypoints_seq, \
               heatmaps_seq, \
               pred_keyp_seq,\
               pred_action_seq

    def unroll(self, img_seq, action_seq):
        keypoints_seq, _ = self.images_to_keypoints_net(img_seq)
        keypoint_0 = keypoints_seq[:, 0, :, :2]
        pred_keypoints_seq = self.keyp_pred_net.unroll(keypoint_0, action_seq)
        pred_keypoints_seq = torch.cat((pred_keypoints_seq, keypoints_seq[:, 1:, :, 2].unsqueeze(3)), dim=3)

        return pred_keypoints_seq

    def img_to_keyp(self, img_seq):
        keypoints_seq, _ = self.images_to_keypoints_net(img_seq)

        return keypoints_seq

    # def keyp_to_img(self, keyp_seq):
    #     pred_img_seq = self.keypoints_to_images_net(keyp_seq)
    #     return pred_img_seq

    def step(self, batch, batch_idx, is_train=True):
        data = batch
        img_seq = data['image']
        action_seq = data['action']

        keypoints_seq, heatmaps_seq, pred_keyp_seq, pred_action_seq= self.forward(img_seq, action_seq)
        heatmap_loss = get_heatmap_seq_loss(heatmaps_seq)

        pred_action_loss = F.mse_loss(pred_action_seq, action_seq[:,:-1], reduction='sum')
        pred_action_loss /= (pred_action_seq.shape[0] * pred_action_seq.shape[1])

        pred_keyp_coord_seq, keyp_coord_seq = pred_keyp_seq[Ellipsis, :2], keypoints_seq[:, 1:, :, :2]
        pred_keyp_loss = F.mse_loss(pred_keyp_coord_seq, keyp_coord_seq, reduction='sum')
        pred_keyp_loss /= (pred_keyp_coord_seq.shape[0] * pred_keyp_coord_seq.shape[1])

        T = self.cfg.observed_steps
        temporal_loss = temporal_separation_loss(self.cfg, keypoints_seq[:, :T])

        loss = (heatmap_loss * self.cfg.heatmap_regularization) + \
               (temporal_loss * self.cfg.separation_loss_scale) + \
               (pred_action_loss * self.cfg.pred_action_loss_scale) + \
               (pred_keyp_loss * self.cfg.pred_keyp_loss_scale)

        pfx = '' if is_train else 'test_'
        output = {
            pfx + 'loss': loss,
            pfx + 'hmap_loss': heatmap_loss,
            pfx + 'temporal_loss': temporal_loss,
            pfx + 'pred_action_loss': pred_action_loss,
            pfx + 'pred_keyp_loss': pred_keyp_loss
        }

        if self.cfg.log_training and is_train:
            if self.global_step % 500 == 0:  self.log_train_viz()

        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, False)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, False)

    def aggregate_metrics(self, outputs, is_train=True):
        pfx = '' if is_train else 'test_'
        avg_loss = torch.stack([x[pfx + 'loss'] for x in outputs]).mean()
        avg_hmap_loss = torch.stack([x[pfx + 'hmap_loss'] for x in outputs]).mean()
        avg_temporal_loss = torch.stack([x[pfx + 'temporal_loss'] for x in outputs]).mean()
        avg_action_loss = torch.stack([x[pfx + 'pred_action_loss'] for x in outputs]).mean()
        avg_keyp_pred_loss = torch.stack([x[pfx + 'pred_keyp_loss'] for x in outputs]).mean()

        pfx = "train/" if is_train else "test/"
        logs = {
            pfx + 'loss': avg_loss,
            pfx + 'hmap_loss': avg_hmap_loss,
            pfx + 'temporal_loss': avg_temporal_loss,
            pfx + 'pred_keyp_loss': avg_keyp_pred_loss,
            pfx + 'pred_action_loss': avg_action_loss
        }
        return logs

    def training_epoch_end(self, outputs):
        logs = self.aggregate_metrics(outputs, True)
        return {'log': logs, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        logs = self.aggregate_metrics(outputs, False)
        print()
        return {'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        logs = self.aggregate_metrics(outputs, False)
        print()
        return {'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        return [optimizer], [scheduler]
        #return optimizer

    def train_dataloader(self):
        train_loader, _ = datasets.get_sequence_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.train_dir),
            batch_size=self.cfg.batch_size,
            num_timesteps=self.cfg.observed_steps + self.cfg.predicted_steps)

        return train_loader

    def val_dataloader(self):
        self.val_loader, _ = datasets.get_sequence_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.test_dir),
            batch_size=self.cfg.batch_size,
            num_timesteps=self.cfg.observed_steps + self.cfg.predicted_steps, shuffle=False)

        return self.val_loader

    def test_dataloader(self):
        test_loader, _ = datasets.get_sequence_dataset(
            data_dir=os.path.join(self.cfg.data_dir, self.cfg.test_dir),
            batch_size=self.cfg.batch_size,
            num_timesteps=self.cfg.observed_steps + self.cfg.predicted_steps, shuffle=False)

        return test_loader

    def log_train_viz(self):
        print('\n',"*******Logging Intermediate Training: ", self.global_step, '*************\n')
        for data in islice(self.val_loader, 4):
            with torch.no_grad():
                img_seq = data['image'].to(torch.device(self.cfg.device))
                file_id_seq = data['file_idx']
                frame_id_seq = data['frame_ind']

                keyp_seq = self.img_to_keyp(img_seq)

                s_n, s_t = 2, 5
                img_sample_seq = img_seq[s_n:s_n+1, s_t]
                keyp_sample_seq = keyp_seq[s_n:s_n+1, s_t]
                file_id_sample_seq = file_id_seq[s_n:s_n+1, s_t]
                frame_id_sample_seq = frame_id_seq[s_n:s_n+1, s_t]

                self.save_sample_keyp(img_sample_seq, keyp_sample_seq,
                                      file_id_sample_seq, frame_id_sample_seq,
                                      self.global_step, self.cfg.log_training_path)

    def save_sample_keyp(self, img_seq, keyp_seq,
                         file_id_seq, frame_id_seq, step_num, save_dir):
        """
        :param img_seq: N x 3 x H x W
        :param keyp_seq: N x num_keyp x 3
        :param step_num: int
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        img_seq_np = utils.img_torch_to_numpy(img_seq)
        keyp_seq_np = keyp_seq.cpu().numpy()

        file_dir = "file_{}_frame_{}"

        N, num_keyp = keyp_seq_np.shape[:2]

        for n in range(N):
            file_id = file_id_seq[n]
            frame_id = frame_id_seq[n]
            img = img_seq_np[n]
            keyps = keyp_seq_np[n]

            save_file_dir = os.path.join(save_dir, file_dir.format(file_id, frame_id))

            if not os.path.isdir(save_file_dir):
                os.makedirs(save_file_dir)

            keyps_history_path = os.path.join(save_file_dir, "keyps_history.npy")
            if not os.path.isfile(keyps_history_path):
                keyps_history = keyps[np.newaxis,:,:]
            else:
                prev_keyps_history = np.load(keyps_history_path)
                keyps_history = np.concatenate((prev_keyps_history, keyps[np.newaxis,:,:]))

            for k in range(num_keyp):
                save_path = os.path.join(save_file_dir, 'keyp_{}.png'.format(k))
                keyp_history = keyps_history[:, k]
                save_img_keyp(img, keyp_history, save_path, k, step_num)

            np.save(keyps_history_path, keyps_history)

        self.log_steps += 1

def main(args):
    utils.set_seed_everywhere(args.seed)

    cfg = hyperparameters.get_config(args)
    cfg.seed = args.seed
    cfg.base_dir = cfg.base_dir + "_s_" + str(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    time_str = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = os.path.join(cfg.base_dir, time_str)
    checkpoint_dir = os.path.join(exp_dir, cfg.checkpoint_dir)
    log_dir = os.path.join(exp_dir, cfg.log_dir)

    cfg.log_training = args.log_training
    cfg.log_training_path = os.path.join(exp_dir, args.log_training_path)
    cfg.num_steps = args.num_steps
    cfg.device =  str(torch.device("cuda" if args.cuda else "cpu"))

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
    trainer = Trainer(max_epochs=10000,
                      max_steps=args.num_steps,
                      logger=logger,
                      checkpoint_callback=cp_callback,
                      gpus=gpus,
                      #distributed_backend='dp',
                      progress_bar_refresh_rate=1,
                      #gradient_clip_val=cfg.clipnorm,
                      fast_dev_run=False,
                      #train_percent_check=0.1,val_percent_check=0.0,
                      val_percent_check=0.3,
                      track_grad_norm=2,
                      num_sanity_val_steps = 0,
                      show_progress_bar=True)
    trainer.fit(model)
    save_path = os.path.join(checkpoint_dir, "model_final_" + str(args.num_steps) + ".ckpt")
    print("Saving model finally:")
    trainer.save_checkpoint(save_path)

if __name__ == "__main__":
    from register_args import get_argparse, save_config

    args = get_argparse(False).parse_args()

    main(args)