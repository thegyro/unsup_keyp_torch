# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Hyperparameters of the structured video prediction models."""
import os


class ConfigDict(dict):
    """A dictionary whose keys can be accessed as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def get(self, key, default=None):
        """Allows to specify defaults when accessing the config."""
        if key not in self:
            return default
        return self[key]


def get_config(FLAGS):
    """Default values for all hyperparameters."""

    cfg = ConfigDict()
    # Directories:
    cfg.exp_name = FLAGS.exp_name
    cfg.base_dir = os.path.join(FLAGS.base_dir, cfg.exp_name)

    cfg.data_dir = FLAGS.data_dir
    cfg.train_dir = FLAGS.train_dir
    cfg.test_dir = FLAGS.test_dir

    cfg.checkpoint_dir = FLAGS.checkpoint_dir
    cfg.log_dir = FLAGS.logs_dir
    cfg.pretrained_path = FLAGS.pretrained_path

    # Architecture:
    cfg.layers_per_scale = 2
    cfg.conv_layer_kwargs = _conv_layer_kwargs()
    cfg.dense_layer_kwargs = _dense_layer_kwargs()

    # Optimization:
    cfg.batch_size = FLAGS.batch_size
    cfg.steps_per_epoch = FLAGS.steps_per_epoch
    cfg.num_epochs = FLAGS.num_epochs
    cfg.learning_rate = FLAGS.learning_rate
    cfg.clipnorm = FLAGS.clipnorm

    # Image sequence parameters:
    cfg.observed_steps = FLAGS.timesteps
    cfg.predicted_steps = FLAGS.timesteps

    # Keypoint encoding settings:
    cfg.num_keypoints = FLAGS.num_keypoints
    cfg.heatmap_width = 16
    cfg.heatmap_regularization = FLAGS.heatmap_reg
    cfg.keypoint_width = 1.5
    cfg.num_encoder_filters = 32
    cfg.separation_loss_scale = FLAGS.temp_reg
    cfg.separation_loss_sigma = FLAGS.temp_width

    # Dynamics:
    cfg.num_rnn_units = 512
    cfg.prior_net_dim = 128
    cfg.posterior_net_dim = 128
    cfg.latent_code_size = 16
    cfg.kl_loss_scale = FLAGS.kl_reg
    cfg.kl_annealing_steps = 1000
    cfg.use_deterministic_belief = False
    cfg.scheduled_sampling_ramp_steps = (
        cfg.steps_per_epoch * int(cfg.num_epochs * 0.8))
    cfg.scheduled_sampling_p_true_start_obs = 1.0
    cfg.scheduled_sampling_p_true_end_obs = 0.1
    cfg.scheduled_sampling_p_true_start_pred = 1.0
    cfg.scheduled_sampling_p_true_end_pred = 0.5
    cfg.num_samples_for_bom = 1

    cfg.pred_keyp_loss_scale = FLAGS.keyp_reg
    cfg.pred_action_loss_scale = FLAGS.action_reg
    cfg.action_dim = FLAGS.action_dim

    return cfg


def _conv_layer_kwargs():
    """Returns a configDict with default conv layer hyperparameters."""

    cfg = ConfigDict()

    cfg.kernel_size = 3
    cfg.padding = 1

    return cfg


def _dense_layer_kwargs():
    """Returns a configDict with default dense layer hyperparameters."""

    cfg = ConfigDict()
    # cfg.activation = tf.nn.relu

    return cfg
