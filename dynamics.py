from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from losses import kl_divergence_loss

from utils import stack_time, unstack_time

import ops


class PriorNet(nn.Module):
    def __init__(self, cfg):
        super(PriorNet, self).__init__()
        self.cfg = cfg

        self.hidden = nn.Linear(cfg.num_rnn_units, cfg.prior_net_dim)
        self.means = nn.Linear(cfg.prior_net_dim, cfg.latent_code_size)
        self.stds_raw = nn.Linear(cfg.prior_net_dim, cfg.latent_code_size)

    def forward(self, rnn_state):
        h = F.relu(self.hidden(rnn_state))
        mu = self.means(h)
        sigma_raw = self.stds_raw(h)
        sigma = F.softplus(sigma_raw) + 1e-4

        return mu, sigma


class PosteriorNet(nn.Module):
    def __init__(self, cfg):
        super(PosteriorNet, self).__init__()

        hidden_indim = cfg.num_keypoints * 3 + cfg.num_rnn_units
        self.hidden = nn.Linear(hidden_indim, cfg.posterior_net_dim)
        self.means = nn.Linear(cfg.posterior_net_dim, cfg.latent_code_size)
        self.stds_raw = nn.Linear(cfg.posterior_net_dim, cfg.latent_code_size)

    def forward(self, rnn_state, observed_keypoints):
        x = torch.cat([rnn_state, observed_keypoints], dim=-1)
        h = F.relu(self.hidden(x))
        mu = self.means(h)
        sigma_raw = self.stds_raw(h)
        sigma = F.softplus(sigma_raw) + 1e-4

        return mu, sigma


def reparametrize(mu, std):
    eps = torch.randn_like(std).to(std.device)
    return mu + std * eps


class KeypointDecoder(nn.Module):
    def __init__(self, cfg):
        super(KeypointDecoder, self).__init__()
        hidden_indim = cfg.num_rnn_units + cfg.latent_code_size
        self.hidden = nn.Linear(hidden_indim, 128)
        self.keypoints = nn.Linear(128, cfg.num_keypoints * 3)

    def forward(self, rnn_state, latent_code):
        x = torch.cat([rnn_state, latent_code], dim=-1)
        h = F.relu(self.hidden(x))
        keyps = torch.tanh(self.keypoints(h))

        return keyps


class VRNNKeypoints(nn.Module):
    def __init__(self, cfg):
        super(VRNNKeypoints, self).__init__()

        rnn_inputdim = cfg.num_keypoints * 3 + cfg.latent_code_size
        self.rnn_cell = nn.GRUCell(rnn_inputdim, cfg.num_rnn_units)

        self.prior_net = PriorNet(cfg)
        self.posterior_net = PosteriorNet(cfg)
        self.decoder = KeypointDecoder(cfg)

        self.num_timesteps = cfg.observed_steps + cfg.predicted_steps

        # initial RNN state
        self.cfg = cfg

    def vrnn_iteration(self, cfg,
                       input_keypoints,
                       rnn_state, rnn_cell,
                       prior_net, decoder,
                       scheduled_sampler,
                       posterior_net=None):

        batch_size = input_keypoints.size()[0]
        input_keypoints_flat = input_keypoints.view(batch_size, -1)

        mu_prior, std_prior = prior_net(rnn_state)

        if posterior_net:
            mu, std = posterior_net(rnn_state, input_keypoints_flat)
            kl_divergence = kl_divergence_loss(mu, std, mu_prior, std_prior)
        else:
            mu, std = mu_prior.detach(), std_prior.detach()
            kl_divergence = None

        latent_belief = reparametrize(mu, std)
        output_keypoints_flat = decoder(rnn_state, latent_belief)

        # TODO: scheduled sampling
        keypoints_for_rnn = output_keypoints_flat

        # Execute RNN - step forward in time
        rnn_input = torch.cat([keypoints_for_rnn, latent_belief], dim=-1)
        rnn_state = rnn_cell(rnn_input, rnn_state)

        output_keypoints = output_keypoints_flat.view_as(input_keypoints)

        return output_keypoints, rnn_state, kl_divergence

    def forward(self, keypoints_seq):
        keypoints_seq_list = unstack_time(keypoints_seq)

        output_keypoints_list = [None] * self.num_timesteps
        kl_div_list = [None] * self.cfg.observed_steps

        rnn_state = torch.zeros([keypoints_seq.shape[0], self.cfg.num_rnn_units]).to(keypoints_seq.device)

        for t in range(self.cfg.observed_steps):
            output_keypoints_list[t], rnn_state, kl_div_list[t] = self.vrnn_iteration(
                self.cfg, keypoints_seq_list[t],
                rnn_state, self.rnn_cell,
                self.prior_net, self.decoder,
                None, self.posterior_net)

        for t in range(self.cfg.observed_steps, self.num_timesteps):
            output_keypoints_list[t], rnn_state, _ = self.vrnn_iteration(
                self.cfg, keypoints_seq_list[t],
                rnn_state, self.rnn_cell,
                self.prior_net, self.decoder,
                None, None)

        output_keypoints_seq = stack_time(output_keypoints_list)
        kl_div_seq = stack_time(kl_div_list)

        return output_keypoints_seq, kl_div_seq

    def unroll(self, keypoints_seq, T_future):
        keypoints_seq_list = unstack_time(keypoints_seq)
        T_obs = len(keypoints_seq_list)//2
        output_keypoints_list = [None] * (T_obs + T_future)

        rnn_state = torch.zeros([keypoints_seq.shape[0], self.cfg.num_rnn_units]).to(keypoints_seq.device)
        for t in range(T_obs):
            output_keypoints_list[t], rnn_state, _ = self.vrnn_iteration(
                self.cfg, keypoints_seq_list[t],
                rnn_state, self.rnn_cell,
                self.prior_net, self.decoder,
                None, self.posterior_net)

        for t in range(T_obs, T_obs + T_future):
            output_keypoints_list[t], rnn_state, _ = self.vrnn_iteration(
                self.cfg, keypoints_seq_list[-1],
                rnn_state, self.rnn_cell,
                self.prior_net, self.decoder,
                None, None)

        output_keypoints_seq = stack_time(output_keypoints_list)

        return output_keypoints_seq