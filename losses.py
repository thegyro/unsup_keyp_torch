import torch


def get_heatmap_penalty(weight_matrix):
    """L1-loss on mean heatmap activations, to encourage sparsity."""
    weight_shape = list(weight_matrix.shape)
    assert len(weight_shape) == 4, weight_shape

    heatmap_mean = torch.mean(weight_matrix, dim=(2, 3))
    penalty = torch.mean(torch.abs(heatmap_mean))
    return penalty


def get_heatmap_seq_loss(heatmaps_seq):
    losses = []
    num_seq = heatmaps_seq.shape[1]
    for i in range(num_seq):
        heatmaps = heatmaps_seq[:,i, :, :]
        losses.append(get_heatmap_penalty(heatmaps))

    return torch.sum(torch.stack(losses))

def kl_divergence_loss(mu_1, std_1, mu_2, std_2):
    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mu_1 - mu_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element, dim=-1)


def temporal_separation_loss(cfg, coords):
    """Encourages keypoint to have different temporal trajectories.

    If two keypoints move along trajectories that are identical up to a time-
    invariant translation (offset), this suggest that they both represent the same
    object and are redundant, which we want to avoid.

    To measure this similarity of trajectories, we first center each trajectory by
    subtracting its mean. Then, we compute the pairwise distance between all
    trajectories at each timepoint. These distances are higher for trajectories
    that are less similar. To compute the loss, the distances are transformed by
    a Gaussian and averaged across time and across trajectories.

    Args:
      cfg: ConfigDict.
      coords: [batch, time, num_landmarks, 3] coordinate tensor.

    Returns:
      Separation loss.
    """
    x = coords[Ellipsis, 0]
    y = coords[Ellipsis, 1]

    # Center trajectories:
    x = x - torch.mean(x, dim=1, keepdim=True)
    y = y - torch.mean(y, dim=1, keepdim=True)

    # Compute pairwise distance matrix:
    d = ((x[:, :, :, None] - x[:, :, None, :]) ** 2.0 +
         (y[:, :, :, None] - y[:, :, None, :]) ** 2.0)

    # Temporal mean:
    d = torch.mean(d, dim=1)

    # Apply Gaussian function such that loss falls off with distance:
    loss_matrix = torch.exp(-d / (2.0 * cfg.separation_loss_sigma ** 2.0))
    loss_matrix = torch.mean(loss_matrix, dim=0)  # Mean across batch.
    loss = torch.sum(loss_matrix)  # Sum matrix elements.

    # Subtract sum of values on diagonal, which are always 1:
    loss -= cfg.num_keypoints

    # Normalize by maximal possible value. The loss is now scaled between 0 (all
    # keypoints are infinitely far apart) and 1 (all keypoints are at the same
    # location):
    loss /= cfg.num_keypoints * (cfg.num_keypoints - 1)

    return loss

if __name__ == "__main__":
    w = torch.ones()
