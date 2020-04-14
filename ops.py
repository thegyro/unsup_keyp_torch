import enum
import torch


EPSILON = 1e-6  # Constant for numerical stability.


class Axis(enum.Enum):
    """Maps axes to image indices, assuming that 0th dimension is the batch."""
    y = 1
    x = 2

def maps_to_keypoints(heatmaps):
    """Turns feature-detector heatmaps into (x, y, scale) keypoints.

    This function takes a tensor of feature maps as input. Each map is normalized
    to a probability distribution and the location of the mean of the distribution
    (in image coordinates) is computed. This location is used as a low-dimensional
    representation of the heatmap (i.e. a keypoint).

    To model keypoint presence/absence, the mean intensity of each feature map is
    also computed, so that each keypoint is represented by an (x, y, scale)
    triplet.

    Args:
      heatmaps: [batch_size, num_keypoints, H, W] tensors.
    Returns:
      A [batch_size, num_keypoints, 3] tensor with (x, y, scale)-triplets for each
      keypoint. Coordinate range is [-1, 1] for x and y, and [0, 1] for scale.
    """

    # Check that maps are non-negative:
    heatmaps = heatmaps.permute(0, 2, 3, 1)
    map_min = torch.min(heatmaps)
    assert map_min >= 0, "Heatmaps must be non-negative"

    x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
    y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)

    map_scales = torch.mean(heatmaps, dim=(1,2))

    # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
    # degeneracy between the encoder and decoder heatmap scales and ensures that
    # the scales are in a reasonable range for the RNN:
    map_scales /= (EPSILON + torch.max(map_scales, dim=-1, keepdim=True)[0])

    return torch.stack([x_coordinates, y_coordinates, map_scales], dim=-1)

def _maps_to_coordinates(maps, axis):
    """Reduces heatmaps to coordinates along one axis (x or y).

    Args:
      maps: [batch_size, H, W, num_keypoints] tensors.
      axis: Axis Enum.

    Returns:
      A [batch_size, num_keypoints, 2] tensor with (x, y)-coordinates.
    """

    width = maps.size()[axis.value]
    grid = _get_pixel_grid(axis, width).to(maps.device)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    grid = torch.reshape(grid, shape)

    if axis == Axis.x:
        marginalize_dim = 1
    elif axis == Axis.y:
        marginalize_dim = 2

    # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
    weights = torch.sum(maps + EPSILON, dim=marginalize_dim, keepdim=True)
    weights /= torch.sum(weights, dim=axis.value, keepdim=True)

    # Compute the center of mass of the marginalized maps to obtain scalar
    # coordinates:
    coordinates = torch.sum(weights * grid, dim=axis.value, keepdim=True)

    return coordinates.squeeze(2).squeeze(1)

def _get_pixel_grid(axis, width):
    """Returns an array of length `width` containing pixel coordinates."""
    if axis == Axis.x:
        return torch.linspace(-1.0, 1.0, width)  # Left is negative, right is positive.
    elif axis == Axis.y:
        return torch.linspace(1.0, -1.0, width)

def keypoints_to_maps(keypoints, sigma=1.0, heatmap_width=16):
    """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).

    Args:
      keypoints: [batch_size, num_keypoints, 3] tensor of keypoints where the last
        dimension contains (x, y, scale) triplets.
      sigma: Std. dev. of the Gaussian blob, in units of heatmap pixels.
      heatmap_width: Width of output heatmaps in pixels.

    Returns:
      A [batch_size, heatmap_width, heatmap_width, num_keypoints] tensor.
    """

    coordinates, map_scales = torch.split(keypoints, [2, 1], dim=-1)

    def get_grid(axis):
        grid = _get_pixel_grid(axis, heatmap_width).to(coordinates.device)
        shape = [1, 1, 1, 1]
        shape[axis.value] = -1
        return torch.reshape(grid, shape)

    # Expand to [batch_size, 1, 1, num_keypoints] for broadcasting later:
    x_coordinates = coordinates[:, None, None, :, 0]
    y_coordinates = coordinates[:, None, None, :, 1]

    # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
    keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0
    x_vec = torch.exp(-((get_grid(Axis.x) - x_coordinates)**2.0)/keypoint_width)
    y_vec = torch.exp(-((get_grid(Axis.y) - y_coordinates)**2.0)/keypoint_width)
    maps = x_vec * y_vec

    maps = maps * map_scales[:, None, None, :, 0]
    return maps.permute(0, 3, 1, 2)


def add_coord_channels(image_tensor):
    """Adds channels containing pixel indices (x and y coordinates) to an image.

    Note: This has nothing to do with keypoint coordinates. It is just a data
    augmentation to allow convolutional networks to learn non-translation-
    equivariant outputs. This is similar to the "CoordConv" layers:
    https://arxiv.org/abs/1603.09382.

    Args:
    image_tensor: [batch_size, C, H, W] tensor.

    Returns:
    [batch_size, C + 2, H, W] tensor with x and y coordinate channels.
    """

    batch_size, C, y_size, x_size = image_tensor.shape

    x_grid = torch.linspace(-1.0, 1.0, x_size).to(image_tensor.device)
    x_map = x_grid[None, None, None, :].repeat((batch_size, 1, y_size, 1))

    y_grid = torch.linspace(1.0, -1.0, y_size).to(image_tensor.device)
    y_map = y_grid[None, None, :, None].repeat((batch_size, 1, 1, x_size))

    return torch.cat([image_tensor, x_map, y_map], dim=1)