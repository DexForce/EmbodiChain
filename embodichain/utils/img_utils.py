# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
# ----------------------------------------------------------------------------

import torch


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """Convert binary masks to bounding boxes.

    Args:
        masks (torch.Tensor): A tensor of shape (..., H, W) containing binary masks
            where non-zero values indicate the presence of the object.

    Returns:
        torch.Tensor: A tensor of shape (..., 4) containing the bounding boxes
        in XYXY format.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def gen_disp_colormap(inputs, normalize=True, torch_transpose=True):
    """
    Generate an RGB visualization using the "plasma" colormap for 2D/3D/4D scalar image inputs.

    This utility maps scalar image(s) to an RGB colormap suitable for display or further processing.
    It accepts either a NumPy array or a torch.Tensor (torch tensors are detached, moved to CPU and
    converted to NumPy). The matplotlib "plasma" colormap with 256 entries is used.

    Parameters
    - inputs (numpy.ndarray or torch.Tensor):
        Scalar image data with one of the following dimensionalities:
          * 2D: (H, W)               -> a single image
          * 3D: (N, H, W)            -> a batch of N single-channel images
          * 4D: (N, C, H, W)         -> a batch with channel dimension; expected C==1 (first channel used)
        The function will convert torch.Tensor input to numpy internally.
    - normalize (bool, default True):
        If True, input values are linearly scaled to [0, 1] using (x - min) / (max - min).
        If the input is constant (min == max), a small divisor (1e5) is used to avoid division
        by zero, which effectively maps values near 0. If False, values are assumed to already be
        in the [0, 1] range (no scaling is performed).
    - torch_transpose (bool, default True):
        Controls the output channel ordering to match common PyTorch conventions:
          * If True: outputs are transposed to channel-first form:
              - 2D input  -> (3, H, W)
              - 3D input  -> (N, 3, H, W)
              - 4D input  -> (N, 3, H, W)  (uses the first channel)
          * If False: outputs keep channel-last ordering:
              - 2D input  -> (H, W, 3)
              - 3D input  -> (N, H, W, 3)
              - 4D input  -> (N, H, W, 3)

    Returns
    - numpy.ndarray:
        RGB image(s) with float values in [0, 1]. The exact output shape depends on the input
        dimensionality and the value of torch_transpose (see above). The alpha channel produced by
        the colormap is discarded; only the RGB channels are returned.

    Notes and behavior
    - The function uses matplotlib.pyplot.get_cmap("plasma", 256).
    - For 4D inputs the code selects the first channel (index 0) before applying the colormap.
    - Inputs with dimensionality other than 2, 3, or 4 are not supported and will likely raise
      an error or produce unintended results.
    - This function is non-destructive: it returns a new NumPy array and does not modify the input.
    - Typical use cases: visualizing depth maps, single-channel activation maps, or other scalar
      images as colored RGB images for inspection or logging.

    Examples
    - 2D array (H, W) -> returns (3, H, W) if torch_transpose=True
    - 3D array (N, H, W) -> returns (N, 3, H, W) if torch_transpose=True
    - 4D array (N, 1, H, W) -> returns (N, 3, H, W) if torch_transpose=True
    """
    import matplotlib.pyplot as plt
    import torch

    _DEPTH_COLORMAP = plt.get_cmap("plasma", 256)  # for plotting
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
