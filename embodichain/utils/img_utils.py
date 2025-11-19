# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
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
