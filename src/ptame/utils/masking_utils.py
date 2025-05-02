from functools import partial

import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import functional as F

from ptame.data.imagenet_datamodule import norm


def minmax_4d(cam_maps: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Normalize a 4D tensor to [0, 1] range with min-max. Per batch and
    channel.

    The normalization is done per 2d map
    """
    max_vals = cam_maps.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1)
    min_vals = cam_maps.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(-1)
    return (cam_maps - min_vals) / torch.clamp(max_vals - min_vals, min=eps)


def channel_minmax_4d(cam_maps: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Normalize a 4D tensor to [0, 1] range.

    The normalization is done per batch element
    """
    max_vals = (
        cam_maps.flatten(1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    )
    min_vals = (
        cam_maps.flatten(1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    )
    return (cam_maps - min_vals) / torch.clamp(max_vals - min_vals, min=eps)


def random_map_select(maps, targets):
    """Psuedo-randomly select the maps based on the targets of the batch.

    For each image, the mmapsasks corresponding to all the targets are
    selected. For N images and C targets, N x C maps are selected. C has to be
    less than N.
    """
    return maps[:, targets, :, :]


def single_map_select(maps, targets):
    """Select a single map per image based on the targets of the batch.

    For N images, N maps are selected.
    """
    return maps[torch.arange(maps.size(0)), targets, :, :].unsqueeze(1)


def single_map_select_thresh(maps, targets, threshold=0.25):
    """Select a single map per image based on the targets of the batch. Also
    applies a threshold on the map. The theory is that no more than 25% of the
    image can be important for the classification. The aim is to avoid using
    the area loss to force the map to be of low activation.

    For N images, N maps are selected.
    """
    masks = maps[torch.arange(maps.size(0)), targets, :, :].unsqueeze(1)
    masks = classic_mask_thresholding(masks, 1 - threshold)
    return masks


def naive_masking(x, masks):
    """Naive masking procedure that applies the masks to the input tensor.

    The masks are resized to the input tensor size. Then, the masks are applied
    to the input tensor by element-wise multiplication.
    """
    x_masked = x * masks
    return x_masked


def gaussian_masking(x, masks, kernel_size=55):
    # Apply Gaussian blur to the images
    blurred_x = torch.vmap(
        lambda img: TF.gaussian_blur(img, kernel_size=kernel_size)
    )(x)

    # Combine the original and blurred images using the mask
    masked_image = masks * x + (1 - masks) * blurred_x
    return masked_image


def mean_masking(x, masks):
    # Apply mean masking to the images
    means = x.mean(dim=(2, 3), keepdim=True)

    # replace the masked regions with the mean of the image
    masked_image = masks * x + (1 - masks) * means
    return masked_image


def basic_resize(masks, x_size):
    """Resize the masks to the input tensor size.

    The masks are resized to the size of the input tensor using bilinear
    interpolation.
    """
    _, _, H, W = x_size
    masks = F.interpolate(masks, size=(H, W), mode="bilinear")

    return masks


def nearest_resize(masks, x_size):
    """Resize the masks to the input tensor size.

    The masks are resized to the size of the input tensor using nearest
    interpolation. Current function's purposeis to identify if there is problem
    w/ AMP & interpolate inspired from related issues:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    https://github.com/pytorch/pytorch/issues/104157
    """
    _, _, H, W = x_size
    masks = F.interpolate(masks, size=(H, W), mode="nearest")
    return masks


def classic_mask_thresholding(
    masks: Tensor, percent: float, reverse=False
) -> Tensor:
    """Threshold the masks.

    :param masks: The masks to threshold.
    :param percent: The percentage of the mask to REMOVE (turn to 0).
    """
    percent = 1 - percent

    if percent > 0:
        flat_masks = masks.flatten(2)
        threshold = torch.kthvalue(
            flat_masks, int(flat_masks[0].numel() * percent)
        ).values[:, :, None, None]
        if not reverse:
            threshold_mask = masks > threshold
        else:
            threshold_mask = masks < threshold
        return masks * threshold_mask
    else:
        return masks


def binary_mask_thresholding(
    masks: Tensor, percent: float, most: bool = False
) -> Tensor:
    """Generate binary masks.

    The percentage refers to the percentage of the mask to remove, in both MoRF
    and LeRF cases.
    """

    if most:
        percent = 1 - percent

    def single_mask_threshold(mask: Tensor) -> Tensor:
        flat_mask = mask.flatten()
        threshold = torch.kthvalue(
            flat_mask, int(flat_mask.numel() * percent)
        ).values
        if not most:
            threshold_mask = mask > threshold
        else:
            threshold_mask = mask < threshold
        return threshold_mask

    vectorized = torch.vmap(single_mask_threshold)
    return vectorized(masks)


def resize_mask(resize=basic_resize, masking=naive_masking):
    """Compose the norm and masking procedures.

    The norm is applied in reverse to the input tensor. The masks are applied
    to the input tensor. The resulting tensor is normalized.
    """

    def resize_mask_inner(x, masks):
        masks = resize(masks, x.size())
        x_masked = masking(x, masks)
        return x_masked

    return resize_mask_inner


def instance_standardize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Standardize the input tensor (B, C, H, W) per instance (B), per
    channel(C)."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + eps
    return (x - mean) / std


def norm_resize_mask(
    unnorm=partial(norm, reverse=True),
    resize=basic_resize,
    masking=naive_masking,
    norm=norm,
):
    """Compose the norm and masking procedures.

    The norm is applied in reverse to the input tensor. The masks are applied
    to the input tensor. The resulting tensor is normalized.
    """

    def norm_resize_mask_inner(x, masks):
        x = unnorm(x)
        masks = resize(masks, x.size())
        x_masked = masking(x, masks)
        x_masked = norm(x_masked)
        return x_masked

    return norm_resize_mask_inner


def resize_thresh(
    resize=partial(basic_resize, x_size=(1, 1, 224, 224)),
    thresholding=classic_mask_thresholding,
):
    """Compose the resize and thresholding procedures.

    The masks are resized to the input tensor size. The masks are thresholded.
    """

    def resize_thresh_inner(masks, percent):
        masks = resize(
            masks,
        )
        masks = thresholding(masks, percent)
        return masks

    return resize_thresh_inner


def norm_mask(norm=norm, masking=naive_masking):
    """Compose the norm and masking procedures.

    The norm is applied in reverse to the input tensor. The masks are applied
    to the input tensor. The resulting tensor is normalized.
    """

    def norm_resize_mask_inner(x, masks):
        x = norm(x, reverse=True)
        x_masked = masking(x, masks)
        x_masked = norm(x_masked)
        return x_masked

    return norm_resize_mask_inner
