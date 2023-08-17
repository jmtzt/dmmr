from pathlib import Path
from typing import Optional, Union, Dict, Set, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn as nn, Tensor

from deepali.core import functional as U, PathStr, Grid
from deepali.data import Image


def append_data(
        data: Optional[Tensor], channels: Dict[str, Tuple[int, int]], name: str, other: Tensor
) -> Tensor:
    r"""Append image data."""
    if data is None:
        data = other
    else:
        data = torch.cat([data, other], dim=0)
    channels[name] = (data.shape[0] - other.shape[0], data.shape[0])
    return data


def read_images(
        sample: Union[PathStr, Dict[str, PathStr]], names: Set[str], device: torch.device
) -> Tuple[Image, Dict[str, Tuple[int, int]]]:
    r"""Read image data from input files."""
    data = None
    grid = None
    if isinstance(sample, (Path, str)):
        sample = {"img": sample}
    img_path = sample.get("img")
    seg_path = sample.get("seg")
    sdf_path = sample.get("sdf")
    for path in (img_path, seg_path, sdf_path):
        if not path:
            continue
        grid = Grid.from_file(path).align_corners_(True)
        break
    else:
        raise ValueError("One of 'img', 'seg', or 'sdf' input image file paths is required")
    assert grid is not None
    dtype = torch.float32
    channels = {}
    if "img" in names:
        temp = Image.read(img_path, dtype=dtype, device=device)
        data = append_data(data, channels, "img", temp.tensor())
    if "seg" in names:
        if seg_path is None:
            raise ValueError("Missing segmentation label image file path")
        temp = Image.read(seg_path, dtype=torch.int64, device=device)
        temp_grid = temp.grid()
        num_classes = int(temp.max()) + 1
        temp = temp.tensor().unsqueeze(0)
        temp = U.as_one_hot_tensor(temp, num_classes).to(dtype=dtype)
        temp = temp.squeeze(0)
        temp = Image(temp, grid=temp_grid).sample(grid)
        data = append_data(data, channels, "seg", temp.tensor())
    if "sdf" in names:
        if sdf_path is None:
            raise ValueError("Missing segmentation boundary signed distance field file path")
        temp = Image.read(sdf_path, dtype=dtype, device=device)
        temp = temp.sample(grid)
        data = append_data(data, channels, "sdf", temp.tensor())
    if data is None:
        if img_path is None:
            raise ValueError("Missing intensity image file path")
        data = Image.read(img_path, dtype=dtype, device=device)
        channels = {"img": (0, 1)}
    image = Image(data, grid=grid)
    return image, channels


def normalize_data_(
        image: Image,
        channels: Dict[str, Tuple[int, int]],
        shift: Optional[Dict[str, Tensor]] = None,
        scale: Optional[Dict[str, Tensor]] = None,
) -> Image:
    r"""Normalize image data."""
    if shift is None:
        shift = {}
    if scale is None:
        scale = {}
    for channel, (start, stop) in channels.items():
        data = image.tensor().narrow(0, start, stop - start)
        offset = shift.get(channel)
        if offset is not None:
            data -= offset
        norm = scale.get(channel)
        if norm is not None:
            data /= norm
        if channel in ("msk", "seg"):
            data.clamp_(min=0, max=1)
    return image


def mask_to_3d_bbox(mask):
    mask = mask.squeeze()
    bounding_boxes = torch.zeros((6), device=mask.device, dtype=torch.float)

    z, y, x = torch.where(mask != 0)
    bounding_boxes[0] = torch.min(x)
    bounding_boxes[1] = torch.min(y)
    bounding_boxes[2] = torch.min(z)
    bounding_boxes[3] = torch.max(x)
    bounding_boxes[4] = torch.max(y)
    bounding_boxes[5] = torch.max(z)

    return bounding_boxes


def extract_patches(tensor, mask, size=17):
    wc, ws, wa = size, size, size  # window size
    sc, ss, sa = size, size, size  # stride

    x_min, y_min, z_min, x_max, y_max, z_max = mask_to_3d_bbox(mask)

    x_min, y_min, z_min = int(x_min.item()), int(y_min.item()), int(z_min.item())
    x_max, y_max, z_max = int(x_max.item()), int(y_max.item()), int(z_max.item())

    tensor = tensor[:, :, z_min:z_max, y_min:y_max, x_min:x_max]

    # Pad the input such that it is divisible by the window size
    padding_values = []
    for dim_size in tensor.shape[2:]:
        remainder = dim_size % wc
        if remainder != 0:
            padding = wc - remainder
        else:
            padding = 0
        padding_values.extend([padding // 2, padding - padding // 2])

    padding_values.reverse()
    padded = F.pad(tensor, padding_values, 'constant')

    # Create the patches of wc x ws x wa
    patches = padded.unfold(2, wc, sc).unfold(3, ws, ss).unfold(4, wa, sa)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, wc, ws, wa)

    return patches.unsqueeze(1)


def show_image(
        image: Tensor,
        label: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
) -> None:
    r"""Render image data in last two tensor dimensions using matplotlib.pyplot.imshow().

    Args:
        image: Image tensor of shape ``(..., H, W)``.
        ax: Figure axes to render the image in. If ``None``, a new figure is created.
        label: Image label to display in the axes title.
        kwargs: Keyword arguments to pass on to ``matplotlib.pyplot.imshow()``.
            When ``ax`` is ``None``, can contain ``figsize`` to specify the size of
            the figure created for displaying the image.

    """
    if ax is None:
        figsize = kwargs.pop("figsize", (4, 4))
        _, ax = plt.subplots(figsize=figsize)
    kwargs["cmap"] = kwargs.get("cmap", "gray")
    im = ax.imshow(image.reshape((-1,) + image.shape[-2:])[0].cpu().numpy(), **kwargs)
    if label:
        ax.set_title(label, fontsize=12, y=1.04)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return im


class DMMRLoss(nn.Module):

    def __init__(self,
                 patch_size=17,
                 model_path='/home/joao/repos/midir-thesis/dmmr_models/complete_camcan_tanh_hinge.pt',
                 zero_percentage_threshold=0.5,
                 ):
        super(DMMRLoss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path).to(self.device)
        self.zero_percentage_threshold = zero_percentage_threshold

    def forward(self, source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        binary_mask = torch.zeros_like(target).to(self.device)
        binary_mask[target > 0] = 1

        fixed_patches = extract_patches(target, binary_mask, size=self.patch_size)
        moving_patches = extract_patches(source, binary_mask, size=self.patch_size)

        keep_mask = torch.zeros(fixed_patches.shape[0])
        for i, patch in enumerate(fixed_patches):
            patch = patch[0].squeeze()
            zero_percentage = torch.mean((patch.squeeze() == 0).float()).item()
            if zero_percentage > self.zero_percentage_threshold:
                keep_mask[i] = 0
            else:
                keep_mask[i] = 1

        fixed_patches = fixed_patches[keep_mask.bool()]
        moving_patches = moving_patches[keep_mask.bool()]
        out = self.model(fixed_patches, moving_patches)
        value = torch.mean(out)

        return value
