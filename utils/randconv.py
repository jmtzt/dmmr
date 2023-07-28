import torch
import torchvision
from torch import nn
from torch.distributions import Uniform
import torchvision.transforms.functional as F

from scipy.ndimage import gaussian_filter


from utils.image import normalise_intensity


def randconv(image: torch.Tensor, in_channels: int, K: int, mix: bool, p: float) -> torch.Tensor:
    """
    Outputs the image or the random convolution applied on the image.

    Args:
        image (torch.Tensor): input image
        in_channels (int): number of input channels
        K (int): maximum kernel size of the random convolution
        mix (bool): whether to mix the image with the random convolution
        p (float): probability of applying the random convolution
    Returns:
        torch.Tensor: output image after applying the random convolution
    """

    p0 = torch.rand(1).item()
    if p0 < p:
        return image
    else:
        k = torch.randint(0, K+1, (1, )).item()
        random_convolution = nn.Conv3d(in_channels, in_channels, 2*k + 1, padding=k).to(image.device)
        init_factor = (1. / (3 * k * k)) if k != 0 else (1. / (3 * (k+1) * (k+1)))
        torch.nn.init.uniform_(random_convolution.weight,
                              0, init_factor)
        image_rc = random_convolution(image).to(image.device).detach()

        if mix:
            alpha = torch.rand(1,)
            return alpha * image + (1 - alpha) * image_rc
        else:
            return image_rc


def apply_randconv(image: torch.Tensor, mask: torch.Tensor, K: int, mix: bool, p: float, dim: int = 2,
                   random_convolution: torch.nn.modules.conv.Conv3d = None) -> torch.Tensor:
    """
    Outputs the image or the random convolution applied on the image.

    Args:
        image (torch.Tensor): input image
        K (int): maximum kernel size of the random convolution
    """
    # image = torch.tensor(image).unsqueeze(0)

    p0 = torch.rand(1).item()
    if p0 < p:
        image = image
    else:
        if not random_convolution:
            k = K
            if dim == 2:
                random_convolution = nn.Conv2d(in_channels=1, out_channels=1,
                                               kernel_size=1, padding=0).to(image.device)
            else:
                kernel_size = 2 * k + 1
                random_convolution = nn.Conv3d(in_channels=1, out_channels=1,
                                               kernel_size=kernel_size, padding=k).to(image.device)

            torch.nn.init.normal_(random_convolution.weight, 1. / 9 + (3 * k * k + torch.finfo(torch.float32).eps))
        else:
            random_convolution = random_convolution

        # TODO: ask whether detaching the tensor wouldn't affect training with randconvs
        image_rc = random_convolution(image).to(image.device).detach()

        # TODO: include for training with brains
        image_rc = -torch.abs(image_rc)
        negative_slope = 0.4
        image_rc = nn.LeakyReLU(negative_slope=negative_slope)(image_rc)

        # image_rc = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=[0.1, 2])(image_rc)
        # image_rc = nn.LeakyReLU(negative_slope=0.2)(random_convolution(image).to(image.device).detach())

        if mix:
            alpha = torch.rand(1, ).to(image.device)
            image = alpha * image * (1 - alpha) * image_rc
            # re-normalise the intensity between 0 and 1 after applying the random convolution kernel
            image = normalise_intensity(image, min_out=0.0, max_out=1.0, mode="minmax", clip=True)
        else:
            image = image_rc

    if dim == 2:
        # image[0, torch.where(mask[0][0] == 0)[0], torch.where(mask[0][0] == 0)[1]] = 0
        pass
    else:
        # image[:, :, torch.where(mask <= 0.0)[2], torch.where(mask <= 0.0)[3], torch.where(mask <= 0.0)[4]] = 0
        # image[mask <= 0.1] = 0
        image = image*mask
    image = normalise_intensity(torch.abs(image), min_out=0.0, max_out=1.0, mode="minmax", clip=True)

    return image, random_convolution


def apply_sin_act(image: torch.Tensor,
                  mask: torch.Tensor,
                  k: int,
                  dim: int = 3,
                  random_convolution: torch.nn.modules.conv.Conv3d = None):

    if not random_convolution:
        if dim == 2:
            random_convolution = nn.Conv2d(in_channels=1, out_channels=1,
                                           kernel_size=1, padding=0).to(image.device)
        else:
            kernel_size = 2 * k + 1
            random_convolution = nn.Conv3d(in_channels=1, out_channels=1,
                                           kernel_size=kernel_size, padding=k).to(image.device)

        torch.nn.init.normal_(random_convolution.weight, 1. / 9 + (3 * k * k + torch.finfo(torch.float32).eps))
    else:
        random_convolution = random_convolution

    image_rc = random_convolution(image).to(image.device).detach()

    a, b = torch.rand(1).item(), torch.rand(1).item()
    image_rc_act = torch.sin(5 * a * (image_rc - b * torch.pi))
    image_out = normalise_intensity(image_rc_act,
                                    min_out=0.0,
                                    max_out=1.0,
                                    mode="minmax",
                                    clip=True)

    image_out = image_out * mask

    return image_out, random_convolution
