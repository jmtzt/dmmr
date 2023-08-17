import torch
from torch import nn as nn
from torchmetrics import HingeLoss

from model.network import DMMRNet, DMMRTanh


def get_loss_fn_dmmr(hparams):
    if hparams.loss.sim_loss == "hinge":
        return HingeLoss(task="binary")
    elif hparams.loss.sim_loss == "bce":
        return nn.BCELoss(weight=torch.tensor([hparams.loss.pos_weight]))


def get_network_dmmr(hparams):
    """Configure network"""
    if hparams.network.type == "dmmr_net_sigmoid":
        network = DMMRNet()
    elif hparams.network.type == "dmmr_net_tanh":
        network = DMMRTanh()
    else:
        raise ValueError(f"Network config ({hparams.network.name}) not recognised.")
    return network
