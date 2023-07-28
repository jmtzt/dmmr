import torch
from omegaconf import OmegaConf, open_dict
import torchio as tio
import torch.nn as nn
from lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import SGD, Adam

from model.dmmr_utils import get_loss_fn_dmmr, get_network_dmmr


class DeepMetricModel(LightningModule):
    def __init__(self, hparams):
        super(DeepMetricModel, self).__init__()
        OmegaConf.set_struct(hparams, True)
        with open_dict(hparams):
            hparams.lr = hparams.training.lr
        self.save_hyperparameters(hparams)

        self.network = get_network_dmmr(self.hparams)
        self.loss_fn = get_loss_fn_dmmr(self.hparams)
        self.all_labels = []
        # self.hparams['lr'] = self.hparams.training.lr
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def configure_optimizers(self):
        if self.hparams.training.optimizer == 'sgd':
            optimizer = SGD(self.parameters(),
                            lr=self.hparams.lr,
                            momentum=self.hparams.training.momentum,
                            weight_decay=self.hparams.training.weight_decay)
        elif self.hparams.training.optimizer == 'adam':
            optimizer = Adam(self.parameters(),
                             lr=self.hparams.lr,
                             weight_decay=self.hparams.training.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {self.hparams.training.optimizer} not implemented')

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.training.lr_decay_step,
                                                    gamma=0.1,
                                                    last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #                                                 max_lr=self.hparams.training.lr * 10,
        #                                                 total_steps=20)

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
        }

    def forward(self, tar, src):
        net_out = self.network(tar, src)
        return net_out

    def _step(self, batch):
        """ Forward pass inference + compute loss """
        mod1 = batch['mod1'][tio.DATA].float()
        mod2 = batch['mod2'][tio.DATA].float()
        labels = batch['label'].float()

        out = self.forward(mod1, mod2)

        losses = self.loss_fn(out, labels)

        return losses, out, labels

    @staticmethod
    def compute_metrics(preds, labels):
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
        auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())

        return acc, f1, auc

    def training_step(self, batch, batch_idx):
        train_loss, out, labels = self._step(batch)

        if self.hparams.network.type == 'dmmr_net':
            preds = out > 0.5
        elif self.hparams.network.type == 'dmmr_net_tanh' and self.hparams.loss.sim_loss == 'hinge':
            preds = out > 0.0
            labels[labels == 0] = -1  # convert positive label 0 to -1
            labels[labels == -1] = False  # Positive is False in this case
            labels[labels == 1] = True  # Negative is True in this case

        acc, f1, auc = self.compute_metrics(preds, labels)

        metrics_dict = {'loss': train_loss, 'train_acc': acc, 'train_f1': f1, 'train_auc': auc}

        for k, metric in metrics_dict.items():
            self.log(f'{k}', metric, batch_size=self.hparams.data.training_batch_size)

        return metrics_dict

    def validation_step(self, batch, batch_idx):
        val_loss, out, labels = self._step(batch)

        if self.hparams.loss.sim_loss != 'hinge':
            preds = torch.sigmoid(out) > 0.5
        elif self.hparams.network.type == 'dmmr_net_tanh' and self.hparams.loss.sim_loss == 'hinge':
            preds = out > 0.0
            labels[labels == 0] = -1  # convert positive label 0 to -1
            labels[labels == -1] = False  # Positive is False in this case
            labels[labels == 1] = True  # Negative is True in this case
        else:
            preds = out > 0.5
        acc, f1, auc = self.compute_metrics(preds, labels)

        metrics_dict = {'val_loss': val_loss, 'val_acc': acc, 'val_f1': f1, 'val_auc': auc}

        for k, metric in metrics_dict.items():
            self.log(f'{k}', metric, batch_size=self.hparams.data.validation_batch_size)

        return metrics_dict
