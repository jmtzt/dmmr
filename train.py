import random
import os
import torch
import hydra
import lightning as pl

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig

from data.datamodules import CamCANDataModule, IXIDataModule
from model.dmmr import DeepMetricModel

seed = 1337  # for reproducibility
random.seed(seed)
torch.manual_seed(seed)

os.environ["DISPLAY"] = "localhost:10.0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['OC_CAUSE'] = '1'


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # set via CLI hydra.run.dir
    model_dir = os.getcwd()

    torch.set_float32_matmul_precision('medium')

    if cfg.data.name == 'camcan':
        dm = CamCANDataModule(**cfg.data)
    elif cfg.data.name == 'ixi':
        dm = IXIDataModule(**cfg.data)
    else:
        raise NotImplementedError(f'Unknown dataset: {cfg.data.name}')

    checkpoint_path = f'{model_dir}/checkpoints/'
    cfg['training']['checkpoint_path'] = checkpoint_path
    print('saving to ', checkpoint_path)

    if not cfg['debug']:
        ckpt_callback = ModelCheckpoint(save_last=True,
                                        dirpath=checkpoint_path,
                                        every_n_epochs=20,
                                        verbose=True
                                        )

        project = f'overfit_{cfg.data.name}_dmmr' if cfg['data']['overfit'] else f'complete_{cfg.data.name}_dmmr'
        tags = [f'{cfg.data.modality.lower()}_{cfg.network.type.lower()}_{cfg.loss.sim_loss.lower()}',]
        wandb_logger = WandbLogger(
            project=project,
            tags=tags,)
        wandb_logger.experiment.config["batch_size"] = cfg.data.training_batch_size

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[ckpt_callback,
                       EarlyStopping(monitor="val_loss", mode="min", patience=3)],
            **cfg.training.trainer,
            auto_lr_find=True)
    else:
        trainer = pl.Trainer(**cfg.training.trainer, auto_lr_find=True)

    model = DeepMetricModel(cfg)

    if not cfg['debug']:
        wandb_logger.watch(model, log='all', log_freq=cfg.training.gradients_log_interval)

    # trainer.tune(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
