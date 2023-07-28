import random
import os
import torch
import hydra
import lightning as pl

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from data.datamodules import BraTSDataModule, CamCANDataModule
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

    if cfg.data.name == 'brats':
        dm = BraTSDataModule(**cfg.data)
    elif cfg.data.name == 'camcan':
        dm = CamCANDataModule(**cfg.data)
    else:
        raise NotImplementedError(f'Unknown dataset: {cfg.data.name}')

    checkpoint_path = f'{model_dir}/checkpoints/'
    cfg['training']['checkpoint_path'] = checkpoint_path
    print('saving to ', checkpoint_path)

    if not cfg['debug']:
        ckpt_callback = ModelCheckpoint(save_last=True,
                                        dirpath=checkpoint_path,
                                        every_n_epochs=50,
                                        verbose=True
                                        )

        project = f'overfit_{cfg.data.name}_dmmr' if cfg['data']['overfit'] else f'complete_{cfg.data.name}_dmmr'
        tags = [f'{cfg.data.modality.lower()}_{cfg.network.type.lower()}_{cfg.loss.sim_loss.lower()}', 'delete']
        wandb_logger = WandbLogger(
            project=project,
            tags=tags,)
        wandb_logger.experiment.config["batch_size"] = cfg.data.training_batch_size

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[ckpt_callback],
            **cfg.training.trainer,
            auto_lr_find=True)
    else:
        trainer = pl.Trainer(**cfg.training.trainer, auto_lr_find=True)

    model = DeepMetricModel(cfg)

    if not cfg['debug']:
        wandb_logger.watch(model, log='all', log_freq=cfg.training.gradients_log_interval)

    trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()

# TODO: stick with previous implementation of patch labelling
# TODO: apply the transformations on the _step function on the fly
# TODO: use the randomized binary mask to select pos/neg patches/augmentation -> find a native pytorch fn for this
# TODO: profile the code to see what is taking so long
# TODO: add script to get curves for the trained dmmr model
# TODO: add script to compile dmmr model into a torchscript .pt