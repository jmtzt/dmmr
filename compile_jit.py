import argparse
import os

from omegaconf import OmegaConf

from model.dmmr import DeepMetricModel


def check_weights_frozen(model, verbose=False):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            pass
        else:
            if verbose:
                print(f'Parameter {name} is trainable.')
            raise ValueError(f'Parameter {name} is trainable.')


def main(args):
    ckpt_path = args.ckpt_path

    model = DeepMetricModel.load_from_checkpoint(ckpt_path)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if args.verbose:
        print(model)

    check_weights_frozen(model, verbose=args.verbose)

    # Load the config.yaml file from the checkpoint folder
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    config_file = os.path.join(parent_dir, ".hydra", "config.yaml")
    config = OmegaConf.load(config_file)

    # Get the required parameters from the config
    data_name = config.data.name
    modality = config.data.modality
    network_type = config.network.type
    loss_sim_loss = config.loss.sim_loss
    training_lr = config.training.lr
    max_epochs = config.training.trainer.max_epochs
    if not args.model_name:
        model_name = f'{data_name}_{modality}_{network_type}_{loss_sim_loss}_lr{training_lr}_epochs{max_epochs}.pt'
    else:
        model_name = args.model_name

    model.to_torchscript(f'outputs/dmmr_models/{model_name}')
    if args.verbose:
        print(f'Compiled model saved to outputs/dmmr_models/{model_name}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and compile DeepMetricModel')
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint')
    parser.add_argument('--model_name', type=str, help='Name of the compiled model')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()

    main(args)
