# A Deep Metric for Multimodal Registration

This repository contains code for the A Deep Metric for Multimodal Registration framework presented in the paper ([DMMR 2016](https://arxiv.org/pdf/1609.05396.pdf))

## Installation
1. Clone this repository
2. In a fresh Python 3.7+ virtual environment, install dependencies by running:
    ```
    pip install -r <path_to_cloned_repository>/requirements.txt
    ```

## Configurations
We use [Hydra](https://hydra.cc/docs/intro) for structured configurations. 
Each directory in `conf/` is a *config group* which contains alternative configurations for that group. 
The final configuration is the *composition* of all *config groups*.
The default options for the groups are set in `conf/config.yaml`. 
To use a different configuration for a group, for example the loss function:
```
python loss=<hinge/bce/ce> ...
```

Any configuration in this structure can be conveniently over-written in CLI at runtime. For example, to change the regularisation weight at runtime:
```
python training.lr=<your_lr> ...
```

See [Hydra documentation](https://hydra.cc/docs/intro) for more details.

   
## File Structure
```
├── conf
│   ├── config.yaml           # Main config file
│   ├── data                  # Dataset specific config files
│   │   ├── brats.yaml
│   │   ├── camcan.yaml
│   │   └── ixi.yaml
│   ├── loss                  # Loss specific config files
│   │   ├── bce.yaml
│   │   ├── ce.yaml
│   │   └── hinge.yaml
│   ├── network               # Network specific config files
│   │   ├── dmmr_multiclass.yaml
│   │   ├── dmmr_sigmoid.yaml
│   │   └── dmmr_tanh.yaml
│   └── training              # Training/trainer related configuration
│       └── training.yaml
├── data
│   └── datamodules.py        # Datamodules for different datasets
├── compile_jit.py            # Utility for JIT compilation of DMMR model
├── inference.py              # Inference script
├── model
│   ├── dmmr.py               # DMMR Lightning module definition
│   ├── dmmr_utils.py         # DMMR utility functions
│   └── network.py            # DMMR network specification
├── train.py                  # Training script
└── utils                     # Handy utility functions

```

## Run Training
To train the default model, simply change the `.yaml` files in the `conf` folder and then run:
```
python train.py 
```
Training logs and outputs will be saved in `outputs/YYYY-MM-DD/HH-min-sec`.
On default settings, a checkpoint of the model will be saved at `{output_dir}/checkpoints/last.ckpt`
A copy of the configurations will be saved to `model_dir/.hydra` automatically.
As mentioned above, you can overwrite any configuration in CLI at runtime, or change the default values in `conf/`

## JIT Compilation

To compile the model to a TorchScript JIT model, run:
```
python compile_jit.py --ckpt_path <path_to_checkpoint> --model_name <name_of_model> --verbose
```

The model will then be saved under `outputs/dmmr_models/` with the specified name `<name_of_model>`.

## Run Inference
To run inference on a trained model, run:
```
python inference.py --source_path <path_to_source_image> --target_path <path_to_target_image> --model_path <path_to_jit_model> --angle_range <angle_range> (format: start_angle:end_angle:step) --translation_range <translation_range> (format start_translation:end_translation:step) --zero_percentage_threshold <threshold> --patch_size <patch_size> --axis <axis> (can be x, y, z or xyz)
```

This script will then plot the similarity curves with respect to the specified axis (both for translation and rotation)
for the specified source and target images given the trained model.

