import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml
import numpy as np
import os

from lightning_modules import LigandPocketDDPM


import torch
import numpy as np
import random

def set_seed(seed: int):
    # Set random seed for Python
    random.seed(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for CUDA
    torch.cuda.manual_seed_all(seed)  # for all GPUs
    torch.backends.cudnn.deterministic = True  # ensures deterministic results
    torch.backends.cudnn.benchmark = False  # disables auto-tuning (to ensure deterministic results)
    
    # If you are using any other libraries that generate random numbers, set their seed too
    # For example, if you are using PyTorch Lightning, you can set the seed for it as well.
    pl.seed_everything(seed)

# Set a specific seed (for example, 42)
set_seed(42)



def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config
import pytorch_lightning as pl
import csv
class LossLoggerCallback(pl.Callback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.fields = ['epoch', 'train_loss', 'val_loss', 'sampling_results']

        # Initialize CSV file with headers
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('loss/train', None)
        val_loss = trainer.callback_metrics.get('loss/val', None)
        sampling_results = trainer.callback_metrics.get('sampling_results', None)

        # Handle missing or None values
        sampling_results = str(sampling_results) if sampling_results is not None else 'N/A'

        # Write to CSV
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss.item() if train_loss else 'N/A',
                'val_loss': val_loss.item() if val_loss else 'N/A',
                'sampling_results': sampling_results
            })

# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)


    out_dir = Path(args.logdir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        alg=args.alg,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=0
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='ligand-pocket-ddpm',
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )
    loss_logger = LossLoggerCallback(output_path=Path(out_dir, "results.csv"))
    # loss_logger = LossLoggerCallback(output_path=Path(out_dir, "results.csv"))

    # trainer = pl.Trainer(
    #     max_epochs=args.n_epochs,
    #     logger=logger,
    #     callbacks=[checkpoint_callback, loss_logger],
    #     enable_progress_bar=args.enable_progress_bar,
    #     num_sanity_val_steps=args.num_sanity_val_steps,
    #     accelerator='cpu', devices=args.gpus,
    #     strategy=('ddp' if args.gpus > 1 else None)
    # )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback,loss_logger],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        strategy='ddp' if args.gpus > 1 else "auto"
    )



    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
