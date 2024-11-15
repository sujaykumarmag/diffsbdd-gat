from sklearn.model_selection import KFold
import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml
import numpy as np
import os
import csv

from lightning_modules import LigandPocketDDPM


import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    pl.seed_everything(seed)

set_seed(42)



from lightning_modules import LigandPocketDDPM

class LossLoggerCallback(pl.Callback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.fields = ['fold', 'epoch', 'train_loss', 'val_loss']
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def on_validation_epoch_end(self, trainer, pl_module):
        fold = pl_module.fold_idx
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('loss/train', None)
        val_loss = trainer.callback_metrics.get('loss/val', None)
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

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



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--n_splits', type=int, default=10, help="Number of folds for cross-validation")
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

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

    dataset = args.dataset
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    loss_logger = LossLoggerCallback(output_path=Path(out_dir, "cross_val_results.csv"))

    fold_idx = 0
    for train_idx, val_idx in kf.split(dataset):
        print(f"Starting fold {fold_idx + 1}/{args.n_splits}")

        # Initialize model for each fold 
        # (Explanation : https://stackoverflow.com/questions/73118350/how-to-initialize-the-parameter-in-the-cross-validation-method-and-get-the-final)
        pl_module = LigandPocketDDPM(
            outdir=out_dir,
            dataset=dataset,
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
            node_histogram=histogram,
            pocket_representation=args.pocket_representation,
            virtual_nodes=0
        )

        # Split data for the current fold
        pl_module.train_idx = train_idx
        pl_module.val_idx = val_idx
        pl_module.fold_idx = fold_idx

        logger = pl.loggers.WandbLogger(
            save_dir=args.logdir,
            project='ligand-pocket-ddpm',
            group=args.wandb_params.group,
            name=f"{args.run_name}_fold_{fold_idx}",
            id=f"{args.run_name}_fold_{fold_idx}",
            resume='must' if args.resume is not None else False,
            entity=args.wandb_params.entity,
            mode=args.wandb_params.mode,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=Path(out_dir, f'checkpoints_fold_{fold_idx}'),
            filename="best-model-epoch={epoch:02d}",
            monitor="loss/val",
            save_top_k=1,
            save_last=True,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_callback, loss_logger],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='gpu' if args.gpus > 0 else 'cpu',
            devices=args.gpus if args.gpus > 0 else 1,
            strategy='ddp' if args.gpus > 1 else "auto"
        )

        # Fit model on current fold
        trainer.fit(model=pl_module, ckpt_path=ckpt_path)
        fold_idx += 1

    print("Cross-validation complete.")
