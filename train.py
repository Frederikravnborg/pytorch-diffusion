import torch
from data import DiffSet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
import wandb
import time
import warnings


def main():
    # Training hyperparameters
    diffusion_steps = 10
    dataset_choice = "CIFAR"
    max_epoch = 1
    batch_size = 128

    train_fraction = 2
    val_fraction = 2

    continue_training = False
    ckpt_path = '/Users/fredmac/Documents/DTU-FredMac/pytorch-diffusion/checkpoints/06.30-22.28.05/10_steps-epoch=00-loss=0.00.ckpt'

    wandb_name = f'{diffusion_steps}_steps'

    # Set the device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Loading parameters
    load_model = False
    load_version_num = 1

    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    if load_model:
        pass_version = load_version_num
        last_checkpoint = glob.glob(
            f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
        )[-1]

    # Create datasets and data loaders
    train_dataset = DiffSet(True, dataset_choice)
    val_dataset = DiffSet(False, dataset_choice)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, persistent_workers=True)

    # Create model and trainer
    # if load_model:
    #     model = DiffusionModel.load_from_checkpoint(last_checkpoint, in_size=train_dataset.size*train_dataset.size, t_range=diffusion_steps, img_depth=train_dataset.depth)
    # else:
    model = DiffusionModel(train_dataset.size*train_dataset.size, diffusion_steps, train_dataset.depth)


    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss',
        dirpath = f'checkpoints/{time.strftime("%m/%d-%H.%M.%S")}',
        filename = f'{diffusion_steps}_steps-' + '{epoch:02d}-{loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    wandb_logger = pl.loggers.WandbLogger(
        name=f'{time.strftime("%m/%d-%H.%M.%S")} - {wandb_name}',
        version=pass_version,
        project="Diffusion_Project",
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch, 
        log_every_n_steps=10, 
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        limit_train_batches=train_fraction,
        limit_val_batches=val_fraction,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=ckpt_path if continue_training else None)

if __name__ == '__main__':
    main()