import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from prompt_datamodule import PromptDataModule, BufferPromptDataModule
from trainers.am_trainer import AMTrainer
from trainers.buffer_am_trainer import BufferAMTrainer
from config_utils import load_config
from core_utils import ConfigViewerCallback


def main():
    # Parse command line arguments
    config = load_config()

    run_name = f"rm{config.reward_multiplier}_smooth{config.smooth_gradients}_buffer{config.use_buffer}"
     
    print(f"Run name: {run_name}")
    
    if config.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = Path(config.resume_from_checkpoint)
        assert config.resume_from_checkpoint.exists()
    
    if config.use_tf32:
        # Enable TF32 globally
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing logger")
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=run_name,  # This will be used for the checkpoint folder
        save_dir=config.save_dir,
        dir=config.save_dir,
        offline=False,
        config=config,
    )

    # Define explicit checkpoint directory with the run name
    checkpoint_dir = Path(config.save_dir) / config.wandb_project / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoints to: {checkpoint_dir}")

    # Best checkpoint based on new_reward with epoch in filename
    best_reward_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),  # Explicitly set the checkpoint directory
        save_last=False,          # Don't save last in this callback
        save_top_k=1,             # Save only the best model
        every_n_epochs=config.checkpoint_every_n_epochs,
        filename="best_{val_reward_mean:.4f}_{epoch}",  # Include epoch and reward
        monitor="val_reward_mean",     # Monitor new_reward instead of val_loss
        mode="max",               # Higher reward is better
        verbose=True,             # Print when a better model is saved
    )
 
    # Last checkpoint with epoch in filename
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),  # Explicitly set the checkpoint directory
        save_last=True,           # Save the last checkpoint
        save_top_k=0,             # Don't save any top-k models
        every_n_epochs=config.checkpoint_every_n_epochs,
        filename="last_{epoch}",  
        verbose=True,
    )

    pl.seed_everything(config.seed)

    if config.use_buffer:
        datamodule = BufferPromptDataModule(
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            training_prompt_path=config.training_prompt_path,
            validation_prompt_path=config.validation_prompt_path,
        )
    else:
        datamodule = PromptDataModule(
            batch_size=config.batch_size,
            training_prompt_path=config.training_prompt_path,
            validation_prompt_path=config.validation_prompt_path,
        )
    config.iterations_per_epoch = len(datamodule.train_dataloader())

    if config.resume_from_checkpoint is not None:
        if config.use_buffer:
            am_trainer = BufferAMTrainer.load_from_checkpoint(config.resume_from_checkpoint, config=config)
        else:
            am_trainer = AMTrainer.load_from_checkpoint(config.resume_from_checkpoint, config=config)
    else:
        if config.use_buffer:
            am_trainer = BufferAMTrainer(config=config)
        else: 
            am_trainer = AMTrainer(config=config)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        logger=wandb_logger,
        devices=torch.cuda.device_count(),
        accelerator="auto",
        strategy="ddp",  # Use DistributedDataParallel
        callbacks=[
            best_reward_callback,         # Track best by reward values
            last_checkpoint_callback,     # Save last checkpoint
            lr_monitor, 
            ConfigViewerCallback()
        ],
        deterministic=False,
        gradient_clip_val=config.gradient_clip,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=1,
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.accum_grad_steps,
        max_epochs=config.max_epochs,
        precision=config.precision,
        log_every_n_steps=1,
    )
    print(f"val_check_interval: {trainer.val_check_interval}")
    print(f"accumulate_grad_batches: {trainer.accumulate_grad_batches}")
    print(f"limit_val_batches: {trainer.limit_val_batches}")
    print(f"check_val_every_n_epoch: {trainer.check_val_every_n_epoch}")
    trainer.fit(am_trainer, datamodule)


if __name__ == "__main__":
    main()
