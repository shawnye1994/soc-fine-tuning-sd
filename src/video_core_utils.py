"""
Utility functions for the SOC pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import json
import torchvision

import wandb
import pytorch_lightning as pl

from soc_pipeline_svd import SOCStableVideoDiffusionPipeline

def get_model(
    model_name,
    use_compile=True,
    bfloat_dtype=True,
    scheduler="edm_ancestral",
    use_for_training=True,
):
    """
    Get the SVD model based on the model name.
    """
    torch_dtype = torch.bfloat16 if bfloat_dtype else torch.float32 #torch.float16

    if model_name == "stable-video-diffusion":
        pipeline = SOCStableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch_dtype
                )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if scheduler == "euler_discrete":
        pass #which is the default scheduler
    elif scheduler == 'edm_ancestral':
        # used for the soc finetuning
        pipeline.set_edm_ancestral_scheduler()
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    if use_compile:
        print("Run torch compile")
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)

        pipeline.unet = torch.compile(
            pipeline.unet, mode="reduce-overhead", fullgraph=True
        )
        pipeline.vae.decode = torch.compile(
            pipeline.vae.decode, mode="reduce-overhead", fullgraph=True
        )

    if use_for_training:
        pipeline.vae.requires_grad_(False)
        pipeline.unet.requires_grad_(False)

    return pipeline



def torchvision_grid(images, nrow=8):
    """
    Make a grid of images using torchvision.utils.make_grid.
    """

    # convert PIL images to torch tensors
    images = [torchvision.transforms.ToTensor()(image) for image in images]
    images = torch.stack(images, dim=0)
    images = images * 255.0
    images = torch.clamp(images, min=0.0, max=255.0)

    grid = torchvision.utils.make_grid(images, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    grid = grid.unsqueeze(0)
    grid = grid.numpy().astype(np.uint8)
    return grid

def reinitialize_last_layer(layer, mean=0.0, std=0.01):
    if isinstance(layer, nn.Conv2d):
        init.normal_(layer.weight, mean=mean, std=std)
        # Optionally, set bias to a small constant (or zero)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)

class ConfigViewerCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        if not getattr(trainer, "logger", None):
            return
        config = pl_module.config
        
        if hasattr(config, "config_path"):
            artifact = wandb.Artifact("config-artifact", type="config")
            artifact.add_file(config.config_path)
            trainer.logger.experiment.log_artifact(artifact)