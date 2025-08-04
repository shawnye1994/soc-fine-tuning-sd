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
from diffusers.loaders import PeftAdapterMixin

import torch.nn as nn

# class _LoraWrapper(nn.Module):
#     """
#     A thin wrapper that forwards to `self.core` but
#       • enables a given adapter     when `adapter_name` is not None
#       • disables every adapter      when `adapter_name` is None
#     Nothing is copied, both views share the same parameter tensors.
#     Use for the initialization of unet_int and trainable unet
#     """
#     def __init__(self, core_unet, adapter_name: str | None):
#         super().__init__()
#         self.core = core_unet
#         self.adapter_name = adapter_name        # None → LoRA off

#     # ---- mandatory: forward pass ----------------------------------
#     def forward(self, *args, **kwargs):
#         if self.adapter_name is None:
#             self.core.set_adapter([])                   # LoRA OFF
#         else:
#             self.core.set_adapter([self.adapter_name])  # LoRA ON
#         return self.core(*args, **kwargs)

#     # ---- transparently expose every other attribute ---------------
#     def __getattr__(self, item):
#         return getattr(self.core, item)

def attach_peft_mixin(model):
    """
    Make an *already-instantiated* UNet gain the PEFT/LoRA API
    (add_adapter, set_adapter, enable_adapters, …) without touching
    the original source code.

    Returns the same object, now with the extra methods.
    """
    # 1.  Do nothing if it already has the mixin
    if isinstance(model, PeftAdapterMixin):
        return model

    # 2.  Create a new class that merges the current class + mixin
    NewCls = type(
        f"PeftWrapped{model.__class__.__name__}",
        (PeftAdapterMixin, model.__class__),   # MRO: mixin first
        {}
    )

    # 3.  Point the instance to the new class
    model.__class__ = NewCls

    # 4.  Run the mixin’s __init__ (it just sets a few attributes)
    PeftAdapterMixin.__init__(model)

    return model

def get_model(
    model_name,
    use_compile=True,
    bfloat_dtype=True,
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

    # add the peft functions to the svd unet
    pipeline.unet = attach_peft_mixin(pipeline.unet)
    
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