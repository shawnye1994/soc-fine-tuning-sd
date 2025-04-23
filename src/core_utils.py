"""
Utility functions for the SOC pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import json
from diffusers import DDIMScheduler
import torchvision

import wandb
import pytorch_lightning as pl

# from fkd_pipeline_sdxl import FKDStableDiffusionXL
from soc_pipeline_sd import SOCStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def get_model(
    model_name,
    use_compile=True,
    bfloat_dtype=True,
    scheduler="ddim",
    use_for_training=True,
):
    """
    Get the FKD-supported model based on the model name.
    """
    torch_dtype = torch.bfloat16 if bfloat_dtype else torch.float32 #torch.float16

    if model_name == "stable-diffusion-xl":
        pipeline = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype
        )
    elif model_name == "stable-diffusion-v1-5":
        pipeline = SOCStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype
        )
    elif model_name == "stable-diffusion-v1-4":
        pipeline = SOCStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype
        )
    elif model_name == "stable-diffusion-2-1":
        pipeline = SOCStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch_dtype
        )
    elif model_name == "dpo-sdxl-text2image-v1":
        pipeline = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        unet_id = "mhdang/dpo-sdxl-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipeline.unet = unet
    elif model_name == "dpo-sd1.5-text2image-v1":
        pipeline = SOCStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # load finetuned model
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipeline.unet = unet
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

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
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(False)

    return pipeline


# load prompt data
def load_data(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data


def generate_split_indices(data, prompt_splits, split_idx, seed):
    # Generate split indices
    indices = np.arange(len(data))
    split_indices = np.array_split(indices, prompt_splits)
    return split_indices[split_idx]


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