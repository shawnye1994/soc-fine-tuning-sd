import argparse
import yaml
import os
from pathlib import Path

import torch
from pytorch_lightning import seed_everything
import numpy as np
# Import your trainer and data module
from svd_trainers.am_trainer import AMTrainer
from video_datamodule import VideoDataModule
from types import SimpleNamespace
from PIL import Image

def export_to_gif(frames, output_path, fps=7):
    """
    Exports a list of PIL images to a GIF file.
    
    Args:
        frames (List[PIL.Image.Image]): List of PIL images to export as a GIF.
        output_path (str): Path to save the output GIF file.
        fps (int, optional): Frames per second. Controls the speed of the GIF. Defaults to 7.
    
    Returns:
        str: The path to the saved GIF file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Calculate duration in milliseconds between frames
    duration = int(1000 / fps)
    
    # Save as GIF
    frames[0].save(
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,  # 0 means loop forever
        optimize=True
    )
    return output_path


def load_model(config, lora_ckpt, device="cuda", use_base_model=False):
    print(f"Loading checkpoint: {lora_ckpt}")
    model = AMTrainer.load_from_checkpoint(lora_ckpt, config=config)
    model = model.eval()

    if use_base_model:
        # deactivate the lora
        model.unet.set_adapter([])

    pipeline = model.soc_pipeline.to(device)
    return pipeline

@torch.no_grad()
def evaluate_checkpoint(ckpt_path, out_dir, config, device="cuda", use_base_model=False):
    seed_everything(config.seed)
    # Load data
    datamodule = VideoDataModule(
            batch_size=1,
            num_workers=config.num_workers,
            train_video_path_json=config.train_video_path_json,
            val_video_path_json=config.val_video_path_json,
            target_vid_size=config.target_vid_size,
            vid_data_type=config.vid_data_type,
        )
    datamodule.setup("validate")
    val_dataloader = datamodule.val_dataloader()

    # load model
    pipeline = load_model(config, ckpt_path, device, use_base_model=use_base_model)

    for batch_idx, batch in enumerate(val_dataloader):
        gt_video, init_frame = batch
        frames = pipeline(
            init_frame,
            num_frames=config.target_vid_size[0],
            height=config.target_vid_size[1],
            width=config.target_vid_size[2],
            num_inference_steps=config.num_inference_steps,
            store_traj=False,
            store_noise=False,
            store_noise_pred=False,
            use_soc_scheduler=True,
            learn_offset=config.learn_offset,
            output_type="pil"
        )[0]

        # export to gif
        vid_path = export_to_gif(frames, os.path.join(out_dir, f"gen_vid_{batch_idx}.gif"))
        print(f"Generated video saved at: {vid_path}")

        # convert the gt_video from pytorch tensors to a list of PIL image
        gt_video = gt_video*255
        gt_video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for frame in gt_video[0]]
        gt_vid_path = export_to_gif(gt_video, os.path.join(out_dir, f"gt_vid_{batch_idx}.gif"))
        print(f"GT video saved at: {gt_vid_path}")


def load_config_from_checkpoint(ckpt_path):
    """Extract configuration from checkpoint file"""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Option 1: Use the embedded YAML content
    if 'config_yaml' in checkpoint:
        print("Loading config from embedded YAML in checkpoint")
        config_dict = yaml.safe_load(checkpoint['config_yaml'])
        return SimpleNamespace(**config_dict)
    
    # Option 2: Use the stored config dictionary
    elif 'config_dict' in checkpoint:
        print("Loading config from stored dictionary in checkpoint")
        return SimpleNamespace(**checkpoint['config_dict'])
    
    # No embedded config found
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to checkpoint .ckpt file, or 'base_model' for base model")
    parser.add_argument("--use_base_model", action="store_true", help="Use base model for evaluation")
    parser.add_argument("--config", type=str, required=False, help="Path to config .yaml file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output file")
    args = parser.parse_args()

    # Check if output_dir exists, if not, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # First try to load config from checkpoint
    config = load_config_from_checkpoint(args.lora_ckpt)
    print('Config loaded from checkpoint:')
    print(config)
    
    # If not found or provided explicitly, load from file
    if config is None and args.config:
        print(f"Loading config from file: {args.config}")
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)
    elif config is None:
        raise ValueError("No config found in checkpoint and no config file provided")

    # Run evaluation and get output
    evaluate_checkpoint(args.lora_ckpt, args.output_dir, config, device="cuda", use_base_model=args.use_base_model)