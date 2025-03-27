import argparse
import yaml

import torch
from pytorch_lightning import seed_everything

# Import your trainer and data module
from src.am_trainer import AMTrainer
from src.prompt_datamodule import PromptDataModule
from src.metrics import (
    do_image_reward,
    do_clip_score,
    do_human_preference_score,
    do_dreamsim_diversity,
)
from types import SimpleNamespace

@torch.no_grad()
def evaluate_checkpoint(ckpt_path, config, device="cuda", eta=1.0, num_samples_per_prompt=5):
    seed_everything(config.seed)

    # Load data
    datamodule = PromptDataModule(
        batch_size=config.batch_size,
        training_prompt_path=config.training_prompt_path,
        validation_prompt_path=config.validation_prompt_path,
    )
    datamodule.setup("validate")
    val_dataloader = datamodule.val_dataloader()

    # Load and prepare model
    if ckpt_path == "base_model":
        print("Evaluating base model (no checkpoint)")
        # Create trainer with default (untrained) model
        model = AMTrainer(config=config)
        model.eval().to(device)
    else:
        print(f"Loading checkpoint: {ckpt_path}")
        model = AMTrainer.load_from_checkpoint(ckpt_path, config=config)
        model.eval().to(device)

    if hasattr(model, 'soc_pipeline'):
        model.soc_pipeline.to(device)

    # Lists to store per-prompt averages for existing metrics
    prompt_avg_ir = []
    prompt_avg_cs = []
    prompt_avg_hp = []
    
    # List to store DreamSim variance per prompt
    prompt_avg_ds = []

    for batch_idx, batch in enumerate(val_dataloader):
        prompt_texts = batch["text"]
        batch_size = len(prompt_texts)

        # Repeat each prompt num_samples_per_prompt times
        repeated_prompts = []
        for p in prompt_texts:
            repeated_prompts.extend([p] * num_samples_per_prompt)

        # Generate images in one call
        result = model.soc_pipeline(
            repeated_prompts,
            num_inference_steps=config.num_inference_steps,
            eta=eta,
            output_type="pil",
            device=device,
            store_traj=False,
            use_custom_scheduler=True,
            learn_offset=config.learn_offset,
        )
        images = result.images

        # Compute metrics for all images
        ir_vals = do_image_reward(images=images, prompts=repeated_prompts)
        cs_vals = do_clip_score(images=images, prompts=repeated_prompts)
        hp_vals = do_human_preference_score(images=images, prompts=repeated_prompts)

        # Group by prompt, compute per-prompt averages
        for i in range(batch_size):
            start = i * num_samples_per_prompt
            end = (i + 1) * num_samples_per_prompt

            # Slice images and metrics for this prompt
            prompt_images = images[start:end]
            ir_slice = ir_vals[start:end]
            cs_slice = cs_vals[start:end]
            hp_slice = hp_vals[start:end]

            ir_mean = float(torch.tensor(ir_slice).mean())
            cs_mean = float(torch.tensor(cs_slice).mean())
            hp_mean = float(torch.tensor(hp_slice).mean())

            # Store IR, CS, HP
            prompt_avg_ir.append(ir_mean)
            prompt_avg_cs.append(cs_mean)
            prompt_avg_hp.append(hp_mean)

            # Now compute DreamSim diversity for these images
            # The function do_dreamsim_diversity should:
            #   1) Preprocess images
            #   2) Embed them with dreamsim_model
            #   3) Return (variance, variance_variance)
            ds_var, _ = do_dreamsim_diversity(prompt_images, device=device)
            prompt_avg_ds.append(ds_var)

    # Compute overall means (across prompt-level means)
    if len(prompt_avg_ir) > 0:
        ir_tensor = torch.tensor(prompt_avg_ir)
        cs_tensor = torch.tensor(prompt_avg_cs)
        hp_tensor = torch.tensor(prompt_avg_hp)
        ds_tensor = torch.tensor(prompt_avg_ds)

        # Means
        avg_ir = ir_tensor.mean().item()
        avg_cs = cs_tensor.mean().item()
        avg_hp = hp_tensor.mean().item()
        avg_ds = ds_tensor.mean().item()

        # Standard deviations
        std_ir = ir_tensor.std().item()
        std_cs = cs_tensor.std().item()
        std_hp = hp_tensor.std().item()
        std_ds = ds_tensor.std().item()

        # Standard errors (std / sqrt(N_prompts))
        n_prompts = len(prompt_avg_ir)
        se_ir = std_ir / (n_prompts**0.5)
        se_cs = std_cs / (n_prompts**0.5)
        se_hp = std_hp / (n_prompts**0.5)
        se_ds = std_ds / (n_prompts**0.5)
    else:
        avg_ir = avg_cs = avg_hp = avg_ds = 0.0
        se_ir = se_cs = se_hp = se_ds = 0.0

    print(f"Checkpoint: {ckpt_path}")
    print(f"Avg ImageReward : {avg_ir:.4f} ± {se_ir:.4f}")
    print(f"Avg CLIP-Score  : {avg_cs:.4f} ± {se_cs:.4f}")
    print(f"Avg HPS         : {avg_hp:.4f} ± {se_hp:.4f}")
    print(f"Avg DreamSim Var: {avg_ds:.4f} ± {se_ds:.4f}")

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
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file, or 'base_model' for base model")
    parser.add_argument("--eta", type=float, default=1.0, help="Sampling noise multiplier")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samples generated per prompt")
    parser.add_argument("--config", type=str, required=False, help="Path to config .yaml file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Load config appropriately
    if args.ckpt == "base_model":
        # Base model requires explicit config
        if not args.config:
            raise ValueError("Config file must be provided when evaluating base model")
        print(f"Loading config from file: {args.config}")
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)
    else:
        # First try to load config from checkpoint
        config = load_config_from_checkpoint(args.ckpt)
        
        # If not found or provided explicitly, load from file
        if config is None and args.config:
            print(f"Loading config from file: {args.config}")
            with open(args.config, "r") as f:
                config_dict = yaml.safe_load(f)
            config = SimpleNamespace(**config_dict)
        elif config is None:
            raise ValueError("No config found in checkpoint and no config file provided")

    evaluate_checkpoint(args.ckpt, 
                        config, 
                        device=args.device, 
                        eta=args.eta, 
                        num_samples_per_prompt=args.num_samples_per_prompt
                        )