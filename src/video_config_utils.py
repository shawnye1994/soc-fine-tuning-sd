import yaml
import argparse
from pathlib import Path
import os

def load_config(config_path=None, override_args=None):
    """
    Load configuration from YAML file with optional command-line overrides.
    
    Args:
        config_path: Path to YAML config file
        override_args: Command line arguments that override YAML settings
    
    Returns:
        config: Namespace containing all configuration values
    """
    parser = argparse.ArgumentParser(description="Adjoint Matching for Stable Diffusion")
    parser.add_argument("--config", type=str, default=config_path, help="Path to config YAML file")
    
    # Allow overriding specific config values from command line
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--precision", type=str, help="Training precision (32, 16, bf16)")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--save_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--verbose", type=str, help="Enable verbose logging")
    parser.add_argument("--use_buffer", type=bool, help="Use buffer")
    
    args = parser.parse_args(override_args)
    
    # If config path provided through command line, use that
    config_path = args.config if args.config else config_path
    
    if not config_path:
        raise ValueError("No config file specified. Use --config or provide config_path.")
    
    # Load YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to Namespace
    config = argparse.Namespace(**config_dict)
    
    # Override with any command line arguments that were explicitly provided
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            setattr(config, key, value)
    
    # Ensure paths are Path objects
    if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint:
        config.resume_from_checkpoint = Path(config.resume_from_checkpoint)
    
    # Store the config file path for reference
    config.config_path = config_path

    # Only use guidance in adjoint computation and control computation if guidance scale is not 1.0
    # and the respective flags are set
    # defualt guidace in adjoint computation is false
    config.cfg_adjoint = config.guidance_in_adjoint_computation
    config.cfg_control = config.guidance_in_control_computation
    
    return config