import torch
import torch.distributed as dist
from diffusers.models import UNet2DConditionModel

import pytorch_lightning as pl
from core_utils import get_model, reinitialize_last_layer

from soc_pipeline_sd import retrieve_timesteps, latent_to_decode
from metrics import reward_function
from SOC_DDIM_scheduler import SOCDDIMScheduler

class SOCTrainer(pl.LightningModule):
    def __init__(self, config):
        """Initialize the base Stochastic Optimal Control trainer"""
        super().__init__()
        self.config = config
        
        # Initialize pipeline, model and scheduler
        self._initialize_pipeline()
        self._initialize_unet()
        self._initialize_scheduler()
        
        # Define basic functions
        self._setup_utility_functions()
        
        # Initialize tracking variables
        self.eval_step_outputs = dict(val=[], test=[])
        self.global_batch_step = 0

    def _initialize_pipeline(self):
        """Setup the Stable Diffusion pipeline"""
        self.soc_pipeline = get_model(
            self.config.model_name,
            use_compile=False,
            bfloat_dtype=(self.config.precision == 'bf16'),
        )
        
        # Setup appropriate data type
        if self.config.precision == 'bf16':
            self.torch_dtype = torch.bfloat16
        elif self.config.precision == '32':
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{self.config.precision}'. Allowed values are: 'bf16', '32'.")

    def _initialize_unet(self):
        """Initialize and setup UNet model"""
        # Initialize UNet model
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=self.torch_dtype,
        )
        self.unet.requires_grad_(True)

        # Store initial weights
        self.soc_pipeline.unet_init = self.soc_pipeline.unet
        self.unet_init = self.soc_pipeline.unet_init
        self.unet_init.requires_grad_(False)

        # Set the pipeline to use our trainable UNet
        self.soc_pipeline.unet = self.unet
        
        # Handle last layer initialization if needed
        if self.config.learn_offset:
            self._reinitialize_last_layer()

    def _initialize_scheduler(self):
        """Setup the diffusion scheduler with custom parameters"""
        mutable_dict = dict(self.soc_pipeline.scheduler.config)
        mutable_dict['beta_start'] = self.config.beta_start
        mutable_dict['beta_end'] = self.config.beta_end
        self.soc_pipeline.scheduler = SOCDDIMScheduler.from_config(mutable_dict)
        self.soc_pipeline.scheduler.config.timestep_spacing = 'trailing'
        
        # Initialize timesteps
        self._setup_timesteps()

    def _setup_timesteps(self):
        """Setup diffusion timesteps"""
        timesteps, num_inference_steps = retrieve_timesteps(
            self.soc_pipeline.scheduler,
            self.config.num_inference_steps,
            device=self.device,
            timesteps=None,
            sigmas=None,
        )
        
        self.time_steps = timesteps.to(self.torch_dtype)
        
        # Adjust final alpha as needed
        self.soc_pipeline.scheduler.final_alpha_cumprod = (
            1 + self.soc_pipeline.scheduler.alphas_cumprod[timesteps[-1]]
        ) / 2

    def _setup_utility_functions(self):
        """Define utility functions for the pipeline"""
        self.latent_to_decode_fn = lambda x: latent_to_decode(
            model=self.soc_pipeline, output_type="pt", latents=x,
        )
        
        self.reward_fn = lambda x, prompt: reward_function(
            x, prompt, 
            model=self.soc_pipeline, 
            guidance_reward_fn=self.config.guidance_reward_fn, 
            verbose=self.config.verbose
        )

    def _reinitialize_last_layer(self):
        """Reinitialize the last layer for offset learning"""
        original_norm = self.unet.conv_out.weight.norm()
        print("Original weight norm:", original_norm.item())
        reinitialize_last_layer(self.unet.conv_out)
        new_norm = self.unet.conv_out.weight.norm()
        print("New weight norm:", new_norm.item())

    def on_fit_start(self):
        """Setup before training starts"""
        self.soc_pipeline.to(self.device)
        self.time_steps = self.time_steps.to(self.device)
        self.unet.to(self.device).train()

    def get_prompt_embeds(self, prompt):
        """Get CLIP embeddings for prompts"""
        num_images_per_prompt = 1
        prompt_embeds, negative_prompt_embeds = self.soc_pipeline.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            True,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        return prompt_embeds, negative_prompt_embeds

    def switch_to_train(self):
        """Set models to training mode"""
        self.unet.train()
        self.unet_init.train()

    def switch_to_eval(self):
        """Set models to evaluation mode"""
        self.unet.eval()
        self.unet_init.eval()

    def gather_across_gpus(self, tensor):
        """Concatenate a tensor across all GPUs"""
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def average_across_gpus(self, tensor):
        """Average a tensor across all GPUs"""
        world_size = dist.get_world_size()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
        return tensor

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler"""
        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.unet.parameters(), 
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2), 
                weight_decay=0.0
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

        # Setup scheduler
        if self.config.scheduler == "stepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.accum_grad_steps, gamma=0.99
            )
        elif self.config.scheduler == "linear_warmup": 
            def lr_lambda(current_step):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return 1.0

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")

        return [optimizer], [scheduler]
    
    def control_from_noise_preds(
        self,
        noise_pred,
        noise_pred_init,
        t_eval,
        x_eval,
        batch_size,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
    ):
        """
        Calculate the control vector field from given noise predictions under the current model.
        and the reference model (uncontrolled).
        
        This function computes the control vector field component of the diffusion process
        by taking the difference between controlled and uncontrolled denoising steps
        and dividing it by the timestep standard deviation.
        
        Args:
            noise_pred: Noise prediction from the trained model
            noise_pred_init: Noise prediction from the base/initial model (uncontrolled)
            t_eval: Timestep(s) at which to evaluate
            x_eval: Current latent representations
            batch_size: Number of samples in the batch
            eta: Controls the stochasticity of the process (0=deterministic, 1=full stochasticity)
            use_clipped_model_output: Whether to clip model outputs
            generator: Random number generator for reproducibility
            
        Returns:
            control_times_sqrt_dt: Control vector field multiplied by sqrt(delta t)
            
        Mathematical relationship:
        - Controlled update: prev_sample = x_eval + δ * (b(x_eval,t) + σ(t) * u(x_eval,t)) + √δ * σ(t) * ε
        - Uncontrolled update: prev_sample_init = x_eval + δ * b(x_eval,t) + √δ * σ(t) * ε
        - Difference (prev_sample_diff): δ * σ(t) * u(x_eval,t)
        - control_times_sqrt_dt: √δ * u(x_eval,t) = prev_sample_diff / σ(t)
        """
        #Get difference between controlled and uncontrolled denoising steps
        _, _, prev_sample_diff, std_dev_t, _ = self.soc_pipeline.scheduler.step(
            noise_pred,
            noise_pred_init,
            t_eval.unsqueeze(0).repeat(batch_size, 1).to(torch.int),
            x_eval,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=torch.zeros_like(noise_pred, device=noise_pred.device),
            return_dict=False,
        )
        # Extract the control signal by dividing the difference by standard deviation
        control_times_sqrt_dt = prev_sample_diff / std_dev_t
        return control_times_sqrt_dt
    
    def evaluate_controls(
        self, 
        batch_size, 
        num_timesteps_to_load, 
        prompt_embeds, 
        negative_prompt_embeds, 
        all_x_t, 
        all_t, 
        indices_t,
        noise_pred_init=None,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
        **kwargs,
    ):
        """
        Calculate the control vector field from the latents, times, and prompt embeddings,
        for a batch of samples across multiple timesteps.
        
        This function computes the control vector field component of the diffusion process
        by taking the difference between controlled and uncontrolled denoising steps
        and dividing it by the timestep standard deviation.
        
        Args:
            batch_size: Number of samples in the batch
            num_timesteps_to_load: Number of timesteps to process
            prompt_embeds: Embedded text prompts
            negative_prompt_embeds: Embedded negative prompts (used for CFG)
            all_x_t: Latent representations at all timesteps [batch_size, num_timesteps, 4, 64, 64]
            all_t: All timestep values
            indices_t: Indices of timesteps to evaluate
            noise_pred_init: Pre-computed noise predictions from reference model (optional)
            eta: Controls the stochasticity (0=deterministic, 1=full stochasticity)
            use_clipped_model_output: Whether to clip model outputs
            generator: Random number generator for reproducibility
            
        Returns:
            prev_sample: Controlled denoised latent for next timestep
            control_times_sqrt_dt: Control vector field times square root of timestep size
            noise_pred_eval: Noise prediction from trained model
            noise_pred_init_eval: Noise prediction from reference model
            std_dev_t: Standard deviation at timestep t (for scaling in loss calculation)
            
        Processing steps:
            1. Extract latents and timesteps for evaluation
            2. Apply conditional guidance if enabled
            3. Generate noise predictions from both models
            4. Compute controlled and uncontrolled denoising steps
            5. Extract control signals from the difference
        """
        x_eval = all_x_t[:, indices_t, :, :, :]
        t_eval = all_t[indices_t]
        t_eval_repeat = t_eval.repeat(batch_size)
        x_eval = x_eval.view(batch_size * num_timesteps_to_load, 4, 64, 64)

        # expand the latents if we are doing classifier free guidance
        x_eval_model_input = torch.cat([x_eval] * 2) if self.config.cfg_control else x_eval
        t_eval_repeat = torch.cat([t_eval_repeat] * 2) if self.config.cfg_control else t_eval_repeat

        x_eval_model_input = self.soc_pipeline.scheduler.scale_model_input(x_eval_model_input, t_eval_repeat)

        # Make batch-level decision for prompt dropout
        use_negative_prompts = False
        if not self.config.cfg_control and hasattr(self.config, 'prompt_dropout') and self.config.prompt_dropout > 0:
            # Single random value for the entire batch
            use_negative_prompts = torch.rand(1, device=prompt_embeds.device).item() < self.config.prompt_dropout
        
        if use_negative_prompts:
            # Use negative prompts for the entire batch
            prompt_embeds_expanded = negative_prompt_embeds.repeat_interleave(num_timesteps_to_load, dim=0)
            # Force recomputation of noise_pred_init since we're using different prompts
            noise_pred_init = None
        else:
            # Original behavior for positive prompts
            prompt_embeds_expanded = prompt_embeds.repeat_interleave(num_timesteps_to_load, dim=0)
            if self.config.cfg_control:
                negative_prompt_embeds_expanded = negative_prompt_embeds.repeat_interleave(num_timesteps_to_load, dim=0)
                prompt_embeds_expanded = torch.cat([negative_prompt_embeds_expanded, prompt_embeds_expanded], dim=0)

        if self.config.learn_offset:
            noise_pred_offset_eval = self.unet(
                x_eval_model_input.to(self.unet.dtype),
                t_eval_repeat,
                encoder_hidden_states=prompt_embeds_expanded.to(self.unet.dtype),
                return_dict=False,
            )[0]
            
            if noise_pred_init is not None:
                noise_pred_init_eval = noise_pred_init
            else:
                with torch.no_grad():
                    noise_pred_init_eval = self.unet_init(
                        x_eval_model_input.to(self.unet_init.dtype),
                        t_eval_repeat,
                        encoder_hidden_states=prompt_embeds_expanded.to(self.unet_init.dtype),
                        return_dict=False,
                    )[0].detach()

            # perform guidance
            if self.config.cfg_control:
                noise_pred_offset_eval_uncond, noise_pred_offset_eval_text = noise_pred_offset_eval.chunk(2)
                noise_pred_offset_eval_uncond = noise_pred_offset_eval_uncond.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                noise_pred_offset_eval_text = noise_pred_offset_eval_text.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                noise_pred_offset_eval = noise_pred_offset_eval_uncond + self.config.guidance_scale * (
                    noise_pred_offset_eval_text - noise_pred_offset_eval_uncond
                )

                if noise_pred_init is None:
                    noise_pred_init_eval_uncond, noise_pred_init_eval_text = noise_pred_init_eval.chunk(2)
                    noise_pred_init_eval_uncond = noise_pred_init_eval_uncond.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                    noise_pred_init_eval_text = noise_pred_init_eval_text.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                    noise_pred_init_eval = noise_pred_init_eval_uncond + self.config.guidance_scale * (
                        noise_pred_init_eval_text - noise_pred_init_eval_uncond
                    )
            else:
                noise_pred_offset_eval = noise_pred_offset_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                noise_pred_init_eval = noise_pred_init_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)

            noise_pred_eval = noise_pred_init_eval + noise_pred_offset_eval

        else:
            noise_pred_eval = self.unet(
                x_eval_model_input.to(self.unet.dtype),
                t_eval_repeat,
                encoder_hidden_states=prompt_embeds_expanded.to(self.unet.dtype),
                return_dict=False,
            )[0]
            if noise_pred_init is not None:
                noise_pred_init_eval = noise_pred_init
            else:
                with torch.no_grad():
                    noise_pred_init_eval = self.unet_init(
                        x_eval_model_input.to(self.unet_init.dtype),
                        t_eval_repeat,
                        encoder_hidden_states=prompt_embeds_expanded.to(self.unet_init.dtype),
                        return_dict=False,
                    )[0].detach()

            # perform guidance
            if self.config.cfg_control:
                noise_pred_eval_uncond, noise_pred_eval_text = noise_pred_eval.chunk(2)
                noise_pred_eval_uncond = noise_pred_eval_uncond.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                noise_pred_eval_text = noise_pred_eval_text.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                noise_pred_eval = noise_pred_eval_uncond + self.config.guidance_scale * (
                    noise_pred_eval_text - noise_pred_eval_uncond
                )

                if noise_pred_init is None:
                    noise_pred_init_eval_uncond, noise_pred_init_eval_text = noise_pred_init_eval.chunk(2)
                    noise_pred_init_eval_uncond = noise_pred_init_eval_uncond.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                    noise_pred_init_eval_text = noise_pred_init_eval_text.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                    noise_pred_init_eval = noise_pred_init_eval_uncond + self.config.guidance_scale * (
                        noise_pred_init_eval_text - noise_pred_init_eval_uncond
                    )

            else:
                noise_pred_eval = noise_pred_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
                noise_pred_init_eval = noise_pred_init_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)

        x_eval = x_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
        prev_sample, prev_sample_init, prev_sample_diff, std_dev_t, _ = self.soc_pipeline.scheduler.step(
            noise_pred_eval, 
            noise_pred_init_eval, 
            t_eval.unsqueeze(0).repeat(batch_size, 1).to(torch.int), 
            x_eval,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=torch.zeros_like(noise_pred_eval, device=noise_pred_eval.device),
            return_dict=False,
        )
        ## Controlled update: 
        ##      prev_sample = x_eval + \delta * (b(x_eval,t_eval) + \sigma(t_eval) u(x_eval,t_eval)) + \sqrt{\delta} \sigma(t_eval) \epsilon
        ## Un-controlled update: 
        ##      prev_sample_init = x_eval + \delta * b(x_eval,t_eval) + \sqrt{\delta} \sigma(t_eval) \epsilon
        ## ==> prev_sample_diff = prev_sample - prev_sample_init = \delta \sigma(t_eval) u(x_eval,t_eval)
        
        control_times_sqrt_dt = prev_sample_diff / std_dev_t
        ## ==> control_times_sqrt_dt = \sqrt{\delta} u(x_eval,t_eval)

        # Return components needed for loss calculation:
        # - control_times_sqrt_dt: control vector field times square root of time step size
        # - prev_sample: Controlled denoised latent for next timestep
        # - noise_pred_eval: Noise prediction from our models
        # - std_dev_t: Standard deviation at timestep t (used for scaling in loss)
        return control_times_sqrt_dt, prev_sample, noise_pred_eval, noise_pred_init_eval, std_dev_t
    
    def grad_rewards(
        self,
        x,
        prompts,
        smooth_gradients=False,
        num_smooth_samples=20,
        noise_std=0.02,
        clip_quantile=0.9,
    ):
        """
        Calculate gradients of reward function with respect to latent representations.
        
        Can use standard backpropagation or gradient smoothing with clipping for improved stability:
        - Standard: Direct backpropagation through reward model
        - Smoothed: Average gradients over multiple noise samples, clip to quantile threshold
        
        Args:
            x: Latent representations
            prompts: Text prompts dict
            smooth_gradients: Whether to use gradient smoothing
            num_smooth_samples: Number of noise samples for smoothing
            noise_std: Standard deviation of added noise
            clip_quantile: Quantile threshold for gradient clipping
            
        Returns:
            Tuple of (gradients, reward_values)
        """
        if not smooth_gradients:
            # Original implementation
            with torch.enable_grad():
                x = x.requires_grad_(True)
                image = self.latent_to_decode_fn(x)
                reward_values = self.reward_fn(image, prompts['text'])
                output = torch.autograd.grad(
                    reward_values.sum(), x
                )[0]
                return output.detach(), reward_values.detach()
        else:
            batch_size = x.shape[0]
        
            # Store all gradients and rewards for processing after collection
            all_grads = []
            all_rewards = []
            
            # Process each noise sample individually
            for i in range(num_smooth_samples):
                # Generate single noise sample
                noise = torch.randn_like(x) * noise_std
                noisy_x = x + noise
                noisy_x = noisy_x.requires_grad_(True)
                
                # Forward pass for this single noise sample
                with torch.enable_grad():
                    images = self.latent_to_decode_fn(noisy_x)
                    rewards = self.reward_fn(images, prompts['text'])
                    grads = torch.autograd.grad(rewards.sum(), noisy_x, retain_graph=False)[0]
                
                # Store gradients and rewards
                all_grads.append(grads.detach())
                all_rewards.append(rewards.detach())
                
                # Optional progress reporting
                if i % 5 == 0 and self.global_rank == 0 and self.config.verbose:
                    print(f"Smoothing progress: {i+1}/{num_smooth_samples}")
            
            # Stack all gradients [num_samples, batch_size, 4, 64, 64]
            all_grads = torch.stack(all_grads)
            all_rewards = torch.stack(all_rewards)
            
            # Compute gradient norms: [num_samples, batch_size]
            grad_norms = torch.norm(all_grads.view(num_smooth_samples, batch_size, -1), dim=2)
            
            # For each prompt, compute the quantile of the gradient norms
            # Shape: [batch_size]
            quantile_vals = torch.quantile(grad_norms, clip_quantile, dim=0)
            
            # Clip the gradients - create a scaling factor
            # First, compute the max of quantile and actual norm to avoid division by zero
            # Shape: [num_samples, batch_size]
            scaling = torch.minimum(
                quantile_vals.unsqueeze(0) / torch.clamp(grad_norms, min=1e-8),
                torch.ones_like(grad_norms)
            )
            
            # Apply scaling to each gradient
            # Reshape scaling to broadcast: [num_samples, batch_size, 1, 1, 1]
            scaling = scaling.view(num_smooth_samples, batch_size, 1, 1, 1)
            clipped_grads = all_grads * scaling
            
            # Average over all samples
            avg_grads = clipped_grads.mean(dim=0)
            avg_rewards = all_rewards.mean(dim=0)
            
            if self.global_rank == 0 and self.config.verbose:
                print(f"Raw grad norm - min: {grad_norms.min().item():.4f}, max: {grad_norms.max().item():.4f}, mean: {grad_norms.mean().item():.4f}")
                print(f"Quantile values: {quantile_vals.mean().item():.4f}")
                print(f"Final grad norm: {torch.norm(avg_grads.view(batch_size, -1), dim=1).mean().item():.4f}")
            
            torch.cuda.empty_cache()
            return avg_grads, avg_rewards
    
    def log_metrics(self, prefix, metrics_dict, batch_size):
        """Log all metrics with consistent naming and parameters"""
        for name, value in metrics_dict.items():
            self.log(
                f"{prefix}_{name}",
                value.detach(),
                on_step=(prefix == "train"), 
                on_epoch=True,
                prog_bar=True, 
                logger=True,
                sync_dist=True,
                rank_zero_only=False,
                batch_size=batch_size,
            )
    
    def on_save_checkpoint(self, checkpoint):
        """Save config directly in the checkpoint file"""
        if hasattr(self, "config"):
            import yaml
            from pathlib import Path
            
            if hasattr(self.config, "config_path"):
                config_path = Path(self.config.config_path)
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        yaml_content = f.read()
                    checkpoint['config_yaml'] = yaml_content
                    checkpoint['config_path'] = str(config_path)
            
            checkpoint['config_dict'] = vars(self.config)
    
    # Abstract methods to be implemented by subclasses
    def training_step(self, batch, batch_id):
        """Main training loop, must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def validation_step(self, batch, batch_idx, stage):
        """Validation loop, must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")