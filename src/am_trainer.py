import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
from diffusers.models import UNet2DConditionModel
import numpy as np
from typing import Optional
import math
import json

import pytorch_lightning as pl
from core_utils import get_model
from pytorch_lightning.utilities import grad_norm

from soc_pipeline_sd import retrieve_timesteps, latent_to_decode
from metrics import do_image_reward, do_clip_score, do_human_preference_score
from custom_DDIM_scheduler import CustomDDIMScheduler

def reward_function(x, prompt, model, guidance_reward_fn="ImageReward", use_no_grad=False, use_score_from_prompt_batched=True, verbose=False):
    """
    Computes reward values for generated images using selected reward model.
    
    Args:
        x: Tensor of decoded images to evaluate
        prompt: Text prompts corresponding to images
        model: Model containing image processor for post-processing
        guidance_reward_fn: Which reward function to use ("ImageReward", "Clip-Score", or "HumanPreference")
        use_no_grad: Whether to disable gradient tracking
        use_score_from_prompt_batched: For ImageReward, whether to use batched scoring
        
    Returns:
        Tensor of reward values (with or without gradients based on use_no_grad)
    """
    # convert to pil image
    imagesx = model.image_processor.postprocess(x, output_type="pt")
    imagesx = [image for image in imagesx]

    if guidance_reward_fn == "ImageReward":
        rewards = do_image_reward(
            images=imagesx, 
            prompts=prompt, 
            use_no_grad=use_no_grad, 
            use_score_from_prompt_batched=use_score_from_prompt_batched,
        )
    elif guidance_reward_fn == "Clip-Score":
        rewards = do_clip_score(images=imagesx, prompts=prompt)
    elif guidance_reward_fn == "HumanPreference":
        rewards = do_human_preference_score(images=imagesx, prompts=prompt)
    else:
        raise ValueError(f"Unknown metric: {guidance_reward_fn}")

    if use_no_grad:
        return torch.tensor(rewards).to(x.device)
    else:
        if verbose:
            print(f'rewards.requires_grad in reward_function: {rewards.requires_grad}')
        return rewards

def reinitialize_last_layer(layer, mean=0.0, std=0.01):
    if isinstance(layer, nn.Conv2d):
        init.normal_(layer.weight, mean=mean, std=std)
        # Optionally, set bias to a small constant (or zero)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)


class AMTrainer(pl.LightningModule):
    def __init__(self, config):
        """
        Initialize Adjoint Matching Trainer for Stable Diffusion.
        
        Sets up:
        - Stable Diffusion UNet model and initialization
        - Custom DDIM scheduler with specified parameters
        - Reward and decoding functions
        - Timestep configuration
        - Optional layer reinitialization for offset learning
        
        Args:
            config: Configuration object with training parameters
        """
        super().__init__()
        self.config = config
        soc_pipeline = get_model(
            self.config.model_name,
            use_compile=False,
            bfloat_dtype=(self.config.precision == 'bf16'),
        )
        self.soc_pipeline = soc_pipeline

        # get timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.soc_pipeline.scheduler,
            self.config.num_inference_steps,
            device=self.device,
            timesteps=None,
            sigmas=None,
        )
        self.eval_step_outputs = dict(val=[], test=[])

        assert (
            num_inference_steps == self.config.num_inference_steps
        ), num_inference_steps

        if self.config.precision == 'bf16':
            self.torch_dtype = torch.bfloat16
        elif self.config.precision == '32-true':
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{self.config.precision}'. Allowed values are: 'bf16', '32-true'.")

        self.time_steps = timesteps.to(self.torch_dtype)

        # initialize unet model
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=self.torch_dtype,
        )
        self.unet.requires_grad_(True)

        self.soc_pipeline.unet_init = self.soc_pipeline.unet
        self.unet_init = self.soc_pipeline.unet_init
        self.unet_init.requires_grad_(False)

        self.soc_pipeline.unet = self.unet

        print(f'self.unet.dtype: {self.unet.dtype}, self.unet_init.dtype: {self.unet_init.dtype}')

        mutable_dict = dict(self.soc_pipeline.scheduler.config)
        mutable_dict['beta_start'] = config.beta_start
        mutable_dict['beta_end'] = config.beta_end
        self.soc_pipeline.scheduler = CustomDDIMScheduler.from_config(mutable_dict)
        self.soc_pipeline.scheduler.config.timestep_spacing = 'trailing'
        self.EMA_updates = 0
        self.EMA_value = -1
        self.EMA_decay = 0.9

        # define decoding and reward functions
        self.latent_to_decode_fn = lambda x: latent_to_decode(
            model=self.soc_pipeline, output_type="pt", latents=x,
        )
        self.reward_fn = lambda x, prompt: reward_function(x, 
                                                           prompt, 
                                                           model=self.soc_pipeline, 
                                                           guidance_reward_fn=self.config.guidance_reward_fn, 
                                                           verbose=self.config.verbose
                                                           )

        # get timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.soc_pipeline.scheduler,
            self.config.num_inference_steps,
            device=self.device,
            timesteps=None,
            sigmas=None,
        )
        print(f'self.soc_pipeline.scheduler.final_alpha_cumprod before: {self.soc_pipeline.scheduler.final_alpha_cumprod}')
        self.soc_pipeline.scheduler.final_alpha_cumprod = (1 + self.soc_pipeline.scheduler.alphas_cumprod[timesteps[-1]]) / 2
        print(f'self.soc_pipeline.scheduler.final_alpha_cumprod after: {self.soc_pipeline.scheduler.final_alpha_cumprod}')
        print(f'timesteps: {timesteps}')
        print(f'num_inference_steps: {num_inference_steps}')
        print(f'self.soc_pipeline.scheduler.beta_schedule: {self.soc_pipeline.scheduler.beta_schedule}')
        print(f'self.soc_pipeline.scheduler.betas: {self.soc_pipeline.scheduler.betas}')
        print(f'self.soc_pipeline.scheduler.alphas_cumprod: {self.soc_pipeline.scheduler.alphas_cumprod}')
        print(f'self.soc_pipeline.scheduler.config: {self.soc_pipeline.scheduler.config}')
        self.eval_step_outputs = dict(val=[], test=[])

        assert (
            num_inference_steps == self.config.num_inference_steps
        ), num_inference_steps

        self.time_steps = timesteps

        with open(self.config.validation_prompt_path, "r") as f:
            data = json.load(f)
        self.validation_prompts = [x['text'] for x in data]

        self.global_batch_step = 0

        if config.learn_offset:
            original_norm = self.unet.conv_out.weight.norm()
            print("Original weight norm:", original_norm.item())
            reinitialize_last_layer(self.unet.conv_out)
            new_norm = self.unet.conv_out.weight.norm()
            print("New weight norm:", new_norm.item())

    def on_fit_start(self):
        self.soc_pipeline.to(self.device)
        self.time_steps = self.time_steps.to(self.device)
        self.unet.to(self.device).train()

    def get_prompt_embeds(self, prompt):
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

        return prompt_embeds

    def prep_batch(self, batch):
        prompt_embeds, x_t, prompts = batch

        batch_size = len(prompt_embeds)
        assert (batch_size == x_t.shape[0]), (batch_size, x_t.shape[0])

        t = self.time_steps

        return prompt_embeds, x_t, t, prompts

    def aggregate_eval_outputs(self, stage):
        eval_outputs = self.self.eval_step_outputs[stage]

        eval_outputs = np.array(eval_outputs)
        eval_outputs = eval_outputs.mean()

        self.log(
            "val_loss",
            eval_outputs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.eval_step_outputs[stage] = []

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.unet, norm_type=2)
        norms = sum(norms.values())

        self.log(
            "grad_norm",
            norms,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

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
        
    def grad_inner_product(
        self, 
        x: torch.Tensor, 
        t: int, 
        vectors: torch.Tensor,
        prompt_embeds: torch.Tensor,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
    ):
        """
        Compute gradients of inner product between:
        1. (prev_sample_init - x) and 
        2. The adjoint vectors
        
        This is a key component of the adjoint method where we backpropagate the inner product
        to compute how noise predictions impact the reward through the denoising process.
        
        Args:
            x: Current latent state
            t: Current timestep
            vectors: The adjoint vectors from later timesteps
            prompt_embeds: Encoded text prompts
            eta: Noise parameter for scheduler
            use_clipped_model_output: Whether to clip model outputs
            generator: Optional random generator
            
        Returns:
            Tuple of (gradient, noise_prediction)
        """

        def inner_product(x):
            # x with shape (batch_size, 4, 64, 64)
            x_model_input = self.soc_pipeline.scheduler.scale_model_input(x, t)

            noise_pred = self.unet_init(
                # x_model_input.bfloat16(),
                x_model_input.to(self.unet_init.dtype),
                # t.bfloat16(),
                t,
                # encoder_hidden_states=prompt_embeds.bfloat16(),
                encoder_hidden_states=prompt_embeds.to(self.unet_init.dtype),
                return_dict=False,
            )[0]

            prev_sample_init, _, _, _ = self.soc_pipeline.scheduler.step(
                noise_pred, 
                None, 
                t, 
                x,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
                variance_noise=torch.zeros_like(noise_pred, device=noise_pred.device),
                return_dict=False,
            )
            sum_inner_prod = torch.sum((prev_sample_init - x) * vectors, dim=[0, 1, 2, 3])
            return sum_inner_prod, noise_pred
        
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sum_inner_prod, noise_pred = inner_product(x)
            output = torch.autograd.grad(
                sum_inner_prod,
                x,
            )[0]
            x = x.requires_grad_(False)
            return output.detach(), noise_pred.detach()

    def compute_adjoints(
        self, 
        all_x_t: torch.Tensor, 
        all_t: torch.Tensor, 
        prompt_embeds: torch.Tensor,
        prompts: list[str],
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
    ):
        """
        Compute adjoint states using backward propagation through the diffusion process.
        
        This is the core of the Adjoint Matching algorithm:
        1. Compute reward gradients at the final timestep
        2. Initialize adjoint state with reward gradients
        3. Propagate adjoints backward through timesteps using inner product gradients
        
        The adjoint states capture how changes to noise predictions at each timestep
        impact the final reward.
        
        Args:
            all_x_t: Sequence of latent states for all timesteps
            all_t: Timestep indices
            prompt_embeds: Encoded text prompts
            prompts: Text prompts
            eta: Noise parameter for scheduler
            use_clipped_model_output: Whether to clip model outputs
            generator: Optional random generator
            
        Returns:
            Tuple of (adjoint_states, reward_values, noise_predictions)
        """
        reward_grads, reward_values = self.grad_rewards(all_x_t[:,-1], 
                                                        prompts, 
                                                        smooth_gradients=self.config.smooth_gradients,
                                                        num_smooth_samples=self.config.smooth_samples,
                                                        noise_std=self.config.smooth_noise_std,
                                                        clip_quantile=self.config.smooth_clipping_quantile,
                                                        )
        assert all_x_t[:,:-1].shape[1] == len(all_t)
        num_timesteps = all_x_t.shape[1]
        if self.global_rank == 0 and self.config.verbose:
            print(f'all_x_t.shape: {all_x_t.shape}, all_x_t[:,-1].shape: {all_x_t[:,-1].shape}')
            print(f'num_timesteps: {num_timesteps}')
        
        with torch.no_grad():
            adjoint_states = torch.zeros_like(all_x_t)
            all_noise_pred_init = torch.zeros_like(all_x_t)
            a = -self.config.reward_multiplier * reward_grads.to(torch.float32)
            adjoint_states[:,-1] = a

            for k in range(num_timesteps - 2, -1, -1):
                grad_inner_prod, noise_pred_init = self.grad_inner_product(
                    all_x_t[:,k],
                    all_t[k],
                    a,
                    prompt_embeds,
                    eta=eta,
                    use_clipped_model_output=use_clipped_model_output,
                    generator=generator,
                )
                a += grad_inner_prod
                adjoint_states[:,k] = a
                if self.global_rank == 0 and self.config.verbose:
                    print(f'k: {k}, all_t[k]: {all_t[k]}, torch.sum(a**2): {torch.sum(a**2)}, torch.sum(all_x_t[:,k]**2): {torch.sum(all_x_t[:,k]**2)}, a.dtype: {a.dtype}')
                all_noise_pred_init[:,k] = noise_pred_init
                del grad_inner_prod, noise_pred_init

            _, noise_pred_init = self.grad_inner_product(
                all_x_t[:,-1],
                all_t[-1],
                a,
                prompt_embeds,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
            )
            all_noise_pred_init[:,-1] = noise_pred_init

        return adjoint_states, reward_values, all_noise_pred_init
    
    def switch_to_train(self):
        self.unet.train()
        self.unet_init.train()

    def switch_to_eval(self):
        self.unet.eval()
        self.unet_init.eval()

    def gather_across_gpus(self, tensor):
        """Concatenate a tensor across all GPUs."""
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def average_across_gpus(self, tensor):
        """Averages a tensor across all GPUs instead of concatenating."""
        world_size = dist.get_world_size()  # Get the number of GPUs
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # Sum across all GPUs
        tensor /= world_size  # Compute the average
        return tensor
    
    def EMA_update(self, value):
        if self.EMA_updates == 0:
            self.EMA_value = value
        elif self.EMA_updates < 1/self.EMA_decay:
            self.EMA_value = value / (self.EMA_updates + 1) + (1 - 1 / (self.EMA_updates + 1)) * self.EMA_value
        else:
            self.EMA_value = self.EMA_decay * self.EMA_value + (1 - self.EMA_decay) * value
        
        self.EMA_updates += 1
        return self.EMA_value 

    def training_step_args(
        self, 
        batch,
        num_timesteps_to_load,
        loss_clipping_threshold: float = -1.0,
        per_sample_threshold_quantile: float = 1.0,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
    ):
        """
        Core training logic for adjoint matching:
        
        1. Computes adjoint states for the denoising process
        2. Samples a subset of timesteps for computing the loss
        3. Runs UNet to predict noise for these timesteps 
        4. Computes losses by comparing:
           - Drift from UNet (prev_sample_diff / std_dev_t)
           - Target drift from adjoints (adjoint_states * std_dev_t)
        5. Applies various loss clipping techniques for stability
        
        Args:
            batch: Input data (prompt_embeds, latent trajectories, prompts)
            num_timesteps_to_load: Number of timesteps to sample for loss computation
            loss_clipping_threshold: Absolute threshold for loss clipping (-1 to disable)
            per_sample_threshold_quantile: Quantile-based threshold (e.g., 0.9 for 90th percentile)
            eta: Noise parameter for scheduler
            use_clipped_model_output: Whether to clip model outputs
            generator: Optional random generator
            variance_noise: Optional custom noise
            
        Returns:
            Tuple of (loss, reward_mean, control_norm, AM_target_norm, prev_sample_norm)
        """
        prompt_embeds, all_x_t, all_t, prompts = self.prep_batch(batch)
        batch_size = len(prompt_embeds)
        if self.global_rank == 0 and self.config.verbose:
            print(f'prompts: {prompts}')

        self.switch_to_eval()

        adjoint_states, reward_values, all_noise_pred_init = self.compute_adjoints(
            all_x_t,
            all_t.to(torch.int),
            prompt_embeds,
            prompts,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
        )
        if not hasattr(self, "EMA_adjoint_states"):
            self.EMA_adjoint_states = self.average_across_gpus(adjoint_states.mean(dim=0))
        else:
            self.EMA_adjoint_states = 0.99 * self.EMA_adjoint_states + (1 - 0.99) * self.average_across_gpus(adjoint_states.mean(dim=0))

        num_timesteps = len(all_t)
        middle_timestep = round(num_timesteps * 0.6)     
        indices_t_1 = np.random.choice(
            np.arange(0, middle_timestep), num_timesteps_to_load // 2, replace=False
        )
        indices_t_2 = np.random.choice(
            np.arange(middle_timestep, num_timesteps), 
            num_timesteps_to_load - num_timesteps_to_load // 2, 
            replace=False
        )
        indices_t = np.concatenate((indices_t_1, indices_t_2))  
        indices_t = np.sort(indices_t)
        indices_t = torch.tensor(indices_t, dtype=torch.long, device=all_x_t.device)

        x_eval = all_x_t[:, indices_t, :, :, :]
        t_eval = all_t[indices_t]
        t_eval_repeat = t_eval.repeat(batch_size)
        x_eval = x_eval.view(batch_size * num_timesteps_to_load, 4, 64, 64)

        x_eval_model_input = self.soc_pipeline.scheduler.scale_model_input(x_eval, t_eval_repeat)
        prompt_embeds_expanded = prompt_embeds.repeat_interleave(num_timesteps_to_load, dim=0)

        self.switch_to_train()

        if self.config.learn_offset:
            noise_pred_offset_eval = self.unet(
                x_eval_model_input.to(self.unet.dtype),
                t_eval_repeat,
                encoder_hidden_states=prompt_embeds_expanded.to(self.unet.dtype),
                return_dict=False,
            )[0]
            noise_pred_offset_eval = noise_pred_offset_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
            with torch.no_grad():
                noise_pred_init_eval = self.unet_init(
                    x_eval_model_input.to(self.unet_init.dtype),
                    t_eval_repeat,
                    encoder_hidden_states=prompt_embeds_expanded.to(self.unet_init.dtype),
                    return_dict=False,
                )[0].detach()
                noise_pred_init_eval = noise_pred_init_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
            noise_pred_eval = noise_pred_init_eval + noise_pred_offset_eval

        else:
            noise_pred_eval = self.unet(
                x_eval_model_input.to(self.unet.dtype),
                t_eval_repeat,
                encoder_hidden_states=prompt_embeds_expanded.to(self.unet.dtype),
                return_dict=False,
            )[0]
            noise_pred_eval = noise_pred_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
            with torch.no_grad():
                noise_pred_init_eval = self.unet_init(
                    x_eval_model_input.to(self.unet_init.dtype),
                    t_eval_repeat,
                    encoder_hidden_states=prompt_embeds_expanded.to(self.unet_init.dtype),
                    return_dict=False,
                )[0].detach()
                noise_pred_init_eval = noise_pred_init_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
        
        adjoint_states_eval = adjoint_states[:, indices_t, :, :, :]

        x_eval = x_eval.view(batch_size, num_timesteps_to_load, 4, 64, 64)
        prev_sample, prev_sample_init, prev_sample_diff, std_dev_t = self.soc_pipeline.scheduler.step(
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

        EMA_adjoint_states_expanded = self.EMA_adjoint_states[None,indices_t,:,:,:].expand(prev_sample.shape[0], -1, -1, -1, -1)
        if self.global_rank == 0 and self.config.verbose:
            print(f'torch.sum((prev_sample_diff / std_dev_t) ** 2): {torch.sum((prev_sample_diff / std_dev_t) ** 2)}')
            print(f'torch.sum((adjoint_states_eval * std_dev_t) ** 2): {torch.sum((adjoint_states_eval * std_dev_t) ** 2)}')
            print(f'indices_t: {indices_t}')
            print(f'std_dev_t[0,:,0,0,0]: {std_dev_t[0,:,0,0,0]}')
            print(f'torch.sum((prev_sample_diff / std_dev_t) ** 2, dim=[0,2,3,4]): {torch.sum((prev_sample_diff / std_dev_t) ** 2, dim=[0,2,3,4])}')
            print(f'torch.sum((prev_sample / std_dev_t) ** 2, dim=[0,2,3,4]): {torch.sum((prev_sample / std_dev_t) ** 2, dim=[0,2,3,4])}')
            print(f'torch.sum((adjoint_states_eval * std_dev_t) ** 2, dim=[0,2,3,4]): {torch.sum((adjoint_states_eval * std_dev_t) ** 2, dim=[0,2,3,4])}')
            print(f'loss_clipping_threshold: {loss_clipping_threshold}, per_sample_threshold_quantile: {per_sample_threshold_quantile}')
            print(f'torch.sum((EMA_adjoint_states_expanded * std_dev_t) ** 2, dim=[0,2,3,4]): {torch.sum((EMA_adjoint_states_expanded * std_dev_t) ** 2, dim=[0,2,3,4])}')

        loss_evals = torch.sum(
            (prev_sample_diff / std_dev_t + adjoint_states_eval * std_dev_t) ** 2,
            dim = [2,3,4]
        )
        control_norms = torch.sum(
            (prev_sample_diff.detach() / std_dev_t) ** 2,
            dim = [2,3,4]
        )
        AM_target_norms = torch.sum(
            (adjoint_states_eval * std_dev_t) ** 2,
            dim = [2,3,4]
        )
        prev_sample_norms = torch.sum(
            (prev_sample.detach() / std_dev_t) ** 2,
            dim = [2,3,4]
        )

        if loss_clipping_threshold >= 0:
            clip_mask = (loss_evals < loss_clipping_threshold).int()
            loss_evals = loss_evals * clip_mask
            loss_eval = torch.sum(loss_evals) / torch.sum(clip_mask)
            control_norm = torch.sum(control_norms) / torch.sum(clip_mask)
            AM_target_norm = torch.sum(AM_target_norms) / torch.sum(clip_mask)
            prev_sample_norm = torch.sum(prev_sample_norms) / torch.sum(clip_mask)
        elif per_sample_threshold_quantile >= 0:
            loss_eval_values = self.gather_across_gpus(loss_evals.detach())
            if self.global_rank == 0 and self.config.verbose:
                print(f'loss_evals.shape: {loss_evals.shape}, loss_eval_values.shape: {loss_eval_values.shape}')
                print(f'loss_eval_values: {loss_eval_values}')

            # Compute statistics
            min_val = torch.min(torch.sqrt(loss_eval_values)).item()
            mean_val = torch.mean(torch.sqrt(loss_eval_values)).item()
            median_val = torch.median(torch.sqrt(loss_eval_values)).item()
            max_val = torch.max(torch.sqrt(loss_eval_values)).item()
            percentile_90 = torch.quantile(torch.sqrt(loss_eval_values), 0.9).item()  # 90th percentile
            percentile_75 = torch.quantile(torch.sqrt(loss_eval_values), 0.75).item()  # 75th percentile
            quantile_threshold = torch.quantile(torch.sqrt(loss_eval_values), per_sample_threshold_quantile).item() # quantile threshold
            effective_sample_size = torch.sum(torch.sqrt(loss_eval_values)) ** 2 / torch.sum(loss_eval_values)

            quantile_threshold_EMA = self.EMA_update(quantile_threshold)

            per_sample_clipping_mask = (torch.sqrt(loss_evals.detach()) < quantile_threshold_EMA).to(torch.int32)

            if self.global_rank == 0 and self.config.verbose:
                # Print results
                print(f"Min: {min_val}, Mean: {mean_val}, Median: {median_val}, Max: {max_val}")
                print(f"90th Percentile: {percentile_90}, 75th Percentile: {percentile_75}")
                print(f"{round(per_sample_threshold_quantile * 100)}th Percentile: {quantile_threshold}")
                print(f"Effective Sample Size: {effective_sample_size}")
                print(f"{round(per_sample_threshold_quantile * 100)}th Percentile EMA: {quantile_threshold_EMA}, self.EMA_updates: {self.EMA_updates}")
                print(f'per_sample_clipping_mask.shape: {per_sample_clipping_mask.shape}')
                print(f'per_sample_clipping_mask: {per_sample_clipping_mask}')

            loss_eval = torch.mean(loss_evals * per_sample_clipping_mask)
            control_norm = torch.mean(control_norms * per_sample_clipping_mask)
            AM_target_norm = torch.mean(AM_target_norms * per_sample_clipping_mask)
            prev_sample_norm = torch.mean(prev_sample_norms * per_sample_clipping_mask)
        else:
            loss_eval = torch.mean(loss_evals)
            control_norm = torch.mean(control_norms)
            AM_target_norm = torch.mean(AM_target_norms)
            prev_sample_norm = torch.mean(prev_sample_norms)

        if self.global_rank == 0 and self.config.verbose:
            print(f'reward_values.shape: {reward_values.shape}, torch.mean(reward_values): {torch.mean(reward_values)}')
        return loss_eval, torch.mean(reward_values), control_norm, AM_target_norm, prev_sample_norm
    
    def log_metrics(self, prefix, loss, reward_mean, control_norm, target_norm, prev_sample_norm, batch_size):
        """Log all metrics with consistent naming and parameters"""
        for name, value in {
            "loss": loss,
            "reward_mean": reward_mean,
            "control_norm": control_norm, 
            "AM_target_norm": target_norm,
            "prev_sample_norm": prev_sample_norm
        }.items():
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

    def training_step(self, batch, batch_id):
        """
        Main training step that:
        1. Sets random seed for reproducibility
        2. Runs Stable Diffusion to generate trajectories
        3. Calls training_step_args to compute loss
        4. Logs metrics and returns loss for backpropagation
        
        Args:
            batch: Input batch (text prompts)
            batch_id: Batch index
            
        Returns:
            Training loss
        """
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(gpu_num + self.config.seed + 10 * self.global_batch_step)
        prompts = batch

        _, trajectories, prompt_embeds = self.soc_pipeline(
            prompts['text'],
            num_inference_steps=self.config.num_inference_steps,
            eta=self.config.eta,
            store_traj=True,
            use_custom_scheduler=True,
            learn_offset=self.config.learn_offset,
        )
        x_t = trajectories
        batch = prompt_embeds, x_t, prompts
        if self.global_rank == 0 and self.config.verbose:
            print(f'self.config.num_timesteps_to_load_train: {self.config.num_timesteps_to_load_train}, self.config.per_sample_threshold_quantile: {self.config.per_sample_threshold_quantile}, self.global_batch_step: {self.global_batch_step}')
        train_loss, train_reward_mean, train_control_norm, train_AM_target_norm, train_prev_sample_norm = self.training_step_args(
            batch,
            num_timesteps_to_load=self.config.num_timesteps_to_load_train,
            loss_clipping_threshold=-1.0,
            per_sample_threshold_quantile=self.config.per_sample_threshold_quantile,
            eta=self.config.eta,
            use_clipped_model_output=False,
            generator=None,
            variance_noise=None,
        )

        # Calculate batch size to use in logging
        batch_size = self.config.batch_size

        self.log_metrics(prefix="train", 
                         loss=train_loss.detach(), 
                         reward_mean=train_reward_mean.detach(), 
                         control_norm=train_control_norm.detach(),
                         target_norm=train_AM_target_norm.detach(), 
                         prev_sample_norm=train_prev_sample_norm.detach(), 
                         batch_size=batch_size)
        
        self.global_batch_step += 1
        del train_reward_mean, train_control_norm, train_AM_target_norm, train_prev_sample_norm

        return train_loss
    
    def evaluate_step(self, batch, batch_idx, stage):
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(gpu_num + self.config.seed + 10 * batch_idx)
        prompts = batch

        _, trajectories, prompt_embeds = self.soc_pipeline(
            prompts['text'],
            num_inference_steps=self.config.num_inference_steps,
            eta=self.config.eta,
            store_traj=True,
            use_custom_scheduler=True,
            learn_offset=self.config.learn_offset,
        )
        x_t = trajectories
        batch = prompt_embeds, x_t, prompts
        with torch.no_grad():
            val_loss, val_reward_mean, val_control_norm, val_AM_target_norm, val_prev_sample_norm = self.training_step_args(
                batch,
                num_timesteps_to_load=self.config.num_timesteps_to_load_train,
                loss_clipping_threshold=-1.0,
                per_sample_threshold_quantile=self.config.per_sample_threshold_quantile,
                eta=self.config.eta,
                use_clipped_model_output=False,
                generator=None,
                variance_noise=None,
            )
            # Calculate batch size to use in logging
            batch_size = self.config.batch_size

            self.log_metrics(prefix="val", 
                             loss=val_loss.detach(), 
                             reward_mean=val_reward_mean.detach(), 
                             control_norm=val_control_norm.detach(),
                             target_norm=val_AM_target_norm.detach(), 
                             prev_sample_norm=val_prev_sample_norm.detach(), 
                             batch_size=batch_size)
            
            del val_reward_mean, val_control_norm, val_AM_target_norm, val_prev_sample_norm
            torch.cuda.empty_cache()
            return val_loss
        
    def validation_step(self, batch, batch_idx):
        print(f'VALIDATION STEP, batch_idx: {batch_idx}')
        self.evaluate_step(batch, batch_idx, "val")
    
    def configure_optimizers(self):
        """
        Sets up optimizer and learning rate scheduler:
        - Optimizer: AdamW with configurable learning rate and betas
        - Scheduler options:
          1. StepLR: Decay learning rate by gamma every step_size steps
          2. Linear warmup: Linearly increase learning rate during warmup steps
        
        Returns:
            Configuration for PyTorch Lightning optimizer and scheduler
        """
        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.unet.parameters(), 
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2), 
                weight_decay=0.0
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

        if self.config.scheduler == "stepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.accum_grad_steps, gamma=0.99
            )
        elif self.config.scheduler == "linear_warmup": 
            # LambdaLR scheduler (linear warm-up)
            def lr_lambda(current_step):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))  # Linear warm-up
                return 1.0  # Normal LR after warm-up

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
                'interval': 'step',  # ⚡ Apply the scheduler per training step
                'frequency': 1,      # Run every step
            }
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")

        return [optimizer], [scheduler]
    
    def on_save_checkpoint(self, checkpoint):
        """Save config directly in the checkpoint file"""
        # Save the config object as part of the checkpoint
        if hasattr(self, "config"):
            import yaml
            from pathlib import Path
            
            # Get config_path if available, otherwise create a minimal config dict
            if hasattr(self.config, "config_path"):
                config_path = Path(self.config.config_path)
                if config_path.exists():
                    # Read the YAML file content
                    with open(config_path, 'r') as f:
                        yaml_content = f.read()
                        
                    # Store the YAML content and path in checkpoint
                    checkpoint['config_yaml'] = yaml_content
                    checkpoint['config_path'] = str(config_path)
            
            # Also store the config object attributes as a dictionary
            checkpoint['config_dict'] = vars(self.config)