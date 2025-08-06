import torch
import numpy as np
from typing import Optional
import pytorch_lightning as pl
import gc
from svd_trainers.soc_trainer import SOCTrainer
from torch.amp import autocast

class AMTrainer(SOCTrainer):
    """
    Adjoint Matching Trainer for training diffusion model control mechanisms.
    
    This class implements the Adjoint Matching algorithm which uses adjoint methods to efficiently
    compute optimal control vector fields for guided diffusion models. The approach backpropagates
    reward gradients through the diffusion process using adjoint states to create supervision signals
    for training the UNet model to predict optimal control directions.
    
    Key features:
    - Computes adjoint states that capture how noise predictions affect final rewards
    - Provides loss functions that match model outputs to optimal control directions
    - Uses quantile-based loss clipping for training stability
    - Supports conditional generation with classifier-free guidance
    
    The trainer inherits core diffusion infrastructure from SOCTrainer and adds the adjoint-specific
    optimization machinery needed for more efficient optimization of the diffusion policy.
    """
    def __init__(self, config):
        super().__init__(config)
        # Additional AM-specific initialization
        self.EMA_updates = 0
        self.EMA_value = -1
        self.EMA_decay = 0.9
        
    def grad_inner_product(
        self, 
        x: torch.Tensor, 
        t, 
        vectors: torch.Tensor,
        image_latents: torch.Tensor,
        image_embeddings: torch.Tensor,
        added_time_ids: torch.Tensor,
        generator=None,
    ):
        """
        Compute gradients of inner product between:
        1. (prev_sample_init - x) and 
        2. The adjoint vectors
        
        This is a key component of the adjoint method where we backpropagate the inner product
        to compute how noise predictions impact the reward through the denoising process.
        
        Args:
            x: Current latent state, (B, T, C, H, W)
            t: Current timestep, (B, )
            vectors: The adjoint vectors from later timesteps, (B, T, C, H, W)
            image_latents: Image latents, (2*B, T, C, H, W)
            image_embeddings: Image embeddings, (2*B, 1, 1024)
            added_time_ids: Added time ids, (2*B, 3)
            generator: Optional random generator
            
        Returns:
            Tuple of (gradient, noise_prediction)
        """

        def inner_product(x, image_latents, image_embeddings, added_time_ids):
            # x with shape (batch_size, T, C, H, W)
            x_model_input = torch.cat([x] * 2) if self.config.cfg_adjoint else x
            image_latents = image_latents if self.config.cfg_adjoint else image_latents.chunk(2)[1]
            image_embeddings = image_embeddings if self.config.cfg_adjoint else image_embeddings.chunk(2)[1]
            added_time_ids = added_time_ids if self.config.cfg_adjoint else added_time_ids.chunk(2)[1]

            x_model_input = torch.cat([x_model_input, image_latents], dim=2)
            batch_size = x_model_input.shape[0]
            x_model_input = self.soc_pipeline.scheduler.batch_scale_model_input(x_model_input, t.unsqueeze(0).repeat(batch_size))

            self.unet.set_adapter([]) #deactivate the lora during finetuning
            noise_pred = self.unet(
                x_model_input.to(self.unet.dtype),
                t,
                encoder_hidden_states=image_embeddings.to(self.unet.dtype),
                added_time_ids=added_time_ids.to(self.unet.dtype),
                return_dict=False,
            )[0]
            self.unet.set_adapter([self.lora_name]) #activate the lora during finetuning
            
            # perform guidance
            if self.config.cfg_adjoint:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.soc_pipeline.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            
            prev_sample_init, _, _, _, _ = self.soc_pipeline.scheduler.batch_step(
                noise_pred, 
                None, 
                t.unsqueeze(0).repeat(batch_size), 
                x,
                generator=generator,
                noise=torch.zeros_like(noise_pred, device=noise_pred.device),
                return_dict=False,
            )
            
            ## prev_sample_init = x + \Delta t b(x,t) + \sqrt{\Delta t} \sigma(t) \epsilon with \epsilon = 0
            ## ==> prev_sample_init - x = \Delta t b(x,t)

            sum_inner_prod = torch.sum((prev_sample_init - x) * vectors, dim=[0, 1, 2, 3, 4])
            return sum_inner_prod, noise_pred
        
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sum_inner_prod, noise_pred = inner_product(x, image_latents, image_embeddings, added_time_ids)
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
        image_latents: torch.Tensor,
        image_embeddings: torch.Tensor,
        added_time_ids: torch.Tensor,
        **kwargs,
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
            **kwargs: Additional arguments to pass to compute_loss, including:
                - eta: Noise parameter for scheduler
                - use_clipped_model_output: Whether to clip model outputs
                - generator: Optional random generator
            
        Returns:
            Tuple of (adjoint_states, reward_values, noise_predictions)
        """
        reward_grads, reward_values = self.grad_rewards(all_x_t[:,-1],
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
            all_noise_pred_init = torch.zeros_like(all_x_t[:, :-1])
            a = -self.config.reward_multiplier * reward_grads.to(torch.float32)
            adjoint_states[:,-1] = a

            for k in range(num_timesteps - 2, -1, -1):
                grad_inner_prod, noise_pred_init = self.grad_inner_product(
                    all_x_t[:,k],
                    all_t[k],
                    a,
                    image_latents,
                    image_embeddings,
                    added_time_ids,
                    **kwargs,
                )
                a += grad_inner_prod
                adjoint_states[:,k] = a
                if self.global_rank == 0 and self.config.verbose:
                    print(f'k: {k}, all_t[k]: {all_t[k]}, torch.sum(a**2): {torch.sum(a**2)}, torch.sum(all_x_t[:,k]**2): {torch.sum(all_x_t[:,k]**2)}, a.dtype: {a.dtype}')
                all_noise_pred_init[:,k] = noise_pred_init
                del grad_inner_prod, noise_pred_init

        return adjoint_states, reward_values, all_noise_pred_init
    
    def EMA_update(self, value):
        if self.EMA_updates == 0:
            self.EMA_value = value
        elif self.EMA_updates < 1/self.EMA_decay:
            self.EMA_value = value / (self.EMA_updates + 1) + (1 - 1 / (self.EMA_updates + 1)) * self.EMA_value
        else:
            self.EMA_value = self.EMA_decay * self.EMA_value + (1 - self.EMA_decay) * value
        
        self.EMA_updates += 1
        return self.EMA_value

    def sample_time_indices(self, all_t, all_x_t, num_timesteps_to_load):
        """
        Sample timestep indices for loss computation, using a stratified approach.
        
        We sample timesteps from both early and later parts of the diffusion process
        to ensure the model learns to control across the entire denoising trajectory.
        
        Args:
            all_t: All available timesteps 
            all_x_t: Tensor containing latents for all timesteps
            num_timesteps_to_load: Number of timesteps to sample for training
            
        Returns:
            indices_t: Tensor of selected timestep indices, sorted in ascending order
        """
        num_timesteps = len(all_t)
        # Split at 60% of the diffusion process - balances between
        # early timesteps (high noise) and later timesteps (more structure)
        middle_timestep = round(num_timesteps * 0.6)     
        
        # Sample first half from early timesteps (higher noise)
        indices_t_1 = np.random.choice(
            np.arange(0, middle_timestep), num_timesteps_to_load // 2, replace=False
        )
        
        # Sample second half from later timesteps (more structure)
        indices_t_2 = np.random.choice(
            np.arange(middle_timestep, num_timesteps), 
            num_timesteps_to_load - num_timesteps_to_load // 2, 
            replace=False
        )
        
        # Combine and sort the indices
        indices_t = np.concatenate((indices_t_1, indices_t_2))  
        indices_t = np.sort(indices_t)
        indices_t = torch.tensor(indices_t, dtype=torch.long, device=all_x_t.device)
        return indices_t
    
    def loss_quantile_statistics(self, loss_eval_values, per_sample_threshold_quantile):
        stats = {
            'min_val': torch.min(torch.sqrt(loss_eval_values)).item(),
            'mean_val': torch.mean(torch.sqrt(loss_eval_values)).item(),
            'median_val': torch.median(torch.sqrt(loss_eval_values)).item(),
            'max_val': torch.max(torch.sqrt(loss_eval_values)).item(),
            'percentile_90': torch.quantile(torch.sqrt(loss_eval_values), 0.9).item(),  # 90th percentile
            'percentile_75': torch.quantile(torch.sqrt(loss_eval_values), 0.75).item(),  # 75th percentile
            'quantile_threshold': torch.quantile(torch.sqrt(loss_eval_values), per_sample_threshold_quantile).item(),  # quantile threshold
            'effective_sample_size': torch.sum(torch.sqrt(loss_eval_values)) ** 2 / torch.sum(loss_eval_values)
        }
        if self.global_rank == 0 and self.config.verbose:
            # Print results
            print(f"Min: {stats['min_val']}, Mean: {stats['mean_val']}, Median: {stats['median_val']}, Max: {stats['max_val']}")
            print(f"90th Percentile: {stats['percentile_90']}, 75th Percentile: {stats['percentile_75']}")
            print(f"{round(per_sample_threshold_quantile * 100)}th Percentile: {stats['quantile_threshold']}")
            print(f"Effective Sample Size: {stats['effective_sample_size']}")
        return stats

    def compute_loss(
        self, 
        batch,
        batch_idx,
        num_timesteps_to_load,
        per_sample_threshold_quantile: float = 1.0,
        **kwargs,
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
            batch: Input data (x_t, t, image_latents, image_embeddings, added_time_ids)
            num_timesteps_to_load: Number of timesteps to sample for loss computation
            per_sample_threshold_quantile: Quantile-based threshold (e.g., 0.9 for 90th percentile)
            **kwargs: Additional arguments to pass to compute_loss, including:
                - use_clipped_model_output: Whether to clip model outputs
                - generator: Optional random generator
            
        Returns:
            Tuple of (loss, reward_mean, control_norm, AM_target_norm, prev_sample_norm)
        """
        all_x_t, all_t, image_latents, image_embeddings, added_time_ids = batch
        batch_size = len(all_x_t)

        self.switch_to_eval()

        adjoint_states, reward_values, all_noise_pred_init = self.compute_adjoints(
            all_x_t,
            all_t,
            image_latents,
            image_embeddings,
            added_time_ids,
            **kwargs,
        )

        indices_t = self.sample_time_indices(all_t, all_x_t, num_timesteps_to_load)
        
        adjoint_states_eval = adjoint_states[:, indices_t, :, :, :, :]
        noise_pred_init_eval = all_noise_pred_init[:, indices_t, :, :, :, :]

        self.switch_to_train()

        if self.config.cfg_adjoint == self.config.cfg_control:
            noise_pred_init_argument = noise_pred_init_eval
        else:
            noise_pred_init_argument = None

        # control_times_sqrt_dt: (batch_size, num_timesteps_to_load, T, C, H, W)
        # prev_sample: (batch_size, num_timesteps_to_load, T, C, H, W)
        # std_dev_t: (batch_size, num_timesteps_to_load, T, C, H, W)
        control_times_sqrt_dt, prev_sample, _, _, std_dev_t = self.evaluate_controls(
            batch_size, 
            num_timesteps_to_load, 
            image_latents, 
            image_embeddings,
            added_time_ids,
            all_x_t, 
            all_t, 
            indices_t,
            noise_pred_init=noise_pred_init_argument,
            **kwargs,
        )
        

        loss_evals = torch.sum(
            (control_times_sqrt_dt + adjoint_states_eval * std_dev_t) ** 2,
            dim = [2,3,4,5]
        )
        ## control_times_sqrt_dt = \sqrt{\delta} u(x_t,t)
        ## std_dev_t = \sqrt{\delta} \sigma(t)
        ## ==> adjoint_states_eval * std_dev_t = \sqrt{\delta} \sigma(t) \tilde{a}
        ## ==> (control_times_sqrt_dt + adjoint_states_eval * std_dev_t) ** 2 =
        ##      \delta * (u(x_t,t) + \sigma(t) \tilde{a}) ** 2   
        
        control_norms = torch.sum(
            control_times_sqrt_dt.detach() ** 2,
            dim = [2,3,4,5]
        )
        AM_target_norms = torch.sum(
            (adjoint_states_eval * std_dev_t) ** 2,
            dim = [2,3,4,5]
        )
        prev_sample_norms = torch.sum(
            (prev_sample.detach() / std_dev_t) ** 2,
            dim = [2,3,4,5]
        )

        if per_sample_threshold_quantile >= 0:
            gathered_loss_values = self.gather_across_gpus(loss_evals.detach())
            if self.global_rank == 0 and self.config.verbose:
                print(f'loss_evals.shape: {loss_evals.shape}, gathered_loss_values.shape: {gathered_loss_values.shape}')
                print(f'gathered_loss_values: {gathered_loss_values}')

            # Compute statistics
            stats = self.loss_quantile_statistics(gathered_loss_values, per_sample_threshold_quantile)

            quantile_threshold_EMA = self.EMA_update(stats['quantile_threshold'])

            per_sample_clipping_mask = (torch.sqrt(loss_evals.detach()) < quantile_threshold_EMA).to(torch.int32)

            if self.global_rank == 0 and self.config.verbose:
                # Print results
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
    
    def compute_training_loss(
        self, 
        batch,
        batch_idx,
        num_timesteps_to_load,
        per_sample_threshold_quantile: float = 1.0,
        **kwargs,
    ):
        """
        Wrapper for the compute_loss function to handle training-specific logic.
        
        Args:
            batch: Input data (x_t, t, image_latents, image_embeddings, added_time_ids)
            batch_idx: Batch index
            num_timesteps_to_load: Number of timesteps to sample for loss computation
            per_sample_threshold_quantile: Quantile-based threshold (e.g., 0.9 for 90th percentile)
            **kwargs: Additional arguments to pass to compute_loss, including:
                - eta: Noise parameter for scheduler
                - use_clipped_model_output: Whether to clip model outputs
                - generator: Optional random generator
            
        Returns:
            Tuple of (loss, reward_mean, control_norm, AM_target_norm, prev_sample_norm)
        """
        return self.compute_loss(
            batch,
            batch_idx,
            num_timesteps_to_load=num_timesteps_to_load,
            per_sample_threshold_quantile=per_sample_threshold_quantile,
            **kwargs,
        )
    
    def compute_evaluation_loss(
        self, 
        batch,
        batch_idx,
        num_timesteps_to_load,
        per_sample_threshold_quantile: float = 1.0,
        **kwargs,
    ):
        """
        Wrapper for the compute_loss function to handle evaluation-specific logic.
        
        Args:
            batch: Input data (prompt_embeds, latent trajectories, prompts)
            batch_idx: Batch index
            num_timesteps_to_load: Number of timesteps to sample for loss computation
            per_sample_threshold_quantile: Quantile-based threshold (e.g., 0.9 for 90th percentile)
            **kwargs: Additional arguments to pass to compute_loss, including:
                - eta: Noise parameter for scheduler
                - use_clipped_model_output: Whether to clip model outputs
                - generator: Optional random generator
            
        Returns:
            Tuple of (loss, reward_mean, control_norm, AM_target_norm, prev_sample_norm)
        """
        if self.config.quick_evaluation: # Whether to skip adjoint computation (for evaluation)
            all_x_t, all_t, image_latents, image_embeddings, added_time_ids = batch
            batch_size = len(all_x_t)
            self.switch_to_eval()

            # disble the autocast here to avoid OOM during evaluation step
            # pytorchlightning automatically enables autocast in validation step
            with autocast(device_type = "cuda", enabled=False):
                reward_grads, reward_values = self.grad_rewards(
                    all_x_t[:,-1],
                    smooth_gradients=self.config.smooth_gradients,
                    num_smooth_samples=self.config.smooth_samples,
                    noise_std=self.config.smooth_noise_std,
                    clip_quantile=self.config.smooth_clipping_quantile,
                )
            indices_t = self.sample_time_indices(all_t, all_x_t, num_timesteps_to_load)
            control_times_sqrt_dt, prev_sample, _, _, std_dev_t = self.evaluate_controls(
                batch_size, 
                num_timesteps_to_load, 
                image_latents, 
                image_embeddings,
                added_time_ids,
                all_x_t, 
                all_t, 
                indices_t,
                noise_pred_init=None,
                **kwargs,
            )
            control_norms = torch.sum(
                control_times_sqrt_dt.detach() ** 2,
                dim = [2,3,4,5]
            )
            prev_sample_norms = torch.sum(
                (prev_sample.detach() / std_dev_t) ** 2,
                dim = [2,3,4,5]
            )
            control_norm = torch.mean(control_norms)
            prev_sample_norm = torch.mean(prev_sample_norms)

            # Return dummy values for consistency with the full evaluation
            dummy_val = torch.tensor(0.0, device=reward_values.device)
            return dummy_val, torch.mean(reward_values), control_norm, dummy_val, prev_sample_norm
        else:
            return self.compute_loss(
                batch,
                batch_idx,
                num_timesteps_to_load=num_timesteps_to_load,
                per_sample_threshold_quantile=per_sample_threshold_quantile,
                **kwargs,
            )

    def generate_trajectories(self, batch):
        init_frame = batch[1]
        _, _, _, trajectories, image_latents, image_embeddings, added_time_ids, _ = self.soc_pipeline(
            init_frame,
            num_frames=self.config.target_vid_size[0],
            height=self.config.target_vid_size[1],
            width=self.config.target_vid_size[2],
            num_inference_steps=self.config.num_inference_steps,
            store_traj=True,
            use_soc_scheduler=True,
            learn_offset=self.config.learn_offset,
            output_type="latent",
        )
        x_t = trajectories
        t = self.time_steps
        batch = x_t, t, image_latents, image_embeddings, added_time_ids
        return batch

    def generate_trajectories_train(self, batch, batch_idx):
        return self.generate_trajectories(batch)
    
    def generate_trajectories_eval(self, batch, batch_idx):
        return self.generate_trajectories(batch)

    def training_step(self, batch, batch_idx):
        """
        Main training step that:
        1. Sets random seed for reproducibility
        2. Runs Stable Diffusion to generate trajectories
        3. Calls compute_training_loss to compute loss
        4. Logs metrics and returns loss for backpropagation
        
        Args:
            batch: Input batch, tuple of (gt_video_tensor, init_frame)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(gpu_num + self.config.seed + 10 * self.global_batch_step)

        # batch = (x_t, t, image_latents, image_embeddings, added_time_ids)
        batch = self.generate_trajectories_train(batch, batch_idx)
        
        if self.global_rank == 0 and self.config.verbose:
            print(f'self.config.num_timesteps_to_load_train: {self.config.num_timesteps_to_load_train}, self.config.per_sample_threshold_quantile: {self.config.per_sample_threshold_quantile}, self.global_batch_step: {self.global_batch_step}')
        train_loss, train_reward_mean, train_control_norm, train_AM_target_norm, train_prev_sample_norm = self.compute_training_loss(
            batch,
            batch_idx,
            num_timesteps_to_load=self.config.num_timesteps_to_load_train,
            per_sample_threshold_quantile=self.config.per_sample_threshold_quantile,
            use_clipped_model_output=False,
            generator=None,
        )

        # Calculate batch size to use in logging
        batch_size = self.config.batch_size

        # Log all metrics
        metrics_dict = {
            "loss": train_loss.detach(),
            "reward_mean": train_reward_mean.detach(),
            "control_norm": train_control_norm.detach(),
            "target_norm": train_AM_target_norm.detach(),
            "prev_sample_norm": train_prev_sample_norm.detach(),
        }
        self.log_metrics(
            prefix="train", 
            metrics_dict=metrics_dict,
            batch_size=batch_size,
        )
        
        self.global_batch_step += 1
        del train_reward_mean, train_control_norm, train_AM_target_norm, train_prev_sample_norm

        return train_loss
    
    def evaluate_step(self, batch, batch_idx, stage):
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(gpu_num + self.config.seed + 10 * batch_idx)

        batch = self.generate_trajectories_eval(batch, batch_idx)
        with torch.no_grad():
            val_loss, val_reward_mean, val_control_norm, val_AM_target_norm, val_prev_sample_norm = self.compute_evaluation_loss(
                batch,
                batch_idx,
                num_timesteps_to_load=self.config.num_timesteps_to_load_train,
                per_sample_threshold_quantile=self.config.per_sample_threshold_quantile,
                generator=None,
            )
            # Calculate batch size to use in logging
            batch_size = self.config.batch_size

            # Log all metrics
            if self.config.quick_evaluation:
                metrics_dict = {
                    "reward_mean": val_reward_mean.detach(),
                    "control_norm": val_control_norm.detach(),
                    "prev_sample_norm": val_prev_sample_norm.detach(),
                }
            else:
                metrics_dict = {
                    "loss": val_loss.detach(),
                    "reward_mean": val_reward_mean.detach(),
                    "control_norm": val_control_norm.detach(),
                    "target_norm": val_AM_target_norm.detach(),
                    "prev_sample_norm": val_prev_sample_norm.detach(),
                }
            self.log_metrics(
                prefix="val", 
                metrics_dict=metrics_dict,
                batch_size=batch_size,
            )
            
            del val_reward_mean, val_control_norm, val_AM_target_norm, val_prev_sample_norm
            torch.cuda.empty_cache()
            return val_loss
        
    def validation_step(self, batch, batch_idx):
        print(f'VALIDATION STEP, batch_idx: {batch_idx}')
        if next(self.soc_pipeline.vae.encoder.parameters()).device == torch.device('cpu'):
            print('move vae to gpu')
            self.soc_pipeline.vae.encoder.to(self.device)
            self.soc_pipeline.vae.decoder.to(self.device)

        self.evaluate_step(batch, batch_idx, "val")

        self.soc_pipeline.vae.encoder.to('cpu')
        self.soc_pipeline.vae.decoder.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        # Use the parent class's load_from_checkpoint method
        return super().load_from_checkpoint(checkpoint_path, config=config, **kwargs)