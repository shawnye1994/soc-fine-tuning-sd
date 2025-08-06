import torch
import numpy as np
from typing import Optional
import pytorch_lightning as pl
from einops import rearrange

from svd_trainers.soc_trainer import SOCTrainer
from svd_trainers.buffer_soc_trainer import BufferSOCTrainer
from svd_trainers.am_trainer import AMTrainer

class BufferAMTrainer(BufferSOCTrainer, AMTrainer):
    """
    Buffer-based Adjoint Matching Trainer for diffusion models.
    
    This class combines BufferSOCTrainer and AMTrainer to create a memory-efficient
    implementation of the Adjoint Matching algorithm. It leverages advanced buffer
    management to optimize the compute-intensive process of generating and backpropagating
    through diffusion trajectories.
    
    Key features:
    - Implements efficient buffer management for trajectories and adjoints
    - Uses importance sampling to handle distribution shifts between buffer and current policy
    - Computes log importance weights for accurate gradient estimation
    - Monitors effective sample size (ESS) to detect degradation in sampling quality
    - Supports quantile-based loss clipping for stability
    - Optimizes memory usage across multiple GPUs
    
    The buffer stores multiple components required for training:
    - Trajectories: Latent paths through the diffusion process
    - Adjoint states: Backpropagated reward gradients through time
    - Noise predictions: From both current and reference models
    - Random noises: Used for importance weight computation
    - Rewards: Evaluation metrics for generated samples
    - Prompt embeddings: Text conditioning information
    
    This approach significantly improves training throughput while maintaining
    the mathematical correctness of the adjoint matching optimization.
    """
    def __init__(self, config):
        # Initialize SOCTrainer first (base class)
        SOCTrainer.__init__(self, config)
        
        # Initialize AMTrainer-specific attributes
        self.EMA_updates = 0
        self.EMA_value = -1
        self.EMA_decay = 0.9
        
        # Initialize buffer-specific attributes
        self.buffer_size = config.buffer_size
        self.buffer_device = config.buffer_device
        self.iterations_per_chunk = self.buffer_size // (torch.cuda.device_count() * config.batch_size)
        self.iterations_per_epoch = config.iterations_per_epoch
        self.chunks_per_epoch = self.iterations_per_epoch
        
        # Initialize buffer structures for adjoint states and trajectories
        self.buffer_variables = [
            'adjoint_states', 'trajectories', 'rewards', 'random_noises', 
            'noise_preds', 'noise_preds_init', 'image_latents', 'image_embeddings', 'added_time_ids'
        ]
        self.buffer = {}
        for var in self.buffer_variables:
            self.buffer[var] = None
        
        # Additional buffer-specific tracking
        self.current_buffer_index = 0
        self.buffer_initialized = False
        self.buffer_update_frequency = self.iterations_per_chunk * config.passes_per_buffer
        self.passes_per_buffer = config.passes_per_buffer
        self.current_pass = 0  # Track which pass through the buffer we're on

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        # Use the parent class's load_from_checkpoint method
        return super().load_from_checkpoint(checkpoint_path, config=config, **kwargs)
        
    def collect_data(self, init_frame, batch_start):
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(gpu_num + self.config.seed + 10 * self.global_batch_step + 100 * batch_start)
        # First, compute trajectories and the corresponding random noises for computing the importance weights
        _, all_random_noises, all_noise_preds, all_x_t, image_latents, image_embeddings, added_time_ids, timesteps = self.soc_pipeline(
            # todo can we compute the control norms already here?
            init_frame,
            num_frames=self.config.target_vid_size[0],
            height=self.config.target_vid_size[1],
            width=self.config.target_vid_size[2],
            num_inference_steps=self.config.num_inference_steps,
            store_traj=True,
            store_noise=True,
            store_noise_pred=True,
            use_soc_scheduler=True,
            learn_offset=self.config.learn_offset,
            output_type="latent"
        )
        all_t = timesteps
        self.switch_to_eval()

        adjoint_states, reward_values, all_noise_pred_init = self.compute_adjoints(
            all_x_t,
            all_t,
            image_latents,
            image_embeddings,
            added_time_ids,
            generator=None,
        )

        output = {
            'trajectories': all_x_t,
            'adjoint_states': adjoint_states,
            'rewards': reward_values,
            'random_noises': all_random_noises,
            'noise_preds': all_noise_preds,
            'noise_preds_init': all_noise_pred_init,
            'image_latents': image_latents,
            'image_embeddings': image_embeddings,
            'added_time_ids': added_time_ids,
        }
        return output

    def control_difference_from_buffer(
        self,
        noise_pred_eval,
        noise_pred_buffer,
        t_eval,
        x_eval,
        batch_size,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None, 
    ):  
        nttl = noise_pred_eval.shape[1]
        # noise_pred_eval should be b t c h w
        noise_pred_eval = rearrange(noise_pred_eval, 'b nttl t c h w -> (b nttl) t c h w')
        noise_pred_buffer = rearrange(noise_pred_buffer, 'b nttl t c h w -> (b nttl) t c h w')
        x_eval = rearrange(x_eval, 'b nttl t c h w -> (b nttl) t c h w')
        t_eval_repeat = t_eval.unsqueeze(0).repeat(batch_size, 1)
        t_eval_repeat = rearrange(t_eval_repeat, 'b nttl -> (b nttl)')
        _, _, prev_sample_diff_from_buffer, std_dev_t, _ = self.soc_pipeline.scheduler.batch_step(
            noise_pred_eval, 
            noise_pred_buffer, 
            t_eval_repeat, 
            x_eval,
            generator=generator,
            noise=torch.zeros_like(noise_pred_eval, device=noise_pred_eval.device),
            return_dict=False,
            scheduler_timesteps=self.time_steps
        )
        prev_sample_diff_from_buffer = rearrange(prev_sample_diff_from_buffer, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=nttl)
        std_dev_t = rearrange(std_dev_t, '(b nttl) t c h w-> b nttl t c h w', b=batch_size, nttl=nttl)
        # prev_sample_diff_from_buffer = \delta \sigma(t) (u(x_t,t) - u_{buffer}(x_t,t))
        # ==> prev_sample_diff_from_buffer / std_dev_t = \sqrt{\delta} (u(x_t,t) - u_{buffer}(x_t,t))
        return prev_sample_diff_from_buffer / std_dev_t
    
    def stepwise_log_importance_weights(self, control_difference, noise):
        # Integrated control norms - 0.5 ||u_1(x_t,t) - u_2(x_t,t)||² dt  [batchsize, timesteps, T, C, H, W] -> [batchsize, timesteps]
        deterministic_path_weights = -0.5 * torch.sum(
            control_difference ** 2,
            dim=[2, 3, 4, 5]
        )

        # Stochastic costs <u_1(x_t,t) - u_2(x_t,t), dB_t>  [batchsize, timesteps, T, C, H, W] -> [batchsize, timesteps]
        stochastic_path_weights = torch.sum(
            control_difference * noise,
            dim=[2, 3, 4, 5]
        )

        # The logarithm of the ratio between the conditional density for u_1
        # and the conditional density of u_2 is given by:
        # log w(X) = - 0.5 ||u_1(x_t,t) - u_2(x_t,t)||² dt + <u_1(x_t,t) - u_2(x_t,t), dB_t>
        # Proof via Girsanov's theorem, or just by writing out the ratio of the two conditional Gaussian densities and
        # simplifying.
        return deterministic_path_weights + stochastic_path_weights
    
    def compute_training_loss(
        self, 
        batch,
        batch_idx,
        num_timesteps_to_load,
        per_sample_threshold_quantile: float = 1.0,
        **kwargs,
    ):
        """
        Core training logic for adjoint matching:
        
        1. Takes adjoint states from buffer
        2. Samples a subset of timesteps for computing the loss
        3. Runs UNet to predict noise for these timesteps 
        4. Computes losses by comparing:
           - Control vector field control_times_sqrt_dt = prev_sample_diff / std_dev_t
           - Target vector field from adjoints (adjoint_states * std_dev_t)
        5. Applies various loss clipping techniques for stability
        
        Args:
            batch: Input data (prompt_embeds, latent trajectories, prompts)
            num_timesteps_to_load: Number of timesteps to sample for loss computation
            per_sample_threshold_quantile: Quantile-based threshold (e.g., 0.9 for 90th percentile)
            **kwargs: Additional arguments for the scheduler, including:
                - eta: Noise scale for the scheduler 
                - use_clipped_model_output: Whether to clip model outputs
                - generator: Optional random generator
            
        Returns:
            Tuple of (loss, reward_mean, control_norm, AM_target_norm, prev_sample_norm)
        """
        image_latents, image_embeddings, added_time_ids, all_x_t, all_t, adjoint_states, reward_values, noise_preds, noise_preds_init, random_noises = batch
        batch_size = len(all_x_t)
        if self.global_rank == 0 and self.config.verbose:
            print(f'image_latents: {image_latents}')

        self.switch_to_eval()

        indices_t = self.sample_time_indices(all_t, all_x_t, num_timesteps_to_load)
        
        adjoint_states_eval = adjoint_states[:, indices_t, :, :, :, :]
        noise_pred_buffer = noise_preds[:, indices_t, :, :, :, :]
        noise_pred_init_buffer = noise_preds_init[:, indices_t, :, :, :, :]
        random_noises_buffer = random_noises[:, indices_t, :, :, :, :]

        self.switch_to_train()

        if self.config.cfg_adjoint == self.config.cfg_control:
            noise_pred_init_argument = noise_pred_init_buffer
        else:
            noise_pred_init_argument = None

        control_times_sqrt_dt, prev_sample, noise_pred_eval, _, std_dev_t = self.evaluate_controls(
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

        control_difference_from_buffer = self.control_difference_from_buffer(
            noise_pred_eval.detach(),
            noise_pred_buffer,
            all_t[indices_t],
            all_x_t[:, indices_t, :, :, :, :],
            batch_size,
            **kwargs,
        )

        stepwise_log_weights = self.stepwise_log_importance_weights(
            control_difference_from_buffer,
            random_noises_buffer
        )    
        stepwise_weights = torch.exp(stepwise_log_weights)
        gathered_stepwise_weights = self.gather_across_gpus(stepwise_weights.detach())
        ess_stepwise = torch.sum(gathered_stepwise_weights) ** 2 / torch.sum(gathered_stepwise_weights ** 2)
        self.log_metrics(
            prefix="train", 
            metrics_dict={'ess_stepwise': ess_stepwise},
            batch_size=batch_size,
        )
        if self.global_rank == 0 and self.config.verbose:
            print(f'stepwise_weights.shape: {stepwise_weights.shape}')
            print(f'{stepwise_weights}')

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
            loss_eval_values = self.gather_across_gpus(loss_evals.detach())
            if self.global_rank == 0 and self.config.verbose:
                print(f'loss_evals.shape: {loss_evals.shape}, loss_eval_values.shape: {loss_eval_values.shape}')
                print(f'loss_eval_values: {loss_eval_values}')

            # Compute statistics
            stats = self.loss_quantile_statistics(loss_eval_values, per_sample_threshold_quantile)

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
    
    def generate_trajectories_train(self, batch, batch_idx):
        batch_idx_within_chunk = batch_idx % self.iterations_per_chunk
        t = self.time_steps

        image_latents = self.buffer['image_latents'][batch_idx_within_chunk]
        image_embeddings = self.buffer['image_embeddings'][batch_idx_within_chunk]
        added_time_ids = self.buffer['added_time_ids'][batch_idx_within_chunk]
        x_t = self.buffer['trajectories'][batch_idx_within_chunk]
        adjoint_states = self.buffer['adjoint_states'][batch_idx_within_chunk]
        reward_values = self.buffer['rewards'][batch_idx_within_chunk]
        noise_preds = self.buffer['noise_preds'][batch_idx_within_chunk]
        noise_preds_init = self.buffer['noise_preds_init'][batch_idx_within_chunk]
        random_noises = self.buffer['random_noises'][batch_idx_within_chunk]

        return image_latents, image_embeddings, added_time_ids, x_t, t, adjoint_states, reward_values, noise_preds, noise_preds_init, random_noises

    def generate_trajectories_eval(self, batch, batch_idx):
        return self.generate_trajectories(batch)