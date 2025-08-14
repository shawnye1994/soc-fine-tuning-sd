import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import pytorch_lightning as pl
from video_core_utils import get_model, reinitialize_last_layer

from soc_pipeline_svd import retrieve_timesteps, latent_to_decode
from video_metrics import reward_function
from video_asthetics_reward_utils import video_asthetics_rm_load
from video_reward_utils import video_rm_load

from SOC_EDM_Ancestral_scheduler import SOCEDMAncestralScheduler
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.training_utils import cast_training_params
# from video_core_utils import _LoraWrapper
from einops import rearrange, repeat
import gc
# from torchviz import make_dot


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
        """Setup the Stable Video Diffusion pipeline"""
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
        # enable gradient checkpointing
        self.soc_pipeline.unet.enable_gradient_checkpointing()
        self.soc_pipeline.vae.enable_gradient_checkpointing()

    def _initialize_unet(self):
        """Initialize and setup UNet model"""
        # Store initial weights
        # self.soc_pipeline.unet_init = self.soc_pipeline.unet
        # self.unet_init = self.soc_pipeline.unet_init
        # self.unet_init.requires_grad_(False)
        self.unet = self.soc_pipeline.unet
        self.unet.requires_grad_(False)

        # Initialize UNet Lora
        lora_config = self.config.lora_config
        if self.config.only_lora_on_temporal_module:
            #only inject lora to all the motion_modules layers
            motion_target_modules = []
            for k, _ in self.unet.named_modules():
                if "temporal_transformer_blocks" in k:
                    for m in lora_config['target_modules']:
                        if m in k:
                            motion_target_modules.append(k)
        else:
            # inject lora to all blocks other than temporal_transformer_blocks
            # used in the aethestics finetuning debug
            motion_target_modules = []
            for k, _ in self.unet.named_modules():
                if "temporal_transformer_blocks" not in k:
                    for m in lora_config['target_modules']:
                        if m in k:
                            motion_target_modules.append(k)
        lora_config['target_modules'] = motion_target_modules
        unet_lora_config = LoraConfig(**lora_config)
        # Add adapter and make sure the trainable params are in float32.
        # Normally, LoRA doesn't benefit from EMA: https://github.com/huggingface/diffusers/issues/9998
        self.unet.add_adapter(unet_lora_config, adapter_name=self.config.lora_adapter_name)

        assert self.config.lora_precision in ["fp32", "bfloat16", "fp16"], f"Invalid LoRA precision: {self.config.lora_precision}"
        if self.config.lora_precision == "fp32":
            # only upcast LoRA parameters into fp32
            cast_training_params(self.unet, dtype=torch.float32)

        lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        trainable_params = lora_layers
        # count the totale number of parameters in the lora_layers
        total_params = sum(p.numel() for p in lora_layers)
        print(f"Total trainable LoRA parameters: {total_params/1e6} M")

        # use the _LoraWrapper to wrap the unet_init and the unet
        # base_unet = self.unet_init
        self.lora_name = self.config.lora_adapter_name
        self.soc_pipeline.lora_name = self.lora_name
        # self.unet_init = _LoraWrapper(base_unet, None)
        # self.unet = _LoraWrapper(base_unet, lora_name)

        # Set the pipeline to use our trainable UNet
        self.soc_pipeline.unet = self.unet
        
        # Handle last layer initialization if needed
        if self.config.learn_offset:
            self._reinitialize_last_layer()
        
        self.soc_pipeline.unet.enable_gradient_checkpointing()

    def _initialize_scheduler(self):
        """Setup the diffusion scheduler with custom parameters"""
        self.soc_pipeline.set_edm_ancestral_scheduler()
        
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
        
        # self.time_steps = timesteps.to(self.torch_dtype)
        self.time_steps = timesteps
        
        # TO DO: Should only be used for the DDIM
        # Adjust final alpha as needed
        # self.soc_pipeline.scheduler.final_alpha_cumprod = (
        #     1 + self.soc_pipeline.scheduler.alphas_cumprod[timesteps[-1]]
        # ) / 2

    def _setup_utility_functions(self):
        """Define utility functions for the pipeline"""
        self.latent_to_decode_fn = lambda x: latent_to_decode(
            model=self.soc_pipeline, output_type="pt", latents=x, decode_chunk_size=self.config.vae_decode_chunk_size
        )
        
        # setup reward function
        if self.config.reward_func == "VideoReward":
            self.reward_model = video_rm_load(traj_discriminator_config=self.config.reward_model_config,
                                              device=self.device)
        elif self.config.reward_func == "VideoAestheticsReward":
            self.reward_model = video_asthetics_rm_load(asthetics_model_config=self.config.reward_model_config,
                                                        device=self.device)
        else:
            raise ValueError(f"Unknown metric: {reward_func}")
        
        # self.reward_fn = lambda x: reward_function(
        #     x,
        #     traj_discriminator_config=self.config.reward_model_config,
        #     device=self.device,
        #     reward_func=self.config.reward_func,
        #     verbose=self.config.verbose
        # )

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

    def switch_to_train(self):
        """Set models to training mode"""
        self.unet.train()
        # self.unet_init.train()

    def switch_to_eval(self):
        """Set models to evaluation mode"""
        self.unet.eval()
        # self.unet_init.eval()

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
        prev_sample, prev_sample_init, prev_sample_diff, std_dev_t, _ = self.soc_pipeline.scheduler.batch_step(
            noise_pred, 
            noise_pred_init, 
            t_eval.unsqueeze(0).repeat(batch_size, 1),
            x_eval,
            generator=generator,
            noise=torch.zeros_like(noise_pred, device=noise_pred.device),
            return_dict=False,
        )
        # Extract the control signal by dividing the difference by standard deviation
        control_times_sqrt_dt = prev_sample_diff / std_dev_t
        return control_times_sqrt_dt
    
    def evaluate_controls(
        self, 
        batch_size, 
        num_timesteps_to_load,
        image_latents,
        image_embeddings,
        added_time_ids,
        all_x_t, 
        all_t, 
        indices_t,
        noise_pred_init=None,
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
            num_timesteps_to_load: Number of timesteps to process, randomly sampled from all_t
            image_latents: [2*batch_size, T, C, H, W], Encoded conditional image_latents, include the negative/zero image latents if the svd_soc_pipeline used cfg. [0:1, ...] corresponds to the negative/zero latent.
            image_embeddings: [2*batch_size, 1, 1024], Embedded conditional image_embeddings for the cross attention, include the negative/zero image embeddings if the svd_soc_pipeline used cfg, [0:1, ...] corresponds to the negative/zero embedding.
            added_time_ids: [2*batch_size, 3], Added time ids for the cross attention, include the negative/zero time ids if the svd_soc_pipeline used cfg (batch size doubled), [0:1, ...] corresponds to the negative/zero time id.
            
            all_x_t: Latent representations at all timesteps [1, num_inference_timesteps, T, C, H, W]
            all_t: All timestep values, [num_inference_timesteps,]
            indices_t: Indices of timesteps to evaluate, len(indices_t) = num_timesteps_to_load
            noise_pred_init: Pre-computed noise predictions from reference model (optional)
            generator: Random number generator for reproducibility
            
        Returns:
            prev_sample: Controlled denoised latent for next timestep, (batch_size*num_timesteps_to_load, T, C, H, W)
            control_times_sqrt_dt: Control vector field times square root of timestep size, (batch_size*num_timesteps_to_load, T, C, H, W)
            noise_pred_eval: Noise prediction from trained model, (batch_size*num_timesteps_to_load, T, C, H, W)
            noise_pred_init_eval: Noise prediction from reference model, (batch_size*num_timesteps_to_load, T, C, H, W)
            std_dev_t: Standard deviation at timestep t (for scaling in loss calculation), (batch_size*num_timesteps_to_load, T, C, H, W)
            
        Processing steps:
            1. Extract latents and timesteps for evaluation
            2. Apply conditional guidance if enabled
            3. Generate noise predictions from both models
            4. Compute controlled and uncontrolled denoising steps
            5. Extract control signals from the difference
        """
        x_eval = all_x_t[:, indices_t, :, :, :, :]
        x_eval = rearrange(x_eval, 'b num_timesteps_to_load t c h w -> (b num_timesteps_to_load) t c h w')
        t_eval = all_t[indices_t] #(num_timesteps_to_load, )
        t_eval_repeat = t_eval.unsqueeze(0).repeat(batch_size, 1) #(b, num_timesteps_to_load)
        t_eval_repeat = rearrange(t_eval_repeat, 'b num_timesteps_to_load -> (b num_timesteps_to_load)') #(b*num_timesteps_to_load, )
        # expand the latents if we are doing classifier free guidance
        x_eval_model_input = torch.cat([x_eval] * 2) if self.config.cfg_control else x_eval

        # Make batch-level decision for prompt dropout
        use_negative_prompts = False
        if not self.config.cfg_control and hasattr(self.config, 'prompt_dropout') and self.config.prompt_dropout > 0:
            # Single random value for the entire batch
            use_negative_prompts = torch.rand(1, device=image_embeddings.device).item() < self.config.prompt_dropout
        
        if use_negative_prompts:
            # Use negative prompts for the entire batch
            image_latents_expanded = image_latents.chunk(2)[0].repeat_interleave(num_timesteps_to_load, dim=0)
            image_embeddings_expanded = image_embeddings.chunk(2)[0].repeat_interleave(num_timesteps_to_load, dim=0)
            added_time_ids_expanded = added_time_ids.chunk(2)[0].repeat_interleave(num_timesteps_to_load, dim=0)
            # Force recomputation of noise_pred_init since we're using different prompts
            noise_pred_init = None
        else:
            # Original behavior for positive prompts
            image_latents_expanded = image_latents.chunk(2)[1].repeat_interleave(num_timesteps_to_load, dim=0)
            image_embeddings_expanded = image_embeddings.chunk(2)[1].repeat_interleave(num_timesteps_to_load, dim=0)
            added_time_ids_expanded = added_time_ids.chunk(2)[1].repeat_interleave(num_timesteps_to_load, dim=0)
            if self.config.cfg_control:
                negative_image_latents = image_latents.chunk(2)[0].repeat_interleave(num_timesteps_to_load, dim=0)
                negative_image_embeddings = image_embeddings.chunk(2)[0].repeat_interleave(num_timesteps_to_load, dim=0)
                negative_added_time_ids = added_time_ids.chunk(2)[0].repeat_interleave(num_timesteps_to_load, dim=0)
                image_latents_expanded = torch.cat([negative_image_latents, image_latents_expanded], dim=0)
                image_embeddings_expanded = torch.cat([negative_image_embeddings, image_embeddings_expanded], dim=0)
                added_time_ids_expanded = torch.cat([negative_added_time_ids, added_time_ids_expanded], dim=0)

        # Concatenate image_latents over channels dimension
        x_eval_model_input = torch.cat([x_eval_model_input, image_latents_expanded], dim=2)
        x_eval_model_input = self.soc_pipeline.scheduler.batch_scale_model_input(x_eval_model_input, t_eval_repeat, self.time_steps)

        if self.config.learn_offset:
            raise NotImplementedError("Offset learning is not implemented yet")

        else:
            noise_pred_eval = self.unet(
                x_eval_model_input.to(self.unet.dtype),
                t_eval_repeat,
                encoder_hidden_states=image_embeddings_expanded,
                added_time_ids=added_time_ids_expanded,
                return_dict=False,
            )[0]
            
            if noise_pred_init is not None:
                noise_pred_init_eval = noise_pred_init
                noise_pred_init_eval = rearrange(noise_pred_init_eval, 'b num_timesteps_to_load t c h w -> (b num_timesteps_to_load) t c h w')
            else:
                with torch.no_grad():
                    self.unet.set_adapter([]) #deactivate the lora
                    noise_pred_init_eval = self.unet(
                        x_eval_model_input.to(self.unet.dtype),
                        t_eval_repeat,
                        encoder_hidden_states=image_embeddings_expanded.to(self.unet.dtype),
                        added_time_ids=added_time_ids_expanded.to(self.unet.dtype),
                        return_dict=False,
                    )[0].detach()
                    self.unet.set_adapter([self.lora_name]) #activate the lora
            # perform guidance
            if self.config.cfg_control:
                noise_pred_eval_uncond, noise_pred_eval_cond = noise_pred_eval.chunk(2)
                noise_pred_eval = noise_pred_eval_uncond + self.soc_pipeline.guidance_scale * (
                    noise_pred_eval_cond - noise_pred_eval_uncond
                )

                if noise_pred_init is None:
                    noise_pred_init_eval_uncond, noise_pred_init_eval_cond = noise_pred_init_eval.chunk(2)
                    noise_pred_init_eval = noise_pred_init_eval_uncond + self.soc_pipeline.guidance_scale * (
                        noise_pred_init_eval_cond - noise_pred_init_eval_uncond
                    )
        
        prev_sample, prev_sample_init, prev_sample_diff, std_dev_t, _ = self.soc_pipeline.scheduler.batch_step(
            noise_pred_eval, 
            noise_pred_init_eval, 
            t_eval_repeat, 
            x_eval,
            generator=generator,
            noise=torch.zeros_like(noise_pred_eval, device=noise_pred_eval.device),
            return_dict=False,
            scheduler_timesteps=self.time_steps,
        )
        ## Controlled update: 
        ##      prev_sample = x_eval + \delta * (b(x_eval,t_eval) + \sigma(t_eval) u(x_eval,t_eval)) + \sqrt{\delta} \sigma(t_eval) \epsilon
        ## Un-controlled update: 
        ##      prev_sample_init = x_eval + \delta * b(x_eval,t_eval) + \sqrt{\delta} \sigma(t_eval) \epsilon
        ## ==> prev_sample_diff = prev_sample - prev_sample_init = \delta \sigma(t_eval) u(x_eval,t_eval)
        prev_sample = rearrange(prev_sample, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=num_timesteps_to_load)
        prev_sample_init = rearrange(prev_sample_init, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=num_timesteps_to_load)
        prev_sample_diff = rearrange(prev_sample_diff, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=num_timesteps_to_load)
        std_dev_t = rearrange(std_dev_t, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=num_timesteps_to_load) #(b nttl 1 1 1 1)
        noise_pred_eval = rearrange(noise_pred_eval, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=num_timesteps_to_load)
        noise_pred_init_eval = rearrange(noise_pred_init_eval, '(b nttl) t c h w -> b nttl t c h w', b=batch_size, nttl=num_timesteps_to_load)

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
            smooth_gradients: Whether to use gradient smoothing
            num_smooth_samples: Number of noise samples for smoothing
            noise_std: Standard deviation of added noise
            clip_quantile: Quantile threshold for gradient clipping
            
        Returns:
            Tuple of (gradients, reward_values)
        """
        # print("autocast-enabled :", torch.is_autocast_enabled("cuda"))
        # print("cache_enabled    :", torch.is_autocast_cache_enabled())

        # offload the unet and encoder to cpu
        unet_device = self.soc_pipeline.unet.device
        encoder_device = next(self.soc_pipeline.vae.encoder.parameters()).device
        self.soc_pipeline.unet.to('cpu')
        self.soc_pipeline.vae.encoder.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

        if not smooth_gradients:
            # Original implementation
            with torch.enable_grad():
                x = x.requires_grad_(True)
                vid = self.latent_to_decode_fn(x)
                # postprocess -> pixel range in [0, 1]
                vid = self.soc_pipeline.video_processor.postprocess_video(video=vid, output_type='pt')
                # output = torch.autograd.grad(
                #     vid, x, grad_outputs=torch.ones_like(vid))[0]
                reward_values = self.reward_model(vid)
                if self.global_rank == 0:
                    print('reward value', reward_values)
                
                output = torch.autograd.grad(
                    reward_values.sum(), x
                )[0]

                torch.cuda.empty_cache()
                gc.collect()
                self.soc_pipeline.unet.to(unet_device)
                self.soc_pipeline.vae.encoder.to(encoder_device)
                
                return output.detach(), reward_values.detach()
        else:
            if self.config.reward_func == "VideoReward":
                # for the random query sampling method of cotracker reward model
                assert self.config.reward_model_config['discriminator_config']['spatio_query_method'] == 'Random', "Gradient smoothing is only for the random query sampling method of cotracker reward model"
            # step 1, decoding video without gradient, with a larger decod chunk size
            
            vid = latent_to_decode(model=self.soc_pipeline, output_type='pt', latents=x, decode_chunk_size=x.shape[1])
            vid = self.soc_pipeline.video_processor.postprocess_video(video=vid, output_type='pt') #(B, T, C, H, W)
            # step 2, repeat each video num_smooth_samples times, and compute the average reward gradient w.r.t each vid
            batch_size = vid.shape[0]
            vid = repeat(vid, 'b t c h w -> (ns b) t c h w', ns=num_smooth_samples)
            with torch.enable_grad():
                vid = vid.requires_grad_(True)
                reward_values = self.reward_model(vid)
                vid_grads = torch.autograd.grad(reward_values.sum(), vid, retain_graph=False)[0]
            vid_grads = rearrange(vid_grads.detach(), '(ns b) t c h w -> ns b t c h w', ns=num_smooth_samples)
            torch.cuda.empty_cache()
            gc.collect()

            # clip the gradient before average
            # Compute gradient norms: [num_samples, batch_size]
            grad_norms = torch.norm(vid_grads.view(num_smooth_samples, batch_size, -1), dim=2)
            # For each video, compute the quantile of the gradient norms
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
            # Reshape scaling to broadcast: [num_samples, batch_size, 1, 1, 1, 1]
            scaling = scaling.view(num_smooth_samples, batch_size, 1, 1, 1, 1)
            vid_grads = vid_grads * scaling
            
            # compute the mean reward values and mean grads
            vid_grads = vid_grads.mean(dim=0) #(b t c h w)
            reward_values = rearrange(reward_values.detach(), '(ns b) -> ns b', ns=num_smooth_samples)
            print('reward_values', reward_values)
            avg_rewards = reward_values.mean(dim=0) #(b,)
            print('avg_rewards', avg_rewards)
            print('reward grad', vid_grads.min(), vid_grads.max(), vid_grads.mean())
            # step 3, compute the reward gradient w.r.t x
            with torch.enable_grad():
                # here we need to compute the vid again
                x = x.requires_grad_(True)
                vid = self.latent_to_decode_fn(x)
                # postprocess -> pixel range in [0, 1]
                vid = self.soc_pipeline.video_processor.postprocess_video(video=vid, output_type='pt')
                # compute the reward gradient w.r.t x
                grads = torch.autograd.grad(vid, x, grad_outputs=vid_grads)[0]
            
            torch.cuda.empty_cache()
            gc.collect()
            self.soc_pipeline.unet.to(unet_device)
            self.soc_pipeline.vae.encoder.to(encoder_device)

            return grads.detach(), avg_rewards.detach()

        # else:
        #     batch_size = x.shape[0]
        
        #     # Store all gradients and rewards for processing after collection
        #     all_grads = []
        #     all_rewards = []
            
        #     # Process each noise sample individually
        #     for i in range(num_smooth_samples):
        #         # Generate single noise sample
        #         noise = torch.randn_like(x) * noise_std
        #         noisy_x = x + noise
        #         noisy_x = noisy_x.requires_grad_(True)
                
        #         # Forward pass for this single noise sample
        #         with torch.enable_grad():
        #             vid = self.latent_to_decode_fn(noisy_x)
        #             rewards = self.reward_fn(vid)
        #             grads = torch.autograd.grad(rewards.sum(), noisy_x, retain_graph=False)[0]
                
        #         # Store gradients and rewards
        #         all_grads.append(grads.detach())
        #         all_rewards.append(rewards.detach())
                
        #         # Optional progress reporting
        #         if i % 5 == 0 and self.global_rank == 0 and self.config.verbose:
        #             print(f"Smoothing progress: {i+1}/{num_smooth_samples}")
            
        #     # Stack all gradients [num_samples, batch_size, T, C, H, W]
        #     all_grads = torch.stack(all_grads)
        #     all_rewards = torch.stack(all_rewards)
            
        #     # Compute gradient norms: [num_samples, batch_size]
        #     grad_norms = torch.norm(all_grads.view(num_smooth_samples, batch_size, -1), dim=2)
            
        #     # For each prompt, compute the quantile of the gradient norms
        #     # Shape: [batch_size]
        #     quantile_vals = torch.quantile(grad_norms, clip_quantile, dim=0)
            
        #     # Clip the gradients - create a scaling factor
        #     # First, compute the max of quantile and actual norm to avoid division by zero
        #     # Shape: [num_samples, batch_size]
        #     scaling = torch.minimum(
        #         quantile_vals.unsqueeze(0) / torch.clamp(grad_norms, min=1e-8),
        #         torch.ones_like(grad_norms)
        #     )
            
        #     # Apply scaling to each gradient
        #     # Reshape scaling to broadcast: [num_samples, batch_size, 1, 1, 1, 1]
        #     scaling = scaling.view(num_smooth_samples, batch_size, 1, 1, 1, 1)
        #     clipped_grads = all_grads * scaling
            
        #     # Average over all samples
        #     avg_grads = clipped_grads.mean(dim=0)
        #     avg_rewards = all_rewards.mean(dim=0)
            
        #     if self.global_rank == 0 and self.config.verbose:
        #         print(f"Raw grad norm - min: {grad_norms.min().item():.4f}, max: {grad_norms.max().item():.4f}, mean: {grad_norms.mean().item():.4f}")
        #         print(f"Quantile values: {quantile_vals.mean().item():.4f}")
        #         print(f"Final grad norm: {torch.norm(avg_grads.view(batch_size, -1), dim=1).mean().item():.4f}")
            
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     return avg_grads, avg_rewards
    
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
        # get lora state
        lora_sd = get_peft_model_state_dict(
            self.unet, adapter_name=self.lora_name
        )

        # Lightning expects keys to be prefixed with the attribute name (`unet.`)
        checkpoint["state_dict"] = {f"unet.{k}": v.cpu() for k, v in lora_sd.items()}


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
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        """
        Custom load_from_checkpoint method to handle LoRA weights loading.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            config: Configuration object (if None, will try to load from checkpoint)
            **kwargs: Additional arguments to pass to __init__
        """
        # Load the checkpoint
        if not isinstance(checkpoint_path, (str, Path)):
            raise ValueError(f"checkpoint_path must be a string or Path, got {type(checkpoint_path)}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        
        # If config is not provided, try to load it from the checkpoint
        if config is None:
            if 'config_dict' in checkpoint:
                # Reconstruct config from saved dict
                from config_utils import Config
                config_dict = checkpoint['config_dict']
                config = Config(**config_dict)
                print(f"Loaded config from checkpoint: {checkpoint_path}")
            else:
                raise ValueError("Config not provided and not found in checkpoint")
        
        # Create the model instance
        try:
            model = cls(config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create model instance: {e}")
        
        # Load LoRA weights from the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Extract LoRA weights (remove 'unet.' prefix)
            lora_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('unet.'):
                    # Remove the 'unet.' prefix to get the LoRA parameter name
                    lora_key = key[5:]  # Remove 'unet.' prefix
                    lora_state_dict[lora_key] = value
            
            # Load the LoRA weights into the model
            if lora_state_dict:
                try:
                    set_peft_model_state_dict(model.unet, lora_state_dict, adapter_name=model.lora_name)
                    print(f"Successfully loaded {len(lora_state_dict)} LoRA parameters from checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Failed to load LoRA weights: {e}")
                    print("Continuing with randomly initialized LoRA weights...")
            else:
                print(f"Warning: No LoRA weights found in checkpoint: {checkpoint_path}")
                print("Continuing with randomly initialized LoRA weights...")
        else:
            print(f"Warning: No state_dict found in checkpoint: {checkpoint_path}")
            print("Continuing with randomly initialized LoRA weights...")
        
        return model


    # Abstract methods to be implemented by subclasses
    def training_step(self, batch, batch_id):
        """Main training loop, must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def validation_step(self, batch, batch_idx, stage):
        """Validation loop, must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")