import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from diffusers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Tuple, Union
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class SOCEDMAncestralSchedulerOutput(BaseOutput):
    """
    Custom output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
        prev_sample_diff (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Difference between `prev_sample` according to the fine-tuned model, and `prev_sample` according to the 
            pre-trained (original) model.
        std_dev_t (`torch.FloatTensor` of shape `(batch_size)` for images):
            diffusion standard deviation
    """

    # The first attributes are the attributes of DDIMSchedulerOutput
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    # The next two attributes are not in DDIMSchedulerOutput
    prev_sample_diff: Optional[torch.Tensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    variance_noise: Optional[torch.FloatTensor] = None

class SOCEDMAncestralScheduler(EulerDiscreteScheduler):
    """
    `SOCEDMAncestralScheduler` is a modification of `EulerDiscreteScheduler` where the `step` function takes in `model_output` as well
    as `model_output_init` (the output of the original model), as well as other changes tailored for stochastic optimal
    control fine-tuning. Besides, the step function follows EulerAncestralDiscreteScheduler.

    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """
    def batch_scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        Batched scale model input
        Args:
            sample (`torch.Tensor`): (B, T, C, H, W)
                The input sample.
            timestep (`int`, *optional*): (B,)
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        step_indices = [self.index_for_timestep(t) for t in timestep]
        sigma = self.sigmas[step_indices].flatten().to(sample.device)
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = self.precondition_inputs(sample, sigma)

        return sample

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        alpha_prod_t_prev = (self.alphas_cumprod[prev_timestep] * (prev_timestep >= 0).to(torch.int) 
                             + self.final_alpha_cumprod * (prev_timestep < 0).to(torch.int))
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    
    def step(
        self,
        model_output: torch.Tensor,
        model_output_init: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SOCEDMAncestralSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        init = model_output_init is not None
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        # gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0 #default gamma should be 0
        gamma = 0.0
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
            eps = noise * s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
            if init:
                pred_original_sample_init = model_output_init
            else:
                pred_original_sample_init = None
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
            if init:
                pred_original_sample_init = sample - sigma_hat * model_output_init
            else:
                pred_original_sample_init = None
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            if init:
                pred_original_sample_init = model_output_init * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            else:
                pred_original_sample_init = None
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        """THE ORIGINAL EULER DISCRETE SCHEDULER STEP FUNCTION
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat
        if init:
            derivative_init = (sample - pred_original_sample_init) / sigma_hat
        
        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        if init:
            prev_sample_init = sample + derivative_init * dt
        """
        # <-------Euler ancestral update------->
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        if init:
            derivative_init = (sample - pred_original_sample_init) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt
        if init:
            prev_sample_init = sample + derivative_init * dt

        prev_sample = prev_sample + noise * sigma_up
        if init:
            prev_sample_init = prev_sample_init + noise * sigma_up

        # <-------Euler ancestral update------->

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
        if init:
            prev_sample_init = prev_sample_init.to(model_output.dtype)
        
        prev_sample_diff = prev_sample - prev_sample_init if init else None

        # upon completion increase step index by one
        self._step_index += 1

        variance_noise = noise
        std_dev_t = sigma_up
        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
                prev_sample_diff,
                std_dev_t,
                variance_noise
            )

        return SOCEDMAncestralSchedulerOutput(prev_sample=prev_sample, 
                                     pred_original_sample=pred_original_sample, 
                                     prev_sample_diff=prev_sample_diff,
                                     std_dev_t=std_dev_t,
                                     variance_noise=variance_noise)
    
    def batch_step(
        self,
        model_output: torch.Tensor,
        model_output_init: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SOCEDMAncestralSchedulerOutput, Tuple]:
        """
        Used for the controal evaluation.
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`): (B, T, C, H, W)
                The direct output from learned diffusion model.
            timestep (`float`): (B, 1)
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`): (B, T, C, H, W)
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        init = model_output_init is not None
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        step_indices = [self.index_for_timestep(t) for t in timestep]
        sigma = self.sigmas[step_indices].flatten().to(sample.device)
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        # gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0 #default gamma should be 0
        gamma = 0.0
        sigma_hat = sigma * (gamma + 1)

        # if gamma > 0:
        #     if noise is None:
        #         noise = randn_tensor(
        #             model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        #         )
        #     eps = noise * s_noise
        #     sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
            if init:
                pred_original_sample_init = model_output_init
            else:
                pred_original_sample_init = None
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
            if init:
                pred_original_sample_init = sample - sigma_hat * model_output_init
            else:
                pred_original_sample_init = None
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            if init:
                pred_original_sample_init = model_output_init * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            else:
                pred_original_sample_init = None
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        """THE ORIGINAL EULER DISCRETE SCHEDULER STEP FUNCTION
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat
        if init:
            derivative_init = (sample - pred_original_sample_init) / sigma_hat
        
        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        if init:
            prev_sample_init = sample + derivative_init * dt
        """
        # <-------Euler ancestral update------->


        sigma_from = self.sigmas[step_indices].flatten().to(sample.device)
        while len(sigma_from.shape) < len(sample.shape):
            sigma_from = sigma_from.unsqueeze(-1)

        sigma_to = self.sigmas[[s + 1 for s in step_indices]].flatten().to(sample.device)
        while len(sigma_to.shape) < len(sample.shape):
            sigma_to = sigma_to.unsqueeze(-1)

        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        if init:
            derivative_init = (sample - pred_original_sample_init) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt
        if init:
            prev_sample_init = sample + derivative_init * dt
        if noise is None:
            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)
        prev_sample = prev_sample + noise * sigma_up
        if init:
            prev_sample_init = prev_sample_init + noise * sigma_up

        # <-------Euler ancestral update------->

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
        if init:
            prev_sample_init = prev_sample_init.to(model_output.dtype)
        
        prev_sample_diff = prev_sample - prev_sample_init if init else None

        # upon completion increase step index by one
        self._step_index += 1

        variance_noise = noise
        std_dev_t = sigma_up
        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
                prev_sample_diff,
                std_dev_t,
                variance_noise
            )

        return SOCEDMAncestralSchedulerOutput(prev_sample=prev_sample, 
                                     pred_original_sample=pred_original_sample, 
                                     prev_sample_diff=prev_sample_diff,
                                     std_dev_t=std_dev_t,
                                     variance_noise=variance_noise)