import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from diffusers import DDIMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Tuple, Union

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class SOCDDIMSchedulerOutput(BaseOutput):
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

class SOCDDIMScheduler(DDIMScheduler):
    """
    `SOCDDIMScheduler` is a modification of `DDIMScheduler` where the `step` function takes in `model_output` as well
    as `model_output_init` (the output of the original model), as well as other changes tailored for stochastic optimal
    control fine-tuning. 

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
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SOCDDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            model_output_init (`torch.Tensor`):
                The direct output from pre-trained (original) diffusion model. If None, the behavior of this function
                defaults to the behavior of the `step` function of the `DDIMScheduler` class.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        init = model_output_init is not None

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (self.alphas_cumprod[prev_timestep] * (prev_timestep >= 0).to(torch.int) 
                             + self.final_alpha_cumprod * (prev_timestep < 0).to(torch.int))
        beta_prod_t = 1 - alpha_prod_t

        alpha_prod_t = alpha_prod_t[...,None,None,None]
        alpha_prod_t_prev = alpha_prod_t_prev[...,None,None,None]
        beta_prod_t = beta_prod_t[...,None,None,None]
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
            if init:
                pred_original_sample_init = (sample - beta_prod_t ** (0.5) * model_output_init) / alpha_prod_t ** (0.5)
                pred_epsilon_init = model_output_init
            else:
                pred_original_sample_init = None
                pred_epsilon_init = None
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            if init:
                pred_original_sample_init = model_output_init
                pred_epsilon_init = (sample - alpha_prod_t ** (0.5) * pred_original_sample_init) / beta_prod_t ** (0.5)
            else:
                pred_original_sample_init = None
                pred_epsilon_init = None
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
            if init:
                pred_original_sample_init = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output_init
                pred_epsilon_init = (alpha_prod_t**0.5) * model_output_init + (beta_prod_t**0.5) * sample
            else:
                pred_original_sample_init = None
                pred_epsilon_init = None
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
            if init:
                pred_original_sample_init = self._threshold_sample(pred_original_sample_init)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
            if init:
                pred_original_sample_init = pred_original_sample_init.clamp(
                    -self.config.clip_sample_range, self.config.clip_sample_range
                )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        # print(f'variance: {variance}')
        std_dev_t = eta * variance ** (0.5)
        std_dev_t = std_dev_t[...,None,None,None]

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            if init:
                pred_epsilon_init = (sample - alpha_prod_t ** (0.5) * pred_original_sample_init) / beta_prod_t ** (0.5)

        sqrt_arg_sign = torch.sign(1 - alpha_prod_t_prev - std_dev_t**2)
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.
        pred_sample_direction = sqrt_arg_sign * (sqrt_arg_sign * (1 - alpha_prod_t_prev - std_dev_t**2)) ** (0.5) * pred_epsilon
        if init:
            pred_sample_direction_init = sqrt_arg_sign * (sqrt_arg_sign * (1 - alpha_prod_t_prev - std_dev_t**2)) ** (0.5) * pred_epsilon_init

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if init:
            prev_sample_init = alpha_prod_t_prev ** (0.5) * pred_original_sample_init + pred_sample_direction_init

        prev_sample_diff = prev_sample - prev_sample_init if init else None

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance
            if init:
                prev_sample_init = prev_sample_init + variance

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
                prev_sample_diff,
                std_dev_t,
                variance_noise,
            )

        return SOCDDIMSchedulerOutput(
            prev_sample=prev_sample, 
            pred_original_sample=pred_original_sample,
            prev_sample_diff=prev_sample_diff,
            std_dev_t=std_dev_t,
            variance_noise=variance_noise,
        )